"""Mac에서 같은 검증 라벨로 모델별 결과를 채점한다. GPU에는 이 파일이 필요 없다."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywhn2xyxy

from wlcv.config import ANNOTATIONS, DATASET
from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation import onnx_plate, onnx_pose
from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.plate_track import BOX, KEY, select_centers
from wlcv.evaluation.precision_inputs import B02, POSE_REF, source_image
from wlcv.evaluation.schedule_review import review_inputs
from wlcv.annotation.plate_blur import REVIEW_FILE


def plate_metrics(manifest, outputs, folder):
    validator = DetectionValidator(save_dir=folder)
    metric = DetMetrics(names={0: "plate"})
    counts, boxes = [], []
    for row in manifest.itertuples():
        if row.key not in outputs:
            continue
        raw = outputs[row.key]
        pred = torch.from_numpy(
            onnx_plate.candidates(raw, (640, 384), (row.height, row.width, 3))
        )
        pred = pred[pred[:, 4].argsort(descending=True, stable=True)]
        if row.source == "video":
            for x1, y1, x2, y2, confidence, _ in pred.tolist():
                boxes.append([row.video, row.frame, confidence, x1, y1, x2, y2])
            continue
        stem = f"{row.video}_f{row.frame}"
        truth = torch.tensor(
            read_boxes(DATASET / "labels/val" / (stem + ".txt")), dtype=torch.float32
        ).reshape(-1, 4)
        truth = xywhn2xyxy(truth, w=row.width, h=row.height)
        labels = torch.zeros(len(truth))
        matched = validator.match_predictions(
            pred[:, 5], labels, box_iou(truth, pred[:, :4])
        )
        metric.update_stats(
            dict(
                tp=matched.numpy(),
                conf=pred[:, 4].numpy(),
                pred_cls=pred[:, 5].numpy(),
                target_cls=labels.numpy(),
                target_img=labels.unique().numpy(),
                im_name=stem,
            )
        )
        direct = pred[pred[:, 4] >= 0.10]
        found = validator.match_predictions(
            direct[:, 5], labels, box_iou(truth, direct[:, :4])
        )
        tp = int(found[:, 0].sum())
        counts.append(
            dict(
                image=stem,
                video=row.video,
                person=row.person,
                frame=row.frame,
                active=row.active,
                tp=tp,
                fp=len(direct) - tp,
                fn=len(truth) - tp,
            )
        )
    metric.process()
    return (
        dict(mAP50=metric.box.map50, mAP50_95=metric.box.map),
        pd.DataFrame(counts),
        pd.DataFrame(boxes, columns=KEY + ["confidence"] + BOX),
    )


def score_hubs(boxes):
    # 이 출발점에서 직접 검출이 성립하면 더 앞 프레임 없이도 기존 궤적과 같은 이력을 얻는다.
    segments = []
    for window in ((20,), tuple(range(30, 37))):
        frames = pd.DataFrame({"video_id": B02, "frame_index": window})
        selected = select_centers(
            frames, boxes, distance_ratio=0.5, show_progress=False
        )
        selected = selected[selected.method.eq("Past link")].copy()
        # 직전 문맥을 모르는 구간에서 두 번 연속 직접 검출하면 그 뒤 이력은 확정된다.
        high = selected.source.eq("high")
        known = high | (high & high.shift(fill_value=False)).cummax()
        selected.loc[~known, "source"] = "history_unknown"
        selected.loc[~known, ["confidence", "cx", "cy", *BOX]] = np.nan
        segments.append(selected)
    track = pd.concat(segments, ignore_index=True)
    _, truth, _ = review_inputs(DATASET / "data.yaml")
    truth = truth[truth.video_id.eq(B02)].set_index(
        "frame_index", verify_integrity=True
    )
    blind = pd.read_csv(REVIEW_FILE)
    manual = (
        blind[
            blind.video_id.eq(B02)
            & blind.hub_source.isin(["clicked", "previous_blind"])
        ]
        .set_index("frame_index")
        .manual_y.to_dict()
    )
    lift = pd.read_csv(ANNOTATIONS / "hub_lift_review" / f"{B02}.csv").query(
        "frame_index == 20 and visibility == 'visible'"
    )
    manual[20] = float(lift.manual_y.item())
    track["manual_y"] = [manual[i] for i in track.frame_index]
    track["iou"] = [
        iou(r[BOX].to_numpy(float), truth.loc[r.frame_index, BOX].to_numpy(float))
        for _, r in track.iterrows()
    ]
    track["match"] = track.iou.ge(0.5)
    track["dy_px"] = (track.cy - track.manual_y).where(track.match)
    track["abs_dy_px"] = track.dy_px.abs()
    endpoints = track.set_index("frame_index").loc[[20, 30]]
    measured = (
        endpoints.cy.iloc[0] - endpoints.cy.iloc[1] if endpoints.match.all() else np.nan
    )
    manual_rise = manual[20] - manual[30]
    rise = dict(
        manual_rise_px=manual_rise,
        model_rise_px=measured,
        rise_error_px=measured - manual_rise,
        matched_30_36=int(track[track.frame_index >= 30].match.sum()),
        mean_abs_30_36=track[track.frame_index >= 30].abs_dy_px.mean(),
    )
    return track, rise


@torch.inference_mode()
def pose_metrics(manifest, inputs, reference, outputs):
    original = pd.read_csv(POSE_REF / "frames.csv").set_index(
        ["video", "frame"], verify_integrity=True
    )
    model = YOLO(str(onnx_pose.WEIGHTS))
    model.predict(
        np.zeros((640, 384, 3), np.uint8),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device="cpu",
        verbose=False,
    )
    frames, joints = [], []
    for row in manifest.itertuples():
        if row.key not in outputs:
            continue
        case = original.loc[row.video, row.frame]
        bgr = source_image(row)
        tensor = torch.from_numpy(inputs[row.key].astype(np.float32) / 255)
        if not torch.equal(tensor, model.predictor.preprocess([bgr])):
            raise ValueError("참조 자세 전처리와 평가 입력이 다릅니다.")
        compared = []
        for archive in (reference, outputs):
            (result,) = model.predictor.postprocess(
                torch.from_numpy(archive[row.key].copy()), tensor, [bgr]
            )
            compared.append(result)
        info = pd.Series(dict(person_id=row.person, near_side=row.side))
        detail, points = onnx_pose.record_frame(
            info, row.video, case.phase, row.frame, compared
        )
        if detail["pt_id"] != case.pt_id:
            raise ValueError("기존 수행자 참고 박스를 확인하세요.")
        if row.video == B02 and row.frame == 30:
            hand = pd.read_csv(ANNOTATIONS / "performer_selection/b02.csv")
            target = hand.loc[hand.frame_index.eq(30), BOX].to_numpy(dtype=np.float32)
            overlap = box_iou(torch.from_numpy(target), compared[0].boxes.xyxy)[0]
            matches = torch.where(overlap >= 0.50)[0].tolist()
            if matches != [int(detail["pt_id"]) - 1]:
                raise ValueError("b02 30번 수동 수행자 박스가 기준 모델 후보와 다릅니다.")
        frames.append(
            dict(
                video=row.video,
                frame=row.frame,
                phase=case.phase,
                matched=detail["ort_id"] is not None,
                performer_iou=detail["performer_iou"],
                angle_deg=detail["ort_angle_deg"],
            )
        )
        for pt in points:
            joints.append(
                dict(
                    video=row.video,
                    frame=row.frame,
                    side=row.side,
                    joint=pt["joint"],
                    manual_status=pt["manual_status"],
                    status=pt["ort_status"],
                    score=pt["ort_score"],
                    x=pt["ort_x"],
                    y=pt["ort_y"],
                    manual_error_px=pt["ort_manual_px"],
                )
            )
    return pd.DataFrame(frames), pd.DataFrame(joints)


def score(folder, runtimes):
    folder = Path(folder)
    payload = folder / "gpu_input"
    results = folder / "predictions"
    spec = json.loads((payload / "jobs.json").read_text())
    if (
        sha256((payload / "inputs.npz").read_bytes()).hexdigest()
        != spec["input_sha256"]
    ):
        raise ValueError("평가 입력 체크섬이 다릅니다.")
    if any(r.startswith("trt-") for r in runtimes):
        receipt = json.loads((results / "real_run.json").read_text())
        if (
            receipt["input_sha256"] != spec["input_sha256"]
            or receipt["graph_sha256"] != spec["graph_sha256"]
            or receipt["device"] != "NVIDIA GeForce RTX 3090"
            or receipt["tf32"] is not False
        ):
            raise ValueError("회수한 GPU 결과의 입력·모델·장치가 다릅니다.")
    manifest = pd.read_csv(folder / "local_frames.csv")
    tables = {
        name: []
        for name in ("map", "photo_counts", "hubs", "rise", "performers", "joints")
    }
    for runtime in runtimes:
        archives = {}
        for model, channels in (("plate", 5), ("pose", 56)):
            with np.load(
                results / f"{runtime}-{model}.npz", allow_pickle=False
            ) as data:
                if set(data.files) != set(spec["jobs"][model]):
                    raise ValueError(f"{runtime} {model} 입력 목록이 다릅니다.")
                archives[model] = dict(data)
            for raw in archives[model].values():
                if (
                    raw.shape != (1, channels, 5040)
                    or raw.dtype != np.float32
                    or not np.isfinite(raw).all()
                ):
                    raise ValueError(f"{runtime} {model} 출력 형식이 다릅니다.")
        metrics, photos, boxes = plate_metrics(manifest, archives["plate"], folder)
        hubs, rise = score_hubs(boxes)
        with (
            np.load(payload / "inputs.npz") as inputs,
            np.load(results / "pytorch-fp32-pose.npz") as reference,
        ):
            people, joints = pose_metrics(manifest, inputs, reference, archives["pose"])
        for name, table in zip(
            tables,
            (
                pd.DataFrame([metrics]),
                photos,
                hubs,
                pd.DataFrame([rise]),
                people,
                joints,
            ),
            strict=True,
        ):
            tables[name].append(table.assign(runtime=runtime))
    output = folder / ("trt_quality" if "trt-fp16" in runtimes else "local_controls")
    output.mkdir(exist_ok=True)
    tables = {key: pd.concat(parts, ignore_index=True) for key, parts in tables.items()}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    left, right = runtimes[-2:]
    photos = tables["photo_counts"].pivot(
        index=["person", "video", "frame", "active"], columns="runtime", values="fn"
    )
    photos["new_fn"] = (photos[right] - photos[left]).clip(lower=0)
    photos["recovered"] = (photos[left] - photos[right]).clip(lower=0)
    photos.to_csv(output / "paired_misses.csv")
    joints = tables["joints"].pivot(
        index=["video", "frame", "side", "joint"], columns="runtime", values="status"
    )
    joints["changed"] = joints[left] != joints[right]
    joints.to_csv(output / "paired_joints.csv")
    return tables
