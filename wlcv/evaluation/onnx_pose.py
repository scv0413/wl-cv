"""같은 원본 픽셀에서 추론한 자세를 수행자·관절·동작 시점별로 대응한다."""

from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torchvision.ops import box_iou
from tqdm.auto import tqdm
import ultralytics
from ultralytics import YOLO

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.annotation.trunk_repeat import FIRST_PAIRS, repeat_frames
from wlcv.config import ANNOTATIONS, ROOT, SPLIT, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.pose import JOINTS, joint_table
from wlcv.evaluation.trunk import vertical_angle

WEIGHTS = ROOT / "models/yolo26n-pose.pt"
TRUNK = ANNOTATIONS / "trunk_inclination"
INPUT = (640, 384)
PERSON_CONF, JOINT_CONF = 0.25, 0.50
RUNTIMES = ("pt", "ort")


def export_pose(folder):
    """학습 가중치 옆에 파일을 쓰지 않도록 실행 폴더에 복사해서 변환한다."""
    with TemporaryDirectory(dir=folder) as temp:
        copy = Path(temp) / "pose.pt"
        copyfile(WEIGHTS, copy)
        # 이 버전에서 nms=None은 23번 추론과 같은 원시 자세 헤드를 내보낸다.
        file = YOLO(str(copy)).export(format="onnx", device="cpu", imgsz=INPUT, batch=1,
                                      opset=17, half=False, dynamic=False, simplify=False, nms=None)
        path = Path(file).replace(folder / "pose.onnx")
    onnx.checker.check_model(str(path))
    return path


def match_performer(pt, ort_result, video, frame):
    """23번에서 확인한 사람 박스와 IoU 0.50 이상인 ONNX 후보가 하나일 때만 대응한다."""
    multiple = (video, frame) == ("b02_sn_h_02", 30)
    expected_count = 2 if multiple else 1
    if len(pt.boxes) != expected_count:
        return None, None, np.nan
    index = expected_count - 1
    if not len(ort_result.boxes):
        return index, None, np.nan
    overlaps = box_iou(pt.boxes.xyxy[index:index+1], ort_result.boxes.xyxy)[0]
    matches = torch.where(overlaps >= 0.50)[0]
    if len(matches) != 1:
        return index, None, float(overlaps.max())
    return index, int(matches[0]), float(overlaps[matches[0]].item())


def near_points(result, index, side):
    if index is None:
        return {}
    points = joint_table(result.numpy(), index + 1, JOINT_CONF)
    return points.loc[points["방향"] == side].set_index("관절").to_dict("index")


def record_frame(info, video, phase, frame, results):
    left, right, overlap = match_performer(*results, video, frame)
    key = dict(person=info.person_id, video=video, phase=phase, frame=frame, side=info.near_side)
    row = dict(**key, pt_people=len(results[0].boxes), ort_people=len(results[1].boxes),
               pt_id=None if left is None else left + 1, ort_id=None if right is None else right + 1,
               performer_iou=overlap)
    # 21번 1차 체간 판독을 우선하며, 같은 프레임의 기존 무릎·발목 판독도 불러온다.
    manual = load_points(pose_point_path(ANNOTATIONS, video, frame))
    manual.update(load_points(pose_point_path(TRUNK, video, frame)))
    joints = {name: dict(**key, joint=name) for name in JOINTS}
    manual_trunk = []
    for name, item in joints.items():
        label = manual.get(name, {})
        if label and (label["side"] != info.near_side or label["split"] != "validation"
                      or label["person_id"] != info.person_id or label["video_id"] != video
                      or int(label["frame_index"]) != frame):
            raise ValueError(f"수동점의 대상 또는 프레임을 확인하세요: {video} {frame} {name}")
        item["manual_status"] = label.get("status", "unlabeled")
        item["manual_reason"] = label.get("reason", "")
        item["manual_x"] = float(label.get("x_px") or np.nan)
        item["manual_y"] = float(label.get("y_px") or np.nan)
        if name in ("어깨", "골반") and item["manual_status"] == "located":
            manual_trunk.append((item["manual_x"], item["manual_y"]))
    row["manual_angle_deg"] = vertical_angle(*manual_trunk) if len(manual_trunk) == 2 else np.nan

    for runtime, result, index in zip(RUNTIMES, results, (left, right)):
        points = near_points(result, index, info.near_side)
        trunk = []
        for name, item in joints.items():
            point = points.get(name, {})
            x, y = point.get("x (px)", np.nan), point.get("y (px)", np.nan)
            status = point.get("표시 상태", "수행자 미확인")
            item.update({runtime+"_x": x, runtime+"_y": y, runtime+"_status": status,
                         runtime+"_score": point.get("관절 신뢰도", np.nan)})
            item[runtime+"_manual_px"] = np.nan
            if status == "표시":
                if item["manual_status"] == "located":
                    item[runtime+"_manual_px"] = np.hypot(x-item["manual_x"], y-item["manual_y"])
                if name in ("어깨", "골반"):
                    trunk.append((x, y))
        row[runtime+"_angle_deg"] = vertical_angle(*trunk) if len(trunk) == 2 else np.nan
    for item in joints.values():
        item["distance_px"] = np.hypot(item["pt_x"]-item["ort_x"], item["pt_y"]-item["ort_y"])
        item["score_diff"] = abs(item["pt_score"] - item["ort_score"])
    return row, list(joints.values())


@torch.inference_mode()
def compare(folder):
    split = pd.read_csv(SPLIT)
    validation = split.loc[split.role.eq("validation"), ["video_id", "person_id"]]
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv")
    videos = validation.merge(meta, on=["video_id", "person_id"], validate="one_to_one").set_index("video_id")
    cases = sorted(repeat_frames(videos, TRUNK))
    people = videos.loc[list(FIRST_PAIRS), "person_id"]
    if len(cases) != 12 or people.nunique() != 6 or set(people) & set(split.loc[split.role.eq("train"), "person_id"]):
        raise ValueError("기존 검증 6명의 확정 시점 12장을 확인하세요.")
    path = export_pose(folder)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    entry, = session.get_inputs()
    output, = session.get_outputs()
    if entry.shape != [1, 3, *INPUT] or entry.type != "tensor(float)" or output.shape != [1, 56, 5040]:
        raise ValueError(f"자세 모델의 입출력을 확인하세요. {entry.shape} → {output.shape}")
    model = YOLO(str(WEIGHTS))
    frame_rows, joint_rows = [], []
    for video, frame in tqdm(cases, desc="두 런타임의 자세 비교", unit="장"):
        info = videos.loc[video]
        bgr = read_rgb(VIDEOS / info.filename, frame)[:, :, ::-1].copy()
        if model.predictor is None:
            model.predict(bgr, imgsz=640, conf=PERSON_CONF, iou=0.7, device="cpu", verbose=False)
            torch.set_num_threads(4)
        predictor = model.predictor
        tensor = predictor.preprocess([bgr])
        if list(tensor.shape) != entry.shape or tensor.dtype != torch.float32:
            raise ValueError(f"{video} {frame}의 640 전처리 크기를 확인하세요.")
        raw_pt = predictor.model(tensor)
        if isinstance(raw_pt, (list, tuple)):
            raw_pt = raw_pt[0]
        array, = session.run([output.name], {entry.name: tensor.numpy()})
        raw_ort = torch.from_numpy(array)
        results = []
        for raw in (raw_pt, raw_ort):
            if list(raw.shape) != output.shape or not torch.isfinite(raw).all():
                raise ValueError(f"{video} {frame}의 원시 출력에 문제가 있습니다.")
            # 동일한 신뢰도 필터와 NMS, 원본 해상도로의 박스·관절 복원을 적용한다.
            result, = predictor.postprocess(raw, tensor, [bgr])
            results.append(result)
        phase = "lift_off" if frame == FIRST_PAIRS[video][0] else "knee_pass"
        row, points = record_frame(info, video, phase, frame, results)
        frame_rows.append(row)
        joint_rows.extend(points)
    frames, joints = pd.DataFrame(frame_rows), pd.DataFrame(joint_rows)
    frames["angle_diff_deg"] = (frames.pt_angle_deg - frames.ort_angle_deg).abs()
    changes = pd.DataFrame(index=people.to_list())
    changes.index.name = "person"
    for runtime in RUNTIMES:
        angles = frames.pivot(index="person", columns="phase", values=runtime+"_angle_deg")
        changes[runtime+"_change_deg"] = angles.knee_pass - angles.lift_off
    changes["diff_deg"] = (changes.pt_change_deg - changes.ort_change_deg).abs()
    for name, table in (("frames", frames), ("joints", joints), ("changes", changes.reset_index())):
        table.to_csv(folder / f"{name}.csv", index=False)
    settings = dict(torch=torch.__version__, onnx=onnx.__version__, ort=ort.__version__,
                    ultralytics=ultralytics.__version__, provider=session.get_providers(),
                    input=entry.shape, output=output.shape, dtype="float32", opset=17,
                    weights_sha256=sha256(WEIGHTS.read_bytes()).hexdigest(),
                    onnx_sha256=sha256(path.read_bytes()).hexdigest(), person_conf=PERSON_CONF,
                    joint_conf=JOINT_CONF, match_iou=0.5, nms_iou=0.7, threads=torch.get_num_threads())
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    return frames, joints, changes


def overlay(person, frames, joints):
    """두 시점에서 같은 원본 위치에 두 런타임의 점과 직접·추정 수동점을 겹친다."""
    pair = frames[frames.person.eq(person)].sort_values("frame")
    if len(pair) != 2:
        raise ValueError("검증 선수 한 명의 두 시점을 선택하세요.")
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), layout="constrained")
    for ax, row in zip(axes, pair.itertuples()):
        rgb = read_rgb(VIDEOS / f"{row.video}.mp4", row.frame)
        points = joints[joints.video.eq(row.video) & joints.frame.eq(row.frame)]
        ax.imshow(rgb)
        for rt, marker, color in (("pt", "o", "cyan"), ("ort", "+", "magenta")):
            valid = points[rt+"_status"].eq("표시")
            ax.plot(points[rt+"_x"].where(valid), points[rt+"_y"].where(valid),
                    marker=marker, markerfacecolor="none", color=color, ls="-" if rt=="pt" else ":",
                    ms=9, label="PyTorch" if rt=="pt" else "ONNX Runtime")
        for status, marker, label in (("located", "x", "Manual"), ("estimated", "^", "Estimated")):
            marked = points[points.manual_status.eq(status)]
            ax.scatter(marked.manual_x, marked.manual_y, marker=marker, color="gold", label=label)
        ax.set_title(f"{person} / {row.phase} / frame {row.frame}")
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    return fig
