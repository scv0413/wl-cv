"""세 원판 런타임에 같은 원본 입력과 배포 추적 규칙을 적용한다."""

import argparse
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, COSINE_WEIGHTS, CUTMIX_WEIGHTS, DATASET, VIDEOS
from wlcv.data.frames import frames_rgb
from wlcv.evaluation import onnx_plate
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.evaluation.plate_endpoint_rise import check_endpoint, select_rise
from wlcv.evaluation.plate_patience_review import LABELS
from wlcv.evaluation.plate_track import BOX, KEY, PlateHistory, choose_plate
from wlcv.evaluation.schedule_review import review_inputs

WEIGHTS = {"baseline": onnx_plate.WEIGHTS, "cutmix": CUTMIX_WEIGHTS, "cosine": COSINE_WEIGHTS}
SHAPE = (1, 3, 640, 384)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def export_graph(weights, target):
    with TemporaryDirectory(dir=target.parent) as temp:
        copy = Path(temp) / (target.stem + ".pt")
        shutil.copyfile(weights, copy)
        exported = YOLO(str(copy)).export(
            format="onnx",
            imgsz=SHAPE[2:],
            batch=1,
            device="cpu",
            dynamic=False,
            opset=17,
            simplify=False,
            nms=None,
        )
        graph = onnx.load(exported)
        del graph.metadata_props[:]  # 원격 실행에는 학습 경로나 부가 설명이 필요 없다.
        onnx.checker.check_model(graph)
        onnx.save(graph, target)


def prepare(folder):
    payload = folder / "gpu_input"
    payload.mkdir(parents=True, exist_ok=False)
    graphs = {}
    videos, _, _ = review_inputs(DATASET / "data.yaml")
    for name, weights in WEIGHTS.items():
        detector = validation_plate_model(weights, videos)
        target = payload / (name + ".onnx")
        export_graph(weights, target)
        graphs[name] = digest(target)
    first = np.zeros((640, 384, 3), dtype=np.uint8)
    detector.predict(first, imgsz=640, device="cpu", verbose=False)
    rows, tensors, events = [], {}, []
    for video in videos:
        phases = load_events(ANNOTATIONS, video, 1)
        last = max(phases.values())
        for index, rgb in enumerate(
            tqdm(islice(frames_rgb(VIDEOS / f"{video}.mp4"), last + 1), total=last + 1, desc=video)
        ):
            image = detector.predictor.preprocess([rgb[:, :, ::-1].copy()]).cpu().numpy()
            if image.shape != SHAPE:
                raise ValueError(f"{video} {index}: 입력 크기 {image.shape}")
            key = f"f{len(rows):04d}"
            packed = np.rint(image * 255).astype(np.uint8)
            np.testing.assert_array_equal(packed.astype(np.float32) / 255, image)
            tensors[key] = packed
            is_endpoint = index in (phases["transition"], phases["drive"])
            rows.append(
                dict(
                    key=key,
                    video_id=video,
                    frame_index=index,
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    in_motion=index >= phases["ready"],
                    endpoint=is_endpoint,
                )
            )
            if is_endpoint:
                events.append(key)
    manifest = pd.DataFrame(rows)
    if (len(manifest), manifest.in_motion.sum(), len(events)) != (533, 478, 12):
        raise ValueError("검증 6명의 프레임 범위를 확인하세요.")
    manifest.to_csv(folder / "local_frames.csv", index=False)
    np.savez_compressed(payload / "inputs.npz", **tensors)
    spec = dict(
        graph_sha256=graphs,
        input_sha256=digest(payload / "inputs.npz"),
        jobs={"baseline": list(tensors), "cutmix": list(tensors), "cosine": events},
    )
    (payload / "jobs.json").write_text(json.dumps(spec))
    (folder / "weights_sha256.json").write_text(
        json.dumps({n: digest(p) for n, p in WEIGHTS.items()}, indent=2)
    )


@torch.inference_mode()
def controls(folder):
    payload = folder / "gpu_input"
    jobs = json.loads((payload / "jobs.json").read_text())["jobs"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    torch.set_num_threads(2)
    with np.load(payload / "inputs.npz", allow_pickle=False) as inputs:
        for name, keys in jobs.items():
            model = YOLO(str(WEIGHTS[name]))
            model.predict(np.zeros((640, 384, 3), np.uint8), imgsz=640, device="cpu", verbose=False)
            session = ort.InferenceSession(
                str(payload / f"{name}.onnx"), options, providers=["CPUExecutionProvider"]
            )
            (entry,) = session.get_inputs()
            if entry.shape != list(SHAPE) or session.get_outputs()[0].shape != [1, 5, 5040]:
                raise ValueError(f"{name} 그래프 입출력을 확인하세요.")
            pt, ort_result = {}, {}
            for key in tqdm(keys, desc=name):
                x = inputs[key].astype(np.float32) / 255
                pt[key] = onnx_plate.raw_pytorch(model.predictor, torch.from_numpy(x)).copy()
                (ort_result[key],) = session.run(None, {entry.name: x})
            for engine, pred in [("pytorch", pt), ("onnx", ort_result)]:
                np.savez_compressed(folder / f"{engine}-{name}.npz", **pred)


def select_candidate(raw, row, history):
    # 공통 NMS와 원본 좌표 복원을 적용한 뒤 기존 배포 규칙으로 선택한다.
    boxes = onnx_plate.candidates(raw, SHAPE[2:], (row.height, row.width))
    candidates = pd.DataFrame(boxes[:, :4], columns=BOX)
    candidates["confidence"] = boxes[:, 4]
    candidates["video_id"], candidates["frame_index"] = row.video_id, row.frame_index
    return choose_plate(
        history, candidates, row.video_id, row.frame_index, row.width, row.height, "continuity"
    )


RUNTIME_CASES = {
    "pytorch": ("pytorch", "pytorch", "pytorch"),
    "onnx": ("onnx", "onnx", "onnx"),
    "cutmix-fp16": ("pytorch", "trt", "pytorch"),
    "cosine-fp16": ("pytorch", "pytorch", "trt"),
    "aux-fp16": ("pytorch", "trt", "trt"),
    "trt": ("trt", "trt", "trt"),
}


def score(folder, engine):
    from contextlib import ExitStack
    from wlcv.evaluation.box_size_review import iou

    manifest = pd.read_csv(folder / "local_frames.csv")
    videos, boxes, _ = review_inputs(DATASET / "data.yaml")
    truth = boxes.set_index(KEY, verify_integrity=True)
    hubs = pd.read_csv(LABELS).set_index(KEY, verify_integrity=True)
    spec = json.loads((folder / "gpu_input/jobs.json").read_text())
    records, points, people = [], [], []
    with ExitStack() as stack:
        outputs = {}
        for name, runtime in zip(("baseline", "cutmix", "cosine"), RUNTIME_CASES[engine], strict=True):
            src = folder / "gpu_result" if runtime == "trt" else folder
            file = src / f"{runtime}-{name}.npz"
            outputs[name] = stack.enter_context(np.load(file, allow_pickle=False))
        for name, predictions in outputs.items():
            if set(predictions.files) != set(spec["jobs"][name]):
                raise ValueError(f"{engine} {name} 결과와 입력 프레임이 다릅니다.")
        for video in videos:
            histories = {name: PlateHistory() for name in ("baseline", "cutmix")}
            endpoints, manual_y, clip = [], [], []
            rows = manifest[manifest.video_id.eq(video)]
            if rows.frame_index.tolist() != list(range(len(rows))):
                raise ValueError("시간 연결은 영상의 0번부터 순서대로 진행해야 합니다.")
            for row in rows.itertuples():
                track, selected, source = None, "none", "missing"
                for name, history in histories.items():
                    candidate, origin = select_candidate(outputs[name][row.key], row, history)
                    if track is None and candidate is not None:
                        track, selected, source = candidate, name, origin
                coords = np.full(4, np.nan) if track is None else track[BOX].to_numpy(float)
                cy = np.nan if track is None else float(track.cy)
                key = (video, row.frame_index)
                has_label = key in truth.index and truth.loc[key, BOX].notna().all()
                overlap = iou(coords, truth.loc[key, BOX].to_numpy(float)) if has_label else np.nan
                item = dict(
                    video_id=video,
                    frame_index=row.frame_index,
                    in_motion=row.in_motion,
                    scored=bool(has_label),
                    hit=bool(has_label and overlap >= 0.50),
                    detected=track is not None,
                    iou=overlap,
                    selected_model=selected,
                    source=source,
                    cy=cy,
                    **dict(zip(BOX, coords)),
                )
                clip.append(item)
                if row.endpoint:
                    cosine, _ = select_candidate(outputs["cosine"][row.key], row, PlateHistory())
                    point = check_endpoint(row.frame_index, coords, cy, selected, cosine)
                    endpoints.append(point)
                    points.append(dict(video_id=video, **point))
                    click = hubs.loc[key]
                    if click.hub_source not in ("clicked", "previous_blind"):
                        raise ValueError("직접 찍은 허브가 필요합니다.")
                    manual_y.append(float(click.manual_y))
            records.extend(clip)
            motion = pd.DataFrame(clip).query("in_motion")
            judged = motion[motion.scored]
            gap = ~motion.detected
            method, rise = select_rise(endpoints)
            manual = manual_y[0] - manual_y[1]
            people.append(
                dict(
                    video_id=video,
                    scored=len(judged),
                    tp=int(judged.hit.sum()),
                    fn=int((~judged.hit).sum()),
                    fp=int((judged.detected & ~judged.hit).sum()),
                    gap=int(gap.sum()),
                    longest=int(gap.groupby((~gap).cumsum()).sum().max()),
                    source=method,
                    rise_px=rise,
                    manual_px=manual,
                    error_px=None if rise is None else rise - manual,
                )
            )
    athletes = pd.DataFrame(people)
    if athletes.scored.sum() != 270 or len(athletes) != 6:
        raise ValueError("동작 중 수동 박스 270장과 검증 6명을 확인하세요.")
    for name, rows in [("frames", records), ("endpoints", points), ("athletes", people)]:
        pd.DataFrame(rows).to_csv(folder / f"{engine}-{name}.csv", index=False)
    print(
        engine,
        athletes[["scored", "tp", "fn", "fp", "gap"]].sum().to_dict(),
        "measured",
        athletes.error_px.count(),
        "MAE",
        athletes.error_px.abs().mean(),
        flush=True,
    )


def compare(folder):
    """엔진이 바뀌며 복구하거나 잃은 프레임과 선수별 상승량 차이를 확인한다."""
    spec = json.loads((folder / "gpu_input/jobs.json").read_text())
    record = json.loads((folder / "gpu_result/real_run.json").read_text())
    if (
        record["graph_sha256"] != spec["graph_sha256"]
        or record["input_sha256"] != spec["input_sha256"]
        or record["device"] != "NVIDIA GeForce RTX 3090"
        or record["tf32"]
    ):
        raise ValueError("입력, 그래프, GPU 설정이 현재 실행과 다릅니다.")
    for name, stats in record["models"].items():
        if stats["frames"] != len(spec["jobs"][name]) or stats["half_layers"] < 1:
            raise ValueError(f"{name}의 FP16 실행 기록을 확인하세요.")

    reference = pd.read_csv(folder / "pytorch-frames.csv").set_index(KEY).sort_index()
    summaries, cases = [], []
    for runtime in RUNTIME_CASES:
        frames = pd.read_csv(folder / f"{runtime}-frames.csv").set_index(KEY).sort_index()
        if not frames.index.equals(reference.index) or not frames[["scored", "in_motion"]].equals(
            reference[["scored", "in_motion"]]
        ):
            raise ValueError("채점 프레임 또는 동작 구간이 다릅니다.")
        active = frames.in_motion
        judged = active & frames.scored
        gained = judged & ~reference.hit & frames.hit
        lost = judged & reference.hit & ~frames.hit
        changed = active & frames.selected_model.ne(reference.selected_model)
        cases.append(
            frames.loc[gained | lost | changed].assign(engine=runtime, gained=gained, lost=lost)
        )
        athletes = pd.read_csv(folder / f"{runtime}-athletes.csv")
        endpoints = pd.read_csv(folder / f"{runtime}-endpoints.csv")
        summaries.append(
            dict(
                engine=runtime,
                judged=int(judged.sum()),
                tp=int(frames.loc[judged, "hit"].sum()),
                fn=int((~frames.loc[judged, "hit"]).sum()),
                fp=int((judged & frames.detected & ~frames.hit).sum()),
                motion=int(active.sum()),
                gap=int((active & ~frames.detected).sum()),
                longest=int(athletes.longest.max()),
                gained=int(gained.sum()),
                lost=int(lost.sum()),
                changed_model=int(changed.sum()),
                endpoints=int(endpoints.accepted.sum()),
                cosine_rises=int(athletes.source.eq("cosine").sum()),
                measured=int(athletes.error_px.count()),
                mae=athletes.error_px.abs().mean(),
            )
        )
    pd.concat(cases).to_csv(folder / "changes.csv")
    summary = pd.DataFrame(summaries)
    summary.to_csv(folder / "summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("step", choices=["prepare", "controls", *RUNTIME_CASES, "compare"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.step == "prepare":
        prepare(args.folder)
    elif args.step == "controls":
        controls(args.folder)
    elif args.step == "compare":
        print(compare(args.folder).to_string(index=False))
    else:
        score(args.folder, args.step)
