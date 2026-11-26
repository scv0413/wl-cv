"""Mac에서 같은 원판·자세 모델의 PyTorch CPU, MPS, ONNX Runtime CPU를 비교한다."""
from copy import deepcopy
from hashlib import sha256
import json
from time import perf_counter_ns

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.analyze import read_phases
from wlcv.config import ROOT, SPLIT, VIDEOS
from wlcv.evaluation.onnx_latency import VIDEO, WARMUP, compare_outputs, cpu_session
from wlcv.evaluation.onnx_plate import WEIGHTS as PLATE_WEIGHTS, candidates
from wlcv.evaluation.onnx_pose import WEIGHTS as POSE_WEIGHTS
from wlcv.evaluation.plate_track import BOX, KEY, select_centers
from wlcv.evaluation.pose import JOINTS
from wlcv.evaluation.trunk import vertical_angle

ORDER = ("pt_cpu", "pt_mps", "ort_cpu", "ort_cpu", "pt_mps", "pt_cpu")


def load_models(plate_path, pose_path, first):
    models = {"pt_cpu": [], "pt_mps": [], "ort_cpu": []}
    for weights, path in [(PLATE_WEIGHTS, plate_path), (POSE_WEIGHTS, pose_path)]:
        meta = json.loads((path.parent / "settings.json").read_text())
        if (sha256(weights.read_bytes()).hexdigest() != meta["weights_sha256"] or
                sha256(path.read_bytes()).hexdigest() != meta["onnx_sha256"]):
            raise ValueError("46·47번과 같은 가중치와 그래프인지 확인하세요.")
        yolo = YOLO(str(weights))
        yolo.predict(first, imgsz=640, conf=.25, iou=.7, device="cpu", verbose=False)
        predictor = yolo.predictor
        native = predictor.model.backend.model
        models["pt_cpu"].append(native)
        models["pt_mps"].append(deepcopy(native).to("mps").eval())
        models["ort_cpu"].append(cpu_session(path, threads=4))
    return models, predictor  # 마지막으로 읽은 자세 모델의 전처리·후처리


def forward_pair(runtime, models, tensor):
    if runtime == "ort_cpu":
        feed = {"images": tensor.numpy()}
        return [torch.from_numpy(model.run(None, feed)[0]) for model in models]
    if runtime == "pt_mps":
        tensor = tensor.to("mps")
    outputs = []
    for model in models:
        raw = model(tensor)
        outputs.append(raw[0] if isinstance(raw, (tuple, list)) else raw)
    if runtime == "pt_mps":
        torch.mps.synchronize()
    return [raw.cpu() for raw in outputs]


def infer_frame(runtime, models, predictor, bgr):
    tensor = predictor.preprocess([bgr])
    raw_plate, raw_pose = forward_pair(runtime, models, tensor)
    if tuple(raw_plate.shape) != (1, 5, 5040) or tuple(raw_pose.shape) != (1, 56, 5040):
        raise ValueError("기존 FP32 원시 출력의 형태를 확인하세요.")
    boxes = candidates(raw_plate.numpy(), tensor.shape[2:], bgr.shape)
    person, = predictor.postprocess(raw_pose, tensor, [bgr])
    return boxes, person.boxes.data.numpy(), person.keypoints.data.numpy()


def trunk_angle(people, points, side):
    if side not in ("left", "right") or len(people) != 1:
        return np.nan
    i = int(side == "right")
    shoulder, hip = points[0, [JOINTS["어깨"][i], JOINTS["골반"][i]]]
    if min(shoulder[2], hip[2]) < .50 or not np.any(shoulder[:2]) or not np.any(hip[:2]):
        return np.nan
    angle = vertical_angle(shoulder[:2], hip[:2])
    return np.nan if angle is None else angle


@torch.inference_mode()
def benchmark(plate_path, pose_path, folder):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal GPU를 사용할 수 있는 Mac의 PyTorch 커널로 실행하세요.")
    split = pd.read_csv(SPLIT).set_index("video_id")
    if split.at[VIDEO, "role"] != "validation":
        raise ValueError("기존 검증 영상인지 확인하세요.")
    info = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id").loc[VIDEO]
    path = VIDEOS / info.filename
    video = cv2.VideoCapture(str(path))
    try:
        ok, first = video.read()
        if not ok:
            raise OSError(f"{path} 첫 프레임을 읽지 못했습니다.")
    finally:
        video.release()
    models, predictor = load_models(plate_path, pose_path, first)
    torch.set_num_threads(4)
    rows, reference, summaries = [], [], []
    lift, knee = read_phases(VIDEO)
    for trial, runtime in enumerate(tqdm(ORDER, desc="Mac CPU·GPU 비교"), 1):
        for _ in range(WARMUP):
            infer_frame(runtime, models[runtime], predictor, first)
        video = cv2.VideoCapture(str(path))
        boxes_table, angles = [], []
        try:
            for frame in range(info.frame_count):
                start = perf_counter_ns()
                ok, bgr = video.read()
                if not ok:
                    raise OSError(f"{frame}번 디코딩 실패")
                values = infer_frame(runtime, models[runtime], predictor, bgr)
                elapsed = (perf_counter_ns() - start) / 1e6
                if trial == 1:
                    reference.append(tuple(a.copy() for a in values))
                same, coord, score = compare_outputs(values, reference[frame])
                rows.append((runtime, trial, frame, elapsed, same, coord, score))
                boxes, people, points = values
                for x1, y1, x2, y2, confidence, _ in boxes:
                    boxes_table.append((VIDEO, frame, confidence, x1, y1, x2, y2))
                angles.append(trunk_angle(people, points, info.near_side))
            if video.read()[0]:
                raise ValueError("영상이 등록된 프레임 수보다 깁니다.")
        finally:
            video.release()
        indices = pd.DataFrame({"video_id":VIDEO, "frame_index":range(info.frame_count)})
        boxes = pd.DataFrame(boxes_table, columns=KEY + ["confidence"] + BOX)
        selected = select_centers(indices, boxes, show_progress=False).query("method == 'Past link'").set_index("frame_index")
        rise = selected.at[lift, "cy"] - selected.at[knee, "cy"]
        angle_change = angles[knee] - angles[lift]
        summaries.append((runtime, trial, selected.source.eq("high").sum(), selected.source.eq("linked").sum(),
                          selected.source.eq("missing").sum(), rise, angle_change))
    rows = pd.DataFrame(rows, columns=["runtime", "trial", "frame", "total_ms", "same_counts_and_states",
                                       "coordinate_delta_px", "score_delta"])
    domain = pd.DataFrame(summaries, columns=["runtime", "trial", "direct", "linked", "missing", "rise_px", "change_deg"])
    rows.to_csv(folder / "frames.csv", index=False)
    domain.to_csv(folder / "domain.csv", index=False)
    settings = dict(video=VIDEO, torch=torch.__version__, order=ORDER, dtype="float32",
                    cpu_threads=4, mps_synchronized=True, warmup=WARMUP, plate=str(plate_path), pose=str(pose_path))
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    return rows, domain
