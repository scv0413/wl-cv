"""같은 c04 영상에서 ORT의 연산 스레드 수만 바꾸어 두 모델의 지연을 측정한다."""
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import ROOT, SPLIT, VIDEOS
from wlcv.evaluation.onnx_plate import candidates
from wlcv.evaluation.onnx_pose import WEIGHTS

VIDEO = "c04_sn_f_01"
ORDER = (4, 2, 6, 6, 2, 4)
WARMUP = 10
STAGES = ["decode", "prepare", "plate", "pose", "post", "total"]


def cpu_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    entry, = session.get_inputs()
    if entry.name != "images" or entry.shape != [1, 3, 640, 384] or entry.type != "tensor(float)":
        raise ValueError(f"{path.name}의 입력이 기존 실험과 다릅니다.")
    return session


def outputs(plate, pose, predictor, bgr):
    """프레임을 한 번 전처리해서 두 세션에 넣고 원본 영상 좌표를 복원한다."""
    start = perf_counter_ns()
    tensor = predictor.preprocess([bgr])
    feed = {"images": tensor.numpy()}
    prepared = perf_counter_ns()
    plate_raw, = plate.run(None, feed)
    plate_done = perf_counter_ns()
    pose_raw, = pose.run(None, feed)
    pose_done = perf_counter_ns()
    boxes = candidates(plate_raw, tensor.shape[2:], bgr.shape)
    person, = predictor.postprocess(torch.from_numpy(pose_raw), tensor, [bgr])
    values = (boxes, person.boxes.data.numpy(), person.keypoints.data.numpy())
    end = perf_counter_ns()
    times = np.diff([start, prepared, plate_done, pose_done, end]) / 1e6
    return values, times, end


def compare_outputs(current, reference):
    """NMS 후 모든 원판·사람 후보와 17개 관절을 반환 순서대로 대조한다."""
    if any(a.shape != b.shape for a, b in zip(current, reference)):
        return False, np.nan, np.nan
    plate, person, joints = [np.abs(a - b) for a, b in zip(current, reference)]
    coord = max(plate[:, :4].max(initial=0), person[:, :4].max(initial=0),
                np.linalg.norm(joints[:, :, :2], axis=2).max(initial=0))
    score = max(plate[:, 4].max(initial=0), person[:, 4].max(initial=0),
                joints[:, :, 2].max(initial=0))
    statuses_match = (
        np.array_equal(current[0][:, 4] >= .1, reference[0][:, 4] >= .1)
        and np.array_equal(current[2][:, :, 2] >= .5, reference[2][:, :, 2] >= .5)
    )
    return bool(statuses_match), float(coord), float(score)


@torch.inference_mode()
def benchmark(plate_path, pose_path, folder):
    for path in (plate_path, pose_path):
        settings = json.loads((path.parent / "settings.json").read_text())
        if sha256(path.read_bytes()).hexdigest() != settings["onnx_sha256"]:
            raise ValueError(f"{path}의 해시가 이전 실행 기록과 다릅니다.")
    split = pd.read_csv(SPLIT).set_index("video_id")
    info = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id").loc[VIDEO]
    if split.at[VIDEO, "role"] != "validation":
        raise ValueError("검증 영상이 필요합니다.")
    video = VIDEOS / info.filename
    capture = cv2.VideoCapture(str(video))
    try:
        ok, first = capture.read()
        if not ok:
            raise OSError(f"{video} 첫 장을 열 수 없습니다.")
    finally:
        capture.release()

    # 47번의 원본 해상도 전처리와 자세 후처리를 재사용한다. 최초 설정은 측정 밖이다.
    model = YOLO(str(WEIGHTS))
    model.predict(first, imgsz=640, conf=.25, iou=.7, device="cpu", verbose=False)
    predictor = model.predictor
    torch.set_num_threads(4)
    rows, reference = [], []
    for trial, threads in enumerate(tqdm(ORDER, desc="CPU 스레드 비교"), 1):
        plate = cpu_session(plate_path, threads)
        pose = cpu_session(pose_path, threads)
        if plate.get_outputs()[0].shape != [1, 5, 5040] or pose.get_outputs()[0].shape != [1, 56, 5040]:
            raise ValueError("46·47번의 원시 출력 그래프를 선택하세요.")
        for _ in range(WARMUP):
            outputs(plate, pose, predictor, first)
        capture = cv2.VideoCapture(str(video))
        frame = 0
        try:
            while True:
                start = perf_counter_ns()
                ok, image = capture.read()
                decoded = perf_counter_ns()
                if not ok:
                    break
                prediction, times, end = outputs(plate, pose, predictor, image)
                if trial == 1:
                    reference.append(tuple(array.copy() for array in prediction))
                same, coord, score = compare_outputs(prediction, reference[frame])
                rows.append((trial, threads, frame, (decoded-start)/1e6, *times, (end-start)/1e6,
                             same, coord, score))
                frame += 1
        finally:
            capture.release()
        if frame != info.frame_count:
            raise ValueError(f"{trial}회 디코딩한 {frame}장이 메타데이터와 다릅니다.")
    detail = pd.DataFrame(rows, columns=["trial", "threads", "frame", *STAGES,
                                        "same_counts_and_states", "coordinate_delta_px", "score_delta"])
    detail.to_csv(folder / "frames.csv", index=False)
    settings = dict(video=VIDEO, order=ORDER, warmup=WARMUP, torch_threads=4,
                    cpu_provider="CPUExecutionProvider", ort_version=ort.__version__,
                    threads_spinning=False, plate=str(plate_path), pose=str(pose_path),
                    graph_optimization=ort.SessionOptions().graph_optimization_level.name)
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    return detail
