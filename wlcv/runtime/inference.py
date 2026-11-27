"""원본 한 프레임에서 원판과 자세를 차례로 추론하고 다음 프레임으로 넘어간다."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.pose import PosePredictor

from wlcv.data.frames import timed_frames_rgb
from wlcv.runtime.foreground import performer_plates, select_performer
from wlcv.runtime.tracking import BOX, PlateHistory, choose_plate


def load_plate_model(weights):
    path = Path(weights).resolve(strict=True)
    model = YOLO(str(path), task="detect")
    if path.suffix == ".engine":
        # 엔진을 두 번 GPU에 올리지 않고 준비 단계에서 저장한 클래스 이름을 읽는다.
        from ultralytics.nn.backends.base import BaseBackend

        _, metadata = BaseBackend.engine_header(path)
        names = {int(key): value for key, value in metadata.get("names", {}).items()}
    else:
        names = model.names
    if names != {0: "plate"}:
        raise ValueError("원판 탐지 가중치가 필요합니다.")
    return model


def predict_rgb(model, frame, *, conf=0.25, imgsz=640, device="cpu"):
    """RGB 프레임을 Ultralytics의 BGR 배열 입력으로 변환한다."""
    return model.predict(frame[:, :, ::-1].copy(), imgsz=imgsz, conf=conf,
                         device=device, verbose=False)[0]


class PlateInput(DetectionPredictor):
    def preprocess(self, images):
        tensor = super().preprocess(images)
        self.prepared_frame = (images[0], tensor)
        return tensor


def shared_pose_predictor(plate_model):
    """Ultralytics가 영상 내내 같은 추론 객체를 재사용하도록 클래스를 한 번 만든다."""
    class PoseWithPlateInput(PosePredictor):
        def preprocess(self, images):
            plate = plate_model.predictor
            frame, tensor = plate.prepared_frame
            if (images[0] is not frame or self.imgsz != plate.imgsz or self.device != plate.device
                    or self.model.stride != plate.model.stride or self.args.rect != plate.args.rect
                    or self.scale_fill != plate.scale_fill or self.model.fp16 != plate.model.fp16):
                raise ValueError("두 모델이 같은 프레임과 입력 크기·형식을 사용해야 합니다.")
            plate.prepared_frame = None
            return tensor

    return PoseWithPlateInput


def plate_candidate_frames(path, plate_model, device, timings, cutmix_model, plate_imgsz, share_input):
    models = {"baseline": plate_model}
    if cutmix_model is not None:
        models["cutmix"] = cutmix_model
    for index, (time_ms, rgb) in enumerate(timed_frames_rgb(path)):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        candidates = {}
        start = perf_counter()
        for name, model in models.items():
            predictor = PlateInput if share_input and name == "baseline" else None
            result, = model.predict(bgr, conf=.001, imgsz=plate_imgsz, device=device,
                                    verbose=False, predictor=predictor)
            boxes = result.boxes.cpu()
            table = pd.DataFrame(boxes.xyxy.tolist(), columns=BOX)
            table["confidence"] = boxes.conf.tolist()
            table["video_id"], table["frame_index"] = path.stem, index
            candidates[name] = table
        timings["plate_inference"] += perf_counter() - start
        yield time_ms, bgr, candidates


def track_candidates(tables, histories, video, index, shape, edge_rule, people=(), selected=None):
    chosen, source, selected_model = None, "missing", "none"
    height, width = shape[:2]
    for name, table in tables.items():
        candidates = performer_plates(table, people, selected)
        candidate, origin = choose_plate(histories[name], candidates, video, index,
                                         width, height, edge_rule)
        if chosen is None and candidate is not None:
            chosen, source, selected_model = candidate, origin, name
    values = {k: np.nan if chosen is None else chosen[k] for k in ["confidence", "cx", "cy", *BOX]}
    return SimpleNamespace(video_id=video, frame_index=index, source=source,
                           selected_model=selected_model, **values)


def tracked_plate_frames(path, plate_model, device, timings, *, cutmix_model=None,
                         plate_imgsz=640, share_input=False, edge_rule="continuity"):
    """원판 단독 실험용. 주·보조 모델이 각자의 이력을 유지한다."""
    frames = plate_candidate_frames(path, plate_model, device, timings, cutmix_model,
                                    plate_imgsz, share_input)
    histories = {"baseline": PlateHistory(), "cutmix": PlateHistory()}
    for index, (time_ms, bgr, tables) in enumerate(frames):
        start = perf_counter()
        plate = track_candidates(tables, histories, path.stem, index, bgr.shape, edge_rule)
        timings["plate_tracking"] += perf_counter() - start
        yield time_ms, bgr, plate


def reread_candidates(path, frames, timings):
    """두 번 읽기 모드에서는 원본 영상 대신 첫 번째 패스의 원판 후보만 메모리에 보관한다."""
    start = perf_counter()
    cached = [(time_ms, tables) for time_ms, _, tables in frames]
    if "plate_load" in timings:
        timings["plate_pass"] = timings.pop("plate_load") + perf_counter() - start
        del timings["plate_inference"]
    for (time_ms, rgb), (first_time, tables) in zip(timed_frames_rgb(path), cached, strict=True):
        if time_ms != first_time:
            raise ValueError("두 번 읽은 영상 프레임의 시각이 다릅니다.")
        yield time_ms, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), tables


def single_pass_frames(path, plate_model, pose_model, device, share_input, timings, *,
                       plate_imgsz=640, cutmix_model=None, edge_rule="off", reread=False):
    pose_predictor = shared_pose_predictor(plate_model) if share_input else None
    frames = plate_candidate_frames(path, plate_model, device, timings, cutmix_model,
                                    plate_imgsz, share_input)
    if reread:
        frames = reread_candidates(path, frames, timings)
    histories = {"baseline": PlateHistory(), "cutmix": PlateHistory()}
    previous = None
    for index, (time_ms, bgr, tables) in enumerate(frames):
        start = perf_counter()
        pose_result, = pose_model.predict(bgr, conf=.25, imgsz=640, device=device,
                                          predictor=pose_predictor, verbose=False)
        result = pose_result.cpu().numpy()
        people = result.boxes.xyxy
        selected, reason = select_performer(people, previous)
        previous = None if selected is None else people[selected].copy()
        detected = perf_counter()
        timings["pose"] += detected - start
        plate = track_candidates(tables, histories, path.stem, index, bgr.shape,
                                 edge_rule, people, selected)
        plate.performer_number = selected + 1 if selected is not None else None
        plate.performer_selection = reason
        timings["plate_tracking"] += perf_counter() - detected
        yield time_ms, bgr, plate, result


def detect_endpoint(model, bgr, video, index, device):
    """깨끗한 원본 한 장에서만 검출한다. 직전 프레임의 이력은 사용하지 않는다."""
    result, = model.predict(bgr, conf=0.001, imgsz=640, device=device, verbose=False)
    boxes = result.boxes.cpu()
    candidates = pd.DataFrame(boxes.xyxy.tolist(), columns=BOX)
    candidates["confidence"] = boxes.conf.tolist()
    candidates["video_id"], candidates["frame_index"] = video, index
    height, width = bgr.shape[:2]
    chosen, _ = choose_plate(PlateHistory(), candidates, video, index, width, height, "continuity")
    return chosen
