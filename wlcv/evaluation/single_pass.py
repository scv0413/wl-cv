"""원본 한 프레임에서 원판과 자세를 차례로 추론하고 다음 프레임으로 넘어간다."""
from types import SimpleNamespace
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.pose import PosePredictor

from wlcv.data.frames import timed_frames_rgb
from wlcv.evaluation.plate_track import BOX, PlateHistory, choose_plate


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


def tracked_plate_frames(path, plate_model, device, timings, *, cutmix_model=None,
                         plate_imgsz=640, share_input=False, edge_rule="continuity"):
    """두 모델이 각자의 이력을 갱신한다. 주 모델이 비어 있을 때만 보조 박스를 선택한다."""
    models = [("baseline", plate_model)]
    if cutmix_model is not None:
        models.append(("cutmix", cutmix_model))
    histories = {name: PlateHistory() for name, _ in models}
    for index, (time_ms, rgb) in enumerate(timed_frames_rgb(path)):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        chosen, source, selected_model = None, "missing", "none"
        for name, model in models:
            start = perf_counter()
            predictor = PlateInput if share_input and name == "baseline" else None
            result, = model.predict(bgr, conf=.001, imgsz=plate_imgsz, device=device,
                                    verbose=False, predictor=predictor)
            boxes = result.boxes.cpu()
            detected = perf_counter()
            candidates = pd.DataFrame(boxes.xyxy.tolist(), columns=BOX)
            candidates["confidence"] = boxes.conf.tolist()
            candidates["video_id"] = path.stem
            candidates["frame_index"] = index
            candidate, origin = choose_plate(histories[name], candidates, path.stem, index,
                                             width, height, edge_rule)
            if chosen is None and candidate is not None:
                chosen, source, selected_model = candidate, origin, name
            timings["plate_inference"] += detected - start
            timings["plate_tracking"] += perf_counter() - detected
        values = {k: np.nan if chosen is None else chosen[k] for k in ["confidence", "cx", "cy", *BOX]}
        plate = SimpleNamespace(video_id=path.stem, frame_index=index, source=source,
                                selected_model=selected_model, **values)
        yield time_ms, bgr, plate


def single_pass_frames(path, plate_model, pose_model, device, share_input, timings, *,
                       plate_imgsz=640, cutmix_model=None, edge_rule="off"):
    pose_predictor = shared_pose_predictor(plate_model) if share_input else None
    plates = tracked_plate_frames(path, plate_model, device, timings, cutmix_model=cutmix_model,
                                  plate_imgsz=plate_imgsz, share_input=share_input, edge_rule=edge_rule)
    for time_ms, bgr, plate in plates:
        start = perf_counter()
        pose_result, = pose_model.predict(bgr, conf=.25, imgsz=640, device=device,
                                          predictor=pose_predictor, verbose=False)
        result = pose_result.cpu().numpy()
        timings["pose"] += perf_counter() - start
        yield time_ms, bgr, plate, result
