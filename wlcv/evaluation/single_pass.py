"""원본 한 프레임에서 원판과 자세를 차례로 추론하고 다음 프레임으로 넘어간다."""
from types import SimpleNamespace
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.pose import PosePredictor

from wlcv.data.frames import timed_frames_rgb
from wlcv.evaluation.plate_track import BOX, PlateHistory, prepare_candidates


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


def single_pass_frames(path, plate_model, pose_model, device, share_input, timings, *, plate_imgsz=640):
    history = PlateHistory()
    options = dict(device=device, verbose=False)
    plate_predictor = PlateInput if share_input else None
    pose_predictor = shared_pose_predictor(plate_model) if share_input else None
    for index, (time_ms, rgb) in enumerate(timed_frames_rgb(path)):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        start = perf_counter()
        plate_result, = plate_model.predict(bgr, conf=.001, imgsz=plate_imgsz, predictor=plate_predictor, **options)
        boxes = plate_result.boxes.cpu()
        detected = perf_counter()
        # 일괄 처리와 같은 정렬·반경·이력 규칙을 현재 프레임 후보에 적용한다.
        candidates = pd.DataFrame(boxes.xyxy.tolist(), columns=BOX)
        candidates["confidence"] = boxes.conf.tolist()
        candidates["video_id"] = path.stem
        candidates["frame_index"] = index
        _, chosen, source = history.choose(prepare_candidates(candidates), path.stem, index)
        values = {k: np.nan if chosen is None else chosen[k] for k in ["confidence", "cx", "cy", *BOX]}
        plate = SimpleNamespace(video_id=path.stem, frame_index=index, source=source, **values)
        tracked = perf_counter()
        pose_result, = pose_model.predict(bgr, conf=.25, imgsz=640, predictor=pose_predictor, **options)
        result = pose_result.cpu().numpy()
        timings["plate_inference"] += detected - start
        timings["plate_tracking"] += tracked - detected
        timings["pose"] += perf_counter() - tracked
        yield time_ms, bgr, plate, result
