"""무릎 통과와 접촉 두 장에서 코사인 모델로 원판 상승량을 측정한다."""
import numpy as np
import pandas as pd

from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.plate_track import BOX, PlateHistory, choose_plate


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


def check_endpoint(index, track_box, track_y, tracking_model, candidate, *, model_loaded=True):
    point = {"frame_index": int(index), "tracking_model": tracking_model,
             "track_y_px": float(track_y) if np.isfinite(track_y) else None,
             "cosine_y_px": None if candidate is None else float(candidate.cy),
             "inter_model_iou": None, "accepted": False, "reason": ""}
    if point["track_y_px"] is None or not np.isfinite(track_box).all():
        point["reason"] = "추적 박스 누락"
    elif not model_loaded:
        point["reason"] = "코사인 가중치 없음"
    elif candidate is None:
        point["reason"] = "코사인 직접 검출 없음"
    else:
        overlap = float(iou(track_box, candidate[BOX].to_numpy(float)))
        point["inter_model_iou"] = overlap
        point["accepted"] = overlap >= 0.50
        if not point["accepted"]:
            point["reason"] = "두 모델의 박스 IoU 0.50 미만"
    return point


def select_rise(points):
    """한 쪽만 통과해도 양 끝을 같은 추적 방식으로 계산한다."""
    if len(points) != 2:
        raise ValueError("무릎 통과·접촉 두 시점이 필요합니다.")
    if all(p["accepted"] for p in points):
        return "cosine", points[0]["cosine_y_px"] - points[1]["cosine_y_px"]
    if all(p["track_y_px"] is not None for p in points):
        return "tracking", points[0]["track_y_px"] - points[1]["track_y_px"]
    return None, None
