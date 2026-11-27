"""원본 픽셀과 프레임 시각으로 계산하는 스내치 측정값."""
from math import atan2, degrees

import numpy as np
import pandas as pd

from wlcv.runtime.apex import drive_to_peak
from wlcv.runtime.inference import detect_endpoint
from wlcv.runtime.tracking import BOX, iou


def vertical_angle(shoulder, hip):
    """원본 좌표로 위쪽 수직선과의 각도를 구한다. 모델의 소수 픽셀 좌표도 유지한다."""
    dx = shoulder[0] - hip[0]
    up = hip[1] - shoulder[1]
    if dx == 0 and up == 0:
        return None
    return degrees(atan2(abs(dx), up))


def vertical_speeds(time_ms, height_y):
    """속도는 원본 px/s, 위쪽이 양수. 1 px 이하의 후반 상승량은 비율 계산을 보류한다."""
    dt = np.diff(np.asarray(time_ms, dtype=float)) / 1000
    if len(dt) != 2 or not np.isfinite(dt).all() or (dt <= 0).any():
        return dict(early=np.nan, late=np.nan, ratio=np.nan, reason="시각 확인 필요")
    rise = -np.diff(np.asarray(height_y, dtype=float))
    early, late = rise / dt
    if not np.isfinite(rise).all():
        reason = "끝점 누락"
    elif rise[1] <= 1:
        reason = "후반 상승량 1 px 이하"
    else:
        reason = ""
    return dict(early=early, late=late, ratio=np.nan if reason else early/late, reason=reason)


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
    """두 시점이 모두 통과하면 코사인, 하나라도 보류되면 양쪽 모두 추적 좌표를 쓴다."""
    if len(points) != 2:
        raise ValueError("무릎 통과·접촉 두 시점이 필요합니다.")
    if all(p["accepted"] for p in points):
        return "cosine", points[0]["cosine_y_px"] - points[1]["cosine_y_px"]
    if all(p["track_y_px"] is not None for p in points):
        return "tracking", points[0]["track_y_px"] - points[1]["track_y_px"]
    return None, None


def phase_summary(records, lift, knee):
    """원본의 y는 아래로 증가하므로 이륙 y에서 무릎 통과 y를 뺀다."""
    if lift is None or knee is None:
        return {"lift_off_frame": lift, "knee_pass_frame": knee, "rise_px": None,
                "missing_endpoint_frames": [], "reason": "이륙 또는 무릎 통과 시점 미지정"}
    interval = records.set_index("frame_index").loc[lift:knee]
    first, last = interval.loc[lift], interval.loc[knee]
    missing = [int(i) for i in [lift, knee] if pd.isna(interval.at[i, "plate_cy_px"])]
    rise = None if missing else round(float(first.plate_cy_px - last.plate_cy_px), 3)
    return {
        "lift_off_frame": lift, "knee_pass_frame": knee,
        "lift_off_y_px": None if lift in missing else float(first.plate_cy_px),
        "knee_pass_y_px": None if knee in missing else float(last.plate_cy_px),
        "rise_px": rise, "missing_endpoint_frames": missing,
        "observed_frames": int(interval.plate_cy_px.count()), "interval_frames": len(interval),
    }


def coaching_summary(rows, lift, knee, events, facing):
    """원본 시각과 모델 관측으로 계산하며 측정할 수 없는 값은 JSON null로 남긴다."""
    frames = rows.set_index("frame_index", verify_integrity=True)
    result = {"units": "source px, source-video seconds", "landmarks": "model plate box center and near shoulder"}
    contact = events.get("drive")
    if lift is not None and knee is not None and contact in frames.index and lift < knee < contact:
        endpoints = frames.loc[[lift, knee, contact]]
        values = vertical_speeds(endpoints.source_time_ms, endpoints.plate_cy_px)
        result["pulls"] = {k: (float(v) if np.isfinite(v) else None) for k, v in values.items() if k != "reason"}
        result["pulls"].update(reason=values["reason"], endpoint_frames=[int(lift), int(knee), int(contact)],
                               plate_states=endpoints.plate_status.tolist())
    else:
        result["pulls"] = {"early": None, "late": None, "ratio": None, "reason": "이륙·무릎 통과·접촉 중 미지정 시점이 있습니다."}
    catch = events.get("catch")
    result["drive_to_peak"] = drive_to_peak(frames, contact, catch)
    if catch not in frames.index:
        result["catch"] = {"forward_px": None, "up_px": None, "reason": "캐치 시점 미지정"}
        return result
    point = frames.loc[catch]
    info = {"frame": int(catch), "forward_px": None, "up_px": None}
    if point[["plate_cx_px", "plate_cy_px", "shoulder_x_px", "shoulder_y_px"]].isna().any():
        info["reason"] = "원판 또는 선택된 수행자의 가까운 어깨 관측 누락"
    else:
        info["up_px"] = float(point.shoulder_y_px - point.plate_cy_px)
        if facing in (-1, 1):
            info["forward_px"] = float(facing * (point.plate_cx_px - point.shoulder_x_px))
            info["reason"] = ""
        else:
            info["reason"] = "화면에서 선수가 향한 방향 미확인"
    result["catch"] = info
    return result


def measure_knee_to_contact(rows, video, knee, contact, images, cosine, weights, device):
    """무릎 통과→접촉 상승량과 두 시점의 채택·보류 이유를 기록한다."""
    result = {
        "knee_pass_frame": knee, "contact_frame": contact, "source": None, "rise_px": None,
        "endpoints": [], "cosine_weights": None, "reason": "무릎 통과 또는 접촉 시점 미지정",
    }
    if knee is None or contact is None:
        return result

    tracks = rows.set_index("frame_index")
    for index in (knee, contact):
        tracked = tracks.loc[index]
        candidate = None
        if cosine is not None:
            candidate = detect_endpoint(cosine, images[index], video, index, device)
        box = tracked[["plate_x1", "plate_y1", "plate_x2", "plate_y2"]].to_numpy(float)
        result["endpoints"].append(check_endpoint(
            index, box, tracked.plate_cy_px, tracked.selected_model, candidate,
            model_loaded=cosine is not None,
        ))

    result["source"], result["rise_px"] = select_rise(result["endpoints"])
    result["reason"] = " · ".join(
        f"{point['frame_index']}번 {point['reason']}" for point in result["endpoints"] if point["reason"]
    )
    result["cosine_weights"] = str(weights) if cosine is not None else None
    return result
