"""업로드 시 후보를 제안하고, 저장한 끝점 좌표로 다시 코칭 수치를 계산한다."""
from time import perf_counter

import numpy as np

from wlcv.data.mp4_input import VideoInputError, validate_events
from wlcv.runtime.inference import detect_endpoint
from wlcv.runtime.measurements import check_endpoint, coaching_summary, knee_contact_measurement, phase_summary
from wlcv.runtime.pull_phases import PULL_KEYS, suggest_from_rows


def cache_endpoints(stream, model, video, device, checks, timings):
    """원본을 임시 폴더에서 지우기 전에 사용자가 선택할 수 있는 프레임의 좌표를 저장한다."""
    for ms, image, plate, people in stream:
        start = perf_counter()
        candidate = None
        if model is not None and np.isfinite(plate.cy):
            candidate = detect_endpoint(model, image, video, plate.frame_index, device)
        box = np.array([plate.x1, plate.y1, plate.x2, plate.y2])
        checks[plate.frame_index] = check_endpoint(plate.frame_index, box, plate.cy, plate.selected_model,
                                                   candidate, model_loaded=model is not None)
        timings["cosine_measurement"] += perf_counter() - start
        yield ms, image, plate, people


def initial_review(rows, manual):
    suggested, reasons = suggest_from_rows(rows)
    effective = dict(manual)
    sources = {}
    for key in PULL_KEYS:
        if key in manual:
            sources[key] = "manual"
        elif key != "drive" and suggested[key] is not None:
            trial = {**effective, key: suggested[key]}
            try:
                validate_events(trial, len(rows))
            except VideoInputError:
                reasons[key] += " · 입력 시점과 순서가 맞지 않아 보류"
            else:
                effective = trial
                sources[key] = "automatic"
    selected = {key: effective[key] for key in PULL_KEYS if key in effective}
    return {"suggested": suggested, "reasons": reasons, "events": selected, "sources": sources,
            "status": review_status(sources)}


def review_status(sources):
    checked = all(sources.get(k) in ("manual", "reviewed") for k in ("lift_off", "knee_pass"))
    return "confirmed" if checked else "needs_review"


def revised_review(previous, updates, other_events, count):
    events = {**previous["events"], **updates}
    if previous["sources"].get("drive") == "automatic" and "drive" not in updates:
        events.pop("drive", None)
    validate_events({**other_events, **events}, count)
    sources = {**previous["sources"], **dict.fromkeys(updates, "reviewed")}
    sources = {k: sources[k] for k in events}
    return {**previous, "events": events, "sources": sources, "status": review_status(sources)}


def reviewed_metrics(rows, events, checks, cosine_weights, direction):
    lift, knee, drive = (events.get(key) for key in PULL_KEYS)
    rise = knee_contact_measurement(knee, drive, checks, cosine_weights)
    coaching = coaching_summary(rows, lift, knee, events, direction)
    angles = rows.set_index("frame_index").trunk_angle_deg.reindex([lift, knee])
    change = float(angles.iloc[1] - angles.iloc[0]) if angles.notna().all() else None
    return {"knee_to_contact_rise": rise, "manual_phase_interval": phase_summary(rows, lift, knee),
            "trunk_change_deg": change, "coaching": coaching}
