"""업로드 시 후보를 제안하고, 저장한 끝점 좌표로 다시 코칭 수치를 계산한다."""
from wlcv.data.mp4_input import VideoInputError, validate_events
from wlcv.runtime.measurements import check_pull_endpoints, coaching_summary, knee_contact_measurement, phase_summary
from wlcv.runtime.pull_phases import PULL_KEYS, suggest_from_rows


def initial_review(rows, manual):
    suggested, reasons = suggest_from_rows(rows)
    if "drive" not in manual:
        suggested.pop("drive", None)
        reasons.pop("drive", None)
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
    if knee is not None and drive is not None and not {knee, drive}.issubset(checks):
        # 수정한 시점에 코사인 기록이 없으면 두 장 모두 저장된 추적 중심으로 측정한다.
        checks = check_pull_endpoints(rows, (knee, drive))
        cosine_weights = None
        for point in checks.values():
            if point["reason"] == "코사인 가중치 없음":
                point["reason"] = "해당 끝점 쌍의 코사인 검출 기록 없음"
    rise = knee_contact_measurement(knee, drive, checks, cosine_weights)
    coaching = coaching_summary(rows, lift, knee, events, direction)
    angles = rows.set_index("frame_index").trunk_angle_deg.reindex([lift, knee])
    change = float(angles.iloc[1] - angles.iloc[0]) if angles.notna().all() else None
    return {"knee_to_contact_rise": rise, "manual_phase_interval": phase_summary(rows, lift, knee),
            "trunk_change_deg": change, "coaching": coaching}
