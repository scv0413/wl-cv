"""수동 시점 없이 한 번의 측면 스내치에서 이륙·무릎 통과·접촉 후보를 제안한다."""

import numpy as np
import pandas as pd

from wlcv.runtime.tracking import iou


def find_lift(t, *, allow_missing_ankle=False):
    """발목 부근에서 잠시 정지한 다음 두 장 연속 올라가는 첫 장을 찾는다."""
    y = t.plate_y.to_numpy()
    time = t.time_ms.to_numpy()
    ground = None
    first_low_y = None
    last_seen = None
    for i in range(len(t)):
        if pd.isna(y[i]):
            continue
        size = t.diameter.iat[i]
        if not np.isfinite(size) or size <= 0:
            continue
        knee, ankle = t.knee.iat[i], t.ankle.iat[i]
        at_floor = np.isfinite(ankle) and abs(y[i] - ankle) <= 0.65 * size
        if allow_missing_ankle and not np.isfinite(ankle):
            # 발목이 가려진 경우만 정지·상승 궤적으로 출발을 판단한다.
            at_floor = True
        if np.isfinite(knee):
            at_floor = at_floor and y[i] > knee
        if ground is None:
            if (np.isfinite(knee) and y[i] <= knee) or (
                first_low_y is not None and first_low_y - y[i] > 0.18 * size
            ):
                return None, "정지한 출발을 확인하기 전에 원판 상승 또는 무릎 높이 도달"
            if at_floor and first_low_y is None:
                first_low_y = y[i]
            # 정지 폭은 원판 지름의 2.5%(최소 2px). 최소 0.1초를 연속 관측한다.
            j = max(0, np.searchsorted(time, time[i] - 100, side="right") - 1)
            window = t.iloc[j:i+1]
            if not at_floor or time[i]-time[j] < 99.9 or len(window) < 3:
                continue
            if not window.subject.notna().all() or window.subject.nunique() != 1:
                continue
            values = y[j:i+1]
            band = max(2.0, 0.025 * size)
            if np.isfinite(values).all() and np.ptp(values) <= band:
                ground = float(np.median(values))
                ground_size = size
                ground_subject = t.subject.iat[i]
                last_seen = i
            continue
        if time[i] - time[last_seen] > 250 or t.subject.iat[i] != ground_subject:
            return None, "출발 기준을 세운 뒤 원판·수행자 연결이 끊김"
        last_seen = i
        band = max(2.0, 0.025 * ground_size)
        if i+1 < len(t) and y[i] < ground-band and y[i+1] < ground-band:
            if t.subject.iat[i+1] == ground_subject and y[i+1] < y[i]:
                return i, "바닥 정지 후 원판 중심이 연속 상승"
        if at_floor and abs(y[i]-ground) <= band:
            ground = (ground+y[i])/2
    return None, "바닥 정지와 연속 상승을 함께 관측하지 못함"


def first_ascent(t, lift):
    """결측·수행자 변경·하강 이후의 다른 동작을 끌어다 쓰지 않는다."""
    initial = t.iloc[lift]
    high = initial.plate_y
    last_seen = lift
    descent = 0
    for i in range(lift+1, len(t)):
        r = t.iloc[i]
        if pd.notna(r.subject) and r.subject != initial.subject:
            return i
        if r.time_ms - t.time_ms.iat[last_seen] > 250:
            return i
        if not np.isfinite(r.plate_y):
            continue
        last_seen = i
        high = min(high, r.plate_y)
        descent = descent + 1 if r.plate_y > high + 0.30 * initial.diameter else 0
        if descent >= 2:
            return i-1
    return len(t)


def first_crossing(gap, start, stop):
    # 앞 장은 기준 아래, 현재·다음 장은 기준 이상. 결측값은 비교가 성립하지 않는다.
    for i in range(start, stop-1):
        if gap[i-1] < 0 <= gap[i] and gap[i+1] >= 0:
            return i
    return None


def suggest(t, *, allow_missing_ankle=False):
    if t.empty or not np.array_equal(t.frame.to_numpy(), np.arange(len(t))):
        raise ValueError("0번부터 끝까지 순서대로 저장한 궤적이 필요합니다.")
    if not np.isfinite(t.time_ms).all() or not t.time_ms.diff().iloc[1:].gt(0).all():
        raise ValueError("프레임 표시 시각을 확인하세요.")
    lift, reason = find_lift(t, allow_missing_ankle=allow_missing_ankle)
    out = dict(lift=lift, knee=None, contact=None, lift_reason=reason,
               knee_reason="이륙 보류", contact_reason="무릎 통과 보류")
    if lift is None:
        return out
    stop = first_ascent(t, lift)
    knee = first_crossing((t.knee-t.plate_y).to_numpy(), lift+1, stop)
    out.update(knee=knee, knee_reason="무릎 높이 교차" if knee is not None else "첫 상승 중 연속된 무릎 교차 없음")
    if knee is None:
        return out
    # 허벅지 위쪽 높이는 바와 몸의 실제 접촉을 직접 관측한 값이 아니다.
    thigh = t.hip + 0.25 * (t.knee-t.hip)
    contact = first_crossing((thigh-t.plate_y).to_numpy(), knee+1, stop)
    out.update(contact=contact, contact_reason="허벅지 위쪽 높이 교차" if contact is not None
               else "첫 상승 중 연속된 상부 허벅지 교차 없음")
    return out


PULL_KEYS = ("lift_off", "knee_pass", "drive")


def suggest_from_rows(rows):
    """검수 실험과 같은 원판·가까운 관절·수행자 연속 구간을 구성한다."""
    track = pd.DataFrame({
        "frame": rows.frame_index, "time_ms": rows.source_time_ms, "plate_y": rows.plate_cy_px,
        "diameter": rows.plate_y2 - rows.plate_y1, "knee": rows.knee_y_px,
        "hip": rows.hip_y_px, "ankle": rows.ankle_y_px,
    }, dtype=float)
    names = ["performer_x1", "performer_y1", "performer_x2", "performer_y2"]
    subjects = []
    previous = None
    segment = 0
    for box in rows[names].to_numpy(dtype=float):
        if not np.isfinite(box).all():
            previous = None
            subjects.append(np.nan)
            continue
        if previous is None or iou(box, previous) < 0.30:
            segment += 1
        previous = box
        subjects.append(segment)
    track["subject"] = subjects
    if not np.isfinite(track.time_ms).all() or not track.time_ms.diff().iloc[1:].gt(0).all():
        return dict.fromkeys(PULL_KEYS), dict.fromkeys(PULL_KEYS, "원본 프레임 시각 확인 필요")
    result = suggest(track, allow_missing_ankle=True)
    names = ("lift", "knee", "contact")
    return ({k: result[name] for k, name in zip(PULL_KEYS, names)},
            {k: result[name + "_reason"] for k, name in zip(PULL_KEYS, names)})
