"""같은 YOLO 후보를 현재 프레임만으로 선택하거나 과거 두 점으로 연결한다."""
from itertools import groupby

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

KEY = ["video_id", "frame_index"]
BOX = ["x1", "y1", "x2", "y2"]


def expected_center(history, frame):
    """직전 두 관측의 속도를 사용한다. 한 점만 있으면 마지막 위치를 쓴다."""
    last_frame, last = history[-1]
    expected_x, expected_y = last.cx, last.cy
    if len(history) == 2:
        before_frame, before = history[0]
        step = (frame - last_frame) / (last_frame - before_frame)
        expected_x += (last.cx - before.cx) * step
        expected_y += (last.cy - before.cy) * step
    return expected_x, expected_y


def link_low_candidate(candidates, history, frame, *, distance_ratio=0.5, checks=None):
    """예상 중심에서 허용 반지름 안에 있는 가장 가까운 후보를 선택한다."""
    _, last = history[-1]
    expected_x, expected_y = expected_center(history, frame)
    distance = np.hypot(candidates.cx - expected_x, candidates.cy - expected_y)
    last_width = last.x2 - last.x1
    radius = last_width * distance_ratio
    nearby = candidates[distance <= radius]
    if checks is not None:
        checks.append(dict(video_id=last.video_id, frame_index=frame, distance_ratio=distance_ratio,
                           expected_x=expected_x, expected_y=expected_y, last_width=last_width,
                           radius_px=radius, nearest_px=distance.min(), candidates=len(candidates),
                           inside=len(nearby)))
    if nearby.empty:
        return None
    # candidates가 신뢰도 내림차순이어서 거리가 같을 때 신뢰도가 높은 후보가 먼저 온다.
    return nearby.loc[distance.loc[nearby.index].idxmin()]


def select_centers(frames, candidates, *, distance_ratio=0.5, checks=None,
                   gate_high_history=False, high_checks=None):
    """정답 좌표·IoU 없이 후보를 선택한다. 두 방법의 고신뢰 선택은 같다."""
    if not np.isfinite(distance_ratio) or distance_ratio <= 0:
        raise ValueError("연결 거리 배율은 양의 유한한 수여야 합니다.")
    c = candidates[KEY + ["confidence"] + BOX].copy()
    c["cx"] = (c.x1 + c.x2) / 2
    c["cy"] = (c.y1 + c.y2) / 2
    c = c[c.confidence >= .001].sort_values("confidence", ascending=False, kind="stable")
    grouped = c.groupby(KEY)
    rows = []
    for video, segment in frames[KEY].groupby("video_id"):
        history = []
        last_high = None
        indices = segment.frame_index.sort_values()
        if not indices.diff().dropna().eq(1).all():
            raise ValueError(f"{video}의 연속 프레임을 확인하세요.")
        for frame in tqdm(indices, desc=video, unit="장"):
            key = (video, frame)
            now = grouped.get_group(key) if key in grouped.groups else c.iloc[:0]
            strong = now[now.confidence >= .10]
            high = None if strong.empty else strong.iloc[0]
            chosen = high
            source = "high" if high is not None else "missing"
            if last_high is not None and frame - last_high > 2:
                history = []
            add_to_history = True
            if high is not None:
                # 고신뢰 출력과 2프레임 시계는 두 조건에서 동일하다.
                last_high = frame
                if gate_high_history and history:
                    expected_x, expected_y = expected_center(history, frame)
                    last = history[-1][1]
                    radius = (last.x2 - last.x1) * distance_ratio
                    distance = np.hypot(high.cx - expected_x, high.cy - expected_y)
                    add_to_history = distance <= radius
                    if high_checks is not None:
                        high_checks.append(dict(video_id=video, frame_index=frame,
                                                last_history_frame=history[-1][0],
                                                expected_x=expected_x, expected_y=expected_y,
                                                radius_px=radius, distance_px=distance,
                                                stored=bool(add_to_history)))
            elif history:
                chosen = link_low_candidate(now, history, frame,
                                            distance_ratio=distance_ratio, checks=checks)
                if chosen is not None:
                    source = "linked"
            if chosen is not None and add_to_history:
                history = (history + [(frame, chosen)])[-2:]
            for method, choice, origin in [("YOLO", high, "high" if high is not None else "missing"),
                                            ("Past link", chosen, source)]:
                row = dict(video_id=video, frame_index=frame, method=method, source=origin)
                for col in ["confidence", "cx", "cy"] + BOX:
                    row[col] = np.nan if choice is None else choice[col]
                rows.append(row)
    return pd.DataFrame(rows)


def score_centers(selected, centers, boxes):
    """후보 선택이 끝난 뒤에만 수동 중심과 원판 정답 박스를 결합한다."""
    if set(map(tuple, centers[KEY].values)) != set(map(tuple, boxes[KEY].values)):
        raise ValueError("중심 판독표와 평가 프레임이 다릅니다.")
    ground = boxes[KEY + BOX].rename(columns={col:"gt_" + col for col in BOX})
    d = selected.merge(centers, on=KEY, validate="many_to_one").merge(ground, on=KEY, validate="many_to_one")
    width = (np.minimum(d.x2, d.gt_x2) - np.maximum(d.x1, d.gt_x1)).clip(lower=0)
    height = (np.minimum(d.y2, d.gt_y2) - np.maximum(d.y1, d.gt_y1)).clip(lower=0)
    intersection = width * height
    box_area = (d.x2 - d.x1) * (d.y2 - d.y1)
    gt_width = d.gt_x2 - d.gt_x1
    d["iou"] = intersection / (box_area + gt_width * (d.gt_y2 - d.gt_y1) - intersection)
    d["match"] = d.iou.ge(.50)
    d["error_px"] = np.hypot(d.cx - d.manual_x, d.cy - d.manual_y)
    d["error_width_pct"] = 100 * d.error_px / gt_width
    return d


def summarize_centers(detail):
    rows = []
    for (video, method), g in detail.groupby(["video_id", "method"], sort=False):
        g = g.sort_values("frame_index")
        readable = g[g.manual_y.notna()]
        scored = readable[readable.match]
        gaps = [len(list(part)) for missing, part in groupby(~g.match) if missing]
        # 최고점은 화면에서 y가 가장 작은 점이다. 비교 가능한 구간의 관측 최고점을 센다.
        with_point = readable.dropna(subset=["cy"])
        first_peak = readable.loc[readable.manual_y.idxmin()] if len(readable) else None
        peak = with_point.loc[with_point.cy.idxmin()] if len(with_point) else None
        rows.append(dict(video_id=video, method=method, frames=len(g), readable=len(readable),
                         outputs=g.cx.count(), hits=g.match.sum(), readable_hits=len(scored),
                         wrong_boxes=(g.cx.notna() & ~g.match).sum(), longest_gap=max(gaps, default=0),
                         median_error_px=scored.error_px.median(), median_error_pct=scored.error_width_pct.median(),
                         max_error_px=scored.error_px.max(),
                         manual_peak_frame=first_peak.frame_index if first_peak is not None else np.nan,
                         peak_frame=peak.frame_index if peak is not None else np.nan,
                         peak_y_difference_px=(peak.cy-first_peak.manual_y)
                         if peak is not None and first_peak is not None else np.nan))
    return pd.DataFrame(rows)
