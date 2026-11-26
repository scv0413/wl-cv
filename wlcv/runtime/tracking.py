"""같은 YOLO 후보를 현재 프레임만으로 선택하거나 과거 두 점으로 연결한다."""

import numpy as np

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


def prepare_candidates(candidates):
    """중심을 계산하고 같은 신뢰도에서는 원래 후보 순서를 보존한다."""
    c = candidates[KEY + ["confidence"] + BOX].copy()
    c["cx"] = (c.x1 + c.x2) / 2
    c["cy"] = (c.y1 + c.y2) / 2
    c = c[c.confidence >= .001].sort_values("confidence", ascending=False, kind="stable")
    return c


class PlateHistory:
    """한 영상의 최근 관측 두 점과 마지막 고신뢰 프레임을 기억한다."""

    def __init__(self):
        self.history = []
        self.last_high = None

    def choose(self, now, video, frame, *, distance_ratio=0.5, checks=None,
               gate_high_history=False, high_checks=None, history_checks=None):
        """prepare_candidates로 정렬된 현재 프레임의 후보를 연결한다."""
        strong = now[now.confidence >= .10]
        high = None if strong.empty else strong.iloc[0]
        chosen = high
        source = "high" if high is not None else "missing"
        if self.last_high is not None and frame - self.last_high > 2:
            self.history = []
        # 선택 직전의 이력을 기록한다. 직접 검출에서는 이 반경을 적용하지 않는다.
        if history_checks is not None and self.history:
            ex, ey = expected_center(self.history, frame)
            last = self.history[-1][1]
            history_checks.append(dict(video_id=video, frame_index=frame, expected_x=ex,
                                       expected_y=ey, radius_px=(last.x2-last.x1)*distance_ratio))
        add_to_history = True
        if high is not None:
            # 고신뢰 출력과 2프레임 시계는 두 조건에서 동일하다.
            self.last_high = frame
            if gate_high_history and self.history:
                expected_x, expected_y = expected_center(self.history, frame)
                last = self.history[-1][1]
                radius = (last.x2 - last.x1) * distance_ratio
                distance = np.hypot(high.cx - expected_x, high.cy - expected_y)
                add_to_history = distance <= radius
                if high_checks is not None:
                    high_checks.append(dict(video_id=video, frame_index=frame,
                                            last_history_frame=self.history[-1][0],
                                            expected_x=expected_x, expected_y=expected_y,
                                            radius_px=radius, distance_px=distance,
                                            stored=bool(add_to_history)))
        elif self.history:
            chosen = link_low_candidate(now, self.history, frame,
                                        distance_ratio=distance_ratio, checks=checks)
            if chosen is not None:
                source = "linked"
        if chosen is not None and add_to_history:
            self.history = (self.history + [(frame, chosen)])[-2:]
        return high, chosen, source


def choose_plate(history, candidates, video, frame, width, height, rule):
    """현재 후보를 거른 뒤에만 이력을 갱신한다. 원본 픽셀 좌표를 사용한다."""
    now = prepare_candidates(candidates)
    if rule not in ("off", "strict", "continuity"):
        raise ValueError(rule)
    if rule != "off" and not now.empty:
        bw, bh = now.x2 - now.x1, now.y2 - now.y1
        edge = (now.x1 <= 1) | (now.y1 <= 1) | (width-now.x2 <= 1) | (height-now.y2 <= 1)
        blocked = edge & (bh > 0) & (bw >= 3 * bh)
        if (rule == "continuity" and blocked.any() and history.history
                and history.last_high is not None and frame - history.last_high <= 2):
            x, y = expected_center(history.history, frame)
            _, previous = history.history[-1]
            radius = (previous.x2 - previous.x1) * 0.5
            blocked &= np.hypot(now.cx - x, now.cy - y) > radius
        now = now.loc[~blocked]
    _, box, source = history.choose(now, video, frame, distance_ratio=0.5)
    return box, source


def iou(box, truth):
    if not np.isfinite(box).all():
        return np.nan
    overlap = np.maximum(0, np.minimum(box[2:], truth[2:]) - np.maximum(box[:2], truth[:2])).prod()
    return overlap / (np.prod(box[2:] - box[:2]) + np.prod(truth[2:] - truth[:2]) - overlap)
