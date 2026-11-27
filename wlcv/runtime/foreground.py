"""고정 측면 카메라에서 화면에 크게 보이는 전경 선수와 그 주변 원판을 고른다."""
from itertools import combinations

import numpy as np

from wlcv.runtime.tracking import iou


def select_performer(boxes, previous, *, area_ratio=1.25):
    """후보 번호는 0부터 센다. 크기가 비슷할 때는 직전 선수와 겹치는지 확인한다."""
    if len(boxes) == 0:
        return None, "missing"
    if len(boxes) == 1:
        return 0, "single"

    sizes = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    ranked = np.argsort(-sizes, kind="stable")
    first, second = ranked[:2]
    # 같은 몸에 넓은 박스와 좁은 박스가 함께 생길 수 있다. 이 경우 면적 순위는 불안정하다.
    if previous is not None and iou(boxes[first], boxes[second]) >= .50:
        overlaps = np.array([iou(box, previous) for box in boxes])
        best, runner_up = np.argsort(-overlaps, kind="stable")[:2]
        in_front = iou(boxes[best], boxes[first]) >= .50 or sizes[best] * area_ratio >= sizes[first]
        if in_front and overlaps[best] >= .50 and overlaps[best] - overlaps[runner_up] >= .10:
            return int(best), "previous_box"
        return None, "ambiguous"
    if sizes[first] >= area_ratio * sizes[second]:
        return int(first), "foreground_size"
    if previous is not None:
        comparable = [i for i in ranked if area_ratio * sizes[i] >= sizes[first]]
        linked = [i for i in comparable if iou(boxes[i], previous) >= .50]
        if len(linked) == 1:
            return int(linked[0]), "previous_box"
    return None, "ambiguous"


def performer_plates(candidates, people, selected, *, margin=.15):
    """다인 장면에서 선수 옆·머리 위 원판은 허용하고 다른 사람 쪽 원판은 거른다."""
    if len(people) < 2 or candidates.empty:
        return candidates
    if selected is None:
        # 같은 사람의 전신·부분 박스가 겹칠 때는 공통 위치의 원판을 추적할 수 있다.
        if any(iou(a, b) < .50 for a, b in combinations(people, 2)):
            return candidates.iloc[:0]
        targets = people
    else:
        targets = [people[selected]]

    cx = (candidates.x1 + candidates.x2) / 2
    cy = (candidates.y1 + candidates.y2) / 2
    rx = (candidates.x2 - candidates.x1) / 2
    ry = (candidates.y2 - candidates.y1) / 2
    keep = np.ones(len(candidates), dtype=bool)
    for x1, y1, x2, y2 in targets:
        # 원판이 사람 박스 밖으로 나오는 캐치와 바닥 출발 자세의 여유 폭.
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin
        keep &= cx.between(x1 - pad_x - rx, x2 + pad_x + rx)
        keep &= cy.between(y1 - pad_y - ry, y2 + pad_y + ry)
    return candidates.loc[keep]
