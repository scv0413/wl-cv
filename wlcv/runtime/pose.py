"""가까운 쪽 관절을 표시하기 위한 COCO 좌표 판독."""
import numpy as np
import pandas as pd


# COCO 17점에서 수행자 본인의 왼쪽과 오른쪽 관절 번호
JOINTS = {"어깨": (5, 6), "골반": (11, 12), "무릎": (13, 14), "발목": (15, 16)}
SIDES = ("left", "right")


def joint_table(result, person_number, min_confidence):
    """한 사람의 양쪽 어깨·골반·무릎·발목 좌표와 신뢰도를 표로 만든다."""
    if not 1 <= person_number <= len(result.boxes):
        raise ValueError("그림에 표시된 사람 번호를 선택하세요.")
    if not 0 <= min_confidence <= 1:
        raise ValueError("관절 신뢰도 기준은 0부터 1 사이로 정하세요.")

    points = result.keypoints.data[person_number - 1]
    height, width = result.orig_shape
    rows = []
    for joint, indices in JOINTS.items():
        for side, index in zip(SIDES, indices):
            x, y, confidence = points[index]
            if not (0 <= x < width and 0 <= y < height) or (x == 0 and y == 0):
                x, y = np.nan, np.nan
                status = "좌표 없음"
            elif confidence >= min_confidence:
                status = "표시"
            else:
                status = "저신뢰"
            rows.append((side, joint, x, y, confidence, status))
    return pd.DataFrame(rows, columns=["방향", "관절", "x (px)", "y (px)", "관절 신뢰도", "표시 상태"])
