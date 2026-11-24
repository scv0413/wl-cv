"""검증 영상에서 수행자 후보와 YOLO26n-pose의 네 관절을 원본 좌표에 표시한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# COCO 17점에서 수행자 본인의 왼쪽과 오른쪽 관절 번호
JOINTS = {"어깨": (5, 6), "골반": (11, 12), "무릎": (13, 14), "발목": (15, 16)}
SIDES = ("left", "right")


def show_candidates(rgb, result, title):
    """사람이 여러 명이면 그림의 번호로 수행자를 고른다. 미검출 프레임도 보여 준다."""
    fig, ax = plt.subplots(figsize=(5, 6), layout="constrained")
    ax.imshow(rgb)
    for number, (x1, y1, x2, y2) in enumerate(result.boxes.xyxy, start=1):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor="gold", linewidth=1.5))
        ax.text(x1, y1, str(number), va="top", color="black",
                bbox=dict(facecolor="gold", edgecolor="none", pad=1))
    ax.set_title(f"{title} | {len(result.boxes)} person candidates")
    ax.axis("off")
    return fig


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


def plot_pose(rgb, result, person_number, joints, title, near_side, zoom=True, *, manual=None):
    """원본 전체와 같은 프레임의 관절 예측을 나란히 그린다."""
    fig, (original, overlay) = plt.subplots(1, 2, figsize=(9, 5.5), layout="constrained")
    for ax in (original, overlay):
        ax.imshow(rgb)
        ax.axis("off")
    original.set_title("Original frame")
    overlay.set_title(f"Pose #{person_number}   |   Camera side: {near_side}")
    fig.suptitle(title)

    for side, color, marker in [("left", "#00e5ff", "o"), ("right", "#ffab40", "^")]:
        points = joints[joints["방향"] == side]
        if points.empty:
            continue
        x, y = points["x (px)"], points["y (px)"]
        visible = points["표시 상태"] == "표시"
        # NaN에서 선이 끊겨 무릎이 빠졌을 때 골반과 발목을 바로 잇지 않는다.
        overlay.plot(x.where(visible), y.where(visible), color=color, linewidth=1.6,
                     marker=marker, markersize=7, markeredgecolor="black", label=side.title())
        weak = points["표시 상태"] == "저신뢰"
        overlay.scatter(x[weak], y[weak], facecolors="none", edgecolors=color,
                        marker=marker, s=70, linewidths=1.7)
    # 수동 원본 좌표는 예측과 같은 축에 표시한다.
    for state, marker, label in [("located", "+", "Manual"), ("estimated", "D", "Manual estimate")]:
        rows = [row for row in (manual or {}).values()
                if row["side"] == near_side and row["status"] == state]
        if rows:
            overlay.plot([int(row["x_px"]) for row in rows], [int(row["y_px"]) for row in rows],
                         linestyle="none", marker=marker, color="#cd2ffe", markerfacecolor="none",
                         markeredgewidth=1.7, markersize=9, label=label)
    overlay.legend(loc="upper right", fontsize=8)

    if zoom:
        x1, y1, x2, y2 = result.boxes.xyxy[person_number - 1]
        pad = 0.65 * max(x2 - x1, y2 - y1)
        height, width = rgb.shape[:2]
        overlay.set_xlim(max(0, x1 - pad), min(width, x2 + pad))
        overlay.set_ylim(min(height, y2 + pad), max(0, y1 - pad))
    return fig


def compare_manual_points(joints, records, near_side):
    """수동으로 판독한 가까운 쪽 관절을 모델과 같은 원본 픽셀 단위로 비교한다."""
    rows = []
    for _, prediction in joints.loc[joints["방향"] == near_side].iterrows():
        joint = prediction["관절"]
        record = records.get(joint)
        dx = dy = distance = estimate_distance = np.nan
        status = "미검수"
        if record and record["side"] == near_side:
            if record["status"] in ("located", "estimated"):
                dx = int(record["x_px"]) - prediction["x (px)"]
                dy = int(record["y_px"]) - prediction["y (px)"]
                if record["status"] == "estimated":
                    status = "소켓 위치 추정" if record["reason"] == "hip_socket" else "가림 추정"
                    estimate_distance = np.hypot(dx, dy)
                else:
                    status = "좌표 확인"
                    distance = np.hypot(dx, dy)
            elif record["reason"] == "occluded":
                status = "판독 불가 · 가림"
            else:
                status = "판독 불가 · 불확실"
        rows.append((joint, status, dx, dy, distance, estimate_distance))
    return pd.DataFrame(rows, columns=["관절", "수동 판독", "Δx (px)", "Δy (px)", "거리 (px)", "추정점 참고 거리 (px)"])


def compare_pose_reads(first, second):
    """같은 관절을 두 번 짚은 거리와 판독 상태의 변화를 확인한다."""
    labels = {"located": "직접 표시", "estimated": "추정", "unreadable": "판독 불가"}
    rows = []
    for joint in JOINTS:
        a, b = first.get(joint), second.get(joint)
        if not a or not b:
            raise ValueError(f"{joint}의 1차와 2차 판독을 모두 확인하세요.")
        if any(a[key] != b[key] for key in ("video_id", "frame_index", "person_id", "side", "split")):
            raise ValueError(f"{joint}의 영상, 프레임, 선수 또는 방향이 다릅니다.")

        states = [a["status"], b["status"]]
        distance = estimate_distance = np.nan
        if "unreadable" not in states:
            dx = int(b["x_px"]) - int(a["x_px"])
            dy = int(b["y_px"]) - int(a["y_px"])
            if "estimated" in states:
                estimate_distance = np.hypot(dx, dy)
            else:
                distance = np.hypot(dx, dy)
        rows.append((joint, labels[states[0]], labels[states[1]], distance, estimate_distance))

    return pd.DataFrame(rows, columns=["관절", "1차 상태", "2차 상태", "직접 표시 간 거리 (px)",
                                       "추정 포함 참고 거리 (px)"])
