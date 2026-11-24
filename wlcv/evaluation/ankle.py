"""무릎 통과 프레임에서 가까운 복사뼈의 수동점과 자세 모델의 발목점을 비교한다."""

import matplotlib.pyplot as plt
import numpy as np

from wlcv.evaluation.pose import compare_manual_points, joint_table, show_candidates


def review_ankle(rgb, result, record, *, title, person_number=None, min_confidence=0.50):
    """한 프레임의 판독을 표 한 행과 확대 그림으로 반환한다."""
    side = record["side"]
    status = record["status"]
    status_label = {"located": "직접 표시", "estimated": "가림 추정", "unreadable": "판독 불가"}
    report = {"수동 판독": status_label[status], "판독 사유": record["reason"],
              "수동 x": np.nan, "수동 y": np.nan, "후보 번호": person_number,
              "모델 상태": "", "모델 x": np.nan, "모델 y": np.nan, "발목 신뢰도": np.nan,
              "Δy (px)": np.nan, "추정점 Δy (px)": np.nan}

    if status != "unreadable":
        report["수동 x"], report["수동 y"] = int(record["x_px"]), int(record["y_px"])
    count = len(result.boxes)
    if count == 1 and person_number is None:
        person_number = 1
    if person_number is None or not 1 <= person_number <= count:
        report["모델 상태"] = "사람 미검출" if count == 0 else "수행자 선택 필요"
        return report, show_candidates(rgb, result, title)

    joints = joint_table(result, person_number, min_confidence)
    ankle = joints.loc[(joints["관절"] == "발목") & (joints["방향"] == side)]
    point = ankle.iloc[0]
    report.update({"후보 번호": person_number, "모델 상태": point["표시 상태"],
                   "모델 x": point["x (px)"], "모델 y": point["y (px)"],
                   "발목 신뢰도": point["관절 신뢰도"]})
    comparison = compare_manual_points(ankle, {"발목": record}, side).iloc[0]
    column = "추정점 Δy (px)" if status == "estimated" else "Δy (px)"
    report[column] = comparison["Δy (px)"]

    fig, ax = plt.subplots(figsize=(4.4, 4), layout="constrained")
    ax.imshow(rgb)
    xs, ys = [], []
    markers = [("모델", "YOLO", "o", "#00c9df"),
               ("수동", "Estimate" if status == "estimated" else "Manual",
                "D" if status == "estimated" else "+", "#ff41b3")]
    for prefix, label, marker, color in markers:
        x, y = report[prefix + " x"], report[prefix + " y"]
        if np.isfinite(x) and np.isfinite(y):
            ax.plot(x, y, marker=marker, linestyle="none", markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=2, markersize=10, label=label)
            xs.append(x)
            ys.append(y)
    if xs:
        height, width = rgb.shape[:2]
        pad = max(24, round(height * 0.055))
        ax.set_xlim(max(0, min(xs) - pad), min(width, max(xs) + pad))
        ax.set_ylim(min(height, max(ys) + pad), max(0, min(ys) - pad))
        ax.legend(loc="upper right")
    ax.set_title(f"{title} | {side} ankle")
    ax.axis("off")
    return report, fig


def compare_ankle_landmarks(rgb, previous, fold):
    """같은 프레임의 고정 모델점에서 두 수동 기준점까지의 직선거리를 비교한다."""
    status = {"located": "직접 표시", "estimated": "추정", "unreadable": "판독 불가"}[fold["status"]]
    model = np.array([previous["모델 x"], previous["모델 y"]], dtype=float)
    bone = np.array([previous["수동 x"], previous["수동 y"]], dtype=float)
    crease = np.array([np.nan, np.nan])
    if fold["status"] != "unreadable":
        crease = np.array([int(fold["x_px"]), int(fold["y_px"])])

    bone_distance = np.linalg.norm(model - bone)
    fold_distance = np.linalg.norm(model - crease)
    difference = np.nan
    nearer = None
    if previous["수동 판독"] == "직접 표시" and fold["status"] == "located":
        difference = bone_distance - fold_distance
        if np.isfinite(difference):
            nearer = "같은 거리"
            if difference > 0:
                nearer = "접힘점"
            elif difference < 0:
                nearer = "복사뼈"

    video = fold["video_id"]
    frame = int(fold["frame_index"])
    report = {"영상": video, "프레임": frame, "가까운 쪽": fold["side"],
              "복사뼈 판독": previous["수동 판독"], "접힘점 판독": status,
              "접힘점 사유": fold["reason"], "모델 상태": previous["모델 상태"],
              "발목 신뢰도": previous["발목 신뢰도"], "모델–복사뼈 (px)": bone_distance,
              "모델–접힘점 (px)": fold_distance, "거리 차이 (px)": difference, "더 가까운 기준점": nearer}

    fig, ax = plt.subplots(figsize=(4.6, 4), layout="constrained")
    ax.imshow(rgb)
    points = []
    for point, marker, color, label in [(model, "o", "#00c9df", "YOLO"),
                                         (bone, "+", "#ff41b3", "Malleolus"),
                                         (crease, "^", "#ffcf20", "Front bend")]:
        if np.isfinite(point).all():
            ax.plot(*point, marker=marker, color=color, markerfacecolor="none", markeredgewidth=2,
                    markersize=12, linestyle="none", label=label)
            points.append(point)
    if points:
        xs, ys = np.array(points).T
        height, width = rgb.shape[:2]
        pad = max(24, round(height * 0.055))
        ax.set_xlim(max(0, min(xs) - pad), min(width, max(xs) + pad))
        ax.set_ylim(min(height, max(ys) + pad), max(0, min(ys) - pad))
        ax.legend(loc="upper right")
    ax.set_title(f"{video} / {frame} | {fold['side']} ankle")
    ax.axis("off")
    return report, fig
