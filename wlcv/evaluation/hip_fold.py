"""b04 두 시점에서 왼쪽 어깨를 고정하고 골반 쪽 끝점만 바꿔 비교한다."""

from math import dist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wlcv.annotation.pose_points import HIP_FOLD, load_points, pose_point_path
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.trunk import vertical_angle

FRAMES = (9, 21)
VIDEO = "b04_sn_f_01"
SOCKET = "소켓 추정 중점"
MODEL = "YOLO 골반"
METHODS = (SOCKET, HIP_FOLD, MODEL)


def fold_reads(folder):
    return {frame: load_points(pose_point_path(folder, VIDEO, frame)).get(HIP_FOLD)
            for frame in FRAMES}


def manual_xy(point, allowed):
    if not point or point["status"] not in allowed:
        return np.array([np.nan, np.nan])
    return np.array([float(point["x_px"]), float(point["y_px"])])


def compare(video_path, model, fold_folder, socket_folder):
    folds = fold_reads(fold_folder)
    if not all(folds.values()):
        raise ValueError("두 프레임의 접힘점 또는 판독 불가 상태를 먼저 저장하세요.")
    if video_path.stem != VIDEO:
        raise ValueError("b04_sn_f_01 원본을 선택하세요.")

    rows = []
    for frame in FRAMES:
        rgb = read_rgb(video_path, frame)
        result = predict_rgb(model, rgb).cpu().numpy()
        # 이 두 장은 원본에서 수행자 한 명씩 확인했다.
        if len(result.boxes) != 1:
            raise ValueError(f"{frame}번의 수행자 후보 수가 달라졌습니다. 원본을 확인하세요.")
        joints = joint_table(result, 1, min_confidence=0.50)
        left = joints.loc[joints["방향"] == "left"].set_index("관절")
        coords = left[["x (px)", "y (px)"]].copy()
        coords.loc[left["표시 상태"] != "표시"] = np.nan
        shoulder = coords.loc["어깨"].to_numpy()
        model_hip = coords.loc["골반"].to_numpy()

        socket_reads = []
        for number in (1, 2):
            path = pose_point_path(socket_folder, VIDEO, frame, pass_number=number)
            point = load_points(path).get("골반")
            socket_reads.append(manual_xy(point, {"estimated", "located"}))
        socket = np.mean(socket_reads, axis=0)
        fold = manual_xy(folds[frame], {"located"})
        for method, endpoint in zip(METHODS, (socket, fold, model_hip)):
            rows.append({"프레임": frame, "기준": method,
                         "어깨 x": shoulder[0], "어깨 y": shoulder[1],
                         "끝점 x": endpoint[0], "끝점 y": endpoint[1],
                         "각도 (°)": vertical_angle(shoulder, endpoint),
                         "YOLO와 거리 (px)": dist(endpoint, model_hip) if method != MODEL else np.nan,
                         "어깨 신뢰도": left.at["어깨", "관절 신뢰도"],
                         "골반 신뢰도": left.at["골반", "관절 신뢰도"]})
    return pd.DataFrame(rows)


def plot_comparison(video_path, points):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    styles = [("socket mean", "#ffbf34", "o"), ("visible fold", "#12e6db", "x"),
              ("YOLO hip", "#eb79ff", "^")]
    for ax, frame in zip(axes, FRAMES):
        rgb = read_rgb(video_path, frame)
        height, width = rgb.shape[:2]
        ax.imshow(rgb)
        rows = points.loc[points["프레임"] == frame].set_index("기준")
        shoulder = rows.loc[MODEL, ["어깨 x", "어깨 y"]].to_numpy(dtype=float)
        for method, (label, color, marker) in zip(METHODS, styles):
            x, y = rows.loc[method, ["끝점 x", "끝점 y"]]
            ax.plot([shoulder[0], x], [shoulder[1], y], color=color, alpha=0.7)
            ax.plot(x, y, marker=marker, ms=9, mew=2, mfc="none", color=color,
                    linestyle="none", label=label)
        ax.plot(*shoulder, "w+", ms=12, mew=2, label="fixed shoulder")
        ax.set_xlim(width * .31, width * .72)
        ax.set_ylim(height * .86, height * .51)
        ax.set_title(f"b04  |  frame {frame}")
        ax.axis("off")
    axes[1].legend(loc="upper right", fontsize=8)
    return fig
