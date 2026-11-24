"""b04 두 시점에 수동 견봉·소켓과 YOLO 어깨·골반을 각각 연결한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.data.frames import read_rgb
from wlcv.evaluation.hip_fold import FRAMES, VIDEO, left_pose_joints, manual_xy, socket_midpoint
from wlcv.evaluation.trunk import point_label, vertical_angle

MANUAL = "수동 견봉·소켓 중점"
MODEL = "YOLO 왼쪽 어깨·골반"


def acromion_reads(folder):
    return {frame: load_points(pose_point_path(folder, VIDEO, frame)).get("어깨")
            for frame in FRAMES}


def compare_trunk(video_path, model, shoulder_folder, socket_folder):
    shoulders = acromion_reads(shoulder_folder)
    if video_path.stem != VIDEO or not all(shoulders.values()):
        raise ValueError("b04 9번과 21번의 견봉을 각각 판독한 뒤 비교하세요.")

    rows = []
    for frame in FRAMES:
        manual = shoulders[frame]
        if (manual["side"], manual["split"], manual["video_id"], int(manual["frame_index"])) != (
            "left", "train", VIDEO, frame
        ):
            raise ValueError(f"{frame}번 견봉의 영상·프레임·방향·분할을 확인하세요.")
        shoulder = manual_xy(manual, {"located", "estimated"})
        socket = socket_midpoint(socket_folder, frame)
        joints = left_pose_joints(model, read_rgb(video_path, frame), frame)
        model_shoulder = joints.loc["어깨", ["x (px)", "y (px)"]].to_numpy(dtype=float)
        model_hip = joints.loc["골반", ["x (px)", "y (px)"]].to_numpy(dtype=float)
        socket_status = "소켓 1·2차 평균 (추정)" if np.isfinite(socket).all() else "소켓 판독 부족"
        conditions = [
            (MANUAL, shoulder, socket, point_label(manual), socket_status),
            (MODEL, model_shoulder, model_hip, joints.at["어깨", "표시 상태"], joints.at["골반", "표시 상태"]),
        ]
        for method, top, bottom, shoulder_status, hip_status in conditions:
            rows.append({"프레임": frame, "기준": method,
                         "어깨 x": top[0], "어깨 y": top[1], "골반 x": bottom[0], "골반 y": bottom[1],
                         "어깨 판독": shoulder_status, "골반 판독": hip_status, "각도 (°)": vertical_angle(top, bottom)})
    return pd.DataFrame(rows)


def plot_lines(video_path, points):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), layout="constrained")
    for ax, frame in zip(axes, FRAMES):
        rgb = read_rgb(video_path, frame)
        height, width = rgb.shape[:2]
        ax.imshow(rgb)
        rows = points.loc[points["프레임"] == frame].set_index("기준")
        for method, color, marker, name in [(MANUAL, "#ffd34d", "o", "Manual"), (MODEL, "#00dbef", "^", "YOLO")]:
            row = rows.loc[method]
            ax.plot([row["어깨 x"], row["골반 x"]], [row["어깨 y"], row["골반 y"]],
                    color=color, marker=marker, mfc="none", lw=2, ms=8, label=name)
        ax.set_xlim(width * 0.31, width * 0.72)
        ax.set_ylim(height * 0.86, height * 0.51)
        ax.set_title(f"b04  |  frame {frame}")
        ax.legend(loc="upper right")
        ax.axis("off")
    return fig
