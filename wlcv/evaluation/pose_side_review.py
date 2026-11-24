"""a05의 오른쪽 수동 무릎·발목에 같은 선수의 좌우 예측을 각각 대조한다."""

from math import hypot, isclose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.config import ANNOTATIONS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.pose_comparison import JOINT_CONF, PERSON_CONF

VIDEO_ID = "a05_sn_f_01"
FRAMES = (30, 41, 60)
JOINTS = ("무릎", "발목")
EXISTING = ANNOTATIONS / "knee_ankle_candidates" / "validation"
NEW_READS = ANNOTATIONS / "a05_side_review" / "validation"


def read_right_points(folder=NEW_READS):
    """41번은 예비 판독 원본, 나머지는 이번 판독에서 읽는다. 미검수도 한 행으로 남긴다."""
    rows = []
    for frame in FRAMES:
        source = EXISTING if frame == 41 else folder
        records = load_points(pose_point_path(source, VIDEO_ID, frame))
        for joint in JOINTS:
            record = records.get(joint)
            x = y = np.nan
            status = "미검수"
            if record is not None:
                keys = ("video_id", "person_id", "split", "frame_index", "side")
                if tuple(record[key] for key in keys) != (VIDEO_ID, "a05", "validation", str(frame), "right"):
                    raise ValueError(f"{frame}번 {joint}의 영상·선수·분할·방향을 확인하세요.")
                if record["status"] == "located":
                    x, y = float(record["x_px"]), float(record["y_px"])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        raise ValueError(f"{frame}번 {joint}의 직접점을 확인하세요.")
                    status = "직접 확인"
                elif record["status"] == "unreadable" and not (record["x_px"] or record["y_px"]):
                    status = "판독 불가"
                else:
                    raise ValueError(f"{frame}번 {joint}는 직접 확인 또는 판독 불가로 저장하세요.")
            if frame == 41 and status != "직접 확인":
                raise ValueError("41번 원본에서 기존 오른쪽 무릎·발목 직접점을 확인하세요.")
            rows.append({"프레임": frame, "관절": joint, "수동 판독": status, "수동 x": x, "수동 y": y})
    return pd.DataFrame(rows)


def compare_sides(model, video, manual):
    """수동점 하나에 대해 같은 관절의 왼쪽·오른쪽 예측까지 원본 픽셀 거리를 잰다."""
    if manual["수동 판독"].eq("미검수").any():
        raise ValueError("30번과 60번의 오른쪽 무릎·발목을 모두 저장하세요.")
    rows = []
    for frame in tqdm(FRAMES, desc="좌우 관절 비교", unit="장"):
        rgb = read_rgb(video, frame)
        result = predict_rgb(model, rgb, conf=PERSON_CONF, imgsz=640, device="cpu").cpu().numpy()
        count = len(result.boxes)
        points = None
        if count == 1:
            points = joint_table(result, 1, JOINT_CONF).set_index(["관절", "방향"])
        for row in manual.loc[manual["프레임"] == frame].to_dict("records"):
            if row["수동 판독"] == "직접 확인" and not (
                0 <= row["수동 x"] < rgb.shape[1] and 0 <= row["수동 y"] < rgb.shape[0]
            ):
                raise ValueError(f"{frame}번 {row['관절']} 수동점이 원본 밖에 있습니다.")
            for side, label in [("left", "왼쪽"), ("right", "오른쪽")]:
                x = y = confidence = distance = np.nan
                status = "수행자 미탐" if count == 0 else "수행자 선택 불확실"
                if points is not None:
                    point = points.loc[(row["관절"], side)]
                    confidence, status = point["관절 신뢰도"], point["표시 상태"]
                    if status == "표시":
                        x, y = point["x (px)"], point["y (px)"]
                        if row["수동 판독"] == "직접 확인":
                            distance = hypot(x - row["수동 x"], y - row["수동 y"])
                row.update({f"{label} x": x, f"{label} y": y, f"{label} 신뢰도": confidence,
                            f"{label} 상태": status, f"{label} 거리(px)": distance})
            left, right = row["왼쪽 거리(px)"], row["오른쪽 거리(px)"]
            if row["수동 판독"] == "판독 불가":
                nearest = "수동 판독 불가"
            elif not (np.isfinite(left) and np.isfinite(right)):
                nearest = "좌우 비교 불가"
            elif isclose(left, right, abs_tol=1e-8):
                nearest = "같은 거리"
            else:
                nearest = "왼쪽" if left < right else "오른쪽"
            row["더 가까운 예측"] = nearest
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_figure(video, report):
    """열은 프레임, 행은 관절이다. 세 점을 같은 원본 프레임 위에 그린다."""
    styles = [("수동", "Manual right", "o", "#ffdf00"),
              ("왼쪽", "YOLO left", "^", "#00cfff"),
              ("오른쪽", "YOLO right", "X", "#ff6699")]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5), layout="constrained")
    for column, frame in enumerate(FRAMES):
        rgb = read_rgb(video, frame)
        subset = report[report["프레임"] == frame].set_index("관절")
        for index, joint in enumerate(JOINTS):
            ax, row = axes[index, column], subset.loc[joint]
            ax.imshow(rgb)
            coords = []
            for prefix, _, marker, color in styles:
                x, y = row[f"{prefix} x"], row[f"{prefix} y"]
                if pd.notna(x) and pd.notna(y):
                    coords.append((x, y))
                    ax.scatter(x, y, marker=marker, s=95, color=color, edgecolors="black", linewidths=1)
            if coords:
                xs, ys = zip(*coords)
                pad = 45
                ax.set_xlim(max(0, min(xs)-pad), min(rgb.shape[1], max(xs)+pad))
                ax.set_ylim(min(rgb.shape[0], max(ys)+pad), max(0, min(ys)-pad))
            label = {"왼쪽": "Left closer", "오른쪽": "Right closer", "같은 거리": "Tie",
                     "수동 판독 불가": "Manual unreadable", "좌우 비교 불가": "Pair unavailable"}
            joint_name = "Knee" if joint == "무릎" else "Ankle"
            ax.set_title(f"#{frame} {joint_name} | {label[row['더 가까운 예측']]}", fontsize=10)
            ax.axis("off")
    handles = [Line2D([], [], marker=m, color=c, markeredgecolor="black", linestyle="none", label=name)
               for _, name, m, c in styles]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return fig
