"""캐치에서 가까운 원판 중심과 견봉의 상대 위치를 원본 좌표로 비교한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.phase_speed import prior_hubs
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table

# 원본 캐치에서 얼굴이 향한 화면 방향을 확인했다. 영상이 바뀌면 다시 판독한다.
FACING = {"a03_sn_f_01": -1, "a05_sn_f_01": 1, "b03_sn_f_01": -1,
          "c04_sn_f_01": -1, "c08_sn_f_01": -1}
READS = ANNOTATIONS / "catch_shoulders"


def shoulder_read(video, frame, side):
    for folder in (READS, ANNOTATIONS):
        row = load_points(pose_point_path(folder, video, frame)).get("어깨")
        if row:
            if (row["video_id"], int(row["frame_index"]), row["side"]) != (video, frame, side):
                raise ValueError("견봉 판독의 영상·프레임·방향을 확인하세요.")
            return row
    return {}


def predicted_shoulder(result, side):
    if len(result.boxes) != 1:
        return np.nan, np.nan, f"수행자 후보 {len(result.boxes)}명"
    joints = joint_table(result, 1, 0.5)
    point = joints[(joints["방향"] == side) & (joints["관절"] == "어깨")].iloc[0]
    if point["표시 상태"] != "표시":
        return np.nan, np.nan, point["표시 상태"]
    return float(point["x (px)"]), float(point["y (px)"]), "표시"


def catch_comparison(plates):
    hubs = prior_hubs()
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    selected = plates.set_index(["video_id", "frame_index"], verify_integrity=True)
    pose = YOLO(str((ROOT / "models/yolo26n-pose.pt").resolve(strict=True)))
    rows = []
    for video in sorted(plates.video_id.unique()):
        frame = load_events(ANNOTATIONS, video, pass_number=1).get("catch")
        if frame is None:
            rows.append(dict(video_id=video, shoulder_status="캐치 라벨 없음"))
            continue
        key = video, frame
        side = meta.at[video, "near_side"]
        manual = shoulder_read(video, frame, side)
        status = manual.get("status", "미판독")
        sx, sy = np.nan, np.nan
        if status in ("located", "estimated"):
            sx, sy = float(manual["x_px"]), float(manual["y_px"])
        hx, hy = np.nan, np.nan
        if key in hubs.index and hubs.at[key, "visibility"] == "visible":
            hx, hy = hubs.at[key, "manual_x"], hubs.at[key, "manual_y"]
        rgb = read_rgb(VIDEOS / f"{video}.mp4", frame)
        result = predict_rgb(pose, rgb, conf=.25, imgsz=640, device="cpu").cpu().numpy()
        px, py, pose_status = predicted_shoulder(result, side)
        box = selected.loc[key]
        facing = FACING.get(video, np.nan)
        rows.append(dict(video_id=video, frame_index=frame, side=side, facing=facing,
                         shoulder_status=status, pose_status=pose_status, plate_source=box.source,
                         manual_hx=hx, manual_hy=hy, manual_sx=sx, manual_sy=sy,
                         model_hx=box.cx, model_hy=box.cy, model_sx=px, model_sy=py,
                         manual_forward=facing*(hx-sx), manual_up=sy-hy,
                         model_forward=facing*(box.cx-px), model_up=py-box.cy))
    return pd.DataFrame(rows)


def plot_catches(table):
    rows = table.dropna(subset=["frame_index"])
    fig, axes = plt.subplots(1, len(rows), figsize=(3*len(rows), 4), squeeze=False)
    for ax, row in zip(axes[0], rows.itertuples(), strict=True):
        ax.imshow(read_rgb(VIDEOS / f"{row.video_id}.mp4", int(row.frame_index)))
        for source, color in [("manual", "gold"), ("model", "deepskyblue")]:
            x = [getattr(row, source+"_hx"), getattr(row, source+"_sx")]
            y = [getattr(row, source+"_hy"), getattr(row, source+"_sy")]
            ax.plot(x, y, "o-", fillstyle="none", color=color, label=source)
        ax.set_title(f"{row.video_id[:3]} · {int(row.frame_index)}")
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig
