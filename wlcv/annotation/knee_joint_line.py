"""c04 두 프레임에서 선수의 왼쪽 무릎 관절선을 다시 판독한다."""
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wlcv.annotation.pose_points import load_points, pose_point_editor, pose_point_path
from wlcv.config import ANNOTATIONS, ROOT, VIDEOS
from wlcv.data.frames import read_rgb

VIDEO = "c04_sn_f_01"
FRAMES = (27, 31)
CROP = (370, 1290, 830, 1600)  # 양쪽 프레임에 동일한 원본 픽셀 영역
REVIEW = ANNOTATIONS / "c04_knee_joint_line"
SOURCE = ROOT / "outputs/knee_pass_height/20261126-000518-592842/frames.csv"
STATUS = {"located": "직접 확인", "estimated": "추정", "unreadable": "판독 불가"}


def editor(folder=REVIEW):
    panels = [
        pose_point_editor(VIDEOS / f"{VIDEO}.mp4", frame, "c04", "left",
                          split="validation", annotations=folder, joints=("무릎",),
                          trunk_review=True, crop=CROP)
        for frame in FRAMES
    ]
    tabs = widgets.Tab(children=panels)
    for index, frame in enumerate(FRAMES):
        tabs.set_title(index, f"{frame}번")
    return tabs


def compare(folder=REVIEW, source=None):
    """두 판독을 모두 저장한 후에만 58번 예측을 읽는다."""
    manual = {}
    for frame in FRAMES:
        point = load_points(pose_point_path(folder, VIDEO, frame)).get("무릎")
        if point is not None:
            if (point["video_id"], point["person_id"], point["split"], point["side"],
                point["frame_index"]) != (VIDEO, "c04", "validation", "left", str(frame)):
                raise ValueError(f"{frame}번 수동점의 대상이 다릅니다.")
            manual[frame] = point
    print(f"왼쪽 무릎 판독 {len(manual)}/2장 저장")
    if len(manual) != len(FRAMES):
        print("남은 프레임:", [f for f in FRAMES if f not in manual])
        return

    if source is None:
        source = SOURCE
        if not source.is_file():
            runs = sorted(SOURCE.parent.parent.glob("*/frames.csv"))
            if len(runs) != 1:
                raise FileNotFoundError("58번 노트북을 실행한 뒤 compare(source=프레임_CSV_경로)로 선택하세요.")
            source = runs[0]
    track = pd.read_csv(source)
    track = track[track.video.eq(VIDEO) & track.frame.isin(FRAMES)].set_index("frame", verify_integrity=True)
    if set(track.index) != set(FRAMES) or not track.side.eq("left").all():
        raise ValueError("58번의 두 프레임과 가까운 방향을 확인하세요.")

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    x0, y0, x1, y1 = CROP
    for ax, frame in zip(axes, FRAMES):
        point = manual[frame]
        pose = track.loc[frame]
        ax.imshow(read_rgb(VIDEOS / f"{VIDEO}.mp4", frame)[y0:y1, x0:x1])
        ax.set_title(f"c04 · frame {frame}")
        ax.axis("off")
        knee_dx = knee_dy = plate_dx = plate_dy = np.nan
        if pd.notna(pose.plate_y):
            ax.plot(pose.plate_x-x0, pose.plate_y-y0, "+", color="gold",
                    ms=14, mew=2, label="Plate box center")
        if pd.notna(pose.knee_y):
            ax.plot(pose.knee_x-x0, pose.knee_y-y0, "x", color="cyan", ms=12, mew=2, label="YOLO knee")
        if point["status"] != "unreadable":
            x, y = float(point["x_px"]), float(point["y_px"])
            direct = point["status"] == "located"
            ax.scatter(x-x0, y-y0, marker="o" if direct else "D", facecolors="none",
                       edgecolors="magenta", s=110, linewidths=2,
                       label="Manual knee" if direct else "Estimated knee")
            knee_dx, knee_dy = pose.knee_x - x, pose.knee_y - y
            plate_dx, plate_dy = pose.plate_x - x, pose.plate_y - y
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left")
        knee_distance = np.hypot(knee_dx, knee_dy)
        rows.append({
            "프레임": frame,
            "수동 판독": STATUS[point["status"]],
            "무릎 신뢰도": pose.knee_score,
            "YOLO 무릎 세로 차이(px)": knee_dy,
            "원판 중심 세로 차이(px)": plate_dy,
            "수동점까지 YOLO 무릎 거리(px)": knee_distance,
            "수동점까지 원판 중심 거리(px)": np.hypot(plate_dx, plate_dy),
            "직접 확인한 무릎 위치 오차(px)": knee_distance if point["status"] == "located" else np.nan,
        })
    return pd.DataFrame(rows), fig
