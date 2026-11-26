"""같은 원본 좌표에서 동작 중 미탐, 허브 세로 오차, 이륙부터의 상승량을 비교한다."""
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wlcv.annotation.hub_points import read_hubs
from wlcv.annotation.plate_blur import REVIEW_FILE
from wlcv.annotation.storage import events_for, load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT
from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.plate_size_mps import B02, C04
from wlcv.evaluation.plate_track import BOX, KEY
from wlcv.evaluation.schedule_review import review_inputs

LIFT_HUB = ANNOTATIONS / "hub_lift_review" / f"{B02}.csv"


def verified_frames(runs):
    """각 MP4의 끝까지 읽고, 같은 조건의 반복 결과와 두 크기의 자세 결과를 대조한다."""
    first = {}
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    expected = {(video, size, block) for video in (C04, B02) for size in (640, 512) for block in range(1, 5)}
    if len(runs) != len(expected) or set(runs[["video_id", "size", "block"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("두 영상·두 크기의 네 블록이 모두 끝난 측정표가 필요합니다.")
    for row in runs.itertuples(index=False):
        folder = ROOT / row.run
        summary = json.loads((folder / "summary.json").read_text())
        if ((summary["video_id"], summary["device"], summary["read_mode"], summary["plate_imgsz"], summary["pose_imgsz"])
                != (row.video_id, "mps", "single-pass", row.size, 640)):
            raise ValueError(f"실행 조건 불일치: {folder}")
        frame_table = pd.read_csv(folder / "frames.csv")
        info = metadata.loc[row.video_id]
        if frame_table.frame_index.tolist() != list(range(info.frame_count)) or not frame_table.video_id.eq(row.video_id).all():
            raise ValueError(f"원본 프레임 목록 불일치: {folder}")
        capture = cv2.VideoCapture(str(folder / "overlay.mp4"))
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (info.height, info.width):
                    raise ValueError(f"원본과 출력 크기가 다릅니다: {folder}")
                count += 1
        finally:
            capture.release()
        if count != len(frame_table):
            raise ValueError(f"오버레이 재생 장수 불일치: {folder}")
        key = (row.video_id, row.size)
        if key in first:
            pd.testing.assert_frame_equal(first[key], frame_table, check_exact=True)
        else:
            first[key] = frame_table
    for video in (C04, B02):
        columns = [name for name in first[video, 640] if not name.startswith("plate_")]
        pd.testing.assert_frame_equal(first[video, 640][columns], first[video, 512][columns], check_exact=True)
    return pd.concat([frames.assign(size=size) for (video, size), frames in first.items()], ignore_index=True)


def motion_rows(frames):
    videos, boxes, _ = review_inputs(DATASET / "data.yaml")
    if not {B02, C04}.issubset(videos):
        raise ValueError("베이스라인 검증셋과 원본 박스 라벨 경로를 확인하세요.")
    records = []
    for video, group in frames.groupby("video_id"):
        events = load_events(ANNOTATIONS, video, 1)
        end = events[events_for(video)[-1]]
        active = group[group.frame_index.between(events["ready"], end)].copy()
        truth = boxes[boxes.video_id.eq(video)].rename(columns={col: "gt_"+col for col in BOX})
        active = active.merge(truth, how="left", on=KEY, validate="many_to_one")
        scored = active[["gt_"+col for col in BOX]].notna().all(axis=1)
        active["iou"] = np.nan
        for i, row in active[scored].iterrows():
            predicted = row[["plate_"+c for c in BOX]].to_numpy(float)
            label = row[["gt_"+c for c in BOX]].to_numpy(float)
            active.at[i, "iou"] = iou(predicted, label)
        active["match"] = active.iou.ge(.50).astype("boolean").where(scored)
        records.append(active)
    return pd.concat(records, ignore_index=True)


def hub_and_rise(frames):
    # 같은 날의 연속 블라인드 판독을 한 번씩만 가중한다. 재판독은 섞지 않는다.
    blind = pd.read_csv(REVIEW_FILE)
    hubs = blind[blind.video_id.eq(B02) & blind.hub_source.isin(["clicked", "previous_blind"])]
    b02 = frames[frames.video_id.eq(B02)]
    points = b02.merge(hubs[KEY+["manual_y"]], on=KEY, validate="many_to_one")
    points["dy_px"] = points.plate_cy_px - points.manual_y
    points["abs_dy_px"] = points.dy_px.abs()
    events = load_events(ANNOTATIONS, B02, 1)
    phase = pd.read_csv(ANNOTATIONS / "trunk_inclination/frames" / f"{B02}.csv").iloc[0]
    lift, knee = int(phase["이륙"]), int(phase["무릎 통과"])
    if phase["판독"] != "확인" or events["transition"] != knee or not lift < knee:
        raise ValueError("수동 이륙·무릎 통과 시점을 확인하세요.")
    lift_points = read_hubs(LIFT_HUB, frames=(lift,), video=B02)
    visible = lift_points[lift_points.visibility.eq("visible")]
    manual_rise = np.nan
    if not visible.empty and knee in set(hubs.frame_index):
        manual_rise = float(visible.manual_y.iloc[0]) - hubs.set_index("frame_index").at[knee, "manual_y"]
    rises = []
    for size, group in b02.groupby("size"):
        center = group.set_index("frame_index", verify_integrity=True).plate_cy_px
        rise = center.at[lift] - center.at[knee]
        rises.append(dict(size=size, lift=lift, knee=knee, lift_y=center.at[lift], knee_y=center.at[knee],
                          rise_px=rise, manual_rise_px=manual_rise, error_px=rise-manual_rise))
    return points, pd.DataFrame(rises)


def evaluate(folder):
    runs = pd.read_csv(folder / "runs.csv")
    frames = verified_frames(runs)
    motion = motion_rows(frames)
    points, rises = hub_and_rise(frames)
    motion.to_csv(folder / "motion_frames.csv", index=False)
    points.to_csv(folder / "b02_hubs.csv", index=False)
    rises.to_csv(folder / "b02_rise.csv", index=False)
    return runs, motion, points, rises


def plot_comparison(runs, points):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    colors = {640: "#176b87", 512: "#b65517"}
    for size, color in colors.items():
        row = runs[runs["size"] == size]
        offset = -.07 if size == 640 else .07
        for i, video in enumerate((C04, B02)):
            times = row[row.video_id.eq(video)].command_s
            axes[0].scatter(np.repeat(i+offset, len(times)), times, color=color, alpha=.55)
            axes[0].plot([i+offset-.045, i+offset+.045], [times.median()]*2, color=color, lw=3)
        p = points[points["size"] == size].sort_values("frame_index")
        axes[1].plot(p.frame_index, p.dy_px, "o-" if size == 640 else "s--", color=color, label=str(size))
    axes[0].set(xticks=[0, 1], xticklabels=["c04", "b02"], ylabel="Command to exit (s)", title="Four runs per input size")
    axes[1].set(xlabel="Source frame", ylabel="Box center minus hub (px, down +)", title="b02 | manual hub")
    axes[1].axhline(0, color="gray", lw=.8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(title="Plate input")
    return fig
