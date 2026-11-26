"""같은 원본 프레임의 원판 중심과 가까운 무릎 높이로 통과 시점을 찾는다."""
from contextlib import closing
from datetime import datetime
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.trunk_repeat import FIRST_PAIRS
from wlcv.config import ROOT, RUNS, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.evaluation.performer_continuity import select
from wlcv.evaluation.phase_speed import phases
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.single_pass import single_pass_frames

PARENT = ROOT / "outputs/knee_pass_height"


def observe(video, info, contact, plate_model, pose_model, device):
    """0번부터 이력을 쌓는다. 수동 무릎 통과 시점은 입력으로 받지 않는다."""
    previous = None
    rows = []
    timings = dict(plate_inference=0., plate_tracking=0., pose=0.)
    stream = single_pass_frames(VIDEOS / info.filename, plate_model, pose_model,
                                device, True, timings, plate_imgsz=640)
    with closing(stream):
        for time_ms, _, plate, pose in islice(stream, contact + 1):
            boxes = pose.boxes.xyxy
            who = select(boxes, previous)
            previous = boxes[who].copy() if who is not None else None
            knee_x = knee_y = knee_conf = np.nan
            status = "수행자 보류"
            if who is not None:
                points = joint_table(pose, who + 1, .50)
                knee = points[(points["방향"] == info.near_side) & (points["관절"] == "무릎")].iloc[0]
                status = knee["표시 상태"]
                knee_conf = knee["관절 신뢰도"]
                if status == "표시":
                    knee_x, knee_y = knee["x (px)"], knee["y (px)"]
            rows.append(dict(video=video, frame=plate.frame_index, time_ms=time_ms,
                             side=info.near_side, people=len(boxes), person=None if who is None else who+1,
                             plate_status=plate.source, plate_score=plate.confidence,
                             plate_x=plate.cx, plate_y=plate.cy, x1=plate.x1, x2=plate.x2,
                             y1=plate.y1, y2=plate.y2, knee_status=status,
                             knee_score=knee_conf, knee_x=knee_x, knee_y=knee_y))
    track = pd.DataFrame(rows)
    if (track.frame.tolist() != list(range(contact+1)) or not np.isfinite(track.time_ms).all()
            or not track.time_ms.diff().iloc[1:].gt(0).all()):
        raise ValueError(f"{video}의 원본 프레임 수 또는 타임스탬프를 확인하세요.")
    track["gap_px"] = track.knee_y - track.plate_y
    return track


def first_crossing(gap):
    """아래(-) → 같은 높이 또는 위(0 이상). 결측을 가로질러 프레임을 정하지 않는다."""
    observed = gap.dropna()
    above = observed[observed >= 0]
    if above.empty:
        if observed.empty:
            reason = "두 좌표 동시 관측 없음"
        elif gap.isna().any():
            reason = "교차 미관측·결측 있음"
        else:
            reason = "교차 미관측"
        return None, reason
    frame = int(above.index[0])
    if frame == observed.index[0]:
        return None, "첫 관측부터 같은 높이 이상"
    if pd.isna(gap.loc[frame - 1]):
        return None, "교차 사이 결측"
    return frame, "관측"


def compare(track, marks):
    """선택을 마친 후 정답과 비교한다. 시각 차이는 각 프레임의 원본 타임스탬프를 쓴다."""
    at = track.set_index("frame", verify_integrity=True)
    lift, manual, contact = marks["lift"], marks["knee"], marks["contact"]
    gap = at.loc[lift:contact, "gap_px"]
    auto, status = first_crossing(gap)
    error = None if auto is None else auto - manual
    return dict(video=track.video.iat[0], lift=lift, manual=manual, contact=contact,
                automatic=auto, status=status, error_frames=error,
                error_ms=np.nan if auto is None else at.at[auto, "time_ms"] - at.at[manual, "time_ms"],
                window_frames=len(gap), valid_frames=int(gap.count()), gap_at_manual=gap.at[manual],
                manual_event_rise_px=at.at[lift, "plate_y"] - at.at[manual, "plate_y"],
                auto_event_rise_px=np.nan if auto is None else at.at[lift, "plate_y"] - at.at[auto, "plate_y"])


def run(device="mps"):
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id", verify_integrity=True)
    plate_weights = RUNS / "plate_baseline/weights/best.pt"
    pose_weights = (ROOT / "models/yolo26n-pose.pt").resolve(strict=True)
    plate_model = validation_plate_model(plate_weights, FIRST_PAIRS)
    pose_model = YOLO(str(pose_weights))
    tracks, scores = [], []
    for video in tqdm(FIRST_PAIRS, desc="검증 영상", unit="편"):
        marks = phases(video)
        info = meta.loc[video]
        if info.near_side not in ("left", "right"):
            raise ValueError(f"{video} 가까운 쪽이 지정되지 않았습니다.")
        track = observe(video, info, marks["contact"], plate_model, pose_model, device)
        tracks.append(track)
        scores.append(compare(track, marks))
    settings = dict(device=device, imgsz=640, plate_direct_conf=.10, plate_candidate_conf=.001,
                    link_radius_ratio=.50, person_conf=.25, person_match_iou=.50, knee_conf=.50,
                    rule="first adjacent negative to nonnegative gap from manual lift through contact",
                    plate_sha256=sha256(plate_weights.read_bytes()).hexdigest(),
                    pose_sha256=sha256(pose_weights.read_bytes()).hexdigest())
    PARENT.mkdir(parents=True, exist_ok=True)
    folder = PARENT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with TemporaryDirectory(dir=PARENT) as tmp:
        staging = Path(tmp) / "complete"
        staging.mkdir()
        pd.concat(tracks, ignore_index=True).to_csv(staging / "frames.csv", index=False)
        pd.DataFrame(scores).to_csv(staging / "summary.csv", index=False)
        (staging / "settings.json").write_text(json.dumps(settings, indent=2))
        staging.rename(folder)
    return folder


def plot_curves(frames, summary):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, case in zip(axes.flat, summary.itertuples(), strict=True):
        data = frames.loc[frames.video.eq(case.video) & frames.frame.between(case.lift, case.contact)]
        ax.axhline(0, color="0.6", lw=.8)
        ax.axvline(case.manual, color="#b14d12", ls="--", lw=2)
        if pd.notna(case.automatic):
            ax.axvline(case.automatic, color="#0050a3", ls=":", lw=2)
        ax.plot(data.frame, data.gap_px, color="#176e63", marker=".", ms=5)
        missing = data.loc[data.gap_px.isna(), "frame"]
        ax.plot(missing, [0.04]*len(missing), "|", color="0.4", transform=ax.get_xaxis_transform())
        auto = "held" if pd.isna(case.automatic) else str(int(case.automatic))
        ax.set_title(f"{case.video[:3]} · manual {case.manual} / rule {auto}")
        ax.set_xlabel("Source frame")
        ax.set_ylabel("Knee y - plate center y (px)")
        ax.grid(axis="y", alpha=.15)
    handles = [Line2D([], [], color="#176e63", marker=".", label="Above knee when positive"),
               Line2D([], [], color="#b14d12", ls="--", label="Manual knee pass"),
               Line2D([], [], color="#0050a3", ls=":", label="First crossing"),
               Line2D([], [], color="0.4", ls="none", marker="|", label="Unobserved")]
    fig.legend(handles=handles, loc="outside lower center", ncols=4)
    return fig


def plot_case(frames, summary, video):
    """수동 프레임과 규칙 프레임을 원본에 표시한다. 보류 시 수동 시점 전후를 본다."""
    case = summary.set_index("video").loc[video]
    manual = int(case.manual)
    if pd.notna(case.automatic):
        items = [(manual, "Manual"), (int(case.automatic), "Rule")]
    else:
        items = [(i, "Manual" if i == manual else "Neighbor") for i in
                 range(max(int(case.lift), manual-1), min(int(case.contact), manual+1)+1)]
    table = frames[frames.video.eq(video)].set_index("frame")
    fig, axes = plt.subplots(1, len(items), squeeze=False, figsize=(3.8*len(items), 5.3), layout="constrained")
    for ax, (i, label) in zip(axes.flat, items):
        row = table.loc[i]
        ax.imshow(read_rgb(VIDEOS / f"{video}.mp4", i))
        if pd.notna(row.plate_y):
            ax.add_patch(Rectangle((row.x1,row.y1),row.x2-row.x1,row.y2-row.y1,
                                   fill=False,edgecolor="gold",lw=1.5))
            ax.plot(row.plate_x, row.plate_y, "+", color="gold", ms=13, mew=2)
        if pd.notna(row.knee_y):
            ax.axhline(row.knee_y, color="cyan", ls=":", lw=1)
            ax.plot(row.knee_x, row.knee_y, "x", color="cyan", ms=9, mew=2)
        ax.set_title(f"{label} · frame {i}")
        ax.axis("off")
    fig.suptitle(f"{video} · gold: plate center / cyan: near knee")
    return fig
