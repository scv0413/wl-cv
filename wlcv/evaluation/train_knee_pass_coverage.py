"""학습 선수의 무릎 통과 전후에 원판과 가까운 무릎이 함께 보이는지 확인한다."""
from datetime import datetime
from hashlib import sha256
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT, RUNS, SPLIT
from wlcv.evaluation.knee_pass_height import observe

PARENT = ROOT / "outputs/train_knee_pass_coverage"
VIDEOS = ("a01_sn_f_01", "a06_sn_h_01", "a08_sn_f_01", "b00_sn_f_01",
          "b01_sn_f_01", "b04_sn_f_01", "c02_sn_f_01")


def extract(device="mps"):
    """처음부터 끝까지 추론한다. 두 수동 판독은 나중에 분석할 구간을 고를 때만 쓴다."""
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    split = pd.read_csv(SPLIT).set_index("video_id").loc[list(VIDEOS)]
    if not split.role.eq("train").all() or split.person_id.nunique() != len(VIDEOS):
        raise ValueError("7명 모두 서로 다른 학습 선수여야 합니다.")
    plate_path = RUNS / "plate_baseline/weights/best.pt"
    pose_path = ROOT / "models/yolo26n-pose.pt"
    plate = YOLO(str(plate_path.resolve(strict=True)))
    pose = YOLO(str(pose_path.resolve(strict=True)))
    if plate.names != {0: "plate"} or pose.task != "pose":
        raise ValueError("원판 검출과 자세 모델을 확인하세요.")
    trained = set(pd.read_csv(plate_path.parent.parent / "train_people.csv").person_id)
    tracks, readings = [], []
    for video in tqdm(VIDEOS, desc="원본 영상", unit="편"):
        info = meta.loc[video]
        if info.near_side not in ("left", "right"):
            raise ValueError(f"{video}의 가까운 쪽을 확인하세요.")
        track = observe(video, info, int(info.frame_count) - 1, plate, pose, device)
        tracks.append(track)
        first, second = [load_events(ANNOTATIONS, video, n)["transition"] for n in (1, 2)]
        if not (0 <= min(first, second) <= max(first, second) < len(track)):
            raise ValueError(f"{video}의 판독 프레임이 원본 범위 밖입니다.")
        readings.append(dict(video=video, first=first, second=second, near_side=info.near_side,
                             plate_train_person=split.at[video, "person_id"] in trained))

    PARENT.mkdir(parents=True, exist_ok=True)
    dest = PARENT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with TemporaryDirectory(dir=PARENT) as temp:
        staging = Path(temp) / "complete"
        staging.mkdir()
        pd.concat(tracks, ignore_index=True).to_csv(staging / "frames.csv", index=False)
        pd.DataFrame(readings).to_csv(staging / "readings.csv", index=False)
        settings = dict(device=device, imgsz=640, plate_direct=.10, plate_candidate=.001,
                        link_radius_ratio=.5, person_conf=.25, person_iou=.5, knee_conf=.5,
                        plate_sha256=sha256(plate_path.read_bytes()).hexdigest(),
                        pose_sha256=sha256(pose_path.read_bytes()).hexdigest())
        (staging / "settings.json").write_text(json.dumps(settings, indent=2))
        staging.rename(dest)
    return dest



def missing_reason(row):
    reasons = []
    if pd.isna(row.plate_x) or pd.isna(row.plate_y):
        reasons.append("원판 미관측")
    if pd.isna(row.knee_x) or pd.isna(row.knee_y):
        reasons.append(row.knee_status)
    return " + ".join(reasons) if reasons else "동시 관측"


def intervals(data, column):
    """원본 프레임 순서대로 같은 값이 이어진 구간을 양 끝 포함으로 센다."""
    groups = data[column].ne(data[column].shift()).cumsum()
    return data.groupby(groups, sort=False).agg(
        start=("frame", "first"), end=("frame", "last"), length=("frame", "size"),
        state=(column, "first")).reset_index(drop=True)


def measure(folder, padding=5):
    """두 판독 범위를 포함하는 연속 길이와 분석 창 안의 모든 결측을 센다."""
    if padding < 0:
        raise ValueError("전후 장수는 0 이상이어야 합니다.")
    tracks = pd.read_csv(folder / "frames.csv")
    readings = pd.read_csv(folder / "readings.csv")
    summaries, segments, windows = [], [], []
    for mark in readings.itertuples():
        track = tracks[tracks.video.eq(mark.video)].copy()
        if track.frame.tolist() != list(range(len(track))):
            raise ValueError(f"{mark.video}의 원본 프레임 순서를 확인하세요.")
        track["reason"] = [missing_reason(row) for row in track.itertuples()]
        track["paired"] = track.reason.eq("동시 관측")
        lo, hi = sorted((mark.first, mark.second))
        window = track[track.frame.between(lo - padding, hi + padding)].copy()
        windows.append(window)
        runs = intervals(window, "paired")
        detail = intervals(window, "reason")
        detail.insert(0, "video", mark.video)
        segments.append(detail)

        # 창 밖까지 이어지는 경우도 영상 전체에서 끊길 때까지 센다.
        full = intervals(track, "paired")
        span = full[full.state & full.start.le(lo) & full.end.ge(hi)]
        start = end = None
        before = after = None
        if not span.empty:
            block = span.iloc[0]
            start, end = int(block.start), int(block.end)
            before, after = lo - start, end - hi
        available = runs[runs.state]
        longest = 0 if available.empty else int(available.length.max())
        summaries.append(dict(video=mark.video, first=mark.first, second=mark.second,
                              plate_train_person=mark.plate_train_person,
                              window_start=window.frame.iat[0], window_end=window.frame.iat[-1],
                              window_ms=window.time_ms.iat[-1] - window.time_ms.iat[0],
                              total=len(window), paired=int(window.paired.sum()), longest=longest,
                              linked_paired=int((window.paired & window.plate_status.eq("linked")).sum()),
                              core_observed=not span.empty, full_start=start, full_end=end,
                              full_length=0 if start is None else end-start+1,
                              before=before, after=after))
    return (pd.DataFrame(summaries).convert_dtypes(), pd.concat(segments, ignore_index=True),
            pd.concat(windows, ignore_index=True))


def plot_window(summary, windows):
    """색은 실제 원본 프레임에서 어느 좌표가 빠졌는지 나타낸다."""
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    colors = {"Both": "#148476", "Plate missing": "#dd914b", "Knee missing": "#6878ad",
              "Both missing": "#9d5965", "Performer selection withheld": "#b66040"}
    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    used = set()
    labels = []
    for i, mark in enumerate(summary.itertuples()):
        lo, hi = sorted((mark.first, mark.second))
        labels.append(mark.video[:3])
        window = windows[windows.video.eq(mark.video)]
        for row in window.itertuples():
            has_plate = pd.notna(row.plate_x) and pd.notna(row.plate_y)
            if row.paired:
                state = "Both"
            elif has_plate and row.knee_status == "수행자 보류":
                state = "Performer selection withheld"
            elif has_plate:
                state = "Knee missing"
            elif pd.isna(row.knee_x) or pd.isna(row.knee_y):
                state = "Both missing"
            else:
                state = "Plate missing"
            used.add(state)
            ax.add_patch(Rectangle((row.frame-lo-.5, i-.32), 1, .64,
                                   facecolor=colors[state], edgecolor="white"))
        ax.add_patch(Rectangle((-.5, i-.36), hi-lo+1, .72,
                               facecolor="none", edgecolor="#1c232b", lw=1.8))
    ax.set_yticks(range(len(labels)), labels)
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Source frames relative to earlier manual reading",
           title="Plate center and near-side knee around knee pass")
    handles = [Patch(color=c, label=k) for k,c in colors.items() if k in used]
    handles.append(Patch(facecolor="none", edgecolor="#1c232b", label="Manual range"))
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return fig
