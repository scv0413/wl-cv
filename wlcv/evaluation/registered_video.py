"""등록 영상의 라벨과 학습·검증 분할을 읽어 공통 영상 분석을 실행한다."""
from functools import partial
from time import perf_counter

import pandas as pd

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT, VIDEOS
from wlcv.evaluation.catch_offset import FACING
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.runtime.models import experiment_models
from wlcv.runtime.pipeline import analyze_video


def read_phases(video):
    file = ANNOTATIONS / "trunk_inclination" / "frames" / f"{video}.csv"
    rows = pd.read_csv(file).to_dict("records")
    if len(rows) != 1 or rows[0]["판독"] != "확인":
        raise ValueError(f"확인된 이륙·무릎 통과 라벨 한 쌍이 필요합니다: {file}")
    return int(rows[0]["이륙"]), int(rows[0]["무릎 통과"])


def analyze_registered(video, *, device, read_mode, plate_imgsz, plate_mode):
    """기존 선수별 분할과 가중치별 학습 명단을 확인해 등록 영상을 분석한다."""
    start = perf_counter()
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    if video not in metadata.index:
        raise ValueError(f"등록된 검증 영상 ID 또는 MP4 경로를 입력하세요: {video}")
    info = metadata.loc[video]
    timing = pd.read_csv(ROOT / "data/metadata/video_timing.csv").set_index("video_id")
    fps = float(timing.at[video, "fps_from_median_gap"])
    lift, knee = read_phases(video)
    if not 0 <= lift < knee < info.frame_count or info.near_side not in ("left", "right") or not fps > 0:
        raise ValueError("수동 시점 순서, 가까운 쪽과 재생 fps를 확인하세요.")
    events = load_events(ANNOTATIONS, video, pass_number=1)
    direction = FACING.get(video)
    path = VIDEOS / info.filename
    parent = ROOT / "outputs/video_analysis" / video
    events = dict(events, lift_off=lift, knee_pass=knee)
    return analyze_video(
        path, info, video=video, fps=fps, direction=direction, events=events, parent=parent,
        weights=experiment_models(), input_mode="validation_id", start=start, device=device,
        read_mode=read_mode, plate_imgsz=plate_imgsz, plate_mode=plate_mode,
        detector=partial(validation_plate_model, video_ids=[video]),
    )
