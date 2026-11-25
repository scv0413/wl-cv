"""이륙·무릎 통과·바 접촉의 원본 시각과 수동 허브로 두 구간 속도를 비교한다."""

import numpy as np
import pandas as pd

from wlcv.annotation.plate_blur import REVIEW_FILE
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, RUNS, VIDEOS
from wlcv.data.frames import timed_frames_rgb
from wlcv.evaluation.full_video import detect_videos
from wlcv.evaluation.plate_track import select_centers

PHASES = ("lift", "knee", "contact")


def phases(video):
    saved = pd.read_csv(ANNOTATIONS / "trunk_inclination/frames" / f"{video}.csv")
    if len(saved) != 1 or saved.iloc[0]["판독"] != "확인":
        raise ValueError(f"{video} 이륙 판독을 확인하세요.")
    events = load_events(ANNOTATIONS, video, pass_number=1)
    lift, knee = int(saved.iloc[0]["이륙"]), int(saved.iloc[0]["무릎 통과"])
    contact = events["drive"]
    if knee != events["transition"] or not 0 <= lift < knee < contact:
        raise ValueError(f"{video} 동작 순서를 확인하세요.")
    return dict(zip(PHASES, [lift, knee, contact]))


def prior_hubs():
    """기존 시간 연결 실험에서 시각 판독한 좌표와 판독 불가 상태를 함께 읽는다."""
    files = ["temporal_plate_centers.csv", "additional_temporal_review.csv"]
    rows = pd.concat([pd.read_csv(ANNOTATIONS / name) for name in files], ignore_index=True)
    return rows.set_index(["video_id", "frame_index"], verify_integrity=True)


def model_plates(videos):
    frames, candidates = detect_videos(RUNS / "plate_baseline/weights/best.pt", videos)
    choices = select_centers(frames, candidates, distance_ratio=0.5)
    return choices[choices.method == "Past link"].copy()


def vertical_speeds(time_ms, height_y):
    """속도는 원본 px/s, 위쪽이 양수. 1 px 이하의 후반 상승량은 비율 계산을 보류한다."""
    dt = np.diff(np.asarray(time_ms, dtype=float)) / 1000
    if len(dt) != 2 or not np.isfinite(dt).all() or (dt <= 0).any():
        return dict(early=np.nan, late=np.nan, ratio=np.nan, reason="시각 확인 필요")
    rise = -np.diff(np.asarray(height_y, dtype=float))
    early, late = rise / dt
    if not np.isfinite(rise).all():
        reason = "끝점 누락"
    elif rise[1] <= 1:
        reason = "후반 상승량 1 px 이하"
    else:
        reason = ""
    return dict(early=early, late=late, ratio=np.nan if reason else early/late, reason=reason)


def compare_speeds(plates):
    blind = pd.read_csv(REVIEW_FILE).set_index(["video_id", "frame_index"], verify_integrity=True)
    prior = prior_hubs()
    selected = plates.set_index(["video_id", "frame_index"], verify_integrity=True)
    points, trials = [], []
    for video in sorted(blind.index.get_level_values(0).unique()):
        phase = phases(video)
        times = {i:t for i,(t, _) in enumerate(timed_frames_rgb(VIDEOS / f"{video}.mp4")) if i in phase.values()}
        rows = []
        for event, i in phase.items():
            key = (video, i)
            state, y = "미판독", np.nan
            if event == "lift" and key in prior.index:
                point = prior.loc[key]
                state = point.visibility
                y = point.manual_y if state == "visible" else np.nan
            elif event != "lift":
                point = blind.loc[key]
                state = point.hub_source
                y = point.manual_y if state in ("clicked", "previous_blind") else np.nan
            chosen = selected.loc[key]
            rows.append(dict(video_id=video, phase=event, frame_index=i, time_ms=times[i],
                             hub_status=state, manual_y=y, model_y=chosen.cy, source=chosen.source))
        points.extend(rows)
        tab = pd.DataFrame(rows)
        result = {"video_id": video}
        for method in ["manual", "model"]:
            speeds = vertical_speeds(tab.time_ms, tab[f"{method}_y"])
            result.update({f"{method}_{key}": value for key, value in speeds.items()})
        trials.append(result)
    return pd.DataFrame(points), pd.DataFrame(trials)
