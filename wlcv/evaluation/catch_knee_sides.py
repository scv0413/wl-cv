"""두 검증 선수의 1차 캐치에서 수동 왼쪽과 오른쪽 각각의 최근접 예측을 비교한다."""

import math

import pandas as pd

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT, SPLIT
from wlcv.evaluation.bilateral_knees import read_knees

SCENES = {"c04_sn_f_01": 78, "c08_sn_f_01": 58}
CROP = (400, 1320, 735, 1650)
FOLDER = ANNOTATIONS / "catch_knee_sides" / "validation"
# c04의 1차 무릎 직접점은 기존 판독 파일에서 읽는다.
ORIGINAL = {("c04_sn_f_01", "left"): ANNOTATIONS}


def check_scenes():
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id", verify_integrity=True)
    splits = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    for video, frame in SCENES.items():
        meta, split = metadata.loc[video], splits.loc[video]
        if (meta.person_id, meta.near_side, split.role) != (split.person_id, "left", "validation"):
            raise ValueError(f"{video}의 선수·검증 분할·카메라 방향을 확인하세요.")
        if load_events(ANNOTATIONS, video, 1).get("catch") != frame:
            raise ValueError(f"{video}의 1차 캐치 프레임이 달라졌습니다.")


def read_catches(folder=FOLDER):
    tables = []
    for video, frame in SCENES.items():
        sources = {(frame, side): path for (vid, side), path in ORIGINAL.items() if vid == video}
        points = read_knees(folder, video_id=video, frames=[frame], roi=CROP, original_sources=sources)
        points.insert(0, "영상", video)
        tables.append(points)
    return pd.concat(tables, ignore_index=True)


def opposite_pattern(score):
    """한 수동 무릎 안에서 같은 쪽 거리와 반대쪽 거리를 비교한다. 양수면 반대쪽이 가깝다."""
    left = score["수동L–예측L"] - score["수동L–예측R"]
    right = score["수동R–예측R"] - score["수동R–예측L"]
    if not (math.isfinite(left) and math.isfinite(right)):
        status = "판단 보류"
    elif left > 0 and right > 0:
        status = "관찰"
    else:
        status = "미관찰"
    return {"왼쪽 거리 차(px)": left, "오른쪽 거리 차(px)": right, "두 무릎 모두 반대쪽": status}


def repeat_summary(scores):
    states = scores["두 무릎 모두 반대쪽"]
    observed = states.eq("관찰").sum()
    valid = states.ne("판단 보류").sum()
    if valid == 2:
        conclusion = "두 선수 모두에서 반복 관찰됨" if observed == 2 else "두 선수 모두에서 반복되는 패턴은 확인되지 않음"
    else:
        conclusion = "두 선수 비교 미완료"
    return f"{conclusion} · 관찰 {observed}명 / 판정 가능 {valid}명 (전체 2명)"
