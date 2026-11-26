"""같은 검증 프레임에서 미탐과 직접 판독한 허브의 세로 오차를 비교한다."""

import pandas as pd

from wlcv.evaluation.schedule_review import METHODS


def summarize_epochs(pair):
    """미탐은 모든 평가 프레임, 위치 오차는 두 모델이 모두 맞힌 직접 판독점에서 집계한다."""
    rows = []
    for method, frames in pair.groupby("method"):
        groups = [("전체", frames), *frames.groupby("video_id")]
        for video, group in groups:
            visible = group[group.visibility.eq("visible") & group.manual_y.notna()]
            common = visible[visible.match_65 & visible.match_95]
            for epoch in (65, 95):
                signed = common[f"dy_px_{epoch}"]
                rows.append({
                    "영상": video, "방식": METHODS[method], "에폭": epoch,
                    "박스 평가 프레임": len(group),
                    "미탐": int((~group[f"match_{epoch}"]).sum()),
                    "직접 판독 허브": len(visible), "공통 정탐 허브": len(common),
                    "허브 세로 평균 절대 오차(px)": signed.abs().mean(),
                    "허브 세로 평균 편향(px)": signed.mean(),
                })
    return pd.DataFrame(rows)
