"""b02 두 끝점을 다시 직접 읽은 뒤, 같은 모델 출력의 상승량 오차를 비교한다."""
import numpy as np
import pandas as pd

from wlcv.annotation.hub_points import VIDEO
from wlcv.config import ANNOTATIONS

FRAMES = (30, 36)
POINTS = ANNOTATIONS / "hub_rise_repeat" / f"{VIDEO}.csv"


def compare_repeat(baseline, repeat):
    """37번 실험의 수동 높이와 모델 출력을 고정해 두 판독의 결과를 계산한다."""
    if set(repeat.frame_index) != set(FRAMES) or not repeat.visibility.eq("visible").all():
        raise ValueError("두 프레임 모두 직접 판독한 경우에만 상승량을 비교합니다.")
    saved = baseline[baseline.video_id.eq(VIDEO)].set_index("imgsz", verify_integrity=True)
    if (set(saved.index) != {640, 800} or not saved.knee_frame.eq(30).all()
            or not saved.contact_frame.eq(36).all()):
        raise ValueError("37번 실험의 b02 30·36번과 두 입력 크기를 확인하세요.")
    points = repeat.set_index("frame_index", verify_integrity=True)
    first = saved.loc[640]
    point_table = pd.DataFrame({
        "프레임": FRAMES,
        "1차 허브 y": [first.knee_manual_y, first.contact_manual_y],
        "2차 허브 y": [points.at[i, "manual_y"] for i in FRAMES],
    })
    point_table["y 변화(px, 아래 +)"] = point_table["2차 허브 y"] - point_table["1차 허브 y"]
    rows = []
    for reading in ["1차 허브 y", "2차 허브 y"]:
        knee_y, contact_y = point_table[reading]
        manual_rise = knee_y - contact_y
        errors = saved.model_rise_px - manual_rise
        closer = "동일"
        if not np.isclose(abs(errors[640]), abs(errors[800]), rtol=0, atol=1e-6):
            closer = str(errors.abs().idxmin())
        rows.append({"판독": reading[:2], "수동 상승량(px)": manual_rise,
                     "640 상승량 오차(px)": errors[640], "800 상승량 오차(px)": errors[800],
                     "두 입력 모두 과소측정": bool((errors < 0).all()), "절대 오차가 작은 입력": closer,
                     "800 무릎 통과 오차(아래 +)": saved.at[800, "knee_model_y"] - knee_y,
                     "800 접촉 오차(아래 +)": saved.at[800, "contact_model_y"] - contact_y})
    return point_table, pd.DataFrame(rows)
