import numpy as np
import pandas as pd
import pytest

from wlcv.evaluation.bilateral_knees import compare_pair


@pytest.mark.parametrize(("predicted", "same", "crossed", "verdict"), [
    ((1, 9), 2, 18, "양쪽 모두 같은 쪽 대응 지지"),
    ((9, 1), 18, 2, "양쪽 모두 교차 대응 지지"),
    ((4, 4), 10, 10, "동률"),
    ((3, 1), 12, 8, "교차 합만 작음"),
])
def test_knee_assignment_uses_both_manual_points(predicted, same, crossed, verdict):
    rows = pd.DataFrame({"방향": ["왼쪽", "오른쪽"], "판독": ["직접 확인"] * 2,
                         "수동 x": [0, 10], "수동 y": [0, 0], "예측 상태": ["표시"] * 2,
                         "예측 x": predicted, "예측 y": [0, 0]})
    result = compare_pair(rows)
    assert (result["같은 쪽 합(px)"], result["교차 합(px)"], result["판정"]) == (same, crossed, verdict)


def test_unreadable_knee_keeps_partial_distances():
    rows = pd.DataFrame({"방향": ["왼쪽", "오른쪽"], "판독": ["판독 불가", "직접 확인"],
                         "수동 x": [np.nan, 10], "수동 y": [np.nan, 10], "예측 상태": ["표시"] * 2,
                         "예측 x": [11, 10], "예측 y": [10, 10]})
    result = compare_pair(rows)
    assert result["판정"] == "수동 판독 불가"
    assert result["수동R–예측L"] == 1
    assert np.isnan(result["같은 쪽 합(px)"]) and np.isnan(result["교차 합(px)"])
