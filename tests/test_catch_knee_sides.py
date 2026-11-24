import numpy as np
import pandas as pd

from wlcv.evaluation.catch_knee_sides import opposite_pattern, repeat_summary


def test_each_manual_knee_compares_its_own_two_distances():
    distances = {"수동L–예측L": 20, "수동L–예측R": 16, "수동R–예측L": 7, "수동R–예측R": 10}
    assert opposite_pattern(distances)["두 무릎 모두 반대쪽"] == "관찰"
    distances["수동R–예측R"] = 7
    assert opposite_pattern(distances)["두 무릎 모두 반대쪽"] == "미관찰"
    distances["수동R–예측R"] = np.nan
    assert opposite_pattern(distances)["두 무릎 모두 반대쪽"] == "판단 보류"
    scores = pd.DataFrame({"두 무릎 모두 반대쪽": ["관찰", "판단 보류"]})
    assert "관찰 1명 / 판정 가능 1명 (전체 2명)" in repeat_summary(scores)
