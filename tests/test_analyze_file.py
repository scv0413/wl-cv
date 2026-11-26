"""수동 시점을 일부만 입력해도 해당 시점에 필요한 지표만 계산한다."""
import pandas as pd
import pytest

from wlcv.analyze import coaching_summary, phase_summary
from wlcv.data.mp4_input import validate_events


def test_catch_without_pull_labels_uses_explicit_facing():
    rows = pd.DataFrame([{"frame_index": 0, "plate_cx_px": 40., "plate_cy_px": 10.,
                          "shoulder_x_px": 25., "shoulder_y_px": 30.}])
    for direction in (1, -1):
        scores = coaching_summary(rows, None, None, {"catch": 0}, direction)
        assert scores["catch"]["forward_px"] == direction * 15.
        assert scores["catch"]["up_px"] == 20.
        assert scores["pulls"]["ratio"] is None
    assert phase_summary(rows, None, 0)["rise_px"] is None


def test_manual_finish_must_be_inside_decoded_frames():
    validate_events({"catch": 0, "finish": 2}, frame_count=3)
    with pytest.raises(ValueError, match="0~2"):
        validate_events({"finish": 3}, frame_count=3)
    with pytest.raises(ValueError, match="순서"):
        validate_events({"lift_off": 3, "finish": 2})
