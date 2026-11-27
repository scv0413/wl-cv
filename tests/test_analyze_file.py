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


def test_peak_uses_source_time_and_excludes_catch_from_search_and_range():
    from wlcv.runtime.apex import drive_to_peak

    frames = pd.DataFrame({"plate_cy_px": [100., 80., 70., 71., 40.],
                           "source_time_ms": [20., 60., 120., 220., 245.]})
    peak = drive_to_peak(frames, 0, 4)
    assert peak["peak_frame"] == 2 and peak["rise_px"] == 30
    assert peak["mean_speed_px_s"] == pytest.approx(300)
    assert peak["speed_range_px_s"] == pytest.approx([145, 500])
    boundary = drive_to_peak(frames, 0, 3)
    assert boundary["peak_frame"] == 2 and boundary["speed_range_px_s"] is None
    assert boundary["range_reason"]


def test_endpoint_iou_rejection_uses_both_tracking_centers():
    import numpy as np
    from wlcv.evaluation.plate_endpoint_rise import check_endpoint, select_rise

    box = np.array([0., 0., 20., 20.])
    cosine = pd.Series({"x1": 0., "y1": 2., "x2": 20., "y2": 22., "cy": 12.})
    knee = check_endpoint(1, box, 10., "baseline", cosine)
    contact = check_endpoint(2, box + 100., 110., "cutmix", cosine)
    assert knee["accepted"] and not contact["accepted"]
    assert select_rise([knee, contact]) == ("tracking", -100.)
    missing = check_endpoint(2, np.full(4, np.nan), np.nan, "none", cosine)
    assert select_rise([knee, missing]) == (None, None)


def test_explicit_deployment_folder_keeps_missing_models_in_that_folder(monkeypatch, tmp_path):
    """배포 폴더에 없는 모델을 로컬 학습 폴더에서 조용히 읽지 않는다."""
    from wlcv.runtime.models import experiment_models, file_models

    monkeypatch.delenv("WL_MODEL_DIR", raising=False)
    assert file_models() == experiment_models()
    monkeypatch.setenv("WL_MODEL_DIR", str(tmp_path))
    weights = file_models()
    assert weights.plate == tmp_path / "pytorch/plate_baseline.pt"
    assert weights.cosine == tmp_path / "pytorch/plate_cosine.pt"
