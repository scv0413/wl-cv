from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from ultralytics.engine.results import Results

from wlcv.data.pose_dataset import PoseFrame
from wlcv.evaluation.pose_comparison import detection_summary, measure_frame, paired_errors


def test_errors_use_the_matched_performer_and_the_labeled_side():
    frame = PoseFrame(Path("scene.mp4"), 10, "val", "right", (10, 10, 50, 90),
                      {"knee": (30, 55), "ankle": (30, 80)})
    boxes = torch.tensor([[70, 10, 99, 80, .99, 0], [10, 10, 50, 90, .80, 0]])
    points = torch.zeros(2, 17, 3)
    points[1, 13] = torch.tensor([30, 55, .99])  # 반대쪽 무릎은 정답과 겹친다.
    points[1, 14] = torch.tensor([33, 59, .80])
    points[1, 16] = torch.tensor([30, 80, .30])
    pred = Results(np.zeros((100, 100, 3), dtype=np.uint8), "scene", {0: "person"},
                   boxes=boxes, keypoints=points)
    before = pd.DataFrame(measure_frame(frame, pred))
    knee, ankle = before.iloc[0], before.iloc[1]
    assert not knee["수행자 미탐"] and knee["오차(px)"] == pytest.approx(5)
    assert knee["박스 높이 대비(%)"] == pytest.approx(6.25)
    assert ankle["상태"] == "저신뢰" and np.isnan(ankle["오차(px)"])

    # 같은 프레임에서 배경 사람만 남으면 오차가 0으로 집계되지 않아야 한다.
    after = pd.DataFrame(measure_frame(frame, pred[:1]))
    counts = detection_summary({"before": before, "after": after})
    assert counts["수행자 미탐"].tolist() == [0, 1]
    detail, summary = paired_errors(before, after)
    assert len(detail) == 2 and detail["변화(px)"].isna().all()
    assert summary["양쪽에서 측정한 수"].tolist() == [0, 0]
    _, same = paired_errors(before, before)
    assert same.at["무릎", "평균 변화(px)"] == 0
    assert same.at["무릎", "양쪽에서 측정한 수"] == 1
    empty = pd.DataFrame(measure_frame(frame, pred[:0]))
    assert empty["수행자 미탐"].all() and empty["후보 IoU"].isna().all()
