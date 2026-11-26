"""주 모델이 보이는 동안에도 보조 이력이 쌓여 다음 장의 저신뢰 후보를 연결한다."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes

from wlcv.evaluation import single_pass


def test_primary_priority_and_independent_cutmix_history(monkeypatch):
    frames = [(i * 40, np.zeros((120, 200, 3), dtype=np.uint8)) for i in range(3)]
    monkeypatch.setattr(single_pass, "timed_frames_rgb", lambda path: iter(frames))

    class Detector:
        def __init__(self, outputs):
            self.outputs = iter(outputs)

        def predict(self, image, **options):
            boxes = np.array(next(self.outputs), dtype=np.float32).reshape(-1, 6)
            return [SimpleNamespace(boxes=Boxes(boxes, image.shape[:2]))]

    primary = Detector([[[140, 80, 160, 100, .8, 0]], [], []])
    auxiliary = Detector([[[10, 10, 30, 30, .9, 0]], [[11, 10, 31, 30, .05, 0]], []])
    times = {"plate_inference": 0.0, "plate_tracking": 0.0}
    stream = single_pass.tracked_plate_frames(Path("test.mp4"), primary, "cpu", times,
                                              cutmix_model=auxiliary)
    sources = [(p.selected_model, p.source) for _, _, p in stream]
    assert sources == [("baseline", "high"), ("cutmix", "linked"), ("none", "missing")]
