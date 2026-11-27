"""같은 프레임의 사람·원판 후보 순서와 무관하게 전경 선수의 결과를 선택한다."""
import numpy as np
import pandas as pd

from wlcv.runtime.foreground import select_performer
from wlcv.runtime.inference import track_candidates
from wlcv.runtime.tracking import BOX, PlateHistory


def test_other_lifters_high_confidence_plate_does_not_enter_history():
    background = [5, 10, 45, 85]
    foreground = [125, 25, 220, 185]
    people = np.array([background, foreground], dtype=float)
    selected, reason = select_performer(people, None)
    assert selected == 1 and reason == "foreground_size"
    # 앞 선수의 원판은 캐치 때 사람 박스 바로 위에 위치한다.
    table = pd.DataFrame([[12, 30, 36, 54, .99], [149, 2, 173, 26, .7]],
                         columns=[*BOX, "confidence"])
    table["video_id"] = "lift"
    table["frame_index"] = 0
    histories = {"baseline": PlateHistory(), "cutmix": PlateHistory()}
    for frame, available in enumerate((table, table.iloc[:1])):
        plate = track_candidates({"baseline": available}, histories, "lift", frame,
                                 (200, 250), "continuity", people, selected)
        if frame == 0:
            assert plate.cx == 161 and plate.source == "high"
        else:
            assert plate.source == "missing"
    assert len(histories["baseline"].history) == 1
    assert select_performer(people[::-1], None)[0] == 0


def test_similar_sizes_use_previous_person_or_withhold():
    boxes = np.array([[5, 10, 65, 110], [135, 10, 197, 110]], dtype=float)
    assert select_performer(boxes, None) == (None, "ambiguous")
    assert select_performer(boxes, boxes[1]) == (1, "previous_box")


def test_partial_and_full_candidates_on_same_lifter_use_continuity():
    previous = np.array([100, 20, 185, 190])
    boxes = np.array([[83, 20, 210, 190], [101, 20, 185, 190]])
    assert select_performer(boxes, previous) == (1, "previous_box")
    assert select_performer(np.stack([previous, previous]), previous) == (None, "ambiguous")
