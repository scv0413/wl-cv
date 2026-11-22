import csv

import pytest

from wlcv.annotation.storage import events_for, load_boxes, load_events, save_boxes, save_event


def test_personal_boxes_preserve_original_and_explicit_empty(tmp_path):
    video_id = "a01_sn_f_01"
    original_dir = tmp_path / "plate_boxes"
    original_dir.mkdir()
    original = original_dir / f"{video_id}_f3.txt"
    original.write_text("0 0.5 0.5 0.2 0.2\n")
    baseline = load_boxes(tmp_path, video_id, 3)
    assert baseline == ("원본 라벨", [(0.5, 0.5, 0.2, 0.2)])
    boxes = [*baseline[1], (0.2, 0.4, 0.1, 0.2)]

    path = save_boxes(tmp_path, video_id, 3, boxes, previous=baseline)
    assert path.name == f"{video_id}_frame_0003.txt"
    personal = load_boxes(tmp_path, video_id, 3)
    assert personal == ("개인 라벨", boxes)
    with pytest.raises(ValueError):
        save_boxes(tmp_path, video_id, 3, [], previous=personal)
    with pytest.raises(ValueError):
        save_boxes(tmp_path, video_id, 3, boxes[:1], previous=baseline)
    assert len(path.read_text().splitlines()) == 2
    save_boxes(tmp_path, video_id, 3, [], previous=personal, reviewed_empty=True)
    assert load_boxes(tmp_path, video_id, 3) == ("개인 라벨", [])
    assert load_boxes(tmp_path, video_id, 4) == ("미라벨", [])
    assert original.read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_event_edit_keeps_other_records_and_extra_columns(tmp_path):
    video_id = "a01_sn_f_01"
    original = tmp_path / "point_labels.csv"
    contents = ("video_id,event,frame,pass,notes\n"
                "a01_sn_f_01,ready,3,1,keep\n"
                "a01_sn_f_01,ready,2,2,other pass\n"
                "a06_sn_h_01,drive,27,1,other video\n")
    original.write_text(contents)
    with pytest.raises(ValueError, match="잠겨"):
        save_event(tmp_path, video_id, 1, "ready", 4, previous=3)
    target = save_event(tmp_path, video_id, 2, "ready", 4, previous=2)
    save_event(tmp_path, video_id, 2, "finish", 81, previous=None)
    assert load_events(tmp_path, video_id, 2) == {"ready": 4, "finish": 81}
    with target.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [r["notes"] for r in rows] == ["keep", "other pass", "other video", ""]
    assert [r["frame"] for r in rows] == ["3", "4", "27", "81"]
    assert events_for("a06_sn_h_01") == ("ready", "transition", "drive")
    with pytest.raises(ValueError):
        save_event(tmp_path, "a06_sn_h_01", 1, "catch", 9, previous=None)
    with pytest.raises(ValueError):
        save_event(tmp_path, video_id, 2, "ready", 7, previous=3)
    assert original.read_text() == contents


def test_starting_second_pass_locks_first_pass_for_same_video(tmp_path):
    video_id = "a01_sn_f_01"
    save_event(tmp_path, video_id, 1, "ready", 3, previous=None)
    save_event(tmp_path, video_id, 2, "ready", 6, previous=None)
    with pytest.raises(ValueError, match="잠겨"):
        save_event(tmp_path, video_id, 1, "ready", 5, previous=3)
    assert load_events(tmp_path, video_id, 1) == {"ready": 3}
    save_event(tmp_path, "a06_sn_h_01", 1, "ready", 4, previous=None)
