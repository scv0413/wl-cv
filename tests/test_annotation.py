import csv

import pytest

from wlcv.annotation.storage import load_boxes, save_boxes, save_event


def test_personal_boxes_preserve_original_and_explicit_empty(tmp_path):
    video_id = "a01_sn_f_01"
    original_dir = tmp_path / "plate_boxes"
    original_dir.mkdir()
    original = original_dir / f"{video_id}_f3.txt"
    original.write_text("0 0.5 0.5 0.2 0.2\n")
    baseline = load_boxes(tmp_path, video_id, 3)
    assert baseline == ("원본 라벨", [(0.5, 0.5, 0.2, 0.2)])
    boxes = [*baseline[1], (0.2, 0.4, 0.1, 0.2)]

    save_boxes(tmp_path, video_id, 3, boxes, previous=baseline)
    personal = load_boxes(tmp_path, video_id, 3)
    assert personal == ("개인 라벨", boxes)
    with pytest.raises(ValueError):
        save_boxes(tmp_path, video_id, 3, [], previous=personal)
    with pytest.raises(ValueError):
        save_boxes(tmp_path, video_id, 3, boxes[:1], previous=baseline)
    save_boxes(tmp_path, video_id, 3, [], previous=personal, reviewed_empty=True)
    assert load_boxes(tmp_path, video_id, 3) == ("개인 라벨", [])
    assert load_boxes(tmp_path, video_id, 4) == ("미라벨", [])
    assert original.read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_second_pass_locks_only_that_video_and_preserves_source(tmp_path):
    video_id = "a01_sn_f_01"
    other_video = "a06_sn_h_01"
    original = tmp_path / "point_labels.csv"
    contents = ("video_id,event,frame,pass,notes\n"
                "a01_sn_f_01,ready,3,1,first pass\n"
                "a06_sn_h_01,drive,27,1,other video\n")
    original.write_text(contents)

    target = save_event(tmp_path, video_id, 2, "ready", 2, previous=None)
    save_event(tmp_path, video_id, 2, "ready", 4, previous=2)
    with pytest.raises(ValueError):
        save_event(tmp_path, video_id, 1, "ready", 5, previous=3)
    with pytest.raises(ValueError):
        save_event(tmp_path, video_id, 2, "ready", 7, previous=2)

    # 한 영상의 2차 판독은 다른 영상의 1차 판독을 잠그지 않는다.
    save_event(tmp_path, other_video, 1, "drive", 28, previous=27)
    with pytest.raises(ValueError):
        save_event(tmp_path, other_video, 1, "catch", 30, previous=None)

    with target.open(newline="") as source:
        rows = sorted((r["video_id"], r["event"], r["pass"], r["frame"], r["notes"])
                      for r in csv.DictReader(source))
    assert rows == [
        (video_id, "ready", "1", "3", "first pass"),
        (video_id, "ready", "2", "4", ""),
        (other_video, "drive", "1", "28", "other video"),
    ]
    assert original.read_text() == contents
