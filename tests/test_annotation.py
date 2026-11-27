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


def test_one_pixel_box_edit_saves_and_reopens_at_source_resolution(tmp_path):
    import cv2
    import numpy as np
    import ipywidgets as widgets
    from wlcv.annotation.boxes import plate_box_editor
    from wlcv.annotation.pixel_edges import BoxEdgeSliders

    # 960px 원본은 편집기에서 720px로 축소 표시된다.
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (960, 540))
    try:
        writer.write(np.zeros((540, 960, 3), dtype=np.uint8))
    finally:
        writer.release()
    folder = tmp_path / "plate_boxes"
    folder.mkdir()
    original = folder / "sample_f0.txt"
    original.write_text("0 0.5 0.5 0.5 0.5\n")

    editor = plate_box_editor(video, tmp_path)
    panel = next(w for w in editor.children if isinstance(w, BoxEdgeSliders))
    left, top, right, bottom = panel.edges
    assert [s.value for s in panel.edges] == [240, 135, 720, 405]
    assert [s.max for s in panel.edges] == [960, 540, 960, 540]
    preview = panel.detail.value
    left.value += 1
    assert panel.detail.value != preview
    assert load_boxes(tmp_path, "sample", 0)[0] == "원본 라벨"

    buttons = next(w for w in editor.children if type(w) is widgets.Box)
    next(b for b in buttons.children if b.description == "박스 저장").click()
    source, saved = load_boxes(tmp_path, "sample", 0)
    assert source == "개인 라벨"
    x, y, w, h = saved[0]
    assert [(x-w/2)*960, (y-h/2)*540, (x+w/2)*960, (y+h/2)*540] == pytest.approx(
        [241, 135, 720, 405], abs=0.00002)
    reopened = plate_box_editor(video, tmp_path)
    loaded = next(w for w in reopened.children if isinstance(w, BoxEdgeSliders))
    assert [s.value for s in loaded.edges] == [241, 135, 720, 405]
    assert original.read_text() == "0 0.5 0.5 0.5 0.5\n"
