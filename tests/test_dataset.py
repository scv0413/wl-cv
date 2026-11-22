"""작은 영상으로 프레임과 라벨의 대응을 확인한다."""

import cv2
import numpy as np
import pytest

from wlcv.data.dataset import prepare_dataset, read_splits
from wlcv.data.frames import read_rgb


@pytest.fixture
def sample(tmp_path):
    videos = tmp_path / "videos"
    labels = tmp_path / "boxes"
    videos.mkdir()
    labels.mkdir()
    (tmp_path / "people.csv").write_text("video_id,person_id,role\none,a,train\ntwo,b,validation\n")
    for name in ("one", "two"):
        writer = cv2.VideoWriter(str(videos / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        try:
            for bgr in ((200, 40, 10), (10, 40, 200)):
                writer.write(np.full((24, 32, 3), bgr, dtype=np.uint8))
        finally:
            writer.release()
    (labels / "one_f0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels / "two_f1.txt").touch()
    return tmp_path


def build(root):
    return prepare_dataset(root / "videos", root / "boxes", root / "people.csv", root / "dataset")


def test_frame_index_color_and_exported_labels(sample):
    rgb = read_rgb(sample / "videos/two.avi", 1)
    assert np.allclose(rgb[0, 0], [200, 40, 10], atol=4)

    output = build(sample).parent
    images = {p.relative_to(output).as_posix() for p in output.rglob("*.jpg")}
    assert images == {"images/train/one_f0.jpg", "images/val/two_f1.jpg"}
    assert (output / "labels/val/two_f1.txt").read_bytes() == b""

    (output / "images/train/one_f0.jpg").write_bytes(b"broken")
    with pytest.raises(FileExistsError):
        build(sample)


def test_missing_frame_leaves_no_partial_dataset(sample):
    (sample / "boxes/one_f8.txt").touch()
    with pytest.raises(IndexError):
        build(sample)
    assert not (sample / "dataset").exists()


def test_person_cannot_appear_in_both_splits(tmp_path):
    split = tmp_path / "people.csv"
    split.write_text("video_id,person_id,role\none,a,train\ntwo,a,validation\n")
    with pytest.raises(ValueError):
        read_splits(split)
