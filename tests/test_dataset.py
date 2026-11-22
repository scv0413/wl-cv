"""작은 가상 영상으로 프레임 번호, 색 순서, 데이터 분할을 확인한다."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from wlcv.data.dataset import prepare_dataset
from wlcv.data.frames import read_rgb


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.videos = self.root / "videos"
        self.labels = self.root / "boxes"
        self.split = self.root / "people.csv"
        self.output = self.root / "dataset"
        self.videos.mkdir()
        self.labels.mkdir()
        self.split.write_text(
            "video_id,person_id,role\none,a,train\ntwo,b,validation\n", encoding="utf-8"
        )
        for name in ("one", "two"):
            writer = cv2.VideoWriter(
                str(self.videos / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            try:
                for bgr in ((200, 40, 10), (10, 40, 200), (60, 90, 120)):
                    writer.write(np.full((24, 32, 3), bgr, dtype=np.uint8))
            finally:
                writer.release()
        (self.labels / "one_f0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (self.labels / "two_f1.txt").touch()

    def build(self):
        return prepare_dataset(self.videos, self.labels, self.split, self.output)

    def test_frame_index_color_and_exported_labels(self):
        self.assertTrue(np.allclose(read_rgb(self.videos / "two.avi", 1)[0, 0], [200, 40, 10], atol=4))
        config = self.build()
        filenames = {p.relative_to(self.output).as_posix() for p in self.output.rglob("*.jpg")}
        self.assertEqual(filenames, {"images/train/one_f0.jpg", "images/val/two_f1.jpg"})
        self.assertEqual((self.output / "labels/val/two_f1.txt").read_bytes(), b"")
        first_build = config.stat().st_mtime_ns
        self.assertEqual(self.build().stat().st_mtime_ns, first_build)
        (self.output / "images/train/one_f0.jpg").write_bytes(b"broken")
        with self.assertRaises(FileExistsError):
            self.build()

    def test_person_leakage_fails_before_export(self):
        self.split.write_text("video_id,person_id,role\none,a,train\ntwo,a,validation\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "학습과 검증"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_missing_frame_leaves_no_partial_dataset(self):
        (self.labels / "one_f8.txt").touch()
        with self.assertRaises(IndexError):
            self.build()
        self.assertFalse(self.output.exists())
