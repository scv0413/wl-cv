"""영상의 첫 장을 0번으로 세고, 시간 순서대로 RGB 이미지를 읽는다."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np


def frames_rgb(path: str | Path) -> Iterator[np.ndarray]:
    """영상의 모든 프레임을 차례로 읽는다. 각 배열의 형태는 (높이, 너비, 3)이다."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"영상을 열 수 없습니다: {path}")
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def read_rgb(video_path: str | Path, frame_index: int) -> np.ndarray:
    """앞에서부터 디코딩하여 라벨에 기록된 프레임 번호와 일치하는 한 장을 읽는다."""
    if frame_index < 0:
        raise ValueError("프레임 번호는 0 이상이어야 합니다.")
    with closing(frames_rgb(video_path)) as frames:
        for index, rgb in enumerate(frames):
            if index == frame_index:
                return rgb
    raise IndexError(f"{video_path}: {frame_index}번 프레임에 도달하기 전에 영상이 끝났습니다.")
