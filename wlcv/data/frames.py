"""영상의 첫 장을 0번으로 세고, 시간 순서대로 RGB 이미지를 읽는다."""

import os
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np


def frames_rgb(path: str | Path) -> Iterator[np.ndarray]:
    """영상의 모든 프레임을 차례로 읽는다. 각 배열의 형태는 (높이, 너비, 3)이다."""
    for _, rgb in timed_frames_rgb(path):
        yield rgb


def timed_frames_rgb(path: str | Path) -> Iterator[tuple[float, np.ndarray]]:
    """원본 표시 시각(ms)과 회전이 반영된 RGB 배열을 디코딩 순서대로 돌려준다."""
    if os.environ.get("WL_VIDEO_DECODER") == "pyav":
        yield from pyav_frames_rgb(path)
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"영상을 열 수 없습니다: {path}")
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            time_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            yield time_ms, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def pyav_frames_rgb(path: str | Path) -> Iterator[tuple[float, np.ndarray]]:
    """Linux 배포에서 FFmpeg 버전과 색 변환을 고정하고 원본 표시 시각을 보존한다."""
    import av

    from wlcv.data.mp4_input import VideoInputError

    rotations = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE,
                 180: cv2.ROTATE_180, -180: cv2.ROTATE_180}
    try:
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            # 스트림 하나를 순서대로 읽는다. 모델 추론 중 디코더 작업 스레드를 유지하지 않는다.
            stream.codec_context.thread_count = 1
            start = float((stream.start_time or 0) * stream.time_base)
            for frame in video.decode(stream):
                if frame.time is None:
                    raise VideoInputError("영상 프레임의 표시 시각을 읽을 수 없습니다.")
                # Mac 기준 디코더와 같은 방식으로 색차 채널을 확대해 BGR로 변환한다.
                bgr = frame.reformat(format="bgr24", interpolation="BICUBIC").to_ndarray()
                rotation = rotations.get(frame.rotation)
                if rotation is not None:
                    bgr = cv2.rotate(bgr, rotation)
                yield (frame.time - start) * 1000, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except av.FFmpegError as exc:
        raise VideoInputError("영상 스트림을 디코딩할 수 없습니다.") from exc


def read_rgb(video_path: str | Path, frame_index: int) -> np.ndarray:
    """앞에서부터 디코딩하여 라벨에 기록된 프레임 번호와 일치하는 한 장을 읽는다."""
    if frame_index < 0:
        raise ValueError("프레임 번호는 0 이상이어야 합니다.")
    with closing(frames_rgb(video_path)) as frames:
        for index, rgb in enumerate(frames):
            if index == frame_index:
                return rgb
    raise IndexError(f"{video_path}: {frame_index}번 프레임에 도달하기 전에 영상이 끝났습니다.")
