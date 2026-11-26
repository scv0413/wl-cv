"""검증 목록이나 수동 라벨 없이 MP4 자체에서 오버레이에 필요한 정보를 읽는다."""
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from statistics import median

import cv2


@dataclass
class FileVideoInfo:
    width: int
    height: int
    near_side: str
    fps: float
    # 편집된 MP4는 헤더에 실제로 표시하지 않는 프레임까지 담을 수 있다.
    frame_count: int | None = None


def inspect_mp4(path, near_side):
    if path.suffix.lower() != ".mp4" or not path.is_file():
        raise ValueError(f"읽을 수 있는 MP4 파일 경로를 확인하세요: {path}")
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=format_name:stream=time_base,avg_frame_rate:packet=pts,flags",
             "-of", "json", str(path.absolute())], capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ValueError("영상 시각을 읽으려면 ffprobe가 필요합니다 (brew install ffmpeg).") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"MP4 컨테이너를 읽을 수 없습니다: {path}") from exc
    header = json.loads(probe.stdout)
    if "mp4" not in header["format"]["format_name"].split(",") or not header.get("streams"):
        raise ValueError("MP4의 영상 스트림을 찾을 수 없습니다.")
    stream = header["streams"][0]
    # B 프레임의 저장 순서와 상관없이 표시 시각 순서로 간격을 구한다.
    pts = sorted({p["pts"] for p in header.get("packets", [])
                  if "pts" in p and "D" not in p.get("flags", "")})
    if len(pts) >= 2:
        step = median(b - a for a, b in zip(pts, pts[1:]))
        fps = 1 / (float(Fraction(stream["time_base"])) * step)
    else:
        try:
            fps = float(Fraction(stream["avg_frame_rate"]))
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("영상 재생 FPS를 확인할 수 없습니다.") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        # 추론에 쓰는 디코더에서 회전 정보가 적용된 가로·세로를 읽는다.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or min(width, height, fps) <= 0:
            raise ValueError("MP4의 표시 크기 또는 FPS를 확인할 수 없습니다.")
    finally:
        cap.release()
    return FileVideoInfo(width, height, near_side, fps)


EVENT_KEYS = ("lift_off", "knee_pass", "drive", "catch", "finish")


def validate_events(events, frame_count=None):
    if set(events).difference(EVENT_KEYS):
        raise ValueError("지원하지 않는 수동 시점입니다.")
    values = [events[name] for name in EVENT_KEYS if name in events]
    if any(type(frame) is not int or frame < 0 for frame in values):
        raise ValueError("수동 시점은 0부터 세는 정수 프레임 번호입니다.")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("이륙 < 무릎 통과 < 접촉 < 캐치 < 피니시 순서로 입력하세요.")
    if frame_count is not None and values and values[-1] >= frame_count:
        raise ValueError(f"수동 시점은 실제 디코딩된 0~{frame_count - 1}번 안에 있어야 합니다.")
