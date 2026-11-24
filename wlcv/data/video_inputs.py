"""학습에 쓰는 영상의 형식과 마지막 라벨 프레임을 검사한다."""

import json
import shutil
import subprocess
from collections import defaultdict

import pandas as pd

from .frames import read_rgb
from .plate_labels import label_frame

# ffprobe는 같은 계열의 컨테이너 이름을 쉼표로 묶어 반환한다.
EXPECTED_FORMATS = {
    ".mp4": "mp4",
    ".mov": "mov",
    ".m4v": "mp4",
    ".avi": "avi",
    ".mkv": "matroska",
}


def _probe_video(ffprobe, path):
    """파일에서 컨테이너, 헤더의 브랜드, 첫 영상 스트림의 코덱을 읽는다."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=format_name:format_tags=major_brand:stream=codec_name",
         "-of", "json", str(path.resolve())],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise ValueError(detail[0] if detail else "컨테이너를 읽을 수 없습니다.")

    metadata = json.loads(result.stdout)
    container = metadata.get("format", {})
    streams = metadata.get("streams", [])
    codec = streams[0].get("codec_name") if streams else None
    brand = container.get("tags", {}).get("major_brand", "").strip()
    return container.get("format_name", ""), brand, codec


def inspect_video_inputs(video_dir, labels_dir):
    """라벨이 있는 영상만 검사하고 영상당 한 행을 반환한다."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe가 필요합니다. Mac에서는 터미널에서 brew install ffmpeg를 실행하세요.")

    last_frames = {}
    for label in labels_dir.glob("*.txt"):
        video_id, frame = label_frame(label)
        last_frames[video_id] = max(last_frames.get(video_id, 0), frame)
    if not last_frames:
        raise ValueError(f"검사할 원판 라벨이 없습니다: {labels_dir}")

    files_by_id = defaultdict(list)
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.stem in last_frames:
            files_by_id[path.stem].append(path)

    rows = []
    for video_id, last_frame in sorted(last_frames.items()):
        candidates = files_by_id[video_id]
        supported = [path for path in candidates if path.suffix.lower() in EXPECTED_FORMATS]
        # 지원 영상이 없으면 같은 이름의 파일을 보여주어 확장자 문제를 찾을 수 있게 한다.
        candidates = supported or candidates
        row = {
            "영상 ID": video_id,
            "확장자": ", ".join(path.suffix or "(없음)" for path in candidates) or "—",
            "헤더 브랜드": "—",
            "코덱": "—",
            "마지막 라벨 프레임": last_frame,
            "프레임 읽기": "미검사",
            "메모": "",
        }
        rows.append(row)
        if len(candidates) != 1:
            row["메모"] = "원본 영상 없음" if not candidates else "같은 영상 ID의 파일이 여러 개"
            continue

        path = candidates[0]
        expected = EXPECTED_FORMATS.get(path.suffix.lower())
        notes = []
        if expected is None:
            notes.append("학습 파이프라인에서 지원하지 않는 확장자")
        try:
            container, brand, codec = _probe_video(ffprobe, path)
            row["헤더 브랜드"] = brand or "—"
            row["코덱"] = codec or "—"
            if not codec:
                notes.append("영상 스트림 또는 코덱을 확인할 수 없음")
            if expected and expected not in container.split(","):
                notes.append(f"확장자와 컨테이너 불일치 ({container or '알 수 없음'})")
            if path.suffix.lower() in {".mp4", ".m4v"} and brand == "qt":
                notes.append("QuickTime 브랜드(qt)")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            notes.append(f"형식 검사 실패 ({error})")

        try:
            read_rgb(path, last_frame)
            row["프레임 읽기"] = "성공"
        except (OSError, IndexError) as error:
            row["프레임 읽기"] = "실패"
            notes.append(str(error))
        row["메모"] = " · ".join(notes) or "—"

    return pd.DataFrame(rows)
