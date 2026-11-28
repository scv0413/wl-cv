"""합성 MP4 한 편을 FastAPI에 제출하고 다운로드한 오버레이·CSV를 실제로 읽는다."""
import argparse
import csv
import hashlib
import shutil
import json
import subprocess
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import requests


FRAMES = 24


def generate_video(path):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=size=384x640:rate=24:duration=1", "-frames:v", str(FRAMES),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2", str(path)], check=True)


def read_overlay(source, overlay):
    stream = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
         "-of", "default=nokey=1:noprint_wrappers=1", str(overlay)], text=True).strip()
    original, rendered = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(overlay))
    count = changed = 0
    try:
        while True:
            ok_src, src = original.read()
            ok_out, out = rendered.read()
            if not ok_src and not ok_out:
                break
            if not ok_src or not ok_out or src.shape != out.shape:
                raise ValueError("원본과 오버레이의 디코딩 길이 또는 해상도가 다릅니다.")
            # 원판 위치에 따라 분석 자막은 상단 또는 하단에 표시된다.
            difference = cv2.absdiff(src, out)
            if max(np.mean(difference[12:150]), np.mean(difference[-150:-12])) > 5:
                changed += 1
            count += 1
    finally:
        original.release()
        rendered.release()
    if stream != "h264" or count != FRAMES or changed != FRAMES:
        raise ValueError(f"오버레이 검사 실패: {stream}, {count}장, 자막 영역 변화 {changed}장")
    return count


def verify(url, output, cpu_check=False, input_video=None):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "synthetic.mp4"
    if input_video is None:
        generate_video(source)
    else:
        shutil.copyfile(input_video, source)
    input_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    start = perf_counter()
    with source.open("rb") as fp:
        response = requests.post(url + "/analyses", files={"video": (source.name, fp, "video/mp4")},
                                 data={"near_side": "left", "facing": "left"},
                                 timeout=600)
    if response.status_code != 201:
        raise RuntimeError(f"POST /analyses 실패 ({response.status_code}): {response.text[:500]}")
    seconds = perf_counter() - start
    info = response.json()
    saved = requests.get(url + info["summary_url"], timeout=30)
    saved.raise_for_status()
    if saved.json() != info or info["frame_count"] != FRAMES or info["plate_mode"] != "cutmix":
        raise ValueError("저장된 분석 요약을 확인하세요.")
    engines = info.get("inference_backends", {})
    if not cpu_check and engines != dict.fromkeys(("baseline", "cutmix", "pose"), "TensorRTBackend"):
        raise ValueError(f"기본 세 모델의 TensorRT 실행 결과가 예상과 다릅니다: {engines}")
    for name in ("overlay.mp4", "frames.csv"):
        artifact = requests.get(url + info["files"][name], timeout=60)
        artifact.raise_for_status()
        (output / name).write_bytes(artifact.content)
    with (output / "frames.csv").open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if ([int(row["frame_index"]) for row in rows] != list(range(FRAMES))
            or not {"plate_status", "selected_model", "trunk_angle_deg"}.issubset(rows[0])):
        raise ValueError("프레임별 검출·자세 CSV를 확인하세요.")
    count = read_overlay(source, output / "overlay.mp4")
    report = dict(check="cpu-api" if cpu_check else "tensorrt-api", analysis_id=info["analysis_id"],
                  input_sha256=input_sha256, http_status=response.status_code, http_seconds=round(seconds, 3), backends=engines,
                  overlay_frames=count, csv_rows=len(rows), selected_models=info["selected_model_counts"])
    (output / "response.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="이번 실행을 저장할 새 폴더")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--cpu-check", action="store_true", help="Mac에서 HTTP와 출력 파일만 비교할 때 사용")
    parser.add_argument("--input", type=Path, help="기존 검증에 사용한 합성 MP4 원본")
    args = parser.parse_args()
    verify(args.url.rstrip("/"), args.output, args.cpu_check, args.input)
