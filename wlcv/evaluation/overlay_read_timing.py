"""같은 영상의 세 분석 명령을 교대로 실행하고 완성 파일을 대조한다."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import pandas as pd
from tqdm.auto import tqdm

from wlcv.config import ROOT

MODES = ("two-pass", "single-pass", "single-shared")
ORDERS = (MODES, (MODES[2], MODES[0], MODES[1]), (MODES[1], MODES[2], MODES[0]))


def timed_runs(folder):
    """각 방식이 블록의 첫째·둘째·셋째 자리에 한 번씩 오도록 한다."""
    rows = []
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    with tqdm(total=9, desc="새 프로세스에서 영상 완성", unit="편") as progress:
        for block, order in enumerate(ORDERS, 1):
            for mode in order:
                command = [sys.executable, "-m", "wlcv.analyze", "c04_sn_f_01",
                           "--device", "mps", "--read-mode", mode]
                progress.set_postfix_str(f"{block} / {mode}")
                start = perf_counter()
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
                elapsed = perf_counter() - start
                if result.returncode:
                    raise RuntimeError(result.stderr[-4000:])
                run = Path(next(line.removeprefix("로컬 결과: ") for line in result.stdout.splitlines()
                                if line.startswith("로컬 결과: ")))
                timing = json.loads((run / "timing.json").read_text())
                rows.append(dict(block=block, mode=mode, run=str(run), command_s=elapsed,
                                 pipeline_s=timing["pipeline_seconds"]))
                pd.DataFrame(rows).to_csv(folder / "runs.csv", index=False)
                progress.update()
    return pd.DataFrame(rows)


def decoded_frames(path):
    """파일 포맷과 재생 정보를 확인하고 압축을 푼 픽셀을 프레임별로 비교한다."""
    capture = cv2.VideoCapture(str(path))
    hashes = []
    try:
        if not capture.isOpened():
            raise OSError(f"완성 영상을 다시 열 수 없습니다: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape != (1920, 1080, 3):
                raise ValueError("오버레이 해상도가 다릅니다.")
            hashes.append(sha256(image).digest())
    finally:
        capture.release()
    if len(hashes) != 116 or abs(fps - 30) > .01 or codec.lower() not in ("avc1", "h264"):
        raise ValueError(f"완성 영상의 장수·재생 속도·코덱을 확인하세요: {len(hashes)}, {fps}, {codec}")
    return hashes


def verify_runs(runs, folder):
    """시간 측정을 모두 마친 후 첫 두 번 읽기 결과와 프레임별로 대조한다."""
    reference = Path(runs.iloc[0].run)
    reference_frames = pd.read_csv(reference / "frames.csv")
    reference_summary = json.loads((reference / "summary.json").read_text())
    reference_pixels = decoded_frames(reference / "overlay.mp4")
    keys = ["video_id", "frame_index", "source_time_ms", "manual_phase", "plate_status",
            "person_candidates", "pose_side", "joints_drawn"]
    metrics = {"position_px": ["plate_cx_px", "plate_cy_px", "shoulder_x_px", "shoulder_y_px"],
               "score": ["plate_confidence"], "angle_deg": ["trunk_angle_deg"]}
    results = []
    for row in tqdm(runs.itertuples(index=False), total=len(runs), desc="완성 파일 검수"):
        path = Path(row.run)
        data = pd.read_csv(path / "frames.csv")
        if list(data.columns) != list(reference_frames.columns) or data.frame_index.tolist() != list(range(116)):
            raise ValueError("프레임 번호 또는 CSV 열 구성이 바뀌었습니다.")
        summary = json.loads((path / "summary.json").read_text())
        if summary["device"] != "mps" or summary["read_mode"] != row.mode:
            raise ValueError("실행 장치 또는 읽기 방식을 확인하세요.")
        hashes = decoded_frames(path / "overlay.mp4")
        result = dict(block=row.block, mode=row.mode, equal_labels=data[keys].equals(reference_frames[keys]),
                      equal_missing=data.isna().equals(reference_frames.isna()),
                      pixel_mismatch_frames=sum(a != b for a,b in zip(reference_pixels, hashes, strict=True)),
                      direct=summary["plate_counts"]["직접 검출"],
                      rise_px=summary["manual_phase_interval"]["rise_px"], change_deg=summary["trunk_change_deg"])
        for name, columns in metrics.items():
            error = (data[columns] - reference_frames[columns]).abs()
            result[name] = error.max().max()
        result["same_summary"] = all(summary[k] == reference_summary[k] for k in
                                     ("plate_counts", "manual_phase_interval", "trunk_change_deg", "coaching"))
        results.append(result)
    checks = pd.DataFrame(results)
    checks.to_csv(folder / "checks.csv", index=False)
    return checks
