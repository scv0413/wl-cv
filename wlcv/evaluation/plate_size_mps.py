"""Mac에서 두 영상의 오버레이를 완성하고 원판 입력 크기만 비교한다."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm.auto import tqdm

from wlcv.config import ROOT

C04 = "c04_sn_f_01"
B02 = "b02_sn_h_02"
PARENT = ROOT / "outputs/plate_size_mps"


def benchmark():
    """각 영상에서 640을 먼저 두 번, 512를 먼저 두 번 실행한다."""
    folder = PARENT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True)
    rows = []
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    with tqdm(total=16, desc="MPS 오버레이 저장", unit="편") as progress:
        for block in range(4):
            videos = (C04, B02) if block % 2 == 0 else (B02, C04)
            for video in videos:
                sizes = (640, 512) if (block in (0, 3)) == (video == C04) else (512, 640)
                for size in sizes:
                    command = [sys.executable, "-m", "wlcv.analyze", video, "--device", "mps",
                               "--read-mode", "single-pass", "--plate-imgsz", str(size)]
                    start = perf_counter()
                    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
                    elapsed = perf_counter() - start
                    if result.returncode:
                        raise RuntimeError(result.stderr[-4000:])
                    path = next(line.removeprefix("로컬 결과: ") for line in result.stdout.splitlines()
                                if line.startswith("로컬 결과: "))
                    run = Path(path).relative_to(ROOT)
                    timing = json.loads((ROOT / run / "timing.json").read_text())
                    rows.append(dict(block=block+1, video_id=video, size=size, run=str(run),
                                     command_s=elapsed, pipeline_s=timing["pipeline_seconds"]))
                    pd.DataFrame(rows).to_csv(folder / "runs.csv", index=False)
                    progress.update()
    return folder


if __name__ == "__main__":
    print(benchmark())
