"""3090 대조군의 익명 입력 묶음에서 학습 일정만 바꾼다. 원본 yolo26n.pt로 시작한다."""
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO, settings

INPUT_SHA256 = "aaf184492fc91e32969b77e51a6bd2ec7f5b36908ca662a36ef8d53168c381c9"


def main():
    root = Path(__file__).resolve().parent
    if hashlib.sha256((root / "input.tar").read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("기존 3090 대조군과 같은 입력 묶음이 필요합니다.")
    if (torch.__version__ != "2.8.0+cu128" or ultralytics.__version__ != "8.4.202"
            or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090"
            or torch.cuda.get_device_capability(0) != (8, 6)):
        raise RuntimeError("대조군의 GPU 및 패키지 버전을 확인하세요.")
    settings.update({k: False for k in ("sync", "wandb", "clearml", "comet", "mlflow", "neptune",
                                        "raytune", "hub", "dvc", "tensorboard") if k in settings})
    args = json.loads((root / "settings.json").read_text())
    if (args["epochs"], args["patience"], args["cutmix"], args["resume"]) != (100, 100, 0, False):
        raise ValueError("기존 대조군 설정과 다릅니다.")
    args.update(epochs=200, patience=30)
    run = root / "runs/schedule200"
    if run.exists():
        raise FileExistsError(run)
    started = time.perf_counter()
    YOLO(str(root / "yolo26n.pt")).train(data=str(root / "dataset/data.yaml"), device=0,
                                         project=str(run.parent), name=run.name, **args)
    elapsed = time.perf_counter() - started
    with (run / "results.csv").open() as source:
        rows = list(csv.DictReader(source))
    epochs = [int(r["epoch"]) for r in rows]
    if epochs != list(range(1, len(rows) + 1)) or not 30 <= len(rows) <= 200:
        raise RuntimeError("학습 로그가 불완전합니다.")
    for name in ("best.pt", "last.pt"):
        if not (run / "weights" / name).stat().st_size:
            raise RuntimeError(name)
    print(f"학습 완료 {len(rows)}에폭, 학습 및 최종 검증 소요 {elapsed:.2f}초", flush=True)


if __name__ == "__main__":
    main()
