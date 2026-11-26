"""프로젝트 경로와 학습 장치. 실행 환경은 .env.example을 참고한다."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def local_path(variable, default):
    path = Path(os.environ.get(variable) or default).expanduser()
    return (ROOT / path).resolve()


def training_device():
    requested = os.environ.get("WL_TRAIN_DEVICE")
    if requested:
        return requested

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


VIDEOS = local_path("WL_VIDEO_DIR", "data/raw/videos")
ANNOTATIONS = ROOT / "data/annotations"
SPLIT = ROOT / "data/splits/person_split.csv"
DATASET = local_path("WL_DATASET_DIR", "outputs/dataset")
RUNS = local_path("WL_RUNS_DIR", "outputs/runs")
WEIGHTS = local_path("WL_WEIGHTS", "models/yolo26n.pt")
CUTMIX_WEIGHTS = ROOT / "outputs/plate_cutmix_3090/20261126/runs/cutmix/weights/best.pt"
COSINE_WEIGHTS = local_path("WL_COSINE_WEIGHTS", "outputs/plate_lr_grid/20261126/runs/lr0005_cosine/weights/best.pt")
DEVICE = training_device()
