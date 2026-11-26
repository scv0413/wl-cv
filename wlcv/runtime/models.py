"""분석에 사용할 원판·자세 가중치 위치."""
import os
from dataclasses import dataclass
from pathlib import Path

from wlcv.config import COSINE_WEIGHTS, CUTMIX_WEIGHTS, ROOT, RUNS, local_path


@dataclass(frozen=True)
class ModelPaths:
    plate: Path
    pose: Path
    cutmix: Path
    cosine: Path


def experiment_models():
    """등록 영상 검증에서는 각 가중치 곁에 저장된 학습 선수 명단도 사용한다."""
    return ModelPaths(
        plate=RUNS / "plate_baseline/weights/best.pt",
        pose=ROOT / "models/yolo26n-pose.pt",
        cutmix=CUTMIX_WEIGHTS,
        cosine=COSINE_WEIGHTS,
    )


def file_models():
    """배포 경로를 지정하면 해당 폴더만 사용한다. 기존 설치의 경로도 지원한다."""
    if not os.environ.get("WL_MODEL_DIR"):
        return experiment_models()
    folder = local_path("WL_MODEL_DIR", "models") / "pytorch"
    return ModelPaths(
        plate=folder / "plate_baseline.pt",
        pose=folder / "yolo26n-pose.pt",
        cutmix=folder / "plate_cutmix.pt",
        cosine=folder / "plate_cosine.pt",
    )
