"""원판 위치를 이용해 학습 이미지에만 부분 가림을 추가한다."""

import json
import math
from pathlib import Path
from random import Random

import yaml
from ultralytics import YOLO, __version__
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect import DetectionTrainer

from wlcv.config import DEVICE, WEIGHTS
from wlcv.training.train import train_baseline

PROBABILITY = 0.30
AREA_RANGE = (0.25, 0.40)
MASK_VALUE = 114


def apply_plate_occlusion(images, image_indices, boxes, rng, *, probability=PROBABILITY):
    """[0, 1] 이미지 배치에서 정규화된 원판 박스 안쪽만 가린다."""
    height, width = images.shape[-2:]
    for image_index, (x, y, box_width, box_height) in zip(image_indices, boxes):
        if rng.random() >= probability:
            continue
        # 박스 경계에 보이는 원판을 남겨 원래 좌표를 그대로 사용한다.
        left = max(0, math.ceil((x - box_width / 2) * width) + 1)
        right = min(width, math.floor((x + box_width / 2) * width) - 1)
        top = max(0, math.ceil((y - box_height / 2) * height) + 1)
        bottom = min(height, math.floor((y + box_height / 2) * height) - 1)
        scale = math.sqrt(rng.uniform(*AREA_RANGE))
        mask_width = round((right - left) * scale)
        mask_height = round((bottom - top) * scale)
        if mask_width < 1 or mask_height < 1:
            continue
        mask_x = rng.randint(left, right - mask_width)
        mask_y = rng.randint(top, bottom - mask_height)
        images[image_index, :, mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = MASK_VALUE / 255


class PlateOcclusionTrainer(DetectionTrainer):
    """Ultralytics의 기본 증강을 거친 학습 배치에 가림을 추가한다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occlusion_rng = Random(self.args.seed)
        spec = {"probability_per_plate": PROBABILITY, "inner_box_area_range": AREA_RANGE,
                "gray_value_0_to_255": MASK_VALUE, "seed": self.args.seed}
        (self.save_dir / "occlusion.json").write_text(json.dumps(spec, indent=2) + "\n")

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        indices = batch["batch_idx"].detach().cpu().long().tolist()
        boxes = batch["bboxes"].detach().cpu().tolist()
        apply_plate_occlusion(batch["img"], indices, boxes, self.occlusion_rng)
        return batch


def train_occlusion(data_yaml, baseline_dir):
    """기존 Mac 베이스라인과 조건을 대조한 뒤 가림 적용 모델을 학습한다."""
    baseline = yaml.safe_load((baseline_dir / "args.yaml").read_text())
    control = YOLO(str(baseline_dir / "weights" / "best.pt"))
    if control.ckpt.get("version") != __version__:
        raise ValueError("베이스라인과 같은 Ultralytics 버전에서 실행하세요.")
    if Path(baseline["data"]).resolve() != data_yaml.resolve():
        raise ValueError("베이스라인과 같은 데이터셋을 지정하세요.")
    if Path(baseline["model"]).resolve() != WEIGHTS.resolve():
        raise ValueError("베이스라인과 같은 사전학습 가중치를 지정하세요.")
    if str(baseline["device"]) != str(DEVICE):
        raise ValueError("이 비교는 베이스라인과 같은 장치에서 실행하세요.")

    # args.yaml의 기본 학습 설정과 설치된 라이브러리를 비교한다.
    paths_and_device = {"model", "data", "device", "project", "name"}
    changed = [
        key for key, default in DEFAULT_CFG_DICT.items()
        if key not in paths_and_device and baseline.get(key) != default
    ]
    if changed:
        raise ValueError(f"베이스라인과 다른 학습 설정을 확인하세요: {', '.join(changed)}")
    return train_baseline(data_yaml, name="plate_occlusion", trainer=PlateOcclusionTrainer)
