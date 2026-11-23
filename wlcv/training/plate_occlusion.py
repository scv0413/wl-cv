"""원판 위치를 이용해 학습 이미지에만 부분 가림을 추가한다."""

import json
import math
from functools import partial
from pathlib import Path
from random import Random

import yaml
from ultralytics import YOLO, __version__
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect import DetectionTrainer

from wlcv.config import DEVICE, WEIGHTS
from wlcv.training.background_textures import resize_background, sample_training_backgrounds
from wlcv.training.train import train_baseline

PROBABILITY = 0.30
AREA_RANGE = (0.25, 0.40)
MASK_VALUE = 114


def apply_plate_occlusion(images, image_indices, boxes, rng, *, probability=PROBABILITY, make_fill=None):
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
        fill = MASK_VALUE / 255
        if make_fill is not None:
            fill = make_fill(mask_width, mask_height).to(images.device)
        images[image_index, :, mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = fill


class PlateOcclusionTrainer(DetectionTrainer):
    """Ultralytics의 기본 증강을 거친 학습 배치에 가림을 추가한다."""

    fill_kind = "gray"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occlusion_rng = Random(self.args.seed)
        spec = {"probability_per_plate": PROBABILITY, "inner_box_area_range": AREA_RANGE,
                "fill": self.fill_kind, "seed": self.args.seed}
        self.make_fill = None
        if self.fill_kind == "training_background":
            patches = sample_training_backgrounds(Path(self.args.data).parent)
            # 질감을 고르는 난수는 마스크 위치·면적 난수와 별도로 사용한다.
            texture_rng = Random(self.args.seed + 1)
            self.make_fill = partial(resize_background, patches, texture_rng)
            spec.update(texture_count=len(patches), texture_seed=self.args.seed + 1)
        else:
            spec["gray_value_0_to_255"] = MASK_VALUE
        (self.save_dir / "occlusion.json").write_text(json.dumps(spec, indent=2) + "\n")

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        indices = batch["batch_idx"].detach().cpu().long().tolist()
        boxes = batch["bboxes"].detach().cpu().tolist()
        apply_plate_occlusion(batch["img"], indices, boxes, self.occlusion_rng,
                              make_fill=self.make_fill)
        return batch


class PlateTextureTrainer(PlateOcclusionTrainer):
    """회색 대신 학습 프레임에서 잘라낸 배경으로 마스크를 채운다."""

    fill_kind = "training_background"


def train_occlusion(data_yaml, baseline_dir, *, gray_reference=None):
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
    if gray_reference is None:
        return train_baseline(data_yaml, name="plate_occlusion", trainer=PlateOcclusionTrainer)

    gray_args = yaml.safe_load((gray_reference / "args.yaml").read_text())
    experiment_paths = {"name", "save_dir"}
    if any(baseline.get(key) != gray_args.get(key) for key in baseline.keys() | gray_args.keys()
           if key not in experiment_paths):
        raise ValueError("회색 가림 실험과 베이스라인의 학습 설정이 다릅니다.")
    gray_spec = json.loads((gray_reference / "occlusion.json").read_text())
    settings = (gray_spec["probability_per_plate"], tuple(gray_spec["inner_box_area_range"]),
                gray_spec["seed"], gray_spec.get("gray_value_0_to_255"))
    if settings != (PROBABILITY, AREA_RANGE, baseline["seed"], MASK_VALUE):
        raise ValueError("기존 회색 가림 실험의 확률·면적·시드를 확인하세요.")
    return train_baseline(data_yaml, name="plate_occlusion_texture", trainer=PlateTextureTrainer)
