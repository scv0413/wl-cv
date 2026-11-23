"""학습 프레임에서 원판 라벨과 겹치지 않는 질감을 가져온다."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

from wlcv.data.plate_labels import read_boxes


def sample_training_backgrounds(dataset_dir):
    """학습 JPG의 3×3 위치를 살펴보고 비어 있는 작은 영역을 모은다."""
    dataset_dir = Path(dataset_dir)
    patches = []
    for path in sorted((dataset_dir / "images" / "train").glob("*.jpg")):
        labels = dataset_dir / "labels" / "train" / f"{path.stem}.txt"
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes = [
            ((x - bw / 2) * width, (y - bh / 2) * height,
             (x + bw / 2) * width, (y + bh / 2) * height)
            for x, y, bw, bh in read_boxes(labels)
        ]
        size = min(width, height) // 4
        for left in (0, (width - size) // 2, width - size):
            for top in (0, (height - size) // 2, height - size):
                right, bottom = left + size, top + size
                overlaps = any(left < x2 and right > x1 and top < y2 and bottom > y1
                               for x1, y1, x2, y2 in boxes)
                if overlaps:
                    continue
                crop = image.crop((left, top, right, bottom)).resize((64, 64), Image.Resampling.BILINEAR)
                pixels = np.array(crop)
                # 평평한 영역은 회색 채움과 비슷하므로 질감 후보에서 뺀다.
                if pixels.std(axis=(0, 1)).mean() >= 3:
                    patches.append(torch.from_numpy(pixels).permute(2, 0, 1))
    if not patches:
        raise ValueError("학습 이미지에서 사용할 배경 질감을 찾지 못했습니다.")
    return patches


def resize_background(patches, rng, width, height):
    """배경 한 조각을 가림 영역의 크기에 맞춘다."""
    patch = rng.choice(patches)[None].float() / 255
    return interpolate(patch, size=(height, width), mode="bilinear", align_corners=False)[0]
