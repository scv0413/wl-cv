"""원판 탐지 증강을 눈으로 확인하고 같은 원본 검증 프레임에서 모델을 비교한다."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from wlcv.data.plate_labels import label_frame, read_boxes
from wlcv.evaluation.occlusion_dose import score_frame


def preview_augmentation(path):
    """같은 사진의 중앙 확대·축소와 HSV 밝기 변화만 따로 보여 준다."""
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    fig, axes = plt.subplots(2, 3, figsize=(8, 6), layout="constrained")
    for axis, factor in zip(axes[0], [0.5, 1.0, 1.5]):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, factor)
        zoom = cv2.warpAffine(rgb, matrix, (w, h), borderValue=(114, 114, 114))
        axis.imshow(zoom)
        axis.set_title(f"Zoom ×{factor:.1f}")
    for axis, factor in zip(axes[1], [0.6, 1.0, 1.4]):
        example = hsv.copy()
        example[:, :, 2] = (hsv[:, :, 2] * factor).clip(0, 255).astype(np.uint8)
        axis.imshow(cv2.cvtColor(example, cv2.COLOR_HSV2RGB))
        axis.set_title(f"Brightness ×{factor:.1f}")
    for axis in axes.flat:
        axis.axis("off")
    return fig


def evaluate_plates(model, dataset):
    """고정 검증셋의 가까운 원판 한 개를 평가한다. 원판이 있는 라벨만 대상이다."""
    rows = []
    for path in tqdm(sorted((dataset / "images/val").glob("*.jpg")), desc="원판 검증", unit="장"):
        targets = read_boxes(dataset / "labels/val" / f"{path.stem}.txt")
        if not targets:
            continue
        (cx, cy, bw, bh), = targets
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        truth = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
        confidence, iou = score_frame(model, rgb, truth)
        hit = confidence >= 0.10
        x1, y1, x2, y2 = np.rint(truth).astype(int)
        value = rgb.max(axis=2)
        box = value[y1:y2, x1:x2]
        # 박스 너비의 10%만큼 바깥으로 띠를 만들어 원판 주변 밝기를 측정한다.
        pad = max(1, round((x2 - x1) * 0.10))
        x0, y0 = max(0, x1-pad), max(0, y1-pad)
        nearby = value[y0:y2+pad, x0:x2+pad]
        ring = np.ones(nearby.shape, dtype=bool)
        ring[y1-y0:y2-y0, x1-x0:x2-x0] = False
        contrast = abs(np.median(box) - np.median(nearby[ring])) if ring.any() else np.nan
        video, index = label_frame(path)
        rows.append(dict(video=video, frame=index, box_width_640=bw*w*640/max(h,w),
                         scene_v=np.median(value), box_background_delta_v=contrast, confidence=confidence,
                         hit=hit, missed=not hit, hit_iou=iou if hit else np.nan))
    return pd.DataFrame(rows)
