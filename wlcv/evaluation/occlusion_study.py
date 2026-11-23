"""가림 실험 모델을 동일한 검증 프레임에서 비교한다."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from wlcv.data.plate_labels import label_frame, read_boxes
from wlcv.evaluation.occlusion import plot_occlusion_comparison


def validation_occlusion_scores(model, dataset_dir, *, confidence=0.10):
    """정답 박스 중앙 50%를 가린 고정 조건과 원본에서 매칭 수를 비교한다."""
    images = sorted((dataset_dir / "images" / "val").glob("*.jpg"))
    if not images:
        raise ValueError("검증 프레임을 먼저 준비하세요.")
    rows = []
    for path in tqdm(images, desc="검증 프레임", unit="장"):
        video, index = label_frame(path)
        with Image.open(path) as image:
            frame = np.array(image.convert("RGB"))
        target, = read_boxes(dataset_dir / "labels" / "val" / f"{path.stem}.txt")
        height, width = frame.shape[:2]
        cx, cy, box_width, box_height = target
        mask = (round(box_width * width * math.sqrt(0.5)),
                round(box_height * height * math.sqrt(0.5)))
        figure, scores = plot_occlusion_comparison(
            model, frame, target, {"Plate 50%": (cx, cy)}, mask,
        )
        plt.close(figure)
        scores["video_id"] = video
        scores["frame_index"] = index
        rows.append(scores)

    scores = pd.concat(rows, ignore_index=True)
    scores["accepted"] = scores["confidence"].ge(confidence)
    summary = scores.groupby("condition", sort=False).agg(
        frames=("accepted", "size"),
        matched_at_threshold=("accepted", "sum"),
        median_iou=("iou", "median"),
    )
    return summary, scores
