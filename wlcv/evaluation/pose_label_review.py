"""저장된 YOLO 라벨을 원본 픽셀로 되돌려 박스와 직접점을 대조한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from wlcv.data.frames import read_rgb
from wlcv.evaluation.pose import JOINTS, SIDES


def review_pose_labels(data_yaml, frames):
    """YOLO TXT에서 복원한 좌표만 그린다. 표의 오차는 원래 CSV와의 차이다."""
    dataset = data_yaml.parent
    fig, axes = plt.subplots(2, len(frames), figsize=(10, 8), squeeze=False, layout="constrained")
    checks = []
    for column, sample in enumerate(frames):
        rgb = read_rgb(sample.video, sample.frame)
        height, width = rgb.shape[:2]
        with Image.open(dataset / "images" / sample.split / f"{sample.stem}.png") as image:
            if not np.array_equal(rgb, np.asarray(image)):
                raise ValueError(f"{sample.stem}의 학습 이미지와 원본 프레임이 다릅니다.")

        label = dataset / "labels" / sample.split / f"{sample.stem}.txt"
        values = np.loadtxt(label)
        if values.shape != (56,) or values[0] != 0:
            raise ValueError(f"{label.name}에 수행자 한 명과 COCO 17점이 필요합니다.")
        cx, cy, w, h = values[1:5] * [width, height, width, height]
        box = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        keypoints = values[5:].reshape(17, 3)
        side = SIDES.index(sample.side)
        indices = [JOINTS["무릎"][side], JOINTS["발목"][side]]
        expected_visibility = np.zeros(17)
        expected_visibility[indices] = 2
        missing = expected_visibility == 0
        if not np.array_equal(keypoints[:, 2], expected_visibility) or np.any(keypoints[missing, :2] != 0):
            raise ValueError(f"{sample.stem}의 좌우 또는 미주석 표시가 다릅니다.")
        knee, ankle = keypoints[indices, :2] * [width, height]

        # TXT 반올림을 감안해 원본 픽셀의 1/100까지 허용한다.
        errors = [np.max(np.abs(box - sample.box)), np.max(np.abs(knee - sample.points["knee"])),
                  np.max(np.abs(ankle - sample.points["ankle"]))]
        if not np.all(np.asarray(errors) <= 0.01):
            raise ValueError(f"{sample.stem}의 복원 좌표가 CSV와 다릅니다. 오차(px) {errors}")
        checks.append([sample.video.stem, sample.frame, sample.split, sample.side, *errors])

        left, top, right, bottom = box
        full, zoom = axes[:, column]
        for ax in (full, zoom):
            ax.imshow(rgb)
            ax.add_patch(Rectangle((left, top), w, h, fill=False, edgecolor="limegreen", linewidth=2))
            ax.plot(*knee, marker="+", color="cyan", markersize=11, markeredgewidth=2.2)
            ax.plot(*ankle, marker="D", color="gold", markersize=6, markeredgecolor="black")
            ax.axis("off")
        full.set_title(f"{sample.video.stem} · {sample.frame}\n{sample.split} | {sample.side}")
        pad = h * 0.12
        zoom.set_xlim(max(0, left - pad), min(width, right + pad))
        zoom.set_ylim(min(height, bottom + pad), max(0, top - pad))
    fig.suptitle("Decoded YOLO labels  |  green box: performer    cyan +: knee    gold diamond: ankle")
    columns = ["영상", "프레임", "분할", "가까운 쪽", "박스 오차(px)", "무릎 오차(px)", "발목 오차(px)"]
    return fig, pd.DataFrame(checks, columns=columns)
