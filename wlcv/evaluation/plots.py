from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from wlcv.data.dataset import read_boxes


def predict_rgb(model, frame, *, conf=0.25, imgsz=640, device="cpu"):
    """RGB 프레임을 Ultralytics의 BGR 배열 입력으로 변환한다."""
    return model.predict(frame[:, :, ::-1].copy(), imgsz=imgsz, conf=conf,
                         device=device, verbose=False)[0]


def plot_comparison(frame, label_file, prediction, title):
    """같은 프레임에 표시한 정답과 모델 예측을 비교한다."""
    _, (truth, predicted) = plt.subplots(1, 2, figsize=(8, 6), layout="constrained")
    truth.imshow(frame)
    height, width = frame.shape[:2]
    for cx, cy, w, h in read_boxes(label_file):
        corner = ((cx - w / 2) * width, (cy - h / 2) * height)
        truth.add_patch(Rectangle(corner, w * width, h * height, fill=False,
                                  edgecolor="lime", linewidth=2))
    truth.set_title("Ground truth")
    predicted.imshow(prediction.plot()[:, :, ::-1])
    predicted.set_title(title)
    for axis in (truth, predicted):
        axis.axis("off")
    plt.show()


def plot_training(run_dir: Path):
    """results.csv의 에폭별 검출 손실 합계와 검증 지표를 그린다."""
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = results.columns.str.strip()
    _, (loss, score) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for split in ("train", "val"):
        columns = results.filter(regex=rf"^{split}/.+_loss$")
        if columns.empty:
            raise ValueError(f"{split} 손실 열이 없습니다.")
        loss.plot(results["epoch"], columns.sum(axis=1, min_count=1), label=split)
    for key in ("mAP50", "mAP50-95", "precision", "recall"):
        score.plot(results["epoch"], results[f"metrics/{key}(B)"], label=key)
    loss.set_ylabel("Detection loss (sum)")
    score.set_ylabel("Validation score")
    score.set_ylim(-0.02, 1.02)
    for axis in (loss, score):
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    plt.show()
