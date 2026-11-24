"""학습 일정 비교에 쓰는 로컬 시각화와 선택형 실험 도구."""

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm.auto import tqdm

from wlcv.annotation.hub_points import FRAMES, VIDEO
from wlcv.config import SPLIT, VIDEOS
from wlcv.data.frames import read_rgb


def curves(runs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for label, run in runs.items():
        history = pd.read_csv(run / "results.csv")
        for ax, col in zip(axes, ["train/box_loss", "val/box_loss", "metrics/mAP50-95(B)"]):
            ax.plot(history.epoch, history[col], label=label)
            ax.set(xlabel="Epoch", title=col)
    axes[0].legend()
    return fig


def hub_figure(pair):
    rows = pair[pair.video_id.eq(VIDEO) & pair.method.eq("Past link")].set_index("frame_index")
    fig, axes = plt.subplots(1, 3, figsize=(9, 5), layout="constrained")
    for index, ax in zip(FRAMES, axes):
        rgb = read_rgb(VIDEOS / f"{VIDEO}.mp4", index)
        ax.imshow(rgb)
        row = rows.loc[index]
        for tag, color, mark in [("100", "cyan", "+"), ("200", "orange", "x")]:
            x = (row[f"x1_{tag}"] + row[f"x2_{tag}"]) / 2
            y = (row[f"y1_{tag}"] + row[f"y2_{tag}"]) / 2
            if pd.notna(x):
                ax.plot(x, y, marker=mark, color=color, markersize=13, label=tag)
        if row.visibility == "visible":
            ax.plot(row.manual_x, row.manual_y, "o", markerfacecolor="none",
                    markeredgecolor="magenta", markersize=10, label="Manual hub")
        ax.set_title(f"{index} | manual: {row.visibility}")
        ax.axis("off")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right")
    return fig


def fiftyone_comparison(runs, data):
    """검증 JPEG 39장의 같은 정답 위에 두 best.pt의 예측을 올린다."""
    import fiftyone as fo
    from fiftyone import ViewField as F
    from fiftyone.utils.ultralytics import to_detections
    from ultralytics import YOLO
    from wlcv.evaluation.fiftyone_review import load_validation_frames

    review = load_validation_frames(data, SPLIT)
    for label, run in runs.items():
        model = YOLO(str(run / "weights/best.pt"))
        field, key = f"model_{label}", f"eval_{label}"
        for sample in tqdm(review.iter_samples(autosave=True), total=len(review), desc=field):
            result = model.predict(sample.filepath, conf=0.10, imgsz=640, device="cpu", verbose=False)[0]
            sample[field] = to_detections(result)
        review.evaluate_detections(field, gt_field="ground_truth", eval_key=key, method="coco", iou=0.5)
    review.save_view("Recovered", review.match(F("eval_200_tp") > F("eval_100_tp")))
    review.save_view("Lost", review.match(F("eval_200_tp") < F("eval_100_tp")))
    return fo.launch_app(review, address="127.0.0.1")


def log_scalar_history(runs, folder):
    """완료한 실행의 설정과 스칼라 곡선만 W&B 오프라인 로그에 저장한다."""
    import wandb

    for label, run in runs.items():
        settings = yaml.safe_load((run / "args.yaml").read_text())
        keys = ["epochs", "patience", "optimizer", "seed", "batch", "imgsz", "scale", "hsv_v", "mosaic", "close_mosaic"]
        with wandb.init(project="wl-cv", name=f"plate-{label}", mode="offline", dir=str(folder),
                        save_code=False, config={key: settings[key] for key in keys}) as log:
            for _, epoch in pd.read_csv(run / "results.csv").iterrows():
                log.log({k: float(epoch[k]) for k in epoch.index if k.startswith(("train/", "val/", "metrics/", "lr/"))},
                        step=int(epoch["epoch"]))
