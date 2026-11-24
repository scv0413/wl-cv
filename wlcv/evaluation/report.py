"""데이터 분할, 검출 결과, 학습 기록을 보고서 그림으로 정리한다."""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from wlcv.data.plate_labels import label_frame, read_boxes


def audit_dataset(dataset_dir, split_csv, run_dirs):
    """프레임별 메타데이터와 각 학습 실행의 검증 사람 중복을 확인한다."""
    split = pd.read_csv(split_csv).set_index("video_id", verify_integrity=True)
    rows = []
    for role in ("train", "val"):
        for path in sorted((dataset_dir / "images" / role).glob("*.jpg")):
            video_id, frame_index = label_frame(path)
            person = split.loc[video_id]
            expected_role = "validation" if role == "val" else role
            if person["role"] != expected_role:
                raise ValueError(f"{video_id}의 분할표와 프레임 폴더가 다릅니다.")
            boxes = read_boxes(dataset_dir / "labels" / role / f"{path.stem}.txt")
            with Image.open(path) as image:
                # 파일 메타데이터가 달라도 픽셀이 같으면 같은 해시가 나온다.
                pixels = np.asarray(image.convert("RGB"))
                digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                brightness = float(pixels.mean())
            rows.append((role, person["person_id"], video_id, frame_index,
                         [width * height * 100 for _, _, width, height in boxes],
                         brightness, digest, path.name))

    frames = pd.DataFrame(rows, columns=["split", "person_id", "video_id", "frame_index",
                                        "box_area_pct", "brightness", "hash", "frame"])
    train = frames[frames["split"] == "train"]
    val = frames[frames["split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("베이스라인 노트북에서 학습·검증 프레임을 먼저 준비하세요.")
    if set(train["person_id"]) & set(val["person_id"]):
        raise ValueError("학습·검증 프레임에 같은 사람이 포함되어 있습니다.")
    duplicates = sorted(set(train["hash"]) & set(val["hash"]))
    duplicate_frames = frames.loc[frames["hash"].isin(duplicates), ["split", "person_id", "frame"]]
    runs = []
    for run in run_dirs:
        people = set(pd.read_csv(run / "train_people.csv", dtype=str)["person_id"])
        runs.append({"run": run.name, "train_people": len(people),
                     "validation_overlap": ", ".join(sorted(people & set(val["person_id"]))) or "—"})
    return frames, pd.DataFrame(runs), duplicate_frames


def plot_data_coverage(frames):
    """사람별 표본 수와 표본별 원판 크기, 밝기를 비교한다."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    counts = frames.groupby(["person_id", "split"]).size().unstack(fill_value=0)
    counts.plot.barh(ax=axes[0], stacked=True, color=["#2563a6", "#d47729"])
    axes[0].legend(loc="lower right")
    axes[0].set(xlabel="Labeled frames", ylabel="Person", title="Frames per person")
    sizes = frames.explode("box_area_pct")["box_area_pct"].dropna().astype(float)
    edges = np.histogram_bin_edges(sizes, bins=10) if len(sizes) else 10
    for split, group in frames.groupby("split", sort=False):
        values = group.explode("box_area_pct")["box_area_pct"].dropna().astype(float)
        axes[1].hist(values, bins=edges, alpha=0.65, label=split)
        axes[2].hist(group["brightness"], bins=np.linspace(0, 255, 17), alpha=0.65, label=split)
    axes[1].set(title="Plate box sizes", xlabel="Area (% of frame)", ylabel="Boxes")
    axes[2].set(title="Mean frame brightness", xlabel="Mean RGB intensity (0–255)", ylabel="Frames")
    for axis in axes[1:]:
        axis.legend()
    return fig


def plot_training_report(run_dir):
    """손실 구성 요소, 검증 mAP, 실제 학습률 로그를 그린다."""
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    epoch = history["epoch"]
    losses = [name.split("/", 1)[1] for name in history.filter(regex=r"^train/.+_loss$")]
    fig, axes = plt.subplots(1, len(losses) + 2, figsize=(3.3 * (len(losses) + 2), 3.8),
                             layout="constrained")
    for axis, loss in zip(axes, losses):
        for split in ("train", "val"):
            axis.plot(epoch, history[f"{split}/{loss}"], label=split)
        axis.set_title(loss.replace("_", " "))
        axis.legend()
    for metric in ("mAP50", "mAP50-95"):
        axes[-2].plot(epoch, history[f"metrics/{metric}(B)"], label=metric)
    axes[-2].set(title="Validation detection", ylim=(0, 1.02))
    axes[-2].legend()
    for name in history.filter(regex=r"^lr/"):
        axes[-1].plot(epoch, history[name], label=name.split("/")[1])
    axes[-1].set(title="Learning rate", yscale="log")
    axes[-1].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    return fig


def collect_detections(review):
    """FiftyOne의 박스별 대응과 정답 박스 수를 프레임별로 저장한다."""
    ground_truth, predictions = [], []
    for sample in review:
        info = {"video_id": sample.video_id, "frame_index": sample.frame_index}
        width, height = sample.metadata.width, sample.metadata.height
        ground_truth.append({**info, "person_id": sample.person_id,
                             "targets": len(sample.ground_truth.detections)})
        targets = {d.id: d for d in sample.ground_truth.detections}
        for box in sample.predictions.detections:
            is_match = box["eval"] == "tp"
            center_error = np.nan
            overlap = np.nan
            if is_match:
                target = targets[box["eval_id"]]
                x, y, w, h = box.bounding_box
                tx, ty, tw, th = target.bounding_box
                center_error = np.hypot((x + w / 2 - tx - tw / 2) * width,
                                        (y + h / 2 - ty - th / 2) * height)
                overlap = box["eval_iou"]
            predictions.append({**info, "confidence": box.confidence, "matched": is_match,
                                "iou": overlap, "center_error_px": center_error})
    return pd.DataFrame(ground_truth), pd.DataFrame(
        predictions, columns=["video_id", "frame_index", "confidence", "matched", "iou", "center_error_px"]
    )


def compare_thresholds(frames, predictions, thresholds):
    """신뢰도 순으로 매칭된 단일 클래스 예측의 기준값을 바꿔 검출 건수를 센다."""
    target_count = int(frames["targets"].sum())
    rows = []
    for threshold in thresholds:
        accepted = predictions[predictions["confidence"] >= threshold]
        tp = int(accepted["matched"].sum())
        fp = len(accepted) - tp
        rows.append((threshold, tp, fp, target_count - tp,
                     tp / len(accepted) if len(accepted) else np.nan, tp / target_count))
    return pd.DataFrame(rows, columns=["confidence", "TP", "FP", "FN", "precision", "recall"])


def plot_frame_errors(frames, predictions, video_id, threshold):
    """한 영상의 검증 프레임에서 후보 점수와 매칭 박스의 중심 오차를 표시한다."""
    valid = frames[frames["video_id"] == video_id].sort_values("frame_index")
    if valid.empty:
        raise ValueError(f"검증 데이터에 없는 영상입니다: {video_id}")
    boxes = predictions[predictions["video_id"] == video_id]
    is_match = boxes["matched"].astype(bool)
    matched = boxes[is_match]
    unmatched = boxes[~is_match]
    accepted = matched[matched["confidence"] >= threshold]
    matched_counts = accepted.groupby("frame_index").size()
    matches_per_frame = valid["frame_index"].map(matched_counts).fillna(0)
    missed = valid[matches_per_frame < valid["targets"]]
    fig, (score, error) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    score.scatter(matched["frame_index"], matched["confidence"], color="#2563a6", label="Matched candidate")
    score.scatter(unmatched["frame_index"], unmatched["confidence"], color="#d47729",
                  marker="x", label="Unmatched candidate")
    score.axhline(threshold, color="gray", linestyle="--", label=f"Threshold {threshold:.2f}")
    error.scatter(accepted["frame_index"], accepted["center_error_px"], label="Matched plate")
    for index, frame in enumerate(missed["frame_index"]):
        error.axvline(frame, color="#b74949", alpha=0.55, linestyle=":",
                      label="Miss at threshold" if index == 0 else None)
    score.set(title=f"{video_id} · labeled validation frames", ylabel="Confidence", ylim=(-0.02, 1.02))
    error.set(xlabel="Frame index (0-based)", ylabel="Center error (source px)")
    error.set_xticks(valid["frame_index"])
    for axis in (score, error):
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    return fig
