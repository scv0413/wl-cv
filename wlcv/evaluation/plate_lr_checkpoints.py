"""같은 실행의 세 에폭과 기존 코사인 모델을 동일한 원본 프레임에서 비교한다."""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from wlcv.config import DATASET, ROOT
from wlcv.evaluation.plate_lr_review import save_false_positives
from wlcv.evaluation.plate_patience_review import measure
from wlcv.evaluation.schedule_review import evaluate_motion
from wlcv.training.plate_cutmix import check_same_labels
from wlcv.training.plate_lr_grid import CHECKPOINT_EPOCHS, FULL_RUN

CONTROL = ROOT / "outputs/plate_lr_grid/20261126/runs/lr0005_cosine"
PREVIOUS = ROOT / "outputs/plate_lr_0001/20261126/runs/lr0001_linear"


def compare(experiment):
    run = experiment / "runs" / FULL_RUN
    previous = yaml.safe_load((PREVIOUS / "args.yaml").read_text())
    args = yaml.safe_load((run / "args.yaml").read_text())
    changed = {k for k in previous.keys() | args.keys() if previous.get(k) != args.get(k)}
    if changed - {"name", "save_dir", "patience"} or args["patience"] != 0:
        raise ValueError(f"이전 선형 실험과 달라진 설정을 확인하세요: {changed}")
    weights = {"cosine_0005": CONTROL / "weights/best.pt"}
    for epoch in CHECKPOINT_EPOCHS:
        weights[f"epoch{epoch}"] = run / "weights" / f"epoch{epoch}.pt"

    review = experiment / "review"
    review.mkdir(parents=True, exist_ok=True)
    reference = None
    people, frames, gaps = [], [], []
    for name, checkpoint in weights.items():
        detail = evaluate_motion(checkpoint, DATASET / "data.yaml")
        if reference is None:
            reference = detail
        else:
            check_same_labels(reference, detail)
        selected = detail[detail.method.eq("Past link")].copy()
        frames.append(selected.assign(checkpoint=name))
        motion = selected[selected.in_motion].copy()
        # 수동 박스와 매칭되지 않은 출력 박스만 오탐으로 센다.
        wrong = motion.x1.notna() & motion.match.eq(False).fillna(False)
        fp = wrong.groupby(motion.video_id).sum()
        athletes = measure(detail)
        athletes["false_positives"] = athletes.video_id.map(fp)
        people.append(athletes.assign(checkpoint=name))
        for video, group in motion.groupby("video_id"):
            missing = group.loc[group.cy.isna(), "frame_index"].sort_values()
            for _, streak in missing.groupby(missing.diff().ne(1).cumsum()):
                gaps.append((name, video, int(streak.min()), int(streak.max()), len(streak)))

    athletes = pd.concat(people, ignore_index=True)
    common = athletes.pivot(index="video_id", columns="checkpoint", values="abs_rise_error_px").dropna().index
    athletes["common_error"] = athletes.abs_rise_error_px.where(athletes.video_id.isin(common))
    summary = athletes.groupby("checkpoint", sort=False).agg(
        scored=("scored_boxes", "sum"), missed=("misses", "sum"), fp=("false_positives", "sum"),
        motion=("motion_frames", "sum"), missing=("gap_frames", "sum"), longest=("longest_gap", "max"),
        measured=("abs_rise_error_px", "count"), common=("common_error", "count"), rise_mae=("common_error", "mean"))
    all_frames = pd.concat(frames, ignore_index=True)
    save_false_positives(all_frames.rename(columns={"checkpoint": "case"}), review)
    all_frames.to_csv(review / "frames.csv", index=False)
    athletes.to_csv(review / "athletes.csv", index=False)
    pd.DataFrame(gaps, columns=["checkpoint", "video_id", "first", "last", "length"]).to_csv(
        review / "gaps.csv", index=False)
    summary.to_csv(review / "summary.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    torch.set_num_threads(2)
    print(compare(parser.parse_args().experiment.resolve()).round(3))
