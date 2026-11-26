"""각 조건을 같은 Mac CPU로 채점하고 한 설정만 다른 조건끼리 차이를 구한다."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from wlcv.config import ANNOTATIONS, DATASET
from wlcv.evaluation.full_video import summarize_phase_false_positives
from wlcv.evaluation.plate_patience_review import CONTROL, measure
from wlcv.evaluation.schedule_review import evaluate_run
from wlcv.training.plate_cutmix import check_same_labels
from wlcv.training.plate_lr_grid import CASES, LOW_LR_CASES


def save_false_positives(frames, review):
    """수동 박스와 매칭되지 않은 출력 박스를 동작·준비·종료 구간별로 센다."""
    tables = []
    for name, group in frames.groupby("case", sort=False):
        counts = summarize_phase_false_positives(group, ANNOTATIONS)
        tables.append(counts.assign(case=name))
    pd.concat(tables, ignore_index=True).to_csv(review / "false_positives.csv", index=False)


def compare(experiment, low_lr_experiment=None):
    """추가 실험이 있으면 이전 가중치도 재채점해 새 실험 폴더에만 기록한다."""
    review = (low_lr_experiment or experiment) / "review"
    review.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_load((CONTROL / "args.yaml").read_text())
    expected_changes = {"name", "save_dir", "optimizer", "momentum", "warmup_bias_lr", "lr0", "cos_lr"}
    runs = {"reference": CONTROL}
    runs.update({name: experiment / "runs" / name for name in CASES})
    settings = CASES.copy()
    if low_lr_experiment is not None:
        settings.update(LOW_LR_CASES)
        runs.update({name: low_lr_experiment / "runs" / name for name in LOW_LR_CASES})
    athletes, scores, selections = [], [], []
    reference_frames = None
    for name, run in runs.items():
        if name in settings:
            args = yaml.safe_load((run / "args.yaml").read_text())
            changes = {key for key in original.keys() | args.keys() if original.get(key) != args.get(key)}
            if changes - expected_changes or (args["lr0"], args["cos_lr"]) != settings[name]:
                raise ValueError(f"{name}의 고정 조건이 달라졌습니다: {changes}")
            audit = json.loads((run / "audit.json").read_text())
            if audit["initial_lr"] != args["lr0"] or audit["betas"] != [0.9, 0.999]:
                raise ValueError(f"{name}의 실제 옵티마이저 설정을 확인하세요.")
        metric, detail = evaluate_run(run, DATASET / "data.yaml", review)
        if reference_frames is None:
            reference_frames = detail
        else:
            check_same_labels(reference_frames, detail)
        athletes.append(measure(detail).assign(case=name))
        scores.append({"case": name, "map50_95": metric})
        selections.append(detail[detail.method.eq("Past link")].assign(case=name))

    people = pd.concat(athletes, ignore_index=True)
    shared = people.pivot(index="video_id", columns="case", values="abs_rise_error_px").dropna().index
    people["common_error"] = people.abs_rise_error_px.where(people.video_id.isin(shared))
    summary = people.groupby("case", sort=False).agg(
        motion_misses=("misses","sum"), scored_boxes=("scored_boxes","sum"),
        missing_centers=("gap_frames","sum"), motion_frames=("motion_frames","sum"),
        longest_gap=("longest_gap","max"), measured_rises=("abs_rise_error_px","count"),
        common_athletes=("common_error","count"), rise_mae_px=("common_error","mean"))
    summary = summary.join(pd.DataFrame(scores).set_index("case"))
    pairs = [("lr001_linear", "lr001_cosine"), ("lr0005_linear", "lr0005_cosine"),
             ("lr001_linear", "lr0005_linear"), ("lr001_cosine", "lr0005_cosine")]
    if low_lr_experiment is not None:
        pairs.extend([("lr0005_linear", "lr0001_linear"), ("lr0005_cosine", "lr0001_cosine"),
                      ("lr0001_linear", "lr0001_cosine")])
    contrasts = []
    for before, after in pairs:
        metrics = ["map50_95", "motion_misses", "missing_centers", "longest_gap", "rise_mae_px"]
        difference = summary.loc[after, metrics] - summary.loc[before, metrics]
        contrasts.append({"before": before, "after": after, **difference.to_dict()})
    people.to_csv(review / "athletes.csv", index=False)
    summary.to_csv(review / "summary.csv")
    frames = pd.concat(selections, ignore_index=True)
    frames.to_csv(review / "frames.csv", index=False)
    save_false_positives(frames, review)
    pd.DataFrame(contrasts).to_csv(review / "contrasts.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--low-lr-experiment", type=Path, help="추가 0.0001 실험 폴더")
    args = parser.parse_args()
    torch.set_num_threads(2)
    extra = args.low_lr_experiment.resolve() if args.low_lr_experiment else None
    print(compare(args.experiment.resolve(), extra).round(4))
