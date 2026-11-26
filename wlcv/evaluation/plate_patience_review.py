"""세 가중치의 미탐, 좌표가 끊긴 길이, 무릎 통과→접촉 원판 상승량을 비교한다."""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT
from wlcv.evaluation.schedule_review import evaluate_motion
from wlcv.training.plate_cutmix import check_same_labels

CONTROL = ROOT / "outputs/plate_3090_schedule/20261126/runs/schedule200"
LABELS = ANNOTATIONS / "plate_blur_review/validation.csv"


def measure(detail):
    """수동 박스가 있을 때만 미탐을 세고, 좌표 누락은 동작 구간의 모든 프레임에서 센다."""
    motion = detail[detail.in_motion & detail.method.eq("Past link")].copy()
    if motion.match.count() != 270 or motion.video_id.nunique() != 6:
        raise ValueError("검증 6명의 동작 구간과 채점 가능한 수동 박스 270장을 확인하세요.")
    hubs = pd.read_csv(LABELS).set_index(["video_id", "frame_index"], verify_integrity=True)
    rows = []
    for video, group in motion.groupby("video_id"):
        frames = group.sort_values("frame_index").set_index("frame_index")
        if frames.index.tolist() != list(range(frames.index.min(), frames.index.max() + 1)):
            raise ValueError(f"{video}의 동작 중간 프레임이 빠졌습니다.")
        missing = frames.cx.isna() | frames.cy.isna()
        gap_lengths = missing.groupby((~missing).cumsum()).sum()
        gaps = frames.index[missing].tolist()
        events = load_events(ANNOTATIONS, video, pass_number=1)
        knee, contact = events["transition"], events["drive"]
        points = hubs.loc[video].loc[[knee, contact]]
        manual = float("nan")
        if points.hub_source.isin(["clicked", "previous_blind"]).all():
            manual = points.at[knee, "manual_y"] - points.at[contact, "manual_y"]
        rise = frames.at[knee, "cy"] - frames.at[contact, "cy"]
        rows.append(dict(video_id=video, motion_frames=len(frames), scored_boxes=frames.match.count(),
                         misses=int(frames.match.eq(False).sum()), gap_frames=len(gaps),
                         longest_gap=int(gap_lengths.max()), gap_indices=" ".join(map(str, gaps)),
                         knee=knee, contact=contact, manual_rise_px=manual, model_rise_px=rise,
                         rise_error_px=rise-manual, abs_rise_error_px=abs(rise-manual)))
    return pd.DataFrame(rows)


def compare(trial, output):
    before = yaml.safe_load((CONTROL / "args.yaml").read_text())
    after = yaml.safe_load((trial / "args.yaml").read_text())
    diff = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if diff - {"epochs", "patience", "name", "save_dir"} or (after["epochs"], after["patience"]) != (1000, 50):
        raise ValueError(f"학습 일정 밖의 설정을 확인하세요: {diff}")
    weights = {"200/30": CONTROL / "weights/best.pt", "1000/30": trial / "weights/best_patience30.pt",
               "1000/50": trial / "weights/best.pt"}
    output.mkdir(parents=True, exist_ok=True)
    people, observations = [], []
    reference = None
    for label, path in weights.items():
        frames = evaluate_motion(path, DATASET / "data.yaml")
        if reference is None:
            reference = frames
        else:
            check_same_labels(reference, frames)
        people.append(measure(frames).assign(schedule=label))
        observations.append(frames.assign(schedule=label))
    results = pd.concat(people, ignore_index=True)
    common = results.pivot(index="video_id", columns="schedule", values="abs_rise_error_px").dropna().index
    results["common_rise"] = results.video_id.isin(common)
    results["common_abs_error"] = results.abs_rise_error_px.where(results.common_rise)
    summary = results.groupby("schedule", sort=False).agg(
        scored_boxes=("scored_boxes","sum"), misses=("misses","sum"), motion_frames=("motion_frames","sum"),
        gap_frames=("gap_frames","sum"), longest_gap=("longest_gap","max"),
        measurable_rises=("abs_rise_error_px","count"), common_athletes=("common_abs_error","count"),
        common_rise_mae_px=("common_abs_error","mean"))
    pd.concat(observations, ignore_index=True).to_csv(output / "frames.csv", index=False)
    results.to_csv(output / "athletes.csv", index=False)
    summary.to_csv(output / "summary.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(compare(args.trial, args.output).round(3))
