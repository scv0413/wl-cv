"""같은 GPU에서 학습한 100/100과 200/30을 Mac CPU에서 짝지어 채점한다."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from wlcv.config import DATASET, ROOT
from wlcv.evaluation.schedule_review import evaluate_run
from wlcv.training.plate_cutmix import score_pair

CONTROL = ROOT / "outputs/plate_cutmix_3090/20261126/runs/control"


def compare(trial, output):
    output.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_load((CONTROL / "args.yaml").read_text())
    new = yaml.safe_load((trial / "args.yaml").read_text())
    differences = {k for k in original.keys() | new.keys() if original.get(k) != new.get(k)}
    if differences - {"epochs", "patience", "name", "save_dir"} or (new["epochs"], new["patience"]) != (200, 30):
        raise ValueError(f"대조군과 학습 일정 밖의 설정을 확인하세요: {differences}")
    frames, metrics = [], []
    for run in (CONTROL, trial):
        metric, observations = evaluate_run(run, DATASET / "data.yaml", output)
        # 빈 정답이 섞인 열에도 불리언 반전(~)을 안전하게 적용한다.
        observations["match"] = observations["match"].astype("boolean")
        frames.append(observations)
        metrics.append(metric)
    pair, scores = score_pair(*frames)
    if len(pair) != 540 or pair.video_id.nunique() != 6:
        raise ValueError("검증 선수 6명·동작 중 270장의 두 추론 방식을 확인하세요.")
    pair.to_csv(output / "frames.csv", index=False)
    scores.to_csv(output / "by_video.csv", index=False)
    summary = []
    for method, group in pair.groupby("method"):
        common = group[group.visibility.eq("visible") & group.manual_y.notna()
                       & group.match_base & group.match_trial]
        motion = scores[(scores["방법"] == method) & (scores["평가 구간"] == "동작 중 전체")].iloc[0]
        a03 = scores[(scores["방법"] == method) & (scores["평가 구간"] == "a03_sn_f_01")].iloc[0]
        for index, (tag, prefix) in enumerate([("base", "기존"), ("trial", "비교")]):
            summary.append(dict(method=method, schedule=["100/100", "200/30"][index],
                                map50_95=metrics[index], motion_misses=int(motion[prefix+" 미탐"]),
                                a03_longest=int(a03[prefix+" 최장 연속 미탐"]), common_hubs=len(common),
                                hub_y_mae_px=common["dy_px_"+tag].abs().mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.csv", index=False)
    linked = summary[summary.method.eq("Past link")].set_index("schedule")
    criteria = ["motion_misses", "a03_longest", "hub_y_mae_px"]
    delta = linked.loc["200/30", criteria].astype(float) - linked.loc["100/100", criteria].astype(float)
    if not (linked.common_hubs > 0).all():
        raise ValueError("두 모델 모두 정탐인 직접 허브 판독점이 없습니다.")
    accepted = bool(delta.motion_misses <= 0 and delta.a03_longest <= 0 and delta.hub_y_mae_px <= 1)
    decision = dict(accept_3090_schedule=accepted, motion_misses_delta=int(delta.motion_misses),
                    a03_longest_delta=int(delta.a03_longest), hub_y_mae_delta_px=float(delta.hub_y_mae_px))
    (output / "decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    return summary, decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary, decision = compare(args.trial, args.output)
    print(summary.to_string(index=False))
    print(decision)
