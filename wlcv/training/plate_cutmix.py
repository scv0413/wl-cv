"""실제 학습 입력에서 CutMix의 적용 빈도를 세고 같은 베이스라인 설정으로 비교한다."""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import CutMix
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.metrics import bbox_ioa

from wlcv.config import RUNS
from wlcv.training.train import train_baseline

PROBABILITY = .30


def baseline_settings():
    cfg = yaml.safe_load((RUNS/"plate_baseline/args.yaml").read_text())
    if cfg["cutmix"] != 0:
        raise ValueError("대조군의 CutMix를 확인하세요.")
    cfg["cutmix"] = PROBABILITY
    for name in ("project", "name", "save_dir"):
        cfg.pop(name,None)
    return cfg


def preview(repeats=3):
    """학습 파이프라인에서 CutMix 직전까지 동일하게 처리한 입력으로 변환을 점검한다."""
    cfg = baseline_settings()
    data = yaml.safe_load(Path(cfg["data"]).read_text())
    data["names"] = {0:"plate"}
    params = get_cfg(overrides=cfg)
    ds = YOLODataset(img_path=str(Path(cfg["data"]).parent/"images/train"),imgsz=cfg["imgsz"],
                     augment=True,hyp=params,data=data,cache=False,batch_size=cfg["batch"],task="detect")
    random.seed(0)
    np.random.seed(0)
    rows=[]
    transforms=ds.transforms.transforms
    step=next(i for i,t in enumerate(transforms) if isinstance(t,CutMix))
    mixer=transforms[step]
    for i in tqdm(range(repeats*len(ds)),desc="학습 입력 확인",unit="장"):
        sample=ds.get_image_and_label(i%len(ds))
        for before in transforms[:step]:
            sample=before(sample)
        row=dict(source=Path(ds.im_files[i%len(ds)]).stem,attempted=False,applied=False,
                 original_overlap=False,donor_boxes=0,partial_donor=0)
        if random.uniform(0,1) <= mixer.p:
            row["attempted"]=True
            info=mixer.get_params(sample)
            if not info.get("skip"):
                area=info["area"][None]
                orig=sample["instances"].bboxes
                donor=sample["mix_labels"][0]["instances"].bboxes[info["indexes2"]]
                ratios=bbox_ioa(area,donor)[0]
                original_count=len(orig)
                new=mixer.apply_instances(mixer.apply_image(sample,info),info)
                assert len(new["instances"])==original_count+len(donor)
                row.update(applied=True,original_overlap=bool((bbox_ioa(area,orig)>0).any()),
                           donor_boxes=len(donor),partial_donor=int((ratios<.999).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def train():
    cfg=baseline_settings()
    data=Path(cfg.pop("data"))
    weights=Path(cfg.pop("model"))
    device=cfg.pop("device")
    run=train_baseline(data,weights=weights,device=device,name="plate_cutmix_030",settings=cfg)
    for filename in ("best.pt","last.pt"):
        (run/"weights"/filename).resolve(strict=True)
    return run


def longest_miss(group, column):
    """평가되지 않은 프레임을 건너뛰어 연속 미탐으로 세지 않는다."""
    longest = current = 0
    previous_video = previous_frame = None
    for video, frame, hit in group[["video_id", "frame_index", column]].itertuples(index=False, name=None):
        if video != previous_video or frame != previous_frame + 1 or hit:
            current = 0
        if not hit:
            current += 1
            longest = max(longest, current)
        previous_video, previous_frame = video, frame
    return longest


def score_pair(control, trial):
    """동일한 동작 구간·수동 박스에서 두 모델을 평가하고 구간별 최장 미탐을 계산한다."""
    from wlcv.evaluation.schedule_review import compare_runs

    keys = ["video_id", "frame_index", "method"]
    truth = ["in_motion", "visibility", "manual_x", "manual_y", "gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    first = control.set_index(keys).sort_index()
    second = trial.set_index(keys).sort_index()
    same_labels = first[truth].equals(second[truth])
    same_scored_frames = first["match"].notna().equals(second["match"].notna())
    if not (same_labels and same_scored_frames):
        raise ValueError("두 실행의 평가 구간, 채점 프레임 또는 수동 정답이 다릅니다.")
    pair = compare_runs(control, trial, labels=("base", "trial")).sort_values(keys)
    early = pair.video_id.eq("a03_sn_f_01") & pair.frame_index.between(2, 14)
    scopes = [("a03 2~14번", pair[early]), ("동작 중 전체", pair)]
    scopes.extend(pair.groupby("video_id", sort=True))
    rows = []
    for scope, subset in scopes:
        for method, group in subset.groupby("method", sort=True):
            rows.append({
                "평가 구간": scope, "방법": method, "평가 장수": len(group),
                "기존 미탐": int((~group.match_base).sum()),
                "비교 미탐": int((~group.match_trial).sum()),
                "복구": int(group.change.eq("복구").sum()), "손실": int(group.change.eq("손실").sum()),
                "기존 최장 연속 미탐": longest_miss(group, "match_base"),
                "비교 최장 연속 미탐": longest_miss(group, "match_trial"),
            })
    return pair, pd.DataFrame(rows)


def compare(trial, *, control=None, output=None):
    """같은 CPU 조건에서 검출 mAP와 원본 동작 중 연속 미탐·복구·손실을 보존한다."""
    from wlcv.evaluation.schedule_review import evaluate_run

    trial = Path(trial)
    control = RUNS / "plate_baseline" if control is None else Path(control)
    data = Path(baseline_settings()["data"])
    folder = trial / "paired_review" if output is None else Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    metrics, frames = [], {}
    for name, run in (("baseline", control), ("cutmix", trial)):
        metric, detail = evaluate_run(run, data, folder)
        detail.to_csv(folder / f"{name}_frames.csv", index=False)
        frames[name] = detail
        metrics.append({"실행": name, "실행 폴더": run.name, "mAP50–95": metric})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(folder / "metrics.csv", index=False)
    pair, summary = score_pair(frames["baseline"], frames["cutmix"])
    pair.to_csv(folder / "paired_frames.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    print(metrics.to_string(index=False))
    return summary
