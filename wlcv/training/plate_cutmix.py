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


def compare(trial):
    """같은 CPU 판정에서 두 모델의 a03 준비 구간과 검증 동작 중 확보·손실을 센다."""
    from wlcv.evaluation.schedule_review import evaluate_run

    control = RUNS / "plate_baseline"
    data = Path(baseline_settings()["data"])
    folder = trial / "paired_review"
    folder.mkdir(exist_ok=True)
    frames = {}
    for name, run in (("baseline",control),("cutmix",trial)):
        metric, detail = evaluate_run(run,data,folder)
        detail.to_csv(folder/f"{name}_frames.csv",index=False)
        frames[name] = detail[detail.in_motion & detail.match.notna()].copy()
        print(name, "mAP50–95", metric)
    keys = ["video_id","frame_index","method"]
    pair = frames["baseline"].merge(frames["cutmix"],on=keys,suffixes=("_base","_trial"),
                                    how="outer",validate="one_to_one",indicator=True)
    if not pair["_merge"].eq("both").all():
        raise ValueError("두 모델의 평가 프레임이 다릅니다.")
    pair["recovered"] = ~pair.match_base & pair.match_trial
    pair["lost"] = pair.match_base & ~pair.match_trial
    pair.to_csv(folder/"paired_frames.csv",index=False)
    early = pair.video_id.eq("a03_sn_f_01") & pair.frame_index.between(2,14)
    return pd.concat({scope: sub.groupby("method")[["match_base","match_trial","recovered","lost"]].sum()
                      for scope,sub in (("a03 2~14번",pair[early]),("동작 중 전체",pair))})
