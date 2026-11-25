"""100에폭 기준 실행에서 학습 일정 두 값만 바꾸고 새 best.pt를 보존한다."""

from pathlib import Path

import json

import pandas as pd
import yaml
from filelock import FileLock
from ultralytics import YOLO

from wlcv.annotation.hub_points import FRAMES, read_hubs
from wlcv.config import RUNS, SPLIT
from wlcv.data.dataset import _output_signature, read_splits
from wlcv.data.plate_labels import label_frame
from wlcv.training.train import train_baseline

BASE = RUNS / "plate_baseline"
TRIAL = RUNS / "plate_200_patience50"
SCHEDULE = {"epochs": 200, "patience": 50}


def baseline_settings():
    """원본 베이스라인의 설정, 학습 가중치와 사람별 분할을 확인한다."""
    settings = yaml.safe_load((BASE / "args.yaml").read_text())
    weights = Path(settings["model"]).resolve(strict=True)
    data = Path(settings["data"]).resolve(strict=True)
    if weights.name != "yolo26n.pt" or settings["epochs"] != 100 or settings["resume"]:
        raise ValueError("100에폭 베이스라인의 사전학습 모델과 실행 설정을 확인하세요.")
    manifest = json.loads((data.parent / ".dataset.json").read_text())
    if _output_signature(data.parent) != manifest["outputs"]:
        raise ValueError("준비한 데이터셋의 이미지 또는 라벨이 바뀌었습니다.")
    splits = read_splits(SPLIT)
    used = {}
    for role in ("train", "val"):
        images = sorted((data.parent / "images" / role).glob("*.jpg"))
        labels = {p.stem for p in (data.parent / "labels" / role).glob("*.txt")}
        videos = {label_frame(p)[0] for p in images}
        if {p.stem for p in images} != labels or any(splits[v] != role for v in videos):
            raise ValueError(f"{role} 이미지와 라벨, 사람별 분할을 확인하세요.")
        used[role] = videos
        expected = 45 if role == "train" else 39
        if len(images) != expected:
            raise ValueError(f"{role} 프레임이 기준 실행의 {expected}장과 다릅니다.")
    training = pd.read_csv(BASE / "train_people.csv")
    people = pd.read_csv(SPLIT).set_index("video_id").person_id
    if (set(training.video_id) != used["train"] or len(set(people.loc[list(used["val"])])) != 6
            or set(training.person_id) & set(people.loc[list(used["val"])])):
        raise ValueError("기준 실행의 학습 선수와 검증 6명의 분할을 확인하세요.")
    return settings


def settings_for_trial():
    return {**baseline_settings(), **SCHEDULE}


def train_schedule():
    """판독 완료 후 시작한다. 재실행하면 완료 결과를 열거나 중단된 마지막 에폭부터 잇는다."""
    if set(read_hubs().frame_index) != set(FRAMES):
        raise ValueError("14번 노트북에서 세 장을 직접 확인 또는 판독 불가로 저장하세요.")
    settings = settings_for_trial()
    weights = Path(settings.pop("model"))
    data = Path(settings.pop("data"))
    device = settings.pop("device")
    for key in ("name", "project", "save_dir"):
        settings.pop(key, None)

    marker = TRIAL / "schedule_complete.txt"
    with FileLock(str(RUNS / ".plate_200_patience50.lock"), timeout=0):
        if marker.exists():
            (TRIAL / "weights/best.pt").resolve(strict=True)
            (TRIAL / "weights/last.pt").resolve(strict=True)
            return TRIAL
        if TRIAL.exists():
            last = TRIAL / "weights/last.pt"
            if not last.is_file():
                raise ValueError(f"아직 재개 가능한 last.pt가 없습니다. 이전 실행 폴더를 확인하세요: {TRIAL}")
            prior = yaml.safe_load((TRIAL / "args.yaml").read_text())
            for key, value in settings.items():
                if key != "resume" and prior.get(key) != value:
                    raise ValueError(f"중단된 실행 설정이 다릅니다: {key}")
            if Path(prior["model"]).resolve() not in (weights.resolve(), last.resolve()) or Path(prior["data"]).resolve() != data.resolve():
                raise ValueError("이전 실행이 같은 사전학습 모델·데이터인지 확인하세요.")
            model = YOLO(str(last))
            if model.ckpt.get("epoch", -1) >= 0:
                model.train(resume=True, device=device)
        else:
            train_baseline(data, device=device, name=TRIAL.name, weights=weights, settings=settings)
        for filename in ("best.pt", "last.pt"):
            (TRIAL / "weights" / filename).resolve(strict=True)
        count = len(pd.read_csv(TRIAL / "results.csv"))
        marker.write_text(f"epochs_completed={count}\n")
    return TRIAL
