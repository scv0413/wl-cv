"""원래 학습 선수 8명만 세 번 나눠 각 회차를 같은 사전학습 가중치에서 시작한다."""

from pathlib import Path

from filelock import FileLock
import pandas as pd

from wlcv.annotation.fold_hubs import FOLDS
from wlcv.config import ROOT, RUNS, SPLIT, VIDEOS
from wlcv.data.dataset import prepare_dataset
from wlcv.data.plate_labels import label_frame
from wlcv.training.plate_epoch_pair import train_checkpoint_pair
from wlcv.training.plate_schedule import baseline_settings

OUTPUT = ROOT / "outputs/plate_3fold"


def run_dir(fold):
    return RUNS / f"plate_3fold_{fold}"


def prepare_fold(fold, settings):
    """원본 베이스라인의 학습 라벨만 가져와 이 회차의 학습·검증 폴더를 만든다."""
    plan = pd.read_csv(FOLDS)
    if fold not in (1, 2, 3) or set(plan.heldout_fold) != {1, 2, 3}:
        raise ValueError("fold 번호와 분할표를 확인하세요.")
    if plan.video_id.duplicated().any() or plan.groupby("person_id").heldout_fold.nunique().ne(1).any():
        raise ValueError("한 영상 또는 선수가 여러 검증 회차에 들어갔습니다.")
    people = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True).loc[plan.video_id]
    if not people.role.eq("train").all() or people.person_id.tolist() != plan.person_id.tolist():
        raise ValueError("기존 학습 선수와 3-fold 명단이 다릅니다.")

    labels = Path(settings["data"]).parent / "labels/train"
    wanted = {(r.video_id, int(i)) for r in plan.itertuples() for i in r.frame_indices.split()}
    if wanted != {label_frame(p) for p in labels.glob("*.txt")}:
        raise ValueError("분할표의 프레임 번호와 베이스라인의 학습 라벨이 다릅니다.")

    split = plan[["video_id", "person_id"]].copy()
    split["role"] = "train"
    split.loc[plan.heldout_fold.eq(fold), "role"] = "validation"
    folder = OUTPUT / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "split.csv"
    with FileLock(str(folder / ".prepare.lock"), timeout=30):
        content = split.to_csv(index=False)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"기존 분할이 현재 계획과 다릅니다. {path}")
        if not path.exists():
            temp = path.with_suffix(".tmp")
            temp.write_text(content)
            temp.replace(path)
        data = prepare_dataset(VIDEOS, labels, path, folder / "dataset")
    return data, path


def train_fold(fold):
    settings = baseline_settings()
    data, split_csv = prepare_fold(fold, settings)
    return train_checkpoint_pair(run_dir(fold), data, settings, split_csv)
