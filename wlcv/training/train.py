"""YOLO26n 원판 탐지 베이스라인 학습."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import ANNOTATIONS, DATASET, DEVICE, RUNS, SPLIT, VIDEOS, WEIGHTS
from wlcv.data.dataset import prepare_dataset
from wlcv.data.plate_labels import label_frame


def train_baseline(
    data_yaml: Path, *, device=DEVICE, name="plate_baseline", weights=WEIGHTS, split_csv=SPLIT,
) -> Path:
    """사전학습 YOLO26n을 학습하고 이번 실행의 결과 폴더를 반환한다."""
    # 빈 배경 라벨도 학습에 사용된다. 이번 실행에 들어간 영상과 사람을 보존한다.
    train_labels = Path(data_yaml).parent / "labels" / "train"
    videos = sorted({label_frame(path)[0] for path in train_labels.glob("*.txt")})
    split = pd.read_csv(split_csv, dtype=str).set_index("video_id", verify_integrity=True)
    selected = split.loc[videos]
    if selected.empty or not selected["role"].eq("train").all():
        raise ValueError("학습 라벨과 사람별 분할표를 확인하세요.")
    model = YOLO(str(Path(weights).resolve(strict=True)))
    with tqdm(desc="학습", unit="epoch") as progress:
        def start_progress(trainer):
            progress.reset(total=trainer.epochs)
            selected[["person_id"]].to_csv(trainer.save_dir / "train_people.csv")

        model.add_callback("on_train_start", start_progress)
        model.add_callback("on_train_epoch_end", lambda trainer: progress.update())
        model.train(
            data=str(data_yaml), device=device, project=str(RUNS), name=name,
        )
    return Path(model.trainer.save_dir)


def main():
    parser = argparse.ArgumentParser(description="사람별 분할 데이터로 원판 검출 모델을 학습합니다.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--name", default="plate_baseline")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    data_yaml = prepare_dataset(VIDEOS, ANNOTATIONS / "plate_boxes", SPLIT, DATASET)
    run_dir = train_baseline(data_yaml, **vars(args))
    print(f"학습 결과 {run_dir}")


if __name__ == "__main__":
    main()
