"""100에폭 베이스라인을 한 번 학습하며 지정한 두 에폭의 가중치를 보관한다."""

from pathlib import Path
from shutil import copyfile

from filelock import FileLock
import pandas as pd
from ultralytics import YOLO

from wlcv.config import RUNS
from wlcv.training.plate_schedule import baseline_settings
from wlcv.training.train import train_baseline

EPOCHS_TO_KEEP = (65, 95)
RUN = RUNS / "plate_baseline_epoch65_95"


def _copy_epoch(last, number):
    destination = last.with_name(f"epoch_{number}.pt")
    temporary = destination.with_suffix(".tmp")
    copyfile(last, temporary)
    temporary.replace(destination)


def save_selected_epoch(trainer):
    # 내부 번호 64는 화면에서 65번째 에폭이다. 현재 에폭의 last.pt가 기록된 직후 호출된다.
    number = trainer.epoch + 1
    if number in EPOCHS_TO_KEEP:
        _copy_epoch(trainer.last, number)
        print(f"{number}에폭 저장 완료")


def completed_weights():
    """완료한 한 실행에서 두 가중치 파일의 내부 에폭 번호도 확인한다."""
    history = pd.read_csv(RUN / "results.csv")
    if history.epoch.tolist() != list(range(1, 101)):
        raise ValueError("100에폭 학습을 마친 뒤 비교 셀을 실행하세요.")
    folder = RUN / "weights"
    for name in ("last.pt", "best.pt"):
        (folder / name).resolve(strict=True)
    weights = {}
    for number in EPOCHS_TO_KEEP:
        path = folder / f"epoch_{number}.pt"
        checkpoint = YOLO(str(path.resolve(strict=True))).ckpt
        if (checkpoint["epoch"] + 1 != number
                or Path(checkpoint["train_args"]["save_dir"]).resolve() != RUN.resolve()):
            raise ValueError(f"{path.name}의 내부 에폭 번호와 실행 폴더를 확인하세요.")
        weights[number] = path
    return weights


def train_epoch_pair():
    """처음에는 사전학습 가중치로 시작하고 중단된 실행은 같은 폴더에서 재개한다."""
    settings = baseline_settings()
    weights = Path(settings.pop("model"))
    data = Path(settings.pop("data"))
    device = settings.pop("device")
    for key in ("name", "project", "save_dir"):
        settings.pop(key, None)

    with FileLock(str(RUN.parent / f".{RUN.name}.lock"), timeout=0):
        if not RUN.exists():
            train_baseline(data, device=device, weights=weights, name=RUN.name,
                           settings=settings, on_model_save=save_selected_epoch)
        else:
            last = RUN / "weights/last.pt"
            if not last.is_file():
                raise FileNotFoundError(f"재개할 last.pt가 없습니다. 첫 에폭 전에 중단됐다면 {RUN}을 다른 이름으로 옮긴 뒤 다시 실행하세요.")
            model = YOLO(str(last))
            if model.ckpt["epoch"] == -1:  # 학습을 마치면 Ultralytics가 재개 정보를 지운다.
                return completed_weights()
            saved_args = model.ckpt["train_args"]
            if (Path(saved_args["save_dir"]).resolve() != RUN.resolve()
                    or Path(saved_args["data"]).resolve() != data.resolve()):
                raise ValueError("체크포인트의 실행 폴더와 데이터 경로를 확인하세요.")
            finished = model.ckpt["epoch"] + 1
            # 마지막 에폭 복사 도중 커널이 꺼졌으면 보관 파일을 last.pt에서 복구한다.
            if finished in EPOCHS_TO_KEEP:
                _copy_epoch(last, finished)
            for number in EPOCHS_TO_KEEP:
                if 0 < number < finished and not last.with_name(f"epoch_{number}.pt").is_file():
                    raise FileNotFoundError(f"{number}에폭 파일이 없는 실행입니다. {RUN}을 보관한 뒤 새로 학습하세요.")
            if 0 < finished < settings["epochs"]:
                model.add_callback("on_model_save", save_selected_epoch)
                model.train(resume=str(last), data=str(data), device=device, save_dir=str(RUN))
        return completed_weights()
