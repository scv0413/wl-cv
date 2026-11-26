"""RTX 3090에서 한 번 학습하며 patience 30과 50이 선택하는 가중치를 보관한다."""
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.utils.torch_utils import strip_optimizer

INPUT_SHA256 = "aaf184492fc91e32969b77e51a6bd2ec7f5b36908ca662a36ef8d53168c381c9"


def remember_best(trainer, record):
    """저장 직후에만 검사하고 첫 30에폭 정체 시점의 best를 한 번 복사한다."""
    snapshot = trainer.best.with_name("best_patience30.pt")
    epoch = trainer.epoch + 1
    # best.pt는 최고 fitness와 동점이어도 덮어쓴다. 실제 파일이 선택된 에폭도 기록한다.
    if trainer.fitness == trainer.best_fitness:
        record["best_weight_epoch"] = epoch
    record["last_epoch"] = epoch
    record["strict_best_epoch"] = int(trainer.stopper.best_epoch)
    if record["first_30_epoch"] is None and epoch - trainer.stopper.best_epoch >= 30:
        shutil.copyfile(trainer.best, snapshot)
        record["first_30_epoch"] = epoch
        record["p30_weight_epoch"] = record["best_weight_epoch"]
        (trainer.save_dir / "patience.json").write_text(json.dumps(record, indent=2))
        print(f"patience 30 첫 도달: {epoch}에폭", flush=True)

def train(root):
    if hashlib.sha256((root / "input.tar").read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("기존 3090 대조군의 입력 묶음을 확인하세요.")
    if not (torch.__version__ == "2.8.0+cu128" and ultralytics.__version__ == "8.4.202"
            and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"):
        raise RuntimeError("3090 대조군의 GPU와 패키지 버전을 확인하세요.")
    integrations = ("sync", "wandb", "clearml", "comet", "mlflow", "neptune", "raytune", "hub", "dvc", "tensorboard")
    settings.update({k: False for k in integrations if k in settings})
    args = json.loads((root / "settings.json").read_text())
    if (args["epochs"], args["patience"], args["cutmix"], args["resume"]) != (100, 100, 0, False):
        raise ValueError("초기 대조군 설정을 확인하세요.")
    args.update(epochs=1000, patience=50)
    run = root / "runs/schedule1000"
    if run.exists():
        raise FileExistsError(run)
    snapshot = run / "weights/best_patience30.pt"
    record = {"first_30_epoch": None, "p30_weight_epoch": None}

    model = YOLO(str(root / "yolo26n.pt"))
    model.add_callback("on_model_save", partial(remember_best, record=record))
    started = time.perf_counter()
    model.train(data=str(root / "dataset/data.yaml"), project=str(run.parent), name=run.name, device=0, **args)
    record["train_and_val_seconds"] = round(time.perf_counter() - started, 2)
    if record["first_30_epoch"] is None:
        # 마지막까지 정체가 30에폭 미만이면 두 규칙 모두 상한에서 같은 best를 선택한다.
        shutil.copyfile(run / "weights/best.pt", snapshot)
        record["p30_weight_epoch"] = record["best_weight_epoch"]
    else:
        # 사본에서만 옵티마이저를 제거해 최종 best와 같은 추론용 형식으로 보관한다.
        strip_optimizer(snapshot)
    (run / "patience.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


if __name__ == "__main__":
    train(Path("/workspace/cutmix"))
