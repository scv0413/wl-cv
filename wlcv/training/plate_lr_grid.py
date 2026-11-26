"""같은 RTX 3090에서 원본 가중치로 학습률 조건을 독립적으로 학습한다."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfile
import time

import torch
from ultralytics import YOLO, __version__, settings

CASES = {
    "lr001_linear": (0.001, False),
    "lr001_cosine": (0.001, True),
    "lr0005_linear": (0.0005, False),
    "lr0005_cosine": (0.0005, True),
}
LOW_LR_CASES = {
    "lr0001_linear": (0.0001, False),
    "lr0001_cosine": (0.0001, True),
}
FULL_RUN = "lr0001_linear_200"
CHECKPOINT_EPOCHS = (106, 150, 200)
TRAIN_CASES = CASES | LOW_LR_CASES | {FULL_RUN: (0.0001, False)}
INPUT_HASH = "aaf184492fc91e32969b77e51a6bd2ec7f5b36908ca662a36ef8d53168c381c9"


def train(case):
    root = Path("/workspace/cutmix")
    if hashlib.sha256((root / "input.tar").read_bytes()).hexdigest() != INPUT_HASH:
        raise ValueError("기존 3090 학습 입력과 다릅니다.")
    if (__version__ != "8.4.202" or torch.__version__ != "2.8.0+cu128"
            or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090"):
        raise RuntimeError("PyTorch, Ultralytics, 실제 GPU를 확인하세요.")
    plugins = ["sync", "wandb", "clearml", "comet", "mlflow", "neptune", "raytune", "hub", "dvc", "tensorboard"]
    settings.update({key: False for key in plugins if key in settings})
    lr, cosine = TRAIN_CASES[case]
    args = json.loads((root / "settings.json").read_text())
    # 기존 auto가 실제로 고른 AdamW의 beta1과 bias 워밍업 시작값을 유지한다.
    args.update(epochs=200, patience=0 if case == FULL_RUN else 30, optimizer="AdamW", momentum=0.9,
                warmup_bias_lr=0.0, lr0=lr, cos_lr=cosine)
    run = root / "runs" / case
    if run.exists():
        raise FileExistsError(run)
    audit = {}

    def check_optimizer(trainer):
        groups = trainer.optimizer.param_groups
        if (type(trainer.optimizer).__name__ != "AdamW" or trainer.args.warmup_bias_lr != 0
                or any(g["initial_lr"] != lr or tuple(g["betas"]) != (0.9, 0.999) for g in groups)
                or bool(trainer.args.cos_lr) != cosine):
            raise ValueError("실제 옵티마이저 또는 스케줄이 계획과 다릅니다.")
        audit.update(optimizer="AdamW", initial_lr=lr, betas=list(groups[0]["betas"]),
                     warmup_bias_lr=trainer.args.warmup_bias_lr, cos_lr=cosine)

    def record_best(trainer):
        if trainer.fitness == trainer.best_fitness:
            audit["best_epoch"] = trainer.epoch + 1

    def save_checkpoints(trainer):
        # Ultralytics는 0부터 세므로 노트북에 표시되는 에폭으로 변환한다.
        epoch = trainer.epoch + 1
        if epoch in CHECKPOINT_EPOCHS:
            copyfile(trainer.last, trainer.last.parent / f"epoch{epoch}.pt")

    model = YOLO(str(root / "yolo26n.pt"))
    if case == FULL_RUN:
        model.add_callback("on_model_save", save_checkpoints)
    model.add_callback("on_train_start", check_optimizer)
    model.add_callback("on_model_save", record_best)
    start = time.perf_counter()
    model.train(data=str(root / "dataset/data.yaml"), device=0, project=str(run.parent), name=case, **args)
    audit["train_and_val_seconds"] = round(time.perf_counter() - start, 2)
    with (run / "results.csv").open() as f:
        history = list(csv.DictReader(f))
    audit["epochs_run"] = int(history[-1]["epoch"])
    if not all((run / "weights" / f).is_file() for f in ("best.pt", "last.pt")):
        raise FileNotFoundError("체크포인트가 빠졌습니다.")
    if case == FULL_RUN:
        if audit["epochs_run"] != 200:
            raise RuntimeError("200에폭까지 완료되지 않았습니다.")
        for epoch in CHECKPOINT_EPOCHS:
            path = run / "weights" / f"epoch{epoch}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["epoch"] + 1 != epoch:
                raise ValueError(f"{path.name}의 내부 에폭이 다릅니다.")
        audit["checkpoint_epochs"] = list(CHECKPOINT_EPOCHS)
    (run / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=TRAIN_CASES)
    train(parser.parse_args().case)
