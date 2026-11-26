"""동일한 GPU에서 같은 정렬 순서·초기 가중치로 한 조건을 100에폭 학습한다."""
import argparse
import json
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("control", "cutmix"))
    condition = parser.parse_args().condition
    root = Path(__file__).resolve().parent
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0)
    if "3090" not in gpu or ultralytics.__version__ != "8.4.202":
        raise RuntimeError((gpu, ultralytics.__version__))
    settings.update({key: False for key in (
        "sync", "wandb", "clearml", "comet", "mlflow", "neptune", "raytune",
        "hub", "dvc", "tensorboard"
    ) if key in settings})
    args = json.loads((root / "settings.json").read_text())
    assert args["cutmix"] == 0 and args["epochs"] == 100 and args["imgsz"] == 640
    assert args["scale"] == 0.5
    args["scale"] = 0.25
    args["cutmix"] = 0.30 if condition == "cutmix" else 0.0
    run = root / "runs" / condition
    if run.exists():
        raise FileExistsError(run)
    print({"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
           "ultralytics": ultralytics.__version__, "cutmix": args["cutmix"]}, flush=True)
    YOLO(root / "yolo26n.pt").train(
        data=str(root / "dataset/data.yaml"), device=0,
        project=str(run.parent), name=condition, **args
    )
    import csv
    with (run / "results.csv").open() as f:
        epochs = [int(row["epoch"]) for row in csv.DictReader(f)]
    assert epochs == list(range(1, 101))
    for filename in ("best.pt", "last.pt"):
        assert (run / "weights" / filename).stat().st_size > 0
    print(f"완료 {condition} {len(epochs)} epochs", flush=True)


if __name__ == "__main__":
    main()
