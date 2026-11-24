"""검수한 자세 데이터로 YOLO26n-pose 시험 학습 또는 첫 파인튜닝을 실행한다."""

from pathlib import Path

from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import DEVICE, ROOT, RUNS

ORIGINAL_POSE = ROOT / "models/yolo26n-pose.pt"


def train_pose(data_yaml, *, epochs, name, device=DEVICE, **options):
    """원본 사전학습 가중치로 시작해 실행별 best.pt와 last.pt를 확인한다."""
    model = YOLO(str(ORIGINAL_POSE.resolve(strict=True)))
    with tqdm(total=epochs, desc="자세 학습", unit="epoch") as progress:
        model.add_callback("on_train_epoch_end", lambda trainer: progress.update())
        model.train(
            data=str(data_yaml.resolve(strict=True)),
            epochs=epochs,
            batch=1,
            workers=0,
            device=device,
            project=str(RUNS),
            name=name,
            exist_ok=False,
            save=True,
            **options,
        )
    run_dir = Path(model.trainer.save_dir)
    for path in (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"이번 실행의 결과 파일을 확인하세요. {path}")
    return run_dir
