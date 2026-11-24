"""검수한 자세 데이터로 YOLO26n-pose 학습·검증 한 바퀴를 실행한다."""

from pathlib import Path

from ultralytics import YOLO

from wlcv.config import DEVICE, ROOT, RUNS


def train_pose_smoke(data_yaml, device=DEVICE):
    """Mac 노트북에서 한 장을 학습하고, 두 장을 검증한 뒤 실행 폴더를 반환한다."""
    weights = ROOT / "models/yolo26n-pose.pt"
    model = YOLO(str(weights.resolve(strict=True)))
    model.train(
        data=str(data_yaml.resolve(strict=True)),
        epochs=1,
        batch=1,
        workers=0,
        device=device,
        project=str(RUNS),
        name="pose_smoke",
        exist_ok=False,
        save=True,
    )
    run_dir = Path(model.trainer.save_dir)
    for path in (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"이번 실행의 결과 파일을 확인하세요. {path}")
    return run_dir
