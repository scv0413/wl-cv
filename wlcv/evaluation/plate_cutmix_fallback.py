"""기존 원판 모델이 박스를 내지 않을 때만 두 번째 모델의 출력을 사용한다."""
import pandas as pd

from wlcv.config import ROOT
from wlcv.training.plate_cutmix import check_same_labels, score_pair


def fill_missing_boxes(primary, secondary):
    """영상·프레임·방법을 맞추고, 정답을 참조하지 않고 선택한다."""
    keys = ["video_id", "frame_index", "method"]
    a = primary.set_index(keys, verify_integrity=True).sort_index()
    b = secondary.set_index(keys, verify_integrity=True).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("두 모델에서 평가한 프레임 또는 방법이 다릅니다.")
    take = a.x1.isna() & b.x1.notna()
    result = pd.concat([a.loc[~take], b.loc[take]]).sort_index()
    result["used_cutmix"] = take
    return result.reset_index()


def replay():
    """Mac 베이스라인과 3090 CutMix가 각각 영상 전체를 추론해 둔 표를 읽는다."""
    root = ROOT / "outputs"
    primary = pd.read_csv(root / "plate_cutmix_preview/mac_reference/frames.csv",
                          dtype={"match": "boolean"})
    secondary = pd.read_csv(root / "plate_cutmix_3090/20261126/paired_review/cutmix_frames.csv",
                            dtype={"match": "boolean"})
    # 두 실행이 동일한 정답과 채점 범위를 사용했는지 확인한다.
    check_same_labels(primary, secondary)
    result = fill_missing_boxes(primary, secondary)
    pairs, summary = score_pair(primary, result)
    destination = root / "plate_cutmix_fallback"
    destination.mkdir(exist_ok=True)
    result.to_csv(destination / "frames.csv", index=False)
    pairs.to_csv(destination / "paired.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    return result, summary


def check_training_backgrounds():
    """b05·b06은 학습에서 본 빈 배경이다. 독립 검증 점수와 구분해 센다."""
    from ultralytics import YOLO

    from wlcv.config import DATASET, RUNS
    from wlcv.data.plate_labels import read_boxes
    from wlcv.evaluation.full_video import predict_plate_videos
    from wlcv.evaluation.plate_track import select_centers

    videos = ["b05_sn_f_01", "b06_sn_f_01"]
    for video in videos:
        labels = list((DATASET / "labels/train").glob(f"{video}_f*.txt"))
        if not labels or any(read_boxes(path) for path in labels):
            raise ValueError(f"{video}의 학습용 빈 배경 라벨을 확인하세요.")
    gpu_runs = ROOT / "outputs/plate_cutmix_3090/20261126/runs"
    runs = {"기존 모델": RUNS / "plate_baseline", "3090 대조군": gpu_runs / "control",
            "CutMix": gpu_runs / "cutmix"}
    tracks = {}
    for name, run in runs.items():
        model = YOLO(str((run / "weights/best.pt").resolve(strict=True)))
        if model.names != {0: "plate"}:
            raise ValueError("원판 모델을 지정하세요.")
        frames, candidates = predict_plate_videos(model, videos)
        tracks[name] = select_centers(frames, candidates, show_progress=False)
    tracks["빈 출력 보완"] = fill_missing_boxes(tracks["기존 모델"], tracks["CutMix"])
    rows = []
    for name, data in tracks.items():
        for (video, method), group in data.groupby(["video_id", "method"]):
            seen = group.sort_values("frame_index").x1.notna()
            longest = int(seen.groupby((~seen).cumsum()).sum().max())
            rows.append((name, video, method, len(seen), int(seen.sum()), longest))
    table = pd.DataFrame(rows, columns=["모델", "영상", "방법", "전체", "오탐 프레임", "최장 연속"])
    folder = ROOT / "outputs/plate_cutmix_fallback"
    folder.mkdir(exist_ok=True)
    pd.concat(tracks, names=["model"]).reset_index(level=0).to_csv(
        folder / "background_frames.csv", index=False
    )
    table.to_csv(folder / "training_backgrounds.csv", index=False)
    return table
