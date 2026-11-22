from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import ANNOTATIONS, ROOT, SPLIT, VIDEOS
from wlcv.data.dataset import label_frame, read_boxes
from wlcv.data.frames import frames_rgb
from wlcv.evaluation.plots import predict_rgb

MANIFEST = ROOT / "data" / "metadata" / "negative_eval.csv"


def evaluate_negatives(
    weights: Path, manifest=MANIFEST, *, video_dir=VIDEOS, split_csv=SPLIT,
    labels_dir=ANNOTATIONS / "plate_boxes", conf=0.25,
) -> pd.DataFrame:
    """전 구간에 원판이 없는 영상에서 오탐 박스 수와 오탐이 나온 프레임 비율을 센다."""
    candidates = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    fields = ["video_id", "person_id", "filename"]
    candidates = candidates[fields].apply(lambda column: column.str.strip())
    columns = ["video_id", "person_id", "frames", "fp_frames", "fp_boxes", "fp_frame_rate"]
    if candidates.empty:
        print("미측정 · 전 구간에 원판이 없다고 확인한 영상이 없습니다.")
        return pd.DataFrame(columns=columns)
    if candidates.eq("").any().any() or candidates["video_id"].duplicated().any():
        raise ValueError("평가 목록에 빈 값 또는 중복 영상이 있습니다.")

    weights = Path(weights)
    snapshot = weights.parent.parent / "train_people.csv"
    if not snapshot.is_file():
        raise FileNotFoundError(f"가중치와 함께 저장한 학습 인물 목록이 필요합니다. {snapshot}")
    trained = pd.read_csv(snapshot, dtype=str)
    if trained.empty or trained[["video_id", "person_id"]].isna().any().any():
        raise ValueError("학습 인물 목록이 비어 있거나 값이 누락되었습니다.")
    trained_people = set(trained["person_id"])
    trained_videos = set(trained["video_id"])
    split = pd.read_csv(split_csv, dtype=str).set_index("video_id", verify_integrity=True)
    known_people = set(split["person_id"]) | trained_people

    # 모든 후보를 확인한 다음에만 추론을 시작한다.
    for video, person, filename in candidates.itertuples(index=False, name=None):
        if Path(filename).name != filename or Path(filename).stem != video:
            raise ValueError(f"{video}의 영상 ID와 파일명이 일치해야 합니다.")
        if video in split.index and split.at[video, "person_id"] != person:
            raise ValueError(f"{video}의 사람 ID가 분할표와 다릅니다.")
        prefix = video.split("_", 1)[0]
        if prefix in known_people and prefix != person:
            raise ValueError(f"{filename}의 사람 ID를 확인하세요.")
        if person in trained_people or video in trained_videos:
            raise ValueError(f"{video}의 수행자 {person}은 학습에 사용되었습니다.")
        if not (Path(video_dir) / filename).is_file():
            raise FileNotFoundError(Path(video_dir) / filename)
    candidate_ids = set(candidates["video_id"])
    for label in Path(labels_dir).glob("*.txt"):
        video, _ = label_frame(label)
        if video in candidate_ids and read_boxes(label):
            raise ValueError(f"{video}에 원판이 표시된 프레임이 있습니다.")

    model = YOLO(str(weights.resolve(strict=True)))
    if model.names != {0: "plate"}:
        raise ValueError("원판 검출용으로 학습한 가중치를 선택하세요.")
    rows = []
    for video, person, filename in candidates.itertuples(index=False, name=None):
        frames = fp_frames = fp_boxes = 0
        for frame in tqdm(frames_rgb(Path(video_dir) / filename), desc=video, unit="frame"):
            prediction = predict_rgb(model, frame, conf=conf, device="cpu")
            boxes = len(prediction.boxes)
            frames += 1
            fp_frames += int(boxes > 0)
            fp_boxes += boxes
        if frames == 0:
            raise ValueError(f"디코딩 가능한 프레임이 없습니다. {filename}")
        rows.append((video, person, frames, fp_frames, fp_boxes, fp_frames / frames))
    return pd.DataFrame(rows, columns=columns)
