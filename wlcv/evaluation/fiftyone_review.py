"""사람별 검증셋의 원판 정답과 예측을 FiftyOne에서 확인한다."""

import fiftyone as fo
import pandas as pd
from fiftyone.utils.ultralytics import to_detections
from tqdm.auto import tqdm

from wlcv.data.plate_labels import label_frame


def load_validation_frames(data_yaml, split_csv):
    """학습에 사용한 data.yaml에서 val 프레임과 정답 박스를 읽는다."""
    # YOLOv5Dataset은 박스 라벨의 파일 형식을 가리킨다. YOLO26도 같은 형식을 쓴다.
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path=str(data_yaml),
        split="val",
        label_field="ground_truth",
    )
    people = pd.read_csv(split_csv, dtype=str)
    validation = people[people["role"] == "validation"].set_index("video_id")

    for sample in dataset.iter_samples(autosave=True):
        video_id, frame_index = label_frame(sample.filepath)
        sample["person_id"] = validation.at[video_id, "person_id"]
        sample["video_id"] = video_id
        sample["frame_index"] = frame_index
    return dataset


def evaluate_plates(dataset, model, confidence=0.25):
    """검수용 신뢰도에서 예측하고, IoU 0.5를 기준으로 정답과 대응시킨다."""
    if model.names != {0: "plate"}:
        raise ValueError("원판 한 클래스를 학습한 실험 폴더의 weights/best.pt를 사용하세요.")

    for sample in tqdm(dataset.iter_samples(autosave=True), total=len(dataset), desc="검증 프레임 예측"):
        result = model.predict(sample.filepath, conf=confidence, device="cpu", verbose=False)[0]
        sample["predictions"] = to_detections(result)

    dataset.evaluate_detections(
        "predictions", gt_field="ground_truth", eval_key="eval", method="coco", iou=0.5,
    )
