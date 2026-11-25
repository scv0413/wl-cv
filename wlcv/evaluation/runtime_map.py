"""고정 검증 사진에서 같은 입력과 같은 채점기로 원판 검출 mAP를 비교한다."""
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywhn2xyxy

from wlcv.config import ANNOTATIONS, DATASET, SPLIT
from wlcv.data.plate_labels import label_frame, read_boxes
from wlcv.evaluation.onnx_plate import WEIGHTS, THREADS, candidates, raw_pytorch


def validation_photos():
    """사진·라벨·선수 분할이 원판 베이스라인의 고정 검증셋인지 확인한다."""
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    trained = set(pd.read_csv(WEIGHTS.parent.parent / "train_people.csv").person_id)
    originals = {}
    for label in (ANNOTATIONS / "plate_boxes").glob("*.txt"):
        video, _ = label_frame(label)
        if split.at[video, "role"] == "validation":
            originals[label.stem] = label
    images = sorted((DATASET / "images/val").glob("*.jpg"))
    labels = {path.stem: path for path in (DATASET / "labels/val").glob("*.txt")}
    if set(originals) != set(labels) or set(labels) != {path.stem for path in images}:
        raise ValueError("검증 사진과 원본 라벨 목록이 다릅니다.")
    people = {split.at[label_frame(path)[0], "person_id"] for path in images}
    if len(images) != 39 or len(people) != 6 or people & trained:
        raise ValueError("원판 학습 8명과 분리된 검증 6명의 사진 39장을 확인하세요.")
    for name, label in labels.items():
        if originals[name].read_bytes() != label.read_bytes():
            raise ValueError(f"원본과 다른 라벨: {name}")
    return images, labels


@torch.inference_mode()
def compare_map(graph_folder, output_folder):
    """원본 화면 좌표에서 NMS 후 모든 후보를 채점한다. 시간 연결 전 검출기 평가다."""
    graph_folder = Path(graph_folder)
    graph = graph_folder / "plate.onnx"
    settings = json.loads((graph_folder / "settings.json").read_text())
    for path, key in [(WEIGHTS, "weights_sha256"), (graph, "onnx_sha256")]:
        if sha256(path.read_bytes()).hexdigest() != settings[key]:
            raise ValueError(f"46번 실험 파일과 체크섬이 다릅니다: {path.name}")
    images, labels = validation_photos()
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    input_spec, = session.get_inputs()
    if input_spec.shape != [1, 3, 640, 384] or input_spec.type != "tensor(float)":
        raise ValueError("배포 후보인 고정 FP32 그래프를 확인하세요.")

    detector = YOLO(str(WEIGHTS))
    detector.predict(str(images[0]), device="cpu", imgsz=640, conf=0.001, verbose=False)
    torch.set_num_threads(THREADS)
    validator = DetectionValidator(save_dir=Path(output_folder))
    metrics = {name: DetMetrics(names={0: "plate"}) for name in ("pytorch", "onnx")}
    frames = []
    for image in tqdm(images, desc="검증 사진", unit="장"):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError(f"읽을 수 없는 사진: {image.name}")
        height, width = bgr.shape[:2]
        boxes = torch.tensor(read_boxes(labels[image.stem]), dtype=torch.float32).reshape(-1, 4)
        truth = xywhn2xyxy(boxes, w=width, h=height)
        classes = torch.zeros(len(truth))
        tensor = detector.predictor.preprocess([bgr])
        if list(tensor.shape) != input_spec.shape or tensor.dtype != torch.float32:
            raise ValueError(f"{image.name}의 입력 크기 또는 정밀도가 다릅니다.")
        outputs = {
            "pytorch": raw_pytorch(detector.predictor, tensor),
            "onnx": session.run(None, {input_spec.name: tensor.numpy()})[0],
        }
        for name, output in outputs.items():
            if output.shape != (1, 5, 5040) or not np.isfinite(output).all():
                raise ValueError(f"{image.name}: {name} 원시 출력을 확인하세요.")
            pred = torch.from_numpy(candidates(output, tensor.shape[2:], bgr.shape))
            pred = pred[pred[:, 4].argsort(descending=True, stable=True)]
            matched = validator.match_predictions(pred[:, 5], classes, box_iou(truth, pred[:, :4]))
            metrics[name].update_stats({
                "tp": matched.numpy(), "conf": pred[:, 4].numpy(), "pred_cls": pred[:, 5].numpy(),
                "target_cls": classes.numpy(), "target_img": classes.unique().numpy(), "im_name": image.stem,
            })
            direct = pred[:, 4] >= 0.10
            tp = int(matched[direct, 0].sum())
            frames.append(dict(image=image.stem, runtime=name, targets=len(truth), candidates=len(pred),
                               direct_tp=tp, direct_fp=int(direct.sum()) - tp, direct_fn=len(truth) - tp))

    summary, by_iou = [], []
    for name, metric in metrics.items():
        metric.process()
        summary.append(dict(runtime=name, mAP50=metric.box.map50, mAP50_95=metric.box.map))
        for threshold, ap in zip(validator.iouv.tolist(), metric.box.all_ap[0], strict=True):
            by_iou.append(dict(runtime=name, iou=round(threshold, 2), ap=ap))
    return pd.DataFrame(summary), pd.DataFrame(by_iou), pd.DataFrame(frames)
