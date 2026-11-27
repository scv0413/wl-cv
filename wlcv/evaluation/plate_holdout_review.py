"""저장된 배포 출력을 수동 정답과 겹치고 미탐 한 장의 필터 전 후보를 복원한다."""
from pathlib import Path

import cv2
import fiftyone as fo
from fiftyone import ViewField as F
import numpy as np
import pandas as pd
from ultralytics import YOLO

from wlcv.annotation.storage import box_path
from wlcv.config import ROOT, VIDEOS
from wlcv.data.frames import frames_rgb
from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation.plate_holdout import BOX, REVIEW
from wlcv.runtime.tracking import iou


CASE_VIDEO = "a07_sn_h_02"
CASE_FRAME = 54


def detection(label, box, width, height, confidence):
    x1, y1, x2, y2 = box
    return fo.Detection(label=label, confidence=float(confidence), bounding_box=[
        float(x1 / width), float(y1 / height),
        float((x2 - x1) / width), float((y2 - y1) / height),
    ])


def add_case_candidates(sample, rgb, frame, models, previous):
    """53~55번은 사람 후보를, 54번에는 두 원판 모델의 필터 전 후보도 표시한다."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = rgb.shape[:2]
    for name, model in models.items():
        pose = name == "people_candidates"
        if not pose and frame != CASE_FRAME:
            continue
        result = model.predict(bgr, conf=.25 if pose else .001, imgsz=640,
                               device="cpu", verbose=False)[0]
        proposals = []
        for index, (box, conf) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())):
            proposal = detection("person" if pose else "plate", box, width, height, conf)
            proposal["number"] = index + 1
            if pose and previous is not None:
                proposal["iou_with_previous_performer"] = float(iou(np.array(box), previous))
            proposals.append(proposal)
        sample[name] = fo.Detections(detections=proposals)


def review_dataset(folder):
    folder = Path(folder).resolve()
    name = "wlcv-plate-holdout-" + folder.name.replace("_", "-")
    scores = pd.read_csv(folder / "scored_samples.csv").set_index(["video_id", "frame"])
    runs = pd.read_csv(folder / "runs.csv").set_index("video_id")
    videos = pd.read_csv(folder / "by_video.csv").set_index("video_id")
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        if dataset.info.get("source") != str(folder):
            raise ValueError("같은 이름의 데이터셋이 다른 결과 폴더를 가리킵니다.")
        check_scores(dataset, scores)
        return dataset

    models = {field: YOLO(ROOT / "models/pytorch" / filename) for field, filename in [
        ("people_candidates", "yolo26n-pose.pt"), ("baseline_raw", "plate_baseline.pt"),
        ("cutmix_raw", "plate_cutmix.pt"),
    ]}
    images = folder / "fiftyone_frames"
    images.mkdir(exist_ok=True)
    samples = []
    for video, run in runs.iterrows():
        rows = pd.read_csv(folder / run.run / "frames.csv").set_index("frame_index")
        labels = scores.loc[video]
        wanted = set(labels.index)
        case_frames = {53, 54, 55} if video == CASE_VIDEO else set()
        wanted.update(case_frames)
        for frame, rgb in enumerate(frames_rgb(VIDEOS / f"{video}.mp4")):
            if frame > max(wanted):
                break
            if frame not in wanted:
                continue
            image = images / f"{video}_{frame:04d}.png"
            if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"{image} 저장에 실패했습니다.")
            row = rows.loc[frame]
            sample = fo.Sample(filepath=str(image), person_id=videos.at[video, "person_id"],
                               video_id=video, frame_index=frame, plate_status=row.plate_status,
                               performer_selection=row.performer_selection,
                               person_candidates=int(row.person_candidates))
            if frame in labels.index and labels.at[frame, "scored"]:
                sample.tags.append("scored")
                truth = read_boxes(box_path(REVIEW, video, frame), inside_frame=True)
                sample["ground_truth"] = fo.Detections(detections=[
                    fo.Detection(label="plate", bounding_box=[x-w/2, y-h/2, w, h])
                    for x, y, w, h in truth])
            if frame in case_frames:
                sample.tags.append("case")
                prev = rows.loc[frame-1, ["performer_x1","performer_y1","performer_x2","performer_y2"]]
                previous = prev.to_numpy(float) if prev.notna().all() else None
                add_case_candidates(sample, rgb, frame, models, previous)
            pred = []
            if row[BOX].notna().all():
                height, width = rgb.shape[:2]
                pred = [detection("plate", row[BOX], width, height, row.plate_confidence)]
            sample["deployment"] = fo.Detections(detections=pred)
            samples.append(sample)

    dataset = fo.Dataset(name)
    dataset.add_samples(samples)
    reviewed = dataset.match_tags("scored")
    reviewed.evaluate_detections("deployment", gt_field="ground_truth", eval_key="eval",
                                 method="coco", iou=.5, compute_mAP=False)
    check_scores(dataset, scores)
    dataset.save_view("scored", reviewed)
    dataset.save_view("final-misses", reviewed.match(F("eval_fn") > 0))
    dataset.save_view("a07-54-context", dataset.match_tags("case").sort_by("frame_index"))
    dataset.info = {"source": str(folder)}
    dataset.save()
    dataset.persistent = True
    return dataset


def check_scores(dataset, scores):
    reviewed = dataset.match_tags("scored")
    fields = ["video_id", "frame_index", "eval_tp", "eval_fn", "eval_fp"]
    values = pd.DataFrame(dict(zip(fields, reviewed.values(fields))))
    actual = values.set_index(fields[:2]).sort_index()
    expected = scores.loc[scores.scored, ["tp", "fn", "fp"]].sort_index()
    actual.columns = expected.columns
    pd.testing.assert_frame_equal(actual, expected, check_names=False, check_dtype=False)
    if dataset.match_tags("case").sort_by("frame_index").values("frame_index") != [53, 54, 55]:
        raise ValueError("미탐 전후 프레임을 확인하세요.")
