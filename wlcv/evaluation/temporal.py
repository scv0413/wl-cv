"""준비부터 받기까지, 가까운 원판을 연속 몇 프레임 놓치는지 측정한다."""

from itertools import groupby, islice

import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

from wlcv.config import ANNOTATIONS, SPLIT, VIDEOS
from wlcv.data.frames import frames_rgb
from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation.plots import predict_rgb


def evaluate_frames(weights, review_csv, video_ids):
    """원본 프레임에서 한 번 추론하고 각 후보의 신뢰도와 정답 IoU를 남긴다."""
    model = YOLO(str(weights.resolve(strict=True)))
    if model.names != {0: "plate"}:
        raise ValueError("원판 탐지 가중치를 확인하세요.")
    reviewed = pd.read_csv(review_csv).set_index(["video_id", "frame_index"], verify_integrity=True)
    events = pd.read_csv(ANNOTATIONS / "point_labels.csv")
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    trained = set(pd.read_csv(weights.parent.parent / "train_people.csv")["person_id"])
    frame_rows, candidates = [], []
    for video in video_ids:
        if split.at[video, "role"] != "validation" or split.at[video, "person_id"] in trained:
            raise ValueError(f"{video}의 사람별 검증 분할을 확인하세요.")
        points = events[(events.video_id == video) & (events["pass"] == 1)].set_index("event")
        start, end = int(points.at["ready", "frame"]), int(points.at["catch", "frame"])
        if not 0 <= start <= end:
            raise ValueError(f"{video}의 준비·받기 프레임을 확인하세요.")
        last = None
        frames = frames_rgb(VIDEOS / f"{video}.mp4")
        try:
            segment = islice(frames, start, end + 1)
            progress = tqdm(segment, total=end - start + 1, desc=video, unit="frame")
            for index, rgb in enumerate(progress, start=start):
                label_path = ANNOTATIONS / "plate_boxes" / f"{video}_f{index}.txt"
                h, w = rgb.shape[:2]
                if label_path.exists():
                    [(cx, cy, bw, bh)] = read_boxes(label_path)
                    target = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
                    source = "existing"
                else:
                    target = reviewed.loc[(video, index), ["x1", "y1", "x2", "y2"]].tolist()
                    source = "visual_review"
                x1, y1, x2, y2 = target
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f"{video} {index}번 정답 좌표를 확인하세요.")
                frame_rows.append((video, index, source, *target))
                boxes = predict_rgb(model, rgb, conf=0.001, imgsz=640, device="cpu").boxes.cpu()
                overlaps = box_iou(torch.tensor([target], dtype=torch.float32), boxes.xyxy)[0]
                for box, score, iou in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), overlaps.tolist()):
                    candidates.append((video, index, score, iou, *box))
                last = index
        finally:
            frames.close()
        if last != end:
            raise OSError(f"{video}의 {end}번까지 디코딩할 수 없습니다.")
    keys = ["video_id", "frame_index"]
    coords = ["x1", "y1", "x2", "y2"]
    return (pd.DataFrame(frame_rows, columns=keys + ["label_source"] + coords),
            pd.DataFrame(candidates, columns=keys + ["confidence", "iou"] + coords))


def compare_thresholds(frames, candidates, thresholds=(0.25, 0.10)):
    """같은 후보에 기준을 적용한다. 한 대상의 중복 검출도 추가 박스로 센다."""
    keys = ["video_id", "frame_index"]
    matches = candidates[candidates.iou >= 0.50].sort_values("confidence", ascending=False)
    best = matches.drop_duplicates(keys)[keys + ["confidence", "iou"]]
    scores = frames.merge(best, on=keys, how="left", validate="one_to_one")
    details, runs, summary = [], [], []
    for threshold in thresholds:
        counts = candidates[candidates.confidence >= threshold].groupby(keys).size()
        result = scores.join(counts.rename("boxes"), on=keys)
        result["detected"] = result.confidence.ge(threshold)
        result["extra_boxes"] = result.boxes.fillna(0).astype(int) - result.detected.astype(int)
        result["threshold"] = threshold
        details.append(result)
        for video, clip in result.groupby("video_id", sort=False):
            clip = clip.sort_values("frame_index")
            if not clip.frame_index.diff().dropna().eq(1).all():
                raise ValueError(f"{video}의 연속 평가 구간에 빠진 프레임이 있습니다.")
            lengths = []
            for detected, run in groupby(clip.itertuples(), key=lambda row: row.detected):
                if detected:
                    continue
                indices = [row.frame_index for row in run]
                lengths.append(len(indices))
                runs.append((video, threshold, indices[0], indices[-1], len(indices)))
            summary.append((video, threshold, len(clip), sum(lengths), max(lengths, default=0),
                            int(clip.extra_boxes.gt(0).sum())))
    intervals = pd.DataFrame(runs, columns=["video_id", "threshold", "start", "end", "length"])
    totals = pd.DataFrame(summary, columns=["video_id", "threshold", "frames", "missed",
                                            "longest_miss", "extra_box_frames"])
    return pd.concat(details, ignore_index=True), intervals, totals
