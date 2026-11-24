"""같은 검증용 바닥 질감을 원판과 빈 바닥에 놓아 위치별 반응을 비교한다."""

import hashlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.utils.metrics import box_iou

from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation.plots import predict_rgb


def reviewed_floor_frames(positions):
    table = pd.read_csv(positions)
    floor = table[(table.condition == "Floor") & (table.usage == "apply")]
    return floor.sort_values(["person", "video", "frame"]).set_index("stem", verify_integrity=True)


def load_image(dataset, stem):
    with Image.open(dataset / "images/val" / f"{stem}.jpg") as image:
        return np.array(image.convert("RGB"))


def make_example(dataset, floors, stem):
    """각 위치에 사용할 원본 좌표와 픽셀을 정한다. 추론 결과는 사용하지 않는다."""
    row = floors.loc[stem]
    image = load_image(dataset, stem)
    height, width = image.shape[:2]
    [(cx, cy, bw, bh)] = read_boxes(dataset / "labels/val" / f"{stem}.txt")
    truth = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
    x1, y1, x2, y2 = np.rint(truth).astype(int)
    floor = tuple(int(row[k]) for k in ["left", "top", "right", "bottom"])
    fx, fy, fr, fb = floor
    w = fr - fx
    h = (fb - fy) // 2
    # 같은 사람의 비슷한 바닥을 제자리에 복사하는 경우를 피한다.
    others = floors[floors.person != row.person]
    donor = others.iloc[floors.index.get_loc(stem) % len(others)]
    source = load_image(dataset, donor.name)
    sx, sy, sr, sb = [int(donor[k]) for k in ["left", "top", "right", "bottom"]]
    tile = np.asarray(Image.fromarray(source[sy:sb, sx:sr]).resize((w, 2*h), Image.Resampling.BILINEAR))
    strip = tile[:h]
    left = (x1 + x2 - w) // 2
    mid = (y1 + y2 - h) // 2
    bottom = y2 - h
    if not (w > 0 and h > 0 and x1 <= left and left+w <= x2 and y1 <= mid
            and mid+h <= y2 and y2+h <= height):
        raise ValueError(f"{stem}에 같은 크기의 가림 대조군을 만들 수 없습니다.")
    if not (fr <= x1 or fx >= x2 or fb <= y1 or fy >= y2):
        raise ValueError(f"{stem}의 빈 바닥과 원판이 겹칩니다.")
    # 아래 끝과 윤곽 횡단은 박스 안쪽의 픽셀이 같다. 횡단에서만 아래 절반을 더 붙인다.
    variants = {
        "Original": None,
        "Center strip": (left, mid, strip),
        "Lower strip": (left, bottom, strip),
        "Cross lower edge": (left, bottom, tile),
        "Floor strip": (fx, fy, strip),
        "Center full": (left, (y1+y2-2*h)//2, tile),
        "Floor full": (fx, fy, tile),
    }
    return image, truth, floor, donor.name, variants


def paste_patch(image, placement):
    if placement is None:
        return image
    x, y, patch = placement
    h, w = patch.shape[:2]
    changed = image.copy()
    changed[y:y+h, x:x+w] = patch
    return changed


def evaluate_positions(model, dataset, floors, *, threshold=0.10):
    rows = []
    for stem in tqdm(floors.index, desc="패치 위치 비교", unit="장"):
        image, truth, floor, donor, variants = make_example(dataset, floors, stem)
        tx, ty, tr, tb = np.rint(truth).astype(int)
        for condition, placement in variants.items():
            changed = paste_patch(image, placement)
            output = predict_rgb(model, changed, conf=0.001, imgsz=640, device="cpu")
            boxes = output.boxes.cpu()
            overlaps = box_iou(torch.tensor([truth], dtype=torch.float32), boxes.xyxy)[0]
            matched = torch.where((boxes.cls == 0) & (overlaps >= 0.50))[0]
            record = dict(stem=stem, person=floors.at[stem, "person"], frame=int(floors.at[stem, "frame"]),
                          condition=condition, donor=donor, confidence=np.nan, iou=np.nan, detected=False,
                          floor_score=np.nan, floor_fp=False, coverage=0.0)
            if len(matched):
                idx = matched[boxes.conf[matched].argmax()]
                record.update(confidence=float(boxes.conf[idx]), iou=float(overlaps[idx]),
                              detected=float(boxes.conf[idx]) >= threshold)
            # 모든 조건에서 동일한 빈 바닥 ROI에 중심이 있는 비매칭 후보를 센다.
            fx, fy, fr, fb = floor
            cx, cy = boxes.xywh[:, 0], boxes.xywh[:, 1]
            floor_scores = boxes.conf[(boxes.cls == 0) & (overlaps < 0.50)
                                      & (cx >= fx) & (cx < fr) & (cy >= fy) & (cy < fb)]
            if len(floor_scores):
                score = float(floor_scores.max())
                record.update(floor_score=score, floor_fp=score >= threshold)
            if placement is not None:
                x, y, patch = placement
                h, w = patch.shape[:2]
                overlap = max(0, min(x+w, tr)-max(x, tx)) * max(0, min(y+h, tb)-max(y, ty))
                if np.array_equal(image[y:y+h, x:x+w], patch):
                    raise ValueError(f"{stem} {condition}에서 패치가 입력을 바꾸지 않았습니다.")
                record.update(x=x, y=y, width=w, height=h, coverage=overlap/((tr-tx)*(tb-ty)),
                              patch_hash=hashlib.sha256(patch.tobytes()).hexdigest())
            rows.append(record)
    return pd.DataFrame(rows)


def summarize(detail):
    original = detail[detail.condition == "Original"].set_index("stem")
    paired = detail.join(original[["detected", "confidence", "iou"]].add_prefix("original_"), on="stem")
    paired["lost"] = paired.original_detected & ~paired.detected
    paired["recovered"] = ~paired.original_detected & paired.detected
    paired["delta_conf"] = paired.confidence - paired.original_confidence
    paired["delta_iou"] = paired.iou - paired.original_iou
    summary = paired.groupby("condition", sort=False).agg(
        frames=("stem", "size"), detected=("detected", "sum"), lost=("lost", "sum"),
        recovered=("recovered", "sum"), floor_fp=("floor_fp", "sum"), paired_matches=("delta_conf", "count"),
        median_delta_conf=("delta_conf", "median"), median_delta_iou=("delta_iou", "median"))
    return paired, summary
