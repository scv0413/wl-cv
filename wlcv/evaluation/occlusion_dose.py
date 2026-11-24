"""같은 질감의 높이만 늘려 원판 아래쪽과 중앙에서 탐지 후보를 비교한다."""
import hashlib

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics.utils.metrics import box_iou

from wlcv.evaluation.patch_position import make_example, paste_patch
from wlcv.evaluation.plots import predict_rgb

DOSES = (0, 10, 20, 25)
POSITIONS = ("Center", "Lower")
PREVIOUS_NAMES = {"Center": "Center strip", "Lower": "Lower strip"}


def make_doses(truth, variants):
    """0%는 원본이다. 두 위치에 같은 패치 조각을 사용한다."""
    x1, y1, x2, y2 = np.rint(truth).astype(int)
    area = (x2 - x1) * (y2 - y1)
    left, _, strip = variants["Lower strip"]
    max_height, width = strip.shape[:2]
    placements = {}
    for dose in DOSES[1:]:
        height = max_height if dose == 25 else round(dose / 100 * area / width)
        if not 0 < height <= max_height:
            raise ValueError(f"{dose}% 높이를 기존 25% 패치에서 잘라낼 수 없습니다.")
        patch = strip[-height:]  # 아래쪽 고정 픽셀을 보존하면서 윗줄만 추가한다.
        placements["Lower", dose] = (left, y2 - height, patch)
        placements["Center", dose] = (left, (y1 + y2 - height) // 2, patch)
    return placements, area


def check_endpoints(floors, dataset, previous):
    """추론 전에 25%의 재료, 위치, 크기, 실제 픽셀 해시를 지난 두 실행과 대조한다."""
    endpoints = previous[previous.condition.isin(PREVIOUS_NAMES.values())]
    old = endpoints.set_index(["model", "stem", "condition"])
    if len(old) != 4 * len(floors) or not old.index.is_unique:
        raise ValueError("모델별 중앙·아래쪽 대조 기록을 확인하세요.")
    for stem in floors.index:
        _, truth, _, donor, variants = make_example(dataset, floors, stem)
        placements, area = make_doses(truth, variants)
        for position in POSITIONS:
            x, y, patch = placements[position, 25]
            h, w = patch.shape[:2]
            digest = hashlib.sha256(patch.tobytes()).hexdigest()
            for name in ("Baseline", "Texture"):
                row = old.loc[name, stem, PREVIOUS_NAMES[position]]
                if ((donor, x, y, w, h, digest) != (row.donor, row.x, row.y, row.width, row.height, row.patch_hash)
                        or not np.isclose(w*h/area, row.coverage)):
                    raise ValueError(f"{stem} {position} 25%가 보관된 입력과 다릅니다.")


def score_frame(model, image, truth):
    """정답 IoU 0.50 이상인 후보의 최고 신뢰도. 해당 후보가 없으면 결측이다."""
    boxes = predict_rgb(model, image, imgsz=640, conf=0.001, device="cpu").boxes.cpu()
    overlaps = box_iou(torch.tensor([truth], dtype=torch.float32), boxes.xyxy)[0]
    matches = torch.where((boxes.cls == 0) & (overlaps >= 0.50))[0]
    if not len(matches):
        return np.nan, np.nan
    best = matches[boxes.conf[matches].argmax()]
    return float(boxes.conf[best]), float(overlaps[best])


def evaluate_doses(model, floors, dataset):
    rows = []
    for stem in tqdm(floors.index, desc="가림 비율 비교", unit="장"):
        image, truth, _, donor, variants = make_example(dataset, floors, stem)
        placements, area = make_doses(truth, variants)
        clean_score, clean_iou = score_frame(model, image, truth)
        for position in POSITIONS:
            for dose in DOSES:
                x = y = None
                h = w = 0
                digest = None
                confidence, iou = clean_score, clean_iou
                if dose:
                    x, y, patch = placements[position, dose]
                    h, w = patch.shape[:2]
                    digest = hashlib.sha256(patch.tobytes()).hexdigest()
                    confidence, iou = score_frame(model, paste_patch(image, (x, y, patch)), truth)
                rows.append(dict(stem=stem, person=floors.at[stem, "person"], frame=int(floors.at[stem, "frame"]),
                                 donor=donor, position=position, dose=dose, actual_pct=100*w*h/area,
                                 x=x, y=y, width=w, height=h, patch_hash=digest,
                                 confidence=confidence, iou=iou, detected=confidence >= 0.10))
    return pd.DataFrame(rows)


def verify_scores(detail, previous):
    """원본과 25%에서 저장된 두 모델의 예측을 다시 얻었는지 확인한다."""
    for position in POSITIONS:
        for dose, condition in [(0, "Original"), (25, PREVIOUS_NAMES[position])]:
            new = detail[(detail.position == position) & (detail.dose == dose)]
            old = previous[previous.condition == condition]
            paired = new.merge(old, on=["model", "stem"], suffixes=("_new", "_old"),
                               validate="one_to_one")
            if len(paired) != len(new):
                raise ValueError(f"{position} {dose}%의 이전 기록이 부족합니다.")
            for column in ["confidence", "iou", "detected"]:
                np.testing.assert_allclose(paired[column + "_new"], paired[column + "_old"],
                                           rtol=0, atol=1e-6, equal_nan=True)


def mark_states(detail, previous_misses):
    """0.001 후보 하한과 0.10 판정 기준을 구별하고, 바로 전 단계와 비교한다."""
    keys = ["model", "stem", "position"]
    data = detail.sort_values(keys + ["dose"]).copy()
    data["prior_miss"] = data.stem.isin(previous_misses)
    data["matched"] = data.confidence.notna()
    data["state"] = np.select([~data.matched, data.detected], ["no_match", "detected"],
                              default="low_confidence")
    before = data.groupby(keys).matched.shift()
    data["lost_match"] = before.eq(True) & ~data.matched
    data["regained_match"] = before.eq(False) & data.matched
    return data


def count_frames(data):
    table = data.groupby(["model", "position", "dose"]).agg(
        frames=("stem", "size"), matched=("matched", "sum"), detected=("detected", "sum"),
        lost_match=("lost_match", "sum"), regained_match=("regained_match", "sum"))
    table["low_confidence"] = table.matched - table.detected
    table["no_match"] = table.frames - table.matched
    return table
