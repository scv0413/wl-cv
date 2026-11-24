"""합성 입력의 동일한 탐지 후보를 합성 전 정답과 보이는 부분 정답에 각각 대조한다."""
import hashlib

import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics.utils.metrics import box_iou

from wlcv.evaluation.patch_position import make_example, paste_patch
from wlcv.evaluation.plots import predict_rgb

CONDITIONS = ("Lower strip", "Cross lower edge")
COORDS = ["x1", "y1", "x2", "y2"]


def compare_visible_boxes(model, dataset, floors, visible_csv):
    """후보 한 목록에서 두 IoU를 계산한다. 정답만 바뀌고 신뢰도는 그대로다."""
    labels = pd.read_csv(visible_csv).set_index(["stem", "condition"])
    expected = {(stem, cond) for stem in floors.index for cond in CONDITIONS}
    if not labels.index.is_unique or set(labels.index) != expected:
        raise ValueError("각 검증 프레임과 가림 조건마다 보이는 부분 박스 하나가 필요합니다.")
    predictions, decisions = [], []
    for stem in tqdm(floors.index, desc="두 정답 박스 비교", unit="장"):
        image, original, _, donor, patches = make_example(dataset, floors, stem)
        for condition in CONDITIONS:
            visible = labels.loc[(stem, condition), COORDS].tolist()
            x1, y1, x2, y2 = visible
            if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                raise ValueError(f"{stem} {condition}의 보이는 부분 박스 좌표를 확인하세요.")
            placement = patches[condition]
            x, y, patch = placement
            frame = paste_patch(image, placement)
            result = predict_rgb(model, frame, conf=0.001, imgsz=640, device="cpu")
            boxes = result.boxes.cpu()
            boxes = boxes[boxes.cls == 0]
            truth = torch.tensor([original, visible], dtype=torch.float32)
            ious = box_iou(truth, boxes.xyxy)
            scores = boxes.conf
            at_threshold = scores >= 0.10
            record = dict(stem=stem, person=floors.at[stem, "person"], condition=condition,
                          donor=donor, x=x, y=y, width=patch.shape[1], height=patch.shape[0],
                          patch_hash=hashlib.sha256(patch.tobytes()).hexdigest(),
                          candidates=len(scores), above_threshold=int(at_threshold.sum()))
            for label_name, overlaps in zip(["original", "visible"], ious):
                matching = torch.where(overlaps >= 0.50)[0]
                hit = False
                candidate_id, confidence, iou = None, None, None
                if len(matching):
                    candidate_id = int(matching[scores[matching].argmax()])
                    confidence = float(scores[candidate_id])
                    iou = float(overlaps[candidate_id])
                    hit = confidence >= 0.10
                record.update({f"{label_name}_hit": hit, f"{label_name}_candidate": candidate_id,
                               f"{label_name}_score": confidence, f"{label_name}_iou": iou})
            # 두 정답 중 하나에라도 맞는 실제 출력 중 최고 점수 후보를 고정해 IoU 이동을 비교한다.
            union = torch.where(at_threshold & (ious.max(dim=0).values >= 0.50))[0]
            record.update(fixed_candidate=None, fixed_original_iou=None, fixed_visible_iou=None)
            if len(union):
                idx = int(union[scores[union].argmax()])
                record.update(fixed_candidate=idx, fixed_original_iou=float(ious[0,idx]),
                              fixed_visible_iou=float(ious[1,idx]))
            decisions.append(record)
            for idx, (rect, conf) in enumerate(zip(boxes.xyxy.tolist(), scores.tolist())):
                predictions.append((stem, condition, idx, *rect, conf, *ious[:,idx].tolist()))
    columns = ["stem", "condition", "candidate", *COORDS, "confidence", "original_iou", "visible_iou"]
    decisions = pd.DataFrame(decisions)
    decisions["gained"] = ~decisions.original_hit & decisions.visible_hit
    decisions["lost"] = decisions.original_hit & ~decisions.visible_hit
    decisions["below_both"] = (~decisions.original_hit & ~decisions.visible_hit
                               & decisions[["original_score", "visible_score"]].notna().any(axis=1))
    decisions["no_match_at_floor"] = decisions[["original_score", "visible_score"]].isna().all(axis=1)
    decisions["iou_change"] = decisions.fixed_visible_iou - decisions.fixed_original_iou
    return decisions, pd.DataFrame(predictions, columns=columns)


def summarize_comparison(results):
    return results.groupby(["model", "condition"], sort=False).agg(
        frames=("stem", "size"), original_tp=("original_hit", "sum"), visible_tp=("visible_hit", "sum"),
        gained=("gained", "sum"), lost=("lost", "sum"), low_score_match=("below_both", "sum"),
        no_match_at_0001=("no_match_at_floor", "sum"), fixed_n=("fixed_candidate", "count"),
        median_iou_change=("iou_change", "median"))
