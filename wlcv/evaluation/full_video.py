"""전체 영상을 처음부터 읽고, 정답과 독립적으로 두 방법의 원판 후보를 선택한다."""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import EVENTS, load_events
from wlcv.config import ROOT, SPLIT, VIDEOS
from wlcv.data.frames import frames_rgb
from wlcv.evaluation.plate_track import BOX, KEY, select_centers
from wlcv.evaluation.plots import predict_rgb


def detect_videos(weights, video_ids):
    """신뢰도 0.001 이상의 후보를 보관한다. 후보가 없는 프레임도 프레임 표에 남긴다."""
    model = YOLO(str(Path(weights).resolve(strict=True)))
    if model.names != {0: "plate"}:
        raise ValueError("원판 탐지 가중치가 필요합니다.")
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    trained = set(pd.read_csv(weights.parent.parent / "train_people.csv").person_id)
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    frames, candidates = [], []
    for video in video_ids:
        if split.at[video, "role"] != "validation" or split.at[video, "person_id"] in trained:
            raise ValueError(f"{video}의 학습·검증 분할을 확인하세요.")
        expected = int(metadata.at[video, "frame_count"])
        decoded = 0
        for index, rgb in enumerate(tqdm(frames_rgb(VIDEOS / f"{video}.mp4"),
                                         total=expected, desc=video, unit="장")):
            frames.append((video, index))
            boxes = predict_rgb(model, rgb, conf=0.001, imgsz=640, device="cpu").boxes.cpu()
            for conf, coords in zip(boxes.conf.tolist(), boxes.xyxy.tolist()):
                candidates.append((video, index, conf, *coords))
            decoded += 1
        if decoded != expected:
            raise ValueError(f"{video}의 실제 프레임 수 {decoded}와 기록 {expected}가 다릅니다.")
    return pd.DataFrame(frames, columns=KEY), pd.DataFrame(candidates, columns=KEY + ["confidence"] + BOX)


def compare_videos(frames, candidates, review, *, gate_high_history=False, high_checks=None):
    """선택을 끝낸 뒤 검수 박스와 비교한다. 판독 불가와 미탐을 구분한다."""
    checks = []
    selected = select_centers(frames, candidates, distance_ratio=0.5, checks=checks,
                              gate_high_history=gate_high_history, high_checks=high_checks)
    truth = review.rename(columns={col: "gt_" + col for col in BOX})
    d = selected.merge(truth, on=KEY, how="left", validate="many_to_one", indicator=True)
    if not d["_merge"].eq("both").all() or len(review) != len(frames):
        raise ValueError("평가용 검수표와 디코딩한 프레임이 일치해야 합니다.")
    d = d.drop(columns="_merge")
    counts = candidates.groupby(KEY).size()
    d["candidate_count"] = d.join(counts.rename("n"), on=KEY).n.fillna(0).astype(int)
    width = (np.minimum(d.x2, d.gt_x2) - np.maximum(d.x1, d.gt_x1)).clip(lower=0)
    height = (np.minimum(d.y2, d.gt_y2) - np.maximum(d.y1, d.gt_y1)).clip(lower=0)
    overlap = width * height
    area = (d.x2 - d.x1) * (d.y2 - d.y1)
    target_area = (d.gt_x2 - d.gt_x1) * (d.gt_y2 - d.gt_y1)
    d["iou"] = overlap / (area + target_area - overlap)
    d["match"] = d.iou.ge(.5).astype("boolean").mask(d.gt_x1.isna())
    checks = pd.DataFrame(checks, columns=KEY + ["distance_ratio", "expected_x", "expected_y",
                          "last_width", "radius_px", "nearest_px", "candidates", "inside"])
    return d, checks


def summarize_video_scope(detail):
    """연속 미탐은 같은 영상·평가 구간 안에서만 센다. 판독 불가이면 연속 길이를 끊는다."""
    rows = []
    for (v, s, m), group in detail.groupby(["video_id", "scope", "method"], sort=False):
        group = group.sort_values("frame_index")
        missing = group.match.eq(False).fillna(False)
        lengths = missing.groupby((~missing).cumsum()).sum()
        linked = group.source.eq("linked")
        rows.append(dict(video_id=v, scope=s, method=m, frames=len(group),
                         scored=group.match.count(), hits=group.match.sum(), missed=missing.sum(),
                         longest_miss=int(lengths.max()), linked=linked.sum(),
                         recovered=(linked & group.match).sum(), wrong_link=(linked & missing).sum(),
                         unreadable_link=(linked & group.match.isna()).sum()))
    return pd.DataFrame(rows)


def summarize_phase_false_positives(detail, annotations):
    """전체 영상에서 방법별로 최종 선택한 원판 박스의 오탐을 동작 구간별로 센다."""
    phases = ["동작 중", "ready 전", "finish 후", "구간 미확인"]
    groups = []
    for video, frames in detail.groupby("video_id", sort=False):
        events = load_events(annotations, video, pass_number=1)
        known = [events[name] for name in EVENTS if name in events]
        if known != sorted(known):
            raise ValueError(f"{video}의 동작 이벤트 순서를 확인하세요.")

        frames = frames.copy()
        index = frames["frame_index"]
        frames["phase"] = "구간 미확인"
        if known:
            frames.loc[index.between(known[0], known[-1]), "phase"] = "동작 중"
        if "ready" in events:
            frames.loc[index < events["ready"], "phase"] = "ready 전"
        if "finish" in events:
            frames.loc[index > events["finish"], "phase"] = "finish 후"
        groups.append(frames)

    frames = pd.concat(groups, ignore_index=True)
    frames["phase"] = pd.Categorical(frames["phase"], categories=phases, ordered=True)
    frames["false_positive"] = frames.x1.notna() & frames.match.eq(False).fillna(False)
    summary = frames.groupby(["method", "phase"], observed=False).agg(
        frames=("frame_index", "size"),
        scored=("match", "count"),
        fp_frames=("false_positive", "sum"),
    ).reset_index()
    denominator = summary.scored.where(summary.scored > 0)
    summary["fp_percent"] = 100 * summary.fp_frames / denominator
    return summary
