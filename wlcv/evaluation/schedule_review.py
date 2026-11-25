"""두 best.pt를 같은 검증 사진과 연속 영상에서 판정한다."""

import numpy as np
import pandas as pd
from ultralytics import YOLO

from wlcv.annotation.hub_points import read_hubs
from wlcv.annotation.storage import EVENTS, load_events
from wlcv.config import ANNOTATIONS, ROOT
from wlcv.data.plate_labels import label_frame, read_boxes
from wlcv.evaluation.full_video import compare_videos, detect_videos
from wlcv.evaluation.plate_track import BOX, KEY

METHODS = {"YOLO": "직접 검출", "Past link": "시간 연결 적용"}


def review_inputs(data):
    """고정 검증 사진의 박스와 추가로 검수된 원본 프레임을 영상·프레임 번호로 합친다."""
    sizes = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    original = []
    for path in sorted((data.parent / "labels/val").glob("*.txt")):
        video, index = label_frame(path)
        (cx, cy, bw, bh), = read_boxes(path)
        w, h = sizes.at[video, "width"], sizes.at[video, "height"]
        original.append((video, index, (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    fixed = pd.DataFrame(original, columns=KEY + BOX)
    videos = sorted(fixed.video_id.unique())
    extra = pd.read_csv(ANNOTATIONS / "additional_temporal_review.csv")
    dense = pd.read_csv(ANNOTATIONS / "b02_c07_plate_review.csv")
    sparse = pd.read_csv(ANNOTATIONS / "temporal_review_boxes.csv")
    boxes = pd.concat([fixed, sparse, extra, dense], ignore_index=True)[KEY + BOX]
    boxes = boxes[boxes.video_id.isin(videos)].drop_duplicates(KEY, keep="first")
    hubs = pd.concat([pd.read_csv(ANNOTATIONS / "temporal_plate_centers.csv"),
                      extra, read_hubs()], ignore_index=True)
    hubs = hubs[hubs.video_id.isin(videos)][KEY + ["visibility", "manual_x", "manual_y"]]
    for col in ('manual_x', 'manual_y'):
        hubs[col] = pd.to_numeric(hubs[col])
    return videos, boxes, hubs


def evaluate_run(run, data, output):
    weights = run / "weights/best.pt"
    model = YOLO(str(weights.resolve(strict=True)))
    metric = model.val(data=str(data), split="val", imgsz=640, batch=16, conf=0.001,
                       iou=0.7, max_det=300, device="cpu", workers=0, plots=False,
                       project=str(output), name=run.name)
    return float(metric.box.map), evaluate_motion(weights, data)


def evaluate_motion(weights, data):
    """검증 원본 영상에서 직접 검출과 시간 연결 결과를 수동 박스·허브와 대조한다."""
    videos, boxes, hubs = review_inputs(data)
    frames, candidates = detect_videos(weights, videos)
    review = frames.merge(boxes, on=KEY, how="left", validate="one_to_one")
    detail, _ = compare_videos(frames, candidates, review)
    detail = detail.merge(hubs, on=KEY, how="left", validate="many_to_one")
    detail["in_motion"] = False
    for video in videos:
        events = load_events(ANNOTATIONS, video, pass_number=1)
        # 하프 동작은 finish가 없으므로 마지막으로 확인된 이벤트까지만 포함한다.
        if "ready" in events:
            last = max(events[k] for k in EVENTS if k in events)
            in_range = detail.frame_index.between(events["ready"], last)
            detail.loc[detail.video_id.eq(video) & in_range, "in_motion"] = True

    visible = detail.visibility.eq("visible")
    detail["dy_px"] = (detail.cy - detail.manual_y).where(visible)
    detail["error_px"] = np.hypot(detail.cx - detail.manual_x, detail.dy_px).where(visible)
    return detail


def compare_runs(control, trial, *, labels=("100", "200")):
    """같은 프레임·후처리에서 어느 모델이 정탐을 얻고 잃었는지 연결한다."""
    keys = KEY + ["method"]
    fixed = ["in_motion", "visibility", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "manual_x", "manual_y"]
    measured = ["source", "confidence", "match", "iou", "error_px", "dy_px"] + BOX
    first, second = labels
    left = control[keys + fixed + measured]
    pair = left.merge(trial[keys + measured], on=keys, validate="one_to_one",
                      suffixes=(f"_{first}", f"_{second}"), how="outer", indicator=True)
    if not pair.pop("_merge").eq("both").all():
        raise ValueError("두 실행의 프레임 목록을 확인하세요.")
    match_first = f"match_{first}"
    match_second = f"match_{second}"
    valid = pair.in_motion & pair[match_first].notna() & pair[match_second].notna()
    result = pair.loc[valid].copy()
    result["change"] = "모두 미탐"
    result.loc[result[match_first] & result[match_second], "change"] = "모두 정탐"
    result.loc[~result[match_first] & result[match_second], "change"] = "복구"
    result.loc[result[match_first] & ~result[match_second], "change"] = "손실"
    return result


def hub_summary(pair):
    """같은 수동점에서 둘 다 정탐인 쌍의 오차를 비교하고, 빠진 프레임 수도 남긴다."""
    rows = []
    known = pair[pair.visibility.eq("visible")]
    for (video, method), group in known.groupby(["video_id", "method"]):
        both = group[group.match_100 & group.match_200]
        rows.append({"영상": video, "방식": METHODS[method], "직접 판독": len(group),
                     "100 정탐": int(group.match_100.sum()), "200 정탐": int(group.match_200.sum()),
                     "공통 정탐": len(both), "100 중앙오차(px)": both.error_px_100.median(),
                     "200 중앙오차(px)": both.error_px_200.median()})
    return pd.DataFrame(rows)
