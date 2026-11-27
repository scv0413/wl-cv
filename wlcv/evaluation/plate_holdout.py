"""고정한 테스트 판독과 Mac 배포 파이프라인이 저장한 frames.csv를 비교한다."""
import argparse
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wlcv.annotation.storage import box_path, load_events
from wlcv.config import ANNOTATIONS, ROOT, SPLIT
from wlcv.data.plate_labels import read_boxes
from wlcv.runtime.tracking import iou

BOX = ["plate_x1", "plate_y1", "plate_x2", "plate_y2"]
REVIEW = ANNOTATIONS / "plate_holdout"


def score_samples(video, rows, targets, width, height):
    results = []
    for frame in targets:
        row = rows.loc[frame]
        pred = row[BOX].to_numpy(dtype=float)
        present = np.isfinite(pred).all()
        label = box_path(REVIEW, video, frame)
        uncertain = REVIEW / "판독불가" / f"{video}_f{frame}.txt"
        boxes = [] if uncertain.exists() else read_boxes(label, inside_frame=True)
        if len(boxes) > 1:
            raise ValueError(f"{video} {frame}번에서 가까운 원판 하나를 확인하세요.")
        overlap = np.nan
        if boxes and present:
            x, y, w, h = boxes[0]
            truth = np.array([x-w/2, y-h/2, x+w/2, y+h/2]) * [width, height, width, height]
            overlap = iou(pred, truth)
        match = bool(overlap >= 0.5)
        scored = not uncertain.exists()
        results.append(dict(video_id=video, frame=frame, scored=scored, gt_present=bool(boxes),
                            plate_status=row.plate_status, model=row.selected_model, iou=overlap,
                            tp=int(match), fn=int(scored and bool(boxes) and not match),
                            fp=int(scored and present and not match)))
    return pd.DataFrame(results)


def missing_ranges(rows):
    missing = rows.plate_cx_px.isna() | rows.plate_cy_px.isna()
    ranges = []
    for is_missing, frames in groupby(rows.index, key=missing.__getitem__):
        if is_missing:
            run = list(frames)
            ranges.append((run[0], run[-1]))
    return ranges


def read_indexed(path, key):
    table = pd.read_csv(path).set_index(key)
    if not table.index.is_unique:
        raise ValueError(f"{path.name}의 {key}가 중복되었습니다.")
    return table


def score_run(folder):
    cohort = read_indexed(SPLIT.with_name("plate_holdout_videos.csv"), "video_id")
    split = read_indexed(SPLIT, "video_id")
    runs = read_indexed(folder / "runs.csv", "video_id")
    test = split.loc[split.role.eq("test")]
    if set(test.index) != set(cohort.index) or set(runs.index) != set(cohort.index):
        raise ValueError("test 분할, 고정 원본 목록, 추론한 영상이 일치하지 않습니다.")
    sizes = pd.read_csv(ROOT / "data/metadata/video_timing.csv").set_index("video_id")
    per_video, samples, gaps = [], [], []
    for video, item in cohort.iterrows():
        if item.person_id != test.at[video, "person_id"]:
            raise ValueError(f"{video}의 테스트 선수 ID가 다릅니다.")
        dest = folder / runs.at[video, "run"]
        summary = json.loads((dest / "summary.json").read_text())
        rows = read_indexed(dest / "frames.csv", "frame_index")
        if (list(rows.index) != list(range(int(item.frames))) or not rows.video_id.eq(video).all()
                or summary["plate_mode"] != "cutmix" or summary["plate_imgsz"] != 640
                or summary["pose_imgsz"] != 640 or summary["device"] != "cpu"
                or summary["read_mode"] != "single-pass"):
            raise ValueError(f"{video}의 추론 설정 또는 프레임 번호가 다릅니다.")
        coords = rows[[*BOX, "plate_cx_px", "plate_cy_px"]].notna()
        if not (coords.all(axis=1) | ~coords.any(axis=1)).all():
            raise ValueError(f"{video}의 박스 또는 중심 좌표 중 일부만 비어 있습니다.")
        events = load_events(REVIEW, video, 1)
        start = events["ready"]
        last_key = "drive" if "_h_" in video else "finish"
        end = events[last_key]
        if ((start, end) != (summary["manual_events"]["lift_off"], summary["manual_events"][last_key])
                or not 0 <= start < end < len(rows)):
            raise ValueError(f"{video}의 추론과 판독 구간이 다릅니다.")
        motion = rows.loc[start:end]
        targets = sorted({start + round((end-start)*i/4) for i in range(5)})
        size = sizes.loc[video]
        detail = score_samples(video, rows, targets, size.width, size.height)
        samples.append(detail)
        result = dict(person_id=item.person_id, video_id=video, frames=len(rows), motion_frames=len(motion),
                      start=start, end=end, scored=int(detail.scored.sum()), tp=int(detail.tp.sum()),
                      fn=int(detail.fn.sum()), fp=int(detail.fp.sum()))
        for scope, region in (("motion", motion), ("all", rows)):
            ranges = missing_ranges(region)
            lengths = [b-a+1 for a,b in ranges]
            result[scope+"_missing"] = sum(lengths)
            result[scope+"_longest"] = max(lengths, default=0)
            gaps.extend((item.person_id, video, scope, a, b, b-a+1) for a,b in ranges)
        per_video.append(result)
    videos = pd.DataFrame(per_video)
    totals = ["frames", "motion_frames", "scored", "tp", "fn", "fp", "motion_missing", "all_missing"]
    metrics = dict.fromkeys(totals, "sum")
    metrics.update(motion_longest="max", all_longest="max")
    by_person = videos.groupby("person_id")[list(metrics)].agg(metrics)
    pd.concat(samples).to_csv(folder / "scored_samples.csv", index=False)
    videos.to_csv(folder / "by_video.csv", index=False)
    by_person.to_csv(folder / "by_person.csv")
    pd.DataFrame(gaps, columns=["person_id","video_id","scope","start","end","frames"]).to_csv(
        folder / "missing_ranges.csv", index=False)
    return by_person


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="runs.csv와 영상별 분석 결과가 있는 폴더")
    print(score_run(parser.parse_args().folder).to_string())
