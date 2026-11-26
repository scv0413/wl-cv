"""배포용 원판 추적기를 동일한 영상에 적용해 단독·보조·코사인을 비교한다."""
import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from tqdm.auto import tqdm

from wlcv.annotation.storage import EVENTS, load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT, VIDEOS
from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.evaluation.plate_edge_filter import WEIGHTS
from wlcv.evaluation.plate_lr_checkpoints import CONTROL as COSINE_RUN
from wlcv.evaluation.plate_lr_review import save_false_positives
from wlcv.evaluation.plate_patience_review import measure
from wlcv.evaluation.plate_track import BOX, KEY
from wlcv.evaluation.schedule_review import review_inputs
from wlcv.evaluation.single_pass import tracked_plate_frames

MODES = {
    "mac_single": (WEIGHTS["mac"], None),
    "mac_cutmix": (WEIGHTS["mac"], WEIGHTS["cutmix"]),
    "cosine_single": (COSINE_RUN / "weights/best.pt", None),
}


def track_videos(mode, videos, metadata):
    primary, auxiliary = MODES[mode]
    model = validation_plate_model(primary, videos)
    helper = validation_plate_model(auxiliary, videos) if auxiliary else None
    rows, times = [], []
    for video in tqdm(videos, desc=mode, unit="편"):
        path = VIDEOS / metadata.at[video, "filename"]
        count = int(metadata.at[video, "frame_count"])
        timing = {"plate_inference": 0.0, "plate_tracking": 0.0}
        start = perf_counter()
        stream = tracked_plate_frames(path, model, "cpu", timing, cutmix_model=helper,
                                      plate_imgsz=640, edge_rule="continuity")
        clip = [vars(plate) for _, _, plate in stream]
        elapsed = perf_counter() - start
        if [(r["video_id"], r["frame_index"]) for r in clip] != [(video, i) for i in range(count)]:
            raise ValueError(f"{video}의 디코딩 결과와 메타데이터가 다릅니다.")
        times.append(dict(mode=mode, video_id=video, frames=count, seconds=elapsed, **timing))
        rows.extend(clip)
    tracks = pd.DataFrame(rows)
    tracks["selected_model"] = tracks.selected_model.replace(
        {"baseline": "cosine" if mode == "cosine_single" else "mac"})
    return tracks, times


def score(tracks, boxes, videos):
    truth = boxes.rename(columns={col: "gt_" + col for col in BOX})
    detail = tracks.merge(truth, on=KEY, how="left", validate="one_to_one")
    detail["iou"] = [iou(box, target) for box, target in zip(
        detail[BOX].to_numpy(), detail[["gt_" + c for c in BOX]].to_numpy())]
    detail["match"] = detail.iou.ge(0.5).astype("boolean").mask(detail.gt_x1.isna())
    detail["method"] = "Past link"
    detail["in_motion"] = False
    for video in videos:
        events = load_events(ANNOTATIONS, video, pass_number=1)
        last = max(events[k] for k in EVENTS if k in events)
        detail.loc[detail.video_id.eq(video) & detail.frame_index.between(events["ready"], last), "in_motion"] = True
    return detail


def compare(output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    videos, boxes, _ = review_inputs(DATASET / "data.yaml")
    frames, people, times = [], [], []
    for mode in MODES:
        tracks, elapsed = track_videos(mode, videos, metadata)
        detail = score(tracks, boxes, videos)
        athletes = measure(detail)
        motion = detail[detail.in_motion]
        wrong = motion.x1.notna() & motion.match.eq(False).fillna(False)
        athletes["false_positives"] = athletes.video_id.map(wrong.groupby(motion.video_id).sum())
        times.extend(elapsed)
        frames.append(detail.assign(mode=mode))
        people.append(athletes.assign(mode=mode))

    records = pd.concat(frames, ignore_index=True)
    athletes = pd.concat(people, ignore_index=True)
    shared = athletes.pivot(index="video_id", columns="mode", values="abs_rise_error_px").dropna().index
    athletes["common_error"] = athletes.abs_rise_error_px.where(athletes.video_id.isin(shared))
    summary = athletes.groupby("mode", sort=False).agg(
        scored=("scored_boxes","sum"), missed=("misses","sum"), fp=("false_positives","sum"),
        motion=("motion_frames","sum"), missing=("gap_frames","sum"), longest=("longest_gap","max"),
        measured=("abs_rise_error_px","count"), common=("common_error","count"), rise_mae=("common_error","mean"))
    records.to_csv(output / "frames.csv", index=False)
    save_false_positives(records.rename(columns={"mode": "case"}), output)
    athletes.to_csv(output / "athletes.csv", index=False)
    summary.to_csv(output / "summary.csv")
    pd.DataFrame(times).to_csv(output / "timing.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    torch.set_num_threads(2)
    print(compare(parser.parse_args().output).round(3))
