"""같은 원판 추적을 독립적으로 재생하고 정해진 순서로 빈 프레임만 채운다."""
import argparse
import hashlib
from time import perf_counter

import pandas as pd
import torch
from tqdm.auto import tqdm

from wlcv.config import DATASET, ROOT, RUNS, VIDEOS
from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.evaluation.plate_deployment_comparison import score
from wlcv.evaluation.plate_edge_filter import BACKGROUNDS, WEIGHTS
from wlcv.evaluation.plate_lr_checkpoints import CONTROL as COSINE
from wlcv.evaluation.plate_lr_review import save_false_positives
from wlcv.evaluation.plate_patience_review import CONTROL as SCHEDULE, LABELS, measure
from wlcv.evaluation.plate_track import KEY
from wlcv.evaluation.schedule_review import review_inputs
from wlcv.evaluation.single_pass import tracked_plate_frames

MODELS = {
    "cosine": COSINE / "weights/best.pt",
    **WEIGHTS,
    "gray": RUNS / "plate_occlusion/weights/best.pt",
    "texture": RUNS / "plate_occlusion_texture/weights/best.pt",
    "scale_off": RUNS / "plate_scale_off/weights/best.pt",
    "scale025": ROOT / "outputs/plate_scale_cutmix/runs/control/weights/best.pt",
    "hsv_v_off": RUNS / "plate_hsv_v_off/weights/best.pt",
    "schedule": SCHEDULE / "weights/best.pt",
}
OUTPUT = ROOT / "outputs/plate_cosine_fallback"


def infer(output):
    """추적 이력은 모델·영상마다 새로 시작한다. b05와 b06은 별도 학습 배경이다."""
    val, _, _ = review_inputs(DATASET / "data.yaml")
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    for video in BACKGROUNDS:
        labels = list((DATASET / "labels/train").glob(f"{video}_f*.txt"))
        if not labels or any(read_boxes(path) for path in labels):
            raise ValueError(f"{video}의 빈 배경 라벨을 확인하세요.")

    folder = output / "tracks"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, weights in MODELS.items():
        model = validation_plate_model(weights, val)
        rows = []
        start = perf_counter()
        for video in tqdm(val + BACKGROUNDS, desc=name, unit="편"):
            path = VIDEOS / metadata.at[video, "filename"]
            timing = {"plate_inference": 0.0, "plate_tracking": 0.0}
            frames = tracked_plate_frames(path, model, "cpu", timing, plate_imgsz=640, edge_rule="continuity")
            clip = [vars(plate) for _, _, plate in frames]
            expected = range(int(metadata.at[video, "frame_count"]))
            if [(r["video_id"], r["frame_index"]) for r in clip] != [(video, i) for i in expected]:
                raise ValueError(f"{video}의 영상 길이 또는 프레임 순서가 다릅니다.")
            rows.extend(clip)
        seconds = perf_counter() - start
        tracks = pd.DataFrame(rows)
        tracks["selected_model"] = tracks.selected_model.replace({"baseline": name})
        tracks.to_csv(folder / f"{name}.csv", index=False)
        manifest.append(dict(model=name, weights=str(weights.relative_to(ROOT)),
                             sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), seconds=seconds))
    pd.DataFrame(manifest).to_csv(output / "models.csv", index=False)


def choose_by_priority(tracks, order):
    """각 모델은 이미 자신의 궤적을 갱신했다. 선순위 모델이 빈 프레임만 보완한다."""
    chosen = tracks[order[0]].copy()
    for name in order[1:]:
        other = tracks[name]
        if not chosen.index.equals(other.index):
            raise ValueError(f"{name} 모델의 영상·프레임 목록이 다릅니다.")
        take = chosen.x1.isna() & other.x1.notna()
        chosen.loc[take] = other.loc[take]
    return chosen.reset_index()


def compare(output):
    val, boxes, _ = review_inputs(DATASET / "data.yaml")
    tracks = {}
    for name in MODELS:
        table = pd.read_csv(output / "tracks" / f"{name}.csv", float_precision="round_trip")
        tracks[name] = table.set_index(KEY, verify_integrity=True).sort_index()

    orders = [(name,) for name in MODELS]
    orders.append(("mac", "cutmix"))
    for name in MODELS:
        if name != "cosine":
            orders.extend([("cosine", name), (name, "cosine")])
    orders.extend([("cosine", "mac", "cutmix"), ("mac", "cutmix", "cosine")])

    all_frames, people, backgrounds = [], [], []
    for order in orders:
        case = "+".join(order)
        chosen = choose_by_priority(tracks, order)
        detail = score(chosen[chosen.video_id.isin(val)], boxes, val)
        all_frames.append(detail.assign(case=case))
        athletes = measure(detail)
        people.append(athletes.assign(case=case))
        for video in BACKGROUNDS:
            present = chosen.loc[chosen.video_id.eq(video), "x1"].notna()
            longest = present.groupby((~present).cumsum()).sum().max()
            backgrounds.append(dict(case=case, video=video, frames=len(present),
                                    fp=int(present.sum()), longest=int(longest)))

    frames = pd.concat(all_frames, ignore_index=True)
    athletes = pd.concat(people, ignore_index=True)
    bg = pd.DataFrame(backgrounds)
    matches = frames.loc[frames.in_motion & frames.match.notna()].pivot(
        index=KEY, columns="case", values="match")
    if len(matches) != 270 or matches.isna().any().any():
        raise ValueError("모든 모드가 같은 동작 중 수동 박스 270장을 평가해야 합니다.")
    shared = athletes.pivot(index="video_id", columns="case", values="abs_rise_error_px").dropna().index
    athletes["common_abs_error"] = athletes.abs_rise_error_px.where(athletes.video_id.isin(shared))
    summary = athletes.groupby("case", sort=False).agg(
        scored=("scored_boxes", "sum"), fn=("misses", "sum"), motion=("motion_frames", "sum"),
        gaps=("gap_frames", "sum"), longest=("longest_gap", "max"), rises=("abs_rise_error_px", "count"),
        rise_mae=("abs_rise_error_px", "mean"), common=("common_abs_error", "count"),
        common_rise_mae=("common_abs_error", "mean"))
    on_motion = frames[frames.in_motion]
    wrong = on_motion.x1.notna() & on_motion.match.eq(False).fillna(False)
    summary["fp"] = wrong.groupby(on_motion.case).sum()
    baseline = matches["mac+cutmix"]
    summary["gained"] = matches.loc[~baseline].sum()
    summary["lost"] = (~matches.loc[baseline]).sum()
    summary["bg_fp"] = bg.groupby("case").fp.sum()
    hubs = pd.read_csv(LABELS)
    hubs = hubs[hubs.hub_source.isin(["clicked", "previous_blind"])]
    hub_errors = frames.merge(hubs[KEY + ["manual_y"]], on=KEY, validate="many_to_one")
    hub_errors["vertical_error_px"] = hub_errors.cy - hub_errors.manual_y
    hub_errors[["case", *KEY, "selected_model", "cy", "manual_y", "vertical_error_px"]].to_csv(
        output / "hub_frames.csv", index=False)
    save_false_positives(frames, output)
    frames.to_csv(output / "frames.csv", index=False)
    athletes.to_csv(output / "athletes.csv", index=False)
    bg.to_csv(output / "backgrounds.csv", index=False)
    summary.to_csv(output / "summary.csv")
    return summary


def endpoint_readout(output):
    """수동 시점 12장만 새로 추론한다. 두 장 모두 추적 박스와 겹칠 때만 상승량을 교체한다."""
    from wlcv.data.frames import read_rgb
    from wlcv.evaluation.box_size_review import iou
    from wlcv.evaluation.plate_track import BOX, PlateHistory, choose_plate
    from wlcv.evaluation.plots import predict_rgb

    athletes = pd.read_csv(output / "athletes.csv")
    reference = athletes[athletes.case.eq("mac")]
    frames = pd.read_csv(output / "frames.csv").set_index(["case", *KEY])
    model = validation_plate_model(MODELS["cosine"], reference.video_id)
    points, rises = [], []
    for athlete in reference.itertuples():
        video = athlete.video_id
        by_frame = {}
        for index in [athlete.knee, athlete.contact]:
            image = read_rgb(VIDEOS / f"{video}.mp4", index)
            boxes = predict_rgb(model, image, conf=0.001, imgsz=640, device="cpu").boxes.cpu()
            candidates = pd.DataFrame(boxes.xyxy.tolist(), columns=BOX)
            candidates["confidence"] = boxes.conf.tolist()
            candidates["video_id"], candidates["frame_index"] = video, index
            height, width = image.shape[:2]
            chosen, _ = choose_plate(PlateHistory(), candidates, video, index, width, height, "continuity")
            by_frame[index] = chosen

        for case in ["mac", "mac+cutmix"]:
            cosine_y, baseline_y = [], []
            for index, chosen in by_frame.items():
                prior = frames.loc[(case, video, index)]
                baseline_y.append(prior.cy)
                overlap = float("nan")
                if chosen is not None and pd.notna(prior.cx):
                    overlap = iou(prior[BOX].to_numpy(float), chosen[BOX].to_numpy(float))
                    if overlap >= 0.50:
                        cosine_y.append(chosen.cy)
                points.append(dict(case=case, video_id=video, frame_index=index, inter_model_iou=overlap,
                                   accepted=overlap >= 0.50))
            use_cosine = len(cosine_y) == 2
            first, last = cosine_y if use_cosine else baseline_y
            rise = first - last
            rises.append(dict(case=case, video_id=video, source="cosine" if use_cosine else case,
                              rise_px=rise, manual_rise_px=athlete.manual_rise_px,
                              error_px=rise - athlete.manual_rise_px))
    pd.DataFrame(points).to_csv(output / "endpoint_checks.csv", index=False)
    pd.DataFrame(rises).to_csv(output / "endpoint_rises.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["infer", "compare", "endpoints", "all"], default="all", nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.step in ("all", "infer"):
        infer(OUTPUT)
    if args.step in ("all", "compare"):
        print(compare(OUTPUT).round(2))

    if args.step in ("all", "endpoints"):
        endpoint_readout(OUTPUT)
