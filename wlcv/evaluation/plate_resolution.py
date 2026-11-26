"""같은 원본 프레임에서 입력 크기만 바꿔 원판 예측과 소요 시간을 비교한다."""
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import ultralytics
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.plate_blur import REVIEW_FILE, review_frames
from wlcv.config import ANNOTATIONS, DATASET, ROOT, RUNS, SPLIT, VIDEOS
from wlcv.data.frames import frames_rgb, read_rgb
from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation.full_video import compare_videos
from wlcv.evaluation.plate_track import BOX, KEY
from wlcv.evaluation.plots import predict_rgb

SIZES = (640, 800)
WEIGHTS = RUNS / "plate_baseline/weights/best.pt"


def review_targets():
    """선명도·허브는 최초 판독본, 박스는 고정 검증 라벨과 연속 프레임 검수본에서 읽는다."""
    review = pd.read_csv(REVIEW_FILE)
    if set(map(tuple, review[KEY].values)) != set(review_frames()) or review.duplicated(KEY).any():
        raise ValueError("블라인드 판독과 구간의 영상·프레임 번호를 확인하세요.")
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    trained = set(pd.read_csv(WEIGHTS.parent.parent / "train_people.csv").person_id)
    tables = []
    for name in ("temporal_review_boxes", "additional_temporal_review", "b02_c07_plate_review"):
        data = pd.read_csv(ANNOTATIONS / f"{name}.csv")[KEY + BOX].dropna()
        tables.append(data.assign(box_source=name))
    backup = pd.concat(tables).set_index(KEY)
    rows = []
    for row in review.itertuples():
        v, i = row.video_id, row.frame_index
        person = split.at[v, "person_id"]
        if split.at[v, "role"] != "validation" or person in trained:
            raise ValueError(f"{v}의 사람별 분할을 확인하세요.")
        w, h = meta.at[v, "width"], meta.at[v, "height"]
        label = DATASET / "labels/val" / f"{v}_f{i}.txt"
        if label.exists():
            (cx, cy, bw, bh), = read_boxes(label)
            coords = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
            source = "baseline_val"
        else:
            other = backup.loc[(v, i), :]
            coords, source = other[BOX].tolist(), other.box_source
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f"{v} {i}번 검수 박스가 원본 크기를 벗어납니다.")
        rows.append((v, i, person, source, *coords))
    return review.merge(pd.DataFrame(rows, columns=KEY + ["person_id", "box_source"] + BOX),
                        on=KEY, validate="one_to_one")


def run_resolution(output):
    """영상을 처음부터 읽는다. 매 프레임의 두 호출 순서를 번갈아 Mac CPU 시간을 잰다."""
    targets = review_targets()
    output.mkdir(parents=True, exist_ok=False)
    models = {size: YOLO(str(WEIGHTS.resolve(strict=True))) for size in SIZES}
    if models[640].names != {0: "plate"}:
        raise ValueError("베이스라인 원판 모델을 확인하세요.")
    frame_counts = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id").frame_count
    timings, boxes = [], []
    for video in sorted(targets.video_id.unique()):
        count = 0
        for i, rgb in enumerate(tqdm(frames_rgb(VIDEOS / f"{video}.mp4"),
                                     total=frame_counts[video], desc=video, unit="장")):
            if i == 0:
                for size in SIZES:
                    for _ in range(3):
                        predict_rgb(models[size], rgb, conf=.001, imgsz=size, device="cpu")
            for size in (SIZES if i % 2 == 0 else reversed(SIZES)):
                start = perf_counter()
                result = predict_rgb(models[size], rgb, conf=.001, imgsz=size, device="cpu")
                elapsed = (perf_counter() - start) * 1000
                timings.append((video, i, size, elapsed))
                for conf, xyxy in zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
                    boxes.append((video, i, size, conf, *xyxy))
            count += 1
        if count != frame_counts[video]:
            raise ValueError(f"{video}에서 {count}장만 읽었습니다.")
    times = pd.DataFrame(timings, columns=KEY + ["imgsz", "ms"])
    times.to_csv(output / "timings.csv", index=False)
    candidates = pd.DataFrame(boxes, columns=KEY + ["imgsz", "confidence"] + BOX)
    frames = times[KEY].drop_duplicates()
    truth = frames.merge(targets, on=KEY, how="left", validate="one_to_one")
    results = []
    for size in SIZES:
        detail, _ = compare_videos(frames, candidates[candidates.imgsz.eq(size)], truth)
        detail = detail[detail.person_id.notna()].copy()
        detail["imgsz"] = size
        detail["dy_px"] = detail.cy - detail.manual_y
        results.append(detail)
    detail = pd.concat(results, ignore_index=True)
    detail.to_csv(output / "frame_results.csv", index=False)
    settings = dict(device="cpu", platform=platform.platform(), machine=platform.machine(),
                    torch=torch.__version__, ultralytics=ultralytics.__version__,
                    threads=torch.get_num_threads(), weights=WEIGHTS.name,
                    weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                    sizes=SIZES, batch=1, candidate_conf=.001, direct_conf=.10, link_ratio=.5,
                    max_frames_since_high=2, match_iou=.5, warmup_per_video_size=3)
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    return detail, times

def paired_frames(detail):
    """검수 박스 IoU 0.50으로 판정하고, 같은 프레임끼리 연결해 복구와 손실을 구분한다."""
    keys = KEY + ["method"]
    fixed = ["person_id", "clarity", "occlusion", "hub_source", "manual_x", "manual_y",
             "box_source"] + ["gt_" + c for c in BOX]
    values = ["match", "source", "confidence", "iou", "cx", "cy", "dy_px"] + BOX
    small = detail.loc[detail.imgsz == 640, keys + fixed + values]
    large = detail.loc[detail.imgsz == 800, keys + values]
    pair = small.merge(large, on=keys, how="outer", suffixes=("_640", "_800"),
                       validate="one_to_one", indicator=True)
    if not pair.pop("_merge").eq("both").all() or pair[["match_640", "match_800"]].isna().any().any():
        raise ValueError("두 입력 크기의 평가 대상 또는 검수 박스가 다릅니다.")
    pair["change"] = np.select(
        [pair.match_640 & pair.match_800, ~pair.match_640 & pair.match_800,
         pair.match_640 & ~pair.match_800], ["공통 정탐", "복구", "손실"], default="모두 미탐")
    pair["common_hub"] = pair.change.eq("공통 정탐") & pair.hub_source.isin(["clicked", "previous_blind"])
    return pair


def summarize(pair, groups=("method",)):
    rows = []
    for key, g in pair.groupby(list(groups), sort=False):
        hubs = g[g.common_hub]
        row = dict(zip(groups, key))
        row.update(frames=len(g), hits_640=int(g.match_640.sum()), hits_800=int(g.match_800.sum()),
                   recovered=int(g.change.eq("복구").sum()), lost=int(g.change.eq("손실").sum()),
                   common_hubs=len(hubs), abs_y_640=hubs.dy_px_640.abs().median(),
                   abs_y_800=hubs.dy_px_800.abs().median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pair(pair, video, index, method="Past link"):
    """수동점과 두 박스 중심을 같은 축·같은 원본 위에 겹친다."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    row = pair.set_index(KEY + ["method"]).loc[(video, index, method)]
    rgb = read_rgb(VIDEOS / f"{video}.mp4", index)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), layout="constrained", sharex=True, sharey=True)
    for ax, size in zip(axes, SIZES):
        ax.imshow(rgb)
        x, y, x2, y2 = [row["gt_" + c] for c in BOX]
        ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="gold", linestyle="--", label="Manual box"))
        ax.plot(row.manual_x, row.manual_y, "o", color="gold", fillstyle="none", label="Manual hub")
        source = row[f"source_{size}"]
        if source != "missing":
            a, b, c, d = [row[f"{k}_{size}"] for k in BOX]
            ax.add_patch(Rectangle((a, b), c-a, d-b, fill=False, edgecolor="deepskyblue", label="Prediction"))
            ax.plot(row[f"cx_{size}"], row[f"cy_{size}"], "x", color="deepskyblue", label="Box center")
        score = row[f"iou_{size}"]
        ax.set_title(f"{size}  |  {source}  |  IoU {score:.2f}" if pd.notna(score) else f"{size}  |  {source}")
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{video}  /  frame {index}  /  {method}")
    return fig


def fiftyone_dataset(pair, output):
    """56장의 원본과 두 시간 연결 결과를 로컬 FiftyOne에서 필터링한다."""
    import fiftyone as fo

    folder = output / "review_images"
    folder.mkdir(exist_ok=True)
    dataset = fo.Dataset()
    samples = []
    for r in tqdm(pair[pair.method.eq("Past link")].to_dict("records"), desc="FiftyOne 준비"):
        video, index = r["video_id"], r["frame_index"]
        rgb = read_rgb(VIDEOS / f"{video}.mp4", index)
        height, width = rgb.shape[:2]
        path = folder / f"{video}_f{index}.png"
        Image.fromarray(rgb).save(path)
        sample = fo.Sample(filepath=str(path))
        for name in ["video_id", "frame_index", "person_id", "clarity", "occlusion", "change"]:
            sample[name] = r[name]
        if pd.notna(r["manual_x"]):
            sample["manual_hub"] = fo.Keypoint(label="hub", points=[(r["manual_x"]/width, r["manual_y"]/height)])
        for name, cols in [("ground_truth", ["gt_" + c for c in BOX]),
                           ("size_640", [c + "_640" for c in BOX]), ("size_800", [c + "_800" for c in BOX])]:
            x, y, x2, y2 = [r[col] for col in cols]
            if pd.notna(x):
                sample[name] = fo.Detection(label="plate", bounding_box=[x/width, y/height, (x2-x)/width, (y2-y)/height])
        sample["relative_box_area"] = (r["gt_x2"]-r["gt_x1"])*(r["gt_y2"]-r["gt_y1"])/(width*height)
        for size in SIZES:
            sample[f"source_{size}"] = r[f"source_{size}"]
            if pd.notna(r[f"dy_px_{size}"]):
                sample[f"dy_px_{size}"] = r[f"dy_px_{size}"]
        samples.append(sample)
    dataset.add_samples(samples)
    return dataset

def fixed_validation(output):
    """기존 박스 검증셋의 mAP도 같은 두 입력 크기로 비교한다."""
    model = YOLO(str(WEIGHTS.resolve(strict=True)))
    rows = []
    for size in SIZES:
        result = model.val(data=str(DATASET / "data.yaml"), split="val", imgsz=size, device="cpu",
                           batch=16, workers=0, conf=.001, iou=.7, max_det=300, plots=False,
                           project=str(output.resolve()), name=f"validation_{size}", verbose=False)
        rows.append({"imgsz": size, "map50": result.box.map50, "map50_95": result.box.map})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "fixed_validation.csv", index=False)
    return metrics
