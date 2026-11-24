"""검증 원본에서 원판 외곽 검출과 수동 허브의 세로 오차를 함께 평가한다."""
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plate_resolution import SIZES, WEIGHTS, review_targets
from wlcv.evaluation.plate_track import BOX, KEY
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.resolution_edges import box_iou


def evaluate_heights(reference):
    """선택 박스는 원본 전체를 재생한 33번 결과에서 불러오고 두 입력 크기의 후보를 다시 읽는다."""
    settings = json.loads((reference / "settings.json").read_text())
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != settings["weights_sha256"]:
        raise ValueError("저장된 선택 결과와 같은 원판 베이스라인 가중치가 필요합니다.")
    targets = review_targets().set_index(KEY, verify_integrity=True)
    saved = pd.read_csv(reference / "paired_frames.csv")
    saved = saved[saved.method == "Past link"].set_index(KEY, verify_integrity=True)
    if set(targets.index) != set(saved.index):
        raise ValueError("전체 검증 프레임과 33번 실험의 선택 결과를 확인하세요.")

    model = YOLO(str(WEIGHTS))
    frame_rows, candidate_rows = [], []
    for (video, index), target in tqdm(targets.iterrows(), total=len(targets), desc="원본 후보 비교"):
        rgb = read_rgb(VIDEOS / f"{video}.mp4", index)
        choice = saved.loc[(video, index)]
        outer = target[BOX].to_numpy(float)
        readable = target.hub_source in ("clicked", "previous_blind") and pd.notna(target.manual_y)
        for size in SIZES:
            key = dict(person_id=target.person_id, video_id=video, frame_index=index, imgsz=size)
            boxes = predict_rgb(model, rgb, imgsz=size, conf=.001, device="cpu").boxes
            chosen_box = choice[[f"{name}_{size}" for name in BOX]].to_numpy(float)
            source = choice[f"source_{size}"]
            selected_iou = selected_error = np.nan
            matching_errors = []
            selected_count = outer_count = 0
            for conf, coords in zip(boxes.conf.tolist(), boxes.xyxy.tolist()):
                box = np.asarray(coords)
                iou = box_iou(box, outer)
                center_y = (box[1] + box[3]) / 2
                error = abs(center_y - target.manual_y) if readable else np.nan
                selected = (np.allclose(box, chosen_box, atol=.001, rtol=0)
                            and np.isclose(conf, choice[f"confidence_{size}"], atol=1e-6, rtol=0))
                if selected:
                    selected_count += 1
                    selected_iou, selected_error = iou, error
                if iou >= .50:
                    outer_count += 1
                    if readable:
                        matching_errors.append(error)
                candidate_rows.append({**key, **dict(zip(BOX, coords)), "confidence": conf,
                                       "selected": selected, "outer_iou": iou, "height_error_px": error})
            if selected_count != int(source != "missing") or not np.isclose(
                selected_iou, choice[f"iou_{size}"], equal_nan=True, atol=1e-5
            ):
                raise ValueError(f"{video} {index}번의 {size} 선택을 재현할 수 없습니다.")
            eligible = selected_iou < .50 and bool(matching_errors)
            closest_height = min(matching_errors, default=np.nan)
            frame_rows.append({**key, "source": source, "readable": readable,
                               "outer_iou": selected_iou, "height_error_px": selected_error,
                               "candidates": len(boxes), "matching_candidates": outer_count,
                               "min_matching_error_px": closest_height, "eligible": eligible,
                               "pattern": eligible and selected_error < closest_height})
    return pd.DataFrame(frame_rows), pd.DataFrame(candidate_rows)


def summarize_heights(frames):
    return frames.groupby(["imgsz", "person_id"]).agg(
        frames=("frame_index", "size"), manual_hubs=("readable", "sum"),
        missing=("source", lambda s: s.eq("missing").sum()),
        outer_hits=("outer_iou", lambda s: s.ge(.50).sum()),
        outer_misses_with_box=("outer_iou", lambda s: s.lt(.50).sum()),
        eligible=("eligible", "sum"), pattern=("pattern", "sum"),
        median_height_error_px=("height_error_px", "median"),
    ).reset_index()


def plot_heights(frames):
    """한 점은 한 프레임의 실제 선택 박스다. 640·800은 같은 원본 픽셀 좌표로 평가한다."""
    people = sorted(frames.person_id.unique())
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), sharex=True, sharey=True, layout="constrained")
    for person, ax in zip(people, axes.flat, strict=True):
        for size, marker, color in [(640, "o", "#08719c"), (800, "x", "#bf5917")]:
            rows = frames[(frames.person_id == person) & (frames.imgsz == size)]
            ax.scatter(rows.outer_iou, rows.height_error_px, marker=marker, color=color,
                       alpha=.85, s=35, label=f"{size} input")
        ax.axvline(.50, color="gray", ls="--", lw=1)
        ax.set_title(f"{person}  |  {len(rows)} frames")
        ax.grid(alpha=.15)
        ax.set_xlim(0, 1.03)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Selected box IoU with the manual outer plate")
    fig.supylabel("Vertical error from the manual hub (px)")
    return fig
