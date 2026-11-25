"""현재 한 프레임의 채도만 바꿔 원판 검출·외곽·허브 오차를 비교한다."""
from itertools import islice

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import VIDEOS
from wlcv.data.frames import frames_rgb, read_rgb
from wlcv.evaluation.plate_resolution import WEIGHTS
from wlcv.evaluation.plate_track import BOX, KEY, select_centers
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.resolution_edges import box_iou

FACTORS = (1.0, 0.5, 1.5)


def change_saturation(rgb, factor):
    """원본은 그대로 반환한다. 변형에서는 HSV의 S만 바꾼 뒤 RGB로 돌린다."""
    if factor == 1:
        return rgb
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 1)
    return np.rint(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255).astype(np.uint8)


def detect(model, rgb, video, frame):
    boxes = predict_rgb(model, rgb, imgsz=640, conf=.001, device="cpu").boxes
    return [(video, frame, score, *box)
            for score, box in zip(boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True)]


def score_frame(target, choice, factor):
    truth = target[BOX].to_numpy(float)
    pred = choice[BOX].to_numpy(float)
    iou = box_iou(pred, truth) if choice.source != "missing" else np.nan
    readable = target.hub_source in ("clicked", "previous_blind") and pd.notna(target.manual_y)
    return {
        "factor": factor, "source": choice.source, "match": bool(iou >= .50),
        "confidence": choice.confidence, "iou": iou, **dict(zip(BOX, pred)),
        "top_px": pred[1] - truth[1], "bottom_px": pred[3] - truth[3],
        "center_px": choice.cy - (truth[1] + truth[3]) / 2,
        "hub_y_error_px": choice.cy - target.manual_y if readable else np.nan,
    }


def run_review(targets):
    """0번부터 디코딩하며, 각 비교가 끝난 뒤 원본의 후보만 다음 이력에 넣는다."""
    if len(targets) != 56 or targets.person_id.nunique() != 6 or targets.duplicated(KEY).any():
        raise ValueError("검증 6명의 판독 프레임 56장을 확인하세요.")
    model = YOLO(str(WEIGHTS.resolve(strict=True)))
    if model.names != {0: "plate"}:
        raise ValueError("원판 베이스라인 가중치를 확인하세요.")
    rows = []
    for video, group in targets.groupby("video_id"):
        labels = group.set_index("frame_index")
        stop = int(labels.index.max()) + 1
        history = []
        stream = islice(frames_rgb(VIDEOS / f"{video}.mp4"), stop)
        for frame, rgb in enumerate(tqdm(stream, total=stop, desc=video, unit="장")):
            original = detect(model, rgb, video, frame)
            if frame in labels.index:
                prefix = pd.DataFrame({"video_id": video, "frame_index": range(frame + 1)})
                for factor in FACTORS:
                    now = original if factor == 1 else detect(model, change_saturation(rgb, factor), video, frame)
                    candidates = pd.DataFrame(history + now, columns=KEY + ["confidence"] + BOX)
                    selected = select_centers(prefix, candidates, distance_ratio=.5, show_progress=False)
                    choice = selected[(selected.frame_index == frame) & (selected.method == "Past link")].iloc[0]
                    scores = score_frame(labels.loc[frame], choice, factor)
                    rows.append({"video_id": video, "frame_index": frame, **scores})
            history.extend(original)
    if len(rows) != len(targets) * len(FACTORS):
        raise ValueError("접촉 프레임까지 디코딩한 장수를 확인하세요.")
    truth = targets.rename(columns={name: "gt_" + name for name in BOX})
    return pd.DataFrame(rows).merge(truth, on=KEY, validate="many_to_one")


def check_original(detail, reference):
    """33번에서 저장한 640의 선택 출처·좌표·신뢰도와 정답 매칭을 프레임별로 대조한다."""
    saved = pd.read_csv(reference).query("method == 'Past link'").set_index(KEY, verify_integrity=True).sort_index()
    current = detail[detail.factor.eq(1)].set_index(KEY, verify_integrity=True).sort_index()
    if not current.index.equals(saved.index):
        raise ValueError("33번 실험과 이번 실험의 대상 프레임이 다릅니다.")
    same = (current.source == saved.source_640) & (current.match == saved.match_640)
    for column in BOX + ["confidence", "iou"]:
        tolerance = .05 if column in BOX else .0001
        same &= np.isclose(current[column], saved[column + "_640"], atol=tolerance, rtol=0, equal_nan=True)
    if not same.all():
        raise ValueError(f"원본 비교가 달라진 프레임입니다. 가중치와 실행 환경을 확인하세요. {same[~same].index.tolist()}")


def paired_summary(detail):
    """각 변형과 원본이 모두 맞힌 프레임끼리 선수별 변화량을 계산한다."""
    metrics = ["match", "bottom_px", "top_px", "center_px", "hub_y_error_px", "iou"]
    original = detail.loc[detail.factor.eq(1), KEY + metrics]
    variants = detail[detail.factor.ne(1)].merge(original, on=KEY, suffixes=("", "_original"), validate="many_to_one")
    rows = []
    for (person, factor), group in variants.groupby(["person_id", "factor"]):
        common = group[group.match & group.match_original]
        hub = common.dropna(subset=["hub_y_error_px", "hub_y_error_px_original"])
        row = {
            "person_id": person, "factor": factor, "frames": len(group), "common": len(common),
            "recovered": int((group.match & ~group.match_original).sum()),
            "lost": int((~group.match & group.match_original).sum()),
            "direct_hits": int((group.match & group.source.eq("high")).sum()),
            "linked_hits": int((group.match & group.source.eq("linked")).sum()),
            "hub_pairs": len(hub),
        }
        for field in ["bottom_px", "top_px", "center_px", "iou"]:
            row["delta_" + field] = (common[field] - common[field + "_original"]).mean()
        for field, pairs in [("bottom_px", common), ("hub_y_error_px", hub)]:
            row["delta_abs_" + field] = (pairs[field].abs() - pairs[field + "_original"].abs()).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def show_frame(targets, video, frame, detail=None):
    """같은 시야에서 채도별 수동 박스·허브와 선택 결과를 확대한다."""
    target = targets.set_index(KEY).loc[(video, frame)]
    rgb = read_rgb(VIDEOS / f"{video}.mp4", frame)
    truth = target[BOX].to_numpy(float)
    choices = None if detail is None else detail.set_index(KEY + ["factor"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained", sharex=True, sharey=True)
    for ax, factor in zip(axes, FACTORS, strict=True):
        ax.imshow(change_saturation(rgb, factor))
        boxes = [(truth, "#ffc242", "--")]
        title = f"S × {factor:.1f}"
        if choices is not None:
            choice = choices.loc[(video, frame, factor)]
            title += f" | {choice.source}"
            if choice.source != "missing":
                boxes.append((choice[BOX].to_numpy(float), "#26d8ff", "-"))
                title += f"\nconf {choice.confidence:.3f} | IoU {choice.iou:.3f}"
        for (x, y, right, bottom), color, style in boxes:
            ax.add_patch(Rectangle((x, y), right-x, bottom-y, fill=False, ec=color, ls=style, lw=1.8))
        if target.hub_source in ("clicked", "previous_blind"):
            ax.plot(target.manual_x, target.manual_y, "+", color="white", ms=10)
        pad = .65 * max(truth[2]-truth[0], truth[3]-truth[1])
        ax.set(xlim=(truth[0]-pad, truth[2]+pad), ylim=(truth[3]+pad, truth[1]-pad), title=title)
        ax.axis("off")
    fig.suptitle(f"{video} / frame {frame}    dashed gold = manual box    + = manual hub    blue = prediction")
    return fig
