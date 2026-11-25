"""같은 검증 프레임에서 수행자 미탐과 가까운 무릎·발목의 위치 오차를 비교한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm
from ultralytics.utils.metrics import box_iou

from wlcv.data.frames import read_rgb
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table

PERSON_CONF, MATCH_IOU, JOINT_CONF = 0.25, 0.50, 0.50
FRAME = ["영상", "프레임", "가까운 쪽"]
KEY = [*FRAME, "관절"]


def measure_frame(sample, result):
    """배경에 사람이 있어도 수동 수행자 박스와 가장 많이 겹친 후보를 대조한다."""
    boxes = result.boxes
    best_box = None
    overlap = np.nan
    points = None
    if len(boxes):
        truth = torch.tensor([sample.box], dtype=torch.float32)
        ious = box_iou(truth, boxes.xyxy.float())[0]
        best = int(ious.argmax())
        overlap = float(ious[best])
        best_box = tuple(boxes.xyxy[best].tolist())
        if overlap >= MATCH_IOU:
            points = joint_table(result.numpy(), best + 1, JOINT_CONF)
            points = points[points["방향"] == sample.side].set_index("관절")

    rows = []
    for joint, name in [("무릎", "knee"), ("발목", "ankle")]:
        x = y = conf = distance = np.nan
        status = "수행자 미탐"
        if points is not None:
            point = points.loc[joint]
            conf, status = point["관절 신뢰도"], point["표시 상태"]
            if status == "표시":
                x, y = point["x (px)"], point["y (px)"]
                tx, ty = sample.points[name]
                distance = float(np.hypot(x - tx, y - ty))
                status = "측정"
        height = sample.box[3] - sample.box[1]
        rows.append({"영상": sample.video.stem, "프레임": sample.frame, "가까운 쪽": sample.side,
                     "관절": joint, "수행자 미탐": points is None, "후보 IoU": overlap,
                     "후보 박스": best_box, "상태": status, "관절 신뢰도": conf,
                     "예측 x": x, "예측 y": y, "오차(px)": distance,
                     "박스 높이 대비(%)": 100 * distance / height})
    return rows


def evaluate_pose(model, frames):
    rows = []
    for sample in tqdm(frames, desc="자세 평가", unit="장"):
        rgb = read_rgb(sample.video, sample.frame)
        prediction = predict_rgb(model, rgb, conf=PERSON_CONF, imgsz=640, device="cpu")
        rows.extend(measure_frame(sample, prediction.cpu()))
    return pd.DataFrame(rows)


def detection_summary(models):
    rows = []
    for name, table in models.items():
        frames = table.drop_duplicates(FRAME)
        measured = table[table["상태"] == "측정"]
        rows.append({"모델": name, "평가 프레임": len(frames), "수행자 미탐": int(frames["수행자 미탐"].sum()),
                     "무릎 측정 수": (measured["관절"] == "무릎").sum(),
                     "발목 측정 수": (measured["관절"] == "발목").sum()})
    return pd.DataFrame(rows)


def paired_errors(before, after):
    """양쪽에서 같은 관절을 측정했을 때만 차이를 낸다. 미탐 행은 표에 남긴다."""
    fields = ["상태", "오차(px)", "박스 높이 대비(%)"]
    left = before[KEY + fields].set_index(KEY)
    right = after[KEY + fields].set_index(KEY)
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise ValueError("두 모델의 영상·프레임·방향·관절 구성이 다릅니다.")
    table = left.add_prefix("전 ").join(right.add_prefix("후 ")).reset_index()
    table["변화(px)"] = table["후 오차(px)"] - table["전 오차(px)"]
    table["변화(%p)"] = table["후 박스 높이 대비(%)"] - table["전 박스 높이 대비(%)"]
    paired = table.dropna(subset=["변화(px)"])
    summary = paired.groupby("관절").agg(**{
        "양쪽에서 측정한 수": ("변화(px)", "size"),
        "전 평균(px)": ("전 오차(px)", "mean"), "후 평균(px)": ("후 오차(px)", "mean"),
        "평균 변화(px)": ("변화(px)", "mean"),
        "전 평균(%)": ("전 박스 높이 대비(%)", "mean"), "후 평균(%)": ("후 박스 높이 대비(%)", "mean"),
        "평균 변화(%p)": ("변화(%p)", "mean"),
    }).reindex(["무릎", "발목"])
    summary["양쪽에서 측정한 수"] = summary["양쪽에서 측정한 수"].fillna(0).astype(int)
    return table, summary


def comparison_figure(frames, models):
    """열은 프레임, 행은 모델이다. 같은 범위에서 수동점과 모델점을 겹쳐 그린다."""
    fig, axes = plt.subplots(len(models), len(frames), squeeze=False,
                             figsize=(max(4.7, 2.8 * len(frames)), 3.5 * len(models) + 0.6),
                             layout="constrained")
    for column, sample in enumerate(frames):
        rgb = read_rgb(sample.video, sample.frame)
        left, top, right, bottom = sample.box
        pad = (bottom - top) * 0.15
        for row, (model_name, table) in enumerate(models.items()):
            ax = axes[row, column]
            subset = table[(table["영상"] == sample.video.stem) & (table["프레임"] == sample.frame)]
            first = subset.iloc[0]
            ax.imshow(rgb)
            ax.add_patch(Rectangle((left, top), right-left, bottom-top, fill=False, color="limegreen"))
            if first["후보 박스"] is not None:
                x1, y1, x2, y2 = first["후보 박스"]
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color="orangered", linestyle="--"))
            for joint, name, color in [("무릎", "knee", "cyan"), ("발목", "ankle", "gold")]:
                truth = sample.points[name]
                ax.plot(*truth, "o", color=color, markeredgecolor="black", markersize=7)
                p = subset[subset["관절"] == joint].iloc[0]
                if p["상태"] == "측정":
                    x, y = p["예측 x"], p["예측 y"]
                    ax.plot([truth[0], x], [truth[1], y], color=color)
                    ax.plot(x, y, "x", color=color, markersize=9, markeredgewidth=2)
            state = "MISS" if first["수행자 미탐"] else f'{(subset["상태"] == "측정").sum()}/2 joints'
            ax.set_title(f"{sample.video.stem} #{sample.frame}\n{model_name} | {state}", fontsize=10)
            ax.set_xlim(max(0, left-pad), min(rgb.shape[1], right+pad))
            ax.set_ylim(min(rgb.shape[0], bottom+pad), max(0, top-pad))
            ax.axis("off")
    fig.legend(handles=[Line2D([], [], color="limegreen", label="Manual box"),
                        Line2D([], [], color="orangered", ls="--", label="Candidate box"),
                        Line2D([], [], color="black", ls="", marker="o", label="Manual point"),
                        Line2D([], [], color="black", ls="", marker="x", label="Prediction")],
               loc="outside lower center", ncol=2 if len(frames) == 1 else 4, fontsize=9)
    return fig
