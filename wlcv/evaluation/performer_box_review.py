"""두 검증 장면에서 같은 예측을 1차 박스와 보이는 신체 기준 박스에 대조한다."""

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from ultralytics.utils.metrics import box_iou

from wlcv.annotation.performer_box import PerformerBox
from wlcv.config import ANNOTATIONS, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.pose_comparison import MATCH_IOU


# 1차 박스는 당시 판독한 노트북에 따라 두 폴더로 나뉘어 있다.
FIRST_FOLDERS = {
    ("c04_sn_f_01", 78): ANNOTATIONS / "pose_box_review" / "validation",
    ("c07_sn_f_01", 41): ANNOTATIONS / "knee_ankle_candidates" / "validation",
}
REVIEW_FOLDER = ANNOTATIONS / "visible_performer_review" / "validation"
MODELS = {"원본": "before", "파인튜닝": "after"}
BOX_STYLES = [
    ("First label", "#0072B2", "--"),
    ("Visible label", "#009E73", "-"),
    ("Prediction", "#D55E00", ":"),
]


def compare_saved_boxes(review_folder, prediction_folder):
    """기존 실행에서 저장한 수행자 후보를 고정한다. 양쪽 IoU에 같은 좌표를 쓴다."""
    labels = {}
    for (video, frame), first_folder in FIRST_FOLDERS.items():
        first = PerformerBox(first_folder, video, frame).saved
        visible = PerformerBox(review_folder, video, frame).saved
        if first is None or visible is None:
            raise ValueError(f"{video} {frame}번의 1차 박스와 2차 저장 여부를 확인하세요.")
        labels[video, frame] = first, visible

    rows = []
    for model, suffix in MODELS.items():
        table = pd.read_csv(prediction_folder / f"validation_{suffix}.csv")
        # 같은 후보가 무릎·발목 행에 반복되므로 프레임별 박스 한 개만 읽는다.
        for (video, frame), (first, visible) in labels.items():
            matched = table[(table["영상"] == video) & (table["프레임"] == frame)]
            candidates = matched[["후보 박스", "후보 IoU"]].drop_duplicates()
            if len(candidates) != 1 or candidates.isna().any().any():
                raise ValueError(f"{model}의 {video} {frame}번 저장 후보를 확인하세요.")
            row = candidates.iloc[0]
            prediction = literal_eval(row["후보 박스"])
            ground_truth = torch.tensor([first, visible], dtype=torch.float32)
            candidate = torch.tensor([prediction], dtype=torch.float32)
            first_iou, visible_iou = box_iou(ground_truth, candidate)[:, 0].tolist()
            if not np.isclose(first_iou, row["후보 IoU"], atol=1e-5):
                raise ValueError(f"{video} {frame}번의 1차 박스가 기존 평가와 다릅니다.")
            rows.append({"영상": video, "프레임": frame, "모델": model,
                         "윗변 변화(px)": visible[1] - first[1],
                         "1차 IoU": first_iou, "2차 IoU": visible_iou,
                         "IoU 변화": visible_iou - first_iou,
                         "1차 매칭": first_iou >= MATCH_IOU, "2차 매칭": visible_iou >= MATCH_IOU,
                         "1차 박스": first, "2차 박스": visible, "예측 박스": prediction})
    return pd.DataFrame(rows)


def draw_comparison(report):
    """같은 프레임의 두 모델을 같은 크기로 표시하고 세 박스의 선 모양을 구분한다."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 9), layout="constrained")
    for column, ((video, frame), group) in enumerate(report.groupby(["영상", "프레임"], sort=False)):
        rgb = read_rgb(VIDEOS / f"{video}.mp4", int(frame))
        all_boxes = np.array(group[["1차 박스", "2차 박스", "예측 박스"]].values.tolist()).reshape(-1, 4)
        left, top = all_boxes[:, :2].min(axis=0)
        right, bottom = all_boxes[:, 2:].max(axis=0)
        pad = (bottom - top) * 0.12
        for row, (_, item) in enumerate(group.iterrows()):
            ax = axes[row, column]
            ax.imshow(rgb)
            boxes = [item["1차 박스"], item["2차 박스"], item["예측 박스"]]
            for (label, color, style), (x1, y1, x2, y2) in zip(BOX_STYLES, boxes):
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linestyle=style, linewidth=2))
            name = "Original" if item["모델"] == "원본" else "Fine-tuned"
            ax.set_title(f"{video[:3]} #{frame} | {name}\nIoU {item['1차 IoU']:.3f} → {item['2차 IoU']:.3f}")
            ax.set_xlim(max(0, left - pad), min(rgb.shape[1], right + pad))
            ax.set_ylim(min(rgb.shape[0], bottom + pad), max(0, top - pad))
            ax.axis("off")
    handles = [Line2D([], [], color=color, linestyle=style, label=label)
               for label, color, style in BOX_STYLES]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return fig
