"""b02 손실 4장에서 밝은 면의 영상 경계와 저장된 원판 예측 박스를 비교한다."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from wlcv.config import VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plate_track import BOX


def bright_face(gray, outer, cutoff):
    """검수한 전체 원판 중앙에서 밝은 연결 영역의 영상 좌표를 구한다."""
    x1, y1, x2, y2 = outer
    # 중앙 76%에 밝은 면을 담고, 원판 주변의 밝은 랙·천장을 배제한다.
    dx, dy = (x2-x1)*.12, (y2-y1)*.12
    left, top, right, bottom = np.rint([x1+dx, y1+dy, x2-dx, y2-dy]).astype(int)
    roi = gray[top:bottom, left:right]
    _, mask = cv2.threshold(roi, cutoff, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("밝은 원판 면의 경계를 찾을 수 없습니다.")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if x == 0 or y == 0 or x+w == roi.shape[1] or y+h == roi.shape[0]:
        raise ValueError("밝은 면이 탐색창 경계에 닿았습니다. 원본에서 확인하세요.")
    box = np.array([left+x, top+y, left+x+w, top+y+h])
    return box, contour.reshape(-1, 2) + [left, top]


def box_iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return overlap / ((a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - overlap)


def compare_bright_edges(pairs):
    """같은 영상에서 밝기 기준을 바꿔도 박스의 경계 비교 결론이 유지되는지 확인한다."""
    video = "b02_sn_h_02"
    lost = pairs[(pairs.video_id == video) & (pairs.method == "Past link") &
                 pairs.frame_index.between(32, 35)].sort_values("frame_index")
    if lost.frame_index.tolist() != [32, 33, 34, 35]:
        raise ValueError("b02 32~35번 시간 연결 결과 네 행이 필요합니다.")
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
    colors = {"outer": "#ffbd44", "inner": "#e975d2", 640: "#38b6ff", 800: "#38b6ff"}
    records = []
    for col, (_, row) in enumerate(lost.iterrows()):
        index = int(row.frame_index)
        rgb = read_rgb(VIDEOS / f"{video}.mp4", index)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        outer = row[["gt_" + s for s in BOX]].to_numpy(float)
        for cutoff in (110, 130, 150):
            inner, contour = bright_face(gray, outer, cutoff)
            for size in (640, 800):
                pred = row[[s + f"_{size}" for s in BOX]].to_numpy(float)
                if not np.isfinite(pred).all():
                    raise ValueError(f"{index}번 {size}에 저장된 예측 박스가 없습니다.")
                records.append(dict(video_id=video, frame_index=index, imgsz=size, cutoff=cutoff,
                                    source=row[f"source_{size}"], pred_width=pred[2]-pred[0],
                                    pred_height=pred[3]-pred[1], pred_area=(pred[2:]-pred[:2]).prod(),
                                    inner_width=inner[2]-inner[0], inner_height=inner[3]-inner[1],
                                    iou_outer=box_iou(pred,outer), iou_inner=box_iou(pred,inner),
                                    inner_edge_mae=np.abs(pred-inner).mean(),
                                    outer_edge_mae=np.abs(pred-outer).mean()))
            if cutoff == 130:
                trace = np.vstack([contour, contour[0]])
        for level, axis in enumerate(axes[:, col]):
            axis.imshow(rgb)
            x, y, x2, y2 = outer
            axis.set(xlim=(x-30, x2+30), ylim=(y2+32, y-32), aspect="equal")
            axis.axis("off")
            axis.add_patch(Rectangle((x,y), x2-x, y2-y, fill=False, linestyle="--",
                                     edgecolor=colors["outer"], linewidth=1.4))
            axis.plot(*trace.T, color=colors["inner"], linewidth=1)
            title = f"Frame {index} | reference"
            if level:
                size = (640, 800)[level-1]
                a,b,c,d = [row[s + f"_{size}"] for s in BOX]
                axis.add_patch(Rectangle((a,b), c-a, d-b, fill=False,
                                         edgecolor=colors[size], linewidth=2))
                title = f"{size} | {row[f'source_{size}']}"
            axis.set_title(title, fontsize=10)
    fig.suptitle("b02   |   dashed gold: reviewed outer box   /   pink: bright face   /   blue: selected box", fontsize=12)
    return pd.DataFrame(records), fig
