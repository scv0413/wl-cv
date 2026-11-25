"""b02 원판 안쪽의 밝기 변화가 선택한 원판 박스에 미치는 영향을 비교한다."""
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
from wlcv.evaluation.resolution_edges import box_iou, bright_face

VIDEO = "b02_sn_h_02"
INDICES = (32, 33, 34, 35)
CONDITIONS = ("original", "face", "ceiling")


def darkened_frames(rgb, outer, amount=40):
    """같은 RGB 감산 지도를 밝은 면 또는 오른쪽 천장으로 옮긴다."""
    if not 0 < amount <= 255:
        raise ValueError("밝기 감소량은 1~255에서 선택하세요.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inner, contour = bright_face(gray, outer, cutoff=130)
    x, y, right, bottom = inner
    w, h = right-x, bottom-y
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
    mask[gray <= 130] = 0
    # 원래 밝은 픽셀만 바꾸고, 마스크 안쪽 5 px에 걸쳐 서서히 어둡게 한다.
    alpha = np.minimum(cv2.distanceTransform(mask, cv2.DIST_L2, 5) / 5, 1)[y:bottom, x:right]
    # b02 720×1280 영상에서 원판·선수가 지나지 않는 오른쪽 천장의 고정 중심.
    bx, by = 555-w//2, 170-h//2
    face = rgb[y:bottom, x:right]
    sky = rgb[by:by+h, bx:bx+w]
    if sky.shape != face.shape or by+h >= outer[1]:
        raise ValueError("천장 대조 영역과 원판 위치를 확인하세요.")
    # 두 위치 모두에서 RGB 값이 0 아래로 잘리지 않도록 같은 양을 뺀다.
    delta = np.minimum(np.rint(alpha*amount), np.minimum(face.min(2), sky.min(2))).astype(np.uint8)
    images = {"original": rgb, "face": rgb.copy(), "ceiling": rgb.copy()}
    images["face"][y:bottom, x:right] -= delta[:, :, None]
    images["ceiling"][by:by+h, bx:bx+w] -= delta[:, :, None]
    return images, inner, (bx, by, bx+w, by+h)


def candidates(model, rgb, index):
    boxes = predict_rgb(model, rgb, imgsz=800, conf=.001, device="cpu").boxes
    return [(VIDEO, index, conf, *coords) for conf, coords in zip(boxes.conf.tolist(), boxes.xyxy.tolist())]


def run_face_brightness(pairs, amount=40):
    """한 번에 현재 한 장만 편집한다. 각 조건은 같은 원본 후보 이력에서 시작한다."""
    ref = pairs[(pairs.video_id == VIDEO) & (pairs.method == "Past link") &
                pairs.frame_index.isin(INDICES)].set_index("frame_index", verify_integrity=True)
    if sorted(ref.index) != list(INDICES):
        raise ValueError("33번 실험에서 b02 32~35번의 기준 결과를 확인하세요.")
    model = YOLO(str(WEIGHTS.resolve(strict=True)))
    if model.names != {0: "plate"}:
        raise ValueError("원판 베이스라인 best.pt가 필요합니다.")
    history = []
    results = []
    for index, rgb in enumerate(tqdm(islice(frames_rgb(VIDEOS / f"{VIDEO}.mp4"), 36),
                                     total=36, desc="원본 이력과 밝기 비교", unit="장")):
        original = candidates(model, rgb, index)
        if index in INDICES:
            row = ref.loc[index]
            outer = row[["gt_" + c for c in BOX]].to_numpy(float)
            images, inner, _ = darkened_frames(rgb, outer, amount)
            frames = pd.DataFrame({KEY[0]: VIDEO, KEY[1]: range(index+1)})
            for condition, image in images.items():
                now = original if condition == "original" else candidates(model, image, index)
                prefix = pd.DataFrame(history + now, columns=KEY + ["confidence"] + BOX)
                selected = select_centers(frames, prefix, distance_ratio=.5)
                choice = selected[(selected.frame_index == index) & (selected.method == "Past link")].iloc[0]
                pred = choice[BOX].to_numpy(float)
                found = choice.source != "missing"
                changed = rgb[:, :, 0].astype(int) - image[:, :, 0]
                hits = [box_iou(np.array(c[3:]), outer) for c in now]
                results.append(dict(frame_index=index, condition=condition, max_decrement=amount,
                                    source=choice.source,
                                    confidence=choice.confidence, **dict(zip(BOX, pred)),
                                    iou_outer=box_iou(pred,outer) if found else np.nan,
                                    iou_inner=box_iou(pred,inner) if found else np.nan,
                                    width_px=pred[2]-pred[0], max_candidate_outer_iou=max(hits, default=0),
                                    dy_px=choice.cy-row.manual_y, changed_pixels=int((changed>0).sum()),
                                    mean_decrement=changed[changed>0].mean() if changed.any() else 0))
        history.extend(original)
    detail = pd.DataFrame(results)
    if len(detail) != len(INDICES)*len(CONDITIONS):
        raise ValueError("영상 종료 시점과 조건별 프레임 수를 확인하세요.")
    return detail


def show_frame(pairs, results, index=35):
    """변경 영역의 실제 위치와 선택된 박스를 세 조건에서 같은 배율로 표시한다."""
    ref = pairs[(pairs.video_id == VIDEO) & (pairs.method == "Past link") &
                (pairs.frame_index == index)].iloc[0]
    outer = ref[["gt_" + c for c in BOX]].to_numpy(float)
    amount, = results.max_decrement.unique()
    images, _, ceiling = darkened_frames(read_rgb(VIDEOS / f"{VIDEO}.mp4", index), outer, amount)
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.8), layout="constrained")
    for ax, condition in zip(axes, CONDITIONS):
        r = results[(results.frame_index == index) & (results.condition == condition)].iloc[0]
        ax.imshow(images[condition])
        for coords, color, style in [(outer, "#ffc742", "--"), (r[BOX].to_numpy(float), "#1ed7ff", "-")]:
            if np.isfinite(coords).all():
                x,y,x2,y2=coords
                ax.add_patch(Rectangle((x,y), x2-x,y2-y, fill=False, ec=color, ls=style, lw=1.5))
        if condition == "ceiling":
            x,y,x2,y2=ceiling
            ax.add_patch(Rectangle((x,y), x2-x,y2-y, fill=False, ec="white", ls=":", lw=1))
        ax.set(xlim=(0,720), ylim=(outer[3]+45,40))
        ax.set_title(f"{condition}  |  {r.source}\nouter IoU {r.iou_outer:.3f}" if pd.notna(r.iou_outer)
                     else f"{condition}  |  {r.source}")
        ax.axis("off")
    fig.suptitle(f"b02 frame {index}  |  gold: manual outer box   blue: selected box")
    return fig
