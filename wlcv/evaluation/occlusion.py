"""같은 프레임에 같은 크기의 마스크를 씌워 원판 검출의 변화를 비교한다."""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from ultralytics.utils.metrics import box_iou

from .plots import predict_rgb


def plot_occlusion_comparison(
    model, frame, target, regions, mask_size, *, min_confidence=0.001, fills=None,
):
    """정답과 IoU가 0.5 이상인 후보 중 가장 높은 신뢰도를 고른다."""
    height, width = frame.shape[:2]
    cx, cy, bw, bh = target
    ground_truth = torch.tensor([[(cx - bw / 2) * width, (cy - bh / 2) * height,
                                   (cx + bw / 2) * width, (cy + bh / 2) * height]])
    mask_width, mask_height = mask_size
    if not (0 < mask_width <= width and 0 < mask_height <= height):
        raise ValueError("마스크 크기는 프레임 안에 들어와야 합니다.")

    panels = [("Original", None), *regions.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 5), layout="constrained")
    results = []
    for axis, (name, center) in zip(axes, panels):
        image = frame.copy()
        if center is not None:
            x = round(center[0] * width - mask_width / 2)
            y = round(center[1] * height - mask_height / 2)
            if not (0 <= x <= width - mask_width and 0 <= y <= height - mask_height):
                raise ValueError(f"{name} 마스크가 프레임을 벗어납니다.")
            fill = 114 if fills is None else fills.get(name, 114)
            image[y:y + mask_height, x:x + mask_width] = fill
        output = predict_rgb(model, image, conf=min_confidence)
        boxes = output.boxes.cpu()
        overlaps = box_iou(ground_truth, boxes.xyxy)[0]
        matches = torch.where((overlaps >= 0.5) & (boxes.cls == 0))[0]

        axis.imshow(image)
        axis.set_axis_off()
        gx1, gy1, gx2, gy2 = ground_truth[0].tolist()
        axis.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False,
                                 edgecolor="#80ff80", linewidth=2))
        confidence, overlap = None, None
        if len(matches):
            index = matches[boxes.conf[matches].argmax()]
            confidence = float(boxes.conf[index])
            overlap = float(overlaps[index])
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                     edgecolor="#ffd067", linewidth=2, linestyle="--"))
        subtitle = f"conf {confidence:.3f} · IoU {overlap:.2f}" if confidence is not None else "No matched box"
        axis.set_title(f"{name}\n{subtitle}")
        results.append({"condition": name, "confidence": confidence, "iou": overlap})
    return fig, pd.DataFrame(results)
