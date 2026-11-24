"""원본 b02의 실제 선택과 선택되지 않은 바깥 원판 후보를 비교한다."""
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import VIDEOS
from wlcv.data.frames import frames_rgb, read_rgb
from wlcv.evaluation.face_brightness import INDICES, VIDEO, candidates
from wlcv.evaluation.plate_resolution import WEIGHTS
from wlcv.evaluation.plate_track import BOX, KEY, select_centers
from wlcv.evaluation.resolution_edges import box_iou


def review_candidates(pairs):
    """과거 원본에서 실제 선택을 재현한 뒤 각 후보를 수동 외곽·허브에 대조한다."""
    ref = pairs[(pairs.video_id == VIDEO) & (pairs.method == "Past link") &
                pairs.frame_index.isin(INDICES)].set_index("frame_index", verify_integrity=True)
    if sorted(ref.index) != list(INDICES) or not ref.hub_source.isin(["clicked", "previous_blind"]).all():
        raise ValueError("네 프레임의 수동 외곽·허브와 33번 실험 결과가 필요합니다.")
    model = YOLO(str(WEIGHTS.resolve(strict=True)))
    frames = pd.DataFrame({"video_id": VIDEO, "frame_index": range(max(INDICES) + 1)})
    raw = []
    video = frames_rgb(VIDEOS / f"{VIDEO}.mp4")
    count = 0
    for index, rgb in enumerate(tqdm(islice(video, len(frames)), total=len(frames), desc="원본 추론")):
        raw.extend(candidates(model, rgb, index))
        count += 1
    if count != len(frames):
        raise ValueError("요청한 프레임까지 읽을 수 없습니다.")
    boxes = pd.DataFrame(raw, columns=KEY + ["confidence"] + BOX)
    history = []
    choices = select_centers(frames, boxes, distance_ratio=0.5, history_checks=history)
    selected = choices[choices.method == "Past link"].set_index("frame_index")
    context = pd.DataFrame(history).set_index("frame_index")
    tables = []
    for frame in INDICES:
        gt = ref.loc[frame]
        choice = selected.loc[frame]
        if choice.source != gt.source_800 or not np.allclose(choice[BOX].to_numpy(float),
                gt[[c + "_800" for c in BOX]].to_numpy(float), rtol=0, atol=0.001):
            raise ValueError(f"{frame}번 원본 선택이 33번 실험과 다릅니다.")
        before = context.loc[frame]
        now = boxes[boxes.frame_index == frame].sort_values("confidence", ascending=False, kind="stable").copy()
        now["rank"] = range(1, len(now) + 1)
        now["cx"] = (now.x1 + now.x2) / 2
        now["cy"] = (now.y1 + now.y2) / 2
        now["distance_px"] = np.hypot(now.cx - before.expected_x, now.cy - before.expected_y)
        now["radius_px"] = before.radius_px
        now["inside_radius"] = now.distance_px <= before.radius_px
        now["expected_x"], now["expected_y"] = before.expected_x, before.expected_y
        now["manual_x"], now["manual_y"] = gt.manual_x, gt.manual_y
        now["dy_px"] = now.cy - gt.manual_y
        now["hub_error_px"] = np.hypot(now.cx - gt.manual_x, now.dy_px)
        outer = gt[["gt_" + c for c in BOX]].to_numpy(float)
        now["iou_outer"] = [box_iou(box, outer) for box in now[BOX].to_numpy(float)]
        now["selected"] = (now[BOX].eq(choice[BOX]).all(axis=1) & now.confidence.eq(choice.confidence))
        if now.selected.sum() != 1:
            raise ValueError(f"{frame}번에서 실제 선택한 후보를 특정할 수 없습니다.")
        now["best_outer"] = now.index == now.iou_outer.idxmax()
        now["source"] = choice.source
        tables.append(now)
    return pd.concat(tables, ignore_index=True)


def plot_candidates(detail, pairs):
    """같은 원본에 두 박스와 허브, 선택 직전의 예상 중심·반경을 겹친다."""
    ref = pairs[(pairs.video_id == VIDEO) & (pairs.method == "Past link")].set_index("frame_index")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for ax, frame in zip(axes.flat, INDICES):
        group = detail[detail.frame_index == frame]
        actual = group[group.selected].iloc[0]
        outer = group[group.best_outer].iloc[0]
        gt = ref.loc[frame]
        ax.imshow(read_rgb(VIDEOS / f"{VIDEO}.mp4", frame))
        for box, color, style in [(gt[["gt_" + c for c in BOX]], "gold", "--"),
                                  (actual[BOX], "deepskyblue", "-"), (outer[BOX], "magenta", ":")]:
            x, y, x2, y2 = map(float, box)
            ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor=color, ls=style, lw=2))
        ax.plot(actual.cx, actual.cy, "x", color="deepskyblue", ms=9)
        ax.plot(outer.cx, outer.cy, "+", color="magenta", ms=10)
        ax.plot(gt.manual_x, gt.manual_y, "o", color="gold", fillstyle="none", ms=9)
        ax.plot(actual.expected_x, actual.expected_y, ".", color="white")
        ax.add_patch(Circle((actual.expected_x, actual.expected_y), actual.radius_px,
                            fill=False, color="white", alpha=.8, ls="--"))
        ax.set(xlim=(0, 610), ylim=(gt.gt_y2 + 65, gt.gt_y1 - 65),
               title=f"Frame {frame}  |  {actual.source}  |  radius {actual.radius_px:.0f} px")
        ax.axis("off")
    handles = [Line2D([], [], color=c, ls=s, marker=m, fillstyle="none", label=label)
               for c, s, m, label in [("gold", "--", "o", "Manual outer / hub"),
                                      ("deepskyblue", "-", "x", "Selected / center"),
                                      ("magenta", ":", "+", "Best outer IoU / center"),
                                      ("gray", "--", ".", "Expected center / radius")]]
    fig.legend(handles=handles, loc="outside lower center", ncols=2)
    return fig
