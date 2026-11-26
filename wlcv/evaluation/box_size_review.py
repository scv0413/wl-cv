"""선택된 원판 박스의 중심을 고정하고 직전 원본 박스의 크기만 반영한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from wlcv.config import VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.runtime.tracking import iou


KEY = ["video_id", "frame_index"]
BOX = ["x1", "y1", "x2", "y2"]


def smooth_size(current, previous, width, height):
    """이전 장이 미탐이거나 보정 박스가 화면 밖이면 기존 좌표를 유지한다."""
    if not np.isfinite(current).all():
        return current, "no_current"
    if not np.isfinite(previous).all():
        return current, "no_previous"
    center = (current[:2] + current[2:]) / 2
    size = ((current[2:] - current[:2]) + (previous[2:] - previous[:2])) / 2
    result = np.r_[center - size/2, center + size/2]
    if np.any(result[:2] < 0) or np.any(result[2:] > [width, height]):
        return current, "outside_frame"
    return result, "averaged"


def compare_sizes(frames, history, metadata):
    """44번의 현재 박스와 같은 영상의 t-1 원본 선택을 짝짓는다."""
    original = frames[frames.factor.eq(1)].set_index(KEY, verify_integrity=True)
    if (len(original) != 56 or original.person_id.nunique() != 6
            or frames.duplicated(KEY + ["factor"]).any()
            or not frames.groupby(KEY).factor.agg(set).map(lambda s: s == {0.5, 1.0, 1.5}).all()
            or len(frames) != 3 * len(original)):
        raise ValueError("44번 검증 6명·56장·세 채도 조건을 확인하세요.")
    history = history[history.method.eq("Past link")].set_index(KEY, verify_integrity=True)
    same = history.loc[original.index]
    if not (same.source == original.source).all() or not np.allclose(
        same[BOX + ["confidence"]], original[BOX + ["confidence"]], atol=1e-5, rtol=0, equal_nan=True
    ):
        raise ValueError("전체 영상 이력이 44번 원본 박스·점수와 다릅니다.")
    sizes = metadata.set_index("video_id", verify_integrity=True)
    rows = []
    for row in tqdm(frames.itertuples(index=False), total=len(frames), desc="박스 크기 비교", unit="장"):
        current = np.array([row.x1, row.y1, row.x2, row.y2])
        truth = np.array([row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2])
        # 고정한 56장 모두 0번 이후다. 누락된 이력 행과 실제 미탐을 구분한다.
        prior = history.loc[(row.video_id, row.frame_index - 1)]
        prev_box = prior[BOX].to_numpy(float)
        width, height = sizes.loc[row.video_id, ["width", "height"]]
        corrected, reason = smooth_size(current, prev_box, width, height)
        before, after = iou(current, truth), iou(corrected, truth)
        if not np.isclose(before, row.iou, equal_nan=True):
            raise ValueError(f"{row.video_id} {row.frame_index}번의 원본 IoU를 확인하세요.")
        shift = ((corrected[:2] + corrected[2:]) - (current[:2] + current[2:])) / 2
        if np.isfinite(current).all() and not np.allclose(shift, 0, atol=1e-8, rtol=0):
            raise AssertionError("보정 과정에서 박스 중심이 움직였습니다.")
        rows.append(dict(reason=reason, previous_source=prior.source,
                         previous_width=prev_box[2]-prev_box[0], previous_height=prev_box[3]-prev_box[1],
                         iou_before=before, iou_after=after, shift_x=shift[0], shift_y=shift[1],
                         bottom_before=current[3]-truth[3], bottom_after=corrected[3]-truth[3],
                         **dict(zip(("fixed_" + c for c in BOX), corrected))))
    result = pd.concat([frames.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result["delta_iou"] = result.iou_after - result.iou_before
    result["recovered"] = ~result.iou_before.ge(.5) & result.iou_after.ge(.5)
    result["lost"] = result.iou_before.ge(.5) & ~result.iou_after.ge(.5)
    return result


def summarize(frames, by):
    rows = []
    for key, group in frames.groupby(by, sort=True):
        rows.append({**dict(zip(by, key)), "frames": len(group), "averaged": group.reason.eq("averaged").sum(),
                     "hits_before": group.iou_before.ge(.5).sum(), "hits_after": group.iou_after.ge(.5).sum(),
                     "recovered": group.recovered.sum(), "lost": group.lost.sum(),
                     "better": group.delta_iou.gt(1e-9).sum(), "worse": group.delta_iou.lt(-1e-9).sum(),
                     "mean_iou_before": group.iou_before.mean(), "mean_iou_after": group.iou_after.mean(),
                     "bottom_mae_before": group.bottom_before.abs().mean(),
                     "bottom_mae_after": group.bottom_after.abs().mean()})
    return pd.DataFrame(rows)


def plot_example(frames, video="b02_sn_h_02", frame=33):
    """현재 위치와 배율을 고정해 세 박스를 비교한다. 배경은 원본 RGB다."""
    rgb = read_rgb(VIDEOS / f"{video}.mp4", frame)
    case = frames[frames.video_id.eq(video) & frames.frame_index.eq(frame)].set_index("factor")
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 4), layout="constrained")
    for ax, factor in zip(axes, [1.0, .5, 1.5], strict=True):
        row = case.loc[factor]
        ax.imshow(rgb)
        for prefix, name, color, style in [("gt_", "manual", "white", ":"),
                                          ("", "before", "orange", "--"), ("fixed_", "after", "#00c7ee", "-")]:
            x, y, right, bottom = row[[prefix + col for col in BOX]]
            ax.add_patch(Rectangle((x,y), right-x, bottom-y, fill=False, edgecolor=color,
                                   linestyle=style, linewidth=2, label=name))
        ax.plot((row.x1+row.x2)/2, (row.y1+row.y2)/2, "+", color="#00c7ee", ms=10)
        ax.set(xlim=(row.gt_x1-60, row.gt_x2+60), ylim=(row.gt_y2+60, row.gt_y1-60),
               title=f"S × {factor:g}   |   IoU {row.iou_before:.4f} → {row.iou_after:.4f}")
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{video}   /   frame {frame}   /   boxes on the same original image")
    return fig
