"""무릎 통과부터 접촉까지 모델·수동 원판 박스의 수직 경계 차이를 확인한다."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from wlcv.annotation.plate_blur import review_frames
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plate_track import KEY


def edge_frames(pairs):
    """입력 640의 기존 선택을 읽는다. 아래로 갈수록 원본 y좌표가 증가한다."""
    selected = pairs[pairs.method.eq("Past link")].copy().sort_values(KEY)
    keys = list(selected[KEY].itertuples(index=False, name=None))
    if len(set(keys)) != len(keys) or set(keys) != set(review_frames()):
        raise ValueError("원본에서 판독한 연속 구간과 선택 표가 일치해야 합니다.")
    selected["offset"] = 0
    for video, group in selected.groupby("video_id"):
        contact = load_events(ANNOTATIONS, video, pass_number=1)["drive"]
        selected.loc[group.index, "offset"] = group.frame_index - contact

    height = selected.gt_y2 - selected.gt_y1
    if not (height > 0).all():
        raise ValueError("모든 판독 프레임에 유효한 수동 박스가 필요합니다.")
    selected["bottom"] = selected.y2_640 - selected.gt_y2
    selected["top"] = selected.y1_640 - selected.gt_y1
    selected["center"] = selected.cy_640 - (selected.gt_y1 + selected.gt_y2) / 2
    if not np.allclose(selected.center, (selected.top + selected.bottom) / 2, equal_nan=True):
        raise ValueError("선택 박스와 저장된 중심 좌표가 다릅니다.")
    selected["bottom_height_pct"] = 100 * selected.bottom / height
    return selected


def trailing_run(flags):
    """마지막 프레임에서 거꾸로 세고, 처음 조건을 벗어난 곳에서 멈춘다."""
    length = 0
    for flag in reversed(flags.tolist()):
        if not flag:
            break
        length += 1
    return length


def summarize_edges(frames, threshold_px=0):
    """연속 길이는 접촉 포함이다. 접촉 직전까지만 센 값도 보존한다."""
    summaries = []
    for video, group in frames.groupby("video_id"):
        group = group.sort_values("offset")
        if not np.array_equal(group.offset, np.arange(1-len(group), 1)):
            raise ValueError(f"{video}의 접촉 프레임과 구간 연속성을 확인하세요.")
        lower = group.bottom.gt(threshold_px)
        upper = group.top.gt(threshold_px)
        center = group.center.gt(threshold_px)
        count = trailing_run(lower)
        last = group.iloc[-1]
        summaries.append(dict(video_id=video, person_id=last.person_id, frames=len(group),
                              missing=group.source_640.eq("missing").sum(), contact_frame=last.frame_index,
                              bottom_start=last.frame_index-count+1 if count else pd.NA,
                              bottom_run=count, bottom_run_before=trailing_run(lower.iloc[:-1]),
                              top_run=trailing_run(upper), center_run=trailing_run(center),
                              all_run=trailing_run(lower & upper & center),
                              reaches_window_start=count==len(group), bottom_px=last.bottom,
                              top_px=last.top, center_px=last.center, bottom_height_pct=last.bottom_height_pct))
    return pd.DataFrame(summaries)


def plot_edge_traces(frames):
    """각 선수에서 접촉을 0에 맞춰 세로 오차를 원본 px로 그린다."""
    groups = list(frames.groupby("person_id", sort=True))
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), layout="constrained")
    for ax, (person, group) in zip(axes.flat, groups, strict=True):
        for field, color, marker in [("bottom", "#b34b12", "o"), ("top", "#23699c", "^"),
                                     ("center", "#382e45", "x")]:
            ax.plot(group.offset, group[field], marker=marker, ms=4, lw=1.5, color=color, label=field)
        ax.axhline(0, color="grey", lw=.8)
        ax.set_title(f"{person} | {len(group)} frames")
        ax.grid(alpha=.15)
    axes[0, 0].legend(ncols=3, fontsize=7)
    fig.supxlabel("Frames relative to contact (0 = contact)")
    fig.supylabel("Model minus manual box (px, down +)")
    return fig


def plot_contact_examples(frames, people=("b02", "c04")):
    """마지막 세 원본과 두 박스를 겹쳐 선수별 경계 차이를 확인한다."""
    fig, axes = plt.subplots(len(people), 3, figsize=(9, 3*len(people)), layout="constrained", squeeze=False)
    for row_axes, person in zip(axes, people, strict=True):
        group = frames[frames.person_id.eq(person) & frames.offset.between(-2, 0)].sort_values("offset")
        for ax, row in zip(row_axes, group.itertuples(), strict=True):
            rgb = read_rgb(VIDEOS / f"{row.video_id}.mp4", row.frame_index)
            ax.imshow(rgb)
            truth = (row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2)
            pred = (row.x1_640, row.y1_640, row.x2_640, row.y2_640)
            for box, color, style in [(truth, "gold", "--"), (pred, "deepskyblue", "-")]:
                if np.isfinite(box).all():
                    x, y, right, bottom = box
                    ax.add_patch(Rectangle((x, y), right-x, bottom-y, fill=False, ec=color, ls=style, lw=1.8))
            bounds = np.array([truth, pred])
            ax.set(xlim=(np.nanmin(bounds[:, 0])-20, np.nanmax(bounds[:, 2])+20),
                   ylim=(np.nanmax(bounds[:, 3])+20, np.nanmin(bounds[:, 1])-20),
                   title=f"{person} frame {row.frame_index} | {row.offset:+d}")
            ax.axis("off")
    fig.suptitle("Gold dashed: manual plate box   |   Blue: selected 640 box")
    return fig
