"""원본 좌표를 유지하며 운동 시점별 골반 판독을 나란히 보여준다."""

import matplotlib.pyplot as plt

from wlcv.data.frames import read_rgb


def plot_review(events, people, video_files, video_dir, show_points=True):
    """행은 이륙·무릎 통과, 열은 사람이다. 표시를 끄면 원본만 검수할 수 있다."""
    phases = ["이륙 후보", "무릎 통과"]
    fig, axes = plt.subplots(2, len(people), figsize=(3 * len(people), 7), squeeze=False)
    for col, person in enumerate(people):
        rows = events.loc[events["person_id"].eq(person)].set_index("phase")
        for row, phase in enumerate(phases):
            frame = rows.loc[phase]
            rgb = read_rgb(video_dir / video_files[frame.video_id], int(frame.frame_index))
            height, width = rgb.shape[:2]
            ax = axes[row, col]
            ax.imshow(rgb)
            # 같은 영상의 두 시점을 같은 비율로 확대한다. 좌표 단위는 원본 픽셀이다.
            ax.set_xlim(width * 0.16, width * 0.85)
            ax.set_ylim(height * 0.86, height * 0.41)
            candidate = frame.screen == "추정 후보"
            if show_points and candidate:
                ax.plot(frame.read1_x, frame.read1_y, "+", color="#00f5ed", ms=13, mew=2)
                ax.plot(frame.read2_x, frame.read2_y, "o", mfc="none", mec="#ffad38", ms=10, mew=2)
            status = "estimated" if candidate else "unresolved"
            stage = "Lift-off" if row == 0 else "Knee pass"
            ax.set_title(f"{person} ({frame.near_side})   f{int(frame.frame_index)}\n{stage}  |  {status}",
                         fontsize=11)
            ax.axis("off")
    if show_points:
        fig.suptitle("Near-side hip estimates   |   + first read     ○ second read", fontsize=16)
    else:
        fig.suptitle("Near-side hip review   |   original frames", fontsize=16)
    fig.tight_layout()
    return fig


def plot_hip_reads(points, video_files, video_dir):
    """한 열에 한 장면을 놓고 아래에서 같은 위치를 원본 픽셀 기준으로 확대한다."""
    from matplotlib.lines import Line2D

    groups = points.groupby(["video_id", "frame_index"], sort=False)
    fig, axes = plt.subplots(2, len(groups), figsize=(4 * len(groups), 7), squeeze=False)
    colors = {"트레이너": "#ffd746", "예비": "#00e1ff"}
    markers = {1: "o", 2: "+"}
    for col, ((video, frame), rows) in enumerate(groups):
        rgb = read_rgb(video_dir / video_files[video], frame)
        height, width = rgb.shape[:2]
        valid = rows.dropna(subset=["x", "y"])
        for ax in axes[:, col]:
            ax.imshow(rgb)
            for point in valid.itertuples():
                ax.plot(point.x, point.y, marker=markers[point.round_number],
                        color=colors[point.reader], markerfacecolor="none",
                        markersize=10, markeredgewidth=1.8)
            ax.axis("off")
        ax = axes[0, col]
        ax.set_xlim(width * 0.16, width * 0.90)
        ax.set_ylim(height * 0.88, height * 0.40)
        ax.set_title(f"{video[:3]}   |   frame {frame}")
        if not valid.empty:
            x, y = valid[["x", "y"]].mean()
            axes[1, col].set_xlim(x - 75, x + 75)
            axes[1, col].set_ylim(y + 75, y - 75)
    handles = [Line2D([], [], color=color, linewidth=4, label=label)
               for color, label in [(colors["트레이너"], "Trainer"), (colors["예비"], "Preliminary")]]
    handles.extend(Line2D([], [], color="black", marker=marker, markerfacecolor="none",
                          linestyle="none", markersize=9, label=f"Read {number}")
                   for number, marker in markers.items())
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
