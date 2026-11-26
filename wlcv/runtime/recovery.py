"""캐치부터 피니시까지의 원판과 어깨 거리·체간 기울기."""
import pandas as pd


def landmarks(rows):
    """중간의 최대 후방 간격과 시작·종료 프레임을 같은 그림에서 확인한다."""
    valid = rows.forward_px.dropna()
    frames = [rows.index[0], rows.index[-1]]
    if not valid.empty:
        frames.append(valid.idxmin())
    return sorted(set(frames))


def plot_interval(rows, *, title, note, fig=None):
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()  # 노트북에서 표시할 그림
    fig.set_size_inches(9.8, 6.0)
    fig.set_layout_engine("constrained")
    axes = fig.subplots(2, 1, sharex=True)
    fig.suptitle(f"{title}    /    Catch to finish", fontweight="bold")
    guides = landmarks(rows)
    for ax, column, color, title, units in [
        (axes[0], "forward_px", "#006f69", "Plate relative to near shoulder   (+ forward / - behind)", "Distance (px)"),
        (axes[1], "trunk_angle_deg", "#8d4a18", "Near shoulder–hip line from vertical   (0 = upright)", "Angle (deg)"),
    ]:
        y = rows[column]
        ax.plot(rows.index, y, ".-", color=color, linewidth=2, markersize=5)
        ax.axhline(0, color="0.4", linewidth=.8)
        for frame in guides:
            ax.axvline(frame, color="0.75", linestyle=":", linewidth=1)
            if pd.notna(y.loc[frame]):
                ax.scatter(frame, y.loc[frame], s=56, facecolors="white", edgecolors=color, zorder=3)
                ax.annotate(f"{y.loc[frame]:.1f}", (frame, y.loc[frame]), xytext=(0, 9),
                            textcoords="offset points", ha="center", color=color)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(units)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.22, x=.05)
    ticks = sorted(set(range(rows.index[0], rows.index[-1] + 1, 5)) | set(guides))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xlabel("Original frame (0-based)")
    fig.supxlabel(note,
                  fontsize=9, color="0.3")
    return fig


def save_recovery_curve(rows, events, folder):
    """수동 캐치·피니시가 있을 때만 해당 구간의 모델 관측을 표시한다."""
    catch, finish = events.get("catch"), events.get("finish")
    if catch is None or finish is None:
        return {"reason": "캐치 또는 피니시 시점 미지정"}
    interval = rows.set_index("frame_index").loc[catch:finish]
    offsets = interval.forward_px.dropna()
    angles = interval.trunk_angle_deg
    result = {"catch_frame": catch, "finish_frame": finish, "distance_frames": len(offsets),
              "angle_frames": int(angles.count()), "minimum_forward_frame": None, "minimum_forward_px": None,
              "trunk_change_deg": None, "reason": ""}
    if not offsets.empty:
        result.update(minimum_forward_frame=int(offsets.idxmin()), minimum_forward_px=float(offsets.min()))
    if pd.notna(angles.loc[catch]) and pd.notna(angles.loc[finish]):
        result["trunk_change_deg"] = float(angles.loc[finish] - angles.loc[catch])
    if offsets.empty and angles.isna().all():
        result["reason"] = "구간 내 원판·어깨 거리와 체간 관측이 없습니다."
        return result
    from matplotlib.figure import Figure
    # HTTP 요청은 작업 스레드에서 실행된다. PNG 저장에는 화면 창이 필요 없다.
    figure = plot_interval(interval, title="Plate and pose",
                           note="Model estimates in source pixels and degrees.", fig=Figure())
    figure.savefig(folder / "catch_to_finish.png", dpi=145)
    return result
