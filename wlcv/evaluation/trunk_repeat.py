"""두 회차의 판독이 끝난 뒤 체간 각도와 그룹 평균을 비교한다."""

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.annotation.trunk_repeat import FIRST_PAIRS, JOINTS, pending_frames, repeat_frames
from wlcv.data.frames import read_rgb
from wlcv.evaluation.trunk import review_trunk

GROUPS = {"A": "국가대표", "B": "현직 트레이너", "C": "동호인"}


def compare_repeats(videos, folder):
    """직접·추정·판독 불가를 보존하며 같은 선수의 두 회차만 짝지어 비교한다."""
    if pending_frames(repeat_frames(videos, folder), folder):
        raise ValueError("12장의 두 관절을 모두 판독한 뒤 비교하세요.")
    frames, trials = [], []
    for video in tqdm(FIRST_PAIRS, desc="두 회차 비교", unit="편"):
        info = videos.loc[video]
        trial = {"선수": info.person_id, "그룹": GROUPS[info.group]}
        phases = {}
        for number in (1, 2):
            row, _ = review_trunk(video, info, folder, pass_number=number)
            trial[f"{number}차 상태"] = row["판독"]
            change = row["직접 변화 (°)"] if row["판독"] == "직접 확인" else row["참고 변화 (°)"]
            trial[f"{number}차 변화 (°)"] = change
            for phase in ("이륙", "무릎 통과"):
                entry = phases.setdefault(phase, {"선수": info.person_id, "영상": video, "시점": phase,
                                                  "프레임": row[f"{phase} 프레임"]})
                entry[f"{number}차 어깨"] = row[f"{phase} 어깨"]
                entry[f"{number}차 골반"] = row[f"{phase} 골반"]
                entry[f"{number}차 각도 (°)"] = row[f"{phase} (°)"]
        frames.extend(phases.values())
        if None in (trial["1차 변화 (°)"], trial["2차 변화 (°)"]):
            trial["비교 구분"] = "비교 불가"
        elif "추정 포함" in (trial["1차 상태"], trial["2차 상태"]):
            trial["비교 구분"] = "추정 포함 참고"
        else:
            trial["비교 구분"] = "직접 확인"
        trials.append(trial)
    frames = pd.DataFrame(frames)
    trials = pd.DataFrame(trials)
    for number in (1, 2):
        frames[f"{number}차 각도 (°)"] = pd.to_numeric(frames[f"{number}차 각도 (°)"])
        change = pd.to_numeric(trials[f"{number}차 변화 (°)"])
        trials[f"{number}차 크기 (°)"] = change.abs()
    frames["각도 차이 (2차−1차, °)"] = frames["2차 각도 (°)"] - frames["1차 각도 (°)"]
    trials["크기 차이 (2차−1차, °)"] = trials["2차 크기 (°)"] - trials["1차 크기 (°)"]
    return frames, trials


def group_comparison(trials):
    """두 회차 모두 비교 가능한 같은 선수만 평균에 넣고 그 수를 표시한다."""
    paired = trials.loc[trials["비교 구분"] != "비교 불가"]
    summary = paired.groupby(["그룹", "비교 구분"], sort=False).agg(
        비교인원=("선수", "size"), **{"1차 평균 크기 (°)": ("1차 크기 (°)", "mean"),
                                "2차 평균 크기 (°)": ("2차 크기 (°)", "mean")})
    return summary.reset_index()


def plot_repeats(trials):
    paired = trials.loc[trials["비교 구분"] != "비교 불가"]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    for i, (_, row) in enumerate(paired.iterrows()):
        ax.plot([row["1차 크기 (°)"], row["2차 크기 (°)"]], [i, i], color="0.75", zorder=1)
    ax.scatter(paired["1차 크기 (°)"], range(len(paired)), s=70, facecolors="none",
               edgecolors="#354c61", label="Pass 1")
    ax.scatter(paired["2차 크기 (°)"], range(len(paired)), marker="x", color="#057d72", label="Pass 2")
    ax.set_yticks(range(len(paired)), paired["선수"])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute change from lift-off to knee pass (°)")
    ax.legend()
    return fig


def plot_point_comparison(frame, videos, folder, video_dir):
    """각도 차이가 큰 한 장을 골라 같은 확대 범위에서 두 판독의 점을 확인한다."""
    video, index = frame["영상"], int(frame["프레임"])
    rgb = read_rgb(video_dir / videos.loc[video, "filename"], index)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.7), sharex=True, sharey=True, layout="constrained")
    xs, ys = [], []
    for number, ax in enumerate(axes, start=1):
        points = load_points(pose_point_path(folder, video, index, pass_number=number))
        ax.imshow(rgb)
        coords = []
        for joint in JOINTS:
            point = points[joint]
            x, y = int(point["x_px"]), int(point["y_px"])
            coords.append((x, y))
            xs.append(x)
            ys.append(y)
            marker = "D" if point["status"] == "estimated" else "+"
            ax.plot(x, y, color="#ffcb48", marker=marker, markersize=9, markerfacecolor="none", zorder=3)
        ax.plot(*zip(*coords), "--", color="#00cfda")
        angle = frame[f"{number}차 각도 (°)"]
        ax.set_title(f"Pass {number}  /  {angle:.1f}°")
        ax.axis("off")
    pad = max(24, (max(ys) - min(ys)) * 0.45)
    axes[0].set_xlim(max(0, min(xs)-pad), min(rgb.shape[1], max(xs)+pad))
    axes[0].set_ylim(min(rgb.shape[0], max(ys)+pad), max(0, min(ys)-pad))
    fig.suptitle(f"{video}  /  frame {index}")
    return fig
