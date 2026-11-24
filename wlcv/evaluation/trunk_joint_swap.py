"""같은 체간 선에서 어깨 또는 골반만 수동점으로 바꿔 두 시점의 각도 변화를 비교한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wlcv.annotation.trunk_repeat import FIRST_PAIRS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.trunk import vertical_angle


# 왼쪽 항목은 어깨의 출처, 오른쪽 항목은 골반의 출처다.
CONDITIONS = {
    "원래 모델": ("모델", "모델"),
    "어깨만 교체": ("수동", "모델"),
    "골반만 교체": ("모델", "수동"),
    "두 점 수동": ("수동", "수동"),
}
VIDEOS_TO_REVIEW = ("a05_sn_f_01", "b02_sn_h_02")
PHASES = ("이륙", "무릎 통과")


def swap_angles(frames, joints):
    """23번 실험이 저장한 원본 픽셀 좌표로 각도를 다시 계산한다."""
    frames = frames.set_index(["영상", "프레임"], verify_integrity=True)
    joints = joints.set_index(["영상", "프레임", "관절"], verify_integrity=True)
    rows = []
    for video in tqdm(VIDEOS_TO_REVIEW, desc="관절 교체 비교", unit="편"):
        for phase, index in zip(PHASES, FIRST_PAIRS[video]):
            info = frames.loc[video, index]
            shoulder = joints.loc[video, index, "어깨"]
            hip = joints.loc[video, index, "골반"]
            if (info["시점"] != phase or info["모델 상태"] != "기준 충족"
                    or any(p["시점"] != phase or p["방향"] != info["방향"]
                           or p["선수"] != info["선수"] for p in (shoulder, hip))):
                raise ValueError(f"{video} {index}번의 시점·수행자·모델 상태를 확인하세요.")
            for n in (1, 2):
                sources = {"수동": f"{n}차", "모델": "모델"}
                for name, (shoulder_source, hip_source) in CONDITIONS.items():
                    s = sources[shoulder_source]
                    h = sources[hip_source]
                    sx, sy = shoulder[s+" x"], shoulder[s+" y"]
                    hx, hy = hip[h+" x"], hip[h+" y"]
                    angle = vertical_angle((sx, sy), (hx, hy))
                    if angle is None or not np.isfinite(angle):
                        raise ValueError(f"{video} {index}번 {name}의 좌표를 확인하세요.")
                    # 양 끝 대조군이 직전 실험의 모델 각도와 해당 수동 회차 각도를 재현해야 한다.
                    if s == h and not np.isclose(angle, info[s+" 각도 (°)"], rtol=0, atol=0.001):
                        raise ValueError(f"{video} {index}번에서 23번 실험의 각도를 재현하지 못했습니다.")
                    rows.append({"영상": video, "선수": info["선수"], "시점": phase, "프레임": index,
                                 "방향": info["방향"], "판독 회차": n, "조건": name,
                                 "수동 어깨 판독": shoulder[f"{n}차 판독"], "수동 골반 판독": hip[f"{n}차 판독"],
                                 "어깨 x": sx, "어깨 y": sy, "골반 x": hx, "골반 y": hy, "각도 (°)": angle})
    return pd.DataFrame(rows)


def changes_by_condition(angles):
    wide = angles.pivot(index=["선수", "판독 회차", "조건"], columns="시점", values="각도 (°)")
    change = (wide["무릎 통과"] - wide["이륙"]).unstack("조건")
    return change[list(CONDITIONS)].reset_index().rename_axis(columns=None)


def plot_changes(change):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True, layout="constrained")
    names = ["YOLO", "Manual shoulder", "Manual hip", "Manual both"]
    for ax, (person, group) in zip(axes, change.groupby("선수", sort=False)):
        ax.axvline(0, color="#718086", lw=1)
        for _, row in group.iterrows():
            n = int(row["판독 회차"])
            offset = -0.09 if n == 1 else 0.09
            ax.scatter(row[list(CONDITIONS)].astype(float), np.arange(4) + offset,
                       marker="o" if n == 1 else "x", color=["#265789", "#b85216"][n-1],
                       s=48, label=f"Manual pass {n}")
        ax.set_yticks(range(4), names)
        ax.set_title(person)
        ax.set_xlabel("Knee pass minus lift-off (degrees)")
        ax.grid(axis="x", alpha=0.18)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="upper right")
    return fig


def plot_overlays(angles, video, pass_number, video_path):
    """원래 모델선을 점선으로 남기고 단독 교체한 두 조건을 나란히 보여 준다."""
    subset = angles.loc[(angles["영상"] == video) & (angles["판독 회차"] == pass_number)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True, layout="constrained")
    for i, phase in enumerate(PHASES):
        points = subset.loc[subset["시점"] == phase].set_index("조건")
        base = points.loc["원래 모델"]
        frame = int(base["프레임"])
        rgb = read_rgb(video_path, frame)
        for col, condition in enumerate(("어깨만 교체", "골반만 교체")):
            ax = axes[i, col]
            ax.imshow(rgb)
            ax.plot([base["어깨 x"], base["골반 x"]], [base["어깨 y"], base["골반 y"]],
                    "--", color="white", lw=1.7, label="Original YOLO")
            p = points.loc[condition]
            ax.plot([p["어깨 x"], p["골반 x"]], [p["어깨 y"], p["골반 y"]], color="#00d0d4", lw=2)
            joint = ("어깨", "골반")[col]
            ax.plot(p[joint+" x"], p[joint+" y"], marker="D", color="#ffd256", markerfacecolor="none",
                    markersize=8, markeredgewidth=2)
            title = ("Manual shoulder", "Manual hip")[col]
            ax.set_title(f"{title} / frame {frame} / {p['각도 (°)']:.1f}°", fontsize=10)
            ax.axis("off")
    xs = subset[["어깨 x", "골반 x"]].to_numpy()
    ys = subset[["어깨 y", "골반 y"]].to_numpy()
    pad = 0.32 * max(np.ptp(xs), np.ptp(ys))
    axes[0, 0].set_xlim(max(0, xs.min()-pad), min(rgb.shape[1], xs.max()+pad))
    axes[0, 0].set_ylim(min(rgb.shape[0], ys.max()+pad), max(0, ys.min()-pad))
    fig.suptitle(f"{video} / manual pass {pass_number}")
    return fig
