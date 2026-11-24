"""검증 12프레임에서 가까운 어깨·골반의 모델 예측과 두 수동 판독을 비교한다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from wlcv.annotation.trunk_repeat import FIRST_PAIRS, JOINTS, pending_frames, repeat_frames
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.trunk import review_trunk, vertical_angle


def predict_frames(model, videos, folder, video_dir):
    frames = repeat_frames(videos, folder)
    if pending_frames(frames, folder):
        raise ValueError("22번 실험의 2차 판독을 먼저 완료하세요.")
    results = {}
    for video, frame in tqdm(sorted(frames), desc="자세 추론", unit="장"):
        rgb = read_rgb(video_dir / videos.at[video, "filename"], frame)
        pred = predict_rgb(model, rgb, conf=0.25, imgsz=640, device="cpu")
        results[video, frame] = pred.cpu().numpy()
    return results


def candidate_gallery(results):
    fig, axes = plt.subplots(3, 4, figsize=(11, 9), layout="constrained")
    for ax, ((video, frame), result) in zip(axes.flat, results.items()):
        ax.imshow(result.orig_img[:, :, ::-1])
        for n, (x, y, right, bottom) in enumerate(result.boxes.xyxy, 1):
            ax.add_patch(Rectangle((x, y), right-x, bottom-y, fill=False, color="cyan", lw=1))
            ax.text(x, y, str(n), color="gold", backgroundcolor="black", fontsize=8)
        ax.set_title(f"{video} / {frame}", fontsize=9)
        ax.axis("off")
    return fig


def compare_frames(results, videos, folder, choices, min_confidence=0.50):
    frames, joints = [], []
    for video, pair in FIRST_PAIRS.items():
        info = videos.loc[video]
        reads = [review_trunk(video, info, folder, pass_number=n) for n in (1, 2)]
        for phase, frame in zip(("이륙", "무릎 통과"), pair):
            result = results[video, frame]
            count = len(result.boxes)
            number = choices.get((video, frame), 1 if count == 1 else None)
            base = {"영상": video, "선수": info.person_id, "시점": phase, "프레임": frame, "방향": info.near_side}
            row = {**base, "후보 수": count, "선택 번호": number, "사람 신뢰도": np.nan,
                   "모델 상태": "수행자 미확인" if count else "사람 미검출",
                   "모델 각도 (°)": np.nan, "후보 각도 (°)": np.nan}
            pred = None
            if number is not None and 1 <= number <= count:
                row["사람 신뢰도"] = float(result.boxes.conf[number-1])
                points = joint_table(result, number, min_confidence)
                pred = points[points["방향"] == info.near_side].set_index("관절")

            for n, (manual, _) in enumerate(reads, 1):
                row[f"{n}차 각도 (°)"] = manual[f"{phase} (°)"]
            xy, states = [], []
            for joint in JOINTS:
                x = y = conf = np.nan
                state = row["모델 상태"]
                if pred is not None:
                    point = pred.loc[joint]
                    x, y, conf, state = point[["x (px)", "y (px)", "관절 신뢰도", "표시 상태"]]
                xy.append((x, y))
                states.append(state)
                item = {**base, "관절": joint, "모델 x": x, "모델 y": y, "신뢰도": conf, "상태": state}
                row[joint+" 신뢰도"] = conf
                for n, (manual, details) in enumerate(reads, 1):
                    p = details[phase].get(joint)
                    mx = my = np.nan
                    if p and p["status"] != "unreadable":
                        mx, my = int(p["x_px"]), int(p["y_px"])
                    item.update({f"{n}차 x": mx, f"{n}차 y": my, f"{n}차 판독": manual[f"{phase} {joint}"],
                                 f"모델−{n}차 거리 (px)": np.hypot(x-mx, y-my)})
                joints.append(item)
            if pred is not None:
                angle = vertical_angle(*xy) if np.isfinite(xy).all() else None
                if angle is None:
                    row["모델 상태"] = "좌표 없음 또는 겹침"
                else:
                    row["후보 각도 (°)"] = angle
                    if states == ["표시", "표시"]:
                        row["모델 상태"], row["모델 각도 (°)"] = "기준 충족", angle
                    else:
                        row["모델 상태"] = "관절 저신뢰"
            frames.append(row)
    frames = pd.DataFrame(frames)
    for n in (1, 2):
        frames[f"모델−{n}차 (°)"] = frames["모델 각도 (°)"] - pd.to_numeric(frames[f"{n}차 각도 (°)"])
    return frames, pd.DataFrame(joints)


def trial_changes(frames):
    rows = []
    for person, pair in frames.groupby("선수", sort=False):
        pair = pair.set_index("시점")
        row = {"선수": person}
        for source in ("1차", "2차", "모델"):
            angles = pd.to_numeric(pair[source+" 각도 (°)"])
            row[source] = angles["무릎 통과"] - angles["이륙"]
        a, b, model = row["1차"], row["2차"], row["모델"]
        if not np.isfinite([a, b]).all():
            verdict = "수동 비교 불가"
        elif np.sign(a) != np.sign(b):
            verdict = "수동 방향 불일치"
        elif not np.isfinite(model):
            verdict = "모델 비교 불가"
        else:
            verdict = "같은 방향" if np.sign(a) == np.sign(model) else "방향 불일치"
        row.update({"모델−1차 (°)": model-a, "모델−2차 (°)": model-b, "방향 판정": verdict})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_changes(trials):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    y = np.arange(len(trials))
    for source, label, marker, color, offset in [
            ("1차", "Manual 1", "o", "#225587", -0.13), ("2차", "Manual 2", "x", "#00896e", 0),
            ("모델", "YOLO", "^", "#c64b10", 0.13)]:
        ax.scatter(trials[source], y + offset, label=label, marker=marker, color=color, s=48)
    ax.axvline(0, color="0.55", lw=1)
    ax.set_yticks(y, trials["선수"])
    ax.invert_yaxis()
    ax.set_xlabel("Knee pass minus lift-off (degrees)")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(ncol=3)
    return fig


def plot_trial(person, frames, joints, results):
    """같은 영상의 두 원본을 같은 범위로 확대한다. 저신뢰 모델선은 점선으로 표시한다."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), sharex=True, sharey=True, layout="constrained")
    xs, ys = [], []
    for ax, (_, row) in zip(axes, frames.loc[frames["선수"] == person].iterrows()):
        video, frame = row["영상"], row["프레임"]
        ax.imshow(results[video, frame].orig_img[:, :, ::-1])
        pts = joints.loc[(joints["영상"] == video) & (joints["프레임"] == frame)]
        for label, prefix, marker, color in [("Manual 1", "1차", "+", "#ffcb36"),
                                              ("Manual 2", "2차", "x", "#41f3ef"),
                                              ("YOLO", "모델", "o", "#ff42ba")]:
            px, py = pts[prefix+" x"], pts[prefix+" y"]
            weak = prefix == "모델" and row["모델 상태"] != "기준 충족"
            ax.plot(px, py, ":" if weak else "-", marker=marker, markerfacecolor="none",
                    markersize=9, markeredgewidth=1.7, lw=1.6, color=color, label=label)
            xs.extend(px.dropna())
            ys.extend(py.dropna())
        phase = "Lift-off" if row["시점"] == "이륙" else "Knee pass"
        ax.set_title(f"{phase} / {frame}")
        ax.axis("off")
    if xs and ys:
        pad = max(30, 0.50 * max(max(xs)-min(xs), max(ys)-min(ys)))
        axes[0].set_xlim(min(xs)-pad, max(xs)+pad)
        axes[0].set_ylim(max(ys)+pad, min(ys)-pad)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{video} / {row['방향']} side")
    return fig
