"""무릎 통과와 바 접촉 사이의 수동 허브·예측 박스 중심 상승량을 비교한다."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wlcv.annotation.plate_blur import REVIEW_FILE
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, SPLIT, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plate_track import KEY

SIZES = (640, 800)


def phase_rises(pairs):
    """1차 판독의 두 이벤트만 사용한다. 판독 불가나 출력 누락은 NaN으로 남긴다."""
    saved = pairs[pairs.method == "Past link"].set_index(KEY, verify_integrity=True)
    hubs = pd.read_csv(REVIEW_FILE).set_index(KEY, verify_integrity=True)
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    if set(saved.index) != set(hubs.index):
        raise ValueError("기존 판독과 저장된 원본 선택 결과의 프레임이 다릅니다.")

    rows = []
    for video in sorted(hubs.index.get_level_values("video_id").unique()):
        if split.at[video, "role"] != "validation":
            raise ValueError(f"{video}의 검증 분할을 확인하세요.")
        events = load_events(ANNOTATIONS, video, pass_number=1)
        knee, contact = events["transition"], events["drive"]
        if knee >= contact:
            raise ValueError(f"{video}의 무릎 통과와 접촉 순서를 확인하세요.")
        for size in SIZES:
            row = dict(video_id=video, person_id=split.at[video, "person_id"], imgsz=size)
            for phase, frame in [("knee", knee), ("contact", contact)]:
                truth = hubs.loc[(video, frame)]
                choice = saved.loc[(video, frame)]
                readable = truth.hub_source in ("clicked", "previous_blind")
                manual_y = truth.manual_y if readable else np.nan
                source = choice[f"source_{size}"]
                model_y = choice[f"cy_{size}"] if source != "missing" else np.nan
                row.update({f"{phase}_frame": frame, f"{phase}_source": source,
                            f"{phase}_manual_y": manual_y, f"{phase}_model_y": model_y,
                            f"{phase}_bias_px": model_y - manual_y})
            # 화면 위로 갈수록 y좌표가 작아지므로 시작점에서 끝점을 뺀다.
            row["manual_rise_px"] = row["knee_manual_y"] - row["contact_manual_y"]
            row["model_rise_px"] = row["knee_model_y"] - row["contact_model_y"]
            row["rise_error_px"] = row["model_rise_px"] - row["manual_rise_px"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_rise_error(rises):
    """왼쪽은 실제 두 끝점 오차, 오른쪽은 두 오차의 차이로 생긴 상승량 오차다."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    people = sorted(rises.person_id.unique())
    for size, offset, color in [(640, -.15, "#256590"), (800, .15, "#c35620")]:
        rows = rises[rises.imgsz == size].set_index("person_id").loc[people]
        x = np.arange(len(rows)) + offset
        axes[0].vlines(x, rows.knee_bias_px, rows.contact_bias_px, color=color)
        axes[0].scatter(x, rows.knee_bias_px, marker="o", facecolors="white", edgecolors=color)
        axes[0].scatter(x, rows.contact_bias_px, marker="x", color=color)
        axes[1].bar(x, rows.rise_error_px, width=.28, color=color, label=str(size))
    for ax in axes:
        ax.set_xticks(range(len(people)), people)
        ax.axhline(0, color="#50555a", lw=.8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
    axes[0].set(title="Knee pass (o) to contact (×)", ylabel="Center minus manual hub (px, down +)")
    axes[1].set(title="Rise error by athlete", ylabel="Model rise minus manual rise (px)")
    axes[1].legend(title="Input size")
    return fig


def plot_endpoints(pairs, rises, video):
    """실제 원본의 같은 좌표계에 허브와 두 모델 입력 크기의 박스 중심을 겹친다."""
    phase = rises[rises.video_id == video].iloc[0]
    saved = pairs[(pairs.video_id == video) & (pairs.method == "Past link")].set_index("frame_index")
    hubs = pd.read_csv(REVIEW_FILE).set_index(KEY)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, index, label in zip(axes, [phase.knee_frame, phase.contact_frame], ["Knee pass", "Contact"]):
        row = saved.loc[index]
        point = hubs.loc[(video, index)]
        ax.imshow(read_rgb(VIDEOS / f"{video}.mp4", index))
        if point.hub_source in ("clicked", "previous_blind"):
            ax.plot(point.manual_x, point.manual_y, "o", color="gold", fillstyle="none", ms=12, mew=2, label="Manual hub")
        for size, color, marker in [(640, "deepskyblue", "x"), (800, "magenta", "+")]:
            ax.plot(row[f"cx_{size}"], row[f"cy_{size}"], marker, color=color, ms=12, mew=2, label=f"{size} center")
        ax.set(xlim=(row.gt_x1-25, row.gt_x2+25), ylim=(row.gt_y2+25, row.gt_y1-25),
               title=f"{label} | frame {index}")
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    return fig
