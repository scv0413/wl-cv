"""보이는 양쪽 무릎에서 같은 쪽 연결과 좌우를 바꾼 연결의 픽셀 거리를 비교한다."""

from math import dist, isclose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.config import ANNOTATIONS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.pose_comparison import JOINT_CONF, PERSON_CONF

VIDEO_ID = "a05_sn_f_01"
FRAMES = (84, 87)
SIDES = {"left": "왼쪽", "right": "오른쪽"}
# 원본만 보고 고른 하체 표시 영역 (왼쪽, 위, 오른쪽, 아래). 추론은 전체 프레임을 사용한다.
ROI = (140, 470, 290, 670)
FOLDER = ANNOTATIONS / "a05_bilateral_knees" / "validation"


def read_knees(folder, *, video_id=VIDEO_ID, frames=FRAMES, roi=ROI, original_sources=None):
    """기존 직접점은 지정된 원본 폴더에서만 읽고 새 판독과 섞어 쓰지 않는다."""
    original_sources = original_sources or {}
    rows = []
    for frame in frames:
        for side, name in SIDES.items():
            source = original_sources.get((frame, side), folder / side)
            path = pose_point_path(source, video_id, frame)
            point = load_points(path).get("무릎")
            if (frame, side) in original_sources and (not point or point["status"] != "located"):
                raise ValueError(f"{path.name}에서 기존 무릎 직접점을 확인하세요.")
            status, x, y = "미검수", np.nan, np.nan
            if point is not None:
                keys = ("video_id", "frame_index", "person_id", "split", "side")
                if tuple(point[k] for k in keys) != (video_id, str(frame), video_id.split("_")[0], "validation", side):
                    raise ValueError(f"{path.name}의 영상·프레임·선수·분할·좌우를 확인하세요.")
                if point["status"] == "located":
                    status, x, y = "직접 확인", float(point["x_px"]), float(point["y_px"])
                    if not (roi[0] <= x < roi[2] and roi[1] <= y < roi[3]):
                        raise ValueError(f"{frame}번 {name} 무릎 좌표가 판독 영역 밖에 있습니다.")
                elif point["status"] == "unreadable" and not (point["x_px"] or point["y_px"]):
                    status = "판독 불가"
                else:
                    raise ValueError(f"{frame}번 {name} 무릎을 직접 확인 또는 판독 불가로 저장하세요.")
            rows.append({"프레임": frame, "방향": name, "판독": status, "수동 x": x, "수동 y": y})
    return pd.DataFrame(rows)


def compare_pair(rows):
    """수동 왼쪽/오른쪽을 기준으로 네 거리를 계산한다. 누락된 쌍은 합산하지 않는다."""
    points = rows.set_index("방향")
    manual = points.loc[["왼쪽", "오른쪽"], ["수동 x", "수동 y"]].to_numpy(dtype=float)
    predicted = points.loc[["왼쪽", "오른쪽"], ["예측 x", "예측 y"]].to_numpy(dtype=float)
    ll, lr = (dist(manual[0], p) for p in predicted)
    rl, rr = (dist(manual[1], p) for p in predicted)
    same, crossed = ll + rr, lr + rl
    if points["판독"].ne("직접 확인").any():
        verdict = "수동 판독 불가"
    elif points["예측 상태"].ne("표시").any():
        verdict = "예측 비교 불가"
    elif isclose(same, crossed, abs_tol=1e-8):
        verdict = "동률"
    elif max(lr, rl) < min(ll, rr):
        verdict = "양쪽 모두 교차 대응 지지"
    elif max(ll, rr) < min(lr, rl):
        verdict = "양쪽 모두 같은 쪽 대응 지지"
    else:
        verdict = "교차 합만 작음" if crossed < same else "같은 쪽 합만 작음"
    return {"수동L–예측L": ll, "수동L–예측R": lr, "수동R–예측L": rl, "수동R–예측R": rr,
            "같은 쪽 합(px)": same, "교차 합(px)": crossed, "같은 쪽 − 교차(px)": same - crossed,
            "판정": verdict}


def evaluate(model, video, manual, *, frames=FRAMES):
    if manual["판독"].eq("미검수").any():
        raise ValueError("대상 프레임의 왼쪽·오른쪽 판독을 모두 저장하세요.")
    coordinates, scores = [], []
    for frame in tqdm(frames, desc="양쪽 무릎 비교", unit="장"):
        rgb = read_rgb(video, frame)
        result = predict_rgb(model, rgb, conf=PERSON_CONF, imgsz=640, device="cpu").cpu().numpy()
        count = len(result.boxes)
        joints = joint_table(result, 1, JOINT_CONF) if count == 1 else None
        frame_rows = manual[manual["프레임"] == frame].copy().set_index("방향")
        for side, name in SIDES.items():
            x = y = confidence = np.nan
            status = "미탐" if count == 0 else "수행자 구분 불가"
            if joints is not None:
                point = joints[(joints["관절"] == "무릎") & (joints["방향"] == side)].iloc[0]
                status, confidence = point["표시 상태"], point["관절 신뢰도"]
                if status == "표시":
                    x, y = point["x (px)"], point["y (px)"]
            frame_rows.loc[name, ["예측 x", "예측 y", "신뢰도", "예측 상태"]] = [x, y, confidence, status]
        frame_rows = frame_rows.reset_index()
        coordinates.append(frame_rows)
        scores.append({"프레임": frame, **compare_pair(frame_rows)})
    return pd.concat(coordinates, ignore_index=True), pd.DataFrame(scores)


def draw_assignments(video, coordinates, scores, *, frames=FRAMES, roi=ROI):
    """한 프레임에 두 패널을 써서 동일한 네 점의 연결만 바꾸어 보여 준다."""
    fig, axes = plt.subplots(len(frames), 2, figsize=(8, 3.5 * len(frames)), squeeze=False, layout="constrained")
    colors = {"왼쪽": "#28d7fd", "오른쪽": "#ffbf32"}
    for index, frame in enumerate(frames):
        points = coordinates[coordinates["프레임"] == frame].set_index("방향")
        score = scores.set_index("프레임").loc[frame]
        rgb = read_rgb(video, frame)
        for col, (name, field, targets) in enumerate([
            ("Same side", "같은 쪽 합(px)", ("왼쪽", "오른쪽")),
            ("Crossed", "교차 합(px)", ("오른쪽", "왼쪽")),
        ]):
            ax = axes[index, col]
            ax.imshow(rgb)
            for side, target in zip(("왼쪽", "오른쪽"), targets):
                mx, my = points.loc[side, ["수동 x", "수동 y"]]
                px, py = points.loc[target, ["예측 x", "예측 y"]]
                ax.plot([mx, px], [my, py], color=colors[side], linestyle="--", linewidth=1.7)
                ax.plot(mx, my, "o", markersize=12, markerfacecolor="none", markeredgewidth=2,
                        markeredgecolor=colors[side])
                ax.plot(px, py, "X", markersize=7, color=colors[target], markeredgecolor="black")
            # 비교에 쓰인 점이 판독 영역 밖에 있어도 그림 안에 함께 담는다.
            xs = points[["수동 x", "예측 x"]].stack()
            ys = points[["수동 y", "예측 y"]].stack()
            ax.set_xlim(max(0, min(roi[0], xs.min()-10)), min(rgb.shape[1], max(roi[2], xs.max()+10)))
            ax.set_ylim(min(rgb.shape[0], max(roi[3], ys.max()+10)), max(0, min(roi[1], ys.min()-10)))
            total = score[field]
            value = f"{total:.1f} px" if pd.notna(total) else "unavailable"
            ax.set_title(f"#{frame} | {name} | {value}", fontsize=10)
            ax.axis("off")
    handles = [Line2D([], [], color=c, label=n) for n, c in zip(("Left", "Right"), colors.values())]
    handles.extend([Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color="black", label="Manual"),
                    Line2D([], [], marker="X", linestyle="none", color="black", label="YOLO")])
    fig.legend(handles=handles, loc="outside lower center", ncols=4)
    return fig
