"""같은 영상·같은 후보로 시간 연결을 켜고 끈 결과를 프레임 단위로 대조한다."""

from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics.utils.metrics import box_iou

from wlcv.data.frames import frames_rgb
from wlcv.evaluation.plate_track import BOX
from wlcv.evaluation.video_overlay import PLATE_STATES, draw_plate


def phase_table(detail, lift, knee, drive):
    rows = []
    for label, start, end in [("이륙 → 무릎 통과", lift, knee), ("무릎 통과 → 바 접촉", knee, drive)]:
        interval = detail[detail.frame_index.between(start, end)]
        for method, group in interval.groupby("method", sort=False):
            frames = group.set_index("frame_index", verify_integrity=True)
            rise = frames.at[start, "cy"] - frames.at[end, "cy"]
            rows.append({"구간": label, "시작": start, "끝": end, "방식": method,
                         "전체": len(frames), "판독 가능": frames.match.count(),
                         "정탐": frames.match.sum(), "시간 연결": frames.source.eq("linked").sum(),
                         "선택한 원판의 상승량(px)": rise})
    return pd.DataFrame(rows)


def trajectory(detail, start, end):
    """가운데 한 프레임을 놓치면 선이 끊어진다. y축 방향은 원본과 같다."""
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 3.5), layout="constrained")
    interval = detail[detail.frame_index.between(start, end)]
    for ax, method in zip(axes, ["YOLO", "Past link"]):
        rows = interval[interval.method == method].sort_values("frame_index")
        ref_y = (rows.gt_y1 + rows.gt_y2) / 2
        ax.plot(rows.frame_index, ref_y, "--", color="#9b4a81", label="Reviewed box")
        ax.plot(rows.frame_index, rows.cy, "o-", color="#287450", label="Selected box")
        linked = rows[rows.source == "linked"]
        ax.scatter(linked.frame_index, linked.cy, marker="D", s=60, c="#f4b244",
                   edgecolor="black", zorder=4, label="Linked candidate")
        ax.set(title=method, xlabel="Frame (0-based)", xticks=range(start, end+1))
        ax.grid(axis="y", alpha=.25)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Plate box center y (source px)")
    axes[0].legend(fontsize=8)
    return fig


def audit_candidates(detail, candidates, checks, indices):
    """선택이 끝난 뒤 검수 박스를 열어 해당 위치에 후보가 있었는지 확인한다."""
    selected = detail[detail.method.eq("Past link")].set_index("frame_index")
    rows = []
    for index in indices:
        result = selected.loc[index]
        boxes = candidates[candidates.frame_index.eq(index)].reset_index(drop=True)
        row = {"frame_index": index, "candidate_count": len(boxes)}
        if len(boxes) and pd.notna(result.gt_x1):
            gt = torch.tensor([[result.gt_x1, result.gt_y1, result.gt_x2, result.gt_y2]], dtype=torch.float32)
            predictions = torch.tensor(boxes[BOX].to_numpy(), dtype=torch.float32)
            overlaps = box_iou(gt, predictions)[0]
            best = int(overlaps.argmax())
            row.update(best_candidate_iou=float(overlaps[best]), candidate_confidence=boxes.at[best, "confidence"])
        rows.append(row)
    distances = checks[["frame_index", "expected_x", "expected_y", "nearest_px", "radius_px", "inside"]]
    return pd.DataFrame(rows).merge(distances, on="frame_index", how="left", validate="one_to_one")


def panel(bgr, row, trail, title, check=None):
    """원본 좌표로 박스와 반경을 그리고, 작은 영상에서도 읽히도록 제목은 별도로 붙인다."""
    image = bgr.copy()
    draw_plate(image, row, trail)
    if pd.notna(row.gt_x1):
        cv2.rectangle(image, (round(row.gt_x1), round(row.gt_y1)),
                      (round(row.gt_x2), round(row.gt_y2)), (215, 75, 225), 2)
    if check is not None:
        center = (round(check.expected_x), round(check.expected_y))
        cv2.circle(image, center, round(check.radius_px), (255, 215, 60), 2)
        cv2.drawMarker(image, center, (255, 215, 60), cv2.MARKER_TILTED_CROSS, 18, 2)
    height, width = image.shape[:2]
    preview = cv2.resize(image, (360, 2 * round(180 * height / width)), interpolation=cv2.INTER_AREA)
    result = cv2.copyMakeBorder(preview, 70, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 27, 29))
    _, source, color = PLATE_STATES[row.source]
    score = "" if pd.isna(row.confidence) else f" {row.confidence:.3f}"
    overlap = "--" if pd.isna(row.iou) else f"{row.iou:.3f}"
    cv2.putText(result, f"{title}   |   frame {row.frame_index}", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .66, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(result, f"{source}{score}    IoU {overlap}", (8, 53),
                cv2.FONT_HERSHEY_SIMPLEX, .62, color, 1, cv2.LINE_AA)
    return result


def save_comparison(video_path, detail, checks, fps, indices, folder):
    """두 칸이 같은 원본 프레임을 쓰도록 한 번 디코딩해 비교 영상과 대표 장면을 저장한다."""
    folder = Path(folder)
    movie = folder / "comparison.mp4"
    direct = detail[detail.method == "YOLO"].sort_values("frame_index")
    linked = detail[detail.method == "Past link"].sort_values("frame_index")
    lookup = {row.frame_index: row for row in checks.itertuples()}
    left_trail, right_trail = deque(maxlen=12), deque(maxlen=12)
    writer = None
    try:
        pairs = zip(frames_rgb(video_path), direct.itertuples(), linked.itertuples(), strict=True)
        for index, (rgb, left, right) in enumerate(tqdm(pairs, total=len(direct), desc="비교 영상", unit="장")):
            if left.frame_index != index or right.frame_index != index:
                raise ValueError(f"{index}번 영상과 판정 행을 확인하세요.")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pair = cv2.hconcat([panel(bgr, left, left_trail, "Link OFF"),
                                panel(bgr, right, right_trail, "Link ON", lookup.get(index))])
            if writer is None:
                height, width = pair.shape[:2]
                writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError("H.264 비교 영상을 열 수 없습니다.")
            writer.write(pair)
            if index in indices and not cv2.imwrite(str(folder / f"frame_{index:03}.png"), pair):
                raise OSError(f"{index}번 비교 이미지를 저장하지 못했습니다.")
    finally:
        if writer is not None:
            writer.release()
    return movie
