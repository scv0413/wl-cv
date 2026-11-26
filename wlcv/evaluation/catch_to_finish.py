"""c04에서 캐치 후 일어서는 구간을 기존 단일 모델의 출력으로 다시 본다."""

import json

import cv2
import pandas as pd

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS
from wlcv.evaluation.catch_offset import FACING
from wlcv.runtime.recovery import landmarks, plot_interval as draw_interval


def plot_interval(rows, *, title="c04", note="Model estimates per frame. At catch, the manual shoulder and hip were both estimated."):
    return draw_interval(rows, title=title, note=note)


VIDEO = "c04_sn_f_01"


def read_interval(run):
    summary = json.loads((run / "summary.json").read_text())
    if (summary["video_id"] != VIDEO or summary["near_side"] != "left"
            or summary["plate_mode"] != "single" or summary["plate_imgsz"] != 640
            or summary["pose_imgsz"] != 640):
        raise ValueError("c04 왼쪽 관절과 단일 원판 모델을 입력 640으로 분석한 폴더를 선택하세요.")
    events = load_events(ANNOTATIONS, VIDEO, pass_number=1)
    start, end = events["catch"], events["finish"]
    rows = pd.read_csv(run / "frames.csv").set_index("frame_index", verify_integrity=True)
    rows = rows.loc[start:end].copy()
    if (start >= end or list(rows.index) != list(range(start, end + 1))
            or not rows.video_id.eq(VIDEO).all() or not rows.pose_side.eq("left").all()
            or not rows.source_time_ms.diff().iloc[1:].gt(0).all()):
        raise ValueError("캐치부터 피니시까지 원본 프레임과 시각 순서를 확인하세요.")
    rows["elapsed_s"] = (rows.source_time_ms - rows.source_time_ms.iloc[0]) / 1000
    # c04는 화면 왼쪽을 본다. 양수는 선수 앞쪽, 음수는 뒤쪽이다.
    rows["forward_px"] = FACING[VIDEO] * (rows.plate_cx_px - rows.shoulder_x_px)
    for frame in rows.index:
        points = load_points(pose_point_path(ANNOTATIONS, VIDEO, frame))
        for joint, column in [("어깨", "shoulder_review"), ("골반", "hip_review")]:
            point = points.get(joint)
            rows.loc[frame, column] = point["status"] if point and point["side"] == "left" else "unreviewed"
    return rows


def write_clip(run, rows, destination):
    """기존 오버레이 위에 거리와 수직선을 추가하고 원본 1장당 1장씩 0.25배속으로 저장한다."""
    cap = cv2.VideoCapture(str(run / "overlay.mp4"))
    writer = None
    try:
        if not cap.isOpened():
            raise OSError("같은 실행의 overlay.mp4를 열 수 없습니다.")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, int(rows.index[0])):
            raise ValueError("오버레이 시각 또는 시작 프레임을 확인하세요.")
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"avc1"), fps / 4, (w, h + 172))
        if not writer.isOpened():
            raise OSError("H.264 동영상을 만들 수 없습니다.")
        for row in rows.itertuples():
            ok, image = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != row.Index + 1:
                raise ValueError(f"{row.Index}번 원본과 오버레이 순서가 맞지 않습니다.")
            if pd.notna(row.forward_px):
                x, y = round(row.shoulder_x_px), round(row.plate_cy_px)
                shoulder_y = round(row.shoulder_y_px)
                for top in range(min(y, shoulder_y), max(y, shoulder_y) + 81, 22):
                    cv2.line(image, (x, top), (x, top + 10), (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(image, (x, y), (round(row.plate_cx_px), y), (35, 195, 255), 6, cv2.LINE_AA)
            panel = cv2.copyMakeBorder(image, 0, 172, 0, 0, cv2.BORDER_CONSTANT, value=(31, 30, 23))
            phase = "CATCH" if row.Index == rows.index[0] else "FINISH" if row.Index == rows.index[-1] else "RISING"
            offset = f"{row.forward_px:+.1f} px  (+ forward)" if pd.notna(row.forward_px) else "unavailable"
            angle = f"{row.trunk_angle_deg:.1f} deg from vertical" if pd.notna(row.trunk_angle_deg) else "unavailable"
            lines = [f"{phase}    frame {row.Index}    +{row.elapsed_s:.2f} s    |    0.25x playback",
                     f"Plate vs shoulder    {offset}", f"Trunk    {angle}"]
            for line, label in enumerate(lines):
                cv2.putText(panel, label, (28, h + 44 + 52 * line),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.05, (245, 245, 245), 2, cv2.LINE_AA)
            writer.write(panel)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
