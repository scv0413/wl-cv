"""확인한 세 시점의 코칭 결과를 같은 분석 ID에 저장한다."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import pandas as pd
from fastapi import HTTPException

from wlcv.api.recovery_review import current_review
from wlcv.runtime.pull_review import revised_review, reviewed_metrics


def reviewed_frame(folder, frame):
    capture = cv2.VideoCapture(str(folder / "overlay.mp4"))
    try:
        if frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise HTTPException(500, "오버레이의 해당 프레임을 찾을 수 없습니다.")
        ok, image = capture.read()
        if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
            raise HTTPException(500, "요청한 오버레이 프레임을 읽을 수 없습니다.")
        ok, jpeg = cv2.imencode(".jpg", image)
        if not ok:
            raise HTTPException(500, "해당 프레임을 표시할 수 없습니다.")
        return jpeg.tobytes()
    finally:
        capture.release()


def add_saved_clipping(data, folder):
    """같은 과거 궤적을 조회하거나 재검수할 때 화면 경계도 같은 기준으로 복원한다."""
    if "plate_clipped" not in data and {"plate_x1", "plate_y1", "plate_x2", "plate_y2"}.issubset(data):
        data["plate_clipped"] = (data.plate_x1 <= 0) | (data.plate_y1 <= 0)
        if (folder / "overlay.mp4").is_file():
            capture = cv2.VideoCapture(str(folder / "overlay.mp4"))
            try:
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if width > 0 and height > 0:
                    data["plate_clipped"] |= (data.plate_x2 >= width - 1) | (data.plate_y2 >= height - 1)
            finally:
                capture.release()


def save_pull_review(folder, summary, updates):
    events = dict(summary["manual_events"])
    recovery = current_review(folder)
    if recovery:
        events.update(catch=recovery["catch_frame"], finish=recovery["finish_frame"])
    review = revised_review(summary["pull_review"], updates, events, summary["frame_count"])
    events.update(review["events"])
    rows = pd.read_csv(folder / "frames.csv")
    add_saved_clipping(rows, folder)
    points = json.loads((folder / "pull_endpoints.json").read_text())
    checks = {point["frame_index"]: point for point in points}
    direction = {"left": -1, "right": 1}[summary["facing"]]
    metrics = reviewed_metrics(rows, events, checks, summary.get("pull_cosine_weights"), direction)
    metrics["coaching"] = {**summary["coaching"], **metrics["coaching"]}
    summary.update(metrics, pull_review=review)
    with TemporaryDirectory(dir=folder) as temp:
        target = Path(temp) / "summary.json"
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        target.replace(folder / target.name)
