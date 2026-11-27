"""저장된 프레임 측정값에서 사용자가 고른 캐치·피니시 그래프를 만든다."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from wlcv.data.mp4_input import VideoInputError, validate_events
from wlcv.runtime.recovery import save_recovery_curve


def current_review(folder):
    selection = folder / "recovery_review.json"
    if selection.is_file():
        return json.loads(selection.read_text(encoding="utf-8"))
    return None


def chart_path(folder, review):
    if review is None:
        return folder / "catch_to_finish.png"
    return folder / "reviews" / f"{review['catch_frame']}-{review['finish_frame']}.png"


def create_review(folder, summary, catch, finish):
    events = {**summary["manual_events"], **summary.get("pull_review", {}).get("events", {}),
              "catch": catch, "finish": finish}
    validate_events(events, summary["frame_count"])
    rows = pd.read_csv(folder / "frames.csv", usecols=["frame_index", "forward_px", "trunk_angle_deg"])
    with TemporaryDirectory(dir=folder) as temp:
        staged = Path(temp)
        review = save_recovery_curve(rows, events, staged)
        if review["reason"]:
            raise VideoInputError(review["reason"])
        target = chart_path(folder, review)
        target.parent.mkdir(exist_ok=True)
        (staged / "catch_to_finish.png").replace(target)
        # 완성된 PNG가 준비된 후에만 다운로드 링크를 새 판독으로 전환한다.
        selection = staged / "recovery_review.json"
        selection.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
        selection.replace(folder / selection.name)
    return review
