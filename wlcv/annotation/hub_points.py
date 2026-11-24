"""b02의 원본 프레임에서 허브 중심을 직접 판독한다. 예측은 읽지 않는다."""

import csv

import cv2
import ipywidgets as widgets
import pandas as pd
from filelock import FileLock
from ipyevents import Event

from wlcv.annotation.storage import _replacement
from wlcv.config import ANNOTATIONS, VIDEOS
from wlcv.data.frames import read_rgb

VIDEO = "b02_sn_h_02"
FRAMES = (34, 35, 36)
POINTS = ANNOTATIONS / "hub_review" / f"{VIDEO}.csv"
FIELDS = ["video_id", "frame_index", "visibility", "manual_x", "manual_y"]


def read_hubs(path=POINTS, *, frames=FRAMES):
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    rows = pd.read_csv(path).loc[:, FIELDS]
    if (not rows.video_id.eq(VIDEO).all() or rows.frame_index.duplicated().any()
            or not rows.frame_index.isin(frames).all()):
        raise ValueError("허브 표의 영상과 프레임 번호를 확인하세요.")
    xy = rows[["manual_x", "manual_y"]]
    visible = rows.visibility.eq("visible")
    if (not rows.visibility.isin(["visible", "unreadable"]).all()
            or xy[visible].isna().any().any() or xy[~visible].notna().any().any()):
        raise ValueError("직접 확인한 행에만 x, y를 기록하세요.")
    return rows


def save_hub(path, index, xy, previous, *, frames=FRAMES):
    if index not in frames:
        raise ValueError("이번 판독 대상에 없는 프레임입니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".write.lock"), timeout=5):
        rows = read_hubs(path, frames=frames)
        old = rows[rows.frame_index.eq(index)]
        old_values = old.fillna("").to_dict("records")
        if old_values != previous.fillna("").to_dict("records"):
            raise ValueError("다른 창에서 이 점을 수정했습니다. 판독 셀을 다시 여세요.")
        values = [VIDEO, index, "unreadable", None, None] if xy is None else [VIDEO, index, "visible", *xy]
        with _replacement(path) as out:
            writer = csv.writer(out)
            writer.writerow(FIELDS)
            for row in rows[~rows.frame_index.eq(index)].itertuples(index=False, name=None):
                writer.writerow(["" if pd.isna(v) else v for v in row])
            writer.writerow(values)


def hub_editor(index, *, path=POINTS, frames=FRAMES):
    if index not in frames:
        raise ValueError("이번 판독 대상에 없는 프레임입니다.")
    rgb = read_rgb(VIDEOS / f"{VIDEO}.mp4", index)
    h, w = rgb.shape[:2]
    photo = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
    zoom = widgets.Checkbox(value=False, description="원본 크기로 보기")
    unreadable = widgets.Button(description="판독 불가로 저장")
    status = widgets.Label()
    rows = read_hubs(path, frames=frames)
    previous = rows[rows.frame_index.eq(index)]

    def draw(*_):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        status.value = f"{index}번 · 미판독"
        if len(previous):
            row = previous.iloc[0]
            status.value = f"{index}번 · 판독 불가 저장됨"
            if row.visibility == "visible":
                xy = (int(row.manual_x), int(row.manual_y))
                cv2.drawMarker(image, xy, (0, 215, 255), cv2.MARKER_CROSS, 15, 2)
                status.value = f"{index}번 · 저장한 내 좌표 {xy}"
        photo.layout.width = f"{w if zoom.value else min(w,450)}px"
        photo.value = cv2.imencode(".png", image)[1].tobytes()

    def save(xy):
        nonlocal previous
        try:
            save_hub(path, index, xy, previous, frames=frames)
        except (ValueError, OSError, TimeoutError) as exc:
            status.value = str(exc)
            return
        rows = read_hubs(path, frames=frames)
        previous = rows[rows.frame_index.eq(index)]
        draw()

    def click(event):
        rx = event["relativeX"] / event["boundingRectWidth"]
        ry = event["relativeY"] / event["boundingRectHeight"]
        if 0 <= rx < 1 and 0 <= ry < 1:
            save((int(rx*w), int(ry*h)))

    clicks = Event(source=photo, watched_events=["click"])
    clicks.on_dom_event(click)
    unreadable.on_click(lambda _: save(None))
    zoom.observe(draw, names="value")
    draw()
    return widgets.VBox([status, widgets.HBox([zoom, unreadable]), photo])
