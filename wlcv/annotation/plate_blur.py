"""검증 원본의 무릎 통과부터 접촉까지 흐림·가림·허브를 독립적으로 기록한다."""

import csv

import cv2
import ipywidgets as widgets
import pandas as pd
from filelock import FileLock
from ipyevents import Event

from wlcv.annotation.storage import _replacement, load_events
from wlcv.config import ANNOTATIONS, SPLIT, VIDEOS
from wlcv.data.dataset import read_splits
from wlcv.data.frames import read_rgb
from wlcv.data.plate_labels import label_frame

REVIEW_FILE = ANNOTATIONS / "plate_blur_review" / "validation.csv"
CLARITY = ["선명", "움직임으로 흐림", "판단 어려움"]
OCCLUSION = ["가림 없음", "가림 있음", "판단 어려움"]
FIELDS = ["video_id", "frame_index", "clarity", "occlusion", "hub_source", "manual_x", "manual_y"]


def review_frames():
    """기존 원판 라벨이 있는 검증 영상에서 1차 이벤트의 양 끝을 모두 포함한다."""
    split = read_splits(SPLIT)
    labels = {label_frame(path)[0] for path in (ANNOTATIONS / "plate_boxes").glob("*.txt")}
    videos = sorted(video for video in labels if split[video] == "val")
    if not videos:
        raise ValueError("기존 원판 라벨과 검증 분할을 확인하세요.")
    frames = []
    for video in videos:
        events = load_events(ANNOTATIONS, video, pass_number=1)
        start, end = events.get("transition"), events.get("drive")
        if start is None or end is None or not 0 <= start <= end:
            raise ValueError(f"{video}의 1차 무릎 통과·접촉 라벨을 확인하세요.")
        frames.extend((video, index) for index in range(start, end + 1))
    return frames


def read_reviews(path=REVIEW_FILE):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"판독 파일의 열을 확인하세요. {path}")
        reviews = {}
        for row in reader:
            key = (row["video_id"], int(row["frame_index"]))
            if key in reviews:
                raise ValueError(f"판독 파일에 중복된 프레임이 있습니다. {key}")
            reviews[key] = row
        return reviews


def save_review(path, key, row, previous):
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".write.lock"), timeout=5):
        reviews = read_reviews(path)
        if reviews.get(key) != previous:
            raise ValueError("다른 창에서 이 프레임을 저장했습니다. 판독 셀을 다시 여세요.")
        reviews[key] = row
        with _replacement(path) as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(reviews[key] for key in sorted(reviews))


def previous_blind_hubs():
    """14번 블라인드 판독에서 직접 찍은 좌표만 읽는다."""
    saved = {}
    for path in (ANNOTATIONS / "hub_review").glob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["visibility"] == "visible":
                    key = (row["video_id"], int(row["frame_index"]))
                    saved[key] = (int(float(row["manual_x"])), int(float(row["manual_y"])))
    return saved


def frame_form(key, original_hub, previous, on_save):
    """좌표는 원본 픽셀로 보관한다. 선택한 프레임마다 새 입력 폼을 만든다."""
    video, index = key
    rgb = read_rgb(VIDEOS / f"{video}.mp4", index)
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    photo = widgets.Image(format="png", layout=widgets.Layout(
        width=f"{min(width, 500)}px", max_width="100%", height="auto", align_self="flex-start"))
    zoom = widgets.Checkbox(description="원본 크기로 보기")
    clarity = widgets.Dropdown(options=CLARITY, value=None, description="테두리 선명도")
    occlusion = widgets.Dropdown(options=OCCLUSION, value=None, description="원판 가림")
    options = [("직접 찍기", "clicked"), ("판독 불가", "unreadable")]
    if original_hub is not None:
        options.append(("기존 블라인드 직접점 사용", "previous_blind"))
    hub = widgets.Dropdown(options=options, value="previous_blind" if original_hub else "clicked",
                           description="허브 중심")
    point = None
    if previous:
        clarity.value, occlusion.value = previous["clarity"], previous["occlusion"]
        hub.value = previous["hub_source"]
        if previous["hub_source"] == "clicked":
            point = (int(previous["manual_x"]), int(previous["manual_y"]))

    notice = widgets.Label()
    save = widgets.Button(description="저장하고 다음", button_style="success")
    def draw(*_):
        canvas = bgr.copy()
        if hub.value == "clicked" and point is not None:
            cv2.drawMarker(canvas, point, (0, 210, 255), cv2.MARKER_CROSS, 17, 2)
        photo.layout.width = f"{width if zoom.value else min(width, 500)}px"
        photo.value = cv2.imencode(".png", canvas)[1].tobytes()

    def click(event):
        nonlocal point
        x = int(width * event["relativeX"] / event["boundingRectWidth"])
        y = int(height * event["relativeY"] / event["boundingRectHeight"])
        if 0 <= x < width and 0 <= y < height:
            point = (x, y)
            hub.value = "clicked"
            notice.value = f"선택한 허브 ({x}, {y}) · 저장하려면 버튼을 누르세요."
            draw()

    def submit(_):
        if clarity.value is None or occlusion.value is None:
            notice.value = "테두리 선명도와 가림 여부를 각각 선택하세요."
            return
        xy = point if hub.value == "clicked" else original_hub
        if hub.value == "unreadable":
            xy = ("", "")
        elif xy is None:
            notice.value = "허브 중심을 클릭하거나 ‘판독 불가’를 선택하세요."
            return
        row = dict(zip(FIELDS, [video, index, clarity.value, occlusion.value, hub.value, *xy]))
        try:
            on_save(key, row, previous)
        except (OSError, ValueError, TimeoutError) as exc:
            notice.value = str(exc)

    clicks = Event(source=photo, watched_events=["click"])
    clicks.on_dom_event(click)
    zoom.observe(draw, names="value")
    hub.observe(draw, names="value")
    save.on_click(submit)
    draw()
    return widgets.VBox([zoom, photo, clarity, occlusion, hub, save, notice])


def review_editor(frames, path=REVIEW_FILE):
    reviews = read_reviews(path)
    hubs = previous_blind_hubs()
    first = next((key for key in frames if key not in reviews), frames[0])
    picker = widgets.Dropdown(options=[(f"{video} · {index}번", (video, index))
                                       for video, index in frames], value=first, description="프레임")
    status = widgets.Label()
    panel = widgets.VBox()

    def open_frame(*_):
        saved = read_reviews(path)
        key = picker.value
        count = sum(frame in saved for frame in frames)
        state = "저장됨" if key in saved else "미판독"
        status.value = f"{count}/{len(frames)}장 저장 · 현재 프레임 {state}"
        panel.children = ()
        try:
            panel.children = (frame_form(key, hubs.get(key), saved.get(key), submit),)
        except (IndexError, OSError, ValueError) as exc:
            status.value += f" · {exc}"

    def submit(key, row, previous):
        save_review(path, key, row, previous)
        saved = read_reviews(path)
        remaining = [frame for frame in frames if frame not in saved]
        if remaining:
            picker.value = next((frame for frame in remaining if frame > key), remaining[0])
        else:
            open_frame()

    picker.observe(open_frame, names="value")
    open_frame()
    return widgets.VBox([picker, status, panel])


def progress_table(frames, path=REVIEW_FILE):
    saved = read_reviews(path)
    rows = []
    for key in frames:
        row = saved.get(key, {})
        rows.append({"영상": key[0], "대상": 1, "저장": int(bool(row)),
                     "직접 허브": int(row.get("hub_source") in ("clicked", "previous_blind")),
                     "허브 판독 불가": int(row.get("hub_source") == "unreadable")})
    return pd.DataFrame(rows).groupby("영상").sum()
