"""원본 프레임에서 수행자 박스를 편집한다. 필요하면 기존 관절점을 표시한다."""

import cv2
import ipywidgets as widgets
from ipyevents import Event

from .frame_view import FrameView
from .pose_points import load_points


def performer_box_editor(video_path, frame, *, source, person, split, side, box):
    view = FrameView(video_path, frame)
    width, height = view.read()
    preview_h, preview_w = view.rgb.shape[:2]
    # 관절은 화면에만 그린다. 저장 버튼은 전달받은 수행자 박스만 쓴다.
    points = load_points(source)
    markers = {"무릎": cv2.MARKER_CROSS, "발목": cv2.MARKER_DIAMOND}
    for joint, marker in markers.items():
        row = points.get(joint)
        expected = (video_path.stem, str(frame), person, split, side, "located")
        if row is None or tuple(row[k] for k in ("video_id", "frame_index", "person_id", "split", "side", "status")) != expected:
            raise ValueError(f"{source.name}에서 {joint}의 직접 판독과 장면 정보를 확인하세요.")
        x, y = float(row["x_px"]), float(row["y_px"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{source.name}의 {joint} 좌표가 원본 밖에 있습니다.")
        pixel = (min(preview_w - 1, int(x * preview_w / width)),
                 min(preview_h - 1, int(y * preview_h / height)))
        cv2.drawMarker(view.rgb, pixel, (0, 0, 0), marker, 14, 4)
        cv2.drawMarker(view.rgb, pixel, (255, 218, 68), marker, 14, 2)

    return _box_widget(view, box, width, height, "머리부터 발끝까지 드래그하세요.")


def visible_box_editor(video_path, frame, box):
    """처음에는 원본만 보여 주고 재개할 때는 이번 판독의 저장본만 불러온다."""
    view = FrameView(video_path, frame)
    width, height = view.read()
    return _box_widget(view, box, width, height, "보이는 수행자의 양 끝을 대각선으로 드래그하세요.")


def _box_widget(view, box, width, height, instruction):
    save = widgets.Button(description="박스 저장", button_style="success", disabled=True)
    reload = widgets.Button(description="저장본 다시 열기")
    status = widgets.Label()
    anchor = None
    draft = None

    def show_saved():
        if box.saved is None:
            view.show()
            status.value = f"미저장 · {instruction}"
        else:
            x1, y1, x2, y2 = box.saved
            view.show([((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                        (x2 - x1) / width, (y2 - y1) / height)])
            status.value = f"저장됨 · ({x1}, {y1}) — ({x2}, {y2})"

    def preview(a, b):
        view.show(preview=((a[0] / width, a[1] / height), (b[0] / width, b[1] / height)))

    def restore(*_):
        nonlocal anchor, draft
        box.reload()
        anchor = draft = None
        save.disabled = True
        show_saved()

    def show_draft():
        if draft is None:
            show_saved()
        else:
            preview(draft[:2], draft[2:])
            status.value = "그린 박스를 확인한 뒤 ‘박스 저장’을 누르세요."
        save.disabled = draft is None

    def mouse(event):
        nonlocal anchor, draft
        kind = event["type"]
        if kind == "mouseleave":
            if anchor is not None:
                anchor = None
                show_draft()
            return
        if kind != "mousedown" and anchor is None:
            return
        if kind != "mousemove" and event.get("button", 0) != 0:
            return
        # 확대·축소된 화면에서 드래그해도 CSV에는 원본 픽셀 좌표를 저장한다.
        x = int(event["relativeX"] / event["boundingRectWidth"] * width)
        y = int(event["relativeY"] / event["boundingRectHeight"] * height)
        point = (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
        if kind == "mousedown":
            anchor = point
            save.disabled = True
        elif kind == "mousemove":
            preview(anchor, point)
        elif kind == "mouseup":
            x1, x2 = sorted((anchor[0], point[0]))
            y1, y2 = sorted((anchor[1], point[1]))
            anchor = None
            if x1 < x2 and y1 < y2:
                draft = (x1, y1, x2, y2)
            show_draft()

    def persist(_):
        nonlocal draft
        if draft is None:
            return
        try:
            box.save(draft)
        except (OSError, ValueError) as error:
            status.value = str(error)
            return
        draft = None
        show_draft()

    drag = Event(source=view.image, watched_events=["mousedown", "mouseup", "mouseleave"],
                 prevent_default_action=True)
    move = Event(source=view.image, watched_events=["mousemove"], wait=75, throttle_or_debounce="throttle")
    drag.on_dom_event(mouse)
    move.on_dom_event(mouse)
    save.on_click(persist)
    reload.on_click(restore)
    show_saved()
    return widgets.VBox([widgets.HBox([save, reload]), status, view.image])
