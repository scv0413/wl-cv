"""주피터 셀에서 원판 테두리를 마우스로 드래그해 표시한다."""

import ipywidgets as widgets
from ipyevents import Event

from wlcv.config import ANNOTATIONS
from .frame_view import FrameView
from .pixel_edges import BoxEdgeSliders, normalized_box
from .storage import load_boxes, save_boxes


def _labelled_frames(annotations, video_id):
    frames = set()
    for folder, prefix in (("plate_boxes", f"{video_id}_f"),
                           ("personal_plate_boxes", f"{video_id}_frame_")):
        for path in (annotations / folder).glob(prefix + "*.txt"):
            number = path.stem[len(prefix):]
            if number.isdigit():
                frames.add(int(number))
    return sorted(frames)


def plate_box_editor(video_path, annotations=ANNOTATIONS):
    """원본 또는 개인 박스를 읽는다. 저장 버튼을 눌렀을 때만 기록한다."""
    video_id = video_path.stem
    labelled = _labelled_frames(annotations, video_id)
    view = FrameView(video_path, labelled[0] if labelled else 0)
    jump = widgets.Dropdown(description="라벨 위치")
    status = widgets.Label()
    undo = widgets.Button(description="마지막 박스 삭제")
    save = widgets.Button(description="박스 저장", button_style="success")
    empty = widgets.Button(description="원판 없음 저장")
    reload = widgets.Button(description="다시 읽기")
    boxes = []
    previous = None
    anchor = None

    def update_controls():
        dirty = previous is not None and boxes != previous[1]
        view.slider.disabled = dirty
        jump.disabled = dirty
        undo.disabled = not boxes
        save.disabled = previous is None or not boxes
        empty.disabled = previous is None or bool(boxes)
        status.value = f"{previous[0]} · 박스 {len(boxes)}개" if previous else ""
        if dirty:
            status.value += " · 프레임을 이동하려면 저장하거나 다시 읽으세요."

    def redraw():
        view.show(boxes)
        update_controls()

    edges = BoxEdgeSliders(boxes, view, on_edit=redraw)

    def refresh(*_):
        nonlocal previous, anchor
        anchor = None
        previous = None
        boxes.clear()
        update_controls()
        try:
            view.read(keep_original=True)
            previous = load_boxes(annotations, video_id, view.slider.value)
            boxes.extend(previous[1])
            view.show(boxes)
            update_controls()
        except (OSError, ValueError, RuntimeError, IndexError) as error:
            status.value = str(error)
        edges.select_last()

    def update_jumps():
        options = [(f"프레임 {f}", f) for f in _labelled_frames(annotations, video_id)
                   if f <= view.slider.max]
        jump.options = [("선택하세요", None), *options]
        jump.value = None

    def jump_to(change):
        if change["new"] is not None:
            view.slider.value = change["new"]
            jump.value = None

    def undo_box(_):
        boxes.pop()
        edges.select_last()
        redraw()

    def persist(button):
        try:
            path = save_boxes(annotations, video_id, view.slider.value, boxes,
                              previous=previous, reviewed_empty=button is empty)
            refresh()
            update_jumps()
            status.value += f" · 저장함 {path.name}"
        except (OSError, ValueError) as error:
            status.value = str(error)

    def mouse(event):
        nonlocal anchor
        if previous is None:
            return
        if event["type"] == "mouseleave":
            if anchor is not None:
                anchor = None
                view.show(boxes)
            return
        if event["type"] == "mousemove" and anchor is None:
            return
        if event["type"] != "mousemove" and event.get("button", 0) != 0:
            return
        point = tuple(max(0, min(1, event[key] / event[size])) for key, size in
                      (("relativeX", "boundingRectWidth"), ("relativeY", "boundingRectHeight")))
        if event["type"] == "mousedown":
            anchor = point
        elif anchor is not None and event["type"] == "mousemove":
            view.show(boxes, preview=(anchor, point))
        elif anchor is not None and event["type"] == "mouseup":
            x, y = anchor
            w, h = abs(point[0] - x), abs(point[1] - y)
            anchor = None
            if w * event["boundingRectWidth"] >= 2 and h * event["boundingRectHeight"] >= 2:
                height, width = view.original_rgb.shape[:2]
                left, right = sorted((round(x * width), round(point[0] * width)))
                top, bottom = sorted((round(y * height), round(point[1] * height)))
                if left < right and top < bottom:
                    boxes.append(normalized_box(left, top, right, bottom, width, height))
                    edges.select_last()
            redraw()

    for button in (save, empty):
        button.on_click(persist)
    undo.on_click(undo_box)
    reload.on_click(refresh)
    view.slider.observe(refresh, names="value")
    update_jumps()
    jump.observe(jump_to, names="value")
    drag = Event(source=view.image, watched_events=["mousedown", "mouseup", "mouseleave"],
                 prevent_default_action=True)
    move = Event(source=view.image, watched_events=["mousemove"], wait=90,
                 throttle_or_debounce="throttle")
    drag.on_dom_event(mouse)
    move.on_dom_event(mouse)
    refresh()
    actions = widgets.Box([undo, save, empty, reload], layout=widgets.Layout(
        display="flex", flex_flow="row wrap"))
    return widgets.VBox([widgets.Label(video_path.name), jump, view.slider,
                         view.image, actions, status, edges])
