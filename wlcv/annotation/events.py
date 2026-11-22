"""프레임을 보며 동작 이벤트 한 건씩 판독하고 저장한다."""

import ipywidgets as widgets

from wlcv.config import ANNOTATIONS
from .frame_view import FrameView
from .storage import events_for, load_events, save_event


def phase_event_editor(video_path, pass_number=1, annotations=ANNOTATIONS):
    """선택한 영상과 판독 차수에 속한 이벤트만 화면에 표시한다."""
    if pass_number not in (1, 2):
        raise ValueError("판독 차수는 1 또는 2입니다.")
    video_id = video_path.stem
    names = events_for(video_id)
    records = load_events(annotations, video_id, pass_number)
    locked = pass_number == 1 and bool(load_events(annotations, video_id, 2))
    view = FrameView(video_path, records.get(names[0], 0))
    event = widgets.Dropdown(description="동작", options=names)
    save = widgets.Button(description="현재 프레임 기록", button_style="success")
    reload = widgets.Button(description="기록 다시 읽기")
    summary = widgets.Label()
    status = widgets.Label()
    lock_notice = " · 2차 판독을 시작해 1차 기록은 읽기 전용입니다."

    def frame_changed(*_):
        save.disabled = True
        status.value = ""
        try:
            view.read()
            view.show()
            save.disabled = locked
        except (OSError, ValueError, RuntimeError, IndexError) as error:
            status.value = str(error)

    def select_event(*_):
        stored = records.get(event.value)
        if stored is not None and 0 <= stored <= view.slider.max:
            view.slider.value = stored
        summary.value = "   |   ".join(f"{name} {records.get(name, '미기록')}" for name in names)

    def refresh(*_):
        nonlocal records, locked
        try:
            records = load_events(annotations, video_id, pass_number)
            locked = pass_number == 1 and bool(load_events(annotations, video_id, 2))
            save.disabled = locked or view.rgb is None
            title.value = heading + (lock_notice if locked else "")
            select_event()
            if view.rgb is not None:
                status.value = "저장된 기록을 다시 읽었습니다."
        except (OSError, ValueError) as error:
            status.value = str(error)

    def persist(_):
        try:
            path = save_event(annotations, video_id, pass_number, event.value,
                              view.slider.value, previous=records.get(event.value))
            refresh()
            status.value = f"{event.value} · 프레임 {view.slider.value} 저장함 · {path.name}"
        except (OSError, ValueError) as error:
            status.value = str(error)

    save.on_click(persist)
    reload.on_click(refresh)
    event.observe(select_event, names="value")
    view.slider.observe(frame_changed, names="value")
    heading = f"{video_path.name} · {pass_number}차 판독"
    title = widgets.Label(heading + (lock_notice if locked else ""))
    frame_changed()
    select_event()
    return widgets.VBox([title,
                         event, summary, view.slider, view.image,
                         widgets.HBox([save, reload]), status])
