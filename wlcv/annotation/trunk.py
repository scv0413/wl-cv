"""체간 판독 전, 두 사건을 원본에서 확인하고 가까운 쪽 관절을 찍는다."""

import csv

import ipywidgets as widgets
from filelock import FileLock

from wlcv.annotation.frame_view import FrameView
from wlcv.annotation.pose_points import pose_point_editor
from wlcv.annotation.storage import _replacement


def pair_path(folder, video):
    return folder / "frames" / f"{video}.csv"


def read_pair(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as source:
        row, = csv.DictReader(source)
    return row


def save_pair(path, row, previous):
    """현재 영상의 체간 판독용 시점만 저장한다. 원래 이벤트 표는 그대로 보존한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".write.lock"), timeout=5):
        if read_pair(path) != previous:
            raise ValueError("다른 창에서 시점을 수정했습니다. 영상을 다시 선택하세요.")
        with _replacement(path) as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)


def paired_panels(left, right):
    """두 프레임을 같은 폭으로 나란히 배치한다."""
    for panel in (left, right):
        panel.layout = widgets.Layout(width="49%", min_width="0", overflow="auto")
    return widgets.HBox([left, right], layout=widgets.Layout(width="100%", justify_content="space-between"))


def trunk_pair_editor(video_path, info, folder):
    """시점을 명시적으로 확인한 후에만 해당 프레임의 어깨·골반 판독기를 연다."""
    path = pair_path(folder, video_path.stem)
    saved = read_pair(path)
    confirmed = saved is not None and saved["판독"] == "확인"
    start = int(saved["이륙"]) if confirmed else int(info["탐색 시작"])
    end = int(saved["무릎 통과"]) if confirmed else int(info["통과 라벨"])
    first, last = FrameView(video_path, start), FrameView(video_path, end)
    confirm = widgets.Button(description="두 시점 확인", button_style="success")
    skip = widgets.Button(description="시점 판독 불가")
    notice = widgets.Label()
    points = widgets.VBox()

    def show_frames(view, title):
        def refresh(*_):
            points.children = ()
            notice.value = "원본을 확인한 뒤 두 시점을 저장하세요."
            view.read()
            view.show()

        refresh()
        view.slider.observe(refresh, names="value")
        return widgets.VBox([widgets.HTML(f"<b>{title}</b>"), view.slider, view.image])

    previews = paired_panels(show_frames(first, "이륙 후보"), show_frames(last, "무릎 통과 후보"))
    chooser = widgets.Accordion(children=[previews], selected_index=0)
    chooser.set_title(0, "프레임을 앞뒤로 넘겨 두 시점 확인")

    def open_editors():
        panels = []
        for label, index in [("이륙", first.slider.value), ("무릎 통과", last.slider.value)]:
            editor = pose_point_editor(video_path, index, info["person_id"], info["near_side"],
                                       split="validation", annotations=folder,
                                       joints=("어깨", "골반"), trunk_review=True)
            panels.append(widgets.VBox([widgets.HTML(f"<b>{label} · {index}번</b>"), editor]))
        points.children = (paired_panels(*panels),)
        chooser.selected_index = None
        notice.value = "두 시점을 저장했습니다. 아래에서 어깨와 골반을 선택하고 클릭하세요."

    def persist(button):
        nonlocal saved
        use = button is confirm
        if use and first.slider.value >= last.slider.value:
            notice.value = "이륙은 무릎 통과보다 앞선 프레임이어야 합니다."
            return
        row = {"판독": "확인" if use else "판독 불가",
               "이륙": str(first.slider.value) if use else "",
               "무릎 통과": str(last.slider.value) if use else ""}
        try:
            save_pair(path, row, saved)
        except (ValueError, OSError) as error:
            notice.value = str(error)
            return
        saved = row
        if use:
            open_editors()
        else:
            points.children = ()
            notice.value = "시점 판독 불가로 저장했습니다."

    confirm.on_click(persist)
    skip.on_click(persist)
    if confirmed:
        open_editors()
    elif saved is not None:
        notice.value = "이 영상은 시점 판독 불가로 기록되어 있습니다."
    return widgets.VBox([chooser, widgets.HBox([confirm, skip]), notice, points])
