"""가까운 쪽 관절을 한 번 클릭해 원본 픽셀에 기록한다."""

import csv

import cv2
import ipywidgets as widgets
from filelock import FileLock
from ipyevents import Event

from wlcv.config import ANNOTATIONS
from wlcv.data.frames import read_rgb
from .storage import _replacement

JOINTS = ("어깨", "골반", "무릎", "발목")
ANKLE_FOLD = "앞쪽 접힘점"
HIP_FOLD = "고관절 앞쪽 접힘점"
FIELDS = ("video_id", "person_id", "split", "frame_index", "side", "joint",
          "status", "reason", "x_px", "y_px")


def pose_point_path(annotations, video_id, frame_index, *, pass_number=1, fold=False):
    folder = annotations / ("ankle_fold_points" if fold else "personal_pose_points")
    if pass_number == 2:
        folder = folder / "pass_2"
    elif pass_number != 1:
        raise ValueError("판독 차수는 1 또는 2입니다.")
    return folder / f"{video_id}_frame_{frame_index:04d}.csv"


def load_points(path):
    """검수한 관절만 들어 있다. 판독 불가 행의 좌표 칸은 비어 있다."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as source:
        return {row["joint"]: row for row in csv.DictReader(source)}


def _save_point(path, row, previous):
    """같은 관절에 대한 동시 수정을 확인하고 다른 관절의 기록은 보존한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".write.lock"), timeout=5):
        points = load_points(path)
        if points.get(row["joint"]) != previous:
            raise ValueError("이 관절이 다른 창에서 바뀌었습니다. 다시 읽은 뒤 확인하세요.")
        points[row["joint"]] = row
        with _replacement(path) as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(points[joint] for joint in (*JOINTS, ANKLE_FOLD, HIP_FOLD) if joint in points)
        return points


def pose_point_editor(video_path, frame_index, person_id, side, *, split, annotations=ANNOTATIONS,
                      pass_number=1, joints=JOINTS, trunk_review=False, frame_label=None, performer_box=None):
    """한 원본 프레임을 열고 사람이 선택한 관절 한 개씩 자동 저장한다."""
    if side not in ("left", "right"):
        raise ValueError(f"{video_path.stem}: 메타데이터에서 카메라에 가까운 쪽을 확인하세요.")
    fold = tuple(joints) == (ANKLE_FOLD,)
    hip_fold = tuple(joints) == (HIP_FOLD,)
    if not (fold or hip_fold) and (not joints or not set(joints).issubset(JOINTS)):
        raise ValueError("판독할 관절을 확인하세요.")
    rgb = read_rgb(video_path, frame_index)
    height, width = rgb.shape[:2]
    path = pose_point_path(annotations, video_path.stem, frame_index, pass_number=pass_number, fold=fold)
    frame_info = dict(video_id=video_path.stem, frame_index=str(frame_index),
                      person_id=person_id, split=split, side=side)
    points = load_points(path)
    side_label = "왼쪽" if side == "left" else "오른쪽"
    frame_label = frame_label or f"{video_path.stem} · {frame_index}번"
    title = widgets.Label(f"{pass_number}차 판독 · {frame_label} · 선수의 {side_label}")
    box_option = "수행자 박스"
    options = (box_option, *joints) if performer_box is not None else joints
    joint = widgets.ToggleButtons(options=options, description="표시 항목" if performer_box else
                                  ("기준점" if fold or hip_fold else "관절"))
    zoom = widgets.Checkbox(value=False, description="원본 크기로 확대")
    photo = widgets.Image(format="jpeg", layout=widgets.Layout(
        max_width="none", height="auto", display="block"))
    viewport = widgets.Box([photo], layout=widgets.Layout(
        display="block", width="100%", max_height="650px", overflow="auto"))
    hidden = widgets.Button(description="가려서 판독 불가")
    unsure = widgets.Button(description="위치 불확실")
    quality = widgets.Button(layout=widgets.Layout(width="auto"))
    reload_button = widgets.Button(description="다시 읽기")
    summary = widgets.Label()
    message = widgets.Label()

    # 체간 실험에서는 점을 찍기 전에 가시성을 고른다. 소켓은 추정을 기본으로 한다.
    visibility = widgets.Dropdown(description="찍을 점", layout=widgets.Layout(width="auto"))

    def select_visibility(*_):
        options = [("직접 확인", "located"), ("옷에 가려 추정", "clothing"),
                   ("그 밖의 가림으로 추정", "occluded")]
        editing_box = joint.value == box_option
        visibility.disabled = hidden.disabled = unsure.disabled = editing_box
        if performer_box is not None:
            performer_box.first_corner = None
        default = "located"
        if joint.value == "골반":
            options.append(("소켓 위치 추정", "hip_socket"))
            default = "hip_socket"
        visibility.options = options
        previous = points.get(joint.value, {})
        reason = previous.get("reason")
        if previous.get("status") == "located":
            default = "located"
        elif previous.get("status") == "estimated" and reason in dict(options).values():
            default = reason
        visibility.value = default

    def quality_button(*_):
        row = points.get(joint.value, {})
        quality.disabled = joint.value == "골반" or row.get("status") not in ("located", "estimated")
        if joint.value == "골반":
            quality.description = "소켓 기준 추정점"
        else:
            quality.description = "직접 확인점으로 변경" if row.get("status") == "estimated" else "가림 추정점으로 변경"

    def draw():
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        labels = []
        if performer_box is not None:
            performer_box.draw(canvas)
            labels.append("박스 저장됨" if performer_box.saved else "박스 미검수")
        for number, name in enumerate(joints, start=1):
            row = points.get(name)
            if row is None:
                label = "미검수"
            elif row["status"] == "unreadable":
                label = "판독 불가(가림)" if row["reason"] == "occluded" else "판독 불가(불확실)"
            else:
                # 소켓 1·2차 평균은 반 픽셀일 수 있다. 표시할 때만 반올림한다.
                x, y = (round(float(row[key])) for key in ("x_px", "y_px"))
                estimated = row["status"] == "estimated"
                marker = cv2.MARKER_DIAMOND if estimated else cv2.MARKER_CROSS
                cv2.drawMarker(canvas, (x, y), (0, 225, 255), marker, 15, 2)
                cv2.putText(canvas, str(number), (x + 7, max(15, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 225, 255), 2)
                xy_label = f"({row['x_px']}, {row['y_px']})"
                label = f"추정 {xy_label}" if estimated else f"직접 확인 {xy_label}"
                if trunk_review and row["reason"] == "clothing":
                    label = f"옷 가림 추정 ({x}, {y})"
            labels.append(f"{number} {name} {label}")
        summary.value = " · ".join(labels)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise ValueError("원본 프레임을 표시할 수 없습니다.")
        photo.value = encoded.tobytes()
        quality_button()

    def resize(*_):
        scale = 1 if zoom.value else min(1, 720 / width, 540 / height)
        photo.layout.width = f"{round(width * scale)}px"

    def save(xy=None, reason="", status=None):
        nonlocal points
        name = joint.value
        if xy is None:
            row = dict(frame_info, joint=name, status="unreadable", reason=reason, x_px="", y_px="")
        else:
            x, y = xy
            old = points.get(name, {})
            if trunk_review:
                reason = "" if visibility.value == "located" else visibility.value
                status = "estimated" if reason else "located"
            elif name == "골반":
                status, reason = "estimated", "hip_socket"
            elif status is None:
                status = "estimated" if old.get("status") == "estimated" else "located"
            if status == "estimated" and not reason:
                reason = old.get("reason") if old.get("status") == "estimated" else "occluded"
            row = dict(frame_info, joint=name, status=status, reason=reason, x_px=str(x), y_px=str(y))
        try:
            points = _save_point(path, row, points.get(name))
        except (OSError, ValueError) as error:
            message.value = str(error)
            return
        draw()
        message.value = f"{name} 저장됨 · {sum(name in points for name in joints)}/{len(joints)}개 검수"

    def change_quality(_):
        row = points[joint.value]
        status = "located" if row["status"] == "estimated" else "estimated"
        save((row["x_px"], row["y_px"]), status=status)

    def click(event):
        if event.get("button", 0) != 0:
            return
        # 브라우저 확대나 스크롤 후에도 클릭한 이미지의 크기를 기준으로 환산한다.
        rx = event["relativeX"] / event["boundingRectWidth"]
        ry = event["relativeY"] / event["boundingRectHeight"]
        if 0 <= rx < 1 and 0 <= ry < 1:
            point = (int(rx * width), int(ry * height))
            if performer_box is not None and joint.value == box_option:
                try:
                    message.value = performer_box.click(point)
                except (OSError, ValueError) as error:
                    message.value = str(error)
                draw()
            else:
                save(point)

    def reload(*_):
        nonlocal points
        points = load_points(path)
        if performer_box is not None:
            performer_box.reload()
        if trunk_review:
            select_visibility()
        draw()
        message.value = f"저장본을 읽었습니다 · {sum(name in points for name in joints)}/{len(joints)}개 검수"

    clicks = Event(source=photo, watched_events=["click"])
    clicks.on_dom_event(click)
    hidden.on_click(lambda _: save(reason="occluded"))
    unsure.on_click(lambda _: save(reason="uncertain"))
    quality.on_click(change_quality)
    joint.observe(quality_button, names="value")
    reload_button.on_click(reload)
    zoom.observe(resize, names="value")
    resize()
    draw()
    if trunk_review:
        joint.observe(select_visibility, names="value")
        select_visibility()
    if performer_box is not None:
        def change_tool(*_):
            draw()
            message.value = ("수행자의 전신을 감싸는 두 모서리를 차례로 누르세요."
                             if joint.value == box_option else "판독 상태를 선택하고 관절을 누르세요.")
        joint.observe(change_tool, names="value")
        change_tool()
    buttons = [hidden, unsure, reload_button]
    if not hip_fold:
        buttons.insert(0, visibility if trunk_review else quality)
    actions = widgets.Box(buttons, layout=widgets.Layout(flex_flow="row wrap"))
    return widgets.VBox([title, joint, actions, summary, message, zoom, viewport])
