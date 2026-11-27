"""원판과 가까운 쪽 자세 예측을 원본 픽셀 좌표에 그린다."""

from collections import deque
from time import perf_counter

import cv2
import pandas as pd
from tqdm import tqdm

from wlcv.data.frames import timed_frames_rgb
from wlcv.data.mp4_input import VideoInputError
from wlcv.runtime.foreground import select_performer
from wlcv.runtime.inference import predict_rgb
from wlcv.runtime.measurements import vertical_angle
from wlcv.runtime.pose import joint_table
from wlcv.runtime.video_writer import open_h264_writer

PLATE_STATES = {
    "high": ("직접 검출", "DIRECT", (95, 230, 100)),
    "linked": ("시간 연결", "LINKED", (30, 190, 255)),
    "missing": ("미탐", "MISSING", (80, 80, 255)),
}
POSE_COLOR = (255, 220, 60)


def draw_pose(image, result, near_side, *, selection=None):
    """선택한 수행자의 가까운 쪽 어깨·골반·무릎·발목을 연결한다."""
    count = len(result.boxes)
    if selection is None:
        index, reason = select_performer(result.boxes.xyxy, None)
        number = None if index is None else index + 1
    else:
        number, reason = selection
    record = {"person_candidates": count, "performer_number": number, "performer_selection": reason, "pose_side": near_side, "joints_drawn": 0,
              "shoulder_x_px": None, "shoulder_y_px": None, "trunk_angle_deg": None}
    box = result.boxes.xyxy[number - 1] if number is not None else (None,) * 4
    for key, value in zip(("performer_x1", "performer_y1", "performer_x2", "performer_y2"), box):
        record[key] = float(value) if value is not None else None
    if number is None:
        return record
    if count > 1:
        left, top, right, bottom = map(round, box)
        cv2.rectangle(image, (left, top), (right, bottom), POSE_COLOR, 2)

    points = joint_table(result, person_number=number, min_confidence=0.50)
    previous = None
    for point in points.loc[points["방향"] == near_side].to_dict("records"):
        if point["표시 상태"] != "표시":
            previous = None
            continue
        if point["관절"] == "어깨":
            record["shoulder_x_px"] = float(point["x (px)"])
            record["shoulder_y_px"] = float(point["y (px)"])
        if point["관절"] == "골반" and record["shoulder_x_px"] is not None:
            shoulder = (record["shoulder_x_px"], record["shoulder_y_px"])
            record["trunk_angle_deg"] = vertical_angle(shoulder, (point["x (px)"], point["y (px)"]))
        xy = (round(point["x (px)"]), round(point["y (px)"]))
        if previous is not None:
            cv2.line(image, previous, xy, POSE_COLOR, 4, cv2.LINE_AA)
        cv2.circle(image, xy, 8, POSE_COLOR, -1, cv2.LINE_AA)
        previous = xy
        record["joints_drawn"] += 1
    return record


def draw_plate(image, plate, trail):
    """직접 검출과 시간 연결을 구분하고 미탐을 가로질러 궤적을 잇지 않는다."""
    _, _, color = PLATE_STATES[plate.source]
    if plate.source == "missing":
        trail.clear()
        return
    center = (round(plate.cx), round(plate.cy))
    cv2.rectangle(image, (round(plate.x1), round(plate.y1)),
                  (round(plate.x2), round(plate.y2)), color, 3)
    trail.append(center)
    for previous, current in zip(list(trail)[:-1], list(trail)[1:]):
        cv2.line(image, previous, current, color, 3, cv2.LINE_AA)
    cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 24, 4)


def draw_caption(image, plate, pose, phase):
    _, label, color = PLATE_STATES[plate.source]
    score = "" if plate.source == "missing" else f"  {plate.confidence:.3f}"
    lines = [(f"Frame {plate.frame_index}   Plate {label}{score}", color)]
    if hasattr(plate, "selected_model"):
        lines.append((f"Plate model: {plate.selected_model.upper()}", color))
    if pose["performer_number"] is not None:
        detail = (f"{pose['performer_number']}/{pose['person_candidates']}   {pose['pose_side']}   "
                  f"{pose['joints_drawn']}/4 joints")
    else:
        detail = f"withheld   {pose['person_candidates']} candidates"
    lines.append((f"Pose: {detail}", POSE_COLOR))
    if phase:
        lines.append((f"Manual: {phase}", (255, 255, 255)))
    right = min(image.shape[1] - 1, 690)
    height = 13 + 42 * len(lines)
    top = 12
    if (plate.source != "missing" and plate.x1 < right and plate.x2 > 12
            and plate.y1 < top + height and plate.y2 > top):
        top = max(12, image.shape[0] - height - 12)
    cv2.rectangle(image, (12, top), (right, top + height), (24, 24, 24), -1)
    for number, (message, color) in enumerate(lines, start=1):
        cv2.putText(image, message, (24, top + 42*number), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, color, 2, cv2.LINE_AA)


def two_pass_frames(path, plates, pose_model, device, timings):
    pairs = zip(timed_frames_rgb(path), plates.itertuples(index=False), strict=True)
    for (time_ms, rgb), plate in pairs:
        started = perf_counter()
        result = predict_rgb(pose_model, rgb, conf=0.25, imgsz=640, device=device).cpu().numpy()
        timings["pose"] += perf_counter() - started
        yield time_ms, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), plate, result


def render_frames(frame_stream, info, fps, lift, knee, output, phase_events, timings, *, endpoint_images=None):
    size = (int(info.width + info.width % 2), int(info.height + info.height % 2))
    writer = open_h264_writer(output, fps, size)
    records = []
    draw_seconds = write_seconds = 0.0
    trail = deque(maxlen=12)
    events = {lift: ("이륙", "lift-off"), knee: ("무릎 통과", "knee pass")}
    for name, korean in [("drive", "접촉"), ("catch", "캐치"), ("finish", "피니시")]:
        if name in (phase_events or {}):
            events[phase_events[name]] = (korean, name)
    try:
        for index, (time_ms, image, plate, result) in enumerate(
                tqdm(frame_stream, total=info.frame_count, desc="자세와 오버레이", unit="장")):
            if index != plate.frame_index or image.shape[:2] != (info.height, info.width):
                raise ValueError(f"{index}번 원본의 크기 또는 프레임 번호를 확인하세요.")
            if endpoint_images is not None and index in endpoint_images:
                endpoint_images[index] = image.copy()  # 그리기 전 BGR 원본 두 장만 보존한다.
            drawing = perf_counter()
            selection = ((plate.performer_number, plate.performer_selection)
                         if hasattr(plate, "performer_selection") else None)
            pose = draw_pose(image, result, info.near_side, selection=selection)
            draw_plate(image, plate, trail)
            phase, caption = events.get(index, ("", ""))
            draw_caption(image, plate, pose, caption)
            drawn = perf_counter()
            draw_seconds += drawn - drawing
            # H.264는 짝수 크기로 저장한다. 홀수 원본의 마지막 행·열도 보존한다.
            if info.height % 2 or info.width % 2:
                image = cv2.copyMakeBorder(image, 0, info.height % 2, 0, info.width % 2, cv2.BORDER_CONSTANT)
            writer.write(image)
            write_seconds += perf_counter() - drawn
            records.append({
                "video_id": plate.video_id, "frame_index": index, "source_time_ms": time_ms,
                "manual_phase": phase, "plate_status": PLATE_STATES[plate.source][0],
                "plate_confidence": plate.confidence, "plate_cx_px": plate.cx, "plate_cy_px": plate.cy,
                "plate_x1": plate.x1, "plate_y1": plate.y1,
                "plate_x2": plate.x2, "plate_y2": plate.y2, **pose,
            })
            if hasattr(plate, "selected_model"):
                records[-1]["selected_model"] = plate.selected_model
    finally:
        started = perf_counter()
        writer.release()  # 인코더에 남은 프레임과 MP4 컨테이너를 마무리한다.
        write_seconds += perf_counter() - started
    if not records:
        raise VideoInputError("영상에서 디코딩한 프레임이 없습니다.")
    if info.frame_count is not None and len(records) != info.frame_count:
        raise ValueError("원본 디코딩 장수와 등록된 프레임 수가 다릅니다.")
    timings.update(drawing=draw_seconds, writer=write_seconds)
    return pd.DataFrame(records)


def render(path, plates, info, fps, pose_model, lift, knee, output, phase_events=None, *, device="cpu", timings=None):
    """기존 두 번 읽기 렌더링 호출도 동일한 그리기·인코딩 경로를 사용한다."""
    if timings is None:
        timings = {}
    timings["pose"] = 0.0
    stream = two_pass_frames(path, plates, pose_model, device, timings)
    return render_frames(stream, info, fps, lift, knee, output, phase_events, timings)
