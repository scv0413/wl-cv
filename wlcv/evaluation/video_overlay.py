"""원판과 가까운 쪽 자세 예측을 원본 픽셀 좌표에 그린다."""

import cv2

from wlcv.evaluation.pose import joint_table

PLATE_STATES = {
    "high": ("직접 검출", "DIRECT", (95, 230, 100)),
    "linked": ("시간 연결", "LINKED", (30, 190, 255)),
    "missing": ("미탐", "MISSING", (80, 80, 255)),
}
POSE_COLOR = (255, 220, 60)


def draw_pose(image, result, near_side):
    """수행자 후보가 하나일 때만 가까운 쪽 어깨·골반·무릎·발목을 연결한다."""
    count = len(result.boxes)
    record = {"person_candidates": count, "pose_side": near_side, "joints_drawn": 0}
    if count != 1:
        return record

    points = joint_table(result, person_number=1, min_confidence=0.50)
    previous = None
    for point in points.loc[points["방향"] == near_side].to_dict("records"):
        if point["표시 상태"] != "표시":
            previous = None
            continue
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
    if pose["person_candidates"] == 1:
        detail = f"{pose['pose_side']}   {pose['joints_drawn']}/4 joints"
    else:
        detail = f"withheld   {pose['person_candidates']} candidates"
    lines.append((f"Pose: {detail}", POSE_COLOR))
    if phase:
        lines.append((f"Manual: {phase}", (255, 255, 255)))
    cv2.rectangle(image, (12, 12), (min(image.shape[1]-1, 690), 25 + 42*len(lines)), (24, 24, 24), -1)
    for number, (message, color) in enumerate(lines, start=1):
        cv2.putText(image, message, (24, 12 + 42*number), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, color, 2, cv2.LINE_AA)
