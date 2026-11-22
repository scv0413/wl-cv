"""원본 라벨을 읽고 개인 작업 파일에 수정 내용을 저장한다."""

import csv
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock

from wlcv.data.plate_labels import check_box, read_boxes

EVENTS = ("ready", "transition", "drive", "catch", "finish")


@contextmanager
def _replacement(path):
    """파일을 모두 쓴 뒤 교체한다. 저장 도중 실패하면 이전 파일을 유지한다."""
    temp = NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            yield temp
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def box_path(annotations, video_id, frame):
    return annotations / "personal_plate_boxes" / f"{video_id}_frame_{frame:04d}.txt"


def load_boxes(annotations, video_id, frame):
    """개인 검수본을 우선한다. 빈 파일도 사람이 검수한 라벨이다."""
    personal = box_path(annotations, video_id, frame)
    original = annotations / "plate_boxes" / f"{video_id}_f{frame}.txt"
    for name, path in (("개인 라벨", personal), ("원본 라벨", original)):
        if path.is_file():
            return name, read_boxes(path, inside_frame=True)
    return "미라벨", []


def save_boxes(annotations, video_id, frame, boxes, *, previous, reviewed_empty=False):
    """확인한 프레임만 저장한다. 다른 창에서 먼저 고치면 다시 열도록 알린다."""
    if not boxes and not reviewed_empty:
        raise ValueError("원판 없는 프레임은 ‘원판 없음 저장’을 누르세요.")
    for box in boxes:
        check_box(box, inside_frame=True)
    path = box_path(annotations, video_id, frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".write.lock"), timeout=5):
        if load_boxes(annotations, video_id, frame) != previous:
            raise ValueError("다른 창에서 라벨을 바꿨습니다. 다시 읽고 수정하세요.")
        with _replacement(path) as output:
            for box in boxes:
                output.write("0 " + " ".join(f"{v:.8f}" for v in box) + "\n")
    return path


def events_for(video_id):
    parts = video_id.split("_")
    if len(parts) < 4 or parts[-2] not in {"f", "h"}:
        raise ValueError("영상 ID에서 전체(f) 또는 하프(h) 동작을 확인하세요.")
    return EVENTS[:3] if parts[-2] == "h" else EVENTS


def _read_events(annotations):
    personal = annotations / "personal_point_labels.csv"
    path = personal if personal.is_file() else annotations / "point_labels.csv"
    fields = ["video_id", "event", "frame", "pass"]
    if not path.is_file():
        return fields, []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"이벤트 CSV의 열 이름을 확인하세요. {path.name}")
        return reader.fieldnames, list(reader)


def load_events(annotations, video_id, pass_number):
    _, rows = _read_events(annotations)
    return {r["event"]: int(r["frame"]) for r in rows
            if r["video_id"] == video_id and r["pass"] == str(pass_number)}


def save_event(annotations, video_id, pass_number, event, frame, *, previous):
    if pass_number not in (1, 2) or frame < 0 or event not in events_for(video_id):
        raise ValueError("판독 차수와 이벤트, 프레임 번호를 확인하세요.")
    path = annotations / "personal_point_labels.csv"
    annotations.mkdir(parents=True, exist_ok=True)
    with FileLock(str(annotations / ".personal_point_labels.lock"), timeout=5):
        fields, rows = _read_events(annotations)
        if pass_number == 1 and any(r["video_id"] == video_id and r["pass"] == "2" for r in rows):
            raise ValueError("2차 판독을 시작해 이 영상의 1차 기록은 잠겨 있습니다.")
        matches = [r for r in rows if r["video_id"] == video_id and
                   r["event"] == event and r["pass"] == str(pass_number)]
        if len(matches) > 1:
            raise ValueError("이 이벤트가 CSV에 두 번 기록되어 있습니다.")
        current = int(matches[0]["frame"]) if matches else None
        if current != previous:
            raise ValueError("다른 창에서 이 이벤트를 바꿨습니다. 다시 읽고 수정하세요.")
        if matches:
            matches[0]["frame"] = str(frame)
        else:
            rows.append({"video_id": video_id, "event": event,
                         "frame": str(frame), "pass": str(pass_number)})
        with _replacement(path) as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path
