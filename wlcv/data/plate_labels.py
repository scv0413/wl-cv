"""프레임 번호와 YOLO 원판 박스 좌표를 읽는다."""

import re
from pathlib import Path


def label_frame(path: str | Path) -> tuple[str, int]:
    """예: a01_sn_f_01_f25.txt → ('a01_sn_f_01', 25)."""
    path = Path(path)
    match = re.fullmatch(r"(.+)_f(\d+)", path.stem)
    if not match:
        raise ValueError(f"라벨 이름은 영상ID_f프레임번호.txt 형식이어야 합니다: {path.name}")
    return match[1], int(match[2])


def check_box(box, *, inside_frame=False):
    x, y, width, height = box
    if inside_frame:
        # 손으로 그린 박스는 프레임 안에 있어야 한다. 저장할 때의 반올림 차이는 허용한다.
        margin = 1e-6
        position_ok = (-margin <= x - width / 2 and x + width / 2 <= 1 + margin
                       and -margin <= y - height / 2 and y + height / 2 <= 1 + margin)
    else:
        position_ok = 0 <= x <= 1 and 0 <= y <= 1
    if not (position_ok and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("원판 박스의 중심과 크기를 확인하세요.")


def read_boxes(path: str | Path, *, inside_frame=False) -> list[tuple[float, float, float, float]]:
    """빈 파일은 원판 없음, 한 줄은 클래스 0과 정규화된 중심 x/y, 너비, 높이다."""
    path = Path(path)
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"{path} {number}행: 클래스 0과 좌표 4개가 필요합니다.")
        try:
            x, y, width, height = map(float, fields[1:])
            box = (x, y, width, height)
            check_box(box, inside_frame=inside_frame)
        except ValueError as error:
            raise ValueError(f"{path} {number}행: {error}") from error
        boxes.append(box)
    return boxes
