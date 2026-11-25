"""두 모서리를 클릭해 한 프레임의 수행자 박스를 원본 픽셀로 저장한다."""

import csv

import cv2
from filelock import FileLock

from .storage import _replacement


class PerformerBox:
    def __init__(self, folder, video_id, frame):
        self.path = folder / "person_boxes" / f"{video_id}_frame_{frame:04d}.csv"
        self.frame = {"video_id": video_id, "frame_index": frame}
        self.reload()

    def read(self):
        if not self.path.is_file():
            return None
        with self.path.open(encoding="utf-8", newline="") as source:
            row, = csv.DictReader(source)
        if row["video_id"] != self.frame["video_id"] or int(row["frame_index"]) != self.frame["frame_index"]:
            raise ValueError(f"수행자 박스의 프레임 정보를 확인하세요. {self.path.name}")
        return tuple(int(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))

    def reload(self):
        self.saved = self.read()
        self.first_corner = None

    def click(self, point):
        if self.first_corner is None:
            self.first_corner = point
            return "첫 모서리를 선택했습니다. 대각선 반대편 모서리를 누르세요."
        x1, x2 = sorted((self.first_corner[0], point[0]))
        y1, y2 = sorted((self.first_corner[1], point[1]))
        if x1 == x2 or y1 == y2:
            return "가로와 세로가 모두 생기도록 반대편 모서리를 다시 누르세요."
        self.save((x1, y1, x2, y2))
        return "수행자 박스를 저장했습니다. 관절을 선택하거나 박스를 다시 그릴 수 있습니다."

    def save(self, bounds):
        """확정한 두 모서리를 기록한다. 다른 창이 먼저 바꾼 박스는 보호한다."""
        x1, y1, x2, y2 = bounds
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("왼쪽 위와 오른쪽 아래 모서리를 확인하세요.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path.parent / ".write.lock"), timeout=5):
            if self.read() != self.saved:
                raise ValueError("다른 창에서 박스를 바꿨습니다. 다시 읽기를 누르세요.")
            row = dict(self.frame, xmin=x1, ymin=y1, xmax=x2, ymax=y2)
            with _replacement(self.path) as output:
                writer = csv.DictWriter(output, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
        self.saved = (x1, y1, x2, y2)
        self.first_corner = None

    def draw(self, canvas):
        if self.saved:
            x1, y1, x2, y2 = self.saved
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (65, 230, 130), 3)
        if self.first_corner:
            cv2.drawMarker(canvas, self.first_corner, (65, 230, 130), cv2.MARKER_TILTED_CROSS, 20, 3)
