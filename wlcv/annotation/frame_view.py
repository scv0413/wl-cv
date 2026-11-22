"""두 라벨링 도구에서 같은 크기와 번호로 프레임을 표시한다."""

import cv2
import ipywidgets as widgets
import numpy as np

from wlcv.data.frames import read_rgb


class FrameView:
    def __init__(self, video_path, start_frame=0):
        self.video_path = video_path
        cap = cv2.VideoCapture(str(video_path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if count < 1:
            raise ValueError(f"영상을 열 수 없습니다. {video_path}")
        self.slider = widgets.IntSlider(description="프레임", min=0, max=count - 1,
                                        value=start_frame, continuous_update=False,
                                        layout=widgets.Layout(width="95%"))
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(
            max_width="100%", height="auto", align_self="flex-start"))
        self.rgb = None

    def read(self):
        self.rgb = None
        self.image.value = b""
        rgb = read_rgb(self.video_path, self.slider.value)
        height, width = rgb.shape[:2]
        scale = min(1, 720 / width, 540 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self.rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        self.image.layout.width = f"{size[0]}px"

    def show(self, boxes=(), preview=None):
        canvas = self.rgb.copy()
        height, width = canvas.shape[:2]
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            start = self._pixel(x - w / 2, y - h / 2)
            end = self._pixel(x + w / 2, y + h / 2)
            cv2.rectangle(canvas, start, end, (35, 235, 105), 2)
            cv2.putText(canvas, str(i), (start[0], min(height - 1, start[1] + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 235, 105), 1)
        if preview:
            a, b = preview
            cv2.rectangle(canvas, self._pixel(*a), self._pixel(*b), (255, 180, 30), 2)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("프레임을 표시할 수 없습니다.")
        self.image.value = jpg.tobytes()

    def _pixel(self, x, y):
        height, width = self.rgb.shape[:2]
        return (int(np.clip(round(x * width), 0, width - 1)),
                int(np.clip(round(y * height), 0, height - 1)))
