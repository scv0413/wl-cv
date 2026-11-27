"""원판 박스의 네 변을 원본 프레임 좌표로 조정하는 주피터 위젯."""

import cv2
import ipywidgets as widgets


def normalized_box(left, top, right, bottom, width, height):
    return ((left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


class BoxEdgeSliders(widgets.VBox):
    def __init__(self, boxes, view, on_edit):
        self.boxes = boxes
        self.view = view
        self.on_edit = on_edit
        self.syncing = False
        self.selector = widgets.Dropdown(description="조정할 박스")
        self.edges = [
            widgets.IntSlider(description=label, step=1, continuous_update=False,
                              layout=widgets.Layout(width="95%"))
            for label in ("왼쪽 (px)", "위 (px)", "오른쪽 (px)", "아래 (px)")
        ]
        self.detail = widgets.Image(format="png", layout=widgets.Layout(
            max_width="100%", height="auto", align_self="flex-start"))
        hint = widgets.Label("원본 픽셀 좌표 · 슬라이더를 선택하고 방향키로 1 px씩 이동")
        super().__init__([self.selector, hint, *self.edges, self.detail])
        self.selector.observe(self.sync, names="value")
        for slider in self.edges:
            slider.observe(self.edit, names="value")

    def select_last(self):
        """새로 그린 박스 또는 다시 읽은 프레임에서 마지막 박스를 선택한다."""
        self.syncing = True
        try:
            self.selector.options = [(f"박스 {i + 1}", i) for i in range(len(self.boxes))]
            self.selector.index = len(self.boxes) - 1 if self.boxes else None
            self.selector.layout.display = "" if len(self.boxes) > 1 else "none"
        finally:
            self.syncing = False
        self.sync()

    def pixel_edges(self):
        height, width = self.view.original_rgb.shape[:2]
        x, y, w, h = self.boxes[self.selector.value]
        # 예전 라벨의 소수점 좌표도 그대로 읽고 사용자가 움직인 변만 변경한다.
        return [(x - w / 2) * width, (y - h / 2) * height,
                (x + w / 2) * width, (y + h / 2) * height]

    def sync(self, _=None):
        if self.syncing:
            return
        active = self.selector.value is not None and self.view.original_rgb is not None
        self.layout.display = "" if active else "none"
        if not active:
            self.detail.value = b""
            return
        height, width = self.view.original_rgb.shape[:2]
        bounds = [round(v) for v in self.pixel_edges()]
        self.syncing = True
        try:
            for slider, value, limit in zip(self.edges, bounds, (width, height, width, height)):
                slider.max = limit
                slider.value = value
        finally:
            self.syncing = False
        self.show_detail(*bounds)

    def edit(self, change):
        if self.syncing or self.selector.value is None:
            return
        height, width = self.view.original_rgb.shape[:2]
        values = self.pixel_edges()
        edge = self.edges.index(change["owner"])
        value = change["new"]
        # 오른쪽·아래쪽은 이미지 끝 경계까지 허용하고 두 변이 교차하지 않게 한다.
        if edge < 2:
            value = min(value, max(0, round(values[edge + 2]) - 1))
        else:
            limit = width if edge == 2 else height
            value = max(value, min(limit, round(values[edge - 2]) + 1))
        values[edge] = value
        self.boxes[self.selector.value] = normalized_box(*values, width, height)
        self.sync()
        self.on_edit()

    def show_detail(self, left, top, right, bottom):
        """원본에서 박스 주변만 잘라 한 픽셀 두께로 표시한다."""
        rgb = self.view.original_rgb
        height, width = rgb.shape[:2]
        x0, y0 = max(0, left - 16), max(0, top - 16)
        crop = rgb[y0:min(height, bottom + 16), x0:min(width, right + 16)].copy()
        cv2.rectangle(crop, (left - x0, top - y0),
                      (max(left, right - 1) - x0, max(top, bottom - 1) - y0),
                      (35, 235, 105), 1)
        ok, png = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("원판 테두리 미리보기를 표시할 수 없습니다.")
        self.detail.value = png.tobytes()
