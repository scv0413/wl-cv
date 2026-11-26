"""3-fold 학습 선수의 원판 양성 프레임에서 직접 판독한 허브만 모은다."""

from contextlib import ExitStack

import ipywidgets as widgets
import pandas as pd

from wlcv.annotation.hub_points import hub_editor, read_hubs
from wlcv.config import ANNOTATIONS, SPLIT
from wlcv.data.plate_labels import read_boxes

FOLDS = SPLIT.with_name("plate_person_3fold.csv")
POINT_DIR = ANNOTATIONS / "plate_fold_hubs"


def fold_targets():
    """분할표에 포함된 원본 YOLO 라벨을 확인하고 양성 프레임의 키만 반환한다."""
    plan = pd.read_csv(FOLDS)
    people = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    if plan.groupby("person_id").heldout_fold.nunique().gt(1).any():
        raise ValueError("같은 선수가 여러 검증 fold에 배정되어 있습니다.")
    rows = []
    for row in plan.itertuples(index=False):
        video = row.video_id
        if people.at[video, "role"] != "train" or people.at[video, "person_id"] != row.person_id:
            raise ValueError(f"{video}의 기존 학습 분할과 선수 ID를 확인하세요.")
        indices = [int(i) for i in row.frame_indices.split()]
        if len(indices) != row.labeled_frames or len(set(indices)) != len(indices):
            raise ValueError(f"{video}의 분할표 프레임 번호를 확인하세요.")
        positive = [i for i in indices if read_boxes(ANNOTATIONS / "plate_boxes" / f"{video}_f{i}.txt")]
        if len(positive) != row.positive_frames:
            raise ValueError(f"{video}의 분할표와 실제 원판 라벨 장수가 다릅니다.")
        rows.extend((row.heldout_fold, row.person_id, video, i) for i in positive)
    targets = pd.DataFrame(rows, columns=["fold", "person_id", "video_id", "frame_index"])
    if targets.empty or targets.duplicated(["video_id", "frame_index"]).any():
        raise ValueError("분할표의 원판 양성 프레임 목록을 확인하세요.")
    return targets


def review_status(targets, folder=POINT_DIR):
    """각 프레임을 저장된 직접점, 판독 불가, 미판독으로 구분한다."""
    states = {}
    for video, group in targets.groupby("video_id"):
        saved = read_hubs(folder / f"{video}.csv", video=video, frames=group.frame_index)
        for row in saved.itertuples(index=False):
            states[video, row.frame_index] = "직접 확인" if row.visibility == "visible" else "판독 불가"
    keys = targets[["video_id", "frame_index"]].itertuples(index=False, name=None)
    return targets.assign(status=[states.get(key, "미판독") for key in keys])


def progress_table(targets, folder=POINT_DIR):
    status = review_status(targets, folder)
    table = pd.crosstab([status.fold, status.person_id], status.status)
    table = table.reindex(columns=["직접 확인", "판독 불가", "미판독"], fill_value=0)
    table.insert(0, "대상", table.sum(axis=1))
    return table.rename_axis(index=["Fold", "선수"], columns=None)


def fold_hub_editor(targets, folder=POINT_DIR):
    keys = list(targets[["video_id", "frame_index"]].itertuples(index=False, name=None))
    options = [(f"Fold {r.fold} · {r.person_id} · {r.frame_index}번", (r.video_id, r.frame_index))
               for r in targets.itertuples(index=False)]
    picker = widgets.Dropdown(options=options, description="프레임")
    next_frame = widgets.Button(description="다음 미판독")
    progress = widgets.Label()
    panel = widgets.VBox()
    cleanup = ExitStack()

    def remaining():
        status = review_status(targets, folder)
        return [key for key, state in zip(keys, status.status) if state == "미판독"]

    def refresh():
        pending = remaining()
        progress.value = f"{len(keys) - len(pending)}/{len(keys)}장 저장 · 미판독 {len(pending)}장"
        next_frame.disabled = not pending

    def open_frame(*_):
        video, index = picker.value
        frames = targets.loc[targets.video_id.eq(video), "frame_index"].tolist()
        panel.children = ()
        cleanup.close()
        try:
            view = hub_editor(index, path=folder / f"{video}.csv", frames=frames,
                              video=video, on_saved=refresh, cleanup=cleanup)
            panel.children = (view,)
            refresh()
        except (OSError, ValueError, IndexError) as exc:
            progress.value = f"프레임을 열 수 없습니다. {exc}"

    def go_next(_):
        pending = set(remaining())
        start = keys.index(picker.value) + 1
        for key in keys[start:] + keys[:start]:
            if key in pending:
                picker.value = key
                break

    pending = remaining()
    picker.value = pending[0] if pending else keys[0]
    picker.observe(open_frame, names="value")
    next_frame.on_click(go_next)
    open_frame()
    return widgets.VBox([progress, widgets.HBox([picker, next_frame]), panel])
