"""각 선수는 자신을 학습에 쓰지 않은 한 fold에서만 채점한다."""

import numpy as np
import pandas as pd

from wlcv.annotation.fold_hubs import POINT_DIR
from wlcv.annotation.hub_points import read_hubs
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT
from wlcv.data.plate_labels import label_frame, read_boxes
from wlcv.evaluation.epoch_pair import summarize_epochs
from wlcv.evaluation.full_video import compare_videos, detect_videos
from wlcv.evaluation.plate_track import KEY, BOX
from wlcv.evaluation.schedule_review import METHODS, compare_runs
from wlcv.training.plate_epoch_pair import completed_weights
from wlcv.training.plate_folds import OUTPUT, run_dir


def fold_truth(data):
    sizes = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    rows = []
    for path in sorted((data.parent / "labels/val").glob("*.txt")):
        video, index = label_frame(path)
        boxes = read_boxes(path)
        coords = [np.nan] * 4
        if boxes:
            (cx, cy, bw, bh), = boxes
            w, h = sizes.loc[video, ["width", "height"]]
            coords = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
        rows.append((video, index, *coords))
    truth = pd.DataFrame(rows, columns=KEY + BOX)
    hubs = []
    truth["in_motion"] = False
    for video, group in truth[truth.x1.notna()].groupby("video_id"):
        points = read_hubs(POINT_DIR / f"{video}.csv", video=video, frames=group.frame_index)
        if set(points.frame_index) != set(group.frame_index):
            raise ValueError(f"{video}에 아직 저장하지 않은 허브 판독이 있습니다.")
        visible = points[points.visibility.eq("visible")]
        w, h = sizes.loc[video, ["width", "height"]]
        if not (visible.manual_x.between(0, w, inclusive="left") &
                visible.manual_y.between(0, h, inclusive="left")).all():
            raise ValueError(f"{video}의 허브 좌표가 화면 밖에 있습니다.")
        hubs.append(points)
        events = load_events(ANNOTATIONS, video, pass_number=1)
        if "ready" in events:
            inside = group.frame_index.between(events["ready"], max(events.values()))
            truth.loc[group.index[inside], "in_motion"] = True
    return truth.merge(pd.concat(hubs), on=KEY, how="left", validate="one_to_one")


def background_stats(details, truth, fold, epoch):
    """영상 전체가 원판 없는 것으로 확인된 선수만 오탐과 최장 연속 길이를 잰다."""
    rows = []
    for method, name in METHODS.items():
        videos = truth.loc[truth.x1.isna(), "video_id"].unique()
        for video in videos:
            if video not in {"b05_sn_f_01", "b06_sn_f_01"}:
                raise ValueError(f"{video}의 영상 전체가 원판 없는지 먼저 확인하세요.")
            group = details[details.video_id.eq(video) & details.method.eq(method)].sort_values("frame_index")
            found = group.x1.notna()
            labeled = group.frame_index.isin(truth.loc[truth.video_id.eq(video), "frame_index"])
            longest = int(found.groupby((~found).cumsum()).sum().max())
            rows.append([fold, epoch, name, video, int(labeled.sum()), int((found & labeled).sum()),
                         len(group), int(found.sum()), found.mean(), longest])
        if not len(videos):
            rows.append([fold, epoch, name, "대상 없음", 0, 0, 0, 0, np.nan, np.nan])
    return pd.DataFrame(rows, columns=["fold", "에폭", "방식", "배경 영상", "음성 라벨 장수",
                        "음성 라벨 오탐", "전체 프레임", "오탐 프레임", "오탐 비율", "최장 연속 오탐(장)"])


def evaluate_fold(fold, data, split_csv):
    weights = completed_weights(run_dir(fold), data=data)
    truth = fold_truth(data)
    videos = sorted(truth.video_id.unique())
    records, negatives = {}, []
    for epoch, checkpoint in weights.items():
        frames, candidates = detect_videos(checkpoint, videos, split_csv=split_csv)
        if len(truth.merge(frames, on=KEY, validate="one_to_one")) != len(truth):
            raise ValueError("원본에서 빠진 검증 라벨 프레임이 있습니다.")
        review = frames.merge(truth, on=KEY, how="left", validate="one_to_one")
        detail, _ = compare_videos(frames, candidates, review)
        detail["dy_px"] = detail.cy - detail.manual_y
        detail["error_px"] = np.hypot(detail.cx - detail.manual_x, detail.dy_px)
        records[epoch] = detail
        negatives.append(background_stats(detail, truth, fold, epoch))

    pair = compare_runs(records[65], records[95], labels=("65", "95"), motion_only=False)
    if len(pair) != truth.x1.notna().sum() * len(METHODS):
        raise ValueError("비교표의 양성 프레임 수를 확인하세요.")
    pair.insert(0, "fold", fold)
    background = pd.concat(negatives, ignore_index=True)
    folder = OUTPUT / f"fold_{fold}"
    pair.to_csv(folder / "paired_frames.csv", index=False)
    background.to_csv(folder / "background.csv", index=False)
    return pair, background


def summarize_folds(pairs):
    """미탐은 모든 양성에서 세고, 세로 오차는 양쪽 모두 정탐인 같은 수동점에서 비교한다."""
    if pairs.duplicated(KEY + ["method"]).any():
        raise ValueError("같은 선수의 프레임이 두 번 포함됐습니다.")
    rows = []
    for label, group in [("선택 회차 합계", pairs), *pairs.groupby("fold")]:
        for scope, sample in [("양성 전체", group), ("동작 중", group[group.in_motion.eq(True)])]:
            table = summarize_epochs(sample)
            if label == "선택 회차 합계":
                table = table[table["영상"].eq("전체")]
            rows.append(table.assign(fold=label, 범위=scope))
    return pd.concat(rows, ignore_index=True)
