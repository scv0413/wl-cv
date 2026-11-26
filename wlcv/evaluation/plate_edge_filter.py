"""같은 원본 후보에서 화면 경계 필터만 바꿔 두 모델의 시간 연결을 다시 계산한다."""
import pandas as pd

from wlcv.annotation.storage import EVENTS, load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT, RUNS, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.full_video import compare_videos, predict_plate_videos, validation_plate_model
from wlcv.evaluation.plate_cutmix_fallback import fill_missing_boxes
from wlcv.evaluation.plate_track import BOX, KEY
from wlcv.evaluation.schedule_review import review_inputs

OUTPUT = ROOT / "outputs/plate_edge_filter"
BACKGROUNDS = ["b05_sn_f_01", "b06_sn_f_01"]
WEIGHTS = {
    "mac": RUNS / "plate_baseline/weights/best.pt",
    "cutmix": ROOT / "outputs/plate_cutmix_3090/20261126/runs/cutmix/weights/best.pt",
}


def infer_originals():
    """두 모델 모두 프레임 0부터 추론한다. 대조·실험 조건은 같은 후보를 공유한다."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    val, _, _ = review_inputs(DATASET / "data.yaml")
    for name, weights in WEIGHTS.items():
        model = validation_plate_model(weights, val)
        frames, candidates = predict_plate_videos(model, val + BACKGROUNDS, device="cpu", imgsz=640)
        frames.to_csv(OUTPUT / f"{name}_frames.csv", index=False)
        candidates.to_csv(OUTPUT / f"{name}_candidates.csv", index=False)


def edge_mask(candidates, dimensions):
    """xyxy는 원본 픽셀 경계 좌표다. 오른쪽과 아래쪽 끝은 각각 width와 height다."""
    width = candidates.video_id.map(dimensions.width)
    height = candidates.video_id.map(dimensions.height)
    box_width = candidates.x2 - candidates.x1
    box_height = candidates.y2 - candidates.y1
    touches_edge = ((candidates.x1 <= 1) | (candidates.y1 <= 1)
                    | (width - candidates.x2 <= 1) | (height - candidates.y2 <= 1))
    return touches_edge & (box_height > 0) & (box_width >= 3 * box_height)


def replay():
    val, boxes, _ = review_inputs(DATASET / "data.yaml")
    videos = val + BACKGROUNDS
    dimensions = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id").loc[videos]
    for video, size in dimensions.iterrows():
        if read_rgb(VIDEOS / f"{video}.mp4", 0).shape[:2] != (size.height, size.width):
            raise ValueError(f"{video}의 원본 크기를 확인하세요.")
    tracks = {"before": {}, "after": {}}
    excluded = []
    for model in WEIGHTS:
        frames = pd.read_csv(OUTPUT / f"{model}_frames.csv")
        candidates = pd.read_csv(OUTPUT / f"{model}_candidates.csv", float_precision="round_trip")
        expected = [(video, i) for video in videos for i in range(int(dimensions.at[video, "frame_count"]))]
        if list(frames[KEY].itertuples(index=False, name=None)) != expected:
            raise ValueError("원본의 연속 프레임 목록이 다릅니다.")
        review = frames.merge(boxes, on=KEY, how="left", validate="one_to_one")
        rejected = edge_mask(candidates, dimensions)
        excluded.append(candidates[rejected].assign(model=model))
        for condition, pool in [("before", candidates), ("after", candidates[~rejected])]:
            # 각 조건마다 영상 첫 프레임에서 새 이력으로 시작한다.
            tracks[condition][model], _ = compare_videos(frames, pool, review)
    rejected = pd.concat(excluded, ignore_index=True)
    rejected.to_csv(OUTPUT / "rejected.csv", index=False)
    keys = KEY + ["method"]
    combined = {}
    for condition, models in tracks.items():
        data = fill_missing_boxes(models["mac"], models["cutmix"])
        data.to_csv(OUTPUT / f"{condition}.csv", index=False)
        combined[condition] = data.set_index(keys, verify_integrity=True)
    pair = combined["before"].join(combined["after"], lsuffix="_before", rsuffix="_after", validate="one_to_one")
    pair = pair.reset_index()
    pair["in_motion"] = False
    for video in val:
        events = load_events(ANNOTATIONS, video, pass_number=1)
        last = max(events[k] for k in EVENTS if k in events)
        mask = pair.video_id.eq(video) & pair.frame_index.between(events["ready"], last)
        pair.loc[mask, "in_motion"] = True
    pair.to_csv(OUTPUT / "paired.csv", index=False)
    return pair


def summarize(pair):
    """시간 연결 후 같은 프레임끼리 대조하고, 빈 배경은 프레임 단위로 별도 집계한다."""
    linked = pair[pair.method.eq("Past link")]
    scored = linked[linked.in_motion & linked.match_before.notna()].copy()
    if len(scored) != 270 or scored.match_after.isna().any():
        raise ValueError("동작 중 270장의 수동 정답을 확인하세요.")
    recovered = scored[scored.video_id.eq("a03_sn_f_01") & scored.used_cutmix_before & scored.match_before]
    if set(recovered.frame_index) != {4, 5, 6, 10, 11, 12, 13, 14}:
        raise ValueError("기존 가림 복구 8장을 재현하지 못했습니다.")
    counts = []
    for label, group in [("동작 중", scored), ("a03 기존 복구", recovered)]:
        before = group.match_before.astype(bool)
        after = group.match_after.astype(bool)
        counts.append((label, len(group), int(before.sum()), int(after.sum()),
                       int((before & ~after).sum()), int((~before & after).sum())))
    motion = pd.DataFrame(counts, columns=["구간","프레임","기존 정탐","적용 정탐","손실","복구"])
    backgrounds = []
    for video in BACKGROUNDS:
        for method, group in pair[pair.video_id.eq(video)].groupby("method"):
            for condition in ["before","after"]:
                present = group.sort_values("frame_index")[f"x1_{condition}"].notna()
                longest = int(present.groupby((~present).cumsum()).sum().max())
                backgrounds.append((video, method, condition, len(group), int(present.sum()), longest))
    bg = pd.DataFrame(backgrounds, columns=["video","method","condition","frames","fp","longest"])
    motion.to_csv(OUTPUT / "motion.csv", index=False)
    bg.to_csv(OUTPUT / "backgrounds.csv", index=False)
    recovered.to_csv(OUTPUT / "a03_recoveries.csv", index=False)
    return motion, bg


def changed_outputs(pair):
    """같은 영상·프레임에서 선택된 박스나 출처가 바뀐 경우만 남긴다."""
    same = pd.Series(True, index=pair.index)
    for column in [*BOX, "source", "used_cutmix"]:
        a, b = pair[f"{column}_before"], pair[f"{column}_after"]
        same &= a.eq(b) | (a.isna() & b.isna())
    changed = pair.loc[~same].copy()
    changed.to_csv(OUTPUT / "changed_outputs.csv", index=False)
    return changed
