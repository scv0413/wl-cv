"""같은 3090 모델 종류에서 scale과 CutMix 네 조합을 Mac CPU로 채점한다."""
import json

import pandas as pd
import yaml

from wlcv.config import DATASET, ROOT, SPLIT
from wlcv.data.plate_labels import label_frame
from wlcv.evaluation.schedule_review import evaluate_run
from wlcv.training.plate_cutmix import check_same_labels, longest_miss, score_pair

PREVIOUS = ROOT / "outputs/plate_cutmix_3090/20261126"
CURRENT = ROOT / "outputs/plate_scale_cutmix"
REVIEW = CURRENT / "review"


def evaluate_new_runs():
    """새 best.pt 두 개만 추론한다. 기존 0.5 실행과 입력·설정의 일치도 확인한다."""
    REVIEW.mkdir(parents=True, exist_ok=True)
    meta = [json.loads((folder / "environment.json").read_text()) for folder in (PREVIOUS, CURRENT)]
    if meta[0]["data_sha256"] != meta[1]["data_sha256"] or meta[0]["packages"] != meta[1]["packages"]:
        raise ValueError("두 GPU 학습 실행의 입력 또는 패키지 버전이 다릅니다.")
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    video_ids = sorted({label_frame(p)[0] for p in (DATASET / "labels/train").glob("*.txt")})
    people = split.loc[video_ids]
    if not people.role.eq("train").all() or people.person_id.nunique() != 8:
        raise ValueError("학습 분할을 확인하세요.")
    metrics = []
    for arm, probability in (("control", 0.0), ("cutmix", 0.30)):
        previous = PREVIOUS / "runs" / arm
        current = CURRENT / "runs" / arm
        prev, now = [yaml.safe_load((p / "args.yaml").read_text()) for p in (previous, current)]
        changes = {key for key in set(prev) | set(now) if prev.get(key) != now.get(key)}
        if changes != {"scale"} or (prev["scale"], now["scale"], now["cutmix"]) != (0.5, 0.25, probability):
            raise ValueError(f"{arm} 실행 설정의 차이를 확인하세요: {changes}")
        if pd.read_csv(current / "results.csv").epoch.tolist() != list(range(1, 101)):
            raise ValueError(f"{arm}의 학습 기록이 완전하지 않습니다.")
        people[["person_id"]].to_csv(current / "train_people.csv")
        metric, frames = evaluate_run(current, DATASET / "data.yaml", REVIEW)
        frames.to_csv(REVIEW / f"{arm}_frames.csv", index=False)
        metrics.append({"실행 폴더": arm, "mAP50–95": metric})
    pd.DataFrame(metrics).to_csv(REVIEW / "metrics.csv", index=False)


def summarize():
    """동작 중 장수, 실제 가림의 연속 길이, 원판 박스의 크기를 원본 좌표로 비교한다."""
    scores, sizes, pairs = [], [], {}
    for scale, folder in [(0.5, PREVIOUS / "paired_review"), (0.25, REVIEW)]:
        metrics = pd.read_csv(folder / "metrics.csv").set_index("실행 폴더")
        for arm, probability in [("control", 0.0), ("cutmix", 0.3)]:
            name = "baseline" if scale == 0.5 and arm == "control" else arm
            frame = pd.read_csv(folder / f"{name}_frames.csv", dtype={"match": "boolean"})
            if pairs:
                check_same_labels(next(iter(pairs.values())), frame)
            pairs[(scale, probability)] = frame
            for method, g in frame.groupby("method"):
                motion = g[g.in_motion & g.match.notna()].sort_values(["video_id", "frame_index"])
                early = motion[motion.video_id.eq("a03_sn_f_01") & motion.frame_index.between(2, 14)]
                b02 = motion[motion.video_id.eq("b02_sn_h_02") & motion.frame_index.between(32, 35)]
                if len(motion) != 270 or len(early) != 13 or len(b02) != 4:
                    raise ValueError("동일한 동작 프레임이 모두 들어와야 합니다.")
                scores.append((scale, probability, method, metrics.at[arm,"mAP50–95"],
                               int((~motion.match).sum()), int((~early.match).sum()),
                               longest_miss(early, "match"), int(b02.match.sum())))
                for row in b02.itertuples():
                    width = (row.x2 - row.x1) / (row.gt_x2 - row.gt_x1)
                    height = (row.y2 - row.y1) / (row.gt_y2 - row.gt_y1)
                    center_dx = (row.x1 + row.x2 - row.gt_x1 - row.gt_x2) / 2
                    center_dy = (row.y1 + row.y2 - row.gt_y1 - row.gt_y2) / 2
                    sizes.append((scale, probability, method, row.frame_index, row.source,
                                  width, height, width * height, center_dx, center_dy, row.iou, row.match))
    scores = pd.DataFrame(scores, columns=["scale","cutmix","method","map","motion_misses",
                                          "a03_misses","a03_longest","b02_hits"])
    sizes = pd.DataFrame(sizes, columns=["scale","cutmix","method","frame","source",
                                         "width_ratio","height_ratio","area_ratio","box_dx_px","box_dy_px","iou","match"])
    scores.to_csv(REVIEW / "four_conditions.csv", index=False)
    sizes.to_csv(REVIEW / "b02_box_sizes.csv", index=False)
    for prob in (0.0, 0.3):
        paired, summary = score_pair(pairs[(0.5,prob)], pairs[(0.25,prob)])
        paired.to_csv(REVIEW / f"paired_scale_cutmix_{prob}.csv", index=False)
        summary.to_csv(REVIEW / f"scale_change_cutmix_{prob}.csv", index=False)
    return scores, sizes


def plot_boxes(video_id, frame_index):
    """동일 원본을 네 모델에 공통으로 확대하고 수동·예측 박스를 그린다."""
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    from wlcv.config import VIDEOS
    from wlcv.data.frames import read_rgb

    rgb = read_rgb(VIDEOS / f"{video_id}.mp4", frame_index)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for row, (scale, folder) in enumerate([(0.5, PREVIOUS / "paired_review"), (0.25, REVIEW)]):
        for col, (arm, probability) in enumerate([("control", 0), ("cutmix", 0.3)]):
            name = "baseline" if scale == 0.5 and arm == "control" else arm
            frames = pd.read_csv(folder / f"{name}_frames.csv")
            sample = frames[(frames.video_id == video_id) & (frames.frame_index == frame_index)
                            & (frames.method == "Past link")].squeeze()
            gt = [sample.gt_x1, sample.gt_y1, sample.gt_x2, sample.gt_y2]
            if not all(pd.notna(gt)):
                raise ValueError("이 프레임에는 수동 원판 박스가 없습니다.")
            x, y, right, bottom = gt
            width, height = right - x, bottom - y
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="lime", linewidth=2))
            if pd.notna(sample.x1):
                ax.add_patch(Rectangle((sample.x1, sample.y1), sample.x2 - sample.x1,
                                       sample.y2 - sample.y1, fill=False, edgecolor="darkorange",
                                       linestyle="--", linewidth=2))
                result = f"IoU {sample.iou:.3f}"
            else:
                result = "no box"
            ax.set(xlim=(max(0, x-width), min(rgb.shape[1], right+width)),
                   ylim=(min(rgb.shape[0], bottom+height), max(0, y-height)),
                   title=f"scale {scale}, CutMix {probability} | {result}")
            ax.axis("off")
    fig.suptitle(f"{video_id} / frame {frame_index}")
    return fig


def check_backgrounds():
    """기존 학습에서 본 빈 배경 b05·b06만 사용해 새 두 모델의 오탐을 별도로 센다."""
    from ultralytics import YOLO

    from wlcv.data.plate_labels import read_boxes
    from wlcv.evaluation.full_video import predict_plate_videos
    from wlcv.evaluation.plate_track import select_centers

    videos = ["b05_sn_f_01", "b06_sn_f_01"]
    for video in videos:
        labels = list((DATASET / "labels/train").glob(f"{video}_f*.txt"))
        if not labels or any(read_boxes(path) for path in labels):
            raise ValueError(f"{video} 학습 라벨이 비어 있는지 확인하세요.")
    rows = []
    for arm, prob in [("control", 0.0), ("cutmix", 0.3)]:
        weights = CURRENT / "runs" / arm / "weights/best.pt"
        model = YOLO(str(weights.resolve(strict=True)))
        frames, candidates = predict_plate_videos(model, videos)
        track = select_centers(frames, candidates, show_progress=False)
        for (video, method), group in track.groupby(["video_id", "method"]):
            present = group.sort_values("frame_index").x1.notna()
            longest = int(present.groupby((~present).cumsum()).sum().max())
            rows.append((0.25, prob, video, method, len(group), int(present.sum()), longest))
    counts = pd.DataFrame(rows, columns=["scale", "cutmix", "video", "method", "frames",
                                         "false_positive_frames", "longest_run"])
    counts.to_csv(REVIEW / "training_backgrounds.csv", index=False)
    return counts
