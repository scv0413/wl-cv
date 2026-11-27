"""같은 원본 후보를 다섯 번 재생해 전경 선수와 원판 선택 기준을 검토한다."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT, VIDEOS
from wlcv.evaluation.schedule_review import review_inputs
from wlcv.runtime.foreground import performer_plates, select_performer
from wlcv.runtime.inference import load_plate_model, plate_candidate_frames, track_candidates
from wlcv.runtime.models import file_models
from wlcv.runtime.tracking import BOX, PlateHistory, iou

CONTROL = "control"
CONDITIONS = {
    CONTROL: (1.25, .15),
    "area_110": (1.10, .15),
    "area_150": (1.50, .15),
    "margin_0": (1.25, 0.0),
    "margin_10": (1.25, .10),
}
KEY = ["video_id", "frame_index"]


def collect(folder):
    """검증에 사용했던 영상별로 사람·원판 후보 좌표만 저장한다. 모델 입력은 원본이다."""
    videos, _, _ = review_inputs(DATASET / "data.yaml")
    if len(videos) != 6:
        raise ValueError("기존 검증 6명의 YOLO 데이터셋과 영상 경로를 확인하세요.")
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    weights = file_models()
    primary, helper = load_plate_model(weights.plate), load_plate_model(weights.cutmix)
    pose = YOLO(str(weights.pose), task="pose")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for video in tqdm(videos, desc="검증 영상 후보 저장"):
        path = VIDEOS / metadata.at[video, "filename"]
        stream = plate_candidate_frames(path, primary, "cpu", {"plate_inference": 0.}, helper, 640, False)
        frames = []
        for index, (_, bgr, tables) in enumerate(stream):
            result, = pose.predict(bgr, imgsz=640, conf=.25, device="cpu", verbose=False)
            frames.append(dict(frame_index=index, shape=bgr.shape[:2], people=result.boxes.xyxy.tolist(),
                               plates={k: t[BOX + ["confidence"]].values.tolist() for k, t in tables.items()}))
        if len(frames) != int(metadata.at[video, "frame_count"]):
            raise ValueError(f"{video}의 디코딩 장수를 확인하세요.")
        (folder / f"{video}.json").write_text(json.dumps(frames))
    return videos


def replay(video, frames, area_ratio, margin, plate_truth, people_truth):
    """조건마다 빈 이력에서 시작한다. 미선택과 사람 박스의 판독 모호성을 구분한다."""
    events = load_events(ANNOTATIONS, video, 1)
    start, end = events["ready"], max(events.values())
    previous = None
    histories = {"baseline": PlateHistory(), "cutmix": PlateHistory()}
    records = []
    for index, frame in enumerate(frames):
        if index != frame["frame_index"]:
            raise ValueError(f"{video}의 프레임 순서를 확인하세요.")
        people = np.asarray(frame["people"], dtype=np.float32).reshape(-1, 4)
        selected, reason = select_performer(people, previous, area_ratio=area_ratio)
        previous = None if selected is None else people[selected].copy()
        kept, pool = {}, []
        for model, boxes in frame["plates"].items():
            table = pd.DataFrame(boxes, columns=BOX + ["confidence"])
            table["video_id"], table["frame_index"] = video, index
            kept[model] = performer_plates(table, people, selected, margin=margin)
            pool.extend(f"{model}:{i}" for i in kept[model].index)
        # 사람과의 위치 조건을 적용한 후보로만 두 모델의 시간 이력을 갱신한다.
        plate = track_candidates(kept, histories, video, index, frame["shape"], "continuity")
        phase = "before" if index < start else "after" if index > end else "motion"
        record = dict(video_id=video, frame_index=index, phase=phase, people=len(people),
                      selected=0 if selected is None else int(selected + 1), reason=reason,
                      pool=" ".join(pool), source=plate.source, model=plate.selected_model,
                      plate_conf=plate.confidence, **{k: getattr(plate, k) for k in BOX})
        key = video, index
        if key in people_truth.index:
            truth = people_truth.loc[key, BOX].to_numpy(float)
            matches = [i for i, box in enumerate(people) if iou(box, truth) >= .50]
            if selected is None:
                verdict = "보류"
            elif not matches:
                verdict = "대응 없음"
            elif selected not in matches:
                verdict = "오선택"
            else:
                verdict = "유일 일치" if len(matches) == 1 else "복수 일치"
            record["person_review"] = verdict
            record["person_iou"] = np.nan if previous is None else iou(previous, truth)
        if key in plate_truth.index:
            truth = plate_truth.loc[key, BOX].to_numpy(float)
            overlap = iou(np.array([record[k] for k in BOX]), truth)
            record["plate_iou"] = overlap
            record["plate_fn"] = not (overlap >= .50)
            record["plate_fp"] = bool(overlap < .50)  # 빈 예측은 미탐만, 잘못 놓인 박스는 둘 다
        records.append(record)
    return records


def compare(folder):
    folder = Path(folder)
    videos, truth, _ = review_inputs(DATASET / "data.yaml")
    plate_truth = truth.set_index(KEY, verify_integrity=True)
    people_truth = pd.concat(pd.read_csv(file) for file in
                             sorted((ANNOTATIONS / "performer_selection").glob("*.csv")))
    people_truth = people_truth.set_index(KEY, verify_integrity=True)
    records = []
    for video in tqdm(videos, desc="조건별 시간 연결"):
        frames = json.loads((folder / "candidates" / f"{video}.json").read_text())
        for name, (ratio, margin) in CONDITIONS.items():
            run = replay(video, frames, ratio, margin, plate_truth, people_truth)
            records.extend(dict(condition=name, **row) for row in run)
    details = pd.DataFrame(records)
    control = details[details.condition == CONTROL].set_index(KEY)
    pairs = details.join(control[["selected", "reason", "pool", "source", "model", "plate_conf", *BOX]],
                         on=KEY, rsuffix="_control", validate="many_to_one")
    pairs["person_changed"] = pairs.selected != pairs.selected_control
    pairs["reason_changed"] = pairs.reason != pairs.reason_control
    pairs["pool_changed"] = pairs.pool != pairs.pool_control
    pairs["plate_changed"] = (pairs.source != pairs.source_control) | (pairs.model != pairs.model_control)
    for col in [*BOX, "plate_conf"]:
        pairs["plate_changed"] |= ~np.isclose(pairs[col], pairs[col + "_control"], rtol=0, atol=0, equal_nan=True)
    pairs["output_changed"] = pairs.person_changed | pairs.plate_changed
    summaries = []
    for (name, phase), rows in pairs.groupby(["condition", "phase"], sort=False):
        review = rows.person_review
        summaries.append(dict(condition=name, phase=phase, frames=len(rows), multi=int(rows.people.ge(2).sum()),
                              person_changed=int(rows.person_changed.sum()), plate_changed=int(rows.plate_changed.sum()),
                              reason_changed=int(rows.reason_changed.sum()), pool_changed=int(rows.pool_changed.sum()),
                              multi_withheld=int((rows.people.ge(2) & rows.selected.eq(0)).sum()),
                              ref_unique=int(review.eq("유일 일치").sum()), ref_multiple=int(review.eq("복수 일치").sum()),
                              ref_wrong=int(review.eq("오선택").sum()), ref_withheld=int(review.eq("보류").sum()),
                              ref_unmatched=int(review.eq("대응 없음").sum()), plate_labeled=int(rows.plate_fn.count()),
                              fn=int(rows.plate_fn.sum()), fp=int(rows.plate_fp.sum())))
    summary = pd.DataFrame(summaries)
    pairs.to_csv(folder / "frames.csv", index=False)
    pairs[pairs.output_changed].to_csv(folder / "changed.csv", index=False)
    pairs[pairs.output_changed | pairs.reason_changed | pairs.pool_changed].to_csv(folder / "review.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    return summary, pairs


def plot_change(folder, pairs, video, index, condition, *, show_video=True):
    """대조군과 실험군의 선택 박스를 같은 원본 또는 빈 좌표 평면에서 비교한다."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from wlcv.data.frames import read_rgb

    frame = json.loads((Path(folder) / "candidates" / f"{video}.json").read_text())[index]
    rows = pairs[(pairs.video_id == video) & (pairs.frame_index == index)].set_index("condition")
    fig, axes = plt.subplots(1, 2, figsize=(6, 5), layout="constrained")
    height, width = frame["shape"]
    rgb = read_rgb(VIDEOS / f"{video}.mp4", index) if show_video else None
    for ax, name in zip(axes, [CONTROL, condition]):
        if rgb is not None:
            ax.imshow(rgb)
        row = rows.loc[name]
        for number, (x1, y1, x2, y2) in enumerate(frame["people"], 1):
            selected = number == row.selected
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                   ec="#007a62" if selected else "#9098a1",
                                   lw=3 if selected else 1, ls="-" if selected else "--"))
        if pd.notna(row.x1):
            ax.add_patch(Rectangle((row.x1, row.y1), row.x2-row.x1, row.y2-row.y1,
                                   fill=False, ec="#c66d00", lw=2))
        area, margin = CONDITIONS[name]
        ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal",
               title=f"Area {area:.2f}  |  Margin {margin:.0%}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{video}  /  frame {index}")
    fig.supxlabel("Green: selected person     Dashed: other candidate     Orange: plate", fontsize=9)
    return fig
