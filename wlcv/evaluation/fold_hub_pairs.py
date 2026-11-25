"""원판 출력 누락과 수동 박스 위치 미매칭을 분리해 판독 대상을 고른다."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from wlcv.config import ANNOTATIONS, ROOT
from wlcv.data.plate_labels import read_boxes
from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.plate_track import BOX, KEY


def reference_boxes(frames):
    """현재 원판 라벨을 우선하고, 다른 판독 출처는 겹치지 않는 프레임만 보충한다."""
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    rows = []
    for video in frames.video_id.unique():
        w, h = meta.loc[video, ["width", "height"]]
        for folder, suffix in [("personal_plate_boxes", "_frame_"), ("plate_boxes", "_f"),
                               ("reference_plate_boxes", "_frame_")]:
            for path in sorted((ANNOTATIONS / folder).glob(f"{video}{suffix}*.txt")):
                frame = int(path.stem.split(suffix)[-1])
                boxes = read_boxes(path)
                if not boxes:
                    # 명시적으로 비어 있는 최신 판독에 예전 양성 박스를 덧씌우지 않는다.
                    rows.append([video, frame, folder, *([np.nan] * 4)])
                    continue
                (x, y, bw, bh), = boxes
                rows.append([video, frame, folder, (x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
    legacy = pd.read_csv(ANNOTATIONS / "reference_frame_boxes.csv")
    for row in legacy[legacy.plate_present.eq("yes") & legacy.video_id.isin(frames.video_id)].itertuples():
        x, y = row.plate_bbox_x, row.plate_bbox_y
        rows.append([row.video_id, row.frame_index, "reference_frame_boxes", x, y,
                     x+row.plate_bbox_width, y+row.plate_bbox_height])
    return pd.DataFrame(rows, columns=KEY+["box_source"]+["gt_"+k for k in BOX]).drop_duplicates(KEY)


def classify(frames):
    scored = frames.merge(reference_boxes(frames), on=KEY, how="left", validate="one_to_one")
    scored["iou"] = np.nan
    for i, row in scored[scored.gt_x1.notna()].iterrows():
        scored.at[i, "iou"] = iou(row[BOX].to_numpy(float), row[["gt_"+k for k in BOX]].to_numpy(float))
    scored["reason"] = scored.source.map({"linked":"linked", "missing":"no_output", "high":"direct"})
    scored.loc[scored.iou.lt(.50), "reason"] = "box_mismatch"
    return scored


def make_pairs(scored):
    """같은 영상·같은 수동 구간 ±5장, 직접 신뢰도 ≥0.50. 대조는 한 번만 사용한다."""
    has_target_or_unlabeled = scored.gt_x1.notna() | scored.box_source.isna()
    active = scored[scored.phase.str.contains("-") & has_target_or_unlabeled]
    pairs = []
    for (video, phase), segment in active.groupby(["video_id", "phase"]):
        cases = segment[segment.reason.ne("direct")].sort_values("frame_index")
        controls = segment[segment.reason.eq("direct") & segment.confidence.ge(.50)].sort_values("frame_index")
        controls = controls.frame_index.to_numpy()
        indices = cases.frame_index.to_numpy()
        if not len(indices):
            continue
        # 먼저 가능한 짝의 수를 최대화한 뒤 프레임 간격 합이 가장 작은 짝을 선택한다.
        distance = np.abs(indices[:, None] - controls)
        unpaired = (len(indices)+1)*6
        cost = np.where(distance <= 5, distance, 2*unpaired)
        cost = np.column_stack([cost, np.full((len(indices), len(indices)), unpaired)])
        case_rows, matches = linear_sum_assignment(cost)
        for i, j in zip(case_rows, matches):
            ref = int(controls[j]) if j < len(controls) else None
            pairs.append(dict(video_id=video, phase=phase, case_frame=int(indices[i]),
                              reason=cases.iloc[i].reason, control_frame=ref))
    return pd.DataFrame(pairs, columns=["video_id", "phase", "case_frame", "reason", "control_frame"])


def prepare_queue(folder):
    frames = pd.read_csv(folder / "frames.csv")
    scored = classify(frames)
    pairs = make_pairs(scored)
    paired = pairs.dropna(subset=["control_frame"])
    ids = pd.concat([paired[["video_id", col]].rename(columns={col:"frame_index"})
                     for col in ["case_frame","control_frame"]], ignore_index=True)
    ids["frame_index"] = ids.frame_index.astype(int)
    queue = ids.merge(frames[KEY+["fold","person_id"]], on=KEY, validate="one_to_one")
    queue = queue.sample(frac=1, random_state=53).reset_index(drop=True)
    scored.to_csv(folder / "screen.csv", index=False)
    pairs.to_csv(folder / "pairs.csv", index=False)
    queue.to_csv(folder / "blind_queue.csv", index=False)
    return queue
