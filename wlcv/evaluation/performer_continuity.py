"""같은 YOLO26n-pose 후보를 두 방식으로 선택한다. 모든 번호는 원본 프레임 기준이다."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import events_for, load_events
from wlcv.config import ANNOTATIONS, ROOT, VIDEOS
from wlcv.data.frames import frames_rgb, read_rgb
from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.trunk import vertical_angle

PARENT = ROOT / "outputs/performer_selection"
REFERENCE = ANNOTATIONS / "performer_selection"
BOX = ["x1", "y1", "x2", "y2"]
PRIMARY = ("b02_sn_h_02", "c04_sn_f_01")
OTHERS = ("a03_sn_f_01", "a05_sn_f_01", "b03_sn_f_01", "c08_sn_f_01")


def select(boxes, previous):
    """직전 프레임에서 선택한 박스와 IoU ≥ 0.50인 후보가 하나일 때만 다인 연결한다."""
    if len(boxes) == 1:
        return 0
    if previous is None:
        return None
    matches = [i for i, box in enumerate(boxes) if iou(box, previous) >= .50]
    return matches[0] if len(matches) == 1 else None


def analyze(videos):
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    weights = ROOT / "models/yolo26n-pose.pt"
    model = YOLO(str(weights.resolve(strict=True)))
    frames, candidates = [], []
    for video in videos:
        info = meta.loc[video]
        events = load_events(ANNOTATIONS, video, 1)
        start, end = events["ready"], events[events_for(video)[-1]]
        previous = None
        stream = frames_rgb(VIDEOS / info.filename)
        count = 0
        for i, image in enumerate(tqdm(stream, total=info.frame_count, desc=video)):
            result = predict_rgb(model, image, device="mps", imgsz=640, conf=.25).cpu().numpy()
            boxes = result.boxes.xyxy
            single = 0 if len(boxes) == 1 else None
            linked = select(boxes, previous)
            angle = np.nan
            shown = 0
            for n, box in enumerate(boxes):
                candidates.append(dict(video_id=video, frame_index=i, candidate=n+1,
                                       previous_iou=np.nan if previous is None else iou(box, previous),
                                       **dict(zip(BOX, box))))
            if linked is not None:
                points = joint_table(result, linked+1, .50)
                near = points[points["방향"].eq(info.near_side) & points["표시 상태"].eq("표시")].set_index("관절")
                shown = len(near)
                if {"어깨", "골반"}.issubset(near.index):
                    angle = vertical_angle(near.loc["어깨", ["x (px)", "y (px)"]].to_numpy(),
                                           near.loc["골반", ["x (px)", "y (px)"]].to_numpy())
            frames.append(dict(video_id=video, frame_index=i, in_motion=start <= i <= end, count=len(boxes),
                               single=None if single is None else single+1, linked=None if linked is None else linked+1,
                               single_angle=angle if single is not None else np.nan, linked_angle=angle,
                               linked_joints=shown))
            previous = None if linked is None else boxes[linked].copy()
            count += 1
        if count != info.frame_count:
            raise ValueError(f"{video}: 프레임 수 불일치")
    PARENT.mkdir(parents=True, exist_ok=True)
    folder = PARENT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with TemporaryDirectory(dir=PARENT) as staging:
        tmp = Path(staging) / "complete"
        tmp.mkdir()
        pd.DataFrame(frames).to_csv(tmp / "frames.csv", index=False)
        pd.DataFrame(candidates).to_csv(tmp / "boxes.csv", index=False)
        (tmp / "run.json").write_text(json.dumps(dict(videos=list(videos), device="mps", imgsz=640,
            pose_weights=weights.name, sha256=sha256(weights.read_bytes()).hexdigest(), previous_iou=.50), indent=2))
        tmp.rename(folder)
    return folder


def score(folder, reference):
    """원본 판독 박스와 유일하게 IoU 0.50을 넘긴 후보를 정답으로 대응시킨다."""
    ref = pd.read_csv(reference)
    frames = pd.read_csv(folder / "frames.csv").set_index(["video_id", "frame_index"], verify_integrity=True)
    boxes = pd.read_csv(folder / "boxes.csv")
    rows = []
    for point in ref.itertuples(index=False):
        frame = frames.loc[point.video_id, point.frame_index]
        at = boxes[boxes.video_id.eq(point.video_id) & boxes.frame_index.eq(point.frame_index)]
        truth = np.array([getattr(point, key) for key in BOX])
        matched = [c.candidate for c in at.itertuples() if iou(np.array([c.x1,c.y1,c.x2,c.y2]), truth) >= .50]
        row = dict(video_id=point.video_id, frame_index=point.frame_index, matches=len(matched))
        for name in ("single","linked"):
            chosen = frame[name]
            if pd.isna(chosen):
                status = "보류"
            elif len(matched) > 1:
                status = "대응 모호"
            else:
                status = "정선택" if chosen in matched else "오선택"
            row[name] = status
        rows.append(row)
    return pd.DataFrame(rows)


def plot_review(folder, reference):
    refs = pd.read_csv(reference)
    frames = pd.read_csv(folder / "frames.csv").set_index(["video_id", "frame_index"])
    boxes = pd.read_csv(folder / "boxes.csv")
    fig, axes = plt.subplots(1, len(refs), figsize=(2.7*len(refs), 4.5), squeeze=False, layout="constrained")
    for ax, row in zip(axes[0], refs.itertuples()):
        ax.imshow(read_rgb(VIDEOS / f"{row.video_id}.mp4", row.frame_index))
        selected = frames.loc[(row.video_id,row.frame_index), "linked"]
        for p in boxes[boxes.video_id.eq(row.video_id) & boxes.frame_index.eq(row.frame_index)].itertuples():
            color = "#11dfbf" if p.candidate == selected else "#bd64ff"
            ax.add_patch(Rectangle((p.x1,p.y1),p.x2-p.x1,p.y2-p.y1,fill=False,color=color,lw=2))
        ax.add_patch(Rectangle((row.x1,row.y1),row.x2-row.x1,row.y2-row.y1,fill=False,
                               edgecolor="white",ls="--",lw=1.3))
        ax.set_title(f"{row.video_id[:3]} · {row.frame_index}")
        ax.axis("off")
    fig.suptitle("White: original-only reference  |  Green: selected  |  Purple: other")
    return fig
