"""기존 후보를 두 번 재생해 박스 포함 관계만 원판 필터에 추가한 결과를 비교한다."""
import argparse
import json
import sys
from itertools import combinations, groupby
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", type=Path, required=True, help="고정한 배포 코드 체크아웃")
    parser.add_argument("--case", action="store_true", help="가설이 나온 54번도 별도로 재생")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))

from wlcv.runtime.foreground import select_performer, performer_plates
from wlcv.runtime.inference import track_candidates
from wlcv.runtime.tracking import BOX, PlateHistory, iou

keys = ["video_id", "frame_index"]


def included_pairs(boxes):
    """교집합 / 작은 박스 면적. 작은 박스가 더 짧고 그 중심이 큰 박스의 위쪽에 있어야 한다."""
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    matches = []
    for i, j in combinations(range(len(boxes)), 2):
        small, large = (i, j) if areas[i] < areas[j] else (j, i)
        a, b = boxes[small], boxes[large]
        if areas[small] <= 0 or (a[3]-a[1]) >= (b[3]-b[1]) or a[1]+a[3] >= b[1]+b[3]:
            continue
        inside = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
        coverage = float(inside / areas[small])
        if coverage >= .90:
            matches.append((small, large, coverage))
    return matches


def group_members(people, selected, pairs):
    linked = {frozenset((a,b)) for a,b,_ in pairs}
    if selected is not None:
        # 연결된 후보를 타고 제3자까지 합치지 않는다.
        return sorted({index for pair in linked if selected in pair for index in pair})
    if pairs and all(frozenset((i,j)) in linked or iou(people[i],people[j]) >= .50
                     for i,j in combinations(range(len(people)),2)):
        return list(range(len(people)))
    return []


def replay(video, stream):
    previous = None
    histories = {mode:{name:PlateHistory() for name in ("baseline","cutmix")}
                 for mode in ("control","containment")}
    rows, reviews = [], []
    for index, frame in enumerate(stream):
        assert index == frame["frame_index"]
        people = np.asarray(frame["people"], dtype=np.float32).reshape(-1,4)
        selected, reason = select_performer(people, previous)
        previous = None if selected is None else people[selected].copy()
        pairs = included_pairs(people)
        members = group_members(people, selected, pairs)
        for small, large, coverage in pairs:
            reviews.append(dict(video_id=video, frame_index=index, small=small+1, large=large+1,
                                coverage=coverage, pair_iou=iou(people[small],people[large]),
                                used=small in members and large in members))
        for mode, history in histories.items():
            kept, pool = {}, []
            for name, packed in frame["plates"].items():
                t = pd.DataFrame(packed, columns=BOX+["confidence"])
                t["video_id"],t["frame_index"] = video,index
                keep = performer_plates(t,people,selected).index
                if mode == "containment":
                    for member in members:
                        keep = keep.union(performer_plates(t,people,member).index)
                kept[name] = t.loc[t.index.isin(keep)]
                pool.extend(f"{name}:{i}" for i in kept[name].index)
            plate = track_candidates(kept, history, video, index, frame["shape"], "continuity")
            rows.append(dict(video_id=video, frame_index=index, condition=mode, people=len(people),
                             person=0 if selected is None else selected+1, reason=reason,
                             pool=" ".join(pool), model=plate.selected_model, source=plate.source,
                             confidence=plate.confidence, **{k:getattr(plate,k) for k in BOX}))
    return rows, reviews


def longest_missing(rows):
    return max((sum(1 for _ in run) for missing,run in groupby(rows.x1.isna()) if missing),default=0)


def validation(project, output):
    from wlcv.evaluation.schedule_review import review_inputs

    prior = project / "outputs/foreground_thresholds/20261127-154950"
    previous = pd.read_csv(prior / "frames.csv").query("condition == 'control'").set_index(keys)
    videos, manual, _ = review_inputs(project / "outputs/dataset/data.yaml")
    assert set(videos) == set(previous.index.unique("video_id"))
    truth = manual.set_index(keys)[BOX].add_prefix("gt_")
    assert previous.index.is_unique and truth.index.is_unique
    assert truth.index.isin(previous.index).all()
    rows, reviews = [], []
    for video in previous.index.unique("video_id"):
        sequence = json.loads((prior/"candidates"/f"{video}.json").read_text())
        r, audit = replay(video,sequence)
        rows.extend(r)
        reviews.extend(audit)
    frames = pd.DataFrame(rows).join(previous[["phase"]], on=keys).join(truth, on=keys)
    frames["labeled"] = frames[["gt_"+k for k in BOX]].notna().all(axis=1)
    frames["iou"] = [iou(a,b) if np.isfinite(b).all() else np.nan for a,b in
                     zip(frames[BOX].to_numpy(),frames[["gt_"+k for k in BOX]].to_numpy())]
    frames["tp"] = frames.labeled & frames.iou.ge(.5)
    frames["fn"] = frames.labeled & ~frames.tp
    frames["fp"] = frames.labeled & frames.x1.notna() & ~frames.tp
    base = frames.query("condition == 'control'").set_index(keys).sort_index()
    saved = previous.loc[base.index]
    for current, old in [("person","selected"),("reason","reason"),("source","source"),
                         ("model","model"),("pool","pool")]:
        assert base[current].eq(saved[old].fillna("")).all(), current
    for current, old in [("confidence","plate_conf"),("iou","plate_iou")]+[(k,k) for k in BOX]:
        np.testing.assert_allclose(base[current],saved[old],rtol=0,atol=1e-5,equal_nan=True)
    assert base.query("phase=='motion'").labeled.sum()==270
    exp = frames.query("condition=='containment'").set_index(keys).loc[base.index]
    changes = base[["phase","people","labeled","person","reason","tp","fn","fp"]].copy()
    changes["pool_changed"] = base.pool.ne(exp.pool)
    changes["plate_changed"] = base.source.ne(exp.source) | base.model.ne(exp.model)
    for k in BOX+["confidence"]:
        changes["plate_changed"] |= ~np.isclose(base[k],exp[k],rtol=0,atol=1e-7,equal_nan=True)
    changes["recovered"] = base.fn & exp.tp
    changes["lost"] = base.tp & exp.fn
    frames.to_csv(output/"frames.csv",index=False)
    changes.to_csv(output/"frame_comparison.csv")
    audit = pd.DataFrame(reviews).join(changes[["phase","pool_changed","plate_changed"]],on=keys)
    audit.to_csv(output/"pairs.csv",index=False)
    metrics=[]
    for (mode,video,phase),group in frames.groupby(["condition","video_id","phase"],sort=False):
        metrics.append(dict(condition=mode,video_id=video,phase=phase,frames=len(group),
                            labeled=int(group.labeled.sum()),tp=int(group.tp.sum()),fn=int(group.fn.sum()),
                            fp=int(group.fp.sum()),missing=int(group.x1.isna().sum()),longest=longest_missing(group)))
    byvideo = pd.DataFrame(metrics)
    byvideo.to_csv(output/"by_video.csv",index=False)
    print(byvideo.groupby(["condition","phase"])[["frames","labeled","tp","fn","fp","missing"]].sum())
    print(changes.groupby("phase")[["pool_changed","plate_changed","recovered","lost"]].sum())
    print(audit.groupby("phase").used.agg(["count","sum"]))
    print("Missed motion samples:\n",base.query("phase=='motion' and fn")[["people","reason","source"]])


def origin_case(project, output):
    from ultralytics import YOLO
    from wlcv.runtime.inference import load_plate_model,plate_candidate_frames
    from wlcv.data.plate_labels import read_boxes
    video = "a07_sn_h_02"
    weights = project/"models/pytorch"
    plate,helper = [load_plate_model(weights/name) for name in ("plate_baseline.pt","plate_cutmix.pt")]
    pose = YOLO(str(weights/"yolo26n-pose.pt"))
    seq=[]
    path = project/"data/raw/videos"/f"{video}.mp4"
    for i,(_,image,tables) in enumerate(plate_candidate_frames(path,plate,"cpu",{"plate_inference":0.},helper,640,False)):
        if i>55:
            break
        result, = pose.predict(image,conf=.25,imgsz=640,device="cpu",verbose=False)
        seq.append(dict(frame_index=i,shape=image.shape[:2],people=result.boxes.xyxy.tolist(),
                        plates={k:t[BOX+["confidence"]].values.tolist() for k,t in tables.items()}))
    rows,_=replay(video,seq)
    out=pd.DataFrame(rows).query("frame_index>=53").copy()
    (cx,cy,bw,bh),=read_boxes(project/"data/annotations/plate_holdout/personal_plate_boxes"/f"{video}_frame_0054.txt")
    h,w=seq[54]["shape"]
    gt=np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])*[w,h,w,h]
    out["iou_54"]=[iou(b,gt) if t==54 else np.nan for t,b in zip(out.frame_index,out[BOX].values)]
    out.to_csv(output/"origin_case.csv",index=False)
    print(out[["frame_index","condition","reason","model","source","confidence","iou_54"]].to_string(index=False))


if __name__=="__main__":
    project = args.project.resolve()
    output = project / "outputs/performer_containment/20261127"
    output.mkdir(parents=True, exist_ok=True)
    validation(project, output)
    if args.case:
        origin_case(project, output)
