"""수행자 선택을 보류한 프레임에서만 원판 중심을 포함하는 유일한 사람 박스를 검토한다."""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.config import ANNOTATIONS, ROOT, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.pose import joint_table
from wlcv.evaluation.box_size_review import iou


def run(source, whole_phase=False):
    if whole_phase:
        from wlcv.annotation.storage import load_events
        tracks = pd.read_csv(source/"frames.csv")
        parts = []
        for video, track in tracks.groupby("video", sort=False):
            e = load_events(ANNOTATIONS,video,1)
            parts.append(track[track.frame.between(e["ready"],e["drive"])])
        frames = pd.concat(parts,ignore_index=True)
        frames["paired"] = frames[["plate_x","plate_y","knee_x","knee_y"]].notna().all(axis=1)
    else:
        from wlcv.evaluation.train_knee_pass_coverage import measure
        _, _, frames = measure(source)
    model = YOLO(str((ROOT/"models/yolo26n-pose.pt").resolve(strict=True)))
    rows = []
    for r in tqdm(frames.itertuples(),total=len(frames),desc="동일 원본",unit="장"):
        rgb = read_rgb(VIDEOS/f"{r.video}.mp4",r.frame)
        pose = predict_rgb(model,rgb,device="mps",conf=.25,imgsz=640).cpu().numpy()
        boxes = pose.boxes.xyxy
        if len(boxes)!=r.people:
            raise ValueError(f"{r.video} {r.frame}: 기존 사람 후보 수와 다릅니다.")
        chosen = None if pd.isna(r.person) else int(r.person)-1
        origin = "기존 선택" if chosen is not None else "보류"
        if chosen is None and r.plate_status=="high":
            inside=[i for i,(x1,y1,x2,y2) in enumerate(boxes)
                    if x1<=r.plate_x<=x2 and y1<=r.plate_y<=y2]
            if len(inside)==1:
                chosen=inside[0]
                origin="원판 중심"
        knee_x=knee_y=overlap=np.nan
        matched_box=None
        if chosen is not None:
            pts=joint_table(pose,chosen+1,.5)
            knee=pts[(pts["방향"]==r.side) & (pts["관절"]=="무릎")].iloc[0]
            if knee["표시 상태"]=="표시":
                knee_x,knee_y=knee["x (px)"],knee["y (px)"]
            if origin=="기존 선택" and not np.allclose([knee_x,knee_y],[r.knee_x,r.knee_y],atol=.1,equal_nan=True):
                raise ValueError(f"{r.video} {r.frame}: 기존 무릎 좌표 재현을 확인하세요.")
            # 이미 직접 그려 저장된 수행자 박스만 독립적으로 대조한다.
            files=list(ANNOTATIONS.glob(f"*/train/person_boxes/{r.video}_frame_{r.frame:04d}.csv"))
            files+=list((ANNOTATIONS/"train_pose_candidates/person_boxes").glob(f"{r.video}_frame_{r.frame:04d}.csv"))
            if files:
                refs=np.concatenate([pd.read_csv(f)[["xmin","ymin","xmax","ymax"]].to_numpy() for f in files])
                if not np.allclose(refs,refs[0]): raise ValueError("수동 박스 버전을 확인하세요.")
                overlap=iou(boxes[chosen],refs[0])
            matched_box=boxes[chosen]
        record=dict(video=r.video,frame=r.frame,origin=origin,original_paired=r.paired,
                    paired=bool(np.isfinite(r.plate_y) and np.isfinite(knee_y)),knee_x=knee_x,knee_y=knee_y,
                    reference_iou=overlap,selected_person=None if chosen is None else chosen+1)
        record.update(dict(zip(("x1","y1","x2","y2"), matched_box if matched_box is not None else [np.nan]*4)))
        rows.append(record)
    return pd.DataFrame(rows)

def compare_timing(source, output):
    """모든 선수를 동일한 ready~drive 구간에서 재처리하고 원판 좌표·라벨·학습 설정을 고정한다."""
    from wlcv.evaluation.knee_pass_tcn import run as fit_folds
    output.mkdir(parents=True, exist_ok=True)
    choices = run(source, whole_phase=True)
    choices.to_csv(output/"choices.csv",index=False)
    original = pd.read_csv(source/"frames.csv")
    feature_dir = output/"tcn_input"
    feature_dir.mkdir(exist_ok=True)
    features = original.merge(choices[["video","frame","origin","knee_x","knee_y"]],
                              on=["video","frame"],how="left",validate="one_to_one",suffixes=("","_new"))
    recovered = features.origin.eq("원판 중심")
    for col in ("knee_x","knee_y"):
        features.loc[recovered,col] = features.loc[recovered,col+"_new"]
    features["gap_px"] = features.knee_y-features.plate_y
    # 모델 입력 열만 보관해 보완 전의 수행자 상태가 새 진단으로 오인되는 것을 막는다.
    features[["video","frame","time_ms","knee_y","plate_y","gap_px"]].to_csv(feature_dir/"frames.csv",index=False)
    pd.read_csv(source/"readings.csv").to_csv(feature_dir/"readings.csv",index=False)
    return fit_folds(feature_dir, output_parent=output/"folds")
