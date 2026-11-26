"""같은 궤적 규칙으로 Mac MPS에서 영상 열기부터 H.264 저장 완료까지 비교한다."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.analyze import read_phases, render_frames
from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT, VIDEOS
from wlcv.data.frames import timed_frames_rgb
from wlcv.evaluation.edge_continuity import choose_plate
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.evaluation.plate_edge_filter import WEIGHTS
from wlcv.evaluation.plate_track import BOX, PlateHistory

OUTPUT = ROOT / "outputs/plate_fallback_timing"
VIDEOS_TO_TIME = ("a03_sn_f_01", "c04_sn_f_01")


def render_run(video, mode, folder):
    if video not in VIDEOS_TO_TIME or mode not in ("single", "dual"):
        raise ValueError("측정 영상과 모델 조건을 확인하세요.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("이 실험은 Mac MPS에서 실행합니다.")
    folder.mkdir(parents=True, exist_ok=False)
    info = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id").loc[video]
    fps = pd.read_csv(ROOT / "data/metadata/video_timing.csv").set_index("video_id").at[video,"fps_from_median_gap"]
    events = load_events(ANNOTATIONS,video,pass_number=1)
    lift,knee = read_phases(video)
    start = perf_counter()
    names = ["mac","cutmix"] if mode == "dual" else ["mac"]
    models = {name:validation_plate_model(WEIGHTS[name],[video]) for name in names}
    histories = {name:PlateHistory() for name in names}
    pose = YOLO(str((ROOT / "models/yolo26n-pose.pt").resolve(strict=True)))
    used = []
    options = dict(imgsz=640, device="mps", verbose=False)

    def stream():
        for index,(time_ms,rgb) in enumerate(timed_frames_rgb(VIDEOS / info.filename)):
            image = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            chosen, origin, selected_name = None,"missing",""
            for name,model in models.items():
                # 보조 모델도 매 프레임을 보고 독립적인 과거 궤적을 유지한다.
                result, = model.predict(image,conf=.001,**options)
                boxes = result.boxes.cpu()
                pool = pd.DataFrame(boxes.xyxy.tolist(),columns=BOX)
                pool["confidence"] = boxes.conf.tolist()
                pool["video_id"],pool["frame_index"] = video,index
                candidate,source = choose_plate(histories[name],pool,video,index,info.width,info.height,"continuity")
                if chosen is None and candidate is not None:
                    chosen,origin,selected_name = candidate,source,name
            used.append(selected_name)
            values = {k: np.nan if chosen is None else chosen[k] for k in ["confidence","cx","cy",*BOX]}
            plate = SimpleNamespace(video_id=video,frame_index=index,source=origin,**values)
            posture, = pose.predict(image,conf=.25,**options)
            yield time_ms,image,plate,posture.cpu().numpy()

    video_start = perf_counter()
    rows = render_frames(stream(),info,fps,lift,knee,folder/"overlay.mp4",events,{})
    saved = perf_counter()  # writer.release()까지 완료
    rows["selected_model"] = used
    rows.to_csv(folder/"frames.csv",index=False)
    timing = dict(video=video,mode=mode,frames=len(rows),overlay_s=saved-video_start,
                  with_model_load_s=saved-start, fps=float(fps))
    (folder/"timing.json").write_text(json.dumps(timing,indent=2))


def benchmark():
    session = OUTPUT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    session.mkdir(parents=True)
    runs = []
    env = {**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0"}
    for block in tqdm(range(4),desc="교차 순서로 영상 저장",unit="회"):
        for vi,video in enumerate(VIDEOS_TO_TIME):
            modes = ["single","dual"] if (block+vi)%2 == 0 else ["dual","single"]
            for mode in modes:
                folder = session / f"{video}_{block+1}_{mode}"
                command = [sys.executable,"-m","wlcv.evaluation.plate_fallback_timing","--one",video,mode,str(folder)]
                start = perf_counter()
                with (session / f"{folder.name}.log").open("w") as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
                wall = perf_counter()-start
                item=json.loads((folder/"timing.json").read_text())
                runs.append(dict(**item,block=block+1,command_s=wall,folder=folder.name))
                pd.DataFrame(runs).to_csv(session/"times.csv",index=False)
    return session


def verify_runs(session):
    """측정이 모두 끝난 뒤 MP4 전체를 디코딩하고 같은 수동 프레임에서 정오를 비교한다."""
    from wlcv.evaluation.box_size_review import iou
    from wlcv.evaluation.plate_edge_filter import OUTPUT as INPUTS

    times = pd.read_csv(session / "times.csv")
    if len(times) != 16 or not times.groupby(["video","mode"]).size().eq(4).all():
        raise ValueError("두 영상의 조건별 4회 측정이 완료돼야 합니다.")
    truth = pd.read_csv(INPUTS / "paired.csv").query("method == 'Past link' and in_motion and gt_x1_before == gt_x1_before")
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    results, checks = [], []
    for (video, block), runs in times.groupby(["video","block"]):
        hand = truth[truth.video_id.eq(video)].set_index("frame_index")
        truth_boxes = hand[[f"gt_{c}_before" for c in BOX]].to_numpy()
        pair = {}
        for r in runs.itertuples():
            path = session / r.folder
            frames = pd.read_csv(path / "frames.csv").set_index("frame_index",verify_integrity=True)
            capture = cv2.VideoCapture(str(path / "overlay.mp4"))
            count = 0
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4,"little").decode("ascii").lower()
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    if image.shape[:2] != (meta.at[video,"height"],meta.at[video,"width"]):
                        raise ValueError("원본 해상도와 다릅니다.")
                    count += 1
            finally:
                capture.release()
            if (count != r.frames or count != meta.at[video,"frame_count"]
                    or frames.index.tolist() != list(range(count)) or abs(fps-r.fps)>.02
                    or codec not in ("avc1","h264")):
                raise ValueError(f"완성 영상·CSV 검수 실패: {path}")
            pred = frames.loc[hand.index,[f"plate_{c}" for c in BOX]].to_numpy()
            overlap = [iou(box,target) for box,target in zip(pred,truth_boxes)]
            pair[r.mode] = pd.Series(overlap,index=hand.index).ge(.5)
            checks.append(dict(video=video,block=block,mode=r.mode,decoded=count,codec=codec,fps=fps,
                               scored=len(hand),hits=int(pair[r.mode].sum()),
                               missing_outputs=int(frames.plate_x1.isna().sum()),
                               aux_selected=int(frames.selected_model.eq("cutmix").sum())))
        single, dual = pair["single"],pair["dual"]
        for i in hand.index:
            results.append(dict(video=video,block=block,frame=int(i),single=bool(single.at[i]),dual=bool(dual.at[i])))
    pd.DataFrame(results).to_csv(session/"paired_quality.csv",index=False)
    checked = pd.DataFrame(checks)
    checked.to_csv(session/"checks.csv",index=False)
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", nargs=3,metavar=("VIDEO","MODE","FOLDER"))
    args=parser.parse_args()
    if args.one:
        render_run(args.one[0],args.one[1],Path(args.one[2]))
    else:
        print(benchmark())
