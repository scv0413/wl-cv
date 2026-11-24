"""저장된 두 모델로 검증 영상 한 편을 분석한다. 프레임 번호는 0부터 센다."""

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from wlcv.config import ANNOTATIONS, ROOT, RUNS, VIDEOS
from wlcv.data.frames import timed_frames_rgb
from wlcv.evaluation.full_video import detect_videos
from wlcv.evaluation.plate_track import select_centers
from wlcv.evaluation.plots import predict_rgb
from wlcv.evaluation.video_overlay import PLATE_STATES, draw_caption, draw_plate, draw_pose


def read_phases(video):
    file = ANNOTATIONS / "trunk_inclination" / "frames" / f"{video}.csv"
    rows = pd.read_csv(file).to_dict("records")
    if len(rows) != 1 or rows[0]["판독"] != "확인":
        raise ValueError(f"확인된 이륙·무릎 통과 라벨 한 쌍이 필요합니다: {file}")
    return int(rows[0]["이륙"]), int(rows[0]["무릎 통과"])


def phase_summary(records, lift, knee):
    """원본의 y는 아래로 증가하므로 이륙 y에서 무릎 통과 y를 뺀다."""
    interval = records.set_index("frame_index").loc[lift:knee]
    first, last = interval.loc[lift], interval.loc[knee]
    missing = [int(i) for i in [lift, knee] if pd.isna(interval.at[i, "plate_cy_px"])]
    rise = None if missing else round(float(first.plate_cy_px - last.plate_cy_px), 3)
    return {
        "lift_off_frame": lift, "knee_pass_frame": knee,
        "lift_off_y_px": None if lift in missing else float(first.plate_cy_px),
        "knee_pass_y_px": None if knee in missing else float(last.plate_cy_px),
        "rise_px": rise, "missing_endpoint_frames": missing,
        "observed_frames": int(interval.plate_cy_px.count()), "interval_frames": len(interval),
    }


def render(path, plates, info, fps, pose_model, lift, knee, output):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps,
                             (int(info.width), int(info.height)))
    if not writer.isOpened():
        raise OSError("H.264 영상을 저장할 수 없습니다. OpenCV의 avc1 지원을 확인하세요.")
    records = []
    trail = deque(maxlen=12)
    events = {lift: ("이륙", "lift-off"), knee: ("무릎 통과", "knee pass")}
    try:
        pairs = zip(timed_frames_rgb(path), plates.itertuples(index=False), strict=True)
        for index, ((time_ms, rgb), plate) in enumerate(tqdm(pairs, total=len(plates), desc="자세와 오버레이", unit="장")):
            if index != plate.frame_index or rgb.shape[:2] != (info.height, info.width):
                raise ValueError(f"{index}번 원본의 크기 또는 프레임 번호를 확인하세요.")
            result = predict_rgb(pose_model, rgb, conf=0.25, imgsz=640, device="cpu").cpu().numpy()
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pose = draw_pose(image, result, info.near_side)
            draw_plate(image, plate, trail)
            phase, caption = events.get(index, ("", ""))
            draw_caption(image, plate, pose, caption)
            writer.write(image)
            records.append({
                "video_id": plate.video_id, "frame_index": index, "source_time_ms": time_ms,
                "manual_phase": phase, "plate_status": PLATE_STATES[plate.source][0],
                "plate_confidence": plate.confidence, "plate_cx_px": plate.cx, "plate_cy_px": plate.cy,
                **pose,
            })
    finally:
        writer.release()
    return pd.DataFrame(records)


def analyze(video):
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    if video not in metadata.index:
        raise ValueError(f"등록된 검증 영상 ID를 입력하세요: {video}")
    info = metadata.loc[video]
    timing = pd.read_csv(ROOT / "data/metadata/video_timing.csv").set_index("video_id")
    fps = float(timing.at[video, "fps_from_median_gap"])
    lift, knee = read_phases(video)
    if not 0 <= lift < knee < info.frame_count or info.near_side not in ("left", "right") or not fps > 0:
        raise ValueError("수동 시점 순서, 가까운 쪽과 재생 fps를 확인하세요.")

    plate_weights = RUNS / "plate_baseline/weights/best.pt"
    pose_weights = (ROOT / "models/yolo26n-pose.pt").resolve(strict=True)
    frames, candidates = detect_videos(plate_weights, [video])
    selected = select_centers(frames, candidates, distance_ratio=0.5)
    plates = selected.loc[selected.method == "Past link"]
    pose_model = YOLO(str(pose_weights))
    if pose_model.task != "pose":
        raise ValueError(f"자세 추정 가중치를 확인하세요: {pose_weights}")

    parent = ROOT / "outputs/video_analysis" / video
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # 한 영상의 세 결과가 모두 준비되면 공개한다. 실패해도 지난 실행은 다시 열 수 있다.
    with TemporaryDirectory(dir=parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        overlay = folder / "overlay.mp4"
        rows = render(VIDEOS / info.filename, plates, info, fps, pose_model, lift, knee, overlay)
        check = cv2.VideoCapture(str(overlay))
        try:
            if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                raise OSError("오버레이와 프레임 기록 수가 일치하지 않습니다.")
        finally:
            check.release()
        rows.to_csv(folder / "frames.csv", index=False)
        counts = {name: int(rows.plate_status.eq(name).sum()) for name, _, _ in PLATE_STATES.values()}
        summary = {"video_id": video, "frame_count": len(rows), "plate_counts": counts,
                   "overlay_fps": fps, "near_side": info.near_side,
                   "plate_weights": str(plate_weights), "pose_weights": str(pose_weights),
                   "device": "cpu", "manual_phase_interval": phase_summary(rows, lift, knee)}
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        folder.rename(destination)
    return destination, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="검증 영상 ID (예: c04_sn_f_01)")
    output, summary = analyze(parser.parse_args().video_id)
    interval = summary["manual_phase_interval"]
    print(f"전체 {summary['frame_count']}프레임 · {summary['plate_counts']}")
    if interval["rise_px"] is None:
        print(f"상승량 계산 보류 · 원판 미탐 프레임 {interval['missing_endpoint_frames']}")
    else:
        print(f"수동 이륙 {interval['lift_off_frame']} → 무릎 통과 {interval['knee_pass_frame']}: "
              f"원판 중심 {interval['rise_px']:+.2f} px (위쪽이 +)")
    print(f"로컬 결과: {output}")


if __name__ == "__main__":
    main()
