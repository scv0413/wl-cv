"""저장된 두 모델로 검증 영상 한 편을 분석한다. 프레임 번호는 0부터 센다."""

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import load_events
from wlcv.evaluation.phase_speed import vertical_speeds
from wlcv.evaluation.catch_offset import FACING
from wlcv.config import ANNOTATIONS, CUTMIX_WEIGHTS, ROOT, RUNS, VIDEOS
from wlcv.data.frames import timed_frames_rgb
from wlcv.evaluation.full_video import validation_plate_model
from wlcv.evaluation.single_pass import single_pass_frames, tracked_plate_frames
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


def coaching_summary(rows, lift, knee, events, facing):
    """원본 시각과 모델 관측으로 계산하며 측정할 수 없는 값은 JSON null로 남긴다."""
    frames = rows.set_index("frame_index", verify_integrity=True)
    result = {"units": "source px, source-video seconds", "landmarks": "model plate box center and near shoulder"}
    contact = events.get("drive")
    if contact in frames.index and lift < knee < contact:
        endpoints = frames.loc[[lift, knee, contact]]
        values = vertical_speeds(endpoints.source_time_ms, endpoints.plate_cy_px)
        result["pulls"] = {k: (float(v) if np.isfinite(v) else None) for k, v in values.items() if k != "reason"}
        result["pulls"].update(reason=values["reason"], endpoint_frames=[int(lift), int(knee), int(contact)],
                               plate_states=endpoints.plate_status.tolist())
    else:
        result["pulls"] = {"reason": "접촉 시점 미확인"}
    catch = events.get("catch")
    if catch not in frames.index:
        result["catch"] = {"reason": "캐치 시점 미확인"}
        return result
    point = frames.loc[catch]
    info = {"frame": int(catch), "forward_px": None, "up_px": None}
    if point[["plate_cx_px", "plate_cy_px", "shoulder_x_px", "shoulder_y_px"]].isna().any():
        info["reason"] = "원판 또는 단일 수행자의 가까운 어깨 관측 누락"
    else:
        info["up_px"] = float(point.shoulder_y_px - point.plate_cy_px)
        if facing in (-1, 1):
            info["forward_px"] = float(facing * (point.plate_cx_px - point.shoulder_x_px))
            info["reason"] = ""
        else:
            info["reason"] = "화면에서 선수가 향한 방향 미확인"
    result["catch"] = info
    return result


def two_pass_frames(path, plates, pose_model, device, timings):
    pairs = zip(timed_frames_rgb(path), plates.itertuples(index=False), strict=True)
    for (time_ms, rgb), plate in pairs:
        started = perf_counter()
        result = predict_rgb(pose_model, rgb, conf=0.25, imgsz=640, device=device).cpu().numpy()
        timings["pose"] += perf_counter() - started
        yield time_ms, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), plate, result


def render_frames(frame_stream, info, fps, lift, knee, output, phase_events, timings):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps,
                             (int(info.width), int(info.height)))
    if not writer.isOpened():
        raise OSError("H.264 영상을 저장할 수 없습니다. OpenCV의 avc1 지원을 확인하세요.")
    records = []
    draw_seconds = write_seconds = 0.0
    trail = deque(maxlen=12)
    events = {lift: ("이륙", "lift-off"), knee: ("무릎 통과", "knee pass")}
    for name, korean in [("drive", "접촉"), ("catch", "캐치")]:
        if name in (phase_events or {}):
            events[phase_events[name]] = (korean, name)
    try:
        for index, (time_ms, image, plate, result) in enumerate(
                tqdm(frame_stream, total=int(info.frame_count), desc="자세와 오버레이", unit="장")):
            if index != plate.frame_index or image.shape[:2] != (info.height, info.width):
                raise ValueError(f"{index}번 원본의 크기 또는 프레임 번호를 확인하세요.")
            drawing = perf_counter()
            pose = draw_pose(image, result, info.near_side)
            draw_plate(image, plate, trail)
            phase, caption = events.get(index, ("", ""))
            draw_caption(image, plate, pose, caption)
            drawn = perf_counter()
            draw_seconds += drawn - drawing
            writer.write(image)
            write_seconds += perf_counter() - drawn
            records.append({
                "video_id": plate.video_id, "frame_index": index, "source_time_ms": time_ms,
                "manual_phase": phase, "plate_status": PLATE_STATES[plate.source][0],
                "plate_confidence": plate.confidence, "plate_cx_px": plate.cx, "plate_cy_px": plate.cy,
                "plate_x1": plate.x1, "plate_y1": plate.y1,
                "plate_x2": plate.x2, "plate_y2": plate.y2, **pose,
            })
            if hasattr(plate, "selected_model"):
                records[-1]["selected_model"] = plate.selected_model
    finally:
        started = perf_counter()
        writer.release()  # 인코더에 남은 프레임과 MP4 컨테이너를 마무리한다.
        write_seconds += perf_counter() - started
    if len(records) != info.frame_count:
        raise ValueError("원본 디코딩 장수와 등록된 프레임 수가 다릅니다.")
    timings.update(drawing=draw_seconds, writer=write_seconds)
    return pd.DataFrame(records)


def render(path, plates, info, fps, pose_model, lift, knee, output, phase_events=None, *, device="cpu", timings=None):
    """기존 두 번 읽기 렌더링 호출도 동일한 그리기·인코딩 경로를 사용한다."""
    if timings is None:
        timings = {}
    timings["pose"] = 0.0
    stream = two_pass_frames(path, plates, pose_model, device, timings)
    return render_frames(stream, info, fps, lift, knee, output, phase_events, timings)


def analyze(video, *, device="cpu", read_mode="single-pass", plate_imgsz=640, plate_mode="single"):
    start = perf_counter()
    if plate_mode not in ("single", "cutmix"):
        raise ValueError("원판 분석 모드는 single 또는 cutmix입니다.")
    if read_mode not in ("two-pass", "single-pass", "single-shared"):
        raise ValueError("지원하지 않는 영상 읽기 방식입니다.")
    if plate_imgsz <= 0 or plate_imgsz % 32:
        raise ValueError("원판 입력 크기는 양의 32 배수여야 합니다.")
    if read_mode == "single-shared" and plate_imgsz != 640:
        raise ValueError("입력 텐서를 공유하려면 원판과 자세 크기가 모두 640이어야 합니다.")
    if device not in ("cpu", "mps"):
        raise ValueError("cpu 또는 mps 중 하나를 선택하세요.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("이 실행 환경에서 MPS를 사용할 수 없습니다.")
    metadata = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    if video not in metadata.index:
        raise ValueError(f"등록된 검증 영상 ID를 입력하세요: {video}")
    info = metadata.loc[video]
    timing = pd.read_csv(ROOT / "data/metadata/video_timing.csv").set_index("video_id")
    fps = float(timing.at[video, "fps_from_median_gap"])
    lift, knee = read_phases(video)
    if not 0 <= lift < knee < info.frame_count or info.near_side not in ("left", "right") or not fps > 0:
        raise ValueError("수동 시점 순서, 가까운 쪽과 재생 fps를 확인하세요.")

    events = load_events(ANNOTATIONS, video, pass_number=1)
    plate_weights = RUNS / "plate_baseline/weights/best.pt"
    pose_weights = (ROOT / "models/yolo26n-pose.pt").resolve(strict=True)
    configured = perf_counter()
    timings = {"setup": configured - start, "pose": 0.0}
    path = VIDEOS / info.filename
    plate_model = validation_plate_model(plate_weights, [video])
    cutmix_model = validation_plate_model(CUTMIX_WEIGHTS, [video]) if plate_mode == "cutmix" else None
    timings.update(plate_load=perf_counter()-configured, plate_inference=0.0, plate_tracking=0.0)
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
        video_start = perf_counter()
        if read_mode == "two-pass":
            observations = tracked_plate_frames(path, plate_model, device, timings,
                                                cutmix_model=cutmix_model, plate_imgsz=plate_imgsz)
            plates = pd.DataFrame.from_records(vars(plate) for _, _, plate in observations)
            # 기존 두 번 읽기 시간표의 plate_pass에는 모델 준비·읽기·추론이 들어간다.
            timings["plate_pass"] = timings.pop("plate_load") + perf_counter() - video_start - timings["plate_tracking"]
            del timings["plate_inference"]
            stream = two_pass_frames(path, plates, pose_model, device, timings)
        else:
            stream = single_pass_frames(path, plate_model, pose_model, device, read_mode == "single-shared",
                                        timings, plate_imgsz=plate_imgsz, cutmix_model=cutmix_model,
                                        edge_rule="continuity")
        rows = render_frames(stream, info, fps, lift, knee, overlay, events, timings)
        rendered = perf_counter()
        timings["render_other"] = rendered - start - sum(timings.values())
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
                   "plate_mode": plate_mode, "overlay_seconds": rendered - video_start,
                   "selected_model_counts": {name: int(rows.selected_model.eq(name).sum())
                                             for name in ("baseline", "cutmix", "none")},
                   "cutmix_weights": str(CUTMIX_WEIGHTS) if cutmix_model is not None else None,
                   "plate_weights": str(plate_weights), "pose_weights": str(pose_weights),
                   "coaching": coaching_summary(rows, lift, knee, events, FACING.get(video)),
                   "plate_imgsz": plate_imgsz, "pose_imgsz": 640,
                   "device": device, "read_mode": read_mode, "manual_phase_interval": phase_summary(rows, lift, knee)}
        endpoint_angles = rows.set_index("frame_index").loc[[lift, knee], "trunk_angle_deg"]
        summary["trunk_change_deg"] = (float(endpoint_angles.loc[knee] - endpoint_angles.loc[lift])
                                       if endpoint_angles.notna().all() else None)
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        saved = perf_counter()
        timings["verify_and_save"] = saved - rendered
        elapsed = saved - start
        timing = {"pipeline_seconds": elapsed, "frames_per_second": len(rows) / elapsed,
                  "stage_seconds": timings}
        (folder / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        folder.rename(destination)
    return destination, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="검증 영상 ID (예: c04_sn_f_01)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="원판·자세 추론 장치")
    parser.add_argument("--read-mode", choices=("two-pass", "single-pass", "single-shared"),
                        default="single-pass", help="원본 읽기 횟수와 모델 전처리 공유 여부")
    parser.add_argument("--plate-imgsz", type=int, default=640, help="원판 모델 입력 긴 변 (자세는 640)")
    parser.add_argument("--plate-mode", choices=("single", "cutmix"), default="single",
                        help="원판 베이스라인 단독 또는 빈 출력을 CutMix로 보완")
    args = parser.parse_args()
    output, summary = analyze(args.video_id, device=args.device, read_mode=args.read_mode,
                              plate_imgsz=args.plate_imgsz, plate_mode=args.plate_mode)
    interval = summary["manual_phase_interval"]
    print(f"전체 {summary['frame_count']}프레임 · {summary['plate_counts']}")
    if interval["rise_px"] is None:
        print(f"상승량 계산 보류 · 원판 미탐 프레임 {interval['missing_endpoint_frames']}")
    else:
        print(f"수동 이륙 {interval['lift_off_frame']} → 무릎 통과 {interval['knee_pass_frame']}: "
              f"원판 중심 {interval['rise_px']:+.2f} px (위쪽이 +)")
    pulls = summary["coaching"]["pulls"]
    if pulls.get("ratio") is not None:
        print(f"이륙→무릎 {pulls['early']:.2f} px/s · 무릎→접촉 {pulls['late']:.2f} px/s · 앞/뒤 {pulls['ratio']:.3f}")
    else:
        print("속도 비율 보류 ·", pulls["reason"])
    catch = summary["coaching"]["catch"]
    if catch.get("forward_px") is not None:
        print(f"캐치 바−어깨 · 앞 {catch['forward_px']:+.2f} px · 위 {catch['up_px']:+.2f} px (모델)")
    else:
        print("캐치 위치 ·", catch["reason"])
    print(f"로컬 결과: {output}")


if __name__ == "__main__":
    main()
