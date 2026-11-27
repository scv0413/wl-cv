"""MP4 한 편에서 원판·자세를 추론하고 영상, 표, 요약을 함께 저장한다."""
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from wlcv.config import local_path
from wlcv.data.mp4_input import inspect_mp4, validate_events
from wlcv.runtime.inference import load_plate_model, single_pass_frames
from wlcv.runtime.measurements import coaching_summary, measure_knee_to_contact, phase_summary
from wlcv.runtime.models import file_models
from wlcv.runtime.overlay import PLATE_STATES, render_frames
from wlcv.runtime.recovery import save_recovery_curve


def analyze_file(path, *, near_side, facing, manual_events=None, device="cpu",
                 read_mode="single-pass", plate_imgsz=640, plate_mode="single", output_parent=None):
    """새 MP4의 공개 진입점. 시점은 0부터 세는 원본 프레임 번호다."""
    start = perf_counter()
    if near_side not in ("left", "right") or facing not in ("left", "right"):
        raise ValueError("파일 입력에는 --near-side와 --facing을 각각 left 또는 right로 지정하세요.")
    path = Path(path).expanduser()
    events = dict(manual_events or {})
    validate_events(events)
    info = inspect_mp4(path, near_side)
    parent = local_path("WL_OUTPUT_DIR", "outputs/file_analysis") if output_parent is None else Path(output_parent)
    return analyze_video(
        path, info, video=path.stem, fps=info.fps, direction={"left": -1, "right": 1}[facing],
        events=events, parent=parent,
        weights=file_models(), input_mode="file", start=start, device=device,
        read_mode=read_mode, plate_imgsz=plate_imgsz, plate_mode=plate_mode,
    )


def analyze_video(path, info, *, video, fps, direction, events, parent, weights, input_mode,
                  start, device, read_mode, plate_imgsz, plate_mode, detector=load_plate_model):
    """입력 정보가 준비된 한 영상에 같은 추론·측정 규칙을 적용한다."""
    lift, knee = events.get("lift_off"), events.get("knee_pass")
    if plate_mode not in ("single", "cutmix"):
        raise ValueError("원판 분석 모드는 single 또는 cutmix입니다.")
    if read_mode not in ("two-pass", "single-pass", "single-shared"):
        raise ValueError("지원하지 않는 영상 읽기 방식입니다.")
    if plate_imgsz <= 0 or plate_imgsz % 32:
        raise ValueError("원판 입력 크기는 양의 32 배수여야 합니다.")
    if read_mode == "single-shared" and plate_imgsz != 640:
        raise ValueError("입력 텐서를 공유하려면 원판과 자세 크기가 모두 640이어야 합니다.")
    use_engines = weights.plate.suffix == ".engine"
    if use_engines and device != "0":
        raise ValueError("TensorRT 엔진에는 WL_API_DEVICE=0을 지정하세요.")
    if device not in ("cpu", "mps", "0"):
        raise ValueError("cpu, mps 또는 0(첫 번째 CUDA GPU)을 지정하세요.")
    if device == "0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("이 실행 환경에서 MPS를 사용할 수 없습니다.")
    configured = perf_counter()
    timings = {"setup": configured - start, "pose": 0.0}
    plate_model = detector(weights.plate)
    cutmix_model = detector(weights.cutmix) if plate_mode == "cutmix" else None
    timings.update(plate_load=perf_counter()-configured, plate_inference=0.0, plate_tracking=0.0)
    pose_model = YOLO(str(weights.pose.resolve(strict=True)), task="pose" if use_engines else None)
    if pose_model.task != "pose":
        raise ValueError(f"자세 추정 가중치를 확인하세요: {weights.pose}")

    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # 영상·표·요약 파일을 모두 저장한 뒤 결과 폴더를 공개한다. 실패해도 지난 실행은 다시 열 수 있다.
    with TemporaryDirectory(dir=parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        overlay = folder / "overlay.mp4"
        video_start = perf_counter()
        stream = single_pass_frames(path, plate_model, pose_model, device, read_mode == "single-shared",
                                    timings, plate_imgsz=plate_imgsz, cutmix_model=cutmix_model,
                                    edge_rule="continuity", reread=read_mode == "two-pass")
        contact = events.get("drive")
        has_pair = knee is not None and contact is not None
        if has_pair and knee >= contact:
            raise ValueError("무릎 통과가 접촉보다 먼저여야 합니다.")
        endpoint_images = {knee: None, contact: None} if has_pair and weights.cosine.is_file() else {}
        rows = render_frames(stream, info, fps, lift, knee, overlay, events, timings,
                             endpoint_images=endpoint_images)
        rendered = perf_counter()
        timings["render_other"] = rendered - start - sum(timings.values())
        check = cv2.VideoCapture(str(overlay))
        try:
            if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                raise OSError("오버레이와 프레임 기록 수가 일치하지 않습니다.")
        finally:
            check.release()
        if input_mode == "file":
            validate_events(events, len(rows))
        measurement_start = perf_counter()
        cosine = detector(weights.cosine) if endpoint_images else None
        measurement = measure_knee_to_contact(
            rows, video, knee, contact, endpoint_images, cosine, weights.cosine, device
        )
        timings["cosine_measurement"] = perf_counter() - measurement_start if has_pair else 0.0
        rows["forward_px"] = (rows.plate_cx_px - rows.shoulder_x_px) * (direction or np.nan)
        rows.to_csv(folder / "frames.csv", index=False)
        counts = {name: int(rows.plate_status.eq(name).sum()) for name, _, _ in PLATE_STATES.values()}
        summary = {"video_id": video, "frame_count": len(rows), "plate_counts": counts,
                   "overlay_fps": fps, "near_side": info.near_side,
                   "input_mode": input_mode,
                   "facing": {-1: "left", 1: "right"}.get(direction),
                   "manual_events": {k: v for k, v in {"lift_off": lift, "knee_pass": knee,
                                     **{k: events.get(k) for k in ("drive", "catch", "finish")}}.items()
                                     if v is not None},
                   "knee_to_contact_rise": measurement,
                   "plate_mode": plate_mode, "overlay_seconds": rendered - video_start,
                   "selected_model_counts": {name: int(rows.selected_model.eq(name).sum())
                                             for name in ("baseline", "cutmix", "none")},
                   "cutmix_weights": str(weights.cutmix) if cutmix_model is not None else None,
                   "plate_weights": str(weights.plate), "pose_weights": str(weights.pose),
                   "coaching": coaching_summary(rows, lift, knee, events, direction, knee_to_contact=measurement),
                   "plate_imgsz": plate_imgsz, "pose_imgsz": 640,
                   "device": device, "read_mode": read_mode, "manual_phase_interval": phase_summary(rows, lift, knee)}
        if use_engines:
            active = {"baseline": plate_model, "pose": pose_model, "cutmix": cutmix_model, "cosine": cosine}
            backends = {}
            for name, model in active.items():
                if model is None:
                    continue
                # predict를 실제 호출한 뒤에만 predictor가 존재한다.
                runtime = model.predictor.model
                if runtime.format != "engine" or runtime.device.type != "cuda":
                    raise RuntimeError(f"{name} 모델이 CUDA TensorRT 엔진으로 실행되지 않았습니다.")
                backends[name] = type(runtime.backend).__name__
            summary["inference_backends"] = backends
        endpoint_angles = rows.set_index("frame_index").trunk_angle_deg.reindex([lift, knee])
        summary["trunk_change_deg"] = (float(endpoint_angles.loc[knee] - endpoint_angles.loc[lift])
                                       if endpoint_angles.notna().all() else None)
        if input_mode == "file":
            summary["coaching"]["catch_to_finish"] = save_recovery_curve(rows, events, folder)
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        saved = perf_counter()
        timings["verify_and_save"] = saved - rendered - timings["cosine_measurement"]
        elapsed = saved - start
        timing = {"pipeline_seconds": elapsed, "frames_per_second": len(rows) / elapsed,
                  "stage_seconds": timings}
        (folder / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        folder.rename(destination)
    return destination, summary
