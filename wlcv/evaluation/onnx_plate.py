"""같은 FP32 텐서에서 원판 베이스라인의 PyTorch·ONNX 출력을 비교한다."""
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from tqdm.auto import tqdm
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from wlcv.config import ROOT, RUNS, SPLIT, VIDEOS
from wlcv.data.frames import frames_rgb, read_rgb
from wlcv.evaluation.plate_track import BOX, KEY, select_centers

WEIGHTS = RUNS / "plate_baseline/weights/best.pt"
RUNTIMES = ("pytorch", "onnx")
BOX_TOL = 0.10       # 원본 영상 좌표(px)
SCORE_TOL = 0.0001
THREADS = 4
WARMUP = 10


def load_review(reference_dir):
    settings = json.loads((reference_dir / "settings.json").read_text())
    if sha256(WEIGHTS.read_bytes()).hexdigest() != settings["weights_sha256"]:
        raise ValueError("44번 실험에 사용한 원판 best.pt와 현재 파일이 다릅니다.")
    rows = pd.read_csv(reference_dir / "frames.csv").query("factor == 1").copy()
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    trained = set(pd.read_csv(WEIGHTS.parent.parent / "train_people.csv").person_id)
    if (len(rows) != 56 or rows.duplicated(KEY).any() or rows.person_id.nunique() != 6
            or not split.loc[rows.video_id, "role"].eq("validation").all()
            or trained.intersection(rows.person_id)
            or split.loc[rows.video_id, "person_id"].tolist() != rows.person_id.tolist()):
        raise ValueError("검증 6명의 원본 56장과 사람별 분할을 확인하세요.")
    return rows


def export_graph(folder):
    """원본 checkpoint는 보존하고 이 실행 폴더에 배포용 ONNX만 남긴다."""
    with TemporaryDirectory(dir=folder) as temp:
        local_weights = Path(temp) / "plate.pt"
        shutil.copyfile(WEIGHTS, local_weights)
        path = YOLO(str(local_weights)).export(format="onnx", imgsz=(640, 384), batch=1,
                                               device="cpu", dynamic=False,
                                               nms=None, opset=17, simplify=False)
        graph_path = Path(path).replace(folder / "plate.onnx")
    graph = onnx.load(graph_path)
    onnx.checker.check_model(graph)
    return graph_path


def raw_pytorch(predictor, tensor):
    output = predictor.model(tensor)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output.numpy()


def candidates(raw, input_hw, image_shape):
    """단일 클래스 검출 헤드에 같은 NMS와 원본 좌표 복원을 적용한다."""
    pred = non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=.001,
                               iou_thres=.7, max_det=300)[0]
    if len(pred):
        pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], image_shape)
    return pred.numpy()


def compare_boxes(left, right):
    """NMS 뒤 출력 순서가 달라도 실제 좌표가 가까운 박스끼리 일대일 대응한다."""
    coord_error = score_error = 0.0
    if len(left) and len(right):
        distances = np.abs(left[:, None, :4] - right[None, :, :4]).max(axis=2)
        i, j = linear_sum_assignment(distances)
        coord_error = float(distances[i, j].max())
        score_error = float(np.abs(left[i, 4] - right[j, 4]).max())
    return abs(len(left)-len(right)), coord_error, score_error


@torch.inference_mode()
def run_comparison(folder, review):
    graph = export_graph(folder)
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # ORT의 유휴 작업 스레드가 교대로 실행하는 PyTorch의 CPU 시간을 사용하지 않게 한다.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    entry, = session.get_inputs()
    output, = session.get_outputs()
    if entry.shape != [1, 3, 640, 384] or entry.type != "tensor(float)" or output.shape != [1, 5, 5040]:
        raise ValueError(f"예상과 다른 그래프 입출력: {entry.shape}, {entry.type}, {output.shape}")

    videos = sorted(review.video_id.unique())
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    first = read_rgb(VIDEOS / f"{videos[0]}.mp4", 0)
    detector = YOLO(str(WEIGHTS))
    if detector.names != {0: "plate"}:
        raise ValueError("원판 단일 클래스 가중치를 확인하세요.")
    detector.predict(first[:, :, ::-1].copy(), conf=.001, imgsz=640, device="cpu", verbose=False)
    predictor = detector.predictor
    torch.set_num_threads(THREADS)
    times, comparisons = [], []
    boxes = {name: [] for name in RUNTIMES}
    for video in videos:
        count = 0
        path = VIDEOS / f"{video}.mp4"
        for frame, rgb in enumerate(tqdm(frames_rgb(path), total=int(meta.at[video, "frame_count"]), desc=video)):
            tensor = predictor.preprocess([rgb[:, :, ::-1].copy()])
            if list(tensor.shape) != entry.shape or tensor.dtype != torch.float32:
                raise ValueError(f"{video} {frame}: 기존 640 전처리와 ONNX 입력이 다릅니다.")
            array = tensor.numpy()  # 두 런타임이 같은 CPU 메모리·픽셀을 읽는다.
            feed = {entry.name: array}
            if frame == 0:
                for _ in range(WARMUP):
                    raw_pytorch(predictor, tensor)
                    session.run([output.name], feed)
            raw = {}
            for name in (RUNTIMES if len(comparisons) % 2 == 0 else reversed(RUNTIMES)):
                start = perf_counter_ns()
                if name == "pytorch":
                    pred = raw_pytorch(predictor, tensor)
                else:
                    pred = session.run([output.name], feed)[0]
                elapsed = (perf_counter_ns() - start) / 1e6
                times.append((video, frame, name, elapsed))
                raw[name] = pred
            if any(p.shape != (1, 5, 5040) or not np.isfinite(p).all() for p in raw.values()):
                raise ValueError(f"{video} {frame}: 원시 출력의 모양 또는 값이 올바르지 않습니다.")
            delta = np.abs(raw["pytorch"] - raw["onnx"])[0]
            now = {}
            for name in RUNTIMES:
                now[name] = candidates(raw[name], tensor.shape[2:], rgb.shape)
                for x1, y1, x2, y2, score, cls in now[name]:
                    if cls != 0:
                        raise ValueError("원판 외 클래스가 있습니다.")
                    boxes[name].append((video, frame, score, x1, y1, x2, y2))
            unmatched, xy, conf = compare_boxes(now["pytorch"], now["onnx"])
            comparisons.append((video, frame, delta[:4].max(), delta[4].max(), len(now["pytorch"]),
                                len(now["onnx"]), unmatched, xy, conf))
            count += 1
        if count != meta.at[video, "frame_count"]:
            raise ValueError(f"{video}: 영상의 실제 프레임 수를 확인하세요.")

    detail = pd.DataFrame(comparisons, columns=KEY + ["raw_xy_px", "raw_score", "n_pytorch", "n_onnx",
                                                      "unmatched", "box_max_px", "score_max"])
    timing = pd.DataFrame(times, columns=KEY + ["runtime", "ms"])
    selections = []
    for name in RUNTIMES:
        table = pd.DataFrame(boxes[name], columns=KEY + ["confidence"] + BOX)
        selected = select_centers(detail[KEY], table, distance_ratio=.5, show_progress=False)
        selections.append(selected[selected.method.eq("Past link")].assign(runtime=name))
    tracks = pd.concat(selections, ignore_index=True)
    detail.to_csv(folder / "frames.csv", index=False)
    timing.to_csv(folder / "latency.csv", index=False)
    tracks.to_csv(folder / "tracks.csv", index=False)
    settings = dict(weights_sha256=sha256(WEIGHTS.read_bytes()).hexdigest(), onnx_sha256=sha256(graph.read_bytes()).hexdigest(),
                    model_bytes=graph.stat().st_size, torch=torch.__version__, onnx=onnx.__version__, ort=ort.__version__,
                    ultralytics=ultralytics.__version__, platform=platform.platform(),
                    cpu=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                    if platform.system() == "Darwin" else platform.machine(),
                    providers=session.get_providers(), threads=torch.get_num_threads(), warmup_per_video=WARMUP,
                    input_name=entry.name, input_shape=entry.shape, output_name=output.name, output_shape=output.shape,
                    opset=17, dtype="float32", candidate_conf=.001, nms_iou=.7, direct_conf=.10,
                    box_tolerance_px=BOX_TOL, confidence_tolerance=SCORE_TOL)
    (folder / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    return detail, timing, tracks


def evaluate(detail, timing, tracks, review):
    """운영 선택은 전체 영상에서, 수동 박스와 IoU 비교는 라벨이 있는 56장에서 계산한다."""
    selected = tracks.pivot(index=KEY, columns="runtime", values="source")
    if selected.isna().any().any() or len(selected) != len(detail):
        raise ValueError("런타임별 프레임 목록이 다릅니다.")
    labels = review[KEY + ["source", "confidence", "match"] + BOX + ["gt_" + c for c in BOX]].rename(
        columns={c: "reference_" + c for c in ["source", "confidence", "match"] + BOX})
    scored = tracks.merge(labels, on=KEY, validate="many_to_one")
    intersection = (np.minimum(scored.x2, scored.gt_x2)-np.maximum(scored.x1, scored.gt_x1)).clip(lower=0) * (
        np.minimum(scored.y2, scored.gt_y2)-np.maximum(scored.y1, scored.gt_y1)).clip(lower=0)
    area = (scored.x2-scored.x1) * (scored.y2-scored.y1)
    truth_area = (scored.gt_x2-scored.gt_x1) * (scored.gt_y2-scored.gt_y1)
    scored["iou"] = intersection / (area + truth_area - intersection)
    scored["match"] = scored.iou.ge(.50)
    scored["reference_box_error"] = np.abs(scored[BOX].to_numpy() -
                                     scored[["reference_" + c for c in BOX]].to_numpy()).max(axis=1)
    scored["reference_score_error"] = (scored.confidence - scored.reference_confidence).abs()
    indexed = tracks.set_index(["runtime"] + KEY, verify_integrity=True).sort_index()
    summary = []
    for name in RUNTIMES:
        g = scored[scored.runtime.eq(name)]
        b02 = indexed.loc[(name, "b02_sn_h_02", 35)]
        c04 = indexed.loc[(name, "c04_sn_f_01")]
        ms = timing.loc[timing.runtime.eq(name), "ms"]
        summary.append(dict(runtime=name, frames=len(ms), direct=g.source.eq("high").sum(),
                            linked=g.source.eq("linked").sum(), matched_56=int(g.match.sum()),
                            b02_f35_source=b02.source, b02_f35_score=b02.confidence,
                            c04_rise_px=c04.at[11, "cy"] - c04.at[27, "cy"],
                            p50_ms=ms.median(), p95_ms=ms.quantile(.95)))
    summary = pd.DataFrame(summary).set_index("runtime")
    left = indexed.loc["pytorch"]
    right = indexed.loc["onnx"].reindex(left.index)
    checks = {
        "candidate_count_diff_frames": int(detail.unmatched.gt(0).sum()),
        "candidate_box_max_px": float(detail.box_max_px.max()),
        "candidate_score_max": float(detail.score_max.max()),
        "selected_box_max_px": float((left[BOX] - right[BOX]).abs().max().max()),
        "selected_score_max": float((left.confidence-right.confidence).abs().max()),
        "source_diff_frames": int(selected.pytorch.ne(selected.onnx).sum()),
        "reference_mismatch_rows": int(((scored.source != scored.reference_source)
                                       | (scored.match != scored.reference_match)
                                       | (scored.reference_box_error > BOX_TOL)
                                       | (scored.reference_score_error > SCORE_TOL)).sum()),
        "raw_xy_max_input_px": float(detail.raw_xy_px.max()), "raw_score_max": float(detail.raw_score.max()),
    }
    checks["passed"] = bool(
        checks["candidate_count_diff_frames"] == checks["source_diff_frames"] == checks["reference_mismatch_rows"] == 0
        and max(checks["candidate_box_max_px"], checks["selected_box_max_px"]) <= BOX_TOL
        and max(checks["candidate_score_max"], checks["selected_score_max"]) <= SCORE_TOL
        and summary.b02_f35_source.eq("linked").all() and (summary.c04_rise_px-160.51).abs().le(BOX_TOL).all()
    )
    return summary, scored, checks
