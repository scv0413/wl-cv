"""합성 H.264 영상 116장에서 GPU 추론과 CPU 영상 처리 시간을 따로 잰다."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
import torch
import tensorrt as trt
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

# Pod에서 두 파일을 같은 폴더에 놓는다.
from benchmark import (
    ENGINES,
    GPU,
    LOGGER,
    ROOT,
    SHAPE,
    native_runner,
    ort_runner,
    quality,
)

FRAMES = 116
SIZE = "1080x1920"


def load_plan(model, channels, precision):
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(
        (ROOT / "result" / f"{model}-{precision}.engine").read_bytes()
    )
    context = engine.create_execution_context()
    y = torch.empty((1, channels, 5040), device=GPU)
    src, dst = (engine.get_tensor_name(i) for i in range(2))
    assert context.set_tensor_address(dst, y.data_ptr())

    def run(x):
        _ = (runtime, engine)
        assert context.set_tensor_address(src, x.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return y

    return run


def overlay(rgb, outputs, index):
    # 연산·그리기 비용을 보기 위한 합성 장면이다. 선수 동작 지표는 만들지 않는다.
    counts = []
    for pred, threshold in zip(outputs, (0.001, 0.25)):
        rows = non_max_suppression(pred, conf_thres=threshold, iou_thres=0.70, nc=1)[0]
        counts.append(len(rows))
        if len(rows):
            boxes = ops.scale_boxes(SHAPE[2:], rows[:, :4].clone(), rgb.shape).int()
            for x1, y1, x2, y2 in boxes.tolist():
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (40, 210, 140), 2)
    cv2.putText(
        rgb,
        f"Synthetic {index} | plate {counts[0]} | person {counts[1]}",
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return counts


def process_video(folder, source, name, runs):
    resizer = LetterBox(new_shape=SHAPE[2:], auto=False, stride=32)
    totals = defaultdict(float)
    records = []
    start = perf_counter()
    reader = cv2.VideoCapture(str(source))
    output = folder / f"{name}.mp4"
    encode = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            SIZE,
            "-r",
            "30",
            "-i",
            "-",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-threads",
            "2",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(FRAMES):
            t = perf_counter()
            ok, bgr = reader.read()
            if not ok:
                raise RuntimeError(f"decode failed at {i}")
            totals["decode"] += perf_counter() - t
            t = perf_counter()
            chw = np.ascontiguousarray(
                resizer(image=bgr)[:, :, ::-1].transpose(2, 0, 1)
            )
            x = torch.from_numpy(chw).to(GPU).unsqueeze(0).float().div_(255)
            torch.cuda.synchronize()
            totals["preprocess_h2d"] += perf_counter() - t
            t = perf_counter()
            raw = [run(x) for run in runs]
            torch.cuda.synchronize()
            totals["two_models"] += perf_counter() - t
            t = perf_counter()
            counts = overlay(bgr, [y.cpu() for y in raw], i)
            records.append([i, *counts])
            totals["d2h_nms_overlay"] += perf_counter() - t
            t = perf_counter()
            encode.stdin.write(bgr.tobytes())
            totals["encoder_pipe"] += perf_counter() - t
    finally:
        reader.release()
        t = perf_counter()
        encode.stdin.close()
        if encode.wait() != 0:
            raise RuntimeError("H.264 encoding failed")
        totals["encoder_finish"] += perf_counter() - t
    with (folder / "counts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "plate_candidates", "person_candidates"])
        writer.writerows(records)
    elapsed = perf_counter() - start
    return {
        "engine": name,
        "seconds": elapsed,
        "fps": FRAMES / elapsed,
        "stages_s": dict(totals),
        "synthetic_plate_candidates": sum(a[1] for a in records),
        "synthetic_person_candidates": sum(a[2] for a in records),
    }


@torch.inference_mode()
def run_one(folder, engine):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(2)
    start = perf_counter()
    models = []
    for model, channels in (("plate", 5), ("pose", 56)):
        if engine == "pytorch-fp32":
            run = native_runner(ROOT / f"{model}.pt")
        elif engine == "ort-cuda-fp32":
            run, _ = ort_runner(ROOT / f"{model}.onnx", (1, channels, 5040))
        else:
            run = load_plan(model, channels, engine.split("-")[1])
        models.append(run)
    torch.cuda.synchronize()
    load_s = perf_counter() - start
    row = process_video(folder, folder / "input.mp4", engine, models)
    row["model_load_s"] = load_s
    (folder / "run.json").write_text(json.dumps(row))


@torch.inference_mode()
def compare_trt_precision():
    """같은 합성 텐서·후보에서 내부 FP16 허용의 차이만 측정한다."""
    rng = np.random.default_rng(560)
    inputs = [
        torch.from_numpy(rng.random(SHAPE, dtype=np.float32)).to(GPU) for _ in range(12)
    ]
    result = {}
    for model, channels, threshold in (("plate", 5, 0.10), ("pose", 56, 0.25)):
        outputs = {}
        for precision in ("fp32", "fp16"):
            run = load_plan(model, channels, precision)
            outputs[precision] = np.concatenate(
                [run(x).cpu().numpy().copy() for x in inputs]
            )
        baseline, lower = outputs["fp32"], outputs["fp16"]
        if not np.isfinite(baseline).all() or not np.isfinite(lower).all():
            raise ValueError(f"{model} 비정상 출력")
        high32, high16 = baseline[:, 4] >= threshold, lower[:, 4] >= threshold
        result[model] = {
            "raw_deltas": quality(lower, baseline, model),
            "threshold": threshold,
            "candidates": int(high32.size),
            "above_fp32": int(high32.sum()),
            "above_fp16": int(high16.sum()),
            "threshold_crossings": int(np.count_nonzero(high32 != high16)),
        }
    (ROOT / "result/precision.json").write_text(json.dumps(result, indent=2))


def main():
    rows = []
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        src = folder / "input.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={SIZE}:rate=30",
                "-frames:v",
                str(FRAMES),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-threads",
                "2",
                str(src),
            ],
            check=True,
        )
        # 엔진마다 새 Python 프로세스를 시작한다. 두 번째 반복은 역순으로 실행한다.
        for name in (*ENGINES, *reversed(ENGINES)):
            start = perf_counter()
            subprocess.run([sys.executable, __file__, str(folder), name], check=True)
            wall = perf_counter() - start
            row = json.loads((folder / "run.json").read_text())
            check = cv2.VideoCapture(str(folder / f"{name}.mp4"))
            frames = 0
            while check.grab():
                frames += 1
            check.release()
            with (folder / "counts.csv").open() as f:
                csv_rows = sum(1 for _ in csv.DictReader(f))
            if frames != FRAMES or csv_rows != FRAMES:
                raise RuntimeError(f"저장 결과 확인: 영상 {frames}장, CSV {csv_rows}행")
            row.update(
                command_s=wall,
                command_fps=frames / wall,
                verified_frames=frames,
                verified_csv_rows=csv_rows,
            )
            rows.append(row)
            (ROOT / "result/synthetic_video.json").write_text(
                json.dumps(rows, indent=2)
            )
            print(name, len(rows), "of 8", round(wall, 2), "s", flush=True)

    compare_trt_precision()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] in ENGINES:
        run_one(Path(sys.argv[1]), sys.argv[2])
    elif sys.argv[1:] == ["--precision"]:
        compare_trt_precision()
    else:
        main()
