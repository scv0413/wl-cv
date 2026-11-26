"""Mac에서 같은 입력에 PyTorch FP32와 ORT CPU FP32 대조군을 실행한다."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.evaluation import onnx_plate, onnx_pose


@torch.inference_mode()
def run_controls(folder):
    folder = Path(folder)
    payload = folder / "gpu_input"
    output = folder / "predictions"
    output.mkdir(exist_ok=True)
    jobs = json.loads((payload / "jobs.json").read_text())["jobs"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    torch.set_num_threads(4)
    with np.load(payload / "inputs.npz", allow_pickle=False) as frames:
        for name, weights in (
            ("plate", onnx_plate.WEIGHTS),
            ("pose", onnx_pose.WEIGHTS),
        ):
            model = YOLO(str(weights))
            model.predict(
                np.zeros((640, 384, 3), np.uint8),
                imgsz=640,
                device="cpu",
                verbose=False,
            )
            session = ort.InferenceSession(
                str(payload / f"{name}.onnx"),
                options,
                providers=["CPUExecutionProvider"],
            )
            pt, ort_rows = {}, {}
            for key in tqdm(jobs[name], desc=f"Mac FP32 {name}", unit="장"):
                image = frames[key].astype(np.float32) / 255
                pt[key] = onnx_plate.raw_pytorch(
                    model.predictor, torch.from_numpy(image)
                )
                (ort_rows[key],) = session.run(
                    None, {session.get_inputs()[0].name: image}
                )
            np.savez_compressed(output / f"pytorch-fp32-{name}.npz", **pt)
            np.savez_compressed(output / f"ort-cpu-fp32-{name}.npz", **ort_rows)
