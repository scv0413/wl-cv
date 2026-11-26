"""번호가 붙은 입력으로 RTX 3090 FP16 엔진을 빌드하고 원시 출력을 보존한다."""

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from tensorrt_3090 import OUTPUT, trt_runner


@torch.inference_mode()
def main():
    folder = Path(__file__).resolve().parent
    device = torch.cuda.get_device_name()
    if device != "NVIDIA GeForce RTX 3090" or os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("RTX 3090 및 TF32 설정을 확인하세요.")
    spec = json.loads((folder / "jobs.json").read_text())
    inputs = folder / "inputs.npz"
    if sha256(inputs.read_bytes()).hexdigest() != spec["input_sha256"]:
        raise ValueError("입력 체크섬이 다릅니다.")
    OUTPUT.mkdir(exist_ok=True)
    info = dict(
        device=device,
        trt=trt.__version__,
        torch=str(torch.__version__),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        tf32=False,
        input_sha256=spec["input_sha256"],
        graph_sha256=spec["graph_sha256"],
        models={},
    )
    with np.load(inputs, allow_pickle=False) as frames:
        for name, keys in spec["jobs"].items():
            graph = folder / f"{name}.onnx"
            if sha256(graph.read_bytes()).hexdigest() != spec["graph_sha256"][name]:
                raise ValueError(f"{name} 그래프 체크섬이 다릅니다.")
            print(f"BUILD {name}", flush=True)
            run, seconds = trt_runner(graph, (1, 5, 5040), True)
            layers = json.loads((OUTPUT / f"{name}-fp16-layers.json").read_text())["Layers"]
            half = sum(
                any("Half" in t.get("Format/Datatype", "") for t in l.get("Outputs", []))
                for l in layers
            )
            if not half:
                raise RuntimeError(f"{name} 내부 FP16 사용을 확인할 수 없습니다.")
            values = {}
            for k in keys:
                x = torch.from_numpy(frames[k].astype(np.float32) / 255).cuda()
                y = run(x).cpu().numpy().copy()
                if y.shape != (1, 5, 5040) or not np.isfinite(y).all():
                    raise ValueError(f"{name} {k} 출력 오류")
                values[k] = y
            np.savez_compressed(OUTPUT / f"trt-{name}.npz", **values)
            info["models"][name] = dict(frames=len(keys), build_seconds=seconds, half_layers=half)
            print(f"DONE {name} {len(keys)} frames {half} FP16 layers", flush=True)
    (OUTPUT / "real_run.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
