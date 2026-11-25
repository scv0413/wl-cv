"""승인된 RTX 3090에서 실행한다. 결과는 번호별 원시 예측만 담는다."""

from hashlib import sha256
import json
import os
from pathlib import Path
import numpy as np
import tensorrt as trt
import torch

from tensorrt_3090 import OUTPUT, trt_runner


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parent
    device = torch.cuda.get_device_name(0)
    if device != "NVIDIA GeForce RTX 3090" or os.getenv("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("RTX 3090과 NVIDIA_TF32_OVERRIDE=0 설정을 확인하세요.")
    spec = json.loads((root / "jobs.json").read_text())
    inputs = root / "inputs.npz"
    if sha256(inputs.read_bytes()).hexdigest() != spec["input_sha256"]:
        raise ValueError("입력 체크섬이 다릅니다.")
    OUTPUT.mkdir(exist_ok=True)
    with np.load(inputs, allow_pickle=False) as frames:
        for model, channels in (("plate", 5), ("pose", 56)):
            graph = root / f"{model}.onnx"
            if sha256(graph.read_bytes()).hexdigest() != spec["graph_sha256"][model]:
                raise ValueError(f"{model} 그래프 체크섬이 다릅니다.")
            for precision in ("fp32", "fp16"):
                print(model, precision, flush=True)
                shape = (1, channels, 5040)
                run, _ = trt_runner(graph, shape, precision == "fp16")
                predictions = {}
                for key in spec["jobs"][model]:
                    x = torch.from_numpy(frames[key].astype(np.float32) / 255).to(
                        "cuda"
                    )
                    y = run(x).cpu().numpy().copy()
                    if y.shape != shape or not np.isfinite(y).all():
                        raise ValueError(
                            f"{model} {precision} {key} 원시 출력을 확인하세요."
                        )
                    predictions[key] = y
                if precision == "fp16":
                    layers = json.loads(
                        (OUTPUT / f"{model}-fp16-layers.json").read_text()
                    )["Layers"]
                    if not any(
                        "Half" in t.get("Format/Datatype", "")
                        for layer in layers
                        for t in layer.get("Outputs", [])
                    ):
                        raise ValueError("엔진 내부 FP16 사용을 확인할 수 없습니다.")
                np.savez_compressed(
                    OUTPUT / f"trt-{precision}-{model}.npz", **predictions
                )
    (OUTPUT / "real_run.json").write_text(
        json.dumps(
            {
                "device": device,
                "tensorrt": trt.__version__,
                "tf32": False,
                "input_sha256": spec["input_sha256"],
                "graph_sha256": spec["graph_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
