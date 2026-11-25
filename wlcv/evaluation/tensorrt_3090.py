"""RTX 3090 한 대에서 같은 GPU 텐서의 원판·자세 추론을 측정한다."""

import collections
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "result"
SHAPE = (1, 3, 640, 384)
GPU = torch.device("cuda:0")
ENGINES = ("pytorch-fp32", "ort-cuda-fp32", "trt-fp32", "trt-fp16")
LOGGER = trt.Logger(trt.Logger.WARNING)


def native_runner(path):
    model = torch.load(path, weights_only=False, map_location="cpu").eval().to(GPU)

    def run(x):
        y = model(x)
        return y[0] if isinstance(y, (list, tuple)) else y

    return run


def ort_runner(path, output_shape, profile=False):
    options = ort.SessionOptions()
    options.enable_profiling = profile
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "use_tf32": "0",
                "device_id": 0,
                "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
            },
        )
    ]
    session = ort.InferenceSession(str(path), options, providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("ONNX Runtime CUDA provider was not loaded")
    binding = session.io_binding()
    y = torch.empty(output_shape, device=GPU)
    binding.bind_output(
        session.get_outputs()[0].name, "cuda", 0, np.float32, output_shape, y.data_ptr()
    )
    name = session.get_inputs()[0].name

    def run(x):
        binding.bind_input(name, "cuda", 0, np.float32, SHAPE, x.data_ptr())
        session.run_with_iobinding(binding)
        return y

    return run, session


def trt_runner(path, output_shape, fp16):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(path.read_bytes()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    start = perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = perf_counter() - start
    if plan is None:
        raise RuntimeError("TensorRT build failed")
    (OUTPUT / f"{path.stem}-{'fp16' if fp16 else 'fp32'}.engine").write_bytes(
        bytes(plan)
    )
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    src, dst = (engine.get_tensor_name(i) for i in range(2))
    assert engine.get_tensor_mode(src) == trt.TensorIOMode.INPUT
    assert (
        tuple(engine.get_tensor_shape(src)) == SHAPE
        and tuple(engine.get_tensor_shape(dst)) == output_shape
    )
    assert engine.get_tensor_dtype(src) == engine.get_tensor_dtype(dst) == trt.float32
    inspector = engine.create_engine_inspector()
    detail = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    suffix = "fp16" if fp16 else "fp32"
    (OUTPUT / f"{path.stem}-{suffix}-layers.json").write_text(detail)
    y = torch.empty(output_shape, device=GPU)
    assert context.set_tensor_address(dst, y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    def run(x):
        # Keep runtime and engine alive until the last GPU operation completes.
        _ = (runtime, engine)
        if not context.set_tensor_address(
            src, x.data_ptr()
        ) or not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        return y

    return run, build_seconds


def quality(predictions, reference, model):
    delta = np.abs(predictions - reference)
    groups = {"box_input_px": delta[:, :4], "object_score": delta[:, 4:5]}
    if model == "pose":
        points = delta[:, 5:].reshape(len(delta), 17, 3, -1)
        groups.update(joint_input_px=points[:, :, :2], joint_score=points[:, :, 2])
    return {
        k: {"p95": float(np.quantile(a, 0.95)), "max": float(a.max())}
        for k, a in groups.items()
    }


@torch.inference_mode()
def main():
    OUTPUT.mkdir(exist_ok=True)
    assert "3090" in torch.cuda.get_device_name(0) and torch.cuda.device_count() == 1
    assert os.getenv("NVIDIA_TF32_OVERRIDE") == "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(560)
    images = [
        torch.from_numpy(rng.random(SHAPE, dtype=np.float32)).to(GPU) for _ in range(12)
    ]
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ort": ort.__version__,
        "trt": trt.__version__,
        "shape": SHAPE,
        "seed": 560,
        "tf32": False,
        "scope": "GPU resident input to synchronized output; no video or NMS",
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    hashes = json.loads((ROOT / "models.json").read_text())
    (OUTPUT / "environment.json").write_text(json.dumps(info, indent=2))
    for model, channels in [("plate", 5), ("pose", 56)]:
        shape = (1, channels, 5040)
        graph, weights = ROOT / f"{model}.onnx", ROOT / f"{model}.pt"
        for p, key in [(graph, "onnx"), (weights, "eager_pt")]:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[model][key]
        profile, s = ort_runner(graph, shape, profile=True)
        profile(images[0])
        torch.cuda.synchronize()
        events = json.loads(Path(s.end_profiling()).read_text())
        placement = collections.Counter(
            e["args"]["provider"]
            for e in events
            if e.get("cat") == "Node" and "provider" in e.get("args", {})
        )
        if not placement["CUDAExecutionProvider"]:
            raise RuntimeError("CUDA kernel placement not confirmed")
        cpu_ops = {
            e["args"].get("op_name")
            for e in events
            if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        }
        if cpu_ops - {
            "Shape",
            "Size",
            "Gather",
            "Unsqueeze",
            "Squeeze",
            "Concat",
            "Slice",
            "Cast",
            "Reshape",
        }:
            raise RuntimeError(f"Unexpected CPU operators: {cpu_ops}")
        del profile, s
        runs = {
            ENGINES[0]: native_runner(weights),
            ENGINES[1]: ort_runner(graph, shape)[0],
        }
        build = {}
        for name in ENGINES[2:]:
            print(model, "building", name, flush=True)
            runs[name], build[name] = trt_runner(graph, shape, name.endswith("16"))
        # 개별 runner의 첫 호출과 예열 후 반복을 구분한다.
        first_call = {}
        for engine, run in runs.items():
            torch.cuda.synchronize()
            start = perf_counter()
            run(images[0])
            torch.cuda.synchronize()
            first_call[engine] = (perf_counter() - start) * 1000
        reference = np.concatenate(
            [runs[ENGINES[0]](im).cpu().numpy() for im in images]
        )
        assert (
            reference.shape == (len(images), channels, 5040)
            and np.isfinite(reference).all()
        )
        errors = {}
        measures = {name: [] for name in ENGINES}
        for name in ENGINES:
            values = np.concatenate(
                [runs[name](im).cpu().numpy().copy() for im in images]
            )
            if values.shape != reference.shape or not np.isfinite(values).all():
                raise RuntimeError(f"bad output: {model} {name}")
            errors[name] = quality(values, reference, model)
            for i in range(24):
                runs[name](images[i % len(images)])
            torch.cuda.synchronize()
        for cycle in range(4):
            for name in ENGINES[cycle:] + ENGINES[:cycle]:
                for i in range(60):
                    x = images[i % len(images)]
                    torch.cuda.synchronize()
                    start = perf_counter()
                    runs[name](x)
                    torch.cuda.synchronize()
                    measures[name].append((perf_counter() - start) * 1000)
        rows = {
            name: {
                "samples": len(ms),
                "p50_ms": float(np.median(ms)),
                "p95_ms": float(np.quantile(ms, 0.95)),
                "fps": float(1000 / np.mean(ms)),
                "first_call_ms": first_call[name],
                "build_s": build.get(name),
            }
            for name, ms in measures.items()
        }
        report = {
            "model": model,
            "timing": rows,
            "raw_output_deltas": errors,
            "ort_provider_nodes": dict(placement),
            "ort_cpu_ops": sorted(cpu_ops),
            "raw_ms": measures,
        }
        (OUTPUT / f"{model}.json").write_text(json.dumps(report, indent=2))
        print(model, rows, flush=True)
        del runs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
