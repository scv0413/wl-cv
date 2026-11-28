"""RTX 3090에서 검증된 ONNX와 네이티브 엔진을 FastAPI가 읽을 형태로 준비한다."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory



MODEL_FILES = {"baseline": "plate_baseline", "cutmix": "plate_cutmix",
               "cosine": "plate_cosine", "pose": "yolo26n-pose"}
INPUT_SHAPE = (1, 3, 640, 384)


def check_source(path, expected_hash):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"{path.name}이 검증한 모델과 다릅니다.")
    return digest


def model_metadata(role):
    # 원판 그래프는 학습 경로를 제거하면서 클래스 이름도 빠졌다. 체크섬을 대조한 뒤 복원한다.
    is_pose = role == "pose"
    meta = dict(task="pose" if is_pose else "detect", names={0: "person" if is_pose else "plate"},
                batch=1, stride=32, imgsz=list(INPUT_SHAPE[2:]), channels=3, end2end=False,
                args={"dynamic": False, "nms": False})
    if is_pose:
        meta["kpt_shape"] = [17, 3]
    return meta


def build_plan(graph, logger):
    import tensorrt as trt

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(graph.read_bytes()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"{graph.name} 엔진 빌드 실패")
    return bytes(plan)


def inspect_engine(engine, role):
    import tensorrt as trt

    channels = 56 if role == "pose" else 5
    if engine.num_io_tensors != 2:
        raise ValueError(f"{role} 엔진 입출력 개수가 다릅니다.")
    for index, shape, mode in [(0, INPUT_SHAPE, trt.TensorIOMode.INPUT),
                               (1, (1, channels, 5040), trt.TensorIOMode.OUTPUT)]:
        name = engine.get_tensor_name(index)
        if (engine.get_tensor_mode(name) != mode or tuple(engine.get_tensor_shape(name)) != shape
                or engine.get_tensor_dtype(name) != trt.float32):
            raise ValueError(f"{role}/{name}의 형상·자료형이 기존 검증과 다릅니다.")
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    count = sum(any("Half" in out.get("Format/Datatype", "") for out in layer.get("Outputs", []))
                for layer in layers)
    if count == 0:
        raise ValueError(f"{role} 엔진에서 FP16 연산층을 확인할 수 없습니다.")
    return count


def prepare(source, output, *, include_cosine=False):
    import tensorrt as trt
    import torch

    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 3090":
        raise RuntimeError(f"이번 검증은 RTX 3090에서 실행하세요. 현재 장치 {gpu}")
    if output.exists():
        raise FileExistsError(f"기존 엔진을 보존하도록 새 출력 폴더를 지정하세요: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    release = json.loads(Path(__file__).with_name("release.json").read_text())
    report = {"gpu": gpu, "tensorrt": trt.__version__, "models": {}}
    with TemporaryDirectory(dir=output.parent) as temp:
        package = Path(temp) / "tensorrt"
        package.mkdir()
        for role, filename in MODEL_FILES.items():
            if role == "cosine" and not include_cosine:
                continue
            graph = source / f"{role}.onnx"
            graph_sha = check_source(graph, release[role]["onnx_sha256"])
            metadata = model_metadata(role)
            cached = source / f"{role}-fp16.engine"
            engine = None
            if cached.is_file() and "raw_engine_sha256" in release[role]:
                check_source(cached, release[role]["raw_engine_sha256"])
                plan = cached.read_bytes()
                engine = runtime.deserialize_cuda_engine(plan)
            reused = engine is not None
            if not reused:
                print(f"{role} FP16 엔진 빌드", flush=True)
                plan = build_plan(graph, logger)
                engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError(f"{role} 엔진을 RTX 3090에 로드하지 못했습니다.")
            half_layers = inspect_engine(engine, role)
            # Ultralytics는 길이 4바이트, JSON 메타데이터, 원래 엔진 바이트 순서로 읽는다.
            header = json.dumps(metadata).encode()
            target = package / f"{filename}.engine"
            target.write_bytes(len(header).to_bytes(4, "little") + header + plan)
            report["models"][role] = dict(source="cached" if reused else "built",
                                          half_layers=half_layers, graph_sha256=graph_sha,
                                          engine_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            print(f"{role}: FP16 출력이 있는 층 {half_layers}개", flush=True)
        (package / "engines.json").write_text(json.dumps(report, indent=2))
        package.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-cosine", action="store_true", help="이전 무릎 통과·접촉 상승량 실험용 엔진도 준비")
    args = parser.parse_args()
    prepare(args.source, args.output, include_cosine=args.include_cosine)
