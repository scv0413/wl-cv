"""기존 원판·자세 모델에서 추론에 필요한 파일만 준비하고 합성 입력으로 검증한다."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from wlcv.config import ROOT
from wlcv.evaluation.onnx_plate import WEIGHTS as PLATE_WEIGHTS
from wlcv.evaluation.onnx_pose import WEIGHTS as POSE_WEIGHTS


@torch.inference_mode()
def prepare_bundle(destination, plate_reference, pose_reference):
    """참조 폴더는 46·47번 실행에서 모델과 함께 settings.json을 저장한 위치다."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    configs = [("plate", PLATE_WEIGHTS, plate_reference, 5),
               ("pose", POSE_WEIGHTS, pose_reference, 56)]
    manifest = {}
    for name, checkpoint, reference, channels in configs:
        reference = Path(reference)
        meta = json.loads((reference / "settings.json").read_text())
        graph = reference / f"{name}.onnx"
        if sha256(checkpoint.read_bytes()).hexdigest() != meta["weights_sha256"]:
            raise ValueError(f"{name} 베이스라인 가중치가 다릅니다.")
        if sha256(graph.read_bytes()).hexdigest() != meta["onnx_sha256"]:
            raise ValueError(f"{name} ONNX 파일이 다릅니다.")

        model = YOLO(str(checkpoint))
        model.predict(np.zeros((640, 384, 3), dtype=np.uint8), device="cpu", imgsz=640, verbose=False)
        network = deepcopy(model.predictor.model.backend.model).cpu().eval()
        for attr in ("args", "pt_path", "yaml_file", "criterion", "ckpt_path"):
            if hasattr(network, attr):
                delattr(network, attr)
        network.yaml.pop("yaml_file", None)
        path = destination / f"{name}.pt"
        torch.save(network, path)
        native = torch.load(path, weights_only=False, map_location="cpu").eval()
        x = torch.rand((1, 3, 640, 384), generator=torch.Generator().manual_seed(127))
        prediction = native(x)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
        expected, = session.run(None, {"images": x.numpy()})
        if tuple(prediction.shape) != (1, channels, 5040):
            raise ValueError(f"{name} 원시 출력 모양을 확인하세요.")
        np.testing.assert_allclose(prediction.numpy(), expected, rtol=1e-3, atol=0.04)
        shutil.copyfile(graph, destination / graph.name)
        manifest[name] = {
            "source_pt": meta["weights_sha256"], "onnx": meta["onnx_sha256"],
            "eager_pt": sha256(path.read_bytes()).hexdigest(),
            "cpu_synthetic_max_abs": float(np.max(np.abs(prediction.numpy() - expected))),
        }
    (destination / "models.json").write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(ROOT / "wlcv/evaluation/tensorrt_3090.py", destination / "benchmark.py")
    shutil.copyfile(ROOT / "wlcv/evaluation/synthetic_video_timing.py", destination / "synthetic_video.py")
    return manifest
