"""57번 검증 입력을 만든다. 선수와 프레임 번호를 연결하는 표는 Mac에 보관한다."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import onnx
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from wlcv.annotation.storage import events_for, load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT, SPLIT, VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.data.plate_labels import label_frame
from wlcv.evaluation import onnx_plate, onnx_pose
from wlcv.evaluation.plate_size_mps import B02
from wlcv.evaluation.runtime_map import validation_photos

PLATE_REF = ROOT / "outputs/onnx_plate_parity/20261125-194508-519898"
POSE_REF = ROOT / "outputs/onnx_pose_parity/20261125-201352-304100"
MAP_REF = ROOT / "outputs/runtime_quality/20261125-214200-963305"
HUB_REF = ROOT / "outputs/plate_size_mps/20261125-205647-451535"
HUB_FRAMES = (20, *range(30, 37))


def source_image(row):
    if row.source == "jpeg":
        image = cv2.imread(
            str(DATASET / "images/val" / f"{row.video}_f{row.frame}.jpg")
        )
        if image is None:
            raise ValueError(f"검증 사진을 읽을 수 없습니다: {row.key}")
        return image
    return read_rgb(VIDEOS / f"{row.video}.mp4", row.frame)[:, :, ::-1].copy()


def prepare(folder):
    """프레임과 파일 검사가 모두 끝나면 완성된 입력 폴더를 한 번에 저장한다."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"기존 실험 입력을 보존합니다: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as temporary:
        working = Path(temporary) / "complete"
        working.mkdir()
        manifest = _prepare_files(working)
        working.rename(folder)
    return manifest


@torch.inference_mode()
def _prepare_files(folder):
    payload = folder / "gpu_input"
    payload.mkdir()
    photos, _ = validation_photos()
    split = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    pose = pd.read_csv(POSE_REF / "frames.csv")
    if (
        len(pose) != 12
        or pose.person.nunique() != 6
        or pose.duplicated(["video", "frame"]).any()
    ):
        raise ValueError("47번 자세 검증 6명의 12장을 확인하세요.")
    poses = set(zip(pose.video, pose.frame))
    hubs = {(B02, n) for n in HUB_FRAMES}
    sources = [("jpeg", *label_frame(path)) for path in photos]
    sources += [("video", video, frame) for video, frame in sorted(poses | hubs)]
    meta = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    rows = []
    tensors = {}
    jobs = {"plate": [], "pose": []}
    reader = YOLO(str(onnx_plate.WEIGHTS))
    reader.predict(str(photos[0]), device="cpu", imgsz=640, verbose=False)
    for index, (source, video, frame) in enumerate(
        tqdm(sources, desc="검증 입력", unit="장")
    ):
        assert split.at[video, "role"] == "validation"
        key = f"input{index:03d}"
        phase = load_events(ANNOTATIONS, video, 1)
        end = phase.get(events_for(video)[-1])
        active = (
            phase["ready"] <= frame <= end
            if end is not None and "ready" in phase
            else None
        )
        row = dict(
            key=key,
            source=source,
            video=video,
            person=split.at[video, "person_id"],
            frame=int(frame),
            side=meta.at[video, "near_side"],
            active=active,
        )
        bgr = source_image(pd.Series(row))
        x = reader.predictor.preprocess([bgr]).cpu().numpy()
        if x.shape != (1, 3, 640, 384) or x.dtype != np.float32:
            raise ValueError(f"입력 크기 또는 자료형이 다릅니다: {key}")
        rgb8 = np.rint(x * 255).astype(np.uint8)
        assert np.array_equal(rgb8.astype(np.float32) / 255, x)
        tensors[key] = rgb8
        row.update(height=bgr.shape[0], width=bgr.shape[1])
        rows.append(row)
        if source == "jpeg" or (video, frame) in hubs:
            jobs["plate"].append(key)
        if source == "video" and (video, frame) in poses:
            jobs["pose"].append(key)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(folder / "local_frames.csv", index=False)
    if (len(jobs["plate"]), len(jobs["pose"])) != (47, 12):
        raise ValueError("원판 39장·허브 8장·자세 12장의 입력 매핑을 확인하세요.")
    inp = payload / "inputs.npz"
    np.savez_compressed(inp, **tensors)
    hashes = {}
    for name, ref, weights in (
        ("plate", PLATE_REF, onnx_plate.WEIGHTS),
        ("pose", POSE_REF, onnx_pose.WEIGHTS),
    ):
        graph = ref / f"{name}.onnx"
        settings = json.loads((ref / "settings.json").read_text())
        hashes[name] = sha256(graph.read_bytes()).hexdigest()
        if (
            hashes[name] != settings["onnx_sha256"]
            or sha256(weights.read_bytes()).hexdigest() != settings["weights_sha256"]
        ):
            raise ValueError(f"{name} 기준 모델의 체크섬을 확인하세요.")
        exported = onnx.load(graph)
        graph_bytes = exported.graph.SerializeToString()
        for item in exported.metadata_props:
            if item.key == "description":
                item.value = f"{name} inference model"
        target = payload / graph.name
        onnx.save(exported, target)
        if onnx.load(target).graph.SerializeToString() != graph_bytes:
            raise ValueError("내보낸 모델의 연산 그래프 또는 가중치가 달라졌습니다.")
        hashes[name] = sha256(target.read_bytes()).hexdigest()
    spec = {
        "jobs": jobs,
        "graph_sha256": hashes,
        "input_sha256": sha256(inp.read_bytes()).hexdigest(),
    }
    (payload / "jobs.json").write_text(json.dumps(spec, indent=2))
    shutil.copyfile(
        ROOT / "wlcv/evaluation/tensorrt_3090.py", payload / "tensorrt_3090.py"
    )
    shutil.copyfile(
        ROOT / "wlcv/evaluation/precision_gpu.py", payload / "precision_gpu.py"
    )
    return manifest
