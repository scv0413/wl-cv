"""세 장의 수행자 박스와 가까운 무릎·발목 직접점을 COCO 17점 라벨로 내보낸다."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT

from wlcv.annotation.performer_box import PerformerBox
from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.config import ANNOTATIONS, ROOT, SPLIT, VIDEOS
from wlcv.data.frames import read_rgb


@dataclass
class PoseFrame:
    video: Path
    frame: int
    split: str
    side: str
    box: tuple
    points: dict

    @property
    def stem(self):
        return f"{self.video.stem}_frame_{self.frame:04d}"


def load_reviewed_frames():
    """기존 사람별 분할과 직접 판독이 일치하는 세 장만 읽는다."""
    videos = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id", verify_integrity=True)
    roles = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    if roles.groupby("person_id").role.nunique().gt(1).any():
        raise ValueError("한 사람이 두 분할에 들어 있습니다. 사람별 분할표를 확인하세요.")

    scenes = [
        ("a08_sn_f_01", 29, "train", "right", ANNOTATIONS / "a08_right_knee_ankle"),
        ("b02_sn_h_02", 36, "validation", "right", ANNOTATIONS),
        ("c04_sn_f_01", 78, "validation", "left", ANNOTATIONS),
    ]
    frames = []
    for video, frame, role, side, source in scenes:
        info = videos.loc[video]
        split = roles.loc[video]
        if (split.role, split.person_id, info.near_side) != (role, info.person_id, side):
            raise ValueError(f"{video}의 분할, 수행자, 방향을 확인하세요.")
        box = PerformerBox(ANNOTATIONS / "pose_box_review" / role, video, frame).saved
        if box is None:
            raise ValueError(f"{video} {frame}번의 수행자 박스가 없습니다.")
        rows = load_points(pose_point_path(source, video, frame, pass_number=1))
        points = {}
        for joint, name in [("무릎", "knee"), ("발목", "ankle")]:
            row = rows.get(joint, {})
            keys = ("video_id", "frame_index", "person_id", "split", "side", "status")
            if tuple(row.get(key) for key in keys) != (video, str(frame), info.person_id, role, side, "located"):
                raise ValueError(f"{video} {frame}번의 {joint} 직접 판독을 확인하세요.")
            points[name] = (float(row["x_px"]), float(row["y_px"]))
        subset = "train" if role == "train" else "val"
        frames.append(PoseFrame(VIDEOS / info.filename, frame, subset, side, box, points))
    return frames


def encode_pose(frame, width, height, joint_names):
    """픽셀 모서리를 정규화한 중심·크기로, 직접점을 (x, y, 2)로 바꾼다."""
    left, top, right, bottom = frame.box
    if not (0 <= left < right < width and 0 <= top < bottom < height):
        raise ValueError(f"{frame.stem}의 박스가 원본 크기를 벗어납니다.")
    box = [(left + right) / (2 * width), (top + bottom) / (2 * height),
           (right - left) / width, (bottom - top) / height]

    keypoints = np.zeros((17, 3))
    for joint, (x, y) in frame.points.items():
        if not (left <= x <= right and top <= y <= bottom):
            raise ValueError(f"{frame.stem}의 {joint} 좌표가 수행자 박스를 벗어납니다.")
        index = joint_names.index(f"{frame.side}_{joint}")
        keypoints[index] = [x / width, y / height, 2]
    fields = ["0", *(f"{value:.8f}" for value in box)]
    for x, y, visible in keypoints:
        if visible:
            fields.extend([f"{x:.8f}", f"{y:.8f}", "2"])
        else:
            fields.extend(["0", "0", "0"])
    return " ".join(fields) + "\n"


def export_pose_dataset(frames, folder):
    """변환을 마친 세 장을 새 폴더에 보관하고 학습에 전달할 YAML 경로를 반환한다."""
    # 설치된 모델의 관절 순서와 좌우 반전 짝을 그대로 사용한다.
    coco = yaml.safe_load((ULTRALYTICS_ROOT / "cfg/datasets/coco-pose.yaml").read_text())
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder, prefix="build-") as work:
        staging = Path(work) / "dataset"
        # 실행마다 별도 폴더를 써서 이전 실험의 라벨과 학습 캐시를 보존한다.
        dataset = folder / Path(work).name.replace("build-", "dataset-", 1)
        for sample in tqdm(frames, desc="자세 라벨 변환", unit="장"):
            rgb = read_rgb(sample.video, sample.frame)
            height, width = rgb.shape[:2]
            label = encode_pose(sample, width, height, coco["kpt_names"][0])
            images = staging / "images" / sample.split
            labels = staging / "labels" / sample.split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(images / f"{sample.stem}.png")
            (labels / f"{sample.stem}.txt").write_text(label, encoding="utf-8")

        config = {"path": str(dataset), "train": "images/train", "val": "images/val"}
        for key in ("names", "kpt_shape", "flip_idx", "kpt_names"):
            config[key] = coco[key]
        (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        staging.rename(dataset)
    return dataset / "data.yaml"
