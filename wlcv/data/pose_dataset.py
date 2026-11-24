"""수행자 박스와 가까운 무릎·발목 직접점을 COCO 17점 라벨로 내보낸다."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT

from wlcv.annotation.knee_ankle_candidates import REVIEW_FOLDER, load_candidates
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
        box_folder = ANNOTATIONS / "pose_box_review" / role
        frames.append(_read_frame(video, frame, info.filename, info.person_id, role, side, source, box_folder))
    return frames


def _read_frame(video, frame, filename, person, role, side, source, box_folder):
    box = PerformerBox(box_folder, video, frame).saved
    if box is None:
        raise ValueError(f"{video} {frame}번의 수행자 박스가 없습니다.")
    rows = load_points(pose_point_path(source, video, frame, pass_number=1))
    points = {}
    for joint, name in [("무릎", "knee"), ("발목", "ankle")]:
        row = rows.get(joint, {})
        keys = ("video_id", "frame_index", "person_id", "split", "side", "status")
        if tuple(row.get(key) for key in keys) != (video, str(frame), person, role, side, "located"):
            raise ValueError(f"{video} {frame}번의 {joint} 직접 판독을 확인하세요.")
        points[name] = (float(row["x_px"]), float(row["y_px"]))
    subset = "train" if role == "train" else "val"
    return PoseFrame(VIDEOS / filename, frame, subset, side, box, points)


def load_finetune_frames():
    """최초 3장과 추가 판독 8장을 연결하고 예비 프레임을 별도로 반환한다."""
    groups = {"train": [], "val": [], "reserve": []}
    for sample in load_reviewed_frames():
        groups[sample.split].append(sample)
    for row in load_candidates().itertuples():
        folder = REVIEW_FOLDER / row.split
        sample = _read_frame(row.video_id, row.frame, row.filename, row.person,
                             row.split, row.side, folder, folder)
        if row.priority == "예비":
            if (row.video_id, row.frame, row.split) != ("a05_sn_f_01", 41, "validation"):
                raise ValueError("첫 파인튜닝의 예비 프레임을 확인하세요.")
            sample.split = "reserve"
        groups[sample.split].append(sample)
    if [len(group) for group in groups.values()] != [5, 5, 1]:
        raise ValueError("학습 5장, 기본 검증 5장, 예비 1장의 구성을 확인하세요.")
    stems = [sample.stem for group in groups.values() for sample in group]
    if len(stems) != len(set(stems)):
        raise ValueError("두 목록에 같은 프레임이 들어 있습니다.")
    return groups


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
    """변환한 학습·검증 프레임을 새 폴더에 보관하고 학습에 전달할 YAML 경로를 반환한다."""
    # 설치된 모델의 관절 순서와 좌우 반전 짝을 그대로 사용한다.
    coco = yaml.safe_load((ULTRALYTICS_ROOT / "cfg/datasets/coco-pose.yaml").read_text())
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder, prefix="build-") as work:
        staging = Path(work) / "dataset"
        # 실행마다 별도 폴더를 써서 이전 실험의 라벨과 학습 캐시를 보존한다.
        dataset = folder / Path(work).name.replace("build-", "dataset-", 1)
        for sample in tqdm(frames, desc="자세 라벨 변환", unit="장"):
            if sample.split not in ("train", "val"):
                raise ValueError(f"{sample.stem}은 학습·기본 검증용 프레임이 아닙니다.")
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


def check_finetune_files(data_yaml, groups):
    """Ultralytics가 읽을 두 디렉토리에 약속한 5장씩만 있는지 확인한다."""
    data_yaml = Path(data_yaml).resolve(strict=True)
    cfg = yaml.safe_load(data_yaml.read_text())
    if (Path(cfg["path"]).resolve() != data_yaml.parent or cfg["kpt_shape"] != [17, 3]
            or cfg["names"] != {0: "person"}):
        raise ValueError("이번 실험의 데이터 경로와 COCO 17점 설정을 확인하세요.")
    for split in ("train", "val"):
        if cfg[split] != f"images/{split}":
            raise ValueError(f"{split}의 이미지 경로를 확인하세요.")
        expected = {sample.stem for sample in groups[split]}
        for folder, suffix in [("images", ".png"), ("labels", ".txt")]:
            directory = data_yaml.parent / folder / split
            files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
            if files != {stem + suffix for stem in expected}:
                raise ValueError(f"{directory}의 파일이 이번 실험의 다섯 프레임과 다릅니다.")
