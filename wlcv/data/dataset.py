"""사람별 분할표와 프레임 라벨을 YOLO 학습용 폴더로 옮긴다."""

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import yaml
from tqdm import tqdm

from .frames import frames_rgb
from .plate_labels import label_frame, read_boxes


def read_splits(split_csv: str | Path) -> dict[str, str]:
    """한 선수의 영상에는 train, validation, test 중 한 역할만 허용한다."""
    subsets = {"train": "train", "validation": "val", "test": "test"}
    people = {}
    videos = {}
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"video_id", "person_id", "role"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{split_csv}: video_id, person_id, role 열이 필요합니다.")
        for row in reader:
            video, person, role = (row[key].strip() for key in ("video_id", "person_id", "role"))
            if not video or not person or role not in subsets:
                raise ValueError(f"{split_csv} {reader.line_num}행: 영상, 사람 또는 분할 값이 잘못되었습니다.")
            if video in videos:
                raise ValueError(f"분할표에 영상이 중복되었습니다: {video}")
            if person in people and people[person] != role:
                raise ValueError(f"{person}의 영상이 여러 분할에 등록되었습니다: {people[person]}, {role}")
            people[person] = role
            videos[video] = subsets[role]
    return videos


def _signature(files: Iterable[Path], base: Path | None = None) -> str:
    """경로와 내용을 함께 비교하여 오래되었거나 중간에 바뀐 출력을 식별한다."""
    result = hashlib.sha256()
    for path in sorted(files):
        name = path.relative_to(base) if base else path
        result.update(str(name).encode() + b"\0")
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(block)
        result.update(file_hash.digest())
    return result.hexdigest()


def _output_signature(directory: Path) -> str:
    files = [directory / "data.yaml"]
    files.extend(path for path in (directory / "images").rglob("*") if path.is_file())
    files.extend((directory / "labels").rglob("*.txt"))
    return _signature(files, base=directory)


def _reuse(directory: Path, inputs: str) -> Path:
    try:
        manifest = json.loads((directory / ".dataset.json").read_text(encoding="utf-8"))
        if (
            manifest["path"] == str(directory)
            and manifest["inputs"] == inputs
            and manifest["outputs"] == _output_signature(directory)
        ):
            return directory / "data.yaml"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    raise FileExistsError(f"기존 데이터셋과 현재 입력이 일치하지 않습니다. 새 출력 폴더를 지정하세요: {directory}")


def prepare_dataset(
    video_dir: str | Path, labels_dir: str | Path, split_csv: str | Path, output_dir: str | Path
) -> Path:
    """라벨이 있는 프레임만 저장하고 Ultralytics에 전달할 data.yaml 경로를 반환한다."""
    video_dir, labels_dir, split_csv, output_dir = (
        Path(p).expanduser().resolve() for p in (video_dir, labels_dir, split_csv, output_dir)
    )
    subsets = read_splits(split_csv)
    annotations = defaultdict(dict)
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise ValueError(f"프레임 라벨(.txt)을 찾을 수 없습니다: {labels_dir}")
    for label in label_files:
        video, index = label_frame(label)
        if video not in subsets:
            raise ValueError(f"{label.name}의 영상이 분할표에 없습니다.")
        if subsets[video] == "test":
            continue
        if index in annotations[video]:
            raise ValueError(f"{video}의 {index}번 프레임에 라벨 파일이 두 개 있습니다.")
        read_boxes(label)
        annotations[video][index] = label
    if {subsets[video] for video in annotations} != {"train", "val"}:
        raise ValueError("학습과 검증에 각각 한 장 이상의 라벨 프레임이 필요합니다.")

    videos = {}
    for path in video_dir.iterdir():
        if path.stem in annotations and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            if path.stem in videos:
                raise ValueError(f"같은 영상 ID의 파일이 두 개입니다: {path.stem}")
            videos[path.stem] = path
    missing = sorted(annotations.keys() - videos.keys())
    if missing:
        raise FileNotFoundError(f"{video_dir}에 원본 영상이 없습니다: {', '.join(missing)}")

    used_labels = [label for frames in annotations.values() for label in frames.values()]
    inputs = _signature([split_csv, *used_labels, *videos.values()])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        return _reuse(output_dir, inputs)

    # 완성된 폴더만 최종 이름으로 옮긴다. 디코딩 실패나 중단 시 임시 폴더는 자동 정리된다.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=".wlcv-") as temporary:
        build = Path(temporary) / "dataset"
        for subset in ("train", "val"):
            for kind in ("images", "labels"):
                (build / kind / subset).mkdir(parents=True)
        with tqdm(total=len(used_labels), desc="라벨 프레임 추출", unit="장") as progress:
            for video, targets in annotations.items():
                subset = subsets[video]
                last = max(targets)
                with closing(frames_rgb(videos[video])) as frames:
                    for index, rgb in enumerate(frames):
                        if index not in targets:
                            continue
                        label = targets[index]
                        image = build / "images" / subset / (label.stem + ".jpg")
                        if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                            raise OSError(f"이미지를 저장할 수 없습니다: {image}")
                        (build / "labels" / subset / label.name).write_bytes(label.read_bytes())
                        progress.update()
                        if index == last:
                            break
                    else:
                        raise IndexError(f"{video}: {last}번 라벨 프레임에 도달하기 전에 영상이 끝났습니다.")

        config = {"path": str(output_dir), "train": "images/train", "val": "images/val", "names": {0: "plate"}}
        (build / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest = {"path": str(output_dir), "inputs": inputs, "outputs": _output_signature(build)}
        (build / ".dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            build.rename(output_dir)
        except OSError:
            if output_dir.is_dir() and any(output_dir.iterdir()):
                return _reuse(output_dir, inputs)
            raise
    return output_dir / "data.yaml"
