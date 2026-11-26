"""내부 검증 선수의 원본을 처음부터 읽고 95에폭 원판 궤적을 보존한다."""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from wlcv.annotation.fold_hubs import FOLDS
from wlcv.annotation.storage import events_for, load_events
from wlcv.config import ANNOTATIONS, ROOT, SPLIT
from wlcv.evaluation.full_video import detect_videos
from wlcv.evaluation.plate_track import KEY, select_centers
from wlcv.training.plate_epoch_pair import completed_weights
from wlcv.training.plate_folds import OUTPUT, run_dir

PARENT = ROOT / "outputs/fold_hub_screen"
BACKGROUND = {"b05_sn_f_01", "b06_sn_f_01"}


def motion_phases(video, count):
    """ready와 마지막 이벤트가 확인된 범위만 쓴다. 경계 프레임은 시작하는 구간에 넣는다."""
    events = load_events(ANNOTATIONS, video, 1)
    labels = events_for(video)
    phase = ["unknown"] * count
    if video in BACKGROUND:
        return ["background"] * count
    if not all(name in events for name in labels):
        return phase
    indices = [events[name] for name in labels]
    if indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= count:
        raise ValueError(f"{video}의 수동 시점 순서를 확인하세요.")
    phase = ["outside"] * count
    for name, next_name in zip(labels, labels[1:]):
        start, stop = events[name], events[next_name]
        if next_name == labels[-1]:
            stop += 1
        phase[start:stop] = [f"{name}-{next_name}"] * (stop-start)
    return phase


def scan_originals(device="mps"):
    people = pd.read_csv(FOLDS)[["person_id", "heldout_fold"]].drop_duplicates()
    original = pd.read_csv(SPLIT)
    videos = original.merge(people, on="person_id", validate="many_to_one")
    if not videos.role.eq("train").all() or videos.video_id.duplicated().any():
        raise ValueError("기존 검증 선수가 포함됐거나 영상이 중복됐습니다.")
    PARENT.mkdir(parents=True, exist_ok=True)
    destination = PARENT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with TemporaryDirectory(dir=PARENT) as tmp:
        work = Path(tmp) / "screen"
        work.mkdir()
        tables, models = [], []
        for fold, group in videos.groupby("heldout_fold"):
            run = run_dir(fold)
            data = OUTPUT / f"fold_{fold}/dataset/data.yaml"
            weights = completed_weights(run, data=data)[95]
            trained = set(pd.read_csv(run / "train_people.csv").person_id)
            split = pd.read_csv(OUTPUT / f"fold_{fold}/split.csv")
            heldout = set(group.person_id)
            if (trained != set(split.loc[split.role.eq("train"), "person_id"])
                    or heldout != set(split.loc[split.role.eq("validation"), "person_id"])
                    or trained & heldout):
                raise ValueError(f"fold {fold}의 학습·검증 선수를 확인하세요.")
            # 원래 분할은 그대로 두고, 같은 검증 선수의 다른 시기도 이 회차에서만 읽는다.
            expanded = group[["video_id", "person_id"]].assign(role="validation")
            path = work / f"fold_{fold}_videos.csv"
            expanded.to_csv(path, index=False)
            frames, candidates = detect_videos(weights, group.video_id, split_csv=path, device=device)
            tracks = select_centers(frames, candidates, distance_ratio=.5)
            selected = tracks[tracks.method.eq("Past link")].drop(columns="method").copy()
            selected = selected.merge(expanded[["video_id", "person_id"]], on="video_id", validate="many_to_one")
            for video, rows in selected.groupby("video_id"):
                if rows.frame_index.tolist() != list(range(len(rows))):
                    raise ValueError(f"{video} 원본 프레임 번호가 누락됐습니다.")
                selected.loc[rows.index, "phase"] = motion_phases(video, len(rows))
            tables.append(selected.assign(fold=fold))
            models.append(dict(fold=fold, checkpoint=str(weights.relative_to(ROOT)), epoch=95,
                               sha256=sha256(weights.read_bytes()).hexdigest(), device=device,
                               trained=" ".join(sorted(trained)), heldout=" ".join(sorted(heldout))))
        frames = pd.concat(tables, ignore_index=True)
        if frames.duplicated(KEY).any():
            raise ValueError("같은 원본이 여러 fold에 포함됐습니다.")
        frames.to_csv(work / "frames.csv", index=False)
        pd.DataFrame(models).to_csv(work / "models.csv", index=False)
        work.rename(destination)
    return destination


if __name__ == "__main__":
    print(scan_originals())
