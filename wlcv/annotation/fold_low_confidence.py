"""빈 쌍 목록과 별개로, 원하는 경우 새 원본 다섯 장에서 허브를 확인한다."""
import pandas as pd

from wlcv.annotation.fold_hubs import fold_hub_editor, review_status
from wlcv.annotation.hub_points import read_hubs
from wlcv.config import ANNOTATIONS
from wlcv.evaluation.plate_track import KEY

PAIRED = "요청한 짝"
SPOT = "선택 검수"


def load_queue(folder, mode=PAIRED):
    if mode == PAIRED:
        return pd.read_csv(folder / "blind_queue.csv")
    if mode != SPOT:
        raise ValueError("판독 모드를 확인하세요.")
    # 수동 시점만으로 구간 중간에 가까운 새 프레임을 고른다. 점수·오차 순위는 쓰지 않는다.
    frames = pd.read_csv(folder / "frames.csv", usecols=KEY+["fold", "person_id", "phase"])
    targets = []
    for video, group in frames[frames.phase.eq("transition-drive")].groupby("video_id"):
        old = ANNOTATIONS / "plate_fold_hubs" / f"{video}.csv"
        previous = pd.read_csv(old).frame_index if old.exists() else []
        new = group[~group.frame_index.isin(previous)]
        if not new.empty:
            middle = (group.frame_index.min() + group.frame_index.max()) / 2
            targets.append(new.loc[(new.frame_index-middle).abs().idxmin()])
    queue = pd.DataFrame(targets, columns=frames.columns).drop(columns="phase")
    return queue.sample(frac=1, random_state=53).reset_index(drop=True)


def annotation_dir(folder, mode):
    subfolder = {PAIRED: "paired", SPOT: "spot_check"}[mode]
    return ANNOTATIONS / "fold_low_confidence_hubs" / folder.name / subfolder


def editor(folder, mode=PAIRED):
    queue = load_queue(folder, mode)
    if queue.empty:
        print("현재 기준을 충족하는 짝은 0쌍입니다. 이 모드에서 찍을 프레임은 없습니다.")
        return
    return fold_hub_editor(queue, annotation_dir(folder, mode))


def show_after_review(folder, mode):
    queue = load_queue(folder, mode)
    if queue.empty:
        print("이 모드의 판독 대상은 0장입니다.")
        return
    directory = annotation_dir(folder, mode)
    status = review_status(queue, directory)
    done = int(status.status.ne("미판독").sum())
    if done != len(queue):
        print(f"{done}/{len(queue)}장 저장. 모든 판독이 끝나면 원판 예측을 열 수 있습니다.")
        return
    parts = [read_hubs(directory / f"{v}.csv", frames=g.frame_index, video=v)
             for v, g in queue.groupby("video_id")]
    points = pd.concat(parts).merge(pd.read_csv(folder / "screen.csv"), on=KEY, validate="one_to_one")
    points["center_minus_hub_y_px"] = (points.cy - points.manual_y).where(
        points.visibility.eq("visible") & points.source.ne("missing") & points.reason.ne("box_mismatch"))
    if mode == PAIRED:
        pairs = pd.read_csv(folder / "pairs.csv").dropna(subset=["control_frame"]).reset_index(names="pair")
        groups = pairs.melt(id_vars=["video_id", "pair"], value_vars=["case_frame", "control_frame"],
                            var_name="role", value_name="frame_index")
        points = groups.merge(points, on=KEY, validate="one_to_one").sort_values(["pair", "role"])
    columns = (["pair", "role"] if mode == PAIRED else []) + KEY + [
        "visibility", "manual_x", "manual_y", "source", "confidence", "iou", "cy", "center_minus_hub_y_px"]
    return points[columns].round(3)
