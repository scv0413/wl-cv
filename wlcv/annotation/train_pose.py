"""고정한 학습 후보 8장의 목록, b04 기존 판독과 저장 진행률."""

import pandas as pd

from .performer_box import PerformerBox
from .pose_points import JOINTS, _save_point, load_points, pose_point_path

# 각 선수의 이륙 후보, 무릎 통과 순서다. 프레임 번호는 0부터 센다.
FRAME_PAIRS = {"a01_sn_f_01": (5, 13), "b01_sn_f_01": (12, 23),
               "b04_sn_f_01": (9, 21), "c02_sn_f_01": (14, 26)}
STATUS_LABELS = {"located": "직접 확인", "estimated": "추정", "unreadable": "판독 불가"}


def candidate_frames(metadata, split):
    videos = pd.read_csv(metadata / "videos.csv").set_index("video_id", verify_integrity=True)
    roles = pd.read_csv(split).set_index("video_id", verify_integrity=True)
    rows = []
    for video, pair in FRAME_PAIRS.items():
        info = videos.loc[video]
        if roles.at[video, "role"] != "train" or roles.at[video, "person_id"] != info.person_id:
            raise ValueError(f"{video}의 사람별 학습 분할을 확인하세요.")
        if info.near_side != "left":
            raise ValueError(f"{video}의 가까운 쪽이 기존 판독과 다릅니다.")
        for frame, phase in zip(pair, ("이륙 후보", "무릎 통과")):
            rows.append(dict(video_id=video, person_id=info.person_id, filename=info.filename,
                             side=info.near_side, frame=frame, phase=phase))
    return pd.DataFrame(rows)


def import_b04(annotations, folder):
    """원본은 읽기만 하고, 빈 관절 항목만 채워 사용자가 수정한 판독을 보존한다."""
    video = "b04_sn_f_01"
    for frame in FRAME_PAIRS[video]:
        target = pose_point_path(folder, video, frame)
        saved = load_points(target)

        def read_source(subfolder, joint, number=1):
            path = pose_point_path(annotations / subfolder, video, frame, pass_number=number)
            row = load_points(path).get(joint)
            if not row or (row["video_id"], row["frame_index"], row["person_id"], row["side"], row["split"]) != (
                video, str(frame), "b04", "left", "train"
            ):
                raise ValueError(f"{path.name}의 {joint} 원본 판독을 확인하세요.")
            return row.copy()

        if "어깨" not in saved:
            shoulder = read_source("train_acromion_review", "어깨")
            _save_point(target, shoulder, previous=None)
        if "골반" not in saved:
            first = read_source("train_hip_review", "골반")
            second = read_source("train_hip_review", "골반", 2)
            if not all(row["status"] == "estimated" for row in (first, second)):
                raise ValueError(f"b04 {frame}번 소켓의 두 추정점을 확인하세요.")
            hip = dict(first, reason="hip_socket")
            for key in ("x_px", "y_px"):
                hip[key] = f"{(float(first[key]) + float(second[key])) / 2:g}"
            _save_point(target, hip, previous=None)


def progress_table(frames, folder):
    rows = []
    for item in frames.itertuples():
        box = PerformerBox(folder, item.video_id, item.frame)
        points = load_points(pose_point_path(folder, item.video_id, item.frame))
        row = {"선수": item.person_id, "프레임": item.frame, "시점": item.phase,
               "수행자 박스": "저장" if box.saved else "미검수"}
        for joint in JOINTS:
            row[joint] = STATUS_LABELS[points[joint]["status"]] if joint in points else "미검수"
        row["완료"] = box.saved is not None and all(joint in points for joint in JOINTS)
        rows.append(row)
    return pd.DataFrame(rows)
