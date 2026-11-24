"""추가 판독할 무릎·발목 후보를 읽고 사람이 저장한 박스와 관절 상태를 확인한다."""

import pandas as pd

from wlcv.annotation.performer_box import PerformerBox
from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.annotation.storage import load_events
from wlcv.annotation.train_pose import STATUS_LABELS
from wlcv.config import ANNOTATIONS, ROOT, SPLIT

JOINTS = ("무릎", "발목")
REVIEW_FOLDER = ANNOTATIONS / "knee_ankle_candidates"


def load_candidates():
    """선수·분할·가까운 방향을 기존 표와 연결한다. 프레임은 0번부터 센다."""
    metadata = ROOT / "data/metadata"
    candidates = pd.read_csv(metadata / "knee_ankle_candidates.csv", keep_default_na=False)
    roles = pd.read_csv(SPLIT).set_index("video_id", verify_integrity=True)
    videos = pd.read_csv(metadata / "videos.csv").set_index("video_id", verify_integrity=True)
    if roles.groupby("person_id").role.nunique().gt(1).any():
        raise ValueError("한 사람이 학습과 검증에 함께 있습니다. 사람별 분할표를 확인하세요.")
    if candidates.duplicated(["video_id", "frame"]).any():
        raise ValueError("후보 목록에서 같은 영상과 프레임이 반복됩니다.")

    rows = []
    for row in candidates.itertuples(index=False):
        info, role = videos.loc[row.video_id], roles.loc[row.video_id]
        if (role.person_id, role.role) != (info.person_id, row.split):
            raise ValueError(f"{row.video_id}의 선수 또는 분할이 기존 표와 다릅니다.")
        if info.near_side not in ("left", "right") or not 0 <= row.frame < info.frame_count:
            raise ValueError(f"{row.video_id}의 가까운 방향과 프레임 번호를 확인하세요.")
        basis = "원본 영상 육안 확인"
        if row.anchor:
            events = load_events(ANNOTATIONS, row.video_id, pass_number=1)
            offset = int(row.offset)
            if row.anchor not in events or events[row.anchor] + offset != row.frame:
                raise ValueError(f"{row.video_id} {row.frame}번의 1차 동작 시점 라벨이 달라졌습니다.")
            basis = f"1차 {row.anchor} {events[row.anchor]}번"
            if offset:
                basis += f" + {offset}프레임"
        rows.append(dict(video_id=row.video_id, person=info.person_id, filename=info.filename,
                         frame=row.frame, split=role.role, side=info.near_side, phase=row.phase,
                         basis=basis, priority=row.priority, cue=row.cue))
    return pd.DataFrame(rows)


def review_status(frames, folder=REVIEW_FOLDER):
    """박스와 두 관절의 저장 상태를 읽고 직접점이 모두 박스 안에 있는지 센다."""
    rows = []
    for frame in frames.itertuples():
        dest = folder / frame.split
        box = PerformerBox(dest, frame.video_id, frame.frame).saved
        points = load_points(pose_point_path(dest, frame.video_id, frame.frame))
        result = {"분할": frame.split, "선수": frame.person, "프레임": frame.frame,
                  "수행자 박스": "저장" if box else "미검수"}
        for joint in JOINTS:
            point = points.get(joint)
            result[joint] = STATUS_LABELS[point["status"]] if point else "미검수"
        result["판독 완료"] = box is not None and all(joint in points for joint in JOINTS)
        result["두 직접점 모두 박스 안"] = "—"
        if box and all(result[joint] == "직접 확인" for joint in JOINTS):
            left, top, right, bottom = box
            inside = all(left <= float(points[j]["x_px"]) <= right and
                         top <= float(points[j]["y_px"]) <= bottom for j in JOINTS)
            result["두 직접점 모두 박스 안"] = "예" if inside else "박스·점 재확인"
        rows.append(result)
    return pd.DataFrame(rows)
