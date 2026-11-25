"""같은 12프레임을 섞어 익명 번호로 2차 판독한다."""

from random import Random

import ipywidgets as widgets

from wlcv.annotation.pose_points import load_points, pose_point_editor, pose_point_path
from wlcv.annotation.trunk import pair_path, read_pair


# 21번 실험에서 트레이너가 확정한 이륙·무릎 통과. 두 번째 판독에서도 시점을 고정한다.
FIRST_PAIRS = {"a03_sn_f_01": (2, 19), "a05_sn_f_01": (4, 18), "b02_sn_h_02": (20, 30),
               "b03_sn_f_01": (8, 24), "c04_sn_f_01": (11, 27), "c08_sn_f_01": (10, 24)}
JOINTS = ("어깨", "골반")


def repeat_frames(videos, folder):
    frames = []
    for video, (start, end) in FIRST_PAIRS.items():
        saved = read_pair(pair_path(folder, video))
        if (video not in videos.index or not saved or saved["판독"] != "확인"
                or (int(saved["이륙"]), int(saved["무릎 통과"])) != (start, end)):
            raise ValueError("21번 실험의 검증 영상과 확정한 두 시점을 확인하세요.")
        frames.extend([(video, start), (video, end)])
    # 고정된 순서이므로 다시 열어도 익명 번호가 같은 프레임을 가리킨다.
    Random(1).shuffle(frames)
    return frames


def pending_frames(frames, folder):
    remaining = []
    for index, (video, frame) in enumerate(frames):
        points = load_points(pose_point_path(folder, video, frame, pass_number=2))
        if any(joint not in points for joint in JOINTS):
            remaining.append(index)
    return remaining


def blind_editor(frames, videos, folder, video_dir):
    """익명 번호와 가까운 쪽만 표시한다. 중단했다면 첫 미완료 프레임에서 연다."""
    left = pending_frames(frames, folder)
    picker = widgets.Dropdown(options=[(f"프레임 {i+1:02d}", i) for i in range(len(frames))],
                              value=left[0] if left else 0, description="판독")
    advance = widgets.Button(description="다음 미완료")
    progress = widgets.Label()
    panel = widgets.VBox()

    def open_frame(*_):
        index = picker.value
        video, frame = frames[index]
        info = videos.loc[video]
        editor = pose_point_editor(video_dir / info.filename, frame, info.person_id, info.near_side,
                                   annotations=folder, split="validation", pass_number=2,
                                   joints=JOINTS, trunk_review=True, frame_label=f"프레임 {index+1:02d}")
        panel.children = (editor,)
        progress.value = f"저장 완료 {len(frames) - len(pending_frames(frames, folder))}/{len(frames)}장"

    def next_pending(_):
        left = pending_frames(frames, folder)
        if not left:
            progress.value = f"{len(frames)}장 판독 완료 · 마지막 셀을 실행하세요."
        elif picker.value in left:
            progress.value = "현재 프레임의 어깨와 골반을 모두 기록하세요. 판독 불가도 저장됩니다."
        else:
            picker.value = left[0]

    picker.observe(open_frame, names="value")
    advance.on_click(next_pending)
    open_frame()
    return widgets.VBox([widgets.HBox([picker, advance]), progress, panel])
