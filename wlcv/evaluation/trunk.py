"""수동 어깨와 골반을 이은 선이 화면 수직선에서 얼마나 기울었는지 비교한다."""

import matplotlib.pyplot as plt

from wlcv.annotation.pose_points import load_points, pose_point_path
from wlcv.annotation.trunk import pair_path, read_pair
from wlcv.data.frames import read_rgb
from wlcv.runtime.measurements import vertical_angle


def point_label(point):
    if point is None:
        return "미판독"
    if point["status"] == "located":
        return "직접 확인"
    if point["status"] == "unreadable":
        return "판독 불가"
    return {"clothing": "옷 가림 추정", "hip_socket": "소켓 위치 추정",
            "occluded": "기타 가림 추정"}[point["reason"]]


def frame_angle(points):
    """위쪽 수직선을 0도로 둔다. 좌우 어느 쪽을 보는 선수에게나 같은 기준이다."""
    shoulder, hip = points.get("어깨"), points.get("골반")
    if any(p is None or p["status"] == "unreadable" for p in (shoulder, hip)):
        return None
    return vertical_angle((int(shoulder["x_px"]), int(shoulder["y_px"])),
                          (int(hip["x_px"]), int(hip["y_px"])))


def review_trunk(video, info, folder, *, pass_number=1):
    """두 시점의 라벨을 다시 읽는다. 추정점이 하나라도 있으면 변화를 참고 열에만 넣는다."""
    pair = read_pair(pair_path(folder, video))
    row = {"영상": video, "가까운 쪽": info["near_side"], "판독": "시점 미확인",
           "이륙 프레임": None, "무릎 통과 프레임": None,
           "이륙 어깨": None, "이륙 골반": None, "무릎 통과 어깨": None, "무릎 통과 골반": None,
           "이륙 (°)": None, "무릎 통과 (°)": None, "직접 변화 (°)": None, "참고 변화 (°)": None}
    if not pair or pair["판독"] != "확인":
        if pair:
            row["판독"] = "시점 판독 불가"
        return row, {}

    details = {}
    states = []
    for phase in ("이륙", "무릎 통과"):
        frame = int(pair[phase])
        points = load_points(pose_point_path(folder, video, frame, pass_number=pass_number))
        for joint in ("어깨", "골반"):
            p = points.get(joint)
            if p and (p["video_id"] != video or int(p["frame_index"]) != frame
                      or p["person_id"] != info["person_id"] or p["split"] != "validation"
                      or p["side"] != info["near_side"]):
                raise ValueError(f"{video} {frame}번 {joint}의 판독 정보를 확인하세요.")
            row[f"{phase} {joint}"] = point_label(p)
            states.append(p["status"] if p else "missing")
        row[f"{phase} 프레임"] = frame
        row[f"{phase} (°)"] = frame_angle(points)
        details[phase] = points

    if "unreadable" in states:
        row["판독"] = "관절 판독 불가"
    elif "missing" in states:
        row["판독"] = "관절 미판독"
    elif row["이륙 (°)"] is None or row["무릎 통과 (°)"] is None:
        row["판독"] = "두 점 겹침"
    else:
        estimated = "estimated" in states
        row["판독"] = "추정 포함" if estimated else "직접 확인"
        column = "참고 변화 (°)" if estimated else "직접 변화 (°)"
        row[column] = row["무릎 통과 (°)"] - row["이륙 (°)"]
    return row, details


def plot_trunk_pair(video_path, row, details):
    """원본 위에 수직 기준선, 체간 선과 직접점(+)·추정점(마름모)을 표시한다."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    for ax, phase, title in zip(axes, ("이륙", "무릎 통과"), ("Lift-off", "Knee pass")):
        frame = row[f"{phase} 프레임"]
        ax.imshow(read_rgb(video_path, frame))
        points = details[phase]
        for joint in ("어깨", "골반"):
            p = points.get(joint)
            if p and p["status"] != "unreadable":
                marker = "D" if p["status"] == "estimated" else "+"
                ax.plot(int(p["x_px"]), int(p["y_px"]), marker=marker, markersize=10,
                        color="#ffd34d", markerfacecolor="none", markeredgewidth=2, zorder=3)
        angle = row[f"{phase} (°)"]
        value = "unavailable"
        if angle is not None:
            hip, shoulder = points["골반"], points["어깨"]
            xs = [int(p["x_px"]) for p in (hip, shoulder)]
            ys = [int(p["y_px"]) for p in (hip, shoulder)]
            length = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
            ax.plot([xs[0], xs[0]], [ys[0], ys[0] - length], "w:", linewidth=2)
            estimated = any(p["status"] == "estimated" for p in (hip, shoulder))
            ax.plot(xs, ys, "--" if estimated else "-", color="#00cdd9", linewidth=2)
            value = f"{angle:.1f}°" + (" (estimate)" if estimated else "")
        ax.set_title(f"{title} / frame {frame} / {value}")
        ax.axis("off")
    fig.suptitle(f"{row['영상']} | {row['가까운 쪽']} side")
    return fig
