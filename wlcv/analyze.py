"""검증 영상 ID 또는 MP4 경로로 원판과 가까운 쪽 관절을 분석한다. 프레임 번호는 0부터 센다."""
import argparse
from pathlib import Path

from wlcv.data.mp4_input import EVENT_KEYS
from wlcv.runtime.measurements import coaching_summary as coaching_summary, phase_summary as phase_summary
from wlcv.runtime.overlay import render as render, render_frames as render_frames, two_pass_frames as two_pass_frames
from wlcv.runtime.pipeline import analyze_file
from wlcv.runtime.recovery import save_recovery_curve as save_recovery_curve


def read_phases(video):
    from wlcv.evaluation.registered_video import read_phases as read_registered_phases
    return read_registered_phases(video)


def analyze(video, *, device="cpu", read_mode="single-pass", plate_imgsz=640, plate_mode="single",
            near_side=None, facing=None, manual_events=None):
    """기존 CLI와 노트북에서 사용하던 파일 경로·검증 영상 ID를 모두 지원한다."""
    options = dict(device=device, read_mode=read_mode, plate_imgsz=plate_imgsz, plate_mode=plate_mode)
    if isinstance(video, Path) or Path(video).suffix or "/" in str(video):
        return analyze_file(video, near_side=near_side, facing=facing, manual_events=manual_events, **options)
    if near_side is not None or facing is not None or manual_events:
        raise ValueError("방향·시점을 직접 지정하려면 영상 ID 대신 MP4 경로를 입력하세요.")
    from wlcv.evaluation.registered_video import analyze_registered
    return analyze_registered(video, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="검증 영상 ID 또는 로컬 MP4 파일 경로")
    parser.add_argument("--near-side", choices=("left", "right"), help="카메라에 가까운 쪽 (선수의 좌우)")
    parser.add_argument("--facing", choices=("left", "right"), help="선수가 보는 방향 (화면의 좌우)")
    for key, help_text in zip(EVENT_KEYS, ("이륙", "무릎 통과", "접촉", "캐치", "피니시")):
        parser.add_argument("--" + key.replace("_", "-"), type=int, metavar="FRAME",
                            help=f"{help_text} 수동 시점 (0부터, 선택)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="원판·자세 추론 장치")
    parser.add_argument("--read-mode", choices=("two-pass", "single-pass", "single-shared"),
                        default="single-pass", help="원본 읽기 횟수와 모델 전처리 공유 여부")
    parser.add_argument("--plate-imgsz", type=int, default=640, help="원판 모델 입력 긴 변 (자세는 640)")
    parser.add_argument("--plate-mode", choices=("single", "cutmix"), default="single",
                        help="원판 베이스라인 단독 또는 빈 출력을 CutMix로 보완")
    args = parser.parse_args()
    events = {key: getattr(args, key) for key in EVENT_KEYS if getattr(args, key) is not None}
    try:
        output, summary = analyze(args.source, device=args.device, read_mode=args.read_mode,
                                  plate_imgsz=args.plate_imgsz, plate_mode=args.plate_mode,
                                  near_side=args.near_side, facing=args.facing, manual_events=events)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    interval = summary["manual_phase_interval"]
    print(f"전체 {summary['frame_count']}프레임 · {summary['plate_counts']}")
    if interval["rise_px"] is None:
        print("이륙→무릎 통과 상승량 보류 ·", interval.get("reason") or f"원판 미탐 프레임 {interval['missing_endpoint_frames']}")
    else:
        print(f"수동 이륙 {interval['lift_off_frame']} → 무릎 통과 {interval['knee_pass_frame']}: "
              f"원판 중심 {interval['rise_px']:+.2f} px (위쪽이 +)")
    measurement = summary["knee_to_contact_rise"]
    if measurement["rise_px"] is None:
        print("무릎 통과→접촉 상승량 보류 ·", measurement["reason"])
    else:
        source = "코사인 직접 검출" if measurement["source"] == "cosine" else "기존 추적 좌표"
        print(f"무릎 통과→접촉 상승량 {measurement['rise_px']:+.2f} px · {source}", measurement["reason"])
    pulls = summary["coaching"]["pulls"]
    if pulls.get("ratio") is not None:
        print(f"이륙→무릎 {pulls['early']:.2f} px/s · 무릎→접촉 {pulls['late']:.2f} px/s · 앞/뒤 {pulls['ratio']:.3f}")
    else:
        if pulls.get("late") is not None:
            print(f"무릎→접촉 평균 수직속도 {pulls['late']:.2f} px/s")
        print("속도 비율 보류 ·", pulls["reason"])
    peak = summary["coaching"]["drive_to_peak"]
    if peak["peak_frame"] is None:
        print("접촉→최고점 산출 보류 ·", peak["reason"])
    else:
        print(f"접촉→최고점 {peak['peak_frame']}번 · 상승량 {peak['rise_px']:.2f} px · "
              f"평균 수직속도 {peak['mean_speed_px_s']:,.0f} px/s")
        bounds = peak["speed_range_px_s"]
        if bounds is None:
            print("±1프레임 범위 보류 ·", peak["range_reason"])
        else:
            print(f"±1프레임 속도 범위 {bounds[0]:,.0f}~{bounds[1]:,.0f} px/s")
    catch = summary["coaching"]["catch"]
    if catch.get("forward_px") is not None:
        print(f"캐치 바−어깨 · 앞 {catch['forward_px']:+.2f} px · 위 {catch['up_px']:+.2f} px (모델)")
    else:
        print("캐치 위치 ·", catch["reason"])
    print(f"로컬 결과: {output}")


if __name__ == "__main__":
    main()
