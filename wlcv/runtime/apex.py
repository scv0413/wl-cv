"""접촉부터 캐치 직전까지 관측한 원판 최고점과 구간 평균 수직속도."""
import numpy as np


def drive_to_peak(frames, drive, catch):
    """frame_index로 인덱싱된 원본 시각(ms)과 추적 중심 y(px)를 사용한다."""
    result = dict(status="산출 보류", drive_frame=drive, catch_frame=catch, peak_frame=None,
                  rise_px=None, elapsed_s=None, mean_speed_px_s=None, speed_range_px_s=None,
                  missing_frames=None, reason="", range_reason="")
    if drive is None or catch is None:
        missing = "·".join(name for name, value in (("접촉", drive), ("캐치", catch)) if value is None)
        result["reason"] = f"{missing} 시점 미지정"
        return result
    if drive >= catch:
        result["reason"] = "접촉이 캐치보다 앞서야 합니다."
        return result
    if frames is None or not {"plate_cy_px", "source_time_ms"}.issubset(frames.columns):
        result["reason"] = "저장된 원판 좌표 또는 원본 시각 없음"
        return result
    if drive not in frames.index or catch not in frames.index:
        result["reason"] = "접촉 또는 캐치 프레임 기록 누락"
        return result

    # 캐치를 포함하지 않는다. 화면에서 y가 가장 작은 프레임이 가장 높다.
    interval = frames.reindex(range(drive, catch))
    height = interval.plate_cy_px.where(np.isfinite(interval.plate_cy_px))
    result["missing_frames"] = int(height.isna().sum())
    if not np.isfinite(height.at[drive]):
        result["reason"] = "접촉 프레임 원판 좌표 누락"
        return result
    if not height.iloc[1:].notna().any():
        result["reason"] = "접촉 후 캐치 전 원판 관측 없음"
        return result
    peak = int(height.idxmin())  # 동일한 높이는 처음 도달한 프레임을 쓴다.
    if peak == drive:
        result["reason"] = "접촉 이후 추가 상승이 관측되지 않았습니다."
        return result
    time_ms = interval.source_time_ms
    if not np.isfinite(time_ms).all() or (time_ms.diff().dropna() <= 0).any():
        result["reason"] = "원본 프레임 시각 누락 또는 순서 오류"
        return result
    rise = height.at[drive] - height
    seconds = (time_ms - time_ms.at[drive]) / 1000
    speed = rise.iloc[1:] / seconds.iloc[1:]
    result.update(status="산출", peak_frame=peak, rise_px=float(rise.at[peak]),
                  elapsed_s=float(seconds.at[peak]), mean_speed_px_s=float(speed.at[peak]))

    # 종료 프레임을 바꿀 때마다 그 프레임의 상승량과 경과 시간을 모두 바꾼다.
    neighbors = speed.reindex([peak - 1, peak, peak + 1])
    if neighbors.notna().all():
        result["speed_range_px_s"] = [float(neighbors.min()), float(neighbors.max())]
    else:
        result["range_reason"] = "최고점 앞뒤 한 프레임의 구간 속도를 모두 계산할 수 없습니다."
    return result
