"""MP4를 제출하고 완료된 분석의 요약, 오버레이, CSV를 조회한다."""
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi import Path as PathParameter
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, NonNegativeInt
from pydantic.json_schema import SkipJsonSchema

from wlcv.api.recovery_review import chart_path, create_review, current_review
from wlcv.api.review_page import review_page
from wlcv.config import local_path
from wlcv.data.mp4_input import VideoInputError
from wlcv.runtime.apex import drive_to_peak
from wlcv.runtime.measurements import pull_summary, speed_verdict
from wlcv.runtime.pipeline import analyze_file


app = FastAPI(title="스내치 영상 분석", description="측면 MP4 한 편에서 원판과 가까운 쪽 관절을 추적합니다.")
# 한 프로세스에서 동시에 여러 영상의 모델을 메모리에 올리지 않는다.
analysis_lock = Lock()
Side = Literal["left", "right"]
Mode = Literal["single", "cutmix"]
AnalysisId = Annotated[str, PathParameter(pattern=r"^[0-9]{8}-[0-9]{6}-[0-9]{6}$")]
# multipart 폼에서 빈 문자열은 시점 생략, 숫자 0은 첫 프레임이다.
Frame = Annotated[
    NonNegativeInt | Literal[""] | None,
    Form(examples=[""], description="선택 사항입니다. 시점을 모르면 빈칸으로 두세요. 0은 영상의 첫 프레임입니다."),
]
Artifact = Literal["overlay.mp4", "frames.csv", "catch_to_finish.png"]
MEDIA_TYPES = {"overlay.mp4": "video/mp4", "frames.csv": "text/csv", "catch_to_finish.png": "image/png"}


class DriveToPeak(BaseModel):
    status: Literal["산출", "산출 보류"]
    drive_frame: int | None
    catch_frame: int | None
    peak_frame: int | None = Field(description="접촉부터 캐치 직전까지 원판 중심이 가장 높았던 첫 프레임(0부터).")
    rise_px: float | None = Field(description="접촉부터 관측 최고점까지의 수직 상승량(px).")
    elapsed_s: float | None = Field(description="접촉부터 관측 최고점까지의 원본 경과 시간(s).")
    mean_speed_px_s: float | None = Field(description="상승량을 경과 시간으로 나눈 평균 수직속도(px/s).")
    speed_range_px_s: tuple[float, float] | None = Field(description="최고점의 전·해당·다음 프레임을 각각 끝으로 계산한 평균속도의 [최솟값, 최댓값](px/s).")
    missing_frames: int | None = Field(description="접촉부터 캐치 직전까지 원판 좌표가 없는 프레임 수.")
    reason: str
    range_reason: str


class PullSpeedRange(BaseModel):
    early_speed_px_s: tuple[float, float] = Field(description="이륙→무릎 통과 평균 수직속도 범위(px/s)")
    late_speed_px_s: tuple[float, float] = Field(description="무릎 통과→접촉 평균 수직속도 범위(px/s)")
    early_over_late: tuple[float, float] = Field(description="앞/뒤 속도 비율의 최솟값과 최댓값")
    late_over_early: tuple[float, float] = Field(description="판정에 쓰는 뒤/앞 속도 비율의 최솟값과 최댓값")


class AnalysisResult(BaseModel):
    analysis_id: str
    frame_count: int
    near_side: Side
    facing: Side
    plate_mode: Mode
    plate_counts: dict[str, int]
    selected_model_counts: dict[str, int]
    inference_backends: dict[str, str]
    manual_events: dict[str, int]
    lift_to_knee_rise_px: float | None
    knee_to_contact_rise_px: float | None
    knee_to_contact_source: str | None
    knee_to_contact_reason: str
    lift_to_knee_speed_px_s: float | None = Field(description="이륙 → 무릎 통과의 원판 평균 수직속도(px/s). 위쪽이 양수입니다.")
    trunk_change_deg: float | None = Field(description="무릎 통과 때의 체간 기울기에서 이륙 때의 기울기를 뺀 값(°). 음수면 몸통이 더 세워졌습니다.")
    trunk_change_reason: str = Field(description="체간 변화량을 표시할 수 없는 이유. 측정되면 빈 문자열입니다.")
    knee_to_contact_speed_px_s: float | None = Field(description="채택한 knee_to_contact_rise_px를 무릎 통과 → 접촉의 원본 경과 시간으로 나눈 평균 수직속도(px/s).")
    pull_speed_ratio: float | None = Field(description="앞 구간 평균 수직속도 ÷ 뒤 구간 평균 수직속도. 두 구간 모두 상승할 때 1보다 작으면 뒤 구간이 더 빠릅니다.")
    pull_speed_late_over_early: float | None = Field(description="입력 시점에서 뒤 평균속도 ÷ 앞 평균속도. 두 구간 모두 상승할 때 계산합니다.")
    pull_speed_verdict: Literal["느림", "비슷", "빠름", "보류"] = Field(description="뒤/앞이 0.90 미만이면 느림, 1.10 초과면 빠름, 경곗값을 포함한 그 사이는 비슷입니다.")
    pull_speed_evaluated_shifts: int | None = Field(description="판정 가능한 ±1프레임 변경 수(0~4). 예전 분석에서 검사하지 않았으면 null입니다.")
    pull_speed_range: PullSpeedRange | None = Field(default=None, exclude_if=lambda value: value is None,
        description="무릎 통과·접촉 중 하나를 한 프레임 옮겨 판정이 바뀔 때만 표시합니다. 입력 시점과 판독 가능한 이동의 최솟값·최댓값입니다.")
    pull_speed_reason: str = Field(description="구간 속도 또는 비율을 표시할 수 없는 이유. 모두 측정되면 빈 문자열입니다.")
    drive_to_peak: DriveToPeak
    summary_url: str
    recovery_review_url: str
    catch_to_finish: dict | None
    files: dict[str, str]


def output_root():
    return local_path("WL_OUTPUT_DIR", "outputs/file_analysis") / "api"


def saved_measurement_frames(folder):
    """이전 분석과 캐치 재판독에는 영상 재추론 없이 원본 시각과 좌표를 읽는다."""
    path = folder / "frames.csv"
    if path.is_file():
        columns = {"frame_index", "source_time_ms", "plate_cy_px"}
        data = pd.read_csv(path, usecols=columns.__contains__)
        if "frame_index" in data:
            return data.set_index("frame_index")
    return pd.DataFrame()


def public_result(request, folder, summary):
    rise = summary["knee_to_contact_rise"]
    coaching = summary.get("coaching", {})
    pulls = coaching.get("pulls", {})
    events = summary["manual_events"]
    trunk_change = summary.get("trunk_change_deg")
    trunk_reason = ""
    if trunk_change is None:
        if events.get("lift_off") is None or events.get("knee_pass") is None:
            trunk_reason = "이륙 또는 무릎 통과 시점 미지정"
        else:
            trunk_reason = "이륙 또는 무릎 통과에서 가까운 쪽 어깨·골반의 기울기 측정 불가"
    counts = summary["plate_counts"].copy()
    counts["좌표 누락"] = counts.pop("미탐")  # 새 영상에는 수동 정답이 없다.
    review = current_review(folder)
    catch = review["catch_frame"] if review else events.get("catch")
    lift, knee, drive = events.get("lift_off"), events.get("knee_pass"), events.get("drive")
    peak = coaching.get("drive_to_peak")
    update_peak = peak is None or (peak["drive_frame"], peak["catch_frame"]) != (drive, catch)
    update_pulls = "late_source" not in pulls and knee is not None and (drive is not None or lift is not None)
    frames = None
    if update_pulls or (update_peak and drive is not None and catch is not None):
        frames = saved_measurement_frames(folder)
    if update_pulls:
        previous_early = pulls.get("early")
        pulls = pull_summary(frames, lift, knee, drive, rise)
        if frames.empty and previous_early is not None:
            pulls["early"] = previous_early
    if update_peak:
        peak = drive_to_peak(frames, drive, catch)
    files = {}
    for filename in MEDIA_TYPES:
        path = chart_path(folder, review) if filename == "catch_to_finish.png" else folder / filename
        if path.is_file():
            files[filename] = request.url_for("download_result", analysis_id=folder.name, filename=filename).path

    factor, verdict = speed_verdict(pulls.get("early"), pulls.get("late"), pulls.get("ratio"))
    return AnalysisResult(
        analysis_id=folder.name, frame_count=summary["frame_count"], near_side=summary["near_side"],
        facing=summary["facing"], plate_mode=summary["plate_mode"], plate_counts=counts,
        selected_model_counts=summary["selected_model_counts"], manual_events=events,
        inference_backends=summary.get("inference_backends", {}),
        lift_to_knee_rise_px=summary["manual_phase_interval"]["rise_px"],
        knee_to_contact_rise_px=rise["rise_px"], knee_to_contact_source=rise["source"],
        knee_to_contact_reason=rise["reason"], lift_to_knee_speed_px_s=pulls.get("early"),
        trunk_change_deg=trunk_change, trunk_change_reason=trunk_reason,
        knee_to_contact_speed_px_s=pulls.get("late"), pull_speed_ratio=pulls.get("ratio"),
        pull_speed_late_over_early=factor, pull_speed_verdict=verdict,
        pull_speed_evaluated_shifts=pulls.get("evaluated_shifts"), pull_speed_range=pulls.get("sensitivity_range"),
        pull_speed_reason=pulls.get("reason", "저장된 분석에 구간 속도 기록 없음"),
        drive_to_peak=peak,
        summary_url=request.url_for("get_result", analysis_id=folder.name).path, files=files,
        recovery_review_url=request.url_for("review_recovery", analysis_id=folder.name).path,
        catch_to_finish=review or coaching.get("catch_to_finish"),
    )


@app.post(
    "/analyses", status_code=201, summary="MP4 한 편 분석",
    description=(
        "원판 추적이 끊기면 CutMix로 자동 보완합니다. "
        "두 구간의 평균 수직속도(px/s)를 보려면 이륙(lift_off), 무릎 통과(knee_pass), "
        "바와 몸의 접촉(drive) 프레임을 입력하세요. 이륙·무릎 통과만 지정해도 체간 기울기 변화(°)를 확인할 수 있습니다. "
        "무릎 통과 또는 접촉을 ±1프레임 옮겨 뒤/앞 속도 판정이 바뀌면 pull_speed_range도 표시합니다. "
        "접촉과 캐치(catch)를 입력하면 drive_to_peak에서 상승량·최고점 프레임·평균속도와 ±1프레임 속도 범위를 볼 수 있습니다. "
        "Execute 후 Response body에서 측정값을 확인하세요. 첫 프레임은 0이며 모르는 시점은 빈칸으로 두세요."
    ),
)
def create_analysis(
    request: Request,
    response: Response,
    video: UploadFile,
    near_side: Annotated[Side, Form(description="카메라에 가까운 쪽. 선수의 왼쪽 또는 오른쪽")],
    facing: Annotated[Side, Form(description="선수가 향한 쪽. 화면의 왼쪽 또는 오른쪽")],
    # 예전 비교 스크립트의 명시적 모드는 지원하고, 업로드 화면은 자동 보완을 쓴다.
    plate_mode: Annotated[SkipJsonSchema[Mode], Form()] = "cutmix",
    lift_off: Frame = None,
    knee_pass: Frame = None,
    drive: Frame = None,
    catch: Frame = None,
    finish: Frame = None,
) -> AnalysisResult:
    filename = Path((video.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(415, "MP4 파일을 선택하세요.")
    events = {"lift_off": lift_off, "knee_pass": knee_pass, "drive": drive, "catch": catch, "finish": finish}
    events = {key: frame for key, frame in events.items() if frame not in (None, "")}

    with TemporaryDirectory() as temp:
        path = Path(temp) / filename
        with path.open("wb") as saved:
            shutil.copyfileobj(video.file, saved)
        try:
            with analysis_lock:
                folder, summary = analyze_file(
                    path, near_side=near_side, facing=facing, manual_events=events,
                    plate_mode=plate_mode, device=os.environ.get("WL_API_DEVICE") or "cpu",
                    output_parent=output_root(),
                )
        except VideoInputError as error:
            raise HTTPException(422, str(error).replace(str(path), filename)) from error

    result = public_result(request, folder, summary)
    response.headers["Location"] = result.summary_url
    return result


def saved_summary(folder):
    summary = folder / "summary.json"
    if not summary.is_file():
        raise HTTPException(404, "완료된 분석을 찾을 수 없습니다.")
    return json.loads(summary.read_text(encoding="utf-8"))


@app.get("/analyses/{analysis_id}", summary="저장된 분석 요약")
def get_result(request: Request, analysis_id: AnalysisId) -> AnalysisResult:
    folder = output_root() / analysis_id
    return public_result(request, folder, saved_summary(folder))


@app.get("/analyses/{analysis_id}/catch-to-finish", response_class=HTMLResponse, summary="캐치·피니시 판독 화면")
def review_recovery(request: Request, analysis_id: AnalysisId):
    folder = output_root() / analysis_id
    summary = saved_summary(folder)
    review = current_review(folder)
    events = summary["manual_events"]
    if review:
        events = {"catch": review["catch_frame"], "finish": review["finish_frame"]}
    overlay = request.url_for("download_result", analysis_id=analysis_id, filename="overlay.mp4").path
    chart = request.url_for("download_result", analysis_id=analysis_id, filename="catch_to_finish.png").path
    return review_page(overlay, chart, summary["frame_count"], summary["overlay_fps"], events,
                       has_chart=chart_path(folder, review).is_file())


@app.post("/analyses/{analysis_id}/catch-to-finish", summary="저장된 CSV로 캐치·피니시 그래프 만들기")
def save_review(request: Request, analysis_id: AnalysisId, catch: Annotated[NonNegativeInt, Form()],
                finish: Annotated[NonNegativeInt, Form()]) -> AnalysisResult:
    folder = output_root() / analysis_id
    summary = saved_summary(folder)
    try:
        with analysis_lock:
            create_review(folder, summary, catch, finish)
            return public_result(request, folder, summary)
    except VideoInputError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/analyses/{analysis_id}/files/{filename}", response_class=FileResponse, summary="분석 결과 파일 열기")
def download_result(analysis_id: AnalysisId, filename: Artifact):
    folder = output_root() / analysis_id
    path = chart_path(folder, current_review(folder)) if filename == "catch_to_finish.png" else folder / filename
    if not path.is_file():
        raise HTTPException(404, "이 분석에서 생성된 파일을 찾을 수 없습니다.")
    disposition = "attachment" if filename == "frames.csv" else "inline"
    return FileResponse(path, media_type=MEDIA_TYPES[filename], filename=filename, content_disposition_type=disposition)
