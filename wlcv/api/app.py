"""MP4를 제출하고 완료된 분석의 요약, 오버레이, CSV를 조회한다."""
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

import pandas as pd
from filelock import FileLock
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi import Path as PathParameter
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, NonNegativeInt
from pydantic.json_schema import SkipJsonSchema

from wlcv.api.pull_review import add_saved_clipping, reviewed_frame, save_pull_review
from wlcv.api.pull_page import pull_page
from wlcv.api.recovery_review import chart_path, create_review, current_review
from wlcv.api.review_page import review_page
from wlcv.config import local_path
from wlcv.data.mp4_input import VideoInputError
from wlcv.runtime.apex import drive_to_peak
from wlcv.runtime.measurements import first_peak_speeds
from wlcv.runtime.pull_review import review_status
from wlcv.runtime.pipeline import analyze_file


app = FastAPI(
    title="스내치 영상 분석",
    version="0.2.0",
    description="측면 MP4 한 편에서 원판과 가까운 쪽 관절을 추적합니다. 처음에는 **MP4 한 편 분석**을 실행하세요.",
    openapi_tags=[
        {"name": "영상 분석", "description": "영상을 제출하거나 저장된 분석의 요약과 결과 파일을 엽니다."},
        {"name": "시점 검수", "description": "분석 후 시점을 수정할 때 사용합니다. 원본 영상을 다시 제출할 필요가 없습니다."},
    ],
)
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
FrameValue = NonNegativeInt | Literal[""] | None
ReadyFrame = Annotated[FrameValue, Form(examples=[""], description="원판이 바닥에서 올라가기 시작하는 프레임. 첫 프레임은 0번입니다.")]
TransitionFrame = Annotated[FrameValue, Form(examples=[""], description="원판 중심이 가까운 쪽 무릎 높이를 통과하는 프레임.")]
DriveFrame = Annotated[FrameValue, Form(examples=[""], description="바와 몸이 닿는 프레임. 접촉 관련 추가 측정을 원할 때만 입력하세요.")]
LegacyFrame = Annotated[SkipJsonSchema[FrameValue], Form()]
PHASE_NAMES = {"lift_off": "ready", "knee_pass": "transition", "drive": "drive"}


def frame_alias(name, value, previous):
    if value in (None, ""):
        return previous
    if previous not in (None, "", value):
        raise HTTPException(422, f"{name}에 서로 다른 프레임 번호가 입력됐습니다.")
    return value


def public_pull_review(review):
    if review is None:
        return None
    result = dict(review)
    for field in ("suggested", "events", "sources", "reasons"):
        result[field] = {PHASE_NAMES[k]: v for k, v in review[field].items()}
    result["status"] = review_status(review["sources"])
    return result


Artifact = Literal["overlay.mp4", "frames.csv", "catch_to_finish.png"]
MEDIA_TYPES = {"overlay.mp4": "video/mp4", "frames.csv": "text/csv", "catch_to_finish.png": "image/png"}


class DriveToPeak(BaseModel):
    status: Literal["산출", "산출 보류"]
    drive_frame: int | None
    catch_frame: int | None
    peak_frame: int | None = Field(description="drive부터 catch 직전까지 원판 중심이 가장 높았던 첫 프레임(0부터).")
    rise_px: float | None = Field(description="drive부터 관측 최고점까지의 수직 상승량(px).")
    elapsed_s: float | None = Field(description="drive부터 관측 최고점까지의 원본 경과 시간(s).")
    mean_speed_px_s: float | None = Field(description="상승량을 경과 시간으로 나눈 평균 수직속도(px/s).")
    speed_range_px_s: tuple[float, float] | None = Field(description="최고점의 전·해당·다음 프레임을 각각 끝으로 계산한 평균속도의 [최솟값, 최댓값](px/s).")
    missing_frames: int | None = Field(description="drive부터 catch 직전까지 원판 좌표가 없는 프레임 수.")
    reason: str
    range_reason: str


class PullSpeedRange(BaseModel):
    early_speed_px_s: tuple[float, float] = Field(description="ready → transition 평균 수직속도 범위(px/s)")
    late_speed_px_s: tuple[float, float] = Field(description="transition → 첫 상승 최고점 평균 수직속도 범위(px/s)")
    late_over_early: tuple[float, float] = Field(description="판정에 쓰는 뒤/앞 속도 비율의 최솟값과 최댓값")


class PullReview(BaseModel):
    suggested: dict[str, int | None] = Field(description="ready, transition, drive의 자동 후보. 판독 보류는 null입니다.")
    reasons: dict[str, str]
    events: dict[str, int] = Field(description="현재 코칭 계산에 사용한 시점.")
    sources: dict[str, Literal["automatic", "manual", "reviewed"]]
    status: Literal["needs_review", "confirmed"] = Field(description="시점의 수동 확인 여부입니다. 자동 시점으로 계산한 결과도 바로 사용할 수 있습니다.")


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
    lift_to_knee_speed_px_s: float | None = Field(description="ready → transition 구간의 원판 평균 수직속도(px/s). 위쪽이 양수입니다.")
    trunk_change_deg: float | None = Field(description="transition에서 측정한 체간 기울기에서 ready에서 측정한 기울기를 뺀 값(°). 음수면 몸통이 더 세워졌습니다.")
    trunk_change_reason: str = Field(description="체간 변화량을 표시할 수 없는 이유. 측정되면 빈 문자열입니다.")
    first_peak_frame: int | None = Field(description="transition 이후 원판이 처음 상승했다가 세 프레임 연속 내려오기 직전의 프레임 번호(0부터).")
    transition_to_first_peak_speed_px_s: float | None = Field(description="transition부터 첫 상승 최고점까지 추적 중심의 평균 수직속도(px/s). 최고점 미확인 시 null.")
    pull_speed_ratio: float | None = Field(description="후반 평균 수직속도 ÷ 전반 평균 수직속도. 1보다 크면 후반이 더 빠릅니다.")
    pull_speed_late_over_early: float | None = Field(description="입력 시점에서 뒤 평균속도 ÷ 앞 평균속도. 두 구간 모두 상승할 때 계산합니다.")
    pull_speed_verdict: Literal["느림", "비슷", "빠름", "보류"] = Field(description="뒤/앞이 0.90 미만이면 느림, 1.10 초과면 빠름, 경곗값을 포함한 그 사이는 비슷입니다.")
    pull_speed_evaluated_shifts: int | None = Field(description="판정 가능한 ±1프레임 변경 수(0~4). 예전 분석에서 검사하지 않았으면 null입니다.")
    pull_speed_range: PullSpeedRange | None = Field(default=None, exclude_if=lambda value: value is None,
        description="transition 또는 첫 상승 최고점을 한 프레임 옮겨 판정이 바뀔 때만 표시합니다. 입력 시점과 판독 가능한 이동의 최솟값·최댓값입니다.")
    pull_speed_reason: str = Field(description="구간 속도 또는 비율을 표시할 수 없는 이유. 모두 측정되면 빈 문자열입니다.")
    drive_to_peak: DriveToPeak
    summary_url: str
    pull_review: PullReview | None = None
    pull_review_url: str | None = Field(default=None, description="프레임을 보며 ready와 transition을 수정하는 화면 주소.")
    recovery_review_url: str = Field(description="오버레이를 보며 catch와 finish를 선택해 그래프를 만드는 화면 주소.")
    catch_to_finish: dict | None
    files: dict[str, str]


def output_root():
    return local_path("WL_OUTPUT_DIR", "outputs/file_analysis") / "api"


def saved_measurement_frames(folder):
    """이전 분석과 캐치 재판독에는 영상 재추론 없이 원본 시각과 좌표를 읽는다."""
    path = folder / "frames.csv"
    if path.is_file():
        columns = {"frame_index", "source_time_ms", "plate_cy_px", "plate_clipped", "plate_x1", "plate_y1", "plate_x2", "plate_y2"}
        data = pd.read_csv(path, usecols=columns.__contains__)
        add_saved_clipping(data, folder)
        if "frame_index" in data:
            return data.set_index("frame_index")
    return pd.DataFrame()


def public_result(request, folder, summary):
    rise = summary["knee_to_contact_rise"]
    coaching = summary.get("coaching", {})
    pulls = coaching.get("pulls", {})
    pull_review = summary.get("pull_review")
    events = {**summary["manual_events"], **(pull_review["events"] if pull_review else {})}
    trunk_change = summary.get("trunk_change_deg")
    trunk_reason = ""
    if trunk_change is None:
        if events.get("lift_off") is None or events.get("knee_pass") is None:
            trunk_reason = "ready 또는 transition 시점 미지정"
        else:
            trunk_reason = "ready 또는 transition에서 가까운 쪽 어깨·골반의 기울기 측정 불가"
    counts = summary["plate_counts"].copy()
    counts["좌표 누락"] = counts.pop("미탐")  # 새 영상에는 수동 정답이 없다.
    review = current_review(folder)
    catch = review["catch_frame"] if review else events.get("catch")
    lift, knee, drive = events.get("lift_off"), events.get("knee_pass"), events.get("drive")
    peak = coaching.get("drive_to_peak")
    update_peak = peak is None or (peak["drive_frame"], peak["catch_frame"]) != (drive, catch)
    # 예전 요약의 후반 속도는 drive에서 끝났다. 저장한 궤적으로 현재 구간을 다시 계산한다.
    update_pulls = "first_peak_frame" not in pulls
    frames = None
    if update_pulls or (update_peak and drive is not None and catch is not None):
        frames = saved_measurement_frames(folder)
    if update_pulls:
        old_early = pulls.get("early")
        pulls = first_peak_speeds(frames, lift, knee)
        if pulls["early"] is None and old_early is not None and not {"plate_cy_px", "source_time_ms"}.issubset(frames):
            pulls["early"] = old_early
    if update_peak:
        peak = drive_to_peak(frames, drive, catch)
    files = {}
    for filename in MEDIA_TYPES:
        path = chart_path(folder, review) if filename == "catch_to_finish.png" else folder / filename
        if path.is_file():
            files[filename] = request.url_for("download_result", analysis_id=folder.name, filename=filename).path

    return AnalysisResult(
        analysis_id=folder.name, frame_count=summary["frame_count"], near_side=summary["near_side"],
        facing=summary["facing"], plate_mode=summary["plate_mode"], plate_counts=counts,
        selected_model_counts=summary["selected_model_counts"], manual_events=summary["manual_events"],
        inference_backends=summary.get("inference_backends", {}),
        lift_to_knee_rise_px=summary["manual_phase_interval"]["rise_px"],
        knee_to_contact_rise_px=rise["rise_px"], knee_to_contact_source=rise["source"],
        knee_to_contact_reason=rise["reason"], lift_to_knee_speed_px_s=pulls.get("early"),
        trunk_change_deg=trunk_change, trunk_change_reason=trunk_reason,
        first_peak_frame=pulls["first_peak_frame"], transition_to_first_peak_speed_px_s=pulls["late"],
        pull_speed_ratio=pulls["late_over_early"], pull_speed_late_over_early=pulls["late_over_early"],
        pull_speed_verdict=pulls["verdict"],
        pull_speed_evaluated_shifts=pulls.get("evaluated_shifts"), pull_speed_range=pulls.get("sensitivity_range"),
        pull_speed_reason=pulls.get("reason", "저장된 분석에 구간 속도 기록 없음"),
        drive_to_peak=peak,
        summary_url=request.url_for("get_result", analysis_id=folder.name).path, files=files,
        pull_review=public_pull_review(pull_review),
        pull_review_url=request.url_for("review_pulls", analysis_id=folder.name).path if pull_review else None,
        recovery_review_url=request.url_for("review_recovery", analysis_id=folder.name).path,
        catch_to_finish=review or coaching.get("catch_to_finish"),
    )


@app.post(
    "/analyses", status_code=201, tags=["영상 분석"], summary="MP4 한 편 분석",
    description=(
        "MP4와 촬영 방향을 입력하세요. ready와 transition을 비우면 자동으로 찾아 "
        "ready → transition과 transition → 첫 상승 최고점의 평균 수직속도 및 뒤/앞 비율을 계산합니다. "
        "최고점을 확인할 수 없으면 후반 속도와 비율은 null입니다. "
        "결과의 `files`에서 오버레이와 CSV를 열 수 있습니다."
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
    ready: ReadyFrame = None,
    transition: TransitionFrame = None,
    drive: DriveFrame = None,
    lift_off: LegacyFrame = None,
    knee_pass: LegacyFrame = None,
    catch: Frame = None,
    finish: Frame = None,
) -> AnalysisResult:
    filename = Path((video.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(415, "MP4 파일을 선택하세요.")
    events = {"lift_off": frame_alias("ready", ready, lift_off),
              "knee_pass": frame_alias("transition", transition, knee_pass),
              "drive": drive, "catch": catch, "finish": finish}
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
                    output_parent=output_root(), enable_pull_review=True,
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


@app.get("/analyses/{analysis_id}", tags=["영상 분석"], summary="저장된 분석 요약 조회")
def get_result(request: Request, analysis_id: AnalysisId) -> AnalysisResult:
    folder = output_root() / analysis_id
    return public_result(request, folder, saved_summary(folder))


def reviewable_summary(folder):
    summary = saved_summary(folder)
    if "pull_review" not in summary:
        raise HTTPException(409, "이전 분석에는 시점 검수용 좌표가 없습니다. MP4를 다시 제출하세요.")
    return summary


@app.get("/analyses/{analysis_id}/pull-review", response_class=HTMLResponse, include_in_schema=False)
def review_pulls(request: Request, analysis_id: AnalysisId):
    folder = output_root() / analysis_id
    summary = reviewable_summary(folder)
    return pull_page(public_result(request, folder, summary))


@app.get("/analyses/{analysis_id}/pull-review/frames/{frame}", response_class=Response, include_in_schema=False)
def get_pull_frame(analysis_id: AnalysisId, frame: NonNegativeInt):
    folder = output_root() / analysis_id
    summary = reviewable_summary(folder)
    if frame >= summary["frame_count"]:
        raise HTTPException(422, "영상의 프레임 범위를 벗어났습니다.")
    return Response(reviewed_frame(folder, frame), media_type="image/jpeg")


@app.post("/analyses/{analysis_id}/pull-review", tags=["시점 검수"], summary="ready·transition 수정 및 평균속도 재계산")
def confirm_pulls(request: Request, analysis_id: AnalysisId,
                  ready: ReadyFrame = None, transition: TransitionFrame = None,
                  drive: LegacyFrame = None,
                  lift_off: LegacyFrame = None, knee_pass: LegacyFrame = None) -> AnalysisResult:
    folder = output_root() / analysis_id
    lift_off = frame_alias("ready", ready, lift_off)
    knee_pass = frame_alias("transition", transition, knee_pass)
    edits = {k: v for k, v in (("lift_off", lift_off), ("knee_pass", knee_pass), ("drive", drive))
             if v not in (None, "")}
    if not edits:
        raise HTTPException(422, "확인한 시점의 프레임 번호를 입력하세요.")
    reviewable_summary(folder)
    try:
        with analysis_lock, FileLock(str(folder / ".review.lock")):
            summary = reviewable_summary(folder)
            save_pull_review(folder, summary, edits)
            return public_result(request, folder, summary)
    except VideoInputError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/analyses/{analysis_id}/catch-to-finish", response_class=HTMLResponse, include_in_schema=False)
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


@app.post("/analyses/{analysis_id}/catch-to-finish", tags=["시점 검수"], summary="catch·finish 지정 및 어깨 기준 원판 위치 그래프 생성")
def save_review(request: Request, analysis_id: AnalysisId, catch: Annotated[NonNegativeInt, Form()],
                finish: Annotated[NonNegativeInt, Form()]) -> AnalysisResult:
    folder = output_root() / analysis_id
    saved_summary(folder)
    try:
        with analysis_lock, FileLock(str(folder / ".review.lock")):
            summary = saved_summary(folder)
            create_review(folder, summary, catch, finish)
            return public_result(request, folder, summary)
    except VideoInputError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/analyses/{analysis_id}/files/{filename}", tags=["영상 분석"], response_class=FileResponse, summary="결과 파일 열기")
def download_result(analysis_id: AnalysisId, filename: Artifact):
    folder = output_root() / analysis_id
    path = chart_path(folder, current_review(folder)) if filename == "catch_to_finish.png" else folder / filename
    if not path.is_file():
        raise HTTPException(404, "이 분석에서 생성된 파일을 찾을 수 없습니다.")
    disposition = "attachment" if filename == "frames.csv" else "inline"
    return FileResponse(path, media_type=MEDIA_TYPES[filename], filename=filename, content_disposition_type=disposition)
