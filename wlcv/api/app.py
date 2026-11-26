"""MP4를 제출하고 완료된 분석의 요약, 오버레이, CSV를 조회한다."""
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi import Path as PathParameter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema

from wlcv.config import local_path
from wlcv.data.mp4_input import VideoInputError
from wlcv.runtime.pipeline import analyze_file


app = FastAPI(title="스내치 영상 분석", description="측면 MP4 한 편에서 원판과 가까운 쪽 관절을 추적합니다.")
# 한 프로세스에서 동시에 여러 영상의 모델을 메모리에 올리지 않는다.
analysis_lock = Lock()
Side = Literal["left", "right"]
Mode = Literal["single", "cutmix"]
AnalysisId = Annotated[str, PathParameter(pattern=r"^[0-9]{8}-[0-9]{6}-[0-9]{6}$")]
Frame = Annotated[int | None, Form(ge=0, description="원본 프레임 번호. 첫 장은 0입니다.")]
Artifact = Literal["overlay.mp4", "frames.csv", "catch_to_finish.png"]
MEDIA_TYPES = {"overlay.mp4": "video/mp4", "frames.csv": "text/csv", "catch_to_finish.png": "image/png"}


class AnalysisResult(BaseModel):
    analysis_id: str
    frame_count: int
    near_side: Side
    facing: Side
    plate_mode: Mode
    plate_counts: dict[str, int]
    selected_model_counts: dict[str, int]
    manual_events: dict[str, int]
    lift_to_knee_rise_px: float | None
    knee_to_contact_rise_px: float | None
    knee_to_contact_source: str | None
    knee_to_contact_reason: str
    trunk_change_deg: float | None
    summary_url: str
    files: dict[str, str]


def output_root():
    return local_path("WL_OUTPUT_DIR", "outputs/file_analysis") / "api"


def public_result(request, folder, summary):
    rise = summary["knee_to_contact_rise"]
    counts = summary["plate_counts"].copy()
    counts["좌표 누락"] = counts.pop("미탐")  # 새 영상에는 수동 정답이 없다.
    files = {}
    for filename in MEDIA_TYPES:
        if (folder / filename).is_file():
            files[filename] = request.url_for("download_result", analysis_id=folder.name, filename=filename).path

    return AnalysisResult(
        analysis_id=folder.name, frame_count=summary["frame_count"], near_side=summary["near_side"],
        facing=summary["facing"], plate_mode=summary["plate_mode"], plate_counts=counts,
        selected_model_counts=summary["selected_model_counts"], manual_events=summary["manual_events"],
        lift_to_knee_rise_px=summary["manual_phase_interval"]["rise_px"],
        knee_to_contact_rise_px=rise["rise_px"], knee_to_contact_source=rise["source"],
        knee_to_contact_reason=rise["reason"], trunk_change_deg=summary["trunk_change_deg"],
        summary_url=request.url_for("get_result", analysis_id=folder.name).path, files=files,
    )


@app.post(
    "/analyses", status_code=201, summary="MP4 한 편 분석",
    description="베이스라인이 직접 검출과 시간 연결로도 원판을 찾지 못한 프레임만 CutMix로 자동 보완합니다.",
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
    events = {key: frame for key, frame in events.items() if frame is not None}

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


@app.get("/analyses/{analysis_id}", summary="저장된 분석 요약")
def get_result(request: Request, analysis_id: AnalysisId) -> AnalysisResult:
    folder = output_root() / analysis_id
    summary = folder / "summary.json"
    if not summary.is_file():
        raise HTTPException(404, "완료된 분석을 찾을 수 없습니다.")
    return public_result(request, folder, json.loads(summary.read_text(encoding="utf-8")))


@app.get("/analyses/{analysis_id}/files/{filename}", response_class=FileResponse, summary="분석 결과 파일 열기")
def download_result(analysis_id: AnalysisId, filename: Artifact):
    path = output_root() / analysis_id / filename
    if not path.is_file():
        raise HTTPException(404, "이 분석에서 생성된 파일을 찾을 수 없습니다.")
    disposition = "attachment" if filename == "frames.csv" else "inline"
    return FileResponse(path, media_type=MEDIA_TYPES[filename], filename=filename, content_disposition_type=disposition)
