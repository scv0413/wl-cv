"""HTTP로 게시하는 파일과 측정값의 범위를 확인한다."""
import asyncio
import json

import httpx

from wlcv.api.app import app


def sample_summary(mode):
    return {
        "frame_count": 3, "near_side": "left", "facing": "right", "plate_mode": mode,
        "plate_counts": {"직접 검출": 2, "시간 연결": 0, "미탐": 1},
        "manual_events": {}, "selected_model_counts": {"baseline": 2, "cutmix": 0, "none": 1},
        "manual_phase_interval": {"rise_px": None}, "trunk_change_deg": None,
        "plate_weights": "/private/models/plate.pt",
        "knee_to_contact_rise": {"rise_px": None, "source": None, "reason": "시점 미지정"},
    }


def test_saved_result_exposes_measurements_and_only_result_files(tmp_path, monkeypatch):
    monkeypatch.setenv("WL_OUTPUT_DIR", str(tmp_path))
    analysis_id = "20261126-123456-000001"
    folder = tmp_path / "api" / analysis_id
    folder.mkdir(parents=True)
    summary = sample_summary("single")
    (folder / "summary.json").write_text(json.dumps(summary))
    (folder / "frames.csv").write_text("frame_index\n0\n1\n2\n")

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            url = f"/analyses/{analysis_id}"
            response = await client.get(url)
            assert response.status_code == 200
            assert "/private/models" not in response.text
            result = response.json()
            assert result["plate_counts"]["좌표 누락"] == 1
            assert result["lift_to_knee_rise_px"] is None
            csv = await client.get(result["files"]["frames.csv"])
            assert csv.text == (folder / "frames.csv").read_text()
            assert (await client.get(url + "/files/summary.json")).status_code == 422
            assert (await client.get(url + "/files/catch_to_finish.png")).status_code == 404

    asyncio.run(check())


def test_saving_catch_graph_works_without_a_gui(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import pandas as pd

    from wlcv.runtime.recovery import save_recovery_curve

    def no_gui(*args, **kwargs):
        raise AssertionError("파일 저장 중 화면 창을 열었습니다.")

    monkeypatch.setattr(plt, "figure", no_gui)
    rows = pd.DataFrame({"frame_index": [0, 1], "forward_px": [1., 2.], "trunk_angle_deg": [8., 5.]})
    save_recovery_curve(rows, {"catch": 0, "finish": 1}, tmp_path)
    assert (tmp_path / "catch_to_finish.png").read_bytes().startswith(b"\x89PNG")


def test_upload_uses_automatic_fallback_without_a_mode_selector(tmp_path, monkeypatch):
    def fake_analysis(path, *, plate_mode, **options):
        return tmp_path / "20261126-123456-000001", sample_summary(plate_mode)

    monkeypatch.setattr("wlcv.api.app.analyze_file", fake_analysis)
    schema = app.openapi()
    upload = schema["paths"]["/analyses"]["post"]["requestBody"]["content"]["multipart/form-data"]
    form = upload["schema"]["$ref"].split("/")[-1]
    assert "plate_mode" not in schema["components"]["schemas"][form]["properties"]

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for extra, expected in [({}, "cutmix"), ({"plate_mode": "single"}, "single")]:
                response = await client.post(
                    "/analyses", files={"video": ("snatch.mp4", b"test")},
                    data={"near_side": "left", "facing": "right", **extra},
                )
                assert response.status_code == 201
                assert response.json()["plate_mode"] == expected

    asyncio.run(check())
