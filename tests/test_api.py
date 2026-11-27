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
            assert result["lift_to_knee_speed_px_s"] is None
            assert result["pull_speed_reason"] == "저장된 분석에 구간 속도 기록 없음"
            assert "시점 미지정" in result["trunk_change_reason"]
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


def test_review_saved_frames_without_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("WL_OUTPUT_DIR", str(tmp_path))
    folder = tmp_path / "api/20261126-123456-000001"
    folder.mkdir(parents=True)
    summary = {**sample_summary("cutmix"), "overlay_fps": 30}
    (folder / "summary.json").write_text(json.dumps(summary))
    csv = "frame_index,forward_px,trunk_angle_deg\n0,3,\n1,,\n2,,\n"
    (folder / "frames.csv").write_text(csv)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            info = (await client.get(f"/analyses/{folder.name}")).json()
            url = info["recovery_review_url"]
            assert "캐치·피니시 판독" in (await client.get(url)).text
            result = (await client.post(url, data={"catch": 0, "finish": 2})).raise_for_status().json()
            assert result["manual_events"] == {}
            assert result["catch_to_finish"]["distance_frames"] == 1
            assert result["catch_to_finish"]["angle_frames"] == 0
            chart = result["files"]["catch_to_finish.png"]
            png = (await client.get(chart)).content
            assert png.startswith(b"\x89PNG")
            for start, end in [(1, 2), (2, 1), (0, 3)]:
                assert (await client.post(url, data={"catch": start, "finish": end})).status_code == 422
            assert (await client.get(chart)).content == png
            assert (await client.get(info["summary_url"])).json() == result
    asyncio.run(check())
    assert (folder / "frames.csv").read_text() == csv
    assert json.loads((folder / "summary.json").read_text()) == summary


def test_upload_uses_automatic_fallback_without_a_mode_selector(tmp_path, monkeypatch):
    def fake_analysis(path, *, plate_mode, manual_events, **options):
        summary = sample_summary(plate_mode)
        summary["manual_events"] = manual_events
        if "drive" in manual_events:
            summary["coaching"] = {"pulls": {"early": 0., "late": 400., "ratio": 0., "reason": ""}}
            summary["trunk_change_deg"] = -7.5
        folder = tmp_path / "api" / "20261126-123456-000001"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary))
        return folder, summary

    monkeypatch.setenv("WL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("wlcv.api.app.analyze_file", fake_analysis)
    schema = app.openapi()
    upload = schema["paths"]["/analyses"]["post"]["requestBody"]["content"]["multipart/form-data"]
    form = upload["schema"]["$ref"].split("/")[-1]
    properties = schema["components"]["schemas"][form]["properties"]
    assert "plate_mode" not in properties
    frames = ("lift_off", "knee_pass", "drive", "catch", "finish")
    for key in frames:
        assert properties[key]["examples"] == [""]

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            cases = [
                ({}, "cutmix", {}),
                (dict.fromkeys(frames, ""), "cutmix", {}),
                ({"plate_mode": "single", "lift_off": "0"}, "single", {"lift_off": 0}),
                ({"lift_off": "0", "knee_pass": "1", "drive": "2"}, "cutmix", {"lift_off": 0, "knee_pass": 1, "drive": 2}),
            ]
            for extra, expected, events in cases:
                response = await client.post(
                    "/analyses", files={"video": ("snatch.mp4", b"test")},
                    data={"near_side": "left", "facing": "right", **extra},
                )
                assert response.status_code == 201
                assert response.json()["plate_mode"] == expected
                assert response.json()["manual_events"] == events
                if "drive" in events:
                    metrics = response.json()
                    assert metrics["lift_to_knee_speed_px_s"] == metrics["pull_speed_ratio"] == 0.
                    assert metrics["knee_to_contact_speed_px_s"] == 400.
                    assert metrics["trunk_change_deg"] == -7.5
                    assert metrics["trunk_change_reason"] == metrics["pull_speed_reason"] == ""
                    assert (await client.get(metrics["summary_url"])).json() == metrics

    asyncio.run(check())
