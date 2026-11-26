"""Mac에서 실행해 같은 영상의 네이티브 API 결과와 로컬 Docker API 결과를 비교한다."""
import argparse
import json
import subprocess
from io import BytesIO
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import pandas as pd
from PIL import Image

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, DATASET, ROOT, VIDEOS
from wlcv.evaluation.schedule_review import review_inputs
from wlcv.runtime.tracking import BOX, iou


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_folder(parent, video, events):
    for summary in sorted(parent.glob("*/summary.json"), reverse=True):
        data = load_json(summary)
        if video == "b00_sn_f_01" and not (summary.parent / "recovery_review.json").is_file():
            continue
        if (data["video_id"] == video and data["manual_events"] == events
                and data["plate_mode"] == "cutmix" and data["device"] == "cpu"
                and data["near_side"] == "left" and data["facing"] == "left"):
            return summary.parent
    raise FileNotFoundError(f"{video}의 Mac CPU 참조 결과가 없습니다: {parent}")


def hits(frames, labels):
    matched = set()
    for _, label in labels.iterrows():
        index = int(label.frame_index)
        box = frames.loc[index, ["plate_" + key for key in BOX]].to_numpy(float)
        if iou(box, label[BOX].to_numpy(float)) >= 0.50:
            matched.add(index)
    return matched


def probe_overlay(path):
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                           "-show_entries", "stream=codec_name,nb_read_frames", "-of", "json", str(path)],
                          capture_output=True, text=True, check=True)
    stream, = json.loads(proc.stdout)["streams"]
    return stream["codec_name"], int(stream["nb_read_frames"])


def verify(client, video, events, references, results, truth):
    ref = reference_folder(references, video, events)
    before = pd.read_csv(ref / "frames.csv").set_index("frame_index")
    with (VIDEOS / f"{video}.mp4").open("rb") as source:
        start = perf_counter()
        response = client.post("/analyses", files={"video": (Path(source.name).name, source, "video/mp4")},
                               data={"near_side": "left", "facing": "left", **events})
        response.raise_for_status()
        info = response.json()
    elapsed = perf_counter() - start
    out = results / "api" / info["analysis_id"]
    after = pd.read_csv(BytesIO(client.get(info["files"]["frames.csv"]).raise_for_status().content)).set_index("frame_index")
    checks = {}
    details = {"analysis_id": out.name, "reference_id": ref.name, "upload_to_response_s": round(elapsed, 2)}
    checks["frames"] = after.index.equals(before.index) and info["frame_count"] == len(before)
    checks["source_time_ms"] = bool(np.allclose(after["source_time_ms"], before["source_time_ms"], atol=0.001, rtol=0)) if checks["frames"] else False
    checks["model_and_status"] = after[["selected_model", "plate_status"]].equals(before[["selected_model", "plate_status"]])
    checks["overlay"] = probe_overlay(out / "overlay.mp4") == ("h264", len(before))
    details["model_counts"] = info["selected_model_counts"]
    if video.startswith(("a03", "c04")):
        times = load_events(ANNOTATIONS, video, pass_number=1)
        labels = truth[truth.video_id.eq(video) & truth.frame_index.between(times["ready"], times["finish"])].dropna(subset=BOX)
        native_hits, docker_hits = hits(before, labels), hits(after, labels)
        details["motion_hits"] = [len(docker_hits), len(labels)]
        checks["same_hits"] = native_hits == docker_hits
        expected = [35, 40] if video.startswith("a03") else [69, 69]
        checks["expected_motion_hits"] = details["motion_hits"] == expected
        if video.startswith("a03"):
            details["cutmix_hits"] = sorted(docker_hits.intersection(after.index[after.selected_model.eq("cutmix")]))
            checks["recovered_eight"] = details["cutmix_hits"] == [4, 5, 6, 10, 11, 12, 13, 14]
        else:
            rise = info["lift_to_knee_rise_px"]
            native_rise = load_json(ref / "summary.json")["manual_phase_interval"]["rise_px"]
            details["rise_px"] = [native_rise, rise]
            checks["rise_within_0.1px"] = abs(rise - native_rise) <= 0.1
            contact = load_json(ref / "summary.json")["knee_to_contact_rise"]
            details["knee_to_contact_px"] = [contact["rise_px"], info["knee_to_contact_rise_px"]]
            checks["cosine_rise"] = (
                info["knee_to_contact_source"] == contact["source"] == "cosine"
                and abs(info["knee_to_contact_rise_px"] - contact["rise_px"]) <= 0.1
            )
    else:
        review = load_json(ref / "recovery_review.json")
        first, last = review["catch_frame"], review["finish_frame"]
        info = client.post(info["recovery_review_url"], data={"catch": first, "finish": last}).raise_for_status().json()
        chart = info["catch_to_finish"]
        details["review_frames"] = [first, last]
        details["native_chart_counts"] = [review["distance_frames"], review["angle_frames"]]
        details["chart_counts"] = [chart["distance_frames"], chart["angle_frames"]]
        checks["graph_counts"] = details["chart_counts"] == details["native_chart_counts"]
        for col in ["forward_px", "trunk_angle_deg"]:
            a, b = before.loc[first:last, col], after.loc[first:last, col]
            details[col + "_coverage_disagreements"] = int((a.isna() != b.isna()).sum())
            checks[col] = bool(np.allclose(a, b, atol=0.1, rtol=0, equal_nan=True))
            details[col + "_max_delta"] = round(float((a-b).abs().max()), 5)
        png = client.get(info["files"]["catch_to_finish.png"]).raise_for_status().content
        with Image.open(BytesIO(png)) as image:
            image.verify()
            checks["png"] = image.format == "PNG"
    checks["saved_summary"] = client.get(info["summary_url"]).raise_for_status().json() == info
    return {**details, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--references", type=Path, default=ROOT / "outputs/file_analysis/api")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/docker_analysis")
    options = parser.parse_args()
    _, truth, _ = review_inputs(DATASET / "data.yaml")
    report = {}
    cases = {"a03_sn_f_01": {"knee_pass": 19, "drive": 25},
             "c04_sn_f_01": {"lift_off": 11, "knee_pass": 27, "drive": 43, "catch": 78, "finish": 107},
             "b00_sn_f_01": {}}
    with httpx.Client(base_url=f"http://127.0.0.1:{options.port}", trust_env=False, timeout=600) as client:
        for video, events in cases.items():
            report[video] = verify(client, video, events, options.references, options.results, truth)
            print(video, json.dumps(report[video], ensure_ascii=False), flush=True)
            (options.results / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if not all(all(row["checks"].values()) for row in report.values()):
        raise SystemExit("Mac 비교 결과에 차이가 있습니다. comparison.json을 확인하세요.")


if __name__ == "__main__":
    main()
