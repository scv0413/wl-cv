"""같은 두 끝점의 허브와 수동·모델 외곽 박스 중심으로 상승량을 대조한다."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from wlcv.config import VIDEOS
from wlcv.data.frames import read_rgb
from wlcv.evaluation.phase_hub_rise import phase_rises
from wlcv.evaluation.plate_track import BOX, KEY


def compare_box_rises(pairs, repeat):
    """검증 6명의 첫 판독과 b02의 별도 판독을 분리한다. 이미 선택한 640 박스만 사용한다."""
    phases = phase_rises(pairs)
    saved = pairs[pairs.method.eq("Past link")].set_index(KEY, verify_integrity=True)
    repeated = repeat.set_index(KEY, verify_integrity=True)
    endpoints, intervals = [], []
    for phase in phases[phases.imgsz.eq(640)].itertuples():
        ends = []
        for event, index in [("knee", phase.knee_frame), ("contact", phase.contact_frame)]:
            key = (phase.video_id, index)
            row = saved.loc[key]
            box_y = (row.gt_y1 + row.gt_y2) / 2
            model_y = (row.y1_640 + row.y2_640) / 2
            if not np.isclose(model_y, row.cy_640, equal_nan=True):
                raise ValueError(f"{key}에서 선택 박스와 중심 좌표가 다릅니다.")
            entry = dict(video_id=phase.video_id, person_id=phase.person_id, event=event,
                         frame_index=index, box_source=row.box_source, model_source=row.source_640,
                         box_y=box_y, model_y=model_y, model_minus_box_y=model_y-box_y,
                         top_error=row.y1_640-row.gt_y1, bottom_error=row.y2_640-row.gt_y2)
            endpoints.append(entry)
            ends.append(entry)
        knee, contact = ends
        box_rise = knee["box_y"] - contact["box_y"]
        model_rise = knee["model_y"] - contact["model_y"]
        readings = {"first": phase.manual_rise_px}
        if phase.video_id == "b02_sn_h_02":
            pts = repeated.loc[[(phase.video_id, e["frame_index"]) for e in ends]]
            readings["repeat"] = pts.manual_y.iloc[0] - pts.manual_y.iloc[1] if pts.visibility.eq("visible").all() else np.nan
        for reading, hub_rise in readings.items():
            intervals.append(dict(person_id=phase.person_id, video_id=phase.video_id, reading=reading,
                                  knee_frame=phase.knee_frame, contact_frame=phase.contact_frame,
                                  hub_rise=hub_rise, box_rise=box_rise, model_rise=model_rise,
                                  box_minus_hub=box_rise-hub_rise, model_minus_box=model_rise-box_rise,
                                  model_minus_hub=model_rise-hub_rise))
    return pd.DataFrame(endpoints), pd.DataFrame(intervals)


def plot_components(intervals):
    primary = intervals[intervals.reading.eq("first")]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    x = np.arange(len(primary))
    ax.bar(x-.16, primary.box_minus_hub, width=.30, color="#b76421", label="Manual box minus hub")
    ax.bar(x+.16, primary.model_minus_box, width=.30, color="#2276a5", label="Model box minus manual box")
    ax.scatter(x, primary.model_minus_hub, marker="D", color="#20272f", label="Total (model minus hub)", zorder=3)
    ax.set(xticks=x, xticklabels=primary.person_id, ylabel="Difference in rise (px)",
           title="Knee pass to contact | input 640 | first blind reading")
    ax.axhline(0, color="gray", lw=.8)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    return fig


def show_b02(pairs, endpoints):
    """금색 테두리와 +는 수동 외곽 박스와 중심, 원은 1차 수동 허브다."""
    video = "b02_sn_h_02"
    saved = pairs[(pairs.video_id == video) & (pairs.method == "Past link")].set_index("frame_index")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), layout="constrained")
    for ax, end in zip(axes, endpoints[endpoints.video_id.eq(video)].itertuples(), strict=True):
        row = saved.loc[end.frame_index]
        ax.imshow(read_rgb(VIDEOS / f"{video}.mp4", end.frame_index))
        for keys, color, style, marker in [(["gt_"+c for c in BOX], "gold", "--", "+"),
                                           ([c+"_640" for c in BOX], "deepskyblue", "-", "x")]:
            x1, y1, x2, y2 = row[keys].to_numpy(float)
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1, fill=False,edgecolor=color,ls=style,lw=2))
            ax.plot((x1+x2)/2,(y1+y2)/2,marker,color=color,ms=11,mew=2)
        ax.plot(row.manual_x,row.manual_y,"o",fillstyle="none",color="white",ms=12,mew=2)
        ax.set(xlim=(min(row.gt_x1,row.x1_640)-30,max(row.gt_x2,row.x2_640)+30),
               ylim=(max(row.gt_y2,row.y2_640)+30,min(row.gt_y1,row.y1_640)-30),
               title=f"{end.event} | frame {end.frame_index}")
        ax.axis("off")
    fig.suptitle("b02 | white circle: manual hub   gold: manual box   blue: model box")
    return fig
