"""경계의 납작한 박스가 직전 궤적을 잇는 경우에만 허용한다."""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wlcv.config import ROOT
from wlcv.evaluation.box_size_review import iou
from wlcv.evaluation.plate_cutmix_fallback import fill_missing_boxes
from wlcv.evaluation.plate_edge_filter import OUTPUT as CANDIDATES
from wlcv.evaluation.plate_track import BOX, KEY, PlateHistory, choose_plate

OUTPUT = ROOT / "outputs/plate_edge_continuity"


def model_track(model, rule, sizes):
    frames = pd.read_csv(CANDIDATES / f"{model}_frames.csv")
    candidates = pd.read_csv(CANDIDATES / f"{model}_candidates.csv", float_precision="round_trip")
    groups = dict(tuple(candidates.groupby(KEY)))
    empty = candidates.iloc[:0]
    rows = []
    for video, segment in tqdm(frames.groupby("video_id"), desc=f"{model} {rule}", unit="편"):
        width, height, length = sizes.loc[video, ["width","height","frame_count"]]
        if segment.frame_index.tolist() != list(range(length)):
            raise ValueError(f"{video}의 연속 프레임이 누락됐습니다.")
        history = PlateHistory()
        for frame in segment.frame_index:
            box, source = choose_plate(history, groups.get((video, frame), empty), video, frame,
                                       width, height, rule)
            coords = [np.nan]*4 if box is None else box[BOX].tolist()
            rows.append([video, frame, "Past link", source, *coords])
    return pd.DataFrame(rows, columns=KEY+["method","source",*BOX])


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sizes = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id")
    previous = pd.read_csv(CANDIDATES / "paired.csv").query("method == 'Past link'").set_index(KEY)
    truth_cols = [f"gt_{c}_before" for c in BOX]
    runs = []
    for rule in ("off","strict","continuity"):
        tracks = [model_track(model, rule, sizes) for model in ("mac","cutmix")]
        result = fill_missing_boxes(*tracks).set_index(KEY).sort_index()
        if not result.index.equals(previous.sort_index().index) or len(result) != 863:
            raise ValueError("검증 영상과 배경의 프레임이 일치해야 합니다.")
        if rule in ("off","strict"):
            suffix = "before" if rule == "off" else "after"
            for col in ["source","used_cutmix",*BOX]:
                pd.testing.assert_series_equal(result[col],previous[f"{col}_{suffix}"].sort_index(),
                                               check_names=False,check_dtype=False)
        result = result.join(previous[["in_motion",*truth_cols]])
        result["iou"] = [iou(box, truth) if np.isfinite(truth).all() else np.nan
                         for box, truth in zip(result[BOX].to_numpy(),result[truth_cols].to_numpy())]
        result["hit"] = result.iou.ge(0.5).where(result[truth_cols[0]].notna()).astype("boolean")
        runs.append(result.reset_index().assign(rule=rule))
    all_frames = pd.concat(runs,ignore_index=True)
    all_frames.to_csv(OUTPUT / "frames.csv",index=False)
    counts = []
    for rule, g in all_frames.groupby("rule",sort=False):
        motion = g[g.in_motion & g.hit.notna()].set_index(KEY)
        reference = previous.loc[motion.index,"match_before"].astype(bool)
        a03 = motion.loc["a03_sn_f_01"].loc[[4,5,6,10,11,12,13,14]]
        bg = g[g.video_id.eq("b06_sn_f_01")].sort_values("frame_index").x1.notna()
        b02 = g[g.video_id.eq("b02_sn_h_02")].set_index("frame_index")
        counts.append(dict(rule=rule, scored=len(motion),hits=int(motion.hit.sum()),
                           lost=int((reference & ~motion.hit).sum()),gained=int((~reference & motion.hit).sum()),
                           a03_saved=int(a03.hit.sum()),b06_fp=int(bg.sum()),
                           b06_longest=int(bg.groupby((~bg).cumsum()).sum().max()),
                           b05_fp=int(g.loc[g.video_id.eq("b05_sn_f_01"),"x1"].count()),
                           b02_iou_77=b02.at[77,"iou"],b02_iou_78=b02.at[78,"iou"]))
    table=pd.DataFrame(counts)
    table.to_csv(OUTPUT / "summary.csv",index=False)
    return table
