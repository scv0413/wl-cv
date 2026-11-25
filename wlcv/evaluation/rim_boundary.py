"""원본에서 읽은 원판 아래 윤곽을 기준으로 패치를 수직 이동한다."""
from hashlib import sha256

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wlcv.evaluation.occlusion_dose import make_doses, score_frame
from wlcv.evaluation.patch_position import make_example, paste_patch

# 각 점은 원본 픽셀 (x, y)이다. 하단 중앙에서 다리·그림자와 겹치는 곳은 근삿값이다.
PROFILES = {
    "a03_sn_f_01_f19": [(347, 848), (355, 855), (360, 858), (366, 861), (373, 863),
                         (380, 864), (388, 863), (395, 861), (400, 859), (405, 856),
                         (410, 854), (415, 851), (418, 849), (421, 846)],
    "b02_sn_h_02_f30": [(197, 949), (210, 959), (225, 968), (240, 976), (250, 982),
                         (268, 989), (285, 996), (305, 1000), (325, 1002), (341, 998),
                         (358, 992), (377, 990), (389, 987), (400, 980), (410, 976),
                         (422, 969), (435, 959), (448, 950), (451, 948)],
}
SETTINGS = {"a03_sn_f_01_f19": (405, 3), "b02_sn_h_02_f30": (412, 5)}
CONDITIONS = ("Original", "Inside", "Outside", "Boundary")
SIZES = ("Small", "Previous20")


def place_at_rim(stem, left, patch):
    """패치 전 폭에서 안팎이 분리되도록 원판 안·밖으로 여유 픽셀을 둔다."""
    height, width = patch.shape[:2]
    path_x, path_y = np.array(PROFILES[stem]).T
    columns = np.arange(left, left + width)
    if columns[0] < path_x[0] or columns[-1] > path_x[-1]:
        raise ValueError("패치가 판독한 윤곽의 가로 범위를 벗어났습니다.")
    edge = np.interp(columns, path_x, path_y)
    gap = SETTINGS[stem][1]
    # 각 열에서 y가 경계보다 작으면 원판 쪽이다.
    tops = {"Inside": int(np.floor(edge.min())) - gap - height,
            "Outside": int(np.ceil(edge.max())) + gap,
            "Boundary": round(edge.mean() - (height - 1) / 2)}
    placements = {"Original": None}
    shares = {"Original": np.nan}
    for condition, top in tops.items():
        placements[condition] = (left, top, patch)
        ys = np.arange(top, top + height)[:, None]
        shares[condition] = (ys < edge).mean()
    if shares["Inside"] != 1 or shares["Outside"] != 0 or not .4 <= shares["Boundary"] <= .6:
        raise ValueError(f"{stem}의 안쪽·바깥·경계 위치를 다시 확인하세요.")
    return placements, shares


def prepare_cases(dataset, floors, previous):
    cases = {}
    examples = {}
    for stem in PROFILES:
        image, truth, _, donor, variants = make_example(dataset, floors, stem)
        placements, area = make_doses(truth, variants)
        left, _, patch = placements["Lower", 20]
        digest = sha256(patch.tobytes()).hexdigest()
        old = previous[(previous.stem == stem) & (previous.model == "Baseline")
                       & (previous.position == "Lower") & (previous.dose == 20)].iloc[0]
        if digest != old.patch_hash or donor != old.donor or left != old.x:
            raise ValueError(f"{stem}의 과거 20% 패치와 다릅니다.")
        examples[stem] = (image, truth, donor, patch, left, area)
    # 16×10 패치는 두 프레임에도 같은 픽셀을 붙이기 위해 a03의 기존 패치에서 한 번만 자른다.
    shared = examples["a03_sn_f_01_f19"][3][9:19, 27:43].copy()
    for stem, (image, truth, donor, large, left, area) in examples.items():
        for size in SIZES:
            x, patch, source = left, large, donor
            if size == "Small":
                x = SETTINGS[stem][0]
                patch = shared
                source = examples["a03_sn_f_01_f19"][2]
            placements, shares = place_at_rim(stem, x, patch)
            cases[stem, size] = dict(image=image, truth=truth, area=area, source=source,
                                     placements=placements, shares=shares)
    return cases


def measure(model, cases):
    rows = []
    originals = {}
    for (stem, size), case in tqdm(cases.items(), desc="두 프레임의 하단 가림", unit="묶음"):
        for condition, placement in case["placements"].items():
            image = paste_patch(case["image"], placement)
            if condition == "Original" and stem in originals:
                confidence, iou = originals[stem]
            else:
                confidence, iou = score_frame(model, image, case["truth"])
            if condition == "Original":
                originals[stem] = (confidence, iou)
            x = y = digest = None
            width = height = 0
            if placement is not None:
                x, y, patch = placement
                height, width = patch.shape[:2]
                digest = sha256(patch.tobytes()).hexdigest()
            rows.append(dict(stem=stem, size=size, condition=condition, x=x, y=y, width=width,
                             height=height, patch_hash=digest, donor=case["source"] if width else None,
                             pct_box_area=100 * width * height / case["area"],
                             pct_patch_inside=100 * case["shares"][condition], confidence=confidence,
                             iou=iou, detected=confidence >= .10))
    return pd.DataFrame(rows)
