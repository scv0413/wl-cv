"""학습 선수를 한 명씩 제외해 무릎 통과 시점을 예측한다. Mac CPU용 소규모 비교."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from wlcv.annotation.storage import load_events
from wlcv.config import ANNOTATIONS, ROOT, SPLIT
from wlcv.evaluation.knee_pass_height import first_crossing


def clips(folder):
    tracks = pd.read_csv(folder / "frames.csv")
    marks = pd.read_csv(folder / "readings.csv").set_index("video")
    split = pd.read_csv(SPLIT).set_index("video_id").loc[marks.index]
    if not split.role.eq("train").all() or split.person_id.nunique() != len(marks):
        raise ValueError("학습 선수당 영상 한 편이 필요합니다.")
    heights = pd.read_csv(ROOT / "data/metadata/videos.csv").set_index("video_id").height
    data = {}
    for video, mark in marks.iterrows():
        events = load_events(ANNOTATIONS, video, 1)
        # 실제로 제공받을 수 있는 두 끝점만 사용한다. 통과 라벨은 채점과 훈련 목표에만 쓴다.
        part = tracks[tracks.video.eq(video) & tracks.frame.between(events["ready"], events["drive"])].set_index("frame")
        if part.index.tolist() != list(range(events["ready"], events["drive"] + 1)):
            raise ValueError(f"{video} 프레임 순서 오류")
        h = heights.at[video]
        plate, knee = part.plate_y/h, part.knee_y/h
        paired = plate.notna() & knee.notna()
        channels = [plate.fillna(0), knee.fillna(0), (knee-plate).fillna(0), plate.notna(), knee.notna()]
        x = torch.tensor(np.array(channels, dtype=np.float32)).unsqueeze(0)
        # 두 겹의 합성곱이 보는 7장 안에 실제 동시 관측이 한 장 이상 있는 후보만 채점한다.
        support = torch.tensor(paired.rolling(7, center=True, min_periods=1).max().astype(bool).to_numpy())
        lo, hi = sorted((mark["first"], mark["second"]))
        target = torch.tensor(part.index.to_series().between(lo, hi).to_numpy())
        data[video] = dict(x=x, support=support, target=target, lo=lo, hi=hi, track=part)
    return data


def network():
    return nn.Sequential(nn.Conv1d(5,8,3,padding=1),nn.ReLU(),
                         nn.Conv1d(8,8,3,dilation=2,padding=2),nn.ReLU(),nn.Conv1d(8,1,1))


def fit(train, path, seed, epochs=40):
    torch.manual_seed(seed)
    model = network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    examples = [s for s in train if (s["support"] & s["target"]).any()]
    if not examples:
        raise ValueError("학습 가능한 양성 시점이 없습니다.")
    best_loss = float("inf")
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = []
        for s in examples:
            logits = model(s["x"])[0,0].masked_fill(~s["support"], -torch.inf)
            # 정답 범위 안에 들어간 확률 질량을 높인다. 모든 시도의 비중은 같다.
            log_p = F.log_softmax(logits, dim=0)
            losses.append(-torch.logsumexp(log_p[s["target"] & s["support"]], dim=0))
        loss = torch.stack(losses).mean()
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), path)
        loss.backward()
        optimizer.step()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval(), len(examples)


def result(video, s, method, frame, seed=None, n_train=None):
    lo, hi, track = s["lo"], s["hi"], s["track"]
    row = dict(video=video, method=method, seed=seed, train_used=n_train,
               first_bound=lo, last_bound=hi, predicted=frame, outside_ms=np.nan, outside_frames=np.nan)
    if frame is not None:
        boundary = max(lo,min(frame,hi))
        row.update(outside_frames=frame-boundary, outside_ms=track.at[frame,"time_ms"]-track.at[boundary,"time_ms"])
    return row


def run(source, seeds=(0,1,2)):
    """7회차에 동일 설정을 쓰고 seed 0을 주 비교로 지정한다. 가중치는 로컬에 보관한다."""
    data = clips(source)
    dest = ROOT / "outputs/knee_pass_tcn" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dest.mkdir(parents=True)
    rows = []
    for video, s in tqdm(data.items(), desc="제외 선수", unit="명"):
        t = s["track"]
        midpoint = (t.time_ms.iloc[0]+t.time_ms.iloc[-1])/2
        mid_frame = int((t.time_ms-midpoint).abs().idxmin())
        auto, _ = first_crossing(t.gap_px)
        rows += [result(video,s,"시간 중앙",mid_frame), result(video,s,"높이 규칙",auto)]
        for seed in seeds:
            ckpt = dest / f"{video}_seed{seed}"
            ckpt.mkdir()
            model,n = fit([ex for name,ex in data.items() if name != video], ckpt/"best.pt",seed)
            frame = None
            if s["support"].any():
                with torch.no_grad():
                    scores = model(s["x"])[0,0].masked_fill(~s["support"],-torch.inf)
                frame = int(t.index[int(scores.argmax())])
            rows.append(result(video,s,"TCN",frame,seed,n))
    table = pd.DataFrame(rows)
    table.to_csv(dest/"scores.csv",index=False)
    (dest/"source.txt").write_text(str(Path(source).resolve())+"\n")
    return dest,table
