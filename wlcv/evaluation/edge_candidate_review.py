"""제외한 후보마다 원본의 해당 영역을 확대해 프레임 순서대로 검수한다."""
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from wlcv.config import VIDEOS
from wlcv.data.frames import frames_rgb, read_rgb
from wlcv.evaluation.plate_edge_filter import OUTPUT

COLORS = {"mac": "#ff7f0e", "cutmix": "#00a5e0"}
TILE_W, TILE_H = 210, 108
COLUMNS, ROWS = 6, 6


def crop_candidate(image, row):
    """원본에서 후보 주변을 여유 있게 잘라내고 선택된 후보를 테두리로 표시한다."""
    width, height = image.size
    pad = max(24, round((row.x2 - row.x1) / 4))
    left, top = max(0, int(row.x1)-pad), max(0, int(row.y1)-pad)
    right, bottom = min(width, int(row.x2)+pad+1), min(height, int(row.y2)+pad+1)
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((row.x1-left,row.y1-top,min(row.x2,right-1)-left,min(row.y2,bottom-1)-top),
                   outline=COLORS[row.model], width=max(1, crop.width // 170))
    return crop


def candidate_sheets():
    """페이지와 칸 번호를 CSV에 남겨 모든 제외 후보를 원본까지 되짚을 수 있게 한다."""
    dest = OUTPUT / "candidate_sheets"
    dest.mkdir(exist_ok=True)
    rejected = pd.read_csv(OUTPUT / "rejected.csv")
    index = []
    for video, group in rejected.groupby("video_id"):
        pages = []
        start = len(index)
        by_frame = {i: g for i, g in group.groupby("frame_index")}
        for i, rgb in enumerate(frames_rgb(VIDEOS / f"{video}.mp4")):
            if i not in by_frame:
                continue
            original = Image.fromarray(rgb)
            for row in by_frame[i].itertuples():
                page, slot = divmod(len(index) - start, ROWS*COLUMNS)
                if slot == 0:
                    pages.append(Image.new("RGB", (TILE_W*COLUMNS, TILE_H*ROWS), "#ffffff"))
                sheet = pages[page]
                x, y = (slot % COLUMNS)*TILE_W, (slot // COLUMNS)*TILE_H
                crop = ImageOps.contain(crop_candidate(original, row), (TILE_W-5, TILE_H-20))
                sheet.paste(crop,(x+(TILE_W-crop.width)//2,y+17))
                ImageDraw.Draw(sheet).text((x+4,y+2), f"f{i} {row.model}  p={row.confidence:.3f}",fill="#111111")
                index.append((row.Index, video, i, row.model, page+1, slot+1))
        for n, sheet in enumerate(pages, start=1):
            sheet.save(dest / f"{video}_{n:02d}.jpg", quality=88)
    listing = pd.DataFrame(index, columns=["candidate","video_id","frame_index","model","page","slot"])
    if len(listing) != len(rejected):
        raise ValueError("제외 후보 수와 원본 확대 그림 수가 다릅니다.")
    listing.to_csv(dest / "index.csv", index=False)
    return listing


def plot_frame(video, frame, *, index_in_frame=0):
    """후보 한 개의 원본 위치와 확대, 같은 프레임의 다른 제외 후보를 보여 준다."""
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    boxes = pd.read_csv(OUTPUT / "rejected.csv")
    rows = boxes[(boxes.video_id == video) & (boxes.frame_index == frame)]
    row = rows.iloc[index_in_frame]
    image = Image.fromarray(read_rgb(VIDEOS / f"{video}.mp4", frame))
    fig, axes = plt.subplots(1,2,figsize=(8,4.4),layout="constrained")
    axes[0].imshow(image)
    for r in rows.itertuples():
        axes[0].add_patch(Rectangle((r.x1,r.y1), r.x2-r.x1,r.y2-r.y1,fill=False,
                                    edgecolor=COLORS[r.model],linewidth=2))
    axes[1].imshow(crop_candidate(image,row))
    axes[0].set_title("Rejected candidates on original")
    axes[1].set_title(f"{row.model} | confidence {row.confidence:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{video} / frame {frame}")
    return fig


def plot_decisions(pair, examples):
    """필터 전후의 최종 출력과 수동 정답을 같은 원본에 각각 표시한다."""
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3.6*len(examples)), squeeze=False,
                             layout="constrained")
    for axes_row, (video, frame) in zip(axes, examples):
        item = pair[pair.video_id.eq(video) & pair.frame_index.eq(frame)
                    & pair.method.eq("Past link")].iloc[0]
        image = read_rgb(VIDEOS / f"{video}.mp4", frame)
        for ax, condition, label in zip(axes_row, ["truth","before","after"],
                                        ["Manual", "Filter off", "Filter on"]):
            ax.imshow(image)
            columns = [f"gt_{c}_before" if condition == "truth" else f"{c}_{condition}"
                       for c in ["x1", "y1", "x2", "y2"]]
            x, y, right, bottom = item[columns]
            if pd.notna(x):
                color = "#1fb964" if condition == "truth" else "#e94f13"
                ax.add_patch(Rectangle((x, y), right-x, bottom-y, fill=False, color=color, linewidth=2.5))
            value = "" if condition == "truth" else item[f"iou_{condition}"]
            if pd.notna(value) and condition != "truth":
                label += f" | IoU {value:.3f}"
            elif pd.isna(x):
                label += " | no box"
            ax.set_title(label)
            ax.set_xlabel(f"{video} / f{frame}")
            ax.set_xticks([])
            ax.set_yticks([])
    return fig
