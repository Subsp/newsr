from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def iter_images(render_dir: Path) -> List[Path]:
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in render_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def select_uniform(paths: List[Path], max_images: int) -> List[Path]:
    if int(max_images) <= 0 or len(paths) <= int(max_images):
        return paths
    ids = np.unique(np.linspace(0, len(paths) - 1, num=int(max_images), dtype=np.int64))
    return [paths[int(idx)] for idx in ids.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a contact sheet from rendered images.")
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_images", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb_width", type=int, default=360)
    parser.add_argument("--label_height", type=int, default=24)
    parser.add_argument("--padding", type=int, default=8)
    args = parser.parse_args()

    render_dir = Path(args.render_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not render_dir.is_dir():
        raise FileNotFoundError(f"render_dir not found: {render_dir}")

    paths = select_uniform(iter_images(render_dir), int(args.max_images))
    if not paths:
        raise FileNotFoundError(f"No render images found under: {render_dir}")

    thumbs = []
    thumb_width = max(int(args.thumb_width), 32)
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, thumb_width), Image.Resampling.LANCZOS)
        tile = ImageOps.pad(image, (thumb_width, thumb_width), method=Image.Resampling.LANCZOS, color=(18, 18, 18))
        thumbs.append((path, tile))

    columns = max(int(args.columns), 1)
    rows = int(math.ceil(len(thumbs) / columns))
    padding = max(int(args.padding), 0)
    label_height = max(int(args.label_height), 0)
    tile_w = thumb_width
    tile_h = thumb_width + label_height
    sheet_w = columns * tile_w + (columns + 1) * padding
    sheet_h = rows * tile_h + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_w, sheet_h), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (path, tile) in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)
        sheet.paste(tile, (x, y))
        if label_height > 0:
            label = path.stem
            draw.text((x + 4, y + thumb_width + 5), label[:48], fill=(230, 230, 230), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"[contact-sheet] saved: {output_path}")
    print(f"[contact-sheet] source: {render_dir}")
    print(f"[contact-sheet] images: {len(thumbs)} / {len(iter_images(render_dir))}")


if __name__ == "__main__":
    main()
