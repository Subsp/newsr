#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _parse_layer(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Layer must use label=render_dir format, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Layer label is empty: {raw}")
    return label, Path(path).expanduser().resolve()


def _iter_images(render_dir: Path) -> List[Path]:
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in render_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _select_uniform(paths: List[Path], max_images: int) -> List[Path]:
    if int(max_images) <= 0 or len(paths) <= int(max_images):
        return paths
    ids = np.unique(np.linspace(0, len(paths) - 1, num=int(max_images), dtype=np.int64))
    return [paths[int(idx)] for idx in ids.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a row-by-view, column-by-layer contact sheet.")
    parser.add_argument("--layer", action="append", required=True, help="Layer in label=render_dir format.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--thumb_width", type=int, default=360)
    parser.add_argument("--label_height", type=int, default=28)
    parser.add_argument("--padding", type=int, default=8)
    args = parser.parse_args()

    layers = [_parse_layer(item) for item in args.layer]
    for label, render_dir in layers:
        if not render_dir.is_dir():
            raise FileNotFoundError(f"Layer '{label}' render dir not found: {render_dir}")

    reference_paths = _select_uniform(_iter_images(layers[0][1]), int(args.max_images))
    if not reference_paths:
        raise FileNotFoundError(f"No render images found under: {layers[0][1]}")

    thumb_width = max(int(args.thumb_width), 32)
    label_height = max(int(args.label_height), 0)
    padding = max(int(args.padding), 0)
    columns = len(layers)
    rows = len(reference_paths)
    tile_w = thumb_width
    tile_h = thumb_width + label_height
    sheet_w = columns * tile_w + (columns + 1) * padding
    sheet_h = rows * tile_h + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_w, sheet_h), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row, ref_path in enumerate(reference_paths):
        for col, (label, render_dir) in enumerate(layers):
            path = render_dir / ref_path.name
            if not path.exists():
                image = Image.new("RGB", (thumb_width, thumb_width), (42, 12, 12))
                tile_label = f"{label}: missing {ref_path.name}"
            else:
                image = Image.open(path).convert("RGB")
                image.thumbnail((thumb_width, thumb_width), Image.Resampling.LANCZOS)
                tile_label = f"{label}: {ref_path.stem}"
            tile = ImageOps.pad(image, (thumb_width, thumb_width), method=Image.Resampling.LANCZOS, color=(18, 18, 18))
            x = padding + col * (tile_w + padding)
            y = padding + row * (tile_h + padding)
            sheet.paste(tile, (x, y))
            if label_height > 0:
                draw.text((x + 4, y + thumb_width + 6), tile_label[:64], fill=(230, 230, 230), font=font)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"[layer-sheet] saved: {output_path}")
    print(f"[layer-sheet] layers: {', '.join(label for label, _ in layers)}")
    print(f"[layer-sheet] images: {len(reference_paths)}")


if __name__ == "__main__":
    main()
