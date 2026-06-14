#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose matching images from two directories into side-by-side pairs.")
    parser.add_argument("--left_dir", type=Path, required=True)
    parser.add_argument("--right_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--left_label", type=str, default="before")
    parser.add_argument("--right_label", type=str, default="after")
    parser.add_argument("--add_labels", action="store_true")
    parser.add_argument("--gap", type=int, default=8)
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def draw_label(image: Image.Image, text: str, x: int, y: int) -> None:
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((x, y), text)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))


def main() -> None:
    args = parse_args()
    left_dir = args.left_dir.expanduser().resolve()
    right_dir = args.right_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    left_files = {path.name: path for path in iter_images(left_dir)}
    right_files = {path.name: path for path in iter_images(right_dir)}
    common = sorted(set(left_files) & set(right_files))
    if not common:
        raise FileNotFoundError(f"No matching image names between {left_dir} and {right_dir}")

    for name in common:
        with Image.open(left_files[name]).convert("RGB") as left_img, Image.open(right_files[name]).convert("RGB") as right_img:
            width = left_img.width + int(args.gap) + right_img.width
            height = max(left_img.height, right_img.height)
            canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
            canvas.paste(left_img, (0, 0))
            canvas.paste(right_img, (left_img.width + int(args.gap), 0))
            if bool(args.add_labels):
                draw_label(canvas, str(args.left_label), 8, 8)
                draw_label(canvas, str(args.right_label), left_img.width + int(args.gap) + 8, 8)
            canvas.save(output_dir / name)

    print(f"[compose-side-by-side-v0] paired {len(common)} images -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
