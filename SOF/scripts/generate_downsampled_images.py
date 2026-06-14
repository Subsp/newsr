#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a downsampled image directory from an existing image directory."
    )
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument(
        "--resize_filter",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_path", type=Path, default=None)
    return parser.parse_args()


def resize_filter(name: str) -> int:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping[name]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else output_dir / "downsample_summary.json"

    if args.scale <= 1.0:
        raise ValueError("--scale must be > 1.0")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    filter_mode = resize_filter(args.resize_filter)
    source_images = sorted(path for path in source_dir.rglob("*") if is_image(path))
    if not source_images:
        raise FileNotFoundError(f"No images found under: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for src in source_images:
        rel = src.relative_to(source_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.overwrite:
            continue

        with Image.open(src) as image:
            out_w = max(1, int(round(image.width / args.scale)))
            out_h = max(1, int(round(image.height / args.scale)))
            resized = image.resize((out_w, out_h), resample=filter_mode)
            resized.save(dst)
            entries.append(
                {
                    "relative_path": str(rel),
                    "source_size": [image.width, image.height],
                    "output_size": [out_w, out_h],
                    "output_path": str(dst),
                }
            )

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "scale": float(args.scale),
        "resize_filter": args.resize_filter,
        "processed_count": len(entries),
        "entries": entries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[generate-downsampled-images] done")
    print(f"  source_dir     : {source_dir}")
    print(f"  output_dir     : {output_dir}")
    print(f"  processed_count: {len(entries)}")
    print(f"  summary        : {summary_path}")


if __name__ == "__main__":
    main()
