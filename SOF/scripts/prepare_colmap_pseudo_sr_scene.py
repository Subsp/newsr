#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a COLMAP-style pseudo-SR scene alias by resizing source images "
            "to match a higher-resolution reference image directory while keeping the "
            "original sparse metadata."
        )
    )
    parser.add_argument("--scene_root", type=Path, required=True)
    parser.add_argument("--scene_alias_dir", type=Path, required=True)
    parser.add_argument("--source_images_subdir", type=str, required=True)
    parser.add_argument("--target_images_subdir", type=str, required=True)
    parser.add_argument(
        "--resize_filter",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
    )
    return parser.parse_args()


def _resize_filter(name: str) -> int:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping[name]


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _copy_non_image_metadata(
    scene_root: Path,
    scene_alias_dir: Path,
    source_images_subdir: str,
    target_images_subdir: str,
) -> list[str]:
    copied: list[str] = []
    for entry in sorted(scene_root.iterdir()):
        if entry.name in {"images", source_images_subdir, target_images_subdir}:
            continue
        if entry.name.startswith("images_"):
            continue
        dst = scene_alias_dir / entry.name
        _symlink(entry, dst)
        copied.append(entry.name)
    return copied


def _collect_source_images(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*") if _is_image(path))


def _resize_images(
    *,
    source_dir: Path,
    target_dir: Path,
    output_images_dir: Path,
    resize_filter_name: str,
) -> dict[str, object]:
    resize_filter = _resize_filter(resize_filter_name)
    entries: list[dict[str, object]] = []
    source_images = _collect_source_images(source_dir)
    if not source_images:
        raise FileNotFoundError(f"No source images found under: {source_dir}")

    for src in source_images:
        rel = src.relative_to(source_dir)
        target = target_dir / rel
        if not target.is_file():
            raise FileNotFoundError(f"Missing target-size reference image for: {rel}")

        dst = output_images_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as src_img, Image.open(target) as target_img:
            resized = src_img.resize(target_img.size, resize_filter)
            resized.save(dst)
            entries.append(
                {
                    "relative_path": str(rel),
                    "source_image": str(src.resolve()),
                    "target_reference": str(target.resolve()),
                    "output_image": str(dst.resolve()),
                    "source_size": [src_img.width, src_img.height],
                    "target_size": [target_img.width, target_img.height],
                }
            )

    return {
        "num_images": len(entries),
        "entries": entries,
    }


def main() -> None:
    args = _parse_args()
    scene_root = args.scene_root.resolve()
    scene_alias_dir = args.scene_alias_dir.resolve()
    source_dir = (scene_root / args.source_images_subdir).resolve()
    target_dir = (scene_root / args.target_images_subdir).resolve()
    output_images_dir = scene_alias_dir / "images"

    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image dir not found: {source_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target image dir not found: {target_dir}")

    scene_alias_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    copied_metadata = _copy_non_image_metadata(
        scene_root=scene_root,
        scene_alias_dir=scene_alias_dir,
        source_images_subdir=args.source_images_subdir,
        target_images_subdir=args.target_images_subdir,
    )
    image_summary = _resize_images(
        source_dir=source_dir,
        target_dir=target_dir,
        output_images_dir=output_images_dir,
        resize_filter_name=args.resize_filter,
    )

    summary = {
        "scene_root": str(scene_root),
        "scene_alias_dir": str(scene_alias_dir),
        "source_images_subdir": args.source_images_subdir,
        "target_images_subdir": args.target_images_subdir,
        "resize_filter": args.resize_filter,
        "copied_metadata": copied_metadata,
        **image_summary,
    }

    summary_path = scene_alias_dir / "pseudo_sr_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("[prepare-colmap-pseudo-sr-scene] done")
    print(f"  scene_alias_dir : {scene_alias_dir}")
    print(f"  source_images   : {source_dir}")
    print(f"  target_images   : {target_dir}")
    print(f"  alias_images    : {output_images_dir}")
    print(f"  num_images      : {summary['num_images']}")
    print(f"  resize_filter   : {args.resize_filter}")
    print(f"  summary         : {summary_path}")


if __name__ == "__main__":
    main()
