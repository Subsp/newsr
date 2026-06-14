#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageFilter


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_contents(src_root: Path, dst_root: Path) -> int:
    if not src_root.is_dir():
        return 0
    copied = 0
    for src_path in sorted(src_root.rglob("*")):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        if src_path.is_dir():
            ensure_dir(dst_path)
            continue
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)
        copied += 1
    return copied


def blur_prior(prior_path: Path, output_path: Path, radius: float) -> None:
    with Image.open(prior_path) as image:
        rgb = image.convert("RGB")
        blurred = rgb.filter(ImageFilter.BoxBlur(radius=radius))
        ensure_dir(output_path.parent)
        blurred.save(output_path)


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Build a low-frequency prior cache by box-blurring prepared fused_priors "
            "and copying masks/anchors for staged curriculum runs."
        )
    )
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--anchor_subdir", type=str, default="aligned_references")
    parser.add_argument("--kernel_size", type=int, default=9)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    prior_src = source_root / str(args.prior_subdir)
    mask_src = source_root / str(args.mask_subdir)
    anchor_src = source_root / str(args.anchor_subdir)

    if not prior_src.is_dir():
        raise FileNotFoundError(f"Missing prior source dir: {prior_src}")

    kernel_size = max(int(args.kernel_size), 1)
    radius = float(max(kernel_size - 1, 0)) / 2.0

    prior_dst = output_root / "fused_priors"
    mask_dst = output_root / "usable_masks"
    anchor_dst = output_root / "aligned_references"
    ensure_dir(prior_dst)
    ensure_dir(mask_dst)
    ensure_dir(anchor_dst)

    processed = 0
    for prior_path in iter_image_files(prior_src):
        rel = prior_path.relative_to(prior_src)
        blur_prior(prior_path, prior_dst / rel, radius=radius)
        processed += 1

    copied_masks = copy_tree_contents(mask_src, mask_dst)
    copied_anchors = copy_tree_contents(anchor_src, anchor_dst)

    manifest = {
        "version": "build_lowpass_prior_cache_v0",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "prior_subdir": str(args.prior_subdir),
        "mask_subdir": str(args.mask_subdir),
        "anchor_subdir": str(args.anchor_subdir),
        "kernel_size": kernel_size,
        "box_blur_radius": radius,
        "processed_priors": processed,
        "copied_masks": copied_masks,
        "copied_anchors": copied_anchors,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
