#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prepared-prior cache whose fused_priors keep the aligned reference "
            "low-frequency content and inject only high-pass SR-prior residuals."
        )
    )
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--anchor_subdir", type=str, default="aligned_references")
    parser.add_argument("--mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--highpass_radius", type=float, default=3.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--delta_clip", type=float, default=0.18)
    parser.add_argument("--mask_power", type=float, default=0.75)
    parser.add_argument("--mask_floor", type=float, default=0.0)
    parser.add_argument("--copy_missing_anchor", action="store_true")
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def index_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in paths:
        if path.stem in index:
            duplicates.setdefault(path.stem, [str(index[path.stem])]).append(str(path))
            continue
        index[path.stem] = path
    if duplicates:
        examples = {key: vals[:4] for key, vals in list(duplicates.items())[:8]}
        raise ValueError(f"Duplicate {label} stems are ambiguous: {examples}")
    return index


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path | None, shape: tuple[int, int], *, mask_power: float, mask_floor: float) -> np.ndarray:
    if path is None or not path.is_file():
        mask = np.ones(shape, dtype=np.float32)
    else:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if mask.shape != shape:
            mask = np.asarray(
                Image.fromarray(np.clip(np.round(mask * 255.0), 0, 255).astype(np.uint8), mode="L").resize(
                    (shape[1], shape[0]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            ) / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    if float(mask_power) != 1.0:
        mask = np.power(np.clip(mask, 1e-8, 1.0), float(mask_power))
    if float(mask_floor) > 0.0:
        mask = np.clip(mask, float(mask_floor), 1.0)
    return mask[..., None]


def gaussian_blur_rgb(image: np.ndarray, radius: float) -> np.ndarray:
    if float(radius) <= 0.0:
        return image.astype(np.float32, copy=True)
    u8 = np.clip(np.round(np.clip(image, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    blurred = Image.fromarray(u8, mode="RGB").filter(ImageFilter.GaussianBlur(float(radius)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.round(np.clip(image, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def copy_tree(src: Path, dst: Path) -> int:
    if not src.is_dir():
        return 0
    copied = 0
    for item in sorted(src.rglob("*")):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    prior_dir = source_root / args.prior_subdir
    anchor_dir = source_root / args.anchor_subdir
    mask_dir = source_root / args.mask_subdir

    if not prior_dir.is_dir():
        raise FileNotFoundError(f"Missing prior dir: {prior_dir}")
    if not anchor_dir.is_dir():
        raise FileNotFoundError(f"Missing anchor/reference dir: {anchor_dir}")

    prior_by_stem = index_by_stem(iter_images(prior_dir), "prior")
    mask_by_stem = index_by_stem(iter_images(mask_dir), "mask")
    anchor_paths = iter_images(anchor_dir)
    if not anchor_paths:
        raise FileNotFoundError(f"No anchor images found under: {anchor_dir}")

    fused_out = output_root / "fused_priors"
    anchor_out = output_root / "aligned_references"
    mask_out = output_root / "usable_masks"
    fused_out.mkdir(parents=True, exist_ok=True)
    anchor_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    missing_priors: list[str] = []
    hp_abs_means: list[float] = []
    masked_abs_means: list[float] = []

    for anchor_path in anchor_paths:
        rel = anchor_path.relative_to(anchor_dir)
        prior_path = prior_by_stem.get(anchor_path.stem)
        if prior_path is None:
            missing_priors.append(str(rel))
            if bool(args.copy_missing_anchor):
                shutil.copy2(anchor_path, fused_out / rel)
                shutil.copy2(anchor_path, anchor_out / rel)
            continue

        anchor = load_rgb(anchor_path)
        prior = load_rgb(prior_path)
        if prior.shape != anchor.shape:
            prior_img = Image.open(prior_path).convert("RGB").resize(
                (anchor.shape[1], anchor.shape[0]),
                Image.Resampling.BICUBIC,
            )
            prior = np.asarray(prior_img, dtype=np.float32) / 255.0

        mask = load_mask(
            mask_by_stem.get(anchor_path.stem),
            anchor.shape[:2],
            mask_power=float(args.mask_power),
            mask_floor=float(args.mask_floor),
        )
        prior_high = prior - gaussian_blur_rgb(prior, float(args.highpass_radius))
        anchor_high = anchor - gaussian_blur_rgb(anchor, float(args.highpass_radius))
        high_delta = prior_high - anchor_high
        if float(args.delta_clip) > 0.0:
            high_delta = np.clip(high_delta, -float(args.delta_clip), float(args.delta_clip))
        injected = anchor + float(args.gain) * mask * high_delta
        injected = np.clip(injected, 0.0, 1.0)

        save_rgb(fused_out / rel, injected)
        shutil.copy2(anchor_path, anchor_out / rel)
        if anchor_path.stem in mask_by_stem:
            dst_mask = mask_out / rel.with_suffix(mask_by_stem[anchor_path.stem].suffix)
            dst_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_by_stem[anchor_path.stem], dst_mask)

        hp_abs = float(np.mean(np.abs(high_delta)))
        masked_abs = float(np.mean(np.abs(mask * high_delta)))
        hp_abs_means.append(hp_abs)
        masked_abs_means.append(masked_abs)
        frames.append(
            {
                "image_name": rel.name,
                "relative_path": str(rel),
                "prior_path": str(prior_path),
                "anchor_path": str(anchor_path),
                "output_path": str(fused_out / rel),
                "mask_path": str(mask_by_stem.get(anchor_path.stem)) if anchor_path.stem in mask_by_stem else None,
                "highpass_abs_mean": hp_abs,
                "masked_highpass_abs_mean": masked_abs,
            }
        )

    if not frames:
        raise RuntimeError("No HF residual prior frames were produced.")

    copied_masks = copy_tree(mask_dir, mask_out)
    manifest = {
        "version": "build_prior_hf_residual_cache_v0",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "prior_subdir": str(args.prior_subdir),
        "anchor_subdir": str(args.anchor_subdir),
        "mask_subdir": str(args.mask_subdir),
        "highpass_radius": float(args.highpass_radius),
        "gain": float(args.gain),
        "delta_clip": float(args.delta_clip),
        "mask_power": float(args.mask_power),
        "mask_floor": float(args.mask_floor),
        "num_frames": len(frames),
        "num_missing_priors": len(missing_priors),
        "missing_prior_examples": missing_priors[:16],
        "copied_masks": copied_masks,
        "highpass_abs_mean": float(np.mean(hp_abs_means)),
        "masked_highpass_abs_mean": float(np.mean(masked_abs_means)),
        "frames": frames,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
