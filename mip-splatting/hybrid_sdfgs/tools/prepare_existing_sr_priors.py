#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare IE-SRGS-style discrepancy / usable-mask / fused-prior outputs "
            "from an existing prior image folder and a reference image folder."
        )
    )
    parser.add_argument("--prior_dir", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mask_threshold", type=float, default=0.12)
    parser.add_argument("--mask_mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument(
        "--discrepancy_floor",
        type=float,
        default=0.05,
        help="Lower bound on reference luma in the discrepancy denominator.",
    )
    parser.add_argument("--save_fused_priors", action="store_true")
    parser.add_argument("--copy_raw_priors", action="store_true")
    parser.add_argument("--save_discrepancy_npz", action="store_true")
    parser.add_argument(
        "--disable_usable_masks",
        action="store_true",
        help=(
            "Do not save usable_masks / masked_* artifacts, and treat fused priors "
            "as the raw prior images instead of reference-masked blends."
        ),
    )
    return parser.parse_args()


def _collect_images(folder: Path) -> list[Path]:
    images = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found under: {folder}")
    return images


def _index_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in _collect_images(folder)}


def _load_rgb01(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb01(path: Path, rgb: np.ndarray) -> None:
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    _atomic_save_image(Image.fromarray(rgb_u8, mode="RGB"), path)


def _save_gray01(path: Path, gray: np.ndarray) -> None:
    gray_u8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    _atomic_save_image(Image.fromarray(gray_u8, mode="L"), path)


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        image.save(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _ensure_same_size(reference_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if reference_rgb.shape[:2] == (target_h, target_w):
        return reference_rgb
    ref_u8 = np.clip(np.round(reference_rgb * 255.0), 0, 255).astype(np.uint8)
    resized = Image.fromarray(ref_u8, mode="RGB").resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _compute_discrepancy_and_mask(
    external_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    threshold: float,
    floor: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = _ensure_same_size(reference_rgb, external_rgb.shape[:2])
    ext_l = _rgb_to_luma(external_rgb)
    ref_l = _rgb_to_luma(ref)
    denom = np.maximum(np.abs(ref_l), floor)
    discrepancy = np.abs(ext_l - ref_l) / denom
    if mode == "hard":
        usable = (discrepancy < threshold).astype(np.float32)
    else:
        usable = np.clip(1.0 - discrepancy / max(threshold, 1e-8), 0.0, 1.0)
    fused = usable[..., None] * external_rgb + (1.0 - usable[..., None]) * ref
    return discrepancy.astype(np.float32), usable.astype(np.float32), fused.astype(np.float32)


def main() -> None:
    args = _parse_args()
    args.prior_dir = args.prior_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.output_root = args.output_root.resolve()

    if not args.prior_dir.is_dir():
        raise FileNotFoundError(f"Prior dir not found: {args.prior_dir}")
    if not args.reference_dir.is_dir():
        raise FileNotFoundError(f"Reference dir not found: {args.reference_dir}")

    prior_by_stem = _index_by_stem(args.prior_dir)
    ref_by_stem = _index_by_stem(args.reference_dir)

    priors_dir = args.output_root / "priors"
    discrepancy_dir = args.output_root / "discrepancy"
    mask_dir = args.output_root / "usable_masks"
    aligned_reference_dir = args.output_root / "aligned_references"
    masked_prior_dir = args.output_root / "masked_priors"
    masked_reference_dir = args.output_root / "masked_references"
    fused_dir = args.output_root / "fused_priors"
    npz_dir = args.output_root / "discrepancy_npz"
    save_usable_mask_artifacts = not bool(args.disable_usable_masks)

    args.output_root.mkdir(parents=True, exist_ok=True)
    if save_usable_mask_artifacts:
        discrepancy_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        aligned_reference_dir.mkdir(parents=True, exist_ok=True)
        masked_prior_dir.mkdir(parents=True, exist_ok=True)
        masked_reference_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_raw_priors:
        priors_dir.mkdir(parents=True, exist_ok=True)
    if args.save_fused_priors:
        fused_dir.mkdir(parents=True, exist_ok=True)
    if args.save_discrepancy_npz:
        npz_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, float | str]] = []
    missing_ref: list[str] = []

    for idx, (stem, prior_path) in enumerate(sorted(prior_by_stem.items()), start=1):
        print(f"[existing-prior] {idx}/{len(prior_by_stem)} {prior_path.name}")
        ref_path = ref_by_stem.get(stem)
        if ref_path is None:
            missing_ref.append(stem)
            continue

        if args.copy_raw_priors:
            shutil.copy2(prior_path, priors_dir / f"{stem}.png")

        sr_rgb = _load_rgb01(prior_path)
        ref_rgb = _load_rgb01(ref_path)
        discrepancy, usable, fused = _compute_discrepancy_and_mask(
            external_rgb=sr_rgb,
            reference_rgb=ref_rgb,
            threshold=args.mask_threshold,
            floor=args.discrepancy_floor,
            mode=args.mask_mode,
        )

        ref_resized = _ensure_same_size(ref_rgb, sr_rgb.shape[:2])
        if bool(args.disable_usable_masks):
            fused = sr_rgb
        else:
            masked_prior = usable[..., None] * sr_rgb
            masked_reference = usable[..., None] * ref_resized
            disc_vis = np.clip(discrepancy / max(args.mask_threshold * 2.0, 1e-8), 0.0, 1.0)
            _save_gray01(discrepancy_dir / f"{stem}.png", disc_vis)
            _save_gray01(mask_dir / f"{stem}.png", usable)
            _save_rgb01(aligned_reference_dir / f"{stem}.png", ref_resized)
            _save_rgb01(masked_prior_dir / f"{stem}.png", masked_prior)
            _save_rgb01(masked_reference_dir / f"{stem}.png", masked_reference)
        if args.save_fused_priors:
            _save_rgb01(fused_dir / f"{stem}.png", fused)
        if args.save_discrepancy_npz:
            np.savez_compressed(
                npz_dir / f"{stem}.npz",
                discrepancy=discrepancy,
                usable_mask=usable,
                fused=fused,
            )

        stats.append(
            {
                "image_name": prior_path.name,
                "stem": stem,
                "usable_mean": float(usable.mean()),
                "discrepancy_mean": float(discrepancy.mean()),
                "discrepancy_p90": float(np.percentile(discrepancy, 90.0)),
            }
        )

    manifest = {
        "prior_dir": str(args.prior_dir),
        "reference_dir": str(args.reference_dir),
        "output_root": str(args.output_root),
        "mask_threshold": args.mask_threshold,
        "mask_mode": args.mask_mode,
        "discrepancy_floor": args.discrepancy_floor,
        "save_fused_priors": bool(args.save_fused_priors),
        "copy_raw_priors": bool(args.copy_raw_priors),
        "save_discrepancy_npz": bool(args.save_discrepancy_npz),
        "disable_usable_masks": bool(args.disable_usable_masks),
        "num_priors": len(prior_by_stem),
        "num_matched": len(stats),
        "missing_reference_count": len(missing_ref),
        "missing_reference_examples": missing_ref[:20],
        "usable_mean": None if not stats else float(np.mean([x["usable_mean"] for x in stats])),
        "discrepancy_mean": None if not stats else float(np.mean([x["discrepancy_mean"] for x in stats])),
        "frames": stats,
    }
    _atomic_write_text(args.output_root / "manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if bool(args.disable_usable_masks):
        print("[existing-prior] usable masks disabled; training will use consistency gating only.")
    else:
        print(f"[existing-prior] usable masks saved to {mask_dir}")
    if args.save_fused_priors:
        print(f"[existing-prior] fused priors saved to {fused_dir}")


if __name__ == "__main__":
    main()
