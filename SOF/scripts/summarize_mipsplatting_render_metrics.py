#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize PSNR/SSIM for a mip-splatting render directory."
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
    )
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    args = parse_args()
    root = args.model_dir.resolve() / args.split / f"ours_{args.iteration}"
    renders_dir = root / f"test_preds_{args.resolution}"
    gt_dir = root / f"gt_{args.resolution}"

    if not renders_dir.is_dir():
      raise FileNotFoundError(f"Render dir not found: {renders_dir}")
    if not gt_dir.is_dir():
      raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    render_files = iter_images(renders_dir)
    if not render_files:
        raise FileNotFoundError(f"No render images found in: {renders_dir}")

    psnrs: list[float] = []
    ssims: list[float] = []
    per_view: dict[str, dict[str, float]] = {}

    for render_path in render_files:
        gt_path = gt_dir / render_path.name
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT image for render: {gt_path}")

        render = load_rgb01(render_path)
        gt = load_rgb01(gt_path)

        psnr_value = float(peak_signal_noise_ratio(gt, render, data_range=1.0))
        ssim_value = float(structural_similarity(gt, render, channel_axis=2, data_range=1.0))

        psnrs.append(psnr_value)
        ssims.append(ssim_value)
        per_view[render_path.name] = {
            "PSNR": psnr_value,
            "SSIM": ssim_value,
        }

    summary = {
        "model_dir": str(args.model_dir.resolve()),
        "split": args.split,
        "iteration": int(args.iteration),
        "resolution": int(args.resolution),
        "n_views": len(render_files),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "per_view": per_view,
    }

    out_path = args.model_dir.resolve() / "results_psnr_ssim.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Scene: {args.model_dir}")
    print(f"Split: {args.split}")
    print(f"Views: {summary['n_views']}")
    print(f"  SSIM : {summary['SSIM']:>12.7f}")
    print(f"  PSNR : {summary['PSNR']:>12.7f}")
    print()
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
