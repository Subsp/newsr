#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone x8-to-x2 prior fusion pipeline for MipNeRF360-style scenes."
    )
    parser.add_argument("--scene_root", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_subdir", type=str, default="images_8")
    parser.add_argument("--gt_subdir", type=str, default="images_2")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--hf_weight", type=float, default=0.75)
    parser.add_argument("--delta_clip", type=float, default=0.10)
    parser.add_argument("--consistency_tau", type=float, default=0.08)
    parser.add_argument("--energy_floor", type=float, default=0.002)
    parser.add_argument("--gain_power", type=float, default=1.0)
    parser.add_argument("--write_debug", action="store_true")
    args = parser.parse_args()

    from pipeline import FusionConfig, run_pipeline

    if args.scene_root:
        scene_root = Path(args.scene_root)
        input_dir = Path(args.input_dir) if args.input_dir else scene_root / args.input_subdir
        gt_dir = Path(args.gt_dir) if args.gt_dir else scene_root / args.gt_subdir
        gt_dir_str = str(gt_dir) if gt_dir.exists() else ""
    else:
        if not args.input_dir:
            raise ValueError("Either --scene_root or --input_dir must be provided.")
        input_dir = Path(args.input_dir)
        gt_dir_str = args.gt_dir

    cfg = FusionConfig(
        levels=args.levels,
        hf_weight=args.hf_weight,
        delta_clip=args.delta_clip,
        consistency_tau=args.consistency_tau,
        energy_floor=args.energy_floor,
        gain_power=args.gain_power,
        scale=args.scale,
        write_debug=args.write_debug,
    )

    summary = run_pipeline(
        input_dir=str(input_dir),
        prior_dir=args.prior_dir,
        output_dir=args.output_dir,
        gt_dir=gt_dir_str if gt_dir_str else None,
        cfg=cfg,
    )

    print("[x8to2] finished")
    print(f"[x8to2] frames: {summary['num_frames']}")
    print(f"[x8to2] renders: {Path(args.output_dir) / 'renders'}")
    if "quick_eval" in summary:
        fused = summary["quick_eval"]["fused"]
        bicubic = summary["quick_eval"]["bicubic"]
        delta = summary["quick_eval"]["delta"]
        print(
            "[x8to2] fused   "
            f"PSNR={fused['psnr']:.4f} SSIM={fused['ssim']:.4f} MAE={fused['mae']:.6f}"
        )
        print(
            "[x8to2] bicubic "
            f"PSNR={bicubic['psnr']:.4f} SSIM={bicubic['ssim']:.4f} MAE={bicubic['mae']:.6f}"
        )
        print(
            "[x8to2] delta   "
            f"PSNR={delta['psnr']:.4f} SSIM={delta['ssim']:.4f} MAE={delta['mae']:.6f}"
        )


if __name__ == "__main__":
    main()
