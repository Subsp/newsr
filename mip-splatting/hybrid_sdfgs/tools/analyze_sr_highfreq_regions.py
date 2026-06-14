#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def _progress(iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        print(f"[hf-analysis] {desc}...")
        return iterable


def _list_images(directory: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
    root = Path(directory)
    return sorted([str(p) for p in root.iterdir() if p.suffix in exts], key=lambda x: Path(x).name)


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path).convert("RGB") as img:
        return _to_float01(np.asarray(img))


def _save_rgb(path: str, arr: np.ndarray):
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _resize(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = img.resize((w, h), Image.BICUBIC)
    return _to_float01(np.asarray(img))


def _blur(arr: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return _to_float01(np.asarray(img))


def _rgb_to_luma(arr: np.ndarray) -> np.ndarray:
    return (
        0.299 * arr[..., 0]
        + 0.587 * arr[..., 1]
        + 0.114 * arr[..., 2]
    ).astype(np.float32)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(mean(vals)) if vals else float("nan")


def _heat_to_rgb01(score01: np.ndarray) -> np.ndarray:
    score01 = np.clip(score01, 0.0, 1.0).astype(np.float32)
    r = np.clip(1.5 * score01 - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * score01 - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * score01, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _overlay(base: np.ndarray, mask01: np.ndarray, color_rgb01: tuple[float, float, float], alpha_scale: float) -> np.ndarray:
    alpha = np.clip(mask01, 0.0, 1.0)[..., None] * alpha_scale
    color = np.zeros_like(base) + np.array(color_rgb01, dtype=np.float32)
    return np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 1.0)


def _write_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether SR outputs contain effective high-frequency details and visualize usable regions."
    )
    parser.add_argument("--sr_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--blur_radius", type=float, default=1.6)
    parser.add_argument("--hf_quantile", type=float, default=0.75)
    parser.add_argument("--improve_margin", type=float, default=0.01)
    parser.add_argument("--align_thresh", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=24)
    args = parser.parse_args()

    sr_dir = os.path.abspath(os.path.expanduser(args.sr_dir))
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    gt_dir = os.path.abspath(os.path.expanduser(args.gt_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    sr_paths = _list_images(sr_dir)
    input_paths = _list_images(input_dir)
    gt_paths = _list_images(gt_dir)
    if not sr_paths:
        raise FileNotFoundError(f"No SR images found in {sr_dir}")
    if not input_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")
    if not gt_paths:
        raise FileNotFoundError(f"No GT images found in {gt_dir}")

    sr_by_stem = {Path(p).stem: p for p in sr_paths}
    input_by_stem = {Path(p).stem: p for p in input_paths}
    gt_by_stem = {Path(p).stem: p for p in gt_paths}
    stems = [Path(p).stem for p in gt_paths if Path(p).stem in sr_by_stem and Path(p).stem in input_by_stem]
    if not stems:
        raise RuntimeError("No common stems across SR/LR/GT directories")

    vis_root = Path(output_dir) / "visuals"
    mask_root = vis_root / "usable_masks"
    overlay_root = vis_root / "usable_overlays"
    heat_root = vis_root / "improve_heatmaps"
    side_root = vis_root / "side_by_side"
    crop_root = vis_root / "top_regions"
    for d in [mask_root, overlay_root, heat_root, side_root, crop_root]:
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    top_regions: list[dict[str, Any]] = []

    for stem in _progress(stems, "analyzing high-frequency usefulness", total=len(stems)):
        sr = _load_rgb(sr_by_stem[stem])
        gt = _load_rgb(gt_by_stem[stem])
        lr = _load_rgb(input_by_stem[stem])
        h, w = gt.shape[:2]
        if sr.shape[:2] != (h, w):
            sr = _resize(sr, (h, w))
        bicubic = _resize(lr, (h, w))

        gt_lp = _blur(gt, args.blur_radius)
        sr_lp = _blur(sr, args.blur_radius)
        bi_lp = _blur(bicubic, args.blur_radius)

        hf_gt = gt - gt_lp
        hf_sr = sr - sr_lp
        hf_bi = bicubic - bi_lp

        hf_gt_l = _rgb_to_luma(hf_gt)
        hf_sr_l = _rgb_to_luma(hf_sr)
        hf_bi_l = _rgb_to_luma(hf_bi)

        target_strength = np.abs(hf_gt_l)
        err_sr = np.abs(hf_sr_l - hf_gt_l)
        err_bi = np.abs(hf_bi_l - hf_gt_l)
        improve = err_bi - err_sr

        denom = np.maximum(np.abs(hf_gt_l), 1e-4)
        align = 1.0 - np.clip(np.abs(hf_sr_l - hf_gt_l) / denom, 0.0, 1.0)

        hf_thresh = float(np.quantile(target_strength, args.hf_quantile))
        useful_mask = (
            (target_strength >= hf_thresh)
            & (improve > args.improve_margin)
            & (align >= args.align_thresh)
        )

        positive_improve = np.maximum(improve, 0.0)
        useful_score = positive_improve * useful_mask.astype(np.float32)
        max_score = float(useful_score.max())
        useful_score01 = useful_score / max(max_score, 1e-6)

        heat = _heat_to_rgb01(np.clip((improve - args.improve_margin) / max(0.05, max_score + 1e-6), 0.0, 1.0))
        mask_rgb = np.repeat(useful_mask[..., None].astype(np.float32), 3, axis=-1)
        overlay = _overlay(sr, useful_score01, (0.0, 1.0, 0.1), 0.55)

        side = np.concatenate([bicubic, sr, gt, overlay], axis=1)
        _save_rgb(str(mask_root / f"{stem}.png"), mask_rgb)
        _save_rgb(str(overlay_root / f"{stem}.png"), overlay)
        _save_rgb(str(heat_root / f"{stem}.png"), heat)
        _save_rgb(str(side_root / f"{stem}.png"), side)

        usable_ratio = float(np.mean(useful_mask))
        mean_improve = float(np.mean(improve))
        mean_useful_improve = float(np.mean(improve[useful_mask])) if np.any(useful_mask) else 0.0
        hf_psnr_sr = _psnr(hf_sr_l * 0.5 + 0.5, hf_gt_l * 0.5 + 0.5)
        hf_psnr_bi = _psnr(hf_bi_l * 0.5 + 0.5, hf_gt_l * 0.5 + 0.5)

        ys, xs = np.where(useful_mask)
        if len(xs) > 0:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            box_area = max((y1 - y0) * (x1 - x0), 1)
            region_score = float(useful_score[y0:y1, x0:x1].mean())
            top_regions.append({
                "stem": stem,
                "region_score": region_score,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "usable_pixels": int(useful_mask.sum()),
                "usable_ratio": usable_ratio,
            })
            sr_crop = sr[y0:y1, x0:x1]
            gt_crop = gt[y0:y1, x0:x1]
            bi_crop = bicubic[y0:y1, x0:x1]
            overlay_crop = overlay[y0:y1, x0:x1]
            crop_panel = np.concatenate([bi_crop, sr_crop, gt_crop, overlay_crop], axis=1)
            _save_rgb(str(crop_root / f"{stem}.png"), crop_panel)
        else:
            top_regions.append({
                "stem": stem,
                "region_score": 0.0,
                "x0": 0,
                "y0": 0,
                "x1": 0,
                "y1": 0,
                "usable_pixels": 0,
                "usable_ratio": usable_ratio,
            })

        rows.append({
            "stem": stem,
            "usable_ratio": usable_ratio,
            "usable_pixels": int(useful_mask.sum()),
            "hf_thresh": hf_thresh,
            "mean_improve": mean_improve,
            "mean_useful_improve": mean_useful_improve,
            "hf_psnr_sr": hf_psnr_sr,
            "hf_psnr_bicubic": hf_psnr_bi,
            "hf_psnr_gain": hf_psnr_sr - hf_psnr_bi,
            "sr_psnr": _psnr(sr, gt),
            "bicubic_psnr": _psnr(bicubic, gt),
            "delta_psnr_over_bicubic": _psnr(sr, gt) - _psnr(bicubic, gt),
            "max_useful_score": max_score,
        })

    top_regions = sorted(top_regions, key=lambda x: x["region_score"], reverse=True)
    _write_csv(str(Path(output_dir) / "per_frame_highfreq.csv"), rows)
    _write_csv(str(Path(output_dir) / "top_regions.csv"), top_regions[: args.top_k])

    summary = {
        "sr_dir": sr_dir,
        "input_dir": input_dir,
        "gt_dir": gt_dir,
        "frames": len(rows),
        "mean_usable_ratio": _safe_mean([r["usable_ratio"] for r in rows]),
        "mean_hf_psnr_gain": _safe_mean([r["hf_psnr_gain"] for r in rows]),
        "mean_delta_psnr_over_bicubic": _safe_mean([r["delta_psnr_over_bicubic"] for r in rows]),
        "mean_mean_useful_improve": _safe_mean([r["mean_useful_improve"] for r in rows]),
        "top_k_regions": top_regions[: args.top_k],
        "visual_dirs": {
            "usable_masks": str(mask_root),
            "usable_overlays": str(overlay_root),
            "improve_heatmaps": str(heat_root),
            "side_by_side": str(side_root),
            "top_regions": str(crop_root),
        },
    }
    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[hf-analysis] done")
    print(f"  sr_dir            : {sr_dir}")
    print(f"  input_dir         : {input_dir}")
    print(f"  gt_dir            : {gt_dir}")
    print(f"  output_dir        : {output_dir}")
    print(f"  frames            : {len(rows)}")
    print(f"  mean usable ratio : {summary['mean_usable_ratio']:.4f}")
    print(f"  mean HF gain      : {summary['mean_hf_psnr_gain']:.4f}")
    print(f"  mean dPSNR        : {summary['mean_delta_psnr_over_bicubic']:.4f}")


if __name__ == "__main__":
    main()
