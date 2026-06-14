#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _progress(iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        print(f"[prior-mask-eval] {desc}...")
        return iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SR prior and usable-mask quality from an existing prior output root. "
            "Produces per-frame metrics, aggregate summaries, and qualitative review panels."
        )
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--analysis_dir", type=Path, default=None)
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--reference_dir", type=Path, default=None)
    parser.add_argument("--prior_subdir", type=str, default="priors")
    parser.add_argument("--mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--reference_subdir", type=str, default="aligned_references")
    parser.add_argument("--fused_subdir", type=str, default="fused_priors")
    parser.add_argument("--hard_mask_threshold", type=float, default=0.5)
    parser.add_argument(
        "--oracle_margin",
        type=float,
        default=0.0,
        help="A pixel is oracle-good only if prior error <= bicubic error - oracle_margin.",
    )
    parser.add_argument(
        "--discrepancy_floor",
        type=float,
        default=-1.0,
        help="If negative, reuse discrepancy_floor from manifest.json when available.",
    )
    parser.add_argument("--top_k", type=int, default=12)
    return parser.parse_args()


def _collect_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    images = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found under: {folder}")
    return images


def _index_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in _collect_images(folder)}


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path).convert("RGB") as img:
        return _to_float01(np.asarray(img))


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path).convert("L") as img:
        return _to_float01(np.asarray(img))


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path)


def _resize_rgb(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    img = img.resize((w, h), resample=Image.Resampling.BICUBIC)
    return _to_float01(np.asarray(img))


def _resize_gray(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    img = img.resize((w, h), resample=Image.Resampling.BILINEAR)
    return _to_float01(np.asarray(img))


def _ensure_rgb_size(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    return arr if arr.shape[:2] == size_hw else _resize_rgb(arr, size_hw)


def _ensure_gray_size(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    return arr if arr.shape[:2] == size_hw else _resize_gray(arr, size_hw)


def _rgb_to_luma(arr: np.ndarray) -> np.ndarray:
    return (
        0.299 * arr[..., 0]
        + 0.587 * arr[..., 1]
        + 0.114 * arr[..., 2]
    ).astype(np.float32)


def _safe_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(v)]
    return float(mean(vals)) if vals else float("nan")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _weighted_psnr(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    w = np.clip(weights.astype(np.float32), 0.0, 1.0)
    denom = float(np.sum(w) * a.shape[2])
    if denom <= 1e-8:
        return float("nan")
    mse = float(np.sum(((a - b) ** 2) * w[..., None]) / denom)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _weighted_mae(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    w = np.clip(weights.astype(np.float32), 0.0, 1.0)
    denom = float(np.sum(w) * a.shape[2])
    if denom <= 1e-8:
        return float("nan")
    return float(np.sum(np.abs(a - b) * w[..., None]) / denom)


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    try:
        import cv2
    except Exception:
        return float("nan")
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    denom = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / denom
    return float(np.mean(ssim_map))


def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    vals = [_ssim_channel(a[..., c], b[..., c]) for c in range(3)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _colorize_score01(score01: np.ndarray) -> np.ndarray:
    score01 = np.clip(score01.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 * score01 - 0.2, 0.0, 1.0)
    g = np.clip(1.4 - np.abs(2.0 * score01 - 1.0) * 1.4, 0.0, 1.0)
    b = np.clip(1.2 - 1.4 * score01, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _error_heatmap(error_map: np.ndarray) -> np.ndarray:
    q = float(np.quantile(error_map, 0.95))
    denom = max(q, 1e-6)
    return _colorize_score01(np.clip(error_map / denom, 0.0, 1.0))


def _overlay_mask(base: np.ndarray, mask01: np.ndarray, color_rgb01: tuple[float, float, float], alpha: float) -> np.ndarray:
    w = np.clip(mask01.astype(np.float32), 0.0, 1.0)[..., None] * alpha
    color = np.zeros_like(base) + np.array(color_rgb01, dtype=np.float32)
    return np.clip(base * (1.0 - w) + color * w, 0.0, 1.0)


def _bool_to_rgb(mask: np.ndarray, fg: tuple[float, float, float], bg: tuple[float, float, float]) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.float32)
    out[mask] = np.array(fg, dtype=np.float32)
    out[~mask] = np.array(bg, dtype=np.float32)
    return out


def _label_tile(arr: np.ndarray, title: str) -> Image.Image:
    rgb_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_u8, mode="RGB")
    band_h = 20
    canvas = Image.new("RGB", (img.width, img.height + band_h), color=(12, 12, 12))
    canvas.paste(img, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 4), title, fill=(230, 230, 230), font=font)
    return canvas


def _grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    if not tiles:
        raise ValueError("tiles must not be empty")
    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(0, 0, 0))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        canvas.paste(tile, (x, y))
    return canvas


def _load_manifest(output_root: Path) -> dict[str, Any]:
    path = output_root / "manifest.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _build_panel(
    stem: str,
    prior_by_stem: dict[str, Path],
    mask_by_stem: dict[str, Path],
    ref_by_stem: dict[str, Path],
    fused_by_stem: dict[str, Path],
    input_by_stem: dict[str, Path],
    hard_mask_threshold: float,
) -> np.ndarray:
    prior = _load_rgb(prior_by_stem[stem])
    mask = _load_gray(mask_by_stem[stem])
    ref = _load_rgb(ref_by_stem[stem])
    h, w = ref.shape[:2]
    prior = _ensure_rgb_size(prior, (h, w))
    mask = _ensure_gray_size(mask, (h, w))
    hard_mask = mask >= hard_mask_threshold
    fused = None
    if stem in fused_by_stem:
        fused = _ensure_rgb_size(_load_rgb(fused_by_stem[stem]), (h, w))
    bicubic = None
    if stem in input_by_stem:
        bicubic = _resize_rgb(_load_rgb(input_by_stem[stem]), (h, w))

    prior_err = np.mean(np.abs(prior - ref), axis=2)
    oracle_rgb = np.zeros((h, w, 3), dtype=np.float32)
    if bicubic is not None:
        bicubic_err = np.mean(np.abs(bicubic - ref), axis=2)
        oracle_good = prior_err <= bicubic_err
        oracle_rgb = _bool_to_rgb(oracle_good, fg=(0.1, 0.95, 0.2), bg=(0.95, 0.2, 0.15))

    tiles = []
    if bicubic is not None:
        tiles.append(_label_tile(bicubic, "bicubic"))
    tiles.append(_label_tile(prior, "prior"))
    if fused is not None:
        tiles.append(_label_tile(fused, "fused"))
    tiles.append(_label_tile(ref, "reference"))
    tiles.append(_label_tile(np.repeat(mask[..., None], 3, axis=2), "mask_soft"))
    tiles.append(_label_tile(np.repeat(hard_mask[..., None].astype(np.float32), 3, axis=2), "mask_hard"))
    tiles.append(_label_tile(_overlay_mask(prior, mask, (0.0, 1.0, 0.1), 0.55), "prior_mask_overlay"))
    if bicubic is not None:
        tiles.append(_label_tile(oracle_rgb, "oracle_prior_better"))
    tiles.append(_label_tile(_error_heatmap(prior_err), "prior_error"))
    if fused is not None:
        fused_err = np.mean(np.abs(fused - ref), axis=2)
        tiles.append(_label_tile(_error_heatmap(fused_err), "fused_error"))
    return np.asarray(_grid(tiles, cols=4), dtype=np.uint8).astype(np.float32) / 255.0


def main() -> None:
    args = _parse_args()
    args.output_root = args.output_root.resolve()
    analysis_dir = (args.analysis_dir or (args.output_root / "quality_eval")).resolve()
    manifest = _load_manifest(args.output_root)

    prior_dir = args.output_root / args.prior_subdir
    mask_dir = args.output_root / args.mask_subdir
    reference_dir = args.reference_dir.resolve() if args.reference_dir else (args.output_root / args.reference_subdir)
    fused_dir = args.output_root / args.fused_subdir
    fused_dir = fused_dir if fused_dir.is_dir() else None
    input_dir = args.input_dir.resolve() if args.input_dir else None

    if not prior_dir.is_dir():
        raise FileNotFoundError(f"Prior dir not found: {prior_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference dir not found: {reference_dir}")
    if input_dir is not None and not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    discrepancy_floor = (
        float(manifest.get("discrepancy_floor", 0.05))
        if args.discrepancy_floor < 0.0
        else float(args.discrepancy_floor)
    )

    prior_by_stem = _index_by_stem(prior_dir)
    mask_by_stem = _index_by_stem(mask_dir)
    ref_by_stem = _index_by_stem(reference_dir)
    common_stems = [stem for stem in sorted(prior_by_stem) if stem in mask_by_stem and stem in ref_by_stem]
    if input_dir is not None:
        input_by_stem = _index_by_stem(input_dir)
        common_stems = [stem for stem in common_stems if stem in input_by_stem]
    else:
        input_by_stem = {}
    fused_by_stem = _index_by_stem(fused_dir) if fused_dir is not None else {}
    if not common_stems:
        raise RuntimeError("No common stems across prior/mask/reference directories")

    review_root = analysis_dir / "review_panels"
    best_root = review_root / "best_selectivity"
    worst_root = review_root / "worst_selectivity"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for stem in _progress(common_stems, "evaluating prior + mask quality", total=len(common_stems)):
        prior = _load_rgb(prior_by_stem[stem])
        ref = _load_rgb(ref_by_stem[stem])
        h, w = ref.shape[:2]
        prior = _ensure_rgb_size(prior, (h, w))
        mask_soft = _ensure_gray_size(_load_gray(mask_by_stem[stem]), (h, w))
        mask_soft = np.clip(mask_soft, 0.0, 1.0)
        mask_hard = mask_soft >= float(args.hard_mask_threshold)
        fused = None
        if fused_dir is not None and stem in _index_by_stem(fused_dir):
            fused = _ensure_rgb_size(_load_rgb(_index_by_stem(fused_dir)[stem]), (h, w))

        prior_psnr = _psnr(prior, ref)
        prior_ssim = _ssim_rgb(prior, ref)
        prior_mae = _mae(prior, ref)

        prior_keep_psnr = _weighted_psnr(prior, ref, mask_soft)
        prior_reject_psnr = _weighted_psnr(prior, ref, 1.0 - mask_soft)
        prior_keep_mae = _weighted_mae(prior, ref, mask_soft)
        prior_reject_mae = _weighted_mae(prior, ref, 1.0 - mask_soft)

        ref_l = _rgb_to_luma(ref)
        prior_l = _rgb_to_luma(prior)
        discrepancy = np.abs(prior_l - ref_l) / np.maximum(np.abs(ref_l), discrepancy_floor)
        discrepancy_mean = float(np.mean(discrepancy))
        discrepancy_p90 = float(np.percentile(discrepancy, 90.0))

        fused_psnr = float("nan")
        fused_ssim = float("nan")
        fused_mae = float("nan")
        fused_gain_over_prior = float("nan")
        if fused is not None:
            fused_psnr = _psnr(fused, ref)
            fused_ssim = _ssim_rgb(fused, ref)
            fused_mae = _mae(fused, ref)
            fused_gain_over_prior = fused_psnr - prior_psnr

        keep_ratio_soft = float(mask_soft.mean())
        keep_ratio_hard = float(mask_hard.mean())

        bicubic_psnr = float("nan")
        bicubic_keep_psnr = float("nan")
        bicubic_reject_psnr = float("nan")
        bicubic_mae = float("nan")
        oracle_good_ratio = float("nan")
        mask_precision = float("nan")
        mask_recall = float("nan")
        mask_f1 = float("nan")
        mask_iou = float("nan")
        mask_soft_precision = float("nan")
        mask_soft_recall = float("nan")
        selectivity_gain = float("nan")
        prior_gain_over_bicubic = float("nan")

        if input_dir is not None:
            bicubic = _resize_rgb(_load_rgb(input_by_stem[stem]), (h, w))
            bicubic_psnr = _psnr(bicubic, ref)
            bicubic_mae = _mae(bicubic, ref)
            bicubic_keep_psnr = _weighted_psnr(bicubic, ref, mask_soft)
            bicubic_reject_psnr = _weighted_psnr(bicubic, ref, 1.0 - mask_soft)
            prior_gain_over_bicubic = prior_psnr - bicubic_psnr

            prior_err = np.mean(np.abs(prior - ref), axis=2)
            bicubic_err = np.mean(np.abs(bicubic - ref), axis=2)
            oracle_good = prior_err <= (bicubic_err - float(args.oracle_margin))
            oracle_good_ratio = float(np.mean(oracle_good))

            inter = float(np.logical_and(mask_hard, oracle_good).sum())
            pred = float(mask_hard.sum())
            gt = float(oracle_good.sum())
            union = float(np.logical_or(mask_hard, oracle_good).sum())
            mask_precision = inter / pred if pred > 0 else float("nan")
            mask_recall = inter / gt if gt > 0 else float("nan")
            mask_f1 = (
                (2.0 * mask_precision * mask_recall) / (mask_precision + mask_recall)
                if not math.isnan(mask_precision) and not math.isnan(mask_recall) and (mask_precision + mask_recall) > 0
                else float("nan")
            )
            mask_iou = inter / union if union > 0 else float("nan")
            soft_denom = float(mask_soft.sum())
            mask_soft_precision = (
                float((mask_soft * oracle_good.astype(np.float32)).sum()) / soft_denom
                if soft_denom > 1e-8
                else float("nan")
            )
            mask_soft_recall = (
                float((mask_soft * oracle_good.astype(np.float32)).sum()) / gt
                if gt > 1e-8
                else float("nan")
            )
            selectivity_gain = (prior_keep_psnr - bicubic_keep_psnr) - (prior_reject_psnr - bicubic_reject_psnr)

        rows.append(
            {
                "stem": stem,
                "prior_psnr": prior_psnr,
                "prior_ssim": prior_ssim,
                "prior_mae": prior_mae,
                "prior_keep_psnr": prior_keep_psnr,
                "prior_reject_psnr": prior_reject_psnr,
                "prior_keep_mae": prior_keep_mae,
                "prior_reject_mae": prior_reject_mae,
                "fused_psnr": fused_psnr,
                "fused_ssim": fused_ssim,
                "fused_mae": fused_mae,
                "fused_gain_over_prior": fused_gain_over_prior,
                "bicubic_psnr": bicubic_psnr,
                "bicubic_mae": bicubic_mae,
                "bicubic_keep_psnr": bicubic_keep_psnr,
                "bicubic_reject_psnr": bicubic_reject_psnr,
                "prior_gain_over_bicubic": prior_gain_over_bicubic,
                "keep_ratio_soft": keep_ratio_soft,
                "keep_ratio_hard": keep_ratio_hard,
                "oracle_good_ratio": oracle_good_ratio,
                "mask_precision": mask_precision,
                "mask_recall": mask_recall,
                "mask_f1": mask_f1,
                "mask_iou": mask_iou,
                "mask_soft_precision": mask_soft_precision,
                "mask_soft_recall": mask_soft_recall,
                "selectivity_gain": selectivity_gain,
                "discrepancy_mean": discrepancy_mean,
                "discrepancy_p90": discrepancy_p90,
            }
        )

    summary = {
        "count": len(rows),
        "backend": manifest.get("backend"),
        "mask_threshold_generation": manifest.get("mask_threshold"),
        "discrepancy_floor": discrepancy_floor,
        "prior_psnr_mean": _safe_mean([r["prior_psnr"] for r in rows]),
        "prior_ssim_mean": _safe_mean([r["prior_ssim"] for r in rows]),
        "prior_gain_over_bicubic_mean": _safe_mean([r["prior_gain_over_bicubic"] for r in rows]),
        "fused_psnr_mean": _safe_mean([r["fused_psnr"] for r in rows]),
        "fused_gain_over_prior_mean": _safe_mean([r["fused_gain_over_prior"] for r in rows]),
        "keep_ratio_soft_mean": _safe_mean([r["keep_ratio_soft"] for r in rows]),
        "keep_ratio_hard_mean": _safe_mean([r["keep_ratio_hard"] for r in rows]),
        "oracle_good_ratio_mean": _safe_mean([r["oracle_good_ratio"] for r in rows]),
        "mask_precision_mean": _safe_mean([r["mask_precision"] for r in rows]),
        "mask_recall_mean": _safe_mean([r["mask_recall"] for r in rows]),
        "mask_f1_mean": _safe_mean([r["mask_f1"] for r in rows]),
        "mask_iou_mean": _safe_mean([r["mask_iou"] for r in rows]),
        "mask_soft_precision_mean": _safe_mean([r["mask_soft_precision"] for r in rows]),
        "mask_soft_recall_mean": _safe_mean([r["mask_soft_recall"] for r in rows]),
        "selectivity_gain_mean": _safe_mean([r["selectivity_gain"] for r in rows]),
        "discrepancy_mean": _safe_mean([r["discrepancy_mean"] for r in rows]),
    }

    rows_by_selectivity = sorted(
        rows,
        key=lambda r: (-1e9 if math.isnan(r["selectivity_gain"]) else r["selectivity_gain"]),
        reverse=True,
    )
    rows_by_prior = sorted(rows, key=lambda r: r["prior_psnr"], reverse=True)
    summary["best_selectivity_frames"] = rows_by_selectivity[: args.top_k]
    summary["worst_selectivity_frames"] = list(reversed(rows_by_selectivity[-args.top_k :]))
    summary["best_prior_frames"] = rows_by_prior[: args.top_k]
    summary["worst_prior_frames"] = list(reversed(rows_by_prior[-args.top_k :]))

    _write_csv(analysis_dir / "per_frame_metrics.csv", rows)
    with (analysis_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        f"count: {summary['count']}",
        f"backend: {summary['backend']}",
        f"prior_psnr_mean: {summary['prior_psnr_mean']:.4f}",
        f"prior_ssim_mean: {summary['prior_ssim_mean']:.4f}",
        f"prior_gain_over_bicubic_mean: {summary['prior_gain_over_bicubic_mean']:.4f}",
        f"fused_psnr_mean: {summary['fused_psnr_mean']:.4f}",
        f"fused_gain_over_prior_mean: {summary['fused_gain_over_prior_mean']:.4f}",
        f"keep_ratio_soft_mean: {summary['keep_ratio_soft_mean']:.4f}",
        f"keep_ratio_hard_mean: {summary['keep_ratio_hard_mean']:.4f}",
        f"mask_precision_mean: {summary['mask_precision_mean']:.4f}",
        f"mask_recall_mean: {summary['mask_recall_mean']:.4f}",
        f"mask_f1_mean: {summary['mask_f1_mean']:.4f}",
        f"mask_iou_mean: {summary['mask_iou_mean']:.4f}",
        f"mask_soft_precision_mean: {summary['mask_soft_precision_mean']:.4f}",
        f"mask_soft_recall_mean: {summary['mask_soft_recall_mean']:.4f}",
        f"selectivity_gain_mean: {summary['selectivity_gain_mean']:.4f}",
        f"discrepancy_mean: {summary['discrepancy_mean']:.4f}",
    ]
    with (analysis_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    best_root.mkdir(parents=True, exist_ok=True)
    worst_root.mkdir(parents=True, exist_ok=True)
    for row in summary["best_selectivity_frames"]:
        panel = _build_panel(
            stem=row["stem"],
            prior_by_stem=prior_by_stem,
            mask_by_stem=mask_by_stem,
            ref_by_stem=ref_by_stem,
            fused_by_stem=fused_by_stem,
            input_by_stem=input_by_stem,
            hard_mask_threshold=float(args.hard_mask_threshold),
        )
        _save_rgb(best_root / f"{row['stem']}.png", panel)
    for row in summary["worst_selectivity_frames"]:
        panel = _build_panel(
            stem=row["stem"],
            prior_by_stem=prior_by_stem,
            mask_by_stem=mask_by_stem,
            ref_by_stem=ref_by_stem,
            fused_by_stem=fused_by_stem,
            input_by_stem=input_by_stem,
            hard_mask_threshold=float(args.hard_mask_threshold),
        )
        _save_rgb(worst_root / f"{row['stem']}.png", panel)

    print("[prior-mask-eval] done")
    print(f"  output_root            : {args.output_root}")
    print(f"  analysis_dir           : {analysis_dir}")
    print(f"  frames                 : {summary['count']}")
    print(f"  prior_psnr_mean        : {summary['prior_psnr_mean']:.4f}")
    print(f"  prior_ssim_mean        : {summary['prior_ssim_mean']:.4f}")
    print(f"  prior_gain_over_bicubic: {summary['prior_gain_over_bicubic_mean']:.4f}")
    print(f"  fused_gain_over_prior  : {summary['fused_gain_over_prior_mean']:.4f}")
    print(f"  mask_precision_mean    : {summary['mask_precision_mean']:.4f}")
    print(f"  mask_recall_mean       : {summary['mask_recall_mean']:.4f}")
    print(f"  mask_f1_mean           : {summary['mask_f1_mean']:.4f}")
    print(f"  selectivity_gain_mean  : {summary['selectivity_gain_mean']:.4f}")


if __name__ == "__main__":
    main()
