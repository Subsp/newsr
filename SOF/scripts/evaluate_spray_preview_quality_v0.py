#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify sprayed-2DGS preview variants against base and GT. "
            "Metrics emphasize effective high-frequency hit, leak, low-frequency pollution, and luminance artifacts."
        )
    )
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--variant", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--match_policy", choices=["stem", "order"], default="order")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--lowpass_kernel", type=int, default=21)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--target_threshold", type=float, default=0.18)
    parser.add_argument("--change_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _box_blur_rgb(image: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)


def _normalize_map(value: np.ndarray, percentile: float) -> np.ndarray:
    value = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(value, float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def _luma(image: np.ndarray) -> np.ndarray:
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _corr(a: np.ndarray, b: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if weight is None:
        w = np.ones_like(x, dtype=np.float32)
    else:
        w = np.asarray(weight, dtype=np.float32).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 1e-7)
    if int(np.count_nonzero(keep)) < 16:
        return 0.0
    x = x[keep]
    y = y[keep]
    w = w[keep]
    w = w / max(float(w.sum()), 1e-8)
    xm = float(np.sum(w * x))
    ym = float(np.sum(w * y))
    xv = x - xm
    yv = y - ym
    denom = math.sqrt(float(np.sum(w * xv * xv)) * float(np.sum(w * yv * yv)))
    if denom <= 1e-8:
        return 0.0
    return float(np.sum(w * xv * yv) / denom)


def _mean(rows: Sequence[Dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _pairs(base_paths: List[Path], gt_paths: List[Path], current_paths: List[Path], policy: str):
    if policy == "order":
        for idx, (base, gt, current) in enumerate(zip(base_paths, gt_paths, current_paths)):
            yield f"{idx:05d}", base, gt, current
        return
    gt_by_stem = {p.stem: p for p in gt_paths}
    current_by_stem = {p.stem: p for p in current_paths}
    for base in base_paths:
        gt = gt_by_stem.get(base.stem)
        current = current_by_stem.get(base.stem)
        if gt is not None and current is not None:
            yield base.stem, base, gt, current


def _quality_row(
    *,
    stem: str,
    base_path: Path,
    gt_path: Path,
    current_path: Path,
    highpass_kernel: int,
    lowpass_kernel: int,
    norm_percentile: float,
    target_threshold: float,
    change_threshold: float,
) -> Dict[str, float | str]:
    base = _load_rgb(base_path)
    size = (base.shape[1], base.shape[0])
    gt = _load_rgb(gt_path, size=size)
    current = _load_rgb(current_path, size=size)

    hp_base = base - _box_blur_rgb(base, highpass_kernel)
    hp_gt = gt - _box_blur_rgb(gt, highpass_kernel)
    hp_current = current - _box_blur_rgb(current, highpass_kernel)
    target_hf = _normalize_map((0.65 * np.abs(hp_gt - hp_base) + 0.35 * np.abs(hp_gt)).mean(axis=2), norm_percentile)
    injected_hf = _normalize_map(np.abs(hp_current - hp_base).mean(axis=2), norm_percentile)
    target_mask = target_hf >= float(target_threshold)
    off_mask = ~target_mask

    low_base = _box_blur_rgb(base, lowpass_kernel)
    low_current = _box_blur_rgb(current, lowpass_kernel)
    low_gt = _box_blur_rgb(gt, lowpass_kernel)
    lf_change = np.abs(low_current - low_base).mean(axis=2)
    lf_gt_improve = np.abs(low_base - low_gt).mean(axis=2) - np.abs(low_current - low_gt).mean(axis=2)

    luma_base = _luma(base)
    luma_current = _luma(current)
    luma_gt = _luma(gt)
    dark = np.maximum(luma_base - luma_current, 0.0)
    bright = np.maximum(luma_current - luma_base, 0.0)
    changed = np.max(np.abs(current - base), axis=2) > float(change_threshold)

    on_energy = float(injected_hf[target_mask].mean()) if np.any(target_mask) else 0.0
    off_energy = float(injected_hf[off_mask].mean()) if np.any(off_mask) else 0.0
    target_energy = float(target_hf[target_mask].mean()) if np.any(target_mask) else 0.0
    leak_ratio = float(off_energy / max(on_energy, 1e-8))
    hf_corr = _corr(injected_hf, target_hf)
    hf_corr_active = _corr(injected_hf, target_hf, target_mask.astype(np.float32))
    hit_precision = float(np.mean(target_hf[changed] >= float(target_threshold))) if np.any(changed) else 0.0
    hit_recall = float(np.mean(injected_hf[target_mask] >= float(target_threshold))) if np.any(target_mask) else 0.0
    dark_changed = float(dark[changed].mean()) if np.any(changed) else 0.0
    bright_changed = float(bright[changed].mean()) if np.any(changed) else 0.0
    luma_error_delta = float(np.mean(np.abs(luma_current - luma_gt) - np.abs(luma_base - luma_gt)))

    return {
        "stem": stem,
        "base": str(base_path),
        "gt": str(gt_path),
        "current": str(current_path),
        "psnr_vs_gt": _psnr(current, gt),
        "psnr_base_vs_gt": _psnr(base, gt),
        "l1_vs_gt": float(np.mean(np.abs(current - gt))),
        "l1_base_vs_gt": float(np.mean(np.abs(base - gt))),
        "luma_error_delta": luma_error_delta,
        "target_active_ratio": float(target_mask.mean()),
        "changed_ratio": float(changed.mean()),
        "hf_on_energy": on_energy,
        "hf_off_energy": off_energy,
        "hf_target_energy": target_energy,
        "hf_leak_ratio": leak_ratio,
        "hf_corr": hf_corr,
        "hf_corr_active": hf_corr_active,
        "hit_precision": hit_precision,
        "hit_recall": hit_recall,
        "lf_change_mean": float(lf_change.mean()),
        "lf_gt_improve_mean": float(lf_gt_improve.mean()),
        "dark_mean": float(dark.mean()),
        "bright_mean": float(bright.mean()),
        "dark_changed_mean": dark_changed,
        "bright_changed_mean": bright_changed,
        "dark_to_bright": float(dark.sum() / max(float(bright.sum()), 1e-8)),
    }


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_paths = _list_images(base_dir)
    gt_paths = _list_images(gt_dir)
    if int(args.limit) > 0:
        base_paths = base_paths[: int(args.limit)]
        gt_paths = gt_paths[: int(args.limit)]

    summaries: List[Dict[str, object]] = []
    for name, variant_dir_raw in args.variant:
        variant_dir = Path(variant_dir_raw).expanduser().resolve()
        current_paths = _list_images(variant_dir)
        if int(args.limit) > 0:
            current_paths = current_paths[: int(args.limit)]
        rows: List[Dict[str, float | str]] = []
        for stem, base_path, gt_path, current_path in _pairs(
            base_paths,
            gt_paths,
            current_paths,
            str(args.match_policy),
        ):
            rows.append(
                _quality_row(
                    stem=stem,
                    base_path=base_path,
                    gt_path=gt_path,
                    current_path=current_path,
                    highpass_kernel=int(args.highpass_kernel),
                    lowpass_kernel=int(args.lowpass_kernel),
                    norm_percentile=float(args.norm_percentile),
                    target_threshold=float(args.target_threshold),
                    change_threshold=float(args.change_threshold),
                )
            )
        if not rows:
            raise RuntimeError(f"No common frames found for variant={name}: {variant_dir}")
        summary = {
            "name": str(name),
            "dir": str(variant_dir),
            "num_frames": len(rows),
            "psnr_vs_gt_mean": _mean(rows, "psnr_vs_gt"),
            "psnr_base_vs_gt_mean": _mean(rows, "psnr_base_vs_gt"),
            "psnr_delta_mean": _mean(rows, "psnr_vs_gt") - _mean(rows, "psnr_base_vs_gt"),
            "l1_vs_gt_mean": _mean(rows, "l1_vs_gt"),
            "l1_base_vs_gt_mean": _mean(rows, "l1_base_vs_gt"),
            "l1_delta_mean": _mean(rows, "l1_vs_gt") - _mean(rows, "l1_base_vs_gt"),
            "luma_error_delta_mean": _mean(rows, "luma_error_delta"),
            "target_active_ratio_mean": _mean(rows, "target_active_ratio"),
            "changed_ratio_mean": _mean(rows, "changed_ratio"),
            "hf_on_energy_mean": _mean(rows, "hf_on_energy"),
            "hf_off_energy_mean": _mean(rows, "hf_off_energy"),
            "hf_leak_ratio_mean": _mean(rows, "hf_leak_ratio"),
            "hf_corr_mean": _mean(rows, "hf_corr"),
            "hf_corr_active_mean": _mean(rows, "hf_corr_active"),
            "hit_precision_mean": _mean(rows, "hit_precision"),
            "hit_recall_mean": _mean(rows, "hit_recall"),
            "lf_change_mean": _mean(rows, "lf_change_mean"),
            "lf_gt_improve_mean": _mean(rows, "lf_gt_improve_mean"),
            "dark_mean": _mean(rows, "dark_mean"),
            "bright_mean": _mean(rows, "bright_mean"),
            "dark_changed_mean": _mean(rows, "dark_changed_mean"),
            "bright_changed_mean": _mean(rows, "bright_changed_mean"),
            "dark_to_bright_mean": _mean(rows, "dark_to_bright"),
            "rows": rows,
        }
        (output_dir / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)

    compact = [
        {key: value for key, value in summary.items() if key not in {"rows"}}
        for summary in summaries
    ]
    out = {
        "version": "evaluate_spray_preview_quality_v0",
        "base_dir": str(base_dir),
        "gt_dir": str(gt_dir),
        "output_dir": str(output_dir),
        "match_policy": str(args.match_policy),
        "highpass_kernel": int(args.highpass_kernel),
        "lowpass_kernel": int(args.lowpass_kernel),
        "target_threshold": float(args.target_threshold),
        "summaries": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
