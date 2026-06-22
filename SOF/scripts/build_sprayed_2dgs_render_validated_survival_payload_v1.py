#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render-validated survival payload for sprayed 2DGS HF Gaussians. "
            "This test version uses GT high-frequency location as the target evidence."
        )
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--base_render_dir", required=True)
    parser.add_argument("--merged_render_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--primitive_dir", default="")
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--summary_path", default="")
    parser.add_argument("--tags_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit_views", type=int, default=0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--lowpass_kernel", type=int, default=21)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--target_coverage_multiplier", type=float, default=1.0)
    parser.add_argument("--probation_coverage_multiplier", type=float, default=1.8)
    parser.add_argument("--min_keep_fraction", type=float, default=0.01)
    parser.add_argument("--max_keep_fraction", type=float, default=0.45)
    parser.add_argument("--min_score_floor", type=float, default=0.05)
    parser.add_argument("--group_size_ref", type=float, default=16.0)
    parser.add_argument("--distance_sigma_px", type=float, default=1.75)
    parser.add_argument("--bad_distance_px", type=float, default=4.0)
    parser.add_argument("--lf_penalty_weight", type=float, default=0.32)
    parser.add_argument("--dark_penalty_weight", type=float, default=0.30)
    parser.add_argument("--over_penalty_weight", type=float, default=0.22)
    parser.add_argument("--orientation_kernel", type=int, default=9)
    parser.add_argument("--footprint_long_scale", type=float, default=0.85)
    parser.add_argument("--footprint_short_scale", type=float, default=0.85)
    parser.add_argument("--footprint_max_radius_px", type=float, default=12.0)
    parser.add_argument("--min_direction_align", type=float, default=0.52)
    parser.add_argument("--direction_penalty_weight", type=float, default=0.28)
    parser.add_argument("--footprint_leak_penalty_weight", type=float, default=0.34)
    parser.add_argument("--owner_direction_bins", type=int, default=12)
    parser.add_argument("--owner_top_bins", type=int, default=2)
    parser.add_argument("--owner_min_group_size", type=int, default=4)
    parser.add_argument("--owner_penalty_weight", type=float, default=0.18)
    return parser.parse_args()


def _list_files(root: Path, exts: set[str]) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _box_blur_scalar(image: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)


def _central_grad(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(image, dtype=np.float32)
    gx = np.zeros_like(value, dtype=np.float32)
    gy = np.zeros_like(value, dtype=np.float32)
    gx[:, 1:-1] = (value[:, 2:] - value[:, :-2]) * 0.5
    gx[:, 0] = value[:, 1] - value[:, 0]
    gx[:, -1] = value[:, -1] - value[:, -2]
    gy[1:-1, :] = (value[2:, :] - value[:-2, :]) * 0.5
    gy[0, :] = value[1, :] - value[0, :]
    gy[-1, :] = value[-1, :] - value[-2, :]
    return gx, gy


def _structure_tensor_orientation(energy: np.ndarray, kernel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx, gy = _central_grad(energy)
    jxx = _box_blur_scalar(gx * gx, kernel)
    jyy = _box_blur_scalar(gy * gy, kernel)
    jxy = _box_blur_scalar(gx * gy, kernel)
    grad_theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    tangent = grad_theta + np.float32(np.pi * 0.5)
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / np.maximum(trace, 1e-8)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    return (
        tangent.astype(np.float32),
        coherence,
        np.cos(2.0 * tangent).astype(np.float32),
        np.sin(2.0 * tangent).astype(np.float32),
    )


def _normalize_map(value: np.ndarray, percentile: float) -> np.ndarray:
    value = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(value, float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def _otsu_threshold(value: np.ndarray, bins: int = 128) -> float:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size < 16:
        return 1.0
    lo = float(flat.min())
    hi = float(flat.max())
    if hi <= lo + 1e-8:
        return hi
    hist, edges = np.histogram(flat, bins=int(bins), range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return hi
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1e-8)
    mean_fg = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1e-8))[::-1]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    idx = int(np.nanargmax(between))
    return float(centers[idx])


def _sample_bilinear(gray: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    x = np.clip(xy[:, 0].astype(np.float32), 0.0, max(w - 1, 0))
    y = np.clip(xy[:, 1].astype(np.float32), 0.0, max(h - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    v00 = gray[y0, x0]
    v10 = gray[y0, x1]
    v01 = gray[y1, x0]
    v11 = gray[y1, x1]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    ).astype(np.float32)


def _torch_flatnonzero(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    model_dir = Path(args.model_dir).expanduser().resolve()
    point_dir = model_dir / "point_cloud" / f"iteration_{int(args.iteration)}"
    tags_path = Path(args.tags_path).expanduser().resolve() if args.tags_path else point_dir / "gaussian_tags.pt"
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else model_dir / "spray_2dgs_hf_carrier_to_gaussian_layer_v0_summary.json"
    )
    summary = _read_json(summary_path)
    if args.metadata_path:
        metadata_path = Path(args.metadata_path).expanduser().resolve()
    else:
        metadata_name = "sprayed_2dgs_gaussian_layer_metadata_v0.npz"
        metadata_candidates: List[Path] = []
        for key in ("merged_metadata", "newborn_metadata"):
            value = summary.get(key)
            if value:
                metadata_candidates.append(Path(str(value)).expanduser().resolve())
        metadata_candidates.append(point_dir / metadata_name)
        newborn_model_dir = summary.get("newborn_model_dir")
        if newborn_model_dir:
            metadata_candidates.append(
                Path(str(newborn_model_dir)).expanduser().resolve()
                / "point_cloud"
                / f"iteration_{int(args.iteration)}"
                / metadata_name
            )
        seen = set()
        deduped_candidates: List[Path] = []
        for candidate in metadata_candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                deduped_candidates.append(candidate)
        metadata_path = next((candidate for candidate in deduped_candidates if candidate.is_file()), deduped_candidates[0])
        if not metadata_path.is_file():
            candidates = "\n  ".join(str(candidate) for candidate in deduped_candidates)
            raise FileNotFoundError(
                "sprayed 2DGS metadata not found. Re-run the spray step with the current code, "
                "or pass --metadata_path explicitly. Tried:\n  "
                f"{candidates}"
            )
    primitive_dir = (
        Path(args.primitive_dir).expanduser().resolve()
        if args.primitive_dir
        else Path(str(summary["primitive_dir"])).expanduser().resolve()
    )
    return tags_path, summary_path, metadata_path, primitive_dir, Path(args.output_dir).expanduser().resolve()


def _group_support(matched_base_index: np.ndarray, source_view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, inverse, counts = np.unique(matched_base_index.astype(np.int64), return_inverse=True, return_counts=True)
    pairs = np.stack([inverse.astype(np.int64), source_view.astype(np.int64)], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    view_counts = np.bincount(unique_pairs[:, 0], minlength=counts.shape[0]).astype(np.int32)
    return counts[inverse].astype(np.int32), view_counts[inverse].astype(np.int32)


def _normalize_group_size(group_size: np.ndarray, group_size_ref: float) -> np.ndarray:
    ref = max(float(group_size_ref), 1.0)
    return np.clip(np.log1p(group_size.astype(np.float32)) / np.log1p(ref), 0.0, 1.0)


def _build_full_mask(prior_indices: torch.Tensor, newborn_mask: np.ndarray, total: int) -> torch.Tensor:
    out = torch.zeros((total,), dtype=torch.bool)
    out[prior_indices] = torch.from_numpy(newborn_mask.astype(np.bool_))
    return out


def _view_maps(
    *,
    base_path: Path,
    merged_path: Path,
    gt_path: Path,
    highpass_kernel: int,
    lowpass_kernel: int,
    orientation_kernel: int,
    norm_percentile: float,
) -> Dict[str, np.ndarray | float]:
    base = _load_rgb(base_path)
    size = (base.shape[1], base.shape[0])
    merged = _load_rgb(merged_path, size=size)
    gt = _load_rgb(gt_path, size=size)

    hp_base = base - _box_blur_rgb(base, highpass_kernel)
    hp_merged = merged - _box_blur_rgb(merged, highpass_kernel)
    hp_gt = gt - _box_blur_rgb(gt, highpass_kernel)
    target_raw = (0.65 * np.abs(hp_gt - hp_base) + 0.35 * np.abs(hp_gt)).mean(axis=2)
    delta_raw = np.abs(hp_merged - hp_base).mean(axis=2)
    lf_raw = np.abs(_box_blur_rgb(merged, lowpass_kernel) - _box_blur_rgb(base, lowpass_kernel)).mean(axis=2)
    luma_base = (0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]).astype(np.float32)
    luma_merged = (0.299 * merged[..., 0] + 0.587 * merged[..., 1] + 0.114 * merged[..., 2]).astype(np.float32)
    dark_raw = np.maximum(luma_base - luma_merged, 0.0)

    target = _normalize_map(target_raw, norm_percentile)
    delta = _normalize_map(delta_raw, norm_percentile)
    lf = _normalize_map(lf_raw, norm_percentile)
    dark = _normalize_map(dark_raw, norm_percentile)
    tangent, coherence, tangent_cos2, tangent_sin2 = _structure_tensor_orientation(target, orientation_kernel)
    target_thr = max(_otsu_threshold(target), 0.02)
    active_ratio = float(np.mean(target >= target_thr))
    return {
        "target": target,
        "delta": delta,
        "lf": lf,
        "dark": dark,
        "tangent": tangent,
        "coherence": coherence,
        "tangent_cos2": tangent_cos2,
        "tangent_sin2": tangent_sin2,
        "target_threshold": float(target_thr),
        "target_active_ratio": active_ratio,
    }


def _cholesky_axes(cholesky: Optional[np.ndarray], count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cholesky is None or cholesky.shape[0] != count or cholesky.shape[1] < 3:
        return (
            np.zeros((count,), dtype=np.float32),
            np.full((count,), 5.0, dtype=np.float32),
            np.full((count,), 0.8, dtype=np.float32),
        )
    a = np.asarray(cholesky[:, 0], dtype=np.float32)
    b = np.asarray(cholesky[:, 1], dtype=np.float32)
    c = np.asarray(cholesky[:, 2], dtype=np.float32)
    xx = a * a
    xy = a * b
    yy = b * b + c * c
    trace = xx + yy
    diff = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy * xy, 1e-12))
    large = np.maximum((trace + diff) * 0.5, 1e-8)
    small = np.maximum((trace - diff) * 0.5, 1e-8)
    theta = 0.5 * np.arctan2(2.0 * xy, xx - yy + 1e-12)
    return theta.astype(np.float32), np.sqrt(large).astype(np.float32), np.sqrt(small).astype(np.float32)


def _load_source_geometry(
    primitive_paths: Sequence[Path],
    source_view: np.ndarray,
    source_primitive: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_out = np.zeros((source_view.shape[0], 2), dtype=np.float32)
    theta_out = np.zeros((source_view.shape[0],), dtype=np.float32)
    long_out = np.ones((source_view.shape[0],), dtype=np.float32)
    short_out = np.ones((source_view.shape[0],), dtype=np.float32)
    for view_id in np.unique(source_view):
        view_int = int(view_id)
        if view_int < 0 or view_int >= len(primitive_paths):
            raise IndexError(f"source_view={view_int} outside primitive path list ({len(primitive_paths)})")
        ids = np.flatnonzero(source_view == view_int)
        primitive = np.load(primitive_paths[view_int])
        mu = np.asarray(primitive["mu_xy"], dtype=np.float32)
        primitive_ids = source_primitive[ids].astype(np.int64)
        mu_out[ids] = mu[primitive_ids]
        cholesky = np.asarray(primitive["cholesky"], dtype=np.float32) if "cholesky" in primitive else None
        if cholesky is None:
            theta, long_px, short_px = _cholesky_axes(None, int(mu.shape[0]))
        else:
            theta, long_px, short_px = _cholesky_axes(cholesky, int(cholesky.shape[0]))
        theta_out[ids] = theta[primitive_ids]
        long_out[ids] = long_px[primitive_ids]
        short_out[ids] = short_px[primitive_ids]
    return mu_out, theta_out, long_out, short_out


def _dynamic_threshold(scores: np.ndarray, active_ratio: float, multiplier: float, min_fraction: float, max_fraction: float) -> float:
    if scores.size == 0:
        return 1.0
    keep_fraction = float(np.clip(active_ratio * float(multiplier), float(min_fraction), float(max_fraction)))
    quantile_thr = float(np.quantile(scores, max(0.0, min(1.0, 1.0 - keep_fraction))))
    otsu_thr = _otsu_threshold(scores)
    return float(max(quantile_thr, otsu_thr))


def _sample_footprint_metrics(
    maps: Dict[str, np.ndarray | float],
    mu_xy: np.ndarray,
    theta: np.ndarray,
    long_px: np.ndarray,
    short_px: np.ndarray,
    *,
    long_scale: float,
    short_scale: float,
    max_radius_px: float,
) -> Dict[str, np.ndarray]:
    n = int(mu_xy.shape[0])
    if n == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "target": empty,
            "delta": empty,
            "lf": empty,
            "dark": empty,
            "leak": empty,
            "over": empty,
            "direction_align": empty,
            "direction_coherence": empty,
        }

    long_r = np.clip(long_px.astype(np.float32) * float(long_scale), 0.25, float(max_radius_px))
    short_r = np.clip(short_px.astype(np.float32) * float(short_scale), 0.20, float(max_radius_px))
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    long_vec = np.stack([cos_t * long_r, sin_t * long_r], axis=1)
    short_vec = np.stack([-sin_t * short_r, cos_t * short_r], axis=1)
    primitive_cos2 = np.cos(2.0 * theta).astype(np.float32)
    primitive_sin2 = np.sin(2.0 * theta).astype(np.float32)

    # Canonical samples cover center, long-axis support, and short-axis leakage.
    offsets = (
        (0.0, 0.0, 1.00),
        (0.65, 0.0, 0.75),
        (-0.65, 0.0, 0.75),
        (1.25, 0.0, 0.35),
        (-1.25, 0.0, 0.35),
        (0.0, 0.75, 0.45),
        (0.0, -0.75, 0.45),
    )
    weight_sum = float(sum(w for _, _, w in offsets))
    target_acc = np.zeros((n,), dtype=np.float32)
    delta_acc = np.zeros((n,), dtype=np.float32)
    lf_acc = np.zeros((n,), dtype=np.float32)
    dark_acc = np.zeros((n,), dtype=np.float32)
    leak_acc = np.zeros((n,), dtype=np.float32)
    over_acc = np.zeros((n,), dtype=np.float32)
    align_num = np.zeros((n,), dtype=np.float32)
    align_den = np.zeros((n,), dtype=np.float32)
    coherence_num = np.zeros((n,), dtype=np.float32)
    coherence_den = np.zeros((n,), dtype=np.float32)

    for u, v, sample_weight in offsets:
        xy = mu_xy + float(u) * long_vec + float(v) * short_vec
        target = _sample_bilinear(maps["target"], xy)
        delta = _sample_bilinear(maps["delta"], xy)
        lf = _sample_bilinear(maps["lf"], xy)
        dark = _sample_bilinear(maps["dark"], xy)
        coherence = _sample_bilinear(maps["coherence"], xy)
        cos2 = _sample_bilinear(maps["tangent_cos2"], xy)
        sin2 = _sample_bilinear(maps["tangent_sin2"], xy)
        align = np.abs(primitive_cos2 * cos2 + primitive_sin2 * sin2).astype(np.float32)
        orient_weight = float(sample_weight) * target * (0.25 + 0.75 * coherence)

        target_acc += float(sample_weight) * target
        delta_acc += float(sample_weight) * delta
        lf_acc += float(sample_weight) * lf
        dark_acc += float(sample_weight) * dark
        leak_acc += float(sample_weight) * (1.0 - target) * (0.35 + 0.65 * delta)
        over_acc += float(sample_weight) * delta * np.clip(1.0 - target, 0.0, 1.0)
        align_num += orient_weight * align
        align_den += orient_weight
        coherence_num += float(sample_weight) * target * coherence
        coherence_den += float(sample_weight) * target

    direction_align = np.where(align_den > 1e-7, align_num / np.maximum(align_den, 1e-7), 0.5).astype(np.float32)
    direction_coherence = np.where(
        coherence_den > 1e-7,
        coherence_num / np.maximum(coherence_den, 1e-7),
        0.0,
    ).astype(np.float32)
    return {
        "target": np.clip(target_acc / weight_sum, 0.0, 1.0).astype(np.float32),
        "delta": np.clip(delta_acc / weight_sum, 0.0, 1.0).astype(np.float32),
        "lf": np.clip(lf_acc / weight_sum, 0.0, 1.0).astype(np.float32),
        "dark": np.clip(dark_acc / weight_sum, 0.0, 1.0).astype(np.float32),
        "leak": np.clip(leak_acc / weight_sum, 0.0, 1.0).astype(np.float32),
        "over": np.clip(over_acc / weight_sum, 0.0, 1.0).astype(np.float32),
        "direction_align": np.clip(direction_align, 0.0, 1.0),
        "direction_coherence": np.clip(direction_coherence, 0.0, 1.0),
    }


def _owner_direction_support(
    matched_base_index: np.ndarray,
    source_theta: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int,
    top_bins: int,
    min_group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(matched_base_index.shape[0])
    support = np.ones((n,), dtype=np.float32)
    in_top_mode = np.ones((n,), dtype=bool)
    bin_count = max(int(bins), 2)
    keep_bins = max(1, min(int(top_bins), bin_count))
    order = np.argsort(matched_base_index.astype(np.int64), kind="stable")
    sorted_owner = matched_base_index[order].astype(np.int64)
    sorted_phi = np.mod(source_theta[order].astype(np.float32), np.pi)
    sorted_bin = np.clip(np.floor(sorted_phi / np.pi * float(bin_count)).astype(np.int64), 0, bin_count - 1)
    sorted_weight = np.clip(weights[order].astype(np.float32), 1e-4, None)
    start = 0
    while start < n:
        end = start + 1
        owner = sorted_owner[start]
        while end < n and sorted_owner[end] == owner:
            end += 1
        ids = order[start:end]
        if end - start >= int(min_group_size):
            local_bins = sorted_bin[start:end]
            local_weight = sorted_weight[start:end]
            hist = np.bincount(local_bins, weights=local_weight, minlength=bin_count).astype(np.float32)
            best = np.argsort(hist)[-keep_bins:]
            best_max = float(np.max(hist[best])) if best.size else float(hist.max())
            local_support = hist[local_bins] / max(best_max, 1e-6)
            support[ids] = np.clip(0.25 + 0.75 * local_support, 0.0, 1.0).astype(np.float32)
            in_top_mode[ids] = np.isin(local_bins, best)
        start = end
    return support, in_top_mode


def main() -> None:
    args = _parse_args()
    tags_path, summary_path, metadata_path, primitive_dir, output_dir = _resolve_paths(args)
    summary = _read_json(summary_path)
    tags = torch.load(tags_path, map_location="cpu")
    source_tag = torch.as_tensor(tags["source_tag"]).reshape(-1).to(dtype=torch.int64)
    total = int(source_tag.shape[0])
    prior_mask_t = source_tag == 1
    prior_indices = _torch_flatnonzero(prior_mask_t)
    base_count = int(summary.get("base_gaussians", total - int(prior_indices.shape[0])))

    meta = np.load(metadata_path)
    weight = np.asarray(meta["weight"], dtype=np.float32).reshape(-1)
    primitive_opacity = np.asarray(meta["primitive_opacity"], dtype=np.float32).reshape(-1)
    matched_distance = np.asarray(meta["matched_pixel_distance"], dtype=np.float32).reshape(-1)
    opacity = np.asarray(meta["opacity"], dtype=np.float32).reshape(-1)
    scale = np.asarray(meta["scale"], dtype=np.float32)
    source_view = np.asarray(meta["source_view"], dtype=np.int32).reshape(-1)
    source_primitive = np.asarray(meta["source_primitive"], dtype=np.int64).reshape(-1)
    matched_base_index = np.asarray(meta["matched_base_index"], dtype=np.int64).reshape(-1)
    n = int(weight.shape[0])
    if int(prior_indices.shape[0]) != n:
        if base_count + n != total:
            raise ValueError(f"prior count mismatch: prior={int(prior_indices.shape[0])} metadata={n} total={total}")
        prior_indices = torch.arange(base_count, total, dtype=torch.long)

    primitive_paths = _list_files(primitive_dir, {".npz"})
    if int(args.limit_views) > 0:
        primitive_paths = primitive_paths[: int(args.limit_views)]
    base_paths = _list_files(Path(args.base_render_dir).expanduser().resolve(), IMAGE_EXTS)
    merged_paths = _list_files(Path(args.merged_render_dir).expanduser().resolve(), IMAGE_EXTS)
    gt_paths = _list_files(Path(args.gt_dir).expanduser().resolve(), IMAGE_EXTS)
    max_view = int(source_view.max()) if source_view.size else -1
    required_views = max_view + 1
    if len(base_paths) < required_views or len(merged_paths) < required_views or len(gt_paths) < required_views:
        raise ValueError(
            f"not enough render/gt frames for source views: required={required_views} "
            f"base={len(base_paths)} merged={len(merged_paths)} gt={len(gt_paths)}"
        )
    mu_xy, source_theta, source_long_px, source_short_px = _load_source_geometry(
        primitive_paths,
        source_view,
        source_primitive,
    )

    target_sample = np.zeros((n,), dtype=np.float32)
    delta_sample = np.zeros((n,), dtype=np.float32)
    lf_sample = np.zeros((n,), dtype=np.float32)
    dark_sample = np.zeros((n,), dtype=np.float32)
    footprint_leak = np.zeros((n,), dtype=np.float32)
    footprint_over = np.zeros((n,), dtype=np.float32)
    direction_align = np.zeros((n,), dtype=np.float32)
    direction_coherence = np.zeros((n,), dtype=np.float32)
    center_target_sample = np.zeros((n,), dtype=np.float32)
    view_active_ratio: Dict[int, float] = {}
    view_target_threshold: Dict[int, float] = {}
    for view_id in sorted(int(v) for v in np.unique(source_view)):
        ids = np.flatnonzero(source_view == view_id)
        maps = _view_maps(
            base_path=base_paths[view_id],
            merged_path=merged_paths[view_id],
            gt_path=gt_paths[view_id],
            highpass_kernel=int(args.highpass_kernel),
            lowpass_kernel=int(args.lowpass_kernel),
            orientation_kernel=int(args.orientation_kernel),
            norm_percentile=float(args.norm_percentile),
        )
        center_target_sample[ids] = _sample_bilinear(maps["target"], mu_xy[ids])
        footprint = _sample_footprint_metrics(
            maps,
            mu_xy[ids],
            source_theta[ids],
            source_long_px[ids],
            source_short_px[ids],
            long_scale=float(args.footprint_long_scale),
            short_scale=float(args.footprint_short_scale),
            max_radius_px=float(args.footprint_max_radius_px),
        )
        target_sample[ids] = footprint["target"]
        delta_sample[ids] = footprint["delta"]
        lf_sample[ids] = footprint["lf"]
        dark_sample[ids] = footprint["dark"]
        footprint_leak[ids] = footprint["leak"]
        footprint_over[ids] = footprint["over"]
        direction_align[ids] = footprint["direction_align"]
        direction_coherence[ids] = footprint["direction_coherence"]
        view_active_ratio[view_id] = float(maps["target_active_ratio"])
        view_target_threshold[view_id] = float(maps["target_threshold"])

    group_size, group_views = _group_support(matched_base_index, source_view)
    evidence = np.sqrt(np.clip(weight, 0.0, 1.0)) * (0.35 + 0.65 * np.clip(primitive_opacity, 0.0, 1.0))
    dist_score = np.exp(-0.5 * (matched_distance / max(float(args.distance_sigma_px), 1e-6)) ** 2).astype(np.float32)
    metadata_score = evidence * dist_score
    view_support = np.clip(group_views.astype(np.float32) / 2.0, 0.0, 1.0)
    size_support = _normalize_group_size(group_size, float(args.group_size_ref))
    owner_support, owner_top_mode = _owner_direction_support(
        matched_base_index,
        source_theta,
        metadata_score + target_sample + 1e-4,
        bins=int(args.owner_direction_bins),
        top_bins=int(args.owner_top_bins),
        min_group_size=int(args.owner_min_group_size),
    )
    agreement = np.sqrt(np.clip(target_sample, 0.0, 1.0) * np.clip(delta_sample, 0.0, 1.0))
    over_hf = np.maximum(footprint_over, delta_sample * np.clip(1.0 - target_sample, 0.0, 1.0))
    direction_quality = direction_align * (0.35 + 0.65 * direction_coherence)
    direction_badness = np.clip(float(args.min_direction_align) - direction_align, 0.0, 1.0) * direction_coherence
    owner_badness = np.where(owner_top_mode, 0.0, 1.0 - owner_support).astype(np.float32)
    render_risk = (
        float(args.lf_penalty_weight) * lf_sample
        + float(args.dark_penalty_weight) * dark_sample
        + float(args.over_penalty_weight) * over_hf
        + float(args.footprint_leak_penalty_weight) * footprint_leak
        + float(args.direction_penalty_weight) * direction_badness
        + float(args.owner_penalty_weight) * owner_badness
    )
    score = (
        0.16 * metadata_score
        + 0.30 * target_sample
        + 0.18 * agreement
        + 0.18 * direction_quality
        + 0.08 * owner_support
        + 0.06 * view_support
        + 0.04 * size_support
        - render_risk
    ).astype(np.float32)
    score = np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    survive = np.zeros((n,), dtype=bool)
    probation = np.zeros((n,), dtype=bool)
    thresholds_by_view: Dict[str, Dict[str, float]] = {}
    for view_id in sorted(int(v) for v in np.unique(source_view)):
        ids = np.flatnonzero(source_view == view_id)
        view_scores = score[ids]
        active_ratio = view_active_ratio[view_id]
        survive_thr = max(
            float(args.min_score_floor),
            _dynamic_threshold(
                view_scores,
                active_ratio,
                float(args.target_coverage_multiplier),
                float(args.min_keep_fraction),
                float(args.max_keep_fraction),
            ),
        )
        probation_thr = max(
            float(args.min_score_floor) * 0.5,
            _dynamic_threshold(
                view_scores,
                active_ratio,
                float(args.probation_coverage_multiplier),
                float(args.min_keep_fraction),
                float(args.max_keep_fraction),
            ),
        )
        target_floor = max(0.05, 0.35 * view_target_threshold[view_id])
        bad = (
            (matched_distance[ids] > float(args.bad_distance_px))
            | ((lf_sample[ids] + dark_sample[ids]) > (target_sample[ids] + 0.45))
            | ((direction_coherence[ids] > 0.22) & (direction_align[ids] < float(args.min_direction_align)))
            | ((footprint_leak[ids] > target_sample[ids] + 0.35) & (target_sample[ids] < 0.45))
        )
        survive_ids = ids[(view_scores >= survive_thr) & (target_sample[ids] >= target_floor) & ~bad]
        probation_ids = ids[
            (view_scores >= probation_thr)
            & ~np.isin(ids, survive_ids)
            & (target_sample[ids] >= target_floor * 0.5)
            & ~bad
        ]
        survive[survive_ids] = True
        probation[probation_ids] = True
        thresholds_by_view[str(view_id)] = {
            "active_ratio": active_ratio,
            "target_threshold": view_target_threshold[view_id],
            "survive_threshold": float(survive_thr),
            "probation_threshold": float(probation_thr),
            "survive": int(survive_ids.shape[0]),
            "probation": int(probation_ids.shape[0]),
            "total": int(ids.shape[0]),
        }

    suppress = (~survive) & (~probation) & (score >= float(args.min_score_floor))
    prune = ~(survive | probation | suppress)
    keep = survive
    candidate = survive | probation
    drop_keep = ~keep
    drop_candidate = ~candidate

    state = np.zeros((n,), dtype=np.int8)
    state[survive] = 1
    state[probation] = 2
    state[suppress] = 3
    state[prune] = 4

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "sprayed_2dgs_render_validated_survival_payload_v1.pt"
    payload = {
        "prior": prior_mask_t,
        "survive_prior": _build_full_mask(prior_indices, survive, total),
        "probation_prior": _build_full_mask(prior_indices, probation, total),
        "suppress_prior": _build_full_mask(prior_indices, suppress, total),
        "prune_prior": _build_full_mask(prior_indices, prune, total),
        "keep_prior": _build_full_mask(prior_indices, keep, total),
        "candidate_prior": _build_full_mask(prior_indices, candidate, total),
        "drop_prior": _build_full_mask(prior_indices, drop_keep, total),
        "drop_candidate_prior": _build_full_mask(prior_indices, drop_candidate, total),
        "state": torch.zeros((total,), dtype=torch.int8),
        "survival_score": torch.zeros((total,), dtype=torch.float32),
        "base_count": torch.tensor(base_count, dtype=torch.int64),
    }
    payload["state"][prior_indices] = torch.from_numpy(state)
    payload["survival_score"][prior_indices] = torch.from_numpy(score)
    torch.save(payload, payload_path)

    scores_path = output_dir / "sprayed_2dgs_render_validated_survival_scores_v1.npz"
    np.savez_compressed(
        scores_path,
        score=score,
        state=state,
        target_sample=target_sample,
        center_target_sample=center_target_sample,
        delta_sample=delta_sample,
        lf_sample=lf_sample,
        dark_sample=dark_sample,
        footprint_leak=footprint_leak,
        footprint_over=footprint_over,
        direction_align=direction_align,
        direction_coherence=direction_coherence,
        direction_quality=direction_quality,
        owner_support=owner_support,
        owner_top_mode=owner_top_mode,
        metadata_score=metadata_score,
        agreement=agreement,
        over_hf=over_hf,
        render_risk=render_risk,
        group_size=group_size,
        group_views=group_views,
        source_view=source_view,
        source_primitive=source_primitive,
        matched_base_index=matched_base_index,
        matched_pixel_distance=matched_distance,
        mu_xy=mu_xy,
        source_theta=source_theta,
        source_long_px=source_long_px,
        source_short_px=source_short_px,
    )
    counts = {
        "total": n,
        "survive": int(np.count_nonzero(survive)),
        "probation": int(np.count_nonzero(probation)),
        "suppress": int(np.count_nonzero(suppress)),
        "prune": int(np.count_nonzero(prune)),
        "keep": int(np.count_nonzero(keep)),
        "candidate": int(np.count_nonzero(candidate)),
    }
    out_summary = {
        "version": "build_sprayed_2dgs_render_validated_survival_payload_v2_footprint_direction_owner",
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "iteration": int(args.iteration),
        "tags_path": str(tags_path),
        "spray_summary": str(summary_path),
        "metadata_path": str(metadata_path),
        "primitive_dir": str(primitive_dir),
        "base_render_dir": str(Path(args.base_render_dir).expanduser().resolve()),
        "merged_render_dir": str(Path(args.merged_render_dir).expanduser().resolve()),
        "gt_dir": str(Path(args.gt_dir).expanduser().resolve()),
        "payload_path": str(payload_path),
        "scores_path": str(scores_path),
        "counts": counts,
        "ratios": {key: float(value / max(n, 1)) for key, value in counts.items() if key != "total"},
        "score_stats": {
            "score_mean": float(score.mean()),
            "score_p50": float(np.percentile(score, 50)),
            "score_p90": float(np.percentile(score, 90)),
            "target_mean": float(target_sample.mean()),
            "center_target_mean": float(center_target_sample.mean()),
            "delta_mean": float(delta_sample.mean()),
            "lf_mean": float(lf_sample.mean()),
            "dark_mean": float(dark_sample.mean()),
            "footprint_leak_mean": float(footprint_leak.mean()),
            "direction_align_mean": float(direction_align.mean()),
            "direction_coherence_mean": float(direction_coherence.mean()),
            "direction_quality_mean": float(direction_quality.mean()),
            "owner_support_mean": float(owner_support.mean()),
            "owner_top_mode_ratio": float(owner_top_mode.mean()),
            "risk_mean": float(render_risk.mean()),
        },
        "thresholds_by_view": thresholds_by_view,
    }
    summary_path_out = output_dir / "summary.json"
    summary_path_out.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "score_stats": out_summary["score_stats"], "payload": str(payload_path)}, indent=2))
    print(f"[render-validated-survival-v1] summary: {summary_path_out}")


if __name__ == "__main__":
    main()
