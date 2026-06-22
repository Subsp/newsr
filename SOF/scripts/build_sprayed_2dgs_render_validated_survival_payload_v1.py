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
    metadata_path = (
        Path(args.metadata_path).expanduser().resolve()
        if args.metadata_path
        else Path(str(summary["newborn_metadata"])).expanduser().resolve()
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
    target_thr = max(_otsu_threshold(target), 0.02)
    active_ratio = float(np.mean(target >= target_thr))
    return {
        "target": target,
        "delta": delta,
        "lf": lf,
        "dark": dark,
        "target_threshold": float(target_thr),
        "target_active_ratio": active_ratio,
    }


def _load_source_mu(primitive_paths: Sequence[Path], source_view: np.ndarray, source_primitive: np.ndarray) -> np.ndarray:
    out = np.zeros((source_view.shape[0], 2), dtype=np.float32)
    for view_id in np.unique(source_view):
        view_int = int(view_id)
        if view_int < 0 or view_int >= len(primitive_paths):
            raise IndexError(f"source_view={view_int} outside primitive path list ({len(primitive_paths)})")
        ids = np.flatnonzero(source_view == view_int)
        primitive = np.load(primitive_paths[view_int])
        mu = np.asarray(primitive["mu_xy"], dtype=np.float32)
        out[ids] = mu[source_primitive[ids].astype(np.int64)]
    return out


def _dynamic_threshold(scores: np.ndarray, active_ratio: float, multiplier: float, min_fraction: float, max_fraction: float) -> float:
    if scores.size == 0:
        return 1.0
    keep_fraction = float(np.clip(active_ratio * float(multiplier), float(min_fraction), float(max_fraction)))
    quantile_thr = float(np.quantile(scores, max(0.0, min(1.0, 1.0 - keep_fraction))))
    otsu_thr = _otsu_threshold(scores)
    return float(max(quantile_thr, otsu_thr))


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
    mu_xy = _load_source_mu(primitive_paths, source_view, source_primitive)

    target_sample = np.zeros((n,), dtype=np.float32)
    delta_sample = np.zeros((n,), dtype=np.float32)
    lf_sample = np.zeros((n,), dtype=np.float32)
    dark_sample = np.zeros((n,), dtype=np.float32)
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
            norm_percentile=float(args.norm_percentile),
        )
        target_sample[ids] = _sample_bilinear(maps["target"], mu_xy[ids])
        delta_sample[ids] = _sample_bilinear(maps["delta"], mu_xy[ids])
        lf_sample[ids] = _sample_bilinear(maps["lf"], mu_xy[ids])
        dark_sample[ids] = _sample_bilinear(maps["dark"], mu_xy[ids])
        view_active_ratio[view_id] = float(maps["target_active_ratio"])
        view_target_threshold[view_id] = float(maps["target_threshold"])

    group_size, group_views = _group_support(matched_base_index, source_view)
    evidence = np.sqrt(np.clip(weight, 0.0, 1.0)) * (0.35 + 0.65 * np.clip(primitive_opacity, 0.0, 1.0))
    dist_score = np.exp(-0.5 * (matched_distance / max(float(args.distance_sigma_px), 1e-6)) ** 2).astype(np.float32)
    metadata_score = evidence * dist_score
    view_support = np.clip(group_views.astype(np.float32) / 2.0, 0.0, 1.0)
    size_support = _normalize_group_size(group_size, float(args.group_size_ref))
    agreement = np.sqrt(np.clip(target_sample, 0.0, 1.0) * np.clip(delta_sample, 0.0, 1.0))
    over_hf = delta_sample * np.clip(1.0 - target_sample, 0.0, 1.0)
    render_risk = (
        float(args.lf_penalty_weight) * lf_sample
        + float(args.dark_penalty_weight) * dark_sample
        + float(args.over_penalty_weight) * over_hf
    )
    score = (
        0.24 * metadata_score
        + 0.38 * target_sample
        + 0.20 * agreement
        + 0.10 * view_support
        + 0.08 * size_support
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
        bad = (matched_distance[ids] > float(args.bad_distance_px)) | (
            (lf_sample[ids] + dark_sample[ids]) > (target_sample[ids] + 0.45)
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
        delta_sample=delta_sample,
        lf_sample=lf_sample,
        dark_sample=dark_sample,
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
        "version": "build_sprayed_2dgs_render_validated_survival_payload_v1",
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
            "delta_mean": float(delta_sample.mean()),
            "lf_mean": float(lf_sample.mean()),
            "dark_mean": float(dark_sample.mean()),
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
