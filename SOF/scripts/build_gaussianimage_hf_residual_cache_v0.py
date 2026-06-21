#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit trusted image-domain HF residuals with the official GaussianImage 2DGS implementation. "
            "This script only prepares HF targets, weighted loss, and diagnostics; 2D Gaussian raster/model code "
            "is imported from --external_repo_root."
        )
    )
    parser.add_argument("--external_repo_root", required=True, help="Clone of https://github.com/Xinjie-Q/GaussianImage.")
    parser.add_argument("--target_dir", required=True, help="Usually NPSE edge_target.")
    parser.add_argument("--anchor_dir", required=True, help="Anchor render directory.")
    parser.add_argument("--mask_dir", required=True, help="Trusted edge/HF mask directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--detail_alpha", type=float, default=0.8)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--confidence_power", type=float, default=1.5)
    parser.add_argument("--mask_power", type=float, default=1.0)
    parser.add_argument("--background_weight", type=float, default=0.02)
    parser.add_argument("--fit_target_mode", default="hf_residual", choices=["hf_residual", "rgb", "rgb_delta"])
    parser.add_argument("--rgb_loss_weight_mode", default="full", choices=["full", "trust", "trust_plus_background"])
    parser.add_argument("--num_gaussians", type=int, default=4096)
    parser.add_argument("--model", default="cholesky", choices=["cholesky", "rs"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adan"])
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="l1_l2", choices=["l1", "l2", "l1_l2"])
    parser.add_argument("--lambda_l1", type=float, default=0.5)
    parser.add_argument("--lambda_l2", type=float, default=0.5)
    parser.add_argument("--init_random", action="store_true")
    parser.add_argument("--init_min_score", type=float, default=0.035)
    parser.add_argument("--init_nms_radius_px", type=int, default=2)
    parser.add_argument("--init_max_candidates", type=int, default=0)
    parser.add_argument("--init_weight_power", type=float, default=0.5)
    parser.add_argument("--init_orientation_radius_px", type=int, default=5)
    parser.add_argument("--init_sigma_long_px", type=float, default=5.0)
    parser.add_argument("--init_sigma_short_px", type=float, default=0.8)
    parser.add_argument("--init_coherence_long_boost", type=float, default=0.75)
    parser.add_argument("--segment_init", action="store_true")
    parser.add_argument("--no_segment_init", dest="segment_init", action="store_false")
    parser.set_defaults(segment_init=True)
    parser.add_argument("--segment_samples_per_seed", type=int, default=7)
    parser.add_argument("--segment_step_px", type=float, default=2.0)
    parser.add_argument("--segment_seed_nms_radius_px", type=int, default=8)
    parser.add_argument("--segment_trace_search_radius_px", type=int, default=2)
    parser.add_argument("--segment_turn_min_cos", type=float, default=0.45)
    parser.add_argument("--segment_min_score", type=float, default=-1.0)
    parser.add_argument("--segment_anchor_weight", type=float, default=0.015)
    parser.add_argument("--segment_pair_weight", type=float, default=0.030)
    parser.add_argument("--segment_shape_weight", type=float, default=0.001)
    parser.add_argument("--segment_color_smooth_weight", type=float, default=0.002)
    parser.add_argument("--light_visuals", action="store_true", help="Also write white-background diagnostic images.")
    parser.add_argument("--light_visual_strength", type=float, default=0.75)
    parser.add_argument("--neutral_outside_mask", action="store_true")
    parser.add_argument("--no_neutral_outside_mask", dest="neutral_outside_mask", action="store_false")
    parser.set_defaults(neutral_outside_mask=False)
    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(
    paths: Sequence[Path],
    lookup: Dict[str, Path],
    reference_path: Path,
    index: int,
    match_policy: str,
) -> Optional[Path]:
    if match_policy in {"stem", "order_if_needed"}:
        found = lookup.get(reference_path.stem.lower())
        if found is not None:
            return found
        if match_policy == "stem":
            return None
    if match_policy in {"order", "order_if_needed"} and index < len(paths):
        return paths[index]
    return None


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("L")
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
    return (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    ).astype(np.float32) / float(k * k)


def _central_grad(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = gray.astype(np.float32, copy=False)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gy[0, :] = gray[1, :] - gray[0, :]
    gy[-1, :] = gray[-1, :] - gray[-2, :]
    return gx, gy


def _box_sum(gray: np.ndarray, radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r <= 0:
        return gray.astype(np.float32, copy=True)
    k = 2 * r + 1
    padded = np.pad(gray.astype(np.float32), ((r, r), (r, r)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]).astype(np.float32)


def _structure_tensor_tangent(energy: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    gx, gy = _central_grad(energy)
    jxx = _box_sum(gx * gx, radius)
    jyy = _box_sum(gy * gy, radius)
    jxy = _box_sum(gx * gy, radius)
    grad_theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    tangent = grad_theta + np.float32(math.pi * 0.5)
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / np.maximum(trace, 1e-8)
    return tangent.astype(np.float32), np.clip(coherence, 0.0, 1.0).astype(np.float32)


def _select_energy_pixels(
    energy: np.ndarray,
    count: int,
    min_score: float,
    nms_radius_px: int,
    max_candidates: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = energy.shape
    total = int(h * w)
    count = max(0, int(count))
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    topk = int(max_candidates) if int(max_candidates) > 0 else max(count * 32, count)
    topk = min(max(topk, count), total)
    flat = energy.reshape(-1)
    if topk < total:
        candidates = np.argpartition(flat, -topk)[-topk:]
        candidates = candidates[np.argsort(flat[candidates])[::-1]]
    else:
        candidates = np.argsort(flat)[::-1]
    suppressed = np.zeros((h, w), dtype=bool)
    r = max(0, int(nms_radius_px))
    coords: List[Tuple[int, int]] = []
    scores: List[float] = []
    for idx in candidates.tolist():
        score = float(flat[idx])
        if score < float(min_score):
            break
        y = int(idx // w)
        x = int(idx - y * w)
        if suppressed[y, x]:
            continue
        coords.append((x, y))
        scores.append(score)
        if len(coords) >= count:
            break
        if r > 0:
            suppressed[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)] = True
        else:
            suppressed[y, x] = True
    if not coords:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _angle_abs_cos(theta_a: float, theta_b: float) -> float:
    return float(abs(math.cos(float(theta_a) - float(theta_b))))


def _trace_segment_one_side(
    seed_xy: np.ndarray,
    seed_theta: float,
    sign: float,
    energy: np.ndarray,
    tangent: np.ndarray,
    min_score: float,
    step_px: float,
    half_samples: int,
    search_radius_px: int,
    turn_min_cos: float,
) -> List[Tuple[float, float]]:
    h, w = energy.shape
    pos = np.asarray(seed_xy, dtype=np.float32).copy()
    theta = float(seed_theta)
    out: List[Tuple[float, float]] = []
    r = max(0, int(search_radius_px))
    for _ in range(max(0, int(half_samples))):
        direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32) * float(sign)
        pred = pos + float(step_px) * direction
        px = int(round(float(pred[0])))
        py = int(round(float(pred[1])))
        best_score = -1.0
        best_xy: Optional[Tuple[int, int]] = None
        best_theta = theta
        for yy in range(max(0, py - r), min(h, py + r + 1)):
            for xx in range(max(0, px - r), min(w, px + r + 1)):
                score = float(energy[yy, xx])
                if score < float(min_score):
                    continue
                cand_theta = float(tangent[yy, xx])
                continuity = _angle_abs_cos(theta, cand_theta)
                if continuity < float(turn_min_cos):
                    continue
                # Prefer high-energy ridge pixels, but keep tangent continuity as
                # a real part of the score so we do not jump across nearby edges.
                combined = score * (0.35 + 0.65 * continuity)
                if combined > best_score:
                    best_score = combined
                    best_xy = (xx, yy)
                    best_theta = cand_theta
        if best_xy is None:
            break
        pos = np.asarray(best_xy, dtype=np.float32)
        theta = best_theta
        out.append((float(pos[0]), float(pos[1])))
    return out


def _build_segment_init_points(
    energy: np.ndarray,
    residual: np.ndarray,
    color_image: np.ndarray,
    count: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = energy.shape
    tangent, coherence = _structure_tensor_tangent(energy, int(args.init_orientation_radius_px))
    samples_per_seed = max(1, int(args.segment_samples_per_seed))
    half_samples = max(0, (samples_per_seed - 1) // 2)
    seed_count = int(math.ceil(max(int(count), 1) / float(samples_per_seed)) * 1.35)
    min_score = float(args.init_min_score if float(args.segment_min_score) < 0.0 else args.segment_min_score)
    seed_xy, _ = _select_energy_pixels(
        energy,
        seed_count,
        min_score,
        int(args.segment_seed_nms_radius_px),
        int(args.init_max_candidates),
    )
    coords: List[Tuple[float, float]] = []
    thetas: List[float] = []
    segment_ids: List[int] = []
    segment_orders: List[int] = []
    pair_edges: List[Tuple[int, int]] = []
    segment_id = 0
    for seed in seed_xy:
        sx = int(np.clip(round(float(seed[0])), 0, w - 1))
        sy = int(np.clip(round(float(seed[1])), 0, h - 1))
        seed_theta = float(tangent[sy, sx])
        back = _trace_segment_one_side(
            seed,
            seed_theta,
            -1.0,
            energy,
            tangent,
            min_score,
            float(args.segment_step_px),
            half_samples,
            int(args.segment_trace_search_radius_px),
            float(args.segment_turn_min_cos),
        )
        fwd = _trace_segment_one_side(
            seed,
            seed_theta,
            1.0,
            energy,
            tangent,
            min_score,
            float(args.segment_step_px),
            half_samples,
            int(args.segment_trace_search_radius_px),
            float(args.segment_turn_min_cos),
        )
        chain = list(reversed(back)) + [(float(seed[0]), float(seed[1]))] + fwd
        if len(chain) < 2:
            continue
        start_index = len(coords)
        for order, (x, y) in enumerate(chain):
            if len(coords) >= int(count):
                break
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            coords.append((float(xi), float(yi)))
            thetas.append(float(tangent[yi, xi]))
            segment_ids.append(segment_id)
            segment_orders.append(order)
            if order > 0:
                pair_edges.append((len(coords) - 2, len(coords) - 1))
        segment_id += 1
        if len(coords) >= int(count):
            break
    if len(coords) < int(count):
        missing = int(count) - len(coords)
        extra_xy, _ = _select_energy_pixels(
            energy,
            missing,
            min_score,
            int(args.init_nms_radius_px),
            int(args.init_max_candidates),
        )
        if extra_xy.shape[0] < missing:
            rng = np.random.default_rng(12345)
            extra_random = np.stack(
                [
                    rng.uniform(0.0, max(float(w - 1), 1.0), size=missing - extra_xy.shape[0]),
                    rng.uniform(0.0, max(float(h - 1), 1.0), size=missing - extra_xy.shape[0]),
                ],
                axis=1,
            ).astype(np.float32)
            extra_xy = np.concatenate([extra_xy, extra_random], axis=0)
        for x, y in extra_xy[:missing]:
            xi = int(np.clip(round(float(x)), 0, w - 1))
            yi = int(np.clip(round(float(y)), 0, h - 1))
            coords.append((float(xi), float(yi)))
            thetas.append(float(tangent[yi, xi]))
            segment_ids.append(-1)
            segment_orders.append(-1)
    coords_arr = np.asarray(coords[: int(count)], dtype=np.float32)
    theta_arr = np.asarray(thetas[: int(count)], dtype=np.float32)
    xy_int = np.rint(coords_arr).astype(np.int64)
    xs = np.clip(xy_int[:, 0], 0, w - 1)
    ys = np.clip(xy_int[:, 1], 0, h - 1)
    coh = coherence[ys, xs]
    sigma_long = float(args.init_sigma_long_px) * (1.0 + float(args.init_coherence_long_boost) * coh)
    sigma_short = np.full_like(sigma_long, float(args.init_sigma_short_px), dtype=np.float32)
    colors = color_image[ys, xs, :].astype(np.float32)
    segment_ids_arr = np.asarray(segment_ids[: int(count)], dtype=np.int32)
    pair_arr = np.asarray(
        [(a, b) for a, b in pair_edges if a < int(count) and b < int(count)],
        dtype=np.int64,
    )
    if pair_arr.size == 0:
        pair_arr = np.zeros((0, 2), dtype=np.int64)
    return coords_arr, theta_arr, sigma_long.astype(np.float32), sigma_short, colors, segment_ids_arr, pair_arr


def _cholesky_from_orientation(theta: np.ndarray, sigma_long: np.ndarray, sigma_short: np.ndarray) -> np.ndarray:
    out = np.zeros((theta.shape[0], 3), dtype=np.float32)
    for i in range(theta.shape[0]):
        c = float(math.cos(float(theta[i])))
        s = float(math.sin(float(theta[i])))
        sl = max(float(sigma_long[i]), 1e-3)
        ss = max(float(sigma_short[i]), 1e-3)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        cov = rot @ np.diag([sl * sl, ss * ss]).astype(np.float32) @ rot.T
        chol = np.linalg.cholesky(cov + np.eye(2, dtype=np.float32) * 1e-5)
        out[i, 0] = chol[0, 0]
        out[i, 1] = chol[1, 0]
        out[i, 2] = chol[1, 1]
    return out


def _target_init_gaussianimage(
    model,
    signed_target: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    residual_clip: float,
    args: argparse.Namespace,
    init_color_image: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    h, w = signed_target.shape[:2]
    n = int(model._xyz.shape[0])
    color_image = signed_target if init_color_image is None else init_color_image
    energy = np.clip(np.abs(residual).mean(axis=2) / max(float(residual_clip), 1e-8), 0.0, 1.0)
    energy = (energy * (np.clip(weight, 0.0, 1.0) ** max(float(args.init_weight_power), 0.0))).astype(np.float32)
    if bool(args.segment_init):
        coords, theta, sigma_long, sigma_short, colors, segment_ids, pair_edges = _build_segment_init_points(
            energy,
            residual,
            color_image,
            n,
            args,
        )
    else:
        coords, _ = _select_energy_pixels(
            energy,
            n,
            float(args.init_min_score),
            int(args.init_nms_radius_px),
            int(args.init_max_candidates),
        )
        if coords.shape[0] < n:
            missing = n - int(coords.shape[0])
            rng = np.random.default_rng(12345)
            extra_xy = np.stack(
                [
                    rng.uniform(0.0, max(float(w - 1), 1.0), size=missing),
                    rng.uniform(0.0, max(float(h - 1), 1.0), size=missing),
                ],
                axis=1,
            ).astype(np.float32)
            coords = np.concatenate([coords, extra_xy], axis=0)
        xy_int = np.rint(coords).astype(np.int64)
        xs = np.clip(xy_int[:, 0], 0, w - 1)
        ys = np.clip(xy_int[:, 1], 0, h - 1)
        tangent, coherence = _structure_tensor_tangent(energy, int(args.init_orientation_radius_px))
        theta = tangent[ys, xs]
        coh = coherence[ys, xs]
        sigma_long = float(args.init_sigma_long_px) * (1.0 + float(args.init_coherence_long_boost) * coh)
        sigma_short = np.full_like(sigma_long, float(args.init_sigma_short_px), dtype=np.float32)
        colors = color_image[ys, xs, :].astype(np.float32)
        segment_ids = np.full((n,), -1, dtype=np.int32)
        pair_edges = np.zeros((0, 2), dtype=np.int64)
    cholesky = _cholesky_from_orientation(theta.astype(np.float32), sigma_long.astype(np.float32), sigma_short)
    means = np.empty((n, 2), dtype=np.float32)
    means[:, 0] = np.clip(coords[:, 0] / max(float(w - 1), 1.0) * 2.0 - 1.0, -0.999, 0.999)
    means[:, 1] = np.clip(coords[:, 1] / max(float(h - 1), 1.0) * 2.0 - 1.0, -0.999, 0.999)
    xyz = np.arctanh(means).astype(np.float32)
    bound = np.asarray([0.5, 0.0, 0.5], dtype=np.float32)[None, :]
    cholesky_param = cholesky - bound
    device = model._xyz.device
    with torch.no_grad():
        model._xyz.copy_(torch.from_numpy(xyz).to(device=device, dtype=model._xyz.dtype))
        model._features_dc.copy_(torch.from_numpy(colors).to(device=device, dtype=model._features_dc.dtype))
        if hasattr(model, "_cholesky"):
            model._cholesky.copy_(torch.from_numpy(cholesky_param).to(device=device, dtype=model._cholesky.dtype))
        if hasattr(model, "_opacity"):
            model._opacity.fill_(1.0)
        if hasattr(model, "background"):
            model.background.fill_(0.5)
    return {
        "means_norm": means.astype(np.float32),
        "cholesky": cholesky.astype(np.float32),
        "colors": colors.astype(np.float32),
        "segment_id": segment_ids.astype(np.int32),
        "pair_edges": pair_edges.astype(np.int64),
    }


def _import_gaussianimage(repo_root: Path, model_name: str):
    cholesky_py = repo_root / "gaussianimage_cholesky.py"
    rs_py = repo_root / "gaussianimage_rs.py"
    selected_py = cholesky_py if model_name == "cholesky" else rs_py
    if not selected_py.is_file():
        raise FileNotFoundError(
            f"GaussianImage model file not found under {repo_root}: {selected_py.name}. "
            "Clone it with: git clone --recursive https://github.com/Xinjie-Q/GaussianImage.git"
        )
    # GaussianImage imports quantize.py and pytorch_msssim at module import time,
    # but our HF fitting path always uses quantize=False and computes its own
    # weighted L1/L2 loss. Tiny stubs avoid pulling optional codec/SSIM
    # dependencies that would otherwise try to upgrade the existing torch env.
    if "quantize" not in sys.modules:
        stub = types.ModuleType("quantize")
        stub.__all__ = []
        sys.modules["quantize"] = stub
    if "pytorch_msssim" not in sys.modules:
        msssim_stub = types.ModuleType("pytorch_msssim")

        def _unused_msssim(*_args, **_kwargs):
            raise RuntimeError("pytorch_msssim is stubbed; this adapter does not use GaussianImage SSIM losses.")

        class _UnusedSSIM:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, *_args, **_kwargs):
                return _unused_msssim()

        msssim_stub.ms_ssim = _unused_msssim
        msssim_stub.ssim = _unused_msssim
        msssim_stub.SSIM = _UnusedSSIM
        sys.modules["pytorch_msssim"] = msssim_stub
    sys.path.insert(0, str(repo_root))
    gsplat_root = repo_root / "gsplat"
    if gsplat_root.is_dir():
        sys.path.insert(0, str(gsplat_root))

    def _load(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import GaussianImage module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if model_name == "cholesky":
        cholesky_module = _load(cholesky_py, "gaussianimage_cholesky")
        if not hasattr(cholesky_module, "GaussianImage_Cholesky"):
            raise ImportError(f"Missing GaussianImage_Cholesky in {cholesky_py}")
        return {"GaussianImage_Cholesky": cholesky_module.GaussianImage_Cholesky}
    rs_module = _load(rs_py, "gaussianimage_rs")
    if not hasattr(rs_module, "GaussianImage_RS"):
        raise ImportError(f"Missing GaussianImage_RS in {rs_py}")
    return {"GaussianImage_RS": rs_module.GaussianImage_RS}


def _target_signed_hf(
    target: np.ndarray,
    anchor: np.ndarray,
    mask: np.ndarray,
    highpass_kernel: int,
    detail_alpha: float,
    residual_clip: float,
    confidence_power: float,
    mask_power: float,
    neutral_outside_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_hp = target - _box_blur_rgb(target, highpass_kernel)
    anchor_hp = anchor - _box_blur_rgb(anchor, highpass_kernel)
    residual = detail_alpha * (target_hp - anchor_hp)
    residual = np.clip(residual, -residual_clip, residual_clip).astype(np.float32)
    trust = np.clip(mask, 0.0, 1.0) ** max(mask_power, 0.0)
    trust = trust ** max(confidence_power, 0.0)
    if neutral_outside_mask:
        residual = residual * trust[..., None]
    signed = np.clip(0.5 + residual / (2.0 * max(residual_clip, 1e-8)), 0.0, 1.0).astype(np.float32)
    return signed, residual, trust.astype(np.float32)


def _target_signed_rgb_delta(
    target: np.ndarray,
    anchor: np.ndarray,
    mask: np.ndarray,
    detail_alpha: float,
    residual_clip: float,
    confidence_power: float,
    mask_power: float,
    neutral_outside_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = float(detail_alpha) * (target.astype(np.float32) - anchor.astype(np.float32))
    residual = np.clip(residual, -float(residual_clip), float(residual_clip)).astype(np.float32)
    trust = np.clip(mask, 0.0, 1.0) ** max(float(mask_power), 0.0)
    trust = trust ** max(float(confidence_power), 0.0)
    if neutral_outside_mask:
        residual = residual * trust[..., None]
    signed = np.clip(0.5 + residual / (2.0 * max(float(residual_clip), 1e-8)), 0.0, 1.0).astype(np.float32)
    return signed, residual, trust.astype(np.float32)


def _fit_gaussianimage(
    module,
    fit_target: np.ndarray,
    target_residual: np.ndarray,
    weight: np.ndarray,
    num_gaussians: int,
    model_name: str,
    iterations: int,
    lr: float,
    optimizer: str,
    loss_name: str,
    lambda_l1: float,
    lambda_l2: float,
    background_weight: float,
    args: argparse.Namespace,
    init_color_image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, object, List[float], Dict[str, np.ndarray]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GaussianImage fitting requires CUDA for diff_gaussian_rasterization.")
    h, w = fit_target.shape[:2]
    device = torch.device("cuda")
    image_t = torch.from_numpy(fit_target).to(device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
    weight_t = torch.from_numpy(np.clip(weight, 0.0, 1.0)).to(device=device, dtype=torch.float32)[None, :, :]
    weight_t = torch.clamp(weight_t + float(background_weight), 0.0, 1.0)
    model_cls = module["GaussianImage_Cholesky"] if model_name == "cholesky" else module["GaussianImage_RS"]
    model = model_cls(
        loss_type="L2",
        opt_type=str(optimizer),
        num_points=int(num_gaussians),
        H=int(h),
        W=int(w),
        BLOCK_H=16,
        BLOCK_W=16,
        device=device,
        lr=float(lr),
        quantize=False,
    ).to(device)
    if hasattr(model, "background"):
        with torch.no_grad():
            model.background.fill_(0.5)
    init_meta: Dict[str, np.ndarray] = {}
    if not bool(args.init_random):
        init_meta = _target_init_gaussianimage(
            model,
            fit_target,
            target_residual,
            weight,
            float(args.residual_clip),
            args,
            init_color_image=init_color_image,
        )
    init_means_t: Optional[torch.Tensor] = None
    init_cholesky_t: Optional[torch.Tensor] = None
    pair_edges_t: Optional[torch.Tensor] = None
    segment_valid_t: Optional[torch.Tensor] = None
    if init_meta:
        init_means_t = torch.from_numpy(init_meta["means_norm"]).to(device=device, dtype=torch.float32)
        init_cholesky_t = torch.from_numpy(init_meta["cholesky"]).to(device=device, dtype=torch.float32)
        segment_ids_np = init_meta["segment_id"].reshape(-1)
        segment_valid_t = torch.from_numpy(segment_ids_np >= 0).to(device=device)
        if init_meta["pair_edges"].shape[0] > 0:
            pair_edges_t = torch.from_numpy(init_meta["pair_edges"]).to(device=device, dtype=torch.long)
    losses: List[float] = []
    for _ in range(int(iterations)):
        model.optimizer.zero_grad(set_to_none=True)
        rendered = model.forward()["render"].squeeze(0)
        diff = rendered - image_t
        if loss_name == "l1":
            loss = (diff.abs() * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
        elif loss_name == "l2":
            loss = ((diff * diff) * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
        else:
            l1 = (diff.abs() * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
            l2 = ((diff * diff) * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
            loss = float(lambda_l1) * l1 + float(lambda_l2) * l2
        if init_means_t is not None and segment_valid_t is not None and bool(torch.any(segment_valid_t)):
            current_means = torch.tanh(model._xyz)
            valid_means = segment_valid_t.to(device=current_means.device)
            anchor_weight = float(args.segment_anchor_weight)
            if anchor_weight > 0.0:
                loss = loss + anchor_weight * F.smooth_l1_loss(current_means[valid_means], init_means_t[valid_means])
            if pair_edges_t is not None and int(pair_edges_t.shape[0]) > 0:
                p0 = pair_edges_t[:, 0]
                p1 = pair_edges_t[:, 1]
                pair_weight = float(args.segment_pair_weight)
                if pair_weight > 0.0:
                    curr_delta = current_means[p1] - current_means[p0]
                    init_delta = init_means_t[p1] - init_means_t[p0]
                    loss = loss + pair_weight * F.smooth_l1_loss(curr_delta, init_delta)
                color_weight = float(args.segment_color_smooth_weight)
                if color_weight > 0.0:
                    colors = torch.clamp(model.get_features, 0.0, 1.0)
                    loss = loss + color_weight * F.smooth_l1_loss(colors[p1], colors[p0])
            shape_weight = float(args.segment_shape_weight)
            if shape_weight > 0.0 and init_cholesky_t is not None and hasattr(model, "get_cholesky_elements"):
                loss = loss + shape_weight * F.smooth_l1_loss(model.get_cholesky_elements[valid_means], init_cholesky_t[valid_means])
        loss.backward()
        model.optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    with torch.no_grad():
        rendered = model.forward()["render"].squeeze(0).detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return rendered, model, losses, init_meta


def _extract_primitives(model, model_name: str, h: int, w: int) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        xyz = torch.tanh(model._xyz).detach().cpu().numpy().astype(np.float32)
        mu_xy = np.empty((xyz.shape[0], 2), dtype=np.float32)
        mu_xy[:, 0] = (xyz[:, 0] * 0.5 + 0.5) * float(w - 1)
        mu_xy[:, 1] = (xyz[:, 1] * 0.5 + 0.5) * float(h - 1)
        features = torch.clamp(model.get_features, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        opacity = model.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32)
        payload = {
            "mu_xy": mu_xy,
            "color": features,
            "opacity": opacity,
        }
        if model_name == "cholesky" and hasattr(model, "_cholesky"):
            payload["cholesky"] = model.get_cholesky_elements.detach().cpu().numpy().astype(np.float32)
        if model_name == "rs":
            if hasattr(model, "_scaling"):
                payload["scaling"] = model._scaling.detach().cpu().numpy().astype(np.float32)
            if hasattr(model, "_rotation"):
                payload["rotation"] = model._rotation.detach().cpu().numpy().astype(np.float32)
    return payload


def _signed_to_residual(signed: np.ndarray, residual_clip: float) -> np.ndarray:
    return (signed.astype(np.float32) - 0.5) * (2.0 * float(residual_clip))


def _weighted_l1(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (a.shape[2] if a.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    return float((np.abs(a - b) * weight[..., None]).sum() / denom)


def _weighted_energy(value: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (value.shape[2] if value.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    return float((np.abs(value) * weight[..., None]).sum() / denom)


def _pearson_abs(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    aa = np.abs(a).mean(axis=2).reshape(-1)
    bb = np.abs(b).mean(axis=2).reshape(-1)
    ww = weight.reshape(-1)
    keep = ww > 1e-6
    if int(keep.sum()) < 4:
        return float("nan")
    aa = aa[keep]
    bb = bb[keep]
    ww = ww[keep]
    ww = ww / max(float(ww.sum()), 1e-8)
    ma = float((aa * ww).sum())
    mb = float((bb * ww).sum())
    da = aa - ma
    db = bb - mb
    den = float(np.sqrt((ww * da * da).sum() * (ww * db * db).sum()))
    if den <= 1e-8:
        return float("nan")
    return float((ww * da * db).sum() / den)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _abs_vis(value: np.ndarray, residual_clip: float) -> np.ndarray:
    gray = np.clip(np.abs(value).mean(axis=2) / max(float(residual_clip), 1e-8), 0.0, 1.0)
    return np.repeat(gray[..., None], 3, axis=2)


def _overlay(target_abs: np.ndarray, recon_abs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*target_abs.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(target_abs, 0.0, 1.0)
    rgb[..., 1] = np.clip(recon_abs, 0.0, 1.0)
    rgb[..., 2] = np.clip(recon_abs, 0.0, 1.0)
    return np.clip(rgb * (0.12 + 0.88 * np.clip(weight[..., None], 0.0, 1.0)), 0.0, 1.0)


def _light_abs(gray: np.ndarray, strength: float) -> np.ndarray:
    value = 1.0 - float(strength) * np.clip(gray, 0.0, 1.0)
    return np.repeat(np.clip(value[..., None], 0.0, 1.0), 3, axis=2)


def _light_signed(value: np.ndarray, strength: float) -> np.ndarray:
    # Signed HF uses 0.5 as neutral. Show residual magnitude on a white canvas
    # so sparse line structure is easier to inspect than on a black canvas.
    residual = np.abs(np.clip(value, 0.0, 1.0) - 0.5) * 2.0
    return 1.0 - float(strength) * residual


def _light_overlay(target_abs: np.ndarray, recon_abs: np.ndarray, weight: np.ndarray, strength: float) -> np.ndarray:
    target = np.clip(target_abs, 0.0, 1.0) * np.clip(weight, 0.0, 1.0)
    recon = np.clip(recon_abs, 0.0, 1.0) * np.clip(weight, 0.0, 1.0)
    rgb = np.ones((*target.shape, 3), dtype=np.float32)
    rgb[..., 1] -= float(strength) * target
    rgb[..., 2] -= float(strength) * target
    rgb[..., 0] -= float(strength) * recon
    return np.clip(rgb, 0.0, 1.0)


def _light_primitive_overlay(primitive_overlay: np.ndarray, strength: float) -> np.ndarray:
    gray = np.clip(primitive_overlay.mean(axis=2), 0.0, 1.0)
    return _light_abs(gray, strength)


def _rgb_with_hf(base: np.ndarray, residual: np.ndarray, weight: Optional[np.ndarray] = None, strength: float = 1.0) -> np.ndarray:
    if weight is not None:
        residual = residual * np.clip(weight[..., None], 0.0, 1.0)
    return np.clip(base + float(strength) * residual, 0.0, 1.0)


def _rgb_hf_error_overlay(
    base: np.ndarray,
    target_abs: np.ndarray,
    recon_abs: np.ndarray,
    weight: np.ndarray,
    strength: float,
) -> np.ndarray:
    # Red marks target HF not explained by 2DGS, cyan marks extra 2DGS HF.
    target = np.clip(target_abs, 0.0, 1.0) * np.clip(weight, 0.0, 1.0)
    recon = np.clip(recon_abs, 0.0, 1.0) * np.clip(weight, 0.0, 1.0)
    missing = np.clip(target - recon, 0.0, 1.0)
    extra = np.clip(recon - target, 0.0, 1.0)
    rgb = 0.72 * np.clip(base, 0.0, 1.0) + 0.18
    rgb[..., 0] += float(strength) * missing
    rgb[..., 1] += float(strength) * extra
    rgb[..., 2] += float(strength) * extra
    return np.clip(rgb, 0.0, 1.0)


def _sample_rgb_bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(xy[:, 0].astype(np.float32), 0.0, max(float(w - 1), 0.0))
    y = np.clip(xy[:, 1].astype(np.float32), 0.0, max(float(h - 1), 0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0.astype(np.float32))[:, None]
    wy = (y - y0.astype(np.float32))[:, None]
    c00 = image[y0, x0, :]
    c10 = image[y0, x1, :]
    c01 = image[y1, x0, :]
    c11 = image[y1, x1, :]
    return (
        c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy
    ).astype(np.float32)


def _render_model_with_colors(model, colors: np.ndarray, background: float = 0.0) -> np.ndarray:
    device = model._features_dc.device
    old_features = model._features_dc.detach().clone()
    old_background = model.background.detach().clone() if hasattr(model, "background") else None
    try:
        with torch.no_grad():
            model._features_dc.copy_(torch.from_numpy(colors).to(device=device, dtype=model._features_dc.dtype))
            if hasattr(model, "background"):
                model.background.fill_(float(background))
            render = (
                model.forward()["render"]
                .squeeze(0)
                .detach()
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    finally:
        with torch.no_grad():
            model._features_dc.copy_(old_features)
            if old_background is not None:
                model.background.copy_(old_background)
    return render


def _composite_over(base: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)
    if a.ndim == 2:
        a = a[..., None]
    return np.clip(foreground * a + base * (1.0 - a), 0.0, 1.0)


def _primitive_overlay(primitives: Dict[str, np.ndarray], h: int, w: int, max_draw: int = 4096) -> np.ndarray:
    image = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    mu = primitives["mu_xy"]
    color = primitives["color"]
    opacity = primitives.get("opacity", np.ones((mu.shape[0],), dtype=np.float32))
    order = np.argsort(opacity)[::-1][: min(int(max_draw), int(mu.shape[0]))]
    cholesky = primitives.get("cholesky")
    scaling = primitives.get("scaling")
    rotation = primitives.get("rotation")
    for i in order.tolist():
        x, y = float(mu[i, 0]), float(mu[i, 1])
        rgb = tuple(int(np.clip(color[i, j] * 255.0, 0, 255)) for j in range(3))
        if cholesky is not None and cholesky.shape[1] >= 3:
            a = float(abs(cholesky[i, 0]))
            b = float(abs(cholesky[i, 1]))
            c = float(abs(cholesky[i, 2]))
            length = max(1.0, min(12.0, 160.0 * max(a, c)))
            theta = math.atan2(b, a + 1e-8)
        elif scaling is not None and rotation is not None:
            sx = float(abs(scaling[i, 0])) if scaling.ndim > 1 else float(abs(scaling[i]))
            length = max(1.0, min(12.0, 160.0 * sx))
            theta = float(rotation[i, 0]) if rotation.ndim > 1 else float(rotation[i])
        else:
            length = 2.0
            theta = 0.0
        dx = math.cos(theta) * length
        dy = math.sin(theta) * length
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=rgb, width=1)
    return np.asarray(image, dtype=np.float32) / 255.0


def _panel(rgb: np.ndarray, label: str) -> Image.Image:
    rgb = np.clip(rgb, 0.0, 1.0)
    image = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(560, image.width), 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return image


def _write_sheet(
    path: Path,
    signed_target: np.ndarray,
    signed_render: np.ndarray,
    target_residual: np.ndarray,
    recon_residual: np.ndarray,
    error: np.ndarray,
    weight: np.ndarray,
    primitive_overlay: np.ndarray,
    residual_clip: float,
) -> None:
    target_abs = _abs_vis(target_residual, residual_clip)
    recon_abs = _abs_vis(recon_residual, residual_clip)
    panels = [
        _panel(signed_target, "target signed HF"),
        _panel(signed_render, "GaussianImage signed HF"),
        _panel(target_abs, "target abs HF"),
        _panel(recon_abs, "GaussianImage abs HF"),
        _panel(np.repeat(np.clip(weight[..., None], 0.0, 1.0), 3, axis=2), "weighted trust edge"),
        _panel(_abs_vis(error, residual_clip), "abs residual error"),
        _panel(_overlay(target_abs[..., 0], recon_abs[..., 0], weight), "overlay target=red 2DGS=cyan"),
        _panel(primitive_overlay, "exported 2D Gaussian primitives"),
    ]
    width = max(p.width for p in panels)
    height = max(p.height for p in panels)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * width, (i // cols) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _write_rgb_sheet(
    path: Path,
    anchor: np.ndarray,
    target: np.ndarray,
    target_rgb_hf: np.ndarray,
    recon_rgb_hf: np.ndarray,
    recon_rgb_hf_weighted: np.ndarray,
    rgb_error_overlay: np.ndarray,
) -> None:
    panels = [
        _panel(anchor, "anchor RGB"),
        _panel(target, "edge target RGB"),
        _panel(target_rgb_hf, "anchor + target HF"),
        _panel(recon_rgb_hf, "anchor + 2DGS HF"),
        _panel(recon_rgb_hf_weighted, "anchor + trusted 2DGS HF"),
        _panel(rgb_error_overlay, "RGB error: red missing, cyan extra"),
    ]
    width = max(p.width for p in panels)
    height = max(p.height for p in panels)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * width, (i // cols) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _mean(rows: Sequence[Dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.external_repo_root).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {output_dir}")
    module = _import_gaussianimage(repo_root, str(args.model))

    target_paths = _list_images(target_dir)
    anchor_paths = _list_images(anchor_dir)
    mask_paths = _list_images(mask_dir)
    if int(args.limit) > 0:
        target_paths = target_paths[: int(args.limit)]

    dirs = {
        "target_hf": output_dir / "target_hf",
        "recon_hf": output_dir / "recon_hf",
        "target_abs": output_dir / "target_abs",
        "recon_abs": output_dir / "recon_abs",
        "edge_recon": output_dir / "edge_recon",
        "overlay": output_dir / "overlay",
        "primitive_overlay": output_dir / "primitive_overlay",
        "rgb_anchor": output_dir / "rgb_anchor",
        "rgb_target": output_dir / "rgb_target",
        "rgb_recon": output_dir / "rgb_recon",
        "rgb_recon_error": output_dir / "rgb_recon_error",
        "rgb_delta_target": output_dir / "rgb_delta_target",
        "rgb_delta_recon": output_dir / "rgb_delta_recon",
        "rgb_delta_apply": output_dir / "rgb_delta_apply",
        "rgb_delta_apply_error": output_dir / "rgb_delta_apply_error",
        "rgb_delta_recon_trust": output_dir / "rgb_delta_recon_trust",
        "rgb_delta_apply_trust": output_dir / "rgb_delta_apply_trust",
        "rgb_delta_apply_trust_error": output_dir / "rgb_delta_apply_trust_error",
        "rgb_delta_extra_outside": output_dir / "rgb_delta_extra_outside",
        "rgb_target_hf": output_dir / "rgb_target_hf",
        "rgb_recon_hf": output_dir / "rgb_recon_hf",
        "rgb_recon_hf_weighted": output_dir / "rgb_recon_hf_weighted",
        "rgb_error_overlay": output_dir / "rgb_error_overlay",
        "rgb_sheet": output_dir / "rgb_sheet",
        "carrier_alpha": output_dir / "carrier_alpha",
        "carrier_rgb_target": output_dir / "carrier_rgb_target",
        "carrier_rgb_anchor": output_dir / "carrier_rgb_anchor",
        "carrier_rgb_target_over_anchor": output_dir / "carrier_rgb_target_over_anchor",
        "sheet": output_dir / "sheet",
        "primitives": output_dir / "primitives",
    }
    if bool(args.light_visuals):
        dirs.update(
            {
                "target_abs_light": output_dir / "target_abs_light",
                "recon_abs_light": output_dir / "recon_abs_light",
                "target_hf_light": output_dir / "target_hf_light",
                "recon_hf_light": output_dir / "recon_hf_light",
                "overlay_light": output_dir / "overlay_light",
                "primitive_overlay_light": output_dir / "primitive_overlay_light",
            }
        )
    if bool(args.save_pt):
        dirs["pt"] = output_dir / "pt"
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    anchor_lookup = _lookup(anchor_paths)
    mask_lookup = _lookup(mask_paths)
    rows: List[Dict[str, float]] = []
    frames: List[Dict[str, object]] = []

    print(f"[gaussianimage-hf-v0] repo   : {repo_root}")
    print(f"[gaussianimage-hf-v0] target : {target_dir}")
    print(f"[gaussianimage-hf-v0] anchor : {anchor_dir}")
    print(f"[gaussianimage-hf-v0] mask   : {mask_dir}")
    print(f"[gaussianimage-hf-v0] output : {output_dir}")
    print(
        f"[gaussianimage-hf-v0] fit    : model={args.model} n={args.num_gaussians} "
        f"iters={args.iterations} lr={args.lr} loss={args.loss} mode={args.fit_target_mode}"
    )

    for index, target_path in enumerate(tqdm(target_paths, desc="GaussianImage HF")):
        anchor_path = _resolve(anchor_paths, anchor_lookup, target_path, index, args.match_policy)
        mask_path = _resolve(mask_paths, mask_lookup, target_path, index, args.match_policy)
        if anchor_path is None or mask_path is None:
            continue
        target = _load_rgb(target_path)
        size = (target.shape[1], target.shape[0])
        anchor = _load_rgb(anchor_path, size=size)
        mask = _load_gray(mask_path, size=size)
        signed_hf_target, hf_target_residual, weight = _target_signed_hf(
            target,
            anchor,
            mask,
            int(args.highpass_kernel),
            float(args.detail_alpha),
            float(args.residual_clip),
            float(args.confidence_power),
            float(args.mask_power),
            bool(args.neutral_outside_mask),
        )
        if str(args.fit_target_mode) == "rgb_delta":
            signed_target, target_residual, weight = _target_signed_rgb_delta(
                target,
                anchor,
                mask,
                float(args.detail_alpha),
                float(args.residual_clip),
                float(args.confidence_power),
                float(args.mask_power),
                bool(args.neutral_outside_mask),
            )
            fit_target = signed_target
            if str(args.rgb_loss_weight_mode) == "full":
                fit_weight = np.ones_like(weight, dtype=np.float32)
            elif str(args.rgb_loss_weight_mode) == "trust_plus_background":
                fit_weight = np.clip(weight + float(args.background_weight), 0.0, 1.0).astype(np.float32)
            else:
                fit_weight = weight
            init_color_image = None
        elif str(args.fit_target_mode) == "rgb":
            signed_target = signed_hf_target
            target_residual = hf_target_residual
            fit_target = target
            if str(args.rgb_loss_weight_mode) == "full":
                fit_weight = np.ones_like(weight, dtype=np.float32)
            elif str(args.rgb_loss_weight_mode) == "trust_plus_background":
                fit_weight = np.clip(weight + float(args.background_weight), 0.0, 1.0).astype(np.float32)
            else:
                fit_weight = weight
            init_color_image = target
        else:
            signed_target = signed_hf_target
            target_residual = hf_target_residual
            fit_target = signed_target
            fit_weight = weight
            init_color_image = None
        fit_render, model, losses, init_meta = _fit_gaussianimage(
            module,
            fit_target,
            target_residual,
            fit_weight,
            int(args.num_gaussians),
            str(args.model),
            int(args.iterations),
            float(args.lr),
            str(args.optimizer),
            str(args.loss),
            float(args.lambda_l1),
            float(args.lambda_l2),
            float(args.background_weight),
            args,
            init_color_image=init_color_image,
        )
        if str(args.fit_target_mode) == "rgb_delta":
            signed_render = fit_render
            recon_residual = _signed_to_residual(signed_render, float(args.residual_clip))
            rgb_render = np.clip(anchor + recon_residual, 0.0, 1.0)
            rgb_l1 = _weighted_l1(rgb_render, target, np.ones_like(weight, dtype=np.float32))
        elif str(args.fit_target_mode) == "rgb":
            rgb_render = fit_render
            recon_residual = float(args.detail_alpha) * (
                rgb_render - _box_blur_rgb(rgb_render, int(args.highpass_kernel)) - (anchor - _box_blur_rgb(anchor, int(args.highpass_kernel)))
            )
            recon_residual = np.clip(recon_residual, -float(args.residual_clip), float(args.residual_clip)).astype(np.float32)
            signed_render = np.clip(
                0.5 + recon_residual / (2.0 * max(float(args.residual_clip), 1e-8)),
                0.0,
                1.0,
            ).astype(np.float32)
            rgb_l1 = _weighted_l1(rgb_render, target, np.ones_like(weight, dtype=np.float32))
        else:
            signed_render = fit_render
            rgb_render = np.clip(anchor + _signed_to_residual(signed_render, float(args.residual_clip)), 0.0, 1.0)
            recon_residual = _signed_to_residual(signed_render, float(args.residual_clip))
            rgb_l1 = float("nan")
        error = recon_residual - target_residual
        target_abs = np.clip(np.abs(target_residual).mean(axis=2) / max(float(args.residual_clip), 1e-8), 0.0, 1.0)
        recon_abs = np.clip(np.abs(recon_residual).mean(axis=2) / max(float(args.residual_clip), 1e-8), 0.0, 1.0)
        recon_residual_trust = recon_residual * np.clip(weight[..., None], 0.0, 1.0)
        signed_render_trust = np.clip(
            0.5 + recon_residual_trust / (2.0 * max(float(args.residual_clip), 1e-8)),
            0.0,
            1.0,
        ).astype(np.float32)
        rgb_render_trust = np.clip(anchor + recon_residual_trust, 0.0, 1.0)
        extra_outside = np.clip(
            np.abs(recon_residual).mean(axis=2) * (1.0 - np.clip(weight, 0.0, 1.0)) / max(float(args.residual_clip), 1e-8),
            0.0,
            1.0,
        )
        overlay = _overlay(target_abs, recon_abs, weight)
        target_rgb_hf = _rgb_with_hf(anchor, target_residual)
        recon_rgb_hf = _rgb_with_hf(anchor, recon_residual)
        recon_rgb_hf_weighted = _rgb_with_hf(anchor, recon_residual, weight=weight)
        rgb_error_overlay = _rgb_hf_error_overlay(anchor, target_abs, recon_abs, weight, 0.65)
        primitives = _extract_primitives(model, str(args.model), target.shape[0], target.shape[1])
        if init_meta:
            primitives["segment_id"] = init_meta["segment_id"]
            primitives["pair_edges"] = init_meta["pair_edges"]
            primitives["init_mu_norm"] = init_meta["means_norm"]
            primitives["init_cholesky"] = init_meta["cholesky"]
        carrier_target_colors = _sample_rgb_bilinear(target, primitives["mu_xy"])
        carrier_anchor_colors = _sample_rgb_bilinear(anchor, primitives["mu_xy"])
        carrier_rgb_target = _render_model_with_colors(model, carrier_target_colors, background=0.0)
        carrier_rgb_anchor = _render_model_with_colors(model, carrier_anchor_colors, background=0.0)
        carrier_alpha_rgb = _render_model_with_colors(
            model,
            np.ones_like(carrier_target_colors, dtype=np.float32),
            background=0.0,
        )
        carrier_alpha = np.clip(carrier_alpha_rgb.mean(axis=2), 0.0, 1.0)
        carrier_rgb_target_over_anchor = _composite_over(anchor, carrier_rgb_target, carrier_alpha)
        primitive_overlay = _primitive_overlay(primitives, target.shape[0], target.shape[1])

        stem = target_path.stem
        _save_rgb(dirs["target_hf"] / f"{stem}.png", signed_target)
        _save_rgb(dirs["recon_hf"] / f"{stem}.png", signed_render)
        _save_gray(dirs["target_abs"] / f"{stem}.png", target_abs)
        _save_gray(dirs["recon_abs"] / f"{stem}.png", recon_abs)
        _save_rgb(dirs["edge_recon"] / f"{stem}.png", recon_rgb_hf)
        _save_rgb(dirs["overlay"] / f"{stem}.png", overlay)
        _save_rgb(dirs["primitive_overlay"] / f"{stem}.png", primitive_overlay)
        _save_rgb(dirs["rgb_anchor"] / f"{stem}.png", anchor)
        _save_rgb(dirs["rgb_target"] / f"{stem}.png", target)
        _save_rgb(dirs["rgb_recon"] / f"{stem}.png", rgb_render)
        _save_rgb(dirs["rgb_recon_error"] / f"{stem}.png", np.repeat(np.clip(np.abs(rgb_render - target).mean(axis=2, keepdims=True) * 4.0, 0.0, 1.0), 3, axis=2))
        _save_rgb(dirs["rgb_delta_target"] / f"{stem}.png", signed_target)
        _save_rgb(dirs["rgb_delta_recon"] / f"{stem}.png", signed_render)
        _save_rgb(dirs["rgb_delta_apply"] / f"{stem}.png", rgb_render)
        _save_rgb(dirs["rgb_delta_apply_error"] / f"{stem}.png", np.repeat(np.clip(np.abs(rgb_render - target).mean(axis=2, keepdims=True) * 4.0, 0.0, 1.0), 3, axis=2))
        _save_rgb(dirs["rgb_delta_recon_trust"] / f"{stem}.png", signed_render_trust)
        _save_rgb(dirs["rgb_delta_apply_trust"] / f"{stem}.png", rgb_render_trust)
        _save_rgb(dirs["rgb_delta_apply_trust_error"] / f"{stem}.png", np.repeat(np.clip(np.abs(rgb_render_trust - target).mean(axis=2, keepdims=True) * 4.0, 0.0, 1.0), 3, axis=2))
        _save_gray(dirs["rgb_delta_extra_outside"] / f"{stem}.png", extra_outside)
        _save_rgb(dirs["rgb_target_hf"] / f"{stem}.png", target_rgb_hf)
        _save_rgb(dirs["rgb_recon_hf"] / f"{stem}.png", recon_rgb_hf)
        _save_rgb(dirs["rgb_recon_hf_weighted"] / f"{stem}.png", recon_rgb_hf_weighted)
        _save_rgb(dirs["rgb_error_overlay"] / f"{stem}.png", rgb_error_overlay)
        _save_gray(dirs["carrier_alpha"] / f"{stem}.png", carrier_alpha)
        _save_rgb(dirs["carrier_rgb_target"] / f"{stem}.png", carrier_rgb_target)
        _save_rgb(dirs["carrier_rgb_anchor"] / f"{stem}.png", carrier_rgb_anchor)
        _save_rgb(dirs["carrier_rgb_target_over_anchor"] / f"{stem}.png", carrier_rgb_target_over_anchor)
        if bool(args.light_visuals):
            light_strength = float(args.light_visual_strength)
            _save_rgb(dirs["target_abs_light"] / f"{stem}.png", _light_abs(target_abs, light_strength))
            _save_rgb(dirs["recon_abs_light"] / f"{stem}.png", _light_abs(recon_abs, light_strength))
            _save_rgb(dirs["target_hf_light"] / f"{stem}.png", _light_signed(signed_target, light_strength))
            _save_rgb(dirs["recon_hf_light"] / f"{stem}.png", _light_signed(signed_render, light_strength))
            _save_rgb(dirs["overlay_light"] / f"{stem}.png", _light_overlay(target_abs, recon_abs, weight, light_strength))
            _save_rgb(
                dirs["primitive_overlay_light"] / f"{stem}.png",
                _light_primitive_overlay(primitive_overlay, light_strength),
            )
        if index < int(args.debug_limit):
            _write_sheet(
                dirs["sheet"] / f"{stem}.png",
                signed_target,
                signed_render,
                target_residual,
                recon_residual,
                error,
                weight,
                primitive_overlay,
                float(args.residual_clip),
            )
            _write_rgb_sheet(
                dirs["rgb_sheet"] / f"{stem}.png",
                anchor,
                target,
                target_rgb_hf,
                recon_rgb_hf,
                recon_rgb_hf_weighted,
                rgb_error_overlay,
            )
        np.savez_compressed(dirs["primitives"] / f"{stem}.npz", **primitives, losses=np.asarray(losses, dtype=np.float32))
        if bool(args.save_pt):
            torch.save(model.state_dict(), dirs["pt"] / f"{stem}.pt")

        row = {
            "index": float(index),
            "num_primitives": float(args.num_gaussians),
            "loss_start": float(losses[0]) if losses else float("nan"),
            "loss_final": float(losses[-1]) if losses else float("nan"),
            "target_energy": _weighted_energy(target_residual, weight),
            "recon_energy": _weighted_energy(recon_residual, weight),
            "l1": _weighted_l1(recon_residual, target_residual, weight),
            "corr_abs": _pearson_abs(recon_residual, target_residual, weight),
            "rgb_l1": rgb_l1,
            "weight_mean": float(weight.mean()),
        }
        rows.append(row)
        frames.append(
            {
                "stem": stem,
                "target": str(target_path),
                "anchor": str(anchor_path),
                "mask": str(mask_path),
                "num_primitives": int(args.num_gaussians),
                "segment_init": bool(args.segment_init and not args.init_random),
                "num_segments": int(len(set(int(x) for x in primitives.get("segment_id", np.asarray([], dtype=np.int32)).tolist() if int(x) >= 0))),
                "num_segment_pairs": int(primitives.get("pair_edges", np.zeros((0, 2), dtype=np.int64)).shape[0]),
                "loss_final": row["loss_final"],
            }
        )
        del model
        torch.cuda.empty_cache()

    summary = {
        "external_repo_root": str(repo_root),
        "target_dir": str(target_dir),
        "anchor_dir": str(anchor_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
        "match_policy": args.match_policy,
        "model": args.model,
        "fit_target_mode": args.fit_target_mode,
        "rgb_loss_weight_mode": args.rgb_loss_weight_mode,
        "num_frames": len(rows),
        "num_gaussians": int(args.num_gaussians),
        "iterations": int(args.iterations),
        "loss_final_mean": _mean(rows, "loss_final"),
        "rgb_l1_mean": _mean(rows, "rgb_l1"),
        "l1_mean": _mean(rows, "l1"),
        "corr_abs_mean": _mean(rows, "corr_abs"),
        "target_energy_mean": _mean(rows, "target_energy"),
        "recon_energy_mean": _mean(rows, "recon_energy"),
        "weight_mean": _mean(rows, "weight_mean"),
        "frames": frames,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "num_primitives",
                "loss_start",
                "loss_final",
                "target_energy",
                "recon_energy",
                "l1",
                "corr_abs",
                "rgb_l1",
                "weight_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))
    print(f"[gaussianimage-hf-v0] summary: {output_dir / 'summary.json'}")
    print(f"[gaussianimage-hf-v0] inspect: {dirs['sheet']}")


if __name__ == "__main__":
    main()
