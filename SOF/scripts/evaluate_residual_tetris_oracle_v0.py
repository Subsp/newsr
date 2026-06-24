#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spray_2dgs_posterior_to_gaussian_layer_v0 import (  # noqa: E402
    ViewPrimitiveSet,
    _camera_basis,
    _cluster_camera_angle,
    _cluster_to_result,
    _greedy_clusters,
    _load_base_vertices,
    _load_cameras,
    _load_view_primitives,
    _project_point,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class EvalObs:
    view_slot: int
    primitive_index: int
    view_index: int
    stem: str
    split: str
    center: np.ndarray
    theta: float
    long_px: float
    short_px: float
    support: np.ndarray
    responsibility: np.ndarray
    core_weight: np.ndarray
    target_hf: np.ndarray
    q_parent: np.ndarray
    roi: Tuple[int, int, int, int]


@dataclass
class Candidate:
    piece: str
    scale: float
    orientation_deg: float
    phase: int

    @property
    def name(self) -> str:
        sign = "p" if self.phase >= 0 else "n"
        angle = f"{self.orientation_deg:+.0f}".replace("+", "p").replace("-", "m")
        scale = f"{self.scale:.2f}".replace(".", "p")
        return f"{self.piece}_s{scale}_a{angle}_{sign}"


def _parse_csv(value: str, cast=float) -> List:
    out = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Residual Tetris V0 oracle. It separates signed basis capacity, "
            "normalized donor support compatibility, and physical delivery capacity "
            "for 2DGS-derived high-frequency residual cells."
        )
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--primitive_dir", required=True)
    parser.add_argument("--base_render_dir", required=True)
    parser.add_argument("--sr_dir", required=True)
    parser.add_argument("--weight_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--q_parent_dir", default="")
    parser.add_argument("--carrier_rgb_dir", default="")
    parser.add_argument("--carrier_render_dir", default="")
    parser.add_argument("--match_policy", choices=["stem", "order_if_needed", "order"], default="order_if_needed")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_primitives_per_view", type=int, default=32768)
    parser.add_argument("--max_clusters", type=int, default=256)
    parser.add_argument("--min_weight", type=float, default=0.02)
    parser.add_argument("--min_q", type=float, default=0.01)
    parser.add_argument("--min_primitive_opacity", type=float, default=0.0)
    parser.add_argument("--min_cluster_views", type=int, default=3)
    parser.add_argument("--min_fit_views", type=int, default=1)
    parser.add_argument("--min_selection_views", type=int, default=1)
    parser.add_argument("--min_test_views", type=int, default=1)
    parser.add_argument("--min_camera_angle_deg", type=float, default=1.5)
    parser.add_argument("--min_target_energy", type=float, default=1e-5)
    parser.add_argument("--min_active_area", type=int, default=24)
    parser.add_argument("--max_overlap_ratio", type=float, default=0.98)
    parser.add_argument("--max_connected_components", type=int, default=128)

    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--lowpass_kernel", type=int, default=21)
    parser.add_argument("--responsibility_bg_tau", type=float, default=0.25)
    parser.add_argument("--core_weight_threshold", type=float, default=0.015)
    parser.add_argument("--support_threshold", type=float, default=0.03)
    parser.add_argument("--tolerance_radius", type=int, default=3)
    parser.add_argument("--roi_pad_px", type=int, default=12)

    parser.add_argument("--piece_types", default="signed_single,dipole,dog,split")
    parser.add_argument("--piece_scales", default="0.75,1.0,1.5")
    parser.add_argument("--orientation_degs", default="-10,0,10")
    parser.add_argument("--phases", default="-1,1")
    parser.add_argument("--dipole_spacing", type=float, default=0.85)
    parser.add_argument("--dog_large_scale", type=float, default=2.35)
    parser.add_argument("--split_spacing", type=float, default=0.65)

    parser.add_argument("--beta_max", type=float, default=0.35)
    parser.add_argument("--beta_ridge", type=float, default=1e-4)
    parser.add_argument("--lambda_off", type=float, default=0.25)
    parser.add_argument("--lambda_lp", type=float, default=0.05)
    parser.add_argument("--lambda_dc", type=float, default=0.05)
    parser.add_argument("--q_percentile", type=float, default=95.0)
    parser.add_argument("--q_tau", type=float, default=0.03)
    parser.add_argument("--num_wrong_slots", type=int, default=5)
    parser.add_argument("--num_shuffled_q", type=int, default=5)
    parser.add_argument("--null_percentile", type=float, default=95.0)
    parser.add_argument("--selector_corr_threshold", type=float, default=0.20)
    parser.add_argument("--selector_ee_threshold", type=float, default=0.05)
    parser.add_argument("--selector_gain_min", type=float, default=0.0)
    parser.add_argument("--selector_leak_max", type=float, default=0.30)
    parser.add_argument("--selector_lp_drift_max", type=float, default=1.0)
    parser.add_argument("--selector_delivery_retention_min", type=float, default=0.50)
    parser.add_argument("--selector_beta_saturation_max", type=float, default=0.50)
    parser.add_argument("--selector_beta_max_fraction_max", type=float, default=0.90)
    parser.add_argument("--selector_null_margin_min", type=float, default=0.0)
    parser.add_argument("--test_good_corr_threshold", type=float, default=0.20)
    parser.add_argument("--test_good_ee_threshold", type=float, default=0.05)
    parser.add_argument("--test_good_leak_max", type=float, default=0.30)
    parser.add_argument("--test_good_lp_drift_max", type=float, default=1.0)

    parser.add_argument("--base_opacity_min", type=float, default=0.02)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--layer_search_radius_px", type=int, default=3)
    parser.add_argument("--footprint_sample_scale", type=float, default=1.25)
    parser.add_argument("--mode_depth_rel", type=float, default=0.018)
    parser.add_argument("--mode_depth_abs", type=float, default=0.006)
    parser.add_argument("--mode_position_radius", type=float, default=0.018)
    parser.add_argument("--min_mode_dominance", type=float, default=0.42)
    parser.add_argument("--max_mode_entropy", type=float, default=0.78)
    parser.add_argument("--association_radius_px", type=float, default=7.0)
    parser.add_argument("--association_cell_px", type=float, default=8.0)
    parser.add_argument("--association_color_weight", type=float, default=0.35)
    parser.add_argument("--association_shape_weight", type=float, default=0.25)
    parser.add_argument("--association_max_cost", type=float, default=3.25)
    parser.add_argument("--fit_error_tau", type=float, default=0.08)
    parser.add_argument("--fit_error_floor", type=float, default=0.15)
    parser.add_argument("--localization_sigma_px", type=float, default=1.4)
    parser.add_argument("--localization_footprint_beta", type=float, default=0.08)
    parser.add_argument("--surface_sigma", type=float, default=0.006)
    parser.add_argument("--tangent_prior_weight", type=float, default=0.002)
    parser.add_argument("--map_iterations", type=int, default=6)
    parser.add_argument("--map_huber_px", type=float, default=4.0)
    parser.add_argument("--map_damping", type=float, default=1e-4)
    parser.add_argument("--max_reproj_rms_px", type=float, default=3.8)
    parser.add_argument("--max_center_std", type=float, default=0.045)
    parser.add_argument("--max_hessian_cond", type=float, default=2.5e5)
    parser.add_argument("--screen_filter_sigma_px", type=float, default=0.45)
    parser.add_argument("--extract_sigma_px", type=float, default=0.35)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--scale_min", type=float, default=4e-4)
    parser.add_argument("--scale_max", type=float, default=9e-3)
    parser.add_argument("--normal_scale_ratio", type=float, default=0.20)
    parser.add_argument("--normal_scale_min", type=float, default=2.5e-4)
    parser.add_argument("--normal_scale_max", type=float, default=1.6e-3)

    parser.add_argument("--debug_limit", type=int, default=12)
    return parser.parse_args()


def _list_files(root: Path, exts: Iterable[str] = IMAGE_EXTS) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    ext_set = {x.lower() for x in exts}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ext_set)


def _image_lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve_path(paths: Sequence[Path], lookup: Dict[str, Path], stem: str, index: int, policy: str) -> Path:
    found = lookup.get(stem.lower())
    if found is not None:
        return found
    if policy == "stem":
        raise KeyError(f"No stem match for {stem}")
    if index >= len(paths):
        raise IndexError(f"No order match for view index {index}")
    return paths[index]


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _box_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return np.asarray(image, dtype=np.float32).copy()
    pad = k // 2
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)


def _highpass(image: np.ndarray, kernel: int) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) - _box_blur(image, kernel)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    radius = int(radius)
    if radius <= 0:
        return base.copy()
    out = np.zeros_like(base, dtype=bool)
    padded = np.pad(base, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + base.shape[0], dx : dx + base.shape[1]]
    return out


def _connected_components(mask: np.ndarray) -> int:
    active = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(active, dtype=bool)
    h, w = active.shape
    count = 0
    for y in range(h):
        for x in range(w):
            if not active[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and active[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return count


def _weighted_corr(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> Optional[float]:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim < np.asarray(a).ndim:
        while w.ndim < np.asarray(a).ndim:
            w = w[..., None]
    w = np.broadcast_to(w, np.asarray(a).shape).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 1e-8)
    if int(np.count_nonzero(keep)) < 16:
        return None
    x = x[keep]
    y = y[keep]
    w = w[keep]
    w = w / max(float(w.sum()), 1e-8)
    xm = float(np.sum(w * x))
    ym = float(np.sum(w * y))
    xv = x - xm
    yv = y - ym
    denom = math.sqrt(float(np.sum(w * xv * xv)) * float(np.sum(w * yv * yv)))
    if denom <= 1e-10:
        return None
    return float(np.sum(w * xv * yv) / denom)


def _energy(x: np.ndarray, weight: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    while w.ndim < arr.ndim:
        w = w[..., None]
    return float(np.sum(w * arr * arr))


def _safe_percentile(value: np.ndarray, percentile: float) -> float:
    arr = np.asarray(value, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return 0.0
    return float(np.percentile(arr, percentile))


def _concat_nonempty(values: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(v).reshape(-1) for v in values if np.asarray(v).size > 0]
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def _roi_bounds(center: np.ndarray, long_px: float, short_px: float, shape: Tuple[int, int], pad: int) -> Tuple[int, int, int, int]:
    h, w = shape
    r = int(math.ceil(max(float(long_px), float(short_px), 1.0) * 4.0 + float(pad)))
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    return x0, y0, x1, y1


def _coords(roi: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return xx.astype(np.float32), yy.astype(np.float32)


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    theta: float,
    long_px: float,
    short_px: float,
) -> np.ndarray:
    ct = math.cos(float(theta))
    st = math.sin(float(theta))
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    long_px = max(float(long_px), 0.35)
    short_px = max(float(short_px), 0.35)
    val = np.exp(-0.5 * ((u / long_px) ** 2 + (v / short_px) ** 2))
    return val.astype(np.float32)


def _piece(
    *,
    candidate: Candidate,
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    theta: float,
    long_px: float,
    short_px: float,
    args: argparse.Namespace,
) -> np.ndarray:
    theta = float(theta) + math.radians(float(candidate.orientation_deg))
    scale = float(candidate.scale)
    long_s = max(float(long_px) * scale, 0.35)
    short_s = max(float(short_px) * scale, 0.35)
    phase = 1.0 if int(candidate.phase) >= 0 else -1.0
    if candidate.piece == "signed_single":
        return phase * _gaussian_2d(xx, yy, center, theta, long_s, short_s)

    ct_b = -math.sin(theta)
    st_b = math.cos(theta)
    if candidate.piece == "dipole":
        d = float(args.dipole_spacing) * short_s
        c_pos = np.asarray([center[0] + ct_b * d, center[1] + st_b * d], dtype=np.float32)
        c_neg = np.asarray([center[0] - ct_b * d, center[1] - st_b * d], dtype=np.float32)
        return phase * (
            _gaussian_2d(xx, yy, c_pos, theta, long_s, short_s)
            - _gaussian_2d(xx, yy, c_neg, theta, long_s, short_s)
        )
    if candidate.piece == "dog":
        small = _gaussian_2d(xx, yy, center, theta, long_s, short_s)
        large = _gaussian_2d(
            xx,
            yy,
            center,
            theta,
            long_s * float(args.dog_large_scale),
            short_s * float(args.dog_large_scale),
        )
        kappa = float(np.sum(small) / max(float(np.sum(large)), 1e-8))
        return phase * (small - kappa * large)
    if candidate.piece == "split":
        d = float(args.split_spacing) * short_s
        c_pos = np.asarray([center[0] + ct_b * d, center[1] + st_b * d], dtype=np.float32)
        c_neg = np.asarray([center[0] - ct_b * d, center[1] - st_b * d], dtype=np.float32)
        return phase * (
            _gaussian_2d(xx, yy, c_pos, theta, long_s, max(short_s * 0.75, 0.35))
            - _gaussian_2d(xx, yy, c_neg, theta, long_s, max(short_s * 0.75, 0.35))
        )
    raise ValueError(f"Unknown piece type: {candidate.piece}")


def _project_frame(
    *,
    xyz: np.ndarray,
    normal: np.ndarray,
    source_view: ViewPrimitiveSet,
    source_primitive: int,
    target_view: ViewPrimitiveSet,
) -> Tuple[np.ndarray, float, float, float, bool]:
    src_theta = float(source_view.theta[source_primitive])
    _pos, cam_x, cam_y, _cam_z, fx, fy, _cw, _ch = _camera_basis(source_view.camera)
    t = math.cos(src_theta) * cam_x + math.sin(src_theta) * cam_y
    normal = normal.astype(np.float32)
    t = t - normal * float(np.dot(t, normal))
    t_norm = float(np.linalg.norm(t))
    if t_norm <= 1e-8:
        t = cam_x - normal * float(np.dot(cam_x, normal))
        t_norm = float(np.linalg.norm(t))
    t = (t / max(t_norm, 1e-8)).astype(np.float32)
    b = np.cross(normal, t)
    b = (b / max(float(np.linalg.norm(b)), 1e-8)).astype(np.float32)

    center_src, depth_src, ok_src = _project_point(source_view.camera, source_view.source_size, xyz)
    center_tgt, _depth_tgt, ok_tgt = _project_point(target_view.camera, target_view.source_size, xyz)
    if not ok_src or not ok_tgt:
        return center_tgt, 0.0, 0.0, 0.0, False
    focal = max((float(fx) + float(fy)) * 0.5, 1e-8)
    pixel_to_world = max(float(depth_src), 1e-8) / focal
    long_w = max(float(source_view.long_px[source_primitive]), 0.5) * pixel_to_world
    short_w = max(float(source_view.short_px[source_primitive]), 0.5) * pixel_to_world
    p_t, _, ok_t = _project_point(target_view.camera, target_view.source_size, xyz + t * long_w)
    p_b, _, ok_b = _project_point(target_view.camera, target_view.source_size, xyz + b * short_w)
    if not ok_t or not ok_b:
        return center_tgt, 0.0, 0.0, 0.0, False
    axis_t = p_t - center_tgt
    axis_b = p_b - center_tgt
    long_px = max(float(np.linalg.norm(axis_t)), 0.45)
    short_px = max(float(np.linalg.norm(axis_b)), 0.45)
    theta = math.atan2(float(axis_t[1]), float(axis_t[0]))
    return center_tgt.astype(np.float32), theta, long_px, short_px, True


def _split_obs(obs: Sequence[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    sorted_obs = sorted(obs, key=lambda item: item[0])
    out = {"fit": [], "selection": [], "test": []}
    names = ("fit", "selection", "test")
    for i, item in enumerate(sorted_obs):
        out[names[i % 3]].append(item)
    return out


def _make_spray_args(args: argparse.Namespace) -> SimpleNamespace:
    values = vars(args).copy()
    values.setdefault("color_mode", "base_anchor_additive")
    values.setdefault("color_gain", 0.32)
    values.setdefault("opacity_floor", 0.006)
    values.setdefault("opacity_scale", 0.055)
    values.setdefault("opacity_power", 0.70)
    values.setdefault("opacity_min", 0.004)
    values.setdefault("opacity_max", 0.075)
    return SimpleNamespace(**values)


def _candidate_grid(args: argparse.Namespace) -> List[Candidate]:
    pieces = [x for x in _parse_csv(args.piece_types, str) if x]
    scales = _parse_csv(args.piece_scales, float)
    angles = _parse_csv(args.orientation_degs, float)
    phases = _parse_csv(args.phases, int)
    return [
        Candidate(piece=piece, scale=float(scale), orientation_deg=float(angle), phase=int(phase))
        for piece in pieces
        for scale in scales
        for angle in angles
        for phase in phases
    ]


def _fit_beta(
    obs_list: Sequence[EvalObs],
    candidate: Candidate,
    args: argparse.Namespace,
    q_mode: str,
    beta_mode: str = "rgb",
    q_global_scale: Optional[float] = None,
    q_variant: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if beta_mode not in {"rgb", "luma"}:
        raise ValueError(beta_mode)
    channels = 1 if beta_mode == "luma" else 3
    numerator = np.zeros((channels,), dtype=np.float64)
    denominator = np.full((channels,), float(args.beta_ridge), dtype=np.float64)
    target_energy = 0.0
    fit_pixels = 0
    q_values = []
    for obs in obs_list:
        basis, target, fit_w, off_w, raw_basis, lp_basis = _basis_arrays(
            obs,
            candidate,
            args,
            q_mode,
            q_global_scale,
            q_variant=q_variant,
        )
        if beta_mode == "luma":
            target_use = np.sum(target * LUMA[None, None, :], axis=2, keepdims=True)
            basis_use = basis[..., None]
            raw_use = raw_basis[..., None]
            lp_use = lp_basis[..., None]
        else:
            target_use = target
            basis_use = np.repeat(basis[..., None], 3, axis=2)
            raw_use = np.repeat(raw_basis[..., None], 3, axis=2)
            lp_use = np.repeat(lp_basis[..., None], 3, axis=2)
        fw = fit_w[..., None]
        ow = off_w[..., None]
        numerator += np.sum(fw * basis_use * target_use, axis=(0, 1))
        denominator += np.sum(fw * basis_use * basis_use, axis=(0, 1))
        denominator += float(args.lambda_off) * np.sum(ow * raw_use * raw_use, axis=(0, 1))
        denominator += float(args.lambda_lp) * np.sum(fw * lp_use * lp_use, axis=(0, 1))
        dc = np.sum(fw * raw_use, axis=(0, 1)) / max(float(np.sum(fw)), 1e-8)
        denominator += float(args.lambda_dc) * dc * dc
        target_energy += float(np.sum(fw * target_use * target_use))
        fit_pixels += int(np.count_nonzero(fit_w > 1e-8))
        q_values.append(obs.q_parent[obs.core_weight > float(args.core_weight_threshold)])
    beta = numerator / np.maximum(denominator, 1e-8)
    beta = np.clip(beta, -float(args.beta_max), float(args.beta_max)).astype(np.float32)
    if beta_mode == "luma":
        beta_rgb = np.asarray([beta[0], beta[0], beta[0]], dtype=np.float32)
    else:
        beta_rgb = beta.astype(np.float32)
    q_concat = _concat_nonempty(q_values)
    stats = {
        "fit_pixels": float(fit_pixels),
        "target_energy": float(target_energy),
        "beta_abs_max": float(np.max(np.abs(beta_rgb))),
        "beta_l2": float(np.linalg.norm(beta_rgb)),
        "beta_saturation": float(np.mean(np.abs(beta_rgb) >= float(args.beta_max) * 0.98)),
        "beta_max_fraction": float(np.max(np.abs(beta_rgb)) / max(float(args.beta_max), 1e-8)),
        "beta_margin": float(1.0 - np.max(np.abs(beta_rgb)) / max(float(args.beta_max), 1e-8)),
        "q_peak": float(np.max(q_concat)) if q_concat.size else 0.0,
        "q_mean_on_core": float(np.mean(q_concat)) if q_concat.size else 0.0,
        "q_core_coverage": float(np.mean(q_concat > float(args.q_tau))) if q_concat.size else 0.0,
    }
    return beta_rgb, stats


def _basis_arrays(
    obs: EvalObs,
    candidate: Candidate,
    args: argparse.Namespace,
    q_mode: str,
    q_global_scale: Optional[float] = None,
    q_variant: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = obs.roi
    xx, yy = _coords(obs.roi)
    phi = _piece(
        candidate=candidate,
        xx=xx,
        yy=yy,
        center=obs.center,
        theta=obs.theta,
        long_px=obs.long_px,
        short_px=obs.short_px,
        args=args,
    )
    if q_mode == "A":
        q = np.clip(obs.support, 0.0, 1.0)
    elif q_mode == "B_shape":
        scale = _safe_percentile(obs.q_parent[obs.support > float(args.support_threshold)], float(args.q_percentile))
        q = np.clip(obs.q_parent / max(scale, 1e-8), 0.0, 1.0)
    elif q_mode == "B_relative":
        q = np.clip(obs.q_parent / max(float(q_global_scale or 0.0), 1e-8), 0.0, 1.0)
    elif q_mode == "C":
        q = np.clip(obs.q_parent, 0.0, 1.0)
    elif q_mode == "wrong_slot":
        n_slots = max(int(getattr(args, "num_wrong_slots", 1)), 1)
        shift = max(float(obs.short_px) * 2.0, 2.0)
        # Use nearby equal-radius negative slots instead of one hand-picked
        # offset; this gives a small matched null distribution per cluster.
        angle = float(obs.theta) + math.pi * 0.5 + 2.0 * math.pi * (int(q_variant) % n_slots) / float(n_slots)
        shifted = obs.center + np.asarray([math.cos(angle) * shift, math.sin(angle) * shift], dtype=np.float32)
        phi = _piece(
            candidate=candidate,
            xx=xx,
            yy=yy,
            center=shifted,
            theta=obs.theta,
            long_px=obs.long_px,
            short_px=obs.short_px,
            args=args,
        )
        q = np.clip(obs.q_parent, 0.0, 1.0)
    elif q_mode == "shuffled_q":
        n_shuffle = max(int(getattr(args, "num_shuffled_q", 1)), 1)
        base_q = np.clip(obs.q_parent, 0.0, 1.0)
        dx = max(1, int(round(base_q.shape[1] * float((int(q_variant) % n_shuffle) + 1) / float(n_shuffle + 1))))
        dy = max(1, int(round(base_q.shape[0] * float(((int(q_variant) * 7) % n_shuffle) + 1) / float(n_shuffle + 1))))
        q = np.roll(np.roll(base_q, shift=dx, axis=1), shift=dy, axis=0)
    else:
        raise ValueError(f"Unknown q_mode: {q_mode}")
    raw_basis = q * phi
    basis_hp = _highpass(raw_basis, int(args.highpass_kernel))
    lp_basis = _box_blur(raw_basis, int(args.lowpass_kernel))
    target = obs.target_hf
    fit_w = np.clip(obs.core_weight, 0.0, 1.0)
    tolerance = _dilate(fit_w > float(args.core_weight_threshold), int(args.tolerance_radius))
    off_w = np.where(tolerance, 0.0, np.clip(obs.support, 0.0, 1.0)).astype(np.float32)
    return basis_hp.astype(np.float32), target.astype(np.float32), fit_w.astype(np.float32), off_w, raw_basis.astype(np.float32), lp_basis.astype(np.float32)


def _eval_beta(
    obs_list: Sequence[EvalObs],
    candidate: Candidate,
    beta: np.ndarray,
    args: argparse.Namespace,
    q_mode: str,
    q_global_scale: Optional[float] = None,
    q_variant: int = 0,
) -> Dict[str, float]:
    target_energy = 0.0
    residual_energy = 0.0
    pred_energy = 0.0
    off_energy = 0.0
    lp_energy = 0.0
    signed_pred: List[np.ndarray] = []
    signed_target: List[np.ndarray] = []
    signed_weight: List[np.ndarray] = []
    fit_pixels = 0
    active_area = 0
    target_norm = 0.0
    q_vals = []
    for obs in obs_list:
        basis, target, fit_w, off_w, raw_basis, lp_basis = _basis_arrays(
            obs,
            candidate,
            args,
            q_mode,
            q_global_scale,
            q_variant=q_variant,
        )
        pred_hp = basis[..., None] * beta[None, None, :]
        pred_raw = raw_basis[..., None] * beta[None, None, :]
        pred_lp = lp_basis[..., None] * beta[None, None, :]
        fw = fit_w[..., None]
        ow = off_w[..., None]
        target_energy += float(np.sum(fw * target * target))
        residual_energy += float(np.sum(fw * (target - pred_hp) * (target - pred_hp)))
        pred_energy += float(np.sum(fw * pred_hp * pred_hp))
        off_energy += float(np.sum(ow * pred_raw * pred_raw))
        lp_energy += float(np.sum(fw * pred_lp * pred_lp))
        fit_pixels += int(np.count_nonzero(fit_w > 1e-8))
        active_area += int(np.count_nonzero(fit_w > float(args.core_weight_threshold)))
        target_norm += float(np.sum(fw * np.abs(target)))
        signed_pred.append(np.sum(pred_hp * LUMA[None, None, :], axis=2))
        signed_target.append(np.sum(target * LUMA[None, None, :], axis=2))
        signed_weight.append(fit_w)
        q_vals.append(obs.q_parent[fit_w > float(args.core_weight_threshold)])
    if not obs_list:
        return _empty_metrics()
    if target_energy <= 1e-10:
        ee = float("-inf")
    else:
        ee = 1.0 - residual_energy / max(target_energy, 1e-10)
    pred_flat = np.concatenate([x.reshape(-1) for x in signed_pred], axis=0)
    target_flat = np.concatenate([x.reshape(-1) for x in signed_target], axis=0)
    weight_flat = np.concatenate([x.reshape(-1) for x in signed_weight], axis=0)
    corr = _weighted_corr(pred_flat, target_flat, weight_flat)
    amp = math.sqrt(pred_energy / max(target_energy, 1e-10)) if target_energy > 0 else float("nan")
    leak = off_energy / max(pred_energy, 1e-10)
    precision = pred_energy / max(pred_energy + off_energy, 1e-10)
    lp = lp_energy / max(target_energy, 1e-10)
    q_concat = _concat_nonempty(q_vals)
    return {
        "views": float(len(obs_list)),
        "fit_pixels": float(fit_pixels),
        "active_area": float(active_area),
        "corr": float("nan") if corr is None else float(corr),
        "explained_energy": float(ee),
        "gain": float(target_energy - residual_energy),
        "target_energy": float(target_energy),
        "pred_energy": float(pred_energy),
        "amplitude_ratio": float(amp),
        "leak_ratio": float(leak),
        "target_precision": float(precision),
        "lp_drift": float(lp),
        "target_norm": float(target_norm),
        "q_peak": float(np.max(q_concat)) if q_concat.size else 0.0,
        "q_mean_on_core": float(np.mean(q_concat)) if q_concat.size else 0.0,
        "q_core_coverage": float(np.mean(q_concat > float(args.q_tau))) if q_concat.size else 0.0,
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "views": 0.0,
        "fit_pixels": 0.0,
        "active_area": 0.0,
        "corr": float("nan"),
        "explained_energy": float("-inf"),
        "gain": float("-inf"),
        "target_energy": 0.0,
        "pred_energy": 0.0,
        "amplitude_ratio": float("nan"),
        "leak_ratio": float("inf"),
        "target_precision": 0.0,
        "lp_drift": float("inf"),
        "target_norm": 0.0,
        "q_peak": 0.0,
        "q_mean_on_core": 0.0,
        "q_core_coverage": 0.0,
    }


def _prefix(prefix: str, data: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in data.items()}


def _score(metrics: Dict[str, float]) -> float:
    corr = float(metrics.get("corr", 0.0))
    if not math.isfinite(corr):
        corr = -1.0
    ee = float(metrics.get("explained_energy", -1.0))
    leak = float(metrics.get("leak_ratio", 1.0))
    lp = float(metrics.get("lp_drift", 1.0))
    amp = float(metrics.get("amplitude_ratio", 0.0))
    amp_penalty = abs(math.log(max(amp, 1e-4))) if math.isfinite(amp) else 4.0
    return corr + 0.35 * ee - 0.20 * min(leak, 10.0) - 0.10 * min(lp, 10.0) - 0.03 * amp_penalty


def _finite_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _metric_score_from_row(row: Dict[str, object], prefix: str) -> float:
    return _score(
        {
            "corr": _finite_float(row.get(f"{prefix}_corr"), -1.0),
            "explained_energy": _finite_float(row.get(f"{prefix}_explained_energy"), -1.0),
            "leak_ratio": _finite_float(row.get(f"{prefix}_leak_ratio"), 1.0),
            "lp_drift": _finite_float(row.get(f"{prefix}_lp_drift"), 1.0),
            "amplitude_ratio": _finite_float(row.get(f"{prefix}_amplitude_ratio"), 0.0),
        }
    )


def _quality_pass(row: Dict[str, object], prefix: str, args: argparse.Namespace, *, use_test_thresholds: bool = False) -> bool:
    corr_thr = float(args.test_good_corr_threshold if use_test_thresholds else args.selector_corr_threshold)
    ee_thr = float(args.test_good_ee_threshold if use_test_thresholds else args.selector_ee_threshold)
    leak_max = float(args.test_good_leak_max if use_test_thresholds else args.selector_leak_max)
    lp_max = float(args.test_good_lp_drift_max if use_test_thresholds else args.selector_lp_drift_max)
    return (
        _finite_float(row.get(f"{prefix}_corr"), -1.0) > corr_thr
        and _finite_float(row.get(f"{prefix}_explained_energy"), -1.0) > ee_thr
        and _finite_float(row.get(f"{prefix}_gain"), -1.0) > float(args.selector_gain_min)
        and _finite_float(row.get(f"{prefix}_leak_ratio"), 1e9) < leak_max
        and _finite_float(row.get(f"{prefix}_lp_drift"), 1e9) < lp_max
    )


def _null_summary(prefix: str, real_metrics: Dict[str, float], null_metrics: Sequence[Dict[str, float]], args: argparse.Namespace) -> Dict[str, float]:
    if not null_metrics:
        return {
            f"{prefix}_null_count": 0.0,
            f"{prefix}_null_score_p95": float("nan"),
            f"{prefix}_null_corr_p95": float("nan"),
            f"{prefix}_null_EE_p95": float("nan"),
            f"{prefix}_null_gain_p95": float("nan"),
            f"{prefix}_margin_score": float("nan"),
            f"{prefix}_margin_corr": float("nan"),
            f"{prefix}_margin_EE": float("nan"),
            f"{prefix}_margin_gain": float("nan"),
            f"{prefix}_empirical_tail": float("nan"),
        }
    percentile = float(args.null_percentile)
    null_scores = np.asarray([_score(m) for m in null_metrics], dtype=np.float32)
    null_corr = np.asarray([_finite_float(m.get("corr"), -1.0) for m in null_metrics], dtype=np.float32)
    null_ee = np.asarray([_finite_float(m.get("explained_energy"), -1.0) for m in null_metrics], dtype=np.float32)
    null_gain = np.asarray([_finite_float(m.get("gain"), -1e9) for m in null_metrics], dtype=np.float32)
    real_score = _score(real_metrics)
    real_corr = _finite_float(real_metrics.get("corr"), -1.0)
    real_ee = _finite_float(real_metrics.get("explained_energy"), -1.0)
    real_gain = _finite_float(real_metrics.get("gain"), -1e9)
    score_p = float(np.percentile(null_scores, percentile))
    corr_p = float(np.percentile(null_corr, percentile))
    ee_p = float(np.percentile(null_ee, percentile))
    gain_p = float(np.percentile(null_gain, percentile))
    tail = float((1 + int(np.count_nonzero(null_scores >= real_score))) / float(len(null_scores) + 1))
    return {
        f"{prefix}_null_count": float(len(null_metrics)),
        f"{prefix}_null_score_p95": score_p,
        f"{prefix}_null_corr_p95": corr_p,
        f"{prefix}_null_EE_p95": ee_p,
        f"{prefix}_null_gain_p95": gain_p,
        f"{prefix}_margin_score": float(real_score - score_p),
        f"{prefix}_margin_corr": float(real_corr - corr_p),
        f"{prefix}_margin_EE": float(real_ee - ee_p),
        f"{prefix}_margin_gain": float(real_gain - gain_p),
        f"{prefix}_empirical_tail": tail,
    }


def _eval_null_family(
    *,
    fit: Sequence[EvalObs],
    eval_obs: Sequence[EvalObs],
    cand: Candidate,
    args: argparse.Namespace,
    q_mode: str,
    count: int,
    q_global_scale: Optional[float] = None,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for variant in range(max(int(count), 0)):
        beta, _ = _fit_beta(
            fit,
            cand,
            args,
            q_mode=q_mode,
            beta_mode="rgb",
            q_global_scale=q_global_scale,
            q_variant=variant,
        )
        out.append(
            _eval_beta(
                eval_obs,
                cand,
                beta,
                args,
                q_mode=q_mode,
                q_global_scale=q_global_scale,
                q_variant=variant,
            )
        )
    return out


def _apply_selector_labels(row: Dict[str, object], args: argparse.Namespace) -> None:
    row["test_positive"] = bool(_finite_float(row.get("C_test_gain"), -1.0) > 0.0)
    row["test_good_strict"] = bool(_quality_pass(row, "C_test", args, use_test_thresholds=True))
    row["pass_A_selection"] = bool(_quality_pass(row, "A_selection", args))
    row["pass_B_selection"] = bool(
        _quality_pass(row, "B_shape_selection", args)
        and _quality_pass(row, "B_relative_selection", args)
    )
    row["pass_C_selection"] = bool(
        _quality_pass(row, "C_selection", args)
        and _finite_float(row.get("delivery_retention_selection"), -1.0) > float(args.selector_delivery_retention_min)
    )
    row["pass_controls_selection"] = bool(
        _finite_float(row.get("slot_selection_margin_score"), -1.0) > float(args.selector_null_margin_min)
        and _finite_float(row.get("q_selection_margin_score"), -1.0) > float(args.selector_null_margin_min)
    )
    beta_sat = _finite_float(row.get("C_fit_beta_beta_saturation"), 1.0)
    beta_frac = _finite_float(row.get("C_fit_beta_beta_max_fraction"), 1.0)
    row["pass_capacity"] = bool(
        beta_sat < float(args.selector_beta_saturation_max)
        and beta_frac < float(args.selector_beta_max_fraction_max)
    )
    ordered = [
        ("reject_A_basis", bool(row["pass_A_selection"])),
        ("reject_B_support", bool(row["pass_B_selection"])),
        ("reject_C_delivery", bool(row["pass_C_selection"])),
        ("reject_matched_null", bool(row["pass_controls_selection"])),
        ("reject_beta_saturation", bool(row["pass_capacity"])),
    ]
    reject_reason = "pass"
    for reason, ok in ordered:
        if not ok:
            reject_reason = reason
            break
    row["selector_reject_reason"] = reject_reason
    row["selector_core_pass"] = reject_reason == "pass"
    row["selector_capacity_limited_pass"] = bool(
        row["pass_A_selection"]
        and row["pass_B_selection"]
        and row["pass_C_selection"]
        and row["pass_controls_selection"]
        and not row["pass_capacity"]
    )
    margins = [
        _finite_float(row.get("A_selection_corr"), -1.0) - float(args.selector_corr_threshold),
        _finite_float(row.get("A_selection_explained_energy"), -1.0) - float(args.selector_ee_threshold),
        _finite_float(row.get("B_shape_selection_explained_energy"), -1.0) - float(args.selector_ee_threshold),
        _finite_float(row.get("B_relative_selection_explained_energy"), -1.0) - float(args.selector_ee_threshold),
        _finite_float(row.get("C_selection_corr"), -1.0) - float(args.selector_corr_threshold),
        _finite_float(row.get("C_selection_explained_energy"), -1.0) - float(args.selector_ee_threshold),
        _finite_float(row.get("slot_selection_margin_score"), -1.0) - float(args.selector_null_margin_min),
        _finite_float(row.get("q_selection_margin_score"), -1.0) - float(args.selector_null_margin_min),
        float(args.selector_beta_max_fraction_max) - _finite_float(row.get("C_fit_beta_beta_max_fraction"), 1.0),
    ]
    row["selector_score"] = float(min(margins))


def _build_eval_obs(
    *,
    cluster_obs: Sequence[Tuple[int, int]],
    result,
    views: Sequence[ViewPrimitiveSet],
    base_paths: Sequence[Path],
    base_lookup: Dict[str, Path],
    sr_paths: Sequence[Path],
    sr_lookup: Dict[str, Path],
    weight_paths: Sequence[Path],
    weight_lookup: Dict[str, Path],
    q_paths: Sequence[Path],
    q_lookup: Dict[str, Path],
    args: argparse.Namespace,
) -> Tuple[List[EvalObs], Dict[str, float]]:
    split_by_obs = _split_obs(cluster_obs)
    split_map = {item: split for split, items in split_by_obs.items() for item in items}
    source_slot, source_i = cluster_obs[0]
    source_view = views[source_slot]
    out: List[EvalObs] = []
    stats = {
        "support_view_count": float(len(cluster_obs)),
        "camera_baseline_diversity": float(_cluster_camera_angle(views, cluster_obs, result.xyz.astype(np.float32))),
        "target_energy": 0.0,
        "active_area": 0.0,
        "overlap_ratio": 0.0,
        "responsibility_entropy": 0.0,
        "connected_component_count": 0.0,
    }
    for view_slot, prim_i in cluster_obs:
        view = views[view_slot]
        center, theta, long_px, short_px, ok = _project_frame(
            xyz=result.xyz.astype(np.float32),
            normal=result.normal.astype(np.float32),
            source_view=source_view,
            source_primitive=source_i,
            target_view=view,
        )
        if not ok:
            continue
        base_path = _resolve_path(base_paths, base_lookup, view.stem, view.view_index, args.match_policy)
        sr_path = _resolve_path(sr_paths, sr_lookup, view.stem, view.view_index, args.match_policy)
        weight_path = _resolve_path(weight_paths, weight_lookup, view.stem, view.view_index, args.match_policy)
        base_img = _load_rgb(base_path)
        size = (base_img.shape[1], base_img.shape[0])
        sr_img = _load_rgb(sr_path, size=size)
        weight_img = _load_gray(weight_path, size=size)
        if q_paths:
            q_path = _resolve_path(q_paths, q_lookup, view.stem, view.view_index, args.match_policy)
            q_img = _load_gray(q_path, size=size)
        else:
            q_img = weight_img.copy()
        target_delta = _highpass(sr_img - base_img, int(args.highpass_kernel))
        roi = _roi_bounds(center, long_px, short_px, (base_img.shape[0], base_img.shape[1]), int(args.roi_pad_px))
        x0, y0, x1, y1 = roi
        xx, yy = _coords(roi)
        support = _gaussian_2d(xx, yy, center, theta, long_px, short_px)
        responsibility = support / (float(args.responsibility_bg_tau) + support)
        weight_roi = np.clip(weight_img[y0:y1, x0:x1], 0.0, 1.0)
        core_weight = np.clip(responsibility * weight_roi, 0.0, 1.0).astype(np.float32)
        # Keep the signed HF target in raw residual units. The cluster
        # responsibility and effective-HF confidence are used as fit weights,
        # otherwise the target would be attenuated twice.
        target_roi = target_delta[y0:y1, x0:x1].astype(np.float32)
        q_roi = np.clip(q_img[y0:y1, x0:x1], 0.0, 1.0).astype(np.float32)
        obs = EvalObs(
            view_slot=view_slot,
            primitive_index=prim_i,
            view_index=view.view_index,
            stem=view.stem,
            split=split_map[(view_slot, prim_i)],
            center=center,
            theta=float(theta),
            long_px=float(long_px),
            short_px=float(short_px),
            support=support.astype(np.float32),
            responsibility=responsibility.astype(np.float32),
            core_weight=core_weight,
            target_hf=target_roi,
            q_parent=q_roi,
            roi=roi,
        )
        out.append(obs)
        active = core_weight > float(args.core_weight_threshold)
        stats["target_energy"] += _energy(target_roi, core_weight)
        stats["active_area"] += float(np.count_nonzero(active))
        stats["overlap_ratio"] += float(np.mean(responsibility > 0.75))
        p = np.clip(responsibility, 1e-6, 1.0)
        stats["responsibility_entropy"] += float(np.mean(-(p * np.log(p) + (1.0 - p) * np.log(np.clip(1.0 - p, 1e-6, 1.0)))))
        stats["connected_component_count"] += float(_connected_components(active))
    denom = max(float(len(out)), 1.0)
    for key in ("overlap_ratio", "responsibility_entropy", "connected_component_count"):
        stats[key] = float(stats[key] / denom)
    return out, stats


def _eligible(obs: Sequence[EvalObs], stats: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    split_counts = {name: sum(1 for item in obs if item.split == name) for name in ("fit", "selection", "test")}
    if len(obs) < int(args.min_cluster_views):
        reasons.append("support_views")
    if split_counts["fit"] < int(args.min_fit_views):
        reasons.append("fit_views")
    if split_counts["selection"] < int(args.min_selection_views):
        reasons.append("selection_views")
    if split_counts["test"] < int(args.min_test_views):
        reasons.append("test_views")
    if float(stats["target_energy"]) < float(args.min_target_energy):
        reasons.append("target_energy")
    if float(stats["active_area"]) < float(args.min_active_area):
        reasons.append("active_area")
    if float(stats["overlap_ratio"]) > float(args.max_overlap_ratio):
        reasons.append("overlap_ratio")
    if float(stats["connected_component_count"]) > float(args.max_connected_components):
        reasons.append("fragmented_support")
    return not reasons, reasons


def _eval_cluster(
    obs: Sequence[EvalObs],
    candidates: Sequence[Candidate],
    args: argparse.Namespace,
) -> Dict[str, object]:
    fit = [o for o in obs if o.split == "fit"]
    selection = [o for o in obs if o.split == "selection"]
    test = [o for o in obs if o.split == "test"]
    best: Optional[Tuple[float, Candidate, np.ndarray, Dict[str, float], Dict[str, float], Dict[str, float]]] = None
    all_a: List[Dict[str, object]] = []
    for cand in candidates:
        beta, fit_stats = _fit_beta(fit, cand, args, q_mode="A", beta_mode="rgb")
        sel_metrics = _eval_beta(selection, cand, beta, args, q_mode="A")
        test_metrics = _eval_beta(test, cand, beta, args, q_mode="A")
        row = {"candidate": cand.name, "beta": [float(x) for x in beta], **_prefix("selection", sel_metrics)}
        all_a.append(row)
        score = _score(sel_metrics)
        if best is None or score > best[0]:
            best = (score, cand, beta, fit_stats, sel_metrics, test_metrics)
    if best is None:
        return {"valid": False, "reason": "no_candidate"}
    _score_a, cand, beta_a, fit_stats_a, sel_a, test_a = best
    q_values = []
    for o in fit:
        q_values.append(o.q_parent[o.core_weight > float(args.core_weight_threshold)])
    q_concat = _concat_nonempty(q_values)
    q_global = _safe_percentile(q_concat, float(args.q_percentile))

    beta_luma, _fit_luma = _fit_beta(fit, cand, args, q_mode="A", beta_mode="luma")
    per_view_gains = []
    for o in obs:
        beta_v, _ = _fit_beta([o], cand, args, q_mode="A", beta_mode="rgb")
        per_view_gains.append(_eval_beta([o], cand, beta_v, args, q_mode="A")["gain"])

    beta_b, _ = _fit_beta(fit, cand, args, q_mode="B_shape", beta_mode="rgb")
    beta_br, _ = _fit_beta(fit, cand, args, q_mode="B_relative", beta_mode="rgb", q_global_scale=q_global)
    beta_c, fit_stats_c = _fit_beta(fit, cand, args, q_mode="C", beta_mode="rgb")
    slot_selection_null = _eval_null_family(
        fit=fit,
        eval_obs=selection,
        cand=cand,
        args=args,
        q_mode="wrong_slot",
        count=int(args.num_wrong_slots),
    )
    slot_test_null = _eval_null_family(
        fit=fit,
        eval_obs=test,
        cand=cand,
        args=args,
        q_mode="wrong_slot",
        count=int(args.num_wrong_slots),
    )
    q_selection_null = _eval_null_family(
        fit=fit,
        eval_obs=selection,
        cand=cand,
        args=args,
        q_mode="shuffled_q",
        count=int(args.num_shuffled_q),
    )
    q_test_null = _eval_null_family(
        fit=fit,
        eval_obs=test,
        cand=cand,
        args=args,
        q_mode="shuffled_q",
        count=int(args.num_shuffled_q),
    )
    beta_wrong, _ = _fit_beta(fit, cand, args, q_mode="wrong_slot", beta_mode="rgb", q_variant=0)
    beta_shuf, _ = _fit_beta(fit, cand, args, q_mode="shuffled_q", beta_mode="rgb", q_variant=0)
    c_selection = _eval_beta(selection, cand, beta_c, args, q_mode="C")
    c_test = _eval_beta(test, cand, beta_c, args, q_mode="C")

    result: Dict[str, object] = {
        "valid": True,
        "best_candidate": cand.name,
        "best_piece": cand.piece,
        "best_scale": float(cand.scale),
        "best_orientation_deg": float(cand.orientation_deg),
        "best_phase": int(cand.phase),
        "q_global_scale": float(q_global),
        "A_beta": [float(x) for x in beta_a],
        "A_luma_beta": [float(x) for x in beta_luma],
        "C_beta": [float(x) for x in beta_c],
        "per_view_beta_gain_mean": float(np.mean(per_view_gains)) if per_view_gains else float("nan"),
        "per_view_beta_gain_positive_ratio": float(np.mean(np.asarray(per_view_gains) > 0.0)) if per_view_gains else float("nan"),
        "A_grid_top": sorted(all_a, key=lambda r: float(r.get("selection_gain", -1e30)), reverse=True)[:8],
    }
    result.update(_prefix("A_fit", _eval_beta(fit, cand, beta_a, args, q_mode="A")))
    result.update(_prefix("A_selection", sel_a))
    result.update(_prefix("A_test", test_a))
    result.update(_prefix("A_luma_selection", _eval_beta(selection, cand, beta_luma, args, q_mode="A")))
    result.update(_prefix("A_luma_test", _eval_beta(test, cand, beta_luma, args, q_mode="A")))
    result.update(_prefix("B_shape_selection", _eval_beta(selection, cand, beta_b, args, q_mode="B_shape")))
    result.update(_prefix("B_shape_test", _eval_beta(test, cand, beta_b, args, q_mode="B_shape")))
    result.update(_prefix("B_relative_selection", _eval_beta(selection, cand, beta_br, args, q_mode="B_relative", q_global_scale=q_global)))
    result.update(_prefix("B_relative_test", _eval_beta(test, cand, beta_br, args, q_mode="B_relative", q_global_scale=q_global)))
    result.update(_prefix("C_fit", _eval_beta(fit, cand, beta_c, args, q_mode="C")))
    result.update(_prefix("C_selection", c_selection))
    result.update(_prefix("C_test", c_test))
    result.update(_prefix("wrong_slot_selection", _eval_beta(selection, cand, beta_wrong, args, q_mode="wrong_slot", q_variant=0)))
    result.update(_prefix("wrong_slot_test", _eval_beta(test, cand, beta_wrong, args, q_mode="wrong_slot", q_variant=0)))
    result.update(_prefix("shuffled_q_selection", _eval_beta(selection, cand, beta_shuf, args, q_mode="shuffled_q", q_variant=0)))
    result.update(_prefix("shuffled_q_test", _eval_beta(test, cand, beta_shuf, args, q_mode="shuffled_q", q_variant=0)))
    result.update(_prefix("C_fit_beta", fit_stats_c))
    result.update(_null_summary("slot_selection", c_selection, slot_selection_null, args))
    result.update(_null_summary("slot_test", c_test, slot_test_null, args))
    result.update(_null_summary("q_selection", c_selection, q_selection_null, args))
    result.update(_null_summary("q_test", c_test, q_test_null, args))
    ga_sel = max(float(result.get("A_selection_gain", float("-inf"))), 0.0)
    gc_sel = float(result.get("C_selection_gain", float("-inf")))
    result["delivery_retention_selection"] = float(gc_sel / ga_sel) if ga_sel > 1e-10 else float("nan")
    ga = max(float(result.get("A_test_gain", float("-inf"))), 0.0)
    gc = float(result.get("C_test_gain", float("-inf")))
    result["delivery_retention_test"] = float(gc / ga) if ga > 1e-10 else float("nan")
    result["delivery_retention"] = result["delivery_retention_test"]
    result["Delta_EE_wrong_slot_selection"] = float(result["C_selection_explained_energy"] - result["wrong_slot_selection_explained_energy"])
    result["Delta_corr_wrong_slot_selection"] = float(result["C_selection_corr"] - result["wrong_slot_selection_corr"])
    result["Delta_EE_shuffled_q_selection"] = float(result["C_selection_explained_energy"] - result["shuffled_q_selection_explained_energy"])
    result["Delta_corr_shuffled_q_selection"] = float(result["C_selection_corr"] - result["shuffled_q_selection_corr"])
    result["Delta_EE_wrong_slot"] = float(result["C_test_explained_energy"] - result["wrong_slot_test_explained_energy"])
    result["Delta_corr_wrong_slot"] = float(result["C_test_corr"] - result["wrong_slot_test_corr"])
    result["Delta_EE_shuffled_q"] = float(result["C_test_explained_energy"] - result["shuffled_q_test_explained_energy"])
    result["Delta_corr_shuffled_q"] = float(result["C_test_corr"] - result["shuffled_q_test_corr"])
    _apply_selector_labels(result, args)
    return result


def _mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _median(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.median(values)) if values else float("nan")


def _quantile(rows: Sequence[Dict[str, object]], key: str, q: float) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.percentile(values, q)) if values else float("nan")


def _sum_positive(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(sum(max(_finite_float(row.get(key), 0.0), 0.0) for row in rows))


def _sum_metric(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(sum(_finite_float(row.get(key), 0.0) for row in rows))


def _selector_report(
    selected_rows: Sequence[Dict[str, object]],
    eligible_rows: Sequence[Dict[str, object]],
    *,
    label_key: str = "test_good_strict",
) -> Dict[str, float]:
    selected = list(selected_rows)
    eligible = list(eligible_rows)
    selected_count = len(selected)
    eligible_count = len(eligible)
    good_total = sum(1 for row in eligible if bool(row.get(label_key)))
    selected_good = sum(1 for row in selected if bool(row.get(label_key)))
    selected_bad = selected_count - selected_good
    selected_energy = _sum_positive(selected, "cluster_target_energy")
    eligible_energy = _sum_positive(eligible, "cluster_target_energy")
    selected_good_energy = _sum_positive([r for r in selected if bool(r.get(label_key))], "cluster_target_energy")
    selected_bad_energy = max(selected_energy - selected_good_energy, 0.0)
    selected_gain = _sum_positive(selected, "C_test_gain")
    oracle_good = [r for r in eligible if bool(r.get(label_key))]
    oracle_gain = _sum_positive(oracle_good, "C_test_gain")
    oracle_energy = _sum_positive(oracle_good, "cluster_target_energy")
    positive_gain_count = sum(1 for row in selected if _finite_float(row.get("C_test_gain"), -1.0) > 0.0)
    return {
        "eligible_count": float(eligible_count),
        "selected_count": float(selected_count),
        "selected_ratio": float(selected_count / max(eligible_count, 1)),
        "test_good_total": float(good_total),
        "selected_test_good": float(selected_good),
        "selected_false_positive": float(selected_bad),
        "selected_precision": float(selected_good / max(selected_count, 1)),
        "selected_recall": float(selected_good / max(good_total, 1)),
        "selected_FDR": float(selected_bad / max(selected_count, 1)),
        "selected_positive_gain_ratio": float(positive_gain_count / max(selected_count, 1)),
        "selected_target_energy": selected_energy,
        "eligible_target_energy": eligible_energy,
        "selected_target_energy_coverage": float(selected_energy / max(eligible_energy, 1e-10)),
        "selected_good_target_energy": selected_good_energy,
        "selected_bad_target_energy": selected_bad_energy,
        "weighted_FDR": float(selected_bad_energy / max(selected_energy, 1e-10)),
        "selected_positive_gain": selected_gain,
        "oracle_positive_gain": oracle_gain,
        "oracle_gain_capture": float(selected_gain / max(oracle_gain, 1e-10)),
        "oracle_good_target_energy": oracle_energy,
        "oracle_energy_capture": float(selected_good_energy / max(oracle_energy, 1e-10)),
        "selected_C_test_corr_median": _median(selected, "C_test_corr"),
        "selected_C_test_EE_median": _median(selected, "C_test_explained_energy"),
        "selected_C_test_gain_sum": _sum_metric(selected, "C_test_gain"),
        "selected_leak_median": _median(selected, "C_test_leak_ratio"),
        "selected_lp_drift_median": _median(selected, "C_test_lp_drift"),
        "selected_beta_saturation_median": _median(selected, "C_fit_beta_beta_saturation"),
    }


def _top_k(rows: Sequence[Dict[str, object]], key: str, k: int, *, descending: bool = True) -> List[Dict[str, object]]:
    def sort_key(row: Dict[str, object]) -> float:
        value = _finite_float(row.get(key), float("-inf") if descending else float("inf"))
        return value

    return sorted(rows, key=sort_key, reverse=descending)[: max(int(k), 0)]


def _baseline_reports(
    eligible_rows: Sequence[Dict[str, object]],
    selected_count: int,
    *,
    label_key: str = "test_good_strict",
) -> Dict[str, Dict[str, float]]:
    rows = list(eligible_rows)
    k = max(int(selected_count), 0)
    if k <= 0:
        return {}
    rng = np.random.default_rng(20240624)
    reports: Dict[str, Dict[str, float]] = {
        "top_target_energy": _selector_report(_top_k(rows, "cluster_target_energy", k), rows, label_key=label_key),
        "top_A_selection_EE": _selector_report(_top_k(rows, "A_selection_explained_energy", k), rows, label_key=label_key),
        "top_C_selection_EE": _selector_report(_top_k(rows, "C_selection_explained_energy", k), rows, label_key=label_key),
        "top_selector_score": _selector_report(_top_k(rows, "selector_score", k), rows, label_key=label_key),
    }
    random_reports = []
    if rows:
        indices = np.arange(len(rows))
        for _ in range(64):
            chosen = rng.choice(indices, size=min(k, len(rows)), replace=False)
            random_reports.append(_selector_report([rows[int(i)] for i in chosen], rows, label_key=label_key))
    if random_reports:
        keys = sorted(random_reports[0].keys())
        reports["random_matched_count_mean"] = {key: float(np.mean([r[key] for r in random_reports])) for key in keys}
        reports["random_matched_count_p90"] = {key: float(np.percentile([r[key] for r in random_reports], 90.0)) for key in keys}
    return reports


def _transition_summary(eligible_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stages = [
        ("preeligible", lambda r: True),
        ("A_selection", lambda r: bool(r.get("pass_A_selection"))),
        ("AB_selection", lambda r: bool(r.get("pass_A_selection")) and bool(r.get("pass_B_selection"))),
        (
            "ABC_selection",
            lambda r: bool(r.get("pass_A_selection")) and bool(r.get("pass_B_selection")) and bool(r.get("pass_C_selection")),
        ),
        (
            "ABC_controls_selection",
            lambda r: bool(r.get("pass_A_selection"))
            and bool(r.get("pass_B_selection"))
            and bool(r.get("pass_C_selection"))
            and bool(r.get("pass_controls_selection")),
        ),
        ("core_pass", lambda r: bool(r.get("selector_core_pass"))),
    ]
    out: Dict[str, object] = {}
    previous_count = None
    for name, predicate in stages:
        count = sum(1 for row in eligible_rows if predicate(row))
        out[name] = {
            "count": int(count),
            "ratio_of_eligible": float(count / max(len(eligible_rows), 1)),
            "ratio_from_previous": float(count / max(previous_count, 1)) if previous_count is not None else 1.0,
        }
        previous_count = count
    reasons: Dict[str, int] = {}
    for row in eligible_rows:
        reason = str(row.get("selector_reject_reason", "not_evaluated"))
        reasons[reason] = reasons.get(reason, 0) + 1
    out["reject_reasons"] = reasons
    return out


def _write_debug(debug_root: Path, row: Dict[str, object], obs: Sequence[EvalObs], args: argparse.Namespace) -> None:
    if not obs or not row.get("valid"):
        return
    cand = Candidate(
        piece=str(row["best_piece"]),
        scale=float(row["best_scale"]),
        orientation_deg=float(row["best_orientation_deg"]),
        phase=int(row["best_phase"]),
    )
    beta = np.asarray(row["C_beta"], dtype=np.float32)
    for idx, o in enumerate(obs[:3]):
        basis, target, fit_w, _off_w, raw_basis, _lp = _basis_arrays(o, cand, args, "C", None)
        pred = basis[..., None] * beta[None, None, :]
        root = debug_root / f"{idx:02d}_{o.split}_{o.stem}"
        _save_gray(root / "support.png", o.support)
        _save_gray(root / "core_weight.png", o.core_weight)
        _save_gray(root / "q_parent.png", o.q_parent)
        _save_rgb(root / "target_signed_x4.png", np.clip(target * 4.0 + 0.5, 0.0, 1.0))
        _save_rgb(root / "pred_signed_x4.png", np.clip(pred * 4.0 + 0.5, 0.0, 1.0))
        _save_gray(root / "raw_basis_abs.png", np.clip(np.abs(raw_basis) / max(float(np.max(np.abs(raw_basis))), 1e-8), 0.0, 1.0))


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    base_ply = base_model_dir / "point_cloud" / f"iteration_{int(args.base_iteration)}" / "point_cloud.ply"
    primitive_dir = Path(args.primitive_dir).expanduser().resolve()
    base_render_dir = Path(args.base_render_dir).expanduser().resolve()
    sr_dir = Path(args.sr_dir).expanduser().resolve()
    weight_dir = Path(args.weight_dir).expanduser().resolve()
    q_parent_dir = Path(args.q_parent_dir).expanduser().resolve() if str(args.q_parent_dir).strip() else None
    for required in (base_model_dir, base_ply, primitive_dir, base_render_dir, sr_dir, weight_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    primitive_paths = _list_files(primitive_dir, [".npz"])
    if int(args.limit) > 0:
        primitive_paths = primitive_paths[: int(args.limit)]
    base_paths = _list_files(base_render_dir)
    sr_paths = _list_files(sr_dir)
    weight_paths = _list_files(weight_dir)
    q_paths = _list_files(q_parent_dir) if q_parent_dir is not None and q_parent_dir.is_dir() else []
    base_lookup = _image_lookup(base_paths)
    sr_lookup = _image_lookup(sr_paths)
    weight_lookup = _image_lookup(weight_paths)
    q_lookup = _image_lookup(q_paths)

    cameras = _load_cameras(base_model_dir)
    base_vertices, base_xyz, base_opacity, base_rgb = _load_base_vertices(base_ply)
    spray_args = _make_spray_args(args)
    carrier_rgb_paths = _list_files(Path(args.carrier_rgb_dir)) if str(args.carrier_rgb_dir).strip() and Path(args.carrier_rgb_dir).is_dir() else []
    carrier_render_paths = _list_files(Path(args.carrier_render_dir)) if str(args.carrier_render_dir).strip() and Path(args.carrier_render_dir).is_dir() else []
    weight_for_loader = weight_paths
    carrier_rgb_lookup = _image_lookup(carrier_rgb_paths)
    carrier_render_lookup = _image_lookup(carrier_render_paths)
    weight_lookup_for_loader = _image_lookup(weight_for_loader)

    print(f"[residual-tetris-v0] base      : {base_model_dir}")
    print(f"[residual-tetris-v0] primitives: {primitive_dir}")
    print(f"[residual-tetris-v0] sr/base   : {sr_dir} / {base_render_dir}")
    print(f"[residual-tetris-v0] weight    : {weight_dir}")
    print(f"[residual-tetris-v0] q_parent  : {q_parent_dir if q_parent_dir else 'proxy=weight'}")

    views: List[ViewPrimitiveSet] = []
    per_view: List[Dict[str, object]] = []
    for view_index, primitive_path in enumerate(primitive_paths):
        view, info = _load_view_primitives(
            primitive_path,
            view_index,
            spray_args,
            cameras,
            base_xyz,
            base_opacity,
            carrier_rgb_paths,
            carrier_rgb_lookup,
            carrier_render_paths,
            carrier_render_lookup,
            weight_for_loader,
            weight_lookup_for_loader,
        )
        per_view.append(info)
        if view is None:
            print(f"[residual-tetris-v0] skip {view_index + 1}/{len(primitive_paths)} {primitive_path.stem}: {info['status']}")
            continue
        views.append(view)
        print(
            f"[residual-tetris-v0] view {view_index + 1}/{len(primitive_paths)} {primitive_path.stem} "
            f"prims={view.mu_xy.shape[0]} q={float(view.q.mean()):.4f}"
        )
    if not views:
        raise RuntimeError("No usable primitive views.")

    clusters = _greedy_clusters(views, spray_args)
    results = [_cluster_to_result(views, cluster, base_xyz, base_rgb, spray_args) for cluster in clusters]
    candidate_grid = _candidate_grid(args)
    print(f"[residual-tetris-v0] clusters  : {len(clusters)}")
    print(f"[residual-tetris-v0] candidates: {len(candidate_grid)}")

    rows: List[Dict[str, object]] = []
    debug_written = 0
    for cluster_id, (cluster, result) in enumerate(zip(clusters, results)):
        if int(args.max_clusters) > 0 and len(rows) >= int(args.max_clusters):
            break
        obs, cstats = _build_eval_obs(
            cluster_obs=cluster,
            result=result,
            views=views,
            base_paths=base_paths,
            base_lookup=base_lookup,
            sr_paths=sr_paths,
            sr_lookup=sr_lookup,
            weight_paths=weight_paths,
            weight_lookup=weight_lookup,
            q_paths=q_paths,
            q_lookup=q_lookup,
            args=args,
        )
        is_eligible, reasons = _eligible(obs, cstats, args)
        row: Dict[str, object] = {
            "cluster_id": int(cluster_id),
            "status": str(result.status),
            "eligible": bool(is_eligible),
            "ineligible_reasons": reasons,
            "source_view": int(result.source_view),
            "source_primitive": int(result.source_primitive),
            "parent_index": int(result.parent_index),
            "cluster_size": int(len(cluster)),
            "obs_used": int(len(obs)),
            **{f"cluster_{k}": float(v) for k, v in cstats.items()},
        }
        if is_eligible:
            eval_result = _eval_cluster(obs, candidate_grid, args)
            row.update(eval_result)
            if debug_written < int(args.debug_limit):
                _write_debug(output_dir / "debug" / f"cluster_{cluster_id:05d}", row, obs, args)
                debug_written += 1
        rows.append(row)
        if (len(rows) % 25) == 0:
            print(f"[residual-tetris-v0] evaluated {len(rows)} clusters")

    eligible_rows = [r for r in rows if bool(r.get("eligible")) and bool(r.get("valid"))]
    numeric_keys = sorted(
        {
            key
            for row in eligible_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    means = {key: _mean(eligible_rows, key) for key in numeric_keys}
    medians = {key: _median(eligible_rows, key) for key in numeric_keys}
    q25 = {key: _quantile(eligible_rows, key, 25.0) for key in numeric_keys}
    q75 = {key: _quantile(eligible_rows, key, 75.0) for key in numeric_keys}
    pass_rows = [
        r
        for r in eligible_rows
        if float(r.get("A_test_corr", -1.0)) > 0.2
        and float(r.get("A_test_gain", -1.0)) > 0.0
        and float(r.get("A_test_leak_ratio", 1e9)) < 0.3
        and float(r.get("A_test_lp_drift", 1e9)) < 1.0
    ]
    c_pass_rows = [
        r
        for r in eligible_rows
        if float(r.get("C_test_corr", -1.0)) > 0.2
        and float(r.get("C_test_gain", -1.0)) > 0.0
        and float(r.get("C_test_leak_ratio", 1e9)) < 0.3
        and float(r.get("delivery_retention", -1.0)) > 0.5
    ]
    selected_rows = [r for r in eligible_rows if bool(r.get("selector_core_pass"))]
    capacity_limited_rows = [r for r in eligible_rows if bool(r.get("selector_capacity_limited_pass"))]
    agg_ga = sum(max(float(r.get("A_test_gain", 0.0)), 0.0) for r in eligible_rows)
    agg_gc = sum(float(r.get("C_test_gain", 0.0)) for r in eligible_rows if float(r.get("A_test_gain", 0.0)) > 0.0)
    selector_validation = _selector_report(selected_rows, eligible_rows, label_key="test_good_strict")
    selector_positive_validation = _selector_report(selected_rows, eligible_rows, label_key="test_positive")
    baselines = _baseline_reports(eligible_rows, len(selected_rows), label_key="test_good_strict")
    transition = _transition_summary(eligible_rows)
    summary = {
        "version": "evaluate_residual_tetris_oracle_v0",
        "base_model_dir": str(base_model_dir),
        "base_iteration": int(args.base_iteration),
        "primitive_dir": str(primitive_dir),
        "base_render_dir": str(base_render_dir),
        "sr_dir": str(sr_dir),
        "weight_dir": str(weight_dir),
        "q_parent_dir": str(q_parent_dir) if q_parent_dir is not None else None,
        "physical_q_source": "q_parent_dir" if q_paths else "proxy_weight_dir",
        "output_dir": str(output_dir),
        "num_views": int(len(views)),
        "num_clusters_total": int(len(clusters)),
        "num_clusters_evaluated": int(len(rows)),
        "num_clusters_eligible": int(len(eligible_rows)),
        "lockbox_note": (
            "A/B/C *_test fields are analysis-test diagnostics in this development run. "
            "Deployable selection must use *_selection fields only; use an unseen final scene/test for final claims."
        ),
        "num_gate_a_pass": int(len(pass_rows)),
        "num_gate_c_pass": int(len(c_pass_rows)),
        "gate_a_pass_ratio": float(len(pass_rows) / max(len(eligible_rows), 1)),
        "gate_c_pass_ratio": float(len(c_pass_rows) / max(len(eligible_rows), 1)),
        "analysis_test_oracle": {
            "num_gate_a_pass": int(len(pass_rows)),
            "num_gate_c_pass": int(len(c_pass_rows)),
            "gate_a_pass_ratio": float(len(pass_rows) / max(len(eligible_rows), 1)),
            "gate_c_pass_ratio": float(len(c_pass_rows) / max(len(eligible_rows), 1)),
            "aggregate_delivery_retention": float(agg_gc / max(agg_ga, 1e-10)) if agg_ga > 0.0 else float("nan"),
        },
        "selector_validation": selector_validation,
        "selector_positive_validation": selector_positive_validation,
        "selector_transition": transition,
        "selector_baselines": baselines,
        "selector_counts": {
            "core_pass": int(len(selected_rows)),
            "capacity_limited_pass": int(len(capacity_limited_rows)),
            "test_good_strict": int(sum(1 for r in eligible_rows if bool(r.get("test_good_strict")))),
            "test_positive": int(sum(1 for r in eligible_rows if bool(r.get("test_positive")))),
        },
        "aggregate_delivery_retention": float(agg_gc / max(agg_ga, 1e-10)) if agg_ga > 0.0 else float("nan"),
        "candidate_grid": [c.name for c in candidate_grid],
        "params": vars(args),
        "means": means,
        "medians": medians,
        "q25": q25,
        "q75": q75,
        "per_view": per_view,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "selected_rows.json").write_text(json.dumps(selected_rows, indent=2) + "\n", encoding="utf-8")
    readme = output_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Residual Tetris V0 oracle.",
                "",
                "A_*: visibility/support-gated signed basis capacity.",
                "B_shape_*: per-view normalized donor support compatibility.",
                "B_relative_*: globally normalized donor support with cross-view relative strength.",
                "C_*: true/proxy q_parent delivery capacity.",
                "wrong_slot_* and shuffled_q_* are matched negative controls.",
                "",
                "Important: *_test fields are analysis-test diagnostics for this run.",
                "Deployable selector flags use *_selection fields, matched-null selection margins,",
                "and beta diagnostics only. See selector_validation and selector_transition.",
                "",
                f"Main summary: {output_dir / 'summary.json'}",
                f"Rows: {output_dir / 'rows.json'}",
                f"Selected rows: {output_dir / 'selected_rows.json'}",
                f"Debug: {output_dir / 'debug'}",
                f"physical_q_source={summary['physical_q_source']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "per_view"}}, indent=2))


if __name__ == "__main__":
    main()
