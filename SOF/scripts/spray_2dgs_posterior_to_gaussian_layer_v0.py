#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from spray_2dgs_hf_carrier_to_3d_v0 import (
    IMAGE_EXTS,
    _bilinear_rgb,
    _bilinear_scalar,
    _camera_basis,
    _cholesky_axes,
    _copy_config,
    _fill_new_vertices,
    _list_files,
    _load_cameras,
    _load_gray,
    _load_rgb,
    _lookup,
    _make_tracking,
    _matrix_to_quaternion_wxyz,
    _normalize,
    _resolve,
)


@dataclass
class ViewPrimitiveSet:
    view_index: int
    stem: str
    camera: Dict[str, object]
    source_size: Tuple[int, int]
    primitive_path: Path
    mu_xy: np.ndarray
    theta: np.ndarray
    long_px: np.ndarray
    short_px: np.ndarray
    color: np.ndarray
    primitive_opacity: np.ndarray
    q_hf: np.ndarray
    q_fit: np.ndarray
    q: np.ndarray
    anchor_xyz: np.ndarray
    anchor_normal: np.ndarray
    anchor_parent: np.ndarray
    mode_dominance: np.ndarray
    mode_entropy: np.ndarray
    mode_depth: np.ndarray
    mode_score: np.ndarray
    original_primitive_id: np.ndarray


@dataclass
class ClusterResult:
    obs: List[Tuple[int, int]]
    xyz: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    opacity: float
    scale: np.ndarray
    rotation: np.ndarray
    score: float
    reproj_rms: float
    max_center_std: float
    hessian_cond: float
    mode_dominance: float
    mode_entropy: float
    status: str
    source_view: int
    source_view_slot: int
    source_primitive: int
    parent_index: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse per-view image-space 2DGS primitives into conservative 3DGS newborns. "
            "This separates image footprint, localization uncertainty, center posterior, "
            "and render covariance: cholesky is used as footprint evidence, while the "
            "3D center is solved by a cluster-level robust reprojection MAP with one "
            "base Gaussian layer point-to-plane prior."
        )
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--primitive_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--newborn_model_dir", default="")
    parser.add_argument("--carrier_rgb_dir", default="")
    parser.add_argument("--carrier_render_dir", default="")
    parser.add_argument("--carrier_weight_dir", default="")
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_primitives_per_view", type=int, default=32768)
    parser.add_argument("--max_total_newborn", type=int, default=0)
    parser.add_argument("--min_weight", type=float, default=0.02)
    parser.add_argument("--min_q", type=float, default=0.01)
    parser.add_argument("--min_primitive_opacity", type=float, default=0.0)
    parser.add_argument("--fit_error_tau", type=float, default=0.08)
    parser.add_argument("--fit_error_floor", type=float, default=0.15)

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
    parser.add_argument("--min_cluster_views", type=int, default=2)
    parser.add_argument("--min_camera_angle_deg", type=float, default=1.5)

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

    parser.add_argument(
        "--color_mode",
        default="base_anchor_additive",
        choices=["primitive", "base_anchor_additive"],
    )
    parser.add_argument("--color_gain", type=float, default=0.32)
    parser.add_argument("--opacity_floor", type=float, default=0.006)
    parser.add_argument("--opacity_scale", type=float, default=0.055)
    parser.add_argument("--opacity_power", type=float, default=0.70)
    parser.add_argument("--opacity_min", type=float, default=0.004)
    parser.add_argument("--opacity_max", type=float, default=0.075)

    parser.add_argument("--write_cpu_merged_preview", action="store_true")
    return parser.parse_args()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _load_base_vertices(base_ply: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(base_ply))
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"Base PLY lacks xyz fields: {base_ply}")
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    if "opacity" in names:
        opacity = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float32).reshape(-1))
    else:
        opacity = np.ones((xyz.shape[0],), dtype=np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_c0 = 0.28209479177387814
        dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(dc * sh_c0 + 0.5, 0.0, 1.0).astype(np.float32)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    return vertices, xyz, opacity, rgb


def _project_xyz(
    xyz: np.ndarray,
    cam: Dict[str, object],
    source_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, cam_x, cam_y, cam_z, fx, fy, cam_w, cam_h = _camera_basis(cam)
    rel = xyz - pos[None, :]
    depth = rel @ cam_z.astype(np.float32)
    x_cam = rel @ cam_x.astype(np.float32)
    y_cam = rel @ cam_y.astype(np.float32)
    px_cam = x_cam / np.maximum(depth, 1e-8) * float(fx) + float(cam_w) * 0.5
    py_cam = y_cam / np.maximum(depth, 1e-8) * float(fy) + float(cam_h) * 0.5
    sw, sh = source_size
    px = px_cam / max(float(cam_w - 1), 1.0) * max(float(sw - 1), 1.0)
    py = py_cam / max(float(cam_h - 1), 1.0) * max(float(sh - 1), 1.0)
    return px.astype(np.float32), py.astype(np.float32), depth.astype(np.float32)


def _build_visible_layer_grid(
    base_xyz: np.ndarray,
    base_opacity: np.ndarray,
    cam: Dict[str, object],
    source_size: Tuple[int, int],
    base_opacity_min: float,
    depth_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    px, py, depth = _project_xyz(base_xyz, cam, source_size)
    sw, sh = source_size
    valid = (
        np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(depth)
        & (depth > float(depth_min))
        & (px >= 0.0)
        & (py >= 0.0)
        & (px <= float(sw - 1))
        & (py <= float(sh - 1))
        & (base_opacity >= float(base_opacity_min))
    )
    grid_count = int(sw) * int(sh)
    grid_index = np.full((grid_count,), -1, dtype=np.int64)
    grid_depth = np.full((grid_count,), np.inf, dtype=np.float32)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return grid_index.reshape(sh, sw), grid_depth.reshape(sh, sw)
    xi = np.rint(px[ids]).astype(np.int64)
    yi = np.rint(py[ids]).astype(np.int64)
    inside = (xi >= 0) & (xi < sw) & (yi >= 0) & (yi < sh)
    ids = ids[inside]
    xi = xi[inside]
    yi = yi[inside]
    if ids.size == 0:
        return grid_index.reshape(sh, sw), grid_depth.reshape(sh, sw)
    lin = yi * int(sw) + xi
    d = depth[ids]
    order = np.lexsort((d, lin))
    sorted_lin = lin[order]
    first = np.concatenate(([True], sorted_lin[1:] != sorted_lin[:-1]))
    winners = order[first]
    win_lin = lin[winners]
    grid_index[win_lin] = ids[winners]
    grid_depth[win_lin] = d[winners]
    return grid_index.reshape(sh, sw), grid_depth.reshape(sh, sw)


def _sample_offsets() -> np.ndarray:
    return np.asarray(
        [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.65),
            (-1.0, 0.0, 0.65),
            (0.0, 1.0, 0.65),
            (0.0, -1.0, 0.65),
            (0.7, 0.7, 0.42),
            (-0.7, 0.7, 0.42),
            (0.7, -0.7, 0.42),
            (-0.7, -0.7, 0.42),
        ],
        dtype=np.float32,
    )


def _search_offsets(radius: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    radius = max(int(radius), 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                out.append((dx, dy))
    out.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    return out


def _lookup_grid_nearest(grid_index: np.ndarray, x: float, y: float, offsets: Sequence[Tuple[int, int]]) -> int:
    h, w = grid_index.shape
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    for dx, dy in offsets:
        xx = cx + int(dx)
        yy = cy + int(dy)
        if 0 <= xx < w and 0 <= yy < h:
            idx = int(grid_index[yy, xx])
            if idx >= 0:
                return idx
    return -1


def _surface_normal_from_parents(parent_xyz: np.ndarray, weights: np.ndarray, camera_pos: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if parent_xyz.shape[0] < 3 or float(weights.sum()) <= 1e-8:
        normal = fallback.astype(np.float32)
    else:
        w = weights.astype(np.float64)
        w = w / max(float(w.sum()), 1e-8)
        center = (parent_xyz.astype(np.float64) * w[:, None]).sum(axis=0)
        delta = parent_xyz.astype(np.float64) - center[None, :]
        cov = (delta * w[:, None]).T @ delta
        try:
            vals, vecs = np.linalg.eigh(cov + np.eye(3) * 1e-10)
            normal = vecs[:, int(np.argmin(vals))].astype(np.float32)
        except np.linalg.LinAlgError:
            normal = fallback.astype(np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    to_cam = camera_pos.astype(np.float32) - parent_xyz.mean(axis=0).astype(np.float32) if parent_xyz.size else fallback
    if float(np.dot(normal, to_cam)) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def _cluster_surface_modes(
    parent_ids: List[int],
    parent_scores: List[float],
    base_xyz: np.ndarray,
    depth_by_id: Dict[int, float],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if not parent_ids:
        return []
    score_by_id: Dict[int, float] = {}
    for idx, score in zip(parent_ids, parent_scores):
        score_by_id[int(idx)] = score_by_id.get(int(idx), 0.0) + float(score)
    items = sorted(score_by_id.items(), key=lambda kv: kv[1], reverse=True)[:32]
    modes: List[Dict[str, object]] = []
    for idx, score in items:
        xyz = base_xyz[int(idx)]
        depth = float(depth_by_id[int(idx)])
        placed = False
        for mode in modes:
            md = float(mode["depth"])
            mx = np.asarray(mode["xyz"], dtype=np.float32)
            depth_tol = max(float(args.mode_depth_abs), float(args.mode_depth_rel) * max(abs(md), abs(depth), 1e-6))
            if abs(depth - md) <= depth_tol and float(np.linalg.norm(xyz - mx)) <= float(args.mode_position_radius):
                old = float(mode["score"])
                new = old + float(score)
                mode["ids"].append(int(idx))
                mode["weights"].append(float(score))
                mode["score"] = new
                mode["xyz"] = (mx * old + xyz * float(score)) / max(new, 1e-8)
                mode["depth"] = (md * old + depth * float(score)) / max(new, 1e-8)
                placed = True
                break
        if not placed:
            modes.append(
                {
                    "ids": [int(idx)],
                    "weights": [float(score)],
                    "score": float(score),
                    "xyz": xyz.astype(np.float32),
                    "depth": float(depth),
                }
            )
    modes.sort(key=lambda mode: float(mode["score"]), reverse=True)
    return modes


def _integrate_primitive_mode(
    mu: np.ndarray,
    theta: float,
    long_px: float,
    short_px: float,
    grid_index: np.ndarray,
    grid_depth: np.ndarray,
    base_xyz: np.ndarray,
    base_opacity: np.ndarray,
    cam_pos: np.ndarray,
    cam_z: np.ndarray,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    offsets = _sample_offsets()
    search = _search_offsets(int(args.layer_search_radius_px))
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    long_vec = np.asarray([c, s], dtype=np.float32)
    short_vec = np.asarray([-s, c], dtype=np.float32)
    parent_ids: List[int] = []
    parent_scores: List[float] = []
    parent_depths: List[float] = []
    scale = float(args.footprint_sample_scale)
    for ox, oy, weight in offsets:
        xy = mu + scale * (float(ox) * float(long_px) * long_vec + float(oy) * float(short_px) * short_vec)
        idx = _lookup_grid_nearest(grid_index, float(xy[0]), float(xy[1]), search)
        if idx < 0:
            continue
        parent_ids.append(int(idx))
        parent_scores.append(float(weight) * float(base_opacity[int(idx)]))
        parent_depths.append(float((base_xyz[int(idx)] - cam_pos) @ cam_z.astype(np.float32)))
    if not parent_ids:
        return None
    depth_by_id = {int(idx): float(depth) for idx, depth in zip(parent_ids, parent_depths)}
    modes = _cluster_surface_modes(parent_ids, parent_scores, base_xyz, depth_by_id, args)
    if not modes:
        return None
    total_score = float(sum(float(mode["score"]) for mode in modes))
    probs = np.asarray([float(mode["score"]) / max(total_score, 1e-8) for mode in modes], dtype=np.float32)
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-8))).sum() / max(math.log(max(len(probs), 2)), 1e-8))
    top = modes[0]
    ids = np.asarray(top["ids"], dtype=np.int64)
    weights = np.asarray(top["weights"], dtype=np.float32)
    fallback = cam_z.astype(np.float32)
    normal = _surface_normal_from_parents(base_xyz[ids], weights, cam_pos, fallback)
    return {
        "parent": int(ids[0]),
        "xyz": np.asarray(top["xyz"], dtype=np.float32),
        "normal": normal,
        "depth": float(top["depth"]),
        "mode_score": float(top["score"]),
        "mode_dominance": float(probs[0]),
        "mode_entropy": entropy,
        "num_modes": int(len(modes)),
    }


def _infer_source_size(primitive: Dict[str, np.ndarray], rgb_path: Optional[Path], weight_path: Optional[Path]) -> Tuple[int, int]:
    for path in (rgb_path, weight_path):
        if path is not None and path.is_file():
            from PIL import Image

            with Image.open(path) as image:
                return image.size
    mu = np.asarray(primitive["mu_xy"], dtype=np.float32)
    return (
        max(1, int(math.ceil(float(np.nanmax(mu[:, 0])) + 1.0))),
        max(1, int(math.ceil(float(np.nanmax(mu[:, 1])) + 1.0))),
    )


def _camera_for_view(cameras: Dict[str, Dict[str, object]], stem: str, view_index: int, policy: str) -> Tuple[Optional[Dict[str, object]], str]:
    cam = cameras.get(stem.lower())
    if cam is not None:
        return cam, "stem"
    if policy == "stem":
        return None, "missing"
    values = list(cameras.values())
    if view_index >= len(values):
        return None, "missing"
    return values[view_index], "order"


def _load_view_primitives(
    primitive_path: Path,
    view_index: int,
    args: argparse.Namespace,
    cameras: Dict[str, Dict[str, object]],
    base_xyz: np.ndarray,
    base_opacity: np.ndarray,
    rgb_paths: Sequence[Path],
    rgb_lookup: Dict[str, Path],
    render_paths: Sequence[Path],
    render_lookup: Dict[str, Path],
    weight_paths: Sequence[Path],
    weight_lookup: Dict[str, Path],
) -> Tuple[Optional[ViewPrimitiveSet], Dict[str, object]]:
    stem = primitive_path.stem
    primitive = dict(np.load(primitive_path))
    if "mu_xy" not in primitive or "color" not in primitive:
        raise KeyError(f"{primitive_path} must contain mu_xy and color")
    cam, cam_match = _camera_for_view(cameras, stem, view_index, str(args.match_policy))
    if cam is None:
        return None, {"stem": stem, "status": "missing_camera"}
    rgb_path = _resolve(rgb_paths, rgb_lookup, stem, view_index, args.match_policy) if rgb_paths else None
    render_path = _resolve(render_paths, render_lookup, stem, view_index, args.match_policy) if render_paths else None
    weight_path = _resolve(weight_paths, weight_lookup, stem, view_index, args.match_policy) if weight_paths else None
    source_size = _infer_source_size(primitive, rgb_path, weight_path)
    mu_all = np.asarray(primitive["mu_xy"], dtype=np.float32)
    color_all = np.asarray(primitive["color"], dtype=np.float32)
    opacity_all = np.asarray(primitive.get("opacity", np.ones((mu_all.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
    theta_all, long_all, short_all = _cholesky_axes(
        np.asarray(primitive["cholesky"], dtype=np.float32) if "cholesky" in primitive else None,
        int(mu_all.shape[0]),
    )
    if weight_path is not None and weight_path.is_file():
        weight_img = _load_gray(weight_path, size=source_size)
        q_hf_all = _bilinear_scalar(weight_img, mu_all, source_size)
    else:
        q_hf_all = np.ones((mu_all.shape[0],), dtype=np.float32)
    rgb_img = _load_rgb(rgb_path, size=source_size) if rgb_path is not None and rgb_path.is_file() else None
    if rgb_img is not None:
        sampled = _bilinear_rgb(rgb_img, mu_all, source_size)
        low_color = np.mean(np.abs(color_all), axis=1, keepdims=True) < 1e-4
        color_all = np.where(low_color, sampled, color_all)
    if rgb_img is not None and render_path is not None and render_path.is_file():
        render_img = _load_rgb(render_path, size=source_size)
        render_sample = _bilinear_rgb(render_img, mu_all, source_size)
        target_sample = _bilinear_rgb(rgb_img, mu_all, source_size)
        fit_error = np.mean(np.abs(render_sample - target_sample), axis=1)
        q_fit_all = np.exp(-fit_error / max(float(args.fit_error_tau), 1e-6)).astype(np.float32)
        q_fit_all = np.clip(q_fit_all, float(args.fit_error_floor), 1.0)
        render_match = str(render_path)
    else:
        q_fit_all = np.ones((mu_all.shape[0],), dtype=np.float32)
        render_match = ""

    finite = np.isfinite(mu_all).all(axis=1) & np.isfinite(color_all).all(axis=1)
    finite &= np.isfinite(opacity_all) & np.isfinite(q_hf_all) & np.isfinite(q_fit_all)
    finite &= q_hf_all >= float(args.min_weight)
    finite &= opacity_all >= float(args.min_primitive_opacity)
    keep = np.flatnonzero(finite)
    if keep.size == 0:
        return None, {"stem": stem, "status": "empty_after_initial_filter"}
    order_score = (q_hf_all * q_fit_all)[keep]
    order = np.argsort(order_score)[::-1]
    if int(args.max_primitives_per_view) > 0:
        order = order[: int(args.max_primitives_per_view)]
    keep = keep[order]
    mu = mu_all[keep]
    color = np.clip(color_all[keep], 0.0, 1.0)
    opacity = np.clip(opacity_all[keep], 0.0, 1.0)
    theta = theta_all[keep]
    long_px = long_all[keep]
    short_px = short_all[keep]
    q_hf = np.clip(q_hf_all[keep], 0.0, 1.0).astype(np.float32)
    q_fit_kept = np.clip(q_fit_all[keep], 0.0, 1.0).astype(np.float32)

    grid_index, grid_depth = _build_visible_layer_grid(
        base_xyz,
        base_opacity,
        cam,
        source_size,
        float(args.base_opacity_min),
        float(args.depth_min),
    )
    cam_pos, _cam_x, _cam_y, cam_z, _fx, _fy, _cw, _ch = _camera_basis(cam)
    anchors: List[Dict[str, object]] = []
    valid_ids: List[int] = []
    for local_i in range(int(mu.shape[0])):
        mode = _integrate_primitive_mode(
            mu[local_i],
            float(theta[local_i]),
            float(long_px[local_i]),
            float(short_px[local_i]),
            grid_index,
            grid_depth,
            base_xyz,
            base_opacity,
            cam_pos.astype(np.float32),
            cam_z.astype(np.float32),
            args,
        )
        if mode is None:
            continue
        q_layer = float(mode["mode_dominance"]) * (1.0 - 0.35 * float(mode["mode_entropy"]))
        q = float(q_hf[local_i]) * float(q_fit_kept[local_i]) * np.clip(q_layer, 0.0, 1.0)
        if q < float(args.min_q):
            continue
        anchors.append(mode)
        valid_ids.append(local_i)
    if not valid_ids:
        return None, {
            "stem": stem,
            "status": "empty_after_surface_mode",
            "initial_candidates": int(keep.size),
            "camera_match": cam_match,
        }
    ids = np.asarray(valid_ids, dtype=np.int64)
    mode_dominance = np.asarray([float(m["mode_dominance"]) for m in anchors], dtype=np.float32)
    mode_entropy = np.asarray([float(m["mode_entropy"]) for m in anchors], dtype=np.float32)
    q_fit = q_fit_kept[ids]
    q_layer = mode_dominance * (1.0 - 0.35 * mode_entropy)
    q = np.clip(q_hf[ids] * q_fit * q_layer, 0.0, 1.0).astype(np.float32)
    view = ViewPrimitiveSet(
        view_index=int(view_index),
        stem=stem,
        camera=cam,
        source_size=source_size,
        primitive_path=primitive_path,
        mu_xy=mu[ids],
        theta=theta[ids],
        long_px=long_px[ids],
        short_px=short_px[ids],
        color=color[ids],
        primitive_opacity=opacity[ids],
        q_hf=q_hf[ids],
        q_fit=q_fit,
        q=q,
        anchor_xyz=np.stack([np.asarray(m["xyz"], dtype=np.float32) for m in anchors], axis=0),
        anchor_normal=np.stack([np.asarray(m["normal"], dtype=np.float32) for m in anchors], axis=0),
        anchor_parent=np.asarray([int(m["parent"]) for m in anchors], dtype=np.int64),
        mode_dominance=mode_dominance,
        mode_entropy=mode_entropy,
        mode_depth=np.asarray([float(m["depth"]) for m in anchors], dtype=np.float32),
        mode_score=np.asarray([float(m["mode_score"]) for m in anchors], dtype=np.float32),
        original_primitive_id=keep[ids].astype(np.int64),
    )
    info = {
        "stem": stem,
        "status": "ok",
        "camera_match": cam_match,
        "available": int(mu_all.shape[0]),
        "initial_candidates": int(keep.size),
        "surface_candidates": int(ids.size),
        "rgb_path": "" if rgb_path is None else str(rgb_path),
        "render_path": render_match,
        "weight_path": "" if weight_path is None else str(weight_path),
        "q_mean": float(q.mean()),
        "q_p90": float(np.percentile(q, 90)),
        "q_fit_mean": float(q_fit.mean()),
        "mode_dominance_mean": float(mode_dominance.mean()),
        "mode_entropy_mean": float(mode_entropy.mean()),
        "source_size": [int(source_size[0]), int(source_size[1])],
    }
    return view, info


def _project_point(cam: Dict[str, object], source_size: Tuple[int, int], xyz: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    px, py, depth = _project_xyz(xyz.reshape(1, 3).astype(np.float32), cam, source_size)
    sw, sh = source_size
    xy = np.asarray([px[0], py[0]], dtype=np.float32)
    valid = (
        np.isfinite(xy).all()
        and np.isfinite(float(depth[0]))
        and float(depth[0]) > 1e-8
        and -16.0 <= float(xy[0]) <= float(sw + 15)
        and -16.0 <= float(xy[1]) <= float(sh + 15)
    )
    return xy, float(depth[0]), valid


def _build_candidate_grid(view: ViewPrimitiveSet, cell_px: float) -> Dict[Tuple[int, int], List[int]]:
    cell = max(float(cell_px), 1.0)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for i, xy in enumerate(view.mu_xy):
        key = (int(math.floor(float(xy[0]) / cell)), int(math.floor(float(xy[1]) / cell)))
        grid.setdefault(key, []).append(i)
    return grid


def _query_candidate_grid(
    view: ViewPrimitiveSet,
    grid: Dict[Tuple[int, int], List[int]],
    xy: np.ndarray,
    radius_px: float,
    cell_px: float,
) -> Iterable[int]:
    cell = max(float(cell_px), 1.0)
    cx = int(math.floor(float(xy[0]) / cell))
    cy = int(math.floor(float(xy[1]) / cell))
    span = int(math.ceil(float(radius_px) / cell)) + 1
    for gy in range(cy - span, cy + span + 1):
        for gx in range(cx - span, cx + span + 1):
            yield from grid.get((gx, gy), [])


def _axis3d(view: ViewPrimitiveSet, idx: int, normal: np.ndarray) -> np.ndarray:
    _pos, cam_x, cam_y, _cam_z, _fx, _fy, _w, _h = _camera_basis(view.camera)
    theta = float(view.theta[idx])
    axis = math.cos(theta) * cam_x.astype(np.float32) + math.sin(theta) * cam_y.astype(np.float32)
    axis = axis - normal.astype(np.float32) * float(np.dot(axis, normal))
    n = float(np.linalg.norm(axis))
    if n <= 1e-8:
        axis = cam_x.astype(np.float32) - normal.astype(np.float32) * float(np.dot(cam_x, normal))
        n = max(float(np.linalg.norm(axis)), 1e-8)
    return (axis / n).astype(np.float32)


def _candidate_match_cost(
    anchor_view: ViewPrimitiveSet,
    anchor_idx: int,
    target_view: ViewPrimitiveSet,
    target_idx: int,
    pred_xy: np.ndarray,
    args: argparse.Namespace,
) -> float:
    delta = target_view.mu_xy[target_idx] - pred_xy
    dist = float(np.linalg.norm(delta))
    loc_sigma = float(args.localization_sigma_px) + float(args.localization_footprint_beta) * math.sqrt(
        max(float(target_view.long_px[target_idx] * target_view.short_px[target_idx]), 1e-8)
    )
    geo = dist / max(loc_sigma, 1e-6)
    log_shape = abs(math.log(max(float(anchor_view.long_px[anchor_idx]), 1e-5) / max(float(target_view.long_px[target_idx]), 1e-5)))
    log_shape += abs(math.log(max(float(anchor_view.short_px[anchor_idx]), 1e-5) / max(float(target_view.short_px[target_idx]), 1e-5)))
    color = float(np.linalg.norm(anchor_view.color[anchor_idx] - target_view.color[target_idx]))
    return geo + float(args.association_shape_weight) * log_shape + float(args.association_color_weight) * color


def _cluster_camera_angle(views: Sequence[ViewPrimitiveSet], obs: Sequence[Tuple[int, int]], xyz: np.ndarray) -> float:
    dirs: List[np.ndarray] = []
    for vi, _idx in obs:
        pos, _x, _y, _z, _fx, _fy, _w, _h = _camera_basis(views[vi].camera)
        d = pos.astype(np.float32) - xyz.astype(np.float32)
        d = d / max(float(np.linalg.norm(d)), 1e-8)
        dirs.append(d)
    if len(dirs) < 2:
        return 0.0
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            cos = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            best = max(best, math.degrees(math.acos(cos)))
    return float(best)


def _numeric_projection_jacobian(cam: Dict[str, object], source_size: Tuple[int, int], xyz: np.ndarray) -> np.ndarray:
    _xy0, depth, valid = _project_point(cam, source_size, xyz)
    eps = max(1e-4, abs(float(depth)) * 1e-4)
    jac = np.zeros((2, 3), dtype=np.float64)
    for k in range(3):
        step = np.zeros((3,), dtype=np.float32)
        step[k] = eps
        xp, _dp, vp = _project_point(cam, source_size, xyz + step)
        xm, _dm, vm = _project_point(cam, source_size, xyz - step)
        if not (valid and vp and vm):
            jac[:, k] = 0.0
        else:
            jac[:, k] = ((xp - xm) / (2.0 * eps)).astype(np.float64)
    return jac


def _solve_cluster_map(
    views: Sequence[ViewPrimitiveSet],
    obs: Sequence[Tuple[int, int]],
    base_xyz: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    anchor_view, anchor_idx = obs[0]
    xb = views[anchor_view].anchor_xyz[anchor_idx].astype(np.float64)
    n = views[anchor_view].anchor_normal[anchor_idx].astype(np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    x = xb.copy()
    tangent = np.eye(3, dtype=np.float64) - np.outer(n, n)
    for _ in range(int(args.map_iterations)):
        h = np.eye(3, dtype=np.float64) * float(args.map_damping)
        g = np.zeros((3,), dtype=np.float64)
        sigma_surface = max(float(args.surface_sigma), 1e-8)
        rb = float(np.dot(n, x - xb)) / sigma_surface
        wb = min(1.0, float(args.map_huber_px) / max(abs(rb), 1e-8))
        jb = n[None, :] / sigma_surface
        h += wb * (jb.T @ jb)
        g += wb * (jb.T.reshape(3) * rb)
        tangent_weight = float(args.tangent_prior_weight)
        if tangent_weight > 0.0:
            rt = tangent @ (x - xb)
            h += tangent_weight * tangent
            g += tangent_weight * rt
        for vi, oi in obs:
            view = views[vi]
            pred, _depth, valid = _project_point(view.camera, view.source_size, x.astype(np.float32))
            if not valid:
                continue
            r = (pred - view.mu_xy[oi]).astype(np.float64)
            loc_sigma = float(args.localization_sigma_px) + float(args.localization_footprint_beta) * math.sqrt(
                max(float(view.long_px[oi] * view.short_px[oi]), 1e-8)
            )
            loc_sigma = max(loc_sigma, 1e-6)
            rn = r / loc_sigma
            norm = float(np.linalg.norm(rn))
            robust = min(1.0, float(args.map_huber_px) / max(norm, 1e-8))
            q = max(float(view.q[oi]), 1e-5)
            jac = _numeric_projection_jacobian(view.camera, view.source_size, x.astype(np.float32)) / loc_sigma
            h += q * robust * (jac.T @ jac)
            g += q * robust * (jac.T @ rn)
        try:
            dx = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(h) @ g
        step = float(np.linalg.norm(dx))
        x += dx
        if step < 1e-5:
            break
    residuals: List[float] = []
    h_final = np.eye(3, dtype=np.float64) * float(args.map_damping)
    sigma_surface = max(float(args.surface_sigma), 1e-8)
    jb = n[None, :] / sigma_surface
    h_final += jb.T @ jb
    for vi, oi in obs:
        view = views[vi]
        pred, _depth, valid = _project_point(view.camera, view.source_size, x.astype(np.float32))
        if not valid:
            residuals.append(float("inf"))
            continue
        r = pred - view.mu_xy[oi]
        residuals.append(float(np.linalg.norm(r)))
        loc_sigma = float(args.localization_sigma_px) + float(args.localization_footprint_beta) * math.sqrt(
            max(float(view.long_px[oi] * view.short_px[oi]), 1e-8)
        )
        jac = _numeric_projection_jacobian(view.camera, view.source_size, x.astype(np.float32)) / max(loc_sigma, 1e-6)
        h_final += max(float(view.q[oi]), 1e-5) * (jac.T @ jac)
    try:
        u = np.linalg.inv(h_final)
    except np.linalg.LinAlgError:
        u = np.linalg.pinv(h_final)
    eig = np.linalg.eigvalsh(np.nan_to_num(u, nan=1e6, posinf=1e6, neginf=1e6))
    max_std = float(math.sqrt(max(float(np.max(eig)), 0.0)))
    cond = float(np.linalg.cond(h_final))
    rms = float(math.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
    return x.astype(np.float32), n.astype(np.float32), rms, max_std, cond, u.astype(np.float32)


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights.astype(np.float64)
    return (values.astype(np.float64) * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-8)


def _tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normal.astype(np.float32)
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(seed, n))) > 0.85:
        seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    t1 = seed - n * float(np.dot(seed, n))
    t1 = t1 / max(float(np.linalg.norm(t1)), 1e-8)
    t2 = np.cross(n, t1).astype(np.float32)
    t2 = t2 / max(float(np.linalg.norm(t2)), 1e-8)
    return t1.astype(np.float32), t2.astype(np.float32)


def _fit_render_shape(
    views: Sequence[ViewPrimitiveSet],
    obs: Sequence[Tuple[int, int]],
    xyz: np.ndarray,
    normal: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray([float(views[vi].q[oi]) for vi, oi in obs], dtype=np.float32)
    weights = np.maximum(weights, 1e-5)
    axes: List[np.ndarray] = []
    long_world: List[float] = []
    short_world: List[float] = []
    for vi, oi in obs:
        view = views[vi]
        axis = _axis3d(view, oi, normal)
        axes.append(axis)
        _pred, depth, _valid = _project_point(view.camera, view.source_size, xyz)
        _pos, _cam_x, _cam_y, _cam_z, fx, fy, _w, _h = _camera_basis(view.camera)
        pix_world = max(float(depth), 1e-8) / max((float(fx) + float(fy)) * 0.5, 1e-8)
        filter2 = float(args.screen_filter_sigma_px) ** 2 + float(args.extract_sigma_px) ** 2
        lp = math.sqrt(max(float(view.long_px[oi]) ** 2 - filter2, 0.25))
        sp = math.sqrt(max(float(view.short_px[oi]) ** 2 - filter2, 0.12))
        long_world.append(lp * pix_world * float(args.scale_multiplier))
        short_world.append(sp * pix_world * float(args.scale_multiplier))
    axes_arr = np.stack(axes, axis=0).astype(np.float64)
    orient = np.zeros((3, 3), dtype=np.float64)
    for axis, weight in zip(axes_arr, weights):
        orient += float(weight) * np.outer(axis, axis)
    t1, t2 = _tangent_basis(normal)
    b = np.stack([t1, t2], axis=1).astype(np.float64)
    orient2 = b.T @ orient @ b
    try:
        vals, vecs = np.linalg.eigh(orient2)
        long2 = vecs[:, int(np.argmax(vals))]
        long_axis = (b @ long2).astype(np.float32)
    except np.linalg.LinAlgError:
        long_axis = t1
    long_axis = long_axis - normal.astype(np.float32) * float(np.dot(long_axis, normal))
    long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-8)
    short_axis = np.cross(normal.astype(np.float32), long_axis).astype(np.float32)
    short_axis = short_axis / max(float(np.linalg.norm(short_axis)), 1e-8)
    rot = np.stack([long_axis, short_axis, normal.astype(np.float32)], axis=1)[None, :, :]
    quat = _matrix_to_quaternion_wxyz(rot)[0]
    lw = np.asarray(long_world, dtype=np.float32)
    sw = np.asarray(short_world, dtype=np.float32)
    scale_long = float(np.average(lw, weights=weights))
    scale_short = float(np.average(sw, weights=weights))
    scale_long = float(np.clip(scale_long, float(args.scale_min), float(args.scale_max)))
    scale_short = float(np.clip(scale_short, float(args.scale_min), float(args.scale_max)))
    scale_normal = float(
        np.clip(
            scale_short * float(args.normal_scale_ratio),
            float(args.normal_scale_min),
            float(args.normal_scale_max),
        )
    )
    scale = np.asarray([scale_long, scale_short, scale_normal], dtype=np.float32)
    return scale, quat.astype(np.float32)


def _resolve_cluster_color_opacity(
    views: Sequence[ViewPrimitiveSet],
    obs: Sequence[Tuple[int, int]],
    base_rgb: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, float]:
    weights = np.asarray([float(views[vi].q[oi]) for vi, oi in obs], dtype=np.float32)
    weights = np.maximum(weights, 1e-5)
    colors = np.stack([views[vi].color[oi] for vi, oi in obs], axis=0).astype(np.float32)
    carrier = _weighted_average(colors, weights).astype(np.float32)
    anchor_view, anchor_idx = obs[0]
    parent = int(views[anchor_view].anchor_parent[anchor_idx])
    if str(args.color_mode) == "base_anchor_additive":
        color = np.clip(base_rgb[parent] + float(args.color_gain) * carrier, 0.0, 1.0)
    else:
        color = np.clip(carrier, 0.0, 1.0)
    score = float(np.clip(weights.mean(), 0.0, 1.0))
    opacity = float(args.opacity_floor) + float(args.opacity_scale) * (score ** float(args.opacity_power))
    opacity = float(np.clip(opacity, float(args.opacity_min), float(args.opacity_max)))
    return color.astype(np.float32), opacity, score


def _greedy_clusters(
    views: Sequence[ViewPrimitiveSet],
    args: argparse.Namespace,
) -> List[List[Tuple[int, int]]]:
    grids = [_build_candidate_grid(view, float(args.association_cell_px)) for view in views]
    flat: List[Tuple[float, int, int]] = []
    for vi, view in enumerate(views):
        for oi, q in enumerate(view.q):
            flat.append((float(q), vi, oi))
    flat.sort(reverse=True)
    used = [np.zeros((view.mu_xy.shape[0],), dtype=bool) for view in views]
    clusters: List[List[Tuple[int, int]]] = []
    for _q, vi, oi in flat:
        if used[vi][oi]:
            continue
        anchor = views[vi]
        anchor_xyz = anchor.anchor_xyz[oi]
        cluster: List[Tuple[int, int]] = [(vi, oi)]
        for vj, target in enumerate(views):
            if vj == vi:
                continue
            pred, _depth, valid = _project_point(target.camera, target.source_size, anchor_xyz)
            if not valid:
                continue
            best: Optional[Tuple[float, int]] = None
            for cand in _query_candidate_grid(
                target,
                grids[vj],
                pred,
                float(args.association_radius_px),
                float(args.association_cell_px),
            ):
                if used[vj][cand]:
                    continue
                dist = float(np.linalg.norm(target.mu_xy[cand] - pred))
                if dist > float(args.association_radius_px):
                    continue
                cost = _candidate_match_cost(anchor, oi, target, cand, pred, args)
                if cost <= float(args.association_max_cost) and (best is None or cost < best[0]):
                    best = (cost, cand)
            if best is not None:
                cluster.append((vj, int(best[1])))
        if len(cluster) >= max(1, int(args.min_cluster_views)):
            mean_xyz = np.mean([views[a].anchor_xyz[b] for a, b in cluster], axis=0)
            angle = _cluster_camera_angle(views, cluster, mean_xyz.astype(np.float32))
            if angle >= float(args.min_camera_angle_deg):
                for a, b in cluster:
                    used[a][b] = True
                clusters.append(cluster)
    return clusters


def _cluster_to_result(
    views: Sequence[ViewPrimitiveSet],
    obs: Sequence[Tuple[int, int]],
    base_xyz: np.ndarray,
    base_rgb: np.ndarray,
    args: argparse.Namespace,
) -> ClusterResult:
    xyz, normal, rms, max_std, cond, _u = _solve_cluster_map(views, obs, base_xyz, args)
    scale, quat = _fit_render_shape(views, obs, xyz, normal, args)
    color, opacity, score = _resolve_cluster_color_opacity(views, obs, base_rgb, args)
    mode_dom = float(np.mean([float(views[vi].mode_dominance[oi]) for vi, oi in obs]))
    mode_ent = float(np.mean([float(views[vi].mode_entropy[oi]) for vi, oi in obs]))
    status = "confirmed"
    if rms > float(args.max_reproj_rms_px) or max_std > float(args.max_center_std) or cond > float(args.max_hessian_cond):
        status = "probation"
    if mode_dom < float(args.min_mode_dominance) or mode_ent > float(args.max_mode_entropy):
        status = "probation"
    source_view, source_idx = obs[0]
    return ClusterResult(
        obs=list(obs),
        xyz=xyz.astype(np.float32),
        normal=normal.astype(np.float32),
        color=color.astype(np.float32),
        opacity=float(opacity),
        scale=scale.astype(np.float32),
        rotation=quat.astype(np.float32),
        score=float(score),
        reproj_rms=float(rms),
        max_center_std=float(max_std),
        hessian_cond=float(cond),
        mode_dominance=mode_dom,
        mode_entropy=mode_ent,
        status=status,
        source_view=int(views[source_view].view_index),
        source_view_slot=int(source_view),
        source_primitive=int(views[source_view].original_primitive_id[source_idx]),
        parent_index=int(views[source_view].anchor_parent[source_idx]),
    )


def _build_newborn_arrays(results: Sequence[ClusterResult]) -> Dict[str, np.ndarray]:
    confirmed = [res for res in results if res.status == "confirmed"]
    if not confirmed:
        return {}
    return {
        "xyz": np.stack([res.xyz for res in confirmed], axis=0).astype(np.float32),
        "color": np.stack([res.color for res in confirmed], axis=0).astype(np.float32),
        "opacity": np.asarray([[res.opacity] for res in confirmed], dtype=np.float32),
        "scale": np.stack([res.scale for res in confirmed], axis=0).astype(np.float32),
        "rotation": np.stack([res.rotation for res in confirmed], axis=0).astype(np.float32),
        "source_view": np.asarray([res.source_view for res in confirmed], dtype=np.int32),
        "source_view_slot": np.asarray([res.source_view_slot for res in confirmed], dtype=np.int32),
        "source_primitive": np.asarray([res.source_primitive for res in confirmed], dtype=np.int64),
        "parent_index": np.asarray([res.parent_index for res in confirmed], dtype=np.int64),
        "score": np.asarray([res.score for res in confirmed], dtype=np.float32),
        "reproj_rms": np.asarray([res.reproj_rms for res in confirmed], dtype=np.float32),
        "max_center_std": np.asarray([res.max_center_std for res in confirmed], dtype=np.float32),
        "hessian_cond": np.asarray([res.hessian_cond for res in confirmed], dtype=np.float32),
        "mode_dominance": np.asarray([res.mode_dominance for res in confirmed], dtype=np.float32),
        "mode_entropy": np.asarray([res.mode_entropy for res in confirmed], dtype=np.float32),
        "cluster_size": np.asarray([len(res.obs) for res in confirmed], dtype=np.int32),
    }


def _result_records(results: Sequence[ClusterResult], views: Sequence[ViewPrimitiveSet]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for i, res in enumerate(results):
        records.append(
            {
                "index": i,
                "status": res.status,
                "cluster_size": len(res.obs),
                "score": res.score,
                "reproj_rms": res.reproj_rms,
                "max_center_std": res.max_center_std,
                "hessian_cond": res.hessian_cond,
                "mode_dominance": res.mode_dominance,
                "mode_entropy": res.mode_entropy,
                "parent_index": res.parent_index,
                "source_view": res.source_view,
                "source_view_slot": res.source_view_slot,
                "source_primitive": res.source_primitive,
                "source": [
                    {
                        "view": int(views[vi].view_index),
                        "view_slot": int(vi),
                        "stem": views[vi].stem,
                        "primitive": int(views[vi].original_primitive_id[oi]),
                        "q": float(views[vi].q[oi]),
                    }
                    for vi, oi in res.obs
                ],
            }
        )
    return records


def _slice_newborn(newborn: Dict[str, np.ndarray], count: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in newborn.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (newborn["xyz"].shape[0],):
            out[key] = value[:count]
        else:
            out[key] = value
    return out


def main() -> None:
    args = _parse_args()
    base_model_dir = Path(args.base_model_dir)
    base_iter = int(args.base_iteration)
    base_point_dir = base_model_dir / "point_cloud" / f"iteration_{base_iter}"
    base_ply = base_point_dir / "point_cloud.ply"
    if not base_ply.is_file():
        raise FileNotFoundError(f"Base PLY not found: {base_ply}")
    primitive_dir = Path(args.primitive_dir)
    output_model_dir = Path(args.output_model_dir)
    newborn_model_dir = Path(args.newborn_model_dir) if args.newborn_model_dir else output_model_dir / "newborn_only_model"
    for candidate in (output_model_dir, newborn_model_dir):
        if candidate.exists() and not bool(args.overwrite):
            raise FileExistsError(f"Output exists; pass --overwrite: {candidate}")
    if output_model_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_model_dir)
    if newborn_model_dir.exists() and bool(args.overwrite):
        shutil.rmtree(newborn_model_dir)

    primitive_paths = _list_files(primitive_dir, [".npz"])
    if int(args.limit) > 0:
        primitive_paths = primitive_paths[: int(args.limit)]
    if not primitive_paths:
        raise RuntimeError(f"No primitive npz files found in {primitive_dir}")
    rgb_paths = _list_files(Path(args.carrier_rgb_dir), IMAGE_EXTS) if args.carrier_rgb_dir else []
    render_paths = _list_files(Path(args.carrier_render_dir), IMAGE_EXTS) if args.carrier_render_dir else []
    weight_paths = _list_files(Path(args.carrier_weight_dir), IMAGE_EXTS) if args.carrier_weight_dir else []
    rgb_lookup = _lookup(rgb_paths)
    render_lookup = _lookup(render_paths)
    weight_lookup = _lookup(weight_paths)
    cameras = _load_cameras(base_model_dir)
    base_vertices, base_xyz, base_opacity, base_rgb = _load_base_vertices(base_ply)

    views: List[ViewPrimitiveSet] = []
    per_view: List[Dict[str, object]] = []
    print(f"[2dgs-posterior-v0] base      : {base_model_dir}")
    print(f"[2dgs-posterior-v0] primitives: {primitive_dir}")
    print(f"[2dgs-posterior-v0] output    : {output_model_dir}")
    for view_index, primitive_path in enumerate(primitive_paths):
        view, info = _load_view_primitives(
            primitive_path,
            view_index,
            args,
            cameras,
            base_xyz,
            base_opacity,
            rgb_paths,
            rgb_lookup,
            render_paths,
            render_lookup,
            weight_paths,
            weight_lookup,
        )
        per_view.append(info)
        if view is None:
            print(f"[2dgs-posterior-v0] skip {view_index + 1}/{len(primitive_paths)} {primitive_path.stem}: {info['status']}")
            continue
        views.append(view)
        print(
            f"[2dgs-posterior-v0] {view_index + 1}/{len(primitive_paths)} {primitive_path.stem} "
            f"surface={view.mu_xy.shape[0]} q={float(view.q.mean()):.4f} "
            f"mode={float(view.mode_dominance.mean()):.3f}/{float(view.mode_entropy.mean()):.3f}"
        )
    if not views:
        raise RuntimeError("No usable 2DGS primitive views were loaded.")

    clusters = _greedy_clusters(views, args)
    print(f"[2dgs-posterior-v0] clusters: {len(clusters)}")
    results = [_cluster_to_result(views, cluster, base_xyz, base_rgb, args) for cluster in clusters]
    confirmed = [res for res in results if res.status == "confirmed"]
    probation = [res for res in results if res.status == "probation"]
    newborn = _build_newborn_arrays(results)
    if not newborn:
        raise RuntimeError("No confirmed newborn gaussians were produced; inspect probation thresholds.")
    if int(args.max_total_newborn) > 0 and newborn["xyz"].shape[0] > int(args.max_total_newborn):
        newborn = _slice_newborn(newborn, int(args.max_total_newborn))

    base_count = int(base_vertices.shape[0])
    new_vertices = _fill_new_vertices(base_vertices.dtype, newborn)
    _copy_config(base_model_dir, newborn_model_dir)
    newborn_point_dir = newborn_model_dir / "point_cloud" / f"iteration_{base_iter}"
    newborn_point_dir.mkdir(parents=True, exist_ok=True)
    newborn_ply = newborn_point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(new_vertices, "vertex")]).write(str(newborn_ply))
    torch.save(_make_tracking(0, int(new_vertices.shape[0]), Path("__missing__"), base_iter), newborn_point_dir / "gaussian_tags.pt")
    metadata_name = "sprayed_2dgs_posterior_metadata_v0.npz"
    metadata_path = newborn_point_dir / metadata_name
    np.savez_compressed(metadata_path, **newborn)

    output_point_dir = output_model_dir / "point_cloud" / f"iteration_{base_iter}"
    output_point_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, output_point_dir / metadata_name)

    cpu_preview_ply = None
    cpu_preview_tags = None
    if bool(args.write_cpu_merged_preview):
        merged = np.empty((base_count + new_vertices.shape[0],), dtype=base_vertices.dtype)
        merged[:base_count] = base_vertices
        merged[base_count:] = new_vertices
        preview_point_dir = output_model_dir / "cpu_merged_preview" / "point_cloud" / f"iteration_{base_iter}"
        preview_point_dir.mkdir(parents=True, exist_ok=True)
        cpu_preview_ply = preview_point_dir / "point_cloud.ply"
        PlyData([PlyElement.describe(merged, "vertex")]).write(str(cpu_preview_ply))
        cpu_preview_tags = preview_point_dir / "gaussian_tags.pt"
        torch.save(_make_tracking(base_count, int(new_vertices.shape[0]), base_point_dir / "gaussian_tags.pt", base_iter), cpu_preview_tags)

    summary = {
        "version": "spray_2dgs_posterior_to_gaussian_layer_v0",
        "base_model_dir": str(base_model_dir),
        "base_iteration": base_iter,
        "base_ply": str(base_ply),
        "primitive_dir": str(primitive_dir),
        "carrier_rgb_dir": str(args.carrier_rgb_dir),
        "carrier_render_dir": str(args.carrier_render_dir),
        "carrier_weight_dir": str(args.carrier_weight_dir),
        "output_model_dir": str(output_model_dir),
        "newborn_model_dir": str(newborn_model_dir),
        "newborn_ply": str(newborn_ply),
        "newborn_metadata": str(metadata_path),
        "merged_metadata": str(output_point_dir / metadata_name),
        "cpu_preview_ply": None if cpu_preview_ply is None else str(cpu_preview_ply),
        "cpu_preview_tags": None if cpu_preview_tags is None else str(cpu_preview_tags),
        "base_gaussians": base_count,
        "confirmed_newborn_gaussians": int(new_vertices.shape[0]),
        "probation_clusters": int(len(probation)),
        "total_clusters": int(len(results)),
        "expected_total_after_merge": int(base_count + new_vertices.shape[0]),
        "num_views": len(views),
        "per_view": per_view,
        "params": vars(args),
        "stats": {
            "confirmed_score_mean": float(np.mean([r.score for r in confirmed])) if confirmed else float("nan"),
            "confirmed_reproj_rms_mean": float(np.mean([r.reproj_rms for r in confirmed])) if confirmed else float("nan"),
            "confirmed_center_std_mean": float(np.mean([r.max_center_std for r in confirmed])) if confirmed else float("nan"),
            "confirmed_cluster_size_mean": float(np.mean([len(r.obs) for r in confirmed])) if confirmed else float("nan"),
            "probation_reproj_rms_mean": float(np.mean([r.reproj_rms for r in probation])) if probation else float("nan"),
            "scale_mean": [float(x) for x in newborn["scale"].mean(axis=0)],
            "scale_p90": [float(x) for x in np.percentile(newborn["scale"], 90, axis=0)],
            "opacity_mean": float(newborn["opacity"].mean()),
            "color_mean": [float(x) for x in newborn["color"].mean(axis=0)],
        },
        "clusters": _result_records(results[:2000], views),
    }
    output_model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_model_dir / "spray_2dgs_posterior_to_gaussian_layer_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "confirmed_newborn_gaussians": summary["confirmed_newborn_gaussians"],
                "probation_clusters": summary["probation_clusters"],
                "expected_total_after_merge": summary["expected_total_after_merge"],
                "newborn_ply": str(newborn_ply),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
