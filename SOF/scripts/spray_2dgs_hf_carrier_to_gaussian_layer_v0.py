#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from spray_2dgs_hf_carrier_to_3d_v0 import (
    IMAGE_EXTS,
    SH_C0,
    _bilinear_rgb,
    _bilinear_scalar,
    _camera_basis,
    _cholesky_axes,
    _copy_config,
    _fill_new_vertices,
    _infer_source_size,
    _list_files,
    _load_cameras,
    _load_gray,
    _load_rgb,
    _lookup,
    _make_tracking,
    _matrix_to_quaternion_wxyz,
    _normalize,
    _resolve,
    _unproject_points,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift 2DGS SR-HF carrier primitives onto the currently visible 3DGS layer. "
            "Unlike the mesh-depth spray path, this uses projected base Gaussian centers as the depth carrier, "
            "so newborn splats compete near the existing rendered layer instead of a possibly hidden mesh layer."
        )
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--primitive_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--newborn_model_dir", default="")
    parser.add_argument("--carrier_rgb_dir", default="")
    parser.add_argument("--carrier_weight_dir", default="")
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_primitives_per_view", type=int, default=65536)
    parser.add_argument("--max_total_newborn", type=int, default=0)
    parser.add_argument("--min_weight", type=float, default=0.01)
    parser.add_argument("--min_primitive_opacity", type=float, default=0.0)
    parser.add_argument("--base_opacity_min", type=float, default=0.02)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--search_radius_px", type=int, default=5)
    parser.add_argument("--front_offset_px", type=float, default=0.35)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--scale_min", type=float, default=5e-4)
    parser.add_argument("--scale_max", type=float, default=1.2e-2)
    parser.add_argument("--normal_scale_ratio", type=float, default=0.35)
    parser.add_argument("--normal_scale_min", type=float, default=4e-4)
    parser.add_argument("--normal_scale_max", type=float, default=3e-3)
    parser.add_argument("--opacity_floor", type=float, default=0.02)
    parser.add_argument("--opacity_scale", type=float, default=0.12)
    parser.add_argument("--opacity_power", type=float, default=0.75)
    parser.add_argument("--opacity_min", type=float, default=0.01)
    parser.add_argument("--opacity_max", type=float, default=0.16)
    parser.add_argument(
        "--color_mode",
        default="primitive",
        choices=["primitive", "base_anchor_additive"],
        help=(
            "primitive uses exported 2DGS primitive RGB as ordinary radiance. "
            "base_anchor_additive anchors each newborn Gaussian to its matched base Gaussian DC color and "
            "adds a bounded positive HF carrier term, avoiding black-background alpha-over darkening."
        ),
    )
    parser.add_argument("--color_gain", type=float, default=0.35)
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
        dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(dc * float(SH_C0) + 0.5, 0.0, 1.0).astype(np.float32)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    return vertices, xyz, opacity, rgb


def _project_base_to_source(
    xyz: np.ndarray,
    opacity: np.ndarray,
    cam: Dict[str, object],
    source_size: Tuple[int, int],
    base_opacity_min: float,
    depth_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
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
    valid = (
        np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(depth)
        & (depth > float(depth_min))
        & (px >= 0.0)
        & (py >= 0.0)
        & (px <= float(sw - 1))
        & (py <= float(sh - 1))
        & (opacity >= float(base_opacity_min))
    )
    return px.astype(np.float32), py.astype(np.float32), depth.astype(np.float32), (sw, sh), valid


def _build_visible_layer_index(
    base_xy_depth: Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    px, py, depth, source_size, valid = base_xy_depth
    sw, sh = source_size
    grid_count = int(sw) * int(sh)
    grid_index = np.full((grid_count,), -1, dtype=np.int64)
    grid_depth = np.full((grid_count,), np.inf, dtype=np.float32)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return grid_index.reshape(sh, sw), grid_depth.reshape(sh, sw)
    xi = np.rint(px[valid_ids]).astype(np.int64)
    yi = np.rint(py[valid_ids]).astype(np.int64)
    inside = (xi >= 0) & (xi < sw) & (yi >= 0) & (yi < sh)
    valid_ids = valid_ids[inside]
    xi = xi[inside]
    yi = yi[inside]
    if valid_ids.size == 0:
        return grid_index.reshape(sh, sw), grid_depth.reshape(sh, sw)
    lin = yi * int(sw) + xi
    d = depth[valid_ids]
    order = np.lexsort((d, lin))
    sorted_lin = lin[order]
    first = np.concatenate(([True], sorted_lin[1:] != sorted_lin[:-1]))
    winners = order[first]
    win_lin = lin[winners]
    grid_index[win_lin] = valid_ids[winners]
    grid_depth[win_lin] = d[winners]
    return grid_index.reshape(sh, sw), grid_depth.reshape(sh, sw)


def _match_primitives_to_visible_layer(
    mu: np.ndarray,
    grid_index: np.ndarray,
    grid_depth: np.ndarray,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid_index.shape
    n = int(mu.shape[0])
    best_index = np.full((n,), -1, dtype=np.int64)
    best_depth = np.full((n,), np.inf, dtype=np.float32)
    best_dist2 = np.full((n,), np.inf, dtype=np.float32)
    radius = max(int(radius), 0)
    offsets: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    center_x = np.rint(mu[:, 0]).astype(np.int64)
    center_y = np.rint(mu[:, 1]).astype(np.int64)
    for dx, dy in offsets:
        xs = center_x + int(dx)
        ys = center_y + int(dy)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(inside):
            continue
        idx = np.full((n,), -1, dtype=np.int64)
        dep = np.full((n,), np.inf, dtype=np.float32)
        idx[inside] = grid_index[ys[inside], xs[inside]]
        hit = idx >= 0
        if not np.any(hit):
            continue
        dep[hit] = grid_depth[ys[hit], xs[hit]]
        dist2 = (xs.astype(np.float32) - mu[:, 0]) ** 2 + (ys.astype(np.float32) - mu[:, 1]) ** 2
        update = hit & ((dist2 < best_dist2) | ((dist2 == best_dist2) & (dep < best_depth)))
        best_index[update] = idx[update]
        best_depth[update] = dep[update]
        best_dist2[update] = dist2[update]
    matched = best_index >= 0
    return matched, best_index, best_depth, np.sqrt(np.maximum(best_dist2, 0.0)).astype(np.float32)


def _camera_for_view(
    cameras: Dict[str, Dict[str, object]],
    stem: str,
    view_index: int,
    match_policy: str,
) -> Tuple[Optional[Dict[str, object]], str]:
    cam = cameras.get(stem.lower())
    if cam is not None:
        return cam, "stem"
    if match_policy == "stem":
        return None, "missing"
    camera_values = list(cameras.values())
    if view_index >= len(camera_values):
        return None, "missing"
    return camera_values[view_index], "order"


def _load_primitive_colors(
    primitive: Dict[str, np.ndarray],
    mu: np.ndarray,
    final_ids: np.ndarray,
    rgb_path: Optional[Path],
    source_size: Tuple[int, int],
) -> np.ndarray:
    color = np.asarray(primitive["color"], dtype=np.float32)
    kept_color = np.clip(color[final_ids], 0.0, 1.0)
    if rgb_path is not None and rgb_path.is_file():
        rgb_img = _load_rgb(rgb_path, size=source_size)
        sampled = _bilinear_rgb(rgb_img, mu, source_size)
        low_color = np.mean(np.abs(kept_color), axis=1, keepdims=True) < 1e-4
        kept_color = np.where(low_color, sampled, kept_color)
    return np.clip(kept_color, 0.0, 1.0).astype(np.float32)


def _resolve_newborn_colors(
    primitive: Dict[str, np.ndarray],
    mu: np.ndarray,
    final_ids: np.ndarray,
    matched_base_idx: np.ndarray,
    base_rgb: np.ndarray,
    rgb_path: Optional[Path],
    source_size: Tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    primitive_color = _load_primitive_colors(primitive, mu, final_ids, rgb_path, source_size)
    mode = str(args.color_mode).strip().lower()
    if mode == "primitive":
        return primitive_color
    if mode == "base_anchor_additive":
        anchor = np.clip(base_rgb[matched_base_idx], 0.0, 1.0)
        # The 2DGS evidence image is a black-background carrier: black means no HF evidence,
        # not black radiance. Anchor to the matched 3D Gaussian and add only the visible carrier energy.
        carrier = np.clip(primitive_color, 0.0, 1.0)
        return np.clip(anchor + float(args.color_gain) * carrier, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported color_mode={args.color_mode!r}")


def _spray_view(
    primitive_path: Path,
    view_index: int,
    args: argparse.Namespace,
    cameras: Dict[str, Dict[str, object]],
    base_xyz: np.ndarray,
    base_opacity: np.ndarray,
    base_rgb: np.ndarray,
    rgb_paths: Sequence[Path],
    rgb_lookup: Dict[str, Path],
    weight_paths: Sequence[Path],
    weight_lookup: Dict[str, Path],
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, object]]:
    stem = primitive_path.stem
    primitive = dict(np.load(primitive_path))
    if "mu_xy" not in primitive or "color" not in primitive:
        raise KeyError(f"{primitive_path} must contain mu_xy and color")
    mu_all = np.asarray(primitive["mu_xy"], dtype=np.float32)
    opacity_2d = np.asarray(primitive.get("opacity", np.ones((mu_all.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)

    cam, camera_match = _camera_for_view(cameras, stem, view_index, str(args.match_policy))
    if cam is None:
        return None, {"stem": stem, "status": "missing_camera"}

    rgb_path = _resolve(rgb_paths, rgb_lookup, stem, view_index, args.match_policy) if rgb_paths else None
    weight_path = _resolve(weight_paths, weight_lookup, stem, view_index, args.match_policy) if weight_paths else None
    source_size = _infer_source_size(primitive, rgb_path, weight_path)
    weight = np.ones((mu_all.shape[0],), dtype=np.float32)
    if weight_path is not None and weight_path.is_file():
        weight_img = _load_gray(weight_path, size=source_size)
        weight = _bilinear_scalar(weight_img, mu_all, source_size)

    finite = np.isfinite(mu_all).all(axis=1) & np.isfinite(opacity_2d)
    finite &= weight >= float(args.min_weight)
    finite &= opacity_2d >= float(args.min_primitive_opacity)
    keep_ids = np.flatnonzero(finite)
    if keep_ids.size == 0:
        return None, {"stem": stem, "status": "empty_after_weight_filter"}
    score = weight[keep_ids] * np.clip(opacity_2d[keep_ids], 0.0, 1.0)
    order = np.argsort(score)[::-1]
    if int(args.max_primitives_per_view) > 0:
        order = order[: int(args.max_primitives_per_view)]
    keep_ids = keep_ids[order]
    candidate_mu = mu_all[keep_ids]

    visible_projection = _project_base_to_source(
        base_xyz,
        base_opacity,
        cam,
        source_size,
        float(args.base_opacity_min),
        float(args.depth_min),
    )
    grid_index, grid_depth = _build_visible_layer_index(visible_projection)
    matched, matched_base_idx, matched_depth, matched_dist = _match_primitives_to_visible_layer(
        candidate_mu,
        grid_index,
        grid_depth,
        int(args.search_radius_px),
    )
    if not np.any(matched):
        return None, {
            "stem": stem,
            "status": "empty_after_gaussian_layer_match",
            "candidate": int(candidate_mu.shape[0]),
            "source_size": [int(source_size[0]), int(source_size[1])],
        }
    final_ids = keep_ids[matched]
    kept_mu = candidate_mu[matched]
    kept_depth = matched_depth[matched]
    kept_weight = weight[final_ids]
    kept_opacity_2d = np.clip(opacity_2d[final_ids], 0.0, 1.0)
    kept_matched_base_idx = matched_base_idx[matched]
    kept_color = _resolve_newborn_colors(
        primitive,
        kept_mu,
        final_ids,
        kept_matched_base_idx,
        base_rgb,
        rgb_path,
        source_size,
        args,
    )
    xyz, cam_x, cam_y, normal, _px, fx, fy = _unproject_points(cam, kept_mu, kept_depth, source_size)
    pix_world = kept_depth / max((float(fx) + float(fy)) * 0.5, 1e-8)
    front_offset = np.maximum(float(args.front_offset_px), 0.0) * pix_world
    xyz = (xyz - normal * front_offset[:, None]).astype(np.float32)

    cholesky = np.asarray(primitive["cholesky"], dtype=np.float32)[final_ids] if "cholesky" in primitive else None
    theta, long_px, short_px = _cholesky_axes(cholesky, int(final_ids.size))
    c = np.cos(theta).astype(np.float32)
    s = np.sin(theta).astype(np.float32)
    long_axis = _normalize(c[:, None] * cam_x[None, :] + s[:, None] * cam_y[None, :])
    short_axis = _normalize(-s[:, None] * cam_x[None, :] + c[:, None] * cam_y[None, :])
    normal_axis = _normalize(normal)
    rot_mats = np.stack([long_axis, short_axis, normal_axis], axis=2)
    quats = _matrix_to_quaternion_wxyz(rot_mats)

    scale_long = long_px * pix_world * float(args.scale_multiplier)
    scale_short = short_px * pix_world * float(args.scale_multiplier)
    scale_long = np.clip(scale_long, float(args.scale_min), float(args.scale_max))
    scale_short = np.clip(scale_short, float(args.scale_min), float(args.scale_max))
    scale_normal = np.clip(
        scale_short * float(args.normal_scale_ratio),
        float(args.normal_scale_min),
        float(args.normal_scale_max),
    )
    scale = np.stack([scale_long, scale_short, scale_normal], axis=1).astype(np.float32)

    opacity = (
        float(args.opacity_floor)
        + float(args.opacity_scale) * np.power(np.clip(kept_weight, 0.0, 1.0), float(args.opacity_power)) * kept_opacity_2d
    )
    opacity = np.clip(opacity, float(args.opacity_min), float(args.opacity_max)).astype(np.float32)
    newborn = {
        "xyz": xyz.astype(np.float32),
        "color": kept_color.astype(np.float32),
        "opacity": opacity.reshape(-1, 1),
        "scale": scale,
        "rotation": quats,
        "source_view": np.full((final_ids.size,), int(view_index), dtype=np.int32),
        "source_primitive": final_ids.astype(np.int64),
        "matched_base_index": kept_matched_base_idx.astype(np.int64),
        "matched_depth": kept_depth.astype(np.float32),
        "matched_pixel_distance": matched_dist[matched].astype(np.float32),
        "weight": kept_weight.astype(np.float32),
        "primitive_opacity": kept_opacity_2d.astype(np.float32),
    }
    visible_valid = visible_projection[-1]
    info = {
        "stem": stem,
        "status": "ok",
        "selected": int(final_ids.size),
        "candidate": int(candidate_mu.shape[0]),
        "available": int(mu_all.shape[0]),
        "camera_match": camera_match,
        "camera_index": int(cam.get("_index", view_index)),
        "source_size": [int(source_size[0]), int(source_size[1])],
        "visible_base_gaussians": int(np.count_nonzero(visible_valid)),
        "weight_mean": float(kept_weight.mean()),
        "opacity_mean": float(opacity.mean()),
        "matched_depth_mean": float(kept_depth.mean()),
        "matched_pixel_distance_mean": float(matched_dist[matched].mean()),
        "scale_long_mean": float(scale_long.mean()),
        "scale_short_mean": float(scale_short.mean()),
        "color_mode": str(args.color_mode),
        "color_gain": float(args.color_gain),
    }
    return newborn, info


def _slice_chunk(chunk: Dict[str, np.ndarray], count: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in chunk.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (chunk["xyz"].shape[0],):
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
    rgb_lookup = _lookup(rgb_paths)
    weight_paths = _list_files(Path(args.carrier_weight_dir), IMAGE_EXTS) if args.carrier_weight_dir else []
    weight_lookup = _lookup(weight_paths)
    cameras = _load_cameras(base_model_dir)
    base_vertices, base_xyz, base_opacity, base_rgb = _load_base_vertices(base_ply)

    per_view: List[Dict[str, object]] = []
    chunks: List[Dict[str, np.ndarray]] = []
    total_new = 0
    for view_index, primitive_path in enumerate(primitive_paths):
        newborn, info = _spray_view(
            primitive_path,
            view_index,
            args,
            cameras,
            base_xyz,
            base_opacity,
            base_rgb,
            rgb_paths,
            rgb_lookup,
            weight_paths,
            weight_lookup,
        )
        per_view.append(info)
        if newborn is None:
            print(f"[spray-2dgs-to-gaussian-layer-v0] skip {view_index + 1}/{len(primitive_paths)} {primitive_path.stem}: {info['status']}")
            continue
        remaining = int(args.max_total_newborn) - total_new if int(args.max_total_newborn) > 0 else None
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and newborn["xyz"].shape[0] > remaining:
            newborn = _slice_chunk(newborn, remaining)
            info["selected_after_global_cap"] = int(remaining)
        chunks.append(newborn)
        total_new += int(newborn["xyz"].shape[0])
        print(
            f"[spray-2dgs-to-gaussian-layer-v0] {view_index + 1}/{len(primitive_paths)} {primitive_path.stem} "
            f"new={newborn['xyz'].shape[0]} total_new={total_new}"
        )
    if not chunks:
        raise RuntimeError("No newborn gaussians were produced.")

    concat_keys = (
        "xyz",
        "color",
        "opacity",
        "scale",
        "rotation",
        "source_view",
        "source_primitive",
        "matched_base_index",
        "matched_depth",
        "matched_pixel_distance",
        "weight",
        "primitive_opacity",
    )
    newborn_all: Dict[str, np.ndarray] = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in concat_keys}

    base_count = int(base_vertices.shape[0])
    new_vertices = _fill_new_vertices(base_vertices.dtype, newborn_all)
    _copy_config(base_model_dir, newborn_model_dir)
    newborn_point_dir = newborn_model_dir / "point_cloud" / f"iteration_{base_iter}"
    newborn_point_dir.mkdir(parents=True, exist_ok=True)
    newborn_ply = newborn_point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(new_vertices, "vertex")]).write(str(newborn_ply))

    newborn_tags = _make_tracking(0, int(new_vertices.shape[0]), Path("__missing__"), base_iter)
    torch.save(newborn_tags, newborn_point_dir / "gaussian_tags.pt")
    metadata_name = "sprayed_2dgs_gaussian_layer_metadata_v0.npz"
    newborn_metadata_path = newborn_point_dir / metadata_name
    np.savez_compressed(newborn_metadata_path, **newborn_all)

    # Keep a copy next to the merged model as well, so cleanup of *_newborn_only
    # does not break later survival/validation passes.
    output_point_dir = output_model_dir / "point_cloud" / f"iteration_{base_iter}"
    output_point_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path = output_point_dir / metadata_name
    shutil.copy2(newborn_metadata_path, output_metadata_path)

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
        tags = _make_tracking(base_count, int(new_vertices.shape[0]), base_point_dir / "gaussian_tags.pt", base_iter)
        cpu_preview_tags = preview_point_dir / "gaussian_tags.pt"
        torch.save(tags, cpu_preview_tags)

    output_model_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "spray_2dgs_hf_carrier_to_gaussian_layer_v0",
        "mode": "newborn_only_for_gaussianmodel_merge",
        "base_model_dir": str(base_model_dir),
        "base_iteration": base_iter,
        "base_ply": str(base_ply),
        "primitive_dir": str(primitive_dir),
        "carrier_rgb_dir": str(args.carrier_rgb_dir),
        "carrier_weight_dir": str(args.carrier_weight_dir),
        "output_model_dir": str(output_model_dir),
        "newborn_model_dir": str(newborn_model_dir),
        "newborn_ply": str(newborn_ply),
        "newborn_tags": str(newborn_point_dir / "gaussian_tags.pt"),
        "newborn_metadata": str(newborn_metadata_path),
        "merged_metadata": str(output_metadata_path),
        "cpu_preview_ply": None if cpu_preview_ply is None else str(cpu_preview_ply),
        "cpu_preview_tags": None if cpu_preview_tags is None else str(cpu_preview_tags),
        "base_gaussians": base_count,
        "newborn_gaussians": int(new_vertices.shape[0]),
        "expected_total_after_merge": int(base_count + new_vertices.shape[0]),
        "num_views": len(primitive_paths),
        "per_view": per_view,
        "params": {
            "max_primitives_per_view": int(args.max_primitives_per_view),
            "max_total_newborn": int(args.max_total_newborn),
            "min_weight": float(args.min_weight),
            "min_primitive_opacity": float(args.min_primitive_opacity),
            "base_opacity_min": float(args.base_opacity_min),
            "depth_min": float(args.depth_min),
            "search_radius_px": int(args.search_radius_px),
            "front_offset_px": float(args.front_offset_px),
            "scale_multiplier": float(args.scale_multiplier),
            "scale_min": float(args.scale_min),
            "scale_max": float(args.scale_max),
            "opacity_floor": float(args.opacity_floor),
            "opacity_scale": float(args.opacity_scale),
            "opacity_max": float(args.opacity_max),
            "color_mode": str(args.color_mode),
            "color_gain": float(args.color_gain),
        },
        "newborn_stats": {
            "weight_mean": float(newborn_all["weight"].mean()),
            "opacity_mean": float(newborn_all["opacity"].mean()),
            "matched_depth_mean": float(newborn_all["matched_depth"].mean()),
            "matched_pixel_distance_mean": float(newborn_all["matched_pixel_distance"].mean()),
            "scale_mean": [float(x) for x in newborn_all["scale"].mean(axis=0)],
            "scale_p90": [float(x) for x in np.percentile(newborn_all["scale"], 90, axis=0)],
            "color_mean": [float(x) for x in newborn_all["color"].mean(axis=0)],
        },
    }
    summary_path = output_model_dir / "spray_2dgs_hf_carrier_to_gaussian_layer_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "base_gaussians": summary["base_gaussians"],
                "newborn_gaussians": summary["newborn_gaussians"],
                "expected_total_after_merge": summary["expected_total_after_merge"],
                "newborn_model_dir": summary["newborn_model_dir"],
                "newborn_ply": summary["newborn_ply"],
            },
            indent=2,
        )
    )
    print(f"[spray-2dgs-to-gaussian-layer-v0] summary: {summary_path}")


if __name__ == "__main__":
    main()
