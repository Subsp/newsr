#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable[object], **_: object) -> Iterable[object]:
        return iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SH_C0 = 0.28209479177387814


KIND_GEOMETRY = 1
KIND_TEXTURE = 2
KIND_NOISE = 3


def _list_files(root: Path, exts: Sequence[str]) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    ext_set = {x.lower() for x in exts}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ext_set)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(paths: Sequence[Path], lookup: Dict[str, Path], stem: str, index: int, policy: str) -> Optional[Path]:
    if policy in {"stem", "order_if_needed"}:
        found = lookup.get(stem.lower())
        if found is not None:
            return found
        if policy == "stem":
            return None
    if policy in {"order", "order_if_needed"} and index < len(paths):
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


def _rgb_luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    ).astype(np.float32)


def _blur_gray(image: np.ndarray, radius: float) -> np.ndarray:
    radius = max(float(radius), 0.0)
    if radius <= 1e-6:
        return image.astype(np.float32)
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return (np.asarray(pil, dtype=np.float32) / 255.0).astype(np.float32)


def _curve_strength_from_image(curve_rgb: Optional[np.ndarray], gate_weight: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mode = str(args.curve_image_mode)
    if mode == "weight":
        strength = gate_weight.astype(np.float32)
    else:
        if curve_rgb is None:
            raise ValueError(f"curve_image_mode={mode} requires --curve_image_dir")
        luma = _rgb_luma(curve_rgb)
        if mode == "sr_hf_luma":
            low = _blur_gray(luma, float(args.curve_highpass_blur_radius))
            strength = np.abs(luma - low).astype(np.float32)
            positive = strength[np.isfinite(strength) & (strength > 0)]
            if positive.size:
                scale = max(float(np.percentile(positive, 99.0)), 1e-6)
                strength = strength / scale
            strength = np.clip(strength, 0.0, 1.0).astype(np.float32)
        elif mode == "luma":
            strength = np.clip(luma, 0.0, 1.0).astype(np.float32)
        else:  # argparse choices should prevent this.
            raise ValueError(f"Unsupported curve_image_mode: {mode}")
    gate = np.clip(gate_weight.astype(np.float32), 0.0, 1.0)
    power = float(args.curve_weight_power)
    if abs(power - 1.0) > 1e-8:
        gate = np.power(gate, power).astype(np.float32)
    return np.clip(strength * gate, 0.0, 1.0).astype(np.float32)


def _load_cameras(base_model_dir: Path) -> Dict[str, Dict[str, object]]:
    path = base_model_dir / "cameras.json"
    if not path.is_file():
        raise FileNotFoundError(f"cameras.json not found: {path}")
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list):
        raise ValueError(f"Expected cameras.json list, got {type(cameras).__name__}")
    out: Dict[str, Dict[str, object]] = {}
    for index, cam in enumerate(cameras):
        name = str(cam.get("img_name", cam.get("image_name", "")))
        if not name:
            continue
        item = dict(cam)
        item["_index"] = index
        out[Path(name).stem.lower()] = item
    return out


def _camera_basis(cam: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    rot = np.asarray(cam["rotation"], dtype=np.float32)
    pos = np.asarray(cam["position"], dtype=np.float32)
    if rot.shape != (3, 3):
        raise ValueError(f"Invalid camera rotation shape: {rot.shape}")
    if pos.shape != (3,):
        raise ValueError(f"Invalid camera position shape: {pos.shape}")
    return pos, rot[:, 0], rot[:, 1], rot[:, 2], float(cam["fx"]), float(cam["fy"]), int(cam["width"]), int(cam["height"])


def _unproject_points(
    cam: Dict[str, object],
    xy: np.ndarray,
    depth: np.ndarray,
    source_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    pos, cam_x, cam_y, cam_z, fx, fy, cam_w, cam_h = _camera_basis(cam)
    sw, sh = source_size
    px = xy[:, 0] / max(float(sw - 1), 1.0) * max(float(cam_w - 1), 1.0)
    py = xy[:, 1] / max(float(sh - 1), 1.0) * max(float(cam_h - 1), 1.0)
    x_cam = (px - float(cam_w) * 0.5) / max(fx, 1e-8) * depth
    y_cam = (py - float(cam_h) * 0.5) / max(fy, 1e-8) * depth
    xyz = (
        pos[None, :]
        + x_cam[:, None] * cam_x[None, :]
        + y_cam[:, None] * cam_y[None, :]
        + depth[:, None] * cam_z[None, :]
    ).astype(np.float32)
    normal = xyz - pos[None, :]
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    return xyz, cam_x.astype(np.float32), cam_y.astype(np.float32), normal.astype(np.float32), px.astype(np.float32), fx, fy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _load_base_vertices(base_ply: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from plyfile import PlyData

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
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
    grid_index = np.full((int(sw) * int(sh),), -1, dtype=np.int64)
    grid_depth = np.full((int(sw) * int(sh),), np.inf, dtype=np.float32)
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
    offsets: List[Tuple[int, int]] = []
    r = max(int(radius), 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift SR-HF 2D primitive evidence onto the visible Gaussian layer and merge it into "
            "LIMAP-style 3D curve/line tracks. This is intentionally a cache builder: it does not "
            "modify the Gaussian field yet."
        )
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--primitive_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--weight_dir", default="")
    parser.add_argument("--rgb_dir", default="")
    parser.add_argument("--curve_image_dir", default="")
    parser.add_argument("--curve_image_mode", default="sr_hf_luma", choices=["sr_hf_luma", "luma", "weight"])
    parser.add_argument("--curve_highpass_blur_radius", type=float, default=4.0)
    parser.add_argument("--curve_weight_power", type=float, default=1.0)
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--curve_source", default="skeleton", choices=["skeleton", "primitive"])
    parser.add_argument("--skeleton_threshold_percentile", type=float, default=86.0)
    parser.add_argument("--skeleton_min_weight", type=float, default=0.025)
    parser.add_argument("--skeleton_min_path_pixels", type=int, default=8)
    parser.add_argument("--skeleton_sample_step_px", type=float, default=3.0)
    parser.add_argument("--skeleton_smooth_window", type=int, default=3)
    parser.add_argument("--skeleton_max_thinning_iters", type=int, default=80)
    parser.add_argument("--dense_stroke_enable", action="store_true")
    parser.add_argument("--dense_stroke_threshold_percentile", type=float, default=78.0)
    parser.add_argument("--dense_stroke_min_strength", type=float, default=0.012)
    parser.add_argument("--dense_stroke_grid_px", type=int, default=2)
    parser.add_argument("--dense_stroke_max_per_view", type=int, default=32768)
    parser.add_argument("--dense_stroke_length_px", type=float, default=4.0)
    parser.add_argument("--dense_stroke_short_px", type=float, default=0.55)
    parser.add_argument("--profile_width_enable", action="store_true")
    parser.add_argument("--profile_width_radius_px", type=int, default=6)
    parser.add_argument("--profile_width_falloff", type=float, default=0.35)
    parser.add_argument("--profile_width_min_px", type=float, default=0.4)
    parser.add_argument("--profile_width_max_px", type=float, default=5.0)
    parser.add_argument("--keep_kinds", default="1,2", help="Comma-separated primitive kind ids. Defaults to geometry+texture.")
    parser.add_argument("--max_primitives_per_view", type=int, default=32768)
    parser.add_argument("--min_score", type=float, default=0.05)
    parser.add_argument("--min_weight", type=float, default=0.01)
    parser.add_argument("--base_opacity_min", type=float, default=0.02)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--search_radius_px", type=int, default=5)
    parser.add_argument("--endpoint_search_radius_px", type=int, default=3)
    parser.add_argument("--require_endpoint_match", action="store_true")
    parser.add_argument("--max_endpoint_depth_delta_px", type=float, default=8.0)
    parser.add_argument("--front_offset_px", type=float, default=0.25)
    parser.add_argument("--segment_length_scale", type=float, default=2.5)
    parser.add_argument("--segment_min_length_px", type=float, default=2.0)
    parser.add_argument("--segment_max_length_px", type=float, default=18.0)

    parser.add_argument("--max_segments_for_merge", type=int, default=50000)
    parser.add_argument("--merge_radius_px", type=float, default=6.0)
    parser.add_argument("--merge_radius_abs", type=float, default=0.006)
    parser.add_argument("--merge_angle_deg", type=float, default=18.0)
    parser.add_argument("--merge_min_overlap", type=float, default=0.05)
    parser.add_argument("--merge_same_view", action="store_true")
    parser.add_argument("--layer_bin_radius_px", type=float, default=8.0)
    parser.add_argument("--layer_bin_radius_abs", type=float, default=0.008)
    parser.add_argument("--layer_dir_bins", type=int, default=8)
    parser.add_argument("--layer_include_kind", action="store_true")
    parser.add_argument("--candidate_radius_px", type=float, default=6.0)
    parser.add_argument("--candidate_radius_abs", type=float, default=0.006)
    parser.add_argument("--candidate_reproj_radius_px", type=float, default=5.0)
    parser.add_argument("--candidate_dir_angle_deg", type=float, default=25.0)
    parser.add_argument("--candidate_normal_angle_deg", type=float, default=60.0)
    parser.add_argument("--candidate_depth_delta_px", type=float, default=12.0)
    parser.add_argument("--candidate_min_survive_views", type=int, default=2)
    parser.add_argument("--candidate_probation_min_source_strength", type=float, default=0.04)
    parser.add_argument("--candidate_probation_max_line_residual_px", type=float, default=8.0)
    parser.add_argument("--candidate_keep_probation", action="store_true")
    parser.add_argument("--track_build_mode", default="source_segments", choices=["source_segments", "merge", "layer_bins", "candidate_graph"])
    parser.add_argument("--min_track_segments", type=int, default=2)
    parser.add_argument("--min_track_views", type=int, default=1)
    parser.add_argument("--strong_track_min_views", type=int, default=2)
    parser.add_argument("--track_min_dir_consistency", type=float, default=0.0)
    parser.add_argument("--track_max_line_residual_px", type=float, default=0.0)
    parser.add_argument("--debug_limit", type=int, default=8)
    parser.add_argument("--max_draw_segments", type=int, default=32768)
    return parser.parse_args()


def _parse_kind_set(text: str) -> set[int]:
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _infer_source_size(
    primitive: Dict[str, np.ndarray],
    rgb_path: Optional[Path],
    weight_path: Optional[Path],
) -> Tuple[int, int]:
    for path in (rgb_path, weight_path):
        if path is not None and path.is_file():
            with Image.open(path) as image:
                return image.size
    xy = np.asarray(primitive["xy"], dtype=np.float32)
    width = int(math.ceil(float(np.nanmax(xy[:, 0])) + 1.0)) if xy.size else 1
    height = int(math.ceil(float(np.nanmax(xy[:, 1])) + 1.0)) if xy.size else 1
    return max(width, 1), max(height, 1)


def _bilinear_scalar(image: np.ndarray, xy: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    sw, sh = size
    x = np.clip(xy[:, 0].astype(np.float32), 0.0, float(sw - 1))
    y = np.clip(xy[:, 1].astype(np.float32), 0.0, float(sh - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, sw - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    wx = (x - x0.astype(np.float32))[:, None]
    wy = (y - y0.astype(np.float32))[:, None]
    v00 = image[y0, x0].reshape(-1, 1)
    v10 = image[y0, x1].reshape(-1, 1)
    v01 = image[y1, x0].reshape(-1, 1)
    v11 = image[y1, x1].reshape(-1, 1)
    v0 = v00 * (1.0 - wx) + v10 * wx
    v1 = v01 * (1.0 - wx) + v11 * wx
    return (v0 * (1.0 - wy) + v1 * wy).reshape(-1).astype(np.float32)


def _bilinear_rgb(image: np.ndarray, xy: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.stack([_bilinear_scalar(image[..., c], xy, size) for c in range(3)], axis=1).astype(np.float32)


def _thin_binary_mask(mask: np.ndarray, max_iters: int) -> np.ndarray:
    """Zhang-Suen thinning. This keeps us dependency-light on the server."""
    img = (mask.astype(np.uint8) > 0).copy()
    if img.size == 0:
        return img.astype(bool)
    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0
    for _ in range(max(1, int(max_iters))):
        changed = False
        for sub_iter in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            center = img[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            n_count = sum(neighbors)
            transitions = np.zeros_like(center, dtype=np.uint8)
            for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]):
                transitions += ((a == 0) & (b == 1)).astype(np.uint8)
            if sub_iter == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8
            remove = (
                (center == 1)
                & (n_count >= 2)
                & (n_count <= 6)
                & (transitions == 1)
                & (m1 == 0)
                & (m2 == 0)
            )
            if np.any(remove):
                center[remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _skeleton_neighbors(point: Tuple[int, int], skel: np.ndarray) -> List[Tuple[int, int]]:
    x, y = point
    h, w = skel.shape
    out: List[Tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx = x + dx
            yy = y + dy
            if 0 <= xx < w and 0 <= yy < h and bool(skel[yy, xx]):
                out.append((xx, yy))
    return out


def _trace_skeleton_paths(skel: np.ndarray, min_pixels: int) -> List[np.ndarray]:
    points_yx = np.argwhere(skel)
    if points_yx.size == 0:
        return []
    points = [(int(x), int(y)) for y, x in points_yx]
    point_set = set(points)
    degree = {p: len(_skeleton_neighbors(p, skel)) for p in points}
    starts = [p for p in points if degree[p] != 2]
    visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    def walk(start: Tuple[int, int], nxt: Tuple[int, int]) -> np.ndarray:
        path = [start, nxt]
        visited_edges.add(edge_key(start, nxt))
        prev = start
        cur = nxt
        while degree.get(cur, 0) == 2:
            candidates = [p for p in _skeleton_neighbors(cur, skel) if p != prev]
            if not candidates:
                break
            nxt2 = candidates[0]
            key = edge_key(cur, nxt2)
            if key in visited_edges:
                break
            visited_edges.add(key)
            path.append(nxt2)
            prev, cur = cur, nxt2
        return np.asarray(path, dtype=np.float32)

    paths: List[np.ndarray] = []
    for start in starts:
        for nxt in _skeleton_neighbors(start, skel):
            if edge_key(start, nxt) in visited_edges:
                continue
            path = walk(start, nxt)
            if path.shape[0] >= int(min_pixels):
                paths.append(path)

    # Closed loops have no endpoints/junctions. Start one path per unvisited loop.
    for start in points:
        neighbors = _skeleton_neighbors(start, skel)
        for nxt in neighbors:
            if edge_key(start, nxt) in visited_edges:
                continue
            path = [start, nxt]
            visited_edges.add(edge_key(start, nxt))
            prev = start
            cur = nxt
            while True:
                candidates = [p for p in _skeleton_neighbors(cur, skel) if p != prev]
                if not candidates:
                    break
                nxt2 = candidates[0]
                key = edge_key(cur, nxt2)
                if key in visited_edges or nxt2 not in point_set:
                    break
                visited_edges.add(key)
                path.append(nxt2)
                prev, cur = cur, nxt2
                if cur == start:
                    break
            arr = np.asarray(path, dtype=np.float32)
            if arr.shape[0] >= int(min_pixels):
                paths.append(arr)
    return paths


def _smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    win = max(1, int(window))
    if win <= 1 or path.shape[0] < 3:
        return path.astype(np.float32)
    if win % 2 == 0:
        win += 1
    half = win // 2
    padded = np.pad(path.astype(np.float32), ((half, half), (0, 0)), mode="edge")
    out = np.zeros_like(path, dtype=np.float32)
    for i in range(path.shape[0]):
        out[i] = padded[i : i + win].mean(axis=0)
    out[0] = path[0]
    out[-1] = path[-1]
    return out


def _resample_path(path: np.ndarray, step_px: float) -> np.ndarray:
    if path.shape[0] <= 1:
        return path.astype(np.float32)
    deltas = np.diff(path.astype(np.float32), axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-6:
        return path[:1].astype(np.float32)
    step = max(float(step_px), 1.0)
    samples = np.arange(0.0, total, step, dtype=np.float32)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate([samples, np.asarray([total], dtype=np.float32)])
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_len).astype(np.float32)])
    out = []
    for s in samples:
        idx = int(np.searchsorted(cum, float(s), side="right") - 1)
        idx = max(0, min(idx, seg_len.shape[0] - 1))
        denom = max(float(seg_len[idx]), 1e-8)
        t = (float(s) - float(cum[idx])) / denom
        out.append(path[idx] * (1.0 - t) + path[idx + 1] * t)
    return np.asarray(out, dtype=np.float32)


def _primitive_from_skeleton(
    strength_img: np.ndarray,
    rgb_img: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    positive = strength_img[np.isfinite(strength_img) & (strength_img > 0)]
    if positive.size == 0:
        return {
            "xy": np.zeros((0, 2), dtype=np.float32),
            "theta": np.zeros((0,), dtype=np.float32),
            "sigma_long": np.zeros((0,), dtype=np.float32),
            "sigma_short": np.zeros((0,), dtype=np.float32),
            "color": np.zeros((0, 3), dtype=np.float32),
            "score": np.zeros((0,), dtype=np.float32),
            "kind": np.zeros((0,), dtype=np.int32),
        }
    threshold = max(
        float(args.skeleton_min_weight),
        float(np.percentile(positive, float(args.skeleton_threshold_percentile))),
    )
    mask = strength_img >= threshold
    skel = _thin_binary_mask(mask, int(args.skeleton_max_thinning_iters))
    paths = _trace_skeleton_paths(skel, int(args.skeleton_min_path_pixels))
    xy_items: List[np.ndarray] = []
    theta_items: List[np.ndarray] = []
    sigma_long_items: List[np.ndarray] = []
    sigma_short_items: List[np.ndarray] = []
    score_items: List[np.ndarray] = []
    color_items: List[np.ndarray] = []
    h, w = strength_img.shape
    size = (int(w), int(h))
    for path in paths:
        path = _smooth_path(path, int(args.skeleton_smooth_window))
        path = _resample_path(path, float(args.skeleton_sample_step_px))
        if path.shape[0] < 2:
            continue
        p0 = path[:-1]
        p1 = path[1:]
        vec = p1 - p0
        length = np.linalg.norm(vec, axis=1)
        ok = length >= 1.0
        if not np.any(ok):
            continue
        p0 = p0[ok]
        p1 = p1[ok]
        vec = vec[ok]
        length = length[ok]
        center = (p0 + p1) * 0.5
        theta = np.arctan2(vec[:, 1], vec[:, 0]).astype(np.float32)
        score = _bilinear_scalar(strength_img, center, size)
        xy_items.append(center.astype(np.float32))
        theta_items.append(theta)
        sigma_long_items.append((length / max(2.0 * float(args.segment_length_scale), 1e-6)).astype(np.float32))
        sigma_short_items.append(np.full((center.shape[0],), 0.5, dtype=np.float32))
        score_items.append(score.astype(np.float32))
        if rgb_img is not None:
            color_items.append(_bilinear_rgb(rgb_img, center, size))
        else:
            color_items.append(np.repeat(score[:, None], 3, axis=1).astype(np.float32))
    if not xy_items:
        return {
            "xy": np.zeros((0, 2), dtype=np.float32),
            "theta": np.zeros((0,), dtype=np.float32),
            "sigma_long": np.zeros((0,), dtype=np.float32),
            "sigma_short": np.zeros((0,), dtype=np.float32),
            "color": np.zeros((0, 3), dtype=np.float32),
            "score": np.zeros((0,), dtype=np.float32),
            "kind": np.zeros((0,), dtype=np.int32),
        }
    xy = np.concatenate(xy_items, axis=0)
    theta = np.concatenate(theta_items, axis=0).astype(np.float32)
    sigma_short = _estimate_profile_half_width_px(strength_img, xy, theta, args)
    return {
        "xy": xy,
        "theta": theta,
        "sigma_long": np.concatenate(sigma_long_items, axis=0).astype(np.float32),
        "sigma_short": sigma_short,
        "color": np.clip(np.concatenate(color_items, axis=0), 0.0, 1.0).astype(np.float32),
        "score": np.concatenate(score_items, axis=0).astype(np.float32),
        "kind": np.full((xy.shape[0],), KIND_GEOMETRY, dtype=np.int32),
    }


def _estimate_profile_half_width_px(strength_img: np.ndarray, xy: np.ndarray, theta: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    fallback = np.full((xy.shape[0],), max(float(args.dense_stroke_short_px), 1e-6), dtype=np.float32)
    if not bool(args.profile_width_enable) or xy.shape[0] == 0:
        return fallback
    radius = max(int(args.profile_width_radius_px), 1)
    falloff = float(np.clip(float(args.profile_width_falloff), 1e-3, 0.99))
    min_px = max(float(args.profile_width_min_px), 1e-6)
    max_px = max(float(args.profile_width_max_px), min_px)
    size = (int(strength_img.shape[1]), int(strength_img.shape[0]))
    peak = _bilinear_scalar(strength_img, xy.astype(np.float32), size)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)
    half_width = fallback.copy()
    for i in range(xy.shape[0]):
        if not np.isfinite(peak[i]) or peak[i] <= 1e-6:
            continue
        threshold = float(peak[i]) * falloff
        side_widths = []
        for sign in (-1.0, 1.0):
            last = 0.0
            for d in range(1, radius + 1):
                sample_xy = (xy[i : i + 1] + sign * float(d) * normal[i : i + 1]).astype(np.float32)
                value = float(_bilinear_scalar(strength_img, sample_xy, size)[0])
                if value >= threshold:
                    last = float(d)
                elif last > 0.0:
                    break
            side_widths.append(last)
        width = max(float(np.mean(side_widths)), min_px)
        half_width[i] = float(np.clip(width, min_px, max_px))
    return half_width.astype(np.float32)


def _empty_primitive() -> Dict[str, np.ndarray]:
    return {
        "xy": np.zeros((0, 2), dtype=np.float32),
        "theta": np.zeros((0,), dtype=np.float32),
        "sigma_long": np.zeros((0,), dtype=np.float32),
        "sigma_short": np.zeros((0,), dtype=np.float32),
        "color": np.zeros((0, 3), dtype=np.float32),
        "score": np.zeros((0,), dtype=np.float32),
        "kind": np.zeros((0,), dtype=np.int32),
    }


def _concat_primitives(primitives: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    non_empty = [item for item in primitives if int(item["xy"].shape[0]) > 0]
    if not non_empty:
        return _empty_primitive()
    keys = ["xy", "theta", "sigma_long", "sigma_short", "color", "score", "kind"]
    return {key: np.concatenate([item[key] for item in non_empty], axis=0) for key in keys}


def _primitive_from_dense_strokes(
    strength_img: np.ndarray,
    rgb_img: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    positive = strength_img[np.isfinite(strength_img) & (strength_img > 0)]
    if positive.size == 0:
        return _empty_primitive()
    threshold = max(
        float(args.dense_stroke_min_strength),
        float(np.percentile(positive, float(args.dense_stroke_threshold_percentile))),
    )
    valid = np.isfinite(strength_img) & (strength_img >= threshold)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return _empty_primitive()

    scores = strength_img[ys, xs].astype(np.float32)
    h, w = strength_img.shape
    grid = max(int(args.dense_stroke_grid_px), 1)
    cell_w = int(math.ceil(float(w) / float(grid)))
    cells = (ys // grid).astype(np.int64) * int(cell_w) + (xs // grid).astype(np.int64)
    order = np.lexsort((-scores, cells))
    sorted_cells = cells[order]
    first = np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
    keep = order[first]
    keep = keep[np.argsort(scores[keep])[::-1]]
    if int(args.dense_stroke_max_per_view) > 0:
        keep = keep[: int(args.dense_stroke_max_per_view)]
    xs_keep = xs[keep].astype(np.float32)
    ys_keep = ys[keep].astype(np.float32)
    score_keep = scores[keep].astype(np.float32)

    gy, gx = np.gradient(strength_img.astype(np.float32))
    grad_x = gx[ys[keep], xs[keep]].astype(np.float32)
    grad_y = gy[ys[keep], xs[keep]].astype(np.float32)
    theta = (np.arctan2(grad_y, grad_x) + math.pi * 0.5).astype(np.float32)
    weak = (grad_x * grad_x + grad_y * grad_y) < 1e-12
    theta[weak] = 0.0

    xy = np.stack([xs_keep, ys_keep], axis=1).astype(np.float32)
    size = (int(w), int(h))
    if rgb_img is not None:
        color = _bilinear_rgb(rgb_img, xy, size)
    else:
        color = np.repeat(score_keep[:, None], 3, axis=1).astype(np.float32)
    dense_len = max(float(args.dense_stroke_length_px), 1e-6)
    sigma_long = np.full(
        (xy.shape[0],),
        dense_len / max(2.0 * float(args.segment_length_scale), 1e-6),
        dtype=np.float32,
    )
    sigma_short = _estimate_profile_half_width_px(strength_img, xy, theta, args)
    return {
        "xy": xy,
        "theta": theta,
        "sigma_long": sigma_long,
        "sigma_short": sigma_short,
        "color": np.clip(color, 0.0, 1.0).astype(np.float32),
        "score": score_keep,
        "kind": np.full((xy.shape[0],), KIND_TEXTURE, dtype=np.int32),
    }


def _load_primitive(path: Path) -> Dict[str, np.ndarray]:
    data = dict(np.load(path))
    if "xy" in data:
        xy = np.asarray(data["xy"], dtype=np.float32)
    elif "mu_xy" in data:
        xy = np.asarray(data["mu_xy"], dtype=np.float32)
    else:
        raise KeyError(f"{path} must contain xy or mu_xy")
    count = int(xy.shape[0])
    if "theta" in data:
        theta = np.asarray(data["theta"], dtype=np.float32).reshape(-1)
    elif "cholesky" in data:
        theta = _theta_from_cholesky(np.asarray(data["cholesky"], dtype=np.float32), count)
    else:
        theta = np.zeros((count,), dtype=np.float32)
    sigma_long = np.asarray(data.get("sigma_long", np.full((count,), 4.0, dtype=np.float32)), dtype=np.float32).reshape(-1)
    sigma_short = np.asarray(data.get("sigma_short", np.full((count,), 0.7, dtype=np.float32)), dtype=np.float32).reshape(-1)
    color = np.asarray(data.get("color", np.ones((count, 3), dtype=np.float32)), dtype=np.float32).reshape(count, 3)
    score = np.asarray(data.get("score", data.get("opacity", np.ones((count,), dtype=np.float32))), dtype=np.float32).reshape(-1)
    kind = np.asarray(data.get("kind", np.full((count,), KIND_GEOMETRY, dtype=np.int32)), dtype=np.int32).reshape(-1)
    return {
        "xy": xy,
        "theta": theta[:count],
        "sigma_long": sigma_long[:count],
        "sigma_short": sigma_short[:count],
        "color": color[:count],
        "score": score[:count],
        "kind": kind[:count],
    }


def _theta_from_cholesky(cholesky: np.ndarray, count: int) -> np.ndarray:
    if cholesky.shape[0] != count or cholesky.shape[1] < 3:
        return np.zeros((count,), dtype=np.float32)
    theta = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        a = float(cholesky[i, 0])
        b = float(cholesky[i, 1])
        c = float(cholesky[i, 2])
        cov = np.asarray([[a * a, a * b], [a * b, b * b + c * c]], dtype=np.float32)
        vals, vecs = np.linalg.eigh(cov + np.eye(2, dtype=np.float32) * 1e-8)
        long_vec = vecs[:, int(np.argmax(vals))]
        theta[i] = math.atan2(float(long_vec[1]), float(long_vec[0]))
    return theta


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
    values = list(cameras.values())
    if view_index >= len(values):
        return None, "missing"
    return values[view_index], "order"


def _project_points_to_source(
    points: np.ndarray,
    cam: Dict[str, object],
    source_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    pos, cam_x, cam_y, cam_z, fx, fy, cam_w, cam_h = _camera_basis(cam)
    rel = points.astype(np.float32) - pos[None, :]
    depth = rel @ cam_z.astype(np.float32)
    x_cam = rel @ cam_x.astype(np.float32)
    y_cam = rel @ cam_y.astype(np.float32)
    px_cam = x_cam / np.maximum(depth, 1e-8) * float(fx) + float(cam_w) * 0.5
    py_cam = y_cam / np.maximum(depth, 1e-8) * float(fy) + float(cam_h) * 0.5
    sw, sh = source_size
    x = px_cam / max(float(cam_w - 1), 1.0) * max(float(sw - 1), 1.0)
    y = py_cam / max(float(cam_h - 1), 1.0) * max(float(sh - 1), 1.0)
    return np.stack([x, y], axis=1).astype(np.float32), depth.astype(np.float32)


def _primitive_segments_2d(primitive: Dict[str, np.ndarray], length_scale: float, min_len: float, max_len: float) -> Tuple[np.ndarray, np.ndarray]:
    xy = primitive["xy"].astype(np.float32)
    theta = primitive["theta"].astype(np.float32)
    half_len = np.clip(primitive["sigma_long"].astype(np.float32) * float(length_scale), float(min_len), float(max_len))
    direction = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    p0 = xy - direction * half_len[:, None]
    p1 = xy + direction * half_len[:, None]
    return p0.astype(np.float32), p1.astype(np.float32)


def _filter_primitive_ids(
    primitive: Dict[str, np.ndarray],
    weight_path: Optional[Path],
    source_size: Tuple[int, int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    xy = primitive["xy"]
    score = primitive["score"].astype(np.float32)
    kind = primitive["kind"].astype(np.int32)
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(score)
    keep_kinds = _parse_kind_set(str(args.keep_kinds))
    finite &= np.asarray([int(k) in keep_kinds for k in kind], dtype=bool)
    finite &= score >= float(args.min_score)
    weight = np.ones((xy.shape[0],), dtype=np.float32)
    if weight_path is not None and weight_path.is_file():
        weight_img = _load_gray(weight_path, size=source_size)
        weight = _bilinear_scalar(weight_img, xy, source_size)
        finite &= weight >= float(args.min_weight)
    ids = np.flatnonzero(finite)
    if ids.size == 0:
        return ids, weight
    rank_score = score[ids] * np.clip(weight[ids], 0.0, 1.0)
    order = np.argsort(rank_score)[::-1]
    if int(args.max_primitives_per_view) > 0:
        order = order[: int(args.max_primitives_per_view)]
    return ids[order], weight


def _lift_view_segments(
    primitive_path: Path,
    view_index: int,
    args: argparse.Namespace,
    cameras: Dict[str, Dict[str, object]],
    base_xyz: np.ndarray,
    base_opacity: np.ndarray,
    rgb_paths: Sequence[Path],
    rgb_lookup: Dict[str, Path],
    weight_paths: Sequence[Path],
    weight_lookup: Dict[str, Path],
    curve_paths: Sequence[Path],
    curve_lookup: Dict[str, Path],
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, object]]:
    stem = primitive_path.stem
    cam, cam_match = _camera_for_view(cameras, stem, view_index, str(args.match_policy))
    if cam is None:
        return None, {"stem": stem, "status": "missing_camera"}
    rgb_path = _resolve(rgb_paths, rgb_lookup, stem, view_index, str(args.match_policy)) if rgb_paths else None
    weight_path = _resolve(weight_paths, weight_lookup, stem, view_index, str(args.match_policy)) if weight_paths else None
    curve_path = _resolve(curve_paths, curve_lookup, stem, view_index, str(args.match_policy)) if curve_paths else None
    if str(args.curve_source) == "skeleton":
        if weight_path is None or not weight_path.is_file():
            return None, {"stem": stem, "status": "missing_weight_for_skeleton"}
        if str(args.curve_image_mode) != "weight" and (curve_path is None or not curve_path.is_file()):
            return None, {"stem": stem, "status": "missing_curve_image_for_skeleton"}
        source_for_size = curve_path if curve_path is not None and curve_path.is_file() else weight_path
        with Image.open(source_for_size) as image:
            source_size = image.size
        weight_img_for_gate = _load_gray(weight_path, size=source_size)
        curve_rgb = _load_rgb(curve_path, size=source_size) if curve_path is not None and curve_path.is_file() else None
        curve_strength = _curve_strength_from_image(curve_rgb, weight_img_for_gate, args)
        skeleton_primitive = _primitive_from_skeleton(curve_strength, curve_rgb, args)
        if bool(args.dense_stroke_enable):
            dense_primitive = _primitive_from_dense_strokes(curve_strength, curve_rgb, args)
            primitive = _concat_primitives([skeleton_primitive, dense_primitive])
            primitive_source = f"skeleton_dense_{args.curve_image_mode}"
        else:
            primitive = skeleton_primitive
            primitive_source = f"skeleton_{args.curve_image_mode}"
        color_rgb_path = curve_path if curve_path is not None and curve_path.is_file() else rgb_path
    else:
        primitive = _load_primitive(primitive_path)
        source_size = _infer_source_size(primitive, rgb_path, weight_path)
        primitive_source = "primitive"
        color_rgb_path = rgb_path
    ids, weight = _filter_primitive_ids(primitive, weight_path, source_size, args)
    if ids.size == 0:
        return None, {"stem": stem, "status": "empty_after_filter", "available": int(primitive["xy"].shape[0])}

    visible_projection = _project_base_to_source(
        base_xyz,
        base_opacity,
        cam,
        source_size,
        float(args.base_opacity_min),
        float(args.depth_min),
    )
    grid_index, grid_depth = _build_visible_layer_index(visible_projection)

    xy = primitive["xy"][ids]
    matched, matched_base_idx, matched_depth, matched_dist = _match_primitives_to_visible_layer(
        xy,
        grid_index,
        grid_depth,
        int(args.search_radius_px),
    )
    if not np.any(matched):
        return None, {
            "stem": stem,
            "status": "empty_after_layer_match",
            "candidate": int(ids.size),
            "source_size": [int(source_size[0]), int(source_size[1])],
        }

    p0_all, p1_all = _primitive_segments_2d(
        primitive,
        float(args.segment_length_scale),
        float(args.segment_min_length_px),
        float(args.segment_max_length_px),
    )
    kept_ids = ids[matched]
    kept_xy = xy[matched]
    kept_depth = matched_depth[matched]
    kept_base_idx = matched_base_idx[matched]
    kept_dist = matched_dist[matched]
    p0 = p0_all[kept_ids]
    p1 = p1_all[kept_ids]

    endpoint_depth = np.repeat(kept_depth[:, None], 2, axis=1).astype(np.float32)
    if int(args.endpoint_search_radius_px) > 0 or bool(args.require_endpoint_match):
        endpoints = np.concatenate([p0, p1], axis=0)
        ep_matched, _ep_idx, ep_depth, _ep_dist = _match_primitives_to_visible_layer(
            endpoints,
            grid_index,
            grid_depth,
            int(args.endpoint_search_radius_px),
        )
        ep_matched = ep_matched.reshape(2, -1).T
        ep_depth = ep_depth.reshape(2, -1).T
        max_delta = np.maximum(float(args.max_endpoint_depth_delta_px), 0.0)
        pixel_world = kept_depth / max((float(cam["fx"]) + float(cam["fy"])) * 0.5, 1e-8)
        valid_depth = ep_matched & (np.abs(ep_depth - kept_depth[:, None]) <= max_delta * pixel_world[:, None])
        endpoint_depth = np.where(valid_depth, ep_depth, endpoint_depth)
        if bool(args.require_endpoint_match):
            ok = valid_depth.all(axis=1)
            kept_ids = kept_ids[ok]
            kept_xy = kept_xy[ok]
            kept_depth = kept_depth[ok]
            kept_base_idx = kept_base_idx[ok]
            kept_dist = kept_dist[ok]
            p0 = p0[ok]
            p1 = p1[ok]
            endpoint_depth = endpoint_depth[ok]
            if kept_ids.size == 0:
                return None, {"stem": stem, "status": "empty_after_endpoint_match", "candidate": int(ids.size)}

    endpoints_2d = np.concatenate([p0, p1], axis=0).astype(np.float32)
    endpoints_depth = np.concatenate([endpoint_depth[:, 0], endpoint_depth[:, 1]], axis=0).astype(np.float32)
    endpoints_3d, cam_x, cam_y, endpoint_normal, _px, fx, fy = _unproject_points(cam, endpoints_2d, endpoints_depth, source_size)
    count = int(kept_ids.size)
    p0_3d = endpoints_3d[:count]
    p1_3d = endpoints_3d[count:]
    center_3d, _cx, _cy, center_normal, _cpx, _cfx, _cfy = _unproject_points(cam, kept_xy, kept_depth, source_size)
    pix_world = kept_depth / max((float(fx) + float(fy)) * 0.5, 1e-8)
    source_pixel_scale = 0.5 * (
        max(float(cam["width"] - 1), 1.0) / max(float(source_size[0] - 1), 1.0)
        + max(float(cam["height"] - 1), 1.0) / max(float(source_size[1] - 1), 1.0)
    )
    pix_world = (pix_world * source_pixel_scale).astype(np.float32)
    if float(args.front_offset_px) > 0:
        offset = float(args.front_offset_px) * pix_world[:, None] * center_normal
        p0_3d = p0_3d - offset
        p1_3d = p1_3d - offset
        center_3d = center_3d - offset

    direction = p1_3d - p0_3d
    length = np.linalg.norm(direction, axis=1)
    valid_len = np.isfinite(length) & (length > 1e-8)
    if not np.any(valid_len):
        return None, {"stem": stem, "status": "empty_after_3d_length_filter", "candidate": int(ids.size)}
    direction = direction[valid_len] / length[valid_len, None]

    final_ids = kept_ids[valid_len]
    rgb_color = np.clip(primitive["color"][final_ids], 0.0, 1.0).astype(np.float32)
    if color_rgb_path is not None and color_rgb_path.is_file():
        rgb_img = _load_rgb(color_rgb_path, size=source_size)
        sampled = _bilinear_rgb(rgb_img, primitive["xy"][final_ids], source_size)
        low = np.mean(np.abs(rgb_color), axis=1, keepdims=True) < 1e-5
        rgb_color = np.where(low, sampled, rgb_color).astype(np.float32)

    lifted = {
        "p0": p0_3d[valid_len].astype(np.float32),
        "p1": p1_3d[valid_len].astype(np.float32),
        "center": center_3d[valid_len].astype(np.float32),
        "direction": direction.astype(np.float32),
        "normal": center_normal[valid_len].astype(np.float32),
        "length": length[valid_len].astype(np.float32),
        "source_view": np.full((int(np.count_nonzero(valid_len)),), int(view_index), dtype=np.int32),
        "source_primitive": final_ids.astype(np.int64),
        "source_stem_id": np.full((int(np.count_nonzero(valid_len)),), int(view_index), dtype=np.int32),
        "source_xy": primitive["xy"][final_ids].astype(np.float32),
        "source_p0": p0[valid_len].astype(np.float32),
        "source_p1": p1[valid_len].astype(np.float32),
        "source_width_px": primitive["sigma_short"][final_ids].astype(np.float32),
        "score": primitive["score"][final_ids].astype(np.float32),
        "weight": weight[final_ids].astype(np.float32),
        "kind": primitive["kind"][final_ids].astype(np.int32),
        "color": rgb_color,
        "matched_base_index": kept_base_idx[valid_len].astype(np.int64),
        "matched_depth": kept_depth[valid_len].astype(np.float32),
        "matched_pixel_distance": kept_dist[valid_len].astype(np.float32),
        "pixel_world": pix_world[valid_len].astype(np.float32),
    }
    info = {
        "stem": stem,
        "status": "ok",
        "primitive_source": primitive_source,
        "camera_match": cam_match,
        "curve_image": "" if curve_path is None else str(curve_path),
        "available": int(primitive["xy"].shape[0]),
        "candidate": int(ids.size),
        "lifted": int(lifted["p0"].shape[0]),
        "source_size": [int(source_size[0]), int(source_size[1])],
        "camera_index": int(cam.get("_index", view_index)),
        "weight_mean": float(lifted["weight"].mean()),
        "score_mean": float(lifted["score"].mean()),
        "pixel_world_mean": float(lifted["pixel_world"].mean()),
        "length_mean": float(lifted["length"].mean()),
        "width_px_mean": float(lifted["source_width_px"].mean()),
        "matched_pixel_distance_mean": float(lifted["matched_pixel_distance"].mean()),
    }
    return lifted, info


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros((n,), dtype=np.int8)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        if p != x:
            self.parent[x] = self.find(p)
        return int(self.parent[x])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _line_distance(point: np.ndarray, center: np.ndarray, direction: np.ndarray) -> float:
    delta = point - center
    proj = float(delta @ direction)
    residual = delta - proj * direction
    return float(np.linalg.norm(residual))


def _overlap_ratio(p0_a: np.ndarray, p1_a: np.ndarray, p0_b: np.ndarray, p1_b: np.ndarray, axis: np.ndarray) -> float:
    a = np.asarray([float(p0_a @ axis), float(p1_a @ axis)], dtype=np.float32)
    b = np.asarray([float(p0_b @ axis), float(p1_b @ axis)], dtype=np.float32)
    amin, amax = float(a.min()), float(a.max())
    bmin, bmax = float(b.min()), float(b.max())
    overlap = max(0.0, min(amax, bmax) - max(amin, bmin))
    denom = max(min(amax - amin, bmax - bmin), 1e-8)
    return float(overlap / denom)


def _should_merge(i: int, j: int, segments: Dict[str, np.ndarray], args: argparse.Namespace) -> bool:
    if not bool(args.merge_same_view) and int(segments["source_view"][i]) == int(segments["source_view"][j]):
        return False
    di = segments["direction"][i]
    dj = segments["direction"][j]
    cos_angle = abs(float(di @ dj))
    if cos_angle < math.cos(math.radians(float(args.merge_angle_deg))):
        return False
    ci = segments["center"][i]
    cj = segments["center"][j]
    radius = max(
        float(args.merge_radius_abs),
        float(args.merge_radius_px) * 0.5 * float(segments["pixel_world"][i] + segments["pixel_world"][j]),
    )
    center_dist = float(np.linalg.norm(ci - cj))
    if center_dist > radius * 2.0:
        return False
    line_dist = max(
        _line_distance(ci, cj, dj),
        _line_distance(cj, ci, di),
    )
    if line_dist > radius:
        return False
    overlap = _overlap_ratio(segments["p0"][i], segments["p1"][i], segments["p0"][j], segments["p1"][j], di)
    return bool(overlap >= float(args.merge_min_overlap) or center_dist <= radius)


def _concat_segments(chunks: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = [
        "p0",
        "p1",
        "center",
        "direction",
        "normal",
        "length",
        "source_view",
        "source_primitive",
        "source_stem_id",
        "source_xy",
        "source_p0",
        "source_p1",
        "source_width_px",
        "score",
        "weight",
        "kind",
        "color",
        "matched_base_index",
        "matched_depth",
        "matched_pixel_distance",
        "pixel_world",
    ]
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _limit_segments_for_merge(segments: Dict[str, np.ndarray], max_count: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n = int(segments["p0"].shape[0])
    if int(max_count) <= 0 or n <= int(max_count):
        ids = np.arange(n, dtype=np.int64)
        return segments, ids
    rank = segments["score"] * np.clip(segments["weight"], 0.0, 1.0)
    ids = np.argsort(rank)[::-1][: int(max_count)].astype(np.int64)
    limited: Dict[str, np.ndarray] = {}
    for key, value in segments.items():
        if value.shape[:1] == (n,):
            limited[key] = value[ids]
        else:
            limited[key] = value
    return limited, ids


def _merge_segments(segments: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n = int(segments["p0"].shape[0])
    if n == 0:
        return _empty_tracks(), np.zeros((0,), dtype=bool)
    uf = _UnionFind(n)
    median_pix = float(np.median(segments["pixel_world"])) if n > 0 else 1e-3
    cell_size = max(float(args.merge_radius_abs), float(args.merge_radius_px) * median_pix, 1e-6)
    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    rank = segments["score"] * np.clip(segments["weight"], 0.0, 1.0)
    order = np.argsort(rank)[::-1]
    centers = segments["center"]
    for idx in order.tolist():
        cell = tuple(np.floor(centers[idx] / cell_size).astype(np.int64).tolist())
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []):
                        if _should_merge(idx, j, segments, args):
                            uf.union(idx, j)
        grid[cell].append(idx)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    tracks = _aggregate_tracks(groups, segments, args)
    keep = _track_keep_mask(tracks, args)
    return tracks, keep.astype(bool)


def _canonical_direction_key(direction: np.ndarray, bins: int) -> Tuple[int, int, int]:
    d = np.asarray(direction, dtype=np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-8)
    for value in d.tolist():
        if abs(float(value)) > 1e-6:
            if float(value) < 0.0:
                d = -d
            break
    q = np.round(d * max(int(bins), 1)).astype(np.int32)
    return int(q[0]), int(q[1]), int(q[2])


def _layer_bin_key(i: int, segments: Dict[str, np.ndarray], cell_size: float, args: argparse.Namespace) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
    cell = tuple(np.floor(segments["center"][i] / max(float(cell_size), 1e-8)).astype(np.int64).tolist())
    direction = _canonical_direction_key(segments["direction"][i], int(args.layer_dir_bins))
    kind = int(segments["kind"][i]) if bool(args.layer_include_kind) else 0
    return (int(cell[0]), int(cell[1]), int(cell[2])), direction, kind


def _aggregate_layer_bins(segments: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n = int(segments["p0"].shape[0])
    if n == 0:
        return _empty_tracks(), np.zeros((0,), dtype=bool)
    median_pix = float(np.median(segments["pixel_world"])) if n > 0 else 1e-3
    cell_size = max(float(args.layer_bin_radius_abs), float(args.layer_bin_radius_px) * median_pix, 1e-6)
    rank = segments["score"] * np.clip(segments["weight"], 0.0, 1.0)
    order = np.argsort(rank)[::-1]
    groups_by_key: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int], int], List[int]] = defaultdict(list)
    for idx in order.tolist():
        groups_by_key[_layer_bin_key(idx, segments, cell_size, args)].append(idx)
    groups = {group_id: ids for group_id, ids in enumerate(groups_by_key.values())}
    tracks = _aggregate_tracks(groups, segments, args)
    keep = _track_keep_mask(tracks, args)
    return tracks, keep.astype(bool)


def _source_segment_direction(segments: Dict[str, np.ndarray], ids: np.ndarray) -> np.ndarray:
    direction = segments["source_p1"][ids].astype(np.float32) - segments["source_p0"][ids].astype(np.float32)
    return direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)


def _candidate_reprojection_pass(
    i: int,
    j: int,
    segments: Dict[str, np.ndarray],
    view_cameras: Sequence[Optional[Dict[str, object]]],
    source_sizes: Sequence[Tuple[int, int]],
    args: argparse.Namespace,
) -> Tuple[bool, float, float]:
    view_j = int(segments["source_view"][j])
    if view_j < 0 or view_j >= len(view_cameras):
        return False, float("inf"), 0.0
    cam_j = view_cameras[view_j]
    if cam_j is None:
        return False, float("inf"), 0.0
    size_j = source_sizes[view_j]
    points = np.stack([segments["center"][i], segments["p0"][i], segments["p1"][i]], axis=0).astype(np.float32)
    xy, depth = _project_points_to_source(points, cam_j, size_j)
    sw, sh = size_j
    if (
        not np.isfinite(xy).all()
        or not np.isfinite(depth).all()
        or float(depth[0]) <= float(args.depth_min)
        or float(xy[0, 0]) < 0.0
        or float(xy[0, 1]) < 0.0
        or float(xy[0, 0]) > float(sw - 1)
        or float(xy[0, 1]) > float(sh - 1)
    ):
        return False, float("inf"), 0.0
    reproj_error = float(np.linalg.norm(xy[0] - segments["source_xy"][j]))
    if reproj_error > float(args.candidate_reproj_radius_px):
        return False, reproj_error, 0.0
    if float(args.candidate_depth_delta_px) > 0.0:
        depth_tol = float(args.candidate_depth_delta_px) * max(float(segments["pixel_world"][j]), 1e-8)
        if abs(float(depth[0]) - float(segments["matched_depth"][j])) > depth_tol:
            return False, reproj_error, 0.0
    projected_dir = xy[2] - xy[1]
    projected_dir = projected_dir / max(float(np.linalg.norm(projected_dir)), 1e-8)
    target_dir = segments["source_p1"][j] - segments["source_p0"][j]
    target_dir = target_dir / max(float(np.linalg.norm(target_dir)), 1e-8)
    dir_score = abs(float(projected_dir @ target_dir))
    if dir_score < math.cos(math.radians(float(args.candidate_dir_angle_deg))):
        return False, reproj_error, dir_score
    return True, reproj_error, dir_score


def _candidate_graph_tracks(
    segments: Dict[str, np.ndarray],
    view_cameras: Sequence[Optional[Dict[str, object]]],
    source_sizes: Sequence[Tuple[int, int]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(segments["p0"].shape[0])
    if n == 0:
        empty = _empty_tracks()
        z = np.zeros((0,), dtype=bool)
        return empty, z, z, z, z
    tracks, _ = _tracks_from_source_segments(segments, args)
    median_pix = float(np.median(segments["pixel_world"])) if n > 0 else 1e-3
    cell_size = max(float(args.candidate_radius_abs), float(args.candidate_radius_px) * median_pix, 1e-6)
    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    centers = segments["center"]
    for idx in range(n):
        cell = tuple(np.floor(centers[idx] / cell_size).astype(np.int64).tolist())
        grid[(int(cell[0]), int(cell[1]), int(cell[2]))].append(idx)

    cos_dir = math.cos(math.radians(float(args.candidate_dir_angle_deg)))
    cos_normal = math.cos(math.radians(float(args.candidate_normal_angle_deg)))
    source_strength = (segments["score"] * np.clip(segments["weight"], 0.0, 1.0)).astype(np.float32)
    support_count = np.ones((n,), dtype=np.int32)
    support_views: List[set[int]] = [set([int(v)]) for v in segments["source_view"].tolist()]
    support_score_sum = source_strength.astype(np.float64).copy()
    reproj_error_sum = np.zeros((n,), dtype=np.float64)
    reproj_dir_sum = np.zeros((n,), dtype=np.float64)
    reproj_count = np.zeros((n,), dtype=np.int32)
    dir_sum = np.ones((n,), dtype=np.float64)
    residual_sum = np.zeros((n,), dtype=np.float64)
    weighted_color = (segments["color"] * source_strength[:, None]).astype(np.float64)
    weight_sum = source_strength.astype(np.float64).copy()
    rank = np.argsort(source_strength)[::-1]
    for i in rank.tolist():
        cell = tuple(np.floor(centers[i] / cell_size).astype(np.int64).tolist())
        radius_i = max(float(args.candidate_radius_abs), float(args.candidate_radius_px) * float(segments["pixel_world"][i]))
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j in grid.get((int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)), []):
                        if i == j or int(segments["source_view"][i]) == int(segments["source_view"][j]):
                            continue
                        radius = max(radius_i, float(args.candidate_radius_abs), float(args.candidate_radius_px) * float(segments["pixel_world"][j]))
                        center_dist = float(np.linalg.norm(segments["center"][i] - segments["center"][j]))
                        if center_dist > radius:
                            continue
                        dir_score_3d = abs(float(segments["direction"][i] @ segments["direction"][j]))
                        if dir_score_3d < cos_dir:
                            continue
                        normal_score = abs(float(segments["normal"][i] @ segments["normal"][j]))
                        if normal_score < cos_normal:
                            continue
                        ok, reproj_error, dir_score_2d = _candidate_reprojection_pass(i, j, segments, view_cameras, source_sizes, args)
                        if not ok:
                            continue
                        score_j = float(source_strength[j])
                        support_count[i] += 1
                        support_views[i].add(int(segments["source_view"][j]))
                        support_score_sum[i] += score_j
                        reproj_error_sum[i] += reproj_error
                        reproj_dir_sum[i] += dir_score_2d
                        reproj_count[i] += 1
                        dir_sum[i] += dir_score_3d
                        residual_sum[i] += _line_distance(segments["center"][j], segments["center"][i], segments["direction"][i])
                        weighted_color[i] += segments["color"][j].astype(np.float64) * score_j
                        weight_sum[i] += score_j

    tracks["segment_count"] = support_count.astype(np.int32)
    tracks["view_count"] = np.asarray([len(v) for v in support_views], dtype=np.int32)
    tracks["score_mean"] = (support_score_sum / np.maximum(support_count.astype(np.float64), 1.0)).astype(np.float32)
    tracks["score_max"] = np.maximum(tracks["score_max"], tracks["score_mean"]).astype(np.float32)
    tracks["weight_mean"] = np.maximum(tracks["weight_mean"], np.clip(tracks["score_mean"], 0.0, 1.0)).astype(np.float32)
    tracks["color"] = np.clip(weighted_color / np.maximum(weight_sum[:, None], 1e-8), 0.0, 1.0).astype(np.float32)
    tracks["source_strength"] = source_strength.astype(np.float32)
    tracks["neighbor_support_score"] = (
        (support_score_sum - source_strength.astype(np.float64)) / np.maximum(support_count.astype(np.float64) - 1.0, 1.0)
    ).astype(np.float32)
    tracks["reproject_error_px"] = (reproj_error_sum / np.maximum(reproj_count.astype(np.float64), 1.0)).astype(np.float32)
    tracks["reproject_dir_consistency"] = (reproj_dir_sum / np.maximum(reproj_count.astype(np.float64), 1.0)).astype(np.float32)
    tracks["dir_consistency"] = (dir_sum / np.maximum(support_count.astype(np.float64), 1.0)).astype(np.float32)
    tracks["line_residual"] = (residual_sum / np.maximum(support_count.astype(np.float64) - 1.0, 1.0)).astype(np.float32)
    tracks["line_residual_px"] = tracks["line_residual"] / np.maximum(tracks["pixel_world_mean"], 1e-8)
    tracks["support_view_ratio"] = tracks["view_count"].astype(np.float32) / np.maximum(tracks["segment_count"].astype(np.float32), 1.0)
    base_keep = _track_keep_mask(tracks, args)
    survive = base_keep & (tracks["view_count"] >= int(args.candidate_min_survive_views))
    probation = (
        base_keep
        & ~survive
        & (tracks["source_strength"] >= float(args.candidate_probation_min_source_strength))
        & (tracks["line_residual_px"] <= float(args.candidate_probation_max_line_residual_px))
    )
    if not bool(args.candidate_keep_probation):
        keep = survive
    else:
        keep = survive | probation
    suppress = ~(survive | probation)
    return tracks, keep.astype(bool), survive.astype(bool), probation.astype(bool), suppress.astype(bool)


def _tracks_from_source_segments(segments: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n = int(segments["p0"].shape[0])
    if n == 0:
        return _empty_tracks(), np.zeros((0,), dtype=bool)
    tracks = _empty_tracks()
    tracks["p0"] = segments["p0"].astype(np.float32)
    tracks["p1"] = segments["p1"].astype(np.float32)
    tracks["center"] = segments["center"].astype(np.float32)
    tracks["direction"] = segments["direction"].astype(np.float32)
    tracks["normal"] = segments["normal"].astype(np.float32)
    tracks["color"] = segments["color"].astype(np.float32)
    tracks["score_mean"] = segments["score"].astype(np.float32)
    tracks["score_max"] = segments["score"].astype(np.float32)
    tracks["weight_mean"] = segments["weight"].astype(np.float32)
    tracks["length"] = segments["length"].astype(np.float32)
    tracks["width_px_mean"] = segments["source_width_px"].astype(np.float32)
    tracks["width_px_max"] = segments["source_width_px"].astype(np.float32)
    tracks["pixel_world_mean"] = segments["pixel_world"].astype(np.float32)
    tracks["segment_count"] = np.ones((n,), dtype=np.int32)
    tracks["view_count"] = np.ones((n,), dtype=np.int32)
    tracks["kind_geometry_count"] = (segments["kind"] == KIND_GEOMETRY).astype(np.int32)
    tracks["kind_texture_count"] = (segments["kind"] == KIND_TEXTURE).astype(np.int32)
    tracks["kind_noise_count"] = (segments["kind"] == KIND_NOISE).astype(np.int32)
    tracks["source_strength"] = (segments["score"] * np.clip(segments["weight"], 0.0, 1.0)).astype(np.float32)
    tracks["neighbor_support_score"] = np.zeros((n,), dtype=np.float32)
    tracks["reproject_error_px"] = np.zeros((n,), dtype=np.float32)
    tracks["reproject_dir_consistency"] = np.zeros((n,), dtype=np.float32)
    tracks["dir_consistency"] = np.ones((n,), dtype=np.float32)
    tracks["line_residual"] = np.zeros((n,), dtype=np.float32)
    tracks["line_residual_px"] = np.zeros((n,), dtype=np.float32)
    tracks["support_view_ratio"] = np.ones((n,), dtype=np.float32)
    keep = _track_keep_mask(tracks, args)
    return tracks, keep.astype(bool)


def _empty_tracks() -> Dict[str, np.ndarray]:
    return {
        "p0": np.zeros((0, 3), dtype=np.float32),
        "p1": np.zeros((0, 3), dtype=np.float32),
        "center": np.zeros((0, 3), dtype=np.float32),
        "direction": np.zeros((0, 3), dtype=np.float32),
        "normal": np.zeros((0, 3), dtype=np.float32),
        "color": np.zeros((0, 3), dtype=np.float32),
        "score_mean": np.zeros((0,), dtype=np.float32),
        "score_max": np.zeros((0,), dtype=np.float32),
        "weight_mean": np.zeros((0,), dtype=np.float32),
        "length": np.zeros((0,), dtype=np.float32),
        "width_px_mean": np.zeros((0,), dtype=np.float32),
        "width_px_max": np.zeros((0,), dtype=np.float32),
        "pixel_world_mean": np.zeros((0,), dtype=np.float32),
        "segment_count": np.zeros((0,), dtype=np.int32),
        "view_count": np.zeros((0,), dtype=np.int32),
        "kind_geometry_count": np.zeros((0,), dtype=np.int32),
        "kind_texture_count": np.zeros((0,), dtype=np.int32),
        "kind_noise_count": np.zeros((0,), dtype=np.int32),
        "source_strength": np.zeros((0,), dtype=np.float32),
        "neighbor_support_score": np.zeros((0,), dtype=np.float32),
        "reproject_error_px": np.zeros((0,), dtype=np.float32),
        "reproject_dir_consistency": np.zeros((0,), dtype=np.float32),
        "dir_consistency": np.zeros((0,), dtype=np.float32),
        "line_residual": np.zeros((0,), dtype=np.float32),
        "line_residual_px": np.zeros((0,), dtype=np.float32),
        "support_view_ratio": np.zeros((0,), dtype=np.float32),
    }


def _track_keep_mask(tracks: Dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    keep = (
        (tracks["segment_count"] >= int(args.min_track_segments))
        & (tracks["view_count"] >= int(args.min_track_views))
        & np.isfinite(tracks["length"])
        & (tracks["length"] > 1e-8)
        & np.isfinite(tracks["score_max"])
        & (tracks["score_max"] >= float(args.min_score))
        & np.isfinite(tracks["weight_mean"])
        & (tracks["weight_mean"] >= float(args.min_weight))
    )
    min_dir = float(args.track_min_dir_consistency)
    if min_dir > 0.0 and "dir_consistency" in tracks:
        keep &= np.isfinite(tracks["dir_consistency"]) & (tracks["dir_consistency"] >= min_dir)
    max_res_px = float(args.track_max_line_residual_px)
    if max_res_px > 0.0 and "line_residual_px" in tracks:
        keep &= np.isfinite(tracks["line_residual_px"]) & (tracks["line_residual_px"] <= max_res_px)
    return keep.astype(bool)


def _aggregate_tracks(groups: Dict[int, List[int]], segments: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, np.ndarray]:
    out: Dict[str, List[object]] = {key: [] for key in _empty_tracks().keys()}
    for ids in groups.values():
        idx = np.asarray(ids, dtype=np.int64)
        weights = np.clip(segments["score"][idx] * segments["weight"][idx], 1e-4, None).astype(np.float32)
        points = np.concatenate([segments["p0"][idx], segments["p1"][idx]], axis=0)
        point_weights = np.repeat(weights, 2)
        center = (points * point_weights[:, None]).sum(axis=0) / max(float(point_weights.sum()), 1e-8)
        centered = points - center[None, :]
        cov = (centered * point_weights[:, None]).T @ centered / max(float(point_weights.sum()), 1e-8)
        vals, vecs = np.linalg.eigh(cov + np.eye(3, dtype=np.float32) * 1e-10)
        direction = vecs[:, int(np.argmax(vals))].astype(np.float32)
        mean_dir = (segments["direction"][idx] * weights[:, None]).sum(axis=0)
        if float(direction @ mean_dir) < 0.0:
            direction = -direction
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        proj = centered @ direction
        p0 = center + float(proj.min()) * direction
        p1 = center + float(proj.max()) * direction
        residual = centered - proj[:, None] * direction[None, :]
        line_residual = float(np.average(np.linalg.norm(residual, axis=1), weights=point_weights))
        line_residual_px = line_residual / max(float(np.mean(segments["pixel_world"][idx])), 1e-8)
        dir_alignment = np.abs(segments["direction"][idx] @ direction)
        dir_consistency = float(np.average(dir_alignment, weights=weights))
        normal = (segments["normal"][idx] * weights[:, None]).sum(axis=0)
        normal = normal - float(normal @ direction) * direction
        if float(np.linalg.norm(normal)) < 1e-8:
            normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            normal = normal - float(normal @ direction) * direction
        normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
        color = (segments["color"][idx] * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-8)
        kinds = segments["kind"][idx]
        views = np.unique(segments["source_view"][idx])
        out["p0"].append(p0.astype(np.float32))
        out["p1"].append(p1.astype(np.float32))
        out["center"].append(center.astype(np.float32))
        out["direction"].append(direction.astype(np.float32))
        out["normal"].append(normal.astype(np.float32))
        out["color"].append(np.clip(color, 0.0, 1.0).astype(np.float32))
        out["score_mean"].append(float(np.mean(segments["score"][idx])))
        out["score_max"].append(float(np.max(segments["score"][idx])))
        out["weight_mean"].append(float(np.mean(segments["weight"][idx])))
        out["length"].append(float(np.linalg.norm(p1 - p0)))
        out["width_px_mean"].append(float(np.average(segments["source_width_px"][idx], weights=weights)))
        out["width_px_max"].append(float(np.max(segments["source_width_px"][idx])))
        out["pixel_world_mean"].append(float(np.mean(segments["pixel_world"][idx])))
        out["segment_count"].append(int(idx.size))
        out["view_count"].append(int(views.size))
        out["kind_geometry_count"].append(int(np.count_nonzero(kinds == KIND_GEOMETRY)))
        out["kind_texture_count"].append(int(np.count_nonzero(kinds == KIND_TEXTURE)))
        out["kind_noise_count"].append(int(np.count_nonzero(kinds == KIND_NOISE)))
        out["source_strength"].append(float(np.max(segments["score"][idx] * np.clip(segments["weight"][idx], 0.0, 1.0))))
        out["neighbor_support_score"].append(float(np.mean(segments["score"][idx] * np.clip(segments["weight"][idx], 0.0, 1.0))))
        out["reproject_error_px"].append(0.0)
        out["reproject_dir_consistency"].append(0.0)
        out["dir_consistency"].append(dir_consistency)
        out["line_residual"].append(line_residual)
        out["line_residual_px"].append(line_residual_px)
        out["support_view_ratio"].append(float(views.size) / max(float(idx.size), 1.0))
    result: Dict[str, np.ndarray] = {}
    for key, values in out.items():
        empty = _empty_tracks()[key]
        if not values:
            result[key] = empty
            continue
        dtype = empty.dtype
        result[key] = np.asarray(values, dtype=dtype)
    return result


def _write_obj(path: Path, tracks: Dict[str, np.ndarray], keep: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    p0 = tracks["p0"][keep]
    p1 = tracks["p1"][keep]
    for a, b in zip(p0, p1):
        lines.append(f"v {float(a[0]):.8f} {float(a[1]):.8f} {float(a[2]):.8f}\n")
        lines.append(f"v {float(b[0]):.8f} {float(b[1]):.8f} {float(b[2]):.8f}\n")
    for i in range(p0.shape[0]):
        lines.append(f"l {2 * i + 1} {2 * i + 2}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _draw_segments_overlay(path: Path, size: Tuple[int, int], segments: Dict[str, np.ndarray], view_id: int, max_draw: int) -> None:
    sw, sh = size
    image = Image.new("RGB", (sw, sh), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    ids = np.flatnonzero(segments["source_view"] == int(view_id))
    if ids.size > int(max_draw):
        score = segments["score"][ids] * np.clip(segments["weight"][ids], 0.0, 1.0)
        ids = ids[np.argsort(score)[::-1][: int(max_draw)]]
    colors = {KIND_GEOMETRY: (255, 70, 25), KIND_TEXTURE: (255, 220, 40), KIND_NOISE: (70, 130, 255)}
    for i in ids.tolist():
        color = colors.get(int(segments["kind"][i]), (255, 255, 255))
        if "source_p0" in segments and "source_p1" in segments:
            x0, y0 = segments["source_p0"][i]
            x1, y1 = segments["source_p1"][i]
            draw.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=1)
        else:
            x, y = segments["source_xy"][i]
            draw.ellipse((float(x) - 1.0, float(y) - 1.0, float(x) + 1.0, float(y) + 1.0), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_track_projection(
    path: Path,
    size: Tuple[int, int],
    cam: Dict[str, object],
    tracks: Dict[str, np.ndarray],
    keep: np.ndarray,
    max_draw: int,
    strong_min_views: int,
) -> None:
    sw, sh = size
    image = Image.new("RGB", (sw, sh), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    ids = np.flatnonzero(keep)
    if ids.size > int(max_draw):
        rank = tracks["score_max"][ids] * np.maximum(tracks["view_count"][ids], 1)
        ids = ids[np.argsort(rank)[::-1][: int(max_draw)]]
    p0_xy, p0_depth = _project_points_to_source(tracks["p0"][ids], cam, size)
    p1_xy, p1_depth = _project_points_to_source(tracks["p1"][ids], cam, size)
    for local, tid in enumerate(ids.tolist()):
        if p0_depth[local] <= 0 or p1_depth[local] <= 0:
            continue
        x0, y0 = p0_xy[local]
        x1, y1 = p1_xy[local]
        if max(x0, x1) < 0 or max(y0, y1) < 0 or min(x0, x1) >= sw or min(y0, y1) >= sh:
            continue
        if int(tracks["view_count"][tid]) >= int(strong_min_views):
            color = (80, 255, 120)
        else:
            color = (255, 210, 50)
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def main() -> None:
    args = _parse_args()
    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    primitive_dir = Path(args.primitive_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    weight_dir = Path(args.weight_dir).expanduser().resolve() if str(args.weight_dir) else None
    rgb_dir = Path(args.rgb_dir).expanduser().resolve() if str(args.rgb_dir) else None
    curve_image_dir = Path(args.curve_image_dir).expanduser().resolve() if str(args.curve_image_dir) else None

    if output_root.exists() and any(output_root.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output root is not empty; use --overwrite: {output_root}")
    if output_root.exists() and bool(args.overwrite):
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_ply = base_model_dir / "point_cloud" / f"iteration_{int(args.base_iteration)}" / "point_cloud.ply"
    if not base_ply.is_file():
        raise FileNotFoundError(f"Base PLY not found: {base_ply}")
    primitive_paths = _list_files(primitive_dir, [".npz"])
    if int(args.limit) > 0:
        primitive_paths = primitive_paths[: int(args.limit)]
    if not primitive_paths:
        raise RuntimeError(f"No primitive files found in {primitive_dir}")
    rgb_paths = _list_files(rgb_dir, IMAGE_EXTS) if rgb_dir is not None and rgb_dir.is_dir() else []
    rgb_lookup = _lookup(rgb_paths)
    weight_paths = _list_files(weight_dir, IMAGE_EXTS) if weight_dir is not None and weight_dir.is_dir() else []
    weight_lookup = _lookup(weight_paths)
    curve_paths = _list_files(curve_image_dir, IMAGE_EXTS) if curve_image_dir is not None and curve_image_dir.is_dir() else []
    curve_lookup = _lookup(curve_paths)
    if str(args.curve_source) == "skeleton" and str(args.curve_image_mode) != "weight" and not curve_paths:
        raise FileNotFoundError(f"curve_image_dir is required for SR-driven skeleton curves: {curve_image_dir}")
    cameras = _load_cameras(base_model_dir)
    _base_vertices, base_xyz, base_opacity, _base_rgb = _load_base_vertices(base_ply)

    chunks: List[Dict[str, np.ndarray]] = []
    per_view: List[Dict[str, object]] = []
    stems: List[str] = []
    source_sizes: List[Tuple[int, int]] = []
    for view_index, primitive_path in enumerate(tqdm(primitive_paths, desc="lift SR-HF curves")):
        lifted, info = _lift_view_segments(
            primitive_path,
            view_index,
            args,
            cameras,
            base_xyz,
            base_opacity,
            rgb_paths,
            rgb_lookup,
            weight_paths,
            weight_lookup,
            curve_paths,
            curve_lookup,
        )
        per_view.append(info)
        stems.append(primitive_path.stem)
        if "source_size" in info:
            source_sizes.append((int(info["source_size"][0]), int(info["source_size"][1])))
        else:
            source_sizes.append((1, 1))
        if lifted is None:
            print(f"[sr-hf-curve-tracks-v0] skip {view_index + 1}/{len(primitive_paths)} {primitive_path.stem}: {info['status']}")
            continue
        chunks.append(lifted)
        print(
            f"[sr-hf-curve-tracks-v0] {view_index + 1}/{len(primitive_paths)} {primitive_path.stem} "
            f"lifted={lifted['p0'].shape[0]}"
        )
    if not chunks:
        raise RuntimeError("No SR-HF segments were lifted.")

    segments_all = _concat_segments(chunks)
    segments_merge, merge_source_ids = _limit_segments_for_merge(segments_all, int(args.max_segments_for_merge))
    view_cameras: List[Optional[Dict[str, object]]] = []
    for view_index, stem in enumerate(stems):
        cam, _cam_match = _camera_for_view(cameras, stem, view_index, str(args.match_policy))
        view_cameras.append(cam)
    if str(args.track_build_mode) == "source_segments":
        tracks, keep = _tracks_from_source_segments(segments_merge, args)
        survive = keep & (tracks["view_count"] >= int(args.strong_track_min_views))
        probation = keep & ~survive
        suppress = ~keep
    elif str(args.track_build_mode) == "merge":
        tracks, keep = _merge_segments(segments_merge, args)
        survive = keep & (tracks["view_count"] >= int(args.strong_track_min_views))
        probation = keep & ~survive
        suppress = ~keep
    elif str(args.track_build_mode) == "layer_bins":
        tracks, keep = _aggregate_layer_bins(segments_merge, args)
        survive = keep & (tracks["view_count"] >= int(args.strong_track_min_views))
        probation = keep & ~survive
        suppress = ~keep
    elif str(args.track_build_mode) == "candidate_graph":
        tracks, keep, survive, probation, suppress = _candidate_graph_tracks(segments_merge, view_cameras, source_sizes, args)
    else:
        raise ValueError(f"Unsupported track_build_mode: {args.track_build_mode}")
    strong = keep & (tracks["view_count"] >= int(args.strong_track_min_views))

    np.savez_compressed(output_root / "segments_all_v0.npz", **segments_all)
    np.savez_compressed(output_root / "segments_for_merge_v0.npz", **segments_merge, source_segment_ids=merge_source_ids)
    np.savez_compressed(
        output_root / "sr_hf_curve_tracks_v0.npz",
        **tracks,
        keep=keep,
        strong=strong,
        survive=survive,
        probation=probation,
        suppress=suppress,
    )
    _write_obj(output_root / "tracks_keep_v0.obj", tracks, keep)
    _write_obj(output_root / "tracks_strong_v0.obj", tracks, strong)

    dirs = {
        "segment_overlay": output_root / "segment_overlay",
        "track_projection": output_root / "track_projection",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    for view_index, stem in enumerate(stems[: int(args.debug_limit)]):
        cam, _cam_match = _camera_for_view(cameras, stem, view_index, str(args.match_policy))
        if cam is None:
            continue
        size = source_sizes[view_index]
        _draw_segments_overlay(
            dirs["segment_overlay"] / f"{stem}.png",
            size,
            segments_all,
            view_index,
            int(args.max_draw_segments),
        )
        _draw_track_projection(
            dirs["track_projection"] / f"{stem}.png",
            size,
            cam,
            tracks,
            keep,
            int(args.max_draw_segments),
            int(args.strong_track_min_views),
        )

    summary = {
        "version": "build_sr_hf_curve_tracks_v0",
        "base_model_dir": str(base_model_dir),
        "base_iteration": int(args.base_iteration),
        "base_ply": str(base_ply),
        "primitive_dir": str(primitive_dir),
        "weight_dir": "" if weight_dir is None else str(weight_dir),
        "rgb_dir": "" if rgb_dir is None else str(rgb_dir),
        "curve_image_dir": "" if curve_image_dir is None else str(curve_image_dir),
        "output_root": str(output_root),
        "num_views": len(primitive_paths),
        "stems": stems,
        "per_view": per_view,
        "params": {
            "keep_kinds": str(args.keep_kinds),
            "max_primitives_per_view": int(args.max_primitives_per_view),
            "min_score": float(args.min_score),
            "min_weight": float(args.min_weight),
            "search_radius_px": int(args.search_radius_px),
            "endpoint_search_radius_px": int(args.endpoint_search_radius_px),
            "require_endpoint_match": bool(args.require_endpoint_match),
            "segment_length_scale": float(args.segment_length_scale),
            "max_segments_for_merge": int(args.max_segments_for_merge),
            "merge_radius_px": float(args.merge_radius_px),
            "merge_radius_abs": float(args.merge_radius_abs),
            "merge_angle_deg": float(args.merge_angle_deg),
            "merge_min_overlap": float(args.merge_min_overlap),
            "merge_same_view": bool(args.merge_same_view),
            "layer_bin_radius_px": float(args.layer_bin_radius_px),
            "layer_bin_radius_abs": float(args.layer_bin_radius_abs),
            "layer_dir_bins": int(args.layer_dir_bins),
            "layer_include_kind": bool(args.layer_include_kind),
            "candidate_radius_px": float(args.candidate_radius_px),
            "candidate_radius_abs": float(args.candidate_radius_abs),
            "candidate_reproj_radius_px": float(args.candidate_reproj_radius_px),
            "candidate_dir_angle_deg": float(args.candidate_dir_angle_deg),
            "candidate_normal_angle_deg": float(args.candidate_normal_angle_deg),
            "candidate_depth_delta_px": float(args.candidate_depth_delta_px),
            "candidate_min_survive_views": int(args.candidate_min_survive_views),
            "candidate_probation_min_source_strength": float(args.candidate_probation_min_source_strength),
            "candidate_probation_max_line_residual_px": float(args.candidate_probation_max_line_residual_px),
            "candidate_keep_probation": bool(args.candidate_keep_probation),
            "curve_source": str(args.curve_source),
            "curve_image_mode": str(args.curve_image_mode),
            "curve_highpass_blur_radius": float(args.curve_highpass_blur_radius),
            "curve_weight_power": float(args.curve_weight_power),
            "skeleton_threshold_percentile": float(args.skeleton_threshold_percentile),
            "skeleton_min_weight": float(args.skeleton_min_weight),
            "skeleton_min_path_pixels": int(args.skeleton_min_path_pixels),
            "skeleton_sample_step_px": float(args.skeleton_sample_step_px),
            "dense_stroke_enable": bool(args.dense_stroke_enable),
            "dense_stroke_threshold_percentile": float(args.dense_stroke_threshold_percentile),
            "dense_stroke_min_strength": float(args.dense_stroke_min_strength),
            "dense_stroke_grid_px": int(args.dense_stroke_grid_px),
            "dense_stroke_max_per_view": int(args.dense_stroke_max_per_view),
            "dense_stroke_length_px": float(args.dense_stroke_length_px),
            "dense_stroke_short_px": float(args.dense_stroke_short_px),
            "profile_width_enable": bool(args.profile_width_enable),
            "profile_width_radius_px": int(args.profile_width_radius_px),
            "profile_width_falloff": float(args.profile_width_falloff),
            "profile_width_min_px": float(args.profile_width_min_px),
            "profile_width_max_px": float(args.profile_width_max_px),
            "track_build_mode": str(args.track_build_mode),
            "min_track_segments": int(args.min_track_segments),
            "min_track_views": int(args.min_track_views),
            "strong_track_min_views": int(args.strong_track_min_views),
            "track_min_dir_consistency": float(args.track_min_dir_consistency),
            "track_max_line_residual_px": float(args.track_max_line_residual_px),
        },
        "counts": {
            "segments_all": int(segments_all["p0"].shape[0]),
            "segments_for_merge": int(segments_merge["p0"].shape[0]),
            "tracks_all": int(tracks["p0"].shape[0]),
            "tracks_keep": int(np.count_nonzero(keep)),
            "tracks_strong": int(np.count_nonzero(strong)),
            "tracks_survive": int(np.count_nonzero(survive)),
            "tracks_probation": int(np.count_nonzero(probation)),
            "tracks_suppress": int(np.count_nonzero(suppress)),
        },
        "stats": {
            "segment_score_mean": _mean(segments_all["score"]),
            "segment_weight_mean": _mean(segments_all["weight"]),
            "segment_length_mean": _mean(segments_all["length"]),
            "segment_width_px_mean": _mean(segments_all["source_width_px"]),
            "track_view_count_mean": _mean(tracks["view_count"].astype(np.float32)),
            "track_segment_count_mean": _mean(tracks["segment_count"].astype(np.float32)),
            "track_length_mean": _mean(tracks["length"]),
            "track_width_px_mean": _mean(tracks["width_px_mean"]),
            "track_source_strength_mean": _mean(tracks["source_strength"]),
            "track_neighbor_support_score_mean": _mean(tracks["neighbor_support_score"]),
            "track_reproject_error_px_mean": _mean(tracks["reproject_error_px"]),
            "track_reproject_dir_consistency_mean": _mean(tracks["reproject_dir_consistency"]),
            "track_dir_consistency_mean": _mean(tracks["dir_consistency"]),
            "track_line_residual_px_mean": _mean(tracks["line_residual_px"]),
            "track_support_view_ratio_mean": _mean(tracks["support_view_ratio"]),
            "keep_view_count_mean": _mean(tracks["view_count"][keep].astype(np.float32)),
            "strong_view_count_mean": _mean(tracks["view_count"][strong].astype(np.float32)),
            "survive_view_count_mean": _mean(tracks["view_count"][survive].astype(np.float32)),
            "probation_source_strength_mean": _mean(tracks["source_strength"][probation]),
        },
        "outputs": {
            "segments_all": str(output_root / "segments_all_v0.npz"),
            "segments_for_merge": str(output_root / "segments_for_merge_v0.npz"),
            "tracks": str(output_root / "sr_hf_curve_tracks_v0.npz"),
            "tracks_keep_obj": str(output_root / "tracks_keep_v0.obj"),
            "tracks_strong_obj": str(output_root / "tracks_strong_v0.obj"),
            "segment_overlay": str(dirs["segment_overlay"]),
            "track_projection": str(dirs["track_projection"]),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "stats": summary["stats"]}, indent=2))
    print(f"[sr-hf-curve-tracks-v0] summary: {output_root / 'summary.json'}")
    print(f"[sr-hf-curve-tracks-v0] tracks : {output_root / 'sr_hf_curve_tracks_v0.npz'}")
    print(f"[sr-hf-curve-tracks-v0] inspect: {dirs['track_projection']}")


if __name__ == "__main__":
    main()
