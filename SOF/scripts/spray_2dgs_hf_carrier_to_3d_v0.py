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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEPTH_EXTS = {".npz", ".npy", ".png", ".tiff", ".tif"}
SH_C0 = 0.28209479177387814
SOURCE_ORIGINAL = 0
SOURCE_PRIOR_INJECTED = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift 2DGS SR-HF carrier primitives into a newborn-only 3DGS PLY using gs2mesh-aligned depth. "
            "The final injection should be done by SOF/merge_gaussian_plys_v0.py so GaussianModel owns the update."
        )
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--depth_dir", required=True)
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
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--depth_key", default="auto")
    parser.add_argument("--depth_png_scale", type=float, default=1000.0)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--scale_min", type=float, default=5e-4)
    parser.add_argument("--scale_max", type=float, default=1.2e-2)
    parser.add_argument("--normal_scale_ratio", type=float, default=0.35)
    parser.add_argument("--normal_scale_min", type=float, default=4e-4)
    parser.add_argument("--normal_scale_max", type=float, default=3e-3)
    parser.add_argument("--opacity_floor", type=float, default=0.015)
    parser.add_argument("--opacity_scale", type=float, default=0.10)
    parser.add_argument("--opacity_power", type=float, default=0.75)
    parser.add_argument("--opacity_min", type=float, default=0.01)
    parser.add_argument("--opacity_max", type=float, default=0.12)
    parser.add_argument("--write_cpu_merged_preview", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def _list_files(root: Path, exts: Sequence[str]) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    ext_set = {x.lower() for x in exts}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ext_set)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(paths: Sequence[Path], lookup: Dict[str, Path], stem: str, index: int, policy: str) -> Optional[Path]:
    key = stem.lower()
    if policy in {"stem", "order_if_needed"}:
        found = lookup.get(key)
        if found is not None:
            return found
        if policy == "stem":
            return None
    if policy in {"order", "order_if_needed"} and index < len(paths):
        return paths[index]
    return None


def _load_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_depth(path: Path, depth_key: str, png_scale: float) -> Tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        return np.squeeze(arr), "npy"
    if suffix == ".npz":
        data = np.load(path)
        if depth_key != "auto":
            if depth_key not in data:
                raise KeyError(f"Depth key '{depth_key}' not found in {path}; keys={list(data.keys())}")
            arr = data[depth_key]
            return np.squeeze(arr).astype(np.float32), depth_key
        preferred = (
            "aligned_depth",
            "depth",
            "mesh_depth",
            "depth_prior_target",
            "arr_0",
        )
        for key in preferred:
            if key in data:
                arr = np.squeeze(data[key]).astype(np.float32)
                if arr.ndim == 2:
                    return arr, key
        for key in data.keys():
            arr = np.squeeze(data[key]).astype(np.float32)
            if arr.ndim == 2:
                return arr, key
        raise ValueError(f"No 2D depth array found in {path}; keys={list(data.keys())}")
    image = Image.open(path)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    scale = max(float(png_scale), 1e-8)
    return arr / scale, f"{suffix}_scale_{scale:g}"


def _bilinear_scalar(arr: np.ndarray, xy: np.ndarray, source_size: Tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[:2]
    sw, sh = source_size
    x = xy[:, 0] / max(float(sw - 1), 1.0) * max(float(w - 1), 1.0)
    y = xy[:, 1] / max(float(sh - 1), 1.0) * max(float(h - 1), 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    wx = (x - x0.astype(np.float32)).astype(np.float32)
    wy = (y - y0.astype(np.float32)).astype(np.float32)
    v00 = arr[y0, x0]
    v10 = arr[y0, x1]
    v01 = arr[y1, x0]
    v11 = arr[y1, x1]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    ).astype(np.float32)


def _bilinear_rgb(arr: np.ndarray, xy: np.ndarray, source_size: Tuple[int, int]) -> np.ndarray:
    channels = []
    for c in range(arr.shape[2]):
        channels.append(_bilinear_scalar(arr[..., c], xy, source_size))
    return np.stack(channels, axis=1).astype(np.float32)


def _infer_source_size(
    primitive: Dict[str, np.ndarray],
    rgb_path: Optional[Path],
    weight_path: Optional[Path],
) -> Tuple[int, int]:
    for path in (rgb_path, weight_path):
        if path is not None and path.is_file():
            with Image.open(path) as image:
                return image.size
    mu = np.asarray(primitive["mu_xy"], dtype=np.float32)
    width = int(math.ceil(float(np.nanmax(mu[:, 0])) + 1.0))
    height = int(math.ceil(float(np.nanmax(mu[:, 1])) + 1.0))
    return max(width, 1), max(height, 1)


def _load_cameras(base_model_dir: Path) -> Dict[str, Dict[str, object]]:
    path = base_model_dir / "cameras.json"
    if not path.is_file():
        raise FileNotFoundError(f"cameras.json not found: {path}")
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list):
        raise ValueError(f"Expected cameras.json list, got {type(cameras).__name__}")
    out: Dict[str, Dict[str, object]] = {}
    for i, cam in enumerate(cameras):
        name = str(cam.get("img_name", cam.get("image_name", "")))
        if not name:
            continue
        stem = Path(name).stem.lower()
        cam = dict(cam)
        cam["_index"] = i
        out[stem] = cam
    return out


def _camera_basis(cam: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    rot = np.asarray(cam["rotation"], dtype=np.float32)
    if rot.shape != (3, 3):
        raise ValueError(f"Invalid camera rotation shape: {rot.shape}")
    pos = np.asarray(cam["position"], dtype=np.float32)
    if pos.shape != (3,):
        raise ValueError(f"Invalid camera position shape: {pos.shape}")
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    width = int(cam["width"])
    height = int(cam["height"])
    return pos, rot[:, 0], rot[:, 1], rot[:, 2], fx, fy, width, height


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
    normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(normal_norm, 1e-8)
    return xyz, cam_x.astype(np.float32), cam_y.astype(np.float32), normal.astype(np.float32), px.astype(np.float32), fx, fy


def _cholesky_axes(cholesky: Optional[np.ndarray], count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cholesky is None or cholesky.shape[0] != count or cholesky.shape[1] < 3:
        theta = np.zeros((count,), dtype=np.float32)
        long_px = np.full((count,), 5.0, dtype=np.float32)
        short_px = np.full((count,), 0.8, dtype=np.float32)
        return theta, long_px, short_px
    theta = np.zeros((count,), dtype=np.float32)
    long_px = np.zeros((count,), dtype=np.float32)
    short_px = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        a = float(cholesky[i, 0])
        b = float(cholesky[i, 1])
        c = float(cholesky[i, 2])
        cov = np.asarray([[a * a, a * b], [a * b, b * b + c * c]], dtype=np.float32)
        vals, vecs = np.linalg.eigh(cov + np.eye(2, dtype=np.float32) * 1e-8)
        order = np.argsort(vals)
        small_i = int(order[0])
        large_i = int(order[-1])
        long_vec = vecs[:, large_i]
        theta[i] = math.atan2(float(long_vec[1]), float(long_vec[0]))
        long_px[i] = math.sqrt(max(float(vals[large_i]), 1e-8))
        short_px[i] = math.sqrt(max(float(vals[small_i]), 1e-8))
    return theta, long_px, short_px


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    quats = np.zeros((matrix.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(matrix.astype(np.float64)):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        else:
            diag = np.diag(m)
            if diag[0] > diag[1] and diag[0] > diag[2]:
                s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
                qw = (m[2, 1] - m[1, 2]) / s
                qx = 0.25 * s
                qy = (m[0, 1] + m[1, 0]) / s
                qz = (m[0, 2] + m[2, 0]) / s
            elif diag[1] > diag[2]:
                s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
                qw = (m[0, 2] - m[2, 0]) / s
                qx = (m[0, 1] + m[1, 0]) / s
                qy = 0.25 * s
                qz = (m[1, 2] + m[2, 1]) / s
            else:
                s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
                qw = (m[1, 0] - m[0, 1]) / s
                qx = (m[0, 2] + m[2, 0]) / s
                qy = (m[1, 2] + m[2, 1]) / s
                qz = 0.25 * s
        q = np.asarray([qw, qx, qy, qz], dtype=np.float32)
        quats[i] = q / max(float(np.linalg.norm(q)), 1e-8)
    return quats


def _rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return (rgb - 0.5) / SH_C0


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _copy_config(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json", "input.ply"):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def _make_tracking(base_count: int, new_count: int, base_tags_path: Path, iteration: int) -> Dict[str, torch.Tensor]:
    if base_tags_path.is_file():
        base_tags = torch.load(base_tags_path, map_location="cpu")
    else:
        base_tags = {}
    total = base_count + new_count
    source_tag = torch.zeros((total,), dtype=torch.int32)
    seed_id = torch.full((total,), -1, dtype=torch.int64)
    root_id = torch.full((total,), -1, dtype=torch.int64)
    generation = torch.zeros((total,), dtype=torch.int32)
    edge_touched = torch.zeros((total,), dtype=torch.bool)
    edge_touch_iter = torch.full((total,), -1, dtype=torch.int32)

    for name, tensor, dtype in (
        ("source_tag", source_tag, torch.int32),
        ("seed_id", seed_id, torch.int64),
        ("root_id", root_id, torch.int64),
        ("generation", generation, torch.int32),
        ("edge_touched", edge_touched, torch.bool),
        ("edge_touch_iter", edge_touch_iter, torch.int32),
    ):
        value = base_tags.get(name)
        if torch.is_tensor(value) and int(value.numel()) >= base_count:
            tensor[:base_count] = value[:base_count].to(dtype=dtype).reshape(-1)

    new_slice = slice(base_count, total)
    source_tag[new_slice] = int(SOURCE_PRIOR_INJECTED)
    seed_id[new_slice] = torch.arange(base_count, total, dtype=torch.int64)
    root_id[new_slice] = seed_id[new_slice]
    generation[new_slice] = 1
    edge_touched[new_slice] = True
    edge_touch_iter[new_slice] = int(iteration)
    return {
        "source_tag": source_tag,
        "seed_id": seed_id,
        "root_id": root_id,
        "generation": generation,
        "edge_touched": edge_touched,
        "edge_touch_iter": edge_touch_iter,
    }


def _fill_new_vertices(base_dtype: np.dtype, newborn: Dict[str, np.ndarray]) -> np.ndarray:
    n = int(newborn["xyz"].shape[0])
    out = np.zeros((n,), dtype=base_dtype)
    names = set(out.dtype.names or ())
    xyz = newborn["xyz"]
    if {"x", "y", "z"}.issubset(names):
        out["x"] = xyz[:, 0]
        out["y"] = xyz[:, 1]
        out["z"] = xyz[:, 2]
    for name in ("nx", "ny", "nz"):
        if name in names:
            out[name] = 0.0
    f_dc = _rgb_to_sh(newborn["color"])
    for i in range(3):
        name = f"f_dc_{i}"
        if name in names:
            out[name] = f_dc[:, i]
    for name in names:
        if name.startswith("f_rest_"):
            out[name] = 0.0
    if "opacity" in names:
        out["opacity"] = _logit(newborn["opacity"])
    scales = np.log(np.clip(newborn["scale"], 1e-8, None)).astype(np.float32)
    for i in range(scales.shape[1]):
        name = f"scale_{i}"
        if name in names:
            out[name] = scales[:, i]
    rot = newborn["rotation"]
    for i in range(rot.shape[1]):
        name = f"rot_{i}"
        if name in names:
            out[name] = rot[:, i]
    if "filter_3D" in names:
        out["filter_3D"] = 0.0
    return out


def _spray_view(
    primitive_path: Path,
    view_index: int,
    args: argparse.Namespace,
    cameras: Dict[str, Dict[str, object]],
    depth_paths: Sequence[Path],
    depth_lookup: Dict[str, Path],
    rgb_paths: Sequence[Path],
    rgb_lookup: Dict[str, Path],
    weight_paths: Sequence[Path],
    weight_lookup: Dict[str, Path],
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, object]]:
    stem = primitive_path.stem
    primitive = dict(np.load(primitive_path))
    if "mu_xy" not in primitive or "color" not in primitive:
        raise KeyError(f"{primitive_path} must contain mu_xy and color")
    mu = np.asarray(primitive["mu_xy"], dtype=np.float32)
    color = np.asarray(primitive["color"], dtype=np.float32)
    opacity_2d = np.asarray(primitive.get("opacity", np.ones((mu.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)

    cam = cameras.get(stem.lower())
    camera_match = "stem"
    if cam is None:
        if args.match_policy == "stem":
            return None, {"stem": stem, "status": "missing_camera"}
        camera_values = list(cameras.values())
        if view_index >= len(camera_values):
            return None, {"stem": stem, "status": "missing_camera"}
        cam = camera_values[view_index]
        camera_match = "order"

    depth_path = _resolve(depth_paths, depth_lookup, stem, view_index, args.match_policy)
    if depth_path is None:
        return None, {"stem": stem, "status": "missing_depth"}
    rgb_path = _resolve(rgb_paths, rgb_lookup, stem, view_index, args.match_policy) if rgb_paths else None
    weight_path = _resolve(weight_paths, weight_lookup, stem, view_index, args.match_policy) if weight_paths else None
    source_size = _infer_source_size(primitive, rgb_path, weight_path)

    depth, depth_key = _load_depth(depth_path, str(args.depth_key), float(args.depth_png_scale))
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze: {depth_path} shape={depth.shape}")
    depth_at_mu = _bilinear_scalar(depth, mu, source_size)
    weight = np.ones((mu.shape[0],), dtype=np.float32)
    if weight_path is not None and weight_path.is_file():
        weight_img = _load_gray(weight_path, size=source_size)
        weight = _bilinear_scalar(weight_img, mu, source_size)
    if rgb_path is not None and rgb_path.is_file():
        rgb_img = _load_rgb(rgb_path, size=source_size)
        # Primitive colors are optimized carrier features. Blend in the evidence image color only
        # where the exported primitive color is near black, which usually means an inactive primitive.
        sampled = _bilinear_rgb(rgb_img, mu, source_size)
        low_color = np.mean(np.abs(color), axis=1, keepdims=True) < 1e-4
        color = np.where(low_color, sampled, color)

    finite = np.isfinite(mu).all(axis=1) & np.isfinite(depth_at_mu) & (depth_at_mu > float(args.depth_min))
    finite &= np.isfinite(color).all(axis=1) & np.isfinite(opacity_2d)
    finite &= weight >= float(args.min_weight)
    finite &= opacity_2d >= float(args.min_primitive_opacity)
    keep_ids = np.flatnonzero(finite)
    if keep_ids.size == 0:
        return None, {
            "stem": stem,
            "status": "empty_after_filter",
            "depth": str(depth_path),
            "depth_key": depth_key,
            "weight_mean": float(weight.mean()) if weight.size else 0.0,
        }
    score = weight[keep_ids] * np.clip(opacity_2d[keep_ids], 0.0, 1.0)
    order = np.argsort(score)[::-1]
    if int(args.max_primitives_per_view) > 0:
        order = order[: int(args.max_primitives_per_view)]
    keep_ids = keep_ids[order]

    kept_mu = mu[keep_ids]
    kept_depth = depth_at_mu[keep_ids]
    kept_weight = weight[keep_ids]
    kept_opacity_2d = np.clip(opacity_2d[keep_ids], 0.0, 1.0)
    kept_color = np.clip(color[keep_ids], 0.0, 1.0)
    xyz, cam_x, cam_y, normal, _px, fx, fy = _unproject_points(cam, kept_mu, kept_depth, source_size)

    theta, long_px, short_px = _cholesky_axes(
        np.asarray(primitive["cholesky"], dtype=np.float32) if "cholesky" in primitive else None,
        mu.shape[0],
    )
    theta = theta[keep_ids]
    long_px = long_px[keep_ids]
    short_px = short_px[keep_ids]
    c = np.cos(theta).astype(np.float32)
    s = np.sin(theta).astype(np.float32)
    long_axis = _normalize(c[:, None] * cam_x[None, :] + s[:, None] * cam_y[None, :])
    short_axis = _normalize(-s[:, None] * cam_x[None, :] + c[:, None] * cam_y[None, :])
    normal_axis = _normalize(normal)
    rot_mats = np.stack([long_axis, short_axis, normal_axis], axis=2)
    quats = _matrix_to_quaternion_wxyz(rot_mats)

    pix_world = kept_depth / max((float(fx) + float(fy)) * 0.5, 1e-8)
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
        "source_view": np.full((keep_ids.size,), int(view_index), dtype=np.int32),
        "source_primitive": keep_ids.astype(np.int64),
        "weight": kept_weight.astype(np.float32),
        "primitive_opacity": kept_opacity_2d.astype(np.float32),
        "stem": np.asarray([stem] * keep_ids.size),
    }
    info = {
        "stem": stem,
        "status": "ok",
        "selected": int(keep_ids.size),
        "available": int(mu.shape[0]),
        "camera_match": camera_match,
        "camera_index": int(cam.get("_index", view_index)),
        "depth": str(depth_path),
        "depth_key": depth_key,
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "source_size": [int(source_size[0]), int(source_size[1])],
        "weight_mean": float(kept_weight.mean()),
        "opacity_mean": float(opacity.mean()),
        "scale_long_mean": float(scale_long.mean()),
        "scale_short_mean": float(scale_short.mean()),
    }
    return newborn, info


def main() -> None:
    args = _parse_args()
    base_model_dir = Path(args.base_model_dir)
    base_iter = int(args.base_iteration)
    base_point_dir = base_model_dir / "point_cloud" / f"iteration_{base_iter}"
    base_ply = base_point_dir / "point_cloud.ply"
    if not base_ply.is_file():
        raise FileNotFoundError(f"Base PLY not found: {base_ply}")
    primitive_dir = Path(args.primitive_dir)
    depth_dir = Path(args.depth_dir)
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
    depth_paths = _list_files(depth_dir, DEPTH_EXTS)
    depth_lookup = _lookup(depth_paths)
    rgb_paths = _list_files(Path(args.carrier_rgb_dir), IMAGE_EXTS) if args.carrier_rgb_dir else []
    rgb_lookup = _lookup(rgb_paths)
    weight_paths = _list_files(Path(args.carrier_weight_dir), IMAGE_EXTS) if args.carrier_weight_dir else []
    weight_lookup = _lookup(weight_paths)
    cameras = _load_cameras(base_model_dir)

    per_view: List[Dict[str, object]] = []
    chunks: List[Dict[str, np.ndarray]] = []
    total_new = 0
    for view_index, primitive_path in enumerate(primitive_paths):
        newborn, info = _spray_view(
            primitive_path,
            view_index,
            args,
            cameras,
            depth_paths,
            depth_lookup,
            rgb_paths,
            rgb_lookup,
            weight_paths,
            weight_lookup,
        )
        per_view.append(info)
        if newborn is None:
            print(f"[spray-2dgs-to-3d-v0] skip {view_index + 1}/{len(primitive_paths)} {primitive_path.stem}: {info['status']}")
            continue
        remaining = int(args.max_total_newborn) - total_new if int(args.max_total_newborn) > 0 else None
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and newborn["xyz"].shape[0] > remaining:
            keep = slice(0, remaining)
            newborn = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (newborn["xyz"].shape[0],) else v) for k, v in newborn.items()}
            info["selected_after_global_cap"] = int(remaining)
        chunks.append(newborn)
        total_new += int(newborn["xyz"].shape[0])
        print(
            f"[spray-2dgs-to-3d-v0] {view_index + 1}/{len(primitive_paths)} {primitive_path.stem} "
            f"new={newborn['xyz'].shape[0]} total_new={total_new}"
        )
    if not chunks:
        raise RuntimeError("No newborn gaussians were produced.")

    newborn_all: Dict[str, np.ndarray] = {}
    for key in ("xyz", "color", "opacity", "scale", "rotation", "source_view", "source_primitive", "weight", "primitive_opacity"):
        newborn_all[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)

    ply = PlyData.read(str(base_ply))
    base_vertices = ply["vertex"].data
    base_count = int(base_vertices.shape[0])
    new_vertices = _fill_new_vertices(base_vertices.dtype, newborn_all)
    _copy_config(base_model_dir, newborn_model_dir)
    newborn_point_dir = newborn_model_dir / "point_cloud" / f"iteration_{base_iter}"
    newborn_point_dir.mkdir(parents=True, exist_ok=True)
    newborn_ply = newborn_point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(new_vertices, "vertex")]).write(str(newborn_ply))

    newborn_tags = _make_tracking(0, int(new_vertices.shape[0]), Path("__missing__"), base_iter)
    torch.save(newborn_tags, newborn_point_dir / "gaussian_tags.pt")
    np.savez_compressed(
        newborn_point_dir / "sprayed_2dgs_metadata_v0.npz",
        source_view=newborn_all["source_view"],
        source_primitive=newborn_all["source_primitive"],
        weight=newborn_all["weight"],
        primitive_opacity=newborn_all["primitive_opacity"],
        xyz=newborn_all["xyz"],
        scale=newborn_all["scale"],
        color=newborn_all["color"],
    )

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
        "version": "spray_2dgs_hf_carrier_to_3d_v0",
        "mode": "newborn_only_for_gaussianmodel_merge",
        "base_model_dir": str(base_model_dir),
        "base_iteration": base_iter,
        "base_ply": str(base_ply),
        "depth_dir": str(depth_dir),
        "primitive_dir": str(primitive_dir),
        "carrier_rgb_dir": str(args.carrier_rgb_dir),
        "carrier_weight_dir": str(args.carrier_weight_dir),
        "output_model_dir": str(output_model_dir),
        "newborn_model_dir": str(newborn_model_dir),
        "newborn_ply": str(newborn_ply),
        "newborn_tags": str(newborn_point_dir / "gaussian_tags.pt"),
        "newborn_metadata": str(newborn_point_dir / "sprayed_2dgs_metadata_v0.npz"),
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
            "scale_multiplier": float(args.scale_multiplier),
            "scale_min": float(args.scale_min),
            "scale_max": float(args.scale_max),
            "opacity_floor": float(args.opacity_floor),
            "opacity_scale": float(args.opacity_scale),
            "opacity_max": float(args.opacity_max),
        },
        "newborn_stats": {
            "weight_mean": float(newborn_all["weight"].mean()),
            "opacity_mean": float(newborn_all["opacity"].mean()),
            "scale_mean": [float(x) for x in newborn_all["scale"].mean(axis=0)],
            "scale_p90": [float(x) for x in np.percentile(newborn_all["scale"], 90, axis=0)],
            "color_mean": [float(x) for x in newborn_all["color"].mean(axis=0)],
        },
    }
    (output_model_dir / "spray_2dgs_hf_carrier_to_3d_v0_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "base_gaussians": summary["base_gaussians"],
        "newborn_gaussians": summary["newborn_gaussians"],
        "expected_total_after_merge": summary["expected_total_after_merge"],
        "newborn_model_dir": summary["newborn_model_dir"],
        "newborn_ply": summary["newborn_ply"],
    }, indent=2))
    print(f"[spray-2dgs-to-3d-v0] summary: {output_model_dir / 'spray_2dgs_hf_carrier_to_3d_v0_summary.json'}")


if __name__ == "__main__":
    main()
