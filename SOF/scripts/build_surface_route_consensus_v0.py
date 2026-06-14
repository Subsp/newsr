#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image
from torch import nn

SOF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIPSPLATTING_ROOT = SOF_ROOT.parent / "mip-splatting"


def _ensure_mipsplatting_imports(mipsplatting_root: Path) -> None:
    root = str(mipsplatting_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=1,
        white_background=white_background,
        data_device="cuda",
        eval=True,
        kernel_size=0.1,
        ray_jitter=False,
        resample_gt_image=False,
        load_allres=False,
        sample_more_highres=False,
    )


def _build_pipe_args() -> Namespace:
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        compute_filter3D_python=False,
        compute_view2gaussian_python=False,
        use_merged_sof_rasterizer=False,
        use_vanilla_sof_rasterizer=False,
        require_merged_sof_aux=False,
        debug=False,
    )


def _resolve_iteration(model_path: Path, iteration: int) -> int:
    if int(iteration) >= 0:
        return int(iteration)
    point_root = model_path / "point_cloud"
    candidates: List[int] = []
    for child in point_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_root}")
    return max(candidates)


def _tensor_1d(value: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().cpu().reshape(-1)
    else:
        out = torch.as_tensor(value).reshape(-1)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _torch_load(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_surface_mask(mask_payload_path: Path, mask_key: str, expected_count: int) -> torch.Tensor:
    payload = _torch_load(mask_payload_path)
    if mask_key not in payload:
        class_id = payload.get("class_id")
        if class_id is not None:
            class_id = _tensor_1d(class_id, dtype=torch.long)
            if int(class_id.shape[0]) != int(expected_count):
                raise ValueError(
                    f"class_id length mismatch in {mask_payload_path}: "
                    f"expected {expected_count}, got {int(class_id.shape[0])}"
                )
            if mask_key == "no_mesh_neutral":
                return class_id == 0
            if mask_key == "surface_carrier":
                return class_id == 1
            if mask_key == "near_surface_uncertain":
                return class_id == 2
            if mask_key == "off_surface_near_mesh":
                return class_id == 3
            if mask_key == "axis_touching_surface":
                return class_id == 4
            if mask_key == "low_opacity_neutral":
                return class_id == 5
            if mask_key == "surface_candidate":
                return (class_id == 1) | (class_id == 4)
            if mask_key == "surface_or_uncertain":
                return (class_id == 1) | (class_id == 2) | (class_id == 4)
        raise KeyError(f"Mask key '{mask_key}' not found in {mask_payload_path}")
    mask = _tensor_1d(payload[mask_key], dtype=torch.bool)
    if int(mask.shape[0]) != int(expected_count):
        raise ValueError(
            f"Mask length mismatch for key '{mask_key}': expected {expected_count}, got {int(mask.shape[0])}"
        )
    return mask


def _select_uniform(items: Sequence[object], max_items: int):
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items), list(range(len(items)))
    ids = np.unique(np.linspace(0, len(items) - 1, num=int(max_items), dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()], [int(idx) for idx in ids.tolist()]


def _safe_stem(path: str) -> str:
    return Path(path).stem


def _extract_trailing_int(name: str) -> int | None:
    import re

    match = re.search(r"(\d+)$", str(name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _sort_key_from_stem(stem: str) -> Tuple[int, int | str]:
    idx = _extract_trailing_int(stem)
    if idx is None:
        return (1, stem)
    return (0, idx)


def _sort_key_from_camera(camera) -> Tuple[int, int | str]:
    idx = _extract_trailing_int(camera.image_name)
    if idx is None:
        return (1, camera.image_name)
    return (0, idx)


def _build_external_image_index(train_cameras, root_dir: Path, exts: Sequence[str]) -> Dict[str, str]:
    root = root_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"External image root not found: {root}")

    ext_set = {f".{str(ext).lower().lstrip('.')}" for ext in exts}
    stem_to_path: Dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in ext_set:
            continue
        stem_to_path.setdefault(child.stem, str(child))

    index: Dict[str, str] = {}
    used_paths: set[str] = set()

    for camera in train_cameras:
        picked = stem_to_path.get(camera.image_name)
        if picked is None or picked in used_paths:
            continue
        index[camera.image_name] = picked
        used_paths.add(picked)

    unmatched = [cam for cam in train_cameras if cam.image_name not in index]
    idx_to_paths: Dict[int, List[str]] = {}
    for stem, path in sorted(stem_to_path.items(), key=lambda item: _sort_key_from_stem(item[0])):
        idx = _extract_trailing_int(stem)
        if idx is None:
            continue
        idx_to_paths.setdefault(idx, []).append(path)

    for camera in unmatched:
        idx = _extract_trailing_int(camera.image_name)
        if idx is None:
            continue
        candidates = [path for path in idx_to_paths.get(idx, []) if path not in used_paths]
        if not candidates:
            continue
        picked = candidates[0]
        index[camera.image_name] = picked
        used_paths.add(picked)

    unmatched = [cam for cam in train_cameras if cam.image_name not in index]
    if unmatched:
        remaining_paths = []
        for stem, path in sorted(stem_to_path.items(), key=lambda item: _sort_key_from_stem(item[0])):
            if path in used_paths:
                continue
            remaining_paths.append(path)
        unmatched_sorted = sorted(unmatched, key=_sort_key_from_camera)
        pair_count = min(len(unmatched_sorted), len(remaining_paths))
        for i in range(pair_count):
            index[unmatched_sorted[i].image_name] = remaining_paths[i]
    return index


def _load_rgb(path: str, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    if tensor.shape[-2:] != (height, width):
        tensor = torch_F.interpolate(
            tensor[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
    return tensor


def _load_mask(path: str, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None].contiguous()
    if tensor.shape[-2:] != (height, width):
        tensor = torch_F.interpolate(
            tensor[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
    return tensor


def _laplacian_highfreq(image_chw: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=image_chw.dtype,
        device=image_chw.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(image_chw.shape[0], 1, 1, 1)
    return torch_F.conv2d(image_chw[None], kernel, padding=1, groups=image_chw.shape[0])[0]


def _avg_lowpass(image_chw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size <= 1:
        return image_chw
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    return torch_F.avg_pool2d(
        image_chw[None],
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )[0]


def _compute_lowfreq_gate(
    prior_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    kernel_size: int,
    tau: float,
) -> torch.Tensor:
    tau = max(float(tau), 1e-6)
    prior_lp = _avg_lowpass(prior_rgb, int(kernel_size))
    anchor_lp = _avg_lowpass(anchor_rgb, int(kernel_size))
    diff = (prior_lp - anchor_lp).abs().mean(dim=0)
    return torch.exp(-diff / tau).clamp(0.0, 1.0)


def _clip_signed_delta(image_chw: torch.Tensor, clip_value: float) -> torch.Tensor:
    clip_value = float(clip_value)
    if clip_value <= 0.0:
        return image_chw
    return image_chw.clamp(min=-clip_value, max=clip_value)


def _clone_subset_gaussians(base, mask: torch.Tensor):
    from scene.gaussian_model import GaussianModel

    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    subset = GaussianModel(base.max_sh_degree)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[mask].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[mask].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[mask].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[mask].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[mask].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[mask].clone().requires_grad_(False))
    if isinstance(getattr(base, "filter_3D", None), torch.Tensor) and int(base.filter_3D.shape[0]) == int(base.get_xyz.shape[0]):
        subset.filter_3D = base.filter_3D.detach()[mask].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    if hasattr(subset, "xyz_gradient_accum_abs"):
        subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    if hasattr(subset, "xyz_gradient_accum_abs_max"):
        subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    for name in ("_source_tag", "_seed_id", "_generation", "_edge_touched", "_edge_touch_iter"):
        value = getattr(base, name, None)
        if torch.is_tensor(value) and int(value.shape[0]) == int(base.get_xyz.shape[0]):
            setattr(subset, name, value.detach()[mask].clone())
    return subset


def _project_to_pixels(xyz: torch.Tensor, camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = torch.tensor(camera.R, device=xyz.device, dtype=xyz.dtype)
    T = torch.tensor(camera.T, device=xyz.device, dtype=xyz.dtype)
    xyz_cam = xyz @ R + T[None, :]
    z = xyz_cam[:, 2].clamp(min=1e-6)
    x = xyz_cam[:, 0] / z * float(camera.focal_x) + float(camera.image_width) / 2.0
    y = xyz_cam[:, 1] / z * float(camera.focal_y) + float(camera.image_height) / 2.0
    return x.detach().cpu().numpy(), y.detach().cpu().numpy(), z.detach().cpu().numpy()


def _select_candidate_pixels(
    *,
    signal_energy: np.ndarray,
    confidence_map: np.ndarray,
    min_prior_mask: float,
    min_residual_energy: float,
    max_candidate_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_valid = (
        (confidence_map >= float(min_prior_mask))
        & (signal_energy >= float(min_residual_energy))
    )
    if not np.any(seed_valid):
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    ys, xs = np.nonzero(seed_valid)
    scores = (confidence_map[ys, xs] * signal_energy[ys, xs]).astype(np.float32, copy=False)
    if int(max_candidate_pixels) > 0 and ys.shape[0] > int(max_candidate_pixels):
        keep = int(max_candidate_pixels)
        partition = np.argpartition(-scores, keep - 1)[:keep]
        order = np.argsort(-scores[partition], kind="stable")
        picked = partition[order]
        ys = ys[picked]
        xs = xs[picked]
        scores = scores[picked]
    return ys.astype(np.int32, copy=False), xs.astype(np.int32, copy=False), scores.astype(np.float32, copy=False)


def _build_gaussian_tile_bins(
    *,
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    radii: np.ndarray,
    opacity: np.ndarray,
    width: int,
    height: int,
    tile_size: int,
    radius_scale: float,
    min_radius_px: float,
) -> Tuple[List[List[int]], np.ndarray, int, int]:
    tile_size = max(int(tile_size), 4)
    tiles_x = max(int(math.ceil(float(width) / float(tile_size))), 1)
    tiles_y = max(int(math.ceil(float(height) / float(tile_size))), 1)
    tile_bins: List[List[int]] = [[] for _ in range(tiles_x * tiles_y)]
    effective_radii = np.maximum(
        radii.astype(np.float32, copy=False) * float(radius_scale),
        float(min_radius_px),
    ).astype(np.float32, copy=False)

    visible_ids = np.nonzero(radii > 0.0)[0].tolist()
    for local_id in visible_ids:
        cx = float(proj_x[local_id])
        cy = float(proj_y[local_id])
        radius = float(effective_radii[local_id])
        alpha = float(opacity[local_id])
        if not np.isfinite(cx) or not np.isfinite(cy) or alpha <= 1e-6:
            continue
        x0 = max(0, int(math.floor(cx - radius)))
        x1 = min(width - 1, int(math.ceil(cx + radius)))
        y0 = max(0, int(math.floor(cy - radius)))
        y1 = min(height - 1, int(math.ceil(cy + radius)))
        if x1 < x0 or y1 < y0:
            continue
        tx0 = max(0, x0 // tile_size)
        tx1 = min(tiles_x - 1, x1 // tile_size)
        ty0 = max(0, y0 // tile_size)
        ty1 = min(tiles_y - 1, y1 // tile_size)
        for ty in range(ty0, ty1 + 1):
            row_offset = ty * tiles_x
            for tx in range(tx0, tx1 + 1):
                tile_bins[row_offset + tx].append(int(local_id))
    return tile_bins, effective_radii, tiles_x, tiles_y


def _query_routes_for_pixels(
    *,
    cand_y: np.ndarray,
    cand_x: np.ndarray,
    confidence_map: np.ndarray,
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    effective_radii: np.ndarray,
    opacity: np.ndarray,
    tile_bins: List[List[int]],
    tiles_x: int,
    tile_size: int,
    top_k: int,
    cell_grid: int,
    min_route_quality: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(top_k) not in (1, 2):
        raise ValueError(f"Only top_k in {{1,2}} is supported in v0, got {top_k}")

    route_keys = np.full((cand_y.shape[0],), -1, dtype=np.int64)
    route_quality = np.zeros((cand_y.shape[0],), dtype=np.float32)
    valid = np.zeros((cand_y.shape[0],), dtype=bool)
    if cand_y.shape[0] == 0:
        return route_keys, route_quality, valid

    route_base = int(proj_x.shape[0]) + 1
    for idx in range(cand_y.shape[0]):
        y = int(cand_y[idx])
        x = int(cand_x[idx])
        tx = max(int(x // tile_size), 0)
        ty = max(int(y // tile_size), 0)
        candidate_ids = tile_bins[ty * tiles_x + tx]
        if not candidate_ids:
            continue

        scores: List[Tuple[float, int]] = []
        for local_id in candidate_ids:
            radius = max(float(effective_radii[local_id]), 1e-6)
            dx = (float(x) - float(proj_x[local_id])) / radius
            dy = (float(y) - float(proj_y[local_id])) / radius
            dist2 = dx * dx + dy * dy
            if dist2 > 1.0:
                continue
            score = float(opacity[local_id]) * math.exp(-0.5 * dist2)
            if score <= 1e-8:
                continue
            scores.append((score, int(local_id)))

        if not scores:
            continue
        scores.sort(key=lambda item: item[0], reverse=True)
        top1_score, top1_id = scores[0]
        if int(top_k) > 1 and len(scores) > 1:
            top2_score, top2_id = scores[1]
        else:
            top2_score, top2_id = 0.0, -1

        score_sum = max(top1_score + top2_score, 1e-6)
        top1_dominance = top1_score / score_sum
        quality = float(confidence_map[y, x]) * float(top1_dominance) * float(np.clip(top1_score, 0.0, 1.0))
        if quality < float(min_route_quality):
            continue

        radius = max(float(effective_radii[top1_id]), 1.0)
        dx = float(np.clip((float(x) - float(proj_x[top1_id])) / radius, -1.0, 1.0))
        dy = float(np.clip((float(y) - float(proj_y[top1_id])) / radius, -1.0, 1.0))
        cell_x = int(np.clip(math.floor((dx + 1.0) * 0.5 * float(cell_grid)), 0, int(cell_grid) - 1))
        cell_y = int(np.clip(math.floor((dy + 1.0) * 0.5 * float(cell_grid)), 0, int(cell_grid) - 1))
        route_key = (
            (
                (int(top1_id) * route_base + (int(top2_id) + 1))
                * int(cell_grid)
                + int(cell_x)
            )
            * int(cell_grid)
            + int(cell_y)
        )
        route_keys[idx] = np.int64(route_key)
        route_quality[idx] = np.float32(quality)
        valid[idx] = True
    return route_keys, route_quality, valid


def _aggregate_view_routes(
    cand_y: np.ndarray,
    cand_x: np.ndarray,
    route_keys: np.ndarray,
    route_quality: np.ndarray,
    valid_mask: np.ndarray,
    residual_hwc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(valid_mask):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    keys = torch.from_numpy(route_keys[valid_mask].astype(np.int64))
    weights = torch.from_numpy(route_quality[valid_mask].astype(np.float32))
    residual = torch.from_numpy(residual_hwc[cand_y[valid_mask], cand_x[valid_mask]].astype(np.float32))
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    count = int(unique_keys.shape[0])

    sum_w = torch.zeros((count,), dtype=torch.float32)
    sum_rgb = torch.zeros((count, 3), dtype=torch.float32)
    pixel_count = torch.zeros((count,), dtype=torch.int32)
    sum_w.index_add_(0, inverse, weights)
    sum_rgb.index_add_(0, inverse, residual * weights[:, None])
    pixel_count.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int32))
    mean_rgb = sum_rgb / sum_w.clamp(min=1e-6)[:, None]
    return (
        unique_keys.numpy(),
        mean_rgb.numpy(),
        sum_w.numpy(),
        pixel_count.numpy(),
    )


def _build_route_consensus(route_observations: Dict[int, List[Tuple[np.ndarray, float]]], min_views: int, var_tau: float):
    route_keys = np.array(sorted(route_observations.keys()), dtype=np.int64)
    if route_keys.size == 0:
        return route_keys, np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32), {}

    fused = np.zeros((route_keys.shape[0], 3), dtype=np.float32)
    conf = np.zeros((route_keys.shape[0],), dtype=np.float32)
    stats: Dict[int, Dict[str, float]] = {}

    for idx, route_key in enumerate(route_keys.tolist()):
        entries = route_observations[int(route_key)]
        means = np.stack([entry[0] for entry in entries], axis=0).astype(np.float32)
        weights = np.asarray([entry[1] for entry in entries], dtype=np.float32)
        safe_weights = np.clip(weights, a_min=1e-6, a_max=None)
        fused_rgb = np.sum(means * safe_weights[:, None], axis=0) / np.sum(safe_weights)
        sq = ((means - fused_rgb[None, :]) ** 2).mean(axis=1)
        variance = float(np.sum(sq * safe_weights) / np.sum(safe_weights))
        view_count = int(len(entries))
        view_gate = min(1.0, float(view_count) / max(int(min_views), 1))
        confidence = float(view_gate * math.exp(-variance / max(float(var_tau), 1e-6)))
        fused[idx] = fused_rgb
        conf[idx] = confidence
        stats[int(route_key)] = {
            "view_count": float(view_count),
            "variance": variance,
            "confidence": confidence,
        }

    return route_keys, fused, conf, stats


def _window_bounds(index: int, total: int, window_size: int) -> Tuple[int, int]:
    total = max(int(total), 0)
    if total <= 0:
        return 0, 0
    window_size = max(1, min(int(window_size), total))
    left = max(0, int(index) - window_size // 2)
    right = left + window_size
    if right > total:
        right = total
        left = max(0, right - window_size)
    return left, right


def _build_local_route_observations(view_caches: Sequence[Dict[str, object]]) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    route_observations: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for view_cache in view_caches:
        route_keys = view_cache["route_unique_keys"]
        route_mean_rgb = view_cache["route_mean_rgb"]
        route_sum_w = view_cache["route_sum_w"]
        for key, rgb, weight in zip(route_keys.tolist(), route_mean_rgb.tolist(), route_sum_w.tolist()):
            route_observations.setdefault(int(key), []).append((np.asarray(rgb, dtype=np.float32), float(weight)))
    return route_observations


def _save_gray(path: Path, image_hw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_hw, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def _save_rgb(path: Path, image_hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_hwc, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build surface route consensus targets for masked SR prior injection.")
    parser.add_argument("--mipsplatting_root", default=str(DEFAULT_MIPSPLATTING_ROOT))
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--surface_state_payload", required=True)
    parser.add_argument("--surface_mask_key", default="surface_candidate")
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--prior_mask_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--cell_grid", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=32)
    parser.add_argument("--max_candidate_pixels", type=int, default=6000)
    parser.add_argument("--local_consensus_views", type=int, default=3)
    parser.add_argument("--route_radius_scale", type=float, default=2.5)
    parser.add_argument("--route_min_radius_px", type=float, default=1.5)
    parser.add_argument("--min_prior_mask", type=float, default=0.05)
    parser.add_argument("--min_route_quality", type=float, default=0.08)
    parser.add_argument("--min_residual_energy", type=float, default=0.01)
    parser.add_argument("--route_min_views", type=int, default=2)
    parser.add_argument("--route_var_tau", type=float, default=0.01)
    parser.add_argument("--prior_delta_clip", type=float, default=0.20)
    parser.add_argument("--anchor_mode", type=str, default="full", choices=["surface", "full"])
    parser.add_argument(
        "--signal_mode",
        type=str,
        default="direct_sr_highfreq",
        choices=["direct_sr_highfreq", "anchor_residual"],
    )
    parser.add_argument("--lowfreq_gate_kernel", type=int, default=9)
    parser.add_argument("--lowfreq_gate_tau", type=float, default=0.08)
    parser.add_argument("--sparse_payload", action="store_true")
    parser.add_argument("--save_debug_png", action="store_true")
    args = parser.parse_args()

    mipsplatting_root = Path(args.mipsplatting_root).expanduser().resolve()
    _ensure_mipsplatting_imports(mipsplatting_root)

    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    surface_state_payload = Path(args.surface_state_payload).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_iteration(model_path, int(args.iteration))
    dataset = _build_dataset_args(str(scene_root), str(model_path), str(args.images_subdir), bool(args.white_background))
    pipe = _build_pipe_args()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)

    total_gaussians = int(gaussians.get_xyz.shape[0])
    surface_mask = _load_surface_mask(surface_state_payload, str(args.surface_mask_key), total_gaussians)
    surface_selected = int(surface_mask.sum().item())
    if surface_selected <= 0:
        raise ValueError(f"Mask '{args.surface_mask_key}' selected zero gaussians.")

    train_cameras = list(scene.getTrainCameras())
    selected_views, selected_indices = _select_uniform(train_cameras, int(args.max_views))

    prior_index = _build_external_image_index(selected_views, Path(args.prior_dir), ("png", "jpg", "jpeg", "webp"))
    prior_mask_index = _build_external_image_index(selected_views, Path(args.prior_mask_dir), ("png", "jpg", "jpeg", "webp"))
    if len(prior_index) == 0:
        raise ValueError(f"No prior images matched train cameras under {args.prior_dir}")
    if len(prior_mask_index) == 0:
        raise ValueError(f"No prior masks matched train cameras under {args.prior_mask_dir}")

    device = gaussians.get_xyz.device
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    surface_subset = _clone_subset_gaussians(gaussians, surface_mask.to(device=device))
    surface_subset.compute_3D_filter(train_cameras)
    surface_opacity = surface_subset.get_opacity_with_3D_filter.detach().cpu().numpy().reshape(-1)

    first_pass_views: List[Dict[str, object]] = []
    cached_views: List[Dict[str, object]] = []

    print(
        f"[surface-route-consensus-v0] scene={scene_root} model={model_path} views={len(selected_views)} "
        f"surface={surface_selected}/{total_gaussians} lowfreq_anchor={args.anchor_mode} "
        f"signal={args.signal_mode}",
        flush=True,
    )

    with torch.no_grad():
        for view_idx, camera in zip(selected_indices, selected_views):
            if camera.image_name not in prior_index or camera.image_name not in prior_mask_index:
                print(f"[surface-route-consensus-v0] skip {camera.image_name}: missing prior or mask", flush=True)
                continue

            render_full = render(
                camera,
                gaussians,
                pipe,
                background,
                kernel_size=float(dataset.kernel_size),
            )["render"][:3].detach().cpu()
            render_surface = render(
                camera,
                surface_subset,
                pipe,
                background,
                kernel_size=float(dataset.kernel_size),
            )
            surface_rgb = render_surface["render"][:3].detach().cpu()
            surface_radii = render_surface["radii"].detach().cpu().numpy().astype(np.float32, copy=False)
            if str(args.anchor_mode) == "surface":
                anchor_rgb_t = surface_rgb
            else:
                anchor_rgb_t = render_full

            width = int(camera.image_width)
            height = int(camera.image_height)
            prior_rgb = _load_rgb(prior_index[camera.image_name], width=width, height=height)
            prior_mask = _load_mask(prior_mask_index[camera.image_name], width=width, height=height)

            lowfreq_gate = _compute_lowfreq_gate(
                prior_rgb=prior_rgb,
                anchor_rgb=anchor_rgb_t,
                kernel_size=int(args.lowfreq_gate_kernel),
                tau=float(args.lowfreq_gate_tau),
            ).cpu()
            surface_highfreq = _laplacian_highfreq(surface_rgb).cpu()
            prior_highfreq = _laplacian_highfreq(prior_rgb).cpu()
            if str(args.signal_mode) == "anchor_residual":
                prior_delta = _clip_signed_delta(prior_rgb - surface_rgb, float(args.prior_delta_clip))
                route_signal_highfreq = _laplacian_highfreq(prior_delta).cpu()
            else:
                route_signal_highfreq = prior_highfreq
            signal_energy = route_signal_highfreq.abs().mean(dim=0).numpy()
            prior_mask_hw = prior_mask.squeeze(0).numpy()
            confidence_hw = prior_mask_hw * lowfreq_gate.numpy()
            cand_y, cand_x, cand_score = _select_candidate_pixels(
                signal_energy=signal_energy,
                confidence_map=confidence_hw,
                min_prior_mask=float(args.min_prior_mask),
                min_residual_energy=float(args.min_residual_energy),
                max_candidate_pixels=int(args.max_candidate_pixels),
            )

            proj_x, proj_y, _ = _project_to_pixels(surface_subset.get_xyz, camera)
            tile_bins, effective_radii, tiles_x, _ = _build_gaussian_tile_bins(
                proj_x=proj_x,
                proj_y=proj_y,
                radii=surface_radii,
                opacity=surface_opacity,
                width=width,
                height=height,
                tile_size=int(args.tile_size),
                radius_scale=float(args.route_radius_scale),
                min_radius_px=float(args.route_min_radius_px),
            )
            route_keys, route_quality, valid = _query_routes_for_pixels(
                cand_y=cand_y,
                cand_x=cand_x,
                confidence_map=confidence_hw,
                proj_x=proj_x,
                proj_y=proj_y,
                effective_radii=effective_radii,
                opacity=surface_opacity,
                tile_bins=tile_bins,
                tiles_x=tiles_x,
                tile_size=int(args.tile_size),
                top_k=int(args.top_k),
                cell_grid=int(args.cell_grid),
                min_route_quality=float(args.min_route_quality),
            )
            unique_keys, mean_rgb, sum_w, pixel_count = _aggregate_view_routes(
                cand_y=cand_y,
                cand_x=cand_x,
                route_keys=route_keys,
                route_quality=route_quality,
                valid_mask=valid,
                residual_hwc=route_signal_highfreq.permute(1, 2, 0).numpy(),
            )
            anchor_highfreq = _laplacian_highfreq(anchor_rgb_t).cpu().permute(1, 2, 0).numpy()
            surface_anchor_highfreq = surface_highfreq.permute(1, 2, 0).numpy()
            cached_views.append(
                {
                    "image_name": camera.image_name,
                    "width": width,
                    "height": height,
                    "cand_y": cand_y,
                    "cand_x": cand_x,
                    "candidate_score": cand_score,
                    "route_keys": route_keys,
                    "route_quality": route_quality,
                    "valid": valid,
                    "route_unique_keys": unique_keys,
                    "route_mean_rgb": mean_rgb,
                    "route_sum_w": sum_w,
                    "anchor_highfreq_samples": anchor_highfreq[cand_y, cand_x].astype(np.float32, copy=False),
                    "surface_anchor_highfreq_samples": surface_anchor_highfreq[cand_y, cand_x].astype(
                        np.float32, copy=False
                    ),
                    "lowfreq_gate_samples": lowfreq_gate.numpy()[cand_y, cand_x].astype(np.float32, copy=False),
                    "anchor_rgb": anchor_rgb_t.permute(1, 2, 0).numpy().astype(np.float32, copy=False),
                }
            )

            first_pass_views.append(
                {
                    "image_name": camera.image_name,
                    "selected_index": int(view_idx),
                    "width": width,
                    "height": height,
                    "matched_prior": prior_index[camera.image_name],
                    "matched_mask": prior_mask_index[camera.image_name],
                    "candidate_pixels": int(cand_y.shape[0]),
                    "valid_pixels": int(valid.sum()),
                    "route_count": int(unique_keys.shape[0]),
                    "surface_mean": float(surface_rgb.mean().item()),
                    "anchor_mean": float(anchor_rgb_t.mean().item()),
                    "mask_mean": float(prior_mask.mean().item()),
                    "lowfreq_gate_mean": float(lowfreq_gate.mean().item()),
                    "valid_ratio": float(valid.sum() / max(int(cand_y.shape[0]), 1)),
                    "total_route_weight": float(route_quality[valid].sum()) if np.any(valid) else 0.0,
                }
            )
            print(
                f"[surface-route-consensus-v0] first-pass view={camera.image_name} "
                f"candidate={int(cand_y.shape[0])} valid={int(valid.sum())} routes={int(unique_keys.shape[0])}",
                flush=True,
            )

    global_route_key_set = set()
    for view_cache in cached_views:
        for route_key in view_cache["route_unique_keys"].tolist():
            global_route_key_set.add(int(route_key))
    route_keys_sorted = np.array(sorted(global_route_key_set), dtype=np.int64)

    per_view_root = output_root / "per_view"
    per_view_root.mkdir(parents=True, exist_ok=True)
    debug_root = output_root / "debug"
    manifest_views: List[Dict[str, object]] = []
    local_consensus_views = max(1, int(args.local_consensus_views))

    for view_idx, view_cache in enumerate(cached_views):
        window_left, window_right = _window_bounds(view_idx, len(cached_views), local_consensus_views)
        local_route_observations = _build_local_route_observations(cached_views[window_left:window_right])
        local_route_keys, local_fused_residuals, local_route_conf, _ = _build_route_consensus(
            route_observations=local_route_observations,
            min_views=int(args.route_min_views),
            var_tau=float(args.route_var_tau),
        )
        route_conf_lookup = {
            int(route_key): (local_fused_residuals[idx], float(local_route_conf[idx]))
            for idx, route_key in enumerate(local_route_keys.tolist())
        }

        width = int(view_cache["width"])
        height = int(view_cache["height"])
        cand_y = view_cache["cand_y"]
        cand_x = view_cache["cand_x"]
        route_keys = view_cache["route_keys"]
        route_quality = view_cache["route_quality"]
        valid = view_cache["valid"]
        anchor_highfreq_samples = view_cache["anchor_highfreq_samples"]
        surface_anchor_highfreq_samples = view_cache["surface_anchor_highfreq_samples"]

        target_highfreq = np.zeros((height, width, 3), dtype=np.float32)
        target_residual = np.zeros((height, width, 3), dtype=np.float32)
        surface_anchor_highfreq = np.zeros((height, width, 3), dtype=np.float32)
        target_weight_hw = np.zeros((height, width), dtype=np.float32)

        valid_ids = np.nonzero(valid)[0]
        matched_pixels = 0
        for local_idx in valid_ids.tolist():
            entry = route_conf_lookup.get(int(route_keys[local_idx]))
            if entry is None:
                continue
            fused_rgb, conf_value = entry
            y = int(cand_y[local_idx])
            x = int(cand_x[local_idx])
            if str(args.signal_mode) == "anchor_residual":
                target_highfreq[y, x] = fused_rgb + surface_anchor_highfreq_samples[local_idx]
            else:
                target_highfreq[y, x] = fused_rgb
            target_residual[y, x] = fused_rgb
            surface_anchor_highfreq[y, x] = surface_anchor_highfreq_samples[local_idx]
            target_weight_hw[y, x] = float(route_quality[local_idx]) * float(conf_value)
            matched_pixels += 1

        if bool(args.sparse_payload):
            nonzero_y, nonzero_x = np.nonzero(target_weight_hw > 0.0)
            payload = {
                "image_name": str(view_cache["image_name"]),
                "height": int(height),
                "width": int(width),
                "sample_y": torch.from_numpy(nonzero_y.astype(np.int32, copy=False)),
                "sample_x": torch.from_numpy(nonzero_x.astype(np.int32, copy=False)),
                "target_highfreq_samples": torch.from_numpy(
                    target_highfreq[nonzero_y, nonzero_x].astype(np.float16, copy=False)
                ),
                "target_residual_samples": torch.from_numpy(
                    target_residual[nonzero_y, nonzero_x].astype(np.float16, copy=False)
                ),
                "surface_anchor_highfreq_samples": torch.from_numpy(
                    surface_anchor_highfreq[nonzero_y, nonzero_x].astype(np.float16, copy=False)
                ),
                "target_weight_samples": torch.from_numpy(
                    target_weight_hw[nonzero_y, nonzero_x].astype(np.float16, copy=False)
                ),
                "signal_mode": str(args.signal_mode),
            }
        else:
            payload = {
                "image_name": str(view_cache["image_name"]),
                "target_highfreq": torch.from_numpy(target_highfreq.transpose(2, 0, 1)).to(dtype=torch.float16),
                "target_residual": torch.from_numpy(target_residual.transpose(2, 0, 1)).to(dtype=torch.float16),
                "surface_anchor_highfreq": torch.from_numpy(
                    surface_anchor_highfreq.transpose(2, 0, 1)
                ).to(dtype=torch.float16),
                "target_weight": torch.from_numpy(target_weight_hw[None]).to(dtype=torch.float16),
                "signal_mode": str(args.signal_mode),
            }
        output_path = per_view_root / f"{view_cache['image_name']}.pt"
        torch.save(payload, output_path)

        if bool(args.save_debug_png):
            _save_gray(debug_root / "weight" / f"{view_cache['image_name']}.png", target_weight_hw)
            target_abs = np.clip(np.abs(target_highfreq).mean(axis=2) * 4.0, 0.0, 1.0)
            _save_gray(debug_root / "target_abs" / f"{view_cache['image_name']}.png", target_abs)
            candidate_mask = np.zeros((height, width), dtype=np.float32)
            if cand_y.shape[0] > 0:
                candidate_mask[cand_y, cand_x] = 1.0
            _save_gray(debug_root / "candidate" / f"{view_cache['image_name']}.png", candidate_mask)
            _save_rgb(debug_root / "anchor" / f"{view_cache['image_name']}.png", view_cache["anchor_rgb"])

        matched_ratio = float(np.mean(target_weight_hw > 0.0))
        manifest_views.append(
            {
                "image_name": str(view_cache["image_name"]),
                "path": str(output_path),
                "width": width,
                "height": height,
                "candidate_pixels": int(cand_y.shape[0]),
                "matched_ratio": matched_ratio,
                "nonzero_weight_pixels": int((target_weight_hw > 0.0).sum()),
            }
        )
        print(
            f"[surface-route-consensus-v0] save view={view_cache['image_name']} "
            f"matched={matched_pixels} ratio={matched_ratio:.4f} "
            f"window={window_right - window_left}",
            flush=True,
        )

    summary = {
        "version": "surface_route_consensus_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "mipsplatting_root": str(mipsplatting_root),
        "surface_state_payload": str(surface_state_payload),
        "surface_mask_key": str(args.surface_mask_key),
        "prior_dir": str(Path(args.prior_dir).expanduser().resolve()),
        "prior_mask_dir": str(Path(args.prior_mask_dir).expanduser().resolve()),
        "output_root": str(output_root),
        "images_subdir": str(args.images_subdir),
        "iteration": int(loaded_iter),
        "counts": {
            "source_gaussians": int(total_gaussians),
            "surface_gaussians": int(surface_selected),
            "selected_views": int(len(manifest_views)),
            "route_keys": int(route_keys_sorted.shape[0]),
        },
        "params": {
            "top_k": int(args.top_k),
            "cell_grid": int(args.cell_grid),
            "tile_size": int(args.tile_size),
            "max_candidate_pixels": int(args.max_candidate_pixels),
            "local_consensus_views": int(args.local_consensus_views),
            "route_radius_scale": float(args.route_radius_scale),
            "route_min_radius_px": float(args.route_min_radius_px),
            "min_prior_mask": float(args.min_prior_mask),
            "min_route_quality": float(args.min_route_quality),
            "min_residual_energy": float(args.min_residual_energy),
            "route_min_views": int(args.route_min_views),
            "route_var_tau": float(args.route_var_tau),
            "prior_delta_clip": float(args.prior_delta_clip),
            "anchor_mode": str(args.anchor_mode),
            "lowfreq_gate_kernel": int(args.lowfreq_gate_kernel),
            "lowfreq_gate_tau": float(args.lowfreq_gate_tau),
            "signal_mode": str(args.signal_mode),
            "sparse_payload": bool(args.sparse_payload),
        },
        "views": manifest_views,
        "first_pass_views": first_pass_views,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2), flush=True)
    print(f"[surface-route-consensus-v0] manifest: {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
