import json
import math
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import render_simple
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_points3D_binary, read_points3D_text
from scene.dataset_readers import fetchPly, storePly
from train_mip_to_sof_surface_v0 import (
    charbonnier,
    compute_depth_prior_distortion_loss,
    compute_depth_prior_self_normal_loss,
    load_cameras_for_split,
    load_depth_prior_for_view,
    masked_weighted_mean,
    normalize_normal,
    parse_subdir_list,
    scheduled_loss_scale,
)
from utils.depth_utils import depth_to_normal
from utils.general_utils import get_expon_lr_func, safe_state
from utils.graphics_utils import BasicPointCloud
from utils.prior_injection import load_rgb_image
from utils.system_utils import mkdir_p


def camera_extent_from_views(cameras: Sequence[object]) -> float:
    centers = []
    for camera in cameras:
        center = camera.camera_center
        if torch.is_tensor(center):
            center = center.detach().cpu().numpy()
        else:
            center = np.asarray(center)
        centers.append(np.asarray(center, dtype=np.float32).reshape(3))
    if not centers:
        return 1.0
    centers_np = np.stack(centers, axis=0)
    mean_center = np.mean(centers_np, axis=0, keepdims=True)
    diagonal = float(np.linalg.norm(centers_np - mean_center, axis=1).max())
    return max(diagonal * 1.1, 1e-6)


def select_uniform_indices(num_items: int, max_items: int) -> List[int]:
    if max_items <= 0 or num_items <= max_items:
        return list(range(num_items))
    ids = np.unique(np.linspace(0, num_items - 1, num=max_items, dtype=np.int64))
    return [int(idx) for idx in ids.tolist()]


def pixel_grid_stride_mask(height: int, width: int, stride: int, phase: int = 0) -> torch.Tensor:
    stride = max(int(stride), 1)
    if stride <= 1:
        return torch.ones((height, width), dtype=torch.bool)
    ys = torch.arange(height, dtype=torch.int64)
    xs = torch.arange(width, dtype=torch.int64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    phase = int(phase) % stride
    return ((grid_y + phase) % stride == 0) & ((grid_x + phase) % stride == 0)


def depth_pixels_to_world(camera, xy_depth: np.ndarray) -> np.ndarray:
    z = xy_depth[:, 2].astype(np.float32, copy=False)
    x_cam = (xy_depth[:, 0].astype(np.float32, copy=False) - float(camera.image_width) / 2.0) / float(camera.focal_x) * z
    y_cam = (xy_depth[:, 1].astype(np.float32, copy=False) - float(camera.image_height) / 2.0) / float(camera.focal_y) * z
    xyz_cam = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32, copy=False)
    r = np.asarray(camera.R, dtype=np.float32)
    t = np.asarray(camera.T, dtype=np.float32)
    return ((xyz_cam - t[None, :]) @ r.T).astype(np.float32, copy=False)


def world_points_to_pixels(camera, xyz_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    r = np.asarray(camera.R, dtype=np.float32)
    t = np.asarray(camera.T, dtype=np.float32)
    xyz_cam = xyz_world @ r + t[None, :]
    z = xyz_cam[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-6)

    x = np.zeros_like(z, dtype=np.float32)
    y = np.zeros_like(z, dtype=np.float32)
    if np.any(valid_z):
        x_valid = xyz_cam[valid_z, 0] / z[valid_z]
        y_valid = xyz_cam[valid_z, 1] / z[valid_z]
        x[valid_z] = x_valid * float(camera.focal_x) + float(camera.image_width) / 2.0
        y[valid_z] = y_valid * float(camera.focal_y) + float(camera.image_height) / 2.0

    valid = (
        valid_z
        & np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= float(camera.image_width - 1))
        & (y >= 0.0)
        & (y <= float(camera.image_height - 1))
    )
    return x, y, z, valid


def load_scene_sparse_pointcloud(scene_root: Path) -> BasicPointCloud | None:
    sparse_root = scene_root / "sparse" / "0"
    if not sparse_root.is_dir():
        return None
    ply_path = sparse_root / "points3D.ply"
    if not ply_path.is_file():
        bin_path = sparse_root / "points3D.bin"
        txt_path = sparse_root / "points3D.txt"
        if bin_path.is_file():
            xyz, rgb, _ = read_points3D_binary(str(bin_path))
            storePly(str(ply_path), xyz, rgb)
        elif txt_path.is_file():
            xyz, rgb, _ = read_points3D_text(str(txt_path))
            storePly(str(ply_path), xyz, rgb)
        else:
            return None
    return fetchPly(str(ply_path))


def build_sfm_fallback_points(
    scene_root: Path,
    cameras: Sequence[object],
    teacher_cache: Sequence[Dict[str, object]],
    *,
    max_points: int,
    base_weight: float,
    only_missing: bool,
    min_visible_views: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    sparse_pcd = load_scene_sparse_pointcloud(scene_root)
    if sparse_pcd is None:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32), {
            "enabled": False,
            "reason": "missing_sparse_points3D",
            "raw_points": 0,
            "kept_points": 0,
        }

    points = np.asarray(sparse_pcd.points, dtype=np.float32)
    colors = np.asarray(sparse_pcd.colors, dtype=np.float32)
    raw_points = int(points.shape[0])
    if raw_points == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32), {
            "enabled": False,
            "reason": "empty_sparse_points3D",
            "raw_points": 0,
            "kept_points": 0,
        }

    if int(max_points) > 0 and raw_points > int(max_points):
        keep_ids = np.unique(np.linspace(0, raw_points - 1, num=int(max_points), dtype=np.int64))
        points = points[keep_ids]
        colors = colors[keep_ids]

    visible_counts = np.zeros((points.shape[0],), dtype=np.int32)
    missing_counts = np.zeros((points.shape[0],), dtype=np.int32)
    color_sum = np.zeros((points.shape[0], 3), dtype=np.float32)
    color_hits = np.zeros((points.shape[0],), dtype=np.int32)

    for camera, target in zip(cameras, teacher_cache):
        x, y, _, valid = world_points_to_pixels(camera, points)
        if not np.any(valid):
            continue

        idx = np.flatnonzero(valid)
        xi = np.rint(x[idx]).astype(np.int64, copy=False)
        yi = np.rint(y[idx]).astype(np.int64, copy=False)

        depth_prior_mask = np.asarray(target["depth_prior_mask"], dtype=bool)
        hole_mask = ~depth_prior_mask[yi, xi]
        visible_counts[idx] += 1

        use_mask = hole_mask if bool(only_missing) else np.ones_like(hole_mask, dtype=bool)
        if not np.any(use_mask):
            continue
        picked = idx[use_mask]
        missing_counts[picked] += 1

        mip_rgb_hwc = np.asarray(target["mip_rgb"].permute(1, 2, 0), dtype=np.float32)
        picked_x = xi[use_mask]
        picked_y = yi[use_mask]
        color_sum[picked] += mip_rgb_hwc[picked_y, picked_x]
        color_hits[picked] += 1

    if bool(only_missing):
        keep_mask = missing_counts >= max(int(min_visible_views), 1)
    else:
        keep_mask = visible_counts >= max(int(min_visible_views), 1)

    if not np.any(keep_mask):
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32), {
            "enabled": True,
            "reason": "no_visible_hole_points",
            "raw_points": int(raw_points),
            "kept_points": 0,
        }

    kept_points = points[keep_mask].astype(np.float32, copy=False)
    kept_colors = colors[keep_mask].astype(np.float32, copy=False)
    if np.any(color_hits[keep_mask] > 0):
        recolor = color_hits[keep_mask] > 0
        kept_colors[recolor] = color_sum[keep_mask][recolor] / np.clip(
            color_hits[keep_mask][recolor, None].astype(np.float32),
            1.0,
            None,
        )

    coverage = np.zeros((keep_mask.sum(),), dtype=np.float32)
    coverage_src = missing_counts if bool(only_missing) else visible_counts
    denom_src = np.maximum(visible_counts[keep_mask], 1)
    coverage = coverage_src[keep_mask].astype(np.float32) / denom_src.astype(np.float32)
    kept_weights = np.clip(float(base_weight) * np.clip(coverage, 0.25, 1.0), 1e-4, 1.0).astype(np.float32, copy=False)
    return kept_points, np.clip(kept_colors, 0.0, 1.0).astype(np.float32, copy=False), kept_weights, {
        "enabled": True,
        "raw_points": int(raw_points),
        "sampled_points": int(points.shape[0]),
        "kept_points": int(kept_points.shape[0]),
        "only_missing": bool(only_missing),
        "min_visible_views": int(min_visible_views),
        "weight_mean": float(kept_weights.mean()) if kept_weights.size else 0.0,
    }


def resize_chw(image_chw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(image_chw.shape[-2:]) == tuple(target_hw):
        return image_chw
    return F.interpolate(
        image_chw.unsqueeze(0).float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0]


def load_render_image_chw(path: Path, target_hw: Tuple[int, int]) -> torch.Tensor:
    image = load_rgb_image(path).permute(2, 0, 1).contiguous()
    return resize_chw(image, target_hw)


def list_render_images(render_root: Path) -> List[Path]:
    image_paths = [
        path
        for path in sorted(render_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not image_paths:
        raise FileNotFoundError(f"No render images found under {render_root}")
    return image_paths


def write_clean_cfg_args(output_model_path: Path, scene_root: Path, images_subdir: str, sh_degree: int) -> None:
    payload = Namespace(
        sh_degree=int(sh_degree),
        source_path=str(scene_root),
        model_path=str(output_model_path),
        images=str(images_subdir),
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )
    with open(output_model_path / "cfg_args", "w", encoding="utf-8") as f:
        f.write(repr(payload))


def weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    *,
    voxel_size: float,
    min_points_per_voxel: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or float(voxel_size) <= 0.0:
        counts = np.ones((points.shape[0],), dtype=np.int32)
        return points, colors, weights, counts

    voxel_ids = np.floor(points / float(voxel_size)).astype(np.int64, copy=False)
    unique_ids, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
    n_voxels = unique_ids.shape[0]

    weight_sum = np.bincount(inverse, weights=weights, minlength=n_voxels).astype(np.float32)
    count_sum = np.bincount(inverse, minlength=n_voxels).astype(np.int32)

    point_sum = np.stack(
        [np.bincount(inverse, weights=points[:, axis] * weights, minlength=n_voxels) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    color_sum = np.stack(
        [np.bincount(inverse, weights=colors[:, axis] * weights, minlength=n_voxels) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    color_plain_sum = np.stack(
        [np.bincount(inverse, weights=colors[:, axis], minlength=n_voxels) for axis in range(3)],
        axis=1,
    ).astype(np.float32)

    denom = np.clip(weight_sum[:, None], 1e-8, None)
    points_ds = point_sum / denom
    colors_ds = color_sum / denom
    zero_weight = weight_sum <= 1e-8
    if np.any(zero_weight):
        colors_ds[zero_weight] = color_plain_sum[zero_weight] / np.clip(count_sum[zero_weight, None], 1, None)
    keep = count_sum >= max(int(min_points_per_voxel), 1)
    if not np.any(keep):
        keep = count_sum > 0
    return (
        points_ds[keep].astype(np.float32, copy=False),
        np.clip(colors_ds[keep], 0.0, 1.0).astype(np.float32, copy=False),
        weight_sum[keep].astype(np.float32, copy=False),
        count_sum[keep].astype(np.int32, copy=False),
    )


def topk_by_weight(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    *,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(max_points) <= 0 or points.shape[0] <= int(max_points):
        return points, colors, weights, counts
    order = np.argsort(-weights)
    keep = order[: int(max_points)]
    keep = np.sort(keep)
    return points[keep], colors[keep], weights[keep], counts[keep]


@torch.no_grad()
def build_mip_rgb_depthprior_cache(
    cameras: Sequence[object],
    mip_render_paths: Sequence[Path],
    *,
    depth_prior_root: Path | None,
    depth_prior_subdirs: Sequence[str],
    depth_prior_confidence_subdirs: Sequence[str],
    depth_prior_confidence_min: float,
    depth_prior_agreement_threshold: float,
    depth_prior_agreement_floor: float,
    depth_prior_align_mode: str,
    depth_prior_align_min_pixels: int,
    depth_prior_surface_weight_boost: float,
    depth_prior_weight_gain: float,
    depth_prior_weight_power: float,
    depth_prior_weight_min: float,
    depth_min: float,
) -> List[Dict[str, object]]:
    if len(cameras) != len(mip_render_paths):
        raise ValueError("cameras and mip_render_paths must have the same length")
    if str(depth_prior_align_mode) != "identity":
        raise ValueError(
            "RGB-only depthprior reinit no longer uses teacher depth; set --depth_prior_align_mode identity."
        )

    cache: List[Dict[str, object]] = []
    for idx, (camera, render_path) in enumerate(zip(cameras, mip_render_paths), start=1):
        target_hw = (int(camera.image_height), int(camera.image_width))
        mip_rgb = load_render_image_chw(render_path, target_hw).to(dtype=torch.float32)

        rgb_mask = torch.ones(target_hw, dtype=torch.bool)
        rgb_weight = torch.ones(target_hw, dtype=torch.float32)
        surface_mask = torch.zeros(target_hw, dtype=torch.bool)
        depth_prior_mask = torch.zeros(target_hw, dtype=torch.bool)
        depth_prior_weight = torch.zeros(target_hw, dtype=torch.float32)
        depth_prior_target = None
        depth_prior_normal = None
        depth_prior_info: Dict[str, object] = {"status": "disabled"}

        if depth_prior_root is not None:
            raw_depth_prior, raw_depth_conf, depth_prior_info = load_depth_prior_for_view(
                camera.image_name,
                depth_prior_root=depth_prior_root,
                target_hw=target_hw,
                depth_subdirs=depth_prior_subdirs,
                confidence_subdirs=depth_prior_confidence_subdirs,
            )
            if raw_depth_prior is not None:
                depth_prior = raw_depth_prior.to(dtype=torch.float32)
                if raw_depth_conf is None:
                    depth_prior_conf = torch.ones_like(depth_prior)
                else:
                    depth_prior_conf = raw_depth_conf.to(dtype=torch.float32).clamp(0.0, 1.0)

                aligned_depth = depth_prior
                depth_prior_info["align"] = {
                    "mode": "identity",
                    "pixels": int(
                        (
                            torch.isfinite(aligned_depth)
                            & (aligned_depth > float(depth_min))
                            & (depth_prior_conf >= float(depth_prior_confidence_min))
                        )
                        .sum()
                        .item()
                    ),
                    "scale": 1.0,
                    "shift": 0.0,
                    "note": "rgb_only_no_teacher_depth",
                }
                if float(depth_prior_agreement_threshold) > 0.0 or float(depth_prior_agreement_floor) > 0.0:
                    depth_prior_info["agreement"] = {
                        "mode": "disabled_without_teacher_depth",
                        "threshold": float(depth_prior_agreement_threshold),
                        "floor": float(depth_prior_agreement_floor),
                    }

                depth_valid = (
                    torch.isfinite(aligned_depth)
                    & (aligned_depth > float(depth_min))
                    & (depth_prior_conf >= float(depth_prior_confidence_min))
                )
                surface_mask = depth_valid
                depth_prior_weight = depth_prior_conf
                if float(depth_prior_weight_power) > 0.0 and float(depth_prior_weight_power) != 1.0:
                    depth_prior_weight = depth_prior_weight.clamp_min(0.0).pow(float(depth_prior_weight_power))
                if float(depth_prior_weight_gain) != 1.0:
                    depth_prior_weight = depth_prior_weight * float(depth_prior_weight_gain)
                depth_prior_weight = depth_prior_weight.clamp(0.0, 1.0)
                if float(depth_prior_weight_min) > 0.0:
                    depth_prior_weight = torch.where(
                        depth_prior_weight >= float(depth_prior_weight_min),
                        depth_prior_weight,
                        torch.zeros_like(depth_prior_weight),
                    )
                depth_prior_weight = torch.where(
                    depth_valid,
                    depth_prior_weight,
                    torch.zeros_like(depth_prior_weight),
                )
                depth_prior_mask = depth_prior_weight > 0.0
                depth_prior_target = aligned_depth.unsqueeze(0).detach()
                if bool(depth_prior_mask.any()):
                    depth_normal_hw3, _ = depth_to_normal(camera, aligned_depth.unsqueeze(0))
                    depth_prior_normal = normalize_normal(depth_normal_hw3.permute(2, 0, 1).detach())
                if float(depth_prior_surface_weight_boost) > 0.0:
                    rgb_weight = rgb_weight + float(depth_prior_surface_weight_boost) * depth_prior_weight

        rgb_weight = rgb_weight.clamp(0.0, 1.0 + max(float(depth_prior_surface_weight_boost), 0.0))
        cache.append(
            {
                "image_name": str(camera.image_name),
                "mip_rgb": mip_rgb.half().cpu(),
                "mip_render_path": str(render_path),
                "rgb_mask": rgb_mask.cpu(),
                "rgb_weight": rgb_weight.half().cpu(),
                "surface_mask": surface_mask.cpu(),
                "depth_prior_target": depth_prior_target.half().cpu() if torch.is_tensor(depth_prior_target) else None,
                "depth_prior_normal": depth_prior_normal.half().cpu() if torch.is_tensor(depth_prior_normal) else None,
                "depth_prior_mask": depth_prior_mask.cpu(),
                "depth_prior_weight": depth_prior_weight.half().cpu(),
                "depth_prior_info": depth_prior_info,
            }
        )
        print(
            f"[mip-depthprior-reinit] cached rgb view {idx}/{len(cameras)} "
            f"rgb=1.0000 depth_prior={float(depth_prior_mask.float().mean().item()):.4f}",
            flush=True,
        )
    return cache


def build_pointcloud_from_depth_prior(
    scene_root: Path,
    cameras: Sequence[object],
    teacher_cache: Sequence[Dict[str, object]],
    *,
    init_max_views: int,
    init_pixel_stride: int,
    init_min_weight: float,
    init_voxel_size: float,
    init_voxel_size_factor: float,
    init_min_points_per_voxel: int,
    init_max_points: int,
    init_sfm_fallback: bool,
    init_sfm_max_points: int,
    init_sfm_weight: float,
    init_sfm_only_missing: bool,
    init_sfm_min_visible_views: int,
) -> Tuple[BasicPointCloud, Dict[str, object]]:
    selected_ids = list(range(len(cameras)))
    if int(init_max_views) > 0 and len(selected_ids) > int(init_max_views):
        selected_ids = np.unique(np.linspace(0, len(cameras) - 1, num=int(init_max_views), dtype=np.int64)).tolist()

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    per_view_counts: List[Dict[str, object]] = []

    for local_idx, view_idx in enumerate(selected_ids):
        camera = cameras[view_idx]
        target = teacher_cache[view_idx]
        depth_prior_target = target.get("depth_prior_target")
        if not torch.is_tensor(depth_prior_target):
            per_view_counts.append(
                {
                    "view_index": int(view_idx),
                    "image_name": str(target["image_name"]),
                    "points_before_downsample": 0,
                }
            )
            continue

        depth_mask = target["depth_prior_mask"]
        depth_weight = target["depth_prior_weight"]
        mip_rgb = target["mip_rgb"]
        if not torch.is_tensor(depth_mask) or not torch.is_tensor(depth_weight) or not torch.is_tensor(mip_rgb):
            continue

        mask = depth_mask.to(dtype=torch.bool)
        if int(init_pixel_stride) > 1:
            mask = mask & pixel_grid_stride_mask(mask.shape[0], mask.shape[1], int(init_pixel_stride), phase=local_idx)
        if float(init_min_weight) > 0.0:
            mask = mask & (depth_weight >= float(init_min_weight))
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            per_view_counts.append(
                {
                    "view_index": int(view_idx),
                    "image_name": str(target["image_name"]),
                    "points_before_downsample": 0,
                }
            )
            continue

        depth_hw = depth_prior_target[0]
        xy_depth = torch.stack(
            [
                xs.to(dtype=torch.float32),
                ys.to(dtype=torch.float32),
                depth_hw[ys, xs].to(dtype=torch.float32),
            ],
            dim=1,
        ).cpu().numpy()
        xyz_world = depth_pixels_to_world(camera, xy_depth)
        rgb = mip_rgb[:, ys, xs].permute(1, 0).cpu().numpy().astype(np.float32, copy=False)
        weights = depth_weight[ys, xs].cpu().numpy().astype(np.float32, copy=False)
        all_points.append(xyz_world)
        all_colors.append(rgb)
        all_weights.append(weights)
        per_view_counts.append(
            {
                "view_index": int(view_idx),
                "image_name": str(target["image_name"]),
                "points_before_downsample": int(xyz_world.shape[0]),
                "weight_mean": float(weights.mean()) if weights.size else 0.0,
            }
        )

    if not all_points:
        raise RuntimeError("Depth prior did not produce any valid initialization points.")

    points = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32, copy=False)
    weights = np.concatenate(all_weights, axis=0).astype(np.float32, copy=False)
    sfm_summary: Dict[str, object] = {"enabled": False, "kept_points": 0}
    if bool(init_sfm_fallback) and float(init_sfm_weight) > 0.0:
        sfm_points, sfm_colors, sfm_weights, sfm_summary = build_sfm_fallback_points(
            scene_root,
            [cameras[idx] for idx in selected_ids],
            [teacher_cache[idx] for idx in selected_ids],
            max_points=int(init_sfm_max_points),
            base_weight=float(init_sfm_weight),
            only_missing=bool(init_sfm_only_missing),
            min_visible_views=int(init_sfm_min_visible_views),
        )
        if sfm_points.shape[0] > 0:
            points = np.concatenate([points, sfm_points], axis=0).astype(np.float32, copy=False)
            colors = np.concatenate([colors, sfm_colors], axis=0).astype(np.float32, copy=False)
            weights = np.concatenate([weights, sfm_weights], axis=0).astype(np.float32, copy=False)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    voxel_size = float(init_voxel_size)
    if voxel_size <= 0.0 and float(init_voxel_size_factor) > 0.0:
        voxel_size = max(bbox_diag * float(init_voxel_size_factor), 1e-6)

    points, colors, weights, counts = weighted_voxel_downsample(
        points,
        colors,
        weights,
        voxel_size=float(voxel_size),
        min_points_per_voxel=int(init_min_points_per_voxel),
    )
    points, colors, weights, counts = topk_by_weight(
        points,
        colors,
        weights,
        counts,
        max_points=int(init_max_points),
    )
    if points.shape[0] == 0:
        raise RuntimeError("All initialization points were filtered out after voxel downsampling.")

    pcd = BasicPointCloud(
        points=points.astype(np.float32, copy=False),
        colors=np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
        normals=np.zeros_like(points, dtype=np.float32),
    )
    summary = {
        "views_used": int(len(selected_ids)),
        "raw_points": int(sum(item["points_before_downsample"] for item in per_view_counts)),
        "final_points": int(points.shape[0]),
        "bbox_diag": float(bbox_diag),
        "voxel_size": float(voxel_size),
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_p90": float(np.percentile(weights, 90.0)) if weights.size else 0.0,
        "count_mean": float(counts.mean()) if counts.size else 0.0,
        "per_view": per_view_counts,
        "sfm_fallback": sfm_summary,
    }
    return pcd, summary


def build_optimizer(student: GaussianModel, args, spatial_lr_scale: float) -> Tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.Adam(
        [
            {"params": [student._xyz], "lr": float(args.xyz_lr_init) * float(spatial_lr_scale), "name": "xyz"},
            {"params": [student._features_dc], "lr": float(args.feature_lr), "name": "f_dc"},
            {"params": [student._features_rest], "lr": float(args.feature_rest_lr), "name": "f_rest"},
            {"params": [student._opacity], "lr": float(args.opacity_lr), "name": "opacity"},
            {"params": [student._scaling], "lr": float(args.scaling_lr), "name": "scaling"},
            {"params": [student._rotation], "lr": float(args.rotation_lr), "name": "rotation"},
        ],
        lr=0.0,
        eps=1e-15,
    )
    xyz_scheduler = get_expon_lr_func(
        lr_init=float(args.xyz_lr_init) * float(spatial_lr_scale),
        lr_final=float(args.xyz_lr_final) * float(spatial_lr_scale),
        lr_delay_mult=float(args.xyz_lr_delay_mult),
        max_steps=int(args.iterations),
    )
    return optimizer, xyz_scheduler


def save_checkpoint(student: GaussianModel, output_dir: Path, iteration: int) -> None:
    point_dir = output_dir / "point_cloud" / f"iteration_{int(iteration)}"
    mkdir_p(str(point_dir))
    student.save_ply(str(point_dir / "point_cloud.ply"))
    student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(student.get_xyz.shape[0])}, f, indent=2)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Reinitialize a SOF-style Gaussian field from mip RGB renders and depth priors.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--mip_model_path", default=None)
    parser.add_argument("--mip_render_root", required=True)
    parser.add_argument("--mip_iteration", type=int, default=30000)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--copy_render_config_from", default=None)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=244)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--init_only", action="store_true")

    parser.add_argument("--init_max_views", type=int, default=48)
    parser.add_argument("--init_pixel_stride", type=int, default=8)
    parser.add_argument("--init_min_weight", type=float, default=0.02)
    parser.add_argument("--init_voxel_size", type=float, default=0.0)
    parser.add_argument("--init_voxel_size_factor", type=float, default=0.002)
    parser.add_argument("--init_min_points_per_voxel", type=int, default=1)
    parser.add_argument("--init_max_points", type=int, default=200000)
    parser.add_argument("--init_sfm_fallback", action="store_true")
    parser.add_argument("--init_sfm_max_points", type=int, default=120000)
    parser.add_argument("--init_sfm_weight", type=float, default=0.20)
    parser.add_argument("--init_sfm_only_missing", action="store_true")
    parser.add_argument("--init_sfm_min_visible_views", type=int, default=1)

    parser.add_argument("--xyz_lr_init", type=float, default=0.00016)
    parser.add_argument("--xyz_lr_final", type=float, default=0.0000016)
    parser.add_argument("--xyz_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--feature_rest_lr", type=float, default=0.000125)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    parser.add_argument("--lambda_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_teacher_depth", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_normal", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_alpha", type=float, default=0.0)
    parser.add_argument("--lambda_opacity_reg", type=float, default=1e-4)
    parser.add_argument("--min_surface_alpha", type=float, default=0.05)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--depth_relative_min", type=float, default=1e-3)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--min_loss_pixels", type=float, default=64.0)

    parser.add_argument("--depth_prior_root", type=str, default=None)
    parser.add_argument("--depth_prior_subdirs", type=str, default="depth,")
    parser.add_argument("--depth_prior_confidence_subdirs", type=str, default="auto")
    parser.add_argument("--depth_prior_confidence_min", type=float, default=0.05)
    parser.add_argument("--depth_prior_agreement_threshold", type=float, default=0.15)
    parser.add_argument("--depth_prior_agreement_floor", type=float, default=0.0)
    parser.add_argument("--depth_prior_align_mode", choices=["identity"], default="identity")
    parser.add_argument("--depth_prior_align_min_pixels", type=int, default=2048)
    parser.add_argument("--depth_prior_surface_weight_boost", type=float, default=0.25)
    parser.add_argument("--depth_prior_weight_gain", type=float, default=1.0)
    parser.add_argument("--depth_prior_weight_power", type=float, default=1.0)
    parser.add_argument("--depth_prior_weight_min", type=float, default=0.0)
    parser.add_argument("--lambda_depth_prior", type=float, default=0.10)
    parser.add_argument("--lambda_depth_prior_normal", type=float, default=0.03)
    parser.add_argument("--lambda_depth_prior_distortion", type=float, default=100.0)
    parser.add_argument("--lambda_depth_prior_self_normal", type=float, default=0.05)
    parser.add_argument("--depth_prior_warmup_start_iter", type=int, default=0)
    parser.add_argument("--depth_prior_warmup_end_iter", type=int, default=4000)
    parser.add_argument("--depth_prior_start_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_end_scale", type=float, default=2.0)
    parser.add_argument("--depth_prior_update_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_schedule_mode", choices=["linear", "smoothstep"], default="smoothstep")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    safe_state(bool(args.quiet))

    if not torch.cuda.is_available():
        raise RuntimeError("train_mip_depthprior_reinit_sof_v0 currently requires CUDA.")

    scene_root = Path(args.scene_root).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve() if args.mip_model_path else None
    mip_render_root = Path(args.mip_render_root).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)
    write_clean_cfg_args(output_model_path, scene_root, str(args.images_subdir), int(args.sh_degree))

    if not scene_root.is_dir():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")
    if not mip_render_root.is_dir():
        raise FileNotFoundError(f"mip_render_root not found: {mip_render_root}")

    depth_prior_root = Path(args.depth_prior_root).expanduser().resolve() if args.depth_prior_root else None
    if depth_prior_root is not None and not depth_prior_root.is_dir():
        raise FileNotFoundError(f"depth_prior_root not found: {depth_prior_root}")

    depth_prior_subdirs = parse_subdir_list(args.depth_prior_subdirs, default_auto=("depth", ""))
    depth_prior_confidence_subdirs = parse_subdir_list(
        args.depth_prior_confidence_subdirs,
        default_auto=("confidence", "conf", "depth_conf", "valid"),
    )

    teacher_iteration = int(args.mip_iteration)
    all_cameras = load_cameras_for_split(scene_root, output_model_path, str(args.images_subdir), str(args.split))
    selected_indices = select_uniform_indices(len(all_cameras), int(args.max_views))
    selected_cameras = [all_cameras[idx] for idx in selected_indices]
    if not selected_cameras:
        raise RuntimeError("No training cameras selected.")

    all_render_paths = list_render_images(mip_render_root)
    if len(all_render_paths) == len(all_cameras):
        selected_render_paths = [all_render_paths[idx] for idx in selected_indices]
    elif len(all_render_paths) == len(selected_cameras):
        selected_render_paths = all_render_paths
    else:
        raise RuntimeError(
            f"Render count mismatch under {mip_render_root}: got {len(all_render_paths)} images "
            f"for {len(all_cameras)} cameras ({len(selected_cameras)} selected)."
        )

    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    teacher_cache = build_mip_rgb_depthprior_cache(
        selected_cameras,
        selected_render_paths,
        depth_prior_root=depth_prior_root,
        depth_prior_subdirs=depth_prior_subdirs,
        depth_prior_confidence_subdirs=depth_prior_confidence_subdirs,
        depth_prior_confidence_min=float(args.depth_prior_confidence_min),
        depth_prior_agreement_threshold=float(args.depth_prior_agreement_threshold),
        depth_prior_agreement_floor=float(args.depth_prior_agreement_floor),
        depth_prior_align_mode=str(args.depth_prior_align_mode),
        depth_prior_align_min_pixels=int(args.depth_prior_align_min_pixels),
        depth_prior_surface_weight_boost=float(args.depth_prior_surface_weight_boost),
        depth_prior_weight_gain=float(args.depth_prior_weight_gain),
        depth_prior_weight_power=float(args.depth_prior_weight_power),
        depth_prior_weight_min=float(args.depth_prior_weight_min),
        depth_min=float(args.depth_min),
    )

    init_pcd, init_summary = build_pointcloud_from_depth_prior(
        scene_root,
        selected_cameras,
        teacher_cache,
        init_max_views=int(args.init_max_views),
        init_pixel_stride=int(args.init_pixel_stride),
        init_min_weight=float(args.init_min_weight),
        init_voxel_size=float(args.init_voxel_size),
        init_voxel_size_factor=float(args.init_voxel_size_factor),
        init_min_points_per_voxel=int(args.init_min_points_per_voxel),
        init_max_points=int(args.init_max_points),
        init_sfm_fallback=bool(args.init_sfm_fallback),
        init_sfm_max_points=int(args.init_sfm_max_points),
        init_sfm_weight=float(args.init_sfm_weight),
        init_sfm_only_missing=bool(args.init_sfm_only_missing),
        init_sfm_min_visible_views=int(args.init_sfm_min_visible_views),
    )

    spatial_lr_scale = camera_extent_from_views(selected_cameras)
    init_ply_path = output_model_path / "depthprior_pointcloud_init_v0.ply"
    storePly(str(init_ply_path), init_pcd.points, np.clip(np.round(init_pcd.colors * 255.0), 0, 255).astype(np.uint8))
    init_iter_dir = output_model_path / "point_cloud" / "iteration_0"
    mkdir_p(str(init_iter_dir))
    gaussian_init = GaussianModel(int(args.sh_degree))
    gaussian_init.create_from_pcd(init_pcd, spatial_lr_scale=spatial_lr_scale, MCMC_init=False)
    gaussian_init.compute_3D_filter(selected_cameras.copy(), CUDA=False)
    gaussian_init.save_ply(str(init_iter_dir / "point_cloud.ply"))
    gaussian_init.save_tracking_metadata(str(init_iter_dir / "gaussian_tags.pt"))
    del gaussian_init

    save_iterations = set(int(item) for item in args.save_iterations)
    if int(args.save_every) > 0:
        save_iterations.update(range(int(args.save_every), int(args.iterations) + 1, int(args.save_every)))
    save_iterations.add(int(args.iterations))
    save_iterations = {item for item in save_iterations if 0 < item <= int(args.iterations)}

    print(
        "[mip-depthprior-reinit] mip rgb source  : "
        f"{mip_render_root} (render-only supervision)"
    )
    print(f"[mip-depthprior-reinit] selected views  : {len(selected_cameras)}")
    print(
        "[mip-depthprior-reinit] init points     : "
        f"{init_summary['final_points']} from {init_summary['raw_points']} raw "
        f"(voxel={init_summary['voxel_size']:.6f})"
    )
    print(
        "[mip-depthprior-reinit] sfm fallback    : "
        f"enabled={bool(init_summary['sfm_fallback'].get('enabled', False))} "
        f"kept={int(init_summary['sfm_fallback'].get('kept_points', 0))}"
    )
    print(f"[mip-depthprior-reinit] depth prior     : {depth_prior_root if depth_prior_root is not None else '(disabled)'}")

    if bool(args.init_only):
        summary = {
            "version": "mip_depthprior_reinit_sof_v0",
            "mode": "init_only",
            "supervision_mode": "mip_rgb_plus_depth_prior",
            "scene_root": str(scene_root),
            "mip_model_path": str(mip_model_path) if mip_model_path is not None else None,
            "mip_render_root": str(mip_render_root),
            "mip_iteration": int(teacher_iteration),
            "output_model_path": str(output_model_path),
            "inputs": {
                "images_subdir": str(args.images_subdir),
                "split": str(args.split),
                "selected_views": [str(item["image_name"]) for item in teacher_cache],
                "selected_view_count": int(len(selected_cameras)),
            },
            "initialization": {
                **init_summary,
                "init_ply": str(init_ply_path),
                "init_gaussian_ply": str(init_iter_dir / "point_cloud.ply"),
                "init_gaussian_tags": str(init_iter_dir / "gaussian_tags.pt"),
                "spatial_lr_scale": float(spatial_lr_scale),
            },
            "params": vars(args),
            "artifacts": {
                "init_point_cloud": str(init_ply_path),
                "init_gaussian_dir": str(init_iter_dir),
            },
        }
        with open(output_model_path / "mip_depthprior_reinit_sof_v0_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        with open(output_model_path / "mip_depthprior_reinit_sof_v0_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"[mip-depthprior-reinit] summary : {output_model_path / 'mip_depthprior_reinit_sof_v0_summary.json'}")
        print(f"[mip-depthprior-reinit] init    : {init_iter_dir / 'point_cloud.ply'}")
        return

    student = GaussianModel(int(args.sh_degree))
    student.create_from_pcd(init_pcd, spatial_lr_scale=spatial_lr_scale, MCMC_init=False)
    student.compute_3D_filter(selected_cameras.copy(), CUDA=False)
    optimizer, xyz_scheduler = build_optimizer(student, args, spatial_lr_scale)

    progress = tqdm(range(1, int(args.iterations) + 1), desc="mip depthprior reinit")
    ema_total = 0.0
    ema_rgb = 0.0
    ema_depth = 0.0
    log_rows: List[Dict[str, float]] = []
    last_metrics: Dict[str, float] = {}

    for iteration in progress:
        for group in optimizer.param_groups:
            if str(group.get("name", "")) == "xyz":
                group["lr"] = xyz_scheduler(iteration)

        view_idx = randint(0, len(selected_cameras) - 1)
        camera = selected_cameras[view_idx]
        target = teacher_cache[view_idx]

        render_pkg = render_simple(camera, student, background)
        rgb = render_pkg["render"].clamp(0.0, 1.0)
        depth = render_pkg["depth"]
        normal = normalize_normal(render_pkg["normal"])

        rgb_mask = target["rgb_mask"].to(device="cuda", dtype=torch.bool)
        rgb_weight = target["rgb_weight"].to(device="cuda", dtype=torch.float32)
        mip_rgb = target["mip_rgb"].to(device="cuda", dtype=torch.float32)

        loss = torch.zeros((), dtype=torch.float32, device="cuda")

        rgb_loss = masked_weighted_mean(
            torch.abs(rgb - mip_rgb),
            rgb_mask,
            float(args.min_loss_pixels),
            rgb_weight,
        )
        if rgb_loss is None:
            rgb_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        loss = loss + float(args.lambda_rgb) * rgb_loss

        depth_prior_scale = scheduled_loss_scale(
            iteration,
            start_iter=int(args.depth_prior_warmup_start_iter),
            end_iter=int(args.depth_prior_warmup_end_iter),
            start_scale=float(args.depth_prior_start_scale),
            end_scale=float(args.depth_prior_end_scale),
            update_scale=float(args.depth_prior_update_scale),
            mode=str(args.depth_prior_schedule_mode),
        )

        depth_prior_mask = target["depth_prior_mask"].to(device="cuda", dtype=torch.bool)
        depth_prior_weight = target["depth_prior_weight"].to(device="cuda", dtype=torch.float32)
        render_target = {
            "depth_prior_mask": depth_prior_mask,
            "depth_prior_weight": depth_prior_weight,
        }

        depth_prior_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior) > 0.0 and target.get("depth_prior_target") is not None:
            depth_prior_target = target["depth_prior_target"].to(device="cuda", dtype=torch.float32)
            prior_depth_rel = (depth - depth_prior_target) / torch.clamp(
                depth_prior_target,
                min=float(args.depth_relative_min),
            )
            maybe_depth_prior_loss = masked_weighted_mean(
                charbonnier(prior_depth_rel, float(args.charbonnier_eps)),
                depth_prior_mask,
                float(args.min_loss_pixels),
                depth_prior_weight,
            )
            if maybe_depth_prior_loss is not None:
                depth_prior_loss = maybe_depth_prior_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior) * depth_prior_loss

        depth_prior_normal_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior_normal) > 0.0 and target.get("depth_prior_normal") is not None:
            prior_normal_target = target["depth_prior_normal"].to(device="cuda", dtype=torch.float32)
            prior_normal_dot = torch.sum(normal * prior_normal_target, dim=0).clamp(-1.0, 1.0)
            maybe_prior_normal_loss = masked_weighted_mean(
                1.0 - prior_normal_dot,
                depth_prior_mask,
                float(args.min_loss_pixels),
                depth_prior_weight,
            )
            if maybe_prior_normal_loss is not None:
                depth_prior_normal_loss = maybe_prior_normal_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_normal) * depth_prior_normal_loss

        depth_prior_distortion_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior_distortion) > 0.0 and target.get("depth_prior_target") is not None:
            maybe_prior_distortion_loss = compute_depth_prior_distortion_loss(
                render_pkg,
                render_target,
                min_pixels=float(args.min_loss_pixels),
            )
            if maybe_prior_distortion_loss is not None:
                depth_prior_distortion_loss = maybe_prior_distortion_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_distortion) * depth_prior_distortion_loss

        depth_prior_self_normal_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior_self_normal) > 0.0 and target.get("depth_prior_target") is not None:
            maybe_prior_self_normal_loss = compute_depth_prior_self_normal_loss(
                camera,
                render_pkg,
                render_target,
                min_pixels=float(args.min_loss_pixels),
            )
            if maybe_prior_self_normal_loss is not None:
                depth_prior_self_normal_loss = maybe_prior_self_normal_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_self_normal) * depth_prior_self_normal_loss

        opacity_reg = student.get_opacity.mean()
        loss = loss + float(args.lambda_opacity_reg) * opacity_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ema_total = 0.4 * float(loss.item()) + 0.6 * ema_total
        ema_rgb = 0.4 * float(rgb_loss.item()) + 0.6 * ema_rgb
        ema_depth = 0.4 * float(depth_prior_loss.item()) + 0.6 * ema_depth
        last_metrics = {
            "loss": float(loss.item()),
            "rgb": float(rgb_loss.item()),
            "depth_prior": float(depth_prior_loss.item()),
            "depth_prior_normal": float(depth_prior_normal_loss.item()),
            "depth_prior_distortion": float(depth_prior_distortion_loss.item()),
            "depth_prior_self_normal": float(depth_prior_self_normal_loss.item()),
            "opacity_reg": float(opacity_reg.item()),
            "depth_prior_scale": float(depth_prior_scale),
        }
        if iteration % 10 == 0:
            progress.set_postfix(
                {
                    "loss": f"{ema_total:.6f}",
                    "rgb": f"{ema_rgb:.6f}",
                    "dpr": f"{ema_depth:.6f}",
                }
            )
        if iteration % 100 == 0 or iteration == 1 or iteration == int(args.iterations):
            log_rows.append({"iteration": int(iteration), **last_metrics})
        if iteration in save_iterations:
            save_checkpoint(student, output_model_path, iteration)

    output_iteration = int(args.iterations)
    save_checkpoint(student, output_model_path, output_iteration)

    summary = {
        "version": "mip_depthprior_reinit_sof_v0",
        "supervision_mode": "mip_rgb_plus_depth_prior",
        "scene_root": str(scene_root),
        "mip_model_path": str(mip_model_path) if mip_model_path is not None else None,
        "mip_render_root": str(mip_render_root),
        "mip_iteration": int(teacher_iteration),
        "output_model_path": str(output_model_path),
        "inputs": {
            "images_subdir": str(args.images_subdir),
            "split": str(args.split),
            "selected_views": [str(item["image_name"]) for item in teacher_cache],
            "selected_view_count": int(len(selected_cameras)),
        },
        "initialization": {
            **init_summary,
            "init_ply": str(init_ply_path),
            "spatial_lr_scale": float(spatial_lr_scale),
        },
        "params": vars(args),
        "training": {
            "iterations": int(args.iterations),
            "last_metrics": last_metrics,
            "ema": {
                "loss": float(ema_total),
                "rgb": float(ema_rgb),
                "depth_prior": float(ema_depth),
            },
            "log_rows": log_rows,
        },
        "artifacts": {
            "final_point_cloud_dir": str(output_model_path / "point_cloud" / f"iteration_{output_iteration}"),
            "final_point_cloud": str(output_model_path / "point_cloud" / f"iteration_{output_iteration}" / "point_cloud.ply"),
        },
    }
    with open(output_model_path / "mip_depthprior_reinit_sof_v0_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(output_model_path / "mip_depthprior_reinit_sof_v0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[mip-depthprior-reinit] summary : {output_model_path / 'mip_depthprior_reinit_sof_v0_summary.json'}")
    print(f"[mip-depthprior-reinit] output  : {output_model_path / 'point_cloud' / f'iteration_{output_iteration}' / 'point_cloud.ply'}")


if __name__ == "__main__":
    main()
