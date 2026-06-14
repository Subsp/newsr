from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scripts.export_lr_artifact_gaussian_scores_v0 import (
    _accumulate_from_visibility,
    _clone_subset_gaussians,
    _downsample_map,
    _stats,
    _write_masked_model,
)
from train_mip_to_sof_surface_v0 import (
    build_dataset_args,
    load_model_ply,
    load_train_cameras_only,
    resolve_iteration,
    select_uniform,
)
from utils.general_utils import build_rotation
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def stats_from_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def robust_scene_diag(xyz: np.ndarray) -> float:
    if xyz.shape[0] == 0:
        return 1.0
    lo = np.percentile(xyz, 1.0, axis=0)
    hi = np.percentile(xyz, 99.0, axis=0)
    return max(float(np.linalg.norm(hi - lo)), 1e-6)


def camera_center_numpy(camera) -> np.ndarray:
    center = camera.camera_center
    if torch.is_tensor(center):
        center = center.detach().cpu().numpy()
    return np.asarray(center, dtype=np.float32).reshape(3)


def project_points_camera(camera, points_xyz: np.ndarray, *, depth_min: float, margin: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(camera.R, dtype=np.float32)
    T = np.asarray(camera.T, dtype=np.float32)
    xyz_cam = points_xyz @ R + T[None, :]
    z = xyz_cam[:, 2]
    valid_z = z > float(depth_min)
    safe_z = np.maximum(z, 1e-6)
    u = xyz_cam[:, 0] / safe_z * float(camera.focal_x) + float(camera.image_width) * 0.5
    v = xyz_cam[:, 1] / safe_z * float(camera.focal_y) + float(camera.image_height) * 0.5
    projected = np.stack([u, v, z], axis=1).astype(np.float32, copy=False)
    valid = (
        valid_z
        & np.isfinite(projected).all(axis=1)
        & (u >= -float(margin))
        & (u < float(camera.image_width + margin))
        & (v >= -float(margin))
        & (v < float(camera.image_height + margin))
    )
    return projected, valid


def cap_mask(candidate: np.ndarray, score: np.ndarray, *, max_fraction: float, max_count: int) -> Tuple[np.ndarray, int, int]:
    mask = candidate.astype(bool, copy=True)
    before = int(np.count_nonzero(mask))
    cap = before
    if float(max_fraction) > 0.0:
        cap = min(cap, max(1, int(round(mask.shape[0] * float(max_fraction)))))
    if int(max_count) > 0:
        cap = min(cap, int(max_count))
    if cap <= 0:
        mask = np.zeros_like(mask, dtype=bool)
    elif before > cap:
        ids = np.flatnonzero(mask).astype(np.int64, copy=False)
        order = np.argsort(-score[ids], kind="stable")[:cap]
        capped = np.zeros_like(mask, dtype=bool)
        capped[ids[order]] = True
        mask = capped
    return mask, before, int(np.count_nonzero(mask))


@torch.no_grad()
def gaussian_geometry(gaussians, *, use_effective_scale: bool) -> Dict[str, np.ndarray]:
    xyz = gaussians.get_xyz.detach().float()
    scaling = gaussians.get_scaling_with_3D_filter if bool(use_effective_scale) else gaussians.get_scaling
    raw_scaling = gaussians.get_scaling
    rotations = build_rotation(gaussians.get_rotation).detach()
    largest_axis_idx = torch.argmax(scaling, dim=1)
    shortest_axis_idx = torch.argmin(scaling, dim=1)
    row_ids = torch.arange(scaling.shape[0], device=scaling.device)
    largest_axis = rotations[row_ids, :, largest_axis_idx]
    largest_axis = largest_axis / torch.clamp(torch.linalg.norm(largest_axis, dim=1, keepdim=True), min=1e-8)
    shortest_axis = rotations[row_ids, :, shortest_axis_idx]
    shortest_axis = shortest_axis / torch.clamp(torch.linalg.norm(shortest_axis, dim=1, keepdim=True), min=1e-8)

    sorted_scales = torch.sort(scaling, dim=1).values
    raw_sorted_scales = torch.sort(torch.clamp(raw_scaling, min=1e-8), dim=1).values
    scale_min = torch.clamp(sorted_scales[:, 0], min=1e-8)
    scale_mid = torch.clamp(sorted_scales[:, 1], min=1e-8)
    scale_max = torch.clamp(sorted_scales[:, 2], min=1e-8)
    raw_scale_max = torch.clamp(raw_sorted_scales[:, 2], min=1e-8)
    volume_radius = torch.clamp(torch.prod(scaling, dim=1), min=1e-24).pow(1.0 / 3.0)
    raw_volume_radius = torch.clamp(torch.prod(torch.clamp(raw_scaling, min=1e-8), dim=1), min=1e-24).pow(1.0 / 3.0)
    if torch.is_tensor(gaussians.filter_3D):
        filter_3d = gaussians.filter_3D.detach().reshape(-1).float()
    else:
        filter_3d = torch.zeros((xyz.shape[0],), dtype=torch.float32, device=xyz.device)

    return {
        "xyz": xyz.detach().cpu().numpy().astype(np.float32, copy=False),
        "axis_matrix": rotations.detach().cpu().numpy().astype(np.float32, copy=False),
        "largest_axis": largest_axis.detach().cpu().numpy().astype(np.float32, copy=False),
        "shortest_axis": shortest_axis.detach().cpu().numpy().astype(np.float32, copy=False),
        "shortest_axis_idx": shortest_axis_idx.detach().cpu().numpy().astype(np.int64, copy=False),
        "scale": scaling.detach().cpu().numpy().astype(np.float32, copy=False),
        "raw_scale": raw_scaling.detach().cpu().numpy().astype(np.float32, copy=False),
        "scale_max": scale_max.detach().cpu().numpy().astype(np.float32, copy=False),
        "scale_mid": scale_mid.detach().cpu().numpy().astype(np.float32, copy=False),
        "scale_min": scale_min.detach().cpu().numpy().astype(np.float32, copy=False),
        "raw_scale_max": raw_scale_max.detach().cpu().numpy().astype(np.float32, copy=False),
        "axis_anisotropy": (scale_max / scale_mid).detach().cpu().numpy().astype(np.float32, copy=False),
        "full_anisotropy": (scale_max / scale_min).detach().cpu().numpy().astype(np.float32, copy=False),
        "volume_radius": volume_radius.detach().cpu().numpy().astype(np.float32, copy=False),
        "raw_volume_radius": raw_volume_radius.detach().cpu().numpy().astype(np.float32, copy=False),
        "filter_inflation": (scale_max / raw_scale_max).detach().cpu().numpy().astype(np.float32, copy=False),
        "filter_scale_ratio": (filter_3d / raw_scale_max).detach().cpu().numpy().astype(np.float32, copy=False),
        "opacity": gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False),
        "filter_3D": filter_3d.detach().cpu().numpy().astype(np.float32, copy=False),
    }


@torch.no_grad()
def collect_view_aligned_stats(
    gaussians,
    cameras: Sequence[object],
    geom: Dict[str, np.ndarray],
    *,
    depth_min: float,
    screen_margin_px: int,
    chunk_size: int,
    background: torch.Tensor,
) -> Dict[str, np.ndarray]:
    xyz = geom["xyz"]
    axes = geom["axis_matrix"]
    scale = geom["scale"]
    largest_axis = geom["largest_axis"]
    n = xyz.shape[0]
    visible_count = np.zeros((n,), dtype=np.int32)
    radius_sum = np.zeros((n,), dtype=np.float64)
    radius_max = np.zeros((n,), dtype=np.float32)
    alignment_sum = np.zeros((n,), dtype=np.float64)
    alignment_max = np.zeros((n,), dtype=np.float32)
    ray_thickness_sum = np.zeros((n,), dtype=np.float64)
    ray_thickness_max = np.zeros((n,), dtype=np.float32)
    train_cross_scale_sum = np.zeros((n,), dtype=np.float64)
    train_cross_scale_min = np.full((n,), np.inf, dtype=np.float32)
    side_explosion_sum = np.zeros((n,), dtype=np.float64)
    side_explosion_max = np.zeros((n,), dtype=np.float32)
    side_radius_sum = np.zeros((n,), dtype=np.float64)
    side_radius_max = np.zeros((n,), dtype=np.float32)

    step = max(int(chunk_size), 1)
    for camera in tqdm(cameras, desc="view-aligned volume score"):
        render_pkg = render_simple(camera, gaussians, background)
        radii = render_pkg["radii"].detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        render_visible = render_pkg["visibility_filter"].detach().cpu().numpy().astype(bool).reshape(-1)
        projected, projection_valid = project_points_camera(
            camera,
            xyz,
            depth_min=float(depth_min),
            margin=int(screen_margin_px),
        )
        active = render_visible & projection_valid & (radii > 0.0)
        active_ids = np.flatnonzero(active).astype(np.int64, copy=False)
        if active_ids.size == 0:
            continue
        cam_center = camera_center_numpy(camera)
        focal_mean = 0.5 * (float(camera.focal_x) + float(camera.focal_y))

        for begin in range(0, active_ids.shape[0], step):
            ids = active_ids[begin : begin + step]
            ray = xyz[ids] - cam_center[None, :]
            ray_norm = np.clip(np.linalg.norm(ray, axis=1, keepdims=True), 1e-8, None)
            ray_dir = ray / ray_norm

            alignment = np.abs(np.sum(largest_axis[ids] * ray_dir, axis=1)).astype(np.float32, copy=False)
            local_ray = np.einsum("nji,nj->ni", axes[ids], ray_dir, optimize=True)
            local_ray = np.clip(local_ray, -1.0, 1.0)
            ray_thickness = np.sqrt(np.sum(np.square(local_ray * scale[ids]), axis=1) + 1e-16).astype(
                np.float32,
                copy=False,
            )
            perp_weight = np.sqrt(np.clip(1.0 - np.square(local_ray), 0.0, 1.0))
            train_cross_scale = np.max(scale[ids] * perp_weight, axis=1).astype(np.float32, copy=False)
            train_cross_scale = np.maximum(train_cross_scale, 1e-8)
            side_explosion = (geom["scale_max"][ids] / train_cross_scale).astype(np.float32, copy=False)
            side_radius = (focal_mean * geom["scale_max"][ids] / np.maximum(projected[ids, 2], 1e-6)).astype(
                np.float32,
                copy=False,
            )

            visible_count[ids] += 1
            radius_sum[ids] += radii[ids].astype(np.float64)
            radius_max[ids] = np.maximum(radius_max[ids], radii[ids])
            alignment_sum[ids] += alignment.astype(np.float64)
            alignment_max[ids] = np.maximum(alignment_max[ids], alignment)
            ray_thickness_sum[ids] += ray_thickness.astype(np.float64)
            ray_thickness_max[ids] = np.maximum(ray_thickness_max[ids], ray_thickness)
            train_cross_scale_sum[ids] += train_cross_scale.astype(np.float64)
            train_cross_scale_min[ids] = np.minimum(train_cross_scale_min[ids], train_cross_scale)
            side_explosion_sum[ids] += side_explosion.astype(np.float64)
            side_explosion_max[ids] = np.maximum(side_explosion_max[ids], side_explosion)
            side_radius_sum[ids] += side_radius.astype(np.float64)
            side_radius_max[ids] = np.maximum(side_radius_max[ids], side_radius)

    denom = np.maximum(visible_count.astype(np.float32), 1.0)
    train_cross_scale_min[~np.isfinite(train_cross_scale_min)] = 0.0
    ray_thickness_mean = (ray_thickness_sum / denom).astype(np.float32)
    train_cross_scale_mean = (train_cross_scale_sum / denom).astype(np.float32)
    ray_thickness_ratio = (
        ray_thickness_mean / np.maximum(train_cross_scale_mean.astype(np.float32, copy=False), 1e-8)
    ).astype(np.float32)
    return {
        "visible_count": visible_count,
        "visible_fraction": (visible_count.astype(np.float32) / max(len(cameras), 1)).astype(np.float32),
        "radius_mean": (radius_sum / denom).astype(np.float32),
        "radius_max": radius_max,
        "axis_alignment_mean": (alignment_sum / denom).astype(np.float32),
        "axis_alignment_max": alignment_max,
        "ray_thickness_mean": ray_thickness_mean,
        "ray_thickness_max": ray_thickness_max,
        "train_cross_scale_mean": train_cross_scale_mean,
        "train_cross_scale_min": train_cross_scale_min,
        "ray_thickness_ratio": ray_thickness_ratio,
        "side_explosion_mean": (side_explosion_sum / denom).astype(np.float32),
        "side_explosion_max": side_explosion_max,
        "side_radius_mean": (side_radius_sum / denom).astype(np.float32),
        "side_radius_max": side_radius_max,
    }


def build_short_axis_stress_model(
    gaussians,
    *,
    axis_source: str,
    scale_factor: float,
    min_axis_to_max_ratio: float,
) -> object:
    all_mask = torch.ones((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=gaussians.get_xyz.device)
    stressed = _clone_subset_gaussians(gaussians, all_mask)
    with torch.no_grad():
        raw_scale = torch.clamp(stressed.get_scaling.detach().clone(), min=1e-8)
        if isinstance(stressed.filter_3D, torch.Tensor) and stressed.filter_3D.ndim > 0:
            filter_3d = stressed.filter_3D.detach().reshape(-1, 1).float()
        else:
            filter_3d = torch.zeros((raw_scale.shape[0], 1), dtype=torch.float32, device=raw_scale.device)
        axis_scale = stressed.get_scaling_with_3D_filter if axis_source == "effective" else raw_scale
        shortest_axis_idx = torch.argmin(axis_scale, dim=1)
        row_ids = torch.arange(raw_scale.shape[0], device=raw_scale.device)
        max_axis_scale = torch.clamp(axis_scale.max(dim=1).values, min=1e-8)
        current_axis_scale = torch.clamp(axis_scale[row_ids, shortest_axis_idx], min=1e-8)
        target = torch.maximum(
            current_axis_scale * float(scale_factor),
            max_axis_scale * float(min_axis_to_max_ratio),
        )
        if axis_source == "effective":
            filter_axis = filter_3d[row_ids, 0]
            target = torch.sqrt(torch.clamp(target.square() - filter_axis.square(), min=0.0))
        raw_scale[row_ids, shortest_axis_idx] = torch.maximum(
            raw_scale[row_ids, shortest_axis_idx],
            torch.clamp(target, min=1e-8),
        )
        stressed._scaling = nn.Parameter(torch.log(torch.clamp(raw_scale, min=1e-8)).requires_grad_(False))
    return stressed


@torch.no_grad()
def collect_short_axis_stress_stats(
    gaussians,
    stressed_gaussians,
    cameras: Sequence[object],
    *,
    visibility_downsample: int,
    visibility_topk: int,
    visibility_max_visible: int,
    visibility_max_patch_radius: int,
    major_impact_threshold: float,
    major_radius_gain_threshold: float,
    large_radius_threshold_px: float,
    depth_min: float,
    background: torch.Tensor,
) -> Dict[str, np.ndarray]:
    n = int(gaussians.get_xyz.shape[0])
    impact_sum = np.zeros((n,), dtype=np.float64)
    impact_denom = np.zeros((n,), dtype=np.float64)
    impact_view_sum = np.zeros((n,), dtype=np.float64)
    raw_impact_sum = np.zeros((n,), dtype=np.float64)
    raw_impact_denom = np.zeros((n,), dtype=np.float64)
    impact_max = np.zeros((n,), dtype=np.float32)
    stress_visible_count = np.zeros((n,), dtype=np.int32)
    stress_major_impact_count = np.zeros((n,), dtype=np.int32)
    stress_radius_gain_sum = np.zeros((n,), dtype=np.float64)
    stress_radius_gain_max = np.ones((n,), dtype=np.float32)
    stress_radius_max = np.zeros((n,), dtype=np.float32)
    stress_large_radius_count = np.zeros((n,), dtype=np.int32)

    vis_cfg = VisibilityRecordConfig(
        downsample=int(visibility_downsample),
        topk=int(visibility_topk),
        max_visible_per_view=int(visibility_max_visible),
        min_opacity=0.0,
        min_depth=float(depth_min),
        max_patch_radius=int(visibility_max_patch_radius),
    )

    for camera in tqdm(cameras, desc="short-axis stress score"):
        base_pkg = render_simple(camera, gaussians, background)
        stress_pkg = render_simple(camera, stressed_gaussians, background)
        base_rgb = base_pkg["render"].detach().clamp(0.0, 1.0)
        stress_rgb = stress_pkg["render"].detach().clamp(0.0, 1.0)
        raw_diff = torch.mean(torch.abs(stress_rgb - base_rgb), dim=0)
        norm = torch.quantile(raw_diff.reshape(-1).float(), 0.98).clamp_min(1e-6)
        diff_norm = torch.clamp(raw_diff / norm.to(device=raw_diff.device, dtype=raw_diff.dtype), 0.0, 1.0)

        image_hw = (int(camera.image_height), int(camera.image_width))
        records = build_coarse_visibility_records(
            stressed_gaussians,
            [camera],
            [stress_pkg],
            image_hw=image_hw,
            cfg=vis_cfg,
        )
        coarse_h, coarse_w = [int(v) for v in records["coarse_hw"].tolist()]
        ids = records["gaussian_ids"][0, 0].numpy()
        weights = records["weights"][0, 0].numpy()
        impact_coarse = _downsample_map(diff_norm, (coarse_h, coarse_w))
        raw_impact_coarse = _downsample_map(raw_diff, (coarse_h, coarse_w))
        view_impact_sum = np.zeros((n,), dtype=np.float64)
        view_impact_denom = np.zeros((n,), dtype=np.float64)
        _accumulate_from_visibility(ids, weights, impact_coarse, impact_sum, impact_denom)
        _accumulate_from_visibility(ids, weights, raw_impact_coarse, raw_impact_sum, raw_impact_denom)
        _accumulate_from_visibility(ids, weights, impact_coarse, view_impact_sum, view_impact_denom)
        view_impact = (view_impact_sum / np.maximum(view_impact_denom, 1e-8)).astype(np.float32)
        view_impact_valid = view_impact_denom > 0.0
        if np.any(view_impact_valid):
            impact_view_sum[view_impact_valid] += view_impact[view_impact_valid].astype(np.float64)
            impact_max[view_impact_valid] = np.maximum(impact_max[view_impact_valid], view_impact[view_impact_valid])
            major_mask = view_impact_valid & (view_impact >= float(major_impact_threshold))
            stress_major_impact_count[major_mask] += 1

        base_radii = base_pkg["radii"].detach().float().cpu().numpy().reshape(-1)
        stress_radii = stress_pkg["radii"].detach().float().cpu().numpy().reshape(-1)
        stress_visible = stress_pkg["visibility_filter"].detach().cpu().numpy().astype(bool).reshape(-1)
        stress_ids = np.flatnonzero(stress_visible & np.isfinite(stress_radii) & (stress_radii > 0.0))
        if stress_ids.size > 0:
            gain = stress_radii[stress_ids] / np.maximum(base_radii[stress_ids], 1.0)
            stress_visible_count[stress_ids] += 1
            stress_radius_gain_sum[stress_ids] += gain.astype(np.float64)
            stress_radius_gain_max[stress_ids] = np.maximum(stress_radius_gain_max[stress_ids], gain.astype(np.float32))
            stress_radius_max[stress_ids] = np.maximum(stress_radius_max[stress_ids], stress_radii[stress_ids])
            large_radius = (stress_radii[stress_ids] >= float(large_radius_threshold_px)) & (
                gain >= float(major_radius_gain_threshold)
            )
            stress_large_radius_count[stress_ids[large_radius]] += 1

    impact_score = (impact_sum / np.maximum(impact_denom, 1e-8)).astype(np.float32)
    raw_impact_score = (raw_impact_sum / np.maximum(raw_impact_denom, 1e-8)).astype(np.float32)
    impact_view_mean = (impact_view_sum / np.maximum(stress_visible_count.astype(np.float64), 1.0)).astype(np.float32)
    stress_radius_gain_mean = (
        stress_radius_gain_sum / np.maximum(stress_visible_count.astype(np.float64), 1.0)
    ).astype(np.float32)
    stress_radius_gain_mean[stress_visible_count == 0] = 1.0
    return {
        "stress_impact_score": impact_score,
        "stress_raw_impact_score": raw_impact_score,
        "stress_impact_max": impact_max,
        "stress_impact_view_mean": impact_view_mean,
        "stress_visible_count": stress_visible_count,
        "stress_visible_fraction": (stress_visible_count.astype(np.float32) / max(len(cameras), 1)).astype(np.float32),
        "stress_major_impact_count": stress_major_impact_count,
        "stress_major_impact_visible_fraction": (
            stress_major_impact_count.astype(np.float32) / np.maximum(stress_visible_count.astype(np.float32), 1.0)
        ).astype(np.float32),
        "stress_radius_gain_mean": stress_radius_gain_mean,
        "stress_radius_gain_max": stress_radius_gain_max,
        "stress_radius_max": stress_radius_max,
        "stress_large_radius_count": stress_large_radius_count,
        "stress_large_radius_visible_fraction": (
            stress_large_radius_count.astype(np.float32) / np.maximum(stress_visible_count.astype(np.float32), 1.0)
        ).astype(np.float32),
    }


def compute_surface_distance(
    mesh_path: Path,
    xyz: np.ndarray,
    *,
    mode: str,
    sample_count: int,
    chunk_size: int,
) -> Tuple[np.ndarray, str]:
    from select_mesh_outside_gaussians_v0 import load_triangle_mesh, query_mesh_surface

    mesh_obj = load_triangle_mesh(str(mesh_path))
    payload = query_mesh_surface(
        mesh_obj=mesh_obj,
        points_xyz=xyz,
        mode=str(mode),
        sample_count=int(sample_count),
        chunk_size=int(chunk_size),
    )
    return payload["surface_distance"].astype(np.float32, copy=False), str(payload["surface_query_mode_used"])


def build_view_aligned_prune_mask(
    geom: Dict[str, np.ndarray],
    view: Dict[str, np.ndarray],
    stress: Dict[str, np.ndarray],
    *,
    scene_diag: float,
    surface_distance: np.ndarray | None,
    delete_quantile: float,
    min_visible_views: int,
    max_visible_fraction: float,
    max_opacity: float,
    min_axis_alignment: float,
    min_axis_anisotropy: float,
    min_ray_thickness_ratio: float,
    min_side_explosion: float,
    min_side_radius_px: float,
    min_radius_px: float,
    min_effective_scale_ratio: float,
    min_volume_radius_ratio: float,
    min_filter_inflation: float,
    min_filter_scale_ratio: float,
    min_stress_impact: float,
    min_stress_radius_gain: float,
    min_stress_radius_px: float,
    min_stress_major_impact_views: int,
    min_stress_major_impact_visible_fraction: float,
    candidate_mode: str,
    surface_distance_threshold: float,
    require_surface_outside: bool,
    max_prune_fraction: float,
    max_prune_count: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | int | bool | str]]:
    visible = view["visible_count"] >= int(min_visible_views)
    opacity = np.clip(geom["opacity"], 0.0, 1.0)
    scale_max = geom["scale_max"]
    volume_radius = geom["volume_radius"]
    min_effective_scale = max(float(scene_diag) * float(min_effective_scale_ratio), 1e-8)
    min_volume_radius = max(float(scene_diag) * float(min_volume_radius_ratio), 1e-8)
    surface_risk = np.zeros_like(scale_max, dtype=np.float32)
    surface_outside = np.ones_like(visible, dtype=bool)
    if surface_distance is not None:
        surface_outside = surface_distance >= float(surface_distance_threshold)
        surface_risk = np.clip(surface_distance / max(float(surface_distance_threshold), 1e-8) - 1.0, 0.0, 3.0) / 3.0

    alignment_risk = np.maximum(
        np.clip((view["axis_alignment_mean"] - float(min_axis_alignment)) / max(1.0 - float(min_axis_alignment), 1e-6), 0.0, 1.0),
        np.clip(view["ray_thickness_ratio"] / max(float(min_ray_thickness_ratio), 1e-6) - 1.0, 0.0, 2.0) / 2.0,
    ).astype(np.float32)
    anisotropy_risk = np.clip(geom["axis_anisotropy"] / max(float(min_axis_anisotropy), 1e-6) - 1.0, 0.0, 3.0) / 3.0
    scale_risk = np.clip(scale_max / min_effective_scale - 1.0, 0.0, 3.0) / 3.0
    volume_risk = np.clip(volume_radius / min_volume_radius - 1.0, 0.0, 3.0) / 3.0
    radius_risk = np.clip(view["radius_max"] / max(float(min_radius_px), 1.0) - 1.0, 0.0, 3.0) / 3.0
    filter_inflation_risk = np.clip(
        geom["filter_inflation"] / max(float(min_filter_inflation), 1e-6) - 1.0,
        0.0,
        3.0,
    ) / 3.0
    filter_scale_risk = np.clip(
        geom["filter_scale_ratio"] / max(float(min_filter_scale_ratio), 1e-6) - 1.0,
        0.0,
        3.0,
    ) / 3.0
    filter_risk = np.maximum(filter_inflation_risk, filter_scale_risk).astype(np.float32)
    stress_impact_risk = np.clip(
        stress["stress_impact_score"] / max(float(min_stress_impact), 1e-6) - 1.0,
        0.0,
        3.0,
    ) / 3.0
    stress_radius_gain_risk = np.clip(
        stress["stress_radius_gain_max"] / max(float(min_stress_radius_gain), 1e-6) - 1.0,
        0.0,
        3.0,
    ) / 3.0
    stress_radius_risk = np.clip(
        stress["stress_radius_max"] / max(float(min_stress_radius_px), 1.0) - 1.0,
        0.0,
        3.0,
    ) / 3.0
    stress_major_view_risk = np.zeros_like(scale_max, dtype=np.float32)
    if int(min_stress_major_impact_views) > 0:
        stress_major_view_risk = np.maximum(
            stress_major_view_risk,
            np.clip(
                stress["stress_major_impact_count"] / max(float(min_stress_major_impact_views), 1.0) - 1.0,
                0.0,
                3.0,
            )
            / 3.0,
        )
    stress_major_fraction_risk = np.zeros_like(scale_max, dtype=np.float32)
    if float(min_stress_major_impact_visible_fraction) > 0.0:
        stress_major_fraction_risk = np.clip(
            stress["stress_major_impact_visible_fraction"] / max(float(min_stress_major_impact_visible_fraction), 1e-6)
            - 1.0,
            0.0,
            3.0,
        ) / 3.0
    stress_persistent_risk = np.maximum(stress_major_view_risk, stress_major_fraction_risk).astype(np.float32)
    stress_risk = np.maximum.reduce(
        [stress_impact_risk, stress_radius_gain_risk, stress_radius_risk, stress_persistent_risk]
    ).astype(np.float32)
    side_risk = np.maximum(
        np.clip(view["side_explosion_max"] / max(float(min_side_explosion), 1e-6) - 1.0, 0.0, 3.0) / 3.0,
        np.clip(view["side_radius_max"] / max(float(min_side_radius_px), 1.0) - 1.0, 0.0, 3.0) / 3.0,
    ).astype(np.float32)
    opacity_risk = np.clip((float(max_opacity) - opacity) / max(float(max_opacity), 1e-6), 0.0, 1.0)

    size_risk = np.maximum.reduce([scale_risk, volume_risk, radius_risk]).astype(np.float32)
    footprint_risk = np.maximum(radius_risk, side_risk).astype(np.float32)
    shape_risk = np.maximum(anisotropy_risk, filter_risk).astype(np.float32)
    geometry_risk = np.maximum.reduce([size_risk, footprint_risk, filter_risk, stress_risk]).astype(np.float32)
    footprint_or_stress_risk = np.maximum.reduce([footprint_risk, filter_risk, stress_persistent_risk]).astype(np.float32)
    delete_score = (
        (0.20 + 0.80 * geometry_risk)
        * (0.25 + 0.75 * stress_risk)
        * (0.35 + 0.65 * footprint_or_stress_risk)
        * (0.45 + 0.55 * np.maximum(shape_risk, stress_risk))
        * (0.65 + 0.35 * opacity_risk)
        * (0.70 + 0.30 * surface_risk)
    ).astype(np.float32)
    delete_score[~np.isfinite(delete_score)] = 0.0

    view_aligned = (view["axis_alignment_mean"] >= float(min_axis_alignment)) | (
        view["ray_thickness_ratio"] >= float(min_ray_thickness_ratio)
    )
    large = (
        (scale_max >= min_effective_scale)
        | (volume_radius >= min_volume_radius)
        | (view["radius_max"] >= float(min_radius_px))
        | (stress["stress_radius_max"] >= float(min_stress_radius_px))
    )
    side_bad = (view["side_explosion_max"] >= float(min_side_explosion)) | (
        view["side_radius_max"] >= float(min_side_radius_px)
    )
    filter_bad = (geom["filter_inflation"] >= float(min_filter_inflation)) | (
        geom["filter_scale_ratio"] >= float(min_filter_scale_ratio)
    )
    footprint_bad = side_bad | (view["radius_max"] >= float(min_radius_px))
    stress_persistent = (
        (stress["stress_major_impact_count"] >= int(min_stress_major_impact_views))
        | (stress["stress_major_impact_visible_fraction"] >= float(min_stress_major_impact_visible_fraction))
    )
    stress_bad = (
        (stress["stress_impact_score"] >= float(min_stress_impact)) & stress_persistent
    ) | (
        (stress["stress_radius_gain_max"] >= float(min_stress_radius_gain))
        & (stress["stress_radius_max"] >= float(min_stress_radius_px))
        & ((stress["stress_large_radius_count"] >= int(max(min_stress_major_impact_views, 1))) | stress_persistent)
    )
    view_aligned_valid = (
        visible
        & view_aligned
        & large
        & side_bad
    )
    volume_valid = (
        visible
        & large
        & (footprint_bad | filter_bad)
        & (side_bad | filter_bad | (geom["axis_anisotropy"] >= float(min_axis_anisotropy)))
    )
    stress_valid = (
        visible
        & large
        & stress_bad
        & (footprint_bad | filter_bad | stress_persistent | (stress["stress_radius_max"] >= float(min_stress_radius_px)))
    )
    if candidate_mode == "view_aligned":
        valid = view_aligned_valid
    elif candidate_mode == "volume":
        valid = volume_valid
    elif candidate_mode == "short_axis_stress":
        valid = stress_valid
    elif candidate_mode == "volume_stress":
        valid = volume_valid | stress_valid
    else:
        valid = view_aligned_valid | volume_valid | stress_valid
    valid = (
        valid
        & (view["visible_fraction"] <= float(max_visible_fraction))
        & (opacity <= float(max_opacity))
        & np.isfinite(delete_score)
    )
    if bool(require_surface_outside):
        valid &= surface_outside

    threshold = float(np.quantile(delete_score[valid], float(delete_quantile))) if np.any(valid) else 1.0
    candidate = valid & (delete_score >= threshold)
    prune_mask, before_cap, after_cap = cap_mask(
        candidate,
        delete_score,
        max_fraction=float(max_prune_fraction),
        max_count=int(max_prune_count),
    )
    return prune_mask.astype(bool), delete_score, {
        "delete_score_threshold": float(threshold),
        "candidate_count_before_cap": int(before_cap),
        "candidate_count_after_cap": int(after_cap),
        "valid_count": int(np.count_nonzero(valid)),
        "surface_gate_enabled": bool(require_surface_outside),
        "surface_distance_available": bool(surface_distance is not None),
        "min_effective_scale": float(min_effective_scale),
        "min_volume_radius": float(min_volume_radius),
        "candidate_mode": str(candidate_mode),
        "view_aligned_valid_count": int(np.count_nonzero(view_aligned_valid)),
        "volume_valid_count": int(np.count_nonzero(volume_valid)),
        "stress_valid_count": int(np.count_nonzero(stress_valid)),
        "stress_persistent_count": int(np.count_nonzero(stress_persistent)),
        "filter_bad_count": int(np.count_nonzero(filter_bad)),
        "footprint_bad_count": int(np.count_nonzero(footprint_bad)),
        "stress_bad_count": int(np.count_nonzero(stress_bad)),
    }


def main() -> None:
    parser = ArgumentParser(description="Hard-prune view-aligned high-volume mip Gaussians before SOF surface pull.")
    parser.add_argument("--scene_root", type=str, required=True)
    parser.add_argument("--mip_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_iteration", type=int, default=30000)
    parser.add_argument("--images_subdir", type=str, default="images_8")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--depth_min", type=float, default=0.05)
    parser.add_argument("--screen_margin_px", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=131072)
    parser.add_argument("--use_effective_scale", type=int, default=1)
    parser.add_argument("--recompute_filter3d", type=int, default=1)
    parser.add_argument("--surface_mesh_path", type=str, default="")
    parser.add_argument("--surface_query_mode", choices=["auto", "exact", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=500000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--surface_distance_threshold", type=float, default=0.035)
    parser.add_argument("--require_surface_outside", action="store_true")
    parser.add_argument(
        "--candidate_mode",
        choices=["view_aligned", "volume", "short_axis_stress", "volume_stress", "hybrid"],
        default="volume_stress",
    )
    parser.add_argument("--stress_axis_source", choices=["raw", "effective"], default="effective")
    parser.add_argument("--stress_short_axis_scale_factor", type=float, default=8.0)
    parser.add_argument("--stress_min_axis_to_max_ratio", type=float, default=1.0)
    parser.add_argument("--stress_visibility_downsample", type=int, default=8)
    parser.add_argument("--stress_visibility_topk", type=int, default=4)
    parser.add_argument("--stress_visibility_max_visible", type=int, default=60000)
    parser.add_argument("--stress_visibility_max_patch_radius", type=int, default=4)
    parser.add_argument("--stress_major_impact_threshold", type=float, default=0.12)
    parser.add_argument("--delete_quantile", type=float, default=0.975)
    parser.add_argument("--min_visible_views", type=int, default=1)
    parser.add_argument("--max_visible_fraction", type=float, default=1.0)
    parser.add_argument("--max_opacity", type=float, default=1.0)
    parser.add_argument("--min_axis_alignment", type=float, default=0.88)
    parser.add_argument("--min_axis_anisotropy", type=float, default=1.70)
    parser.add_argument("--min_ray_thickness_ratio", type=float, default=1.80)
    parser.add_argument("--min_side_explosion", type=float, default=1.70)
    parser.add_argument("--min_side_radius_px", type=float, default=24.0)
    parser.add_argument("--min_radius_px", type=float, default=18.0)
    parser.add_argument("--min_effective_scale_ratio", type=float, default=0.0025)
    parser.add_argument("--min_volume_radius_ratio", type=float, default=0.0015)
    parser.add_argument("--min_filter_inflation", type=float, default=1.25)
    parser.add_argument("--min_filter_scale_ratio", type=float, default=0.60)
    parser.add_argument("--min_stress_impact", type=float, default=0.025)
    parser.add_argument("--min_stress_radius_gain", type=float, default=1.20)
    parser.add_argument("--min_stress_radius_px", type=float, default=22.0)
    parser.add_argument("--min_stress_major_impact_views", type=int, default=2)
    parser.add_argument("--min_stress_major_impact_visible_fraction", type=float, default=0.35)
    parser.add_argument("--max_prune_fraction", type=float, default=0.04)
    parser.add_argument("--max_prune_count", type=int, default=0)
    parser.add_argument("--export_deleted_model", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    iteration = resolve_iteration(mip_model_path, int(args.iteration))
    output_iteration = int(args.output_iteration)
    if output_iteration < 0:
        output_iteration = int(iteration)

    dataset_args = build_dataset_args(str(scene_root), str(mip_model_path), str(args.images_subdir))
    cameras_all = load_train_cameras_only(scene_root, mip_model_path, str(args.images_subdir))
    cameras = select_uniform(cameras_all, int(args.max_views))
    if not cameras:
        raise RuntimeError(f"No train cameras found for scene={scene_root} images={args.images_subdir}")

    mip = load_model_ply(mip_model_path, iteration, int(dataset_args.sh_degree))
    if bool(int(args.recompute_filter3d)):
        mip.compute_3D_filter(cameras, CUDA=False)

    geom = gaussian_geometry(mip, use_effective_scale=bool(int(args.use_effective_scale)))
    scene_diag = robust_scene_diag(geom["xyz"])
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    print(f"[view-volume-cleanup-v0] scene       : {scene_root}")
    print(f"[view-volume-cleanup-v0] mip model   : {mip_model_path} iter={iteration} n={geom['xyz'].shape[0]}")
    print(f"[view-volume-cleanup-v0] views       : {len(cameras)} from {args.images_subdir}")
    print(f"[view-volume-cleanup-v0] scene diag  : {scene_diag:.6f}")
    print(f"[view-volume-cleanup-v0] output      : {output_model_path}")

    view = collect_view_aligned_stats(
        mip,
        cameras,
        geom,
        depth_min=float(args.depth_min),
        screen_margin_px=int(args.screen_margin_px),
        chunk_size=int(args.chunk_size),
        background=background,
    )
    stressed_mip = build_short_axis_stress_model(
        mip,
        axis_source=str(args.stress_axis_source),
        scale_factor=float(args.stress_short_axis_scale_factor),
        min_axis_to_max_ratio=float(args.stress_min_axis_to_max_ratio),
    )
    stress = collect_short_axis_stress_stats(
        mip,
        stressed_mip,
        cameras,
        visibility_downsample=int(args.stress_visibility_downsample),
        visibility_topk=int(args.stress_visibility_topk),
        visibility_max_visible=int(args.stress_visibility_max_visible),
        visibility_max_patch_radius=int(args.stress_visibility_max_patch_radius),
        major_impact_threshold=float(args.stress_major_impact_threshold),
        major_radius_gain_threshold=float(args.min_stress_radius_gain),
        large_radius_threshold_px=float(args.min_stress_radius_px),
        depth_min=float(args.depth_min),
        background=background,
    )

    surface_distance = None
    surface_query_mode_used = None
    surface_mesh_path = Path(args.surface_mesh_path).expanduser().resolve() if str(args.surface_mesh_path).strip() else None
    if surface_mesh_path is not None:
        if not surface_mesh_path.is_file():
            raise FileNotFoundError(f"surface mesh not found: {surface_mesh_path}")
        print(f"[view-volume-cleanup-v0] surface mesh: {surface_mesh_path}")
        surface_distance, surface_query_mode_used = compute_surface_distance(
            surface_mesh_path,
            geom["xyz"],
            mode=str(args.surface_query_mode),
            sample_count=int(args.mesh_surface_sample_count),
            chunk_size=int(args.surface_query_chunk_size),
        )

    prune_mask, delete_score, delete_info = build_view_aligned_prune_mask(
        geom,
        view,
        stress,
        scene_diag=float(scene_diag),
        surface_distance=surface_distance,
        delete_quantile=float(args.delete_quantile),
        min_visible_views=int(args.min_visible_views),
        max_visible_fraction=float(args.max_visible_fraction),
        max_opacity=float(args.max_opacity),
        min_axis_alignment=float(args.min_axis_alignment),
        min_axis_anisotropy=float(args.min_axis_anisotropy),
        min_ray_thickness_ratio=float(args.min_ray_thickness_ratio),
        min_side_explosion=float(args.min_side_explosion),
        min_side_radius_px=float(args.min_side_radius_px),
        min_radius_px=float(args.min_radius_px),
        min_effective_scale_ratio=float(args.min_effective_scale_ratio),
        min_volume_radius_ratio=float(args.min_volume_radius_ratio),
        min_filter_inflation=float(args.min_filter_inflation),
        min_filter_scale_ratio=float(args.min_filter_scale_ratio),
        min_stress_impact=float(args.min_stress_impact),
        min_stress_radius_gain=float(args.min_stress_radius_gain),
        min_stress_radius_px=float(args.min_stress_radius_px),
        min_stress_major_impact_views=int(args.min_stress_major_impact_views),
        min_stress_major_impact_visible_fraction=float(args.min_stress_major_impact_visible_fraction),
        candidate_mode=str(args.candidate_mode),
        surface_distance_threshold=float(args.surface_distance_threshold),
        require_surface_outside=bool(args.require_surface_outside),
        max_prune_fraction=float(args.max_prune_fraction),
        max_prune_count=int(args.max_prune_count),
    )
    keep_mask = ~prune_mask

    cleaned_model = _write_masked_model(
        mip,
        keep_mask,
        output_model_path.parent,
        output_model_path.name,
        mip_model_path,
        output_iteration,
        meta={
            "source_model_path": str(mip_model_path),
            "source_iteration": int(iteration),
            "prune_count": int(np.count_nonzero(prune_mask)),
        },
    )
    deleted_model = None
    if bool(args.export_deleted_model) and np.any(prune_mask):
        deleted_model = _write_masked_model(
            mip,
            prune_mask,
            output_model_path.parent,
            output_model_path.name + "_deleted_view_aligned_volume_artifacts",
            mip_model_path,
            output_iteration,
            meta={
                "source_model_path": str(mip_model_path),
                "source_iteration": int(iteration),
                "prune_count": int(np.count_nonzero(prune_mask)),
            },
        )

    payload_geom_keys = (
        "xyz",
        "largest_axis",
        "shortest_axis",
        "shortest_axis_idx",
        "scale",
        "raw_scale",
        "scale_max",
        "scale_mid",
        "scale_min",
        "raw_scale_max",
        "axis_anisotropy",
        "full_anisotropy",
        "volume_radius",
        "raw_volume_radius",
        "filter_inflation",
        "filter_scale_ratio",
        "opacity",
        "filter_3D",
    )
    payload = {
        "version": "cleanup_mip_view_aligned_volume_artifacts_v0",
        "score_version": "short_axis_stress_v3",
        "prune_mask": torch.from_numpy(prune_mask.astype(bool))[:, None],
        "keep_mask": torch.from_numpy(keep_mask.astype(bool))[:, None],
        "delete_score": torch.from_numpy(delete_score.astype(np.float32))[:, None],
        "surface_distance": torch.from_numpy(surface_distance.astype(np.float32))[:, None]
        if surface_distance is not None
        else None,
        "geom": {key: torch.from_numpy(geom[key].copy()) for key in payload_geom_keys},
        "view": {key: torch.from_numpy(value.copy()) for key, value in view.items()},
        "stress": {key: torch.from_numpy(value.copy()) for key, value in stress.items()},
        "meta": {
            "scene_root": str(scene_root),
            "mip_model_path": str(mip_model_path),
            "iteration": int(iteration),
            "output_model_path": str(output_model_path),
            "output_iteration": int(output_iteration),
            "images_subdir": str(args.images_subdir),
            "selected_views": [str(cam.image_name) for cam in cameras],
            "scene_diag": float(scene_diag),
            "surface_mesh_path": str(surface_mesh_path) if surface_mesh_path is not None else None,
            "surface_query_mode_used": surface_query_mode_used,
            "delete_info": delete_info,
            "args": vars(args),
        },
    }
    output_model_path.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_model_path / "view_aligned_volume_cleanup_payload.pt")

    visible_mask = view["visible_count"] > 0
    summary = {
        "version": "cleanup_mip_view_aligned_volume_artifacts_v0",
        "score_version": "short_axis_stress_v3",
        "scene_root": str(scene_root),
        "mip_model_path": str(mip_model_path),
        "iteration": int(iteration),
        "output_model_path": str(output_model_path),
        "output_iteration": int(output_iteration),
        "cleaned_model": cleaned_model,
        "deleted_model": deleted_model,
        "num_gaussians": int(prune_mask.shape[0]),
        "visible_gaussians": int(np.count_nonzero(visible_mask)),
        "pruned_count": int(np.count_nonzero(prune_mask)),
        "pruned_ratio": float(np.mean(prune_mask)),
        "delete_info": delete_info,
        "score_stats_visible": {
            "delete_score": _stats(delete_score, visible_mask),
            "axis_alignment_mean": _stats(view["axis_alignment_mean"], visible_mask),
            "ray_thickness_ratio": _stats(view["ray_thickness_ratio"], visible_mask),
            "side_explosion_max": _stats(view["side_explosion_max"], visible_mask),
            "side_radius_max": _stats(view["side_radius_max"], visible_mask),
            "radius_max": _stats(view["radius_max"], visible_mask),
            "stress_impact_score": _stats(stress["stress_impact_score"], visible_mask),
            "stress_raw_impact_score": _stats(stress["stress_raw_impact_score"], visible_mask),
            "stress_impact_view_mean": _stats(stress["stress_impact_view_mean"], visible_mask),
            "stress_impact_max": _stats(stress["stress_impact_max"], visible_mask),
            "stress_major_impact_count": _stats(stress["stress_major_impact_count"], visible_mask),
            "stress_major_impact_visible_fraction": _stats(stress["stress_major_impact_visible_fraction"], visible_mask),
            "stress_radius_gain_max": _stats(stress["stress_radius_gain_max"], visible_mask),
            "stress_radius_max": _stats(stress["stress_radius_max"], visible_mask),
            "stress_large_radius_count": _stats(stress["stress_large_radius_count"], visible_mask),
            "scale_max": _stats(geom["scale_max"], visible_mask),
            "volume_radius": _stats(geom["volume_radius"], visible_mask),
            "filter_inflation": _stats(geom["filter_inflation"], visible_mask),
            "filter_scale_ratio": _stats(geom["filter_scale_ratio"], visible_mask),
            "opacity": _stats(geom["opacity"], visible_mask),
            "surface_distance": _stats(surface_distance, visible_mask) if surface_distance is not None else None,
        },
        "score_stats_pruned": {
            "delete_score": _stats(delete_score, prune_mask),
            "axis_alignment_mean": _stats(view["axis_alignment_mean"], prune_mask),
            "ray_thickness_ratio": _stats(view["ray_thickness_ratio"], prune_mask),
            "side_explosion_max": _stats(view["side_explosion_max"], prune_mask),
            "side_radius_max": _stats(view["side_radius_max"], prune_mask),
            "radius_max": _stats(view["radius_max"], prune_mask),
            "stress_impact_score": _stats(stress["stress_impact_score"], prune_mask),
            "stress_raw_impact_score": _stats(stress["stress_raw_impact_score"], prune_mask),
            "stress_impact_view_mean": _stats(stress["stress_impact_view_mean"], prune_mask),
            "stress_impact_max": _stats(stress["stress_impact_max"], prune_mask),
            "stress_major_impact_count": _stats(stress["stress_major_impact_count"], prune_mask),
            "stress_major_impact_visible_fraction": _stats(stress["stress_major_impact_visible_fraction"], prune_mask),
            "stress_radius_gain_max": _stats(stress["stress_radius_gain_max"], prune_mask),
            "stress_radius_max": _stats(stress["stress_radius_max"], prune_mask),
            "stress_large_radius_count": _stats(stress["stress_large_radius_count"], prune_mask),
            "scale_max": _stats(geom["scale_max"], prune_mask),
            "volume_radius": _stats(geom["volume_radius"], prune_mask),
            "filter_inflation": _stats(geom["filter_inflation"], prune_mask),
            "filter_scale_ratio": _stats(geom["filter_scale_ratio"], prune_mask),
            "opacity": _stats(geom["opacity"], prune_mask),
            "surface_distance": _stats(surface_distance, prune_mask) if surface_distance is not None else None,
        },
        "raw_stats": {
            "scale_max_all": stats_from_array(geom["scale_max"]),
            "filter_3D_all": stats_from_array(geom["filter_3D"]),
            "axis_anisotropy_all": stats_from_array(geom["axis_anisotropy"]),
            "ray_thickness_ratio_all": stats_from_array(view["ray_thickness_ratio"]),
            "side_explosion_max_all": stats_from_array(view["side_explosion_max"]),
            "stress_impact_score_all": stats_from_array(stress["stress_impact_score"]),
            "stress_impact_view_mean_all": stats_from_array(stress["stress_impact_view_mean"]),
            "stress_impact_max_all": stats_from_array(stress["stress_impact_max"]),
            "stress_major_impact_count_all": stats_from_array(stress["stress_major_impact_count"]),
            "stress_major_impact_visible_fraction_all": stats_from_array(stress["stress_major_impact_visible_fraction"]),
            "stress_radius_gain_max_all": stats_from_array(stress["stress_radius_gain_max"]),
            "stress_radius_max_all": stats_from_array(stress["stress_radius_max"]),
            "stress_large_radius_count_all": stats_from_array(stress["stress_large_radius_count"]),
            "filter_inflation_all": stats_from_array(geom["filter_inflation"]),
            "filter_scale_ratio_all": stats_from_array(geom["filter_scale_ratio"]),
        },
    }
    (output_model_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] cleaned mip model : {cleaned_model}")
    print(f"[done] output ply        : {output_model_path}/point_cloud/iteration_{output_iteration}/point_cloud.ply")
    print(f"[done] deleted subset    : {deleted_model}")
    print(f"[done] summary           : {output_model_path}/summary.json")
    print(f"[done] payload           : {output_model_path}/view_aligned_volume_cleanup_payload.pt")


if __name__ == "__main__":
    main()
