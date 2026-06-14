from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from joint_judge_mip_sof_v0 import (
    apply_manual_prune,
    apply_soft_suppression,
    cap_mask,
    collect_render_metrics,
    save_payload,
    static_gaussian_metrics,
    stats_from_array,
)
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import (
    build_dataset_args,
    build_prepared_sr_cache,
    clamp_xyz_displacement,
    compute_prepared_sr_losses,
    copy_render_config,
    evaluate_sr_prior_losses,
    load_model_ply,
    load_train_cameras_only,
    resolve_iteration,
    select_uniform,
)
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.prior_fusion import project_points_camera
from utils.system_utils import mkdir_p


def freeze_for_global_refine(
    gaussians: GaussianModel,
    *,
    train_xyz: bool,
    train_opacity: bool,
    train_scale: bool,
) -> None:
    gaussians._xyz.requires_grad_(bool(train_xyz))
    gaussians._opacity.requires_grad_(bool(train_opacity))
    gaussians._scaling.requires_grad_(bool(train_scale))
    gaussians._rotation.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)


def build_optimizer(
    gaussians: GaussianModel,
    *,
    xyz_lr: float,
    opacity_lr: float,
    scale_lr: float,
    train_opacity: bool,
    train_scale: bool,
) -> torch.optim.Optimizer:
    params = [{"params": [gaussians._xyz], "lr": float(xyz_lr), "name": "xyz"}]
    if bool(train_opacity):
        params.append({"params": [gaussians._opacity], "lr": float(opacity_lr), "name": "opacity"})
    if bool(train_scale):
        params.append({"params": [gaussians._scaling], "lr": float(scale_lr), "name": "scale"})
    return torch.optim.Adam(params, eps=1e-15)


@torch.no_grad()
def cache_mip_renders(
    cameras: Sequence[object],
    mip_anchor: GaussianModel,
    background: torch.Tensor,
) -> List[torch.Tensor]:
    cache: List[torch.Tensor] = []
    for camera in tqdm(cameras, desc="cache mip anchors"):
        cache.append(render_simple(camera, mip_anchor, background)["render"].detach().clamp(0.0, 1.0))
    return cache


def camera_rgb(camera, device: torch.device) -> torch.Tensor:
    image = camera.original_image
    if image.ndim != 3:
        raise ValueError(f"camera.original_image must be 3D, got {tuple(image.shape)}")
    if image.shape[0] == 3:
        return image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported camera.original_image shape: {tuple(image.shape)}")


def build_dense_risk_weight(
    static: Dict[str, np.ndarray],
    render: Dict[str, np.ndarray],
    *,
    radius_threshold: float,
    lr_residual_threshold: float,
    anisotropy_threshold: float,
    min_support_views: int,
) -> torch.Tensor:
    radius_score = np.clip(
        (render["max_radius"] - float(radius_threshold)) / max(float(radius_threshold), 1.0),
        0.0,
        1.0,
    )
    residual_score = np.clip(
        (render["residual_max"] - float(lr_residual_threshold)) / max(1.0 - float(lr_residual_threshold), 1e-6),
        0.0,
        1.0,
    )
    anisotropy = np.maximum(static["anisotropy"], 1.0)
    anisotropy_score = np.clip(
        np.log(anisotropy) / max(np.log(float(anisotropy_threshold)), 1e-6) - 1.0,
        0.0,
        1.0,
    )
    support_score = np.clip(
        (float(min_support_views) - render["visible_count"].astype(np.float32)) / max(float(min_support_views), 1.0),
        0.0,
        1.0,
    )
    risk = radius_score * (0.50 * residual_score + 0.25 * support_score + 0.15 * anisotropy_score + 0.10)
    risk = np.maximum(risk, 0.15 * anisotropy_score * support_score)
    return torch.from_numpy(np.clip(risk, 0.0, 1.0).astype(np.float32))


def mean_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a - b))


@torch.no_grad()
def evaluate_rgb_losses(
    cameras: Sequence[object],
    mip_cache: Sequence[torch.Tensor],
    student: GaussianModel,
    background: torch.Tensor,
) -> Dict[str, float]:
    mip_losses = []
    lr_losses = []
    for camera, mip_rgb in zip(cameras, mip_cache):
        rgb = render_simple(camera, student, background)["render"].detach().clamp(0.0, 1.0)
        mip_losses.append(float(mean_l1(rgb, mip_rgb.to(device=rgb.device)).item()))
        lr_losses.append(float(mean_l1(rgb, camera_rgb(camera, background.device)).item()))
    return {
        "mip_rgb": float(np.mean(mip_losses)) if mip_losses else 0.0,
        "lr_rgb": float(np.mean(lr_losses)) if lr_losses else 0.0,
    }


@torch.no_grad()
def collect_sr_signal_metrics(
    gaussians: GaussianModel,
    cameras: Sequence[object],
    sr_cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    *,
    background: torch.Tensor,
    support_threshold: float,
    depth_min: float,
) -> Dict[str, np.ndarray]:
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    total = int(xyz.shape[0])
    visible_count = np.zeros((total,), dtype=np.int32)
    support_count = np.zeros((total,), dtype=np.int32)
    mask_sum = np.zeros((total,), dtype=np.float64)
    mask_max = np.zeros((total,), dtype=np.float32)
    residual_weight_sum = np.zeros((total,), dtype=np.float64)
    residual_weight = np.zeros((total,), dtype=np.float64)
    residual_max = np.zeros((total,), dtype=np.float32)

    for camera, target in tqdm(list(zip(cameras, sr_cache)), desc="SR signal metrics"):
        package = render_simple(camera, gaussians, background)
        radii = package["radii"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        active = radii > 0
        if not np.any(active):
            continue

        projected, valid = project_points_camera(camera, xyz, depth_min=float(depth_min), margin=0)
        active &= valid
        active_ids = np.flatnonzero(active).astype(np.int64, copy=False)
        if active_ids.size == 0:
            continue

        render_rgb = package["render"].detach().clamp(0.0, 1.0)
        prior_rgb = target["prior_rgb"]
        prior_mask = target["prior_mask"]
        if not torch.is_tensor(prior_rgb) or not torch.is_tensor(prior_mask):
            continue
        prior_rgb = prior_rgb.to(device=background.device, dtype=render_rgb.dtype).clamp(0.0, 1.0)
        prior_mask = prior_mask.to(device=background.device, dtype=render_rgb.dtype).clamp(0.0, 1.0)

        residual_map = torch.mean(torch.abs(render_rgb - prior_rgb), dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        mask_map = prior_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        xy = projected[active_ids, :2]
        x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, mask_map.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, mask_map.shape[0] - 1)
        sampled_mask = mask_map[y, x].astype(np.float32, copy=False)
        sampled_residual = residual_map[y, x].astype(np.float32, copy=False)

        visible_count[active_ids] += 1
        mask_sum[active_ids] += sampled_mask.astype(np.float64)
        mask_max[active_ids] = np.maximum(mask_max[active_ids], sampled_mask)
        supported = sampled_mask >= float(support_threshold)
        if np.any(supported):
            support_count[active_ids[supported]] += 1
        residual_weight_sum[active_ids] += (sampled_residual * sampled_mask).astype(np.float64)
        residual_weight[active_ids] += sampled_mask.astype(np.float64)
        residual_max[active_ids] = np.maximum(residual_max[active_ids], sampled_residual * sampled_mask)

    mask_mean = np.divide(
        mask_sum,
        np.maximum(visible_count, 1),
        out=np.zeros_like(mask_sum, dtype=np.float64),
        where=visible_count > 0,
    ).astype(np.float32)
    residual_mean = np.divide(
        residual_weight_sum,
        np.maximum(residual_weight, 1e-6),
        out=np.zeros_like(residual_weight_sum, dtype=np.float64),
        where=residual_weight > 1e-6,
    ).astype(np.float32)
    support_ratio = np.divide(
        support_count.astype(np.float32),
        np.maximum(visible_count, 1).astype(np.float32),
        out=np.zeros_like(mask_mean, dtype=np.float32),
        where=visible_count > 0,
    ).astype(np.float32)
    return {
        "visible_count": visible_count,
        "support_count": support_count,
        "support_ratio": support_ratio,
        "mask_mean": mask_mean,
        "mask_max": mask_max,
        "prior_residual_mean": residual_mean,
        "prior_residual_max_weighted": residual_max,
    }


def build_lr_sr_cleanup_masks(
    static: Dict[str, np.ndarray],
    lr_render: Dict[str, np.ndarray],
    sr_signal: Dict[str, np.ndarray],
    risk_weight: np.ndarray,
    *,
    cleanup_mode: str,
    lr_protect_threshold: float,
    lr_soft_bad_threshold: float,
    lr_hard_bad_threshold: float,
    sr_support_threshold: float,
    sr_protect_residual_threshold: float,
    sr_bad_residual_threshold: float,
    soft_risk_threshold: float,
    hard_risk_threshold: float,
    radius_threshold: float,
    anisotropy_threshold: float,
    max_visible_views: int,
    max_large_views: int,
    max_prune_fraction: float,
    max_prune_count: int,
    max_suppress_fraction: float,
    max_suppress_count: int,
) -> Dict[str, np.ndarray]:
    if cleanup_mode == "off":
        total = int(static["opacity"].shape[0])
        empty = np.zeros((total,), dtype=bool)
        return {
            "hard_prune_mask": empty,
            "soft_suppress_mask": empty.copy(),
            "candidate_hard_mask": empty.copy(),
            "candidate_soft_mask": empty.copy(),
            "candidate_delete_mask": empty.copy(),
            "score": np.zeros((total,), dtype=np.float32),
            "protected_by_signal": empty.copy(),
        }

    lr_signal = np.maximum(lr_render["residual_max"], lr_render["large_residual_max"])
    lr_good = (lr_render["visible_count"] > 0) & (lr_render["residual_mean"] <= float(lr_protect_threshold))
    lr_soft_bad = (lr_render["visible_count"] > 0) & (lr_signal >= float(lr_soft_bad_threshold))
    lr_hard_bad = (lr_render["visible_count"] > 0) & (lr_signal >= float(lr_hard_bad_threshold))

    sr_visible = sr_signal["visible_count"] > 0
    sr_supported = sr_visible & (
        (sr_signal["mask_max"] >= float(sr_support_threshold))
        | (sr_signal["support_count"] > 0)
    )
    sr_good = sr_supported & (sr_signal["prior_residual_mean"] <= float(sr_protect_residual_threshold))
    sr_bad = sr_supported & (sr_signal["prior_residual_mean"] >= float(sr_bad_residual_threshold))
    sr_no_support = sr_visible & (~sr_supported)
    protected_by_signal = lr_good | sr_good

    large = lr_render["max_radius"] >= float(radius_threshold)
    sparse = (
        (lr_render["visible_count"] <= int(max_visible_views))
        | ((lr_render["large_view_count"] > 0) & (lr_render["large_view_count"] <= int(max_large_views)))
    )
    anisotropic = static["anisotropy"] >= float(anisotropy_threshold)
    soft_geometry = (risk_weight >= float(soft_risk_threshold)) | large | anisotropic
    hard_geometry = (risk_weight >= float(hard_risk_threshold)) & (large | anisotropic) & (sparse | anisotropic)

    # LR/SR are the gate. In hybrid mode only the strongest LR/SR disagreements
    # are pruned; delete mode promotes the same gated soft candidates to prune.
    signal_hard = lr_hard_bad & (sr_bad | sr_no_support)
    signal_soft = lr_soft_bad | sr_bad | sr_no_support
    candidate_hard = signal_hard & hard_geometry & (~protected_by_signal)
    if cleanup_mode == "soft":
        candidate_hard &= False
    candidate_soft = signal_soft & soft_geometry & (~protected_by_signal) & (~candidate_hard)

    lr_score = np.clip(
        (lr_signal - float(lr_soft_bad_threshold)) / max(1.0 - float(lr_soft_bad_threshold), 1e-6),
        0.0,
        1.0,
    )
    sr_score = np.clip(
        (sr_signal["prior_residual_mean"] - float(sr_protect_residual_threshold))
        / max(1.0 - float(sr_protect_residual_threshold), 1e-6),
        0.0,
        1.0,
    )
    radius_score = np.clip(
        (lr_render["max_radius"] - float(radius_threshold)) / max(float(radius_threshold), 1.0),
        0.0,
        2.0,
    )
    anisotropy_score = np.clip(
        np.log(np.maximum(static["anisotropy"], 1.0)) / max(np.log(float(anisotropy_threshold)), 1e-6) - 1.0,
        0.0,
        2.0,
    )
    no_sr_support_score = sr_no_support.astype(np.float32)
    score = (
        2.0 * lr_score
        + 1.5 * sr_score
        + 1.0 * risk_weight
        + 0.5 * radius_score
        + 0.5 * anisotropy_score
        + 0.5 * no_sr_support_score
    ).astype(np.float32)
    if cleanup_mode == "delete":
        prune_candidates = candidate_hard | candidate_soft
    else:
        prune_candidates = candidate_hard
    if float(max_prune_fraction) <= 0.0 and int(max_prune_count) <= 0:
        hard = np.zeros_like(prune_candidates, dtype=bool)
    else:
        hard = cap_mask(prune_candidates, score, max_fraction=float(max_prune_fraction), max_count=int(max_prune_count))
    if cleanup_mode == "delete" or (float(max_suppress_fraction) <= 0.0 and int(max_suppress_count) <= 0):
        soft = np.zeros_like(candidate_soft, dtype=bool)
    else:
        soft = cap_mask(candidate_soft, score, max_fraction=float(max_suppress_fraction), max_count=int(max_suppress_count))
    return {
        "hard_prune_mask": hard.astype(bool),
        "soft_suppress_mask": soft.astype(bool),
        "candidate_hard_mask": candidate_hard.astype(bool),
        "candidate_soft_mask": candidate_soft.astype(bool),
        "candidate_delete_mask": (candidate_hard | candidate_soft).astype(bool),
        "score": score,
        "protected_by_signal": protected_by_signal.astype(bool),
        "lr_good": lr_good.astype(bool),
        "lr_soft_bad": lr_soft_bad.astype(bool),
        "lr_hard_bad": lr_hard_bad.astype(bool),
        "sr_supported": sr_supported.astype(bool),
        "sr_good": sr_good.astype(bool),
        "sr_bad": sr_bad.astype(bool),
        "sr_no_support": sr_no_support.astype(bool),
        "hard_geometry": hard_geometry.astype(bool),
        "soft_geometry": soft_geometry.astype(bool),
    }


def main() -> None:
    parser = ArgumentParser(description="Global dense-weight refinement starting from SOF-prior field.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--start_sof_model_path", required=True)
    parser.add_argument("--mip_anchor_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--start_iteration", type=int, default=32000)
    parser.add_argument("--mip_iteration", type=int, default=30000)
    parser.add_argument("--output_iteration", type=int, default=33000)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--sr_images_subdir", default="images_2")
    parser.add_argument("--sr_prior_root", type=str, default=None)
    parser.add_argument("--sr_prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--sr_prior_mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--sr_anchor_subdir", type=str, default="aligned_references")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--sr_max_views", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--xyz_lr", type=float, default=5e-6)
    parser.add_argument("--opacity_lr", type=float, default=5e-4)
    parser.add_argument("--scale_lr", type=float, default=1e-4)
    parser.add_argument("--enable_opacity_update", action="store_true")
    parser.add_argument("--enable_scale_update", action="store_true")
    parser.add_argument("--lambda_mip_rgb", type=float, default=0.35)
    parser.add_argument("--lambda_lr_rgb", type=float, default=0.10)
    parser.add_argument("--lambda_sr_l1", type=float, default=0.05)
    parser.add_argument("--lambda_sr_hf", type=float, default=0.30)
    parser.add_argument("--lambda_sr_mip_rgb", type=float, default=0.10)
    parser.add_argument("--lambda_xyz_anchor", type=float, default=35.0)
    parser.add_argument("--lambda_opacity_anchor", type=float, default=0.05)
    parser.add_argument("--lambda_scale_anchor", type=float, default=0.20)
    parser.add_argument("--lambda_risk_opacity", type=float, default=0.02)
    parser.add_argument("--lambda_risk_scale", type=float, default=0.002)
    parser.add_argument("--risk_radius_threshold", type=float, default=56.0)
    parser.add_argument("--risk_lr_residual_threshold", type=float, default=0.12)
    parser.add_argument("--risk_anisotropy_threshold", type=float, default=20.0)
    parser.add_argument("--risk_min_support_views", type=int, default=4)
    parser.add_argument("--sr_prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--sr_prior_consistency_threshold", type=float, default=0.12)
    parser.add_argument("--sr_prior_min_valid_ratio", type=float, default=0.50)
    parser.add_argument("--sr_prior_min_pixels", type=float, default=64.0)
    parser.add_argument("--sr_prior_delta_clip", type=float, default=0.15)
    parser.add_argument("--disable_sr_prior_hf_residual", action="store_true")
    parser.add_argument("--max_displacement_ratio", type=float, default=0.001)
    parser.add_argument("--max_displacement_abs", type=float, default=0.0)
    parser.add_argument("--cleanup_mode", choices=("off", "soft", "hybrid", "delete"), default="off")
    parser.add_argument("--cleanup_lr_protect_threshold", type=float, default=0.08)
    parser.add_argument("--cleanup_lr_soft_bad_threshold", type=float, default=0.14)
    parser.add_argument("--cleanup_lr_hard_bad_threshold", type=float, default=0.20)
    parser.add_argument("--cleanup_sr_support_threshold", type=float, default=0.25)
    parser.add_argument("--cleanup_sr_protect_residual_threshold", type=float, default=0.10)
    parser.add_argument("--cleanup_sr_bad_residual_threshold", type=float, default=0.20)
    parser.add_argument("--cleanup_soft_risk_threshold", type=float, default=0.35)
    parser.add_argument("--cleanup_hard_risk_threshold", type=float, default=0.60)
    parser.add_argument("--cleanup_radius_threshold", type=float, default=56.0)
    parser.add_argument("--cleanup_anisotropy_threshold", type=float, default=24.0)
    parser.add_argument("--cleanup_max_visible_views", type=int, default=3)
    parser.add_argument("--cleanup_max_large_views", type=int, default=1)
    parser.add_argument("--cleanup_max_prune_fraction", type=float, default=0.002)
    parser.add_argument("--cleanup_max_prune_count", type=int, default=0)
    parser.add_argument("--cleanup_max_suppress_fraction", type=float, default=0.02)
    parser.add_argument("--cleanup_max_suppress_count", type=int, default=0)
    parser.add_argument("--cleanup_soft_opacity_scale", type=float, default=0.65)
    parser.add_argument("--cleanup_soft_scale_shrink", type=float, default=0.85)
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    start_model_path = Path(args.start_sof_model_path).expanduser().resolve()
    mip_model_path = Path(args.mip_anchor_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    sr_prior_root = Path(args.sr_prior_root).expanduser().resolve() if args.sr_prior_root else None
    if sr_prior_root is None:
        raise ValueError("--sr_prior_root is required for global refine v0")
    sr_prior_dir = sr_prior_root / str(args.sr_prior_subdir)
    sr_mask_dir = sr_prior_root / str(args.sr_prior_mask_subdir)
    sr_anchor_dir = sr_prior_root / str(args.sr_anchor_subdir)
    for label, path in (("sr_prior_dir", sr_prior_dir), ("sr_mask_dir", sr_mask_dir), ("sr_anchor_dir", sr_anchor_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")

    dataset_args = build_dataset_args(str(scene_root), str(start_model_path), str(args.images_subdir))
    lr_all_cameras = load_train_cameras_only(scene_root, start_model_path, str(args.images_subdir))
    lr_cameras = select_uniform(lr_all_cameras, int(args.max_views))
    if not lr_cameras:
        raise RuntimeError("No LR cameras found.")
    sr_all_cameras = load_train_cameras_only(scene_root, start_model_path, str(args.sr_images_subdir))
    selected_names = {normalize_image_name(cam.image_name) for cam in lr_cameras}
    sr_candidates = [cam for cam in sr_all_cameras if normalize_image_name(cam.image_name) in selected_names]
    if not sr_candidates:
        sr_candidates = sr_all_cameras
    sr_cameras = select_uniform(sr_candidates, int(args.sr_max_views))
    if not sr_cameras:
        raise RuntimeError("No SR cameras found.")

    start_iter = resolve_iteration(start_model_path, int(args.start_iteration))
    mip_iter = resolve_iteration(mip_model_path, int(args.mip_iteration))
    student = load_model_ply(start_model_path, start_iter, int(dataset_args.sh_degree))
    mip_anchor = load_model_ply(mip_model_path, mip_iter, int(dataset_args.sh_degree))
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    filter_cameras = lr_cameras + [cam for cam in sr_cameras if cam not in lr_cameras]
    student.compute_3D_filter(filter_cameras, CUDA=False)
    mip_anchor.compute_3D_filter(filter_cameras, CUDA=False)

    sr_index = index_image_dir(str(sr_prior_dir))
    sr_anchor_index = index_image_dir(str(sr_anchor_dir))
    sr_cameras, sr_cache = build_prepared_sr_cache(
        sr_cameras,
        student,
        background,
        sr_prior_index=sr_index,
        sr_anchor_index=sr_anchor_index,
        sr_prior_mask_dir=sr_mask_dir,
        sr_prior_mask_suffix="",
        prior_consistency_threshold=float(args.sr_prior_consistency_threshold),
        prior_mask_floor=float(args.sr_prior_mask_floor),
    )
    if not sr_cache:
        raise RuntimeError(f"No SR priors matched SR cameras under {sr_prior_root}")

    lr_mip_cache = cache_mip_renders(lr_cameras, mip_anchor, background)
    sr_mip_cache = cache_mip_renders(sr_cameras, mip_anchor, background)

    static = static_gaussian_metrics(student)
    render_metrics = collect_render_metrics(
        student,
        lr_cameras,
        large_radius_px=float(args.risk_radius_threshold),
        residual_against_camera=True,
        depth_min=0.02,
        background=background,
    )
    risk_weight = build_dense_risk_weight(
        static,
        render_metrics,
        radius_threshold=float(args.risk_radius_threshold),
        lr_residual_threshold=float(args.risk_lr_residual_threshold),
        anisotropy_threshold=float(args.risk_anisotropy_threshold),
        min_support_views=int(args.risk_min_support_views),
    ).to(device=background.device)

    xyz_init = student._xyz.detach().clone()
    opacity_init = student.get_opacity.detach().clone()
    scale_init = student.get_scaling.detach().clone()
    bbox_diag = torch.linalg.norm(torch.max(xyz_init, dim=0).values - torch.min(xyz_init, dim=0).values).clamp_min(1e-6)
    max_displacement = (
        float(args.max_displacement_abs)
        if float(args.max_displacement_abs) > 0.0
        else float(args.max_displacement_ratio) * float(bbox_diag.item())
    )

    freeze_for_global_refine(
        student,
        train_xyz=True,
        train_opacity=bool(args.enable_opacity_update),
        train_scale=bool(args.enable_scale_update),
    )
    optimizer = build_optimizer(
        student,
        xyz_lr=float(args.xyz_lr),
        opacity_lr=float(args.opacity_lr),
        scale_lr=float(args.scale_lr),
        train_opacity=bool(args.enable_opacity_update),
        train_scale=bool(args.enable_scale_update),
    )

    print(f"[global-refine-sof-v0] scene       : {scene_root}")
    print(f"[global-refine-sof-v0] start SOF   : {start_model_path} iter={start_iter} n={student.get_xyz.shape[0]}")
    print(f"[global-refine-sof-v0] mip anchor  : {mip_model_path} iter={mip_iter}")
    print(f"[global-refine-sof-v0] output      : {output_model_path}")
    print(f"[global-refine-sof-v0] views       : lr={len(lr_cameras)} sr={len(sr_cameras)}")
    print(f"[global-refine-sof-v0] train       : iter={args.iterations} xyz_lr={args.xyz_lr} op={args.enable_opacity_update} scale={args.enable_scale_update}")
    print(f"[global-refine-sof-v0] risk mean   : {float(risk_weight.mean().item()):.6f} max={float(risk_weight.max().item()):.6f}")
    print(f"[global-refine-sof-v0] cleanup     : mode={args.cleanup_mode} lr/sr-gated")

    before_rgb = evaluate_rgb_losses(lr_cameras, lr_mip_cache, student, background)
    before_sr = evaluate_sr_prior_losses(
        sr_cameras,
        sr_cache,
        student,
        background,
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
        disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
    )

    progress = tqdm(range(1, int(args.iterations) + 1), desc="global SOF refine")
    log_rows = []
    for iteration in progress:
        loss = torch.zeros((), dtype=torch.float32, device="cuda")
        lr_idx = randint(0, len(lr_cameras) - 1)
        lr_camera = lr_cameras[lr_idx]
        lr_rgb = render_simple(lr_camera, student, background)["render"].clamp(0.0, 1.0)
        mip_lr = lr_mip_cache[lr_idx].to(device=lr_rgb.device, dtype=lr_rgb.dtype)
        lr_anchor = camera_rgb(lr_camera, background.device).to(dtype=lr_rgb.dtype)
        loss_mip_rgb = mean_l1(lr_rgb, mip_lr)
        loss_lr_rgb = mean_l1(lr_rgb, lr_anchor)
        loss = loss + float(args.lambda_mip_rgb) * loss_mip_rgb
        loss = loss + float(args.lambda_lr_rgb) * loss_lr_rgb

        sr_l1_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        sr_hf_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        sr_mip_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if sr_cache:
            sr_idx = randint(0, len(sr_cache) - 1)
            sr_camera = sr_cameras[sr_idx]
            sr_rgb = render_simple(sr_camera, student, background)["render"].clamp(0.0, 1.0)
            sr_mip = sr_mip_cache[sr_idx].to(device=sr_rgb.device, dtype=sr_rgb.dtype)
            sr_mip_loss = mean_l1(sr_rgb, sr_mip)
            maybe_sr_l1, maybe_sr_hf = compute_prepared_sr_losses(
                sr_rgb,
                sr_cache[sr_idx],
                min_pixels=float(args.sr_prior_min_pixels),
                min_valid_ratio=float(args.sr_prior_min_valid_ratio),
                prior_delta_clip=float(args.sr_prior_delta_clip),
                disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
            )
            if maybe_sr_l1 is not None:
                sr_l1_loss = maybe_sr_l1
                loss = loss + float(args.lambda_sr_l1) * sr_l1_loss
            if maybe_sr_hf is not None:
                sr_hf_loss = maybe_sr_hf
                loss = loss + float(args.lambda_sr_hf) * sr_hf_loss
            loss = loss + float(args.lambda_sr_mip_rgb) * sr_mip_loss

        xyz_delta = (student._xyz - xyz_init) / bbox_diag
        loss_xyz_anchor = torch.mean(xyz_delta * xyz_delta)
        loss = loss + float(args.lambda_xyz_anchor) * loss_xyz_anchor

        current_opacity = student.get_opacity
        current_scale = student.get_scaling
        loss_opacity_anchor = torch.mean(torch.abs(current_opacity - opacity_init))
        rel_scale = (current_scale - scale_init) / torch.clamp(scale_init, min=1e-8)
        loss_scale_anchor = torch.mean(rel_scale * rel_scale)
        if bool(args.enable_opacity_update):
            loss = loss + float(args.lambda_opacity_anchor) * loss_opacity_anchor
            loss = loss + float(args.lambda_risk_opacity) * torch.mean(risk_weight[:, None] * current_opacity)
        if bool(args.enable_scale_update):
            loss = loss + float(args.lambda_scale_anchor) * loss_scale_anchor
            scale_max = current_scale.max(dim=1).values
            init_scale_max = torch.clamp(scale_init.max(dim=1).values, min=1e-8)
            loss = loss + float(args.lambda_risk_scale) * torch.mean(risk_weight * scale_max / init_scale_max)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        clamp_xyz_displacement(student, xyz_init, max_displacement=max_displacement)

        row = {
            "iter": int(iteration),
            "loss": float(loss.detach().item()),
            "mip_rgb": float(loss_mip_rgb.detach().item()),
            "lr_rgb": float(loss_lr_rgb.detach().item()),
            "sr_l1": float(sr_l1_loss.detach().item()),
            "sr_hf": float(sr_hf_loss.detach().item()),
            "sr_mip": float(sr_mip_loss.detach().item()),
            "xyz_anchor": float(loss_xyz_anchor.detach().item()),
            "opacity_anchor": float(loss_opacity_anchor.detach().item()),
            "scale_anchor": float(loss_scale_anchor.detach().item()),
        }
        log_rows.append(row)
        if iteration % 10 == 0:
            progress.set_postfix(
                {
                    "loss": f"{row['loss']:.5f}",
                    "mip": f"{row['mip_rgb']:.4f}",
                    "lr": f"{row['lr_rgb']:.4f}",
                    "sr": f"{row['sr_l1']:.4f}/{row['sr_hf']:.4f}",
                    "anc": f"{row['xyz_anchor']:.6f}",
                }
            )

        if int(args.save_every) > 0 and iteration % int(args.save_every) == 0:
            point_dir = output_model_path / "point_cloud" / f"iteration_{iteration}"
            mkdir_p(str(point_dir))
            student.save_ply(str(point_dir / "point_cloud.ply"))
            student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    student.compute_3D_filter(filter_cameras, CUDA=False)
    pre_cleanup_rgb = evaluate_rgb_losses(lr_cameras, lr_mip_cache, student, background)
    pre_cleanup_sr = evaluate_sr_prior_losses(
        sr_cameras,
        sr_cache,
        student,
        background,
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
        disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
    )

    final_xyz_reference = xyz_init
    cleanup_summary = {
        "mode": str(args.cleanup_mode),
        "before_count": int(student.get_xyz.shape[0]),
        "after_count": int(student.get_xyz.shape[0]),
        "candidate_hard_count": 0,
        "candidate_soft_count": 0,
        "candidate_delete_count": 0,
        "hard_pruned_count": 0,
        "soft_suppressed_count": 0,
        "payload_path": None,
        "signal_priority": "LR/SR gates cleanup; geometry only acts as auxiliary evidence.",
        "config": {
            "lr_protect_threshold": float(args.cleanup_lr_protect_threshold),
            "lr_soft_bad_threshold": float(args.cleanup_lr_soft_bad_threshold),
            "lr_hard_bad_threshold": float(args.cleanup_lr_hard_bad_threshold),
            "sr_support_threshold": float(args.cleanup_sr_support_threshold),
            "sr_protect_residual_threshold": float(args.cleanup_sr_protect_residual_threshold),
            "sr_bad_residual_threshold": float(args.cleanup_sr_bad_residual_threshold),
            "soft_risk_threshold": float(args.cleanup_soft_risk_threshold),
            "hard_risk_threshold": float(args.cleanup_hard_risk_threshold),
            "radius_threshold": float(args.cleanup_radius_threshold),
            "anisotropy_threshold": float(args.cleanup_anisotropy_threshold),
            "max_visible_views": int(args.cleanup_max_visible_views),
            "max_large_views": int(args.cleanup_max_large_views),
            "max_prune_fraction": float(args.cleanup_max_prune_fraction),
            "max_prune_count": int(args.cleanup_max_prune_count),
            "max_suppress_fraction": float(args.cleanup_max_suppress_fraction),
            "max_suppress_count": int(args.cleanup_max_suppress_count),
            "soft_opacity_scale": float(args.cleanup_soft_opacity_scale),
            "soft_scale_shrink": float(args.cleanup_soft_scale_shrink),
        },
    }
    if str(args.cleanup_mode) != "off":
        cleanup_static = static_gaussian_metrics(student)
        cleanup_lr_render = collect_render_metrics(
            student,
            lr_cameras,
            large_radius_px=float(args.cleanup_radius_threshold),
            residual_against_camera=True,
            depth_min=0.02,
            background=background,
        )
        cleanup_sr_signal = collect_sr_signal_metrics(
            student,
            sr_cameras,
            sr_cache,
            background=background,
            support_threshold=float(args.cleanup_sr_support_threshold),
            depth_min=0.02,
        )
        cleanup_masks = build_lr_sr_cleanup_masks(
            cleanup_static,
            cleanup_lr_render,
            cleanup_sr_signal,
            risk_weight.detach().cpu().numpy().astype(np.float32, copy=False),
            cleanup_mode=str(args.cleanup_mode),
            lr_protect_threshold=float(args.cleanup_lr_protect_threshold),
            lr_soft_bad_threshold=float(args.cleanup_lr_soft_bad_threshold),
            lr_hard_bad_threshold=float(args.cleanup_lr_hard_bad_threshold),
            sr_support_threshold=float(args.cleanup_sr_support_threshold),
            sr_protect_residual_threshold=float(args.cleanup_sr_protect_residual_threshold),
            sr_bad_residual_threshold=float(args.cleanup_sr_bad_residual_threshold),
            soft_risk_threshold=float(args.cleanup_soft_risk_threshold),
            hard_risk_threshold=float(args.cleanup_hard_risk_threshold),
            radius_threshold=float(args.cleanup_radius_threshold),
            anisotropy_threshold=float(args.cleanup_anisotropy_threshold),
            max_visible_views=int(args.cleanup_max_visible_views),
            max_large_views=int(args.cleanup_max_large_views),
            max_prune_fraction=float(args.cleanup_max_prune_fraction),
            max_prune_count=int(args.cleanup_max_prune_count),
            max_suppress_fraction=float(args.cleanup_max_suppress_fraction),
            max_suppress_count=int(args.cleanup_max_suppress_count),
        )
        hard_prune_mask = cleanup_masks["hard_prune_mask"].astype(bool)
        soft_suppress_mask = cleanup_masks["soft_suppress_mask"].astype(bool)
        if np.any(soft_suppress_mask):
            apply_soft_suppression(
                student,
                soft_suppress_mask,
                opacity_scale=float(args.cleanup_soft_opacity_scale),
                scale_shrink=float(args.cleanup_soft_scale_shrink),
            )
        if np.any(hard_prune_mask):
            keep_mask = torch.from_numpy(~hard_prune_mask).to(device=xyz_init.device, dtype=torch.bool)
            final_xyz_reference = xyz_init[keep_mask]
            apply_manual_prune(student, hard_prune_mask)

        output_model_path.mkdir(parents=True, exist_ok=True)
        cleanup_payload_path = output_model_path / "global_refine_sof_v0_cleanup_payload.pt"
        save_payload(
            cleanup_payload_path,
            {
                "masks": cleanup_masks,
                "lr_render": cleanup_lr_render,
                "sr_signal": cleanup_sr_signal,
                "static": cleanup_static,
                "risk_weight": risk_weight.detach().cpu().numpy().astype(np.float32, copy=False),
            },
        )
        cleanup_summary.update(
            {
                "after_count": int(student.get_xyz.shape[0]),
                "candidate_hard_count": int(np.count_nonzero(cleanup_masks["candidate_hard_mask"])),
                "candidate_soft_count": int(np.count_nonzero(cleanup_masks["candidate_soft_mask"])),
                "candidate_delete_count": int(np.count_nonzero(cleanup_masks["candidate_delete_mask"])),
                "hard_pruned_count": int(np.count_nonzero(hard_prune_mask)),
                "soft_suppressed_count": int(np.count_nonzero(soft_suppress_mask)),
                "protected_by_signal_count": int(np.count_nonzero(cleanup_masks["protected_by_signal"])),
                "payload_path": str(cleanup_payload_path),
                "score": stats_from_array(cleanup_masks["score"].astype(np.float32, copy=False)),
                "lr_signal": {
                    "residual_mean": stats_from_array(cleanup_lr_render["residual_mean"]),
                    "residual_max": stats_from_array(cleanup_lr_render["residual_max"]),
                    "large_residual_max": stats_from_array(cleanup_lr_render["large_residual_max"]),
                },
                "sr_signal": {
                    "mask_mean": stats_from_array(cleanup_sr_signal["mask_mean"]),
                    "mask_max": stats_from_array(cleanup_sr_signal["mask_max"]),
                    "prior_residual_mean": stats_from_array(cleanup_sr_signal["prior_residual_mean"]),
                    "support_ratio": stats_from_array(cleanup_sr_signal["support_ratio"]),
                },
            }
        )
        print(
            "[global-refine-sof-v0] cleanup "
            f"mode={args.cleanup_mode} hard={cleanup_summary['hard_pruned_count']} "
            f"soft={cleanup_summary['soft_suppressed_count']} "
            f"protected={cleanup_summary['protected_by_signal_count']}",
            flush=True,
        )

    student.compute_3D_filter(filter_cameras, CUDA=False)
    output_model_path.mkdir(parents=True, exist_ok=True)
    copy_render_config(start_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
    mkdir_p(str(point_dir))
    student.save_ply(str(point_dir / "point_cloud.ply"))
    student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    after_rgb = evaluate_rgb_losses(lr_cameras, lr_mip_cache, student, background)
    after_sr = evaluate_sr_prior_losses(
        sr_cameras,
        sr_cache,
        student,
        background,
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
        disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
    )
    displacement = torch.linalg.norm(student._xyz.detach() - final_xyz_reference, dim=1)
    summary = {
        "version": "global_refine_sof_v0",
        "scene_root": str(scene_root),
        "start_sof_model_path": str(start_model_path),
        "mip_anchor_model_path": str(mip_model_path),
        "output_model_path": str(output_model_path),
        "start_iteration": int(start_iter),
        "mip_iteration": int(mip_iter),
        "output_iteration": int(args.output_iteration),
        "images_subdir": str(args.images_subdir),
        "sr_images_subdir": str(args.sr_images_subdir),
        "iterations": int(args.iterations),
        "train_opacity": bool(args.enable_opacity_update),
        "train_scale": bool(args.enable_scale_update),
        "max_displacement": float(max_displacement),
        "loss_weights": {
            "mip_rgb": float(args.lambda_mip_rgb),
            "lr_rgb": float(args.lambda_lr_rgb),
            "sr_l1": float(args.lambda_sr_l1),
            "sr_hf": float(args.lambda_sr_hf),
            "sr_mip_rgb": float(args.lambda_sr_mip_rgb),
            "xyz_anchor": float(args.lambda_xyz_anchor),
            "opacity_anchor": float(args.lambda_opacity_anchor),
            "scale_anchor": float(args.lambda_scale_anchor),
            "risk_opacity": float(args.lambda_risk_opacity),
            "risk_scale": float(args.lambda_risk_scale),
        },
        "risk": {
            "mean": float(risk_weight.mean().item()),
            "p90": float(torch.quantile(risk_weight, 0.90).item()),
            "p99": float(torch.quantile(risk_weight, 0.99).item()),
            "max": float(risk_weight.max().item()),
            "config": {
                "radius_threshold": float(args.risk_radius_threshold),
                "lr_residual_threshold": float(args.risk_lr_residual_threshold),
                "anisotropy_threshold": float(args.risk_anisotropy_threshold),
                "min_support_views": int(args.risk_min_support_views),
            },
        },
        "before_rgb": before_rgb,
        "pre_cleanup_rgb": pre_cleanup_rgb,
        "after_rgb": after_rgb,
        "before_sr": before_sr,
        "pre_cleanup_sr": pre_cleanup_sr,
        "after_sr": after_sr,
        "cleanup": cleanup_summary,
        "displacement": stats_from_array(displacement.detach().cpu().numpy().astype(np.float32)),
        "log_tail": log_rows[-50:],
        "sr_prior_root": str(sr_prior_root),
    }
    summary_path = output_model_path / "global_refine_sof_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] output model: {output_model_path}")
    print(f"[done] output ply  : {point_dir / 'point_cloud.ply'}")
    print(f"[done] summary     : {summary_path}")


if __name__ == "__main__":
    main()
