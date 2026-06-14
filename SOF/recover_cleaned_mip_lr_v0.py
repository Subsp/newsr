from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from joint_judge_mip_sof_v0 import collect_render_metrics, static_gaussian_metrics, stats_from_array
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import (
    apply_gaussian_update_mask,
    build_prepared_sr_cache,
    build_scale_update_mask,
    build_dataset_args,
    clamp_xyz_displacement,
    compute_mip_closure_losses,
    compute_mip_closure_over_losses,
    compute_premul_hf_excess_loss,
    compute_prepared_sr_losses,
    copy_render_config,
    evaluate_mip_closure_losses,
    evaluate_mip_closure_over_losses,
    evaluate_premul_hf_excess_losses,
    evaluate_sr_prior_losses,
    load_gaussian_update_mask_payload,
    load_splat_settings,
    load_model_ply,
    load_train_cameras_only,
    render_mip_closure_cache,
    resolve_iteration,
    select_uniform,
    summarize_sr_cache,
)
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.system_utils import mkdir_p


def freeze_for_lr_recover(
    gaussians: GaussianModel,
    *,
    train_xyz: bool,
    train_opacity: bool,
    train_scale: bool,
    train_dc: bool,
    train_rest: bool,
) -> None:
    gaussians._xyz.requires_grad_(bool(train_xyz))
    gaussians._opacity.requires_grad_(bool(train_opacity))
    gaussians._scaling.requires_grad_(bool(train_scale))
    gaussians._rotation.requires_grad_(False)
    gaussians._features_dc.requires_grad_(bool(train_dc))
    gaussians._features_rest.requires_grad_(bool(train_rest))


def build_optimizer(
    gaussians: GaussianModel,
    *,
    train_xyz: bool,
    xyz_lr: float,
    opacity_lr: float,
    scale_lr: float,
    dc_lr: float,
    rest_lr: float,
    train_opacity: bool,
    train_scale: bool,
    train_dc: bool,
    train_rest: bool,
) -> torch.optim.Optimizer:
    params = []
    if bool(train_xyz):
        params.append({"params": [gaussians._xyz], "lr": float(xyz_lr), "name": "xyz"})
    if bool(train_opacity):
        params.append({"params": [gaussians._opacity], "lr": float(opacity_lr), "name": "opacity"})
    if bool(train_scale):
        params.append({"params": [gaussians._scaling], "lr": float(scale_lr), "name": "scale"})
    if bool(train_dc):
        params.append({"params": [gaussians._features_dc], "lr": float(dc_lr), "name": "features_dc"})
    if bool(train_rest):
        params.append({"params": [gaussians._features_rest], "lr": float(rest_lr), "name": "features_rest"})
    if not params:
        raise ValueError("Optimizer requires at least one trainable parameter group.")
    return torch.optim.Adam(params, eps=1e-15)


def camera_rgb(camera, device: torch.device) -> torch.Tensor:
    image = camera.original_image
    if image.ndim != 3:
        raise ValueError(f"camera.original_image must be 3D, got {tuple(image.shape)}")
    if image.shape[0] == 3:
        return image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported camera.original_image shape: {tuple(image.shape)}")


def mean_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a - b))


def _tau_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = torch.clamp(alpha, min=1e-6, max=1.0 - 1e-6)
    return -torch.log1p(-alpha)


def blur_chw_box(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = int(kernel_size)
    if kernel <= 1:
        return image
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    value = image.unsqueeze(0)
    value = F.pad(value, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(value, kernel_size=kernel, stride=1).squeeze(0)


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    mask = mask_hw.to(device=a.device, dtype=a.dtype).clamp(0.0, 1.0)
    denom = torch.clamp(mask.sum() * a.shape[0], min=1.0)
    return (torch.abs(a - b) * mask.unsqueeze(0)).sum() / denom


def grad_l2_norm(value: torch.Tensor) -> float:
    grad = value.grad
    if grad is None:
        return 0.0
    return float(torch.linalg.norm(grad.detach()).item())


def _load_optional_bool_mask(path: Path | None, total_count: int, *, device: torch.device) -> torch.Tensor:
    if path is None:
        return torch.zeros((int(total_count),), dtype=torch.bool, device=device)
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        tensor = obj.reshape(-1)
    elif isinstance(obj, dict):
        if "selected_mask" in obj:
            tensor = torch.as_tensor(obj["selected_mask"])
        elif "selected_ids" in obj:
            ids = torch.as_tensor(obj["selected_ids"]).reshape(-1).to(dtype=torch.int64)
            mask = torch.zeros((int(total_count),), dtype=torch.bool)
            if ids.numel() > 0:
                mask[ids] = True
            return mask.to(device=device)
        else:
            raise KeyError(f"Mask payload missing selected_mask/selected_ids: {path}")
    else:
        raise ValueError(f"Unsupported mask payload type for {path}: {type(obj)!r}")

    tensor = tensor.reshape(-1)
    if tensor.dtype == torch.bool:
        if int(tensor.shape[0]) != int(total_count):
            raise ValueError(f"Mask length mismatch for {path}: {tensor.shape[0]} vs {total_count}")
        return tensor.to(device=device, dtype=torch.bool)
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        ids = tensor.to(dtype=torch.int64)
        mask = torch.zeros((int(total_count),), dtype=torch.bool)
        if ids.numel() > 0:
            if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(total_count):
                raise ValueError(f"Mask ids out of range for {path}: n={total_count}")
            mask[ids] = True
        return mask.to(device=device)
    raise ValueError(f"Unsupported mask tensor dtype for {path}: {tensor.dtype}")


def mean_abs_delta(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(value.detach() - reference.detach())).item())


def combine_optional_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    active_masks = [mask for mask in masks if mask is not None]
    if not active_masks:
        return None
    combined = active_masks[0].to(device="cuda", dtype=torch.bool).reshape(-1)
    for mask in active_masks[1:]:
        current = mask.to(device="cuda", dtype=torch.bool).reshape(-1)
        if current.shape != combined.shape:
            raise ValueError(f"Gaussian mask length mismatch: {tuple(current.shape)} vs {tuple(combined.shape)}")
        combined = combined & current
    return combined


def _dilate_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask.to(dtype=torch.bool)
    value = mask.to(dtype=torch.float32)[None, None]
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(value, kernel_size=kernel, stride=1, padding=radius)
    return dilated[0, 0] > 0.5


def build_projected_gaussian_touch_mask(
    viewpoint_cam,
    gaussians: GaussianModel,
    image_mask: torch.Tensor,
    *,
    visibility_filter: torch.Tensor | None = None,
    radii: torch.Tensor | None = None,
    depth_min: float = 1e-6,
    min_touch_radius_px: float = 2.0,
    radius_scale: float = 0.5,
    max_touch_radius_px: float = 16.0,
) -> torch.Tensor:
    xyz = gaussians.get_xyz.detach()
    if xyz.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device="cuda")

    R = torch.as_tensor(viewpoint_cam.R, device=xyz.device, dtype=xyz.dtype)
    T = torch.as_tensor(viewpoint_cam.T, device=xyz.device, dtype=xyz.dtype)
    xyz_cam = xyz @ R + T.unsqueeze(0)
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / torch.clamp_min(z, 1e-6) * float(viewpoint_cam.focal_x) + float(viewpoint_cam.image_width) / 2.0
    y = xyz_cam[:, 1] / torch.clamp_min(z, 1e-6) * float(viewpoint_cam.focal_y) + float(viewpoint_cam.image_height) / 2.0

    valid = z > float(depth_min)
    if visibility_filter is not None:
        valid = valid & visibility_filter.to(device=xyz.device, dtype=torch.bool)

    xi = torch.round(x).to(dtype=torch.int64)
    yi = torch.round(y).to(dtype=torch.int64)
    valid = valid & (xi >= 0) & (xi < int(viewpoint_cam.image_width)) & (yi >= 0) & (yi < int(viewpoint_cam.image_height))

    touched = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
    if not torch.any(valid):
        return touched
    valid_idx = valid.nonzero(as_tuple=True)[0]
    image_mask_bool = image_mask.to(device=xyz.device, dtype=torch.bool)
    touched[valid_idx] = image_mask_bool[yi[valid_idx], xi[valid_idx]]

    if radii is None:
        touch_radius = int(round(float(min_touch_radius_px)))
        if touch_radius > 0:
            dilated_mask = _dilate_binary_mask(image_mask_bool, touch_radius)
            touched[valid_idx] |= dilated_mask[yi[valid_idx], xi[valid_idx]]
        return touched

    radii = radii.detach().to(device=xyz.device, dtype=torch.float32)
    valid_radii = torch.ceil(
        torch.clamp(
            radii[valid_idx] * float(radius_scale),
            min=float(min_touch_radius_px),
            max=float(max_touch_radius_px),
        )
    ).to(dtype=torch.int64)
    max_radius = int(valid_radii.max().item()) if valid_radii.numel() > 0 else 0
    for radius in range(1, max_radius + 1):
        bucket = valid_radii == radius
        if not torch.any(bucket):
            continue
        dilated_mask = _dilate_binary_mask(image_mask_bool, radius)
        bucket_idx = valid_idx[bucket]
        touched[bucket_idx] |= dilated_mask[yi[bucket_idx], xi[bucket_idx]]
    return touched


@torch.no_grad()
def build_sr_touch_prefilter(
    cameras: Sequence[object],
    sr_cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    base_mask: torch.Tensor | None,
    view_limit: int,
    min_touch_views: int,
    min_visible_views: int,
    min_touch_ratio: float,
    min_candidate_opacity: float,
    radius_scale: float,
    min_touch_radius_px: float,
    max_touch_radius_px: float,
) -> Dict[str, object]:
    total = int(student.get_xyz.shape[0])
    candidate_mask = (
        torch.ones((total,), dtype=torch.bool, device="cuda")
        if base_mask is None
        else base_mask.to(device="cuda", dtype=torch.bool).reshape(-1)
    )
    visible_view_count = torch.zeros((total,), dtype=torch.int32, device="cuda")
    touch_view_count = torch.zeros((total,), dtype=torch.int32, device="cuda")
    pairs = list(zip(cameras, sr_cache))
    if int(view_limit) > 0:
        pairs = pairs[: int(view_limit)]

    matched_view_names: List[str] = []
    skipped_missing_mask = 0
    for camera, target in tqdm(pairs, desc="prefilter sr touch"):
        prior_mask = target.get("prior_mask")
        if not torch.is_tensor(prior_mask):
            skipped_missing_mask += 1
            continue
        prior_mask = prior_mask.to(device="cuda", dtype=torch.float32)
        if float(prior_mask.sum().item()) <= 0.0:
            skipped_missing_mask += 1
            continue
        render_pkg = render_simple(camera, student, background)
        visible = render_pkg["visibility_filter"].detach().to(device="cuda", dtype=torch.bool).reshape(-1)
        radii = render_pkg["radii"].detach().to(device="cuda", dtype=torch.float32).reshape(-1)
        visible_view_count += visible.to(dtype=torch.int32)
        touched = build_projected_gaussian_touch_mask(
            camera,
            student,
            prior_mask > 0.0,
            visibility_filter=visible,
            radii=radii,
            min_touch_radius_px=float(min_touch_radius_px),
            radius_scale=float(radius_scale),
            max_touch_radius_px=float(max_touch_radius_px),
        )
        touch_view_count += touched.to(dtype=torch.int32)
        matched_view_names.append(str(camera.image_name))

    opacity = student.get_opacity.detach().reshape(-1)
    visible_float = torch.clamp(visible_view_count.to(dtype=torch.float32), min=1.0)
    touch_ratio = touch_view_count.to(dtype=torch.float32) / visible_float
    selected_mask = (
        candidate_mask
        & (visible_view_count >= int(min_visible_views))
        & (touch_view_count >= int(min_touch_views))
        & (touch_ratio >= float(min_touch_ratio))
        & (opacity >= float(min_candidate_opacity))
    )
    selected_ids = torch.nonzero(selected_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    summary = {
        "mode": "sr_touch_v0",
        "candidate_count": int(candidate_mask.sum().item()),
        "selected_count": int(selected_ids.numel()),
        "selected_ratio": float(selected_ids.numel() / max(int(candidate_mask.sum().item()), 1)),
        "matched_view_count": int(len(matched_view_names)),
        "skipped_missing_mask_views": int(skipped_missing_mask),
        "matched_views": matched_view_names,
        "config": {
            "view_limit": int(view_limit),
            "min_touch_views": int(min_touch_views),
            "min_visible_views": int(min_visible_views),
            "min_touch_ratio": float(min_touch_ratio),
            "min_candidate_opacity": float(min_candidate_opacity),
            "radius_scale": float(radius_scale),
            "min_touch_radius_px": float(min_touch_radius_px),
            "max_touch_radius_px": float(max_touch_radius_px),
        },
        "visible_view_count": stats_from_array(visible_view_count.detach().cpu().numpy().astype(np.float32)),
        "touch_view_count": stats_from_array(touch_view_count.detach().cpu().numpy().astype(np.float32)),
        "touch_ratio": stats_from_array(touch_ratio.detach().cpu().numpy().astype(np.float32)),
        "selected_visible_view_count": stats_from_array(
            visible_view_count[selected_mask].detach().cpu().numpy().astype(np.float32)
        ),
        "selected_touch_view_count": stats_from_array(
            touch_view_count[selected_mask].detach().cpu().numpy().astype(np.float32)
        ),
        "selected_touch_ratio": stats_from_array(
            touch_ratio[selected_mask].detach().cpu().numpy().astype(np.float32)
        ),
    }
    return {
        "selected_mask": selected_mask,
        "selected_ids": selected_ids,
        "visible_view_count": visible_view_count,
        "touch_view_count": touch_view_count,
        "touch_ratio": touch_ratio,
        "summary": summary,
    }


@torch.no_grad()
def cache_anchor_renders(
    cameras: Sequence[object],
    anchor_model: GaussianModel,
    background: torch.Tensor,
) -> List[torch.Tensor]:
    cache: List[torch.Tensor] = []
    for camera in tqdm(cameras, desc="cache anchor renders"):
        cache.append(render_simple(camera, anchor_model, background)["render"].detach().clamp(0.0, 1.0).cpu())
    return cache


@torch.no_grad()
def evaluate_rgb_losses(
    cameras: Sequence[object],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    anchor_cache: Sequence[torch.Tensor] | None = None,
) -> Dict[str, float]:
    lr_losses = []
    anchor_losses = []
    for idx, camera in enumerate(cameras):
        rgb = render_simple(camera, student, background)["render"].detach().clamp(0.0, 1.0)
        lr_losses.append(float(mean_l1(rgb, camera_rgb(camera, background.device).to(dtype=rgb.dtype)).item()))
        if anchor_cache is not None:
            anchor_rgb = anchor_cache[idx].to(device=rgb.device, dtype=rgb.dtype)
            anchor_losses.append(float(mean_l1(rgb, anchor_rgb).item()))
    return {
        "lr_rgb": float(np.mean(lr_losses)) if lr_losses else 0.0,
        "anchor_rgb": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
    }


def sr_target_with_anchor(
    target: Dict[str, torch.Tensor | str | float | None],
    anchor_rgb: torch.Tensor,
) -> Dict[str, torch.Tensor | str | float | None]:
    patched = dict(target)
    patched["anchor_rgb"] = anchor_rgb.detach().cpu()
    return patched


def compute_mip_hr_lowfreq_loss(
    rgb: torch.Tensor,
    mip_anchor_rgb: torch.Tensor,
    target: Dict[str, torch.Tensor | str | float | None],
    *,
    kernel_size: int,
    consistency_threshold: float,
    mask_floor: float,
) -> torch.Tensor | None:
    prior_mask = target.get("prior_mask")
    reference_rgb = target.get("anchor_rgb")
    if not torch.is_tensor(prior_mask) or not torch.is_tensor(reference_rgb):
        return None

    mip_anchor_rgb = mip_anchor_rgb.to(device=rgb.device, dtype=rgb.dtype)
    prior_mask = prior_mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
    reference_rgb = reference_rgb.to(device=rgb.device, dtype=rgb.dtype)

    rgb_low = blur_chw_box(rgb, int(kernel_size))
    mip_low = blur_chw_box(mip_anchor_rgb, int(kernel_size))
    ref_low = blur_chw_box(reference_rgb, int(kernel_size))
    mask = prior_mask
    if float(consistency_threshold) > 0.0:
        consistency = torch.mean(torch.abs(mip_low - ref_low), dim=0)
        mask = mask * (consistency <= float(consistency_threshold)).to(dtype=mask.dtype)
    if float(mask_floor) > 0.0:
        mask = mask.clamp(min=float(mask_floor), max=1.0)
    if float(mask.sum().detach().item()) < 1.0:
        return None
    return masked_l1(rgb_low, mip_low, mask)


@torch.no_grad()
def evaluate_mip_hr_anchor_losses(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    mip_anchor_cache: Sequence[torch.Tensor],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    residual_anchor_mode: str,
    min_pixels: float,
    min_valid_ratio: float,
    prior_delta_clip: float,
    disable_hf_residual: bool,
    lowfreq_kernel: int,
    lowfreq_consistency_threshold: float,
    lowfreq_mask_floor: float,
) -> Dict[str, float]:
    sr_l1_losses = []
    sr_hf_losses = []
    mip_lowfreq_losses = []
    for camera, target, mip_anchor_rgb in zip(cameras, cache, mip_anchor_cache):
        rgb = render_simple(camera, student, background)["render"].detach().clamp(0.0, 1.0)
        loss_target = target
        if str(residual_anchor_mode) == "mip_hr":
            loss_target = sr_target_with_anchor(target, mip_anchor_rgb)
        loss_l1, loss_hf = compute_prepared_sr_losses(
            rgb,
            loss_target,
            min_pixels=min_pixels,
            min_valid_ratio=min_valid_ratio,
            prior_delta_clip=prior_delta_clip,
            disable_hf_residual=disable_hf_residual,
        )
        if loss_l1 is not None:
            sr_l1_losses.append(float(loss_l1.item()))
        if loss_hf is not None:
            sr_hf_losses.append(float(loss_hf.item()))
        loss_mip = compute_mip_hr_lowfreq_loss(
            rgb,
            mip_anchor_rgb,
            target,
            kernel_size=int(lowfreq_kernel),
            consistency_threshold=float(lowfreq_consistency_threshold),
            mask_floor=float(lowfreq_mask_floor),
        )
        if loss_mip is not None:
            mip_lowfreq_losses.append(float(loss_mip.item()))
    return {
        "sr_l1": float(np.mean(sr_l1_losses)) if sr_l1_losses else 0.0,
        "sr_hf": float(np.mean(sr_hf_losses)) if sr_hf_losses else 0.0,
        "mip_hr_lowfreq": float(np.mean(mip_lowfreq_losses)) if mip_lowfreq_losses else 0.0,
    }


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


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probability / torch.clamp(1.0 - probability, min=1e-6))


def _opacity_raw_with_tau_scale(original_raw: torch.Tensor, tau_scale: float, min_alpha: float) -> torch.Tensor:
    alpha = torch.sigmoid(original_raw)
    tau = -torch.log(torch.clamp(1.0 - alpha, min=1e-6))
    alpha_eff = 1.0 - torch.exp(-torch.clamp(tau * float(tau_scale), min=0.0))
    alpha_eff = torch.clamp(alpha_eff, min=float(min_alpha), max=1.0 - 1e-6)
    return _logit(alpha_eff)


def _release_fraction(iteration: int, start_iter: int, end_iter: int, mode: str) -> float:
    if int(end_iter) <= int(start_iter):
        return 1.0 if int(iteration) >= int(start_iter) else 0.0
    t = (float(iteration) - float(start_iter)) / max(float(end_iter - start_iter), 1.0)
    t = float(np.clip(t, 0.0, 1.0))
    if str(mode) == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    return t


def scheduled_sr_prior_scale(
    iteration: int,
    *,
    start_iter: int,
    end_iter: int,
    start_scale: float,
    end_scale: float,
    update_scale: float,
    mode: str,
) -> float:
    release = _release_fraction(int(iteration), int(start_iter), int(end_iter), str(mode))
    scheduled = float(start_scale) + release * (float(end_scale) - float(start_scale))
    return float(update_scale) * scheduled


def phase_train_flags(args, iteration: int) -> tuple[str, Dict[str, bool]]:
    mode = str(args.phase_mode).strip().lower()
    appearance_flags = {
        "train_xyz": False,
        "train_opacity": False,
        "train_scale": False,
        "train_dc": bool(args.enable_dc_update),
        "train_rest": bool(args.enable_rest_update),
    }
    geometry_flags = {
        "train_xyz": True,
        "train_opacity": bool(args.enable_opacity_update),
        "train_scale": bool(args.enable_scale_update),
        "train_dc": False,
        "train_rest": False,
    }
    if mode == "joint":
        return "joint", {
            "train_xyz": True,
            "train_opacity": bool(args.enable_opacity_update),
            "train_scale": bool(args.enable_scale_update),
            "train_dc": bool(args.enable_dc_update),
            "train_rest": bool(args.enable_rest_update),
        }
    if mode == "appearance_only":
        return "appearance", appearance_flags
    if mode == "geometry_only":
        return "geometry", geometry_flags
    if mode == "alternating":
        block = max(int(args.phase_block_steps), 1)
        phase_index = ((int(iteration) - 1) // block) % 2
        if phase_index == 0:
            return "appearance", appearance_flags
        return "geometry", geometry_flags
    raise ValueError(f"Unsupported phase_mode: {args.phase_mode!r}")


def phase_union_train_flags(args) -> Dict[str, bool]:
    mode = str(args.phase_mode).strip().lower()
    if mode == "joint":
        _, flags = phase_train_flags(args, 1)
        return flags
    if mode == "appearance_only":
        _, flags = phase_train_flags(args, 1)
        return flags
    if mode == "geometry_only":
        _, flags = phase_train_flags(args, 1)
        return flags
    if mode == "alternating":
        return {
            "train_xyz": True,
            "train_opacity": bool(args.enable_opacity_update),
            "train_scale": bool(args.enable_scale_update),
            "train_dc": bool(args.enable_dc_update),
            "train_rest": bool(args.enable_rest_update),
        }
    raise ValueError(f"Unsupported phase_mode: {args.phase_mode!r}")


def load_reparam_settle_state(
    *,
    student: GaussianModel,
    output_source_idx_path: Path | None,
    parent_mask_path: Path | None,
    child_mask_path: Path | None,
    geometry_parent_mask_path: Path | None,
) -> Dict[str, object] | None:
    if (
        output_source_idx_path is None
        and parent_mask_path is None
        and child_mask_path is None
        and geometry_parent_mask_path is None
    ):
        return None
    if output_source_idx_path is None:
        raise ValueError("Reparameterization settle controls require --reparam_output_source_idx_path")

    output_source_idx = torch.as_tensor(torch.load(output_source_idx_path, map_location="cpu")).reshape(-1)
    output_source_idx = output_source_idx.to(dtype=torch.int64, device=student._xyz.device)
    total_count = int(student.get_xyz.shape[0])
    if int(output_source_idx.shape[0]) != total_count:
        raise ValueError(
            f"Reparameterization lineage length mismatch: {output_source_idx.shape[0]} vs model n={total_count}"
        )

    parent_mask = _load_optional_bool_mask(parent_mask_path, total_count, device=student._xyz.device)
    child_mask = _load_optional_bool_mask(child_mask_path, total_count, device=student._xyz.device)
    geometry_parent_mask = _load_optional_bool_mask(geometry_parent_mask_path, total_count, device=student._xyz.device)
    active_mask = parent_mask | child_mask
    start_tau = _tau_from_alpha(student.get_opacity.detach().reshape(-1))

    selected_ids = torch.nonzero(active_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    if selected_ids.numel() == 0:
        return {
            "enabled": False,
            "output_source_idx_path": str(output_source_idx_path),
            "parent_mask_path": str(parent_mask_path) if parent_mask_path is not None else None,
            "child_mask_path": str(child_mask_path) if child_mask_path is not None else None,
            "geometry_parent_mask_path": str(geometry_parent_mask_path) if geometry_parent_mask_path is not None else None,
            "parent_count": int(parent_mask.sum().item()),
            "child_count": int(child_mask.sum().item()),
            "geometry_parent_count": int(geometry_parent_mask.sum().item()),
            "group_count": 0,
        }

    group_source_idx, selected_group_index = torch.unique(
        output_source_idx[selected_ids],
        sorted=True,
        return_inverse=True,
    )
    group_start_tau = torch.zeros(
        (int(group_source_idx.shape[0]),),
        dtype=start_tau.dtype,
        device=start_tau.device,
    )
    group_start_tau.scatter_add_(0, selected_group_index, start_tau[selected_ids])

    child_ids = torch.nonzero(child_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    geometry_parent_ids = torch.nonzero(geometry_parent_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    return {
        "enabled": True,
        "output_source_idx_path": str(output_source_idx_path),
        "output_source_idx": output_source_idx,
        "parent_mask_path": str(parent_mask_path) if parent_mask_path is not None else None,
        "child_mask_path": str(child_mask_path) if child_mask_path is not None else None,
        "geometry_parent_mask_path": str(geometry_parent_mask_path) if geometry_parent_mask_path is not None else None,
        "selected_ids": selected_ids,
        "selected_start_tau": start_tau[selected_ids],
        "selected_group_index": selected_group_index,
        "group_source_idx": group_source_idx,
        "group_start_tau": group_start_tau,
        "child_ids": child_ids,
        "child_start_tau": start_tau[child_ids] if child_ids.numel() > 0 else torch.empty((0,), dtype=start_tau.dtype, device=start_tau.device),
        "geometry_parent_ids": geometry_parent_ids,
        "geometry_parent_start_tau": (
            start_tau[geometry_parent_ids]
            if geometry_parent_ids.numel() > 0
            else torch.empty((0,), dtype=start_tau.dtype, device=start_tau.device)
        ),
        "parent_count": int(parent_mask.sum().item()),
        "child_count": int(child_mask.sum().item()),
        "geometry_parent_count": int(geometry_parent_mask.sum().item()),
        "group_count": int(group_source_idx.shape[0]),
    }


def compute_reparam_mass_cap_loss(
    student: GaussianModel,
    state: Dict[str, object] | None,
    *,
    eps: float,
) -> tuple[torch.Tensor, Dict[str, float | int]]:
    zero = torch.zeros((), dtype=torch.float32, device=student._xyz.device)
    if state is None or not bool(state.get("enabled", False)):
        return zero, {"over_groups": 0, "overflow_mean": 0.0, "overflow_max": 0.0}

    selected_ids = state["selected_ids"]
    if int(selected_ids.numel()) == 0:
        return zero, {"over_groups": 0, "overflow_mean": 0.0, "overflow_max": 0.0}
    current_tau = _tau_from_alpha(student.get_opacity.reshape(-1))
    group_current_tau = torch.zeros_like(state["group_start_tau"])
    group_current_tau.scatter_add_(0, state["selected_group_index"], current_tau[selected_ids])
    group_limit_tau = state["group_start_tau"] * (1.0 + max(float(eps), 0.0))
    overflow = torch.relu(group_current_tau - group_limit_tau)
    loss = torch.mean(overflow * overflow)
    return loss, {
        "over_groups": int(torch.count_nonzero(overflow > 0).item()),
        "overflow_mean": float(overflow.mean().detach().item()) if overflow.numel() > 0 else 0.0,
        "overflow_max": float(overflow.max().detach().item()) if overflow.numel() > 0 else 0.0,
    }


def compute_reparam_child_tau_cap_loss(
    student: GaussianModel,
    state: Dict[str, object] | None,
    *,
    tau_scale: float,
    tau_floor: float,
) -> tuple[torch.Tensor, Dict[str, float | int]]:
    zero = torch.zeros((), dtype=torch.float32, device=student._xyz.device)
    if state is None or not bool(state.get("enabled", False)):
        return zero, {"over_children": 0, "overflow_mean": 0.0, "overflow_max": 0.0}

    child_ids = state["child_ids"]
    if int(child_ids.numel()) == 0:
        return zero, {"over_children": 0, "overflow_mean": 0.0, "overflow_max": 0.0}
    current_tau = _tau_from_alpha(student.get_opacity.reshape(-1))[child_ids]
    start_tau = state["child_start_tau"]
    child_cap_tau = torch.clamp(start_tau * max(float(tau_scale), 0.0), min=max(float(tau_floor), 0.0))
    overflow = torch.relu(current_tau - child_cap_tau)
    loss = torch.mean(overflow * overflow)
    return loss, {
        "over_children": int(torch.count_nonzero(overflow > 0).item()),
        "overflow_mean": float(overflow.mean().detach().item()) if overflow.numel() > 0 else 0.0,
        "overflow_max": float(overflow.max().detach().item()) if overflow.numel() > 0 else 0.0,
    }


def compute_geometry_parent_tau_brake_loss(
    student: GaussianModel,
    state: Dict[str, object] | None,
    *,
    tau_scale: float,
) -> tuple[torch.Tensor, Dict[str, float | int]]:
    zero = torch.zeros((), dtype=torch.float32, device=student._xyz.device)
    if state is None or not bool(state.get("enabled", False)):
        return zero, {"over_parents": 0, "overflow_mean": 0.0, "overflow_max": 0.0}

    parent_ids = state["geometry_parent_ids"]
    if int(parent_ids.numel()) == 0:
        return zero, {"over_parents": 0, "overflow_mean": 0.0, "overflow_max": 0.0}
    current_tau = _tau_from_alpha(student.get_opacity.reshape(-1))[parent_ids]
    start_tau = state["geometry_parent_start_tau"]
    target_tau = start_tau * max(float(tau_scale), 0.0)
    overflow = torch.relu(current_tau - target_tau)
    loss = torch.mean(overflow * overflow)
    return loss, {
        "over_parents": int(torch.count_nonzero(overflow > 0).item()),
        "overflow_mean": float(overflow.mean().detach().item()) if overflow.numel() > 0 else 0.0,
        "overflow_max": float(overflow.max().detach().item()) if overflow.numel() > 0 else 0.0,
    }


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for chunk in str(text).split(","):
        chunk = str(chunk).strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


@torch.no_grad()
def prune_student_with_optimizer(
    student: GaussianModel,
    optimizer: torch.optim.Optimizer,
    prune_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prune_mask = prune_mask.reshape(-1).to(device=student._xyz.device, dtype=torch.bool)
    total_count = int(student.get_xyz.shape[0])
    if int(prune_mask.shape[0]) != total_count:
        raise ValueError(f"prune mask length mismatch: {prune_mask.shape[0]} vs model n={total_count}")
    prune_count = int(prune_mask.sum().item())
    valid_mask = ~prune_mask
    old_to_new = torch.full((total_count,), -1, dtype=torch.int64, device=student._xyz.device)
    if prune_count <= 0:
        old_to_new[valid_mask] = torch.arange(total_count, device=student._xyz.device, dtype=torch.int64)
        return valid_mask, old_to_new, 0

    old_to_new[valid_mask] = torch.arange(int(valid_mask.sum().item()), device=student._xyz.device, dtype=torch.int64)

    new_optimizer_state: Dict[torch.nn.Parameter, Dict[str, object]] = {}
    group_params: Dict[str, torch.nn.Parameter] = {}
    for group in optimizer.param_groups:
        old_param = group["params"][0]
        new_data = old_param.detach()[valid_mask].clone()
        new_param = torch.nn.Parameter(new_data.requires_grad_(old_param.requires_grad))
        stored_state = optimizer.state.get(old_param, None)
        if stored_state is not None:
            next_state: Dict[str, object] = {}
            for key, value in stored_state.items():
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total_count:
                    next_state[key] = value[valid_mask].clone()
                elif torch.is_tensor(value):
                    next_state[key] = value.clone()
                else:
                    next_state[key] = value
            new_optimizer_state[new_param] = next_state
        group["params"][0] = new_param
        group_params[str(group.get("name", ""))] = new_param
    optimizer.state = new_optimizer_state

    def _slice_param(param: torch.Tensor, *, requires_grad: bool) -> torch.nn.Parameter:
        return torch.nn.Parameter(param.detach()[valid_mask].clone().requires_grad_(requires_grad))

    student._xyz = group_params.get("xyz", _slice_param(student._xyz, requires_grad=student._xyz.requires_grad))
    student._opacity = group_params.get("opacity", _slice_param(student._opacity, requires_grad=student._opacity.requires_grad))
    student._scaling = group_params.get("scale", _slice_param(student._scaling, requires_grad=student._scaling.requires_grad))
    student._features_dc = group_params.get("features_dc", _slice_param(student._features_dc, requires_grad=student._features_dc.requires_grad))
    student._features_rest = group_params.get("features_rest", _slice_param(student._features_rest, requires_grad=student._features_rest.requires_grad))
    student._rotation = _slice_param(student._rotation, requires_grad=False)

    if hasattr(student, "_source_tag") and int(student._source_tag.shape[0]) == total_count:
        student._source_tag = student._source_tag[valid_mask]
    if hasattr(student, "_seed_id") and int(student._seed_id.shape[0]) == total_count:
        student._seed_id = student._seed_id[valid_mask]
    if hasattr(student, "_generation") and int(student._generation.shape[0]) == total_count:
        student._generation = student._generation[valid_mask]
    if hasattr(student, "_edge_touched") and int(student._edge_touched.shape[0]) == total_count:
        student._edge_touched = student._edge_touched[valid_mask]
    if hasattr(student, "_edge_touch_iter") and int(student._edge_touch_iter.shape[0]) == total_count:
        student._edge_touch_iter = student._edge_touch_iter[valid_mask]
    if isinstance(student.filter_3D, torch.Tensor) and int(student.filter_3D.shape[0]) == total_count:
        student.filter_3D = student.filter_3D[valid_mask]
    if isinstance(student.max_radii2D, torch.Tensor) and int(student.max_radii2D.shape[0]) == total_count:
        student.max_radii2D = student.max_radii2D[valid_mask]
    if isinstance(student.xyz_gradient_accum, torch.Tensor) and int(student.xyz_gradient_accum.shape[0]) == total_count:
        student.xyz_gradient_accum = student.xyz_gradient_accum[valid_mask]
    if isinstance(student.xyz_gradient_accum_abs, torch.Tensor) and int(student.xyz_gradient_accum_abs.shape[0]) == total_count:
        student.xyz_gradient_accum_abs = student.xyz_gradient_accum_abs[valid_mask]
    if isinstance(student.xyz_gradient_accum_abs_max, torch.Tensor) and int(student.xyz_gradient_accum_abs_max.shape[0]) == total_count:
        student.xyz_gradient_accum_abs_max = student.xyz_gradient_accum_abs_max[valid_mask]
    if isinstance(student.denom, torch.Tensor) and int(student.denom.shape[0]) == total_count:
        student.denom = student.denom[valid_mask]

    return valid_mask, old_to_new, prune_count


def remap_ids_after_prune(ids: torch.Tensor, valid_mask: torch.Tensor, old_to_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(ids.numel()) <= 0:
        empty_long = torch.empty((0,), dtype=torch.int64, device=valid_mask.device)
        empty_bool = torch.empty((0,), dtype=torch.bool, device=valid_mask.device)
        return empty_long, empty_bool
    keep = valid_mask[ids]
    return old_to_new[ids[keep]], keep


def update_reparam_state_after_prune(
    state: Dict[str, object] | None,
    valid_mask: torch.Tensor,
    old_to_new: torch.Tensor,
) -> Dict[str, object] | None:
    if state is None:
        return None
    output_source_idx = state["output_source_idx"][valid_mask]
    selected_ids, keep_selected = remap_ids_after_prune(state["selected_ids"], valid_mask, old_to_new)
    child_ids, keep_children = remap_ids_after_prune(state["child_ids"], valid_mask, old_to_new)
    geometry_parent_ids, keep_geometry = remap_ids_after_prune(state["geometry_parent_ids"], valid_mask, old_to_new)

    state["output_source_idx"] = output_source_idx
    state["selected_ids"] = selected_ids
    state["selected_start_tau"] = state["selected_start_tau"][keep_selected]
    state["child_ids"] = child_ids
    state["child_start_tau"] = state["child_start_tau"][keep_children]
    state["geometry_parent_ids"] = geometry_parent_ids
    state["geometry_parent_start_tau"] = state["geometry_parent_start_tau"][keep_geometry]
    state["parent_count"] = int(selected_ids.numel() - child_ids.numel())
    state["child_count"] = int(child_ids.numel())
    state["geometry_parent_count"] = int(geometry_parent_ids.numel())
    if int(selected_ids.numel()) <= 0:
        state["enabled"] = False
        state["group_count"] = 0
        state["selected_group_index"] = torch.empty((0,), dtype=torch.int64, device=valid_mask.device)
        state["group_source_idx"] = torch.empty((0,), dtype=torch.int64, device=valid_mask.device)
        state["group_start_tau"] = torch.empty((0,), dtype=state["selected_start_tau"].dtype, device=valid_mask.device)
        return state

    group_source_idx, selected_group_index = torch.unique(
        output_source_idx[selected_ids],
        sorted=True,
        return_inverse=True,
    )
    group_start_tau = torch.zeros(
        (int(group_source_idx.shape[0]),),
        dtype=state["selected_start_tau"].dtype,
        device=valid_mask.device,
    )
    group_start_tau.scatter_add_(0, selected_group_index, state["selected_start_tau"])
    state["enabled"] = True
    state["selected_group_index"] = selected_group_index
    state["group_source_idx"] = group_source_idx
    state["group_start_tau"] = group_start_tau
    state["group_count"] = int(group_source_idx.shape[0])
    return state


def update_star_release_state_after_prune(
    state: Dict[str, object] | None,
    valid_mask: torch.Tensor,
    old_to_new: torch.Tensor,
) -> Dict[str, object] | None:
    if state is None:
        return None
    selected_ids, keep = remap_ids_after_prune(state["selected_ids"], valid_mask, old_to_new)
    state["selected_ids"] = selected_ids
    state["selected_count"] = int(selected_ids.numel())
    if "original_features_rest" in state:
        state["original_features_rest"] = state["original_features_rest"][keep]
    if "original_opacity" in state:
        state["original_opacity"] = state["original_opacity"][keep]
    if int(selected_ids.numel()) <= 0:
        state["enabled"] = False
    return state


def compute_reparam_prune_mask(
    student: GaussianModel,
    state: Dict[str, object] | None,
    *,
    risk_weight: torch.Tensor,
    child_dead_tau: float,
    child_spike_tau_scale: float,
    child_spike_tau_abs: float,
    child_spike_anisotropy: float,
    child_spike_risk_min: float,
) -> tuple[torch.Tensor, Dict[str, float | int]]:
    count = int(student.get_xyz.shape[0])
    prune_mask = torch.zeros((count,), dtype=torch.bool, device=student._xyz.device)
    if state is None or not bool(state.get("enabled", False)):
        return prune_mask, {"pruned": 0, "child_dead": 0, "child_spike": 0, "child_total": 0}

    child_ids = state["child_ids"]
    if int(child_ids.numel()) <= 0:
        return prune_mask, {"pruned": 0, "child_dead": 0, "child_spike": 0, "child_total": 0}

    current_tau = _tau_from_alpha(student.get_opacity.reshape(-1))[child_ids]
    current_scale = student.get_scaling[child_ids]
    current_scale_max = torch.max(current_scale, dim=1).values
    current_scale_min = torch.clamp(torch.min(current_scale, dim=1).values, min=1e-8)
    current_anisotropy = current_scale_max / current_scale_min
    child_risk = risk_weight[child_ids]

    dead_mask = current_tau <= max(float(child_dead_tau), 0.0)
    spike_tau_cap = torch.clamp(
        state["child_start_tau"] * max(float(child_spike_tau_scale), 0.0),
        min=max(float(child_spike_tau_abs), 0.0),
    )
    spike_mask = (
        (current_tau >= spike_tau_cap)
        & (current_anisotropy >= max(float(child_spike_anisotropy), 1.0))
        & (child_risk >= max(float(child_spike_risk_min), 0.0))
    )
    selected = dead_mask | spike_mask
    if int(selected.numel()) > 0:
        prune_mask[child_ids[selected]] = True
    return prune_mask, {
        "pruned": int(prune_mask.sum().item()),
        "child_dead": int(dead_mask.sum().item()),
        "child_spike": int(spike_mask.sum().item()),
        "child_total": int(child_ids.numel()),
    }


def load_star_release_state(
    payload_path: Path | None,
    student: GaussianModel,
    *,
    rest_start_scale: float,
    rest_end_scale: float,
    tau_start_scale: float,
    tau_end_scale: float,
    release_start_iter: int,
    release_end_iter: int,
    release_mode: str,
    release_opacity: bool,
    min_alpha: float,
) -> Dict[str, object] | None:
    if payload_path is None:
        return None
    payload = torch.load(payload_path, map_location="cpu")
    if str(payload.get("version", "")) != "star_quarantine_v0":
        raise ValueError(f"Unsupported star quarantine payload version in {payload_path}")

    selected_ids = torch.as_tensor(payload.get("selected_ids", torch.empty((0,), dtype=torch.int64))).reshape(-1).long()
    if selected_ids.numel() == 0:
        return {
            "payload_path": str(payload_path),
            "selected_ids": selected_ids.to(device=student._features_rest.device),
            "selected_count": 0,
            "enabled": False,
        }
    count = int(student.get_xyz.shape[0])
    if int(selected_ids.min().item()) < 0 or int(selected_ids.max().item()) >= count:
        raise ValueError(f"Star quarantine selected ids are out of range for model n={count}: {payload_path}")
    original_features_rest = torch.as_tensor(payload["original_features_rest"]).to(
        device=student._features_rest.device,
        dtype=student._features_rest.dtype,
    )
    original_opacity = torch.as_tensor(payload["original_opacity"]).to(
        device=student._opacity.device,
        dtype=student._opacity.dtype,
    )
    if int(original_features_rest.shape[0]) != int(selected_ids.numel()):
        raise ValueError("Star quarantine payload original_features_rest length mismatch")
    if int(original_opacity.shape[0]) != int(selected_ids.numel()):
        raise ValueError("Star quarantine payload original_opacity length mismatch")

    payload_args = payload.get("args", {})
    payload_rest_scale = float(payload_args.get("rest_scale", 0.10)) if isinstance(payload_args, dict) else 0.10
    payload_tau_scale = float(payload_args.get("tau_scale", 0.35)) if isinstance(payload_args, dict) else 0.35
    resolved_rest_start = payload_rest_scale if float(rest_start_scale) < 0.0 else float(rest_start_scale)
    resolved_tau_start = payload_tau_scale if float(tau_start_scale) < 0.0 else float(tau_start_scale)
    return {
        "payload_path": str(payload_path),
        "selected_ids": selected_ids.to(device=student._features_rest.device),
        "selected_count": int(selected_ids.numel()),
        "original_features_rest": original_features_rest,
        "original_opacity": original_opacity,
        "rest_start_scale": float(resolved_rest_start),
        "rest_end_scale": float(rest_end_scale),
        "tau_start_scale": float(resolved_tau_start),
        "tau_end_scale": float(tau_end_scale),
        "release_start_iter": int(release_start_iter),
        "release_end_iter": int(release_end_iter),
        "release_mode": str(release_mode),
        "release_opacity": bool(release_opacity),
        "min_alpha": float(min_alpha),
        "enabled": True,
    }


@torch.no_grad()
def apply_star_release(student: GaussianModel, state: Dict[str, object] | None, iteration: int) -> Dict[str, float | int]:
    if state is None or not bool(state.get("enabled", False)):
        return {"release": 1.0, "rest_scale": 1.0, "tau_scale": 1.0, "selected_count": 0}
    selected_ids = state["selected_ids"]
    release = _release_fraction(
        int(iteration),
        int(state["release_start_iter"]),
        int(state["release_end_iter"]),
        str(state["release_mode"]),
    )
    rest_scale = float(state["rest_start_scale"]) + release * (
        float(state["rest_end_scale"]) - float(state["rest_start_scale"])
    )
    tau_scale = float(state["tau_start_scale"]) + release * (
        float(state["tau_end_scale"]) - float(state["tau_start_scale"])
    )
    student._features_rest[selected_ids] = state["original_features_rest"] * float(rest_scale)
    if bool(state.get("release_opacity", False)):
        student._opacity[selected_ids] = _opacity_raw_with_tau_scale(
            state["original_opacity"],
            tau_scale=float(tau_scale),
            min_alpha=float(state["min_alpha"]),
        )
    return {
        "release": float(release),
        "rest_scale": float(rest_scale),
        "tau_scale": float(tau_scale),
        "selected_count": int(state["selected_count"]),
    }


def main() -> None:
    parser = ArgumentParser(description="Recover LR render quality after mip cleanup while anchoring risky GS.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--start_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--anchor_model_path", default="")
    parser.add_argument("--start_iteration", type=int, default=30000)
    parser.add_argument("--anchor_iteration", type=int, default=30000)
    parser.add_argument("--output_iteration", type=int, default=-1)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--xyz_lr", type=float, default=5e-6)
    parser.add_argument("--opacity_lr", type=float, default=5e-4)
    parser.add_argument("--scale_lr", type=float, default=1e-4)
    parser.add_argument("--dc_lr", type=float, default=0.0)
    parser.add_argument("--rest_lr", type=float, default=0.0)
    parser.add_argument("--lambda_lr_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_anchor_rgb", type=float, default=0.0)
    parser.add_argument("--lambda_xyz_anchor", type=float, default=45.0)
    parser.add_argument("--lambda_opacity_anchor", type=float, default=0.05)
    parser.add_argument("--lambda_scale_anchor", type=float, default=0.18)
    parser.add_argument("--lambda_dc_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_rest_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_risk_opacity", type=float, default=0.035)
    parser.add_argument("--lambda_risk_scale", type=float, default=0.005)
    parser.add_argument("--lambda_risk_rest", type=float, default=0.0)
    parser.add_argument("--lambda_sr_risk_opacity", type=float, default=0.0)
    parser.add_argument("--lambda_sr_risk_scale", type=float, default=0.0)
    parser.add_argument("--lambda_sr_risk_rest", type=float, default=0.0)
    parser.add_argument("--sr_risk_boost", type=float, default=1.0)
    parser.add_argument("--sr_residual_anchor", choices=["prepared", "mip_hr"], default="prepared")
    parser.add_argument("--lambda_mip_hr_lowfreq", type=float, default=0.0)
    parser.add_argument("--mip_hr_lowfreq_kernel", type=int, default=17)
    parser.add_argument("--mip_hr_lowfreq_consistency_threshold", type=float, default=0.16)
    parser.add_argument("--mip_hr_lowfreq_mask_floor", type=float, default=0.0)
    parser.add_argument("--mip_closure_model_path", default="")
    parser.add_argument("--mip_closure_iteration", type=int, default=30000)
    parser.add_argument("--mip_closure_images_subdir", default="")
    parser.add_argument("--mip_closure_max_views", type=int, default=0)
    parser.add_argument("--lambda_mip_closure_alpha", type=float, default=0.0)
    parser.add_argument("--lambda_mip_closure_premul", type=float, default=0.0)
    parser.add_argument("--lambda_mip_closure_depth", type=float, default=0.0)
    parser.add_argument("--lambda_mip_closure_alpha_over", type=float, default=0.0)
    parser.add_argument("--lambda_mip_closure_premul_over", type=float, default=0.0)
    parser.add_argument("--mip_closure_kernel", type=int, default=25)
    parser.add_argument("--mip_closure_alpha_threshold", type=float, default=0.05)
    parser.add_argument("--mip_closure_reference_lowpass", type=int, default=1)
    parser.add_argument("--mip_closure_min_pixels", type=float, default=256.0)
    parser.add_argument("--mip_closure_depth_relative_min", type=float, default=0.5)
    parser.add_argument("--mip_closure_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--sr_prior_root", default="")
    parser.add_argument("--sr_prior_subdir", default="fused_priors")
    parser.add_argument("--sr_prior_mask_subdir", default="usable_masks")
    parser.add_argument("--sr_anchor_subdir", default="aligned_references")
    parser.add_argument("--sr_prior_mask_suffix", default="")
    parser.add_argument("--sr_images_subdir", default="images_2")
    parser.add_argument("--sr_max_views", type=int, default=0)
    parser.add_argument("--sr_view_mode", choices=["selected_lr", "all_train"], default="selected_lr")
    parser.add_argument("--lambda_sr_prior_l1", "--sr_prior_l1_weight", dest="lambda_sr_prior_l1", type=float, default=0.0)
    parser.add_argument("--lambda_sr_prior_hf", "--sr_prior_hf_weight", dest="lambda_sr_prior_hf", type=float, default=0.0)
    parser.add_argument("--sr_prior_warmup_start_iter", type=int, default=0)
    parser.add_argument("--sr_prior_warmup_end_iter", type=int, default=0)
    parser.add_argument("--sr_prior_start_scale", type=float, default=1.0)
    parser.add_argument("--sr_prior_end_scale", type=float, default=1.0)
    parser.add_argument("--sr_prior_update_scale", type=float, default=1.0)
    parser.add_argument("--sr_prior_schedule_mode", choices=["linear", "smoothstep"], default="smoothstep")
    parser.add_argument("--lambda_premul_hf_excess", type=float, default=0.0)
    parser.add_argument("--premul_hf_excess_kernel", type=int, default=9)
    parser.add_argument("--premul_hf_excess_ratio", type=float, default=1.25)
    parser.add_argument("--premul_hf_excess_margin", type=float, default=0.01)
    parser.add_argument("--sr_prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--sr_prior_consistency_threshold", type=float, default=0.12)
    parser.add_argument("--sr_prior_min_valid_ratio", type=float, default=0.50)
    parser.add_argument("--sr_prior_min_pixels", type=float, default=64.0)
    parser.add_argument("--sr_prior_delta_clip", type=float, default=0.15)
    parser.add_argument("--disable_sr_prior_hf_residual", action="store_true")
    parser.add_argument("--risk_radius_threshold", type=float, default=48.0)
    parser.add_argument("--risk_lr_residual_threshold", type=float, default=0.10)
    parser.add_argument("--risk_anisotropy_threshold", type=float, default=20.0)
    parser.add_argument("--risk_min_support_views", type=int, default=4)
    parser.add_argument("--max_displacement_ratio", type=float, default=0.0015)
    parser.add_argument("--max_displacement_abs", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--recompute_filter3d_before_train", type=int, default=0)
    parser.add_argument("--recompute_filter3d", type=int, default=1)
    parser.add_argument("--star_quarantine_payload_path", default="")
    parser.add_argument("--star_release_start_iter", type=int, default=0)
    parser.add_argument("--star_release_end_iter", type=int, default=0)
    parser.add_argument("--star_release_mode", choices=["linear", "smoothstep"], default="smoothstep")
    parser.add_argument("--star_release_rest_start_scale", type=float, default=-1.0)
    parser.add_argument("--star_release_rest_end_scale", type=float, default=1.0)
    parser.add_argument("--star_release_tau_start_scale", type=float, default=-1.0)
    parser.add_argument("--star_release_tau_end_scale", type=float, default=1.0)
    parser.add_argument("--star_release_min_alpha", type=float, default=1e-6)
    parser.add_argument("--star_release_opacity", action="store_true")
    parser.add_argument("--reparam_output_source_idx_path", default="")
    parser.add_argument("--reparam_parent_mask_path", default="")
    parser.add_argument("--reparam_child_mask_path", default="")
    parser.add_argument("--geometry_parent_mask_path", default="")
    parser.add_argument("--lambda_reparam_mass_cap", type=float, default=0.0)
    parser.add_argument("--reparam_mass_cap_eps", type=float, default=0.10)
    parser.add_argument("--lambda_reparam_child_tau_cap", type=float, default=0.0)
    parser.add_argument("--reparam_child_tau_cap_scale", type=float, default=2.5)
    parser.add_argument("--reparam_child_tau_cap_abs", type=float, default=0.03)
    parser.add_argument("--lambda_geometry_parent_tau_brake", type=float, default=0.0)
    parser.add_argument("--geometry_parent_tau_scale", type=float, default=1.0)
    parser.add_argument("--reparam_prune_iters", default="")
    parser.add_argument("--reparam_prune_child_dead_tau", type=float, default=0.0)
    parser.add_argument("--reparam_prune_child_spike_tau_scale", type=float, default=0.0)
    parser.add_argument("--reparam_prune_child_spike_tau_abs", type=float, default=0.0)
    parser.add_argument("--reparam_prune_child_spike_anisotropy", type=float, default=0.0)
    parser.add_argument("--reparam_prune_child_spike_risk_min", type=float, default=0.0)
    parser.add_argument("--phase_mode", choices=["joint", "appearance_only", "geometry_only", "alternating"], default="joint")
    parser.add_argument("--phase_block_steps", type=int, default=0)
    parser.add_argument("--optimize_gaussian_mask_payload", default="")
    parser.add_argument("--optimize_gaussian_mask_key", default="selected_mask")
    parser.add_argument("--gaussian_update_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_scale_axis_mode", choices=["all", "major_only"], default="all")
    parser.add_argument("--prior_prefilter_mode", choices=["none", "sr_touch_v0"], default="none")
    parser.add_argument("--prior_prefilter_mask_payload", default="")
    parser.add_argument("--prior_prefilter_mask_key", default="selected_mask")
    parser.add_argument("--prior_prefilter_view_limit", type=int, default=0)
    parser.add_argument("--prior_prefilter_min_touch_views", type=int, default=1)
    parser.add_argument("--prior_prefilter_min_visible_views", type=int, default=1)
    parser.add_argument("--prior_prefilter_min_touch_ratio", type=float, default=0.0)
    parser.add_argument("--prior_prefilter_min_candidate_opacity", type=float, default=0.0)
    parser.add_argument("--prior_prefilter_radius_scale", type=float, default=0.5)
    parser.add_argument("--prior_prefilter_min_touch_radius_px", type=float, default=2.0)
    parser.add_argument("--prior_prefilter_max_touch_radius_px", type=float, default=16.0)
    parser.add_argument("--prior_prefilter_save_path", default="")
    parser.add_argument("--enable_opacity_update", action="store_true")
    parser.add_argument("--enable_scale_update", action="store_true")
    parser.add_argument("--enable_dc_update", action="store_true")
    parser.add_argument("--enable_rest_update", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    start_model_path = Path(args.start_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    anchor_model_path = Path(args.anchor_model_path).expanduser().resolve() if str(args.anchor_model_path).strip() else None
    star_quarantine_payload_path = (
        Path(args.star_quarantine_payload_path).expanduser().resolve()
        if str(args.star_quarantine_payload_path).strip()
        else None
    )
    reparam_output_source_idx_path = (
        Path(args.reparam_output_source_idx_path).expanduser().resolve()
        if str(args.reparam_output_source_idx_path).strip()
        else None
    )
    reparam_parent_mask_path = (
        Path(args.reparam_parent_mask_path).expanduser().resolve()
        if str(args.reparam_parent_mask_path).strip()
        else None
    )
    reparam_child_mask_path = (
        Path(args.reparam_child_mask_path).expanduser().resolve()
        if str(args.reparam_child_mask_path).strip()
        else None
    )
    geometry_parent_mask_path = (
        Path(args.geometry_parent_mask_path).expanduser().resolve()
        if str(args.geometry_parent_mask_path).strip()
        else None
    )
    optimize_gaussian_mask_payload_path = (
        Path(args.optimize_gaussian_mask_payload).expanduser().resolve()
        if str(args.optimize_gaussian_mask_payload).strip()
        else None
    )
    prior_prefilter_mask_payload_path = (
        Path(args.prior_prefilter_mask_payload).expanduser().resolve()
        if str(args.prior_prefilter_mask_payload).strip()
        else None
    )
    prior_prefilter_save_path = (
        Path(args.prior_prefilter_save_path).expanduser().resolve()
        if str(args.prior_prefilter_save_path).strip()
        else None
    )
    mip_closure_model_path = (
        Path(args.mip_closure_model_path).expanduser().resolve()
        if str(args.mip_closure_model_path).strip()
        else None
    )
    sr_prior_root = Path(args.sr_prior_root).expanduser().resolve() if str(args.sr_prior_root).strip() else None
    sr_prior_dir = sr_prior_root / str(args.sr_prior_subdir) if sr_prior_root is not None and args.sr_prior_subdir else sr_prior_root
    sr_prior_mask_dir = (
        sr_prior_root / str(args.sr_prior_mask_subdir)
        if sr_prior_root is not None and args.sr_prior_mask_subdir
        else None
    )
    sr_anchor_dir = (
        sr_prior_root / str(args.sr_anchor_subdir)
        if sr_prior_root is not None and args.sr_anchor_subdir
        else None
    )

    if sr_prior_root is not None and not sr_prior_root.is_dir():
        raise FileNotFoundError(f"SR prior root not found: {sr_prior_root}")
    if sr_prior_dir is not None and not sr_prior_dir.is_dir():
        raise FileNotFoundError(f"SR fused prior dir not found: {sr_prior_dir}")
    if sr_prior_mask_dir is not None and not sr_prior_mask_dir.is_dir():
        print(
            f"[recover-cleaned-mip-lr-v0] warning: SR prior mask dir missing, using consistency gate only: {sr_prior_mask_dir}",
            flush=True,
        )
        sr_prior_mask_dir = None
    if sr_anchor_dir is not None and not sr_anchor_dir.is_dir():
        raise FileNotFoundError(
            f"SR GT-free anchor dir not found: {sr_anchor_dir}. "
            "Rebuild prepared SR priors so aligned_references are available."
        )
    if star_quarantine_payload_path is not None and not star_quarantine_payload_path.is_file():
        raise FileNotFoundError(f"Star quarantine payload not found: {star_quarantine_payload_path}")
    for label, path in (
        ("reparam_output_source_idx_path", reparam_output_source_idx_path),
        ("reparam_parent_mask_path", reparam_parent_mask_path),
        ("reparam_child_mask_path", reparam_child_mask_path),
        ("geometry_parent_mask_path", geometry_parent_mask_path),
        ("optimize_gaussian_mask_payload", optimize_gaussian_mask_payload_path),
        ("prior_prefilter_mask_payload", prior_prefilter_mask_payload_path),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    use_mip_closure = (
        float(args.lambda_mip_closure_alpha) > 0.0
        or float(args.lambda_mip_closure_premul) > 0.0
        or float(args.lambda_mip_closure_depth) > 0.0
        or float(args.lambda_mip_closure_alpha_over) > 0.0
        or float(args.lambda_mip_closure_premul_over) > 0.0
    )
    use_reparam_settle_controls = (
        float(args.lambda_reparam_mass_cap) > 0.0
        or float(args.lambda_reparam_child_tau_cap) > 0.0
        or float(args.lambda_geometry_parent_tau_brake) > 0.0
    )
    if use_reparam_settle_controls:
        if reparam_output_source_idx_path is None:
            raise ValueError("Reparameterization settle controls require --reparam_output_source_idx_path")
        if float(args.lambda_reparam_mass_cap) > 0.0 and reparam_parent_mask_path is None:
            raise ValueError("Reparameterization mass cap requires --reparam_parent_mask_path")
        if float(args.lambda_reparam_child_tau_cap) > 0.0 and reparam_child_mask_path is None:
            raise ValueError("Child tau cap requires --reparam_child_mask_path")
        if float(args.lambda_geometry_parent_tau_brake) > 0.0 and geometry_parent_mask_path is None:
            raise ValueError("Geometry parent tau brake requires --geometry_parent_mask_path")
    if use_mip_closure and mip_closure_model_path is None:
        raise ValueError("MIP closure recovery loss requires --mip_closure_model_path")
    if mip_closure_model_path is not None and not mip_closure_model_path.is_dir():
        raise FileNotFoundError(f"MIP closure model not found: {mip_closure_model_path}")
    dataset_args = build_dataset_args(str(scene_root), str(start_model_path), str(args.images_subdir))
    cameras_all = load_train_cameras_only(scene_root, start_model_path, str(args.images_subdir))
    cameras = select_uniform(cameras_all, int(args.max_views))
    if not cameras:
        raise RuntimeError(f"No train cameras found for scene={scene_root} images={args.images_subdir}")

    start_iter = resolve_iteration(start_model_path, int(args.start_iteration))
    output_iteration = int(args.output_iteration)
    if output_iteration < 0:
        output_iteration = int(start_iter) + int(args.iterations)

    student = load_model_ply(start_model_path, start_iter, int(dataset_args.sh_degree))
    anchor_model = None
    anchor_iter = None
    if anchor_model_path is not None:
        anchor_iter = resolve_iteration(anchor_model_path, int(args.anchor_iteration))
        anchor_model = load_model_ply(anchor_model_path, anchor_iter, int(dataset_args.sh_degree))
    if float(args.lambda_anchor_rgb) > 0.0 and anchor_model is None:
        raise ValueError("--lambda_anchor_rgb > 0 requires --anchor_model_path")
    use_mip_hr_anchor = (
        str(args.sr_residual_anchor) == "mip_hr" or float(args.lambda_mip_hr_lowfreq) > 0.0
    )
    if use_mip_hr_anchor and anchor_model is None:
        raise ValueError("MIP HR anchor supervision requires --anchor_model_path")
    sr_prior_index = index_image_dir(str(sr_prior_dir)) if sr_prior_dir is not None else None
    sr_anchor_index = index_image_dir(str(sr_anchor_dir)) if sr_anchor_dir is not None else None
    use_sr_prior = sr_prior_index is not None and (
        float(args.lambda_sr_prior_l1) > 0.0
        or float(args.lambda_sr_prior_hf) > 0.0
        or float(args.lambda_premul_hf_excess) > 0.0
    )
    use_sr_supervision = sr_prior_index is not None and (use_sr_prior or use_mip_hr_anchor)

    mip_closure_model = None
    mip_closure_iter = None
    mip_closure_splat_args = None
    if use_mip_closure and mip_closure_model_path is not None:
        mip_closure_iter = resolve_iteration(mip_closure_model_path, int(args.mip_closure_iteration))
        mip_closure_model = load_model_ply(mip_closure_model_path, mip_closure_iter, int(dataset_args.sh_degree))
        mip_closure_splat_args = load_splat_settings(mip_closure_model_path)

    star_release_state = load_star_release_state(
        star_quarantine_payload_path,
        student,
        rest_start_scale=float(args.star_release_rest_start_scale),
        rest_end_scale=float(args.star_release_rest_end_scale),
        tau_start_scale=float(args.star_release_tau_start_scale),
        tau_end_scale=float(args.star_release_tau_end_scale),
        release_start_iter=int(args.star_release_start_iter),
        release_end_iter=int(args.star_release_end_iter),
        release_mode=str(args.star_release_mode),
        release_opacity=bool(args.star_release_opacity),
        min_alpha=float(args.star_release_min_alpha),
    )
    star_release_status = apply_star_release(student, star_release_state, 0)
    reparam_settle_state = load_reparam_settle_state(
        student=student,
        output_source_idx_path=reparam_output_source_idx_path,
        parent_mask_path=reparam_parent_mask_path,
        child_mask_path=reparam_child_mask_path,
        geometry_parent_mask_path=geometry_parent_mask_path,
    )

    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    anchor_cache = None
    if anchor_model is not None and float(args.lambda_anchor_rgb) > 0.0:
        anchor_cache = cache_anchor_renders(cameras, anchor_model, background)

    sr_cameras: List[object] = []
    sr_cache: List[Dict[str, torch.Tensor | str | float | None]] = []
    sr_images_subdir = str(args.sr_images_subdir).strip() or str(args.images_subdir)
    if use_sr_supervision:
        if sr_images_subdir == str(args.images_subdir):
            sr_all_cameras = cameras_all
        else:
            sr_all_cameras = load_train_cameras_only(scene_root, start_model_path, sr_images_subdir)
        if str(args.sr_view_mode) == "all_train":
            sr_candidates = sr_all_cameras
        else:
            selected_names = {normalize_image_name(cam.image_name) for cam in cameras}
            sr_candidates = [cam for cam in sr_all_cameras if normalize_image_name(cam.image_name) in selected_names]
            if not sr_candidates:
                print(
                    "[recover-cleaned-mip-lr-v0] warning: no SR cameras matched LR selected names; "
                    f"falling back to all {sr_images_subdir} train cameras",
                    flush=True,
                )
                sr_candidates = sr_all_cameras
        if int(args.sr_max_views) > 0:
            sr_max_views = int(args.sr_max_views)
        elif str(args.sr_view_mode) == "all_train":
            sr_max_views = 0
        else:
            sr_max_views = int(args.max_views)
        sr_cameras = select_uniform(sr_candidates, sr_max_views)
        if not sr_cameras:
            raise RuntimeError("SR prior was enabled but no SR supervision cameras were found.")

    mip_closure_cameras: List[object] = []
    mip_closure_cache: List[Dict[str, torch.Tensor | str]] = []
    mip_closure_images_subdir = str(args.mip_closure_images_subdir).strip() or sr_images_subdir
    if use_mip_closure and mip_closure_model_path is not None:
        closure_all_cameras = load_train_cameras_only(scene_root, mip_closure_model_path, mip_closure_images_subdir)
        closure_max_views = int(args.mip_closure_max_views) if int(args.mip_closure_max_views) > 0 else int(args.max_views)
        mip_closure_cameras = select_uniform(closure_all_cameras, closure_max_views)
        if not mip_closure_cameras:
            raise RuntimeError(
                f"MIP closure was enabled but no cameras were found for images={mip_closure_images_subdir}"
            )

    filter_cameras = list(cameras) + [cam for cam in sr_cameras if cam not in cameras]
    filter_cameras += [cam for cam in mip_closure_cameras if cam not in filter_cameras]
    if bool(int(args.recompute_filter3d_before_train)) and filter_cameras:
        student.compute_3D_filter(filter_cameras, CUDA=False)
        if anchor_model is not None and use_mip_hr_anchor:
            anchor_model.compute_3D_filter(filter_cameras, CUDA=False)
    if mip_closure_model is not None and mip_closure_cameras:
        mip_closure_model.compute_3D_filter(mip_closure_cameras, CUDA=False)

    sr_cameras, sr_cache = build_prepared_sr_cache(
        sr_cameras,
        student,
        background,
        sr_prior_index=sr_prior_index,
        sr_anchor_index=sr_anchor_index,
        sr_prior_mask_dir=sr_prior_mask_dir,
        sr_prior_mask_suffix=str(args.sr_prior_mask_suffix),
        prior_consistency_threshold=float(args.sr_prior_consistency_threshold),
        prior_mask_floor=float(args.sr_prior_mask_floor),
    )
    if use_sr_supervision and not sr_cache:
        if use_sr_prior:
            raise RuntimeError(
                "SR prior was enabled but no prepared prior images matched the selected SR cameras. "
                f"root={sr_prior_root}, subdir={args.sr_prior_subdir}, images={sr_images_subdir}"
            )
        raise RuntimeError(
            "MIP-HR lowfreq supervision was enabled but no prepared prior images matched the selected SR cameras. "
            f"root={sr_prior_root}, subdir={args.sr_prior_subdir}, images={sr_images_subdir}"
        )
    if mip_closure_model is not None and mip_closure_splat_args is not None:
        mip_closure_cache = render_mip_closure_cache(
            mip_closure_cameras,
            mip_closure_model,
            background,
            mip_splat_args=mip_closure_splat_args,
        )
    mip_hr_anchor_cache: List[torch.Tensor] = []
    if use_mip_hr_anchor and sr_cameras:
        if not sr_cache:
            raise RuntimeError("MIP HR anchor supervision needs SR cameras/cache.")
        mip_hr_anchor_cache = cache_anchor_renders(sr_cameras, anchor_model, background)
    hf_excess_cache: Sequence[Dict[str, torch.Tensor | str | float | None]] = sr_cache
    if str(args.sr_residual_anchor) == "mip_hr" and mip_hr_anchor_cache:
        hf_excess_cache = [
            sr_target_with_anchor(target, mip_anchor_rgb)
            for target, mip_anchor_rgb in zip(sr_cache, mip_hr_anchor_cache)
        ]

    static = static_gaussian_metrics(student)
    lr_render = collect_render_metrics(
        student,
        cameras,
        large_radius_px=float(args.risk_radius_threshold),
        residual_against_camera=True,
        depth_min=0.02,
        background=background,
    )
    risk_weight = build_dense_risk_weight(
        static,
        lr_render,
        radius_threshold=float(args.risk_radius_threshold),
        lr_residual_threshold=float(args.risk_lr_residual_threshold),
        anisotropy_threshold=float(args.risk_anisotropy_threshold),
        min_support_views=int(args.risk_min_support_views),
    ).to(device="cuda")
    prior_risk_weight = risk_weight
    if use_sr_prior and float(args.sr_risk_boost) > 1.0:
        prior_risk_weight = torch.clamp(prior_risk_weight * float(args.sr_risk_boost), max=1.0)

    phase_name, phase_flags = phase_train_flags(args, 1)
    union_train_flags = phase_union_train_flags(args)
    if not any(bool(value) for value in union_train_flags.values()):
        raise ValueError(f"Phase mode {args.phase_mode!r} produced no trainable parameters.")

    external_update_mask = load_gaussian_update_mask_payload(
        str(optimize_gaussian_mask_payload_path) if optimize_gaussian_mask_payload_path is not None else "",
        str(args.optimize_gaussian_mask_key),
        int(student.get_xyz.shape[0]),
    )
    prefilter_candidate_mask = load_gaussian_update_mask_payload(
        str(prior_prefilter_mask_payload_path) if prior_prefilter_mask_payload_path is not None else "",
        str(args.prior_prefilter_mask_key),
        int(student.get_xyz.shape[0]),
    )
    prior_prefilter_state: Dict[str, object] | None = None
    prior_prefilter_mask: torch.Tensor | None = None
    if str(args.prior_prefilter_mode) != "none":
        if not sr_cache:
            raise ValueError("Prior prefilter requires SR prior cache/views to be available.")
        prior_prefilter_state = build_sr_touch_prefilter(
            sr_cameras,
            sr_cache,
            student,
            background,
            base_mask=prefilter_candidate_mask,
            view_limit=int(args.prior_prefilter_view_limit),
            min_touch_views=int(args.prior_prefilter_min_touch_views),
            min_visible_views=int(args.prior_prefilter_min_visible_views),
            min_touch_ratio=float(args.prior_prefilter_min_touch_ratio),
            min_candidate_opacity=float(args.prior_prefilter_min_candidate_opacity),
            radius_scale=float(args.prior_prefilter_radius_scale),
            min_touch_radius_px=float(args.prior_prefilter_min_touch_radius_px),
            max_touch_radius_px=float(args.prior_prefilter_max_touch_radius_px),
        )
        prior_prefilter_mask = prior_prefilter_state["selected_mask"]
        if prior_prefilter_save_path is None:
            prior_prefilter_save_path = output_model_path / "prior_prefilter_selected_mask.pt"
        prior_prefilter_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "selected_mask": prior_prefilter_state["selected_mask"].detach().cpu(),
                "selected_ids": prior_prefilter_state["selected_ids"].detach().cpu(),
                "visible_view_count": prior_prefilter_state["visible_view_count"].detach().cpu(),
                "touch_view_count": prior_prefilter_state["touch_view_count"].detach().cpu(),
                "touch_ratio": prior_prefilter_state["touch_ratio"].detach().cpu(),
                "summary": prior_prefilter_state["summary"],
            },
            prior_prefilter_save_path,
        )
    gaussian_update_mask = combine_optional_masks(external_update_mask, prior_prefilter_mask)
    if gaussian_update_mask is not None and int(gaussian_update_mask.sum().item()) <= 0:
        raise RuntimeError("Gaussian update mask selected zero gaussians after combining prefilter and external mask.")

    xyz_init = student._xyz.detach().clone()
    opacity_init = student.get_opacity.detach().clone()
    scale_init = student.get_scaling.detach().clone()
    dc_init = student._features_dc.detach().clone()
    rest_init = student._features_rest.detach().clone()
    bbox_diag = torch.linalg.norm(torch.max(xyz_init, dim=0).values - torch.min(xyz_init, dim=0).values).clamp_min(1e-6)
    max_displacement = (
        float(args.max_displacement_abs)
        if float(args.max_displacement_abs) > 0.0
        else float(args.max_displacement_ratio) * float(bbox_diag.item())
    )
    reparam_prune_iters = {value for value in parse_int_list(str(args.reparam_prune_iters)) if int(value) > 0}
    use_reparam_prune = bool(reparam_prune_iters) and reparam_settle_state is not None and bool(reparam_settle_state.get("enabled", False))
    prune_history: List[Dict[str, float | int]] = []
    scale_param_mask = build_scale_update_mask(
        scale_init,
        update_mask=gaussian_update_mask,
        axis_mode=str(args.gaussian_scale_axis_mode),
    )

    freeze_for_lr_recover(
        student,
        train_xyz=bool(phase_flags["train_xyz"]),
        train_opacity=bool(phase_flags["train_opacity"]),
        train_scale=bool(phase_flags["train_scale"]),
        train_dc=bool(phase_flags["train_dc"]),
        train_rest=bool(phase_flags["train_rest"]),
    )
    optimizer = build_optimizer(
        student,
        train_xyz=bool(union_train_flags["train_xyz"]),
        xyz_lr=float(args.xyz_lr),
        opacity_lr=float(args.opacity_lr),
        scale_lr=float(args.scale_lr),
        dc_lr=float(args.dc_lr),
        rest_lr=float(args.rest_lr),
        train_opacity=bool(union_train_flags["train_opacity"]),
        train_scale=bool(union_train_flags["train_scale"]),
        train_dc=bool(union_train_flags["train_dc"]),
        train_rest=bool(union_train_flags["train_rest"]),
    )

    print(f"[recover-cleaned-mip-lr-v0] scene       : {scene_root}")
    print(f"[recover-cleaned-mip-lr-v0] start model : {start_model_path} iter={start_iter} n={student.get_xyz.shape[0]}")
    if anchor_model_path is not None:
        print(f"[recover-cleaned-mip-lr-v0] anchor model: {anchor_model_path} iter={anchor_iter}")
    print(f"[recover-cleaned-mip-lr-v0] output      : {output_model_path}")
    print(f"[recover-cleaned-mip-lr-v0] views       : {len(cameras)} from {args.images_subdir}")
    print(
        f"[recover-cleaned-mip-lr-v0] train       : iter={args.iterations} "
        f"xyz_lr={args.xyz_lr} op={args.enable_opacity_update} scale={args.enable_scale_update} "
        f"dc={args.enable_dc_update} rest={args.enable_rest_update}"
    )
    print(f"[recover-cleaned-mip-lr-v0] SR prior   : {sr_prior_root if sr_prior_root is not None else '(disabled)'}")
    if use_sr_prior:
        print(
            f"[recover-cleaned-mip-lr-v0] SR views   : {sr_images_subdir} cached={len(sr_cache)} "
            f"weights={float(args.lambda_sr_prior_l1):.4g}/{float(args.lambda_sr_prior_hf):.4g} "
            f"schedule={float(args.sr_prior_start_scale):.3g}->{float(args.sr_prior_end_scale):.3g} "
            f"update={float(args.sr_prior_update_scale):.3g} "
            f"iter={int(args.sr_prior_warmup_start_iter)}->{int(args.sr_prior_warmup_end_iter)} "
            f"hf_excess={float(args.lambda_premul_hf_excess):.4g} "
            f"k={int(args.premul_hf_excess_kernel)} ratio={float(args.premul_hf_excess_ratio):.4g} "
            f"margin={float(args.premul_hf_excess_margin):.4g}"
        )
    elif use_sr_supervision:
        print(
            f"[recover-cleaned-mip-lr-v0] SR cache   : {sr_images_subdir} cached={len(sr_cache)} "
            f"(used for MIP-HR lowfreq / consistency only)"
        )
    print(
        f"[recover-cleaned-mip-lr-v0] MIP-HR sup : anchor={args.sr_residual_anchor} "
        f"lowfreq={float(args.lambda_mip_hr_lowfreq):.4g} cached={len(mip_hr_anchor_cache)}"
    )
    print(
        f"[recover-cleaned-mip-lr-v0] MIP closure: model={mip_closure_model_path if use_mip_closure else '(disabled)'} "
        f"views={len(mip_closure_cache)} weights={float(args.lambda_mip_closure_alpha):.4g}/"
        f"{float(args.lambda_mip_closure_premul):.4g}/{float(args.lambda_mip_closure_depth):.4g} "
        f"over={float(args.lambda_mip_closure_alpha_over):.4g}/"
        f"{float(args.lambda_mip_closure_premul_over):.4g} "
        f"kernel={int(args.mip_closure_kernel)} ref_lowpass={int(args.mip_closure_reference_lowpass)}"
    )
    if star_release_state is not None:
        print(
            f"[recover-cleaned-mip-lr-v0] star release: payload={star_quarantine_payload_path} "
            f"n={int(star_release_status['selected_count'])} "
            f"iter={int(args.star_release_start_iter)}->{int(args.star_release_end_iter)} "
            f"rest={float(star_release_status['rest_scale']):.4g}->{float(args.star_release_rest_end_scale):.4g} "
            f"tau={float(star_release_status['tau_scale']):.4g}->{float(args.star_release_tau_end_scale):.4g} "
            f"opacity={bool(args.star_release_opacity)} mode={args.star_release_mode}"
        )
    if reparam_settle_state is not None:
        print(
            f"[recover-cleaned-mip-lr-v0] reparam ctrl: "
            f"groups={int(reparam_settle_state.get('group_count', 0))} "
            f"parents={int(reparam_settle_state.get('parent_count', 0))} "
            f"children={int(reparam_settle_state.get('child_count', 0))} "
            f"geom_parents={int(reparam_settle_state.get('geometry_parent_count', 0))} "
            f"weights={float(args.lambda_reparam_mass_cap):.4g}/"
            f"{float(args.lambda_reparam_child_tau_cap):.4g}/"
            f"{float(args.lambda_geometry_parent_tau_brake):.4g}"
        )
    if use_reparam_prune:
        print(
            f"[recover-cleaned-mip-lr-v0] reparam prune: "
            f"iters={sorted(reparam_prune_iters)} dead_tau={float(args.reparam_prune_child_dead_tau):.4g} "
            f"spike_tau={float(args.reparam_prune_child_spike_tau_scale):.4g}/{float(args.reparam_prune_child_spike_tau_abs):.4g} "
            f"aniso={float(args.reparam_prune_child_spike_anisotropy):.4g} risk>={float(args.reparam_prune_child_spike_risk_min):.4g}"
        )
    print(
        f"[recover-cleaned-mip-lr-v0] filter3D   : "
        f"before_train={int(args.recompute_filter3d_before_train)} final={int(args.recompute_filter3d)}"
    )
    print(
        f"[recover-cleaned-mip-lr-v0] risk mean   : {float(risk_weight.mean().item()):.6f} "
        f"max={float(risk_weight.max().item()):.6f} sr_boost={float(args.sr_risk_boost):.3g}"
    )
    print(
        f"[recover-cleaned-mip-lr-v0] phase       : mode={args.phase_mode} "
        f"block={int(args.phase_block_steps)} current={phase_name}"
    )
    if gaussian_update_mask is not None:
        print(
            f"[recover-cleaned-mip-lr-v0] update mask : "
            f"selected={int(gaussian_update_mask.sum().item())}/{int(gaussian_update_mask.shape[0])} "
            f"scale={float(args.gaussian_update_scale):.4g} axis={args.gaussian_scale_axis_mode}"
        )
    else:
        print("[recover-cleaned-mip-lr-v0] update mask : disabled")
    if prior_prefilter_state is not None:
        prefilter_summary = prior_prefilter_state["summary"]
        print(
            f"[recover-cleaned-mip-lr-v0] prefilter   : mode={args.prior_prefilter_mode} "
            f"selected={prefilter_summary['selected_count']}/{prefilter_summary['candidate_count']} "
            f"views={prefilter_summary['matched_view_count']} "
            f"save={prior_prefilter_save_path}"
        )

    before_rgb = evaluate_rgb_losses(cameras, student, background, anchor_cache=anchor_cache)
    if mip_hr_anchor_cache:
        before_sr = evaluate_mip_hr_anchor_losses(
            sr_cameras,
            sr_cache,
            mip_hr_anchor_cache,
            student,
            background,
            residual_anchor_mode=str(args.sr_residual_anchor),
            min_pixels=float(args.sr_prior_min_pixels),
            min_valid_ratio=float(args.sr_prior_min_valid_ratio),
            prior_delta_clip=float(args.sr_prior_delta_clip),
            disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
            lowfreq_kernel=int(args.mip_hr_lowfreq_kernel),
            lowfreq_consistency_threshold=float(args.mip_hr_lowfreq_consistency_threshold),
            lowfreq_mask_floor=float(args.mip_hr_lowfreq_mask_floor),
        )
    else:
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
    before_premul_hf_excess = evaluate_premul_hf_excess_losses(
        sr_cameras,
        hf_excess_cache,
        student,
        background,
        kernel_size=int(args.premul_hf_excess_kernel),
        excess_ratio=float(args.premul_hf_excess_ratio),
        margin=float(args.premul_hf_excess_margin),
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
    ) if hf_excess_cache else {"premul_hf_excess": 0.0}
    before_mip_closure = evaluate_mip_closure_losses(
        mip_closure_cameras,
        mip_closure_cache,
        student,
        background,
        splat_args=mip_closure_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.mip_closure_min_pixels),
        depth_relative_min=float(args.mip_closure_depth_relative_min),
        charbonnier_eps=float(args.mip_closure_charbonnier_eps),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    ) if mip_closure_cache and mip_closure_splat_args is not None else {"alpha": 0.0, "premul": 0.0, "depth": 0.0}
    before_mip_closure_over = evaluate_mip_closure_over_losses(
        mip_closure_cameras,
        mip_closure_cache,
        student,
        background,
        splat_args=mip_closure_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.mip_closure_min_pixels),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    ) if mip_closure_cache and mip_closure_splat_args is not None else {"alpha_over": 0.0, "premul_luma_over": 0.0}

    progress = tqdm(range(1, int(args.iterations) + 1), desc="recover cleaned mip lr")
    log_rows = []
    current_phase_name = ""
    for iteration in progress:
        phase_name, phase_flags = phase_train_flags(args, iteration)
        if phase_name != current_phase_name:
            freeze_for_lr_recover(
                student,
                train_xyz=bool(phase_flags["train_xyz"]),
                train_opacity=bool(phase_flags["train_opacity"]),
                train_scale=bool(phase_flags["train_scale"]),
                train_dc=bool(phase_flags["train_dc"]),
                train_rest=bool(phase_flags["train_rest"]),
            )
            current_phase_name = phase_name
        star_release_status = apply_star_release(student, star_release_state, iteration)
        loss = torch.zeros((), dtype=torch.float32, device="cuda")
        view_idx = randint(0, len(cameras) - 1)
        camera = cameras[view_idx]
        rgb = render_simple(camera, student, background)["render"].clamp(0.0, 1.0)
        lr_anchor = camera_rgb(camera, background.device).to(dtype=rgb.dtype)

        loss_lr_rgb = mean_l1(rgb, lr_anchor)
        loss = loss + float(args.lambda_lr_rgb) * loss_lr_rgb

        loss_anchor_rgb = torch.zeros((), dtype=torch.float32, device="cuda")
        if anchor_cache is not None and float(args.lambda_anchor_rgb) > 0.0:
            anchor_rgb = anchor_cache[view_idx].to(device=rgb.device, dtype=rgb.dtype)
            loss_anchor_rgb = mean_l1(rgb, anchor_rgb)
            loss = loss + float(args.lambda_anchor_rgb) * loss_anchor_rgb

        loss_sr_l1 = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_sr_hf = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_premul_hf_excess = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_mip_hr_lowfreq = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_mip_closure_alpha = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_mip_closure_premul = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_mip_closure_depth = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_mip_closure_alpha_over = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_mip_closure_premul_over = torch.zeros((), dtype=torch.float32, device="cuda")
        sr_valid_ratio_step = 0.0
        sr_mask_pixels_step = 0.0
        sr_mask_mean_step = 0.0
        sr_loss_active_step = 0
        sr_prior_scale = scheduled_sr_prior_scale(
            iteration,
            start_iter=int(args.sr_prior_warmup_start_iter),
            end_iter=int(args.sr_prior_warmup_end_iter),
            start_scale=float(args.sr_prior_start_scale),
            end_scale=float(args.sr_prior_end_scale),
            update_scale=float(args.sr_prior_update_scale),
            mode=str(args.sr_prior_schedule_mode),
        )
        if sr_cache and (
            float(args.lambda_sr_prior_l1) > 0.0
            or float(args.lambda_sr_prior_hf) > 0.0
            or float(args.lambda_premul_hf_excess) > 0.0
            or float(args.lambda_mip_hr_lowfreq) > 0.0
        ):
            sr_idx = randint(0, len(sr_cache) - 1)
            sr_camera = sr_cameras[sr_idx]
            sr_target = sr_cache[sr_idx]
            sr_pkg = render_simple(sr_camera, student, background)
            sr_rgb = sr_pkg["render"].clamp(0.0, 1.0)
            loss_target = sr_target
            if str(args.sr_residual_anchor) == "mip_hr" and mip_hr_anchor_cache:
                loss_target = sr_target_with_anchor(sr_target, mip_hr_anchor_cache[sr_idx])
            sr_valid_ratio_step = float(loss_target.get("valid_ratio", 0.0) or 0.0)
            prior_mask_step = loss_target.get("prior_mask")
            if torch.is_tensor(prior_mask_step):
                prior_mask_step = prior_mask_step.to(device=sr_rgb.device, dtype=sr_rgb.dtype).clamp(0.0, 1.0)
                sr_mask_pixels_step = float(prior_mask_step.sum().detach().item())
                sr_mask_mean_step = float(prior_mask_step.mean().detach().item())
            maybe_sr_l1, maybe_sr_hf = compute_prepared_sr_losses(
                sr_rgb,
                loss_target,
                min_pixels=float(args.sr_prior_min_pixels),
                min_valid_ratio=float(args.sr_prior_min_valid_ratio),
                prior_delta_clip=float(args.sr_prior_delta_clip),
                disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
            )
            if maybe_sr_l1 is not None:
                sr_loss_active_step = 1
                loss_sr_l1 = maybe_sr_l1
                loss = loss + sr_prior_scale * float(args.lambda_sr_prior_l1) * loss_sr_l1
            if maybe_sr_hf is not None:
                sr_loss_active_step = 1
                loss_sr_hf = maybe_sr_hf
                loss = loss + sr_prior_scale * float(args.lambda_sr_prior_hf) * loss_sr_hf
            if mip_hr_anchor_cache and float(args.lambda_mip_hr_lowfreq) > 0.0:
                maybe_mip_hr = compute_mip_hr_lowfreq_loss(
                    sr_rgb,
                    mip_hr_anchor_cache[sr_idx],
                    sr_target,
                    kernel_size=int(args.mip_hr_lowfreq_kernel),
                    consistency_threshold=float(args.mip_hr_lowfreq_consistency_threshold),
                    mask_floor=float(args.mip_hr_lowfreq_mask_floor),
                )
                if maybe_mip_hr is not None:
                    loss_mip_hr_lowfreq = maybe_mip_hr
                    loss = loss + float(args.lambda_mip_hr_lowfreq) * loss_mip_hr_lowfreq
            if float(args.lambda_premul_hf_excess) > 0.0:
                maybe_hf_excess = compute_premul_hf_excess_loss(
                    sr_pkg,
                    loss_target,
                    kernel_size=int(args.premul_hf_excess_kernel),
                    excess_ratio=float(args.premul_hf_excess_ratio),
                    margin=float(args.premul_hf_excess_margin),
                    min_pixels=float(args.sr_prior_min_pixels),
                    min_valid_ratio=float(args.sr_prior_min_valid_ratio),
                    prior_delta_clip=float(args.sr_prior_delta_clip),
                )
                if maybe_hf_excess is not None:
                    sr_loss_active_step = 1
                    loss_premul_hf_excess = maybe_hf_excess
                    loss = loss + sr_prior_scale * float(args.lambda_premul_hf_excess) * loss_premul_hf_excess

        if mip_closure_cache and mip_closure_splat_args is not None and (
            float(args.lambda_mip_closure_alpha) > 0.0
            or float(args.lambda_mip_closure_premul) > 0.0
            or float(args.lambda_mip_closure_depth) > 0.0
            or float(args.lambda_mip_closure_alpha_over) > 0.0
            or float(args.lambda_mip_closure_premul_over) > 0.0
        ):
            closure_idx = randint(0, len(mip_closure_cache) - 1)
            closure_pkg = render_simple(
                mip_closure_cameras[closure_idx],
                student,
                background,
                splat_args=mip_closure_splat_args,
            )
            maybe_alpha, maybe_premul, maybe_depth = compute_mip_closure_losses(
                closure_pkg,
                mip_closure_cache[closure_idx],
                kernel_size=int(args.mip_closure_kernel),
                alpha_threshold=float(args.mip_closure_alpha_threshold),
                min_pixels=float(args.mip_closure_min_pixels),
                depth_relative_min=float(args.mip_closure_depth_relative_min),
                charbonnier_eps=float(args.mip_closure_charbonnier_eps),
                reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
            )
            if maybe_alpha is not None:
                loss_mip_closure_alpha = maybe_alpha
                loss = loss + float(args.lambda_mip_closure_alpha) * loss_mip_closure_alpha
            if maybe_premul is not None:
                loss_mip_closure_premul = maybe_premul
                loss = loss + float(args.lambda_mip_closure_premul) * loss_mip_closure_premul
            if maybe_depth is not None:
                loss_mip_closure_depth = maybe_depth
                loss = loss + float(args.lambda_mip_closure_depth) * loss_mip_closure_depth
            maybe_alpha_over, maybe_premul_over = compute_mip_closure_over_losses(
                closure_pkg,
                mip_closure_cache[closure_idx],
                kernel_size=int(args.mip_closure_kernel),
                alpha_threshold=float(args.mip_closure_alpha_threshold),
                min_pixels=float(args.mip_closure_min_pixels),
                reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
            )
            if maybe_alpha_over is not None:
                loss_mip_closure_alpha_over = maybe_alpha_over
                loss = loss + float(args.lambda_mip_closure_alpha_over) * loss_mip_closure_alpha_over
            if maybe_premul_over is not None:
                loss_mip_closure_premul_over = maybe_premul_over
                loss = loss + float(args.lambda_mip_closure_premul_over) * loss_mip_closure_premul_over

        xyz_delta = (student._xyz - xyz_init) / bbox_diag
        loss_xyz_anchor = torch.mean(xyz_delta * xyz_delta)
        loss = loss + float(args.lambda_xyz_anchor) * loss_xyz_anchor

        current_opacity = student.get_opacity
        current_scale = student.get_scaling
        loss_opacity_anchor = torch.mean(torch.abs(current_opacity - opacity_init))
        rel_scale = (current_scale - scale_init) / torch.clamp(scale_init, min=1e-8)
        loss_scale_anchor = torch.mean(rel_scale * rel_scale)
        loss_dc_anchor = torch.mean(torch.abs(student._features_dc - dc_init))
        loss_rest_anchor = torch.mean(torch.abs(student._features_rest - rest_init))
        rest_abs = torch.mean(
            torch.abs(student._features_rest).reshape(int(student.get_xyz.shape[0]), -1),
            dim=1,
        )
        loss_reparam_mass_cap = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_reparam_child_tau_cap = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_geometry_parent_tau_brake = torch.zeros((), dtype=torch.float32, device="cuda")
        reparam_mass_stats = {"over_groups": 0, "overflow_mean": 0.0, "overflow_max": 0.0}
        reparam_child_stats = {"over_children": 0, "overflow_mean": 0.0, "overflow_max": 0.0}
        geometry_parent_stats = {"over_parents": 0, "overflow_mean": 0.0, "overflow_max": 0.0}

        loss_risk_opacity = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_risk_scale = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_risk_rest = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_sr_risk_opacity = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_sr_risk_scale = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_sr_risk_rest = torch.zeros((), dtype=torch.float32, device="cuda")
        if bool(args.enable_opacity_update):
            loss = loss + float(args.lambda_opacity_anchor) * loss_opacity_anchor
            loss_risk_opacity = torch.mean(risk_weight[:, None] * current_opacity)
            loss = loss + float(args.lambda_risk_opacity) * loss_risk_opacity
            if use_sr_prior and float(args.lambda_sr_risk_opacity) > 0.0:
                loss_sr_risk_opacity = torch.mean(prior_risk_weight[:, None] * current_opacity)
                loss = loss + float(args.lambda_sr_risk_opacity) * loss_sr_risk_opacity
        if bool(args.enable_scale_update):
            loss = loss + float(args.lambda_scale_anchor) * loss_scale_anchor
            scale_max = current_scale.max(dim=1).values
            init_scale_max = torch.clamp(scale_init.max(dim=1).values, min=1e-8)
            scale_ratio = scale_max / init_scale_max
            loss_risk_scale = torch.mean(risk_weight * scale_ratio)
            loss = loss + float(args.lambda_risk_scale) * loss_risk_scale
            if use_sr_prior and float(args.lambda_sr_risk_scale) > 0.0:
                loss_sr_risk_scale = torch.mean(prior_risk_weight * scale_ratio)
                loss = loss + float(args.lambda_sr_risk_scale) * loss_sr_risk_scale
        if bool(args.enable_dc_update):
            loss = loss + float(args.lambda_dc_anchor) * loss_dc_anchor
        if bool(args.enable_rest_update):
            loss = loss + float(args.lambda_rest_anchor) * loss_rest_anchor
            if float(args.lambda_risk_rest) > 0.0:
                loss_risk_rest = torch.mean(risk_weight * rest_abs)
                loss = loss + float(args.lambda_risk_rest) * loss_risk_rest
            if use_sr_prior and float(args.lambda_sr_risk_rest) > 0.0:
                loss_sr_risk_rest = torch.mean(prior_risk_weight * rest_abs)
                loss = loss + float(args.lambda_sr_risk_rest) * loss_sr_risk_rest
        if float(args.lambda_reparam_mass_cap) > 0.0:
            loss_reparam_mass_cap, reparam_mass_stats = compute_reparam_mass_cap_loss(
                student,
                reparam_settle_state,
                eps=float(args.reparam_mass_cap_eps),
            )
            loss = loss + float(args.lambda_reparam_mass_cap) * loss_reparam_mass_cap
        if float(args.lambda_reparam_child_tau_cap) > 0.0:
            loss_reparam_child_tau_cap, reparam_child_stats = compute_reparam_child_tau_cap_loss(
                student,
                reparam_settle_state,
                tau_scale=float(args.reparam_child_tau_cap_scale),
                tau_floor=float(args.reparam_child_tau_cap_abs),
            )
            loss = loss + float(args.lambda_reparam_child_tau_cap) * loss_reparam_child_tau_cap
        if float(args.lambda_geometry_parent_tau_brake) > 0.0:
            loss_geometry_parent_tau_brake, geometry_parent_stats = compute_geometry_parent_tau_brake_loss(
                student,
                reparam_settle_state,
                tau_scale=float(args.geometry_parent_tau_scale),
            )
            loss = loss + float(args.lambda_geometry_parent_tau_brake) * loss_geometry_parent_tau_brake

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        apply_gaussian_update_mask(
            optimizer,
            total_gaussians=int(student.get_xyz.shape[0]),
            update_mask=gaussian_update_mask,
            update_scale=float(args.gaussian_update_scale),
            scale_param_mask=scale_param_mask,
        )
        grad_xyz = grad_l2_norm(student._xyz)
        grad_opacity = grad_l2_norm(student._opacity)
        grad_scale = grad_l2_norm(student._scaling)
        grad_dc = grad_l2_norm(student._features_dc)
        grad_rest = grad_l2_norm(student._features_rest)
        optimizer.step()
        clamp_xyz_displacement(student, xyz_init, max_displacement=max_displacement)
        star_release_status = apply_star_release(student, star_release_state, iteration)

        prune_step_count = 0
        prune_child_dead_count = 0
        prune_child_spike_count = 0
        if use_reparam_prune and int(iteration) in reparam_prune_iters:
            prune_mask, prune_stats = compute_reparam_prune_mask(
                student,
                reparam_settle_state,
                risk_weight=prior_risk_weight,
                child_dead_tau=float(args.reparam_prune_child_dead_tau),
                child_spike_tau_scale=float(args.reparam_prune_child_spike_tau_scale),
                child_spike_tau_abs=float(args.reparam_prune_child_spike_tau_abs),
                child_spike_anisotropy=float(args.reparam_prune_child_spike_anisotropy),
                child_spike_risk_min=float(args.reparam_prune_child_spike_risk_min),
            )
            prune_step_count = int(prune_stats["pruned"])
            prune_child_dead_count = int(prune_stats["child_dead"])
            prune_child_spike_count = int(prune_stats["child_spike"])
            if prune_step_count > 0:
                valid_mask, old_to_new, _ = prune_student_with_optimizer(student, optimizer, prune_mask)
                xyz_init = xyz_init[valid_mask]
                opacity_init = opacity_init[valid_mask]
                scale_init = scale_init[valid_mask]
                dc_init = dc_init[valid_mask]
                rest_init = rest_init[valid_mask]
                risk_weight = risk_weight[valid_mask]
                prior_risk_weight = prior_risk_weight[valid_mask]
                if gaussian_update_mask is not None:
                    gaussian_update_mask = gaussian_update_mask[valid_mask]
                scale_param_mask = build_scale_update_mask(
                    scale_init,
                    update_mask=gaussian_update_mask,
                    axis_mode=str(args.gaussian_scale_axis_mode),
                )
                reparam_settle_state = update_reparam_state_after_prune(reparam_settle_state, valid_mask, old_to_new)
                star_release_state = update_star_release_state_after_prune(star_release_state, valid_mask, old_to_new)
                star_release_status = apply_star_release(student, star_release_state, iteration)
                prune_history.append(
                    {
                        "iter": int(iteration),
                        "pruned": int(prune_step_count),
                        "child_dead": int(prune_child_dead_count),
                        "child_spike": int(prune_child_spike_count),
                        "remaining": int(student.get_xyz.shape[0]),
                    }
                )

        row = {
            "iter": int(iteration),
            "phase": str(current_phase_name),
            "loss": float(loss.detach().item()),
            "lr_rgb": float(loss_lr_rgb.detach().item()),
            "anchor_rgb": float(loss_anchor_rgb.detach().item()),
            "sr_l1": float(loss_sr_l1.detach().item()),
            "sr_hf": float(loss_sr_hf.detach().item()),
            "sr_prior_scale": float(sr_prior_scale),
            "sr_valid_ratio": float(sr_valid_ratio_step),
            "sr_mask_pixels": float(sr_mask_pixels_step),
            "sr_mask_mean": float(sr_mask_mean_step),
            "sr_loss_active": int(sr_loss_active_step),
            "premul_hf_excess": float(loss_premul_hf_excess.detach().item()),
            "mip_hr_lowfreq": float(loss_mip_hr_lowfreq.detach().item()),
            "mip_closure_alpha": float(loss_mip_closure_alpha.detach().item()),
            "mip_closure_premul": float(loss_mip_closure_premul.detach().item()),
            "mip_closure_depth": float(loss_mip_closure_depth.detach().item()),
            "mip_closure_alpha_over": float(loss_mip_closure_alpha_over.detach().item()),
            "mip_closure_premul_over": float(loss_mip_closure_premul_over.detach().item()),
            "xyz_anchor": float(loss_xyz_anchor.detach().item()),
            "opacity_anchor": float(loss_opacity_anchor.detach().item()),
            "scale_anchor": float(loss_scale_anchor.detach().item()),
            "dc_anchor": float(loss_dc_anchor.detach().item()),
            "rest_anchor": float(loss_rest_anchor.detach().item()),
            "risk_opacity": float(loss_risk_opacity.detach().item()),
            "risk_scale": float(loss_risk_scale.detach().item()),
            "risk_rest": float(loss_risk_rest.detach().item()),
            "sr_risk_opacity": float(loss_sr_risk_opacity.detach().item()),
            "sr_risk_scale": float(loss_sr_risk_scale.detach().item()),
            "sr_risk_rest": float(loss_sr_risk_rest.detach().item()),
            "reparam_mass_cap": float(loss_reparam_mass_cap.detach().item()),
            "reparam_child_tau_cap": float(loss_reparam_child_tau_cap.detach().item()),
            "geometry_parent_tau_brake": float(loss_geometry_parent_tau_brake.detach().item()),
            "reparam_mass_over_groups": int(reparam_mass_stats["over_groups"]),
            "reparam_mass_overflow_mean": float(reparam_mass_stats["overflow_mean"]),
            "reparam_mass_overflow_max": float(reparam_mass_stats["overflow_max"]),
            "reparam_child_over": int(reparam_child_stats["over_children"]),
            "reparam_child_overflow_mean": float(reparam_child_stats["overflow_mean"]),
            "reparam_child_overflow_max": float(reparam_child_stats["overflow_max"]),
            "geometry_parent_over": int(geometry_parent_stats["over_parents"]),
            "geometry_parent_overflow_mean": float(geometry_parent_stats["overflow_mean"]),
            "geometry_parent_overflow_max": float(geometry_parent_stats["overflow_max"]),
            "grad_xyz": float(grad_xyz),
            "grad_opacity": float(grad_opacity),
            "grad_scale": float(grad_scale),
            "grad_dc": float(grad_dc),
            "grad_rest": float(grad_rest),
            "star_release": float(star_release_status["release"]),
            "star_rest_scale": float(star_release_status["rest_scale"]),
            "star_tau_scale": float(star_release_status["tau_scale"]),
            "star_selected_count": int(star_release_status["selected_count"]),
            "reparam_pruned": int(prune_step_count),
            "reparam_prune_child_dead": int(prune_child_dead_count),
            "reparam_prune_child_spike": int(prune_child_spike_count),
            "gaussian_count": int(student.get_xyz.shape[0]),
        }
        log_rows.append(row)
        if iteration % 10 == 0:
            progress.set_postfix(
                {
                    "ph": str(current_phase_name),
                    "loss": f"{row['loss']:.5f}",
                    "lr": f"{row['lr_rgb']:.4f}",
                    "anc": f"{row['anchor_rgb']:.4f}",
                    "sr": f"{row['sr_l1']:.4f}/{row['sr_hf']:.4f}",
                    "srs": f"{row['sr_prior_scale']:.2f}",
                    "mask": f"{row['sr_mask_mean']:.3f}",
                    "hfex": f"{row['premul_hf_excess']:.4f}",
                    "gR": f"{row['grad_rest']:.2e}",
                    "miphr": f"{row['mip_hr_lowfreq']:.4f}",
                    "mcl": f"{row['mip_closure_alpha']:.4f}/{row['mip_closure_premul']:.4f}",
                    "mcl+": f"{row['mip_closure_alpha_over']:.4f}/{row['mip_closure_premul_over']:.4f}",
                    "mass": f"{row['reparam_mass_cap']:.4f}",
                    "ctau": f"{row['reparam_child_tau_cap']:.4f}",
                    "pr": f"{row['reparam_pruned']}",
                    "star": f"{row['star_release']:.2f}",
                    "xyz": f"{row['xyz_anchor']:.6f}",
                }
            )

        if int(args.save_every) > 0 and iteration % int(args.save_every) == 0:
            point_dir = output_model_path / "point_cloud" / f"iteration_{int(start_iter) + iteration}"
            mkdir_p(str(point_dir))
            student.save_ply(str(point_dir / "point_cloud.ply"))
            student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    star_release_status = apply_star_release(student, star_release_state, int(args.iterations))
    if bool(int(args.recompute_filter3d)):
        student.compute_3D_filter(filter_cameras, CUDA=False)

    output_model_path.mkdir(parents=True, exist_ok=True)
    copy_render_config(start_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(output_iteration)}"
    mkdir_p(str(point_dir))
    student.save_ply(str(point_dir / "point_cloud.ply"))
    student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    after_rgb = evaluate_rgb_losses(cameras, student, background, anchor_cache=anchor_cache)
    if mip_hr_anchor_cache:
        after_sr = evaluate_mip_hr_anchor_losses(
            sr_cameras,
            sr_cache,
            mip_hr_anchor_cache,
            student,
            background,
            residual_anchor_mode=str(args.sr_residual_anchor),
            min_pixels=float(args.sr_prior_min_pixels),
            min_valid_ratio=float(args.sr_prior_min_valid_ratio),
            prior_delta_clip=float(args.sr_prior_delta_clip),
            disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
            lowfreq_kernel=int(args.mip_hr_lowfreq_kernel),
            lowfreq_consistency_threshold=float(args.mip_hr_lowfreq_consistency_threshold),
            lowfreq_mask_floor=float(args.mip_hr_lowfreq_mask_floor),
        )
    else:
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
    after_premul_hf_excess = evaluate_premul_hf_excess_losses(
        sr_cameras,
        hf_excess_cache,
        student,
        background,
        kernel_size=int(args.premul_hf_excess_kernel),
        excess_ratio=float(args.premul_hf_excess_ratio),
        margin=float(args.premul_hf_excess_margin),
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
    ) if hf_excess_cache else {"premul_hf_excess": 0.0}
    after_mip_closure = evaluate_mip_closure_losses(
        mip_closure_cameras,
        mip_closure_cache,
        student,
        background,
        splat_args=mip_closure_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.mip_closure_min_pixels),
        depth_relative_min=float(args.mip_closure_depth_relative_min),
        charbonnier_eps=float(args.mip_closure_charbonnier_eps),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    ) if mip_closure_cache and mip_closure_splat_args is not None else {"alpha": 0.0, "premul": 0.0, "depth": 0.0}
    after_mip_closure_over = evaluate_mip_closure_over_losses(
        mip_closure_cameras,
        mip_closure_cache,
        student,
        background,
        splat_args=mip_closure_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.mip_closure_min_pixels),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    ) if mip_closure_cache and mip_closure_splat_args is not None else {"alpha_over": 0.0, "premul_luma_over": 0.0}
    displacement = torch.linalg.norm(student._xyz.detach() - xyz_init, dim=1)
    parameter_delta = {
        "xyz_mean": float(displacement.mean().item()),
        "xyz_max": float(displacement.max().item()),
        "opacity_mean_abs": mean_abs_delta(student.get_opacity, opacity_init),
        "scale_mean_abs": mean_abs_delta(student.get_scaling, scale_init),
        "dc_mean_abs": mean_abs_delta(student._features_dc, dc_init),
        "rest_mean_abs": mean_abs_delta(student._features_rest, rest_init),
        "rest_max_abs": float(torch.max(torch.abs(student._features_rest.detach() - rest_init.detach())).item()),
    }
    final_reparam_mass_cap, final_reparam_mass_stats = compute_reparam_mass_cap_loss(
        student,
        reparam_settle_state,
        eps=float(args.reparam_mass_cap_eps),
    )
    final_reparam_child_tau_cap, final_reparam_child_stats = compute_reparam_child_tau_cap_loss(
        student,
        reparam_settle_state,
        tau_scale=float(args.reparam_child_tau_cap_scale),
        tau_floor=float(args.reparam_child_tau_cap_abs),
    )
    final_geometry_parent_tau_brake, final_geometry_parent_stats = compute_geometry_parent_tau_brake_loss(
        student,
        reparam_settle_state,
        tau_scale=float(args.geometry_parent_tau_scale),
    )
    summary = {
        "version": "recover_cleaned_mip_lr_v0",
        "scene_root": str(scene_root),
        "start_model_path": str(start_model_path),
        "anchor_model_path": str(anchor_model_path) if anchor_model_path is not None else None,
        "output_model_path": str(output_model_path),
        "start_iteration": int(start_iter),
        "anchor_iteration": None if anchor_iter is None else int(anchor_iter),
        "output_iteration": int(output_iteration),
        "images_subdir": str(args.images_subdir),
        "selected_views": [str(cam.image_name) for cam in cameras],
        "iterations": int(args.iterations),
        "phase": {
            "mode": str(args.phase_mode),
            "block_steps": int(args.phase_block_steps),
            "initial_phase": str(phase_name),
        },
        "train_opacity": bool(args.enable_opacity_update),
        "train_scale": bool(args.enable_scale_update),
        "train_dc": bool(args.enable_dc_update),
        "train_rest": bool(args.enable_rest_update),
        "filter3d": {
            "before_train": bool(int(args.recompute_filter3d_before_train)),
            "final": bool(int(args.recompute_filter3d)),
        },
        "max_displacement": float(max_displacement),
        "loss_weights": {
            "lr_rgb": float(args.lambda_lr_rgb),
            "anchor_rgb": float(args.lambda_anchor_rgb),
            "xyz_anchor": float(args.lambda_xyz_anchor),
            "opacity_anchor": float(args.lambda_opacity_anchor),
            "scale_anchor": float(args.lambda_scale_anchor),
            "dc_anchor": float(args.lambda_dc_anchor),
            "rest_anchor": float(args.lambda_rest_anchor),
            "risk_opacity": float(args.lambda_risk_opacity),
            "risk_scale": float(args.lambda_risk_scale),
            "risk_rest": float(args.lambda_risk_rest),
            "sr_risk_opacity": float(args.lambda_sr_risk_opacity),
            "sr_risk_scale": float(args.lambda_sr_risk_scale),
            "sr_risk_rest": float(args.lambda_sr_risk_rest),
            "sr_prior_l1": float(args.lambda_sr_prior_l1),
            "sr_prior_hf": float(args.lambda_sr_prior_hf),
            "premul_hf_excess": float(args.lambda_premul_hf_excess),
            "mip_hr_lowfreq": float(args.lambda_mip_hr_lowfreq),
            "mip_closure_alpha": float(args.lambda_mip_closure_alpha),
            "mip_closure_premul": float(args.lambda_mip_closure_premul),
            "mip_closure_depth": float(args.lambda_mip_closure_depth),
            "mip_closure_alpha_over": float(args.lambda_mip_closure_alpha_over),
            "mip_closure_premul_over": float(args.lambda_mip_closure_premul_over),
            "reparam_mass_cap": float(args.lambda_reparam_mass_cap),
            "reparam_child_tau_cap": float(args.lambda_reparam_child_tau_cap),
            "geometry_parent_tau_brake": float(args.lambda_geometry_parent_tau_brake),
        },
        "prior_inputs": {
            "sr_prior_root": str(sr_prior_root) if sr_prior_root is not None else None,
            "sr_prior_dir": str(sr_prior_dir) if sr_prior_dir is not None else None,
            "sr_prior_mask_dir": str(sr_prior_mask_dir) if sr_prior_mask_dir is not None else None,
            "sr_anchor_dir": str(sr_anchor_dir) if sr_anchor_dir is not None else None,
            "sr_prior_subdir": str(args.sr_prior_subdir),
            "sr_prior_mask_subdir": str(args.sr_prior_mask_subdir),
            "sr_anchor_subdir": str(args.sr_anchor_subdir),
            "sr_images_subdir": str(sr_images_subdir),
            "sr_view_mode": str(args.sr_view_mode),
            "mip_hr_anchor_model_path": str(anchor_model_path) if use_mip_hr_anchor and anchor_model_path is not None else None,
            "sr_selected_views": [str(cam.image_name) for cam in sr_cameras],
            "mip_closure_model_path": str(mip_closure_model_path) if mip_closure_model_path is not None else None,
            "mip_closure_iteration": None if mip_closure_iter is None else int(mip_closure_iter),
            "mip_closure_images_subdir": str(mip_closure_images_subdir),
            "mip_closure_selected_views": [str(cam.image_name) for cam in mip_closure_cameras],
        },
        "gaussian_update_mask": {
            "payload": str(optimize_gaussian_mask_payload_path) if optimize_gaussian_mask_payload_path is not None else None,
            "key": str(args.optimize_gaussian_mask_key),
            "selected": int(gaussian_update_mask.sum().item()) if gaussian_update_mask is not None else None,
            "total": int(gaussian_update_mask.shape[0]) if gaussian_update_mask is not None else int(student.get_xyz.shape[0]),
            "update_scale": float(args.gaussian_update_scale),
            "scale_axis_mode": str(args.gaussian_scale_axis_mode),
        },
        "prior_prefilter": None if prior_prefilter_state is None else {
            "payload": str(prior_prefilter_save_path) if prior_prefilter_save_path is not None else None,
            **prior_prefilter_state["summary"],
        },
        "prior_config": {
            "sr_residual_anchor": str(args.sr_residual_anchor),
            "premul_hf_excess_kernel": int(args.premul_hf_excess_kernel),
            "premul_hf_excess_ratio": float(args.premul_hf_excess_ratio),
            "premul_hf_excess_margin": float(args.premul_hf_excess_margin),
            "mip_hr_lowfreq_kernel": int(args.mip_hr_lowfreq_kernel),
            "mip_hr_lowfreq_consistency_threshold": float(args.mip_hr_lowfreq_consistency_threshold),
            "mip_hr_lowfreq_mask_floor": float(args.mip_hr_lowfreq_mask_floor),
            "mip_closure_kernel": int(args.mip_closure_kernel),
            "mip_closure_alpha_threshold": float(args.mip_closure_alpha_threshold),
            "mip_closure_reference_lowpass": bool(int(args.mip_closure_reference_lowpass)),
            "mip_closure_min_pixels": float(args.mip_closure_min_pixels),
            "mip_closure_depth_relative_min": float(args.mip_closure_depth_relative_min),
            "sr_prior_mask_floor": float(args.sr_prior_mask_floor),
            "sr_prior_consistency_threshold": float(args.sr_prior_consistency_threshold),
            "sr_prior_min_valid_ratio": float(args.sr_prior_min_valid_ratio),
            "sr_prior_min_pixels": float(args.sr_prior_min_pixels),
            "sr_prior_delta_clip": float(args.sr_prior_delta_clip),
            "disable_sr_prior_hf_residual": bool(args.disable_sr_prior_hf_residual),
            "sr_prior_warmup_start_iter": int(args.sr_prior_warmup_start_iter),
            "sr_prior_warmup_end_iter": int(args.sr_prior_warmup_end_iter),
            "sr_prior_start_scale": float(args.sr_prior_start_scale),
            "sr_prior_end_scale": float(args.sr_prior_end_scale),
            "sr_prior_update_scale": float(args.sr_prior_update_scale),
            "sr_prior_schedule_mode": str(args.sr_prior_schedule_mode),
        },
        "star_release": {
            "payload_path": str(star_quarantine_payload_path) if star_quarantine_payload_path is not None else None,
            "selected_count": int(star_release_status["selected_count"]),
            "enabled": bool(star_release_state is not None and star_release_state.get("enabled", False)),
            "release_start_iter": int(args.star_release_start_iter),
            "release_end_iter": int(args.star_release_end_iter),
            "release_mode": str(args.star_release_mode),
            "release_opacity": bool(args.star_release_opacity),
            "rest_start_scale": None if star_release_state is None else float(star_release_state.get("rest_start_scale", 1.0)),
            "rest_end_scale": None if star_release_state is None else float(star_release_state.get("rest_end_scale", 1.0)),
            "tau_start_scale": None if star_release_state is None else float(star_release_state.get("tau_start_scale", 1.0)),
            "tau_end_scale": None if star_release_state is None else float(star_release_state.get("tau_end_scale", 1.0)),
            "final_release": float(star_release_status["release"]),
            "final_rest_scale": float(star_release_status["rest_scale"]),
            "final_tau_scale": float(star_release_status["tau_scale"]),
        },
        "reparam_settle": {
            "enabled": bool(reparam_settle_state is not None and reparam_settle_state.get("enabled", False)),
            "output_source_idx_path": None if reparam_settle_state is None else reparam_settle_state.get("output_source_idx_path"),
            "parent_mask_path": None if reparam_settle_state is None else reparam_settle_state.get("parent_mask_path"),
            "child_mask_path": None if reparam_settle_state is None else reparam_settle_state.get("child_mask_path"),
            "geometry_parent_mask_path": None if reparam_settle_state is None else reparam_settle_state.get("geometry_parent_mask_path"),
            "group_count": 0 if reparam_settle_state is None else int(reparam_settle_state.get("group_count", 0)),
            "parent_count": 0 if reparam_settle_state is None else int(reparam_settle_state.get("parent_count", 0)),
            "child_count": 0 if reparam_settle_state is None else int(reparam_settle_state.get("child_count", 0)),
            "geometry_parent_count": 0 if reparam_settle_state is None else int(reparam_settle_state.get("geometry_parent_count", 0)),
            "config": {
                "mass_cap_eps": float(args.reparam_mass_cap_eps),
                "child_tau_cap_scale": float(args.reparam_child_tau_cap_scale),
                "child_tau_cap_abs": float(args.reparam_child_tau_cap_abs),
                "geometry_parent_tau_scale": float(args.geometry_parent_tau_scale),
            },
            "prune": {
                "enabled": bool(use_reparam_prune),
                "iters": sorted(reparam_prune_iters),
                "child_dead_tau": float(args.reparam_prune_child_dead_tau),
                "child_spike_tau_scale": float(args.reparam_prune_child_spike_tau_scale),
                "child_spike_tau_abs": float(args.reparam_prune_child_spike_tau_abs),
                "child_spike_anisotropy": float(args.reparam_prune_child_spike_anisotropy),
                "child_spike_risk_min": float(args.reparam_prune_child_spike_risk_min),
                "history": prune_history,
            },
            "final_losses": {
                "mass_cap": float(final_reparam_mass_cap.detach().item()),
                "child_tau_cap": float(final_reparam_child_tau_cap.detach().item()),
                "geometry_parent_tau_brake": float(final_geometry_parent_tau_brake.detach().item()),
            },
            "final_stats": {
                "mass_cap": final_reparam_mass_stats,
                "child_tau_cap": final_reparam_child_stats,
                "geometry_parent_tau_brake": final_geometry_parent_stats,
            },
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
                "sr_risk_boost": float(args.sr_risk_boost),
            },
            "boosted_mean": float(prior_risk_weight.mean().item()),
            "boosted_p90": float(torch.quantile(prior_risk_weight, 0.90).item()),
            "boosted_p99": float(torch.quantile(prior_risk_weight, 0.99).item()),
        },
        "sr_prior_cache": summarize_sr_cache(sr_cache),
        "before_rgb": before_rgb,
        "after_rgb": after_rgb,
        "before_sr": before_sr,
        "after_sr": after_sr,
        "before_premul_hf_excess": before_premul_hf_excess,
        "after_premul_hf_excess": after_premul_hf_excess,
        "before_mip_closure": before_mip_closure,
        "after_mip_closure": after_mip_closure,
        "before_mip_closure_over": before_mip_closure_over,
        "after_mip_closure_over": after_mip_closure_over,
        "displacement": stats_from_array(displacement.detach().cpu().numpy().astype(np.float32)),
        "parameter_delta": parameter_delta,
        "log_tail": log_rows[-50:],
    }
    summary_path = output_model_path / "recover_cleaned_mip_lr_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] output model: {output_model_path}")
    print(f"[done] output ply  : {point_dir / 'point_cloud.ply'}")
    print(f"[done] summary     : {summary_path}")


if __name__ == "__main__":
    main()
