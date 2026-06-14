#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
import json
import torch.nn.functional as F
from pathlib import Path
from random import randint
from scipy.spatial import cKDTree
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, SplattingSettings, OptimizationParams, SplattingSettings, MeshingParams
from utils.depth_utils import depths_to_points, depth_to_normal, central_diff
from utils.vis_utils import gui_visualize, export_image
from scene.gaussian_model import build_scaling_rotation
from diff_gaussian_rasterization import ExtendedSettings, DebugVisualization
import matplotlib
import numpy as np
import trimesh
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
import matplotlib
from scene.densifier import AbsGradDensifier, MCMCDensifier
from scene.gaussian_model import GaussianSourceTag
from utils.general_utils import build_rotation
from utils.prior_injection import index_image_dir, load_rgb_image, load_mask, normalize_image_name
from utils.route_executor import (
    load_route_payload as load_gaussian_route_payload,
    resolve_route_runtime_state as resolve_gaussian_route_runtime_state,
)
from utils.training_diagnostics import (
    DiagnosticBasisProvider,
    GaussianGradientTracker,
    TwoDDropoutGradientDiagnostic,
    compute_training_phase,
    export_diagnostic_bundle,
    resolve_diagnostic_start_iter,
)

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    fused_ssim = None
    FUSED_SSIM_AVAILABLE = False

RED = '\033[31m'
RESET = '\033[0m'

_FUSED_SSIM_WARNING_SHOWN = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if lr_init == 0:
            return 0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return (delay_rate * log_lerp)

    return helper


def compute_ssim_loss(image, gt_image):
    global _FUSED_SSIM_WARNING_SHOWN

    image_batched = image.unsqueeze(0)
    gt_batched = gt_image.unsqueeze(0)
    if FUSED_SSIM_AVAILABLE:
        return 1.0 - fused_ssim(image_batched, gt_batched)

    if not _FUSED_SSIM_WARNING_SHOWN:
        print(
            f"{RED}[WARN]{RESET} fused_ssim is not installed; "
            "falling back to utils.loss_utils.ssim."
        )
        _FUSED_SSIM_WARNING_SHOWN = True
    return 1.0 - ssim(image_batched, gt_batched)


def _resize_hw3_tensor(image_hw3: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    image = image_hw3.permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    resized = F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).detach().cpu()


def _resize_mask_tensor(mask_hw: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    mask = mask_hw[None, None].to(device="cuda", dtype=torch.float32)
    resized = F.interpolate(mask, size=(target_height, target_width), mode="nearest")
    return resized[0, 0].detach().cpu()


def load_external_rgb_cached(image_name, index, cache, height, width):
    if index is None:
        return None
    key = (image_name, height, width)
    if key not in cache:
        path = lookup_indexed_image_path(index, image_name)
        if path is None:
            cache[key] = None
        else:
            image = load_rgb_image(path)
            if image.shape[0] != height or image.shape[1] != width:
                image = _resize_hw3_tensor(image, target_height=height, target_width=width)
            cache[key] = image
    image = cache[key]
    if image is None:
        return None
    return image.to(device="cuda", dtype=torch.float32)


def lookup_indexed_image_path(index, image_name):
    if index is None:
        return None
    candidates = [
        image_name,
        normalize_image_name(image_name),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    for key in candidates:
        path = index.get(str(key))
        if path is not None:
            return path
    return None


def has_external_mask(image_name, mask_dir, suffix: str = "_inject.png") -> bool:
    if mask_dir is None:
        return False
    stem = normalize_image_name(image_name)
    candidates = [
        os.path.join(mask_dir, f"{image_name}{suffix}"),
        os.path.join(mask_dir, f"{stem}{suffix}"),
    ]
    return any(os.path.exists(path) for path in candidates)


def load_external_mask_cached(image_name, mask_dir, cache, height, width, suffix: str = "_inject.png"):
    if mask_dir is None:
        return None
    key = (image_name, height, width, suffix)
    if key not in cache:
        stem = normalize_image_name(image_name)
        candidates = [
            os.path.join(mask_dir, f"{image_name}{suffix}"),
            os.path.join(mask_dir, f"{stem}{suffix}"),
        ]
        path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
        if path is None:
            cache[key] = None
        else:
            mask = load_mask(Path(path))
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = _resize_mask_tensor(mask, target_height=height, target_width=width)
            cache[key] = mask
    mask = cache[key]
    if mask is None:
        return None
    return (mask.to(device="cuda", dtype=torch.float32) > 0.5)


def compute_masked_l1_loss(image, target, mask):
    if mask is None:
        return None
    mask = mask.to(dtype=image.dtype, device=image.device)
    active = float(mask.sum().item())
    if active <= 0:
        return None
    mask3 = mask.unsqueeze(0)
    return (torch.abs(image - target) * mask3).sum() / (mask3.sum() * image.shape[0]).clamp_min(1.0)


def _odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def blur_image_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _odd_kernel_size(kernel_size)
    if kernel_size <= 1:
        return image
    pad = kernel_size // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0]


def compute_weighted_l1_loss(image: torch.Tensor, target: torch.Tensor, weight_hw: torch.Tensor):
    weight_hw = weight_hw.to(dtype=image.dtype, device=image.device)
    active = weight_hw.sum()
    if float(active.item()) <= 0:
        return None
    return (torch.abs(image - target) * weight_hw.unsqueeze(0)).sum() / (active * image.shape[0]).clamp_min(1.0)


def compute_weighted_gradient_l1_loss(image: torch.Tensor, target: torch.Tensor, weight_hw: torch.Tensor):
    weight_hw = weight_hw.to(dtype=image.dtype, device=image.device)
    dx_image = image[:, :, 1:] - image[:, :, :-1]
    dx_target = target[:, :, 1:] - target[:, :, :-1]
    dy_image = image[:, 1:, :] - image[:, :-1, :]
    dy_target = target[:, 1:, :] - target[:, :-1, :]
    weight_x = 0.5 * (weight_hw[:, 1:] + weight_hw[:, :-1])
    weight_y = 0.5 * (weight_hw[1:, :] + weight_hw[:-1, :])
    loss_x = compute_weighted_l1_loss(dx_image, dx_target, weight_x)
    loss_y = compute_weighted_l1_loss(dy_image, dy_target, weight_y)
    if loss_x is None and loss_y is None:
        return None
    if loss_x is None:
        return loss_y
    if loss_y is None:
        return loss_x
    return 0.5 * (loss_x + loss_y)


def get_prior_edge_detail_alpha(runtime_args, iteration: int, train_start_iter: int) -> float:
    start = float(runtime_args.prior_edge_detail_alpha)
    final = float(runtime_args.prior_edge_detail_alpha_final)
    if final < 0.0:
        return start
    warmup = max(int(runtime_args.prior_edge_detail_warmup_iters), 0)
    if warmup <= 0:
        return final
    t = max(0.0, min(1.0, float(iteration - train_start_iter) / float(warmup)))
    return start + (final - start) * t


def compute_prior_edge_detail_loss(
    image: torch.Tensor,
    prior_target: torch.Tensor,
    image_mask: torch.Tensor,
    runtime_args,
    detail_alpha: float,
    lowfreq_anchor: torch.Tensor = None,
):
    if image_mask is None:
        return None
    image_mask = image_mask.to(dtype=image.dtype, device=image.device)
    if float(image_mask.sum().item()) <= 0:
        return None

    blur_kernel = int(runtime_args.prior_edge_detail_blur_kernel)
    image_low = blur_image_chw(image, blur_kernel)
    prior_low = blur_image_chw(prior_target, blur_kernel)
    anchor = image.detach() if lowfreq_anchor is None else lowfreq_anchor.detach()
    anchor_low = blur_image_chw(anchor, blur_kernel)

    image_high = image - image_low
    prior_high = prior_target - prior_low
    anchor_high = image.detach() - blur_image_chw(image.detach(), blur_kernel)

    lowfreq_diff = torch.abs(prior_low - anchor_low).mean(dim=0)
    lowfreq_threshold = float(runtime_args.prior_edge_lowfreq_threshold)
    if lowfreq_threshold > 0.0:
        confidence = torch.clamp(1.0 - lowfreq_diff / lowfreq_threshold, min=0.0, max=1.0)
    else:
        confidence = torch.ones_like(image_mask)

    detail_min_gain = float(runtime_args.prior_edge_detail_min_gain)
    if detail_min_gain > 0.0:
        prior_detail = torch.abs(prior_high).mean(dim=0)
        current_detail = torch.abs(anchor_high).mean(dim=0)
        detail_confidence = torch.clamp(
            (prior_detail - current_detail - detail_min_gain) / detail_min_gain,
            min=0.0,
            max=1.0,
        )
        confidence = confidence * detail_confidence

    confidence_power = float(runtime_args.prior_edge_confidence_power)
    if confidence_power != 1.0:
        confidence = torch.clamp(confidence, min=0.0, max=1.0).pow(confidence_power)
    confidence = confidence * image_mask

    if float(confidence.sum().item()) < float(runtime_args.prior_edge_min_pixels):
        return None

    detail_alpha = max(0.0, min(1.0, float(detail_alpha)))
    high_target = (1.0 - detail_alpha) * anchor_high + detail_alpha * prior_high
    detail_loss = compute_weighted_l1_loss(image_high, high_target.detach(), confidence)
    if detail_loss is None:
        return None

    total_loss = float(runtime_args.prior_edge_detail_weight) * detail_loss

    lowfreq_weight = float(runtime_args.prior_edge_lowfreq_weight)
    if lowfreq_weight > 0.0:
        low_loss = compute_weighted_l1_loss(image_low, anchor_low.detach(), image_mask)
        if low_loss is not None:
            total_loss = total_loss + lowfreq_weight * low_loss

    grad_weight = float(runtime_args.prior_edge_grad_weight)
    if grad_weight > 0.0:
        grad_target = anchor + detail_alpha * prior_high
        grad_loss = compute_weighted_gradient_l1_loss(image, grad_target.detach(), confidence)
        if grad_loss is not None:
            total_loss = total_loss + grad_weight * grad_loss

    return total_loss


def _dilate_binary_mask(mask_hw: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return mask_hw.to(dtype=torch.bool)
    mask = mask_hw.to(device="cuda", dtype=torch.float32)[None, None]
    kernel = int(radius_px) * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=int(radius_px))
    return dilated[0, 0] > 0.5


def build_projected_gaussian_touch_mask(
    viewpoint_cam,
    gaussians,
    image_mask,
    visibility_filter=None,
    radii=None,
    depth_min: float = 1e-6,
    min_touch_radius_px: float = 2.0,
    radius_scale: float = 0.5,
    max_touch_radius_px: float = 16.0,
):
    if image_mask is None:
        return None
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
    mask_values = image_mask_bool[yi[valid_idx], xi[valid_idx]]
    touched[valid_idx] = mask_values

    # Edge masks are thin by construction; center-only tests miss Gaussians whose
    # splat footprint overlaps the edge band but whose center sits just outside it.
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


def build_source_tag_train_mask(gaussians, optimize_source_tag: str):
    if optimize_source_tag == "all":
        return None
    source_tag = gaussians._source_tag
    if optimize_source_tag == "prior":
        return source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    if optimize_source_tag == "probe":
        return source_tag == int(GaussianSourceTag.EXTENSION_PROBE)
    if optimize_source_tag == "added":
        return source_tag != int(GaussianSourceTag.ORIGINAL)
    raise ValueError(f"Unsupported optimize_source_tag: {optimize_source_tag}")


def load_gaussian_update_mask_payload(path: str, key: str, total_gaussians: int) -> torch.Tensor:
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        mask = payload.reshape(-1)
        if mask.shape[0] == int(total_gaussians):
            return mask.to(device="cuda", dtype=torch.bool)
        if mask.ndim == 1 and mask.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
            ids = mask.to(dtype=torch.int64)
            out = torch.zeros((int(total_gaussians),), dtype=torch.bool)
            out[ids] = True
            return out.to(device="cuda")
        raise ValueError(
            f"Raw tensor Gaussian mask payload has unsupported shape: {tuple(mask.shape)} "
            f"vs total_gaussians={total_gaussians}"
        )
    if key in payload:
        mask = payload[key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.reshape(-1)
        if mask.shape[0] != int(total_gaussians):
            raise ValueError(
                f"Gaussian update mask '{key}' length mismatch: "
                f"{tuple(mask.shape)} vs total_gaussians={total_gaussians}"
            )
        return mask.to(device="cuda", dtype=torch.bool)
    if "selected_ids" in payload:
        ids = payload["selected_ids"]
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        ids = ids.to(dtype=torch.int64)
        mask = torch.zeros((int(total_gaussians),), dtype=torch.bool)
        mask[ids] = True
        return mask.to(device="cuda")
    raise KeyError(f"Mask payload must contain '{key}' or 'selected_ids': {path}")


def _unwrap_gaussian_action_payload(payload):
    if "update_strength" in payload:
        return payload
    if "gs_action_payload" in payload and isinstance(payload["gs_action_payload"], dict):
        return payload["gs_action_payload"]
    if "hrgs_outputs" in payload:
        nested = payload["hrgs_outputs"]
        if isinstance(nested, dict) and "gs_action_payload" in nested and isinstance(nested["gs_action_payload"], dict):
            return nested["gs_action_payload"]
    raise KeyError(
        "Gaussian action payload must contain 'update_strength' directly "
        "or a nested 'gs_action_payload' dictionary."
    )


def _as_gaussian_action_vector(value, key: str, total_gaussians: int, default_value: float | None = None) -> torch.Tensor:
    if value is None:
        if default_value is None:
            raise KeyError(f"Missing Gaussian action payload key: {key}")
        return torch.full((int(total_gaussians),), float(default_value), dtype=torch.float32)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().to(dtype=torch.float32).reshape(-1)
    if value.shape[0] != int(total_gaussians):
        raise ValueError(
            f"Gaussian action vector '{key}' length mismatch: "
            f"{value.shape[0]} vs total_gaussians={total_gaussians}"
        )
    return value.cpu()


def load_gaussian_action_payload(path: str, total_gaussians: int):
    if not path:
        return None
    payload_path = Path(path)
    suffix = payload_path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(payload_path)
        raw_payload = {key: loaded[key] for key in loaded.files}
    else:
        raw_payload = torch.load(payload_path, map_location="cpu")
    if not isinstance(raw_payload, dict):
        raise TypeError(f"Unsupported Gaussian action payload object type: {type(raw_payload)!r}")

    payload = _unwrap_gaussian_action_payload(raw_payload)
    update_strength = _as_gaussian_action_vector(payload.get("update_strength"), "update_strength", total_gaussians)
    attach_strength = _as_gaussian_action_vector(
        payload.get("attach_strength"),
        "attach_strength",
        total_gaussians,
        default_value=1.0,
    )
    detail_weight = _as_gaussian_action_vector(
        payload.get("detail_weight"),
        "detail_weight",
        total_gaussians,
        default_value=1.0,
    )
    prior_color_strength = _as_gaussian_action_vector(
        payload.get("prior_color_strength"),
        "prior_color_strength",
        total_gaussians,
        default_value=1.0,
    )
    return {
        "update_strength": update_strength,
        "attach_strength": attach_strength,
        "detail_weight": detail_weight,
        "prior_color_strength": prior_color_strength,
    }


def _align_gaussian_action_vector(weight: torch.Tensor, total_gaussians: int, fill_value: float) -> torch.Tensor:
    weight = weight.to(device="cuda", dtype=torch.float32).reshape(-1)
    current = int(weight.shape[0])
    target = int(total_gaussians)
    if current == target:
        return weight
    if current > target:
        return weight[:target]
    pad = torch.full((target - current,), float(fill_value), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, pad), dim=0)


def resolve_gaussian_action_runtime_state(
    action_payload,
    total_gaussians: int,
    min_weight: float,
    attach_scale: float,
    detail_scale: float,
    prior_color_scale: float,
):
    if action_payload is None:
        return None
    update_strength = _align_gaussian_action_vector(action_payload["update_strength"], total_gaussians, fill_value=1.0)
    attach_strength = _align_gaussian_action_vector(action_payload["attach_strength"], total_gaussians, fill_value=1.0)
    detail_weight = _align_gaussian_action_vector(action_payload["detail_weight"], total_gaussians, fill_value=1.0)
    prior_color_strength = _align_gaussian_action_vector(
        action_payload["prior_color_strength"],
        total_gaussians,
        fill_value=1.0,
    )

    update_strength = torch.clamp(update_strength, 0.0, 1.0)
    attach_strength = torch.clamp(attach_strength * float(attach_scale), 0.0, 1.0)
    detail_weight = torch.clamp(detail_weight * float(detail_scale), 0.0, 1.0)
    prior_color_strength = torch.clamp(prior_color_strength * float(prior_color_scale), 0.0, 1.0)
    if float(min_weight) > 0.0:
        update_strength = torch.where(
            update_strength >= float(min_weight),
            update_strength,
            torch.zeros_like(update_strength),
        )

    active_mask = update_strength > 0.0
    geometry_weight = torch.clamp(update_strength * (0.5 + 0.5 * attach_strength), 0.0, 1.0)
    appearance_weight = torch.clamp(update_strength * torch.maximum(detail_weight, prior_color_strength), 0.0, 1.0)
    opacity_weight = update_strength
    return {
        "active_mask": active_mask,
        "update_strength": update_strength,
        "attach_strength": attach_strength,
        "detail_weight": detail_weight,
        "prior_color_strength": prior_color_strength,
        "geometry_weight": geometry_weight,
        "appearance_weight": appearance_weight,
        "opacity_weight": opacity_weight,
    }


def build_prior_edge_camera_pool(train_cameras, prior_index, mask_dir):
    cameras = []
    missing_prior = []
    missing_mask = []
    for camera in train_cameras:
        image_name = camera.image_name
        if lookup_indexed_image_path(prior_index, image_name) is None:
            if len(missing_prior) < 16:
                missing_prior.append(image_name)
            continue
        if not has_external_mask(image_name, mask_dir):
            if len(missing_mask) < 16:
                missing_mask.append(image_name)
            continue
        cameras.append(camera)
    return cameras, missing_prior, missing_mask


def combine_gaussian_update_masks(*masks):
    active_masks = [mask for mask in masks if mask is not None]
    if not active_masks:
        return None
    combined = active_masks[0].to(device="cuda", dtype=torch.bool)
    for mask in active_masks[1:]:
        mask = mask.to(device="cuda", dtype=torch.bool)
        if mask.shape[0] != combined.shape[0]:
            raise ValueError(f"Gaussian update mask length mismatch: {mask.shape[0]} vs {combined.shape[0]}")
        combined = combined & mask
    return combined


def _reshape_weight_for_tensor(weight: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return weight.view(weight.shape[0], *([1] * (tensor.ndim - 1)))


def apply_gaussian_update_mask(
    gaussians,
    update_mask: torch.Tensor,
    freeze_appearance: bool,
    update_scale: float = 1.0,
    geometry_weight: torch.Tensor = None,
    position_weight: torch.Tensor = None,
    scaling_weight: torch.Tensor = None,
    rotation_weight: torch.Tensor = None,
    appearance_weight: torch.Tensor = None,
    opacity_weight: torch.Tensor = None,
    freeze_geometry_mask: torch.Tensor = None,
):
    if (
        update_mask is None
        and geometry_weight is None
        and position_weight is None
        and scaling_weight is None
        and rotation_weight is None
        and appearance_weight is None
        and opacity_weight is None
        and freeze_geometry_mask is None
        and not freeze_appearance
    ):
        return

    optimizer = gaussians.optimizer
    gaussian_group_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
    total_gaussians = int(gaussians.get_xyz.shape[0])
    if position_weight is None:
        position_weight = geometry_weight
    if scaling_weight is None:
        scaling_weight = geometry_weight
    if rotation_weight is None:
        rotation_weight = geometry_weight

    if update_mask is not None:
        update_mask = update_mask.to(device="cuda", dtype=torch.bool)
        if update_mask.shape[0] != total_gaussians:
            raise ValueError(
                f"Gaussian update mask length mismatch: {update_mask.shape[0]} vs {total_gaussians}. "
                "Disable densification/pruning or regenerate the mask for this checkpoint."
            )
        inverse_mask = ~update_mask
    else:
        inverse_mask = None

    if freeze_geometry_mask is not None:
        freeze_geometry_mask = freeze_geometry_mask.to(device="cuda", dtype=torch.bool)
        if freeze_geometry_mask.shape[0] != total_gaussians:
            raise ValueError(
                f"Freeze-geometry mask length mismatch: {freeze_geometry_mask.shape[0]} vs {total_gaussians}. "
                "Regenerate the mask for this checkpoint or disable --freeze_geometry_mask_payload."
            )

    geometry_weight = None if geometry_weight is None else geometry_weight.to(device="cuda", dtype=torch.float32)
    position_weight = None if position_weight is None else position_weight.to(device="cuda", dtype=torch.float32)
    scaling_weight = None if scaling_weight is None else scaling_weight.to(device="cuda", dtype=torch.float32)
    rotation_weight = None if rotation_weight is None else rotation_weight.to(device="cuda", dtype=torch.float32)
    appearance_weight = None if appearance_weight is None else appearance_weight.to(device="cuda", dtype=torch.float32)
    opacity_weight = None if opacity_weight is None else opacity_weight.to(device="cuda", dtype=torch.float32)

    def _group_weight(name: str):
        if name == "xyz":
            return position_weight
        if name == "scaling":
            return scaling_weight
        if name == "rotation":
            return rotation_weight
        if name in {"f_dc", "f_rest"}:
            return appearance_weight
        if name == "opacity":
            return opacity_weight
        return None

    for group in optimizer.param_groups:
        name = group.get("name", "")
        group_weight = _group_weight(name)
        for param in group["params"]:
            if param.grad is not None:
                if name in gaussian_group_names and param.grad.shape[0] == total_gaussians:
                    if group_weight is not None:
                        param.grad.mul_(_reshape_weight_for_tensor(group_weight, param.grad))
                    if float(update_scale) != 1.0:
                        param.grad.mul_(float(update_scale))
                    if inverse_mask is not None:
                        param.grad[inverse_mask] = 0
                    if freeze_geometry_mask is not None and name in {"xyz", "scaling", "rotation"}:
                        param.grad[freeze_geometry_mask] = 0
                elif freeze_appearance and "appearance" in name:
                    param.grad.zero_()

            state = optimizer.state.get(param, None)
            if state is None:
                continue
            for value in state.values():
                if not torch.is_tensor(value):
                    continue
                if name in gaussian_group_names and value.ndim > 0 and value.shape[0] == total_gaussians:
                    if group_weight is not None:
                        value.mul_(_reshape_weight_for_tensor(group_weight, value))
                    if inverse_mask is not None:
                        value[inverse_mask] = 0
                    if freeze_geometry_mask is not None and name in {"xyz", "scaling", "rotation"}:
                        value[freeze_geometry_mask] = 0
                elif freeze_appearance and "appearance" in name:
                    value.zero_()


def _load_trimesh_surface(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh geometry found in scene: {mesh_path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh object type from {mesh_path}: {type(loaded)!r}")
    if loaded.vertices.shape[0] == 0 or loaded.faces.shape[0] == 0:
        raise ValueError(f"Mesh has no vertices/faces: {mesh_path}")
    return loaded


class MeshSurfaceThinningRegularizer:
    """Lightweight mesh-surface prior for LR cases where GS can grow into thick shells."""

    def __init__(self, args):
        self.enabled = float(args.lambda_surface_thin) > 0.0
        self.args = args
        self.tree = None
        self.surface_points_np = None
        self.surface_normals_np = None
        self.anchor_points = None
        self.anchor_normals = None
        self.anchor_count = 0
        self.last_update_iter = -1

        if not self.enabled:
            return
        if not args.surface_thin_mesh_path:
            raise ValueError("--lambda_surface_thin > 0 requires --surface_thin_mesh_path")

        mesh = _load_trimesh_surface(args.surface_thin_mesh_path)
        sample_count = max(1, int(args.surface_thin_sample_count))
        surface_points, face_ids = trimesh.sample.sample_surface(mesh, sample_count)
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        surface_normals = face_normals[np.asarray(face_ids, dtype=np.int64)]
        normal_norm = np.linalg.norm(surface_normals, axis=1, keepdims=True)
        surface_normals = surface_normals / np.maximum(normal_norm, 1e-8)

        self.surface_points_np = np.asarray(surface_points, dtype=np.float32)
        self.surface_normals_np = np.asarray(surface_normals, dtype=np.float32)
        self.tree = cKDTree(self.surface_points_np)
        print(
            "[surface-thin] loaded mesh surface prior: "
            f"samples={self.surface_points_np.shape[0]}, vertices={len(mesh.vertices)}, faces={len(mesh.faces)}"
        )

    def _active_at(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if iteration < int(self.args.surface_thin_from_iter):
            return False
        until_iter = int(self.args.surface_thin_until_iter)
        if until_iter > 0 and iteration > until_iter:
            return False
        return True

    def _refresh_anchors(self, gaussians, iteration: int):
        xyz_np = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        try:
            _, nearest_ids = self.tree.query(xyz_np, k=1, workers=-1)
        except TypeError:
            _, nearest_ids = self.tree.query(xyz_np, k=1)
        nearest_ids = np.asarray(nearest_ids, dtype=np.int64)
        self.anchor_points = torch.as_tensor(self.surface_points_np[nearest_ids], device="cuda", dtype=torch.float32)
        self.anchor_normals = torch.as_tensor(self.surface_normals_np[nearest_ids], device="cuda", dtype=torch.float32)
        self.anchor_count = int(xyz_np.shape[0])
        self.last_update_iter = int(iteration)
        print(f"[surface-thin] refreshed nearest mesh anchors for {self.anchor_count} GS at iter {iteration}")

    def loss(self, gaussians, iteration: int, point_weight: torch.Tensor = None):
        if not self._active_at(iteration):
            return None
        total = int(gaussians.get_xyz.shape[0])
        if total <= 0:
            return None
        needs_refresh = (
            self.anchor_points is None
            or self.anchor_count != total
            or (int(iteration) - self.last_update_iter) >= int(self.args.surface_thin_update_interval)
        )
        if needs_refresh:
            self._refresh_anchors(gaussians, iteration)

        sample_count = int(self.args.surface_thin_gaussian_sample_count)
        if sample_count > 0 and total > sample_count:
            ids = torch.randperm(total, device="cuda")[:sample_count]
        else:
            ids = torch.arange(total, device="cuda")

        xyz = gaussians.get_xyz[ids]
        anchor_points = self.anchor_points[ids].to(dtype=xyz.dtype)
        anchor_normals = self.anchor_normals[ids].to(dtype=xyz.dtype)

        sample_weight = None
        weight_norm = None
        if point_weight is not None:
            point_weight = point_weight.to(device="cuda", dtype=xyz.dtype).reshape(-1)
            if point_weight.shape[0] != total:
                raise ValueError(
                    f"surface thinning weight length mismatch: {point_weight.shape[0]} vs total_gaussians={total}"
                )
            sample_weight = point_weight[ids].clamp_min(0.0)
            weight_norm = sample_weight.sum().clamp_min(1e-6)

        def _weighted_mean(values: torch.Tensor) -> torch.Tensor:
            if sample_weight is None:
                return values.mean()
            return (values * sample_weight).sum() / weight_norm

        signed_offset = torch.sum((xyz - anchor_points) * anchor_normals, dim=-1)
        margin = float(self.args.surface_thin_offset_margin)
        offset_loss = _weighted_mean(torch.relu(torch.abs(signed_offset) - margin).pow(2))

        target = float(self.args.surface_thin_normal_scale_target)
        weight = float(self.args.surface_thin_normal_scale_weight)
        if target <= 0.0 or weight <= 0.0:
            return offset_loss

        rotations = build_rotation(gaussians.get_rotation[ids])
        local_normals = torch.bmm(rotations.transpose(1, 2), anchor_normals.unsqueeze(-1)).squeeze(-1)
        normal_extent = torch.sqrt(
            torch.sum((local_normals * gaussians.get_scaling[ids]).pow(2), dim=-1).clamp_min(1e-12)
        )
        normal_scale_loss = _weighted_mean(torch.relu(normal_extent - target).pow(2))
        return offset_loss + weight * normal_scale_loss

def training(dataset, opt, pipe : PipelineParams, mesh : MeshingParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, splat_args: ExtendedSettings, runtime_args):
    import time
    start_event = time.time()
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, splat_args, opt, pipe, mesh)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe.convert_SBs_python)
    scene = Scene(dataset, gaussians, MCMC_init=mesh.cap_max != -1)
    trainCameras = scene.getTrainCameras().copy() 
    if runtime_args.start_ply:
        start_ply_path = Path(runtime_args.start_ply).expanduser().resolve()
        if not start_ply_path.is_file():
            raise FileNotFoundError(f"start_ply not found: {start_ply_path}")
        gaussians.load_ply(str(start_ply_path))
        tags_path = start_ply_path.parent / "gaussian_tags.pt"
        if tags_path.is_file():
            gaussians.load_tracking_metadata(str(tags_path))
        else:
            gaussians.init_tracking_state(gaussians.get_xyz.shape[0])
        print(f"[start-ply] loaded initial Gaussians from {start_ply_path}")
    
    appearance_embedding = None
    if mesh.use_decoupled_appearance:
        appearance_embedding = AppearanceEmbedding(num_views=len(trainCameras))
    if mesh.use_pgsr_appearance:
        appearance_embedding = PGSREmbedding(num_views=len(trainCameras))
    
    gaussians.training_setup(opt, mesh, appearance_embedding)
    if checkpoint:
        checkpoint_blob = torch.load(checkpoint)
        if not isinstance(checkpoint_blob, (tuple, list)):
            raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint_blob)!r}")
        if len(checkpoint_blob) == 2:
            model_params, first_iter = checkpoint_blob
            _appearance_embedding, _appearance_net = (None, None)
        elif len(checkpoint_blob) == 3:
            model_params, first_iter, (_appearance_embedding, _appearance_net) = checkpoint_blob
        else:
            raise RuntimeError(f"Unsupported top-level checkpoint tuple length: {len(checkpoint_blob)}")
        if appearance_embedding is not None and _appearance_embedding is not None:
            appearance_embedding.restore(_appearance_embedding, _appearance_net)
        gaussians.restore(model_params, opt, mesh, appearance_embedding)
        

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if mesh.cap_max == -1:
        densifier = AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    else:
        densifier = MCMCDensifier(gaussians, opt, mesh, dataset, pipe)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx
        
    # at first, we don't need the opacity
    splat_args.render_opacity = False
    
    gaussians.compute_3D_filter(cameras=trainCameras, CUDA=not pipe.compute_filter3D_python)
    surface_thinning_regularizer = MeshSurfaceThinningRegularizer(runtime_args)
    global_image_index = index_image_dir(runtime_args.global_image_dir) if runtime_args.global_image_dir else None
    prior_local_index = index_image_dir(runtime_args.prior_local_dir) if runtime_args.prior_local_dir else None
    prior_local_mask_dir = runtime_args.prior_local_mask_dir if runtime_args.prior_local_mask_dir else None
    prior_edge_index = index_image_dir(runtime_args.prior_edge_dir) if runtime_args.prior_edge_dir else None
    prior_edge_mask_dir = runtime_args.prior_edge_mask_dir if runtime_args.prior_edge_mask_dir else None
    global_image_cache = {}
    prior_local_cache = {}
    prior_local_mask_cache = {}
    prior_edge_cache = {}
    prior_edge_mask_cache = {}
    external_update_mask = load_gaussian_update_mask_payload(
        runtime_args.optimize_gaussian_mask_payload,
        runtime_args.optimize_gaussian_mask_key,
        total_gaussians=gaussians.get_xyz.shape[0],
    )
    freeze_geometry_mask = load_gaussian_update_mask_payload(
        runtime_args.freeze_geometry_mask_payload,
        runtime_args.freeze_geometry_mask_key,
        total_gaussians=gaussians.get_xyz.shape[0],
    )
    gaussian_action_payload = load_gaussian_route_payload(
        runtime_args.gaussian_action_payload,
        total_gaussians=gaussians.get_xyz.shape[0],
    )
    if external_update_mask is not None:
        selected_count = int(external_update_mask.sum().item())
        print(
            f"[prior-edge] gaussian update mask '{runtime_args.optimize_gaussian_mask_key}': "
            f"{selected_count}/{external_update_mask.shape[0]}"
        )
        if selected_count <= 0:
            raise ValueError("External Gaussian update mask is empty; regenerate the edge-region GS payload.")
    if gaussian_action_payload is not None:
        print(
            "[hrgs-action] loaded Gaussian action payload: "
            f"update_mean={gaussian_action_payload['update_strength'].mean().item():.4f}, "
            f"attach_mean={gaussian_action_payload['attach_strength'].mean().item():.4f}, "
            f"detail_mean={gaussian_action_payload['detail_weight'].mean().item():.4f}, "
            f"prior_color_mean={gaussian_action_payload['prior_color_strength'].mean().item():.4f}, "
            f"suppress_mean={gaussian_action_payload['suppress_strength'].mean().item():.4f}"
        )
    if freeze_geometry_mask is not None:
        print(
            "[hrgs-action] freeze geometry mask: "
            f"{int(freeze_geometry_mask.sum().item())}/{freeze_geometry_mask.shape[0]}"
        )
    active_train_cameras = trainCameras
    prior_edge_skip_count = 0
    if bool(runtime_args.prior_only_edge_finetune):
        if runtime_args.lambda_prior_edge <= 0.0:
            raise ValueError("--prior_only_edge_finetune requires --lambda_prior_edge > 0")
        if prior_edge_index is None or prior_edge_mask_dir is None:
            raise ValueError("--prior_only_edge_finetune requires --prior_edge_dir and --prior_edge_mask_dir")
        if external_update_mask is None:
            raise ValueError("--prior_only_edge_finetune requires --optimize_gaussian_mask_payload")
        active_train_cameras, missing_prior_samples, missing_mask_samples = build_prior_edge_camera_pool(
            trainCameras,
            prior_edge_index,
            prior_edge_mask_dir,
        )
        print(
            f"[prior-edge] prior-only camera pool: {len(active_train_cameras)}/{len(trainCameras)} "
            "train views with both prior image and edge mask"
        )
        if missing_prior_samples:
            print(f"[prior-edge] missing prior samples: {missing_prior_samples}")
        if missing_mask_samples:
            print(f"[prior-edge] missing edge-mask samples: {missing_mask_samples}")
        if len(active_train_cameras) == 0:
            raise ValueError("No train cameras have both prior images and edge masks for prior-only edge finetune.")

    diagnostics_enabled = bool(runtime_args.enable_gradient_tracking) or bool(runtime_args.enable_2d_dropout_diagnostic)
    diagnostics_output_root = None
    gradient_tracker = None
    dropout_diagnostic = None
    if diagnostics_enabled:
        diagnostics_output_root = Path(scene.model_path) / runtime_args.diagnostic_output_subdir
        diagnostics_output_root.mkdir(parents=True, exist_ok=True)
        basis_provider = DiagnosticBasisProvider(
            basis_mode=runtime_args.diagnostic_basis_mode,
            surface_payload_path=runtime_args.diagnostic_surface_payload,
        )
        gradient_start_iter = resolve_diagnostic_start_iter(runtime_args.gradient_tracking_from_iter, opt, mesh)
        dropout_start_iter = resolve_diagnostic_start_iter(runtime_args.dropout_diagnostic_from_iter, opt, mesh)
        gradient_tracker = GaussianGradientTracker(
            enabled=bool(runtime_args.enable_gradient_tracking),
            start_iter=gradient_start_iter,
            snapshot_interval=int(runtime_args.gradient_tracking_snapshot_interval),
            tile_size=int(runtime_args.gradient_tracking_tile_size),
            basis_provider=basis_provider,
            reset_on_phase_change=not bool(runtime_args.diagnostic_disable_phase_reset),
        )
        dropout_diagnostic = TwoDDropoutGradientDiagnostic(
            enabled=bool(runtime_args.enable_2d_dropout_diagnostic),
            start_iter=dropout_start_iter,
            interval=int(runtime_args.dropout_diagnostic_interval),
            num_masks=int(runtime_args.dropout_diagnostic_num_masks),
            tile_size=int(runtime_args.dropout_diagnostic_tile_size),
            keep_ratio=float(runtime_args.dropout_diagnostic_keep_ratio),
            basis_provider=basis_provider,
            loss_mode=runtime_args.dropout_diagnostic_loss_mode,
            alpha_threshold=float(runtime_args.dropout_diagnostic_alpha_threshold),
            min_active_pixels=int(runtime_args.dropout_diagnostic_min_active_pixels),
        )
        print(
            "[diagnostics] enabled: "
            f"gradient={bool(runtime_args.enable_gradient_tracking)}@{gradient_start_iter}, "
            f"dropout={bool(runtime_args.enable_2d_dropout_diagnostic)}@{dropout_start_iter}, "
            f"basis={runtime_args.diagnostic_basis_mode}, output={diagnostics_output_root}"
        )
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    train_start_iter = first_iter + 1
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, message = network_gui.receive()
                if custom_cam != None:
                    with torch.no_grad():
                        debugVis = DebugVisualization(**message["debug_data"])
                        net_image = render(custom_cam, gaussians, pipe, background, message["scaling_modifier"], splat_args=splat_args, debugVis=debugVis)["render"]

                    if debugVis.type == 0:
                        image = gui_visualize(
                            render_cam=custom_cam,
                            alpha=net_image[7:8],
                            distortion=net_image[8:9],
                            depth=net_image[6:7],
                            normal=net_image[3:6],
                            render=net_image[:3],
                            other_args=message
                        )
                    else:
                        image = net_image[:3]

                    image = torch.clamp(image, 0., 1.)
                    net_image_bytes = memoryview((image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                net_image_bytes = memoryview((image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if bool(message["train"]) and ((iteration < int(opt.iterations)) or not bool(message["keep_alive"])):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = active_train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        phase_name = compute_training_phase(iteration, opt, mesh)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if iteration > mesh.distortion_from_iter and mesh.lambda_opacity_field > 0.0:
            splat_args.render_opacity = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, splat_args=splat_args)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image = rendering[:3, :, :]
        gaussian_action_state = resolve_gaussian_route_runtime_state(
            gaussian_action_payload,
            total_gaussians=gaussians.get_xyz.shape[0],
            min_weight=float(runtime_args.gaussian_action_min_weight),
            attach_scale=float(runtime_args.gaussian_action_attach_scale),
            detail_scale=float(runtime_args.gaussian_action_detail_scale),
            prior_color_scale=float(runtime_args.gaussian_action_prior_color_scale),
            geometry_scale=float(runtime_args.gaussian_action_geometry_scale),
            appearance_scale=float(runtime_args.gaussian_action_appearance_scale),
            suppress_scale=float(runtime_args.gaussian_action_suppress_scale),
        )

        # Loss
        gt_image = load_external_rgb_cached(
            viewpoint_cam.image_name,
            global_image_index,
            global_image_cache,
            height=image.shape[1],
            width=image.shape[2],
        )
        if gt_image is None:
            gt_image = viewpoint_cam.original_image.cuda()
        else:
            gt_image = gt_image.permute(2, 0, 1)
        L_SSIM = compute_ssim_loss(image, gt_image)
        if mesh.use_decoupled_appearance:
            Ll1 = appearance_embedding.L1_loss_appearance(image, gt_image, viewpoint_cam.idx)
        if mesh.use_pgsr_appearance and L_SSIM < 0.5:
            Ll1 = appearance_embedding.L1_loss_appearance(image, gt_image, viewpoint_cam.idx)
        else:
            Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * L_SSIM

        prior_local_loss = None
        if runtime_args.lambda_prior_local > 0.0 and prior_local_index is not None and prior_local_mask_dir is not None:
            prior_image = load_external_rgb_cached(
                viewpoint_cam.image_name,
                prior_local_index,
                prior_local_cache,
                height=image.shape[1],
                width=image.shape[2],
            )
            prior_mask = load_external_mask_cached(
                viewpoint_cam.image_name,
                prior_local_mask_dir,
                prior_local_mask_cache,
                height=image.shape[1],
                width=image.shape[2],
            )
            if prior_image is not None and prior_mask is not None:
                if float(prior_mask.sum().item()) >= float(runtime_args.prior_local_min_pixels):
                    prior_local_loss = compute_masked_l1_loss(
                        image,
                        prior_image.permute(2, 0, 1),
                        prior_mask,
                    )

        prior_edge_loss = None
        prior_edge_touch_mask = None
        prior_edge_detail_alpha_value = None
        if runtime_args.lambda_prior_edge > 0.0 and prior_edge_index is not None and prior_edge_mask_dir is not None:
            prior_image = load_external_rgb_cached(
                viewpoint_cam.image_name,
                prior_edge_index,
                prior_edge_cache,
                height=image.shape[1],
                width=image.shape[2],
            )
            prior_mask = load_external_mask_cached(
                viewpoint_cam.image_name,
                prior_edge_mask_dir,
                prior_edge_mask_cache,
                height=image.shape[1],
                width=image.shape[2],
            )
            if prior_image is not None and prior_mask is not None:
                if float(prior_mask.sum().item()) >= float(runtime_args.prior_edge_min_pixels):
                    prior_target = prior_image.permute(2, 0, 1)
                    if runtime_args.prior_edge_loss_mode == "detail_v1":
                        prior_edge_detail_alpha_value = get_prior_edge_detail_alpha(
                            runtime_args,
                            iteration,
                            train_start_iter,
                        )
                        lowfreq_anchor = gt_image if runtime_args.prior_edge_lowfreq_anchor == "gt" else None
                        prior_edge_loss = compute_prior_edge_detail_loss(
                            image,
                            prior_target,
                            prior_mask,
                            runtime_args,
                            detail_alpha=prior_edge_detail_alpha_value,
                            lowfreq_anchor=lowfreq_anchor,
                        )
                    else:
                        blend_alpha = float(runtime_args.prior_edge_blend_alpha)
                        if blend_alpha < 1.0:
                            blend_alpha = max(0.0, blend_alpha)
                            prior_target = blend_alpha * prior_target + (1.0 - blend_alpha) * image.detach()
                        prior_edge_loss = compute_masked_l1_loss(
                            image,
                            prior_target,
                            prior_mask,
                        )
                    prior_edge_touch_mask = build_projected_gaussian_touch_mask(
                        viewpoint_cam,
                        gaussians,
                        prior_mask,
                        visibility_filter=visibility_filter,
                        radii=radii,
                        min_touch_radius_px=runtime_args.prior_edge_touch_min_radius_px,
                        radius_scale=runtime_args.prior_edge_touch_radius_scale,
                        max_touch_radius_px=runtime_args.prior_edge_touch_max_radius_px,
                    )
                    if prior_edge_touch_mask is not None and external_update_mask is not None:
                        prior_edge_touch_mask = prior_edge_touch_mask & external_update_mask

        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        if depth.isnan().sum() > 0:
            print("DEPTH IS NAN!!!!!")
            depth[depth.isnan()] = 0.0
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        # c2w = (viewpoint_cam.world_view_transform.T).inverse()
        # if we only need the rotation, why bother with the inverse
        c2w = (viewpoint_cam.world_view_transform)
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        # Keep image-gradient regularization on the same device as the renderer
        # when training images are stored on CPU to save VRAM.
        nabla_I = central_diff(gt_image.permute(1, 2, 0))
        
        normal_error = (1 - (render_normal_world * depth_normal).sum(dim=0))
        depth_normal_loss = normal_error.mean()
        
        lambda_distortion = mesh.lambda_distortion if iteration >= mesh.distortion_from_iter else 0.0
        lambda_depth_normal = mesh.lambda_depth_normal if iteration >= mesh.depth_normal_from_iter else 0.0
            
        # Normal regularization (smoothness)
        normal_loss = central_diff(render_normal_world.permute(1,2,0)) * torch.exp(-nabla_I)
        normal_loss = normal_loss.mean()
        lambda_normal = mesh.lambda_smoothness if iteration >= mesh.depth_normal_from_iter else 0.0

        lambda_opacity_field = mesh.lambda_opacity_field if iteration >= mesh.distortion_from_iter else 0.0
        opacity = rendering[7]
        opa_loss = (opacity - 0.5)**2

        #Ll1opacity_smoothness = central_diff(rendering[7][..., None]) * torch.exp(-nabla_I)
        opa_loss = opa_loss.mean()
        
        lambda_extent = mesh.lambda_extent if iteration >= mesh.distortion_from_iter else 0.0
        extent_loss = rendering[9]
        extent_loss = extent_loss.mean()
        surface_thin_loss = None
        
        if bool(runtime_args.prior_only_edge_finetune):
            if prior_edge_loss is None:
                prior_edge_skip_count += 1
                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "Loss": "skip",
                            "Skip": str(prior_edge_skip_count),
                            "Size": f"{len(gaussians._xyz)}",
                        }
                    )
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                continue
            loss = runtime_args.lambda_prior_edge * prior_edge_loss
        else:
            # Final loss
            loss =  rgb_loss + \
                    depth_normal_loss   * lambda_depth_normal + \
                    distortion_loss     * lambda_distortion +  \
                    normal_loss         * lambda_normal + \
                    opa_loss            * lambda_opacity_field + \
                    extent_loss         * lambda_extent

            if prior_local_loss is not None:
                loss = loss + runtime_args.lambda_prior_local * prior_local_loss
            if prior_edge_loss is not None:
                loss = loss + runtime_args.lambda_prior_edge * prior_edge_loss
            surface_thin_loss = surface_thinning_regularizer.loss(
                gaussians,
                iteration,
                point_weight=None if gaussian_action_state is None else gaussian_action_state["attach_strength"],
            )
            if surface_thin_loss is not None:
                loss = loss + float(runtime_args.lambda_surface_thin) * surface_thin_loss
        if prior_edge_touch_mask is not None:
            gaussians.mark_edge_touched(prior_edge_touch_mask, iteration)
                
        if not bool(runtime_args.prior_only_edge_finetune):
            # MCMC losses
            loss += mesh.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            loss += mesh.scale_reg * torch.abs(gaussians.get_scaling).mean()
            loss += mesh.min_scale_reg * torch.min(gaussians.get_scaling, dim=-1).values.mean()

        gradient_snapshot = None
        loss.backward()
        source_update_mask = build_source_tag_train_mask(gaussians, runtime_args.optimize_source_tag)
        action_update_mask = None if gaussian_action_state is None else gaussian_action_state["active_mask"]
        update_mask = combine_gaussian_update_masks(source_update_mask, external_update_mask, action_update_mask)
        apply_gaussian_update_mask(
            gaussians,
            update_mask=update_mask,
            freeze_appearance=bool(runtime_args.freeze_appearance_during_tag_optimization),
            update_scale=float(runtime_args.prior_edge_update_scale) * float(runtime_args.gaussian_action_update_scale),
            geometry_weight=None if gaussian_action_state is None else gaussian_action_state["geometry_weight"],
            position_weight=None if gaussian_action_state is None else gaussian_action_state["position_weight"],
            scaling_weight=None if gaussian_action_state is None else gaussian_action_state["scaling_weight"],
            rotation_weight=None if gaussian_action_state is None else gaussian_action_state["rotation_weight"],
            appearance_weight=None if gaussian_action_state is None else gaussian_action_state["appearance_weight"],
            opacity_weight=None if gaussian_action_state is None else gaussian_action_state["opacity_weight"],
            freeze_geometry_mask=freeze_geometry_mask,
        )
        if gradient_tracker is not None:
            gradient_snapshot = gradient_tracker.update(
                iteration=iteration,
                phase_name=phase_name,
                viewpoint_cam=viewpoint_cam,
                gaussians=gaussians,
                visibility_filter=visibility_filter,
                gt_image=gt_image.detach(),
                render_image=image.detach(),
                gradient_mask=update_mask,
            )
        iter_end.record()

        diagnostics_runtime = None
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress = {"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{len(gaussians._xyz)}"}
                if prior_edge_loss is not None:
                    progress["Edge"] = f"{prior_edge_loss.item():.{4}f}"
                if prior_edge_detail_alpha_value is not None:
                    progress["A"] = f"{prior_edge_detail_alpha_value:.{2}f}"
                if surface_thin_loss is not None:
                    progress["Thin"] = f"{surface_thin_loss.item():.{4}e}"
                progress_bar.set_postfix(progress)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), normal_loss=depth_normal_loss * lambda_depth_normal, distortion_loss=distortion_loss * lambda_distortion)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Keep edge prior-only finetuning aligned with the external GS mask.
            if not bool(runtime_args.prior_only_edge_finetune):
                # Densification (AbsGrad or MCMC)
                densifier.densify(
                    iteration=iteration,
                    visibility_filter=visibility_filter,
                    radii=radii,
                    viewspace_point_tensor=viewspace_point_tensor,
                    cameras_extent=scene.cameras_extent,
                    trainCameras=trainCameras
                )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if not bool(runtime_args.prior_only_edge_finetune):
                    densifier.postfix(xyz_lr=xyz_lr)

            dropout_snapshot = None
            if dropout_diagnostic is not None:
                with torch.enable_grad():
                    dropout_snapshot = dropout_diagnostic.run(
                        iteration=iteration,
                        phase_name=phase_name,
                        viewpoint_cam=viewpoint_cam,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=background,
                        gt_image=gt_image.detach(),
                        base_rendering=rendering.detach(),
                        splat_args=splat_args,
                        render_fn=render,
                        build_touch_mask_fn=build_projected_gaussian_touch_mask,
                        gradient_mask=update_mask,
                    )
                    gaussians.optimizer.zero_grad(set_to_none=True)
            if diagnostics_output_root is not None:
                diagnostics_runtime = export_diagnostic_bundle(
                    output_root=diagnostics_output_root,
                    iteration=iteration,
                    phase_name=phase_name,
                    camera_name=viewpoint_cam.image_name,
                    gradient_snapshot=gradient_snapshot,
                    dropout_snapshot=dropout_snapshot,
                )
                if tb_writer and diagnostics_runtime is not None:
                    grad_summary = gradient_snapshot["summary"] if gradient_snapshot is not None else {}
                    drop_summary = dropout_snapshot["summary"] if dropout_snapshot is not None else {}
                    if "running_jitter_mean" in grad_summary:
                        tb_writer.add_scalar("diagnostics/grad_running_jitter_mean", grad_summary["running_jitter_mean"], iteration)
                    if "running_flip_rate_mean" in grad_summary:
                        tb_writer.add_scalar("diagnostics/grad_running_flip_mean", grad_summary["running_flip_rate_mean"], iteration)
                    if "dropout_std_mean" in drop_summary:
                        tb_writer.add_scalar("diagnostics/dropout_std_mean", drop_summary["dropout_std_mean"], iteration)
                    if "dropout_sign_agreement_mean" in drop_summary:
                        tb_writer.add_scalar("diagnostics/dropout_sign_agreement_mean", drop_summary["dropout_sign_agreement_mean"], iteration)
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                appearance_state = appearance_embedding.capture() if appearance_embedding is not None else (None, None)
                torch.save((gaussians.capture(), iteration, appearance_state), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    end_event = time.time() 
    
    print(f'Training in {end_event - start_event :.4f} seconds!')

def prepare_output_and_logger(args, settings: ExtendedSettings, opt, pipe, mesh):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
    # write config file
    with open(os.path.join(args.model_path, "config.json"), 'w') as config_json:
        json.dump(settings.to_dict(), config_json)

    # write output config files for opt, pipe, mesh
    with open(os.path.join(args.model_path, "mesh_args"), 'w') as f:
        f.write(str(Namespace(**vars(mesh))))
    with open(os.path.join(args.model_path, "rem_args"), 'w') as f:
        f.write(str(Namespace(**{**vars(opt), **vars(pipe)})))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed, normal_loss, distortion_loss):
    if tb_writer:
        tb_writer.add_scalar('train_loss/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        if iteration > 15000:
            tb_writer.add_scalar('additional_losses/normal_loss', normal_loss.item(), iteration)
            tb_writer.add_scalar('additional_losses/distortion_loss', distortion_loss.item(), iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MeshingParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_ply", type=str, default=None)
    parser.add_argument("--optimize_source_tag", choices=["all", "prior", "probe", "added"], default="all")
    parser.add_argument("--optimize_gaussian_mask_payload", type=str, default=None)
    parser.add_argument("--optimize_gaussian_mask_key", type=str, default="selected_mask")
    parser.add_argument("--freeze_geometry_mask_payload", type=str, default=None)
    parser.add_argument("--freeze_geometry_mask_key", type=str, default="selected_mask")
    parser.add_argument("--gaussian_action_payload", type=str, default=None)
    parser.add_argument("--gaussian_action_min_weight", type=float, default=0.0)
    parser.add_argument("--gaussian_action_update_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_action_attach_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_action_detail_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_action_prior_color_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_action_geometry_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_action_appearance_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_action_suppress_scale", type=float, default=1.0)
    parser.add_argument("--freeze_appearance_during_tag_optimization", action="store_true")
    parser.add_argument("--global_image_dir", type=str, default=None)
    parser.add_argument("--prior_local_dir", type=str, default=None)
    parser.add_argument("--prior_local_mask_dir", type=str, default=None)
    parser.add_argument("--lambda_prior_local", type=float, default=0.0)
    parser.add_argument("--prior_local_min_pixels", type=float, default=64.0)
    parser.add_argument("--prior_edge_dir", type=str, default=None)
    parser.add_argument("--prior_edge_mask_dir", type=str, default=None)
    parser.add_argument("--lambda_prior_edge", type=float, default=0.0)
    parser.add_argument("--prior_only_edge_finetune", action="store_true")
    parser.add_argument("--prior_edge_loss_mode", choices=["rgb", "detail_v1"], default="rgb")
    parser.add_argument("--prior_edge_blend_alpha", type=float, default=1.0)
    parser.add_argument("--prior_edge_update_scale", type=float, default=1.0)
    parser.add_argument("--prior_edge_min_pixels", type=float, default=64.0)
    parser.add_argument("--prior_edge_touch_min_radius_px", type=float, default=2.0)
    parser.add_argument("--prior_edge_touch_radius_scale", type=float, default=0.5)
    parser.add_argument("--prior_edge_touch_max_radius_px", type=float, default=16.0)
    parser.add_argument("--prior_edge_detail_blur_kernel", type=int, default=9)
    parser.add_argument("--prior_edge_detail_alpha", type=float, default=0.6)
    parser.add_argument("--prior_edge_detail_alpha_final", type=float, default=-1.0)
    parser.add_argument("--prior_edge_detail_warmup_iters", type=int, default=0)
    parser.add_argument("--prior_edge_detail_weight", type=float, default=1.0)
    parser.add_argument("--prior_edge_lowfreq_weight", type=float, default=0.05)
    parser.add_argument("--prior_edge_grad_weight", type=float, default=0.0)
    parser.add_argument("--prior_edge_lowfreq_threshold", type=float, default=0.08)
    parser.add_argument("--prior_edge_lowfreq_anchor", choices=["render", "gt"], default="render")
    parser.add_argument("--prior_edge_detail_min_gain", type=float, default=0.0)
    parser.add_argument("--prior_edge_confidence_power", type=float, default=1.0)
    parser.add_argument("--surface_thin_mesh_path", type=str, default=None)
    parser.add_argument("--lambda_surface_thin", type=float, default=0.0)
    parser.add_argument("--surface_thin_from_iter", type=int, default=0)
    parser.add_argument("--surface_thin_until_iter", type=int, default=0)
    parser.add_argument("--surface_thin_sample_count", type=int, default=500000)
    parser.add_argument("--surface_thin_update_interval", type=int, default=500)
    parser.add_argument("--surface_thin_gaussian_sample_count", type=int, default=65536)
    parser.add_argument("--surface_thin_offset_margin", type=float, default=0.02)
    parser.add_argument("--surface_thin_normal_scale_target", type=float, default=0.0)
    parser.add_argument("--surface_thin_normal_scale_weight", type=float, default=1.0)
    parser.add_argument("--diagnostic_output_subdir", type=str, default="training_diagnostics_v0")
    parser.add_argument("--diagnostic_basis_mode", choices=["gaussian_frame", "surface_payload"], default="gaussian_frame")
    parser.add_argument("--diagnostic_surface_payload", type=str, default=None)
    parser.add_argument("--diagnostic_disable_phase_reset", action="store_true")
    parser.add_argument("--enable_gradient_tracking", action="store_true")
    parser.add_argument("--gradient_tracking_from_iter", type=int, default=-1)
    parser.add_argument("--gradient_tracking_snapshot_interval", type=int, default=500)
    parser.add_argument("--gradient_tracking_tile_size", type=int, default=32)
    parser.add_argument("--enable_2d_dropout_diagnostic", action="store_true")
    parser.add_argument("--dropout_diagnostic_from_iter", type=int, default=-1)
    parser.add_argument("--dropout_diagnostic_interval", type=int, default=1000)
    parser.add_argument("--dropout_diagnostic_num_masks", type=int, default=6)
    parser.add_argument("--dropout_diagnostic_tile_size", type=int, default=48)
    parser.add_argument("--dropout_diagnostic_keep_ratio", type=float, default=0.75)
    parser.add_argument("--dropout_diagnostic_loss_mode", choices=["masked_l1", "masked_gradient_l1"], default="masked_l1")
    parser.add_argument("--dropout_diagnostic_alpha_threshold", type=float, default=0.1)
    parser.add_argument("--dropout_diagnostic_min_active_pixels", type=int, default=256)

    args = parser.parse_args(sys.argv[1:])
    if args.start_checkpoint and args.start_ply:
        raise ValueError("Pass only one of --start_checkpoint or --start_ply.")
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    splat_args = ss.get_settings(args)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), mp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, splat_args, args)

    # All done
    print("\nTraining complete.")
