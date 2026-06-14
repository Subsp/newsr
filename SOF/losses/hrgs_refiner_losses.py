from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class HRGSRefinerLossConfig:
    depth_weight: float = 1.0
    normal_weight: float = 0.2
    multiview_weight: float = 0.0
    delta_weight: float = 0.02
    surface_mask_weight: float = 0.2
    update_mask_weight: float = 0.1
    detail_mask_weight: float = 0.1
    confidence_weight: float = 0.1
    prior_color_weight: float = 0.05
    vggt_conf_threshold: float = 0.3
    gs_alpha_threshold: float = 0.05
    detail_grad_threshold: float = 0.08
    reprojection_tolerance: float = 0.05


def _mean_with_mask(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    denom = mask.sum().clamp_min(eps)
    return (value * mask).sum() / denom


def _depth_to_world_normals(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, v, _, h, w = depth.shape
    normals = torch.zeros((b, v, 3, h, w), device=depth.device, dtype=depth.dtype)
    valid = torch.zeros((b, v, 1, h, w), device=depth.device, dtype=torch.bool)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype) + 0.5,
        torch.arange(w, device=depth.device, dtype=depth.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)

    for bi in range(b):
        for vi in range(v):
            K = intrinsics[bi, vi]
            c2w = cam_to_world[bi, vi]
            rays_cam = pixels @ torch.linalg.inv(K).transpose(0, 1)
            d = depth[bi, vi, 0].reshape(-1, 1)
            points_cam = rays_cam * d
            rot = c2w[:3, :3]
            trans = c2w[:3, 3]
            points = (points_cam @ rot.transpose(0, 1) + trans.unsqueeze(0)).reshape(h, w, 3)

            dx = points[2:, 1:-1] - points[:-2, 1:-1]
            dy = points[1:-1, 2:] - points[1:-1, :-2]
            normal = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-6)
            normals[bi, vi, :, 1:-1, 1:-1] = normal.permute(2, 0, 1)

            local_valid = torch.isfinite(depth[bi, vi, 0]) & (depth[bi, vi, 0] > 1e-6)
            valid[bi, vi, 0, 1:-1, 1:-1] = (
                local_valid[1:-1, 1:-1]
                & local_valid[2:, 1:-1]
                & local_valid[:-2, 1:-1]
                & local_valid[1:-1, 2:]
                & local_valid[1:-1, :-2]
            )

    return normals, valid


def _image_grad_mag(image: torch.Tensor) -> torch.Tensor:
    dx = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
    dy = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
    dx = F.pad(dx.abs().mean(dim=2, keepdim=True), (0, 0, 0, 1))
    dy = F.pad(dy.abs().mean(dim=2, keepdim=True), (0, 1, 0, 0))
    return dx + dy


def _normalize_map(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = value.flatten(2)
    vmax = flat.amax(dim=-1, keepdim=True).clamp_min(eps)
    vmin = flat.amin(dim=-1, keepdim=True)
    normalized = (flat - vmin) / (vmax - vmin).clamp_min(eps)
    return normalized.view_as(value)


def _safe_bce(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    target = target.to(dtype=prob.dtype).clamp(0.0, 1.0)
    autocast_ctx = torch.cuda.amp.autocast(enabled=False) if prob.is_cuda else nullcontext()
    with autocast_ctx:
        return F.binary_cross_entropy(prob.float(), target.float())


def _project_world_points(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_view: torch.Tensor,
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = target_hw
    points_cam = points_world.reshape(-1, 3) @ world_to_view[:3, :3].transpose(0, 1)
    points_cam = points_cam + world_to_view[:3, 3].unsqueeze(0)
    z = points_cam[:, 2].reshape(h, w)
    z_safe = z.clamp_min(1e-6)
    uv = points_cam @ intrinsics.transpose(0, 1)
    u = (uv[:, 0] / z_safe.reshape(-1)).reshape(h, w)
    v = (uv[:, 1] / z_safe.reshape(-1)).reshape(h, w)
    grid_x = ((u + 0.5) / max(float(w), 1.0)) * 2.0 - 1.0
    grid_y = ((v + 0.5) / max(float(h), 1.0)) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1), z


def _backproject_depth_map(
    depth_hw: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
) -> torch.Tensor:
    h, w = depth_hw.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth_hw.device, dtype=depth_hw.dtype) + 0.5,
        torch.arange(w, device=depth_hw.device, dtype=depth_hw.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    rays_cam = pixels @ torch.linalg.inv(intrinsics).transpose(0, 1)
    points_cam = rays_cam * depth_hw.reshape(-1, 1)
    rot = cam_to_world[:3, :3]
    trans = cam_to_world[:3, 3]
    points_world = points_cam @ rot.transpose(0, 1) + trans.unsqueeze(0)
    return points_world.reshape(h, w, 3)


def _multiview_depth_consistency(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    cameras: Dict[str, torch.Tensor],
    tolerance_ratio: float,
) -> torch.Tensor:
    b, v, _, h, w = depth.shape
    intrinsics = cameras["intrinsics"]
    world_to_view = cameras["world_to_view"]
    cam_to_world = cameras["cam_to_world"]

    total = depth.new_tensor(0.0)
    count = 0
    for bi in range(b):
        for ref_idx in range(v):
            ref_depth = depth[bi, ref_idx, 0]
            ref_valid = valid_mask[bi, ref_idx, 0]
            world_points = _backproject_depth_map(ref_depth, intrinsics[bi, ref_idx], cam_to_world[bi, ref_idx])
            for src_idx in range(v):
                if src_idx == ref_idx:
                    continue
                grid, proj_depth = _project_world_points(
                    world_points,
                    intrinsics[bi, src_idx],
                    world_to_view[bi, src_idx],
                    (h, w),
                )
                sampled_depth = F.grid_sample(
                    depth[bi : bi + 1, src_idx],
                    grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0]
                sampled_valid = F.grid_sample(
                    valid_mask[bi : bi + 1, src_idx].to(dtype=depth.dtype),
                    grid.unsqueeze(0),
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0] > 0.5
                inside = (
                    (grid[..., 0] >= -1.0)
                    & (grid[..., 0] <= 1.0)
                    & (grid[..., 1] >= -1.0)
                    & (grid[..., 1] <= 1.0)
                    & (proj_depth > 1e-6)
                )
                mask = inside & ref_valid & sampled_valid
                if mask.any():
                    tol = torch.clamp(proj_depth.abs(), min=1.0) * float(tolerance_ratio)
                    total = total + _mean_with_mask(torch.abs(sampled_depth - proj_depth) / tol.clamp_min(1e-6), mask)
                    count += 1
    if count == 0:
        return depth.new_tensor(0.0)
    return total / float(count)


def compute_hrgsrefiner_losses(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    sample: Dict[str, Dict[str, torch.Tensor]],
    cfg: HRGSRefinerLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or HRGSRefinerLossConfig()

    surface = outputs["surface_2d"]
    update = outputs["update_2d"]
    gs_buffers = sample["lr_gs_buffers"]
    vggt_prior = sample["vggt_prior"]
    cameras = sample["cameras"]
    targets = sample["targets"]
    images = sample["images"]

    depth_surf = surface["depth_surf"]
    normal_surf = surface["normal_surf"]
    delta_depth = surface["delta_depth"]
    conf_geo = surface.get("conf_surface", surface["conf_geo"])
    mask_surface = surface.get("surface_ownership", surface["mask_surface"])
    mask_detail = surface.get("appearance_ownership", surface["mask_detail"])
    mask_update2d = surface["mask_update2d"]
    prior_color_weight2d = update["prior_color_weight2d"]

    oracle_depth = targets["oracle_depth"]
    valid_depth = targets["valid_depth"].to(dtype=torch.bool)
    gs_alpha = gs_buffers["alpha"]
    vggt_conf = vggt_prior["conf_hr"]

    depth_valid = valid_depth & torch.isfinite(depth_surf) & torch.isfinite(oracle_depth)
    safe_depth_surf = torch.where(depth_valid, depth_surf, torch.zeros_like(depth_surf))
    safe_oracle_depth = torch.where(depth_valid, oracle_depth, torch.zeros_like(oracle_depth))
    loss_depth = _mean_with_mask(
        F.smooth_l1_loss(safe_depth_surf, safe_oracle_depth, reduction="none"),
        depth_valid,
    )

    oracle_normal, oracle_normal_valid = _depth_to_world_normals(
        oracle_depth,
        cameras["intrinsics"],
        cameras["cam_to_world"],
    )
    oracle_normal = torch.where(torch.isfinite(oracle_normal), oracle_normal, torch.zeros_like(oracle_normal))
    normal_valid = oracle_normal_valid & depth_valid
    safe_normal_surf = torch.where(normal_valid.expand_as(normal_surf), normal_surf, torch.zeros_like(normal_surf))
    safe_oracle_normal = torch.where(normal_valid.expand_as(oracle_normal), oracle_normal, torch.zeros_like(oracle_normal))
    loss_normal = _mean_with_mask(
        1.0 - (safe_normal_surf * safe_oracle_normal).sum(dim=2, keepdim=True).clamp(-1.0, 1.0),
        normal_valid,
    )

    loss_delta = delta_depth.abs().mean()

    render_error = (gs_buffers["render_rgb"] - images["images_sr"]).abs().mean(dim=2, keepdim=True)
    update_target = _normalize_map(render_error) * gs_alpha.clamp(0.0, 1.0)
    grad_mag = _normalize_map(_image_grad_mag(images["images_sr"]))
    surface_target = (
        valid_depth.to(dtype=depth_surf.dtype)
        * (vggt_conf >= float(cfg.vggt_conf_threshold)).to(dtype=depth_surf.dtype)
    ).clamp(0.0, 1.0)
    appearance_seed = torch.maximum(_normalize_map(render_error), grad_mag)
    detail_target = (
        gs_alpha.clamp(0.0, 1.0)
        * (1.0 - surface_target)
        * appearance_seed
    ).clamp(0.0, 1.0)
    conf_residual = torch.abs(safe_depth_surf.detach() - safe_oracle_depth) / torch.clamp(safe_oracle_depth.abs(), min=1.0)
    conf_target = torch.where(
        valid_depth,
        torch.exp(-conf_residual),
        torch.zeros_like(conf_residual),
    )

    loss_surface_mask = _safe_bce(mask_surface, surface_target)
    loss_update_mask = _safe_bce(mask_update2d, update_target)
    loss_detail_mask = _safe_bce(mask_detail, detail_target)
    loss_conf = _safe_bce(conf_geo, conf_target)
    loss_prior_color = _safe_bce(prior_color_weight2d, detail_target)

    loss_multiview = depth_surf.new_tensor(0.0)
    if float(cfg.multiview_weight) > 0.0 and depth_surf.shape[1] > 1:
        loss_multiview = _multiview_depth_consistency(
            depth_surf,
            depth_valid,
            cameras,
            tolerance_ratio=float(cfg.reprojection_tolerance),
        )

    total = (
        float(cfg.depth_weight) * loss_depth
        + float(cfg.normal_weight) * loss_normal
        + float(cfg.multiview_weight) * loss_multiview
        + float(cfg.delta_weight) * loss_delta
        + float(cfg.surface_mask_weight) * loss_surface_mask
        + float(cfg.update_mask_weight) * loss_update_mask
        + float(cfg.detail_mask_weight) * loss_detail_mask
        + float(cfg.confidence_weight) * loss_conf
        + float(cfg.prior_color_weight) * loss_prior_color
    )

    return {
        "total": total,
        "depth": loss_depth.detach(),
        "normal": loss_normal.detach(),
        "multiview": loss_multiview.detach(),
        "delta": loss_delta.detach(),
        "surface_mask": loss_surface_mask.detach(),
        "update_mask": loss_update_mask.detach(),
        "detail_mask": loss_detail_mask.detach(),
        "confidence": loss_conf.detach(),
        "prior_color": loss_prior_color.detach(),
        "total_backprop": total,
    }
