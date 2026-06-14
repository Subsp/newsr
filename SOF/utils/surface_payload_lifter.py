from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SurfacePayloadLifterConfig:
    min_confidence: float = 0.35
    min_mask_value: float = 0.5
    min_depth: float = 1e-4
    max_points_per_view: int = 12000
    voxel_size: float = 0.0
    voxel_size_scale: float = 2.0
    min_views_per_cluster: int = 1
    min_points_per_cluster: int = 1
    max_disagreement: float = 0.10
    thickness_ratio: float = 0.05
    min_scale: float = 1e-4
    max_scale: float = 0.25
    pixel_footprint_scale: float = 1.0


def _canonicalize_camera_tensors(
    cameras: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "intrinsics" not in cameras:
        raise KeyError("cameras must contain 'intrinsics'.")
    intrinsics = cameras["intrinsics"]
    if "cam_to_world" in cameras:
        cam_to_world = cameras["cam_to_world"]
        world_to_view = torch.linalg.inv(cam_to_world)
    elif "world_to_view" in cameras:
        world_to_view = cameras["world_to_view"]
        cam_to_world = torch.linalg.inv(world_to_view)
    else:
        raise KeyError("cameras must contain either 'cam_to_world' or 'world_to_view'.")
    return intrinsics, cam_to_world, world_to_view


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _resize_intrinsics(
    intrinsics: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> torch.Tensor:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = intrinsics.clone()
    out[..., 0, 0] *= float(dst_w) / float(src_w)
    out[..., 1, 1] *= float(dst_h) / float(src_h)
    out[..., 0, 2] *= float(dst_w) / float(src_w)
    out[..., 1, 2] *= float(dst_h) / float(src_h)
    return out


def _backproject_depth(
    depth_hw: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
) -> torch.Tensor:
    h, w = depth_hw.shape
    device = depth_hw.device
    dtype = depth_hw.dtype
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) + 0.5,
        torch.arange(w, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    rays_cam = pixels @ torch.linalg.inv(intrinsics).transpose(0, 1)
    points_cam = rays_cam * depth_hw.reshape(-1, 1)
    rot = cam_to_world[:3, :3]
    trans = cam_to_world[:3, 3]
    points_world = points_cam @ rot.transpose(0, 1) + trans.unsqueeze(0)
    return points_world.reshape(h, w, 3)


def _orthonormal_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype).expand_as(normals)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=normals.device, dtype=normals.dtype).expand_as(normals)
    use_y = torch.abs(normals[:, 0]) > 0.9
    ref = torch.where(use_y[:, None], y_axis, x_axis)
    tangent_u = _normalize(torch.cross(ref, normals, dim=-1))
    tangent_v = _normalize(torch.cross(normals, tangent_u, dim=-1))
    return tangent_u, tangent_v


def _cluster_stats(
    points: torch.Tensor,
    normals: torch.Tensor,
    colors: torch.Tensor,
    weights: torch.Tensor,
    view_ids: torch.Tensor,
    base_scale_u: torch.Tensor,
    base_scale_v: torch.Tensor,
    cfg: SurfacePayloadLifterConfig,
) -> Dict[str, torch.Tensor]:
    weight_sum = weights.sum().clamp_min(1e-8)
    center = (points * weights[:, None]).sum(dim=0) / weight_sum
    normal = _normalize((normals * weights[:, None]).sum(dim=0, keepdim=True))[0]
    fused_rgb = (colors * weights[:, None]).sum(dim=0) / weight_sum
    confidence = (weights.sum() / float(max(weights.numel(), 1))).clamp(0.0, 1.0)

    tangent_u, tangent_v = _orthonormal_basis(normal[None])
    tangent_u = tangent_u[0]
    tangent_v = tangent_v[0]
    delta = points - center[None]
    offset_n = torch.abs((delta * normal[None]).sum(dim=-1))
    offset_u = torch.abs((delta * tangent_u[None]).sum(dim=-1))
    offset_v = torch.abs((delta * tangent_v[None]).sum(dim=-1))
    disagreement = (
        offset_n.mean()
        + 0.5 * (1.0 - torch.clamp((normals * normal[None]).sum(dim=-1), -1.0, 1.0)).mean()
    )

    scale_u = torch.maximum(base_scale_u.mean(), offset_u.quantile(0.75) * 2.0).clamp(
        min=cfg.min_scale,
        max=cfg.max_scale,
    )
    scale_v = torch.maximum(base_scale_v.mean(), offset_v.quantile(0.75) * 2.0).clamp(
        min=cfg.min_scale,
        max=cfg.max_scale,
    )
    scale_n = (torch.minimum(scale_u, scale_v) * float(cfg.thickness_ratio)).clamp(
        min=cfg.min_scale,
        max=cfg.max_scale,
    )
    unique_views = torch.unique(view_ids).numel()
    valid = (
        (float(confidence.item()) >= float(cfg.min_confidence))
        and (int(unique_views) >= int(cfg.min_views_per_cluster))
        and (int(points.shape[0]) >= int(cfg.min_points_per_cluster))
        and (float(disagreement.item()) <= float(cfg.max_disagreement))
    )
    return {
        "center": center,
        "normal": normal,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "scale_u": scale_u.unsqueeze(0),
        "scale_v": scale_v.unsqueeze(0),
        "scale_n": scale_n.unsqueeze(0),
        "fused_rgb": fused_rgb,
        "confidence": confidence.unsqueeze(0),
        "disagreement": disagreement.unsqueeze(0),
        "view_count": torch.tensor([unique_views], device=points.device, dtype=torch.int32),
        "valid_mask": torch.tensor([valid], device=points.device, dtype=torch.bool),
    }


def save_surface_payload_npz(path: str | Path, payload: Dict[str, torch.Tensor]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in payload.items():
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bool:
            serializable[key] = tensor.numpy().astype(np.bool_)
        else:
            serializable[key] = tensor.numpy()
    np.savez(path, **serializable)
    return path


def lift_surface_payload(
    depth_surf: torch.Tensor,
    normal_surf: torch.Tensor,
    conf_geo: torch.Tensor,
    mask_surface: torch.Tensor,
    sr_images: torch.Tensor,
    cameras: Dict[str, torch.Tensor],
    cfg: SurfacePayloadLifterConfig | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or SurfacePayloadLifterConfig()
    intrinsics, cam_to_world, _ = _canonicalize_camera_tensors(cameras)

    b, v, _, h, w = depth_surf.shape
    intrinsics = _resize_intrinsics(intrinsics, (h, w), (h, w))

    points_all: List[torch.Tensor] = []
    normals_all: List[torch.Tensor] = []
    colors_all: List[torch.Tensor] = []
    weights_all: List[torch.Tensor] = []
    view_ids_all: List[torch.Tensor] = []
    batch_ids_all: List[torch.Tensor] = []
    scale_u_all: List[torch.Tensor] = []
    scale_v_all: List[torch.Tensor] = []

    for bi in range(b):
        for vi in range(v):
            depth_hw = depth_surf[bi, vi, 0]
            conf_hw = conf_geo[bi, vi, 0]
            mask_hw = mask_surface[bi, vi, 0]
            valid = (
                (depth_hw > float(cfg.min_depth))
                & torch.isfinite(depth_hw)
                & (conf_hw >= float(cfg.min_confidence))
                & (mask_hw >= float(cfg.min_mask_value))
            )
            if valid.sum().item() <= 0:
                continue

            scores = (conf_hw * mask_hw).masked_fill(~valid, -1.0)
            if int(cfg.max_points_per_view) > 0 and valid.sum().item() > int(cfg.max_points_per_view):
                topk = torch.topk(scores.reshape(-1), k=int(cfg.max_points_per_view))
                keep = torch.zeros_like(scores.reshape(-1), dtype=torch.bool)
                keep[topk.indices] = True
                valid = keep.reshape(h, w)

            world_points = _backproject_depth(depth_hw, intrinsics[bi, vi], cam_to_world[bi, vi])
            normals = _normalize(normal_surf[bi, vi].permute(1, 2, 0).reshape(-1, 3))
            colors = sr_images[bi, vi].permute(1, 2, 0).reshape(-1, 3)
            weights = (conf_hw * mask_hw).reshape(-1)
            valid_flat = valid.reshape(-1)

            yy, xx = torch.meshgrid(
                torch.arange(h, device=depth_hw.device, dtype=depth_hw.dtype) + 0.5,
                torch.arange(w, device=depth_hw.device, dtype=depth_hw.dtype) + 0.5,
                indexing="ij",
            )
            depth_flat = depth_hw.reshape(-1)
            fx = intrinsics[bi, vi, 0, 0].clamp_min(1e-6)
            fy = intrinsics[bi, vi, 1, 1].clamp_min(1e-6)
            base_scale_u = depth_flat / fx * float(cfg.pixel_footprint_scale)
            base_scale_v = depth_flat / fy * float(cfg.pixel_footprint_scale)

            points_all.append(world_points.reshape(-1, 3)[valid_flat])
            normals_all.append(normals[valid_flat])
            colors_all.append(colors[valid_flat])
            weights_all.append(weights[valid_flat].clamp_min(1e-6))
            view_ids_all.append(torch.full((int(valid_flat.sum().item()),), vi, device=depth_hw.device, dtype=torch.int64))
            batch_ids_all.append(torch.full((int(valid_flat.sum().item()),), bi, device=depth_hw.device, dtype=torch.int64))
            scale_u_all.append(base_scale_u[valid_flat].clamp(min=cfg.min_scale, max=cfg.max_scale))
            scale_v_all.append(base_scale_v[valid_flat].clamp(min=cfg.min_scale, max=cfg.max_scale))

    device = depth_surf.device
    if not points_all:
        empty = torch.empty((0, 3), device=device, dtype=depth_surf.dtype)
        empty1 = torch.empty((0, 1), device=device, dtype=depth_surf.dtype)
        return {
            "centers": empty,
            "normals": empty,
            "tangent_u": empty,
            "tangent_v": empty,
            "scale_u": empty1,
            "scale_v": empty1,
            "scale_n": empty1,
            "fused_rgb": empty,
            "confidence": empty1,
            "disagreement": empty1,
            "view_count": torch.empty((0, 1), device=device, dtype=torch.int32),
            "valid_mask": torch.empty((0, 1), device=device, dtype=torch.bool),
        }

    points = torch.cat(points_all, dim=0)
    normals = _normalize(torch.cat(normals_all, dim=0))
    colors = torch.cat(colors_all, dim=0).clamp(0.0, 1.0)
    weights = torch.cat(weights_all, dim=0)
    view_ids = torch.cat(view_ids_all, dim=0)
    batch_ids = torch.cat(batch_ids_all, dim=0)
    base_scale_u = torch.cat(scale_u_all, dim=0)
    base_scale_v = torch.cat(scale_v_all, dim=0)

    if float(cfg.voxel_size) > 0.0:
        voxel_size = float(cfg.voxel_size)
    else:
        voxel_size = float(torch.median(torch.cat([base_scale_u, base_scale_v])) * float(cfg.voxel_size_scale))
    voxel_size = max(voxel_size, float(cfg.min_scale))

    centers_list: List[torch.Tensor] = []
    normals_list: List[torch.Tensor] = []
    tangent_u_list: List[torch.Tensor] = []
    tangent_v_list: List[torch.Tensor] = []
    scale_u_list: List[torch.Tensor] = []
    scale_v_list: List[torch.Tensor] = []
    scale_n_list: List[torch.Tensor] = []
    colors_list: List[torch.Tensor] = []
    conf_list: List[torch.Tensor] = []
    disagreement_list: List[torch.Tensor] = []
    view_count_list: List[torch.Tensor] = []
    valid_list: List[torch.Tensor] = []

    for bi in range(b):
        batch_mask = batch_ids == bi
        if batch_mask.sum().item() <= 0:
            continue
        p_batch = points[batch_mask]
        n_batch = normals[batch_mask]
        c_batch = colors[batch_mask]
        w_batch = weights[batch_mask]
        v_batch = view_ids[batch_mask]
        su_batch = base_scale_u[batch_mask]
        sv_batch = base_scale_v[batch_mask]

        voxel_ids = torch.floor(p_batch / voxel_size).to(dtype=torch.int64)
        _, inverse = torch.unique(voxel_ids, dim=0, return_inverse=True)
        for cluster_id in range(int(inverse.max().item()) + 1):
            keep = inverse == cluster_id
            stats = _cluster_stats(
                p_batch[keep],
                n_batch[keep],
                c_batch[keep],
                w_batch[keep],
                v_batch[keep],
                su_batch[keep],
                sv_batch[keep],
                cfg,
            )
            centers_list.append(stats["center"].unsqueeze(0))
            normals_list.append(stats["normal"].unsqueeze(0))
            tangent_u_list.append(stats["tangent_u"].unsqueeze(0))
            tangent_v_list.append(stats["tangent_v"].unsqueeze(0))
            scale_u_list.append(stats["scale_u"])
            scale_v_list.append(stats["scale_v"])
            scale_n_list.append(stats["scale_n"])
            colors_list.append(stats["fused_rgb"].unsqueeze(0))
            conf_list.append(stats["confidence"])
            disagreement_list.append(stats["disagreement"])
            view_count_list.append(stats["view_count"])
            valid_list.append(stats["valid_mask"])

    return {
        "centers": torch.cat(centers_list, dim=0),
        "normals": _normalize(torch.cat(normals_list, dim=0)),
        "tangent_u": _normalize(torch.cat(tangent_u_list, dim=0)),
        "tangent_v": _normalize(torch.cat(tangent_v_list, dim=0)),
        "scale_u": torch.cat(scale_u_list, dim=0).reshape(-1, 1),
        "scale_v": torch.cat(scale_v_list, dim=0).reshape(-1, 1),
        "scale_n": torch.cat(scale_n_list, dim=0).reshape(-1, 1),
        "fused_rgb": torch.cat(colors_list, dim=0).clamp(0.0, 1.0),
        "confidence": torch.cat(conf_list, dim=0).reshape(-1, 1),
        "disagreement": torch.cat(disagreement_list, dim=0).reshape(-1, 1),
        "view_count": torch.cat(view_count_list, dim=0).reshape(-1, 1),
        "valid_mask": torch.cat(valid_list, dim=0).reshape(-1, 1),
    }
