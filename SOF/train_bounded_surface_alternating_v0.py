from __future__ import annotations

import json
import math
import random
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_soflr_vggt_bound_gs_correction_v0 import bind_gaussians_to_mesh
from recover_cleaned_mip_lr_v0 import build_sr_touch_prefilter
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import (
    build_prepared_sr_cache,
    copy_render_config,
    load_model_ply,
    load_train_cameras_only,
    resolve_iteration,
    select_uniform,
    summarize_sr_cache,
)
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir
from utils.sh_utils import RGB2SH, SH2RGB
from utils.sof_mesh_patch_enhancer_v0 import load_triangle_mesh, stats_from_array
from utils.system_utils import mkdir_p


TRUST_SURFACE = 0
TRUST_LOOSE = 1
TRUST_OUTLIER = 2

MEM_OBSERVE = 0
MEM_COLOR_READY = 1
MEM_COLOR_STABLE = 2
MEM_LOW_CONF = 3
MEM_REJECTED = 4


def features_dc_to_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    if features_dc.ndim == 3:
        if features_dc.shape[1] == 1:
            sh_dc = features_dc[:, 0, :]
        elif features_dc.shape[2] == 1:
            sh_dc = features_dc[:, :, 0]
        else:
            sh_dc = features_dc.reshape(features_dc.shape[0], -1)[:, :3]
    elif features_dc.ndim == 2:
        sh_dc = features_dc[:, :3]
    else:
        raise ValueError(f"Unsupported features_dc shape for RGB conversion: {tuple(features_dc.shape)}")
    return torch.nan_to_num(SH2RGB(sh_dc), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def rgb_to_features_dc_like(rgb: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    sh_dc = RGB2SH(rgb)
    if reference.ndim == 3:
        if reference.shape[1] == 1:
            return sh_dc[:, None, :].to(device=reference.device, dtype=reference.dtype)
        if reference.shape[2] == 1:
            return sh_dc[:, :, None].to(device=reference.device, dtype=reference.dtype)
        raise ValueError(f"Unsupported reference features_dc shape: {tuple(reference.shape)}")
    if reference.ndim == 2:
        return sh_dc.to(device=reference.device, dtype=reference.dtype)
    raise ValueError(f"Unsupported reference features_dc shape: {tuple(reference.shape)}")


def charbonnier_rows(delta: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(delta * delta + float(eps) ** 2).mean(dim=-1)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(weight.sum(), min=1e-6)
    return (value * weight).sum() / denom


def tensor_stats(value: torch.Tensor) -> Dict[str, float]:
    return stats_from_array(value.detach().cpu().numpy().astype(np.float32, copy=False))


def apply_base_support_mode(base_support: torch.Tensor, *, mode: str, floor: float) -> torch.Tensor:
    mode = str(mode)
    floor_value = float(floor)
    if mode == "multiply":
        return base_support
    if mode == "floor":
        return torch.maximum(base_support, torch.full_like(base_support, floor_value))
    if mode == "blend":
        return floor_value + (1.0 - floor_value) * base_support
    if mode == "none":
        return torch.ones_like(base_support)
    raise ValueError(f"Unsupported base_support_mode: {mode}")


def apply_residual_sample_clip(prior_sample: torch.Tensor, base_sample: torch.Tensor, residual_clip: float) -> torch.Tensor:
    clip = float(residual_clip)
    if clip <= 0.0:
        return prior_sample
    residual = torch.clamp(prior_sample - base_sample, min=-clip, max=clip)
    return torch.clamp(base_sample + residual, min=0.0, max=1.0)


def residual_l1_gate(
    prior_sample: torch.Tensor,
    base_sample: torch.Tensor,
    *,
    min_l1: float,
    max_l1: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_l1 = torch.mean(torch.abs(prior_sample - base_sample), dim=1)
    gate = torch.ones_like(residual_l1)
    if float(min_l1) > 0.0:
        gate = gate * (residual_l1 >= float(min_l1)).to(dtype=gate.dtype)
    if float(max_l1) > 0.0:
        gate = gate * (residual_l1 <= float(max_l1)).to(dtype=gate.dtype)
    return residual_l1, gate


def lowpass_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = int(kernel_size)
    if kernel <= 1:
        return image
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = F.pad(image[None], (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1, padding=0)[0]


def prepare_residual_band_views(
    prior_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    *,
    band: str,
    lowpass_kernel: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    band_name = str(band)
    if band_name == "raw":
        return prior_rgb, base_rgb, anchor_rgb
    if band_name == "lowmid":
        kernel = int(lowpass_kernel)
        return lowpass_chw(prior_rgb, kernel), lowpass_chw(base_rgb, kernel), lowpass_chw(anchor_rgb, kernel)
    raise ValueError(f"Unsupported aggregation_residual_band: {band_name}")


def save_rgb_png(path: Path, image_chw: torch.Tensor) -> None:
    array = image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB").save(path)


def save_mask_png(path: Path, mask_hw: torch.Tensor) -> None:
    array = mask_hw.detach().cpu().clamp(0.0, 1.0).numpy()
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L").save(path)


def dump_masked_prior_inputs(
    cameras: Sequence[object],
    sr_cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    output_dir: Path,
    *,
    max_views: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_cameras, dump_cache = _select_pair_uniform(cameras, sr_cache, int(max_views))
    manifest: List[Dict[str, object]] = []

    for index, (camera, cache_item) in enumerate(zip(dump_cameras, dump_cache), start=1):
        view_name = str(camera.image_name)
        stem = f"{index:03d}_{view_name}"
        prior_rgb = cache_item["prior_rgb"].to(dtype=torch.float32)
        anchor_rgb = cache_item["anchor_rgb"].to(dtype=torch.float32)
        base_rgb = cache_item["base_rgb"].to(dtype=torch.float32)
        prior_mask = cache_item["prior_mask"].to(dtype=torch.float32).clamp(0.0, 1.0)
        consistency = cache_item["consistency"].to(dtype=torch.float32)
        masked_prior = prior_rgb * prior_mask.unsqueeze(0)

        save_rgb_png(output_dir / f"{stem}_prior.png", prior_rgb)
        save_rgb_png(output_dir / f"{stem}_masked_prior.png", masked_prior)
        save_rgb_png(output_dir / f"{stem}_anchor.png", anchor_rgb)
        save_rgb_png(output_dir / f"{stem}_base.png", base_rgb)
        save_mask_png(output_dir / f"{stem}_mask.png", prior_mask)
        if consistency.numel() > 0:
            consistency_vis = consistency / max(float(consistency.max().item()), 1e-6)
            save_mask_png(output_dir / f"{stem}_consistency.png", consistency_vis)

        manifest.append(
            {
                "view_name": view_name,
                "prior_path": str(output_dir / f"{stem}_prior.png"),
                "masked_prior_path": str(output_dir / f"{stem}_masked_prior.png"),
                "mask_path": str(output_dir / f"{stem}_mask.png"),
                "anchor_path": str(output_dir / f"{stem}_anchor.png"),
                "base_path": str(output_dir / f"{stem}_base.png"),
                "consistency_path": str(output_dir / f"{stem}_consistency.png"),
                "mask_mean": float(prior_mask.mean().item()),
                "consistency_mean": float(consistency.mean().item()) if consistency.numel() > 0 else 0.0,
                "valid_ratio": float(cache_item.get("valid_ratio", 0.0) or 0.0),
            }
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"count": len(manifest), "items": manifest}, handle, indent=2, ensure_ascii=False)
    return {"dir": str(output_dir), "manifest": str(manifest_path), "count": len(manifest)}


def _face_edge_mean_lengths(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    e01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1).astype(np.float32)
    e12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1).astype(np.float32)
    e20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1).astype(np.float32)
    return ((e01 + e12 + e20) / 3.0).astype(np.float32, copy=False)


def _select_pair_uniform(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    max_items: int,
) -> Tuple[List[object], List[Dict[str, torch.Tensor | str | float | None]]]:
    if max_items <= 0 or len(cameras) <= max_items:
        return list(cameras), list(cache)
    indices = np.unique(np.linspace(0, len(cameras) - 1, num=max_items, dtype=np.int64))
    return [cameras[int(idx)] for idx in indices.tolist()], [cache[int(idx)] for idx in indices.tolist()]


def project_points_camera_torch(
    camera,
    points_xyz: torch.Tensor,
    *,
    depth_min: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r = torch.as_tensor(camera.R, device=points_xyz.device, dtype=points_xyz.dtype)
    t = torch.as_tensor(camera.T, device=points_xyz.device, dtype=points_xyz.dtype)
    xyz_cam = points_xyz @ r + t.unsqueeze(0)
    z = xyz_cam[:, 2]
    safe_z = torch.clamp_min(z, 1e-6)
    x = xyz_cam[:, 0] / safe_z * float(camera.focal_x) + float(camera.image_width) / 2.0
    y = xyz_cam[:, 1] / safe_z * float(camera.focal_y) + float(camera.image_height) / 2.0
    valid = z > float(depth_min)
    valid = valid & (x >= 0.0) & (x < float(camera.image_width)) & (y >= 0.0) & (y < float(camera.image_height))
    return torch.stack([x, y, z], dim=1), valid


def sample_chw_at_pixels(image_chw: torch.Tensor, pixel_xy: torch.Tensor) -> torch.Tensor:
    if image_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image_chw.shape)}")
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError(f"Expected xy shape [N, 2], got {tuple(pixel_xy.shape)}")
    height, width = image_chw.shape[-2:]
    if int(pixel_xy.shape[0]) <= 0:
        return torch.empty((0, image_chw.shape[0]), device=image_chw.device, dtype=image_chw.dtype)
    gx = (pixel_xy[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
    gy = (pixel_xy[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image_chw.unsqueeze(0).float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1).to(device=image_chw.device, dtype=image_chw.dtype)


def sample_hw_at_pixels(value_hw: torch.Tensor, pixel_xy: torch.Tensor) -> torch.Tensor:
    sampled = sample_chw_at_pixels(value_hw.unsqueeze(0), pixel_xy)
    return sampled[:, 0]


def build_feature_optimizer(student: GaussianModel, lr: float) -> torch.optim.Optimizer:
    student._features_dc.requires_grad_(True)
    student._features_rest.requires_grad_(False)
    student._opacity.requires_grad_(False)
    student._scaling.requires_grad_(False)
    student._rotation.requires_grad_(False)
    if isinstance(student._xyz, torch.Tensor):
        student._xyz.requires_grad_(False)
    return torch.optim.Adam([{"params": [student._features_dc], "lr": float(lr), "name": "features_dc"}], eps=1e-15)


def slice_parameter(param: torch.Tensor, valid_mask: torch.Tensor, *, requires_grad: Optional[bool] = None) -> nn.Parameter:
    if requires_grad is None:
        requires_grad = bool(param.requires_grad)
    return nn.Parameter(param.detach()[valid_mask].clone().requires_grad_(requires_grad))


@dataclass
class BoundedState:
    uid: torch.Tensor
    parent_uid: torch.Tensor
    lineage_group_id: torch.Tensor
    source_type: torch.Tensor
    seed_id: torch.Tensor
    generation: torch.Tensor
    face_ids: torch.Tensor
    bary_coords: torch.Tensor
    surface_points: torch.Tensor
    normals: torch.Tensor
    tangent_u: torch.Tensor
    tangent_v: torch.Tensor
    u0: torch.Tensor
    v0: torch.Tensor
    d0: torch.Tensor
    u_current: torch.Tensor
    v_current: torch.Tensor
    d_current: torch.Tensor
    tau_surface: torch.Tensor
    d_norm: torch.Tensor
    trust_class: torch.Tensor
    selected_mask: torch.Tensor
    bad_streak: torch.Tensor

    def clone_current_xyz(self) -> torch.Tensor:
        return (
            self.surface_points
            + self.tangent_u * self.u_current[:, None]
            + self.tangent_v * self.v_current[:, None]
            + self.normals * self.d_current[:, None]
        )

    def compose_xyz(self, u_value: torch.Tensor, v_value: torch.Tensor, d_value: torch.Tensor) -> torch.Tensor:
        return (
            self.surface_points
            + self.tangent_u * u_value[:, None]
            + self.tangent_v * v_value[:, None]
            + self.normals * d_value[:, None]
        )

    def subset(self, valid_mask: torch.Tensor) -> "BoundedState":
        def _slice(value: torch.Tensor) -> torch.Tensor:
            return value[valid_mask].clone()

        return BoundedState(
            uid=_slice(self.uid),
            parent_uid=_slice(self.parent_uid),
            lineage_group_id=_slice(self.lineage_group_id),
            source_type=_slice(self.source_type),
            seed_id=_slice(self.seed_id),
            generation=_slice(self.generation),
            face_ids=_slice(self.face_ids),
            bary_coords=_slice(self.bary_coords),
            surface_points=_slice(self.surface_points),
            normals=_slice(self.normals),
            tangent_u=_slice(self.tangent_u),
            tangent_v=_slice(self.tangent_v),
            u0=_slice(self.u0),
            v0=_slice(self.v0),
            d0=_slice(self.d0),
            u_current=_slice(self.u_current),
            v_current=_slice(self.v_current),
            d_current=_slice(self.d_current),
            tau_surface=_slice(self.tau_surface),
            d_norm=_slice(self.d_norm),
            trust_class=_slice(self.trust_class),
            selected_mask=_slice(self.selected_mask),
            bad_streak=_slice(self.bad_streak),
        )

    def save_payload(self, path: Path, *, initial_uid_count: int) -> None:
        uid_to_active = torch.full((int(initial_uid_count),), -1, dtype=torch.int64)
        uid_to_active[self.uid.detach().cpu()] = torch.arange(self.uid.shape[0], dtype=torch.int64)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "active_index_to_uid": self.uid.detach().cpu(),
                "uid_to_active_index": uid_to_active,
                "parent_uid": self.parent_uid.detach().cpu(),
                "lineage_group_id": self.lineage_group_id.detach().cpu(),
                "source_type": self.source_type.detach().cpu(),
                "seed_id": self.seed_id.detach().cpu(),
                "generation": self.generation.detach().cpu(),
                "face_ids": self.face_ids.detach().cpu(),
                "bary_coords": self.bary_coords.detach().cpu(),
                "surface_points": self.surface_points.detach().cpu(),
                "normals": self.normals.detach().cpu(),
                "tangent_u": self.tangent_u.detach().cpu(),
                "tangent_v": self.tangent_v.detach().cpu(),
                "u0": self.u0.detach().cpu(),
                "v0": self.v0.detach().cpu(),
                "d0": self.d0.detach().cpu(),
                "u_current": self.u_current.detach().cpu(),
                "v_current": self.v_current.detach().cpu(),
                "d_current": self.d_current.detach().cpu(),
                "tau_surface": self.tau_surface.detach().cpu(),
                "d_norm": self.d_norm.detach().cpu(),
                "trust_class": self.trust_class.detach().cpu(),
                "selected_mask": self.selected_mask.detach().cpu(),
                "bad_streak": self.bad_streak.detach().cpu(),
            },
            path,
        )


@dataclass
class AggregatedTargets:
    target_rgb: torch.Tensor
    base_rgb: torch.Tensor
    anchor_rgb: torch.Tensor
    confidence: torch.Tensor
    disagreement: torch.Tensor
    view_count: torch.Tensor
    weight_sum: torch.Tensor
    valid_mask: torch.Tensor
    summary: Dict[str, object]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            target_rgb=self.target_rgb.detach().cpu().numpy().astype(np.float32),
            base_rgb=self.base_rgb.detach().cpu().numpy().astype(np.float32),
            anchor_rgb=self.anchor_rgb.detach().cpu().numpy().astype(np.float32),
            confidence=self.confidence.detach().cpu().numpy().astype(np.float32),
            disagreement=self.disagreement.detach().cpu().numpy().astype(np.float32),
            view_count=self.view_count.detach().cpu().numpy().astype(np.int32),
            weight_sum=self.weight_sum.detach().cpu().numpy().astype(np.float32),
            valid_mask=self.valid_mask.detach().cpu().numpy().astype(np.uint8),
        )


@dataclass
class SurfacePriorMemory:
    target_rgb: torch.Tensor
    base_rgb: torch.Tensor
    anchor_rgb: torch.Tensor
    confidence: torch.Tensor
    disagreement: torch.Tensor
    view_count: torch.Tensor
    weight_sum: torch.Tensor
    update_count: torch.Tensor
    state: torch.Tensor

    @classmethod
    def create(cls, count: int, device: torch.device) -> "SurfacePriorMemory":
        return cls(
            target_rgb=torch.zeros((count, 3), dtype=torch.float32, device=device),
            base_rgb=torch.zeros((count, 3), dtype=torch.float32, device=device),
            anchor_rgb=torch.zeros((count, 3), dtype=torch.float32, device=device),
            confidence=torch.zeros((count,), dtype=torch.float32, device=device),
            disagreement=torch.zeros((count,), dtype=torch.float32, device=device),
            view_count=torch.zeros((count,), dtype=torch.int32, device=device),
            weight_sum=torch.zeros((count,), dtype=torch.float32, device=device),
            update_count=torch.zeros((count,), dtype=torch.int32, device=device),
            state=torch.full((count,), MEM_OBSERVE, dtype=torch.int32, device=device),
        )

    def subset(self, valid_mask: torch.Tensor) -> "SurfacePriorMemory":
        return SurfacePriorMemory(
            target_rgb=self.target_rgb[valid_mask].clone(),
            base_rgb=self.base_rgb[valid_mask].clone(),
            anchor_rgb=self.anchor_rgb[valid_mask].clone(),
            confidence=self.confidence[valid_mask].clone(),
            disagreement=self.disagreement[valid_mask].clone(),
            view_count=self.view_count[valid_mask].clone(),
            weight_sum=self.weight_sum[valid_mask].clone(),
            update_count=self.update_count[valid_mask].clone(),
            state=self.state[valid_mask].clone(),
        )

    def update(
        self,
        aggregated: AggregatedTargets,
        selected_mask: torch.Tensor,
        *,
        beta_max: float,
        min_confidence: float,
        max_disagreement: float,
        min_views: int,
        stable_updates: int,
        stable_min_confidence: float,
        stable_max_disagreement: float,
    ) -> Dict[str, object]:
        reliable = (
            selected_mask
            & aggregated.valid_mask
            & (aggregated.confidence >= float(min_confidence))
            & (aggregated.disagreement <= float(max_disagreement))
            & (aggregated.view_count >= int(min_views))
        )
        reliable_ids = torch.nonzero(reliable, as_tuple=False).squeeze(1).to(dtype=torch.int64)
        if int(reliable_ids.numel()) > 0:
            old_count = self.update_count[reliable_ids]
            first_update = old_count <= 0
            beta = (float(beta_max) * aggregated.confidence[reliable_ids].clamp(0.0, 1.0)).clamp(0.0, float(beta_max))
            beta_rgb = beta[:, None]

            self.target_rgb[reliable_ids] = torch.where(
                first_update[:, None],
                aggregated.target_rgb[reliable_ids],
                (1.0 - beta_rgb) * self.target_rgb[reliable_ids] + beta_rgb * aggregated.target_rgb[reliable_ids],
            )
            self.base_rgb[reliable_ids] = torch.where(
                first_update[:, None],
                aggregated.base_rgb[reliable_ids],
                (1.0 - beta_rgb) * self.base_rgb[reliable_ids] + beta_rgb * aggregated.base_rgb[reliable_ids],
            )
            self.anchor_rgb[reliable_ids] = torch.where(
                first_update[:, None],
                aggregated.anchor_rgb[reliable_ids],
                (1.0 - beta_rgb) * self.anchor_rgb[reliable_ids] + beta_rgb * aggregated.anchor_rgb[reliable_ids],
            )
            self.confidence[reliable_ids] = torch.where(
                first_update,
                aggregated.confidence[reliable_ids],
                (1.0 - beta) * self.confidence[reliable_ids] + beta * aggregated.confidence[reliable_ids],
            )
            self.disagreement[reliable_ids] = torch.where(
                first_update,
                aggregated.disagreement[reliable_ids],
                (1.0 - beta) * self.disagreement[reliable_ids] + beta * aggregated.disagreement[reliable_ids],
            )
            self.view_count[reliable_ids] = torch.maximum(self.view_count[reliable_ids], aggregated.view_count[reliable_ids])
            self.weight_sum[reliable_ids] = torch.maximum(self.weight_sum[reliable_ids], aggregated.weight_sum[reliable_ids])
            self.update_count[reliable_ids] += 1

        has_memory = self.update_count > 0
        color_ready = selected_mask & has_memory & (self.confidence >= float(min_confidence))
        color_stable = (
            color_ready
            & (self.update_count >= int(stable_updates))
            & (self.confidence >= float(stable_min_confidence))
            & (self.disagreement <= float(stable_max_disagreement))
        )
        no_memory_low_conf = selected_mask & (~has_memory)
        self.state[no_memory_low_conf] = MEM_LOW_CONF
        self.state[color_ready] = MEM_COLOR_READY
        self.state[color_stable] = MEM_COLOR_STABLE
        self.state[~selected_mask] = MEM_OBSERVE

        return {
            "reliable_updates": int(reliable.sum().item()),
            "state_counts": self.state_counts(),
            "memory_confidence": tensor_stats(self.confidence[selected_mask]) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
            "memory_disagreement": tensor_stats(self.disagreement[selected_mask]) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
        }

    def state_counts(self) -> Dict[str, int]:
        return {
            "observe": int(torch.count_nonzero(self.state == MEM_OBSERVE).item()),
            "color_ready": int(torch.count_nonzero(self.state == MEM_COLOR_READY).item()),
            "color_stable": int(torch.count_nonzero(self.state == MEM_COLOR_STABLE).item()),
            "low_conf": int(torch.count_nonzero(self.state == MEM_LOW_CONF).item()),
            "rejected": int(torch.count_nonzero(self.state == MEM_REJECTED).item()),
        }

    def to_aggregated(self, selected_mask: torch.Tensor, active_states: Sequence[int], *, summary_name: str) -> AggregatedTargets:
        active_state_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
        for state_value in active_states:
            active_state_mask |= self.state == int(state_value)
        valid_mask = selected_mask & active_state_mask & (self.update_count > 0)
        summary = {
            "name": str(summary_name),
            "count": int(selected_mask.shape[0]),
            "selected": int(selected_mask.sum().item()),
            "observed": int(torch.count_nonzero(self.update_count > 0).item()),
            "valid": int(valid_mask.sum().item()),
            "mean_confidence": float(self.confidence[valid_mask].mean().item()) if torch.any(valid_mask) else 0.0,
            "mean_disagreement": float(self.disagreement[valid_mask].mean().item()) if torch.any(valid_mask) else 0.0,
            "state_counts": self.state_counts(),
        }
        return AggregatedTargets(
            target_rgb=self.target_rgb,
            base_rgb=self.base_rgb,
            anchor_rgb=self.anchor_rgb,
            confidence=self.confidence,
            disagreement=self.disagreement,
            view_count=self.view_count,
            weight_sum=self.weight_sum,
            valid_mask=valid_mask,
            summary=summary,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "target_rgb": self.target_rgb.detach().cpu(),
                "base_rgb": self.base_rgb.detach().cpu(),
                "anchor_rgb": self.anchor_rgb.detach().cpu(),
                "confidence": self.confidence.detach().cpu(),
                "disagreement": self.disagreement.detach().cpu(),
                "view_count": self.view_count.detach().cpu(),
                "weight_sum": self.weight_sum.detach().cpu(),
                "update_count": self.update_count.detach().cpu(),
                "state": self.state.detach().cpu(),
                "state_counts": self.state_counts(),
                "state_legend": {
                    "OBSERVE": MEM_OBSERVE,
                    "COLOR_READY": MEM_COLOR_READY,
                    "COLOR_STABLE": MEM_COLOR_STABLE,
                    "LOW_CONF": MEM_LOW_CONF,
                    "REJECTED": MEM_REJECTED,
                },
            },
            path,
        )


def build_bounded_state(
    student: GaussianModel,
    mesh_path: str,
    *,
    face_k: int,
    chunk_size: int,
    tau_floor: float,
    tau_edge_scale: float,
) -> BoundedState:
    mesh = load_triangle_mesh(mesh_path)
    face_mean_edge = _face_edge_mean_lengths(mesh)
    xyz = student.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    bound = bind_gaussians_to_mesh(xyz, mesh, face_k=int(face_k), chunk_size=int(chunk_size))

    device = student.get_xyz.device
    uid = torch.arange(xyz.shape[0], dtype=torch.int64, device=device)
    seed_id = student._seed_id.detach().clone()
    lineage_group_id = torch.where(seed_id >= 0, seed_id, uid)
    parent_uid = torch.full_like(uid, -1)
    face_ids = torch.from_numpy(bound["face_ids"]).to(device=device, dtype=torch.int64)
    tau_surface = torch.from_numpy(
        np.maximum(face_mean_edge[bound["face_ids"]] * float(tau_edge_scale), float(tau_floor)).astype(np.float32)
    ).to(device=device)
    d0 = torch.from_numpy(bound["normal_offset"]).to(device=device, dtype=torch.float32)
    d_norm = torch.abs(d0) / torch.clamp(tau_surface, min=1e-6)
    trust_class = torch.full((xyz.shape[0],), TRUST_SURFACE, dtype=torch.int64, device=device)
    trust_class[d_norm > 1.5] = TRUST_LOOSE
    trust_class[d_norm > 3.0] = TRUST_OUTLIER
    selected_mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=device)
    bad_streak = torch.zeros((xyz.shape[0],), dtype=torch.int32, device=device)
    return BoundedState(
        uid=uid,
        parent_uid=parent_uid,
        lineage_group_id=lineage_group_id.detach().clone(),
        source_type=student._source_tag.detach().clone(),
        seed_id=seed_id.detach().clone(),
        generation=student._generation.detach().clone(),
        face_ids=face_ids,
        bary_coords=torch.from_numpy(bound["bary_coords"]).to(device=device, dtype=torch.float32),
        surface_points=torch.from_numpy(bound["surface_points"]).to(device=device, dtype=torch.float32),
        normals=torch.from_numpy(bound["normals"]).to(device=device, dtype=torch.float32),
        tangent_u=torch.from_numpy(bound["tangent_u"]).to(device=device, dtype=torch.float32),
        tangent_v=torch.from_numpy(bound["tangent_v"]).to(device=device, dtype=torch.float32),
        u0=torch.from_numpy(bound["tangent_offset_u"]).to(device=device, dtype=torch.float32),
        v0=torch.from_numpy(bound["tangent_offset_v"]).to(device=device, dtype=torch.float32),
        d0=d0,
        u_current=torch.from_numpy(bound["tangent_offset_u"]).to(device=device, dtype=torch.float32),
        v_current=torch.from_numpy(bound["tangent_offset_v"]).to(device=device, dtype=torch.float32),
        d_current=d0.clone(),
        tau_surface=tau_surface,
        d_norm=d_norm,
        trust_class=trust_class,
        selected_mask=selected_mask,
        bad_streak=bad_streak,
    )


def build_memory_eligible_mask(
    bound: BoundedState,
    prefilter: Dict[str, torch.Tensor | Dict[str, object]],
    *,
    mode: str,
    loose_min_visible_views: int,
    loose_min_touch_views: int,
    loose_min_touch_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    candidate = bound.selected_mask
    device = candidate.device
    trust_surface = bound.trust_class == TRUST_SURFACE
    trust_loose = bound.trust_class == TRUST_LOOSE
    trust_outlier = bound.trust_class == TRUST_OUTLIER

    visible_view_count = prefilter.get("visible_view_count")
    touch_view_count = prefilter.get("touch_view_count")
    touch_ratio = prefilter.get("touch_ratio")
    if torch.is_tensor(visible_view_count):
        visible_view_count = visible_view_count.to(device=device).reshape(-1)
    else:
        visible_view_count = torch.zeros_like(bound.uid, dtype=torch.int32)
    if torch.is_tensor(touch_view_count):
        touch_view_count = touch_view_count.to(device=device).reshape(-1)
    else:
        touch_view_count = torch.zeros_like(bound.uid, dtype=torch.int32)
    if torch.is_tensor(touch_ratio):
        touch_ratio = touch_ratio.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        touch_ratio = torch.zeros_like(bound.d0, dtype=torch.float32)

    loose_visible = (
        trust_loose
        & (visible_view_count >= int(loose_min_visible_views))
        & (touch_view_count >= int(loose_min_touch_views))
        & (touch_ratio >= float(loose_min_touch_ratio))
    )
    mode = str(mode)
    if mode == "selected":
        eligible = candidate
    elif mode == "trust_surface":
        eligible = candidate & trust_surface
    elif mode == "non_outlier":
        eligible = candidate & (~trust_outlier)
    elif mode == "trust_surface_or_visible_loose":
        eligible = candidate & (trust_surface | loose_visible)
    elif mode == "trust_loose_only":
        eligible = candidate & trust_loose
    elif mode == "outlier_only":
        eligible = candidate & trust_outlier
    else:
        raise ValueError(f"Unsupported memory eligibility mode: {mode}")

    summary = {
        "mode": mode,
        "candidate_selected": int(candidate.sum().item()),
        "eligible": int(eligible.sum().item()),
        "eligible_ratio_of_candidate": float(eligible.sum().item() / max(float(candidate.sum().item()), 1.0)),
        "eligible_trust_surface": int((eligible & trust_surface).sum().item()),
        "eligible_trust_loose": int((eligible & trust_loose).sum().item()),
        "eligible_trust_outlier": int((eligible & trust_outlier).sum().item()),
        "candidate_trust_surface": int((candidate & trust_surface).sum().item()),
        "candidate_trust_loose": int((candidate & trust_loose).sum().item()),
        "candidate_trust_outlier": int((candidate & trust_outlier).sum().item()),
        "loose_visible_candidates": int((candidate & loose_visible).sum().item()),
        "loose_min_visible_views": int(loose_min_visible_views),
        "loose_min_touch_views": int(loose_min_touch_views),
        "loose_min_touch_ratio": float(loose_min_touch_ratio),
    }
    return eligible, summary


def aggregate_surface_targets(
    cameras: Sequence[object],
    sr_cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    bound: BoundedState,
    *,
    selected_mask_override: Optional[torch.Tensor] = None,
    depth_min: float,
    min_support_views: int,
    min_sample_weight: float,
    agreement_sigma: float,
    base_sigma: float,
    base_support_mode: str,
    base_support_floor: float,
    residual_sample_clip: float,
    residual_band: str,
    residual_lowpass_kernel: int,
    residual_min_l1: float,
    residual_max_l1: float,
    disagreement_sigma: float,
    aggregation_mode: str,
    robust_trim_sigma: float,
    robust_trim_disagreement_scale: float,
) -> AggregatedTargets:
    device = bound.surface_points.device
    total = int(bound.uid.shape[0])
    target_rgb = torch.zeros((total, 3), dtype=torch.float32, device=device)
    base_rgb = torch.zeros((total, 3), dtype=torch.float32, device=device)
    anchor_rgb = torch.zeros((total, 3), dtype=torch.float32, device=device)
    confidence = torch.zeros((total,), dtype=torch.float32, device=device)
    disagreement = torch.zeros((total,), dtype=torch.float32, device=device)
    view_count = torch.zeros((total,), dtype=torch.int32, device=device)
    weight_sum = torch.zeros((total,), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((total,), dtype=torch.bool, device=device)

    selected_mask = bound.selected_mask if selected_mask_override is None else selected_mask_override.to(device=device, dtype=torch.bool).reshape(-1)
    selected_ids = torch.nonzero(selected_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    if int(selected_ids.numel()) <= 0:
        summary = {
            "count": total,
            "selected": 0,
            "observed": 0,
            "valid": 0,
            "mean_confidence": 0.0,
            "mean_disagreement": 0.0,
            "candidate_selected": int(bound.selected_mask.sum().item()),
        }
        return AggregatedTargets(
            target_rgb=target_rgb,
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            confidence=confidence,
            disagreement=disagreement,
            view_count=view_count,
            weight_sum=weight_sum,
            valid_mask=valid_mask,
            summary=summary,
        )

    xyz_selected = bound.clone_current_xyz()[selected_ids]
    local_count = int(selected_ids.numel())
    local_weight_sum = torch.zeros((local_count,), dtype=torch.float32, device=device)
    local_view_count = torch.zeros((local_count,), dtype=torch.int32, device=device)
    local_rgb_sum = torch.zeros((local_count, 3), dtype=torch.float32, device=device)
    local_base_sum = torch.zeros((local_count, 3), dtype=torch.float32, device=device)
    local_anchor_sum = torch.zeros((local_count, 3), dtype=torch.float32, device=device)
    local_rgb_sq_sum = torch.zeros((local_count, 3), dtype=torch.float32, device=device)

    for camera, cache_item in tqdm(list(zip(cameras, sr_cache)), desc="aggregate surface targets"):
        prior_rgb = cache_item["prior_rgb"].to(device=device, dtype=torch.float32)
        anchor_view = cache_item["anchor_rgb"].to(device=device, dtype=torch.float32)
        base_view = cache_item["base_rgb"].to(device=device, dtype=torch.float32)
        prior_residual_view, base_residual_view, anchor_residual_view = prepare_residual_band_views(
            prior_rgb,
            base_view,
            anchor_view,
            band=str(residual_band),
            lowpass_kernel=int(residual_lowpass_kernel),
        )
        prior_mask = cache_item["prior_mask"].to(device=device, dtype=torch.float32)

        proj, valid = project_points_camera_torch(camera, xyz_selected, depth_min=float(depth_min))
        if not torch.any(valid):
            continue
        valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(1).to(dtype=torch.int64)
        xy = proj[valid_ids, :2]
        prior_sample = sample_chw_at_pixels(prior_residual_view, xy)
        anchor_sample = sample_chw_at_pixels(anchor_residual_view, xy)
        base_sample = sample_chw_at_pixels(base_residual_view, xy)
        mask_sample = sample_hw_at_pixels(prior_mask, xy).clamp(0.0, 1.0)
        agreement = torch.exp(-torch.mean(torch.abs(prior_sample - anchor_sample), dim=1) / max(float(agreement_sigma), 1e-6))
        base_support = torch.exp(-torch.mean(torch.abs(prior_sample - base_sample), dim=1) / max(float(base_sigma), 1e-6))
        _, residual_gate = residual_l1_gate(
            prior_sample,
            base_sample,
            min_l1=float(residual_min_l1),
            max_l1=float(residual_max_l1),
        )
        base_weight = apply_base_support_mode(base_support, mode=str(base_support_mode), floor=float(base_support_floor))
        sample_weight = mask_sample * agreement * base_weight * residual_gate
        good = sample_weight > float(min_sample_weight)
        if not torch.any(good):
            continue

        local_ids = valid_ids[good]
        target_sample = apply_residual_sample_clip(prior_sample, base_sample, float(residual_sample_clip))
        rgb = target_sample[good]
        base_rgb_view = base_sample[good]
        anchor_rgb_view = anchor_sample[good]
        weight = sample_weight[good]
        local_weight_sum[local_ids] += weight
        local_view_count[local_ids] += 1
        local_rgb_sum[local_ids] += rgb * weight[:, None]
        local_base_sum[local_ids] += base_rgb_view * weight[:, None]
        local_anchor_sum[local_ids] += anchor_rgb_view * weight[:, None]
        local_rgb_sq_sum[local_ids] += rgb * rgb * weight[:, None]

    observed = local_weight_sum > 0.0
    if torch.any(observed):
        denom = torch.clamp(local_weight_sum[observed], min=1e-6)[:, None]
        local_target = local_rgb_sum[observed] / denom
        local_base = local_base_sum[observed] / denom
        local_anchor = local_anchor_sum[observed] / denom
        local_var = torch.clamp(local_rgb_sq_sum[observed] / denom - local_target * local_target, min=0.0)
        local_disagreement = torch.sqrt(local_var.mean(dim=1) + 1e-8)

        if str(aggregation_mode) == "trimmed_mean" and float(robust_trim_sigma) > 0.0:
            preliminary_target = torch.zeros_like(local_rgb_sum)
            preliminary_disagreement = torch.zeros((local_count,), dtype=torch.float32, device=device)
            preliminary_target[observed] = local_target
            preliminary_disagreement[observed] = local_disagreement
            trim_weight_sum = torch.zeros_like(local_weight_sum)
            trim_view_count = torch.zeros_like(local_view_count)
            trim_rgb_sum = torch.zeros_like(local_rgb_sum)
            trim_base_sum = torch.zeros_like(local_base_sum)
            trim_anchor_sum = torch.zeros_like(local_anchor_sum)
            trim_rgb_sq_sum = torch.zeros_like(local_rgb_sq_sum)

            for camera, cache_item in tqdm(list(zip(cameras, sr_cache)), desc="trim surface targets"):
                prior_rgb = cache_item["prior_rgb"].to(device=device, dtype=torch.float32)
                anchor_view = cache_item["anchor_rgb"].to(device=device, dtype=torch.float32)
                base_view = cache_item["base_rgb"].to(device=device, dtype=torch.float32)
                prior_residual_view, base_residual_view, anchor_residual_view = prepare_residual_band_views(
                    prior_rgb,
                    base_view,
                    anchor_view,
                    band=str(residual_band),
                    lowpass_kernel=int(residual_lowpass_kernel),
                )
                prior_mask = cache_item["prior_mask"].to(device=device, dtype=torch.float32)

                proj, valid = project_points_camera_torch(camera, xyz_selected, depth_min=float(depth_min))
                if not torch.any(valid):
                    continue
                valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(1).to(dtype=torch.int64)
                xy = proj[valid_ids, :2]
                prior_sample = sample_chw_at_pixels(prior_residual_view, xy)
                anchor_sample = sample_chw_at_pixels(anchor_residual_view, xy)
                base_sample = sample_chw_at_pixels(base_residual_view, xy)
                mask_sample = sample_hw_at_pixels(prior_mask, xy).clamp(0.0, 1.0)
                agreement = torch.exp(-torch.mean(torch.abs(prior_sample - anchor_sample), dim=1) / max(float(agreement_sigma), 1e-6))
                base_support = torch.exp(-torch.mean(torch.abs(prior_sample - base_sample), dim=1) / max(float(base_sigma), 1e-6))
                _, residual_gate = residual_l1_gate(
                    prior_sample,
                    base_sample,
                    min_l1=float(residual_min_l1),
                    max_l1=float(residual_max_l1),
                )
                base_weight = apply_base_support_mode(base_support, mode=str(base_support_mode), floor=float(base_support_floor))
                sample_weight = mask_sample * agreement * base_weight * residual_gate
                good = sample_weight > float(min_sample_weight)
                if not torch.any(good):
                    continue

                local_ids = valid_ids[good]
                target_sample = apply_residual_sample_clip(prior_sample, base_sample, float(residual_sample_clip))
                rgb = target_sample[good]
                residual = torch.mean(torch.abs(rgb - preliminary_target[local_ids]), dim=1)
                trim_threshold = torch.maximum(
                    torch.full_like(residual, float(robust_trim_sigma)),
                    preliminary_disagreement[local_ids] * float(robust_trim_disagreement_scale),
                )
                keep = observed[local_ids] & (residual <= trim_threshold)
                if not torch.any(keep):
                    continue

                local_ids = local_ids[keep]
                rgb = rgb[keep]
                base_rgb_view = base_sample[good][keep]
                anchor_rgb_view = anchor_sample[good][keep]
                weight = sample_weight[good][keep]
                trim_weight_sum[local_ids] += weight
                trim_view_count[local_ids] += 1
                trim_rgb_sum[local_ids] += rgb * weight[:, None]
                trim_base_sum[local_ids] += base_rgb_view * weight[:, None]
                trim_anchor_sum[local_ids] += anchor_rgb_view * weight[:, None]
                trim_rgb_sq_sum[local_ids] += rgb * rgb * weight[:, None]

            trim_observed = trim_weight_sum > 0.0
            if torch.any(trim_observed):
                trim_denom = torch.clamp(trim_weight_sum[trim_observed], min=1e-6)[:, None]
                trim_target = trim_rgb_sum[trim_observed] / trim_denom
                trim_base = trim_base_sum[trim_observed] / trim_denom
                trim_anchor = trim_anchor_sum[trim_observed] / trim_denom
                trim_var = torch.clamp(trim_rgb_sq_sum[trim_observed] / trim_denom - trim_target * trim_target, min=0.0)
                trim_disagreement = torch.sqrt(trim_var.mean(dim=1) + 1e-8)
                local_target = trim_target
                local_base = trim_base
                local_anchor = trim_anchor
                local_disagreement = trim_disagreement
                observed = trim_observed
                local_weight_sum = trim_weight_sum
                local_view_count = trim_view_count

    if not torch.any(observed):
        summary = {
            "count": total,
            "selected": int(selected_mask.sum().item()),
            "observed": 0,
            "valid": 0,
            "mean_confidence": 0.0,
            "mean_disagreement": 0.0,
            "candidate_selected": int(bound.selected_mask.sum().item()),
            "selected_view_count": tensor_stats(view_count[selected_mask].to(dtype=torch.float32)) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
            "selected_confidence": tensor_stats(confidence[selected_mask]) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
            "selected_disagreement": tensor_stats(disagreement[selected_mask]) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
        }
        return AggregatedTargets(
            target_rgb=target_rgb,
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            confidence=confidence,
            disagreement=disagreement,
            view_count=view_count,
            weight_sum=weight_sum,
            valid_mask=valid_mask,
            summary=summary,
        )

    local_view_gate = torch.clamp(
        local_view_count[observed].to(dtype=torch.float32) / max(float(min_support_views), 1.0),
        min=0.0,
        max=1.0,
    )
    local_mean_weight = local_weight_sum[observed] / torch.clamp(local_view_count[observed].to(dtype=torch.float32), min=1.0)
    local_confidence = local_mean_weight * torch.exp(-local_disagreement / max(float(disagreement_sigma), 1e-6)) * local_view_gate

    active_ids = selected_ids[observed]
    target_rgb[active_ids] = local_target
    base_rgb[active_ids] = local_base
    anchor_rgb[active_ids] = local_anchor
    disagreement[active_ids] = local_disagreement
    confidence[active_ids] = local_confidence
    weight_sum[active_ids] = local_weight_sum[observed]
    view_count[active_ids] = local_view_count[observed]
    valid_mask[active_ids] = local_view_count[observed] >= int(min_support_views)

    selected_valid = valid_mask & selected_mask
    summary = {
        "count": total,
        "selected": int(selected_mask.sum().item()),
        "candidate_selected": int(bound.selected_mask.sum().item()),
        "observed": int(torch.count_nonzero(observed).item()),
        "valid": int(selected_valid.sum().item()),
        "mean_confidence": float(confidence[selected_valid].mean().item()) if torch.any(selected_valid) else 0.0,
        "mean_disagreement": float(disagreement[selected_valid].mean().item()) if torch.any(selected_valid) else 0.0,
        "selected_view_count": tensor_stats(view_count[selected_mask].to(dtype=torch.float32)) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
        "selected_confidence": tensor_stats(confidence[selected_mask]) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
        "selected_disagreement": tensor_stats(disagreement[selected_mask]) if torch.any(selected_mask) else stats_from_array(np.asarray([], dtype=np.float32)),
        "base_support_mode": str(base_support_mode),
        "base_support_floor": float(base_support_floor),
        "residual_sample_clip": float(residual_sample_clip),
        "residual_band": str(residual_band),
        "residual_lowpass_kernel": int(residual_lowpass_kernel),
        "residual_min_l1": float(residual_min_l1),
        "residual_max_l1": float(residual_max_l1),
    }
    return AggregatedTargets(
        target_rgb=target_rgb,
        base_rgb=base_rgb,
        anchor_rgb=anchor_rgb,
        confidence=confidence,
        disagreement=disagreement,
        view_count=view_count,
        weight_sum=weight_sum,
        valid_mask=valid_mask,
        summary=summary,
    )


@torch.no_grad()
def build_appearance_target_rgb(
    aggregated: AggregatedTargets,
    dc_anchor_ref: torch.Tensor,
    *,
    target_mode: str,
    residual_clip: float,
    residual_scale: float,
) -> torch.Tensor:
    mode = str(target_mode)
    target_rgb = aggregated.target_rgb.detach()
    if mode == "absolute":
        return target_rgb
    if mode == "residual_clipped":
        anchor_rgb = features_dc_to_rgb(dc_anchor_ref.detach()).to(device=target_rgb.device, dtype=target_rgb.dtype)
        residual = target_rgb - aggregated.base_rgb.detach()
        clip = float(residual_clip)
        if clip > 0.0:
            residual = torch.clamp(residual, min=-clip, max=clip)
        return torch.clamp(anchor_rgb + float(residual_scale) * residual, min=0.0, max=1.0).detach()
    raise ValueError(f"Unsupported appearance_target_mode: {mode}")


@torch.no_grad()
def evaluate_color_residual(
    student: GaussianModel,
    aggregated: AggregatedTargets,
    bound: BoundedState,
    dc_anchor_ref: torch.Tensor,
    *,
    target_mode: str,
    residual_clip: float,
    residual_scale: float,
) -> float:
    active = bound.selected_mask & aggregated.valid_mask
    if not torch.any(active):
        return 0.0
    rgb = features_dc_to_rgb(student._features_dc.detach())[active]
    target = build_appearance_target_rgb(
        aggregated,
        dc_anchor_ref,
        target_mode=str(target_mode),
        residual_clip=float(residual_clip),
        residual_scale=float(residual_scale),
    )[active]
    weight = aggregated.confidence[active]
    loss = weighted_mean(torch.mean(torch.abs(rgb - target), dim=1), torch.clamp(weight, min=1e-6))
    return float(loss.detach().item())


def run_appearance_phase(
    student: GaussianModel,
    aggregated: AggregatedTargets,
    bound: BoundedState,
    *,
    steps: int,
    lr: float,
    dc_anchor_ref: torch.Tensor,
    lambda_anchor: float,
    lambda_base_guard: float,
    target_mode: str,
    residual_clip: float,
    residual_scale: float,
    charbonnier_eps: float,
) -> Dict[str, float | int]:
    before = evaluate_color_residual(
        student,
        aggregated,
        bound,
        dc_anchor_ref,
        target_mode=str(target_mode),
        residual_clip=float(residual_clip),
        residual_scale=float(residual_scale),
    )
    active = bound.selected_mask & aggregated.valid_mask
    if int(steps) <= 0 or not torch.any(active):
        return {
            "before": before,
            "after": before,
            "steps": 0,
            "target_mode": str(target_mode),
            "residual_clip": float(residual_clip),
            "residual_scale": float(residual_scale),
        }

    optimizer = build_feature_optimizer(student, lr=float(lr))
    target_rgb = build_appearance_target_rgb(
        aggregated,
        dc_anchor_ref,
        target_mode=str(target_mode),
        residual_clip=float(residual_clip),
        residual_scale=float(residual_scale),
    )
    base_rgb = aggregated.base_rgb.detach()
    confidence = aggregated.confidence.detach()
    target_dc = rgb_to_features_dc_like(target_rgb, student._features_dc.detach())

    active_ids = torch.nonzero(active, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    for _ in tqdm(range(int(steps)), desc="appearance phase"):
        optimizer.zero_grad(set_to_none=True)
        current_rgb = features_dc_to_rgb(student._features_dc)[active_ids]
        current_dc = student._features_dc[active_ids]
        target_rgb_active = target_rgb[active_ids]
        base_rgb_active = base_rgb[active_ids]
        target_dc_active = target_dc[active_ids]
        weight = torch.clamp(confidence[active_ids], min=1e-6)

        loss_color = weighted_mean(charbonnier_rows(current_rgb - target_rgb_active, charbonnier_eps), weight)
        loss_anchor = torch.mean(torch.abs(current_dc - dc_anchor_ref[active_ids]))
        loss_base = weighted_mean(torch.mean(torch.abs(current_rgb - base_rgb_active), dim=1), weight)
        loss_dc_target = torch.mean(torch.abs(current_dc - target_dc_active))
        loss = loss_color + float(lambda_anchor) * loss_anchor + float(lambda_base_guard) * loss_base + 0.25 * loss_dc_target
        loss.backward()
        optimizer.step()

    after = evaluate_color_residual(
        student,
        aggregated,
        bound,
        dc_anchor_ref,
        target_mode=str(target_mode),
        residual_clip=float(residual_clip),
        residual_scale=float(residual_scale),
    )
    return {
        "before": before,
        "after": after,
        "steps": int(steps),
        "target_mode": str(target_mode),
        "residual_clip": float(residual_clip),
        "residual_scale": float(residual_scale),
    }


@torch.no_grad()
def evaluate_geometry_residual(
    cameras: Sequence[object],
    sr_cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    bound: BoundedState,
    dc_rgb_frozen: torch.Tensor,
    *,
    depth_min: float,
    geometry_active_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    geom_mask = bound.selected_mask & (bound.trust_class != TRUST_OUTLIER)
    if geometry_active_mask is not None:
        geom_mask = geom_mask & geometry_active_mask.to(device=geom_mask.device, dtype=torch.bool)
    geom_ids = torch.nonzero(geom_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    if int(geom_ids.numel()) <= 0:
        return {"photo": 0.0, "samples": 0}

    xyz = bound.clone_current_xyz()[geom_ids]
    dc_rgb = dc_rgb_frozen[geom_ids]
    loss_acc = 0.0
    sample_count = 0
    for camera, cache_item in zip(cameras, sr_cache):
        prior_rgb = cache_item["prior_rgb"].to(device=xyz.device, dtype=torch.float32)
        prior_mask = cache_item["prior_mask"].to(device=xyz.device, dtype=torch.float32)
        proj, valid = project_points_camera_torch(camera, xyz, depth_min=float(depth_min))
        if not torch.any(valid):
            continue
        valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(1).to(dtype=torch.int64)
        xy = proj[valid_ids, :2]
        prior_sample = sample_chw_at_pixels(prior_rgb, xy)
        mask_sample = sample_hw_at_pixels(prior_mask, xy).clamp(0.0, 1.0)
        good = mask_sample > 0.0
        if not torch.any(good):
            continue
        loss_acc += float(torch.mean(torch.abs(prior_sample[good] - dc_rgb[valid_ids[good]])).item()) * int(good.sum().item())
        sample_count += int(good.sum().item())
    return {"photo": float(loss_acc / max(sample_count, 1)), "samples": int(sample_count)}


def run_geometry_phase(
    cameras: Sequence[object],
    sr_cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    student: GaussianModel,
    bound: BoundedState,
    *,
    steps: int,
    lr: float,
    depth_min: float,
    prior_mask_threshold: float,
    lambda_uv: float,
    lambda_delta: float,
    charbonnier_eps: float,
    trusted_tangent_scale: float,
    loose_tangent_scale: float,
    trusted_normal_scale: float,
    loose_normal_scale: float,
    geometry_active_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float | int]:
    dc_rgb_frozen = features_dc_to_rgb(student._features_dc.detach()).detach()
    before = evaluate_geometry_residual(
        cameras,
        sr_cache,
        bound,
        dc_rgb_frozen,
        depth_min=float(depth_min),
        geometry_active_mask=geometry_active_mask,
    )
    geom_mask = bound.selected_mask & (bound.trust_class != TRUST_OUTLIER)
    if geometry_active_mask is not None:
        geom_mask = geom_mask & geometry_active_mask.to(device=geom_mask.device, dtype=torch.bool)
    geom_ids = torch.nonzero(geom_mask, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    if int(steps) <= 0 or int(geom_ids.numel()) <= 0 or len(cameras) <= 0:
        return {
            "before_photo": float(before["photo"]),
            "after_photo": float(before["photo"]),
            "sample_count": int(before["samples"]),
            "steps": 0,
        }

    trust = bound.trust_class[geom_ids]
    tau = bound.tau_surface[geom_ids]
    tangent_radius = torch.where(
        trust == TRUST_SURFACE,
        tau * float(trusted_tangent_scale),
        tau * float(loose_tangent_scale),
    )
    normal_radius = torch.where(
        trust == TRUST_SURFACE,
        tau * float(trusted_normal_scale),
        tau * float(loose_normal_scale),
    )
    u_base = bound.u_current[geom_ids].detach().clone()
    v_base = bound.v_current[geom_ids].detach().clone()
    d_base = bound.d_current[geom_ids].detach().clone()

    u_param = nn.Parameter(u_base.clone().requires_grad_(True))
    v_param = nn.Parameter(v_base.clone().requires_grad_(True))
    d_param = nn.Parameter(d_base.clone().requires_grad_(True))
    optimizer = torch.optim.Adam(
        [
            {"params": [u_param], "lr": float(lr), "name": "u"},
            {"params": [v_param], "lr": float(lr), "name": "v"},
            {"params": [d_param], "lr": float(lr), "name": "d"},
        ],
        eps=1e-15,
    )

    valid_view_indices = [
        index
        for index, cache_item in enumerate(sr_cache)
        if torch.is_tensor(cache_item.get("prior_mask")) and float(cache_item["prior_mask"].sum().item()) > 0.0
    ]
    if not valid_view_indices:
        return {
            "before_photo": float(before["photo"]),
            "after_photo": float(before["photo"]),
            "sample_count": int(before["samples"]),
            "steps": 0,
        }

    for _ in tqdm(range(int(steps)), desc="geometry phase"):
        optimizer.zero_grad(set_to_none=True)
        view_index = random.choice(valid_view_indices)
        camera = cameras[view_index]
        cache_item = sr_cache[view_index]
        prior_rgb = cache_item["prior_rgb"].to(device=u_param.device, dtype=torch.float32)
        prior_mask = cache_item["prior_mask"].to(device=u_param.device, dtype=torch.float32)
        xyz = bound.surface_points[geom_ids] + bound.tangent_u[geom_ids] * u_param[:, None] + bound.tangent_v[geom_ids] * v_param[:, None] + bound.normals[geom_ids] * d_param[:, None]
        proj, valid = project_points_camera_torch(camera, xyz, depth_min=float(depth_min))
        if not torch.any(valid):
            continue
        valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(1).to(dtype=torch.int64)
        xy = proj[valid_ids, :2]
        prior_sample = sample_chw_at_pixels(prior_rgb, xy)
        mask_sample = sample_hw_at_pixels(prior_mask, xy).clamp(0.0, 1.0)
        good = mask_sample > float(prior_mask_threshold)
        if not torch.any(good):
            continue

        local_ids = valid_ids[good]
        weight = mask_sample[good]
        dc_target = dc_rgb_frozen[geom_ids][local_ids]
        photo_rows = charbonnier_rows(prior_sample[good] - dc_target, charbonnier_eps)
        loss_photo = weighted_mean(photo_rows, weight)
        loss_uv = (((u_param - u_base) / torch.clamp(tangent_radius, min=1e-6)) ** 2 + ((v_param - v_base) / torch.clamp(tangent_radius, min=1e-6)) ** 2).mean()
        loss_delta = (((d_param - d_base) / torch.clamp(normal_radius, min=1e-6)) ** 2).mean()
        loss = loss_photo + float(lambda_uv) * loss_uv + float(lambda_delta) * loss_delta
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u_param.clamp_(min=u_base - tangent_radius, max=u_base + tangent_radius)
            v_param.clamp_(min=v_base - tangent_radius, max=v_base + tangent_radius)
            d_param.clamp_(min=d_base - normal_radius, max=d_base + normal_radius)

    with torch.no_grad():
        bound.u_current[geom_ids] = u_param.detach()
        bound.v_current[geom_ids] = v_param.detach()
        bound.d_current[geom_ids] = d_param.detach()
        student._xyz = bound.clone_current_xyz().detach()

    after = evaluate_geometry_residual(
        cameras,
        sr_cache,
        bound,
        dc_rgb_frozen,
        depth_min=float(depth_min),
        geometry_active_mask=geometry_active_mask,
    )
    return {
        "before_photo": float(before["photo"]),
        "after_photo": float(after["photo"]),
        "sample_count": int(after["samples"]),
        "steps": int(steps),
    }


def prune_absurd_gaussians(
    student: GaussianModel,
    bound: BoundedState,
    aggregated: AggregatedTargets,
    dc_anchor_ref: torch.Tensor,
    *,
    prune_bad_streak: int,
    prune_confidence_threshold: float,
    prune_disagreement_threshold: float,
    prune_color_error_threshold: float,
    prune_min_views: int,
    protect_confidence_threshold: float,
) -> Tuple[GaussianModel, BoundedState, torch.Tensor, Dict[str, object], Optional[torch.Tensor]]:
    active = bound.selected_mask
    if not torch.any(active):
        return student, bound, dc_anchor_ref, {"pruned": 0, "remaining": int(bound.uid.shape[0]), "selected_remaining": 0}, None

    current_rgb = features_dc_to_rgb(student._features_dc.detach())
    color_error = torch.mean(torch.abs(current_rgb - aggregated.target_rgb), dim=1)
    hard_outlier = (
        active
        & (bound.trust_class == TRUST_OUTLIER)
        & (aggregated.confidence < float(prune_confidence_threshold))
        & (aggregated.view_count < int(prune_min_views))
    )
    unsupported = (
        active
        & (aggregated.confidence < float(prune_confidence_threshold))
        & (aggregated.disagreement > float(prune_disagreement_threshold))
        & (color_error > float(prune_color_error_threshold))
    )
    bad_now = hard_outlier | unsupported
    bound.bad_streak[bad_now] += 1
    bound.bad_streak[~bad_now] = 0
    prune_mask = active & (bound.bad_streak >= int(prune_bad_streak))

    if torch.any(prune_mask):
        unique_groups = torch.unique(bound.lineage_group_id[prune_mask])
        for group_id in unique_groups.tolist():
            group = bound.lineage_group_id == int(group_id)
            group_prune = prune_mask & group
            if not torch.any(group_prune):
                continue
            group_conf = aggregated.confidence[group]
            group_views = aggregated.view_count[group]
            protect = (group_conf >= float(protect_confidence_threshold)) | (group_views >= int(prune_min_views) + 1)
            if torch.any(protect) and int(torch.count_nonzero(group_prune).item()) == int(torch.count_nonzero(group).item()):
                group_ids = torch.nonzero(group, as_tuple=False).squeeze(1).to(dtype=torch.int64)
                best_local = torch.argmax(group_conf)
                prune_mask[group_ids[int(best_local.item())]] = False

    prune_count = int(prune_mask.sum().item())
    if prune_count <= 0:
        return student, bound, dc_anchor_ref, {
            "pruned": 0,
            "remaining": int(bound.uid.shape[0]),
            "selected_remaining": int(bound.selected_mask.sum().item()),
        }, None

    valid_mask = ~prune_mask
    student._xyz = slice_parameter(student._xyz, valid_mask, requires_grad=False)
    student._features_dc = slice_parameter(student._features_dc, valid_mask, requires_grad=False)
    student._features_rest = slice_parameter(student._features_rest, valid_mask, requires_grad=False)
    student._opacity = slice_parameter(student._opacity, valid_mask, requires_grad=False)
    student._scaling = slice_parameter(student._scaling, valid_mask, requires_grad=False)
    student._rotation = slice_parameter(student._rotation, valid_mask, requires_grad=False)
    student._source_tag = student._source_tag[valid_mask]
    student._seed_id = student._seed_id[valid_mask]
    student._generation = student._generation[valid_mask]
    student._edge_touched = student._edge_touched[valid_mask]
    student._edge_touch_iter = student._edge_touch_iter[valid_mask]
    if isinstance(student.filter_3D, torch.Tensor) and int(student.filter_3D.shape[0]) == int(valid_mask.shape[0]):
        student.filter_3D = student.filter_3D[valid_mask]
    if isinstance(student.max_radii2D, torch.Tensor) and int(student.max_radii2D.shape[0]) == int(valid_mask.shape[0]):
        student.max_radii2D = student.max_radii2D[valid_mask]

    bound = bound.subset(valid_mask)
    dc_anchor_ref = dc_anchor_ref[valid_mask].clone()
    student._xyz = bound.clone_current_xyz().detach()
    return student, bound, dc_anchor_ref, {
        "pruned": int(prune_count),
        "remaining": int(bound.uid.shape[0]),
        "selected_remaining": int(bound.selected_mask.sum().item()),
    }, valid_mask.detach().clone()


def save_checkpoint(
    student: GaussianModel,
    bound: BoundedState,
    output_model_path: Path,
    *,
    iteration: int,
    initial_uid_count: int,
    summary_name: str,
    save_surface_map: bool = True,
) -> None:
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(iteration)}"
    mkdir_p(str(point_dir))
    student.save_ply(str(point_dir / "point_cloud.ply"))
    student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    if bool(save_surface_map):
        bound.save_payload(output_model_path / summary_name, initial_uid_count=int(initial_uid_count))


def main() -> None:
    parser = ArgumentParser(description="Bound all SOF GS to a surface frame and alternate prior-guided color/geometry updates.")
    parser.add_argument("-s", "--scene_root", type=str, required=True)
    parser.add_argument("--start_model_path", type=str, required=True)
    parser.add_argument("--start_iteration", type=int, default=-1)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--images_subdir", type=str, default="images_2")
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--sr_prior_root", type=str, required=True)
    parser.add_argument("--sr_anchor_root", type=str, required=True)
    parser.add_argument("--sr_prior_mask_dir", type=str, default="")
    parser.add_argument("--sr_prior_mask_suffix", type=str, default="")
    parser.add_argument("--sr_prior_consistency_threshold", type=float, default=0.08)
    parser.add_argument("--sr_prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--prior_prefilter_view_limit", type=int, default=0)
    parser.add_argument("--prior_prefilter_min_touch_views", type=int, default=1)
    parser.add_argument("--prior_prefilter_min_visible_views", type=int, default=1)
    parser.add_argument("--prior_prefilter_min_touch_ratio", type=float, default=0.0)
    parser.add_argument("--prior_prefilter_min_candidate_opacity", type=float, default=0.0)
    parser.add_argument("--prior_prefilter_radius_scale", type=float, default=0.5)
    parser.add_argument("--prior_prefilter_min_touch_radius_px", type=float, default=2.0)
    parser.add_argument("--prior_prefilter_max_touch_radius_px", type=float, default=16.0)
    parser.add_argument("--dump_masked_prior_inputs", action="store_true")
    parser.add_argument("--dump_masked_prior_max_views", type=int, default=16)
    parser.add_argument("--phase_mode", choices=["alternating", "appearance_only", "geometry_only"], default="alternating")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--total_steps", type=int, default=0)
    parser.add_argument("--appearance_steps", type=int, default=200)
    parser.add_argument("--geometry_steps", type=int, default=200)
    parser.add_argument("--save_every_cycles", type=int, default=1)
    parser.add_argument("--save_initial_surface_map", type=int, default=1)
    parser.add_argument("--save_cycle_surface_maps", type=int, default=1)
    parser.add_argument("--save_cycle_surface_targets", type=int, default=1)
    parser.add_argument("--save_cycle_memory", type=int, default=1)
    parser.add_argument("--save_final_surface_map", type=int, default=1)
    parser.add_argument("--save_final_memory", type=int, default=1)
    parser.add_argument("--output_iteration", type=int, default=-1)
    parser.add_argument("--face_k", type=int, default=8)
    parser.add_argument("--bind_chunk_size", type=int, default=16384)
    parser.add_argument("--tau_floor", type=float, default=0.002)
    parser.add_argument("--tau_edge_scale", type=float, default=0.4)
    parser.add_argument("--min_support_views", type=int, default=1)
    parser.add_argument("--min_sample_weight", type=float, default=0.05)
    parser.add_argument("--agreement_sigma", type=float, default=0.07)
    parser.add_argument("--base_sigma", type=float, default=0.08)
    parser.add_argument("--base_support_mode", choices=["multiply", "floor", "blend", "none"], default="multiply")
    parser.add_argument("--base_support_floor", type=float, default=0.0)
    parser.add_argument("--aggregation_residual_sample_clip", type=float, default=0.0)
    parser.add_argument("--aggregation_residual_band", choices=["raw", "lowmid"], default="raw")
    parser.add_argument("--aggregation_residual_lowpass_kernel", type=int, default=5)
    parser.add_argument("--aggregation_residual_min_l1", type=float, default=0.0)
    parser.add_argument("--aggregation_residual_max_l1", type=float, default=0.0)
    parser.add_argument("--disagreement_sigma", type=float, default=0.10)
    parser.add_argument("--aggregation_mode", choices=["mean", "trimmed_mean"], default="trimmed_mean")
    parser.add_argument("--robust_trim_sigma", type=float, default=0.12)
    parser.add_argument("--robust_trim_disagreement_scale", type=float, default=2.5)
    parser.add_argument("--enable_surface_memory", type=int, default=1)
    parser.add_argument("--memory_beta", type=float, default=0.20)
    parser.add_argument("--memory_min_confidence", type=float, default=0.05)
    parser.add_argument("--memory_max_disagreement", type=float, default=0.16)
    parser.add_argument("--memory_stable_updates", type=int, default=2)
    parser.add_argument("--memory_stable_min_confidence", type=float, default=0.08)
    parser.add_argument("--memory_stable_max_disagreement", type=float, default=0.12)
    parser.add_argument(
        "--memory_eligibility",
        choices=["selected", "trust_surface", "non_outlier", "trust_surface_or_visible_loose", "trust_loose_only", "outlier_only"],
        default="selected",
    )
    parser.add_argument("--memory_loose_min_visible_views", type=int, default=2)
    parser.add_argument("--memory_loose_min_touch_views", type=int, default=1)
    parser.add_argument("--memory_loose_min_touch_ratio", type=float, default=0.0)
    parser.add_argument("--appearance_lr", type=float, default=5e-4)
    parser.add_argument("--lambda_dc_anchor", type=float, default=0.05)
    parser.add_argument("--lambda_base_guard", type=float, default=0.10)
    parser.add_argument("--appearance_target_mode", choices=["absolute", "residual_clipped"], default="absolute")
    parser.add_argument("--appearance_residual_clip", type=float, default=0.0)
    parser.add_argument("--appearance_residual_scale", type=float, default=1.0)
    parser.add_argument("--geometry_lr", type=float, default=5e-4)
    parser.add_argument("--geometry_prior_mask_threshold", type=float, default=0.05)
    parser.add_argument("--lambda_uv", type=float, default=0.05)
    parser.add_argument("--lambda_delta", type=float, default=0.05)
    parser.add_argument("--trusted_tangent_scale", type=float, default=2.5)
    parser.add_argument("--loose_tangent_scale", type=float, default=1.5)
    parser.add_argument("--trusted_normal_scale", type=float, default=1.5)
    parser.add_argument("--loose_normal_scale", type=float, default=0.75)
    parser.add_argument("--prune_after_cycle", action="store_true")
    parser.add_argument("--prune_bad_streak", type=int, default=2)
    parser.add_argument("--prune_confidence_threshold", type=float, default=0.02)
    parser.add_argument("--prune_disagreement_threshold", type=float, default=0.18)
    parser.add_argument("--prune_color_error_threshold", type=float, default=0.12)
    parser.add_argument("--prune_min_views", type=int, default=1)
    parser.add_argument("--protect_confidence_threshold", type=float, default=0.08)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(bool(args.quiet))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    scene_root = Path(args.scene_root).expanduser().resolve()
    start_model_path = Path(args.start_model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    sr_prior_root = Path(args.sr_prior_root).expanduser().resolve()
    sr_anchor_root = Path(args.sr_anchor_root).expanduser().resolve()
    sr_prior_mask_dir = Path(args.sr_prior_mask_dir).expanduser().resolve() if str(args.sr_prior_mask_dir).strip() else None

    start_iteration = resolve_iteration(start_model_path, int(args.start_iteration))
    output_model_path.mkdir(parents=True, exist_ok=True)
    copy_render_config(start_model_path, output_model_path)

    student = load_model_ply(start_model_path, start_iteration, sh_degree=3)
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    cameras = load_train_cameras_only(scene_root, start_model_path, args.images_subdir)
    cameras = select_uniform(cameras, int(args.max_views))
    sr_prior_index = index_image_dir(sr_prior_root)
    sr_anchor_index = index_image_dir(sr_anchor_root)
    sr_cameras, sr_cache = build_prepared_sr_cache(
        cameras,
        student,
        background,
        sr_prior_index=sr_prior_index,
        sr_anchor_index=sr_anchor_index,
        sr_prior_mask_dir=sr_prior_mask_dir,
        sr_prior_mask_suffix=str(args.sr_prior_mask_suffix),
        prior_consistency_threshold=float(args.sr_prior_consistency_threshold),
        prior_mask_floor=float(args.sr_prior_mask_floor),
    )
    if not sr_cache:
        raise RuntimeError("No prepared SR prior views matched the current scene cameras.")
    sr_cameras, sr_cache = _select_pair_uniform(sr_cameras, sr_cache, int(args.max_views))
    masked_prior_dump: Dict[str, object] | None = None
    if bool(args.dump_masked_prior_inputs):
        masked_prior_dump = dump_masked_prior_inputs(
            sr_cameras,
            sr_cache,
            output_model_path / "masked_prior_inputs",
            max_views=int(args.dump_masked_prior_max_views),
        )

    prefilter = build_sr_touch_prefilter(
        sr_cameras,
        sr_cache,
        student,
        background,
        base_mask=None,
        view_limit=int(args.prior_prefilter_view_limit),
        min_touch_views=int(args.prior_prefilter_min_touch_views),
        min_visible_views=int(args.prior_prefilter_min_visible_views),
        min_touch_ratio=float(args.prior_prefilter_min_touch_ratio),
        min_candidate_opacity=float(args.prior_prefilter_min_candidate_opacity),
        radius_scale=float(args.prior_prefilter_radius_scale),
        min_touch_radius_px=float(args.prior_prefilter_min_touch_radius_px),
        max_touch_radius_px=float(args.prior_prefilter_max_touch_radius_px),
    )
    torch.save(
        {
            "selected_mask": prefilter["selected_mask"].detach().cpu(),
            "selected_ids": prefilter["selected_ids"].detach().cpu(),
            "visible_view_count": prefilter["visible_view_count"].detach().cpu(),
            "touch_view_count": prefilter["touch_view_count"].detach().cpu(),
            "touch_ratio": prefilter["touch_ratio"].detach().cpu(),
            "summary": prefilter["summary"],
        },
        output_model_path / "prior_prefilter_selected_mask.pt",
    )

    bound = build_bounded_state(
        student,
        str(mesh_path),
        face_k=int(args.face_k),
        chunk_size=int(args.bind_chunk_size),
        tau_floor=float(args.tau_floor),
        tau_edge_scale=float(args.tau_edge_scale),
    )
    bound.selected_mask = prefilter["selected_mask"].to(device=bound.selected_mask.device, dtype=torch.bool).reshape(-1)
    memory_eligible_mask, memory_eligibility_summary = build_memory_eligible_mask(
        bound,
        prefilter,
        mode=str(args.memory_eligibility),
        loose_min_visible_views=int(args.memory_loose_min_visible_views),
        loose_min_touch_views=int(args.memory_loose_min_touch_views),
        loose_min_touch_ratio=float(args.memory_loose_min_touch_ratio),
    )
    student._xyz = bound.clone_current_xyz().detach()
    dc_anchor_ref = student._features_dc.detach().clone()
    initial_uid_count = int(bound.uid.shape[0])
    surface_memory = SurfacePriorMemory.create(int(bound.uid.shape[0]), device=bound.uid.device)
    if bool(int(args.save_initial_surface_map)):
        bound.save_payload(output_model_path / "bounded_surface_map_cycle00.pt", initial_uid_count=initial_uid_count)

    appearance_steps = int(args.appearance_steps)
    geometry_steps = int(args.geometry_steps)
    if str(args.phase_mode) == "appearance_only":
        geometry_steps = 0
    elif str(args.phase_mode) == "geometry_only":
        appearance_steps = 0
    cycle_step_budget = max(appearance_steps + geometry_steps, 1)
    cycles = int(args.cycles)
    if int(args.total_steps) > 0:
        cycles = max(1, math.ceil(int(args.total_steps) / cycle_step_budget))

    executed_total_steps = 0
    cycle_summaries: List[Dict[str, object]] = []
    prune_history: List[Dict[str, object]] = []

    for cycle in range(1, cycles + 1):
        aggregated_before = aggregate_surface_targets(
            sr_cameras,
            sr_cache,
            bound,
            selected_mask_override=memory_eligible_mask,
            depth_min=float(args.depth_min),
            min_support_views=int(args.min_support_views),
            min_sample_weight=float(args.min_sample_weight),
            agreement_sigma=float(args.agreement_sigma),
            base_sigma=float(args.base_sigma),
            base_support_mode=str(args.base_support_mode),
            base_support_floor=float(args.base_support_floor),
            residual_sample_clip=float(args.aggregation_residual_sample_clip),
            residual_band=str(args.aggregation_residual_band),
            residual_lowpass_kernel=int(args.aggregation_residual_lowpass_kernel),
            residual_min_l1=float(args.aggregation_residual_min_l1),
            residual_max_l1=float(args.aggregation_residual_max_l1),
            disagreement_sigma=float(args.disagreement_sigma),
            aggregation_mode=str(args.aggregation_mode),
            robust_trim_sigma=float(args.robust_trim_sigma),
            robust_trim_disagreement_scale=float(args.robust_trim_disagreement_scale),
        )
        if bool(int(args.save_cycle_surface_targets)):
            aggregated_before.save(output_model_path / f"surface_targets_cycle{cycle:02d}_before.npz")

        memory_before_summary: Dict[str, object] | None = None
        memory_after_summary: Dict[str, object] | None = None
        memory_path: Path | None = None
        if bool(int(args.enable_surface_memory)):
            memory_before_summary = surface_memory.update(
                aggregated_before,
                memory_eligible_mask,
                beta_max=float(args.memory_beta),
                min_confidence=float(args.memory_min_confidence),
                max_disagreement=float(args.memory_max_disagreement),
                min_views=int(args.min_support_views),
                stable_updates=int(args.memory_stable_updates),
                stable_min_confidence=float(args.memory_stable_min_confidence),
                stable_max_disagreement=float(args.memory_stable_max_disagreement),
            )
            appearance_targets = surface_memory.to_aggregated(
                memory_eligible_mask,
                [MEM_COLOR_READY, MEM_COLOR_STABLE],
                summary_name=f"cycle{cycle:02d}_appearance_memory",
            )
            geometry_targets = surface_memory.to_aggregated(
                memory_eligible_mask,
                [MEM_COLOR_STABLE],
                summary_name=f"cycle{cycle:02d}_geometry_memory",
            )
            geometry_active_mask = geometry_targets.valid_mask
        else:
            appearance_targets = aggregated_before
            geometry_targets = aggregated_before
            geometry_active_mask = None

        appearance_summary = run_appearance_phase(
            student,
            appearance_targets,
            bound,
            steps=appearance_steps,
            lr=float(args.appearance_lr),
            dc_anchor_ref=dc_anchor_ref,
            lambda_anchor=float(args.lambda_dc_anchor),
            lambda_base_guard=float(args.lambda_base_guard),
            target_mode=str(args.appearance_target_mode),
            residual_clip=float(args.appearance_residual_clip),
            residual_scale=float(args.appearance_residual_scale),
            charbonnier_eps=float(args.charbonnier_eps),
        )
        executed_total_steps += int(appearance_summary["steps"])

        geometry_summary = run_geometry_phase(
            sr_cameras,
            sr_cache,
            student,
            bound,
            steps=geometry_steps,
            lr=float(args.geometry_lr),
            depth_min=float(args.depth_min),
            prior_mask_threshold=float(args.geometry_prior_mask_threshold),
            lambda_uv=float(args.lambda_uv),
            lambda_delta=float(args.lambda_delta),
            charbonnier_eps=float(args.charbonnier_eps),
            trusted_tangent_scale=float(args.trusted_tangent_scale),
            loose_tangent_scale=float(args.loose_tangent_scale),
            trusted_normal_scale=float(args.trusted_normal_scale),
            loose_normal_scale=float(args.loose_normal_scale),
            geometry_active_mask=geometry_active_mask,
        )
        executed_total_steps += int(geometry_summary["steps"])

        aggregated_after = aggregate_surface_targets(
            sr_cameras,
            sr_cache,
            bound,
            selected_mask_override=memory_eligible_mask,
            depth_min=float(args.depth_min),
            min_support_views=int(args.min_support_views),
            min_sample_weight=float(args.min_sample_weight),
            agreement_sigma=float(args.agreement_sigma),
            base_sigma=float(args.base_sigma),
            base_support_mode=str(args.base_support_mode),
            base_support_floor=float(args.base_support_floor),
            residual_sample_clip=float(args.aggregation_residual_sample_clip),
            residual_band=str(args.aggregation_residual_band),
            residual_lowpass_kernel=int(args.aggregation_residual_lowpass_kernel),
            residual_min_l1=float(args.aggregation_residual_min_l1),
            residual_max_l1=float(args.aggregation_residual_max_l1),
            disagreement_sigma=float(args.disagreement_sigma),
            aggregation_mode=str(args.aggregation_mode),
            robust_trim_sigma=float(args.robust_trim_sigma),
            robust_trim_disagreement_scale=float(args.robust_trim_disagreement_scale),
        )
        if bool(int(args.save_cycle_surface_targets)):
            aggregated_after.save(output_model_path / f"surface_targets_cycle{cycle:02d}_after.npz")

        if bool(int(args.enable_surface_memory)):
            memory_after_summary = surface_memory.update(
                aggregated_after,
                memory_eligible_mask,
                beta_max=float(args.memory_beta),
                min_confidence=float(args.memory_min_confidence),
                max_disagreement=float(args.memory_max_disagreement),
                min_views=int(args.min_support_views),
                stable_updates=int(args.memory_stable_updates),
                stable_min_confidence=float(args.memory_stable_min_confidence),
                stable_max_disagreement=float(args.memory_stable_max_disagreement),
            )
            memory_path = output_model_path / f"surface_prior_memory_cycle{cycle:02d}.pt"
            if bool(int(args.save_cycle_memory)):
                surface_memory.save(memory_path)
            else:
                memory_path = None

        prune_summary = {"pruned": 0, "remaining": int(bound.uid.shape[0]), "selected_remaining": int(bound.selected_mask.sum().item())}
        if bool(args.prune_after_cycle):
            student, bound, dc_anchor_ref, prune_summary, kept_mask = prune_absurd_gaussians(
                student,
                bound,
                aggregated_after,
                dc_anchor_ref,
                prune_bad_streak=int(args.prune_bad_streak),
                prune_confidence_threshold=float(args.prune_confidence_threshold),
                prune_disagreement_threshold=float(args.prune_disagreement_threshold),
                prune_color_error_threshold=float(args.prune_color_error_threshold),
                prune_min_views=int(args.prune_min_views),
                protect_confidence_threshold=float(args.protect_confidence_threshold),
            )
            if kept_mask is not None:
                surface_memory = surface_memory.subset(kept_mask)
                memory_eligible_mask = memory_eligible_mask[kept_mask].clone()
                if bool(int(args.enable_surface_memory)):
                    memory_path = output_model_path / f"surface_prior_memory_cycle{cycle:02d}_post_prune.pt"
                    if bool(int(args.save_cycle_memory)):
                        surface_memory.save(memory_path)
                    else:
                        memory_path = None
            prune_history.append({"cycle": int(cycle), **prune_summary})

        if bool(int(args.save_cycle_surface_maps)):
            bound.save_payload(output_model_path / f"bounded_surface_map_cycle{cycle:02d}.pt", initial_uid_count=initial_uid_count)
        if int(args.save_every_cycles) > 0 and cycle % int(args.save_every_cycles) == 0:
            checkpoint_iteration = start_iteration + executed_total_steps
            save_checkpoint(
                student,
                bound,
                output_model_path,
                iteration=checkpoint_iteration,
                initial_uid_count=initial_uid_count,
                summary_name=f"bounded_surface_map_cycle{cycle:02d}_latest.pt",
                save_surface_map=bool(int(args.save_cycle_surface_maps)),
            )

        cycle_summaries.append(
            {
                "cycle": int(cycle),
                "aggregation_before": aggregated_before.summary,
                "memory_before": memory_before_summary,
                "appearance_targets": appearance_targets.summary,
                "appearance": appearance_summary,
                "geometry_targets": geometry_targets.summary,
                "geometry": geometry_summary,
                "aggregation_after": aggregated_after.summary,
                "memory_after": memory_after_summary,
                "memory_path": None if memory_path is None else str(memory_path),
                "prune": prune_summary,
            }
        )

    final_iteration = int(args.output_iteration) if int(args.output_iteration) >= 0 else start_iteration + executed_total_steps
    student._xyz = bound.clone_current_xyz().detach()
    save_checkpoint(
        student,
        bound,
        output_model_path,
        iteration=final_iteration,
        initial_uid_count=initial_uid_count,
        summary_name="bounded_surface_map_latest.pt",
        save_surface_map=bool(int(args.save_final_surface_map)),
    )
    surface_memory_latest_path: Path | None = None
    if bool(int(args.save_final_memory)):
        surface_memory_latest_path = output_model_path / "surface_prior_memory_latest.pt"
        surface_memory.save(surface_memory_latest_path)
    with open(output_model_path / "prune_history.jsonl", "w", encoding="utf-8") as handle:
        for item in prune_history:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "version": "bounded_surface_alternating_v0",
        "scene_root": str(scene_root),
        "start_model_path": str(start_model_path),
        "start_iteration": int(start_iteration),
        "mesh_path": str(mesh_path),
        "output_model_path": str(output_model_path),
        "sr_prior_root": str(sr_prior_root),
        "sr_anchor_root": str(sr_anchor_root),
        "sr_prior_mask_dir": str(sr_prior_mask_dir) if sr_prior_mask_dir is not None else None,
        "masked_prior_dump": masked_prior_dump,
        "prefilter": prefilter["summary"],
        "sr_prior_cache": summarize_sr_cache(sr_cache),
        "binding": {
            "count": int(bound.uid.shape[0]),
            "selected_count": int(bound.selected_mask.sum().item()),
            "memory_eligible_count": int(memory_eligible_mask.sum().item()),
            "surface_distance": tensor_stats(torch.abs(bound.d0)),
            "tau_surface": tensor_stats(bound.tau_surface),
            "d_norm": tensor_stats(bound.d_norm),
            "trust_surface": int(torch.count_nonzero(bound.trust_class == TRUST_SURFACE).item()),
            "trust_loose": int(torch.count_nonzero(bound.trust_class == TRUST_LOOSE).item()),
            "trust_outlier": int(torch.count_nonzero(bound.trust_class == TRUST_OUTLIER).item()),
        },
        "memory_eligibility": memory_eligibility_summary,
        "params": vars(args),
        "planned_cycles": int(cycles),
        "planned_total_steps": int(args.total_steps),
        "executed_total_steps": int(executed_total_steps),
        "cycles": cycle_summaries,
        "artifacts": {
            "prefilter_payload": str(output_model_path / "prior_prefilter_selected_mask.pt"),
            "masked_prior_inputs_dir": None if masked_prior_dump is None else masked_prior_dump["dir"],
            "masked_prior_inputs_manifest": None if masked_prior_dump is None else masked_prior_dump["manifest"],
            "bounded_surface_map_latest": str(output_model_path / "bounded_surface_map_latest.pt") if bool(int(args.save_final_surface_map)) else None,
            "surface_prior_memory_latest": None if surface_memory_latest_path is None else str(surface_memory_latest_path),
            "final_point_cloud_dir": str(output_model_path / "point_cloud" / f"iteration_{int(final_iteration)}"),
            "prune_history": str(output_model_path / "prune_history.jsonl"),
        },
    }
    summary_path = output_model_path / "bounded_surface_alternating_v0_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
