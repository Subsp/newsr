from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
import trimesh

from gaussian_renderer import render_simple
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import load_cameras_for_split, load_model_ply, resolve_iteration, select_uniform
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.prior_fusion import _quaternion_from_rotation_matrix
from utils.sh_utils import RGB2SH


def is_camera_source_root(path: Path) -> bool:
    return (path / "sparse").exists() or (path / "transforms_train.json").is_file()


def resolve_scene_root(args) -> Path:
    if getattr(args, "scene_root", None):
        return Path(args.scene_root).expanduser().resolve()
    if getattr(args, "prior_dir", None):
        prior_root = Path(args.prior_dir).expanduser().resolve()
        base_root = prior_root.parent
        if is_camera_source_root(base_root):
            return base_root
        child_candidates = sorted(path for path in base_root.iterdir() if path.is_dir())
        matched = [path for path in child_candidates if is_camera_source_root(path)]
        if len(matched) == 1:
            return matched[0]
        return base_root
    raise ValueError("scene_root could not be resolved because neither --scene_root nor --prior_dir was provided.")


def resolve_camera_model_path(args, sof_model_path: Path) -> Path:
    value = getattr(args, "camera_model_path", None)
    if value:
        return Path(value).expanduser().resolve()
    return sof_model_path


def stats_from_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def lookup_indexed_path(index: Dict[str, Path], image_name: str) -> Optional[Path]:
    candidates = [
        str(image_name),
        normalize_image_name(str(image_name)),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    for candidate in candidates:
        value = index.get(candidate)
        if value is not None:
            return Path(value)
    lower_index = {str(key).lower(): Path(value) for key, value in index.items()}
    for candidate in candidates:
        value = lower_index.get(str(candidate).lower())
        if value is not None:
            return value
    return None


def to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    Image.fromarray(to_uint8_rgb(image_chw), mode="RGB").save(path)


def load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_rgb_uint8(image: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    if pil.size != (target_w, target_h):
        resampling = getattr(Image, "Resampling", Image)
        pil = pil.resize((target_w, target_h), resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def scalar_to_rgb(values: np.ndarray, invert: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = np.clip(1.0 - x if invert else x, 0.0, 1.0)
    r = np.clip(1.8 * x, 0.0, 1.0)
    g = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32, copy=False)


def detail_residual_to_rgb(detail_rgb: np.ndarray, clamp_value: float) -> np.ndarray:
    limit = max(float(clamp_value), 1.0e-6)
    x = np.clip(np.asarray(detail_rgb, dtype=np.float32) / limit, -1.0, 1.0)
    return np.clip(0.5 + 0.5 * x, 0.0, 1.0).astype(np.float32, copy=False)


def scalar_image_to_rgb(image_hw: np.ndarray, invert: bool = False) -> np.ndarray:
    value = np.asarray(image_hw, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        norm = np.zeros_like(value, dtype=np.float32)
    else:
        vmax = max(float(np.percentile(finite, 99.0)), 1.0e-6)
        norm = np.clip(value / vmax, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    r = np.clip(1.8 * norm, 0.0, 1.0)
    g = np.clip(1.8 * norm - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * norm, 0.0, 1.0)
    return np.clip(np.stack([r, g, b], axis=-1) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def export_point_cloud(path: Path, points: np.ndarray, colors_rgb: np.ndarray, max_points: int, seed: int) -> int:
    points = np.asarray(points, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.float32)
    if points.shape[0] != colors_rgb.shape[0]:
        raise ValueError(f"Point/color count mismatch: {points.shape[0]} vs {colors_rgb.shape[0]}")
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        ids = np.sort(rng.choice(points.shape[0], size=int(max_points), replace=False))
        points = points[ids]
        colors_rgb = colors_rgb[ids]
    colors_u8 = np.clip(colors_rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    trimesh.points.PointCloud(points, colors=colors_u8).export(path)
    return int(points.shape[0])


def _clone_subset_gaussians(base: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    subset = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[mask].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[mask].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[mask].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[mask].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[mask].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[mask].clone().requires_grad_(False))
    if (
        isinstance(base.filter_3D, torch.Tensor)
        and base.filter_3D.ndim > 0
        and base.filter_3D.shape[0] == base.get_xyz.shape[0]
    ):
        subset.filter_3D = base.filter_3D.detach()[mask].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.init_tracking_state(count)
    if base._source_tag.shape[0] == base.get_xyz.shape[0]:
        subset._source_tag = base._source_tag.detach()[mask].clone()
    if base._seed_id.shape[0] == base.get_xyz.shape[0]:
        subset._seed_id = base._seed_id.detach()[mask].clone()
    if base._generation.shape[0] == base.get_xyz.shape[0]:
        subset._generation = base._generation.detach()[mask].clone()
    if base._edge_touched.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touched = base._edge_touched.detach()[mask].clone()
    if base._edge_touch_iter.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touch_iter = base._edge_touch_iter.detach()[mask].clone()
    return subset


def apply_preview_opacity_in_place(
    gaussians: GaussianModel,
    opacity_gate: np.ndarray,
    *,
    min_opacity: float,
    max_opacity: float,
    gate_power: float,
) -> None:
    gate = torch.from_numpy(np.asarray(opacity_gate, dtype=np.float32)).to(device=gaussians.get_xyz.device)
    gate = torch.clamp(gate, 0.0, 1.0)
    if float(gate_power) != 1.0:
        gate = gate.pow(float(gate_power))
    base_opacity = gaussians.get_opacity.detach().reshape(-1)
    target = torch.clamp(base_opacity * gate, min=float(min_opacity), max=float(max_opacity))
    gaussians._opacity = nn.Parameter(gaussians.inverse_opacity_activation(target[:, None]).detach().requires_grad_(False))


def assign_rgb_to_gaussians_in_place(gaussians: GaussianModel, global_ids: np.ndarray, rgb: np.ndarray, zero_rest: bool = False) -> None:
    ids_t = torch.from_numpy(np.asarray(global_ids, dtype=np.int64)).to(device=gaussians.get_xyz.device, dtype=torch.long)
    rgb_t = torch.from_numpy(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)).to(device=gaussians.get_xyz.device, dtype=torch.float32)
    with torch.no_grad():
        if gaussians._features_dc.ndim == 3:
            if gaussians._features_dc.shape[1] == 1 and gaussians._features_dc.shape[2] == 3:
                gaussians._features_dc[ids_t, 0, :] = RGB2SH(rgb_t)
            elif gaussians._features_dc.shape[1] == 3 and gaussians._features_dc.shape[2] == 1:
                gaussians._features_dc[ids_t, :, 0] = RGB2SH(rgb_t)
            else:
                raise RuntimeError(f"Unsupported _features_dc shape: {tuple(gaussians._features_dc.shape)}")
        elif gaussians._features_dc.ndim == 2 and gaussians._features_dc.shape[1] == 3:
            if bool(gaussians.use_SBs):
                gaussians._features_dc[ids_t, :] = rgb_t
            else:
                gaussians._features_dc[ids_t, :] = RGB2SH(rgb_t)
        else:
            raise RuntimeError(f"Unsupported _features_dc shape: {tuple(gaussians._features_dc.shape)}")
        if bool(zero_rest):
            gaussians._features_rest[ids_t] = 0.0


def configure_local_child_overlay_model_in_place(
    child_model: GaussianModel,
    *,
    centers: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    normals: np.ndarray,
    scale_u: np.ndarray,
    scale_v: np.ndarray,
    scale_n: np.ndarray,
    rgb: np.ndarray,
    opacity_gate: np.ndarray,
    scale_xy: float,
    scale_n_mul: float,
    opacity_scale: float,
    opacity_min: float,
    opacity_max: float,
) -> None:
    device = child_model.get_xyz.device
    centers_t = torch.from_numpy(np.asarray(centers, dtype=np.float32)).to(device=device, dtype=torch.float32)
    basis_t = torch.from_numpy(
        np.stack(
            [
                np.asarray(tangent_u, dtype=np.float32),
                np.asarray(tangent_v, dtype=np.float32),
                np.asarray(normals, dtype=np.float32),
            ],
            axis=2,
        )
    ).to(device=device, dtype=torch.float32)
    rotation_t = _quaternion_from_rotation_matrix(basis_t)
    scales_t = torch.from_numpy(
        np.stack(
            [
                np.maximum(np.asarray(scale_u, dtype=np.float32) * float(scale_xy), 1e-6),
                np.maximum(np.asarray(scale_v, dtype=np.float32) * float(scale_xy), 1e-6),
                np.maximum(np.asarray(scale_n, dtype=np.float32) * float(scale_n_mul), 1e-6),
            ],
            axis=1,
        )
    ).to(device=device, dtype=torch.float32)
    rgb_t = torch.from_numpy(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)).to(device=device, dtype=torch.float32)
    gate_t = torch.from_numpy(np.clip(np.asarray(opacity_gate, dtype=np.float32), 0.0, 1.0)).to(device=device, dtype=torch.float32)
    opacity_prob = torch.clamp(
        child_model.get_opacity.detach().reshape(-1) * gate_t * float(opacity_scale),
        min=float(opacity_min),
        max=float(opacity_max),
    )

    with torch.no_grad():
        child_model._xyz = nn.Parameter(centers_t.detach().clone().requires_grad_(False))
        child_model._rotation = nn.Parameter(rotation_t.detach().clone().requires_grad_(False))
        child_model._scaling = nn.Parameter(torch.log(scales_t).detach().clone().requires_grad_(False))
        child_model._opacity = nn.Parameter(
            child_model.inverse_opacity_activation(opacity_prob[:, None]).detach().clone().requires_grad_(False)
        )
        if child_model._features_dc.ndim == 3:
            if child_model._features_dc.shape[1] == 1 and child_model._features_dc.shape[2] == 3:
                child_model._features_dc = nn.Parameter(RGB2SH(rgb_t)[:, None, :].detach().clone().requires_grad_(False))
            elif child_model._features_dc.shape[1] == 3 and child_model._features_dc.shape[2] == 1:
                child_model._features_dc = nn.Parameter(RGB2SH(rgb_t)[:, :, None].detach().clone().requires_grad_(False))
            else:
                raise RuntimeError(f"Unsupported _features_dc shape: {tuple(child_model._features_dc.shape)}")
        elif child_model._features_dc.ndim == 2 and child_model._features_dc.shape[1] == 3:
            if bool(child_model.use_SBs):
                child_model._features_dc = nn.Parameter(rgb_t.detach().clone().requires_grad_(False))
            else:
                child_model._features_dc = nn.Parameter(RGB2SH(rgb_t).detach().clone().requires_grad_(False))
        else:
            raise RuntimeError(f"Unsupported _features_dc shape: {tuple(child_model._features_dc.shape)}")
        child_model._features_rest = nn.Parameter(torch.zeros_like(child_model._features_rest).detach().clone().requires_grad_(False))


def merge_gaussian_models(surface: GaussianModel, detail: GaussianModel) -> GaussianModel:
    merged = GaussianModel(max(int(surface.max_sh_degree), int(detail.max_sh_degree)), use_SBs=bool(surface.use_SBs))
    merged.active_sh_degree = max(int(surface.active_sh_degree), int(detail.active_sh_degree))
    merged.spatial_lr_scale = float(surface.spatial_lr_scale)
    merged._xyz = nn.Parameter(torch.cat((surface._xyz.detach(), detail._xyz.detach()), dim=0).requires_grad_(False))
    merged._features_dc = nn.Parameter(
        torch.cat((surface._features_dc.detach(), detail._features_dc.detach()), dim=0).requires_grad_(False)
    )
    merged._features_rest = nn.Parameter(
        torch.cat((surface._features_rest.detach(), detail._features_rest.detach()), dim=0).requires_grad_(False)
    )
    merged._opacity = nn.Parameter(torch.cat((surface._opacity.detach(), detail._opacity.detach()), dim=0).requires_grad_(False))
    merged._scaling = nn.Parameter(torch.cat((surface._scaling.detach(), detail._scaling.detach()), dim=0).requires_grad_(False))
    merged._rotation = nn.Parameter(torch.cat((surface._rotation.detach(), detail._rotation.detach()), dim=0).requires_grad_(False))
    merged.filter_3D = torch.cat((surface.filter_3D.detach(), detail.filter_3D.detach()), dim=0)
    merged.max_radii2D = torch.zeros((merged._xyz.shape[0],), dtype=torch.float32, device=surface.get_xyz.device)
    merged.xyz_gradient_accum = torch.zeros((merged._xyz.shape[0], 1), dtype=torch.float32, device=surface.get_xyz.device)
    merged.xyz_gradient_accum_abs = torch.zeros((merged._xyz.shape[0], 1), dtype=torch.float32, device=surface.get_xyz.device)
    merged.xyz_gradient_accum_abs_max = torch.zeros((merged._xyz.shape[0], 1), dtype=torch.float32, device=surface.get_xyz.device)
    merged.denom = torch.zeros((merged._xyz.shape[0], 1), dtype=torch.float32, device=surface.get_xyz.device)
    merged.init_tracking_state(int(merged._xyz.shape[0]))
    merged._source_tag = torch.cat((surface._source_tag.detach(), detail._source_tag.detach()), dim=0)
    merged._seed_id = torch.cat((surface._seed_id.detach(), detail._seed_id.detach()), dim=0)
    merged._generation = torch.cat((surface._generation.detach(), detail._generation.detach()), dim=0)
    merged._edge_touched = torch.cat((surface._edge_touched.detach(), detail._edge_touched.detach()), dim=0)
    merged._edge_touch_iter = torch.cat((surface._edge_touch_iter.detach(), detail._edge_touch_iter.detach()), dim=0)
    return merged


def build_overlay_gaussian_model(
    base: GaussianModel,
    *,
    xyz: np.ndarray,
    rgb: np.ndarray,
    rotation_matrix: np.ndarray,
    scales: np.ndarray,
    opacity_prob: np.ndarray,
    source_tag: Optional[np.ndarray] = None,
    seed_id: Optional[np.ndarray] = None,
    generation: Optional[np.ndarray] = None,
) -> GaussianModel:
    device = base.get_xyz.device
    xyz_t = torch.from_numpy(np.asarray(xyz, dtype=np.float32)).to(device=device, dtype=torch.float32)
    rgb_t = torch.from_numpy(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)).to(device=device, dtype=torch.float32)
    rotation_t = _quaternion_from_rotation_matrix(
        torch.from_numpy(np.asarray(rotation_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    )
    scales_t = torch.from_numpy(np.maximum(np.asarray(scales, dtype=np.float32), 1e-6)).to(device=device, dtype=torch.float32)
    opacity_t = torch.from_numpy(
        np.clip(np.asarray(opacity_prob, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    ).to(device=device, dtype=torch.float32)

    model = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    model.active_sh_degree = int(base.active_sh_degree)
    model.spatial_lr_scale = float(base.spatial_lr_scale)
    model._xyz = nn.Parameter(xyz_t.detach().clone().requires_grad_(False))
    if base._features_dc.ndim == 3:
        if base._features_dc.shape[1] == 1 and base._features_dc.shape[2] == 3:
            model._features_dc = nn.Parameter(RGB2SH(rgb_t)[:, None, :].detach().clone().requires_grad_(False))
            model._features_rest = nn.Parameter(
                torch.zeros((xyz_t.shape[0], *base._features_rest.shape[1:]), dtype=base._features_rest.dtype, device=device).requires_grad_(False)
            )
        elif base._features_dc.shape[1] == 3 and base._features_dc.shape[2] == 1:
            model._features_dc = nn.Parameter(RGB2SH(rgb_t)[:, :, None].detach().clone().requires_grad_(False))
            model._features_rest = nn.Parameter(
                torch.zeros((xyz_t.shape[0], *base._features_rest.shape[1:]), dtype=base._features_rest.dtype, device=device).requires_grad_(False)
            )
        else:
            raise RuntimeError(f"Unsupported _features_dc shape: {tuple(base._features_dc.shape)}")
    elif base._features_dc.ndim == 2 and base._features_dc.shape[1] == 3:
        if bool(base.use_SBs):
            model._features_dc = nn.Parameter(rgb_t.detach().clone().requires_grad_(False))
        else:
            model._features_dc = nn.Parameter(RGB2SH(rgb_t).detach().clone().requires_grad_(False))
        model._features_rest = nn.Parameter(
            torch.zeros((xyz_t.shape[0], *base._features_rest.shape[1:]), dtype=base._features_rest.dtype, device=device).requires_grad_(False)
        )
    else:
        raise RuntimeError(f"Unsupported _features_dc shape: {tuple(base._features_dc.shape)}")

    model._opacity = nn.Parameter(base.inverse_opacity_activation(opacity_t[:, None]).detach().clone().requires_grad_(False))
    model._scaling = nn.Parameter(torch.log(scales_t).detach().clone().requires_grad_(False))
    model._rotation = nn.Parameter(rotation_t.detach().clone().requires_grad_(False))
    model.filter_3D = torch.zeros((xyz_t.shape[0], 1), dtype=torch.float32, device=device)
    model.max_radii2D = torch.zeros((xyz_t.shape[0],), dtype=torch.float32, device=device)
    model.xyz_gradient_accum = torch.zeros((xyz_t.shape[0], 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs = torch.zeros((xyz_t.shape[0], 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs_max = torch.zeros((xyz_t.shape[0], 1), dtype=torch.float32, device=device)
    model.denom = torch.zeros((xyz_t.shape[0], 1), dtype=torch.float32, device=device)
    model.init_tracking_state(int(xyz_t.shape[0]))
    if source_tag is not None:
        model._source_tag = torch.from_numpy(np.asarray(source_tag, dtype=np.int32)).to(device=device, dtype=torch.int32)
    if seed_id is not None:
        model._seed_id = torch.from_numpy(np.asarray(seed_id, dtype=np.int64)).to(device=device, dtype=torch.int64)
    if generation is not None:
        model._generation = torch.from_numpy(np.asarray(generation, dtype=np.int32)).to(device=device, dtype=torch.int32)
    return model


def build_patch_child_overlay_model(
    base: GaussianModel,
    *,
    centers: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    normals: np.ndarray,
    scale_u: np.ndarray,
    scale_v: np.ndarray,
    scale_n: np.ndarray,
    rgb: np.ndarray,
    opacity_gate: np.ndarray,
    patch_offsets_uv: np.ndarray,
    offset_scale: float,
    scale_xy: float,
    scale_n_mul: float,
    opacity_scale: float,
    opacity_min: float,
    opacity_max: float,
    source_tag: Optional[np.ndarray] = None,
    seed_id: Optional[np.ndarray] = None,
    generation: Optional[np.ndarray] = None,
) -> GaussianModel:
    centers = np.asarray(centers, dtype=np.float32)
    tangent_u = np.asarray(tangent_u, dtype=np.float32)
    tangent_v = np.asarray(tangent_v, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    scale_u = np.asarray(scale_u, dtype=np.float32)
    scale_v = np.asarray(scale_v, dtype=np.float32)
    scale_n = np.asarray(scale_n, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    opacity_gate = np.asarray(opacity_gate, dtype=np.float32)
    patch_offsets_uv = np.asarray(patch_offsets_uv, dtype=np.float32)

    if patch_offsets_uv.ndim != 2 or patch_offsets_uv.shape[1] != 2:
        raise ValueError(f"patch_offsets_uv must be Nx2, got shape {tuple(patch_offsets_uv.shape)}")
    patch_count = int(patch_offsets_uv.shape[0])
    count = int(centers.shape[0])

    offset_u = patch_offsets_uv[None, :, 0:1] * scale_u[:, None, None] * float(offset_scale)
    offset_v = patch_offsets_uv[None, :, 1:2] * scale_v[:, None, None] * float(offset_scale)
    xyz = centers[:, None, :] + tangent_u[:, None, :] * offset_u + tangent_v[:, None, :] * offset_v
    xyz = xyz.reshape(-1, 3)

    rotation_matrix = np.repeat(
        np.stack([tangent_u, tangent_v, normals], axis=2),
        patch_count,
        axis=0,
    ).astype(np.float32, copy=False)
    scales = np.repeat(
        np.stack(
            [
                np.maximum(scale_u * float(scale_xy), 1e-6),
                np.maximum(scale_v * float(scale_xy), 1e-6),
                np.maximum(scale_n * float(scale_n_mul), 1e-6),
            ],
            axis=1,
        ),
        patch_count,
        axis=0,
    ).astype(np.float32, copy=False)
    rgb_rep = np.repeat(rgb.astype(np.float32, copy=False), patch_count, axis=0)
    opacity_prob = np.repeat(
        np.clip(opacity_gate * float(opacity_scale) / math.sqrt(float(max(patch_count, 1))), float(opacity_min), float(opacity_max)),
        patch_count,
        axis=0,
    ).astype(np.float32, copy=False)

    src_rep = np.repeat(source_tag, patch_count, axis=0) if source_tag is not None else None
    seed_rep = np.repeat(seed_id, patch_count, axis=0) if seed_id is not None else None
    gen_rep = np.repeat(generation, patch_count, axis=0) if generation is not None else None
    return build_overlay_gaussian_model(
        base,
        xyz=xyz,
        rgb=rgb_rep,
        rotation_matrix=rotation_matrix,
        scales=scales,
        opacity_prob=opacity_prob,
        source_tag=src_rep,
        seed_id=seed_rep,
        generation=gen_rep,
    )


def alpha_overlay(base_rgb: torch.Tensor, overlay_rgb: torch.Tensor, alpha_hw: torch.Tensor, strength: float) -> torch.Tensor:
    alpha = alpha_hw.detach().clamp(0.0, 1.0).unsqueeze(0) * float(strength)
    return torch.clamp(base_rgb * (1.0 - alpha) + overlay_rgb * alpha, 0.0, 1.0)


def make_labeled_grid(tiles: Sequence[Tuple[str, np.ndarray]], columns: int, pad: int = 8, label_height: int = 24) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles provided for grid generation.")
    columns = max(1, int(columns))
    sample = tiles[0][1]
    tile_h, tile_w = int(sample.shape[0]), int(sample.shape[1])
    rows = int(math.ceil(len(tiles) / float(columns)))
    canvas_w = columns * tile_w + (columns + 1) * pad
    canvas_h = rows * (tile_h + label_height) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image_np) in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x0 = pad + col * (tile_w + pad)
        y0 = pad + row * (tile_h + label_height + pad)
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + label_height - 1], fill=(32, 32, 32))
        draw.text((x0 + 6, y0 + 4), str(label), fill=(235, 235, 235))
        canvas.paste(Image.fromarray(image_np, mode="RGB"), (x0, y0 + label_height))
    return canvas


def load_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def choose_preview_mask(payload: Dict[str, np.ndarray], args) -> np.ndarray:
    valid_mask = np.asarray(payload["valid_mask"]).astype(bool, copy=False)
    confidence = np.asarray(payload["confidence"], dtype=np.float32)
    detail_confidence = np.asarray(payload["detail_confidence"], dtype=np.float32)
    risk = np.asarray(payload["risk"], dtype=np.float32)

    mask = valid_mask.copy()
    mask &= confidence >= float(args.min_confidence)
    mask &= risk <= float(args.max_risk)
    if bool(args.require_detail):
        mask &= detail_confidence >= float(args.min_detail_confidence)
    return mask


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Inspect/render preview assets for a SOF gaussian surface-response payload.")
    parser.add_argument("--response_payload_path", required=True)
    parser.add_argument("--sof_model_path", required=True)
    parser.add_argument("--sof_iteration", type=int, default=-1)
    parser.add_argument("--scene_root", default="")
    parser.add_argument("--camera_model_path", default="")
    parser.add_argument("--prior_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--preview_views", type=int, default=6)
    parser.add_argument("--preview_grid_columns", type=int, default=4)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--min_detail_confidence", type=float, default=0.0)
    parser.add_argument("--max_risk", type=float, default=1.0)
    parser.add_argument("--require_detail", action="store_true")
    parser.add_argument("--preview_point_cap", type=int, default=250000)
    parser.add_argument("--preview_point_seed", type=int, default=0)
    parser.add_argument("--preview_min_opacity", type=float, default=0.03)
    parser.add_argument("--preview_max_opacity", type=float, default=0.95)
    parser.add_argument("--preview_gate_power", type=float, default=0.75)
    parser.add_argument("--child_overlay_scale_xy", type=float, default=0.18)
    parser.add_argument("--child_overlay_scale_n", type=float, default=0.75)
    parser.add_argument("--child_overlay_offset_scale", type=float, default=1.0)
    parser.add_argument("--child_overlay_opacity_scale", type=float, default=0.45)
    parser.add_argument("--child_overlay_opacity_min", type=float, default=0.01)
    parser.add_argument("--child_overlay_opacity_max", type=float, default=0.18)
    parser.add_argument("--overlay_strength", type=float, default=0.65)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    safe_state(bool(args.quiet))

    if not torch.cuda.is_available():
        raise RuntimeError("inspect_sof_gaussian_response_v0 currently requires CUDA.")

    response_payload_path = Path(args.response_payload_path).expanduser().resolve()
    sof_model_path = Path(args.sof_model_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else response_payload_path.parent / "inspect_sof_gaussian_response_v0"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not response_payload_path.is_file():
        raise FileNotFoundError(f"response_payload_path not found: {response_payload_path}")
    if not sof_model_path.is_dir():
        raise FileNotFoundError(f"sof_model_path not found: {sof_model_path}")

    prior_dir = Path(args.prior_dir).expanduser().resolve() if str(args.prior_dir).strip() else None
    if prior_dir is not None and not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_dir not found: {prior_dir}")

    scene_root = resolve_scene_root(args)
    camera_model_path = resolve_camera_model_path(args, sof_model_path)
    if not scene_root.is_dir():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")
    if not camera_model_path.is_dir():
        raise FileNotFoundError(f"camera_model_path not found: {camera_model_path}")

    payload = load_payload(response_payload_path)
    required_keys = [
        "gaussian_ids",
        "centers",
        "fused_rgb",
        "confidence",
        "detail_confidence",
        "risk",
        "target_high_rgb",
        "valid_mask",
    ]
    for key in required_keys:
        if key not in payload:
            raise KeyError(f"Payload missing required key: {key}")

    sof_iteration = resolve_iteration(sof_model_path, int(args.sof_iteration))
    model = load_model_ply(sof_model_path, sof_iteration, sh_degree=3)
    cameras = load_cameras_for_split(scene_root, camera_model_path, str(args.images_subdir), str(args.split))
    cameras = select_uniform(cameras, int(args.preview_views))
    if not cameras:
        raise RuntimeError("No preview cameras available for inspection.")

    gaussian_ids = np.asarray(payload["gaussian_ids"], dtype=np.int64)
    total_gaussians = int(model.get_xyz.shape[0])
    if gaussian_ids.ndim != 1:
        raise ValueError(f"gaussian_ids must be 1D, got shape {tuple(gaussian_ids.shape)}")
    if np.any(gaussian_ids < 0) or np.any(gaussian_ids >= total_gaussians):
        raise ValueError(
            f"Payload gaussian_ids exceed current model size: max_id={int(gaussian_ids.max())} total={total_gaussians}"
        )

    preview_mask_np = choose_preview_mask(payload, args)
    preview_count = int(preview_mask_np.sum())
    if preview_count <= 0:
        raise RuntimeError("No payload carriers remain after preview filtering.")

    fused_rgb = np.asarray(payload["fused_rgb"], dtype=np.float32)
    anchor_rgb = np.asarray(payload["anchor_rgb"], dtype=np.float32) if "anchor_rgb" in payload else None
    confidence = np.asarray(payload["confidence"], dtype=np.float32)
    detail_confidence = np.asarray(payload["detail_confidence"], dtype=np.float32)
    risk = np.asarray(payload["risk"], dtype=np.float32)
    target_high_rgb = np.asarray(payload["target_high_rgb"], dtype=np.float32)
    fused_delta_rgb = np.asarray(payload["fused_delta_rgb"], dtype=np.float32) if "fused_delta_rgb" in payload else (fused_rgb - anchor_rgb if anchor_rgb is not None else None)
    centers = np.asarray(payload["centers"], dtype=np.float32)
    tangent_u = np.asarray(payload["tangent_u"], dtype=np.float32)
    tangent_v = np.asarray(payload["tangent_v"], dtype=np.float32)
    normals = np.asarray(payload["normals"], dtype=np.float32)
    scale_u = np.asarray(payload["scale_u"], dtype=np.float32)
    scale_v = np.asarray(payload["scale_v"], dtype=np.float32)
    scale_n = np.asarray(payload["scale_n"], dtype=np.float32)

    conf_rgb = scalar_to_rgb(confidence, invert=False)
    detail_conf_rgb = scalar_to_rgb(detail_confidence, invert=False)
    risk_rgb = scalar_to_rgb(risk, invert=True)
    detail_rgb = detail_residual_to_rgb(target_high_rgb, clamp_value=0.08)

    preview_global_mask = np.zeros((total_gaussians,), dtype=bool)
    preview_global_mask[gaussian_ids[preview_mask_np]] = True
    preview_mask_t = torch.from_numpy(preview_global_mask).to(device=model.get_xyz.device, dtype=torch.bool)
    preview_subset = _clone_subset_gaussians(model, preview_mask_t)
    baked_model = _clone_subset_gaussians(model, torch.ones((total_gaussians,), device=model.get_xyz.device, dtype=torch.bool))
    assign_rgb_to_gaussians_in_place(
        baked_model,
        gaussian_ids[preview_mask_np],
        fused_rgb[preview_mask_np],
        zero_rest=False,
    )

    opacity_gate = np.clip(confidence[preview_mask_np], 0.05, 1.0)
    apply_preview_opacity_in_place(
        preview_subset,
        opacity_gate=opacity_gate,
        min_opacity=float(args.preview_min_opacity),
        max_opacity=float(args.preview_max_opacity),
        gate_power=float(args.preview_gate_power),
    )
    if "detail_apply" in payload:
        child_opacity_gate = np.maximum(payload["low_apply"], payload["detail_apply"]).astype(np.float32, copy=False)[
            preview_mask_np
        ]
    elif "low_apply" in payload:
        child_opacity_gate = np.asarray(payload["low_apply"], dtype=np.float32)[preview_mask_np]
    else:
        child_opacity_gate = confidence[preview_mask_np]
    patch_offsets_uv = np.asarray(payload.get("patch_offsets_uv", np.asarray([[0.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    local_child_model = build_patch_child_overlay_model(
        model,
        centers=centers[preview_mask_np],
        tangent_u=tangent_u[preview_mask_np],
        tangent_v=tangent_v[preview_mask_np],
        normals=normals[preview_mask_np],
        scale_u=scale_u[preview_mask_np],
        scale_v=scale_v[preview_mask_np],
        scale_n=scale_n[preview_mask_np],
        rgb=fused_rgb[preview_mask_np],
        opacity_gate=child_opacity_gate,
        patch_offsets_uv=patch_offsets_uv,
        offset_scale=float(args.child_overlay_offset_scale),
        scale_xy=float(args.child_overlay_scale_xy),
        scale_n_mul=float(args.child_overlay_scale_n),
        opacity_scale=float(args.child_overlay_opacity_scale),
        opacity_min=float(args.child_overlay_opacity_min),
        opacity_max=float(args.child_overlay_opacity_max),
        source_tag=np.asarray(payload["source_tag"], dtype=np.int32)[preview_mask_np] if "source_tag" in payload else None,
        seed_id=np.asarray(payload["seed_ids"], dtype=np.int64)[preview_mask_np] if "seed_ids" in payload else None,
        generation=np.asarray(payload["generation"], dtype=np.int32)[preview_mask_np] if "generation" in payload else None,
    )
    merged_local_overlay_model = merge_gaussian_models(model, local_child_model)

    override_fused = torch.from_numpy(np.clip(fused_rgb[preview_mask_np], 0.0, 1.0)).to(device=model.get_xyz.device, dtype=torch.float32)
    override_conf = torch.from_numpy(conf_rgb[preview_mask_np]).to(device=model.get_xyz.device, dtype=torch.float32)
    override_detail_conf = torch.from_numpy(detail_conf_rgb[preview_mask_np]).to(device=model.get_xyz.device, dtype=torch.float32)
    override_risk = torch.from_numpy(risk_rgb[preview_mask_np]).to(device=model.get_xyz.device, dtype=torch.float32)
    override_detail = torch.from_numpy(detail_rgb[preview_mask_np]).to(device=model.get_xyz.device, dtype=torch.float32)

    background = torch.ones((3,), dtype=torch.float32, device="cuda") if bool(args.white_background) else torch.zeros((3,), dtype=torch.float32, device="cuda")

    point_dir = output_dir / "point_cloud_previews"
    point_dir.mkdir(parents=True, exist_ok=True)
    exported_point_counts = {
        "anchor_rgb": export_point_cloud(
            point_dir / "carrier_anchor_rgb_preview_v0.ply",
            centers[preview_mask_np],
            np.clip(anchor_rgb[preview_mask_np], 0.0, 1.0) if anchor_rgb is not None else np.clip(fused_rgb[preview_mask_np], 0.0, 1.0),
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "fused_rgb": export_point_cloud(
            point_dir / "carrier_fused_rgb_preview_v0.ply",
            centers[preview_mask_np],
            np.clip(fused_rgb[preview_mask_np], 0.0, 1.0),
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "confidence": export_point_cloud(
            point_dir / "carrier_confidence_preview_v0.ply",
            centers[preview_mask_np],
            conf_rgb[preview_mask_np],
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "detail_confidence": export_point_cloud(
            point_dir / "carrier_detail_confidence_preview_v0.ply",
            centers[preview_mask_np],
            detail_conf_rgb[preview_mask_np],
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "risk": export_point_cloud(
            point_dir / "carrier_risk_preview_v0.ply",
            centers[preview_mask_np],
            risk_rgb[preview_mask_np],
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "detail_rgb": export_point_cloud(
            point_dir / "carrier_detail_rgb_preview_v0.ply",
            centers[preview_mask_np],
            detail_rgb[preview_mask_np],
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
    }
    if fused_delta_rgb is not None:
        exported_point_counts["fused_delta_rgb"] = export_point_cloud(
            point_dir / "carrier_fused_delta_rgb_preview_v0.ply",
            centers[preview_mask_np],
            detail_residual_to_rgb(fused_delta_rgb[preview_mask_np], clamp_value=0.12),
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        )

    prior_index = index_image_dir(str(prior_dir)) if prior_dir is not None else {}
    render_root = output_dir / "render_previews"
    render_root.mkdir(parents=True, exist_ok=True)
    view_summaries: List[Dict[str, object]] = []
    contact_tiles: List[Tuple[str, np.ndarray]] = []

    for view_idx, camera in enumerate(cameras):
        base_pkg = render_simple(camera, model, background)
        baked_pkg = render_simple(camera, baked_model, background)
        local_child_pkg = render_simple(camera, local_child_model, background)
        merged_local_pkg = render_simple(camera, merged_local_overlay_model, background)
        fused_pkg = render_simple(camera, preview_subset, background, override_color=override_fused)
        conf_pkg = render_simple(camera, preview_subset, background, override_color=override_conf)
        detail_conf_pkg = render_simple(camera, preview_subset, background, override_color=override_detail_conf)
        risk_pkg = render_simple(camera, preview_subset, background, override_color=override_risk)
        detail_pkg = render_simple(camera, preview_subset, background, override_color=override_detail)

        base_rgb = base_pkg["render"]
        baked_rgb = baked_pkg["render"]
        local_child_rgb = local_child_pkg["render"]
        merged_local_rgb = merged_local_pkg["render"]
        fused_rgb_img = fused_pkg["render"]
        conf_rgb_img = conf_pkg["render"]
        detail_conf_rgb_img = detail_conf_pkg["render"]
        risk_rgb_img = risk_pkg["render"]
        detail_rgb_img = detail_pkg["render"]
        fused_alpha = fused_pkg["alpha"][0]
        overlay_rgb = alpha_overlay(base_rgb, fused_rgb_img, fused_alpha, strength=float(args.overlay_strength))
        baked_delta = torch.mean(torch.abs(baked_rgb - base_rgb), dim=0)
        local_overlay_delta = torch.mean(torch.abs(merged_local_rgb - base_rgb), dim=0)

        view_name = normalize_image_name(str(camera.image_name))
        view_dir = render_root / f"{view_idx:03d}_{view_name}"
        view_dir.mkdir(parents=True, exist_ok=True)

        base_u8 = to_uint8_rgb(base_rgb)
        baked_u8 = to_uint8_rgb(baked_rgb)
        local_child_u8 = to_uint8_rgb(local_child_rgb)
        merged_local_u8 = to_uint8_rgb(merged_local_rgb)
        fused_u8 = to_uint8_rgb(fused_rgb_img)
        overlay_u8 = to_uint8_rgb(overlay_rgb)
        conf_u8 = to_uint8_rgb(conf_rgb_img)
        detail_conf_u8 = to_uint8_rgb(detail_conf_rgb_img)
        risk_u8 = to_uint8_rgb(risk_rgb_img)
        detail_u8 = to_uint8_rgb(detail_rgb_img)
        baked_delta_u8 = scalar_image_to_rgb(baked_delta.detach().cpu().numpy(), invert=False)
        local_overlay_delta_u8 = scalar_image_to_rgb(local_overlay_delta.detach().cpu().numpy(), invert=False)

        Image.fromarray(base_u8, mode="RGB").save(view_dir / "base_render.png")
        Image.fromarray(baked_u8, mode="RGB").save(view_dir / "baked_fused_render.png")
        Image.fromarray(baked_delta_u8, mode="RGB").save(view_dir / "baked_delta_heatmap.png")
        Image.fromarray(local_child_u8, mode="RGB").save(view_dir / "local_child_overlay_render.png")
        Image.fromarray(merged_local_u8, mode="RGB").save(view_dir / "merged_local_overlay_render.png")
        Image.fromarray(local_overlay_delta_u8, mode="RGB").save(view_dir / "local_overlay_delta_heatmap.png")
        Image.fromarray(local_child_u8, mode="RGB").save(view_dir / "patch_child_overlay_render.png")
        Image.fromarray(merged_local_u8, mode="RGB").save(view_dir / "merged_patch_overlay_render.png")
        Image.fromarray(local_overlay_delta_u8, mode="RGB").save(view_dir / "patch_overlay_delta_heatmap.png")
        Image.fromarray(fused_u8, mode="RGB").save(view_dir / "fused_render.png")
        Image.fromarray(fused_u8, mode="RGB").save(view_dir / "carrier_fused_render.png")
        Image.fromarray(overlay_u8, mode="RGB").save(view_dir / "overlay_render.png")
        Image.fromarray(overlay_u8, mode="RGB").save(view_dir / "carrier_overlay_preview.png")
        Image.fromarray(conf_u8, mode="RGB").save(view_dir / "confidence_render.png")
        Image.fromarray(detail_conf_u8, mode="RGB").save(view_dir / "detail_confidence_render.png")
        Image.fromarray(risk_u8, mode="RGB").save(view_dir / "risk_render.png")
        Image.fromarray(detail_u8, mode="RGB").save(view_dir / "detail_rgb_render.png")
        Image.fromarray(np.clip(fused_alpha.detach().cpu().numpy() * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8), mode="L").save(
            view_dir / "fused_alpha.png"
        )

        tiles: List[Tuple[str, np.ndarray]] = []
        prior_path = lookup_indexed_path(prior_index, str(camera.image_name)) if prior_index else None
        if prior_path is not None:
            prior_rgb = resize_rgb_uint8(load_rgb_uint8(prior_path), size_hw=base_u8.shape[:2])
            Image.fromarray(prior_rgb, mode="RGB").save(view_dir / "prior_rgb.png")
            tiles.append(("prior", prior_rgb))
        tiles.extend(
            [
                ("base", base_u8),
                ("merged_patch", merged_local_u8),
                ("patch_delta", local_overlay_delta_u8),
                ("baked_fused", baked_u8),
                ("baked_delta", baked_delta_u8),
                ("patch_child", local_child_u8),
                ("fused", fused_u8),
                ("carrier_overlay", overlay_u8),
                ("confidence", conf_u8),
                ("detail_conf", detail_conf_u8),
                ("risk", risk_u8),
                ("detail_rgb", detail_u8),
            ]
        )
        grid = make_labeled_grid(tiles, columns=int(args.preview_grid_columns))
        grid.save(view_dir / "comparison_grid.png")

        contact_tiles.append((f"{view_idx:03d}_{view_name}", np.asarray(grid.convert("RGB"), dtype=np.uint8)))
        view_summaries.append(
            {
                "view_index": int(view_idx),
                "image_name": str(camera.image_name),
                "prior_path": str(prior_path) if prior_path is not None else None,
                "output_dir": str(view_dir.resolve()),
                "fused_alpha_mean": float(fused_alpha.mean().item()),
                "fused_alpha_p95": float(np.percentile(fused_alpha.detach().cpu().numpy().reshape(-1), 95.0)),
                "baked_delta_mean": float(baked_delta.mean().item()),
                "baked_delta_p95": float(np.percentile(baked_delta.detach().cpu().numpy().reshape(-1), 95.0)),
                "local_overlay_delta_mean": float(local_overlay_delta.mean().item()),
                "local_overlay_delta_p95": float(np.percentile(local_overlay_delta.detach().cpu().numpy().reshape(-1), 95.0)),
            }
        )

    overview = make_labeled_grid(contact_tiles, columns=1)
    overview.save(output_dir / "comparison_overview_v0.png")

    summary = {
        "mode": "inspect_sof_gaussian_response_v0",
        "response_payload_path": str(response_payload_path),
        "sof_model_path": str(sof_model_path),
        "sof_iteration": int(sof_iteration),
        "scene_root": str(scene_root),
        "camera_model_path": str(camera_model_path),
        "prior_dir": str(prior_dir) if prior_dir is not None else "",
        "output_dir": str(output_dir),
        "preview_count": int(preview_count),
        "payload_count": int(gaussian_ids.shape[0]),
        "selected_ratio": float(preview_count / max(int(gaussian_ids.shape[0]), 1)),
        "point_preview_counts": exported_point_counts,
        "preview_filter": {
            "min_confidence": float(args.min_confidence),
            "min_detail_confidence": float(args.min_detail_confidence),
            "max_risk": float(args.max_risk),
            "require_detail": bool(args.require_detail),
        },
        "local_child_overlay": {
            "scale_xy": float(args.child_overlay_scale_xy),
            "scale_n": float(args.child_overlay_scale_n),
            "offset_scale": float(args.child_overlay_offset_scale),
            "opacity_scale": float(args.child_overlay_opacity_scale),
            "opacity_min": float(args.child_overlay_opacity_min),
            "opacity_max": float(args.child_overlay_opacity_max),
            "patch_count": int(patch_offsets_uv.shape[0]),
        },
        "stats": {
            "confidence": stats_from_array(confidence[preview_mask_np]),
            "detail_confidence": stats_from_array(detail_confidence[preview_mask_np]),
            "risk": stats_from_array(risk[preview_mask_np]),
            "detail_energy": stats_from_array(np.mean(np.abs(target_high_rgb[preview_mask_np]), axis=1)),
        },
        "paths": {
            "comparison_overview": str((output_dir / "comparison_overview_v0.png").resolve()),
            "point_cloud_previews": str(point_dir.resolve()),
            "render_previews": str(render_root.resolve()),
        },
        "view_summaries": view_summaries,
        "note": (
            "This inspector renders the original SOF field and a carrier-only preview subset colored by "
            "fused_rgb/confidence/detail_confidence/risk so the response payload can be checked before "
            "it is wired into the final render-field training. baked_fused_render renders a full cloned "
            "SOF field with selected carrier responses written back once, while overlay/carrier_fused "
            "remain carrier-only previews that can look harsher because they add an extra splat layer. "
            "merged_local_overlay_render instead keeps the parent field unchanged and adds a bundle of "
            "small tangent-plane child gaussians expanded from patch_offsets_uv, which better matches the "
            "intended semantics that overlay evidence should not affect the parent gaussian outside the "
            "supported local patch."
        ),
    }
    summary_path = output_dir / "inspect_sof_gaussian_response_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
