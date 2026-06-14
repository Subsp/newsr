#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene import Scene
from scene.gaussian_model import GaussianModel
from score_mesh_delta_star_gaussians_v0 import read_vertex_table, xyz_from_table
from utils.general_utils import build_rotation


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=white_background,
        data_device="cpu",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
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


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def _ensure_tracking_state(gaussians: GaussianModel) -> None:
    count = int(gaussians.get_xyz.shape[0])
    if not hasattr(gaussians, "_source_tag") or int(gaussians._source_tag.shape[0]) != count:
        gaussians.init_tracking_state(count)


def _make_static_gaussian_model(
    *,
    base: GaussianModel,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    tracking_state: Dict[str, torch.Tensor],
) -> GaussianModel:
    model = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    model.active_sh_degree = int(base.active_sh_degree)
    model.spatial_lr_scale = float(base.spatial_lr_scale)
    model._xyz = nn.Parameter(xyz.detach().clone().requires_grad_(False))
    model._features_dc = nn.Parameter(features_dc.detach().clone().requires_grad_(False))
    model._features_rest = nn.Parameter(features_rest.detach().clone().requires_grad_(False))
    model._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(False))
    model._scaling = nn.Parameter(scaling.detach().clone().requires_grad_(False))
    model._rotation = nn.Parameter(rotation.detach().clone().requires_grad_(False))
    model.filter_3D = filter_3d.detach().clone()
    count = int(xyz.shape[0])
    device = xyz.device
    model.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=device)
    model.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.denom = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.init_tracking_state(count)
    model._source_tag = tracking_state["source_tag"].to(device=device, dtype=torch.int32)
    model._seed_id = tracking_state["seed_id"].to(device=device, dtype=torch.int64)
    model._generation = tracking_state["generation"].to(device=device, dtype=torch.int32)
    model._edge_touched = tracking_state["edge_touched"].to(device=device, dtype=torch.bool)
    model._edge_touch_iter = tracking_state["edge_touch_iter"].to(device=device, dtype=torch.int32)
    return model


def _save_model_snapshot(
    *,
    output_root: Path,
    loaded_iter: int,
    source_model_path: Path,
    base_model: GaussianModel,
    state: Dict[str, torch.Tensor],
) -> None:
    _copy_render_config(source_model_path, output_root)
    point_dir = output_root / "point_cloud" / f"iteration_{loaded_iter}"
    point_dir.mkdir(parents=True, exist_ok=True)
    model = _make_static_gaussian_model(
        base=base_model,
        xyz=state["xyz"],
        features_dc=state["features_dc"],
        features_rest=state["features_rest"],
        opacity=state["opacity"],
        scaling=state["scaling"],
        rotation=state["rotation"],
        filter_3d=state["filter_3d"],
        tracking_state=state["tracking_state"],
    )
    model.save_ply(str(point_dir / "point_cloud.ply"))
    model.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probability / torch.clamp(1.0 - probability, min=1e-6))


def _tau_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = torch.clamp(alpha, min=1e-6, max=1.0 - 1e-6)
    return -torch.log(torch.clamp(1.0 - alpha, min=1e-6))


def _alpha_from_tau(tau: torch.Tensor) -> torch.Tensor:
    tau = torch.clamp(tau, min=0.0)
    return 1.0 - torch.exp(-tau)


def _opacity_mass_metric(scales: torch.Tensor, mode: str) -> torch.Tensor:
    safe = torch.clamp(scales, min=1e-12)
    if mode == "volume":
        return torch.prod(safe, dim=1, keepdim=True)
    if mode == "area":
        sorted_scales = torch.sort(safe, dim=1).values
        return torch.prod(sorted_scales[:, -2:], dim=1, keepdim=True)
    return torch.ones((safe.shape[0], 1), dtype=safe.dtype, device=safe.device)


def _line_split_pattern(split_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    count = max(2, int(split_count))
    x = torch.linspace(-1.0, 1.0, steps=count, dtype=dtype, device=device)
    zeros = torch.zeros_like(x)
    return torch.stack((x, zeros, zeros), dim=1)


def _grid_split_pattern(split_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = max(2, int(np.ceil(np.sqrt(max(int(split_count), 4)))))
    coords = torch.linspace(-1.0, 1.0, steps=side, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    zz = torch.zeros_like(xx)
    pattern = torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=1)
    if int(split_count) > 0 and int(split_count) < int(pattern.shape[0]):
        pattern = pattern[: int(split_count)]
    return pattern


def _adaptive_chunk_counts(
    sorted_selected_scale: torch.Tensor,
    *,
    base_split_count: int,
    max_split_count: int,
    chunk_aspect_target: float,
) -> torch.Tensor:
    base = max(2, int(base_split_count))
    cap = max(base, int(max_split_count))
    safe = torch.clamp(sorted_selected_scale, min=1e-8)
    major = safe[:, 0]
    support = torch.maximum(safe[:, 1], safe[:, 2])
    ratio = major / torch.clamp(support, min=1e-8)
    target = max(float(chunk_aspect_target), 1.0)
    counts = torch.ceil(ratio / target).to(dtype=torch.int64)
    return torch.clamp(counts, min=base, max=cap)


def _scalar_stats(values: np.ndarray | torch.Tensor) -> Dict[str, float | int]:
    if isinstance(values, torch.Tensor):
        arr = values.detach().float().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _load_mesh_vertices(mesh_path: str) -> np.ndarray | None:
    mesh_text = str(mesh_path).strip()
    if not mesh_text:
        return None
    path = Path(mesh_text).expanduser().resolve()
    if not path.is_file():
        return None
    mesh_table, _ = read_vertex_table(path)
    mesh_xyz = xyz_from_table(mesh_table)
    return np.asarray(mesh_xyz, dtype=np.float32)


def _load_tensor_1d(payload: dict, key: str, dtype: torch.dtype) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"Key '{key}' not found in payload.")
    value = payload[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(dtype=dtype, device="cpu").reshape(-1)


@dataclass
class ModulePayload:
    path: Path
    mask_key: str
    score_key: str
    source_mask: np.ndarray
    source_score: np.ndarray
    nearest_surface_index: np.ndarray | None


@dataclass
class ModuleConfig:
    name: str
    enabled: bool
    payload: ModulePayload | None
    apply_to_children: bool
    exclude_selected_key: str
    max_fraction: float
    max_count: int
    split_count: int
    max_split_count: int
    child_layout: str
    chunk_aspect_target: float
    offset_scale: float
    parent_tau_keep: float
    child_tau_ratio: float
    mass_cap_eps: float
    parent_dc_scale: float
    parent_rest_scale: float
    child_major_scale_multiplier: float
    child_minor_scale_multiplier: float
    child_normal_scale_multiplier: float
    child_dc_scale: float
    child_rest_scale: float
    child_filter_scale: float
    filter_cap_ratio: float
    energy_conserve_mode: str
    mesh_pull_lambda: float


def _load_module_payload(
    *,
    payload_path: str,
    mask_key: str,
    score_key: str,
    nearest_surface_key: str,
    expected_count: int,
) -> ModulePayload | None:
    path_text = str(payload_path).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)!r}")
    source_mask = _load_tensor_1d(payload, str(mask_key), torch.bool)
    if int(source_mask.shape[0]) != int(expected_count):
        raise ValueError(
            f"Mask '{mask_key}' length mismatch in {path}: expected {expected_count}, got {int(source_mask.shape[0])}"
        )
    if str(score_key).strip():
        if str(score_key) in payload:
            source_score = _load_tensor_1d(payload, str(score_key), torch.float32)
        else:
            source_score = torch.ones((expected_count,), dtype=torch.float32)
    else:
        source_score = torch.ones((expected_count,), dtype=torch.float32)
    if int(source_score.shape[0]) != int(expected_count):
        raise ValueError(
            f"Score '{score_key}' length mismatch in {path}: expected {expected_count}, got {int(source_score.shape[0])}"
        )
    nearest_surface_index = None
    if str(nearest_surface_key).strip() and str(nearest_surface_key) in payload:
        nearest_surface = _load_tensor_1d(payload, str(nearest_surface_key), torch.int64)
        if int(nearest_surface.shape[0]) != int(expected_count):
            raise ValueError(
                f"Nearest-surface key '{nearest_surface_key}' length mismatch in {path}: "
                f"expected {expected_count}, got {int(nearest_surface.shape[0])}"
            )
        nearest_surface_index = nearest_surface.numpy().astype(np.int64, copy=False)
    return ModulePayload(
        path=path,
        mask_key=str(mask_key),
        score_key=str(score_key),
        source_mask=source_mask.numpy().astype(bool, copy=False),
        source_score=source_score.numpy().astype(np.float32, copy=False),
        nearest_surface_index=nearest_surface_index,
    )


def _module_payload_to_summary(payload: ModulePayload | None) -> dict | None:
    if payload is None:
        return None
    return {
        "path": str(payload.path),
        "mask_key": payload.mask_key,
        "score_key": payload.score_key,
        "source_positive_count": int(np.count_nonzero(payload.source_mask)),
    }


def _as_int64_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.int64)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.astype(np.int64, copy=False)).to(torch.int64)
    if value is None:
        return torch.empty((0,), dtype=torch.int64)
    return torch.as_tensor(value, dtype=torch.int64)


def _build_initial_state(gaussians: GaussianModel) -> Dict[str, torch.Tensor]:
    count = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    if isinstance(gaussians.filter_3D, torch.Tensor) and gaussians.filter_3D.ndim > 0:
        filter_3d = gaussians.filter_3D.detach().clone()
        if filter_3d.ndim == 1:
            filter_3d = filter_3d.reshape(-1, 1)
    else:
        filter_3d = torch.zeros((count, 1), dtype=torch.float32, device=device)
    return {
        "xyz": gaussians._xyz.detach().clone(),
        "features_dc": gaussians._features_dc.detach().clone(),
        "features_rest": gaussians._features_rest.detach().clone(),
        "opacity": gaussians._opacity.detach().clone(),
        "scaling": gaussians._scaling.detach().clone(),
        "rotation": gaussians._rotation.detach().clone(),
        "filter_3d": filter_3d,
        "tracking_state": {
            "source_tag": gaussians._source_tag.detach().clone(),
            "seed_id": gaussians._seed_id.detach().clone(),
            "generation": gaussians._generation.detach().clone(),
            "edge_touched": gaussians._edge_touched.detach().clone(),
            "edge_touch_iter": gaussians._edge_touch_iter.detach().clone(),
        },
        "output_source_idx": torch.arange(count, dtype=torch.int64, device=device),
        "is_child_output_mask": torch.zeros((count,), dtype=torch.bool, device=device),
        "geometry_selected_output_mask": torch.zeros((count,), dtype=torch.bool, device=device),
        "geometry_child_output_mask": torch.zeros((count,), dtype=torch.bool, device=device),
        "highlight_selected_output_mask": torch.zeros((count,), dtype=torch.bool, device=device),
        "highlight_child_output_mask": torch.zeros((count,), dtype=torch.bool, device=device),
    }


def _extend_bool_masks(state: Dict[str, torch.Tensor], new_count: int) -> None:
    if int(new_count) <= 0:
        return
    device = state["xyz"].device
    keys = [
        "is_child_output_mask",
        "geometry_selected_output_mask",
        "geometry_child_output_mask",
        "highlight_selected_output_mask",
        "highlight_child_output_mask",
    ]
    for key in keys:
        state[key] = torch.cat((state[key], torch.zeros((new_count,), dtype=torch.bool, device=device)), dim=0)


def _select_output_indices(state: Dict[str, torch.Tensor], payload: ModulePayload, apply_to_children: bool) -> torch.Tensor:
    source_idx = state["output_source_idx"].detach().cpu().numpy().astype(np.int64, copy=False)
    candidate = payload.source_mask[source_idx]
    if not apply_to_children:
        candidate &= ~state["is_child_output_mask"].detach().cpu().numpy().astype(bool, copy=False)
    candidate_ids = np.flatnonzero(candidate).astype(np.int64, copy=False)
    if candidate_ids.size == 0:
        return torch.zeros((0,), dtype=torch.long, device=state["xyz"].device)
    return torch.from_numpy(candidate_ids).to(device=state["xyz"].device, dtype=torch.long)


def _cap_selected_indices(
    *,
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    max_fraction: float,
    max_count: int,
    reference_count: int,
) -> torch.Tensor:
    selected = candidate_indices
    cap = int(candidate_indices.shape[0])
    if float(max_fraction) > 0.0:
        cap = min(cap, max(1, int(round(float(max_fraction) * float(reference_count)))))
    if int(max_count) > 0:
        cap = min(cap, int(max_count))
    if cap <= 0:
        return candidate_indices[:0]
    if int(candidate_indices.shape[0]) <= cap:
        return selected
    order = torch.argsort(candidate_scores, descending=True, stable=True)[:cap]
    return selected[order]


def _apply_module(
    *,
    state: Dict[str, torch.Tensor],
    config: ModuleConfig,
    mesh_vertices: np.ndarray | None,
) -> Dict[str, object]:
    selected_key = f"{config.name}_selected_output_mask"
    child_key = f"{config.name}_child_output_mask"
    before_count = int(state["xyz"].shape[0])

    summary: Dict[str, object] = {
        "enabled": bool(config.enabled and config.payload is not None),
        "payload": _module_payload_to_summary(config.payload),
        "candidate_count_before_cap": 0,
        "selected_count": 0,
        "child_count": 0,
        "output_count_before": before_count,
        "output_count_after": before_count,
    }
    if not config.enabled or config.payload is None:
        return summary

    payload = config.payload
    candidate_indices = _select_output_indices(state, payload, bool(config.apply_to_children))
    if str(config.exclude_selected_key).strip():
        exclude_mask = state[str(config.exclude_selected_key)]
        candidate_indices = candidate_indices[~exclude_mask[candidate_indices]]
    summary["candidate_count_before_cap"] = int(candidate_indices.shape[0])
    if int(candidate_indices.shape[0]) <= 0:
        return summary

    source_idx = state["output_source_idx"][candidate_indices]
    candidate_scores = torch.from_numpy(payload.source_score).to(device=state["xyz"].device, dtype=torch.float32)[source_idx]
    candidate_scores = torch.nan_to_num(candidate_scores, nan=0.0, posinf=0.0, neginf=0.0)
    selected_indices = _cap_selected_indices(
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        max_fraction=float(config.max_fraction),
        max_count=int(config.max_count),
        reference_count=int(payload.source_mask.shape[0]),
    )
    selected_count = int(selected_indices.shape[0])
    summary["selected_count"] = selected_count
    if selected_count <= 0:
        return summary

    selected_source_idx = state["output_source_idx"][selected_indices]
    selected_scores = torch.from_numpy(payload.source_score).to(device=state["xyz"].device, dtype=torch.float32)[selected_source_idx]
    selected_scores = torch.nan_to_num(selected_scores, nan=0.0, posinf=0.0, neginf=0.0)

    device = state["xyz"].device
    dtype = state["xyz"].dtype
    state[selected_key][selected_indices] = True

    xyz = state["xyz"]
    features_dc = state["features_dc"]
    features_rest = state["features_rest"]
    opacity = state["opacity"]
    scaling = state["scaling"]
    rotation = state["rotation"]
    filter_3d = state["filter_3d"]

    scale = torch.exp(scaling)
    filter_col = filter_3d.reshape(-1, 1).to(device=device, dtype=scale.dtype)
    effective_scale = torch.sqrt(torch.square(scale) + torch.square(filter_col))
    selected_scale = effective_scale[selected_indices]
    axis_order = torch.argsort(selected_scale, dim=1, descending=True)
    rotations = build_rotation(rotation[selected_indices])
    axis_basis = torch.gather(rotations, 2, axis_order[:, None, :].expand(-1, 3, -1))
    sorted_selected_scale = torch.gather(selected_scale, 1, axis_order)

    adaptive_chunk_mode = str(config.child_layout) == "major_axis_adaptive_chunk"
    if str(config.child_layout) == "grid":
        pattern = _grid_split_pattern(int(config.split_count), device=device, dtype=dtype)
        split_counts = torch.full((selected_count,), int(pattern.shape[0]), dtype=torch.int64, device=device)
    elif adaptive_chunk_mode:
        pattern = None
        split_counts = _adaptive_chunk_counts(
            sorted_selected_scale,
            base_split_count=int(config.split_count),
            max_split_count=int(config.max_split_count),
            chunk_aspect_target=float(config.chunk_aspect_target),
        )
    else:
        pattern = _line_split_pattern(int(config.split_count), device=device, dtype=dtype)
        split_counts = torch.full((selected_count,), int(pattern.shape[0]), dtype=torch.int64, device=device)
    actual_split_count = int(split_counts.max().item()) if int(split_counts.numel()) > 0 else 0

    if adaptive_chunk_mode:
        child_slot_ids = torch.arange(actual_split_count, device=device, dtype=dtype)[None, :]
        split_center = (split_counts.to(dtype=dtype) - 1.0)[:, None] * 0.5
        split_denom = torch.clamp(split_center, min=1.0)
        normalized_major = (child_slot_ids - split_center) / split_denom
        active_child_mask = child_slot_ids < split_counts[:, None].to(dtype=dtype)
        local_offsets = torch.zeros((selected_count, actual_split_count, 3), dtype=dtype, device=device)
        local_offsets[:, :, 0] = normalized_major * sorted_selected_scale[:, None, 0] * float(config.offset_scale)
    else:
        active_child_mask = torch.ones((selected_count, actual_split_count), dtype=torch.bool, device=device)
        local_offsets = pattern[None, :, :] * sorted_selected_scale[:, None, :] * float(config.offset_scale)
    parent_centers = xyz[selected_indices]

    mesh_pull_count = 0
    if mesh_vertices is not None and payload.nearest_surface_index is not None and float(config.mesh_pull_lambda) > 0.0:
        selected_source_idx = state["output_source_idx"][selected_indices].detach().cpu().numpy().astype(np.int64, copy=False)
        nearest_surface = payload.nearest_surface_index[selected_source_idx]
        valid = (nearest_surface >= 0) & (nearest_surface < mesh_vertices.shape[0])
        if np.any(valid):
            mesh_points = parent_centers.detach().clone()
            mesh_points_valid = torch.from_numpy(mesh_vertices[nearest_surface[valid]]).to(device=device, dtype=parent_centers.dtype)
            mesh_points[torch.from_numpy(valid).to(device=device, dtype=torch.bool)] = mesh_points_valid
            parent_centers = (1.0 - float(config.mesh_pull_lambda)) * parent_centers + float(config.mesh_pull_lambda) * mesh_points
            mesh_pull_count = int(np.count_nonzero(valid))

    child_xyz_all = parent_centers[:, None, :] + torch.einsum("bij,bnj->bni", axis_basis, local_offsets)

    selected_raw_scale = scale[selected_indices]
    sorted_raw_scale = torch.gather(selected_raw_scale, 1, axis_order)
    sorted_child_scale = sorted_raw_scale[:, None, :].expand(-1, actual_split_count, -1).clone()
    if adaptive_chunk_mode:
        chunk_major_factor = torch.minimum(
            torch.full((selected_count,), float(config.child_major_scale_multiplier), dtype=sorted_child_scale.dtype, device=device),
            2.0 / torch.clamp(split_counts.to(dtype=sorted_child_scale.dtype), min=2.0),
        )
        sorted_child_scale[:, :, 0] = sorted_child_scale[:, :, 0] * chunk_major_factor[:, None]
    else:
        sorted_child_scale[:, :, 0] = sorted_child_scale[:, :, 0] * float(config.child_major_scale_multiplier)
    sorted_child_scale[:, :, 1] = sorted_child_scale[:, :, 1] * float(config.child_minor_scale_multiplier)
    sorted_child_scale[:, :, 2] = sorted_child_scale[:, :, 2] * float(config.child_normal_scale_multiplier)
    updated_child_scale = torch.zeros((selected_count, actual_split_count, 3), dtype=sorted_child_scale.dtype, device=device)
    gather_index = axis_order[:, None, :].expand(-1, actual_split_count, -1)
    updated_child_scale.scatter_(2, gather_index, sorted_child_scale)
    child_scale = torch.clamp(updated_child_scale, min=1e-8)

    child_filter = filter_3d[selected_indices, None, :].expand(-1, actual_split_count, -1).clone()
    child_filter = torch.clamp(child_filter * float(config.child_filter_scale), min=0.0)
    if float(config.filter_cap_ratio) > 0.0:
        scene_extent = torch.clamp(torch.norm(torch.max(xyz, dim=0).values - torch.min(xyz, dim=0).values), min=1e-6)
        child_filter = torch.clamp(child_filter, max=scene_extent * float(config.filter_cap_ratio))

    parent_alpha = torch.sigmoid(opacity[selected_indices]).reshape(-1)
    parent_tau = _tau_from_alpha(parent_alpha)
    parent_tau_keep = max(0.0, float(config.parent_tau_keep))
    mass_cap_eps = max(0.0, float(config.mass_cap_eps))
    max_child_ratio = max(0.0, 1.0 + mass_cap_eps - parent_tau_keep)
    effective_child_tau_ratio = min(max(0.0, float(config.child_tau_ratio)), max_child_ratio)

    new_parent_tau = parent_tau * parent_tau_keep
    opacity[selected_indices] = _logit(_alpha_from_tau(new_parent_tau).reshape(-1, 1))
    if float(config.parent_dc_scale) != 1.0:
        features_dc[selected_indices] = features_dc[selected_indices] * float(config.parent_dc_scale)
    if float(config.parent_rest_scale) != 1.0:
        features_rest[selected_indices] = features_rest[selected_indices] * float(config.parent_rest_scale)

    if str(config.energy_conserve_mode) == "none":
        child_tau = (parent_tau * effective_child_tau_ratio / torch.clamp(split_counts.to(dtype=parent_tau.dtype), min=1.0))[:, None].expand(-1, actual_split_count)
        child_tau = child_tau * active_child_mask.to(dtype=child_tau.dtype)
        effective_ratio_tensor = torch.full_like(parent_tau, fill_value=effective_child_tau_ratio)
    else:
        parent_metric = _opacity_mass_metric(selected_scale, str(config.energy_conserve_mode)).reshape(-1)
        child_effective_scale = torch.sqrt(
            torch.square(child_scale) + torch.square(child_filter)
        )
        child_metric = _opacity_mass_metric(
            child_effective_scale.reshape(-1, 3),
            str(config.energy_conserve_mode),
        ).reshape(selected_count, actual_split_count)
        child_metric = child_metric * active_child_mask.to(dtype=child_metric.dtype)
        child_metric_sum = torch.clamp(child_metric.sum(dim=1), min=1e-12)
        target_mass = parent_tau * parent_metric * effective_child_tau_ratio
        child_tau = (target_mass / child_metric_sum)[:, None].expand(-1, actual_split_count)
        realized_mass = child_tau * child_metric
        effective_ratio_tensor = realized_mass.sum(dim=1) / torch.clamp(parent_tau * parent_metric, min=1e-12)
    active_child_mask_flat = active_child_mask.reshape(-1)
    child_xyz = child_xyz_all.reshape(-1, 3)[active_child_mask_flat]
    child_scale = child_scale.reshape(-1, 3)[active_child_mask_flat]
    child_scaling = torch.log(child_scale)
    child_filter = child_filter.reshape(-1, filter_3d.shape[1])[active_child_mask_flat]
    child_tau = child_tau.reshape(-1, 1)[active_child_mask_flat]
    child_opacity = _logit(_alpha_from_tau(child_tau))

    child_features_dc = features_dc[selected_indices, None, ...].expand(
        -1,
        actual_split_count,
        *features_dc.shape[1:],
    ).reshape(-1, *features_dc.shape[1:])[active_child_mask_flat].clone()
    child_features_rest = features_rest[selected_indices, None, ...].expand(
        -1,
        actual_split_count,
        *features_rest.shape[1:],
    ).reshape(-1, *features_rest.shape[1:])[active_child_mask_flat].clone()
    child_features_dc = child_features_dc * float(config.child_dc_scale)
    child_features_rest = child_features_rest * float(config.child_rest_scale)
    child_rotation = rotation[selected_indices, None, :].expand(-1, actual_split_count, -1).reshape(-1, rotation.shape[1])[active_child_mask_flat]

    child_count = int(child_xyz.shape[0])
    tracking_state = state["tracking_state"]
    child_tracking = {
        "source_tag": tracking_state["source_tag"][selected_indices].repeat_interleave(split_counts),
        "seed_id": tracking_state["seed_id"][selected_indices].repeat_interleave(split_counts),
        "generation": (tracking_state["generation"][selected_indices] + 1).repeat_interleave(split_counts),
        "edge_touched": tracking_state["edge_touched"][selected_indices].repeat_interleave(split_counts),
        "edge_touch_iter": tracking_state["edge_touch_iter"][selected_indices].repeat_interleave(split_counts),
    }

    state["xyz"] = torch.cat((xyz, child_xyz), dim=0)
    state["features_dc"] = torch.cat((features_dc, child_features_dc), dim=0)
    state["features_rest"] = torch.cat((features_rest, child_features_rest), dim=0)
    state["opacity"] = torch.cat((opacity, child_opacity), dim=0)
    state["scaling"] = torch.cat((scaling, child_scaling), dim=0)
    state["rotation"] = torch.cat((rotation, child_rotation), dim=0)
    state["filter_3d"] = torch.cat((filter_3d, child_filter), dim=0)
    state["output_source_idx"] = torch.cat(
        (state["output_source_idx"], state["output_source_idx"][selected_indices].repeat_interleave(split_counts)),
        dim=0,
    )
    state["tracking_state"] = {
        "source_tag": torch.cat((tracking_state["source_tag"], child_tracking["source_tag"]), dim=0),
        "seed_id": torch.cat((tracking_state["seed_id"], child_tracking["seed_id"]), dim=0),
        "generation": torch.cat((tracking_state["generation"], child_tracking["generation"]), dim=0),
        "edge_touched": torch.cat((tracking_state["edge_touched"], child_tracking["edge_touched"]), dim=0),
        "edge_touch_iter": torch.cat((tracking_state["edge_touch_iter"], child_tracking["edge_touch_iter"]), dim=0),
    }
    _extend_bool_masks(state, child_count)
    state["is_child_output_mask"][-child_count:] = True
    state[child_key][-child_count:] = True

    summary.update(
        {
            "child_count": child_count,
            "output_count_after": int(state["xyz"].shape[0]),
            "split_count": int(actual_split_count),
            "split_count_stats": _scalar_stats(split_counts),
            "parent_tau_keep": float(parent_tau_keep),
            "child_tau_ratio_requested": float(config.child_tau_ratio),
            "child_tau_ratio_effective": float(effective_child_tau_ratio),
            "mass_cap_eps": float(mass_cap_eps),
            "mesh_pull_lambda": float(config.mesh_pull_lambda),
            "mesh_pull_count": int(mesh_pull_count),
            "candidate_score_stats": _scalar_stats(candidate_scores),
            "selected_score_stats": _scalar_stats(selected_scores),
            "selected_parent_tau_stats": _scalar_stats(parent_tau),
            "selected_effective_child_ratio_stats": _scalar_stats(effective_ratio_tensor),
            "selected_source_idx": _as_int64_tensor(selected_source_idx),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply mask-guided regulatedGS reparameterization with independent geometry/highlight modules."
    )
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--save_intermediate_models", action="store_true")

    parser.add_argument("--geometry_payload_path", default="")
    parser.add_argument("--geometry_mask_key", default="geometry_candidate_mask")
    parser.add_argument("--geometry_score_key", default="candidate_score")
    parser.add_argument("--geometry_nearest_surface_key", default="nearest_surface_index")
    parser.add_argument("--geometry_mesh_path", default="")
    parser.add_argument("--geometry_apply_to_children", action="store_true")
    parser.add_argument("--geometry_max_fraction", type=float, default=0.015)
    parser.add_argument("--geometry_max_count", type=int, default=24000)
    parser.add_argument("--geometry_split_count", type=int, default=4)
    parser.add_argument("--geometry_max_split_count", type=int, default=10)
    parser.add_argument("--geometry_child_layout", choices=["major_axis", "grid", "major_axis_adaptive_chunk"], default="grid")
    parser.add_argument("--geometry_chunk_aspect_target", type=float, default=1.8)
    parser.add_argument("--geometry_offset_scale", type=float, default=0.55)
    parser.add_argument("--geometry_parent_tau_keep", type=float, default=0.85)
    parser.add_argument("--geometry_child_tau_ratio", type=float, default=0.35)
    parser.add_argument("--geometry_mass_cap_eps", type=float, default=0.10)
    parser.add_argument("--geometry_parent_dc_scale", type=float, default=1.0)
    parser.add_argument("--geometry_parent_rest_scale", type=float, default=1.0)
    parser.add_argument("--geometry_child_major_scale_multiplier", type=float, default=0.55)
    parser.add_argument("--geometry_child_minor_scale_multiplier", type=float, default=0.72)
    parser.add_argument("--geometry_child_normal_scale_multiplier", type=float, default=0.72)
    parser.add_argument("--geometry_child_dc_scale", type=float, default=1.0)
    parser.add_argument("--geometry_child_rest_scale", type=float, default=0.0)
    parser.add_argument("--geometry_child_filter_scale", type=float, default=0.35)
    parser.add_argument("--geometry_filter_cap_ratio", type=float, default=0.0015)
    parser.add_argument("--geometry_energy_conserve_mode", choices=["none", "area", "volume"], default="area")
    parser.add_argument("--geometry_mesh_pull_lambda", type=float, default=0.15)

    parser.add_argument("--highlight_payload_path", default="")
    parser.add_argument("--highlight_mask_key", default="brightness_mask")
    parser.add_argument("--highlight_score_key", default="dc_luma")
    parser.add_argument("--highlight_exclude_selected_key", default="")
    parser.add_argument("--highlight_apply_to_children", action="store_true")
    parser.add_argument("--highlight_max_fraction", type=float, default=0.020)
    parser.add_argument("--highlight_max_count", type=int, default=32000)
    parser.add_argument("--highlight_split_count", type=int, default=2)
    parser.add_argument("--highlight_max_split_count", type=int, default=2)
    parser.add_argument("--highlight_child_layout", choices=["major_axis", "grid", "major_axis_adaptive_chunk"], default="major_axis")
    parser.add_argument("--highlight_chunk_aspect_target", type=float, default=1.8)
    parser.add_argument("--highlight_offset_scale", type=float, default=0.28)
    parser.add_argument("--highlight_parent_tau_keep", type=float, default=0.95)
    parser.add_argument("--highlight_child_tau_ratio", type=float, default=0.20)
    parser.add_argument("--highlight_mass_cap_eps", type=float, default=0.08)
    parser.add_argument("--highlight_parent_dc_scale", type=float, default=1.0)
    parser.add_argument("--highlight_parent_rest_scale", type=float, default=0.50)
    parser.add_argument("--highlight_child_major_scale_multiplier", type=float, default=0.80)
    parser.add_argument("--highlight_child_minor_scale_multiplier", type=float, default=0.88)
    parser.add_argument("--highlight_child_normal_scale_multiplier", type=float, default=0.92)
    parser.add_argument("--highlight_child_dc_scale", type=float, default=1.0)
    parser.add_argument("--highlight_child_rest_scale", type=float, default=0.0)
    parser.add_argument("--highlight_child_filter_scale", type=float, default=0.45)
    parser.add_argument("--highlight_filter_cap_ratio", type=float, default=0.0015)
    parser.add_argument("--highlight_energy_conserve_mode", choices=["none", "area", "volume"], default="area")

    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_model_iteration(model_path, int(args.iteration))
    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    _ensure_tracking_state(gaussians)

    initial_count = int(gaussians.get_xyz.shape[0])
    geometry_payload = _load_module_payload(
        payload_path=str(args.geometry_payload_path),
        mask_key=str(args.geometry_mask_key),
        score_key=str(args.geometry_score_key),
        nearest_surface_key=str(args.geometry_nearest_surface_key),
        expected_count=initial_count,
    )
    highlight_payload = _load_module_payload(
        payload_path=str(args.highlight_payload_path),
        mask_key=str(args.highlight_mask_key),
        score_key=str(args.highlight_score_key),
        nearest_surface_key="",
        expected_count=initial_count,
    )
    geometry_mesh_vertices = _load_mesh_vertices(str(args.geometry_mesh_path))

    state = _build_initial_state(gaussians)
    geometry_cfg = ModuleConfig(
        name="geometry",
        enabled=geometry_payload is not None,
        payload=geometry_payload,
        apply_to_children=bool(args.geometry_apply_to_children),
        exclude_selected_key="",
        max_fraction=float(args.geometry_max_fraction),
        max_count=int(args.geometry_max_count),
        split_count=int(args.geometry_split_count),
        max_split_count=int(args.geometry_max_split_count),
        child_layout=str(args.geometry_child_layout),
        chunk_aspect_target=float(args.geometry_chunk_aspect_target),
        offset_scale=float(args.geometry_offset_scale),
        parent_tau_keep=float(args.geometry_parent_tau_keep),
        child_tau_ratio=float(args.geometry_child_tau_ratio),
        mass_cap_eps=float(args.geometry_mass_cap_eps),
        parent_dc_scale=float(args.geometry_parent_dc_scale),
        parent_rest_scale=float(args.geometry_parent_rest_scale),
        child_major_scale_multiplier=float(args.geometry_child_major_scale_multiplier),
        child_minor_scale_multiplier=float(args.geometry_child_minor_scale_multiplier),
        child_normal_scale_multiplier=float(args.geometry_child_normal_scale_multiplier),
        child_dc_scale=float(args.geometry_child_dc_scale),
        child_rest_scale=float(args.geometry_child_rest_scale),
        child_filter_scale=float(args.geometry_child_filter_scale),
        filter_cap_ratio=float(args.geometry_filter_cap_ratio),
        energy_conserve_mode=str(args.geometry_energy_conserve_mode),
        mesh_pull_lambda=float(args.geometry_mesh_pull_lambda),
    )
    highlight_cfg = ModuleConfig(
        name="highlight",
        enabled=highlight_payload is not None,
        payload=highlight_payload,
        apply_to_children=bool(args.highlight_apply_to_children),
        exclude_selected_key=str(args.highlight_exclude_selected_key),
        max_fraction=float(args.highlight_max_fraction),
        max_count=int(args.highlight_max_count),
        split_count=int(args.highlight_split_count),
        max_split_count=int(args.highlight_max_split_count),
        child_layout=str(args.highlight_child_layout),
        chunk_aspect_target=float(args.highlight_chunk_aspect_target),
        offset_scale=float(args.highlight_offset_scale),
        parent_tau_keep=float(args.highlight_parent_tau_keep),
        child_tau_ratio=float(args.highlight_child_tau_ratio),
        mass_cap_eps=float(args.highlight_mass_cap_eps),
        parent_dc_scale=float(args.highlight_parent_dc_scale),
        parent_rest_scale=float(args.highlight_parent_rest_scale),
        child_major_scale_multiplier=float(args.highlight_child_major_scale_multiplier),
        child_minor_scale_multiplier=float(args.highlight_child_minor_scale_multiplier),
        child_normal_scale_multiplier=float(args.highlight_child_normal_scale_multiplier),
        child_dc_scale=float(args.highlight_child_dc_scale),
        child_rest_scale=float(args.highlight_child_rest_scale),
        child_filter_scale=float(args.highlight_child_filter_scale),
        filter_cap_ratio=float(args.highlight_filter_cap_ratio),
        energy_conserve_mode=str(args.highlight_energy_conserve_mode),
        mesh_pull_lambda=0.0,
    )

    geometry_summary = _apply_module(state=state, config=geometry_cfg, mesh_vertices=geometry_mesh_vertices)
    if bool(args.save_intermediate_models) and bool(geometry_summary.get("selected_count", 0)):
        _save_model_snapshot(
            output_root=output_model_path / "intermediate_geometry_module",
            loaded_iter=loaded_iter,
            source_model_path=model_path,
            base_model=gaussians,
            state=state,
        )

    highlight_summary = _apply_module(state=state, config=highlight_cfg, mesh_vertices=None)
    if bool(args.save_intermediate_models) and bool(highlight_summary.get("selected_count", 0)):
        _save_model_snapshot(
            output_root=output_model_path / "intermediate_highlight_module",
            loaded_iter=loaded_iter,
            source_model_path=model_path,
            base_model=gaussians,
            state=state,
        )

    _save_model_snapshot(
        output_root=output_model_path,
        loaded_iter=loaded_iter,
        source_model_path=model_path,
        base_model=gaussians,
        state=state,
    )

    masks_dir = output_model_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state["output_source_idx"].detach().cpu().to(torch.int64), masks_dir / "output_source_idx.pt")
    torch.save(state["is_child_output_mask"].detach().cpu().to(torch.bool), masks_dir / "is_child_output_mask.pt")
    torch.save(state["geometry_selected_output_mask"].detach().cpu().to(torch.bool), masks_dir / "geometry_selected_output_mask.pt")
    torch.save(state["geometry_child_output_mask"].detach().cpu().to(torch.bool), masks_dir / "geometry_child_output_mask.pt")
    torch.save(state["highlight_selected_output_mask"].detach().cpu().to(torch.bool), masks_dir / "highlight_selected_output_mask.pt")
    torch.save(state["highlight_child_output_mask"].detach().cpu().to(torch.bool), masks_dir / "highlight_child_output_mask.pt")

    geometry_selected_source_idx = _as_int64_tensor(
        geometry_summary.pop("selected_source_idx", torch.empty((0,), dtype=torch.int64))
    )
    highlight_selected_source_idx = _as_int64_tensor(
        highlight_summary.pop("selected_source_idx", torch.empty((0,), dtype=torch.int64))
    )
    torch.save(geometry_selected_source_idx, masks_dir / "geometry_selected_source_idx.pt")
    torch.save(highlight_selected_source_idx, masks_dir / "highlight_selected_source_idx.pt")

    summary = {
        "version": "mask_guided_reparameterization_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "output_model_path": str(output_model_path),
        "iteration": int(loaded_iter),
        "input_gaussians": int(initial_count),
        "output_gaussians": int(state["xyz"].shape[0]),
        "save_intermediate_models": bool(args.save_intermediate_models),
        "geometry_module": geometry_summary,
        "highlight_module": highlight_summary,
        "args": vars(args),
    }
    (output_model_path / "mask_guided_reparameterization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
