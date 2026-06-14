from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from train_meshgs_prior_v0 import load_carrier_payload_arrays
from utils.general_utils import inverse_sigmoid
from utils.route_executor import (
    RouteExecutionConfig,
    apply_route_suppression_to_detail_model,
    build_execution_payload_from_route,
    load_route_payload,
    vector_stats,
)
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p, searchForMaxIteration


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return int(iteration)
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")
    return int(searchForMaxIteration(str(point_cloud_root)))


def _resolve_input_ply(model_path: Path, iteration: int) -> Path:
    point_dir = model_path / "point_cloud" / f"iteration_{int(iteration)}"
    ply_path = point_dir / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"point cloud not found: {ply_path}")
    return ply_path


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "config.json", "cameras.json"]:
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=False,
        alpha_mask=False,
        init_type="sfm",
    )


def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def _quaternion_from_rotation_matrix_np(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64, copy=False)
    q = np.empty((m.shape[0], 4), dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    positive = trace > 0.0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        q[positive, 0] = 0.25 * s
        q[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
        q[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
        q[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s

    negative = ~positive
    if np.any(negative):
        idx = np.where(negative)[0]
        mn = m[idx]
        choice = np.argmax(np.stack([mn[:, 0, 0], mn[:, 1, 1], mn[:, 2, 2]], axis=1), axis=1)
        for axis in range(3):
            local = idx[choice == axis]
            if local.size == 0:
                continue
            ml = m[local]
            if axis == 0:
                s = np.sqrt(1.0 + ml[:, 0, 0] - ml[:, 1, 1] - ml[:, 2, 2]) * 2.0
                q[local, 0] = (ml[:, 2, 1] - ml[:, 1, 2]) / s
                q[local, 1] = 0.25 * s
                q[local, 2] = (ml[:, 0, 1] + ml[:, 1, 0]) / s
                q[local, 3] = (ml[:, 0, 2] + ml[:, 2, 0]) / s
            elif axis == 1:
                s = np.sqrt(1.0 + ml[:, 1, 1] - ml[:, 0, 0] - ml[:, 2, 2]) * 2.0
                q[local, 0] = (ml[:, 0, 2] - ml[:, 2, 0]) / s
                q[local, 1] = (ml[:, 0, 1] + ml[:, 1, 0]) / s
                q[local, 2] = 0.25 * s
                q[local, 3] = (ml[:, 1, 2] + ml[:, 2, 1]) / s
            else:
                s = np.sqrt(1.0 + ml[:, 2, 2] - ml[:, 0, 0] - ml[:, 1, 1]) * 2.0
                q[local, 0] = (ml[:, 1, 0] - ml[:, 0, 1]) / s
                q[local, 1] = (ml[:, 0, 2] + ml[:, 2, 0]) / s
                q[local, 2] = (ml[:, 1, 2] + ml[:, 2, 1]) / s
                q[local, 3] = 0.25 * s

    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def _load_action_payload(path: Path) -> Dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(path)
        raw: Dict[str, Any] = {key: loaded[key] for key in loaded.files}
    else:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported action payload object type: {type(raw)!r}")
    if "update_strength" in raw:
        payload = raw
    elif "gs_action_payload" in raw and isinstance(raw["gs_action_payload"], dict):
        payload = raw["gs_action_payload"]
    elif "hrgs_outputs" in raw and isinstance(raw["hrgs_outputs"], dict) and isinstance(raw["hrgs_outputs"].get("gs_action_payload"), dict):
        payload = raw["hrgs_outputs"]["gs_action_payload"]
    else:
        raise KeyError("Action payload missing 'update_strength' or nested 'gs_action_payload'.")

    out: Dict[str, torch.Tensor] = {}
    for key in ["update_strength", "attach_strength", "detail_weight", "prior_color_strength"]:
        value = payload.get(key)
        if value is None:
            raise KeyError(f"Missing action payload key: {key}")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        out[key] = value.detach().cpu().float().reshape(-1, 1)
    return out


def _build_surface_gaussians(
    payload_path: Path,
    sh_degree: int,
    min_confidence: float,
    max_disagreement: float,
    min_views: int,
    max_count: int,
    seed: int,
    scale_multiplier: float,
    thickness_multiplier: float,
    min_scale: float,
    init_opacity: float,
) -> tuple[GaussianModel, np.ndarray, Dict[str, float]]:
    payload = load_carrier_payload_arrays(str(payload_path))
    valid = payload["valid_mask"].astype(bool)
    valid &= payload["confidence"] >= float(min_confidence)
    valid &= payload["disagreement"] <= float(max_disagreement)
    valid &= payload["view_count"] >= int(min_views)
    indices = np.flatnonzero(valid)
    if max_count > 0 and indices.size > int(max_count):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=int(max_count), replace=False))
    if indices.size == 0:
        raise RuntimeError("No valid carriers remained after filtering.")

    centers = payload["centers"][indices].astype(np.float32)
    colors = np.clip(payload["fused_rgb"][indices].astype(np.float32), 0.0, 1.0)
    normals = _normalize_np(payload["normals"][indices].astype(np.float32))
    tangent_u = _normalize_np(payload["tangent_u"][indices].astype(np.float32))
    tangent_v = _normalize_np(np.cross(normals, tangent_u))
    tangent_u = _normalize_np(np.cross(tangent_v, normals))

    rotation_matrix = np.stack([tangent_u, tangent_v, normals], axis=2)
    rotations = _quaternion_from_rotation_matrix_np(rotation_matrix)

    scales = np.stack(
        [
            payload["scale_u"][indices],
            payload["scale_v"][indices],
            payload["scale_n"][indices],
        ],
        axis=1,
    ).astype(np.float32)
    scales *= float(scale_multiplier)
    scales[:, 2] *= float(thickness_multiplier)
    scales = np.clip(scales, float(min_scale), None)

    confidence = np.clip(payload["confidence"][indices].astype(np.float32), 0.0, 1.0)
    disagreement = payload["disagreement"][indices].astype(np.float32)
    if max_disagreement > 0:
        disagreement_gate = 1.0 - np.clip(disagreement / float(max_disagreement), 0.0, 1.0)
    else:
        disagreement_gate = np.ones_like(confidence)
    opacity = np.clip(float(init_opacity) * (0.25 + 0.75 * confidence * disagreement_gate), 1e-4, 0.95)

    surface = GaussianModel(sh_degree, use_SBs=False)
    fused_color = RGB2SH(torch.from_numpy(colors).float().cuda())
    features = torch.zeros((indices.size, 3, (sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
    features[:, :3, 0] = fused_color

    surface.spatial_lr_scale = 1.0
    surface._xyz = nn.Parameter(torch.from_numpy(centers).float().cuda().requires_grad_(False))
    surface._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False))
    surface._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))
    surface._opacity = nn.Parameter(inverse_sigmoid(torch.from_numpy(opacity[:, None]).float().cuda()).requires_grad_(False))
    surface._scaling = nn.Parameter(torch.log(torch.from_numpy(scales).float().cuda()).requires_grad_(False))
    surface._rotation = nn.Parameter(torch.from_numpy(rotations).float().cuda().requires_grad_(False))
    surface.filter_3D = torch.zeros((indices.size, 1), dtype=torch.float32, device="cuda")
    surface.max_radii2D = torch.zeros((indices.size,), dtype=torch.float32, device="cuda")
    surface.init_tracking_state(indices.size, source_tag=int(GaussianSourceTag.PRIOR_INJECTED))
    surface.active_sh_degree = 0

    summary = {
        "payload_count": int(payload["valid_mask"].shape[0]),
        "selected_surface_count": int(indices.size),
        "selected_surface_ratio": float(indices.size / max(payload["valid_mask"].shape[0], 1)),
        "mean_surface_confidence": float(confidence.mean()) if confidence.size else 0.0,
    }
    return surface, confidence, summary


def _concat_tracking(detail: GaussianModel, surface: GaussianModel) -> Dict[str, torch.Tensor]:
    return {
        "source_tag": torch.cat((surface._source_tag.detach(), detail._source_tag.detach()), dim=0),
        "seed_id": torch.cat((surface._seed_id.detach(), detail._seed_id.detach()), dim=0),
        "generation": torch.cat((surface._generation.detach(), detail._generation.detach()), dim=0),
        "edge_touched": torch.cat((surface._edge_touched.detach(), detail._edge_touched.detach()), dim=0),
        "edge_touch_iter": torch.cat((surface._edge_touch_iter.detach(), detail._edge_touch_iter.detach()), dim=0),
    }


def _merge_gaussian_models(surface: GaussianModel, detail: GaussianModel) -> GaussianModel:
    merged = GaussianModel(detail.max_sh_degree, use_SBs=False)
    merged.active_sh_degree = max(int(surface.active_sh_degree), int(detail.active_sh_degree))
    merged.spatial_lr_scale = float(detail.spatial_lr_scale)

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
    merged.max_radii2D = torch.zeros((merged._xyz.shape[0],), dtype=torch.float32, device="cuda")
    merged.init_tracking_state(int(merged._xyz.shape[0]))
    tracking = _concat_tracking(detail, surface)
    merged._source_tag = tracking["source_tag"]
    merged._seed_id = tracking["seed_id"]
    merged._generation = tracking["generation"]
    merged._edge_touched = tracking["edge_touched"]
    merged._edge_touch_iter = tracking["edge_touch_iter"]
    return merged


def _save_merged_action_payload(
    action_payload_path: Path | None,
    output_path: Path,
    surface_confidence: np.ndarray,
    detail_count: int,
    surface_count: int,
    detail_attach_scale: float,
    detail_update_scale: float,
    detail_detail_scale: float,
    detail_prior_color_scale: float,
    radius_gate: torch.Tensor | None,
    detail_route_payload: Dict[str, torch.Tensor] | None,
    detail_execution_payload: Dict[str, torch.Tensor] | None,
    execution_config: RouteExecutionConfig | None,
) -> tuple[Path | None, Dict[str, Any]]:
    if action_payload_path is None and detail_route_payload is None and detail_execution_payload is None:
        return None, {"enabled": False}
    if detail_execution_payload is not None:
        detail_payload = {
            key: value.detach().cpu().float().reshape(-1, 1)
            for key, value in detail_execution_payload.items()
            if torch.is_tensor(value) and value.ndim >= 1
        }
    elif detail_route_payload is not None:
        detail_payload = {
            key: value.detach().cpu().float().reshape(-1, 1)
            for key, value in detail_route_payload.items()
            if torch.is_tensor(value) and value.ndim >= 1
        }
    else:
        detail_payload = _load_action_payload(action_payload_path)
    if int(detail_payload["update_strength"].shape[0]) != int(detail_count):
        raise ValueError(
            f"Detail payload length mismatch: {detail_payload['update_strength'].shape[0]} "
            f"vs detail_count={detail_count}"
        )

    surface_conf = torch.from_numpy(np.clip(surface_confidence, 0.0, 1.0)).float().reshape(-1, 1)
    gate = radius_gate if radius_gate is not None else torch.ones((detail_count, 1), dtype=torch.float32)
    gate = gate.float().reshape(-1, 1)
    detail_payload["update_strength"] = torch.clamp(
        detail_payload["update_strength"] * float(detail_update_scale) * gate,
        min=0.0,
        max=1.0,
    )
    detail_payload["attach_strength"] = torch.clamp(
        detail_payload["attach_strength"] * float(detail_attach_scale) * gate,
        min=0.0,
        max=1.0,
    )
    detail_payload["detail_weight"] = torch.clamp(
        detail_payload["detail_weight"] * float(detail_detail_scale),
        min=0.0,
        max=1.0,
    )
    detail_payload["prior_color_strength"] = torch.clamp(
        detail_payload["prior_color_strength"] * float(detail_prior_color_scale) * gate,
        min=0.0,
        max=1.0,
    )
    if "suppress_strength" in detail_payload:
        detail_payload["suppress_strength"] = torch.clamp(detail_payload["suppress_strength"], min=0.0, max=1.0)
    if "geometry_update_strength" in detail_payload:
        detail_payload["geometry_update_strength"] = torch.clamp(
            detail_payload["geometry_update_strength"] * float(detail_update_scale) * gate,
            min=0.0,
            max=1.0,
        )
    if "color_update_strength" in detail_payload:
        detail_payload["color_update_strength"] = torch.clamp(
            detail_payload["color_update_strength"] * float(detail_detail_scale),
            min=0.0,
            max=1.0,
        )
    if "execution_attach_weight" in detail_payload:
        detail_payload["execution_attach_weight"] = torch.clamp(
            detail_payload["execution_attach_weight"] * float(detail_attach_scale) * gate,
            min=0.0,
            max=1.0,
        )
    if "execution_detail_weight" in detail_payload:
        detail_payload["execution_detail_weight"] = torch.clamp(
            detail_payload["execution_detail_weight"] * float(detail_detail_scale),
            min=0.0,
            max=1.0,
        )
    if "execution_prior_color_strength" in detail_payload:
        detail_payload["execution_prior_color_strength"] = torch.clamp(
            detail_payload["execution_prior_color_strength"] * float(detail_prior_color_scale) * gate,
            min=0.0,
            max=1.0,
        )
    for key in ("execution_position_weight", "execution_scaling_weight", "execution_rotation_weight"):
        if key in detail_payload:
            detail_payload[key] = torch.clamp(
                detail_payload[key] * float(detail_update_scale) * gate,
                min=0.0,
                max=1.0,
            )
    if "execution_appearance_weight" in detail_payload:
        detail_payload["execution_appearance_weight"] = torch.clamp(
            detail_payload["execution_appearance_weight"] * float(detail_detail_scale),
            min=0.0,
            max=1.0,
        )
    if "execution_suppress_strength" in detail_payload:
        detail_payload["execution_suppress_strength"] = torch.clamp(
            detail_payload["execution_suppress_strength"],
            min=0.0,
            max=1.0,
        )
    if "execution_opacity_weight" in detail_payload:
        detail_payload["execution_opacity_weight"] = torch.clamp(
            detail_payload["execution_opacity_weight"],
            min=0.0,
            max=1.0,
        )
    if "execution_appearance_weight" in detail_payload or "execution_position_weight" in detail_payload:
        position = detail_payload.get("execution_position_weight", detail_payload.get("geometry_update_strength"))
        scaling = detail_payload.get("execution_scaling_weight", position)
        rotation = detail_payload.get("execution_rotation_weight", position)
        appearance = detail_payload.get("execution_appearance_weight", detail_payload.get("color_update_strength"))
        opacity = detail_payload.get("execution_opacity_weight", detail_payload.get("update_strength"))
        if position is not None:
            detail_payload["execution_position_weight"] = torch.clamp(position, min=0.0, max=1.0)
            detail_payload["geometry_update_strength"] = detail_payload["execution_position_weight"].clone()
        if scaling is not None:
            detail_payload["execution_scaling_weight"] = torch.clamp(scaling, min=0.0, max=1.0)
        if rotation is not None:
            detail_payload["execution_rotation_weight"] = torch.clamp(rotation, min=0.0, max=1.0)
        if appearance is not None:
            detail_payload["execution_appearance_weight"] = torch.clamp(appearance, min=0.0, max=1.0)
            detail_payload["color_update_strength"] = detail_payload["execution_appearance_weight"].clone()
        if opacity is not None:
            detail_payload["execution_opacity_weight"] = torch.clamp(opacity, min=0.0, max=1.0)
        if position is not None or appearance is not None or opacity is not None:
            base_update = torch.maximum(
                detail_payload.get("execution_position_weight", torch.zeros_like(detail_payload["update_strength"])),
                detail_payload.get("execution_appearance_weight", torch.zeros_like(detail_payload["update_strength"])),
            )
            if "execution_opacity_weight" in detail_payload:
                base_update = torch.maximum(base_update, detail_payload["execution_opacity_weight"])
            detail_payload["execution_update_strength"] = torch.clamp(base_update, min=0.0, max=1.0)
            detail_payload["update_strength"] = detail_payload["execution_update_strength"].clone()
    detail_action_summary = {
        "enabled": True,
        "update_strength_stats": _scalar_stats(detail_payload["update_strength"]),
        "attach_strength_stats": _scalar_stats(detail_payload["attach_strength"]),
        "detail_weight_stats": _scalar_stats(detail_payload["detail_weight"]),
        "prior_color_strength_stats": _scalar_stats(detail_payload["prior_color_strength"]),
    }
    if execution_config is not None:
        detail_action_summary["execution_config"] = dict(vars(execution_config))
    for key in (
        "execution_attach_weight",
        "execution_detail_weight",
        "execution_prior_color_strength",
        "execution_suppress_strength",
        "execution_position_weight",
        "execution_scaling_weight",
        "execution_rotation_weight",
        "execution_appearance_weight",
        "execution_opacity_weight",
        "execution_update_strength",
        "clean_detail_strength",
        "risky_useful_strength",
        "harmful_outlier_strength",
    ):
        value = detail_payload.get(key)
        if value is not None:
            detail_action_summary[f"{key}_stats"] = _scalar_stats(value)
    surface_payload = {
        "update_strength": surface_conf.clone(),
        "attach_strength": torch.ones((surface_count, 1), dtype=torch.float32),
        "detail_weight": torch.zeros((surface_count, 1), dtype=torch.float32),
        "prior_color_strength": surface_conf.clone(),
    }
    if detail_route_payload is not None:
        surface_payload.update(
            {
                "radius_gate": torch.ones((surface_count, 1), dtype=torch.float32),
                "suppress_strength": torch.zeros((surface_count, 1), dtype=torch.float32),
                "geometry_update_strength": surface_conf.clone(),
                "color_update_strength": surface_conf.clone(),
                "p_attach": torch.ones((surface_count, 1), dtype=torch.float32),
                "p_detail": torch.zeros((surface_count, 1), dtype=torch.float32),
                "p_suppress": torch.zeros((surface_count, 1), dtype=torch.float32),
                "p_promote": torch.zeros((surface_count, 1), dtype=torch.float32),
                "p_spawn_source": torch.zeros((surface_count, 1), dtype=torch.float32),
            }
        )
    if detail_execution_payload is not None:
        surface_payload.update(
            {
                "execution_update_strength": surface_conf.clone(),
                "execution_attach_weight": torch.ones((surface_count, 1), dtype=torch.float32),
                "execution_detail_weight": torch.zeros((surface_count, 1), dtype=torch.float32),
                "execution_prior_color_strength": surface_conf.clone(),
                "execution_suppress_strength": torch.zeros((surface_count, 1), dtype=torch.float32),
                "execution_position_weight": surface_conf.clone(),
                "execution_scaling_weight": surface_conf.clone(),
                "execution_rotation_weight": surface_conf.clone(),
                "execution_appearance_weight": surface_conf.clone(),
                "execution_opacity_weight": surface_conf.clone(),
            }
        )
    merged = {
        key: torch.cat((surface_payload[key], detail_payload[key]), dim=0)
        for key in surface_payload.keys()
    }
    if detail_route_payload is not None or detail_execution_payload is not None:
        for key in (
            "radius_gate",
            "suppress_strength",
            "geometry_update_strength",
            "color_update_strength",
            "p_attach",
            "p_detail",
            "p_suppress",
            "p_promote",
            "p_spawn_source",
            "execution_update_strength",
            "execution_attach_weight",
            "execution_detail_weight",
            "execution_prior_color_strength",
            "execution_suppress_strength",
            "execution_position_weight",
            "execution_scaling_weight",
            "execution_rotation_weight",
            "execution_appearance_weight",
            "execution_opacity_weight",
            "clean_detail_strength",
            "risky_useful_strength",
            "harmful_outlier_strength",
        ):
            if key not in detail_payload:
                continue
            if key not in surface_payload:
                fill_value = 0.0
                if key in {
                    "radius_gate",
                    "execution_attach_weight",
                    "execution_position_weight",
                    "execution_scaling_weight",
                    "execution_rotation_weight",
                    "execution_appearance_weight",
                    "execution_opacity_weight",
                    "execution_update_strength",
                }:
                    fill_value = 1.0
                surface_payload[key] = torch.full((surface_count, 1), float(fill_value), dtype=torch.float32)
            merged[key] = torch.cat((surface_payload[key], detail_payload[key]), dim=0)
    if execution_config is not None:
        merged["meta"] = {"route_execution_config": dict(vars(execution_config))}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    return output_path, detail_action_summary


def _scalar_stats(values: torch.Tensor) -> Dict[str, float | None]:
    if values.numel() == 0:
        return {"min": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    flat = values.detach().reshape(-1).float()
    return {
        "min": float(flat.min().item()),
        "median": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "max": float(flat.max().item()),
    }


def _estimate_detail_projected_radius_proxy(
    detail: GaussianModel,
    scene_root: Path,
    model_path: Path,
    images_subdir: str,
    max_views: int,
) -> torch.Tensor:
    dataset_args = _build_dataset_args(str(scene_root), str(model_path), images_subdir)
    dataset = ModelParams(None).extract(dataset_args)
    scene = Scene(
        dataset,
        GaussianModel(dataset.sh_degree, use_SBs=False),
        load_iteration=-1,
        shuffle=False,
        skip_train=False,
        skip_test=True,
    )
    cameras = scene.getTrainCameras().copy()
    if int(max_views) > 0 and len(cameras) > int(max_views):
        ids = np.linspace(0, len(cameras) - 1, num=int(max_views), dtype=np.int64)
        ids = np.unique(ids)
        cameras = [cameras[int(i)] for i in ids.tolist()]

    xyz = detail.get_xyz.detach()
    scaling = detail.get_scaling.detach()
    max_scale = scaling.max(dim=1).values
    radius_proxy = torch.zeros((xyz.shape[0],), device=xyz.device, dtype=torch.float32)
    for camera in cameras:
        R = torch.as_tensor(camera.R, device=xyz.device, dtype=xyz.dtype)
        T = torch.as_tensor(camera.T, device=xyz.device, dtype=xyz.dtype)
        xyz_cam = xyz @ R + T[None, :]
        z = xyz_cam[:, 2]
        valid = z > 1e-6
        if not torch.any(valid):
            continue
        focal = max(float(camera.focal_x), float(camera.focal_y))
        local = torch.zeros_like(radius_proxy)
        local[valid] = max_scale[valid] * float(focal) / z[valid]
        radius_proxy = torch.maximum(radius_proxy, local.float())
    return radius_proxy


def _build_detail_radius_gate(
    detail: GaussianModel,
    scene_root: Path | None,
    model_path: Path,
    images_subdir: str,
    radius_ref_px: float,
    min_gate: float,
    max_views: int,
) -> tuple[torch.Tensor | None, Dict[str, float | None]]:
    if scene_root is None:
        return None, {"enabled": False}
    radius_proxy = _estimate_detail_projected_radius_proxy(
        detail=detail,
        scene_root=scene_root,
        model_path=model_path,
        images_subdir=images_subdir,
        max_views=max_views,
    )
    ref_px = max(float(radius_ref_px), 1e-6)
    gate = torch.clamp(ref_px / torch.clamp(radius_proxy, min=ref_px), min=float(min_gate), max=1.0).unsqueeze(1)
    summary = {
        "enabled": True,
        "radius_ref_px": float(radius_ref_px),
        "radius_min_gate": float(min_gate),
        "radius_proxy_stats": _scalar_stats(radius_proxy),
        "radius_gate_stats": _scalar_stats(gate),
    }
    return gate.detach().cpu(), summary


def main() -> None:
    parser = ArgumentParser(description="Merge HRGS surface carriers with a SOF-native mip detail field.")
    parser.add_argument("--detail_model_path", required=True)
    parser.add_argument("--carrier_payload", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--action_payload", default=None)
    parser.add_argument("--route_payload", default=None)
    parser.add_argument("--detail_iteration", type=int, default=-1)
    parser.add_argument("--output_iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--surface_min_confidence", type=float, default=0.05)
    parser.add_argument("--surface_max_disagreement", type=float, default=0.08)
    parser.add_argument("--surface_min_views", type=int, default=2)
    parser.add_argument("--surface_max_count", type=int, default=0)
    parser.add_argument("--surface_seed", type=int, default=0)
    parser.add_argument("--surface_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--surface_thickness_multiplier", type=float, default=0.5)
    parser.add_argument("--surface_min_scale", type=float, default=1e-5)
    parser.add_argument("--surface_init_opacity", type=float, default=0.35)
    parser.add_argument("--detail_attach_scale", type=float, default=0.1)
    parser.add_argument("--detail_update_scale", type=float, default=1.0)
    parser.add_argument("--detail_detail_scale", type=float, default=1.0)
    parser.add_argument("--detail_prior_color_scale", type=float, default=1.0)
    parser.add_argument("--detail_radius_gate", action="store_true")
    parser.add_argument("--detail_radius_images_subdir", default="images_2")
    parser.add_argument("--detail_radius_ref_px", type=float, default=96.0)
    parser.add_argument("--detail_radius_min_gate", type=float, default=0.1)
    parser.add_argument("--detail_radius_max_views", type=int, default=32)
    parser.add_argument("--route_execution_mode", default="detail_preserve_v0p8")
    parser.add_argument("--route_detail_protect_beta", type=float, default=0.7)
    parser.add_argument("--route_detail_attach_strength", type=float, default=0.0)
    parser.add_argument("--route_detail_geometry_strength", type=float, default=0.0)
    parser.add_argument("--route_prior_color_strength_scale", type=float, default=0.25)
    parser.add_argument("--route_suppress_update_floor", type=float, default=0.15)
    parser.add_argument("--route_suppress_opacity_scale", type=float, default=0.25)
    parser.add_argument("--route_min_scale_gate", type=float, default=0.3)
    parser.add_argument("--route_scale_gate_power", type=float, default=1.0)
    parser.add_argument("--route_scale_gate_suppress_coupling", type=float, default=1.0)
    parser.add_argument("--route_risky_useful_opacity_coupling", type=float, default=0.35)
    parser.add_argument("--route_risky_useful_scale_coupling", type=float, default=0.2)
    parser.add_argument("--route_harmful_scale_coupling", type=float, default=1.0)
    parser.add_argument("--render_sanity", action="store_true")
    parser.add_argument("--scene_root", default=None)
    parser.add_argument("--render_sanity_images_subdir", default="images_2")
    parser.add_argument("--render_sanity_resolution", type=int, default=1)
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    detail_model_path = Path(args.detail_model_path).expanduser().resolve()
    carrier_payload_path = Path(args.carrier_payload).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    action_payload_path = Path(args.action_payload).expanduser().resolve() if args.action_payload else None
    route_payload_path = Path(args.route_payload).expanduser().resolve() if args.route_payload else None
    scene_root = Path(args.scene_root).expanduser().resolve() if args.scene_root else None

    detail_iteration = _resolve_model_iteration(detail_model_path, int(args.detail_iteration))
    output_iteration = detail_iteration if int(args.output_iteration) < 0 else int(args.output_iteration)
    detail_ply = _resolve_input_ply(detail_model_path, detail_iteration)

    detail = GaussianModel(int(args.sh_degree), use_SBs=False)
    detail.load_ply(str(detail_ply))
    detail_tags_path = detail_model_path / "point_cloud" / f"iteration_{detail_iteration}" / "gaussian_tags.pt"
    detail.load_tracking_metadata(str(detail_tags_path))
    detail_route_payload = load_route_payload(
        route_payload_path,
        total_gaussians=int(detail.get_xyz.shape[0]),
    )
    execution_cfg = RouteExecutionConfig(
        mode=str(args.route_execution_mode),
        detail_protect_beta=float(args.route_detail_protect_beta),
        detail_attach_strength=float(args.route_detail_attach_strength),
        detail_geometry_strength=float(args.route_detail_geometry_strength),
        detail_prior_color_strength_scale=float(args.route_prior_color_strength_scale),
        suppress_update_floor=float(args.route_suppress_update_floor),
        suppress_opacity_scale=float(args.route_suppress_opacity_scale),
        min_scale_gate=float(args.route_min_scale_gate),
        scale_gate_power=float(args.route_scale_gate_power),
        scale_gate_suppress_coupling=float(args.route_scale_gate_suppress_coupling),
        risky_useful_opacity_coupling=float(args.route_risky_useful_opacity_coupling),
        risky_useful_scale_coupling=float(args.route_risky_useful_scale_coupling),
        harmful_scale_coupling=float(args.route_harmful_scale_coupling),
    )
    detail_execution_payload = (
        build_execution_payload_from_route(detail_route_payload, execution_cfg)
        if detail_route_payload is not None
        else None
    )
    route_summary: Dict[str, Any] = {"enabled": detail_route_payload is not None}
    if detail_route_payload is not None:
        route_summary = apply_route_suppression_to_detail_model(
            detail=detail,
            route_payload=detail_route_payload,
            suppress_opacity_scale=float(args.route_suppress_opacity_scale),
            min_scale_gate=float(args.route_min_scale_gate),
            scale_gate_power=float(args.route_scale_gate_power),
            mode=str(args.route_execution_mode),
            detail_protect_beta=float(args.route_detail_protect_beta),
            scale_gate_suppress_coupling=float(args.route_scale_gate_suppress_coupling),
            risky_useful_opacity_coupling=float(args.route_risky_useful_opacity_coupling),
            risky_useful_scale_coupling=float(args.route_risky_useful_scale_coupling),
            harmful_scale_coupling=float(args.route_harmful_scale_coupling),
        )
        route_summary["execution_config"] = dict(vars(execution_cfg))
        for key in ("p_attach", "p_detail", "p_suppress"):
            vector = detail_route_payload.get(key)
            if vector is not None:
                route_summary[f"{key}_stats"] = vector_stats(vector)
        if detail_execution_payload is not None:
            for key in (
                "execution_attach_weight",
                "execution_detail_weight",
                "execution_prior_color_strength",
                "execution_suppress_strength",
                "execution_position_weight",
                "execution_scaling_weight",
                "execution_rotation_weight",
                "execution_appearance_weight",
                "execution_opacity_weight",
                "execution_update_strength",
                "clean_detail_strength",
                "risky_useful_strength",
                "harmful_outlier_strength",
            ):
                vector = detail_execution_payload.get(key)
                if vector is not None:
                    route_summary[f"{key}_stats"] = vector_stats(vector)

    surface, surface_confidence, surface_summary = _build_surface_gaussians(
        payload_path=carrier_payload_path,
        sh_degree=int(args.sh_degree),
        min_confidence=float(args.surface_min_confidence),
        max_disagreement=float(args.surface_max_disagreement),
        min_views=int(args.surface_min_views),
        max_count=int(args.surface_max_count),
        seed=int(args.surface_seed),
        scale_multiplier=float(args.surface_scale_multiplier),
        thickness_multiplier=float(args.surface_thickness_multiplier),
        min_scale=float(args.surface_min_scale),
        init_opacity=float(args.surface_init_opacity),
    )
    merged = _merge_gaussian_models(surface, detail)

    _copy_render_config(detail_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(output_iteration)}"
    mkdir_p(str(point_dir))
    output_ply = point_dir / "point_cloud.ply"
    merged.save_ply(str(output_ply))
    merged.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    masks_dir = output_model_path / "masks"
    mkdir_p(str(masks_dir))
    surface_count = int(surface.get_xyz.shape[0])
    detail_count = int(detail.get_xyz.shape[0])
    total_count = int(merged.get_xyz.shape[0])
    surface_mask = torch.cat(
        (
            torch.ones((surface_count,), dtype=torch.bool),
            torch.zeros((detail_count,), dtype=torch.bool),
        ),
        dim=0,
    )
    detail_mask = ~surface_mask
    torch.save(surface_mask, masks_dir / "surface_mask.pt")
    torch.save(detail_mask, masks_dir / "detail_mask.pt")
    torch.save(torch.arange(surface_count, dtype=torch.int64), masks_dir / "surface_indices.pt")
    torch.save(torch.arange(detail_count, dtype=torch.int64) + surface_count, masks_dir / "detail_indices.pt")
    if detail_route_payload is not None:
        for key in ("p_attach", "p_detail", "p_suppress"):
            vector = detail_route_payload.get(key)
            if vector is None:
                continue
            route_mask = vector.detach().cpu().reshape(-1) >= 0.5
            torch.save(route_mask, masks_dir / f"{key}_mask.pt")

    radius_gate = None
    radius_gate_summary: Dict[str, Any] = {"enabled": False}
    if bool(args.detail_radius_gate):
        radius_gate, radius_gate_summary = _build_detail_radius_gate(
            detail=detail,
            scene_root=scene_root,
            model_path=detail_model_path,
            images_subdir=str(args.detail_radius_images_subdir),
            radius_ref_px=float(args.detail_radius_ref_px),
            min_gate=float(args.detail_radius_min_gate),
            max_views=int(args.detail_radius_max_views),
        )
        if radius_gate is not None:
            torch.save(radius_gate, masks_dir / "detail_radius_gate.pt")

    merged_action_path, detail_action_summary = _save_merged_action_payload(
        action_payload_path=action_payload_path,
        output_path=output_model_path / "merged_action_payload.pt",
        surface_confidence=surface_confidence,
        detail_count=detail_count,
        surface_count=surface_count,
        detail_attach_scale=float(args.detail_attach_scale),
        detail_update_scale=float(args.detail_update_scale),
        detail_detail_scale=float(args.detail_detail_scale),
        detail_prior_color_scale=float(args.detail_prior_color_scale),
        radius_gate=radius_gate,
        detail_route_payload=detail_route_payload,
        detail_execution_payload=detail_execution_payload,
        execution_config=execution_cfg if detail_route_payload is not None else None,
    )

    summary = {
        "mode": "prepare_hrgs_surface_detail_init",
        "detail_model_path": str(detail_model_path),
        "detail_iteration": int(detail_iteration),
        "detail_ply": str(detail_ply),
        "carrier_payload": str(carrier_payload_path),
        "action_payload": str(action_payload_path) if action_payload_path is not None else None,
        "route_payload": str(route_payload_path) if route_payload_path is not None else None,
        "output_model_path": str(output_model_path),
        "output_iteration": int(output_iteration),
        "output_ply": str(output_ply),
        "surface_count": surface_count,
        "detail_count": detail_count,
        "total_count": total_count,
        "merged_action_payload": str(merged_action_path) if merged_action_path is not None else None,
        "detail_action_summary": detail_action_summary,
        "detail_attach_scale": float(args.detail_attach_scale),
        "detail_update_scale": float(args.detail_update_scale),
        "detail_detail_scale": float(args.detail_detail_scale),
        "detail_prior_color_scale": float(args.detail_prior_color_scale),
        "route_execution_mode": str(args.route_execution_mode),
        "detail_radius_gate_summary": radius_gate_summary,
        "detail_route_summary": route_summary,
        **surface_summary,
    }
    meta_path = output_model_path / "surface_detail_init_summary.json"
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (point_dir / "num_gaussians.json").write_text(
        json.dumps(
            {
                "num_gaussians": total_count,
                "surface_count": surface_count,
                "detail_count": detail_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    render_sanity_pass = None
    render_sanity_dir = None
    if args.render_sanity:
        if scene_root is None:
            raise ValueError("--render_sanity requires --scene_root")
        cmd = [
            str(args.python_bin),
            str(REPO_ROOT / "render.py"),
            "-m",
            str(output_model_path),
            "-s",
            str(scene_root),
            "-i",
            str(args.render_sanity_images_subdir),
            "-r",
            str(int(args.render_sanity_resolution)),
            "--iteration",
            str(int(output_iteration)),
            "--eval",
            "--skip_train",
            "--data_device",
            "cpu",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        render_sanity_pass = True
        render_sanity_dir = str(output_model_path / "test" / f"ours_{int(output_iteration)}" / "renders")
        summary["render_sanity_pass"] = render_sanity_pass
        summary["render_sanity_dir"] = render_sanity_dir
        meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
