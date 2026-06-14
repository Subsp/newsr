from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from scipy.spatial import cKDTree

from arguments import ModelParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from train_meshgs_prior_v0 import load_carrier_payload_arrays
from utils.route_executor import load_route_payload, vector_stats


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str):
    class _Args:
        pass

    args = _Args()
    args.sh_degree = 3
    args.source_path = scene_root
    args.model_path = model_path
    args.images = images_subdir
    args.resolution = -1
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.alpha_mask = False
    args.init_type = "sfm"
    return args


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return int(iteration)

    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")

    candidates = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_root}")
    return int(max(candidates))


def _normalize_score(values: np.ndarray, q: float = 0.95, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    ref = np.quantile(values, q)
    ref = max(float(ref), float(eps))
    return np.clip(values / ref, 0.0, 1.0)


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _tensor_dict_to_numpy(payload: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        for key, value in payload.items()
        if torch.is_tensor(value)
    }


def _estimate_projected_radius_proxy(
    detail: GaussianModel,
    scene_root: Path,
    model_path: Path,
    images_subdir: str,
    max_views: int,
) -> np.ndarray:
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
    max_scale = detail.get_scaling.detach().max(dim=1).values
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
    return radius_proxy.detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass
class HeuristicGaussianRouteConfig:
    images_subdir: str = "images_2"
    max_views: int = 32
    radius_ref_px: float = 96.0
    radius_temperature_px: float = 24.0
    radius_gate_min: float = 0.1
    surface_confidence_floor: float = 0.05
    surface_distance_scale: float = 2.0
    opacity_center: float = 0.35
    opacity_temperature: float = 0.15
    suppress_update_floor: float = 0.15
    detail_boost: float = 1.0


def build_gaussian_route_features(
    detail: GaussianModel,
    carrier_payload_path: str | Path,
    *,
    scene_root: str | Path,
    model_path: str | Path,
    action_payload_path: str | Path | None = None,
    cfg: HeuristicGaussianRouteConfig | None = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or HeuristicGaussianRouteConfig()
    carrier_payload = load_carrier_payload_arrays(str(Path(carrier_payload_path).expanduser().resolve()))

    xyz = detail.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    scaling = detail.get_scaling.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = detail.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)

    radius_proxy = _estimate_projected_radius_proxy(
        detail=detail,
        scene_root=Path(scene_root).expanduser().resolve(),
        model_path=Path(model_path).expanduser().resolve(),
        images_subdir=str(cfg.images_subdir),
        max_views=int(cfg.max_views),
    )

    valid_surface = carrier_payload["valid_mask"].astype(bool)
    valid_surface &= carrier_payload["confidence"] >= float(cfg.surface_confidence_floor)
    if not np.any(valid_surface):
        raise RuntimeError("Carrier payload has no valid surface carriers for heuristic routing.")

    centers = carrier_payload["centers"][valid_surface].astype(np.float32, copy=False)
    normals = carrier_payload["normals"][valid_surface].astype(np.float32, copy=False)
    confidence = carrier_payload["confidence"][valid_surface].astype(np.float32, copy=False)
    support_scale = np.maximum(
        carrier_payload["scale_u"][valid_surface].astype(np.float32, copy=False),
        carrier_payload["scale_v"][valid_surface].astype(np.float32, copy=False),
    )
    tree = cKDTree(centers)
    surface_distance, nearest_idx = tree.query(xyz, k=1)
    surface_distance = np.asarray(surface_distance, dtype=np.float32).reshape(-1)
    nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
    nearest_centers = centers[nearest_idx]
    nearest_normals = normals[nearest_idx]
    nearest_conf = confidence[nearest_idx]
    nearest_scale = np.clip(support_scale[nearest_idx], 1e-6, None)
    offset = xyz - nearest_centers
    signed_normal_distance = np.abs(np.sum(offset * nearest_normals, axis=1))
    relative_surface_distance = surface_distance / nearest_scale

    total_gaussians = int(xyz.shape[0])
    if action_payload_path is not None:
        action_payload = load_route_payload(action_payload_path, total_gaussians=total_gaussians)
        if action_payload is None:
            raise RuntimeError(f"Failed to load action payload: {action_payload_path}")
        action = _tensor_dict_to_numpy(action_payload)
    else:
        ones = np.ones((total_gaussians,), dtype=np.float32)
        zeros = np.zeros((total_gaussians,), dtype=np.float32)
        action = {
            "update_strength": ones,
            "attach_strength": ones,
            "detail_weight": ones,
            "prior_color_strength": ones,
            "suppress_strength": zeros,
            "radius_gate": ones,
        }

    contribution_sum = action.get("contribution_sum", action["update_strength"])
    support_count = action.get("support_count", action["update_strength"])

    return {
        "opacity": opacity,
        "scale_min": scaling.min(axis=1),
        "scale_mean": scaling.mean(axis=1),
        "scale_max": scaling.max(axis=1),
        "anisotropy": scaling.max(axis=1) / np.clip(scaling.min(axis=1), 1e-6, None),
        "radius_proxy": radius_proxy,
        "surface_distance": surface_distance,
        "signed_normal_distance": signed_normal_distance.astype(np.float32, copy=False),
        "relative_surface_distance": relative_surface_distance.astype(np.float32, copy=False),
        "surface_confidence": nearest_conf.astype(np.float32, copy=False),
        "support_deficit": (1.0 - np.clip(nearest_conf, 0.0, 1.0)).astype(np.float32, copy=False),
        "update_strength": np.asarray(action["update_strength"], dtype=np.float32),
        "attach_strength": np.asarray(action["attach_strength"], dtype=np.float32),
        "detail_weight": np.asarray(action["detail_weight"], dtype=np.float32),
        "prior_color_strength": np.asarray(action["prior_color_strength"], dtype=np.float32),
        "contribution_score": _normalize_score(np.log1p(np.asarray(contribution_sum, dtype=np.float32))),
        "support_score": _normalize_score(np.asarray(support_count, dtype=np.float32)),
    }


def build_heuristic_route_payload(
    detail: GaussianModel,
    carrier_payload_path: str | Path,
    *,
    scene_root: str | Path,
    model_path: str | Path,
    action_payload_path: str | Path | None = None,
    cfg: HeuristicGaussianRouteConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or HeuristicGaussianRouteConfig()
    features = build_gaussian_route_features(
        detail=detail,
        carrier_payload_path=carrier_payload_path,
        scene_root=scene_root,
        model_path=model_path,
        action_payload_path=action_payload_path,
        cfg=cfg,
    )

    radius_proxy = features["radius_proxy"]
    opacity = features["opacity"]
    relative_surface_distance = features["relative_surface_distance"]
    surface_conf = np.clip(features["surface_confidence"], 0.0, 1.0)
    update_strength = np.clip(features["update_strength"], 0.0, 1.0)
    attach_strength = np.clip(features["attach_strength"], 0.0, 1.0)
    detail_weight = np.clip(features["detail_weight"], 0.0, 1.0)
    prior_color_strength = np.clip(features["prior_color_strength"], 0.0, 1.0)
    contribution_score = np.clip(features["contribution_score"], 0.0, 1.0)
    support_score = np.clip(features["support_score"], 0.0, 1.0)

    radius_gate = np.clip(
        float(cfg.radius_ref_px) / np.maximum(radius_proxy, float(cfg.radius_ref_px)),
        float(cfg.radius_gate_min),
        1.0,
    ).astype(np.float32, copy=False)
    large_radius_score = _sigmoid_np((radius_proxy - float(cfg.radius_ref_px)) / max(float(cfg.radius_temperature_px), 1e-6))
    high_opacity = _sigmoid_np((opacity - float(cfg.opacity_center)) / max(float(cfg.opacity_temperature), 1e-6))
    surface_near = np.exp(-relative_surface_distance / max(float(cfg.surface_distance_scale), 1e-6)).astype(np.float32, copy=False)

    base_attach = np.clip(0.6 * attach_strength + 0.4 * update_strength, 0.0, 1.0)
    base_detail = np.clip(
        0.40 * detail_weight
        + 0.35 * np.maximum(prior_color_strength, contribution_score)
        + 0.25 * support_score,
        0.0,
        1.0,
    )
    floating_score = np.clip((1.0 - surface_near * surface_conf) * high_opacity, 0.0, 1.0)
    p_suppress = np.clip(np.maximum(large_radius_score, floating_score), 0.0, 1.0)
    p_attach = np.clip(
        surface_near * surface_conf * radius_gate * (0.25 + 0.75 * base_attach) * (1.0 - p_suppress),
        0.0,
        1.0,
    )
    # v0.7: detail and suppress are allowed to co-exist. Risk should not erase
    # appearance ownership; executor handles "preserve appearance, limit geometry".
    raw_detail = np.clip(base_detail * (1.0 - 0.5 * p_attach), 0.0, 1.0)
    detail_boost = max(float(cfg.detail_boost), 1e-6)
    if abs(detail_boost - 1.0) > 1e-6:
        # Boost detail ownership smoothly without hard-thresholding. Values in
        # the mid range are lifted first, which is useful for "make the current
        # detail skeleton show up more clearly" experiments.
        p_detail = 1.0 - np.power(1.0 - raw_detail, detail_boost)
    else:
        p_detail = raw_detail
    p_detail = np.clip(p_detail, 0.0, 1.0).astype(np.float32, copy=False)
    p_attach = p_attach.astype(np.float32, copy=False)
    p_suppress = p_suppress.astype(np.float32, copy=False)

    p_promote = np.zeros_like(p_attach, dtype=np.float32)
    p_spawn_source = np.clip(base_detail * surface_conf * (1.0 - surface_near), 0.0, 1.0)

    geometry_update_strength = np.clip(update_strength * p_attach * radius_gate * (1.0 - p_suppress), 0.0, 1.0)
    color_update_strength = np.clip(
        update_strength * np.maximum(p_detail, 0.5 * base_detail),
        0.0,
        1.0,
    )
    effective_attach = np.clip(attach_strength * p_attach * radius_gate * (1.0 - p_suppress), 0.0, 1.0)
    effective_detail = np.clip(detail_weight * np.maximum(p_detail, 0.5 * base_detail), 0.0, 1.0)
    effective_prior = np.clip(
        prior_color_strength * np.maximum(p_detail, 0.5 * base_detail),
        0.0,
        1.0,
    )
    effective_suppress = np.clip(p_suppress, 0.0, 1.0)
    update_out = np.clip(
        np.maximum.reduce(
            [
                geometry_update_strength,
                color_update_strength,
                effective_suppress * float(cfg.suppress_update_floor),
            ]
        ),
        0.0,
        1.0,
    )

    payload = {
        "version": "gaussian_route_v0p7_heuristic",
        "num_gaussians": int(radius_proxy.shape[0]),
        "p_attach": torch.from_numpy(p_attach[:, None]),
        "p_detail": torch.from_numpy(p_detail[:, None]),
        "p_suppress": torch.from_numpy(p_suppress[:, None]),
        "p_promote": torch.from_numpy(p_promote[:, None]),
        "p_spawn_source": torch.from_numpy(p_spawn_source[:, None].astype(np.float32, copy=False)),
        "radius_gate": torch.from_numpy(radius_gate[:, None]),
        "geometry_update_strength": torch.from_numpy(geometry_update_strength[:, None].astype(np.float32, copy=False)),
        "color_update_strength": torch.from_numpy(color_update_strength[:, None].astype(np.float32, copy=False)),
        "update_strength": torch.from_numpy(update_out[:, None].astype(np.float32, copy=False)),
        "attach_strength": torch.from_numpy(effective_attach[:, None].astype(np.float32, copy=False)),
        "detail_weight": torch.from_numpy(effective_detail[:, None].astype(np.float32, copy=False)),
        "prior_color_strength": torch.from_numpy(effective_prior[:, None].astype(np.float32, copy=False)),
        "suppress_strength": torch.from_numpy(effective_suppress[:, None].astype(np.float32, copy=False)),
        "effective_attach": torch.from_numpy(effective_attach[:, None].astype(np.float32, copy=False)),
        "effective_detail": torch.from_numpy(effective_detail[:, None].astype(np.float32, copy=False)),
        "effective_suppress": torch.from_numpy(effective_suppress[:, None].astype(np.float32, copy=False)),
        "diagnostics": {
            key: torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(-1, 1))
            for key, value in features.items()
        },
        "stats": {
            "attach": vector_stats(torch.from_numpy(p_attach)),
            "detail": vector_stats(torch.from_numpy(p_detail)),
            "suppress": vector_stats(torch.from_numpy(p_suppress)),
            "radius_gate": vector_stats(torch.from_numpy(radius_gate)),
            "radius_proxy": vector_stats(torch.from_numpy(radius_proxy)),
            "surface_distance": vector_stats(torch.from_numpy(features["surface_distance"])),
            "relative_surface_distance": vector_stats(torch.from_numpy(relative_surface_distance)),
        },
        "meta": {
            "route_mode": "heuristic",
            "route_semantics": "overlapping_detail_and_suppress_v0p7",
            "heuristic_config": asdict(cfg),
            "scene_root": str(Path(scene_root).expanduser().resolve()),
            "model_path": str(Path(model_path).expanduser().resolve()),
            "carrier_payload_path": str(Path(carrier_payload_path).expanduser().resolve()),
            "action_payload_path": str(Path(action_payload_path).expanduser().resolve()) if action_payload_path else None,
        },
    }
    return payload
