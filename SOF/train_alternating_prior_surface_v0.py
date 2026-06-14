from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scene.sugar_like_meshgs_model import SugarLikeMeshGaussianModel, _load_mesh_np
from train_mip_to_sof_surface_v0 import copy_render_config, load_cameras_for_split, select_uniform
from train_sugar_like_meshgs_prior_v0 import save_sugar_like_meshgs
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.sh_utils import SH2RGB


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


def blur_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0]


def load_image_chw(path: Path, device: torch.device) -> torch.Tensor:
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32)


def sample_image_at_xy(image_chw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if xy.numel() == 0:
        return torch.zeros((0, 3), dtype=image_chw.dtype, device=image_chw.device)
    height = image_chw.shape[1]
    width = image_chw.shape[2]
    if width <= 1:
        grid_x = torch.zeros_like(xy[:, 0])
    else:
        grid_x = (xy[:, 0] / float(width - 1)) * 2.0 - 1.0
    if height <= 1:
        grid_y = torch.zeros_like(xy[:, 1])
    else:
        grid_y = (xy[:, 1] / float(height - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image_chw.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def project_points_camera_torch(camera, points_xyz: torch.Tensor, depth_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
    R = torch.as_tensor(camera.R, device=points_xyz.device, dtype=points_xyz.dtype)
    T = torch.as_tensor(camera.T, device=points_xyz.device, dtype=points_xyz.dtype)
    xyz_cam = points_xyz @ R + T.unsqueeze(0)
    z = xyz_cam[:, 2]
    z_safe = torch.clamp_min(z, 1e-6)
    x = xyz_cam[:, 0] / z_safe * float(camera.focal_x) + float(camera.image_width) / 2.0
    y = xyz_cam[:, 1] / z_safe * float(camera.focal_y) + float(camera.image_height) / 2.0
    projected = torch.stack([x, y, z], dim=1)
    valid = z > float(depth_min)
    valid &= x >= 0.0
    valid &= x <= float(camera.image_width - 1)
    valid &= y >= 0.0
    valid &= y <= float(camera.image_height - 1)
    return projected, valid


def weighted_rgb_l1(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(weight.sum() * 3.0, min=1e-8)
    return (torch.abs(pred_rgb - target_rgb) * weight[:, None]).sum() / denom


def clamp_surface_vertex_displacement(
    meshgs: SugarLikeMeshGaussianModel,
    surface_vertices_init: torch.Tensor,
    max_displacement: float,
) -> None:
    if float(max_displacement) <= 0.0:
        return
    with torch.no_grad():
        delta = meshgs._surface_vertices - surface_vertices_init
        norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        scale = torch.clamp(float(max_displacement) / torch.clamp(norm, min=1e-12), max=1.0)
        meshgs._surface_vertices.copy_(surface_vertices_init + delta * scale)


def current_dc_rgb(meshgs: SugarLikeMeshGaussianModel) -> torch.Tensor:
    return torch.clamp(SH2RGB(meshgs._features_dc[:, 0, :]), 0.0, 1.0)


class ImageCache:
    def __init__(self, device: torch.device, blur_kernel: int):
        self.device = device
        self.blur_kernel = int(blur_kernel)
        self.rgb_cache: Dict[Path, torch.Tensor] = {}
        self.low_cache: Dict[Path, torch.Tensor] = {}

    def load(self, path: Path) -> torch.Tensor:
        tensor = self.rgb_cache.get(path)
        if tensor is None:
            tensor = load_image_chw(path, self.device)
            self.rgb_cache[path] = tensor
        return tensor

    def load_lowpass(self, path: Path) -> torch.Tensor:
        tensor = self.low_cache.get(path)
        if tensor is None:
            tensor = blur_chw(self.load(path), self.blur_kernel)
            self.low_cache[path] = tensor
        return tensor


def lookup_observation_path(observation_root: Path, image_name: str) -> Optional[Path]:
    names = [
        str(image_name),
        normalize_image_name(str(image_name)),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    for name in names:
        candidate = observation_root / f"{name}.npz"
        if candidate.is_file():
            return candidate
    lower_index = {child.stem.lower(): child for child in observation_root.glob("*.npz")}
    for name in names:
        candidate = lower_index.get(str(name).lower())
        if candidate is not None:
            return candidate
    return None


def lookup_indexed_path(index: Optional[Dict[str, Path]], image_name: str) -> Optional[Path]:
    if index is None:
        return None
    names = [
        str(image_name),
        normalize_image_name(str(image_name)),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    for name in names:
        candidate = index.get(name)
        if candidate is not None:
            return Path(candidate)
    lower_index = {str(key).lower(): Path(value) for key, value in index.items()}
    for name in names:
        candidate = lower_index.get(str(name).lower())
        if candidate is not None:
            return candidate
    return None


def weighted_color_variance(
    local_ids: torch.Tensor,
    colors: torch.Tensor,
    weights: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if local_ids.numel() <= 0:
        return torch.zeros((), dtype=colors.dtype, device=colors.device)
    sum_w = torch.zeros((count,), dtype=colors.dtype, device=colors.device)
    sum_rgb = torch.zeros((count, 3), dtype=colors.dtype, device=colors.device)
    sum_rgb2 = torch.zeros((count, 3), dtype=colors.dtype, device=colors.device)
    sum_w.scatter_add_(0, local_ids, weights)
    sum_rgb.scatter_add_(0, local_ids[:, None].expand(-1, 3), colors * weights[:, None])
    sum_rgb2.scatter_add_(0, local_ids[:, None].expand(-1, 3), colors.pow(2) * weights[:, None])
    valid = sum_w > 1e-8
    if not torch.any(valid):
        return torch.zeros((), dtype=colors.dtype, device=colors.device)
    mean = sum_rgb[valid] / sum_w[valid, None]
    mean2 = sum_rgb2[valid] / sum_w[valid, None]
    var = torch.clamp(mean2 - mean.pow(2), min=0.0)
    return var.mean()


def save_payload_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_meshgs_checkpoint(meshgs: SugarLikeMeshGaussianModel, output_dir: Path, iteration: int) -> None:
    point_dir = output_dir / "point_cloud" / f"iteration_{int(iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    classic = meshgs.materialize_gaussian_model()
    classic.save_ply(str(point_dir / "point_cloud.ply"))
    classic.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    meshgs.save_bound_metadata(str(point_dir / "mesh_bound_state.pt"))
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(classic.get_xyz.shape[0])}, f, indent=2)


def load_patch_bank(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = (
        "centers",
        "normals",
        "tangent_u",
        "tangent_v",
        "scale_u",
        "scale_v",
        "scale_n",
        "face_ids",
        "bary_coords",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"Patch bank is missing '{key}': {path}")
    return {key: np.asarray(data[key]) for key in data.files}


def load_view_records(
    *,
    observation_root: Path,
    cameras: Sequence[object],
    prior_index: Dict[str, Path],
    anchor_index: Optional[Dict[str, Path]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for camera in cameras:
        obs_path = lookup_observation_path(observation_root, str(camera.image_name))
        if obs_path is None:
            continue
        obs = np.load(obs_path)
        patch_ids = np.asarray(obs["patch_ids"]).reshape(-1).astype(np.int64, copy=False)
        if patch_ids.size == 0:
            continue
        sample_weight = np.asarray(obs["sample_weight"]).reshape(-1).astype(np.float32, copy=False)
        prior_path = lookup_indexed_path(prior_index, str(camera.image_name))
        anchor_path = lookup_indexed_path(anchor_index, str(camera.image_name))
        if prior_path is None:
            continue
        records.append(
            {
                "camera": camera,
                "image_name": normalize_image_name(str(camera.image_name)),
                "observation_path": obs_path,
                "global_ids": patch_ids,
                "sample_weight": sample_weight,
                "prior_path": prior_path,
                "anchor_path": anchor_path,
            }
        )
    if not records:
        raise RuntimeError(f"No patch observation records matched the selected cameras under {observation_root}")
    return records


def build_local_view_records(
    view_records: Sequence[Dict[str, object]],
    global_to_local: np.ndarray,
    device: torch.device,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for record in view_records:
        global_ids = np.asarray(record["global_ids"], dtype=np.int64)
        local_ids = global_to_local[global_ids]
        keep = local_ids >= 0
        if not np.any(keep):
            continue
        local_ids = local_ids[keep].astype(np.int64, copy=False)
        sample_weight = np.asarray(record["sample_weight"], dtype=np.float32)[keep]
        records.append(
            {
                "camera": record["camera"],
                "image_name": record["image_name"],
                "observation_path": record["observation_path"],
                "prior_path": record["prior_path"],
                "anchor_path": record["anchor_path"],
                "local_ids_np": local_ids,
                "sample_weight_np": sample_weight,
                "local_ids_t": torch.from_numpy(local_ids).to(device=device, dtype=torch.long),
                "sample_weight_t": torch.from_numpy(sample_weight).to(device=device, dtype=torch.float32),
            }
        )
    if not records:
        raise RuntimeError("No selected carriers remain visible in any observation view.")
    return records


@torch.no_grad()
def aggregate_surface_targets(
    *,
    xyz_world: torch.Tensor,
    view_records: Sequence[Dict[str, object]],
    image_cache: ImageCache,
    min_views: int,
    min_confidence: float,
    max_disagreement: float,
    depth_min: float,
    anchor_lowfreq_threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    count = int(xyz_world.shape[0])
    device = xyz_world.device
    dtype = xyz_world.dtype
    color_sum = torch.zeros((count, 3), dtype=dtype, device=device)
    weight_sum = torch.zeros((count,), dtype=dtype, device=device)
    anchor_color_sum = torch.zeros((count, 3), dtype=dtype, device=device)
    anchor_weight_sum = torch.zeros((count,), dtype=dtype, device=device)
    anchor_view_count = torch.zeros((count,), dtype=torch.int32, device=device)
    view_count = torch.zeros((count,), dtype=torch.int32, device=device)
    agreement_sum = torch.zeros((count,), dtype=dtype, device=device)

    for record in view_records:
        local_ids = record["local_ids_t"]
        if local_ids.numel() == 0:
            continue
        projected, valid = project_points_camera_torch(record["camera"], xyz_world[local_ids], depth_min=depth_min)
        if not torch.any(valid):
            continue
        xy = projected[valid, :2]
        prior_rgb = sample_image_at_xy(image_cache.load(record["prior_path"]), xy)
        sample_weight = record["sample_weight_t"][valid]
        if record["anchor_path"] is not None:
            anchor_rgb = sample_image_at_xy(image_cache.load(record["anchor_path"]), xy)
            anchor_keep = sample_weight > 1e-8
            if torch.any(anchor_keep):
                anchor_ids = local_ids[valid][anchor_keep]
                anchor_rgb = anchor_rgb[anchor_keep]
                anchor_sample_weight = sample_weight[anchor_keep]
                anchor_color_sum.scatter_add_(
                    0,
                    anchor_ids[:, None].expand(-1, 3),
                    anchor_rgb * anchor_sample_weight[:, None],
                )
                anchor_weight_sum.scatter_add_(0, anchor_ids, anchor_sample_weight)
                anchor_view_count.scatter_add_(0, anchor_ids, torch.ones_like(anchor_ids, dtype=anchor_view_count.dtype))
        if record["anchor_path"] is not None and float(anchor_lowfreq_threshold) > 0.0:
            prior_low = sample_image_at_xy(image_cache.load_lowpass(record["prior_path"]), xy)
            anchor_low = sample_image_at_xy(image_cache.load_lowpass(record["anchor_path"]), xy)
            low_delta = torch.mean(torch.abs(prior_low - anchor_low), dim=1)
            agreement = torch.clamp(1.0 - low_delta / float(anchor_lowfreq_threshold), min=0.0, max=1.0)
        else:
            agreement = torch.ones_like(sample_weight)
        weight = sample_weight * agreement
        keep = weight > 1e-8
        if not torch.any(keep):
            continue
        ids = local_ids[valid][keep]
        prior_rgb = prior_rgb[keep]
        weight = weight[keep]
        agreement = agreement[keep]
        color_sum.scatter_add_(0, ids[:, None].expand(-1, 3), prior_rgb * weight[:, None])
        weight_sum.scatter_add_(0, ids, weight)
        view_count.scatter_add_(0, ids, torch.ones_like(ids, dtype=view_count.dtype))
        agreement_sum.scatter_add_(0, ids, agreement)

    fused_rgb = torch.zeros((count, 3), dtype=dtype, device=device)
    supported = weight_sum > 1e-8
    fused_rgb[supported] = color_sum[supported] / weight_sum[supported, None]
    anchor_rgb = torch.zeros((count, 3), dtype=dtype, device=device)
    anchor_supported = anchor_weight_sum > 1e-8
    anchor_rgb[anchor_supported] = anchor_color_sum[anchor_supported] / anchor_weight_sum[anchor_supported, None]

    disagreement_sum = torch.zeros((count,), dtype=dtype, device=device)
    for record in view_records:
        local_ids = record["local_ids_t"]
        if local_ids.numel() == 0:
            continue
        projected, valid = project_points_camera_torch(record["camera"], xyz_world[local_ids], depth_min=depth_min)
        if not torch.any(valid):
            continue
        xy = projected[valid, :2]
        prior_rgb = sample_image_at_xy(image_cache.load(record["prior_path"]), xy)
        sample_weight = record["sample_weight_t"][valid]
        if record["anchor_path"] is not None and float(anchor_lowfreq_threshold) > 0.0:
            prior_low = sample_image_at_xy(image_cache.load_lowpass(record["prior_path"]), xy)
            anchor_low = sample_image_at_xy(image_cache.load_lowpass(record["anchor_path"]), xy)
            low_delta = torch.mean(torch.abs(prior_low - anchor_low), dim=1)
            agreement = torch.clamp(1.0 - low_delta / float(anchor_lowfreq_threshold), min=0.0, max=1.0)
        else:
            agreement = torch.ones_like(sample_weight)
        weight = sample_weight * agreement
        keep = weight > 1e-8
        if not torch.any(keep):
            continue
        ids = local_ids[valid][keep]
        prior_rgb = prior_rgb[keep]
        weight = weight[keep]
        ref = fused_rgb[ids]
        delta = torch.mean(torch.abs(prior_rgb - ref), dim=1)
        disagreement_sum.scatter_add_(0, ids, delta * weight)

    disagreement = torch.zeros((count,), dtype=dtype, device=device)
    disagreement[supported] = disagreement_sum[supported] / weight_sum[supported]
    disagreement_gate = torch.clamp(1.0 - disagreement / max(float(max_disagreement), 1e-8), min=0.0, max=1.0)
    view_gate = torch.clamp(view_count.to(dtype=dtype) / max(float(min_views), 1.0), min=0.0, max=1.0)
    mean_weight = torch.zeros((count,), dtype=dtype, device=device)
    mean_weight[supported] = weight_sum[supported] / torch.clamp(view_count[supported].to(dtype=dtype), min=1.0)
    mean_agreement = torch.ones((count,), dtype=dtype, device=device)
    observed = view_count > 0
    mean_agreement[observed] = agreement_sum[observed] / torch.clamp(view_count[observed].to(dtype=dtype), min=1.0)
    confidence = mean_weight * mean_agreement * view_gate * disagreement_gate
    valid_mask = observed & (view_count >= int(min_views)) & (confidence >= float(min_confidence)) & (
        disagreement <= float(max_disagreement)
    )

    payload = {
        "centers": xyz_world.detach().cpu().numpy().astype(np.float32, copy=False),
        "normals": np.zeros((count, 3), dtype=np.float32),
        "tangent_u": np.zeros((count, 3), dtype=np.float32),
        "tangent_v": np.zeros((count, 3), dtype=np.float32),
        "scale_u": np.zeros((count,), dtype=np.float32),
        "scale_v": np.zeros((count,), dtype=np.float32),
        "scale_n": np.zeros((count,), dtype=np.float32),
        "face_ids": np.zeros((count,), dtype=np.int64),
        "bary_coords": np.zeros((count, 3), dtype=np.float32),
        "fused_rgb": fused_rgb.detach().cpu().numpy().astype(np.float32, copy=False),
        "anchor_rgb": anchor_rgb.detach().cpu().numpy().astype(np.float32, copy=False),
        "confidence": confidence.detach().cpu().numpy().astype(np.float32, copy=False),
        "disagreement": disagreement.detach().cpu().numpy().astype(np.float32, copy=False),
        "view_count": view_count.detach().cpu().numpy().astype(np.int32, copy=False),
        "valid_mask": valid_mask.detach().cpu().numpy().astype(bool, copy=False),
        "weight_sum": weight_sum.detach().cpu().numpy().astype(np.float32, copy=False),
        "anchor_weight_sum": anchor_weight_sum.detach().cpu().numpy().astype(np.float32, copy=False),
        "anchor_view_count": anchor_view_count.detach().cpu().numpy().astype(np.int32, copy=False),
        "mean_agreement": mean_agreement.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    summary = {
        "count": int(count),
        "observed": int(observed.sum().item()),
        "valid": int(valid_mask.sum().item()),
        "mean_confidence": float(confidence.mean().item()) if count > 0 else 0.0,
        "mean_disagreement": float(disagreement.mean().item()) if count > 0 else 0.0,
    }
    return payload, summary


def merge_geometry_fields(payload: Dict[str, np.ndarray], bank: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = dict(payload)
    for key in ("normals", "tangent_u", "tangent_v", "scale_u", "scale_v", "scale_n", "face_ids", "bary_coords"):
        merged[key] = np.asarray(bank[key])
    return merged


def subset_patch_bank_geometry(patch_bank: Dict[str, np.ndarray], selected_indices: np.ndarray) -> Dict[str, np.ndarray]:
    subset: Dict[str, np.ndarray] = {}
    for key in ("normals", "tangent_u", "tangent_v", "scale_u", "scale_v", "scale_n", "face_ids", "bary_coords"):
        values = np.asarray(patch_bank[key])
        subset[key] = values[selected_indices]
    return subset


def select_payload_indices(
    payload: Dict[str, np.ndarray],
    *,
    min_confidence: float,
    max_disagreement: float,
    min_views: int,
    max_count: int,
    seed: int,
) -> np.ndarray:
    valid = np.asarray(payload["valid_mask"]).astype(bool, copy=False)
    valid &= np.asarray(payload["confidence"]).reshape(-1) >= float(min_confidence)
    valid &= np.asarray(payload["disagreement"]).reshape(-1) <= float(max_disagreement)
    valid &= np.asarray(payload["view_count"]).reshape(-1) >= int(min_views)
    indices = np.flatnonzero(valid).astype(np.int64, copy=False)
    if int(max_count) > 0 and indices.size > int(max_count):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=int(max_count), replace=False)).astype(np.int64, copy=False)
    return indices


def resolve_initial_colors(
    aggregated_payload: Dict[str, np.ndarray],
    selected_indices: np.ndarray,
    *,
    init_color_source: str,
    init_color_gray_value: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    source = str(init_color_source)
    fused_rgb = np.clip(aggregated_payload["fused_rgb"][selected_indices].astype(np.float32, copy=False), 0.0, 1.0)
    summary: Dict[str, object] = {"init_color_source": source}
    if source == "fused_rgb":
        summary["init_color_fallback_count"] = 0
        return fused_rgb, summary
    if source == "anchor_rgb":
        anchor_weight_sum = np.asarray(aggregated_payload["anchor_weight_sum"]).reshape(-1).astype(np.float32, copy=False)
        anchor_supported = anchor_weight_sum[selected_indices] > 1e-8
        if not np.any(anchor_supported):
            raise RuntimeError("init_color_source=anchor_rgb requested, but no selected carrier has anchor RGB support.")
        colors = fused_rgb.copy()
        anchor_rgb = np.clip(aggregated_payload["anchor_rgb"][selected_indices].astype(np.float32, copy=False), 0.0, 1.0)
        colors[anchor_supported] = anchor_rgb[anchor_supported]
        summary["init_color_fallback_count"] = int((~anchor_supported).sum())
        summary["init_color_anchor_supported_count"] = int(anchor_supported.sum())
        return colors, summary
    if source == "neutral_gray":
        gray_value = float(np.clip(init_color_gray_value, 0.0, 1.0))
        colors = np.full((selected_indices.size, 3), gray_value, dtype=np.float32)
        summary["init_color_fallback_count"] = 0
        summary["init_color_gray_value"] = gray_value
        return colors, summary
    raise ValueError(f"Unsupported init_color_source='{source}'")


def initialize_meshgs_from_payload(
    *,
    mesh_path: str,
    patch_bank: Dict[str, np.ndarray],
    aggregated_payload: Dict[str, np.ndarray],
    selected_indices: np.ndarray,
    sh_degree: int,
    scale_multiplier: float,
    thickness_multiplier: float,
    init_opacity: float,
    max_disagreement: float,
    init_color_source: str,
    init_color_gray_value: float,
) -> Tuple[SugarLikeMeshGaussianModel, Dict[str, float]]:
    if selected_indices.size == 0:
        raise RuntimeError("No valid surface carriers remain after aggregation filters.")

    vertices, faces = _load_mesh_np(mesh_path)
    colors, color_summary = resolve_initial_colors(
        aggregated_payload,
        selected_indices,
        init_color_source=init_color_source,
        init_color_gray_value=init_color_gray_value,
    )
    confidence = np.clip(aggregated_payload["confidence"][selected_indices].astype(np.float32), 0.0, 1.0)
    disagreement = aggregated_payload["disagreement"][selected_indices].astype(np.float32)
    disagreement_gate = 1.0 - np.clip(disagreement / max(float(max_disagreement), 1e-8), 0.0, 1.0)
    opacity = np.clip(float(init_opacity) * (0.25 + 0.75 * confidence * disagreement_gate), 1e-4, 0.95)

    meshgs = SugarLikeMeshGaussianModel(sh_degree, use_SBs=False)
    meshgs.initialize_from_arrays(
        vertices=vertices,
        faces=faces,
        face_ids=patch_bank["face_ids"][selected_indices].astype(np.int64, copy=False),
        bary_coords=patch_bank["bary_coords"][selected_indices].astype(np.float32, copy=False),
        colors=colors,
        scale_u=patch_bank["scale_u"][selected_indices].astype(np.float32, copy=False) * float(scale_multiplier),
        scale_v=patch_bank["scale_v"][selected_indices].astype(np.float32, copy=False) * float(scale_multiplier),
        scale_n=patch_bank["scale_n"][selected_indices].astype(np.float32, copy=False)
        * float(scale_multiplier)
        * float(thickness_multiplier),
        opacity=opacity,
        learn_surface_vertices=True,
        learn_plane_scales=False,
        learn_inplane_rotation=False,
        build_normal_pairs=True,
    )
    summary = {
        "payload_count": int(aggregated_payload["valid_mask"].shape[0]),
        "selected_count": int(selected_indices.size),
        "selected_ratio": float(selected_indices.size / max(int(aggregated_payload["valid_mask"].shape[0]), 1)),
        "normal_consistency_pairs": int(meshgs._normal_consistency_pairs.shape[0]),
    }
    summary.update(color_summary)
    return meshgs, summary


def build_cycle_plan(
    *,
    cycles: int,
    total_steps: int,
    appearance_steps: int,
    structure_steps: int,
) -> List[Dict[str, int]]:
    a_default = max(int(appearance_steps), 0)
    b_default = max(int(structure_steps), 0)
    if a_default <= 0 and b_default <= 0:
        raise ValueError("At least one of appearance_steps or structure_steps must be positive.")

    plan: List[Dict[str, int]] = []
    cumulative_steps = 0
    if int(total_steps) > 0:
        remaining = int(total_steps)
        cycle_idx = 1
        while remaining > 0:
            a_steps = min(a_default, remaining) if a_default > 0 else 0
            remaining -= a_steps
            b_steps = min(b_default, remaining) if b_default > 0 else 0
            remaining -= b_steps
            cycle_total = a_steps + b_steps
            if cycle_total <= 0:
                break
            plan.append(
                {
                    "cycle": int(cycle_idx),
                    "appearance_steps": int(a_steps),
                    "structure_steps": int(b_steps),
                    "step_begin": int(cumulative_steps + 1),
                    "step_end": int(cumulative_steps + cycle_total),
                }
            )
            cumulative_steps += cycle_total
            cycle_idx += 1
        if cumulative_steps <= 0:
            raise ValueError("total_steps produced an empty alternating schedule.")
        return plan

    for cycle_idx in range(1, int(cycles) + 1):
        cycle_total = a_default + b_default
        plan.append(
            {
                "cycle": int(cycle_idx),
                "appearance_steps": int(a_default),
                "structure_steps": int(b_default),
                "step_begin": int(cumulative_steps + 1),
                "step_end": int(cumulative_steps + cycle_total),
            }
        )
        cumulative_steps += cycle_total
    return plan


def optimize_appearance(
    *,
    meshgs: SugarLikeMeshGaussianModel,
    target_rgb: torch.Tensor,
    target_weight: torch.Tensor,
    anchor_rgb: torch.Tensor,
    steps: int,
    lr: float,
    lambda_anchor: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    progress_label: str = "appearance-only A",
) -> Dict[str, float]:
    if int(steps) <= 0 or float(lr) <= 0.0:
        current = current_dc_rgb(meshgs)
        loss = weighted_rgb_l1(current, target_rgb, target_weight)
        return {"before": float(loss.item()), "after": float(loss.item()), "steps": 0}

    local_optimizer = optimizer
    if local_optimizer is None:
        local_optimizer = torch.optim.Adam([{"params": [meshgs._features_dc], "lr": float(lr), "name": "f_dc"}], eps=1e-15)
    with torch.no_grad():
        before = weighted_rgb_l1(current_dc_rgb(meshgs), target_rgb, target_weight)

    progress = tqdm(range(int(steps)), desc=str(progress_label), leave=False)
    after = before
    for _ in progress:
        local_optimizer.zero_grad(set_to_none=True)
        current = current_dc_rgb(meshgs)
        loss = weighted_rgb_l1(current, target_rgb, target_weight)
        if float(lambda_anchor) > 0.0:
            loss = loss + float(lambda_anchor) * weighted_rgb_l1(current, anchor_rgb, target_weight)
        loss.backward()
        local_optimizer.step()
        after = loss.detach()
        progress.set_postfix(loss=f"{float(after.item()):.6f}")
    return {"before": float(before.item()), "after": float(after.item()), "steps": int(steps)}


def compute_structure_losses(
    *,
    meshgs: SugarLikeMeshGaussianModel,
    view_records: Sequence[Dict[str, object]],
    image_cache: ImageCache,
    fixed_rgb: torch.Tensor,
    carrier_weight: torch.Tensor,
    depth_min: float,
    anchor_lowfreq_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    photo_terms: List[torch.Tensor] = []
    variance_ids: List[torch.Tensor] = []
    variance_rgb: List[torch.Tensor] = []
    variance_weight: List[torch.Tensor] = []

    for record in view_records:
        local_ids = record["local_ids_t"]
        if local_ids.numel() == 0:
            continue
        xyz = meshgs.get_xyz[local_ids]
        projected, valid = project_points_camera_torch(record["camera"], xyz, depth_min=depth_min)
        if not torch.any(valid):
            continue
        xy = projected[valid, :2]
        prior_rgb = sample_image_at_xy(image_cache.load(record["prior_path"]), xy)
        weight = record["sample_weight_t"][valid] * carrier_weight[local_ids[valid]]
        if record["anchor_path"] is not None and float(anchor_lowfreq_threshold) > 0.0:
            prior_low = sample_image_at_xy(image_cache.load_lowpass(record["prior_path"]), xy)
            anchor_low = sample_image_at_xy(image_cache.load_lowpass(record["anchor_path"]), xy)
            low_delta = torch.mean(torch.abs(prior_low - anchor_low), dim=1)
            agreement = torch.clamp(1.0 - low_delta / float(anchor_lowfreq_threshold), min=0.0, max=1.0)
            weight = weight * agreement
        keep = weight > 1e-8
        if not torch.any(keep):
            continue
        ids = local_ids[valid][keep]
        prior_rgb = prior_rgb[keep]
        weight = weight[keep]
        target = fixed_rgb[ids]
        photo_terms.append(torch.abs(prior_rgb - target) * weight[:, None])
        variance_ids.append(ids)
        variance_rgb.append(prior_rgb)
        variance_weight.append(weight)

    if not photo_terms:
        zero = torch.zeros((), dtype=fixed_rgb.dtype, device=fixed_rgb.device)
        return zero, zero, 0

    photo_cat = torch.cat(photo_terms, dim=0)
    weight_cat = torch.cat(variance_weight, dim=0)
    photo_loss = photo_cat.sum() / torch.clamp(weight_cat.sum() * 3.0, min=1e-8)
    variance_loss = weighted_color_variance(
        torch.cat(variance_ids, dim=0),
        torch.cat(variance_rgb, dim=0),
        weight_cat,
        count=int(fixed_rgb.shape[0]),
    )
    return photo_loss, variance_loss, int(weight_cat.shape[0])


def optimize_structure(
    *,
    meshgs: SugarLikeMeshGaussianModel,
    view_records: Sequence[Dict[str, object]],
    image_cache: ImageCache,
    fixed_rgb: torch.Tensor,
    carrier_weight: torch.Tensor,
    surface_vertices_init: torch.Tensor,
    steps: int,
    lr: float,
    lambda_photo: float,
    lambda_mv: float,
    lambda_delta: float,
    lambda_normal: float,
    depth_min: float,
    max_surface_vertex_displacement: float,
    anchor_lowfreq_threshold: float,
    step_view_limit: int,
    seed: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    progress_label: str = "structure-only B",
) -> Dict[str, float]:
    if int(steps) <= 0 or float(lr) <= 0.0:
        before_photo, before_mv, sample_count = compute_structure_losses(
            meshgs=meshgs,
            view_records=view_records,
            image_cache=image_cache,
            fixed_rgb=fixed_rgb,
            carrier_weight=carrier_weight,
            depth_min=depth_min,
            anchor_lowfreq_threshold=anchor_lowfreq_threshold,
        )
        return {
            "before_photo": float(before_photo.item()),
            "after_photo": float(before_photo.item()),
            "before_mv": float(before_mv.item()),
            "after_mv": float(before_mv.item()),
            "sample_count": int(sample_count),
            "steps": 0,
        }

    local_optimizer = optimizer
    if local_optimizer is None:
        local_optimizer = torch.optim.Adam([{"params": [meshgs._surface_vertices], "lr": float(lr), "name": "surface_vertices"}], eps=1e-15)
    with torch.no_grad():
        before_photo, before_mv, before_samples = compute_structure_losses(
            meshgs=meshgs,
            view_records=view_records,
            image_cache=image_cache,
            fixed_rgb=fixed_rgb,
            carrier_weight=carrier_weight,
            depth_min=depth_min,
            anchor_lowfreq_threshold=anchor_lowfreq_threshold,
        )

    rng = Random(int(seed))
    after_photo = before_photo
    after_mv = before_mv
    after_samples = before_samples
    progress = tqdm(range(int(steps)), desc=str(progress_label), leave=False)
    for _ in progress:
        if int(step_view_limit) > 0 and len(view_records) > int(step_view_limit):
            indices = sorted(rng.sample(range(len(view_records)), int(step_view_limit)))
            batch = [view_records[idx] for idx in indices]
        else:
            batch = list(view_records)

        local_optimizer.zero_grad(set_to_none=True)
        photo_loss, mv_loss, sample_count = compute_structure_losses(
            meshgs=meshgs,
            view_records=batch,
            image_cache=image_cache,
            fixed_rgb=fixed_rgb,
            carrier_weight=carrier_weight,
            depth_min=depth_min,
            anchor_lowfreq_threshold=anchor_lowfreq_threshold,
        )
        delta = meshgs._surface_vertices - surface_vertices_init
        delta_loss = delta.pow(2).sum(dim=1).mean()
        normal_loss = meshgs.normal_consistency_loss()
        loss = (
            float(lambda_photo) * photo_loss
            + float(lambda_mv) * mv_loss
            + float(lambda_delta) * delta_loss
            + float(lambda_normal) * normal_loss
        )
        loss.backward()
        local_optimizer.step()
        clamp_surface_vertex_displacement(meshgs, surface_vertices_init, max_surface_vertex_displacement)
        after_photo = photo_loss.detach()
        after_mv = mv_loss.detach()
        after_samples = sample_count
        progress.set_postfix(photo=f"{float(after_photo.item()):.6f}", mv=f"{float(after_mv.item()):.6f}")

    return {
        "before_photo": float(before_photo.item()),
        "after_photo": float(after_photo.item()),
        "before_mv": float(before_mv.item()),
        "after_mv": float(after_mv.item()),
        "sample_count": int(after_samples),
        "steps": int(steps),
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Alternating prior surface optimization: appearance-only A, structure-only B.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--camera_model_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--patch_observation_root", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--anchor_dir", default=None)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--total_steps", type=int, default=0)
    parser.add_argument("--appearance_steps", type=int, default=200)
    parser.add_argument("--structure_steps", type=int, default=200)
    parser.add_argument("--structure_view_limit", type=int, default=4)
    parser.add_argument("--appearance_lr", type=float, default=0.02)
    parser.add_argument("--structure_lr", type=float, default=5e-4)
    parser.add_argument("--appearance_anchor_lambda", type=float, default=0.05)
    parser.add_argument("--lambda_structure_photo", type=float, default=1.0)
    parser.add_argument("--lambda_structure_mv", type=float, default=0.25)
    parser.add_argument("--lambda_structure_delta", type=float, default=0.05)
    parser.add_argument("--lambda_structure_normal", type=float, default=0.02)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.05)
    parser.add_argument("--max_disagreement", type=float, default=0.10)
    parser.add_argument("--max_count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--anchor_lowfreq_threshold", type=float, default=0.08)
    parser.add_argument("--anchor_lowfreq_kernel", type=int, default=15)
    parser.add_argument("--max_surface_vertex_displacement", type=float, default=0.02)
    parser.add_argument("--meshgs_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--meshgs_thickness_multiplier", type=float, default=0.5)
    parser.add_argument("--meshgs_init_opacity", type=float, default=0.35)
    parser.add_argument("--init_color_source", choices=["fused_rgb", "anchor_rgb", "neutral_gray"], default="fused_rgb")
    parser.add_argument("--init_color_gray_value", type=float, default=0.5)
    parser.add_argument("--save_every_cycles", type=int, default=0)
    parser.add_argument("--output_iteration", type=int, default=0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    safe_state(bool(args.quiet))

    if not torch.cuda.is_available():
        raise RuntimeError("alternating_prior_surface_v0 currently requires CUDA.")
    device = torch.device("cuda")

    scene_root = Path(args.scene_root).expanduser().resolve()
    camera_model_path = Path(args.camera_model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    patch_root = Path(args.patch_observation_root).expanduser().resolve()
    prior_dir = Path(args.prior_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve() if args.anchor_dir else None
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    patch_bank_path = patch_root / "mesh_patch_bank_v0.npz"
    observation_dir = patch_root / "camera_patch_observations"
    if not camera_model_path.is_dir():
        raise FileNotFoundError(f"camera_model_path not found: {camera_model_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    if not patch_bank_path.is_file():
        raise FileNotFoundError(f"patch bank not found: {patch_bank_path}")
    if not observation_dir.is_dir():
        raise FileNotFoundError(f"patch observation dir not found: {observation_dir}")
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior dir not found: {prior_dir}")
    if anchor_dir is not None and not anchor_dir.is_dir():
        raise FileNotFoundError(f"anchor dir not found: {anchor_dir}")

    cameras_all = load_cameras_for_split(
        scene_root,
        camera_model_path,
        str(args.images_subdir),
        str(args.split),
    )

    patch_bank = load_patch_bank(patch_bank_path)
    prior_index = index_image_dir(str(prior_dir))
    anchor_index = index_image_dir(str(anchor_dir)) if anchor_dir is not None else None
    matched_view_records = load_view_records(
        observation_root=observation_dir,
        cameras=cameras_all,
        prior_index=prior_index,
        anchor_index=anchor_index,
    )
    full_view_records = select_uniform(matched_view_records, int(args.max_views))
    if not full_view_records:
        raise RuntimeError("No matched observation/prior views selected for alternating prior surface training.")
    cameras = [record["camera"] for record in full_view_records]
    image_cache = ImageCache(device=device, blur_kernel=int(args.anchor_lowfreq_kernel))

    patch_centers = torch.from_numpy(np.asarray(patch_bank["centers"], dtype=np.float32)).to(device=device)
    initial_payload, initial_summary = aggregate_surface_targets(
        xyz_world=patch_centers,
        view_records=build_local_view_records(
            full_view_records,
            np.arange(patch_centers.shape[0], dtype=np.int64),
            device=device,
        ),
        image_cache=image_cache,
        min_views=int(args.min_views),
        min_confidence=float(args.min_confidence),
        max_disagreement=float(args.max_disagreement),
        depth_min=float(args.depth_min),
        anchor_lowfreq_threshold=float(args.anchor_lowfreq_threshold),
    )
    initial_payload = merge_geometry_fields(initial_payload, patch_bank)
    initial_payload_path = output_model_path / "carrier_payload_cycle00_init.npz"
    save_payload_npz(initial_payload_path, initial_payload)

    selected_indices = select_payload_indices(
        initial_payload,
        min_confidence=float(args.min_confidence),
        max_disagreement=float(args.max_disagreement),
        min_views=int(args.min_views),
        max_count=int(args.max_count),
        seed=int(args.seed),
    )
    meshgs, init_summary = initialize_meshgs_from_payload(
        mesh_path=str(mesh_path),
        patch_bank=patch_bank,
        aggregated_payload=initial_payload,
        selected_indices=selected_indices,
        sh_degree=int(args.sh_degree),
        scale_multiplier=float(args.meshgs_scale_multiplier),
        thickness_multiplier=float(args.meshgs_thickness_multiplier),
        init_opacity=float(args.meshgs_init_opacity),
        max_disagreement=float(args.max_disagreement),
        init_color_source=str(args.init_color_source),
        init_color_gray_value=float(args.init_color_gray_value),
    )
    copy_render_config(camera_model_path, output_model_path)

    global_to_local = np.full((patch_centers.shape[0],), -1, dtype=np.int64)
    global_to_local[selected_indices] = np.arange(selected_indices.shape[0], dtype=np.int64)
    view_records = build_local_view_records(full_view_records, global_to_local, device=device)

    surface_vertices_init = meshgs._surface_vertices.detach().clone()
    anchor_rgb = current_dc_rgb(meshgs).detach().clone()
    appearance_optimizer = (
        torch.optim.Adam([{"params": [meshgs._features_dc], "lr": float(args.appearance_lr), "name": "f_dc"}], eps=1e-15)
        if float(args.appearance_lr) > 0.0
        else None
    )
    structure_optimizer = (
        torch.optim.Adam([{"params": [meshgs._surface_vertices], "lr": float(args.structure_lr), "name": "surface_vertices"}], eps=1e-15)
        if float(args.structure_lr) > 0.0
        else None
    )
    cycle_plan = build_cycle_plan(
        cycles=int(args.cycles),
        total_steps=int(args.total_steps),
        appearance_steps=int(args.appearance_steps),
        structure_steps=int(args.structure_steps),
    )
    executed_total_steps = 0

    cycle_summaries: List[Dict[str, object]] = []
    local_patch_bank = subset_patch_bank_geometry(patch_bank, selected_indices)
    for planned in cycle_plan:
        cycle_idx = int(planned["cycle"])
        cycle_a_steps = int(planned["appearance_steps"])
        cycle_b_steps = int(planned["structure_steps"])
        cycle_step_begin = int(planned["step_begin"])
        appearance_step_end = int(cycle_step_begin + max(cycle_a_steps, 0) - 1) if cycle_a_steps > 0 else int(cycle_step_begin - 1)
        structure_step_begin = int(appearance_step_end + 1)
        structure_step_end = int(structure_step_begin + max(cycle_b_steps, 0) - 1) if cycle_b_steps > 0 else int(structure_step_begin - 1)
        local_payload, local_summary = aggregate_surface_targets(
            xyz_world=meshgs.get_xyz.detach(),
            view_records=view_records,
            image_cache=image_cache,
            min_views=int(args.min_views),
            min_confidence=float(args.min_confidence),
            max_disagreement=float(args.max_disagreement),
            depth_min=float(args.depth_min),
            anchor_lowfreq_threshold=float(args.anchor_lowfreq_threshold),
        )
        local_payload = merge_geometry_fields(local_payload, local_patch_bank)
        local_payload_path = output_model_path / f"carrier_payload_cycle{cycle_idx:02d}_before_b.npz"
        save_payload_npz(local_payload_path, local_payload)

        target_rgb = torch.from_numpy(local_payload["fused_rgb"].astype(np.float32, copy=False)).to(device=device)
        target_weight = torch.from_numpy(local_payload["confidence"].astype(np.float32, copy=False)).to(device=device)
        target_weight = torch.clamp(target_weight, min=0.0)
        appearance_summary = optimize_appearance(
            meshgs=meshgs,
            target_rgb=target_rgb,
            target_weight=target_weight,
            anchor_rgb=anchor_rgb,
            steps=cycle_a_steps,
            lr=float(args.appearance_lr),
            lambda_anchor=float(args.appearance_anchor_lambda),
            optimizer=appearance_optimizer,
            progress_label=f"appearance-only A [{cycle_step_begin}-{appearance_step_end}]",
        )

        fixed_rgb = current_dc_rgb(meshgs).detach()
        structure_summary = optimize_structure(
            meshgs=meshgs,
            view_records=view_records,
            image_cache=image_cache,
            fixed_rgb=fixed_rgb,
            carrier_weight=torch.clamp(target_weight, min=0.0),
            surface_vertices_init=surface_vertices_init,
            steps=cycle_b_steps,
            lr=float(args.structure_lr),
            lambda_photo=float(args.lambda_structure_photo),
            lambda_mv=float(args.lambda_structure_mv),
            lambda_delta=float(args.lambda_structure_delta),
            lambda_normal=float(args.lambda_structure_normal),
            depth_min=float(args.depth_min),
            max_surface_vertex_displacement=float(args.max_surface_vertex_displacement),
            anchor_lowfreq_threshold=float(args.anchor_lowfreq_threshold),
            step_view_limit=int(args.structure_view_limit),
            seed=int(args.seed + cycle_idx),
            optimizer=structure_optimizer,
            progress_label=f"struct-only B [{structure_step_begin}-{structure_step_end}]",
        )

        refreshed_payload, refreshed_summary = aggregate_surface_targets(
            xyz_world=meshgs.get_xyz.detach(),
            view_records=view_records,
            image_cache=image_cache,
            min_views=int(args.min_views),
            min_confidence=float(args.min_confidence),
            max_disagreement=float(args.max_disagreement),
            depth_min=float(args.depth_min),
            anchor_lowfreq_threshold=float(args.anchor_lowfreq_threshold),
        )
        refreshed_payload = merge_geometry_fields(refreshed_payload, local_patch_bank)
        refreshed_payload_path = output_model_path / f"carrier_payload_cycle{cycle_idx:02d}_after_b.npz"
        save_payload_npz(refreshed_payload_path, refreshed_payload)
        executed_total_steps += int(cycle_a_steps + cycle_b_steps)
        if int(args.save_every_cycles) > 0 and cycle_idx % int(args.save_every_cycles) == 0:
            save_meshgs_checkpoint(meshgs, output_model_path, executed_total_steps)

        cycle_summaries.append(
            {
                "cycle": int(cycle_idx),
                "step_begin": int(planned["step_begin"]),
                "step_end": int(planned["step_end"]),
                "appearance_step_begin": int(cycle_step_begin),
                "appearance_step_end": int(appearance_step_end),
                "structure_step_begin": int(structure_step_begin),
                "structure_step_end": int(structure_step_end),
                "aggregation_before_b": local_summary,
                "appearance": appearance_summary,
                "structure": structure_summary,
                "aggregation_after_b": refreshed_summary,
                "payload_before_b": str(local_payload_path),
                "payload_after_b": str(refreshed_payload_path),
            }
        )

    output_iteration = int(args.output_iteration) if int(args.output_iteration) > 0 else int(executed_total_steps if int(args.total_steps) > 0 else len(cycle_plan))
    final_summary = {
        "version": "alternating_prior_surface_v0",
        "scene_root": str(scene_root),
        "camera_model_path": str(camera_model_path),
        "mesh_path": str(mesh_path),
        "patch_observation_root": str(patch_root),
        "prior_dir": str(prior_dir),
        "anchor_dir": str(anchor_dir) if anchor_dir is not None else None,
        "output_model_path": str(output_model_path),
        "inputs": {
            "images_subdir": str(args.images_subdir),
            "split": str(args.split),
            "requested_max_views": int(args.max_views),
            "selected_views": [normalize_image_name(str(record["image_name"])) for record in full_view_records],
        },
        "selection": {
            "initial_aggregation": initial_summary,
            "selected_global_carriers": int(selected_indices.size),
            "selected_global_indices_path": str(output_model_path / "selected_global_patch_ids.json"),
            **init_summary,
        },
        "params": vars(args),
        "schedule": {
            "planned_cycles": int(len(cycle_plan)),
            "executed_cycles": int(len(cycle_summaries)),
            "planned_total_steps": int(sum(int(item["appearance_steps"]) + int(item["structure_steps"]) for item in cycle_plan)),
            "executed_total_steps": int(executed_total_steps),
            "appearance_block_steps": int(args.appearance_steps),
            "structure_block_steps": int(args.structure_steps),
        },
        "cycles": cycle_summaries,
        "artifacts": {
            "initial_payload": str(initial_payload_path),
            "final_point_cloud_dir": str(output_model_path / "point_cloud" / f"iteration_{output_iteration}"),
            "final_point_cloud_role": "carrier_layer_only",
            "final_eval_note": "Merge or compose this sparse carrier layer with a full base GS model for final-scene rendering; standalone renders are expected to be mostly black.",
        },
    }

    (output_model_path / "selected_global_patch_ids.json").write_text(
        json.dumps(selected_indices.tolist(), indent=2),
        encoding="utf-8",
    )
    save_sugar_like_meshgs(meshgs, output_model_path, output_iteration, args, final_summary)
    summary_path = output_model_path / "alternating_prior_surface_v0_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))
    print(f"[alternating-prior-surface-v0] summary : {summary_path}")
    print(f"[alternating-prior-surface-v0] output  : {output_model_path / 'point_cloud' / f'iteration_{output_iteration}' / 'point_cloud.ply'}")


if __name__ == "__main__":
    main()
