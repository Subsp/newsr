from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import load_cameras_for_split, load_model_ply, resolve_iteration, select_uniform
from utils.general_utils import build_rotation, safe_state
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.sh_utils import SH2RGB


def is_camera_source_root(path: Path) -> bool:
    return (path / "sparse").exists() or (path / "transforms_train.json").is_file()


def resolve_scene_root(args) -> Path:
    if getattr(args, "scene_root", None):
        return Path(args.scene_root).expanduser().resolve()
    prior_root = Path(args.prior_dir).expanduser().resolve()
    base_root = prior_root.parent
    if is_camera_source_root(base_root):
        return base_root
    child_candidates = sorted(path for path in base_root.iterdir() if path.is_dir())
    matched = [path for path in child_candidates if is_camera_source_root(path)]
    if len(matched) == 1:
        return matched[0]
    return base_root


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


def load_image_chw(path: Path, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32)


def blur_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0]


class ImageCache:
    def __init__(self, device: torch.device, lowpass_kernel: int):
        self.device = device
        self.lowpass_kernel = int(lowpass_kernel)
        self.rgb_cache: Dict[Path, torch.Tensor] = {}
        self.low_cache: Dict[Path, torch.Tensor] = {}
        self.high_cache: Dict[Path, torch.Tensor] = {}

    def load_rgb(self, path: Path) -> torch.Tensor:
        tensor = self.rgb_cache.get(path)
        if tensor is None:
            tensor = load_image_chw(path, self.device)
            self.rgb_cache[path] = tensor
        return tensor

    def load_low(self, path: Path) -> torch.Tensor:
        tensor = self.low_cache.get(path)
        if tensor is None:
            tensor = blur_chw(self.load_rgb(path), self.lowpass_kernel)
            self.low_cache[path] = tensor
        return tensor

    def load_high(self, path: Path) -> torch.Tensor:
        tensor = self.high_cache.get(path)
        if tensor is None:
            tensor = self.load_rgb(path) - self.load_low(path)
            self.high_cache[path] = tensor
        return tensor


def sample_image_at_xy(image_chw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if xy.numel() == 0:
        return torch.zeros((0, image_chw.shape[0]), dtype=image_chw.dtype, device=image_chw.device)
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


def sample_scalar_at_xy(image_hw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    sampled = sample_image_at_xy(image_hw.unsqueeze(0), xy)
    return sampled[:, 0]


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


def extract_anchor_rgb(model: GaussianModel) -> torch.Tensor:
    dc = model._features_dc.detach()
    if dc.ndim == 3:
        if dc.shape[1] == 1 and dc.shape[2] == 3:
            return torch.clamp(SH2RGB(dc[:, 0, :]), 0.0, 1.0)
        if dc.shape[1] == 3 and dc.shape[2] == 1:
            return torch.clamp(SH2RGB(dc[:, :, 0]), 0.0, 1.0)
        raise RuntimeError(f"Unsupported _features_dc shape for anchor extraction: {tuple(dc.shape)}")
    if dc.ndim == 2 and dc.shape[1] == 3:
        if bool(model.use_SBs):
            return torch.clamp(dc, 0.0, 1.0)
        return torch.clamp(SH2RGB(dc), 0.0, 1.0)
    raise RuntimeError(f"Unsupported _features_dc shape for anchor extraction: {tuple(dc.shape)}")


def smooth_surface_field_knn(
    centers: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    value: np.ndarray,
    support: np.ndarray,
    *,
    k: int,
    radius_scale: float,
    normal_min_cos: float,
    blend: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if cKDTree is None or int(k) <= 1 or int(centers.shape[0]) <= 1:
        return value.astype(np.float32, copy=False), support.astype(np.float32, copy=False)

    num_points = int(centers.shape[0])
    query_k = max(2, min(int(k), num_points))
    tree = cKDTree(np.asarray(centers, dtype=np.float32))
    dist, nbr = tree.query(np.asarray(centers, dtype=np.float32), k=query_k, workers=1)
    dist = np.asarray(dist, dtype=np.float32)
    nbr = np.asarray(nbr, dtype=np.int64)
    if dist.ndim == 1:
        dist = dist[:, None]
        nbr = nbr[:, None]

    radii = np.maximum(np.asarray(radii, dtype=np.float32), 1e-6)
    support = np.clip(np.asarray(support, dtype=np.float32), 0.0, 1.0)
    value = np.asarray(value, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    sigma = np.maximum(radii[:, None] * float(radius_scale), 1e-6)
    normal_cos = np.sum(normals[:, None, :] * normals[nbr], axis=2)
    normal_cos = np.clip(normal_cos, 0.0, 1.0)
    distance_weight = np.exp(-(dist / sigma) ** 2).astype(np.float32, copy=False)
    valid = dist <= sigma
    valid &= normal_cos >= float(normal_min_cos)
    valid[:, 0] = True
    base_weight = distance_weight * valid.astype(np.float32, copy=False)
    normal_weight = np.clip(
        (normal_cos - float(normal_min_cos)) / max(1.0 - float(normal_min_cos), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    normal_weight[:, 0] = 1.0
    base_weight *= np.maximum(normal_weight, 1e-4)

    neighbor_support = support[nbr]
    support_sum = np.clip(base_weight.sum(axis=1), 1e-6, None)
    smoothed_support = (base_weight * neighbor_support).sum(axis=1) / support_sum

    support_weight = base_weight * neighbor_support
    support_weight_sum = np.clip(support_weight.sum(axis=1), 1e-6, None)
    if value.ndim == 1:
        smoothed_value = (support_weight * value[nbr]).sum(axis=1) / support_weight_sum
    else:
        smoothed_value = (support_weight[:, :, None] * value[nbr]).sum(axis=1) / support_weight_sum[:, None]

    alpha = float(np.clip(blend, 0.0, 1.0))
    blended_value = ((1.0 - alpha) * value + alpha * smoothed_value).astype(np.float32, copy=False)
    blended_support = ((1.0 - alpha) * support + alpha * smoothed_support).astype(np.float32, copy=False)
    return blended_value, np.clip(blended_support, 0.0, 1.0).astype(np.float32, copy=False)


def initialize_consensus_state(num_items: int, feature_dim: int) -> Dict[str, np.ndarray]:
    return {
        "best_feat": np.zeros((num_items, feature_dim), dtype=np.float32),
        "best_weight": np.zeros((num_items,), dtype=np.float32),
        "best_count": np.zeros((num_items,), dtype=np.int32),
        "best_peak_weight": np.zeros((num_items,), dtype=np.float32),
        "best_view_id": np.full((num_items,), -1, dtype=np.int32),
        "alt_feat": np.zeros((num_items, feature_dim), dtype=np.float32),
        "alt_weight": np.zeros((num_items,), dtype=np.float32),
        "alt_count": np.zeros((num_items,), dtype=np.int32),
        "alt_peak_weight": np.zeros((num_items,), dtype=np.float32),
        "alt_view_id": np.full((num_items,), -1, dtype=np.int32),
        "total_weight": np.zeros((num_items,), dtype=np.float32),
        "sample_count": np.zeros((num_items,), dtype=np.int32),
    }


def update_consensus_state(
    state: Dict[str, np.ndarray],
    item_ids: np.ndarray,
    sample_feat: np.ndarray,
    sample_weight: np.ndarray,
    *,
    view_id: int,
    distance_threshold: float,
) -> None:
    if item_ids.size == 0:
        return

    ids = np.asarray(item_ids, dtype=np.int64)
    feat = np.asarray(sample_feat, dtype=np.float32)
    weight = np.asarray(sample_weight, dtype=np.float32)

    state["total_weight"][ids] += weight
    state["sample_count"][ids] += 1

    best_empty = state["best_count"][ids] == 0
    if np.any(best_empty):
        ids_assign = ids[best_empty]
        state["best_feat"][ids_assign] = feat[best_empty]
        state["best_weight"][ids_assign] = weight[best_empty]
        state["best_count"][ids_assign] = 1
        state["best_peak_weight"][ids_assign] = weight[best_empty]
        state["best_view_id"][ids_assign] = int(view_id)

    remaining = ~best_empty
    if not np.any(remaining):
        return

    ids_r = ids[remaining]
    feat_r = feat[remaining]
    weight_r = weight[remaining]

    best_delta = np.mean(np.abs(feat_r - state["best_feat"][ids_r]), axis=1)
    match_best = best_delta <= float(distance_threshold)
    if np.any(match_best):
        ids_match = ids_r[match_best]
        prev_weight = state["best_weight"][ids_match]
        add_weight = weight_r[match_best]
        merged = (
            state["best_feat"][ids_match] * prev_weight[:, None] + feat_r[match_best] * add_weight[:, None]
        ) / np.clip(prev_weight[:, None] + add_weight[:, None], 1e-8, None)
        state["best_feat"][ids_match] = merged
        state["best_weight"][ids_match] = prev_weight + add_weight
        state["best_count"][ids_match] += 1
        replace_peak = add_weight > state["best_peak_weight"][ids_match]
        if np.any(replace_peak):
            ids_peak = ids_match[replace_peak]
            state["best_peak_weight"][ids_peak] = add_weight[replace_peak]
            state["best_view_id"][ids_peak] = int(view_id)

    non_best = ~match_best
    if not np.any(non_best):
        return

    ids_r = ids_r[non_best]
    feat_r = feat_r[non_best]
    weight_r = weight_r[non_best]

    alt_empty = state["alt_count"][ids_r] == 0
    if np.any(alt_empty):
        ids_assign = ids_r[alt_empty]
        state["alt_feat"][ids_assign] = feat_r[alt_empty]
        state["alt_weight"][ids_assign] = weight_r[alt_empty]
        state["alt_count"][ids_assign] = 1
        state["alt_peak_weight"][ids_assign] = weight_r[alt_empty]
        state["alt_view_id"][ids_assign] = int(view_id)

    remaining_alt = ~alt_empty
    if not np.any(remaining_alt):
        return

    ids_r = ids_r[remaining_alt]
    feat_r = feat_r[remaining_alt]
    weight_r = weight_r[remaining_alt]

    alt_delta = np.mean(np.abs(feat_r - state["alt_feat"][ids_r]), axis=1)
    match_alt = alt_delta <= float(distance_threshold)
    if np.any(match_alt):
        ids_match = ids_r[match_alt]
        prev_weight = state["alt_weight"][ids_match]
        add_weight = weight_r[match_alt]
        merged = (
            state["alt_feat"][ids_match] * prev_weight[:, None] + feat_r[match_alt] * add_weight[:, None]
        ) / np.clip(prev_weight[:, None] + add_weight[:, None], 1e-8, None)
        state["alt_feat"][ids_match] = merged
        state["alt_weight"][ids_match] = prev_weight + add_weight
        state["alt_count"][ids_match] += 1
        replace_peak = add_weight > state["alt_peak_weight"][ids_match]
        if np.any(replace_peak):
            ids_peak = ids_match[replace_peak]
            state["alt_peak_weight"][ids_peak] = add_weight[replace_peak]
            state["alt_view_id"][ids_peak] = int(view_id)

    replace_alt = ~match_alt
    if np.any(replace_alt):
        ids_replace = ids_r[replace_alt]
        stronger = weight_r[replace_alt] > state["alt_weight"][ids_replace]
        if np.any(stronger):
            ids_stronger = ids_replace[stronger]
            feat_stronger = feat_r[replace_alt][stronger]
            weight_stronger = weight_r[replace_alt][stronger]
            state["alt_feat"][ids_stronger] = feat_stronger
            state["alt_weight"][ids_stronger] = weight_stronger
            state["alt_count"][ids_stronger] = 1
            state["alt_peak_weight"][ids_stronger] = weight_stronger
            state["alt_view_id"][ids_stronger] = int(view_id)


def finalize_consensus_state(
    state: Dict[str, np.ndarray],
    *,
    min_views: int,
    min_confidence: float,
) -> Dict[str, np.ndarray]:
    choose_alt = state["alt_weight"] > state["best_weight"]
    consensus_feat = state["best_feat"].copy()
    consensus_weight = state["best_weight"].copy()
    consensus_peak_view = state["best_view_id"].copy()
    consensus_cluster_count = state["best_count"].copy()

    if np.any(choose_alt):
        consensus_feat[choose_alt] = state["alt_feat"][choose_alt]
        consensus_weight[choose_alt] = state["alt_weight"][choose_alt]
        consensus_peak_view[choose_alt] = state["alt_view_id"][choose_alt]
        consensus_cluster_count[choose_alt] = state["alt_count"][choose_alt]

    total_weight = np.clip(state["total_weight"], 1e-8, None)
    support_ratio = consensus_weight / total_weight
    coverage_factor = np.clip(state["sample_count"].astype(np.float32) / max(float(min_views), 1.0), 0.0, 1.0)
    confidence = support_ratio * coverage_factor
    disagreement = 1.0 - support_ratio
    mode_count = np.zeros_like(state["sample_count"], dtype=np.int32)
    mode_count[state["best_weight"] > 0.0] += 1
    mode_count[state["alt_weight"] > 0.0] += 1
    valid_mask = (
        (state["sample_count"] >= int(min_views))
        & (consensus_cluster_count > 0)
        & (confidence >= float(min_confidence))
    )
    return {
        "consensus_feat": consensus_feat.astype(np.float32, copy=False),
        "confidence": confidence.astype(np.float32, copy=False),
        "disagreement": disagreement.astype(np.float32, copy=False),
        "support_count": state["sample_count"].astype(np.int32, copy=False),
        "cluster_support_count": consensus_cluster_count.astype(np.int32, copy=False),
        "support_ratio": support_ratio.astype(np.float32, copy=False),
        "mode_count": mode_count.astype(np.int32, copy=False),
        "exemplar_view_id": consensus_peak_view.astype(np.int32, copy=False),
        "valid_mask": valid_mask.astype(bool, copy=False),
    }


def build_patch_offsets(pattern: str, radius_scale: float) -> np.ndarray:
    radius = float(radius_scale)
    if str(pattern) == "cross5":
        return np.asarray(
            [
                [0.0, 0.0],
                [radius, 0.0],
                [-radius, 0.0],
                [0.0, radius],
                [0.0, -radius],
            ],
            dtype=np.float32,
        )
    if str(pattern) == "grid3":
        coords = [-radius, 0.0, radius]
        return np.asarray([[u, v] for v in coords for u in coords], dtype=np.float32)
    raise ValueError(f"Unsupported patch pattern: {pattern}")


def extract_surface_carriers(
    model: GaussianModel,
    *,
    min_opacity: float,
    max_thickness_ratio: float,
    max_count: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        xyz = model.get_xyz.detach()
        scales = model.get_scaling.detach()
        rotations = build_rotation(model.get_rotation.detach())
        opacity = model.get_opacity.detach().reshape(-1)
        source_tag = model._source_tag.detach().reshape(-1)
        seed_id = model._seed_id.detach().reshape(-1)
        generation = model._generation.detach().reshape(-1)

        axis_order = torch.argsort(scales, dim=1, descending=True)
        major_axis = axis_order[:, 0]
        minor_axis = axis_order[:, 1]
        normal_axis = axis_order[:, 2]

        batch_ids = torch.arange(scales.shape[0], device=scales.device)
        scale_u = scales[batch_ids, major_axis]
        scale_v = scales[batch_ids, minor_axis]
        scale_n = scales[batch_ids, normal_axis]
        thickness_ratio = scale_n / torch.clamp(0.5 * (scale_u + scale_v), min=1e-8)

        tangent_u = rotations[batch_ids, :, major_axis]
        tangent_v = rotations[batch_ids, :, minor_axis]
        normal = rotations[batch_ids, :, normal_axis]

        valid = opacity >= float(min_opacity)
        if float(max_thickness_ratio) > 0.0:
            valid &= thickness_ratio <= float(max_thickness_ratio)

        indices = torch.nonzero(valid).reshape(-1)
        if int(max_count) > 0 and int(indices.numel()) > int(max_count):
            rng = np.random.default_rng(int(seed))
            chosen = np.sort(rng.choice(indices.detach().cpu().numpy(), size=int(max_count), replace=False))
            indices = torch.from_numpy(chosen).to(device=indices.device, dtype=indices.dtype)

        if int(indices.numel()) <= 0:
            raise RuntimeError("No SOF Gaussian carriers remain after opacity/thickness filtering.")

        chosen_seed_id = seed_id[indices].clone()
        fallback = chosen_seed_id < 0
        if torch.any(fallback):
            chosen_seed_id[fallback] = indices[fallback].to(dtype=chosen_seed_id.dtype)

        return {
            "gaussian_ids": indices.detach().cpu().numpy().astype(np.int64, copy=False),
            "seed_ids": chosen_seed_id.detach().cpu().numpy().astype(np.int64, copy=False),
            "source_tag": source_tag[indices].detach().cpu().numpy().astype(np.int32, copy=False),
            "generation": generation[indices].detach().cpu().numpy().astype(np.int32, copy=False),
            "xyz": xyz[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "tangent_u": tangent_u[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "tangent_v": tangent_v[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "normal": normal[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "scale_u": scale_u[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "scale_v": scale_v[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "scale_n": scale_n[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "opacity": opacity[indices].detach().cpu().numpy().astype(np.float32, copy=False),
            "thickness_ratio": thickness_ratio[indices].detach().cpu().numpy().astype(np.float32, copy=False),
        }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Gaussian-centric multiview SR surface response payload from a SOF field.")
    parser.add_argument("--scene_root", default="")
    parser.add_argument("--camera_model_path", default="")
    parser.add_argument("--mesh_path", default="")
    parser.add_argument("--sof_model_path", required=True)
    parser.add_argument("--sof_iteration", type=int, default=-1)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--carrier_min_opacity", type=float, default=0.05)
    parser.add_argument("--carrier_max_thickness_ratio", type=float, default=0.35)
    parser.add_argument("--carrier_max_count", type=int, default=0)
    parser.add_argument("--carrier_seed", type=int, default=0)
    parser.add_argument("--lowpass_kernel", type=int, default=15)
    parser.add_argument("--patch_pattern", choices=["cross5", "grid3"], default="cross5")
    parser.add_argument("--patch_radius_scale", type=float, default=0.5)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--render_depth_abs_tolerance", type=float, default=0.03)
    parser.add_argument("--render_depth_rel_tolerance", type=float, default=0.02)
    parser.add_argument("--min_render_alpha", type=float, default=0.05)
    parser.add_argument("--min_screen_radius", type=float, default=0.5)
    parser.add_argument("--min_frontality", type=float, default=0.2)
    parser.add_argument("--max_radius_weight", type=float, default=4.0)
    parser.add_argument("--low_color_threshold", type=float, default=0.08)
    parser.add_argument("--high_patch_threshold", type=float, default=0.05)
    parser.add_argument("--min_views_low", type=int, default=2)
    parser.add_argument("--min_views_high", type=int, default=2)
    parser.add_argument("--min_low_confidence", type=float, default=0.05)
    parser.add_argument("--min_high_confidence", type=float, default=0.05)
    parser.add_argument("--detail_gamma", type=float, default=0.30)
    parser.add_argument("--high_clamp", type=float, default=0.06)
    parser.add_argument("--risk_high_energy_scale", type=float, default=0.04)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fusion_mode", choices=["anchor_residual", "absolute_consensus"], default="anchor_residual")
    parser.add_argument("--low_residual_clamp", type=float, default=0.07)
    parser.add_argument("--low_confidence_power", type=float, default=2.0)
    parser.add_argument("--detail_confidence_power", type=float, default=1.5)
    parser.add_argument("--disable_detail_risk_suppression", action="store_true")
    parser.add_argument("--low_injection_scale", type=float, default=0.75)
    parser.add_argument("--spatial_smooth_knn", type=int, default=8)
    parser.add_argument("--spatial_smooth_radius_scale", type=float, default=4.0)
    parser.add_argument("--spatial_smooth_normal_min_cos", type=float, default=0.55)
    parser.add_argument("--spatial_smooth_blend", type=float, default=0.75)
    parser.add_argument("--disable_spatial_smoothing", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    safe_state(bool(args.quiet))
    suppress_detail_by_risk = not bool(args.disable_detail_risk_suppression)

    if not torch.cuda.is_available():
        raise RuntimeError("prepare_sof_gaussian_response_v0 currently requires CUDA.")

    device = torch.device("cuda")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_dir = Path(args.prior_dir).expanduser().resolve()
    sof_model_path = Path(args.sof_model_path).expanduser().resolve()
    scene_root = resolve_scene_root(args)
    camera_model_path = resolve_camera_model_path(args, sof_model_path)
    mesh_path = Path(args.mesh_path).expanduser().resolve() if str(args.mesh_path).strip() else None

    if not scene_root.is_dir():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")
    if not camera_model_path.is_dir():
        raise FileNotFoundError(f"camera_model_path not found: {camera_model_path}")
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_dir not found: {prior_dir}")
    if not sof_model_path.is_dir():
        raise FileNotFoundError(f"sof_model_path not found: {sof_model_path}")
    if mesh_path is not None and not mesh_path.is_file():
        raise FileNotFoundError(f"mesh_path not found: {mesh_path}")

    sof_iteration = resolve_iteration(sof_model_path, int(args.sof_iteration))
    cameras = load_cameras_for_split(scene_root, camera_model_path, str(args.images_subdir), str(args.split))
    if int(args.max_views) > 0:
        cameras = select_uniform(cameras, int(args.max_views))
    if not cameras:
        raise RuntimeError("No cameras selected for SOF Gaussian response preparation.")

    model = load_model_ply(sof_model_path, sof_iteration, sh_degree=3)
    carriers = extract_surface_carriers(
        model,
        min_opacity=float(args.carrier_min_opacity),
        max_thickness_ratio=float(args.carrier_max_thickness_ratio),
        max_count=int(args.carrier_max_count),
        seed=int(args.carrier_seed),
    )

    carrier_ids_np = carriers["gaussian_ids"]
    carrier_count = int(carrier_ids_np.shape[0])
    carrier_ids_t = torch.from_numpy(carrier_ids_np).to(device=device, dtype=torch.long)
    anchor_rgb_all_t = extract_anchor_rgb(model)
    anchor_rgb_t = anchor_rgb_all_t[carrier_ids_t]
    centers_t = torch.from_numpy(carriers["xyz"]).to(device=device, dtype=torch.float32)
    tangent_u_t = torch.from_numpy(carriers["tangent_u"]).to(device=device, dtype=torch.float32)
    tangent_v_t = torch.from_numpy(carriers["tangent_v"]).to(device=device, dtype=torch.float32)
    normal_t = torch.from_numpy(carriers["normal"]).to(device=device, dtype=torch.float32)
    scale_u_t = torch.from_numpy(carriers["scale_u"]).to(device=device, dtype=torch.float32)
    scale_v_t = torch.from_numpy(carriers["scale_v"]).to(device=device, dtype=torch.float32)

    patch_offsets_np = build_patch_offsets(str(args.patch_pattern), float(args.patch_radius_scale))
    patch_offsets_t = torch.from_numpy(patch_offsets_np).to(device=device, dtype=torch.float32)
    patch_point_count = int(patch_offsets_np.shape[0])

    low_state = initialize_consensus_state(carrier_count, 3)
    high_state = initialize_consensus_state(carrier_count, patch_point_count * 3)
    sample_frontality_sum = np.zeros((carrier_count,), dtype=np.float32)
    sample_radius_sum = np.zeros((carrier_count,), dtype=np.float32)
    sample_alpha_sum = np.zeros((carrier_count,), dtype=np.float32)
    sample_depth_delta_sum = np.zeros((carrier_count,), dtype=np.float32)
    visible_view_count = np.zeros((carrier_count,), dtype=np.int32)
    missing_prior_views: List[str] = []
    view_summaries: List[Dict[str, object]] = []

    prior_index = index_image_dir(str(prior_dir))
    image_cache = ImageCache(device=device, lowpass_kernel=int(args.lowpass_kernel))
    background = torch.zeros((3,), dtype=torch.float32, device=device)

    from gaussian_renderer import render_simple

    view_iter = tqdm(cameras, desc="SOF gaussian response", disable=bool(args.quiet), dynamic_ncols=True)
    for view_idx, camera in enumerate(view_iter):
        prior_path = lookup_indexed_path(prior_index, str(camera.image_name))
        if prior_path is None:
            if len(missing_prior_views) < 64:
                missing_prior_views.append(str(camera.image_name))
            continue

        with torch.no_grad():
            render_pkg = render_simple(camera, model, background)
            visible_global = render_pkg["visibility_filter"].reshape(-1)
            radii_global = render_pkg["radii"].reshape(-1).float()
            depth_img = render_pkg["depth"][0]
            alpha_img = render_pkg["alpha"][0].clamp(0.0, 1.0)

            visible_selected = visible_global[carrier_ids_t]
            radius_selected = radii_global[carrier_ids_t]
            candidate_mask = visible_selected & (radius_selected >= float(args.min_screen_radius))
            candidate_ids = torch.nonzero(candidate_mask).reshape(-1)
            if int(candidate_ids.numel()) <= 0:
                view_summaries.append(
                    {
                        "image_name": normalize_image_name(str(camera.image_name)),
                        "prior_path": str(prior_path),
                        "visible_candidates": 0,
                        "valid_low": 0,
                        "valid_high": 0,
                    }
                )
                continue

            centers = centers_t[candidate_ids]
            projected_center, center_valid = project_points_camera_torch(camera, centers, depth_min=float(args.depth_min))
            if not torch.any(center_valid):
                view_summaries.append(
                    {
                        "image_name": normalize_image_name(str(camera.image_name)),
                        "prior_path": str(prior_path),
                        "visible_candidates": int(candidate_ids.numel()),
                        "valid_low": 0,
                        "valid_high": 0,
                    }
                )
                continue

            candidate_ids = candidate_ids[center_valid]
            projected_center = projected_center[center_valid]
            centers = centers[center_valid]
            tangent_u = tangent_u_t[candidate_ids]
            tangent_v = tangent_v_t[candidate_ids]
            normals = normal_t[candidate_ids]
            scale_u = scale_u_t[candidate_ids]
            scale_v = scale_v_t[candidate_ids]
            radius_px = radius_selected[candidate_ids]

            view_dir = torch.as_tensor(camera.camera_center, device=device, dtype=torch.float32).reshape(1, 3) - centers
            view_dir = view_dir / torch.clamp(torch.norm(view_dir, dim=1, keepdim=True), min=1e-6)
            frontality = torch.abs(torch.sum(normals * view_dir, dim=1))
            front_ok = frontality >= float(args.min_frontality)

            alpha_at_center = sample_scalar_at_xy(alpha_img, projected_center[:, :2])
            depth_at_center = sample_scalar_at_xy(depth_img, projected_center[:, :2])
            depth_delta = torch.abs(depth_at_center - projected_center[:, 2])
            depth_limit = torch.max(
                torch.full_like(depth_delta, float(args.render_depth_abs_tolerance)),
                projected_center[:, 2] * float(args.render_depth_rel_tolerance),
            )
            visibility_ok = alpha_at_center >= float(args.min_render_alpha)
            visibility_ok &= depth_delta <= depth_limit
            low_ok = front_ok & visibility_ok

            if not torch.any(low_ok):
                view_summaries.append(
                    {
                        "image_name": normalize_image_name(str(camera.image_name)),
                        "prior_path": str(prior_path),
                        "visible_candidates": int(candidate_ids.numel()),
                        "valid_low": 0,
                        "valid_high": 0,
                    }
                )
                continue

            candidate_ids_low = candidate_ids[low_ok]
            projected_low = projected_center[low_ok]
            frontality_low = frontality[low_ok]
            radius_low = radius_px[low_ok]
            alpha_low = alpha_at_center[low_ok]
            depth_delta_low = depth_delta[low_ok]

            visible_view_count[candidate_ids_low.detach().cpu().numpy().astype(np.int64, copy=False)] += 1
            sample_frontality_sum[candidate_ids_low.detach().cpu().numpy().astype(np.int64, copy=False)] += (
                frontality_low.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            sample_radius_sum[candidate_ids_low.detach().cpu().numpy().astype(np.int64, copy=False)] += (
                radius_low.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            sample_alpha_sum[candidate_ids_low.detach().cpu().numpy().astype(np.int64, copy=False)] += (
                alpha_low.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            sample_depth_delta_sum[candidate_ids_low.detach().cpu().numpy().astype(np.int64, copy=False)] += (
                depth_delta_low.detach().cpu().numpy().astype(np.float32, copy=False)
            )

            prior_low = image_cache.load_low(prior_path)
            prior_high = image_cache.load_high(prior_path)
            low_rgb = sample_image_at_xy(prior_low, projected_low[:, :2])
            if str(args.fusion_mode) == "anchor_residual":
                low_feat = low_rgb - anchor_rgb_t[candidate_ids_low]
            else:
                low_feat = low_rgb
            radius_weight = torch.clamp(torch.sqrt(torch.clamp(radius_low, min=1e-4)), min=1.0, max=float(args.max_radius_weight))
            low_weight = torch.clamp(frontality_low * radius_weight, min=1e-4)

            update_consensus_state(
                low_state,
                candidate_ids_low.detach().cpu().numpy().astype(np.int64, copy=False),
                low_feat.detach().cpu().numpy().astype(np.float32, copy=False),
                low_weight.detach().cpu().numpy().astype(np.float32, copy=False),
                view_id=int(view_idx),
                distance_threshold=float(args.low_color_threshold),
            )

            offsets_u = patch_offsets_t[:, 0].view(1, patch_point_count, 1) * scale_u_t[candidate_ids_low].view(-1, 1, 1)
            offsets_v = patch_offsets_t[:, 1].view(1, patch_point_count, 1) * scale_v_t[candidate_ids_low].view(-1, 1, 1)
            patch_world = (
                centers_t[candidate_ids_low].view(-1, 1, 3)
                + tangent_u_t[candidate_ids_low].view(-1, 1, 3) * offsets_u
                + tangent_v_t[candidate_ids_low].view(-1, 1, 3) * offsets_v
            )
            flat_patch = patch_world.reshape(-1, 3)
            projected_patch, patch_valid_flat = project_points_camera_torch(camera, flat_patch, depth_min=float(args.depth_min))
            patch_valid = patch_valid_flat.view(-1, patch_point_count).all(dim=1)
            high_ok = patch_valid

            if torch.any(high_ok):
                candidate_ids_high = candidate_ids_low[high_ok]
                projected_patch = projected_patch.view(-1, patch_point_count, 3)[high_ok]
                flat_xy = projected_patch[:, :, :2].reshape(-1, 2)
                high_patch = sample_image_at_xy(prior_high, flat_xy).reshape(-1, patch_point_count * 3)
                high_weight = low_weight[high_ok]
                update_consensus_state(
                    high_state,
                    candidate_ids_high.detach().cpu().numpy().astype(np.int64, copy=False),
                    high_patch.detach().cpu().numpy().astype(np.float32, copy=False),
                    high_weight.detach().cpu().numpy().astype(np.float32, copy=False),
                    view_id=int(view_idx),
                    distance_threshold=float(args.high_patch_threshold),
                )
                valid_high_count = int(candidate_ids_high.numel())
            else:
                valid_high_count = 0

            view_summaries.append(
                {
                    "image_name": normalize_image_name(str(camera.image_name)),
                    "prior_path": str(prior_path),
                    "visible_candidates": int(candidate_ids.numel()),
                    "valid_low": int(candidate_ids_low.numel()),
                    "valid_high": int(valid_high_count),
                    "frontality_mean": float(frontality_low.mean().item()) if int(candidate_ids_low.numel()) > 0 else 0.0,
                    "radius_mean": float(radius_low.mean().item()) if int(candidate_ids_low.numel()) > 0 else 0.0,
                }
            )
            view_iter.set_postfix(low=int(candidate_ids_low.numel()), high=int(valid_high_count))

    low_result = finalize_consensus_state(
        low_state,
        min_views=int(args.min_views_low),
        min_confidence=float(args.min_low_confidence),
    )
    high_result = finalize_consensus_state(
        high_state,
        min_views=int(args.min_views_high),
        min_confidence=float(args.min_high_confidence),
    )

    anchor_rgb = anchor_rgb_t.detach().cpu().numpy().astype(np.float32, copy=False)
    raw_low_consensus = low_result["consensus_feat"].astype(np.float32, copy=False)
    if str(args.fusion_mode) == "anchor_residual":
        target_low_residual = raw_low_consensus.copy()
        low_residual_clamp = float(args.low_residual_clamp)
        if low_residual_clamp > 0.0:
            target_low_residual = np.clip(target_low_residual, -low_residual_clamp, low_residual_clamp).astype(np.float32, copy=False)
        low_apply = np.power(np.clip(low_result["confidence"], 0.0, 1.0), float(args.low_confidence_power)).astype(np.float32, copy=False)
        target_low_rgb = np.clip(anchor_rgb + low_apply[:, None] * target_low_residual, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        target_low_residual = (raw_low_consensus - anchor_rgb).astype(np.float32, copy=False)
        low_apply = np.ones((carrier_count,), dtype=np.float32)
        target_low_rgb = np.clip(raw_low_consensus, 0.0, 1.0).astype(np.float32, copy=False)
    target_high_patch = high_result["consensus_feat"].astype(np.float32, copy=False)
    target_high_rgb = target_high_patch[:, 0:3].astype(np.float32, copy=False)
    high_clamp = float(args.high_clamp)
    if high_clamp > 0.0:
        target_high_rgb = np.clip(target_high_rgb, -high_clamp, high_clamp).astype(np.float32, copy=False)
    detail_confidence = high_result["confidence"].astype(np.float32, copy=False)
    detail_confidence[~high_result["valid_mask"]] = 0.0
    target_high_rgb[~high_result["valid_mask"]] = 0.0

    high_energy = np.mean(np.abs(target_high_patch), axis=1).astype(np.float32, copy=False)
    risk = np.clip(
        high_result["disagreement"] + np.clip(high_energy / max(float(args.risk_high_energy_scale), 1e-6), 0.0, 1.0) * (1.0 - detail_confidence),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    detail_apply = np.power(np.clip(detail_confidence, 0.0, 1.0), float(args.detail_confidence_power)).astype(np.float32, copy=False)
    if bool(suppress_detail_by_risk):
        detail_apply = (detail_apply * (1.0 - risk)).astype(np.float32, copy=False)
    if not bool(args.disable_spatial_smoothing):
        support_radius = 0.5 * (
            carriers["scale_u"].astype(np.float32, copy=False) + carriers["scale_v"].astype(np.float32, copy=False)
        )
        target_low_residual, low_neighborhood_support = smooth_surface_field_knn(
            centers=carriers["xyz"],
            normals=carriers["normal"],
            radii=support_radius,
            value=target_low_residual,
            support=low_apply,
            k=int(args.spatial_smooth_knn),
            radius_scale=float(args.spatial_smooth_radius_scale),
            normal_min_cos=float(args.spatial_smooth_normal_min_cos),
            blend=float(args.spatial_smooth_blend),
        )
        target_high_rgb, detail_neighborhood_support = smooth_surface_field_knn(
            centers=carriers["xyz"],
            normals=carriers["normal"],
            radii=support_radius,
            value=target_high_rgb,
            support=detail_apply,
            k=int(args.spatial_smooth_knn),
            radius_scale=float(args.spatial_smooth_radius_scale),
            normal_min_cos=float(args.spatial_smooth_normal_min_cos),
            blend=float(args.spatial_smooth_blend),
        )
        low_apply = (low_apply * low_neighborhood_support).astype(np.float32, copy=False)
        detail_apply = (detail_apply * detail_neighborhood_support).astype(np.float32, copy=False)
    else:
        low_neighborhood_support = low_apply.astype(np.float32, copy=False)
        detail_neighborhood_support = detail_apply.astype(np.float32, copy=False)
    if str(args.fusion_mode) == "anchor_residual":
        target_low_rgb = np.clip(anchor_rgb + low_apply[:, None] * target_low_residual, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        target_low_rgb = np.clip(anchor_rgb + target_low_residual, 0.0, 1.0).astype(np.float32, copy=False)
    if str(args.fusion_mode) == "anchor_residual":
        fused_delta_rgb = (
            float(args.low_injection_scale) * low_apply[:, None] * target_low_residual
            + float(args.detail_gamma) * detail_apply[:, None] * target_high_rgb
        ).astype(np.float32, copy=False)
        fused_rgb = np.clip(anchor_rgb + fused_delta_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        fused_delta_rgb = (
            float(args.low_injection_scale) * (target_low_rgb - anchor_rgb)
            + float(args.detail_gamma) * detail_apply[:, None] * target_high_rgb
        ).astype(np.float32, copy=False)
        fused_rgb = np.clip(target_low_rgb + float(args.detail_gamma) * detail_apply[:, None] * target_high_rgb, 0.0, 1.0).astype(np.float32, copy=False)

    support_denom = np.clip(visible_view_count.astype(np.float32), 1.0, None)
    mean_frontality = (sample_frontality_sum / support_denom).astype(np.float32, copy=False)
    mean_radius_px = (sample_radius_sum / support_denom).astype(np.float32, copy=False)
    mean_alpha = (sample_alpha_sum / support_denom).astype(np.float32, copy=False)
    mean_depth_delta = (sample_depth_delta_sum / support_denom).astype(np.float32, copy=False)

    valid_mask = low_result["valid_mask"].astype(bool, copy=False)
    payload = {
        "gaussian_ids": carriers["gaussian_ids"].astype(np.int64, copy=False),
        "seed_ids": carriers["seed_ids"].astype(np.int64, copy=False),
        "source_tag": carriers["source_tag"].astype(np.int32, copy=False),
        "generation": carriers["generation"].astype(np.int32, copy=False),
        "centers": carriers["xyz"].astype(np.float32, copy=False),
        "tangent_u": carriers["tangent_u"].astype(np.float32, copy=False),
        "tangent_v": carriers["tangent_v"].astype(np.float32, copy=False),
        "normals": carriers["normal"].astype(np.float32, copy=False),
        "scale_u": carriers["scale_u"].astype(np.float32, copy=False),
        "scale_v": carriers["scale_v"].astype(np.float32, copy=False),
        "scale_n": carriers["scale_n"].astype(np.float32, copy=False),
        "opacity": carriers["opacity"].astype(np.float32, copy=False),
        "thickness_ratio": carriers["thickness_ratio"].astype(np.float32, copy=False),
        "anchor_rgb": anchor_rgb.astype(np.float32, copy=False),
        "raw_low_consensus": raw_low_consensus.astype(np.float32, copy=False),
        "target_low_residual": target_low_residual.astype(np.float32, copy=False),
        "low_apply": low_apply.astype(np.float32, copy=False),
        "low_neighborhood_support": low_neighborhood_support.astype(np.float32, copy=False),
        "target_low_rgb": target_low_rgb,
        "target_high_rgb": target_high_rgb.astype(np.float32, copy=False),
        "target_high_patch": target_high_patch.astype(np.float32, copy=False),
        "detail_apply": detail_apply.astype(np.float32, copy=False),
        "detail_neighborhood_support": detail_neighborhood_support.astype(np.float32, copy=False),
        "fused_delta_rgb": fused_delta_rgb.astype(np.float32, copy=False),
        "fused_rgb": fused_rgb,
        "confidence": low_result["confidence"].astype(np.float32, copy=False),
        "disagreement": low_result["disagreement"].astype(np.float32, copy=False),
        "support_count": low_result["support_count"].astype(np.int32, copy=False),
        "support_ratio": low_result["support_ratio"].astype(np.float32, copy=False),
        "detail_confidence": detail_confidence.astype(np.float32, copy=False),
        "detail_disagreement": high_result["disagreement"].astype(np.float32, copy=False),
        "detail_support_count": high_result["support_count"].astype(np.int32, copy=False),
        "detail_support_ratio": high_result["support_ratio"].astype(np.float32, copy=False),
        "risk": risk,
        "visible_view_count": visible_view_count.astype(np.int32, copy=False),
        "mean_frontality": mean_frontality,
        "mean_radius_px": mean_radius_px,
        "mean_alpha": mean_alpha,
        "mean_depth_delta": mean_depth_delta,
        "valid_mask": valid_mask,
        "low_exemplar_view_id": low_result["exemplar_view_id"].astype(np.int32, copy=False),
        "high_exemplar_view_id": high_result["exemplar_view_id"].astype(np.int32, copy=False),
        "patch_offsets_uv": patch_offsets_np.astype(np.float32, copy=False),
    }
    payload_path = output_dir / "sof_gaussian_surface_response_v0.npz"
    np.savez_compressed(payload_path, **payload)

    summary = {
        "mode": "prepare_sof_gaussian_response_v0",
        "scene_root": str(scene_root),
        "camera_model_path": str(camera_model_path),
        "mesh_path": str(mesh_path) if mesh_path is not None else "",
        "sof_model_path": str(sof_model_path),
        "sof_iteration": int(sof_iteration),
        "prior_dir": str(prior_dir),
        "output_dir": str(output_dir),
        "camera_count": int(len(cameras)),
        "carrier_count": int(carrier_count),
        "valid_count": int(valid_mask.sum()),
        "selected_ratio": float(valid_mask.sum() / max(carrier_count, 1)),
        "missing_prior_view_samples": missing_prior_views,
        "outputs": {
            "response_payload_npz": str(payload_path.resolve()),
        },
        "carrier_stats": {
            "opacity": stats_from_array(carriers["opacity"]),
            "thickness_ratio": stats_from_array(carriers["thickness_ratio"]),
            "scale_u": stats_from_array(carriers["scale_u"]),
            "scale_v": stats_from_array(carriers["scale_v"]),
            "scale_n": stats_from_array(carriers["scale_n"]),
        },
        "response_stats": {
            "confidence": stats_from_array(payload["confidence"][valid_mask]),
            "disagreement": stats_from_array(payload["disagreement"][valid_mask]),
            "detail_confidence": stats_from_array(payload["detail_confidence"][valid_mask]),
            "detail_disagreement": stats_from_array(payload["detail_disagreement"][valid_mask]),
            "risk": stats_from_array(payload["risk"][valid_mask]),
            "support_count": stats_from_array(payload["support_count"][valid_mask].astype(np.float32)),
            "detail_support_count": stats_from_array(payload["detail_support_count"][valid_mask].astype(np.float32)),
            "mean_frontality": stats_from_array(payload["mean_frontality"][valid_mask]),
            "mean_radius_px": stats_from_array(payload["mean_radius_px"][valid_mask]),
        },
        "parameters": {
            "carrier_min_opacity": float(args.carrier_min_opacity),
            "carrier_max_thickness_ratio": float(args.carrier_max_thickness_ratio),
            "carrier_max_count": int(args.carrier_max_count),
            "lowpass_kernel": int(args.lowpass_kernel),
            "patch_pattern": str(args.patch_pattern),
            "patch_radius_scale": float(args.patch_radius_scale),
            "depth_min": float(args.depth_min),
            "render_depth_abs_tolerance": float(args.render_depth_abs_tolerance),
            "render_depth_rel_tolerance": float(args.render_depth_rel_tolerance),
            "min_render_alpha": float(args.min_render_alpha),
            "min_screen_radius": float(args.min_screen_radius),
            "min_frontality": float(args.min_frontality),
            "low_color_threshold": float(args.low_color_threshold),
            "high_patch_threshold": float(args.high_patch_threshold),
            "min_views_low": int(args.min_views_low),
            "min_views_high": int(args.min_views_high),
            "min_low_confidence": float(args.min_low_confidence),
            "min_high_confidence": float(args.min_high_confidence),
            "detail_gamma": float(args.detail_gamma),
            "high_clamp": float(args.high_clamp),
            "risk_high_energy_scale": float(args.risk_high_energy_scale),
            "fusion_mode": str(args.fusion_mode),
            "low_residual_clamp": float(args.low_residual_clamp),
            "low_confidence_power": float(args.low_confidence_power),
            "detail_confidence_power": float(args.detail_confidence_power),
            "suppress_detail_by_risk": bool(suppress_detail_by_risk),
            "low_injection_scale": float(args.low_injection_scale),
            "spatial_smooth_knn": int(args.spatial_smooth_knn),
            "spatial_smooth_radius_scale": float(args.spatial_smooth_radius_scale),
            "spatial_smooth_normal_min_cos": float(args.spatial_smooth_normal_min_cos),
            "spatial_smooth_blend": float(args.spatial_smooth_blend),
            "spatial_smoothing_enabled": not bool(args.disable_spatial_smoothing),
        },
        "view_summaries": view_summaries,
        "note": (
            "This v0 payload treats a SOF Gaussian field as a surface-carrier domain. "
            "Low-frequency responses are fused by view consensus, while high-frequency responses "
            "are only retained when local Gaussian patches agree across multiple views. "
            "The optional mesh_path is currently recorded for provenance/debugging and is not yet "
            "used to modify carrier geometry in this v0 preprocessing stage. "
            "In anchor_residual mode the payload preserves each carrier's original SOF anchor color "
            "and injects only confidence-gated multiview residuals instead of repainting gaussians "
            "to a single consensus color."
        ),
    }
    summary_path = output_dir / "prepare_sof_gaussian_response_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
