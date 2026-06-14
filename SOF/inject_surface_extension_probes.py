import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from arguments import (
    MeshingParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    SplattingSettings,
    get_combined_args,
)
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state


BASELINE_EXTENSION_DEFAULTS = {
    "search_mode": "budget",
    "anchor_area_fraction": 0.10,
    "support_area_fraction": 0.30,
    "candidate_target_sample_count": 60000,
    "target_budget": 1000,
    "min_normal_dot": 0.6,
    "target_score_power": 1.0,
    "pair_target_gap_power": 1.0,
    "min_translation_scale_mult": 1.0,
    "max_translation_scale_mult": 8.0,
    "target_dedupe_radius_scale_mult": 2.0,
    "global_shell_edges_scale_mults": "8,20,40",
    "global_shell_stride_scale_mults": "12,20,32,48",
    "global_shell_priority_weights": "1.0,0.72,0.45,0.25",
    "global_target_density_scale": 1.0,
    "global_candidate_to_target_ratio": 6.0,
    "global_min_candidate_target_count": 120000,
    "global_max_candidate_target_count": 1200000,
    "global_max_target_budget": 100000,
    "subject_core_bias_strength": 0.0,
    "subject_proxy_mode": "image_center",
    "subject_core_camera_stride": 4,
    "subject_core_center_sigma": 0.35,
    "subject_core_visibility_power": 0.5,
    "gs_structure_neighbor_k": 24,
    "gs_structure_radius_scale_mult": 12.0,
    "gs_structure_batch_size": 8192,
    "use_foreground_mask": False,
    "foreground_depth_quantile": 0.6,
    "foreground_min_score": 0.2,
    "foreground_min_visible_views": 2,
    "foreground_camera_stride": 4,
    "foreground_depth_min": 0.2,
    "foreground_anchor_pool_mult": 4.0,
    "foreground_target_pool_mult": 2.0,
    "foreground_anchor_rank_foreground_strength": 0.5,
    "foreground_anchor_rank_subject_strength": 0.8,
    "enable_subject_interior_targets": False,
    "interior_subject_core_min": 0.35,
    "interior_seed_quantile": 0.35,
    "interior_min_visible_views": 2,
    "interior_weight_scale": 1.0,
    "interior_target_budget": 0,
    "interior_target_fraction": 0.0,
}


TARGET_SOURCE_FRONTIER = 0
TARGET_SOURCE_INTERIOR = 1
TARGET_SOURCE_NAME = {
    TARGET_SOURCE_FRONTIER: "frontier",
    TARGET_SOURCE_INTERIOR: "interior",
}

EXTENSION_PRESETS = {
    "baseline_v1": dict(BASELINE_EXTENSION_DEFAULTS),
    "aggressive_v1": {
        "anchor_area_fraction": 0.08,
        "candidate_target_sample_count": 180000,
        "target_budget": 2000,
        "min_normal_dot": 0.35,
        "target_score_power": 2.0,
        "pair_target_gap_power": 2.0,
        "min_translation_scale_mult": 2.0,
        "max_translation_scale_mult": 18.0,
        "target_dedupe_radius_scale_mult": 3.0,
    },
    "global_mesh_search_v1": {
        "search_mode": "global_mesh",
        "anchor_area_fraction": 0.08,
        "candidate_target_sample_count": 0,
        "target_budget": 0,
        "min_normal_dot": 0.25,
        "target_score_power": 2.0,
        "pair_target_gap_power": 1.8,
        "min_translation_scale_mult": 1.0,
        "max_translation_scale_mult": 24.0,
        "target_dedupe_radius_scale_mult": 0.0,
        "global_shell_edges_scale_mults": "8,20,40",
        "global_shell_stride_scale_mults": "12,20,32,48",
        "global_shell_priority_weights": "1.0,0.72,0.45,0.25",
        "global_target_density_scale": 1.0,
        "global_candidate_to_target_ratio": 6.0,
        "global_min_candidate_target_count": 120000,
        "global_max_candidate_target_count": 1200000,
        "global_max_target_budget": 100000,
        "subject_core_bias_strength": 1.5,
        "subject_core_camera_stride": 4,
        "subject_core_center_sigma": 0.32,
        "subject_core_visibility_power": 0.5,
    },
    "foreground_global_mesh_search_v1": {
        "search_mode": "global_mesh",
        "anchor_area_fraction": 0.18,
        "support_area_fraction": 0.45,
        "candidate_target_sample_count": 0,
        "target_budget": 0,
        "min_normal_dot": 0.25,
        "target_score_power": 2.0,
        "pair_target_gap_power": 1.8,
        "min_translation_scale_mult": 1.0,
        "max_translation_scale_mult": 24.0,
        "target_dedupe_radius_scale_mult": 0.0,
        "global_shell_edges_scale_mults": "8,20,40",
        "global_shell_stride_scale_mults": "12,20,32,48",
        "global_shell_priority_weights": "1.0,0.72,0.45,0.25",
        "global_target_density_scale": 1.0,
        "global_candidate_to_target_ratio": 6.0,
        "global_min_candidate_target_count": 120000,
        "global_max_candidate_target_count": 1200000,
        "global_max_target_budget": 100000,
        "subject_core_bias_strength": 1.5,
        "subject_core_camera_stride": 4,
        "subject_core_center_sigma": 0.32,
        "subject_core_visibility_power": 0.5,
        "use_foreground_mask": True,
        "foreground_depth_quantile": 0.6,
        "foreground_min_score": 0.2,
        "foreground_min_visible_views": 2,
        "foreground_camera_stride": 4,
        "foreground_depth_min": 0.2,
        "foreground_anchor_pool_mult": 4.0,
        "foreground_target_pool_mult": 2.0,
        "foreground_anchor_rank_foreground_strength": 0.5,
        "foreground_anchor_rank_subject_strength": 0.8,
    },
}


def build_appearance_embedding(mesh_args, num_views: int):
    if mesh_args.use_decoupled_appearance:
        return AppearanceEmbedding(num_views=num_views)
    if mesh_args.use_pgsr_appearance:
        return PGSREmbedding(num_views=num_views)
    return None


def resolve_start_checkpoint(model_path: str, start_checkpoint: Optional[str], iteration: int) -> str:
    if start_checkpoint:
        return start_checkpoint
    if iteration < 0:
        raise ValueError("iteration must be explicit when start_checkpoint is not provided.")
    return os.path.join(model_path, f"chkpnt{iteration}.pth")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    q = float(np.clip(quantile, 0.0, 1.0))
    if values.ndim != 1 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must be 1D arrays with matching lengths")
    if values.shape[0] == 0:
        raise ValueError("weighted_quantile received an empty array")
    total = float(weights.sum())
    if total <= 1e-12:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    threshold = q * total
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(max(index, 0), len(values_sorted) - 1)
    return float(values_sorted[index])


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.full(weights.shape[0], 1.0 / max(weights.shape[0], 1), dtype=np.float64)
    return weights / total


def parse_label_set(value: str) -> Set[str]:
    labels = {chunk.strip() for chunk in str(value).split(",") if chunk.strip()}
    if not labels:
        raise ValueError("label set cannot be empty")
    return labels


def load_target_face_ids_from_records(path: Optional[str], label_field: Optional[str] = None, keep_labels: Optional[Set[str]] = None) -> np.ndarray:
    if not path:
        return np.empty((0,), dtype=np.int64)
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list at {path}")
    target_face_ids: List[int] = []
    for item in records:
        if "target_face_id" not in item:
            continue
        if label_field is not None and keep_labels is not None:
            label = item.get(label_field)
            if label not in keep_labels:
                continue
        target_face_ids.append(int(item["target_face_id"]))
    if not target_face_ids:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.asarray(target_face_ids, dtype=np.int64))


def parse_float_sequence(value: str, name: str) -> np.ndarray:
    parts = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    if not parts:
        raise ValueError(f"{name} cannot be empty")
    try:
        return np.asarray([float(chunk) for chunk in parts], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"{name} contains non-float values: {value}") from exc


def assign_shell_ids(distance_scale: np.ndarray, shell_edges_scale_mults: np.ndarray) -> np.ndarray:
    shell_ids = np.zeros(distance_scale.shape[0], dtype=np.int32)
    for edge_idx, edge in enumerate(shell_edges_scale_mults):
        shell_ids[distance_scale >= edge] = edge_idx + 1
    return shell_ids


def resolve_global_search_config(args, gaussian_scale_max_median: float) -> Dict[str, np.ndarray]:
    shell_edges_scale_mults = parse_float_sequence(args.global_shell_edges_scale_mults, "global_shell_edges_scale_mults").astype(np.float64)
    shell_stride_scale_mults = parse_float_sequence(args.global_shell_stride_scale_mults, "global_shell_stride_scale_mults").astype(np.float64)
    shell_priority_weights = parse_float_sequence(args.global_shell_priority_weights, "global_shell_priority_weights").astype(np.float64)
    expected_shell_count = int(shell_edges_scale_mults.shape[0] + 1)
    if shell_stride_scale_mults.shape[0] != expected_shell_count:
        raise ValueError("global_shell_stride_scale_mults length must equal len(global_shell_edges_scale_mults) + 1")
    if shell_priority_weights.shape[0] != expected_shell_count:
        raise ValueError("global_shell_priority_weights length must equal len(global_shell_edges_scale_mults) + 1")
    shell_stride = np.clip(shell_stride_scale_mults * float(gaussian_scale_max_median), 1e-6, None)
    return {
        "shell_edges_scale_mults": shell_edges_scale_mults,
        "shell_stride_scale_mults": shell_stride_scale_mults,
        "shell_priority_weights": shell_priority_weights,
        "shell_stride": shell_stride,
    }


def resolve_auto_candidate_target_count(total_target_area: float, stride: float, args) -> int:
    stride = max(float(stride), 1e-6)
    estimate = int(
        np.ceil(
            float(args.global_target_density_scale)
            * float(args.global_candidate_to_target_ratio)
            * float(total_target_area)
            / (stride * stride)
        )
    )
    estimate = max(estimate, int(args.global_min_candidate_target_count))
    if int(args.global_max_candidate_target_count) > 0:
        estimate = min(estimate, int(args.global_max_candidate_target_count))
    return estimate


def allocate_capped_budgets(raw_budgets: np.ndarray, available_counts: np.ndarray, cap: int) -> np.ndarray:
    raw_budgets = np.asarray(raw_budgets, dtype=np.float64)
    available_counts = np.asarray(available_counts, dtype=np.int64)
    budgets = np.minimum(np.ceil(raw_budgets).astype(np.int64), available_counts)
    if cap <= 0:
        return budgets
    total = int(budgets.sum())
    if total <= cap:
        return budgets

    positive = raw_budgets > 0.0
    scaled = raw_budgets * (float(cap) / max(float(raw_budgets.sum()), 1e-12))
    base = np.floor(scaled).astype(np.int64)
    base = np.minimum(base, available_counts)
    base[(positive) & (available_counts > 0) & (base <= 0)] = 1

    current = int(base.sum())
    if current > cap:
        order = np.argsort(scaled - base)
        for idx in order:
            if current <= cap:
                break
            if base[idx] > 0:
                base[idx] -= 1
                current -= 1
        return base

    remainders = scaled - base
    for idx in np.argsort(-remainders):
        if current >= cap:
            break
        if base[idx] >= available_counts[idx]:
            continue
        base[idx] += 1
        current += 1
    return base


def _select_targets_global_search_subset(
    matched: Dict[str, np.ndarray],
    pair_priority: np.ndarray,
    face_areas: np.ndarray,
    gaussian_scale_max_median: float,
    args,
    global_search_config: Dict[str, np.ndarray],
    budget_cap: int,
    fill_to_budget: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    translation_scale = matched["translation_distance"] / max(float(gaussian_scale_max_median), 1e-6)
    shell_ids = assign_shell_ids(translation_scale, global_search_config["shell_edges_scale_mults"])
    weighted_priority = pair_priority * global_search_config["shell_priority_weights"][shell_ids].astype(np.float32)
    matched_target_areas = face_areas[matched["target_face_ids"]]
    shell_count = int(global_search_config["shell_priority_weights"].shape[0])

    raw_shell_budgets = np.zeros((shell_count,), dtype=np.float64)
    available_counts = np.zeros((shell_count,), dtype=np.int64)
    shell_stats = []
    for shell_idx in range(shell_count):
        shell_mask = shell_ids == shell_idx
        available_counts[shell_idx] = int(shell_mask.sum())
        shell_area = float(matched_target_areas[shell_mask].sum())
        stride = float(global_search_config["shell_stride"][shell_idx])
        raw_budget = (
            float(args.global_target_density_scale) * shell_area / max(stride * stride, 1e-12)
            if available_counts[shell_idx] > 0
            else 0.0
        )
        raw_shell_budgets[shell_idx] = raw_budget
        shell_stats.append(
            {
                "shell_index": shell_idx,
                "distance_scale_min": 0.0 if shell_idx == 0 else float(global_search_config["shell_edges_scale_mults"][shell_idx - 1]),
                "distance_scale_max": (
                    float(global_search_config["shell_edges_scale_mults"][shell_idx])
                    if shell_idx < shell_count - 1
                    else None
                ),
                "stride_scale_mult": float(global_search_config["shell_stride_scale_mults"][shell_idx]),
                "stride": stride,
                "priority_weight": float(global_search_config["shell_priority_weights"][shell_idx]),
                "matched_face_count": int(available_counts[shell_idx]),
                "matched_face_area": shell_area,
                "raw_budget": float(raw_budget),
            }
        )

    if fill_to_budget and budget_cap > 0 and int(available_counts.sum()) > 0:
        fill_weights = raw_shell_budgets.copy()
        if float(fill_weights.sum()) <= 1e-12:
            fill_weights = available_counts.astype(np.float64)
        positive_mask = available_counts > 0
        fill_weights[(positive_mask) & (fill_weights <= 0.0)] = 1.0
        fill_raw_budgets = np.zeros_like(raw_shell_budgets)
        fill_raw_budgets[positive_mask] = (
            float(budget_cap) * fill_weights[positive_mask] / max(float(fill_weights[positive_mask].sum()), 1e-12)
        )
        allocated = allocate_capped_budgets(fill_raw_budgets, available_counts, budget_cap)
    else:
        allocated = allocate_capped_budgets(raw_shell_budgets, available_counts, budget_cap)

    selected_ids = []
    for shell_idx in range(shell_count):
        shell_mask = np.flatnonzero(shell_ids == shell_idx)
        budget = int(allocated[shell_idx])
        shell_stats[shell_idx]["allocated_budget"] = budget
        if shell_mask.size == 0 or budget <= 0:
            shell_stats[shell_idx]["selected_count"] = 0
            continue
        local = voxel_sparse_select(
            points=matched["target_centers"][shell_mask],
            priorities=weighted_priority[shell_mask],
            max_count=budget,
            voxel_size=float(global_search_config["shell_stride"][shell_idx]),
        )
        chosen = shell_mask[local]
        selected_ids.append(chosen)
        shell_stats[shell_idx]["selected_count"] = int(chosen.size)

    selected_local_ids = (
        np.concatenate(selected_ids, axis=0).astype(np.int64, copy=False)
        if selected_ids
        else np.empty((0,), dtype=np.int64)
    )
    return selected_local_ids, {
        "shell_ids": shell_ids.astype(np.int32, copy=False),
        "pair_priority_weighted": weighted_priority.astype(np.float32, copy=False),
        "shell_stats": shell_stats,
        "total_raw_budget": float(raw_shell_budgets.sum()),
        "total_allocated_budget": int(allocated.sum()),
    }


def select_targets_global_search(
    matched: Dict[str, np.ndarray],
    pair_priority: np.ndarray,
    face_areas: np.ndarray,
    gaussian_scale_max_median: float,
    args,
    global_search_config: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, object]]:
    budget_cap = int(args.target_budget)
    if budget_cap <= 0 and int(args.global_max_target_budget) > 0:
        budget_cap = int(args.global_max_target_budget)

    source_codes = matched.get("target_source_code")
    if (
        budget_cap <= 0
        or source_codes is None
        or np.count_nonzero(source_codes == TARGET_SOURCE_INTERIOR) == 0
        or (int(args.interior_target_budget) <= 0 and float(args.interior_target_fraction) <= 0.0)
    ):
        return _select_targets_global_search_subset(
            matched=matched,
            pair_priority=pair_priority,
            face_areas=face_areas,
            gaussian_scale_max_median=gaussian_scale_max_median,
            args=args,
            global_search_config=global_search_config,
            budget_cap=budget_cap,
        )

    interior_available = int(np.count_nonzero(source_codes == TARGET_SOURCE_INTERIOR))
    reserved_interior = int(args.interior_target_budget)
    if reserved_interior <= 0 and float(args.interior_target_fraction) > 0.0:
        reserved_interior = int(np.round(float(args.interior_target_fraction) * float(budget_cap)))
    reserved_interior = max(0, min(reserved_interior, interior_available, budget_cap))
    if reserved_interior <= 0:
        return _select_targets_global_search_subset(
            matched=matched,
            pair_priority=pair_priority,
            face_areas=face_areas,
            gaussian_scale_max_median=gaussian_scale_max_median,
            args=args,
            global_search_config=global_search_config,
            budget_cap=budget_cap,
        )

    frontier_mask = source_codes != TARGET_SOURCE_INTERIOR
    interior_mask = source_codes == TARGET_SOURCE_INTERIOR

    def slice_matched(mask: np.ndarray) -> Dict[str, np.ndarray]:
        return {key: value[mask] for key, value in matched.items()}

    frontier_budget = max(budget_cap - reserved_interior, 0)
    selected_blocks = []
    source_selection_summary = {
        "budget_cap": int(budget_cap),
        "reserved_interior_budget": int(reserved_interior),
        "frontier_available": int(np.count_nonzero(frontier_mask)),
        "interior_available": int(interior_available),
    }

    if np.any(frontier_mask) and frontier_budget > 0:
        frontier_selected_local, frontier_summary = _select_targets_global_search_subset(
            matched=slice_matched(frontier_mask),
            pair_priority=pair_priority[frontier_mask],
            face_areas=face_areas,
            gaussian_scale_max_median=gaussian_scale_max_median,
            args=args,
            global_search_config=global_search_config,
            budget_cap=frontier_budget,
            fill_to_budget=False,
        )
        selected_blocks.append(np.flatnonzero(frontier_mask)[frontier_selected_local])
        source_selection_summary["frontier_selected"] = int(frontier_selected_local.size)
    else:
        source_selection_summary["frontier_selected"] = 0

    if np.any(interior_mask) and reserved_interior > 0:
        interior_selected_local, interior_summary_select = _select_targets_global_search_subset(
            matched=slice_matched(interior_mask),
            pair_priority=pair_priority[interior_mask],
            face_areas=face_areas,
            gaussian_scale_max_median=gaussian_scale_max_median,
            args=args,
            global_search_config=global_search_config,
            budget_cap=reserved_interior,
            fill_to_budget=True,
        )
        selected_blocks.append(np.flatnonzero(interior_mask)[interior_selected_local])
        source_selection_summary["interior_selected"] = int(interior_selected_local.size)
    else:
        source_selection_summary["interior_selected"] = 0

    selected_local_ids = (
        np.concatenate(selected_blocks, axis=0).astype(np.int64, copy=False)
        if selected_blocks
        else np.empty((0,), dtype=np.int64)
    )

    selected_local_ids = np.unique(selected_local_ids)
    used = int(selected_local_ids.size)
    leftover = max(budget_cap - used, 0)
    source_selection_summary["leftover_budget"] = int(leftover)

    if leftover > 0:
        remaining_mask = np.ones(pair_priority.shape[0], dtype=bool)
        remaining_mask[selected_local_ids] = False
        if np.any(remaining_mask):
            remaining_selected_local, _ = _select_targets_global_search_subset(
                matched=slice_matched(remaining_mask),
                pair_priority=pair_priority[remaining_mask],
                face_areas=face_areas,
                gaussian_scale_max_median=gaussian_scale_max_median,
                args=args,
                global_search_config=global_search_config,
                budget_cap=leftover,
                fill_to_budget=False,
            )
            if remaining_selected_local.size > 0:
                selected_local_ids = np.unique(
                    np.concatenate([selected_local_ids, np.flatnonzero(remaining_mask)[remaining_selected_local]], axis=0)
                ).astype(np.int64, copy=False)

    base_selected_ids, base_summary = _select_targets_global_search_subset(
        matched=matched,
        pair_priority=pair_priority,
        face_areas=face_areas,
        gaussian_scale_max_median=gaussian_scale_max_median,
        args=args,
        global_search_config=global_search_config,
        budget_cap=budget_cap,
        fill_to_budget=False,
    )
    selected_shell_ids = base_summary["shell_ids"][selected_local_ids]
    for shell_stat in base_summary["shell_stats"]:
        shell_idx = int(shell_stat["shell_index"])
        shell_stat["selected_count"] = int(np.count_nonzero(selected_shell_ids == shell_idx))
    base_summary["total_allocated_budget"] = int(selected_local_ids.size)
    base_summary["source_budgeting"] = source_selection_summary
    return selected_local_ids, base_summary


def sample_weighted_face_ids(
    candidate_ids: np.ndarray,
    candidate_weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or candidate_ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    if candidate_ids.size <= count:
        return candidate_ids.astype(np.int64, copy=False)
    probabilities = normalize_weights(candidate_weights)
    sampled = rng.choice(candidate_ids, size=count, replace=False, p=probabilities)
    return np.asarray(sampled, dtype=np.int64)


def compute_face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tris = vertices[faces[face_ids]]
    centers = tris.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(normal_norm, 1e-12, None)
    return centers.astype(np.float32), normals.astype(np.float32)


def compute_subject_core_scores(
    points_xyz: np.ndarray,
    cameras,
    depth_min: float,
    camera_stride: int,
    center_sigma: float,
    visibility_power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    stride = max(int(camera_stride), 1)
    selected_cameras = list(cameras)[::stride]
    if len(selected_cameras) == 0:
        return np.zeros((points_xyz.shape[0],), dtype=np.float32), np.zeros((points_xyz.shape[0],), dtype=np.int32)

    scores = np.zeros((points_xyz.shape[0],), dtype=np.float64)
    visible_counts = np.zeros((points_xyz.shape[0],), dtype=np.int32)
    sigma = max(float(center_sigma), 1e-6)

    for camera in selected_cameras:
        R = np.asarray(camera.R, dtype=np.float32)
        T = np.asarray(camera.T, dtype=np.float32)
        xyz_cam = points_xyz @ R + T[None, :]
        z = xyz_cam[:, 2]
        valid = z > float(depth_min)
        if not np.any(valid):
            continue

        z_safe = np.clip(z, 1e-6, None)
        x = xyz_cam[:, 0] / z_safe * float(camera.focal_x) + float(camera.image_width) / 2.0
        y = xyz_cam[:, 1] / z_safe * float(camera.focal_y) + float(camera.image_height) / 2.0

        valid &= x >= 0.0
        valid &= x < float(camera.image_width)
        valid &= y >= 0.0
        valid &= y < float(camera.image_height)
        if not np.any(valid):
            continue

        x_norm = (x[valid] / max(float(camera.image_width), 1.0)) - 0.5
        y_norm = (y[valid] / max(float(camera.image_height), 1.0)) - 0.5
        centrality = np.exp(-0.5 * ((x_norm / sigma) ** 2 + (y_norm / sigma) ** 2))
        scores[valid] += centrality
        visible_counts[valid] += 1

    if len(selected_cameras) <= 0:
        return np.zeros((points_xyz.shape[0],), dtype=np.float32), visible_counts

    mean_centrality = scores / np.maximum(visible_counts, 1)
    visibility_fraction = visible_counts.astype(np.float64) / float(len(selected_cameras))
    subject_core = mean_centrality * np.power(np.clip(visibility_fraction, 0.0, 1.0), max(float(visibility_power), 0.0))
    if np.max(subject_core) > 0:
        subject_core = subject_core / float(np.max(subject_core))
    return subject_core.astype(np.float32), visible_counts


def compute_foreground_support_scores(
    points_xyz: np.ndarray,
    cameras,
    depth_min: float,
    camera_stride: int,
    near_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 0:
        empty_float = np.empty((0,), dtype=np.float32)
        empty_int = np.empty((0,), dtype=np.int32)
        return empty_float, empty_int, empty_int

    stride = max(int(camera_stride), 1)
    selected_cameras = list(cameras)[::stride]
    if len(selected_cameras) == 0:
        empty_score = np.zeros((points_xyz.shape[0],), dtype=np.float32)
        empty_int = np.zeros((points_xyz.shape[0],), dtype=np.int32)
        return empty_score, empty_int, empty_int

    q = float(np.clip(near_quantile, 0.0, 1.0))
    visible_counts = np.zeros((points_xyz.shape[0],), dtype=np.int32)
    near_counts = np.zeros((points_xyz.shape[0],), dtype=np.int32)

    for camera in selected_cameras:
        R = np.asarray(camera.R, dtype=np.float32)
        T = np.asarray(camera.T, dtype=np.float32)
        xyz_cam = points_xyz @ R + T[None, :]
        z = xyz_cam[:, 2]
        valid = z > float(depth_min)
        if not np.any(valid):
            continue

        z_safe = np.clip(z, 1e-6, None)
        x = xyz_cam[:, 0] / z_safe * float(camera.focal_x) + float(camera.image_width) / 2.0
        y = xyz_cam[:, 1] / z_safe * float(camera.focal_y) + float(camera.image_height) / 2.0

        valid &= x >= 0.0
        valid &= x < float(camera.image_width)
        valid &= y >= 0.0
        valid &= y < float(camera.image_height)
        if not np.any(valid):
            continue

        visible_ids = np.flatnonzero(valid)
        visible_depth = z[valid]
        if q <= 0.0:
            depth_cutoff = float(np.min(visible_depth))
        elif q >= 1.0:
            depth_cutoff = float(np.max(visible_depth))
        else:
            depth_cutoff = float(np.quantile(visible_depth, q))
        near_local = visible_depth <= depth_cutoff
        visible_counts[visible_ids] += 1
        near_counts[visible_ids] += near_local.astype(np.int32)

    score = near_counts.astype(np.float32) / np.maximum(visible_counts, 1).astype(np.float32)
    return score.astype(np.float32), visible_counts, near_counts


def compute_gs_structure_scores(
    points_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
    gaussian_opacity: np.ndarray,
    gaussian_scaling: np.ndarray,
    neighbor_k: int,
    radius: float,
    scale_ref: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if points_xyz.shape[0] == 0:
        empty_f = np.empty((0,), dtype=np.float32)
        empty_i = np.empty((0,), dtype=np.int32)
        return empty_f, empty_i, {
            "density_raw": empty_f,
            "density_norm": empty_f,
            "small_scale": empty_f,
            "anisotropy": empty_f,
            "nonplanar_raw": empty_f,
            "nonplanar_norm": empty_f,
        }
    if gaussian_xyz.shape[0] == 0:
        zeros_f = np.zeros((points_xyz.shape[0],), dtype=np.float32)
        zeros_i = np.zeros((points_xyz.shape[0],), dtype=np.int32)
        return zeros_f, zeros_i, {
            "density_raw": zeros_f,
            "density_norm": zeros_f,
            "small_scale": zeros_f,
            "anisotropy": zeros_f,
            "nonplanar_raw": zeros_f,
            "nonplanar_norm": zeros_f,
        }

    tree = cKDTree(gaussian_xyz.astype(np.float32, copy=False))
    k = min(max(int(neighbor_k), 1), gaussian_xyz.shape[0])
    batch = max(int(batch_size), 1)
    radius = max(float(radius), 1e-6)
    scale_ref = max(float(scale_ref), 1e-6)

    gaussian_scale_mean = np.mean(gaussian_scaling, axis=1).astype(np.float32, copy=False)
    gaussian_scale_max = np.max(gaussian_scaling, axis=1).astype(np.float32, copy=False)
    gaussian_scale_min = np.min(gaussian_scaling, axis=1).astype(np.float32, copy=False)
    gaussian_small_scale = np.exp(-gaussian_scale_mean / scale_ref).astype(np.float32, copy=False)
    gaussian_anisotropy = (1.0 - (gaussian_scale_min / np.clip(gaussian_scale_max, 1e-6, None))).astype(np.float32, copy=False)

    density_raw = np.zeros((points_xyz.shape[0],), dtype=np.float32)
    small_scale_score = np.zeros((points_xyz.shape[0],), dtype=np.float32)
    anisotropy_score = np.zeros((points_xyz.shape[0],), dtype=np.float32)
    nonplanar_raw = np.zeros((points_xyz.shape[0],), dtype=np.float32)
    neighbor_count = np.zeros((points_xyz.shape[0],), dtype=np.int32)

    for start in range(0, points_xyz.shape[0], batch):
        end = min(start + batch, points_xyz.shape[0])
        batch_points = points_xyz[start:end].astype(np.float32, copy=False)
        distances, neighbor_ids = tree.query(batch_points, k=k)
        if k == 1:
            distances = distances[:, None]
            neighbor_ids = neighbor_ids[:, None]

        batch_dist = np.asarray(distances, dtype=np.float32)
        batch_ids = np.asarray(neighbor_ids, dtype=np.int64)
        batch_opacity = gaussian_opacity[batch_ids]
        distance_weight = np.exp(-0.5 * ((batch_dist / radius) ** 2)).astype(np.float32, copy=False)
        weight = distance_weight * batch_opacity
        weight_sum = np.clip(weight.sum(axis=1), 1e-6, None)
        weight_norm = weight / weight_sum[:, None]

        density_raw[start:end] = weight_sum.astype(np.float32, copy=False)
        neighbor_count[start:end] = (weight > 1e-6).sum(axis=1).astype(np.int32, copy=False)
        small_scale_score[start:end] = (
            (weight * gaussian_small_scale[batch_ids]).sum(axis=1) / weight_sum
        ).astype(np.float32, copy=False)
        anisotropy_score[start:end] = (
            (weight * gaussian_anisotropy[batch_ids]).sum(axis=1) / weight_sum
        ).astype(np.float32, copy=False)

        neighbor_xyz = gaussian_xyz[batch_ids]
        local_mean = (weight_norm[..., None] * neighbor_xyz).sum(axis=1)
        centered = neighbor_xyz - local_mean[:, None, :]
        cov = np.einsum("bk,bki,bkj->bij", weight_norm, centered, centered, optimize=True).astype(np.float32, copy=False)
        eigvals = np.linalg.eigvalsh(cov)
        lam_max = np.clip(eigvals[:, 2], 1e-6, None)
        nonplanar_raw[start:end] = (eigvals[:, 0] / lam_max).astype(np.float32, copy=False)

    density_norm = np.clip(density_raw / max(float(np.percentile(density_raw, 95)), 1e-6), 0.0, 1.0).astype(np.float32, copy=False)
    nonplanar_norm = np.clip(nonplanar_raw / max(float(np.percentile(nonplanar_raw, 95)), 1e-6), 0.0, 1.0).astype(np.float32, copy=False)
    structure_score = (
        density_norm * np.sqrt(np.clip(small_scale_score * anisotropy_score * nonplanar_norm, 0.0, 1.0))
    ).astype(np.float32, copy=False)
    if np.max(structure_score) > 0:
        structure_score = (structure_score / float(np.max(structure_score))).astype(np.float32, copy=False)

    return structure_score, neighbor_count, {
        "density_raw": density_raw,
        "density_norm": density_norm,
        "small_scale": small_scale_score,
        "anisotropy": anisotropy_score,
        "nonplanar_raw": nonplanar_raw,
        "nonplanar_norm": nonplanar_norm,
    }


def blend_score_with_context(
    base_score: np.ndarray,
    foreground_score: Optional[np.ndarray] = None,
    subject_score: Optional[np.ndarray] = None,
    foreground_strength: float = 0.0,
    subject_strength: float = 0.0,
) -> np.ndarray:
    score = np.clip(np.asarray(base_score, dtype=np.float32), 1e-6, None)
    if foreground_score is not None and foreground_strength > 0.0:
        fg = np.clip(np.asarray(foreground_score, dtype=np.float32), 0.0, 1.0)
        score = score * ((1.0 - float(foreground_strength)) + float(foreground_strength) * fg)
    if subject_score is not None and subject_strength > 0.0:
        subj = np.clip(np.asarray(subject_score, dtype=np.float32), 0.0, 1.0)
        score = score * ((1.0 - float(subject_strength)) + float(subject_strength) * subj)
    return score.astype(np.float32)


def voxel_sparse_select(
    points: np.ndarray,
    priorities: np.ndarray,
    max_count: int,
    voxel_size: float,
) -> np.ndarray:
    if max_count <= 0 or points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if voxel_size <= 0.0:
        order = np.argsort(-priorities)
        return order[:max_count].astype(np.int64)

    order = np.argsort(-priorities)
    used_cells = set()
    selected: List[int] = []
    inv_voxel = 1.0 / voxel_size
    for idx in order:
        cell = tuple(np.floor(points[idx] * inv_voxel).astype(np.int64).tolist())
        if cell in used_cells:
            continue
        used_cells.add(cell)
        selected.append(int(idx))
        if len(selected) >= max_count:
            break
    return np.asarray(selected, dtype=np.int64)


def pick_anchor_for_targets(
    anchor_centers: np.ndarray,
    anchor_normals: np.ndarray,
    anchor_face_ids: np.ndarray,
    anchor_scores: np.ndarray,
    target_centers: np.ndarray,
    target_normals: np.ndarray,
    target_face_ids: np.ndarray,
    target_scores: np.ndarray,
    target_subject_core: np.ndarray,
    k_neighbors: int,
    min_translation: float,
    max_translation: float,
    min_normal_dot: float,
) -> Dict[str, np.ndarray]:
    if anchor_centers.shape[0] == 0 or target_centers.shape[0] == 0:
        return {
            "anchor_face_ids": np.empty((0,), dtype=np.int64),
            "target_face_ids": np.empty((0,), dtype=np.int64),
            "anchor_centers": np.empty((0, 3), dtype=np.float32),
            "target_centers": np.empty((0, 3), dtype=np.float32),
            "anchor_normals": np.empty((0, 3), dtype=np.float32),
            "target_normals": np.empty((0, 3), dtype=np.float32),
            "anchor_scores": np.empty((0,), dtype=np.float32),
            "target_scores": np.empty((0,), dtype=np.float32),
            "target_subject_core": np.empty((0,), dtype=np.float32),
            "translation_distance": np.empty((0,), dtype=np.float32),
            "normal_dot": np.empty((0,), dtype=np.float32),
        }

    tree = cKDTree(anchor_centers)
    k = min(max(k_neighbors, 1), anchor_centers.shape[0])
    distances, neighbor_ids = tree.query(target_centers, k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_ids = neighbor_ids[:, None]

    matched_anchor_ids = []
    matched_target_ids = []
    matched_anchor_centers = []
    matched_target_centers = []
    matched_anchor_normals = []
    matched_target_normals = []
    matched_anchor_scores = []
    matched_target_scores = []
    matched_target_subject_core = []
    matched_distances = []
    matched_normal_dot = []

    for idx in range(target_centers.shape[0]):
        chosen = None
        for local_dist, local_anchor in zip(distances[idx], neighbor_ids[idx]):
            if local_dist < min_translation or local_dist > max_translation:
                continue
            dot = float(abs(np.dot(anchor_normals[local_anchor], target_normals[idx])))
            if dot < min_normal_dot:
                continue
            chosen = (int(local_anchor), float(local_dist), dot)
            break
        if chosen is None:
            continue
        anchor_local_id, dist_value, dot_value = chosen
        matched_anchor_ids.append(int(anchor_face_ids[anchor_local_id]))
        matched_target_ids.append(int(target_face_ids[idx]))
        matched_anchor_centers.append(anchor_centers[anchor_local_id])
        matched_target_centers.append(target_centers[idx])
        matched_anchor_normals.append(anchor_normals[anchor_local_id])
        matched_target_normals.append(target_normals[idx])
        matched_anchor_scores.append(float(anchor_scores[anchor_local_id]))
        matched_target_scores.append(float(target_scores[idx]))
        matched_target_subject_core.append(float(target_subject_core[idx]))
        matched_distances.append(dist_value)
        matched_normal_dot.append(dot_value)

    return {
        "anchor_face_ids": np.asarray(matched_anchor_ids, dtype=np.int64),
        "target_face_ids": np.asarray(matched_target_ids, dtype=np.int64),
        "anchor_centers": np.asarray(matched_anchor_centers, dtype=np.float32),
        "target_centers": np.asarray(matched_target_centers, dtype=np.float32),
        "anchor_normals": np.asarray(matched_anchor_normals, dtype=np.float32),
        "target_normals": np.asarray(matched_target_normals, dtype=np.float32),
        "anchor_scores": np.asarray(matched_anchor_scores, dtype=np.float32),
        "target_scores": np.asarray(matched_target_scores, dtype=np.float32),
        "target_subject_core": np.asarray(matched_target_subject_core, dtype=np.float32),
        "translation_distance": np.asarray(matched_distances, dtype=np.float32),
        "normal_dot": np.asarray(matched_normal_dot, dtype=np.float32),
    }


def choose_parent_gaussians(
    gaussian_tree: cKDTree,
    gaussian_xyz: np.ndarray,
    gaussian_opacity: np.ndarray,
    anchor_centers: np.ndarray,
    k_neighbors: int,
    min_parent_opacity: float,
) -> np.ndarray:
    if anchor_centers.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    k = min(max(k_neighbors, 1), gaussian_xyz.shape[0])
    distances, neighbor_ids = gaussian_tree.query(anchor_centers, k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_ids = neighbor_ids[:, None]

    chosen_parent = np.empty((anchor_centers.shape[0],), dtype=np.int64)
    for row in range(anchor_centers.shape[0]):
        ids = np.asarray(neighbor_ids[row], dtype=np.int64)
        dists = np.asarray(distances[row], dtype=np.float32)
        opacity = gaussian_opacity[ids]
        valid = opacity >= min_parent_opacity
        if np.any(valid):
            ids = ids[valid]
            dists = dists[valid]
            opacity = opacity[valid]
        rank = opacity / np.maximum(dists, 1e-4)
        chosen_parent[row] = int(ids[int(np.argmax(rank))])
    return chosen_parent


def save_point_cloud(points: np.ndarray, path: str):
    if points.shape[0] == 0:
        return
    cloud = trimesh.points.PointCloud(points)
    cloud.export(path)


def export_face_subset(mesh: trimesh.Trimesh, face_ids: np.ndarray, path: str):
    if face_ids.size == 0:
        return
    submesh = mesh.submesh([np.unique(face_ids)], append=True, repair=False)
    submesh.export(path)


def stats_from_array(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def counts_from_source_codes(source_codes: np.ndarray) -> Dict[str, int]:
    counts = {name: 0 for name in TARGET_SOURCE_NAME.values()}
    if source_codes.size == 0:
        return counts
    unique_codes, unique_counts = np.unique(source_codes.astype(np.int32, copy=False), return_counts=True)
    for code, count in zip(unique_codes.tolist(), unique_counts.tolist()):
        counts[TARGET_SOURCE_NAME.get(int(code), f"unknown_{int(code)}")] = int(count)
    return counts


def build_default_paths(dataset, checkpoint_iteration: int, output_checkpoint: Optional[str], output_summary: Optional[str], output_manifest: Optional[str], output_preview_dir: Optional[str]):
    preview_dir = output_preview_dir or os.path.join(dataset.model_path, f"extension_probe_preview_iter{checkpoint_iteration}")
    checkpoint_path = output_checkpoint or os.path.join(dataset.model_path, f"chkpnt{checkpoint_iteration}_extensionprobe.pth")
    summary_path = output_summary or os.path.join(dataset.model_path, f"extension_probe_summary_iter{checkpoint_iteration}.json")
    manifest_path = output_manifest or os.path.join(preview_dir, "extension_probe_manifest.json")
    return checkpoint_path, summary_path, manifest_path, preview_dir


def apply_extension_preset(args):
    preset_name = getattr(args, "extension_preset", "baseline_v1")
    if preset_name not in EXTENSION_PRESETS:
        raise ValueError(f"Unsupported extension_preset: {preset_name}")

    preset = EXTENSION_PRESETS[preset_name]
    applied_overrides = {}
    for key, preset_value in preset.items():
        if getattr(args, key) == BASELINE_EXTENSION_DEFAULTS[key]:
            setattr(args, key, preset_value)
            if preset_value != BASELINE_EXTENSION_DEFAULTS[key]:
                applied_overrides[key] = preset_value

    print(f"Using extension preset: {preset_name}")
    if applied_overrides:
        print("Applied preset overrides:")
        for key, value in applied_overrides.items():
            print(f"  {key}: {value}")
    print("Resolved extension parameters:")
    for key in (
        "search_mode",
        "anchor_area_fraction",
        "support_area_fraction",
        "candidate_target_sample_count",
        "target_budget",
        "min_normal_dot",
        "target_score_power",
        "pair_target_gap_power",
        "min_translation_scale_mult",
        "max_translation_scale_mult",
        "target_dedupe_radius_scale_mult",
        "subject_core_bias_strength",
        "subject_proxy_mode",
        "subject_core_camera_stride",
        "subject_core_center_sigma",
        "subject_core_visibility_power",
        "gs_structure_neighbor_k",
        "gs_structure_radius_scale_mult",
        "gs_structure_batch_size",
        "use_foreground_mask",
        "foreground_depth_quantile",
        "foreground_min_score",
        "foreground_min_visible_views",
        "foreground_camera_stride",
        "foreground_depth_min",
        "foreground_anchor_pool_mult",
        "foreground_target_pool_mult",
        "foreground_anchor_rank_foreground_strength",
        "foreground_anchor_rank_subject_strength",
        "enable_subject_interior_targets",
        "interior_subject_core_min",
        "interior_seed_quantile",
        "interior_min_visible_views",
        "interior_weight_scale",
        "interior_target_budget",
        "interior_target_fraction",
    ):
        print(f"  {key}: {getattr(args, key)}")
    if getattr(args, "search_mode", "budget") == "global_mesh":
        for key in (
            "global_shell_edges_scale_mults",
            "global_shell_stride_scale_mults",
            "global_shell_priority_weights",
            "global_target_density_scale",
            "global_candidate_to_target_ratio",
            "global_min_candidate_target_count",
            "global_max_candidate_target_count",
            "global_max_target_budget",
        ):
            print(f"  {key}: {getattr(args, key)}")


def main():
    parser = ArgumentParser(description="Inject surface-extension probe Gaussians copied from reliable anchors.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--proxy_npz", type=str, required=True)
    parser.add_argument(
        "--extension_preset",
        choices=sorted(EXTENSION_PRESETS.keys()),
        default="baseline_v1",
    )
    parser.add_argument("--search_mode", choices=["budget", "global_mesh"], default=BASELINE_EXTENSION_DEFAULTS["search_mode"])
    parser.add_argument("--anchor_area_fraction", type=float, default=BASELINE_EXTENSION_DEFAULTS["anchor_area_fraction"])
    parser.add_argument("--support_area_fraction", type=float, default=BASELINE_EXTENSION_DEFAULTS["support_area_fraction"])
    parser.add_argument("--anchor_face_sample_count", type=int, default=20000)
    parser.add_argument(
        "--candidate_target_sample_count",
        type=int,
        default=BASELINE_EXTENSION_DEFAULTS["candidate_target_sample_count"],
    )
    parser.add_argument("--target_budget", type=int, default=BASELINE_EXTENSION_DEFAULTS["target_budget"])
    parser.add_argument("--anchor_neighbor_k", type=int, default=8)
    parser.add_argument("--parent_neighbor_k", type=int, default=16)
    parser.add_argument("--min_parent_opacity", type=float, default=0.05)
    parser.add_argument("--min_normal_dot", type=float, default=BASELINE_EXTENSION_DEFAULTS["min_normal_dot"])
    parser.add_argument("--target_score_power", type=float, default=BASELINE_EXTENSION_DEFAULTS["target_score_power"])
    parser.add_argument("--pair_target_gap_power", type=float, default=BASELINE_EXTENSION_DEFAULTS["pair_target_gap_power"])
    parser.add_argument("--min_translation", type=float, default=None)
    parser.add_argument("--max_translation", type=float, default=None)
    parser.add_argument(
        "--min_translation_scale_mult",
        type=float,
        default=BASELINE_EXTENSION_DEFAULTS["min_translation_scale_mult"],
    )
    parser.add_argument(
        "--max_translation_scale_mult",
        type=float,
        default=BASELINE_EXTENSION_DEFAULTS["max_translation_scale_mult"],
    )
    parser.add_argument("--target_dedupe_radius", type=float, default=None)
    parser.add_argument(
        "--target_dedupe_radius_scale_mult",
        type=float,
        default=BASELINE_EXTENSION_DEFAULTS["target_dedupe_radius_scale_mult"],
    )
    parser.add_argument("--global_shell_edges_scale_mults", type=str, default=BASELINE_EXTENSION_DEFAULTS["global_shell_edges_scale_mults"])
    parser.add_argument("--global_shell_stride_scale_mults", type=str, default=BASELINE_EXTENSION_DEFAULTS["global_shell_stride_scale_mults"])
    parser.add_argument("--global_shell_priority_weights", type=str, default=BASELINE_EXTENSION_DEFAULTS["global_shell_priority_weights"])
    parser.add_argument("--global_target_density_scale", type=float, default=BASELINE_EXTENSION_DEFAULTS["global_target_density_scale"])
    parser.add_argument("--global_candidate_to_target_ratio", type=float, default=BASELINE_EXTENSION_DEFAULTS["global_candidate_to_target_ratio"])
    parser.add_argument("--global_min_candidate_target_count", type=int, default=BASELINE_EXTENSION_DEFAULTS["global_min_candidate_target_count"])
    parser.add_argument("--global_max_candidate_target_count", type=int, default=BASELINE_EXTENSION_DEFAULTS["global_max_candidate_target_count"])
    parser.add_argument("--global_max_target_budget", type=int, default=BASELINE_EXTENSION_DEFAULTS["global_max_target_budget"])
    parser.add_argument("--subject_core_bias_strength", type=float, default=BASELINE_EXTENSION_DEFAULTS["subject_core_bias_strength"])
    parser.add_argument("--subject_proxy_mode", choices=["image_center", "gs_structure"], default=BASELINE_EXTENSION_DEFAULTS["subject_proxy_mode"])
    parser.add_argument("--subject_core_camera_stride", type=int, default=BASELINE_EXTENSION_DEFAULTS["subject_core_camera_stride"])
    parser.add_argument("--subject_core_center_sigma", type=float, default=BASELINE_EXTENSION_DEFAULTS["subject_core_center_sigma"])
    parser.add_argument("--subject_core_visibility_power", type=float, default=BASELINE_EXTENSION_DEFAULTS["subject_core_visibility_power"])
    parser.add_argument("--gs_structure_neighbor_k", type=int, default=BASELINE_EXTENSION_DEFAULTS["gs_structure_neighbor_k"])
    parser.add_argument("--gs_structure_radius_scale_mult", type=float, default=BASELINE_EXTENSION_DEFAULTS["gs_structure_radius_scale_mult"])
    parser.add_argument("--gs_structure_batch_size", type=int, default=BASELINE_EXTENSION_DEFAULTS["gs_structure_batch_size"])
    parser.add_argument("--use_foreground_mask", action="store_true", default=BASELINE_EXTENSION_DEFAULTS["use_foreground_mask"])
    parser.add_argument("--foreground_depth_quantile", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_depth_quantile"])
    parser.add_argument("--foreground_min_score", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_min_score"])
    parser.add_argument("--foreground_min_visible_views", type=int, default=BASELINE_EXTENSION_DEFAULTS["foreground_min_visible_views"])
    parser.add_argument("--foreground_camera_stride", type=int, default=BASELINE_EXTENSION_DEFAULTS["foreground_camera_stride"])
    parser.add_argument("--foreground_depth_min", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_depth_min"])
    parser.add_argument("--foreground_anchor_pool_mult", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_anchor_pool_mult"])
    parser.add_argument("--foreground_target_pool_mult", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_target_pool_mult"])
    parser.add_argument("--foreground_anchor_rank_foreground_strength", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_anchor_rank_foreground_strength"])
    parser.add_argument("--foreground_anchor_rank_subject_strength", type=float, default=BASELINE_EXTENSION_DEFAULTS["foreground_anchor_rank_subject_strength"])
    parser.add_argument("--enable_subject_interior_targets", action="store_true", default=BASELINE_EXTENSION_DEFAULTS["enable_subject_interior_targets"])
    parser.add_argument("--interior_subject_core_min", type=float, default=BASELINE_EXTENSION_DEFAULTS["interior_subject_core_min"])
    parser.add_argument("--interior_seed_quantile", type=float, default=BASELINE_EXTENSION_DEFAULTS["interior_seed_quantile"])
    parser.add_argument("--interior_min_visible_views", type=int, default=BASELINE_EXTENSION_DEFAULTS["interior_min_visible_views"])
    parser.add_argument("--interior_weight_scale", type=float, default=BASELINE_EXTENSION_DEFAULTS["interior_weight_scale"])
    parser.add_argument("--interior_target_budget", type=int, default=BASELINE_EXTENSION_DEFAULTS["interior_target_budget"])
    parser.add_argument("--interior_target_fraction", type=float, default=BASELINE_EXTENSION_DEFAULTS["interior_target_fraction"])
    parser.add_argument("--promoted_consensus_records", type=str, default=None, help="per_target_consensus json from a previous round; selected labels are added into support")
    parser.add_argument("--promoted_keep_labels", type=str, default="validated", help="Comma-separated validation_status labels to promote into support")
    parser.add_argument("--exclude_probe_records", type=str, default=None, help="manifest/probe_outcomes json whose target_face_id values should be excluded from the next-round target pool")
    parser.add_argument("--reject_good_zone_reentry", action="store_true", help="Reject new probes whose init position is still closer to existing support than to the intended new target")
    parser.add_argument("--good_zone_reentry_ratio", type=float, default=1.0, help="Reject when distance(new_probe, support) <= ratio * distance(new_probe, target)")
    parser.add_argument("--opacity_scale", type=float, default=0.4)
    parser.add_argument("--min_probe_opacity", type=float, default=0.03)
    parser.add_argument("--max_probe_opacity", type=float, default=0.12)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--output_summary", type=str, default=None)
    parser.add_argument("--output_manifest", type=str, default=None)
    parser.add_argument("--output_preview_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    apply_extension_preset(args)

    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for extension probe injection.")
        args.data_device = "cpu"

    safe_state(args.quiet)
    rng = np.random.default_rng(args.random_seed)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)
    splatting.get_settings(args)

    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe_args.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()
    appearance_embedding = build_appearance_embedding(mesh_args, num_views=len(train_cameras))
    gaussians.training_setup(opt_args, mesh_args, appearance_embedding)

    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    checkpoint_path = resolve_start_checkpoint(dataset.model_path, args.start_checkpoint, loaded_iteration)
    checkpoint_iteration = loaded_iteration
    if os.path.exists(checkpoint_path):
        model_params, checkpoint_iteration, appearance_state = torch.load(checkpoint_path)
        if appearance_embedding is not None and appearance_state[0] is not None:
            appearance_embedding.restore(*appearance_state)
        gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)
        checkpoint_origin = checkpoint_path
    else:
        checkpoint_origin = os.path.join(
            dataset.model_path,
            "point_cloud",
            f"iteration_{loaded_iteration}",
            "point_cloud.ply",
        )
        print(
            f"Checkpoint not found at {checkpoint_path}; "
            f"falling back to loaded point cloud state from iteration {loaded_iteration}."
        )

    gaussians.compute_3D_filter(train_cameras.copy(), CUDA=not pipe_args.compute_filter3D_python)

    checkpoint_out, summary_out, manifest_out, preview_dir = build_default_paths(
        dataset,
        checkpoint_iteration=checkpoint_iteration,
        output_checkpoint=args.output_checkpoint,
        output_summary=args.output_summary,
        output_manifest=args.output_manifest,
        output_preview_dir=args.output_preview_dir,
    )
    os.makedirs(preview_dir, exist_ok=True)

    mesh_obj = trimesh.load_mesh(args.mesh_path, process=False)
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    face_areas = np.asarray(mesh_obj.area_faces, dtype=np.float64)

    proxy_payload = np.load(args.proxy_npz)
    if "seed_face_score" not in proxy_payload:
        raise ValueError(f"proxy npz does not contain seed_face_score: {args.proxy_npz}")
    seed_face_score = np.asarray(proxy_payload["seed_face_score"], dtype=np.float32)
    if seed_face_score.shape[0] != faces.shape[0]:
        raise ValueError("seed_face_score length does not match mesh face count")

    gaussian_xyz_cpu = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    gaussian_opacity_cpu = gaussians.get_opacity.detach().squeeze(-1).cpu().numpy().astype(np.float32)
    gaussian_scaling_cpu = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
    gaussian_scale_max_median = float(np.median(np.max(gaussian_scaling_cpu, axis=1)))
    gs_structure_radius = max(float(args.gs_structure_radius_scale_mult) * gaussian_scale_max_median, 1e-6)

    foreground_summary = None
    foreground_anchor_pool_ids = np.empty((0,), dtype=np.int64)
    foreground_target_pool_ids = np.empty((0,), dtype=np.int64)
    support_candidate_ids = np.empty((0,), dtype=np.int64)
    interior_target_candidate_ids = np.empty((0,), dtype=np.int64)
    interior_target_candidate_scores = np.empty((0,), dtype=np.float32)
    interior_target_candidate_weights = np.empty((0,), dtype=np.float32)
    interior_target_gap_strength = np.empty((0,), dtype=np.float32)
    interior_summary = None
    foreground_anchor_score_lookup: Dict[int, float] = {}
    foreground_anchor_visible_lookup: Dict[int, int] = {}
    foreground_anchor_subject_lookup: Dict[int, float] = {}
    foreground_anchor_rank_lookup: Dict[int, float] = {}
    foreground_anchor_subject_components = None
    promoted_support_face_ids = load_target_face_ids_from_records(
        args.promoted_consensus_records,
        label_field="validation_status",
        keep_labels=parse_label_set(args.promoted_keep_labels),
    )
    excluded_target_face_ids = load_target_face_ids_from_records(args.exclude_probe_records)
    anchor_score_raw_sampled = None
    target_score_raw_sampled = None
    support_threshold = None
    support_seed_threshold = None
    support_area_fraction = max(float(args.support_area_fraction), float(args.anchor_area_fraction))

    if args.use_foreground_mask:
        all_face_ids = np.arange(faces.shape[0], dtype=np.int64)
        anchor_pool_count = int(
            min(
                faces.shape[0],
                max(
                    args.anchor_face_sample_count,
                    int(np.ceil(args.anchor_face_sample_count * max(args.foreground_anchor_pool_mult, 1.0))),
                ),
            )
        )
        all_anchor_weights = face_areas * np.clip(seed_face_score, 1e-6, None)
        foreground_anchor_pool_ids = sample_weighted_face_ids(
            all_face_ids,
            all_anchor_weights,
            anchor_pool_count,
            rng,
        )
        if foreground_anchor_pool_ids.size == 0:
            raise RuntimeError("Foreground mask enabled, but no anchor-pool faces were sampled.")

        foreground_anchor_pool_centers, _ = compute_face_geometry(vertices, faces, foreground_anchor_pool_ids)
        foreground_anchor_score, foreground_anchor_visible, foreground_anchor_near = compute_foreground_support_scores(
            points_xyz=foreground_anchor_pool_centers,
            cameras=train_cameras,
            depth_min=args.foreground_depth_min,
            camera_stride=args.foreground_camera_stride,
            near_quantile=args.foreground_depth_quantile,
        )
        if args.subject_proxy_mode == "gs_structure":
            foreground_anchor_subject_score, _, foreground_anchor_subject_components = compute_gs_structure_scores(
                points_xyz=foreground_anchor_pool_centers,
                gaussian_xyz=gaussian_xyz_cpu,
                gaussian_opacity=gaussian_opacity_cpu,
                gaussian_scaling=gaussian_scaling_cpu,
                neighbor_k=args.gs_structure_neighbor_k,
                radius=gs_structure_radius,
                scale_ref=gaussian_scale_max_median,
                batch_size=args.gs_structure_batch_size,
            )
            foreground_anchor_subject_visible = foreground_anchor_visible
        else:
            foreground_anchor_subject_score, foreground_anchor_subject_visible = compute_subject_core_scores(
                points_xyz=foreground_anchor_pool_centers,
                cameras=train_cameras,
                depth_min=args.foreground_depth_min,
                camera_stride=args.subject_core_camera_stride,
                center_sigma=args.subject_core_center_sigma,
                visibility_power=args.subject_core_visibility_power,
            )
        foreground_anchor_rank_score = blend_score_with_context(
            base_score=seed_face_score[foreground_anchor_pool_ids],
            foreground_score=foreground_anchor_score,
            subject_score=foreground_anchor_subject_score,
            foreground_strength=args.foreground_anchor_rank_foreground_strength,
            subject_strength=args.foreground_anchor_rank_subject_strength,
        )
        foreground_anchor_score_lookup = {
            int(face_id): float(score)
            for face_id, score in zip(foreground_anchor_pool_ids.tolist(), foreground_anchor_score.tolist())
        }
        foreground_anchor_visible_lookup = {
            int(face_id): int(count)
            for face_id, count in zip(foreground_anchor_pool_ids.tolist(), foreground_anchor_visible.tolist())
        }
        foreground_anchor_subject_lookup = {
            int(face_id): float(score)
            for face_id, score in zip(foreground_anchor_pool_ids.tolist(), foreground_anchor_subject_score.tolist())
        }
        foreground_anchor_rank_lookup = {
            int(face_id): float(score)
            for face_id, score in zip(foreground_anchor_pool_ids.tolist(), foreground_anchor_rank_score.tolist())
        }
        foreground_anchor_mask = (
            (foreground_anchor_score >= float(args.foreground_min_score))
            & (foreground_anchor_visible >= int(args.foreground_min_visible_views))
        )
        anchor_foreground_ids = foreground_anchor_pool_ids[foreground_anchor_mask]
        anchor_foreground_rank_scores = foreground_anchor_rank_score[foreground_anchor_mask]
        if anchor_foreground_ids.size == 0:
            raise RuntimeError("Foreground mask removed every anchor candidate. Try lowering foreground_min_score or foreground_min_visible_views.")

        support_threshold = weighted_quantile(
            anchor_foreground_rank_scores,
            face_areas[anchor_foreground_ids],
            1.0 - support_area_fraction,
        )
        support_seed_threshold = weighted_quantile(
            seed_face_score[anchor_foreground_ids],
            face_areas[anchor_foreground_ids],
            1.0 - support_area_fraction,
        )
        support_candidate_mask = anchor_foreground_rank_scores >= support_threshold
        support_candidate_ids = anchor_foreground_ids[support_candidate_mask]

        anchor_threshold = weighted_quantile(
            anchor_foreground_rank_scores,
            face_areas[anchor_foreground_ids],
            1.0 - args.anchor_area_fraction,
        )
        anchor_candidate_mask = anchor_foreground_rank_scores >= anchor_threshold
        anchor_candidate_ids = anchor_foreground_ids[anchor_candidate_mask]
        anchor_candidate_scores = anchor_foreground_rank_scores[anchor_candidate_mask]
        if anchor_candidate_ids.size == 0:
            raise RuntimeError("Foreground-aware anchor threshold left no anchor candidates.")

        all_target_gap = np.clip(float(support_seed_threshold) - seed_face_score, 0.0, None)
        target_pool_source_ids = np.flatnonzero(all_target_gap > 0.0)
        total_target_candidate_area = float(face_areas[target_pool_source_ids].sum())
    else:
        support_threshold = weighted_quantile(seed_face_score, face_areas, 1.0 - support_area_fraction)
        support_seed_threshold = support_threshold
        support_mask = seed_face_score >= support_threshold
        support_candidate_ids = np.flatnonzero(support_mask)
        anchor_threshold = weighted_quantile(seed_face_score, face_areas, 1.0 - args.anchor_area_fraction)
        anchor_mask = seed_face_score >= anchor_threshold
        anchor_candidate_ids = np.flatnonzero(anchor_mask)
        anchor_candidate_scores = seed_face_score[anchor_candidate_ids]
        target_pool_source_ids = np.flatnonzero(seed_face_score < support_threshold)
        total_target_candidate_area = float(face_areas[target_pool_source_ids].sum())

    support_injection_summary = None
    if promoted_support_face_ids.size > 0:
        valid_promoted_support = promoted_support_face_ids[
            (promoted_support_face_ids >= 0)
            & (promoted_support_face_ids < faces.shape[0])
        ]
        support_candidate_ids = np.unique(np.concatenate([support_candidate_ids, valid_promoted_support], axis=0))
        support_injection_summary = {
            "promoted_support_face_count": int(promoted_support_face_ids.size),
            "valid_promoted_support_face_count": int(valid_promoted_support.size),
        }

    exclusion_face_ids = support_candidate_ids
    if excluded_target_face_ids.size > 0:
        valid_excluded_target_faces = excluded_target_face_ids[
            (excluded_target_face_ids >= 0)
            & (excluded_target_face_ids < faces.shape[0])
        ]
        exclusion_face_ids = np.unique(np.concatenate([support_candidate_ids, valid_excluded_target_faces], axis=0))
        if support_injection_summary is None:
            support_injection_summary = {}
        support_injection_summary["excluded_previous_target_face_count"] = int(excluded_target_face_ids.size)
        support_injection_summary["valid_excluded_previous_target_face_count"] = int(valid_excluded_target_faces.size)

    if exclusion_face_ids.size > 0:
        keep_target_mask = ~np.isin(target_pool_source_ids, exclusion_face_ids)
        target_pool_source_ids = target_pool_source_ids[keep_target_mask]
        total_target_candidate_area = float(face_areas[target_pool_source_ids].sum())

    if args.use_foreground_mask and args.enable_subject_interior_targets:
        interior_pool_ids = support_candidate_ids
        if anchor_candidate_ids.size > 0:
            interior_pool_ids = interior_pool_ids[~np.isin(interior_pool_ids, anchor_candidate_ids)]
        if excluded_target_face_ids.size > 0:
            interior_pool_ids = interior_pool_ids[~np.isin(interior_pool_ids, excluded_target_face_ids)]

        if interior_pool_ids.size > 0:
            interior_raw_seed = seed_face_score[interior_pool_ids]
            interior_subject = np.asarray(
                [foreground_anchor_subject_lookup.get(int(face_id), 0.0) for face_id in interior_pool_ids.tolist()],
                dtype=np.float32,
            )
            interior_visible = np.asarray(
                [foreground_anchor_visible_lookup.get(int(face_id), 0) for face_id in interior_pool_ids.tolist()],
                dtype=np.int32,
            )
            interior_foreground = np.asarray(
                [foreground_anchor_score_lookup.get(int(face_id), 0.0) for face_id in interior_pool_ids.tolist()],
                dtype=np.float32,
            )
            interior_rank = np.asarray(
                [foreground_anchor_rank_lookup.get(int(face_id), 0.0) for face_id in interior_pool_ids.tolist()],
                dtype=np.float32,
            )
            interior_seed_threshold = weighted_quantile(
                interior_raw_seed,
                face_areas[interior_pool_ids],
                float(np.clip(args.interior_seed_quantile, 0.0, 1.0)),
            )
            interior_visible_min = max(int(args.foreground_min_visible_views), int(args.interior_min_visible_views))
            interior_mask = (
                (interior_subject >= float(args.interior_subject_core_min))
                & (interior_visible >= interior_visible_min)
                & (interior_foreground >= float(args.foreground_min_score))
                & (interior_raw_seed <= float(interior_seed_threshold))
            )
            interior_target_candidate_ids = interior_pool_ids[interior_mask]
            if interior_target_candidate_ids.size > 0:
                interior_gap_strength = np.clip(
                    float(interior_seed_threshold) - interior_raw_seed[interior_mask],
                    1e-6,
                    None,
                ).astype(np.float32)
                interior_target_gap_strength = interior_gap_strength
                interior_target_candidate_scores = (
                    float(support_threshold) - interior_gap_strength
                ).astype(np.float32)
                interior_target_candidate_weights = (
                    face_areas[interior_target_candidate_ids]
                    * (interior_gap_strength ** max(args.target_score_power, 1e-6))
                    * np.clip(0.5 + 0.5 * interior_subject[interior_mask], 1e-3, None)
                    * max(float(args.interior_weight_scale), 1e-6)
                ).astype(np.float32)
            interior_summary = {
                "enabled": True,
                "subject_core_min": float(args.interior_subject_core_min),
                "seed_quantile": float(args.interior_seed_quantile),
                "seed_threshold": float(interior_seed_threshold),
                "min_visible_views": int(interior_visible_min),
                "weight_scale": float(args.interior_weight_scale),
                "pool_face_count": int(interior_pool_ids.size),
                "candidate_face_count": int(interior_target_candidate_ids.size),
                "candidate_face_area": float(face_areas[interior_target_candidate_ids].sum()) if interior_target_candidate_ids.size > 0 else 0.0,
                "pool_subject_core_score": stats_from_array(interior_subject),
                "pool_visible_count": stats_from_array(interior_visible.astype(np.float32)),
                "pool_foreground_score": stats_from_array(interior_foreground),
                "pool_rank_score": stats_from_array(interior_rank),
                "pool_seed_score": stats_from_array(interior_raw_seed.astype(np.float32)),
                "candidate_gap_strength": stats_from_array(interior_target_gap_strength.astype(np.float32)),
            }
        else:
            interior_summary = {
                "enabled": True,
                "subject_core_min": float(args.interior_subject_core_min),
                "seed_quantile": float(args.interior_seed_quantile),
                "min_visible_views": int(max(int(args.foreground_min_visible_views), int(args.interior_min_visible_views))),
                "weight_scale": float(args.interior_weight_scale),
                "pool_face_count": 0,
                "candidate_face_count": 0,
                "candidate_face_area": 0.0,
            }

    anchor_weights = face_areas[anchor_candidate_ids] * np.clip(anchor_candidate_scores, 1e-6, None)
    sampled_anchor_face_ids = sample_weighted_face_ids(
        anchor_candidate_ids,
        anchor_weights,
        args.anchor_face_sample_count,
        rng,
    )
    if sampled_anchor_face_ids.size == 0:
        raise RuntimeError("No anchor faces were sampled from the proxy strong seed region.")

    anchor_centers, anchor_normals = compute_face_geometry(vertices, faces, sampled_anchor_face_ids)
    if args.use_foreground_mask:
        anchor_score_lookup = {int(face_id): float(score) for face_id, score in zip(anchor_candidate_ids.tolist(), anchor_candidate_scores.tolist())}
        anchor_scores = np.asarray([anchor_score_lookup[int(face_id)] for face_id in sampled_anchor_face_ids], dtype=np.float32)
    else:
        anchor_scores = seed_face_score[sampled_anchor_face_ids]
    anchor_score_raw_sampled = seed_face_score[sampled_anchor_face_ids]

    global_search_config = None
    if args.search_mode == "global_mesh":
        global_search_config = resolve_global_search_config(args, gaussian_scale_max_median)

    target_priority_raw_all = np.clip(float(support_seed_threshold) - seed_face_score[target_pool_source_ids], 1e-6, None) ** max(args.target_score_power, 1e-6)
    target_weights_all = face_areas[target_pool_source_ids] * target_priority_raw_all
    target_candidate_source_codes = np.empty((0,), dtype=np.int32)

    if args.use_foreground_mask:
        resolved_candidate_target_sample_count = int(args.candidate_target_sample_count)
        if args.search_mode == "global_mesh":
            if resolved_candidate_target_sample_count <= 0:
                resolved_candidate_target_sample_count = resolve_auto_candidate_target_count(
                    total_target_area=total_target_candidate_area,
                    stride=float(global_search_config["shell_stride"][0]),
                    args=args,
                )
            resolved_candidate_target_sample_count = min(resolved_candidate_target_sample_count, int(target_pool_source_ids.size))
        else:
            resolved_candidate_target_sample_count = int(args.candidate_target_sample_count)
            resolved_candidate_target_sample_count = min(resolved_candidate_target_sample_count, int(target_pool_source_ids.size))

        target_pool_count = int(
            min(
                target_pool_source_ids.size,
                max(
                    resolved_candidate_target_sample_count,
                    int(np.ceil(resolved_candidate_target_sample_count * max(args.foreground_target_pool_mult, 1.0))),
                ),
            )
        )
        foreground_target_pool_ids = sample_weighted_face_ids(
            target_pool_source_ids,
            target_weights_all,
            target_pool_count,
            rng,
        )
        if foreground_target_pool_ids.size == 0:
            raise RuntimeError("Foreground mask enabled, but no target-pool faces were sampled.")

        foreground_target_pool_centers, _ = compute_face_geometry(vertices, faces, foreground_target_pool_ids)
        foreground_target_score, foreground_target_visible, foreground_target_near = compute_foreground_support_scores(
            points_xyz=foreground_target_pool_centers,
            cameras=train_cameras,
            depth_min=args.foreground_depth_min,
            camera_stride=args.foreground_camera_stride,
            near_quantile=args.foreground_depth_quantile,
        )
        if args.subject_proxy_mode == "gs_structure":
            foreground_target_subject_score = np.ones((foreground_target_pool_ids.shape[0],), dtype=np.float32)
            foreground_target_subject_visible = foreground_target_visible
        else:
            foreground_target_subject_score, foreground_target_subject_visible = compute_subject_core_scores(
                points_xyz=foreground_target_pool_centers,
                cameras=train_cameras,
                depth_min=args.foreground_depth_min,
                camera_stride=args.subject_core_camera_stride,
                center_sigma=args.subject_core_center_sigma,
                visibility_power=args.subject_core_visibility_power,
            )
        foreground_target_rank_score = blend_score_with_context(
            base_score=seed_face_score[foreground_target_pool_ids],
            foreground_score=foreground_target_score,
            subject_score=foreground_target_subject_score if args.subject_proxy_mode != "gs_structure" else None,
            foreground_strength=args.foreground_anchor_rank_foreground_strength,
            subject_strength=args.foreground_anchor_rank_subject_strength if args.subject_proxy_mode != "gs_structure" else 0.0,
        )
        foreground_target_mask = (
            (foreground_target_score >= float(args.foreground_min_score))
            & (foreground_target_visible >= int(args.foreground_min_visible_views))
            & (foreground_target_rank_score < float(support_threshold))
        )
        target_candidate_ids = foreground_target_pool_ids[foreground_target_mask]
        if target_candidate_ids.size == 0 and interior_target_candidate_ids.size == 0:
            raise RuntimeError("Foreground mask removed every target candidate. Try lowering foreground_min_score or foreground_min_visible_views.")
        target_candidate_scores = foreground_target_rank_score[foreground_target_mask]
        target_priority_raw = np.clip(float(support_threshold) - target_candidate_scores, 1e-6, None) ** max(args.target_score_power, 1e-6)
        target_fg_score = foreground_target_score[foreground_target_mask]
        target_weights = face_areas[target_candidate_ids] * target_priority_raw * np.clip(0.5 + 0.5 * target_fg_score, 1e-3, None)
        target_candidate_source_codes = np.full(target_candidate_ids.shape[0], TARGET_SOURCE_FRONTIER, dtype=np.int32)

        if interior_target_candidate_ids.size > 0:
            target_candidate_ids = np.concatenate([target_candidate_ids, interior_target_candidate_ids], axis=0)
            target_candidate_scores = np.concatenate([target_candidate_scores, interior_target_candidate_scores], axis=0)
            target_weights = np.concatenate([target_weights, interior_target_candidate_weights], axis=0)
            target_candidate_source_codes = np.concatenate(
                [
                    target_candidate_source_codes,
                    np.full(interior_target_candidate_ids.shape[0], TARGET_SOURCE_INTERIOR, dtype=np.int32),
                ],
                axis=0,
            )
        foreground_summary = {
            "enabled": True,
            "depth_quantile": float(args.foreground_depth_quantile),
            "min_score": float(args.foreground_min_score),
            "min_visible_views": int(args.foreground_min_visible_views),
            "subject_proxy_mode": str(args.subject_proxy_mode),
            "camera_stride": int(args.foreground_camera_stride),
            "depth_min": float(args.foreground_depth_min),
            "target_pool_source_face_count": int(target_pool_source_ids.size),
            "target_pool_source_face_area": total_target_candidate_area,
            "support_area_fraction": float(support_area_fraction),
            "support_threshold": float(support_threshold),
            "support_seed_threshold": float(support_seed_threshold),
            "anchor_pool_face_count": int(foreground_anchor_pool_ids.size),
            "anchor_pool_foreground_face_count": int(anchor_foreground_ids.size),
            "support_face_count": int(support_candidate_ids.size),
            "anchor_pool_subject_core_score": stats_from_array(foreground_anchor_subject_score),
            "anchor_pool_subject_visible_count": stats_from_array(foreground_anchor_subject_visible.astype(np.float32)),
            "anchor_pool_rank_score": stats_from_array(foreground_anchor_rank_score),
            "target_pool_face_count": int(foreground_target_pool_ids.size),
            "target_pool_foreground_face_count": int(target_candidate_ids.size),
            "anchor_pool_foreground_score": stats_from_array(foreground_anchor_score),
            "anchor_pool_visible_count": stats_from_array(foreground_anchor_visible.astype(np.float32)),
            "target_pool_foreground_score": stats_from_array(foreground_target_score),
            "target_pool_visible_count": stats_from_array(foreground_target_visible.astype(np.float32)),
            "target_pool_subject_core_score": stats_from_array(foreground_target_subject_score),
            "target_pool_subject_visible_count": stats_from_array(foreground_target_subject_visible.astype(np.float32)),
            "target_pool_rank_score": stats_from_array(foreground_target_rank_score),
        }
        if foreground_anchor_subject_components is not None:
            foreground_summary["anchor_pool_gs_structure_score"] = stats_from_array(foreground_anchor_subject_score)
            foreground_summary["anchor_pool_gs_density_raw"] = stats_from_array(foreground_anchor_subject_components["density_raw"])
            foreground_summary["anchor_pool_gs_density_norm"] = stats_from_array(foreground_anchor_subject_components["density_norm"])
            foreground_summary["anchor_pool_gs_small_scale"] = stats_from_array(foreground_anchor_subject_components["small_scale"])
            foreground_summary["anchor_pool_gs_anisotropy"] = stats_from_array(foreground_anchor_subject_components["anisotropy"])
            foreground_summary["anchor_pool_gs_nonplanar"] = stats_from_array(foreground_anchor_subject_components["nonplanar_norm"])
    else:
        resolved_candidate_target_sample_count = int(args.candidate_target_sample_count)
        if args.search_mode == "global_mesh":
            if resolved_candidate_target_sample_count <= 0:
                resolved_candidate_target_sample_count = resolve_auto_candidate_target_count(
                    total_target_area=total_target_candidate_area,
                    stride=float(global_search_config["shell_stride"][0]),
                    args=args,
                )
            resolved_candidate_target_sample_count = min(resolved_candidate_target_sample_count, int(target_pool_source_ids.size))
        else:
            resolved_candidate_target_sample_count = int(args.candidate_target_sample_count)
            resolved_candidate_target_sample_count = min(resolved_candidate_target_sample_count, int(target_pool_source_ids.size))
        target_candidate_ids = target_pool_source_ids
        target_candidate_scores = seed_face_score[target_candidate_ids]
        target_priority_raw = np.clip(float(support_threshold) - target_candidate_scores, 1e-6, None) ** max(args.target_score_power, 1e-6)
        target_weights = target_weights_all
        target_candidate_source_codes = np.full(target_candidate_ids.shape[0], TARGET_SOURCE_FRONTIER, dtype=np.int32)

    target_candidate_area = float(face_areas[target_candidate_ids].sum())
    if args.search_mode == "global_mesh":
        if int(args.candidate_target_sample_count) <= 0:
            resolved_candidate_target_sample_count = resolve_auto_candidate_target_count(
                total_target_area=target_candidate_area,
                stride=float(global_search_config["shell_stride"][0]),
                args=args,
            )
        else:
            resolved_candidate_target_sample_count = int(args.candidate_target_sample_count)
    else:
        resolved_candidate_target_sample_count = int(args.candidate_target_sample_count)
    resolved_candidate_target_sample_count = min(resolved_candidate_target_sample_count, int(target_candidate_ids.size))

    sampled_target_face_ids = sample_weighted_face_ids(
        target_candidate_ids,
        target_weights,
        resolved_candidate_target_sample_count,
        rng,
    )
    if sampled_target_face_ids.size == 0:
        raise RuntimeError("No non-anchor target faces were sampled.")
    target_centers, target_normals = compute_face_geometry(vertices, faces, sampled_target_face_ids)
    if args.use_foreground_mask:
        target_score_lookup = {int(face_id): float(score) for face_id, score in zip(target_candidate_ids.tolist(), target_candidate_scores.tolist())}
        target_scores = np.asarray([target_score_lookup[int(face_id)] for face_id in sampled_target_face_ids], dtype=np.float32)
    else:
        target_scores = seed_face_score[sampled_target_face_ids]
    target_source_lookup = {
        int(face_id): int(source_code)
        for face_id, source_code in zip(target_candidate_ids.tolist(), target_candidate_source_codes.tolist())
    }
    sampled_target_source_codes = np.asarray(
        [target_source_lookup[int(face_id)] for face_id in sampled_target_face_ids],
        dtype=np.int32,
    )
    target_score_raw_sampled = seed_face_score[sampled_target_face_ids]

    if args.subject_proxy_mode == "gs_structure":
        target_subject_core, target_subject_visible_count, target_subject_components = compute_gs_structure_scores(
            points_xyz=target_centers,
            gaussian_xyz=gaussian_xyz_cpu,
            gaussian_opacity=gaussian_opacity_cpu,
            gaussian_scaling=gaussian_scaling_cpu,
            neighbor_k=args.gs_structure_neighbor_k,
            radius=gs_structure_radius,
            scale_ref=gaussian_scale_max_median,
            batch_size=args.gs_structure_batch_size,
        )
    else:
        target_subject_core, target_subject_visible_count = compute_subject_core_scores(
            points_xyz=target_centers,
            cameras=train_cameras,
            depth_min=0.2,
            camera_stride=args.subject_core_camera_stride,
            center_sigma=args.subject_core_center_sigma,
            visibility_power=args.subject_core_visibility_power,
        )
        target_subject_components = None

    min_translation = (
        float(args.min_translation)
        if args.min_translation is not None
        else float(args.min_translation_scale_mult) * gaussian_scale_max_median
    )
    max_translation = (
        float(args.max_translation)
        if args.max_translation is not None
        else float(args.max_translation_scale_mult) * gaussian_scale_max_median
    )
    target_dedupe_radius = (
        float(args.target_dedupe_radius)
        if args.target_dedupe_radius is not None
        else float(args.target_dedupe_radius_scale_mult) * gaussian_scale_max_median
    )

    matched = pick_anchor_for_targets(
        anchor_centers=anchor_centers,
        anchor_normals=anchor_normals,
        anchor_face_ids=sampled_anchor_face_ids,
        anchor_scores=anchor_scores,
        target_centers=target_centers,
        target_normals=target_normals,
        target_face_ids=sampled_target_face_ids,
        target_scores=target_scores,
        target_subject_core=target_subject_core,
        k_neighbors=args.anchor_neighbor_k,
        min_translation=min_translation,
        max_translation=max_translation,
        min_normal_dot=args.min_normal_dot,
    )

    if matched["target_face_ids"].size == 0:
        raise RuntimeError("No valid anchor-target pairs survived the frontier compatibility filters.")

    matched_target_source_lookup = {
        int(face_id): int(source_code)
        for face_id, source_code in zip(sampled_target_face_ids.tolist(), sampled_target_source_codes.tolist())
    }
    matched["target_source_code"] = np.asarray(
        [matched_target_source_lookup[int(face_id)] for face_id in matched["target_face_ids"].tolist()],
        dtype=np.int32,
    )

    pair_priority_base = (
        np.clip(matched["anchor_scores"], 1e-6, None)
        * (np.clip(float(support_threshold) - matched["target_scores"], 1e-6, None) ** max(args.pair_target_gap_power, 1e-6))
        / np.maximum(1.0 + matched["translation_distance"], 1e-6)
    ).astype(np.float32)
    pair_priority = (
        pair_priority_base
        * (1.0 + float(args.subject_core_bias_strength) * np.clip(matched["target_subject_core"], 0.0, None))
    ).astype(np.float32)

    if args.search_mode == "global_mesh":
        selected_local_ids, global_search_selection = select_targets_global_search(
            matched=matched,
            pair_priority=pair_priority,
            face_areas=face_areas,
            gaussian_scale_max_median=gaussian_scale_max_median,
            args=args,
            global_search_config=global_search_config,
        )
    else:
        selected_local_ids = voxel_sparse_select(
            points=matched["target_centers"],
            priorities=pair_priority,
            max_count=args.target_budget,
            voxel_size=target_dedupe_radius,
        )
        global_search_selection = None

    if selected_local_ids.size == 0:
        raise RuntimeError("No extension probes were selected after target dedupe / global shell search.")

    selected = {key: value[selected_local_ids] for key, value in matched.items()}
    selected["pair_priority_base"] = pair_priority_base[selected_local_ids]
    selected["pair_priority"] = pair_priority[selected_local_ids]
    if global_search_selection is not None:
        selected["pair_priority_weighted"] = global_search_selection["pair_priority_weighted"][selected_local_ids]
        selected["shell_id"] = global_search_selection["shell_ids"][selected_local_ids]
    else:
        selected["pair_priority_weighted"] = selected["pair_priority"]
        selected["shell_id"] = np.full((selected["target_face_ids"].shape[0],), -1, dtype=np.int32)
    selected["translation_scale"] = (
        selected["translation_distance"] / max(float(gaussian_scale_max_median), 1e-6)
    ).astype(np.float32)

    gaussian_tree = cKDTree(gaussian_xyz_cpu)
    parent_gaussian_ids = choose_parent_gaussians(
        gaussian_tree=gaussian_tree,
        gaussian_xyz=gaussian_xyz_cpu,
        gaussian_opacity=gaussian_opacity_cpu,
        anchor_centers=selected["anchor_centers"],
        k_neighbors=args.parent_neighbor_k,
        min_parent_opacity=args.min_parent_opacity,
    )

    parent_idx = torch.from_numpy(parent_gaussian_ids).to(device="cuda", dtype=torch.long)
    parent_xyz = gaussians.get_xyz.detach()[parent_idx]
    parent_features_dc = gaussians._features_dc.detach()[parent_idx].clone()
    parent_features_rest = gaussians._features_rest.detach()[parent_idx].clone()
    parent_scaling = gaussians._scaling.detach()[parent_idx].clone()
    parent_rotation = gaussians._rotation.detach()[parent_idx].clone()
    parent_opacity = gaussians.get_opacity.detach()[parent_idx]
    parent_generation = gaussians._generation.detach()[parent_idx]
    parent_seed_id = gaussians._seed_id.detach()[parent_idx]
    parent_source_tag = gaussians._source_tag.detach()[parent_idx]

    anchor_centers_torch = torch.from_numpy(selected["anchor_centers"]).to(device="cuda", dtype=torch.float32)
    target_centers_torch = torch.from_numpy(selected["target_centers"]).to(device="cuda", dtype=torch.float32)
    anchor_normals_torch = torch.from_numpy(selected["anchor_normals"]).to(device="cuda", dtype=torch.float32)

    offset = target_centers_torch - anchor_centers_torch
    tangent_offset = offset - torch.sum(offset * anchor_normals_torch, dim=1, keepdim=True) * anchor_normals_torch
    new_xyz = parent_xyz + tangent_offset

    iterative_mode_enabled = bool(args.promoted_consensus_records or args.exclude_probe_records)
    good_zone_reentry_enabled = bool(args.reject_good_zone_reentry or iterative_mode_enabled)
    good_zone_reentry_summary = {
        "enabled": good_zone_reentry_enabled,
        "good_zone_reentry_ratio": float(args.good_zone_reentry_ratio),
        "iterative_mode_enabled": iterative_mode_enabled,
    }
    reentry_rejected_xyz = np.empty((0, 3), dtype=np.float32)
    if good_zone_reentry_enabled and support_candidate_ids.size > 0 and new_xyz.shape[0] > 0:
        support_centers, _ = compute_face_geometry(vertices, faces, support_candidate_ids)
        support_tree = cKDTree(support_centers.astype(np.float32))
        d_support, _ = support_tree.query(new_xyz.detach().cpu().numpy().astype(np.float32), k=1)
        d_target = np.linalg.norm(new_xyz.detach().cpu().numpy().astype(np.float32) - selected["target_centers"].astype(np.float32), axis=1)
        keep_mask_np = d_support > float(args.good_zone_reentry_ratio) * np.maximum(d_target, 1e-6)
        reentry_rejected_xyz = new_xyz.detach().cpu().numpy().astype(np.float32)[~keep_mask_np]
        good_zone_reentry_summary.update(
            {
                "support_face_count": int(support_candidate_ids.size),
                "kept_probe_count": int(keep_mask_np.sum()),
                "rejected_probe_count": int((~keep_mask_np).sum()),
                "distance_to_support": stats_from_array(d_support.astype(np.float32)),
                "distance_to_target": stats_from_array(d_target.astype(np.float32)),
            }
        )
        if not np.any(keep_mask_np):
            raise RuntimeError("Good-zone reentry filter rejected every probe. Try lowering good_zone_reentry_ratio or disabling the filter.")
        keep_mask_torch = torch.from_numpy(keep_mask_np).to(device="cuda", dtype=torch.bool)
        selected = {key: value[keep_mask_np] for key, value in selected.items()}
        parent_gaussian_ids = parent_gaussian_ids[keep_mask_np]
        parent_idx = parent_idx[keep_mask_torch]
        parent_xyz = parent_xyz[keep_mask_torch]
        parent_features_dc = parent_features_dc[keep_mask_torch]
        parent_features_rest = parent_features_rest[keep_mask_torch]
        parent_scaling = parent_scaling[keep_mask_torch]
        parent_rotation = parent_rotation[keep_mask_torch]
        parent_opacity = parent_opacity[keep_mask_torch]
        parent_generation = parent_generation[keep_mask_torch]
        parent_seed_id = parent_seed_id[keep_mask_torch]
        parent_source_tag = parent_source_tag[keep_mask_torch]
        anchor_centers_torch = anchor_centers_torch[keep_mask_torch]
        target_centers_torch = target_centers_torch[keep_mask_torch]
        anchor_normals_torch = anchor_normals_torch[keep_mask_torch]
        tangent_offset = tangent_offset[keep_mask_torch]
        new_xyz = new_xyz[keep_mask_torch]
    else:
        good_zone_reentry_summary["support_face_count"] = int(support_candidate_ids.size)

    new_opacity_values = torch.clamp(
        parent_opacity * args.opacity_scale,
        min=args.min_probe_opacity,
        max=args.max_probe_opacity,
    )
    new_opacity = gaussians.inverse_opacity_activation(new_opacity_values)

    next_seed_id = int((gaussians._seed_id.max().item() + 1) if gaussians._seed_id.numel() > 0 else 0)
    num_probes = int(new_xyz.shape[0])
    probe_seed_ids = torch.arange(next_seed_id, next_seed_id + num_probes, device="cuda", dtype=torch.int64)
    probe_generation = (parent_generation + 1).to(dtype=torch.int32)
    tracking_state = gaussians._build_tracking_extension(
        num_probes,
        source_tag=torch.full(
            (num_probes,),
            int(GaussianSourceTag.EXTENSION_PROBE),
            device="cuda",
            dtype=torch.int32,
        ),
        seed_id=probe_seed_ids,
        generation=probe_generation,
    )

    gaussians.densification_postfix(
        new_xyz=new_xyz,
        new_features_dc=parent_features_dc,
        new_features_rest=parent_features_rest,
        new_opacities=new_opacity,
        new_scaling=parent_scaling,
        new_rotation=parent_rotation,
        tracking_state=tracking_state,
    )
    gaussians.compute_3D_filter(train_cameras.copy(), CUDA=not pipe_args.compute_filter3D_python)

    checkpoint_dir = os.path.dirname(checkpoint_out)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    appearance_capture = appearance_embedding.capture() if appearance_embedding is not None else (None, None)
    torch.save((gaussians.capture(), checkpoint_iteration, appearance_capture), checkpoint_out)

    preview_tags = os.path.join(preview_dir, "gaussian_tags.pt")
    preview_point_cloud = os.path.join(preview_dir, "point_cloud.ply")
    gaussians.save_ply(preview_point_cloud)
    gaussians.save_tracking_metadata(preview_tags)

    selected_anchor_faces_path = os.path.join(preview_dir, "selected_anchor_faces.ply")
    support_candidate_faces_path = os.path.join(preview_dir, "support_candidate_faces.ply")
    selected_target_faces_path = os.path.join(preview_dir, "selected_target_faces.ply")
    selected_frontier_target_faces_path = os.path.join(preview_dir, "selected_frontier_target_faces.ply")
    selected_interior_target_faces_path = os.path.join(preview_dir, "selected_interior_target_faces.ply")
    interior_target_candidate_faces_path = os.path.join(preview_dir, "interior_target_candidate_faces.ply")
    promoted_support_faces_path = os.path.join(preview_dir, "promoted_support_faces.ply")
    export_face_subset(mesh_obj, selected["anchor_face_ids"], selected_anchor_faces_path)
    export_face_subset(mesh_obj, support_candidate_ids, support_candidate_faces_path)
    export_face_subset(mesh_obj, selected["target_face_ids"], selected_target_faces_path)
    export_face_subset(
        mesh_obj,
        selected["target_face_ids"][selected["target_source_code"] == TARGET_SOURCE_FRONTIER],
        selected_frontier_target_faces_path,
    )
    export_face_subset(
        mesh_obj,
        selected["target_face_ids"][selected["target_source_code"] == TARGET_SOURCE_INTERIOR],
        selected_interior_target_faces_path,
    )
    export_face_subset(mesh_obj, interior_target_candidate_ids, interior_target_candidate_faces_path)
    export_face_subset(mesh_obj, promoted_support_face_ids, promoted_support_faces_path)
    foreground_anchor_pool_faces_path = os.path.join(preview_dir, "foreground_anchor_pool_faces.ply")
    foreground_target_pool_faces_path = os.path.join(preview_dir, "foreground_target_pool_faces.ply")
    if args.use_foreground_mask:
        export_face_subset(mesh_obj, foreground_anchor_pool_ids, foreground_anchor_pool_faces_path)
        export_face_subset(mesh_obj, foreground_target_pool_ids, foreground_target_pool_faces_path)

    probe_points_path = os.path.join(preview_dir, "probe_points_init.ply")
    save_point_cloud(new_xyz.detach().cpu().numpy(), probe_points_path)
    rejected_reentry_points_path = os.path.join(preview_dir, "probe_points_reentry_rejected.ply")
    if reentry_rejected_xyz.shape[0] > 0:
        save_point_cloud(reentry_rejected_xyz, rejected_reentry_points_path)

    manifest_records = []
    new_xyz_cpu = new_xyz.detach().cpu().numpy()
    tangent_offset_cpu = tangent_offset.detach().cpu().numpy()
    parent_xyz_cpu = parent_xyz.detach().cpu().numpy()
    probe_opacity_cpu = new_opacity_values.detach().cpu().numpy().reshape(-1)
    parent_opacity_cpu_selected = parent_opacity.detach().cpu().numpy().reshape(-1)
    parent_generation_cpu = parent_generation.detach().cpu().numpy().reshape(-1)
    parent_seed_id_cpu = parent_seed_id.detach().cpu().numpy().reshape(-1)
    parent_source_tag_cpu = parent_source_tag.detach().cpu().numpy().reshape(-1)
    probe_seed_id_cpu = probe_seed_ids.detach().cpu().numpy().reshape(-1)
    probe_generation_cpu = probe_generation.detach().cpu().numpy().reshape(-1)

    for idx in range(num_probes):
        manifest_records.append(
            {
                "probe_index": idx,
                "seed_id": int(probe_seed_id_cpu[idx]),
                "generation": int(probe_generation_cpu[idx]),
                "parent_gaussian_id": int(parent_gaussian_ids[idx]),
                "parent_seed_id": int(parent_seed_id_cpu[idx]),
                "parent_source_tag": int(parent_source_tag_cpu[idx]),
                "parent_generation": int(parent_generation_cpu[idx]),
                "anchor_face_id": int(selected["anchor_face_ids"][idx]),
                "target_face_id": int(selected["target_face_ids"][idx]),
                "target_source_code": int(selected["target_source_code"][idx]),
                "target_source": TARGET_SOURCE_NAME[int(selected["target_source_code"][idx])],
                "anchor_score": float(selected["anchor_scores"][idx]),
                "target_score": float(selected["target_scores"][idx]),
                "target_subject_core": float(selected["target_subject_core"][idx]),
                "pair_priority_base": float(selected["pair_priority_base"][idx]),
                "pair_priority": float(selected["pair_priority"][idx]),
                "pair_priority_weighted": float(selected["pair_priority_weighted"][idx]),
                "shell_id": int(selected["shell_id"][idx]),
                "translation_distance": float(selected["translation_distance"][idx]),
                "translation_scale": float(selected["translation_scale"][idx]),
                "normal_dot": float(selected["normal_dot"][idx]),
                "parent_opacity": float(parent_opacity_cpu_selected[idx]),
                "probe_opacity_init": float(probe_opacity_cpu[idx]),
                "parent_xyz": parent_xyz_cpu[idx].tolist(),
                "anchor_center": selected["anchor_centers"][idx].tolist(),
                "target_center": selected["target_centers"][idx].tolist(),
                "tangent_offset": tangent_offset_cpu[idx].tolist(),
                "probe_xyz_init": new_xyz_cpu[idx].tolist(),
            }
        )

    manifest_dir = os.path.dirname(manifest_out)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    Path(manifest_out).write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")

    summary = {
        "mode": "surface_extension_probe",
        "baseline_checkpoint": checkpoint_origin,
        "output_checkpoint": checkpoint_out,
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "proxy_npz": str(Path(args.proxy_npz).resolve()),
        "preview_dir": str(Path(preview_dir).resolve()),
        "manifest_path": str(Path(manifest_out).resolve()),
        "parameters": {
            "extension_preset": args.extension_preset,
            "search_mode": args.search_mode,
            "anchor_area_fraction": args.anchor_area_fraction,
            "support_area_fraction": support_area_fraction,
            "anchor_face_sample_count": args.anchor_face_sample_count,
            "candidate_target_sample_count": args.candidate_target_sample_count,
            "resolved_candidate_target_sample_count": resolved_candidate_target_sample_count,
            "target_budget": args.target_budget,
            "global_target_density_scale": args.global_target_density_scale,
            "global_candidate_to_target_ratio": args.global_candidate_to_target_ratio,
            "global_min_candidate_target_count": args.global_min_candidate_target_count,
            "global_max_candidate_target_count": args.global_max_candidate_target_count,
            "global_max_target_budget": args.global_max_target_budget,
            "subject_core_bias_strength": args.subject_core_bias_strength,
            "subject_proxy_mode": args.subject_proxy_mode,
            "subject_core_camera_stride": args.subject_core_camera_stride,
            "subject_core_center_sigma": args.subject_core_center_sigma,
            "subject_core_visibility_power": args.subject_core_visibility_power,
            "gs_structure_neighbor_k": args.gs_structure_neighbor_k,
            "gs_structure_radius_scale_mult": args.gs_structure_radius_scale_mult,
            "gs_structure_batch_size": args.gs_structure_batch_size,
            "use_foreground_mask": bool(args.use_foreground_mask),
            "foreground_depth_quantile": args.foreground_depth_quantile,
            "foreground_min_score": args.foreground_min_score,
            "foreground_min_visible_views": args.foreground_min_visible_views,
            "foreground_camera_stride": args.foreground_camera_stride,
            "foreground_depth_min": args.foreground_depth_min,
            "foreground_anchor_pool_mult": args.foreground_anchor_pool_mult,
            "foreground_target_pool_mult": args.foreground_target_pool_mult,
            "foreground_anchor_rank_foreground_strength": args.foreground_anchor_rank_foreground_strength,
            "foreground_anchor_rank_subject_strength": args.foreground_anchor_rank_subject_strength,
            "enable_subject_interior_targets": bool(args.enable_subject_interior_targets),
            "interior_subject_core_min": args.interior_subject_core_min,
            "interior_seed_quantile": args.interior_seed_quantile,
            "interior_min_visible_views": args.interior_min_visible_views,
            "interior_weight_scale": args.interior_weight_scale,
            "interior_target_budget": args.interior_target_budget,
            "interior_target_fraction": args.interior_target_fraction,
            "promoted_consensus_records": args.promoted_consensus_records,
            "promoted_keep_labels": args.promoted_keep_labels,
            "exclude_probe_records": args.exclude_probe_records,
            "reject_good_zone_reentry": bool(args.reject_good_zone_reentry),
            "good_zone_reentry_ratio": float(args.good_zone_reentry_ratio),
            "anchor_neighbor_k": args.anchor_neighbor_k,
            "parent_neighbor_k": args.parent_neighbor_k,
            "min_parent_opacity": args.min_parent_opacity,
            "min_normal_dot": args.min_normal_dot,
            "target_score_power": args.target_score_power,
            "pair_target_gap_power": args.pair_target_gap_power,
            "min_translation": min_translation,
            "max_translation": max_translation,
            "min_translation_scale_mult": args.min_translation_scale_mult,
            "max_translation_scale_mult": args.max_translation_scale_mult,
            "target_dedupe_radius": target_dedupe_radius,
            "target_dedupe_radius_scale_mult": args.target_dedupe_radius_scale_mult,
            "opacity_scale": args.opacity_scale,
            "min_probe_opacity": args.min_probe_opacity,
            "max_probe_opacity": args.max_probe_opacity,
            "random_seed": args.random_seed,
        },
        "anchor_zone": {
            "core_threshold": float(anchor_threshold),
            "support_threshold": float(support_threshold),
            "support_seed_threshold": float(support_seed_threshold),
            "threshold": anchor_threshold,
            "face_count": int(anchor_candidate_ids.size),
            "area": float(face_areas[anchor_candidate_ids].sum()),
            "area_ratio": float(face_areas[anchor_candidate_ids].sum() / max(face_areas.sum(), 1e-12)),
            "sampled_face_count": int(sampled_anchor_face_ids.size),
        },
        "support_zone": {
            "face_count": int(support_candidate_ids.size),
            "area": float(face_areas[support_candidate_ids].sum()) if support_candidate_ids.size > 0 else 0.0,
            "area_ratio": float(face_areas[support_candidate_ids].sum() / max(face_areas.sum(), 1e-12)) if support_candidate_ids.size > 0 else 0.0,
        },
        "target_zone": {
            "candidate_face_count": int(target_candidate_ids.size),
            "candidate_face_area": target_candidate_area,
            "sampled_face_count": int(sampled_target_face_ids.size),
            "valid_pair_count": int(matched["target_face_ids"].size),
            "selected_probe_count": int(num_probes),
            "candidate_source_counts": counts_from_source_codes(target_candidate_source_codes),
            "sampled_source_counts": counts_from_source_codes(sampled_target_source_codes),
            "matched_source_counts": counts_from_source_codes(matched["target_source_code"]),
            "selected_source_counts": counts_from_source_codes(selected["target_source_code"]),
        },
        "iterative_search": support_injection_summary,
        "good_zone_reentry_filter": good_zone_reentry_summary,
        "target_subject_core": {
            "score": stats_from_array(selected["target_subject_core"]),
            "visible_count_sampled": stats_from_array(target_subject_visible_count.astype(np.float32)),
        },
        "translation_distance": stats_from_array(selected["translation_distance"]),
        "translation_scale": stats_from_array(selected["translation_scale"]),
        "normal_dot": stats_from_array(selected["normal_dot"]),
        "anchor_score": stats_from_array(selected["anchor_scores"]),
        "anchor_score_raw": stats_from_array(anchor_score_raw_sampled.astype(np.float32)),
        "target_score": stats_from_array(selected["target_scores"]),
        "target_score_raw": stats_from_array(target_score_raw_sampled.astype(np.float32)),
        "pair_priority_base": stats_from_array(selected["pair_priority_base"]),
        "pair_priority": stats_from_array(selected["pair_priority"]),
        "pair_priority_weighted": stats_from_array(selected["pair_priority_weighted"]),
        "parent_opacity": stats_from_array(parent_opacity_cpu_selected),
        "probe_opacity_init": stats_from_array(probe_opacity_cpu),
        "paths": {
            "preview_point_cloud": preview_point_cloud,
            "preview_tags": preview_tags,
            "selected_anchor_faces": selected_anchor_faces_path,
            "support_candidate_faces": support_candidate_faces_path,
            "selected_target_faces": selected_target_faces_path,
            "selected_frontier_target_faces": selected_frontier_target_faces_path,
            "selected_interior_target_faces": selected_interior_target_faces_path,
            "interior_target_candidate_faces": interior_target_candidate_faces_path,
            "promoted_support_faces": promoted_support_faces_path,
            "probe_points_init": probe_points_path,
            "probe_points_reentry_rejected": rejected_reentry_points_path,
        },
    }
    if foreground_summary is not None:
        summary["foreground_mask"] = foreground_summary
        summary["paths"]["foreground_anchor_pool_faces"] = foreground_anchor_pool_faces_path
        summary["paths"]["foreground_target_pool_faces"] = foreground_target_pool_faces_path
    else:
        summary["foreground_mask"] = {"enabled": False}
    summary["interior_targeting"] = interior_summary or {"enabled": False}
    if target_subject_components is not None:
        summary["target_subject_core"]["proxy_mode"] = "gs_structure"
        summary["target_subject_core"]["density_raw"] = stats_from_array(target_subject_components["density_raw"])
        summary["target_subject_core"]["density_norm"] = stats_from_array(target_subject_components["density_norm"])
        summary["target_subject_core"]["small_scale"] = stats_from_array(target_subject_components["small_scale"])
        summary["target_subject_core"]["anisotropy"] = stats_from_array(target_subject_components["anisotropy"])
        summary["target_subject_core"]["nonplanar"] = stats_from_array(target_subject_components["nonplanar_norm"])
    else:
        summary["target_subject_core"]["proxy_mode"] = "image_center"
    if global_search_selection is not None:
        summary["global_search"] = {
            "shell_edges_scale_mults": [float(x) for x in global_search_config["shell_edges_scale_mults"].tolist()],
            "shell_stride_scale_mults": [float(x) for x in global_search_config["shell_stride_scale_mults"].tolist()],
            "shell_priority_weights": [float(x) for x in global_search_config["shell_priority_weights"].tolist()],
            "shell_stride": [float(x) for x in global_search_config["shell_stride"].tolist()],
            "total_raw_budget": float(global_search_selection["total_raw_budget"]),
            "total_allocated_budget": int(global_search_selection["total_allocated_budget"]),
            "shells": global_search_selection["shell_stats"],
        }
        if "source_budgeting" in global_search_selection:
            summary["global_search"]["source_budgeting"] = global_search_selection["source_budgeting"]
    summary_dir = os.path.dirname(summary_out)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    Path(summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Injected {num_probes} extension probes.")
    print(f"Checkpoint saved to: {checkpoint_out}")
    print(f"Summary saved to: {summary_out}")
    print(f"Manifest saved to: {manifest_out}")


if __name__ == "__main__":
    main()
