import json
import os
import shutil
from argparse import ArgumentParser, Namespace
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
from inject_surface_extension_probes import compute_gs_structure_scores, save_point_cloud
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state


def build_appearance_embedding(mesh_args, num_views: int):
    if mesh_args.use_decoupled_appearance:
        return AppearanceEmbedding(num_views=num_views)
    if mesh_args.use_pgsr_appearance:
        return PGSREmbedding(num_views=num_views)
    return None


def write_clean_cfg_args(output_model_path: Path, dataset) -> None:
    payload = Namespace(
        sh_degree=int(dataset.sh_degree),
        source_path=str(Path(dataset.source_path).resolve()),
        model_path=str(Path(output_model_path).resolve()),
        images=str(dataset.images),
        resolution=-1,
        white_background=bool(dataset.white_background),
        data_device="cuda",
        eval=bool(dataset.eval),
        alpha_mask=bool(getattr(dataset, "alpha_mask", False)),
        init_type="sfm",
        kernel_size=float(getattr(dataset, "kernel_size", 0.1) or 0.1),
        ray_jitter=bool(getattr(dataset, "ray_jitter", False)),
        resample_gt_image=bool(getattr(dataset, "resample_gt_image", False)),
        load_allres=bool(getattr(dataset, "load_allres", False)),
        sample_more_highres=bool(getattr(dataset, "sample_more_highres", False)),
        vanilla_mip_mode=bool(getattr(dataset, "vanilla_mip_mode", False)),
    )
    with open(output_model_path / "cfg_args", "w", encoding="utf-8") as f:
        f.write(repr(payload))


def load_checkpoint_any(checkpoint_path: str):
    raw = torch.load(checkpoint_path)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 3:
            model_params, checkpoint_iteration, appearance_state = raw
        elif len(raw) == 2:
            model_params, checkpoint_iteration = raw
            appearance_state = (None, None)
        else:
            raise ValueError(f"Unsupported checkpoint tuple length {len(raw)} for {checkpoint_path}")
        return model_params, int(checkpoint_iteration), appearance_state
    raise TypeError(f"Unsupported checkpoint payload type {type(raw)!r} for {checkpoint_path}")


def resolve_start_checkpoint(model_path: str, start_checkpoint: Optional[str], iteration: int) -> str:
    if start_checkpoint:
        return start_checkpoint
    if iteration < 0:
        raise ValueError("iteration must be explicit when start_checkpoint is not provided.")
    return os.path.join(model_path, f"chkpnt{iteration}.pth")


def default_output_paths(
    dataset,
    checkpoint_iteration: int,
    output_checkpoint: Optional[str],
    output_summary: Optional[str],
    output_preview_dir: Optional[str],
):
    root = Path(dataset.model_path)
    checkpoint_path = Path(output_checkpoint) if output_checkpoint else root / f"chkpnt{checkpoint_iteration}_meshrelax.pth"
    summary_path = Path(output_summary) if output_summary else checkpoint_path.with_suffix(".summary.json")
    preview_dir = Path(output_preview_dir) if output_preview_dir else checkpoint_path.with_suffix("")
    return checkpoint_path, summary_path, preview_dir


def stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def counts_by_source_tag(source_tag: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, int]:
    if mask is not None:
        source_tag = source_tag[mask]
    mapping = {
        "original": int(GaussianSourceTag.ORIGINAL),
        "prior": int(GaussianSourceTag.PRIOR_INJECTED),
        "probe": int(GaussianSourceTag.EXTENSION_PROBE),
    }
    return {name: int(np.sum(source_tag == value)) for name, value in mapping.items()}


def compute_camera_distance(points_xyz: np.ndarray, camera_centers: np.ndarray) -> Dict[str, np.ndarray | float]:
    if points_xyz.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"nearest_distance": empty, "distance_ref": 1.0}
    if camera_centers.shape[0] == 0:
        ones = np.ones((points_xyz.shape[0],), dtype=np.float32)
        return {"nearest_distance": ones, "distance_ref": 1.0}
    tree = cKDTree(camera_centers.astype(np.float32, copy=False))
    nearest_distance, _ = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float32)
    distance_ref = max(float(np.median(nearest_distance)), 1e-6)
    return {"nearest_distance": nearest_distance, "distance_ref": distance_ref}


def compute_mesh_surface_relation(
    points_xyz: np.ndarray,
    gaussian_scaling: np.ndarray,
    mesh_obj: trimesh.Trimesh,
    sample_count: int,
    band_scale_mult: float,
    surface_points: Optional[np.ndarray] = None,
    surface_normals: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | float]:
    if points_xyz.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "surface_distance": empty,
            "surface_normal_offset": empty,
            "surface_tangent_offset": empty,
            "surface_score": empty,
            "surface_band": empty,
            "sample_count": 0,
        }

    if surface_points is None or surface_normals is None:
        sampled_points, face_ids = trimesh.sample.sample_surface(mesh_obj, max(int(sample_count), 1))
        surface_points = sampled_points.astype(np.float32, copy=False)
        face_ids = np.asarray(face_ids, dtype=np.int64)
        surface_normals = mesh_obj.face_normals[face_ids].astype(np.float32, copy=False)
    else:
        surface_points = surface_points.astype(np.float32, copy=False)
        surface_normals = surface_normals.astype(np.float32, copy=False)

    tree = cKDTree(surface_points)
    nearest_distance, nearest_ids = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float32)
    nearest_ids = np.asarray(nearest_ids, dtype=np.int64)

    nearest_points = surface_points[nearest_ids]
    nearest_normals = surface_normals[nearest_ids]
    delta = points_xyz.astype(np.float32, copy=False) - nearest_points
    normal_offset = np.abs(np.sum(delta * nearest_normals, axis=1)).astype(np.float32, copy=False)
    tangent_sq = np.clip(np.square(nearest_distance) - np.square(normal_offset), 0.0, None)
    tangent_offset = np.sqrt(tangent_sq).astype(np.float32, copy=False)

    mean_scale = np.mean(gaussian_scaling, axis=1).astype(np.float32, copy=False)
    surface_band = np.clip(mean_scale * float(band_scale_mult), 1e-6, None).astype(np.float32, copy=False)
    surface_score = np.exp(-normal_offset / surface_band).astype(np.float32, copy=False)

    return {
        "surface_distance": nearest_distance,
        "surface_normal_offset": normal_offset,
        "surface_tangent_offset": tangent_offset,
        "surface_score": surface_score,
        "surface_band": surface_band,
        "sample_count": int(sample_count),
        "nearest_surface_point": nearest_points.astype(np.float32, copy=False),
        "nearest_surface_normal": nearest_normals.astype(np.float32, copy=False),
    }


def select_spatially_balanced_ids(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_xyz: np.ndarray,
    voxel_size: float,
    max_count: int,
    per_voxel_cap: int,
) -> np.ndarray:
    if candidate_ids.size == 0 or max_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if voxel_size <= 0.0 or per_voxel_cap <= 0:
        order = np.argsort(-candidate_scores)
        return candidate_ids[order[:max_count]].astype(np.int64, copy=False)

    voxel_keys = np.floor(candidate_xyz / voxel_size).astype(np.int64)
    groups: Dict[Tuple[int, int, int], List[int]] = {}
    for local_idx, key in enumerate(map(tuple, voxel_keys.tolist())):
        groups.setdefault(key, []).append(local_idx)

    ordered_groups: List[np.ndarray] = []
    group_front_score: List[float] = []
    for local_indices in groups.values():
        local_indices_arr = np.asarray(local_indices, dtype=np.int64)
        order = np.argsort(-candidate_scores[local_indices_arr])
        ranked = local_indices_arr[order[:per_voxel_cap]]
        if ranked.size == 0:
            continue
        ordered_groups.append(ranked)
        group_front_score.append(float(candidate_scores[ranked[0]]))

    if not ordered_groups:
        return np.empty((0,), dtype=np.int64)

    group_order = np.argsort(-np.asarray(group_front_score, dtype=np.float32))
    pointers = np.zeros((len(ordered_groups),), dtype=np.int64)
    selected_local: List[int] = []

    while len(selected_local) < int(max_count):
        progressed = False
        for group_idx in group_order.tolist():
            ptr = int(pointers[group_idx])
            group = ordered_groups[group_idx]
            if ptr >= group.size:
                continue
            selected_local.append(int(group[ptr]))
            pointers[group_idx] = ptr + 1
            progressed = True
            if len(selected_local) >= int(max_count):
                break
        if not progressed:
            break

    return candidate_ids[np.asarray(selected_local, dtype=np.int64)].astype(np.int64, copy=False)


def match_projected_surface_samples(
    cam,
    projected_xy: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    depth_min: float,
    query_radius_px: float,
    query_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_projected, surface_valid = project_points_camera(cam, surface_points, depth_min=depth_min, margin=0)
    if not np.any(surface_valid):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    valid_surface_ids = np.flatnonzero(surface_valid)
    surface_xy = surface_projected[valid_surface_ids, :2]
    surface_z = surface_projected[valid_surface_ids, 2]
    tree = cKDTree(surface_xy.astype(np.float32, copy=False))

    k = max(1, min(int(query_k), surface_xy.shape[0]))
    dists, nn_ids = tree.query(
        projected_xy.astype(np.float32, copy=False),
        k=k,
        distance_upper_bound=max(float(query_radius_px), 1e-6),
    )
    if k == 1:
        dists = dists[:, None]
        nn_ids = nn_ids[:, None]

    matched_rows: List[int] = []
    hit_points: List[np.ndarray] = []
    hit_normals: List[np.ndarray] = []

    for row in range(projected_xy.shape[0]):
        valid_neighbor_mask = np.isfinite(dists[row]) & (nn_ids[row] < surface_xy.shape[0])
        if not np.any(valid_neighbor_mask):
            continue
        candidates = nn_ids[row][valid_neighbor_mask].astype(np.int64, copy=False)
        candidate_depth = surface_z[candidates]
        best_local = int(candidates[np.argmin(candidate_depth)])
        world_point = surface_points[valid_surface_ids[best_local]]
        world_normal = surface_normals[valid_surface_ids[best_local]]
        matched_rows.append(row)
        hit_points.append(world_point.astype(np.float32, copy=False))
        hit_normals.append(world_normal.astype(np.float32, copy=False))

    if not matched_rows:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    return (
        np.asarray(matched_rows, dtype=np.int64),
        np.asarray(hit_points, dtype=np.float32),
        np.asarray(hit_normals, dtype=np.float32),
    )


def project_points_camera(cam, points_xyz: np.ndarray, depth_min: float, margin: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(cam.R, dtype=np.float32)
    T = np.asarray(cam.T, dtype=np.float32)
    xyz_cam = points_xyz @ R + T[None, :]
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / np.clip(z, 1e-6, None) * float(cam.focal_x) + float(cam.image_width) / 2.0
    y = xyz_cam[:, 1] / np.clip(z, 1e-6, None) * float(cam.focal_y) + float(cam.image_height) / 2.0
    valid = z > float(depth_min)
    valid &= x >= float(margin)
    valid &= x < float(cam.image_width - margin)
    valid &= y >= float(margin)
    valid &= y < float(cam.image_height - margin)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False), valid


def ray_dirs_to_projected_pixels(cam, projected_xy: np.ndarray) -> np.ndarray:
    intrins = np.asarray(
        [
            [float(cam.focal_x), 0.0, float(cam.image_width) / 2.0],
            [0.0, float(cam.focal_y), float(cam.image_height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    c2w = torch.inverse(cam.world_view_transform.T).detach().cpu().numpy().astype(np.float32, copy=False)
    points = np.empty((projected_xy.shape[0], 3), dtype=np.float32)
    points[:, :2] = projected_xy
    points[:, 2] = 1.0
    dirs_world = points @ np.linalg.inv(intrins).T @ c2w[:3, :3].T
    dirs_world = dirs_world / np.clip(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-6, None)
    return dirs_world.astype(np.float32, copy=False)


def first_ray_hits(
    intersector,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
    )
    if len(index_ray) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    locations = np.asarray(locations, dtype=np.float32)
    index_ray = np.asarray(index_ray, dtype=np.int64)
    index_tri = np.asarray(index_tri, dtype=np.int64)
    ray_hit_distance = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    best_hit_per_ray: Dict[int, Tuple[float, np.ndarray, int]] = {}
    for hit_idx, ray_idx in enumerate(index_ray.tolist()):
        dist = float(ray_hit_distance[hit_idx])
        existing = best_hit_per_ray.get(ray_idx)
        if existing is None or dist < existing[0]:
            best_hit_per_ray[ray_idx] = (dist, locations[hit_idx], int(index_tri[hit_idx]))

    ray_ids = sorted(best_hit_per_ray.keys())
    hit_points = np.asarray([best_hit_per_ray[idx][1] for idx in ray_ids], dtype=np.float32)
    hit_tris = np.asarray([best_hit_per_ray[idx][2] for idx in ray_ids], dtype=np.int64)
    return ray_ids, hit_points, hit_tris


def robust_hit_consensus(
    hit_points: np.ndarray,
    consensus_radius: float,
    min_inliers: int,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if hit_points.shape[0] == 0:
        return None, np.zeros((0,), dtype=bool)
    center = np.median(hit_points, axis=0).astype(np.float32, copy=False)
    d = np.linalg.norm(hit_points - center[None, :], axis=1)
    inlier_mask = d <= max(float(consensus_radius), 1e-6)
    if int(np.sum(inlier_mask)) < int(min_inliers):
        return None, inlier_mask
    target = np.mean(hit_points[inlier_mask], axis=0).astype(np.float32, copy=False)
    return target, inlier_mask


def main():
    parser = ArgumentParser(description="Soft-relax suspicious near-field gaussians toward trusted mesh surface hits.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--mesh_surface_sample_count", type=int, default=300000)
    parser.add_argument("--mesh_surface_band_scale_mult", type=float, default=2.0)
    parser.add_argument("--neighbor_k", type=int, default=24)
    parser.add_argument("--radius_scale_mult", type=float, default=12.0)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--focus_near_field", action="store_true")
    parser.add_argument("--near_field_distance_quantile", type=float, default=0.4)
    parser.add_argument("--suspicious_surface_power", type=float, default=1.0)
    parser.add_argument("--suspicious_structure_power", type=float, default=0.5)
    parser.add_argument("--suspicious_opacity_power", type=float, default=0.5)
    parser.add_argument("--min_normal_ratio", type=float, default=1.0)
    parser.add_argument("--suspicious_score_quantile", type=float, default=0.9)
    parser.add_argument("--max_suspicious_count", type=int, default=5000)
    parser.add_argument("--suspicious_balance_voxel_scale_mult", type=float, default=10.0)
    parser.add_argument("--suspicious_balance_per_voxel", type=int, default=24)
    parser.add_argument("--disable_spatial_balance", action="store_true")
    parser.add_argument("--camera_stride", type=int, default=4)
    parser.add_argument("--depth_min", type=float, default=0.2)
    parser.add_argument("--min_hit_views", type=int, default=2)
    parser.add_argument("--min_consensus_hits", type=int, default=2)
    parser.add_argument("--consensus_radius_scale_mult", type=float, default=4.0)
    parser.add_argument("--consensus_radius_abs", type=float, default=0.02)
    parser.add_argument("--enable_anchor_fallback", action="store_true")
    parser.add_argument("--anchor_fallback_min_hit_views", type=int, default=3)
    parser.add_argument("--anchor_fallback_radius_scale_mult", type=float, default=6.0)
    parser.add_argument("--anchor_fallback_radius_abs", type=float, default=0.03)
    parser.add_argument("--anchor_fallback_blend", type=float, default=0.6)
    parser.add_argument("--surface_query_radius_px", type=float, default=6.0)
    parser.add_argument("--surface_query_k", type=int, default=8)
    parser.add_argument("--relocation_strength", type=float, default=1.5)
    parser.add_argument("--max_relocation_alpha", type=float, default=1.0)
    parser.add_argument("--scaling_shrink_strength", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--output_summary", type=str, default=None)
    parser.add_argument("--output_preview_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for mesh relaxation.")
        args.data_device = "cpu"

    safe_state(args.quiet)

    dataset = model.extract(args)
    dataset.init_type = "sfm"
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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_params, checkpoint_iteration, appearance_state = load_checkpoint_any(checkpoint_path)
    if appearance_embedding is not None and appearance_state[0] is not None:
        appearance_embedding.restore(*appearance_state)
    gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)

    mesh_obj = trimesh.load_mesh(args.mesh_path, process=False)
    if not isinstance(mesh_obj, trimesh.Trimesh):
        if hasattr(mesh_obj, "dump"):
            dump = mesh_obj.dump(concatenate=True)
            if isinstance(dump, trimesh.Trimesh):
                mesh_obj = dump
        if not isinstance(mesh_obj, trimesh.Trimesh):
            raise ValueError(f"Failed to load a triangle mesh from {args.mesh_path}")

    surface_points, surface_face_ids = trimesh.sample.sample_surface(mesh_obj, max(int(args.mesh_surface_sample_count), 1))
    surface_points = np.asarray(surface_points, dtype=np.float32)
    surface_face_ids = np.asarray(surface_face_ids, dtype=np.int64)
    surface_normals = np.asarray(mesh_obj.face_normals[surface_face_ids], dtype=np.float32)

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    scaling = gaussians.get_scaling_with_3D_filter.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussians.get_opacity_with_3D_filter.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    source_tag = gaussians._source_tag.detach().cpu().numpy().astype(np.int32, copy=False)

    camera_centers = np.stack(
        [cam.camera_center.detach().cpu().numpy() for cam in train_cameras],
        axis=0,
    ).astype(np.float32, copy=False)
    distance_payload = compute_camera_distance(xyz, camera_centers)
    near_distance = distance_payload["nearest_distance"]
    near_field_threshold = float(np.quantile(near_distance, float(np.clip(args.near_field_distance_quantile, 0.0, 1.0))))
    near_field_mask = near_distance <= near_field_threshold if args.focus_near_field else np.ones_like(near_distance, dtype=bool)

    structure_score, _, structure_components = compute_gs_structure_scores(
        points_xyz=xyz,
        gaussian_xyz=xyz,
        gaussian_opacity=opacity,
        gaussian_scaling=scaling,
        neighbor_k=int(args.neighbor_k),
        radius=max(float(args.radius_scale_mult) * max(float(np.median(np.mean(scaling, axis=1))), 1e-6), 1e-6),
        scale_ref=max(float(np.median(np.mean(scaling, axis=1))), 1e-6),
        batch_size=int(args.batch_size),
    )
    opacity_ref = max(float(np.percentile(opacity, 95)), 1e-6)
    opacity_norm = np.clip(opacity / opacity_ref, 0.0, 1.0).astype(np.float32, copy=False)

    surface_payload = compute_mesh_surface_relation(
        points_xyz=xyz,
        gaussian_scaling=scaling,
        mesh_obj=mesh_obj,
        sample_count=int(args.mesh_surface_sample_count),
        band_scale_mult=float(args.mesh_surface_band_scale_mult),
        surface_points=surface_points,
        surface_normals=surface_normals,
    )
    normal_ratio = surface_payload["surface_normal_offset"] / np.clip(surface_payload["surface_band"], 1e-6, None)
    surface_mismatch = 1.0 - np.clip(surface_payload["surface_score"], 0.0, 1.0)

    suspicious_score = (
        np.power(np.clip(surface_mismatch, 0.0, 1.0), float(args.suspicious_surface_power))
        * np.power(np.clip(structure_score, 0.0, 1.0), float(args.suspicious_structure_power))
        * np.power(np.clip(opacity_norm, 0.0, 1.0), float(args.suspicious_opacity_power))
        * np.clip(normal_ratio / max(float(args.min_normal_ratio), 1e-6), 0.0, 1.0)
    ).astype(np.float32, copy=False)

    suspicious_pool_mask = near_field_mask & (normal_ratio >= float(args.min_normal_ratio))
    suspicious_pool_scores = suspicious_score[suspicious_pool_mask]
    if suspicious_pool_scores.size == 0:
        raise RuntimeError("No suspicious gaussians found in the near-field pool.")
    suspicious_threshold = float(np.quantile(
        suspicious_pool_scores,
        float(np.clip(args.suspicious_score_quantile, 0.0, 1.0)),
    ))
    suspicious_mask = suspicious_pool_mask & (suspicious_score >= suspicious_threshold)

    suspicious_ids = np.flatnonzero(suspicious_mask)
    if args.max_suspicious_count > 0 and suspicious_ids.size > int(args.max_suspicious_count):
        if args.disable_spatial_balance:
            order = np.argsort(-suspicious_score[suspicious_ids])
            suspicious_ids = suspicious_ids[order[: int(args.max_suspicious_count)]]
        else:
            median_scale = max(float(np.median(np.mean(scaling[suspicious_ids], axis=1))), 1e-6)
            voxel_size = float(args.suspicious_balance_voxel_scale_mult) * median_scale
            suspicious_ids = select_spatially_balanced_ids(
                candidate_ids=suspicious_ids,
                candidate_scores=suspicious_score[suspicious_ids],
                candidate_xyz=xyz[suspicious_ids],
                voxel_size=voxel_size,
                max_count=int(args.max_suspicious_count),
                per_voxel_cap=int(args.suspicious_balance_per_voxel),
            )
        suspicious_mask = np.zeros_like(suspicious_mask, dtype=bool)
        suspicious_mask[suspicious_ids] = True

    suspicious_xyz = xyz[suspicious_ids]
    suspicious_scale_mean = np.mean(scaling[suspicious_ids], axis=1).astype(np.float32, copy=False)
    suspicious_score_sel = suspicious_score[suspicious_ids]

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_obj)
    face_normals = np.asarray(mesh_obj.face_normals, dtype=np.float32)
    match_mode = "ray_intersector"
    raycast_error_count = 0
    raycast_error_message: Optional[str] = None

    hit_lists: List[List[np.ndarray]] = [[] for _ in range(suspicious_ids.shape[0])]
    hit_normal_lists: List[List[np.ndarray]] = [[] for _ in range(suspicious_ids.shape[0])]
    visible_view_count = np.zeros((suspicious_ids.shape[0],), dtype=np.int32)
    hit_view_count = np.zeros((suspicious_ids.shape[0],), dtype=np.int32)

    camera_indices = list(range(0, len(train_cameras), max(int(args.camera_stride), 1)))
    if not camera_indices:
        camera_indices = list(range(len(train_cameras)))

    for cam_idx in camera_indices:
        cam = train_cameras[cam_idx]
        projected, valid = project_points_camera(cam, suspicious_xyz, depth_min=float(args.depth_min), margin=0)
        if not np.any(valid):
            continue
        valid_ids = np.flatnonzero(valid)
        visible_view_count[valid_ids] += 1

        projected_xy = projected[valid_ids, :2]
        ray_origins = np.repeat(cam.camera_center.detach().cpu().numpy()[None, :], valid_ids.shape[0], axis=0).astype(np.float32, copy=False)
        ray_directions = ray_dirs_to_projected_pixels(cam, projected_xy)
        try:
            ray_ids, hit_points, hit_tris = first_ray_hits(intersector, ray_origins, ray_directions)
            ray_ids_arr = np.asarray(ray_ids, dtype=np.int64)
            hit_normals = face_normals[hit_tris]
            if ray_ids_arr.size == 0:
                raise RuntimeError("no_ray_hits")
            view_dirs = ray_origins[ray_ids_arr] - hit_points
            front_facing = np.einsum("ij,ij->i", hit_normals, view_dirs) > 0.0
            if not np.any(front_facing):
                raise RuntimeError("no_front_facing_hits")

            ray_ids_arr = ray_ids_arr[front_facing]
            hit_points = hit_points[front_facing]
            hit_normals = hit_normals[front_facing]
            mapped_ids = valid_ids[ray_ids_arr]
        except Exception as exc:
            raycast_error_count += 1
            if raycast_error_message is None:
                raycast_error_message = repr(exc)
            match_mode = "projected_surface_samples_fallback"
            ray_ids_arr, hit_points, hit_normals = match_projected_surface_samples(
                cam=cam,
                projected_xy=projected_xy,
                surface_points=surface_points,
                surface_normals=surface_normals,
                depth_min=float(args.depth_min),
                query_radius_px=float(args.surface_query_radius_px),
                query_k=int(args.surface_query_k),
            )
            if ray_ids_arr.size == 0:
                continue
            mapped_ids = valid_ids[ray_ids_arr]

        hit_view_count[mapped_ids] += 1
        for local_row, point, normal in zip(mapped_ids.tolist(), hit_points.tolist(), hit_normals.tolist()):
            hit_lists[local_row].append(np.asarray(point, dtype=np.float32))
            hit_normal_lists[local_row].append(np.asarray(normal, dtype=np.float32))

    relocated_mask_local = np.zeros((suspicious_ids.shape[0],), dtype=bool)
    target_points = np.full((suspicious_ids.shape[0], 3), np.nan, dtype=np.float32)
    relocation_alpha = np.zeros((suspicious_ids.shape[0],), dtype=np.float32)
    relocation_distance = np.zeros((suspicious_ids.shape[0],), dtype=np.float32)
    consensus_inlier_count = np.zeros((suspicious_ids.shape[0],), dtype=np.int32)
    target_normals = np.full((suspicious_ids.shape[0], 3), np.nan, dtype=np.float32)
    anchor_fallback_mask_local = np.zeros((suspicious_ids.shape[0],), dtype=bool)

    new_xyz = xyz.copy()
    new_scaling = scaling.copy()

    for row, global_id in enumerate(suspicious_ids.tolist()):
        if hit_view_count[row] < int(args.min_hit_views):
            continue
        hits = np.asarray(hit_lists[row], dtype=np.float32)
        if hits.shape[0] == 0:
            continue
        normals = np.asarray(hit_normal_lists[row], dtype=np.float32)
        consensus_radius = max(float(args.consensus_radius_abs), float(args.consensus_radius_scale_mult) * float(suspicious_scale_mean[row]))
        target, inlier_mask = robust_hit_consensus(hits, consensus_radius=consensus_radius, min_inliers=int(args.min_consensus_hits))
        if target is None and args.enable_anchor_fallback and hit_view_count[row] >= int(args.anchor_fallback_min_hit_views):
            anchor_point = surface_payload["nearest_surface_point"][global_id].astype(np.float32, copy=False)
            anchor_radius = max(
                float(args.anchor_fallback_radius_abs),
                float(args.anchor_fallback_radius_scale_mult) * float(surface_payload["surface_band"][global_id]),
            )
            anchor_dist = np.linalg.norm(hits - anchor_point[None, :], axis=1)
            anchor_inlier_mask = anchor_dist <= anchor_radius
            if int(np.sum(anchor_inlier_mask)) >= int(args.min_consensus_hits):
                anchor_mean = np.mean(hits[anchor_inlier_mask], axis=0).astype(np.float32, copy=False)
                blend = float(np.clip(args.anchor_fallback_blend, 0.0, 1.0))
                target = (blend * anchor_point + (1.0 - blend) * anchor_mean).astype(np.float32, copy=False)
                inlier_mask = anchor_inlier_mask
                anchor_fallback_mask_local[row] = True
        if target is None:
            continue
        inlier_count = int(np.sum(inlier_mask))
        consensus_inlier_count[row] = inlier_count
        target_points[row] = target
        target_normals[row] = np.mean(normals[inlier_mask], axis=0).astype(np.float32, copy=False)

        move = target - xyz[global_id]
        move_dist = float(np.linalg.norm(move))
        relocation_distance[row] = move_dist
        band = max(float(surface_payload["surface_band"][global_id]), 1e-6)
        alpha = float(args.relocation_strength) * float(suspicious_score_sel[row]) * move_dist / (move_dist + band)
        alpha = float(np.clip(alpha, 0.0, float(args.max_relocation_alpha)))
        if alpha <= 0.0:
            continue

        relocated_mask_local[row] = True
        relocation_alpha[row] = alpha
        new_xyz[global_id] = xyz[global_id] + alpha * move
        if float(args.scaling_shrink_strength) > 0.0:
            shrink = max(0.0, 1.0 - float(args.scaling_shrink_strength) * alpha)
            new_scaling[global_id] = np.clip(scaling[global_id] * shrink, 1e-6, None)

    relocated_ids = suspicious_ids[relocated_mask_local]
    relocated_mask_global = np.zeros((xyz.shape[0],), dtype=bool)
    relocated_mask_global[relocated_ids] = True

    output_checkpoint, output_summary, output_preview_dir = default_output_paths(
        dataset=dataset,
        checkpoint_iteration=checkpoint_iteration,
        output_checkpoint=args.output_checkpoint,
        output_summary=args.output_summary,
        output_preview_dir=args.output_preview_dir,
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_preview_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "suspicious_mask": torch.from_numpy(suspicious_mask.copy()),
        "relocated_mask": torch.from_numpy(relocated_mask_global.copy()),
        "suspicious_score": torch.from_numpy(suspicious_score.copy()),
        "surface_score": torch.from_numpy(surface_payload["surface_score"].copy()),
        "surface_normal_offset": torch.from_numpy(surface_payload["surface_normal_offset"].copy()),
        "surface_band": torch.from_numpy(surface_payload["surface_band"].copy()),
        "nearest_camera_distance": torch.from_numpy(near_distance.copy()),
        "relocation_alpha": torch.from_numpy(relocation_alpha.copy()),
        "relocation_distance": torch.from_numpy(relocation_distance.copy()),
        "suspicious_ids": torch.from_numpy(suspicious_ids.copy()),
        "target_points": torch.from_numpy(target_points.copy()),
        "target_normals": torch.from_numpy(target_normals.copy()),
        "visible_view_count": torch.from_numpy(visible_view_count.copy()),
        "hit_view_count": torch.from_numpy(hit_view_count.copy()),
        "consensus_inlier_count": torch.from_numpy(consensus_inlier_count.copy()),
    }
    payload_path = output_preview_dir / "mesh_relax_payload.pt"
    torch.save(payload, str(payload_path))

    suspicious_before_path = output_preview_dir / "suspicious_points_before.ply"
    suspicious_after_path = output_preview_dir / "suspicious_points_after.ply"
    target_points_path = output_preview_dir / "target_surface_points.ply"
    save_point_cloud(suspicious_xyz, str(suspicious_before_path))
    save_point_cloud(new_xyz[suspicious_ids], str(suspicious_after_path))
    valid_targets = np.isfinite(target_points[:, 0])
    save_point_cloud(target_points[valid_targets], str(target_points_path))

    if not args.dry_run:
        with torch.no_grad():
            gaussians._xyz.data = torch.tensor(new_xyz, dtype=gaussians._xyz.dtype, device=gaussians._xyz.device)
            if float(args.scaling_shrink_strength) > 0.0:
                gaussians._scaling.data = gaussians.scaling_inverse_activation(
                    torch.tensor(new_scaling, dtype=gaussians.get_scaling.dtype, device=gaussians._scaling.device)
                )
            gaussians.compute_3D_filter(train_cameras.copy(), CUDA=not pipe_args.compute_filter3D_python)

        standard_point_cloud_dir = output_checkpoint.parent / "point_cloud" / f"iteration_{checkpoint_iteration}"
        standard_point_cloud_dir.mkdir(parents=True, exist_ok=True)
        torch.save((gaussians.capture(), checkpoint_iteration, appearance_state), str(output_checkpoint))
        gaussians.save_ply(str(standard_point_cloud_dir / "point_cloud.ply"))
        gaussians.save_tracking_metadata(str(standard_point_cloud_dir / "gaussian_tags.pt"))
        gaussians.save_ply(str(output_preview_dir / "point_cloud.ply"))
        gaussians.save_tracking_metadata(str(output_preview_dir / "gaussian_tags.pt"))

        write_clean_cfg_args(output_checkpoint.parent, dataset)
        input_config_json = Path(dataset.model_path) / "config.json"
        if input_config_json.exists():
            shutil.copy2(input_config_json, output_checkpoint.parent / "config.json")
        input_cameras_json = Path(dataset.model_path) / "cameras.json"
        if input_cameras_json.exists():
            shutil.copy2(input_cameras_json, output_checkpoint.parent / "cameras.json")

    summary = {
        "mode": "mesh_first_suspicious_gaussian_relaxation",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "input_checkpoint": str(Path(checkpoint_path).resolve()),
        "output_checkpoint": None if args.dry_run else str(output_checkpoint.resolve()),
        "checkpoint_iteration": int(checkpoint_iteration),
        "dry_run": bool(args.dry_run),
        "parameters": {
            "mesh_path": str(Path(args.mesh_path).resolve()),
            "focus_near_field": bool(args.focus_near_field),
            "near_field_distance_quantile": float(args.near_field_distance_quantile),
            "near_field_distance_threshold": float(near_field_threshold),
            "neighbor_k": int(args.neighbor_k),
            "radius_scale_mult": float(args.radius_scale_mult),
            "mesh_surface_sample_count": int(args.mesh_surface_sample_count),
            "mesh_surface_band_scale_mult": float(args.mesh_surface_band_scale_mult),
            "camera_stride": int(args.camera_stride),
            "depth_min": float(args.depth_min),
            "min_normal_ratio": float(args.min_normal_ratio),
            "suspicious_score_quantile": float(args.suspicious_score_quantile),
            "max_suspicious_count": int(args.max_suspicious_count),
            "suspicious_balance_voxel_scale_mult": float(args.suspicious_balance_voxel_scale_mult),
            "suspicious_balance_per_voxel": int(args.suspicious_balance_per_voxel),
            "disable_spatial_balance": bool(args.disable_spatial_balance),
            "min_hit_views": int(args.min_hit_views),
            "min_consensus_hits": int(args.min_consensus_hits),
            "consensus_radius_scale_mult": float(args.consensus_radius_scale_mult),
            "consensus_radius_abs": float(args.consensus_radius_abs),
            "enable_anchor_fallback": bool(args.enable_anchor_fallback),
            "anchor_fallback_min_hit_views": int(args.anchor_fallback_min_hit_views),
            "anchor_fallback_radius_scale_mult": float(args.anchor_fallback_radius_scale_mult),
            "anchor_fallback_radius_abs": float(args.anchor_fallback_radius_abs),
            "anchor_fallback_blend": float(args.anchor_fallback_blend),
            "surface_query_radius_px": float(args.surface_query_radius_px),
            "surface_query_k": int(args.surface_query_k),
            "relocation_strength": float(args.relocation_strength),
            "max_relocation_alpha": float(args.max_relocation_alpha),
            "scaling_shrink_strength": float(args.scaling_shrink_strength),
            "surface_match_mode": match_mode,
        },
        "counts": {
            "total_gaussians": int(xyz.shape[0]),
            "source_counts": counts_by_source_tag(source_tag),
            "near_field_count": int(np.sum(near_field_mask)),
            "suspicious_pool_count": int(np.sum(suspicious_pool_mask)),
            "suspicious_count": int(suspicious_ids.shape[0]),
            "relocated_count": int(relocated_ids.shape[0]),
            "anchor_fallback_relocated_count": int(np.sum(anchor_fallback_mask_local)),
            "skipped_no_hit_views": int(np.sum((hit_view_count < int(args.min_hit_views)))),
            "skipped_no_consensus": int(np.sum((hit_view_count >= int(args.min_hit_views)) & (~relocated_mask_local))),
            "raycast_error_count": int(raycast_error_count),
        },
        "score_stats": {
            "suspicious_score_all": stats_from_array(suspicious_score),
            "suspicious_score_selected": stats_from_array(suspicious_score_sel),
            "surface_score_selected": stats_from_array(surface_payload["surface_score"][suspicious_ids]),
            "surface_normal_offset_selected": stats_from_array(surface_payload["surface_normal_offset"][suspicious_ids]),
            "normal_ratio_selected": stats_from_array(normal_ratio[suspicious_ids]),
            "visible_view_count_selected": stats_from_array(visible_view_count.astype(np.float32)),
            "hit_view_count_selected": stats_from_array(hit_view_count.astype(np.float32)),
            "consensus_inlier_count": stats_from_array(consensus_inlier_count.astype(np.float32)),
            "relocation_alpha": stats_from_array(relocation_alpha[relocated_mask_local]),
            "relocation_distance": stats_from_array(relocation_distance[relocated_mask_local]),
        },
        "paths": {
            "payload": str(payload_path.resolve()),
            "suspicious_points_before": str(suspicious_before_path.resolve()),
            "suspicious_points_after": str(suspicious_after_path.resolve()),
            "target_surface_points": str(target_points_path.resolve()),
        },
    }
    if raycast_error_message is not None:
        summary["parameters"]["raycast_error_message"] = raycast_error_message
    if not args.dry_run:
        summary["paths"]["standard_point_cloud"] = str((output_checkpoint.parent / "point_cloud" / f"iteration_{checkpoint_iteration}" / "point_cloud.ply").resolve())
        summary["paths"]["standard_tags"] = str((output_checkpoint.parent / "point_cloud" / f"iteration_{checkpoint_iteration}" / "gaussian_tags.pt").resolve())
        summary["paths"]["preview_point_cloud"] = str((output_preview_dir / "point_cloud.ply").resolve())
        summary["paths"]["preview_tags"] = str((output_preview_dir / "gaussian_tags.pt").resolve())

    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run complete; no checkpoint was written.")
    else:
        print(f"Relaxed checkpoint saved to: {output_checkpoint}")
    print(f"Summary saved to: {output_summary}")


if __name__ == "__main__":
    main()
