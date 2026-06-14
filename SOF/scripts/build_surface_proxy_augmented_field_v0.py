#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from scripts.export_gaussian_mask_subset_v0 import (
    _build_dataset_args,
    _clone_subset_gaussians,
    _copy_render_config,
    _iter_views,
    _load_mask_payload,
    _resolve_iteration,
    _save_rgb,
    _select_uniform,
)
from utils.general_utils import build_rotation, inverse_sigmoid


def _payload_tensor(payload: Dict[str, object], key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"Missing payload key: {key}")
    value = payload[key]
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 1 else tensor.reshape(-1)


def _normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=eps)


def _stable_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype).expand_as(normals)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=normals.device, dtype=normals.dtype).expand_as(normals)
    use_x = torch.abs(normals[:, 0]) < 0.9
    ref = torch.where(use_x[:, None], ref_x, ref_y)
    tangent_u = _normalize(torch.cross(normals, ref, dim=1))
    tangent_v = _normalize(torch.cross(normals, tangent_u, dim=1))
    tangent_u = _normalize(torch.cross(tangent_v, normals, dim=1))
    return tangent_u, tangent_v


def _quaternion_from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
                    1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                    1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                    1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
                ],
                dim=-1,
            ),
            min=1e-8,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[:, 0] ** 2, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], dim=-1),
            torch.stack([m[:, 2, 1] - m[:, 1, 2], q_abs[:, 1] ** 2, m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0]], dim=-1),
            torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] + m[:, 0, 1], q_abs[:, 2] ** 2, m[:, 2, 1] + m[:, 1, 2]], dim=-1),
            torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 2, 1] + m[:, 1, 2], q_abs[:, 3] ** 2], dim=-1),
        ],
        dim=1,
    )
    quat_candidates = quat_by_rijk / (2.0 * torch.clamp(q_abs[:, :, None], min=1e-8))
    out = quat_candidates[torch.arange(matrix.shape[0], device=matrix.device), torch.argmax(q_abs, dim=-1)]
    return _normalize(out)


def _select_donor_ids(
    *,
    payload: Dict[str, object],
    donor_mask: torch.Tensor,
    max_donors: int,
    model_opacity: torch.Tensor | None = None,
) -> np.ndarray:
    donor_mask = donor_mask.to(dtype=torch.bool, device="cpu").reshape(-1)
    donor_ids = torch.nonzero(donor_mask, as_tuple=False).reshape(-1)
    if donor_ids.numel() == 0:
        raise ValueError("Donor mask selected zero gaussians.")

    if {"opacity", "suggested_attach_weight", "mesh_coverage_weight", "center_d_norm"}.issubset(payload.keys()):
        opacity = _payload_tensor(payload, "opacity", dtype=torch.float32)
        attach = _payload_tensor(payload, "suggested_attach_weight", dtype=torch.float32)
        coverage = _payload_tensor(payload, "mesh_coverage_weight", dtype=torch.float32)
        center_d_norm = _payload_tensor(payload, "center_d_norm", dtype=torch.float32).clamp_min(0.0)
        score = opacity * (0.10 + attach) * (0.25 + coverage) * torch.exp(-center_d_norm)
    elif model_opacity is not None:
        score = model_opacity.detach().cpu().reshape(-1).to(dtype=torch.float32)
    else:
        raise KeyError(
            "Donor scoring needs surface-state keys "
            "(opacity/suggested_attach_weight/mesh_coverage_weight/center_d_norm) "
            "or model_opacity fallback."
        )
    donor_scores = score[donor_ids]

    if int(max_donors) > 0 and int(donor_ids.numel()) > int(max_donors):
        topk = torch.topk(donor_scores, k=int(max_donors), largest=True, sorted=True).indices
        donor_ids = donor_ids[topk]
        donor_scores = donor_scores[topk]
    else:
        order = torch.argsort(donor_scores, descending=True)
        donor_ids = donor_ids[order]
        donor_scores = donor_scores[order]

    selected = donor_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    if selected.size == 0:
        raise ValueError("No donors remained after scoring.")
    return selected


def _stats(values: np.ndarray) -> Dict[str, float | int]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        return {"count": int(values.size), "finite_count": int(finite.sum())}
    return {
        "count": int(values.size),
        "finite_count": int(finite.sum()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _load_triangle_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load triangle mesh from {mesh_path}")


def _resolve_mesh_path(mask_payload_path: Path, mesh_path: str) -> Path:
    if str(mesh_path).strip():
        resolved = Path(mesh_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"mesh_path not found: {resolved}")
        return resolved

    summary_candidates = [
        mask_payload_path.with_name("gaussian_surface_state_v0_summary.json"),
        mask_payload_path.with_name("summary.json"),
    ]
    for summary_path in summary_candidates:
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidate = summary.get("mesh_path")
        if candidate:
            resolved = Path(str(candidate)).expanduser().resolve()
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(
                f"mesh_path from {summary_path} does not exist: {resolved}. "
                "Pass --mesh_path explicitly if the mesh moved."
            )
    raise FileNotFoundError(
        "Could not infer mesh_path. Pass --mesh_path or keep "
        "gaussian_surface_state_v0_summary.json next to the surface-state payload."
    )


def _project_points_camera(cam, points_xyz: np.ndarray, depth_min: float, margin: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(cam.R, dtype=np.float32)
    t = np.asarray(cam.T, dtype=np.float32)
    xyz_cam = points_xyz @ r + t[None, :]
    z = xyz_cam[:, 2]
    safe_z = np.clip(z, 1e-6, None)
    x = xyz_cam[:, 0] / safe_z * float(cam.focal_x) + float(cam.image_width) / 2.0
    y = xyz_cam[:, 1] / safe_z * float(cam.focal_y) + float(cam.image_height) / 2.0
    valid = z > float(depth_min)
    valid &= x >= float(margin)
    valid &= x < float(cam.image_width - margin)
    valid &= y >= float(margin)
    valid &= y < float(cam.image_height - margin)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False), valid


def _build_open3d_raycast_scene(mesh_obj: trimesh.Trimesh):
    import open3d as o3d

    vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
    faces = np.asarray(mesh_obj.faces, dtype=np.int32)
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(vertices)
    legacy.triangles = o3d.utility.Vector3iVector(faces)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def _cast_rays_open3d(
    ray_scene,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    ray_chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d

    rays = np.concatenate(
        [
            ray_origins.astype(np.float32, copy=False),
            ray_directions.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    ray_count = int(rays.shape[0])
    hit_mask = np.zeros((ray_count,), dtype=bool)
    hit_points = np.zeros((ray_count, 3), dtype=np.float32)
    hit_tris = np.full((ray_count,), -1, dtype=np.int64)
    hit_dist = np.full((ray_count,), np.inf, dtype=np.float32)
    invalid_primitive = np.iinfo(np.uint32).max
    chunk = max(int(ray_chunk), 1)
    for begin in range(0, ray_count, chunk):
        end = min(begin + chunk, ray_count)
        result = ray_scene.cast_rays(o3d.core.Tensor(rays[begin:end], dtype=o3d.core.Dtype.Float32))
        t_hit = result["t_hit"].numpy().astype(np.float32, copy=False)
        primitive = result["primitive_ids"].numpy().astype(np.uint64, copy=False)
        local_hit = np.isfinite(t_hit) & (primitive != invalid_primitive)
        if not np.any(local_hit):
            continue
        local_ids = np.flatnonzero(local_hit)
        global_ids = begin + local_ids
        hit_mask[global_ids] = True
        hit_dist[global_ids] = t_hit[local_ids]
        hit_tris[global_ids] = primitive[local_ids].astype(np.int64, copy=False)
        hit_points[global_ids] = (
            ray_origins[global_ids] + ray_directions[global_ids] * t_hit[local_ids, None]
        ).astype(np.float32, copy=False)
    return hit_mask, hit_points, hit_tris, hit_dist


def _cast_rays_trimesh(
    intersector,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
    )
    ray_count = int(ray_origins.shape[0])
    hit_mask = np.zeros((ray_count,), dtype=bool)
    hit_points = np.zeros((ray_count, 3), dtype=np.float32)
    hit_tris = np.full((ray_count,), -1, dtype=np.int64)
    hit_dist = np.full((ray_count,), np.inf, dtype=np.float32)
    if len(index_ray) == 0:
        return hit_mask, hit_points, hit_tris, hit_dist

    locations = np.asarray(locations, dtype=np.float32)
    index_ray = np.asarray(index_ray, dtype=np.int64)
    index_tri = np.asarray(index_tri, dtype=np.int64)
    distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
    order = np.argsort(distances)
    for hit_idx in order.tolist():
        ray_id = int(index_ray[hit_idx])
        if hit_mask[ray_id]:
            continue
        hit_mask[ray_id] = True
        hit_points[ray_id] = locations[hit_idx]
        hit_tris[ray_id] = int(index_tri[hit_idx])
        hit_dist[ray_id] = float(distances[hit_idx])
    return hit_mask, hit_points, hit_tris, hit_dist


def _orient_normals_toward_donors(
    normals: np.ndarray,
    hit_points: np.ndarray,
    donor_xyz: np.ndarray,
) -> np.ndarray:
    oriented = np.asarray(normals, dtype=np.float32).copy()
    to_donor = np.asarray(donor_xyz, dtype=np.float32) - np.asarray(hit_points, dtype=np.float32)
    flip = np.einsum("ij,ij->i", oriented, to_donor) < 0.0
    oriented[flip] *= -1.0
    return oriented


def _build_payload_anchor_specs(
    *,
    payload: Dict[str, object],
    donor_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor | Dict[str, object]]:
    donor_ids_t = torch.from_numpy(donor_ids).to(device=device, dtype=torch.long)
    anchor_xyz = torch.as_tensor(payload["anchor_xyz"], device=device, dtype=torch.float32)[donor_ids_t]
    anchor_normal = _normalize(torch.as_tensor(payload["anchor_normal"], device=device, dtype=torch.float32)[donor_ids_t])
    return {
        "anchor_xyz": anchor_xyz,
        "anchor_normal": anchor_normal,
        "cloned_donor_ids": donor_ids,
        "stats": {
            "mode": "anchor",
            "selected_proxy_gaussians": int(donor_ids.shape[0]),
            "ray_hit_count": 0,
            "fallback_anchor_count": int(donor_ids.shape[0]),
        },
    }


def _build_ray_surface_anchor_specs(
    *,
    mesh_path: Path,
    scene: Scene,
    base: GaussianModel,
    payload: Dict[str, object],
    donor_ids: np.ndarray,
    camera_stride: int,
    max_views: int,
    depth_min: float,
    ray_chunk: int,
    surface_mode: str,
    max_hit_to_donor_gap: float,
    max_hit_to_donor_gap_ratio: float,
    min_depth_scale: float,
    projected_center_tolerance_px: float,
    fallback_to_payload_anchor: bool,
    relax_unanchored_with_mesh: bool,
    relaxed_max_hit_to_donor_gap: float,
    relaxed_max_hit_to_donor_gap_ratio: float,
    relaxed_projected_center_tolerance_px: float,
) -> Dict[str, torch.Tensor | np.ndarray | Dict[str, object]]:
    mesh_obj = _load_triangle_mesh(mesh_path)
    face_normals = np.asarray(mesh_obj.face_normals, dtype=np.float32)
    ray_backend = "open3d"
    raycaster = None
    try:
        raycaster = _build_open3d_raycast_scene(mesh_obj)
    except Exception as exc:
        print(f"[surface-proxy-v0] Open3D raycaster failed ({exc}); trying trimesh ray intersector.", flush=True)
        ray_backend = "trimesh"
        raycaster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_obj)

    donor_xyz_all = base.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)[donor_ids]
    donor_scale_max_all = (
        base.get_scaling.detach().cpu().numpy().astype(np.float32, copy=False)[donor_ids].max(axis=1)
    )
    anchor_xyz = np.zeros_like(donor_xyz_all, dtype=np.float32)
    anchor_normal = np.zeros_like(donor_xyz_all, dtype=np.float32)
    anchor_view_index = np.full((donor_ids.shape[0],), -1, dtype=np.int32)
    anchor_face_id = np.full((donor_ids.shape[0],), -1, dtype=np.int64)
    hit_distance = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    donor_distance = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    hit_to_donor_gap = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    depth_scale = np.full((donor_ids.shape[0],), np.nan, dtype=np.float32)
    projected_center_delta_px = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    visible_view_count = np.zeros((donor_ids.shape[0],), dtype=np.int32)
    anchored = np.zeros((donor_ids.shape[0],), dtype=bool)
    anchor_source = np.zeros((donor_ids.shape[0],), dtype=np.int8)

    raw_mesh_hit_seen = np.zeros((donor_ids.shape[0],), dtype=bool)
    raw_best_gap = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    raw_best_projected_center_delta_px = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    relaxed_candidate = np.zeros((donor_ids.shape[0],), dtype=bool)
    relaxed_anchor_xyz = np.zeros_like(donor_xyz_all, dtype=np.float32)
    relaxed_anchor_normal = np.zeros_like(donor_xyz_all, dtype=np.float32)
    relaxed_anchor_view_index = np.full((donor_ids.shape[0],), -1, dtype=np.int32)
    relaxed_anchor_face_id = np.full((donor_ids.shape[0],), -1, dtype=np.int64)
    relaxed_hit_distance = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    relaxed_donor_distance = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    relaxed_hit_to_donor_gap = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)
    relaxed_depth_scale = np.full((donor_ids.shape[0],), np.nan, dtype=np.float32)
    relaxed_projected_center_delta_px = np.full((donor_ids.shape[0],), np.inf, dtype=np.float32)

    train_cameras = list(scene.getTrainCameras())
    stride = max(int(camera_stride), 1)
    camera_indices = list(range(0, len(train_cameras), stride))
    if int(max_views) > 0 and len(camera_indices) > int(max_views):
        ids = np.unique(np.linspace(0, len(camera_indices) - 1, num=int(max_views), dtype=np.int64))
        camera_indices = [camera_indices[int(idx)] for idx in ids.tolist()]
    if not camera_indices:
        raise ValueError("No cameras available for ray-surface proxy anchoring.")
    surface_mode = str(surface_mode)
    if surface_mode not in {"push_away", "pull_front"}:
        raise ValueError(f"Unsupported ray surface mode: {surface_mode!r}")

    raycast_error_count = 0
    raycast_error_message = None
    for camera_idx in camera_indices:
        remaining = np.flatnonzero(~anchored)
        if remaining.size == 0:
            break
        cam = train_cameras[int(camera_idx)]
        projected, valid = _project_points_camera(cam, donor_xyz_all[remaining], depth_min=float(depth_min), margin=0)
        if not np.any(valid):
            continue
        local_visible = np.flatnonzero(valid)
        active_rows = remaining[local_visible]
        visible_view_count[active_rows] += 1

        origin = cam.camera_center.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1, 3)
        ray_origins = np.repeat(origin, active_rows.shape[0], axis=0).astype(np.float32, copy=False)
        ray_vectors = donor_xyz_all[active_rows] - ray_origins
        ray_donor_distance = np.linalg.norm(ray_vectors, axis=1).astype(np.float32, copy=False)
        ray_directions = ray_vectors / np.clip(ray_donor_distance[:, None], 1e-6, None)
        if surface_mode == "push_away":
            post_donor_eps = np.full_like(ray_donor_distance, 1e-5, dtype=np.float32)
            cast_origins = donor_xyz_all[active_rows] + ray_directions * post_donor_eps[:, None]
            hit_distance_offset = ray_donor_distance + post_donor_eps
        else:
            cast_origins = ray_origins
            hit_distance_offset = np.zeros_like(ray_donor_distance, dtype=np.float32)

        try:
            if ray_backend == "open3d":
                hit_mask, hit_points, hit_tris, cast_hit_dist = _cast_rays_open3d(
                    raycaster,
                    cast_origins,
                    ray_directions,
                    ray_chunk=int(ray_chunk),
                )
            else:
                hit_mask, hit_points, hit_tris, cast_hit_dist = _cast_rays_trimesh(
                    raycaster,
                    cast_origins,
                    ray_directions,
                )
        except Exception as exc:
            raycast_error_count += 1
            if raycast_error_message is None:
                raycast_error_message = repr(exc)
            continue

        hit_dist = hit_distance_offset + cast_hit_dist
        if surface_mode == "push_away":
            hit_mask &= hit_dist > (ray_donor_distance + 1e-5)
            gap = hit_dist - ray_donor_distance
        else:
            hit_mask &= hit_dist < (ray_donor_distance - 1e-5)
            gap = ray_donor_distance - hit_dist
        hit_mask &= hit_tris >= 0

        raw_hit_mask = hit_mask.copy()
        if np.any(raw_hit_mask):
            raw_local = np.flatnonzero(raw_hit_mask)
            raw_rows = active_rows[raw_local]
            raw_points = hit_points[raw_local].astype(np.float32, copy=False)
            raw_projected, raw_projected_valid = _project_points_camera(
                cam,
                raw_points,
                depth_min=float(depth_min),
                margin=0,
            )
            raw_center_delta = np.linalg.norm(
                raw_projected[:, :2] - projected[local_visible[raw_local], :2],
                axis=1,
            )
            raw_depth_scale = (
                hit_dist[raw_local] / np.clip(ray_donor_distance[raw_local], 1e-6, None)
            ).astype(np.float32, copy=False)
            raw_basic = raw_projected_valid & np.isfinite(raw_depth_scale) & (raw_depth_scale > 0.0)
            if np.any(raw_basic):
                raw_local = raw_local[raw_basic]
                raw_rows = raw_rows[raw_basic]
                raw_points = raw_points[raw_basic]
                raw_center_delta = raw_center_delta[raw_basic].astype(np.float32, copy=False)
                raw_depth_scale = raw_depth_scale[raw_basic]
                raw_gap = gap[raw_local].astype(np.float32, copy=False)
                raw_faces = np.clip(hit_tris[raw_local], 0, face_normals.shape[0] - 1)
                raw_normals = _orient_normals_toward_donors(face_normals[raw_faces], raw_points, donor_xyz_all[raw_rows])

                raw_update = raw_gap < raw_best_gap[raw_rows]
                if np.any(raw_update):
                    update_rows = raw_rows[raw_update]
                    raw_mesh_hit_seen[update_rows] = True
                    raw_best_gap[update_rows] = raw_gap[raw_update]
                    raw_best_projected_center_delta_px[update_rows] = raw_center_delta[raw_update]

                near_mask = np.ones((raw_rows.shape[0],), dtype=bool)
                if float(relaxed_max_hit_to_donor_gap) > 0.0:
                    near_mask &= raw_gap <= float(relaxed_max_hit_to_donor_gap)
                if float(relaxed_max_hit_to_donor_gap_ratio) > 0.0:
                    relaxed_max_gap = float(relaxed_max_hit_to_donor_gap_ratio) * donor_scale_max_all[raw_rows]
                    near_mask &= raw_gap <= relaxed_max_gap
                if float(relaxed_projected_center_tolerance_px) > 0.0:
                    near_mask &= raw_center_delta <= float(relaxed_projected_center_tolerance_px)
                if np.any(near_mask):
                    near_rows = raw_rows[near_mask]
                    near_gap = raw_gap[near_mask]
                    relaxed_update = near_gap < relaxed_hit_to_donor_gap[near_rows]
                    if np.any(relaxed_update):
                        update_rows = near_rows[relaxed_update]
                        near_points = raw_points[near_mask][relaxed_update]
                        near_normals = raw_normals[near_mask][relaxed_update]
                        near_faces = raw_faces[near_mask][relaxed_update]
                        near_local = raw_local[near_mask][relaxed_update]
                        relaxed_candidate[update_rows] = True
                        relaxed_anchor_xyz[update_rows] = near_points
                        relaxed_anchor_normal[update_rows] = near_normals
                        relaxed_anchor_view_index[update_rows] = int(camera_idx)
                        relaxed_anchor_face_id[update_rows] = near_faces
                        relaxed_hit_distance[update_rows] = hit_dist[near_local]
                        relaxed_donor_distance[update_rows] = ray_donor_distance[near_local]
                        relaxed_hit_to_donor_gap[update_rows] = gap[near_local]
                        relaxed_depth_scale[update_rows] = raw_depth_scale[near_mask][relaxed_update]
                        relaxed_projected_center_delta_px[update_rows] = raw_center_delta[near_mask][relaxed_update]

        if float(max_hit_to_donor_gap) > 0.0:
            hit_mask &= gap <= float(max_hit_to_donor_gap)
        if float(max_hit_to_donor_gap_ratio) > 0.0:
            max_gap = float(max_hit_to_donor_gap_ratio) * donor_scale_max_all[active_rows]
            hit_mask &= gap <= max_gap
        if not np.any(hit_mask):
            continue

        hit_local = np.flatnonzero(hit_mask)
        rows = active_rows[hit_local]
        points = hit_points[hit_local].astype(np.float32, copy=False)
        hit_projected, hit_projected_valid = _project_points_camera(cam, points, depth_min=float(depth_min), margin=0)
        center_delta = np.linalg.norm(hit_projected[:, :2] - projected[local_visible[hit_local], :2], axis=1)
        local_depth_scale = (hit_dist[hit_local] / np.clip(ray_donor_distance[hit_local], 1e-6, None)).astype(
            np.float32,
            copy=False,
        )
        consistency_mask = hit_projected_valid & np.isfinite(local_depth_scale) & (local_depth_scale > 0.0)
        if float(min_depth_scale) > 0.0:
            consistency_mask &= local_depth_scale >= float(min_depth_scale)
        if float(projected_center_tolerance_px) > 0.0:
            consistency_mask &= center_delta <= float(projected_center_tolerance_px)
        if not np.any(consistency_mask):
            continue

        hit_local = hit_local[consistency_mask]
        rows = rows[consistency_mask]
        points = points[consistency_mask]
        center_delta = center_delta[consistency_mask]
        local_depth_scale = local_depth_scale[consistency_mask]
        faces = np.clip(hit_tris[hit_local], 0, face_normals.shape[0] - 1)
        normals = _orient_normals_toward_donors(face_normals[faces], points, donor_xyz_all[rows])
        anchor_xyz[rows] = points
        anchor_normal[rows] = normals
        anchor_view_index[rows] = int(camera_idx)
        anchor_face_id[rows] = faces
        hit_distance[rows] = hit_dist[hit_local]
        donor_distance[rows] = ray_donor_distance[hit_local]
        hit_to_donor_gap[rows] = gap[hit_local]
        depth_scale[rows] = local_depth_scale
        projected_center_delta_px[rows] = center_delta
        anchored[rows] = True
        anchor_source[rows] = 1

    strict_anchor_count = int(np.sum(anchor_source == 1))
    relaxed_anchor_count = 0
    if bool(relax_unanchored_with_mesh) and np.any(~anchored):
        relaxed_rows = np.flatnonzero((~anchored) & relaxed_candidate)
        if relaxed_rows.size > 0:
            anchor_xyz[relaxed_rows] = relaxed_anchor_xyz[relaxed_rows]
            anchor_normal[relaxed_rows] = relaxed_anchor_normal[relaxed_rows]
            anchor_view_index[relaxed_rows] = relaxed_anchor_view_index[relaxed_rows]
            anchor_face_id[relaxed_rows] = relaxed_anchor_face_id[relaxed_rows]
            hit_distance[relaxed_rows] = relaxed_hit_distance[relaxed_rows]
            donor_distance[relaxed_rows] = relaxed_donor_distance[relaxed_rows]
            hit_to_donor_gap[relaxed_rows] = relaxed_hit_to_donor_gap[relaxed_rows]
            depth_scale[relaxed_rows] = relaxed_depth_scale[relaxed_rows]
            projected_center_delta_px[relaxed_rows] = relaxed_projected_center_delta_px[relaxed_rows]
            anchored[relaxed_rows] = True
            anchor_source[relaxed_rows] = 2
            relaxed_anchor_count = int(relaxed_rows.shape[0])

    unresolved_after_ray = ~anchored
    unresolved_not_visible = unresolved_after_ray & (visible_view_count <= 0)
    unresolved_visible_no_mesh = unresolved_after_ray & (visible_view_count > 0) & (~raw_mesh_hit_seen)
    unresolved_mesh_rejected = unresolved_after_ray & raw_mesh_hit_seen

    fallback_count = 0
    if fallback_to_payload_anchor and np.any(~anchored):
        if "anchor_xyz" not in payload or "anchor_normal" not in payload:
            raise KeyError("Ray fallback requested, but payload lacks anchor_xyz/anchor_normal.")
        fallback_rows = np.flatnonzero(~anchored)
        fallback_ids_t = torch.from_numpy(donor_ids[fallback_rows]).to(device=base.get_xyz.device, dtype=torch.long)
        fallback_xyz = torch.as_tensor(payload["anchor_xyz"], device=base.get_xyz.device, dtype=torch.float32)[fallback_ids_t]
        fallback_normal = _normalize(
            torch.as_tensor(payload["anchor_normal"], device=base.get_xyz.device, dtype=torch.float32)[fallback_ids_t]
        )
        anchor_xyz[fallback_rows] = fallback_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        anchor_normal[fallback_rows] = fallback_normal.detach().cpu().numpy().astype(np.float32, copy=False)
        anchored[fallback_rows] = True
        anchor_source[fallback_rows] = 3
        fallback_count = int(fallback_rows.shape[0])

    cloned_rows = np.flatnonzero(anchored)
    if cloned_rows.size == 0:
        raise ValueError("Ray-surface proxy anchoring produced zero cloned donors.")

    cloned_donor_ids = donor_ids[cloned_rows].astype(np.int64, copy=False)
    strict_rows = np.flatnonzero(anchor_source == 1)
    relaxed_rows = np.flatnonzero(anchor_source == 2)
    fallback_rows = np.flatnonzero(anchor_source == 3)
    stats = {
        "mode": "ray_surface",
        "mesh_path": str(mesh_path),
        "ray_backend": str(ray_backend),
        "camera_stride": int(camera_stride),
        "max_views": int(max_views),
        "camera_count_used": int(len(camera_indices)),
        "depth_min": float(depth_min),
        "ray_chunk": int(ray_chunk),
        "surface_mode": str(surface_mode),
        "max_hit_to_donor_gap": float(max_hit_to_donor_gap),
        "max_hit_to_donor_gap_ratio": float(max_hit_to_donor_gap_ratio),
        "min_depth_scale": float(min_depth_scale),
        "projected_center_tolerance_px": float(projected_center_tolerance_px),
        "fallback_to_payload_anchor": bool(fallback_to_payload_anchor),
        "relax_unanchored_with_mesh": bool(relax_unanchored_with_mesh),
        "relaxed_max_hit_to_donor_gap": float(relaxed_max_hit_to_donor_gap),
        "relaxed_max_hit_to_donor_gap_ratio": float(relaxed_max_hit_to_donor_gap_ratio),
        "relaxed_projected_center_tolerance_px": float(relaxed_projected_center_tolerance_px),
        "selected_donors_before_anchor": int(donor_ids.shape[0]),
        "selected_proxy_gaussians": int(cloned_donor_ids.shape[0]),
        "strict_anchor_count": int(strict_anchor_count),
        "relaxed_anchor_count": int(relaxed_anchor_count),
        "ray_hit_count": int(np.sum(anchor_view_index[cloned_rows] >= 0)),
        "fallback_anchor_count": int(fallback_count),
        "uncloned_donor_count": int(donor_ids.shape[0] - cloned_donor_ids.shape[0]),
        "raw_mesh_hit_count": int(raw_mesh_hit_seen.sum()),
        "relaxed_candidate_count": int(relaxed_candidate.sum()),
        "unresolved_after_ray_count": int(unresolved_after_ray.sum()),
        "unresolved_not_visible_count": int(unresolved_not_visible.sum()),
        "unresolved_visible_no_mesh_count": int(unresolved_visible_no_mesh.sum()),
        "unresolved_mesh_rejected_count": int(unresolved_mesh_rejected.sum()),
        "raycast_error_count": int(raycast_error_count),
        "raycast_error_message": raycast_error_message,
        "visible_view_count": _stats(visible_view_count),
        "raw_hit_to_donor_gap": _stats(raw_best_gap[np.isfinite(raw_best_gap)]),
        "raw_projected_center_delta_px": _stats(
            raw_best_projected_center_delta_px[np.isfinite(raw_best_projected_center_delta_px)]
        ),
        "hit_distance": _stats(hit_distance[np.isfinite(hit_distance)]),
        "donor_distance": _stats(donor_distance[np.isfinite(donor_distance)]),
        "hit_to_donor_gap": _stats(hit_to_donor_gap[np.isfinite(hit_to_donor_gap)]),
        "depth_scale": _stats(depth_scale[np.isfinite(depth_scale)]),
        "projected_center_delta_px": _stats(projected_center_delta_px[np.isfinite(projected_center_delta_px)]),
    }
    ray_depth_scale = depth_scale[cloned_rows].astype(np.float32, copy=False)
    ray_depth_scale = np.where(np.isfinite(ray_depth_scale), ray_depth_scale, 1.0).astype(np.float32, copy=False)
    return {
        "anchor_xyz": torch.from_numpy(anchor_xyz[cloned_rows]).to(device=base.get_xyz.device, dtype=torch.float32),
        "anchor_normal": _normalize(
            torch.from_numpy(anchor_normal[cloned_rows]).to(device=base.get_xyz.device, dtype=torch.float32)
        ),
        "cloned_donor_ids": cloned_donor_ids,
        "classification": {
            "selected_donor_ids": donor_ids.astype(np.int64, copy=False),
            "strict_cloned_donor_ids": donor_ids[strict_rows].astype(np.int64, copy=False),
            "relaxed_cloned_donor_ids": donor_ids[relaxed_rows].astype(np.int64, copy=False),
            "fallback_cloned_donor_ids": donor_ids[fallback_rows].astype(np.int64, copy=False),
            "raw_mesh_hit_donor_ids": donor_ids[raw_mesh_hit_seen].astype(np.int64, copy=False),
            "relaxed_candidate_donor_ids": donor_ids[relaxed_candidate].astype(np.int64, copy=False),
            "unresolved_not_visible_donor_ids": donor_ids[unresolved_not_visible].astype(np.int64, copy=False),
            "unresolved_visible_no_mesh_donor_ids": donor_ids[unresolved_visible_no_mesh].astype(np.int64, copy=False),
            "unresolved_mesh_rejected_donor_ids": donor_ids[unresolved_mesh_rejected].astype(np.int64, copy=False),
        },
        "ray_depth_scale": torch.from_numpy(ray_depth_scale).to(device=base.get_xyz.device, dtype=torch.float32),
        "stats": stats,
    }


def _gather_axes(rotation_mats: torch.Tensor, local_ids: torch.Tensor) -> torch.Tensor:
    gather_idx = local_ids[:, None, None].expand(-1, 3, 1)
    return torch.gather(rotation_mats, dim=2, index=gather_idx).squeeze(2)


def _build_proxy_model(
    *,
    base: GaussianModel,
    payload: Dict[str, object],
    donor_ids: np.ndarray,
    anchor_xyz_override: torch.Tensor | None,
    anchor_normal_override: torch.Tensor | None,
    tangent_scale_factor_override: torch.Tensor | None,
    tangent_offset_scale: float,
    tangent_scale_multiplier: float,
    normal_scale_ratio: float,
    normal_offset_ratio: float,
    opacity_scale: float,
    min_proxy_scale: float,
) -> GaussianModel:
    device = base.get_xyz.device
    donor_ids_t = torch.from_numpy(donor_ids).to(device=device, dtype=torch.long)

    donor_xyz = base._xyz.detach()[donor_ids_t]
    donor_features_dc = base._features_dc.detach()[donor_ids_t]
    donor_features_rest = base._features_rest.detach()[donor_ids_t]
    donor_opacity = base.get_opacity.detach()[donor_ids_t].reshape(-1, 1)
    donor_scales = base.get_scaling.detach()[donor_ids_t]
    donor_rot = build_rotation(base._rotation.detach()[donor_ids_t])

    if anchor_xyz_override is not None and anchor_normal_override is not None:
        anchor_xyz = anchor_xyz_override.to(device=device, dtype=torch.float32)
        anchor_normal = _normalize(anchor_normal_override.to(device=device, dtype=torch.float32))
        if int(anchor_xyz.shape[0]) != int(donor_ids_t.shape[0]):
            raise ValueError(
                f"anchor_xyz_override length mismatch: {int(anchor_xyz.shape[0])} vs {int(donor_ids_t.shape[0])}"
            )
    else:
        anchor_xyz = torch.as_tensor(payload["anchor_xyz"], device=device, dtype=torch.float32)[donor_ids_t]
        anchor_normal = _normalize(torch.as_tensor(payload["anchor_normal"], device=device, dtype=torch.float32)[donor_ids_t])

    offset = donor_xyz - anchor_xyz
    tangent_offset = offset - torch.sum(offset * anchor_normal, dim=1, keepdim=True) * anchor_normal

    sorted_scales, sorted_local_ids = torch.sort(donor_scales, dim=1, descending=True)
    major_axis = _gather_axes(donor_rot, sorted_local_ids[:, 0])
    mid_axis = _gather_axes(donor_rot, sorted_local_ids[:, 1])

    major_proj = major_axis - torch.sum(major_axis * anchor_normal, dim=1, keepdim=True) * anchor_normal
    mid_proj = mid_axis - torch.sum(mid_axis * anchor_normal, dim=1, keepdim=True) * anchor_normal
    stable_u, _ = _stable_tangent_basis(anchor_normal)

    major_norm = torch.linalg.norm(major_proj, dim=1, keepdim=True)
    mid_norm = torch.linalg.norm(mid_proj, dim=1, keepdim=True)
    tangent_u = stable_u.clone()
    use_mid = (major_norm[:, 0] <= 1e-6) & (mid_norm[:, 0] > 1e-6)
    use_major = major_norm[:, 0] > 1e-6
    if torch.any(use_mid):
        tangent_u[use_mid] = _normalize(mid_proj[use_mid])
    if torch.any(use_major):
        tangent_u[use_major] = _normalize(major_proj[use_major])
    tangent_v = _normalize(torch.cross(anchor_normal, tangent_u, dim=1))
    tangent_u = _normalize(torch.cross(tangent_v, anchor_normal, dim=1))

    tangent_scale = sorted_scales[:, :2] * float(tangent_scale_multiplier)
    if tangent_scale_factor_override is not None:
        tangent_scale_factor = torch.clamp(
            tangent_scale_factor_override.to(device=device, dtype=torch.float32).reshape(-1, 1),
            min=1e-6,
        )
        if int(tangent_scale_factor.shape[0]) != int(donor_ids_t.shape[0]):
            raise ValueError(
                "tangent_scale_factor_override length mismatch: "
                f"{int(tangent_scale_factor.shape[0])} vs {int(donor_ids_t.shape[0])}"
            )
        tangent_scale = tangent_scale * tangent_scale_factor
    tangent_scale = torch.clamp(tangent_scale, min=float(min_proxy_scale))
    normal_scale = torch.minimum(
        sorted_scales[:, 2],
        float(normal_scale_ratio) * torch.minimum(tangent_scale[:, 0], tangent_scale[:, 1]),
    )
    normal_scale = torch.clamp(normal_scale, min=float(min_proxy_scale))
    proxy_scales = torch.stack((tangent_scale[:, 0], tangent_scale[:, 1], normal_scale), dim=1)

    proxy_xyz = anchor_xyz + float(tangent_offset_scale) * tangent_offset + (float(normal_offset_ratio) * normal_scale[:, None]) * anchor_normal
    proxy_rot_mats = torch.stack((tangent_u, tangent_v, anchor_normal), dim=2)
    proxy_quat = _quaternion_from_rotation_matrix(proxy_rot_mats)
    proxy_opacity = torch.clamp(donor_opacity * float(opacity_scale), min=1e-4, max=0.995)

    proxy = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    proxy.active_sh_degree = int(base.active_sh_degree)
    proxy.spatial_lr_scale = float(base.spatial_lr_scale)
    proxy._xyz = nn.Parameter(proxy_xyz.detach().clone().requires_grad_(False))
    proxy._features_dc = nn.Parameter(donor_features_dc.detach().clone().requires_grad_(False))
    proxy._features_rest = nn.Parameter(donor_features_rest.detach().clone().requires_grad_(False))
    proxy._opacity = nn.Parameter(inverse_sigmoid(proxy_opacity).detach().clone().requires_grad_(False))
    proxy._scaling = nn.Parameter(torch.log(proxy_scales).detach().clone().requires_grad_(False))
    proxy._rotation = nn.Parameter(proxy_quat.detach().clone().requires_grad_(False))
    proxy.filter_3D = torch.zeros((int(donor_ids_t.shape[0]), 1), dtype=torch.float32, device=device)
    proxy.max_radii2D = torch.zeros((int(donor_ids_t.shape[0]),), dtype=torch.float32, device=device)
    proxy.xyz_gradient_accum = torch.zeros((int(donor_ids_t.shape[0]), 1), dtype=torch.float32, device=device)
    proxy.xyz_gradient_accum_abs = torch.zeros((int(donor_ids_t.shape[0]), 1), dtype=torch.float32, device=device)
    proxy.xyz_gradient_accum_abs_max = torch.zeros((int(donor_ids_t.shape[0]), 1), dtype=torch.float32, device=device)
    proxy.denom = torch.zeros((int(donor_ids_t.shape[0]), 1), dtype=torch.float32, device=device)
    proxy.init_tracking_state(int(donor_ids_t.shape[0]), source_tag=int(GaussianSourceTag.PRIOR_INJECTED))
    proxy._seed_id = donor_ids_t.to(dtype=torch.int64)
    donor_generation = getattr(base, "_generation", None)
    if torch.is_tensor(donor_generation) and int(donor_generation.shape[0]) == int(base.get_xyz.shape[0]):
        proxy._generation = donor_generation.detach()[donor_ids_t].to(dtype=torch.int32) + 1
    return proxy


def _tensor_stats(values: torch.Tensor) -> Dict[str, float | int]:
    arr = values.detach().reshape(-1).float().cpu().numpy()
    return _stats(arr)


def _concat_models(*, base: GaussianModel, surface: GaussianModel, proxy: GaussianModel) -> GaussianModel:
    device = base.get_xyz.device
    out = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    out.active_sh_degree = max(int(surface.active_sh_degree), int(proxy.active_sh_degree))
    out.spatial_lr_scale = float(base.spatial_lr_scale)

    out._xyz = nn.Parameter(torch.cat((surface._xyz.detach(), proxy._xyz.detach()), dim=0).requires_grad_(False))
    out._features_dc = nn.Parameter(
        torch.cat((surface._features_dc.detach(), proxy._features_dc.detach()), dim=0).requires_grad_(False)
    )
    out._features_rest = nn.Parameter(
        torch.cat((surface._features_rest.detach(), proxy._features_rest.detach()), dim=0).requires_grad_(False)
    )
    out._opacity = nn.Parameter(torch.cat((surface._opacity.detach(), proxy._opacity.detach()), dim=0).requires_grad_(False))
    out._scaling = nn.Parameter(torch.cat((surface._scaling.detach(), proxy._scaling.detach()), dim=0).requires_grad_(False))
    out._rotation = nn.Parameter(torch.cat((surface._rotation.detach(), proxy._rotation.detach()), dim=0).requires_grad_(False))

    count = int(out._xyz.shape[0])
    out.filter_3D = torch.cat((surface.filter_3D.detach(), proxy.filter_3D.detach()), dim=0)
    out.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=device)
    out.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=device)
    out.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=device)
    out.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=device)
    out.denom = torch.zeros((count, 1), dtype=torch.float32, device=device)
    out.init_tracking_state(count)
    for name in ("_source_tag", "_seed_id", "_generation", "_edge_touched", "_edge_touch_iter"):
        lhs = getattr(surface, name, None)
        rhs = getattr(proxy, name, None)
        if torch.is_tensor(lhs) and torch.is_tensor(rhs):
            setattr(out, name, torch.cat((lhs.detach(), rhs.detach()), dim=0))
    return out


def export_augmented_model(
    *,
    scene_root: Path,
    model_path: Path,
    iteration: int,
    mask_payload_path: Path,
    surface_mask_key: str,
    donor_mask_key: str,
    output_root: Path,
    images_subdir: str,
    split: str,
    max_views: int,
    white_background: bool,
    max_donors: int,
    proxy_anchor_mode: str,
    proxy_output_mode: str,
    mesh_path: str,
    proxy_ray_camera_stride: int,
    proxy_ray_max_views: int,
    proxy_ray_depth_min: float,
    proxy_ray_chunk: int,
    proxy_ray_surface_mode: str,
    proxy_ray_max_hit_to_donor_gap: float,
    proxy_ray_max_hit_to_donor_gap_ratio: float,
    proxy_ray_min_depth_scale: float,
    proxy_ray_projected_center_tolerance_px: float,
    proxy_ray_preserve_screen_scale: bool,
    proxy_ray_fallback_to_anchor: bool,
    proxy_ray_relax_unanchored_with_mesh: bool,
    proxy_ray_relaxed_max_hit_to_donor_gap: float,
    proxy_ray_relaxed_max_hit_to_donor_gap_ratio: float,
    proxy_ray_relaxed_projected_center_tolerance_px: float,
    tangent_offset_scale: float,
    tangent_scale_multiplier: float,
    normal_scale_ratio: float,
    normal_offset_ratio: float,
    opacity_scale: float,
    min_proxy_scale: float,
) -> Dict[str, object]:
    dataset = _build_dataset_args(str(scene_root), str(model_path), images_subdir, white_background)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)

    payload = torch.load(mask_payload_path, map_location="cpu")
    total = int(gaussians.get_xyz.shape[0])
    surface_mask = _load_mask_payload(mask_payload_path, surface_mask_key, total).to(device="cuda")
    donor_mask = _load_mask_payload(mask_payload_path, donor_mask_key, total).to(device="cpu")
    donor_mask &= ~surface_mask.detach().cpu().bool()

    surface_count = int(surface_mask.sum().item())
    donor_candidate_count = int(donor_mask.sum().item())
    if surface_count <= 0:
        raise ValueError(f"Surface mask '{surface_mask_key}' selected zero gaussians.")
    if donor_candidate_count <= 0:
        raise ValueError(f"Donor mask '{donor_mask_key}' selected zero candidate gaussians after excluding surface mask.")

    donor_ids = _select_donor_ids(
        payload=payload,
        donor_mask=donor_mask,
        max_donors=max_donors,
        model_opacity=gaussians.get_opacity.detach().cpu(),
    )
    if str(proxy_anchor_mode) == "ray_surface":
        resolved_mesh_path = _resolve_mesh_path(mask_payload_path, str(mesh_path))
        anchor_specs = _build_ray_surface_anchor_specs(
            mesh_path=resolved_mesh_path,
            scene=scene,
            base=gaussians,
            payload=payload,
            donor_ids=donor_ids,
            camera_stride=int(proxy_ray_camera_stride),
            max_views=int(proxy_ray_max_views),
            depth_min=float(proxy_ray_depth_min),
            ray_chunk=int(proxy_ray_chunk),
            surface_mode=str(proxy_ray_surface_mode),
            max_hit_to_donor_gap=float(proxy_ray_max_hit_to_donor_gap),
            max_hit_to_donor_gap_ratio=float(proxy_ray_max_hit_to_donor_gap_ratio),
            min_depth_scale=float(proxy_ray_min_depth_scale),
            projected_center_tolerance_px=float(proxy_ray_projected_center_tolerance_px),
            fallback_to_payload_anchor=bool(proxy_ray_fallback_to_anchor),
            relax_unanchored_with_mesh=bool(proxy_ray_relax_unanchored_with_mesh),
            relaxed_max_hit_to_donor_gap=float(proxy_ray_relaxed_max_hit_to_donor_gap),
            relaxed_max_hit_to_donor_gap_ratio=float(proxy_ray_relaxed_max_hit_to_donor_gap_ratio),
            relaxed_projected_center_tolerance_px=float(proxy_ray_relaxed_projected_center_tolerance_px),
        )
    elif str(proxy_anchor_mode) == "anchor":
        anchor_specs = _build_payload_anchor_specs(
            payload=payload,
            donor_ids=donor_ids,
            device=gaussians.get_xyz.device,
        )
    else:
        raise ValueError(f"Unsupported proxy_anchor_mode={proxy_anchor_mode!r}")

    cloned_donor_ids = np.asarray(anchor_specs["cloned_donor_ids"], dtype=np.int64)
    surface_subset = _clone_subset_gaussians(gaussians, surface_mask)
    proxy_model = _build_proxy_model(
        base=gaussians,
        payload=payload,
        donor_ids=cloned_donor_ids,
        anchor_xyz_override=anchor_specs["anchor_xyz"],
        anchor_normal_override=anchor_specs["anchor_normal"],
        tangent_scale_factor_override=anchor_specs.get("ray_depth_scale")
        if bool(proxy_ray_preserve_screen_scale)
        else None,
        tangent_offset_scale=tangent_offset_scale,
        tangent_scale_multiplier=tangent_scale_multiplier,
        normal_scale_ratio=normal_scale_ratio,
        normal_offset_ratio=normal_offset_ratio,
        opacity_scale=opacity_scale,
        min_proxy_scale=min_proxy_scale,
    )
    removed_original_donor_gaussians = 0
    removed_selected_uncloned_donor_gaussians = 0
    replacement_policy = str(proxy_output_mode)

    if str(proxy_output_mode) == "surface_plus_proxy":
        augmented = _concat_models(base=gaussians, surface=surface_subset, proxy=proxy_model)
        kept_original_gaussians = int(surface_count)
        removed_original_donor_gaussians = int(total - surface_count)
    elif str(proxy_output_mode) in {"replace_cloned_donors", "replace_successful_donors"}:
        keep_mask = torch.ones((total,), dtype=torch.bool, device=gaussians.get_xyz.device)
        remove_ids_t = torch.from_numpy(cloned_donor_ids).to(device=gaussians.get_xyz.device, dtype=torch.long)
        keep_mask[remove_ids_t] = False
        kept_model = _clone_subset_gaussians(gaussians, keep_mask)
        augmented = _concat_models(base=gaussians, surface=kept_model, proxy=proxy_model)
        kept_original_gaussians = int(keep_mask.sum().item())
        removed_original_donor_gaussians = int(cloned_donor_ids.shape[0])
        replacement_policy = "remove_cloned_donors_add_surface_proxies"
    elif str(proxy_output_mode) == "replace_selected_donors":
        # Aggressive ablation: remove the whole selected off-surface donor set. This can
        # expose no-mesh coverage holes, so the default keeps unresolved donors available.
        keep_mask = torch.ones((total,), dtype=torch.bool, device=gaussians.get_xyz.device)
        keep_mask[torch.from_numpy(donor_ids).to(device=gaussians.get_xyz.device, dtype=torch.long)] = False
        kept_model = _clone_subset_gaussians(gaussians, keep_mask)
        augmented = _concat_models(base=gaussians, surface=kept_model, proxy=proxy_model)
        kept_original_gaussians = int(keep_mask.sum().item())
        removed_original_donor_gaussians = int(donor_ids.shape[0])
        removed_selected_uncloned_donor_gaussians = int(donor_ids.shape[0] - cloned_donor_ids.shape[0])
        replacement_policy = "remove_selected_donors_add_cloned_surface_proxies"
    else:
        raise ValueError(f"Unsupported proxy_output_mode={proxy_output_mode!r}")

    augmented_root = output_root / "augmented_model"
    point_dir = augmented_root / "point_cloud" / f"iteration_{loaded_iter}"
    point_dir.mkdir(parents=True, exist_ok=True)
    _copy_render_config(model_path, augmented_root)
    augmented.save_ply(str(point_dir / "point_cloud.ply"))
    augmented.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    classification_masks_path = augmented_root / "surface_proxy_migration_masks.pt"
    classification = anchor_specs.get("classification", {})
    classification_payload: Dict[str, object] = {
        "source_gaussians": int(total),
        "surface_mask_key": str(surface_mask_key),
        "donor_mask_key": str(donor_mask_key),
    }
    if isinstance(classification, dict):
        for key, value in classification.items():
            ids = np.asarray(value, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < total)]
            classification_payload[key] = torch.from_numpy(ids.astype(np.int64, copy=False))
            mask = torch.zeros((total,), dtype=torch.bool)
            if ids.size > 0:
                mask[torch.from_numpy(ids.astype(np.int64, copy=False))] = True
            mask_key = str(key)
            if mask_key.endswith("_ids"):
                mask_key = f"{mask_key[:-4]}_mask"
            else:
                mask_key = f"{mask_key}_mask"
            classification_payload[mask_key] = mask
    torch.save(classification_payload, classification_masks_path)

    render_dataset = _build_dataset_args(str(scene_root), str(augmented_root), images_subdir, white_background)
    render_gaussians = GaussianModel(render_dataset.sh_degree)
    render_scene = Scene(
        render_dataset,
        render_gaussians,
        load_iteration=loaded_iter,
        shuffle=False,
        skip_test=False,
        skip_train=False,
    )
    render_gaussians.compute_3D_filter(render_scene.getTrainCameras().copy(), CUDA=False)
    background = torch.tensor(
        [1, 1, 1] if white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    render_summary: Dict[str, object] = {}
    for split_name, views in _iter_views(render_scene, split):
        selected_views, selected_indices = _select_uniform(list(views), max_views)
        render_root = output_root / split_name / f"ours_{loaded_iter}" / "renders"
        render_root.mkdir(parents=True, exist_ok=True)
        for output_idx, view in zip(selected_indices, selected_views):
            render_pkg = render_simple(view, render_gaussians, background)
            _save_rgb(render_root / f"{output_idx:05d}.png", render_pkg["render"][:3])
        render_summary[split_name] = {
            "num_views": int(len(selected_views)),
            "source_num_views": int(len(views)),
            "selected_indices": selected_indices,
            "render_root": str(render_root),
        }

    summary = {
        "mode": "build_surface_proxy_augmented_field_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(loaded_iter),
        "mask_payload_path": str(mask_payload_path),
        "surface_mask_key": str(surface_mask_key),
        "donor_mask_key": str(donor_mask_key),
        "images_subdir": str(images_subdir),
        "split": str(split),
        "max_views": int(max_views),
        "white_background": bool(white_background),
        "proxy_parameters": {
            "max_donors": int(max_donors),
            "anchor_mode": str(proxy_anchor_mode),
            "output_mode": str(proxy_output_mode),
            "replacement_policy": replacement_policy,
            "anchor_stats": anchor_specs["stats"],
            "preserve_screen_scale": bool(proxy_ray_preserve_screen_scale),
            "tangent_offset_scale": float(tangent_offset_scale),
            "tangent_scale_multiplier": float(tangent_scale_multiplier),
            "normal_scale_ratio": float(normal_scale_ratio),
            "normal_offset_ratio": float(normal_offset_ratio),
            "opacity_scale": float(opacity_scale),
            "min_proxy_scale": float(min_proxy_scale),
        },
        "counts": {
            "source_gaussians": int(total),
            "surface_subset_gaussians": int(surface_count),
            "donor_candidate_gaussians": int(donor_candidate_count),
            "selected_donor_gaussians_before_anchor": int(donor_ids.shape[0]),
            "selected_proxy_gaussians": int(cloned_donor_ids.shape[0]),
            "kept_original_gaussians": int(kept_original_gaussians),
            "removed_original_donor_gaussians": int(removed_original_donor_gaussians),
            "removed_selected_uncloned_donor_gaussians": int(removed_selected_uncloned_donor_gaussians),
            "augmented_gaussians": int(augmented.get_xyz.shape[0]),
        },
        "proxy_stats": {
            "xyz": _tensor_stats(proxy_model.get_xyz),
            "opacity": _tensor_stats(proxy_model.get_opacity.reshape(-1)),
            "scale": _tensor_stats(proxy_model.get_scaling.reshape(-1)),
            "scale_max": _tensor_stats(torch.max(proxy_model.get_scaling, dim=1).values),
            "filter_3d": _tensor_stats(proxy_model.filter_3D.reshape(-1)),
        },
        "paths": {
            "augmented_model_root": str(augmented_root),
            "augmented_ply": str(point_dir / "point_cloud.ply"),
            "augmented_tags": str(point_dir / "gaussian_tags.pt"),
            "surface_proxy_migration_masks": str(classification_masks_path),
        },
        "renders": render_summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a surface-only field augmented with proxies projected from active non-surface Gaussians.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--mask_payload_path", required=True)
    parser.add_argument("--surface_mask_key", default="surface_candidate")
    parser.add_argument("--donor_mask_key", default="surface_complement_active")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--max_donors", type=int, default=200000)
    parser.add_argument("--proxy_anchor_mode", choices=["ray_surface", "anchor"], default="ray_surface")
    parser.add_argument(
        "--proxy_output_mode",
        choices=["replace_cloned_donors", "replace_successful_donors", "replace_selected_donors", "surface_plus_proxy"],
        default="replace_cloned_donors",
    )
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--proxy_ray_camera_stride", type=int, default=1)
    parser.add_argument("--proxy_ray_max_views", type=int, default=0)
    parser.add_argument("--proxy_ray_depth_min", type=float, default=0.01)
    parser.add_argument("--proxy_ray_chunk", type=int, default=262144)
    parser.add_argument("--proxy_ray_surface_mode", choices=["push_away", "pull_front"], default="push_away")
    parser.add_argument("--proxy_ray_max_hit_to_donor_gap", type=float, default=0.0)
    parser.add_argument("--proxy_ray_max_hit_to_donor_gap_ratio", type=float, default=0.0)
    parser.add_argument("--proxy_ray_min_depth_scale", type=float, default=0.0)
    parser.add_argument("--proxy_ray_projected_center_tolerance_px", type=float, default=1.0)
    parser.add_argument("--proxy_ray_disable_preserve_screen_scale", action="store_true")
    parser.add_argument("--proxy_ray_fallback_to_anchor", action="store_true")
    parser.add_argument("--proxy_ray_relax_unanchored_with_mesh", action="store_true")
    parser.add_argument("--proxy_ray_relaxed_max_hit_to_donor_gap", type=float, default=0.0)
    parser.add_argument("--proxy_ray_relaxed_max_hit_to_donor_gap_ratio", type=float, default=0.0)
    parser.add_argument("--proxy_ray_relaxed_projected_center_tolerance_px", type=float, default=0.0)
    parser.add_argument("--proxy_ray_allow_hit_after_donor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--proxy_tangent_offset_scale", type=float, default=1.0)
    parser.add_argument("--proxy_tangent_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--proxy_normal_scale_ratio", type=float, default=0.35)
    parser.add_argument("--proxy_normal_offset_ratio", type=float, default=0.10)
    parser.add_argument("--proxy_opacity_scale", type=float, default=1.0)
    parser.add_argument("--min_proxy_scale", type=float, default=1e-4)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    mask_payload_path = Path(args.mask_payload_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    iteration = _resolve_iteration(model_path, int(args.iteration))

    summary = export_augmented_model(
        scene_root=scene_root,
        model_path=model_path,
        iteration=iteration,
        mask_payload_path=mask_payload_path,
        surface_mask_key=str(args.surface_mask_key),
        donor_mask_key=str(args.donor_mask_key),
        output_root=output_root,
        images_subdir=str(args.images_subdir),
        split=str(args.split),
        max_views=int(args.max_views),
        white_background=bool(args.white_background),
        max_donors=int(args.max_donors),
        proxy_anchor_mode=str(args.proxy_anchor_mode),
        proxy_output_mode=str(args.proxy_output_mode),
        mesh_path=str(args.mesh_path),
        proxy_ray_camera_stride=int(args.proxy_ray_camera_stride),
        proxy_ray_max_views=int(args.proxy_ray_max_views),
        proxy_ray_depth_min=float(args.proxy_ray_depth_min),
        proxy_ray_chunk=int(args.proxy_ray_chunk),
        proxy_ray_surface_mode="push_away" if bool(args.proxy_ray_allow_hit_after_donor) else str(args.proxy_ray_surface_mode),
        proxy_ray_max_hit_to_donor_gap=float(args.proxy_ray_max_hit_to_donor_gap),
        proxy_ray_max_hit_to_donor_gap_ratio=float(args.proxy_ray_max_hit_to_donor_gap_ratio),
        proxy_ray_min_depth_scale=float(args.proxy_ray_min_depth_scale),
        proxy_ray_projected_center_tolerance_px=float(args.proxy_ray_projected_center_tolerance_px),
        proxy_ray_preserve_screen_scale=not bool(args.proxy_ray_disable_preserve_screen_scale),
        proxy_ray_fallback_to_anchor=bool(args.proxy_ray_fallback_to_anchor),
        proxy_ray_relax_unanchored_with_mesh=bool(args.proxy_ray_relax_unanchored_with_mesh),
        proxy_ray_relaxed_max_hit_to_donor_gap=float(args.proxy_ray_relaxed_max_hit_to_donor_gap),
        proxy_ray_relaxed_max_hit_to_donor_gap_ratio=float(args.proxy_ray_relaxed_max_hit_to_donor_gap_ratio),
        proxy_ray_relaxed_projected_center_tolerance_px=float(args.proxy_ray_relaxed_projected_center_tolerance_px),
        tangent_offset_scale=float(args.proxy_tangent_offset_scale),
        tangent_scale_multiplier=float(args.proxy_tangent_scale_multiplier),
        normal_scale_ratio=float(args.proxy_normal_scale_ratio),
        normal_offset_ratio=float(args.proxy_normal_offset_ratio),
        opacity_scale=float(args.proxy_opacity_scale),
        min_proxy_scale=float(args.min_proxy_scale),
    )
    print(json.dumps(summary, indent=2))
    print(f"[done] summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
