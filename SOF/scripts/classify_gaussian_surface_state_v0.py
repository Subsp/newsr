#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from plyfile import PlyData
except ModuleNotFoundError:
    PlyData = None


CLASS_NAMES = {
    0: "no_mesh_neutral",
    1: "surface_carrier",
    2: "near_surface_uncertain",
    3: "off_surface_near_mesh",
    4: "axis_touching_surface",
    5: "low_opacity_neutral",
}


CLASS_COLORS = {
    0: (130, 130, 130, 255),
    1: (40, 210, 120, 255),
    2: (245, 200, 55, 255),
    3: (230, 70, 55, 255),
    4: (75, 155, 255, 255),
    5: (70, 70, 70, 255),
}


def stats_from_array(values: np.ndarray) -> dict[str, Any]:
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
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def load_triangle_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load triangle mesh from {mesh_path}")


def query_surface_sampled(mesh_obj: trimesh.Trimesh, points_xyz: np.ndarray, sample_count: int) -> dict[str, np.ndarray]:
    surface_points, face_ids = trimesh.sample.sample_surface(mesh_obj, max(int(sample_count), 1))
    surface_points = np.asarray(surface_points, dtype=np.float32)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)
    tree = cKDTree(surface_points)
    distance, nearest = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    return {
        "surface_query_mode_used": "sample",
        "surface_distance": np.asarray(distance, dtype=np.float32),
        "nearest_surface_point": surface_points[nearest].astype(np.float32, copy=False),
        "nearest_surface_normal": normals[nearest].astype(np.float32, copy=False),
        "nearest_face_id": face_ids[nearest].astype(np.int64, copy=False),
    }


def query_surface_open3d(mesh_obj: trimesh.Trimesh, points_xyz: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    import open3d as o3d

    vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
    faces = np.asarray(mesh_obj.faces, dtype=np.int32)
    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    legacy_mesh.triangles = o3d.utility.Vector3iVector(faces)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    closest_points = np.empty_like(points_xyz, dtype=np.float32)
    face_ids = np.empty((points_xyz.shape[0],), dtype=np.int64)
    chunk = max(int(chunk_size), 1)
    for begin in range(0, points_xyz.shape[0], chunk):
        end = min(begin + chunk, points_xyz.shape[0])
        query = o3d.core.Tensor(points_xyz[begin:end].astype(np.float32, copy=False), dtype=o3d.core.Dtype.Float32)
        result = scene.compute_closest_points(query)
        closest_points[begin:end] = result["points"].numpy().astype(np.float32, copy=False)
        tri_ids = result["primitive_ids"].numpy().astype(np.int64, copy=False)
        face_ids[begin:end] = np.clip(tri_ids, 0, len(mesh_obj.faces) - 1)

    delta = points_xyz.astype(np.float32, copy=False) - closest_points
    return {
        "surface_query_mode_used": "exact_open3d",
        "surface_distance": np.linalg.norm(delta, axis=1).astype(np.float32, copy=False),
        "nearest_surface_point": closest_points,
        "nearest_surface_normal": np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32),
        "nearest_face_id": face_ids,
    }


def query_surface(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    mode: str,
    sample_count: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    if mode in {"auto", "exact_open3d"}:
        try:
            return query_surface_open3d(mesh_obj, points_xyz, chunk_size=chunk_size)
        except Exception as exc:
            if mode == "exact_open3d":
                raise
            print(f"[surface-state-v0] exact Open3D query failed ({exc}); falling back to sampled surface.")
    return query_surface_sampled(mesh_obj, points_xyz, sample_count=sample_count)


def _ply_prop_names(names: list[str], prefix: str) -> list[str]:
    return sorted(
        [name for name in names if name.startswith(prefix)],
        key=lambda item: int(item.rsplit("_", 1)[1]) if item.rsplit("_", 1)[-1].isdigit() else item,
    )


def _stack_prop_dict(data: Dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    if not names:
        first = next(iter(data.values()))
        return np.empty((int(np.asarray(first).reshape(-1).shape[0]), 0), dtype=np.float32)
    return np.stack([np.asarray(data[name], dtype=np.float32).reshape(-1) for name in names], axis=1)


def _read_vertex_properties(point_cloud_path: Path) -> Dict[str, np.ndarray]:
    if PlyData is not None:
        ply = PlyData.read(str(point_cloud_path))
        vertex = ply["vertex"]
        return {
            name: np.asarray(vertex[name]).reshape(-1)
            for name in list(vertex.data.dtype.names or [])
        }

    loaded = trimesh.load(str(point_cloud_path), process=False)
    raw = getattr(loaded, "metadata", {}).get("_ply_raw", {})
    if "vertex" not in raw or "data" not in raw["vertex"]:
        raise ModuleNotFoundError(
            "plyfile is not installed, and trimesh did not expose raw PLY vertex properties."
        )
    return {
        name: np.asarray(value).reshape(-1)
        for name, value in raw["vertex"]["data"].items()
    }


def sigmoid_np(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def normalize_np(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), eps, None)


def quaternion_to_rotation_np(quat: np.ndarray) -> np.ndarray:
    q = normalize_np(quat.astype(np.float32, copy=False))
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - w * z)
    rot[:, 0, 2] = 2.0 * (x * z + w * y)
    rot[:, 1, 0] = 2.0 * (x * y + w * z)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - w * x)
    rot[:, 2, 0] = 2.0 * (x * z - w * y)
    rot[:, 2, 1] = 2.0 * (y * z + w * x)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def load_gaussian_ply_full(point_cloud_path: Path) -> Dict[str, np.ndarray]:
    props = _read_vertex_properties(point_cloud_path)
    names = list(props.keys())
    name_set = set(names)
    total = int(np.asarray(props["x"]).reshape(-1).shape[0])

    xyz = _stack_prop_dict(props, ["x", "y", "z"]).astype(np.float32, copy=False)
    scale_raw = _stack_prop_dict(props, _ply_prop_names(names, "scale_")).astype(np.float32, copy=False)
    if scale_raw.shape[1] != 3:
        raise ValueError(f"Expected 3 scale channels in {point_cloud_path}, got {scale_raw.shape[1]}")
    scale = np.exp(scale_raw).astype(np.float32, copy=False)

    rot_raw = _stack_prop_dict(props, _ply_prop_names(names, "rot")).astype(np.float32, copy=False)
    if rot_raw.shape[1] != 4:
        raise ValueError(f"Expected 4 quaternion channels in {point_cloud_path}, got {rot_raw.shape[1]}")
    rotation = quaternion_to_rotation_np(rot_raw)

    filter_3d = (
        np.asarray(props["filter_3D"], dtype=np.float32).reshape(-1, 1)
        if "filter_3D" in name_set
        else np.zeros((total, 1), dtype=np.float32)
    )
    effective_scale = np.sqrt(scale * scale + filter_3d * filter_3d).astype(np.float32, copy=False)

    opacity_raw = (
        np.asarray(props["opacity"], dtype=np.float32).reshape(-1)
        if "opacity" in name_set
        else np.zeros((total,), dtype=np.float32)
    )
    opacity = sigmoid_np(opacity_raw)

    scale_major = np.max(effective_scale, axis=1).astype(np.float32, copy=False)
    scale_minor = np.min(effective_scale, axis=1).astype(np.float32, copy=False)
    scale_mid = np.partition(effective_scale, 1, axis=1)[:, 1].astype(np.float32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = (scale_major / np.maximum(scale_minor, 1e-8)).astype(np.float32, copy=False)

    return {
        "xyz": xyz,
        "scale_raw": scale_raw,
        "scale": scale,
        "effective_scale": effective_scale,
        "rotation": rotation,
        "opacity_raw": opacity_raw,
        "opacity": opacity,
        "scale_major": scale_major,
        "scale_mid": scale_mid,
        "scale_minor": scale_minor,
        "effective_anisotropy": anisotropy,
        "filter_3D": filter_3d.reshape(-1).astype(np.float32, copy=False),
    }


def mesh_edge_lengths(mesh_obj: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float32)
    tri = vertices[faces]
    edges = np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ],
        axis=1,
    )
    edge = np.mean(edges, axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(edge) & (edge > 0.0)
    fallback = float(np.median(edge[finite])) if np.any(finite) else 1e-3
    edge[~finite] = fallback
    return edge


def edge_for_faces(edge: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    if edge.size == 0:
        return np.ones_like(face_ids, dtype=np.float32) * 1e-3
    clipped = np.clip(face_ids.astype(np.int64, copy=False), 0, edge.shape[0] - 1)
    return edge[clipped].astype(np.float32, copy=False)


def tangent_basis(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = normalize_np(normals.astype(np.float32, copy=False))
    ref = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (normal.shape[0], 1))
    use_y = np.abs(normal[:, 0]) > 0.9
    ref[use_y] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = normalize_np(np.cross(ref, normal))
    tangent_v = normalize_np(np.cross(normal, tangent_u))
    return tangent_u.astype(np.float32, copy=False), tangent_v.astype(np.float32, copy=False)


def sigma_along(rotations: np.ndarray, scales: np.ndarray, directions: np.ndarray) -> np.ndarray:
    local = np.einsum("nij,ni->nj", rotations, directions.astype(np.float32, copy=False))
    return np.sqrt(np.sum((local * scales) ** 2, axis=1)).astype(np.float32, copy=False)


def build_axis_samples(
    xyz: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    anisotropy: np.ndarray,
    *,
    anisotropy_threshold: float,
    sample_count: int,
    sample_extent: float,
) -> Dict[str, np.ndarray]:
    total = int(xyz.shape[0])
    center_ids = np.arange(total, dtype=np.int64)
    source_parts = [center_ids]
    t_parts = [np.zeros((total,), dtype=np.float32)]

    use_axis = (anisotropy >= float(anisotropy_threshold)) & (int(sample_count) > 1)
    axis_ids = np.flatnonzero(use_axis).astype(np.int64, copy=False)
    if axis_ids.size > 0:
        t = np.linspace(-1.0, 1.0, num=max(int(sample_count), 2), dtype=np.float32)
        t = t[np.abs(t) > 1e-6]
        if t.size > 0:
            source_parts.append(np.repeat(axis_ids, t.shape[0]).astype(np.int64, copy=False))
            t_parts.append(np.tile(t, axis_ids.shape[0]).astype(np.float32, copy=False))

    source_idx = np.concatenate(source_parts).astype(np.int64, copy=False)
    sample_t = np.concatenate(t_parts).astype(np.float32, copy=False)
    major_axis_id = np.argmax(scales, axis=1).astype(np.int64, copy=False)
    major_scale = np.max(scales, axis=1).astype(np.float32, copy=False)
    major_axis = rotations[source_idx, :, major_axis_id[source_idx]]
    points = xyz[source_idx] + major_axis * (
        sample_t[:, None] * major_scale[source_idx, None] * float(sample_extent)
    )
    sample_count_per_gaussian = np.bincount(source_idx, minlength=total).astype(np.int32)
    return {
        "points": points.astype(np.float32, copy=False),
        "source_idx": source_idx,
        "sample_t": sample_t,
        "axis_sampled": use_axis.astype(bool, copy=False),
        "sample_count_per_gaussian": sample_count_per_gaussian,
    }


def best_sample_by_distance(source_idx: np.ndarray, distance: np.ndarray, total: int) -> np.ndarray:
    order = np.lexsort((distance, source_idx))
    ordered_source = source_idx[order]
    unique, first = np.unique(ordered_source, return_index=True)
    best = np.zeros((total,), dtype=np.int64)
    best[unique] = order[first]
    return best


def reduce_min(source_idx: np.ndarray, values: np.ndarray, total: int, fill: float = np.inf) -> np.ndarray:
    out = np.full((total,), fill, dtype=np.float32)
    np.minimum.at(out, source_idx, values.astype(np.float32, copy=False))
    out[~np.isfinite(out)] = 0.0
    return out


def reduce_max(source_idx: np.ndarray, values: np.ndarray, total: int) -> np.ndarray:
    out = np.zeros((total,), dtype=np.float32)
    np.maximum.at(out, source_idx, values.astype(np.float32, copy=False))
    return out


def export_class_preview(path: Path, xyz: np.ndarray, class_id: np.ndarray, score: np.ndarray, max_points: int) -> None:
    if xyz.shape[0] == 0:
        return
    ids = np.arange(xyz.shape[0], dtype=np.int64)
    if int(max_points) > 0 and ids.size > int(max_points):
        order = np.argsort(-score, kind="stable")
        ids = order[: int(max_points)].astype(np.int64, copy=False)
    colors = np.asarray([CLASS_COLORS[int(cls)] for cls in class_id[ids]], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(xyz[ids], colors=colors).export(path)


def export_scalar_preview(path: Path, xyz: np.ndarray, scalar: np.ndarray, max_points: int) -> None:
    if xyz.shape[0] == 0:
        return
    ids = np.arange(xyz.shape[0], dtype=np.int64)
    if int(max_points) > 0 and ids.size > int(max_points):
        ids = np.argsort(-scalar, kind="stable")[: int(max_points)].astype(np.int64, copy=False)
    value = np.clip(scalar[ids], 0.0, 1.0)
    colors = np.stack(
        [
            255.0 * value,
            210.0 * (1.0 - np.abs(value - 0.5) * 2.0),
            255.0 * (1.0 - value),
            np.full_like(value, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(xyz[ids], colors=colors).export(path)


def payload_tensor(value: np.ndarray):
    array = np.asarray(value).copy()
    if torch is None:
        return array
    return torch.from_numpy(array)


def save_payload(path: Path, payload: Dict[str, Any]) -> Path:
    if torch is not None:
        torch.save(payload, path)
        return path

    npz_path = path.with_suffix(".npz")
    serializable = {
        key: np.asarray(value)
        for key, value in payload.items()
        if key != "class_names"
    }
    np.savez_compressed(npz_path, **serializable)
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a frozen Gaussian field against a partial mesh without moving the field. "
            "No-mesh regions are kept neutral, and elongated Gaussians can be checked by major-axis samples."
        )
    )
    parser.add_argument("--point_cloud_ply", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_query_mode", choices=["auto", "exact_open3d", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=1_000_000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--axis_anisotropy_threshold", type=float, default=4.0)
    parser.add_argument("--axis_sample_count", type=int, default=3)
    parser.add_argument("--axis_sample_extent", type=float, default=1.0)
    parser.add_argument("--coverage_abs_radius", type=float, default=0.08)
    parser.add_argument("--coverage_scale_ratio", type=float, default=3.0)
    parser.add_argument("--coverage_edge_ratio", type=float, default=0.5)
    parser.add_argument("--coverage_tau_floor", type=float, default=0.02)
    parser.add_argument("--attach_tau_abs", type=float, default=0.03)
    parser.add_argument("--attach_tau_scale_ratio", type=float, default=1.5)
    parser.add_argument("--attach_tau_edge_ratio", type=float, default=0.1)
    parser.add_argument("--attach_tau_floor", type=float, default=0.01)
    parser.add_argument("--surface_conf_min", type=float, default=0.35)
    parser.add_argument("--surface_max_d_norm", type=float, default=1.0)
    parser.add_argument("--near_surface_max_d_norm", type=float, default=2.5)
    parser.add_argument("--off_surface_d_norm", type=float, default=3.5)
    parser.add_argument("--sigma_dist", type=float, default=1.25)
    parser.add_argument("--cov_beta", type=float, default=0.5)
    parser.add_argument("--normal_score_floor", type=float, default=0.35)
    parser.add_argument("--min_action_opacity", type=float, default=0.01)
    parser.add_argument("--preview_max_points", type=int, default=250000)
    args = parser.parse_args()

    point_cloud_path = Path(args.point_cloud_ply).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gs = load_gaussian_ply_full(point_cloud_path)
    mesh_obj = load_triangle_mesh(mesh_path)
    total = int(gs["xyz"].shape[0])
    print(f"[surface-state-v0] gaussians: {total}")
    print(f"[surface-state-v0] mesh: vertices={len(mesh_obj.vertices)} faces={len(mesh_obj.faces)}")

    samples = build_axis_samples(
        gs["xyz"],
        gs["rotation"],
        gs["effective_scale"],
        gs["effective_anisotropy"],
        anisotropy_threshold=float(args.axis_anisotropy_threshold),
        sample_count=int(args.axis_sample_count),
        sample_extent=float(args.axis_sample_extent),
    )
    sample_points = samples["points"]
    sample_source = samples["source_idx"]
    print(
        f"[surface-state-v0] samples: {sample_points.shape[0]} "
        f"(axis-sampled gaussians={int(np.sum(samples['axis_sampled']))})"
    )

    surface = query_surface(
        mesh_obj=mesh_obj,
        points_xyz=sample_points,
        mode=str(args.surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )
    sample_dist = surface["surface_distance"].astype(np.float32, copy=False)
    sample_face = surface["nearest_face_id"].astype(np.int64, copy=False)
    sample_anchor = surface["nearest_surface_point"].astype(np.float32, copy=False)
    sample_normal = normalize_np(surface["nearest_surface_normal"].astype(np.float32, copy=False))
    local_edge = edge_for_faces(mesh_edge_lengths(mesh_obj), sample_face)

    coverage_tau = np.maximum.reduce(
        [
            np.full((sample_points.shape[0],), float(args.coverage_abs_radius), dtype=np.float32),
            float(args.coverage_scale_ratio) * gs["scale_major"][sample_source],
            float(args.coverage_edge_ratio) * local_edge,
            np.full((sample_points.shape[0],), float(args.coverage_tau_floor), dtype=np.float32),
        ]
    ).astype(np.float32)
    sample_covered = sample_dist <= coverage_tau
    covered_count = np.bincount(
        sample_source,
        weights=sample_covered.astype(np.float32),
        minlength=total,
    ).astype(np.float32)
    mesh_covered = covered_count > 0.0
    mesh_coverage_weight = np.clip(
        covered_count / np.maximum(samples["sample_count_per_gaussian"].astype(np.float32), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    attach_tau = np.maximum.reduce(
        [
            np.full((sample_points.shape[0],), float(args.attach_tau_abs), dtype=np.float32),
            float(args.attach_tau_scale_ratio) * gs["scale_major"][sample_source],
            float(args.attach_tau_edge_ratio) * local_edge,
            np.full((sample_points.shape[0],), float(args.attach_tau_floor), dtype=np.float32),
        ]
    ).astype(np.float32)
    sample_d_norm = (sample_dist / np.clip(attach_tau, 1e-8, None)).astype(np.float32, copy=False)
    best_sample = best_sample_by_distance(sample_source, sample_d_norm, total=total)
    min_surface_distance = reduce_min(sample_source, sample_dist, total=total)
    min_d_norm = reduce_min(sample_source, sample_d_norm, total=total)
    max_coverage_score = reduce_max(sample_source, np.exp(-sample_dist / np.clip(coverage_tau, 1e-8, None)), total=total)

    center_surface_distance = sample_dist[:total].astype(np.float32, copy=False)
    center_d_norm = sample_d_norm[:total].astype(np.float32, copy=False)
    center_covered = sample_covered[:total].astype(bool, copy=False)
    best_anchor = sample_anchor[best_sample].astype(np.float32, copy=False)
    best_normal = sample_normal[best_sample].astype(np.float32, copy=False)
    best_face = sample_face[best_sample].astype(np.int64, copy=False)
    best_sample_t = samples["sample_t"][best_sample].astype(np.float32, copy=False)

    tangent_u, tangent_v = tangent_basis(best_normal)
    sigma_normal = sigma_along(gs["rotation"], gs["effective_scale"], best_normal)
    sigma_tangent_u = sigma_along(gs["rotation"], gs["effective_scale"], tangent_u)
    sigma_tangent_v = sigma_along(gs["rotation"], gs["effective_scale"], tangent_v)
    sigma_tangent_mean = 0.5 * (sigma_tangent_u + sigma_tangent_v)
    q_dist = np.exp(-np.square(min_d_norm / max(float(args.sigma_dist), 1e-6))).astype(np.float32)
    q_cov = np.exp(
        -sigma_normal / np.maximum(float(args.cov_beta) * np.maximum(sigma_tangent_mean, 1e-8), 1e-8)
    ).astype(np.float32)
    min_axis_id = np.argmin(gs["effective_scale"], axis=1).astype(np.int64, copy=False)
    min_axis_dir = gs["rotation"][np.arange(total), :, min_axis_id]
    q_normal_raw = np.abs(np.sum(min_axis_dir * best_normal, axis=1)).clip(0.0, 1.0).astype(np.float32)
    q_normal = (float(args.normal_score_floor) + (1.0 - float(args.normal_score_floor)) * q_normal_raw).astype(np.float32)
    attach_conf = (q_dist * q_cov * q_normal * mesh_coverage_weight).clip(0.0, 1.0).astype(np.float32)
    attach_conf[~mesh_covered] = 0.0

    low_opacity = gs["opacity"] < float(args.min_action_opacity)
    axis_touching = (
        mesh_covered
        & samples["axis_sampled"]
        & (np.abs(best_sample_t) > 1e-6)
        & (min_d_norm <= float(args.near_surface_max_d_norm))
        & (center_d_norm > float(args.near_surface_max_d_norm))
    )

    class_id = np.full((total,), 0, dtype=np.int64)
    class_id[mesh_covered & low_opacity] = 5
    active = mesh_covered & (~low_opacity)
    surface_carrier = active & (attach_conf >= float(args.surface_conf_min)) & (min_d_norm <= float(args.surface_max_d_norm))
    class_id[surface_carrier] = 1
    class_id[active & (~surface_carrier) & axis_touching] = 4
    near_uncertain = active & (class_id == 0) & (min_d_norm <= float(args.near_surface_max_d_norm))
    class_id[near_uncertain] = 2
    off_surface = active & (class_id == 0) & (min_d_norm > float(args.near_surface_max_d_norm))
    class_id[off_surface] = 3

    off_progress = np.clip(
        (min_d_norm - float(args.near_surface_max_d_norm))
        / max(float(args.off_surface_d_norm) - float(args.near_surface_max_d_norm), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    opacity_gate = np.clip(gs["opacity"] / max(float(args.min_action_opacity), 1e-6), 0.0, 1.0).astype(np.float32)
    carrier_weight = np.zeros((total,), dtype=np.float32)
    carrier_weight[class_id == 1] = attach_conf[class_id == 1]
    carrier_weight[class_id == 4] = 0.5 * attach_conf[class_id == 4]
    suggested_attach_weight = np.zeros((total,), dtype=np.float32)
    suggested_attach_weight[class_id == 1] = attach_conf[class_id == 1]
    suggested_attach_weight[class_id == 2] = 0.25 * attach_conf[class_id == 2]
    suggested_attach_weight[class_id == 3] = 0.20 * (1.0 - attach_conf[class_id == 3]) * off_progress[class_id == 3]
    suggested_attach_weight[class_id == 4] = 0.30 * attach_conf[class_id == 4]
    suggested_suppress_weight = np.zeros((total,), dtype=np.float32)
    suggested_suppress_weight[class_id == 3] = (
        off_progress[class_id == 3]
        * opacity_gate[class_id == 3]
        * (1.0 - attach_conf[class_id == 3])
    ).astype(np.float32)
    neutral_weight = ((class_id == 0) | (class_id == 5)).astype(np.float32)

    delta_to_anchor = (gs["xyz"] - best_anchor).astype(np.float32, copy=False)
    signed_normal_offset = np.sum(delta_to_anchor * best_normal, axis=1).astype(np.float32, copy=False)
    tangent_offset = np.sqrt(
        np.maximum(np.sum(delta_to_anchor * delta_to_anchor, axis=1) - signed_normal_offset * signed_normal_offset, 0.0)
    ).astype(np.float32)

    payload = {
        "version": "gaussian_surface_state_v0",
        "class_names": CLASS_NAMES,
        "gaussian_id": payload_tensor(np.arange(total, dtype=np.int64)),
        "class_id": payload_tensor(class_id),
        "mesh_covered": payload_tensor(mesh_covered.astype(bool, copy=False)),
        "no_mesh_neutral": payload_tensor((class_id == 0).astype(bool, copy=False)),
        "surface_carrier": payload_tensor((class_id == 1).astype(bool, copy=False)),
        "near_surface_uncertain": payload_tensor((class_id == 2).astype(bool, copy=False)),
        "off_surface_near_mesh": payload_tensor((class_id == 3).astype(bool, copy=False)),
        "surface_candidate": payload_tensor(((class_id == 1) | (class_id == 4)).astype(bool, copy=False)),
        "surface_or_uncertain": payload_tensor(((class_id == 1) | (class_id == 2) | (class_id == 4)).astype(bool, copy=False)),
        "low_opacity_neutral": payload_tensor((class_id == 5).astype(bool, copy=False)),
        "axis_touching_surface": payload_tensor(axis_touching.astype(bool, copy=False)),
        "xyz": payload_tensor(gs["xyz"].astype(np.float32, copy=False)),
        "opacity": payload_tensor(gs["opacity"].astype(np.float32, copy=False)),
        "scale_major": payload_tensor(gs["scale_major"].astype(np.float32, copy=False)),
        "scale_mid": payload_tensor(gs["scale_mid"].astype(np.float32, copy=False)),
        "scale_minor": payload_tensor(gs["scale_minor"].astype(np.float32, copy=False)),
        "effective_anisotropy": payload_tensor(gs["effective_anisotropy"].astype(np.float32, copy=False)),
        "anchor_xyz": payload_tensor(best_anchor),
        "anchor_normal": payload_tensor(best_normal),
        "nearest_face_id": payload_tensor(best_face),
        "best_sample_t": payload_tensor(best_sample_t),
        "center_surface_distance": payload_tensor(center_surface_distance),
        "min_axis_surface_distance": payload_tensor(min_surface_distance),
        "center_d_norm": payload_tensor(center_d_norm),
        "min_axis_d_norm": payload_tensor(min_d_norm),
        "signed_normal_offset": payload_tensor(signed_normal_offset),
        "tangent_offset": payload_tensor(tangent_offset),
        "mesh_coverage_weight": payload_tensor(mesh_coverage_weight),
        "mesh_coverage_score": payload_tensor(max_coverage_score),
        "covered_axis_sample_count": payload_tensor(covered_count.astype(np.float32, copy=False)),
        "attach_conf": payload_tensor(attach_conf),
        "q_dist": payload_tensor(q_dist),
        "q_cov": payload_tensor(q_cov),
        "q_normal": payload_tensor(q_normal),
        "q_normal_raw": payload_tensor(q_normal_raw),
        "sigma_normal": payload_tensor(sigma_normal),
        "sigma_tangent_mean": payload_tensor(sigma_tangent_mean.astype(np.float32, copy=False)),
        "carrier_weight": payload_tensor(carrier_weight),
        "suggested_attach_weight": payload_tensor(suggested_attach_weight),
        "suggested_suppress_weight": payload_tensor(suggested_suppress_weight),
        "neutral_weight": payload_tensor(neutral_weight),
    }
    payload_path = save_payload(output_dir / "gaussian_surface_state_v0.pt", payload)

    export_class_preview(
        output_dir / "gaussian_surface_state_classes_v0.ply",
        gs["xyz"],
        class_id,
        np.maximum.reduce([attach_conf, suggested_suppress_weight, mesh_coverage_weight]),
        max_points=int(args.preview_max_points),
    )
    export_scalar_preview(
        output_dir / "gaussian_surface_attach_conf_v0.ply",
        gs["xyz"],
        attach_conf,
        max_points=int(args.preview_max_points),
    )
    export_scalar_preview(
        output_dir / "gaussian_surface_suppress_suggestion_v0.ply",
        gs["xyz"],
        suggested_suppress_weight,
        max_points=int(args.preview_max_points),
    )

    class_counts = {
        CLASS_NAMES[int(cls)]: int(np.sum(class_id == int(cls)))
        for cls in sorted(CLASS_NAMES)
    }
    summary: Dict[str, Any] = {
        "version": "gaussian_surface_state_v0",
        "point_cloud_ply": str(point_cloud_path),
        "mesh_path": str(mesh_path),
        "output_dir": str(output_dir),
        "surface_query_mode_requested": str(args.surface_query_mode),
        "surface_query_mode_used": str(surface.get("surface_query_mode_used", args.surface_query_mode)),
        "counts": {
            "gaussian_count": total,
            "sample_count": int(sample_points.shape[0]),
            "axis_sampled_gaussians": int(np.sum(samples["axis_sampled"])),
            "mesh_covered": int(np.sum(mesh_covered)),
            "no_mesh_neutral": int(np.sum(class_id == 0)),
            "axis_touching_surface": int(np.sum(class_id == 4)),
            "classes": class_counts,
        },
        "parameters": vars(args),
        "stats": {
            "center_surface_distance": stats_from_array(center_surface_distance),
            "min_axis_surface_distance": stats_from_array(min_surface_distance),
            "center_d_norm": stats_from_array(center_d_norm),
            "min_axis_d_norm": stats_from_array(min_d_norm),
            "attach_conf": stats_from_array(attach_conf),
            "mesh_coverage_weight": stats_from_array(mesh_coverage_weight),
            "suggested_suppress_weight": stats_from_array(suggested_suppress_weight),
            "opacity": stats_from_array(gs["opacity"]),
            "anisotropy": stats_from_array(gs["effective_anisotropy"]),
        },
        "paths": {
            "payload": str(payload_path),
            "class_preview": str(output_dir / "gaussian_surface_state_classes_v0.ply"),
            "attach_preview": str(output_dir / "gaussian_surface_attach_conf_v0.ply"),
            "suppress_preview": str(output_dir / "gaussian_surface_suppress_suggestion_v0.ply"),
            "summary": str(output_dir / "gaussian_surface_state_v0_summary.json"),
        },
        "notes": [
            "No-mesh Gaussians are neutral by construction: missing mesh coverage does not imply off-surface conflict.",
            "For elongated Gaussians, major-axis samples can make an axis-touching Gaussian avoid false off-surface suppression.",
            "This payload is diagnostic only; it does not modify the input Gaussian field.",
        ],
    }
    summary_path = output_dir / "gaussian_surface_state_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] payload: {payload_path}")
    print(f"[done] preview: {output_dir / 'gaussian_surface_state_classes_v0.ply'}")


if __name__ == "__main__":
    main()
