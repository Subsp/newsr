from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree


SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def stats_from_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        return {"count": 0, "finite_count": int(finite.sum())}
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


def prop_names(vertex: Any, prefix: str) -> list[str]:
    names = list(vertex.data.dtype.names or [])
    return sorted(
        [name for name in names if name.startswith(prefix)],
        key=lambda item: int(item.rsplit("_", 1)[1]) if item.rsplit("_", 1)[-1].isdigit() else item,
    )


def stack_props(vertex: Any, names: list[str]) -> np.ndarray:
    if not names:
        return np.empty((len(vertex.data), 0), dtype=np.float32)
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def load_gaussian_ply(point_cloud_path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(str(point_cloud_path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or [])
    n = len(vertex.data)

    xyz = stack_props(vertex, ["x", "y", "z"])
    scale_names = prop_names(vertex, "scale_")
    scale_raw = stack_props(vertex, scale_names)
    scale = np.exp(scale_raw) if scale_raw.size else np.zeros((n, 3), dtype=np.float32)
    filter_3d = np.asarray(vertex["filter_3D"], dtype=np.float32) if "filter_3D" in names else np.zeros((n,), dtype=np.float32)
    effective_scale = np.sqrt(scale * scale + filter_3d[:, None] * filter_3d[:, None])

    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float32) if "opacity" in names else np.zeros((n,), dtype=np.float32)
    opacity = sigmoid(opacity_raw).astype(np.float32)

    dc_names = prop_names(vertex, "f_dc_")
    dc = stack_props(vertex, dc_names)
    if dc.shape[1] >= 3:
        rgb_dc = dc[:, :3] * SH_C0 + 0.5
        dc_luma = (0.2126 * rgb_dc[:, 0] + 0.7152 * rgb_dc[:, 1] + 0.0722 * rgb_dc[:, 2]).astype(np.float32)
        dc_chroma_delta = (np.max(rgb_dc[:, :3], axis=1) - np.min(rgb_dc[:, :3], axis=1)).astype(np.float32)
    else:
        dc_luma = np.zeros((n,), dtype=np.float32)
        dc_chroma_delta = np.zeros((n,), dtype=np.float32)

    rest_norm = np.zeros((n,), dtype=np.float32)
    for name in prop_names(vertex, "f_rest_"):
        arr = np.asarray(vertex[name], dtype=np.float32)
        rest_norm += arr * arr
    rest_norm = np.sqrt(rest_norm)

    scale_major = np.max(effective_scale, axis=1).astype(np.float32)
    scale_minor = np.min(effective_scale, axis=1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = (scale_major / np.maximum(scale_minor, 1e-8)).astype(np.float32)

    return {
        "xyz": xyz.astype(np.float32, copy=False),
        "scale_raw": scale_raw.astype(np.float32, copy=False),
        "scale_activated": scale.astype(np.float32, copy=False),
        "filter_3D": filter_3d.astype(np.float32, copy=False),
        "effective_scale": effective_scale.astype(np.float32, copy=False),
        "scale_major": scale_major,
        "scale_minor": scale_minor,
        "effective_anisotropy": anisotropy,
        "opacity_raw": opacity_raw.astype(np.float32, copy=False),
        "opacity": opacity,
        "dc_luma": dc_luma,
        "dc_chroma_delta": dc_chroma_delta,
        "rest_norm": rest_norm.astype(np.float32, copy=False),
    }


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


def query_surface(mesh_obj: trimesh.Trimesh, points_xyz: np.ndarray, mode: str, sample_count: int, chunk_size: int) -> dict[str, np.ndarray]:
    if mode in {"auto", "exact_open3d"}:
        try:
            return query_surface_open3d(mesh_obj, points_xyz, chunk_size=chunk_size)
        except Exception as exc:
            if mode == "exact_open3d":
                raise
            print(f"[mesh-align-score] exact Open3D query failed ({exc}); falling back to sampled surface.")
    return query_surface_sampled(mesh_obj, points_xyz, sample_count=sample_count)


def write_point_cloud(path: Path, xyz: np.ndarray, score: np.ndarray, max_points: int) -> None:
    if xyz.shape[0] == 0:
        return
    ids = np.arange(xyz.shape[0])
    if int(max_points) > 0 and xyz.shape[0] > int(max_points):
        ids = np.argsort(-score, kind="stable")[: int(max_points)]
    score_sel = score[ids]
    denom = max(float(np.percentile(score_sel, 99)), 1e-6)
    heat = np.clip(score_sel / denom, 0.0, 1.0)
    colors = np.stack(
        [
            np.full_like(heat, 255.0),
            160.0 * (1.0 - heat),
            30.0 * (1.0 - heat),
            np.full_like(heat, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    trimesh.points.PointCloud(xyz[ids], colors=colors).export(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Gaussian centers against a local mesh surface and export mesh-alignment features."
    )
    parser.add_argument("--point_cloud_ply", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface_query_mode", choices=["auto", "exact_open3d", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=1_000_000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--surface_distance_threshold", type=float, default=0.03)
    parser.add_argument("--min_candidate_opacity", type=float, default=0.02)
    parser.add_argument("--min_effective_anisotropy", type=float, default=4.0)
    parser.add_argument("--min_normal_over_minor", type=float, default=1.5)
    parser.add_argument("--min_distance_over_major", type=float, default=0.75)
    parser.add_argument("--min_candidate_score", type=float, default=1.0)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    args = parser.parse_args()

    point_cloud_path = Path(args.point_cloud_ply).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gs = load_gaussian_ply(point_cloud_path)
    mesh_obj = load_triangle_mesh(mesh_path)
    surface = query_surface(
        mesh_obj=mesh_obj,
        points_xyz=gs["xyz"],
        mode=str(args.surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )

    delta = gs["xyz"] - surface["nearest_surface_point"]
    signed_normal_offset = np.sum(delta * surface["nearest_surface_normal"], axis=1).astype(np.float32)
    normal_offset_abs = np.abs(signed_normal_offset).astype(np.float32)
    surface_distance = surface["surface_distance"].astype(np.float32, copy=False)
    tangent_offset = np.sqrt(np.maximum(surface_distance * surface_distance - normal_offset_abs * normal_offset_abs, 0.0)).astype(np.float32)

    scale_major = gs["scale_major"]
    scale_minor = gs["scale_minor"]
    distance_over_major = (surface_distance / np.maximum(scale_major, 1e-6)).astype(np.float32)
    normal_over_minor = (normal_offset_abs / np.maximum(scale_minor, 1e-6)).astype(np.float32)
    tangent_over_major = (tangent_offset / np.maximum(scale_major, 1e-6)).astype(np.float32)

    opacity_gate = gs["opacity"] >= float(args.min_candidate_opacity)
    distance_gate = surface_distance >= float(args.surface_distance_threshold)
    aniso_gate = gs["effective_anisotropy"] >= float(args.min_effective_anisotropy)
    normal_gate = normal_over_minor >= float(args.min_normal_over_minor)
    footprint_gate = distance_over_major >= float(args.min_distance_over_major)

    score = (
        np.maximum(normal_over_minor - float(args.min_normal_over_minor), 0.0)
        + 0.65 * np.maximum(distance_over_major - float(args.min_distance_over_major), 0.0)
        + 0.25 * np.maximum(tangent_over_major - 1.0, 0.0)
        + 0.10 * np.log1p(np.maximum(gs["effective_anisotropy"] - 1.0, 0.0))
    ).astype(np.float32)
    candidate_mask = opacity_gate & distance_gate & aniso_gate & (normal_gate | footprint_gate) & (score >= float(args.min_candidate_score))
    candidate_ids = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)

    payload = {
        "mode": "gaussian_mesh_alignment_v0",
        "candidate_mask": torch.from_numpy(candidate_mask.copy()),
        "candidate_ids": torch.from_numpy(candidate_ids.copy()),
        "candidate_score": torch.from_numpy(score.copy()),
        "surface_distance": torch.from_numpy(surface_distance.copy()),
        "nearest_surface_point": torch.from_numpy(surface["nearest_surface_point"].copy()),
        "nearest_surface_normal": torch.from_numpy(surface["nearest_surface_normal"].copy()),
        "nearest_face_id": torch.from_numpy(surface["nearest_face_id"].copy()),
        "signed_normal_offset": torch.from_numpy(signed_normal_offset.copy()),
        "normal_offset_abs": torch.from_numpy(normal_offset_abs.copy()),
        "tangent_offset": torch.from_numpy(tangent_offset.copy()),
        "distance_over_major": torch.from_numpy(distance_over_major.copy()),
        "normal_over_minor": torch.from_numpy(normal_over_minor.copy()),
        "tangent_over_major": torch.from_numpy(tangent_over_major.copy()),
        "scale_major": torch.from_numpy(scale_major.copy()),
        "scale_minor": torch.from_numpy(scale_minor.copy()),
        "effective_anisotropy": torch.from_numpy(gs["effective_anisotropy"].copy()),
        "opacity": torch.from_numpy(gs["opacity"].copy()),
        "dc_luma": torch.from_numpy(gs["dc_luma"].copy()),
        "dc_chroma_delta": torch.from_numpy(gs["dc_chroma_delta"].copy()),
        "rest_norm": torch.from_numpy(gs["rest_norm"].copy()),
    }
    payload_path = output_dir / "gaussian_mesh_alignment_v0.pt"
    torch.save(payload, str(payload_path))

    preview_path = output_dir / "mesh_alignment_candidates_v0.ply"
    write_point_cloud(preview_path, gs["xyz"][candidate_ids], score[candidate_ids], max_points=int(args.preview_max_points))

    summary = {
        "mode": "gaussian_mesh_alignment_v0",
        "point_cloud_ply": str(point_cloud_path),
        "mesh_path": str(mesh_path),
        "output_dir": str(output_dir),
        "payload_path": str(payload_path),
        "preview_ply": str(preview_path),
        "surface_query_mode_requested": str(args.surface_query_mode),
        "surface_query_mode_used": str(surface["surface_query_mode_used"]),
        "parameters": {
            "surface_distance_threshold": float(args.surface_distance_threshold),
            "min_candidate_opacity": float(args.min_candidate_opacity),
            "min_effective_anisotropy": float(args.min_effective_anisotropy),
            "min_normal_over_minor": float(args.min_normal_over_minor),
            "min_distance_over_major": float(args.min_distance_over_major),
            "min_candidate_score": float(args.min_candidate_score),
        },
        "counts": {
            "gaussian_count": int(gs["xyz"].shape[0]),
            "opacity_gate_count": int(np.sum(opacity_gate)),
            "distance_gate_count": int(np.sum(distance_gate)),
            "anisotropy_gate_count": int(np.sum(aniso_gate)),
            "candidate_count": int(candidate_ids.shape[0]),
            "candidate_ratio": float(candidate_ids.shape[0] / max(int(gs["xyz"].shape[0]), 1)),
        },
        "stats": {
            "surface_distance_all": stats_from_array(surface_distance),
            "surface_distance_candidates": stats_from_array(surface_distance[candidate_mask]),
            "normal_over_minor_all": stats_from_array(normal_over_minor),
            "normal_over_minor_candidates": stats_from_array(normal_over_minor[candidate_mask]),
            "distance_over_major_all": stats_from_array(distance_over_major),
            "distance_over_major_candidates": stats_from_array(distance_over_major[candidate_mask]),
            "candidate_score_all": stats_from_array(score),
            "candidate_score_candidates": stats_from_array(score[candidate_mask]),
            "anisotropy_candidates": stats_from_array(gs["effective_anisotropy"][candidate_mask]),
            "opacity_candidates": stats_from_array(gs["opacity"][candidate_mask]),
        },
    }
    summary_path = output_dir / "gaussian_mesh_alignment_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] payload: {payload_path}")
    print(f"[done] preview: {preview_path}")


if __name__ == "__main__":
    main()
