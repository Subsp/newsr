#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score_gaussian_mesh_alignment_v0 import load_triangle_mesh


def stats_from_array(values: np.ndarray) -> Dict[str, Any]:
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
        "p05": float(np.percentile(arr, 5.0)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.maximum(norm, eps)).astype(np.float32, copy=False)


def load_offset_payload(path: Path | None, vertex_count: int) -> Dict[str, np.ndarray | bool]:
    if path is None or not path.is_file():
        return {
            "loaded": False,
            "active_vertex_ids": np.zeros((0,), dtype=np.int64),
            "active_vertex_delta": np.zeros((0,), dtype=np.float32),
            "active_vertex_normals": np.zeros((0, 3), dtype=np.float32),
        }
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected offset payload dict, got {type(payload)!r}")
    ids = payload.get("active_vertex_ids")
    delta = payload.get("active_vertex_delta")
    normals = payload.get("active_vertex_normals")
    if ids is None or delta is None or normals is None:
        raise KeyError("offset payload must contain active_vertex_ids, active_vertex_delta, active_vertex_normals")
    ids_np = ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids)
    delta_np = delta.detach().cpu().numpy() if torch.is_tensor(delta) else np.asarray(delta)
    normals_np = normals.detach().cpu().numpy() if torch.is_tensor(normals) else np.asarray(normals)
    ids_np = ids_np.reshape(-1).astype(np.int64, copy=False)
    delta_np = delta_np.reshape(-1).astype(np.float32, copy=False)
    normals_np = normals_np.astype(np.float32, copy=False)
    valid = (ids_np >= 0) & (ids_np < int(vertex_count)) & np.isfinite(delta_np)
    if normals_np.ndim == 2 and normals_np.shape[0] == ids_np.shape[0] and normals_np.shape[1] == 3:
        valid &= np.all(np.isfinite(normals_np), axis=1)
    else:
        normals_np = np.zeros((ids_np.shape[0], 3), dtype=np.float32)
    return {
        "loaded": True,
        "active_vertex_ids": ids_np[valid],
        "active_vertex_delta": delta_np[valid],
        "active_vertex_normals": normals_np[valid],
    }


def sample_edge_lengths(vertices: np.ndarray, faces: np.ndarray, max_faces: int, seed: int) -> np.ndarray:
    if faces.shape[0] == 0 or int(max_faces) == 0:
        return np.zeros((0,), dtype=np.float32)
    if int(max_faces) > 0 and faces.shape[0] > int(max_faces):
        rng = np.random.default_rng(int(seed))
        ids = rng.choice(faces.shape[0], size=int(max_faces), replace=False)
        faces = faces[ids]
    tri = vertices[faces]
    e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    return np.concatenate([e01, e12, e20]).astype(np.float32, copy=False)


def colorize_values(values: np.ndarray, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    scale = max(float(vmax), 1e-8)
    t = np.clip(values / scale, 0.0, 1.0)
    # Blue -> cyan -> yellow -> red, intentionally simple and readable in PLY viewers.
    r = np.clip(2.0 * t - 0.35, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(1.0 - 2.0 * t, 0.0, 1.0)
    return np.round(np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)


def export_point_cloud(path: Path, points: np.ndarray, values: np.ndarray, max_points: int, seed: int, vmax: float) -> None:
    points = np.asarray(points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if points.shape[0] == 0 or values.shape[0] == 0 or int(max_points) == 0:
        return
    ids = np.arange(points.shape[0])
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        ids = rng.choice(points.shape[0], size=int(max_points), replace=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points[ids], colors=colorize_values(values[ids], vmax=vmax)).export(path)


def threshold_counts(values: np.ndarray, thresholds: list[float]) -> Dict[str, Dict[str, float | int]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    total = max(int(values.shape[0]), 1)
    out: Dict[str, Dict[str, float | int]] = {}
    for threshold in thresholds:
        count = int(np.count_nonzero(values > float(threshold)))
        out[f">{threshold:g}"] = {"count": count, "ratio": float(count / total)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how much a refined mesh moved relative to its source mesh.")
    parser.add_argument("--source_mesh_path", required=True)
    parser.add_argument("--refined_mesh_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--offset_payload_path", default="")
    parser.add_argument("--edge_sample_faces", type=int, default=500000)
    parser.add_argument("--debug_point_cap", type=int, default=200000)
    parser.add_argument("--debug_vmax", type=float, default=0.02)
    parser.add_argument("--compute_full_vertex_normals", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_mesh_path = Path(args.source_mesh_path).expanduser().resolve()
    refined_mesh_path = Path(args.refined_mesh_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not source_mesh_path.is_file():
        raise FileNotFoundError(f"source mesh not found: {source_mesh_path}")
    if not refined_mesh_path.is_file():
        raise FileNotFoundError(f"refined mesh not found: {refined_mesh_path}")
    offset_payload_path = Path(args.offset_payload_path).expanduser().resolve() if str(args.offset_payload_path).strip() else None

    source_mesh = load_triangle_mesh(source_mesh_path)
    refined_mesh = load_triangle_mesh(refined_mesh_path)
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float32)
    refined_vertices = np.asarray(refined_mesh.vertices, dtype=np.float32)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    refined_faces = np.asarray(refined_mesh.faces, dtype=np.int64)
    same_vertex_count = int(source_vertices.shape[0]) == int(refined_vertices.shape[0])
    same_face_count = int(source_faces.shape[0]) == int(refined_faces.shape[0])
    same_faces = same_face_count and np.array_equal(source_faces, refined_faces)
    if not same_vertex_count:
        raise ValueError(
            f"Cannot compare vertex-wise: source vertices={source_vertices.shape[0]}, refined={refined_vertices.shape[0]}"
        )

    delta_vec = refined_vertices - source_vertices
    delta_norm = np.linalg.norm(delta_vec, axis=1).astype(np.float32, copy=False)
    bounds = np.asarray(source_mesh.bounds, dtype=np.float32)
    bbox_extent = bounds[1] - bounds[0]
    bbox_diag = float(max(np.linalg.norm(bbox_extent), 1e-8))
    edge_lengths = sample_edge_lengths(source_vertices, source_faces, int(args.edge_sample_faces), int(args.seed))
    edge_median = float(np.median(edge_lengths)) if edge_lengths.size > 0 else 0.0
    edge_p95 = float(np.percentile(edge_lengths, 95.0)) if edge_lengths.size > 0 else 0.0

    payload = load_offset_payload(offset_payload_path, int(source_vertices.shape[0]))
    active_ids = np.asarray(payload["active_vertex_ids"], dtype=np.int64)
    active_mask = np.zeros((source_vertices.shape[0],), dtype=bool)
    active_mask[active_ids] = True
    inactive_mask = ~active_mask
    changed_mask = delta_norm > 1e-8
    payload_delta = np.asarray(payload["active_vertex_delta"], dtype=np.float32)
    payload_normals = normalize_np(np.asarray(payload["active_vertex_normals"], dtype=np.float32))
    if bool(payload["loaded"]) and active_ids.size == payload_delta.size and payload_normals.shape[0] == active_ids.shape[0]:
        signed_normal_active = np.sum(delta_vec[active_ids] * payload_normals, axis=1).astype(np.float32, copy=False)
        tangent_active = np.sqrt(np.maximum(delta_norm[active_ids] ** 2 - signed_normal_active**2, 0.0)).astype(
            np.float32,
            copy=False,
        )
        payload_delta_error = signed_normal_active - payload_delta
    else:
        signed_normal_active = np.zeros((0,), dtype=np.float32)
        tangent_active = np.zeros((0,), dtype=np.float32)
        payload_delta_error = np.zeros((0,), dtype=np.float32)
    if bool(args.compute_full_vertex_normals):
        source_normals = normalize_np(np.asarray(source_mesh.vertex_normals, dtype=np.float32))
        signed_normal_delta = np.sum(delta_vec * source_normals, axis=1).astype(np.float32, copy=False)
        tangent_delta = np.sqrt(np.maximum(delta_norm * delta_norm - signed_normal_delta * signed_normal_delta, 0.0)).astype(
            np.float32,
            copy=False,
        )
    else:
        signed_normal_delta = np.zeros((0,), dtype=np.float32)
        tangent_delta = np.zeros((0,), dtype=np.float32)

    debug_root = output_root / "debug_pointclouds"
    export_point_cloud(
        debug_root / "all_vertices_displacement_heat.ply",
        refined_vertices,
        delta_norm,
        int(args.debug_point_cap),
        int(args.seed),
        float(args.debug_vmax),
    )
    export_point_cloud(
        debug_root / "active_vertices_displacement_heat.ply",
        refined_vertices[active_mask],
        delta_norm[active_mask],
        int(args.debug_point_cap),
        int(args.seed),
        float(args.debug_vmax),
    )
    if np.any(changed_mask):
        top_count = min(int(args.debug_point_cap), int(np.count_nonzero(changed_mask)))
        top_ids = np.argsort(-delta_norm, kind="stable")[:top_count]
        export_point_cloud(
            debug_root / "top_displacement_vertices.ply",
            refined_vertices[top_ids],
            delta_norm[top_ids],
            -1,
            int(args.seed),
            float(args.debug_vmax),
        )

    thresholds_world = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1.5e-2, 1.9e-2]
    summary = {
        "version": "score_mesh_refine_delta_v0",
        "source_mesh_path": str(source_mesh_path),
        "refined_mesh_path": str(refined_mesh_path),
        "offset_payload_path": str(offset_payload_path) if offset_payload_path is not None else None,
        "output": {
            "summary": str(output_root / "mesh_refine_delta_v0_summary.json"),
            "debug_pointclouds": str(debug_root),
        },
        "topology": {
            "same_vertex_count": bool(same_vertex_count),
            "same_face_count": bool(same_face_count),
            "same_faces": bool(same_faces),
            "vertex_count": int(source_vertices.shape[0]),
            "face_count": int(source_faces.shape[0]),
        },
        "scale": {
            "bbox_min": bounds[0].tolist(),
            "bbox_max": bounds[1].tolist(),
            "bbox_extent": bbox_extent.tolist(),
            "bbox_diag": bbox_diag,
            "edge_sample_faces": int(min(max(int(args.edge_sample_faces), 0), int(source_faces.shape[0]))),
            "edge_length_sample": stats_from_array(edge_lengths),
            "compute_full_vertex_normals": bool(args.compute_full_vertex_normals),
        },
        "counts": {
            "active_vertices": int(np.count_nonzero(active_mask)),
            "active_vertex_ratio": float(np.mean(active_mask.astype(np.float32))),
            "changed_vertices_eps1e-8": int(np.count_nonzero(changed_mask)),
            "changed_vertex_ratio_eps1e-8": float(np.mean(changed_mask.astype(np.float32))),
        },
        "delta_world": {
            "norm_all": stats_from_array(delta_norm),
            "norm_active": stats_from_array(delta_norm[active_mask]),
            "norm_inactive": stats_from_array(delta_norm[inactive_mask]),
            "signed_normal_all": stats_from_array(signed_normal_delta),
            "signed_normal_active_from_payload_normals": stats_from_array(signed_normal_active),
            "tangent_all": stats_from_array(tangent_delta),
            "tangent_active_from_payload_normals": stats_from_array(tangent_active),
            "threshold_counts": threshold_counts(delta_norm, thresholds_world),
        },
        "delta_normalized": {
            "norm_over_bbox_diag": stats_from_array(delta_norm / bbox_diag),
            "norm_active_over_bbox_diag": stats_from_array(delta_norm[active_mask] / bbox_diag),
            "norm_over_edge_median": stats_from_array(delta_norm / max(edge_median, 1e-8)) if edge_median > 0.0 else {"count": 0},
            "norm_active_over_edge_median": stats_from_array(delta_norm[active_mask] / max(edge_median, 1e-8))
            if edge_median > 0.0
            else {"count": 0},
            "norm_over_edge_p95": stats_from_array(delta_norm / max(edge_p95, 1e-8)) if edge_p95 > 0.0 else {"count": 0},
        },
        "payload_check": {
            "loaded": bool(payload["loaded"]),
            "active_vertex_ids": int(active_ids.size),
            "payload_delta": stats_from_array(payload_delta),
            "signed_normal_minus_payload_delta": stats_from_array(payload_delta_error),
        },
    }
    summary_path = output_root / "mesh_refine_delta_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[mesh-refine-delta-v0] summary : {summary_path}")
    print(f"[mesh-refine-delta-v0] debug   : {debug_root}")


if __name__ == "__main__":
    main()
