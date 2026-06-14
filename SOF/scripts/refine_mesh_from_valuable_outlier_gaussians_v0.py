#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from classify_gaussian_surface_state_v0 import (  # noqa: E402
    load_gaussian_ply_full,
    load_triangle_mesh,
    mesh_edge_lengths,
    normalize_np,
    query_surface,
    stats_from_array,
)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_payload(path: Path) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to load .pt surface-state payloads.")
    return torch.load(str(path), map_location="cpu")


def payload_mask(payload: dict[str, Any], keys: list[str], total: int, *, default: bool = False) -> np.ndarray:
    if not keys:
        return np.full((total,), bool(default), dtype=bool)
    out = np.zeros((total,), dtype=bool)
    found = False
    for key in keys:
        if not key:
            continue
        if key not in payload:
            continue
        arr = tensor_to_numpy(payload[key]).astype(bool, copy=False).reshape(-1)
        if arr.shape[0] != total:
            raise ValueError(f"Payload mask '{key}' length mismatch: {arr.shape[0]} vs {total}")
        out |= arr
        found = True
    if not found:
        raise KeyError(f"None of the requested payload mask keys were found: {keys}")
    return out


def split_keys(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def robust_score_linear(value: np.ndarray, low: float, high: float, invert: bool = False) -> np.ndarray:
    denom = max(float(high) - float(low), 1e-8)
    score = np.clip((value.astype(np.float32, copy=False) - float(low)) / denom, 0.0, 1.0)
    if invert:
        score = 1.0 - score
    return score.astype(np.float32, copy=False)


def write_point_cloud(path: Path, xyz: np.ndarray, score: np.ndarray, max_points: int) -> None:
    if xyz.shape[0] == 0:
        return
    ids = np.arange(xyz.shape[0], dtype=np.int64)
    if int(max_points) > 0 and ids.size > int(max_points):
        ids = np.argsort(-score, kind="stable")[: int(max_points)]
    value = score[ids].astype(np.float32, copy=False)
    denom = max(float(np.percentile(value, 99)), 1e-6)
    heat = np.clip(value / denom, 0.0, 1.0)
    colors = np.stack(
        [
            255.0 * heat,
            180.0 * (1.0 - np.abs(heat - 0.5) * 2.0),
            255.0 * (1.0 - heat),
            np.full_like(heat, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(xyz[ids], colors=colors).export(path)


def build_vertex_neighbor_edges(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    undirected = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    undirected = undirected[(undirected[:, 0] >= 0) & (undirected[:, 1] >= 0)]
    undirected = undirected[(undirected[:, 0] < vertex_count) & (undirected[:, 1] < vertex_count)]
    directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    return directed[:, 0].astype(np.int64, copy=False), directed[:, 1].astype(np.int64, copy=False)


def smooth_offsets(
    offsets: np.ndarray,
    data_offsets: np.ndarray,
    data_weight: np.ndarray,
    faces: np.ndarray,
    vertex_count: int,
    iterations: int,
    smooth_lambda: float,
    data_anchor: float,
    clamp_abs: float,
) -> np.ndarray:
    src, dst = build_vertex_neighbor_edges(faces, vertex_count)
    current = offsets.astype(np.float32, copy=True)
    has_data = data_weight > 0.0
    lam = np.clip(float(smooth_lambda), 0.0, 1.0)
    anchor = np.clip(float(data_anchor), 0.0, 1.0)
    for _ in range(max(int(iterations), 0)):
        neighbor_sum = np.zeros_like(current)
        neighbor_count = np.zeros_like(current)
        if src.size > 0:
            np.add.at(neighbor_sum, src, current[dst])
            np.add.at(neighbor_count, src, 1.0)
        neighbor_mean = np.divide(
            neighbor_sum,
            np.maximum(neighbor_count, 1.0),
            out=np.zeros_like(current),
            where=neighbor_count > 0.0,
        )
        smoothed = (1.0 - lam) * current + lam * neighbor_mean
        current = np.where(has_data, anchor * data_offsets + (1.0 - anchor) * smoothed, smoothed)
        if float(clamp_abs) > 0.0:
            current = np.clip(current, -float(clamp_abs), float(clamp_abs))
    return current.astype(np.float32, copy=False)


def export_offset_mesh(
    source_mesh: trimesh.Trimesh,
    offsets: np.ndarray,
    vertex_normals: np.ndarray,
    refined_vertices: np.ndarray,
    path: Path,
) -> None:
    mesh = trimesh.Trimesh(
        vertices=refined_vertices.astype(np.float64, copy=False),
        faces=np.asarray(source_mesh.faces, dtype=np.int64).copy(),
        process=False,
    )
    value = np.abs(offsets.astype(np.float32, copy=False))
    denom = max(float(np.percentile(value[value > 0.0], 99)) if np.any(value > 0.0) else 0.0, 1e-6)
    heat = np.clip(value / denom, 0.0, 1.0)
    colors = np.stack(
        [
            255.0 * heat,
            220.0 * (1.0 - heat),
            255.0 * (1.0 - heat),
            np.full_like(heat, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refine an initialization mesh by letting valuable outlier Gaussians propose a small, "
            "normal-only mesh displacement field. This is an offline diagnostic/refinement stage; "
            "it does not modify the Gaussian field."
        )
    )
    parser.add_argument("--point_cloud_ply", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--surface_state_payload", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_keys", default="near_surface_uncertain,off_surface_near_mesh")
    parser.add_argument("--exclude_keys", default="surface_carrier,low_opacity_neutral,no_mesh_neutral")
    parser.add_argument("--surface_query_mode", choices=["auto", "exact_open3d", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=1_000_000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--opacity_high", type=float, default=0.35)
    parser.add_argument("--thin_ratio_good", type=float, default=0.12)
    parser.add_argument("--thin_ratio_bad", type=float, default=0.45)
    parser.add_argument("--min_anisotropy", type=float, default=2.0)
    parser.add_argument("--anisotropy_high", type=float, default=8.0)
    parser.add_argument("--min_normal_fraction", type=float, default=0.45)
    parser.add_argument("--min_abs_normal_offset", type=float, default=0.001)
    parser.add_argument("--max_abs_normal_offset", type=float, default=0.035)
    parser.add_argument("--max_offset_scale_ratio", type=float, default=1.5)
    parser.add_argument("--min_value_score", type=float, default=0.04)
    parser.add_argument("--huber_delta", type=float, default=0.012)
    parser.add_argument("--max_vertex_offset", type=float, default=0.015)
    parser.add_argument("--max_vertex_offset_edge_ratio", type=float, default=0.25)
    parser.add_argument("--normal_offset_gain", type=float, default=0.65)
    parser.add_argument("--smooth_iterations", type=int, default=8)
    parser.add_argument("--smooth_lambda", type=float, default=0.45)
    parser.add_argument("--data_anchor", type=float, default=0.70)
    parser.add_argument("--preview_max_points", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    point_cloud_path = Path(args.point_cloud_ply).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    payload_path = Path(args.surface_state_payload).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gs = load_gaussian_ply_full(point_cloud_path)
    mesh_obj = load_triangle_mesh(mesh_path)
    payload = load_payload(payload_path)

    xyz = gs["xyz"].astype(np.float32, copy=False)
    total = int(xyz.shape[0])
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Input mesh must contain vertices and faces.")

    candidate_keys = split_keys(args.candidate_keys)
    exclude_keys = split_keys(args.exclude_keys)
    candidate_mask = payload_mask(payload, candidate_keys, total, default=False)
    if exclude_keys:
        exclude_mask = payload_mask(payload, exclude_keys, total, default=False)
        candidate_mask &= ~exclude_mask

    surface = query_surface(
        mesh_obj=mesh_obj,
        points_xyz=xyz,
        mode=str(args.surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )
    nearest_point = surface["nearest_surface_point"].astype(np.float32, copy=False)
    nearest_normal = normalize_np(surface["nearest_surface_normal"].astype(np.float32, copy=False))
    nearest_face_id = surface["nearest_face_id"].astype(np.int64, copy=False)
    delta = xyz - nearest_point
    signed_normal_offset = np.sum(delta * nearest_normal, axis=1).astype(np.float32, copy=False)
    abs_normal_offset = np.abs(signed_normal_offset).astype(np.float32, copy=False)
    surface_distance = surface["surface_distance"].astype(np.float32, copy=False)
    normal_fraction = (
        abs_normal_offset / np.maximum(surface_distance, 1e-8)
    ).astype(np.float32, copy=False)

    scale = gs["effective_scale"].astype(np.float32, copy=False)
    scale_major = gs["scale_major"].astype(np.float32, copy=False)
    scale_minor = gs["scale_minor"].astype(np.float32, copy=False)
    scale_mid = gs["scale_mid"].astype(np.float32, copy=False)
    thin_ratio = (scale_minor / np.maximum(scale_mid, 1e-8)).astype(np.float32, copy=False)
    anisotropy = gs["effective_anisotropy"].astype(np.float32, copy=False)
    opacity = gs["opacity"].astype(np.float32, copy=False)

    opacity_score = robust_score_linear(opacity, float(args.min_opacity), float(args.opacity_high))
    thin_score = robust_score_linear(
        thin_ratio,
        float(args.thin_ratio_good),
        float(args.thin_ratio_bad),
        invert=True,
    )
    anisotropy_score = robust_score_linear(anisotropy, float(args.min_anisotropy), float(args.anisotropy_high))
    normal_fraction_score = robust_score_linear(
        normal_fraction,
        float(args.min_normal_fraction),
        1.0,
    )
    offset_window_score = robust_score_linear(
        abs_normal_offset,
        float(args.min_abs_normal_offset),
        float(args.max_abs_normal_offset),
    ) * robust_score_linear(
        abs_normal_offset,
        0.0,
        float(args.max_abs_normal_offset),
        invert=True,
    )

    max_candidate_offset = np.minimum(
        float(args.max_abs_normal_offset),
        float(args.max_offset_scale_ratio) * np.maximum(scale_major, 1e-8),
    ).astype(np.float32, copy=False)
    offset_gate = (
        (abs_normal_offset >= float(args.min_abs_normal_offset))
        & (abs_normal_offset <= max_candidate_offset)
    )

    value_score = (
        opacity_score
        * np.maximum(thin_score, 0.15 * anisotropy_score)
        * np.maximum(normal_fraction_score, 0.05)
        * np.maximum(offset_window_score, 0.05)
    ).astype(np.float32, copy=False)
    valuable_mask = (
        candidate_mask
        & offset_gate
        & (opacity >= float(args.min_opacity))
        & (anisotropy >= float(args.min_anisotropy))
        & (normal_fraction >= float(args.min_normal_fraction))
        & (value_score >= float(args.min_value_score))
    )
    valuable_ids = np.flatnonzero(valuable_mask).astype(np.int64, copy=False)

    median_edge = float(np.median(mesh_edge_lengths(mesh_obj))) if faces.size > 0 else 0.0
    max_vertex_offset = max(
        float(args.max_vertex_offset),
        float(args.max_vertex_offset_edge_ratio) * max(median_edge, 0.0),
    )
    huber_delta = max(float(args.huber_delta), 1e-8)
    raw_target_offset = signed_normal_offset.astype(np.float32, copy=False)
    robust_target_offset = (
        np.sign(raw_target_offset)
        * np.minimum(np.abs(raw_target_offset), huber_delta)
        * float(args.normal_offset_gain)
    ).astype(np.float32, copy=False)
    robust_target_offset = np.clip(
        robust_target_offset,
        -float(max_vertex_offset),
        float(max_vertex_offset),
    ).astype(np.float32, copy=False)

    vertex_weight = np.zeros((vertices.shape[0],), dtype=np.float32)
    vertex_offset_accum = np.zeros((vertices.shape[0],), dtype=np.float32)
    face_ids = np.clip(nearest_face_id[valuable_ids], 0, faces.shape[0] - 1)
    face_vertices = faces[face_ids]
    weights = value_score[valuable_ids].astype(np.float32, copy=False)
    offsets = robust_target_offset[valuable_ids].astype(np.float32, copy=False)
    for corner in range(3):
        vids = face_vertices[:, corner]
        np.add.at(vertex_weight, vids, weights)
        np.add.at(vertex_offset_accum, vids, weights * offsets)
    vertex_data_offset = np.divide(
        vertex_offset_accum,
        np.maximum(vertex_weight, 1e-8),
        out=np.zeros_like(vertex_offset_accum),
        where=vertex_weight > 0.0,
    ).astype(np.float32, copy=False)
    vertex_offset = smooth_offsets(
        offsets=vertex_data_offset,
        data_offsets=vertex_data_offset,
        data_weight=vertex_weight,
        faces=faces,
        vertex_count=vertices.shape[0],
        iterations=int(args.smooth_iterations),
        smooth_lambda=float(args.smooth_lambda),
        data_anchor=float(args.data_anchor),
        clamp_abs=float(max_vertex_offset),
    )
    vertex_normals = normalize_np(np.asarray(mesh_obj.vertex_normals, dtype=np.float32))
    refined_vertices = (vertices + vertex_offset[:, None] * vertex_normals).astype(np.float32, copy=False)

    refined_mesh_path = output_dir / "refined_mesh_from_outlier_gs_v0.ply"
    displacement_mesh_path = output_dir / "refined_mesh_offset_heat_v0.ply"
    valuable_preview_path = output_dir / "valuable_outlier_gaussians_v0.ply"
    moved_vertex_preview_path = output_dir / "moved_mesh_vertices_v0.ply"
    export_offset_mesh(mesh_obj, vertex_offset, vertex_normals, refined_vertices, refined_mesh_path)
    export_offset_mesh(mesh_obj, vertex_offset, vertex_normals, refined_vertices, displacement_mesh_path)
    write_point_cloud(valuable_preview_path, xyz[valuable_ids], value_score[valuable_ids], int(args.preview_max_points))
    moved_vertex_mask = np.abs(vertex_offset) > 1e-8
    write_point_cloud(
        moved_vertex_preview_path,
        refined_vertices[moved_vertex_mask],
        np.abs(vertex_offset[moved_vertex_mask]),
        int(args.preview_max_points),
    )

    if torch is not None:
        evidence_payload = {
            "version": "valuable_outlier_mesh_refine_v0",
            "point_cloud_ply": str(point_cloud_path),
            "source_mesh_path": str(mesh_path),
            "refined_mesh_path": str(refined_mesh_path),
            "surface_state_payload": str(payload_path),
            "candidate_mask": torch.from_numpy(candidate_mask.copy()),
            "valuable_mask": torch.from_numpy(valuable_mask.copy()),
            "valuable_ids": torch.from_numpy(valuable_ids.copy()),
            "value_score": torch.from_numpy(value_score.copy()),
            "signed_normal_offset": torch.from_numpy(signed_normal_offset.copy()),
            "surface_distance": torch.from_numpy(surface_distance.copy()),
            "normal_fraction": torch.from_numpy(normal_fraction.copy()),
            "nearest_face_id": torch.from_numpy(nearest_face_id.copy()),
            "nearest_surface_point": torch.from_numpy(nearest_point.copy()),
            "nearest_surface_normal": torch.from_numpy(nearest_normal.copy()),
            "vertex_offset": torch.from_numpy(vertex_offset.copy()),
            "vertex_weight": torch.from_numpy(vertex_weight.copy()),
            "vertex_data_offset": torch.from_numpy(vertex_data_offset.copy()),
            "vertex_normals": torch.from_numpy(vertex_normals.copy()),
            "refined_vertices": torch.from_numpy(refined_vertices.copy()),
        }
        evidence_payload_path = output_dir / "valuable_outlier_mesh_refine_v0.pt"
        torch.save(evidence_payload, str(evidence_payload_path))
    else:
        evidence_payload_path = output_dir / "valuable_outlier_mesh_refine_v0.npz"
        np.savez_compressed(
            evidence_payload_path,
            candidate_mask=candidate_mask,
            valuable_mask=valuable_mask,
            valuable_ids=valuable_ids,
            value_score=value_score,
            signed_normal_offset=signed_normal_offset,
            surface_distance=surface_distance,
            normal_fraction=normal_fraction,
            nearest_face_id=nearest_face_id,
            vertex_offset=vertex_offset,
            vertex_weight=vertex_weight,
            vertex_data_offset=vertex_data_offset,
            vertex_normals=vertex_normals,
            refined_vertices=refined_vertices,
        )

    summary = {
        "version": "valuable_outlier_mesh_refine_v0",
        "point_cloud_ply": str(point_cloud_path),
        "source_mesh_path": str(mesh_path),
        "surface_state_payload": str(payload_path),
        "output_dir": str(output_dir),
        "paths": {
            "refined_mesh": str(refined_mesh_path),
            "offset_heat_mesh": str(displacement_mesh_path),
            "valuable_gaussians_preview": str(valuable_preview_path),
            "moved_vertices_preview": str(moved_vertex_preview_path),
            "payload": str(evidence_payload_path),
            "summary": str(output_dir / "valuable_outlier_mesh_refine_v0_summary.json"),
        },
        "surface_query": {
            "requested": str(args.surface_query_mode),
            "used": str(surface.get("surface_query_mode_used", args.surface_query_mode)),
        },
        "parameters": vars(args),
        "counts": {
            "gaussian_count": total,
            "mesh_vertices": int(vertices.shape[0]),
            "mesh_faces": int(faces.shape[0]),
            "candidate_count": int(np.sum(candidate_mask)),
            "valuable_count": int(valuable_ids.shape[0]),
            "valuable_ratio_vs_candidates": float(valuable_ids.shape[0] / max(int(np.sum(candidate_mask)), 1)),
            "moved_vertex_count": int(np.sum(moved_vertex_mask)),
            "moved_vertex_ratio": float(np.sum(moved_vertex_mask) / max(vertices.shape[0], 1)),
        },
        "stats": {
            "value_score_candidates": stats_from_array(value_score[candidate_mask]),
            "value_score_valuable": stats_from_array(value_score[valuable_mask]),
            "opacity_valuable": stats_from_array(opacity[valuable_mask]),
            "thin_ratio_valuable": stats_from_array(thin_ratio[valuable_mask]),
            "anisotropy_valuable": stats_from_array(anisotropy[valuable_mask]),
            "surface_distance_valuable": stats_from_array(surface_distance[valuable_mask]),
            "signed_normal_offset_valuable": stats_from_array(signed_normal_offset[valuable_mask]),
            "normal_fraction_valuable": stats_from_array(normal_fraction[valuable_mask]),
            "vertex_data_offset": stats_from_array(vertex_data_offset[vertex_weight > 0.0]),
            "vertex_offset_final": stats_from_array(vertex_offset[moved_vertex_mask]),
            "vertex_weight": stats_from_array(vertex_weight[vertex_weight > 0.0]),
        },
        "notes": [
            "Only payload-selected non-surface candidates can pull the mesh.",
            "Pulls are normal-only at mesh vertices, robust-clipped, and smoothed over mesh adjacency.",
            "This script changes the mesh, not the Gaussian field or old surface-state labels.",
            "Re-run surface-state classification against the refined mesh before judging layer redistribution.",
        ],
    }
    summary_path = output_dir / "valuable_outlier_mesh_refine_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] refined mesh: {refined_mesh_path}")
    print(f"[done] payload     : {evidence_payload_path}")
    print(f"[done] summary     : {summary_path}")


if __name__ == "__main__":
    main()
