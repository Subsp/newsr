from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from score_mesh_delta_star_gaussians_v0 import (
    estimate_pca_normals_chunked,
    gaussian_features,
    linspace_sample_count,
    query_tree_chunked,
    read_vertex_table,
    save_payload,
    stats,
    write_point_cloud_ply,
    xyz_from_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Gaussians using one SOF mesh surface and long-axis tangent alignment."
    )
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--point_cloud_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_mesh_reference_vertices", type=int, default=2_000_000)
    parser.add_argument("--query_chunk_size", type=int, default=262144)
    parser.add_argument("--gaussian_distance_mode", choices=["center", "major_endpoints"], default="major_endpoints")
    parser.add_argument("--major_endpoint_scale", type=float, default=1.0)
    parser.add_argument("--endpoint_distance_reduce", choices=["min", "max"], default="min")
    parser.add_argument("--radius_mode", choices=["absolute", "scale_max"], default="absolute")
    parser.add_argument("--radius_abs", type=float, default=0.08)
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--tangent_angle_k", type=int, default=12)
    parser.add_argument("--tangent_angle_distance_floor", type=float, default=0.15)
    parser.add_argument("--selection_mode", choices=["surface_tangent", "off_surface"], default="surface_tangent")
    parser.add_argument("--min_surface_distance", type=float, default=0.04)
    parser.add_argument("--min_tangent_angle_to_tangent", type=float, default=0.35)
    parser.add_argument("--min_opacity", type=float, default=0.0001)
    parser.add_argument("--min_anisotropy", type=float, default=1.0)
    parser.add_argument("--max_scale_major", type=float, default=0.0)
    parser.add_argument("--min_dc_luma", type=float, default=-999999.0)
    parser.add_argument("--dc_luma_quantile", type=float, default=0.0)
    parser.add_argument("--min_candidate_score", type=float, default=0.00005)
    parser.add_argument("--candidate_score_quantile", type=float, default=50.0)
    parser.add_argument("--max_candidates", type=int, default=800_000)
    parser.add_argument("--preview_max_points", type=int, default=200_000)
    args = parser.parse_args()

    mesh_path = Path(args.mesh_path).expanduser().resolve()
    point_cloud_path = Path(args.point_cloud_ply).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_table, mesh_header = read_vertex_table(mesh_path)
    mesh_xyz = xyz_from_table(mesh_table)
    mesh_ids = linspace_sample_count(mesh_xyz.shape[0], int(args.max_mesh_reference_vertices))
    mesh_reference = np.asarray(mesh_xyz[mesh_ids], dtype=np.float32)
    mesh_tree = cKDTree(mesh_reference)
    print(f"[single-mesh-tangent-v0] mesh reference vertices: {mesh_reference.shape[0]}/{mesh_xyz.shape[0]}")

    gs_table, _ = read_vertex_table(point_cloud_path)
    gs = gaussian_features(gs_table)
    gs_xyz = gs["xyz"]
    print(f"[single-mesh-tangent-v0] gaussian count: {gs_xyz.shape[0]}")

    center_dist_to_surface, center_nearest_surface = query_tree_chunked(
        mesh_tree, gs_xyz, chunk_size=int(args.query_chunk_size)
    )
    if str(args.gaussian_distance_mode) == "major_endpoints":
        endpoint_offset = gs["major_axis_dir"] * (gs["scale_major"] * float(args.major_endpoint_scale))[:, None]
        endpoint_pos = gs_xyz + endpoint_offset
        endpoint_neg = gs_xyz - endpoint_offset
        dist_pos, nearest_pos = query_tree_chunked(mesh_tree, endpoint_pos, chunk_size=int(args.query_chunk_size))
        dist_neg, nearest_neg = query_tree_chunked(mesh_tree, endpoint_neg, chunk_size=int(args.query_chunk_size))
        if str(args.endpoint_distance_reduce) == "max":
            use_pos = dist_pos >= dist_neg
        else:
            use_pos = dist_pos <= dist_neg
        dist_to_surface = np.where(use_pos, dist_pos, dist_neg).astype(np.float32, copy=False)
        nearest_surface = np.where(use_pos, nearest_pos, nearest_neg).astype(np.int64, copy=False)
        distance_query_points = np.where(use_pos[:, None], endpoint_pos, endpoint_neg).astype(np.float32, copy=False)
    else:
        dist_to_surface = center_dist_to_surface
        nearest_surface = center_nearest_surface
        distance_query_points = gs_xyz

    normals = estimate_pca_normals_chunked(
        mesh_tree,
        mesh_reference,
        distance_query_points,
        k=int(args.tangent_angle_k),
        chunk_size=int(args.query_chunk_size),
    )
    axis_dot_normal = np.abs(np.sum(gs["major_axis_dir"] * normals, axis=1)).clip(0.0, 1.0)
    tangent_angle_to_tangent = (np.arcsin(axis_dot_normal) / (0.5 * math.pi)).astype(np.float32)
    tangent_strength = (1.0 - tangent_angle_to_tangent).astype(np.float32)
    floor = float(np.clip(args.tangent_angle_distance_floor, 0.0, 1.0))
    tangent_distance_factor = (floor + (1.0 - floor) * tangent_angle_to_tangent).astype(np.float32)
    effective_dist_to_surface = (dist_to_surface * tangent_distance_factor).astype(np.float32)

    if str(args.radius_mode) == "absolute":
        radius = np.full_like(gs["scale_major"], float(args.radius_abs), dtype=np.float32)
    else:
        radius = np.maximum(float(args.radius_abs), gs["scale_major"] * float(args.radius_scale)).astype(np.float32)

    proximity = np.exp(-effective_dist_to_surface / np.maximum(radius, 1e-6)).astype(np.float32)
    surface_outlier_strength = (1.0 - np.exp(-dist_to_surface / np.maximum(radius, 1e-6))).astype(np.float32)
    normal_alignment_strength = tangent_angle_to_tangent
    aniso_strength = np.log1p(np.maximum(gs["effective_anisotropy"] - 1.0, 0.0)).astype(np.float32)
    opacity_strength = np.sqrt(np.clip(gs["opacity"], 0.0, 1.0)).astype(np.float32)
    max_scale_mask = (
        np.ones((gs_xyz.shape[0],), dtype=bool)
        if float(args.max_scale_major) <= 0.0
        else gs["scale_major"] <= float(args.max_scale_major)
    )
    if str(args.selection_mode) == "off_surface":
        candidate_score = (
            surface_outlier_strength
            * (0.25 + normal_alignment_strength)
            * (0.35 + aniso_strength)
            * opacity_strength
        ).astype(np.float32)
        geometry_mask = (
            (dist_to_surface >= float(args.min_surface_distance))
            & (tangent_angle_to_tangent >= float(args.min_tangent_angle_to_tangent))
            & (gs["opacity"] >= float(args.min_opacity))
            & (gs["effective_anisotropy"] >= float(args.min_anisotropy))
            & max_scale_mask
            & (candidate_score >= float(args.min_candidate_score))
        )
    else:
        candidate_score = (
            proximity
            * (0.35 + aniso_strength)
            * opacity_strength
            * (0.5 + tangent_strength)
        ).astype(np.float32)
        geometry_mask = (
            (effective_dist_to_surface <= radius)
            & (gs["opacity"] >= float(args.min_opacity))
            & (gs["effective_anisotropy"] >= float(args.min_anisotropy))
            & max_scale_mask
            & (candidate_score >= float(args.min_candidate_score))
        )

    luma_threshold = float(args.min_dc_luma)
    dc_luma_quantile = float(np.clip(float(args.dc_luma_quantile), 0.0, 100.0))
    if dc_luma_quantile > 0.0 and np.any(geometry_mask):
        quantile_threshold = float(
            np.percentile(gs["dc_luma"][geometry_mask], dc_luma_quantile)
        )
        luma_threshold = max(luma_threshold, quantile_threshold)
    brightness_mask = gs["dc_luma"] >= luma_threshold
    base_mask = geometry_mask & brightness_mask

    if np.any(base_mask):
        score_threshold = max(
            float(np.percentile(candidate_score[base_mask], float(args.candidate_score_quantile))),
            float(args.min_candidate_score),
        )
    else:
        score_threshold = float("inf")
    base_candidate_ids = np.flatnonzero(base_mask).astype(np.int64, copy=False)
    candidate_ids = np.flatnonzero(base_mask & (candidate_score >= score_threshold)).astype(np.int64, copy=False)
    if int(args.max_candidates) > 0 and candidate_ids.size > int(args.max_candidates):
        keep = np.argsort(-candidate_score[candidate_ids], kind="stable")[: int(args.max_candidates)]
        candidate_ids = candidate_ids[keep].astype(np.int64, copy=False)
    candidate_mask = np.zeros((gs_xyz.shape[0],), dtype=bool)
    candidate_mask[candidate_ids] = True

    payload_path = output_dir / "mesh_delta_star_gaussian_candidates_v0.pt"
    preview_path = output_dir / "mesh_delta_star_gaussian_candidates_v0.ply"
    surface_preview_path = output_dir / "single_mesh_surface_reference_v0.ply"
    saved_payload_path = save_payload(
        payload_path,
        {
            "candidate_mask": candidate_mask,
            "candidate_ids": candidate_ids,
            "base_candidate_mask": base_mask.astype(bool, copy=False),
            "base_candidate_ids": base_candidate_ids,
            "geometry_candidate_mask": geometry_mask.astype(bool, copy=False),
            "brightness_mask": brightness_mask.astype(bool, copy=False),
            "candidate_score": candidate_score,
            "dist_to_surface_mesh": dist_to_surface,
            "effective_dist_to_surface_mesh": effective_dist_to_surface,
            "center_dist_to_surface_mesh": center_dist_to_surface,
            "nearest_surface_index": nearest_surface,
            "tangent_angle_to_tangent": tangent_angle_to_tangent,
            "tangent_distance_factor": tangent_distance_factor,
            "surface_outlier_strength": surface_outlier_strength,
            "normal_alignment_strength": normal_alignment_strength,
            "delta_radius_used": radius,
            "opacity": gs["opacity"],
            "scale_major": gs["scale_major"],
            "scale_minor": gs["scale_minor"],
            "major_axis_idx": gs["major_axis_idx"],
            "major_axis_dir": gs["major_axis_dir"],
            "effective_anisotropy": gs["effective_anisotropy"],
            "dc_luma": gs["dc_luma"],
            "dc_luma_threshold_used": np.asarray([luma_threshold], dtype=np.float32),
            "rest_norm": gs["rest_norm"],
            "max_scale_major_mask": max_scale_mask.astype(bool, copy=False),
        },
    )
    write_point_cloud_ply(preview_path, gs_xyz[candidate_ids], candidate_score[candidate_ids], max_points=int(args.preview_max_points))
    write_point_cloud_ply(
        surface_preview_path,
        mesh_reference,
        np.ones((mesh_reference.shape[0],), dtype=np.float32),
        max_points=int(args.preview_max_points),
    )

    summary = {
        "mode": "single_mesh_tangent_gaussians_v0",
        "mesh_path": str(mesh_path),
        "point_cloud_ply": str(point_cloud_path),
        "output_dir": str(output_dir),
        "payload_path": saved_payload_path,
        "candidate_preview_ply": str(preview_path),
        "surface_preview_ply": str(surface_preview_path),
        "mesh": {
            "vertex_count": int(mesh_header["elements"]["vertex"]["count"]),
            "face_count": int(mesh_header["elements"].get("face", {}).get("count", 0)),
            "reference_vertex_count": int(mesh_reference.shape[0]),
            "bbox_min": np.min(mesh_reference, axis=0).astype(float).tolist(),
            "bbox_max": np.max(mesh_reference, axis=0).astype(float).tolist(),
        },
        "gaussians": {
            "gaussian_count": int(gs_xyz.shape[0]),
            "geometry_candidate_count": int(np.sum(geometry_mask)),
            "brightness_candidate_count": int(np.sum(brightness_mask)),
            "base_candidate_count": int(np.sum(base_mask)),
            "candidate_score_threshold": float(score_threshold) if np.isfinite(score_threshold) else None,
            "dc_luma_threshold_used": float(luma_threshold),
            "candidate_count": int(candidate_ids.size),
            "candidate_ratio": float(candidate_ids.size / max(gs_xyz.shape[0], 1)),
            "candidate_score_all_stats": stats(candidate_score),
            "candidate_score_selected_stats": stats(candidate_score[candidate_mask]),
            "dist_to_surface_selected_stats": stats(dist_to_surface[candidate_mask]),
            "effective_dist_to_surface_selected_stats": stats(effective_dist_to_surface[candidate_mask]),
            "center_dist_to_surface_selected_stats": stats(center_dist_to_surface[candidate_mask]),
            "tangent_angle_to_tangent_selected_stats": stats(tangent_angle_to_tangent[candidate_mask]),
            "tangent_distance_factor_selected_stats": stats(tangent_distance_factor[candidate_mask]),
            "surface_outlier_strength_selected_stats": stats(surface_outlier_strength[candidate_mask]),
            "normal_alignment_strength_selected_stats": stats(normal_alignment_strength[candidate_mask]),
            "delta_radius_selected_stats": stats(radius[candidate_mask]),
            "opacity_selected_stats": stats(gs["opacity"][candidate_mask]),
            "dc_luma_all_stats": stats(gs["dc_luma"]),
            "dc_luma_geometry_stats": stats(gs["dc_luma"][geometry_mask]),
            "dc_luma_selected_stats": stats(gs["dc_luma"][candidate_mask]),
            "anisotropy_selected_stats": stats(gs["effective_anisotropy"][candidate_mask]),
            "scale_major_selected_stats": stats(gs["scale_major"][candidate_mask]),
            "scale_minor_selected_stats": stats(gs["scale_minor"][candidate_mask]),
            "rest_norm_selected_stats": stats(gs["rest_norm"][candidate_mask]),
        },
        "parameters": {
            "max_mesh_reference_vertices": int(args.max_mesh_reference_vertices),
            "gaussian_distance_mode": str(args.gaussian_distance_mode),
            "major_endpoint_scale": float(args.major_endpoint_scale),
            "endpoint_distance_reduce": str(args.endpoint_distance_reduce),
            "radius_mode": str(args.radius_mode),
            "radius_abs": float(args.radius_abs),
            "radius_scale": float(args.radius_scale),
            "tangent_angle_k": int(args.tangent_angle_k),
            "tangent_angle_distance_floor": float(args.tangent_angle_distance_floor),
            "selection_mode": str(args.selection_mode),
            "min_surface_distance": float(args.min_surface_distance),
            "min_tangent_angle_to_tangent": float(args.min_tangent_angle_to_tangent),
            "min_opacity": float(args.min_opacity),
            "min_anisotropy": float(args.min_anisotropy),
            "max_scale_major": float(args.max_scale_major),
            "min_dc_luma": float(args.min_dc_luma),
            "dc_luma_quantile": dc_luma_quantile,
            "min_candidate_score": float(args.min_candidate_score),
            "candidate_score_quantile": float(args.candidate_score_quantile),
            "max_candidates": int(args.max_candidates),
        },
    }
    summary_path = output_dir / "mesh_delta_star_gaussian_candidates_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] payload          : {saved_payload_path}")
    print(f"[done] candidate preview: {preview_path}")
    print(f"[done] surface preview  : {surface_preview_path}")
    print(f"[done] summary          : {summary_path}")


if __name__ == "__main__":
    main()
