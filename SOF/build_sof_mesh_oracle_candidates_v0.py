from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from utils.sof_mesh_patch_enhancer_v0 import (
    load_triangle_mesh,
    mesh_bbox_normalizer,
    normalize_np,
    points_to_barycentric,
    save_payload_npz,
    stats_from_array,
    write_json,
)


def sample_lr_anchors(mesh: trimesh.Trimesh, count: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces[face_ids]]
    bary = points_to_barycentric(points.astype(np.float32), triangles.astype(np.float32))
    normals = normalize_np(np.asarray(mesh.face_normals, dtype=np.float32)[face_ids])
    return points.astype(np.float32), face_ids.astype(np.int64), bary.astype(np.float32), normals.astype(np.float32)


def sample_hr_reference(mesh: trimesh.Trimesh, count: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    normals = normalize_np(np.asarray(mesh.face_normals, dtype=np.float32)[face_ids])
    return points.astype(np.float32), normals.astype(np.float32)


def tangent_basis_from_normals(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normals = normalize_np(normals.astype(np.float32))
    axis_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    axis_y = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    base = np.tile(axis_x[None], (normals.shape[0], 1))
    parallel = np.abs(np.sum(base * normals, axis=1)) > 0.9
    base[parallel] = axis_y
    tangent_u = normalize_np(base - np.sum(base * normals, axis=1, keepdims=True) * normals)
    tangent_v = normalize_np(np.cross(normals, tangent_u))
    tangent_u = normalize_np(np.cross(tangent_v, normals))
    return tangent_u.astype(np.float32), tangent_v.astype(np.float32)


def colorize_distance(distance: np.ndarray, threshold: float) -> np.ndarray:
    t = np.clip(distance.reshape(-1) / max(float(threshold), 1e-8), 0.0, 1.0)
    colors = np.stack(
        [
            np.round(255.0 * t),
            np.round(230.0 * (1.0 - t)),
            np.round(80.0 * (1.0 - t)),
            np.full_like(t, 255.0),
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def maybe_limit_indices(score: np.ndarray, max_count: int) -> np.ndarray:
    ids = np.arange(score.shape[0], dtype=np.int64)
    if int(max_count) <= 0 or ids.size <= int(max_count):
        return ids
    # Keep the lowest score, where score is usually HR distance adjusted by normal agreement.
    return np.sort(np.argpartition(score, int(max_count) - 1)[: int(max_count)]).astype(np.int64, copy=False)


def build_payload(
    centers: np.ndarray,
    normals: np.ndarray,
    source_face_ids: np.ndarray,
    source_bary: np.ndarray,
    source_offsets: np.ndarray,
    confidence: np.ndarray,
    disagreement: np.ndarray,
    spacing: float,
    thickness_scale: float,
) -> Dict[str, np.ndarray]:
    tangent_u, tangent_v = tangent_basis_from_normals(normals)
    n = centers.shape[0]
    scale_u = np.full((n,), float(spacing), dtype=np.float32)
    scale_v = np.full((n,), float(spacing), dtype=np.float32)
    scale_n = np.full((n,), float(spacing) * float(thickness_scale), dtype=np.float32)
    return {
        "centers": centers.astype(np.float32),
        "normals": normalize_np(normals.astype(np.float32)),
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "scale_u": scale_u,
        "scale_v": scale_v,
        "scale_n": scale_n,
        "fused_rgb": np.zeros((n, 3), dtype=np.float32),
        "confidence": confidence.astype(np.float32),
        "disagreement": disagreement.astype(np.float32),
        "view_count": np.ones((n,), dtype=np.int32),
        "valid_mask": np.ones((n,), dtype=np.bool_),
        "source_face_ids": source_face_ids.astype(np.int64),
        "source_bary_coords": source_bary.astype(np.float32),
        "source_normal_offsets": source_offsets.astype(np.float32),
        # Compatibility aliases for code that expects mesh-bound payload keys.
        # These are source LR bindings, not final SR mesh face bindings.
        "face_ids": source_face_ids.astype(np.int64),
        "bary_coords": source_bary.astype(np.float32),
    }


def main():
    parser = ArgumentParser(
        description=(
            "Oracle staged SOF surface candidate experiment: sample around LRmesh, "
            "select HR-supported candidates, attract selected points to HR, and export sparse carrier candidates."
        )
    )
    parser.add_argument("--lr_mesh_path", type=str, required=True)
    parser.add_argument("--hr_mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--anchor_count", type=int, default=250000)
    parser.add_argument("--hr_reference_samples", type=int, default=500000)
    parser.add_argument("--offset_layers", type=int, default=9)
    parser.add_argument("--offset_radius_ratio", type=float, default=0.003)
    parser.add_argument("--select_distance_ratio", type=float, default=0.0015)
    parser.add_argument("--min_normal_alignment", type=float, default=-0.25)
    parser.add_argument("--selection_mode", choices=["best_per_anchor", "all_valid"], default="best_per_anchor")
    parser.add_argument("--attract_steps", type=int, default=3)
    parser.add_argument("--attract_alpha", type=float, default=0.85)
    parser.add_argument("--max_selected", type=int, default=200000)
    parser.add_argument("--carrier_spacing_ratio", type=float, default=0.001)
    parser.add_argument("--carrier_thickness_scale", type=float, default=0.08)
    parser.add_argument("--preview_all_max", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_mesh = load_triangle_mesh(args.lr_mesh_path)
    hr_mesh = load_triangle_mesh(args.hr_mesh_path)
    _, bbox_diag = mesh_bbox_normalizer(lr_mesh)
    offset_radius = float(args.offset_radius_ratio) * bbox_diag
    select_distance = float(args.select_distance_ratio) * bbox_diag
    carrier_spacing = float(args.carrier_spacing_ratio) * bbox_diag

    print(f"[oracle-candidates] LR mesh vertices={len(lr_mesh.vertices)} faces={len(lr_mesh.faces)}")
    print(f"[oracle-candidates] HR mesh vertices={len(hr_mesh.vertices)} faces={len(hr_mesh.faces)}")
    print(f"[oracle-candidates] bbox_diag={bbox_diag:.6f} offset_radius={offset_radius:.6f} select_distance={select_distance:.6f}")

    anchors, face_ids, bary, lr_normals = sample_lr_anchors(lr_mesh, int(args.anchor_count), int(args.seed))
    hr_points, hr_normals = sample_hr_reference(hr_mesh, int(args.hr_reference_samples), int(args.seed) + 17)
    hr_tree = cKDTree(hr_points)

    offsets = np.linspace(-offset_radius, offset_radius, int(args.offset_layers), dtype=np.float32)
    candidates = anchors[:, None, :] + offsets[None, :, None] * lr_normals[:, None, :]
    flat_candidates = candidates.reshape(-1, 3).astype(np.float32)
    flat_normals = np.repeat(lr_normals, int(args.offset_layers), axis=0).astype(np.float32)
    flat_anchor_ids = np.repeat(np.arange(anchors.shape[0], dtype=np.int64), int(args.offset_layers))
    flat_offsets = np.tile(offsets[None, :], (anchors.shape[0], 1)).reshape(-1).astype(np.float32)

    distances, nearest_ids = hr_tree.query(flat_candidates, k=1)
    distances = distances.astype(np.float32)
    nearest_points = hr_points[nearest_ids].astype(np.float32)
    nearest_normals = hr_normals[nearest_ids].astype(np.float32)
    normal_alignment = np.sum(flat_normals * nearest_normals, axis=1).astype(np.float32)
    valid = (distances <= select_distance) & (normal_alignment >= float(args.min_normal_alignment))

    if args.selection_mode == "best_per_anchor":
        score = distances + select_distance * np.clip(1.0 - normal_alignment, 0.0, 2.0)
        score[~valid] = np.inf
        score_2d = score.reshape(anchors.shape[0], int(args.offset_layers))
        best_layer = np.argmin(score_2d, axis=1)
        best_score = score_2d[np.arange(anchors.shape[0]), best_layer]
        anchor_keep = np.isfinite(best_score)
        selected_flat_ids = np.arange(anchors.shape[0], dtype=np.int64)[anchor_keep] * int(args.offset_layers) + best_layer[anchor_keep]
    else:
        selected_flat_ids = np.flatnonzero(valid).astype(np.int64, copy=False)

    if selected_flat_ids.size == 0:
        raise RuntimeError("No HR-supported candidates selected. Increase --select_distance_ratio or relax --min_normal_alignment.")

    rank_score = distances[selected_flat_ids] + select_distance * np.clip(1.0 - normal_alignment[selected_flat_ids], 0.0, 2.0)
    keep_local = maybe_limit_indices(rank_score, int(args.max_selected))
    selected_flat_ids = selected_flat_ids[keep_local]

    refined = flat_candidates[selected_flat_ids].copy()
    refined_normals = nearest_normals[selected_flat_ids].copy()
    refined_distance = distances[selected_flat_ids].copy()
    refined_nearest = nearest_points[selected_flat_ids].copy()
    for _ in range(max(int(args.attract_steps), 0)):
        refined += float(args.attract_alpha) * (refined_nearest - refined)
        refined_distance, refined_ids = hr_tree.query(refined, k=1)
        refined_distance = refined_distance.astype(np.float32)
        refined_nearest = hr_points[refined_ids].astype(np.float32)
        refined_normals = hr_normals[refined_ids].astype(np.float32)

    source_anchor_ids = flat_anchor_ids[selected_flat_ids]
    source_face_ids = face_ids[source_anchor_ids]
    source_bary = bary[source_anchor_ids]
    source_offsets = flat_offsets[selected_flat_ids]
    confidence = 1.0 - np.clip(refined_distance / max(select_distance, 1e-8), 0.0, 1.0)
    confidence *= np.clip((normal_alignment[selected_flat_ids] + 1.0) * 0.5, 0.0, 1.0)
    disagreement = np.clip(refined_distance / max(select_distance, 1e-8), 0.0, 1.0)

    all_preview_ids = np.arange(flat_candidates.shape[0], dtype=np.int64)
    if int(args.preview_all_max) > 0 and all_preview_ids.size > int(args.preview_all_max):
        rng = np.random.default_rng(int(args.seed))
        all_preview_ids = np.sort(rng.choice(all_preview_ids, size=int(args.preview_all_max), replace=False))
    trimesh.points.PointCloud(
        flat_candidates[all_preview_ids],
        colors=colorize_distance(distances[all_preview_ids], select_distance),
    ).export(output_dir / "all_lr_shell_candidates_preview_v0.ply")
    trimesh.points.PointCloud(
        refined,
        colors=colorize_distance(refined_distance, select_distance),
    ).export(output_dir / "selected_refined_hr_supported_candidates_v0.ply")

    payload = build_payload(
        centers=refined,
        normals=refined_normals,
        source_face_ids=source_face_ids,
        source_bary=source_bary,
        source_offsets=source_offsets,
        confidence=confidence,
        disagreement=disagreement,
        spacing=carrier_spacing,
        thickness_scale=float(args.carrier_thickness_scale),
    )
    payload_path = output_dir / "oracle_hr_supported_candidate_carrier_payload_v0.npz"
    save_payload_npz(payload_path, payload)

    np.savez_compressed(
        output_dir / "oracle_hr_supported_candidate_records_v0.npz",
        anchors=anchors,
        selected_centers=refined,
        selected_normals=refined_normals,
        selected_refined_distance=refined_distance,
        selected_source_anchor_ids=source_anchor_ids,
        selected_source_face_ids=source_face_ids,
        selected_source_bary_coords=source_bary,
        selected_source_normal_offsets=source_offsets,
        selected_confidence=confidence.astype(np.float32),
        select_distance=np.asarray([select_distance], dtype=np.float32),
        offset_radius=np.asarray([offset_radius], dtype=np.float32),
        bbox_diag=np.asarray([bbox_diag], dtype=np.float32),
    )

    summary = {
        "mode": "build_sof_mesh_oracle_candidates_v0",
        "lr_mesh_path": str(Path(args.lr_mesh_path).resolve()),
        "hr_mesh_path": str(Path(args.hr_mesh_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "outputs": {
            "all_candidates_preview": str((output_dir / "all_lr_shell_candidates_preview_v0.ply").resolve()),
            "selected_refined_candidates": str((output_dir / "selected_refined_hr_supported_candidates_v0.ply").resolve()),
            "carrier_payload": str(payload_path.resolve()),
        },
        "parameters": vars(args),
        "derived": {
            "bbox_diag": float(bbox_diag),
            "offset_radius": float(offset_radius),
            "select_distance": float(select_distance),
            "carrier_spacing": float(carrier_spacing),
        },
        "counts": {
            "anchors": int(anchors.shape[0]),
            "offset_layers": int(args.offset_layers),
            "candidate_count": int(flat_candidates.shape[0]),
            "valid_candidate_count": int(np.sum(valid)),
            "selected_count": int(refined.shape[0]),
        },
        "stats": {
            "all_candidate_hr_distance": stats_from_array(distances),
            "valid_candidate_hr_distance": stats_from_array(distances[valid]),
            "selected_initial_hr_distance": stats_from_array(distances[selected_flat_ids]),
            "selected_refined_hr_distance": stats_from_array(refined_distance),
            "selected_normal_alignment": stats_from_array(normal_alignment[selected_flat_ids]),
            "selected_confidence": stats_from_array(confidence),
            "source_normal_offsets": stats_from_array(source_offsets),
        },
        "note": (
            "This is an oracle upper-bound stage: HRmesh is used to choose reliable new surface candidates. "
            "It should be evaluated as candidate geometry, not as a deployable inference pipeline."
        ),
    }
    write_json(output_dir / "build_sof_mesh_oracle_candidates_v0_summary.json", summary)
    print(f"[oracle-candidates] selected {refined.shape[0]} / {flat_candidates.shape[0]} candidates")
    print(f"[oracle-candidates] saved selected point cloud: {output_dir / 'selected_refined_hr_supported_candidates_v0.ply'}")
    print(f"[oracle-candidates] saved carrier payload: {payload_path}")


if __name__ == "__main__":
    main()
