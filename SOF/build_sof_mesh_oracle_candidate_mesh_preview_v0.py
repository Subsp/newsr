from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

from utils.sof_mesh_patch_enhancer_v0 import (
    load_triangle_mesh,
    stats_from_array,
    write_json,
)


def colorize_confidence(confidence: np.ndarray) -> np.ndarray:
    conf = np.clip(confidence.reshape(-1), 0.0, 1.0)
    colors = np.stack(
        [
            np.round(255.0 * (1.0 - conf)),
            np.round(220.0 * conf),
            np.round(60.0 * (1.0 - conf)),
            np.full_like(conf, 255.0),
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def compact_mesh_from_faces(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, face_colors: np.ndarray | None):
    selected_faces = faces[face_ids]
    used_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    compact_faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    compact_vertices = vertices[used_vertices].astype(np.float32, copy=False)
    mesh = trimesh.Trimesh(vertices=compact_vertices, faces=compact_faces, process=False)
    if face_colors is not None:
        mesh.visual.face_colors = face_colors.astype(np.uint8, copy=False)
    return mesh, used_vertices


def per_face_stats(face_ids: np.ndarray, confidence: np.ndarray, displacement_norm: np.ndarray) -> Dict[str, np.ndarray]:
    unique_faces, inverse = np.unique(face_ids, return_inverse=True)
    count = np.zeros((unique_faces.shape[0],), dtype=np.float32)
    conf_sum = np.zeros_like(count)
    disp_sum = np.zeros_like(count)
    np.add.at(count, inverse, 1.0)
    np.add.at(conf_sum, inverse, confidence.astype(np.float32))
    np.add.at(disp_sum, inverse, displacement_norm.astype(np.float32))
    return {
        "face_ids": unique_faces.astype(np.int64),
        "support_count": count.astype(np.int32),
        "mean_confidence": conf_sum / np.clip(count, 1.0, None),
        "mean_displacement": disp_sum / np.clip(count, 1.0, None),
    }


def move_vertices_from_candidate_displacements(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
    displacement: np.ndarray,
    confidence: np.ndarray,
    confidence_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    moved_vertices = vertices.astype(np.float32, copy=True)
    disp_accum = np.zeros_like(moved_vertices, dtype=np.float32)
    weight_accum = np.zeros((moved_vertices.shape[0],), dtype=np.float32)
    sample_faces = faces[face_ids]
    conf_weight = np.power(np.clip(confidence.reshape(-1), 0.0, 1.0), float(confidence_power)).astype(np.float32)
    conf_weight = np.clip(conf_weight, 1e-4, None)
    for corner in range(3):
        vertex_ids = sample_faces[:, corner]
        weights = bary[:, corner].astype(np.float32) * conf_weight
        np.add.at(disp_accum, vertex_ids, displacement * weights[:, None])
        np.add.at(weight_accum, vertex_ids, weights)
    moved_mask = weight_accum > 1e-8
    moved_vertices[moved_mask] += disp_accum[moved_mask] / weight_accum[moved_mask, None]
    return moved_vertices, moved_mask


def main():
    parser = ArgumentParser(
        description=(
            "Convert oracle selected LR-shell candidates into mesh previews with faces. "
            "This is for visual inspection only; it does not claim a watertight SR mesh."
        )
    )
    parser.add_argument("--lr_mesh_path", type=str, required=True)
    parser.add_argument("--candidate_records", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_mesh = load_triangle_mesh(args.lr_mesh_path)
    vertices = np.asarray(lr_mesh.vertices, dtype=np.float32)
    faces = np.asarray(lr_mesh.faces, dtype=np.int64)
    records = np.load(args.candidate_records)

    selected_centers = records["selected_centers"].astype(np.float32)
    face_ids = records["selected_source_face_ids"].astype(np.int64)
    bary = records["selected_source_bary_coords"].astype(np.float32)
    confidence = records["selected_confidence"].astype(np.float32).reshape(-1)
    valid = (face_ids >= 0) & (face_ids < faces.shape[0])
    if not np.all(valid):
        selected_centers = selected_centers[valid]
        face_ids = face_ids[valid]
        bary = bary[valid]
        confidence = confidence[valid]
    if selected_centers.shape[0] == 0:
        raise RuntimeError("No valid selected candidates to mesh-preview.")

    triangles = vertices[faces[face_ids]]
    source_points = np.sum(triangles * bary[:, :, None], axis=1).astype(np.float32)
    displacement = selected_centers - source_points
    displacement_norm = np.linalg.norm(displacement, axis=1).astype(np.float32)
    face_stats = per_face_stats(face_ids, confidence, displacement_norm)
    face_colors = colorize_confidence(face_stats["mean_confidence"])

    moved_vertices, moved_vertex_mask = move_vertices_from_candidate_displacements(
        vertices=vertices,
        faces=faces,
        face_ids=face_ids,
        bary=bary,
        displacement=displacement,
        confidence=confidence,
        confidence_power=float(args.confidence_power),
    )

    source_mesh, used_source_vertices = compact_mesh_from_faces(
        vertices=vertices,
        faces=faces,
        face_ids=face_stats["face_ids"],
        face_colors=face_colors,
    )
    moved_mesh, used_moved_vertices = compact_mesh_from_faces(
        vertices=moved_vertices,
        faces=faces,
        face_ids=face_stats["face_ids"],
        face_colors=face_colors,
    )

    source_mesh_path = output_dir / "selected_source_faces_lrmesh_v0.ply"
    moved_mesh_path = output_dir / "selected_source_faces_oracle_moved_v0.ply"
    source_mesh.export(source_mesh_path)
    moved_mesh.export(moved_mesh_path)

    summary = {
        "mode": "build_sof_mesh_oracle_candidate_mesh_preview_v0",
        "lr_mesh_path": str(Path(args.lr_mesh_path).resolve()),
        "candidate_records": str(Path(args.candidate_records).resolve()),
        "output_dir": str(output_dir.resolve()),
        "outputs": {
            "selected_source_faces_lrmesh": str(source_mesh_path.resolve()),
            "selected_source_faces_oracle_moved": str(moved_mesh_path.resolve()),
        },
        "parameters": {
            "confidence_power": float(args.confidence_power),
        },
        "counts": {
            "selected_candidates": int(selected_centers.shape[0]),
            "selected_unique_faces": int(face_stats["face_ids"].shape[0]),
            "source_preview_vertices": int(used_source_vertices.shape[0]),
            "moved_preview_vertices": int(used_moved_vertices.shape[0]),
            "moved_global_vertices": int(np.sum(moved_vertex_mask)),
        },
        "stats": {
            "candidate_displacement_norm": stats_from_array(displacement_norm),
            "face_support_count": stats_from_array(face_stats["support_count"].astype(np.float32)),
            "face_mean_confidence": stats_from_array(face_stats["mean_confidence"]),
            "face_mean_displacement": stats_from_array(face_stats["mean_displacement"]),
        },
        "note": (
            "selected_source_faces_lrmesh shows which LR faces generated HR-supported candidates. "
            "selected_source_faces_oracle_moved moves those faces by averaged candidate displacements for visual inspection only."
        ),
    }
    write_json(output_dir / "build_sof_mesh_oracle_candidate_mesh_preview_v0_summary.json", summary)
    print(f"[oracle-candidate-mesh-preview] saved source faces: {source_mesh_path}")
    print(f"[oracle-candidate-mesh-preview] saved moved faces : {moved_mesh_path}")


if __name__ == "__main__":
    main()
