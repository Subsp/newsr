import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute a local reliability map for a query mesh against a reference mesh."
    )
    parser.add_argument("--reference_mesh", type=str, required=True, help="Reference mesh, e.g. GT mesh.")
    parser.add_argument("--query_mesh", type=str, required=True, help="Query mesh, e.g. LR mesh.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store outputs.")
    parser.add_argument(
        "--n_reference_samples",
        type=int,
        default=300_000,
        help="Number of reference surface samples used to build the KD-tree.",
    )
    parser.add_argument(
        "--n_query_samples",
        type=int,
        default=300_000,
        help="Number of query surface samples used for sampled-point statistics/visualization.",
    )
    parser.add_argument(
        "--strong_ratio",
        type=float,
        default=0.002,
        help="Strong-core threshold as a fraction of the union bbox diagonal when no absolute threshold is provided.",
    )
    parser.add_argument(
        "--weak_ratio",
        type=float,
        default=0.005,
        help="Weak-support threshold as a fraction of the union bbox diagonal when no absolute threshold is provided.",
    )
    parser.add_argument(
        "--strong_threshold",
        type=float,
        default=None,
        help="Absolute strong-core threshold. Overrides --strong_ratio if set.",
    )
    parser.add_argument(
        "--weak_threshold",
        type=float,
        default=None,
        help="Absolute weak-support threshold. Overrides --weak_ratio if set.",
    )
    return parser.parse_args()


def bbox_diag(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> float:
    bbox_min = np.minimum(mesh_a.bounds[0], mesh_b.bounds[0])
    bbox_max = np.maximum(mesh_a.bounds[1], mesh_b.bounds[1])
    return float(np.linalg.norm(bbox_max - bbox_min))


def distance_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def classify(values: np.ndarray, strong_threshold: float, weak_threshold: float) -> np.ndarray:
    labels = np.zeros_like(values, dtype=np.uint8)
    labels[values <= weak_threshold] = 1
    labels[values <= strong_threshold] = 2
    return labels


def colorize(values: np.ndarray, strong_threshold: float, weak_threshold: float) -> np.ndarray:
    colors = np.zeros((values.shape[0], 3), dtype=np.uint8)
    strong = values <= strong_threshold
    weak = (values > strong_threshold) & (values <= weak_threshold)
    unsupported = values > weak_threshold

    colors[strong] = np.array([0, 200, 0], dtype=np.uint8)

    if np.any(weak):
        t = (values[weak] - strong_threshold) / max(weak_threshold - strong_threshold, 1e-12)
        weak_colors = np.stack(
            [
                (255 * t),
                np.full_like(t, 220.0),
                np.zeros_like(t),
            ],
            axis=1,
        )
        colors[weak] = weak_colors.astype(np.uint8)

    if np.any(unsupported):
        clip = max(weak_threshold * 2.0, weak_threshold + 1e-12)
        t = np.clip((values[unsupported] - weak_threshold) / max(clip - weak_threshold, 1e-12), 0.0, 1.0)
        unsupported_colors = np.stack(
            [
                np.full_like(t, 255.0),
                220.0 * (1.0 - t),
                np.zeros_like(t),
            ],
            axis=1,
        )
        colors[unsupported] = unsupported_colors.astype(np.uint8)

    return colors


def export_submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray, path: Path):
    face_ids = np.flatnonzero(face_mask)
    if face_ids.size == 0:
        return
    submesh = mesh.submesh([face_ids], append=True, repair=False)
    submesh.export(path)


def main():
    args = parse_args()

    reference_mesh = trimesh.load_mesh(args.reference_mesh, process=False)
    query_mesh = trimesh.load_mesh(args.query_mesh, process=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diag = bbox_diag(reference_mesh, query_mesh)
    strong_threshold = float(args.strong_threshold) if args.strong_threshold is not None else float(args.strong_ratio * diag)
    weak_threshold = float(args.weak_threshold) if args.weak_threshold is not None else float(args.weak_ratio * diag)
    if weak_threshold < strong_threshold:
        raise ValueError("weak threshold must be >= strong threshold")

    reference_points, _ = trimesh.sample.sample_surface(reference_mesh, args.n_reference_samples)
    reference_tree = cKDTree(reference_points)

    query_vertices = np.asarray(query_mesh.vertices)
    vertex_distances, _ = reference_tree.query(query_vertices, k=1)

    query_faces = np.asarray(query_mesh.faces)
    face_distances = vertex_distances[query_faces].mean(axis=1)
    vertex_labels = classify(vertex_distances, strong_threshold, weak_threshold)
    face_labels = classify(face_distances, strong_threshold, weak_threshold)

    vertex_colors = colorize(vertex_distances, strong_threshold, weak_threshold)
    colored_query_mesh = trimesh.Trimesh(
        vertices=query_mesh.vertices.copy(),
        faces=query_mesh.faces.copy(),
        vertex_colors=vertex_colors,
        process=False,
    )
    colored_query_mesh.export(output_dir / "query_mesh_local_reliability_colored.ply")

    export_submesh(query_mesh, face_labels == 2, output_dir / "query_mesh_strong_core.ply")
    export_submesh(query_mesh, face_labels == 1, output_dir / "query_mesh_weak_support_only.ply")
    export_submesh(query_mesh, face_labels >= 1, output_dir / "query_mesh_supported_union.ply")
    export_submesh(query_mesh, face_labels == 0, output_dir / "query_mesh_unsupported_only.ply")

    sampled_query_points, sampled_face_ids = trimesh.sample.sample_surface(query_mesh, args.n_query_samples)
    sampled_query_distances, _ = reference_tree.query(sampled_query_points, k=1)
    sampled_query_colors = colorize(sampled_query_distances, strong_threshold, weak_threshold)
    trimesh.points.PointCloud(sampled_query_points, colors=sampled_query_colors).export(
        output_dir / "query_surface_samples_colored.ply"
    )

    face_areas = query_mesh.area_faces
    area_total = float(face_areas.sum())

    def area_for(label: int) -> float:
        return float(face_areas[face_labels == label].sum())

    summary = {
        "reference_mesh": str(Path(args.reference_mesh).resolve()),
        "query_mesh": str(Path(args.query_mesh).resolve()),
        "bbox_diag": diag,
        "thresholds": {
            "strong_threshold": strong_threshold,
            "weak_threshold": weak_threshold,
            "strong_ratio": args.strong_ratio,
            "weak_ratio": args.weak_ratio,
        },
        "vertex_distance": distance_stats(vertex_distances),
        "face_distance": distance_stats(face_distances),
        "sampled_query_distance": distance_stats(sampled_query_distances),
        "vertex_counts": {
            "strong_core": int((vertex_labels == 2).sum()),
            "weak_support": int((vertex_labels == 1).sum()),
            "unsupported": int((vertex_labels == 0).sum()),
            "total": int(vertex_labels.shape[0]),
        },
        "face_counts": {
            "strong_core": int((face_labels == 2).sum()),
            "weak_support": int((face_labels == 1).sum()),
            "unsupported": int((face_labels == 0).sum()),
            "total": int(face_labels.shape[0]),
        },
        "face_area": {
            "strong_core": area_for(2),
            "weak_support": area_for(1),
            "unsupported": area_for(0),
            "total": area_total,
        },
        "face_area_ratio": {
            "strong_core": area_for(2) / max(area_total, 1e-12),
            "weak_support": area_for(1) / max(area_total, 1e-12),
            "unsupported": area_for(0) / max(area_total, 1e-12),
        },
    }

    np.savez_compressed(
        output_dir / "query_mesh_local_reliability.npz",
        vertex_distances=vertex_distances,
        face_distances=face_distances,
        vertex_labels=vertex_labels,
        face_labels=face_labels,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        bbox_diag=diag,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
