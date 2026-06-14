import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score mesh faces by how strongly they are supported by the current Gaussian cloud."
    )
    parser.add_argument("--mesh_path", type=str, required=True, help="Query mesh to score, typically the LR mesh.")
    parser.add_argument(
        "--point_cloud_path",
        type=str,
        default=None,
        help="Path to point_cloud.ply. If omitted, --model_path and --iteration are used.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="SOF model directory that contains point_cloud/iteration_x/point_cloud.ply.",
    )
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration for --model_path point cloud lookup.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store outputs.")
    parser.add_argument(
        "--proxy_variant",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="v2 reduces tiny-triangle bias by using broad face aggregation plus area-weighted thresholds.",
    )
    parser.add_argument(
        "--min_gaussian_opacity",
        type=float,
        default=0.01,
        help="Ignore Gaussians with sigmoid(opacity) below this threshold.",
    )
    parser.add_argument(
        "--radius_scale",
        type=float,
        default=2.0,
        help="Vertex support radius = max(radius_scale * local_mesh_scale, min_radius).",
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=None,
        help="Absolute lower bound for the support radius. Defaults to 3 * median Gaussian max-scale.",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="Optional upper bound for the support radius.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=None,
        help="Absolute slab half-thickness. Defaults to 3 * median Gaussian min-scale.",
    )
    parser.add_argument(
        "--coverage_gate",
        type=float,
        default=0.25,
        help="Normalized vertex support threshold used to mark a face vertex as covered.",
    )
    parser.add_argument(
        "--broad_mix",
        type=float,
        default=0.5,
        help="Blend weight between mean and max vertex support when building the broader v2 support score.",
    )
    parser.add_argument(
        "--support_weight",
        type=float,
        default=0.5,
        help="Weight of normalized support mass in the final face score.",
    )
    parser.add_argument(
        "--thinness_weight",
        type=float,
        default=0.3,
        help="Weight of thinness in the final face score.",
    )
    parser.add_argument(
        "--coverage_weight",
        type=float,
        default=0.2,
        help="Weight of coverage in the final face score.",
    )
    parser.add_argument(
        "--strong_quantile",
        type=float,
        default=0.8,
        help="Default score quantile used to cut the proxy strong core.",
    )
    parser.add_argument(
        "--weak_quantile",
        type=float,
        default=0.5,
        help="Default score quantile used to cut the proxy supported union.",
    )
    parser.add_argument(
        "--threshold_quantile_mode",
        type=str,
        choices=["count", "area"],
        default="area",
        help="Whether the strong/weak thresholds are chosen by face count or face area prevalence.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20000,
        help="Number of mesh vertices processed per batch.",
    )
    parser.add_argument(
        "--reference_mesh",
        type=str,
        default=None,
        help="Optional GT/reference mesh used to build oracle labels on the same query mesh.",
    )
    parser.add_argument(
        "--oracle_npz",
        type=str,
        default=None,
        help="Optional oracle file from compute_local_reliability.py with face_labels.",
    )
    parser.add_argument(
        "--n_reference_samples",
        type=int,
        default=300_000,
        help="Reference surface samples used when --reference_mesh is provided.",
    )
    return parser.parse_args()


def resolve_point_cloud_path(args) -> Path:
    if args.point_cloud_path:
        return Path(args.point_cloud_path)
    if args.model_path is None or args.iteration < 0:
        raise ValueError("Provide --point_cloud_path, or both --model_path and --iteration.")
    return Path(args.model_path) / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussians(point_cloud_path: Path, min_opacity: float) -> Dict[str, np.ndarray]:
    ply = PlyData.read(str(point_cloud_path))
    vertex = ply.elements[0]

    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float32)
    opacity = sigmoid(opacity_raw)

    scale_names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(scale_names) != 3:
        raise ValueError(f"Expected 3 scale_* fields in {point_cloud_path}, found {len(scale_names)}")
    scales_raw = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in scale_names], axis=1)
    scales = np.exp(scales_raw)

    keep = opacity >= min_opacity
    xyz = xyz[keep]
    opacity = opacity[keep]
    scales = scales[keep]

    return {
        "xyz": xyz,
        "opacity": opacity.astype(np.float32),
        "scales": scales.astype(np.float32),
    }


def robust_normalize(values: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    lo = np.quantile(values[finite], low_q)
    hi = np.quantile(values[finite], high_q)
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def weighted_quantile(values: np.ndarray, weights: Optional[np.ndarray], quantile: float) -> float:
    q = float(np.clip(quantile, 0.0, 1.0))
    if weights is None:
        return float(np.quantile(values, q))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != values.shape:
        raise ValueError("weights must match values shape")
    total = float(weights.sum())
    if total <= 1e-12:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    threshold = q * total
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(max(index, 0), len(values_sorted) - 1)
    return float(values_sorted[index])


def distance_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def colorize_score(values: np.ndarray) -> np.ndarray:
    v = np.clip(values.astype(np.float32), 0.0, 1.0)
    colors = np.zeros((values.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.round(255.0 * (1.0 - v)).astype(np.uint8)
    colors[:, 1] = np.round(220.0 * v).astype(np.uint8)
    colors[:, 2] = 0
    return colors


def export_submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray, path: Path):
    face_ids = np.flatnonzero(face_mask)
    if face_ids.size == 0:
        return
    submesh = mesh.submesh([face_ids], append=True, repair=False)
    submesh.export(path)


def compute_vertex_local_scale(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    vertex_area_sum = np.bincount(faces.reshape(-1), weights=np.repeat(face_areas, 3), minlength=len(mesh.vertices))
    vertex_degree = np.bincount(faces.reshape(-1), minlength=len(mesh.vertices))
    mean_incident_area = vertex_area_sum / np.maximum(vertex_degree, 1)
    return np.sqrt(np.clip(mean_incident_area, 1e-12, None)).astype(np.float32)


def compute_vertex_support(
    mesh: trimesh.Trimesh,
    gaussian_xyz: np.ndarray,
    gaussian_opacity: np.ndarray,
    support_radius: np.ndarray,
    thickness: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(normal_norm, 1e-12, None)

    tree = cKDTree(gaussian_xyz)

    support_mass = np.zeros(len(vertices), dtype=np.float32)
    thinness = np.zeros(len(vertices), dtype=np.float32)
    neighbor_count = np.zeros(len(vertices), dtype=np.int32)

    start_time = time.time()
    n_batches = math.ceil(len(vertices) / batch_size)
    for batch_idx in range(n_batches):
        begin = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(vertices))
        points = vertices[begin:end]
        point_normals = normals[begin:end]
        radii = support_radius[begin:end]
        search_radii = np.sqrt(radii * radii + thickness * thickness)
        neighbor_lists = tree.query_ball_point(points, r=search_radii, workers=-1)

        for local_idx, (point, normal, radius, candidate_ids) in enumerate(zip(points, point_normals, radii, neighbor_lists)):
            if len(candidate_ids) == 0:
                continue

            diff = gaussian_xyz[candidate_ids] - point[None, :]
            normal_offset = diff @ normal
            tangent_diff = diff - normal_offset[:, None] * normal[None, :]
            tangent_dist = np.linalg.norm(tangent_diff, axis=1)
            mask = (np.abs(normal_offset) <= thickness) & (tangent_dist <= radius)
            if not np.any(mask):
                continue

            local_normal = np.abs(normal_offset[mask])
            local_tangent = tangent_dist[mask]
            local_opacity = gaussian_opacity[np.asarray(candidate_ids, dtype=np.int64)[mask]]

            tangent_sigma = max(radius * 0.5, 1e-6)
            normal_sigma = max(thickness * 0.5, 1e-6)
            weights = local_opacity
            weights = weights * np.exp(-0.5 * (local_tangent / tangent_sigma) ** 2)
            weights = weights * np.exp(-0.5 * (local_normal / normal_sigma) ** 2)

            total_weight = float(weights.sum())
            if total_weight <= 1e-12:
                continue

            rms_normal = math.sqrt(float(np.sum(weights * (local_normal ** 2)) / total_weight))
            support_mass[begin + local_idx] = total_weight
            thinness[begin + local_idx] = float(np.exp(-rms_normal / max(thickness, 1e-6)))
            neighbor_count[begin + local_idx] = int(mask.sum())

        elapsed = time.time() - start_time
        print(
            f"[compute-gs-support] batch {batch_idx + 1}/{n_batches} "
            f"vertices {end}/{len(vertices)} elapsed={elapsed:.1f}s"
        )

    return support_mass, thinness, neighbor_count


def face_metrics_from_vertices(
    mesh: trimesh.Trimesh,
    vertex_support_mass: np.ndarray,
    vertex_support_norm: np.ndarray,
    vertex_thinness: np.ndarray,
    coverage_gate: float,
    broad_mix: float,
) -> Dict[str, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    blend = float(np.clip(broad_mix, 0.0, 1.0))

    support_mass_face = vertex_support_mass[faces]
    support_norm_face = vertex_support_norm[faces]
    thinness_face = vertex_thinness[faces]
    covered_face = (support_norm_face >= coverage_gate).astype(np.float32)

    face_support_mass = support_mass_face.mean(axis=1).astype(np.float32)
    face_support_norm_mean = support_norm_face.mean(axis=1).astype(np.float32)
    face_support_norm_max = support_norm_face.max(axis=1).astype(np.float32)
    face_support_norm_broad = ((1.0 - blend) * face_support_norm_mean + blend * face_support_norm_max).astype(np.float32)

    face_thinness_mean = thinness_face.mean(axis=1).astype(np.float32)
    face_thinness_max = thinness_face.max(axis=1).astype(np.float32)
    face_thinness_broad = ((1.0 - blend) * face_thinness_mean + blend * face_thinness_max).astype(np.float32)

    face_coverage = covered_face.mean(axis=1).astype(np.float32)
    face_coverage_any = covered_face.max(axis=1).astype(np.float32)
    face_coverage_broad = ((1.0 - blend) * face_coverage + blend * face_coverage_any).astype(np.float32)

    return {
        "face_support_mass": face_support_mass,
        "face_support_norm": face_support_norm_mean,
        "face_support_norm_max": face_support_norm_max,
        "face_support_norm_broad": face_support_norm_broad,
        "face_thinness": face_thinness_mean,
        "face_thinness_max": face_thinness_max,
        "face_thinness_broad": face_thinness_broad,
        "face_coverage": face_coverage,
        "face_coverage_any": face_coverage_any,
        "face_coverage_broad": face_coverage_broad,
    }


def build_oracle_from_reference(
    query_mesh: trimesh.Trimesh,
    reference_mesh: trimesh.Trimesh,
    n_reference_samples: int,
    strong_threshold: Optional[float] = None,
    weak_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    bbox_min = np.minimum(query_mesh.bounds[0], reference_mesh.bounds[0])
    bbox_max = np.maximum(query_mesh.bounds[1], reference_mesh.bounds[1])
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    strong_threshold = float(strong_threshold) if strong_threshold is not None else 0.002 * bbox_diag
    weak_threshold = float(weak_threshold) if weak_threshold is not None else 0.005 * bbox_diag

    reference_points, _ = trimesh.sample.sample_surface(reference_mesh, n_reference_samples)
    reference_tree = cKDTree(reference_points)

    query_vertices = np.asarray(query_mesh.vertices)
    vertex_distances, _ = reference_tree.query(query_vertices, k=1)
    query_faces = np.asarray(query_mesh.faces)
    face_distances = vertex_distances[query_faces].mean(axis=1)

    face_labels = np.zeros_like(face_distances, dtype=np.uint8)
    face_labels[face_distances <= weak_threshold] = 1
    face_labels[face_distances <= strong_threshold] = 2
    return {
        "face_distances": face_distances.astype(np.float32),
        "face_labels": face_labels,
        "bbox_diag": bbox_diag,
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
    }


def load_oracle(args, mesh: trimesh.Trimesh) -> Optional[Dict[str, np.ndarray]]:
    if args.oracle_npz:
        payload = np.load(args.oracle_npz)
        face_labels = payload["face_labels"]
        if face_labels.shape[0] != len(mesh.faces):
            raise ValueError("oracle face_labels length does not match query mesh face count")
        return {
            "face_labels": face_labels.astype(np.uint8),
            "face_distances": payload["face_distances"].astype(np.float32) if "face_distances" in payload.files else None,
            "bbox_diag": float(payload["bbox_diag"]) if "bbox_diag" in payload.files else None,
            "strong_threshold": float(payload["strong_threshold"]) if "strong_threshold" in payload.files else None,
            "weak_threshold": float(payload["weak_threshold"]) if "weak_threshold" in payload.files else None,
        }
    if args.reference_mesh:
        reference_mesh = trimesh.load_mesh(args.reference_mesh, process=False)
        return build_oracle_from_reference(mesh, reference_mesh, args.n_reference_samples)
    return None


def binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
    fn = int(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def weighted_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64)
    tp = float(weights[np.logical_and(pred_mask, gt_mask)].sum())
    fp = float(weights[np.logical_and(pred_mask, np.logical_not(gt_mask))].sum())
    fn = float(weights[np.logical_and(np.logical_not(pred_mask), gt_mask)].sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def threshold_from_prevalence(scores: np.ndarray, target_mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        prevalence = float(target_mask.mean())
    else:
        weights = np.asarray(weights, dtype=np.float64)
        prevalence = float(weights[target_mask].sum() / max(weights.sum(), 1e-12))
    if prevalence <= 0.0:
        return float(np.inf)
    if prevalence >= 1.0:
        return float(np.min(scores) - 1e-6)
    return weighted_quantile(scores, weights, 1.0 - prevalence)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(args.mesh_path, process=False)
    point_cloud_path = resolve_point_cloud_path(args)
    gaussians = load_gaussians(point_cloud_path, min_opacity=args.min_gaussian_opacity)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    threshold_weights = face_areas if args.threshold_quantile_mode == "area" else None

    gaussian_scale_min = np.min(gaussians["scales"], axis=1)
    gaussian_scale_max = np.max(gaussians["scales"], axis=1)
    default_min_radius = float(3.0 * np.median(gaussian_scale_max))
    default_thickness = float(3.0 * np.median(gaussian_scale_min))
    min_radius = float(args.min_radius) if args.min_radius is not None else default_min_radius
    thickness = float(args.thickness) if args.thickness is not None else default_thickness

    vertex_local_scale = compute_vertex_local_scale(mesh)
    support_radius = np.maximum(args.radius_scale * vertex_local_scale, min_radius).astype(np.float32)
    if args.max_radius is not None:
        support_radius = np.minimum(support_radius, float(args.max_radius)).astype(np.float32)

    print("[compute-gs-support] mesh vertices:", len(mesh.vertices))
    print("[compute-gs-support] mesh faces   :", len(mesh.faces))
    print("[compute-gs-support] gaussians    :", len(gaussians["xyz"]))
    print("[compute-gs-support] min_radius   :", min_radius)
    print("[compute-gs-support] thickness    :", thickness)

    vertex_support_mass, vertex_thinness, vertex_neighbor_count = compute_vertex_support(
        mesh=mesh,
        gaussian_xyz=gaussians["xyz"],
        gaussian_opacity=gaussians["opacity"],
        support_radius=support_radius,
        thickness=thickness,
        batch_size=args.batch_size,
    )
    vertex_support_norm = robust_normalize(vertex_support_mass)

    face_metrics = face_metrics_from_vertices(
        mesh=mesh,
        vertex_support_mass=vertex_support_mass,
        vertex_support_norm=vertex_support_norm,
        vertex_thinness=vertex_thinness,
        coverage_gate=args.coverage_gate,
        broad_mix=args.broad_mix,
    )
    total_weight = args.support_weight + args.thinness_weight + args.coverage_weight
    if total_weight <= 1e-12:
        raise ValueError("support/thinness/coverage weights must sum to a positive value")
    seed_face_score = (
        args.support_weight * face_metrics["face_support_norm"]
        + args.thinness_weight * face_metrics["face_thinness"]
        + args.coverage_weight * face_metrics["face_coverage"]
    ) / total_weight
    broad_face_score = (
        args.support_weight * face_metrics["face_support_norm_broad"]
        + args.thinness_weight * face_metrics["face_thinness_broad"]
        + args.coverage_weight * face_metrics["face_coverage_broad"]
    ) / total_weight
    seed_face_score = seed_face_score.astype(np.float32)
    broad_face_score = broad_face_score.astype(np.float32)
    face_score = broad_face_score if args.proxy_variant == "v2" else seed_face_score

    if args.proxy_variant == "v2":
        strong_threshold = weighted_quantile(seed_face_score, threshold_weights, args.strong_quantile)
        weak_threshold = weighted_quantile(broad_face_score, threshold_weights, args.weak_quantile)
        strong_mask = seed_face_score >= strong_threshold
        weak_mask = broad_face_score >= weak_threshold
    else:
        strong_threshold = weighted_quantile(seed_face_score, threshold_weights, args.strong_quantile)
        weak_threshold = weighted_quantile(seed_face_score, threshold_weights, args.weak_quantile)
        strong_mask = seed_face_score >= strong_threshold
        weak_mask = seed_face_score >= weak_threshold

    face_labels = np.zeros(len(mesh.faces), dtype=np.uint8)
    face_labels[weak_mask] = 1
    face_labels[strong_mask] = 2

    vertex_colors = colorize_score(vertex_support_norm)
    vertex_colored_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        vertex_colors=vertex_colors,
        process=False,
    )
    vertex_colored_mesh.export(output_dir / "query_mesh_gs_vertex_support_colored.ply")

    face_colors = colorize_score(face_score)
    colored_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        face_colors=face_colors,
        process=False,
    )
    colored_mesh.export(output_dir / "query_mesh_gs_support_colored.ply")

    export_submesh(mesh, face_labels == 2, output_dir / "query_mesh_gs_proxy_strong_core.ply")
    export_submesh(mesh, face_labels == 1, output_dir / "query_mesh_gs_proxy_weak_support_only.ply")
    export_submesh(mesh, face_labels >= 1, output_dir / "query_mesh_gs_proxy_supported_union.ply")
    export_submesh(mesh, face_labels == 0, output_dir / "query_mesh_gs_proxy_unsupported_only.ply")

    oracle = load_oracle(args, mesh)
    oracle_summary = None
    if oracle is not None:
        oracle_face_labels = oracle["face_labels"]
        oracle_strong = oracle_face_labels == 2
        oracle_supported = oracle_face_labels >= 1
        oracle_area = face_areas.astype(np.float64)

        matched_strong_threshold = threshold_from_prevalence(face_score, oracle_strong)
        matched_supported_threshold = threshold_from_prevalence(face_score, oracle_supported)
        area_matched_strong_threshold = threshold_from_prevalence(face_score, oracle_strong, weights=oracle_area)
        area_matched_supported_threshold = threshold_from_prevalence(face_score, oracle_supported, weights=oracle_area)

        pred_strong_default = face_labels == 2
        pred_supported_default = face_labels >= 1
        pred_strong_matched = face_score >= matched_strong_threshold
        pred_supported_matched = face_score >= matched_supported_threshold
        pred_strong_area_matched = face_score >= area_matched_strong_threshold
        pred_supported_area_matched = face_score >= area_matched_supported_threshold

        oracle_summary = {
            "oracle_counts": {
                "strong_core": int(oracle_strong.sum()),
                "supported_union": int(oracle_supported.sum()),
                "unsupported": int((oracle_face_labels == 0).sum()),
                "total": int(oracle_face_labels.shape[0]),
            },
            "oracle_area_ratio": {
                "strong_core": float(face_areas[oracle_strong].sum() / max(face_areas.sum(), 1e-12)),
                "supported_union": float(face_areas[oracle_supported].sum() / max(face_areas.sum(), 1e-12)),
                "unsupported": float(face_areas[oracle_face_labels == 0].sum() / max(face_areas.sum(), 1e-12)),
            },
            "score_by_oracle_label": {
                "strong_core": distance_stats(face_score[oracle_face_labels == 2]) if np.any(oracle_face_labels == 2) else None,
                "weak_support": distance_stats(face_score[oracle_face_labels == 1]) if np.any(oracle_face_labels == 1) else None,
                "unsupported": distance_stats(face_score[oracle_face_labels == 0]) if np.any(oracle_face_labels == 0) else None,
            },
            "default_threshold_eval": {
                "strong_core": binary_metrics(pred_strong_default, oracle_strong),
                "supported_union": binary_metrics(pred_supported_default, oracle_supported),
                "strong_core_area_weighted": weighted_binary_metrics(pred_strong_default, oracle_strong, oracle_area),
                "supported_union_area_weighted": weighted_binary_metrics(pred_supported_default, oracle_supported, oracle_area),
                "strong_threshold": strong_threshold,
                "weak_threshold": weak_threshold,
            },
            "oracle_prevalence_matched_eval": {
                "strong_core": binary_metrics(pred_strong_matched, oracle_strong),
                "supported_union": binary_metrics(pred_supported_matched, oracle_supported),
                "strong_core_area_weighted": weighted_binary_metrics(pred_strong_matched, oracle_strong, oracle_area),
                "supported_union_area_weighted": weighted_binary_metrics(pred_supported_matched, oracle_supported, oracle_area),
                "strong_threshold": matched_strong_threshold,
                "weak_threshold": matched_supported_threshold,
            },
            "oracle_area_prevalence_matched_eval": {
                "strong_core": binary_metrics(pred_strong_area_matched, oracle_strong),
                "supported_union": binary_metrics(pred_supported_area_matched, oracle_supported),
                "strong_core_area_weighted": weighted_binary_metrics(pred_strong_area_matched, oracle_strong, oracle_area),
                "supported_union_area_weighted": weighted_binary_metrics(pred_supported_area_matched, oracle_supported, oracle_area),
                "strong_threshold": area_matched_strong_threshold,
                "weak_threshold": area_matched_supported_threshold,
            },
            "oracle_reference": {
                "bbox_diag": oracle.get("bbox_diag"),
                "strong_threshold": oracle.get("strong_threshold"),
                "weak_threshold": oracle.get("weak_threshold"),
            },
        }
        if oracle.get("face_distances") is not None:
            oracle_summary["oracle_face_distance"] = distance_stats(oracle["face_distances"])

    np.savez_compressed(
        output_dir / "query_mesh_gs_support_proxy.npz",
        vertex_support_mass=vertex_support_mass,
        vertex_thinness=vertex_thinness,
        vertex_neighbor_count=vertex_neighbor_count,
        vertex_support_norm=vertex_support_norm,
        face_support_mass=face_metrics["face_support_mass"],
        face_support_norm=face_metrics["face_support_norm"],
        face_support_norm_max=face_metrics["face_support_norm_max"],
        face_support_norm_broad=face_metrics["face_support_norm_broad"],
        face_thinness=face_metrics["face_thinness"],
        face_thinness_max=face_metrics["face_thinness_max"],
        face_thinness_broad=face_metrics["face_thinness_broad"],
        face_coverage=face_metrics["face_coverage"],
        face_coverage_any=face_metrics["face_coverage_any"],
        face_coverage_broad=face_metrics["face_coverage_broad"],
        seed_face_score=seed_face_score,
        broad_face_score=broad_face_score,
        face_score=face_score,
        face_labels=face_labels,
        support_radius=support_radius,
        thickness=thickness,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
    )
    area_total = float(face_areas.sum())

    def area_for(label: int) -> float:
        return float(face_areas[face_labels == label].sum())

    summary = {
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "point_cloud_path": str(point_cloud_path.resolve()),
        "gaussian_count_after_opacity_filter": int(len(gaussians["xyz"])),
        "gaussian_opacity": distance_stats(gaussians["opacity"]),
        "gaussian_scale_min": distance_stats(gaussian_scale_min),
        "gaussian_scale_max": distance_stats(gaussian_scale_max),
        "proxy_params": {
            "proxy_variant": args.proxy_variant,
            "min_gaussian_opacity": args.min_gaussian_opacity,
            "radius_scale": args.radius_scale,
            "min_radius": min_radius,
            "max_radius": args.max_radius,
            "thickness": thickness,
            "coverage_gate": args.coverage_gate,
            "broad_mix": args.broad_mix,
            "support_weight": args.support_weight,
            "thinness_weight": args.thinness_weight,
            "coverage_weight": args.coverage_weight,
            "strong_quantile": args.strong_quantile,
            "weak_quantile": args.weak_quantile,
            "threshold_quantile_mode": args.threshold_quantile_mode,
            "batch_size": args.batch_size,
        },
        "vertex_support_mass": distance_stats(vertex_support_mass),
        "vertex_thinness": distance_stats(vertex_thinness),
        "vertex_neighbor_count": distance_stats(vertex_neighbor_count.astype(np.float32)),
        "face_support_mass": distance_stats(face_metrics["face_support_mass"]),
        "face_support_norm": distance_stats(face_metrics["face_support_norm"]),
        "face_support_norm_broad": distance_stats(face_metrics["face_support_norm_broad"]),
        "face_thinness": distance_stats(face_metrics["face_thinness"]),
        "face_thinness_broad": distance_stats(face_metrics["face_thinness_broad"]),
        "face_coverage": distance_stats(face_metrics["face_coverage"]),
        "face_coverage_broad": distance_stats(face_metrics["face_coverage_broad"]),
        "seed_face_score": distance_stats(seed_face_score),
        "broad_face_score": distance_stats(broad_face_score),
        "face_score": distance_stats(face_score),
        "proxy_thresholds": {
            "strong_threshold": strong_threshold,
            "weak_threshold": weak_threshold,
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
        "oracle_compare": oracle_summary,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
