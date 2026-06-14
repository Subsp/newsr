from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


SH_C0 = 0.28209479177387814


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def stats(values: np.ndarray, max_samples: int = 2_000_000) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    total = int(arr.size)
    finite = np.isfinite(arr)
    arr = arr[finite]
    sampled = False
    if arr.size > int(max_samples):
        step = int(math.ceil(arr.size / int(max_samples)))
        arr = arr[::step]
        sampled = True
    if arr.size == 0:
        return {"count": total, "finite_count": int(finite.sum()), "sampled": sampled}
    q = np.percentile(arr, [0, 1, 5, 10, 50, 90, 95, 99, 99.5, 99.9, 100])
    return {
        "count": total,
        "finite_count": int(finite.sum()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p10": float(q[3]),
        "median": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "p995": float(q[8]),
        "p999": float(q[9]),
        "max": float(q[10]),
        "sampled": sampled,
    }


def parse_ply_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = b""
        while b"end_header" not in data:
            chunk = f.read(4096)
            if not chunk:
                raise ValueError(f"PLY header does not contain end_header: {path}")
            data += chunk
    marker = b"end_header"
    marker_idx = data.find(marker)
    newline_idx = data.find(b"\n", marker_idx)
    if newline_idx < 0:
        raise ValueError(f"PLY header end has no newline: {path}")
    header_bytes = data[: newline_idx + 1]
    header = header_bytes.decode("latin1")
    lines = header.splitlines()
    if not any(line.strip() == "format binary_little_endian 1.0" for line in lines):
        raise ValueError(f"Only binary_little_endian PLY is supported: {path}")

    elements: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            current = parts[1]
            elements[current] = {"count": int(parts[2]), "properties": []}
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                elements[current]["properties"].append({"kind": "list", "count_type": parts[2], "item_type": parts[3], "name": parts[4]})
            else:
                elements[current]["properties"].append({"kind": "scalar", "type": parts[1], "name": parts[2]})
    return {"header_size": len(header_bytes), "header": header, "elements": elements}


def vertex_dtype(header: dict[str, Any]) -> np.dtype:
    props = header["elements"]["vertex"]["properties"]
    fields = []
    for prop in props:
        if prop["kind"] != "scalar":
            raise ValueError("List properties in vertex element are not supported.")
        ply_type = prop["type"]
        if ply_type not in PLY_DTYPE_MAP:
            raise ValueError(f"Unsupported PLY scalar type: {ply_type}")
        fields.append((prop["name"], PLY_DTYPE_MAP[ply_type]))
    return np.dtype(fields)


def read_vertex_table(path: Path) -> tuple[np.memmap, dict[str, Any]]:
    header = parse_ply_header(path)
    dtype = vertex_dtype(header)
    count = int(header["elements"]["vertex"]["count"])
    table = np.memmap(path, dtype=dtype, mode="r", offset=int(header["header_size"]), shape=(count,))
    return table, header


def xyz_from_table(table: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.asarray(table["x"], dtype=np.float32),
            np.asarray(table["y"], dtype=np.float32),
            np.asarray(table["z"], dtype=np.float32),
        ],
        axis=1,
    )


def sorted_prop_names(table: np.ndarray, prefix: str) -> list[str]:
    names = list(table.dtype.names or [])
    return sorted(
        [name for name in names if name.startswith(prefix)],
        key=lambda item: int(item.rsplit("_", 1)[1]) if item.rsplit("_", 1)[-1].isdigit() else item,
    )


def stack_props(table: np.ndarray, names: list[str]) -> np.ndarray:
    if not names:
        return np.empty((len(table), 0), dtype=np.float32)
    return np.stack([np.asarray(table[name], dtype=np.float32) for name in names], axis=1)


def rotation_matrix_from_quaternion_wxyz(q_raw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_raw, dtype=np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, np.maximum(norm, 1e-8), out=np.zeros_like(q), where=norm > 0.0)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    matrix = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - w * z)
    matrix[:, 0, 2] = 2.0 * (x * z + w * y)
    matrix[:, 1, 0] = 2.0 * (x * y + w * z)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - w * x)
    matrix[:, 2, 0] = 2.0 * (x * z - w * y)
    matrix[:, 2, 1] = 2.0 * (y * z + w * x)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def linspace_sample_count(total: int, max_count: int) -> np.ndarray:
    if int(max_count) <= 0 or total <= int(max_count):
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=int(max_count), dtype=np.int64)


def query_tree_chunked(tree: cKDTree, points: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty((points.shape[0],), dtype=np.float32)
    indices = np.empty((points.shape[0],), dtype=np.int64)
    chunk = max(int(chunk_size), 1)
    for begin in range(0, points.shape[0], chunk):
        end = min(begin + chunk, points.shape[0])
        d, i = tree.query(points[begin:end], k=1, workers=-1)
        distances[begin:end] = np.asarray(d, dtype=np.float32)
        indices[begin:end] = np.asarray(i, dtype=np.int64)
    return distances, indices


def estimate_pca_normals_chunked(
    tree: cKDTree,
    reference_points: np.ndarray,
    query_points: np.ndarray,
    *,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    neighbor_count = max(int(k), 3)
    normals = np.empty((query_points.shape[0], 3), dtype=np.float32)
    chunk = max(int(chunk_size), 1)
    for begin in range(0, query_points.shape[0], chunk):
        end = min(begin + chunk, query_points.shape[0])
        _, indices = tree.query(query_points[begin:end], k=neighbor_count, workers=-1)
        if indices.ndim == 1:
            indices = indices[:, None]
        neighbors = reference_points[np.asarray(indices, dtype=np.int64)]
        centered = neighbors - np.mean(neighbors, axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / float(neighbor_count)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, :, 0]
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normals[begin:end] = np.divide(
            normal,
            np.maximum(normal_norm, 1e-8),
            out=np.zeros_like(normal, dtype=np.float32),
            where=normal_norm > 0.0,
        ).astype(np.float32, copy=False)
    return normals


def gaussian_features(table: np.ndarray) -> dict[str, np.ndarray]:
    xyz = xyz_from_table(table)
    opacity_raw = np.asarray(table["opacity"], dtype=np.float32) if "opacity" in table.dtype.names else np.zeros((len(table),), dtype=np.float32)
    opacity = sigmoid(opacity_raw)

    scale_names = sorted_prop_names(table, "scale_")
    scale_raw = stack_props(table, scale_names)
    scale = np.exp(scale_raw).astype(np.float32) if scale_raw.size else np.zeros((len(table), 3), dtype=np.float32)
    filter_3d = np.asarray(table["filter_3D"], dtype=np.float32) if "filter_3D" in table.dtype.names else np.zeros((len(table),), dtype=np.float32)
    effective_scale = np.sqrt(scale * scale + filter_3d[:, None] * filter_3d[:, None]).astype(np.float32)
    scale_major = np.max(effective_scale, axis=1).astype(np.float32)
    scale_minor = np.min(effective_scale, axis=1).astype(np.float32)
    anisotropy = (scale_major / np.maximum(scale_minor, 1e-8)).astype(np.float32)
    major_axis_idx = np.argmax(effective_scale, axis=1).astype(np.int64)

    rot_names = sorted_prop_names(table, "rot_")
    if len(rot_names) >= 4:
        rotation = stack_props(table, rot_names[:4])
        rot_mats = rotation_matrix_from_quaternion_wxyz(rotation)
        row_ids = np.arange(len(table), dtype=np.int64)
        major_axis_dir = rot_mats[row_ids, :, major_axis_idx].astype(np.float32, copy=False)
    else:
        major_axis_dir = np.zeros((len(table), 3), dtype=np.float32)
        major_axis_dir[:, 0] = 1.0

    dc_names = sorted_prop_names(table, "f_dc_")
    dc = stack_props(table, dc_names)
    if dc.shape[1] >= 3:
        rgb_dc = dc[:, :3] * SH_C0 + 0.5
        dc_luma = (0.2126 * rgb_dc[:, 0] + 0.7152 * rgb_dc[:, 1] + 0.0722 * rgb_dc[:, 2]).astype(np.float32)
    else:
        dc_luma = np.zeros((len(table),), dtype=np.float32)

    rest_norm = np.zeros((len(table),), dtype=np.float32)
    for name in sorted_prop_names(table, "f_rest_"):
        arr = np.asarray(table[name], dtype=np.float32)
        rest_norm += arr * arr
    rest_norm = np.sqrt(rest_norm).astype(np.float32)

    return {
        "xyz": xyz,
        "opacity": opacity,
        "scale_major": scale_major,
        "scale_minor": scale_minor,
        "major_axis_idx": major_axis_idx.astype(np.int64),
        "major_axis_dir": major_axis_dir,
        "effective_anisotropy": anisotropy,
        "dc_luma": dc_luma,
        "rest_norm": rest_norm,
    }


def write_point_cloud_ply(path: Path, xyz: np.ndarray, score: np.ndarray, max_points: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if xyz.shape[0] == 0:
        path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n", encoding="utf-8")
        return
    ids = np.arange(xyz.shape[0], dtype=np.int64)
    if int(max_points) > 0 and ids.size > int(max_points):
        ids = np.argsort(-score, kind="stable")[: int(max_points)]
    xyz_sel = xyz[ids].astype(np.float32, copy=False)
    score_sel = score[ids].astype(np.float32, copy=False)
    denom = max(float(np.percentile(score_sel, 99.0)), 1e-6)
    heat = np.clip(score_sel / denom, 0.0, 1.0)
    colors = np.stack(
        [
            np.full_like(heat, 255.0),
            210.0 * (1.0 - heat),
            40.0 * (1.0 - heat),
            np.full_like(heat, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")])
    data = np.empty((xyz_sel.shape[0],), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz_sel[:, 0], xyz_sel[:, 1], xyz_sel[:, 2]
    data["red"], data["green"], data["blue"], data["alpha"] = colors[:, 0], colors[:, 1], colors[:, 2], colors[:, 3]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment mesh-delta star gaussian candidate preview\n"
        f"element vertex {xyz_sel.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def save_payload(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch_payload = {}
        for key, value in arrays.items():
            if value.dtype == np.bool_:
                torch_payload[key] = torch.from_numpy(value.astype(bool, copy=False))
            elif np.issubdtype(value.dtype, np.integer):
                torch_payload[key] = torch.from_numpy(value.astype(np.int64, copy=False))
            else:
                torch_payload[key] = torch.from_numpy(value.astype(np.float32, copy=False))
        torch.save(torch_payload, str(path))
        return str(path)
    except Exception as exc:
        npz_path = path.with_suffix(".npz")
        np.savez_compressed(npz_path, **arrays)
        print(f"[mesh-delta-star-v0] torch save failed ({exc}); saved npz payload instead: {npz_path}")
        return str(npz_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use SOF mesh deltas to nominate Gaussians near starburst-like mesh changes.")
    parser.add_argument("--raw_mesh_path", required=True)
    parser.add_argument("--stage_mesh_path", required=True)
    parser.add_argument("--stage_point_cloud_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_raw_reference_vertices", type=int, default=1_000_000)
    parser.add_argument("--max_stage_query_vertices", type=int, default=2_000_000)
    parser.add_argument("--max_delta_points", type=int, default=500_000)
    parser.add_argument("--query_chunk_size", type=int, default=262144)
    parser.add_argument("--mesh_delta_quantile", type=float, default=99.0)
    parser.add_argument("--mesh_delta_distance_threshold", type=float, default=0.0)
    parser.add_argument("--mesh_delta_distance_floor", type=float, default=0.08)
    parser.add_argument("--delta_radius_abs", type=float, default=0.05)
    parser.add_argument("--delta_radius_scale", type=float, default=1.25)
    parser.add_argument("--delta_radius_mode", choices=["scale_max", "absolute"], default="scale_max")
    parser.add_argument("--gaussian_distance_mode", choices=["center", "major_endpoints"], default="center")
    parser.add_argument("--major_endpoint_scale", type=float, default=1.0)
    parser.add_argument("--tangent_angle_mode", choices=["none", "raw_pca"], default="none")
    parser.add_argument("--tangent_angle_k", type=int, default=12)
    parser.add_argument("--tangent_angle_distance_floor", type=float, default=0.15)
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--min_anisotropy", type=float, default=2.0)
    parser.add_argument("--min_candidate_score", type=float, default=0.05)
    parser.add_argument("--candidate_score_quantile", type=float, default=99.0)
    parser.add_argument("--max_candidates", type=int, default=200_000)
    parser.add_argument("--preview_max_points", type=int, default=200_000)
    args = parser.parse_args()

    raw_mesh_path = Path(args.raw_mesh_path).expanduser().resolve()
    stage_mesh_path = Path(args.stage_mesh_path).expanduser().resolve()
    point_cloud_path = Path(args.stage_point_cloud_ply).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_table, raw_header = read_vertex_table(raw_mesh_path)
    stage_table, stage_header = read_vertex_table(stage_mesh_path)
    raw_xyz = xyz_from_table(raw_table)
    stage_xyz = xyz_from_table(stage_table)

    raw_ids = linspace_sample_count(raw_xyz.shape[0], int(args.max_raw_reference_vertices))
    stage_ids = linspace_sample_count(stage_xyz.shape[0], int(args.max_stage_query_vertices))
    raw_reference = np.asarray(raw_xyz[raw_ids], dtype=np.float32)
    stage_query = np.asarray(stage_xyz[stage_ids], dtype=np.float32)

    print(f"[mesh-delta-star-v0] raw reference vertices: {raw_reference.shape[0]}/{raw_xyz.shape[0]}")
    print(f"[mesh-delta-star-v0] stage query vertices  : {stage_query.shape[0]}/{stage_xyz.shape[0]}")
    raw_tree = cKDTree(raw_reference)
    stage_to_raw_distance, _ = query_tree_chunked(raw_tree, stage_query, chunk_size=int(args.query_chunk_size))

    auto_threshold = float(np.percentile(stage_to_raw_distance, float(args.mesh_delta_quantile)))
    threshold = float(args.mesh_delta_distance_threshold)
    if threshold <= 0.0:
        threshold = max(auto_threshold, float(args.mesh_delta_distance_floor))
    delta_mask = stage_to_raw_distance >= threshold
    delta_ids = np.flatnonzero(delta_mask)
    if delta_ids.size == 0:
        raise RuntimeError(f"No mesh delta vertices selected at threshold={threshold:.6f}")
    if int(args.max_delta_points) > 0 and delta_ids.size > int(args.max_delta_points):
        top = np.argsort(-stage_to_raw_distance[delta_ids], kind="stable")[: int(args.max_delta_points)]
        delta_ids = delta_ids[top]
    delta_points = stage_query[delta_ids].astype(np.float32, copy=False)
    delta_magnitude = stage_to_raw_distance[delta_ids].astype(np.float32, copy=False)
    delta_tree = cKDTree(delta_points)

    gs_table, gs_header = read_vertex_table(point_cloud_path)
    gs = gaussian_features(gs_table)
    gs_xyz = gs["xyz"]
    print(f"[mesh-delta-star-v0] gaussian count: {gs_xyz.shape[0]}")
    center_dist_to_delta, center_nearest_delta = query_tree_chunked(delta_tree, gs_xyz, chunk_size=int(args.query_chunk_size))
    if str(args.gaussian_distance_mode) == "major_endpoints":
        endpoint_offset = gs["major_axis_dir"] * (gs["scale_major"] * float(args.major_endpoint_scale))[:, None]
        endpoint_pos = gs_xyz + endpoint_offset
        endpoint_neg = gs_xyz - endpoint_offset
        dist_pos, nearest_pos = query_tree_chunked(delta_tree, endpoint_pos, chunk_size=int(args.query_chunk_size))
        dist_neg, nearest_neg = query_tree_chunked(delta_tree, endpoint_neg, chunk_size=int(args.query_chunk_size))
        use_pos = dist_pos <= dist_neg
        dist_to_delta = np.where(use_pos, dist_pos, dist_neg).astype(np.float32, copy=False)
        nearest_delta = np.where(use_pos, nearest_pos, nearest_neg).astype(np.int64, copy=False)
        distance_query_points = np.where(use_pos[:, None], endpoint_pos, endpoint_neg).astype(np.float32, copy=False)
    else:
        dist_to_delta = center_dist_to_delta
        nearest_delta = center_nearest_delta
        distance_query_points = gs_xyz
    nearest_delta_magnitude = delta_magnitude[nearest_delta]
    dist_to_raw, _ = query_tree_chunked(raw_tree, gs_xyz, chunk_size=int(args.query_chunk_size))

    tangent_angle_to_tangent = np.ones((gs_xyz.shape[0],), dtype=np.float32)
    tangent_distance_factor = np.ones((gs_xyz.shape[0],), dtype=np.float32)
    effective_dist_to_delta = dist_to_delta
    if str(args.tangent_angle_mode) == "raw_pca":
        normals = estimate_pca_normals_chunked(
            raw_tree,
            raw_reference,
            distance_query_points,
            k=int(args.tangent_angle_k),
            chunk_size=int(args.query_chunk_size),
        )
        axis_dot_normal = np.abs(np.sum(gs["major_axis_dir"] * normals, axis=1)).clip(0.0, 1.0)
        tangent_angle_to_tangent = (np.arcsin(axis_dot_normal) / (0.5 * math.pi)).astype(np.float32)
        floor = float(np.clip(args.tangent_angle_distance_floor, 0.0, 1.0))
        tangent_distance_factor = (floor + (1.0 - floor) * tangent_angle_to_tangent).astype(np.float32)
        effective_dist_to_delta = (dist_to_delta * tangent_distance_factor).astype(np.float32)

    if str(args.delta_radius_mode) == "absolute":
        radius = np.full_like(gs["scale_major"], float(args.delta_radius_abs), dtype=np.float32)
    else:
        radius = np.maximum(float(args.delta_radius_abs), gs["scale_major"] * float(args.delta_radius_scale))
    proximity = np.exp(-effective_dist_to_delta / np.maximum(radius, 1e-6)).astype(np.float32)
    mesh_strength = np.log1p(nearest_delta_magnitude / max(threshold, 1e-6)).astype(np.float32)
    aniso_strength = np.log1p(np.maximum(gs["effective_anisotropy"] - 1.0, 0.0)).astype(np.float32)
    opacity_strength = np.sqrt(np.clip(gs["opacity"], 0.0, 1.0)).astype(np.float32)
    raw_offset_strength = np.log1p(dist_to_raw / np.maximum(gs["scale_minor"], 1e-6)).astype(np.float32)
    candidate_score = (proximity * (1.0 + mesh_strength) * (0.35 + aniso_strength) * opacity_strength * (0.5 + 0.25 * raw_offset_strength)).astype(np.float32)

    base_mask = (
        (effective_dist_to_delta <= radius)
        & (gs["opacity"] >= float(args.min_opacity))
        & (gs["effective_anisotropy"] >= float(args.min_anisotropy))
        & (candidate_score >= float(args.min_candidate_score))
    )
    if np.any(base_mask):
        score_threshold = max(float(np.percentile(candidate_score[base_mask], float(args.candidate_score_quantile))), float(args.min_candidate_score))
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
    mesh_delta_preview_path = output_dir / "mesh_delta_vertices_v0.ply"
    saved_payload_path = save_payload(
        payload_path,
        {
            "candidate_mask": candidate_mask,
            "candidate_ids": candidate_ids,
            "base_candidate_mask": base_mask.astype(bool, copy=False),
            "base_candidate_ids": base_candidate_ids,
            "candidate_score": candidate_score,
            "dist_to_delta_mesh": dist_to_delta,
            "effective_dist_to_delta_mesh": effective_dist_to_delta,
            "center_dist_to_delta_mesh": center_dist_to_delta,
            "tangent_angle_to_tangent": tangent_angle_to_tangent,
            "tangent_distance_factor": tangent_distance_factor,
            "nearest_delta_mesh_distance": nearest_delta_magnitude,
            "dist_to_raw_mesh": dist_to_raw,
            "delta_radius_used": radius,
            "mesh_delta_threshold": np.asarray([threshold], dtype=np.float32),
            "opacity": gs["opacity"],
            "scale_major": gs["scale_major"],
            "scale_minor": gs["scale_minor"],
            "major_axis_idx": gs["major_axis_idx"],
            "major_axis_dir": gs["major_axis_dir"],
            "effective_anisotropy": gs["effective_anisotropy"],
            "dc_luma": gs["dc_luma"],
            "rest_norm": gs["rest_norm"],
        },
    )
    write_point_cloud_ply(preview_path, gs_xyz[candidate_ids], candidate_score[candidate_ids], max_points=int(args.preview_max_points))
    write_point_cloud_ply(mesh_delta_preview_path, delta_points, delta_magnitude, max_points=int(args.preview_max_points))

    summary = {
        "mode": "mesh_delta_star_gaussians_v0",
        "raw_mesh_path": str(raw_mesh_path),
        "stage_mesh_path": str(stage_mesh_path),
        "stage_point_cloud_ply": str(point_cloud_path),
        "output_dir": str(output_dir),
        "payload_path": saved_payload_path,
        "candidate_preview_ply": str(preview_path),
        "mesh_delta_preview_ply": str(mesh_delta_preview_path),
        "mesh": {
            "raw_vertex_count": int(raw_header["elements"]["vertex"]["count"]),
            "raw_face_count": int(raw_header["elements"].get("face", {}).get("count", 0)),
            "stage_vertex_count": int(stage_header["elements"]["vertex"]["count"]),
            "stage_face_count": int(stage_header["elements"].get("face", {}).get("count", 0)),
            "stage_vertex_delta": int(stage_header["elements"]["vertex"]["count"] - raw_header["elements"]["vertex"]["count"]),
            "stage_face_delta": int(stage_header["elements"].get("face", {}).get("count", 0) - raw_header["elements"].get("face", {}).get("count", 0)),
            "raw_bbox_min": np.min(raw_reference, axis=0).astype(float).tolist(),
            "raw_bbox_max": np.max(raw_reference, axis=0).astype(float).tolist(),
            "stage_bbox_min": np.min(stage_query, axis=0).astype(float).tolist(),
            "stage_bbox_max": np.max(stage_query, axis=0).astype(float).tolist(),
            "stage_to_raw_distance_sample_stats": stats(stage_to_raw_distance),
            "mesh_delta_threshold": float(threshold),
            "mesh_delta_auto_threshold": float(auto_threshold),
            "mesh_delta_vertex_count": int(delta_ids.size),
            "mesh_delta_vertex_ratio": float(delta_ids.size / max(stage_query.shape[0], 1)),
            "mesh_delta_distance_stats": stats(delta_magnitude),
        },
        "gaussians": {
            "gaussian_count": int(gs_xyz.shape[0]),
            "base_candidate_count": int(np.sum(base_mask)),
            "candidate_score_threshold": float(score_threshold) if np.isfinite(score_threshold) else None,
            "candidate_count": int(candidate_ids.size),
            "candidate_ratio": float(candidate_ids.size / max(gs_xyz.shape[0], 1)),
            "candidate_score_all_stats": stats(candidate_score),
            "candidate_score_selected_stats": stats(candidate_score[candidate_mask]),
            "dist_to_delta_selected_stats": stats(dist_to_delta[candidate_mask]),
            "effective_dist_to_delta_selected_stats": stats(effective_dist_to_delta[candidate_mask]),
            "center_dist_to_delta_selected_stats": stats(center_dist_to_delta[candidate_mask]),
            "tangent_angle_to_tangent_selected_stats": stats(tangent_angle_to_tangent[candidate_mask]),
            "tangent_distance_factor_selected_stats": stats(tangent_distance_factor[candidate_mask]),
            "nearest_delta_distance_selected_stats": stats(nearest_delta_magnitude[candidate_mask]),
            "dist_to_raw_selected_stats": stats(dist_to_raw[candidate_mask]),
            "delta_radius_selected_stats": stats(radius[candidate_mask]),
            "opacity_selected_stats": stats(gs["opacity"][candidate_mask]),
            "anisotropy_selected_stats": stats(gs["effective_anisotropy"][candidate_mask]),
            "scale_major_selected_stats": stats(gs["scale_major"][candidate_mask]),
            "scale_minor_selected_stats": stats(gs["scale_minor"][candidate_mask]),
            "rest_norm_selected_stats": stats(gs["rest_norm"][candidate_mask]),
        },
        "parameters": {
            "max_raw_reference_vertices": int(args.max_raw_reference_vertices),
            "max_stage_query_vertices": int(args.max_stage_query_vertices),
            "max_delta_points": int(args.max_delta_points),
            "mesh_delta_quantile": float(args.mesh_delta_quantile),
            "mesh_delta_distance_threshold": float(args.mesh_delta_distance_threshold),
            "mesh_delta_distance_floor": float(args.mesh_delta_distance_floor),
            "delta_radius_abs": float(args.delta_radius_abs),
            "delta_radius_scale": float(args.delta_radius_scale),
            "delta_radius_mode": str(args.delta_radius_mode),
            "gaussian_distance_mode": str(args.gaussian_distance_mode),
            "major_endpoint_scale": float(args.major_endpoint_scale),
            "tangent_angle_mode": str(args.tangent_angle_mode),
            "tangent_angle_k": int(args.tangent_angle_k),
            "tangent_angle_distance_floor": float(args.tangent_angle_distance_floor),
            "min_opacity": float(args.min_opacity),
            "min_anisotropy": float(args.min_anisotropy),
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
    print(f"[done] mesh delta preview: {mesh_delta_preview_path}")
    print(f"[done] summary          : {summary_path}")


if __name__ == "__main__":
    main()
