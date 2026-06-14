import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
import trimesh
from scipy.spatial import cKDTree


BOUND_BARYCENTRIC_TEMPLATES: Dict[int, np.ndarray] = {
    1: np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32),
    3: np.asarray(
        [
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ],
        dtype=np.float32,
    ),
    4: np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float32,
    ),
    6: np.asarray(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 6.0, 5.0 / 12.0, 5.0 / 12.0],
            [5.0 / 12.0, 1.0 / 6.0, 5.0 / 12.0],
            [5.0 / 12.0, 5.0 / 12.0, 1.0 / 6.0],
        ],
        dtype=np.float32,
    ),
}


FEATURE_SCHEMA = [
    "xyz_norm_x",
    "xyz_norm_y",
    "xyz_norm_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "tangent_u_x",
    "tangent_u_y",
    "tangent_u_z",
    "tangent_v_x",
    "tangent_v_y",
    "tangent_v_z",
    "sqrt_vertex_area_over_diag",
    "mean_edge_over_diag",
    "max_edge_over_diag",
    "normal_consistency",
    "curvature_proxy",
]


def normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def points_to_barycentric(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    safe = np.abs(denom) > 1e-12
    v = np.zeros_like(denom, dtype=np.float32)
    w = np.zeros_like(denom, dtype=np.float32)
    v[safe] = ((d11[safe] * d20[safe] - d01[safe] * d21[safe]) / denom[safe]).astype(np.float32)
    w[safe] = ((d00[safe] * d21[safe] - d01[safe] * d20[safe]) / denom[safe]).astype(np.float32)
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1).astype(np.float32, copy=False)
    bary = np.clip(bary, 0.0, 1.0)
    bary_sum = np.sum(bary, axis=1, keepdims=True)
    degenerate = bary_sum[:, 0] <= 1e-8
    bary = bary / np.clip(bary_sum, 1e-8, None)
    if np.any(degenerate):
        bary[degenerate] = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    return bary


def stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def load_triangle_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load triangle mesh from {mesh_path}")


def mesh_bbox_normalizer(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    center = ((bounds[0] + bounds[1]) * 0.5).astype(np.float32)
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    return center, max(diag, 1e-6)


def compute_vertex_geometry(mesh: trimesh.Trimesh, bbox_center: np.ndarray | None = None, bbox_diag: float | None = None) -> Dict[str, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("Mesh must contain vertices and faces.")

    if bbox_center is None or bbox_diag is None:
        bbox_center, bbox_diag = mesh_bbox_normalizer(mesh)
    bbox_center = np.asarray(bbox_center, dtype=np.float32).reshape(3)
    bbox_diag = float(max(bbox_diag, 1e-6))

    face_normals = normalize_np(np.asarray(mesh.face_normals, dtype=np.float32))
    vertex_normals = normalize_np(np.asarray(mesh.vertex_normals, dtype=np.float32))
    face_areas = np.asarray(mesh.area_faces, dtype=np.float32)
    vertex_count = int(vertices.shape[0])

    vertex_area = np.zeros((vertex_count,), dtype=np.float32)
    np.add.at(vertex_area, faces.reshape(-1), np.repeat(face_areas / 3.0, 3))

    face_vertex_normals = vertex_normals[faces]
    normal_dots = np.sum(face_vertex_normals * face_normals[:, None, :], axis=2)
    normal_dot_sum = np.zeros((vertex_count,), dtype=np.float32)
    normal_dot_count = np.zeros((vertex_count,), dtype=np.float32)
    np.add.at(normal_dot_sum, faces.reshape(-1), normal_dots.reshape(-1).astype(np.float32))
    np.add.at(normal_dot_count, faces.reshape(-1), 1.0)
    normal_consistency = normal_dot_sum / np.clip(normal_dot_count, 1.0, None)
    normal_consistency = np.clip(normal_consistency, -1.0, 1.0).astype(np.float32)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_vec = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    edge_len = np.linalg.norm(edge_vec, axis=1).astype(np.float32)
    edge_sum = np.zeros((vertex_count,), dtype=np.float32)
    edge_count = np.zeros((vertex_count,), dtype=np.float32)
    edge_max = np.zeros((vertex_count,), dtype=np.float32)
    for column in (0, 1):
        ids = edges[:, column]
        np.add.at(edge_sum, ids, edge_len)
        np.add.at(edge_count, ids, 1.0)
        np.maximum.at(edge_max, ids, edge_len)
    mean_edge = edge_sum / np.clip(edge_count, 1.0, None)

    axis_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    axis_y = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    base_axis = np.tile(axis_x[None], (vertex_count, 1))
    parallel = np.abs(np.sum(base_axis * vertex_normals, axis=1)) > 0.9
    base_axis[parallel] = axis_y
    tangent_u = base_axis - np.sum(base_axis * vertex_normals, axis=1, keepdims=True) * vertex_normals
    tangent_u = normalize_np(tangent_u)
    tangent_v = normalize_np(np.cross(vertex_normals, tangent_u))
    tangent_u = normalize_np(np.cross(tangent_v, vertex_normals))

    xyz_norm = (vertices - bbox_center[None, :]) / bbox_diag
    sqrt_area = np.sqrt(np.clip(vertex_area, 0.0, None)) / bbox_diag
    curvature_proxy = 1.0 - np.clip(normal_consistency, 0.0, 1.0)
    features = np.concatenate(
        [
            xyz_norm.astype(np.float32),
            vertex_normals.astype(np.float32),
            tangent_u.astype(np.float32),
            tangent_v.astype(np.float32),
            sqrt_area[:, None].astype(np.float32),
            (mean_edge / bbox_diag)[:, None].astype(np.float32),
            (edge_max / bbox_diag)[:, None].astype(np.float32),
            normal_consistency[:, None].astype(np.float32),
            curvature_proxy[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vertex_normals.astype(np.float32),
        "tangent_u": tangent_u.astype(np.float32),
        "tangent_v": tangent_v.astype(np.float32),
        "vertex_area": vertex_area.astype(np.float32),
        "mean_edge": mean_edge.astype(np.float32),
        "max_edge": edge_max.astype(np.float32),
        "normal_consistency": normal_consistency.astype(np.float32),
        "curvature_proxy": curvature_proxy.astype(np.float32),
        "features": features,
        "bbox_center": bbox_center.astype(np.float32),
        "bbox_diag": np.asarray([bbox_diag], dtype=np.float32),
    }


def sample_reference_surface(mesh: trimesh.Trimesh, sample_count: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, int(sample_count))
    finally:
        np.random.set_state(state)
    normals = normalize_np(np.asarray(mesh.face_normals, dtype=np.float32)[face_ids])
    return points.astype(np.float32), normals.astype(np.float32)


def build_hr_targets(
    lr_geometry: Dict[str, np.ndarray],
    hr_mesh: trimesh.Trimesh,
    reference_sample_count: int,
    strong_threshold: float,
    weak_threshold: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    hr_points, hr_normals = sample_reference_surface(hr_mesh, reference_sample_count, seed)
    tree = cKDTree(hr_points)
    distances, nearest_ids = tree.query(lr_geometry["vertices"], k=1)
    nearest_points = hr_points[nearest_ids].astype(np.float32)
    nearest_normals = hr_normals[nearest_ids].astype(np.float32)

    delta_world = nearest_points - lr_geometry["vertices"]
    tangent_u = lr_geometry["tangent_u"]
    tangent_v = lr_geometry["tangent_v"]
    normals = lr_geometry["normals"]
    bbox_diag = float(lr_geometry["bbox_diag"][0])
    delta_local = np.stack(
        [
            np.sum(delta_world * tangent_u, axis=1),
            np.sum(delta_world * tangent_v, axis=1),
            np.sum(delta_world * normals, axis=1),
        ],
        axis=1,
    ).astype(np.float32) / max(bbox_diag, 1e-6)

    distances = distances.astype(np.float32)
    reliability = np.zeros_like(distances, dtype=np.uint8)
    reliability[distances <= float(weak_threshold)] = 1
    reliability[distances <= float(strong_threshold)] = 2
    train_weight = np.zeros_like(distances, dtype=np.float32)
    train_weight[reliability == 1] = 0.35
    train_weight[reliability == 2] = 1.0
    conf_target = np.zeros_like(distances, dtype=np.float32)
    weak_span = max(float(weak_threshold) - float(strong_threshold), 1e-8)
    weak = reliability == 1
    strong = reliability == 2
    conf_target[strong] = 1.0
    conf_target[weak] = 1.0 - np.clip((distances[weak] - float(strong_threshold)) / weak_span, 0.0, 1.0)

    normal_alignment = np.sum(nearest_normals * normals, axis=1).astype(np.float32)
    return {
        "nearest_hr_points": nearest_points,
        "nearest_hr_normals": nearest_normals,
        "target_delta_local": delta_local.astype(np.float32),
        "target_distance": distances,
        "target_reliability": reliability,
        "target_train_weight": train_weight,
        "target_confidence": conf_target,
        "target_normal_alignment": normal_alignment,
    }


class SOFMeshPatchEnhancerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = []
        dim = int(in_dim)
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 4)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.head(self.backbone(features))
        return {
            "delta_local": raw[:, :3],
            "confidence_logit": raw[:, 3],
        }


def local_delta_to_world(
    delta_local: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    normals: np.ndarray,
    bbox_diag: float,
) -> np.ndarray:
    delta = delta_local.astype(np.float32) * float(bbox_diag)
    return (
        delta[:, 0:1] * tangent_u.astype(np.float32)
        + delta[:, 1:2] * tangent_v.astype(np.float32)
        + delta[:, 2:3] * normals.astype(np.float32)
    ).astype(np.float32)


@torch.no_grad()
def predict_offsets(
    model: nn.Module,
    features_np: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    normals: np.ndarray,
    bbox_diag: float,
    device: str,
    batch_size: int,
    max_delta_ratio: float,
) -> Dict[str, np.ndarray]:
    model.eval()
    features = torch.from_numpy(features_np.astype(np.float32))
    pred_delta_chunks = []
    pred_conf_chunks = []
    for start in range(0, features.shape[0], int(batch_size)):
        batch = features[start : start + int(batch_size)].to(device=device)
        out = model(batch)
        pred_delta_chunks.append(out["delta_local"].detach().cpu())
        pred_conf_chunks.append(torch.sigmoid(out["confidence_logit"]).detach().cpu())
    delta_local = torch.cat(pred_delta_chunks, dim=0).numpy().astype(np.float32)
    confidence = torch.cat(pred_conf_chunks, dim=0).numpy().astype(np.float32)

    if float(max_delta_ratio) > 0.0:
        norm = np.linalg.norm(delta_local, axis=1, keepdims=True)
        scale = np.minimum(1.0, float(max_delta_ratio) / np.clip(norm, 1e-8, None))
        delta_local = delta_local * scale.astype(np.float32)
    delta_world = local_delta_to_world(delta_local, tangent_u, tangent_v, normals, bbox_diag)
    return {
        "delta_local": delta_local,
        "delta_world": delta_world,
        "confidence": confidence,
    }


def colorize_confidence(confidence: np.ndarray) -> np.ndarray:
    conf = np.clip(confidence.reshape(-1), 0.0, 1.0)
    colors = np.stack(
        [
            np.round(255.0 * (1.0 - conf)),
            np.round(220.0 * conf),
            np.round(80.0 * (1.0 - conf)),
            np.full_like(conf, 255.0),
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def _face_frames(mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    normals = normalize_np(np.asarray(mesh.face_normals, dtype=np.float32))
    tangent_u = normalize_np(triangles[:, 1] - triangles[:, 0])
    tangent_v = normalize_np(np.cross(normals, tangent_u))
    tangent_u = normalize_np(np.cross(tangent_v, normals))
    face_centers = triangles.mean(axis=1, keepdims=True)
    offsets = triangles - face_centers
    scale_u = np.max(np.abs(np.sum(offsets * tangent_u[:, None, :], axis=-1)), axis=1).astype(np.float32)
    scale_v = np.max(np.abs(np.sum(offsets * tangent_v[:, None, :], axis=-1)), axis=1).astype(np.float32)
    return {
        "triangles": triangles,
        "normals": normals,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "scale_u": scale_u,
        "scale_v": scale_v,
    }


def build_carrier_payload_from_mesh(
    mesh: trimesh.Trimesh,
    vertex_confidence: np.ndarray,
    carriers_per_face: int,
    thickness_scale: float,
    min_confidence: float,
) -> Dict[str, np.ndarray]:
    if carriers_per_face not in BOUND_BARYCENTRIC_TEMPLATES:
        raise ValueError(f"Unsupported carriers_per_face={carriers_per_face}; expected one of {sorted(BOUND_BARYCENTRIC_TEMPLATES)}")
    frames = _face_frames(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bary_template = BOUND_BARYCENTRIC_TEMPLATES[int(carriers_per_face)]
    face_count = int(faces.shape[0])
    triangles = frames["triangles"]

    centers = (triangles[:, None] * bary_template[None, :, :, None]).sum(axis=2).reshape(-1, 3)
    face_ids = np.repeat(np.arange(face_count, dtype=np.int64), int(carriers_per_face))
    bary_coords = np.tile(bary_template[None, :, :], (face_count, 1, 1)).reshape(-1, 3).astype(np.float32)
    normals = np.repeat(frames["normals"], int(carriers_per_face), axis=0)
    tangent_u = np.repeat(frames["tangent_u"], int(carriers_per_face), axis=0)
    tangent_v = np.repeat(frames["tangent_v"], int(carriers_per_face), axis=0)
    scale_u = np.repeat(frames["scale_u"], int(carriers_per_face)).astype(np.float32)
    scale_v = np.repeat(frames["scale_v"], int(carriers_per_face)).astype(np.float32)
    scale_n = np.minimum(scale_u, scale_v) * float(thickness_scale)

    face_conf = np.mean(np.clip(vertex_confidence[faces], 0.0, 1.0), axis=1).astype(np.float32)
    confidence = np.repeat(face_conf, int(carriers_per_face)).astype(np.float32)
    valid_mask = confidence >= float(min_confidence)

    return {
        "centers": centers.astype(np.float32),
        "normals": normals.astype(np.float32),
        "tangent_u": tangent_u.astype(np.float32),
        "tangent_v": tangent_v.astype(np.float32),
        "scale_u": scale_u.astype(np.float32),
        "scale_v": scale_v.astype(np.float32),
        "scale_n": scale_n.astype(np.float32),
        "fused_rgb": np.zeros((centers.shape[0], 3), dtype=np.float32),
        "confidence": confidence.astype(np.float32),
        "disagreement": (1.0 - confidence).astype(np.float32),
        "view_count": np.ones((centers.shape[0],), dtype=np.int32),
        "valid_mask": valid_mask.astype(np.bool_),
        "face_ids": face_ids.astype(np.int64),
        "bary_coords": bary_coords.astype(np.float32),
    }


def _payload_to_numpy(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


def load_payload_any(path: str) -> Dict[str, np.ndarray]:
    payload_path = Path(path)
    if payload_path.suffix.lower() == ".npz":
        loaded = np.load(payload_path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    raw = torch.load(payload_path, map_location="cpu")
    if "carrier_payload" in raw and isinstance(raw["carrier_payload"], dict):
        raw = raw["carrier_payload"]
    return _payload_to_numpy(raw)


def rebind_carrier_payload_to_mesh(payload: Dict[str, np.ndarray], mesh: trimesh.Trimesh, thickness_scale: float | None = None) -> Dict[str, np.ndarray]:
    if "face_ids" not in payload or "bary_coords" not in payload:
        raise KeyError("Carrier payload needs 'face_ids' and 'bary_coords' to follow a moved mesh.")
    face_ids = np.asarray(payload["face_ids"], dtype=np.int64).reshape(-1)
    bary = np.asarray(payload["bary_coords"], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if np.any((face_ids < 0) | (face_ids >= faces.shape[0])):
        raise ValueError("Carrier payload face_ids are outside the target mesh face range.")

    frames = _face_frames(mesh)
    triangles = frames["triangles"][face_ids]
    out = dict(payload)
    out["centers"] = np.sum(triangles * bary[:, :, None], axis=1).astype(np.float32)
    out["normals"] = frames["normals"][face_ids].astype(np.float32)
    out["tangent_u"] = frames["tangent_u"][face_ids].astype(np.float32)
    out["tangent_v"] = frames["tangent_v"][face_ids].astype(np.float32)
    out["scale_u"] = frames["scale_u"][face_ids].astype(np.float32)
    out["scale_v"] = frames["scale_v"][face_ids].astype(np.float32)
    if thickness_scale is not None or "scale_n" not in out:
        scale = 0.05 if thickness_scale is None else float(thickness_scale)
        out["scale_n"] = (np.minimum(out["scale_u"], out["scale_v"]) * scale).astype(np.float32)
    return out


def save_payload_npz(path: str | Path, payload: Dict[str, np.ndarray]):
    serializable = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.dtype.kind in {"O", "U"}:
            continue
        serializable[key] = arr
    np.savez_compressed(path, **serializable)


def write_json(path: str | Path, payload: Dict[str, Any]):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
