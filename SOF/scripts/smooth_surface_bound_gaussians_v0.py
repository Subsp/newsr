from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_soflr_vggt_bound_gs_correction_v0 import bind_gaussians_to_mesh
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import copy_render_config, load_model_ply, load_train_cameras_only, resolve_iteration, select_uniform
from utils.general_utils import build_rotation, inverse_sigmoid
from utils.sh_utils import SH2RGB
from utils.sof_mesh_patch_enhancer_v0 import load_triangle_mesh, stats_from_array
from utils.system_utils import mkdir_p


def features_dc_to_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    if features_dc.ndim == 3:
        if features_dc.shape[1] == 1:
            sh_dc = features_dc[:, 0, :]
        elif features_dc.shape[2] == 1:
            sh_dc = features_dc[:, :, 0]
        else:
            sh_dc = features_dc.reshape(features_dc.shape[0], -1)[:, :3]
    elif features_dc.ndim == 2:
        sh_dc = features_dc[:, :3]
    else:
        raise ValueError(f"Unsupported features_dc shape: {tuple(features_dc.shape)}")
    return torch.nan_to_num(SH2RGB(sh_dc), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def face_edge_mean_lengths(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    e01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    return ((e01 + e12 + e20) / 3.0).astype(np.float32, copy=False)


def load_or_build_binding(
    xyz: np.ndarray,
    mesh,
    *,
    surface_binding_payload: str,
    face_k: int,
    bind_chunk_size: int,
) -> Dict[str, np.ndarray]:
    if str(surface_binding_payload).strip():
        payload_path = Path(surface_binding_payload).expanduser().resolve()
        payload = torch.load(payload_path, map_location="cpu")
        required = ["face_ids", "surface_points", "normals", "tangent_u", "tangent_v", "u0", "v0", "d0"]
        if all(key in payload for key in required) and int(payload["surface_points"].shape[0]) == int(xyz.shape[0]):
            return {
                "face_ids": np.asarray(payload["face_ids"], dtype=np.int64),
                "surface_points": np.asarray(payload["surface_points"], dtype=np.float32),
                "normals": np.asarray(payload["normals"], dtype=np.float32),
                "tangent_u": np.asarray(payload["tangent_u"], dtype=np.float32),
                "tangent_v": np.asarray(payload["tangent_v"], dtype=np.float32),
                "normal_offset": np.asarray(payload["d0"], dtype=np.float32),
                "tangent_offset_u": np.asarray(payload["u0"], dtype=np.float32),
                "tangent_offset_v": np.asarray(payload["v0"], dtype=np.float32),
            }
        print(f"[surface-smooth] binding payload missing required keys or size mismatch, rebuilding: {payload_path}", flush=True)
    return bind_gaussians_to_mesh(xyz, mesh, face_k=int(face_k), chunk_size=int(bind_chunk_size))


def compute_camera_pixel_tau(
    surface_points: np.ndarray,
    *,
    scene_root: Path | None,
    model_path: Path,
    images_subdir: str,
    max_camera_views: int,
    depth_min: float,
    chunk_size: int,
) -> np.ndarray | None:
    if scene_root is None:
        return None
    cameras = load_train_cameras_only(scene_root, model_path, images_subdir)
    cameras = select_uniform(cameras, int(max_camera_views))
    if not cameras:
        return None

    total = int(surface_points.shape[0])
    accum: List[np.ndarray] = []
    for camera in tqdm(cameras, desc="camera scale tau"):
        r = np.asarray(camera.R, dtype=np.float32)
        t = np.asarray(camera.T, dtype=np.float32)
        focal = max((float(camera.focal_x) + float(camera.focal_y)) * 0.5, 1e-6)
        values = np.full((total,), np.nan, dtype=np.float32)
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            xyz_cam = surface_points[start:end] @ r + t[None, :]
            z = xyz_cam[:, 2]
            valid = z > float(depth_min)
            values[start:end][valid] = (z[valid] / focal).astype(np.float32, copy=False)
        accum.append(values)

    stacked = np.stack(accum, axis=0)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(stacked, axis=0).astype(np.float32, copy=False)
    median[~np.isfinite(median)] = 0.0
    return median


def build_face_adjacency(mesh) -> set[tuple[int, int]]:
    adjacency: set[tuple[int, int]] = set()
    for a, b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        aa, bb = int(a), int(b)
        adjacency.add((aa, bb))
        adjacency.add((bb, aa))
    return adjacency


def build_surface_graph(
    surface_points: np.ndarray,
    normals: np.ndarray,
    face_ids: np.ndarray,
    rgb: np.ndarray,
    tau_surface: np.ndarray,
    confidence: np.ndarray,
    allowed: np.ndarray,
    mesh,
    *,
    k_neighbors: int,
    radius_multiplier: float,
    normal_angle_cut_deg: float,
    sigma_angle_deg: float,
    sigma_color: float,
    require_face_adjacency: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    tree = cKDTree(surface_points)
    query_k = max(int(k_neighbors) * 4 + 1, int(k_neighbors) + 1)
    query_k = min(query_k, int(surface_points.shape[0]))
    _, candidate_ids = tree.query(surface_points, k=query_k, workers=1)
    if candidate_ids.ndim == 1:
        candidate_ids = candidate_ids[:, None]

    face_adjacency = build_face_adjacency(mesh) if bool(require_face_adjacency) else set()
    cos_cut = math.cos(math.radians(float(normal_angle_cut_deg)))
    sigma_angle = max(math.radians(float(sigma_angle_deg)), 1e-6)
    sigma_color = max(float(sigma_color), 1e-6)
    neighbors: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    for i in tqdm(range(int(surface_points.shape[0])), desc="build surface graph"):
        if not bool(allowed[i]):
            neighbors.append(np.empty((0,), dtype=np.int64))
            weights.append(np.empty((0,), dtype=np.float32))
            continue

        n_ids: List[int] = []
        w_vals: List[float] = []
        for raw_j in candidate_ids[i]:
            j = int(raw_j)
            if j == i or not bool(allowed[j]):
                continue
            if bool(require_face_adjacency):
                same_face = int(face_ids[i]) == int(face_ids[j])
                adjacent = (int(face_ids[i]), int(face_ids[j])) in face_adjacency
                if not same_face and not adjacent:
                    continue
            normal_dot = float(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0))
            if normal_dot < cos_cut:
                continue
            dist = float(np.linalg.norm(surface_points[i] - surface_points[j]))
            radius = float(radius_multiplier) * math.sqrt(max(float(tau_surface[i] * tau_surface[j]), 1e-12))
            if dist > radius:
                continue
            angle = math.acos(normal_dot)
            color_diff = float(np.linalg.norm(rgb[i] - rgb[j]))
            dist_w = math.exp(-(dist * dist) / max(2.0 * radius * radius, 1e-12))
            angle_w = math.exp(-(angle * angle) / (2.0 * sigma_angle * sigma_angle))
            color_w = math.exp(-(color_diff * color_diff) / (2.0 * sigma_color * sigma_color))
            conf_w = min(float(confidence[i]), float(confidence[j]))
            weight = dist_w * angle_w * color_w * conf_w
            if weight <= 1e-8:
                continue
            n_ids.append(j)
            w_vals.append(weight)
            if len(n_ids) >= int(k_neighbors):
                break

        neighbors.append(np.asarray(n_ids, dtype=np.int64))
        weights.append(np.asarray(w_vals, dtype=np.float32))
    return neighbors, weights


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size <= 0 or weights.size <= 0 or float(weights.sum()) <= 0.0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    index = min(max(index, 0), int(sorted_values.size) - 1)
    return float(sorted_values[index])


def neighbor_weighted_medians(values: np.ndarray, neighbors: List[np.ndarray], weights: List[np.ndarray]) -> np.ndarray:
    medians = values.copy()
    for i, (nbr, weight) in enumerate(zip(neighbors, weights)):
        if nbr.size <= 0:
            continue
        median = weighted_median(values[nbr], weight)
        if math.isfinite(median):
            medians[i] = median
    return medians


def graph_laplacian_energy(values: np.ndarray, neighbors: List[np.ndarray], weights: List[np.ndarray]) -> float:
    numerator = 0.0
    denom = 0.0
    for i, (nbr, weight) in enumerate(zip(neighbors, weights)):
        if nbr.size <= 0:
            continue
        diff = values[i] - values[nbr]
        numerator += float(np.sum(weight * diff * diff))
        denom += float(np.sum(weight))
    return float(numerator / max(denom, 1e-12))


def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], min=1e-12))
    qx = torch.sign(m[:, 2, 1] - m[:, 1, 2]) * 0.5 * torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=0.0))
    qy = torch.sign(m[:, 0, 2] - m[:, 2, 0]) * 0.5 * torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=0.0))
    qz = torch.sign(m[:, 1, 0] - m[:, 0, 1]) * 0.5 * torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=0.0))
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    return torch.nn.functional.normalize(quat, dim=1)


def covariance_sigma_normal(
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot = build_rotation(rotation)
    scales = torch.exp(scaling)
    normal_local = torch.bmm(rot.transpose(1, 2), normals[:, :, None]).squeeze(2)
    sigma2 = torch.sum(normal_local * normal_local * scales * scales, dim=1).clamp_min(1e-12)
    cov = rot @ torch.diag_embed(scales * scales) @ rot.transpose(1, 2)
    return torch.sqrt(sigma2), cov


def gaussian_major_axis(
    model: GaussianModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = model.get_scaling.detach()
    count = int(scale.shape[0])
    if isinstance(model.filter_3D, torch.Tensor) and model.filter_3D.ndim > 0:
        filter_3d = model.filter_3D.detach().to(device=scale.device, dtype=scale.dtype)
        if int(filter_3d.shape[0]) == count:
            filter_3d = filter_3d.reshape(count, -1)
        else:
            filter_3d = filter_3d.reshape(1, -1).expand(count, -1)
        if int(filter_3d.shape[1]) == 1:
            filter_3d = filter_3d.expand(-1, int(scale.shape[1]))
        elif int(filter_3d.shape[1]) != int(scale.shape[1]):
            filter_3d = filter_3d[:, :1].expand(-1, int(scale.shape[1]))
    else:
        filter_3d = torch.zeros_like(scale)
    effective_scale = torch.sqrt(torch.square(scale) + torch.square(filter_3d))
    major_axis_idx = torch.argmax(effective_scale, dim=1)
    major_scale = torch.gather(effective_scale, 1, major_axis_idx[:, None]).reshape(-1)
    rotations = build_rotation(model._rotation.detach())
    major_axis = torch.gather(rotations, 2, major_axis_idx[:, None, None].expand(-1, 3, 1)).squeeze(2)
    major_axis = torch.nn.functional.normalize(major_axis, dim=1)
    return (
        major_axis.detach().cpu().numpy().astype(np.float32, copy=False),
        major_scale.detach().cpu().numpy().astype(np.float32, copy=False),
        major_axis_idx.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def parse_body_probe_samples(value: str) -> np.ndarray:
    samples: List[float] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        samples.append(float(raw))
    if not samples:
        samples = [-1.0, 0.0, 1.0]
    if not any(abs(sample) <= 1e-6 for sample in samples):
        samples.append(0.0)
    samples = sorted(set(float(sample) for sample in samples))
    return np.asarray(samples, dtype=np.float32)


def compute_body_probe(
    *,
    xyz: np.ndarray,
    major_axis_dir: np.ndarray,
    scale_major: np.ndarray,
    center_normals: np.ndarray,
    center_face_ids: np.ndarray,
    center_smooth_allowed: np.ndarray,
    sigma_normal_norm: np.ndarray,
    tau_surface: np.ndarray,
    pixel_tau: np.ndarray | None,
    face_edge: np.ndarray,
    mesh,
    args,
) -> Dict[str, np.ndarray]:
    samples = parse_body_probe_samples(str(args.body_probe_samples))
    count = int(xyz.shape[0])
    sample_count = int(samples.shape[0])
    offsets = (
        major_axis_dir[:, None, :]
        * scale_major[:, None, None]
        * samples[None, :, None]
        * float(args.body_major_scale)
    )
    probe_points = (xyz[:, None, :] + offsets).reshape(-1, 3).astype(np.float32, copy=False)
    probe_binding = bind_gaussians_to_mesh(
        probe_points,
        mesh,
        face_k=int(args.body_face_k) if int(args.body_face_k) > 0 else int(args.face_k),
        chunk_size=int(args.bind_chunk_size),
    )
    probe_distance = np.asarray(probe_binding["surface_distance"], dtype=np.float32).reshape(count, sample_count)
    probe_face_ids = np.asarray(probe_binding["face_ids"], dtype=np.int64).reshape(count, sample_count)
    probe_normals = np.asarray(probe_binding["normals"], dtype=np.float32).reshape(count, sample_count, 3)

    sample_edge_tau = float(args.k_edge) * face_edge[probe_face_ids]
    if pixel_tau is None:
        sample_tau = np.maximum(sample_edge_tau, np.min(tau_surface)).astype(np.float32, copy=False)
    else:
        sample_tau = np.maximum(sample_edge_tau, float(args.k_px) * pixel_tau[:, None]).astype(np.float32, copy=False)
    sample_tau = np.maximum(sample_tau, np.min(tau_surface)).astype(np.float32, copy=False)
    body_d_norm = probe_distance / np.clip(sample_tau, 1e-8, None)

    normal_dot = np.sum(probe_normals * center_normals[:, None, :], axis=2)
    normal_dot = np.clip(normal_dot, -1.0, 1.0).astype(np.float32, copy=False)
    normal_cut = math.cos(math.radians(float(args.body_normal_angle_cut_deg)))
    sample_valid = (body_d_norm <= float(args.body_d_norm_threshold)) & (normal_dot >= normal_cut)

    center_ids = np.flatnonzero(np.abs(samples) <= 1e-6)
    center_sample_valid = sample_valid[:, center_ids[0]] if center_ids.size else center_smooth_allowed
    center_valid = center_smooth_allowed & center_sample_valid

    max_abs_sample = float(np.max(np.abs(samples))) if sample_count else 0.0
    endpoint_ids = np.flatnonzero(np.abs(np.abs(samples) - max_abs_sample) <= 1e-6)
    if endpoint_ids.size:
        endpoint_distance_norm = body_d_norm[:, endpoint_ids]
        endpoint_valid = np.all(
            (body_d_norm[:, endpoint_ids] <= float(args.body_endpoint_d_norm_threshold))
            & (normal_dot[:, endpoint_ids] >= normal_cut),
            axis=1,
        )
    else:
        endpoint_distance_norm = np.zeros((count, 0), dtype=np.float32)
        endpoint_valid = center_valid.copy()

    sample_weight = np.exp(-0.5 * samples * samples).astype(np.float32)
    sample_weight /= max(float(sample_weight.sum()), 1e-8)
    body_valid_ratio = np.sum(sample_valid.astype(np.float32) * sample_weight[None, :], axis=1).astype(np.float32)
    body_d_p90 = np.percentile(body_d_norm, 90, axis=1).astype(np.float32, copy=False)
    body_d_max = np.max(body_d_norm, axis=1).astype(np.float32, copy=False)
    normal_switch_count = np.sum(normal_dot < normal_cut, axis=1).astype(np.int16, copy=False)
    face_switch_count = np.sum(probe_face_ids != center_face_ids[:, None], axis=1).astype(np.int16, copy=False)

    sigma_thr = float(args.body_sigma_norm_threshold)
    sigma_ok = np.ones((count,), dtype=bool) if sigma_thr <= 0.0 else sigma_normal_norm <= sigma_thr
    body_good = (
        center_valid
        & endpoint_valid
        & (body_valid_ratio >= float(args.body_valid_ratio_threshold))
        & (body_d_p90 <= float(args.body_d_norm_threshold))
        & (normal_switch_count <= int(args.body_max_normal_switch_count))
        & sigma_ok
    )

    label = np.full((count,), 4, dtype=np.int16)
    all_invalid = body_valid_ratio <= float(args.body_all_invalid_ratio)
    label[all_invalid | (~center_valid)] = 3
    multi_surface = center_valid & (normal_switch_count > int(args.body_max_normal_switch_count))
    label[multi_surface] = 2
    endpoint_bad = center_valid & (~endpoint_valid) & (~multi_surface)
    label[endpoint_bad] = 1
    label[body_good] = 0

    return {
        "samples": samples,
        "body_center_valid": center_valid.astype(bool, copy=False),
        "body_endpoint_valid": endpoint_valid.astype(bool, copy=False),
        "body_good": body_good.astype(bool, copy=False),
        "body_valid_ratio": body_valid_ratio,
        "body_d_p90": body_d_p90,
        "body_d_max": body_d_max,
        "body_normal_switch_count": normal_switch_count,
        "body_face_switch_count": face_switch_count,
        "body_case_label": label,
        "body_endpoint_distance_norm": endpoint_distance_norm.astype(np.float32, copy=False),
    }


def replace_normal_variance(
    cov: torch.Tensor,
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
    normals: torch.Tensor,
    sigma_old: torch.Tensor,
    sigma_new: torch.Tensor,
    *,
    cross_decay_power: float,
    min_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    basis = torch.stack([tangent_u, tangent_v, normals], dim=2)
    cov_surface = basis.transpose(1, 2) @ cov @ basis
    sigma_new2 = sigma_new * sigma_new
    cov_surface_new = cov_surface.clone()
    cov_surface_new[:, 2, 2] = sigma_new2
    decay = torch.clamp(sigma_new / torch.clamp(sigma_old, min=1e-8), max=1.0) ** float(cross_decay_power)
    cov_surface_new[:, 0, 2] *= decay
    cov_surface_new[:, 2, 0] *= decay
    cov_surface_new[:, 1, 2] *= decay
    cov_surface_new[:, 2, 1] *= decay
    cov_surface_new = 0.5 * (cov_surface_new + cov_surface_new.transpose(1, 2))
    cov_new = basis @ cov_surface_new @ basis.transpose(1, 2)
    evals, evecs = torch.linalg.eigh(cov_new)
    evals = torch.clamp(evals, min=float(min_scale) ** 2)
    det = torch.linalg.det(evecs)
    flip = det < 0.0
    if torch.any(flip):
        evecs[flip, :, 0] *= -1.0
    scales = torch.sqrt(evals)
    quat = matrix_to_quaternion_wxyz(evecs)
    return torch.log(torch.clamp(scales, min=float(min_scale))), quat


def summarize_value(prefix: str, value: np.ndarray) -> Dict[str, float]:
    finite = value[np.isfinite(value)]
    return {f"{prefix}_{key}": val for key, val in stats_from_array(finite.astype(np.float32, copy=False)).items()}


def save_checkpoint(model: GaussianModel, output_path: Path, iteration: int) -> None:
    point_dir = output_path / "point_cloud" / f"iteration_{int(iteration)}"
    mkdir_p(str(point_dir))
    model.save_ply(str(point_dir / "point_cloud.ply"))
    model.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))


def main() -> None:
    parser = ArgumentParser(description="Static surface-frame smoothing for existing SOF Gaussians.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_path", "--out_path", dest="output_path", required=True)
    parser.add_argument("--scene_root", "--cameras_path", dest="scene_root", default="")
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--mode", choices=["delta_only", "sigma_only", "delta_sigma", "delta_sigma_taucap"], default="delta_sigma")
    parser.add_argument("--surface_binding_payload", default="")
    parser.add_argument("--face_k", type=int, default=8)
    parser.add_argument("--bind_chunk_size", type=int, default=16384)
    parser.add_argument("--max_camera_views", type=int, default=64)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--k_px", type=float, default=2.5)
    parser.add_argument("--k_edge", type=float, default=0.4)
    parser.add_argument("--tau_floor", type=float, default=0.0)
    parser.add_argument("--tau_floor_scene_scale", type=float, default=1e-5)
    parser.add_argument("--sigma_d", type=float, default=1.5)
    parser.add_argument("--sigma_n", type=float, default=1.5)
    parser.add_argument("--min_surface_conf", type=float, default=0.25)
    parser.add_argument("--max_d_norm", type=float, default=3.0)
    parser.add_argument("--k_neighbors", type=int, default=12)
    parser.add_argument("--radius_px", type=float, default=3.0)
    parser.add_argument("--normal_angle_cut_deg", type=float, default=30.0)
    parser.add_argument("--sigma_angle_deg", type=float, default=15.0)
    parser.add_argument("--sigma_color", type=float, default=0.15)
    parser.add_argument("--require_face_adjacency", type=int, default=1)
    parser.add_argument("--smooth_delta_lambda", type=float, default=0.4)
    parser.add_argument("--smooth_sigma_lambda", type=float, default=0.4)
    parser.add_argument("--max_delta_update_px", type=float, default=1.0)
    parser.add_argument("--sigma_normal_cap_px", type=float, default=0.75)
    parser.add_argument("--cross_decay_power", type=float, default=1.0)
    parser.add_argument("--min_scale", type=float, default=1e-6)
    parser.add_argument("--enable_tau_cap", type=int, default=0)
    parser.add_argument("--tau_cap_mad_k", type=float, default=3.0)
    parser.add_argument("--enable_body_probe", type=int, default=1)
    parser.add_argument("--body_probe_samples", default="-1.0,-0.5,0.0,0.5,1.0")
    parser.add_argument("--body_major_scale", type=float, default=1.0)
    parser.add_argument("--body_face_k", type=int, default=0)
    parser.add_argument("--body_valid_ratio_threshold", type=float, default=0.8)
    parser.add_argument("--body_d_norm_threshold", type=float, default=2.0)
    parser.add_argument("--body_endpoint_d_norm_threshold", type=float, default=2.0)
    parser.add_argument("--body_normal_angle_cut_deg", type=float, default=30.0)
    parser.add_argument("--body_sigma_norm_threshold", type=float, default=0.0)
    parser.add_argument("--body_all_invalid_ratio", type=float, default=0.2)
    parser.add_argument("--body_max_normal_switch_count", type=int, default=0)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve() if str(args.scene_root).strip() else None
    iteration = resolve_iteration(model_path, int(args.iteration))

    output_path.mkdir(parents=True, exist_ok=True)
    copy_render_config(model_path, output_path)
    model = load_model_ply(model_path, iteration, sh_degree=3)
    mesh = load_triangle_mesh(str(mesh_path))

    xyz_before = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    scene_diag = float(np.linalg.norm(np.asarray(mesh.bounds[1], dtype=np.float32) - np.asarray(mesh.bounds[0], dtype=np.float32)))
    tau_floor = max(float(args.tau_floor), float(args.tau_floor_scene_scale) * max(scene_diag, 1e-6))
    binding = load_or_build_binding(
        xyz_before,
        mesh,
        surface_binding_payload=str(args.surface_binding_payload),
        face_k=int(args.face_k),
        bind_chunk_size=int(args.bind_chunk_size),
    )

    face_edge = face_edge_mean_lengths(mesh)
    face_ids = np.asarray(binding["face_ids"], dtype=np.int64)
    surface_points = np.asarray(binding["surface_points"], dtype=np.float32)
    normals = np.asarray(binding["normals"], dtype=np.float32)
    tangent_u = np.asarray(binding["tangent_u"], dtype=np.float32)
    tangent_v = np.asarray(binding["tangent_v"], dtype=np.float32)
    delta = np.asarray(binding["normal_offset"], dtype=np.float32)
    edge_tau = float(args.k_edge) * face_edge[face_ids]
    pixel_tau = compute_camera_pixel_tau(
        surface_points,
        scene_root=scene_root,
        model_path=model_path,
        images_subdir=str(args.images_subdir),
        max_camera_views=int(args.max_camera_views),
        depth_min=float(args.depth_min),
        chunk_size=int(args.bind_chunk_size),
    )
    if pixel_tau is None:
        tau_surface = np.maximum(edge_tau, tau_floor).astype(np.float32, copy=False)
    else:
        tau_surface = np.maximum.reduce([float(args.k_px) * pixel_tau, edge_tau, np.full_like(edge_tau, tau_floor)]).astype(np.float32, copy=False)
    delta_norm = delta / np.clip(tau_surface, 1e-8, None)

    normals_t = torch.from_numpy(normals).to(device="cuda", dtype=torch.float32)
    tangent_u_t = torch.from_numpy(tangent_u).to(device="cuda", dtype=torch.float32)
    tangent_v_t = torch.from_numpy(tangent_v).to(device="cuda", dtype=torch.float32)
    sigma_normal_t, cov_t = covariance_sigma_normal(model._scaling.detach(), model._rotation.detach(), normals_t)
    sigma_normal = sigma_normal_t.detach().cpu().numpy().astype(np.float32, copy=False)
    sigma_normal_norm = sigma_normal / np.clip(tau_surface, 1e-8, None)

    rgb = features_dc_to_rgb(model._features_dc.detach()).detach().cpu().numpy().astype(np.float32, copy=False)
    confidence = np.exp(-(delta_norm * delta_norm) / max(float(args.sigma_d) ** 2, 1e-8))
    confidence *= np.exp(-(sigma_normal_norm * sigma_normal_norm) / max(float(args.sigma_n) ** 2, 1e-8))
    center_smooth_allowed = (confidence > float(args.min_surface_conf)) & (np.abs(delta_norm) < float(args.max_d_norm))
    body_probe: Dict[str, np.ndarray] | None = None
    if bool(int(args.enable_body_probe)):
        major_axis_dir, scale_major, major_axis_idx = gaussian_major_axis(model)
        body_probe = compute_body_probe(
            xyz=xyz_before,
            major_axis_dir=major_axis_dir,
            scale_major=scale_major,
            center_normals=normals,
            center_face_ids=face_ids,
            center_smooth_allowed=center_smooth_allowed,
            sigma_normal_norm=sigma_normal_norm,
            tau_surface=tau_surface,
            pixel_tau=pixel_tau,
            face_edge=face_edge,
            mesh=mesh,
            args=args,
        )
        smooth_allowed = center_smooth_allowed & body_probe["body_good"].astype(bool, copy=False)
    else:
        major_axis_idx = np.zeros((xyz_before.shape[0],), dtype=np.int64)
        smooth_allowed = center_smooth_allowed

    neighbors, weights = build_surface_graph(
        surface_points,
        normals,
        face_ids,
        rgb,
        tau_surface,
        confidence.astype(np.float32, copy=False),
        smooth_allowed,
        mesh,
        k_neighbors=int(args.k_neighbors),
        radius_multiplier=float(args.radius_px),
        normal_angle_cut_deg=float(args.normal_angle_cut_deg),
        sigma_angle_deg=float(args.sigma_angle_deg),
        sigma_color=float(args.sigma_color),
        require_face_adjacency=bool(int(args.require_face_adjacency)),
    )
    has_neighbor = np.asarray([nbr.size > 0 for nbr in neighbors], dtype=bool)
    active_smooth = smooth_allowed & has_neighbor

    delta_new = delta.copy()
    if str(args.mode) in {"delta_only", "delta_sigma", "delta_sigma_taucap"}:
        delta_med = neighbor_weighted_medians(delta, neighbors, weights)
        raw_update = float(args.smooth_delta_lambda) * confidence * (delta_med - delta)
        max_update = float(args.max_delta_update_px) * tau_surface
        clipped_update = np.clip(raw_update, -max_update, max_update)
        delta_new[active_smooth] = delta[active_smooth] + clipped_update[active_smooth]

    sigma_new = sigma_normal.copy()
    if str(args.mode) in {"sigma_only", "delta_sigma", "delta_sigma_taucap"}:
        log_sigma = np.log(np.clip(sigma_normal, 1e-8, None))
        log_sigma_med = neighbor_weighted_medians(log_sigma, neighbors, weights)
        log_sigma_new = log_sigma + float(args.smooth_sigma_lambda) * confidence * (log_sigma_med - log_sigma)
        sigma_candidate = np.exp(log_sigma_new).astype(np.float32, copy=False)
        sigma_cap = float(args.sigma_normal_cap_px) * tau_surface
        sigma_new[active_smooth] = np.minimum(sigma_candidate[active_smooth], sigma_cap[active_smooth])

    xyz_new = xyz_before.copy()
    if str(args.mode) in {"delta_only", "delta_sigma", "delta_sigma_taucap"}:
        xyz_new = xyz_before + ((delta_new - delta)[:, None] * normals).astype(np.float32, copy=False)

    scaling_new = model._scaling.detach().clone()
    rotation_new = model._rotation.detach().clone()
    if str(args.mode) in {"sigma_only", "delta_sigma", "delta_sigma_taucap"}:
        sigma_new_t = torch.from_numpy(sigma_new).to(device="cuda", dtype=torch.float32)
        scaling_new, rotation_new = replace_normal_variance(
            cov_t,
            tangent_u_t,
            tangent_v_t,
            normals_t,
            sigma_normal_t,
            sigma_new_t,
            cross_decay_power=float(args.cross_decay_power),
            min_scale=float(args.min_scale),
        )

    opacity_new = model._opacity.detach().clone()
    tau_before = -torch.log(torch.clamp(1.0 - model.get_opacity.detach().reshape(-1), min=1e-6))
    tau_after = tau_before.clone()
    tau_cap_count = 0
    if str(args.mode) == "delta_sigma_taucap" or bool(int(args.enable_tau_cap)):
        tau_np = tau_before.detach().cpu().numpy().astype(np.float32, copy=False)
        tau_med = neighbor_weighted_medians(tau_np, neighbors, weights)
        tau_abs_dev = np.abs(tau_np - tau_med)
        tau_mad = neighbor_weighted_medians(tau_abs_dev, neighbors, weights)
        tau_cap = tau_med + float(args.tau_cap_mad_k) * np.maximum(tau_mad, 1e-6)
        cap_mask = active_smooth & (tau_np > tau_cap)
        tau_np_new = tau_np.copy()
        tau_np_new[cap_mask] = tau_cap[cap_mask]
        tau_after = torch.from_numpy(tau_np_new).to(device="cuda", dtype=torch.float32)
        alpha_new = (1.0 - torch.exp(-tau_after)).clamp(1e-6, 1.0 - 1e-6)
        opacity_new = inverse_sigmoid(alpha_new[:, None])
        tau_cap_count = int(np.count_nonzero(cap_mask))

    delta_after = delta_new
    delta_norm_after = delta_after / np.clip(tau_surface, 1e-8, None)
    sigma_normal_norm_after = sigma_new / np.clip(tau_surface, 1e-8, None)
    changed_xyz = np.linalg.norm(xyz_new - xyz_before, axis=1)
    changed_scale = np.linalg.norm((scaling_new - model._scaling.detach()).detach().cpu().numpy(), axis=1)
    changed_mask = (changed_xyz > 1e-8) | (changed_scale > 1e-8)

    model._xyz = nn.Parameter(torch.from_numpy(xyz_new).to(device="cuda", dtype=torch.float32).requires_grad_(True))
    model._scaling = nn.Parameter(scaling_new.detach().clone().requires_grad_(True))
    model._rotation = nn.Parameter(rotation_new.detach().clone().requires_grad_(True))
    model._opacity = nn.Parameter(opacity_new.detach().clone().requires_grad_(True))
    save_checkpoint(model, output_path, iteration)

    torch.save(torch.from_numpy(delta_norm.astype(np.float32, copy=False)), output_path / "delta_norm_before.pt")
    torch.save(torch.from_numpy(delta_norm_after.astype(np.float32, copy=False)), output_path / "delta_norm_after.pt")
    torch.save(torch.from_numpy(sigma_normal_norm.astype(np.float32, copy=False)), output_path / "sigma_normal_before.pt")
    torch.save(torch.from_numpy(sigma_normal_norm_after.astype(np.float32, copy=False)), output_path / "sigma_normal_after.pt")
    torch.save(torch.from_numpy(changed_mask), output_path / "changed_gaussian_mask.pt")
    if body_probe is not None:
        torch.save(torch.from_numpy(body_probe["samples"].astype(np.float32, copy=False)), output_path / "body_probe_samples.pt")
        torch.save(torch.from_numpy(body_probe["body_center_valid"]), output_path / "body_center_valid.pt")
        torch.save(torch.from_numpy(body_probe["body_endpoint_valid"]), output_path / "body_endpoint_valid.pt")
        torch.save(torch.from_numpy(body_probe["body_good"]), output_path / "body_good_mask.pt")
        torch.save(torch.from_numpy(body_probe["body_valid_ratio"].astype(np.float32, copy=False)), output_path / "body_valid_ratio.pt")
        torch.save(torch.from_numpy(body_probe["body_d_p90"].astype(np.float32, copy=False)), output_path / "body_d_p90.pt")
        torch.save(torch.from_numpy(body_probe["body_d_max"].astype(np.float32, copy=False)), output_path / "body_d_max.pt")
        torch.save(torch.from_numpy(body_probe["body_normal_switch_count"]), output_path / "body_normal_switch_count.pt")
        torch.save(torch.from_numpy(body_probe["body_face_switch_count"]), output_path / "body_face_switch_count.pt")
        torch.save(torch.from_numpy(body_probe["body_case_label"]), output_path / "body_case_label.pt")
        torch.save(
            torch.from_numpy(body_probe["body_endpoint_distance_norm"].astype(np.float32, copy=False)),
            output_path / "body_endpoint_distance_norm.pt",
        )

    body_counts: Dict[str, int] = {}
    body_metrics: Dict[str, object] = {}
    body_artifacts: Dict[str, str] = {}
    if body_probe is not None:
        labels = body_probe["body_case_label"]
        body_counts = {
            "body_good": int(np.count_nonzero(labels == 0)),
            "center_good_endpoint_bad": int(np.count_nonzero(labels == 1)),
            "multi_surface_crossing": int(np.count_nonzero(labels == 2)),
            "all_invalid": int(np.count_nonzero(labels == 3)),
            "uncertain": int(np.count_nonzero(labels == 4)),
        }
        body_metrics = {
            **summarize_value("body_valid_ratio", body_probe["body_valid_ratio"]),
            **summarize_value("body_d_p90", body_probe["body_d_p90"]),
            **summarize_value("body_d_max", body_probe["body_d_max"]),
            "body_endpoint_bad_count": body_counts["center_good_endpoint_bad"],
            "body_normal_switch_count": int(np.count_nonzero(body_probe["body_normal_switch_count"] > 0)),
            "body_face_switch_count": int(np.count_nonzero(body_probe["body_face_switch_count"] > 0)),
        }
        body_artifacts = {
            "body_probe_samples": str(output_path / "body_probe_samples.pt"),
            "body_center_valid": str(output_path / "body_center_valid.pt"),
            "body_endpoint_valid": str(output_path / "body_endpoint_valid.pt"),
            "body_good_mask": str(output_path / "body_good_mask.pt"),
            "body_valid_ratio": str(output_path / "body_valid_ratio.pt"),
            "body_d_p90": str(output_path / "body_d_p90.pt"),
            "body_d_max": str(output_path / "body_d_max.pt"),
            "body_normal_switch_count": str(output_path / "body_normal_switch_count.pt"),
            "body_face_switch_count": str(output_path / "body_face_switch_count.pt"),
            "body_case_label": str(output_path / "body_case_label.pt"),
            "body_endpoint_distance_norm": str(output_path / "body_endpoint_distance_norm.pt"),
        }

    summary: Dict[str, object] = {
        "version": "surface_smooth_filter_v0",
        "model_path": str(model_path),
        "mesh_path": str(mesh_path),
        "output_path": str(output_path),
        "iteration": int(iteration),
        "mode": str(args.mode),
        "count": int(xyz_before.shape[0]),
        "center_smooth_allowed": int(np.count_nonzero(center_smooth_allowed)),
        "smooth_allowed": int(np.count_nonzero(smooth_allowed)),
        "active_smooth": int(np.count_nonzero(active_smooth)),
        "edge_count_directed": int(sum(nbr.size for nbr in neighbors)),
        "body_probe_counts": body_counts,
        "params": vars(args),
        "metrics": {
            **summarize_value("abs_delta_norm_before", np.abs(delta_norm)),
            **summarize_value("abs_delta_norm_after", np.abs(delta_norm_after)),
            **summarize_value("sigma_normal_norm_before", sigma_normal_norm),
            **summarize_value("sigma_normal_norm_after", sigma_normal_norm_after),
            "laplacian_delta_before": graph_laplacian_energy(delta_norm, neighbors, weights),
            "laplacian_delta_after": graph_laplacian_energy(delta_norm_after, neighbors, weights),
            "laplacian_sigma_before": graph_laplacian_energy(np.log(np.clip(sigma_normal_norm, 1e-8, None)), neighbors, weights),
            "laplacian_sigma_after": graph_laplacian_energy(np.log(np.clip(sigma_normal_norm_after, 1e-8, None)), neighbors, weights),
            "num_changed_xyz": int(np.count_nonzero(changed_xyz > 1e-8)),
            "num_changed_scale": int(np.count_nonzero(changed_scale > 1e-8)),
            "max_xyz_shift": float(np.max(changed_xyz)) if changed_xyz.size else 0.0,
            "p95_xyz_shift": float(np.percentile(changed_xyz, 95)) if changed_xyz.size else 0.0,
            "tau_p95_before": float(torch.quantile(tau_before.detach(), 0.95).item()) if tau_before.numel() else 0.0,
            "tau_p95_after": float(torch.quantile(tau_after.detach(), 0.95).item()) if tau_after.numel() else 0.0,
            "tau_cap_count": int(tau_cap_count),
            "major_axis_idx_unique": int(np.unique(major_axis_idx).size) if major_axis_idx.size else 0,
            **body_metrics,
        },
        "artifacts": {
            "point_cloud": str(output_path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"),
            "delta_norm_before": str(output_path / "delta_norm_before.pt"),
            "delta_norm_after": str(output_path / "delta_norm_after.pt"),
            "sigma_normal_before": str(output_path / "sigma_normal_before.pt"),
            "sigma_normal_after": str(output_path / "sigma_normal_after.pt"),
            "changed_gaussian_mask": str(output_path / "changed_gaussian_mask.pt"),
            **body_artifacts,
        },
    }
    with open(output_path / "surface_smooth_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
