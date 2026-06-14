#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gaussian_renderer import render_simple
from joint_judge_mip_sof_v0 import stats_from_array
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from score_gaussian_mesh_alignment_v0 import load_triangle_mesh, query_surface
from train_mip_to_sof_surface_v0 import load_cameras_for_split, load_model_ply, resolve_iteration, select_uniform
from utils.general_utils import build_rotation, inverse_sigmoid
from utils.prior_fusion import project_points_camera
from utils.sh_utils import RGB2SH, SH2RGB


def normalize_np(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), eps, None)


def quaternion_from_rotation_matrix_np(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64, copy=False)
    q = np.empty((m.shape[0], 4), dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    positive = trace > 0.0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        q[positive, 0] = 0.25 * s
        q[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
        q[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
        q[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s

    negative = ~positive
    if np.any(negative):
        idx = np.where(negative)[0]
        mn = m[idx]
        choice = np.argmax(np.stack([mn[:, 0, 0], mn[:, 1, 1], mn[:, 2, 2]], axis=1), axis=1)
        for axis in range(3):
            local = idx[choice == axis]
            if local.size == 0:
                continue
            ml = m[local]
            if axis == 0:
                s = np.sqrt(1.0 + ml[:, 0, 0] - ml[:, 1, 1] - ml[:, 2, 2]) * 2.0
                q[local, 0] = (ml[:, 2, 1] - ml[:, 1, 2]) / s
                q[local, 1] = 0.25 * s
                q[local, 2] = (ml[:, 0, 1] + ml[:, 1, 0]) / s
                q[local, 3] = (ml[:, 0, 2] + ml[:, 2, 0]) / s
            elif axis == 1:
                s = np.sqrt(1.0 + ml[:, 1, 1] - ml[:, 0, 0] - ml[:, 2, 2]) * 2.0
                q[local, 0] = (ml[:, 0, 2] - ml[:, 2, 0]) / s
                q[local, 1] = (ml[:, 0, 1] + ml[:, 1, 0]) / s
                q[local, 2] = 0.25 * s
                q[local, 3] = (ml[:, 1, 2] + ml[:, 2, 1]) / s
            else:
                s = np.sqrt(1.0 + ml[:, 2, 2] - ml[:, 0, 0] - ml[:, 1, 1]) * 2.0
                q[local, 0] = (ml[:, 1, 0] - ml[:, 0, 1]) / s
                q[local, 1] = (ml[:, 0, 2] + ml[:, 2, 0]) / s
                q[local, 2] = (ml[:, 1, 2] + ml[:, 2, 1]) / s
                q[local, 3] = 0.25 * s

    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src = src_model_path / name
        if src.exists():
            shutil.copy2(src, dst_model_path / name)


def tracking_array(gaussians: GaussianModel, name: str, dtype: np.dtype, default: int | bool) -> np.ndarray:
    value = getattr(gaussians, name, None)
    total = int(gaussians.get_xyz.shape[0])
    if torch.is_tensor(value) and int(value.shape[0]) == total:
        return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.full((total,), default, dtype=dtype)


def save_gray(path: Path, image: torch.Tensor) -> None:
    array = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    Image.fromarray((array * 255.0).astype(np.uint8), mode="L").save(path)


def local_mesh_edge_lengths(mesh_obj, face_ids: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    edge_lengths = np.zeros((faces.shape[0],), dtype=np.float32)
    tri = vertices[faces]
    edges = np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ],
        axis=1,
    )
    edge_lengths[:] = np.mean(edges, axis=1).astype(np.float32)
    fallback = float(np.median(edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)])) if edge_lengths.size else 1e-3
    clipped = np.clip(face_ids.astype(np.int64, copy=False), 0, max(edge_lengths.shape[0] - 1, 0))
    out = edge_lengths[clipped] if edge_lengths.size else np.full((face_ids.shape[0],), fallback, dtype=np.float32)
    invalid = (face_ids < 0) | (~np.isfinite(out)) | (out <= 0.0)
    out[invalid] = fallback
    return out.astype(np.float32, copy=False)


def tangent_basis_from_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize_np(normals.astype(np.float32, copy=False))
    ref = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (n.shape[0], 1))
    parallel = np.abs(np.sum(ref * n, axis=1)) > 0.9
    ref[parallel] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    t1 = normalize_np(np.cross(ref, n))
    t2 = normalize_np(np.cross(n, t1))
    return t1.astype(np.float32, copy=False), t2.astype(np.float32, copy=False)


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


def barycentric_for_faces(mesh_obj, points: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    bary = np.tile(np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32), (points.shape[0], 1))
    valid = (face_ids >= 0) & (face_ids < faces.shape[0])
    if np.any(valid):
        bary[valid] = points_to_barycentric(
            points[valid].astype(np.float32, copy=False),
            vertices[faces[face_ids[valid]]].astype(np.float32, copy=False),
        )
    return bary.astype(np.float32, copy=False)


def build_sample_points(
    xyz: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    anisotropy: np.ndarray,
    *,
    long_anisotropy_threshold: float,
    major_sample_count: int,
    major_sample_extent: float,
) -> Dict[str, np.ndarray]:
    total = int(xyz.shape[0])
    source_ids: List[np.ndarray] = []
    sample_ts: List[np.ndarray] = []
    major_axis_ids = np.argmax(scales, axis=1).astype(np.int64, copy=False)
    major_scales = np.max(scales, axis=1).astype(np.float32, copy=False)
    use_multi = (anisotropy >= float(long_anisotropy_threshold)) & (int(major_sample_count) > 1)
    single_ids = np.flatnonzero(~use_multi).astype(np.int64, copy=False)
    if single_ids.size > 0:
        source_ids.append(single_ids)
        sample_ts.append(np.zeros((single_ids.size,), dtype=np.float32))
    multi_ids = np.flatnonzero(use_multi).astype(np.int64, copy=False)
    if multi_ids.size > 0:
        t = np.linspace(-1.0, 1.0, num=max(int(major_sample_count), 2), dtype=np.float32)
        source_ids.append(np.repeat(multi_ids, t.shape[0]))
        sample_ts.append(np.tile(t, multi_ids.shape[0]))
    if source_ids:
        source_idx = np.concatenate(source_ids).astype(np.int64, copy=False)
        sample_t = np.concatenate(sample_ts).astype(np.float32, copy=False)
    else:
        source_idx = np.empty((0,), dtype=np.int64)
        sample_t = np.empty((0,), dtype=np.float32)

    major_axes = rotations[source_idx, :, major_axis_ids[source_idx]]
    points = xyz[source_idx] + major_axes * (
        sample_t[:, None] * major_scales[source_idx, None] * float(major_sample_extent)
    )
    sample_count_per_source = np.bincount(source_idx, minlength=total).astype(np.float32)
    sample_count_per_source = np.maximum(sample_count_per_source, 1.0)
    return {
        "points": points.astype(np.float32, copy=False),
        "source_idx": source_idx,
        "sample_t": sample_t,
        "source_sample_count": sample_count_per_source,
    }


def bilinear_sample_hw(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(xy[:, 0], 0.0, max(float(w - 1), 0.0))
    y = np.clip(xy[:, 1], 0.0, max(float(h - 1), 0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wa = (x1.astype(np.float32) - x) * (y1.astype(np.float32) - y)
    wb = (x - x0.astype(np.float32)) * (y1.astype(np.float32) - y)
    wc = (x1.astype(np.float32) - x) * (y - y0.astype(np.float32))
    wd = (x - x0.astype(np.float32)) * (y - y0.astype(np.float32))
    same_x = x0 == x1
    same_y = y0 == y1
    wa[same_x] = 1.0 - (y[same_x] - y0[same_x].astype(np.float32))
    wc[same_x] = y[same_x] - y0[same_x].astype(np.float32)
    wb[same_x] = 0.0
    wd[same_x] = 0.0
    wa[same_y] = 1.0 - (x[same_y] - x0[same_y].astype(np.float32))
    wb[same_y] = x[same_y] - x0[same_y].astype(np.float32)
    wc[same_y] = 0.0
    wd[same_y] = 0.0
    if image.ndim == 2:
        return image[y0, x0] * wa + image[y0, x1] * wb + image[y1, x0] * wc + image[y1, x1] * wd
    return (
        image[y0, x0] * wa[:, None]
        + image[y0, x1] * wb[:, None]
        + image[y1, x0] * wc[:, None]
        + image[y1, x1] * wd[:, None]
    )


def source_dc_rgb_np(source: GaussianModel) -> np.ndarray:
    with torch.no_grad():
        dc = source._features_dc.detach()
        if source.use_SBs:
            rgb = torch.clamp(dc[:, :3], 0.0, 1.0)
        else:
            rgb = torch.clamp(SH2RGB(dc[:, 0, :]), 0.0, 1.0)
    return rgb.detach().cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def collect_view_support(
    *,
    source: GaussianModel,
    mip_model: GaussianModel | None,
    cameras: Sequence[object],
    sample_points: np.ndarray,
    depth_min: float,
    alpha_ref: float,
    depth_tau_ratio: float,
    projection_chunk_size: int,
    background: torch.Tensor,
) -> Dict[str, np.ndarray]:
    source_total = int(source.get_xyz.shape[0])
    sample_total = int(sample_points.shape[0])
    hit_count = np.zeros((source_total,), dtype=np.int32)
    radius_sum = np.zeros((source_total,), dtype=np.float64)
    radius_max = np.zeros((source_total,), dtype=np.float32)
    pixel_world_sum = np.zeros((sample_total,), dtype=np.float64)
    pixel_world_count = np.zeros((sample_total,), dtype=np.int32)
    mip_support_sum = np.zeros((sample_total,), dtype=np.float64)
    mip_support_count = np.zeros((sample_total,), dtype=np.int32)
    mip_rgb_sum = np.zeros((sample_total, 3), dtype=np.float64)
    mip_rgb_weight_sum = np.zeros((sample_total,), dtype=np.float64)

    for camera in tqdm(cameras, desc="mesh-bounded support views"):
        source_pkg = render_simple(camera, source, background)
        visible = source_pkg["visibility_filter"].detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
        if np.any(visible):
            radii = source_pkg["radii"].detach().float().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            ids = np.flatnonzero(visible).astype(np.int64, copy=False)
            valid_radius = np.maximum(radii[ids], 0.0)
            hit_count[ids] += 1
            radius_sum[ids] += valid_radius.astype(np.float64)
            radius_max[ids] = np.maximum(radius_max[ids], valid_radius)

        mip_alpha = None
        mip_depth = None
        mip_rgb = None
        if mip_model is not None:
            mip_pkg = render_simple(camera, mip_model, background)
            mip_rgb = mip_pkg["render"].detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
            mip_alpha = mip_pkg["alpha"].detach().float().squeeze().cpu().numpy().astype(np.float32, copy=False)
            mip_depth = mip_pkg["depth"].detach().float().squeeze().cpu().numpy().astype(np.float32, copy=False)

        f_mean = 0.5 * (float(camera.focal_x) + float(camera.focal_y))
        chunk = max(int(projection_chunk_size), 1)
        for begin in range(0, sample_total, chunk):
            end = min(begin + chunk, sample_total)
            projected, valid = project_points_camera(camera, sample_points[begin:end], depth_min=float(depth_min), margin=0)
            valid_ids = np.flatnonzero(valid).astype(np.int64, copy=False)
            if valid_ids.size == 0:
                continue
            global_ids = begin + valid_ids
            z = projected[valid_ids, 2]
            pixel_world_sum[global_ids] += (z / max(f_mean, 1e-6)).astype(np.float64)
            pixel_world_count[global_ids] += 1
            if mip_alpha is None or mip_depth is None:
                continue
            xy = projected[valid_ids, :2]
            alpha = bilinear_sample_hw(mip_alpha, xy)
            depth = bilinear_sample_hw(mip_depth, xy)
            alpha_gate = np.clip(alpha / max(float(alpha_ref), 1e-6), 0.0, 1.0)
            valid_depth = np.isfinite(depth) & (depth > float(depth_min))
            depth_tau = np.maximum(float(depth_tau_ratio) * np.maximum(z, float(depth_min)), 1e-6)
            depth_gate = np.zeros_like(alpha_gate, dtype=np.float32)
            depth_gate[valid_depth] = np.exp(-np.abs(depth[valid_depth] - z[valid_depth]) / depth_tau[valid_depth])
            mip_support_sum[global_ids] += (alpha_gate * depth_gate).astype(np.float64)
            mip_support_count[global_ids] += 1
            if mip_rgb is not None:
                rgb = bilinear_sample_hw(mip_rgb, xy)
                weight = (alpha_gate * depth_gate).astype(np.float32, copy=False)
                mip_rgb_sum[global_ids] += (rgb * weight[:, None]).astype(np.float64)
                mip_rgb_weight_sum[global_ids] += weight.astype(np.float64)

    mean_radius = np.divide(
        radius_sum,
        np.maximum(hit_count, 1),
        out=np.zeros_like(radius_sum, dtype=np.float64),
        where=hit_count > 0,
    ).astype(np.float32)
    pixel_world = np.divide(
        pixel_world_sum,
        np.maximum(pixel_world_count, 1),
        out=np.zeros_like(pixel_world_sum, dtype=np.float64),
        where=pixel_world_count > 0,
    ).astype(np.float32)
    if mip_model is None:
        mip_support = np.ones((sample_total,), dtype=np.float32)
        mip_rgb = np.zeros((sample_total, 3), dtype=np.float32)
        mip_rgb_weight = np.zeros((sample_total,), dtype=np.float32)
    else:
        mip_support = np.divide(
            mip_support_sum,
            np.maximum(mip_support_count, 1),
            out=np.zeros_like(mip_support_sum, dtype=np.float64),
            where=mip_support_count > 0,
        ).astype(np.float32)
        mip_rgb = np.divide(
            mip_rgb_sum,
            np.maximum(mip_rgb_weight_sum[:, None], 1e-8),
            out=np.zeros_like(mip_rgb_sum, dtype=np.float64),
            where=mip_rgb_weight_sum[:, None] > 0,
        ).astype(np.float32)
        mip_rgb_weight = mip_rgb_weight_sum.astype(np.float32)
    return {
        "visible_view_count": hit_count,
        "mean_radius": mean_radius,
        "max_radius": radius_max,
        "pixel_world": pixel_world,
        "projected_view_count": pixel_world_count,
        "mip_support": mip_support,
        "mip_support_count": mip_support_count,
        "mip_rgb": mip_rgb,
        "mip_rgb_weight": mip_rgb_weight,
    }


def make_proxy_model(
    source: GaussianModel,
    *,
    centers: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    source_idx: np.ndarray,
    dc_rgb: np.ndarray | None = None,
) -> GaussianModel:
    proxy = GaussianModel(source.max_sh_degree, use_SBs=source.use_SBs)
    proxy.active_sh_degree = 0
    proxy.spatial_lr_scale = float(source.spatial_lr_scale)
    source_dc = source._features_dc.detach()[torch.from_numpy(source_idx).to(device=source._features_dc.device, dtype=torch.long)]
    source_rest = source._features_rest.detach()
    rest_shape = (int(source_idx.shape[0]), *source_rest.shape[1:])
    proxy._xyz = nn.Parameter(torch.from_numpy(centers).float().cuda().requires_grad_(False))
    if dc_rgb is None:
        proxy_dc = source_dc.detach().clone()
    elif source.use_SBs:
        proxy_dc = source_dc.detach().clone()
        proxy_dc[:, :3] = torch.from_numpy(np.clip(dc_rgb, 0.0, 1.0)).float().cuda()
    else:
        proxy_dc = torch.zeros_like(source_dc.detach())
        proxy_dc[:, 0, :] = RGB2SH(torch.from_numpy(np.clip(dc_rgb, 0.0, 1.0)).float().cuda())
    proxy._features_dc = nn.Parameter(proxy_dc.requires_grad_(False))
    proxy._features_rest = nn.Parameter(torch.zeros(rest_shape, dtype=torch.float32, device="cuda").requires_grad_(False))
    proxy._opacity = nn.Parameter(inverse_sigmoid(torch.from_numpy(opacities[:, None]).float().cuda()).requires_grad_(False))
    proxy._scaling = nn.Parameter(torch.log(torch.from_numpy(np.clip(scales, 1e-8, None)).float().cuda()).requires_grad_(False))
    proxy._rotation = nn.Parameter(torch.from_numpy(rotations).float().cuda().requires_grad_(False))
    proxy.filter_3D = torch.zeros((int(source_idx.shape[0]), 1), dtype=torch.float32, device="cuda")
    proxy.max_radii2D = torch.zeros((int(source_idx.shape[0]),), dtype=torch.float32, device="cuda")
    proxy.xyz_gradient_accum = torch.zeros((int(source_idx.shape[0]), 1), dtype=torch.float32, device="cuda")
    proxy.xyz_gradient_accum_abs = torch.zeros((int(source_idx.shape[0]), 1), dtype=torch.float32, device="cuda")
    proxy.xyz_gradient_accum_abs_max = torch.zeros((int(source_idx.shape[0]), 1), dtype=torch.float32, device="cuda")
    proxy.denom = torch.zeros((int(source_idx.shape[0]), 1), dtype=torch.float32, device="cuda")
    proxy.init_tracking_state(int(source_idx.shape[0]), source_tag=int(GaussianSourceTag.PRIOR_INJECTED))
    proxy._seed_id = torch.from_numpy(source_idx.astype(np.int64, copy=False)).to(device="cuda", dtype=torch.int64)
    source_generation = tracking_array(source, "_generation", np.int32, 0)
    proxy._generation = torch.from_numpy((source_generation[source_idx] + 1).astype(np.int32, copy=False)).to(
        device="cuda",
        dtype=torch.int32,
    )
    return proxy


def render_confidence_maps(
    *,
    proxy: GaussianModel,
    cameras: Sequence[object],
    output_dir: Path,
    iteration: int,
    background: torch.Tensor,
) -> Dict[str, object]:
    render_root = output_dir / "surface_confidence_maps" / f"ours_{int(iteration)}"
    render_root.mkdir(parents=True, exist_ok=True)
    selected: List[int] = []
    for idx, camera in enumerate(tqdm(cameras, desc="render mesh-bounded confidence")):
        pkg = render_simple(camera, proxy, background)
        save_gray(render_root / f"{idx:05d}_alpha.png", pkg["alpha"])
        selected.append(int(idx))
    return {
        "render_root": str(render_root),
        "view_count": int(len(cameras)),
        "selected_indices": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a low-opacity zero-SH mesh-bounded proxy Gaussian field from SOFGS evidence.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--mip_model_path", default="")
    parser.add_argument("--mip_iteration", type=int, default=-1)
    parser.add_argument("--surface_query_mode", choices=["auto", "exact_open3d", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=1_000_000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--projection_chunk_size", type=int, default=200000)
    parser.add_argument("--k_px", type=float, default=2.5)
    parser.add_argument("--k_edge", type=float, default=0.4)
    parser.add_argument("--tau_floor", type=float, default=1e-4)
    parser.add_argument("--sigma_dist", type=float, default=1.25)
    parser.add_argument("--sigma_normal", type=float, default=1.25)
    parser.add_argument("--view_conf_views", type=float, default=4.0)
    parser.add_argument("--mip_alpha_ref", type=float, default=0.15)
    parser.add_argument("--mip_depth_tau_ratio", type=float, default=0.03)
    parser.add_argument("--mip_color_blend", type=float, default=1.0)
    parser.add_argument("--min_mip_color_weight", type=float, default=0.03)
    parser.add_argument("--color_sigma", type=float, default=0.18)
    parser.add_argument("--color_confidence_strength", type=float, default=0.25)
    parser.add_argument("--confidence_min", type=float, default=0.30)
    parser.add_argument("--normal_offset_eta", type=float, default=0.5)
    parser.add_argument("--tau_beta", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=0.035)
    parser.add_argument("--normal_scale_cap", type=float, default=0.7)
    parser.add_argument("--tangent_scale_min_ratio", type=float, default=0.25)
    parser.add_argument("--tangent_scale_max_ratio", type=float, default=4.0)
    parser.add_argument("--long_anisotropy_threshold", type=float, default=12.0)
    parser.add_argument("--major_sample_count", type=int, default=3)
    parser.add_argument("--major_sample_extent", type=float, default=1.0)
    parser.add_argument("--render_confidence_maps", action="store_true")
    parser.add_argument("--render_max_views", type=int, default=16)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    iteration = resolve_iteration(model_path, int(args.iteration))
    source = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    mesh_obj = load_triangle_mesh(mesh_path)
    cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    cameras = select_uniform(cameras, int(args.max_views))
    if len(cameras) <= 0:
        raise RuntimeError("No cameras selected for mesh-boundedGS support computation.")

    mip_model = None
    mip_iteration = -1
    if str(args.mip_model_path).strip():
        mip_path = Path(args.mip_model_path).expanduser().resolve()
        mip_iteration = resolve_iteration(mip_path, int(args.mip_iteration))
        mip_model = load_model_ply(mip_path, iteration=mip_iteration, sh_degree=3)

    with torch.no_grad():
        xyz = source.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        opacity = source.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
        scale = source.get_scaling.detach()
        if isinstance(source.filter_3D, torch.Tensor) and int(source.filter_3D.shape[0]) == int(scale.shape[0]):
            filter_3d = source.filter_3D.detach()
            effective_scale_t = torch.sqrt(scale * scale + filter_3d * filter_3d)
        else:
            effective_scale_t = scale
        effective_scale = effective_scale_t.cpu().numpy().astype(np.float32, copy=False)
        rotations = build_rotation(source._rotation.detach()).detach().cpu().numpy().astype(np.float32, copy=False)
    anisotropy = (np.max(effective_scale, axis=1) / np.clip(np.min(effective_scale, axis=1), 1e-8, None)).astype(np.float32)
    source_tau = -np.log(np.clip(1.0 - opacity, 1e-6, 1.0)).astype(np.float32)

    samples = build_sample_points(
        xyz,
        rotations,
        effective_scale,
        anisotropy,
        long_anisotropy_threshold=float(args.long_anisotropy_threshold),
        major_sample_count=int(args.major_sample_count),
        major_sample_extent=float(args.major_sample_extent),
    )
    sample_points = samples["points"]
    sample_source_idx = samples["source_idx"]
    sample_t = samples["sample_t"]
    source_sample_count = samples["source_sample_count"]

    surface = query_surface(
        mesh_obj=mesh_obj,
        points_xyz=sample_points,
        mode=str(args.surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )
    q = surface["nearest_surface_point"].astype(np.float32, copy=False)
    normals = normalize_np(surface["nearest_surface_normal"].astype(np.float32, copy=False))
    face_id = surface["nearest_face_id"].astype(np.int64, copy=False)
    t1, t2 = tangent_basis_from_normals(normals)
    local_edge = local_mesh_edge_lengths(mesh_obj, face_id)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(args.white_background) else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    support = collect_view_support(
        source=source,
        mip_model=mip_model,
        cameras=cameras,
        sample_points=q,
        depth_min=0.01,
        alpha_ref=float(args.mip_alpha_ref),
        depth_tau_ratio=float(args.mip_depth_tau_ratio),
        projection_chunk_size=int(args.projection_chunk_size),
        background=background,
    )

    tau_surface = np.maximum.reduce(
        [
            float(args.k_px) * support["pixel_world"],
            float(args.k_edge) * local_edge,
            np.full((sample_points.shape[0],), float(args.tau_floor), dtype=np.float32),
        ]
    ).astype(np.float32)
    delta = sample_points - q
    signed_offset = np.sum(delta * normals, axis=1).astype(np.float32)
    d_norm = np.abs(signed_offset) / np.clip(tau_surface, 1e-8, None)

    sample_rot = rotations[sample_source_idx]
    sample_scale = effective_scale[sample_source_idx]

    def sigma_along(direction: np.ndarray) -> np.ndarray:
        dot = np.einsum("nij,ni->nj", sample_rot, direction.astype(np.float32, copy=False))
        return np.sqrt(np.sum((dot * sample_scale) ** 2, axis=1)).astype(np.float32)

    sigma_normal = sigma_along(normals)
    sigma_t1 = sigma_along(t1)
    sigma_t2 = sigma_along(t2)
    sigma_normal_norm = sigma_normal / np.clip(tau_surface, 1e-8, None)

    c_dist = np.exp(-np.square(d_norm / max(float(args.sigma_dist), 1e-6))).astype(np.float32)
    c_thin = np.exp(-np.square(sigma_normal_norm / max(float(args.sigma_normal), 1e-6))).astype(np.float32)
    c_view_source = np.clip(
        support["visible_view_count"].astype(np.float32) / max(float(args.view_conf_views), 1e-6),
        0.0,
        1.0,
    )
    c_view = c_view_source[sample_source_idx]
    c_mip = np.clip(support["mip_support"], 0.0, 1.0)
    source_rgb = source_dc_rgb_np(source)
    sample_source_rgb = source_rgb[sample_source_idx]
    has_mip_color = support["mip_rgb_weight"] >= float(args.min_mip_color_weight)
    sample_mip_rgb = np.where(has_mip_color[:, None], support["mip_rgb"], sample_source_rgb).astype(np.float32)
    color_l1 = np.mean(np.abs(sample_source_rgb - sample_mip_rgb), axis=1).astype(np.float32)
    c_color = np.exp(-np.square(color_l1 / max(float(args.color_sigma), 1e-6))).astype(np.float32)
    color_strength = np.clip(float(args.color_confidence_strength), 0.0, 1.0)
    confidence = np.clip(
        c_dist * c_thin * c_view * c_mip * ((1.0 - color_strength) + color_strength * c_color),
        0.0,
        1.0,
    ).astype(np.float32)
    selected = confidence >= float(args.confidence_min)
    if not np.any(selected):
        raise RuntimeError("No mesh-bounded proxy gaussians survived confidence_min.")

    selected_idx = np.flatnonzero(selected).astype(np.int64, copy=False)
    selected_source = sample_source_idx[selected_idx]
    selected_tau_surface = tau_surface[selected_idx]
    selected_signed = signed_offset[selected_idx]
    selected_q = q[selected_idx].astype(np.float32, copy=False)
    selected_normals = normals[selected_idx].astype(np.float32, copy=False)
    selected_face_id = face_id[selected_idx].astype(np.int64, copy=False)
    selected_bary = barycentric_for_faces(mesh_obj, selected_q, selected_face_id)
    clipped_offset = np.clip(
        selected_signed,
        -float(args.normal_offset_eta) * selected_tau_surface,
        float(args.normal_offset_eta) * selected_tau_surface,
    )
    centers = selected_q + clipped_offset[:, None] * selected_normals

    tangent_min = np.maximum(float(args.tangent_scale_min_ratio) * selected_tau_surface, 1e-8)
    tangent_max = np.maximum(float(args.tangent_scale_max_ratio) * selected_tau_surface, tangent_min)
    scale_t1 = np.clip(sigma_t1[selected_idx], tangent_min, tangent_max)
    scale_t2 = np.clip(sigma_t2[selected_idx], tangent_min, tangent_max)
    scale_n = np.minimum(sigma_normal[selected_idx], float(args.normal_scale_cap) * selected_tau_surface)
    scale_n = np.clip(scale_n, 1e-8, None)
    scales = np.stack([scale_t1, scale_t2, scale_n], axis=1).astype(np.float32)
    rotation_matrix = np.stack([t1[selected_idx], t2[selected_idx], selected_normals], axis=2)
    proxy_rotations = quaternion_from_rotation_matrix_np(rotation_matrix)

    tau_bound = (
        float(args.tau_beta)
        * confidence[selected_idx]
        * source_tau[selected_source]
        / source_sample_count[selected_source]
    )
    opacities = np.clip(1.0 - np.exp(-np.clip(tau_bound, 0.0, None)), 1e-6, float(args.alpha_max)).astype(np.float32)
    selected_mip_valid = has_mip_color[selected_idx]
    selected_source_rgb = source_rgb[selected_source]
    color_blend = np.clip(float(args.mip_color_blend), 0.0, 1.0)
    selected_proxy_rgb = np.where(
        selected_mip_valid[:, None],
        (1.0 - color_blend) * selected_source_rgb + color_blend * sample_mip_rgb[selected_idx],
        selected_source_rgb,
    ).astype(np.float32)

    proxy = make_proxy_model(
        source,
        centers=centers.astype(np.float32, copy=False),
        scales=scales,
        rotations=proxy_rotations,
        opacities=opacities,
        source_idx=selected_source,
        dc_rgb=selected_proxy_rgb,
    )
    copy_render_config(model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{iteration}"
    point_dir.mkdir(parents=True, exist_ok=True)
    proxy.save_ply(str(point_dir / "point_cloud.ply"))
    proxy.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    source_confidence = np.zeros((xyz.shape[0],), dtype=np.float32)
    np.maximum.at(source_confidence, sample_source_idx, confidence)
    source_proxy_count = np.bincount(selected_source, minlength=xyz.shape[0]).astype(np.int32)
    source_d_norm_min = np.full((xyz.shape[0],), np.inf, dtype=np.float32)
    source_sigma_normal_norm_min = np.full((xyz.shape[0],), np.inf, dtype=np.float32)
    np.minimum.at(source_d_norm_min, sample_source_idx, d_norm.astype(np.float32))
    np.minimum.at(source_sigma_normal_norm_min, sample_source_idx, sigma_normal_norm.astype(np.float32))
    source_d_norm_min[~np.isfinite(source_d_norm_min)] = 0.0
    source_sigma_normal_norm_min[~np.isfinite(source_sigma_normal_norm_min)] = 0.0

    face_count = int(np.asarray(mesh_obj.faces).shape[0])
    face_product = np.ones((face_count,), dtype=np.float64)
    face_weight_sum = np.zeros((face_count,), dtype=np.float64)
    face_conf_sum = np.zeros((face_count,), dtype=np.float64)
    proxy_face = selected_face_id
    valid_face = (proxy_face >= 0) & (proxy_face < face_count)
    contribution = np.clip(confidence[selected_idx] * opacities / max(float(args.alpha_max), 1e-6), 0.0, 0.95)
    if np.any(valid_face):
        np.multiply.at(face_product, proxy_face[valid_face], 1.0 - contribution[valid_face])
        np.add.at(face_weight_sum, proxy_face[valid_face], contribution[valid_face])
        np.add.at(face_conf_sum, proxy_face[valid_face], confidence[selected_idx][valid_face] * contribution[valid_face])
    face_confidence_product = (1.0 - face_product).astype(np.float32)
    face_confidence_mean = np.divide(
        face_conf_sum,
        np.maximum(face_weight_sum, 1e-8),
        out=np.zeros_like(face_conf_sum, dtype=np.float64),
        where=face_weight_sum > 0,
    ).astype(np.float32)

    payload = {
        "version": "mesh_bounded_gaussians_v0",
        "source_idx": torch.from_numpy(selected_source.astype(np.int64, copy=False)),
        "source_confidence": torch.from_numpy(source_confidence),
        "source_proxy_count": torch.from_numpy(source_proxy_count),
        "source_d_norm_min": torch.from_numpy(source_d_norm_min),
        "source_sigma_normal_norm_min": torch.from_numpy(source_sigma_normal_norm_min),
        "source_visible_view_count": torch.from_numpy(support["visible_view_count"].copy()),
        "proxy_confidence": torch.from_numpy(confidence[selected_idx].astype(np.float32, copy=False)),
        "proxy_d_norm": torch.from_numpy(d_norm[selected_idx].astype(np.float32, copy=False)),
        "proxy_sigma_normal_norm": torch.from_numpy(sigma_normal_norm[selected_idx].astype(np.float32, copy=False)),
        "proxy_mip_support": torch.from_numpy(c_mip[selected_idx].astype(np.float32, copy=False)),
        "proxy_mip_rgb": torch.from_numpy(sample_mip_rgb[selected_idx].astype(np.float32, copy=False)),
        "proxy_source_rgb": torch.from_numpy(selected_source_rgb.astype(np.float32, copy=False)),
        "proxy_rgb": torch.from_numpy(selected_proxy_rgb.astype(np.float32, copy=False)),
        "proxy_mip_color_valid": torch.from_numpy(selected_mip_valid.astype(bool, copy=False)),
        "proxy_source_mip_color_l1": torch.from_numpy(color_l1[selected_idx].astype(np.float32, copy=False)),
        "proxy_color_confidence": torch.from_numpy(c_color[selected_idx].astype(np.float32, copy=False)),
        "proxy_view_confidence": torch.from_numpy(c_view[selected_idx].astype(np.float32, copy=False)),
        "proxy_face_id": torch.from_numpy(proxy_face.astype(np.int64, copy=False)),
        "proxy_source_point_p": torch.from_numpy(sample_points[selected_idx].astype(np.float32, copy=False)),
        "proxy_mesh_point_q": torch.from_numpy(selected_q.astype(np.float32, copy=False)),
        "proxy_mesh_normal": torch.from_numpy(selected_normals.astype(np.float32, copy=False)),
        "proxy_mesh_barycentric": torch.from_numpy(selected_bary.astype(np.float32, copy=False)),
        "proxy_signed_offset": torch.from_numpy(selected_signed.astype(np.float32, copy=False)),
        "proxy_clipped_signed_offset": torch.from_numpy(clipped_offset.astype(np.float32, copy=False)),
        "proxy_local_mesh_edge": torch.from_numpy(local_edge[selected_idx].astype(np.float32, copy=False)),
        "proxy_tangent_u": torch.from_numpy(t1[selected_idx].astype(np.float32, copy=False)),
        "proxy_tangent_v": torch.from_numpy(t2[selected_idx].astype(np.float32, copy=False)),
        "proxy_sample_t": torch.from_numpy(sample_t[selected_idx].astype(np.float32, copy=False)),
        "proxy_tau_surface": torch.from_numpy(selected_tau_surface.astype(np.float32, copy=False)),
        "proxy_opacity": torch.from_numpy(opacities.astype(np.float32, copy=False)),
        "face_confidence_product": torch.from_numpy(face_confidence_product),
        "face_confidence_mean": torch.from_numpy(face_confidence_mean),
        "face_weight_sum": torch.from_numpy(face_weight_sum.astype(np.float32)),
    }
    payload_path = output_model_path / "mesh_bounded_gaussians_v0.pt"
    torch.save(payload, payload_path)

    render_summary: Dict[str, object] = {"enabled": False}
    if bool(args.render_confidence_maps):
        render_cameras = select_uniform(cameras, int(args.render_max_views))
        render_summary = render_confidence_maps(
            proxy=proxy,
            cameras=render_cameras,
            output_dir=output_model_path,
            iteration=iteration,
            background=background,
        )

    summary = {
        "version": "mesh_bounded_gaussians_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "mesh_path": str(mesh_path),
        "mip_model_path": str(args.mip_model_path),
        "loaded_iteration": int(iteration),
        "mip_iteration": int(mip_iteration),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "view_count": int(len(cameras)),
        "source_gaussians": int(xyz.shape[0]),
        "sample_count": int(sample_points.shape[0]),
        "proxy_count": int(centers.shape[0]),
        "proxy_ratio": float(centers.shape[0] / max(int(xyz.shape[0]), 1)),
        "surface_query_mode_used": str(surface.get("surface_query_mode_used", args.surface_query_mode)),
        "thresholds": {
            "confidence_min": float(args.confidence_min),
            "long_anisotropy_threshold": float(args.long_anisotropy_threshold),
            "major_sample_count": int(args.major_sample_count),
            "tau_beta": float(args.tau_beta),
            "alpha_max": float(args.alpha_max),
            "mip_color_blend": float(args.mip_color_blend),
            "min_mip_color_weight": float(args.min_mip_color_weight),
            "color_sigma": float(args.color_sigma),
            "color_confidence_strength": float(args.color_confidence_strength),
        },
        "stats": {
            "confidence_all_samples": stats_from_array(confidence),
            "confidence_selected": stats_from_array(confidence[selected_idx]),
            "d_norm_selected": stats_from_array(d_norm[selected_idx]),
            "sigma_normal_norm_selected": stats_from_array(sigma_normal_norm[selected_idx]),
            "mip_support_selected": stats_from_array(c_mip[selected_idx]),
            "mip_color_weight_selected": stats_from_array(support["mip_rgb_weight"][selected_idx]),
            "source_mip_color_l1_selected": stats_from_array(color_l1[selected_idx]),
            "color_confidence_selected": stats_from_array(c_color[selected_idx]),
            "visible_view_count_source": stats_from_array(support["visible_view_count"].astype(np.float32)),
            "source_proxy_count": stats_from_array(source_proxy_count.astype(np.float32)),
            "face_confidence_product": stats_from_array(face_confidence_product),
        },
        "paths": {
            "proxy_model": str(output_model_path),
            "proxy_ply": str(point_dir / "point_cloud.ply"),
            "payload": str(payload_path),
            "summary": str(output_model_path / "mesh_bounded_gaussians_v0_summary.json"),
        },
        "renders": render_summary,
    }
    summary_path = output_model_path / "mesh_bounded_gaussians_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
