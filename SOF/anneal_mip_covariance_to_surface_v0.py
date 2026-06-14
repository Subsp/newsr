from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import (
    copy_render_config,
    load_cameras_for_split,
    load_model_ply,
    resolve_iteration,
    select_uniform,
)
from cleanup_mip_view_aligned_volume_artifacts_v0 import (
    collect_view_aligned_stats,
    gaussian_geometry,
    robust_scene_diag,
)
from utils.general_utils import build_rotation, safe_state
from utils.prior_fusion import _quaternion_from_rotation_matrix
from utils.sh_utils import SH2RGB
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def load_cfg_namespace(model_path: Path) -> argparse.Namespace | None:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        return None
    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return eval(text, {"Namespace": argparse.Namespace}, {})  # noqa: S307 - trusted local config
    except Exception:
        return None


def resolve_scene_root(args, model_path: Path) -> Path | None:
    if str(getattr(args, "scene_root", "")).strip():
        return Path(args.scene_root).expanduser().resolve()
    cfg = load_cfg_namespace(model_path)
    if cfg is not None and str(getattr(cfg, "source_path", "")).strip():
        return Path(str(cfg.source_path)).expanduser().resolve()
    return None


def resolve_images_subdir(args, model_path: Path) -> str:
    if str(getattr(args, "images_subdir", "")).strip():
        return str(args.images_subdir)
    cfg = load_cfg_namespace(model_path)
    if cfg is not None and str(getattr(cfg, "images", "")).strip():
        return str(cfg.images)
    return "images"


def resolve_white_background(args, model_path: Path) -> bool:
    if bool(getattr(args, "white_background", False)):
        return True
    cfg = load_cfg_namespace(model_path)
    if cfg is not None:
        return bool(getattr(cfg, "white_background", False))
    return False


def stats_from_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def scalar_to_rgb(values: np.ndarray, invert: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = np.clip(1.0 - x if invert else x, 0.0, 1.0)
    r = np.clip(1.8 * x, 0.0, 1.0)
    g = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32, copy=False)


def scalar_image_to_rgb(image_hw: np.ndarray, invert: bool = False) -> np.ndarray:
    value = np.asarray(image_hw, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        norm = np.zeros_like(value, dtype=np.float32)
    else:
        vmax = max(float(np.percentile(finite, 99.0)), 1.0e-6)
        norm = np.clip(value / vmax, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    r = np.clip(1.8 * norm, 0.0, 1.0)
    g = np.clip(1.8 * norm - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * norm, 0.0, 1.0)
    return np.clip(np.stack([r, g, b], axis=-1) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def make_labeled_grid(tiles: Sequence[Tuple[str, np.ndarray]], columns: int, pad: int = 8, label_height: int = 24) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles provided for grid generation.")
    columns = max(1, int(columns))
    sample = tiles[0][1]
    tile_h, tile_w = int(sample.shape[0]), int(sample.shape[1])
    rows = int(math.ceil(len(tiles) / float(columns)))
    canvas_w = columns * tile_w + (columns + 1) * pad
    canvas_h = rows * (tile_h + label_height) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image_np) in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x0 = pad + col * (tile_w + pad)
        y0 = pad + row * (tile_h + label_height + pad)
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + label_height - 1], fill=(32, 32, 32))
        draw.text((x0 + 6, y0 + 4), str(label), fill=(235, 235, 235))
        canvas.paste(Image.fromarray(image_np, mode="RGB"), (x0, y0 + label_height))
    return canvas


def load_triangle_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load a triangle mesh from {mesh_path}")


def query_mesh_surface_by_sampling(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    sample_count: int,
) -> Dict[str, np.ndarray]:
    surface_points, face_ids = trimesh.sample.sample_surface(mesh_obj, max(int(sample_count), 1))
    surface_points = np.asarray(surface_points, dtype=np.float32)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    surface_normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)

    tree = cKDTree(surface_points)
    distance, nearest_ids = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest_ids = np.asarray(nearest_ids, dtype=np.int64)
    nearest_face_ids = face_ids[nearest_ids]
    return {
        "surface_distance": np.asarray(distance, dtype=np.float32),
        "nearest_surface_point": surface_points[nearest_ids].astype(np.float32, copy=False),
        "nearest_surface_normal": surface_normals[nearest_ids].astype(np.float32, copy=False),
        "nearest_face_id": nearest_face_ids.astype(np.int64, copy=False),
        "surface_query_mode_used": "sample",
    }


def query_mesh_surface_exact_open3d(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    import open3d as o3d

    vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
    faces = np.asarray(mesh_obj.faces, dtype=np.int32)
    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    legacy_mesh.triangles = o3d.utility.Vector3iVector(faces)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    closest_points = np.empty_like(points_xyz, dtype=np.float32)
    distances = np.empty((points_xyz.shape[0],), dtype=np.float32)
    face_ids = np.empty((points_xyz.shape[0],), dtype=np.int64)

    chunk = max(int(chunk_size), 1)
    for begin in range(0, points_xyz.shape[0], chunk):
        end = min(begin + chunk, points_xyz.shape[0])
        query_points = o3d.core.Tensor(np.asarray(points_xyz[begin:end], dtype=np.float32), dtype=o3d.core.Dtype.Float32)
        result = scene.compute_closest_points(query_points)
        closest = result["points"].numpy().astype(np.float32, copy=False)
        tri_ids = result["primitive_ids"].numpy().astype(np.int64, copy=False)
        tri_ids = np.clip(tri_ids, 0, len(mesh_obj.faces) - 1)
        closest_points[begin:end] = closest
        distances[begin:end] = np.linalg.norm(closest - points_xyz[begin:end], axis=1).astype(np.float32, copy=False)
        face_ids[begin:end] = tri_ids

    normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)
    return {
        "surface_distance": distances,
        "nearest_surface_point": closest_points,
        "nearest_surface_normal": normals,
        "nearest_face_id": face_ids,
        "surface_query_mode_used": "exact_open3d",
    }


def query_mesh_surface_exact_trimesh(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    query = trimesh.proximity.ProximityQuery(mesh_obj)
    closest_points = np.empty_like(points_xyz, dtype=np.float32)
    distances = np.empty((points_xyz.shape[0],), dtype=np.float32)
    face_ids = np.empty((points_xyz.shape[0],), dtype=np.int64)

    chunk = max(int(chunk_size), 1)
    for begin in range(0, points_xyz.shape[0], chunk):
        end = min(begin + chunk, points_xyz.shape[0])
        closest, distance, tri_ids = query.on_surface(points_xyz[begin:end])
        closest_points[begin:end] = np.asarray(closest, dtype=np.float32)
        distances[begin:end] = np.asarray(distance, dtype=np.float32)
        face_ids[begin:end] = np.asarray(tri_ids, dtype=np.int64)

    normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)
    return {
        "surface_distance": distances,
        "nearest_surface_point": closest_points,
        "nearest_surface_normal": normals,
        "nearest_face_id": face_ids,
        "surface_query_mode_used": "exact_trimesh",
    }


def query_mesh_surface(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    mode: str,
    sample_count: int,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    if mode in {"auto", "exact"}:
        open3d_error = None
        try:
            return query_mesh_surface_exact_open3d(mesh_obj, points_xyz, chunk_size=chunk_size)
        except Exception as exc:
            open3d_error = exc
        try:
            return query_mesh_surface_exact_trimesh(mesh_obj, points_xyz, chunk_size=chunk_size)
        except Exception as exc:
            if mode == "exact":
                raise RuntimeError(
                    "Both exact closest-point backends failed: "
                    f"open3d={open3d_error!r}; trimesh={exc!r}"
                ) from exc
            print(f"[surface-anneal] exact closest-point query failed ({exc}); falling back to sampled query.", flush=True)
    return query_mesh_surface_by_sampling(mesh_obj, points_xyz, sample_count=sample_count)


def face_edge_mean_lengths(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    e01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    return ((e01 + e12 + e20) / 3.0).astype(np.float32, copy=False)


def features_dc_to_rgb(features_dc: torch.Tensor) -> np.ndarray:
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
    return torch.nan_to_num(SH2RGB(sh_dc), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0).detach().cpu().numpy()


def build_tangent_basis(normals: np.ndarray, rotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    axes = np.stack([rotations[:, :, 0], rotations[:, :, 1], rotations[:, :, 2]], axis=1)
    dot = np.sum(axes * normals[:, None, :], axis=2, keepdims=True)
    tangent = axes - dot * normals[:, None, :]
    tangent_norm = np.linalg.norm(tangent, axis=2)
    axis_ids = np.argmax(tangent_norm, axis=1)
    t1 = tangent[np.arange(normals.shape[0]), axis_ids]
    t1_norm = np.linalg.norm(t1, axis=1)

    fallback_x = np.cross(normals, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)[None, :])
    fallback_y = np.cross(normals, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)[None, :])
    fallback = fallback_x
    use_y = np.linalg.norm(fallback_x, axis=1) < 1.0e-6
    fallback[use_y] = fallback_y[use_y]
    fallback_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = fallback / np.clip(fallback_norm, 1.0e-8, None)

    bad = t1_norm < 1.0e-6
    if np.any(bad):
        t1[bad] = fallback[bad]
        t1_norm[bad] = np.linalg.norm(t1[bad], axis=1)
    t1 = t1 / np.clip(t1_norm[:, None], 1.0e-8, None)
    t2 = np.cross(normals, t1)
    t2 = t2 / np.clip(np.linalg.norm(t2, axis=1, keepdims=True), 1.0e-8, None)
    t1 = np.cross(t2, normals)
    t1 = t1 / np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1.0e-8, None)
    return t1.astype(np.float32, copy=False), t2.astype(np.float32, copy=False)


def make_static_copy(base: GaussianModel) -> GaussianModel:
    count = int(base.get_xyz.shape[0])
    model = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    model.active_sh_degree = int(base.active_sh_degree)
    model.spatial_lr_scale = float(base.spatial_lr_scale)
    model._xyz = nn.Parameter(base._xyz.detach().clone().requires_grad_(False))
    model._features_dc = nn.Parameter(base._features_dc.detach().clone().requires_grad_(False))
    model._features_rest = nn.Parameter(base._features_rest.detach().clone().requires_grad_(False))
    model._opacity = nn.Parameter(base._opacity.detach().clone().requires_grad_(False))
    model._scaling = nn.Parameter(base._scaling.detach().clone().requires_grad_(False))
    model._rotation = nn.Parameter(base._rotation.detach().clone().requires_grad_(False))
    if isinstance(base.filter_3D, torch.Tensor) and base.filter_3D.ndim > 0:
        model.filter_3D = base.filter_3D.detach().clone()
    else:
        model.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    model.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    model.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    model.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    model.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    model.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    model.init_tracking_state(count)
    model.restore_tracking_state(base.capture_tracking_state())
    return model


def clone_subset_gaussians(base: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    subset = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[mask].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[mask].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[mask].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[mask].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[mask].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[mask].clone().requires_grad_(False))
    if isinstance(base.filter_3D, torch.Tensor) and base.filter_3D.ndim > 0 and base.filter_3D.shape[0] == base.get_xyz.shape[0]:
        subset.filter_3D = base.filter_3D.detach()[mask].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.init_tracking_state(count)
    subset._source_tag = base._source_tag.detach()[mask].clone()
    subset._seed_id = base._seed_id.detach()[mask].clone()
    subset._generation = base._generation.detach()[mask].clone()
    subset._edge_touched = base._edge_touched.detach()[mask].clone()
    subset._edge_touch_iter = base._edge_touch_iter.detach()[mask].clone()
    return subset


def export_point_cloud(path: Path, points: np.ndarray, colors_rgb: np.ndarray, max_points: int, seed: int) -> int:
    points = np.asarray(points, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.float32)
    if points.shape[0] != colors_rgb.shape[0]:
        raise ValueError(f"Point/color count mismatch: {points.shape[0]} vs {colors_rgb.shape[0]}")
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        ids = np.sort(rng.choice(points.shape[0], size=int(max_points), replace=False))
        points = points[ids]
        colors_rgb = colors_rgb[ids]
    colors_u8 = np.clip(colors_rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    trimesh.points.PointCloud(points, colors=colors_u8).export(path)
    return int(points.shape[0])


def downsample_map(value_hw: torch.Tensor, coarse_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(coarse_hw[0]), int(coarse_hw[1])
    value = value_hw[None, None].float()
    coarse = F.interpolate(value, size=(target_h, target_w), mode="area")
    return coarse[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def accumulate_from_visibility(
    gaussian_ids: np.ndarray,
    weights: np.ndarray,
    metric: np.ndarray,
    sums: np.ndarray,
    denoms: np.ndarray,
) -> None:
    valid = gaussian_ids >= 0
    if not np.any(valid):
        return
    ids = gaussian_ids[valid].astype(np.int64, copy=False)
    w = weights[valid].astype(np.float64, copy=False)
    metric_expanded = np.repeat(metric[..., None], weights.shape[-1], axis=-1)
    vals = metric_expanded[valid].astype(np.float64, copy=False) * w
    np.add.at(sums, ids, vals)
    np.add.at(denoms, ids, w)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively anneal a mip-start Gaussian field toward mesh tangent covariances. "
            "This v0 only rewrites rotation/scale for mesh-supported Gaussians and leaves xyz/color/opacity unchanged."
        )
    )
    parser.add_argument("--model_path", required=True, help="Input mip/SOF model directory containing point_cloud/iteration_x.")
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--output_iteration", type=int, default=-1)
    parser.add_argument("--scene_root", default="")
    parser.add_argument("--images_subdir", default="")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--preview_views", type=int, default=6)
    parser.add_argument("--preview_grid_columns", type=int, default=4)
    parser.add_argument("--preview_point_cap", type=int, default=250000)
    parser.add_argument("--preview_point_seed", type=int, default=0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--surface_query_mode", choices=["auto", "exact", "sample"], default="auto")
    parser.add_argument("--surface_sample_count", type=int, default=4000000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=16384)
    parser.add_argument("--mesh_band_scale_mult", type=float, default=2.0)
    parser.add_argument("--mesh_edge_band_mult", type=float, default=0.60)
    parser.add_argument("--tangent_band_mult", type=float, default=2.0)
    parser.add_argument("--support_distance_band_mult", type=float, default=2.5)
    parser.add_argument("--support_edge_distance_mult", type=float, default=1.25)
    parser.add_argument("--min_surface_conf", type=float, default=0.30)
    parser.add_argument("--max_normal_offset_mult", type=float, default=2.5)
    parser.add_argument("--min_opacity", type=float, default=0.03)
    parser.add_argument("--anneal_strength", type=float, default=0.75)
    parser.add_argument("--surface_conf_power", type=float, default=1.5)
    parser.add_argument("--target_normal_scale_edge_mult", type=float, default=0.35)
    parser.add_argument("--min_normal_scale", type=float, default=1.0e-5)
    parser.add_argument("--cross_term_decay", type=float, default=1.0)
    parser.add_argument("--min_eig_scale", type=float, default=1.0e-6)
    parser.add_argument("--chunk_size", type=int, default=200000)
    parser.add_argument("--artifact_cleanup_payload", default="")
    parser.add_argument("--artifact_cleanup_mask_key", default="prune_mask")
    parser.add_argument("--artifact_guard_max_views", type=int, default=12)
    parser.add_argument("--artifact_guard_depth_min", type=float, default=0.05)
    parser.add_argument("--artifact_guard_screen_margin_px", type=int, default=0)
    parser.add_argument("--artifact_guard_use_effective_scale", type=int, default=1)
    parser.add_argument("--artifact_guard_min_axis_alignment", type=float, default=0.88)
    parser.add_argument("--artifact_guard_min_axis_anisotropy", type=float, default=1.70)
    parser.add_argument("--artifact_guard_min_ray_thickness_ratio", type=float, default=1.80)
    parser.add_argument("--artifact_guard_min_side_explosion", type=float, default=1.70)
    parser.add_argument("--artifact_guard_min_side_radius_px", type=float, default=24.0)
    parser.add_argument("--artifact_guard_min_radius_px", type=float, default=18.0)
    parser.add_argument("--artifact_guard_min_effective_scale_ratio", type=float, default=0.0025)
    parser.add_argument("--artifact_guard_min_volume_radius_ratio", type=float, default=0.0015)
    parser.add_argument("--artifact_guard_min_filter_inflation", type=float, default=1.25)
    parser.add_argument("--artifact_guard_min_filter_scale_ratio", type=float, default=0.60)
    parser.add_argument("--counterfactual_payload", default="")
    parser.add_argument("--counterfactual_mask_key", default="counterfactual_candidate")
    parser.add_argument("--counterfactual_score_key", default="counterfactual_score")
    parser.add_argument("--counterfactual_min_score", type=float, default=-1.0e9)
    parser.add_argument("--render_delta_guard_views", type=int, default=0)
    parser.add_argument("--render_delta_guard_split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--render_delta_guard_downsample", type=int, default=8)
    parser.add_argument("--render_delta_guard_topk", type=int, default=4)
    parser.add_argument("--render_delta_guard_max_visible", type=int, default=60000)
    parser.add_argument("--render_delta_guard_max_patch_radius", type=int, default=2)
    parser.add_argument("--render_delta_guard_min_score", type=float, default=0.06)
    parser.add_argument("--render_delta_guard_quantile", type=float, default=0.97)
    parser.add_argument("--render_delta_guard_max_fraction", type=float, default=0.02)
    parser.add_argument("--quiet", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    safe_state(bool(args.quiet))

    if not torch.cuda.is_available():
        raise RuntimeError("anneal_mip_covariance_to_surface_v0 currently requires CUDA.")

    model_path = Path(args.model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh_path not found: {mesh_path}")
    output_model_path.mkdir(parents=True, exist_ok=True)

    scene_root = resolve_scene_root(args, model_path)
    if scene_root is not None and not scene_root.is_dir():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")
    images_subdir = resolve_images_subdir(args, model_path)
    white_background = resolve_white_background(args, model_path)

    iteration = resolve_iteration(model_path, int(args.iteration))
    output_iteration = int(args.output_iteration) if int(args.output_iteration) >= 0 else int(iteration)

    base = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    output = make_static_copy(base)
    mesh = load_triangle_mesh(str(mesh_path))
    device = base.get_xyz.device

    payload_guard_mask = None
    payload_guard_count = 0
    artifact_payload_path = Path(str(args.artifact_cleanup_payload)).expanduser().resolve() if str(args.artifact_cleanup_payload).strip() else None
    if artifact_payload_path is not None:
        if not artifact_payload_path.is_file():
            raise FileNotFoundError(f"artifact_cleanup_payload not found: {artifact_payload_path}")
        artifact_payload = torch.load(str(artifact_payload_path), map_location="cpu")
        if str(args.artifact_cleanup_mask_key) not in artifact_payload:
            raise KeyError(
                f"artifact_cleanup_payload missing key '{args.artifact_cleanup_mask_key}': {artifact_payload_path}"
            )
        payload_value = artifact_payload[str(args.artifact_cleanup_mask_key)]
        if not torch.is_tensor(payload_value):
            payload_value = torch.as_tensor(payload_value)
        payload_guard_mask = payload_value.reshape(-1).detach().cpu().numpy().astype(bool, copy=False)
        if payload_guard_mask.shape[0] != int(base.get_xyz.shape[0]):
            raise ValueError(
                f"artifact_cleanup mask length mismatch: {payload_guard_mask.shape[0]} vs {int(base.get_xyz.shape[0])}"
            )
        payload_guard_count = int(np.count_nonzero(payload_guard_mask))

    counterfactual_keep_mask = None
    counterfactual_score_np = None
    counterfactual_tested_mask = None
    counterfactual_info: Dict[str, object] = {
        "enabled": False,
        "payload_path": "",
        "mask_key": str(args.counterfactual_mask_key),
        "score_key": str(args.counterfactual_score_key),
        "min_score": float(args.counterfactual_min_score),
    }
    counterfactual_payload_path = (
        Path(str(args.counterfactual_payload)).expanduser().resolve()
        if str(args.counterfactual_payload).strip()
        else None
    )
    if counterfactual_payload_path is not None:
        if not counterfactual_payload_path.is_file():
            raise FileNotFoundError(f"counterfactual_payload not found: {counterfactual_payload_path}")
        counterfactual_payload = torch.load(str(counterfactual_payload_path), map_location="cpu")
        if str(args.counterfactual_mask_key) not in counterfactual_payload:
            raise KeyError(
                f"counterfactual_payload missing key '{args.counterfactual_mask_key}': {counterfactual_payload_path}"
            )
        mask_value = counterfactual_payload[str(args.counterfactual_mask_key)]
        if not torch.is_tensor(mask_value):
            mask_value = torch.as_tensor(mask_value)
        counterfactual_keep_mask = mask_value.reshape(-1).detach().cpu().numpy().astype(bool, copy=False)
        if counterfactual_keep_mask.shape[0] != int(base.get_xyz.shape[0]):
            raise ValueError(
                f"counterfactual mask length mismatch: {counterfactual_keep_mask.shape[0]} vs {int(base.get_xyz.shape[0])}"
            )

        score_key = str(args.counterfactual_score_key).strip()
        if score_key and score_key in counterfactual_payload:
            score_value = counterfactual_payload[score_key]
            if not torch.is_tensor(score_value):
                score_value = torch.as_tensor(score_value)
            counterfactual_score_np = score_value.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            if counterfactual_score_np.shape[0] != int(base.get_xyz.shape[0]):
                raise ValueError(
                    f"counterfactual score length mismatch: {counterfactual_score_np.shape[0]} vs {int(base.get_xyz.shape[0])}"
                )
            counterfactual_keep_mask = counterfactual_keep_mask & (
                counterfactual_score_np >= float(args.counterfactual_min_score)
            )

        tested_key = "tested_candidate"
        tested_count = 0
        if tested_key in counterfactual_payload:
            tested_value = counterfactual_payload[tested_key]
            if not torch.is_tensor(tested_value):
                tested_value = torch.as_tensor(tested_value)
            tested_mask_np = tested_value.reshape(-1).detach().cpu().numpy().astype(bool, copy=False)
            if tested_mask_np.shape[0] == int(base.get_xyz.shape[0]):
                counterfactual_tested_mask = tested_mask_np
                tested_count = int(np.count_nonzero(tested_mask_np))
                # Treat the counterfactual payload as a veto set, not a global whitelist:
                # explicitly rejected tested gaussians are blocked, while untested ones stay eligible.
                counterfactual_keep_mask = counterfactual_keep_mask | (~counterfactual_tested_mask)

        counterfactual_info.update(
            {
                "enabled": True,
                "payload_path": str(counterfactual_payload_path),
                "keep_count": int(np.count_nonzero(counterfactual_keep_mask)),
                "reject_count": int(counterfactual_keep_mask.shape[0] - np.count_nonzero(counterfactual_keep_mask)),
                "tested_count": int(tested_count),
                "mode": "tested_reject_veto",
                "score_stats_kept": (
                    stats_from_array(counterfactual_score_np[counterfactual_keep_mask])
                    if counterfactual_score_np is not None
                    else {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
                ),
                "score_stats_rejected": (
                    stats_from_array(counterfactual_score_np[~counterfactual_keep_mask])
                    if counterfactual_score_np is not None
                    else {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
                ),
            }
        )

    xyz_np = base.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    raw_scale_np = base.get_scaling.detach().cpu().numpy().astype(np.float32, copy=False)
    effective_scale_np = base.get_scaling_with_3D_filter.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity_np = base.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
    source_tag_np = base._source_tag.detach().cpu().numpy().astype(np.int32, copy=False)

    surface_query = query_mesh_surface(
        mesh,
        xyz_np,
        mode=str(args.surface_query_mode),
        sample_count=int(args.surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )
    nearest_distance = np.asarray(surface_query["surface_distance"], dtype=np.float32)
    nearest_point = np.asarray(surface_query["nearest_surface_point"], dtype=np.float32)
    nearest_normal = np.asarray(surface_query["nearest_surface_normal"], dtype=np.float32)
    nearest_face_id = np.asarray(surface_query["nearest_face_id"], dtype=np.int64)

    delta = xyz_np - nearest_point
    signed_distance = np.sum(delta * nearest_normal, axis=1).astype(np.float32, copy=False)
    normal_offset = np.abs(signed_distance).astype(np.float32, copy=False)
    tangent_sq = np.clip(np.square(nearest_distance) - np.square(normal_offset), 0.0, None)
    tangent_offset = np.sqrt(tangent_sq).astype(np.float32, copy=False)

    face_edge = face_edge_mean_lengths(mesh)
    local_mesh_resolution = face_edge[np.clip(nearest_face_id, 0, max(len(face_edge) - 1, 0))].astype(np.float32, copy=False)
    mean_effective_scale = np.mean(effective_scale_np, axis=1).astype(np.float32, copy=False)
    surface_band = np.maximum(mean_effective_scale * float(args.mesh_band_scale_mult), local_mesh_resolution * float(args.mesh_edge_band_mult))
    surface_band = np.maximum(surface_band, float(args.min_normal_scale)).astype(np.float32, copy=False)
    tangent_band = np.maximum(surface_band * float(args.tangent_band_mult), local_mesh_resolution).astype(np.float32, copy=False)
    support_distance = np.maximum(
        surface_band * float(args.support_distance_band_mult),
        local_mesh_resolution * float(args.support_edge_distance_mult),
    ).astype(np.float32, copy=False)

    normal_score = np.exp(-0.5 * np.square(normal_offset / np.clip(surface_band, 1.0e-8, None))).astype(np.float32, copy=False)
    tangent_score = np.exp(-0.5 * np.square(tangent_offset / np.clip(tangent_band, 1.0e-8, None))).astype(np.float32, copy=False)
    surface_conf = (normal_score * tangent_score).astype(np.float32, copy=False)

    artifact_guard_mask = np.zeros((xyz_np.shape[0],), dtype=bool)
    artifact_guard_info: Dict[str, object] = {
        "enabled": bool(scene_root is not None and int(args.artifact_guard_max_views) > 0),
        "payload_enabled": bool(payload_guard_mask is not None),
        "payload_guard_count": int(payload_guard_count),
    }
    if scene_root is not None and int(args.artifact_guard_max_views) > 0:
        guard_cameras = load_cameras_for_split(scene_root, model_path, images_subdir, "train")
        guard_cameras = select_uniform(guard_cameras, int(args.artifact_guard_max_views))
        if guard_cameras:
            geom = gaussian_geometry(base, use_effective_scale=bool(int(args.artifact_guard_use_effective_scale)))
            scene_diag = robust_scene_diag(geom["xyz"])
            guard_background = torch.zeros((3,), dtype=torch.float32, device=device)
            view_guard = collect_view_aligned_stats(
                base,
                guard_cameras,
                geom,
                depth_min=float(args.artifact_guard_depth_min),
                screen_margin_px=int(args.artifact_guard_screen_margin_px),
                chunk_size=max(32768, int(args.chunk_size)),
                background=guard_background,
            )
            min_effective_scale = max(float(scene_diag) * float(args.artifact_guard_min_effective_scale_ratio), 1e-8)
            min_volume_radius = max(float(scene_diag) * float(args.artifact_guard_min_volume_radius_ratio), 1e-8)
            visible = view_guard["visible_count"] >= 1
            view_aligned = (
                (view_guard["axis_alignment_mean"] >= float(args.artifact_guard_min_axis_alignment))
                | (view_guard["ray_thickness_ratio"] >= float(args.artifact_guard_min_ray_thickness_ratio))
            )
            large = (
                (geom["scale_max"] >= min_effective_scale)
                | (geom["volume_radius"] >= min_volume_radius)
                | (view_guard["radius_max"] >= float(args.artifact_guard_min_radius_px))
            )
            side_bad = (
                (view_guard["side_explosion_max"] >= float(args.artifact_guard_min_side_explosion))
                | (view_guard["side_radius_max"] >= float(args.artifact_guard_min_side_radius_px))
            )
            filter_bad = (
                (geom["filter_inflation"] >= float(args.artifact_guard_min_filter_inflation))
                | (geom["filter_scale_ratio"] >= float(args.artifact_guard_min_filter_scale_ratio))
            )
            footprint_bad = side_bad | (view_guard["radius_max"] >= float(args.artifact_guard_min_radius_px))
            artifact_guard_mask = (
                visible
                & large
                & view_aligned
                & (footprint_bad | filter_bad)
                & (side_bad | filter_bad | (geom["axis_anisotropy"] >= float(args.artifact_guard_min_axis_anisotropy)))
            )
            artifact_guard_info.update(
                {
                    "guard_camera_count": int(len(guard_cameras)),
                    "scene_diag": float(scene_diag),
                    "artifact_guard_count": int(np.count_nonzero(artifact_guard_mask)),
                    "visible_count": int(np.count_nonzero(visible)),
                    "view_aligned_count": int(np.count_nonzero(view_aligned)),
                    "large_count": int(np.count_nonzero(large)),
                    "side_bad_count": int(np.count_nonzero(side_bad)),
                    "filter_bad_count": int(np.count_nonzero(filter_bad)),
                    "footprint_bad_count": int(np.count_nonzero(footprint_bad)),
                    "stats_radius_max_guarded": stats_from_array(view_guard["radius_max"][artifact_guard_mask]),
                    "stats_side_radius_guarded": stats_from_array(view_guard["side_radius_max"][artifact_guard_mask]),
                    "stats_side_explosion_guarded": stats_from_array(view_guard["side_explosion_max"][artifact_guard_mask]),
                    "stats_filter_inflation_guarded": stats_from_array(geom["filter_inflation"][artifact_guard_mask]),
                }
            )
        else:
            artifact_guard_info["guard_camera_count"] = 0

    if payload_guard_mask is not None:
        artifact_guard_mask = artifact_guard_mask | payload_guard_mask

    allow_cov_flatten = (
        (nearest_face_id >= 0)
        & (opacity_np >= float(args.min_opacity))
        & (nearest_distance <= support_distance)
        & (normal_offset <= surface_band * float(args.max_normal_offset_mult))
        & (surface_conf >= float(args.min_surface_conf))
    )
    allow_cov_flatten &= ~artifact_guard_mask
    if counterfactual_keep_mask is not None:
        allow_cov_flatten &= counterfactual_keep_mask

    rotations_np = build_rotation(base._rotation.detach()).cpu().numpy().astype(np.float32, copy=False)
    tangent_u_np, tangent_v_np = build_tangent_basis(nearest_normal, rotations_np)

    count = int(xyz_np.shape[0])
    basis_t = torch.from_numpy(np.stack([tangent_u_np, tangent_v_np, nearest_normal], axis=2)).to(device=device, dtype=torch.float32)
    scale_t = base.get_scaling.detach().clone()
    rotation_t = base.get_rotation.detach().clone()
    allow_t = torch.from_numpy(allow_cov_flatten.astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    conf_t = torch.from_numpy(np.clip(surface_conf, 0.0, 1.0)).to(device=device, dtype=torch.float32)
    target_sigma_floor_np = np.maximum(
        local_mesh_resolution * float(args.target_normal_scale_edge_mult),
        float(args.min_normal_scale),
    ).astype(np.float32, copy=False)
    target_sigma_floor_t = torch.from_numpy(target_sigma_floor_np).to(device=device, dtype=torch.float32)

    scaling_new = output._scaling.detach().clone()
    rotation_new = output._rotation.detach().clone()
    sigma_normal_before = np.zeros((count,), dtype=np.float32)
    sigma_normal_after = np.zeros((count,), dtype=np.float32)
    normal_align_before = np.zeros((count,), dtype=np.float32)
    normal_align_after = np.zeros((count,), dtype=np.float32)
    blend_weight_np = np.zeros((count,), dtype=np.float32)
    render_delta_guard_mask = np.zeros((count,), dtype=bool)
    render_delta_guard_score = np.zeros((count,), dtype=np.float32)
    render_delta_guard_info: Dict[str, object] = {
        "enabled": bool(scene_root is not None and int(args.render_delta_guard_views) > 0),
        "guard_count": 0,
        "selected_view_count": 0,
    }

    chunk_size = max(int(args.chunk_size), 1)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        R = build_rotation(rotation_t[start:end])
        scales = scale_t[start:end]
        basis = basis_t[start:end]
        cov = R @ torch.diag_embed(scales * scales) @ R.transpose(1, 2)
        local = basis.transpose(1, 2) @ cov @ basis
        sigma_old = torch.sqrt(torch.clamp(local[:, 2, 2], min=1.0e-12))
        target_sigma = torch.minimum(sigma_old, target_sigma_floor_t[start:end])

        local_target = local.clone()
        local_target[:, 0, 2] = 0.0
        local_target[:, 1, 2] = 0.0
        local_target[:, 2, 0] = 0.0
        local_target[:, 2, 1] = 0.0
        local_target[:, 2, 2] = target_sigma * target_sigma

        weight = float(args.anneal_strength) * allow_t[start:end] * torch.pow(conf_t[start:end], float(args.surface_conf_power))
        sigma_shrink_ratio = torch.clamp((sigma_old - target_sigma) / torch.clamp(sigma_old, min=1.0e-8), min=0.0, max=1.0)
        weight = weight * (sigma_shrink_ratio + (1.0 - sigma_shrink_ratio) * float(args.cross_term_decay))
        weight = torch.clamp(weight, 0.0, 1.0)
        local_new = local + weight[:, None, None] * (local_target - local)
        local_new = 0.5 * (local_new + local_new.transpose(1, 2))

        cov_new = basis @ local_new @ basis.transpose(1, 2)
        evals, evecs = torch.linalg.eigh(cov_new)
        evals = torch.clamp(evals, min=float(args.min_eig_scale) ** 2)
        det = torch.linalg.det(evecs)
        flip = det < 0.0
        if torch.any(flip):
            evecs[flip, :, 0] *= -1.0
        quat = _quaternion_from_rotation_matrix(evecs)
        log_scales = torch.log(torch.sqrt(evals))

        scaling_new[start:end] = log_scales
        rotation_new[start:end] = quat

        sigma_new = torch.sqrt(torch.clamp(local_new[:, 2, 2], min=1.0e-12))
        min_idx_before = torch.argmin(scales, dim=1, keepdim=True)
        min_axis_before = torch.gather(R, 2, min_idx_before[:, None, :].expand(-1, 3, 1)).squeeze(2)
        min_axis_after = evecs[:, :, 0]
        normal_chunk = basis[:, :, 2]
        sigma_old_np = sigma_old.detach().cpu().numpy().astype(np.float32, copy=False)
        sigma_normal_before[start:end] = sigma_old_np
        sigma_normal_after[start:end] = sigma_new.detach().cpu().numpy().astype(np.float32, copy=False)
        normal_align_before[start:end] = torch.abs(torch.sum(min_axis_before * normal_chunk, dim=1)).detach().cpu().numpy().astype(np.float32, copy=False)
        normal_align_after[start:end] = torch.abs(torch.sum(min_axis_after * normal_chunk, dim=1)).detach().cpu().numpy().astype(np.float32, copy=False)
        blend_weight_np[start:end] = weight.detach().cpu().numpy().astype(np.float32, copy=False)

    if scene_root is not None and int(args.render_delta_guard_views) > 0 and np.any(allow_cov_flatten):
        delta_cameras = load_cameras_for_split(scene_root, model_path, images_subdir, str(args.render_delta_guard_split))
        delta_cameras = select_uniform(delta_cameras, int(args.render_delta_guard_views))
        if delta_cameras:
            preview_background = (
                torch.ones((3,), dtype=torch.float32, device=device)
                if bool(white_background)
                else torch.zeros((3,), dtype=torch.float32, device=device)
            )
            tentative_output = make_static_copy(base)
            tentative_output._scaling = nn.Parameter(scaling_new.detach().clone().requires_grad_(False))
            tentative_output._rotation = nn.Parameter(rotation_new.detach().clone().requires_grad_(False))
            selected_mask_t = torch.from_numpy(allow_cov_flatten).to(device=device, dtype=torch.bool)
            selected_subset = clone_subset_gaussians(tentative_output, selected_mask_t)
            subset_scores = np.zeros((int(selected_subset.get_xyz.shape[0]),), dtype=np.float64)
            subset_denoms = np.zeros((int(selected_subset.get_xyz.shape[0]),), dtype=np.float64)

            vis_cfg = VisibilityRecordConfig(
                downsample=int(args.render_delta_guard_downsample),
                topk=int(args.render_delta_guard_topk),
                max_visible_per_view=int(args.render_delta_guard_max_visible),
                min_opacity=0.0,
                min_depth=0.05,
                max_patch_radius=int(args.render_delta_guard_max_patch_radius),
            )

            for camera in delta_cameras:
                base_pkg = render_simple(camera, base, preview_background)
                tentative_pkg = render_simple(camera, tentative_output, preview_background)
                subset_pkg = render_simple(camera, selected_subset, preview_background)

                delta_hw = torch.mean(torch.abs(tentative_pkg["render"] - base_pkg["render"]), dim=0)
                records = build_coarse_visibility_records(
                    selected_subset,
                    [camera],
                    [subset_pkg],
                    image_hw=(int(camera.image_height), int(camera.image_width)),
                    cfg=vis_cfg,
                )
                coarse_h, coarse_w = [int(v) for v in records["coarse_hw"].tolist()]
                delta_coarse = downsample_map(delta_hw, (coarse_h, coarse_w))
                accumulate_from_visibility(
                    records["gaussian_ids"][0, 0].numpy(),
                    records["weights"][0, 0].numpy(),
                    delta_coarse,
                    subset_scores,
                    subset_denoms,
                )

            subset_score = (subset_scores / np.maximum(subset_denoms, 1e-8)).astype(np.float32, copy=False)
            valid_subset = np.isfinite(subset_score)
            if np.any(valid_subset):
                threshold = float(
                    np.quantile(
                        subset_score[valid_subset],
                        float(np.clip(args.render_delta_guard_quantile, 0.0, 1.0)),
                    )
                )
                flagged_subset = valid_subset & (
                    (subset_score >= max(threshold, float(args.render_delta_guard_min_score)))
                )
                max_guard_count = 0
                if float(args.render_delta_guard_max_fraction) > 0.0:
                    max_guard_count = max(
                        1,
                        int(round(float(np.count_nonzero(allow_cov_flatten)) * float(args.render_delta_guard_max_fraction))),
                    )
                if max_guard_count > 0 and int(np.count_nonzero(flagged_subset)) > max_guard_count:
                    ids = np.flatnonzero(flagged_subset).astype(np.int64, copy=False)
                    order = np.argsort(-subset_score[ids], kind="stable")[:max_guard_count]
                    capped = np.zeros_like(flagged_subset, dtype=bool)
                    capped[ids[order]] = True
                    flagged_subset = capped

                selected_ids = np.flatnonzero(allow_cov_flatten).astype(np.int64, copy=False)
                flagged_ids = selected_ids[flagged_subset]
                render_delta_guard_mask[flagged_ids] = True
                render_delta_guard_score[selected_ids] = subset_score
                scaling_new[torch.from_numpy(flagged_ids).to(device=device, dtype=torch.int64)] = base._scaling.detach()[
                    torch.from_numpy(flagged_ids).to(device=device, dtype=torch.int64)
                ]
                rotation_new[torch.from_numpy(flagged_ids).to(device=device, dtype=torch.int64)] = base._rotation.detach()[
                    torch.from_numpy(flagged_ids).to(device=device, dtype=torch.int64)
                ]
                blend_weight_np[flagged_ids] = 0.0
                sigma_normal_after[flagged_ids] = sigma_normal_before[flagged_ids]
                normal_align_after[flagged_ids] = normal_align_before[flagged_ids]
                allow_cov_flatten[flagged_ids] = False
                render_delta_guard_info.update(
                    {
                        "guard_count": int(np.count_nonzero(render_delta_guard_mask)),
                        "selected_view_count": int(len(delta_cameras)),
                        "threshold": float(threshold),
                        "score_stats_flagged": stats_from_array(render_delta_guard_score[render_delta_guard_mask]),
                        "score_stats_selected": stats_from_array(subset_score[valid_subset]),
                    }
                )

    output._scaling = nn.Parameter(scaling_new.detach().clone().requires_grad_(False))
    output._rotation = nn.Parameter(rotation_new.detach().clone().requires_grad_(False))

    copy_render_config(model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(output_iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    output.save_ply(str(point_dir / "point_cloud.ply"))
    output.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    support_rgb = scalar_to_rgb(surface_conf, invert=False)
    weight_rgb = scalar_to_rgb(blend_weight_np, invert=False)
    sigma_ratio = np.clip(sigma_normal_after / np.clip(sigma_normal_before, 1.0e-8, None), 0.0, 1.0)
    sigma_ratio_rgb = scalar_to_rgb(1.0 - sigma_ratio, invert=False)
    preview_dir = output_model_path / "surface_anneal_previews_v0"
    preview_dir.mkdir(parents=True, exist_ok=True)
    point_preview_dir = preview_dir / "point_cloud_previews"
    point_preview_dir.mkdir(parents=True, exist_ok=True)
    point_preview_counts = {
        "surface_confidence": export_point_cloud(
            point_preview_dir / "surface_confidence_preview_v0.ply",
            xyz_np,
            support_rgb,
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "blend_weight": export_point_cloud(
            point_preview_dir / "blend_weight_preview_v0.ply",
            xyz_np,
            weight_rgb,
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "artifact_guard": export_point_cloud(
            point_preview_dir / "artifact_guard_preview_v0.ply",
            xyz_np,
            scalar_to_rgb(artifact_guard_mask.astype(np.float32, copy=False), invert=False),
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "render_delta_guard": export_point_cloud(
            point_preview_dir / "render_delta_guard_preview_v0.ply",
            xyz_np,
            scalar_to_rgb(render_delta_guard_mask.astype(np.float32, copy=False), invert=False),
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "counterfactual_keep": export_point_cloud(
            point_preview_dir / "counterfactual_keep_preview_v0.ply",
            xyz_np,
            scalar_to_rgb(
                (
                    counterfactual_keep_mask.astype(np.float32, copy=False)
                    if counterfactual_keep_mask is not None
                    else np.ones((count,), dtype=np.float32)
                ),
                invert=False,
            ),
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
        "sigma_shrink": export_point_cloud(
            point_preview_dir / "sigma_shrink_preview_v0.ply",
            xyz_np,
            sigma_ratio_rgb,
            max_points=int(args.preview_point_cap),
            seed=int(args.preview_point_seed),
        ),
    }

    payload = {
        "surface_class": torch.from_numpy(allow_cov_flatten.astype(np.int16, copy=False)),
        "allow_cov_flatten": torch.from_numpy(allow_cov_flatten.copy()),
        "artifact_guard_mask": torch.from_numpy(artifact_guard_mask.copy()),
        "counterfactual_keep_mask": torch.from_numpy(
            (
                counterfactual_keep_mask.copy()
                if counterfactual_keep_mask is not None
                else np.ones((count,), dtype=bool)
            )
        ),
        "render_delta_guard_mask": torch.from_numpy(render_delta_guard_mask.copy()),
        "render_delta_guard_score": torch.from_numpy(render_delta_guard_score.astype(np.float32, copy=False)),
        "counterfactual_tested_mask": torch.from_numpy(
            (
                counterfactual_tested_mask.copy()
                if counterfactual_tested_mask is not None
                else np.zeros((count,), dtype=bool)
            )
        ),
        "counterfactual_score": torch.from_numpy(
            (
                counterfactual_score_np.astype(np.float32, copy=False)
                if counterfactual_score_np is not None
                else np.zeros((count,), dtype=np.float32)
            )
        ),
        "surface_conf": torch.from_numpy(surface_conf.astype(np.float32, copy=False)),
        "blend_weight": torch.from_numpy(blend_weight_np.astype(np.float32, copy=False)),
        "nearest_surface_point": torch.from_numpy(nearest_point.astype(np.float32, copy=False)),
        "nearest_surface_normal": torch.from_numpy(nearest_normal.astype(np.float32, copy=False)),
        "tangent_u": torch.from_numpy(tangent_u_np.astype(np.float32, copy=False)),
        "tangent_v": torch.from_numpy(tangent_v_np.astype(np.float32, copy=False)),
        "nearest_face_id": torch.from_numpy(nearest_face_id.astype(np.int64, copy=False)),
        "surface_distance": torch.from_numpy(nearest_distance.astype(np.float32, copy=False)),
        "signed_distance": torch.from_numpy(signed_distance.astype(np.float32, copy=False)),
        "surface_normal_offset": torch.from_numpy(normal_offset.astype(np.float32, copy=False)),
        "surface_tangent_offset": torch.from_numpy(tangent_offset.astype(np.float32, copy=False)),
        "surface_band": torch.from_numpy(surface_band.astype(np.float32, copy=False)),
        "tangent_band": torch.from_numpy(tangent_band.astype(np.float32, copy=False)),
        "support_distance": torch.from_numpy(support_distance.astype(np.float32, copy=False)),
        "local_mesh_resolution": torch.from_numpy(local_mesh_resolution.astype(np.float32, copy=False)),
        "old_xyz": base.get_xyz.detach().cpu(),
        "old_opacity": base.get_opacity.detach().cpu().reshape(-1),
        "old_scale": base.get_scaling.detach().cpu(),
        "old_rotation": base.get_rotation.detach().cpu(),
        "target_normal_scale": torch.from_numpy(np.minimum(sigma_normal_before, target_sigma_floor_np).astype(np.float32, copy=False)),
        "sigma_normal_before": torch.from_numpy(sigma_normal_before.astype(np.float32, copy=False)),
        "sigma_normal_after": torch.from_numpy(sigma_normal_after.astype(np.float32, copy=False)),
        "normal_align_before": torch.from_numpy(normal_align_before.astype(np.float32, copy=False)),
        "normal_align_after": torch.from_numpy(normal_align_after.astype(np.float32, copy=False)),
        "new_scale": output.get_scaling.detach().cpu(),
        "new_rotation": output.get_rotation.detach().cpu(),
        "source_tag": torch.from_numpy(source_tag_np.copy()),
        "surface_query_mode_used": str(surface_query["surface_query_mode_used"]),
    }
    payload_path = output_model_path / "gaussian_surface_payload_v0.pt"
    torch.save(payload, str(payload_path))

    render_summaries: List[Dict[str, object]] = []
    overview_tiles: List[Tuple[str, np.ndarray]] = []
    if scene_root is not None and int(args.preview_views) > 0:
        cameras = load_cameras_for_split(scene_root, model_path, images_subdir, str(args.split))
        cameras = select_uniform(cameras, int(args.preview_views))
        background = torch.ones((3,), dtype=torch.float32, device=device) if bool(white_background) else torch.zeros((3,), dtype=torch.float32, device=device)
        render_dir = preview_dir / "render_previews"
        render_dir.mkdir(parents=True, exist_ok=True)
        selected_mask_t = torch.from_numpy(allow_cov_flatten).to(device=device, dtype=torch.bool)
        selected_subset = clone_subset_gaussians(output, selected_mask_t) if bool(torch.any(selected_mask_t)) else None
        selected_override = (
            torch.from_numpy(support_rgb[allow_cov_flatten]).to(device=device, dtype=torch.float32)
            if selected_subset is not None
            else None
        )

        for view_idx, camera in enumerate(cameras):
            before_pkg = render_simple(camera, base, background)
            after_pkg = render_simple(camera, output, background)
            support_pkg = (
                render_simple(camera, selected_subset, background, override_color=selected_override)
                if selected_subset is not None
                else None
            )

            before_rgb = before_pkg["render"]
            after_rgb = after_pkg["render"]
            support_rgb_img = (
                support_pkg["render"]
                if support_pkg is not None
                else torch.zeros_like(before_rgb)
            )
            delta_hw = torch.mean(torch.abs(after_rgb - before_rgb), dim=0)

            before_u8 = to_uint8_rgb(before_rgb)
            after_u8 = to_uint8_rgb(after_rgb)
            support_u8 = to_uint8_rgb(support_rgb_img)
            delta_u8 = scalar_image_to_rgb(delta_hw.detach().cpu().numpy(), invert=False)

            view_name = Path(str(camera.image_name)).stem or str(camera.image_name)
            view_dir = render_dir / f"{view_idx:03d}_{view_name}"
            view_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(before_u8, mode="RGB").save(view_dir / "base_render.png")
            Image.fromarray(after_u8, mode="RGB").save(view_dir / "annealed_render.png")
            Image.fromarray(support_u8, mode="RGB").save(view_dir / "surface_support_render.png")
            Image.fromarray(delta_u8, mode="RGB").save(view_dir / "render_delta_heatmap.png")

            grid = make_labeled_grid(
                [
                    ("base", before_u8),
                    ("annealed", after_u8),
                    ("delta", delta_u8),
                    ("surface_support", support_u8),
                ],
                columns=max(1, min(4, int(args.preview_grid_columns))),
            )
            grid.save(view_dir / "comparison_grid.png")
            overview_tiles.append((f"{view_idx:03d}_{str(camera.image_name)}", np.asarray(grid.convert("RGB"), dtype=np.uint8)))
            render_summaries.append(
                {
                    "view_index": int(view_idx),
                    "image_name": str(camera.image_name),
                    "delta_mean": float(delta_hw.mean().item()),
                    "delta_p95": float(np.percentile(delta_hw.detach().cpu().numpy().reshape(-1), 95.0)),
                    "output_dir": str(view_dir.resolve()),
                }
            )

        if overview_tiles:
            overview = make_labeled_grid(overview_tiles, columns=1)
            overview.save(preview_dir / "comparison_overview_v0.png")

    summary = {
        "mode": "anneal_mip_covariance_to_surface_v0",
        "model_path": str(model_path),
        "mesh_path": str(mesh_path),
        "scene_root": "" if scene_root is None else str(scene_root),
        "images_subdir": str(images_subdir),
        "iteration": int(iteration),
        "output_model_path": str(output_model_path),
        "output_iteration": int(output_iteration),
        "surface_query_mode_used": str(surface_query["surface_query_mode_used"]),
        "count": int(count),
        "allow_cov_flatten_count": int(np.count_nonzero(allow_cov_flatten)),
        "allow_cov_flatten_ratio": float(np.count_nonzero(allow_cov_flatten) / max(count, 1)),
        "artifact_guard_count": int(np.count_nonzero(artifact_guard_mask)),
        "counterfactual_keep_count": (
            int(np.count_nonzero(counterfactual_keep_mask))
            if counterfactual_keep_mask is not None
            else int(count)
        ),
        "render_delta_guard_count": int(np.count_nonzero(render_delta_guard_mask)),
        "residual_count": int(count - np.count_nonzero(allow_cov_flatten)),
        "params": vars(args),
        "artifact_guard": artifact_guard_info,
        "counterfactual_guard": counterfactual_info,
        "render_delta_guard": render_delta_guard_info,
        "stats": {
            "surface_conf_all": stats_from_array(surface_conf),
            "surface_conf_selected": stats_from_array(surface_conf[allow_cov_flatten]),
            "nearest_distance_selected": stats_from_array(nearest_distance[allow_cov_flatten]),
            "normal_offset_selected": stats_from_array(normal_offset[allow_cov_flatten]),
            "tangent_offset_selected": stats_from_array(tangent_offset[allow_cov_flatten]),
            "sigma_normal_before_selected": stats_from_array(sigma_normal_before[allow_cov_flatten]),
            "sigma_normal_after_selected": stats_from_array(sigma_normal_after[allow_cov_flatten]),
            "normal_align_before_selected": stats_from_array(normal_align_before[allow_cov_flatten]),
            "normal_align_after_selected": stats_from_array(normal_align_after[allow_cov_flatten]),
            "blend_weight_selected": stats_from_array(blend_weight_np[allow_cov_flatten]),
        },
        "source_tag_counts": {
            "original": int(np.count_nonzero(source_tag_np == 0)),
            "prior_injected": int(np.count_nonzero(source_tag_np == 1)),
            "extension_probe": int(np.count_nonzero(source_tag_np == 2)),
        },
        "point_preview_counts": point_preview_counts,
        "render_previews": render_summaries,
        "artifacts": {
            "surface_payload": str(payload_path.resolve()),
            "output_ply": str((point_dir / "point_cloud.ply").resolve()),
            "output_tags": str((point_dir / "gaussian_tags.pt").resolve()),
            "preview_dir": str(preview_dir.resolve()),
        },
        "note": (
            "This v0 stage only anneals covariance for mesh-supported Gaussians. "
            "Centers, opacity, DC/SH color, and unsupported residual Gaussians are left unchanged by design, "
            "so off-mesh compensation structure can continue to preserve render quality while the supported subset "
            "becomes thinner and more tangent-plane aligned."
        ),
    }
    summary_path = output_model_path / "anneal_mip_covariance_to_surface_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved summary to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
