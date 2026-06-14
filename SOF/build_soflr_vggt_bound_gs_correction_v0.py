from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.prior_injection import index_image_dir, load_rgb_image, normalize_image_name
from utils.sof_mesh_patch_enhancer_v0 import load_triangle_mesh, normalize_np, points_to_barycentric, stats_from_array
from utils.vggt_adapter import FrozenVGGTAdapter, VGGTAdapterConfig


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, data_device: str) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=False,
        data_device=data_device,
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def _resolve_iteration(model_path: Path, iteration: int) -> int:
    if int(iteration) >= 0:
        return int(iteration)
    point_root = model_path / "point_cloud"
    candidates: List[int] = []
    for child in point_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_root}")
    return max(candidates)


def _select_uniform(items: Sequence[object], max_items: int) -> List[object]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    ids = np.unique(np.linspace(0, len(items) - 1, num=max_items, dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()]


def _resize_chw(image_chw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(image_chw.shape[-2:]) == tuple(target_hw):
        return image_chw
    image = image_chw.unsqueeze(0).float()
    return F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)[0]


def _load_image_chw(path: Path, target_hw: Tuple[int, int]) -> torch.Tensor:
    image = load_rgb_image(path).permute(2, 0, 1).contiguous()
    return _resize_chw(image, target_hw)


def _camera_name_index(cameras: Sequence[object]) -> Dict[str, object]:
    return {normalize_image_name(cam.image_name): cam for cam in cameras}


def _lookup_image(image_index: Dict[str, Path], image_name: str) -> Path:
    candidates = [
        normalize_image_name(image_name),
        Path(image_name).stem,
        Path(image_name).name,
        image_name,
    ]
    for candidate in candidates:
        if candidate in image_index:
            return image_index[candidate]
    lower = {key.lower(): value for key, value in image_index.items()}
    for candidate in candidates:
        value = lower.get(str(candidate).lower())
        if value is not None:
            return value
    raise FileNotFoundError(f"No image found for camera name '{image_name}'")


def _project_points_camera(cam, points_xyz: np.ndarray, depth_min: float) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(cam.R, dtype=np.float32)
    t = np.asarray(cam.T, dtype=np.float32)
    xyz_cam = points_xyz @ r + t[None, :]
    z = xyz_cam[:, 2]
    safe_z = np.clip(z, 1e-6, None)
    x = xyz_cam[:, 0] / safe_z * float(cam.focal_x) + float(cam.image_width) / 2.0
    y = xyz_cam[:, 1] / safe_z * float(cam.focal_y) + float(cam.image_height) / 2.0
    valid = z > float(depth_min)
    valid &= x >= 0.0
    valid &= x < float(cam.image_width)
    valid &= y >= 0.0
    valid &= y < float(cam.image_height)
    return np.stack([x, y, z], axis=1).astype(np.float32), valid


def _sample_nearest_2d(map_hw: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = map_hw.shape
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, height - 1)
    return map_hw[y, x].astype(np.float32, copy=False)


def _robust_align_depth(
    vggt_depth: np.ndarray,
    ref_depth: np.ndarray,
    mask: np.ndarray,
    min_pixels: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    valid = mask & np.isfinite(vggt_depth) & np.isfinite(ref_depth) & (vggt_depth > 1e-6) & (ref_depth > 1e-6)
    if int(valid.sum()) < int(min_pixels):
        return vggt_depth.astype(np.float32, copy=False), {
            "mode": "identity_insufficient_pixels",
            "pixels": int(valid.sum()),
            "scale": 1.0,
            "shift": 0.0,
        }

    v = vggt_depth[valid].astype(np.float32, copy=False)
    r = ref_depth[valid].astype(np.float32, copy=False)
    v_p10, v_p50, v_p90 = np.percentile(v, [10, 50, 90]).astype(np.float32)
    r_p10, r_p50, r_p90 = np.percentile(r, [10, 50, 90]).astype(np.float32)
    v_span = float(max(v_p90 - v_p10, 1e-6))
    r_span = float(max(r_p90 - r_p10, 1e-6))
    scale = r_span / v_span
    shift = float(r_p50 - scale * v_p50)
    aligned = (vggt_depth.astype(np.float32) * float(scale) + float(shift)).astype(np.float32)
    return aligned, {
        "mode": "robust_p10_p50_p90",
        "pixels": int(valid.sum()),
        "scale": float(scale),
        "shift": float(shift),
        "vggt_p10": float(v_p10),
        "vggt_p50": float(v_p50),
        "vggt_p90": float(v_p90),
        "ref_p10": float(r_p10),
        "ref_p50": float(r_p50),
        "ref_p90": float(r_p90),
    }


def _closest_points_on_candidate_faces(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    candidate_faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_faces = candidate_faces.reshape(-1)
    triangles = vertices[faces[flat_faces]].astype(np.float32, copy=False)
    repeated_points = np.repeat(points.astype(np.float32, copy=False), candidate_faces.shape[1], axis=0)
    bary = points_to_barycentric(repeated_points, triangles)
    closest = np.sum(triangles * bary[:, :, None], axis=1).astype(np.float32, copy=False)
    distance2 = np.sum((closest - repeated_points) ** 2, axis=1).reshape(points.shape[0], candidate_faces.shape[1])
    best_local = np.argmin(distance2, axis=1)
    row = np.arange(points.shape[0], dtype=np.int64)
    best_flat = row * candidate_faces.shape[1] + best_local
    best_faces = flat_faces[best_flat].astype(np.int64, copy=False)
    return closest[best_flat], best_faces, bary[best_flat]


def bind_gaussians_to_mesh(
    xyz: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    face_k: int,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    face_centers = triangles.mean(axis=1).astype(np.float32, copy=False)
    face_normals = normalize_np(np.asarray(mesh.face_normals, dtype=np.float32))
    tree = cKDTree(face_centers)

    total = xyz.shape[0]
    k = max(1, min(int(face_k), int(faces.shape[0])))
    surface_points = np.zeros_like(xyz, dtype=np.float32)
    face_ids = np.zeros((total,), dtype=np.int64)
    bary = np.zeros((total, 3), dtype=np.float32)

    for start in range(0, total, int(chunk_size)):
        end = min(start + int(chunk_size), total)
        _, cand = tree.query(xyz[start:end], k=k, workers=1)
        cand = np.asarray(cand, dtype=np.int64)
        if cand.ndim == 1:
            cand = cand[:, None]
        closest, best_faces, best_bary = _closest_points_on_candidate_faces(
            xyz[start:end],
            vertices,
            faces,
            cand,
        )
        surface_points[start:end] = closest
        face_ids[start:end] = best_faces
        bary[start:end] = best_bary

    normals = face_normals[face_ids].astype(np.float32, copy=False)
    edge_u = vertices[faces[face_ids, 1]] - vertices[faces[face_ids, 0]]
    edge_u = normalize_np(edge_u.astype(np.float32, copy=False))
    tangent_v = normalize_np(np.cross(normals, edge_u))
    tangent_u = normalize_np(np.cross(tangent_v, normals))
    offset = (xyz - surface_points).astype(np.float32, copy=False)
    normal_offset = np.sum(offset * normals, axis=1).astype(np.float32)
    tangent_offset_u = np.sum(offset * tangent_u, axis=1).astype(np.float32)
    tangent_offset_v = np.sum(offset * tangent_v, axis=1).astype(np.float32)
    distance = np.linalg.norm(offset, axis=1).astype(np.float32)
    return {
        "surface_points": surface_points,
        "face_ids": face_ids,
        "bary_coords": bary,
        "normals": normals,
        "tangent_u": tangent_u.astype(np.float32, copy=False),
        "tangent_v": tangent_v.astype(np.float32, copy=False),
        "normal_offset": normal_offset,
        "tangent_offset_u": tangent_offset_u,
        "tangent_offset_v": tangent_offset_v,
        "surface_distance": distance,
    }


def _save_heatmap(path: Path, value_hw: np.ndarray, *, symmetric: bool = False) -> None:
    value = value_hw.astype(np.float32, copy=False)
    finite = np.isfinite(value)
    if not np.any(finite):
        Image.fromarray(np.zeros((*value.shape, 3), dtype=np.uint8)).save(path)
        return
    if symmetric:
        limit = float(np.percentile(np.abs(value[finite]), 98))
        limit = max(limit, 1e-6)
        x = np.clip(value / limit, -1.0, 1.0)
        rgb = np.zeros((*value.shape, 3), dtype=np.float32)
        rgb[..., 0] = np.clip(x, 0.0, 1.0)
        rgb[..., 2] = np.clip(-x, 0.0, 1.0)
        rgb[..., 1] = 1.0 - np.abs(x)
    else:
        lo, hi = np.percentile(value[finite], [2, 98]).astype(np.float32)
        span = float(max(hi - lo, 1e-6))
        x = np.clip((value - float(lo)) / span, 0.0, 1.0)
        rgb = np.stack([x, x, x], axis=-1)
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)


def _save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None, max_points: int = 200000) -> None:
    if points.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        ids = np.sort(rng.choice(points.shape[0], size=int(max_points), replace=False))
        points = points[ids]
        colors = colors[ids] if colors is not None else None
    cloud = trimesh.points.PointCloud(points, colors=colors)
    cloud.export(path)


def _correction_colors(values: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    limit = float(np.percentile(np.abs(values[finite]), 98)) if np.any(finite) else 1.0
    limit = max(limit, 1e-6)
    x = np.clip(values / limit, -1.0, 1.0)
    colors = np.zeros((values.shape[0], 4), dtype=np.uint8)
    colors[:, 0] = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    colors[:, 2] = (np.clip(-x, 0.0, 1.0) * 255).astype(np.uint8)
    colors[:, 1] = (np.clip(1.0 - np.abs(x), 0.0, 1.0) * 255).astype(np.uint8)
    colors[:, 3] = (np.clip(confidence, 0.05, 1.0) * 255).astype(np.uint8)
    return colors


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser(description="Build a conservative VGGT geometry correction for SOFLR Gaussians bound to an LR SOF mesh.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--soflr_model_path", required=True)
    parser.add_argument("--lr_mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_root", default="/root/autodl-tmp/vggt")
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vggt_cache", default=None)
    parser.add_argument("--face_k", type=int, default=8)
    parser.add_argument("--binding_chunk_size", type=int, default=50000)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--min_vggt_confidence", type=float, default=0.05)
    parser.add_argument("--depth_align_min_pixels", type=int, default=2048)
    parser.add_argument("--normal_denominator_min", type=float, default=0.15)
    parser.add_argument("--max_correction_ratio", type=float, default=0.006)
    parser.add_argument("--max_correction_abs", type=float, default=0.0)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.soflr_model_path).expanduser().resolve()
    mesh_path = Path(args.lr_mesh_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_iteration(model_path, int(args.load_iteration))
    dataset_args = _build_dataset_args(str(scene_root), str(model_path), str(args.images_subdir), data_device="cpu")
    dataset = ModelParams(None).extract(dataset_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    cameras = scene.getTrainCameras().copy()
    selected_cameras = _select_uniform(cameras, int(args.max_views))
    if not selected_cameras:
        raise RuntimeError("No cameras available for correction probe.")

    target_hw = (int(selected_cameras[0].image_height), int(selected_cameras[0].image_width))
    image_index = index_image_dir(str(scene_root / str(args.images_subdir)))
    lr_images = []
    image_names = []
    for camera in selected_cameras:
        image_path = _lookup_image(image_index, camera.image_name)
        lr_images.append(_load_image_chw(image_path, target_hw))
        image_names.append(normalize_image_name(camera.image_name))
    lr_images_t = torch.stack(lr_images, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)

    print(f"[bound-gs-vggt] selected views : {len(selected_cameras)}")
    print(f"[bound-gs-vggt] target hw      : {target_hw}")
    print(f"[bound-gs-vggt] render SOFLR depth/alpha")
    gaussians.compute_3D_filter(selected_cameras, CUDA=False)
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    render_depths = []
    render_alphas = []
    for idx, camera in enumerate(selected_cameras):
        render_pkg = render_simple(camera, gaussians, background)
        render_depths.append(render_pkg["depth"][0].detach().cpu().numpy().astype(np.float32))
        render_alphas.append(render_pkg["alpha"][0].detach().cpu().numpy().astype(np.float32))
        if idx < 4:
            _save_heatmap(diag_dir / f"{idx:05d}_soflr_depth.png", render_depths[-1])
            _save_heatmap(diag_dir / f"{idx:05d}_soflr_alpha.png", render_alphas[-1])

    print("[bound-gs-vggt] run vanilla VGGT")
    vggt_adapter = FrozenVGGTAdapter(
        VGGTAdapterConfig(
            vggt_root=str(Path(args.vggt_root).expanduser().resolve()),
            device=device,
        )
    )
    vggt_cache = Path(args.vggt_cache).expanduser().resolve() if args.vggt_cache else output_dir / "vggt_prior.pt"
    vggt_prior = vggt_adapter.run(
        lr_images_t,
        target_hw=target_hw,
        image_names=image_names,
        cache_path=vggt_cache,
    )
    vggt_depths = vggt_prior["depth_hr"][0, :, 0].detach().cpu().numpy().astype(np.float32)
    vggt_confs = vggt_prior["conf_hr"][0, :, 0].detach().cpu().numpy().astype(np.float32)

    print("[bound-gs-vggt] bind SOFLR GS to LR mesh")
    mesh = load_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bbox_diag = float(np.linalg.norm(np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float32)))
    bbox_diag = max(bbox_diag, 1e-6)
    max_correction = float(args.max_correction_abs) if float(args.max_correction_abs) > 0 else bbox_diag * float(args.max_correction_ratio)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    binding = bind_gaussians_to_mesh(
        xyz,
        mesh,
        face_k=int(args.face_k),
        chunk_size=int(args.binding_chunk_size),
    )

    correction_sum = np.zeros((xyz.shape[0],), dtype=np.float64)
    weight_sum = np.zeros((xyz.shape[0],), dtype=np.float64)
    view_count = np.zeros((xyz.shape[0],), dtype=np.int32)
    align_summaries = []
    correction_sample_stats = []

    for view_idx, camera in enumerate(selected_cameras):
        alpha = render_alphas[view_idx]
        soflr_depth = render_depths[view_idx]
        conf = vggt_confs[view_idx]
        vggt_depth = vggt_depths[view_idx]
        align_mask = (alpha >= float(args.min_alpha)) & (conf >= float(args.min_vggt_confidence))
        vggt_aligned, align_summary = _robust_align_depth(
            vggt_depth,
            soflr_depth,
            align_mask,
            min_pixels=int(args.depth_align_min_pixels),
        )
        align_summary["image_name"] = image_names[view_idx]
        align_summaries.append(align_summary)
        delta_map = np.clip(vggt_aligned - soflr_depth, -max_correction * 4.0, max_correction * 4.0)
        if view_idx < 4:
            _save_heatmap(diag_dir / f"{view_idx:05d}_vggt_depth_aligned.png", vggt_aligned)
            _save_heatmap(diag_dir / f"{view_idx:05d}_vggt_conf.png", conf)
            _save_heatmap(diag_dir / f"{view_idx:05d}_depth_delta.png", delta_map, symmetric=True)

        projected, valid = _project_points_camera(camera, xyz, depth_min=float(args.depth_min))
        if not np.any(valid):
            continue
        ids = np.flatnonzero(valid).astype(np.int64, copy=False)
        xy = projected[ids, :2]
        sampled_alpha = _sample_nearest_2d(alpha, xy)
        sampled_conf = _sample_nearest_2d(conf, xy)
        sampled_soflr_depth = _sample_nearest_2d(soflr_depth, xy)
        sampled_vggt_depth = _sample_nearest_2d(vggt_aligned, xy)
        depth_residual = sampled_vggt_depth - sampled_soflr_depth

        r = np.asarray(camera.R, dtype=np.float32)
        z_axis_world_coeff = r[:, 2].astype(np.float32, copy=False)
        denom = np.sum(binding["normals"][ids] * z_axis_world_coeff[None, :], axis=1).astype(np.float32)
        usable = sampled_alpha >= float(args.min_alpha)
        usable &= sampled_conf >= float(args.min_vggt_confidence)
        usable &= np.abs(denom) >= float(args.normal_denominator_min)
        usable &= np.isfinite(depth_residual)
        if not np.any(usable):
            continue

        local_ids = ids[usable]
        normal_step = np.clip(depth_residual[usable] / denom[usable], -max_correction, max_correction).astype(np.float32)
        weights = (sampled_alpha[usable] * sampled_conf[usable] * np.clip(np.abs(denom[usable]), 0.0, 1.0)).astype(np.float32)
        correction_sum[local_ids] += normal_step.astype(np.float64) * weights.astype(np.float64)
        weight_sum[local_ids] += weights.astype(np.float64)
        view_count[local_ids] += 1
        correction_sample_stats.append(stats_from_array(normal_step))

    correction = np.zeros((xyz.shape[0],), dtype=np.float32)
    valid_corr = weight_sum > 1e-8
    correction[valid_corr] = (correction_sum[valid_corr] / weight_sum[valid_corr]).astype(np.float32)
    correction = np.clip(correction, -max_correction, max_correction)
    confidence = np.clip(weight_sum / max(len(selected_cameras), 1), 0.0, 1.0).astype(np.float32)
    corrected_surface_points = (binding["surface_points"] + correction[:, None] * binding["normals"]).astype(np.float32)

    payload_path = output_dir / "correction_payload_v0.npz"
    np.savez_compressed(
        payload_path,
        version=np.asarray(["soflr_vggt_bound_gs_correction_v0"]),
        original_xyz=xyz,
        surface_points=binding["surface_points"],
        corrected_surface_points=corrected_surface_points,
        face_ids=binding["face_ids"],
        bary_coords=binding["bary_coords"],
        normals=binding["normals"],
        tangent_u=binding["tangent_u"],
        tangent_v=binding["tangent_v"],
        normal_offset=binding["normal_offset"],
        tangent_offset_u=binding["tangent_offset_u"],
        tangent_offset_v=binding["tangent_offset_v"],
        surface_distance=binding["surface_distance"],
        normal_correction=correction,
        correction_confidence=confidence,
        correction_weight_sum=weight_sum.astype(np.float32),
        correction_view_count=view_count,
        max_correction=np.asarray([max_correction], dtype=np.float32),
        bbox_diag=np.asarray([bbox_diag], dtype=np.float32),
    )

    colors = _correction_colors(correction, confidence)
    _save_point_cloud(output_dir / "bound_surface_points_preview_v0.ply", binding["surface_points"], max_points=int(args.preview_max_points))
    _save_point_cloud(output_dir / "corrected_surface_points_preview_v0.ply", corrected_surface_points, colors, max_points=int(args.preview_max_points))
    _save_point_cloud(output_dir / "original_soflr_gs_points_preview_v0.ply", xyz, max_points=int(args.preview_max_points))

    summary = {
        "version": "soflr_vggt_bound_gs_correction_v0",
        "scene_root": str(scene_root),
        "soflr_model_path": str(model_path),
        "lr_mesh_path": str(mesh_path),
        "iteration": int(iteration),
        "images_subdir": str(args.images_subdir),
        "selected_views": image_names,
        "target_hw": [int(target_hw[0]), int(target_hw[1])],
        "num_gaussians": int(xyz.shape[0]),
        "num_mesh_vertices": int(vertices.shape[0]),
        "num_mesh_faces": int(len(mesh.faces)),
        "bbox_diag": float(bbox_diag),
        "max_correction": float(max_correction),
        "binding": {
            "face_k": int(args.face_k),
            "surface_distance": stats_from_array(binding["surface_distance"]),
            "normal_offset": stats_from_array(binding["normal_offset"]),
        },
        "correction": {
            "normal_correction": stats_from_array(correction),
            "abs_normal_correction": stats_from_array(np.abs(correction)),
            "confidence": stats_from_array(confidence),
            "view_count": stats_from_array(view_count.astype(np.float32)),
            "active_gaussians": int(np.sum(valid_corr)),
        },
        "depth_alignment": align_summaries,
        "per_view_correction_sample_stats": correction_sample_stats,
        "payload": str(payload_path),
        "vggt_cache": str(vggt_cache),
        "diagnostics_dir": str(diag_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
