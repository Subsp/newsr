from pathlib import Path
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm
from diff_gaussian_rasterization import (
    DebugVisualization,
    ExtendedSettings,
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


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


def save_mask_preview(mask: torch.Tensor, path: str):
    array = (mask.detach().to(dtype=torch.uint8).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def erode_binary_mask(mask_hw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = mask_hw.to(device="cuda", dtype=torch.float32)[None, None]
    if kernel_size <= 1:
        return mask[0, 0] > 0
    pad = kernel_size // 2
    inv = 1.0 - mask
    inv = F.pad(inv, (pad, pad, pad, pad), mode="replicate")
    inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1)
    return (1.0 - inv[0, 0]) > 0.5


def dilate_binary_mask(mask_hw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = mask_hw.to(device="cuda", dtype=torch.float32)[None, None]
    if kernel_size <= 1:
        return mask[0, 0] > 0
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1)[0, 0] > 0.5


def camera_center_numpy(cam) -> np.ndarray:
    center = cam.camera_center
    if torch.is_tensor(center):
        center = center.detach().cpu().numpy()
    return np.asarray(center, dtype=np.float32).reshape(3)


def rasterize_projected_triangles_zbuffer(
    tri_xy: np.ndarray,
    tri_depth: np.ndarray,
    face_ids: np.ndarray,
    height: int,
    width: int,
    depth_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    face_buffer = np.full((height, width), -1, dtype=np.int64)

    for tri, z, face_id in zip(tri_xy, tri_depth, face_ids):
        min_x = max(int(np.floor(np.min(tri[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(tri[:, 0]))), width - 1)
        min_y = max(int(np.floor(np.min(tri[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(tri[:, 1]))), height - 1)
        if max_x < min_x or max_y < min_y:
            continue

        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-8:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx.astype(np.float32) + 0.5
        py = yy.astype(np.float32) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue

        interp_z = w0 * z[0] + w1 * z[1] + w2 * z[2]
        update = inside & (interp_z > float(depth_min))
        if not np.any(update):
            continue

        sub_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update &= interp_z < sub_depth
        if not np.any(update):
            continue
        sub_depth[update] = interp_z[update]
        sub_face = face_buffer[min_y : max_y + 1, min_x : max_x + 1]
        sub_face[update] = int(face_id)

    return depth_buffer, face_buffer


def project_points_camera(cam, points_xyz: np.ndarray, depth_min: float, margin: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(cam.R, dtype=np.float32)
    T = np.asarray(cam.T, dtype=np.float32)
    xyz_cam = points_xyz @ R + T[None, :]
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / np.clip(z, 1e-6, None) * float(cam.focal_x) + float(cam.image_width) / 2.0
    y = xyz_cam[:, 1] / np.clip(z, 1e-6, None) * float(cam.focal_y) + float(cam.image_height) / 2.0
    valid = z > float(depth_min)
    valid &= x >= float(margin)
    valid &= x < float(cam.image_width - margin)
    valid &= y >= float(margin)
    valid &= y < float(cam.image_height - margin)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False), valid


def ray_dirs_to_projected_pixels(cam, projected_xy: np.ndarray) -> np.ndarray:
    intrins = np.asarray(
        [
            [float(cam.focal_x), 0.0, float(cam.image_width) / 2.0],
            [0.0, float(cam.focal_y), float(cam.image_height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    c2w = torch.inverse(cam.world_view_transform.T).detach().cpu().numpy().astype(np.float32, copy=False)
    points = np.empty((projected_xy.shape[0], 3), dtype=np.float32)
    points[:, :2] = projected_xy
    points[:, 2] = 1.0
    dirs_world = points @ np.linalg.inv(intrins).T @ c2w[:3, :3].T
    dirs_world = dirs_world / np.clip(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-6, None)
    return dirs_world.astype(np.float32, copy=False)


def first_ray_hits(intersector, ray_origins: np.ndarray, ray_directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
    )
    if len(index_ray) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    locations = np.asarray(locations, dtype=np.float32)
    index_ray = np.asarray(index_ray, dtype=np.int64)
    index_tri = np.asarray(index_tri, dtype=np.int64)
    ray_hit_distance = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    best_hit_per_ray = {}
    for hit_idx, ray_idx in enumerate(index_ray.tolist()):
        dist = float(ray_hit_distance[hit_idx])
        existing = best_hit_per_ray.get(ray_idx)
        if existing is None or dist < existing[0]:
            best_hit_per_ray[ray_idx] = (dist, locations[hit_idx], int(index_tri[hit_idx]))

    ray_ids = np.asarray(sorted(best_hit_per_ray.keys()), dtype=np.int64)
    hit_points = np.asarray([best_hit_per_ray[int(idx)][1] for idx in ray_ids.tolist()], dtype=np.float32)
    hit_tris = np.asarray([best_hit_per_ray[int(idx)][2] for idx in ray_ids.tolist()], dtype=np.int64)
    return ray_ids, hit_points, hit_tris


def filter_carriers_by_projected_zbuffer(
    candidate_ids: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    depths: np.ndarray,
    width: int,
    depth_tolerance: float,
) -> np.ndarray:
    """Approximate carrier visibility without ray-casting the full mesh.

    Carriers are already mesh-bound surface samples. For the fast fusion path we
    keep carriers that are close to the nearest carrier depth at their projected
    pixel. This avoids the expensive full 12M-face trimesh ray query while still
    rejecting obvious back-facing/behind-surface duplicates inside the ROI.
    """
    if candidate_ids.size == 0:
        return candidate_ids
    pixel_ids = pix_y[candidate_ids].astype(np.int64, copy=False) * int(width) + pix_x[candidate_ids].astype(np.int64, copy=False)
    depth_values = depths[candidate_ids].astype(np.float32, copy=False)

    order = np.lexsort((depth_values, pixel_ids))
    sorted_pixels = pixel_ids[order]
    sorted_depths = depth_values[order]
    unique_pixels, first_indices = np.unique(sorted_pixels, return_index=True)
    min_depths = sorted_depths[first_indices]

    pixel_positions = np.searchsorted(unique_pixels, pixel_ids)
    nearest_depth = min_depths[pixel_positions]
    keep = depth_values <= nearest_depth + float(depth_tolerance)
    return candidate_ids[keep]


def _normalize_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _quaternion_from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert row-major rotation matrices to GraphDeco quaternion order wxyz."""
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
                    1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                    1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                    1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
                ],
                dim=-1,
            ),
            min=1e-8,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[:, 0] ** 2, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], dim=-1),
            torch.stack([m[:, 2, 1] - m[:, 1, 2], q_abs[:, 1] ** 2, m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0]], dim=-1),
            torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] + m[:, 0, 1], q_abs[:, 2] ** 2, m[:, 2, 1] + m[:, 1, 2]], dim=-1),
            torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 2, 1] + m[:, 1, 2], q_abs[:, 3] ** 2], dim=-1),
        ],
        dim=1,
    )
    quat_candidates = quat_by_rijk / (2.0 * torch.clamp(q_abs[:, :, None], min=1e-8))
    out = quat_candidates[torch.arange(matrix.shape[0], device=matrix.device), torch.argmax(q_abs, dim=-1)]
    return _normalize_torch(out)


def carrier_rasterizer_tensors(
    carriers: Dict[str, np.ndarray],
    scale_modifier: float,
    opacity: float,
) -> Dict[str, torch.Tensor]:
    device = "cuda"
    centers = torch.as_tensor(carriers["centers"], device=device, dtype=torch.float32)
    tangent_u = _normalize_torch(torch.as_tensor(carriers["tangent_u"], device=device, dtype=torch.float32))
    tangent_v = _normalize_torch(torch.as_tensor(carriers["tangent_v"], device=device, dtype=torch.float32))
    normals = _normalize_torch(torch.as_tensor(carriers["normals"], device=device, dtype=torch.float32))
    rotations = _quaternion_from_rotation_matrix(torch.stack([tangent_u, tangent_v, normals], dim=2))
    scales = torch.stack(
        [
            torch.as_tensor(carriers["scale_u"], device=device, dtype=torch.float32),
            torch.as_tensor(carriers["scale_v"], device=device, dtype=torch.float32),
            torch.as_tensor(carriers["scale_n"], device=device, dtype=torch.float32),
        ],
        dim=1,
    ) * float(scale_modifier)
    opacities = torch.full((centers.shape[0], 1), float(opacity), device=device, dtype=torch.float32)
    colors = torch.zeros((centers.shape[0], 3), device=device, dtype=torch.float32)
    return {"centers": centers, "rotations": rotations, "scales": scales, "opacities": opacities, "colors": colors}


@torch.no_grad()
def render_carrier_depth_alpha(
    view,
    raster_tensors: Dict[str, torch.Tensor],
    splat_args: Optional[ExtendedSettings],
) -> Tuple[np.ndarray, np.ndarray]:
    if splat_args is None:
        splat_args = ExtendedSettings()
    splat_args.render_opacity = True
    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    raster_settings = GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform,
        inv_viewprojmatrix=view.full_proj_transform_inverse,
        sh_degree=0,
        campos=view.camera_center,
        prefiltered=False,
        settings=splat_args,
        debug_data=DebugVisualization(printing_enabled=False),
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    screenspace = torch.zeros_like(raster_tensors["centers"], device="cuda", dtype=torch.float32)
    rendered, _ = rasterizer(
        means3D=raster_tensors["centers"],
        means2D=screenspace,
        colors_precomp=raster_tensors["colors"],
        opacities=raster_tensors["opacities"],
        scales=raster_tensors["scales"],
        rotations=raster_tensors["rotations"],
    )
    if rendered.shape[0] < 8:
        raise RuntimeError(f"Gaussian rasterizer returned {rendered.shape[0]} channels; expected depth/alpha channels.")
    depth = rendered[6].detach().cpu().numpy().astype(np.float32, copy=False)
    alpha = rendered[7].detach().cpu().numpy().astype(np.float32, copy=False)
    return depth, alpha


def filter_carriers_by_rasterized_depth(
    candidate_ids: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    depths: np.ndarray,
    raster_depth: np.ndarray,
    raster_alpha: np.ndarray,
    depth_tolerance: float,
    alpha_threshold: float,
) -> np.ndarray:
    if candidate_ids.size == 0:
        return candidate_ids
    sampled_depth = raster_depth[pix_y[candidate_ids], pix_x[candidate_ids]]
    sampled_alpha = raster_alpha[pix_y[candidate_ids], pix_x[candidate_ids]]
    keep = sampled_alpha >= float(alpha_threshold)
    keep &= np.abs(depths[candidate_ids] - sampled_depth) <= float(depth_tolerance)
    return candidate_ids[keep]


def build_mesh_visibility(
    mesh: trimesh.Trimesh,
    cam,
    face_ids: Optional[np.ndarray] = None,
    depth_min: float = 0.01,
    front_facing_only: bool = True,
) -> Dict[str, torch.Tensor | np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    if face_ids is None:
        active_face_ids = np.arange(faces_all.shape[0], dtype=np.int64)
    else:
        active_face_ids = np.asarray(face_ids, dtype=np.int64)
    H = int(cam.image_height)
    W = int(cam.image_width)
    if active_face_ids.size == 0:
        empty = torch.zeros((int(cam.image_height), int(cam.image_width)), dtype=torch.bool, device="cuda")
        depth = torch.full((H, W), float("inf"), dtype=torch.float32, device="cuda")
        return {"visible_mask": empty, "visible_face_ids": active_face_ids, "depth": depth}

    active_faces = faces_all[active_face_ids]
    triangles = vertices[active_faces]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)[active_face_ids]
    cam_center = camera_center_numpy(cam)
    face_centers = triangles.mean(axis=1)
    if front_facing_only:
        front_mask = (face_normals * (cam_center[None, :] - face_centers)).sum(axis=1) > 0.0
    else:
        front_mask = np.ones((active_faces.shape[0],), dtype=bool)

    R = np.asarray(cam.R, dtype=np.float32)
    T = np.asarray(cam.T, dtype=np.float32)
    tri_cam = triangles @ R[None, :, :] + T[None, None, :]
    tri_depth = tri_cam[..., 2]
    depth_mask = np.all(tri_depth > float(depth_min), axis=1)
    keep_mask = front_mask & depth_mask

    if not np.any(keep_mask):
        empty = torch.zeros((H, W), dtype=torch.bool, device="cuda")
        depth = torch.full((H, W), float("inf"), dtype=torch.float32, device="cuda")
        return {"visible_mask": empty, "visible_face_ids": active_face_ids[keep_mask], "depth": depth}

    x = tri_cam[..., 0] / np.clip(tri_depth, 1e-6, None) * float(cam.focal_x) + float(W) / 2.0
    y = tri_cam[..., 1] / np.clip(tri_depth, 1e-6, None) * float(cam.focal_y) + float(H) / 2.0
    tri_xy = np.stack([x, y], axis=-1)

    kept_xy = tri_xy[keep_mask]
    kept_depth = tri_depth[keep_mask]
    kept_face_ids = active_face_ids[keep_mask]
    bbox_min = np.floor(np.min(kept_xy, axis=1))
    bbox_max = np.ceil(np.max(kept_xy, axis=1))
    overlaps = (
        (bbox_max[:, 0] >= 0)
        & (bbox_min[:, 0] < W)
        & (bbox_max[:, 1] >= 0)
        & (bbox_min[:, 1] < H)
    )
    if not np.any(overlaps):
        empty = torch.zeros((H, W), dtype=torch.bool, device="cuda")
        depth = torch.full((H, W), float("inf"), dtype=torch.float32, device="cuda")
        return {"visible_mask": empty, "visible_face_ids": kept_face_ids[:0], "depth": depth}

    depth_buffer, face_buffer = rasterize_projected_triangles_zbuffer(
        kept_xy[overlaps],
        kept_depth[overlaps],
        kept_face_ids[overlaps],
        height=H,
        width=W,
        depth_min=depth_min,
    )
    visible_np = np.isfinite(depth_buffer)
    visible_face_ids = np.unique(face_buffer[face_buffer >= 0]).astype(np.int64, copy=False)
    visible_mask = torch.from_numpy(visible_np).to(device="cuda")
    depth = torch.from_numpy(depth_buffer).to(device="cuda", dtype=torch.float32)
    return {"visible_mask": visible_mask, "visible_face_ids": visible_face_ids, "depth": depth}


def _sample_mesh_face_points(triangles: np.ndarray, samples_per_face: int) -> np.ndarray:
    if samples_per_face == 1:
        bary = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32)
    elif samples_per_face == 4:
        bary = np.asarray(
            [
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError("samples_per_face must be 1 or 4")
    return (triangles[:, None] * bary[None, :, :, None]).sum(axis=2)


def _scatter_min_depth(pixel_ids: np.ndarray, depths: np.ndarray, height: int, width: int) -> torch.Tensor:
    depth = torch.full((height * width,), float("inf"), dtype=torch.float32, device="cuda")
    if pixel_ids.size == 0:
        return depth.reshape(height, width)

    pixel_t = torch.from_numpy(pixel_ids.astype(np.int64, copy=False)).to(device="cuda")
    depth_t = torch.from_numpy(depths.astype(np.float32, copy=False)).to(device="cuda")
    if hasattr(depth, "scatter_reduce_"):
        depth.scatter_reduce_(0, pixel_t, depth_t, reduce="amin", include_self=True)
        return depth.reshape(height, width)

    order = np.lexsort((depths, pixel_ids))
    sorted_pixels = pixel_ids[order]
    sorted_depths = depths[order]
    first = np.ones((sorted_pixels.shape[0],), dtype=bool)
    first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
    depth_np = np.full((height * width,), np.inf, dtype=np.float32)
    depth_np[sorted_pixels[first]] = sorted_depths[first]
    return torch.from_numpy(depth_np).to(device="cuda", dtype=torch.float32).reshape(height, width)


def _dilate_depth_min(depth: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return depth
    pad = kernel_size // 2
    large = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1e6))
    dilated = -F.max_pool2d(
        F.pad((-large)[None, None], (pad, pad, pad, pad), mode="replicate"),
        kernel_size=kernel_size,
        stride=1,
    )[0, 0]
    return torch.where(dilated < 1e5, dilated, torch.full_like(depth, float("inf")))


def build_mesh_sample_visibility(
    mesh: trimesh.Trimesh,
    cam,
    depth_min: float = 0.01,
    front_facing_only: bool = True,
    samples_per_face: int = 4,
    splat_kernel: int = 5,
) -> Dict[str, torch.Tensor | np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    active_face_ids = np.arange(faces_all.shape[0], dtype=np.int64)
    height = int(cam.image_height)
    width = int(cam.image_width)
    if faces_all.shape[0] == 0:
        empty = torch.zeros((height, width), dtype=torch.bool, device="cuda")
        depth = torch.full((height, width), float("inf"), dtype=torch.float32, device="cuda")
        return {"visible_mask": empty, "visible_face_ids": active_face_ids, "depth": depth}

    triangles = vertices[faces_all]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    face_centers = triangles.mean(axis=1)
    if front_facing_only:
        cam_center = camera_center_numpy(cam)
        front_mask = (face_normals * (cam_center[None, :] - face_centers)).sum(axis=1) > 0.0
    else:
        front_mask = np.ones((faces_all.shape[0],), dtype=bool)

    if not np.any(front_mask):
        empty = torch.zeros((height, width), dtype=torch.bool, device="cuda")
        depth = torch.full((height, width), float("inf"), dtype=torch.float32, device="cuda")
        return {"visible_mask": empty, "visible_face_ids": active_face_ids[:0], "depth": depth}

    active_face_ids = active_face_ids[front_mask]
    samples_world = _sample_mesh_face_points(triangles[front_mask], int(samples_per_face))
    samples_face_ids = np.repeat(active_face_ids, int(samples_per_face))
    samples = samples_world.reshape(-1, 3)

    R = np.asarray(cam.R, dtype=np.float32)
    T = np.asarray(cam.T, dtype=np.float32)
    samples_cam = samples @ R + T[None, :]
    z = samples_cam[:, 2]
    valid = z > float(depth_min)
    x = samples_cam[:, 0] / np.clip(z, 1e-6, None) * float(cam.focal_x) + float(width) / 2.0
    y = samples_cam[:, 1] / np.clip(z, 1e-6, None) * float(cam.focal_y) + float(height) / 2.0
    pix_x = np.round(x).astype(np.int64)
    pix_y = np.round(y).astype(np.int64)
    valid &= pix_x >= 0
    valid &= pix_x < width
    valid &= pix_y >= 0
    valid &= pix_y < height

    if not np.any(valid):
        empty = torch.zeros((height, width), dtype=torch.bool, device="cuda")
        depth = torch.full((height, width), float("inf"), dtype=torch.float32, device="cuda")
        return {"visible_mask": empty, "visible_face_ids": active_face_ids[:0], "depth": depth}

    valid_pix_x = pix_x[valid]
    valid_pix_y = pix_y[valid]
    pix_linear = (valid_pix_y * width + valid_pix_x).astype(np.int64, copy=False)
    sampled_depth = z[valid].astype(np.float32, copy=False)
    depth_sparse = _scatter_min_depth(pix_linear, sampled_depth, height, width)
    depth_at_samples = (
        depth_sparse[
            torch.from_numpy(valid_pix_y.astype(np.int64, copy=False)).to(device="cuda"),
            torch.from_numpy(valid_pix_x.astype(np.int64, copy=False)).to(device="cuda"),
        ]
        .detach()
        .cpu()
        .numpy()
    )
    depth_epsilon = np.maximum(1e-4, 1e-4 * np.maximum(depth_at_samples, 0.0))
    winning_samples = np.abs(sampled_depth - depth_at_samples) <= depth_epsilon
    visible_face_ids = np.unique(samples_face_ids[valid][winning_samples]).astype(np.int64, copy=False)

    depth = _dilate_depth_min(depth_sparse, int(splat_kernel))
    visible_mask = torch.isfinite(depth)
    return {"visible_mask": visible_mask, "visible_face_ids": visible_face_ids, "depth": depth}


def build_mesh_visible_mask(
    mesh: trimesh.Trimesh,
    cam,
    face_ids: Optional[np.ndarray] = None,
    depth_min: float = 0.01,
    front_facing_only: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    visibility = build_mesh_visibility(
        mesh,
        cam,
        face_ids=face_ids,
        depth_min=depth_min,
        front_facing_only=front_facing_only,
    )
    return visibility["visible_mask"], visibility["visible_face_ids"]


def build_mesh_depth_edge_mask(
    depth: torch.Tensor,
    visible_mask: torch.Tensor,
    abs_threshold: float = 0.03,
    rel_threshold: float = 0.01,
    kernel_size: int = 7,
    dilate_kernel: int = 9,
) -> torch.Tensor:
    if kernel_size <= 1:
        return torch.zeros_like(visible_mask)
    visible = visible_mask.to(device="cuda", dtype=torch.bool)
    if not torch.any(visible):
        return torch.zeros_like(visible)

    depth = depth.to(device="cuda", dtype=torch.float32)
    finite_depth = torch.where(visible & torch.isfinite(depth), depth, torch.zeros_like(depth))
    large = torch.full_like(depth, 1e6)
    depth_for_min = torch.where(visible & torch.isfinite(depth), depth, large)
    pad = kernel_size // 2
    dmax = F.max_pool2d(
        F.pad(finite_depth[None, None], (pad, pad, pad, pad), mode="replicate"),
        kernel_size=kernel_size,
        stride=1,
    )[0, 0]
    dmin = -F.max_pool2d(
        F.pad((-depth_for_min)[None, None], (pad, pad, pad, pad), mode="replicate"),
        kernel_size=kernel_size,
        stride=1,
    )[0, 0]
    local_range = dmax - dmin
    threshold = float(abs_threshold) + float(rel_threshold) * torch.clamp_min(dmin, 0.0)
    edge = visible & (dmin < 1e5) & (local_range > threshold)
    if dilate_kernel > 1:
        edge = dilate_binary_mask(edge, dilate_kernel)
    return edge & visible


def build_mesh_partition_masks(
    prior_valid_mask: torch.Tensor,
    mesh_visible_mask: torch.Tensor,
    interior_kernel: int,
    mesh_edge_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    mesh_core_mask = erode_binary_mask(mesh_visible_mask, interior_kernel)
    if mesh_edge_mask is not None:
        mesh_edge_mask = mesh_edge_mask.to(device="cuda", dtype=torch.bool) & mesh_visible_mask
        mesh_core_mask = mesh_core_mask & (~mesh_edge_mask)
    mesh_fusion_mask = prior_valid_mask & mesh_core_mask
    edge_fusion_mask = prior_valid_mask & (~mesh_core_mask)
    mesh_edge_band = mesh_visible_mask & (~mesh_core_mask)
    return {
        "mesh_visible_mask": mesh_visible_mask,
        "mesh_core_mask": mesh_core_mask,
        "mesh_edge_band": mesh_edge_band,
        "mesh_fusion_mask": mesh_fusion_mask,
        "edge_fusion_mask": edge_fusion_mask,
    }


def build_bounded_carriers(
    mesh: trimesh.Trimesh,
    carriers_per_face: int,
    face_ids: Optional[np.ndarray] = None,
    thickness_scale: float = 0.1,
) -> Dict[str, np.ndarray]:
    if carriers_per_face not in BOUND_BARYCENTRIC_TEMPLATES:
        raise ValueError(f"Unsupported carriers_per_face={carriers_per_face}; expected one of {sorted(BOUND_BARYCENTRIC_TEMPLATES)}")
    bary_template = BOUND_BARYCENTRIC_TEMPLATES[carriers_per_face]

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    if face_ids is None:
        active_face_ids = np.arange(faces_all.shape[0], dtype=np.int64)
    else:
        active_face_ids = np.asarray(face_ids, dtype=np.int64)
    active_faces = faces_all[active_face_ids]
    triangles = vertices[active_faces]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)[active_face_ids]

    centers = (triangles[:, None] * bary_template[None, :, :, None]).sum(axis=2).reshape(-1, 3)
    carrier_face_ids = np.repeat(active_face_ids, carriers_per_face)
    bary_coords = np.tile(bary_template[None, :, :], (active_face_ids.shape[0], 1, 1)).reshape(-1, 3)
    normals = np.repeat(face_normals, carriers_per_face, axis=0)

    edge_u = triangles[:, 1] - triangles[:, 0]
    edge_u = edge_u / np.clip(np.linalg.norm(edge_u, axis=1, keepdims=True), 1e-6, None)
    edge_v = np.cross(face_normals, edge_u)
    edge_v = edge_v / np.clip(np.linalg.norm(edge_v, axis=1, keepdims=True), 1e-6, None)
    face_centers = triangles.mean(axis=1, keepdims=True)
    offsets = triangles - face_centers
    scale_u = np.max(np.abs((offsets * edge_u[:, None, :]).sum(axis=-1)), axis=1)
    scale_v = np.max(np.abs((offsets * edge_v[:, None, :]).sum(axis=-1)), axis=1)
    scale_n = np.minimum(scale_u, scale_v) * float(thickness_scale)
    scale_u = np.repeat(scale_u.astype(np.float32), carriers_per_face)
    scale_v = np.repeat(scale_v.astype(np.float32), carriers_per_face)
    scale_n = np.repeat(scale_n.astype(np.float32), carriers_per_face)
    tangent_u = np.repeat(edge_u.astype(np.float32), carriers_per_face, axis=0)
    tangent_v = np.repeat(edge_v.astype(np.float32), carriers_per_face, axis=0)

    return {
        "centers": centers.astype(np.float32, copy=False),
        "normals": normals.astype(np.float32, copy=False),
        "face_ids": carrier_face_ids.astype(np.int64, copy=False),
        "bary_coords": bary_coords.astype(np.float32, copy=False),
        "scale_u": scale_u,
        "scale_v": scale_v,
        "scale_n": scale_n,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
    }


def bilinear_sample_rgb(image_hw3: np.ndarray, projected_xy: np.ndarray) -> np.ndarray:
    h, w, _ = image_hw3.shape
    x = np.clip(projected_xy[:, 0], 0.0, max(w - 1.0, 0.0))
    y = np.clip(projected_xy[:, 1], 0.0, max(h - 1.0, 0.0))
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

    Ia = image_hw3[y0, x0]
    Ib = image_hw3[y0, x1]
    Ic = image_hw3[y1, x0]
    Id = image_hw3[y1, x1]
    return (
        Ia * wa[:, None]
        + Ib * wb[:, None]
        + Ic * wc[:, None]
        + Id * wd[:, None]
    ).astype(np.float32, copy=False)


def resize_rgb_image_np(image_hw3: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if image_hw3.shape[0] == target_height and image_hw3.shape[1] == target_width:
        return image_hw3.astype(np.float32, copy=False)
    image_u8 = np.round(np.clip(image_hw3, 0.0, 1.0) * 255.0).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = Image.fromarray(image_u8, mode="RGB").resize((target_width, target_height), resampling)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def build_mesh_intersector(mesh: trimesh.Trimesh):
    try:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        # trimesh builds the rtree-backed triangle BVH lazily, so force it here
        # to catch missing optional dependencies before the long fusion loop.
        _ = mesh.triangles_tree
        return intersector
    except Exception as exc:
        print(f"[prior-fusion] warning: failed to build ray intersector ({exc}); visibility will use projection-only fallback.")
        return None


def box_blur_rgb_np(image_hw3: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return image_hw3.astype(np.float32, copy=False)
    if kernel_size % 2 == 0:
        kernel_size += 1
    with torch.no_grad():
        image = torch.from_numpy(image_hw3.astype(np.float32, copy=False)).permute(2, 0, 1)[None].to(device="cuda")
        pad = kernel_size // 2
        image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        blurred = F.avg_pool2d(image, kernel_size=kernel_size, stride=1)
        return blurred[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)


def accumulate_carrier_samples(
    num_carriers: int,
    carrier_ids: np.ndarray,
    sampled_rgb: np.ndarray,
    sampled_low_rgb: Optional[np.ndarray],
    sampled_high_rgb: Optional[np.ndarray],
    weights: np.ndarray,
    samples_rgb,
    samples_low_rgb,
    samples_high_rgb,
    samples_weight,
):
    for sample_idx, (carrier_id, rgb, weight) in enumerate(zip(carrier_ids.tolist(), sampled_rgb, weights.tolist())):
        samples_rgb[carrier_id].append(rgb.astype(np.float32, copy=False))
        if sampled_low_rgb is not None:
            samples_low_rgb[carrier_id].append(sampled_low_rgb[sample_idx].astype(np.float32, copy=False))
        if sampled_high_rgb is not None:
            samples_high_rgb[carrier_id].append(sampled_high_rgb[sample_idx].astype(np.float32, copy=False))
        samples_weight[carrier_id].append(float(weight))


def fuse_carrier_samples_mean(
    num_carriers: int,
    samples_rgb,
    samples_weight,
    min_views_per_carrier: int,
    total_views: int,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    fused_rgb = np.zeros((num_carriers, 3), dtype=np.float32)
    disagreement = np.zeros((num_carriers,), dtype=np.float32)
    view_count = np.zeros((num_carriers,), dtype=np.int32)
    weight_sum = np.zeros((num_carriers,), dtype=np.float32)
    valid_mask = np.zeros((num_carriers,), dtype=np.bool_)
    low_frequency_rgb = np.zeros((num_carriers, 3), dtype=np.float32)
    high_frequency_rgb = np.zeros((num_carriers, 3), dtype=np.float32)
    high_frequency_confidence = np.zeros((num_carriers,), dtype=np.float32)
    fusion_case = np.zeros((num_carriers,), dtype=np.int32)

    for idx in tqdm(
        range(num_carriers),
        desc="Reducing carrier samples",
        disable=not bool(show_progress),
        dynamic_ncols=True,
    ):
        if not samples_rgb[idx]:
            continue
        rgb = np.stack(samples_rgb[idx], axis=0).astype(np.float32, copy=False)
        weights = np.asarray(samples_weight[idx], dtype=np.float32)
        weights = np.clip(weights, 1e-6, None)
        view_count[idx] = rgb.shape[0]
        weight_sum[idx] = float(np.sum(weights))
        if rgb.shape[0] < int(min_views_per_carrier):
            continue
        mean_rgb = np.sum(rgb * weights[:, None], axis=0) / np.clip(weight_sum[idx], 1e-6, None)
        fused_rgb[idx] = mean_rgb.astype(np.float32)
        low_frequency_rgb[idx] = fused_rgb[idx]
        high_frequency_rgb[idx] = 0.0
        second_moment = np.sum((rgb ** 2) * weights[:, None], axis=0) / np.clip(weight_sum[idx], 1e-6, None)
        variance = np.clip(second_moment - mean_rgb ** 2, 0.0, None)
        disagreement[idx] = float(np.sqrt(np.mean(variance)))
        high_frequency_confidence[idx] = 1.0
        fusion_case[idx] = 1 if rgb.shape[0] > 1 else 4
        valid_mask[idx] = True

    confidence = np.zeros((num_carriers,), dtype=np.float32)
    if total_views > 0:
        confidence = view_count.astype(np.float32) / float(total_views)

    return {
        "fused_rgb": fused_rgb,
        "confidence": confidence,
        "disagreement": disagreement,
        "view_count": view_count,
        "weight_sum": weight_sum,
        "valid_mask": valid_mask,
        "low_frequency_rgb": low_frequency_rgb,
        "high_frequency_rgb": high_frequency_rgb,
        "high_frequency_confidence": high_frequency_confidence,
        "fusion_case": fusion_case,
    }


def fuse_carrier_samples_freq_split(
    num_carriers: int,
    samples_rgb,
    samples_low_rgb,
    samples_high_rgb,
    samples_weight,
    min_views_per_carrier: int,
    total_views: int,
    low_frequency_consistency_threshold: float,
    high_frequency_consistency_threshold: float,
    single_view_confidence: float,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    fused_rgb = np.zeros((num_carriers, 3), dtype=np.float32)
    low_frequency_rgb = np.zeros((num_carriers, 3), dtype=np.float32)
    high_frequency_rgb = np.zeros((num_carriers, 3), dtype=np.float32)
    confidence = np.zeros((num_carriers,), dtype=np.float32)
    disagreement = np.zeros((num_carriers,), dtype=np.float32)
    high_frequency_confidence = np.zeros((num_carriers,), dtype=np.float32)
    view_count = np.zeros((num_carriers,), dtype=np.int32)
    weight_sum = np.zeros((num_carriers,), dtype=np.float32)
    valid_mask = np.zeros((num_carriers,), dtype=np.bool_)
    fusion_case = np.zeros((num_carriers,), dtype=np.int32)

    for idx in tqdm(
        range(num_carriers),
        desc="Reducing carrier samples",
        disable=not bool(show_progress),
        dynamic_ncols=True,
    ):
        if not samples_rgb[idx]:
            continue
        rgb = np.stack(samples_rgb[idx], axis=0).astype(np.float32, copy=False)
        low_samples = np.stack(samples_low_rgb[idx], axis=0).astype(np.float32, copy=False)
        high_samples = np.stack(samples_high_rgb[idx], axis=0).astype(np.float32, copy=False)
        weights = np.asarray(samples_weight[idx], dtype=np.float32)
        weights = np.clip(weights, 1e-6, None)
        count = rgb.shape[0]
        view_count[idx] = count
        weight_sum[idx] = float(np.sum(weights))

        if count == 1:
            fused_rgb[idx] = rgb[0]
            low_frequency_rgb[idx] = low_samples[0]
            high_frequency_rgb[idx] = high_samples[0]
            confidence[idx] = float(single_view_confidence)
            disagreement[idx] = 0.0
            high_frequency_confidence[idx] = 0.0
            fusion_case[idx] = 4
            valid_mask[idx] = True
            continue

        low_rgb = np.sum(low_samples * weights[:, None], axis=0) / np.clip(weight_sum[idx], 1e-6, None)
        high_rgb = np.sum(high_samples * weights[:, None], axis=0) / np.clip(weight_sum[idx], 1e-6, None)
        low_residual = low_samples - low_rgb[None, :]
        high_residual = high_samples - high_rgb[None, :]
        low_disagreement = float(np.sqrt(np.average(np.mean(low_residual ** 2, axis=1), weights=weights)))
        high_disagreement = float(np.sqrt(np.average(np.mean(high_residual ** 2, axis=1), weights=weights)))
        low_consistency = 1.0 - np.clip(low_disagreement / max(float(low_frequency_consistency_threshold), 1e-6), 0.0, 1.0)
        high_consistency = 1.0 - np.clip(high_disagreement / max(float(high_frequency_consistency_threshold), 1e-6), 0.0, 1.0)

        low_frequency_rgb[idx] = low_rgb.astype(np.float32)
        disagreement[idx] = low_disagreement
        high_frequency_confidence[idx] = float(high_consistency)

        if low_consistency <= 0.0:
            fused_rgb[idx] = low_rgb.astype(np.float32)
            confidence[idx] = 0.0
            fusion_case[idx] = 3
            valid_mask[idx] = False
            continue

        if high_consistency > 0.0:
            high_residual_norm = np.linalg.norm(high_residual, axis=1)
            robust_weights = weights / np.clip(1.0 + high_residual_norm / max(float(high_frequency_consistency_threshold), 1e-6), 1e-6, None)
            robust_weights = np.clip(robust_weights, 1e-6, None)
            robust_high_rgb = np.sum(high_samples * robust_weights[:, None], axis=0) / np.clip(np.sum(robust_weights), 1e-6, None)
            fused_rgb[idx] = np.clip(low_rgb + high_consistency * robust_high_rgb, 0.0, 1.0).astype(np.float32)
            high_frequency_rgb[idx] = robust_high_rgb.astype(np.float32)
            fusion_case[idx] = 1
        else:
            fused_rgb[idx] = low_rgb.astype(np.float32)
            high_frequency_rgb[idx] = 0.0
            fusion_case[idx] = 2

        overlap_confidence = min(float(count) / max(float(min_views_per_carrier), 1.0), 1.0)
        confidence[idx] = float(low_consistency * (0.5 + 0.5 * high_consistency) * overlap_confidence)
        valid_mask[idx] = count >= 2

    return {
        "fused_rgb": fused_rgb,
        "confidence": confidence,
        "disagreement": disagreement,
        "view_count": view_count,
        "weight_sum": weight_sum,
        "valid_mask": valid_mask,
        "low_frequency_rgb": low_frequency_rgb,
        "high_frequency_rgb": high_frequency_rgb,
        "high_frequency_confidence": high_frequency_confidence,
        "fusion_case": fusion_case,
    }


def fuse_bounded_carriers(
    mesh: trimesh.Trimesh,
    carriers: Dict[str, np.ndarray],
    views,
    prior_index: Dict[str, Path],
    mesh_mask_dir: Optional[Path],
    load_mask_fn,
    load_prior_image_fn,
    min_views_per_carrier: int,
    depth_min: float,
    visibility_epsilon: float,
    ray_chunk_size: int,
    carrier_visibility_mode: str = "rasterizer",
    carrier_visibility_depth_tolerance: float = 0.03,
    carrier_visibility_alpha_threshold: float = 0.02,
    carrier_visibility_scale_modifier: float = 1.0,
    carrier_visibility_opacity: float = 0.95,
    splat_args: Optional[ExtendedSettings] = None,
    fusion_policy: str = "mean_v0",
    low_frequency_consistency_threshold: float = 0.08,
    high_frequency_consistency_threshold: float = 0.04,
    single_view_confidence: float = 0.2,
    frequency_blur_kernel: int = 9,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    centers = carriers["centers"]
    normals = carriers["normals"]
    face_ids = carriers["face_ids"]
    num_carriers = centers.shape[0]
    samples_rgb = [[] for _ in range(num_carriers)]
    samples_low_rgb = [[] for _ in range(num_carriers)]
    samples_high_rgb = [[] for _ in range(num_carriers)]
    samples_weight = [[] for _ in range(num_carriers)]

    intersector = build_mesh_intersector(mesh) if carrier_visibility_mode == "ray" else None
    raster_tensors = None
    if carrier_visibility_mode == "rasterizer":
        raster_tensors = carrier_rasterizer_tensors(
            carriers,
            scale_modifier=float(carrier_visibility_scale_modifier),
            opacity=float(carrier_visibility_opacity),
        )
    view_iter = tqdm(
        views,
        desc="Fusing mesh carriers by view",
        disable=not bool(show_progress),
        dynamic_ncols=True,
    )
    for view in view_iter:
        prior_path = prior_index.get(view.image_name)
        if prior_path is None:
            view_iter.set_postfix(view=view.image_name, status="missing_prior")
            continue
        prior_image = load_prior_image_fn(prior_path).cpu().numpy().astype(np.float32, copy=False)
        mesh_mask = None
        if mesh_mask_dir is not None:
            mask_path = mesh_mask_dir / f"{view.image_name}_inject.png"
            if not mask_path.is_file():
                view_iter.set_postfix(view=view.image_name, status="missing_mask")
                continue
            mesh_mask = load_mask_fn(mask_path).cpu().numpy() > 0.5
            target_height, target_width = mesh_mask.shape
        else:
            target_height = int(view.image_height)
            target_width = int(view.image_width)
        prior_image = resize_rgb_image_np(prior_image, target_height, target_width)
        low_prior_image = box_blur_rgb_np(prior_image, int(frequency_blur_kernel)) if fusion_policy == "freq_split_v1" else None

        projected, valid = project_points_camera(view, centers, depth_min=depth_min, margin=0)
        if projected.shape[0] == 0:
            view_iter.set_postfix(view=view.image_name, visible=0)
            continue

        cam_center = camera_center_numpy(view)
        view_dir = cam_center[None, :] - centers
        view_dir_norm = view_dir / np.clip(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-6, None)
        front_facing = (normals * view_dir_norm).sum(axis=1) > 0.0
        valid &= front_facing

        pix_x = np.round(projected[:, 0]).astype(np.int64)
        pix_y = np.round(projected[:, 1]).astype(np.int64)
        in_bounds = (
            (pix_x >= 0)
            & (pix_x < target_width)
            & (pix_y >= 0)
            & (pix_y < target_height)
        )
        valid &= in_bounds
        if not np.any(valid):
            view_iter.set_postfix(view=view.image_name, visible=0)
            continue
        valid_ids = np.flatnonzero(valid)
        if mesh_mask is not None:
            valid_ids = valid_ids[mesh_mask[pix_y[valid_ids], pix_x[valid_ids]]]
            if valid_ids.size == 0:
                view_iter.set_postfix(view=view.image_name, visible=0)
                continue

        if carrier_visibility_mode == "rasterizer":
            raster_depth, raster_alpha = render_carrier_depth_alpha(view, raster_tensors, splat_args)
            visible_ids = filter_carriers_by_rasterized_depth(
                valid_ids,
                pix_x=pix_x,
                pix_y=pix_y,
                depths=projected[:, 2],
                raster_depth=raster_depth,
                raster_alpha=raster_alpha,
                depth_tolerance=float(carrier_visibility_depth_tolerance),
                alpha_threshold=float(carrier_visibility_alpha_threshold),
            )
        elif carrier_visibility_mode == "carrier_zbuffer":
            visible_ids = filter_carriers_by_projected_zbuffer(
                valid_ids,
                pix_x=pix_x,
                pix_y=pix_y,
                depths=projected[:, 2],
                width=target_width,
                depth_tolerance=float(carrier_visibility_depth_tolerance),
            )
        elif carrier_visibility_mode == "projection":
            visible_ids = valid_ids
        elif carrier_visibility_mode == "ray":
            if intersector is None:
                visible_ids = valid_ids
            else:
                ray_origins = np.repeat(cam_center[None, :], valid_ids.shape[0], axis=0).astype(np.float32, copy=False)
                ray_directions = ray_dirs_to_projected_pixels(view, projected[valid_ids, :2])
                keep_visible = np.zeros((valid_ids.shape[0],), dtype=bool)
                try:
                    for begin in range(0, valid_ids.shape[0], max(int(ray_chunk_size), 1)):
                        end = min(begin + max(int(ray_chunk_size), 1), valid_ids.shape[0])
                        chunk_origins = ray_origins[begin:end]
                        chunk_dirs = ray_directions[begin:end]
                        ray_ids, hit_points, hit_tris = first_ray_hits(intersector, chunk_origins, chunk_dirs)
                        if ray_ids.size == 0:
                            continue
                        local_candidate_ids = valid_ids[begin:end][ray_ids]
                        same_face = hit_tris == face_ids[local_candidate_ids]
                        close_hit = np.linalg.norm(hit_points - centers[local_candidate_ids], axis=1) <= float(visibility_epsilon)
                        accepted = same_face | close_hit
                        keep_visible[begin + ray_ids[accepted]] = True
                    visible_ids = valid_ids[keep_visible]
                except Exception as exc:
                    print(f"[prior-fusion] warning: ray visibility failed during fusion ({exc}); falling back to projection-only visibility.")
                    intersector = None
                    visible_ids = valid_ids
        else:
            raise ValueError(f"Unsupported carrier_visibility_mode: {carrier_visibility_mode}")

        if visible_ids.size == 0:
            view_iter.set_postfix(view=view.image_name, visible=0)
            continue

        sampled_rgb = bilinear_sample_rgb(prior_image, projected[visible_ids, :2])
        sampled_low_rgb = None
        sampled_high_rgb = None
        if low_prior_image is not None:
            sampled_low_rgb = bilinear_sample_rgb(low_prior_image, projected[visible_ids, :2])
            sampled_high_rgb = sampled_rgb - sampled_low_rgb
        weights = np.clip((normals[visible_ids] * view_dir_norm[visible_ids]).sum(axis=1), 0.0, 1.0).astype(np.float32)
        weights = np.clip(weights, 1e-3, None)

        accumulate_carrier_samples(
            num_carriers=num_carriers,
            carrier_ids=visible_ids,
            sampled_rgb=sampled_rgb,
            sampled_low_rgb=sampled_low_rgb,
            sampled_high_rgb=sampled_high_rgb,
            weights=weights,
            samples_rgb=samples_rgb,
            samples_low_rgb=samples_low_rgb,
            samples_high_rgb=samples_high_rgb,
            samples_weight=samples_weight,
        )
        view_iter.set_postfix(view=view.image_name, visible=int(visible_ids.size))

    if fusion_policy == "mean_v0":
        return fuse_carrier_samples_mean(
            num_carriers=num_carriers,
            samples_rgb=samples_rgb,
            samples_weight=samples_weight,
            min_views_per_carrier=min_views_per_carrier,
            total_views=len(views),
            show_progress=show_progress,
        )
    if fusion_policy == "freq_split_v1":
        return fuse_carrier_samples_freq_split(
            num_carriers=num_carriers,
            samples_rgb=samples_rgb,
            samples_low_rgb=samples_low_rgb,
            samples_high_rgb=samples_high_rgb,
            samples_weight=samples_weight,
            min_views_per_carrier=min_views_per_carrier,
            total_views=len(views),
            low_frequency_consistency_threshold=low_frequency_consistency_threshold,
            high_frequency_consistency_threshold=high_frequency_consistency_threshold,
            single_view_confidence=single_view_confidence,
            show_progress=show_progress,
        )
    raise ValueError(f"Unsupported carrier fusion policy: {fusion_policy}")


def export_fused_carrier_point_cloud(
    centers: np.ndarray,
    fused_rgb: np.ndarray,
    valid_mask: np.ndarray,
    path: Path,
):
    if centers.shape[0] == 0 or not np.any(valid_mask):
        return
    colors = np.clip(fused_rgb[valid_mask], 0.0, 1.0)
    color_u8 = np.round(colors * 255.0).astype(np.uint8)
    cloud = trimesh.points.PointCloud(centers[valid_mask], colors=color_u8)
    cloud.export(path)


def _stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _rgba_from_rgb(rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    alpha_channel = np.full((rgb.shape[0], 1), int(alpha), dtype=np.uint8)
    return np.concatenate(
        [np.round(rgb * 255.0).astype(np.uint8), alpha_channel],
        axis=1,
    )


def _colorize_confidence_rgba(confidence: np.ndarray) -> np.ndarray:
    conf = np.clip(np.asarray(confidence, dtype=np.float32).reshape(-1), 0.0, 1.0)
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


def _compact_mesh_from_faces(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    face_colors: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    selected_faces = mesh_faces[np.asarray(face_ids, dtype=np.int64)]
    used_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    compact_faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    compact_vertices = mesh_vertices[used_vertices].astype(np.float32, copy=False)

    if vertex_colors is not None:
        compact_mesh = trimesh.Trimesh(
            vertices=compact_vertices,
            faces=compact_faces,
            vertex_colors=np.asarray(vertex_colors, dtype=np.uint8),
            process=False,
        )
    else:
        compact_mesh = trimesh.Trimesh(vertices=compact_vertices, faces=compact_faces, process=False)
    if face_colors is not None:
        compact_mesh.visual.face_colors = np.asarray(face_colors, dtype=np.uint8)
    return compact_mesh, used_vertices.astype(np.int64, copy=False)


def build_sparse_fused_mesh_overlay(
    mesh: trimesh.Trimesh,
    carriers: Dict[str, np.ndarray],
    fused_payload: Dict[str, np.ndarray],
    confidence_power: float = 1.0,
) -> Dict[str, np.ndarray]:
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    face_ids = np.asarray(carriers["face_ids"], dtype=np.int64).reshape(-1)
    bary_coords = np.asarray(carriers["bary_coords"], dtype=np.float32)
    fused_rgb = np.clip(np.asarray(fused_payload["fused_rgb"], dtype=np.float32), 0.0, 1.0)
    confidence = np.clip(np.asarray(fused_payload["confidence"], dtype=np.float32).reshape(-1), 0.0, 1.0)
    disagreement = np.asarray(fused_payload["disagreement"], dtype=np.float32).reshape(-1)
    view_count = np.asarray(fused_payload["view_count"], dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(fused_payload["valid_mask"]).astype(bool).reshape(-1)
    high_frequency_confidence = np.asarray(
        fused_payload.get("high_frequency_confidence", np.zeros_like(confidence)),
        dtype=np.float32,
    ).reshape(-1)

    if face_ids.shape[0] != bary_coords.shape[0] or face_ids.shape[0] != fused_rgb.shape[0]:
        raise ValueError(
            "carrier geometry and fused payload have inconsistent lengths: "
            f"face_ids={face_ids.shape[0]}, bary={bary_coords.shape[0]}, fused_rgb={fused_rgb.shape[0]}"
        )

    carrier_mask = valid_mask & (face_ids >= 0) & (face_ids < mesh_faces.shape[0])
    if not np.any(carrier_mask):
        return {
            "supported_face_ids": np.zeros((0,), dtype=np.int64),
            "supported_face_rgb": np.zeros((0, 3), dtype=np.float32),
            "supported_face_confidence": np.zeros((0,), dtype=np.float32),
            "supported_face_disagreement": np.zeros((0,), dtype=np.float32),
            "supported_face_view_count": np.zeros((0,), dtype=np.float32),
            "supported_face_support_count": np.zeros((0,), dtype=np.int32),
            "supported_face_high_frequency_confidence": np.zeros((0,), dtype=np.float32),
            "supported_vertex_ids": np.zeros((0,), dtype=np.int64),
            "supported_vertex_rgb": np.zeros((0, 3), dtype=np.float32),
            "supported_vertex_confidence": np.zeros((0,), dtype=np.float32),
            "supported_vertex_disagreement": np.zeros((0,), dtype=np.float32),
            "supported_vertex_view_count": np.zeros((0,), dtype=np.float32),
            "supported_vertex_support_weight": np.zeros((0,), dtype=np.float32),
            "supported_vertex_high_frequency_confidence": np.zeros((0,), dtype=np.float32),
        }

    carrier_ids = np.flatnonzero(carrier_mask)
    carrier_face_ids = face_ids[carrier_ids]
    carrier_bary = bary_coords[carrier_ids].astype(np.float32, copy=False)
    carrier_rgb = fused_rgb[carrier_ids].astype(np.float32, copy=False)
    carrier_confidence = confidence[carrier_ids].astype(np.float32, copy=False)
    carrier_disagreement = disagreement[carrier_ids].astype(np.float32, copy=False)
    carrier_view_count = view_count[carrier_ids].astype(np.float32, copy=False)
    carrier_hf_confidence = high_frequency_confidence[carrier_ids].astype(np.float32, copy=False)
    carrier_weight = np.power(np.clip(carrier_confidence, 0.0, 1.0), float(confidence_power)).astype(np.float32)
    carrier_weight = np.clip(carrier_weight, 1e-6, None)

    unique_face_ids, face_inverse = np.unique(carrier_face_ids, return_inverse=True)
    face_weight = np.zeros((unique_face_ids.shape[0],), dtype=np.float32)
    face_rgb_sum = np.zeros((unique_face_ids.shape[0], 3), dtype=np.float32)
    face_confidence_sum = np.zeros_like(face_weight)
    face_disagreement_sum = np.zeros_like(face_weight)
    face_view_count_sum = np.zeros_like(face_weight)
    face_hf_confidence_sum = np.zeros_like(face_weight)
    face_support_count = np.zeros((unique_face_ids.shape[0],), dtype=np.int32)

    np.add.at(face_weight, face_inverse, carrier_weight)
    np.add.at(face_support_count, face_inverse, 1)
    np.add.at(face_confidence_sum, face_inverse, carrier_weight * carrier_confidence)
    np.add.at(face_disagreement_sum, face_inverse, carrier_weight * carrier_disagreement)
    np.add.at(face_view_count_sum, face_inverse, carrier_weight * carrier_view_count)
    np.add.at(face_hf_confidence_sum, face_inverse, carrier_weight * carrier_hf_confidence)
    for channel in range(3):
        np.add.at(face_rgb_sum[:, channel], face_inverse, carrier_weight * carrier_rgb[:, channel])

    face_weight_safe = np.clip(face_weight, 1e-6, None)
    supported_face_rgb = face_rgb_sum / face_weight_safe[:, None]
    supported_face_confidence = face_confidence_sum / face_weight_safe
    supported_face_disagreement = face_disagreement_sum / face_weight_safe
    supported_face_view_count = face_view_count_sum / face_weight_safe
    supported_face_high_frequency_confidence = face_hf_confidence_sum / face_weight_safe

    carrier_face_vertices = mesh_faces[carrier_face_ids]
    repeated_vertex_ids = carrier_face_vertices.reshape(-1)
    repeated_vertex_weight = (carrier_bary * carrier_weight[:, None]).reshape(-1).astype(np.float32, copy=False)
    repeated_rgb = np.repeat(carrier_rgb, 3, axis=0).astype(np.float32, copy=False)
    repeated_confidence = np.repeat(carrier_confidence, 3).astype(np.float32, copy=False)
    repeated_disagreement = np.repeat(carrier_disagreement, 3).astype(np.float32, copy=False)
    repeated_view_count = np.repeat(carrier_view_count, 3).astype(np.float32, copy=False)
    repeated_hf_confidence = np.repeat(carrier_hf_confidence, 3).astype(np.float32, copy=False)

    unique_vertex_ids, vertex_inverse = np.unique(repeated_vertex_ids, return_inverse=True)
    vertex_weight = np.zeros((unique_vertex_ids.shape[0],), dtype=np.float32)
    vertex_rgb_sum = np.zeros((unique_vertex_ids.shape[0], 3), dtype=np.float32)
    vertex_confidence_sum = np.zeros_like(vertex_weight)
    vertex_disagreement_sum = np.zeros_like(vertex_weight)
    vertex_view_count_sum = np.zeros_like(vertex_weight)
    vertex_hf_confidence_sum = np.zeros_like(vertex_weight)

    np.add.at(vertex_weight, vertex_inverse, repeated_vertex_weight)
    np.add.at(vertex_confidence_sum, vertex_inverse, repeated_vertex_weight * repeated_confidence)
    np.add.at(vertex_disagreement_sum, vertex_inverse, repeated_vertex_weight * repeated_disagreement)
    np.add.at(vertex_view_count_sum, vertex_inverse, repeated_vertex_weight * repeated_view_count)
    np.add.at(vertex_hf_confidence_sum, vertex_inverse, repeated_vertex_weight * repeated_hf_confidence)
    for channel in range(3):
        np.add.at(vertex_rgb_sum[:, channel], vertex_inverse, repeated_vertex_weight * repeated_rgb[:, channel])

    vertex_weight_safe = np.clip(vertex_weight, 1e-6, None)
    supported_vertex_rgb = vertex_rgb_sum / vertex_weight_safe[:, None]
    supported_vertex_confidence = vertex_confidence_sum / vertex_weight_safe
    supported_vertex_disagreement = vertex_disagreement_sum / vertex_weight_safe
    supported_vertex_view_count = vertex_view_count_sum / vertex_weight_safe
    supported_vertex_high_frequency_confidence = vertex_hf_confidence_sum / vertex_weight_safe

    return {
        "supported_face_ids": unique_face_ids.astype(np.int64, copy=False),
        "supported_face_rgb": supported_face_rgb.astype(np.float32, copy=False),
        "supported_face_confidence": supported_face_confidence.astype(np.float32, copy=False),
        "supported_face_disagreement": supported_face_disagreement.astype(np.float32, copy=False),
        "supported_face_view_count": supported_face_view_count.astype(np.float32, copy=False),
        "supported_face_support_count": face_support_count.astype(np.int32, copy=False),
        "supported_face_high_frequency_confidence": supported_face_high_frequency_confidence.astype(np.float32, copy=False),
        "supported_vertex_ids": unique_vertex_ids.astype(np.int64, copy=False),
        "supported_vertex_rgb": supported_vertex_rgb.astype(np.float32, copy=False),
        "supported_vertex_confidence": supported_vertex_confidence.astype(np.float32, copy=False),
        "supported_vertex_disagreement": supported_vertex_disagreement.astype(np.float32, copy=False),
        "supported_vertex_view_count": supported_vertex_view_count.astype(np.float32, copy=False),
        "supported_vertex_support_weight": vertex_weight.astype(np.float32, copy=False),
        "supported_vertex_high_frequency_confidence": supported_vertex_high_frequency_confidence.astype(np.float32, copy=False),
    }


def export_fused_carrier_mesh_assets(
    mesh: trimesh.Trimesh,
    carriers: Dict[str, np.ndarray],
    fused_payload: Dict[str, np.ndarray],
    output_dir: Path,
    confidence_power: float = 1.0,
) -> Dict[str, object]:
    overlay = build_sparse_fused_mesh_overlay(
        mesh=mesh,
        carriers=carriers,
        fused_payload=fused_payload,
        confidence_power=float(confidence_power),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_path = output_dir / "bounded_carrier_sparse_mesh_overlay_v0.npz"
    np.savez_compressed(payload_path, **overlay)

    supported_face_ids = overlay["supported_face_ids"]
    supported_vertex_ids = overlay["supported_vertex_ids"]
    face_rgb_mesh_path = None
    face_confidence_mesh_path = None
    vertex_rgb_mesh_path = None
    vertex_confidence_mesh_path = None

    if supported_face_ids.size > 0:
        face_rgb_mesh, used_vertices = _compact_mesh_from_faces(
            mesh,
            supported_face_ids,
            face_colors=_rgba_from_rgb(overlay["supported_face_rgb"]),
        )
        face_rgb_mesh_path = output_dir / "bounded_carrier_supported_faces_rgb_v0.ply"
        face_rgb_mesh.export(face_rgb_mesh_path)

        face_confidence_mesh, _ = _compact_mesh_from_faces(
            mesh,
            supported_face_ids,
            face_colors=_colorize_confidence_rgba(overlay["supported_face_confidence"]),
        )
        face_confidence_mesh_path = output_dir / "bounded_carrier_supported_faces_confidence_v0.ply"
        face_confidence_mesh.export(face_confidence_mesh_path)

        vertex_color_lut = {int(idx): i for i, idx in enumerate(supported_vertex_ids.tolist())}
        compact_rgb = np.tile(np.asarray([[140, 140, 140, 255]], dtype=np.uint8), (used_vertices.shape[0], 1))
        compact_conf = compact_rgb.copy()
        overlay_vertex_rgb = _rgba_from_rgb(overlay["supported_vertex_rgb"])
        overlay_vertex_conf = _colorize_confidence_rgba(overlay["supported_vertex_confidence"])
        for local_idx, global_vertex_id in enumerate(used_vertices.tolist()):
            payload_vertex_idx = vertex_color_lut.get(int(global_vertex_id))
            if payload_vertex_idx is None:
                continue
            compact_rgb[local_idx] = overlay_vertex_rgb[payload_vertex_idx]
            compact_conf[local_idx] = overlay_vertex_conf[payload_vertex_idx]

        vertex_rgb_mesh, _ = _compact_mesh_from_faces(
            mesh,
            supported_face_ids,
            vertex_colors=compact_rgb,
        )
        vertex_rgb_mesh_path = output_dir / "bounded_carrier_supported_vertices_rgb_v0.ply"
        vertex_rgb_mesh.export(vertex_rgb_mesh_path)

        vertex_confidence_mesh, _ = _compact_mesh_from_faces(
            mesh,
            supported_face_ids,
            vertex_colors=compact_conf,
        )
        vertex_confidence_mesh_path = output_dir / "bounded_carrier_supported_vertices_confidence_v0.ply"
        vertex_confidence_mesh.export(vertex_confidence_mesh_path)

    return {
        "payload_path": str(payload_path.resolve()),
        "supported_face_count": int(supported_face_ids.shape[0]),
        "supported_vertex_count": int(supported_vertex_ids.shape[0]),
        "supported_face_ratio_vs_mesh": float(supported_face_ids.shape[0] / max(len(mesh.faces), 1)),
        "supported_vertex_ratio_vs_mesh": float(supported_vertex_ids.shape[0] / max(len(mesh.vertices), 1)),
        "supported_faces_rgb_mesh_path": str(face_rgb_mesh_path.resolve()) if face_rgb_mesh_path is not None else None,
        "supported_faces_confidence_mesh_path": (
            str(face_confidence_mesh_path.resolve()) if face_confidence_mesh_path is not None else None
        ),
        "supported_vertices_rgb_mesh_path": str(vertex_rgb_mesh_path.resolve()) if vertex_rgb_mesh_path is not None else None,
        "supported_vertices_confidence_mesh_path": (
            str(vertex_confidence_mesh_path.resolve()) if vertex_confidence_mesh_path is not None else None
        ),
        "face_confidence_stats": _stats_from_array(overlay["supported_face_confidence"]),
        "face_disagreement_stats": _stats_from_array(overlay["supported_face_disagreement"]),
        "face_view_count_stats": _stats_from_array(overlay["supported_face_view_count"]),
        "vertex_confidence_stats": _stats_from_array(overlay["supported_vertex_confidence"]),
        "vertex_disagreement_stats": _stats_from_array(overlay["supported_vertex_disagreement"]),
        "vertex_view_count_stats": _stats_from_array(overlay["supported_vertex_view_count"]),
        "note": (
            "The sparse overlay payload is indexed by original mesh face_ids/vertex_ids. "
            "The exported PLY previews are compact submeshes that keep only faces covered by valid fused carriers."
        ),
    }
