import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData


def _parse_triplet(text: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected exactly 3 comma-separated values.")
    return vals[0], vals[1], vals[2]


def _load_gaussian_ply(path: str, device: torch.device):
    ply = PlyData.read(path)
    vertex = ply.elements[0]
    xyz = np.stack(
        [
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ],
        axis=1,
    )
    opacity = np.asarray(vertex["opacity"])[..., None]
    scale_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1)
    rot_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("rot_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rotations = np.stack([np.asarray(vertex[name]) for name in rot_names], axis=1)
    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32, device=device),
        "opacity_logits": torch.tensor(opacity, dtype=torch.float32, device=device),
        "log_scaling": torch.tensor(scales, dtype=torch.float32, device=device),
        "rotation": torch.tensor(rotations, dtype=torch.float32, device=device),
    }


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _build_surfel_frames(rotations_raw: torch.Tensor, scales: torch.Tensor):
    rot = _quaternion_to_rotation_matrix(rotations_raw)
    min_idx = torch.argmin(scales, dim=1)
    batch_idx = torch.arange(scales.shape[0], device=scales.device)

    # Build tangent axes robustly even when multiple scales are tied.
    remaining = torch.tensor([0, 1, 2], device=scales.device)[None, :].repeat(scales.shape[0], 1)
    keep = remaining != min_idx[:, None]
    tangent_idx = remaining[keep].view(scales.shape[0], 2)
    tangent_scales = torch.gather(scales, 1, tangent_idx)
    major_sel = tangent_scales[:, 0] >= tangent_scales[:, 1]
    major_idx = torch.where(major_sel, tangent_idx[:, 0], tangent_idx[:, 1])
    minor_idx = torch.where(major_sel, tangent_idx[:, 1], tangent_idx[:, 0])

    normal = rot[batch_idx, :, min_idx]
    t_major = rot[batch_idx, :, major_idx]
    t_minor = rot[batch_idx, :, minor_idx]
    normal = F.normalize(normal, dim=-1)
    t_major = F.normalize(t_major - (t_major * normal).sum(dim=-1, keepdim=True) * normal, dim=-1)
    t_minor = F.normalize(torch.cross(normal, t_major, dim=-1), dim=-1)

    radius_major = torch.gather(scales, 1, major_idx[:, None])
    radius_minor = torch.gather(scales, 1, minor_idx[:, None])
    thickness = torch.gather(scales, 1, min_idx[:, None])
    return normal, t_major, t_minor, radius_major, radius_minor, thickness


def _orient_normals_toward_outside(xyz: torch.Tensor, normals: torch.Tensor, weights: torch.Tensor):
    center = (xyz * weights).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    outward = xyz - center
    outward_norm = outward.norm(dim=-1, keepdim=True)
    flip = ((normals * outward).sum(dim=-1, keepdim=True) < 0) & (outward_norm > 1e-6)
    return torch.where(flip, -normals, normals), center.squeeze(0)


def _surfel_score(opacity: torch.Tensor, radius_major: torch.Tensor, radius_minor: torch.Tensor, thickness: torch.Tensor):
    sheetness = torch.sqrt((radius_major * radius_minor).clamp(min=1e-8)) / thickness.clamp(min=1e-8)
    return opacity * torch.log1p(sheetness.squeeze(-1))


def _auto_bounds(points: torch.Tensor, padding_ratio: float):
    bmin = points.min(dim=0).values
    bmax = points.max(dim=0).values
    extent = (bmax - bmin).clamp(min=1e-4)
    pad = extent * float(padding_ratio)
    return bmin - pad, bmax + pad


def _focus_bounds_main_object(points: torch.Tensor, weights: torch.Tensor, cluster_ratio: float, inlier_quantile: float, padding_ratio: float):
    n = points.shape[0]
    seed_pool_size = max(1, min(n, max(512, int(0.2 * n))))
    seed_idx_pool = torch.topk(weights.squeeze(-1), k=seed_pool_size, largest=True).indices
    pool = points[seed_idx_pool]
    dist = torch.cdist(pool, pool)
    diag = torch.arange(pool.shape[0], device=points.device)
    dist[diag, diag] = float("inf")
    nn_mean = torch.topk(dist, k=min(16, max(1, pool.shape[0] - 1)), largest=False).values.mean(dim=1)
    seed_score = weights[seed_idx_pool, 0] / nn_mean.clamp(min=1e-6)
    seed_idx = seed_idx_pool[torch.argmax(seed_score)]
    seed = points[seed_idx : seed_idx + 1]

    cluster_size = max(32, min(n, int(max(0.05, cluster_ratio) * n)))
    dist_all = torch.cdist(seed, points).squeeze(0)
    cluster_idx = torch.topk(dist_all, k=cluster_size, largest=False).indices
    cluster_pts = points[cluster_idx]
    cluster_w = weights[cluster_idx]
    center = (cluster_pts * cluster_w).sum(dim=0) / cluster_w.sum(dim=0).clamp(min=1e-6)
    center_dist = torch.linalg.norm(cluster_pts - center[None, :], dim=1)
    radius = torch.quantile(center_dist, q=float(inlier_quantile))
    inlier_mask = center_dist <= radius
    focus_pts = cluster_pts[inlier_mask]
    if focus_pts.shape[0] < 16:
        focus_pts = cluster_pts
    bmin, bmax = _auto_bounds(focus_pts, padding_ratio)
    return bmin, bmax, {
        "mode": "main_object",
        "focus_center": center.detach().cpu().tolist(),
        "focus_size": int(focus_pts.shape[0]),
        "cluster_size": int(cluster_pts.shape[0]),
        "radius": float(radius.item()),
    }


def _manual_bounds(center_text: str, extent_text: str, device: torch.device):
    center = torch.tensor(_parse_triplet(center_text), dtype=torch.float32, device=device)
    extent = torch.tensor(_parse_triplet(extent_text), dtype=torch.float32, device=device).abs().clamp(min=1e-5)
    half = 0.5 * extent
    return center - half, center + half


def _disk_offsets(samples_per_surfel: int, device: torch.device, dtype: torch.dtype):
    base = [
        (0.0, 0.0),
        (0.45, 0.0),
        (-0.45, 0.0),
        (0.0, 0.45),
        (0.0, -0.45),
        (0.32, 0.32),
        (-0.32, 0.32),
        (0.32, -0.32),
        (-0.32, -0.32),
    ]
    samples = max(1, min(samples_per_surfel, len(base)))
    arr = torch.tensor(base[:samples], device=device, dtype=dtype)
    return arr


def _sample_surfel_points(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    t_major: torch.Tensor,
    t_minor: torch.Tensor,
    radius_major: torch.Tensor,
    radius_minor: torch.Tensor,
    samples_per_surfel: int,
):
    offsets = _disk_offsets(samples_per_surfel, device=xyz.device, dtype=xyz.dtype)
    pts = (
        xyz[:, None, :]
        + offsets[None, :, 0:1] * radius_major[:, None, :] * t_major[:, None, :]
        + offsets[None, :, 1:2] * radius_minor[:, None, :] * t_minor[:, None, :]
    )
    nrm = normals[:, None, :].expand(-1, offsets.shape[0], -1)
    return pts.reshape(-1, 3), nrm.reshape(-1, 3)


def _crop_points(points: torch.Tensor, normals: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor):
    mask = ((points >= bmin[None, :]) & (points <= bmax[None, :])).all(dim=1)
    return points[mask], normals[mask], mask


def _write_surfel_ply(path: str, points: np.ndarray, normals: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path, pcd)


def _poisson_reconstruct(points: np.ndarray, normals: np.ndarray, depth: int, density_prune_quantile: float):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(depth),
        scale=1.1,
        linear_fit=True,
    )
    densities = np.asarray(densities)
    if densities.size > 0 and density_prune_quantile > 0:
        keep_thresh = np.quantile(densities, float(density_prune_quantile))
        remove_mask = densities < keep_thresh
        mesh.remove_vertices_by_mask(remove_mask)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(description="Extract a 2DGS-style surfel surface mesh from Gaussian point_cloud.ply.")
    parser.add_argument("--point_cloud_ply", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opacity_min", type=float, default=0.02)
    parser.add_argument("--sheetness_min", type=float, default=2.0)
    parser.add_argument("--max_surfels", type=int, default=60000)
    parser.add_argument("--samples_per_surfel", type=int, default=3)
    parser.add_argument("--normal_axis", type=str, default="min_scale", choices=["x", "y", "z", "min_scale", "max_scale"])
    parser.add_argument("--focus_mode", type=str, default="main_object", choices=["global", "main_object", "manual"])
    parser.add_argument("--focus_cluster_ratio", type=float, default=0.5)
    parser.add_argument("--focus_inlier_quantile", type=float, default=0.9)
    parser.add_argument("--focus_padding_ratio", type=float, default=0.08)
    parser.add_argument("--focus_center", type=str, default="")
    parser.add_argument("--focus_extent", type=str, default="")
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--density_prune_quantile", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    raw = _load_gaussian_ply(args.point_cloud_ply, device=device)
    xyz = raw["xyz"]
    scales = torch.exp(raw["log_scaling"])
    rotations = raw["rotation"]
    opacity = torch.sigmoid(raw["opacity_logits"]).view(-1)

    normal, t_major, t_minor, radius_major, radius_minor, thickness = _build_surfel_frames(rotations, scales)
    if args.normal_axis != "min_scale":
        if args.normal_axis == "max_scale":
            perm = torch.argsort(scales, dim=1, descending=True)
        else:
            axis_map = {"x": 0, "y": 1, "z": 2}
            idx = axis_map[args.normal_axis]
            rot = _quaternion_to_rotation_matrix(rotations)
            normal = F.normalize(rot[:, :, idx], dim=-1)

    sheetness = torch.sqrt((radius_major * radius_minor).clamp(min=1e-8)) / thickness.clamp(min=1e-8)
    valid = (opacity >= float(args.opacity_min)) & (sheetness.squeeze(-1) >= float(args.sheetness_min))
    xyz = xyz[valid]
    normal = normal[valid]
    t_major = t_major[valid]
    t_minor = t_minor[valid]
    radius_major = radius_major[valid]
    radius_minor = radius_minor[valid]
    thickness = thickness[valid]
    opacity = opacity[valid]
    sheetness = sheetness[valid]

    if xyz.shape[0] == 0:
        raise RuntimeError("No surfel-like Gaussians left after opacity/sheetness filtering.")

    score = _surfel_score(opacity, radius_major, radius_minor, thickness)
    if xyz.shape[0] > int(args.max_surfels):
        keep_idx = torch.topk(score, k=int(args.max_surfels), largest=True).indices
        xyz = xyz[keep_idx]
        normal = normal[keep_idx]
        t_major = t_major[keep_idx]
        t_minor = t_minor[keep_idx]
        radius_major = radius_major[keep_idx]
        radius_minor = radius_minor[keep_idx]
        thickness = thickness[keep_idx]
        opacity = opacity[keep_idx]
        sheetness = sheetness[keep_idx]
        score = score[keep_idx]

    normal, centroid = _orient_normals_toward_outside(xyz, normal, opacity[:, None])

    if args.focus_mode == "main_object":
        bmin, bmax, focus_info = _focus_bounds_main_object(
            points=xyz,
            weights=score[:, None],
            cluster_ratio=args.focus_cluster_ratio,
            inlier_quantile=args.focus_inlier_quantile,
            padding_ratio=args.focus_padding_ratio,
        )
    elif args.focus_mode == "manual":
        if not args.focus_center or not args.focus_extent:
            raise ValueError("--focus_center and --focus_extent are required for focus_mode=manual")
        bmin, bmax = _manual_bounds(args.focus_center, args.focus_extent, device=device)
        focus_info = {
            "mode": "manual",
            "focus_center": ((bmin + bmax) * 0.5).detach().cpu().tolist(),
            "focus_extent": (bmax - bmin).detach().cpu().tolist(),
        }
    else:
        bmin, bmax = _auto_bounds(xyz, padding_ratio=args.focus_padding_ratio)
        focus_info = {
            "mode": "global",
            "focus_center": ((bmin + bmax) * 0.5).detach().cpu().tolist(),
            "focus_extent": (bmax - bmin).detach().cpu().tolist(),
        }

    surfel_centers, surfel_normals, center_mask = _crop_points(xyz, normal, bmin, bmax)
    surfel_t_major = t_major[center_mask]
    surfel_t_minor = t_minor[center_mask]
    surfel_radius_major = radius_major[center_mask]
    surfel_radius_minor = radius_minor[center_mask]
    surfel_score = score[center_mask]

    sampled_points, sampled_normals = _sample_surfel_points(
        xyz=surfel_centers,
        normals=surfel_normals,
        t_major=surfel_t_major,
        t_minor=surfel_t_minor,
        radius_major=surfel_radius_major,
        radius_minor=surfel_radius_minor,
        samples_per_surfel=int(args.samples_per_surfel),
    )
    sampled_points, sampled_normals, _ = _crop_points(sampled_points, sampled_normals, bmin, bmax)

    surfel_ply_path = os.path.join(args.output_dir, "surfel_points.ply")
    mesh_path = os.path.join(args.output_dir, "surface_mesh_poisson.ply")
    _write_surfel_ply(
        surfel_ply_path,
        points=sampled_points.detach().cpu().numpy(),
        normals=sampled_normals.detach().cpu().numpy(),
    )
    mesh = _poisson_reconstruct(
        points=sampled_points.detach().cpu().numpy(),
        normals=sampled_normals.detach().cpu().numpy(),
        depth=int(args.poisson_depth),
        density_prune_quantile=float(args.density_prune_quantile),
    )

    import open3d as o3d

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bmin.detach().cpu().numpy(),
        max_bound=bmax.detach().cpu().numpy(),
    )
    mesh = mesh.crop(bbox)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    info = {
        "point_cloud_ply": args.point_cloud_ply,
        "surfel_count": int(surfel_centers.shape[0]),
        "sampled_point_count": int(sampled_points.shape[0]),
        "sheetness_mean": float(sheetness.mean().item()),
        "focus": focus_info,
        "bounds_min": bmin.detach().cpu().tolist(),
        "bounds_max": bmax.detach().cpu().tolist(),
        "centroid": centroid.detach().cpu().tolist(),
        "surface_mesh_poisson": mesh_path,
        "surfel_points": surfel_ply_path,
    }
    with open(os.path.join(args.output_dir, "surface_extract_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("[2dgs-surface] surfels:", int(surfel_centers.shape[0]))
    print("[2dgs-surface] sampled_points:", int(sampled_points.shape[0]))
    print("[2dgs-surface] surfel_ply:", surfel_ply_path)
    print("[2dgs-surface] mesh_ply:", mesh_path)
    print("[2dgs-surface] focus:", focus_info["mode"], "bmin=", info["bounds_min"], "bmax=", info["bounds_max"])


if __name__ == "__main__":
    main()
