import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData


@dataclass
class OfflineGSBootstrapConfig:
    max_anchors: int = 20000
    k_neighbors: int = 8
    opacity_min: float = 0.01
    normal_axis: str = "min_scale"
    sigma_mode: str = "geom_mid_min"
    distance_scale: float = 1.0
    distance_floor: float = 1e-4
    knn_chunk_size: int = 4096
    subsample_mode: str = "sharpness"
    orient_normals_mode: str = "centroid"
    proxy_tangent_reg: float = 0.15
    proxy_max_normal_shift_scale: float = 1.0
    proxy_max_tangent_shift_scale: float = 0.5
    normal_support_scale: float = 2.0
    tangent_support_scale: float = 1.25
    invalid_penalty_scale: float = 1.5


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

    filter_3d = None
    if "filter_3D" in vertex.data.dtype.names:
        filter_3d = np.asarray(vertex["filter_3D"])[..., None]

    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32, device=device),
        "opacity_logits": torch.tensor(opacity, dtype=torch.float32, device=device),
        "log_scaling": torch.tensor(scales, dtype=torch.float32, device=device),
        "rotation": torch.tensor(rotations, dtype=torch.float32, device=device),
        "filter_3d": None if filter_3d is None else torch.tensor(filter_3d, dtype=torch.float32, device=device),
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


def _gaussian_normals(rotations_raw: torch.Tensor, scales: torch.Tensor, axis: str) -> torch.Tensor:
    rotations = _quaternion_to_rotation_matrix(rotations_raw)
    if axis == "x":
        idx = torch.zeros((rotations.shape[0],), dtype=torch.long, device=rotations.device)
    elif axis == "y":
        idx = torch.ones((rotations.shape[0],), dtype=torch.long, device=rotations.device)
    elif axis == "z":
        idx = torch.full((rotations.shape[0],), 2, dtype=torch.long, device=rotations.device)
    elif axis == "max_scale":
        idx = torch.argmax(scales, dim=1)
    else:
        idx = torch.argmin(scales, dim=1)
    batch_idx = torch.arange(rotations.shape[0], device=rotations.device)
    normals = rotations[batch_idx, :, idx]
    return F.normalize(normals, dim=-1)


def _anchor_sigma_from_scales(scales: torch.Tensor, mode: str) -> torch.Tensor:
    scales_sorted, _ = torch.sort(scales, dim=1)
    s_min = scales_sorted[:, 0:1]
    s_mid = scales_sorted[:, 1:2]
    s_max = scales_sorted[:, 2:3]
    if mode == "min":
        sigma = s_min
    elif mode == "mid":
        sigma = s_mid
    elif mode == "max":
        sigma = s_max
    elif mode == "mean":
        sigma = scales.mean(dim=1, keepdim=True)
    else:
        sigma = torch.sqrt((s_min * s_mid).clamp(min=1e-12))
    return sigma


def _anchor_tangent_sigma_from_scales(scales: torch.Tensor) -> torch.Tensor:
    scales_sorted, _ = torch.sort(scales, dim=1)
    s_mid = scales_sorted[:, 1:2]
    s_max = scales_sorted[:, 2:3]
    return torch.sqrt((s_mid * s_max).clamp(min=1e-12))


def _select_anchor_subset(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rotations_raw: torch.Tensor,
    opacity: torch.Tensor,
    normal_sigma: torch.Tensor,
    max_anchors: int,
    mode: str,
):
    if xyz.shape[0] <= int(max_anchors):
        return xyz, scales, rotations_raw, opacity, normal_sigma
    if mode == "random":
        keep_idx = torch.randperm(xyz.shape[0], device=xyz.device)[: int(max_anchors)]
    else:
        sharpness = opacity / normal_sigma.squeeze(-1).clamp(min=1e-6)
        keep_idx = torch.topk(sharpness, k=int(max_anchors), largest=True).indices
    return (
        xyz[keep_idx],
        scales[keep_idx],
        rotations_raw[keep_idx],
        opacity[keep_idx],
        normal_sigma[keep_idx],
    )


def _orient_normals(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    conf: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "none" or xyz.shape[0] == 0:
        return normals
    weights = conf.clamp(min=1e-6)
    center = (xyz * weights).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    outward = xyz - center
    outward_norm = outward.norm(dim=-1, keepdim=True)
    flip = ((normals * outward).sum(dim=-1, keepdim=True) < 0) & (outward_norm > 1e-6)
    return torch.where(flip, -normals, normals)


class OfflineGSBootstrapPseudoSDF:
    def __init__(self, cfg: OfflineGSBootstrapConfig, ply_path: str, device: torch.device):
        self.cfg = cfg
        self.device = device
        raw = _load_gaussian_ply(ply_path, device=device)
        xyz = raw["xyz"]
        scales = torch.exp(raw["log_scaling"])
        rotations_raw = raw["rotation"]
        opacity = torch.sigmoid(raw["opacity_logits"]).view(-1)

        valid = opacity >= float(cfg.opacity_min)
        if valid.any():
            xyz = xyz[valid]
            scales = scales[valid]
            rotations_raw = rotations_raw[valid]
            opacity = opacity[valid]

        normal_sigma = _anchor_sigma_from_scales(scales, cfg.sigma_mode)
        xyz, scales, rotations_raw, opacity, normal_sigma = _select_anchor_subset(
            xyz=xyz,
            scales=scales,
            rotations_raw=rotations_raw,
            opacity=opacity,
            normal_sigma=normal_sigma,
            max_anchors=int(cfg.max_anchors),
            mode=cfg.subsample_mode,
        )

        self.anchor_xyz = xyz
        self.anchor_scales = scales
        self.anchor_normals = _gaussian_normals(rotations_raw, scales, cfg.normal_axis)
        self.anchor_normals = _orient_normals(
            xyz=xyz,
            normals=self.anchor_normals,
            conf=opacity[:, None],
            mode=cfg.orient_normals_mode,
        )
        self.anchor_sigma = (normal_sigma * float(cfg.distance_scale)).clamp(min=float(cfg.distance_floor))
        self.anchor_tangent_sigma = (
            _anchor_tangent_sigma_from_scales(scales) * float(cfg.distance_scale)
        ).clamp(min=float(cfg.distance_floor))
        self.anchor_conf = opacity[:, None].clamp(min=0.0, max=1.0)
        self._refresh_surface_proxies()

    def _knn(self, points: torch.Tensor, k: int):
        n = points.shape[0]
        k = min(int(k), self.anchor_xyz.shape[0])
        if k <= 0:
            empty_dist = torch.zeros((n, 0), dtype=points.dtype, device=points.device)
            empty_idx = torch.zeros((n, 0), dtype=torch.long, device=points.device)
            return empty_dist, empty_idx
        chunk = max(1, int(self.cfg.knn_chunk_size))
        all_dist = []
        all_idx = []
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            dist = torch.cdist(points[start:end], self.anchor_xyz)
            dist_k, idx_k = torch.topk(dist, k=k, dim=1, largest=False, sorted=False)
            all_dist.append(dist_k)
            all_idx.append(idx_k)
        return torch.cat(all_dist, dim=0), torch.cat(all_idx, dim=0)

    @torch.no_grad()
    def query_sdf_and_gradients(self, points: torch.Tensor):
        if self.proxy_xyz.numel() == 0:
            sdf = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
            grad = torch.zeros((points.shape[0], 3), dtype=points.dtype, device=points.device)
            grad[:, 2] = 1.0
            return sdf, grad

        knn_dist, knn_idx = self._knn(points, self.cfg.k_neighbors)
        nbr_xyz = self.proxy_xyz[knn_idx]
        nbr_normals = self.proxy_normals[knn_idx]
        nbr_sigma = self.proxy_sigma[knn_idx].clamp(min=float(self.cfg.distance_floor)).squeeze(-1)
        nbr_tangent_sigma = self.proxy_tangent_sigma[knn_idx].clamp(min=float(self.cfg.distance_floor)).squeeze(-1)
        nbr_conf = self.proxy_conf[knn_idx].clamp(min=0.0).squeeze(-1)

        delta = points[:, None, :] - nbr_xyz
        signed_plane = (delta * nbr_normals).sum(dim=-1)
        tangent_delta = delta - signed_plane[..., None] * nbr_normals
        tangent_dist = tangent_delta.norm(dim=-1)

        normal_support = (float(self.cfg.normal_support_scale) * nbr_sigma).clamp(min=float(self.cfg.distance_floor))
        tangent_support = (float(self.cfg.tangent_support_scale) * nbr_tangent_sigma).clamp(
            min=float(self.cfg.distance_floor)
        )
        score = (
            signed_plane.abs() / normal_support
            + torch.relu(tangent_dist - tangent_support) / tangent_support
            + 0.1 * knn_dist / tangent_support
            - 0.05 * nbr_conf
        )

        best_idx = torch.argmin(score, dim=1)
        row_idx = torch.arange(points.shape[0], device=points.device)
        best_dn = signed_plane[row_idx, best_idx]
        best_tangent_dist = tangent_dist[row_idx, best_idx]
        best_normal_support = normal_support[row_idx, best_idx]
        best_tangent_support = tangent_support[row_idx, best_idx]
        best_normals = nbr_normals[row_idx, best_idx]

        invalid_mag = float(self.cfg.invalid_penalty_scale) * (
            torch.relu(best_tangent_dist - best_tangent_support)
            + torch.relu(best_dn.abs() - best_normal_support)
        )
        abs_sdf = best_dn.abs() + invalid_mag
        sign = torch.where(best_dn >= 0, torch.ones_like(best_dn), -torch.ones_like(best_dn))
        sdf = (sign * abs_sdf).unsqueeze(-1)
        grad = F.normalize(best_normals, dim=-1)
        return sdf, grad

    @torch.no_grad()
    def _refresh_surface_proxies(self):
        if self.anchor_xyz.shape[0] == 0:
            self.proxy_xyz = self.anchor_xyz
            self.proxy_normals = self.anchor_normals
            self.proxy_sigma = self.anchor_sigma
            self.proxy_tangent_sigma = self.anchor_tangent_sigma
            self.proxy_conf = self.anchor_conf
            return

        xyz = self.anchor_xyz
        normals = self.anchor_normals
        normal_sigma = self.anchor_sigma
        tangent_sigma = self.anchor_tangent_sigma
        conf = self.anchor_conf
        n = xyz.shape[0]
        k = min(max(1, int(self.cfg.k_neighbors)), n)
        knn_dist, knn_idx = self._knn(xyz, k)
        nbr_xyz = xyz[knn_idx]
        nbr_normals = normals[knn_idx]
        nbr_tangent_sigma = tangent_sigma[knn_idx].clamp(min=float(self.cfg.distance_floor))
        nbr_conf = conf[knn_idx].clamp(min=0.0)

        ref_normals = normals[:, None, :]
        align = torch.sign((nbr_normals * ref_normals).sum(dim=-1, keepdim=True))
        align = torch.where(align == 0, torch.ones_like(align), align)
        nbr_normals = nbr_normals * align

        weights = torch.exp(-knn_dist / nbr_tangent_sigma.squeeze(-1)) * nbr_conf.squeeze(-1)
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        proxy_normals = F.normalize((weights[..., None] * nbr_normals).sum(dim=1) / weight_sum, dim=-1)

        outer = nbr_normals[..., :, None] * nbr_normals[..., None, :]
        eye = torch.eye(3, device=xyz.device, dtype=xyz.dtype)[None, :, :].expand(n, -1, -1)
        tangent_proj = eye - proxy_normals[:, :, None] * proxy_normals[:, None, :]
        reg = float(self.cfg.proxy_tangent_reg)
        A = (weights[..., None, None] * outer).sum(dim=1) + reg * tangent_proj + 1e-4 * eye
        b = (
            (weights[..., None, None] * outer @ nbr_xyz[..., None]).sum(dim=1).squeeze(-1)
            + reg * (tangent_proj @ xyz[..., None]).squeeze(-1)
        )
        proxy_xyz = torch.linalg.solve(A, b)

        delta = proxy_xyz - xyz
        normal_shift = (delta * proxy_normals).sum(dim=-1, keepdim=True)
        tangent_shift = delta - normal_shift * proxy_normals
        max_normal_shift = float(self.cfg.proxy_max_normal_shift_scale) * normal_sigma
        max_tangent_shift = float(self.cfg.proxy_max_tangent_shift_scale) * tangent_sigma
        normal_shift = torch.maximum(torch.minimum(normal_shift, max_normal_shift), -max_normal_shift)
        tangent_shift_norm = tangent_shift.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        tangent_scale = torch.clamp(max_tangent_shift / tangent_shift_norm, max=1.0)
        tangent_shift = tangent_shift * tangent_scale
        proxy_xyz = xyz + normal_shift * proxy_normals + tangent_shift

        proxy_disp = (proxy_xyz - xyz).norm(dim=-1, keepdim=True)
        proxy_conf = (conf * torch.exp(-proxy_disp / tangent_sigma.clamp(min=float(self.cfg.distance_floor)))).clamp(
            min=0.0,
            max=1.0,
        )

        self.proxy_xyz = proxy_xyz
        self.proxy_normals = proxy_normals
        self.proxy_sigma = normal_sigma
        self.proxy_tangent_sigma = tangent_sigma
        self.proxy_conf = proxy_conf


def _weighted_sharpness(conf: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return conf.squeeze(-1) / sigma.squeeze(-1).clamp(min=1e-6)


def _auto_bounds(anchor_xyz: torch.Tensor, padding_ratio: float):
    bmin = anchor_xyz.min(dim=0).values
    bmax = anchor_xyz.max(dim=0).values
    extent = (bmax - bmin).clamp(min=1e-4)
    pad = extent * float(padding_ratio)
    return bmin - pad, bmax + pad


def _focus_bounds_main_object(
    surface_xyz: torch.Tensor,
    surface_sigma: torch.Tensor,
    surface_conf: torch.Tensor,
    cluster_ratio: float,
    inlier_quantile: float,
    padding_ratio: float,
    seed_pool_size: int,
    seed_knn: int,
):
    n = surface_xyz.shape[0]
    sharpness = _weighted_sharpness(surface_conf, surface_sigma)
    seed_pool_size = max(1, min(int(seed_pool_size), n))
    seed_pool_idx = torch.topk(sharpness, k=seed_pool_size, largest=True).indices
    seed_pool_xyz = surface_xyz[seed_pool_idx]
    local_k = max(1, min(int(seed_knn), max(1, seed_pool_size - 1)))

    if seed_pool_size == 1:
        seed_idx = seed_pool_idx[0]
    else:
        dist = torch.cdist(seed_pool_xyz, seed_pool_xyz)
        diag_idx = torch.arange(seed_pool_size, device=surface_xyz.device)
        dist[diag_idx, diag_idx] = float("inf")
        nn_mean = torch.topk(dist, k=local_k, dim=1, largest=False).values.mean(dim=1)
        seed_score = sharpness[seed_pool_idx] / nn_mean.clamp(min=1e-6)
        seed_idx = seed_pool_idx[torch.argmax(seed_score)]

    seed_xyz = surface_xyz[seed_idx : seed_idx + 1]
    cluster_size = max(local_k * 4, min(n, max(8, int(max(0.05, cluster_ratio) * n))))
    dist_all = torch.cdist(seed_xyz, surface_xyz).squeeze(0)
    cluster_idx = torch.topk(dist_all, k=cluster_size, largest=False).indices

    cluster_xyz = surface_xyz[cluster_idx]
    cluster_sharpness = sharpness[cluster_idx]
    center = (cluster_xyz * cluster_sharpness[:, None]).sum(dim=0) / cluster_sharpness.sum().clamp(min=1e-6)

    center_dist = torch.linalg.norm(cluster_xyz - center[None, :], dim=1)
    inlier_q = float(min(max(inlier_quantile, 0.5), 0.99))
    radius = torch.quantile(center_dist, q=inlier_q)
    inlier_mask = center_dist <= radius
    focus_xyz = cluster_xyz[inlier_mask]
    if focus_xyz.shape[0] < 8:
        focus_xyz = cluster_xyz

    bmin, bmax = _auto_bounds(focus_xyz, padding_ratio=padding_ratio)
    info = {
        "mode": "main_object",
        "seed_index": int(seed_idx.item()),
        "seed_xyz": seed_xyz.squeeze(0).detach().cpu().tolist(),
        "focus_center": center.detach().cpu().tolist(),
        "cluster_size": int(cluster_xyz.shape[0]),
        "focus_size": int(focus_xyz.shape[0]),
        "radius": float(radius.item()),
    }
    return bmin, bmax, info


def _manual_bounds(center_text: str, extent_text: str, device: torch.device):
    center = torch.tensor(_parse_triplet(center_text), dtype=torch.float32, device=device)
    extent = torch.tensor(_parse_triplet(extent_text), dtype=torch.float32, device=device).abs().clamp(min=1e-5)
    half = 0.5 * extent
    return center - half, center + half


def _build_slice_grid(
    bmin: torch.Tensor,
    bmax: torch.Tensor,
    axis: str,
    value: float,
    resolution: int,
    device: torch.device,
):
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    slice_idx = axis_to_idx[axis]
    free_axes = [idx for idx in range(3) if idx != slice_idx]
    xs = torch.linspace(bmin[free_axes[0]], bmax[free_axes[0]], steps=resolution, device=device)
    ys = torch.linspace(bmin[free_axes[1]], bmax[free_axes[1]], steps=resolution, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.zeros((resolution * resolution, 3), dtype=torch.float32, device=device)
    pts[:, free_axes[0]] = grid_x.reshape(-1)
    pts[:, free_axes[1]] = grid_y.reshape(-1)
    pts[:, slice_idx] = float(value)
    return pts, xs, ys, free_axes


def _query_in_chunks(adapter: OfflineGSBootstrapPseudoSDF, points: torch.Tensor, chunk_size: int):
    sdf_chunks = []
    grad_chunks = []
    for start in range(0, points.shape[0], chunk_size):
        end = min(points.shape[0], start + chunk_size)
        sdf, grad = adapter.query_sdf_and_gradients(points[start:end])
        sdf_chunks.append(sdf)
        grad_chunks.append(grad)
    return torch.cat(sdf_chunks, dim=0), torch.cat(grad_chunks, dim=0)


def _save_slice_plot(
    out_path: str,
    sdf_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    axis: str,
    slice_value: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = max(np.percentile(np.abs(sdf_grid), 95), 1e-5)
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=180)
    im = ax.imshow(
        sdf_grid,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    ax.contour(xs, ys, sdf_grid, levels=[0.0], colors="black", linewidths=1.4)
    ax.set_title(f"GS-Bootstrap pseudoSDF slice ({axis}={slice_value:.4f})")
    ax.set_xlabel("world axis")
    ax.set_ylabel("world axis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pseudoSDF value")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _export_mesh(
    adapter: OfflineGSBootstrapPseudoSDF,
    bmin: torch.Tensor,
    bmax: torch.Tensor,
    resolution: int,
    out_path: str,
    chunk_size: int,
):
    from skimage import measure
    import trimesh

    xs = torch.linspace(bmin[0], bmax[0], steps=resolution, device=adapter.device)
    ys = torch.linspace(bmin[1], bmax[1], steps=resolution, device=adapter.device)
    zs = torch.linspace(bmin[2], bmax[2], steps=resolution, device=adapter.device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    sdf, _ = _query_in_chunks(adapter, pts, chunk_size=chunk_size)
    volume = sdf.view(resolution, resolution, resolution).detach().cpu().numpy()

    if np.min(volume) > 0.0 or np.max(volume) < 0.0:
        raise RuntimeError("PseudoSDF volume does not cross zero; marching cubes cannot extract a surface.")

    spacing = (
        float((bmax[2] - bmin[2]).item() / (resolution - 1)),
        float((bmax[1] - bmin[1]).item() / (resolution - 1)),
        float((bmax[0] - bmin[0]).item() / (resolution - 1)),
    )
    verts_zyx, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=spacing)
    verts = np.zeros_like(verts_zyx)
    verts[:, 0] = verts_zyx[:, 2] + float(bmin[0].item())
    verts[:, 1] = verts_zyx[:, 1] + float(bmin[1].item())
    verts[:, 2] = verts_zyx[:, 0] + float(bmin[2].item())
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(out_path)
    return int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Visualize gs_bootstrap pseudoSDF from a saved point_cloud.ply.")
    parser.add_argument("--point_cloud_ply", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_anchors", type=int, default=20000)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--opacity_min", type=float, default=0.01)
    parser.add_argument("--normal_axis", type=str, default="min_scale", choices=["x", "y", "z", "min_scale", "max_scale"])
    parser.add_argument("--sigma_mode", type=str, default="geom_mid_min", choices=["min", "mid", "max", "mean", "geom_mid_min"])
    parser.add_argument("--distance_scale", type=float, default=1.0)
    parser.add_argument("--distance_floor", type=float, default=1e-4)
    parser.add_argument("--knn_chunk_size", type=int, default=4096)
    parser.add_argument("--subsample_mode", type=str, default="sharpness", choices=["random", "sharpness"])
    parser.add_argument("--orient_normals_mode", type=str, default="centroid", choices=["none", "centroid"])
    parser.add_argument("--proxy_tangent_reg", type=float, default=0.15)
    parser.add_argument("--proxy_max_normal_shift_scale", type=float, default=1.0)
    parser.add_argument("--proxy_max_tangent_shift_scale", type=float, default=0.5)
    parser.add_argument("--normal_support_scale", type=float, default=2.0)
    parser.add_argument("--tangent_support_scale", type=float, default=1.25)
    parser.add_argument("--invalid_penalty_scale", type=float, default=1.5)
    parser.add_argument("--slice_axis", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("--slice_value", type=float, default=None)
    parser.add_argument("--slice_resolution", type=int, default=512)
    parser.add_argument("--slice_padding_ratio", type=float, default=0.10)
    parser.add_argument("--slice_bounds_min", type=str, default="")
    parser.add_argument("--slice_bounds_max", type=str, default="")
    parser.add_argument("--query_chunk_size", type=int, default=65536)
    parser.add_argument("--export_slice_png", action="store_true")
    parser.add_argument("--export_slice_npz", action="store_true")
    parser.add_argument("--export_mesh", action="store_true")
    parser.add_argument("--mesh_resolution", type=int, default=128)
    parser.add_argument("--mesh_padding_ratio", type=float, default=0.05)
    parser.add_argument("--mesh_focus_mode", type=str, default="global", choices=["global", "main_object", "manual"])
    parser.add_argument("--mesh_focus_cluster_ratio", type=float, default=0.35)
    parser.add_argument("--mesh_focus_inlier_quantile", type=float, default=0.85)
    parser.add_argument("--mesh_focus_seed_pool_size", type=int, default=4096)
    parser.add_argument("--mesh_focus_seed_knn", type=int, default=32)
    parser.add_argument("--mesh_focus_center", type=str, default="")
    parser.add_argument("--mesh_focus_extent", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    cfg = OfflineGSBootstrapConfig(
        max_anchors=args.max_anchors,
        k_neighbors=args.k_neighbors,
        opacity_min=args.opacity_min,
        normal_axis=args.normal_axis,
        sigma_mode=args.sigma_mode,
        distance_scale=args.distance_scale,
        distance_floor=args.distance_floor,
        knn_chunk_size=args.knn_chunk_size,
        subsample_mode=args.subsample_mode,
        orient_normals_mode=args.orient_normals_mode,
        proxy_tangent_reg=args.proxy_tangent_reg,
        proxy_max_normal_shift_scale=args.proxy_max_normal_shift_scale,
        proxy_max_tangent_shift_scale=args.proxy_max_tangent_shift_scale,
        normal_support_scale=args.normal_support_scale,
        tangent_support_scale=args.tangent_support_scale,
        invalid_penalty_scale=args.invalid_penalty_scale,
    )
    adapter = OfflineGSBootstrapPseudoSDF(cfg=cfg, ply_path=args.point_cloud_ply, device=device)
    if adapter.anchor_xyz.shape[0] == 0:
        raise RuntimeError("No valid Gaussian anchors left after opacity filtering.")
    surface_xyz = adapter.proxy_xyz if adapter.proxy_xyz is not None else adapter.anchor_xyz
    surface_sigma = adapter.proxy_sigma if adapter.proxy_sigma is not None else adapter.anchor_sigma
    surface_conf = adapter.proxy_conf if adapter.proxy_conf is not None else adapter.anchor_conf

    slice_bmin, slice_bmax = _auto_bounds(surface_xyz, padding_ratio=args.slice_padding_ratio)
    if args.slice_bounds_min:
        slice_bmin = torch.tensor(_parse_triplet(args.slice_bounds_min), dtype=torch.float32, device=device)
    if args.slice_bounds_max:
        slice_bmax = torch.tensor(_parse_triplet(args.slice_bounds_max), dtype=torch.float32, device=device)

    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    slice_axis_idx = axis_to_idx[args.slice_axis]
    if args.slice_value is None:
        slice_value = float(0.5 * (slice_bmin[slice_axis_idx] + slice_bmax[slice_axis_idx]).item())
    else:
        slice_value = float(args.slice_value)

    pts, xs, ys, free_axes = _build_slice_grid(
        bmin=slice_bmin,
        bmax=slice_bmax,
        axis=args.slice_axis,
        value=slice_value,
        resolution=args.slice_resolution,
        device=device,
    )
    sdf, grad = _query_in_chunks(adapter, pts, chunk_size=args.query_chunk_size)
    sdf_grid = sdf.view(args.slice_resolution, args.slice_resolution).detach().cpu().numpy()
    grad_grid = grad.view(args.slice_resolution, args.slice_resolution, 3).detach().cpu().numpy()

    out_npz = os.path.join(args.output_dir, "pseudo_sdf_slice.npz")
    out_png = os.path.join(args.output_dir, "pseudo_sdf_slice.png")
    if args.export_slice_npz or not args.export_slice_png:
        np.savez_compressed(
            out_npz,
            sdf=sdf_grid,
            grad=grad_grid,
            xs=xs.detach().cpu().numpy(),
            ys=ys.detach().cpu().numpy(),
            slice_axis=args.slice_axis,
            slice_value=slice_value,
            free_axes=np.asarray(free_axes, dtype=np.int64),
        )
    if args.export_slice_png or not args.export_slice_npz:
        _save_slice_plot(
            out_path=out_png,
            sdf_grid=sdf_grid,
            xs=xs.detach().cpu().numpy(),
            ys=ys.detach().cpu().numpy(),
            axis=args.slice_axis,
            slice_value=slice_value,
        )

    mesh_info = None
    if args.export_mesh:
        mesh_focus_info = {
            "mode": args.mesh_focus_mode,
        }
        if args.mesh_focus_mode == "main_object":
            mesh_bmin, mesh_bmax, focus_info = _focus_bounds_main_object(
                surface_xyz=surface_xyz,
                surface_sigma=surface_sigma,
                surface_conf=surface_conf,
                cluster_ratio=args.mesh_focus_cluster_ratio,
                inlier_quantile=args.mesh_focus_inlier_quantile,
                padding_ratio=args.mesh_padding_ratio,
                seed_pool_size=args.mesh_focus_seed_pool_size,
                seed_knn=args.mesh_focus_seed_knn,
            )
            mesh_focus_info.update(focus_info)
        elif args.mesh_focus_mode == "manual":
            if not args.mesh_focus_center or not args.mesh_focus_extent:
                raise ValueError("--mesh_focus_center and --mesh_focus_extent are required for --mesh_focus_mode manual")
            mesh_bmin, mesh_bmax = _manual_bounds(
                center_text=args.mesh_focus_center,
                extent_text=args.mesh_focus_extent,
                device=device,
            )
            mesh_focus_info["focus_center"] = ((mesh_bmin + mesh_bmax) * 0.5).detach().cpu().tolist()
            mesh_focus_info["focus_extent"] = (mesh_bmax - mesh_bmin).detach().cpu().tolist()
        else:
            mesh_bmin, mesh_bmax = _auto_bounds(surface_xyz, padding_ratio=args.mesh_padding_ratio)
            mesh_focus_info["focus_center"] = ((mesh_bmin + mesh_bmax) * 0.5).detach().cpu().tolist()
            mesh_focus_info["focus_extent"] = (mesh_bmax - mesh_bmin).detach().cpu().tolist()

        mesh_path = os.path.join(args.output_dir, "pseudo_sdf_mesh.ply")
        verts, faces = _export_mesh(
            adapter=adapter,
            bmin=mesh_bmin,
            bmax=mesh_bmax,
            resolution=args.mesh_resolution,
            out_path=mesh_path,
            chunk_size=args.query_chunk_size,
        )
        mesh_info = {
            "path": mesh_path,
            "verts": verts,
            "faces": faces,
            "bounds_min": mesh_bmin.detach().cpu().tolist(),
            "bounds_max": mesh_bmax.detach().cpu().tolist(),
            "focus": mesh_focus_info,
        }
        with open(os.path.join(args.output_dir, "pseudo_sdf_mesh_bounds.json"), "w", encoding="utf-8") as f:
            json.dump(mesh_info, f, indent=2)

    print("[pseudoSDF-viz] anchors:", int(adapter.anchor_xyz.shape[0]))
    print("[pseudoSDF-viz] proxies:", int(surface_xyz.shape[0]))
    print("[pseudoSDF-viz] slice:", out_png if (args.export_slice_png or not args.export_slice_npz) else out_npz)
    if args.export_slice_npz or not args.export_slice_png:
        print("[pseudoSDF-viz] slice_npz:", out_npz)
    if mesh_info is not None:
        print(
            "[pseudoSDF-viz] mesh:",
            mesh_info["path"],
            f"verts={mesh_info['verts']}",
            f"faces={mesh_info['faces']}",
        )
        print(
            "[pseudoSDF-viz] mesh_focus:",
            mesh_info["focus"]["mode"],
            f"bmin={mesh_info['bounds_min']}",
            f"bmax={mesh_info['bounds_max']}",
        )


if __name__ == "__main__":
    main()
