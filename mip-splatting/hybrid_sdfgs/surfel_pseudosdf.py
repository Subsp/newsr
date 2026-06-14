from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData


@dataclass
class SurfelCloud:
    xyz: torch.Tensor
    normal: torch.Tensor
    t_major: torch.Tensor
    t_minor: torch.Tensor
    radius_major: torch.Tensor
    radius_minor: torch.Tensor
    thickness: torch.Tensor
    opacity: torch.Tensor
    score: torch.Tensor

    def to(self, device: torch.device):
        return SurfelCloud(
            xyz=self.xyz.to(device),
            normal=self.normal.to(device),
            t_major=self.t_major.to(device),
            t_minor=self.t_minor.to(device),
            radius_major=self.radius_major.to(device),
            radius_minor=self.radius_minor.to(device),
            thickness=self.thickness.to(device),
            opacity=self.opacity.to(device),
            score=self.score.to(device),
        )

    def index(self, mask: torch.Tensor) -> "SurfelCloud":
        return SurfelCloud(
            xyz=self.xyz[mask],
            normal=self.normal[mask],
            t_major=self.t_major[mask],
            t_minor=self.t_minor[mask],
            radius_major=self.radius_major[mask],
            radius_minor=self.radius_minor[mask],
            thickness=self.thickness[mask],
            opacity=self.opacity[mask],
            score=self.score[mask],
        )


@dataclass
class SurfelConsensusConfig:
    k_neighbors: int = 8
    distance_floor: float = 1e-4
    normal_support_scale: float = 2.0
    tangent_support_scale: float = 1.25
    score_euclid_coef: float = 0.1
    score_conf_coef: float = 0.05
    consensus_cos_thresh: float = 0.5
    invalid_penalty_scale: float = 1.5
    knn_chunk_size: int = 4096


@dataclass
class SurfelMLPConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    num_frequencies: int = 6
    use_skip: bool = True


def _parse_triplet(text: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected exactly 3 comma-separated values.")
    return vals[0], vals[1], vals[2]


def load_gaussian_ply(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
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


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
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


def build_surfel_frames(
    rotations_raw: torch.Tensor,
    scales: torch.Tensor,
    normal_axis: str = "min_scale",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rot = quaternion_to_rotation_matrix(rotations_raw)
    if normal_axis == "min_scale":
        normal_idx = torch.argmin(scales, dim=1)
    elif normal_axis == "max_scale":
        normal_idx = torch.argmax(scales, dim=1)
    else:
        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_idx = torch.full(
            (scales.shape[0],),
            int(axis_map[normal_axis]),
            dtype=torch.long,
            device=scales.device,
        )
    all_idx = torch.arange(scales.shape[0], device=scales.device)
    remaining = torch.tensor([0, 1, 2], device=scales.device)[None, :].repeat(scales.shape[0], 1)
    keep = remaining != normal_idx[:, None]
    tangent_idx = remaining[keep].view(scales.shape[0], 2)
    tangent_scale = torch.gather(scales, 1, tangent_idx)
    major_sel = tangent_scale[:, 0] >= tangent_scale[:, 1]
    major_idx = torch.where(major_sel, tangent_idx[:, 0], tangent_idx[:, 1])
    minor_idx = torch.where(major_sel, tangent_idx[:, 1], tangent_idx[:, 0])

    normal = rot[all_idx, :, normal_idx]
    t_major = rot[all_idx, :, major_idx]
    t_minor = rot[all_idx, :, minor_idx]
    normal = F.normalize(normal, dim=-1)
    t_major = F.normalize(
        t_major - (t_major * normal).sum(dim=-1, keepdim=True) * normal,
        dim=-1,
    )
    t_minor = F.normalize(torch.cross(normal, t_major, dim=-1), dim=-1)

    radius_major = torch.gather(scales, 1, major_idx[:, None])
    radius_minor = torch.gather(scales, 1, minor_idx[:, None])
    thickness = torch.gather(scales, 1, normal_idx[:, None])
    return normal, t_major, t_minor, radius_major, radius_minor, thickness


def orient_normals_toward_outside(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = (xyz * weights).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    outward = xyz - center
    outward_norm = outward.norm(dim=-1, keepdim=True)
    flip = ((normals * outward).sum(dim=-1, keepdim=True) < 0) & (outward_norm > 1e-6)
    return torch.where(flip, -normals, normals), center.squeeze(0)


def surfel_score(
    opacity: torch.Tensor,
    radius_major: torch.Tensor,
    radius_minor: torch.Tensor,
    thickness: torch.Tensor,
) -> torch.Tensor:
    sheetness = torch.sqrt((radius_major * radius_minor).clamp(min=1e-8)) / thickness.clamp(min=1e-8)
    return opacity * torch.log1p(sheetness.squeeze(-1))


def auto_bounds(points: torch.Tensor, padding_ratio: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    bmin = points.min(dim=0).values
    bmax = points.max(dim=0).values
    extent = (bmax - bmin).clamp(min=1e-4)
    pad = extent * float(padding_ratio)
    return bmin - pad, bmax + pad


def focus_bounds_main_object(
    points: torch.Tensor,
    weights: torch.Tensor,
    cluster_ratio: float = 0.5,
    inlier_quantile: float = 0.9,
    padding_ratio: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return auto_bounds(focus_pts, padding_ratio=padding_ratio)


def manual_bounds(center_text: str, extent_text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    center = torch.tensor(_parse_triplet(center_text), dtype=torch.float32, device=device)
    extent = torch.tensor(_parse_triplet(extent_text), dtype=torch.float32, device=device).abs().clamp(min=1e-5)
    half = 0.5 * extent
    return center - half, center + half


def load_surfel_cloud_from_gaussian_ply(
    path: str,
    device: torch.device,
    opacity_min: float = 0.02,
    sheetness_min: float = 2.0,
    max_surfels: int = 60000,
    normal_axis: str = "min_scale",
) -> SurfelCloud:
    raw = load_gaussian_ply(path, device=device)
    xyz = raw["xyz"]
    scales = torch.exp(raw["log_scaling"])
    rotations = raw["rotation"]
    opacity = torch.sigmoid(raw["opacity_logits"]).view(-1)
    normal, t_major, t_minor, radius_major, radius_minor, thickness = build_surfel_frames(
        rotations,
        scales,
        normal_axis=normal_axis,
    )

    sheetness = torch.sqrt((radius_major * radius_minor).clamp(min=1e-8)) / thickness.clamp(min=1e-8)
    valid = (opacity >= float(opacity_min)) & (sheetness.squeeze(-1) >= float(sheetness_min))
    if not valid.any():
        raise RuntimeError("No surfel-like Gaussians left after opacity/sheetness filtering.")

    xyz = xyz[valid]
    normal = normal[valid]
    t_major = t_major[valid]
    t_minor = t_minor[valid]
    radius_major = radius_major[valid]
    radius_minor = radius_minor[valid]
    thickness = thickness[valid]
    opacity = opacity[valid]

    score = surfel_score(opacity, radius_major, radius_minor, thickness)
    if xyz.shape[0] > int(max_surfels):
        keep_idx = torch.topk(score, k=int(max_surfels), largest=True).indices
        xyz = xyz[keep_idx]
        normal = normal[keep_idx]
        t_major = t_major[keep_idx]
        t_minor = t_minor[keep_idx]
        radius_major = radius_major[keep_idx]
        radius_minor = radius_minor[keep_idx]
        thickness = thickness[keep_idx]
        opacity = opacity[keep_idx]
        score = score[keep_idx]

    normal, _ = orient_normals_toward_outside(xyz, normal, opacity[:, None])
    return SurfelCloud(
        xyz=xyz,
        normal=normal,
        t_major=t_major,
        t_minor=t_minor,
        radius_major=radius_major,
        radius_minor=radius_minor,
        thickness=thickness,
        opacity=opacity[:, None],
        score=score[:, None],
    )


def crop_surfel_cloud(surfels: SurfelCloud, bmin: torch.Tensor, bmax: torch.Tensor) -> SurfelCloud:
    mask = ((surfels.xyz >= bmin[None, :]) & (surfels.xyz <= bmax[None, :])).all(dim=1)
    return surfels.index(mask)


class SurfelConsensusPseudoSDF:
    def __init__(self, surfels: SurfelCloud, cfg: Optional[SurfelConsensusConfig] = None):
        self.surfels = surfels
        self.cfg = cfg or SurfelConsensusConfig()

    def _knn(self, points: torch.Tensor, anchors: torch.Tensor, k: int):
        n = points.shape[0]
        k = min(k, anchors.shape[0])
        if k <= 0:
            empty_idx = torch.zeros((n, 0), dtype=torch.long, device=points.device)
            empty_dist = torch.zeros((n, 0), dtype=points.dtype, device=points.device)
            return empty_dist, empty_idx

        chunk = max(1, int(self.cfg.knn_chunk_size))
        all_dist = []
        all_idx = []
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            d = torch.cdist(points[s:e], anchors)
            d_k, i_k = torch.topk(d, k=k, dim=1, largest=False, sorted=False)
            all_dist.append(d_k)
            all_idx.append(i_k)
        return torch.cat(all_dist, dim=0), torch.cat(all_idx, dim=0)

    def _query_chunk(self, points: torch.Tensor):
        surfels = self.surfels
        if surfels.xyz.shape[0] == 0:
            zeros = torch.zeros((points.shape[0], 1), device=points.device, dtype=points.dtype)
            grad = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
            grad[:, 2] = 1.0
            return zeros, grad

        anchors = surfels.xyz.to(device=points.device, dtype=points.dtype)
        normals = surfels.normal.to(device=points.device, dtype=points.dtype)
        t_major = surfels.t_major.to(device=points.device, dtype=points.dtype)
        t_minor = surfels.t_minor.to(device=points.device, dtype=points.dtype)
        radius_major = surfels.radius_major.to(device=points.device, dtype=points.dtype).squeeze(-1)
        radius_minor = surfels.radius_minor.to(device=points.device, dtype=points.dtype).squeeze(-1)
        thickness = surfels.thickness.to(device=points.device, dtype=points.dtype).squeeze(-1)
        conf = surfels.opacity.to(device=points.device, dtype=points.dtype).squeeze(-1)

        knn_dist, knn_idx = self._knn(points, anchors, int(self.cfg.k_neighbors))
        nbr_xyz = anchors[knn_idx]
        nbr_normals = normals[knn_idx]
        nbr_t_major = t_major[knn_idx]
        nbr_t_minor = t_minor[knn_idx]
        nbr_r_major = radius_major[knn_idx].clamp(min=float(self.cfg.distance_floor))
        nbr_r_minor = radius_minor[knn_idx].clamp(min=float(self.cfg.distance_floor))
        nbr_thickness = thickness[knn_idx].clamp(min=float(self.cfg.distance_floor))
        nbr_conf = conf[knn_idx].clamp(min=0.0)

        delta = points[:, None, :] - nbr_xyz
        signed_plane = (delta * nbr_normals).sum(dim=-1)
        u = (delta * nbr_t_major).sum(dim=-1)
        v = (delta * nbr_t_minor).sum(dim=-1)
        tangent_ell = torch.sqrt((u / nbr_r_major) ** 2 + (v / nbr_r_minor) ** 2 + 1e-12)

        normal_support = (
            float(self.cfg.normal_support_scale) * nbr_thickness
        ).clamp(min=float(self.cfg.distance_floor))
        tangent_major_support = (
            float(self.cfg.tangent_support_scale) * nbr_r_major
        ).clamp(min=float(self.cfg.distance_floor))
        tangent_minor_support = (
            float(self.cfg.tangent_support_scale) * nbr_r_minor
        ).clamp(min=float(self.cfg.distance_floor))
        tangent_support_mean = 0.5 * (tangent_major_support + tangent_minor_support)

        score = (
            signed_plane.abs() / normal_support
            + torch.relu(tangent_ell - float(self.cfg.tangent_support_scale))
            + float(self.cfg.score_euclid_coef) * knn_dist / tangent_support_mean
            - float(self.cfg.score_conf_coef) * nbr_conf
        )

        best_idx = torch.argmin(score, dim=1)
        row_idx = torch.arange(points.shape[0], device=points.device)
        best_normals = nbr_normals[row_idx, best_idx]
        best_signed = signed_plane[row_idx, best_idx]
        best_tangent_ell = tangent_ell[row_idx, best_idx]
        best_normal_support = normal_support[row_idx, best_idx]
        best_tangent_support = tangent_support_mean[row_idx, best_idx]

        align_sign = torch.sign((nbr_normals * best_normals[:, None, :]).sum(dim=-1, keepdim=True))
        align_sign = torch.where(align_sign == 0, torch.ones_like(align_sign), align_sign)
        nbr_normals = nbr_normals * align_sign
        signed_plane = signed_plane * align_sign.squeeze(-1)
        align_cos = (nbr_normals * best_normals[:, None, :]).sum(dim=-1)

        weights = torch.exp(
            -0.5
            * (
                (u / tangent_major_support) ** 2
                + (v / tangent_minor_support) ** 2
                + (signed_plane / normal_support) ** 2
            )
        ) * nbr_conf
        weights = weights * (align_cos >= float(self.cfg.consensus_cos_thresh)).float()
        weight_sum = weights.sum(dim=-1, keepdim=True)
        fallback = weight_sum.squeeze(-1) <= 1e-8

        safe_weight_sum = weight_sum.clamp(min=1e-8)
        consensus_signed = (weights * signed_plane).sum(dim=-1) / safe_weight_sum.squeeze(-1)
        consensus_grad = F.normalize(
            (weights[..., None] * nbr_normals).sum(dim=1) / safe_weight_sum,
            dim=-1,
        )

        consensus_signed = torch.where(fallback, best_signed, consensus_signed)
        consensus_grad = torch.where(fallback[:, None], best_normals, consensus_grad)

        invalid_mag = float(self.cfg.invalid_penalty_scale) * (
            torch.relu(best_tangent_ell - float(self.cfg.tangent_support_scale)) * best_tangent_support
            + torch.relu(consensus_signed.abs() - best_normal_support)
        )
        sign = torch.where(consensus_signed >= 0, torch.ones_like(consensus_signed), -torch.ones_like(consensus_signed))
        sdf = (sign * (consensus_signed.abs() + invalid_mag)).unsqueeze(-1)
        grad = F.normalize(consensus_grad, dim=-1)
        return sdf, grad

    @torch.no_grad()
    def query_sdf_and_gradients(self, points: torch.Tensor, chunk_size: Optional[int] = None):
        if chunk_size is None or chunk_size <= 0:
            return self._query_chunk(points)
        sdf_parts = []
        grad_parts = []
        for s in range(0, points.shape[0], int(chunk_size)):
            e = min(points.shape[0], s + int(chunk_size))
            sdf, grad = self._query_chunk(points[s:e])
            sdf_parts.append(sdf)
            grad_parts.append(grad)
        return torch.cat(sdf_parts, dim=0), torch.cat(grad_parts, dim=0)


class FourierEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x] if self.include_input else []
        for i in range(self.num_frequencies):
            freq = (2.0**i) * math.pi
            feats.append(torch.sin(freq * x))
            feats.append(torch.cos(freq * x))
        return torch.cat(feats, dim=-1)


class SurfelPseudoSDFMLP(nn.Module):
    def __init__(self, cfg: Optional[SurfelMLPConfig] = None):
        super().__init__()
        self.cfg = cfg or SurfelMLPConfig()
        self.encoding = FourierEncoding(num_frequencies=self.cfg.num_frequencies, include_input=True)
        in_dim = self.encoding.out_dim
        hidden_dim = int(self.cfg.hidden_dim)
        self.layers = nn.ModuleList()
        last_dim = in_dim
        skip_layer = max(1, self.cfg.num_layers // 2) if self.cfg.use_skip else -1
        self.skip_layer = skip_layer
        for layer_idx in range(int(self.cfg.num_layers)):
            if layer_idx == skip_layer:
                last_dim += in_dim
            linear = nn.Linear(last_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
            last_dim = hidden_dim
        self.head = nn.Linear(last_dim, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoding(x)
        h = enc
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == self.skip_layer:
                h = torch.cat([h, enc], dim=-1)
            h = F.softplus(layer(h), beta=100.0)
        return self.head(h)


def sample_surfel_training_points(
    surfels: SurfelCloud,
    bmin: torch.Tensor,
    bmax: torch.Tensor,
    num_surface: int,
    num_near_surface: int,
    num_uniform: int,
    tangent_scale: float = 0.75,
    normal_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if surfels.xyz.shape[0] == 0:
        raise RuntimeError("Cannot sample training points from an empty surfel cloud.")

    device = surfels.xyz.device
    weights = surfels.score.squeeze(-1).clamp(min=1e-8)
    weights = weights / weights.sum()

    def _sample_ids(count: int):
        return torch.multinomial(weights, num_samples=max(1, count), replacement=True)

    out: Dict[str, torch.Tensor] = {}

    if num_surface > 0:
        ids = _sample_ids(num_surface)
        r = torch.sqrt(torch.rand((num_surface, 1), device=device, dtype=surfels.xyz.dtype))
        theta = 2.0 * math.pi * torch.rand((num_surface, 1), device=device, dtype=surfels.xyz.dtype)
        u = r * torch.cos(theta)
        v = r * torch.sin(theta)
        surface_pts = (
            surfels.xyz[ids]
            + float(tangent_scale) * u * surfels.radius_major[ids] * surfels.t_major[ids]
            + float(tangent_scale) * v * surfels.radius_minor[ids] * surfels.t_minor[ids]
        )
        out["surface"] = surface_pts

    if num_near_surface > 0:
        ids = _sample_ids(num_near_surface)
        r = torch.sqrt(torch.rand((num_near_surface, 1), device=device, dtype=surfels.xyz.dtype))
        theta = 2.0 * math.pi * torch.rand((num_near_surface, 1), device=device, dtype=surfels.xyz.dtype)
        u = r * torch.cos(theta)
        v = r * torch.sin(theta)
        normal_jitter = (
            (torch.rand((num_near_surface, 1), device=device, dtype=surfels.xyz.dtype) * 2.0 - 1.0)
            * float(normal_scale)
            * surfels.thickness[ids]
        )
        near_pts = (
            surfels.xyz[ids]
            + float(tangent_scale) * u * surfels.radius_major[ids] * surfels.t_major[ids]
            + float(tangent_scale) * v * surfels.radius_minor[ids] * surfels.t_minor[ids]
            + normal_jitter * surfels.normal[ids]
        )
        out["near_surface"] = near_pts

    if num_uniform > 0:
        uniform_pts = torch.rand((num_uniform, 3), device=device, dtype=surfels.xyz.dtype)
        uniform_pts = bmin[None, :] + uniform_pts * (bmax - bmin)[None, :]
        out["uniform"] = uniform_pts

    out["all"] = torch.cat([pts for pts in out.values()], dim=0)
    return out


def build_normalization(bmin: torch.Tensor, bmax: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center = 0.5 * (bmin + bmax)
    scale = 0.5 * (bmax - bmin).max().clamp(min=1e-6)
    return center, scale.view(1)


def save_surfel_mlp_checkpoint(
    path: str,
    model: SurfelPseudoSDFMLP,
    model_cfg: SurfelMLPConfig,
    coord_center: torch.Tensor,
    coord_scale: torch.Tensor,
    bmin: torch.Tensor,
    bmax: torch.Tensor,
    meta: Optional[Dict] = None,
):
    checkpoint = {
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "coord_center": coord_center.detach().cpu(),
        "coord_scale": coord_scale.detach().cpu(),
        "bounds_min": bmin.detach().cpu(),
        "bounds_max": bmax.detach().cpu(),
        "meta": meta or {},
    }
    torch.save(checkpoint, path)


def load_surfel_mlp_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model_cfg = SurfelMLPConfig(**checkpoint["model_cfg"])
    model = SurfelPseudoSDFMLP(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    coord_center = checkpoint["coord_center"].to(device=device, dtype=torch.float32)
    coord_scale = checkpoint["coord_scale"].to(device=device, dtype=torch.float32)
    bmin = checkpoint["bounds_min"].to(device=device, dtype=torch.float32)
    bmax = checkpoint["bounds_max"].to(device=device, dtype=torch.float32)
    meta = checkpoint.get("meta", {})
    return model, model_cfg, coord_center, coord_scale, bmin, bmax, meta
