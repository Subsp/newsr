import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from utils.general_utils import build_rotation

from .surfel_pseudosdf import load_surfel_mlp_checkpoint


class BaseSDFAdapter:
    def bind_gaussians(self, gaussians):
        return None

    def step(self, iteration: int):
        return None

    def query_sdf_and_gradients(self, points):
        raise NotImplementedError


@dataclass
class AnalyticSphereSDFAdapter(BaseSDFAdapter):
    center: torch.Tensor
    radius: float

    def query_sdf_and_gradients(self, points):
        center = self.center.to(points.device, dtype=points.dtype)
        diff = points - center[None, :]
        dist = diff.norm(dim=-1, keepdim=True)
        sdf = dist - self.radius
        grad = F.normalize(diff, dim=-1)
        return sdf, grad


class NeuralangeloSDFAdapter(BaseSDFAdapter):
    def __init__(self, neuralangelo_root, config_path, checkpoint_path, device="cuda"):
        self.device = torch.device(device)

        if not os.path.exists(neuralangelo_root):
            raise FileNotFoundError(f"Neuralangelo root not found: {neuralangelo_root}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Neuralangelo config not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Neuralangelo checkpoint not found: {checkpoint_path}")

        if neuralangelo_root not in sys.path:
            sys.path.insert(0, neuralangelo_root)

        from imaginaire.config import Config
        from projects.neuralangelo.model import Model as NeuralangeloModel

        cfg = Config(config_path)
        self.model = NeuralangeloModel(cfg.model, cfg.data).to(self.device)
        self.model.eval()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value

        self.model.load_state_dict(cleaned, strict=False)

        sdf_net = self.model.neural_sdf
        if sdf_net.cfg_sdf.encoding.coarse2fine.enabled:
            sdf_net.warm_up_end = 0
            sdf_net.set_active_levels(current_iter=10**9)
        if sdf_net.cfg_sdf.gradient.mode == "numerical":
            sdf_net.set_normal_epsilon()

    @torch.no_grad()
    def query_sdf_and_gradients(self, points):
        points = points.to(self.device)
        sdf = self.model.neural_sdf.sdf(points)
        grads, _ = self.model.neural_sdf.compute_gradients(points, training=False, sdf=None)
        return sdf, grads


@dataclass
class GSBootstrapConfig:
    update_interval: int = 500
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


class GSBootstrapSDFAdapter(BaseSDFAdapter):
    """A lightweight SDF teacher bootstrapped from current GS primitives.

    The adapter converts Gaussians to local oriented plane priors and performs
    weighted KNN blending to provide pseudo-SDF values and gradients.
    """

    def __init__(self, cfg: GSBootstrapConfig):
        self.cfg = cfg
        self._gaussians = None
        self.anchor_xyz: Optional[torch.Tensor] = None
        self.anchor_normals: Optional[torch.Tensor] = None
        self.anchor_sigma: Optional[torch.Tensor] = None
        self.anchor_tangent_sigma: Optional[torch.Tensor] = None
        self.anchor_conf: Optional[torch.Tensor] = None
        self.proxy_xyz: Optional[torch.Tensor] = None
        self.proxy_normals: Optional[torch.Tensor] = None
        self.proxy_sigma: Optional[torch.Tensor] = None
        self.proxy_tangent_sigma: Optional[torch.Tensor] = None
        self.proxy_conf: Optional[torch.Tensor] = None
        self.last_refresh_iter = -1

    @staticmethod
    def _gaussian_normals(rotations_raw: torch.Tensor, scales: torch.Tensor, axis: str):
        rotations = build_rotation(rotations_raw)
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

    def bind_gaussians(self, gaussians):
        self._gaussians = gaussians
        self.refresh_from_gaussians()

    @staticmethod
    def _anchor_sigma_from_scales(scales: torch.Tensor, mode: str):
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

    @staticmethod
    def _anchor_tangent_sigma_from_scales(scales: torch.Tensor):
        scales_sorted, _ = torch.sort(scales, dim=1)
        s_mid = scales_sorted[:, 1:2]
        s_max = scales_sorted[:, 2:3]
        return torch.sqrt((s_mid * s_max).clamp(min=1e-12))

    @staticmethod
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

    @staticmethod
    def _orient_normals(xyz: torch.Tensor, normals: torch.Tensor, conf: torch.Tensor, mode: str):
        if mode == "none" or xyz.shape[0] == 0:
            return normals
        weights = conf.clamp(min=1e-6)
        center = (xyz * weights).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
        outward = xyz - center
        outward_norm = outward.norm(dim=-1, keepdim=True)
        flip = ((normals * outward).sum(dim=-1, keepdim=True) < 0) & (outward_norm > 1e-6)
        return torch.where(flip, -normals, normals)

    @torch.no_grad()
    def refresh_from_gaussians(self):
        if self._gaussians is None:
            return
        xyz = self._gaussians.get_xyz.detach()
        scales = self._gaussians.get_scaling.detach()
        rotations_raw = self._gaussians._rotation.detach()
        opacity = self._gaussians.get_opacity.detach().view(-1)

        valid = opacity >= float(self.cfg.opacity_min)
        if valid.any():
            xyz = xyz[valid]
            scales = scales[valid]
            rotations_raw = rotations_raw[valid]
            opacity = opacity[valid]

        if xyz.numel() == 0:
            # Keep empty anchors; caller will handle fallback.
            self.anchor_xyz = xyz
            self.anchor_normals = xyz.new_zeros((0, 3))
            self.anchor_sigma = xyz.new_zeros((0, 1))
            self.anchor_tangent_sigma = xyz.new_zeros((0, 1))
            self.anchor_conf = xyz.new_zeros((0, 1))
            self.proxy_xyz = xyz
            self.proxy_normals = xyz.new_zeros((0, 3))
            self.proxy_sigma = xyz.new_zeros((0, 1))
            self.proxy_tangent_sigma = xyz.new_zeros((0, 1))
            self.proxy_conf = xyz.new_zeros((0, 1))
            return

        normals = self._gaussian_normals(rotations_raw, scales, self.cfg.normal_axis)
        normal_sigma = self._anchor_sigma_from_scales(scales, self.cfg.sigma_mode)
        xyz, scales, rotations_raw, opacity, normal_sigma = self._select_anchor_subset(
            xyz=xyz,
            scales=scales,
            rotations_raw=rotations_raw,
            opacity=opacity,
            normal_sigma=normal_sigma,
            max_anchors=int(self.cfg.max_anchors),
            mode=self.cfg.subsample_mode,
        )
        normals = self._gaussian_normals(rotations_raw, scales, self.cfg.normal_axis)
        normals = self._orient_normals(
            xyz=xyz,
            normals=normals,
            conf=opacity[:, None],
            mode=self.cfg.orient_normals_mode,
        )
        normal_sigma = (normal_sigma * float(self.cfg.distance_scale)).clamp(min=float(self.cfg.distance_floor))
        tangent_sigma = self._anchor_tangent_sigma_from_scales(scales)
        tangent_sigma = (tangent_sigma * float(self.cfg.distance_scale)).clamp(min=float(self.cfg.distance_floor))

        self.anchor_xyz = xyz
        self.anchor_normals = normals
        self.anchor_sigma = normal_sigma
        self.anchor_tangent_sigma = tangent_sigma
        self.anchor_conf = opacity[:, None].clamp(min=0.0, max=1.0)
        self._refresh_surface_proxies()

    def step(self, iteration: int):
        interval = int(self.cfg.update_interval)
        if interval <= 0:
            return
        if self._gaussians is None:
            return
        if self.last_refresh_iter < 0 or iteration % interval == 0:
            self.refresh_from_gaussians()
            self.last_refresh_iter = iteration

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

    @torch.no_grad()
    def _refresh_surface_proxies(self):
        if self.anchor_xyz is None or self.anchor_xyz.shape[0] == 0:
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
        knn_dist, knn_idx = self._knn(xyz, xyz, k)
        nbr_xyz = xyz[knn_idx]
        nbr_normals = normals[knn_idx]
        nbr_normal_sigma = normal_sigma[knn_idx].clamp(min=float(self.cfg.distance_floor))
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

    def query_sdf_and_gradients(self, points):
        if self.proxy_xyz is None or self.proxy_xyz.shape[0] == 0:
            zeros = torch.zeros((points.shape[0], 1), device=points.device, dtype=points.dtype)
            default_grad = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
            default_grad[:, 2] = 1.0
            return zeros, default_grad

        proxies = self.proxy_xyz.to(device=points.device, dtype=points.dtype)
        normals = self.proxy_normals.to(device=points.device, dtype=points.dtype)
        sigma = self.proxy_sigma.to(device=points.device, dtype=points.dtype).squeeze(-1)
        tangent_sigma = self.proxy_tangent_sigma.to(device=points.device, dtype=points.dtype).squeeze(-1)
        conf = self.proxy_conf.to(device=points.device, dtype=points.dtype).squeeze(-1)

        k = int(self.cfg.k_neighbors)
        knn_dist, knn_idx = self._knn(points, proxies, k)
        nbr_xyz = proxies[knn_idx]
        nbr_normals = normals[knn_idx]
        nbr_sigma = sigma[knn_idx].clamp(min=float(self.cfg.distance_floor))
        nbr_tangent_sigma = tangent_sigma[knn_idx].clamp(min=float(self.cfg.distance_floor))
        nbr_conf = conf[knn_idx].clamp(min=0.0)

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


class SurfelMLPSDFAdapter(BaseSDFAdapter):
    def __init__(self, checkpoint_path: str, device: str = "cuda", query_chunk_size: int = 8192):
        self.device = torch.device(device)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Surfel MLP checkpoint not found: {checkpoint_path}")
        (
            self.model,
            self.model_cfg,
            self.coord_center,
            self.coord_scale,
            self.bounds_min,
            self.bounds_max,
            self.meta,
        ) = load_surfel_mlp_checkpoint(checkpoint_path, device=self.device)
        self.query_chunk_size = int(query_chunk_size)

    def query_sdf_and_gradients(self, points):
        sdf_chunks = []
        grad_chunks = []
        chunk = max(1, self.query_chunk_size)
        for s in range(0, points.shape[0], chunk):
            e = min(points.shape[0], s + chunk)
            pts = points[s:e].to(device=self.device, dtype=torch.float32).detach().requires_grad_(True)
            pts_norm = (pts - self.coord_center[None, :]) / self.coord_scale
            sdf = self.model(pts_norm)
            grad = torch.autograd.grad(
                sdf,
                pts,
                grad_outputs=torch.ones_like(sdf),
                create_graph=False,
                retain_graph=False,
            )[0]
            sdf_chunks.append(sdf.to(device=points.device, dtype=points.dtype))
            grad_chunks.append(grad.to(device=points.device, dtype=points.dtype))
        return torch.cat(sdf_chunks, dim=0), torch.cat(grad_chunks, dim=0)


def _parse_triplet(text):
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected exactly 3 values for sdf_center, e.g. '0,0,0'")
    return vals


def build_sdf_adapter(args):
    mode = args.sdf_mode
    if mode == "none":
        return None

    if mode == "analytic":
        center_vals = _parse_triplet(args.sdf_center)
        center = torch.tensor(center_vals, dtype=torch.float32)
        return AnalyticSphereSDFAdapter(center=center, radius=float(args.sdf_radius))

    if mode == "neuralangelo":
        return NeuralangeloSDFAdapter(
            neuralangelo_root=args.neuralangelo_root,
            config_path=args.sdf_config,
            checkpoint_path=args.sdf_checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    if mode == "gs_bootstrap":
        return GSBootstrapSDFAdapter(
            GSBootstrapConfig(
                update_interval=args.gs_bootstrap_update_interval,
                max_anchors=args.gs_bootstrap_max_anchors,
                k_neighbors=args.gs_bootstrap_k_neighbors,
                opacity_min=args.gs_bootstrap_opacity_min,
                normal_axis=args.gs_bootstrap_normal_axis,
                sigma_mode=args.gs_bootstrap_sigma_mode,
                distance_scale=args.gs_bootstrap_distance_scale,
                distance_floor=args.gs_bootstrap_distance_floor,
                knn_chunk_size=args.gs_bootstrap_knn_chunk_size,
                subsample_mode=args.gs_bootstrap_subsample_mode,
                orient_normals_mode=args.gs_bootstrap_orient_normals_mode,
                proxy_tangent_reg=args.gs_bootstrap_proxy_tangent_reg,
                proxy_max_normal_shift_scale=args.gs_bootstrap_proxy_max_normal_shift_scale,
                proxy_max_tangent_shift_scale=args.gs_bootstrap_proxy_max_tangent_shift_scale,
                normal_support_scale=args.gs_bootstrap_normal_support_scale,
                tangent_support_scale=args.gs_bootstrap_tangent_support_scale,
                invalid_penalty_scale=args.gs_bootstrap_invalid_penalty_scale,
            )
        )

    if mode == "surfel_mlp":
        return SurfelMLPSDFAdapter(
            checkpoint_path=args.sdf_checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    raise ValueError(f"Unknown sdf_mode: {mode}")
