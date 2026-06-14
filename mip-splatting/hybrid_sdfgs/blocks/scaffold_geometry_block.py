from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation


@dataclass
class ScaffoldGeometryConfig:
    sample_size: int = 2048
    interval: int = 1
    axis: str = "min_scale"


class ScaffoldGeometryBlock:
    """Geometry prior regularization from feed-forward scaffold points/normals."""

    def __init__(
        self,
        cfg: ScaffoldGeometryConfig,
        scaffold_points_cpu: torch.Tensor,
        scaffold_normals_cpu: torch.Tensor | None = None,
    ):
        self.cfg = cfg
        self._points_cpu = scaffold_points_cpu.detach().float().cpu().contiguous()
        self._normals_cpu = (
            None
            if scaffold_normals_cpu is None
            else F.normalize(scaffold_normals_cpu.detach().float().cpu().contiguous(), dim=-1)
        )
        self._points_device_cache = {}
        self._normals_device_cache = {}

    def _points_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._points_device_cache.get(key)
        if cached is None:
            cached = self._points_cpu.to(device=device, dtype=dtype)
            self._points_device_cache[key] = cached
        return cached

    def _normals_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self._normals_cpu is None:
            return None
        key = (str(device), str(dtype))
        cached = self._normals_device_cache.get(key)
        if cached is None:
            cached = self._normals_cpu.to(device=device, dtype=dtype)
            self._normals_device_cache[key] = cached
        return cached

    @staticmethod
    def _gaussian_normals(
        rotations_raw: torch.Tensor,
        scales: torch.Tensor,
        axis: str,
    ) -> torch.Tensor:
        rotations = build_rotation(rotations_raw)

        if axis == "min_scale":
            idx = torch.argmin(scales, dim=1)
        elif axis == "max_scale":
            idx = torch.argmax(scales, dim=1)
        elif axis == "x":
            idx = torch.zeros((scales.shape[0],), device=scales.device, dtype=torch.long)
        elif axis == "y":
            idx = torch.ones((scales.shape[0],), device=scales.device, dtype=torch.long)
        elif axis == "z":
            idx = torch.full((scales.shape[0],), 2, device=scales.device, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported axis mode: {axis}")

        batch_idx = torch.arange(rotations.shape[0], device=rotations.device)
        normals = rotations[batch_idx, :, idx]
        return F.normalize(normals, dim=-1)

    @staticmethod
    def _sample_rows(data: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(data.shape[0])
        if n <= 0:
            raise ValueError("Cannot sample from empty tensor.")
        k = min(max(1, int(count)), n)
        idx = torch.randint(0, n, (k,), device=data.device)
        return data[idx], idx

    def compute(
        self,
        xyz_all: torch.Tensor,
        rotations_raw_all: torch.Tensor,
        scales_all: torch.Tensor,
        iteration: int | None = None,
    ):
        zero = torch.zeros((), device=xyz_all.device, dtype=xyz_all.dtype)
        interval = max(1, int(self.cfg.interval))
        if iteration is not None and (iteration % interval != 0):
            return zero, zero, {"active": 0.0, "selected_gs": 0.0, "selected_scaffold": 0.0}

        points = self._points_on(xyz_all.device, xyz_all.dtype)
        normals = self._normals_on(xyz_all.device, xyz_all.dtype)
        if points.numel() == 0 or xyz_all.numel() == 0:
            return zero, zero, {"active": 0.0, "selected_gs": 0.0, "selected_scaffold": 0.0}

        gs_xyz, gs_idx = self._sample_rows(xyz_all, self.cfg.sample_size)
        scaffold_xyz, scaffold_idx = self._sample_rows(points, self.cfg.sample_size)
        dist = torch.cdist(gs_xyz, scaffold_xyz, p=2.0)

        g2s_dist, g2s_idx = dist.min(dim=1)
        s2g_dist, _ = dist.min(dim=0)
        chamfer = g2s_dist.mean() + s2g_dist.mean()

        normal_loss = zero
        if normals is not None:
            gs_normals = self._gaussian_normals(
                rotations_raw=rotations_raw_all[gs_idx],
                scales=scales_all[gs_idx],
                axis=self.cfg.axis,
            )
            scaffold_normals = normals[scaffold_idx][g2s_idx]
            dot = torch.sum(gs_normals * scaffold_normals, dim=-1).abs()
            normal_loss = (1.0 - dot).mean()

        metrics = {
            "active": 1.0,
            "selected_gs": float(gs_xyz.shape[0]),
            "selected_scaffold": float(scaffold_xyz.shape[0]),
            "chamfer": float(chamfer.detach().item()),
            "normal": float(normal_loss.detach().item()) if normals is not None else 0.0,
            "has_normals": 1.0 if normals is not None else 0.0,
        }
        return chamfer, normal_loss, metrics
