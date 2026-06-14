from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


@dataclass
class VisibilityRecordConfig:
    downsample: int = 8
    topk: int = 4
    max_visible_per_view: int = 30000
    min_opacity: float = 0.02
    min_depth: float = 0.05
    max_patch_radius: int = 1


def _insert_topk(ids_cell: np.ndarray, weight_cell: np.ndarray, gid: int, score: float) -> None:
    replace_idx = int(np.argmin(weight_cell))
    if score <= float(weight_cell[replace_idx]):
        return
    ids_cell[replace_idx] = int(gid)
    weight_cell[replace_idx] = float(score)


def _project_gaussians_to_camera(
    xyz: torch.Tensor,
    camera,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = xyz.device
    dtype = xyz.dtype
    R = torch.as_tensor(camera.R, device=device, dtype=dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=dtype)
    xyz_cam = xyz @ R + T[None, :]
    z = xyz_cam[:, 2].clamp_min(1e-6)
    u = xyz_cam[:, 0] / z * float(camera.focal_x) + float(camera.image_width) / 2.0
    v = xyz_cam[:, 1] / z * float(camera.focal_y) + float(camera.image_height) / 2.0
    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), z.detach().cpu().numpy()


def build_coarse_visibility_records(
    gaussians,
    cameras: Sequence[object],
    render_pkgs: Sequence[Dict[str, torch.Tensor]],
    *,
    image_hw: Tuple[int, int],
    cfg: VisibilityRecordConfig | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or VisibilityRecordConfig()
    if len(cameras) != len(render_pkgs):
        raise ValueError(f"Camera/render_pkg length mismatch: {len(cameras)} vs {len(render_pkgs)}")

    height, width = image_hw
    coarse_h = max(int(height) // int(cfg.downsample), 1)
    coarse_w = max(int(width) // int(cfg.downsample), 1)
    num_views = len(cameras)
    topk = int(cfg.topk)

    gaussian_ids = np.full((1, num_views, coarse_h, coarse_w, topk), -1, dtype=np.int64)
    contribution_weights = np.zeros((1, num_views, coarse_h, coarse_w, topk), dtype=np.float32)

    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy()

    for view_idx, (camera, render_pkg) in enumerate(zip(cameras, render_pkgs)):
        vis_filter = render_pkg["visibility_filter"].detach().cpu().numpy().astype(bool)
        radii = render_pkg["radii"].detach().cpu().numpy()
        u, v, z = _project_gaussians_to_camera(xyz, camera)

        valid = (
            vis_filter
            & np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(z)
            & (opacity >= float(cfg.min_opacity))
            & (z >= float(cfg.min_depth))
            & (u >= 0.0)
            & (u < float(camera.image_width))
            & (v >= 0.0)
            & (v < float(camera.image_height))
        )
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            continue

        scores = opacity[valid_ids] * np.maximum(radii[valid_ids], 0.0) / np.maximum(z[valid_ids], 1e-6)
        if int(cfg.max_visible_per_view) > 0 and valid_ids.size > int(cfg.max_visible_per_view):
            top_local = np.argpartition(scores, -int(cfg.max_visible_per_view))[-int(cfg.max_visible_per_view) :]
            valid_ids = valid_ids[top_local]
            scores = scores[top_local]

        order = np.argsort(scores)[::-1]
        valid_ids = valid_ids[order]
        scores = scores[order]

        for gid, score in zip(valid_ids.tolist(), scores.tolist()):
            cx = int(np.clip(np.rint(u[gid] / float(cfg.downsample) - 0.5), 0, coarse_w - 1))
            cy = int(np.clip(np.rint(v[gid] / float(cfg.downsample) - 0.5), 0, coarse_h - 1))
            coarse_radius = int(
                np.clip(
                    np.rint(max(float(radii[gid]), 1.0) / float(cfg.downsample)),
                    0,
                    int(cfg.max_patch_radius),
                )
            )
            for oy in range(-coarse_radius, coarse_radius + 1):
                py = cy + oy
                if py < 0 or py >= coarse_h:
                    continue
                for ox in range(-coarse_radius, coarse_radius + 1):
                    px = cx + ox
                    if px < 0 or px >= coarse_w:
                        continue
                    local_score = float(score) / float(1 + ox * ox + oy * oy)
                    _insert_topk(
                        gaussian_ids[0, view_idx, py, px],
                        contribution_weights[0, view_idx, py, px],
                        gid,
                        local_score,
                    )

        weights = contribution_weights[0, view_idx]
        denom = np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
        mask = gaussian_ids[0, view_idx] >= 0
        contribution_weights[0, view_idx] = np.where(mask, weights / denom, 0.0)

    return {
        "gaussian_ids": torch.from_numpy(gaussian_ids),
        "weights": torch.from_numpy(contribution_weights),
        "coarse_hw": torch.tensor([coarse_h, coarse_w], dtype=torch.int32),
    }
