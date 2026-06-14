import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SNAPSHOT_SCHEMA_VERSION = "training_diagnostics_snapshot_v0"


def compute_training_phase(iteration: int, opt, mesh) -> str:
    if iteration < int(opt.densify_from_iter):
        densify_stage = "pre_densify"
    elif iteration < int(opt.densify_until_iter):
        densify_stage = "densify"
    else:
        densify_stage = "post_densify"

    reg_tokens = []
    if iteration >= int(mesh.distortion_from_iter):
        reg_tokens.append("dist")
        if float(mesh.lambda_opacity_field) > 0.0:
            reg_tokens.append("opa")
        if float(mesh.lambda_extent) > 0.0:
            reg_tokens.append("ext")
    if iteration >= int(mesh.depth_normal_from_iter):
        reg_tokens.append("dn")
        if float(mesh.lambda_smoothness) > 0.0:
            reg_tokens.append("smooth")
    reg_stage = "+".join(reg_tokens) if reg_tokens else "rgb"
    return f"{densify_stage}|{reg_stage}"


def resolve_diagnostic_start_iter(requested_start: int, opt, mesh) -> int:
    if int(requested_start) >= 0:
        return int(requested_start)
    return max(
        int(opt.densify_until_iter),
        int(mesh.distortion_from_iter),
        int(mesh.depth_normal_from_iter),
    )


def project_points_camera_torch(viewpoint_cam, xyz_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if xyz_world.numel() == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=xyz_world.device)
        return empty, torch.empty((0,), dtype=torch.bool, device=xyz_world.device)
    R = torch.as_tensor(viewpoint_cam.R, device=xyz_world.device, dtype=xyz_world.dtype)
    T = torch.as_tensor(viewpoint_cam.T, device=xyz_world.device, dtype=xyz_world.dtype)
    xyz_cam = xyz_world @ R + T.unsqueeze(0)
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / torch.clamp_min(z, 1e-6) * float(viewpoint_cam.focal_x) + float(viewpoint_cam.image_width) / 2.0
    y = xyz_cam[:, 1] / torch.clamp_min(z, 1e-6) * float(viewpoint_cam.focal_y) + float(viewpoint_cam.image_height) / 2.0
    valid = (
        (z > 1e-6)
        & (x >= 0.0)
        & (x < float(viewpoint_cam.image_width))
        & (y >= 0.0)
        & (y < float(viewpoint_cam.image_height))
    )
    return torch.stack([x, y], dim=1).to(dtype=torch.float32), valid


def chw_to_hwc_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return image.astype(np.float32, copy=False)


def save_rgb_image(path: Path, image_chw: torch.Tensor):
    array = np.clip(chw_to_hwc_numpy(image_chw) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def build_rotation_matrices(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(quaternion, dim=1, keepdim=True).clamp_min(1e-8)
    q = quaternion / norm
    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def expand_tile_grid(grid: np.ndarray, height: int, width: int, tile_size: int) -> np.ndarray:
    expanded = np.repeat(np.repeat(grid, int(tile_size), axis=0), int(tile_size), axis=1)
    return expanded[:height, :width]


def aggregate_points_to_tiles(
    projected_xy: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    tiles_x = int(math.ceil(float(width) / float(tile_size)))
    tiles_y = int(math.ceil(float(height) / float(tile_size)))
    tile_sum = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    tile_count = np.zeros((tiles_y, tiles_x), dtype=np.int32)

    if projected_xy.size > 0:
        px = np.floor(projected_xy[:, 0] / float(tile_size)).astype(np.int64)
        py = np.floor(projected_xy[:, 1] / float(tile_size)).astype(np.int64)
        valid = (
            np.isfinite(projected_xy[:, 0])
            & np.isfinite(projected_xy[:, 1])
            & np.isfinite(values)
            & (px >= 0)
            & (px < tiles_x)
            & (py >= 0)
            & (py < tiles_y)
        )
        if np.any(valid):
            np.add.at(tile_sum, (py[valid], px[valid]), values[valid].astype(np.float32, copy=False))
            np.add.at(tile_count, (py[valid], px[valid]), 1)

    tile_mean = np.full((tiles_y, tiles_x), np.nan, dtype=np.float32)
    nonzero = tile_count > 0
    tile_mean[nonzero] = tile_sum[nonzero] / tile_count[nonzero].astype(np.float32)
    return tile_mean, expand_tile_grid(tile_mean, height=height, width=width, tile_size=tile_size)


def scalar_summary(values: np.ndarray, key_prefix: str) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{key_prefix}_count": 0,
            f"{key_prefix}_mean": 0.0,
            f"{key_prefix}_median": 0.0,
            f"{key_prefix}_p90": 0.0,
        }
    return {
        f"{key_prefix}_count": int(finite.size),
        f"{key_prefix}_mean": float(np.mean(finite)),
        f"{key_prefix}_median": float(np.median(finite)),
        f"{key_prefix}_p90": float(np.percentile(finite, 90.0)),
    }


def _overlay_metric(
    base_image: np.ndarray,
    metric_map: np.ndarray,
    path: Path,
    cmap_name: str,
    signed: bool,
    alpha: float,
):
    base = np.clip(base_image, 0.0, 1.0).astype(np.float32, copy=False)
    overlay = base.copy()
    metric = np.asarray(metric_map, dtype=np.float32)
    finite = np.isfinite(metric)
    if not np.any(finite):
        Image.fromarray(np.clip(base * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)
        return

    values = metric[finite]
    if signed:
        vmax = float(np.percentile(np.abs(values), 95.0))
        vmax = max(vmax, 1e-6)
        normalized = np.clip(metric / vmax, -1.0, 1.0)
        lookup = (normalized + 1.0) * 0.5
        alpha_weight = np.clip(np.abs(metric) / vmax, 0.0, 1.0)
    else:
        vmin = float(np.percentile(values, 5.0))
        vmax = float(np.percentile(values, 95.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        lookup = np.clip((metric - vmin) / (vmax - vmin), 0.0, 1.0)
        alpha_weight = np.clip(lookup, 0.0, 1.0)

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colors = cmap(np.nan_to_num(lookup, nan=0.0))[..., :3].astype(np.float32)
    local_alpha = (float(alpha) * alpha_weight)[..., None]
    overlay[finite] = (1.0 - local_alpha[finite]) * overlay[finite] + local_alpha[finite] * colors[finite]
    Image.fromarray(np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def save_metric_overlay(base_image: np.ndarray, metric_map: np.ndarray, path: Path, signed: bool = False, alpha: float = 0.7):
    cmap_name = "coolwarm" if signed else "magma"
    _overlay_metric(base_image, metric_map, path, cmap_name=cmap_name, signed=signed, alpha=alpha)


def make_contact_sheet(image_paths: List[Path], output_path: Path, columns: int = 2):
    existing = [path for path in image_paths if path.exists()]
    if not existing:
        return
    images = [Image.open(path).convert("RGB") for path in existing]
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    columns = max(1, int(columns))
    rows = int(math.ceil(len(images) / float(columns)))
    cell_w = max(widths)
    cell_h = max(heights)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (24, 24, 24))
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = col * cell_w
        y = row * cell_h
        sheet.paste(image, (x, y))
    sheet.save(output_path)


@dataclass
class DiagnosticRuntimeSummary:
    iteration: int
    phase_name: str
    camera_name: str
    bundle_dir: Path
    summary: Dict[str, float]


class DiagnosticBasisProvider:
    def __init__(self, basis_mode: str = "gaussian_frame", surface_payload_path: Optional[str] = None):
        self.basis_mode = str(basis_mode)
        self.surface_normals = None
        if surface_payload_path:
            payload = torch.load(surface_payload_path, map_location="cpu")
            normals = payload.get("nearest_surface_normal", None)
            if normals is None:
                raise KeyError(f"Surface payload is missing 'nearest_surface_normal': {surface_payload_path}")
            self.surface_normals = torch.as_tensor(normals, dtype=torch.float32)

    def _frame_from_gaussians(self, gaussians) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scales = gaussians.get_scaling.detach()
        rotations = build_rotation_matrices(gaussians.get_rotation.detach())
        sorted_axes = torch.argsort(scales, dim=1)
        axis_n = sorted_axes[:, 0]
        axis_u = sorted_axes[:, 1]
        axis_v = sorted_axes[:, 2]

        def gather_axis(axis_ids: torch.Tensor) -> torch.Tensor:
            gather_idx = axis_ids[:, None, None].expand(-1, 3, 1)
            return torch.gather(rotations, 2, gather_idx).squeeze(-1)

        normal = F.normalize(gather_axis(axis_n), dim=1)
        tangent_u = F.normalize(gather_axis(axis_u), dim=1)
        tangent_v = F.normalize(gather_axis(axis_v), dim=1)
        return normal, tangent_u, tangent_v

    def get_basis(self, gaussians) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal, tangent_u, tangent_v = self._frame_from_gaussians(gaussians)
        if self.basis_mode != "surface_payload" or self.surface_normals is None:
            return normal, tangent_u, tangent_v

        if int(self.surface_normals.shape[0]) != int(gaussians.get_xyz.shape[0]):
            raise ValueError(
                "Surface payload length does not match current Gaussian count: "
                f"{int(self.surface_normals.shape[0])} vs {int(gaussians.get_xyz.shape[0])}"
            )
        normal = self.surface_normals.to(device=gaussians.get_xyz.device, dtype=torch.float32)
        normal = F.normalize(normal, dim=1)
        tangent_u = tangent_u - torch.sum(tangent_u * normal, dim=1, keepdim=True) * normal
        tangent_u_norm = torch.linalg.norm(tangent_u, dim=1, keepdim=True)
        fallback = tangent_u_norm.squeeze(1) <= 1e-6
        tangent_u = tangent_u / torch.clamp_min(tangent_u_norm, 1e-6)
        tangent_u[fallback] = tangent_v[fallback]
        tangent_v = F.normalize(torch.cross(normal, tangent_u, dim=1), dim=1)
        tangent_u = F.normalize(torch.cross(tangent_v, normal, dim=1), dim=1)
        return normal, tangent_u, tangent_v

    @staticmethod
    def project_gradients(
        grad_xyz: torch.Tensor,
        normal: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_n = torch.sum(grad_xyz * normal, dim=1)
        grad_u = torch.sum(grad_xyz * tangent_u, dim=1)
        grad_v = torch.sum(grad_xyz * tangent_v, dim=1)
        grad_t = torch.sqrt(grad_u.pow(2) + grad_v.pow(2) + 1e-12)
        return grad_n, grad_u, grad_v, grad_t


class GaussianGradientTracker:
    def __init__(
        self,
        enabled: bool,
        start_iter: int,
        snapshot_interval: int,
        tile_size: int,
        basis_provider: DiagnosticBasisProvider,
        reset_on_phase_change: bool = True,
        min_significant_grad: float = 1e-8,
    ):
        self.enabled = bool(enabled)
        self.start_iter = int(start_iter)
        self.snapshot_interval = int(snapshot_interval)
        self.tile_size = int(tile_size)
        self.basis_provider = basis_provider
        self.reset_on_phase_change = bool(reset_on_phase_change)
        self.min_significant_grad = float(min_significant_grad)

        self.phase_name = None
        self.sample_count = None
        self.visibility_count = None
        self.sum_signed = None
        self.sum_abs = None
        self.sum_sq = None
        self.sum_tangent = None
        self.flip_count = None
        self.max_abs = None
        self.prev_sign = None

    def _reset(self, gaussian_count: int, phase_name: str, device: torch.device):
        self.phase_name = str(phase_name)
        self.sample_count = torch.zeros((gaussian_count,), dtype=torch.int32, device=device)
        self.visibility_count = torch.zeros((gaussian_count,), dtype=torch.int32, device=device)
        self.sum_signed = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        self.sum_abs = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        self.sum_sq = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        self.sum_tangent = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        self.flip_count = torch.zeros((gaussian_count,), dtype=torch.int32, device=device)
        self.max_abs = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        self.prev_sign = torch.zeros((gaussian_count,), dtype=torch.int8, device=device)

    def _ensure_state(self, gaussian_count: int, phase_name: str, device: torch.device):
        need_reset = self.sample_count is None or int(self.sample_count.shape[0]) != int(gaussian_count)
        if self.reset_on_phase_change and self.phase_name is not None and self.phase_name != str(phase_name):
            need_reset = True
        if need_reset:
            self._reset(gaussian_count=gaussian_count, phase_name=phase_name, device=device)

    def _build_snapshot(
        self,
        iteration: int,
        phase_name: str,
        viewpoint_cam,
        gaussians,
        gt_image: torch.Tensor,
        render_image: torch.Tensor,
        valid: torch.Tensor,
        grad_n: torch.Tensor,
        grad_t: torch.Tensor,
    ) -> Optional[Dict[str, object]]:
        valid_ids = valid.nonzero(as_tuple=True)[0]
        if valid_ids.numel() == 0:
            return None

        xyz = gaussians.get_xyz.detach()[valid_ids]
        projected_xy_t, projected_valid_t = project_points_camera_torch(viewpoint_cam, xyz)
        if not torch.any(projected_valid_t):
            return None
        keep = projected_valid_t
        valid_ids = valid_ids[keep]
        projected_xy = projected_xy_t[keep].cpu().numpy().astype(np.float32, copy=False)

        current_grad_n = grad_n[valid_ids].detach().cpu().numpy().astype(np.float32, copy=False)
        current_grad_t = grad_t[valid_ids].detach().cpu().numpy().astype(np.float32, copy=False)
        count = self.sample_count[valid_ids].to(dtype=torch.float32)
        mean_signed = (self.sum_signed[valid_ids] / torch.clamp_min(count, 1.0)).detach().cpu().numpy().astype(np.float32, copy=False)
        mean_abs = (self.sum_abs[valid_ids] / torch.clamp_min(count, 1.0)).detach().cpu().numpy().astype(np.float32, copy=False)
        std_signed = torch.sqrt(
            torch.clamp_min(self.sum_sq[valid_ids] / torch.clamp_min(count, 1.0) - (self.sum_signed[valid_ids] / torch.clamp_min(count, 1.0)).pow(2), 0.0)
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        flip_rate = (
            self.flip_count[valid_ids].to(dtype=torch.float32) / torch.clamp_min(count - 1.0, 1.0)
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        dominance = (
            self.max_abs[valid_ids] / torch.clamp_min(self.sum_abs[valid_ids], 1e-8)
        ).detach().cpu().numpy().astype(np.float32, copy=False)

        width = int(viewpoint_cam.image_width)
        height = int(viewpoint_cam.image_height)
        maps = {}
        for name, values in {
            "current_signed_normal": current_grad_n,
            "current_abs_normal": np.abs(current_grad_n),
            "current_tangent": current_grad_t,
            "running_bias": mean_signed,
            "running_abs": mean_abs,
            "running_jitter": std_signed,
            "running_flip_rate": flip_rate,
            "running_dominance": dominance,
        }.items():
            tile_grid, expanded = aggregate_points_to_tiles(projected_xy, values, width=width, height=height, tile_size=self.tile_size)
            maps[name] = {
                "tile_grid": tile_grid,
                "expanded_map": expanded,
            }

        summary = {
            "visible_gaussian_count": int(valid_ids.numel()),
            **scalar_summary(current_grad_n, "current_signed_normal"),
            **scalar_summary(np.abs(current_grad_n), "current_abs_normal"),
            **scalar_summary(current_grad_t, "current_tangent"),
            **scalar_summary(mean_signed, "running_bias"),
            **scalar_summary(std_signed, "running_jitter"),
            **scalar_summary(flip_rate, "running_flip_rate"),
            **scalar_summary(dominance, "running_dominance"),
        }
        return {
            "mode": "gradient_tracking",
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "iteration": int(iteration),
            "phase_name": str(phase_name),
            "camera_name": str(viewpoint_cam.image_name),
            "image_width": width,
            "image_height": height,
            "visible_ids": valid_ids.detach().cpu(),
            "projected_xy": projected_xy,
            "current_grad_n": current_grad_n,
            "current_grad_t": current_grad_t,
            "running_mean_signed": mean_signed,
            "running_mean_abs": mean_abs,
            "running_std_signed": std_signed,
            "running_flip_rate": flip_rate,
            "running_dominance": dominance,
            "maps": maps,
            "summary": summary,
            "gt_image": gt_image.detach().cpu(),
            "render_image": render_image.detach().cpu(),
        }

    def update(
        self,
        iteration: int,
        phase_name: str,
        viewpoint_cam,
        gaussians,
        visibility_filter: torch.Tensor,
        gt_image: torch.Tensor,
        render_image: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled or int(iteration) < int(self.start_iter):
            return None
        grad_xyz = gaussians._xyz.grad
        if grad_xyz is None:
            return None
        gaussian_count = int(gaussians.get_xyz.shape[0])
        self._ensure_state(gaussian_count=gaussian_count, phase_name=phase_name, device=gaussians.get_xyz.device)

        normal, tangent_u, tangent_v = self.basis_provider.get_basis(gaussians)
        grad_n, _, _, grad_t = self.basis_provider.project_gradients(
            grad_xyz=grad_xyz,
            normal=normal,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
        )
        finite = torch.isfinite(grad_xyz).all(dim=1)
        valid = visibility_filter.to(device=grad_xyz.device, dtype=torch.bool) & finite
        if gradient_mask is not None:
            valid = valid & gradient_mask.to(device=grad_xyz.device, dtype=torch.bool)
        if not torch.any(valid):
            return None

        self.visibility_count[valid] += 1
        self.sample_count[valid] += 1
        self.sum_signed[valid] += grad_n[valid]
        self.sum_abs[valid] += torch.abs(grad_n[valid])
        self.sum_sq[valid] += grad_n[valid].pow(2)
        self.sum_tangent[valid] += grad_t[valid]
        self.max_abs[valid] = torch.maximum(self.max_abs[valid], torch.abs(grad_n[valid]))

        sign = torch.sign(grad_n)
        sign[torch.abs(grad_n) <= float(self.min_significant_grad)] = 0.0
        sign_int = sign.to(dtype=torch.int8)
        significant = sign_int != 0
        flips = significant & (self.prev_sign != 0) & (self.prev_sign != sign_int)
        self.flip_count[flips] += 1
        self.prev_sign[significant] = sign_int[significant]

        if self.snapshot_interval > 0 and (int(iteration) % int(self.snapshot_interval) == 0):
            return self._build_snapshot(
                iteration=iteration,
                phase_name=phase_name,
                viewpoint_cam=viewpoint_cam,
                gaussians=gaussians,
                gt_image=gt_image,
                render_image=render_image,
                valid=valid,
                grad_n=grad_n,
                grad_t=grad_t,
            )
        return None


class TwoDDropoutGradientDiagnostic:
    def __init__(
        self,
        enabled: bool,
        start_iter: int,
        interval: int,
        num_masks: int,
        tile_size: int,
        keep_ratio: float,
        basis_provider: DiagnosticBasisProvider,
        loss_mode: str = "masked_l1",
        alpha_threshold: float = 0.1,
        min_active_pixels: int = 256,
    ):
        self.enabled = bool(enabled)
        self.start_iter = int(start_iter)
        self.interval = int(interval)
        self.num_masks = int(num_masks)
        self.tile_size = int(tile_size)
        self.keep_ratio = float(keep_ratio)
        self.basis_provider = basis_provider
        self.loss_mode = str(loss_mode)
        self.alpha_threshold = float(alpha_threshold)
        self.min_active_pixels = int(min_active_pixels)

    def should_run(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if int(iteration) < int(self.start_iter):
            return False
        if int(self.interval) <= 0:
            return False
        return int(iteration) % int(self.interval) == 0

    def _focus_mask_from_rendering(self, rendering: torch.Tensor) -> torch.Tensor:
        depth = rendering[6]
        focus = depth > 1e-6
        if rendering.shape[0] > 7:
            opacity = rendering[7]
            focus = focus & (opacity > float(self.alpha_threshold))
        return focus

    def _sample_keep_masks(self, focus_mask: torch.Tensor, iteration: int) -> List[torch.Tensor]:
        height, width = int(focus_mask.shape[0]), int(focus_mask.shape[1])
        tile = max(int(self.tile_size), 1)
        tiles_x = int(math.ceil(float(width) / float(tile)))
        tiles_y = int(math.ceil(float(height) / float(tile)))
        tile_focus = F.max_pool2d(
            focus_mask[None, None].to(dtype=torch.float32),
            kernel_size=tile,
            stride=tile,
            ceil_mode=True,
        )[0, 0] > 0.5
        active_tiles = tile_focus.cpu().numpy().astype(bool)
        masks = []
        for sample_idx in range(int(self.num_masks)):
            rng = np.random.default_rng(int(iteration) * 1009 + sample_idx * 9173)
            tile_keep = np.ones((tiles_y, tiles_x), dtype=bool)
            candidate = np.argwhere(active_tiles)
            if candidate.size > 0:
                num_active = int(candidate.shape[0])
                num_keep = max(1, int(round(float(self.keep_ratio) * float(num_active))))
                chosen = rng.choice(num_active, size=num_keep, replace=False)
                tile_keep[:] = False
                chosen_tiles = candidate[chosen]
                tile_keep[chosen_tiles[:, 0], chosen_tiles[:, 1]] = True
            keep_map = expand_tile_grid(tile_keep.astype(np.float32), height=height, width=width, tile_size=tile) > 0.5
            keep_mask = focus_mask & torch.from_numpy(keep_map).to(device=focus_mask.device, dtype=torch.bool)
            if int(keep_mask.sum().item()) < int(self.min_active_pixels):
                keep_mask = focus_mask
            masks.append(keep_mask)
        return masks

    def _masked_loss(self, rendered_image: torch.Tensor, gt_image: torch.Tensor, keep_mask: torch.Tensor):
        weight = keep_mask.to(dtype=rendered_image.dtype, device=rendered_image.device)
        active = float(weight.sum().item())
        if active <= 0:
            return None
        if self.loss_mode == "masked_gradient_l1":
            dx_render = rendered_image[:, :, 1:] - rendered_image[:, :, :-1]
            dx_gt = gt_image[:, :, 1:] - gt_image[:, :, :-1]
            dy_render = rendered_image[:, 1:, :] - rendered_image[:, :-1, :]
            dy_gt = gt_image[:, 1:, :] - gt_image[:, :-1, :]
            weight_x = 0.5 * (weight[:, 1:] + weight[:, :-1])
            weight_y = 0.5 * (weight[1:, :] + weight[:-1, :])
            loss_x = (torch.abs(dx_render - dx_gt) * weight_x.unsqueeze(0)).sum() / (weight_x.sum() * rendered_image.shape[0]).clamp_min(1.0)
            loss_y = (torch.abs(dy_render - dy_gt) * weight_y.unsqueeze(0)).sum() / (weight_y.sum() * rendered_image.shape[0]).clamp_min(1.0)
            return 0.5 * (loss_x + loss_y)
        return (torch.abs(rendered_image - gt_image) * weight.unsqueeze(0)).sum() / (weight.sum() * rendered_image.shape[0]).clamp_min(1.0)

    def run(
        self,
        iteration: int,
        phase_name: str,
        viewpoint_cam,
        gaussians,
        pipe,
        background: torch.Tensor,
        gt_image: torch.Tensor,
        base_rendering: torch.Tensor,
        splat_args,
        render_fn,
        build_touch_mask_fn,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, object]]:
        if not self.should_run(iteration):
            return None

        focus_mask = self._focus_mask_from_rendering(base_rendering.detach())
        if int(focus_mask.sum().item()) < int(self.min_active_pixels):
            return None
        keep_masks = self._sample_keep_masks(focus_mask, iteration=iteration)
        if not keep_masks:
            return None

        device = gaussians.get_xyz.device
        gaussian_count = int(gaussians.get_xyz.shape[0])
        resp_count = torch.zeros((gaussian_count,), dtype=torch.int32, device=device)
        resp_sum = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        resp_sq = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        sign_sum = torch.zeros((gaussian_count,), dtype=torch.float32, device=device)
        loss_values = []

        normal, tangent_u, tangent_v = self.basis_provider.get_basis(gaussians)

        for keep_mask in keep_masks:
            gaussians.optimizer.zero_grad(set_to_none=True)
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, splat_args=splat_args)
            rendering = render_pkg["render"]
            render_image = rendering[:3]
            diag_loss = self._masked_loss(render_image, gt_image, keep_mask)
            if diag_loss is None:
                continue
            diag_loss.backward()
            grad_xyz = gaussians._xyz.grad
            if grad_xyz is None:
                continue
            if gradient_mask is not None:
                grad_xyz = grad_xyz * gradient_mask.to(device=grad_xyz.device, dtype=grad_xyz.dtype).unsqueeze(1)
            loss_values.append(float(diag_loss.item()))
            grad_n, _, _, _ = self.basis_provider.project_gradients(
                grad_xyz=grad_xyz,
                normal=normal,
                tangent_u=tangent_u,
                tangent_v=tangent_v,
            )
            touched = build_touch_mask_fn(
                viewpoint_cam,
                gaussians,
                keep_mask,
                visibility_filter=render_pkg["visibility_filter"],
                radii=render_pkg["radii"],
            )
            valid = touched & torch.isfinite(grad_xyz).all(dim=1)
            if gradient_mask is not None:
                valid = valid & gradient_mask.to(device=valid.device, dtype=torch.bool)
            if torch.any(valid):
                resp_count[valid] += 1
                resp_sum[valid] += grad_n[valid]
                resp_sq[valid] += grad_n[valid].pow(2)
                sign_sum[valid] += torch.sign(grad_n[valid])
            gaussians.optimizer.zero_grad(set_to_none=True)

        covered = resp_count > 0
        if not torch.any(covered):
            return None

        xyz = gaussians.get_xyz.detach()[covered]
        projected_xy_t, projected_valid_t = project_points_camera_torch(viewpoint_cam, xyz)
        if not torch.any(projected_valid_t):
            return None
        covered_ids = covered.nonzero(as_tuple=True)[0][projected_valid_t]
        projected_xy = projected_xy_t[projected_valid_t].cpu().numpy().astype(np.float32, copy=False)
        count = resp_count[covered_ids].to(dtype=torch.float32)
        mean_resp = (resp_sum[covered_ids] / torch.clamp_min(count, 1.0)).detach().cpu().numpy().astype(np.float32, copy=False)
        std_resp = torch.sqrt(
            torch.clamp_min(resp_sq[covered_ids] / torch.clamp_min(count, 1.0) - (resp_sum[covered_ids] / torch.clamp_min(count, 1.0)).pow(2), 0.0)
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        sign_agreement = torch.abs(sign_sum[covered_ids] / torch.clamp_min(count, 1.0)).detach().cpu().numpy().astype(np.float32, copy=False)
        coverage_ratio = (count / max(float(len(keep_masks)), 1.0)).detach().cpu().numpy().astype(np.float32, copy=False)

        width = int(viewpoint_cam.image_width)
        height = int(viewpoint_cam.image_height)
        maps = {}
        for name, values in {
            "dropout_bias": mean_resp,
            "dropout_std": std_resp,
            "dropout_sign_agreement": sign_agreement,
            "dropout_coverage": coverage_ratio,
        }.items():
            tile_grid, expanded = aggregate_points_to_tiles(projected_xy, values, width=width, height=height, tile_size=self.tile_size)
            maps[name] = {
                "tile_grid": tile_grid,
                "expanded_map": expanded,
            }

        summary = {
            "visible_gaussian_count": int(covered_ids.numel()),
            "num_masks": int(len(keep_masks)),
            "loss_mean": float(np.mean(loss_values)) if loss_values else 0.0,
            "loss_std": float(np.std(loss_values)) if loss_values else 0.0,
            **scalar_summary(mean_resp, "dropout_bias"),
            **scalar_summary(std_resp, "dropout_std"),
            **scalar_summary(sign_agreement, "dropout_sign_agreement"),
            **scalar_summary(coverage_ratio, "dropout_coverage"),
        }
        return {
            "mode": "dropout_gradient",
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "iteration": int(iteration),
            "phase_name": str(phase_name),
            "camera_name": str(viewpoint_cam.image_name),
            "image_width": width,
            "image_height": height,
            "visible_ids": covered_ids.detach().cpu(),
            "projected_xy": projected_xy,
            "dropout_bias": mean_resp,
            "dropout_std": std_resp,
            "dropout_sign_agreement": sign_agreement,
            "dropout_coverage": coverage_ratio,
            "maps": maps,
            "summary": summary,
            "gt_image": gt_image.detach().cpu(),
        }


def export_diagnostic_bundle(
    output_root: Path,
    iteration: int,
    phase_name: str,
    camera_name: str,
    gradient_snapshot: Optional[Dict[str, object]],
    dropout_snapshot: Optional[Dict[str, object]],
) -> Optional[DiagnosticRuntimeSummary]:
    if gradient_snapshot is None and dropout_snapshot is None:
        return None

    output_root = Path(output_root)
    bundle_dir = output_root / f"iter_{int(iteration):06d}_{str(camera_name)}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "iteration": int(iteration),
        "phase_name": str(phase_name),
        "camera_name": str(camera_name),
    }
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "iteration": int(iteration),
        "phase_name": str(phase_name),
        "camera_name": str(camera_name),
        "gradient_tracking": gradient_snapshot,
        "dropout_gradient": dropout_snapshot,
    }
    torch.save(payload, bundle_dir / "snapshot.pt")

    base_image = None
    if gradient_snapshot is not None:
        base_image = chw_to_hwc_numpy(gradient_snapshot["gt_image"])
        save_rgb_image(bundle_dir / "gt.png", gradient_snapshot["gt_image"])
        save_rgb_image(bundle_dir / "render.png", gradient_snapshot["render_image"])
        grad_maps = gradient_snapshot["maps"]
        save_metric_overlay(base_image, grad_maps["current_signed_normal"]["expanded_map"], bundle_dir / "grad_current_signed_overlay.png", signed=True)
        save_metric_overlay(base_image, grad_maps["running_bias"]["expanded_map"], bundle_dir / "grad_running_bias_overlay.png", signed=True)
        save_metric_overlay(base_image, grad_maps["running_jitter"]["expanded_map"], bundle_dir / "grad_running_jitter_overlay.png", signed=False)
        save_metric_overlay(base_image, grad_maps["running_flip_rate"]["expanded_map"], bundle_dir / "grad_running_flip_overlay.png", signed=False)
        save_metric_overlay(base_image, grad_maps["running_dominance"]["expanded_map"], bundle_dir / "grad_running_dominance_overlay.png", signed=False)
        summary.update({f"grad_{key}": value for key, value in gradient_snapshot["summary"].items()})

    if dropout_snapshot is not None:
        if base_image is None:
            base_image = chw_to_hwc_numpy(dropout_snapshot["gt_image"])
            save_rgb_image(bundle_dir / "gt.png", dropout_snapshot["gt_image"])
        drop_maps = dropout_snapshot["maps"]
        save_metric_overlay(base_image, drop_maps["dropout_bias"]["expanded_map"], bundle_dir / "dropout_bias_overlay.png", signed=True)
        save_metric_overlay(base_image, drop_maps["dropout_std"]["expanded_map"], bundle_dir / "dropout_std_overlay.png", signed=False)
        save_metric_overlay(base_image, drop_maps["dropout_sign_agreement"]["expanded_map"], bundle_dir / "dropout_sign_agreement_overlay.png", signed=False)
        save_metric_overlay(base_image, drop_maps["dropout_coverage"]["expanded_map"], bundle_dir / "dropout_coverage_overlay.png", signed=False)
        summary.update({f"drop_{key}": value for key, value in dropout_snapshot["summary"].items()})

    overlay_candidates = [
        bundle_dir / "grad_current_signed_overlay.png",
        bundle_dir / "grad_running_bias_overlay.png",
        bundle_dir / "grad_running_jitter_overlay.png",
        bundle_dir / "grad_running_flip_overlay.png",
        bundle_dir / "dropout_bias_overlay.png",
        bundle_dir / "dropout_std_overlay.png",
        bundle_dir / "dropout_sign_agreement_overlay.png",
        bundle_dir / "dropout_coverage_overlay.png",
    ]
    make_contact_sheet(overlay_candidates, bundle_dir / "contact_sheet.png", columns=2)

    with open(bundle_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return DiagnosticRuntimeSummary(
        iteration=int(iteration),
        phase_name=str(phase_name),
        camera_name=str(camera_name),
        bundle_dir=bundle_dir,
        summary=summary,
    )
