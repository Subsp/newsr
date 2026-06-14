import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as torch_F
from scipy.spatial import cKDTree

from utils.general_utils import build_rotation

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class SOFRegularizationConfig:
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    min_scale_reg: float = 0.0
    lambda_distortion: float = 0.0
    lambda_depth_normal: float = 0.0
    lambda_smoothness: float = 0.0
    lambda_extent: float = 0.0
    lambda_opacity_field: float = 0.0
    distortion_from_iter: int = 0
    depth_normal_from_iter: int = 0
    lambda_surface_thin: float = 0.0
    surface_thin_mesh_path: str = ""
    surface_thin_from_iter: int = 0
    surface_thin_until_iter: int = 0
    surface_thin_sample_count: int = 500000
    surface_thin_update_interval: int = 500
    surface_thin_gaussian_sample_count: int = 65536
    surface_thin_offset_margin: float = 0.02
    surface_thin_normal_scale_target: float = 0.0
    surface_thin_normal_scale_weight: float = 1.0


def _load_trimesh_surface(mesh_path: str):
    if trimesh is None:
        raise ImportError("trimesh is required when lambda_surface_thin > 0")
    loaded = trimesh.load(mesh_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh object type from {mesh_path}: {type(loaded)!r}")
    if loaded.vertices.shape[0] == 0 or loaded.faces.shape[0] == 0:
        raise ValueError(f"Mesh has no vertices/faces: {mesh_path}")
    return loaded


def _camera_to_world(viewpoint_camera: object) -> torch.Tensor:
    return viewpoint_camera.world_view_transform.transpose(0, 1).inverse()


def _depths_to_points(viewpoint_camera: object, depthmap: torch.Tensor) -> torch.Tensor:
    c2w = _camera_to_world(viewpoint_camera)
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    fx = width / (2.0 * math.tan(float(viewpoint_camera.FoVx) / 2.0))
    fy = height / (2.0 * math.tan(float(viewpoint_camera.FoVy) / 2.0))
    intrins = torch.tensor(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=depthmap.dtype,
        device=depthmap.device,
    )
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=depthmap.device, dtype=depthmap.dtype) + 0.5,
        torch.arange(height, device=depthmap.device, dtype=depthmap.dtype) + 0.5,
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ torch.inverse(intrins).transpose(0, 1) @ c2w[:3, :3].transpose(0, 1)
    rays_o = c2w[:3, 3]
    return depthmap.reshape(-1, 1) * rays_d + rays_o


def _depth_to_normal(viewpoint_camera: object, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = _depths_to_points(viewpoint_camera, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points


def _central_diff(image_hwc: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(image_hwc)[:, :, 0]
    dx = torch.cat([image_hwc[2:, 1:-1] - image_hwc[:-2, 1:-1]], dim=0)
    dy = torch.cat([image_hwc[1:-1, 2:] - image_hwc[1:-1, :-2]], dim=1)
    output[1:-1, 1:-1] = torch.norm(dx, dim=-1) + torch.norm(dy, dim=-1)
    return output


def _weighted_mean_map(values: torch.Tensor, weights: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if weights is None:
        return values.mean()
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def _camera_space_xyz(gaussians, viewpoint_camera: object) -> torch.Tensor:
    xyz = gaussians.get_xyz
    R = torch.as_tensor(viewpoint_camera.R, device=xyz.device, dtype=xyz.dtype)
    T = torch.as_tensor(viewpoint_camera.T, device=xyz.device, dtype=xyz.dtype)
    return xyz @ R + T[None, :]


def _gaussian_world_normals(gaussians, viewpoint_camera: object) -> torch.Tensor:
    scales = gaussians.get_scaling_with_3D_filter
    rotations = build_rotation(gaussians.get_rotation)
    min_axis = torch.argmin(scales, dim=-1)
    gather_index = min_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(rotations, 2, gather_index).squeeze(-1)
    view_dirs = viewpoint_camera.camera_center[None, :] - gaussians.get_xyz
    flip_mask = (normals * view_dirs).sum(dim=-1, keepdim=True) < 0
    normals = torch.where(flip_mask, -normals, normals)
    return torch_F.normalize(normals, dim=-1)


def _gaussian_extent_proxy(gaussians, viewpoint_camera: object) -> torch.Tensor:
    xyz_cam = _camera_space_xyz(gaussians, viewpoint_camera)
    depth = xyz_cam[:, 2].clamp_min(1e-4)
    scales = gaussians.get_scaling_with_3D_filter.max(dim=-1).values
    focal = 0.5 * (float(viewpoint_camera.focal_x) + float(viewpoint_camera.focal_y))
    image_span = float(max(int(viewpoint_camera.image_width), int(viewpoint_camera.image_height)))
    return (2.0 * focal * scales / depth / max(image_span, 1.0)).clamp_min(0.0)


def _render_override_image(
    *,
    render_fn: Callable,
    viewpoint_camera: object,
    gaussians,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    subpixel_offset: Optional[torch.Tensor],
    override_color: torch.Tensor,
):
    return render_fn(
        viewpoint_camera,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        override_color=override_color,
        subpixel_offset=subpixel_offset,
    )["render"]


class _MeshSurfaceThinningRegularizer:
    def __init__(self, cfg: SOFRegularizationConfig):
        self.cfg = cfg
        self.enabled = float(cfg.lambda_surface_thin) > 0.0
        self.tree = None
        self.surface_points_np = None
        self.surface_normals_np = None
        self.anchor_points = None
        self.anchor_normals = None
        self.anchor_count = 0
        self.last_update_iter = -1

        if not self.enabled:
            return
        if not cfg.surface_thin_mesh_path:
            raise ValueError("--lambda_surface_thin > 0 requires --surface_thin_mesh_path")

        mesh = _load_trimesh_surface(cfg.surface_thin_mesh_path)
        sample_count = max(1, int(cfg.surface_thin_sample_count))
        surface_points, face_ids = trimesh.sample.sample_surface(mesh, sample_count)
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        surface_normals = face_normals[np.asarray(face_ids, dtype=np.int64)]
        normal_norm = np.linalg.norm(surface_normals, axis=1, keepdims=True)
        surface_normals = surface_normals / np.maximum(normal_norm, 1e-8)

        self.surface_points_np = np.asarray(surface_points, dtype=np.float32)
        self.surface_normals_np = np.asarray(surface_normals, dtype=np.float32)
        self.tree = cKDTree(self.surface_points_np)
        print(
            "[surface-thin] loaded mesh surface prior: "
            f"samples={self.surface_points_np.shape[0]}, vertices={len(mesh.vertices)}, faces={len(mesh.faces)}"
        )

    def _active_at(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if iteration < int(self.cfg.surface_thin_from_iter):
            return False
        until_iter = int(self.cfg.surface_thin_until_iter)
        if until_iter > 0 and iteration > until_iter:
            return False
        return True

    def _refresh_anchors(self, gaussians, iteration: int):
        xyz_np = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        try:
            _, nearest_ids = self.tree.query(xyz_np, k=1, workers=-1)
        except TypeError:
            _, nearest_ids = self.tree.query(xyz_np, k=1)
        nearest_ids = np.asarray(nearest_ids, dtype=np.int64)
        device = gaussians.get_xyz.device
        self.anchor_points = torch.as_tensor(self.surface_points_np[nearest_ids], device=device, dtype=torch.float32)
        self.anchor_normals = torch.as_tensor(self.surface_normals_np[nearest_ids], device=device, dtype=torch.float32)
        self.anchor_count = int(xyz_np.shape[0])
        self.last_update_iter = int(iteration)
        print(f"[surface-thin] refreshed nearest mesh anchors for {self.anchor_count} GS at iter {iteration}")

    def loss(self, gaussians, iteration: int):
        if not self._active_at(iteration):
            return None
        total = int(gaussians.get_xyz.shape[0])
        if total <= 0:
            return None
        needs_refresh = (
            self.anchor_points is None
            or self.anchor_count != total
            or (int(iteration) - self.last_update_iter) >= int(self.cfg.surface_thin_update_interval)
        )
        if needs_refresh:
            self._refresh_anchors(gaussians, iteration)

        sample_count = int(self.cfg.surface_thin_gaussian_sample_count)
        if sample_count > 0 and total > sample_count:
            ids = torch.randperm(total, device=gaussians.get_xyz.device)[:sample_count]
        else:
            ids = torch.arange(total, device=gaussians.get_xyz.device)

        xyz = gaussians.get_xyz[ids]
        anchor_points = self.anchor_points[ids].to(dtype=xyz.dtype)
        anchor_normals = self.anchor_normals[ids].to(dtype=xyz.dtype)

        signed_offset = torch.sum((xyz - anchor_points) * anchor_normals, dim=-1)
        margin = float(self.cfg.surface_thin_offset_margin)
        offset_loss = torch.relu(torch.abs(signed_offset) - margin).pow(2).mean()

        target = float(self.cfg.surface_thin_normal_scale_target)
        weight = float(self.cfg.surface_thin_normal_scale_weight)
        if target <= 0.0 or weight <= 0.0:
            return offset_loss

        rotations = build_rotation(gaussians.get_rotation[ids])
        local_normals = torch.bmm(rotations.transpose(1, 2), anchor_normals.unsqueeze(-1)).squeeze(-1)
        normal_extent = torch.sqrt(
            torch.sum((local_normals * gaussians.get_scaling[ids]).pow(2), dim=-1).clamp_min(1e-12)
        )
        normal_scale_loss = torch.relu(normal_extent - target).pow(2).mean()
        return offset_loss + weight * normal_scale_loss


class SOFRegularizationBlock:
    def __init__(self, cfg: SOFRegularizationConfig):
        self.cfg = cfg
        self.surface_thinner = _MeshSurfaceThinningRegularizer(cfg)

    def _static_enabled(self) -> bool:
        return (
            float(self.cfg.opacity_reg) > 0.0
            or float(self.cfg.scale_reg) > 0.0
            or float(self.cfg.min_scale_reg) > 0.0
            or float(self.cfg.lambda_surface_thin) > 0.0
        )

    def _render_enabled(self, iteration: int) -> bool:
        if iteration >= int(self.cfg.distortion_from_iter):
            if (
                float(self.cfg.lambda_distortion) > 0.0
                or float(self.cfg.lambda_opacity_field) > 0.0
                or float(self.cfg.lambda_extent) > 0.0
            ):
                return True
        if iteration >= int(self.cfg.depth_normal_from_iter):
            if float(self.cfg.lambda_depth_normal) > 0.0 or float(self.cfg.lambda_smoothness) > 0.0:
                return True
        return False

    def _has_render_terms(self) -> bool:
        return (
            float(self.cfg.lambda_distortion) > 0.0
            or float(self.cfg.lambda_depth_normal) > 0.0
            or float(self.cfg.lambda_smoothness) > 0.0
            or float(self.cfg.lambda_extent) > 0.0
            or float(self.cfg.lambda_opacity_field) > 0.0
        )

    @property
    def enabled(self) -> bool:
        return self._static_enabled() or self._has_render_terms()

    def _compute_render_regularizers(
        self,
        *,
        gaussians,
        iteration: int,
        viewpoint_camera: object,
        gt_image: torch.Tensor,
        render_pkg: Optional[dict],
        render_fn: Callable,
        pipe,
        background: torch.Tensor,
        kernel_size: float,
        subpixel_offset: Optional[torch.Tensor],
    ):
        if not self._render_enabled(iteration):
            return None, {}

        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        zero_bg = torch.zeros_like(background)
        render_pkg = render_pkg or {}

        alpha_map = render_pkg.get("alpha")
        if alpha_map is None:
            num_gaussians = int(gaussians.get_xyz.shape[0])
            override_ones = torch.ones((num_gaussians, 3), device=device, dtype=dtype)
            alpha_render = _render_override_image(
                render_fn=render_fn,
                viewpoint_camera=viewpoint_camera,
                gaussians=gaussians,
                pipe=pipe,
                background=zero_bg,
                kernel_size=kernel_size,
                subpixel_offset=subpixel_offset,
                override_color=override_ones,
            )
            alpha_map = alpha_render[:1]
        alpha_map = alpha_map.clamp(0.0, 1.0)
        alpha_hw = alpha_map[0]
        alpha_weight = alpha_hw.detach()
        safe_alpha = alpha_map.clamp_min(1e-4)

        total_loss = torch.zeros((), device=device, dtype=dtype)
        metrics = {}

        xyz_cam = _camera_space_xyz(gaussians, viewpoint_camera)
        depth_values = xyz_cam[:, 2].clamp_min(1e-4)
        near = float(getattr(viewpoint_camera, "znear", 0.01))
        far = float(getattr(viewpoint_camera, "zfar", 100.0))
        mapped_depth = (far * depth_values - far * near) / ((far - near) * depth_values).clamp_min(1e-6)

        need_depth_stats = (
            (iteration >= int(self.cfg.distortion_from_iter) and float(self.cfg.lambda_distortion) > 0.0)
            or (iteration >= int(self.cfg.depth_normal_from_iter) and float(self.cfg.lambda_depth_normal) > 0.0)
        )
        depth_map = render_pkg.get("depth")
        distortion_map = render_pkg.get("distortion")
        if need_depth_stats and depth_map is None:
            depth_pack = torch.stack(
                [depth_values, mapped_depth, mapped_depth.square()],
                dim=1,
            )
            depth_render = _render_override_image(
                render_fn=render_fn,
                viewpoint_camera=viewpoint_camera,
                gaussians=gaussians,
                pipe=pipe,
                background=zero_bg,
                kernel_size=kernel_size,
                subpixel_offset=subpixel_offset,
                override_color=depth_pack,
            )
            depth_map = depth_render[0:1] / safe_alpha
            mapped_depth_map = depth_render[1:2] / safe_alpha
            mapped_depth_sq_map = depth_render[2:3] / safe_alpha
        else:
            mapped_depth_map = None
            mapped_depth_sq_map = None

        if iteration >= int(self.cfg.distortion_from_iter) and float(self.cfg.lambda_distortion) > 0.0:
            if distortion_map is None:
                distortion_map = torch.relu(mapped_depth_sq_map - mapped_depth_map.square()) * alpha_map
            distortion_raw = distortion_map.mean()
            distortion_weighted = float(self.cfg.lambda_distortion) * distortion_raw
            total_loss = total_loss + distortion_weighted
            metrics["distortion_raw"] = float(distortion_raw.detach().item())
            metrics["distortion"] = float(distortion_weighted.detach().item())

        if iteration >= int(self.cfg.distortion_from_iter) and float(self.cfg.lambda_opacity_field) > 0.0:
            opacity_field_raw = ((alpha_map - 0.5) ** 2).mean()
            opacity_field_weighted = float(self.cfg.lambda_opacity_field) * opacity_field_raw
            total_loss = total_loss + opacity_field_weighted
            metrics["opacity_field_raw"] = float(opacity_field_raw.detach().item())
            metrics["opacity_field"] = float(opacity_field_weighted.detach().item())

        if iteration >= int(self.cfg.distortion_from_iter) and float(self.cfg.lambda_extent) > 0.0:
            extent_map = render_pkg.get("extent")
            if extent_map is None:
                extent_values = _gaussian_extent_proxy(gaussians, viewpoint_camera)
                extent_pack = extent_values[:, None].repeat(1, 3)
                extent_render = _render_override_image(
                    render_fn=render_fn,
                    viewpoint_camera=viewpoint_camera,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=zero_bg,
                    kernel_size=kernel_size,
                    subpixel_offset=subpixel_offset,
                    override_color=extent_pack,
                )
                extent_map = extent_render[:1]
            extent_raw = extent_map.mean()
            extent_weighted = float(self.cfg.lambda_extent) * extent_raw
            total_loss = total_loss + extent_weighted
            metrics["extent_raw"] = float(extent_raw.detach().item())
            metrics["extent"] = float(extent_weighted.detach().item())

        need_normals = iteration >= int(self.cfg.depth_normal_from_iter) and (
            float(self.cfg.lambda_depth_normal) > 0.0 or float(self.cfg.lambda_smoothness) > 0.0
        )
        if need_normals:
            normal_render = render_pkg.get("normal")
            if normal_render is None:
                normal_values = _gaussian_world_normals(gaussians, viewpoint_camera)
                normal_pack = (normal_values + 1.0) * 0.5
                normal_render = _render_override_image(
                    render_fn=render_fn,
                    viewpoint_camera=viewpoint_camera,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=zero_bg,
                    kernel_size=kernel_size,
                    subpixel_offset=subpixel_offset,
                    override_color=normal_pack,
                )
                render_normal_world = (normal_render / safe_alpha) * 2.0 - 1.0
                render_normal_world = torch.nn.functional.normalize(render_normal_world, p=2, dim=0)
            else:
                render_normal = torch.nn.functional.normalize(normal_render, p=2, dim=0)
                camera_to_world = viewpoint_camera.world_view_transform[:3, :3]
                render_normal_world = camera_to_world @ render_normal.reshape(3, -1)
                render_normal_world = render_normal_world.reshape_as(render_normal)
            nabla_I = _central_diff(gt_image.permute(1, 2, 0))

            if float(self.cfg.lambda_depth_normal) > 0.0 and depth_map is not None:
                depth_normal, _ = _depth_to_normal(viewpoint_camera, depth_map)
                depth_normal = depth_normal.permute(2, 0, 1)
                normal_error = 1.0 - (render_normal_world * depth_normal).sum(dim=0)
                if "normal" in render_pkg and "depth" in render_pkg:
                    depth_normal_raw = normal_error.mean()
                else:
                    depth_normal_raw = _weighted_mean_map(normal_error, alpha_weight)
                depth_normal_weighted = float(self.cfg.lambda_depth_normal) * depth_normal_raw
                total_loss = total_loss + depth_normal_weighted
                metrics["depth_normal_raw"] = float(depth_normal_raw.detach().item())
                metrics["depth_normal"] = float(depth_normal_weighted.detach().item())

            if float(self.cfg.lambda_smoothness) > 0.0:
                normal_smooth = _central_diff(render_normal_world.permute(1, 2, 0)) * torch.exp(-nabla_I)
                if "normal" in render_pkg:
                    smoothness_raw = normal_smooth.mean()
                else:
                    smoothness_raw = _weighted_mean_map(normal_smooth, alpha_weight)
                smoothness_weighted = float(self.cfg.lambda_smoothness) * smoothness_raw
                total_loss = total_loss + smoothness_weighted
                metrics["smoothness_raw"] = float(smoothness_raw.detach().item())
                metrics["smoothness"] = float(smoothness_weighted.detach().item())

        if not metrics:
            return None, {}
        return total_loss, metrics

    def compute(self, gaussians, iteration: int, *, render_ctx: Optional[dict] = None):
        if not self.enabled:
            return None, {}

        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        total_loss = torch.zeros((), device=device, dtype=dtype)
        metrics = {}

        if float(self.cfg.opacity_reg) > 0.0:
            opacity_raw = torch.abs(gaussians.get_opacity).mean()
            opacity_weighted = float(self.cfg.opacity_reg) * opacity_raw
            total_loss = total_loss + opacity_weighted
            metrics["opacity_raw"] = float(opacity_raw.detach().item())
            metrics["opacity"] = float(opacity_weighted.detach().item())

        if float(self.cfg.scale_reg) > 0.0:
            scale_raw = torch.abs(gaussians.get_scaling).mean()
            scale_weighted = float(self.cfg.scale_reg) * scale_raw
            total_loss = total_loss + scale_weighted
            metrics["scale_raw"] = float(scale_raw.detach().item())
            metrics["scale"] = float(scale_weighted.detach().item())

        if float(self.cfg.min_scale_reg) > 0.0:
            min_scale_raw = torch.min(gaussians.get_scaling, dim=-1).values.mean()
            min_scale_weighted = float(self.cfg.min_scale_reg) * min_scale_raw
            total_loss = total_loss + min_scale_weighted
            metrics["min_scale_raw"] = float(min_scale_raw.detach().item())
            metrics["min_scale"] = float(min_scale_weighted.detach().item())

        if float(self.cfg.lambda_surface_thin) > 0.0:
            surface_thin_raw = self.surface_thinner.loss(gaussians, iteration)
            if surface_thin_raw is not None:
                surface_thin_weighted = float(self.cfg.lambda_surface_thin) * surface_thin_raw
                total_loss = total_loss + surface_thin_weighted
                metrics["surface_thin_raw"] = float(surface_thin_raw.detach().item())
                metrics["surface_thin"] = float(surface_thin_weighted.detach().item())

        if render_ctx is not None:
            render_loss, render_metrics = self._compute_render_regularizers(
                gaussians=gaussians,
                iteration=iteration,
                viewpoint_camera=render_ctx["viewpoint_camera"],
                gt_image=render_ctx["gt_image"],
                render_pkg=render_ctx.get("render_pkg"),
                render_fn=render_ctx["render_fn"],
                pipe=render_ctx["pipe"],
                background=render_ctx["background"],
                kernel_size=render_ctx["kernel_size"],
                subpixel_offset=render_ctx["subpixel_offset"],
            )
            if render_loss is not None:
                total_loss = total_loss + render_loss
                metrics.update(render_metrics)

        if not metrics:
            return None, {}
        metrics["total"] = float(total_loss.detach().item())
        return total_loss, metrics
