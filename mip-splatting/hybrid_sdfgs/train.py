import json
import math
import os
import random
import re
import sys
import uuid
from collections import OrderedDict
from argparse import ArgumentParser, Namespace
from functools import partial
from glob import glob
from random import randint

import numpy as np
import torch
import torch.nn.functional as torch_F
from scipy.spatial import cKDTree
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import (
    network_gui,
    render,
    supports_merged_sof_rasterizer,
    supports_vanilla_sof_rasterizer,
)
from scene import GaussianModel
from scene.gaussian_model import GaussianSourceTag
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.general_utils import build_rotation, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.camera_utils import loadCam
from utils.sh_utils import RGB2SH

from hybrid_sdfgs.losses import (
    linearized_signed_distance,
    normal_alignment_loss,
    offsurface_opacity_loss,
    surface_distance_loss,
)
from hybrid_sdfgs.blocks import (
    FMGuidanceConfig,
    FlowMatchingSDSLikeBlock,
    FrequencyDecompositionBlock,
    FrequencyLossConfig,
    ScaffoldGeometryBlock,
    ScaffoldGeometryConfig,
    SOFPriorBlock,
    SOFPriorConfig,
    SOFRegularizationBlock,
    SOFRegularizationConfig,
    SDFDensifyBlock,
    SDFDensifyConfig,
)
from hybrid_sdfgs.geometry import ScaffoldLoadConfig, load_scaffold_data
from hybrid_sdfgs.camera_bridge import read_transforms_json_scene
from hybrid_sdfgs.scene import HybridScene
from hybrid_sdfgs.scheduler import HybridLossScheduler
from hybrid_sdfgs.sdf_adapter import build_sdf_adapter

try:
    from diff_gaussian_rasterization import ExtendedSettings, GlobalSortOrder, SortMode
except ImportError:
    ExtendedSettings = None
    GlobalSortOrder = None
    SortMode = None

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class SafeSummaryWriter:
    def __init__(self, writer):
        self.writer = writer
        self.enabled = writer is not None

    def __bool__(self):
        return bool(self.enabled and self.writer is not None)

    def _disable(self, exc):
        if not self.enabled:
            return
        print(f"[TENSORBOARD] disabling writer due to logging failure: {exc}")
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        self.writer = None
        self.enabled = False

    def _call(self, method_name, *args, **kwargs):
        if not self:
            return None
        try:
            return getattr(self.writer, method_name)(*args, **kwargs)
        except Exception as exc:
            self._disable(exc)
            return None

    def add_scalar(self, *args, **kwargs):
        return self._call("add_scalar", *args, **kwargs)

    def add_images(self, *args, **kwargs):
        return self._call("add_images", *args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        return self._call("add_histogram", *args, **kwargs)

    def close(self):
        if not self:
            return
        try:
            self.writer.close()
        except Exception as exc:
            self._disable(exc)


def _laplacian_highfreq(image):
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(image.shape[0], 1, 1, 1)
    return torch_F.conv2d(image[None], kernel, padding=1, groups=image.shape[0])[0]


def _box_highpass(image, kernel_size: int):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    low = _box_lowpass(image, kernel_size)
    return image - low


def _box_lowpass(image, kernel_size: int):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    return torch_F.avg_pool2d(
        image[None],
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )[0]


def _proposal_full_stack_enabled(args):
    return bool(getattr(args, "sdf_proposal_full_stack_enable", False))


def _sobel_gradients(image):
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"image must be [1,H,W], got shape {tuple(image.shape)}")
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    grad_x = torch_F.conv2d(image[None], kernel_x, padding=1)[0]
    grad_y = torch_F.conv2d(image[None], kernel_y, padding=1)[0]
    return grad_x, grad_y


def _normalize_vectors_2d(vectors, eps=1e-6):
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp(min=eps)
    return vectors / norms


def _haar_wavelet_highfreq(image):
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"image must be [1,H,W], got shape {tuple(image.shape)}")
    h = image.shape[1]
    w = image.shape[2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h != 0 or pad_w != 0:
        image = torch_F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    x00 = image[:, 0::2, 0::2]
    x01 = image[:, 0::2, 1::2]
    x10 = image[:, 1::2, 0::2]
    x11 = image[:, 1::2, 1::2]
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    energy = lh.abs() + hl.abs() + hh.abs()
    energy = torch_F.interpolate(
        energy[None],
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0]
    return energy


def _build_tangent_basis(normals, hint_dirs=None):
    normals = torch_F.normalize(normals, dim=-1)
    if hint_dirs is not None:
        hint_dirs = torch_F.normalize(hint_dirs, dim=-1)
        tangential_hint = hint_dirs - (hint_dirs * normals).sum(dim=-1, keepdim=True) * normals
        hint_norm = tangential_hint.norm(dim=-1, keepdim=True)
        use_hint = hint_norm.squeeze(-1) > 1e-6
    else:
        tangential_hint = None
        hint_norm = None
        use_hint = None

    ref_x = torch.tensor([1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype)[None, :].expand_as(normals)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=normals.device, dtype=normals.dtype)[None, :].expand_as(normals)
    use_y = (normals[:, 0].abs() > 0.9)[:, None]
    ref = torch.where(use_y, ref_y, ref_x)
    fallback_t1 = torch_F.normalize(torch.cross(normals, ref, dim=-1), dim=-1)
    if tangential_hint is not None:
        safe_hint = tangential_hint / hint_norm.clamp(min=1e-6)
        t1 = torch.where(use_hint[:, None], safe_hint, fallback_t1)
    else:
        t1 = fallback_t1
    t2 = torch_F.normalize(torch.cross(normals, t1, dim=-1), dim=-1)
    return t1, t2


def _rotation_matrix_to_quaternion(rot_mats):
    if rot_mats.ndim != 3 or rot_mats.shape[1:] != (3, 3):
        raise ValueError(f"rot_mats must be [N,3,3], got {tuple(rot_mats.shape)}")
    m = rot_mats
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros((m.shape[0], 4), device=m.device, dtype=m.dtype)

    cond = trace > 0
    if cond.any():
        s = torch.sqrt(trace[cond] + 1.0) * 2.0
        q[cond, 0] = 0.25 * s
        q[cond, 1] = (m[cond, 2, 1] - m[cond, 1, 2]) / s
        q[cond, 2] = (m[cond, 0, 2] - m[cond, 2, 0]) / s
        q[cond, 3] = (m[cond, 1, 0] - m[cond, 0, 1]) / s

    cond1 = (~cond) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if cond1.any():
        s = torch.sqrt(1.0 + m[cond1, 0, 0] - m[cond1, 1, 1] - m[cond1, 2, 2]).clamp(min=1e-8) * 2.0
        q[cond1, 0] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s
        q[cond1, 1] = 0.25 * s
        q[cond1, 2] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / s
        q[cond1, 3] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / s

    cond2 = (~cond) & (~cond1) & (m[:, 1, 1] > m[:, 2, 2])
    if cond2.any():
        s = torch.sqrt(1.0 + m[cond2, 1, 1] - m[cond2, 0, 0] - m[cond2, 2, 2]).clamp(min=1e-8) * 2.0
        q[cond2, 0] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / s
        q[cond2, 1] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / s
        q[cond2, 2] = 0.25 * s
        q[cond2, 3] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / s

    cond3 = (~cond) & (~cond1) & (~cond2)
    if cond3.any():
        s = torch.sqrt(1.0 + m[cond3, 2, 2] - m[cond3, 0, 0] - m[cond3, 1, 1]).clamp(min=1e-8) * 2.0
        q[cond3, 0] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / s
        q[cond3, 1] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / s
        q[cond3, 2] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / s
        q[cond3, 3] = 0.25 * s

    return torch_F.normalize(q, dim=-1)


def _build_rotation_from_surface(normals, scales, view_dirs=None):
    normals = torch_F.normalize(normals, dim=-1)
    t1, t2 = _build_tangent_basis(normals, hint_dirs=view_dirs)
    min_idx = torch.argmin(scales, dim=1)
    rot = torch.zeros((normals.shape[0], 3, 3), device=normals.device, dtype=normals.dtype)

    mask0 = min_idx == 0
    if mask0.any():
        rot[mask0, :, 0] = normals[mask0]
        rot[mask0, :, 1] = t1[mask0]
        rot[mask0, :, 2] = torch.cross(rot[mask0, :, 0], rot[mask0, :, 1], dim=-1)

    mask1 = min_idx == 1
    if mask1.any():
        rot[mask1, :, 0] = t1[mask1]
        rot[mask1, :, 1] = normals[mask1]
        rot[mask1, :, 2] = torch.cross(rot[mask1, :, 0], rot[mask1, :, 1], dim=-1)

    mask2 = min_idx == 2
    if mask2.any():
        rot[mask2, :, 2] = normals[mask2]
        rot[mask2, :, 0] = t1[mask2]
        rot[mask2, :, 1] = torch.cross(rot[mask2, :, 2], rot[mask2, :, 0], dim=-1)

    rot = torch_F.normalize(rot, dim=1)
    return _rotation_matrix_to_quaternion(rot)


def _sample_feature_map_on_grid(feature_map, grid):
    if feature_map is None:
        return None
    if feature_map.ndim != 3:
        raise ValueError(
            f"feature_map must be [C,H,W], got shape {tuple(feature_map.shape)}"
        )
    sample_grid = grid.view(1, -1, 1, 2)
    sampled = torch_F.grid_sample(
        feature_map[None],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(feature_map.shape[0], -1).transpose(0, 1).contiguous()


def _sample_feature_map_on_pixels(feature_map, pixels_xy):
    if feature_map is None or pixels_xy is None or pixels_xy.numel() == 0:
        return None
    h = float(feature_map.shape[1])
    w = float(feature_map.shape[2])
    if h <= 1 or w <= 1:
        return None
    x_ndc = (pixels_xy[:, 0] / (w - 1.0)) * 2.0 - 1.0
    y_ndc = -((pixels_xy[:, 1] / (h - 1.0)) * 2.0 - 1.0)
    grid = torch.stack(
        [
            x_ndc.clamp(min=-1.0, max=1.0),
            y_ndc.clamp(min=-1.0, max=1.0),
        ],
        dim=-1,
    )
    return _sample_feature_map_on_grid(feature_map, grid)


def _project_world_to_grid(points, camera):
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    homog = torch.cat([points, ones], dim=-1)
    clip = homog @ camera.full_proj_transform.to(device=points.device, dtype=points.dtype)

    w = clip[:, 3]
    w_abs = w.abs()
    valid_w = w_abs > 1e-8
    safe_w = torch.where(valid_w, w, torch.ones_like(w))
    ndc = clip[:, :3] / safe_w[:, None]

    # grid_sample expects y=-1 at top, y=1 at bottom, so flip NDC y.
    grid = torch.stack(
        [
            ndc[:, 0].clamp(min=-1.0, max=1.0),
            (-ndc[:, 1]).clamp(min=-1.0, max=1.0),
        ],
        dim=-1,
    )
    in_view = (
        valid_w
        & (w > 0)
        & (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= -1.0)
        & (ndc[:, 2] <= 1.0)
    )
    return grid, in_view


def _sample_image_guidance_on_points(guidance_map, points, camera):
    if guidance_map is None:
        return None, 0.0
    if guidance_map.ndim != 3 or guidance_map.shape[0] != 1:
        raise ValueError(
            f"guidance_map must be [1,H,W], got shape {tuple(guidance_map.shape)}"
        )

    grid, in_view = _project_world_to_grid(points, camera)
    if in_view.sum().item() == 0:
        zeros = torch.zeros((points.shape[0],), device=points.device, dtype=points.dtype)
        return zeros, 0.0

    sample_grid = grid.view(1, -1, 1, 2)
    sampled = torch_F.grid_sample(
        guidance_map[None],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)
    sampled = sampled * in_view.float()
    valid_ratio = float(in_view.float().mean().detach().item())
    return sampled, valid_ratio


def _sample_feature_map_on_points(feature_map, points, camera):
    if feature_map is None:
        return None, None
    grid, in_view = _project_world_to_grid(points, camera)
    sampled = _sample_feature_map_on_grid(feature_map, grid)
    if sampled is None:
        return None, None
    sampled = sampled * in_view[:, None].float()
    return sampled, in_view


def _extract_2d_sr_proposals(guidance_map, topk, min_value):
    """Select top-k 2D SR proposal pixels from a [1,H,W] guidance map."""
    if guidance_map is None:
        return None, None
    if guidance_map.ndim != 3 or guidance_map.shape[0] != 1:
        raise ValueError(
            f"guidance_map must be [1,H,W], got shape {tuple(guidance_map.shape)}"
        )
    flat = guidance_map.view(-1)
    if flat.numel() == 0:
        return None, None

    k = min(max(int(topk), 0), int(flat.shape[0]))
    if k <= 0:
        return None, None

    vals, idx = torch.topk(flat, k=k, largest=True, sorted=False)
    keep = vals >= float(min_value)
    if keep.sum().item() == 0:
        return None, None
    vals = vals[keep]
    idx = idx[keep]
    h = guidance_map.shape[1]
    w = guidance_map.shape[2]
    y = torch.div(idx, w, rounding_mode="floor")
    x = idx - y * w
    pixels = torch.stack([x, y], dim=-1).to(dtype=torch.float32)
    # Clamp for safety against boundary numeric issues.
    pixels[:, 0] = pixels[:, 0].clamp(min=0.0, max=float(w - 1))
    pixels[:, 1] = pixels[:, 1].clamp(min=0.0, max=float(h - 1))
    return pixels, vals


def _extract_thresholded_sr_pixels(guidance_map, threshold, max_points):
    """Select all pixels above a threshold, with optional random cap."""
    if guidance_map is None:
        return None, None
    if guidance_map.ndim != 3 or guidance_map.shape[0] != 1:
        raise ValueError(
            f"guidance_map must be [1,H,W], got shape {tuple(guidance_map.shape)}"
        )
    flat = guidance_map.view(-1)
    if flat.numel() == 0:
        return None, None

    keep = flat >= float(threshold)
    if keep.sum().item() == 0:
        return None, None
    idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    vals = flat[idx]
    if int(max_points) > 0 and idx.numel() > int(max_points):
        perm = torch.randperm(idx.numel(), device=idx.device)[: int(max_points)]
        idx = idx[perm]
        vals = vals[perm]
    h = guidance_map.shape[1]
    w = guidance_map.shape[2]
    y = torch.div(idx, w, rounding_mode="floor")
    x = idx - y * w
    pixels = torch.stack([x, y], dim=-1).to(dtype=torch.float32)
    pixels[:, 0] = pixels[:, 0].clamp(min=0.0, max=float(w - 1))
    pixels[:, 1] = pixels[:, 1].clamp(min=0.0, max=float(h - 1))
    return pixels, vals


def _build_world_rays_from_pixels(camera, pixels_xy):
    """Backproject image pixels to world rays via inverse full projection."""
    if pixels_xy is None or pixels_xy.numel() == 0:
        return None, None

    device = pixels_xy.device
    dtype = pixels_xy.dtype
    h = float(camera.image_height)
    w = float(camera.image_width)
    if h <= 1 or w <= 1:
        return None, None

    x_ndc = (pixels_xy[:, 0] / (w - 1.0)) * 2.0 - 1.0
    y_ndc = -((pixels_xy[:, 1] / (h - 1.0)) * 2.0 - 1.0)

    near_clip = torch.stack(
        [x_ndc, y_ndc, torch.full_like(x_ndc, -1.0), torch.ones_like(x_ndc)],
        dim=-1,
    )
    far_clip = torch.stack(
        [x_ndc, y_ndc, torch.full_like(x_ndc, 1.0), torch.ones_like(x_ndc)],
        dim=-1,
    )

    inv_full = torch.inverse(
        camera.full_proj_transform.to(device=device, dtype=dtype)
    )

    world_near_h = near_clip @ inv_full
    world_far_h = far_clip @ inv_full
    near_w = world_near_h[:, 3:4].abs().clamp(min=1e-8)
    far_w = world_far_h[:, 3:4].abs().clamp(min=1e-8)
    world_near = world_near_h[:, :3] / near_w
    world_far = world_far_h[:, :3] / far_w

    dirs = torch_F.normalize(world_far - world_near, dim=-1)
    origin = camera.camera_center.to(device=device, dtype=dtype)[None, :].expand_as(dirs)
    return origin, dirs


def _gather_image_on_pixels(image_chw, pixels_xy):
    if image_chw is None or pixels_xy is None or pixels_xy.numel() == 0:
        return None
    if image_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image tensor, got shape {tuple(image_chw.shape)}")
    x = torch.round(pixels_xy[:, 0]).to(dtype=torch.long)
    y = torch.round(pixels_xy[:, 1]).to(dtype=torch.long)
    x = x.clamp(min=0, max=int(image_chw.shape[2]) - 1)
    y = y.clamp(min=0, max=int(image_chw.shape[1]) - 1)
    return image_chw[:, y, x].transpose(0, 1).contiguous()


def _gather_scalar_map_on_pixels(map_chw, pixels_xy):
    samples = _gather_image_on_pixels(map_chw, pixels_xy)
    if samples is None:
        return None
    if samples.ndim != 2 or samples.shape[1] != 1:
        raise ValueError(f"Expected single-channel map samples, got shape {tuple(samples.shape)}")
    return samples[:, 0]


def _unproject_pixels_with_camera_z(camera, pixels_xy, depth_z):
    if pixels_xy is None or depth_z is None or pixels_xy.numel() == 0:
        return None
    device = pixels_xy.device
    dtype = pixels_xy.dtype
    depth_z = depth_z.to(device=device, dtype=dtype).reshape(-1).clamp(min=1e-6)
    x_cam = (pixels_xy[:, 0] - float(camera.image_width) / 2.0) / float(camera.focal_x) * depth_z
    y_cam = (pixels_xy[:, 1] - float(camera.image_height) / 2.0) / float(camera.focal_y) * depth_z
    xyz_cam = torch.stack([x_cam, y_cam, depth_z], dim=-1)
    R = torch.as_tensor(camera.R, device=device, dtype=dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=dtype)
    return (xyz_cam - T.unsqueeze(0)) @ R.transpose(0, 1)


def _gather_guidance_gradients_on_pixels(guidance_map, pixels_xy):
    if guidance_map is None or pixels_xy is None or pixels_xy.numel() == 0:
        return None, None
    if guidance_map.ndim != 3:
        raise ValueError(f"Expected [C,H,W] guidance tensor, got shape {tuple(guidance_map.shape)}")
    device = pixels_xy.device
    dtype = pixels_xy.dtype
    guidance = guidance_map[:1].to(device=device, dtype=dtype)
    grad_x = torch.zeros_like(guidance)
    grad_y = torch.zeros_like(guidance)
    if guidance.shape[2] > 1:
        grad_x[:, :, 1:-1] = 0.5 * (guidance[:, :, 2:] - guidance[:, :, :-2])
        grad_x[:, :, 0] = guidance[:, :, 1] - guidance[:, :, 0]
        grad_x[:, :, -1] = guidance[:, :, -1] - guidance[:, :, -2]
    if guidance.shape[1] > 1:
        grad_y[:, 1:-1, :] = 0.5 * (guidance[:, 2:, :] - guidance[:, :-2, :])
        grad_y[:, 0, :] = guidance[:, 1, :] - guidance[:, 0, :]
        grad_y[:, -1, :] = guidance[:, -1, :] - guidance[:, -2, :]
    return (
        _gather_scalar_map_on_pixels(grad_x, pixels_xy),
        _gather_scalar_map_on_pixels(grad_y, pixels_xy),
    )


def _build_hf_seed_shape_from_guidance(
    camera,
    guidance_map,
    pixels_xy,
    depth_z,
    new_xyz,
    base_scale,
    *,
    shape_mode: str,
    long_ratio: float,
    short_ratio: float,
    normal_ratio: float,
    confidence_power: float,
):
    mode = str(shape_mode or "isotropic").lower()
    if mode in {"isotropic", "legacy", "sphere", "round"}:
        scales = base_scale[:, None].repeat(1, 3).clamp(min=1e-7)
        rotations = torch.zeros((new_xyz.shape[0], 4), device=new_xyz.device, dtype=new_xyz.dtype)
        rotations[:, 0] = 1.0
        return scales, rotations, {
            "hf_seed_shape_anisotropy_mean": 1.0,
            "hf_seed_shape_normal_ratio": 1.0,
            "hf_seed_shape_orient_conf_mean": 0.0,
        }

    device = new_xyz.device
    dtype = new_xyz.dtype
    normal_axis = torch_F.normalize(
        new_xyz - camera.camera_center.to(device=device, dtype=dtype)[None, :],
        dim=-1,
    )

    confidence = torch.zeros_like(base_scale)
    tangent_xy = torch.zeros((pixels_xy.shape[0], 2), device=device, dtype=dtype)
    tangent_xy[:, 0] = 1.0
    if mode in {"hf_oriented", "guidance", "edge", "structure"}:
        grad_x, grad_y = _gather_guidance_gradients_on_pixels(guidance_map, pixels_xy)
        if grad_x is not None and grad_y is not None:
            grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
            max_grad = grad_mag.max().clamp(min=1e-8)
            confidence = torch.clamp(grad_mag / max_grad, min=0.0, max=1.0)
            power = max(float(confidence_power), 1e-4)
            confidence = confidence.pow(power)
            tangent_xy = torch.stack([-grad_y, grad_x], dim=-1)
            tangent_norm = tangent_xy.norm(dim=-1, keepdim=True)
            fallback_xy = torch.zeros_like(tangent_xy)
            fallback_xy[:, 0] = 1.0
            tangent_xy = torch.where(tangent_norm > 1e-8, tangent_xy / tangent_norm.clamp(min=1e-8), fallback_xy)
    elif mode not in {"view_flat", "flat", "billboard"}:
        mode = "view_flat"

    shifted_xyz = _unproject_pixels_with_camera_z(camera, pixels_xy + tangent_xy, depth_z)
    if shifted_xyz is None:
        shifted_xyz = new_xyz + torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)[None, :]
    long_axis = shifted_xyz - new_xyz
    long_axis = long_axis - (long_axis * normal_axis).sum(dim=-1, keepdim=True) * normal_axis

    R_cam_to_world = torch.as_tensor(camera.R, device=device, dtype=dtype).transpose(0, 1)
    fallback_long = R_cam_to_world[:, 0][None, :].expand_as(long_axis)
    fallback_long = fallback_long - (fallback_long * normal_axis).sum(dim=-1, keepdim=True) * normal_axis
    basis_t1, _ = _build_tangent_basis(normal_axis)
    fallback_safe = fallback_long.norm(dim=-1, keepdim=True) > 1e-8
    fallback_long = torch.where(fallback_safe, torch_F.normalize(fallback_long, dim=-1), basis_t1)
    safe_long = long_axis.norm(dim=-1, keepdim=True) > 1e-8
    long_axis = torch.where(safe_long, torch_F.normalize(long_axis, dim=-1), fallback_long)
    short_axis = torch_F.normalize(torch.cross(normal_axis, long_axis, dim=-1), dim=-1)
    long_axis = torch_F.normalize(torch.cross(short_axis, normal_axis, dim=-1), dim=-1)

    long_ratio = max(float(long_ratio), 1e-4)
    short_ratio = max(float(short_ratio), 1e-4)
    normal_ratio = max(float(normal_ratio), 1e-4)
    long_scale = base_scale * (1.0 + (long_ratio - 1.0) * confidence)
    short_scale = base_scale * short_ratio
    normal_scale = base_scale * normal_ratio
    scales = torch.stack([long_scale, short_scale, normal_scale], dim=-1).clamp(min=1e-7)

    rot = torch.stack([long_axis, short_axis, normal_axis], dim=-1)
    rotations = _rotation_matrix_to_quaternion(rot)
    anisotropy = scales.max(dim=1).values / scales.min(dim=1).values.clamp(min=1e-12)
    return scales, rotations, {
        "hf_seed_shape_anisotropy_mean": float(anisotropy.mean().detach().item()),
        "hf_seed_shape_anisotropy_p90": float(torch.quantile(anisotropy.detach().float(), 0.90).item()),
        "hf_seed_shape_normal_ratio": float(normal_ratio),
        "hf_seed_shape_orient_conf_mean": float(confidence.mean().detach().item()),
        "hf_seed_shape_orient_conf_p90": float(torch.quantile(confidence.detach().float(), 0.90).item()),
        "hf_seed_shape_mode_id": 2.0 if mode in {"hf_oriented", "guidance", "edge", "structure"} else 1.0,
    }


def _build_visible_seed_reference(
    gaussians,
    camera,
    visibility_filter=None,
    candidate_mask=None,
):
    if gaussians is None or camera is None:
        return None
    xyz = gaussians.get_xyz.detach()
    total = int(xyz.shape[0])
    if total <= 0:
        return None

    valid = torch.ones((total,), device=xyz.device, dtype=torch.bool)
    if visibility_filter is not None and int(visibility_filter.shape[0]) == total:
        valid &= visibility_filter.to(device=xyz.device, dtype=torch.bool)
    if candidate_mask is not None:
        candidate_mask = candidate_mask.reshape(-1).to(device=xyz.device, dtype=torch.bool)
        if int(candidate_mask.shape[0]) != total:
            return None
        valid &= candidate_mask
    if not torch.any(valid):
        return None

    provider_idx = valid.nonzero(as_tuple=True)[0]
    R = torch.as_tensor(camera.R, device=xyz.device, dtype=xyz.dtype)
    T = torch.as_tensor(camera.T, device=xyz.device, dtype=xyz.dtype)
    xyz_cam = xyz[provider_idx] @ R + T.unsqueeze(0)
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / torch.clamp_min(z, 1e-6) * float(camera.focal_x) + float(camera.image_width) / 2.0
    y = xyz_cam[:, 1] / torch.clamp_min(z, 1e-6) * float(camera.focal_y) + float(camera.image_height) / 2.0
    in_view = (
        (z > 1e-6)
        & (x >= 0.0)
        & (x <= float(camera.image_width - 1))
        & (y >= 0.0)
        & (y <= float(camera.image_height - 1))
    )
    if not torch.any(in_view):
        return None

    provider_idx = provider_idx[in_view]
    pixel_xy = torch.stack([x[in_view], y[in_view]], dim=-1)
    pixel_np = np.ascontiguousarray(pixel_xy.detach().cpu().numpy().astype(np.float32, copy=False))
    if hasattr(gaussians, "_root_id") and int(gaussians._root_id.shape[0]) == total:
        provider_root_id = gaussians._root_id[provider_idx].detach()
    else:
        provider_root_id = provider_idx.to(device=provider_idx.device, dtype=torch.int64)
    return {
        "tree": cKDTree(pixel_np),
        "indices": provider_idx,
        "depth_z": z[in_view].detach(),
        "scale_ref": gaussians.get_scaling[provider_idx].detach().mean(dim=-1),
        "seed_id": gaussians._seed_id[provider_idx].detach(),
        "root_id": provider_root_id,
        "generation": gaussians._generation[provider_idx].detach(),
    }


def _build_prior_residual_seed_births(
    gaussians,
    prior_feature_pack,
    prior_render_pkg,
    reference_camera,
    visibility_filter,
    *,
    iteration,
    max_points,
    prior_delta_clip,
    guidance_threshold,
    scale_multiplier,
    scale_mode,
    min_pixel_radius,
    max_pixel_radius,
    max_provider_ratio,
    shape_mode,
    shape_long_ratio,
    shape_short_ratio,
    shape_normal_ratio,
    shape_confidence_power,
    base_opacity,
    jitter_scale,
    color_residual_gain,
    original_only,
):
    metrics = {
        "hf_seed_threshold": float(guidance_threshold),
    }
    if (
        gaussians is None
        or prior_feature_pack is None
        or reference_camera is None
        or int(max_points) <= 0
    ):
        return None, metrics

    guidance_map = prior_feature_pack.get("guidance")
    if guidance_map is None:
        metrics["hf_seed_no_guidance"] = 1.0
        return None, metrics
    flat_guidance = guidance_map.reshape(-1)
    if flat_guidance.numel() > 0:
        metrics["hf_seed_guidance_map_max"] = float(flat_guidance.max().detach().item())
        metrics["hf_seed_guidance_map_nonzero"] = float((flat_guidance > 0).sum().detach().item())
        if float(guidance_threshold) > 0.0:
            metrics["hf_seed_threshold_candidates"] = float(
                (flat_guidance >= float(guidance_threshold)).sum().detach().item()
            )
    if float(guidance_threshold) > 0.0:
        pixels_xy, guidance_vals = _extract_thresholded_sr_pixels(
            guidance_map=guidance_map,
            threshold=float(guidance_threshold),
            max_points=int(max_points),
        )
    else:
        min_guidance = max(float(torch.finfo(guidance_map.dtype).eps), 1e-8)
        pixels_xy, guidance_vals = _extract_2d_sr_proposals(
            guidance_map=guidance_map,
            topk=int(max_points),
            min_value=min_guidance,
        )
    if pixels_xy is None or guidance_vals is None or pixels_xy.shape[0] == 0:
        metrics["hf_seed_empty_candidates"] = 1.0
        return None, metrics
    metrics["hf_seed_selected_pixels"] = float(int(pixels_xy.shape[0]))

    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    pixels_xy = pixels_xy.to(device=device, dtype=dtype)
    guidance_vals = guidance_vals.to(device=device, dtype=dtype).reshape(-1)

    touched_now = None
    if hasattr(gaussians, "_edge_touched") and hasattr(gaussians, "_edge_touch_iter"):
        if int(gaussians._edge_touched.shape[0]) == int(gaussians.get_xyz.shape[0]):
            touched_now = gaussians._edge_touched & (gaussians._edge_touch_iter == int(iteration))
            if not torch.any(touched_now):
                touched_now = None

    source_original_mask = None
    if bool(original_only):
        source_original_mask = gaussians._source_tag == int(GaussianSourceTag.ORIGINAL)

    provider_ref = None
    candidate_masks = []
    if touched_now is not None and source_original_mask is not None:
        candidate_masks.append(touched_now & source_original_mask)
    if touched_now is not None and not bool(original_only):
        candidate_masks.append(touched_now)
    if source_original_mask is not None:
        candidate_masks.append(source_original_mask)
    if not bool(original_only):
        candidate_masks.append(None)
    for candidate_mask in candidate_masks:
        provider_ref = _build_visible_seed_reference(
            gaussians,
            reference_camera,
            visibility_filter=visibility_filter,
            candidate_mask=candidate_mask,
        )
        if provider_ref is not None:
            break
    if provider_ref is None:
        metrics["hf_seed_no_provider"] = 1.0
        return None, metrics

    query_np = np.ascontiguousarray(pixels_xy.detach().cpu().numpy().astype(np.float32, copy=False))
    nn_dist, nn_rows = provider_ref["tree"].query(query_np, k=1)
    nn_dist = np.atleast_1d(np.asarray(nn_dist, dtype=np.float32))
    nn_rows = np.atleast_1d(np.asarray(nn_rows, dtype=np.int64))
    matched_rows = torch.from_numpy(nn_rows).to(device=device, dtype=torch.long)

    provider_idx = provider_ref["indices"][matched_rows]
    provider_depth_z = provider_ref["depth_z"][matched_rows].to(device=device, dtype=dtype)
    provider_scale_ref = provider_ref["scale_ref"][matched_rows].to(device=device, dtype=dtype)
    provider_seed_id = provider_ref["seed_id"][matched_rows].to(device=device, dtype=torch.int64)
    provider_root_id = provider_ref["root_id"][matched_rows].to(device=device, dtype=torch.int64)
    provider_generation = provider_ref["generation"][matched_rows].to(device=device, dtype=torch.int32)

    depth_z = provider_depth_z.clone()
    render_depth_ratio = 0.0
    if isinstance(prior_render_pkg, dict):
        render_depth = prior_render_pkg.get("depth")
        if isinstance(render_depth, torch.Tensor):
            render_depth_z = _gather_scalar_map_on_pixels(render_depth, pixels_xy)
            if render_depth_z is not None:
                render_depth_z = render_depth_z.to(device=device, dtype=dtype)
                render_valid = torch.isfinite(render_depth_z) & (render_depth_z > 1e-6)
                if torch.any(render_valid):
                    depth_z = torch.where(render_valid, render_depth_z, depth_z)
                    render_depth_ratio = float(render_valid.float().mean().detach().item())

    valid = torch.isfinite(depth_z) & (depth_z > 1e-6)
    valid &= torch.isfinite(provider_scale_ref) & (provider_scale_ref > 0.0)
    if not torch.any(valid):
        metrics["hf_seed_no_valid_depth"] = 1.0
        return None, metrics

    pixels_xy = pixels_xy[valid]
    guidance_vals = guidance_vals[valid]
    provider_idx = provider_idx[valid]
    provider_depth_z = provider_depth_z[valid]
    provider_scale_ref = provider_scale_ref[valid]
    provider_seed_id = provider_seed_id[valid]
    provider_root_id = provider_root_id[valid]
    provider_generation = provider_generation[valid]
    depth_z = depth_z[valid]
    nn_dist = nn_dist[valid.detach().cpu().numpy()]

    new_xyz = _unproject_pixels_with_camera_z(reference_camera, pixels_xy, depth_z)
    if new_xyz is None or new_xyz.shape[0] == 0:
        metrics["hf_seed_unproject_fail"] = 1.0
        return None, metrics

    prior_rgb = _gather_image_on_pixels(prior_feature_pack.get("prior_image"), pixels_xy)
    if prior_rgb is None:
        metrics["hf_seed_no_prior_rgb"] = 1.0
        return None, metrics
    anchor_source = prior_feature_pack.get("anchor_image")
    if anchor_source is None:
        anchor_source = prior_feature_pack.get("gt_image")
    anchor_rgb = _gather_image_on_pixels(anchor_source, pixels_xy) if anchor_source is not None else None
    if anchor_rgb is not None:
        residual_rgb = prior_rgb - anchor_rgb
        clip_v = float(prior_delta_clip)
        if clip_v > 0.0:
            residual_rgb = residual_rgb.clamp(min=-clip_v, max=clip_v)
        residual_gain = max(0.0, float(color_residual_gain))
        target_unclamped = anchor_rgb + residual_rgb * residual_gain
        target_rgb = target_unclamped.clamp(0.0, 1.0)
        metrics["hf_seed_residual_abs_mean"] = float(residual_rgb.abs().mean().detach().item())
        metrics["hf_seed_color_residual_gain"] = float(residual_gain)
        metrics["hf_seed_color_clip_ratio"] = float(
            ((target_unclamped < 0.0) | (target_unclamped > 1.0)).float().mean().detach().item()
        )
    else:
        target_rgb = prior_rgb.clamp(0.0, 1.0)
        metrics["hf_seed_color_residual_gain"] = 1.0
        metrics["hf_seed_color_clip_ratio"] = 0.0

    max_guidance = guidance_vals.max().clamp(min=1e-6)
    guidance_strength = torch.clamp(guidance_vals / max_guidance, min=0.0, max=1.0)
    pixel_scale = 0.5 * (
        depth_z / float(reference_camera.focal_x) + depth_z / float(reference_camera.focal_y)
    )
    scale_multiplier = max(float(scale_multiplier), 1e-4)
    scale_mode = str(scale_mode or "legacy_max").lower()
    if scale_mode in {"pixel", "pixel_footprint", "screen"}:
        base_scale = pixel_scale * scale_multiplier
    elif scale_mode in {"min", "min_provider_pixel"}:
        base_scale = torch.minimum(pixel_scale, provider_scale_ref) * scale_multiplier
    elif scale_mode in {"provider", "parent"}:
        base_scale = provider_scale_ref * scale_multiplier
    else:
        base_scale = torch.maximum(pixel_scale, provider_scale_ref) * scale_multiplier
        scale_mode = "legacy_max"

    if float(min_pixel_radius) > 0.0:
        base_scale = torch.maximum(base_scale, pixel_scale * float(min_pixel_radius))
    if float(max_pixel_radius) > 0.0:
        base_scale = torch.minimum(base_scale, pixel_scale * float(max_pixel_radius))
    if float(max_provider_ratio) > 0.0:
        base_scale = torch.minimum(base_scale, provider_scale_ref * float(max_provider_ratio))
    base_scale = base_scale * (0.5 + 0.5 * guidance_strength)
    if float(min_pixel_radius) > 0.0:
        base_scale = torch.maximum(base_scale, pixel_scale * float(min_pixel_radius))
    if float(max_pixel_radius) > 0.0:
        base_scale = torch.minimum(base_scale, pixel_scale * float(max_pixel_radius))
    if float(max_provider_ratio) > 0.0:
        base_scale = torch.minimum(base_scale, provider_scale_ref * float(max_provider_ratio))
    base_scale = base_scale.clamp(min=1e-7)
    if float(jitter_scale) > 0.0:
        new_xyz = new_xyz + torch.randn_like(new_xyz) * base_scale[:, None] * float(jitter_scale)

    new_features_dc = RGB2SH(target_rgb.to(device=device, dtype=torch.float32))[:, None, :].to(
        dtype=gaussians._features_dc.dtype
    )
    new_features_rest = torch.zeros(
        (new_xyz.shape[0], gaussians._features_rest.shape[1], gaussians._features_rest.shape[2]),
        device=device,
        dtype=gaussians._features_rest.dtype,
    )
    opacity_value = min(max(float(base_opacity), 1e-5), 1.0 - 1e-5)
    opacity_values = torch.clamp(
        opacity_value * (0.35 + 0.65 * guidance_strength),
        min=1e-5,
        max=1.0 - 1e-5,
    )
    new_opacities = gaussians.inverse_opacity_activation(
        opacity_values[:, None].to(device=device, dtype=gaussians._opacity.dtype)
    )
    seed_scales, seed_rotations, shape_metrics = _build_hf_seed_shape_from_guidance(
        reference_camera,
        guidance_map,
        pixels_xy,
        depth_z,
        new_xyz,
        base_scale,
        shape_mode=str(shape_mode),
        long_ratio=float(shape_long_ratio),
        short_ratio=float(shape_short_ratio),
        normal_ratio=float(shape_normal_ratio),
        confidence_power=float(shape_confidence_power),
    )
    new_scaling = gaussians.scaling_inverse_activation(
        seed_scales.to(device=device, dtype=dtype).clamp(min=1e-7)
    )
    new_rotation = seed_rotations.to(device=device, dtype=gaussians._rotation.dtype)

    seed_ids = torch.where(
        provider_seed_id >= 0,
        provider_seed_id,
        provider_idx.to(device=device, dtype=torch.int64),
    )
    tracking_state = gaussians._build_tracking_extension(
        new_xyz.shape[0],
        source_tag=gaussians._default_tracking_tensor(
            new_xyz.shape[0],
            int(GaussianSourceTag.PRIOR_INJECTED),
            torch.int32,
        ),
        seed_id=seed_ids,
        root_id=torch.where(
            provider_root_id >= 0,
            provider_root_id,
            provider_idx.to(device=device, dtype=torch.int64),
        ),
        generation=(provider_generation + 1).to(device=device, dtype=torch.int32),
        edge_touched=gaussians._default_tracking_tensor(new_xyz.shape[0], True, torch.bool),
        edge_touch_iter=gaussians._default_tracking_tensor(new_xyz.shape[0], int(iteration), torch.int32),
    )

    metrics.update({
        "hf_seed_candidates": float(int(new_xyz.shape[0])),
        "hf_seed_guidance_mean": float(guidance_vals.mean().detach().item()),
        "hf_seed_guidance_max": float(guidance_vals.max().detach().item()),
        "hf_seed_render_depth_ratio": float(render_depth_ratio),
        "hf_seed_provider_depth_ratio": float(
            torch.isclose(depth_z, provider_depth_z, atol=1e-6, rtol=0.0).float().mean().detach().item()
        ),
        "hf_seed_provider_dist_mean": float(nn_dist.mean()) if nn_dist.size > 0 else 0.0,
        "hf_seed_scale_mode": scale_mode,
        "hf_seed_scale_mean": float(base_scale.mean().detach().item()),
        "hf_seed_scale_p90": float(torch.quantile(base_scale.detach().float(), 0.90).item()),
        "hf_seed_scale_max": float(base_scale.max().detach().item()),
        "hf_seed_pixel_scale_mean": float(pixel_scale.mean().detach().item()),
        "hf_seed_pixel_radius_mean": float((base_scale / pixel_scale.clamp(min=1e-7)).mean().detach().item()),
        "hf_seed_pixel_radius_p90": float(torch.quantile((base_scale / pixel_scale.clamp(min=1e-7)).detach().float(), 0.90).item()),
        "hf_seed_provider_scale_mean": float(provider_scale_ref.mean().detach().item()),
        "hf_seed_provider_ratio_mean": float((base_scale / provider_scale_ref.clamp(min=1e-7)).mean().detach().item()),
        "hf_seed_provider_ratio_p90": float(torch.quantile((base_scale / provider_scale_ref.clamp(min=1e-7)).detach().float(), 0.90).item()),
        "hf_seed_opacity_mean": float(opacity_values.mean().detach().item()),
    })
    metrics.update(shape_metrics)
    return {
        "xyz": new_xyz.to(dtype=gaussians._xyz.dtype),
        "features_dc": new_features_dc,
        "features_rest": new_features_rest,
        "opacities": new_opacities,
        "scaling": new_scaling.to(dtype=gaussians._scaling.dtype),
        "rotation": new_rotation,
        "tracking_state": tracking_state,
    }, metrics


def _build_prior_hf_focus_weight(prior_feature_pack, base_mask, hybrid_args):
    if prior_feature_pack is None or base_mask is None:
        return base_mask
    boost = float(getattr(hybrid_args, "prior_hf_focus_boost", 0.0))
    if boost <= 0.0:
        return base_mask
    guidance_map = prior_feature_pack.get("guidance")
    if guidance_map is None:
        return base_mask
    guidance = guidance_map.to(device=base_mask.device, dtype=base_mask.dtype).clamp(min=0.0)
    max_guidance = guidance.max()
    if float(max_guidance.detach().item()) <= 0.0:
        return base_mask
    power = max(float(getattr(hybrid_args, "prior_hf_focus_power", 1.0)), 1e-6)
    normalized = (guidance / max_guidance.clamp(min=1e-6)).pow(power)
    return base_mask * (1.0 + boost * normalized)


def _build_prior_hf_patch_weight(
    prior_feature_pack,
    base_mask,
    top_fraction: float,
    min_guidance: float,
):
    if prior_feature_pack is None or base_mask is None:
        return None, 0.0
    guidance_map = prior_feature_pack.get("guidance")
    if guidance_map is None:
        return None, 0.0
    guidance = guidance_map.to(device=base_mask.device, dtype=base_mask.dtype).clamp(min=0.0)
    max_guidance = guidance.max()
    if float(max_guidance.detach().item()) <= 0.0:
        return None, 0.0
    normalized = guidance / max_guidance.clamp(min=1e-6)
    valid = (base_mask > 0) & (normalized > 0)
    valid_values = normalized[valid]
    if valid_values.numel() <= 0:
        return None, 0.0

    threshold = max(0.0, float(min_guidance))
    top_fraction = float(top_fraction)
    if 0.0 < top_fraction < 1.0:
        keep = max(1, int(math.ceil(float(valid_values.numel()) * top_fraction)))
        top_values = torch.topk(valid_values.reshape(-1), k=keep, largest=True, sorted=False).values
        threshold = max(threshold, float(top_values.min().detach().item()))

    patch_mask = valid & (normalized >= threshold)
    if not torch.any(patch_mask):
        return None, 0.0
    patch_weight = base_mask * patch_mask.to(dtype=base_mask.dtype) * (0.25 + normalized)
    coverage = patch_mask.to(dtype=base_mask.dtype).sum() / torch.clamp(
        (base_mask > 0).to(dtype=base_mask.dtype).sum(),
        min=1.0,
    )
    return patch_weight, float(coverage.detach().item())


def _build_recent_prior_seed_mask(gaussians, iteration: int, recent_iters: int):
    if gaussians is None or int(recent_iters) <= 0:
        return None
    if not hasattr(gaussians, "_source_tag") or not hasattr(gaussians, "_edge_touch_iter"):
        return None
    total = int(gaussians.get_xyz.shape[0])
    if int(gaussians._source_tag.shape[0]) != total or int(gaussians._edge_touch_iter.shape[0]) != total:
        return None
    recent_mask = gaussians._source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    recent_mask = recent_mask & (gaussians._edge_touch_iter >= 0)
    recent_mask = recent_mask & ((int(iteration) - gaussians._edge_touch_iter) < int(recent_iters))
    if not torch.any(recent_mask):
        return None
    return recent_mask


def _summarize_prior_injected_gaussians(gaussians, large_scale_threshold: float = 0.0):
    if gaussians is None or not hasattr(gaussians, "_source_tag"):
        return {}
    total = int(gaussians.get_xyz.shape[0])
    if int(gaussians._source_tag.shape[0]) != total:
        return {}
    mask = gaussians._source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    count = int(mask.sum().item())
    if count <= 0:
        return {
            "hf_seed_live": 0.0,
            "hf_seed_unique_ids": 0.0,
            "hf_seed_clone_ratio": 0.0,
        }

    scales = gaussians.get_scaling[mask].detach().float()
    max_axis = scales.max(dim=1).values
    geom = scales.clamp(min=1e-12).prod(dim=1).pow(1.0 / 3.0)
    seed_ids = gaussians._seed_id[mask]
    assigned = seed_ids >= 0
    unique_count = int(torch.unique(seed_ids[assigned]).numel()) if torch.any(assigned) else 0
    generations = gaussians._generation[mask].detach().float()

    summary = {
        "hf_seed_live": float(count),
        "hf_seed_unique_ids": float(unique_count),
        "hf_seed_clone_ratio": float(count / max(unique_count, 1)),
        "hf_seed_scale_geom_median": float(torch.quantile(geom, 0.50).item()),
        "hf_seed_scale_geom_p90": float(torch.quantile(geom, 0.90).item()),
        "hf_seed_scale_max_axis_median": float(torch.quantile(max_axis, 0.50).item()),
        "hf_seed_scale_max_axis_p90": float(torch.quantile(max_axis, 0.90).item()),
        "hf_seed_scale_max_axis_max": float(max_axis.max().item()),
        "hf_seed_generation_mean": float(generations.mean().item()),
        "hf_seed_generation_max": float(generations.max().item()),
    }
    if float(large_scale_threshold) > 0.0:
        large = max_axis > float(large_scale_threshold)
        summary["hf_seed_large_count"] = float(large.sum().item())
        summary["hf_seed_large_ratio"] = float(large.float().mean().item())
    return summary


def _clamp_prior_injected_scales(
    gaussians,
    *,
    max_axis: float = 0.0,
    min_axis: float = 0.0,
    max_anisotropy: float = 0.0,
):
    if gaussians is None or not hasattr(gaussians, "_source_tag"):
        return {}
    if max_axis <= 0.0 and min_axis <= 0.0 and max_anisotropy <= 0.0:
        return {}
    total = int(gaussians.get_xyz.shape[0])
    if int(gaussians._source_tag.shape[0]) != total:
        return {}
    mask = gaussians._source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    count = int(mask.sum().item())
    if count <= 0:
        return {"hf_seed_scale_clamp_count": 0.0}

    scales_before = gaussians.get_scaling[mask].detach()
    scales = scales_before.clone().clamp(min=1e-12)
    max_before = scales.max(dim=1).values
    min_before = scales.min(dim=1).values
    anis_before = max_before / min_before.clamp(min=1e-12)

    if float(max_axis) > 0.0:
        scales = scales.clamp(max=float(max_axis))
    if float(min_axis) > 0.0:
        scales = scales.clamp(min=float(min_axis))
    if float(max_anisotropy) > 0.0:
        max_after_partial = scales.max(dim=1, keepdim=True).values
        min_allowed = max_after_partial / float(max_anisotropy)
        scales = torch.maximum(scales, min_allowed)
    if float(max_axis) > 0.0:
        scales = scales.clamp(max=float(max_axis))

    changed = (scales - scales_before).abs().amax(dim=1) > 1e-8
    changed_count = int(changed.sum().item())
    if changed_count > 0:
        gaussians._scaling.data[mask] = gaussians.scaling_inverse_activation(scales).to(
            device=gaussians._scaling.device,
            dtype=gaussians._scaling.dtype,
        )

    max_after = scales.max(dim=1).values
    min_after = scales.min(dim=1).values
    anis_after = max_after / min_after.clamp(min=1e-12)
    return {
        "hf_seed_scale_clamp_count": float(changed_count),
        "hf_seed_scale_clamp_ratio": float(changed.float().mean().item()),
        "hf_seed_scale_clamp_max_before": float(max_before.max().item()),
        "hf_seed_scale_clamp_max_after": float(max_after.max().item()),
        "hf_seed_scale_clamp_anis_p90_before": float(torch.quantile(anis_before.float(), 0.90).item()),
        "hf_seed_scale_clamp_anis_p90_after": float(torch.quantile(anis_after.float(), 0.90).item()),
    }


def _build_prior_bubble_candidate_mask(
    gaussians,
    *,
    opacity_min: float = 0.10,
    max_axis_min: float = 0.0,
    max_axis_max: float = 0.0,
    anisotropy_max: float = 2.5,
    min_generation: int = 0,
):
    if gaussians is None or not hasattr(gaussians, "_source_tag"):
        return None, {}
    total = int(gaussians.get_xyz.shape[0])
    if int(gaussians._source_tag.shape[0]) != total:
        return None, {}

    with torch.no_grad():
        prior_mask = gaussians._source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
        prior_count = int(prior_mask.sum().item())
        if prior_count <= 0:
            return prior_mask, {
                "bubble_cleanup_prior": 0.0,
                "bubble_cleanup_count": 0.0,
                "bubble_cleanup_ratio": 0.0,
            }

        scales = gaussians.get_scaling.detach().float()
        opacities = gaussians.get_opacity.detach().view(-1).float()
        max_axis = scales.max(dim=1).values
        min_axis = scales.min(dim=1).values.clamp(min=1e-12)
        anisotropy = max_axis / min_axis

        candidate = prior_mask
        candidate = candidate & (opacities >= float(opacity_min))
        if float(max_axis_min) > 0.0:
            candidate = candidate & (max_axis >= float(max_axis_min))
        if float(max_axis_max) > 0.0:
            candidate = candidate & (max_axis <= float(max_axis_max))
        if float(anisotropy_max) > 0.0:
            candidate = candidate & (anisotropy <= float(anisotropy_max))
        if int(min_generation) > 0 and hasattr(gaussians, "_generation"):
            generation = gaussians._generation
            if int(generation.shape[0]) == total:
                candidate = candidate & (generation >= int(min_generation))

        count = int(candidate.sum().item())
        metrics = {
            "bubble_cleanup_prior": float(prior_count),
            "bubble_cleanup_count": float(count),
            "bubble_cleanup_ratio": float(count / max(prior_count, 1)),
        }
        if count > 0:
            selected_opacity = opacities[candidate]
            selected_max = max_axis[candidate]
            selected_anis = anisotropy[candidate]
            metrics.update(
                {
                    "bubble_cleanup_opacity_mean": float(selected_opacity.mean().item()),
                    "bubble_cleanup_opacity_p90": float(torch.quantile(selected_opacity, 0.90).item()),
                    "bubble_cleanup_scale_max_p50": float(torch.quantile(selected_max, 0.50).item()),
                    "bubble_cleanup_scale_max_p90": float(torch.quantile(selected_max, 0.90).item()),
                    "bubble_cleanup_anis_p50": float(torch.quantile(selected_anis, 0.50).item()),
                    "bubble_cleanup_anis_p90": float(torch.quantile(selected_anis, 0.90).item()),
                }
            )
        return candidate, metrics


@torch.no_grad()
def _apply_prior_hf_lowfreq_cleanup(
    gaussians,
    *,
    iteration: int,
    from_iter: int = 0,
    until_iter: int = 0,
    interval: int = 1,
    stale_iters: int = 240,
    opacity_min: float = 0.06,
    scale_max_min: float = 0.0,
    scale_max_max: float = 0.0,
    anisotropy_max: float = 0.0,
    opacity_decay: float = 1.0,
    prune_opacity_below: float = 0.0,
    max_prune_fraction: float = 0.0,
    max_prune_count: int = 0,
):
    if gaussians is None or not hasattr(gaussians, "_source_tag") or not hasattr(gaussians, "_edge_touch_iter"):
        return {}
    if int(iteration) < int(from_iter):
        return {}
    if int(until_iter) > 0 and int(iteration) > int(until_iter):
        return {}
    if int(iteration) % max(1, int(interval)) != 0:
        return {}

    total = int(gaussians.get_xyz.shape[0])
    if int(gaussians._source_tag.shape[0]) != total or int(gaussians._edge_touch_iter.shape[0]) != total:
        return {}

    prior_mask = gaussians._source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    prior_count = int(prior_mask.sum().item())
    if prior_count <= 0:
        return {
            "hf_lowfreq_cleanup_prior": 0.0,
            "hf_lowfreq_cleanup_candidates": 0.0,
            "hf_lowfreq_cleanup_pruned": 0.0,
        }

    opacities = gaussians.get_opacity.detach().view(-1).float()
    scales = gaussians.get_scaling.detach().float()
    scale_max = scales.max(dim=1).values
    scale_min = scales.min(dim=1).values.clamp(min=1e-12)
    anisotropy = scale_max / scale_min
    touch_iter = gaussians._edge_touch_iter
    touched = touch_iter >= 0
    age = torch.where(
        touched,
        torch.full_like(touch_iter, int(iteration)) - touch_iter,
        torch.full_like(touch_iter, int(stale_iters)),
    )

    candidate = prior_mask & (age >= int(stale_iters)) & (opacities >= float(opacity_min))
    if float(scale_max_min) > 0.0:
        candidate = candidate & (scale_max >= float(scale_max_min))
    if float(scale_max_max) > 0.0:
        candidate = candidate & (scale_max <= float(scale_max_max))
    if float(anisotropy_max) > 0.0:
        candidate = candidate & (anisotropy <= float(anisotropy_max))

    candidate_count = int(candidate.sum().item())
    metrics = {
        "hf_lowfreq_cleanup_prior": float(prior_count),
        "hf_lowfreq_cleanup_candidates": float(candidate_count),
        "hf_lowfreq_cleanup_ratio": float(candidate_count / max(prior_count, 1)),
        "hf_lowfreq_cleanup_pruned": 0.0,
    }
    if candidate_count <= 0:
        return metrics

    selected_opacity = opacities[candidate]
    selected_scale = scale_max[candidate]
    selected_age = age[candidate].float()
    metrics.update(
        {
            "hf_lowfreq_cleanup_opacity_mean": float(selected_opacity.mean().item()),
            "hf_lowfreq_cleanup_scale_p90": float(torch.quantile(selected_scale, 0.90).item()),
            "hf_lowfreq_cleanup_age_mean": float(selected_age.mean().item()),
        }
    )

    decay = float(opacity_decay)
    new_opacity_all = opacities.clone()
    if 0.0 < decay < 1.0:
        selected_new = torch.clamp(selected_opacity * decay, min=1e-5, max=1.0 - 1e-5)
        gaussians._opacity[candidate] = gaussians.inverse_opacity_activation(
            selected_new[:, None].to(device=gaussians._opacity.device, dtype=gaussians._opacity.dtype)
        )
        new_opacity_all[candidate] = selected_new
        metrics["hf_lowfreq_cleanup_decay"] = float(decay)
        metrics["hf_lowfreq_cleanup_opacity_after"] = float(selected_new.mean().item())

    prune_below = float(prune_opacity_below)
    if prune_below > 0.0:
        prune_mask = candidate & (new_opacity_all <= prune_below)
        prune_count = int(prune_mask.sum().item())
        if prune_count > 0:
            cap = prune_count
            if float(max_prune_fraction) > 0.0:
                cap = min(cap, max(1, int(math.floor(total * float(max_prune_fraction)))))
            if int(max_prune_count) > 0:
                cap = min(cap, int(max_prune_count))
            if cap < prune_count:
                prune_idx = prune_mask.nonzero(as_tuple=True)[0]
                prune_score = (
                    age[prune_idx].float()
                    * scale_max[prune_idx].float().clamp(min=1e-8)
                    * new_opacity_all[prune_idx].float().clamp(min=1e-8)
                )
                keep_local = torch.topk(prune_score, k=max(1, cap), largest=True, sorted=False).indices
                capped_mask = torch.zeros_like(prune_mask)
                capped_mask[prune_idx[keep_local]] = True
                prune_mask = capped_mask
                prune_count = int(prune_mask.sum().item())
            if prune_count > 0:
                gaussians.prune_points(prune_mask)
        metrics["hf_lowfreq_cleanup_pruned"] = float(prune_count)

    return metrics


def _raycast_sdf_surface(
    sdf_adapter,
    ray_origins,
    ray_dirs,
    t_near,
    t_far,
    n_samples,
):
    """Uniformly sample along rays and pick point with minimal |SDF|."""
    if ray_origins is None or ray_origins.numel() == 0:
        return None
    n_rays = ray_origins.shape[0]
    n_steps = max(2, int(n_samples))
    t_vals = torch.linspace(
        float(t_near),
        float(t_far),
        steps=n_steps,
        device=ray_origins.device,
        dtype=ray_origins.dtype,
    )
    pts = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None]
    pts_flat = pts.reshape(-1, 3)
    sdf_vals, sdf_grads = sdf_adapter.query_sdf_and_gradients(pts_flat.detach())
    sdf_vals = sdf_vals.to(device=pts.device, dtype=pts.dtype).view(n_rays, n_steps)
    sdf_grads = sdf_grads.to(device=pts.device, dtype=pts.dtype).view(n_rays, n_steps, 3)

    abs_sdf = sdf_vals.abs()
    min_idx = torch.argmin(abs_sdf, dim=1)
    ridx = torch.arange(n_rays, device=pts.device)
    hit_t = t_vals[min_idx]
    hit_xyz = pts[ridx, min_idx, :]
    hit_sdf = sdf_vals[ridx, min_idx]
    hit_grad = torch_F.normalize(sdf_grads[ridx, min_idx, :], dim=-1)
    return {
        "t": hit_t,
        "xyz": hit_xyz,
        "sdf": hit_sdf,
        "grad": hit_grad,
    }


def _newton_project_to_sdf_surface(sdf_adapter, points, steps):
    if points is None or points.numel() == 0:
        return None
    proj = points
    steps = max(0, int(steps))
    sdf_vals = None
    sdf_grads = None
    for _ in range(steps + 1):
        sdf_vals, sdf_grads = sdf_adapter.query_sdf_and_gradients(proj.detach())
        sdf_vals = sdf_vals.to(device=proj.device, dtype=proj.dtype).view(-1, 1)
        sdf_grads = sdf_grads.to(device=proj.device, dtype=proj.dtype).view(-1, 3)
        if _ == steps:
            break
        denom = sdf_grads.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        proj = proj - sdf_vals * sdf_grads / denom
    return {
        "xyz": proj,
        "sdf": sdf_vals.view(-1),
        "grad": torch_F.normalize(sdf_grads, dim=-1),
    }


def _query_gs_support(gaussians, query_xyz, knn_k):
    """Compute GS support and nearest Gaussian alignment for query points."""
    if query_xyz is None or query_xyz.numel() == 0:
        return None
    centers = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    opacities = gaussians.get_opacity.detach().view(-1)

    k = min(max(1, int(knn_k)), int(centers.shape[0]))
    d = torch.cdist(query_xyz, centers)
    d_k, i_k = torch.topk(d, k=k, dim=1, largest=False, sorted=True)
    sigma = scales.min(dim=1).values[i_k].clamp(min=1e-4)
    opacity_k = opacities[i_k].clamp(min=0.0)
    support = (torch.exp(-d_k / sigma) * opacity_k).sum(dim=1)
    nearest_idx = i_k[:, 0]
    nearest_dist = d_k[:, 0]

    return {
        "support": support,
        "nearest_idx": nearest_idx,
        "nearest_dist": nearest_dist,
        "nearest_sigma": sigma[:, 0],
    }


def _compute_support_sigma_from_scales(scales, sigma_mode, distance_scale, distance_floor):
    scales_sorted, _ = torch.sort(scales, dim=1)
    s_min = scales_sorted[:, 0]
    s_mid = scales_sorted[:, 1]
    s_max = scales_sorted[:, 2]
    if sigma_mode == "min":
        sigma = s_min
    elif sigma_mode == "mid":
        sigma = s_mid
    elif sigma_mode == "max":
        sigma = s_max
    elif sigma_mode == "mean":
        sigma = scales.mean(dim=1)
    else:
        sigma = torch.sqrt((s_min * s_mid).clamp(min=1e-12))
    return (sigma * float(distance_scale)).clamp(min=float(distance_floor))


def _query_gs_support_stable(
    gaussians,
    query_xyz,
    knn_k,
    sigma_mode,
    distance_scale,
    distance_floor,
):
    """Compute GS support with a more stable local scale estimate."""
    if query_xyz is None or query_xyz.numel() == 0:
        return None
    centers = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    opacities = gaussians.get_opacity.detach().view(-1)
    sigma_all = _compute_support_sigma_from_scales(scales, sigma_mode, distance_scale, distance_floor)

    k = min(max(1, int(knn_k)), int(centers.shape[0]))
    d = torch.cdist(query_xyz, centers)
    d_k, i_k = torch.topk(d, k=k, dim=1, largest=False, sorted=True)
    sigma_k = sigma_all[i_k]
    opacity_k = opacities[i_k].clamp(min=0.0)
    support = (torch.exp(-d_k / sigma_k.clamp(min=float(distance_floor))) * opacity_k).sum(dim=1)
    nearest_idx = i_k[:, 0]
    nearest_dist = d_k[:, 0]
    return {
        "support": support,
        "nearest_idx": nearest_idx,
        "nearest_dist": nearest_dist,
        "nearest_sigma": sigma_k[:, 0],
    }


def _diagnose_pseudo_sdf_vs_gs(
    sdf_adapter,
    gaussians,
    max_points,
):
    """Measure how far the pseudo-SDF zero surface has drifted from GS centers."""
    xyz = gaussians.get_xyz.detach()
    if xyz.numel() == 0:
        return {
            "center_abs_sdf_mean": 0.0,
            "center_abs_sdf_p90": 0.0,
            "center_abs_sdf_max": 0.0,
        }

    if xyz.shape[0] > int(max_points):
        idx = torch.randperm(xyz.shape[0], device=xyz.device)[: int(max_points)]
        xyz = xyz[idx]

    sdf_vals, _ = sdf_adapter.query_sdf_and_gradients(xyz)
    abs_sdf = sdf_vals.abs().view(-1)
    q = torch.tensor([0.9], device=abs_sdf.device, dtype=abs_sdf.dtype)
    p90 = torch.quantile(abs_sdf, q).item()
    return {
        "center_abs_sdf_mean": float(abs_sdf.mean().item()),
        "center_abs_sdf_p90": float(p90),
        "center_abs_sdf_max": float(abs_sdf.max().item()),
    }


def _adaptive_gs_refresh_interval(iteration, scheduler, args):
    if scheduler is None:
        return int(args.gs_bootstrap_update_interval)
    stage = scheduler.stage(iteration)
    if stage == "A":
        return int(args.gs_bootstrap_refresh_interval_a)
    if stage == "B":
        return int(args.gs_bootstrap_refresh_interval_b)
    return int(args.gs_bootstrap_refresh_interval_c)


def _maybe_refresh_gs_bootstrap_adapter(sdf_adapter, iteration, scheduler, args):
    if sdf_adapter is None or not hasattr(sdf_adapter, "refresh_from_gaussians"):
        return 0
    interval = _adaptive_gs_refresh_interval(iteration, scheduler, args)
    interval = max(1, int(interval))
    last_refresh_iter = int(getattr(sdf_adapter, "last_refresh_iter", -1))
    if last_refresh_iter < 0 or iteration % interval == 0:
        sdf_adapter.refresh_from_gaussians()
        sdf_adapter.last_refresh_iter = iteration
        return 1
    return 0


def _proposal_trust_from_center_p90(center_abs_sdf_p90, args):
    hard = float(args.sdf_proposal_teacher_p90_hard)
    soft = float(args.sdf_proposal_teacher_p90_soft)
    if center_abs_sdf_p90 >= hard:
        return 0.0, "off"
    if center_abs_sdf_p90 >= soft:
        return float(args.sdf_proposal_teacher_soft_scale), "soft"
    return 1.0, "full"


def _adaptive_support_threshold(support_values, args):
    if support_values is None or support_values.numel() == 0:
        return float(args.sdf_proposal_gs_support_floor)
    q = torch.tensor(
        [float(args.sdf_proposal_gs_support_quantile)],
        device=support_values.device,
        dtype=support_values.dtype,
    )
    qv = float(torch.quantile(support_values, q).item())
    base = float(args.sdf_proposal_gs_support_thresh)
    floor = float(args.sdf_proposal_gs_support_floor)
    return max(floor, min(base, qv))


def _sample_surface_plane_points(
    anchors_xyz,
    anchors_normals,
    base_scale,
    samples_per_anchor,
    tangent_scale,
    normal_scale,
):
    """Generate local (u,v,delta) samples around anchor tangent planes."""
    if anchors_xyz is None or anchors_xyz.numel() == 0:
        return None
    n = anchors_xyz.shape[0]
    spp = max(1, int(samples_per_anchor))
    device = anchors_xyz.device
    dtype = anchors_xyz.dtype

    normals = torch_F.normalize(anchors_normals, dim=-1)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)[None, :].expand(n, -1)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)[None, :].expand(n, -1)
    use_y = (normals[:, 0].abs() > 0.9)[:, None]
    ref = torch.where(use_y, ref_y, ref_x)

    t1 = torch_F.normalize(torch.cross(normals, ref, dim=-1), dim=-1)
    t2 = torch_F.normalize(torch.cross(normals, t1, dim=-1), dim=-1)

    sigma = base_scale.min(dim=1).values.clamp(min=1e-4)
    uv = (torch.rand((n, spp, 2), device=device, dtype=dtype) * 2.0 - 1.0)
    delta = (torch.rand((n, spp, 1), device=device, dtype=dtype) * 2.0 - 1.0)
    u = uv[..., 0:1] * (sigma[:, None, None] * float(tangent_scale))
    v = uv[..., 1:2] * (sigma[:, None, None] * float(tangent_scale))
    d = delta * (sigma[:, None, None] * float(normal_scale))

    pts = (
        anchors_xyz[:, None, :]
        + u * t1[:, None, :]
        + v * t2[:, None, :]
        + d * normals[:, None, :]
    )
    return pts.reshape(-1, 3)


def _concat_densify_bundles(bundle_a, bundle_b):
    if bundle_a is None:
        return bundle_b
    if bundle_b is None:
        return bundle_a
    out = {}
    for key in ("xyz", "rotations_raw", "scales", "opacities"):
        va = bundle_a.get(key)
        vb = bundle_b.get(key)
        if va is None:
            out[key] = vb
        elif vb is None:
            out[key] = va
        else:
            out[key] = torch.cat([va, vb], dim=0)
    return out


def _build_external_prior_index(
    train_cameras,
    external_prior_root,
    prior_subdir="priors",
    exts=("png", "jpg", "jpeg", "webp"),
):
    def _extract_trailing_int(stem):
        match = re.search(r"(\d+)$", stem)
        if match is None:
            return None
        return int(match.group(1))

    def _sort_key_from_stem(stem):
        idx = _extract_trailing_int(stem)
        if idx is None:
            return (1, 0, stem)
        return (0, idx, stem)

    def _sort_key_from_camera(camera):
        return _sort_key_from_stem(camera.image_name)

    root = os.path.abspath(os.path.expanduser(external_prior_root))
    candidates = []
    if prior_subdir:
        candidates.append(os.path.join(root, prior_subdir))
    candidates.append(root)

    allowed_exts = {
        str(ext).strip().lstrip(".").lower()
        for ext in exts
        if str(ext).strip()
    }
    if not allowed_exts:
        allowed_exts = {"png", "jpg", "jpeg", "webp"}

    stem_to_path = {}
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        for dirpath, _, filenames in os.walk(folder):
            for filename in sorted(filenames):
                ext = os.path.splitext(filename)[1].lstrip(".").lower()
                if ext not in allowed_exts:
                    continue
                path = os.path.join(dirpath, filename)
                stem = os.path.splitext(os.path.basename(path))[0]
                if stem not in stem_to_path:
                    stem_to_path[stem] = path

    index = {}
    used_paths = set()
    exact_count = 0
    numeric_count = 0
    order_count = 0

    # Pass 1: strict stem match (recommended path).
    for camera in train_cameras:
        image_name = camera.image_name
        path = stem_to_path.get(image_name)
        if path is None:
            continue
        if path in used_paths:
            continue
        index[image_name] = path
        used_paths.add(path)
        exact_count += 1

    unmatched = [cam for cam in train_cameras if cam.image_name not in index]

    # Pass 2: fallback by trailing frame index (e.g., r_12 <-> 0012.png).
    idx_to_paths = {}
    for stem, path in sorted(stem_to_path.items(), key=lambda x: _sort_key_from_stem(x[0])):
        idx = _extract_trailing_int(stem)
        if idx is None:
            continue
        idx_to_paths.setdefault(idx, []).append(path)

    for camera in unmatched:
        idx = _extract_trailing_int(camera.image_name)
        if idx is None:
            continue
        candidates = [p for p in idx_to_paths.get(idx, []) if p not in used_paths]
        if not candidates:
            continue
        picked = candidates[0]
        index[camera.image_name] = picked
        used_paths.add(picked)
        numeric_count += 1

    unmatched = [cam for cam in train_cameras if cam.image_name not in index]

    # Pass 3: final fallback by sorted order (for plain 0000.png style sequences).
    if unmatched:
        available_paths = []
        for stem, path in sorted(stem_to_path.items(), key=lambda x: _sort_key_from_stem(x[0])):
            if path in used_paths:
                continue
            available_paths.append(path)

        unmatched_sorted = sorted(unmatched, key=_sort_key_from_camera)
        pair_count = min(len(unmatched_sorted), len(available_paths))
        for i in range(pair_count):
            camera = unmatched_sorted[i]
            picked = available_paths[i]
            index[camera.image_name] = picked
            used_paths.add(picked)
            order_count += 1

    missing = len(train_cameras) - len(index)

    print(
        "[PRIOR] external index size: "
        f"{len(index)} matched / {len(train_cameras)} train views "
        f"(missing={missing}, root={root})"
    )
    if len(index) > 0:
        print(
            "[PRIOR] external match breakdown: "
            f"exact={exact_count}, numeric={numeric_count}, order={order_count}"
        )
    if order_count > 0:
        print(
            "[PRIOR] warning: order-based fallback was used. "
            "Verify camera/prior ordering consistency."
        )
    return index


def _parse_external_ext_tokens(spec: str):
    ext_tokens = [
        tok.strip().lstrip(".").lower()
        for tok in str(spec).split(",")
        if tok.strip()
    ]
    return ext_tokens or ["png", "jpg", "jpeg", "webp"]


def _build_optional_external_bank(
    train_cameras,
    root_dir,
    exts,
    bank_cls,
    label: str,
    fallback_roots=(),
):
    candidate_roots = []
    if root_dir:
        candidate_roots.append(root_dir)
    for candidate in fallback_roots or ():
        if not candidate:
            continue
        if any(os.path.abspath(os.path.expanduser(candidate)) == os.path.abspath(os.path.expanduser(existing)) for existing in candidate_roots):
            continue
        candidate_roots.append(candidate)

    if not candidate_roots:
        return None

    first_root = candidate_roots[0]
    for idx, candidate_root in enumerate(candidate_roots):
        external_index = _build_external_prior_index(
            train_cameras=train_cameras,
            external_prior_root=candidate_root,
            prior_subdir="",
            exts=tuple(exts),
        )
        if len(external_index) == 0:
            continue
        if idx > 0:
            print(
                f"[{label}] using fallback asset root after empty primary root: "
                f"primary={first_root} fallback={candidate_root}"
            )
        print(f"[{label}] matched {len(external_index)} train views from {candidate_root}")
        return bank_cls(external_index)

    print(f"[{label}] no usable assets found under: {first_root}")
    return None


def build_source_tag_train_mask(gaussians, optimize_source_tag: str):
    if optimize_source_tag == "all":
        return None
    source_tag = gaussians._source_tag
    if optimize_source_tag == "prior":
        return source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    if optimize_source_tag == "probe":
        return source_tag == int(GaussianSourceTag.EXTENSION_PROBE)
    if optimize_source_tag == "added":
        return source_tag != int(GaussianSourceTag.ORIGINAL)
    raise ValueError(f"Unsupported optimize_source_tag: {optimize_source_tag}")


def count_trainable_source_tag_gaussians(gaussians, source_update_mask) -> int:
    if source_update_mask is None:
        return int(gaussians.get_xyz.shape[0])
    return int(source_update_mask.sum().item())


def _select_evenly_spaced_views(cameras, max_views: int):
    cameras = list(cameras)
    max_views = int(max_views)
    if max_views <= 0 or len(cameras) <= max_views:
        return cameras
    ids = np.unique(np.linspace(0, len(cameras) - 1, num=max_views, dtype=np.int64))
    return [cameras[int(idx)] for idx in ids.tolist()]


def _build_prior_view_curriculum_cameras(active_train_cameras, prior_bank, surface_route_bank, max_views: int):
    cameras = list(active_train_cameras)
    if prior_bank is not None and getattr(prior_bank, "index", None):
        cameras = [camera for camera in cameras if camera.image_name in prior_bank.index]
    elif surface_route_bank is not None:
        cameras = [camera for camera in cameras if surface_route_bank.has_view(camera.image_name)]
    return _select_evenly_spaced_views(cameras, int(max_views))


def _prior_view_curriculum_state(cameras, iteration: int, args):
    if not bool(getattr(args, "prior_view_curriculum_enable", False)):
        return None
    if len(cameras) <= 0:
        return None
    start_iter = int(getattr(args, "prior_view_curriculum_start_iter", 0))
    if start_iter > 0 and int(iteration) < start_iter:
        return None
    primary_iters = max(0, int(getattr(args, "prior_view_curriculum_primary_iters", 0)))
    neighbor_iters = max(0, int(getattr(args, "prior_view_curriculum_neighbor_iters", 0)))
    episode_iters = primary_iters + neighbor_iters
    if episode_iters <= 0:
        return None

    offset = int(iteration) - max(start_iter, 0)
    if offset < 0:
        return None
    coverage_iters = len(cameras) * episode_iters
    if offset >= coverage_iters:
        settle_iters = int(getattr(args, "prior_view_curriculum_settle_iters", 0))
        if settle_iters <= 0 or offset < coverage_iters + settle_iters:
            idx = (offset - coverage_iters) % len(cameras)
            return {
                "phase": "settle",
                "camera": cameras[int(idx)],
                "primary_index": -1,
                "local_iter": int(offset - coverage_iters),
                "view_done": len(cameras),
                "view_total": len(cameras),
                "hf_scale": float(getattr(args, "prior_view_curriculum_settle_hf_scale", 1.0)),
            }
        return None

    primary_index = int(offset // episode_iters)
    local_iter = int(offset % episode_iters)
    if local_iter < primary_iters:
        phase = "primary"
        camera_index = primary_index
        hf_scale = float(getattr(args, "prior_view_curriculum_primary_hf_scale", 1.0))
    else:
        phase = "neighbor"
        radius = max(0, int(getattr(args, "prior_view_curriculum_neighbor_radius", 0)))
        neighbor_indices = [
            idx
            for idx in range(primary_index - radius, primary_index + radius + 1)
            if 0 <= idx < len(cameras) and idx != primary_index
        ]
        if neighbor_indices:
            neighbor_local = local_iter - primary_iters
            camera_index = int(neighbor_indices[int(neighbor_local) % len(neighbor_indices)])
        else:
            camera_index = primary_index
        hf_scale = float(getattr(args, "prior_view_curriculum_neighbor_hf_scale", 1.0))

    return {
        "phase": phase,
        "camera": cameras[int(camera_index)],
        "primary_index": int(primary_index),
        "local_iter": int(local_iter),
        "view_done": int(primary_index),
        "view_total": len(cameras),
        "hf_scale": hf_scale,
    }


def load_gaussian_update_mask_payload(path: str, key: str, total_gaussians: int):
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    if key in payload:
        mask = payload[key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        if mask.ndim != 1 or mask.shape[0] != int(total_gaussians):
            raise ValueError(
                f"Gaussian update mask '{key}' length mismatch: "
                f"{tuple(mask.shape)} vs total_gaussians={total_gaussians}"
            )
        return mask.to(device="cuda", dtype=torch.bool)
    if "class_id" in payload:
        class_id = payload["class_id"]
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)
        class_id = class_id.reshape(-1).to(dtype=torch.long)
        if class_id.shape[0] != int(total_gaussians):
            raise ValueError(
                f"Gaussian class_id length mismatch: {tuple(class_id.shape)} "
                f"vs total_gaussians={total_gaussians}"
            )
        derived_mask = None
        if key == "no_mesh_neutral":
            derived_mask = class_id == 0
        elif key == "surface_carrier":
            derived_mask = class_id == 1
        elif key == "near_surface_uncertain":
            derived_mask = class_id == 2
        elif key == "off_surface_near_mesh":
            derived_mask = class_id == 3
        elif key == "axis_touching_surface":
            derived_mask = class_id == 4
        elif key == "low_opacity_neutral":
            derived_mask = class_id == 5
        elif key == "surface_candidate":
            derived_mask = (class_id == 1) | (class_id == 4)
        elif key == "surface_or_uncertain":
            derived_mask = (class_id == 1) | (class_id == 2) | (class_id == 4)
        elif key in {"non_surface_active", "surface_complement_active", "near_or_off_surface"}:
            low_opacity = class_id == 5
            surface_candidate = (class_id == 1) | (class_id == 4)
            derived_mask = (~low_opacity) & (~surface_candidate)
        elif key == "uncertain_or_off_surface":
            derived_mask = (class_id == 2) | (class_id == 3)
        if derived_mask is not None:
            print(
                f"[SOF-PRIOR] derived gaussian mask '{key}' from class_id in payload: {path}"
            )
            return derived_mask.to(device="cuda", dtype=torch.bool)
    if "selected_ids" in payload:
        ids = payload["selected_ids"]
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        ids = ids.to(dtype=torch.int64)
        mask = torch.zeros((int(total_gaussians),), dtype=torch.bool)
        mask[ids] = True
        return mask.to(device="cuda")
    raise KeyError(f"Mask payload must contain '{key}' or 'selected_ids': {path}")


def build_prior_edge_camera_pool(train_cameras, prior_index, mask_index_or_dir):
    cameras = []
    missing_prior = []
    missing_mask = []
    if isinstance(mask_index_or_dir, dict):
        mask_index = mask_index_or_dir
    else:
        mask_index = _build_external_prior_index(
            train_cameras=train_cameras,
            external_prior_root=mask_index_or_dir,
            prior_subdir="",
            exts=("png", "jpg", "jpeg", "webp"),
        )
    for camera in train_cameras:
        image_name = camera.image_name
        if prior_index.get(image_name) is None:
            if len(missing_prior) < 16:
                missing_prior.append(image_name)
            continue
        if mask_index.get(image_name) is None:
            if len(missing_mask) < 16:
                missing_mask.append(image_name)
            continue
        cameras.append(camera)
    return cameras, missing_prior, missing_mask


def build_masked_residual_prior_camera_pool(
    train_cameras,
    prior_bank,
    anchor_bank,
    soft_mask_bank,
    hard_mask_bank,
    hard_threshold: float,
    min_pixels: float,
):
    if prior_bank is None or anchor_bank is None:
        return [], []

    use_external_masks = soft_mask_bank is not None and hard_mask_bank is not None
    cameras = []
    skipped = []
    for camera in train_cameras:
        image_name = camera.image_name
        if image_name not in prior_bank.index or image_name not in anchor_bank.index:
            if len(skipped) < 16:
                skipped.append((image_name, "missing_asset"))
            continue
        if not use_external_masks:
            cameras.append(camera)
            continue
        if image_name not in soft_mask_bank.index or image_name not in hard_mask_bank.index:
            if len(skipped) < 16:
                skipped.append((image_name, "missing_mask"))
            continue
        soft = soft_mask_bank._load_tensor(image_name)
        hard = hard_mask_bank._load_tensor(image_name)
        if soft is None or hard is None:
            if len(skipped) < 16:
                skipped.append((image_name, "load_failed"))
            continue
        valid_pixels = ((hard >= float(hard_threshold)) & (soft > 0)).float().sum().item()
        if valid_pixels < float(min_pixels):
            if len(skipped) < 16:
                skipped.append((image_name, f"empty_or_tiny:{int(valid_pixels)}"))
            continue
        cameras.append(camera)
    return cameras, skipped


def combine_gaussian_update_masks(*masks):
    active_masks = [mask for mask in masks if mask is not None]
    if not active_masks:
        return None
    combined = active_masks[0].to(device="cuda", dtype=torch.bool)
    for mask in active_masks[1:]:
        mask = mask.to(device="cuda", dtype=torch.bool)
        if mask.shape[0] != combined.shape[0]:
            raise ValueError(f"Gaussian update mask length mismatch: {mask.shape[0]} vs {combined.shape[0]}")
        combined = combined & mask
    return combined


def _clone_frozen_tensor(tensor):
    if not torch.is_tensor(tensor):
        return tensor
    cloned = tensor.detach().clone().to(device="cuda")
    if cloned.is_floating_point():
        cloned.requires_grad_(False)
    return cloned


def load_frozen_gaussian_checkpoint(path: str, sh_degree: int):
    model_args, iteration = torch.load(path)
    tracking_state = None
    filter_3d = None
    if len(model_args) == 12:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            max_radii2d,
            _xyz_gradient_accum,
            _denom,
            _opt_dict,
            spatial_lr_scale,
        ) = model_args
    elif len(model_args) == 13:
        if isinstance(model_args[-1], dict):
            (
                active_sh_degree,
                xyz,
                features_dc,
                features_rest,
                scaling,
                rotation,
                opacity,
                max_radii2d,
                _xyz_gradient_accum,
                _denom,
                _opt_dict,
                spatial_lr_scale,
                tracking_state,
            ) = model_args
        else:
            (
                active_sh_degree,
                xyz,
                features_dc,
                features_rest,
                scaling,
                rotation,
                opacity,
                max_radii2d,
                _xyz_gradient_accum,
                _denom,
                _opt_dict,
                spatial_lr_scale,
                filter_3d,
            ) = model_args
    elif len(model_args) == 14:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            max_radii2d,
            _xyz_gradient_accum,
            _denom,
            _opt_dict,
            spatial_lr_scale,
            filter_3d,
            tracking_state,
        ) = model_args
    else:
        raise ValueError(f"Unsupported Gaussian checkpoint payload length: {len(model_args)}")

    frozen = GaussianModel(sh_degree)
    frozen.active_sh_degree = active_sh_degree
    frozen._xyz = _clone_frozen_tensor(xyz)
    frozen._features_dc = _clone_frozen_tensor(features_dc)
    frozen._features_rest = _clone_frozen_tensor(features_rest)
    frozen._scaling = _clone_frozen_tensor(scaling)
    frozen._rotation = _clone_frozen_tensor(rotation)
    frozen._opacity = _clone_frozen_tensor(opacity)
    frozen.max_radii2D = _clone_frozen_tensor(max_radii2d)
    frozen.spatial_lr_scale = spatial_lr_scale
    if filter_3d is None:
        frozen.filter_3D = torch.zeros(
            (frozen._xyz.shape[0], 1),
            dtype=frozen._xyz.dtype,
            device=frozen._xyz.device,
        )
    else:
        frozen.filter_3D = _clone_frozen_tensor(filter_3d)
    if tracking_state is None:
        frozen.init_tracking_state(frozen._xyz.shape[0])
    else:
        frozen.restore_tracking_state(tracking_state)
    frozen.optimizer = None
    return frozen, int(iteration)


def apply_gaussian_update_mask(gaussians, update_mask: torch.Tensor, update_scale: float = 1.0):
    if update_mask is None:
        return

    optimizer = gaussians.optimizer
    gaussian_group_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
    update_mask = update_mask.to(device="cuda", dtype=torch.bool)
    if update_mask.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(
            f"Gaussian update mask length mismatch: {update_mask.shape[0]} vs {gaussians.get_xyz.shape[0]}. "
            "Disable densification/pruning or regenerate the mask for this checkpoint."
        )
    inverse_mask = ~update_mask

    for group in optimizer.param_groups:
        name = group.get("name", "")
        for param in group["params"]:
            if param.grad is not None and name in gaussian_group_names and param.grad.shape[0] == update_mask.shape[0]:
                if float(update_scale) != 1.0:
                    param.grad[update_mask] *= float(update_scale)
                param.grad[inverse_mask] = 0


def initialize_gaussian_root_ids_for_dynamic_masks(gaussians, *, label: str = "dynamic mask") -> int:
    total = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    if not hasattr(gaussians, "_root_id") or int(gaussians._root_id.shape[0]) != total:
        gaussians._root_id = torch.full((total,), -1, dtype=torch.int64, device=device)
    root_id = gaussians._root_id.to(device=device, dtype=torch.int64)
    missing = root_id < 0
    initialized = int(missing.sum().item())
    if initialized > 0:
        root_id[missing] = torch.arange(total, device=device, dtype=torch.int64)[missing]
        gaussians._root_id = root_id
        print(f"[ROOT-MASK] initialized {initialized}/{total} gaussian roots for {label}.")
    return initialized


def _expand_root_aligned_mask(base_mask: torch.Tensor | None, gaussians, *, label: str):
    if base_mask is None:
        return None
    total = int(gaussians.get_xyz.shape[0])
    base_mask = base_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    if hasattr(gaussians, "_root_id") and int(gaussians._root_id.shape[0]) == total:
        root_id = gaussians._root_id.to(device=gaussians.get_xyz.device, dtype=torch.long)
        valid = (root_id >= 0) & (root_id < int(base_mask.shape[0]))
        expanded = torch.zeros((total,), device=base_mask.device, dtype=torch.bool)
        if torch.any(valid):
            expanded[valid] = base_mask[root_id[valid]]
        return expanded
    if int(base_mask.shape[0]) == total:
        return base_mask
    raise RuntimeError(
        f"{label} requires gaussian root_id tracking after densification. "
        f"base_mask={int(base_mask.shape[0])}, current={total}."
    )


def _expand_root_aligned_values(base_values: torch.Tensor | None, gaussians, *, label: str):
    if base_values is None:
        return None
    total = int(gaussians.get_xyz.shape[0])
    base_values = base_values.to(device=gaussians.get_xyz.device)
    if hasattr(gaussians, "_root_id") and int(gaussians._root_id.shape[0]) == total:
        root_id = gaussians._root_id.to(device=gaussians.get_xyz.device, dtype=torch.long)
        valid = (root_id >= 0) & (root_id < int(base_values.shape[0]))
        expanded = torch.zeros(
            (total, *base_values.shape[1:]),
            device=base_values.device,
            dtype=base_values.dtype,
        )
        if torch.any(valid):
            expanded[valid] = base_values[root_id[valid]]
        return expanded
    if int(base_values.shape[0]) == total:
        return base_values
    raise RuntimeError(
        f"{label} requires gaussian root_id tracking after densification. "
        f"base_values={int(base_values.shape[0])}, current={total}."
    )


def _normalize_3d(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp(min=eps)


def _infer_gaussian_thin_axis_normals(gaussians) -> torch.Tensor:
    scales = gaussians.get_scaling.detach()
    rotations = build_rotation(gaussians._rotation.detach())
    thin_axis = torch.argmin(scales, dim=1)
    gather_idx = thin_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(rotations, dim=2, index=gather_idx).squeeze(2)
    return _normalize_3d(normals)


def _load_surface_normal_lock_payload(path: str, normal_key: str, anchor_key: str, total_gaussians: int):
    if not path:
        return None, None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Surface normal-lock payload not found: {path}")
    payload = torch.load(path, map_location="cpu")
    normals = payload.get(normal_key)
    anchors = payload.get(anchor_key)
    loaded_normals = None
    loaded_anchors = None
    if normals is not None:
        if not torch.is_tensor(normals):
            normals = torch.as_tensor(normals)
        normals = normals.reshape(-1, 3)
        if int(normals.shape[0]) == int(total_gaussians):
            loaded_normals = _normalize_3d(normals.to(device="cuda", dtype=torch.float32))
        else:
            print(
                "[SURFACE-NORMAL-LOCK] ignore payload normals with mismatched length: "
                f"{tuple(normals.shape)} vs total_gaussians={int(total_gaussians)}"
            )
    if anchors is not None:
        if not torch.is_tensor(anchors):
            anchors = torch.as_tensor(anchors)
        anchors = anchors.reshape(-1, 3)
        if int(anchors.shape[0]) == int(total_gaussians):
            loaded_anchors = anchors.to(device="cuda", dtype=torch.float32)
        else:
            print(
                "[SURFACE-NORMAL-LOCK] ignore payload anchors with mismatched length: "
                f"{tuple(anchors.shape)} vs total_gaussians={int(total_gaussians)}"
            )
    return loaded_normals, loaded_anchors


def _build_nearest_surface_migration_anchors(
    gaussians,
    migration_mask: torch.Tensor,
    surface_mask: torch.Tensor,
    payload_normals: torch.Tensor | None = None,
):
    migration_mask = migration_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    surface_mask = surface_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    total = int(gaussians.get_xyz.shape[0])
    if int(migration_mask.shape[0]) != total or int(surface_mask.shape[0]) != total:
        raise ValueError("Nearest-surface migration fallback masks must align with the current Gaussian count.")
    if int(migration_mask.sum().item()) <= 0:
        raise ValueError("Nearest-surface migration fallback has an empty migration mask.")
    if int(surface_mask.sum().item()) <= 0:
        raise ValueError("Nearest-surface migration fallback has an empty surface anchor source mask.")

    xyz = gaussians.get_xyz.detach()
    migration_ids = torch.nonzero(migration_mask, as_tuple=False).reshape(-1)
    surface_ids = torch.nonzero(surface_mask, as_tuple=False).reshape(-1)
    query_np = xyz[migration_ids].detach().cpu().numpy().astype(np.float32, copy=False)
    surface_np = xyz[surface_ids].detach().cpu().numpy().astype(np.float32, copy=False)
    _, nearest_local = cKDTree(surface_np).query(query_np, k=1, workers=1)
    nearest_local = np.asarray(nearest_local, dtype=np.int64).reshape(-1)
    nearest_surface_ids = surface_ids[torch.from_numpy(nearest_local).to(device=surface_ids.device)]

    anchors = xyz.detach().clone()
    anchors[migration_ids] = xyz[nearest_surface_ids]
    inferred_normals = _infer_gaussian_thin_axis_normals(gaussians)
    normals = inferred_normals.detach().clone()
    if payload_normals is not None and int(payload_normals.shape[0]) == total:
        source_normals = payload_normals.to(device=xyz.device, dtype=torch.float32)
    else:
        source_normals = inferred_normals
    normals[migration_ids] = source_normals[nearest_surface_ids]
    return _normalize_3d(normals), anchors


class SurfaceNormalLock:
    def __init__(
        self,
        gaussians,
        mask: torch.Tensor,
        normals: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        dynamic_roots: bool = False,
    ):
        total = int(gaussians.get_xyz.shape[0])
        mask = mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
        if int(mask.shape[0]) != total:
            raise ValueError(f"Surface normal-lock mask length mismatch: {int(mask.shape[0])} vs {total}")
        if int(mask.sum().item()) <= 0:
            raise ValueError("Surface normal-lock mask is empty.")
        if normals is None:
            normals = _infer_gaussian_thin_axis_normals(gaussians)
            normal_source = "thin_axis"
        else:
            normals = _normalize_3d(normals.to(device=gaussians.get_xyz.device, dtype=torch.float32))
            normal_source = "payload"
        if int(normals.shape[0]) != total:
            raise ValueError(f"Surface normal-lock normal length mismatch: {int(normals.shape[0])} vs {total}")
        if anchors is None:
            anchors = gaussians.get_xyz.detach().clone()
            anchor_source = "initial_xyz"
        else:
            anchors = anchors.to(device=gaussians.get_xyz.device, dtype=torch.float32)
            anchor_source = "payload"
        if int(anchors.shape[0]) != total:
            raise ValueError(f"Surface normal-lock anchor length mismatch: {int(anchors.shape[0])} vs {total}")

        self.mask = mask.detach().clone()
        self.normals = normals.detach().clone()
        self.anchors = anchors.detach().clone()
        initial_delta = gaussians.get_xyz.detach() - self.anchors
        self.initial_normal_coord = torch.sum(initial_delta * self.normals, dim=1, keepdim=True).detach().clone()
        self.total = total
        self.normal_source = normal_source
        self.anchor_source = anchor_source
        self.dynamic_roots = bool(dynamic_roots)

    def _check_count(self, gaussians):
        if self.dynamic_roots:
            return
        current = int(gaussians.get_xyz.shape[0])
        if current != int(self.total):
            raise RuntimeError(
                "Surface normal lock requires a fixed Gaussian count. "
                f"Got current={current}, locked={int(self.total)}. Disable densification/pruning for this run."
            )

    def _current_payload(self, gaussians):
        if not self.dynamic_roots:
            self._check_count(gaussians)
            return self.mask, self.normals, self.anchors, self.initial_normal_coord
        mask = _expand_root_aligned_mask(self.mask, gaussians, label="surface normal-lock mask")
        normals = _normalize_3d(
            _expand_root_aligned_values(self.normals, gaussians, label="surface normal-lock normals")
        )
        anchors = _expand_root_aligned_values(self.anchors, gaussians, label="surface normal-lock anchors")
        target_coord = _expand_root_aligned_values(
            self.initial_normal_coord,
            gaussians,
            label="surface normal-lock initial coordinate",
        )
        return mask, normals, anchors, target_coord

    def project_xyz_gradient_to_tangent(self, gaussians):
        grad = gaussians._xyz.grad
        if grad is None:
            return
        mask, normals_all, _, _ = self._current_payload(gaussians)
        selected = mask
        if int(selected.sum().item()) <= 0:
            return
        normals = normals_all[selected].to(dtype=grad.dtype)
        grad_sel = grad[selected]
        normal_grad = torch.sum(grad_sel * normals, dim=1, keepdim=True) * normals
        grad[selected] = grad_sel - normal_grad

    def project_xyz_to_locked_normal_coord(self, gaussians):
        mask, normals_all, anchors_all, target_coord_all = self._current_payload(gaussians)
        selected = mask
        if int(selected.sum().item()) <= 0:
            return
        xyz = gaussians._xyz.data
        normals = normals_all[selected].to(dtype=xyz.dtype)
        anchors = anchors_all[selected].to(dtype=xyz.dtype)
        target_coord = target_coord_all[selected].to(dtype=xyz.dtype)
        delta = xyz[selected] - anchors
        coord = torch.sum(delta * normals, dim=1, keepdim=True)
        xyz[selected] -= (coord - target_coord) * normals


def apply_prior_direct_xyz_nudge(
    gaussians,
    xyz_grad: torch.Tensor | None,
    update_mask: torch.Tensor | None,
    lr: float,
    max_step: float,
    surface_normal_lock: SurfaceNormalLock | None = None,
):
    if xyz_grad is None or update_mask is None:
        return {}
    if float(lr) <= 0.0:
        return {}

    total = int(gaussians.get_xyz.shape[0])
    update_mask = update_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    if int(update_mask.shape[0]) != total:
        raise ValueError(
            f"Prior direct xyz nudge mask length mismatch: {int(update_mask.shape[0])} vs {total}"
        )
    if int(update_mask.sum().item()) <= 0:
        return {}

    grad = xyz_grad.detach().to(device=gaussians.get_xyz.device, dtype=gaussians._xyz.data.dtype)
    if int(grad.shape[0]) != total:
        raise ValueError(f"Prior direct xyz nudge grad length mismatch: {int(grad.shape[0])} vs {total}")

    finite = torch.isfinite(grad).all(dim=1)
    selected = update_mask & finite
    if int(selected.sum().item()) <= 0:
        return {}

    step = -float(lr) * grad[selected]
    if surface_normal_lock is not None:
        lock_mask, lock_normals, _, _ = surface_normal_lock._current_payload(gaussians)
        lock_selected = lock_mask[selected].to(device=step.device, dtype=torch.bool)
        if int(lock_selected.sum().item()) > 0:
            normals = lock_normals[selected][lock_selected].to(dtype=step.dtype)
            step_locked = step[lock_selected]
            normal_step = torch.sum(step_locked * normals, dim=1, keepdim=True) * normals
            step[lock_selected] = step_locked - normal_step

    step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
    if float(max_step) > 0.0:
        scale = torch.clamp(float(max_step) / step_norm.clamp(min=1e-12), max=1.0)
        step = step * scale
        step_norm = torch.linalg.norm(step, dim=1, keepdim=True)

    gaussians._xyz.data[selected] += step
    return {
        "xyz_nudge_count": float(int(selected.sum().item())),
        "xyz_nudge_grad_mean": grad[selected].norm(dim=1).mean().item(),
        "xyz_nudge_step_mean": step_norm.mean().item(),
        "xyz_nudge_step_max": step_norm.max().item(),
    }


class SurfaceMigrationRegularizer:
    def __init__(
        self,
        gaussians,
        mask: torch.Tensor,
        normals: torch.Tensor,
        anchors: torch.Tensor,
        lambda_normal: float = 0.0,
        lambda_tangent: float = 0.0,
        lambda_normal_align: float = 0.0,
        lambda_thin: float = 0.0,
        target_normal_coord: float = 0.0,
        thin_target_ratio: float = 0.25,
        post_step_normal_alpha: float = 0.0,
        post_step_tangent_alpha: float = 0.0,
        from_iter: int = 0,
        until_iter: int = 0,
    ):
        total = int(gaussians.get_xyz.shape[0])
        mask = mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
        if int(mask.shape[0]) != total:
            raise ValueError(f"Surface migration mask length mismatch: {int(mask.shape[0])} vs {total}")
        if int(mask.sum().item()) <= 0:
            raise ValueError("Surface migration mask is empty.")
        if normals is None or anchors is None:
            raise ValueError("Surface migration requires payload anchor normals and anchor xyz.")
        normals = _normalize_3d(normals.to(device=gaussians.get_xyz.device, dtype=torch.float32))
        anchors = anchors.to(device=gaussians.get_xyz.device, dtype=torch.float32)
        if int(normals.shape[0]) != total:
            raise ValueError(f"Surface migration normal length mismatch: {int(normals.shape[0])} vs {total}")
        if int(anchors.shape[0]) != total:
            raise ValueError(f"Surface migration anchor length mismatch: {int(anchors.shape[0])} vs {total}")

        self.mask = mask.detach().clone()
        self.normals = normals.detach().clone()
        self.anchors = anchors.detach().clone()
        self.lambda_normal = float(lambda_normal)
        self.lambda_tangent = float(lambda_tangent)
        self.lambda_normal_align = float(lambda_normal_align)
        self.lambda_thin = float(lambda_thin)
        self.target_normal_coord = float(target_normal_coord)
        self.thin_target_ratio = float(thin_target_ratio)
        self.post_step_normal_alpha = float(post_step_normal_alpha)
        self.post_step_tangent_alpha = float(post_step_tangent_alpha)
        self.from_iter = int(from_iter)
        self.until_iter = int(until_iter)
        self.total = total

    def active(self, iteration: int) -> bool:
        if iteration < self.from_iter:
            return False
        if self.until_iter > 0 and iteration > self.until_iter:
            return False
        return True

    def _check_count(self, gaussians):
        current = int(gaussians.get_xyz.shape[0])
        if current != int(self.total):
            raise RuntimeError(
                "Surface migration requires a fixed Gaussian count. "
                f"Got current={current}, migration={int(self.total)}. Disable densification/pruning for this run."
            )

    def compute(self, gaussians, iteration: int):
        metrics = {}
        if not self.active(iteration):
            return None, metrics
        self._check_count(gaussians)

        selected = self.mask
        xyz = gaussians.get_xyz[selected]
        normals = self.normals[selected].to(dtype=xyz.dtype)
        anchors = self.anchors[selected].to(dtype=xyz.dtype)
        delta = xyz - anchors
        normal_coord = torch.sum(delta * normals, dim=1, keepdim=True)
        target_coord = torch.full_like(normal_coord, float(self.target_normal_coord))
        normal_residual = normal_coord - target_coord
        tangent_delta = delta - normal_coord * normals

        loss = None
        if self.lambda_normal > 0.0:
            normal_loss = normal_residual.abs().mean()
            weighted = self.lambda_normal * normal_loss
            loss = weighted if loss is None else loss + weighted
            metrics["normal"] = normal_loss.detach().item()
            metrics["normal_w"] = weighted.detach().item()
            metrics["normal_abs_mean"] = normal_coord.detach().abs().mean().item()
        if self.lambda_tangent > 0.0:
            tangent_loss = torch.linalg.norm(tangent_delta, dim=1).mean()
            weighted = self.lambda_tangent * tangent_loss
            loss = weighted if loss is None else loss + weighted
            metrics["tangent"] = tangent_loss.detach().item()
            metrics["tangent_w"] = weighted.detach().item()
        if self.lambda_normal_align > 0.0:
            scales = gaussians.get_scaling[selected]
            rotations = build_rotation(gaussians._rotation[selected])
            thin_axis = torch.argmin(scales, dim=1)
            gather_idx = thin_axis[:, None, None].expand(-1, 3, 1)
            thin_normals = torch.gather(rotations, dim=2, index=gather_idx).squeeze(2)
            align = torch.sum(_normalize_3d(thin_normals) * normals, dim=1).abs().clamp(0.0, 1.0)
            align_loss = (1.0 - align).mean()
            weighted = self.lambda_normal_align * align_loss
            loss = weighted if loss is None else loss + weighted
            metrics["align"] = align_loss.detach().item()
            metrics["align_w"] = weighted.detach().item()
        if self.lambda_thin > 0.0:
            sorted_scales, _ = torch.sort(gaussians.get_scaling[selected], dim=1)
            ratio = sorted_scales[:, 0] / sorted_scales[:, 1:].mean(dim=1).clamp(min=1e-8)
            thin_loss = torch.clamp(ratio - float(self.thin_target_ratio), min=0.0).mean()
            weighted = self.lambda_thin * thin_loss
            loss = weighted if loss is None else loss + weighted
            metrics["thin"] = thin_loss.detach().item()
            metrics["thin_w"] = weighted.detach().item()
            metrics["thin_ratio"] = ratio.detach().mean().item()

        if loss is not None:
            metrics["total"] = loss.detach().item()
        return loss, metrics

    def apply_post_step(self, gaussians, iteration: int):
        if not self.active(iteration):
            return
        if self.post_step_normal_alpha <= 0.0 and self.post_step_tangent_alpha <= 0.0:
            return
        self._check_count(gaussians)
        selected = self.mask
        xyz = gaussians._xyz.data
        normals = self.normals[selected].to(dtype=xyz.dtype)
        anchors = self.anchors[selected].to(dtype=xyz.dtype)
        delta = xyz[selected] - anchors
        normal_coord = torch.sum(delta * normals, dim=1, keepdim=True)
        target_coord = torch.full_like(normal_coord, float(self.target_normal_coord))
        if self.post_step_normal_alpha > 0.0:
            alpha = min(max(float(self.post_step_normal_alpha), 0.0), 1.0)
            xyz[selected] -= alpha * (normal_coord - target_coord) * normals
            delta = xyz[selected] - anchors
            normal_coord = torch.sum(delta * normals, dim=1, keepdim=True)
        if self.post_step_tangent_alpha > 0.0:
            alpha = min(max(float(self.post_step_tangent_alpha), 0.0), 1.0)
            tangent_delta = delta - normal_coord * normals
            xyz[selected] -= alpha * tangent_delta


class SurfaceFilter3D:
    def __init__(self, mask: torch.Tensor, scale: float, min_value: float = 0.0):
        if float(scale) < 1.0:
            raise ValueError("surface_filter_3d_scale must be >= 1.0.")
        mask = mask.reshape(-1).to(device="cuda", dtype=torch.bool)
        if int(mask.sum().item()) <= 0:
            raise ValueError("Surface 3D filter mask is empty.")
        self.mask = mask.detach().clone()
        self.scale = float(scale)
        self.min_value = float(min_value)

    def apply_after_compute(self, gaussians):
        total = int(gaussians.get_xyz.shape[0])
        if int(self.mask.shape[0]) != total:
            raise RuntimeError(
                "Surface 3D filter requires a fixed Gaussian count. "
                f"mask={int(self.mask.shape[0])}, current={total}. "
                "Disable densification/pruning or regenerate the mask for the current model."
            )
        if not hasattr(gaussians, "filter_3D") or gaussians.filter_3D is None:
            return
        selected = self.mask.to(device=gaussians.filter_3D.device, dtype=torch.bool)
        values = gaussians.filter_3D[selected] * self.scale
        if self.min_value > 0.0:
            values = torch.maximum(values, torch.full_like(values, self.min_value))
        gaussians.filter_3D[selected] = values


def compute_3d_filter_with_surface_filter(gaussians, cameras, surface_filter_3d=None):
    gaussians.compute_3D_filter(cameras=cameras)
    if surface_filter_3d is not None:
        surface_filter_3d.apply_after_compute(gaussians)


def _iter_gaussian_maskable_params(gaussians):
    gaussian_group_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
    total_gaussians = int(gaussians.get_xyz.shape[0])
    for group in gaussians.optimizer.param_groups:
        name = group.get("name", "")
        if name not in gaussian_group_names:
            continue
        for param in group["params"]:
            if param.requires_grad and param.shape and param.shape[0] == total_gaussians:
                yield name, param


def accumulate_masked_gaussian_loss_grads(
    gaussians,
    loss: torch.Tensor,
    update_mask: torch.Tensor,
    update_scale: float = 1.0,
    retain_graph: bool = False,
):
    if loss is None or update_mask is None:
        return
    update_mask = update_mask.to(device="cuda", dtype=torch.bool)
    inverse_mask = ~update_mask
    params = [param for _, param in _iter_gaussian_maskable_params(gaussians)]
    if not params:
        return
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    scale = float(update_scale)
    for param, grad in zip(params, grads):
        if grad is None:
            continue
        if grad.shape[0] != update_mask.shape[0]:
            continue
        masked_grad = grad.clone()
        masked_grad[inverse_mask] = 0
        if scale != 1.0:
            masked_grad[update_mask] *= scale
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.add_(masked_grad)


def accumulate_masked_viewspace_loss_grads(
    loss: torch.Tensor,
    viewspace_point_tensor: torch.Tensor,
    update_mask: torch.Tensor,
    update_scale: float = 1.0,
    retain_graph: bool = False,
):
    if loss is None or viewspace_point_tensor is None or update_mask is None:
        return
    update_mask = update_mask.to(device=viewspace_point_tensor.device, dtype=torch.bool)
    grad = torch.autograd.grad(
        loss,
        viewspace_point_tensor,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if grad is None or grad.shape[0] != update_mask.shape[0]:
        return
    masked_grad = grad.clone()
    masked_grad[~update_mask] = 0
    scale = float(update_scale)
    if scale != 1.0:
        masked_grad[update_mask] *= scale
    if viewspace_point_tensor.grad is None:
        viewspace_point_tensor.grad = torch.zeros_like(viewspace_point_tensor)
    viewspace_point_tensor.grad.add_(masked_grad)


class GaussianSubsetView:
    def __init__(self, base_gaussians, subset_mask: torch.Tensor):
        subset_mask = subset_mask.to(device=base_gaussians.get_xyz.device, dtype=torch.bool)
        if subset_mask.ndim != 1 or subset_mask.shape[0] != base_gaussians.get_xyz.shape[0]:
            raise ValueError(
                "GaussianSubsetView mask must be a 1D tensor aligned with the current Gaussian count."
            )
        self.base_gaussians = base_gaussians
        self.subset_mask = subset_mask

    @property
    def active_sh_degree(self):
        return self.base_gaussians.active_sh_degree

    @property
    def max_sh_degree(self):
        return self.base_gaussians.max_sh_degree

    @property
    def get_xyz(self):
        return self.base_gaussians.get_xyz[self.subset_mask]

    @property
    def filter_3D(self):
        return self.base_gaussians.filter_3D[self.subset_mask]

    @property
    def get_scaling(self):
        return self.base_gaussians.get_scaling[self.subset_mask]

    @property
    def get_opacity(self):
        return self.base_gaussians.get_opacity[self.subset_mask]

    @property
    def get_features(self):
        return self.base_gaussians.get_features[self.subset_mask]

    @property
    def get_opacity_with_3D_filter(self):
        return self.base_gaussians.get_opacity_with_3D_filter[self.subset_mask]

    @property
    def get_scaling_with_3D_filter(self):
        return self.base_gaussians.get_scaling_with_3D_filter[self.subset_mask]

    @property
    def get_rotation(self):
        return self.base_gaussians.get_rotation[self.subset_mask]

    def get_covariance(self, scaling_modifier=1):
        return self.base_gaussians.get_covariance(scaling_modifier)[self.subset_mask]

    def get_view2gaussian(self, viewmatrix):
        return self.base_gaussians.get_view2gaussian(viewmatrix)[self.subset_mask]


class LayerFrequencyRegularizer:
    def __init__(
        self,
        gaussians,
        non_surface_mask=None,
        surface_mask=None,
        lambda_non_surface_hf: float = 0.0,
        lambda_non_surface_rgb_energy: float = 0.0,
        lambda_non_surface_alpha_hf: float = 0.0,
        lambda_non_surface_alpha_mass: float = 0.0,
        lambda_surface_hf_closure: float = 0.0,
        lambda_surface_start_hf_preserve: float = 0.0,
        start_hf_lowfreq_kernel: int = 15,
        start_hf_lowfreq_threshold: float = 0.05,
        start_hf_energy_threshold: float = 0.01,
        start_hf_mask_power: float = 1.0,
        start_hf_protect_non_surface: bool = False,
        from_iter: int = 0,
        until_iter: int = 0,
        dynamic_roots: bool = False,
    ):
        self.gaussians = gaussians
        self.lambda_non_surface_hf = float(lambda_non_surface_hf)
        self.lambda_non_surface_rgb_energy = float(lambda_non_surface_rgb_energy)
        self.lambda_non_surface_alpha_hf = float(lambda_non_surface_alpha_hf)
        self.lambda_non_surface_alpha_mass = float(lambda_non_surface_alpha_mass)
        self.lambda_surface_hf_closure = float(lambda_surface_hf_closure)
        self.lambda_surface_start_hf_preserve = float(lambda_surface_start_hf_preserve)
        self.start_hf_lowfreq_kernel = int(start_hf_lowfreq_kernel)
        self.start_hf_lowfreq_threshold = float(start_hf_lowfreq_threshold)
        self.start_hf_energy_threshold = float(start_hf_energy_threshold)
        self.start_hf_mask_power = float(start_hf_mask_power)
        self.start_hf_protect_non_surface = bool(start_hf_protect_non_surface)
        self.from_iter = int(from_iter)
        self.until_iter = int(until_iter)
        self.dynamic_roots = bool(dynamic_roots)
        self.surface_mask = None
        self.non_surface_gaussians = None
        self.base_non_surface_mask = None
        self.base_surface_mask = None
        self.has_non_surface_mask = False
        self.has_surface_mask = False

        if (
            self.lambda_non_surface_hf > 0.0
            or self.lambda_non_surface_rgb_energy > 0.0
            or self.lambda_non_surface_alpha_hf > 0.0
            or self.lambda_non_surface_alpha_mass > 0.0
        ):
            if non_surface_mask is not None:
                non_surface_mask = non_surface_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool)
                selected = int(non_surface_mask.sum().item())
                if selected > 0:
                    self.base_non_surface_mask = non_surface_mask.detach().clone()
                    self.has_non_surface_mask = True
                    if not self.dynamic_roots:
                        if selected == int(non_surface_mask.shape[0]):
                            self.non_surface_gaussians = gaussians
                        else:
                            self.non_surface_gaussians = GaussianSubsetView(gaussians, non_surface_mask)

        if (
            self.lambda_surface_hf_closure > 0.0
            or self.lambda_surface_start_hf_preserve > 0.0
        ) and surface_mask is not None:
            surface_mask = surface_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool)
            if int(surface_mask.sum().item()) > 0:
                self.base_surface_mask = surface_mask.detach().clone()
                self.has_surface_mask = True
                self.surface_mask = surface_mask if not self.dynamic_roots else surface_mask.detach().clone()

    def active(self, iteration: int) -> bool:
        if iteration < self.from_iter:
            return False
        if self.until_iter > 0 and iteration > self.until_iter:
            return False
        return True

    def _current_mask(self, base_mask, *, label: str):
        if base_mask is None:
            return None
        if self.dynamic_roots:
            return _expand_root_aligned_mask(base_mask, self.gaussians, label=label)
        total = int(self.gaussians.get_xyz.shape[0])
        if int(base_mask.shape[0]) != total:
            raise RuntimeError(
                f"{label} is fixed-topology but gaussian count changed: "
                f"mask={int(base_mask.shape[0])}, current={total}. "
                "Enable --layer_frequency_dynamic_roots or disable densification/pruning."
            )
        return base_mask

    def _current_non_surface_gaussians(self, metrics):
        if self.dynamic_roots:
            mask = self._current_mask(self.base_non_surface_mask, label="layer-frequency non-surface mask")
            if mask is None:
                return None
            selected = int(mask.sum().item())
            metrics["ns_count"] = float(selected)
            if selected <= 0:
                return None
            if selected == int(mask.shape[0]):
                return self.gaussians
            return GaussianSubsetView(self.gaussians, mask)
        return self.non_surface_gaussians

    def compute(
        self,
        iteration: int,
        viewpoint_camera,
        gt_image,
        render_image,
        render_fn,
        pipe,
        background,
        kernel_size,
        subpixel_offset=None,
        start_image=None,
    ):
        metrics = {}
        direct_loss = None
        surface_loss = None
        if not self.active(iteration):
            return direct_loss, surface_loss, metrics

        start_hf = None
        reliable_weight = None
        if (
            start_image is not None
            and (
                self.lambda_surface_start_hf_preserve > 0.0
                or self.start_hf_protect_non_surface
            )
        ):
            start_image = start_image.detach().to(
                device=render_image.device,
                dtype=render_image.dtype,
            )
            if start_image.shape[-2:] != render_image.shape[-2:]:
                start_image = torch_F.interpolate(
                    start_image[None],
                    size=render_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            start_hf = _laplacian_highfreq(start_image).detach()
            start_low = _box_lowpass(start_image, self.start_hf_lowfreq_kernel)
            gt_low = _box_lowpass(gt_image.detach(), self.start_hf_lowfreq_kernel)
            low_err = (start_low - gt_low).abs().mean(dim=0, keepdim=True)
            start_energy = start_hf.abs().mean(dim=0, keepdim=True)
            if self.start_hf_lowfreq_threshold > 0.0:
                low_weight = (1.0 - low_err / self.start_hf_lowfreq_threshold).clamp(0.0, 1.0)
            else:
                low_weight = torch.ones_like(low_err)
            if self.start_hf_energy_threshold > 0.0:
                energy_weight = (start_energy / self.start_hf_energy_threshold).clamp(0.0, 1.0)
            else:
                energy_weight = torch.ones_like(start_energy)
            reliable_weight = (low_weight * energy_weight).detach()
            if self.start_hf_mask_power != 1.0:
                reliable_weight = reliable_weight.clamp_min(0.0).pow(self.start_hf_mask_power)
            metrics["start_mask"] = reliable_weight.mean().detach().item()
            metrics["start_lowerr"] = low_err.mean().detach().item()
            metrics["start_energy"] = start_energy.mean().detach().item()

        non_surface_gaussians = self._current_non_surface_gaussians(metrics)
        if non_surface_gaussians is not None:
            black_background = torch.zeros_like(background)
            non_surface_pkg = render_fn(
                viewpoint_camera,
                non_surface_gaussians,
                pipe,
                black_background,
                kernel_size=kernel_size,
                subpixel_offset=subpixel_offset,
            )
            non_surface_image = non_surface_pkg["render"]
            if self.lambda_non_surface_rgb_energy > 0.0:
                loss_non_surface_rgb = non_surface_image.abs().mean()
                weighted = self.lambda_non_surface_rgb_energy * loss_non_surface_rgb
                direct_loss = weighted if direct_loss is None else direct_loss + weighted
                metrics["ns_rgb"] = loss_non_surface_rgb.detach().item()
                metrics["ns_rgb_w"] = weighted.detach().item()
            if self.lambda_non_surface_hf > 0.0:
                non_surface_hf_abs = _laplacian_highfreq(non_surface_image).abs()
                if self.start_hf_protect_non_surface and reliable_weight is not None:
                    penalty_weight = (1.0 - reliable_weight).expand_as(non_surface_hf_abs)
                    penalty_mean = penalty_weight.mean()
                    if float(penalty_mean.detach().item()) > 1e-8:
                        loss_non_surface_hf = (non_surface_hf_abs * penalty_weight).mean() / penalty_mean.clamp_min(1e-8)
                    else:
                        loss_non_surface_hf = non_surface_hf_abs.sum() * 0.0
                    metrics["ns_hf_protect"] = reliable_weight.mean().detach().item()
                else:
                    loss_non_surface_hf = non_surface_hf_abs.mean()
                weighted = self.lambda_non_surface_hf * loss_non_surface_hf
                direct_loss = weighted if direct_loss is None else direct_loss + weighted
                metrics["ns_hf"] = loss_non_surface_hf.detach().item()
                metrics["ns_hf_w"] = weighted.detach().item()
            non_surface_alpha = non_surface_pkg.get("alpha")
            if non_surface_alpha is not None:
                if self.lambda_non_surface_alpha_hf > 0.0:
                    loss_alpha_hf = _laplacian_highfreq(non_surface_alpha).abs().mean()
                    weighted = self.lambda_non_surface_alpha_hf * loss_alpha_hf
                    direct_loss = weighted if direct_loss is None else direct_loss + weighted
                    metrics["ns_alpha_hf"] = loss_alpha_hf.detach().item()
                    metrics["ns_alpha_hf_w"] = weighted.detach().item()
                if self.lambda_non_surface_alpha_mass > 0.0:
                    loss_alpha_mass = non_surface_alpha.mean()
                    weighted = self.lambda_non_surface_alpha_mass * loss_alpha_mass
                    direct_loss = weighted if direct_loss is None else direct_loss + weighted
                    metrics["ns_alpha"] = loss_alpha_mass.detach().item()
                    metrics["ns_alpha_w"] = weighted.detach().item()
            elif self.lambda_non_surface_alpha_hf > 0.0 or self.lambda_non_surface_alpha_mass > 0.0:
                metrics["ns_alpha_missing"] = 1.0

        surface_mask = self._current_mask(self.base_surface_mask, label="layer-frequency surface mask")
        self.surface_mask = surface_mask
        if surface_mask is not None:
            metrics["surf_count"] = float(int(surface_mask.sum().item()))
        if surface_mask is not None and int(surface_mask.sum().item()) > 0:
            target_hf = _laplacian_highfreq(gt_image.detach())
            pred_hf = _laplacian_highfreq(render_image)
            if self.lambda_surface_hf_closure > 0.0:
                loss_surface_hf = (pred_hf - target_hf).abs().mean()
                surface_loss = self.lambda_surface_hf_closure * loss_surface_hf
                metrics["surf_hf"] = loss_surface_hf.detach().item()
                metrics["surf_hf_w"] = surface_loss.detach().item()
            if (
                self.lambda_surface_start_hf_preserve > 0.0
                and start_hf is not None
                and reliable_weight is not None
            ):
                expanded_weight = reliable_weight.expand_as(pred_hf)
                weight_mean = expanded_weight.mean()
                if float(weight_mean.detach().item()) > 1e-8:
                    loss_start_hf = ((pred_hf - start_hf).abs() * expanded_weight).mean() / weight_mean.clamp_min(1e-8)
                    weighted = self.lambda_surface_start_hf_preserve * loss_start_hf
                    surface_loss = weighted if surface_loss is None else surface_loss + weighted
                    metrics["start_hf"] = loss_start_hf.detach().item()
                    metrics["start_hf_w"] = weighted.detach().item()

        weighted_total = torch.zeros((), dtype=render_image.dtype, device=render_image.device)
        if direct_loss is not None:
            weighted_total = weighted_total + direct_loss.detach()
        if surface_loss is not None:
            weighted_total = weighted_total + surface_loss.detach()
        if direct_loss is not None or surface_loss is not None:
            metrics["total_w"] = weighted_total.item()
        return direct_loss, surface_loss, metrics


class PriorBubbleCleanupRegularizer:
    """NoSR-style dynamic cleanup for round PRIOR_INJECTED bubble carriers."""

    def __init__(
        self,
        *,
        opacity_min: float = 0.10,
        max_axis_min: float = 0.0,
        max_axis_max: float = 0.0,
        anisotropy_max: float = 2.5,
        min_generation: int = 0,
        lambda_hf: float = 0.0,
        lambda_rgb_energy: float = 0.0,
        lambda_alpha_hf: float = 0.0,
        lambda_alpha_mass: float = 0.0,
        post_opacity_decay: float = 1.0,
        from_iter: int = 0,
        until_iter: int = 0,
        interval: int = 1,
    ):
        self.opacity_min = float(opacity_min)
        self.max_axis_min = float(max_axis_min)
        self.max_axis_max = float(max_axis_max)
        self.anisotropy_max = float(anisotropy_max)
        self.min_generation = int(min_generation)
        self.lambda_hf = float(lambda_hf)
        self.lambda_rgb_energy = float(lambda_rgb_energy)
        self.lambda_alpha_hf = float(lambda_alpha_hf)
        self.lambda_alpha_mass = float(lambda_alpha_mass)
        self.post_opacity_decay = float(post_opacity_decay)
        self.from_iter = int(from_iter)
        self.until_iter = int(until_iter)
        self.interval = max(1, int(interval))

    def active(self, iteration: int) -> bool:
        if iteration < self.from_iter:
            return False
        if self.until_iter > 0 and iteration > self.until_iter:
            return False
        return (iteration % self.interval) == 0

    def _candidate_mask(self, gaussians):
        return _build_prior_bubble_candidate_mask(
            gaussians,
            opacity_min=self.opacity_min,
            max_axis_min=self.max_axis_min,
            max_axis_max=self.max_axis_max,
            anisotropy_max=self.anisotropy_max,
            min_generation=self.min_generation,
        )

    def compute(
        self,
        *,
        iteration: int,
        gaussians,
        viewpoint_camera,
        render_fn,
        pipe,
        background,
        kernel_size,
        subpixel_offset=None,
    ):
        metrics = {}
        if not self.active(iteration):
            return None, metrics
        if (
            self.lambda_hf <= 0.0
            and self.lambda_rgb_energy <= 0.0
            and self.lambda_alpha_hf <= 0.0
            and self.lambda_alpha_mass <= 0.0
        ):
            return None, metrics

        candidate_mask, metrics = self._candidate_mask(gaussians)
        if candidate_mask is None or int(candidate_mask.sum().item()) <= 0:
            return None, metrics

        bubble_gaussians = GaussianSubsetView(gaussians, candidate_mask)
        black_background = torch.zeros_like(background)
        bubble_pkg = render_fn(
            viewpoint_camera,
            bubble_gaussians,
            pipe,
            black_background,
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
        )
        bubble_image = bubble_pkg["render"]
        direct_loss = None

        if self.lambda_rgb_energy > 0.0:
            loss_rgb = bubble_image.abs().mean()
            weighted = self.lambda_rgb_energy * loss_rgb
            direct_loss = weighted if direct_loss is None else direct_loss + weighted
            metrics["bubble_cleanup_rgb"] = loss_rgb.detach().item()
            metrics["bubble_cleanup_rgb_w"] = weighted.detach().item()

        if self.lambda_hf > 0.0:
            loss_hf = _laplacian_highfreq(bubble_image).abs().mean()
            weighted = self.lambda_hf * loss_hf
            direct_loss = weighted if direct_loss is None else direct_loss + weighted
            metrics["bubble_cleanup_hf"] = loss_hf.detach().item()
            metrics["bubble_cleanup_hf_w"] = weighted.detach().item()

        bubble_alpha = bubble_pkg.get("alpha")
        if bubble_alpha is not None:
            if self.lambda_alpha_hf > 0.0:
                loss_alpha_hf = _laplacian_highfreq(bubble_alpha).abs().mean()
                weighted = self.lambda_alpha_hf * loss_alpha_hf
                direct_loss = weighted if direct_loss is None else direct_loss + weighted
                metrics["bubble_cleanup_alpha_hf"] = loss_alpha_hf.detach().item()
                metrics["bubble_cleanup_alpha_hf_w"] = weighted.detach().item()
            if self.lambda_alpha_mass > 0.0:
                loss_alpha = bubble_alpha.mean()
                weighted = self.lambda_alpha_mass * loss_alpha
                direct_loss = weighted if direct_loss is None else direct_loss + weighted
                metrics["bubble_cleanup_alpha"] = loss_alpha.detach().item()
                metrics["bubble_cleanup_alpha_w"] = weighted.detach().item()
        elif self.lambda_alpha_hf > 0.0 or self.lambda_alpha_mass > 0.0:
            metrics["bubble_cleanup_alpha_missing"] = 1.0

        if direct_loss is not None:
            metrics["bubble_cleanup_total_w"] = direct_loss.detach().item()
        return direct_loss, metrics

    @torch.no_grad()
    def apply_post_step(self, gaussians, iteration: int):
        decay = float(self.post_opacity_decay)
        if decay <= 0.0 or decay >= 1.0:
            return {}
        if not self.active(iteration):
            return {}

        candidate_mask, metrics = self._candidate_mask(gaussians)
        if candidate_mask is None or int(candidate_mask.sum().item()) <= 0:
            return metrics

        old_opacity = gaussians.get_opacity.detach().view(-1)
        selected_old = old_opacity[candidate_mask]
        selected_new = torch.clamp(selected_old * decay, min=1e-5, max=1.0 - 1e-5)
        gaussians._opacity[candidate_mask] = gaussians.inverse_opacity_activation(
            selected_new[:, None].to(device=gaussians._opacity.device, dtype=gaussians._opacity.dtype)
        )
        metrics.update(
            {
                "bubble_cleanup_post_count": float(int(candidate_mask.sum().item())),
                "bubble_cleanup_post_decay": float(decay),
                "bubble_cleanup_post_opacity_before": float(selected_old.mean().item()),
                "bubble_cleanup_post_opacity_after": float(selected_new.mean().item()),
            }
        )
        return metrics


class PriorTensorBank:
    def __init__(self, index):
        self.index = index
        self.cache = {}

    def _load_tensor(self, image_name):
        path = self.index.get(image_name)
        if path is None or not os.path.exists(path):
            return None
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def get(self, image_name, width, height, device):
        if image_name not in self.cache:
            tensor = self._load_tensor(image_name)
            if tensor is None:
                return None
            self.cache[image_name] = tensor

        tensor = self.cache[image_name]
        if tensor.shape[-2:] != (height, width):
            tensor = torch_F.interpolate(
                tensor[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]

        return tensor.to(device=device)


class PriorMaskTensorBank:
    def __init__(self, index):
        self.index = index
        self.cache = {}

    def _load_tensor(self, image_name):
        path = self.index.get(image_name)
        if path is None or not os.path.exists(path):
            return None
        from PIL import Image

        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None].contiguous()
        return tensor

    def get(self, image_name, width, height, device):
        if image_name not in self.cache:
            tensor = self._load_tensor(image_name)
            if tensor is None:
                return None
            self.cache[image_name] = tensor

        tensor = self.cache[image_name]
        if tensor.shape[-2:] != (height, width):
            tensor = torch_F.interpolate(
                tensor[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]

        return tensor.to(device=device)


class SurfaceRouteConsensusBank:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.index = self._build_index(self.root_dir)
        self.cache = {}

    def has_view(self, image_name: str) -> bool:
        return str(image_name) in self.index

    @staticmethod
    def _build_index(root_dir: str):
        manifest_path = os.path.join(root_dir, "manifest.json")
        index = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            for view in manifest.get("views", []):
                image_name = str(view.get("image_name", "")).strip()
                path = str(view.get("path", "")).strip()
                if not image_name or not path:
                    continue
                index[image_name] = path
        if index:
            return index

        per_view_root = os.path.join(root_dir, "per_view")
        search_root = per_view_root if os.path.isdir(per_view_root) else root_dir
        for path in glob(os.path.join(search_root, "*.pt")):
            index[os.path.splitext(os.path.basename(path))[0]] = path
        return index

    def _load_payload(self, image_name: str):
        path = self.index.get(image_name)
        if path is None or not os.path.exists(path):
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        target_highfreq = payload.get("target_highfreq")
        target_weight = payload.get("target_weight")
        signal_mode = str(payload.get("signal_mode", "")).strip()
        if target_highfreq is not None and target_weight is not None:
            if not torch.is_tensor(target_highfreq):
                target_highfreq = torch.as_tensor(target_highfreq)
            if not torch.is_tensor(target_weight):
                target_weight = torch.as_tensor(target_weight)
            target_residual = payload.get("target_residual")
            surface_anchor_highfreq = payload.get("surface_anchor_highfreq")
            if target_residual is not None and not torch.is_tensor(target_residual):
                target_residual = torch.as_tensor(target_residual)
            if surface_anchor_highfreq is not None and not torch.is_tensor(surface_anchor_highfreq):
                surface_anchor_highfreq = torch.as_tensor(surface_anchor_highfreq)
            return {
                "mode": "dense",
                "target_highfreq": target_highfreq.to(dtype=torch.float32).contiguous(),
                "target_weight": target_weight.to(dtype=torch.float32).contiguous(),
                "target_residual": None
                if target_residual is None
                else target_residual.to(dtype=torch.float32).contiguous(),
                "surface_anchor_highfreq": None
                if surface_anchor_highfreq is None
                else surface_anchor_highfreq.to(dtype=torch.float32).contiguous(),
                "signal_mode": signal_mode or ("anchor_residual" if target_residual is not None else "absolute_highfreq"),
            }

        sample_y = payload.get("sample_y")
        sample_x = payload.get("sample_x")
        target_highfreq_samples = payload.get("target_highfreq_samples")
        target_residual_samples = payload.get("target_residual_samples")
        surface_anchor_highfreq_samples = payload.get("surface_anchor_highfreq_samples")
        target_weight_samples = payload.get("target_weight_samples")
        height = payload.get("height")
        width = payload.get("width")
        if (
            sample_y is None
            or sample_x is None
            or target_highfreq_samples is None
            or target_weight_samples is None
            or height is None
            or width is None
        ):
            return None
        if not torch.is_tensor(sample_y):
            sample_y = torch.as_tensor(sample_y)
        if not torch.is_tensor(sample_x):
            sample_x = torch.as_tensor(sample_x)
        if not torch.is_tensor(target_highfreq_samples):
            target_highfreq_samples = torch.as_tensor(target_highfreq_samples)
        if target_residual_samples is not None and not torch.is_tensor(target_residual_samples):
            target_residual_samples = torch.as_tensor(target_residual_samples)
        if surface_anchor_highfreq_samples is not None and not torch.is_tensor(surface_anchor_highfreq_samples):
            surface_anchor_highfreq_samples = torch.as_tensor(surface_anchor_highfreq_samples)
        if not torch.is_tensor(target_weight_samples):
            target_weight_samples = torch.as_tensor(target_weight_samples)
        return {
            "mode": "sparse",
            "height": int(height),
            "width": int(width),
            "sample_y": sample_y.to(dtype=torch.long).contiguous(),
            "sample_x": sample_x.to(dtype=torch.long).contiguous(),
            "target_highfreq_samples": target_highfreq_samples.to(dtype=torch.float32).contiguous(),
            "target_residual_samples": None
            if target_residual_samples is None
            else target_residual_samples.to(dtype=torch.float32).contiguous(),
            "surface_anchor_highfreq_samples": None
            if surface_anchor_highfreq_samples is None
            else surface_anchor_highfreq_samples.to(dtype=torch.float32).contiguous(),
            "target_weight_samples": target_weight_samples.to(dtype=torch.float32).contiguous(),
            "signal_mode": signal_mode
            or (
                "anchor_residual"
                if target_residual_samples is not None
                else "absolute_highfreq"
            ),
        }

    def get(self, image_name, width, height, device):
        if image_name not in self.cache:
            payload = self._load_payload(image_name)
            if payload is None:
                return None
            self.cache[image_name] = payload

        payload = self.cache[image_name]
        if payload.get("mode") == "sparse":
            base_height = int(payload["height"])
            base_width = int(payload["width"])
            target_highfreq = torch.zeros((3, base_height, base_width), dtype=torch.float32)
            target_residual = None
            surface_anchor_highfreq = None
            if payload.get("target_residual_samples") is not None:
                target_residual = torch.zeros((3, base_height, base_width), dtype=torch.float32)
            if payload.get("surface_anchor_highfreq_samples") is not None:
                surface_anchor_highfreq = torch.zeros((3, base_height, base_width), dtype=torch.float32)
            target_weight = torch.zeros((1, base_height, base_width), dtype=torch.float32)
            sample_y = payload["sample_y"]
            sample_x = payload["sample_x"]
            if sample_y.numel() > 0:
                target_highfreq[:, sample_y, sample_x] = payload["target_highfreq_samples"].transpose(0, 1)
                if target_residual is not None:
                    target_residual[:, sample_y, sample_x] = payload["target_residual_samples"].transpose(0, 1)
                if surface_anchor_highfreq is not None:
                    surface_anchor_highfreq[:, sample_y, sample_x] = payload[
                        "surface_anchor_highfreq_samples"
                    ].transpose(0, 1)
                target_weight[0, sample_y, sample_x] = payload["target_weight_samples"].reshape(-1)
        else:
            target_highfreq = payload["target_highfreq"]
            target_residual = payload.get("target_residual")
            surface_anchor_highfreq = payload.get("surface_anchor_highfreq")
            target_weight = payload["target_weight"]

        if target_highfreq.shape[-2:] != (height, width):
            target_highfreq = torch_F.interpolate(
                target_highfreq[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        if target_residual is not None and target_residual.shape[-2:] != (height, width):
            target_residual = torch_F.interpolate(
                target_residual[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        if surface_anchor_highfreq is not None and surface_anchor_highfreq.shape[-2:] != (height, width):
            surface_anchor_highfreq = torch_F.interpolate(
                surface_anchor_highfreq[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        if target_weight.shape[-2:] != (height, width):
            target_weight = torch_F.interpolate(
                target_weight[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]

        return {
            "target_highfreq": target_highfreq.to(device=device),
            "target_residual": None if target_residual is None else target_residual.to(device=device),
            "surface_anchor_highfreq": None
            if surface_anchor_highfreq is None
            else surface_anchor_highfreq.to(device=device),
            "target_weight": target_weight.to(device=device),
            "signal_mode": str(payload.get("signal_mode", "absolute_highfreq")),
        }


def _load_scene_camera_infos(dataset, images_override=None, transforms_llffhold=8):
    sparse0 = os.path.join(dataset.source_path, "sparse", "0")
    has_colmap = (
        os.path.exists(os.path.join(sparse0, "images.bin"))
        or os.path.exists(os.path.join(sparse0, "images.txt"))
    )

    if os.path.exists(os.path.join(dataset.source_path, "metadata.json")):
        scene_info = sceneLoadTypeCallbacks["Multi-scale"](
            dataset.source_path,
            dataset.white_background,
            dataset.eval,
            dataset.load_allres,
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](
            dataset.source_path,
            dataset.white_background,
            dataset.eval,
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms.json")):
        scene_info = read_transforms_json_scene(
            dataset.source_path,
            dataset.white_background,
            dataset.eval,
            llffhold=transforms_llffhold,
        )
    elif has_colmap:
        reading_dir = dataset.images if images_override is None else images_override
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path,
            reading_dir,
            dataset.eval,
            llffhold=transforms_llffhold,
        )
    else:
        raise AssertionError(
            "Unsupported dataset format for prior supervision override. Expected one of:\n"
            "  - metadata.json (multi-scale Blender)\n"
            "  - transforms_train.json (Blender)\n"
            "  - transforms.json (Neuralangelo/Instant-NGP style)\n"
            "  - sparse/0/images.bin or sparse/0/images.txt (COLMAP)"
        )

    return scene_info.train_cameras, scene_info.test_cameras


class PriorSupervisionCameraBank:
    def __init__(self, dataset, images_subdir, resolution, transforms_llffhold=8, cache_size=4):
        self.images_subdir = images_subdir
        self.resolution = resolution
        self.cache_size = max(1, int(cache_size))
        self.cache = OrderedDict()

        train_cam_infos, _ = _load_scene_camera_infos(
            dataset,
            images_override=images_subdir,
            transforms_llffhold=transforms_llffhold,
        )
        self.cam_infos_by_name = {cam.image_name: cam for cam in train_cam_infos}

        self.dataset_args = Namespace(**vars(dataset))
        self.dataset_args.images = images_subdir
        self.dataset_args.resolution = resolution

    def get(self, image_name):
        cached = self.cache.get(image_name)
        if cached is not None:
            self.cache.move_to_end(image_name)
            return cached

        cam_info = self.cam_infos_by_name.get(image_name)
        if cam_info is None:
            return None

        camera = loadCam(self.dataset_args, cam_info.uid, cam_info, 1.0)
        self.cache[image_name] = camera
        self.cache.move_to_end(image_name)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return camera


def _make_merged_sof_training_settings():
    if ExtendedSettings is None:
        return None

    # The merged rasterizer is based on SOF's multi-output path. Align its
    # auxiliary-buffer semantics with SOF training instead of relying on the
    # merged binding defaults, which differ for several critical flags.
    settings = ExtendedSettings()
    settings.load_balancing = True
    settings.proper_ewa_scaling = False
    settings.exact_depth = True
    settings.detach_alpha = True
    settings.detach_alpha_extent = True
    settings.include_alpha = False
    settings.render_opacity = False

    if hasattr(settings, "culling_settings"):
        settings.culling_settings.rect_bounding = True
        settings.culling_settings.tight_opacity_bounding = True
        settings.culling_settings.tile_based_culling = False
        settings.culling_settings.hierarchical_4x4_culling = False

    if hasattr(settings, "sort_settings") and SortMode is not None and GlobalSortOrder is not None:
        settings.sort_settings.sort_mode = SortMode.HIER
        settings.sort_settings.sort_order = GlobalSortOrder.PTD_MAX
        if hasattr(settings.sort_settings, "queue_sizes"):
            settings.sort_settings.queue_sizes.tile_4x4 = 64
            settings.sort_settings.queue_sizes.tile_2x2 = 8
            settings.sort_settings.queue_sizes.per_pixel = 4

    return settings


def _load_vanilla_sof_module():
    try:
        import diff_gaussian_rasterization_sof_vanilla as vanilla_mod
    except ImportError:
        return None
    return vanilla_mod


def _make_vanilla_sof_training_settings():
    vanilla_mod = _load_vanilla_sof_module()
    if vanilla_mod is None:
        return None

    settings = vanilla_mod.ExtendedSettings()
    settings.load_balancing = True
    settings.proper_ewa_scaling = False
    settings.exact_depth = True
    settings.detach_alpha = True
    settings.detach_alpha_extent = True
    settings.include_alpha = False
    settings.render_opacity = False

    if hasattr(settings, "culling_settings"):
        settings.culling_settings.rect_bounding = True
        settings.culling_settings.tight_opacity_bounding = True
        settings.culling_settings.tile_based_culling = False
        settings.culling_settings.hierarchical_4x4_culling = False

    sort_mode = getattr(vanilla_mod, "SortMode", None)
    sort_order = getattr(vanilla_mod, "GlobalSortOrder", None)
    if hasattr(settings, "sort_settings") and sort_mode is not None and sort_order is not None:
        settings.sort_settings.sort_mode = sort_mode.HIER
        settings.sort_settings.sort_order = sort_order.PTD_MAX
        if hasattr(settings.sort_settings, "queue_sizes"):
            settings.sort_settings.queue_sizes.tile_4x4 = 64
            settings.sort_settings.queue_sizes.tile_2x2 = 8
            settings.sort_settings.queue_sizes.per_pixel = 4

    return settings


def _load_prior_feature_pack(
    prior_bank,
    camera,
    device,
    hybrid_args,
    prior_mask_bank=None,
    prior_anchor_bank=None,
    prior_soft_mask_bank=None,
    prior_hard_mask_bank=None,
):
    if prior_bank is None or camera is None:
        return None
    prior_image = prior_bank.get(
        camera.image_name,
        width=int(camera.image_width),
        height=int(camera.image_height),
        device=device,
    )
    if prior_image is None:
        return None

    gt_image = camera.original_image.to(device=device)
    prior_anchor_image = None
    if prior_anchor_bank is not None:
        prior_anchor_image = prior_anchor_bank.get(
            camera.image_name,
            width=int(camera.image_width),
            height=int(camera.image_height),
            device=device,
        )

    prior_soft_mask = None
    if prior_soft_mask_bank is not None:
        prior_soft_mask = prior_soft_mask_bank.get(
            camera.image_name,
            width=int(camera.image_width),
            height=int(camera.image_height),
            device=device,
        )

    prior_hard_mask = None
    if prior_hard_mask_bank is not None:
        prior_hard_mask = prior_hard_mask_bank.get(
            camera.image_name,
            width=int(camera.image_width),
            height=int(camera.image_height),
            device=device,
        )

    prior_loss_mode = str(getattr(hybrid_args, "prior_loss_mode", "rgb_hf"))
    if prior_loss_mode == "masked_residual_hf_v1":
        if prior_anchor_image is None:
            return None
        use_external_masks = prior_soft_mask is not None and prior_hard_mask is not None
        if use_external_masks:
            hard_threshold = float(getattr(hybrid_args, "prior_hard_mask_threshold", 0.5))
            soft_power = float(getattr(hybrid_args, "prior_soft_mask_power", 1.0))
            hard_region = (prior_hard_mask >= hard_threshold).float()
            soft_weight = prior_soft_mask.clamp(min=0.0, max=1.0)
            if soft_power != 1.0:
                soft_weight = soft_weight.clamp(min=1e-8, max=1.0).pow(soft_power)
        else:
            hard_region = torch.ones_like(prior_image[:1])
            soft_weight = torch.ones_like(prior_image[:1])
            prior_soft_mask = soft_weight
            prior_hard_mask = hard_region
        prior_mask = hard_region * soft_weight
        masked_min_pixels = float(getattr(hybrid_args, "prior_masked_min_pixels", 64.0))
        valid_pixels = prior_mask.gt(0).float().sum()
        if float(valid_pixels.detach().item()) < masked_min_pixels:
            return None
        prior_delta = prior_image - prior_anchor_image
        if hybrid_args.prior_delta_clip > 0:
            clip_v = float(hybrid_args.prior_delta_clip)
            prior_delta = prior_delta.clamp(min=-clip_v, max=clip_v)
        gray_delta = prior_delta.mean(dim=0, keepdim=True)
        lap_hf = _laplacian_highfreq(prior_delta).abs().mean(dim=0, keepdim=True)
        grad_x, grad_y = _sobel_gradients(gray_delta)
        hf_guidance = lap_hf * prior_mask
        feature_map = torch.cat(
            [
                hf_guidance,
                grad_x * prior_mask,
                grad_y * prior_mask,
            ],
            dim=0,
        )
        return {
            "prior_image": prior_image,
            "gt_image": gt_image,
            "anchor_image": prior_anchor_image,
            "consistency": torch.zeros_like(prior_mask),
            "mask": prior_mask,
            "consistency_mask": hard_region,
            "structure_mask": soft_weight,
            "hard_mask": hard_region,
            "soft_mask": soft_weight,
            "valid_ratio": torch.ones((), dtype=prior_mask.dtype, device=prior_mask.device),
            "guidance": hf_guidance,
            "feature_map": feature_map,
        }

    prior_consistency = (prior_image - gt_image).abs().mean(dim=0, keepdim=True)
    consistency_mask = (
        prior_consistency <= float(hybrid_args.prior_consistency_threshold)
    ).float()
    structure_mask = torch.ones_like(consistency_mask)
    if prior_mask_bank is not None:
        loaded_mask = prior_mask_bank.get(
            camera.image_name,
            width=int(camera.image_width),
            height=int(camera.image_height),
            device=device,
        )
        if loaded_mask is not None:
            structure_mask = loaded_mask.clamp(min=0.0, max=1.0)
            mask_floor = max(0.0, float(getattr(hybrid_args, "prior_mask_floor", 0.0)))
            if mask_floor > 0.0:
                structure_mask = structure_mask.clamp(min=mask_floor, max=1.0)
    prior_mask = consistency_mask * structure_mask
    prior_delta = prior_image - gt_image
    if hybrid_args.prior_delta_clip > 0:
        clip_v = float(hybrid_args.prior_delta_clip)
        prior_delta = prior_delta.clamp(min=-clip_v, max=clip_v)

    gray_prior = prior_image.mean(dim=0, keepdim=True)
    gray_delta = prior_delta.mean(dim=0, keepdim=True)
    grad_x, grad_y = _sobel_gradients(gray_delta)
    lap_hf = _laplacian_highfreq(prior_delta).abs().mean(dim=0, keepdim=True)
    wavelet_delta_hf = _haar_wavelet_highfreq(gray_delta)
    wavelet_prior_hf = _haar_wavelet_highfreq(gray_prior)

    if _proposal_full_stack_enabled(hybrid_args):
        hf_guidance = torch.maximum(
            lap_hf,
            torch.maximum(
                wavelet_delta_hf,
                wavelet_prior_hf * float(hybrid_args.sdf_proposal_prior_hf_prior_weight),
            ),
        )
        grad_src = gray_prior
    else:
        hf_guidance = lap_hf
        grad_src = gray_delta

    hf_guidance = hf_guidance * prior_mask
    grad_src_x, grad_src_y = _sobel_gradients(grad_src)
    feature_channels = [
        hf_guidance,
        grad_x * prior_mask,
        grad_y * prior_mask,
    ]
    if _proposal_full_stack_enabled(hybrid_args):
        feature_channels.extend(
            [
                wavelet_delta_hf * prior_mask,
                wavelet_prior_hf * prior_mask,
                grad_src_x * prior_mask,
                grad_src_y * prior_mask,
            ]
        )
    feature_map = torch.cat(feature_channels, dim=0)
    return {
        "prior_image": prior_image,
        "gt_image": gt_image,
        "anchor_image": prior_anchor_image,
        "consistency": prior_consistency,
        "mask": prior_mask,
        "consistency_mask": consistency_mask,
        "structure_mask": structure_mask,
        "hard_mask": prior_hard_mask,
        "soft_mask": prior_soft_mask,
        "valid_ratio": consistency_mask.mean(),
        "guidance": hf_guidance,
        "feature_map": feature_map,
    }


def _select_neighbor_cameras(
    reference_camera,
    camera_pool,
    max_neighbors,
    prior_bank=None,
    prior_mask_bank=None,
    anchors_xyz=None,
    hybrid_args=None,
    device=None,
):
    if reference_camera is None or camera_pool is None or len(camera_pool) == 0:
        return []
    ref_center = reference_camera.camera_center.detach()
    ranked = []
    for camera in camera_pool:
        if camera.image_name == reference_camera.image_name:
            continue
        center = camera.camera_center.detach()
        dist = torch.linalg.norm(center - ref_center).item()
        view_pack = None
        score = -dist
        if (
            hybrid_args is not None
            and _proposal_full_stack_enabled(hybrid_args)
            and prior_bank is not None
            and anchors_xyz is not None
            and anchors_xyz.numel() > 0
        ):
            view_pack = _load_prior_feature_pack(
                prior_bank=prior_bank,
                camera=camera,
                device=device if device is not None else anchors_xyz.device,
                hybrid_args=hybrid_args,
                prior_mask_bank=prior_mask_bank,
            )
            if view_pack is not None:
                sampled_guidance, valid_ratio = _sample_image_guidance_on_points(
                    guidance_map=view_pack["guidance"],
                    points=anchors_xyz.detach(),
                    camera=camera,
                )
                mean_guidance = 0.0 if sampled_guidance is None else float(sampled_guidance.mean().item())
                score = (
                    float(hybrid_args.sdf_proposal_neighbor_guidance_weight) * mean_guidance
                    + float(hybrid_args.sdf_proposal_neighbor_visibility_weight) * valid_ratio
                    - float(hybrid_args.sdf_proposal_neighbor_distance_weight) * dist
                )
        ranked.append((score, dist, camera, view_pack))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[: max(0, int(max_neighbors))]


def _build_surface_birth_bundle(
    attached_xyz,
    attached_normals,
    base_scale,
    rotations_raw,
    opacities,
    samples_per_anchor,
    tangent_scale,
    normal_scale,
):
    if attached_xyz is None or attached_xyz.numel() == 0:
        return None

    xyz_parts = [attached_xyz]
    rot_parts = [rotations_raw]
    scale_parts = [base_scale]
    opacity_parts = [opacities]

    spp = max(0, int(samples_per_anchor))
    if spp > 0:
        plane_xyz = _sample_surface_plane_points(
            anchors_xyz=attached_xyz,
            anchors_normals=attached_normals,
            base_scale=base_scale,
            samples_per_anchor=spp,
            tangent_scale=tangent_scale,
            normal_scale=normal_scale,
        )
        if plane_xyz is not None and plane_xyz.shape[0] > 0:
            xyz_parts.append(plane_xyz)
            rot_parts.append(rotations_raw.repeat_interleave(spp, dim=0))
            scale_parts.append(base_scale.repeat_interleave(spp, dim=0))
            opacity_parts.append(opacities.repeat_interleave(spp, dim=0))

    return {
        "xyz": torch.cat(xyz_parts, dim=0),
        "rotations_raw": torch.cat(rot_parts, dim=0),
        "scales": torch.cat(scale_parts, dim=0),
        "opacities": torch.cat(opacity_parts, dim=0),
    }


def _build_view_anchored_realign_bundle(
    matched_idx,
    ref_pixels,
    target_xyz,
    target_normals,
    weights,
):
    if (
        matched_idx is None
        or ref_pixels is None
        or target_xyz is None
        or target_normals is None
        or matched_idx.numel() == 0
        or ref_pixels.numel() == 0
        or target_xyz.numel() == 0
        or target_normals.numel() == 0
    ):
        return None
    return {
        "matched_idx": matched_idx,
        "ref_pixels": ref_pixels,
        "target_xyz": target_xyz,
        "target_normals": target_normals,
        "weights": weights,
    }


def _select_best_proposals_per_gaussian(matched_idx, weights):
    if matched_idx is None or matched_idx.numel() == 0:
        return None
    idx_cpu = matched_idx.detach().view(-1).cpu().tolist()
    if weights is None:
        w_cpu = [1.0] * len(idx_cpu)
    else:
        w_cpu = weights.detach().view(-1).cpu().tolist()
    best = {}
    for row, (idx, weight) in enumerate(zip(idx_cpu, w_cpu)):
        prev = best.get(idx)
        if prev is None or weight > prev[0]:
            best[idx] = (weight, row)
    if len(best) == 0:
        return None
    selected_rows = [row for _, row in best.values()]
    selected_rows = torch.tensor(selected_rows, device=matched_idx.device, dtype=torch.long)
    return selected_rows


def _apply_view_anchored_surface_realign(
    gaussians,
    reference_camera,
    realign_bundle,
    hybrid_args,
):
    metrics = {
        "realigned_gs": 0.0,
        "realign_shift_mean": 0.0,
        "realign_depth_delta_mean": 0.0,
        "realign_rot_mean": 0.0,
        "realign_scale_ratio_mean": 1.0,
    }
    if (
        gaussians is None
        or reference_camera is None
        or realign_bundle is None
        or realign_bundle.get("matched_idx") is None
    ):
        return metrics

    matched_idx = realign_bundle["matched_idx"]
    ref_pixels = realign_bundle["ref_pixels"]
    target_xyz = realign_bundle["target_xyz"]
    target_normals = realign_bundle["target_normals"]
    weights = realign_bundle.get("weights")
    if matched_idx.numel() == 0:
        return metrics

    selected_rows = _select_best_proposals_per_gaussian(matched_idx, weights)
    if selected_rows is None or selected_rows.numel() == 0:
        return metrics

    matched_idx = matched_idx[selected_rows]
    ref_pixels = ref_pixels[selected_rows]
    target_xyz = target_xyz[selected_rows]
    target_normals = target_normals[selected_rows]

    ray_origins, ray_dirs = _build_world_rays_from_pixels(reference_camera, ref_pixels)
    if ray_origins is None or ray_dirs is None:
        return metrics

    current_xyz = gaussians.get_xyz.detach()[matched_idx]
    current_depth = ((current_xyz - ray_origins) * ray_dirs).sum(dim=-1)
    target_depth = ((target_xyz - ray_origins) * ray_dirs).sum(dim=-1)

    min_depth = float(hybrid_args.sdf_proposal_realign_min_depth)
    valid = (current_depth > min_depth) & (target_depth > min_depth)
    if valid.sum().item() == 0:
        return metrics

    current_depth = current_depth[valid]
    target_depth = target_depth[valid]
    matched_idx = matched_idx[valid]
    current_xyz = current_xyz[valid]
    ray_origins = ray_origins[valid]
    ray_dirs = ray_dirs[valid]
    target_normals = target_normals[valid]

    raw_ratio = target_depth / current_depth.clamp(min=min_depth)
    max_ratio = max(1.0, float(hybrid_args.sdf_proposal_realign_max_ratio))
    ratio = raw_ratio.clamp(min=1.0 / max_ratio, max=max_ratio)
    target_depth = current_depth * ratio

    depth_delta = target_depth - current_depth
    max_shift = max(0.0, float(hybrid_args.sdf_proposal_realign_max_shift))
    if max_shift > 0:
        depth_delta = depth_delta.clamp(min=-max_shift, max=max_shift)
    target_depth = current_depth + depth_delta
    target_pos = ray_origins + target_depth[:, None] * ray_dirs

    blend = float(hybrid_args.sdf_proposal_realign_ema)
    blend = min(max(blend, 0.0), 1.0)
    new_xyz = (1.0 - blend) * current_xyz + blend * target_pos

    gaussians._xyz[matched_idx] = new_xyz
    if gaussians._xyz.grad is not None:
        gaussians._xyz.grad[matched_idx] = 0

    rot_delta_mean = 0.0
    scale_ratio_mean = 1.0
    if _proposal_full_stack_enabled(hybrid_args):
        current_scales = gaussians.get_scaling.detach()[matched_idx]
        target_quat = _build_rotation_from_surface(
            normals=target_normals,
            scales=current_scales,
            view_dirs=ray_dirs,
        )
        rot_blend = float(hybrid_args.sdf_proposal_realign_rotation_ema)
        rot_blend = min(max(rot_blend, 0.0), 1.0)
        current_quat = torch_F.normalize(gaussians._rotation.detach()[matched_idx], dim=-1)
        blended_quat = torch_F.normalize(
            (1.0 - rot_blend) * current_quat + rot_blend * target_quat,
            dim=-1,
        )
        rot_delta_mean = float((blended_quat - current_quat).norm(dim=-1).mean().item())
        gaussians._rotation[matched_idx] = blended_quat
        if gaussians._rotation.grad is not None:
            gaussians._rotation.grad[matched_idx] = 0

        scale_ratio = (target_depth / current_depth.clamp(min=min_depth)).clamp(
            min=1.0 / max_ratio,
            max=max_ratio,
        )
        scale_ratio_mean = float(scale_ratio.mean().item())
        target_scales = current_scales * scale_ratio[:, None]
        scale_blend = float(hybrid_args.sdf_proposal_realign_scale_ema)
        scale_blend = min(max(scale_blend, 0.0), 1.0)
        current_log_scales = gaussians._scaling.detach()[matched_idx]
        target_log_scales = torch.log(target_scales.clamp(min=1e-6))
        blended_log_scales = (1.0 - scale_blend) * current_log_scales + scale_blend * target_log_scales
        gaussians._scaling[matched_idx] = blended_log_scales
        if gaussians._scaling.grad is not None:
            gaussians._scaling.grad[matched_idx] = 0

    metrics["realigned_gs"] = float(matched_idx.shape[0])
    metrics["realign_shift_mean"] = float((new_xyz - current_xyz).norm(dim=-1).mean().item())
    metrics["realign_depth_delta_mean"] = float(depth_delta.abs().mean().item())
    metrics["realign_rot_mean"] = rot_delta_mean
    metrics["realign_scale_ratio_mean"] = scale_ratio_mean
    return metrics


def _build_native_birth_bundle(
    matched_idx,
    target_xyz,
    target_normals,
    ref_pixels,
    weights,
):
    if (
        matched_idx is None
        or target_xyz is None
        or target_normals is None
        or ref_pixels is None
        or matched_idx.numel() == 0
    ):
        return None
    return {
        "matched_idx": matched_idx,
        "target_xyz": target_xyz,
        "target_normals": target_normals,
        "ref_pixels": ref_pixels,
        "weights": weights,
    }


def _apply_native_surface_birth(
    gaussians,
    birth_bundle,
    reference_camera,
    hybrid_args,
    train_cameras=None,
):
    metrics = {
        "native_births": 0.0,
    }
    if (
        gaussians is None
        or birth_bundle is None
        or birth_bundle.get("matched_idx") is None
    ):
        return metrics

    matched_idx = birth_bundle["matched_idx"]
    target_xyz = birth_bundle["target_xyz"]
    target_normals = birth_bundle["target_normals"]
    ref_pixels = birth_bundle["ref_pixels"]
    weights = birth_bundle.get("weights")
    selected_rows = _select_best_proposals_per_gaussian(matched_idx, weights)
    if selected_rows is None or selected_rows.numel() == 0:
        return metrics

    matched_idx = matched_idx[selected_rows]
    target_xyz = target_xyz[selected_rows]
    target_normals = target_normals[selected_rows]
    ref_pixels = ref_pixels[selected_rows]

    max_births = max(0, int(hybrid_args.sdf_proposal_native_birth_max_points))
    if max_births > 0 and target_xyz.shape[0] > max_births:
        matched_idx = matched_idx[:max_births]
        target_xyz = target_xyz[:max_births]
        target_normals = target_normals[:max_births]
        ref_pixels = ref_pixels[:max_births]

    ray_origins, ray_dirs = _build_world_rays_from_pixels(reference_camera, ref_pixels)
    if ray_dirs is None:
        return metrics

    base_scales = gaussians.get_scaling.detach()[matched_idx]
    new_rotation = _build_rotation_from_surface(
        normals=target_normals,
        scales=base_scales,
        view_dirs=ray_dirs,
    )
    base_dc = gaussians._features_dc.detach()[matched_idx]
    base_rest = gaussians._features_rest.detach()[matched_idx]
    base_opacity = gaussians._opacity.detach()[matched_idx]
    base_log_scaling = gaussians._scaling.detach()[matched_idx]

    gaussians.densification_postfix(
        new_xyz=target_xyz.detach(),
        new_features_dc=base_dc,
        new_features_rest=base_rest,
        new_opacities=base_opacity,
        new_scaling=base_log_scaling,
        new_rotation=new_rotation.detach(),
    )
    if train_cameras is not None:
        gaussians.compute_3D_filter(cameras=train_cameras)
    metrics["native_births"] = float(target_xyz.shape[0])
    return metrics


def _fuse_surface_proposals_multiview(
    sdf_adapter,
    prior_bank,
    prior_mask_bank,
    ref_camera,
    camera_pool,
    ref_pixels,
    ref_feature_pack,
    anchors_xyz,
    anchors_normals,
    base_scale,
    hybrid_args,
):
    if (
        sdf_adapter is None
        or prior_bank is None
        or ref_feature_pack is None
        or ref_pixels is None
        or ref_pixels.numel() == 0
        or anchors_xyz is None
        or anchors_xyz.numel() == 0
    ):
        return None, {
            "neighbor_views": 0.0,
            "fused_points": 0.0,
            "inlier_views_mean": 0.0,
            "delta_abs_mean": 0.0,
        }

    ref_features = _sample_feature_map_on_pixels(
        ref_feature_pack["feature_map"], ref_pixels.to(device=anchors_xyz.device, dtype=anchors_xyz.dtype)
    )
    if ref_features is None:
        return None, {
            "neighbor_views": 0.0,
            "fused_points": 0.0,
            "inlier_views_mean": 0.0,
            "delta_abs_mean": 0.0,
        }

    ref_hf = ref_features[:, 0].clamp(min=0.0)
    ref_dir = _normalize_vectors_2d(ref_features[:, 1:3])
    ref_desc = ref_features[:, 3:] if ref_features.shape[1] > 3 else None
    n = anchors_xyz.shape[0]
    device = anchors_xyz.device
    dtype = anchors_xyz.dtype

    max_neighbors = max(0, int(hybrid_args.sdf_proposal_fuse_max_views))
    neighbor_cameras = _select_neighbor_cameras(
        reference_camera=ref_camera,
        camera_pool=camera_pool,
        max_neighbors=max_neighbors,
        prior_bank=prior_bank,
        prior_mask_bank=prior_mask_bank,
        anchors_xyz=anchors_xyz,
        hybrid_args=hybrid_args,
        device=device,
    )
    if len(neighbor_cameras) == 0:
        return None, {
            "neighbor_views": 0.0,
            "fused_points": 0.0,
            "inlier_views_mean": 0.0,
            "delta_abs_mean": 0.0,
            "tangent_abs_mean": 0.0,
        }

    delta_steps = max(3, int(hybrid_args.sdf_proposal_delta_steps))
    unit_deltas = torch.linspace(-1.0, 1.0, steps=delta_steps, device=device, dtype=dtype)
    if _proposal_full_stack_enabled(hybrid_args):
        tangent_steps = max(1, int(hybrid_args.sdf_proposal_tangent_steps))
        unit_tangent = torch.linspace(-1.0, 1.0, steps=tangent_steps, device=device, dtype=dtype)
    else:
        tangent_steps = 1
        unit_tangent = torch.zeros((1,), device=device, dtype=dtype)
    base_sigma = base_scale.min(dim=1).values.clamp(min=1e-4)
    delta_max = torch.maximum(
        base_sigma * float(hybrid_args.sdf_proposal_delta_scale),
        torch.full_like(base_sigma, float(hybrid_args.sdf_proposal_delta_floor)),
    )
    tangent_max = base_sigma * float(hybrid_args.sdf_proposal_tangent_search_scale)
    ref_ray_origins, ref_ray_dirs = _build_world_rays_from_pixels(ref_camera, ref_pixels.to(device=device, dtype=dtype))
    t1, t2 = _build_tangent_basis(anchors_normals, hint_dirs=ref_ray_dirs)
    delta_grid = unit_deltas[None, None, None, :] * delta_max[:, None, None, None]
    u_grid = unit_tangent[None, :, None, None] * tangent_max[:, None, None, None]
    v_grid = unit_tangent[None, None, :, None] * tangent_max[:, None, None, None]
    candidate_xyz = (
        anchors_xyz[:, None, None, None, :]
        + u_grid[..., None] * t1[:, None, None, None, :]
        + v_grid[..., None] * t2[:, None, None, None, :]
        + delta_grid[..., None] * anchors_normals[:, None, None, None, :]
    )
    num_candidates = candidate_xyz.shape[1] * candidate_xyz.shape[2] * candidate_xyz.shape[3]
    candidate_xyz = candidate_xyz.view(n, num_candidates, 3)
    delta_flat = delta_grid.expand(-1, tangent_steps, tangent_steps, -1).reshape(n, num_candidates)
    u_flat = u_grid.expand(-1, -1, tangent_steps, delta_steps).reshape(n, num_candidates)
    v_flat = v_grid.expand(-1, tangent_steps, -1, delta_steps).reshape(n, num_candidates)

    flat_xyz = candidate_xyz.reshape(-1, 3)
    sdf_vals, _ = sdf_adapter.query_sdf_and_gradients(flat_xyz.detach())
    sdf_vals = sdf_vals.to(device=device, dtype=dtype).view(n, num_candidates).abs()

    total_energy = float(hybrid_args.sdf_proposal_sdf_cost_weight) * sdf_vals
    total_energy = total_energy + float(hybrid_args.sdf_proposal_delta_reg_weight) * (delta_flat / delta_max[:, None].clamp(min=1e-6)).pow(2)
    if _proposal_full_stack_enabled(hybrid_args):
        total_energy = total_energy + float(hybrid_args.sdf_proposal_tangent_reg_weight) * (
            (u_flat / tangent_max[:, None].clamp(min=1e-6)).pow(2)
            + (v_flat / tangent_max[:, None].clamp(min=1e-6)).pow(2)
        )
    total_weight = torch.zeros((n, num_candidates), device=device, dtype=dtype)
    inlier_views = torch.zeros((n, num_candidates), device=device, dtype=torch.int64)
    used_views = 0

    for _, _, camera, preloaded_view_pack in neighbor_cameras:
        view_pack = preloaded_view_pack
        if view_pack is None:
            view_pack = _load_prior_feature_pack(prior_bank, camera, device=device, hybrid_args=hybrid_args)
        if view_pack is None:
            continue
        sampled_features, in_view = _sample_feature_map_on_points(
            view_pack["feature_map"], flat_xyz, camera
        )
        sampled_masks, _ = _sample_feature_map_on_points(view_pack["mask"], flat_xyz, camera)
        if sampled_features is None or in_view is None or sampled_masks is None:
            continue

        sampled_features = sampled_features.view(n, num_candidates, -1)
        sampled_masks = sampled_masks.view(n, num_candidates, -1)[..., 0]
        in_view = in_view.view(n, num_candidates)

        neighbor_hf = sampled_features[..., 0].clamp(min=0.0)
        neighbor_dir = _normalize_vectors_2d(sampled_features[..., 1:3])
        dir_align = (neighbor_dir * ref_dir[:, None, :]).sum(dim=-1).abs().clamp(min=0.0, max=1.0)
        dir_cost = 1.0 - dir_align
        hf_cost = (neighbor_hf - ref_hf[:, None]).abs() / ref_hf[:, None].clamp(min=1e-4)
        if ref_desc is not None and sampled_features.shape[-1] > 3:
            desc_cost = (sampled_features[..., 3:] - ref_desc[:, None, :]).abs().mean(dim=-1)
        else:
            desc_cost = torch.zeros_like(hf_cost)

        valid = (
            in_view
            & (sampled_masks > 0.5)
            & (neighbor_hf >= float(hybrid_args.sdf_proposal_view_min_hf))
        )
        view_weight = torch.where(valid, neighbor_hf, torch.zeros_like(neighbor_hf))
        view_energy = (
            float(hybrid_args.sdf_proposal_feature_hf_weight) * hf_cost
            + float(hybrid_args.sdf_proposal_feature_dir_weight) * dir_cost
            + float(hybrid_args.sdf_proposal_feature_desc_weight) * desc_cost
        )
        total_energy = total_energy + view_energy * view_weight
        total_weight = total_weight + view_weight
        inlier_views = inlier_views + valid.long()
        used_views += 1

    required_neighbor_views = max(1, int(hybrid_args.sdf_proposal_fuse_min_views) - 1)
    if used_views == 0:
        return None, {
            "neighbor_views": 0.0,
            "fused_points": 0.0,
            "inlier_views_mean": 0.0,
            "delta_abs_mean": 0.0,
            "tangent_abs_mean": 0.0,
        }

    candidate_valid = inlier_views >= required_neighbor_views
    avg_energy = total_energy / (1.0 + total_weight)
    avg_energy = torch.where(
        candidate_valid,
        avg_energy,
        torch.full_like(avg_energy, float("inf")),
    )
    has_solution = torch.isfinite(avg_energy).any(dim=1)
    if has_solution.sum().item() == 0:
        return None, {
            "neighbor_views": float(used_views),
            "fused_points": 0.0,
            "inlier_views_mean": 0.0,
            "delta_abs_mean": 0.0,
            "tangent_abs_mean": 0.0,
        }

    best_idx = torch.argmin(avg_energy, dim=1)
    row_idx = torch.arange(n, device=device)
    best_xyz = candidate_xyz[row_idx, best_idx, :]
    best_delta = delta_flat[row_idx, best_idx]
    best_u = u_flat[row_idx, best_idx]
    best_v = v_flat[row_idx, best_idx]
    best_inliers = inlier_views[row_idx, best_idx]

    keep = has_solution & (best_inliers >= required_neighbor_views)
    if keep.sum().item() == 0:
        return None, {
            "neighbor_views": float(used_views),
            "fused_points": 0.0,
            "inlier_views_mean": 0.0,
            "delta_abs_mean": 0.0,
            "tangent_abs_mean": 0.0,
        }

    projected = _newton_project_to_sdf_surface(
        sdf_adapter=sdf_adapter,
        points=best_xyz[keep],
        steps=hybrid_args.sdf_proposal_newton_steps,
    )
    result = {
        "xyz": projected["xyz"],
        "normals": projected["grad"],
        "keep": keep,
        "delta": best_delta[keep],
        "u": best_u[keep],
        "v": best_v[keep],
        "inlier_views": best_inliers[keep].to(dtype=dtype),
        "ref_ray_dirs": ref_ray_dirs[keep],
    }
    metrics = {
        "neighbor_views": float(used_views),
        "fused_points": float(projected["xyz"].shape[0]),
        "inlier_views_mean": float(best_inliers[keep].float().mean().item()),
        "delta_abs_mean": float(best_delta[keep].abs().mean().item()),
        "tangent_abs_mean": float((best_u[keep].pow(2) + best_v[keep].pow(2)).sqrt().mean().item()),
    }
    return result, metrics


def _linear_decay_scale(iteration: int, decay_from: int, decay_until: int, final_scale: float) -> float:
    decay_from = int(decay_from)
    decay_until = int(decay_until)
    final_scale = max(0.0, float(final_scale))
    if decay_from <= 0 or decay_until <= decay_from:
        return 1.0
    iteration = int(iteration)
    if iteration <= decay_from:
        return 1.0
    if iteration >= decay_until:
        return final_scale
    progress = float(iteration - decay_from) / float(decay_until - decay_from)
    return (1.0 - progress) + progress * final_scale


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    hybrid_args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    model_params = None
    resume_from_checkpoint = checkpoint is not None
    if resume_from_checkpoint:
        model_params, first_iter = torch.load(checkpoint)
    scene = HybridScene(
        dataset,
        gaussians,
        transforms_llffhold=hybrid_args.transforms_llffhold,
        skip_initial_pcd=resume_from_checkpoint,
    )
    if resume_from_checkpoint:
        gaussians.restore(model_params, opt)
    else:
        gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()
    all_cameras = train_cameras + test_cameras

    highresolution_index = []
    for index, camera in enumerate(train_cameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=train_cameras)

    sdf_adapter = None
    loss_scheduler = None
    prior_bank = None
    prior_mask_bank = None
    prior_anchor_bank = None
    prior_soft_mask_bank = None
    prior_hard_mask_bank = None
    sequence_lr_anchor_bank = None
    prior_supervision_camera_bank = None
    sof_prior_block = None
    sof_regularization_block = None
    prior_local_bank = None
    prior_local_mask_bank = None
    prior_edge_bank = None
    prior_edge_mask_bank = None
    surface_route_bank = None
    external_update_mask = None
    active_train_cameras = train_cameras
    prior_edge_skip_count = 0
    sdf_densify_block = None
    fm_sds_block = None
    frequency_block = None
    scaffold_block = None
    surface_filter_3d = None
    layer_frequency_regularizer = None
    layer_frequency_start_gaussians = None
    prior_bubble_cleanup_regularizer = None
    surface_migration_regularizer = None
    if hybrid_args.hybrid_enable:
        sdf_adapter = build_sdf_adapter(hybrid_args)
        if sdf_adapter is not None:
            if hasattr(sdf_adapter, "bind_gaussians"):
                sdf_adapter.bind_gaussians(gaussians)
            loss_scheduler = HybridLossScheduler(
                stage_a_end=hybrid_args.hybrid_stage_a_end,
                stage_b_end=hybrid_args.hybrid_stage_b_end,
            )
            print(f"Hybrid regularization enabled (mode={hybrid_args.sdf_mode}).")
        else:
            print("Hybrid regularization requested but no SDF adapter was created.")

    prior_index = {}

    prior_root = ""
    prior_subdir = hybrid_args.external_prior_subdir
    if hybrid_args.external_prior_use_dataset_root:
        prior_root = dataset.source_path
        print(
            "[PRIOR] using dataset root as prior source: "
            f"root={prior_root} subdir={prior_subdir}"
        )
    elif hybrid_args.external_prior_root:
        prior_root = hybrid_args.external_prior_root

    ext_tokens = _parse_external_ext_tokens(hybrid_args.external_prior_exts)

    if prior_root:
        external_index = _build_external_prior_index(
            train_cameras=train_cameras,
            external_prior_root=prior_root,
            prior_subdir=prior_subdir,
            exts=tuple(ext_tokens),
        )
        prior_index.update(external_index)

    if len(prior_index) > 0:
        prior_bank = PriorTensorBank(prior_index)
    elif prior_root:
        print("[PRIOR] no usable priors found, skipping prior-guided losses.")

    if bool(getattr(hybrid_args, "sequence_loss_enable", False)):
        if not hybrid_args.sequence_lr_anchor_root:
            raise ValueError("--sequence_loss_enable requires --sequence_lr_anchor_root")
        sequence_ext_tokens = _parse_external_ext_tokens(hybrid_args.sequence_lr_anchor_exts)
        sequence_lr_anchor_index = _build_external_prior_index(
            train_cameras=train_cameras,
            external_prior_root=hybrid_args.sequence_lr_anchor_root,
            prior_subdir=hybrid_args.sequence_lr_anchor_subdir,
            exts=tuple(sequence_ext_tokens),
        )
        if (
            len(sequence_lr_anchor_index) != len(train_cameras)
            and not bool(getattr(hybrid_args, "sequence_loss_allow_missing_anchor", False))
        ):
            raise RuntimeError(
                "SequenceMatters-style loss requires LR anchors for every train view: "
                f"matched={len(sequence_lr_anchor_index)} total={len(train_cameras)} "
                f"root={hybrid_args.sequence_lr_anchor_root} "
                f"subdir={hybrid_args.sequence_lr_anchor_subdir}"
            )
        if len(sequence_lr_anchor_index) == 0:
            raise RuntimeError(
                "SequenceMatters-style loss found zero LR anchors: "
                f"root={hybrid_args.sequence_lr_anchor_root} "
                f"subdir={hybrid_args.sequence_lr_anchor_subdir}"
            )
        sequence_lr_anchor_bank = PriorTensorBank(sequence_lr_anchor_index)
        print(
            "[SEQUENCE-LOSS] enabled: "
            f"lambda_tex={hybrid_args.sequence_lambda_tex:.3f} "
            f"subpixel={hybrid_args.sequence_subpixel} "
            f"scale={hybrid_args.sequence_subpixel_scale:.3f} "
            f"anchors={len(sequence_lr_anchor_index)}/{len(train_cameras)} "
            f"root={hybrid_args.sequence_lr_anchor_root} "
            f"subdir={hybrid_args.sequence_lr_anchor_subdir}"
        )

    if prior_root and hybrid_args.external_prior_mask_subdir:
        prior_mask_index = _build_external_prior_index(
            train_cameras=train_cameras,
            external_prior_root=prior_root,
            prior_subdir=hybrid_args.external_prior_mask_subdir,
            exts=tuple(ext_tokens),
        )
        if len(prior_mask_index) > 0:
            prior_mask_bank = PriorMaskTensorBank(prior_mask_index)
        else:
            print("[PRIOR] no usable external prior masks found, falling back to consistency mask only.")

    prior_anchor_fallback_roots = []
    if hybrid_args.prior_loss_mode == "masked_residual_hf_v1" and prior_root:
        prior_anchor_fallback_roots.append(os.path.join(prior_root, "aligned_references"))
    prior_anchor_bank = _build_optional_external_bank(
        train_cameras=train_cameras,
        root_dir=hybrid_args.prior_anchor_dir,
        exts=ext_tokens,
        bank_cls=PriorTensorBank,
        label="PRIOR-ANCHOR",
        fallback_roots=prior_anchor_fallback_roots,
    )
    prior_soft_mask_bank = _build_optional_external_bank(
        train_cameras=train_cameras,
        root_dir=hybrid_args.prior_soft_mask_dir,
        exts=ext_tokens,
        bank_cls=PriorMaskTensorBank,
        label="PRIOR-SOFT-MASK",
    )
    prior_hard_mask_bank = _build_optional_external_bank(
        train_cameras=train_cameras,
        root_dir=hybrid_args.prior_hard_mask_dir,
        exts=ext_tokens,
        bank_cls=PriorMaskTensorBank,
        label="PRIOR-HARD-MASK",
    )
    if hybrid_args.prior_loss_mode == "masked_residual_hf_v1":
        print(
            "[PRIOR] masked residual HF mode: "
            f"anchor={hybrid_args.prior_anchor_dir or '<none>'} "
            f"soft={hybrid_args.prior_soft_mask_dir or '<none>'} "
            f"hard={hybrid_args.prior_hard_mask_dir or '<none>'} "
            f"soft_power={hybrid_args.prior_soft_mask_power} "
            f"hard_threshold={hybrid_args.prior_hard_mask_threshold} "
            f"min_pixels={hybrid_args.prior_masked_min_pixels}"
        )

    prior_view_train_cameras = []
    prior_viewpoint_stack = None
    prior_view_sample_prob = max(0.0, min(1.0, float(getattr(hybrid_args, "prior_view_sample_prob", 0.0))))
    if hybrid_args.prior_loss_mode == "masked_residual_hf_v1" and prior_view_sample_prob > 0.0:
        prior_view_train_cameras, prior_view_skipped = build_masked_residual_prior_camera_pool(
            train_cameras=active_train_cameras,
            prior_bank=prior_bank,
            anchor_bank=prior_anchor_bank,
            soft_mask_bank=prior_soft_mask_bank,
            hard_mask_bank=prior_hard_mask_bank,
            hard_threshold=hybrid_args.prior_hard_mask_threshold,
            min_pixels=hybrid_args.prior_masked_min_pixels,
        )
        print(
            "[PRIOR] masked residual view sampling pool: "
            f"{len(prior_view_train_cameras)}/{len(active_train_cameras)} "
            f"active train views (sample_prob={prior_view_sample_prob:.2f})"
        )
        if prior_view_skipped:
            print(f"[PRIOR] masked residual skipped samples: {prior_view_skipped}")
        if len(prior_view_train_cameras) == 0:
            print("[PRIOR] masked residual view sampling disabled because the active pool is empty.")

    need_sof_edge_touch = (
        bool(hybrid_args.prior_hf_seed_enable)
        and bool(hybrid_args.prior_edge_dir)
        and bool(hybrid_args.prior_edge_mask_dir)
    )
    if (
        float(hybrid_args.lambda_prior_local) > 0.0
        or float(hybrid_args.lambda_prior_edge) > 0.0
        or need_sof_edge_touch
    ):
        sof_prior_block = SOFPriorBlock(
            SOFPriorConfig(
                lambda_prior_local=hybrid_args.lambda_prior_local,
                prior_local_min_pixels=hybrid_args.prior_local_min_pixels,
                prior_local_from_iter=hybrid_args.prior_local_from_iter,
                lambda_prior_edge=hybrid_args.lambda_prior_edge,
                prior_edge_loss_mode=hybrid_args.prior_edge_loss_mode,
                prior_edge_blend_alpha=hybrid_args.prior_edge_blend_alpha,
                prior_edge_min_pixels=hybrid_args.prior_edge_min_pixels,
                prior_edge_from_iter=hybrid_args.prior_edge_from_iter,
                prior_edge_touch_min_radius_px=hybrid_args.prior_edge_touch_min_radius_px,
                prior_edge_touch_radius_scale=hybrid_args.prior_edge_touch_radius_scale,
                prior_edge_touch_max_radius_px=hybrid_args.prior_edge_touch_max_radius_px,
                prior_edge_detail_blur_kernel=hybrid_args.prior_edge_detail_blur_kernel,
                prior_edge_detail_alpha=hybrid_args.prior_edge_detail_alpha,
                prior_edge_detail_alpha_final=hybrid_args.prior_edge_detail_alpha_final,
                prior_edge_detail_warmup_iters=hybrid_args.prior_edge_detail_warmup_iters,
                prior_edge_detail_weight=hybrid_args.prior_edge_detail_weight,
                prior_edge_lowfreq_weight=hybrid_args.prior_edge_lowfreq_weight,
                prior_edge_grad_weight=hybrid_args.prior_edge_grad_weight,
                prior_edge_lowfreq_threshold=hybrid_args.prior_edge_lowfreq_threshold,
                prior_edge_lowfreq_anchor=hybrid_args.prior_edge_lowfreq_anchor,
                prior_edge_detail_min_gain=hybrid_args.prior_edge_detail_min_gain,
                prior_edge_confidence_power=hybrid_args.prior_edge_confidence_power,
            )
        )
        print(
            "[SOF-PRIOR] block enabled: "
            f"local={hybrid_args.lambda_prior_local:.3f} "
            f"edge={hybrid_args.lambda_prior_edge:.3f} "
            f"mode={hybrid_args.prior_edge_loss_mode} "
            f"local_from={hybrid_args.prior_local_from_iter} "
            f"edge_from={hybrid_args.prior_edge_from_iter} "
            f"seed_touch_only={1 if need_sof_edge_touch and float(hybrid_args.lambda_prior_edge) <= 0.0 else 0}"
        )

    if float(hybrid_args.lambda_prior_local) > 0.0:
        if hybrid_args.prior_local_dir and hybrid_args.prior_local_mask_dir:
            prior_local_bank = _build_optional_external_bank(
                train_cameras=train_cameras,
                root_dir=hybrid_args.prior_local_dir,
                exts=ext_tokens,
                bank_cls=PriorTensorBank,
                label="SOF-PRIOR-LOCAL",
            )
            prior_local_mask_bank = _build_optional_external_bank(
                train_cameras=train_cameras,
                root_dir=hybrid_args.prior_local_mask_dir,
                exts=ext_tokens,
                bank_cls=PriorMaskTensorBank,
                label="SOF-PRIOR-LOCAL-MASK",
            )
        else:
            print("[SOF-PRIOR] local loss requested but prior_local_dir/mask_dir is incomplete; disabling local branch.")

    if float(hybrid_args.lambda_prior_edge) > 0.0 or need_sof_edge_touch:
        if hybrid_args.prior_edge_dir and hybrid_args.prior_edge_mask_dir:
            prior_edge_bank = _build_optional_external_bank(
                train_cameras=train_cameras,
                root_dir=hybrid_args.prior_edge_dir,
                exts=ext_tokens,
                bank_cls=PriorTensorBank,
                label="SOF-PRIOR-EDGE",
            )
            prior_edge_mask_bank = _build_optional_external_bank(
                train_cameras=train_cameras,
                root_dir=hybrid_args.prior_edge_mask_dir,
                exts=ext_tokens,
                bank_cls=PriorMaskTensorBank,
                label="SOF-PRIOR-EDGE-MASK",
            )
        else:
            print("[SOF-PRIOR] edge branch requested but prior_edge_dir/mask_dir is incomplete; disabling edge branch.")

    if float(hybrid_args.lambda_surface_route_consensus) > 0.0:
        if hybrid_args.surface_route_consensus_root:
            surface_route_bank = SurfaceRouteConsensusBank(hybrid_args.surface_route_consensus_root)
            if len(surface_route_bank.index) > 0:
                print(
                    "[SURFACE-ROUTE] route consensus enabled: "
                    f"root={hybrid_args.surface_route_consensus_root} "
                    f"views={len(surface_route_bank.index)} "
                    f"lambda={hybrid_args.lambda_surface_route_consensus:.3f} "
                    f"from={hybrid_args.surface_route_consensus_from_iter} "
                    f"surface_only={int(bool(hybrid_args.surface_route_surface_only))}"
                )
            else:
                print(
                    "[SURFACE-ROUTE] route consensus root has no usable .pt payloads; "
                    "disabling route loss."
                )
                surface_route_bank = None
        else:
            print("[SURFACE-ROUTE] lambda_surface_route_consensus > 0 but root is empty; disabling route loss.")

    external_update_mask = load_gaussian_update_mask_payload(
        hybrid_args.optimize_gaussian_mask_payload,
        hybrid_args.optimize_gaussian_mask_key,
        total_gaussians=gaussians.get_xyz.shape[0],
    )
    prior_loss_update_mask = load_gaussian_update_mask_payload(
        hybrid_args.prior_loss_gaussian_mask_payload,
        hybrid_args.prior_loss_gaussian_mask_key,
        total_gaussians=gaussians.get_xyz.shape[0],
    )
    prior_loss_update_mask_dynamic_roots = bool(
        getattr(hybrid_args, "prior_loss_gaussian_mask_dynamic_roots", False)
    )
    if external_update_mask is not None:
        selected_count = int(external_update_mask.sum().item())
        print(
            f"[SOF-PRIOR] gaussian update mask '{hybrid_args.optimize_gaussian_mask_key}': "
            f"{selected_count}/{external_update_mask.shape[0]}"
        )
        if selected_count <= 0:
            raise ValueError("External Gaussian update mask is empty; regenerate the edge-region GS payload.")
    if prior_loss_update_mask is not None:
        selected_count = int(prior_loss_update_mask.sum().item())
        print(
            f"[SOF-PRIOR] prior-loss gaussian mask '{hybrid_args.prior_loss_gaussian_mask_key}': "
            f"{selected_count}/{prior_loss_update_mask.shape[0]} "
            f"dynamic_roots={int(prior_loss_update_mask_dynamic_roots)}"
        )
        if selected_count <= 0:
            raise ValueError("Prior-loss Gaussian mask is empty; regenerate the surface-state payload.")
        if prior_loss_update_mask_dynamic_roots:
            initialize_gaussian_root_ids_for_dynamic_masks(
                gaussians,
                label="prior-loss gaussian mask dynamic roots",
            )
    prior_loss_mask_covers_all = bool(
        prior_loss_update_mask is not None
        and int(prior_loss_update_mask.sum().item()) == int(prior_loss_update_mask.shape[0])
    )

    surface_route_gaussians = None
    if surface_route_bank is not None and bool(hybrid_args.surface_route_surface_only):
        surface_route_mask = combine_gaussian_update_masks(prior_loss_update_mask, external_update_mask)
        if surface_route_mask is None:
            print(
                "[SURFACE-ROUTE] surface-only route branch requested but no gaussian mask is available; "
                "falling back to full-render route supervision."
            )
        elif int(surface_route_mask.sum().item()) <= 0:
            print(
                "[SURFACE-ROUTE] surface-only route branch requested but the effective mask is empty; "
                "falling back to full-render route supervision."
            )
        elif int(surface_route_mask.sum().item()) == int(surface_route_mask.shape[0]):
            surface_route_gaussians = gaussians
            print(
                "[SURFACE-ROUTE] surface-only render branch covers all gaussians; "
                "using the full Gaussian set for route supervision."
            )
        else:
            surface_route_gaussians = GaussianSubsetView(gaussians, surface_route_mask)
            print(
                "[SURFACE-ROUTE] surface-only render branch enabled: "
                f"{int(surface_route_mask.sum().item())}/{surface_route_mask.shape[0]} gaussians"
            )

    if prior_loss_mask_covers_all and external_update_mask is None:
        print(
            "[SOF-PRIOR] prior-loss mask covers all gaussians; "
            "treating prior-guided losses as full-field updates."
        )
        prior_loss_update_mask = None

    layer_frequency_requested = (
        float(hybrid_args.lambda_non_surface_hf) > 0.0
        or float(hybrid_args.lambda_non_surface_rgb_energy) > 0.0
        or float(hybrid_args.lambda_non_surface_alpha_hf) > 0.0
        or float(hybrid_args.lambda_non_surface_alpha_mass) > 0.0
        or float(hybrid_args.lambda_surface_hf_closure) > 0.0
        or float(hybrid_args.lambda_surface_start_hf_preserve) > 0.0
    )
    prior_direct_xyz_nudge_mask = None
    prior_direct_xyz_nudge_requested = float(hybrid_args.prior_direct_xyz_nudge_lr) > 0.0
    if layer_frequency_requested:
        layer_payload = str(hybrid_args.layer_frequency_mask_payload or "").strip()
        if not layer_payload:
            layer_payload = str(hybrid_args.prior_loss_gaussian_mask_payload or "").strip()
        layer_frequency_dynamic_roots = bool(getattr(hybrid_args, "layer_frequency_dynamic_roots", False))
        layer_non_surface_mask = None
        layer_surface_mask = None
        needs_non_surface_mask = (
            float(hybrid_args.lambda_non_surface_hf) > 0.0
            or float(hybrid_args.lambda_non_surface_rgb_energy) > 0.0
            or float(hybrid_args.lambda_non_surface_alpha_hf) > 0.0
            or float(hybrid_args.lambda_non_surface_alpha_mass) > 0.0
        )
        if not layer_payload:
            print(
                "[LAYER-FREQ] requested but no layer_frequency_mask_payload was provided; "
                "layer-frequency regularization is disabled."
            )
        else:
            if needs_non_surface_mask:
                layer_non_surface_mask = load_gaussian_update_mask_payload(
                    layer_payload,
                    str(hybrid_args.layer_frequency_non_surface_key),
                    total_gaussians=gaussians.get_xyz.shape[0],
                )
            needs_surface_mask = (
                float(hybrid_args.lambda_surface_hf_closure) > 0.0
                or float(hybrid_args.lambda_surface_start_hf_preserve) > 0.0
            )
            if needs_surface_mask:
                layer_surface_mask = load_gaussian_update_mask_payload(
                    layer_payload,
                    str(hybrid_args.layer_frequency_surface_key),
                    total_gaussians=gaussians.get_xyz.shape[0],
                )
            if layer_frequency_dynamic_roots:
                initialize_gaussian_root_ids_for_dynamic_masks(
                    gaussians,
                    label="layer-frequency dynamic roots",
                )
            layer_frequency_regularizer = LayerFrequencyRegularizer(
                gaussians=gaussians,
                non_surface_mask=layer_non_surface_mask,
                surface_mask=layer_surface_mask,
                lambda_non_surface_hf=float(hybrid_args.lambda_non_surface_hf),
                lambda_non_surface_rgb_energy=float(hybrid_args.lambda_non_surface_rgb_energy),
                lambda_non_surface_alpha_hf=float(hybrid_args.lambda_non_surface_alpha_hf),
                lambda_non_surface_alpha_mass=float(hybrid_args.lambda_non_surface_alpha_mass),
                lambda_surface_hf_closure=float(hybrid_args.lambda_surface_hf_closure),
                lambda_surface_start_hf_preserve=float(hybrid_args.lambda_surface_start_hf_preserve),
                start_hf_lowfreq_kernel=int(hybrid_args.layer_frequency_start_hf_lowfreq_kernel),
                start_hf_lowfreq_threshold=float(hybrid_args.layer_frequency_start_hf_lowfreq_threshold),
                start_hf_energy_threshold=float(hybrid_args.layer_frequency_start_hf_energy_threshold),
                start_hf_mask_power=float(hybrid_args.layer_frequency_start_hf_mask_power),
                start_hf_protect_non_surface=bool(hybrid_args.layer_frequency_start_hf_protect_non_surface),
                from_iter=int(hybrid_args.layer_frequency_from_iter),
                until_iter=int(hybrid_args.layer_frequency_until_iter),
                dynamic_roots=layer_frequency_dynamic_roots,
            )
            if float(hybrid_args.lambda_surface_start_hf_preserve) > 0.0:
                start_hf_checkpoint = str(hybrid_args.layer_frequency_start_hf_checkpoint or "").strip()
                if not start_hf_checkpoint:
                    start_hf_checkpoint = str(checkpoint or "").strip()
                if not start_hf_checkpoint:
                    raise ValueError(
                        "lambda_surface_start_hf_preserve > 0 requires "
                        "--layer_frequency_start_hf_checkpoint or --start_checkpoint."
                    )
                if not os.path.isfile(start_hf_checkpoint):
                    raise FileNotFoundError(f"Layer-frequency start-HF checkpoint not found: {start_hf_checkpoint}")
                layer_frequency_start_gaussians, start_hf_iter = load_frozen_gaussian_checkpoint(
                    start_hf_checkpoint,
                    dataset.sh_degree,
                )
                with torch.no_grad():
                    layer_frequency_start_gaussians.compute_3D_filter(cameras=train_cameras)
                print(
                    "[LAYER-FREQ] start-HF preserve enabled: "
                    f"checkpoint={start_hf_checkpoint} "
                    f"iter={start_hf_iter} "
                    f"lambda={float(hybrid_args.lambda_surface_start_hf_preserve):.6f} "
                    f"lf_kernel={int(hybrid_args.layer_frequency_start_hf_lowfreq_kernel)} "
                    f"lf_thr={float(hybrid_args.layer_frequency_start_hf_lowfreq_threshold):.6f} "
                    f"energy_thr={float(hybrid_args.layer_frequency_start_hf_energy_threshold):.6f} "
                    f"mask_power={float(hybrid_args.layer_frequency_start_hf_mask_power):.3f} "
                    f"protect_ns={int(bool(hybrid_args.layer_frequency_start_hf_protect_non_surface))}"
                )
            ns_count = int(layer_non_surface_mask.sum().item()) if layer_non_surface_mask is not None else 0
            surf_count = int(layer_surface_mask.sum().item()) if layer_surface_mask is not None else 0
            total_count = int(gaussians.get_xyz.shape[0])
            print(
                "[LAYER-FREQ] enabled: "
                f"payload={layer_payload} "
                f"non_surface={str(hybrid_args.layer_frequency_non_surface_key)}:{ns_count}/{total_count} "
                f"surface={str(hybrid_args.layer_frequency_surface_key)}:{surf_count}/{total_count} "
                f"lambda_ns_hf={float(hybrid_args.lambda_non_surface_hf):.6f} "
                f"lambda_ns_rgb={float(hybrid_args.lambda_non_surface_rgb_energy):.6f} "
                f"lambda_ns_alpha_hf={float(hybrid_args.lambda_non_surface_alpha_hf):.6f} "
                f"lambda_ns_alpha={float(hybrid_args.lambda_non_surface_alpha_mass):.6f} "
                f"lambda_surf_hf={float(hybrid_args.lambda_surface_hf_closure):.6f} "
                f"lambda_start_hf={float(hybrid_args.lambda_surface_start_hf_preserve):.6f} "
                f"surface_target={str(getattr(hybrid_args, 'layer_frequency_surface_target', 'gt'))} "
                f"from={int(hybrid_args.layer_frequency_from_iter)} "
                f"until={int(hybrid_args.layer_frequency_until_iter)} "
                f"dynamic_roots={int(layer_frequency_dynamic_roots)}"
            )
            if needs_non_surface_mask and not layer_frequency_regularizer.has_non_surface_mask:
                print("[LAYER-FREQ] warning: non-surface mask is empty; non-surface HF terms are disabled.")
            if needs_surface_mask and not layer_frequency_regularizer.has_surface_mask:
                print("[LAYER-FREQ] warning: surface mask is empty; surface HF terms are disabled.")

    if bool(hybrid_args.prior_bubble_cleanup_enable):
        prior_bubble_cleanup_regularizer = PriorBubbleCleanupRegularizer(
            opacity_min=float(hybrid_args.prior_bubble_cleanup_opacity_min),
            max_axis_min=float(hybrid_args.prior_bubble_cleanup_max_axis_min),
            max_axis_max=float(hybrid_args.prior_bubble_cleanup_max_axis_max),
            anisotropy_max=float(hybrid_args.prior_bubble_cleanup_anisotropy_max),
            min_generation=int(hybrid_args.prior_bubble_cleanup_min_generation),
            lambda_hf=float(hybrid_args.lambda_prior_bubble_hf),
            lambda_rgb_energy=float(hybrid_args.lambda_prior_bubble_rgb_energy),
            lambda_alpha_hf=float(hybrid_args.lambda_prior_bubble_alpha_hf),
            lambda_alpha_mass=float(hybrid_args.lambda_prior_bubble_alpha_mass),
            post_opacity_decay=float(hybrid_args.prior_bubble_cleanup_post_opacity_decay),
            from_iter=int(hybrid_args.prior_bubble_cleanup_from_iter),
            until_iter=int(hybrid_args.prior_bubble_cleanup_until_iter),
            interval=int(hybrid_args.prior_bubble_cleanup_interval),
        )
        print(
            "[PRIOR-BUBBLE] dynamic NoSR-style cleanup enabled: "
            f"opacity>={float(hybrid_args.prior_bubble_cleanup_opacity_min):.4f} "
            f"max_axis=[{float(hybrid_args.prior_bubble_cleanup_max_axis_min):.6f},"
            f"{float(hybrid_args.prior_bubble_cleanup_max_axis_max):.6f}] "
            f"anis<={float(hybrid_args.prior_bubble_cleanup_anisotropy_max):.3f} "
            f"gen>={int(hybrid_args.prior_bubble_cleanup_min_generation)} "
            f"lambda_hf={float(hybrid_args.lambda_prior_bubble_hf):.6f} "
            f"lambda_rgb={float(hybrid_args.lambda_prior_bubble_rgb_energy):.6f} "
            f"lambda_alpha_hf={float(hybrid_args.lambda_prior_bubble_alpha_hf):.6f} "
            f"lambda_alpha={float(hybrid_args.lambda_prior_bubble_alpha_mass):.6f} "
            f"post_decay={float(hybrid_args.prior_bubble_cleanup_post_opacity_decay):.3f} "
            f"from={int(hybrid_args.prior_bubble_cleanup_from_iter)} "
            f"until={int(hybrid_args.prior_bubble_cleanup_until_iter)} "
            f"interval={int(hybrid_args.prior_bubble_cleanup_interval)}"
        )

    if prior_direct_xyz_nudge_requested:
        if (
            layer_frequency_regularizer is not None
            and layer_frequency_regularizer.surface_mask is not None
            and not bool(getattr(layer_frequency_regularizer, "dynamic_roots", False))
        ):
            prior_direct_xyz_nudge_mask = combine_gaussian_update_masks(
                layer_frequency_regularizer.surface_mask,
                external_update_mask,
            )
            print(
                "[PRIOR-XYZ-NUDGE] enabled: "
                f"mask=layer_surface:{int(prior_direct_xyz_nudge_mask.sum().item())}/"
                f"{prior_direct_xyz_nudge_mask.shape[0]} "
                f"lr={float(hybrid_args.prior_direct_xyz_nudge_lr):.6f} "
                f"max_step={float(hybrid_args.prior_direct_xyz_nudge_max_step):.6f}"
            )
        else:
            print(
                "[PRIOR-XYZ-NUDGE] enabled with dynamic update mask: "
                "no layer-frequency surface mask is available, so the current source/mask update filter "
                "will be used each iteration. "
                f"lr={float(hybrid_args.prior_direct_xyz_nudge_lr):.6f} "
                f"max_step={float(hybrid_args.prior_direct_xyz_nudge_max_step):.6f}"
            )

    surface_migration_requested = (
        float(hybrid_args.lambda_surface_migration_normal) > 0.0
        or float(hybrid_args.lambda_surface_migration_tangent) > 0.0
        or float(hybrid_args.lambda_surface_migration_normal_align) > 0.0
        or float(hybrid_args.lambda_surface_migration_thin) > 0.0
        or float(hybrid_args.surface_migration_post_step_normal_alpha) > 0.0
        or float(hybrid_args.surface_migration_post_step_tangent_alpha) > 0.0
    )
    if surface_migration_requested:
        migration_payload = str(hybrid_args.surface_migration_payload or "").strip()
        if not migration_payload:
            migration_payload = str(hybrid_args.layer_frequency_mask_payload or "").strip()
        if not migration_payload:
            migration_payload = str(hybrid_args.prior_loss_gaussian_mask_payload or "").strip()
        if not migration_payload:
            raise ValueError(
                "Surface migration requested but no payload was provided. "
                "Set --surface_migration_payload or LAYER_FREQUENCY_MASK_PAYLOAD."
            )
        migration_mask = load_gaussian_update_mask_payload(
            migration_payload,
            str(hybrid_args.surface_migration_mask_key),
            total_gaussians=gaussians.get_xyz.shape[0],
        )
        migration_normals, migration_anchors = _load_surface_normal_lock_payload(
            migration_payload,
            str(hybrid_args.surface_migration_normal_key),
            str(hybrid_args.surface_migration_anchor_key),
            total_gaussians=gaussians.get_xyz.shape[0],
        )
        migration_anchor_source = "payload"
        if migration_normals is None or migration_anchors is None:
            migration_anchor_source_mask = load_gaussian_update_mask_payload(
                migration_payload,
                str(hybrid_args.surface_migration_anchor_source_key),
                total_gaussians=gaussians.get_xyz.shape[0],
            )
            migration_normals, migration_anchors = _build_nearest_surface_migration_anchors(
                gaussians=gaussians,
                migration_mask=migration_mask,
                surface_mask=migration_anchor_source_mask,
                payload_normals=migration_normals,
            )
            migration_anchor_source = f"nearest:{str(hybrid_args.surface_migration_anchor_source_key)}"
        surface_migration_regularizer = SurfaceMigrationRegularizer(
            gaussians=gaussians,
            mask=migration_mask,
            normals=migration_normals,
            anchors=migration_anchors,
            lambda_normal=float(hybrid_args.lambda_surface_migration_normal),
            lambda_tangent=float(hybrid_args.lambda_surface_migration_tangent),
            lambda_normal_align=float(hybrid_args.lambda_surface_migration_normal_align),
            lambda_thin=float(hybrid_args.lambda_surface_migration_thin),
            target_normal_coord=float(hybrid_args.surface_migration_target_normal_coord),
            thin_target_ratio=float(hybrid_args.surface_migration_thin_target_ratio),
            post_step_normal_alpha=float(hybrid_args.surface_migration_post_step_normal_alpha),
            post_step_tangent_alpha=float(hybrid_args.surface_migration_post_step_tangent_alpha),
            from_iter=int(hybrid_args.surface_migration_from_iter),
            until_iter=int(hybrid_args.surface_migration_until_iter),
        )
        print(
            "[SURFACE-MIGRATE] enabled: "
            f"payload={migration_payload} "
            f"mask={str(hybrid_args.surface_migration_mask_key)}:"
            f"{int(migration_mask.sum().item())}/{migration_mask.shape[0]} "
            f"anchor_source={migration_anchor_source} "
            f"target_normal={float(hybrid_args.surface_migration_target_normal_coord):.6f} "
            f"lambda_normal={float(hybrid_args.lambda_surface_migration_normal):.6f} "
            f"lambda_tangent={float(hybrid_args.lambda_surface_migration_tangent):.6f} "
            f"lambda_align={float(hybrid_args.lambda_surface_migration_normal_align):.6f} "
            f"lambda_thin={float(hybrid_args.lambda_surface_migration_thin):.6f} "
            f"post_normal={float(hybrid_args.surface_migration_post_step_normal_alpha):.6f} "
            f"post_tangent={float(hybrid_args.surface_migration_post_step_tangent_alpha):.6f} "
            f"from={int(hybrid_args.surface_migration_from_iter)} "
            f"until={int(hybrid_args.surface_migration_until_iter)}"
        )

    if bool(hybrid_args.surface_filter_3d):
        surface_filter_mask = None
        surface_filter_payload = str(hybrid_args.surface_filter_3d_payload or "").strip()
        if surface_filter_payload:
            surface_filter_mask = load_gaussian_update_mask_payload(
                surface_filter_payload,
                str(hybrid_args.surface_filter_3d_key),
                total_gaussians=gaussians.get_xyz.shape[0],
            )
        else:
            surface_filter_mask = combine_gaussian_update_masks(prior_loss_update_mask, external_update_mask)

        if surface_filter_mask is None:
            print(
                "[SURFACE-FILTER-3D] requested but no gaussian mask is available; "
                "surface-enhanced mip filter is disabled."
            )
        else:
            surface_filter_3d = SurfaceFilter3D(
                mask=surface_filter_mask,
                scale=float(hybrid_args.surface_filter_3d_scale),
                min_value=float(hybrid_args.surface_filter_3d_min),
            )
            surface_filter_3d.apply_after_compute(gaussians)
            print(
                "[SURFACE-FILTER-3D] enabled: "
                f"{int(surface_filter_mask.sum().item())}/{surface_filter_mask.shape[0]} gaussians "
                f"key={hybrid_args.surface_filter_3d_key} "
                f"mode=global_plus_surface_extra "
                f"scale={float(hybrid_args.surface_filter_3d_scale):.6f} "
                f"min={float(hybrid_args.surface_filter_3d_min):.6f}"
            )

    surface_normal_lock = None
    if bool(hybrid_args.surface_normal_lock):
        normal_lock_dynamic_roots = bool(getattr(hybrid_args, "surface_normal_lock_dynamic_roots", False))
        normal_lock_mask = combine_gaussian_update_masks(prior_loss_update_mask, external_update_mask)
        if normal_lock_mask is None and prior_loss_mask_covers_all:
            if external_update_mask is not None:
                normal_lock_mask = external_update_mask.to(device="cuda", dtype=torch.bool)
            else:
                normal_lock_mask = torch.ones(
                    (int(gaussians.get_xyz.shape[0]),),
                    device=gaussians.get_xyz.device,
                    dtype=torch.bool,
                )
        if normal_lock_mask is None:
            print(
                "[SURFACE-NORMAL-LOCK] requested but no gaussian mask is available; "
                "normal locking is disabled."
            )
        else:
            normal_lock_payload = str(hybrid_args.surface_normal_lock_payload or "").strip()
            if not normal_lock_payload:
                normal_lock_payload = str(hybrid_args.prior_loss_gaussian_mask_payload or "").strip()
            lock_normals, lock_anchors = _load_surface_normal_lock_payload(
                normal_lock_payload,
                str(hybrid_args.surface_normal_lock_normal_key),
                str(hybrid_args.surface_normal_lock_anchor_key),
                total_gaussians=int(gaussians.get_xyz.shape[0]),
            ) if normal_lock_payload else (None, None)
            if normal_lock_dynamic_roots:
                initialize_gaussian_root_ids_for_dynamic_masks(
                    gaussians,
                    label="surface normal-lock dynamic roots",
                )
            surface_normal_lock = SurfaceNormalLock(
                gaussians,
                mask=normal_lock_mask,
                normals=lock_normals,
                anchors=lock_anchors,
                dynamic_roots=normal_lock_dynamic_roots,
            )
            print(
                "[SURFACE-NORMAL-LOCK] enabled: "
                f"{int(normal_lock_mask.sum().item())}/{normal_lock_mask.shape[0]} gaussians "
                f"normal_source={surface_normal_lock.normal_source} "
                f"anchor_source={surface_normal_lock.anchor_source} "
                f"dynamic_roots={int(normal_lock_dynamic_roots)}"
            )

    if bool(hybrid_args.prior_only_edge_finetune):
        if hybrid_args.lambda_prior_edge <= 0.0:
            raise ValueError("--prior_only_edge_finetune requires --lambda_prior_edge > 0")
        if prior_edge_bank is None or not hybrid_args.prior_edge_mask_dir:
            raise ValueError("--prior_only_edge_finetune requires --prior_edge_dir and --prior_edge_mask_dir")
        if prior_loss_update_mask is not None:
            raise ValueError(
                "--prior_only_edge_finetune cannot be combined with "
                "--prior_loss_gaussian_mask_payload; use --optimize_gaussian_mask_payload instead."
            )
        bootstrap_source_tags = {"prior", "added"}
        initial_source_update_mask = build_source_tag_train_mask(gaussians, hybrid_args.optimize_source_tag)
        initial_trainable_count = count_trainable_source_tag_gaussians(gaussians, initial_source_update_mask)
        can_bootstrap_from_hf_seed = bool(hybrid_args.prior_hf_seed_enable) and (
            hybrid_args.optimize_source_tag in bootstrap_source_tags
        )
        if external_update_mask is not None and (
            first_iter < int(opt.densify_until_iter) or bool(hybrid_args.prior_hf_seed_enable)
        ):
            raise ValueError(
                "--prior_only_edge_finetune with a fixed --optimize_gaussian_mask_payload cannot be combined with "
                "densify/HF seed restarts because the Gaussian count changes at runtime. Remove the external mask "
                "and rely on source-tag gating, or disable count-changing ops."
            )
        if external_update_mask is None and initial_trainable_count <= 0 and not can_bootstrap_from_hf_seed:
            raise ValueError(
                "--prior_only_edge_finetune requires --optimize_gaussian_mask_payload unless the restart already "
                "contains trainable gaussians for --optimize_source_tag or HF seeding is enabled for "
                "--optimize_source_tag in {prior,added}."
            )
        active_train_cameras, missing_prior_samples, missing_mask_samples = build_prior_edge_camera_pool(
            train_cameras,
            prior_edge_bank.index,
            prior_edge_mask_bank.index if prior_edge_mask_bank is not None else hybrid_args.prior_edge_mask_dir,
        )
        print(
            f"[SOF-PRIOR] prior-only camera pool: {len(active_train_cameras)}/{len(train_cameras)} "
            "train views with both prior image and edge mask"
        )
        if missing_prior_samples:
            print(f"[SOF-PRIOR] missing prior samples: {missing_prior_samples}")
        if missing_mask_samples:
            print(f"[SOF-PRIOR] missing edge-mask samples: {missing_mask_samples}")
        if external_update_mask is None:
            if initial_trainable_count > 0:
                print(
                    "[SOF-PRIOR] prior-only edge finetune without external GS mask: "
                    f"reusing existing source-tag subset ({initial_trainable_count} gaussians)."
                )
            elif can_bootstrap_from_hf_seed:
                print(
                    "[SOF-PRIOR] prior-only edge finetune without external GS mask: "
                    "bootstrapping trainable gaussians from HF seed births."
                )
        if len(active_train_cameras) == 0:
            raise ValueError("No train cameras have both prior images and edge masks for prior-only edge finetune.")

    surface_route_train_cameras = []
    surface_route_viewpoint_stack = None
    route_view_sample_prob = float(getattr(hybrid_args, "surface_route_view_sample_prob", 0.0))
    route_view_sample_prob = max(0.0, min(1.0, route_view_sample_prob))

    if hybrid_args.prior_supervision_images_subdir:
        prior_supervision_camera_bank = PriorSupervisionCameraBank(
            dataset=dataset,
            images_subdir=hybrid_args.prior_supervision_images_subdir,
            resolution=hybrid_args.prior_supervision_resolution,
            transforms_llffhold=hybrid_args.transforms_llffhold,
            cache_size=hybrid_args.prior_supervision_cache_size,
        )
        print(
            "[PRIOR] HR supervision cameras enabled: "
            f"images={hybrid_args.prior_supervision_images_subdir} "
            f"resolution={hybrid_args.prior_supervision_resolution} "
            f"cache={hybrid_args.prior_supervision_cache_size} "
            f"views={len(prior_supervision_camera_bank.cam_infos_by_name)}"
        )

    if surface_route_bank is not None:
        surface_route_train_cameras = [
            camera for camera in active_train_cameras if surface_route_bank.has_view(camera.image_name)
        ]
        if len(surface_route_train_cameras) > 0:
            print(
                "[SURFACE-ROUTE] route-view sampling pool: "
                f"{len(surface_route_train_cameras)}/{len(active_train_cameras)} active train views "
                f"(sample_prob={route_view_sample_prob:.2f})"
            )
        else:
            print(
                "[SURFACE-ROUTE] no overlap between active train views and route payload views; "
                "route-view oversampling disabled."
            )

    prior_view_curriculum_cameras = []
    if bool(getattr(hybrid_args, "prior_view_curriculum_enable", False)):
        prior_view_curriculum_cameras = _build_prior_view_curriculum_cameras(
            active_train_cameras,
            prior_bank,
            surface_route_bank,
            int(getattr(hybrid_args, "prior_view_curriculum_max_views", 0)),
        )
        if len(prior_view_curriculum_cameras) > 0:
            print(
                "[VIEW-CURRICULUM] enabled: "
                f"views={len(prior_view_curriculum_cameras)} "
                f"start={int(hybrid_args.prior_view_curriculum_start_iter)} "
                f"primary={int(hybrid_args.prior_view_curriculum_primary_iters)} "
                f"neighbor={int(hybrid_args.prior_view_curriculum_neighbor_iters)} "
                f"radius={int(hybrid_args.prior_view_curriculum_neighbor_radius)} "
                f"settle={int(hybrid_args.prior_view_curriculum_settle_iters)} "
                f"birth_primary_only={int(bool(hybrid_args.prior_view_curriculum_birth_primary_only))}"
            )
        else:
            print("[VIEW-CURRICULUM] requested but no active train views have prior/route payloads; disabling.")

    if (
        float(hybrid_args.opacity_reg) > 0.0
        or float(hybrid_args.scale_reg) > 0.0
        or float(hybrid_args.min_scale_reg) > 0.0
        or float(hybrid_args.lambda_distortion) > 0.0
        or float(hybrid_args.lambda_depth_normal) > 0.0
        or float(hybrid_args.lambda_smoothness) > 0.0
        or float(hybrid_args.lambda_extent) > 0.0
        or float(hybrid_args.lambda_opacity_field) > 0.0
        or float(hybrid_args.lambda_surface_thin) > 0.0
    ):
        sof_regularization_block = SOFRegularizationBlock(
            SOFRegularizationConfig(
                opacity_reg=hybrid_args.opacity_reg,
                scale_reg=hybrid_args.scale_reg,
                min_scale_reg=hybrid_args.min_scale_reg,
                lambda_distortion=hybrid_args.lambda_distortion,
                lambda_depth_normal=hybrid_args.lambda_depth_normal,
                lambda_smoothness=hybrid_args.lambda_smoothness,
                lambda_extent=hybrid_args.lambda_extent,
                lambda_opacity_field=hybrid_args.lambda_opacity_field,
                distortion_from_iter=hybrid_args.distortion_from_iter,
                depth_normal_from_iter=hybrid_args.depth_normal_from_iter,
                lambda_surface_thin=hybrid_args.lambda_surface_thin,
                surface_thin_mesh_path=hybrid_args.surface_thin_mesh_path,
                surface_thin_from_iter=hybrid_args.surface_thin_from_iter,
                surface_thin_until_iter=hybrid_args.surface_thin_until_iter,
                surface_thin_sample_count=hybrid_args.surface_thin_sample_count,
                surface_thin_update_interval=hybrid_args.surface_thin_update_interval,
                surface_thin_gaussian_sample_count=hybrid_args.surface_thin_gaussian_sample_count,
                surface_thin_offset_margin=hybrid_args.surface_thin_offset_margin,
                surface_thin_normal_scale_target=hybrid_args.surface_thin_normal_scale_target,
                surface_thin_normal_scale_weight=hybrid_args.surface_thin_normal_scale_weight,
            )
        )
        print(
            "[SOF-REG] block enabled: "
            f"opacity={hybrid_args.opacity_reg:.4f} "
            f"scale={hybrid_args.scale_reg:.4f} "
            f"min_scale={hybrid_args.min_scale_reg:.4f} "
            f"dist={hybrid_args.lambda_distortion:.4f} "
            f"depth_norm={hybrid_args.lambda_depth_normal:.4f} "
            f"smooth={hybrid_args.lambda_smoothness:.4f} "
            f"extent={hybrid_args.lambda_extent:.4f} "
            f"opacity_field={hybrid_args.lambda_opacity_field:.4f} "
            f"surface_thin={hybrid_args.lambda_surface_thin:.4f}"
        )
        print(
            "[SOF-REG] rasterizer mode: "
            + (
                "vanilla SOF single-pass aux"
                if getattr(pipe, "use_vanilla_sof_rasterizer", False)
                else (
                    "merged single-pass aux"
                    if supports_merged_sof_rasterizer() and getattr(pipe, "use_merged_sof_rasterizer", False)
                    else "portable fallback passes"
                )
            )
        )

    merged_splat_args = None
    vanilla_splat_args = None
    train_render_fn = render
    if getattr(pipe, "use_vanilla_sof_rasterizer", False) and (
        getattr(pipe, "use_merged_sof_rasterizer", False) or getattr(pipe, "require_merged_sof_aux", False)
    ):
        raise RuntimeError(
            "Vanilla SOF rasterizer cannot be combined with merged SOF rasterizer or require_merged_sof_aux."
        )
    if getattr(pipe, "use_vanilla_sof_rasterizer", False):
        vanilla_splat_args = _make_vanilla_sof_training_settings()
        if vanilla_splat_args is None or not supports_vanilla_sof_rasterizer():
            raise RuntimeError(
                "Vanilla SOF rasterizer was requested, but diff_gaussian_rasterization_sof_vanilla "
                "is unavailable. Install the vanilla package before running this mode."
            )
        train_render_fn = partial(render, splat_args=vanilla_splat_args)
        print(
            "[SOF-REG] vanilla settings: "
            f"load_balancing={int(bool(vanilla_splat_args.load_balancing))} "
            f"proper_ewa={int(bool(vanilla_splat_args.proper_ewa_scaling))} "
            f"exact_depth={int(bool(vanilla_splat_args.exact_depth))} "
            f"detach_alpha={int(bool(vanilla_splat_args.detach_alpha))} "
            f"detach_alpha_extent={int(bool(vanilla_splat_args.detach_alpha_extent))} "
            f"include_alpha={int(bool(vanilla_splat_args.include_alpha))}"
        )
    elif getattr(pipe, "use_merged_sof_rasterizer", False) or getattr(pipe, "require_merged_sof_aux", False):
        merged_splat_args = _make_merged_sof_training_settings()
        if merged_splat_args is None:
            raise RuntimeError("Merged SOF rasterizer was requested, but ExtendedSettings is unavailable.")
        train_render_fn = partial(render, splat_args=merged_splat_args)
        print(
            "[SOF-REG] merged settings: "
            f"load_balancing={int(bool(merged_splat_args.load_balancing))} "
            f"proper_ewa={int(bool(merged_splat_args.proper_ewa_scaling))} "
            f"exact_depth={int(bool(merged_splat_args.exact_depth))} "
            f"detach_alpha={int(bool(merged_splat_args.detach_alpha))} "
            f"detach_alpha_extent={int(bool(merged_splat_args.detach_alpha_extent))} "
            f"include_alpha={int(bool(merged_splat_args.include_alpha))}"
        )

    if hybrid_args.sdf_densify_enable:
        if sdf_adapter is None:
            print("[SDF-DENSIFY] requested but no SDF adapter is available.")
        else:
            sdf_densify_block = SDFDensifyBlock(
                SDFDensifyConfig(
                    interval=hybrid_args.sdf_densify_interval,
                    topk=hybrid_args.sdf_densify_topk,
                    min_score=hybrid_args.sdf_densify_min_score,
                    surface_coef=hybrid_args.sdf_densify_surface_coef,
                    normal_coef=hybrid_args.sdf_densify_normal_coef,
                    offsurface_coef=hybrid_args.sdf_densify_offsurface_coef,
                    sr_levels=hybrid_args.sdf_densify_sr_levels,
                    sr_samples_per_point=hybrid_args.sdf_densify_sr_samples_per_point,
                    sr_jitter_scale=hybrid_args.sdf_densify_sr_jitter_scale,
                    sr_max_points=hybrid_args.sdf_densify_sr_max_points,
                    sr_score_coef=hybrid_args.sdf_densify_sr_score_coef,
                )
            )
            print("[SDF-DENSIFY] block enabled.")

    if hybrid_args.fm_sds_enable:
        fm_sds_block = FlowMatchingSDSLikeBlock(
            FMGuidanceConfig(
                t_min=hybrid_args.fm_sds_t_min,
                t_max=hybrid_args.fm_sds_t_max,
                gamma=hybrid_args.fm_sds_gamma,
                huber_delta=hybrid_args.fm_sds_huber_delta,
            )
        )
        print("[FM-SDS] block enabled.")

    if hybrid_args.freq_loss_enable:
        frequency_block = FrequencyDecompositionBlock(
            FrequencyLossConfig(
                low_cutoff=hybrid_args.freq_low_cutoff,
                high_cutoff=hybrid_args.freq_high_cutoff,
                low_weight=hybrid_args.freq_low_weight,
                mid_weight=hybrid_args.freq_mid_weight,
                high_weight=hybrid_args.freq_high_weight,
                method=hybrid_args.freq_method,
            )
        )
        print("[FREQ] block enabled.")

    if hybrid_args.scaffold_enable:
        if not hybrid_args.scaffold_path:
            print("[SCAFFOLD] scaffold is enabled but --scaffold_path is empty; disabling block.")
        else:
            try:
                scaffold_data = load_scaffold_data(ScaffoldLoadConfig(path=hybrid_args.scaffold_path))
                scaffold_block = ScaffoldGeometryBlock(
                    ScaffoldGeometryConfig(
                        sample_size=hybrid_args.scaffold_sample_size,
                        interval=hybrid_args.scaffold_interval,
                        axis=hybrid_args.scaffold_axis,
                    ),
                    scaffold_points_cpu=scaffold_data.points,
                    scaffold_normals_cpu=scaffold_data.normals,
                )
                print(
                    "[SCAFFOLD] block enabled: "
                    f"path={scaffold_data.source_path} points={scaffold_data.num_points} "
                    f"normals={int(scaffold_data.has_normals)}"
                )
            except Exception as exc:
                print(f"[SCAFFOLD] disabled due to load failure: {exc}")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    train_start_iter = first_iter
    prior_hf_seed_total = 0
    last_seed_diag = {}
    bootstrap_source_tags = {"prior", "added"}
    initial_source_update_mask = build_source_tag_train_mask(gaussians, hybrid_args.optimize_source_tag)
    initial_trainable_count = count_trainable_source_tag_gaussians(gaussians, initial_source_update_mask)
    seed_bootstrap_required = (
        bool(hybrid_args.prior_hf_seed_enable)
        and hybrid_args.optimize_source_tag in bootstrap_source_tags
        and external_update_mask is None
        and initial_trainable_count <= 0
    )
    if seed_bootstrap_required:
        print(
            "[PRIOR-HF-SEED] bootstrap mode: restart has zero trainable gaussians for "
            f"--optimize_source_tag={hybrid_args.optimize_source_tag}; this run depends on successful seed birth."
        )
    first_cycle_pending_views = {camera.image_name for camera in active_train_cameras}
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if not hybrid_args.disable_gui:
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    (
                        custom_cam,
                        do_training,
                        pipe.convert_SHs_python,
                        pipe.compute_cov3D_python,
                        keep_alive,
                        scaling_modifer,
                    ) = network_gui.receive()
                    if custom_cam is not None:
                        net_image = train_render_fn(
                            custom_cam, gaussians, pipe, background, scaling_modifer
                        )["render"]
                        net_image_bytes = memoryview(
                            (
                                torch.clamp(net_image, min=0, max=1.0)
                                * 255
                            )
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and (
                        (iteration < int(opt.iterations)) or not keep_alive
                    ):
                        break
                except Exception:
                    network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        use_surface_route_view = (
            surface_route_bank is not None
            and len(surface_route_train_cameras) > 0
            and route_view_sample_prob > 0.0
            and random.random() < route_view_sample_prob
        )
        use_prior_view = (
            not use_surface_route_view
            and len(prior_view_train_cameras) > 0
            and prior_view_sample_prob > 0.0
            and random.random() < prior_view_sample_prob
        )
        prior_view_curriculum_state = _prior_view_curriculum_state(
            prior_view_curriculum_cameras,
            iteration,
            hybrid_args,
        )

        if prior_view_curriculum_state is not None:
            viewpoint_cam = prior_view_curriculum_state["camera"]
            use_surface_route_view = False
            use_prior_view = True
        elif use_surface_route_view:
            if not surface_route_viewpoint_stack:
                surface_route_viewpoint_stack = surface_route_train_cameras.copy()
            viewpoint_cam = surface_route_viewpoint_stack.pop(
                randint(0, len(surface_route_viewpoint_stack) - 1)
            )
        elif use_prior_view:
            if not prior_viewpoint_stack:
                prior_viewpoint_stack = prior_view_train_cameras.copy()
            viewpoint_cam = prior_viewpoint_stack.pop(
                randint(0, len(prior_viewpoint_stack) - 1)
            )
        else:
            if not viewpoint_stack:
                viewpoint_stack = active_train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (
            not use_surface_route_view
            and
            not use_prior_view
            and
            prior_view_curriculum_state is None
            and
            random.random() < 0.3
            and dataset.sample_more_highres
            and len(highresolution_index) > 0
        ):
            viewpoint_cam = train_cameras[
                highresolution_index[randint(0, len(highresolution_index) - 1)]
            ]
        prior_hf_seed_first_visit = viewpoint_cam.image_name in first_cycle_pending_views
        if prior_hf_seed_first_visit:
            first_cycle_pending_views.discard(viewpoint_cam.image_name)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        if dataset.ray_jitter:
            subpixel_offset = (
                torch.rand(
                    (
                        int(viewpoint_cam.image_height),
                        int(viewpoint_cam.image_width),
                        2,
                    ),
                    dtype=torch.float32,
                    device="cuda",
                )
                - 0.5
            )
        else:
            subpixel_offset = None

        active_sof_splat_args = vanilla_splat_args if vanilla_splat_args is not None else merged_splat_args
        if active_sof_splat_args is not None:
            active_sof_splat_args.render_opacity = bool(
                iteration >= int(hybrid_args.distortion_from_iter)
                and float(hybrid_args.lambda_opacity_field) > 0.0
            )

        render_pkg = train_render_fn(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size=dataset.kernel_size,
            subpixel_offset=subpixel_offset,
        )
        if active_sof_splat_args is not None and (
            getattr(pipe, "require_merged_sof_aux", False)
            or getattr(pipe, "use_vanilla_sof_rasterizer", False)
        ):
            missing_aux = [key for key in ("normal", "depth", "alpha", "distortion", "extent") if key not in render_pkg]
            if missing_aux:
                raise RuntimeError(
                    "SOF rasterizer was required to provide auxiliary outputs, "
                    f"but render() is missing: {', '.join(missing_aux)}"
                )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        if dataset.resample_gt_image and subpixel_offset is not None:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        prior_image = None
        prior_consistency = None
        prior_mask = None
        prior_valid_ratio = None
        prior_structure_ratio = None
        prior_hf_guidance = None
        prior_feature_pack = None
        prior_render_image = None
        prior_gt_image = None
        prior_anchor_image = None
        prior_camera = viewpoint_cam
        prior_visibility_filter = visibility_filter
        prior_viewspace_point_tensor = viewspace_point_tensor
        prior_radii = radii
        edge_touch_mask = None
        use_prior_supervision = (
            prior_bank is not None
            or prior_local_bank is not None
            or prior_edge_bank is not None
            or surface_route_bank is not None
        )
        if use_prior_supervision:
            prior_camera = viewpoint_cam
            prior_render_image = image
            prior_render_pkg = render_pkg
            if prior_supervision_camera_bank is not None:
                override_camera = prior_supervision_camera_bank.get(viewpoint_cam.image_name)
                if override_camera is not None:
                    prior_camera = override_camera
                    prior_render_pkg = train_render_fn(
                        prior_camera,
                        gaussians,
                        pipe,
                        background,
                        kernel_size=dataset.kernel_size,
                        subpixel_offset=None,
                    )
                    prior_render_image = prior_render_pkg["render"]
                    prior_viewspace_point_tensor = prior_render_pkg["viewspace_points"]
                    prior_visibility_filter = prior_render_pkg["visibility_filter"]
                    prior_radii = prior_render_pkg["radii"]
            if prior_render_image is None and prior_render_pkg is not None:
                prior_render_image = prior_render_pkg["render"]
                prior_viewspace_point_tensor = prior_render_pkg.get("viewspace_points", prior_viewspace_point_tensor)
            if prior_bank is not None:
                prior_feature_pack = _load_prior_feature_pack(
                    prior_bank=prior_bank,
                    camera=prior_camera,
                    device=prior_render_image.device,
                    hybrid_args=hybrid_args,
                    prior_mask_bank=prior_mask_bank,
                    prior_anchor_bank=prior_anchor_bank,
                    prior_soft_mask_bank=prior_soft_mask_bank,
                    prior_hard_mask_bank=prior_hard_mask_bank,
                )
                if prior_feature_pack is not None:
                    prior_image = prior_feature_pack["prior_image"]
                    prior_consistency = prior_feature_pack["consistency"]
                    prior_mask = prior_feature_pack["mask"]
                    prior_valid_ratio = prior_feature_pack["valid_ratio"]
                    prior_structure_ratio = prior_feature_pack["structure_mask"].mean()
                    prior_hf_guidance = prior_feature_pack["guidance"]
                    prior_gt_image = prior_feature_pack["gt_image"]
                    prior_anchor_image = prior_feature_pack.get("anchor_image")

        seed_gate_diag = {
            "hf_seed_gate_enable": 1.0 if bool(hybrid_args.prior_hf_seed_enable) else 0.0,
            "hf_seed_gate_first_visit": 1.0 if prior_hf_seed_first_visit else 0.0,
            "hf_seed_gate_prior_pack": 1.0 if prior_feature_pack is not None else 0.0,
            "hf_seed_gate_guidance": (
                1.0
                if prior_feature_pack is not None and prior_feature_pack.get("guidance") is not None
                else 0.0
            ),
            "hf_seed_gate_prior_bank": 1.0 if prior_bank is not None else 0.0,
            "hf_seed_gate_anchor_bank": 1.0 if prior_anchor_bank is not None else 0.0,
        }
        if prior_camera is not None:
            seed_gate_diag["hf_seed_gate_prior_index"] = (
                1.0 if prior_bank is not None and prior_camera.image_name in prior_bank.index else 0.0
            )
            seed_gate_diag["hf_seed_gate_anchor_index"] = (
                1.0
                if prior_anchor_bank is not None and prior_camera.image_name in prior_anchor_bank.index
                else 0.0
            )
        if seed_bootstrap_required and iteration >= int(hybrid_args.prior_hf_seed_from_iter):
            last_seed_diag.update(seed_gate_diag)

        recent_prior_protect_iters = int(getattr(hybrid_args, "prior_hf_seed_prune_protect_iters", 0))
        recent_prior_seed_mask = _build_recent_prior_seed_mask(
            gaussians,
            iteration=iteration,
            recent_iters=recent_prior_protect_iters,
        )
        seed_first_cycle_only_active = bool(
            getattr(hybrid_args, "prior_hf_seed_first_cycle_only", False)
        )
        hold_prior_densify_for_seed_bootstrap = (
            seed_bootstrap_required
            and seed_first_cycle_only_active
            and iteration >= int(hybrid_args.prior_hf_seed_from_iter)
            and (prior_hf_seed_first_visit or len(first_cycle_pending_views) > 0)
        )
        recent_prior_visible_mask = None
        recent_prior_live_count = 0.0
        recent_prior_visible_count = 0.0
        if recent_prior_seed_mask is not None:
            recent_prior_live_count = float(recent_prior_seed_mask.sum().item())
            if (
                prior_visibility_filter is not None
                and int(prior_visibility_filter.shape[0]) == int(recent_prior_seed_mask.shape[0])
            ):
                recent_prior_visible_mask = recent_prior_seed_mask & prior_visibility_filter
                recent_prior_visible_count = float(recent_prior_visible_mask.sum().item())

        hybrid_metrics = {}
        sdf_densify_metrics = {}
        proposal_metrics = {}
        prior_metrics = {}
        sof_reg_metrics = {}
        fm_metrics = {}
        freq_metrics = {}
        layer_freq_metrics = {}
        layer_frequency_surface_loss = None
        surface_migration_metrics = {}

        Ll1 = l1_loss(image, gt_image)
        loss_tex = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if bool(getattr(hybrid_args, "sequence_loss_enable", False)):
            scale = max(1e-6, float(getattr(hybrid_args, "sequence_subpixel_scale", 4.0)))
            lr_height = max(1, int(round(float(image.shape[-2]) / scale)))
            lr_width = max(1, int(round(float(image.shape[-1]) / scale)))
            lr_anchor = None
            if sequence_lr_anchor_bank is not None:
                lr_anchor = sequence_lr_anchor_bank.get(
                    viewpoint_cam.image_name,
                    width=lr_width,
                    height=lr_height,
                    device=image.device,
                )
            if lr_anchor is None:
                if bool(getattr(hybrid_args, "sequence_loss_allow_missing_anchor", False)):
                    loss = loss_tex
                    sequence_sp_loss = None
                else:
                    raise RuntimeError(
                        "SequenceMatters-style loss is missing LR anchor for "
                        f"view={viewpoint_cam.image_name}"
                    )
            else:
                subpixel_mode = str(getattr(hybrid_args, "sequence_subpixel", "bicubic")).strip().lower()
                if subpixel_mode == "avg":
                    kernel = max(1, int(round(scale)))
                    image_lr = torch_F.avg_pool2d(image[None], kernel_size=kernel, stride=kernel)[0]
                    if image_lr.shape[-2:] != lr_anchor.shape[-2:]:
                        image_lr = torch_F.interpolate(
                            image_lr[None],
                            size=lr_anchor.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                elif subpixel_mode == "bicubic":
                    image_lr = torch_F.interpolate(
                        image[None],
                        size=lr_anchor.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )[0]
                else:
                    raise ValueError(
                        f"Unsupported --sequence_subpixel={hybrid_args.sequence_subpixel}; "
                        "use bicubic or avg"
                    )
                Ll1_sp = l1_loss(image_lr, lr_anchor)
                sequence_sp_loss = (
                    (1.0 - opt.lambda_dssim) * Ll1_sp
                    + opt.lambda_dssim * (1.0 - ssim(image_lr, lr_anchor))
                )
                lambda_tex = max(0.0, min(1.0, float(getattr(hybrid_args, "sequence_lambda_tex", 0.4))))
                loss = lambda_tex * loss_tex + (1.0 - lambda_tex) * sequence_sp_loss
                prior_metrics["sequence_tex"] = float(loss_tex.detach().item())
                prior_metrics["sequence_sp"] = float(sequence_sp_loss.detach().item())
                prior_metrics["sequence_lambda_tex"] = float(lambda_tex)
        else:
            loss = loss_tex

        current_prior_loss_update_mask = (
            _expand_root_aligned_mask(
                prior_loss_update_mask,
                gaussians,
                label="prior-loss gaussian mask",
            )
            if prior_loss_update_mask_dynamic_roots
            else prior_loss_update_mask
        )
        prior_guidance_scale = _linear_decay_scale(
            iteration,
            getattr(hybrid_args, "prior_guidance_decay_from_iter", 0),
            getattr(hybrid_args, "prior_guidance_decay_until_iter", 0),
            getattr(hybrid_args, "prior_guidance_final_scale", 1.0),
        )
        prior_direct_xyz_loss = None
        prior_direct_xyz_grad = None
        prior_direct_xyz_nudge_metrics = {}
        scaffold_metrics = {}
        proposal_realign = None
        proposal_native_birth = None
        freq_total_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        if sdf_adapter is not None:
            refresh_flag = 0
            if (
                hybrid_args.sdf_mode == "gs_bootstrap"
                and hybrid_args.gs_bootstrap_adaptive_refresh_enable
            ):
                refresh_flag = _maybe_refresh_gs_bootstrap_adapter(
                    sdf_adapter=sdf_adapter,
                    iteration=iteration,
                    scheduler=loss_scheduler,
                    args=hybrid_args,
                )
            elif hasattr(sdf_adapter, "step"):
                sdf_adapter.step(iteration)
                refresh_flag = 0
            with torch.no_grad():
                num_points = gaussians.get_xyz.shape[0]
                sample_size = min(hybrid_args.hybrid_points_per_iter, num_points)
                sample_idx = torch.randint(0, num_points, (sample_size,), device="cuda")

            xyz_sample = gaussians.get_xyz[sample_idx]
            sdf_values, sdf_grads = sdf_adapter.query_sdf_and_gradients(xyz_sample.detach())
            sdf_values = sdf_values.to(xyz_sample.device, dtype=xyz_sample.dtype).view(-1, 1)
            sdf_grads = sdf_grads.to(xyz_sample.device, dtype=xyz_sample.dtype).view(-1, 3)

            linearized_sdf = linearized_signed_distance(xyz_sample, sdf_values, sdf_grads)
            loss_surface = surface_distance_loss(
                linearized_sdf,
                epsilon=hybrid_args.surface_epsilon,
            )

            scales_sample = gaussians.get_scaling[sample_idx]
            rotations_sample = gaussians._rotation[sample_idx]
            loss_normal = normal_alignment_loss(
                rotations_sample,
                scales_sample,
                sdf_grads,
                axis=hybrid_args.normal_axis,
            )

            opacities_sample = gaussians.get_opacity[sample_idx]
            loss_offsurface = offsurface_opacity_loss(
                opacities_sample,
                linearized_sdf,
                margin=hybrid_args.offsurface_margin,
            )

            schedule_scale = loss_scheduler.scale(iteration)
            hybrid_loss = schedule_scale * (
                hybrid_args.lambda_surface * loss_surface
                + hybrid_args.lambda_normal * loss_normal
                + hybrid_args.lambda_offsurface * loss_offsurface
            )
            loss = loss + hybrid_loss

            hybrid_metrics = {
                "stage": loss_scheduler.stage(iteration),
                "scale": schedule_scale,
                "surface": loss_surface.detach().item(),
                "normal": loss_normal.detach().item(),
                "offsurface": loss_offsurface.detach().item(),
                "total": hybrid_loss.detach().item(),
            }

            proposal_extra = None
            proposal_realign = None
            proposal_native_birth = None
            if hybrid_args.sdf_proposal_enable and prior_hf_guidance is not None:
                px, pv = _extract_2d_sr_proposals(
                    guidance_map=prior_hf_guidance,
                    topk=hybrid_args.sdf_proposal_topk,
                    min_value=hybrid_args.sdf_proposal_min_hf,
                )
                n_2d = 0 if px is None else int(px.shape[0])
                n_hits = 0
                n_valid = 0
                mean_abs_sdf = 0.0
                mean_support = 0.0
                mean_align = 0.0
                mean_sigma = 0.0
                mean_align_sigma = 0.0
                support_thresh_eff = 0.0
                sdf_pass_rate = 0.0
                support_pass_rate = 0.0
                align_pass_rate = 0.0
                valid_rate = 0.0
                center_abs_sdf_mean = 0.0
                center_abs_sdf_p90 = 0.0
                center_abs_sdf_max = 0.0
                proposal_trust_scale = 1.0
                proposal_trust_state = "full"
                if px is not None and px.shape[0] > 0:
                    px = px.to(device=image.device, dtype=image.dtype)
                    ro, rd = _build_world_rays_from_pixels(viewpoint_cam, px)
                    hits = _raycast_sdf_surface(
                        sdf_adapter=sdf_adapter,
                        ray_origins=ro,
                        ray_dirs=rd,
                        t_near=hybrid_args.sdf_proposal_ray_near,
                        t_far=hybrid_args.sdf_proposal_ray_far,
                        n_samples=hybrid_args.sdf_proposal_ray_samples,
                    )
                    if hits is not None:
                        n_hits = int(hits["xyz"].shape[0])
                        gsq = _query_gs_support_stable(
                            gaussians=gaussians,
                            query_xyz=hits["xyz"],
                            knn_k=hybrid_args.sdf_proposal_gs_knn_k,
                            sigma_mode=hybrid_args.gs_bootstrap_sigma_mode,
                            distance_scale=hybrid_args.gs_bootstrap_distance_scale,
                            distance_floor=hybrid_args.gs_bootstrap_distance_floor,
                        )
                        if gsq is not None:
                            cond_sdf = hits["sdf"].abs() <= hybrid_args.sdf_proposal_surface_thresh
                            support_thresh_eff = _adaptive_support_threshold(gsq["support"], hybrid_args)
                            cond_support = gsq["support"] >= support_thresh_eff
                            cond_align = gsq["nearest_dist"] <= hybrid_args.sdf_proposal_align_thresh
                            align_sigma = gsq["nearest_dist"] / gsq["nearest_sigma"].clamp(min=1e-6)
                            cond_align_sigma = align_sigma <= hybrid_args.sdf_proposal_align_sigma_thresh
                            valid = cond_sdf & cond_support & cond_align & cond_align_sigma
                            n_valid = int(valid.sum().item())
                            mean_abs_sdf = float(hits["sdf"].abs().mean().detach().item())
                            mean_support = float(gsq["support"].mean().detach().item())
                            mean_align = float(gsq["nearest_dist"].mean().detach().item())
                            mean_sigma = float(gsq["nearest_sigma"].mean().detach().item())
                            mean_align_sigma = float(
                                align_sigma.mean().detach().item()
                            )
                            sdf_pass_rate = float(cond_sdf.float().mean().detach().item())
                            support_pass_rate = float(cond_support.float().mean().detach().item())
                            align_pass_rate = float((cond_align & cond_align_sigma).float().mean().detach().item())
                            valid_rate = float(valid.float().mean().detach().item())
                            center_diag = _diagnose_pseudo_sdf_vs_gs(
                                sdf_adapter=sdf_adapter,
                                gaussians=gaussians,
                                max_points=hybrid_args.sdf_proposal_diag_max_points,
                            )
                            center_abs_sdf_mean = center_diag["center_abs_sdf_mean"]
                            center_abs_sdf_p90 = center_diag["center_abs_sdf_p90"]
                            center_abs_sdf_max = center_diag["center_abs_sdf_max"]
                            proposal_trust_scale, proposal_trust_state = _proposal_trust_from_center_p90(
                                center_abs_sdf_p90=center_abs_sdf_p90,
                                args=hybrid_args,
                            )

                            if n_valid > 0 and proposal_trust_scale > 0.0:
                                a_xyz = hits["xyz"][valid]
                                a_nrm = hits["grad"][valid]
                                a_px = px[valid]
                                nearest_idx = gsq["nearest_idx"][valid]
                                if proposal_trust_scale < 1.0:
                                    keep = max(1, int(round(a_xyz.shape[0] * proposal_trust_scale)))
                                    perm = torch.randperm(a_xyz.shape[0], device=a_xyz.device)[:keep]
                                    a_xyz = a_xyz[perm]
                                    a_nrm = a_nrm[perm]
                                    a_px = a_px[perm]
                                    nearest_idx = nearest_idx[perm]
                                    n_valid = int(a_xyz.shape[0])
                                    valid_rate = float(n_valid / max(1, n_hits))
                                base_scale = gaussians.get_scaling[nearest_idx]
                                fused_births, fused_metrics = _fuse_surface_proposals_multiview(
                                    sdf_adapter=sdf_adapter,
                                    prior_bank=prior_bank,
                                    prior_mask_bank=prior_mask_bank,
                                    ref_camera=viewpoint_cam,
                                    camera_pool=train_cameras,
                                    ref_pixels=a_px,
                                    ref_feature_pack=prior_feature_pack,
                                    anchors_xyz=a_xyz,
                                    anchors_normals=a_nrm,
                                    base_scale=base_scale,
                                    hybrid_args=hybrid_args,
                                )
                                proposal_metrics["neighbor_views"] = fused_metrics["neighbor_views"]
                                proposal_metrics["fused_points"] = fused_metrics["fused_points"]
                                proposal_metrics["inlier_views_mean"] = fused_metrics["inlier_views_mean"]
                                proposal_metrics["delta_abs_mean"] = fused_metrics["delta_abs_mean"]
                                proposal_metrics["tangent_abs_mean"] = fused_metrics["tangent_abs_mean"]
                                if fused_births is not None and fused_births["xyz"].shape[0] > 0:
                                    keep_births = fused_births["keep"]
                                    kept_idx = nearest_idx[keep_births]
                                    proposal_realign = _build_view_anchored_realign_bundle(
                                        matched_idx=kept_idx,
                                        ref_pixels=a_px[keep_births],
                                        target_xyz=fused_births["xyz"],
                                        target_normals=fused_births["normals"],
                                        weights=fused_births["inlier_views"],
                                    )
                                    proposal_native_birth = _build_native_birth_bundle(
                                        matched_idx=kept_idx,
                                        target_xyz=fused_births["xyz"],
                                        target_normals=fused_births["normals"],
                                        ref_pixels=a_px[keep_births],
                                        weights=fused_births["inlier_views"],
                                    )
                                    proposal_extra = _build_surface_birth_bundle(
                                        attached_xyz=fused_births["xyz"],
                                        attached_normals=fused_births["normals"],
                                        base_scale=gaussians.get_scaling[kept_idx],
                                        rotations_raw=gaussians._rotation[kept_idx],
                                        opacities=gaussians.get_opacity[kept_idx],
                                        samples_per_anchor=hybrid_args.sdf_proposal_plane_samples,
                                        tangent_scale=hybrid_args.sdf_proposal_tangent_scale,
                                        normal_scale=hybrid_args.sdf_proposal_normal_scale,
                                    )
                            elif proposal_trust_scale <= 0.0:
                                n_valid = 0
                                valid_rate = 0.0
                proposal_metrics = {
                    "proposals_2d": float(n_2d),
                    "hits_3d": float(n_hits),
                    "valid_3d": float(n_valid),
                    "abs_sdf_mean": mean_abs_sdf,
                    "support_mean": mean_support,
                    "align_mean": mean_align,
                    "nearest_sigma_mean": mean_sigma,
                    "align_sigma_mean": mean_align_sigma,
                    "support_thresh_eff": support_thresh_eff,
                    "sdf_pass_rate": sdf_pass_rate,
                    "support_pass_rate": support_pass_rate,
                    "align_pass_rate": align_pass_rate,
                    "valid_rate": valid_rate,
                    "center_abs_sdf_mean": center_abs_sdf_mean,
                    "center_abs_sdf_p90": center_abs_sdf_p90,
                    "center_abs_sdf_max": center_abs_sdf_max,
                    "proposal_trust_scale": proposal_trust_scale,
                    "proposal_trust_state": proposal_trust_state,
                    "teacher_refresh": float(refresh_flag),
                    "sr_value_mean": 0.0 if pv is None else float(pv.mean().detach().item()),
                    "neighbor_views": proposal_metrics.get("neighbor_views", 0.0),
                    "fused_points": proposal_metrics.get("fused_points", 0.0),
                    "inlier_views_mean": proposal_metrics.get("inlier_views_mean", 0.0),
                    "delta_abs_mean": proposal_metrics.get("delta_abs_mean", 0.0),
                    "tangent_abs_mean": proposal_metrics.get("tangent_abs_mean", 0.0),
                    "realigned_gs": 0.0,
                    "realign_shift_mean": 0.0,
                    "realign_depth_delta_mean": 0.0,
                    "realign_rot_mean": 0.0,
                    "realign_scale_ratio_mean": 1.0,
                    "native_births": 0.0,
                }

            densify_extra_for_sr = proposal_extra
            if sdf_densify_block is not None:
                sr_point_weight = None
                sr_valid_ratio = 0.0
                if prior_hf_guidance is not None:
                    sr_point_weight, sr_valid_ratio = _sample_image_guidance_on_points(
                        guidance_map=prior_hf_guidance,
                        points=xyz_sample.detach(),
                        camera=viewpoint_cam,
                    )
                densify_proxy_loss, sdf_densify_metrics, densify_extra = sdf_densify_block.compute(
                    xyz_sample=xyz_sample,
                    linearized_sdf=linearized_sdf,
                    rotations_raw=rotations_sample,
                    scales=scales_sample,
                    sdf_grads=sdf_grads,
                    opacities=opacities_sample,
                    margin=hybrid_args.offsurface_margin,
                    iteration=iteration,
                    sr_point_weight=sr_point_weight,
                )
                weighted_densify = hybrid_args.sdf_densify_loss_weight * densify_proxy_loss
                if hybrid_args.sdf_densify_loss_weight > 0:
                    loss = loss + weighted_densify
                sdf_densify_metrics["proxy"] = densify_proxy_loss.detach().item()
                sdf_densify_metrics["weighted"] = weighted_densify.detach().item()
                sdf_densify_metrics["sr_valid_ratio"] = sr_valid_ratio
                densify_extra_for_sr = _concat_densify_bundles(densify_extra_for_sr, densify_extra)

            if (
                densify_extra_for_sr is not None
                and densify_extra_for_sr.get("xyz") is not None
                and hybrid_args.sdf_densify_sr_weight > 0
            ):
                    xyz_sr = densify_extra_for_sr["xyz"]
                    sdf_values_sr, sdf_grads_sr = sdf_adapter.query_sdf_and_gradients(
                        xyz_sr.detach()
                    )
                    sdf_values_sr = sdf_values_sr.to(xyz_sr.device, dtype=xyz_sr.dtype).view(-1, 1)
                    sdf_grads_sr = sdf_grads_sr.to(xyz_sr.device, dtype=xyz_sr.dtype).view(-1, 3)

                    linearized_sr = linearized_signed_distance(xyz_sr, sdf_values_sr, sdf_grads_sr)
                    loss_surface_sr = surface_distance_loss(
                        linearized_sr,
                        epsilon=hybrid_args.surface_epsilon,
                    )
                    loss_normal_sr = normal_alignment_loss(
                        densify_extra_for_sr["rotations_raw"],
                        densify_extra_for_sr["scales"],
                        sdf_grads_sr,
                        axis=hybrid_args.normal_axis,
                    )
                    loss_offsurface_sr = offsurface_opacity_loss(
                        densify_extra_for_sr["opacities"],
                        linearized_sr,
                        margin=hybrid_args.offsurface_margin,
                    )
                    sr_base = (
                        hybrid_args.lambda_surface * loss_surface_sr
                        + hybrid_args.lambda_normal * loss_normal_sr
                        + hybrid_args.lambda_offsurface * loss_offsurface_sr
                    )
                    sr_weighted = hybrid_args.sdf_densify_sr_weight * sr_base
                    loss = loss + sr_weighted
                    sdf_densify_metrics["sr_surface"] = loss_surface_sr.detach().item()
                    sdf_densify_metrics["sr_normal"] = loss_normal_sr.detach().item()
                    sdf_densify_metrics["sr_offsurface"] = loss_offsurface_sr.detach().item()
                    sdf_densify_metrics["sr_base"] = sr_base.detach().item()
                    sdf_densify_metrics["sr_weighted"] = sr_weighted.detach().item()
                    sdf_densify_metrics["sr_points_total"] = float(xyz_sr.shape[0])
            elif sdf_densify_metrics:
                sdf_densify_metrics["sr_weighted"] = 0.0

        if scaffold_block is not None and (
            hybrid_args.scaffold_chamfer_weight > 0 or hybrid_args.scaffold_normal_weight > 0
        ):
            scaffold_chamfer, scaffold_normal, scaffold_info = scaffold_block.compute(
                xyz_all=gaussians.get_xyz,
                rotations_raw_all=gaussians._rotation,
                scales_all=gaussians.get_scaling,
                iteration=iteration,
            )
            scaffold_loss = (
                hybrid_args.scaffold_chamfer_weight * scaffold_chamfer
                + hybrid_args.scaffold_normal_weight * scaffold_normal
            )
            loss = loss + scaffold_loss
            scaffold_metrics = {
                "total": scaffold_loss.detach().item(),
                "chamfer": scaffold_chamfer.detach().item(),
                "normal": scaffold_normal.detach().item(),
                "selected_gs": scaffold_info["selected_gs"],
                "selected_scaffold": scaffold_info["selected_scaffold"],
                "has_normals": scaffold_info.get("has_normals", 0.0),
            }

        masked_prior_loss = None

        if prior_image is not None and (
            hybrid_args.prior_l1_weight > 0
            or hybrid_args.prior_hf_weight > 0
            or (
                frequency_block is not None
                and hybrid_args.freq_prior_weight > 0
                and hybrid_args.freq_loss_weight > 0
            )
        ):
                # Consistency gate: ignore regions where external prior diverges too far from LR observation.
                consistency = prior_consistency
                mask = prior_mask
                valid_ratio = prior_valid_ratio if prior_valid_ratio is not None else mask.mean()
                render_for_prior = image if prior_render_image is None else prior_render_image
                gt_for_prior = gt_image if prior_gt_image is None else prior_gt_image

                if valid_ratio >= hybrid_args.prior_min_valid_ratio:
                    denom = torch.clamp(mask.sum() * render_for_prior.shape[0], min=1.0)
                    hf_focus_weight = _build_prior_hf_focus_weight(
                        prior_feature_pack,
                        mask,
                        hybrid_args,
                    )
                    hf_focus_denom = torch.clamp(
                        hf_focus_weight.sum() * render_for_prior.shape[0],
                        min=1.0,
                    )
                    if hybrid_args.prior_loss_mode == "masked_residual_hf_v1":
                        if prior_anchor_image is None:
                            raise RuntimeError("masked_residual_hf_v1 requires a prior anchor image for every active prior sample.")
                        anchor_for_prior = prior_anchor_image.to(device=render_for_prior.device, dtype=render_for_prior.dtype)
                        prior_delta = prior_image - anchor_for_prior
                        if hybrid_args.prior_delta_clip > 0:
                            clip_v = hybrid_args.prior_delta_clip
                            prior_delta = prior_delta.clamp(min=-clip_v, max=clip_v)
                        render_delta = render_for_prior - anchor_for_prior
                        primary_render_delta = render_delta
                        primary_prior_delta = prior_delta
                        loss_prior_l1 = ((render_delta - prior_delta).abs() * mask).sum() / denom
                        render_residual_hf = _laplacian_highfreq(render_delta)
                        prior_residual_hf = _laplacian_highfreq(prior_delta)
                        loss_prior_hf = (
                            (render_residual_hf - prior_residual_hf).abs() * hf_focus_weight
                        ).sum() / hf_focus_denom
                    else:
                        loss_prior_l1 = ((render_for_prior - prior_image).abs() * mask).sum() / denom

                    if hybrid_args.prior_loss_mode != "masked_residual_hf_v1" and hybrid_args.disable_prior_hf_residual:
                        image_hf = _laplacian_highfreq(render_for_prior)
                        prior_hf = _laplacian_highfreq(prior_image)
                        primary_render_delta = render_for_prior
                        primary_prior_delta = prior_image
                        render_residual_hf = image_hf
                        prior_residual_hf = prior_hf
                        loss_prior_hf = ((image_hf - prior_hf).abs() * hf_focus_weight).sum() / hf_focus_denom
                    elif hybrid_args.prior_loss_mode != "masked_residual_hf_v1":
                        prior_delta = prior_image - gt_for_prior
                        if hybrid_args.prior_delta_clip > 0:
                            clip_v = hybrid_args.prior_delta_clip
                            prior_delta = prior_delta.clamp(min=-clip_v, max=clip_v)
                        render_delta = render_for_prior - gt_for_prior
                        primary_render_delta = render_delta
                        primary_prior_delta = prior_delta
                        render_residual_hf = _laplacian_highfreq(render_delta)
                        prior_residual_hf = _laplacian_highfreq(prior_delta)
                        loss_prior_hf = (
                            (render_residual_hf - prior_residual_hf).abs() * hf_focus_weight
                        ).sum() / hf_focus_denom

                    primary_hf_energy_loss = None
                    primary_hf_band_loss = None
                    primary_residual_loss = None
                    primary_patch_loss = None
                    primary_patch_lowfreq_loss = None
                    primary_patch_coverage = None
                    recent_seed_imprint_loss = None
                    recent_seed_imprint_count = 0.0
                    recent_seed_imprint_visible = 0.0
                    if (
                        prior_view_curriculum_state is not None
                        and str(prior_view_curriculum_state.get("phase", "")) == "primary"
                    ):
                        primary_bootstrap_iters = int(
                            getattr(hybrid_args, "prior_view_curriculum_primary_bootstrap_iters", 0)
                        )
                        primary_local_iter = int(prior_view_curriculum_state.get("local_iter", 0))
                        primary_energy_weight = float(
                            getattr(hybrid_args, "prior_view_curriculum_primary_hf_energy_weight", 0.0)
                        )
                        primary_band_weight = float(
                            getattr(hybrid_args, "prior_view_curriculum_primary_band_weight", 0.0)
                        )
                        primary_residual_weight = float(
                            getattr(hybrid_args, "prior_view_curriculum_primary_residual_weight", 0.0)
                        )
                        primary_patch_weight = float(
                            getattr(hybrid_args, "prior_view_curriculum_primary_patch_weight", 0.0)
                        )
                        if (
                            (
                                primary_energy_weight > 0.0
                                or primary_band_weight > 0.0
                                or primary_residual_weight > 0.0
                                or primary_patch_weight > 0.0
                            )
                            and (primary_bootstrap_iters <= 0 or primary_local_iter < primary_bootstrap_iters)
                        ):
                            if primary_energy_weight > 0.0:
                                energy_gain = max(
                                    0.0,
                                    float(getattr(hybrid_args, "prior_view_curriculum_primary_hf_energy_gain", 1.0)),
                                )
                                target_energy = prior_residual_hf.abs() * energy_gain
                                render_energy = render_residual_hf.abs()
                                primary_hf_energy_loss = (
                                    torch.relu(target_energy - render_energy) * hf_focus_weight
                                ).sum() / hf_focus_denom
                                loss_prior_hf = loss_prior_hf + primary_energy_weight * primary_hf_energy_loss
                            if primary_band_weight > 0.0:
                                band_kernel = int(
                                    getattr(hybrid_args, "prior_view_curriculum_primary_band_kernel", 7)
                                )
                                render_band = _box_highpass(primary_render_delta, band_kernel)
                                prior_band = _box_highpass(primary_prior_delta, band_kernel)
                                primary_hf_band_loss = (
                                    (render_band - prior_band).abs() * hf_focus_weight
                                ).sum() / hf_focus_denom
                                loss_prior_hf = loss_prior_hf + primary_band_weight * primary_hf_band_loss
                            if primary_residual_weight > 0.0:
                                primary_residual_loss = (
                                    (primary_render_delta - primary_prior_delta).abs() * hf_focus_weight
                                ).sum() / hf_focus_denom
                                loss_prior_hf = loss_prior_hf + primary_residual_weight * primary_residual_loss
                            if primary_patch_weight > 0.0:
                                patch_weight_map, primary_patch_coverage = _build_prior_hf_patch_weight(
                                    prior_feature_pack=prior_feature_pack,
                                    base_mask=mask,
                                    top_fraction=float(
                                        getattr(
                                            hybrid_args,
                                            "prior_view_curriculum_primary_patch_top_fraction",
                                            0.15,
                                        )
                                    ),
                                    min_guidance=float(
                                        getattr(
                                            hybrid_args,
                                            "prior_view_curriculum_primary_patch_min_guidance",
                                            0.08,
                                        )
                                    ),
                                )
                                if patch_weight_map is not None:
                                    patch_delta_gain = max(
                                        0.0,
                                        float(
                                            getattr(
                                                hybrid_args,
                                                "prior_view_curriculum_primary_patch_delta_gain",
                                                1.0,
                                            )
                                        ),
                                    )
                                    patch_target_delta = primary_prior_delta * patch_delta_gain
                                    patch_highpass_kernel = int(
                                        getattr(
                                            hybrid_args,
                                            "prior_view_curriculum_primary_patch_highpass_kernel",
                                            0,
                                        )
                                    )
                                    patch_render_value = primary_render_delta
                                    patch_target_value = patch_target_delta
                                    if patch_highpass_kernel > 1:
                                        patch_render_value = _box_highpass(
                                            primary_render_delta,
                                            patch_highpass_kernel,
                                        )
                                        patch_target_value = (
                                            _box_highpass(primary_prior_delta, patch_highpass_kernel)
                                            * patch_delta_gain
                                        )
                                    patch_denom = torch.clamp(
                                        patch_weight_map.sum() * patch_render_value.shape[0],
                                        min=1.0,
                                    )
                                    primary_patch_loss = (
                                        (patch_render_value - patch_target_value).abs()
                                        * patch_weight_map
                                    ).sum() / patch_denom
                                    loss_prior_hf = loss_prior_hf + primary_patch_weight * primary_patch_loss
                                    patch_lowfreq_guard_weight = float(
                                        getattr(
                                            hybrid_args,
                                            "prior_view_curriculum_primary_patch_lowfreq_guard_weight",
                                            0.0,
                                        )
                                    )
                                    if patch_lowfreq_guard_weight > 0.0:
                                        lowfreq_kernel = int(
                                            getattr(
                                                hybrid_args,
                                                "prior_view_curriculum_primary_patch_lowfreq_guard_kernel",
                                                15,
                                            )
                                        )
                                        render_lowfreq_delta = primary_render_delta - _box_highpass(
                                            primary_render_delta,
                                            lowfreq_kernel,
                                        )
                                        primary_patch_lowfreq_loss = (
                                            render_lowfreq_delta.abs() * patch_weight_map
                                        ).sum() / torch.clamp(
                                            patch_weight_map.sum() * render_lowfreq_delta.shape[0],
                                            min=1.0,
                                        )
                                        loss_prior_hf = (
                                            loss_prior_hf
                                            + patch_lowfreq_guard_weight * primary_patch_lowfreq_loss
                                        )

                    recent_prior_hf_boost = 0.0
                    recent_prior_hf_boost_value = float(
                        getattr(hybrid_args, "prior_hf_seed_recent_hf_boost", 0.0)
                    )
                    if recent_prior_hf_boost_value > 0.0 and recent_prior_visible_count > 0.0:
                        recent_prior_hf_boost = 1.0 + recent_prior_hf_boost_value
                        loss_prior_hf = loss_prior_hf * recent_prior_hf_boost

                    recent_seed_imprint_weight = float(
                        getattr(hybrid_args, "prior_hf_seed_recent_imprint_weight", 0.0)
                    )
                    if recent_seed_imprint_weight > 0.0:
                        imprint_iters = int(
                            getattr(hybrid_args, "prior_hf_seed_recent_imprint_iters", 0)
                        )
                        if imprint_iters <= 0:
                            imprint_iters = int(recent_prior_protect_iters)
                        imprint_mask = _build_recent_prior_seed_mask(
                            gaussians,
                            iteration=iteration,
                            recent_iters=imprint_iters,
                        )
                        if (
                            imprint_mask is not None
                            and prior_visibility_filter is not None
                            and int(prior_visibility_filter.shape[0]) == int(imprint_mask.shape[0])
                        ):
                            recent_seed_imprint_count = float(imprint_mask.sum().item())
                            imprint_visible_mask = imprint_mask & prior_visibility_filter
                            recent_seed_imprint_visible = float(imprint_visible_mask.sum().item())
                            if recent_seed_imprint_visible > 0.0:
                                recent_seed_gaussians = GaussianSubsetView(gaussians, imprint_visible_mask)
                                recent_seed_pkg = train_render_fn(
                                    camera_for_sof_prior,
                                    recent_seed_gaussians,
                                    pipe,
                                    torch.zeros_like(background),
                                    kernel_size=dataset.kernel_size,
                                    subpixel_offset=(subpixel_offset if camera_for_sof_prior is viewpoint_cam else None),
                                )
                                recent_seed_render = recent_seed_pkg["render"]
                                recent_seed_hf_energy = _laplacian_highfreq(recent_seed_render).abs()
                                imprint_gain = max(
                                    0.0,
                                    float(getattr(hybrid_args, "prior_hf_seed_recent_imprint_gain", 1.0)),
                                )
                                target_hf_energy = prior_residual_hf.detach().abs() * imprint_gain
                                recent_seed_imprint_loss = (
                                    torch.relu(target_hf_energy - recent_seed_hf_energy) * hf_focus_weight
                                ).sum() / hf_focus_denom
                                loss_prior_hf = (
                                    loss_prior_hf
                                    + recent_seed_imprint_weight * recent_seed_imprint_loss
                                )

                    prior_hf_weight_eff = float(hybrid_args.prior_hf_weight)
                    if prior_view_curriculum_state is not None:
                        prior_hf_weight_eff *= max(0.0, float(prior_view_curriculum_state.get("hf_scale", 1.0)))
                    prior_hf_weight_eff *= float(prior_guidance_scale)

                    prior_loss = (
                        hybrid_args.prior_l1_weight * loss_prior_l1
                        + prior_hf_weight_eff * loss_prior_hf
                    )
                    prior_residual_masked_update = (
                        bool(getattr(hybrid_args, "prior_residual_masked_update", False))
                        and current_prior_loss_update_mask is not None
                    )
                    if prior_residual_masked_update:
                        masked_prior_loss = (
                            prior_loss
                            if masked_prior_loss is None
                            else masked_prior_loss + prior_loss
                        )
                    else:
                        loss = loss + prior_loss
                    if (
                        hybrid_args.prior_loss_mode == "masked_residual_hf_v1"
                        and float(hybrid_args.prior_direct_xyz_nudge_lr) > 0.0
                    ):
                        prior_direct_xyz_loss = prior_loss
                    prior_metrics = {
                        "total": prior_loss.detach().item(),
                        "l1": loss_prior_l1.detach().item(),
                        "hf": loss_prior_hf.detach().item(),
                        "valid_ratio": valid_ratio.detach().item(),
                        "mask_mean": mask.mean().detach().item(),
                        "hf_focus_mean": hf_focus_weight.mean().detach().item(),
                        "structure_mask_mean": 1.0 if prior_structure_ratio is None else prior_structure_ratio.detach().item(),
                        "hf_recent_prior_live": recent_prior_live_count,
                        "hf_recent_prior_visible": recent_prior_visible_count,
                        "hf_recent_prior_boost": recent_prior_hf_boost,
                        "hf_recent_imprint_count": recent_seed_imprint_count,
                        "hf_recent_imprint_visible": recent_seed_imprint_visible,
                        "hf_weight_eff": prior_hf_weight_eff,
                        "guidance_scale": float(prior_guidance_scale),
                        "masked_update": 1.0 if prior_residual_masked_update else 0.0,
                    }
                    if recent_seed_imprint_loss is not None:
                        prior_metrics["hf_recent_imprint"] = recent_seed_imprint_loss.detach().item()
                    if prior_view_curriculum_state is not None:
                        phase = str(prior_view_curriculum_state.get("phase", ""))
                        phase_id = 1.0 if phase == "primary" else (2.0 if phase == "neighbor" else 3.0)
                        prior_metrics["view_curriculum_phase"] = phase_id
                        prior_metrics["view_curriculum_primary_index"] = float(
                            prior_view_curriculum_state.get("primary_index", -1)
                        )
                        prior_metrics["view_curriculum_view_done"] = float(
                            prior_view_curriculum_state.get("view_done", 0)
                        )
                        prior_metrics["view_curriculum_view_total"] = float(
                            prior_view_curriculum_state.get("view_total", 0)
                        )
                        if primary_hf_energy_loss is not None:
                            prior_metrics["view_curriculum_primary_hf_energy"] = (
                                primary_hf_energy_loss.detach().item()
                            )
                        if primary_hf_band_loss is not None:
                            prior_metrics["view_curriculum_primary_band"] = primary_hf_band_loss.detach().item()
                        if primary_residual_loss is not None:
                            prior_metrics["view_curriculum_primary_residual"] = primary_residual_loss.detach().item()
                        if primary_patch_loss is not None:
                            prior_metrics["view_curriculum_primary_patch"] = primary_patch_loss.detach().item()
                        if primary_patch_lowfreq_loss is not None:
                            prior_metrics["view_curriculum_primary_patch_lowfreq"] = (
                                primary_patch_lowfreq_loss.detach().item()
                            )
                        if primary_patch_coverage is not None:
                            prior_metrics["view_curriculum_primary_patch_coverage"] = primary_patch_coverage

                    if fm_sds_block is not None:
                        fm_base_loss, fm_info = fm_sds_block.compute(
                            render_image=render_for_prior,
                            prior_image=prior_image,
                            mask=mask,
                        )
                        weighted_fm = hybrid_args.fm_sds_weight * fm_base_loss
                        if hybrid_args.fm_sds_weight > 0:
                            loss = loss + weighted_fm
                        fm_metrics = {
                            "total": weighted_fm.detach().item(),
                            "base": fm_base_loss.detach().item(),
                            "t": fm_info["t"],
                            "weight_t": fm_info["weight_t"],
                            "residual": fm_info["residual"],
                        }

                    if (
                        frequency_block is not None
                        and hybrid_args.freq_prior_weight > 0
                        and hybrid_args.freq_loss_weight > 0
                    ):
                        freq_prior_base, freq_prior_info = frequency_block.compute(
                            render_image=render_for_prior,
                            target_image=prior_image,
                            mask=mask,
                        )
                        weighted_freq_prior = (
                            hybrid_args.freq_loss_weight
                            * hybrid_args.freq_prior_weight
                            * freq_prior_base
                        )
                        loss = loss + weighted_freq_prior
                        freq_total_loss = freq_total_loss + weighted_freq_prior
                        freq_metrics["prior_total"] = weighted_freq_prior.detach().item()
                        freq_metrics["prior_low"] = freq_prior_info["low"]
                        freq_metrics["prior_mid"] = freq_prior_info["mid"]
                        freq_metrics["prior_high"] = freq_prior_info["high"]

        render_for_sof_prior = image if prior_render_image is None else prior_render_image
        camera_for_sof_prior = prior_camera if prior_camera is not None else viewpoint_cam

        if sof_prior_block is not None and (prior_local_bank is not None or prior_edge_bank is not None):
            sof_gt_image = (
                prior_gt_image
                if prior_gt_image is not None
                else camera_for_sof_prior.original_image.to(device=render_for_sof_prior.device)
            )

            if prior_local_bank is not None and prior_local_mask_bank is not None:
                local_prior_image = prior_local_bank.get(
                    camera_for_sof_prior.image_name,
                    width=int(camera_for_sof_prior.image_width),
                    height=int(camera_for_sof_prior.image_height),
                    device=render_for_sof_prior.device,
                )
                local_prior_mask = prior_local_mask_bank.get(
                    camera_for_sof_prior.image_name,
                    width=int(camera_for_sof_prior.image_width),
                    height=int(camera_for_sof_prior.image_height),
                    device=render_for_sof_prior.device,
                )
                loss_prior_local = sof_prior_block.compute_local_loss(
                    render_image=render_for_sof_prior,
                    prior_image=local_prior_image,
                    prior_mask=local_prior_mask,
                    iteration=iteration,
                )
                if loss_prior_local is not None:
                    weighted_prior_local = hybrid_args.lambda_prior_local * loss_prior_local
                    if current_prior_loss_update_mask is not None:
                        masked_prior_loss = weighted_prior_local if masked_prior_loss is None else masked_prior_loss + weighted_prior_local
                    else:
                        loss = loss + weighted_prior_local
                    prior_metrics["sof_local"] = loss_prior_local.detach().item()

            if prior_edge_bank is not None and prior_edge_mask_bank is not None:
                edge_prior_image = prior_edge_bank.get(
                    camera_for_sof_prior.image_name,
                    width=int(camera_for_sof_prior.image_width),
                    height=int(camera_for_sof_prior.image_height),
                    device=render_for_sof_prior.device,
                )
                edge_prior_mask = prior_edge_mask_bank.get(
                    camera_for_sof_prior.image_name,
                    width=int(camera_for_sof_prior.image_width),
                    height=int(camera_for_sof_prior.image_height),
                    device=render_for_sof_prior.device,
                )
                lowfreq_anchor = sof_gt_image if hybrid_args.prior_edge_lowfreq_anchor == "gt" else None
                loss_prior_edge, edge_alpha = sof_prior_block.compute_edge_loss(
                    render_image=render_for_sof_prior,
                    prior_image=edge_prior_image,
                    prior_mask=edge_prior_mask,
                    iteration=iteration,
                    train_start_iter=train_start_iter,
                    lowfreq_anchor=lowfreq_anchor,
                )
                if loss_prior_edge is not None:
                    weighted_prior_edge = hybrid_args.lambda_prior_edge * loss_prior_edge
                    if current_prior_loss_update_mask is not None:
                        masked_prior_loss = weighted_prior_edge if masked_prior_loss is None else masked_prior_loss + weighted_prior_edge
                    else:
                        loss = loss + weighted_prior_edge
                    prior_metrics["sof_edge"] = loss_prior_edge.detach().item()
                if edge_alpha is not None:
                    prior_metrics["sof_edge_alpha"] = float(edge_alpha)

                edge_touch_mask = sof_prior_block.build_touch_mask(
                    viewpoint_cam=camera_for_sof_prior,
                    gaussians=gaussians,
                    image_mask=edge_prior_mask,
                    visibility_filter=prior_visibility_filter,
                    radii=prior_radii,
                )
                if edge_touch_mask is not None:
                    touch_update_mask = combine_gaussian_update_masks(external_update_mask, current_prior_loss_update_mask)
                    if touch_update_mask is not None:
                        edge_touch_mask = edge_touch_mask & touch_update_mask
                    if hasattr(gaussians, "mark_edge_touched"):
                        gaussians.mark_edge_touched(edge_touch_mask, iteration)
                    visible_count = float(
                        prior_visibility_filter.sum().item()
                        if prior_visibility_filter is not None
                        else edge_touch_mask.shape[0]
                    )
                    touched_count = float(edge_touch_mask.sum().item())
                    prior_metrics["sof_edge_touch_ratio"] = touched_count / max(visible_count, 1.0)
                    prior_metrics["sof_edge_touched"] = touched_count

        if (
            surface_route_bank is not None
            and iteration >= int(hybrid_args.surface_route_consensus_from_iter)
        ):
            route_payload = surface_route_bank.get(
                camera_for_sof_prior.image_name,
                width=int(camera_for_sof_prior.image_width),
                height=int(camera_for_sof_prior.image_height),
                device=render_for_sof_prior.device,
            )
            if route_payload is not None:
                route_target = route_payload["target_highfreq"]
                route_residual = route_payload.get("target_residual")
                route_surface_anchor = route_payload.get("surface_anchor_highfreq")
                route_signal_mode = str(route_payload.get("signal_mode", "absolute_highfreq"))
                route_weight = route_payload["target_weight"].clamp(min=0.0, max=1.0)
                valid_pixels = float(route_weight.sum().detach().item())
                if valid_pixels >= float(hybrid_args.surface_route_consensus_min_pixels):
                    route_needs_anchor_residual = route_signal_mode == "anchor_residual"
                    use_surface_route_branch = (
                        bool(hybrid_args.surface_route_surface_only)
                        and surface_route_gaussians is not None
                        and (
                            not route_needs_anchor_residual
                            or (route_residual is not None and route_surface_anchor is not None)
                        )
                    )
                    if use_surface_route_branch:
                        surface_route_render_pkg = train_render_fn(
                            camera_for_sof_prior,
                            surface_route_gaussians,
                            pipe,
                            background,
                            kernel_size=dataset.kernel_size,
                            subpixel_offset=None,
                        )
                        pred_highfreq = _laplacian_highfreq(surface_route_render_pkg["render"])
                        if route_signal_mode == "direct_sr_highfreq":
                            pred_route_value = pred_highfreq
                            route_target_value = route_target
                        elif route_signal_mode == "anchor_residual":
                            pred_route_value = pred_highfreq - route_surface_anchor
                            route_target_value = route_residual
                        else:
                            pred_route_value = pred_highfreq
                            route_target_value = route_target
                    else:
                        pred_highfreq = _laplacian_highfreq(render_for_sof_prior)
                        pred_route_value = pred_highfreq
                        route_target_value = route_target
                    denom = torch.clamp(route_weight.sum() * pred_highfreq.shape[0], min=1.0)
                    loss_surface_route = ((pred_route_value - route_target_value).abs() * route_weight).sum() / denom
                    weighted_surface_route = hybrid_args.lambda_surface_route_consensus * loss_surface_route
                    if current_prior_loss_update_mask is not None:
                        masked_prior_loss = (
                            weighted_surface_route
                            if masked_prior_loss is None
                            else masked_prior_loss + weighted_surface_route
                        )
                    else:
                        loss = loss + weighted_surface_route
                    prior_metrics["surface_route"] = loss_surface_route.detach().item()
                    prior_metrics["surface_route_valid_pixels"] = valid_pixels
                    prior_metrics["surface_route_weight_mean"] = route_weight.mean().detach().item()
                    prior_metrics["surface_route_surface_only"] = 1.0 if use_surface_route_branch else 0.0
                    prior_metrics["surface_route_direct_signal"] = 1.0 if route_signal_mode == "direct_sr_highfreq" else 0.0

        if bool(hybrid_args.prior_only_edge_finetune):
            if "sof_edge" not in prior_metrics:
                prior_edge_skip_count += 1
                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "Loss": "skip",
                            "Skip": str(prior_edge_skip_count),
                            "Size": f"{len(gaussians._xyz)}",
                        }
                    )
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                continue
            loss = hybrid_args.lambda_prior_edge * loss_prior_edge

        if sof_regularization_block is not None:
            sof_reg_loss, sof_reg_metrics = sof_regularization_block.compute(
                gaussians,
                iteration,
                render_ctx={
                    "viewpoint_camera": viewpoint_cam,
                    "gt_image": gt_image,
                    "render_pkg": render_pkg,
                    "render_fn": train_render_fn,
                    "pipe": pipe,
                    "background": background,
                    "kernel_size": dataset.kernel_size,
                    "subpixel_offset": subpixel_offset,
                },
            )
            if sof_reg_loss is not None:
                loss = loss + sof_reg_loss

        if layer_frequency_regularizer is not None:
            layer_surface_target = str(
                getattr(hybrid_args, "layer_frequency_surface_target", "gt")
            ).strip().lower()
            layer_target_name = "gt"
            if layer_surface_target in {"prior", "sr_prior"} and prior_image is not None:
                layer_gt_image = prior_image.detach().to(
                    device=render_for_sof_prior.device,
                    dtype=render_for_sof_prior.dtype,
                )
                layer_target_name = "prior"
            elif layer_surface_target in {"anchor", "reference"} and prior_anchor_image is not None:
                layer_gt_image = prior_anchor_image.detach().to(
                    device=render_for_sof_prior.device,
                    dtype=render_for_sof_prior.dtype,
                )
                layer_target_name = "anchor"
            elif prior_gt_image is not None:
                layer_gt_image = prior_gt_image
                if layer_surface_target not in {"gt", "image", "reference_gt"}:
                    layer_target_name = f"fallback_gt_from_{layer_surface_target or 'empty'}"
            elif camera_for_sof_prior is viewpoint_cam:
                layer_gt_image = gt_image
            else:
                layer_gt_image = camera_for_sof_prior.original_image.to(device=render_for_sof_prior.device)
            layer_subpixel_offset = subpixel_offset if camera_for_sof_prior is viewpoint_cam else None
            layer_start_image = None
            if layer_frequency_start_gaussians is not None:
                with torch.no_grad():
                    layer_start_pkg = train_render_fn(
                        camera_for_sof_prior,
                        layer_frequency_start_gaussians,
                        pipe,
                        background,
                        kernel_size=dataset.kernel_size,
                        subpixel_offset=layer_subpixel_offset,
                    )
                    layer_start_image = layer_start_pkg["render"].detach()
            layer_direct_loss, layer_frequency_surface_loss, layer_freq_metrics = (
                layer_frequency_regularizer.compute(
                    iteration=iteration,
                    viewpoint_camera=camera_for_sof_prior,
                    gt_image=layer_gt_image,
                    render_image=render_for_sof_prior,
                    render_fn=train_render_fn,
                    pipe=pipe,
                    background=background,
                    kernel_size=dataset.kernel_size,
                    subpixel_offset=layer_subpixel_offset,
                    start_image=layer_start_image,
                )
            )
            if layer_freq_metrics is not None:
                layer_freq_metrics["target"] = layer_target_name
                layer_freq_metrics["guidance_scale"] = float(prior_guidance_scale)
            if layer_direct_loss is not None:
                loss = loss + layer_direct_loss

        if prior_bubble_cleanup_regularizer is not None:
            bubble_cleanup_loss, bubble_cleanup_metrics = prior_bubble_cleanup_regularizer.compute(
                iteration=iteration,
                gaussians=gaussians,
                viewpoint_camera=camera_for_sof_prior,
                render_fn=train_render_fn,
                pipe=pipe,
                background=background,
                kernel_size=dataset.kernel_size,
                subpixel_offset=(subpixel_offset if camera_for_sof_prior is viewpoint_cam else None),
            )
            if bubble_cleanup_loss is not None:
                loss = loss + bubble_cleanup_loss
            if bubble_cleanup_metrics:
                prior_metrics.update(bubble_cleanup_metrics)

        if surface_migration_regularizer is not None:
            surface_migration_loss, surface_migration_metrics = surface_migration_regularizer.compute(
                gaussians,
                iteration,
            )
            if surface_migration_loss is not None:
                loss = loss + surface_migration_loss

        if (
            frequency_block is not None
            and hybrid_args.freq_gt_weight > 0
            and hybrid_args.freq_loss_weight > 0
        ):
            freq_gt_base, freq_gt_info = frequency_block.compute(
                render_image=image,
                target_image=gt_image,
                mask=None,
            )
            weighted_freq_gt = (
                hybrid_args.freq_loss_weight
                * hybrid_args.freq_gt_weight
                * freq_gt_base
            )
            loss = loss + weighted_freq_gt
            freq_total_loss = freq_total_loss + weighted_freq_gt
            freq_metrics["gt_total"] = weighted_freq_gt.detach().item()
            freq_metrics["gt_low"] = freq_gt_info["low"]
            freq_metrics["gt_mid"] = freq_gt_info["mid"]
            freq_metrics["gt_high"] = freq_gt_info["high"]

        if freq_metrics:
            freq_metrics["total"] = freq_total_loss.detach().item()

        needs_prior_direct_xyz_nudge = prior_direct_xyz_loss is not None
        needs_masked_grad = (
            masked_prior_loss is not None
            or layer_frequency_surface_loss is not None
            or needs_prior_direct_xyz_nudge
        )
        loss.backward(retain_graph=needs_masked_grad)
        if needs_prior_direct_xyz_nudge:
            prior_direct_xyz_grad = torch.autograd.grad(
                prior_direct_xyz_loss,
                gaussians._xyz,
                retain_graph=(masked_prior_loss is not None or layer_frequency_surface_loss is not None),
                allow_unused=True,
            )[0]
            if prior_direct_xyz_grad is not None:
                prior_direct_xyz_grad = prior_direct_xyz_grad.detach()
        if masked_prior_loss is not None:
            masked_prior_update_mask = combine_gaussian_update_masks(
                current_prior_loss_update_mask,
                external_update_mask,
            )
            accumulate_masked_gaussian_loss_grads(
                gaussians,
                masked_prior_loss,
                update_mask=masked_prior_update_mask,
                update_scale=float(hybrid_args.prior_edge_update_scale) * float(prior_guidance_scale),
                retain_graph=(
                    layer_frequency_surface_loss is not None
                    or bool(getattr(hybrid_args, "prior_masked_update_densify_signal", False))
                ),
            )
            if bool(getattr(hybrid_args, "prior_masked_update_densify_signal", False)):
                accumulate_masked_viewspace_loss_grads(
                    masked_prior_loss,
                    prior_viewspace_point_tensor,
                    update_mask=masked_prior_update_mask,
                    update_scale=(
                        float(getattr(hybrid_args, "prior_masked_update_densify_signal_scale", 1.0))
                        * float(prior_guidance_scale)
                    ),
                    retain_graph=layer_frequency_surface_loss is not None,
                )
                if prior_metrics is not None:
                    prior_metrics["masked_densify_signal"] = 1.0
        if layer_frequency_surface_loss is not None and layer_frequency_regularizer.surface_mask is not None:
            accumulate_masked_gaussian_loss_grads(
                gaussians,
                layer_frequency_surface_loss,
                update_mask=combine_gaussian_update_masks(
                    layer_frequency_regularizer.surface_mask,
                    external_update_mask,
                ),
                update_scale=float(hybrid_args.surface_hf_update_scale) * float(prior_guidance_scale),
            )
        iter_end.record()

        with torch.no_grad():
            source_update_mask = build_source_tag_train_mask(gaussians, hybrid_args.optimize_source_tag)
            update_mask = combine_gaussian_update_masks(source_update_mask, external_update_mask)
            apply_gaussian_update_mask(
                gaussians,
                update_mask,
                update_scale=float(hybrid_args.prior_edge_update_scale),
            )
            if surface_normal_lock is not None:
                surface_normal_lock.project_xyz_gradient_to_tangent(gaussians)
            active_prior_direct_xyz_nudge_mask = prior_direct_xyz_nudge_mask
            if active_prior_direct_xyz_nudge_mask is None:
                active_prior_direct_xyz_nudge_mask = update_mask
            if prior_direct_xyz_grad is not None and active_prior_direct_xyz_nudge_mask is not None:
                prior_direct_xyz_nudge_metrics = apply_prior_direct_xyz_nudge(
                    gaussians=gaussians,
                    xyz_grad=prior_direct_xyz_grad,
                    update_mask=active_prior_direct_xyz_nudge_mask,
                    lr=float(hybrid_args.prior_direct_xyz_nudge_lr),
                    max_step=float(hybrid_args.prior_direct_xyz_nudge_max_step),
                    surface_normal_lock=surface_normal_lock,
                )
                if prior_direct_xyz_nudge_metrics:
                    prior_metrics.update(prior_direct_xyz_nudge_metrics)
            if (
                hybrid_args.sdf_proposal_realign_enable
                and sdf_adapter is not None
                and proposal_realign is not None
                and iteration >= int(hybrid_args.sdf_proposal_realign_start_iter)
                and iteration % max(1, int(hybrid_args.sdf_proposal_realign_interval)) == 0
            ):
                realign_metrics = _apply_view_anchored_surface_realign(
                    gaussians=gaussians,
                    reference_camera=viewpoint_cam,
                    realign_bundle=proposal_realign,
                    hybrid_args=hybrid_args,
                )
                proposal_metrics["realigned_gs"] = realign_metrics["realigned_gs"]
                proposal_metrics["realign_shift_mean"] = realign_metrics["realign_shift_mean"]
                proposal_metrics["realign_depth_delta_mean"] = realign_metrics["realign_depth_delta_mean"]
                proposal_metrics["realign_rot_mean"] = realign_metrics["realign_rot_mean"]
                proposal_metrics["realign_scale_ratio_mean"] = realign_metrics["realign_scale_ratio_mean"]

            if (
                _proposal_full_stack_enabled(hybrid_args)
                and proposal_native_birth is not None
                and iteration >= int(hybrid_args.sdf_proposal_native_birth_start_iter)
                and iteration % max(1, int(hybrid_args.sdf_proposal_native_birth_interval)) == 0
            ):
                birth_metrics = _apply_native_surface_birth(
                    gaussians=gaussians,
                    birth_bundle=proposal_native_birth,
                    reference_camera=viewpoint_cam,
                    hybrid_args=hybrid_args,
                    train_cameras=train_cameras,
                )
                proposal_metrics["native_births"] = birth_metrics["native_births"]

            if bool(hybrid_args.prior_hf_seed_scale_clamp_enable):
                scale_clamp_metrics = _clamp_prior_injected_scales(
                    gaussians,
                    max_axis=float(hybrid_args.prior_hf_seed_scale_clamp_max_axis),
                    min_axis=float(hybrid_args.prior_hf_seed_scale_clamp_min_axis),
                    max_anisotropy=float(hybrid_args.prior_hf_seed_scale_clamp_max_anisotropy),
                )
                if scale_clamp_metrics and prior_metrics is not None:
                    prior_metrics.update(scale_clamp_metrics)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, dataset.kernel_size),
                hybrid_metrics=hybrid_metrics,
                sdf_densify_metrics=sdf_densify_metrics,
                proposal_metrics=proposal_metrics,
                prior_metrics=prior_metrics,
                sof_reg_metrics=sof_reg_metrics,
                fm_metrics=fm_metrics,
                freq_metrics=freq_metrics,
                scaffold_metrics=scaffold_metrics,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                densify_candidate_mask = update_mask
                densify_visibility_filter = visibility_filter
                if densify_candidate_mask is not None:
                    densify_visibility_filter = visibility_filter & densify_candidate_mask
                if torch.any(densify_visibility_filter):
                    gaussians.max_radii2D[densify_visibility_filter] = torch.max(
                        gaussians.max_radii2D[densify_visibility_filter],
                        radii[densify_visibility_filter],
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, densify_visibility_filter)
                if (
                    bool(getattr(hybrid_args, "prior_masked_update_densify_signal", False))
                    and prior_viewspace_point_tensor is not viewspace_point_tensor
                    and prior_viewspace_point_tensor.grad is not None
                    and prior_visibility_filter is not None
                    and int(prior_visibility_filter.shape[0]) == int(gaussians.get_xyz.shape[0])
                    and int(prior_radii.shape[0]) == int(gaussians.get_xyz.shape[0])
                ):
                    prior_densify_visibility_filter = prior_visibility_filter
                    if densify_candidate_mask is not None:
                        prior_densify_visibility_filter = prior_densify_visibility_filter & densify_candidate_mask
                    if torch.any(prior_densify_visibility_filter):
                        gaussians.max_radii2D[prior_densify_visibility_filter] = torch.max(
                            gaussians.max_radii2D[prior_densify_visibility_filter],
                            prior_radii[prior_densify_visibility_filter],
                        )
                        gaussians.add_densification_stats(
                            prior_viewspace_point_tensor,
                            prior_densify_visibility_filter,
                        )
                        if prior_metrics is not None:
                            prior_metrics["masked_densify_prior_view"] = 1.0

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    if prior_metrics is not None and hold_prior_densify_for_seed_bootstrap:
                        prior_metrics["hf_seed_densify_hold"] = 1.0
                    if not hold_prior_densify_for_seed_bootstrap:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            candidate_mask=densify_candidate_mask,
                            protect_source_tag=(
                                int(GaussianSourceTag.PRIOR_INJECTED)
                                if recent_prior_protect_iters > 0
                                else None
                            ),
                            protect_recent_iters=recent_prior_protect_iters,
                            current_iteration=iteration,
                        )
                        compute_3d_filter_with_surface_filter(
                            gaussians,
                            cameras=train_cameras,
                            surface_filter_3d=surface_filter_3d,
                        )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    compute_3d_filter_with_surface_filter(
                        gaussians,
                        cameras=train_cameras,
                        surface_filter_3d=surface_filter_3d,
                    )

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if surface_migration_regularizer is not None:
                    surface_migration_regularizer.apply_post_step(gaussians, iteration)
                if surface_normal_lock is not None:
                    surface_normal_lock.project_xyz_to_locked_normal_coord(gaussians)
                if bool(hybrid_args.prior_hf_seed_scale_clamp_enable):
                    scale_clamp_metrics = _clamp_prior_injected_scales(
                        gaussians,
                        max_axis=float(hybrid_args.prior_hf_seed_scale_clamp_max_axis),
                        min_axis=float(hybrid_args.prior_hf_seed_scale_clamp_min_axis),
                        max_anisotropy=float(hybrid_args.prior_hf_seed_scale_clamp_max_anisotropy),
                    )
                    if scale_clamp_metrics and prior_metrics is not None:
                        prior_metrics.update(scale_clamp_metrics)
                if prior_bubble_cleanup_regularizer is not None:
                    bubble_post_metrics = prior_bubble_cleanup_regularizer.apply_post_step(
                        gaussians,
                        iteration,
                    )
                    if bubble_post_metrics:
                        prior_metrics.update(bubble_post_metrics)
                if bool(hybrid_args.prior_hf_lowfreq_cleanup_enable):
                    lowfreq_cleanup_metrics = _apply_prior_hf_lowfreq_cleanup(
                        gaussians,
                        iteration=iteration,
                        from_iter=int(hybrid_args.prior_hf_lowfreq_cleanup_from_iter),
                        until_iter=int(hybrid_args.prior_hf_lowfreq_cleanup_until_iter),
                        interval=int(hybrid_args.prior_hf_lowfreq_cleanup_interval),
                        stale_iters=int(hybrid_args.prior_hf_lowfreq_cleanup_stale_iters),
                        opacity_min=float(hybrid_args.prior_hf_lowfreq_cleanup_opacity_min),
                        scale_max_min=float(hybrid_args.prior_hf_lowfreq_cleanup_scale_max_min),
                        scale_max_max=float(hybrid_args.prior_hf_lowfreq_cleanup_scale_max_max),
                        anisotropy_max=float(hybrid_args.prior_hf_lowfreq_cleanup_anisotropy_max),
                        opacity_decay=float(hybrid_args.prior_hf_lowfreq_cleanup_opacity_decay),
                        prune_opacity_below=float(hybrid_args.prior_hf_lowfreq_cleanup_prune_opacity_below),
                        max_prune_fraction=float(hybrid_args.prior_hf_lowfreq_cleanup_max_prune_fraction),
                        max_prune_count=int(hybrid_args.prior_hf_lowfreq_cleanup_max_prune_count),
                    )
                    if lowfreq_cleanup_metrics:
                        prior_metrics.update(lowfreq_cleanup_metrics)
                        if lowfreq_cleanup_metrics.get("hf_lowfreq_cleanup_pruned", 0.0) > 0.0:
                            compute_3d_filter_with_surface_filter(
                                gaussians,
                                cameras=train_cameras,
                                surface_filter_3d=surface_filter_3d,
                            )
                gaussians.optimizer.zero_grad(set_to_none=True)

                seed_first_cycle_only = bool(getattr(hybrid_args, "prior_hf_seed_first_cycle_only", False))
                seed_allowed_by_view_curriculum = True
                if prior_view_curriculum_state is not None and bool(
                    getattr(hybrid_args, "prior_view_curriculum_birth_primary_only", False)
                ):
                    seed_allowed_by_view_curriculum = (
                        str(prior_view_curriculum_state.get("phase", "")) == "primary"
                    )
                if (
                    hybrid_args.prior_hf_seed_enable
                    and prior_feature_pack is not None
                    and prior_feature_pack.get("guidance") is not None
                    and iteration >= int(hybrid_args.prior_hf_seed_from_iter)
                    and (
                        int(hybrid_args.prior_hf_seed_until_iter) <= 0
                        or iteration <= int(hybrid_args.prior_hf_seed_until_iter)
                    )
                    and iteration % max(1, int(hybrid_args.prior_hf_seed_interval)) == 0
                    and prior_hf_seed_total < int(hybrid_args.prior_hf_seed_max_total)
                    and (not seed_first_cycle_only or prior_hf_seed_first_visit)
                    and seed_allowed_by_view_curriculum
                ):
                    remaining = int(hybrid_args.prior_hf_seed_max_total) - prior_hf_seed_total
                    seed_births, seed_metrics = _build_prior_residual_seed_births(
                        gaussians=gaussians,
                        prior_feature_pack=prior_feature_pack,
                        prior_render_pkg=prior_render_pkg,
                        reference_camera=camera_for_sof_prior,
                        visibility_filter=prior_visibility_filter,
                        iteration=iteration,
                        max_points=min(int(hybrid_args.prior_hf_seed_max_per_iter), remaining),
                        prior_delta_clip=float(hybrid_args.prior_delta_clip),
                        guidance_threshold=float(hybrid_args.prior_hf_seed_guidance_threshold),
                        scale_multiplier=float(hybrid_args.prior_hf_seed_scale_multiplier),
                        scale_mode=str(hybrid_args.prior_hf_seed_scale_mode),
                        min_pixel_radius=float(hybrid_args.prior_hf_seed_min_pixel_radius),
                        max_pixel_radius=float(hybrid_args.prior_hf_seed_max_pixel_radius),
                        max_provider_ratio=float(hybrid_args.prior_hf_seed_max_provider_ratio),
                        shape_mode=str(hybrid_args.prior_hf_seed_shape_mode),
                        shape_long_ratio=float(hybrid_args.prior_hf_seed_shape_long_ratio),
                        shape_short_ratio=float(hybrid_args.prior_hf_seed_shape_short_ratio),
                        shape_normal_ratio=float(hybrid_args.prior_hf_seed_shape_normal_ratio),
                        shape_confidence_power=float(hybrid_args.prior_hf_seed_shape_confidence_power),
                        base_opacity=float(hybrid_args.prior_hf_seed_opacity),
                        jitter_scale=float(hybrid_args.prior_hf_seed_jitter_scale),
                        color_residual_gain=float(hybrid_args.prior_hf_seed_color_residual_gain),
                        original_only=bool(hybrid_args.prior_hf_seed_original_only),
                    )
                    if seed_metrics:
                        prior_metrics.update(seed_metrics)
                        last_seed_diag = dict(seed_metrics)
                        prior_metrics["hf_seed_first_cycle_pending"] = float(len(first_cycle_pending_views))
                        prior_metrics["hf_seed_first_visit"] = 1.0 if prior_hf_seed_first_visit else 0.0
                        prior_metrics["hf_seed_total"] = float(prior_hf_seed_total)
                        if seed_births is None:
                            prior_metrics["hf_seeded"] = 0.0
                    if seed_births is not None:
                        gaussians.densification_postfix(
                            seed_births["xyz"],
                            seed_births["features_dc"],
                            seed_births["features_rest"],
                            seed_births["opacities"],
                            seed_births["scaling"],
                            seed_births["rotation"],
                            tracking_state=seed_births["tracking_state"],
                        )
                        seeded = int(seed_births["xyz"].shape[0])
                        prior_hf_seed_total += int(seeded)
                        compute_3d_filter_with_surface_filter(
                            gaussians,
                            cameras=train_cameras,
                            surface_filter_3d=surface_filter_3d,
                        )
                        if bool(hybrid_args.prior_hf_seed_scale_clamp_enable):
                            scale_clamp_metrics = _clamp_prior_injected_scales(
                                gaussians,
                                max_axis=float(hybrid_args.prior_hf_seed_scale_clamp_max_axis),
                                min_axis=float(hybrid_args.prior_hf_seed_scale_clamp_min_axis),
                                max_anisotropy=float(hybrid_args.prior_hf_seed_scale_clamp_max_anisotropy),
                            )
                            if scale_clamp_metrics:
                                prior_metrics.update(scale_clamp_metrics)
                        prior_metrics["hf_seeded"] = float(seeded)
                        prior_metrics["hf_seed_total"] = float(prior_hf_seed_total)
                    if seed_metrics and tb_writer:
                        if "hf_seeded" in prior_metrics:
                            tb_writer.add_scalar("prior/hf_seeded", prior_metrics["hf_seeded"], iteration)
                        if "hf_seed_total" in prior_metrics:
                            tb_writer.add_scalar("prior/hf_seed_total", prior_metrics["hf_seed_total"], iteration)
                        if "hf_seed_guidance_mean" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_guidance_mean",
                                prior_metrics["hf_seed_guidance_mean"],
                                iteration,
                            )
                        if "hf_seed_residual_abs_mean" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_residual_abs_mean",
                                prior_metrics["hf_seed_residual_abs_mean"],
                                iteration,
                            )
                        if "hf_seed_color_clip_ratio" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_color_clip_ratio",
                                prior_metrics["hf_seed_color_clip_ratio"],
                                iteration,
                            )
                        if "hf_seed_render_depth_ratio" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_render_depth_ratio",
                                prior_metrics["hf_seed_render_depth_ratio"],
                                iteration,
                            )
                        if "hf_seed_provider_dist_mean" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_provider_dist_mean",
                                prior_metrics["hf_seed_provider_dist_mean"],
                                iteration,
                            )
                        if "hf_seed_threshold_candidates" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_threshold_candidates",
                                prior_metrics["hf_seed_threshold_candidates"],
                                iteration,
                            )
                        if "hf_seed_selected_pixels" in prior_metrics:
                            tb_writer.add_scalar(
                                "prior/hf_seed_selected_pixels",
                                prior_metrics["hf_seed_selected_pixels"],
                                iteration,
                            )

            live_prior_count = int(
                (gaussians._source_tag == int(GaussianSourceTag.PRIOR_INJECTED)).sum().item()
            ) if hasattr(gaussians, "_source_tag") else 0
            if prior_metrics:
                prior_metrics.update(
                    _summarize_prior_injected_gaussians(
                        gaussians,
                        large_scale_threshold=float(hybrid_args.prior_hf_seed_diag_large_scale),
                    )
                )
                if "hf_seed_live" not in prior_metrics:
                    prior_metrics["hf_seed_live"] = float(live_prior_count)
                prior_metrics.update(seed_gate_diag)
            if seed_bootstrap_required:
                seed_window_exhausted = False
                if iteration >= int(hybrid_args.prior_hf_seed_from_iter):
                    if seed_first_cycle_only_active and len(first_cycle_pending_views) == 0:
                        seed_window_exhausted = True
                    elif (
                        int(hybrid_args.prior_hf_seed_until_iter) > 0
                        and iteration >= int(hybrid_args.prior_hf_seed_until_iter)
                    ):
                        seed_window_exhausted = True
                    elif iteration >= int(opt.iterations):
                        seed_window_exhausted = True
                if seed_window_exhausted and live_prior_count <= 0:
                    diag_bits = []
                    for key in (
                        "hf_seed_gate_enable",
                        "hf_seed_gate_first_visit",
                        "hf_seed_gate_prior_pack",
                        "hf_seed_gate_guidance",
                        "hf_seed_gate_prior_bank",
                        "hf_seed_gate_anchor_bank",
                        "hf_seed_gate_prior_index",
                        "hf_seed_gate_anchor_index",
                        "hf_seed_threshold_candidates",
                        "hf_seed_selected_pixels",
                        "hf_seed_guidance_map_max",
                        "hf_seed_guidance_map_nonzero",
                        "hf_seed_empty_candidates",
                        "hf_seed_no_provider",
                        "hf_seed_no_valid_depth",
                        "hf_seed_no_prior_rgb",
                    ):
                        if key in last_seed_diag:
                            value = last_seed_diag[key]
                            if isinstance(value, float):
                                if abs(value - round(value)) < 1e-6:
                                    value = int(round(value))
                                else:
                                    value = f"{value:.6f}"
                            diag_bits.append(f"{key}={value}")
                    raise RuntimeError(
                        "HF seed bootstrap ended with zero live PRIOR_INJECTED gaussians. "
                        "The prior branch was born and then fully pruned away, so the run degenerates "
                        "back to the baseline. Increase newborn protection or recent HF supervision. "
                        f"last_seed_total={prior_hf_seed_total} "
                        + (" ".join(diag_bits) if diag_bits else "last_seed_diag=missing")
                    )

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            if (
                hybrid_metrics
                and iteration % max(1, hybrid_args.hybrid_log_interval) == 0
            ):
                print(
                    "[HYBRID] "
                    f"iter={iteration} "
                    f"stage={hybrid_metrics['stage']} "
                    f"scale={hybrid_metrics['scale']:.3f} "
                    f"surf={hybrid_metrics['surface']:.6f} "
                    f"norm={hybrid_metrics['normal']:.6f} "
                    f"off={hybrid_metrics['offsurface']:.6f} "
                    f"total={hybrid_metrics['total']:.6f}"
                )
            if (
                prior_metrics
                and iteration % max(1, hybrid_args.prior_log_interval) == 0
            ):
                prior_parts = []
                if "total" in prior_metrics:
                    prior_parts.extend(
                        [
                            f"total={prior_metrics['total']:.6f}",
                            f"l1={prior_metrics['l1']:.6f}",
                            f"hf={prior_metrics['hf']:.6f}",
                            f"valid={prior_metrics['valid_ratio']:.3f}",
                        ]
                    )
                if "sequence_tex" in prior_metrics:
                    prior_parts.extend(
                        [
                            f"seq_tex={prior_metrics['sequence_tex']:.6f}",
                            f"seq_sp={prior_metrics['sequence_sp']:.6f}",
                            f"seq_ltex={prior_metrics['sequence_lambda_tex']:.2f}",
                        ]
                    )
                if "sof_local" in prior_metrics:
                    prior_parts.append(f"local={prior_metrics['sof_local']:.6f}")
                if "sof_edge" in prior_metrics:
                    prior_parts.append(f"edge={prior_metrics['sof_edge']:.6f}")
                if "sof_edge_alpha" in prior_metrics:
                    prior_parts.append(f"alpha={prior_metrics['sof_edge_alpha']:.3f}")
                if "sof_edge_touch_ratio" in prior_metrics:
                    prior_parts.append(f"touch={prior_metrics['sof_edge_touch_ratio']:.3f}")
                if "hf_seeded" in prior_metrics:
                    prior_parts.extend(
                        [
                            f"seed={int(prior_metrics['hf_seeded'])}",
                            f"seed_total={int(prior_metrics['hf_seed_total'])}",
                        ]
                    )
                if "hf_seed_guidance_mean" in prior_metrics:
                    prior_parts.append(f"seed_hf={prior_metrics['hf_seed_guidance_mean']:.4f}")
                if "hf_seed_color_residual_gain" in prior_metrics:
                    prior_parts.append(f"seed_cgain={prior_metrics['hf_seed_color_residual_gain']:.2f}")
                if "hf_seed_color_clip_ratio" in prior_metrics:
                    prior_parts.append(f"seed_cclip={prior_metrics['hf_seed_color_clip_ratio']:.2f}")
                if "hf_seed_threshold_candidates" in prior_metrics:
                    prior_parts.append(f"seed_px={int(prior_metrics['hf_seed_threshold_candidates'])}")
                if "hf_seed_selected_pixels" in prior_metrics:
                    prior_parts.append(f"seed_sel={int(prior_metrics['hf_seed_selected_pixels'])}")
                if "hf_seed_render_depth_ratio" in prior_metrics:
                    prior_parts.append(f"seed_zr={prior_metrics['hf_seed_render_depth_ratio']:.2f}")
                if "hf_seed_provider_dist_mean" in prior_metrics:
                    prior_parts.append(f"seed_nn={prior_metrics['hf_seed_provider_dist_mean']:.2f}px")
                if "hf_seed_scale_mode" in prior_metrics:
                    prior_parts.append(f"seed_mode={prior_metrics['hf_seed_scale_mode']}")
                if "hf_seed_shape_mode_id" in prior_metrics:
                    shape_id = int(prior_metrics["hf_seed_shape_mode_id"])
                    shape_name = "hf" if shape_id == 2 else ("flat" if shape_id == 1 else "iso")
                    prior_parts.append(f"shape={shape_name}")
                if "hf_seed_shape_anisotropy_p90" in prior_metrics:
                    prior_parts.append(f"shape_a90={prior_metrics['hf_seed_shape_anisotropy_p90']:.2f}")
                if "hf_seed_shape_orient_conf_mean" in prior_metrics:
                    prior_parts.append(f"shape_conf={prior_metrics['hf_seed_shape_orient_conf_mean']:.2f}")
                if "hf_seed_pixel_radius_p90" in prior_metrics:
                    prior_parts.append(f"seed_rpx={prior_metrics['hf_seed_pixel_radius_p90']:.2f}")
                if "hf_seed_provider_ratio_p90" in prior_metrics:
                    prior_parts.append(f"seed_pr={prior_metrics['hf_seed_provider_ratio_p90']:.2f}")
                if "hf_seed_scale_p90" in prior_metrics:
                    prior_parts.append(f"seed_s90={prior_metrics['hf_seed_scale_p90']:.6f}")
                if "hf_seed_first_cycle_pending" in prior_metrics:
                    prior_parts.append(f"seed_rem={int(prior_metrics['hf_seed_first_cycle_pending'])}")
                if "hf_seed_live" in prior_metrics:
                    prior_parts.append(f"seed_live={int(prior_metrics['hf_seed_live'])}")
                if "hf_seed_unique_ids" in prior_metrics:
                    prior_parts.append(f"live_uid={int(prior_metrics['hf_seed_unique_ids'])}")
                if "hf_seed_clone_ratio" in prior_metrics:
                    prior_parts.append(f"live_clone={prior_metrics['hf_seed_clone_ratio']:.2f}")
                if "hf_seed_scale_max_axis_p90" in prior_metrics:
                    prior_parts.append(f"live_s90={prior_metrics['hf_seed_scale_max_axis_p90']:.6f}")
                if "hf_seed_scale_max_axis_max" in prior_metrics:
                    prior_parts.append(f"live_smax={prior_metrics['hf_seed_scale_max_axis_max']:.6f}")
                if "hf_seed_large_ratio" in prior_metrics:
                    prior_parts.append(f"live_large={prior_metrics['hf_seed_large_ratio']:.3f}")
                if "hf_seed_scale_clamp_count" in prior_metrics:
                    prior_parts.append(f"clamp={int(prior_metrics['hf_seed_scale_clamp_count'])}")
                if "hf_seed_scale_clamp_max_after" in prior_metrics:
                    prior_parts.append(f"clamp_max={prior_metrics['hf_seed_scale_clamp_max_after']:.6f}")
                if "hf_seed_scale_clamp_anis_p90_before" in prior_metrics:
                    prior_parts.append(
                        "clamp_anis="
                        f"{prior_metrics['hf_seed_scale_clamp_anis_p90_before']:.1f}->"
                        f"{prior_metrics.get('hf_seed_scale_clamp_anis_p90_after', 0.0):.1f}"
                    )
                if "bubble_cleanup_count" in prior_metrics:
                    prior_parts.append(
                        "bubble="
                        f"{int(prior_metrics['bubble_cleanup_count'])}/"
                        f"{int(prior_metrics.get('bubble_cleanup_prior', 0))}"
                    )
                if "bubble_cleanup_ratio" in prior_metrics:
                    prior_parts.append(f"bubble_r={prior_metrics['bubble_cleanup_ratio']:.3f}")
                if "bubble_cleanup_scale_max_p90" in prior_metrics:
                    prior_parts.append(f"bubble_s90={prior_metrics['bubble_cleanup_scale_max_p90']:.6f}")
                if "bubble_cleanup_anis_p90" in prior_metrics:
                    prior_parts.append(f"bubble_a90={prior_metrics['bubble_cleanup_anis_p90']:.2f}")
                if "bubble_cleanup_total_w" in prior_metrics:
                    prior_parts.append(f"bubble_w={prior_metrics['bubble_cleanup_total_w']:.6f}")
                if "bubble_cleanup_post_count" in prior_metrics:
                    prior_parts.append(
                        "bubble_decay="
                        f"{int(prior_metrics['bubble_cleanup_post_count'])}"
                        f"@{prior_metrics.get('bubble_cleanup_post_decay', 1.0):.2f}"
                    )
                if "hf_seed_gate_prior_pack" in prior_metrics:
                    prior_parts.append(f"seed_pack={int(prior_metrics['hf_seed_gate_prior_pack'])}")
                if "hf_seed_gate_guidance" in prior_metrics:
                    prior_parts.append(f"seed_guid={int(prior_metrics['hf_seed_gate_guidance'])}")
                if "hf_seed_gate_first_visit" in prior_metrics:
                    prior_parts.append(f"seed_visit={int(prior_metrics['hf_seed_gate_first_visit'])}")
                if prior_metrics.get("hf_seed_densify_hold", 0.0) > 0.0:
                    prior_parts.append("seed_hold=1")
                if prior_metrics.get("hf_seed_empty_candidates", 0.0) > 0.0:
                    prior_parts.append("seed_empty=1")
                if prior_metrics.get("hf_seed_no_provider", 0.0) > 0.0:
                    prior_parts.append("seed_ref=0")
                if prior_metrics.get("hf_seed_no_valid_depth", 0.0) > 0.0:
                    prior_parts.append("seed_depth=0")
                if "hf_focus_mean" in prior_metrics:
                    prior_parts.append(f"hf_w={prior_metrics['hf_focus_mean']:.3f}")
                if "hf_weight_eff" in prior_metrics:
                    prior_parts.append(f"hf_eff={prior_metrics['hf_weight_eff']:.2f}")
                if "guidance_scale" in prior_metrics and prior_metrics["guidance_scale"] < 0.999:
                    prior_parts.append(f"guide={prior_metrics['guidance_scale']:.2f}")
                if prior_metrics.get("masked_update", 0.0) > 0.0:
                    prior_parts.append("masked=1")
                if prior_metrics.get("masked_densify_signal", 0.0) > 0.0:
                    prior_parts.append("densig=1")
                if prior_metrics.get("masked_densify_prior_view", 0.0) > 0.0:
                    prior_parts.append("densig_hr=1")
                if "view_curriculum_phase" in prior_metrics:
                    phase_id = int(prior_metrics["view_curriculum_phase"])
                    phase_name = "primary" if phase_id == 1 else ("neighbor" if phase_id == 2 else "settle")
                    prior_parts.append(
                        "vc="
                        f"{phase_name}:{int(prior_metrics.get('view_curriculum_view_done', 0))}/"
                        f"{int(prior_metrics.get('view_curriculum_view_total', 0))}"
                    )
                if "view_curriculum_primary_hf_energy" in prior_metrics:
                    prior_parts.append(f"vc_hfe={prior_metrics['view_curriculum_primary_hf_energy']:.6f}")
                if "view_curriculum_primary_band" in prior_metrics:
                    prior_parts.append(f"vc_hfb={prior_metrics['view_curriculum_primary_band']:.6f}")
                if "view_curriculum_primary_residual" in prior_metrics:
                    prior_parts.append(f"vc_res={prior_metrics['view_curriculum_primary_residual']:.6f}")
                if "view_curriculum_primary_patch" in prior_metrics:
                    prior_parts.append(f"vc_patch={prior_metrics['view_curriculum_primary_patch']:.6f}")
                if "view_curriculum_primary_patch_lowfreq" in prior_metrics:
                    prior_parts.append(f"vc_patch_lf={prior_metrics['view_curriculum_primary_patch_lowfreq']:.6f}")
                if "view_curriculum_primary_patch_coverage" in prior_metrics:
                    prior_parts.append(f"vc_patch_cov={prior_metrics['view_curriculum_primary_patch_coverage']:.3f}")
                if "hf_recent_prior_live" in prior_metrics:
                    prior_parts.append(f"seed_recent={int(prior_metrics['hf_recent_prior_live'])}")
                if "hf_recent_prior_visible" in prior_metrics:
                    prior_parts.append(f"seed_vis={int(prior_metrics['hf_recent_prior_visible'])}")
                if prior_metrics.get("hf_recent_prior_boost", 0.0) > 0.0:
                    prior_parts.append(f"seed_boost={prior_metrics['hf_recent_prior_boost']:.2f}")
                if "hf_recent_imprint" in prior_metrics:
                    prior_parts.append(f"seed_imp={prior_metrics['hf_recent_imprint']:.6f}")
                if "hf_recent_imprint_visible" in prior_metrics:
                    prior_parts.append(f"seed_imp_vis={int(prior_metrics['hf_recent_imprint_visible'])}")
                if "hf_lowfreq_cleanup_candidates" in prior_metrics:
                    prior_parts.append(
                        "lf_clean="
                        f"{int(prior_metrics['hf_lowfreq_cleanup_candidates'])}/"
                        f"{int(prior_metrics.get('hf_lowfreq_cleanup_prior', 0))}"
                    )
                if "hf_lowfreq_cleanup_decay" in prior_metrics:
                    prior_parts.append(f"lf_decay={prior_metrics['hf_lowfreq_cleanup_decay']:.2f}")
                if "hf_lowfreq_cleanup_pruned" in prior_metrics:
                    prior_parts.append(f"lf_prune={int(prior_metrics['hf_lowfreq_cleanup_pruned'])}")
                if "xyz_nudge_count" in prior_metrics:
                    prior_parts.extend(
                        [
                            f"nudge_n={int(prior_metrics['xyz_nudge_count'])}",
                            f"nudge_grad={prior_metrics['xyz_nudge_grad_mean']:.6e}",
                            f"nudge_step={prior_metrics['xyz_nudge_step_mean']:.6e}",
                            f"nudge_max={prior_metrics['xyz_nudge_step_max']:.6e}",
                        ]
                    )
                print(
                    "[PRIOR] "
                    f"iter={iteration} "
                    + " ".join(prior_parts)
                )
            if (
                layer_freq_metrics
                and iteration % max(1, hybrid_args.layer_frequency_log_interval) == 0
            ):
                layer_parts = []
                if "total_w" in layer_freq_metrics:
                    layer_parts.append(f"total_w={layer_freq_metrics['total_w']:.6f}")
                if "ns_count" in layer_freq_metrics:
                    layer_parts.append(f"ns_count={layer_freq_metrics['ns_count']:.0f}")
                if "surf_count" in layer_freq_metrics:
                    layer_parts.append(f"surf_count={layer_freq_metrics['surf_count']:.0f}")
                if "target" in layer_freq_metrics:
                    layer_parts.append(f"target={layer_freq_metrics['target']}")
                if "guidance_scale" in layer_freq_metrics and layer_freq_metrics["guidance_scale"] < 0.999:
                    layer_parts.append(f"guide={layer_freq_metrics['guidance_scale']:.2f}")
                if "ns_rgb" in layer_freq_metrics:
                    layer_parts.append(
                        f"ns_rgb={layer_freq_metrics['ns_rgb']:.6f}"
                    )
                    layer_parts.append(
                        f"ns_rgb_w={layer_freq_metrics['ns_rgb_w']:.6f}"
                    )
                if "ns_hf" in layer_freq_metrics:
                    layer_parts.append(
                        f"ns_hf={layer_freq_metrics['ns_hf']:.6f}"
                    )
                    layer_parts.append(
                        f"ns_hf_w={layer_freq_metrics['ns_hf_w']:.6f}"
                    )
                if "ns_hf_protect" in layer_freq_metrics:
                    layer_parts.append(
                        f"ns_hf_protect={layer_freq_metrics['ns_hf_protect']:.3f}"
                    )
                if "ns_alpha_hf" in layer_freq_metrics:
                    layer_parts.append(
                        f"ns_alpha_hf={layer_freq_metrics['ns_alpha_hf']:.6f}"
                    )
                    layer_parts.append(
                        f"ns_alpha_hf_w={layer_freq_metrics['ns_alpha_hf_w']:.6f}"
                    )
                if "ns_alpha" in layer_freq_metrics:
                    layer_parts.append(
                        f"ns_alpha={layer_freq_metrics['ns_alpha']:.6f}"
                    )
                    layer_parts.append(
                        f"ns_alpha_w={layer_freq_metrics['ns_alpha_w']:.6f}"
                    )
                if "ns_alpha_missing" in layer_freq_metrics:
                    layer_parts.append("ns_alpha=missing")
                if "surf_hf" in layer_freq_metrics:
                    layer_parts.append(
                        f"surf_hf={layer_freq_metrics['surf_hf']:.6f}"
                    )
                    layer_parts.append(
                        f"surf_hf_w={layer_freq_metrics['surf_hf_w']:.6f}"
                    )
                if "start_hf" in layer_freq_metrics:
                    layer_parts.append(
                        f"start_hf={layer_freq_metrics['start_hf']:.6f}"
                    )
                    layer_parts.append(
                        f"start_hf_w={layer_freq_metrics['start_hf_w']:.6f}"
                    )
                if "start_mask" in layer_freq_metrics:
                    layer_parts.append(
                        f"start_mask={layer_freq_metrics['start_mask']:.3f}"
                    )
                if "start_lowerr" in layer_freq_metrics:
                    layer_parts.append(
                        f"start_lowerr={layer_freq_metrics['start_lowerr']:.6f}"
                    )
                if "start_energy" in layer_freq_metrics:
                    layer_parts.append(
                        f"start_energy={layer_freq_metrics['start_energy']:.6f}"
                    )
                print("[LAYER-FREQ] " f"iter={iteration} " + " ".join(layer_parts))
            if (
                surface_migration_metrics
                and iteration % max(1, hybrid_args.surface_migration_log_interval) == 0
            ):
                migrate_parts = []
                if "total" in surface_migration_metrics:
                    migrate_parts.append(f"total={surface_migration_metrics['total']:.6f}")
                if "normal" in surface_migration_metrics:
                    migrate_parts.append(f"normal={surface_migration_metrics['normal']:.6f}")
                    migrate_parts.append(f"normal_w={surface_migration_metrics['normal_w']:.6f}")
                if "normal_abs_mean" in surface_migration_metrics:
                    migrate_parts.append(f"|n|={surface_migration_metrics['normal_abs_mean']:.6f}")
                if "tangent" in surface_migration_metrics:
                    migrate_parts.append(f"tangent={surface_migration_metrics['tangent']:.6f}")
                    migrate_parts.append(f"tangent_w={surface_migration_metrics['tangent_w']:.6f}")
                if "align" in surface_migration_metrics:
                    migrate_parts.append(f"align={surface_migration_metrics['align']:.6f}")
                    migrate_parts.append(f"align_w={surface_migration_metrics['align_w']:.6f}")
                if "thin" in surface_migration_metrics:
                    migrate_parts.append(f"thin={surface_migration_metrics['thin']:.6f}")
                    migrate_parts.append(f"thin_w={surface_migration_metrics['thin_w']:.6f}")
                if "thin_ratio" in surface_migration_metrics:
                    migrate_parts.append(f"ratio={surface_migration_metrics['thin_ratio']:.6f}")
                print("[SURFACE-MIGRATE] " f"iter={iteration} " + " ".join(migrate_parts))
            if (
                sof_reg_metrics
                and iteration % max(1, hybrid_args.sof_reg_log_interval) == 0
            ):
                reg_parts = [f"total={sof_reg_metrics['total']:.6f}"]
                if "opacity" in sof_reg_metrics:
                    reg_parts.append(f"opa={sof_reg_metrics['opacity']:.6f}")
                if "scale" in sof_reg_metrics:
                    reg_parts.append(f"scale={sof_reg_metrics['scale']:.6f}")
                if "min_scale" in sof_reg_metrics:
                    reg_parts.append(f"min={sof_reg_metrics['min_scale']:.6f}")
                if "surface_thin" in sof_reg_metrics:
                    reg_parts.append(f"thin={sof_reg_metrics['surface_thin']:.6f}")
                if "distortion" in sof_reg_metrics:
                    reg_parts.append(f"dist={sof_reg_metrics['distortion']:.6f}")
                if "depth_normal" in sof_reg_metrics:
                    reg_parts.append(f"dn={sof_reg_metrics['depth_normal']:.6f}")
                if "smoothness" in sof_reg_metrics:
                    reg_parts.append(f"smooth={sof_reg_metrics['smoothness']:.6f}")
                if "extent" in sof_reg_metrics:
                    reg_parts.append(f"extent={sof_reg_metrics['extent']:.6f}")
                if "opacity_field" in sof_reg_metrics:
                    reg_parts.append(f"of={sof_reg_metrics['opacity_field']:.6f}")
                print("[SOF-REG] " f"iter={iteration} " + " ".join(reg_parts))
            if (
                sdf_densify_metrics
                and iteration % max(1, hybrid_args.sdf_densify_log_interval) == 0
            ):
                print(
                    "[SDF-DENSIFY] "
                    f"iter={iteration} "
                    f"weighted={sdf_densify_metrics['weighted']:.6f} "
                    f"proxy={sdf_densify_metrics['proxy']:.6f} "
                    f"score_mean={sdf_densify_metrics['score_mean']:.6f} "
                    f"score_max={sdf_densify_metrics['score_max']:.6f} "
                    f"selected={int(sdf_densify_metrics['selected_points'])} "
                    f"refresh={int(sdf_densify_metrics['refresh'])} "
                    f"sr_points={int(sdf_densify_metrics.get('sr_points', 0.0))} "
                    f"sr_score_mean={sdf_densify_metrics.get('sr_score_mean', 0.0):.6f} "
                    f"sr_valid={sdf_densify_metrics.get('sr_valid_ratio', 0.0):.3f} "
                    f"sr_weighted={sdf_densify_metrics.get('sr_weighted', 0.0):.6f}"
                )
            if proposal_metrics and iteration % max(1, hybrid_args.sdf_proposal_log_interval) == 0:
                print(
                    "[SR-PROPOSAL] "
                    f"iter={iteration} "
                    f"proposal2d={int(proposal_metrics.get('proposals_2d', 0.0))} "
                    f"hits3d={int(proposal_metrics.get('hits_3d', 0.0))} "
                    f"valid3d={int(proposal_metrics.get('valid_3d', 0.0))} "
                    f"fused={int(proposal_metrics.get('fused_points', 0.0))} "
                    f"nbr={int(proposal_metrics.get('neighbor_views', 0.0))} "
                    f"inlier={proposal_metrics.get('inlier_views_mean', 0.0):.2f} "
                    f"|delta|={proposal_metrics.get('delta_abs_mean', 0.0):.6f} "
                    f"|tan|={proposal_metrics.get('tangent_abs_mean', 0.0):.6f} "
                    f"realign={int(proposal_metrics.get('realigned_gs', 0.0))} "
                    f"shift={proposal_metrics.get('realign_shift_mean', 0.0):.6f} "
                    f"dd={proposal_metrics.get('realign_depth_delta_mean', 0.0):.6f} "
                    f"drot={proposal_metrics.get('realign_rot_mean', 0.0):.6f} "
                    f"sratio={proposal_metrics.get('realign_scale_ratio_mean', 1.0):.4f} "
                    f"birth={int(proposal_metrics.get('native_births', 0.0))} "
                    f"|sdf|={proposal_metrics.get('abs_sdf_mean', 0.0):.3e} "
                    f"support={proposal_metrics.get('support_mean', 0.0):.6f} "
                    f"support_th={proposal_metrics.get('support_thresh_eff', 0.0):.6f} "
                    f"align={proposal_metrics.get('align_mean', 0.0):.6f} "
                    f"sigma={proposal_metrics.get('nearest_sigma_mean', 0.0):.6f} "
                    f"align/sigma={proposal_metrics.get('align_sigma_mean', 0.0):.3f} "
                    f"pass[sdf={proposal_metrics.get('sdf_pass_rate', 0.0):.3f},"
                    f"sup={proposal_metrics.get('support_pass_rate', 0.0):.3f},"
                    f"ali={proposal_metrics.get('align_pass_rate', 0.0):.3f},"
                    f"all={proposal_metrics.get('valid_rate', 0.0):.3f}] "
                    f"trust={proposal_metrics.get('proposal_trust_state', 'full')}:"
                    f"{proposal_metrics.get('proposal_trust_scale', 1.0):.2f} "
                    f"refresh={int(proposal_metrics.get('teacher_refresh', 0.0))} "
                    f"center|sdf|[mean={proposal_metrics.get('center_abs_sdf_mean', 0.0):.3e},"
                    f"p90={proposal_metrics.get('center_abs_sdf_p90', 0.0):.3e},"
                    f"max={proposal_metrics.get('center_abs_sdf_max', 0.0):.3e}]"
                )
            if fm_metrics and iteration % max(1, hybrid_args.fm_sds_log_interval) == 0:
                print(
                    "[FM-SDS] "
                    f"iter={iteration} "
                    f"total={fm_metrics['total']:.6f} "
                    f"base={fm_metrics['base']:.6f} "
                    f"t={fm_metrics['t']:.4f} "
                    f"wt={fm_metrics['weight_t']:.4f} "
                    f"res={fm_metrics['residual']:.6f}"
                )
            if freq_metrics and iteration % max(1, hybrid_args.freq_log_interval) == 0:
                print(
                    "[FREQ] "
                    f"iter={iteration} "
                    f"total={freq_metrics['total']:.6f} "
                    f"prior={freq_metrics.get('prior_total', 0.0):.6f} "
                    f"gt={freq_metrics.get('gt_total', 0.0):.6f}"
                )
            if (
                scaffold_metrics
                and iteration % max(1, hybrid_args.scaffold_log_interval) == 0
            ):
                print(
                    "[SCAFFOLD] "
                    f"iter={iteration} "
                    f"total={scaffold_metrics['total']:.6f} "
                    f"chamfer={scaffold_metrics['chamfer']:.6f} "
                    f"normal={scaffold_metrics['normal']:.6f} "
                    f"gs={int(scaffold_metrics['selected_gs'])} "
                    f"scf={int(scaffold_metrics['selected_scaffold'])} "
                    f"has_n={int(scaffold_metrics['has_normals'])}"
                )


def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()

    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= width - 1
    id_coords[..., 1] /= height - 1
    id_coords = id_coords * 2 - 1

    image = torch.nn.functional.grid_sample(
        image[None],
        id_coords[None],
        align_corners=True,
        padding_mode="border",
    )[0]
    return image


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if getattr(args, "disable_tensorboard", False):
        print("Tensorboard logging disabled by flag")
    elif TENSORBOARD_FOUND:
        tb_writer = SafeSummaryWriter(SummaryWriter(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss_func,
    elapsed,
    testing_iterations,
    scene,
    render_func,
    render_args,
    hybrid_metrics=None,
    sdf_densify_metrics=None,
    proposal_metrics=None,
    prior_metrics=None,
    sof_reg_metrics=None,
    fm_metrics=None,
    freq_metrics=None,
    scaffold_metrics=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        if hybrid_metrics:
            tb_writer.add_scalar("hybrid/total", hybrid_metrics["total"], iteration)
            tb_writer.add_scalar("hybrid/surface", hybrid_metrics["surface"], iteration)
            tb_writer.add_scalar("hybrid/normal", hybrid_metrics["normal"], iteration)
            tb_writer.add_scalar("hybrid/offsurface", hybrid_metrics["offsurface"], iteration)
        if sdf_densify_metrics:
            tb_writer.add_scalar("sdf_densify/weighted", sdf_densify_metrics["weighted"], iteration)
            tb_writer.add_scalar("sdf_densify/proxy", sdf_densify_metrics["proxy"], iteration)
            tb_writer.add_scalar("sdf_densify/score_mean", sdf_densify_metrics["score_mean"], iteration)
            tb_writer.add_scalar("sdf_densify/selected_points", sdf_densify_metrics["selected_points"], iteration)
            tb_writer.add_scalar("sdf_densify/sr_points", sdf_densify_metrics.get("sr_points", 0.0), iteration)
            tb_writer.add_scalar("sdf_densify/sr_score_mean", sdf_densify_metrics.get("sr_score_mean", 0.0), iteration)
            tb_writer.add_scalar("sdf_densify/sr_valid_ratio", sdf_densify_metrics.get("sr_valid_ratio", 0.0), iteration)
            tb_writer.add_scalar("sdf_densify/sr_weighted", sdf_densify_metrics.get("sr_weighted", 0.0), iteration)
        if proposal_metrics:
            tb_writer.add_scalar("sr_proposal/proposals_2d", proposal_metrics.get("proposals_2d", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/hits_3d", proposal_metrics.get("hits_3d", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/valid_3d", proposal_metrics.get("valid_3d", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/abs_sdf_mean", proposal_metrics.get("abs_sdf_mean", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/support_mean", proposal_metrics.get("support_mean", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/support_thresh_eff", proposal_metrics.get("support_thresh_eff", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/align_mean", proposal_metrics.get("align_mean", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/nearest_sigma_mean", proposal_metrics.get("nearest_sigma_mean", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/align_sigma_mean", proposal_metrics.get("align_sigma_mean", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/sdf_pass_rate", proposal_metrics.get("sdf_pass_rate", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/support_pass_rate", proposal_metrics.get("support_pass_rate", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/align_pass_rate", proposal_metrics.get("align_pass_rate", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/valid_rate", proposal_metrics.get("valid_rate", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/center_abs_sdf_mean", proposal_metrics.get("center_abs_sdf_mean", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/center_abs_sdf_p90", proposal_metrics.get("center_abs_sdf_p90", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/center_abs_sdf_max", proposal_metrics.get("center_abs_sdf_max", 0.0), iteration)
            tb_writer.add_scalar("sr_proposal/proposal_trust_scale", proposal_metrics.get("proposal_trust_scale", 1.0), iteration)
            tb_writer.add_scalar("sr_proposal/teacher_refresh", proposal_metrics.get("teacher_refresh", 0.0), iteration)
        if prior_metrics:
            if "total" in prior_metrics:
                tb_writer.add_scalar("prior/total", prior_metrics["total"], iteration)
                tb_writer.add_scalar("prior/l1", prior_metrics["l1"], iteration)
                tb_writer.add_scalar("prior/hf", prior_metrics["hf"], iteration)
                tb_writer.add_scalar("prior/valid_ratio", prior_metrics["valid_ratio"], iteration)
            if "sequence_tex" in prior_metrics:
                tb_writer.add_scalar("sequence_loss/tex", prior_metrics["sequence_tex"], iteration)
                tb_writer.add_scalar("sequence_loss/subpixel", prior_metrics["sequence_sp"], iteration)
                tb_writer.add_scalar("sequence_loss/lambda_tex", prior_metrics["sequence_lambda_tex"], iteration)
            if "sof_local" in prior_metrics:
                tb_writer.add_scalar("prior/sof_local", prior_metrics["sof_local"], iteration)
            if "sof_edge" in prior_metrics:
                tb_writer.add_scalar("prior/sof_edge", prior_metrics["sof_edge"], iteration)
            if "sof_edge_alpha" in prior_metrics:
                tb_writer.add_scalar("prior/sof_edge_alpha", prior_metrics["sof_edge_alpha"], iteration)
            if "sof_edge_touch_ratio" in prior_metrics:
                tb_writer.add_scalar("prior/sof_edge_touch_ratio", prior_metrics["sof_edge_touch_ratio"], iteration)
            if "hf_seeded" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seeded", prior_metrics["hf_seeded"], iteration)
                tb_writer.add_scalar("prior/hf_seed_total", prior_metrics["hf_seed_total"], iteration)
            if "hf_seed_guidance_mean" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_guidance_mean", prior_metrics["hf_seed_guidance_mean"], iteration)
            if "hf_seed_residual_abs_mean" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_residual_abs_mean", prior_metrics["hf_seed_residual_abs_mean"], iteration)
            if "hf_seed_color_clip_ratio" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_color_clip_ratio", prior_metrics["hf_seed_color_clip_ratio"], iteration)
            if "hf_seed_render_depth_ratio" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_render_depth_ratio", prior_metrics["hf_seed_render_depth_ratio"], iteration)
            if "hf_seed_provider_dist_mean" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_provider_dist_mean", prior_metrics["hf_seed_provider_dist_mean"], iteration)
            for key in (
                "hf_seed_scale_mean",
                "hf_seed_scale_p90",
                "hf_seed_scale_max",
                "hf_seed_color_residual_gain",
                "hf_seed_pixel_scale_mean",
                "hf_seed_pixel_radius_mean",
                "hf_seed_pixel_radius_p90",
                "hf_seed_provider_scale_mean",
                "hf_seed_provider_ratio_mean",
                "hf_seed_provider_ratio_p90",
                "hf_seed_shape_anisotropy_mean",
                "hf_seed_shape_anisotropy_p90",
                "hf_seed_shape_normal_ratio",
                "hf_seed_shape_orient_conf_mean",
                "hf_seed_shape_orient_conf_p90",
                "hf_seed_shape_mode_id",
                "hf_recent_imprint",
                "hf_recent_imprint_count",
                "hf_recent_imprint_visible",
                "hf_lowfreq_cleanup_prior",
                "hf_lowfreq_cleanup_candidates",
                "hf_lowfreq_cleanup_ratio",
                "hf_lowfreq_cleanup_opacity_mean",
                "hf_lowfreq_cleanup_scale_p90",
                "hf_lowfreq_cleanup_age_mean",
                "hf_lowfreq_cleanup_decay",
                "hf_lowfreq_cleanup_opacity_after",
                "hf_lowfreq_cleanup_pruned",
                "hf_seed_unique_ids",
                "hf_seed_clone_ratio",
                "hf_seed_scale_geom_median",
                "hf_seed_scale_geom_p90",
                "hf_seed_scale_max_axis_median",
                "hf_seed_scale_max_axis_p90",
                "hf_seed_scale_max_axis_max",
                "hf_seed_generation_mean",
                "hf_seed_generation_max",
                "hf_seed_large_count",
                "hf_seed_large_ratio",
                "hf_seed_scale_clamp_count",
                "hf_seed_scale_clamp_ratio",
                "hf_seed_scale_clamp_max_before",
                "hf_seed_scale_clamp_max_after",
                "hf_seed_scale_clamp_anis_p90_before",
                "hf_seed_scale_clamp_anis_p90_after",
                "bubble_cleanup_prior",
                "bubble_cleanup_count",
                "bubble_cleanup_ratio",
                "bubble_cleanup_opacity_mean",
                "bubble_cleanup_opacity_p90",
                "bubble_cleanup_scale_max_p50",
                "bubble_cleanup_scale_max_p90",
                "bubble_cleanup_anis_p50",
                "bubble_cleanup_anis_p90",
                "bubble_cleanup_total_w",
                "bubble_cleanup_rgb",
                "bubble_cleanup_rgb_w",
                "bubble_cleanup_hf",
                "bubble_cleanup_hf_w",
                "bubble_cleanup_alpha_hf",
                "bubble_cleanup_alpha_hf_w",
                "bubble_cleanup_alpha",
                "bubble_cleanup_alpha_w",
                "bubble_cleanup_post_count",
                "bubble_cleanup_post_decay",
                "bubble_cleanup_post_opacity_before",
                "bubble_cleanup_post_opacity_after",
            ):
                if key in prior_metrics:
                    tb_writer.add_scalar(f"prior/{key}", prior_metrics[key], iteration)
            if "hf_seed_first_cycle_pending" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_first_cycle_pending", prior_metrics["hf_seed_first_cycle_pending"], iteration)
            if "hf_seed_live" in prior_metrics:
                tb_writer.add_scalar("prior/hf_seed_live", prior_metrics["hf_seed_live"], iteration)
            if "hf_focus_mean" in prior_metrics:
                tb_writer.add_scalar("prior/hf_focus_mean", prior_metrics["hf_focus_mean"], iteration)
            if "view_curriculum_primary_hf_energy" in prior_metrics:
                tb_writer.add_scalar("prior/view_curriculum_primary_hf_energy", prior_metrics["view_curriculum_primary_hf_energy"], iteration)
            if "view_curriculum_primary_band" in prior_metrics:
                tb_writer.add_scalar("prior/view_curriculum_primary_band", prior_metrics["view_curriculum_primary_band"], iteration)
            if "view_curriculum_primary_residual" in prior_metrics:
                tb_writer.add_scalar("prior/view_curriculum_primary_residual", prior_metrics["view_curriculum_primary_residual"], iteration)
            if "view_curriculum_primary_patch" in prior_metrics:
                tb_writer.add_scalar("prior/view_curriculum_primary_patch", prior_metrics["view_curriculum_primary_patch"], iteration)
            if "view_curriculum_primary_patch_lowfreq" in prior_metrics:
                tb_writer.add_scalar("prior/view_curriculum_primary_patch_lowfreq", prior_metrics["view_curriculum_primary_patch_lowfreq"], iteration)
            if "view_curriculum_primary_patch_coverage" in prior_metrics:
                tb_writer.add_scalar("prior/view_curriculum_primary_patch_coverage", prior_metrics["view_curriculum_primary_patch_coverage"], iteration)
            if "hf_recent_prior_live" in prior_metrics:
                tb_writer.add_scalar("prior/hf_recent_prior_live", prior_metrics["hf_recent_prior_live"], iteration)
            if "hf_recent_prior_visible" in prior_metrics:
                tb_writer.add_scalar("prior/hf_recent_prior_visible", prior_metrics["hf_recent_prior_visible"], iteration)
            if "hf_recent_prior_boost" in prior_metrics:
                tb_writer.add_scalar("prior/hf_recent_prior_boost", prior_metrics["hf_recent_prior_boost"], iteration)
        if sof_reg_metrics:
            tb_writer.add_scalar("sof_reg/total", sof_reg_metrics["total"], iteration)
            if "opacity" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/opacity", sof_reg_metrics["opacity"], iteration)
            if "scale" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/scale", sof_reg_metrics["scale"], iteration)
            if "min_scale" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/min_scale", sof_reg_metrics["min_scale"], iteration)
            if "surface_thin" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/surface_thin", sof_reg_metrics["surface_thin"], iteration)
            if "distortion" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/distortion", sof_reg_metrics["distortion"], iteration)
            if "depth_normal" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/depth_normal", sof_reg_metrics["depth_normal"], iteration)
            if "smoothness" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/smoothness", sof_reg_metrics["smoothness"], iteration)
            if "extent" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/extent", sof_reg_metrics["extent"], iteration)
            if "opacity_field" in sof_reg_metrics:
                tb_writer.add_scalar("sof_reg/opacity_field", sof_reg_metrics["opacity_field"], iteration)
        if fm_metrics:
            tb_writer.add_scalar("fm_sds/total", fm_metrics["total"], iteration)
            tb_writer.add_scalar("fm_sds/base", fm_metrics["base"], iteration)
            tb_writer.add_scalar("fm_sds/t", fm_metrics["t"], iteration)
            tb_writer.add_scalar("fm_sds/weight_t", fm_metrics["weight_t"], iteration)
        if freq_metrics:
            tb_writer.add_scalar("freq/total", freq_metrics["total"], iteration)
            tb_writer.add_scalar("freq/prior_total", freq_metrics.get("prior_total", 0.0), iteration)
            tb_writer.add_scalar("freq/gt_total", freq_metrics.get("gt_total", 0.0), iteration)
            tb_writer.add_scalar("freq/prior_low", freq_metrics.get("prior_low", 0.0), iteration)
            tb_writer.add_scalar("freq/prior_mid", freq_metrics.get("prior_mid", 0.0), iteration)
            tb_writer.add_scalar("freq/prior_high", freq_metrics.get("prior_high", 0.0), iteration)
            tb_writer.add_scalar("freq/gt_low", freq_metrics.get("gt_low", 0.0), iteration)
            tb_writer.add_scalar("freq/gt_mid", freq_metrics.get("gt_mid", 0.0), iteration)
            tb_writer.add_scalar("freq/gt_high", freq_metrics.get("gt_high", 0.0), iteration)
        if scaffold_metrics:
            tb_writer.add_scalar("scaffold/total", scaffold_metrics["total"], iteration)
            tb_writer.add_scalar("scaffold/chamfer", scaffold_metrics["chamfer"], iteration)
            tb_writer.add_scalar("scaffold/normal", scaffold_metrics["normal"], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        render_func(viewpoint, scene.gaussians, *render_args)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + f"_view_{viewpoint.image_name}/render",
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + f"_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss_func(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: "
                    f"L1 {l1_test} PSNR {psnr_test}"
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )
            tb_writer.add_scalar(
                "total_points",
                scene.gaussians.get_xyz.shape[0],
                iteration,
            )
        torch.cuda.empty_cache()


def make_parser():
    parser = ArgumentParser(description="Hybrid SDF + Gaussian Splatting training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--sequence_loss_enable", action="store_true")
    parser.add_argument("--sequence_lambda_tex", type=float, default=0.4)
    parser.add_argument("--sequence_subpixel", type=str, default="bicubic", choices=["bicubic", "avg"])
    parser.add_argument("--sequence_subpixel_scale", type=float, default=4.0)
    parser.add_argument("--sequence_lr_anchor_root", type=str, default="")
    parser.add_argument("--sequence_lr_anchor_subdir", type=str, default="images_8")
    parser.add_argument("--sequence_lr_anchor_exts", type=str, default="png,jpg,jpeg,webp")
    parser.add_argument("--sequence_loss_allow_missing_anchor", action="store_true")

    parser.add_argument("--hybrid_enable", action="store_true")
    parser.add_argument(
        "--sdf_mode",
        type=str,
        default="analytic",
        choices=["none", "analytic", "neuralangelo", "gs_bootstrap", "surfel_mlp"],
    )
    parser.add_argument("--sdf_center", type=str, default="0,0,0")
    parser.add_argument("--sdf_radius", type=float, default=1.0)
    parser.add_argument(
        "--neuralangelo_root",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "neuralangelo")
        ),
    )
    parser.add_argument("--sdf_config", type=str, default="")
    parser.add_argument("--sdf_checkpoint", type=str, default="")

    parser.add_argument("--hybrid_stage_a_end", type=int, default=5_000)
    parser.add_argument("--hybrid_stage_b_end", type=int, default=20_000)
    parser.add_argument("--lambda_surface", type=float, default=0.05)
    parser.add_argument("--lambda_normal", type=float, default=0.01)
    parser.add_argument("--lambda_offsurface", type=float, default=0.01)
    parser.add_argument("--surface_epsilon", type=float, default=0.0)
    parser.add_argument("--offsurface_margin", type=float, default=0.05)
    parser.add_argument(
        "--normal_axis",
        type=str,
        default="min_scale",
        choices=["min_scale", "max_scale", "x", "y", "z"],
    )
    parser.add_argument("--hybrid_points_per_iter", type=int, default=4096)
    parser.add_argument("--hybrid_log_interval", type=int, default=100)
    parser.add_argument("--sdf_densify_enable", action="store_true")
    parser.add_argument("--sdf_densify_interval", type=int, default=500)
    parser.add_argument("--sdf_densify_topk", type=int, default=256)
    parser.add_argument("--sdf_densify_min_score", type=float, default=0.05)
    parser.add_argument("--sdf_densify_surface_coef", type=float, default=1.0)
    parser.add_argument("--sdf_densify_normal_coef", type=float, default=0.5)
    parser.add_argument("--sdf_densify_offsurface_coef", type=float, default=0.5)
    parser.add_argument("--sdf_densify_loss_weight", type=float, default=0.0)
    parser.add_argument("--sdf_densify_sr_weight", type=float, default=0.0)
    parser.add_argument("--sdf_densify_sr_levels", type=int, default=2)
    parser.add_argument("--sdf_densify_sr_samples_per_point", type=int, default=2)
    parser.add_argument("--sdf_densify_sr_jitter_scale", type=float, default=0.6)
    parser.add_argument("--sdf_densify_sr_max_points", type=int, default=4096)
    parser.add_argument("--sdf_densify_sr_score_coef", type=float, default=0.5)
    parser.add_argument("--sdf_densify_log_interval", type=int, default=100)

    parser.add_argument("--sdf_proposal_enable", action="store_true")
    parser.add_argument("--sdf_proposal_topk", type=int, default=256)
    parser.add_argument("--sdf_proposal_min_hf", type=float, default=0.02)
    parser.add_argument("--sdf_proposal_ray_near", type=float, default=0.05)
    parser.add_argument("--sdf_proposal_ray_far", type=float, default=6.0)
    parser.add_argument("--sdf_proposal_ray_samples", type=int, default=64)
    parser.add_argument("--sdf_proposal_surface_thresh", type=float, default=0.03)
    parser.add_argument("--sdf_proposal_gs_support_thresh", type=float, default=0.04)
    parser.add_argument("--sdf_proposal_gs_support_floor", type=float, default=0.002)
    parser.add_argument("--sdf_proposal_gs_support_quantile", type=float, default=0.85)
    parser.add_argument("--sdf_proposal_align_thresh", type=float, default=0.06)
    parser.add_argument("--sdf_proposal_align_sigma_thresh", type=float, default=3.0)
    parser.add_argument("--sdf_proposal_gs_knn_k", type=int, default=8)
    parser.add_argument("--sdf_proposal_plane_samples", type=int, default=4)
    parser.add_argument("--sdf_proposal_tangent_scale", type=float, default=0.8)
    parser.add_argument("--sdf_proposal_normal_scale", type=float, default=0.25)
    parser.add_argument("--sdf_proposal_full_stack_enable", action="store_true")
    parser.add_argument("--sdf_proposal_fuse_max_views", type=int, default=4)
    parser.add_argument("--sdf_proposal_fuse_min_views", type=int, default=2)
    parser.add_argument("--sdf_proposal_view_min_hf", type=float, default=0.01)
    parser.add_argument("--sdf_proposal_delta_steps", type=int, default=5)
    parser.add_argument("--sdf_proposal_delta_scale", type=float, default=0.75)
    parser.add_argument("--sdf_proposal_delta_floor", type=float, default=0.005)
    parser.add_argument("--sdf_proposal_tangent_steps", type=int, default=3)
    parser.add_argument("--sdf_proposal_tangent_search_scale", type=float, default=0.35)
    parser.add_argument("--sdf_proposal_feature_hf_weight", type=float, default=1.0)
    parser.add_argument("--sdf_proposal_feature_dir_weight", type=float, default=0.5)
    parser.add_argument("--sdf_proposal_feature_desc_weight", type=float, default=0.35)
    parser.add_argument("--sdf_proposal_sdf_cost_weight", type=float, default=0.25)
    parser.add_argument("--sdf_proposal_delta_reg_weight", type=float, default=0.05)
    parser.add_argument("--sdf_proposal_tangent_reg_weight", type=float, default=0.03)
    parser.add_argument("--sdf_proposal_newton_steps", type=int, default=1)
    parser.add_argument("--sdf_proposal_realign_enable", action="store_true")
    parser.add_argument("--sdf_proposal_realign_start_iter", type=int, default=1000)
    parser.add_argument("--sdf_proposal_realign_interval", type=int, default=10)
    parser.add_argument("--sdf_proposal_realign_ema", type=float, default=0.2)
    parser.add_argument("--sdf_proposal_realign_rotation_ema", type=float, default=0.15)
    parser.add_argument("--sdf_proposal_realign_scale_ema", type=float, default=0.15)
    parser.add_argument("--sdf_proposal_realign_max_shift", type=float, default=0.05)
    parser.add_argument("--sdf_proposal_realign_max_ratio", type=float, default=1.25)
    parser.add_argument("--sdf_proposal_realign_min_depth", type=float, default=1e-3)
    parser.add_argument("--sdf_proposal_native_birth_start_iter", type=int, default=1500)
    parser.add_argument("--sdf_proposal_native_birth_interval", type=int, default=200)
    parser.add_argument("--sdf_proposal_native_birth_max_points", type=int, default=64)
    parser.add_argument("--sdf_proposal_prior_hf_prior_weight", type=float, default=0.25)
    parser.add_argument("--sdf_proposal_neighbor_guidance_weight", type=float, default=1.0)
    parser.add_argument("--sdf_proposal_neighbor_visibility_weight", type=float, default=0.5)
    parser.add_argument("--sdf_proposal_neighbor_distance_weight", type=float, default=0.1)
    parser.add_argument("--sdf_proposal_log_interval", type=int, default=100)
    parser.add_argument("--sdf_proposal_diag_max_points", type=int, default=4096)
    parser.add_argument("--sdf_proposal_teacher_p90_soft", type=float, default=0.02)
    parser.add_argument("--sdf_proposal_teacher_p90_hard", type=float, default=0.04)
    parser.add_argument("--sdf_proposal_teacher_soft_scale", type=float, default=0.5)
    parser.add_argument("--transforms_llffhold", type=int, default=8)
    parser.add_argument("--disable_gui", action="store_true")
    parser.add_argument("--gs_bootstrap_update_interval", type=int, default=500)
    parser.add_argument("--gs_bootstrap_adaptive_refresh_enable", action="store_true")
    parser.add_argument("--gs_bootstrap_refresh_interval_a", type=int, default=50)
    parser.add_argument("--gs_bootstrap_refresh_interval_b", type=int, default=100)
    parser.add_argument("--gs_bootstrap_refresh_interval_c", type=int, default=200)
    parser.add_argument("--gs_bootstrap_max_anchors", type=int, default=20000)
    parser.add_argument("--gs_bootstrap_k_neighbors", type=int, default=8)
    parser.add_argument("--gs_bootstrap_opacity_min", type=float, default=0.01)
    parser.add_argument(
        "--gs_bootstrap_normal_axis",
        type=str,
        default="min_scale",
        choices=["min_scale", "max_scale", "x", "y", "z"],
    )
    parser.add_argument(
        "--gs_bootstrap_sigma_mode",
        type=str,
        default="geom_mid_min",
        choices=["min", "mid", "max", "mean", "geom_mid_min"],
    )
    parser.add_argument("--gs_bootstrap_distance_scale", type=float, default=1.0)
    parser.add_argument("--gs_bootstrap_distance_floor", type=float, default=1e-4)
    parser.add_argument("--gs_bootstrap_knn_chunk_size", type=int, default=4096)
    parser.add_argument(
        "--gs_bootstrap_subsample_mode",
        type=str,
        default="sharpness",
        choices=["random", "sharpness"],
    )
    parser.add_argument(
        "--gs_bootstrap_orient_normals_mode",
        type=str,
        default="centroid",
        choices=["none", "centroid"],
    )
    parser.add_argument("--gs_bootstrap_proxy_tangent_reg", type=float, default=0.15)
    parser.add_argument("--gs_bootstrap_proxy_max_normal_shift_scale", type=float, default=1.0)
    parser.add_argument("--gs_bootstrap_proxy_max_tangent_shift_scale", type=float, default=0.5)
    parser.add_argument("--gs_bootstrap_normal_support_scale", type=float, default=2.0)
    parser.add_argument("--gs_bootstrap_tangent_support_scale", type=float, default=1.25)
    parser.add_argument("--gs_bootstrap_invalid_penalty_scale", type=float, default=1.5)

    parser.add_argument("--external_prior_root", type=str, default="")
    parser.add_argument("--external_prior_subdir", type=str, default="priors")
    parser.add_argument("--external_prior_mask_subdir", type=str, default="")
    parser.add_argument("--external_prior_exts", type=str, default="png,jpg,jpeg,webp")
    parser.add_argument("--external_prior_use_dataset_root", action="store_true")
    parser.add_argument("--prior_loss_mode", type=str, default="rgb_hf", choices=["rgb_hf", "masked_residual_hf_v1"])
    parser.add_argument("--prior_anchor_dir", type=str, default="")
    parser.add_argument("--prior_soft_mask_dir", type=str, default="")
    parser.add_argument("--prior_hard_mask_dir", type=str, default="")
    parser.add_argument("--prior_soft_mask_power", type=float, default=1.0)
    parser.add_argument("--prior_hard_mask_threshold", type=float, default=0.5)
    parser.add_argument("--prior_masked_min_pixels", type=float, default=64.0)
    parser.add_argument("--prior_view_sample_prob", type=float, default=0.0)
    parser.add_argument("--prior_direct_xyz_nudge_lr", type=float, default=0.0)
    parser.add_argument("--prior_direct_xyz_nudge_max_step", type=float, default=0.0)
    parser.add_argument("--prior_residual_masked_update", action="store_true")
    parser.add_argument("--prior_masked_update_densify_signal", action="store_true")
    parser.add_argument("--prior_masked_update_densify_signal_scale", type=float, default=1.0)
    parser.add_argument("--prior_l1_weight", type=float, default=0.05)
    parser.add_argument("--prior_hf_weight", type=float, default=0.1)
    parser.add_argument("--disable_prior_hf_residual", action="store_true")
    parser.add_argument("--prior_delta_clip", type=float, default=0.15)
    parser.add_argument("--prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--prior_consistency_threshold", type=float, default=0.20)
    parser.add_argument("--prior_min_valid_ratio", type=float, default=0.30)
    parser.add_argument("--prior_log_interval", type=int, default=100)
    parser.add_argument("--prior_supervision_images_subdir", type=str, default="")
    parser.add_argument("--prior_supervision_resolution", type=int, default=1)
    parser.add_argument("--prior_supervision_cache_size", type=int, default=4)
    parser.add_argument("--prior_local_dir", type=str, default="")
    parser.add_argument("--prior_local_mask_dir", type=str, default="")
    parser.add_argument("--lambda_prior_local", type=float, default=0.0)
    parser.add_argument("--prior_local_min_pixels", type=float, default=64.0)
    parser.add_argument("--prior_local_from_iter", type=int, default=0)
    parser.add_argument("--prior_edge_dir", type=str, default="")
    parser.add_argument("--prior_edge_mask_dir", type=str, default="")
    parser.add_argument("--lambda_prior_edge", type=float, default=0.0)
    parser.add_argument("--prior_edge_loss_mode", type=str, default="rgb", choices=["rgb", "detail_v1"])
    parser.add_argument("--prior_edge_blend_alpha", type=float, default=1.0)
    parser.add_argument("--prior_edge_min_pixels", type=float, default=64.0)
    parser.add_argument("--prior_edge_from_iter", type=int, default=0)
    parser.add_argument("--prior_edge_touch_min_radius_px", type=float, default=2.0)
    parser.add_argument("--prior_edge_touch_radius_scale", type=float, default=0.5)
    parser.add_argument("--prior_edge_touch_max_radius_px", type=float, default=16.0)
    parser.add_argument("--prior_edge_detail_blur_kernel", type=int, default=9)
    parser.add_argument("--prior_edge_detail_alpha", type=float, default=0.6)
    parser.add_argument("--prior_edge_detail_alpha_final", type=float, default=-1.0)
    parser.add_argument("--prior_edge_detail_warmup_iters", type=int, default=0)
    parser.add_argument("--prior_edge_detail_weight", type=float, default=1.0)
    parser.add_argument("--prior_edge_lowfreq_weight", type=float, default=0.05)
    parser.add_argument("--prior_edge_grad_weight", type=float, default=0.0)
    parser.add_argument("--prior_edge_lowfreq_threshold", type=float, default=0.08)
    parser.add_argument("--prior_edge_lowfreq_anchor", type=str, default="render", choices=["render", "gt"])
    parser.add_argument("--prior_edge_detail_min_gain", type=float, default=0.0)
    parser.add_argument("--prior_edge_confidence_power", type=float, default=1.0)
    parser.add_argument("--prior_edge_update_scale", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_enable", action="store_true")
    parser.add_argument("--prior_hf_seed_from_iter", type=int, default=0)
    parser.add_argument("--prior_hf_seed_until_iter", type=int, default=0)
    parser.add_argument("--prior_hf_seed_interval", type=int, default=100)
    parser.add_argument("--prior_hf_seed_max_per_iter", type=int, default=512)
    parser.add_argument("--prior_hf_seed_max_total", type=int, default=8192)
    parser.add_argument("--prior_hf_seed_scale_multiplier", type=float, default=0.35)
    parser.add_argument("--prior_hf_seed_scale_mode", type=str, default="legacy_max", choices=["legacy_max", "pixel", "pixel_footprint", "screen", "min", "min_provider_pixel", "provider", "parent"])
    parser.add_argument("--prior_hf_seed_min_pixel_radius", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_max_pixel_radius", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_max_provider_ratio", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_shape_mode", type=str, default="isotropic", choices=["isotropic", "view_flat", "hf_oriented"])
    parser.add_argument("--prior_hf_seed_shape_long_ratio", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_shape_short_ratio", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_shape_normal_ratio", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_shape_confidence_power", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_diag_large_scale", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_scale_clamp_enable", action="store_true")
    parser.add_argument("--prior_hf_seed_scale_clamp_max_axis", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_scale_clamp_min_axis", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_scale_clamp_max_anisotropy", type=float, default=0.0)
    parser.add_argument("--prior_bubble_cleanup_enable", action="store_true")
    parser.add_argument("--prior_bubble_cleanup_from_iter", type=int, default=0)
    parser.add_argument("--prior_bubble_cleanup_until_iter", type=int, default=0)
    parser.add_argument("--prior_bubble_cleanup_interval", type=int, default=1)
    parser.add_argument("--prior_bubble_cleanup_opacity_min", type=float, default=0.10)
    parser.add_argument("--prior_bubble_cleanup_max_axis_min", type=float, default=0.0)
    parser.add_argument("--prior_bubble_cleanup_max_axis_max", type=float, default=0.0)
    parser.add_argument("--prior_bubble_cleanup_anisotropy_max", type=float, default=2.5)
    parser.add_argument("--prior_bubble_cleanup_min_generation", type=int, default=0)
    parser.add_argument("--lambda_prior_bubble_hf", type=float, default=0.0)
    parser.add_argument("--lambda_prior_bubble_rgb_energy", type=float, default=0.0)
    parser.add_argument("--lambda_prior_bubble_alpha_hf", type=float, default=0.0)
    parser.add_argument("--lambda_prior_bubble_alpha_mass", type=float, default=0.0)
    parser.add_argument("--prior_bubble_cleanup_post_opacity_decay", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_opacity", type=float, default=0.02)
    parser.add_argument("--prior_hf_seed_jitter_scale", type=float, default=0.15)
    parser.add_argument("--prior_hf_seed_color_residual_gain", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_guidance_threshold", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_first_cycle_only", action="store_true")
    parser.add_argument("--prior_hf_seed_original_only", action="store_true")
    parser.add_argument("--prior_hf_seed_prune_protect_iters", type=int, default=0)
    parser.add_argument("--prior_hf_seed_recent_hf_boost", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_recent_imprint_weight", type=float, default=0.0)
    parser.add_argument("--prior_hf_seed_recent_imprint_gain", type=float, default=1.0)
    parser.add_argument("--prior_hf_seed_recent_imprint_iters", type=int, default=0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_enable", action="store_true")
    parser.add_argument("--prior_hf_lowfreq_cleanup_from_iter", type=int, default=0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_until_iter", type=int, default=0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_interval", type=int, default=20)
    parser.add_argument("--prior_hf_lowfreq_cleanup_stale_iters", type=int, default=240)
    parser.add_argument("--prior_hf_lowfreq_cleanup_opacity_min", type=float, default=0.06)
    parser.add_argument("--prior_hf_lowfreq_cleanup_scale_max_min", type=float, default=0.0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_scale_max_max", type=float, default=0.0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_anisotropy_max", type=float, default=0.0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_opacity_decay", type=float, default=1.0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_prune_opacity_below", type=float, default=0.0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_max_prune_fraction", type=float, default=0.0)
    parser.add_argument("--prior_hf_lowfreq_cleanup_max_prune_count", type=int, default=0)
    parser.add_argument("--prior_hf_focus_boost", type=float, default=0.0)
    parser.add_argument("--prior_hf_focus_power", type=float, default=1.0)
    parser.add_argument("--prior_guidance_decay_from_iter", type=int, default=0)
    parser.add_argument("--prior_guidance_decay_until_iter", type=int, default=0)
    parser.add_argument("--prior_guidance_final_scale", type=float, default=1.0)
    parser.add_argument("--prior_view_curriculum_enable", action="store_true")
    parser.add_argument("--prior_view_curriculum_start_iter", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_primary_iters", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_neighbor_iters", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_neighbor_radius", type=int, default=2)
    parser.add_argument("--prior_view_curriculum_max_views", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_settle_iters", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_birth_primary_only", action="store_true")
    parser.add_argument("--prior_view_curriculum_primary_hf_scale", type=float, default=1.0)
    parser.add_argument("--prior_view_curriculum_neighbor_hf_scale", type=float, default=1.0)
    parser.add_argument("--prior_view_curriculum_settle_hf_scale", type=float, default=1.0)
    parser.add_argument("--prior_view_curriculum_primary_bootstrap_iters", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_primary_hf_energy_weight", type=float, default=0.0)
    parser.add_argument("--prior_view_curriculum_primary_hf_energy_gain", type=float, default=1.0)
    parser.add_argument("--prior_view_curriculum_primary_band_weight", type=float, default=0.0)
    parser.add_argument("--prior_view_curriculum_primary_band_kernel", type=int, default=7)
    parser.add_argument("--prior_view_curriculum_primary_residual_weight", type=float, default=0.0)
    parser.add_argument("--prior_view_curriculum_primary_patch_weight", type=float, default=0.0)
    parser.add_argument("--prior_view_curriculum_primary_patch_top_fraction", type=float, default=0.15)
    parser.add_argument("--prior_view_curriculum_primary_patch_min_guidance", type=float, default=0.08)
    parser.add_argument("--prior_view_curriculum_primary_patch_delta_gain", type=float, default=1.0)
    parser.add_argument("--prior_view_curriculum_primary_patch_highpass_kernel", type=int, default=0)
    parser.add_argument("--prior_view_curriculum_primary_patch_lowfreq_guard_weight", type=float, default=0.0)
    parser.add_argument("--prior_view_curriculum_primary_patch_lowfreq_guard_kernel", type=int, default=15)
    parser.add_argument("--surface_route_consensus_root", type=str, default="")
    parser.add_argument("--lambda_surface_route_consensus", type=float, default=0.0)
    parser.add_argument("--surface_route_consensus_from_iter", type=int, default=0)
    parser.add_argument("--surface_route_consensus_min_pixels", type=float, default=64.0)
    parser.add_argument("--surface_route_view_sample_prob", type=float, default=0.0)
    parser.add_argument("--surface_route_surface_only", action="store_true")
    parser.add_argument("--optimize_source_tag", choices=["all", "prior", "probe", "added"], default="all")
    parser.add_argument("--optimize_gaussian_mask_payload", type=str, default=None)
    parser.add_argument("--optimize_gaussian_mask_key", type=str, default="selected_mask")
    parser.add_argument("--prior_loss_gaussian_mask_payload", type=str, default=None)
    parser.add_argument("--prior_loss_gaussian_mask_key", type=str, default="selected_mask")
    parser.add_argument("--prior_loss_gaussian_mask_dynamic_roots", action="store_true")
    parser.add_argument("--surface_normal_lock", action="store_true")
    parser.add_argument("--surface_normal_lock_payload", type=str, default="")
    parser.add_argument("--surface_normal_lock_normal_key", type=str, default="anchor_normal")
    parser.add_argument("--surface_normal_lock_anchor_key", type=str, default="anchor_xyz")
    parser.add_argument("--surface_normal_lock_dynamic_roots", action="store_true")
    parser.add_argument("--surface_filter_3d", action="store_true")
    parser.add_argument("--surface_filter_3d_payload", type=str, default="")
    parser.add_argument("--surface_filter_3d_key", type=str, default="surface_carrier")
    parser.add_argument("--surface_filter_3d_scale", type=float, default=1.4142135623730951)
    parser.add_argument("--surface_filter_3d_min", type=float, default=0.0)
    parser.add_argument("--layer_frequency_mask_payload", type=str, default="")
    parser.add_argument("--layer_frequency_non_surface_key", type=str, default="non_surface_active")
    parser.add_argument("--layer_frequency_surface_key", type=str, default="surface_carrier")
    parser.add_argument("--layer_frequency_dynamic_roots", action="store_true")
    parser.add_argument("--layer_frequency_surface_target", type=str, default="gt")
    parser.add_argument("--lambda_non_surface_hf", type=float, default=0.0)
    parser.add_argument("--lambda_non_surface_rgb_energy", type=float, default=0.0)
    parser.add_argument("--lambda_non_surface_alpha_hf", type=float, default=0.0)
    parser.add_argument("--lambda_non_surface_alpha_mass", type=float, default=0.0)
    parser.add_argument("--lambda_surface_hf_closure", type=float, default=0.0)
    parser.add_argument("--lambda_surface_start_hf_preserve", type=float, default=0.0)
    parser.add_argument("--layer_frequency_start_hf_checkpoint", type=str, default="")
    parser.add_argument("--layer_frequency_start_hf_lowfreq_kernel", type=int, default=15)
    parser.add_argument("--layer_frequency_start_hf_lowfreq_threshold", type=float, default=0.05)
    parser.add_argument("--layer_frequency_start_hf_energy_threshold", type=float, default=0.01)
    parser.add_argument("--layer_frequency_start_hf_mask_power", type=float, default=1.0)
    parser.add_argument("--layer_frequency_start_hf_protect_non_surface", action="store_true")
    parser.add_argument("--surface_hf_update_scale", type=float, default=1.0)
    parser.add_argument("--layer_frequency_from_iter", type=int, default=0)
    parser.add_argument("--layer_frequency_until_iter", type=int, default=0)
    parser.add_argument("--layer_frequency_log_interval", type=int, default=100)
    parser.add_argument("--surface_migration_payload", type=str, default="")
    parser.add_argument("--surface_migration_mask_key", type=str, default="near_surface_uncertain")
    parser.add_argument("--surface_migration_anchor_source_key", type=str, default="surface_carrier")
    parser.add_argument("--surface_migration_normal_key", type=str, default="anchor_normal")
    parser.add_argument("--surface_migration_anchor_key", type=str, default="anchor_xyz")
    parser.add_argument("--surface_migration_target_normal_coord", type=float, default=0.0)
    parser.add_argument("--lambda_surface_migration_normal", type=float, default=0.0)
    parser.add_argument("--lambda_surface_migration_tangent", type=float, default=0.0)
    parser.add_argument("--lambda_surface_migration_normal_align", type=float, default=0.0)
    parser.add_argument("--lambda_surface_migration_thin", type=float, default=0.0)
    parser.add_argument("--surface_migration_thin_target_ratio", type=float, default=0.25)
    parser.add_argument("--surface_migration_post_step_normal_alpha", type=float, default=0.0)
    parser.add_argument("--surface_migration_post_step_tangent_alpha", type=float, default=0.0)
    parser.add_argument("--surface_migration_from_iter", type=int, default=0)
    parser.add_argument("--surface_migration_until_iter", type=int, default=0)
    parser.add_argument("--surface_migration_log_interval", type=int, default=100)
    parser.add_argument("--prior_only_edge_finetune", action="store_true")
    parser.add_argument("--opacity_reg", type=float, default=0.0)
    parser.add_argument("--scale_reg", type=float, default=0.0)
    parser.add_argument("--min_scale_reg", type=float, default=0.0)
    parser.add_argument("--lambda_distortion", type=float, default=0.0)
    parser.add_argument("--lambda_depth_normal", type=float, default=0.0)
    parser.add_argument("--lambda_smoothness", type=float, default=0.0)
    parser.add_argument("--lambda_extent", type=float, default=0.0)
    parser.add_argument("--lambda_opacity_field", type=float, default=0.0)
    parser.add_argument("--distortion_from_iter", type=int, default=0)
    parser.add_argument("--depth_normal_from_iter", type=int, default=0)
    parser.add_argument("--sof_reg_log_interval", type=int, default=100)
    parser.add_argument("--surface_thin_mesh_path", type=str, default="")
    parser.add_argument("--lambda_surface_thin", type=float, default=0.0)
    parser.add_argument("--surface_thin_from_iter", type=int, default=0)
    parser.add_argument("--surface_thin_until_iter", type=int, default=0)
    parser.add_argument("--surface_thin_sample_count", type=int, default=500000)
    parser.add_argument("--surface_thin_update_interval", type=int, default=500)
    parser.add_argument("--surface_thin_gaussian_sample_count", type=int, default=65536)
    parser.add_argument("--surface_thin_offset_margin", type=float, default=0.02)
    parser.add_argument("--surface_thin_normal_scale_target", type=float, default=0.0)
    parser.add_argument("--surface_thin_normal_scale_weight", type=float, default=1.0)

    parser.add_argument("--fm_sds_enable", action="store_true")
    parser.add_argument("--fm_sds_weight", type=float, default=0.0)
    parser.add_argument("--fm_sds_t_min", type=float, default=0.02)
    parser.add_argument("--fm_sds_t_max", type=float, default=0.98)
    parser.add_argument("--fm_sds_gamma", type=float, default=1.0)
    parser.add_argument("--fm_sds_huber_delta", type=float, default=0.03)
    parser.add_argument("--fm_sds_log_interval", type=int, default=100)

    parser.add_argument("--freq_loss_enable", action="store_true")
    parser.add_argument("--freq_loss_weight", type=float, default=0.0)
    parser.add_argument("--freq_prior_weight", type=float, default=1.0)
    parser.add_argument("--freq_gt_weight", type=float, default=0.0)
    parser.add_argument("--freq_low_cutoff", type=float, default=0.10)
    parser.add_argument("--freq_high_cutoff", type=float, default=0.35)
    parser.add_argument("--freq_low_weight", type=float, default=0.5)
    parser.add_argument("--freq_mid_weight", type=float, default=0.7)
    parser.add_argument("--freq_high_weight", type=float, default=1.0)
    parser.add_argument("--freq_method", type=str, default="fft", choices=["fft", "haar"])
    parser.add_argument("--freq_log_interval", type=int, default=100)

    parser.add_argument("--scaffold_enable", action="store_true")
    parser.add_argument("--scaffold_path", type=str, default="")
    parser.add_argument("--scaffold_sample_size", type=int, default=2048)
    parser.add_argument("--scaffold_interval", type=int, default=1)
    parser.add_argument(
        "--scaffold_axis",
        type=str,
        default="min_scale",
        choices=["min_scale", "max_scale", "x", "y", "z"],
    )
    parser.add_argument("--scaffold_chamfer_weight", type=float, default=0.0)
    parser.add_argument("--scaffold_normal_weight", type=float, default=0.0)
    parser.add_argument("--scaffold_log_interval", type=int, default=100)

    return parser, lp, op, pp


def main():
    parser, lp, op, pp = make_parser()
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    if not args.disable_gui:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
