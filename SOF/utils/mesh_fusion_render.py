from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from utils.prior_fusion import build_mesh_depth_edge_mask, build_mesh_visibility, erode_binary_mask


def load_mesh_fusion_payload(path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    data = np.load(path)
    payload = {}
    required_keys = [
        "centers",
        "normals",
        "tangent_u",
        "tangent_v",
        "scale_u",
        "scale_v",
        "fused_rgb",
        "confidence",
        "disagreement",
        "view_count",
        "valid_mask",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing '{key}' in mesh fusion payload: {path}")
        array = data[key]
        if key == "valid_mask":
            payload[key] = torch.as_tensor(array, device=device, dtype=torch.bool)
        elif key == "view_count":
            payload[key] = torch.as_tensor(array, device=device, dtype=torch.int32)
        else:
            payload[key] = torch.as_tensor(array, device=device, dtype=torch.float32)
    if "scale_n" in data:
        payload["scale_n"] = torch.as_tensor(data["scale_n"], device=device, dtype=torch.float32)
    else:
        payload["scale_n"] = torch.minimum(payload["scale_u"], payload["scale_v"]) * 0.05
    return payload


def load_fusion_region_mask(mask_dir: Optional[str], image_name: str, height: int, width: int) -> Optional[torch.Tensor]:
    if not mask_dir:
        return None
    path = Path(mask_dir) / f"{image_name}_inject.png"
    if not path.is_file():
        return None
    mask = Image.open(path).convert("L")
    if mask.size != (width, height):
        resampling = getattr(Image, "Resampling", Image)
        mask = mask.resize((width, height), resampling.NEAREST)
    array = np.asarray(mask, dtype=np.uint8) > 127
    return torch.from_numpy(array).to(device="cuda", dtype=torch.bool)


def build_runtime_mesh_core_mask(
    mesh: Optional[trimesh.Trimesh],
    view,
    height: int,
    width: int,
    erode_kernel: int,
    depth_min: float,
) -> Optional[torch.Tensor]:
    if mesh is None:
        return None
    visibility = build_mesh_visibility(mesh, view, depth_min=depth_min, front_facing_only=True)
    visible_mask = visibility["visible_mask"]
    depth_edge_mask = build_mesh_depth_edge_mask(visibility["depth"], visible_mask)
    if visible_mask.shape[0] != height or visible_mask.shape[1] != width:
        visible_mask = F.interpolate(
            visible_mask.float()[None, None],
            size=(height, width),
            mode="nearest",
        )[0, 0] > 0.5
        depth_edge_mask = F.interpolate(
            depth_edge_mask.float()[None, None],
            size=(height, width),
            mode="nearest",
        )[0, 0] > 0.5
    return erode_binary_mask(visible_mask, int(erode_kernel)) & (~depth_edge_mask)


def lowpass_chw(image_chw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return image_chw
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = F.pad(image_chw[None], (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0]


def project_payload_points(view, points: torch.Tensor, depth_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
    R = torch.as_tensor(view.R, device=points.device, dtype=points.dtype)
    T = torch.as_tensor(view.T, device=points.device, dtype=points.dtype)
    xyz_cam = points @ R + T.unsqueeze(0)
    z = xyz_cam[:, 2]
    z_safe = torch.clamp_min(z, 1e-6)
    x = xyz_cam[:, 0] / z_safe * float(view.focal_x) + float(view.image_width) / 2.0
    y = xyz_cam[:, 1] / z_safe * float(view.focal_y) + float(view.image_height) / 2.0
    projected = torch.stack((x, y, z), dim=1)
    valid = z > float(depth_min)
    return projected, valid


def _disagreement_gate(disagreement: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if high <= low:
        return (disagreement <= low).to(dtype=torch.float32)
    gate = 1.0 - (disagreement - float(low)) / float(high - low)
    return torch.clamp(gate, 0.0, 1.0)


def _scatter_min_depth(flat_size: int, pixel_ids: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    min_depth = torch.full((flat_size,), float("inf"), device=depth.device, dtype=depth.dtype)
    if pixel_ids.numel() == 0:
        return min_depth
    try:
        min_depth.scatter_reduce_(0, pixel_ids, depth, reduce="amin", include_self=True)
    except AttributeError:
        order = torch.argsort(depth, descending=True)
        min_depth[pixel_ids[order]] = depth[order]
    return min_depth


def render_bounded_carrier_layer(
    view,
    payload: Dict[str, torch.Tensor],
    height: int,
    width: int,
    region_mask: Optional[torch.Tensor],
    min_confidence: float,
    disagreement_low: float,
    disagreement_high: float,
    min_radius_px: float,
    max_radius_px: float,
    radius_scale: float,
    depth_min: float,
    z_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = payload["centers"].device
    centers = payload["centers"]
    normals = payload["normals"]
    confidence = payload["confidence"]
    disagreement = payload["disagreement"]
    valid_mask = payload["valid_mask"]

    if centers.numel() == 0:
        empty_rgb = torch.zeros((3, height, width), device=device, dtype=torch.float32)
        empty_gate = torch.zeros((height, width), device=device, dtype=torch.float32)
        return empty_rgb, empty_gate

    projected, valid = project_payload_points(view, centers, depth_min=depth_min)
    cam_center = view.camera_center.to(device=device, dtype=centers.dtype)
    view_dir = F.normalize(cam_center.unsqueeze(0) - centers, dim=1)
    view_angle = torch.clamp((normals * view_dir).sum(dim=1), 0.0, 1.0)

    xi = torch.round(projected[:, 0]).to(dtype=torch.int64)
    yi = torch.round(projected[:, 1]).to(dtype=torch.int64)
    in_bounds = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    q_gate = _disagreement_gate(disagreement, disagreement_low, disagreement_high)
    carrier_gate = confidence * q_gate * view_angle
    valid = valid & in_bounds & valid_mask & (confidence >= float(min_confidence)) & (carrier_gate > 0.0)
    if region_mask is not None:
        region = region_mask.to(device=device, dtype=torch.bool)
        valid_ids = torch.nonzero(valid, as_tuple=True)[0]
        if valid_ids.numel() > 0:
            valid[valid_ids] = valid[valid_ids] & region[yi[valid_ids], xi[valid_ids]]
    valid_ids = torch.nonzero(valid, as_tuple=True)[0]
    if valid_ids.numel() == 0:
        empty_rgb = torch.zeros((3, height, width), device=device, dtype=torch.float32)
        empty_gate = torch.zeros((height, width), device=device, dtype=torch.float32)
        return empty_rgb, empty_gate

    centers_valid = centers[valid_ids]
    tangent_u = payload["tangent_u"][valid_ids]
    tangent_v = payload["tangent_v"][valid_ids]
    scale_u = payload["scale_u"][valid_ids]
    scale_v = payload["scale_v"][valid_ids]
    projected_valid = projected[valid_ids]
    z_valid = projected_valid[:, 2]

    pu, vu = project_payload_points(view, centers_valid + tangent_u * scale_u[:, None] * float(radius_scale), depth_min=depth_min)
    pv, vv = project_payload_points(view, centers_valid + tangent_v * scale_v[:, None] * float(radius_scale), depth_min=depth_min)
    radius_u = torch.linalg.norm(pu[:, :2] - projected_valid[:, :2], dim=1)
    radius_v = torch.linalg.norm(pv[:, :2] - projected_valid[:, :2], dim=1)
    radius = torch.maximum(radius_u, radius_v)
    radius = torch.where(vu & vv, radius, torch.full_like(radius, float(min_radius_px)))
    radius = torch.ceil(torch.clamp(radius, min=float(min_radius_px), max=float(max_radius_px))).to(dtype=torch.int64)

    pix_x = xi[valid_ids]
    pix_y = yi[valid_ids]
    gate = carrier_gate[valid_ids]
    colors = torch.clamp(payload["fused_rgb"][valid_ids], 0.0, 1.0)
    flat_size = height * width

    pixel_chunks = []
    carrier_chunks = []
    spatial_chunks = []
    depth_chunks = []
    max_radius = int(radius.max().item())
    for r in range(1, max_radius + 1):
        bucket = torch.nonzero(radius == r, as_tuple=True)[0]
        if bucket.numel() == 0:
            continue
        sigma = max(float(r) * 0.5, 1.0)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist2 = float(dx * dx + dy * dy)
                if dist2 > float(r * r):
                    continue
                xx = pix_x[bucket] + dx
                yy = pix_y[bucket] + dy
                inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                if region_mask is not None:
                    region = region_mask.to(device=device, dtype=torch.bool)
                    inside = inside & region[yy.clamp(0, height - 1), xx.clamp(0, width - 1)]
                if not torch.any(inside):
                    continue
                local_ids = bucket[inside]
                pixel_ids = yy[inside] * width + xx[inside]
                spatial = torch.full(
                    (local_ids.shape[0],),
                    np.exp(-dist2 / (2.0 * sigma * sigma)),
                    device=device,
                    dtype=torch.float32,
                )
                pixel_chunks.append(pixel_ids)
                carrier_chunks.append(local_ids)
                spatial_chunks.append(spatial)
                depth_chunks.append(z_valid[local_ids])

    if not pixel_chunks:
        empty_rgb = torch.zeros((3, height, width), device=device, dtype=torch.float32)
        empty_gate = torch.zeros((height, width), device=device, dtype=torch.float32)
        return empty_rgb, empty_gate

    pixel_ids = torch.cat(pixel_chunks, dim=0)
    carrier_ids = torch.cat(carrier_chunks, dim=0)
    spatial = torch.cat(spatial_chunks, dim=0)
    depths = torch.cat(depth_chunks, dim=0)

    min_depth = _scatter_min_depth(flat_size, pixel_ids, depths)
    visible_contrib = depths <= (min_depth[pixel_ids] + float(z_epsilon))
    pixel_ids = pixel_ids[visible_contrib]
    carrier_ids = carrier_ids[visible_contrib]
    spatial = spatial[visible_contrib]
    if pixel_ids.numel() == 0:
        empty_rgb = torch.zeros((3, height, width), device=device, dtype=torch.float32)
        empty_gate = torch.zeros((height, width), device=device, dtype=torch.float32)
        return empty_rgb, empty_gate

    weights = gate[carrier_ids] * spatial
    weight_sum = torch.zeros((flat_size,), device=device, dtype=torch.float32)
    weight_sum.scatter_add_(0, pixel_ids, weights)
    color_sum = torch.zeros((3, flat_size), device=device, dtype=torch.float32)
    weighted_colors = colors[carrier_ids].T * weights.unsqueeze(0)
    color_sum.scatter_add_(1, pixel_ids.unsqueeze(0).expand(3, -1), weighted_colors)

    rgb = color_sum / weight_sum.clamp_min(1e-6).unsqueeze(0)
    coverage = 1.0 - torch.exp(-weight_sum)
    return rgb.reshape(3, height, width), torch.clamp(coverage.reshape(height, width), 0.0, 1.0)


def compose_mesh_fusion_detail(
    sof_rgb: torch.Tensor,
    mesh_rgb: torch.Tensor,
    mesh_gate: torch.Tensor,
    region_mask: Optional[torch.Tensor],
    gate_max: float,
    lowpass_kernel: int,
    low_gate_scale: float,
    high_gate_scale: float,
    low_delta_start: float,
    low_delta_end: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sof_rgb.device
    gate = mesh_gate.to(device=device, dtype=torch.float32)
    if region_mask is not None:
        gate = gate * region_mask.to(device=device, dtype=torch.float32)
    gate = torch.clamp(gate * float(gate_max), 0.0, float(gate_max))

    sof_low = lowpass_chw(sof_rgb, lowpass_kernel)
    mesh_low = lowpass_chw(mesh_rgb, lowpass_kernel)
    low_delta = torch.mean(torch.abs(mesh_low - sof_low), dim=0)
    if low_delta_end <= low_delta_start:
        safety = (low_delta <= low_delta_start).to(dtype=torch.float32)
    else:
        safety = 1.0 - (low_delta - float(low_delta_start)) / float(low_delta_end - low_delta_start)
        safety = torch.clamp(safety, 0.0, 1.0)
    gate = gate * safety

    sof_high = sof_rgb - sof_low
    mesh_high = mesh_rgb - mesh_low
    low_gate = gate * float(low_gate_scale)
    high_gate = gate * float(high_gate_scale)
    composed = sof_rgb + low_gate.unsqueeze(0) * (mesh_low - sof_low) + high_gate.unsqueeze(0) * (mesh_high - sof_high)
    return torch.clamp(composed, 0.0, 1.0), gate
