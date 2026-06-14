from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anneal_mip_covariance_to_surface_v0 import (  # noqa: E402
    build_tangent_basis,
    export_point_cloud,
    make_labeled_grid,
    make_static_copy,
    scalar_image_to_rgb,
    stats_from_array,
    to_uint8_rgb,
)
from gaussian_renderer import render_simple  # noqa: E402
from train_mip_to_sof_surface_v0 import (  # noqa: E402
    copy_render_config,
    load_cameras_for_split,
    load_model_ply,
    resolve_iteration,
    select_uniform,
)
from utils.general_utils import build_rotation  # noqa: E402
from utils.prior_fusion import _quaternion_from_rotation_matrix  # noqa: E402


def load_payload_np(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected payload dict at {path}, got {type(payload)!r}")
    return payload


def payload_array(
    payload: Dict[str, object],
    key: str,
    *,
    count: int,
    dtype: np.dtype = np.float32,
    default: float | int | None = None,
) -> np.ndarray:
    if key not in payload:
        if default is None:
            raise KeyError(f"Payload is missing required key: {key}")
        return np.full((count,), default, dtype=dtype)
    value = payload[key]
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=dtype)
    if arr.shape[0] != int(count):
        raise ValueError(f"Payload key {key!r} length mismatch: {arr.shape[0]} vs gaussian count {count}")
    return arr


def payload_matrix(
    payload: Dict[str, object],
    key: str,
    *,
    count: int,
    width: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Payload is missing required key: {key}")
    value = payload[key]
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=dtype)
    if arr.shape != (int(count), int(width)):
        raise ValueError(f"Payload key {key!r} shape mismatch: {arr.shape} vs {(count, width)}")
    return arr


def unit_scalar_to_rgb(values: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.8 * x, 0.0, 1.0)
    g = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32, copy=False)


def action_class_to_rgb(action_class: np.ndarray) -> np.ndarray:
    colors = np.asarray(
        [
            [0.30, 0.30, 0.34],  # unchanged / residual
            [0.08, 0.86, 0.46],  # surface regulated
            [1.00, 0.22, 0.10],  # off-surface suppressed
        ],
        dtype=np.float32,
    )
    ids = np.clip(np.asarray(action_class, dtype=np.int64), 0, colors.shape[0] - 1)
    return colors[ids]


def vector_stats_torch(values: torch.Tensor) -> Dict[str, float]:
    return stats_from_array(values.detach().cpu().numpy().astype(np.float32, copy=False))


def resize_rgb_for_overview(image: np.ndarray, max_width: int) -> np.ndarray:
    max_width = int(max_width)
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = float(max_width) / float(image.shape[1])
    target = (max_width, max(1, int(round(image.shape[0] * scale))))
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(Image.fromarray(image, mode="RGB").resize(target, resampling.BILINEAR), dtype=np.uint8)


def apply_covariance_surface_regulation(
    *,
    base,
    output,
    nearest_normal: np.ndarray,
    local_mesh_edge: np.ndarray,
    surface_weight: np.ndarray,
    crossing_score: np.ndarray,
    edge_uncertainty: np.ndarray,
    scale_gate: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    device = base.get_xyz.device
    count = int(base.get_xyz.shape[0])
    rotations_np = build_rotation(base._rotation.detach()).detach().cpu().numpy().astype(np.float32, copy=False)
    tangent_u, tangent_v = build_tangent_basis(nearest_normal, rotations_np)
    basis_np = np.stack([tangent_u, tangent_v, nearest_normal], axis=2).astype(np.float32, copy=False)
    basis_t = torch.from_numpy(basis_np).to(device=device, dtype=torch.float32)

    old_scale_t = base.get_scaling.detach()
    scale_t = old_scale_t * scale_gate[:, None]
    rotation_t = base.get_rotation.detach()
    surface_t = torch.from_numpy(np.clip(surface_weight, 0.0, 1.0)).to(device=device, dtype=torch.float32)
    crossing_t = torch.from_numpy(np.clip(crossing_score, 0.0, 1.0)).to(device=device, dtype=torch.float32)
    edge_t = torch.from_numpy(np.clip(edge_uncertainty, 0.0, 1.0)).to(device=device, dtype=torch.float32)
    local_edge_t = torch.from_numpy(np.maximum(local_mesh_edge, 0.0).astype(np.float32, copy=False)).to(device=device)

    target_sigma_floor = torch.clamp(
        local_edge_t * float(args.target_normal_scale_edge_mult),
        min=float(args.min_normal_scale),
    )
    flatten_weight = torch.clamp(
        float(args.cov_flatten_strength)
        * torch.maximum(surface_t, crossing_t * float(args.crossing_flatten_boost))
        * (1.0 - float(args.edge_uncertainty_cov_dampen) * edge_t),
        0.0,
        1.0,
    )
    flatten_weight = torch.where(surface_t >= float(args.min_surface_apply_weight), flatten_weight, torch.zeros_like(flatten_weight))
    sigma_normal_before = torch.zeros((count,), device=device, dtype=torch.float32)
    sigma_normal_after = torch.zeros((count,), device=device, dtype=torch.float32)
    normal_align_before = torch.zeros((count,), device=device, dtype=torch.float32)
    normal_align_after = torch.zeros((count,), device=device, dtype=torch.float32)

    scaling_new = output._scaling.detach().clone()
    rotation_new = output._rotation.detach().clone()
    chunk = max(int(args.chunk_size), 1)
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        R = build_rotation(rotation_t[begin:end])
        scales = scale_t[begin:end]
        basis = basis_t[begin:end]
        cov = R @ torch.diag_embed(scales * scales) @ R.transpose(1, 2)
        local = basis.transpose(1, 2) @ cov @ basis
        sigma_old = torch.sqrt(torch.clamp(local[:, 2, 2], min=1.0e-12))
        target_sigma = torch.minimum(sigma_old, target_sigma_floor[begin:end])

        local_target = local.clone()
        local_target[:, 0, 2] = 0.0
        local_target[:, 1, 2] = 0.0
        local_target[:, 2, 0] = 0.0
        local_target[:, 2, 1] = 0.0
        local_target[:, 2, 2] = target_sigma * target_sigma

        weight = flatten_weight[begin:end]
        local_new = local + weight[:, None, None] * (local_target - local)
        local_new = 0.5 * (local_new + local_new.transpose(1, 2))
        cov_new = basis @ local_new @ basis.transpose(1, 2)
        evals, evecs = torch.linalg.eigh(cov_new)
        evals = torch.clamp(evals, min=float(args.min_eig_scale) ** 2)
        det = torch.linalg.det(evecs)
        flip = det < 0.0
        if torch.any(flip):
            evecs[flip, :, 0] *= -1.0
        quat = _quaternion_from_rotation_matrix(evecs)
        log_scales = torch.log(torch.sqrt(evals))

        scaling_new[begin:end] = log_scales
        rotation_new[begin:end] = quat

        min_idx_before = torch.argmin(scales, dim=1, keepdim=True)
        min_axis_before = torch.gather(R, 2, min_idx_before[:, None, :].expand(-1, 3, 1)).squeeze(2)
        min_axis_after = evecs[:, :, 0]
        normal_chunk = basis[:, :, 2]
        sigma_normal_before[begin:end] = sigma_old
        sigma_normal_after[begin:end] = torch.sqrt(torch.clamp(local_new[:, 2, 2], min=1.0e-12))
        normal_align_before[begin:end] = torch.abs(torch.sum(min_axis_before * normal_chunk, dim=1))
        normal_align_after[begin:end] = torch.abs(torch.sum(min_axis_after * normal_chunk, dim=1))

    output._scaling = nn.Parameter(scaling_new.detach().clone().requires_grad_(False))
    output._rotation = nn.Parameter(rotation_new.detach().clone().requires_grad_(False))
    return {
        "flatten_weight": flatten_weight.detach().cpu().numpy().astype(np.float32, copy=False),
        "sigma_normal_before": sigma_normal_before.detach().cpu().numpy().astype(np.float32, copy=False),
        "sigma_normal_after": sigma_normal_after.detach().cpu().numpy().astype(np.float32, copy=False),
        "normal_align_before": normal_align_before.detach().cpu().numpy().astype(np.float32, copy=False),
        "normal_align_after": normal_align_after.detach().cpu().numpy().astype(np.float32, copy=False),
    }


@torch.no_grad()
def render_apply_previews(
    *,
    base,
    output,
    cameras: Sequence[object],
    output_model_path: Path,
    output_iteration: int,
    action_class_rgb: np.ndarray,
    surface_rgb: np.ndarray,
    suppress_rgb: np.ndarray,
    total_action_rgb: np.ndarray,
    split: str,
    white_background: bool,
    grid_columns: int,
    overview_max_width: int,
) -> List[Dict[str, object]]:
    if not cameras:
        return []
    device = output.get_xyz.device
    background = torch.ones((3,), dtype=torch.float32, device=device) if bool(white_background) else torch.zeros((3,), dtype=torch.float32, device=device)
    render_dir = output_model_path / "mesh_depth_tsdf_apply_previews_v0" / "render_previews"
    render_dir.mkdir(parents=True, exist_ok=True)
    final_render_dir = (
        output_model_path
        / "optimized_renders_no_gt_v0"
        / str(split)
        / f"ours_{int(output_iteration)}"
        / "renders"
    )
    final_render_dir.mkdir(parents=True, exist_ok=True)

    override_specs = [
        ("action_class", torch.from_numpy(action_class_rgb).to(device=device, dtype=torch.float32)),
        ("surface_apply", torch.from_numpy(surface_rgb).to(device=device, dtype=torch.float32)),
        ("opacity_suppress", torch.from_numpy(suppress_rgb).to(device=device, dtype=torch.float32)),
        ("total_action", torch.from_numpy(total_action_rgb).to(device=device, dtype=torch.float32)),
    ]

    overview_tiles: List[Tuple[str, np.ndarray]] = []
    summaries: List[Dict[str, object]] = []
    for view_idx, camera in enumerate(cameras):
        before_pkg = render_simple(camera, base, background)
        after_pkg = render_simple(camera, output, background)
        before_rgb = before_pkg["render"][:3]
        after_rgb = after_pkg["render"][:3]
        delta_hw = torch.mean(torch.abs(after_rgb - before_rgb), dim=0)

        before_u8 = to_uint8_rgb(before_rgb)
        after_u8 = to_uint8_rgb(after_rgb)
        delta_u8 = scalar_image_to_rgb(delta_hw.detach().cpu().numpy(), invert=False)
        view_name = Path(str(camera.image_name)).stem or str(camera.image_name)
        view_dir = render_dir / f"{view_idx:03d}_{view_name}"
        view_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(before_u8, mode="RGB").save(view_dir / "before_render.png")
        Image.fromarray(after_u8, mode="RGB").save(view_dir / "optimized_render.png")
        Image.fromarray(after_u8, mode="RGB").save(final_render_dir / f"{view_idx:05d}.png")
        Image.fromarray(delta_u8, mode="RGB").save(view_dir / "render_delta_heatmap.png")

        tiles: List[Tuple[str, np.ndarray]] = [
            ("before", before_u8),
            ("optimized", after_u8),
            ("delta", delta_u8),
        ]
        for name, override_color in override_specs:
            pkg = render_simple(camera, output, background, override_color=override_color)
            image_u8 = to_uint8_rgb(pkg["render"][:3])
            Image.fromarray(image_u8, mode="RGB").save(view_dir / f"{name}_render.png")
            tiles.append((name, image_u8))

        grid = make_labeled_grid(tiles, columns=max(1, int(grid_columns)))
        grid_path = view_dir / "comparison_grid.png"
        grid.save(grid_path)
        overview_tiles.append(
            (
                f"{view_idx:03d}_{view_name}",
                resize_rgb_for_overview(np.asarray(grid.convert("RGB"), dtype=np.uint8), int(overview_max_width)),
            )
        )
        summaries.append(
            {
                "view_index": int(view_idx),
                "image_name": str(camera.image_name),
                "delta_mean": float(delta_hw.mean().item()),
                "delta_p95": float(np.percentile(delta_hw.detach().cpu().numpy().reshape(-1), 95.0)),
                "output_dir": str(view_dir.resolve()),
                "comparison_grid": str(grid_path.resolve()),
            }
        )

    if overview_tiles:
        make_labeled_grid(overview_tiles, columns=1).save(render_dir / "comparison_overview_v0.png")
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a conservative mesh-depth TSDF regulation payload to a Gaussian model and render before/after previews."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--payload_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--output_iteration", type=int, default=-1)
    parser.add_argument("--iteration_offset", type=int, default=100)
    parser.add_argument("--scene_root", default="")
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--preview_views", type=int, default=8)
    parser.add_argument("--preview_grid_columns", type=int, default=3)
    parser.add_argument("--preview_overview_max_width", type=int, default=1600)
    parser.add_argument("--preview_point_cap", type=int, default=500000)
    parser.add_argument("--preview_point_seed", type=int, default=0)
    parser.add_argument("--white_background", action="store_true")

    parser.add_argument("--min_surface_apply_weight", type=float, default=0.08)
    parser.add_argument("--surface_weight_power", type=float, default=0.85)
    parser.add_argument("--suppress_strength", type=float, default=0.80)
    parser.add_argument("--min_suppress_apply_weight", type=float, default=0.35)
    parser.add_argument("--min_effective_suppress_class", type=float, default=0.20)
    parser.add_argument("--surface_suppress_protect", type=float, default=0.75)
    parser.add_argument("--crossing_suppress_protect", type=float, default=0.55)
    parser.add_argument("--edge_suppress_protect", type=float, default=0.35)
    parser.add_argument("--residual_keep_suppress_protect", type=float, default=0.85)
    parser.add_argument("--min_opacity_gate", type=float, default=0.18)
    parser.add_argument("--min_opacity", type=float, default=1.0e-4)
    parser.add_argument("--uniform_scale_shrink_strength", type=float, default=0.35)
    parser.add_argument("--min_uniform_scale_gate", type=float, default=0.55)

    parser.add_argument("--center_attach_strength", type=float, default=0.45)
    parser.add_argument("--crossing_center_boost", type=float, default=0.45)
    parser.add_argument("--edge_center_dampen", type=float, default=0.65)
    parser.add_argument("--max_center_step_edge_mult", type=float, default=0.75)
    parser.add_argument("--max_center_step_scale_mult", type=float, default=0.50)

    parser.add_argument("--cov_flatten_strength", type=float, default=0.55)
    parser.add_argument("--crossing_flatten_boost", type=float, default=1.15)
    parser.add_argument("--edge_uncertainty_cov_dampen", type=float, default=0.55)
    parser.add_argument("--target_normal_scale_edge_mult", type=float, default=0.35)
    parser.add_argument("--min_normal_scale", type=float, default=1.0e-4)
    parser.add_argument("--min_eig_scale", type=float, default=1.0e-5)
    parser.add_argument("--chunk_size", type=int, default=200000)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    payload_path = Path(args.payload_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve() if str(args.scene_root).strip() else None

    iteration = resolve_iteration(model_path, int(args.iteration))
    output_iteration = int(args.output_iteration)
    if output_iteration <= 0:
        output_iteration = int(iteration) + int(args.iteration_offset)

    base = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    output = make_static_copy(base)
    payload = load_payload_np(payload_path)
    count = int(base.get_xyz.shape[0])

    surface_mass = payload_array(payload, "surface_mass", count=count)
    surface_extract = payload_array(payload, "surface_extract_weight", count=count)
    mesh_bind = payload_array(payload, "mesh_bind_weight", count=count)
    crossing = payload_array(payload, "crossing_score", count=count, default=0.0)
    suppress = payload_array(payload, "opacity_suppress_weight", count=count, default=0.0)
    edge_uncertainty = payload_array(payload, "edge_uncertainty", count=count, default=0.0)
    residual_keep = payload_array(payload, "residual_keep_weight", count=count, default=1.0)
    nearest_normal = payload_matrix(payload, "nearest_surface_normal", count=count, width=3)
    signed_offset = payload_array(payload, "signed_normal_offset", count=count, default=0.0)
    local_edge = payload_array(payload, "local_mesh_edge_length", count=count, default=0.0)
    scale_major = payload_array(payload, "scale_major", count=count, default=0.0)

    nearest_normal = nearest_normal / np.clip(np.linalg.norm(nearest_normal, axis=1, keepdims=True), 1.0e-8, None)
    surface_weight = np.clip(np.maximum(surface_extract, surface_mass * 0.65), 0.0, 1.0)
    surface_weight = np.power(surface_weight, float(args.surface_weight_power)).astype(np.float32, copy=False)
    suppress_threshold = float(np.clip(args.min_suppress_apply_weight, 0.0, 0.95))
    suppress_ramp = np.clip((suppress - suppress_threshold) / max(1.0 - suppress_threshold, 1.0e-6), 0.0, 1.0)
    effective_suppress = np.clip(
        suppress_ramp
        * (1.0 - float(args.surface_suppress_protect) * surface_weight)
        * (1.0 - float(args.crossing_suppress_protect) * np.clip(crossing, 0.0, 1.0))
        * (1.0 - float(args.edge_suppress_protect) * np.clip(edge_uncertainty, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    effective_suppress = np.clip(
        effective_suppress * (1.0 - float(args.residual_keep_suppress_protect) * np.clip(residual_keep, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    center_weight = np.clip(
        float(args.center_attach_strength)
        * np.maximum(mesh_bind, crossing * float(args.crossing_center_boost))
        * (1.0 - float(args.edge_center_dampen) * np.clip(edge_uncertainty, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    center_weight = np.where(surface_weight >= float(args.min_surface_apply_weight), center_weight, 0.0).astype(np.float32)

    device = output.get_xyz.device
    xyz = base.get_xyz.detach()
    old_opacity = base.get_opacity.detach().reshape(-1)
    effective_suppress_t = torch.from_numpy(effective_suppress).to(device=device, dtype=torch.float32)
    opacity_gate = torch.clamp(
        1.0 - float(args.suppress_strength) * effective_suppress_t,
        min=float(args.min_opacity_gate),
        max=1.0,
    )
    new_opacity = torch.clamp(old_opacity * opacity_gate, min=float(args.min_opacity), max=0.999)
    output._opacity = nn.Parameter(output.inverse_opacity_activation(new_opacity[:, None]).detach().clone().requires_grad_(False))

    uniform_scale_gate = torch.clamp(
        1.0 - float(args.uniform_scale_shrink_strength) * effective_suppress_t,
        min=float(args.min_uniform_scale_gate),
        max=1.0,
    )

    center_weight_t = torch.from_numpy(center_weight).to(device=device, dtype=torch.float32)
    normal_t = torch.from_numpy(nearest_normal).to(device=device, dtype=torch.float32)
    signed_t = torch.from_numpy(signed_offset).to(device=device, dtype=torch.float32)
    local_edge_t = torch.from_numpy(np.maximum(local_edge, 0.0).astype(np.float32, copy=False)).to(device=device)
    scale_major_t = torch.from_numpy(np.maximum(scale_major, 0.0).astype(np.float32, copy=False)).to(device=device)
    max_step = torch.maximum(
        local_edge_t * float(args.max_center_step_edge_mult),
        scale_major_t * float(args.max_center_step_scale_mult),
    )
    raw_step = -normal_t * (signed_t * center_weight_t)[:, None]
    raw_step_norm = torch.linalg.norm(raw_step, dim=1)
    step_scale = torch.clamp(max_step / torch.clamp(raw_step_norm, min=1.0e-8), max=1.0)
    applied_step = raw_step * step_scale[:, None]
    output._xyz = nn.Parameter((xyz + applied_step).detach().clone().requires_grad_(False))

    cov_info = apply_covariance_surface_regulation(
        base=base,
        output=output,
        nearest_normal=nearest_normal.astype(np.float32, copy=False),
        local_mesh_edge=local_edge.astype(np.float32, copy=False),
        surface_weight=surface_weight.astype(np.float32, copy=False),
        crossing_score=crossing.astype(np.float32, copy=False),
        edge_uncertainty=edge_uncertainty.astype(np.float32, copy=False),
        scale_gate=uniform_scale_gate,
        args=args,
    )

    total_action = np.clip(
        np.maximum.reduce(
            [
                effective_suppress,
                center_weight,
                cov_info["flatten_weight"],
            ]
        ),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    action_class = np.zeros((count,), dtype=np.int16)
    action_class[cov_info["flatten_weight"] >= float(args.min_surface_apply_weight)] = 1
    action_class[center_weight >= float(args.min_surface_apply_weight)] = 1
    action_class[
        (effective_suppress >= float(args.min_effective_suppress_class))
        & (surface_weight < float(args.min_surface_apply_weight))
    ] = 2

    copy_render_config(model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(output_iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    output.save_ply(str(point_dir / "point_cloud.ply"))
    output.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    preview_root = output_model_path / "mesh_depth_tsdf_apply_previews_v0"
    point_preview_dir = preview_root / "point_cloud_previews"
    point_preview_dir.mkdir(parents=True, exist_ok=True)
    export_point_cloud(point_preview_dir / "action_class_preview_v0.ply", xyz.detach().cpu().numpy(), action_class_to_rgb(action_class), int(args.preview_point_cap), int(args.preview_point_seed))
    export_point_cloud(point_preview_dir / "surface_apply_weight_preview_v0.ply", xyz.detach().cpu().numpy(), unit_scalar_to_rgb(surface_weight), int(args.preview_point_cap), int(args.preview_point_seed))
    export_point_cloud(point_preview_dir / "effective_suppress_preview_v0.ply", xyz.detach().cpu().numpy(), unit_scalar_to_rgb(effective_suppress), int(args.preview_point_cap), int(args.preview_point_seed))
    export_point_cloud(point_preview_dir / "center_attach_weight_preview_v0.ply", xyz.detach().cpu().numpy(), unit_scalar_to_rgb(center_weight), int(args.preview_point_cap), int(args.preview_point_seed))
    export_point_cloud(point_preview_dir / "cov_flatten_weight_preview_v0.ply", xyz.detach().cpu().numpy(), unit_scalar_to_rgb(cov_info["flatten_weight"]), int(args.preview_point_cap), int(args.preview_point_seed))
    export_point_cloud(point_preview_dir / "total_action_weight_preview_v0.ply", xyz.detach().cpu().numpy(), unit_scalar_to_rgb(total_action), int(args.preview_point_cap), int(args.preview_point_seed))

    render_summaries: List[Dict[str, object]] = []
    if scene_root is not None and int(args.preview_views) > 0:
        cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
        cameras = select_uniform(cameras, int(args.preview_views))
        render_summaries = render_apply_previews(
            base=base,
            output=output,
            cameras=cameras,
            output_model_path=output_model_path,
            output_iteration=output_iteration,
            action_class_rgb=action_class_to_rgb(action_class),
            surface_rgb=unit_scalar_to_rgb(surface_weight),
            suppress_rgb=unit_scalar_to_rgb(effective_suppress),
            total_action_rgb=unit_scalar_to_rgb(total_action),
            split=str(args.split),
            white_background=bool(args.white_background),
            grid_columns=int(args.preview_grid_columns),
            overview_max_width=int(args.preview_overview_max_width),
        )

    apply_payload = {
        "version": "mesh_depth_tsdf_regulation_apply_v0",
        "source_payload_path": str(payload_path),
        "action_class": torch.from_numpy(action_class.astype(np.int16, copy=False)),
        "surface_apply_weight": torch.from_numpy(surface_weight.astype(np.float32, copy=False)),
        "effective_suppress_weight": torch.from_numpy(effective_suppress.astype(np.float32, copy=False)),
        "suppress_ramp_weight": torch.from_numpy(suppress_ramp.astype(np.float32, copy=False)),
        "center_attach_weight": torch.from_numpy(center_weight.astype(np.float32, copy=False)),
        "cov_flatten_weight": torch.from_numpy(cov_info["flatten_weight"].astype(np.float32, copy=False)),
        "total_action_weight": torch.from_numpy(total_action.astype(np.float32, copy=False)),
        "old_opacity": old_opacity.detach().cpu(),
        "new_opacity": output.get_opacity.detach().cpu().reshape(-1),
        "opacity_gate": opacity_gate.detach().cpu(),
        "uniform_scale_gate": uniform_scale_gate.detach().cpu(),
        "center_step": applied_step.detach().cpu(),
        "sigma_normal_before": torch.from_numpy(cov_info["sigma_normal_before"]),
        "sigma_normal_after": torch.from_numpy(cov_info["sigma_normal_after"]),
        "normal_align_before": torch.from_numpy(cov_info["normal_align_before"]),
        "normal_align_after": torch.from_numpy(cov_info["normal_align_after"]),
        "residual_keep_weight": torch.from_numpy(residual_keep.astype(np.float32, copy=False)),
    }
    apply_payload_path = output_model_path / "mesh_depth_tsdf_regulation_apply_v0.pt"
    torch.save(apply_payload, apply_payload_path)

    center_step_norm = torch.linalg.norm(applied_step, dim=1)
    summary = {
        "version": "mesh_depth_tsdf_regulation_apply_v0",
        "model_path": str(model_path),
        "iteration": int(iteration),
        "payload_path": str(payload_path),
        "output_model_path": str(output_model_path),
        "output_iteration": int(output_iteration),
        "num_gaussians": int(count),
        "action_counts": {
            "unchanged_or_residual": int(np.count_nonzero(action_class == 0)),
            "surface_regulated": int(np.count_nonzero(action_class == 1)),
            "off_surface_suppressed": int(np.count_nonzero(action_class == 2)),
        },
        "stats": {
            "surface_apply_weight": stats_from_array(surface_weight),
            "effective_suppress_weight": stats_from_array(effective_suppress),
            "suppress_ramp_weight": stats_from_array(suppress_ramp),
            "center_attach_weight": stats_from_array(center_weight),
            "cov_flatten_weight": stats_from_array(cov_info["flatten_weight"]),
            "total_action_weight": stats_from_array(total_action),
            "opacity_gate": vector_stats_torch(opacity_gate),
            "uniform_scale_gate": vector_stats_torch(uniform_scale_gate),
            "center_step_norm": vector_stats_torch(center_step_norm),
            "sigma_normal_before": stats_from_array(cov_info["sigma_normal_before"]),
            "sigma_normal_after": stats_from_array(cov_info["sigma_normal_after"]),
            "normal_align_before": stats_from_array(cov_info["normal_align_before"]),
            "normal_align_after": stats_from_array(cov_info["normal_align_after"]),
        },
        "render_summaries": render_summaries,
        "artifacts": {
            "output_ply": str(point_dir / "point_cloud.ply"),
            "output_tags": str(point_dir / "gaussian_tags.pt"),
            "apply_payload": str(apply_payload_path),
            "preview_root": str(preview_root),
            "render_preview_dir": str(preview_root / "render_previews"),
            "optimized_render_dir": str(output_model_path / "optimized_renders_no_gt_v0"),
        },
        "params": vars(args),
        "note": (
            "This apply stage does not edit color/SH or densify. It conservatively pulls usable near-surface Gaussians "
            "along the mesh normal, flattens their covariance toward the mesh tangent plane, and weakens off-surface "
            "non-usable contributions by opacity/scale gates. Residual Gaussians are mostly left intact for render preservation."
        ),
    }
    summary_path = output_model_path / "mesh_depth_tsdf_regulation_apply_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[mesh-depth-tsdf-apply-v0] output model : {output_model_path}")
    print(f"[mesh-depth-tsdf-apply-v0] output ply   : {point_dir / 'point_cloud.ply'}")
    print(f"[mesh-depth-tsdf-apply-v0] renders      : {preview_root / 'render_previews'}")
    print(f"[mesh-depth-tsdf-apply-v0] summary      : {summary_path}")


if __name__ == "__main__":
    main()
