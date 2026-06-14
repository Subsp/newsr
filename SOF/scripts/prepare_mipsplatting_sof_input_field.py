from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import build_rotation
from utils.sh_utils import SH2RGB
from utils.system_utils import mkdir_p, searchForMaxIteration


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return int(iteration)
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")
    return int(searchForMaxIteration(str(point_cloud_root)))


def _resolve_input_iteration(model_path: Path, iteration: int) -> int:
    return _resolve_model_iteration(model_path, iteration)


def _resolve_input_ply(model_path: Path, iteration: int) -> Path:
    point_dir = model_path / "point_cloud" / f"iteration_{int(iteration)}"
    ply_path = point_dir / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"mip point cloud not found: {ply_path}")
    return ply_path


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "config.json", "cameras.json"]:
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def write_clean_cfg_args(output_model_path: Path, scene_root: Path, images_subdir: str, sh_degree: int) -> None:
    payload = Namespace(
        sh_degree=int(sh_degree),
        source_path=str(scene_root),
        model_path=str(output_model_path),
        images=str(images_subdir),
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
        kernel_size=0.1,
        ray_jitter=False,
        resample_gt_image=False,
        load_allres=False,
        sample_more_highres=False,
        vanilla_mip_mode=False,
    )
    with open(output_model_path / "cfg_args", "w", encoding="utf-8") as f:
        f.write(repr(payload))


def _clone_gaussian_snapshot(
    template: GaussianModel,
    *,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    output_source_idx: torch.Tensor,
    child_output_mask: torch.Tensor | None = None,
    softened_output_mask: torch.Tensor | None = None,
) -> GaussianModel:
    count = int(xyz.shape[0])
    snap = GaussianModel(template.max_sh_degree, use_SBs=template.use_SBs)
    snap.active_sh_degree = int(template.active_sh_degree)
    snap.spatial_lr_scale = float(template.spatial_lr_scale)
    snap._xyz = nn.Parameter(xyz.detach().clone().requires_grad_(False))
    snap._features_dc = nn.Parameter(features_dc.detach().clone().requires_grad_(False))
    snap._features_rest = nn.Parameter(features_rest.detach().clone().requires_grad_(False))
    snap._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(False))
    snap._scaling = nn.Parameter(scaling.detach().clone().requires_grad_(False))
    snap._rotation = nn.Parameter(rotation.detach().clone().requires_grad_(False))
    snap.filter_3D = filter_3d.detach().clone()
    snap.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=xyz.device)
    snap.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=xyz.device)
    snap.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=xyz.device)
    snap.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=xyz.device)
    snap.denom = torch.zeros((count, 1), dtype=torch.float32, device=xyz.device)
    snap.init_tracking_state(count, source_tag=int(GaussianSourceTag.PRIOR_INJECTED))
    source_idx = output_source_idx.to(device=xyz.device, dtype=torch.int64).reshape(-1)
    if int(source_idx.shape[0]) != count:
        raise RuntimeError(
            f"output_source_idx has {int(source_idx.shape[0])} rows but stage snapshot has {count} gaussians"
        )
    child_mask = (
        torch.zeros((count,), dtype=torch.bool, device=xyz.device)
        if child_output_mask is None
        else child_output_mask.to(device=xyz.device, dtype=torch.bool).reshape(-1)
    )
    softened_mask = (
        torch.zeros((count,), dtype=torch.bool, device=xyz.device)
        if softened_output_mask is None
        else softened_output_mask.to(device=xyz.device, dtype=torch.bool).reshape(-1)
    )
    snap._seed_id = source_idx.clone()
    snap._generation = child_mask.to(dtype=torch.int32).clone()
    snap._source_tag = torch.full((count,), int(GaussianSourceTag.PRIOR_INJECTED), dtype=torch.int16, device=xyz.device)
    snap._edge_touched = softened_mask.clone()
    snap._edge_touch_iter = softened_mask.to(dtype=torch.int32).clone()
    return snap


def _format_debug_stage_index(stage_index: int | str) -> str:
    if isinstance(stage_index, int):
        return f"{stage_index:02d}"
    text = str(stage_index).strip()
    return text if text else "xx"


def _save_debug_stage_snapshot(
    *,
    output_model_path: Path,
    output_iteration: int,
    source_model_path: Path,
    template: GaussianModel,
    stage_name: str,
    stage_index: int | str,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    output_source_idx: torch.Tensor,
    child_output_mask: torch.Tensor | None = None,
    softened_output_mask: torch.Tensor | None = None,
    extra_summary: dict | None = None,
) -> None:
    stage_index_label = _format_debug_stage_index(stage_index)
    stage_root = output_model_path / "debug_prepare_stages" / f"debug_stage_{stage_index_label}_{stage_name}"
    point_dir = stage_root / "point_cloud" / f"iteration_{int(output_iteration)}"
    mkdir_p(str(point_dir))
    _copy_render_config(source_model_path, stage_root)
    snap = _clone_gaussian_snapshot(
        template,
        xyz=xyz,
        features_dc=features_dc,
        features_rest=features_rest,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        filter_3d=filter_3d,
        output_source_idx=output_source_idx,
        child_output_mask=child_output_mask,
        softened_output_mask=softened_output_mask,
    )
    snap.save_ply(str(point_dir / "point_cloud.ply"))
    snap.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    masks_dir = stage_root / "masks"
    mkdir_p(str(masks_dir))
    torch.save(output_source_idx.detach().cpu().to(dtype=torch.int64), masks_dir / "output_source_idx.pt")
    if child_output_mask is not None:
        torch.save(child_output_mask.detach().cpu().to(dtype=torch.bool), masks_dir / "init_repair_child_output_mask.pt")
    if softened_output_mask is not None:
        torch.save(softened_output_mask.detach().cpu().to(dtype=torch.bool), masks_dir / "init_repair_softened_output_mask.pt")
    summary = {
        "mode": "prepare_stage_debug_snapshot",
        "stage_name": str(stage_name),
        "stage_index": stage_index,
        "stage_index_label": stage_index_label,
        "output_model_path": str(stage_root),
        "output_iteration": int(output_iteration),
        "num_gaussians": int(xyz.shape[0]),
        "paths": {
            "point_cloud": str(point_dir / "point_cloud.ply"),
            "gaussian_tags": str(point_dir / "gaussian_tags.pt"),
        },
    }
    if extra_summary:
        summary.update(extra_summary)
    (stage_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _load_reference_scene_stats(
    sof_ref_model: Path | None,
    quantile_low: float,
    quantile_high: float,
) -> dict | None:
    if sof_ref_model is None:
        return None
    ref_iteration = _resolve_model_iteration(sof_ref_model, -1)
    ref_ply = _resolve_input_ply(sof_ref_model, ref_iteration)
    ref = GaussianModel(3, use_SBs=False)
    ref.load_ply(str(ref_ply))
    xyz = ref.get_xyz.detach()
    q_low = torch.quantile(xyz, float(quantile_low), dim=0)
    q_high = torch.quantile(xyz, float(quantile_high), dim=0)
    scene_diag = float(torch.linalg.norm(q_high - q_low).item())
    return {
        "ref_model_path": str(sof_ref_model),
        "ref_iteration": int(ref_iteration),
        "ref_ply": str(ref_ply),
        "aabb_min": q_low,
        "aabb_max": q_high,
        "scene_diag": scene_diag,
    }


def _build_finite_mask(gaussians: GaussianModel) -> torch.Tensor:
    pieces = [
        gaussians._xyz,
        gaussians._features_dc.flatten(start_dim=1),
        gaussians._features_rest.flatten(start_dim=1),
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ]
    if isinstance(gaussians.filter_3D, torch.Tensor) and gaussians.filter_3D.ndim > 0:
        pieces.append(gaussians.filter_3D)

    mask = torch.ones((gaussians._xyz.shape[0],), dtype=torch.bool, device=gaussians._xyz.device)
    for tensor in pieces:
        mask &= torch.isfinite(tensor).all(dim=1)
    return mask


def _normalize_rotation(rotation: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(rotation, dim=1, keepdim=True)
    safe = norm > 1e-8
    normalized = torch.zeros_like(rotation)
    normalized[:, 0] = 1.0
    if safe.any():
        safe_rows = safe.squeeze(1)
        normalized[safe_rows] = rotation[safe_rows] / norm[safe_rows]
    return normalized


def _scalar_stats(values: torch.Tensor, max_quantile_samples: int = 2_000_000) -> dict[str, float | int | bool | None]:
    if values.numel() == 0:
        return {
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "count": 0,
            "quantile_sampled": False,
        }
    flat = values.detach().reshape(-1).float()
    quantile_source = flat
    quantile_sampled = False
    if flat.numel() > int(max_quantile_samples):
        # torch.quantile can fail on very large tensors. A deterministic stride
        # sample is enough for diagnostics and does not change the exported GS.
        step = max(1, flat.numel() // int(max_quantile_samples))
        quantile_source = flat[::step]
        quantile_sampled = True
    return {
        "min": float(flat.min().item()),
        "median": float(torch.quantile(quantile_source, 0.50).item()),
        "p90": float(torch.quantile(quantile_source, 0.90).item()),
        "p95": float(torch.quantile(quantile_source, 0.95).item()),
        "p99": float(torch.quantile(quantile_source, 0.99).item()),
        "max": float(flat.max().item()),
        "count": int(flat.numel()),
        "quantile_sampled": bool(quantile_sampled),
    }


def _canonicalize_opacity_raw(opacity: torch.Tensor, opacity_min: float, opacity_max: float) -> tuple[torch.Tensor, str]:
    opacity = torch.nan_to_num(opacity, nan=opacity_min, posinf=opacity_max, neginf=opacity_min)
    if opacity.numel() > 0 and float(opacity.min().item()) >= 0.0 and float(opacity.max().item()) <= 1.0:
        eps = 1e-6
        prob = opacity.clamp(min=eps, max=1.0 - eps)
        raw = torch.log(prob / (1.0 - prob))
        return raw.clamp(min=opacity_min, max=opacity_max), "probability_to_logit"
    return opacity.clamp(min=opacity_min, max=opacity_max), "raw_logit"


def _normalize_scale_clamp_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "min": "min_only",
        "max": "max_only",
        "both": "both",
        "min_only": "min_only",
        "max_only": "max_only",
        "none": "none",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported scale clamp mode: {mode!r}")
    return aliases[mode]


def _apply_scale_clamp(
    activated_scale: torch.Tensor,
    *,
    activated_min: float,
    activated_max: float,
    mode: str,
) -> torch.Tensor:
    clamp_mode = _normalize_scale_clamp_mode(mode)
    out = activated_scale
    if clamp_mode in {"both", "min_only"}:
        out = torch.clamp(out, min=float(activated_min))
    if clamp_mode in {"both", "max_only"}:
        out = torch.clamp(out, max=float(activated_max))
    return out


def _canonicalize_scale_raw(
    scale: torch.Tensor,
    scene_diag: float,
    scale_min_ratio: float,
    scale_max_ratio: float,
    activated_scale_detect_max: float,
    scale_clamp_mode: str = "both",
) -> tuple[torch.Tensor, str, str, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = torch.nan_to_num(scale, nan=-6.0, posinf=3.0, neginf=-8.0)
    if scale.numel() > 0 and float(scale.min().item()) > 0.0 and float(scale.max().item()) < float(activated_scale_detect_max):
        scale_raw = torch.log(torch.clamp(scale, min=1e-8))
        mode = "activated_to_log"
    else:
        scale_raw = scale
        mode = "raw_log"

    activated_before = torch.exp(scale_raw)
    scene_diag = max(float(scene_diag), 1e-6)
    activated_min = scene_diag * float(scale_min_ratio)
    activated_max = scene_diag * float(scale_max_ratio)
    activated_min_only = _apply_scale_clamp(
        activated_before,
        activated_min=activated_min,
        activated_max=activated_max,
        mode="min_only",
    )
    activated_max_only = _apply_scale_clamp(
        activated_before,
        activated_min=activated_min,
        activated_max=activated_max,
        mode="max_only",
    )
    clamp_mode = _normalize_scale_clamp_mode(scale_clamp_mode)
    activated_after = _apply_scale_clamp(
        activated_before,
        activated_min=activated_min,
        activated_max=activated_max,
        mode=clamp_mode,
    )
    return (
        torch.log(torch.clamp(activated_after, min=1e-12)),
        mode,
        clamp_mode,
        activated_min,
        activated_max,
        scale_raw,
        activated_before,
        activated_after,
        activated_min_only,
        activated_max_only,
    )


def _opacity_compensation_metric(scales: torch.Tensor, mode: str) -> torch.Tensor:
    safe = torch.clamp(scales, min=1e-12)
    if mode == "volume":
        return torch.prod(safe, dim=1, keepdim=True)
    sorted_scales = torch.sort(safe, dim=1).values
    return torch.prod(sorted_scales[:, -2:], dim=1, keepdim=True)


def _compensate_opacity_for_scale_shrink(
    opacity_raw: torch.Tensor,
    scale_before: torch.Tensor,
    scale_after: torch.Tensor,
    *,
    mode: str,
    power: float,
    min_opacity_scale: float,
    max_opacity: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    if mode == "none":
        return opacity_raw, {
            "mode": "none",
            "affected_count": 0,
            "opacity_scale_stats": _scalar_stats(torch.ones_like(opacity_raw)),
        }

    before_metric = _opacity_compensation_metric(scale_before, mode)
    after_metric = _opacity_compensation_metric(scale_after, mode)
    shrink_ratio = torch.clamp(after_metric / torch.clamp(before_metric, min=1e-12), min=0.0, max=1.0)
    opacity_scale = torch.pow(shrink_ratio, max(float(power), 0.0))
    opacity_scale = torch.clamp(opacity_scale, min=min(max(float(min_opacity_scale), 0.0), 1.0), max=1.0)
    opacity_prob = torch.sigmoid(opacity_raw)
    max_prob = min(max(float(max_opacity), 1e-6), 1.0 - 1e-6)
    compensated = torch.clamp(opacity_prob * opacity_scale, min=1e-6, max=max_prob)
    compensated_raw = torch.log(compensated / torch.clamp(1.0 - compensated, min=1e-6))
    affected = shrink_ratio < 0.999
    return compensated_raw, {
        "mode": mode,
        "power": float(power),
        "min_opacity_scale": float(min_opacity_scale),
        "max_opacity": float(max_opacity),
        "affected_count": int(affected.sum().item()),
        "affected_ratio": float(affected.float().mean().item()) if affected.numel() else 0.0,
        "scale_metric_ratio_stats": _scalar_stats(shrink_ratio),
        "opacity_scale_stats": _scalar_stats(opacity_scale),
        "opacity_activated_before_stats": _scalar_stats(opacity_prob),
        "opacity_activated_after_stats": _scalar_stats(compensated),
    }


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probability / torch.clamp(1.0 - probability, min=1e-6))


def _resolve_init_repair_modes(mode: str) -> tuple[str, str]:
    mode = str(mode)
    if mode == "none":
        return "none", "none"
    if mode == "split_replace":
        return "split_replace", "none"
    if mode == "energy_split_replace":
        return "energy_split_replace", "none"
    if mode == "soften_bright":
        return "none", "soften_bright"
    if mode == "split_replace_soften_bright":
        return "split_replace", "soften_bright"
    if mode == "energy_split_replace_soften_bright":
        return "energy_split_replace", "soften_bright"
    raise ValueError(f"Unsupported init repair mode: {mode}")


def _features_dc_to_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    if features_dc.ndim == 3:
        if features_dc.shape[1] == 1:
            sh_dc = features_dc[:, 0, :]
        elif features_dc.shape[2] == 1:
            sh_dc = features_dc[:, :, 0]
        else:
            sh_dc = features_dc.reshape(features_dc.shape[0], -1)[:, :3]
    elif features_dc.ndim == 2:
        sh_dc = features_dc[:, :3]
    else:
        raise ValueError(f"Unsupported features_dc shape for RGB conversion: {tuple(features_dc.shape)}")
    return torch.nan_to_num(SH2RGB(sh_dc), nan=0.0, posinf=0.0, neginf=0.0)


def _split_pattern(split_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if int(split_count) <= 2:
        values = [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    elif int(split_count) <= 4:
        values = [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    else:
        values = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    return torch.tensor(values, dtype=dtype, device=device)


def _line_split_pattern(split_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    count = max(2, int(split_count))
    x = torch.linspace(-1.0, 1.0, steps=count, dtype=dtype, device=device)
    zeros = torch.zeros_like(x)
    return torch.stack((x, zeros, zeros), dim=1)


def _apply_init_repair(
    *,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    source_indices: torch.Tensor,
    scene_diag: float,
    mode: str,
    max_repair_fraction: float,
    max_repair_count: int,
    min_opacity: float,
    min_effective_scale_ratio: float,
    min_volume_radius_ratio: float,
    min_filter_scale_ratio: float,
    min_full_anisotropy: float,
    split_count: int,
    child_layout: str,
    child_scale_multiplier: float,
    child_major_scale_multiplier: float,
    child_opacity_scale: float,
    energy_conserve_mode: str,
    filter_scale: float,
    filter_cap_ratio: float,
    offset_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    mode = str(mode)
    child_layout = str(child_layout)
    energy_conserve_mode = str(energy_conserve_mode)
    source_indices = source_indices.to(device=xyz.device, dtype=torch.int64)
    passthrough = {
        "output_source_idx": source_indices.detach().cpu(),
        "init_repair_replaced_source_idx": torch.empty((0,), dtype=torch.int64),
        "init_repair_child_output_mask": torch.zeros((xyz.shape[0],), dtype=torch.bool),
        "init_repair": {
            "mode": mode,
            "enabled": mode != "none",
            "candidate_count_before_cap": 0,
            "replaced_count": 0,
            "child_count": 0,
            "output_count_before": int(xyz.shape[0]),
            "output_count_after": int(xyz.shape[0]),
        },
    }
    if mode == "none":
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, source_indices, passthrough
    if mode not in {"split_replace", "energy_split_replace"}:
        raise ValueError(f"Unsupported init repair mode: {mode}")
    if child_layout not in {"grid", "major_axis"}:
        raise ValueError(f"Unsupported init repair child layout: {child_layout}")
    if energy_conserve_mode not in {"none", "area", "volume"}:
        raise ValueError(f"Unsupported init repair energy conserve mode: {energy_conserve_mode}")
    if mode == "energy_split_replace":
        if child_layout == "grid":
            child_layout = "major_axis"
        if energy_conserve_mode == "none":
            energy_conserve_mode = "area"
    if xyz.shape[0] == 0:
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, source_indices, passthrough

    split_count = int(split_count)
    if split_count < 2:
        raise ValueError("--init_repair_split_count must be >= 2 when init repair is enabled")

    scene_diag = max(float(scene_diag), 1e-6)
    scale = torch.exp(scaling)
    filter_col = filter_3d.reshape(-1, 1).to(device=xyz.device, dtype=scale.dtype)
    effective_scale = torch.sqrt(torch.square(scale) + torch.square(filter_col))
    sorted_scale = torch.sort(torch.clamp(effective_scale, min=1e-8), dim=1).values
    scale_min = sorted_scale[:, 0]
    scale_max = sorted_scale[:, 2]
    volume_radius = torch.clamp(torch.prod(effective_scale, dim=1), min=1e-24).pow(1.0 / 3.0)
    raw_scale_max = torch.clamp(scale.max(dim=1).values, min=1e-8)
    filter_scale_ratio = filter_col.reshape(-1) / raw_scale_max
    full_anisotropy = scale_max / torch.clamp(scale_min, min=1e-8)
    opacity_prob = torch.sigmoid(opacity).reshape(-1)

    min_effective_scale = scene_diag * float(min_effective_scale_ratio)
    min_volume_radius = scene_diag * float(min_volume_radius_ratio)
    scale_risk = scale_max / max(min_effective_scale, 1e-8)
    volume_risk = volume_radius / max(min_volume_radius, 1e-8)
    if float(min_filter_scale_ratio) > 0.0:
        filter_risk = filter_scale_ratio / max(float(min_filter_scale_ratio), 1e-8)
    else:
        filter_risk = torch.zeros_like(filter_scale_ratio)
    size_or_filter_risk = torch.maximum(torch.maximum(scale_risk, volume_risk), filter_risk)

    large_enough = (scale_max >= min_effective_scale) | (volume_radius >= min_volume_radius)
    if float(min_filter_scale_ratio) > 0.0:
        large_enough = large_enough | (filter_scale_ratio >= float(min_filter_scale_ratio))
    if float(min_full_anisotropy) > 0.0:
        large_enough = large_enough & (full_anisotropy >= float(min_full_anisotropy))
    candidate_mask = large_enough & (opacity_prob >= float(min_opacity))
    candidate_count_before_cap = int(candidate_mask.sum().item())
    if candidate_count_before_cap == 0:
        passthrough["init_repair"].update(
            {
                "candidate_count_before_cap": 0,
                "thresholds": {
                    "min_opacity": float(min_opacity),
                    "min_effective_scale": float(min_effective_scale),
                    "min_volume_radius": float(min_volume_radius),
                    "min_filter_scale_ratio": float(min_filter_scale_ratio),
                    "min_full_anisotropy": float(min_full_anisotropy),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, source_indices, passthrough

    score = size_or_filter_risk * torch.clamp(opacity_prob, min=1e-4)
    candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    max_by_fraction = int(max(0, float(max_repair_fraction) * float(xyz.shape[0])))
    max_count = candidate_ids.numel()
    max_count = min(max_count, max_by_fraction)
    if int(max_repair_count) > 0:
        max_count = min(max_count, int(max_repair_count))
    if max_count <= 0:
        passthrough["init_repair"].update(
            {
                "candidate_count_before_cap": candidate_count_before_cap,
                "candidate_count_after_cap": 0,
                "thresholds": {
                    "min_opacity": float(min_opacity),
                    "min_effective_scale": float(min_effective_scale),
                    "min_volume_radius": float(min_volume_radius),
                    "min_filter_scale_ratio": float(min_filter_scale_ratio),
                    "min_full_anisotropy": float(min_full_anisotropy),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, source_indices, passthrough

    selected_order = torch.topk(score[candidate_ids], k=max_count, largest=True, sorted=False).indices
    selected_ids = candidate_ids[selected_order]
    keep_original = torch.ones((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
    keep_original[selected_ids] = False

    if child_layout == "major_axis":
        pattern = _line_split_pattern(split_count, device=xyz.device, dtype=xyz.dtype)
    else:
        pattern = _split_pattern(split_count, device=xyz.device, dtype=xyz.dtype)
    actual_split_count = int(pattern.shape[0])
    selected_scale = effective_scale[selected_ids]
    axis_order = torch.argsort(selected_scale, dim=1, descending=True)
    rotations = build_rotation(rotation[selected_ids])
    axis_basis = torch.gather(rotations, 2, axis_order[:, None, :].expand(-1, 3, -1))
    sorted_selected_scale = torch.gather(selected_scale, 1, axis_order)
    local_offsets = pattern[None, :, :] * sorted_selected_scale[:, None, :] * float(offset_scale)
    child_xyz = xyz[selected_ids, None, :] + torch.einsum("bij,bnj->bni", axis_basis, local_offsets)

    if child_layout == "major_axis":
        selected_raw_scale = scale[selected_ids]
        sorted_raw_scale = torch.gather(selected_raw_scale, 1, axis_order)
        major_scale_multiplier = float(child_major_scale_multiplier)
        if major_scale_multiplier <= 0.0:
            major_scale_multiplier = min(float(child_scale_multiplier), 1.0)
        sorted_child_scale = sorted_raw_scale.clone()
        sorted_child_scale[:, 0] = sorted_child_scale[:, 0] * major_scale_multiplier
        sorted_child_scale[:, 1:] = sorted_child_scale[:, 1:] * float(child_scale_multiplier)
        updated_scale = torch.zeros_like(selected_raw_scale)
        updated_scale.scatter_(1, axis_order, sorted_child_scale)
        child_scale = updated_scale[:, None, :].expand(-1, actual_split_count, -1)
    else:
        child_scale = scale[selected_ids, None, :].expand(-1, actual_split_count, -1)
        child_scale = child_scale * float(child_scale_multiplier)
    child_scale = torch.clamp(child_scale, min=1e-8)
    child_scaling = torch.log(child_scale).reshape(-1, 3)

    child_filter = filter_3d[selected_ids, None, :].expand(-1, actual_split_count, -1)
    child_filter = child_filter * float(filter_scale)
    if float(filter_cap_ratio) > 0.0:
        child_filter = torch.clamp(child_filter, max=scene_diag * float(filter_cap_ratio))
    child_filter = torch.clamp(child_filter, min=0.0)

    parent_opacity = opacity_prob[selected_ids]
    energy_ratio = torch.ones_like(parent_opacity)
    if energy_conserve_mode == "none":
        preserved_child_opacity = 1.0 - torch.pow(torch.clamp(1.0 - parent_opacity, min=1e-6), 1.0 / float(actual_split_count))
        child_opacity_prob_2d = preserved_child_opacity[:, None].expand(-1, actual_split_count)
        child_opacity_prob_2d = child_opacity_prob_2d * float(child_opacity_scale)
    else:
        parent_metric = _opacity_compensation_metric(selected_scale, energy_conserve_mode).reshape(-1)
        child_effective_scale = torch.sqrt(torch.square(child_scale) + torch.square(child_filter))
        child_metric = _opacity_compensation_metric(
            child_effective_scale.reshape(-1, 3),
            energy_conserve_mode,
        ).reshape(-1, actual_split_count)
        child_metric_sum = torch.clamp(child_metric.sum(dim=1), min=1e-12)
        target_mass = parent_opacity * parent_metric * float(child_opacity_scale)
        child_opacity_prob_2d = (target_mass / child_metric_sum)[:, None].expand(-1, actual_split_count)
        realized_mass = torch.clamp(child_opacity_prob_2d, min=1e-6, max=0.95) * child_metric
        energy_ratio = realized_mass.sum(dim=1) / torch.clamp(parent_opacity * parent_metric, min=1e-12)
    child_opacity_prob_2d = torch.clamp(child_opacity_prob_2d, min=1e-6, max=0.95)
    child_opacity = _logit(child_opacity_prob_2d.reshape(-1, 1))
    child_filter = child_filter.reshape(-1, filter_3d.shape[1])

    child_features_dc = features_dc[selected_ids, None, ...].expand(-1, actual_split_count, *features_dc.shape[1:]).reshape(
        -1,
        *features_dc.shape[1:],
    )
    child_features_rest = features_rest[selected_ids, None, ...].expand(
        -1,
        actual_split_count,
        *features_rest.shape[1:],
    ).reshape(-1, *features_rest.shape[1:])
    child_rotation = rotation[selected_ids, None, :].expand(-1, actual_split_count, -1).reshape(-1, rotation.shape[1])
    child_source_indices = source_indices[selected_ids].repeat_interleave(actual_split_count)

    xyz_out = torch.cat((xyz[keep_original], child_xyz.reshape(-1, 3)), dim=0)
    features_dc_out = torch.cat((features_dc[keep_original], child_features_dc), dim=0)
    features_rest_out = torch.cat((features_rest[keep_original], child_features_rest), dim=0)
    opacity_out = torch.cat((opacity[keep_original], child_opacity), dim=0)
    scaling_out = torch.cat((scaling[keep_original], child_scaling), dim=0)
    rotation_out = torch.cat((rotation[keep_original], child_rotation), dim=0)
    filter_out = torch.cat((filter_3d[keep_original], child_filter), dim=0)
    source_indices_out = torch.cat((source_indices[keep_original], child_source_indices), dim=0)
    output_count = int(xyz_out.shape[0])
    for name, tensor in (
        ("features_dc", features_dc_out),
        ("features_rest", features_rest_out),
        ("opacity", opacity_out),
        ("scaling", scaling_out),
        ("rotation", rotation_out),
        ("filter_3D", filter_out),
        ("source_indices", source_indices_out),
    ):
        if int(tensor.shape[0]) != output_count:
            raise RuntimeError(f"init repair produced {output_count} xyz rows but {name} has {int(tensor.shape[0])} rows")
    child_output_mask = torch.cat(
        (
            torch.zeros((int(keep_original.sum().item()),), dtype=torch.bool, device=xyz.device),
            torch.ones((child_source_indices.shape[0],), dtype=torch.bool, device=xyz.device),
        ),
        dim=0,
    )

    selected_score = score[selected_ids]
    selected_summary = {
        "mode": mode,
        "enabled": True,
        "split_count": int(actual_split_count),
        "candidate_count_before_cap": candidate_count_before_cap,
        "candidate_count_after_cap": int(selected_ids.numel()),
        "replaced_count": int(selected_ids.numel()),
        "child_count": int(child_source_indices.numel()),
        "output_count_before": int(xyz.shape[0]),
        "output_count_after": int(xyz_out.shape[0]),
        "max_repair_fraction": float(max_repair_fraction),
        "max_repair_count": int(max_repair_count),
        "child_layout": str(child_layout),
        "child_scale_multiplier": float(child_scale_multiplier),
        "child_major_scale_multiplier": float(child_major_scale_multiplier),
        "child_opacity_scale": float(child_opacity_scale),
        "energy_conserve_mode": str(energy_conserve_mode),
        "filter_scale": float(filter_scale),
        "filter_cap_ratio": float(filter_cap_ratio),
        "offset_scale": float(offset_scale),
        "thresholds": {
            "min_opacity": float(min_opacity),
            "min_effective_scale": float(min_effective_scale),
            "min_volume_radius": float(min_volume_radius),
            "min_filter_scale_ratio": float(min_filter_scale_ratio),
            "min_full_anisotropy": float(min_full_anisotropy),
        },
        "selected_score_stats": _scalar_stats(selected_score),
        "selected_opacity_stats": _scalar_stats(opacity_prob[selected_ids]),
        "selected_effective_scale_max_stats": _scalar_stats(scale_max[selected_ids]),
        "selected_volume_radius_stats": _scalar_stats(volume_radius[selected_ids]),
        "selected_filter_scale_ratio_stats": _scalar_stats(filter_scale_ratio[selected_ids]),
        "selected_full_anisotropy_stats": _scalar_stats(full_anisotropy[selected_ids]),
        "selected_energy_conserve_ratio_stats": _scalar_stats(energy_ratio),
    }
    repair_payload = {
        "output_source_idx": source_indices_out.detach().cpu(),
        "init_repair_replaced_source_idx": source_indices[selected_ids].detach().cpu(),
        "init_repair_child_output_mask": child_output_mask.detach().cpu(),
        "init_repair": selected_summary,
    }
    return (
        xyz_out,
        features_dc_out,
        features_rest_out,
        opacity_out,
        scaling_out,
        rotation_out,
        filter_out,
        source_indices_out,
        repair_payload,
    )


def _apply_bright_soften(
    *,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    output_source_idx: torch.Tensor,
    child_output_mask: torch.Tensor,
    scene_diag: float,
    mode: str,
    target_group: str,
    max_bright_fraction: float,
    max_bright_count: int,
    min_opacity: float,
    max_effective_scale: float,
    luma_quantile: float,
    min_local_luma_ratio: float,
    min_color_delta: float,
    neighbor_k: int,
    expand_scale_multiplier: float,
    smallest_axis_scale_multiplier: float,
    opacity_scale: float,
    dc_scale: float,
    rest_scale: float,
    filter_scale: float,
    filter_cap_ratio: float,
    global_scale_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    mode = str(mode)
    target_group = str(target_group)
    output_source_idx = output_source_idx.to(device=xyz.device, dtype=torch.int64)
    child_output_mask = child_output_mask.to(device=xyz.device, dtype=torch.bool).reshape(-1)
    passthrough = {
        "init_repair_softened_source_idx": torch.empty((0,), dtype=torch.int64),
        "init_repair_softened_output_mask": torch.zeros((xyz.shape[0],), dtype=torch.bool),
        "init_bright_soften": {
            "mode": mode,
            "enabled": mode != "none",
            "candidate_pool_count": 0,
            "candidate_count_before_cap": 0,
            "candidate_count_after_cap": 0,
            "softened_count": 0,
            "output_count": int(xyz.shape[0]),
        },
    }
    if mode == "none":
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough
    if mode != "soften_bright":
        raise ValueError(f"Unsupported init bright soften mode: {mode}")
    if target_group not in {"non_children", "children_only", "all"}:
        raise ValueError(f"Unsupported init bright soften target group: {target_group}")
    if xyz.shape[0] <= 1:
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough
    if int(neighbor_k) < 1:
        raise ValueError("--init_repair_bright_neighbor_k must be >= 1 when bright soften is enabled")

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for --init_repair_mode=soften_bright") from exc

    scene_diag = max(float(scene_diag), 1e-6)
    scale = torch.exp(scaling)
    filter_col = filter_3d.reshape(-1, 1).to(device=xyz.device, dtype=scale.dtype)
    effective_scale = torch.sqrt(torch.square(scale) + torch.square(filter_col))
    effective_scale_max = effective_scale.max(dim=1).values
    opacity_prob = torch.sigmoid(opacity).reshape(-1)
    rgb = torch.clamp(_features_dc_to_rgb(features_dc), min=0.0)
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]

    if target_group == "children_only":
        candidate_base = child_output_mask.clone()
    elif target_group == "all":
        candidate_base = torch.ones_like(child_output_mask)
    else:
        candidate_base = ~child_output_mask
    candidate_base &= torch.isfinite(luma) & torch.isfinite(opacity_prob)
    candidate_base &= opacity_prob >= float(min_opacity)
    if float(max_effective_scale) > 0.0:
        candidate_base &= effective_scale_max <= float(max_effective_scale)
    if not torch.any(candidate_base):
        passthrough["init_bright_soften"].update(
            {
                "thresholds": {
                    "target_group": target_group,
                    "min_opacity": float(min_opacity),
                    "max_effective_scale": float(max_effective_scale),
                    "luma_quantile": float(luma_quantile),
                    "min_local_luma_ratio": float(min_local_luma_ratio),
                    "min_color_delta": float(min_color_delta),
                    "neighbor_k": int(neighbor_k),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough

    clamped_quantile = float(np.clip(float(luma_quantile), 0.0, 1.0))
    luma_threshold = float(torch.quantile(luma[candidate_base].float(), clamped_quantile).item())
    candidate_pool = candidate_base & (luma >= luma_threshold)
    candidate_ids = torch.nonzero(candidate_pool, as_tuple=False).squeeze(1)
    candidate_pool_count = int(candidate_ids.numel())
    if candidate_pool_count == 0:
        passthrough["init_bright_soften"].update(
            {
                "candidate_pool_count": 0,
                "luma_threshold": float(luma_threshold),
                "thresholds": {
                    "target_group": target_group,
                    "min_opacity": float(min_opacity),
                    "max_effective_scale": float(max_effective_scale),
                    "luma_quantile": clamped_quantile,
                    "min_local_luma_ratio": float(min_local_luma_ratio),
                    "min_color_delta": float(min_color_delta),
                    "neighbor_k": int(neighbor_k),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough

    xyz_np = xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    rgb_np = rgb.detach().cpu().numpy().astype(np.float32, copy=False)
    luma_np = luma.detach().cpu().numpy().astype(np.float32, copy=False)
    candidate_ids_np = candidate_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    query_k = min(int(neighbor_k) + 1, xyz_np.shape[0])
    if query_k <= 1:
        passthrough["init_bright_soften"].update(
            {
                "candidate_pool_count": candidate_pool_count,
                "luma_threshold": float(luma_threshold),
                "thresholds": {
                    "target_group": target_group,
                    "min_opacity": float(min_opacity),
                    "max_effective_scale": float(max_effective_scale),
                    "luma_quantile": clamped_quantile,
                    "min_local_luma_ratio": float(min_local_luma_ratio),
                    "min_color_delta": float(min_color_delta),
                    "neighbor_k": int(neighbor_k),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough

    tree = cKDTree(xyz_np)
    _, neighbor_ids = tree.query(xyz_np[candidate_ids_np], k=query_k, workers=1)
    neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64)
    if neighbor_ids.ndim == 1:
        neighbor_ids = neighbor_ids[:, None]
    if neighbor_ids.shape[1] <= 1:
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough
    neighbor_ids = np.clip(neighbor_ids[:, 1:], 0, xyz_np.shape[0] - 1)

    neighbor_luma = luma_np[neighbor_ids]
    neighbor_rgb = rgb_np[neighbor_ids]
    local_luma_ref = np.median(neighbor_luma, axis=1).astype(np.float32, copy=False)
    local_rgb_ref = np.mean(neighbor_rgb, axis=1).astype(np.float32, copy=False)
    local_luma_ratio = np.clip(
        luma_np[candidate_ids_np] / np.clip(local_luma_ref, 1e-4, None),
        0.0,
        1e6,
    ).astype(np.float32, copy=False)
    color_delta = np.linalg.norm(rgb_np[candidate_ids_np] - local_rgb_ref, axis=1).astype(np.float32, copy=False)

    contrast_candidate_mask = (
        (local_luma_ratio >= float(min_local_luma_ratio))
        & (color_delta >= float(min_color_delta))
        & np.isfinite(local_luma_ratio)
        & np.isfinite(color_delta)
    )
    candidate_count_before_cap = int(np.count_nonzero(contrast_candidate_mask))
    if candidate_count_before_cap == 0:
        passthrough["init_bright_soften"].update(
            {
                "candidate_pool_count": candidate_pool_count,
                "candidate_count_before_cap": 0,
                "luma_threshold": float(luma_threshold),
                "local_luma_ratio_stats": _scalar_stats(torch.from_numpy(local_luma_ratio)),
                "color_delta_stats": _scalar_stats(torch.from_numpy(color_delta)),
                "thresholds": {
                    "target_group": target_group,
                    "min_opacity": float(min_opacity),
                    "max_effective_scale": float(max_effective_scale),
                    "luma_quantile": clamped_quantile,
                    "min_local_luma_ratio": float(min_local_luma_ratio),
                    "min_color_delta": float(min_color_delta),
                    "neighbor_k": int(neighbor_k),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough

    selected_pool_ids_np = candidate_ids_np[contrast_candidate_mask]
    selected_local_luma_ratio = local_luma_ratio[contrast_candidate_mask]
    selected_color_delta = color_delta[contrast_candidate_mask]
    selected_luma = luma_np[selected_pool_ids_np]
    selected_opacity = opacity_prob[selected_pool_ids_np].detach().cpu().numpy().astype(np.float32, copy=False)
    selected_score = (
        selected_opacity
        * np.clip(selected_luma / max(float(luma_threshold), 1e-4), 0.0, 4.0)
        * np.clip(selected_local_luma_ratio / max(float(min_local_luma_ratio), 1e-4), 0.0, 4.0)
        * np.clip(selected_color_delta / max(float(min_color_delta), 1e-4), 0.0, 4.0)
    ).astype(np.float32, copy=False)

    max_by_fraction = int(max(0, float(max_bright_fraction) * float(xyz.shape[0])))
    max_count = selected_pool_ids_np.shape[0]
    max_count = min(max_count, max_by_fraction)
    if int(max_bright_count) > 0:
        max_count = min(max_count, int(max_bright_count))
    if max_count <= 0:
        passthrough["init_bright_soften"].update(
            {
                "candidate_pool_count": candidate_pool_count,
                "candidate_count_before_cap": candidate_count_before_cap,
                "candidate_count_after_cap": 0,
                "luma_threshold": float(luma_threshold),
                "local_luma_ratio_stats": _scalar_stats(torch.from_numpy(local_luma_ratio)),
                "color_delta_stats": _scalar_stats(torch.from_numpy(color_delta)),
                "thresholds": {
                    "target_group": target_group,
                    "min_opacity": float(min_opacity),
                    "max_effective_scale": float(max_effective_scale),
                    "luma_quantile": clamped_quantile,
                    "min_local_luma_ratio": float(min_local_luma_ratio),
                    "min_color_delta": float(min_color_delta),
                    "neighbor_k": int(neighbor_k),
                },
            }
        )
        return xyz, features_dc, features_rest, opacity, scaling, rotation, filter_3d, passthrough

    selected_order = np.argsort(-selected_score, kind="stable")[:max_count]
    selected_ids_np = selected_pool_ids_np[selected_order]
    selected_ids = torch.from_numpy(selected_ids_np).to(device=xyz.device, dtype=torch.int64)

    scaling_out = scaling.detach().clone()
    opacity_out = opacity.detach().clone()
    features_dc_out = features_dc.detach().clone()
    features_rest_out = features_rest.detach().clone()
    filter_out = filter_3d.detach().clone()

    selected_scale = torch.exp(scaling_out[selected_ids]).clone()
    axis_order = torch.argsort(selected_scale, dim=1, descending=True)
    sorted_scale = torch.gather(selected_scale, 1, axis_order)
    scale_multiplier = torch.ones_like(sorted_scale)
    scale_multiplier[:, :2] *= float(expand_scale_multiplier)
    scale_multiplier[:, 2] *= float(smallest_axis_scale_multiplier)
    sorted_scale = torch.clamp(
        sorted_scale * scale_multiplier,
        min=1e-8,
        max=max(float(global_scale_cap), 1e-8),
    )
    updated_scale = torch.zeros_like(selected_scale)
    updated_scale.scatter_(1, axis_order, sorted_scale)
    scaling_out[selected_ids] = torch.log(torch.clamp(updated_scale, min=1e-8))

    softened_opacity_prob = torch.clamp(
        opacity_prob[selected_ids].reshape(-1, 1) * float(opacity_scale),
        min=1e-6,
        max=0.95,
    )
    opacity_out[selected_ids] = _logit(softened_opacity_prob)
    features_dc_out[selected_ids] = features_dc_out[selected_ids] * float(dc_scale)
    features_rest_out[selected_ids] = features_rest_out[selected_ids] * float(rest_scale)
    filter_out[selected_ids] = filter_out[selected_ids] * float(filter_scale)
    if float(filter_cap_ratio) > 0.0:
        filter_out[selected_ids] = torch.clamp(filter_out[selected_ids], max=scene_diag * float(filter_cap_ratio))
    filter_out = torch.clamp(filter_out, min=0.0)

    softened_output_mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
    softened_output_mask[selected_ids] = True
    selected_local_luma_ratio_t = torch.from_numpy(selected_local_luma_ratio[selected_order].astype(np.float32, copy=False))
    selected_color_delta_t = torch.from_numpy(selected_color_delta[selected_order].astype(np.float32, copy=False))
    selected_score_t = torch.from_numpy(selected_score[selected_order].astype(np.float32, copy=False))
    selected_source_idx = output_source_idx[selected_ids]
    bright_summary = {
        "mode": mode,
        "enabled": True,
        "candidate_pool_count": candidate_pool_count,
        "candidate_count_before_cap": candidate_count_before_cap,
        "candidate_count_after_cap": int(selected_ids.shape[0]),
        "softened_count": int(selected_ids.shape[0]),
        "output_count": int(xyz.shape[0]),
        "max_bright_fraction": float(max_bright_fraction),
        "max_bright_count": int(max_bright_count),
        "expand_scale_multiplier": float(expand_scale_multiplier),
        "smallest_axis_scale_multiplier": float(smallest_axis_scale_multiplier),
        "opacity_scale": float(opacity_scale),
        "dc_scale": float(dc_scale),
        "rest_scale": float(rest_scale),
        "filter_scale": float(filter_scale),
        "filter_cap_ratio": float(filter_cap_ratio),
        "luma_threshold": float(luma_threshold),
        "thresholds": {
            "target_group": target_group,
            "min_opacity": float(min_opacity),
            "max_effective_scale": float(max_effective_scale),
            "luma_quantile": clamped_quantile,
            "min_local_luma_ratio": float(min_local_luma_ratio),
            "min_color_delta": float(min_color_delta),
            "neighbor_k": int(neighbor_k),
        },
        "local_luma_ratio_stats": _scalar_stats(torch.from_numpy(local_luma_ratio)),
        "color_delta_stats": _scalar_stats(torch.from_numpy(color_delta)),
        "selected_score_stats": _scalar_stats(selected_score_t),
        "selected_luma_stats": _scalar_stats(torch.from_numpy(selected_luma[selected_order].astype(np.float32, copy=False))),
        "selected_local_luma_ratio_stats": _scalar_stats(selected_local_luma_ratio_t),
        "selected_color_delta_stats": _scalar_stats(selected_color_delta_t),
        "selected_opacity_stats": _scalar_stats(opacity_prob[selected_ids]),
        "selected_effective_scale_max_stats": _scalar_stats(effective_scale_max[selected_ids]),
    }
    soften_payload = {
        "init_repair_softened_source_idx": selected_source_idx.detach().cpu(),
        "init_repair_softened_output_mask": softened_output_mask.detach().cpu(),
        "init_bright_soften": bright_summary,
    }
    return (
        xyz,
        features_dc_out,
        features_rest_out,
        opacity_out,
        scaling_out,
        rotation,
        filter_out,
        soften_payload,
    )


def _sanitize_gaussians(
    gaussians: GaussianModel,
    scene_stats: dict | None,
    opacity_min: float,
    opacity_max: float,
    scale_min_ratio: float,
    scale_max_ratio: float,
    activated_scale_detect_max: float,
    opacity_compensate_scale_shrink: str,
    opacity_compensation_power: float,
    min_opacity_compensation_scale: float,
    max_compensated_opacity: float,
    scale_clamp_mode: str,
    feature_clip: float,
    use_aabb_filter: bool,
    aabb_margin_ratio: float,
    filter_mode: str,
    filter_constant: float,
    init_repair_mode: str,
    init_repair_max_fraction: float,
    init_repair_max_count: int,
    init_repair_min_opacity: float,
    init_repair_min_effective_scale_ratio: float,
    init_repair_min_volume_radius_ratio: float,
    init_repair_min_filter_scale_ratio: float,
    init_repair_min_full_anisotropy: float,
    init_repair_split_count: int,
    init_repair_child_layout: str,
    init_repair_child_scale_multiplier: float,
    init_repair_child_major_scale_multiplier: float,
    init_repair_child_opacity_scale: float,
    init_repair_energy_conserve_mode: str,
    init_repair_filter_scale: float,
    init_repair_filter_cap_ratio: float,
    init_repair_offset_scale: float,
    init_repair_bright_max_fraction: float,
    init_repair_bright_max_count: int,
    init_repair_bright_target: str,
    init_repair_bright_min_opacity: float,
    init_repair_bright_max_effective_scale_ratio: float,
    init_repair_bright_luma_quantile: float,
    init_repair_bright_min_local_luma_ratio: float,
    init_repair_bright_min_color_delta: float,
    init_repair_bright_neighbor_k: int,
    init_repair_bright_expand_scale_multiplier: float,
    init_repair_bright_smallest_axis_scale_multiplier: float,
    init_repair_bright_opacity_scale: float,
    init_repair_bright_dc_scale: float,
    init_repair_bright_rest_scale: float,
    init_repair_bright_filter_scale: float,
    init_repair_bright_filter_cap_ratio: float,
    debug_dump_canonicalize_substages: bool = False,
    debug_stage_writer=None,
) -> dict:
    finite_mask = _build_finite_mask(gaussians)
    xyz_all = gaussians._xyz.detach()
    keep_mask = finite_mask.clone()
    dropped_nonfinite = int((~finite_mask).sum().item())
    dropped_aabb = 0

    aabb_min = None
    aabb_max = None
    scene_diag = 1.0
    if scene_stats is not None:
        aabb_min = scene_stats["aabb_min"].to(device=xyz_all.device, dtype=xyz_all.dtype)
        aabb_max = scene_stats["aabb_max"].to(device=xyz_all.device, dtype=xyz_all.dtype)
        scene_diag = max(float(scene_stats["scene_diag"]), 1e-6)
        if use_aabb_filter:
            margin = scene_diag * float(aabb_margin_ratio)
            inside = ((xyz_all >= (aabb_min - margin)) & (xyz_all <= (aabb_max + margin))).all(dim=1)
            keep_mask &= inside
            dropped_aabb = int((finite_mask & ~inside).sum().item())

    xyz = gaussians._xyz.detach()[keep_mask]
    features_dc = gaussians._features_dc.detach()[keep_mask]
    features_rest = gaussians._features_rest.detach()[keep_mask]
    if xyz.shape[0] == 0:
        raise RuntimeError("All mip gaussians were dropped during sanitization; no valid input field remains.")
    source_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).to(device=xyz.device, dtype=torch.int64)
    raw_opacity = gaussians._opacity.detach()[keep_mask]
    raw_scaling = gaussians._scaling.detach()[keep_mask]
    raw_rotation = gaussians._rotation.detach()[keep_mask]
    if isinstance(gaussians.filter_3D, torch.Tensor) and gaussians.filter_3D.ndim > 0:
        raw_filter_3d = gaussians.filter_3D.detach()[keep_mask]
        raw_filter_3d = torch.nan_to_num(raw_filter_3d, nan=0.0, posinf=0.0, neginf=0.0)
        raw_filter_3d = torch.clamp(raw_filter_3d, min=0.0)
    else:
        raw_filter_3d = torch.zeros((xyz.shape[0], 1), dtype=torch.float32, device=xyz.device)
    if debug_stage_writer is not None:
        debug_stage_writer(
            "after_finite_aabb",
            0,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=raw_opacity,
            scaling=raw_scaling,
            rotation=raw_rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "counts": {
                    "dropped_nonfinite": int(dropped_nonfinite),
                    "dropped_aabb": int(dropped_aabb),
                }
            },
        )
    opacity, opacity_mode = _canonicalize_opacity_raw(
        raw_opacity,
        opacity_min=opacity_min,
        opacity_max=opacity_max,
    )
    if debug_stage_writer is not None and bool(debug_dump_canonicalize_substages):
        debug_stage_writer(
            "after_opacity_canonicalize",
            "00a",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=raw_scaling,
            rotation=raw_rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={"opacity_mode": str(opacity_mode)},
        )
    (
        scaling,
        scale_mode,
        scale_clamp_mode_used,
        activated_scale_min,
        activated_scale_max,
        scale_raw_canonical,
        activated_scale_before,
        activated_scale_after,
        activated_scale_min_only,
        activated_scale_max_only,
    ) = _canonicalize_scale_raw(
        raw_scaling,
        scene_diag=scene_diag,
        scale_min_ratio=scale_min_ratio,
        scale_max_ratio=scale_max_ratio,
        activated_scale_detect_max=activated_scale_detect_max,
        scale_clamp_mode=scale_clamp_mode,
    )
    if debug_stage_writer is not None and bool(debug_dump_canonicalize_substages):
        debug_stage_writer(
            "after_scale_domain_only",
            "00b0",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scale_raw_canonical,
            rotation=raw_rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
                "scale_clamp_mode": "none",
                "scale_activated_before_clamp_stats": _scalar_stats(activated_scale_before),
            },
        )
        debug_stage_writer(
            "after_scale_min_clamp_only",
            "00b1",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=torch.log(torch.clamp(activated_scale_min_only, min=1e-12)),
            rotation=raw_rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
                "scale_clamp_mode": "min_only",
                "scale_activated_min": float(activated_scale_min),
                "scale_activated_max": float(activated_scale_max),
                "scale_activated_before_clamp_stats": _scalar_stats(activated_scale_before),
                "scale_activated_after_clamp_stats": _scalar_stats(activated_scale_min_only),
            },
        )
        debug_stage_writer(
            "after_scale_max_clamp_only",
            "00b2",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=torch.log(torch.clamp(activated_scale_max_only, min=1e-12)),
            rotation=raw_rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
                "scale_clamp_mode": "max_only",
                "scale_activated_min": float(activated_scale_min),
                "scale_activated_max": float(activated_scale_max),
                "scale_activated_before_clamp_stats": _scalar_stats(activated_scale_before),
                "scale_activated_after_clamp_stats": _scalar_stats(activated_scale_max_only),
            },
        )
        debug_stage_writer(
            "after_scale_canonicalize",
            "00b3",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=raw_rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
                "scale_clamp_mode": str(scale_clamp_mode_used),
                "scale_activated_min": float(activated_scale_min),
                "scale_activated_max": float(activated_scale_max),
                "scale_activated_before_clamp_stats": _scalar_stats(activated_scale_before),
                "scale_activated_after_clamp_stats": _scalar_stats(activated_scale_after),
            },
        )
    rotation = _normalize_rotation(raw_rotation)
    if debug_stage_writer is not None and bool(debug_dump_canonicalize_substages):
        debug_stage_writer(
            "after_rotation_normalize",
            "00c",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
            },
        )
    features_dc = torch.nan_to_num(features_dc, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        min=-float(feature_clip),
        max=float(feature_clip),
    )
    features_rest = torch.nan_to_num(features_rest, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        min=-float(feature_clip),
        max=float(feature_clip),
    )
    if debug_stage_writer is not None and bool(debug_dump_canonicalize_substages):
        debug_stage_writer(
            "after_feature_clip",
            "00d",
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            filter_3d=raw_filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
                "feature_clip": float(feature_clip),
            },
        )

    rotation_norm = torch.linalg.norm(rotation, dim=1)
    bad_rotation_count = int((rotation_norm < 1e-6).sum().item())

    filter_3d = raw_filter_3d.clone()

    if filter_mode == "zero":
        filter_3d = torch.zeros_like(filter_3d)
    elif filter_mode == "constant":
        filter_3d = torch.full_like(filter_3d, float(filter_constant))

    if debug_stage_writer is not None:
        debug_stage_writer(
            "after_canonicalize",
            1,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            filter_3d=filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_mode": str(opacity_mode),
                "scale_mode": str(scale_mode),
            },
        )

    opacity, opacity_compensation_summary = _compensate_opacity_for_scale_shrink(
        opacity,
        activated_scale_before,
        activated_scale_after,
        mode=str(opacity_compensate_scale_shrink),
        power=float(opacity_compensation_power),
        min_opacity_scale=float(min_opacity_compensation_scale),
        max_opacity=float(max_compensated_opacity),
    )
    if debug_stage_writer is not None:
        debug_stage_writer(
            "after_opacity_comp",
            2,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            filter_3d=filter_3d,
            output_source_idx=source_indices,
            extra_summary={
                "opacity_scale_shrink_compensation": opacity_compensation_summary,
            },
        )

    split_repair_mode, bright_soften_mode = _resolve_init_repair_modes(str(init_repair_mode))
    (
        xyz,
        features_dc,
        features_rest,
        opacity,
        scaling,
        rotation,
        filter_3d,
        output_source_idx,
        init_repair_summary,
    ) = _apply_init_repair(
        xyz=xyz,
        features_dc=features_dc,
        features_rest=features_rest,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        filter_3d=filter_3d,
        source_indices=source_indices,
        scene_diag=scene_diag,
        mode=str(split_repair_mode),
        max_repair_fraction=float(init_repair_max_fraction),
        max_repair_count=int(init_repair_max_count),
        min_opacity=float(init_repair_min_opacity),
        min_effective_scale_ratio=float(init_repair_min_effective_scale_ratio),
        min_volume_radius_ratio=float(init_repair_min_volume_radius_ratio),
        min_filter_scale_ratio=float(init_repair_min_filter_scale_ratio),
        min_full_anisotropy=float(init_repair_min_full_anisotropy),
        split_count=int(init_repair_split_count),
        child_layout=str(init_repair_child_layout),
        child_scale_multiplier=float(init_repair_child_scale_multiplier),
        child_major_scale_multiplier=float(init_repair_child_major_scale_multiplier),
        child_opacity_scale=float(init_repair_child_opacity_scale),
        energy_conserve_mode=str(init_repair_energy_conserve_mode),
        filter_scale=float(init_repair_filter_scale),
        filter_cap_ratio=float(init_repair_filter_cap_ratio),
        offset_scale=float(init_repair_offset_scale),
    )
    init_repair_child_output_mask = init_repair_summary["init_repair_child_output_mask"]
    if debug_stage_writer is not None:
        debug_stage_writer(
            "after_init_repair",
            3,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            filter_3d=filter_3d,
            output_source_idx=output_source_idx,
            child_output_mask=init_repair_child_output_mask,
            extra_summary={"init_repair": init_repair_summary.get("init_repair", {})},
        )
    (
        xyz,
        features_dc,
        features_rest,
        opacity,
        scaling,
        rotation,
        filter_3d,
        bright_soften_summary,
    ) = _apply_bright_soften(
        xyz=xyz,
        features_dc=features_dc,
        features_rest=features_rest,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        filter_3d=filter_3d,
        output_source_idx=output_source_idx,
        child_output_mask=init_repair_child_output_mask,
        scene_diag=scene_diag,
        mode=str(bright_soften_mode),
        target_group=str(init_repair_bright_target),
        max_bright_fraction=float(init_repair_bright_max_fraction),
        max_bright_count=int(init_repair_bright_max_count),
        min_opacity=float(init_repair_bright_min_opacity),
        max_effective_scale=scene_diag * float(init_repair_bright_max_effective_scale_ratio),
        luma_quantile=float(init_repair_bright_luma_quantile),
        min_local_luma_ratio=float(init_repair_bright_min_local_luma_ratio),
        min_color_delta=float(init_repair_bright_min_color_delta),
        neighbor_k=int(init_repair_bright_neighbor_k),
        expand_scale_multiplier=float(init_repair_bright_expand_scale_multiplier),
        smallest_axis_scale_multiplier=float(init_repair_bright_smallest_axis_scale_multiplier),
        opacity_scale=float(init_repair_bright_opacity_scale),
        dc_scale=float(init_repair_bright_dc_scale),
        rest_scale=float(init_repair_bright_rest_scale),
        filter_scale=float(init_repair_bright_filter_scale),
        filter_cap_ratio=float(init_repair_bright_filter_cap_ratio),
        global_scale_cap=float(activated_scale_max),
    )
    if debug_stage_writer is not None:
        debug_stage_writer(
            "after_bright_soften",
            4,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            filter_3d=filter_3d,
            output_source_idx=output_source_idx,
            child_output_mask=init_repair_child_output_mask,
            softened_output_mask=bright_soften_summary["init_repair_softened_output_mask"],
            extra_summary={
                "init_repair": init_repair_summary.get("init_repair", {}),
                "init_bright_soften": bright_soften_summary.get("init_bright_soften", {}),
            },
        )

    gaussians._xyz = nn.Parameter(xyz.requires_grad_(False))
    gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(False))
    gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(False))
    gaussians._opacity = nn.Parameter(opacity.requires_grad_(False))
    gaussians._scaling = nn.Parameter(scaling.requires_grad_(False))
    gaussians._rotation = nn.Parameter(rotation.requires_grad_(False))
    gaussians.filter_3D = filter_3d
    gaussians.max_radii2D = torch.zeros((xyz.shape[0],), dtype=torch.float32, device=xyz.device)
    gaussians.init_tracking_state(xyz.shape[0], source_tag=int(GaussianSourceTag.PRIOR_INJECTED))
    gaussians._seed_id = output_source_idx.to(device=xyz.device, dtype=torch.int64).clone()
    gaussians._generation = init_repair_child_output_mask.to(device=xyz.device, dtype=torch.int32).clone()

    activated_opacity = torch.sigmoid(opacity)
    activated_scaling = torch.exp(scaling)
    return {
        "keep_mask": keep_mask.detach().cpu(),
        "num_input_gaussians": int(keep_mask.shape[0]),
        "num_output_gaussians": int(xyz.shape[0]),
        "dropped_nonfinite": dropped_nonfinite,
        "dropped_aabb": dropped_aabb,
        "filter_mode": filter_mode,
        **init_repair_summary,
        **bright_soften_summary,
        "opacity_mode": opacity_mode,
        "scale_mode": scale_mode,
        "scale_clamp_mode": scale_clamp_mode_used,
        "opacity_scale_shrink_compensation": opacity_compensation_summary,
        "opacity_raw_stats": _scalar_stats(opacity),
        "opacity_activated_stats": _scalar_stats(activated_opacity),
        "scale_raw_stats": _scalar_stats(scaling),
        "scale_activated_stats": _scalar_stats(activated_scaling),
        "scale_activated_before_clamp_stats": _scalar_stats(activated_scale_before),
        "feature_dc_stats": _scalar_stats(features_dc),
        "feature_rest_stats": _scalar_stats(features_rest),
        "bad_rotation_count": bad_rotation_count,
        "scale_activated_min": float(activated_scale_min),
        "scale_activated_max": float(activated_scale_max),
        "tracking_seed_id_assigned": True,
        "tracking_child_generation_count": int(init_repair_child_output_mask.to(dtype=torch.int64).sum().item()),
        "scene_diag": float(scene_diag),
        "aabb_quantile_min": aabb_min.detach().cpu().tolist() if aabb_min is not None else None,
        "aabb_quantile_max": aabb_max.detach().cpu().tolist() if aabb_max is not None else None,
    }


def main() -> None:
    parser = ArgumentParser(description="Convert a mip-splatting point-cloud model into a sanitized SOF-native input field.")
    parser.add_argument("--mip_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--scene_root", default=None)
    parser.add_argument("--sof_ref_model", default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_iteration", type=int, default=-1)
    parser.add_argument("--opacity_raw_min", type=float, default=-8.0)
    parser.add_argument("--opacity_raw_max", type=float, default=1.0)
    parser.add_argument("--scale_min_ratio", type=float, default=1e-5)
    parser.add_argument("--scale_max_ratio", type=float, default=1e-2)
    parser.add_argument("--activated_scale_detect_max", type=float, default=1.0)
    parser.add_argument("--scale_clamp_mode", choices=["both", "min_only", "max_only", "none"], default="both")
    parser.add_argument("--opacity_compensate_scale_shrink", choices=["none", "area", "volume"], default="none")
    parser.add_argument("--opacity_compensation_power", type=float, default=1.0)
    parser.add_argument("--min_opacity_compensation_scale", type=float, default=0.05)
    parser.add_argument("--max_compensated_opacity", type=float, default=0.95)
    parser.add_argument("--feature_clip", type=float, default=10.0)
    parser.add_argument("--use_aabb_filter", action="store_true")
    parser.add_argument("--aabb_quantile_low", type=float, default=0.01)
    parser.add_argument("--aabb_quantile_high", type=float, default=0.99)
    parser.add_argument("--aabb_margin_ratio", type=float, default=0.25)
    parser.add_argument("--filter_mode", choices=["keep", "zero", "constant"], default="keep")
    parser.add_argument("--filter_constant", type=float, default=0.0)
    parser.add_argument(
        "--init_repair_mode",
        choices=[
            "none",
            "split_replace",
            "energy_split_replace",
            "soften_bright",
            "split_replace_soften_bright",
            "energy_split_replace_soften_bright",
        ],
        default="none",
    )
    parser.add_argument("--init_repair_max_fraction", type=float, default=0.04)
    parser.add_argument("--init_repair_max_count", type=int, default=50000)
    parser.add_argument("--init_repair_min_opacity", type=float, default=0.04)
    parser.add_argument("--init_repair_min_effective_scale_ratio", type=float, default=0.003)
    parser.add_argument("--init_repair_min_volume_radius_ratio", type=float, default=0.0015)
    parser.add_argument("--init_repair_min_filter_scale_ratio", type=float, default=0.75)
    parser.add_argument("--init_repair_min_full_anisotropy", type=float, default=0.0)
    parser.add_argument("--init_repair_split_count", type=int, default=4)
    parser.add_argument("--init_repair_child_layout", choices=["grid", "major_axis"], default="grid")
    parser.add_argument("--init_repair_child_scale_multiplier", type=float, default=0.55)
    parser.add_argument("--init_repair_child_major_scale_multiplier", type=float, default=-1.0)
    parser.add_argument("--init_repair_child_opacity_scale", type=float, default=0.75)
    parser.add_argument("--init_repair_energy_conserve_mode", choices=["none", "area", "volume"], default="none")
    parser.add_argument("--init_repair_filter_scale", type=float, default=0.25)
    parser.add_argument("--init_repair_filter_cap_ratio", type=float, default=0.0015)
    parser.add_argument("--init_repair_offset_scale", type=float, default=0.45)
    parser.add_argument("--init_repair_bright_max_fraction", type=float, default=0.01)
    parser.add_argument("--init_repair_bright_max_count", type=int, default=15000)
    parser.add_argument(
        "--init_repair_bright_target",
        choices=["non_children", "children_only", "all"],
        default="non_children",
    )
    parser.add_argument("--init_repair_bright_min_opacity", type=float, default=0.06)
    parser.add_argument("--init_repair_bright_max_effective_scale_ratio", type=float, default=0.0025)
    parser.add_argument("--init_repair_bright_luma_quantile", type=float, default=0.995)
    parser.add_argument("--init_repair_bright_min_local_luma_ratio", type=float, default=1.8)
    parser.add_argument("--init_repair_bright_min_color_delta", type=float, default=0.18)
    parser.add_argument("--init_repair_bright_neighbor_k", type=int, default=8)
    parser.add_argument("--init_repair_bright_expand_scale_multiplier", type=float, default=1.35)
    parser.add_argument("--init_repair_bright_smallest_axis_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--init_repair_bright_opacity_scale", type=float, default=0.6)
    parser.add_argument("--init_repair_bright_dc_scale", type=float, default=0.82)
    parser.add_argument("--init_repair_bright_rest_scale", type=float, default=0.4)
    parser.add_argument("--init_repair_bright_filter_scale", type=float, default=0.5)
    parser.add_argument("--init_repair_bright_filter_cap_ratio", type=float, default=0.001)
    parser.add_argument("--render_sanity", action="store_true")
    parser.add_argument("--render_sanity_images_subdir", default="images_2")
    parser.add_argument("--render_sanity_resolution", type=int, default=1)
    parser.add_argument("--debug_dump_prepare_stages", action="store_true")
    parser.add_argument("--debug_dump_canonicalize_substages", action="store_true")
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve() if args.scene_root else None
    sof_ref_model = Path(args.sof_ref_model).expanduser().resolve() if args.sof_ref_model else None
    input_iteration = _resolve_input_iteration(mip_model_path, int(args.iteration))
    output_iteration = input_iteration if int(args.output_iteration) < 0 else int(args.output_iteration)
    input_ply = _resolve_input_ply(mip_model_path, input_iteration)

    gaussians = GaussianModel(3, use_SBs=False)
    gaussians.load_ply(str(input_ply))
    scene_stats = _load_reference_scene_stats(
        sof_ref_model,
        quantile_low=float(args.aabb_quantile_low),
        quantile_high=float(args.aabb_quantile_high),
    )
    debug_stage_writer = None
    if args.debug_dump_prepare_stages:
        def _writer(
            stage_name: str,
            stage_index: int,
            *,
            xyz: torch.Tensor,
            features_dc: torch.Tensor,
            features_rest: torch.Tensor,
            opacity: torch.Tensor,
            scaling: torch.Tensor,
            rotation: torch.Tensor,
            filter_3d: torch.Tensor,
            output_source_idx: torch.Tensor,
            child_output_mask: torch.Tensor | None = None,
            softened_output_mask: torch.Tensor | None = None,
            extra_summary: dict | None = None,
        ) -> None:
            _save_debug_stage_snapshot(
                output_model_path=output_model_path,
                output_iteration=output_iteration,
                source_model_path=mip_model_path,
                template=gaussians,
                stage_name=stage_name,
                stage_index=stage_index,
                xyz=xyz,
                features_dc=features_dc,
                features_rest=features_rest,
                opacity=opacity,
                scaling=scaling,
                rotation=rotation,
                filter_3d=filter_3d,
                output_source_idx=output_source_idx,
                child_output_mask=child_output_mask,
                softened_output_mask=softened_output_mask,
                extra_summary=extra_summary,
            )

        debug_stage_writer = _writer
    sanitize_summary = _sanitize_gaussians(
        gaussians,
        scene_stats=scene_stats,
        opacity_min=float(args.opacity_raw_min),
        opacity_max=float(args.opacity_raw_max),
        scale_min_ratio=float(args.scale_min_ratio),
        scale_max_ratio=float(args.scale_max_ratio),
        activated_scale_detect_max=float(args.activated_scale_detect_max),
        opacity_compensate_scale_shrink=str(args.opacity_compensate_scale_shrink),
        opacity_compensation_power=float(args.opacity_compensation_power),
        min_opacity_compensation_scale=float(args.min_opacity_compensation_scale),
        max_compensated_opacity=float(args.max_compensated_opacity),
        scale_clamp_mode=str(args.scale_clamp_mode),
        feature_clip=float(args.feature_clip),
        use_aabb_filter=bool(args.use_aabb_filter),
        aabb_margin_ratio=float(args.aabb_margin_ratio),
        filter_mode=str(args.filter_mode),
        filter_constant=float(args.filter_constant),
        init_repair_mode=str(args.init_repair_mode),
        init_repair_max_fraction=float(args.init_repair_max_fraction),
        init_repair_max_count=int(args.init_repair_max_count),
        init_repair_min_opacity=float(args.init_repair_min_opacity),
        init_repair_min_effective_scale_ratio=float(args.init_repair_min_effective_scale_ratio),
        init_repair_min_volume_radius_ratio=float(args.init_repair_min_volume_radius_ratio),
        init_repair_min_filter_scale_ratio=float(args.init_repair_min_filter_scale_ratio),
        init_repair_min_full_anisotropy=float(args.init_repair_min_full_anisotropy),
        init_repair_split_count=int(args.init_repair_split_count),
        init_repair_child_layout=str(args.init_repair_child_layout),
        init_repair_child_scale_multiplier=float(args.init_repair_child_scale_multiplier),
        init_repair_child_major_scale_multiplier=float(args.init_repair_child_major_scale_multiplier),
        init_repair_child_opacity_scale=float(args.init_repair_child_opacity_scale),
        init_repair_energy_conserve_mode=str(args.init_repair_energy_conserve_mode),
        init_repair_filter_scale=float(args.init_repair_filter_scale),
        init_repair_filter_cap_ratio=float(args.init_repair_filter_cap_ratio),
        init_repair_offset_scale=float(args.init_repair_offset_scale),
        init_repair_bright_max_fraction=float(args.init_repair_bright_max_fraction),
        init_repair_bright_max_count=int(args.init_repair_bright_max_count),
        init_repair_bright_target=str(args.init_repair_bright_target),
        init_repair_bright_min_opacity=float(args.init_repair_bright_min_opacity),
        init_repair_bright_max_effective_scale_ratio=float(args.init_repair_bright_max_effective_scale_ratio),
        init_repair_bright_luma_quantile=float(args.init_repair_bright_luma_quantile),
        init_repair_bright_min_local_luma_ratio=float(args.init_repair_bright_min_local_luma_ratio),
        init_repair_bright_min_color_delta=float(args.init_repair_bright_min_color_delta),
        init_repair_bright_neighbor_k=int(args.init_repair_bright_neighbor_k),
        init_repair_bright_expand_scale_multiplier=float(args.init_repair_bright_expand_scale_multiplier),
        init_repair_bright_smallest_axis_scale_multiplier=float(args.init_repair_bright_smallest_axis_scale_multiplier),
        init_repair_bright_opacity_scale=float(args.init_repair_bright_opacity_scale),
        init_repair_bright_dc_scale=float(args.init_repair_bright_dc_scale),
        init_repair_bright_rest_scale=float(args.init_repair_bright_rest_scale),
        init_repair_bright_filter_scale=float(args.init_repair_bright_filter_scale),
        init_repair_bright_filter_cap_ratio=float(args.init_repair_bright_filter_cap_ratio),
        debug_dump_canonicalize_substages=bool(args.debug_dump_canonicalize_substages),
        debug_stage_writer=debug_stage_writer,
    )
    keep_mask_cpu = sanitize_summary.pop("keep_mask")
    output_source_idx = sanitize_summary.pop("output_source_idx")
    init_repair_replaced_source_idx = sanitize_summary.pop("init_repair_replaced_source_idx")
    init_repair_child_output_mask = sanitize_summary.pop("init_repair_child_output_mask")
    init_repair_softened_source_idx = sanitize_summary.pop("init_repair_softened_source_idx")
    init_repair_softened_output_mask = sanitize_summary.pop("init_repair_softened_output_mask")
    keep_idx = torch.nonzero(keep_mask_cpu, as_tuple=False).squeeze(1).to(dtype=torch.int64)
    input_count = int(keep_mask_cpu.shape[0])
    output_count = int(gaussians.get_xyz.shape[0])
    mapping = torch.full((input_count,), -1, dtype=torch.int64)
    if output_source_idx.numel() != output_count:
        raise RuntimeError(
            f"output_source_idx has {output_source_idx.numel()} rows but converted model has {output_count} gaussians"
        )
    for output_idx, source_idx in enumerate(output_source_idx.to(dtype=torch.int64).tolist()):
        if 0 <= int(source_idx) < input_count and int(mapping[int(source_idx)].item()) < 0:
            mapping[int(source_idx)] = int(output_idx)

    _copy_render_config(mip_model_path, output_model_path)
    if scene_root is not None:
        write_clean_cfg_args(
            output_model_path=output_model_path,
            scene_root=scene_root,
            images_subdir=str(args.render_sanity_images_subdir),
            sh_degree=int(gaussians.max_sh_degree),
        )
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(output_iteration)}"
    mkdir_p(str(point_dir))
    output_ply = point_dir / "point_cloud.ply"
    gaussians.save_ply(str(output_ply))
    gaussians.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    masks_dir = output_model_path / "masks"
    mkdir_p(str(masks_dir))
    torch.save(keep_idx, masks_dir / "keep_idx.pt")
    torch.save(mapping, masks_dir / "mip_to_sof_index.pt")
    torch.save(output_source_idx.to(dtype=torch.int64), masks_dir / "output_source_idx.pt")
    torch.save(init_repair_replaced_source_idx.to(dtype=torch.int64), masks_dir / "init_repair_replaced_source_idx.pt")
    torch.save(init_repair_child_output_mask.to(dtype=torch.bool), masks_dir / "init_repair_child_output_mask.pt")
    torch.save(init_repair_softened_source_idx.to(dtype=torch.int64), masks_dir / "init_repair_softened_source_idx.pt")
    torch.save(init_repair_softened_output_mask.to(dtype=torch.bool), masks_dir / "init_repair_softened_output_mask.pt")
    torch.save(torch.ones((output_count,), dtype=torch.bool), masks_dir / "detail_mask.pt")
    torch.save(torch.zeros((output_count,), dtype=torch.bool), masks_dir / "surface_mask.pt")

    summary = {
        "mode": "prepare_mipsplatting_sof_input_field",
        "mip_model_path": str(mip_model_path),
        "input_iteration": int(input_iteration),
        "input_ply": str(input_ply),
        "output_model_path": str(output_model_path),
        "output_iteration": int(output_iteration),
        "output_ply": str(output_ply),
        "scene_root": str(scene_root) if scene_root is not None else None,
        "sof_ref_model": str(sof_ref_model) if sof_ref_model is not None else None,
        "use_aabb_filter": bool(args.use_aabb_filter),
        "aabb_margin_ratio": float(args.aabb_margin_ratio),
        "debug_dump_prepare_stages": bool(args.debug_dump_prepare_stages),
        **sanitize_summary,
    }
    meta_path = output_model_path / "meta.json"
    summary_path = output_model_path / "mip_input_field_summary.json"
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (point_dir / "num_gaussians.json").write_text(
        json.dumps(
            {
                "num_gaussians": int(gaussians.get_xyz.shape[0]),
                "source": "mip_splatting",
                "source_iteration": int(input_iteration),
                "source_tag": int(GaussianSourceTag.PRIOR_INJECTED),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    render_sanity_pass = None
    render_sanity_dir = None
    if args.render_sanity:
        if scene_root is None:
            raise ValueError("--render_sanity requires --scene_root")
        cmd = [
            str(args.python_bin),
            str(REPO_ROOT / "render.py"),
            "-m",
            str(output_model_path),
            "-s",
            str(scene_root),
            "-i",
            str(args.render_sanity_images_subdir),
            "-r",
            str(int(args.render_sanity_resolution)),
            "--iteration",
            str(int(output_iteration)),
            "--eval",
            "--skip_train",
            "--data_device",
            "cpu",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        render_sanity_pass = True
        render_sanity_dir = str(output_model_path / "test" / f"ours_{int(output_iteration)}" / "renders")
    summary["render_sanity_pass"] = render_sanity_pass
    summary["render_sanity_dir"] = render_sanity_dir
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
