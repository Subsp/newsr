#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple  # noqa: E402
from train_bounded_surface_alternating_v0 import features_dc_to_rgb  # noqa: E402
from train_mip_to_sof_surface_v0 import (  # noqa: E402
    build_prepared_sr_cache,
    load_model_ply,
    load_train_cameras_only,
    resolve_iteration,
    select_uniform,
)
from utils.prior_injection import index_image_dir  # noqa: E402


MEM_COLOR_READY = 1
MEM_COLOR_STABLE = 2


def _safe_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    return float(value)


def tensor_stats(values: torch.Tensor | np.ndarray | Sequence[float]) -> Dict[str, float | int]:
    if not torch.is_tensor(values):
        values = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    values = values.detach().flatten().float().cpu()
    values = values[torch.isfinite(values)]
    if int(values.numel()) <= 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "min": 0.0,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.quantile(0.50).item()),
        "p90": float(values.quantile(0.90).item()),
        "p95": float(values.quantile(0.95).item()),
        "p99": float(values.quantile(0.99).item()),
        "max": float(values.max().item()),
        "min": float(values.min().item()),
    }


def mask_stats(values: torch.Tensor, mask: torch.Tensor) -> Dict[str, float | int]:
    mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != int(values.reshape(-1).shape[0]):
        return tensor_stats(torch.empty((0,), dtype=torch.float32))
    return tensor_stats(values.reshape(-1)[mask])


def opacity_to_tau(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.detach().float().reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    return -torch.log1p(-alpha)


def blur_hw(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return value
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return F.avg_pool2d(value[None, None], kernel_size=kernel_size, stride=1, padding=pad)[0, 0]


def luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return torch.sum(rgb * weights, dim=0)


def masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    value = value.detach().float()
    if mask is None:
        return float(value.mean().item()) if int(value.numel()) > 0 else 0.0
    mask = mask.to(device=value.device, dtype=value.dtype).clamp(0.0, 1.0)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(0)
    denom = mask.expand_as(value).sum().clamp_min(1e-6)
    return float((value * mask).sum().item() / float(denom.item()))


def load_optional_memory(path: str, device: torch.device) -> Optional[Dict[str, torch.Tensor | Dict[str, int]]]:
    if not str(path).strip():
        return None
    memory_path = Path(path).expanduser().resolve()
    if not memory_path.is_file():
        return None
    payload = torch.load(memory_path, map_location=device)
    return payload


def load_training_params(model_path: Path) -> Dict[str, object]:
    summary_path = model_path / "bounded_surface_alternating_v0_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    params = payload.get("params", {})
    return params if isinstance(params, dict) else {}


def memory_active_masks(memory: Dict[str, torch.Tensor | Dict[str, int]], count: int, device: torch.device) -> Dict[str, torch.Tensor]:
    state = memory.get("state")
    update_count = memory.get("update_count")
    if not torch.is_tensor(state) or not torch.is_tensor(update_count) or int(state.numel()) != count:
        empty = torch.zeros((count,), dtype=torch.bool, device=device)
        return {"ready": empty, "stable": empty, "active": empty}
    state = state.to(device=device, dtype=torch.int32).reshape(-1)
    update_count = update_count.to(device=device, dtype=torch.int32).reshape(-1)
    ready = (state == MEM_COLOR_READY) & (update_count > 0)
    stable = (state == MEM_COLOR_STABLE) & (update_count > 0)
    active = ready | stable
    return {"ready": ready, "stable": stable, "active": active}


def build_appearance_target_from_memory(
    target: torch.Tensor,
    base: torch.Tensor,
    before_rgb: torch.Tensor,
    *,
    target_mode: str,
    residual_clip: float,
    residual_scale: float,
) -> torch.Tensor:
    mode = str(target_mode)
    if mode == "absolute":
        return target.detach()
    if mode == "residual_clipped":
        residual = target.detach() - base.detach()
        clip = float(residual_clip)
        if clip > 0.0:
            residual = torch.clamp(residual, min=-clip, max=clip)
        return torch.clamp(before_rgb.detach() + float(residual_scale) * residual, min=0.0, max=1.0)
    raise ValueError(f"Unsupported appearance_target_mode: {mode}")


def audit_memory_targets(
    memory: Optional[Dict[str, torch.Tensor | Dict[str, int]]],
    before_rgb: torch.Tensor,
    after_rgb: torch.Tensor,
    *,
    appearance_target_mode: str,
    appearance_residual_clip: float,
    appearance_residual_scale: float,
) -> Dict[str, object]:
    count = int(before_rgb.shape[0])
    device = before_rgb.device
    if memory is None:
        return {"available": False, "reason": "memory_path missing or not found"}

    masks = memory_active_masks(memory, count, device)
    target = memory.get("target_rgb")
    base = memory.get("base_rgb")
    confidence = memory.get("confidence")
    disagreement = memory.get("disagreement")
    update_count = memory.get("update_count")
    if not torch.is_tensor(target) or not torch.is_tensor(base) or int(target.shape[0]) != count:
        return {"available": False, "reason": "memory tensor count mismatch"}

    target = target.to(device=device, dtype=torch.float32).reshape(count, 3)
    base = base.to(device=device, dtype=torch.float32).reshape(count, 3)
    appearance_target = build_appearance_target_from_memory(
        target,
        base,
        before_rgb,
        target_mode=str(appearance_target_mode),
        residual_clip=float(appearance_residual_clip),
        residual_scale=float(appearance_residual_scale),
    )
    target_base_l1 = torch.mean(torch.abs(target - base), dim=1)
    target_before_l1 = torch.mean(torch.abs(target - before_rgb), dim=1)
    after_target_l1 = torch.mean(torch.abs(after_rgb - target), dim=1)
    appearance_target_base_l1 = torch.mean(torch.abs(appearance_target - base), dim=1)
    appearance_target_before_l1 = torch.mean(torch.abs(appearance_target - before_rgb), dim=1)
    after_appearance_target_l1 = torch.mean(torch.abs(after_rgb - appearance_target), dim=1)
    dc_update_l1 = torch.mean(torch.abs(after_rgb - before_rgb), dim=1)

    active = masks["active"]
    stable = masks["stable"]
    ready = masks["ready"]
    result: Dict[str, object] = {
        "available": True,
        "count": count,
        "ready_count": int(ready.sum().item()),
        "stable_count": int(stable.sum().item()),
        "active_count": int(active.sum().item()),
        "active_ratio": float(active.float().mean().item()) if count > 0 else 0.0,
        "appearance_target_mode": str(appearance_target_mode),
        "appearance_residual_clip": float(appearance_residual_clip),
        "appearance_residual_scale": float(appearance_residual_scale),
        "target_minus_base_l1_active": mask_stats(target_base_l1, active),
        "target_minus_before_dc_l1_active": mask_stats(target_before_l1, active),
        "after_dc_minus_target_l1_active": mask_stats(after_target_l1, active),
        "appearance_target_minus_base_l1_active": mask_stats(appearance_target_base_l1, active),
        "appearance_target_minus_before_dc_l1_active": mask_stats(appearance_target_before_l1, active),
        "after_dc_minus_appearance_target_l1_active": mask_stats(after_appearance_target_l1, active),
        "dc_update_l1_active": mask_stats(dc_update_l1, active),
        "dc_update_l1_stable": mask_stats(dc_update_l1, stable),
    }
    if torch.is_tensor(confidence):
        result["confidence_active"] = mask_stats(confidence.to(device=device).float().reshape(-1), active)
    if torch.is_tensor(disagreement):
        result["disagreement_active"] = mask_stats(disagreement.to(device=device).float().reshape(-1), active)
    if torch.is_tensor(update_count):
        result["update_count_active"] = mask_stats(update_count.to(device=device).float().reshape(-1), active)
    state_counts = memory.get("state_counts")
    if isinstance(state_counts, dict):
        result["state_counts"] = {str(key): int(value) for key, value in state_counts.items()}
    return result


@torch.no_grad()
def audit_gaussian_updates(
    before,
    after,
    memory: Optional[Dict[str, torch.Tensor | Dict[str, int]]],
    *,
    appearance_target_mode: str,
    appearance_residual_clip: float,
    appearance_residual_scale: float,
) -> Dict[str, object]:
    before_rgb = features_dc_to_rgb(before._features_dc.detach()).float().clamp(0.0, 1.0)
    after_rgb = features_dc_to_rgb(after._features_dc.detach()).float().clamp(0.0, 1.0)
    same_count = int(before_rgb.shape[0]) == int(after_rgb.shape[0])
    result: Dict[str, object] = {
        "before_count": int(before_rgb.shape[0]),
        "after_count": int(after_rgb.shape[0]),
        "same_count": bool(same_count),
    }
    if not same_count:
        return result

    dc_delta_l1 = torch.mean(torch.abs(after_rgb - before_rgb), dim=1)
    before_alpha = before.get_opacity.detach().float().reshape(-1)
    after_alpha = after.get_opacity.detach().float().reshape(-1)
    alpha_delta = after_alpha - before_alpha
    before_tau = opacity_to_tau(before_alpha)
    after_tau = opacity_to_tau(after_alpha)
    tau_delta = after_tau - before_tau

    result.update(
        {
            "dc_update_l1_all": tensor_stats(dc_delta_l1),
            "opacity_before_all": tensor_stats(before_alpha),
            "opacity_after_all": tensor_stats(after_alpha),
            "opacity_delta_all": tensor_stats(alpha_delta),
            "tau_before_all": tensor_stats(before_tau),
            "tau_after_all": tensor_stats(after_tau),
            "tau_delta_all": tensor_stats(tau_delta),
            "memory_target_audit": audit_memory_targets(
                memory,
                before_rgb,
                after_rgb,
                appearance_target_mode=str(appearance_target_mode),
                appearance_residual_clip=float(appearance_residual_clip),
                appearance_residual_scale=float(appearance_residual_scale),
            ),
        }
    )
    if memory is not None:
        masks = memory_active_masks(memory, int(before_rgb.shape[0]), before_rgb.device)
        active = masks["active"]
        stable = masks["stable"]
        result["dc_update_l1_memory_active"] = mask_stats(dc_delta_l1, active)
        result["dc_update_l1_memory_stable"] = mask_stats(dc_delta_l1, stable)
        result["opacity_memory_active"] = mask_stats(after_alpha, active)
        result["tau_memory_active"] = mask_stats(after_tau, active)
    return result


@torch.no_grad()
def audit_visibility_proxy(model, cameras: Sequence[object], active_mask: Optional[torch.Tensor]) -> Dict[str, object]:
    count = int(model.get_xyz.shape[0])
    device = model.get_xyz.device
    visible_count = torch.zeros((count,), dtype=torch.float32, device=device)
    radii_sum = torch.zeros((count,), dtype=torch.float32, device=device)
    background = torch.zeros((3,), dtype=torch.float32, device=device)

    for camera in tqdm(cameras, desc="audit visibility proxy"):
        pkg = render_simple(camera, model, background)
        visibility = pkg["visibility_filter"].to(device=device, dtype=torch.bool).reshape(-1)
        radii = pkg["radii"].to(device=device, dtype=torch.float32).reshape(-1)
        if int(visibility.numel()) != count:
            continue
        visible_count += visibility.float()
        radii_sum += torch.where(visibility, radii, torch.zeros_like(radii))

    result: Dict[str, object] = {
        "views": int(len(cameras)),
        "visible_view_count_all": tensor_stats(visible_count),
        "radii_sum_all": tensor_stats(radii_sum),
    }
    if active_mask is not None and int(active_mask.numel()) == count:
        active_mask = active_mask.to(device=device, dtype=torch.bool)
        result["visible_view_count_active"] = mask_stats(visible_count, active_mask)
        result["radii_sum_active"] = mask_stats(radii_sum, active_mask)
        result["active_visible_ratio"] = float(((visible_count > 0) & active_mask).float().sum().item() / max(float(active_mask.float().sum().item()), 1.0))
    return result


@torch.no_grad()
def audit_render_response(
    before,
    after,
    cameras: Sequence[object],
    sr_cache: Optional[Sequence[Dict[str, torch.Tensor | str | float | None]]],
    *,
    haze_kernel: int,
) -> Dict[str, object]:
    device = after.get_xyz.device
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    view_items: List[Dict[str, float | str]] = []

    for idx, camera in enumerate(tqdm(cameras, desc="audit render response")):
        before_pkg = render_simple(camera, before, background)
        after_pkg = render_simple(camera, after, background)
        before_rgb = before_pkg["render"].detach().float().clamp(0.0, 1.0)
        after_rgb = after_pkg["render"].detach().float().clamp(0.0, 1.0)
        before_alpha = before_pkg["alpha"].detach().float().clamp(0.0, 1.0)
        after_alpha = after_pkg["alpha"].detach().float().clamp(0.0, 1.0)

        render_delta = torch.abs(after_rgb - before_rgb)
        before_premul = before_rgb * before_alpha
        after_premul = after_rgb * after_alpha
        premul_delta_luma = luma(after_premul - before_premul)
        lowfreq_delta = blur_hw(premul_delta_luma, int(haze_kernel))

        item: Dict[str, float | str] = {
            "view": str(getattr(camera, "image_name", idx)),
            "render_delta_l1": float(render_delta.mean().item()),
            "premul_delta_luma_l1": float(torch.abs(premul_delta_luma).mean().item()),
            "haze_positive_lowfreq": float(torch.relu(lowfreq_delta).mean().item()),
            "haze_abs_lowfreq": float(torch.abs(lowfreq_delta).mean().item()),
            "alpha_delta_l1": float(torch.abs(after_alpha - before_alpha).mean().item()),
        }

        if sr_cache is not None and idx < len(sr_cache):
            target = sr_cache[idx]
            prior_rgb = target["prior_rgb"]
            prior_mask = target["prior_mask"]
            if torch.is_tensor(prior_rgb) and torch.is_tensor(prior_mask):
                prior_rgb = prior_rgb.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                prior_mask = prior_mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                before_err = torch.mean(torch.abs(before_rgb - prior_rgb), dim=0)
                after_err = torch.mean(torch.abs(after_rgb - prior_rgb), dim=0)
                item["prior_mask_mean"] = float(prior_mask.mean().item())
                item["prior_l1_before_masked"] = masked_mean(before_err, prior_mask)
                item["prior_l1_after_masked"] = masked_mean(after_err, prior_mask)
                item["prior_l1_improvement_masked"] = float(item["prior_l1_before_masked"]) - float(item["prior_l1_after_masked"])
        view_items.append(item)

    def collect(key: str) -> List[float]:
        return [float(item[key]) for item in view_items if key in item]

    aggregate = {
        "views": int(len(view_items)),
        "render_delta_l1": tensor_stats(collect("render_delta_l1")),
        "premul_delta_luma_l1": tensor_stats(collect("premul_delta_luma_l1")),
        "haze_positive_lowfreq": tensor_stats(collect("haze_positive_lowfreq")),
        "haze_abs_lowfreq": tensor_stats(collect("haze_abs_lowfreq")),
        "alpha_delta_l1": tensor_stats(collect("alpha_delta_l1")),
    }
    if any("prior_l1_improvement_masked" in item for item in view_items):
        aggregate["prior_l1_before_masked"] = tensor_stats(collect("prior_l1_before_masked"))
        aggregate["prior_l1_after_masked"] = tensor_stats(collect("prior_l1_after_masked"))
        aggregate["prior_l1_improvement_masked"] = tensor_stats(collect("prior_l1_improvement_masked"))
        aggregate["prior_mask_mean"] = tensor_stats(collect("prior_mask_mean"))
    return {"aggregate": aggregate, "views": view_items}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether surface memory changes enter DC and final renders.")
    parser.add_argument("--scene_root", type=str, required=True)
    parser.add_argument("--before_model_path", type=str, required=True)
    parser.add_argument("--after_model_path", type=str, required=True)
    parser.add_argument("--before_iteration", type=int, default=-1)
    parser.add_argument("--after_iteration", type=int, default=-1)
    parser.add_argument("--images_subdir", type=str, default="images_2")
    parser.add_argument("--max_views", type=int, default=24)
    parser.add_argument("--memory_path", type=str, default="")
    parser.add_argument("--sr_prior_root", type=str, default="")
    parser.add_argument("--sr_anchor_root", type=str, default="")
    parser.add_argument("--sr_prior_mask_dir", type=str, default="")
    parser.add_argument("--sr_prior_mask_suffix", type=str, default="")
    parser.add_argument("--sr_prior_consistency_threshold", type=float, default=0.0)
    parser.add_argument("--sr_prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--haze_kernel", type=int, default=31)
    parser.add_argument("--appearance_target_mode", choices=["", "absolute", "residual_clipped"], default="")
    parser.add_argument("--appearance_residual_clip", type=float, default=-1.0)
    parser.add_argument("--appearance_residual_scale", type=float, default=-1.0)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    before_model_path = Path(args.before_model_path).expanduser().resolve()
    after_model_path = Path(args.after_model_path).expanduser().resolve()
    before_iter = resolve_iteration(before_model_path, int(args.before_iteration))
    after_iter = resolve_iteration(after_model_path, int(args.after_iteration))
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else after_model_path / "audit_surface_memory_readout_v0"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_params = load_training_params(after_model_path)
    appearance_target_mode = str(args.appearance_target_mode).strip()
    if not appearance_target_mode:
        appearance_target_mode = str(training_params.get("appearance_target_mode", "absolute"))
    appearance_residual_clip = (
        float(args.appearance_residual_clip)
        if float(args.appearance_residual_clip) >= 0.0
        else float(training_params.get("appearance_residual_clip", 0.0))
    )
    appearance_residual_scale = (
        float(args.appearance_residual_scale)
        if float(args.appearance_residual_scale) >= 0.0
        else float(training_params.get("appearance_residual_scale", 1.0))
    )

    before = load_model_ply(before_model_path, before_iter, sh_degree=3)
    after = load_model_ply(after_model_path, after_iter, sh_degree=3)
    memory = load_optional_memory(str(args.memory_path), device=after.get_xyz.device)

    cameras = load_train_cameras_only(scene_root, after_model_path, str(args.images_subdir))
    cameras = select_uniform(cameras, int(args.max_views))

    sr_cache = None
    if str(args.sr_prior_root).strip() and str(args.sr_anchor_root).strip():
        sr_prior_index = index_image_dir(Path(args.sr_prior_root).expanduser().resolve())
        sr_anchor_index = index_image_dir(Path(args.sr_anchor_root).expanduser().resolve())
        mask_dir = Path(args.sr_prior_mask_dir).expanduser().resolve() if str(args.sr_prior_mask_dir).strip() else None
        background = torch.zeros((3,), dtype=torch.float32, device=after.get_xyz.device)
        cameras, sr_cache = build_prepared_sr_cache(
            cameras,
            before,
            background,
            sr_prior_index=sr_prior_index,
            sr_anchor_index=sr_anchor_index,
            sr_prior_mask_dir=mask_dir,
            sr_prior_mask_suffix=str(args.sr_prior_mask_suffix),
            prior_consistency_threshold=float(args.sr_prior_consistency_threshold),
            prior_mask_floor=float(args.sr_prior_mask_floor),
        )

    gaussian_audit = audit_gaussian_updates(
        before,
        after,
        memory,
        appearance_target_mode=str(appearance_target_mode),
        appearance_residual_clip=float(appearance_residual_clip),
        appearance_residual_scale=float(appearance_residual_scale),
    )
    active_mask = None
    if memory is not None and int(before.get_xyz.shape[0]) == int(after.get_xyz.shape[0]):
        active_mask = memory_active_masks(memory, int(after.get_xyz.shape[0]), after.get_xyz.device)["active"]
    visibility_audit = audit_visibility_proxy(after, cameras, active_mask)
    render_audit = audit_render_response(before, after, cameras, sr_cache, haze_kernel=int(args.haze_kernel))

    summary = {
        "version": "surface_memory_readout_audit_v0",
        "scene_root": str(scene_root),
        "before_model_path": str(before_model_path),
        "after_model_path": str(after_model_path),
        "before_iteration": int(before_iter),
        "after_iteration": int(after_iter),
        "images_subdir": str(args.images_subdir),
        "max_views": int(args.max_views),
        "memory_path": str(args.memory_path) if str(args.memory_path).strip() else None,
        "sr_prior_root": str(args.sr_prior_root) if str(args.sr_prior_root).strip() else None,
        "sr_anchor_root": str(args.sr_anchor_root) if str(args.sr_anchor_root).strip() else None,
        "appearance_target_mode": str(appearance_target_mode),
        "appearance_residual_clip": float(appearance_residual_clip),
        "appearance_residual_scale": float(appearance_residual_scale),
        "gaussian_update_audit": gaussian_audit,
        "visibility_proxy_audit": visibility_audit,
        "render_response_audit": render_audit,
    }

    summary_path = output_dir / "surface_memory_readout_audit_v0_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[audit-surface-memory-readout-v0] summary: {summary_path}")


if __name__ == "__main__":
    main()
