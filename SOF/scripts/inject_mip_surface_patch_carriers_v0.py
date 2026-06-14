from __future__ import annotations

import argparse
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import build_rotation
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=white_background,
        data_device="cpu",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if int(iteration) >= 0:
        return int(iteration)
    point_root = model_path / "point_cloud"
    candidates: List[int] = []
    for child in point_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_root}")
    return max(candidates)


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def _parse_csv_ints(value: str | None) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    out: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _select_views(items: Sequence[object], max_items: int, view_indices: Sequence[int]) -> Tuple[List[object], List[int]]:
    if view_indices:
        selected: List[object] = []
        selected_indices: List[int] = []
        for idx in view_indices:
            if int(idx) < 0 or int(idx) >= len(items):
                raise IndexError(f"view index {idx} out of range for split with {len(items)} cameras")
            selected.append(items[int(idx)])
            selected_indices.append(int(idx))
        return selected, selected_indices
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items), list(range(len(items)))
    ids = np.unique(np.linspace(0, len(items) - 1, num=int(max_items), dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()], [int(idx) for idx in ids.tolist()]


def _iter_views(scene: Scene, split: str) -> List[object]:
    if split == "train":
        return list(scene.getTrainCameras())
    if split == "test":
        return list(scene.getTestCameras())
    if split == "both":
        return list(scene.getTrainCameras()) + list(scene.getTestCameras())
    raise ValueError(f"Unsupported split: {split}")


def _rgb_to_luma(rgb_chw: torch.Tensor) -> torch.Tensor:
    return rgb_chw[0:1] * 0.299 + rgb_chw[1:2] * 0.587 + rgb_chw[2:3] * 0.114


def _box_blur(x_chw: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x_chw
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    x = x_chw.unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=kernel, stride=1)[0]


def _normalize_map(value: torch.Tensor, percentile: float, eps: float = 1e-6) -> torch.Tensor:
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(value)
    finite = torch.isfinite(flat)
    if not bool(torch.any(finite)):
        return torch.zeros_like(value)
    scale = torch.quantile(flat[finite].float(), float(percentile)).clamp_min(float(eps))
    return torch.clamp(value / scale.to(device=value.device, dtype=value.dtype), 0.0, 1.0)


def _downsample_map(value_hw: torch.Tensor, coarse_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(coarse_hw[0]), int(coarse_hw[1])
    value = value_hw[None, None].float()
    coarse = F.interpolate(value, size=(target_h, target_w), mode="area")
    return coarse[0, 0].detach().cpu().numpy().astype(np.float32)


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _save_heatmap(path: Path, value_hw: torch.Tensor | np.ndarray) -> None:
    if isinstance(value_hw, torch.Tensor):
        value = value_hw.detach().float().cpu().numpy()
    else:
        value = np.asarray(value_hw, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        norm = np.zeros_like(value, dtype=np.float32)
    else:
        vmax = max(float(np.percentile(finite, 99.0)), 1e-6)
        norm = np.clip(value / vmax, 0.0, 1.0)
    r = np.clip(1.7 * norm, 0.0, 1.0)
    g = np.clip(1.7 * norm - 0.45, 0.0, 1.0)
    b = np.clip(1.4 * norm - 0.9, 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _save_overlay(path: Path, rgb_chw: torch.Tensor, heat_hw: torch.Tensor) -> None:
    rgb = rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    heat = heat_hw.detach().float().cpu().numpy()
    heat = np.clip(heat / max(float(np.percentile(heat, 99.0)), 1e-6), 0.0, 1.0)
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    red[..., 1] = 0.10
    red[..., 2] = 0.05
    overlay = rgb * (1.0 - 0.50 * heat[..., None]) + red * (0.50 * heat[..., None])
    Image.fromarray(np.clip(overlay * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probability / torch.clamp(1.0 - probability, min=1e-6))


def _opacity_compensation_metric(scales: torch.Tensor, mode: str) -> torch.Tensor:
    safe = torch.clamp(scales, min=1e-12)
    if mode == "volume":
        return torch.prod(safe, dim=1, keepdim=True)
    sorted_scales = torch.sort(safe, dim=1).values
    return torch.prod(sorted_scales[:, -2:], dim=1, keepdim=True)


def _grid_pattern(side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = max(2, int(side))
    coords = torch.linspace(-1.0, 1.0, steps=side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    zz = torch.zeros_like(xx)
    return torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=1)


def _ensure_tracking_state(gaussians: GaussianModel) -> None:
    count = int(gaussians.get_xyz.shape[0])
    if not hasattr(gaussians, "_source_tag") or int(gaussians._source_tag.shape[0]) != count:
        gaussians.init_tracking_state(count)


def _make_static_gaussian_model(
    *,
    base: GaussianModel,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    tracking_state: Dict[str, torch.Tensor],
) -> GaussianModel:
    model = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    model.active_sh_degree = int(base.active_sh_degree)
    model.spatial_lr_scale = float(base.spatial_lr_scale)
    model._xyz = nn.Parameter(xyz.detach().clone().requires_grad_(False))
    model._features_dc = nn.Parameter(features_dc.detach().clone().requires_grad_(False))
    model._features_rest = nn.Parameter(features_rest.detach().clone().requires_grad_(False))
    model._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(False))
    model._scaling = nn.Parameter(scaling.detach().clone().requires_grad_(False))
    model._rotation = nn.Parameter(rotation.detach().clone().requires_grad_(False))
    model.filter_3D = filter_3d.detach().clone()
    count = int(xyz.shape[0])
    device = xyz.device
    model.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=device)
    model.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.denom = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.init_tracking_state(count)
    model._source_tag = tracking_state["source_tag"].to(device=device, dtype=torch.int32)
    model._seed_id = tracking_state["seed_id"].to(device=device, dtype=torch.int64)
    model._generation = tracking_state["generation"].to(device=device, dtype=torch.int32)
    model._edge_touched = tracking_state["edge_touched"].to(device=device, dtype=torch.bool)
    model._edge_touch_iter = tracking_state["edge_touch_iter"].to(device=device, dtype=torch.int32)
    return model


def _accumulate_from_visibility(
    gaussian_ids: np.ndarray,
    weights: np.ndarray,
    metric: np.ndarray,
    sums: np.ndarray,
    denoms: np.ndarray,
) -> None:
    valid = gaussian_ids >= 0
    if not np.any(valid):
        return
    ids = gaussian_ids[valid].astype(np.int64, copy=False)
    w = weights[valid].astype(np.float64, copy=False)
    metric_expanded = np.repeat(metric[..., None], weights.shape[-1], axis=-1)
    vals = metric_expanded[valid].astype(np.float64, copy=False) * w
    np.add.at(sums, ids, vals)
    np.add.at(denoms, ids, w)


def _stats(values: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        arr = arr[np.asarray(mask).reshape(-1)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _build_deficit_maps(
    *,
    current_render: torch.Tensor,
    current_alpha: torch.Tensor,
    mip_render: torch.Tensor,
    mip_alpha: torch.Tensor,
    alpha_margin: float,
    min_mip_alpha: float,
    smooth_kernel: int,
    residual_percentile: float,
    highpass_percentile: float,
) -> Dict[str, torch.Tensor]:
    alpha_deficit = torch.relu(mip_alpha - current_alpha - float(alpha_margin))
    alpha_support = (mip_alpha >= float(min_mip_alpha)).float()
    mip_luma = _rgb_to_luma(mip_render)
    residual = torch.mean(torch.abs(mip_render - current_render), dim=0, keepdim=True)
    residual_unit = _normalize_map(residual, float(residual_percentile))
    surface_highpass = torch.abs(mip_luma - _box_blur(mip_luma, int(smooth_kernel)))
    smooth_surface = 1.0 - _normalize_map(surface_highpass, float(highpass_percentile))
    smooth_surface = torch.clamp(smooth_surface, 0.0, 1.0)
    deficit_map = alpha_deficit * alpha_support * (0.40 + 0.60 * residual_unit) * (0.35 + 0.65 * smooth_surface)
    return {
        "alpha_deficit": alpha_deficit[0],
        "residual": residual[0],
        "smooth_surface": smooth_surface[0],
        "deficit_map": deficit_map[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject MIP-derived surface patch carriers into thin or under-covered SOF regions.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mip_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--mip_iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--view_indices", default="")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--visibility_downsample", type=int, default=4)
    parser.add_argument("--visibility_topk", type=int, default=8)
    parser.add_argument("--visibility_max_visible", type=int, default=80000)
    parser.add_argument("--visibility_max_patch_radius", type=int, default=2)
    parser.add_argument("--alpha_margin", type=float, default=0.02)
    parser.add_argument("--min_mip_alpha", type=float, default=0.08)
    parser.add_argument("--surface_smooth_kernel", type=int, default=11)
    parser.add_argument("--residual_percentile", type=float, default=0.990)
    parser.add_argument("--highpass_percentile", type=float, default=0.990)
    parser.add_argument("--select_quantile", type=float, default=0.985)
    parser.add_argument("--min_deficit_score", type=float, default=0.05)
    parser.add_argument("--min_planarity", type=float, default=5.0)
    parser.add_argument("--min_opacity", type=float, default=0.03)
    parser.add_argument("--min_tangent_extent_ratio", type=float, default=0.0008)
    parser.add_argument("--max_candidate_fraction", type=float, default=0.010)
    parser.add_argument("--max_candidate_count", type=int, default=12000)
    parser.add_argument("--patch_grid_side", type=int, default=2)
    parser.add_argument("--patch_offset_scale", type=float, default=0.75)
    parser.add_argument("--patch_tangent_scale_multiplier", type=float, default=0.55)
    parser.add_argument("--patch_normal_scale_multiplier", type=float, default=0.35)
    parser.add_argument("--patch_opacity_scale", type=float, default=0.65)
    parser.add_argument("--patch_max_opacity", type=float, default=0.32)
    parser.add_argument("--patch_filter_scale", type=float, default=0.25)
    parser.add_argument("--filter_cap_ratio", type=float, default=0.0005)
    parser.add_argument("--energy_conserve_mode", choices=["none", "area", "volume"], default="area")
    parser.add_argument("--features_rest_scale", type=float, default=0.0)
    parser.add_argument("--num_debug_views", type=int, default=4)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)
    debug_dir = output_model_path / "coverage_debug_views"
    debug_dir.mkdir(parents=True, exist_ok=True)

    view_indices = _parse_csv_ints(str(args.view_indices))
    current_iteration = _resolve_model_iteration(model_path, int(args.iteration))
    mip_iteration = _resolve_model_iteration(mip_model_path, int(args.mip_iteration))

    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )
    mip_dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(mip_model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )

    current_gaussians = GaussianModel(dataset.sh_degree)
    current_scene = Scene(dataset, current_gaussians, load_iteration=current_iteration, shuffle=False, skip_test=False, skip_train=False)
    current_loaded_iter = int(current_scene.loaded_iter if current_scene.loaded_iter is not None else current_iteration)
    _ensure_tracking_state(current_gaussians)

    mip_gaussians = GaussianModel(mip_dataset.sh_degree)
    mip_scene = Scene(mip_dataset, mip_gaussians, load_iteration=mip_iteration, shuffle=False, skip_test=False, skip_train=False)
    mip_loaded_iter = int(mip_scene.loaded_iter if mip_scene.loaded_iter is not None else mip_iteration)
    _ensure_tracking_state(mip_gaussians)

    current_views_all = _iter_views(current_scene, str(args.split))
    mip_views_all = _iter_views(mip_scene, str(args.split))
    if len(current_views_all) != len(mip_views_all):
        raise RuntimeError("Current model and MIP model expose different numbers of cameras for the selected split")
    current_views, selected_indices = _select_views(current_views_all, int(args.max_views), view_indices)
    mip_views = [mip_views_all[int(idx)] for idx in selected_indices]
    if not current_views:
        raise RuntimeError(f"No views selected for split={args.split}")

    current_gaussians.compute_3D_filter(current_scene.getTrainCameras().copy(), CUDA=False)
    mip_gaussians.compute_3D_filter(mip_scene.getTrainCameras().copy(), CUDA=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    mip_count = int(mip_gaussians.get_xyz.shape[0])
    deficit_sum = np.zeros((mip_count,), dtype=np.float64)
    metric_weight = np.zeros((mip_count,), dtype=np.float64)
    alpha_sum = np.zeros((mip_count,), dtype=np.float64)
    smooth_sum = np.zeros((mip_count,), dtype=np.float64)
    visible_count = np.zeros((mip_count,), dtype=np.int64)
    view_summaries: List[Dict[str, object]] = []

    for local_idx, source_idx in enumerate(tqdm(selected_indices, desc="coverage-deficit")):
        current_view = current_views[local_idx]
        mip_view = mip_views[local_idx]
        current_render_pkg = render_simple(current_view, current_gaussians, background)
        mip_render_pkg = render_simple(mip_view, mip_gaussians, background)

        current_render = current_render_pkg["render"][:3].detach().clamp(0.0, 1.0)
        current_alpha = current_render_pkg["alpha"].detach().clamp(0.0, 1.0)
        mip_render = mip_render_pkg["render"][:3].detach().clamp(0.0, 1.0)
        mip_alpha = mip_render_pkg["alpha"].detach().clamp(0.0, 1.0)
        maps = _build_deficit_maps(
            current_render=current_render,
            current_alpha=current_alpha,
            mip_render=mip_render,
            mip_alpha=mip_alpha,
            alpha_margin=float(args.alpha_margin),
            min_mip_alpha=float(args.min_mip_alpha),
            smooth_kernel=int(args.surface_smooth_kernel),
            residual_percentile=float(args.residual_percentile),
            highpass_percentile=float(args.highpass_percentile),
        )

        image_hw = (int(mip_view.image_height), int(mip_view.image_width))
        records = build_coarse_visibility_records(
            mip_gaussians,
            [mip_view],
            [mip_render_pkg],
            image_hw=image_hw,
            cfg=VisibilityRecordConfig(
                downsample=int(args.visibility_downsample),
                topk=int(args.visibility_topk),
                max_visible_per_view=int(args.visibility_max_visible),
                max_patch_radius=int(args.visibility_max_patch_radius),
            ),
        )
        coarse_h, coarse_w = [int(v) for v in records["coarse_hw"].tolist()]
        ids = records["gaussian_ids"][0, 0].numpy()
        weights = records["weights"][0, 0].numpy()
        deficit_coarse = _downsample_map(maps["deficit_map"], (coarse_h, coarse_w))
        alpha_coarse = _downsample_map(maps["alpha_deficit"], (coarse_h, coarse_w))
        smooth_coarse = _downsample_map(maps["smooth_surface"], (coarse_h, coarse_w))
        _accumulate_from_visibility(ids, weights, deficit_coarse, deficit_sum, metric_weight)
        _accumulate_from_visibility(ids, weights, alpha_coarse, alpha_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, smooth_coarse, smooth_sum, np.zeros_like(metric_weight))

        visible = mip_render_pkg["visibility_filter"].detach().cpu().numpy().astype(bool)
        visible_ids = np.flatnonzero(visible)
        if visible_ids.size:
            visible_count[visible_ids] += 1

        view_summaries.append(
            {
                "local_view_index": int(local_idx),
                "source_view_index": int(source_idx),
                "image_name": str(current_view.image_name),
                "deficit_mean": float(maps["deficit_map"].mean().item()),
                "alpha_deficit_mean": float(maps["alpha_deficit"].mean().item()),
                "smooth_surface_mean": float(maps["smooth_surface"].mean().item()),
            }
        )
        if local_idx < int(args.num_debug_views):
            prefix = debug_dir / f"view_{local_idx:03d}_src{source_idx:05d}_{str(current_view.image_name).replace('/', '_')}"
            _save_rgb(prefix.with_name(prefix.name + "_current_render.png"), current_render)
            _save_rgb(prefix.with_name(prefix.name + "_mip_render.png"), mip_render)
            _save_heatmap(prefix.with_name(prefix.name + "_current_alpha_heat.png"), current_alpha[0])
            _save_heatmap(prefix.with_name(prefix.name + "_mip_alpha_heat.png"), mip_alpha[0])
            _save_heatmap(prefix.with_name(prefix.name + "_alpha_deficit_heat.png"), maps["alpha_deficit"])
            _save_heatmap(prefix.with_name(prefix.name + "_smooth_surface_heat.png"), maps["smooth_surface"])
            _save_heatmap(prefix.with_name(prefix.name + "_deficit_heat.png"), maps["deficit_map"])
            _save_overlay(prefix.with_name(prefix.name + "_deficit_overlay.png"), current_render, maps["deficit_map"])

    denom = np.maximum(metric_weight, 1e-8)
    deficit_score = np.clip(deficit_sum / denom, 0.0, 1.0).astype(np.float32)
    alpha_score = np.clip(alpha_sum / denom, 0.0, 1.0).astype(np.float32)
    smooth_score = np.clip(smooth_sum / denom, 0.0, 1.0).astype(np.float32)
    visible_mask = visible_count > 0

    mip_scale = mip_gaussians.get_scaling.detach().float().cpu().numpy()
    if isinstance(mip_gaussians.filter_3D, torch.Tensor) and mip_gaussians.filter_3D.ndim > 0:
        mip_filter_3d = mip_gaussians.filter_3D.detach().float().cpu().numpy().reshape(-1, 1)
    else:
        mip_filter_3d = np.zeros((mip_count, 1), dtype=np.float32)
    effective_scale = np.sqrt(np.square(mip_scale) + np.square(mip_filter_3d)).astype(np.float32)
    sorted_scale = np.sort(np.clip(effective_scale, 1e-8, None), axis=1)
    tangent_extent = np.sqrt(sorted_scale[:, 2] * sorted_scale[:, 1]).astype(np.float32)
    planarity = (sorted_scale[:, 1] / np.clip(sorted_scale[:, 0], 1e-8, None)).astype(np.float32)
    mip_opacity = mip_gaussians.get_opacity.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    scene_extent = max(float(current_scene.cameras_extent), 1e-6)
    min_tangent_extent = scene_extent * float(args.min_tangent_extent_ratio)

    candidate = visible_mask.copy()
    candidate &= deficit_score >= float(args.min_deficit_score)
    candidate &= smooth_score >= 0.15
    candidate &= alpha_score >= 0.02
    candidate &= planarity >= float(args.min_planarity)
    candidate &= tangent_extent >= float(min_tangent_extent)
    candidate &= mip_opacity >= float(args.min_opacity)
    candidate_count_before_cap = int(np.sum(candidate))
    if candidate_count_before_cap > 0:
        threshold = float(np.quantile(deficit_score[candidate], float(args.select_quantile)))
        candidate &= deficit_score >= threshold
    else:
        threshold = 1.0

    before_cap = int(np.sum(candidate))
    cap = before_cap
    if float(args.max_candidate_fraction) > 0.0:
        cap = min(cap, max(1, int(round(float(args.max_candidate_fraction) * float(mip_count)))))
    if int(args.max_candidate_count) > 0:
        cap = min(cap, int(args.max_candidate_count))
    if cap <= 0:
        candidate[:] = False
    elif before_cap > cap:
        candidate_ids = np.flatnonzero(candidate)
        order = np.argsort(-deficit_score[candidate_ids], kind="stable")[:cap]
        capped = np.zeros_like(candidate, dtype=bool)
        capped[candidate_ids[order]] = True
        candidate = capped

    selected_ids_np = np.flatnonzero(candidate).astype(np.int64, copy=False)
    selected_count = int(selected_ids_np.size)

    current_xyz = current_gaussians._xyz.detach()
    current_features_dc = current_gaussians._features_dc.detach()
    current_features_rest = current_gaussians._features_rest.detach()
    current_opacity_raw = current_gaussians._opacity.detach()
    current_scaling_raw = current_gaussians._scaling.detach()
    current_rotation_raw = current_gaussians._rotation.detach()
    if isinstance(current_gaussians.filter_3D, torch.Tensor) and current_gaussians.filter_3D.ndim > 0:
        current_filter_3d = current_gaussians.filter_3D.detach()
    else:
        current_filter_3d = torch.zeros((int(current_xyz.shape[0]), 1), dtype=torch.float32, device=current_xyz.device)
    if tuple(mip_gaussians._features_dc.shape[1:]) != tuple(current_features_dc.shape[1:]):
        raise ValueError(
            f"MIP/current features_dc shape mismatch: {tuple(mip_gaussians._features_dc.shape)} vs {tuple(current_features_dc.shape)}"
        )
    if tuple(mip_gaussians._features_rest.shape[1:]) != tuple(current_features_rest.shape[1:]):
        raise ValueError(
            f"MIP/current features_rest shape mismatch: {tuple(mip_gaussians._features_rest.shape)} vs {tuple(current_features_rest.shape)}"
        )

    if selected_count <= 0:
        injected_output_mask = torch.zeros((int(current_xyz.shape[0]),), dtype=torch.bool, device=current_xyz.device)
        injected_source_mip_idx = torch.empty((0,), dtype=torch.int64, device=current_xyz.device)
        tracking_state = {
            "source_tag": current_gaussians._source_tag.detach().clone(),
            "seed_id": current_gaussians._seed_id.detach().clone(),
            "generation": current_gaussians._generation.detach().clone(),
            "edge_touched": current_gaussians._edge_touched.detach().clone(),
            "edge_touch_iter": current_gaussians._edge_touch_iter.detach().clone(),
        }
        merged = _make_static_gaussian_model(
            base=current_gaussians,
            xyz=current_xyz,
            features_dc=current_features_dc,
            features_rest=current_features_rest,
            opacity=current_opacity_raw,
            scaling=current_scaling_raw,
            rotation=current_rotation_raw,
            filter_3d=current_filter_3d,
            tracking_state=tracking_state,
        )
        selected_ids = torch.empty((0,), dtype=torch.int64, device=current_xyz.device)
    else:
        selected_ids = torch.from_numpy(selected_ids_np).to(device=current_xyz.device, dtype=torch.int64)
        patch_pattern = _grid_pattern(int(args.patch_grid_side), device=current_xyz.device, dtype=current_xyz.dtype)
        patch_count = int(patch_pattern.shape[0])

        mip_xyz = mip_gaussians._xyz.detach()
        mip_scale_t = mip_gaussians.get_scaling.detach()
        if isinstance(mip_gaussians.filter_3D, torch.Tensor) and mip_gaussians.filter_3D.ndim > 0:
            mip_filter_t = mip_gaussians.filter_3D.detach()
        else:
            mip_filter_t = torch.zeros((mip_count, 1), dtype=torch.float32, device=mip_xyz.device)
        mip_effective_scale = torch.sqrt(torch.square(mip_scale_t) + torch.square(mip_filter_t.reshape(-1, 1)))
        selected_scale = mip_effective_scale[selected_ids]
        axis_order = torch.argsort(selected_scale, dim=1, descending=True)
        rotations = build_rotation(mip_gaussians._rotation.detach()[selected_ids])
        axis_basis = torch.gather(rotations, 2, axis_order[:, None, :].expand(-1, 3, -1))
        sorted_selected_scale = torch.gather(selected_scale, 1, axis_order)

        local_offsets = patch_pattern[None, :, :] * sorted_selected_scale[:, None, :] * float(args.patch_offset_scale)
        child_xyz = mip_xyz[selected_ids, None, :] + torch.einsum("bij,bnj->bni", axis_basis, local_offsets)

        selected_raw_scale = mip_scale_t[selected_ids]
        sorted_raw_scale = torch.gather(selected_raw_scale, 1, axis_order)
        sorted_child_scale = sorted_raw_scale.clone()
        sorted_child_scale[:, 0] = sorted_child_scale[:, 0] * float(args.patch_tangent_scale_multiplier)
        sorted_child_scale[:, 1] = sorted_child_scale[:, 1] * float(args.patch_tangent_scale_multiplier)
        sorted_child_scale[:, 2] = sorted_child_scale[:, 2] * float(args.patch_normal_scale_multiplier)
        updated_scale = torch.zeros_like(selected_raw_scale)
        updated_scale.scatter_(1, axis_order, sorted_child_scale)
        child_scale = updated_scale[:, None, :].expand(-1, patch_count, -1)
        child_scaling = torch.log(torch.clamp(child_scale, min=1e-8)).reshape(-1, 3)

        child_filter = mip_filter_t[selected_ids, None, :].expand(-1, patch_count, -1)
        child_filter = child_filter * float(args.patch_filter_scale)
        if float(args.filter_cap_ratio) > 0.0:
            child_filter = torch.clamp(child_filter, max=scene_extent * float(args.filter_cap_ratio))
        child_filter = torch.clamp(child_filter, min=0.0).reshape(-1, mip_filter_t.shape[1])

        parent_opacity = mip_gaussians.get_opacity.detach()[selected_ids].reshape(-1)
        selected_deficit = torch.from_numpy(deficit_score[selected_ids_np]).to(device=current_xyz.device, dtype=parent_opacity.dtype)
        target_parent_opacity = torch.clamp(
            parent_opacity * (0.35 + 0.65 * selected_deficit) * float(args.patch_opacity_scale),
            min=1e-4,
            max=float(args.patch_max_opacity),
        )
        if str(args.energy_conserve_mode) == "none":
            child_opacity_prob = 1.0 - torch.pow(
                torch.clamp(1.0 - target_parent_opacity, min=1e-6),
                1.0 / float(patch_count),
            )
            child_opacity_prob = child_opacity_prob[:, None].expand(-1, patch_count)
        else:
            parent_metric = _opacity_compensation_metric(selected_scale, str(args.energy_conserve_mode)).reshape(-1)
            child_effective_scale = torch.sqrt(torch.square(child_scale) + torch.square(child_filter.reshape(selected_count, patch_count, -1)))
            child_metric = _opacity_compensation_metric(
                child_effective_scale.reshape(-1, 3),
                str(args.energy_conserve_mode),
            ).reshape(selected_count, patch_count)
            child_metric_sum = torch.clamp(child_metric.sum(dim=1), min=1e-12)
            target_mass = target_parent_opacity * parent_metric
            child_opacity_prob = (target_mass / child_metric_sum)[:, None].expand(-1, patch_count)
        child_opacity_prob = torch.clamp(child_opacity_prob, min=1e-6, max=0.95)
        child_opacity = _logit(child_opacity_prob.reshape(-1, 1))

        mip_features_dc = mip_gaussians._features_dc.detach()[selected_ids]
        child_features_dc = mip_features_dc[:, None, ...].expand(-1, patch_count, *mip_features_dc.shape[1:]).reshape(
            -1,
            *mip_features_dc.shape[1:],
        ).clone()
        child_features_rest = torch.zeros(
            (selected_count * patch_count, *current_features_rest.shape[1:]),
            dtype=current_features_rest.dtype,
            device=current_features_rest.device,
        )
        if float(args.features_rest_scale) > 0.0:
            mip_features_rest = mip_gaussians._features_rest.detach()[selected_ids]
            child_features_rest = mip_features_rest[:, None, ...].expand(
                -1,
                patch_count,
                *mip_features_rest.shape[1:],
            ).reshape(-1, *mip_features_rest.shape[1:]).clone() * float(args.features_rest_scale)
        child_rotation = mip_gaussians._rotation.detach()[selected_ids, None, :].expand(-1, patch_count, -1).reshape(
            -1,
            current_rotation_raw.shape[1],
        )

        xyz_out = torch.cat((current_xyz, child_xyz.reshape(-1, 3)), dim=0)
        features_dc_out = torch.cat((current_features_dc, child_features_dc), dim=0)
        features_rest_out = torch.cat((current_features_rest, child_features_rest), dim=0)
        opacity_out = torch.cat((current_opacity_raw, child_opacity), dim=0)
        scaling_out = torch.cat((current_scaling_raw, child_scaling), dim=0)
        rotation_out = torch.cat((current_rotation_raw, child_rotation), dim=0)
        filter_out = torch.cat((current_filter_3d, child_filter), dim=0)
        injected_output_mask = torch.cat(
            (
                torch.zeros((int(current_xyz.shape[0]),), dtype=torch.bool, device=current_xyz.device),
                torch.ones((selected_count * patch_count,), dtype=torch.bool, device=current_xyz.device),
            ),
            dim=0,
        )
        injected_source_mip_idx = selected_ids.repeat_interleave(patch_count)
        tracking_state = {
            "source_tag": torch.cat(
                (
                    current_gaussians._source_tag.detach(),
                    torch.full(
                        (selected_count * patch_count,),
                        int(GaussianSourceTag.PRIOR_INJECTED),
                        dtype=torch.int32,
                        device=current_xyz.device,
                    ),
                ),
                dim=0,
            ),
            "seed_id": torch.cat(
                (
                    current_gaussians._seed_id.detach(),
                    selected_ids.repeat_interleave(patch_count).to(dtype=torch.int64),
                ),
                dim=0,
            ),
            "generation": torch.cat(
                (
                    current_gaussians._generation.detach(),
                    torch.ones((selected_count * patch_count,), dtype=torch.int32, device=current_xyz.device),
                ),
                dim=0,
            ),
            "edge_touched": torch.cat(
                (
                    current_gaussians._edge_touched.detach(),
                    torch.zeros((selected_count * patch_count,), dtype=torch.bool, device=current_xyz.device),
                ),
                dim=0,
            ),
            "edge_touch_iter": torch.cat(
                (
                    current_gaussians._edge_touch_iter.detach(),
                    torch.full((selected_count * patch_count,), -1, dtype=torch.int32, device=current_xyz.device),
                ),
                dim=0,
            ),
        }
        merged = _make_static_gaussian_model(
            base=current_gaussians,
            xyz=xyz_out,
            features_dc=features_dc_out,
            features_rest=features_rest_out,
            opacity=opacity_out,
            scaling=scaling_out,
            rotation=rotation_out,
            filter_3d=filter_out,
            tracking_state=tracking_state,
        )

    _copy_render_config(model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{current_loaded_iter}"
    point_dir.mkdir(parents=True, exist_ok=True)
    merged.save_ply(str(point_dir / "point_cloud.ply"))
    merged.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    masks_dir = output_model_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(candidate.astype(bool)), masks_dir / "coverage_candidate_mip_mask.pt")
    torch.save(selected_ids.detach().cpu().to(torch.int64), masks_dir / "coverage_selected_mip_idx.pt")
    torch.save(injected_output_mask.detach().cpu(), masks_dir / "coverage_injected_output_mask.pt")
    torch.save(injected_source_mip_idx.detach().cpu(), masks_dir / "coverage_injected_source_mip_idx.pt")

    deficit_payload = {
        "version": "mip_surface_coverage_deficit_v0",
        "num_mip_gaussians": int(mip_count),
        "deficit_score": torch.from_numpy(deficit_score)[:, None],
        "alpha_score": torch.from_numpy(alpha_score)[:, None],
        "smooth_score": torch.from_numpy(smooth_score)[:, None],
        "visible_count": torch.from_numpy(visible_count.astype(np.int64))[:, None],
        "planarity": torch.from_numpy(planarity.astype(np.float32))[:, None],
        "tangent_extent": torch.from_numpy(tangent_extent.astype(np.float32))[:, None],
        "mip_opacity": torch.from_numpy(mip_opacity.astype(np.float32))[:, None],
        "candidate_mask": torch.from_numpy(candidate.astype(bool))[:, None],
        "meta": {
            "scene_root": str(scene_root),
            "model_path": str(model_path),
            "mip_model_path": str(mip_model_path),
            "output_model_path": str(output_model_path),
            "current_iteration": int(current_loaded_iter),
            "mip_iteration": int(mip_loaded_iter),
            "split": str(args.split),
            "selected_view_indices": selected_indices,
            "deficit_threshold": float(threshold),
            "candidate_count_before_cap": int(before_cap),
            "candidate_count_after_cap": int(np.sum(candidate)),
            "args": vars(args),
        },
    }
    torch.save(deficit_payload, output_model_path / "mip_surface_coverage_deficit_v0.pt")

    summary = {
        "version": "mip_surface_patch_carriers_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "mip_model_path": str(mip_model_path),
        "output_model_path": str(output_model_path),
        "current_iteration": int(current_loaded_iter),
        "mip_iteration": int(mip_loaded_iter),
        "split": str(args.split),
        "selected_view_indices": selected_indices,
        "input_gaussians": int(current_xyz.shape[0]),
        "mip_gaussians": int(mip_count),
        "candidate_count_before_cap": int(before_cap),
        "candidate_count_after_cap": int(np.sum(candidate)),
        "selected_count": int(selected_count),
        "injected_gaussians": int(injected_output_mask.sum().item()),
        "output_gaussians": int(merged.get_xyz.shape[0]),
        "deficit_threshold": float(threshold),
        "score_stats_visible": {
            "deficit_score": _stats(deficit_score, visible_mask),
            "alpha_score": _stats(alpha_score, visible_mask),
            "smooth_score": _stats(smooth_score, visible_mask),
            "planarity": _stats(planarity, visible_mask),
            "tangent_extent": _stats(tangent_extent, visible_mask),
        },
        "score_stats_candidate": {
            "deficit_score": _stats(deficit_score, candidate),
            "alpha_score": _stats(alpha_score, candidate),
            "smooth_score": _stats(smooth_score, candidate),
            "planarity": _stats(planarity, candidate),
            "tangent_extent": _stats(tangent_extent, candidate),
        },
        "view_summaries": view_summaries,
        "args": vars(args),
    }
    (output_model_path / "mip_surface_patch_carriers_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
