from __future__ import annotations

import argparse
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
from scene.gaussian_model import GaussianModel
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


def _build_image_lookup(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    lookup: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        if stem in lookup and lookup[stem] != path:
            raise ValueError(f"Duplicate image stem '{stem}' found under {images_dir}")
        lookup[stem] = path
    if not lookup:
        raise RuntimeError(f"No images found under {images_dir}")
    return lookup


def _load_rgb_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device)


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


def _select_uniform(items: Sequence[object], max_items: int) -> List[object]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    ids = np.unique(np.linspace(0, len(items) - 1, num=max_items, dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()]


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


def _edge_energy(luma_chw: torch.Tensor) -> torch.Tensor:
    device = luma_chw.device
    dtype = luma_chw.dtype
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    luma = luma_chw.unsqueeze(0)
    gx = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), sobel_x)
    gy = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), sobel_y)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-10)[0]
    highpass = torch.abs(luma_chw - _box_blur(luma_chw, 9))
    return grad + 1.5 * highpass


def _normalize_map(value: torch.Tensor, percentile: float, eps: float = 1e-6) -> torch.Tensor:
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(value)
    scale = torch.quantile(flat.float(), float(percentile)).clamp_min(float(eps))
    return torch.clamp(value / scale.to(device=value.device, dtype=value.dtype), 0.0, 1.0)


def _downsample_map(value_hw: torch.Tensor, coarse_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(coarse_hw[0]), int(coarse_hw[1])
    value = value_hw[None, None].float()
    coarse = F.interpolate(value, size=(target_h, target_w), mode="area")
    return coarse[0, 0].detach().cpu().numpy().astype(np.float32)


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
    # Small inferno-like palette without requiring matplotlib.
    r = np.clip(1.7 * norm, 0.0, 1.0)
    g = np.clip(1.7 * norm - 0.45, 0.0, 1.0)
    b = np.clip(1.4 * norm - 0.9, 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _save_overlay(path: Path, rgb_chw: torch.Tensor, heat_hw: torch.Tensor) -> None:
    rgb = rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    heat = heat_hw.detach().float().cpu().numpy()
    heat = np.clip(heat / max(float(np.percentile(heat, 99.0)), 1e-6), 0.0, 1.0)
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    red[..., 1] = 0.05
    overlay = rgb * (1.0 - 0.55 * heat[..., None]) + red * (0.55 * heat[..., None])
    Image.fromarray(np.clip(overlay * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


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


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def _clone_subset_gaussians(base: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    subset = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[mask].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[mask].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[mask].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[mask].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[mask].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[mask].clone().requires_grad_(False))
    if (
        isinstance(base.filter_3D, torch.Tensor)
        and base.filter_3D.ndim > 0
        and base.filter_3D.shape[0] == base.get_xyz.shape[0]
    ):
        subset.filter_3D = base.filter_3D.detach()[mask].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.init_tracking_state(count)
    if base._source_tag.shape[0] == base.get_xyz.shape[0]:
        subset._source_tag = base._source_tag.detach()[mask].clone()
    if base._seed_id.shape[0] == base.get_xyz.shape[0]:
        subset._seed_id = base._seed_id.detach()[mask].clone()
    if base._generation.shape[0] == base.get_xyz.shape[0]:
        subset._generation = base._generation.detach()[mask].clone()
    if base._edge_touched.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touched = base._edge_touched.detach()[mask].clone()
    if base._edge_touch_iter.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touch_iter = base._edge_touch_iter.detach()[mask].clone()
    return subset


def _write_masked_model(
    base: GaussianModel,
    mask_np: np.ndarray,
    output_root: Path,
    name: str,
    model_path: Path,
    iteration: int,
    meta: Dict[str, object] | None = None,
) -> str | None:
    if int(np.sum(mask_np)) <= 0:
        return None
    group_root = output_root / name
    point_dir = group_root / "point_cloud" / f"iteration_{int(iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    _copy_render_config(model_path, group_root)
    mask = torch.from_numpy(mask_np.astype(bool)).to(device=base.get_xyz.device)
    subset = _clone_subset_gaussians(base, mask)
    ply_path = point_dir / "point_cloud.ply"
    tags_path = point_dir / "gaussian_tags.pt"
    subset.save_ply(str(ply_path))
    subset.save_tracking_metadata(str(tags_path))
    (point_dir / "num_gaussians.json").write_text(
        json.dumps(
            {
                "num_gaussians": int(mask.sum().item()),
                "source_count": int(mask_np.shape[0]),
                "mask_name": name,
                "meta": meta or {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(group_root)


def _export_subset_ply(
    base: GaussianModel,
    mask_np: np.ndarray,
    output_root: Path,
    name: str,
    model_path: Path,
    iteration: int,
) -> str | None:
    model_root = _write_masked_model(
        base,
        mask_np,
        output_root,
        f"{name}_ply",
        model_path,
        iteration,
    )
    if model_root is None:
        return None
    ply_path = Path(model_root) / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    return str(ply_path)


def _build_delete_candidate_mask(
    artifact_score: np.ndarray,
    edge_support_score: np.ndarray,
    footprint_risk: np.ndarray,
    lr_forbidden_score: np.ndarray,
    opacity: np.ndarray,
    radius_max: np.ndarray,
    visible_mask: np.ndarray,
    delete_quantile: float,
    delete_min_lr_forbidden: float,
    delete_min_artifact: float,
    delete_max_edge_support: float,
    delete_min_footprint_risk: float,
    delete_max_opacity: float,
    delete_min_radius_px: float,
    delete_max_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | int]]:
    valid = (
        visible_mask
        & np.isfinite(artifact_score)
        & np.isfinite(edge_support_score)
        & np.isfinite(footprint_risk)
        & np.isfinite(lr_forbidden_score)
        & np.isfinite(opacity)
        & np.isfinite(radius_max)
    )
    delete_score = np.zeros_like(artifact_score, dtype=np.float32)
    if not np.any(valid):
        return np.zeros_like(visible_mask, dtype=bool), delete_score, {
            "delete_score_threshold": 1.0,
            "delete_candidate_count_before_cap": 0,
            "delete_candidate_count_after_cap": 0,
        }

    opacity_den = max(float(delete_max_opacity), 1e-6)
    transparent_risk = np.clip((float(delete_max_opacity) - opacity) / opacity_den, 0.0, 1.0)
    delete_score = np.clip(lr_forbidden_score * footprint_risk * transparent_risk, 0.0, 1.0).astype(np.float32)
    delete_score_threshold = float(np.quantile(delete_score[valid], float(delete_quantile)))
    delete_candidate = (
        valid
        & (delete_score >= delete_score_threshold)
        & (lr_forbidden_score >= float(delete_min_lr_forbidden))
        & (artifact_score >= float(delete_min_artifact))
        & (edge_support_score <= float(delete_max_edge_support))
        & (footprint_risk >= float(delete_min_footprint_risk))
        & (opacity <= float(delete_max_opacity))
        & (radius_max >= float(delete_min_radius_px))
    )
    count_before_cap = int(np.sum(delete_candidate))
    if float(delete_max_ratio) > 0.0:
        max_count = int(np.floor(float(delete_max_ratio) * float(delete_candidate.shape[0])))
        if max_count > 0 and count_before_cap > max_count:
            candidate_ids = np.flatnonzero(delete_candidate)
            order = np.argsort(delete_score[candidate_ids])[::-1]
            keep_ids = candidate_ids[order[:max_count]]
            capped = np.zeros_like(delete_candidate, dtype=bool)
            capped[keep_ids] = True
            delete_candidate = capped
        elif max_count <= 0:
            delete_candidate = np.zeros_like(delete_candidate, dtype=bool)
    return delete_candidate, delete_score, {
        "delete_score_threshold": delete_score_threshold,
        "delete_candidate_count_before_cap": count_before_cap,
        "delete_candidate_count_after_cap": int(np.sum(delete_candidate)),
    }


def _build_candidate_masks(
    artifact_score: np.ndarray,
    edge_support_score: np.ndarray,
    footprint_risk: np.ndarray,
    visible_mask: np.ndarray,
    suppress_quantile: float,
    pull_artifact_quantile: float,
    pull_edge_quantile: float,
    pull_max_footprint_risk: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    valid_scores = visible_mask & np.isfinite(artifact_score)
    if not np.any(valid_scores):
        empty = np.zeros_like(visible_mask, dtype=bool)
        return empty, empty, {
            "suppress_threshold": 1.0,
            "pull_artifact_threshold": 0.0,
            "pull_edge_threshold": 1.0,
        }
    lr_forbidden = artifact_score * (1.0 - edge_support_score) * (0.5 + 0.5 * footprint_risk)
    suppress_threshold = float(np.quantile(lr_forbidden[valid_scores], float(suppress_quantile)))
    pull_artifact_threshold = float(np.quantile(artifact_score[valid_scores], float(pull_artifact_quantile)))
    pull_edge_threshold = float(np.quantile(edge_support_score[valid_scores], float(pull_edge_quantile)))
    suppress_candidate = valid_scores & (lr_forbidden >= suppress_threshold) & (edge_support_score < 0.75)
    pull_allowed = (
        valid_scores
        & (artifact_score <= pull_artifact_threshold)
        & (edge_support_score >= pull_edge_threshold)
        & (footprint_risk <= float(pull_max_footprint_risk))
    )
    return suppress_candidate, pull_allowed, {
        "suppress_threshold": suppress_threshold,
        "pull_artifact_threshold": pull_artifact_threshold,
        "pull_edge_threshold": pull_edge_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export light per-GS image-scale legality / artifact scores.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--images_subdir", default=None)
    parser.add_argument("--interaction_images_subdir", default=None)
    parser.add_argument("--reference_images_subdir", default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--max_views", type=int, default=48)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--visibility_downsample", type=int, default=8)
    parser.add_argument("--visibility_topk", type=int, default=4)
    parser.add_argument("--visibility_max_visible", type=int, default=50000)
    parser.add_argument("--visibility_max_patch_radius", type=int, default=1)
    parser.add_argument("--lowpass_kernel", type=int, default=17)
    parser.add_argument("--residual_norm_percentile", type=float, default=0.95)
    parser.add_argument("--lr_edge_percentile", type=float, default=0.82)
    parser.add_argument("--render_hf_percentile", type=float, default=0.92)
    parser.add_argument("--radius_risk_px", type=float, default=18.0)
    parser.add_argument("--suppress_quantile", type=float, default=0.98)
    parser.add_argument("--pull_artifact_quantile", type=float, default=0.60)
    parser.add_argument("--pull_edge_quantile", type=float, default=0.35)
    parser.add_argument("--pull_max_footprint_risk", type=float, default=0.75)
    parser.add_argument("--num_debug_views", type=int, default=4)
    parser.add_argument("--delete_quantile", type=float, default=0.995)
    parser.add_argument("--delete_min_lr_forbidden", type=float, default=0.65)
    parser.add_argument("--delete_min_artifact", type=float, default=0.55)
    parser.add_argument("--delete_max_edge_support", type=float, default=0.45)
    parser.add_argument("--delete_min_footprint_risk", type=float, default=0.70)
    parser.add_argument("--delete_max_opacity", type=float, default=0.08)
    parser.add_argument("--delete_min_radius_px", type=float, default=12.0)
    parser.add_argument("--delete_max_ratio", type=float, default=0.03)
    parser.add_argument("--export_plys", action="store_true")
    parser.add_argument("--export_pruned_model", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_views"
    debug_dir.mkdir(parents=True, exist_ok=True)

    legacy_images_subdir = str(args.images_subdir) if args.images_subdir else None
    interaction_images_subdir = str(args.interaction_images_subdir or legacy_images_subdir or "images_2")
    reference_images_subdir = str(args.reference_images_subdir or legacy_images_subdir or interaction_images_subdir)
    reference_image_lookup = _build_image_lookup(scene_root / reference_images_subdir)
    share_interaction_reference = interaction_images_subdir == reference_images_subdir

    iteration = _resolve_model_iteration(model_path, int(args.iteration))
    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(model_path),
        images_subdir=interaction_images_subdir,
        white_background=bool(args.white_background),
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    all_views = _iter_views(scene, str(args.split))
    views = _select_uniform(all_views, int(args.max_views))
    if not views:
        raise RuntimeError(f"No views found for split={args.split}")

    gaussians.compute_3D_filter(scene.getTrainCameras().copy(), CUDA=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    num_gaussians = int(gaussians.get_xyz.shape[0])

    artifact_sum = np.zeros((num_gaussians,), dtype=np.float64)
    edge_sum = np.zeros((num_gaussians,), dtype=np.float64)
    metric_weight = np.zeros((num_gaussians,), dtype=np.float64)
    visible_count = np.zeros((num_gaussians,), dtype=np.int64)
    radius_sum = np.zeros((num_gaussians,), dtype=np.float64)
    radius_max = np.zeros((num_gaussians,), dtype=np.float64)

    view_summaries: List[Dict[str, object]] = []
    for view_idx, view in enumerate(tqdm(views, desc="lr-artifact-score")):
        render_pkg = render_simple(view, gaussians, background)
        render_rgb = render_pkg["render"][:3].detach().clamp(0.0, 1.0)
        if share_interaction_reference:
            reference_rgb = view.original_image[:3].detach().to(device=render_rgb.device).clamp(0.0, 1.0)
        else:
            image_key = str(view.image_name)
            reference_path = reference_image_lookup.get(image_key)
            if reference_path is None:
                raise KeyError(
                    f"Reference image '{image_key}' was not found under {scene_root / reference_images_subdir}"
                )
            reference_rgb = _load_rgb_tensor(reference_path, render_rgb.device).detach()
        if reference_rgb.shape[-2:] != render_rgb.shape[-2:]:
            reference_rgb = F.interpolate(
                reference_rgb[None],
                size=render_rgb.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )[0].clamp(0.0, 1.0)
        else:
            reference_rgb = reference_rgb.clamp(0.0, 1.0)

        render_low = _box_blur(render_rgb, int(args.lowpass_kernel))
        reference_low = _box_blur(reference_rgb, int(args.lowpass_kernel))
        low_err = torch.sqrt(torch.mean((render_low - reference_low).square(), dim=0) + 1e-10)
        low_err_norm = _normalize_map(low_err, float(args.residual_norm_percentile))

        lr_edge = _edge_energy(_rgb_to_luma(reference_rgb))[0]
        render_edge = _edge_energy(_rgb_to_luma(render_rgb))[0]
        lr_edge_norm = _normalize_map(lr_edge, float(args.lr_edge_percentile))
        render_edge_norm = _normalize_map(render_edge, float(args.render_hf_percentile))
        unsupported_hf = torch.clamp(render_edge_norm * (1.0 - lr_edge_norm), 0.0, 1.0)
        artifact_map = torch.clamp(0.75 * low_err_norm + 0.25 * unsupported_hf, 0.0, 1.0)

        image_hw = (int(view.image_height), int(view.image_width))
        records = build_coarse_visibility_records(
            gaussians,
            [view],
            [render_pkg],
            image_hw=image_hw,
            cfg=VisibilityRecordConfig(
                downsample=int(args.visibility_downsample),
                topk=int(args.visibility_topk),
                max_visible_per_view=int(args.visibility_max_visible),
                max_patch_radius=int(args.visibility_max_patch_radius),
            ),
        )
        coarse_h, coarse_w = [int(v) for v in records["coarse_hw"].tolist()]
        artifact_coarse = _downsample_map(artifact_map, (coarse_h, coarse_w))
        edge_coarse = _downsample_map(lr_edge_norm, (coarse_h, coarse_w))
        ids = records["gaussian_ids"][0, 0].numpy()
        weights = records["weights"][0, 0].numpy()
        _accumulate_from_visibility(ids, weights, artifact_coarse, artifact_sum, metric_weight)
        # Edge support uses the same normalized visibility denominator as the artifact map.
        valid_ids = ids >= 0
        if np.any(valid_ids):
            flat_ids = ids[valid_ids].astype(np.int64, copy=False)
            flat_weights = weights[valid_ids].astype(np.float64, copy=False)
            edge_expanded = np.repeat(edge_coarse[..., None], weights.shape[-1], axis=-1)
            np.add.at(edge_sum, flat_ids, edge_expanded[valid_ids].astype(np.float64, copy=False) * flat_weights)

        radii = render_pkg["radii"].detach().float().cpu().numpy()
        visible = render_pkg["visibility_filter"].detach().cpu().numpy().astype(bool)
        visible_ids = np.flatnonzero(visible)
        if visible_ids.size:
            visible_count[visible_ids] += 1
            radius_sum[visible_ids] += np.maximum(radii[visible_ids], 0.0)
            radius_max[visible_ids] = np.maximum(radius_max[visible_ids], np.maximum(radii[visible_ids], 0.0))

        view_summary = {
            "view_index": int(view_idx),
            "image_name": str(view.image_name),
            "interaction_images_subdir": interaction_images_subdir,
            "reference_images_subdir": reference_images_subdir,
            "artifact_map_mean": float(artifact_map.mean().item()),
            "artifact_map_p95": float(torch.quantile(artifact_map.reshape(-1), 0.95).item()),
            "lr_edge_mean": float(lr_edge_norm.mean().item()),
            "unsupported_hf_mean": float(unsupported_hf.mean().item()),
            "visible_gaussians": int(visible_ids.size),
        }
        view_summaries.append(view_summary)

        if view_idx < int(args.num_debug_views):
            prefix = debug_dir / f"view_{view_idx:03d}_{str(view.image_name).replace('/', '_')}"
            _save_rgb(prefix.with_name(prefix.name + "_render.png"), render_rgb)
            _save_rgb(prefix.with_name(prefix.name + "_reference.png"), reference_rgb)
            _save_heatmap(prefix.with_name(prefix.name + "_artifact_heat.png"), artifact_map)
            _save_overlay(prefix.with_name(prefix.name + "_artifact_overlay.png"), render_rgb, artifact_map)
            _save_heatmap(prefix.with_name(prefix.name + "_lr_edge_heat.png"), lr_edge_norm)
            _save_heatmap(prefix.with_name(prefix.name + "_unsupported_hf_heat.png"), unsupported_hf)

    denom = np.maximum(metric_weight, 1e-8)
    artifact_mean = artifact_sum / denom
    edge_mean = edge_sum / denom
    visible_mask = visible_count > 0
    radius_mean = radius_sum / np.maximum(visible_count, 1)
    footprint_risk = np.clip(0.5 * radius_mean / float(args.radius_risk_px) + 0.5 * radius_max / float(args.radius_risk_px), 0.0, 1.0)
    artifact_score = np.clip(artifact_mean, 0.0, 1.0)
    edge_support_score = np.clip(edge_mean, 0.0, 1.0)
    lr_forbidden_score = artifact_score * (1.0 - edge_support_score) * (0.5 + 0.5 * footprint_risk)
    opacity = gaussians.get_opacity.detach().float().cpu().numpy().reshape(-1)
    suppress_candidate, pull_allowed, thresholds = _build_candidate_masks(
        artifact_score,
        edge_support_score,
        footprint_risk,
        visible_mask,
        suppress_quantile=float(args.suppress_quantile),
        pull_artifact_quantile=float(args.pull_artifact_quantile),
        pull_edge_quantile=float(args.pull_edge_quantile),
        pull_max_footprint_risk=float(args.pull_max_footprint_risk),
    )
    delete_candidate, delete_score, delete_thresholds = _build_delete_candidate_mask(
        artifact_score=artifact_score,
        edge_support_score=edge_support_score,
        footprint_risk=footprint_risk,
        lr_forbidden_score=lr_forbidden_score,
        opacity=opacity,
        radius_max=radius_max,
        visible_mask=visible_mask,
        delete_quantile=float(args.delete_quantile),
        delete_min_lr_forbidden=float(args.delete_min_lr_forbidden),
        delete_min_artifact=float(args.delete_min_artifact),
        delete_max_edge_support=float(args.delete_max_edge_support),
        delete_min_footprint_risk=float(args.delete_min_footprint_risk),
        delete_max_opacity=float(args.delete_max_opacity),
        delete_min_radius_px=float(args.delete_min_radius_px),
        delete_max_ratio=float(args.delete_max_ratio),
    )
    keep_after_delete = ~delete_candidate

    payload = {
        "version": "lr_artifact_gaussian_scores_v0",
        "num_gaussians": int(num_gaussians),
        "artifact_score": torch.from_numpy(artifact_score.astype(np.float32))[:, None],
        "edge_support_score": torch.from_numpy(edge_support_score.astype(np.float32))[:, None],
        "footprint_risk": torch.from_numpy(footprint_risk.astype(np.float32))[:, None],
        "lr_forbidden_score": torch.from_numpy(lr_forbidden_score.astype(np.float32))[:, None],
        "delete_score": torch.from_numpy(delete_score.astype(np.float32))[:, None],
        "opacity": torch.from_numpy(opacity.astype(np.float32))[:, None],
        "visible_count": torch.from_numpy(visible_count.astype(np.int64))[:, None],
        "radius_mean": torch.from_numpy(radius_mean.astype(np.float32))[:, None],
        "radius_max": torch.from_numpy(radius_max.astype(np.float32))[:, None],
        "suppress_candidate": torch.from_numpy(suppress_candidate.astype(bool))[:, None],
        "pull_allowed": torch.from_numpy(pull_allowed.astype(bool))[:, None],
        "delete_candidate": torch.from_numpy(delete_candidate.astype(bool))[:, None],
        "keep_after_delete": torch.from_numpy(keep_after_delete.astype(bool))[:, None],
        "meta": {
            "scene_root": str(scene_root),
            "model_path": str(model_path),
            "images_subdir": interaction_images_subdir,
            "interaction_images_subdir": interaction_images_subdir,
            "reference_images_subdir": reference_images_subdir,
            "legacy_images_subdir_arg": legacy_images_subdir,
            "reference_note": "Artifact legality is measured in reference_images_subdir while rendering/visibility use interaction_images_subdir.",
            "iteration": int(loaded_iter),
            "split": str(args.split),
            "selected_views": [str(view.image_name) for view in views],
            "thresholds": thresholds,
            "delete_thresholds": delete_thresholds,
        },
    }
    torch.save(payload, output_dir / "lr_artifact_scores_v0.pt")

    exported_plys: Dict[str, str | None] = {}
    if bool(args.export_plys):
        exported_plys["suppress_candidate"] = _export_subset_ply(
            gaussians,
            suppress_candidate,
            output_dir,
            "suppress_candidate",
            model_path,
            loaded_iter,
        )
        exported_plys["pull_allowed"] = _export_subset_ply(
            gaussians,
            pull_allowed,
            output_dir,
            "pull_allowed",
            model_path,
            loaded_iter,
        )
        exported_plys["delete_candidate"] = _export_subset_ply(
            gaussians,
            delete_candidate,
            output_dir,
            "delete_candidate",
            model_path,
            loaded_iter,
        )
    pruned_model: str | None = None
    if bool(args.export_pruned_model):
        pruned_model = _write_masked_model(
            gaussians,
            keep_after_delete,
            output_dir,
            "pruned_delete_candidate_model",
            model_path,
            loaded_iter,
            meta={
                "delete_candidate_count": int(np.sum(delete_candidate)),
                "delete_thresholds": delete_thresholds,
            },
        )

    summary = {
        "version": "lr_artifact_gaussian_scores_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "images_subdir": interaction_images_subdir,
        "interaction_images_subdir": interaction_images_subdir,
        "reference_images_subdir": reference_images_subdir,
        "legacy_images_subdir_arg": legacy_images_subdir,
        "reference_note": "Artifact legality is measured in reference_images_subdir while rendering/visibility use interaction_images_subdir.",
        "iteration": int(loaded_iter),
        "split": str(args.split),
        "num_views": int(len(views)),
        "num_gaussians": int(num_gaussians),
        "visible_gaussians": int(np.sum(visible_mask)),
        "suppress_candidate_count": int(np.sum(suppress_candidate)),
        "pull_allowed_count": int(np.sum(pull_allowed)),
        "delete_candidate_count": int(np.sum(delete_candidate)),
        "suppress_candidate_ratio": float(np.mean(suppress_candidate)),
        "pull_allowed_ratio": float(np.mean(pull_allowed)),
        "delete_candidate_ratio": float(np.mean(delete_candidate)),
        "score_stats_visible": {
            "artifact_score": _stats(artifact_score, visible_mask),
            "edge_support_score": _stats(edge_support_score, visible_mask),
            "footprint_risk": _stats(footprint_risk, visible_mask),
            "lr_forbidden_score": _stats(lr_forbidden_score, visible_mask),
            "delete_score": _stats(delete_score, visible_mask),
            "opacity": _stats(opacity, visible_mask),
            "radius_max": _stats(radius_max, visible_mask),
            "visible_count": _stats(visible_count.astype(np.float32), visible_mask),
        },
        "thresholds": thresholds,
        "delete_thresholds": delete_thresholds,
        "args": vars(args),
        "debug_views_dir": str(debug_dir),
        "score_payload": str(output_dir / "lr_artifact_scores_v0.pt"),
        "exported_plys": exported_plys,
        "pruned_model": pruned_model,
        "view_summaries": view_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
