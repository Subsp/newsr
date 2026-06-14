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
from scene.gaussian_model import GaussianModel
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


SH_C0 = 0.28209479177387814


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


def _iter_views(scene: Scene, split: str) -> List[object]:
    if split == "train":
        return list(scene.getTrainCameras())
    if split == "test":
        return list(scene.getTestCameras())
    if split == "both":
        return list(scene.getTrainCameras()) + list(scene.getTestCameras())
    raise ValueError(f"Unsupported split: {split}")


def _parse_csv_ints(value: str | None) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    out: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _parse_csv_floats(value: str) -> List[float]:
    out: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("Expected at least one comma-separated value")
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


def _list_image_paths(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    paths: List[Path] = []
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            paths.append(path)
    if not paths:
        raise RuntimeError(f"No images found under {images_dir}")
    return paths


def _build_image_lookup_from_paths(paths: Sequence[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if stem in lookup and lookup[stem] != path:
            raise ValueError(f"Duplicate image stem '{stem}' found in reference images")
        lookup[stem] = path
    return lookup


def _build_image_lookup(images_dir: Path) -> Dict[str, Path]:
    return _build_image_lookup_from_paths(_list_image_paths(images_dir))


def _resolve_reference_path(
    *,
    lookup: Dict[str, Path],
    paths: Sequence[Path],
    image_name: str,
    source_view_idx: int,
    local_view_idx: int,
    reference_root: Path | None,
) -> Path:
    tried_keys = [str(image_name), f"{int(source_view_idx):05d}", f"{int(local_view_idx):05d}"]
    for key in tried_keys:
        matched = lookup.get(key)
        if matched is not None:
            return matched
    # Mip-Splatting renders are usually saved as 00000.png, 00001.png, ... in view order.
    if 0 <= int(source_view_idx) < len(paths):
        return paths[int(source_view_idx)]
    if 0 <= int(local_view_idx) < len(paths):
        return paths[int(local_view_idx)]
    raise KeyError(
        f"Reference image '{image_name}' was not found under {reference_root}; "
        f"tried stems {tried_keys} and numeric render-order fallback."
    )


def _load_rgb_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device)


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


def _line_kernel(length: int, angle_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    radius = length // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    theta = torch.tensor(float(angle_deg) * np.pi / 180.0, device=device, dtype=dtype)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    along = xx * cos_t + yy * sin_t
    perp = -xx * sin_t + yy * cos_t
    mask = (torch.abs(along) <= float(radius) + 0.25) & (torch.abs(perp) <= 0.55)
    kernel = mask.to(dtype=dtype)
    kernel = kernel / torch.clamp(kernel.sum(), min=1.0)
    return kernel.view(1, 1, length, length)


def _bright_line_ridge(
    luma_chw: torch.Tensor,
    *,
    line_lengths: Sequence[int],
    angles_deg: Sequence[float],
    highpass_kernel: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    luma = luma_chw.float()
    highpass = torch.relu(luma - _box_blur(luma, int(highpass_kernel)))
    angle_responses: List[torch.Tensor] = []
    for angle in angles_deg:
        per_length: List[torch.Tensor] = []
        for length in line_lengths:
            length = int(length)
            if length % 2 == 0:
                length += 1
            pad = length // 2
            kernel = _line_kernel(length, float(angle), device=luma.device, dtype=luma.dtype)
            response = F.conv2d(F.pad(highpass[None], (pad, pad, pad, pad), mode="reflect"), kernel)[0, 0]
            per_length.append(response)
        angle_responses.append(torch.stack(per_length, dim=0).max(dim=0).values)
    responses = torch.stack(angle_responses, dim=0)
    max_response = responses.max(dim=0).values
    if responses.shape[0] >= 2:
        top2 = torch.topk(responses, k=2, dim=0, largest=True, sorted=True).values
        directional_diversity = torch.clamp(top2[1] / torch.clamp(top2[0], min=1e-6), 0.0, 1.0)
    else:
        directional_diversity = torch.zeros_like(max_response)
    return max_response, directional_diversity


def _build_starburst_maps(
    render_rgb: torch.Tensor,
    reference_rgb: torch.Tensor | None,
    *,
    line_lengths: Sequence[int],
    angles_deg: Sequence[float],
    highpass_kernel: int,
    ridge_norm_percentile: float,
    residual_norm_percentile: float,
) -> Dict[str, torch.Tensor]:
    render_luma = _rgb_to_luma(render_rgb)[0]
    render_ridge, diversity = _bright_line_ridge(
        render_luma[None],
        line_lengths=line_lengths,
        angles_deg=angles_deg,
        highpass_kernel=int(highpass_kernel),
    )
    render_ridge_norm = _normalize_map(render_ridge, float(ridge_norm_percentile))
    diversity = torch.clamp(diversity, 0.0, 1.0)

    if reference_rgb is None:
        unsupported = render_ridge_norm
        positive_residual = torch.zeros_like(render_ridge_norm)
        reference_ridge_norm = torch.zeros_like(render_ridge_norm)
        supported_structure = torch.zeros_like(render_ridge_norm)
        star_map = torch.clamp(render_ridge_norm * (0.70 + 0.30 * diversity), 0.0, 1.0)
    else:
        reference_luma = _rgb_to_luma(reference_rgb)[0]
        reference_ridge, _ = _bright_line_ridge(
            reference_luma[None],
            line_lengths=line_lengths,
            angles_deg=angles_deg,
            highpass_kernel=int(highpass_kernel),
        )
        reference_ridge_norm = _normalize_map(reference_ridge, float(ridge_norm_percentile))
        positive_residual = torch.relu(render_luma - reference_luma)
        positive_residual_norm = _normalize_map(positive_residual, float(residual_norm_percentile))
        unsupported = torch.clamp(render_ridge_norm * (1.0 - 0.85 * reference_ridge_norm), 0.0, 1.0)
        supported_structure = torch.clamp(
            torch.minimum(render_ridge_norm, reference_ridge_norm)
            * (0.55 + 0.45 * (1.0 - positive_residual_norm)),
            0.0,
            1.0,
        )
        residual_supported = torch.clamp(0.45 + 0.55 * positive_residual_norm, 0.0, 1.0)
        star_map = torch.clamp(
            (0.70 * unsupported + 0.30 * render_ridge_norm * residual_supported)
            * (0.70 + 0.30 * diversity),
            0.0,
            1.0,
        )
    return {
        "star_map": star_map,
        "render_ridge": render_ridge_norm,
        "reference_ridge": reference_ridge_norm,
        "unsupported_ridge": unsupported,
        "supported_structure": supported_structure,
        "positive_residual": _normalize_map(positive_residual, float(residual_norm_percentile)),
        "directional_diversity": diversity,
    }


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
    r = np.clip(1.8 * norm, 0.0, 1.0)
    g = np.clip(1.6 * norm - 0.35, 0.0, 1.0)
    b = np.clip(1.3 * norm - 0.85, 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _save_overlay(path: Path, rgb_chw: torch.Tensor, heat_hw: torch.Tensor) -> None:
    rgb = rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    heat = heat_hw.detach().float().cpu().numpy()
    heat = np.clip(heat / max(float(np.percentile(heat, 99.0)), 1e-6), 0.0, 1.0)
    mark = np.zeros_like(rgb)
    mark[..., 0] = 1.0
    mark[..., 1] = 0.18
    overlay = rgb * (1.0 - 0.60 * heat[..., None]) + mark * (0.60 * heat[..., None])
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


def _accumulate_star_view_count(
    gaussian_ids: np.ndarray,
    star_metric: np.ndarray,
    counts: np.ndarray,
    *,
    view_star_quantile: float,
    view_star_min: float,
) -> int:
    finite = np.isfinite(star_metric)
    if not np.any(finite):
        return 0
    threshold = float(np.quantile(star_metric[finite], float(view_star_quantile)))
    threshold = max(threshold, float(view_star_min))
    active = finite & (star_metric >= threshold)
    if not np.any(active):
        return 0
    active_ids = gaussian_ids[np.repeat(active[..., None], gaussian_ids.shape[-1], axis=-1) & (gaussian_ids >= 0)]
    if active_ids.size == 0:
        return 0
    unique_ids = np.unique(active_ids.astype(np.int64, copy=False))
    counts[unique_ids] += 1
    return int(unique_ids.size)


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
    for name in ("_source_tag", "_seed_id", "_generation", "_edge_touched", "_edge_touch_iter"):
        value = getattr(base, name, None)
        if torch.is_tensor(value) and int(value.shape[0]) == int(base.get_xyz.shape[0]):
            setattr(subset, name, value.detach()[mask].clone())
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
    subset.save_ply(str(point_dir / "point_cloud.ply"))
    subset.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
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


def _robust_unit(values: np.ndarray, valid: np.ndarray, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.quantile(values[mask], float(quantile))), 1e-6)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def _quantile_threshold(values: np.ndarray, valid: np.ndarray, quantile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    if not np.any(mask):
        return float(default)
    return float(np.quantile(arr[mask], float(quantile)))


def _cap_candidate_mask(
    mask: np.ndarray,
    score: np.ndarray,
    *,
    max_fraction: float,
    max_count: int,
) -> tuple[np.ndarray, int, int]:
    candidate = np.asarray(mask, dtype=bool).copy()
    before_cap = int(np.sum(candidate))
    cap = before_cap
    if float(max_fraction) > 0.0:
        cap = min(cap, max(1, int(round(float(max_fraction) * float(candidate.shape[0])))))
    if int(max_count) > 0:
        cap = min(cap, int(max_count))
    if cap <= 0:
        return np.zeros_like(candidate, dtype=bool), before_cap, 0
    if before_cap > cap:
        candidate_ids = np.flatnonzero(candidate)
        order = np.argsort(-np.asarray(score, dtype=np.float32)[candidate_ids], kind="stable")[:cap]
        capped = np.zeros_like(candidate, dtype=bool)
        capped[candidate_ids[order]] = True
        candidate = capped
    return candidate, before_cap, int(np.sum(candidate))


def _features_dc_luma(gaussians: GaussianModel) -> np.ndarray:
    features_dc = gaussians._features_dc.detach().float()
    if features_dc.ndim == 3:
        dc = features_dc[:, 0, :]
    else:
        dc = features_dc.reshape(features_dc.shape[0], -1)[:, :3]
    rgb = torch.clamp(dc * SH_C0 + 0.5, min=0.0)
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    return luma.detach().cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect starburst-like Gaussian artifacts from image-space bright ridges.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--interaction_images_subdir", default="images_2")
    parser.add_argument("--reference_images_subdir", default="")
    parser.add_argument("--reference_image_dir", default="")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--view_indices", default="")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--visibility_downsample", type=int, default=4)
    parser.add_argument("--visibility_topk", type=int, default=8)
    parser.add_argument("--visibility_max_visible", type=int, default=80000)
    parser.add_argument("--visibility_max_patch_radius", type=int, default=2)
    parser.add_argument("--line_lengths", default="9,17,31")
    parser.add_argument("--angles_deg", default="0,22.5,45,67.5,90,112.5,135,157.5")
    parser.add_argument("--highpass_kernel", type=int, default=21)
    parser.add_argument("--ridge_norm_percentile", type=float, default=0.995)
    parser.add_argument("--residual_norm_percentile", type=float, default=0.990)
    parser.add_argument("--view_star_quantile", type=float, default=0.985)
    parser.add_argument("--view_star_min", type=float, default=0.08)
    parser.add_argument("--select_quantile", type=float, default=0.990)
    parser.add_argument("--min_star_score", type=float, default=0.08)
    parser.add_argument("--min_unsupported_score", type=float, default=0.04)
    parser.add_argument("--min_star_view_count", type=int, default=1)
    parser.add_argument("--global_long_axis_quantile", type=float, default=0.94)
    parser.add_argument("--global_anisotropy_quantile", type=float, default=0.90)
    parser.add_argument("--global_radius_max_quantile", type=float, default=0.94)
    parser.add_argument("--global_bad_support_quantile", type=float, default=0.92)
    parser.add_argument("--global_support_ratio_quantile", type=float, default=0.92)
    parser.add_argument("--global_impact_body_ratio_quantile", type=float, default=0.94)
    parser.add_argument("--global_prefilter_min_hits", type=int, default=2)
    parser.add_argument("--global_prefilter_max_fraction", type=float, default=0.10)
    parser.add_argument("--global_prefilter_max_count", type=int, default=120000)
    parser.add_argument("--max_candidate_fraction", type=float, default=0.015)
    parser.add_argument("--max_candidate_count", type=int, default=30000)
    parser.add_argument("--num_debug_views", type=int, default=4)
    parser.add_argument("--export_candidate_model", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_views"
    debug_dir.mkdir(parents=True, exist_ok=True)

    line_lengths = [int(v) for v in _parse_csv_floats(str(args.line_lengths))]
    angles_deg = _parse_csv_floats(str(args.angles_deg))
    view_indices = _parse_csv_ints(str(args.view_indices))
    reference_images_subdir = str(args.reference_images_subdir).strip()
    reference_image_dir = str(args.reference_image_dir).strip()
    reference_lookup: Dict[str, Path] | None = None
    reference_root: Path | None = None
    reference_paths: List[Path] = []
    if reference_image_dir:
        reference_root = Path(reference_image_dir).expanduser().resolve()
        reference_paths = _list_image_paths(reference_root)
        reference_lookup = _build_image_lookup_from_paths(reference_paths)
    elif reference_images_subdir:
        reference_root = scene_root / reference_images_subdir
        reference_paths = _list_image_paths(reference_root)
        reference_lookup = _build_image_lookup_from_paths(reference_paths)

    iteration = _resolve_model_iteration(model_path, int(args.iteration))
    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(model_path),
        images_subdir=str(args.interaction_images_subdir),
        white_background=bool(args.white_background),
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    all_views = _iter_views(scene, str(args.split))
    views, selected_indices = _select_views(all_views, int(args.max_views), view_indices)
    if not views:
        raise RuntimeError(f"No views selected for split={args.split}")

    gaussians.compute_3D_filter(scene.getTrainCameras().copy(), CUDA=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    num_gaussians = int(gaussians.get_xyz.shape[0])

    star_sum = np.zeros((num_gaussians,), dtype=np.float64)
    unsupported_sum = np.zeros((num_gaussians,), dtype=np.float64)
    supported_sum = np.zeros((num_gaussians,), dtype=np.float64)
    residual_sum = np.zeros((num_gaussians,), dtype=np.float64)
    diversity_sum = np.zeros((num_gaussians,), dtype=np.float64)
    metric_weight = np.zeros((num_gaussians,), dtype=np.float64)
    visible_count = np.zeros((num_gaussians,), dtype=np.int64)
    star_view_count = np.zeros((num_gaussians,), dtype=np.int64)
    radius_sum = np.zeros((num_gaussians,), dtype=np.float64)
    radius_max = np.zeros((num_gaussians,), dtype=np.float64)
    radius_min = np.full((num_gaussians,), np.inf, dtype=np.float64)

    view_summaries: List[Dict[str, object]] = []
    for local_idx, (source_view_idx, view) in enumerate(tqdm(list(zip(selected_indices, views)), desc="starburst-score")):
        render_pkg = render_simple(view, gaussians, background)
        render_rgb = render_pkg["render"][:3].detach().clamp(0.0, 1.0)
        reference_rgb = None
        reference_path = None
        if reference_lookup is not None:
            reference_path = _resolve_reference_path(
                lookup=reference_lookup,
                paths=reference_paths,
                image_name=str(view.image_name),
                source_view_idx=int(source_view_idx),
                local_view_idx=int(local_idx),
                reference_root=reference_root,
            )
            reference_rgb = _load_rgb_tensor(reference_path, render_rgb.device).detach().clamp(0.0, 1.0)
            if reference_rgb.shape[-2:] != render_rgb.shape[-2:]:
                reference_rgb = F.interpolate(
                    reference_rgb[None],
                    size=render_rgb.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )[0].clamp(0.0, 1.0)

        maps = _build_starburst_maps(
            render_rgb,
            reference_rgb,
            line_lengths=line_lengths,
            angles_deg=angles_deg,
            highpass_kernel=int(args.highpass_kernel),
            ridge_norm_percentile=float(args.ridge_norm_percentile),
            residual_norm_percentile=float(args.residual_norm_percentile),
        )

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
        ids = records["gaussian_ids"][0, 0].numpy()
        weights = records["weights"][0, 0].numpy()
        star_coarse = _downsample_map(maps["star_map"], (coarse_h, coarse_w))
        unsupported_coarse = _downsample_map(maps["unsupported_ridge"], (coarse_h, coarse_w))
        supported_coarse = _downsample_map(maps["supported_structure"], (coarse_h, coarse_w))
        residual_coarse = _downsample_map(maps["positive_residual"], (coarse_h, coarse_w))
        diversity_coarse = _downsample_map(maps["directional_diversity"], (coarse_h, coarse_w))
        _accumulate_from_visibility(ids, weights, star_coarse, star_sum, metric_weight)
        _accumulate_from_visibility(ids, weights, unsupported_coarse, unsupported_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, supported_coarse, supported_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, residual_coarse, residual_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, diversity_coarse, diversity_sum, np.zeros_like(metric_weight))
        active_gs = _accumulate_star_view_count(
            ids,
            star_coarse,
            star_view_count,
            view_star_quantile=float(args.view_star_quantile),
            view_star_min=float(args.view_star_min),
        )

        radii = render_pkg["radii"].detach().float().cpu().numpy()
        visible = render_pkg["visibility_filter"].detach().cpu().numpy().astype(bool)
        visible_ids = np.flatnonzero(visible)
        if visible_ids.size:
            valid_radius = np.maximum(radii[visible_ids], 0.0)
            visible_count[visible_ids] += 1
            radius_sum[visible_ids] += valid_radius
            radius_max[visible_ids] = np.maximum(radius_max[visible_ids], valid_radius)
            radius_min[visible_ids] = np.minimum(radius_min[visible_ids], valid_radius)

        view_summary = {
            "local_view_index": int(local_idx),
            "source_view_index": int(source_view_idx),
            "image_name": str(view.image_name),
            "reference_image": str(reference_path) if reference_path is not None else "",
            "star_map_mean": float(maps["star_map"].mean().item()),
            "star_map_p95": float(torch.quantile(maps["star_map"].reshape(-1), 0.95).item()),
            "unsupported_mean": float(maps["unsupported_ridge"].mean().item()),
            "supported_mean": float(maps["supported_structure"].mean().item()),
            "directional_diversity_mean": float(maps["directional_diversity"].mean().item()),
            "active_star_gaussians_in_view": int(active_gs),
            "visible_gaussians": int(visible_ids.size),
        }
        view_summaries.append(view_summary)

        if local_idx < int(args.num_debug_views):
            prefix = debug_dir / f"view_{local_idx:03d}_src{source_view_idx:05d}_{str(view.image_name).replace('/', '_')}"
            _save_rgb(prefix.with_name(prefix.name + "_render.png"), render_rgb)
            if reference_rgb is not None:
                _save_rgb(prefix.with_name(prefix.name + "_reference.png"), reference_rgb)
            for key in ("star_map", "render_ridge", "reference_ridge", "unsupported_ridge", "supported_structure", "positive_residual", "directional_diversity"):
                _save_heatmap(prefix.with_name(prefix.name + f"_{key}_heat.png"), maps[key])
            _save_overlay(prefix.with_name(prefix.name + "_star_overlay.png"), render_rgb, maps["star_map"])

    denom = np.maximum(metric_weight, 1e-8)
    star_score = np.clip(star_sum / denom, 0.0, 1.0).astype(np.float32)
    unsupported_score = np.clip(unsupported_sum / denom, 0.0, 1.0).astype(np.float32)
    supported_score = np.clip(supported_sum / denom, 0.0, 1.0).astype(np.float32)
    residual_score = np.clip(residual_sum / denom, 0.0, 1.0).astype(np.float32)
    diversity_score = np.clip(diversity_sum / denom, 0.0, 1.0).astype(np.float32)
    visible_mask = visible_count > 0
    radius_mean = radius_sum / np.maximum(visible_count, 1)
    radius_min = np.where(np.isfinite(radius_min), radius_min, 0.0)

    scale = gaussians.get_scaling.detach().float().cpu().numpy()
    if isinstance(gaussians.filter_3D, torch.Tensor) and gaussians.filter_3D.ndim > 0:
        filter_3d = gaussians.filter_3D.detach().float().cpu().numpy().reshape(-1, 1)
    else:
        filter_3d = np.zeros((num_gaussians, 1), dtype=np.float32)
    effective_scale = np.sqrt(np.square(scale) + np.square(filter_3d)).astype(np.float32)
    sorted_scale = np.sort(np.clip(effective_scale, 1e-8, None), axis=1)
    effective_scale_max = sorted_scale[:, 2]
    volume_radius = np.clip(np.prod(effective_scale, axis=1), 1e-24, None) ** (1.0 / 3.0)
    anisotropy = sorted_scale[:, 2] / np.clip(sorted_scale[:, 0], 1e-8, None)
    raw_scale_max = np.clip(scale.max(axis=1), 1e-8, None)
    filter_scale_ratio = filter_3d.reshape(-1) / raw_scale_max
    opacity = gaussians.get_opacity.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    dc_luma = _features_dc_luma(gaussians)

    effective_unit = _robust_unit(effective_scale_max, visible_mask, 0.98)
    volume_unit = _robust_unit(volume_radius, visible_mask, 0.98)
    anisotropy_unit = _robust_unit(np.log1p(anisotropy), visible_mask, 0.98)
    filter_unit = _robust_unit(filter_scale_ratio, visible_mask, 0.98)
    luma_unit = _robust_unit(dc_luma, visible_mask, 0.98)
    radius_unit = _robust_unit(radius_max, visible_mask, 0.98)
    geometry_risk = np.clip(
        0.35 * effective_unit + 0.30 * volume_unit + 0.20 * anisotropy_unit + 0.15 * filter_unit,
        0.0,
        1.0,
    ).astype(np.float32)
    body_area_proxy = np.clip(sorted_scale[:, 0] * sorted_scale[:, 1], 1e-12, None).astype(np.float32)
    body_area_unit = _robust_unit(np.sqrt(body_area_proxy), visible_mask, 0.98)
    global_long_axis_score = np.clip(
        0.45 * effective_unit + 0.35 * anisotropy_unit + 0.20 * radius_unit,
        0.0,
        1.0,
    ).astype(np.float32)
    support_ratio = (star_score + 1e-4) / np.clip(supported_score + 1e-4, 1e-4, None)
    support_gap = np.clip(star_score - 0.70 * supported_score, 0.0, None).astype(np.float32)
    impact_body_ratio = support_gap / np.clip(0.12 + 0.88 * body_area_unit, 0.12, None)
    support_ratio_unit = _robust_unit(np.log1p(support_ratio), visible_mask, 0.98)
    impact_body_unit = _robust_unit(np.log1p(impact_body_ratio), visible_mask, 0.98)
    global_prefilter_score = np.clip(
        0.34 * impact_body_unit
        + 0.26 * support_ratio_unit
        + 0.16 * unsupported_score
        + 0.14 * star_score
        + 0.10 * global_long_axis_score,
        0.0,
        1.0,
    ).astype(np.float32)

    starburst_score = np.clip(
        star_score
        * (0.65 + 0.35 * unsupported_score)
        * (0.75 + 0.25 * diversity_score)
        * (0.70 + 0.30 * luma_unit)
        * (0.75 + 0.25 * geometry_risk),
        0.0,
        1.0,
    ).astype(np.float32)

    valid_score = (
        visible_mask
        & np.isfinite(starburst_score)
        & (star_score >= float(args.min_star_score))
        & (star_view_count >= int(args.min_star_view_count))
    )
    if reference_lookup is not None:
        valid_score &= unsupported_score >= float(args.min_unsupported_score)

    long_axis_threshold = _quantile_threshold(
        effective_scale_max,
        visible_mask,
        float(args.global_long_axis_quantile),
    )
    anisotropy_threshold = _quantile_threshold(
        anisotropy,
        visible_mask,
        float(args.global_anisotropy_quantile),
    )
    radius_threshold = _quantile_threshold(
        radius_max,
        visible_mask,
        float(args.global_radius_max_quantile),
    )
    bad_support_threshold = _quantile_threshold(
        star_score,
        visible_mask,
        float(args.global_bad_support_quantile),
    )
    support_ratio_threshold = _quantile_threshold(
        np.log1p(support_ratio),
        visible_mask,
        float(args.global_support_ratio_quantile),
    )
    impact_body_ratio_threshold = _quantile_threshold(
        np.log1p(impact_body_ratio),
        visible_mask,
        float(args.global_impact_body_ratio_quantile),
    )
    global_long_axis_hits = np.zeros((num_gaussians,), dtype=np.int32)
    global_long_axis_hits += (effective_scale_max >= long_axis_threshold).astype(np.int32)
    global_long_axis_hits += (anisotropy >= anisotropy_threshold).astype(np.int32)
    global_long_axis_hits += (radius_max >= radius_threshold).astype(np.int32)
    global_long_axis_candidate = visible_mask & (global_long_axis_hits >= int(args.global_prefilter_min_hits))

    global_hits = global_long_axis_hits.copy()
    global_hits += (star_score >= bad_support_threshold).astype(np.int32)
    global_hits += (np.log1p(support_ratio) >= support_ratio_threshold).astype(np.int32)
    global_hits += (np.log1p(impact_body_ratio) >= impact_body_ratio_threshold).astype(np.int32)
    global_candidate = visible_mask & (global_hits >= int(args.global_prefilter_min_hits))
    global_candidate, global_before_cap, global_after_cap = _cap_candidate_mask(
        global_candidate,
        global_prefilter_score,
        max_fraction=float(args.global_prefilter_max_fraction),
        max_count=int(args.global_prefilter_max_count),
    )

    view_refined_score = np.clip(
        starburst_score
        * (0.45 + 0.35 * global_prefilter_score + 0.20 * impact_body_unit)
        * (0.60 + 0.40 * np.clip(1.0 - supported_score, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)
    if np.any(global_candidate):
        view_refined_candidate = valid_score & global_candidate
    else:
        view_refined_candidate = valid_score.copy()
    if np.any(view_refined_candidate):
        threshold = float(np.quantile(view_refined_score[view_refined_candidate], float(args.select_quantile)))
        candidate = view_refined_candidate & (view_refined_score >= threshold)
    else:
        threshold = 1.0
        candidate = np.zeros((num_gaussians,), dtype=bool)

    candidate, before_cap, after_cap = _cap_candidate_mask(
        candidate,
        view_refined_score,
        max_fraction=float(args.max_candidate_fraction),
        max_count=int(args.max_candidate_count),
    )

    payload = {
        "version": "starburst_gaussian_scores_v0",
        "num_gaussians": int(num_gaussians),
        "starburst_score": torch.from_numpy(starburst_score.astype(np.float32))[:, None],
        "view_refined_score": torch.from_numpy(view_refined_score.astype(np.float32))[:, None],
        "star_score": torch.from_numpy(star_score.astype(np.float32))[:, None],
        "unsupported_score": torch.from_numpy(unsupported_score.astype(np.float32))[:, None],
        "supported_score": torch.from_numpy(supported_score.astype(np.float32))[:, None],
        "positive_residual_score": torch.from_numpy(residual_score.astype(np.float32))[:, None],
        "directional_diversity_score": torch.from_numpy(diversity_score.astype(np.float32))[:, None],
        "geometry_risk": torch.from_numpy(geometry_risk.astype(np.float32))[:, None],
        "global_long_axis_score": torch.from_numpy(global_long_axis_score.astype(np.float32))[:, None],
        "global_prefilter_score": torch.from_numpy(global_prefilter_score.astype(np.float32))[:, None],
        "support_ratio": torch.from_numpy(support_ratio.astype(np.float32))[:, None],
        "impact_body_ratio": torch.from_numpy(impact_body_ratio.astype(np.float32))[:, None],
        "opacity": torch.from_numpy(opacity.astype(np.float32))[:, None],
        "dc_luma": torch.from_numpy(dc_luma.astype(np.float32))[:, None],
        "effective_scale_max": torch.from_numpy(effective_scale_max.astype(np.float32))[:, None],
        "volume_radius": torch.from_numpy(volume_radius.astype(np.float32))[:, None],
        "anisotropy": torch.from_numpy(anisotropy.astype(np.float32))[:, None],
        "filter_scale_ratio": torch.from_numpy(filter_scale_ratio.astype(np.float32))[:, None],
        "body_area_proxy": torch.from_numpy(body_area_proxy.astype(np.float32))[:, None],
        "visible_count": torch.from_numpy(visible_count.astype(np.int64))[:, None],
        "star_view_count": torch.from_numpy(star_view_count.astype(np.int64))[:, None],
        "radius_mean": torch.from_numpy(radius_mean.astype(np.float32))[:, None],
        "radius_min": torch.from_numpy(radius_min.astype(np.float32))[:, None],
        "radius_max": torch.from_numpy(radius_max.astype(np.float32))[:, None],
        "global_long_axis_candidate": torch.from_numpy(global_long_axis_candidate.astype(bool))[:, None],
        "global_prefilter_candidate": torch.from_numpy(global_candidate.astype(bool))[:, None],
        "view_refined_candidate": torch.from_numpy(view_refined_candidate.astype(bool))[:, None],
        "starburst_candidate": torch.from_numpy(candidate.astype(bool))[:, None],
        "meta": {
            "scene_root": str(scene_root),
            "model_path": str(model_path),
            "interaction_images_subdir": str(args.interaction_images_subdir),
            "reference_images_subdir": reference_images_subdir,
            "reference_image_dir": str(reference_root) if reference_root is not None else "",
            "reference_note": "Use LR images, SR-prior images, or LR MIP renders only; do not point this detector at held-out GT.",
            "iteration": int(loaded_iter),
            "split": str(args.split),
            "selected_view_indices": selected_indices,
            "selected_view_names": [str(view.image_name) for view in views],
            "global_long_axis_thresholds": {
                "effective_scale_max": float(long_axis_threshold),
                "anisotropy": float(anisotropy_threshold),
                "radius_max": float(radius_threshold),
                "bad_support": float(bad_support_threshold),
                "support_ratio_log1p": float(support_ratio_threshold),
                "impact_body_ratio_log1p": float(impact_body_ratio_threshold),
            },
            "starburst_score_threshold": float(threshold),
            "global_candidate_count_before_cap": int(global_before_cap),
            "global_candidate_count_after_cap": int(global_after_cap),
            "view_refined_count": int(np.sum(view_refined_candidate)),
            "candidate_count_before_cap": int(before_cap),
            "candidate_count_after_cap": int(after_cap),
            "args": vars(args),
        },
    }
    torch.save(payload, output_dir / "starburst_gaussian_scores_v0.pt")
    torch.save(torch.from_numpy(candidate.astype(bool)), output_dir / "starburst_candidate_mask.pt")

    exported_model = None
    if bool(args.export_candidate_model):
        exported_model = _write_masked_model(
            gaussians,
            candidate,
            output_dir,
            "starburst_candidate_ply",
            model_path,
            loaded_iter,
            meta={
                "global_candidate_count_before_cap": int(global_before_cap),
                "global_candidate_count_after_cap": int(global_after_cap),
                "view_refined_count": int(np.sum(view_refined_candidate)),
                "starburst_score_threshold": float(threshold),
                "candidate_count_before_cap": int(before_cap),
                "candidate_count_after_cap": int(after_cap),
            },
        )

    summary = {
        "version": "starburst_gaussian_scores_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "interaction_images_subdir": str(args.interaction_images_subdir),
        "reference_images_subdir": reference_images_subdir,
        "reference_image_dir": str(reference_root) if reference_root is not None else "",
        "reference_note": "Use LR images, SR-prior images, or LR MIP renders only; do not point this detector at held-out GT.",
        "iteration": int(loaded_iter),
        "split": str(args.split),
        "num_views": int(len(views)),
        "selected_view_indices": selected_indices,
        "num_gaussians": int(num_gaussians),
        "visible_gaussians": int(np.sum(visible_mask)),
        "global_long_axis_candidate_count": int(np.sum(global_long_axis_candidate)),
        "global_candidate_count_before_cap": int(global_before_cap),
        "global_candidate_count_after_cap": int(global_after_cap),
        "view_refined_count": int(np.sum(view_refined_candidate)),
        "candidate_count_before_cap": int(before_cap),
        "candidate_count_after_cap": int(after_cap),
        "candidate_ratio": float(np.mean(candidate)),
        "global_long_axis_thresholds": {
            "effective_scale_max": float(long_axis_threshold),
            "anisotropy": float(anisotropy_threshold),
            "radius_max": float(radius_threshold),
            "bad_support": float(bad_support_threshold),
            "support_ratio_log1p": float(support_ratio_threshold),
            "impact_body_ratio_log1p": float(impact_body_ratio_threshold),
        },
        "starburst_score_threshold": float(threshold),
        "score_stats_visible": {
            "starburst_score": _stats(starburst_score, visible_mask),
            "view_refined_score": _stats(view_refined_score, visible_mask),
            "star_score": _stats(star_score, visible_mask),
            "unsupported_score": _stats(unsupported_score, visible_mask),
            "supported_score": _stats(supported_score, visible_mask),
            "positive_residual_score": _stats(residual_score, visible_mask),
            "directional_diversity_score": _stats(diversity_score, visible_mask),
            "geometry_risk": _stats(geometry_risk, visible_mask),
            "global_long_axis_score": _stats(global_long_axis_score, visible_mask),
            "global_prefilter_score": _stats(global_prefilter_score, visible_mask),
            "support_ratio": _stats(support_ratio, visible_mask),
            "impact_body_ratio": _stats(impact_body_ratio, visible_mask),
            "opacity": _stats(opacity, visible_mask),
            "dc_luma": _stats(dc_luma, visible_mask),
            "effective_scale_max": _stats(effective_scale_max, visible_mask),
            "volume_radius": _stats(volume_radius, visible_mask),
            "anisotropy": _stats(anisotropy, visible_mask),
            "filter_scale_ratio": _stats(filter_scale_ratio, visible_mask),
            "body_area_proxy": _stats(body_area_proxy, visible_mask),
            "radius_max": _stats(radius_max, visible_mask),
            "visible_count": _stats(visible_count.astype(np.float32), visible_mask),
            "star_view_count": _stats(star_view_count.astype(np.float32), visible_mask),
        },
        "score_stats_global_long_axis_candidate": {
            "global_long_axis_score": _stats(global_long_axis_score, global_long_axis_candidate),
            "global_prefilter_score": _stats(global_prefilter_score, global_long_axis_candidate),
            "support_ratio": _stats(support_ratio, global_long_axis_candidate),
            "impact_body_ratio": _stats(impact_body_ratio, global_long_axis_candidate),
        },
        "score_stats_global_candidate": {
            "global_long_axis_score": _stats(global_long_axis_score, global_candidate),
            "global_prefilter_score": _stats(global_prefilter_score, global_candidate),
            "effective_scale_max": _stats(effective_scale_max, global_candidate),
            "anisotropy": _stats(anisotropy, global_candidate),
            "radius_max": _stats(radius_max, global_candidate),
            "star_score": _stats(star_score, global_candidate),
            "unsupported_score": _stats(unsupported_score, global_candidate),
            "supported_score": _stats(supported_score, global_candidate),
            "support_ratio": _stats(support_ratio, global_candidate),
            "impact_body_ratio": _stats(impact_body_ratio, global_candidate),
        },
        "score_stats_view_refined": {
            "view_refined_score": _stats(view_refined_score, view_refined_candidate),
            "starburst_score": _stats(starburst_score, view_refined_candidate),
            "star_score": _stats(star_score, view_refined_candidate),
            "unsupported_score": _stats(unsupported_score, view_refined_candidate),
            "supported_score": _stats(supported_score, view_refined_candidate),
            "geometry_risk": _stats(geometry_risk, view_refined_candidate),
            "global_long_axis_score": _stats(global_long_axis_score, view_refined_candidate),
            "global_prefilter_score": _stats(global_prefilter_score, view_refined_candidate),
            "support_ratio": _stats(support_ratio, view_refined_candidate),
            "impact_body_ratio": _stats(impact_body_ratio, view_refined_candidate),
        },
        "score_stats_candidate": {
            "starburst_score": _stats(starburst_score, candidate),
            "view_refined_score": _stats(view_refined_score, candidate),
            "star_score": _stats(star_score, candidate),
            "unsupported_score": _stats(unsupported_score, candidate),
            "supported_score": _stats(supported_score, candidate),
            "geometry_risk": _stats(geometry_risk, candidate),
            "global_long_axis_score": _stats(global_long_axis_score, candidate),
            "global_prefilter_score": _stats(global_prefilter_score, candidate),
            "support_ratio": _stats(support_ratio, candidate),
            "impact_body_ratio": _stats(impact_body_ratio, candidate),
            "opacity": _stats(opacity, candidate),
            "dc_luma": _stats(dc_luma, candidate),
            "radius_max": _stats(radius_max, candidate),
        },
        "score_payload": str(output_dir / "starburst_gaussian_scores_v0.pt"),
        "candidate_mask": str(output_dir / "starburst_candidate_mask.pt"),
        "exported_candidate_model": exported_model,
        "debug_views_dir": str(debug_dir),
        "view_summaries": view_summaries,
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
