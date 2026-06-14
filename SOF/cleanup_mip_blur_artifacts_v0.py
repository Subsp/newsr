from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from joint_judge_mip_sof_v0 import apply_soft_suppression, collect_render_metrics, stats_from_array
from scripts.export_lr_artifact_gaussian_scores_v0 import (
    _accumulate_from_visibility,
    _box_blur,
    _downsample_map,
    _edge_energy,
    _rgb_to_luma,
    _save_heatmap,
    _save_overlay,
    _save_rgb,
    _stats,
    _write_masked_model,
)
from train_mip_to_sof_surface_v0 import (
    build_dataset_args,
    build_prepared_sr_cache,
    load_model_ply,
    load_train_cameras_only,
    resolve_iteration,
    select_uniform,
)
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def normalize_map(value: torch.Tensor, percentile: float, eps: float = 1e-6) -> torch.Tensor:
    flat = value.detach().reshape(-1).float()
    if flat.numel() == 0:
        return torch.zeros_like(value)
    scale = torch.quantile(flat, float(percentile)).clamp_min(float(eps))
    return torch.clamp(value / scale.to(device=value.device, dtype=value.dtype), 0.0, 1.0)


def tensor_stats(value: torch.Tensor) -> Dict[str, float]:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "max": float(flat.max().item()),
    }


def _cap_mask(
    candidate: np.ndarray,
    score: np.ndarray,
    *,
    max_fraction: float,
    max_count: int,
) -> Tuple[np.ndarray, int, int]:
    capped = candidate.astype(bool, copy=True)
    before_cap = int(np.count_nonzero(capped))
    cap = before_cap
    if float(max_fraction) > 0.0:
        cap = min(cap, max(1, int(round(capped.shape[0] * float(max_fraction)))))
    if int(max_count) > 0:
        cap = min(cap, int(max_count))
    if cap <= 0:
        capped = np.zeros_like(capped, dtype=bool)
    elif before_cap > cap:
        ids = np.flatnonzero(capped).astype(np.int64, copy=False)
        order = np.argsort(-score[ids], kind="stable")[:cap]
        tmp = np.zeros_like(capped, dtype=bool)
        tmp[ids[order]] = True
        capped = tmp
    return capped, before_cap, int(np.count_nonzero(capped))


def build_cleanup_masks(
    *,
    blur_score: np.ndarray,
    sr_support_score: np.ndarray,
    prior_detail_score: np.ndarray,
    render_detail_score: np.ndarray,
    detail_gap_score: np.ndarray,
    sr_residual_score: np.ndarray,
    lowfreq_score: np.ndarray,
    lr_bad_score: np.ndarray,
    footprint_risk: np.ndarray,
    opacity: np.ndarray,
    visible_count: np.ndarray,
    num_score_views: int,
    radius_max: np.ndarray,
    visible_mask: np.ndarray,
    delete_quantile: float,
    min_blur_score: float,
    min_sr_support: float,
    min_prior_detail: float,
    min_detail_gap: float,
    min_lowfreq_score: float,
    min_sr_residual: float,
    min_footprint_risk: float,
    min_radius_px: float,
    prune_max_opacity: float,
    prune_max_visible_fraction: float,
    max_prune_fraction: float,
    max_prune_count: int,
    soft_max_opacity: float,
    max_soft_fraction: float,
    max_soft_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    valid = (
        visible_mask
        & np.isfinite(blur_score)
        & np.isfinite(sr_support_score)
        & np.isfinite(prior_detail_score)
        & np.isfinite(render_detail_score)
        & np.isfinite(detail_gap_score)
        & np.isfinite(sr_residual_score)
        & np.isfinite(lowfreq_score)
        & np.isfinite(lr_bad_score)
        & np.isfinite(footprint_risk)
        & np.isfinite(opacity)
        & np.isfinite(radius_max)
    )
    delete_score = np.zeros_like(blur_score, dtype=np.float32)
    if not np.any(valid):
        empty = np.zeros_like(visible_mask, dtype=bool)
        return empty, empty, delete_score, {
            "delete_score_threshold": 1.0,
            "prune_candidate_count_before_cap": 0,
            "prune_candidate_count_after_cap": 0,
            "soft_candidate_count_before_cap": 0,
            "soft_candidate_count_after_cap": 0,
        }

    opacity = np.clip(opacity.astype(np.float32, copy=False), 0.0, 1.0)
    visible_fraction = np.clip(
        visible_count.astype(np.float32, copy=False) / max(int(num_score_views), 1),
        0.0,
        1.0,
    )
    delete_score = (
        np.clip(blur_score, 0.0, 1.0)
        * (0.35 + 0.65 * np.clip(footprint_risk, 0.0, 1.0))
        * (0.50 + 0.50 * np.clip(sr_support_score, 0.0, 1.0))
        * (0.85 + 0.15 * np.clip(lr_bad_score, 0.0, 1.0))
    ).astype(np.float32)
    threshold = float(np.quantile(delete_score[valid], float(delete_quantile)))
    common_candidate = (
        valid
        & (delete_score >= threshold)
        & (blur_score >= float(min_blur_score))
        & (sr_support_score >= float(min_sr_support))
        & (prior_detail_score >= float(min_prior_detail))
        & ((detail_gap_score >= float(min_detail_gap)) | (lowfreq_score >= float(min_lowfreq_score)))
        & (sr_residual_score >= float(min_sr_residual))
        & (footprint_risk >= float(min_footprint_risk))
        & (radius_max >= float(min_radius_px))
    )
    transparency_risk = np.clip(
        (float(prune_max_opacity) - opacity) / max(float(prune_max_opacity), 1e-6),
        0.0,
        1.0,
    )
    limited_support_risk = np.clip(
        (float(prune_max_visible_fraction) - visible_fraction) / max(float(prune_max_visible_fraction), 1e-6),
        0.0,
        1.0,
    )
    prune_score = (
        delete_score
        * (0.40 + 0.60 * transparency_risk)
        * (0.40 + 0.60 * limited_support_risk)
    ).astype(np.float32)
    prune_candidate = (
        common_candidate
        & (opacity <= float(prune_max_opacity))
        & (visible_fraction <= float(prune_max_visible_fraction))
    )
    prune_mask, prune_before_cap, prune_after_cap = _cap_mask(
        prune_candidate,
        prune_score,
        max_fraction=float(max_prune_fraction),
        max_count=int(max_prune_count),
    )
    soft_score = (
        delete_score
        * (0.50 + 0.50 * np.clip(lowfreq_score, 0.0, 1.0))
        * (0.40 + 0.60 * np.clip(footprint_risk, 0.0, 1.0))
    ).astype(np.float32)
    soft_candidate = common_candidate & ~prune_mask & (opacity <= float(soft_max_opacity))
    soft_mask, soft_before_cap, soft_after_cap = _cap_mask(
        soft_candidate,
        soft_score,
        max_fraction=float(max_soft_fraction),
        max_count=int(max_soft_count),
    )
    return prune_mask.astype(bool), soft_mask.astype(bool), delete_score, {
        "delete_score_threshold": float(threshold),
        "prune_candidate_count_before_cap": int(prune_before_cap),
        "prune_candidate_count_after_cap": int(prune_after_cap),
        "soft_candidate_count_before_cap": int(soft_before_cap),
        "soft_candidate_count_after_cap": int(soft_after_cap),
    }


def main() -> None:
    parser = ArgumentParser(description="Delete mip-stage large-footprint blur artifacts using LR/SR detail evidence.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--mip_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_iteration", type=int, default=30000)
    parser.add_argument("--lr_images_subdir", default="images_8")
    parser.add_argument("--sr_images_subdir", default="images_2")
    parser.add_argument("--sr_prior_root", required=True)
    parser.add_argument("--sr_prior_subdir", default="fused_priors")
    parser.add_argument("--sr_prior_mask_subdir", default="usable_masks")
    parser.add_argument("--sr_anchor_subdir", default="aligned_references")
    parser.add_argument("--sr_prior_consistency_threshold", type=float, default=0.12)
    parser.add_argument("--sr_prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--max_lr_views", type=int, default=16)
    parser.add_argument("--max_sr_views", type=int, default=16)
    parser.add_argument("--visibility_downsample", type=int, default=8)
    parser.add_argument("--visibility_topk", type=int, default=4)
    parser.add_argument("--visibility_max_visible", type=int, default=60000)
    parser.add_argument("--visibility_max_patch_radius", type=int, default=2)
    parser.add_argument("--lowpass_kernel", type=int, default=31)
    parser.add_argument("--veil_weight", type=float, default=0.65)
    parser.add_argument("--detail_gap_weight", type=float, default=0.35)
    parser.add_argument("--detail_norm_percentile", type=float, default=0.92)
    parser.add_argument("--residual_norm_percentile", type=float, default=0.95)
    parser.add_argument("--lowfreq_norm_percentile", type=float, default=0.93)
    parser.add_argument("--radius_risk_px", type=float, default=22.0)
    parser.add_argument("--delete_quantile", type=float, default=0.960)
    parser.add_argument("--min_blur_score", type=float, default=0.08)
    parser.add_argument("--min_sr_support", type=float, default=0.18)
    parser.add_argument("--min_prior_detail", type=float, default=0.05)
    parser.add_argument("--min_detail_gap", type=float, default=0.05)
    parser.add_argument("--min_lowfreq_score", type=float, default=0.08)
    parser.add_argument("--min_sr_residual", type=float, default=0.05)
    parser.add_argument("--min_footprint_risk", type=float, default=0.30)
    parser.add_argument("--min_radius_px", type=float, default=18.0)
    parser.add_argument("--prune_max_opacity", type=float, default=0.30)
    parser.add_argument("--prune_max_visible_fraction", type=float, default=0.45)
    parser.add_argument("--lr_bad_residual_threshold", type=float, default=0.12)
    parser.add_argument("--max_prune_fraction", type=float, default=0.02)
    parser.add_argument("--max_prune_count", type=int, default=0)
    parser.add_argument("--soft_max_opacity", type=float, default=0.70)
    parser.add_argument("--max_soft_fraction", type=float, default=0.04)
    parser.add_argument("--max_soft_count", type=int, default=0)
    parser.add_argument("--soft_opacity_scale", type=float, default=0.75)
    parser.add_argument("--soft_scale_shrink", type=float, default=0.90)
    parser.add_argument("--num_debug_views", type=int, default=4)
    parser.add_argument("--export_deleted_model", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    sr_prior_root = Path(args.sr_prior_root).expanduser().resolve()
    sr_prior_dir = sr_prior_root / str(args.sr_prior_subdir)
    sr_mask_dir = sr_prior_root / str(args.sr_prior_mask_subdir)
    sr_anchor_dir = sr_prior_root / str(args.sr_anchor_subdir)
    for label, path in (("sr_prior_dir", sr_prior_dir), ("sr_mask_dir", sr_mask_dir), ("sr_anchor_dir", sr_anchor_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")

    dataset_args = build_dataset_args(str(scene_root), str(mip_model_path), str(args.lr_images_subdir))
    iteration = resolve_iteration(mip_model_path, int(args.iteration))
    mip = load_model_ply(mip_model_path, iteration, int(dataset_args.sh_degree))
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")

    lr_all = load_train_cameras_only(scene_root, mip_model_path, str(args.lr_images_subdir))
    lr_cameras = select_uniform(lr_all, int(args.max_lr_views))
    sr_all = load_train_cameras_only(scene_root, mip_model_path, str(args.sr_images_subdir))
    if lr_cameras:
        selected_lr_names = {normalize_image_name(cam.image_name) for cam in lr_cameras}
        sr_candidates = [cam for cam in sr_all if normalize_image_name(cam.image_name) in selected_lr_names]
        if not sr_candidates:
            sr_candidates = sr_all
    else:
        sr_candidates = sr_all
    sr_cameras = select_uniform(sr_candidates, int(args.max_sr_views))
    if not sr_cameras:
        raise RuntimeError("No SR cameras found.")

    filter_cameras = lr_cameras + [cam for cam in sr_cameras if cam not in lr_cameras]
    mip.compute_3D_filter(filter_cameras, CUDA=False)
    sr_index = index_image_dir(str(sr_prior_dir))
    sr_anchor_index = index_image_dir(str(sr_anchor_dir))
    sr_cameras, sr_cache = build_prepared_sr_cache(
        sr_cameras,
        mip,
        background,
        sr_prior_index=sr_index,
        sr_anchor_index=sr_anchor_index,
        sr_prior_mask_dir=sr_mask_dir,
        sr_prior_mask_suffix="",
        prior_consistency_threshold=float(args.sr_prior_consistency_threshold),
        prior_mask_floor=float(args.sr_prior_mask_floor),
    )
    if not sr_cache:
        raise RuntimeError(f"No SR priors matched SR cameras under {sr_prior_root}")

    num_gaussians = int(mip.get_xyz.shape[0])
    blur_sum = np.zeros((num_gaussians,), dtype=np.float64)
    support_sum = np.zeros((num_gaussians,), dtype=np.float64)
    prior_detail_sum = np.zeros((num_gaussians,), dtype=np.float64)
    render_detail_sum = np.zeros((num_gaussians,), dtype=np.float64)
    detail_gap_sum = np.zeros((num_gaussians,), dtype=np.float64)
    sr_residual_sum = np.zeros((num_gaussians,), dtype=np.float64)
    lowfreq_sum = np.zeros((num_gaussians,), dtype=np.float64)
    metric_weight = np.zeros((num_gaussians,), dtype=np.float64)
    visible_count = np.zeros((num_gaussians,), dtype=np.int64)
    radius_sum = np.zeros((num_gaussians,), dtype=np.float64)
    radius_max = np.zeros((num_gaussians,), dtype=np.float64)

    output_model_path.mkdir(parents=True, exist_ok=True)
    debug_dir = output_model_path / "debug_sr_blur_views"
    debug_dir.mkdir(parents=True, exist_ok=True)
    view_summaries: List[Dict[str, object]] = []
    vis_cfg = VisibilityRecordConfig(
        downsample=int(args.visibility_downsample),
        topk=int(args.visibility_topk),
        max_visible_per_view=int(args.visibility_max_visible),
        max_patch_radius=int(args.visibility_max_patch_radius),
    )

    print(f"[mip-blur-cleanup-v0] scene      : {scene_root}")
    print(f"[mip-blur-cleanup-v0] mip model  : {mip_model_path} iter={iteration} n={num_gaussians}")
    print(f"[mip-blur-cleanup-v0] SR priors  : {sr_prior_root}")
    print(f"[mip-blur-cleanup-v0] views      : lr={len(lr_cameras)} sr={len(sr_cameras)}")
    for view_idx, (camera, target) in enumerate(tqdm(list(zip(sr_cameras, sr_cache)), desc="mip blur score")):
        render_pkg = render_simple(camera, mip, background)
        render_rgb = render_pkg["render"].detach().clamp(0.0, 1.0)
        prior_rgb = target["prior_rgb"]
        prior_mask = target["prior_mask"]
        if not torch.is_tensor(prior_rgb) or not torch.is_tensor(prior_mask):
            continue
        prior_rgb = prior_rgb.to(device=background.device, dtype=render_rgb.dtype).clamp(0.0, 1.0)
        prior_mask = prior_mask.to(device=background.device, dtype=render_rgb.dtype).clamp(0.0, 1.0)

        prior_detail = normalize_map(
            _edge_energy(_rgb_to_luma(prior_rgb))[0],
            float(args.detail_norm_percentile),
        )
        render_detail = normalize_map(
            _edge_energy(_rgb_to_luma(render_rgb))[0],
            float(args.detail_norm_percentile),
        )
        detail_gap = torch.clamp(prior_detail - render_detail, 0.0, 1.0)
        residual = torch.mean(torch.abs(render_rgb - prior_rgb), dim=0)
        residual_norm = normalize_map(residual, float(args.residual_norm_percentile))
        low_render = _box_blur(render_rgb, int(args.lowpass_kernel))
        low_prior = _box_blur(prior_rgb, int(args.lowpass_kernel))
        lowfreq_residual = torch.sqrt(torch.mean((low_render - low_prior).square(), dim=0) + 1e-10)
        lowfreq_norm = normalize_map(lowfreq_residual, float(args.lowfreq_norm_percentile))
        render_smooth = torch.clamp(1.0 - render_detail, 0.0, 1.0)
        veil_map = torch.clamp(prior_mask * lowfreq_norm * (0.35 + 0.65 * render_smooth), 0.0, 1.0)
        detail_gap_map = torch.clamp(prior_mask * prior_detail * detail_gap * (0.50 + 0.50 * residual_norm), 0.0, 1.0)
        norm_weight = max(float(args.veil_weight) + float(args.detail_gap_weight), 1e-6)
        blur_map = torch.clamp(
            (float(args.veil_weight) * veil_map + float(args.detail_gap_weight) * detail_gap_map) / norm_weight,
            0.0,
            1.0,
        )

        image_hw = (int(camera.image_height), int(camera.image_width))
        records = build_coarse_visibility_records(
            mip,
            [camera],
            [render_pkg],
            image_hw=image_hw,
            cfg=vis_cfg,
        )
        coarse_h, coarse_w = [int(v) for v in records["coarse_hw"].tolist()]
        ids = records["gaussian_ids"][0, 0].numpy()
        weights = records["weights"][0, 0].numpy()
        _accumulate_from_visibility(ids, weights, _downsample_map(blur_map, (coarse_h, coarse_w)), blur_sum, metric_weight)
        _accumulate_from_visibility(ids, weights, _downsample_map(prior_mask, (coarse_h, coarse_w)), support_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, _downsample_map(prior_detail, (coarse_h, coarse_w)), prior_detail_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, _downsample_map(render_detail, (coarse_h, coarse_w)), render_detail_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, _downsample_map(detail_gap, (coarse_h, coarse_w)), detail_gap_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, _downsample_map(residual_norm, (coarse_h, coarse_w)), sr_residual_sum, np.zeros_like(metric_weight))
        _accumulate_from_visibility(ids, weights, _downsample_map(lowfreq_norm, (coarse_h, coarse_w)), lowfreq_sum, np.zeros_like(metric_weight))

        radii = render_pkg["radii"].detach().float().cpu().numpy().reshape(-1)
        visible = render_pkg["visibility_filter"].detach().cpu().numpy().astype(bool).reshape(-1)
        visible_ids = np.flatnonzero(visible).astype(np.int64, copy=False)
        if visible_ids.size > 0:
            visible_count[visible_ids] += 1
            radius_sum[visible_ids] += np.maximum(radii[visible_ids], 0.0)
            radius_max[visible_ids] = np.maximum(radius_max[visible_ids], np.maximum(radii[visible_ids], 0.0))

        view_summaries.append(
            {
                "view_index": int(view_idx),
                "image_name": str(camera.image_name),
                "prior_path": str(target.get("prior_path")),
                "mask_mean": float(prior_mask.mean().item()),
                "prior_detail": tensor_stats(prior_detail),
                "render_detail": tensor_stats(render_detail),
                "detail_gap": tensor_stats(detail_gap),
                "lowfreq_residual": tensor_stats(lowfreq_norm),
                "veil_map": tensor_stats(veil_map),
                "blur_map": tensor_stats(blur_map),
                "visible_gaussians": int(visible_ids.size),
            }
        )
        if view_idx < int(args.num_debug_views):
            prefix = debug_dir / f"view_{view_idx:03d}_{str(camera.image_name).replace('/', '_')}"
            _save_rgb(prefix.with_name(prefix.name + "_render.png"), render_rgb)
            _save_rgb(prefix.with_name(prefix.name + "_prior.png"), prior_rgb)
            _save_heatmap(prefix.with_name(prefix.name + "_prior_mask_heat.png"), prior_mask)
            _save_heatmap(prefix.with_name(prefix.name + "_prior_detail_heat.png"), prior_detail)
            _save_heatmap(prefix.with_name(prefix.name + "_render_detail_heat.png"), render_detail)
            _save_heatmap(prefix.with_name(prefix.name + "_detail_gap_heat.png"), detail_gap)
            _save_heatmap(prefix.with_name(prefix.name + "_lowfreq_residual_heat.png"), lowfreq_norm)
            _save_heatmap(prefix.with_name(prefix.name + "_veil_score_heat.png"), veil_map)
            _save_heatmap(prefix.with_name(prefix.name + "_blur_score_heat.png"), blur_map)
            _save_overlay(prefix.with_name(prefix.name + "_blur_overlay.png"), render_rgb, blur_map)

    denom = np.maximum(metric_weight, 1e-8)
    blur_score = np.clip(blur_sum / denom, 0.0, 1.0).astype(np.float32)
    sr_support_score = np.clip(support_sum / denom, 0.0, 1.0).astype(np.float32)
    prior_detail_score = np.clip(prior_detail_sum / denom, 0.0, 1.0).astype(np.float32)
    render_detail_score = np.clip(render_detail_sum / denom, 0.0, 1.0).astype(np.float32)
    detail_gap_score = np.clip(detail_gap_sum / denom, 0.0, 1.0).astype(np.float32)
    sr_residual_score = np.clip(sr_residual_sum / denom, 0.0, 1.0).astype(np.float32)
    lowfreq_score = np.clip(lowfreq_sum / denom, 0.0, 1.0).astype(np.float32)
    radius_mean = (radius_sum / np.maximum(visible_count, 1)).astype(np.float32)
    footprint_risk = np.clip(
        0.5 * radius_mean / max(float(args.radius_risk_px), 1.0)
        + 0.5 * radius_max / max(float(args.radius_risk_px), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    opacity = mip.get_opacity.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    visible_fraction = np.clip(visible_count.astype(np.float32) / max(len(sr_cameras), 1), 0.0, 1.0)
    visible_mask = visible_count > 0

    lr_metrics = {}
    lr_bad_score = np.zeros((num_gaussians,), dtype=np.float32)
    if lr_cameras:
        lr_render = collect_render_metrics(
            mip,
            lr_cameras,
            large_radius_px=float(args.min_radius_px),
            residual_against_camera=True,
            depth_min=0.02,
            background=background,
        )
        lr_signal = np.maximum(lr_render["residual_max"], lr_render["large_residual_max"])
        lr_bad_score = np.clip(
            (lr_signal - float(args.lr_bad_residual_threshold)) / max(1.0 - float(args.lr_bad_residual_threshold), 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)
        lr_metrics = {
            "residual_mean": stats_from_array(lr_render["residual_mean"]),
            "residual_max": stats_from_array(lr_render["residual_max"]),
            "large_residual_max": stats_from_array(lr_render["large_residual_max"]),
            "lr_bad_score": stats_from_array(lr_bad_score),
            "config": {
                "lr_bad_residual_threshold": float(args.lr_bad_residual_threshold),
            },
        }

    prune_mask, soft_mask, delete_score, delete_info = build_cleanup_masks(
        blur_score=blur_score,
        sr_support_score=sr_support_score,
        prior_detail_score=prior_detail_score,
        render_detail_score=render_detail_score,
        detail_gap_score=detail_gap_score,
        sr_residual_score=sr_residual_score,
        lowfreq_score=lowfreq_score,
        lr_bad_score=lr_bad_score,
        footprint_risk=footprint_risk,
        opacity=opacity,
        visible_count=visible_count,
        num_score_views=len(sr_cameras),
        radius_max=radius_max.astype(np.float32),
        visible_mask=visible_mask,
        delete_quantile=float(args.delete_quantile),
        min_blur_score=float(args.min_blur_score),
        min_sr_support=float(args.min_sr_support),
        min_prior_detail=float(args.min_prior_detail),
        min_detail_gap=float(args.min_detail_gap),
        min_lowfreq_score=float(args.min_lowfreq_score),
        min_sr_residual=float(args.min_sr_residual),
        min_footprint_risk=float(args.min_footprint_risk),
        min_radius_px=float(args.min_radius_px),
        prune_max_opacity=float(args.prune_max_opacity),
        prune_max_visible_fraction=float(args.prune_max_visible_fraction),
        max_prune_fraction=float(args.max_prune_fraction),
        max_prune_count=int(args.max_prune_count),
        soft_max_opacity=float(args.soft_max_opacity),
        max_soft_fraction=float(args.max_soft_fraction),
        max_soft_count=int(args.max_soft_count),
    )
    if np.any(soft_mask):
        apply_soft_suppression(
            mip,
            soft_mask,
            opacity_scale=float(args.soft_opacity_scale),
            scale_shrink=float(args.soft_scale_shrink),
        )
    keep_mask = ~prune_mask

    cleaned_model = _write_masked_model(
        mip,
        keep_mask,
        output_model_path.parent,
        output_model_path.name,
        mip_model_path,
        int(args.output_iteration),
        meta={
            "source_model_path": str(mip_model_path),
            "source_iteration": int(iteration),
            "prune_count": int(np.count_nonzero(prune_mask)),
            "soft_suppress_count": int(np.count_nonzero(soft_mask)),
        },
    )
    deleted_model = None
    if bool(args.export_deleted_model) and np.any(prune_mask):
        deleted_model = _write_masked_model(
            mip,
            prune_mask,
            output_model_path.parent,
            output_model_path.name + "_deleted_blur_artifacts",
            mip_model_path,
            int(args.output_iteration),
            meta={
                "source_model_path": str(mip_model_path),
                "source_iteration": int(iteration),
                "prune_count": int(np.count_nonzero(prune_mask)),
            },
        )

    payload_path = output_model_path / "mip_blur_cleanup_payload.pt"
    payload = {
        "version": "cleanup_mip_blur_artifacts_v0",
        "score_version": "lowfreq_veil_hybrid_v2",
        "blur_score": torch.from_numpy(blur_score)[:, None],
        "sr_support_score": torch.from_numpy(sr_support_score)[:, None],
        "prior_detail_score": torch.from_numpy(prior_detail_score)[:, None],
        "render_detail_score": torch.from_numpy(render_detail_score)[:, None],
        "detail_gap_score": torch.from_numpy(detail_gap_score)[:, None],
        "sr_residual_score": torch.from_numpy(sr_residual_score)[:, None],
        "lowfreq_score": torch.from_numpy(lowfreq_score)[:, None],
        "lr_bad_score": torch.from_numpy(lr_bad_score)[:, None],
        "footprint_risk": torch.from_numpy(footprint_risk)[:, None],
        "opacity": torch.from_numpy(opacity.astype(np.float32))[:, None],
        "visible_fraction": torch.from_numpy(visible_fraction.astype(np.float32))[:, None],
        "delete_score": torch.from_numpy(delete_score.astype(np.float32))[:, None],
        "prune_mask": torch.from_numpy(prune_mask.astype(bool))[:, None],
        "soft_suppress_mask": torch.from_numpy(soft_mask.astype(bool))[:, None],
        "keep_mask": torch.from_numpy(keep_mask.astype(bool))[:, None],
        "visible_count": torch.from_numpy(visible_count.astype(np.int64))[:, None],
        "radius_mean": torch.from_numpy(radius_mean.astype(np.float32))[:, None],
        "radius_max": torch.from_numpy(radius_max.astype(np.float32))[:, None],
        "meta": {
            "scene_root": str(scene_root),
            "mip_model_path": str(mip_model_path),
            "iteration": int(iteration),
            "output_model_path": str(output_model_path),
            "output_iteration": int(args.output_iteration),
            "sr_prior_root": str(sr_prior_root),
            "selected_sr_views": [str(cam.image_name) for cam in sr_cameras],
            "args": vars(args),
            "delete_info": delete_info,
        },
    }
    torch.save(payload, payload_path)

    summary = {
        "version": "cleanup_mip_blur_artifacts_v0",
        "score_version": "lowfreq_veil_hybrid_v2",
        "scene_root": str(scene_root),
        "mip_model_path": str(mip_model_path),
        "iteration": int(iteration),
        "output_model_path": str(output_model_path),
        "output_iteration": int(args.output_iteration),
        "cleaned_model": cleaned_model,
        "deleted_model": deleted_model,
        "sr_prior_root": str(sr_prior_root),
        "num_gaussians": int(num_gaussians),
        "visible_gaussians": int(np.count_nonzero(visible_mask)),
        "pruned_count": int(np.count_nonzero(prune_mask)),
        "pruned_ratio": float(np.mean(prune_mask)),
        "soft_suppressed_count": int(np.count_nonzero(soft_mask)),
        "soft_suppressed_ratio": float(np.mean(soft_mask)),
        "delete_info": delete_info,
        "score_stats_visible": {
            "blur_score": _stats(blur_score, visible_mask),
            "sr_support_score": _stats(sr_support_score, visible_mask),
            "prior_detail_score": _stats(prior_detail_score, visible_mask),
            "render_detail_score": _stats(render_detail_score, visible_mask),
            "detail_gap_score": _stats(detail_gap_score, visible_mask),
            "sr_residual_score": _stats(sr_residual_score, visible_mask),
            "lowfreq_score": _stats(lowfreq_score, visible_mask),
            "lr_bad_score": _stats(lr_bad_score, visible_mask),
            "footprint_risk": _stats(footprint_risk, visible_mask),
            "opacity": _stats(opacity, visible_mask),
            "visible_fraction": _stats(visible_fraction, visible_mask),
            "delete_score": _stats(delete_score, visible_mask),
            "radius_max": _stats(radius_max.astype(np.float32), visible_mask),
        },
        "score_stats_pruned": {
            "blur_score": _stats(blur_score, prune_mask),
            "sr_support_score": _stats(sr_support_score, prune_mask),
            "prior_detail_score": _stats(prior_detail_score, prune_mask),
            "render_detail_score": _stats(render_detail_score, prune_mask),
            "detail_gap_score": _stats(detail_gap_score, prune_mask),
            "sr_residual_score": _stats(sr_residual_score, prune_mask),
            "lowfreq_score": _stats(lowfreq_score, prune_mask),
            "lr_bad_score": _stats(lr_bad_score, prune_mask),
            "footprint_risk": _stats(footprint_risk, prune_mask),
            "opacity": _stats(opacity, prune_mask),
            "visible_fraction": _stats(visible_fraction, prune_mask),
            "delete_score": _stats(delete_score, prune_mask),
            "radius_max": _stats(radius_max.astype(np.float32), prune_mask),
        },
        "score_stats_soft_suppressed": {
            "blur_score": _stats(blur_score, soft_mask),
            "sr_support_score": _stats(sr_support_score, soft_mask),
            "prior_detail_score": _stats(prior_detail_score, soft_mask),
            "render_detail_score": _stats(render_detail_score, soft_mask),
            "detail_gap_score": _stats(detail_gap_score, soft_mask),
            "sr_residual_score": _stats(sr_residual_score, soft_mask),
            "lowfreq_score": _stats(lowfreq_score, soft_mask),
            "lr_bad_score": _stats(lr_bad_score, soft_mask),
            "footprint_risk": _stats(footprint_risk, soft_mask),
            "opacity": _stats(opacity, soft_mask),
            "visible_fraction": _stats(visible_fraction, soft_mask),
            "delete_score": _stats(delete_score, soft_mask),
            "radius_max": _stats(radius_max.astype(np.float32), soft_mask),
        },
        "lr_metrics": lr_metrics,
        "payload": str(payload_path),
        "debug_views_dir": str(debug_dir),
        "view_summaries": view_summaries,
        "args": vars(args),
    }
    summary_path = output_model_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] cleaned mip model: {cleaned_model}")
    print(f"[done] output ply        : {output_model_path}/point_cloud/iteration_{int(args.output_iteration)}/point_cloud.ply")
    print(f"[done] summary           : {summary_path}")
    print(f"[done] payload           : {payload_path}")


if __name__ == "__main__":
    main()
