from __future__ import annotations

import json
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import (
    copy_render_config,
    build_dataset_args,
    load_model_ply,
    load_train_cameras_only,
    resolve_iteration,
    select_uniform,
)
from utils.general_utils import inverse_sigmoid
from utils.prior_fusion import project_points_camera
from utils.system_utils import mkdir_p


def camera_image_chw(camera, device: torch.device) -> torch.Tensor:
    image = camera.original_image
    if image.ndim != 3:
        raise ValueError(f"camera.original_image must be 3D, got {tuple(image.shape)}")
    if image.shape[0] == 3:
        return image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported camera.original_image shape: {tuple(image.shape)}")


def stats_from_array(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def static_gaussian_metrics(gaussians: GaussianModel) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
        scaling = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32, copy=False)
        xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    scale_max = scaling.max(axis=1)
    scale_min = np.clip(scaling.min(axis=1), 1e-8, None)
    return {
        "xyz": xyz,
        "opacity": opacity,
        "scale_max": scale_max.astype(np.float32, copy=False),
        "scale_min": scale_min.astype(np.float32, copy=False),
        "anisotropy": (scale_max / scale_min).astype(np.float32, copy=False),
    }


@torch.no_grad()
def collect_render_metrics(
    gaussians: GaussianModel,
    cameras: Sequence[object],
    *,
    large_radius_px: float,
    residual_against_camera: bool,
    depth_min: float,
    background: torch.Tensor,
) -> Dict[str, np.ndarray]:
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    total = int(xyz.shape[0])
    visible_count = np.zeros((total,), dtype=np.int32)
    large_view_count = np.zeros((total,), dtype=np.int32)
    max_radius = np.zeros((total,), dtype=np.float32)
    radius_sum = np.zeros((total,), dtype=np.float64)
    residual_sum = np.zeros((total,), dtype=np.float64)
    residual_count = np.zeros((total,), dtype=np.int32)
    residual_max = np.zeros((total,), dtype=np.float32)
    large_residual_sum = np.zeros((total,), dtype=np.float64)
    large_residual_count = np.zeros((total,), dtype=np.int32)
    large_residual_max = np.zeros((total,), dtype=np.float32)

    for camera in tqdm(cameras, desc="render metrics"):
        package = render_simple(camera, gaussians, background)
        radii = package["radii"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        active = radii > 0
        if not np.any(active):
            continue
        projected, valid = project_points_camera(camera, xyz, depth_min=float(depth_min), margin=0)
        active &= valid
        active_ids = np.flatnonzero(active).astype(np.int64, copy=False)
        if active_ids.size == 0:
            continue

        active_radii = radii[active_ids]
        visible_count[active_ids] += 1
        radius_sum[active_ids] += active_radii.astype(np.float64)
        max_radius[active_ids] = np.maximum(max_radius[active_ids], active_radii)
        large_ids = active_ids[active_radii >= float(large_radius_px)]
        if large_ids.size > 0:
            large_view_count[large_ids] += 1

        if not residual_against_camera:
            continue

        render_rgb = package["render"].detach().clamp(0.0, 1.0)
        anchor_rgb = camera_image_chw(camera, background.device)
        residual_map = torch.mean(torch.abs(render_rgb - anchor_rgb), dim=0).detach().cpu().numpy().astype(np.float32)
        xy = projected[active_ids, :2]
        x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, residual_map.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, residual_map.shape[0] - 1)
        sampled = residual_map[y, x]
        residual_sum[active_ids] += sampled.astype(np.float64)
        residual_count[active_ids] += 1
        residual_max[active_ids] = np.maximum(residual_max[active_ids], sampled)
        if large_ids.size > 0:
            large_xy = projected[large_ids, :2]
            lx = np.clip(np.rint(large_xy[:, 0]).astype(np.int64), 0, residual_map.shape[1] - 1)
            ly = np.clip(np.rint(large_xy[:, 1]).astype(np.int64), 0, residual_map.shape[0] - 1)
            large_sampled = residual_map[ly, lx]
            large_residual_sum[large_ids] += large_sampled.astype(np.float64)
            large_residual_count[large_ids] += 1
            large_residual_max[large_ids] = np.maximum(large_residual_max[large_ids], large_sampled)

    residual_mean = np.divide(
        residual_sum,
        np.maximum(residual_count, 1),
        out=np.zeros_like(residual_sum, dtype=np.float64),
        where=residual_count > 0,
    ).astype(np.float32)
    large_residual_mean = np.divide(
        large_residual_sum,
        np.maximum(large_residual_count, 1),
        out=np.zeros_like(large_residual_sum, dtype=np.float64),
        where=large_residual_count > 0,
    ).astype(np.float32)
    radius_mean = np.divide(
        radius_sum,
        np.maximum(visible_count, 1),
        out=np.zeros_like(radius_sum, dtype=np.float64),
        where=visible_count > 0,
    ).astype(np.float32)
    return {
        "visible_count": visible_count,
        "large_view_count": large_view_count,
        "max_radius": max_radius,
        "mean_radius": radius_mean,
        "residual_mean": residual_mean,
        "residual_max": residual_max,
        "large_residual_mean": large_residual_mean,
        "large_residual_max": large_residual_max,
        "large_residual_count": large_residual_count,
    }


def load_tracking(path: Path, total: int) -> Dict[str, np.ndarray]:
    tags_path = path / "gaussian_tags.pt"
    if not tags_path.is_file():
        return {
            "source_tag": np.zeros((total,), dtype=np.int32),
            "seed_id": np.full((total,), -1, dtype=np.int64),
            "generation": np.zeros((total,), dtype=np.int32),
        }
    payload = torch.load(tags_path, map_location="cpu")
    out = {}
    for key, dtype, default in (
        ("source_tag", np.int32, 0),
        ("seed_id", np.int64, -1),
        ("generation", np.int32, 0),
    ):
        value = payload.get(key)
        if torch.is_tensor(value) and value.numel() == total:
            out[key] = value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
        else:
            out[key] = np.full((total,), default, dtype=dtype)
    return out


def build_lineage_map(
    mip_point_dir: Path,
    sof_point_dir: Path,
    mip_xyz: np.ndarray,
    sof_xyz: np.ndarray,
) -> Dict[str, object]:
    mip_total = int(mip_xyz.shape[0])
    sof_total = int(sof_xyz.shape[0])
    bbox_diag = float(np.linalg.norm(mip_xyz.max(axis=0) - mip_xyz.min(axis=0)))
    bbox_diag = max(bbox_diag, 1e-6)
    mapped_sof_id = np.full((mip_total,), -1, dtype=np.int64)
    child_count = np.zeros((mip_total,), dtype=np.int32)
    displacement = np.full((mip_total,), np.nan, dtype=np.float32)
    mode = "none"

    if mip_total == sof_total:
        mode = "index"
        mapped_sof_id = np.arange(mip_total, dtype=np.int64)
        child_count.fill(1)
        displacement = (np.linalg.norm(sof_xyz - mip_xyz, axis=1) / bbox_diag).astype(np.float32, copy=False)
    else:
        mip_tracking = load_tracking(mip_point_dir, mip_total)
        sof_tracking = load_tracking(sof_point_dir, sof_total)
        mip_seed = mip_tracking["seed_id"]
        sof_seed = sof_tracking["seed_id"]
        if np.count_nonzero(mip_seed >= 0) > 0 and np.count_nonzero(sof_seed >= 0) > 0:
            mode = "seed_id"
            seed_to_sof: Dict[int, List[int]] = {}
            for idx, seed in enumerate(sof_seed.tolist()):
                if int(seed) < 0:
                    continue
                seed_to_sof.setdefault(int(seed), []).append(int(idx))
            for mip_idx, seed in enumerate(mip_seed.tolist()):
                children = seed_to_sof.get(int(seed), []) if int(seed) >= 0 else []
                if not children:
                    continue
                children_arr = np.asarray(children, dtype=np.int64)
                child_count[mip_idx] = int(children_arr.size)
                center = np.mean(sof_xyz[children_arr], axis=0)
                mapped_sof_id[mip_idx] = int(children_arr[0])
                displacement[mip_idx] = float(np.linalg.norm(center - mip_xyz[mip_idx]) / bbox_diag)

    return {
        "mode": mode,
        "mapped_sof_id": mapped_sof_id,
        "child_count": child_count,
        "displacement_norm": displacement,
    }


def cap_mask(mask: np.ndarray, score: np.ndarray, *, max_fraction: float, max_count: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    selected_ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if selected_ids.size == 0:
        return mask
    cap = selected_ids.size
    if float(max_fraction) > 0.0:
        cap = min(cap, max(1, int(round(mask.shape[0] * float(max_fraction)))))
    if int(max_count) > 0:
        cap = min(cap, int(max_count))
    if selected_ids.size <= cap:
        return mask
    order = np.argsort(-score[selected_ids], kind="stable")[:cap]
    capped = np.zeros_like(mask, dtype=bool)
    capped[selected_ids[order]] = True
    return capped


def build_mip_blob_judge(
    static: Dict[str, np.ndarray],
    render: Dict[str, np.ndarray],
    *,
    min_radius_px: float,
    max_large_views: int,
    max_visible_views: int,
    lr_residual_threshold: float,
    max_blob_opacity: float,
    max_fraction: float,
    max_count: int,
) -> Dict[str, np.ndarray]:
    opacity = static["opacity"]
    radius_excess = np.clip((render["max_radius"] - float(min_radius_px)) / max(float(min_radius_px), 1.0), 0.0, 4.0)
    residual_excess = np.clip((render["large_residual_max"] - float(lr_residual_threshold)) / max(1.0 - float(lr_residual_threshold), 1e-6), 0.0, 1.0)
    opacity_score = np.clip((float(max_blob_opacity) - opacity) / max(float(max_blob_opacity), 1e-6), 0.0, 1.0)
    single_view_score = 1.0 / np.maximum(render["large_view_count"].astype(np.float32), 1.0)
    score = (radius_excess + 2.0 * residual_excess + opacity_score + single_view_score).astype(np.float32)
    mask = (
        (render["max_radius"] >= float(min_radius_px))
        & (render["large_view_count"] > 0)
        & (render["large_view_count"] <= int(max_large_views))
        & (render["visible_count"] <= int(max_visible_views))
        & (render["large_residual_max"] >= float(lr_residual_threshold))
        & (opacity <= float(max_blob_opacity))
    )
    selected = cap_mask(mask, score, max_fraction=float(max_fraction), max_count=int(max_count))
    return {"candidate_mask": mask.astype(bool), "selected_mask": selected.astype(bool), "score": score}


def build_sof_spike_judge(
    static: Dict[str, np.ndarray],
    render: Dict[str, np.ndarray],
    *,
    anisotropy_threshold: float,
    min_radius_px: float,
    max_large_views: int,
    max_visible_views: int,
    max_fraction: float,
    max_count: int,
) -> Dict[str, np.ndarray]:
    anisotropy = static["anisotropy"]
    anisotropy_score = np.clip(np.log(np.maximum(anisotropy, 1.0)) / max(np.log(float(anisotropy_threshold)), 1e-6) - 1.0, 0.0, 4.0)
    radius_score = np.clip((render["max_radius"] - float(min_radius_px)) / max(float(min_radius_px), 1.0), 0.0, 4.0)
    view_score = 1.0 / np.maximum(render["large_view_count"].astype(np.float32), 1.0)
    score = (2.0 * anisotropy_score + radius_score + view_score).astype(np.float32)
    mask = (
        (anisotropy >= float(anisotropy_threshold))
        & (render["max_radius"] >= float(min_radius_px))
        & (render["large_view_count"] > 0)
        & (render["large_view_count"] <= int(max_large_views))
        & (render["visible_count"] <= int(max_visible_views))
    )
    selected = cap_mask(mask, score, max_fraction=float(max_fraction), max_count=int(max_count))
    return {"candidate_mask": mask.astype(bool), "selected_mask": selected.astype(bool), "score": score}


def apply_manual_prune(gaussians: GaussianModel, prune_mask_np: np.ndarray) -> None:
    keep = torch.from_numpy(~prune_mask_np.astype(bool)).to(device=gaussians.get_xyz.device, dtype=torch.bool)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        tensor = getattr(gaussians, name)
        setattr(gaussians, name, nn.Parameter(tensor.detach()[keep].clone().requires_grad_(False)))
    for name in ("max_radii2D", "xyz_gradient_accum", "denom", "filter_3D", "_source_tag", "_seed_id", "_generation", "_edge_touched", "_edge_touch_iter"):
        tensor = getattr(gaussians, name, None)
        if torch.is_tensor(tensor) and tensor.ndim > 0 and tensor.shape[0] == keep.shape[0]:
            setattr(gaussians, name, tensor.detach()[keep].clone())


def apply_soft_suppression(
    gaussians: GaussianModel,
    suppress_mask_np: np.ndarray,
    *,
    opacity_scale: float,
    scale_shrink: float,
) -> None:
    if not np.any(suppress_mask_np):
        return
    mask = torch.from_numpy(suppress_mask_np.astype(bool)).to(device=gaussians.get_xyz.device, dtype=torch.bool)
    with torch.no_grad():
        opacity = gaussians.get_opacity.detach().clone()
        opacity[mask] = torch.clamp(opacity[mask] * float(opacity_scale), min=1e-5, max=0.999)
        scaling = gaussians.get_scaling.detach().clone()
        scaling[mask] = torch.clamp(scaling[mask] * float(scale_shrink), min=1e-8)
    gaussians._opacity = nn.Parameter(inverse_sigmoid(opacity).requires_grad_(False))
    gaussians._scaling = nn.Parameter(torch.log(scaling).requires_grad_(False))


def save_payload(path: Path, payload: Dict[str, object]) -> None:
    tensor_payload = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            tensor_payload[key] = {
                sub_key: torch.from_numpy(sub_value.copy()) if isinstance(sub_value, np.ndarray) else sub_value
                for sub_key, sub_value in value.items()
            }
        elif isinstance(value, np.ndarray):
            tensor_payload[key] = torch.from_numpy(value.copy())
        else:
            tensor_payload[key] = value
    torch.save(tensor_payload, path)


def mask_summary(mask: np.ndarray, score: np.ndarray) -> Dict[str, object]:
    ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    return {
        "count": int(ids.size),
        "fraction": float(ids.size / max(mask.shape[0], 1)),
        "score": stats_from_array(score[ids] if ids.size > 0 else np.zeros((0,), dtype=np.float32)),
        "sample_ids": ids[:20].astype(int).tolist(),
    }


def main() -> None:
    parser = ArgumentParser(description="Conservative joint judge for mip/SOF prior fields.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--mip_model_path", required=True)
    parser.add_argument("--sof_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mip_iteration", type=int, default=-1)
    parser.add_argument("--sof_iteration", type=int, default=-1)
    parser.add_argument("--output_iteration", type=int, default=32000)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--large_radius_px", type=float, default=96.0)
    parser.add_argument("--mip_blob_max_large_views", type=int, default=1)
    parser.add_argument("--mip_blob_max_visible_views", type=int, default=4)
    parser.add_argument("--mip_blob_lr_residual_threshold", type=float, default=0.20)
    parser.add_argument("--mip_blob_max_opacity", type=float, default=0.55)
    parser.add_argument("--mip_blob_max_fraction", type=float, default=0.005)
    parser.add_argument("--mip_blob_max_count", type=int, default=2048)
    parser.add_argument("--sof_spike_anisotropy_threshold", type=float, default=40.0)
    parser.add_argument("--sof_spike_radius_px", type=float, default=48.0)
    parser.add_argument("--sof_spike_max_large_views", type=int, default=2)
    parser.add_argument("--sof_spike_max_visible_views", type=int, default=5)
    parser.add_argument("--sof_spike_max_fraction", type=float, default=0.01)
    parser.add_argument("--sof_spike_max_count", type=int, default=4096)
    parser.add_argument("--cleanup_mode", choices=["dry_run", "soft", "prune"], default="dry_run")
    parser.add_argument("--cleaned_mip_model_path", default="")
    parser.add_argument("--soft_opacity_scale", type=float, default=0.70)
    parser.add_argument("--soft_scale_shrink", type=float, default=0.90)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    sof_model_path = Path(args.sof_model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = build_dataset_args(str(scene_root), str(mip_model_path), str(args.images_subdir))
    cameras_all = load_train_cameras_only(scene_root, mip_model_path, str(args.images_subdir))
    cameras = select_uniform(cameras_all, int(args.max_views))
    if not cameras:
        raise RuntimeError("No cameras available for joint judge.")

    mip_iter = resolve_iteration(mip_model_path, int(args.mip_iteration))
    sof_iter = resolve_iteration(sof_model_path, int(args.sof_iteration))
    mip = load_model_ply(mip_model_path, mip_iter, int(dataset_args.sh_degree))
    sof = load_model_ply(sof_model_path, sof_iter, int(dataset_args.sh_degree))
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    mip.compute_3D_filter(cameras, CUDA=False)
    sof.compute_3D_filter(cameras, CUDA=False)

    print(f"[joint-judge] scene      : {scene_root}")
    print(f"[joint-judge] mip        : {mip_model_path} iter={mip_iter} n={mip.get_xyz.shape[0]}")
    print(f"[joint-judge] sof        : {sof_model_path} iter={sof_iter} n={sof.get_xyz.shape[0]}")
    print(f"[joint-judge] cameras    : {args.images_subdir} views={len(cameras)}")
    print(f"[joint-judge] mode       : {args.cleanup_mode}")

    mip_static = static_gaussian_metrics(mip)
    sof_static = static_gaussian_metrics(sof)
    mip_render = collect_render_metrics(
        mip,
        cameras,
        large_radius_px=float(args.large_radius_px),
        residual_against_camera=True,
        depth_min=float(args.depth_min),
        background=background,
    )
    sof_render = collect_render_metrics(
        sof,
        cameras,
        large_radius_px=float(args.sof_spike_radius_px),
        residual_against_camera=False,
        depth_min=float(args.depth_min),
        background=background,
    )
    lineage = build_lineage_map(
        mip_model_path / "point_cloud" / f"iteration_{mip_iter}",
        sof_model_path / "point_cloud" / f"iteration_{sof_iter}",
        mip_static["xyz"],
        sof_static["xyz"],
    )
    mip_blob = build_mip_blob_judge(
        mip_static,
        mip_render,
        min_radius_px=float(args.large_radius_px),
        max_large_views=int(args.mip_blob_max_large_views),
        max_visible_views=int(args.mip_blob_max_visible_views),
        lr_residual_threshold=float(args.mip_blob_lr_residual_threshold),
        max_blob_opacity=float(args.mip_blob_max_opacity),
        max_fraction=float(args.mip_blob_max_fraction),
        max_count=int(args.mip_blob_max_count),
    )
    sof_spike = build_sof_spike_judge(
        sof_static,
        sof_render,
        anisotropy_threshold=float(args.sof_spike_anisotropy_threshold),
        min_radius_px=float(args.sof_spike_radius_px),
        max_large_views=int(args.sof_spike_max_large_views),
        max_visible_views=int(args.sof_spike_max_visible_views),
        max_fraction=float(args.sof_spike_max_fraction),
        max_count=int(args.sof_spike_max_count),
    )

    payload = {
        "mip_static": {k: v for k, v in mip_static.items() if k != "xyz"},
        "mip_render": mip_render,
        "mip_blob": mip_blob,
        "sof_static": {k: v for k, v in sof_static.items() if k != "xyz"},
        "sof_render": sof_render,
        "sof_spike": sof_spike,
        "lineage": {
            "mapped_sof_id": lineage["mapped_sof_id"],
            "child_count": lineage["child_count"],
            "displacement_norm": lineage["displacement_norm"],
        },
    }
    payload_path = output_dir / "joint_judge_payload.pt"
    save_payload(payload_path, payload)

    cleaned_model_path = None
    if str(args.cleanup_mode) != "dry_run":
        if not args.cleaned_mip_model_path:
            raise ValueError("--cleaned_mip_model_path is required when cleanup_mode is not dry_run")
        cleaned_model_path = Path(args.cleaned_mip_model_path).expanduser().resolve()
        cleaned_model_path.mkdir(parents=True, exist_ok=True)
        if str(args.cleanup_mode) == "soft":
            apply_soft_suppression(
                mip,
                mip_blob["selected_mask"],
                opacity_scale=float(args.soft_opacity_scale),
                scale_shrink=float(args.soft_scale_shrink),
            )
        elif str(args.cleanup_mode) == "prune":
            apply_manual_prune(mip, mip_blob["selected_mask"])
        copy_render_config(mip_model_path, cleaned_model_path)
        point_dir = cleaned_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
        mkdir_p(str(point_dir))
        mip.save_ply(str(point_dir / "point_cloud.ply"))
        mip.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    summary = {
        "version": "joint_judge_mip_sof_v0",
        "scene_root": str(scene_root),
        "mip_model_path": str(mip_model_path),
        "mip_iteration": int(mip_iter),
        "sof_model_path": str(sof_model_path),
        "sof_iteration": int(sof_iter),
        "images_subdir": str(args.images_subdir),
        "views": [str(camera.image_name) for camera in cameras],
        "cleanup_mode": str(args.cleanup_mode),
        "cleaned_mip_model_path": None if cleaned_model_path is None else str(cleaned_model_path),
        "payload_path": str(payload_path),
        "lineage": {
            "mode": str(lineage["mode"]),
            "mip_count": int(mip_static["opacity"].shape[0]),
            "sof_count": int(sof_static["opacity"].shape[0]),
            "mapped_count": int(np.count_nonzero(lineage["mapped_sof_id"] >= 0)),
            "displacement_norm": stats_from_array(lineage["displacement_norm"]),
        },
        "mip": {
            "opacity": stats_from_array(mip_static["opacity"]),
            "scale_max": stats_from_array(mip_static["scale_max"]),
            "anisotropy": stats_from_array(mip_static["anisotropy"]),
            "max_radius": stats_from_array(mip_render["max_radius"]),
            "large_residual_max": stats_from_array(mip_render["large_residual_max"]),
            "blob_candidates": mask_summary(mip_blob["candidate_mask"], mip_blob["score"]),
            "blob_selected": mask_summary(mip_blob["selected_mask"], mip_blob["score"]),
        },
        "sof": {
            "opacity": stats_from_array(sof_static["opacity"]),
            "scale_max": stats_from_array(sof_static["scale_max"]),
            "anisotropy": stats_from_array(sof_static["anisotropy"]),
            "max_radius": stats_from_array(sof_render["max_radius"]),
            "spike_candidates": mask_summary(sof_spike["candidate_mask"], sof_spike["score"]),
            "spike_selected": mask_summary(sof_spike["selected_mask"], sof_spike["score"]),
        },
        "config": vars(args),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if cleaned_model_path is not None:
        shutil.copy2(summary_path, cleaned_model_path / "joint_judge_mip_sof_v0_summary.json")
        print(f"[done] cleaned mip model: {cleaned_model_path}")
    print(f"[done] payload: {payload_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
