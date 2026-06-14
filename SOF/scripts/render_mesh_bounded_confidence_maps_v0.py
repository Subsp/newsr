#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from joint_judge_mip_sof_v0 import stats_from_array
from train_mip_to_sof_surface_v0 import load_cameras_for_split, load_model_ply, resolve_iteration, select_uniform
from utils.general_utils import inverse_sigmoid
from utils.prior_fusion import project_points_camera
from utils.sh_utils import SH2RGB


def save_gray(path: Path, image_hw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_hw.astype(np.float32, copy=False), 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, image_chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_chw.astype(np.float32, copy=False), 0.0, 1.0)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def normalize_percentile(image_hw: np.ndarray, percentile: float) -> np.ndarray:
    denom = max(float(np.percentile(image_hw, float(percentile))), 1e-8)
    return np.clip(image_hw / denom, 0.0, 1.0).astype(np.float32, copy=False)


def log_normalize(image_hw: np.ndarray, gain: float) -> np.ndarray:
    gain = max(float(gain), 1e-8)
    return np.clip(np.log1p(gain * image_hw) / np.log1p(gain), 0.0, 1.0).astype(np.float32, copy=False)


def bilinear_sample_chw(image_chw: np.ndarray, xy: np.ndarray) -> np.ndarray:
    _, h, w = image_chw.shape
    x = np.clip(xy[:, 0], 0.0, max(float(w - 1), 0.0))
    y = np.clip(xy[:, 1], 0.0, max(float(h - 1), 0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    return (
        image_hwc[y0, x0] * wa[:, None]
        + image_hwc[y0, x1] * wb[:, None]
        + image_hwc[y1, x0] * wc[:, None]
        + image_hwc[y1, x1] * wd[:, None]
    ).astype(np.float32, copy=False)


def view_sampled_override_color(gaussians, camera, fallback_rgb: float = 0.0) -> torch.Tensor:
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    projected, valid = project_points_camera(camera, xyz, depth_min=0.01, margin=0)
    ref = camera.original_image[:3].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
    colors = np.full((xyz.shape[0], 3), float(fallback_rgb), dtype=np.float32)
    if np.any(valid):
        ids = np.flatnonzero(valid).astype(np.int64, copy=False)
        colors[ids] = bilinear_sample_chw(ref, projected[ids, :2])
    return torch.from_numpy(colors).to(device=gaussians.get_xyz.device, dtype=torch.float32)


def model_dc_rgb_np(gaussians) -> np.ndarray:
    with torch.no_grad():
        dc = gaussians._features_dc.detach()
        if gaussians.use_SBs:
            rgb = torch.clamp(dc[:, :3], 0.0, 1.0)
        else:
            rgb = torch.clamp(SH2RGB(dc[:, 0, :]), 0.0, 1.0)
    return rgb.detach().cpu().numpy().astype(np.float32, copy=False)


def render_projected_center_diagnostics(
    *,
    gaussians,
    camera,
    output_root: Path,
    view_index: int,
    model_rgb: np.ndarray,
) -> Dict[str, object]:
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    projected, valid = project_points_camera(camera, xyz, depth_min=0.01, margin=0)
    h = int(camera.image_height)
    w = int(camera.image_width)
    density = np.zeros((h, w), dtype=np.float32)
    nearest_idx = np.full((h * w,), -1, dtype=np.int64)
    if np.any(valid):
        ids = np.flatnonzero(valid).astype(np.int64, copy=False)
        x = np.clip(np.rint(projected[ids, 0]).astype(np.int64), 0, w - 1)
        y = np.clip(np.rint(projected[ids, 1]).astype(np.int64), 0, h - 1)
        pix = y * w + x
        np.add.at(density.reshape(-1), pix, 1.0)
        z = projected[ids, 2]
        order = np.argsort(z, kind="stable")[::-1]
        nearest_idx[pix[order]] = ids[order]

    hit = nearest_idx >= 0
    ref = camera.original_image[:3].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
    rgb_map = np.zeros((3, h, w), dtype=np.float32)
    err_map = np.zeros((h, w), dtype=np.float32)
    if np.any(hit):
        flat_rgb = np.zeros((h * w, 3), dtype=np.float32)
        flat_rgb[hit] = model_rgb[nearest_idx[hit]]
        rgb_map = np.transpose(flat_rgb.reshape(h, w, 3), (2, 0, 1))
        ref_hwc = np.transpose(ref, (1, 2, 0))
        err = np.mean(np.abs(flat_rgb[hit] - ref_hwc.reshape(-1, 3)[hit]), axis=1)
        err_map.reshape(-1)[hit] = err.astype(np.float32, copy=False)

    density_root = output_root / "center_density_log"
    rgb_root = output_root / "center_rgb_model_zbuf"
    err_root = output_root / "center_color_l1_error_model_zbuf"
    density_vis = log_normalize(density, gain=10.0)
    save_gray(density_root / f"{int(view_index):05d}.png", density_vis)
    save_rgb(rgb_root / f"{int(view_index):05d}.png", rgb_map)
    save_gray(err_root / f"{int(view_index):05d}.png", np.clip(err_map / 0.35, 0.0, 1.0))
    hit_err = err_map.reshape(-1)[hit]
    return {
        "center_valid_ratio": float(np.mean(hit.astype(np.float32))),
        "center_density": stats_from_array(density[density > 0]),
        "center_color_l1_error_model_zbuf": stats_from_array(hit_err if hit_err.size else np.empty((0,), dtype=np.float32)),
        "center_density_root": str(density_root),
        "center_rgb_model_zbuf_root": str(rgb_root),
        "center_color_l1_error_model_zbuf_root": str(err_root),
    }


def scale_model_opacity(gaussians, scale: float) -> Dict[str, float]:
    with torch.no_grad():
        original_alpha = gaussians.get_opacity.detach()
        scaled_alpha = torch.clamp(original_alpha * float(scale), 1e-6, 1.0 - 1e-6)
        gaussians._opacity.data.copy_(inverse_sigmoid(scaled_alpha))
    return {
        "source_opacity_mean": float(original_alpha.mean().item()),
        "source_opacity_p95": float(torch.quantile(original_alpha.reshape(-1), 0.95).item()),
        "scaled_opacity_mean": float(scaled_alpha.mean().item()),
        "scaled_opacity_p95": float(torch.quantile(scaled_alpha.reshape(-1), 0.95).item()),
    }


def set_model_opacity_from_base(gaussians, base_alpha: torch.Tensor, scale: float) -> Dict[str, float]:
    with torch.no_grad():
        scaled_alpha = torch.clamp(base_alpha * float(scale), 1e-8, 1.0 - 1e-6)
        gaussians._opacity.data.copy_(inverse_sigmoid(scaled_alpha))
    return {
        "source_opacity_mean": float(base_alpha.mean().item()),
        "source_opacity_p95": float(torch.quantile(base_alpha.reshape(-1), 0.95).item()),
        "scaled_opacity_mean": float(scaled_alpha.mean().item()),
        "scaled_opacity_p95": float(torch.quantile(scaled_alpha.reshape(-1), 0.95).item()),
    }


def calibrate_opacity_scale(
    *,
    gaussians,
    base_alpha: torch.Tensor,
    camera,
    initial_scale: float,
    target_alpha_mean: float,
    min_scale: float,
    max_steps: int,
    background: torch.Tensor,
    scale_modifier: float,
) -> Dict[str, object]:
    scale = float(initial_scale)
    history = []
    for _ in range(max(int(max_steps), 1)):
        set_model_opacity_from_base(gaussians, base_alpha, scale)
        pkg = render_simple(camera, gaussians, background, scaling_modifier=float(scale_modifier))
        alpha = pkg["alpha"].detach().float()
        alpha_mean = float(alpha.mean().item())
        alpha_p95 = float(torch.quantile(alpha.reshape(-1), 0.95).item())
        history.append({"opacity_scale": float(scale), "alpha_mean": alpha_mean, "alpha_p95": alpha_p95})
        if alpha_mean <= float(target_alpha_mean) or scale <= float(min_scale):
            return {
                "enabled": True,
                "selected_opacity_scale": float(scale),
                "target_alpha_mean": float(target_alpha_mean),
                "min_scale": float(min_scale),
                "history": history,
            }
        scale *= 0.1
    return {
        "enabled": True,
        "selected_opacity_scale": float(scale),
        "target_alpha_mean": float(target_alpha_mean),
        "min_scale": float(min_scale),
        "history": history,
    }


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src = src_model_path / name
        if src.exists():
            shutil.copy2(src, dst_model_path / name)


def write_debug_visible_model(
    *,
    source_model_path: Path,
    output_root: Path,
    iteration: int,
    alpha: float,
    scale_multiplier: float,
    color_rgb: Sequence[float],
) -> Dict[str, object]:
    debug_model = load_model_ply(source_model_path, iteration=iteration, sh_degree=3)
    debug_model.active_sh_degree = 0
    with torch.no_grad():
        target_alpha = torch.full_like(debug_model.get_opacity, float(alpha)).clamp(1e-6, 1.0 - 1e-6)
        debug_model._opacity.data.copy_(inverse_sigmoid(target_alpha))
        debug_model._scaling.data.add_(float(np.log(max(float(scale_multiplier), 1e-6))))
        color = torch.tensor(list(color_rgb), dtype=torch.float32, device="cuda").clamp(0.0, 1.0)
        sh = rgb_to_sh(color)
        if debug_model.use_SBs:
            debug_model._features_dc.data[:, :3] = sh[None, :]
        else:
            debug_model._features_dc.data[:, 0, :] = sh[None, :]
        debug_model._features_rest.data.zero_()

    model_path = output_root / "debug_visible_mesh_bounded_gaussian_model_v0"
    point_dir = model_path / "point_cloud" / f"iteration_{int(iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    copy_render_config(source_model_path, model_path)
    debug_model.save_ply(str(point_dir / "point_cloud.ply"))
    debug_model.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    return {
        "model_path": str(model_path),
        "ply": str(point_dir / "point_cloud.ply"),
        "alpha": float(alpha),
        "scale_multiplier": float(scale_multiplier),
        "color_rgb": [float(v) for v in color_rgb],
        "count": int(debug_model.get_xyz.shape[0]),
    }


def render_maps(
    *,
    gaussians,
    cameras: Sequence[object],
    output_root: Path,
    background: torch.Tensor,
    pnorm_percentile: float,
    log_gain: float,
    color_error_alpha_min: float,
    scale_modifier: float,
    view_color_mode: str,
    render_center_diagnostics: bool,
) -> Dict[str, object]:
    alpha_root = output_root / "alpha"
    pnorm_root = output_root / f"alpha_p{int(round(float(pnorm_percentile)))}norm"
    log_root = output_root / "alpha_log"
    rgb_root = output_root / "rgb_direct"
    unpremul_root = output_root / "rgb_unpremul"
    err_root = output_root / "color_l1_error"
    selected: List[int] = []
    alpha_stats: List[Dict[str, object]] = []
    model_rgb = model_dc_rgb_np(gaussians)
    for idx, camera in enumerate(tqdm(cameras, desc="render mesh-bounded confidence maps")):
        override_color = None
        if str(view_color_mode) == "camera_sample":
            override_color = view_sampled_override_color(gaussians, camera)
        pkg = render_simple(camera, gaussians, background, scaling_modifier=float(scale_modifier), override_color=override_color)
        alpha = pkg["alpha"].detach().float().squeeze().cpu().numpy().astype(np.float32, copy=False)
        rgb = pkg["render"][:3].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
        alpha_safe = np.maximum(alpha, 1e-6)
        rgb_unpremul = np.clip(rgb / alpha_safe[None, :, :], 0.0, 1.0).astype(np.float32, copy=False)
        ref = camera.original_image[:3].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
        valid = alpha >= float(color_error_alpha_min)
        color_error = np.mean(np.abs(rgb_unpremul - ref), axis=0).astype(np.float32, copy=False)
        color_error_vis = np.where(valid, color_error, 0.0).astype(np.float32, copy=False)
        save_gray(alpha_root / f"{idx:05d}.png", alpha)
        save_gray(pnorm_root / f"{idx:05d}.png", normalize_percentile(alpha, float(pnorm_percentile)))
        save_gray(log_root / f"{idx:05d}.png", log_normalize(alpha, float(log_gain)))
        save_rgb(rgb_root / f"{idx:05d}.png", rgb)
        save_rgb(unpremul_root / f"{idx:05d}.png", rgb_unpremul)
        save_gray(err_root / f"{idx:05d}.png", np.clip(color_error_vis / 0.35, 0.0, 1.0))
        selected.append(int(idx))
        alpha_stats.append(
            {
                "view_index": int(idx),
                "alpha": stats_from_array(alpha),
                "color_l1_error_valid": stats_from_array(color_error[valid] if np.any(valid) else np.empty((0,), dtype=np.float32)),
                "color_valid_ratio": float(np.mean(valid.astype(np.float32))),
                "center_diagnostics": render_projected_center_diagnostics(
                    gaussians=gaussians,
                    camera=camera,
                    output_root=output_root,
                    view_index=idx,
                    model_rgb=model_rgb,
                )
                if bool(render_center_diagnostics)
                else {"enabled": False},
                "alpha_pnorm_percentile": float(pnorm_percentile),
                "alpha_log_gain": float(log_gain),
            }
        )
    return {
        "view_count": int(len(cameras)),
        "selected_indices": selected,
        "alpha_root": str(alpha_root),
        "alpha_pnorm_root": str(pnorm_root),
        "alpha_log_root": str(log_root),
        "rgb_direct_root": str(rgb_root),
        "rgb_unpremul_root": str(unpremul_root),
        "color_l1_error_root": str(err_root),
        "alpha_stats": alpha_stats,
        "scale_modifier": float(scale_modifier),
        "view_color_mode": str(view_color_mode),
        "render_center_diagnostics": bool(render_center_diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render mesh-boundedGS confidence maps with reduced opacity to avoid saturated all-white alpha maps."
    )
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--opacity_scale", type=float, default=0.06)
    parser.add_argument("--auto_opacity_calibrate", action="store_true")
    parser.add_argument("--target_alpha_mean", type=float, default=0.25)
    parser.add_argument("--min_opacity_scale", type=float, default=1e-8)
    parser.add_argument("--opacity_calibration_steps", type=int, default=10)
    parser.add_argument("--render_scale_modifier", type=float, default=0.35)
    parser.add_argument("--view_color_mode", choices=["model", "camera_sample"], default="camera_sample")
    parser.add_argument("--render_center_diagnostics", action="store_true")
    parser.add_argument("--pnorm_percentile", type=float, default=99.0)
    parser.add_argument("--log_gain", type=float, default=30.0)
    parser.add_argument("--color_error_alpha_min", type=float, default=0.02)
    parser.add_argument("--write_debug_visible_model", action="store_true")
    parser.add_argument("--debug_visible_alpha", type=float, default=0.75)
    parser.add_argument("--debug_scale_multiplier", type=float, default=4.0)
    parser.add_argument("--debug_color_rgb", default="1.0,0.05,0.02")
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    iteration = resolve_iteration(model_path, int(args.iteration))
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else model_path / "surface_confidence_maps_low_opacity" / f"ours_{int(iteration)}_opscale_{float(args.opacity_scale):.4f}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    gaussians = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    gaussians.active_sh_degree = 0
    base_alpha = gaussians.get_opacity.detach().clone()
    cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    selected_cameras = select_uniform(cameras, int(args.max_views))
    if len(selected_cameras) <= 0:
        raise RuntimeError("No cameras selected for mesh-bounded confidence rendering.")
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(args.white_background) else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    opacity_calibration: Dict[str, object] = {"enabled": False}
    opacity_scale = float(args.opacity_scale)
    if bool(args.auto_opacity_calibrate):
        opacity_calibration = calibrate_opacity_scale(
            gaussians=gaussians,
            base_alpha=base_alpha,
            camera=selected_cameras[0],
            initial_scale=opacity_scale,
            target_alpha_mean=float(args.target_alpha_mean),
            min_scale=float(args.min_opacity_scale),
            max_steps=int(args.opacity_calibration_steps),
            background=background,
            scale_modifier=float(args.render_scale_modifier),
        )
        opacity_scale = float(opacity_calibration["selected_opacity_scale"])
    opacity_stats = set_model_opacity_from_base(gaussians, base_alpha, opacity_scale)
    render_summary = render_maps(
        gaussians=gaussians,
        cameras=selected_cameras,
        output_root=output_root,
        background=background,
        pnorm_percentile=float(args.pnorm_percentile),
        log_gain=float(args.log_gain),
        color_error_alpha_min=float(args.color_error_alpha_min),
        scale_modifier=float(args.render_scale_modifier),
        view_color_mode=str(args.view_color_mode),
        render_center_diagnostics=bool(args.render_center_diagnostics),
    )
    debug_model_summary: Dict[str, object] = {"enabled": False}
    if bool(args.write_debug_visible_model):
        color_rgb = [float(part) for part in str(args.debug_color_rgb).split(",")]
        if len(color_rgb) != 3:
            raise ValueError("--debug_color_rgb must contain three comma-separated floats")
        debug_model_summary = {
            "enabled": True,
            **write_debug_visible_model(
                source_model_path=model_path,
                output_root=output_root,
                iteration=iteration,
                alpha=float(args.debug_visible_alpha),
                scale_multiplier=float(args.debug_scale_multiplier),
                color_rgb=color_rgb,
            ),
        }
    summary = {
        "version": "render_mesh_bounded_confidence_maps_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(iteration),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "source_view_count": int(len(cameras)),
        "selected_view_count": int(len(selected_cameras)),
        "opacity_scale": float(opacity_scale),
        "requested_opacity_scale": float(args.opacity_scale),
        "render_scale_modifier": float(args.render_scale_modifier),
        "view_color_mode": str(args.view_color_mode),
        "opacity_calibration": opacity_calibration,
        "opacity_stats": opacity_stats,
        "color_error_alpha_min": float(args.color_error_alpha_min),
        "renders": render_summary,
        "debug_visible_model": debug_model_summary,
    }
    summary_path = output_root / "render_mesh_bounded_confidence_maps_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
