from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from build_soflr_vggt_bound_gs_correction_v0 import (
    _lookup_image,
    _project_points_camera,
    _robust_align_depth,
    _sample_nearest_2d,
    _save_heatmap,
    _save_point_cloud,
)
from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel
from apply_soflr_bound_gs_surface_correction_v0 import clone_with_xyz, copy_render_config
from utils.prior_injection import index_image_dir, load_rgb_image, normalize_image_name
from utils.sof_mesh_patch_enhancer_v0 import stats_from_array
from utils.system_utils import mkdir_p
from utils.vggt_adapter import FrozenVGGTAdapter, VGGTAdapterConfig


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, data_device: str) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=False,
        data_device=data_device,
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def _resolve_iteration(model_path: Path, iteration: int) -> int:
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


def _resize_chw(image_chw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(image_chw.shape[-2:]) == tuple(target_hw):
        return image_chw
    return F.interpolate(image_chw[None].float(), size=target_hw, mode="bilinear", align_corners=False)[0]


def _load_image_chw(path: Path, target_hw: Tuple[int, int]) -> torch.Tensor:
    image = load_rgb_image(path).permute(2, 0, 1).contiguous()
    return _resize_chw(image, target_hw)


def _camera_points_to_world(cam, xyz_cam: np.ndarray) -> np.ndarray:
    r = np.asarray(cam.R, dtype=np.float32)
    t = np.asarray(cam.T, dtype=np.float32)
    return ((xyz_cam - t[None, :]) @ r.T).astype(np.float32, copy=False)


def _depth_to_world_at_projected_pixels(cam, xy_depth: np.ndarray) -> np.ndarray:
    z = xy_depth[:, 2].astype(np.float32, copy=False)
    x_cam = (xy_depth[:, 0].astype(np.float32, copy=False) - float(cam.image_width) / 2.0) / float(cam.focal_x) * z
    y_cam = (xy_depth[:, 1].astype(np.float32, copy=False) - float(cam.image_height) / 2.0) / float(cam.focal_y) * z
    xyz_cam = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32, copy=False)
    return _camera_points_to_world(cam, xyz_cam)


def _displacement_colors(displacement: np.ndarray, active: np.ndarray) -> np.ndarray:
    disp = np.asarray(displacement, dtype=np.float32)
    active_disp = disp[active]
    limit = float(np.percentile(active_disp, 98)) if active_disp.size else float(np.percentile(disp, 98))
    limit = max(limit, 1e-6)
    x = np.clip(disp / limit, 0.0, 1.0)
    colors = np.zeros((disp.shape[0], 4), dtype=np.uint8)
    colors[:, 0] = (x * 255).astype(np.uint8)
    colors[:, 1] = ((1.0 - x) * 220).astype(np.uint8)
    colors[:, 2] = 32
    colors[:, 3] = np.where(active, 255, 32).astype(np.uint8)
    return colors


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser(description="Attach visible SOFLR Gaussians to vanilla VGGT aligned depth and export a pseudoGS field.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_root", default="/root/autodl-tmp/vggt")
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--output_iteration", type=int, default=30000)
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vggt_cache", default=None)
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--min_alpha", type=float, default=0.08)
    parser.add_argument("--min_vggt_confidence", type=float, default=0.05)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--min_weight", type=float, default=0.02)
    parser.add_argument("--zbuffer_tolerance_abs", type=float, default=0.03)
    parser.add_argument("--zbuffer_tolerance_rel", type=float, default=0.015)
    parser.add_argument("--max_depth_residual_abs", type=float, default=0.0)
    parser.add_argument("--max_depth_residual_ratio", type=float, default=0.05)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--depth_align_min_pixels", type=int, default=2048)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    scene_root = Path(args.scene_root).expanduser().resolve()
    base_model_path = Path(args.base_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_iteration(base_model_path, int(args.load_iteration))
    dataset_args = _build_dataset_args(str(scene_root), str(base_model_path), str(args.images_subdir), data_device="cpu")
    dataset = ModelParams(None).extract(dataset_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    cameras = scene.getTrainCameras().copy()
    selected_cameras = _select_uniform(cameras, int(args.max_views))
    if not selected_cameras:
        raise RuntimeError("No cameras available for pseudoGS construction.")

    target_hw = (int(selected_cameras[0].image_height), int(selected_cameras[0].image_width))
    image_index = index_image_dir(str(scene_root / str(args.images_subdir)))
    lr_images = []
    image_names = []
    for camera in selected_cameras:
        image_path = _lookup_image(image_index, camera.image_name)
        lr_images.append(_load_image_chw(image_path, target_hw))
        image_names.append(normalize_image_name(camera.image_name))
    lr_images_t = torch.stack(lr_images, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)

    print(f"[vggt-pseudogs] selected views : {len(selected_cameras)}")
    print(f"[vggt-pseudogs] target hw      : {target_hw}")
    print("[vggt-pseudogs] render base SOFLR depth/alpha")
    gaussians.compute_3D_filter(selected_cameras, CUDA=False)
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    render_depths = []
    render_alphas = []
    base_renders = []
    for idx, camera in enumerate(selected_cameras):
        render_pkg = render_simple(camera, gaussians, background)
        render_depths.append(render_pkg["depth"][0].detach().cpu().numpy().astype(np.float32))
        render_alphas.append(render_pkg["alpha"][0].detach().cpu().numpy().astype(np.float32))
        if idx < 4:
            base_renders.append(render_pkg["render"][:3].detach().cpu())
            _save_heatmap(diag_dir / f"{idx:05d}_base_depth.png", render_depths[-1])
            _save_heatmap(diag_dir / f"{idx:05d}_base_alpha.png", render_alphas[-1])
            _save_rgb(diag_dir / f"{idx:05d}_base_render.png", base_renders[-1])

    print("[vggt-pseudogs] run vanilla VGGT")
    vggt_cache = Path(args.vggt_cache).expanduser().resolve() if args.vggt_cache else output_dir / "vggt_prior.pt"
    vggt = FrozenVGGTAdapter(
        VGGTAdapterConfig(
            vggt_root=str(Path(args.vggt_root).expanduser().resolve()),
            device=device,
        )
    )
    vggt_prior = vggt.run(
        lr_images_t,
        target_hw=target_hw,
        image_names=image_names,
        cache_path=vggt_cache,
    )
    vggt_depths = vggt_prior["depth_hr"][0, :, 0].detach().cpu().numpy().astype(np.float32)
    vggt_confs = vggt_prior["conf_hr"][0, :, 0].detach().cpu().numpy().astype(np.float32)

    aligned_vggt_depths = []
    align_summaries = []
    for idx, name in enumerate(image_names):
        align_mask = (render_alphas[idx] >= float(args.min_alpha)) & (vggt_confs[idx] >= float(args.min_vggt_confidence))
        aligned, align_summary = _robust_align_depth(
            vggt_depths[idx],
            render_depths[idx],
            align_mask,
            min_pixels=int(args.depth_align_min_pixels),
        )
        aligned_vggt_depths.append(aligned)
        align_summary["image_name"] = name
        align_summaries.append(align_summary)
        if idx < 4:
            _save_heatmap(diag_dir / f"{idx:05d}_vggt_depth_aligned.png", aligned)
            _save_heatmap(diag_dir / f"{idx:05d}_vggt_conf.png", vggt_confs[idx])
            _save_heatmap(diag_dir / f"{idx:05d}_vggt_minus_base_depth.png", aligned - render_depths[idx], symmetric=True)

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    total = int(xyz.shape[0])
    target_sum = np.zeros((total, 3), dtype=np.float64)
    weight_sum = np.zeros((total,), dtype=np.float64)
    view_count = np.zeros((total,), dtype=np.int32)
    residual_sum = np.zeros((total,), dtype=np.float64)
    abs_residual_sum = np.zeros((total,), dtype=np.float64)
    bbox_min = np.min(xyz, axis=0)
    bbox_max = np.max(xyz, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    bbox_diag = max(bbox_diag, 1e-6)

    print(f"[vggt-pseudogs] attach {total} GS centers to VGGT depth")
    for view_idx, camera in enumerate(selected_cameras):
        alpha = render_alphas[view_idx]
        base_depth = render_depths[view_idx]
        vggt_depth = aligned_vggt_depths[view_idx]
        vggt_conf = vggt_confs[view_idx]

        for start in range(0, total, int(args.chunk_size)):
            end = min(start + int(args.chunk_size), total)
            points = xyz[start:end]
            projected, frustum = _project_points_camera(camera, points, depth_min=float(args.depth_min))
            if not np.any(frustum):
                continue
            local_ids = np.flatnonzero(frustum).astype(np.int64, copy=False)
            global_ids = local_ids + start
            xy = projected[local_ids, :2]
            z = projected[local_ids, 2]
            sampled_alpha = _sample_nearest_2d(alpha, xy)
            sampled_base_depth = _sample_nearest_2d(base_depth, xy)
            sampled_vggt_depth = _sample_nearest_2d(vggt_depth, xy)
            sampled_conf = _sample_nearest_2d(vggt_conf, xy)

            zbuf_tol = float(args.zbuffer_tolerance_abs) + float(args.zbuffer_tolerance_rel) * np.maximum(sampled_base_depth, 1e-6)
            visible = np.abs(z - sampled_base_depth) <= zbuf_tol
            valid = visible
            valid &= sampled_alpha >= float(args.min_alpha)
            valid &= sampled_conf >= float(args.min_vggt_confidence)
            valid &= np.isfinite(sampled_vggt_depth) & (sampled_vggt_depth > float(args.depth_min))
            depth_residual = sampled_vggt_depth - z
            if float(args.max_depth_residual_abs) > 0.0:
                max_res = float(args.max_depth_residual_abs)
            else:
                max_res = float(args.max_depth_residual_ratio) * np.maximum(z, 1e-6)
            valid &= np.abs(depth_residual) <= max_res
            if not np.any(valid):
                continue

            use_ids = global_ids[valid]
            xy_depth = np.stack([xy[valid, 0], xy[valid, 1], sampled_vggt_depth[valid]], axis=1).astype(np.float32, copy=False)
            target_world = _depth_to_world_at_projected_pixels(camera, xy_depth)
            zbuf_gate = np.exp(-np.abs(z[valid] - sampled_base_depth[valid]) / np.maximum(zbuf_tol[valid], 1e-6))
            residual_gate = 1.0 - np.clip(np.abs(depth_residual[valid]) / np.maximum(max_res[valid] if np.ndim(max_res) else max_res, 1e-6), 0.0, 1.0)
            weights = (sampled_alpha[valid] * sampled_conf[valid] * zbuf_gate * residual_gate).astype(np.float32)
            good = weights > 1e-6
            if not np.any(good):
                continue
            use_ids = use_ids[good]
            weights = weights[good]
            target_world = target_world[good]
            residual = depth_residual[valid][good].astype(np.float32)

            target_sum[use_ids] += target_world.astype(np.float64) * weights[:, None].astype(np.float64)
            weight_sum[use_ids] += weights.astype(np.float64)
            view_count[use_ids] += 1
            residual_sum[use_ids] += residual.astype(np.float64) * weights.astype(np.float64)
            abs_residual_sum[use_ids] += np.abs(residual).astype(np.float64) * weights.astype(np.float64)

    active = (view_count >= int(args.min_views)) & (weight_sum >= float(args.min_weight))
    pseudo_xyz_raw = xyz.copy()
    pseudo_xyz_raw[active] = (target_sum[active] / np.clip(weight_sum[active, None], 1e-8, None)).astype(np.float32)
    blend = float(np.clip(args.blend, 0.0, 1.0))
    pseudo_xyz = (xyz * (1.0 - blend) + pseudo_xyz_raw * blend).astype(np.float32)
    displacement = np.linalg.norm(pseudo_xyz - xyz, axis=1).astype(np.float32)
    mean_residual = np.zeros((total,), dtype=np.float32)
    mean_abs_residual = np.zeros((total,), dtype=np.float32)
    mean_residual[active] = (residual_sum[active] / np.clip(weight_sum[active], 1e-8, None)).astype(np.float32)
    mean_abs_residual[active] = (abs_residual_sum[active] / np.clip(weight_sum[active], 1e-8, None)).astype(np.float32)

    output = clone_with_xyz(gaussians, pseudo_xyz)
    copy_render_config(base_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
    mkdir_p(str(point_dir))
    output.save_ply(str(point_dir / "point_cloud.ply"))
    output.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    payload_path = output_dir / "pseudogs_payload_v0.npz"
    np.savez_compressed(
        payload_path,
        version=np.asarray(["vggt_depth_pseudogs_v0"]),
        original_xyz=xyz,
        pseudo_xyz=pseudo_xyz,
        pseudo_xyz_raw=pseudo_xyz_raw,
        active_mask=active,
        displacement=displacement,
        weight_sum=weight_sum.astype(np.float32),
        view_count=view_count,
        mean_depth_residual=mean_residual,
        mean_abs_depth_residual=mean_abs_residual,
        bbox_diag=np.asarray([bbox_diag], dtype=np.float32),
    )

    colors = _displacement_colors(displacement, active)
    _save_point_cloud(output_dir / "original_gs_points_preview_v0.ply", xyz, max_points=int(args.preview_max_points))
    _save_point_cloud(output_dir / "pseudogs_points_preview_v0.ply", pseudo_xyz, colors=colors, max_points=int(args.preview_max_points))

    summary = {
        "version": "vggt_depth_pseudogs_v0",
        "scene_root": str(scene_root),
        "base_model_path": str(base_model_path),
        "output_model_path": str(output_model_path),
        "iteration": int(iteration),
        "output_iteration": int(args.output_iteration),
        "images_subdir": str(args.images_subdir),
        "selected_views": image_names,
        "target_hw": [int(target_hw[0]), int(target_hw[1])],
        "num_gaussians": total,
        "active_gaussians": int(np.sum(active)),
        "active_ratio": float(np.mean(active)),
        "bbox_diag": float(bbox_diag),
        "blend": blend,
        "gates": {
            "min_alpha": float(args.min_alpha),
            "min_vggt_confidence": float(args.min_vggt_confidence),
            "min_views": int(args.min_views),
            "min_weight": float(args.min_weight),
            "zbuffer_tolerance_abs": float(args.zbuffer_tolerance_abs),
            "zbuffer_tolerance_rel": float(args.zbuffer_tolerance_rel),
            "max_depth_residual_abs": float(args.max_depth_residual_abs),
            "max_depth_residual_ratio": float(args.max_depth_residual_ratio),
        },
        "displacement": stats_from_array(displacement),
        "displacement_active": stats_from_array(displacement[active]),
        "displacement_over_bbox_diag_active": stats_from_array(displacement[active] / bbox_diag),
        "view_count_active": stats_from_array(view_count[active].astype(np.float32)),
        "weight_sum_active": stats_from_array(weight_sum[active].astype(np.float32)),
        "mean_depth_residual_active": stats_from_array(mean_residual[active]),
        "mean_abs_depth_residual_active": stats_from_array(mean_abs_residual[active]),
        "depth_alignment": align_summaries,
        "payload": str(payload_path),
        "output_ply": str(point_dir / "point_cloud.ply"),
        "diagnostics_dir": str(diag_dir),
        "vggt_cache": str(vggt_cache),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": total, **summary}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
