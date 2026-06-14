import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import trimesh

from arguments import (
    MeshingParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    SplattingSettings,
    get_combined_args,
)
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.prior_fusion import bilinear_sample_rgb, project_points_camera, resize_rgb_image_np
from utils.prior_injection import (
    index_image_dir,
    load_mask,
    load_prior_image,
    normalize_image_name,
)
from utils.sh_utils import SH2RGB


def build_appearance_embedding(mesh_args, num_views: int):
    if mesh_args.use_decoupled_appearance:
        return AppearanceEmbedding(num_views=num_views)
    if mesh_args.use_pgsr_appearance:
        return PGSREmbedding(num_views=num_views)
    return None


def resolve_start_checkpoint(model_path: str, start_checkpoint: Optional[str], iteration: int) -> str:
    if start_checkpoint:
        return start_checkpoint
    if iteration < 0:
        raise ValueError("iteration must be explicit when start_checkpoint is not provided.")
    return os.path.join(model_path, f"chkpnt{iteration}.pth")


def lookup_indexed_image(index: Dict[str, Path], image_name: str):
    candidates = [
        image_name,
        normalize_image_name(image_name),
        Path(image_name).name,
        Path(image_name).stem,
    ]
    for key in candidates:
        if key in index:
            return index[key]
    lower_index = {str(key).lower(): value for key, value in index.items()}
    for key in candidates:
        value = lower_index.get(str(key).lower())
        if value is not None:
            return value
    return None


def stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def load_candidate_mask(candidate_payload_path: str, total_gaussians: int, key: str) -> Dict[str, np.ndarray]:
    payload = torch.load(candidate_payload_path, map_location="cpu")
    if key in payload:
        mask = payload[key].detach().cpu().numpy().astype(bool, copy=False)
        if mask.shape[0] != total_gaussians:
            raise ValueError(
                f"candidate mask length mismatch: {mask.shape[0]} vs total_gaussians={total_gaussians}"
            )
        selected_ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    elif "selected_ids" in payload:
        selected_ids = payload["selected_ids"].detach().cpu().numpy().astype(np.int64, copy=False)
        mask = np.zeros((total_gaussians,), dtype=bool)
        mask[selected_ids] = True
    else:
        raise KeyError(f"Candidate payload must contain '{key}' or 'selected_ids'.")

    nearest_normals = payload.get("nearest_surface_normal", None)
    if nearest_normals is not None:
        nearest_normals = nearest_normals.detach().cpu().numpy().astype(np.float32, copy=False)
        if nearest_normals.shape[0] != total_gaussians:
            nearest_normals = None

    return {
        "candidate_mask": mask,
        "selected_ids": selected_ids,
        "nearest_surface_normal": nearest_normals,
    }


def load_mask_np(mask_dir: Path, image_name: str, suffix: str):
    mask_path = mask_dir / f"{image_name}{suffix}"
    if not mask_path.is_file():
        return None, None
    mask = load_mask(mask_path).cpu().numpy() > 0.5
    return mask, mask_path


def save_fused_point_cloud(points_xyz: np.ndarray, fused_rgb: np.ndarray, valid_mask: np.ndarray, path: Path, max_points: int):
    valid_ids = np.flatnonzero(valid_mask)
    if valid_ids.size == 0:
        return
    if max_points > 0 and valid_ids.size > int(max_points):
        rng = np.random.default_rng(0)
        valid_ids = rng.choice(valid_ids, size=int(max_points), replace=False)
    colors = np.round(np.clip(fused_rgb[valid_ids], 0.0, 1.0) * 255.0).astype(np.uint8)
    trimesh.points.PointCloud(points_xyz[valid_ids], colors=colors).export(path)


def main():
    parser = ArgumentParser(description="Fuse prior colors directly onto selected existing SOFGS candidates.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    SplattingSettings(parser, render=False)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--candidate_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--fusion_mask_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default="_inject.png")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--screen_margin_px", type=int, default=0)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--min_views_per_gaussian", type=int, default=2)
    parser.add_argument("--opacity_weight_power", type=float, default=1.0)
    parser.add_argument("--view_angle_weight_power", type=float, default=0.0)
    parser.add_argument("--max_disagreement", type=float, default=0.0)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for direct GS fusion.")
        args.data_device = "cpu"

    safe_state(args.quiet)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe_args.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()

    appearance_embedding = build_appearance_embedding(mesh_args, num_views=len(train_cameras))
    gaussians.training_setup(opt_args, mesh_args, appearance_embedding)

    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    checkpoint_path = resolve_start_checkpoint(dataset.model_path, args.start_checkpoint, loaded_iteration)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_params, checkpoint_iteration, appearance_state = torch.load(checkpoint_path)
    if appearance_embedding is not None and appearance_state[0] is not None:
        appearance_embedding.restore(*appearance_state)
    gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    current_dc_rgb = SH2RGB(gaussians._features_dc[:, 0, :].detach()).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    total_gaussians = int(xyz.shape[0])

    candidate_payload = load_candidate_mask(
        candidate_payload_path=args.candidate_payload,
        total_gaussians=total_gaussians,
        key=args.candidate_mask_key,
    )
    candidate_mask = candidate_payload["candidate_mask"]
    candidate_ids = candidate_payload["selected_ids"]
    nearest_normals = candidate_payload["nearest_surface_normal"]
    if candidate_ids.size == 0:
        raise RuntimeError("Candidate payload selected zero gaussians.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_index = index_image_dir(args.prior_dir)
    mask_dir = Path(args.fusion_mask_dir)

    color_sum = np.zeros((total_gaussians, 3), dtype=np.float64)
    color_sq_sum = np.zeros((total_gaussians, 3), dtype=np.float64)
    weight_sum = np.zeros((total_gaussians,), dtype=np.float64)
    view_count = np.zeros((total_gaussians,), dtype=np.int32)

    prior_match_count = 0
    mask_match_count = 0
    processed_view_count = 0
    missing_prior_views = []
    missing_mask_views = []
    selected_cameras = train_cameras[:: max(int(args.camera_stride), 1)]

    print(
        f"[direct-gs-fusion] fusing {candidate_ids.size} candidate gaussians "
        f"across {len(selected_cameras)} train cameras"
    )
    for view_idx, view in enumerate(selected_cameras):
        if view_idx % 20 == 0:
            print(f"[direct-gs-fusion] view {view_idx + 1}/{len(selected_cameras)}: {view.image_name}")
        prior_path = lookup_indexed_image(prior_index, view.image_name)
        if prior_path is None:
            if len(missing_prior_views) < 16:
                missing_prior_views.append(view.image_name)
            continue
        mask, mask_path = load_mask_np(mask_dir, view.image_name, args.mask_suffix)
        if mask is None:
            if len(missing_mask_views) < 16:
                missing_mask_views.append(view.image_name)
            continue

        prior_match_count += 1
        mask_match_count += 1
        prior_image = load_prior_image(prior_path).cpu().numpy().astype(np.float32, copy=False)
        prior_image = resize_rgb_image_np(prior_image, mask.shape[0], mask.shape[1])

        projected, valid = project_points_camera(
            view,
            xyz[candidate_ids],
            depth_min=float(args.depth_min),
            margin=int(args.screen_margin_px),
        )
        if not np.any(valid):
            processed_view_count += 1
            continue

        local_valid_ids = np.flatnonzero(valid)
        mask_x = projected[local_valid_ids, 0] * (float(mask.shape[1]) / float(view.image_width))
        mask_y = projected[local_valid_ids, 1] * (float(mask.shape[0]) / float(view.image_height))
        pix_x = np.round(mask_x).astype(np.int64)
        pix_y = np.round(mask_y).astype(np.int64)
        in_bounds = (
            (pix_x >= 0)
            & (pix_x < mask.shape[1])
            & (pix_y >= 0)
            & (pix_y < mask.shape[0])
        )
        if not np.any(in_bounds):
            processed_view_count += 1
            continue
        local_valid_ids = local_valid_ids[in_bounds]
        pix_x = pix_x[in_bounds]
        pix_y = pix_y[in_bounds]
        in_mask = mask[pix_y, pix_x]
        if not np.any(in_mask):
            processed_view_count += 1
            continue

        local_hit_ids = local_valid_ids[in_mask]
        global_hit_ids = candidate_ids[local_hit_ids]
        sample_xy = projected[local_hit_ids, :2].copy()
        sample_xy[:, 0] *= float(prior_image.shape[1]) / float(view.image_width)
        sample_xy[:, 1] *= float(prior_image.shape[0]) / float(view.image_height)
        sampled_rgb = bilinear_sample_rgb(prior_image, sample_xy)

        weights = np.ones((global_hit_ids.shape[0],), dtype=np.float32)
        if float(args.opacity_weight_power) > 0.0:
            weights *= np.power(
                np.clip(opacity[global_hit_ids], 1e-6, 1.0),
                float(args.opacity_weight_power),
            ).astype(np.float32, copy=False)

        if nearest_normals is not None and float(args.view_angle_weight_power) > 0.0:
            cam_center = view.camera_center.detach().cpu().numpy().astype(np.float32, copy=False)
            view_dir = cam_center[None, :] - xyz[global_hit_ids]
            view_dir /= np.clip(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-6, None)
            facing = np.clip((nearest_normals[global_hit_ids] * view_dir).sum(axis=1), 0.0, 1.0)
            weights *= np.power(
                np.clip(facing, 1e-6, 1.0),
                float(args.view_angle_weight_power),
            ).astype(np.float32, copy=False)

        active_weight = weights > 1e-8
        if not np.any(active_weight):
            processed_view_count += 1
            continue
        global_hit_ids = global_hit_ids[active_weight]
        sampled_rgb = sampled_rgb[active_weight]
        weights = weights[active_weight]

        weight_sum[global_hit_ids] += weights.astype(np.float64)
        color_sum[global_hit_ids] += sampled_rgb.astype(np.float64) * weights[:, None].astype(np.float64)
        color_sq_sum[global_hit_ids] += (sampled_rgb.astype(np.float64) ** 2) * weights[:, None].astype(np.float64)
        view_count[global_hit_ids] += 1
        processed_view_count += 1

    fused_rgb = current_dc_rgb.copy()
    disagreement = np.zeros((total_gaussians,), dtype=np.float32)
    fused_candidate_mask = candidate_mask & (view_count >= int(args.min_views_per_gaussian)) & (weight_sum > 0.0)
    if np.any(fused_candidate_mask):
        fused_rgb[fused_candidate_mask] = (
            color_sum[fused_candidate_mask] / weight_sum[fused_candidate_mask, None]
        ).astype(np.float32)
        second_moment = color_sq_sum[fused_candidate_mask] / weight_sum[fused_candidate_mask, None]
        variance = np.clip(second_moment - fused_rgb[fused_candidate_mask].astype(np.float64) ** 2, 0.0, None)
        disagreement[fused_candidate_mask] = np.sqrt(variance.mean(axis=1)).astype(np.float32)

    if float(args.max_disagreement) > 0.0:
        valid_fused_mask = fused_candidate_mask & (disagreement <= float(args.max_disagreement))
    else:
        valid_fused_mask = fused_candidate_mask

    confidence = np.zeros((total_gaussians,), dtype=np.float32)
    if len(selected_cameras) > 0:
        confidence = (view_count.astype(np.float32) / float(len(selected_cameras))).astype(np.float32)

    payload_path = output_dir / "direct_gs_fusion_v0.pt"
    torch.save(
        {
            "candidate_mask": torch.from_numpy(candidate_mask.copy()),
            "fused_candidate_mask": torch.from_numpy(fused_candidate_mask.copy()),
            "valid_fused_mask": torch.from_numpy(valid_fused_mask.copy()),
            "selected_ids": torch.from_numpy(candidate_ids.astype(np.int64, copy=False)),
            "valid_fused_ids": torch.from_numpy(np.flatnonzero(valid_fused_mask).astype(np.int64, copy=False)),
            "fused_rgb": torch.from_numpy(fused_rgb.copy()),
            "current_dc_rgb": torch.from_numpy(current_dc_rgb.copy()),
            "confidence": torch.from_numpy(confidence.copy()),
            "disagreement": torch.from_numpy(disagreement.copy()),
            "view_count": torch.from_numpy(view_count.copy()),
            "weight_sum": torch.from_numpy(weight_sum.astype(np.float32, copy=False)),
        },
        str(payload_path),
    )

    preview_path = output_dir / "direct_gs_fused_candidates_v0.ply"
    save_fused_point_cloud(
        xyz,
        fused_rgb,
        valid_fused_mask,
        preview_path,
        max_points=int(args.preview_max_points),
    )

    valid_ids = np.flatnonzero(valid_fused_mask)
    fused_ids = np.flatnonzero(fused_candidate_mask)
    rgb_delta = np.linalg.norm(fused_rgb - current_dc_rgb, axis=1).astype(np.float32, copy=False)
    summary = {
        "mode": "direct_gs_fusion_v0",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "input_checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_iteration": int(checkpoint_iteration),
        "candidate_payload": str(Path(args.candidate_payload).resolve()),
        "prior_dir": str(Path(args.prior_dir).resolve()),
        "fusion_mask_dir": str(mask_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "payload_path": str(payload_path.resolve()),
        "preview_path": str(preview_path.resolve()),
        "camera_count_total": int(len(train_cameras)),
        "camera_count_used": int(len(selected_cameras)),
        "views_processed": int(processed_view_count),
        "prior_match_count": int(prior_match_count),
        "mask_match_count": int(mask_match_count),
        "missing_prior_view_samples": missing_prior_views,
        "missing_mask_view_samples": missing_mask_views,
        "parameters": {
            "candidate_mask_key": args.candidate_mask_key,
            "mask_suffix": args.mask_suffix,
            "depth_min": float(args.depth_min),
            "screen_margin_px": int(args.screen_margin_px),
            "camera_stride": int(args.camera_stride),
            "min_views_per_gaussian": int(args.min_views_per_gaussian),
            "opacity_weight_power": float(args.opacity_weight_power),
            "view_angle_weight_power": float(args.view_angle_weight_power),
            "max_disagreement": float(args.max_disagreement),
        },
        "counts": {
            "total_gaussians": int(total_gaussians),
            "candidate_count": int(candidate_ids.size),
            "fused_candidate_count": int(fused_ids.size),
            "valid_fused_count": int(valid_ids.size),
            "fused_candidate_ratio_vs_candidates": float(fused_ids.size / max(candidate_ids.size, 1)),
            "valid_fused_ratio_vs_candidates": float(valid_ids.size / max(candidate_ids.size, 1)),
            "valid_fused_ratio_vs_all": float(valid_ids.size / max(total_gaussians, 1)),
        },
        "stats": {
            "view_count_candidates": stats_from_array(view_count[candidate_ids].astype(np.float32)),
            "view_count_fused": stats_from_array(view_count[fused_ids].astype(np.float32)),
            "view_count_valid": stats_from_array(view_count[valid_ids].astype(np.float32)),
            "confidence_valid": stats_from_array(confidence[valid_ids]),
            "disagreement_fused": stats_from_array(disagreement[fused_ids]),
            "disagreement_valid": stats_from_array(disagreement[valid_ids]),
            "rgb_delta_valid": stats_from_array(rgb_delta[valid_ids]),
            "opacity_valid": stats_from_array(opacity[valid_ids]),
            "weight_sum_valid": stats_from_array(weight_sum[valid_ids].astype(np.float32)),
        },
    }
    summary_path = output_dir / "direct_gs_fusion_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved direct GS fusion payload to: {payload_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
