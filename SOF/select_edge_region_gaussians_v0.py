import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
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
from utils.prior_fusion import project_points_camera
from utils.prior_injection import load_mask, normalize_image_name


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


def load_optional_candidate_mask(path: Optional[str], total_gaussians: int, key: str) -> Dict[str, np.ndarray]:
    if not path:
        mask = np.ones((total_gaussians,), dtype=bool)
        return {"candidate_mask": mask, "candidate_ids": np.arange(total_gaussians, dtype=np.int64)}
    payload = torch.load(path, map_location="cpu")
    if key in payload:
        mask = payload[key].detach().cpu().numpy().astype(bool, copy=False)
        if mask.shape[0] != total_gaussians:
            raise ValueError(f"candidate mask length mismatch: {mask.shape[0]} vs {total_gaussians}")
        ids = np.flatnonzero(mask).astype(np.int64, copy=False)
        return {"candidate_mask": mask, "candidate_ids": ids}
    if "selected_ids" in payload:
        ids = payload["selected_ids"].detach().cpu().numpy().astype(np.int64, copy=False)
        mask = np.zeros((total_gaussians,), dtype=bool)
        mask[ids] = True
        return {"candidate_mask": mask, "candidate_ids": ids}
    raise KeyError(f"Candidate payload must contain '{key}' or 'selected_ids': {path}")


def load_edge_mask(mask_dir: Path, image_name: str, suffix: str):
    stem = normalize_image_name(image_name)
    candidates = [
        mask_dir / f"{image_name}{suffix}",
        mask_dir / f"{stem}{suffix}",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        return None, path
    mask = load_mask(path).cpu().numpy() > 0.5
    return mask, path


def list_mask_file_samples(mask_dir: Path, suffix: str, max_samples: int = 16):
    if not mask_dir.is_dir():
        return []
    return sorted(path.name for path in mask_dir.glob(f"*{suffix}"))[: int(max_samples)]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(device="cuda")
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(mask_t, kernel_size=kernel, stride=1, padding=radius)
    return (dilated[0, 0].detach().cpu().numpy() > 0.5)


def save_selected_point_cloud(points_xyz: np.ndarray, selected_mask: np.ndarray, view_count: np.ndarray, path: Path, max_points: int):
    selected_ids = np.flatnonzero(selected_mask)
    if selected_ids.size == 0:
        return
    if max_points > 0 and selected_ids.size > int(max_points):
        rng = np.random.default_rng(0)
        selected_ids = rng.choice(selected_ids, size=int(max_points), replace=False)
    counts = view_count[selected_ids].astype(np.float32)
    denom = max(float(np.percentile(counts, 95)), 1.0) if counts.size > 0 else 1.0
    heat = np.clip(counts / denom, 0.0, 1.0)
    colors = np.zeros((selected_ids.size, 3), dtype=np.uint8)
    colors[:, 0] = np.round(255.0 * heat).astype(np.uint8)
    colors[:, 1] = np.round(80.0 * (1.0 - heat)).astype(np.uint8)
    colors[:, 2] = 40
    trimesh.points.PointCloud(points_xyz[selected_ids], colors=colors).export(path)


def main():
    parser = ArgumentParser(description="Select existing SOFGS whose projected footprint touches edge fusion masks.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    SplattingSettings(parser, render=False)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--edge_mask_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default="_inject.png")
    parser.add_argument("--candidate_payload", type=str, default=None)
    parser.add_argument("--candidate_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--view_limit", type=int, default=0)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--screen_margin_px", type=int, default=0)
    parser.add_argument("--min_touch_views", type=int, default=1)
    parser.add_argument("--min_visible_views", type=int, default=1)
    parser.add_argument("--min_touch_ratio", type=float, default=0.0)
    parser.add_argument("--min_candidate_opacity", type=float, default=0.0)
    parser.add_argument("--center_only", action="store_true")
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--min_touch_radius_px", type=float, default=1.0)
    parser.add_argument("--max_touch_radius_px", type=float, default=16.0)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for edge-region GS selection.")
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
    scaling = gaussians.get_scaling_with_3D_filter.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    total_gaussians = int(xyz.shape[0])

    candidate_payload = load_optional_candidate_mask(
        args.candidate_payload,
        total_gaussians=total_gaussians,
        key=args.candidate_mask_key,
    )
    candidate_mask = candidate_payload["candidate_mask"] & (opacity >= float(args.min_candidate_opacity))
    candidate_ids = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    if candidate_ids.size == 0:
        raise RuntimeError("No candidate gaussians available after filtering.")

    edge_touch_count = np.zeros((total_gaussians,), dtype=np.int32)
    visible_view_count = np.zeros((total_gaussians,), dtype=np.int32)
    first_touch_view = np.full((total_gaussians,), "", dtype=object)

    mask_dir = Path(args.edge_mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Edge mask directory does not exist: {mask_dir}")
    selected_cameras = train_cameras[:: max(int(args.camera_stride), 1)]
    if int(args.view_limit) > 0:
        selected_cameras = selected_cameras[: int(args.view_limit)]

    mask_match_count = 0
    missing_mask_views = []
    for view_idx, view in enumerate(selected_cameras):
        if view_idx % 20 == 0:
            print(f"[edge-gs-select] view {view_idx + 1}/{len(selected_cameras)}: {view.image_name}")
        edge_mask, mask_path = load_edge_mask(mask_dir, view.image_name, args.mask_suffix)
        if edge_mask is None:
            if len(missing_mask_views) < 16:
                missing_mask_views.append(view.image_name)
            continue
        mask_match_count += 1

        projected, valid = project_points_camera(
            view,
            xyz[candidate_ids],
            depth_min=float(args.depth_min),
            margin=int(args.screen_margin_px),
        )
        if not np.any(valid):
            continue

        local_valid_ids = np.flatnonzero(valid)
        mask_x = projected[local_valid_ids, 0] * (float(edge_mask.shape[1]) / float(view.image_width))
        mask_y = projected[local_valid_ids, 1] * (float(edge_mask.shape[0]) / float(view.image_height))
        pix_x = np.round(mask_x).astype(np.int64)
        pix_y = np.round(mask_y).astype(np.int64)
        in_bounds = (
            (pix_x >= 0)
            & (pix_x < edge_mask.shape[1])
            & (pix_y >= 0)
            & (pix_y < edge_mask.shape[0])
        )
        if not np.any(in_bounds):
            continue

        local_valid_ids = local_valid_ids[in_bounds]
        pix_x = pix_x[in_bounds]
        pix_y = pix_y[in_bounds]
        global_valid_ids = candidate_ids[local_valid_ids]
        visible_view_count[global_valid_ids] += 1

        if bool(args.center_only):
            hit = edge_mask[pix_y, pix_x]
        else:
            z = projected[local_valid_ids, 2]
            max_scale = np.max(scaling[global_valid_ids], axis=1)
            focal = max(float(view.focal_x), float(view.focal_y))
            radii = np.ceil(
                np.clip(
                    focal * max_scale / np.clip(z, 1e-6, None) * float(args.radius_scale),
                    float(args.min_touch_radius_px),
                    float(args.max_touch_radius_px),
                )
            ).astype(np.int64)
            hit = np.zeros((local_valid_ids.shape[0],), dtype=bool)
            for radius in np.unique(radii).tolist():
                bucket = radii == int(radius)
                if not np.any(bucket):
                    continue
                dilated = dilate_mask(edge_mask, int(radius))
                hit[bucket] = dilated[pix_y[bucket], pix_x[bucket]]

        if not np.any(hit):
            continue
        hit_ids = global_valid_ids[hit]
        edge_touch_count[hit_ids] += 1
        new_touch = first_touch_view[hit_ids] == ""
        if np.any(new_touch):
            first_touch_view[hit_ids[new_touch]] = view.image_name

    if mask_match_count == 0:
        raise RuntimeError(
            "No edge masks matched any train view. "
            f"edge_mask_dir={mask_dir.resolve()}, suffix={args.mask_suffix}, "
            f"file_samples={list_mask_file_samples(mask_dir, args.mask_suffix)}, "
            f"missing_view_samples={missing_mask_views}"
        )

    selected_mask = (
        candidate_mask
        & (edge_touch_count >= int(args.min_touch_views))
        & (visible_view_count >= int(args.min_visible_views))
    )
    touch_ratio = edge_touch_count.astype(np.float32) / np.maximum(visible_view_count.astype(np.float32), 1.0)
    if float(args.min_touch_ratio) > 0.0:
        selected_mask = selected_mask & (touch_ratio >= float(args.min_touch_ratio))
    selected_ids = np.flatnonzero(selected_mask).astype(np.int64, copy=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "edge_region_gaussians_v0.pt"
    torch.save(
        {
            "selected_mask": torch.from_numpy(selected_mask.copy()),
            "candidate_mask": torch.from_numpy(candidate_mask.copy()),
            "selected_ids": torch.from_numpy(selected_ids.copy()),
            "edge_touch_count": torch.from_numpy(edge_touch_count.copy()),
            "visible_view_count": torch.from_numpy(visible_view_count.copy()),
            "touch_ratio": torch.from_numpy(touch_ratio.copy()),
            "opacity": torch.from_numpy(opacity.copy()),
        },
        str(payload_path),
    )

    preview_path = output_dir / "edge_region_gaussians_v0.ply"
    save_selected_point_cloud(
        xyz,
        selected_mask,
        edge_touch_count,
        preview_path,
        max_points=int(args.preview_max_points),
    )

    summary = {
        "mode": "edge_region_gaussian_selection_v0",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "input_checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_iteration": int(checkpoint_iteration),
        "edge_mask_dir": str(mask_dir.resolve()),
        "candidate_payload": None if args.candidate_payload is None else str(Path(args.candidate_payload).resolve()),
        "output_dir": str(output_dir.resolve()),
        "payload_path": str(payload_path.resolve()),
        "preview_path": str(preview_path.resolve()),
        "camera_count_total": int(len(train_cameras)),
        "camera_count_used": int(len(selected_cameras)),
        "mask_match_count": int(mask_match_count),
        "missing_mask_view_samples": missing_mask_views,
        "parameters": {
            "mask_suffix": args.mask_suffix,
            "candidate_mask_key": args.candidate_mask_key,
            "view_limit": int(args.view_limit),
            "camera_stride": int(args.camera_stride),
            "depth_min": float(args.depth_min),
            "screen_margin_px": int(args.screen_margin_px),
            "min_touch_views": int(args.min_touch_views),
            "min_visible_views": int(args.min_visible_views),
            "min_touch_ratio": float(args.min_touch_ratio),
            "min_candidate_opacity": float(args.min_candidate_opacity),
            "center_only": bool(args.center_only),
            "radius_scale": float(args.radius_scale),
            "min_touch_radius_px": float(args.min_touch_radius_px),
            "max_touch_radius_px": float(args.max_touch_radius_px),
        },
        "counts": {
            "total_gaussians": int(total_gaussians),
            "candidate_count": int(candidate_ids.size),
            "selected_count": int(selected_ids.size),
            "selected_ratio_vs_candidates": float(selected_ids.size / max(candidate_ids.size, 1)),
            "selected_ratio_vs_all": float(selected_ids.size / max(total_gaussians, 1)),
        },
        "stats": {
            "edge_touch_count_candidates": stats_from_array(edge_touch_count[candidate_ids].astype(np.float32)),
            "edge_touch_count_selected": stats_from_array(edge_touch_count[selected_ids].astype(np.float32)),
            "visible_view_count_selected": stats_from_array(visible_view_count[selected_ids].astype(np.float32)),
            "touch_ratio_candidates": stats_from_array(touch_ratio[candidate_ids]),
            "touch_ratio_selected": stats_from_array(touch_ratio[selected_ids]),
            "opacity_selected": stats_from_array(opacity[selected_ids]),
        },
    }
    summary_path = output_dir / "edge_region_gaussians_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved edge-region GS payload to: {payload_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
