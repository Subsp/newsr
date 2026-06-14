import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch
from scipy.spatial import cKDTree
import trimesh

from arguments import (
    MeshingParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    SplattingSettings,
    get_combined_args,
)
from inject_surface_extension_probes import compute_gs_structure_scores
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state


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


def default_output_paths(
    dataset,
    checkpoint_iteration: int,
    output_checkpoint: Optional[str],
    output_summary: Optional[str],
    output_preview_dir: Optional[str],
):
    root = Path(dataset.model_path)
    checkpoint_path = Path(output_checkpoint) if output_checkpoint else root / f"chkpnt{checkpoint_iteration}_truncated.pth"
    summary_path = Path(output_summary) if output_summary else checkpoint_path.with_suffix(".summary.json")
    preview_dir = Path(output_preview_dir) if output_preview_dir else checkpoint_path.with_suffix("")
    return checkpoint_path, summary_path, preview_dir


def parse_source_tag_list(value: Optional[str]) -> Optional[Set[int]]:
    if value is None or str(value).strip() == "":
        return None
    tag_map = {
        "original": int(GaussianSourceTag.ORIGINAL),
        "prior": int(GaussianSourceTag.PRIOR_INJECTED),
        "probe": int(GaussianSourceTag.EXTENSION_PROBE),
        "added": int(GaussianSourceTag.PRIOR_INJECTED),
        "extension_probe": int(GaussianSourceTag.EXTENSION_PROBE),
        "prior_injected": int(GaussianSourceTag.PRIOR_INJECTED),
    }
    keep_tags: Set[int] = set()
    for chunk in str(value).split(","):
        key = chunk.strip().lower()
        if not key:
            continue
        if key not in tag_map:
            raise ValueError(f"Unknown source tag '{chunk}'. Expected one of: {sorted(tag_map.keys())}")
        keep_tags.add(tag_map[key])
    if not keep_tags:
        raise ValueError("keep_source_tags resolved to an empty set")
    return keep_tags


def stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
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


def counts_by_source_tag(source_tag: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, int]:
    if mask is not None:
        source_tag = source_tag[mask]
    mapping = {
        "original": int(GaussianSourceTag.ORIGINAL),
        "prior": int(GaussianSourceTag.PRIOR_INJECTED),
        "probe": int(GaussianSourceTag.EXTENSION_PROBE),
    }
    return {name: int(np.sum(source_tag == value)) for name, value in mapping.items()}


def compute_camera_distance_weight(
    points_xyz: np.ndarray,
    camera_centers: np.ndarray,
    ref_quantile: float,
    power: float,
    min_weight: float,
) -> Dict[str, np.ndarray | float]:
    if points_xyz.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "nearest_distance": empty,
            "distance_weight": empty,
            "distance_ref": 1.0,
        }
    if camera_centers.shape[0] == 0:
        ones = np.ones((points_xyz.shape[0],), dtype=np.float32)
        return {
            "nearest_distance": ones,
            "distance_weight": ones,
            "distance_ref": 1.0,
        }

    tree = cKDTree(camera_centers.astype(np.float32, copy=False))
    nearest_distance, _ = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float32)

    q = float(np.clip(ref_quantile, 0.0, 1.0))
    distance_ref = max(float(np.quantile(nearest_distance, q)), 1e-6)
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    power = max(float(power), 0.0)

    distance_weight = np.power(
        np.clip(distance_ref / np.clip(nearest_distance, 1e-6, None), 0.0, 1.0),
        power,
    ).astype(np.float32, copy=False)
    if min_weight > 0.0:
        distance_weight = np.clip(distance_weight, min_weight, 1.0).astype(np.float32, copy=False)

    return {
        "nearest_distance": nearest_distance,
        "distance_weight": distance_weight,
        "distance_ref": distance_ref,
    }


def compute_mesh_surface_relation(
    points_xyz: np.ndarray,
    gaussian_scaling: np.ndarray,
    mesh_path: str,
    sample_count: int,
    band_scale_mult: float,
) -> Dict[str, np.ndarray | float]:
    if points_xyz.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "surface_distance": empty,
            "surface_normal_offset": empty,
            "surface_tangent_offset": empty,
            "surface_score": empty,
            "surface_band": empty,
            "sample_count": 0,
        }

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "dump"):
            dump = mesh.dump(concatenate=True)
            if isinstance(dump, trimesh.Trimesh):
                mesh = dump
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load a triangle mesh from {mesh_path}")

    sample_count = max(int(sample_count), 1)
    surface_points, face_ids = trimesh.sample.sample_surface(mesh, sample_count)
    surface_points = surface_points.astype(np.float32, copy=False)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    face_normals = mesh.face_normals[face_ids].astype(np.float32, copy=False)

    tree = cKDTree(surface_points)
    nearest_distance, nearest_ids = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float32)
    nearest_ids = np.asarray(nearest_ids, dtype=np.int64)

    nearest_points = surface_points[nearest_ids]
    nearest_normals = face_normals[nearest_ids]
    delta = points_xyz.astype(np.float32, copy=False) - nearest_points
    normal_offset = np.abs(np.sum(delta * nearest_normals, axis=1)).astype(np.float32, copy=False)
    tangent_sq = np.clip(np.square(nearest_distance) - np.square(normal_offset), 0.0, None)
    tangent_offset = np.sqrt(tangent_sq).astype(np.float32, copy=False)

    mean_scale = np.mean(gaussian_scaling, axis=1).astype(np.float32, copy=False)
    surface_band = np.clip(mean_scale * float(band_scale_mult), 1e-6, None).astype(np.float32, copy=False)
    surface_score = np.exp(-normal_offset / surface_band).astype(np.float32, copy=False)

    return {
        "surface_distance": nearest_distance,
        "surface_normal_offset": normal_offset,
        "surface_tangent_offset": tangent_offset,
        "surface_score": surface_score,
        "surface_band": surface_band,
        "sample_count": int(sample_count),
    }


def main():
    parser = ArgumentParser(description="Scene-wide gaussian truncation experiment using whole-scene confidence.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--keep_source_tags", type=str, default=None, help="Optional comma-separated subset to score/prune: original,prior,probe")
    parser.add_argument("--neighbor_k", type=int, default=24)
    parser.add_argument("--radius_scale_mult", type=float, default=12.0)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--opacity_power", type=float, default=1.0)
    parser.add_argument("--structure_power", type=float, default=1.0)
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--mesh_surface_sample_count", type=int, default=300000)
    parser.add_argument("--mesh_surface_band_scale_mult", type=float, default=2.0)
    parser.add_argument("--mesh_surface_power", type=float, default=1.0)
    parser.add_argument("--focus_near_field", action="store_true")
    parser.add_argument("--near_field_distance_quantile", type=float, default=0.6)
    parser.add_argument("--enable_camera_distance_weight", action="store_true")
    parser.add_argument("--camera_distance_ref_quantile", type=float, default=0.5)
    parser.add_argument("--camera_distance_power", type=float, default=1.0)
    parser.add_argument("--camera_distance_min_weight", type=float, default=0.1)
    parser.add_argument("--min_opacity_norm", type=float, default=0.0)
    parser.add_argument("--min_structure_score", type=float, default=0.0)
    parser.add_argument("--prune_score_quantile", type=float, default=None, help="Prune scored gaussians below this confidence quantile.")
    parser.add_argument("--prune_score_threshold", type=float, default=None, help="Prune scored gaussians below this absolute confidence threshold.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--output_summary", type=str, default=None)
    parser.add_argument("--output_preview_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if args.prune_score_quantile is None and args.prune_score_threshold is None:
        raise ValueError("Specify either --prune_score_quantile or --prune_score_threshold for truncation.")
    if args.prune_score_quantile is not None and args.prune_score_threshold is not None:
        raise ValueError("Use only one of --prune_score_quantile or --prune_score_threshold.")

    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for scene truncation.")
        args.data_device = "cpu"

    safe_state(args.quiet)
    keep_tags = parse_source_tag_list(args.keep_source_tags)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)
    splatting.get_settings(args)

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
    opacity = gaussians.get_opacity_with_3D_filter.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    scaling = gaussians.get_scaling_with_3D_filter.detach().cpu().numpy().astype(np.float32, copy=False)
    source_tag = gaussians._source_tag.detach().cpu().numpy().astype(np.int32, copy=False)

    opacity_ref = max(float(np.percentile(opacity, 95)), 1e-6)
    opacity_norm = np.clip(opacity / opacity_ref, 0.0, 1.0).astype(np.float32, copy=False)
    scale_ref = max(float(np.median(np.mean(scaling, axis=1))), 1e-6)
    radius = max(float(args.radius_scale_mult) * scale_ref, 1e-6)
    camera_centers = np.stack(
        [cam.camera_center.detach().cpu().numpy() for cam in train_cameras],
        axis=0,
    ).astype(np.float32, copy=False)
    structure_score, neighbor_count, structure_components = compute_gs_structure_scores(
        points_xyz=xyz,
        gaussian_xyz=xyz,
        gaussian_opacity=opacity,
        gaussian_scaling=scaling,
        neighbor_k=int(args.neighbor_k),
        radius=radius,
        scale_ref=scale_ref,
        batch_size=int(args.batch_size),
    )

    distance_payload = compute_camera_distance_weight(
        points_xyz=xyz,
        camera_centers=camera_centers,
        ref_quantile=float(args.camera_distance_ref_quantile),
        power=float(args.camera_distance_power),
        min_weight=float(args.camera_distance_min_weight),
    )
    distance_weight = distance_payload["distance_weight"]
    if not args.enable_camera_distance_weight:
        distance_weight = np.ones_like(distance_weight, dtype=np.float32)

    mesh_surface_payload = None
    mesh_surface_score = np.ones((xyz.shape[0],), dtype=np.float32)
    if args.mesh_path:
        mesh_surface_payload = compute_mesh_surface_relation(
            points_xyz=xyz,
            gaussian_scaling=scaling,
            mesh_path=args.mesh_path,
            sample_count=int(args.mesh_surface_sample_count),
            band_scale_mult=float(args.mesh_surface_band_scale_mult),
        )
        mesh_surface_score = np.power(
            np.clip(mesh_surface_payload["surface_score"], 0.0, 1.0),
            float(args.mesh_surface_power),
        ).astype(np.float32, copy=False)

    confidence_score = (
        np.power(np.clip(opacity_norm, 0.0, 1.0), float(args.opacity_power))
        * np.power(np.clip(structure_score, 0.0, 1.0), float(args.structure_power))
        * mesh_surface_score
        * distance_weight
    ).astype(np.float32, copy=False)

    eligible_mask = np.ones((xyz.shape[0],), dtype=bool)
    if keep_tags is not None:
        eligible_mask &= np.isin(source_tag, np.asarray(sorted(keep_tags), dtype=np.int32))
    if args.focus_near_field:
        near_q = float(np.clip(args.near_field_distance_quantile, 0.0, 1.0))
        near_field_threshold = float(np.quantile(distance_payload["nearest_distance"], near_q))
        eligible_mask &= distance_payload["nearest_distance"] <= near_field_threshold
    else:
        near_field_threshold = None
    eligible_mask &= opacity_norm >= float(args.min_opacity_norm)
    eligible_mask &= structure_score >= float(args.min_structure_score)

    eligible_scores = confidence_score[eligible_mask]
    if eligible_scores.size == 0:
        raise RuntimeError("No gaussians remained eligible for truncation after filtering.")

    if args.prune_score_threshold is not None:
        threshold = float(args.prune_score_threshold)
    else:
        q = float(args.prune_score_quantile)
        if not (0.0 <= q <= 1.0):
            raise ValueError("--prune_score_quantile must be within [0, 1]")
        threshold = float(np.quantile(eligible_scores, q))

    prune_mask = eligible_mask & (confidence_score < threshold)
    total_before = int(xyz.shape[0])
    prune_count = int(np.sum(prune_mask))
    retained_count = total_before - prune_count

    output_checkpoint, output_summary, output_preview_dir = default_output_paths(
        dataset=dataset,
        checkpoint_iteration=checkpoint_iteration,
        output_checkpoint=args.output_checkpoint,
        output_summary=args.output_summary,
        output_preview_dir=args.output_preview_dir,
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_preview_dir.mkdir(parents=True, exist_ok=True)

    score_payload = {
        "confidence_score": torch.from_numpy(confidence_score.copy()),
        "opacity_norm": torch.from_numpy(opacity_norm.copy()),
        "opacity_filtered": torch.from_numpy(opacity.copy()),
        "structure_score": torch.from_numpy(structure_score.copy()),
        "neighbor_count": torch.from_numpy(neighbor_count.copy()),
        "source_tag": torch.from_numpy(source_tag.copy()),
        "nearest_camera_distance": torch.from_numpy(distance_payload["nearest_distance"].copy()),
        "camera_distance_weight": torch.from_numpy(distance_weight.copy()),
        "prune_mask": torch.from_numpy(prune_mask.copy()),
    }
    if mesh_surface_payload is not None:
        score_payload["surface_distance"] = torch.from_numpy(mesh_surface_payload["surface_distance"].copy())
        score_payload["surface_normal_offset"] = torch.from_numpy(mesh_surface_payload["surface_normal_offset"].copy())
        score_payload["surface_tangent_offset"] = torch.from_numpy(mesh_surface_payload["surface_tangent_offset"].copy())
        score_payload["surface_score"] = torch.from_numpy(mesh_surface_payload["surface_score"].copy())
        score_payload["surface_band"] = torch.from_numpy(mesh_surface_payload["surface_band"].copy())
    score_payload.update({
        "density_raw": torch.from_numpy(structure_components["density_raw"].copy()),
        "density_norm": torch.from_numpy(structure_components["density_norm"].copy()),
        "small_scale": torch.from_numpy(structure_components["small_scale"].copy()),
        "anisotropy": torch.from_numpy(structure_components["anisotropy"].copy()),
        "nonplanar_norm": torch.from_numpy(structure_components["nonplanar_norm"].copy()),
    })
    score_path = output_preview_dir / "scene_truncation_scores.pt"
    torch.save(score_payload, str(score_path))

    if not args.dry_run:
        standard_point_cloud_dir = output_checkpoint.parent / "point_cloud" / f"iteration_{checkpoint_iteration}"
        standard_point_cloud_dir.mkdir(parents=True, exist_ok=True)
        gaussians.prune_points(torch.from_numpy(prune_mask).to(device=gaussians.get_xyz.device, dtype=torch.bool))
        torch.save((gaussians.capture(), checkpoint_iteration, appearance_state), str(output_checkpoint))
        gaussians.save_ply(str(standard_point_cloud_dir / "point_cloud.ply"))
        gaussians.save_tracking_metadata(str(standard_point_cloud_dir / "gaussian_tags.pt"))
        gaussians.save_ply(str(output_preview_dir / "point_cloud.ply"))
        gaussians.save_tracking_metadata(str(output_preview_dir / "gaussian_tags.pt"))
        input_cfg_args = Path(dataset.model_path) / "cfg_args"
        if input_cfg_args.exists():
            shutil.copy2(input_cfg_args, output_checkpoint.parent / "cfg_args")
        input_config_json = Path(dataset.model_path) / "config.json"
        if input_config_json.exists():
            shutil.copy2(input_config_json, output_checkpoint.parent / "config.json")

    summary = {
        "mode": "scene_gaussian_truncation",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "input_checkpoint": str(Path(checkpoint_path).resolve()),
        "output_checkpoint": None if args.dry_run else str(output_checkpoint.resolve()),
        "checkpoint_iteration": int(checkpoint_iteration),
        "dry_run": bool(args.dry_run),
        "parameters": {
            "keep_source_tags": None if keep_tags is None else sorted(int(x) for x in keep_tags),
            "neighbor_k": int(args.neighbor_k),
            "radius_scale_mult": float(args.radius_scale_mult),
            "batch_size": int(args.batch_size),
            "opacity_power": float(args.opacity_power),
            "structure_power": float(args.structure_power),
            "mesh_path": None if args.mesh_path is None else str(Path(args.mesh_path).resolve()),
            "mesh_surface_sample_count": int(args.mesh_surface_sample_count),
            "mesh_surface_band_scale_mult": float(args.mesh_surface_band_scale_mult),
            "mesh_surface_power": float(args.mesh_surface_power),
            "focus_near_field": bool(args.focus_near_field),
            "near_field_distance_quantile": float(args.near_field_distance_quantile),
            "near_field_distance_threshold": None if near_field_threshold is None else float(near_field_threshold),
            "enable_camera_distance_weight": bool(args.enable_camera_distance_weight),
            "camera_distance_ref_quantile": float(args.camera_distance_ref_quantile),
            "camera_distance_power": float(args.camera_distance_power),
            "camera_distance_min_weight": float(args.camera_distance_min_weight),
            "camera_distance_ref": float(distance_payload["distance_ref"]),
            "min_opacity_norm": float(args.min_opacity_norm),
            "min_structure_score": float(args.min_structure_score),
            "prune_score_quantile": None if args.prune_score_quantile is None else float(args.prune_score_quantile),
            "prune_score_threshold": None if args.prune_score_threshold is None else float(args.prune_score_threshold),
            "resolved_threshold": float(threshold),
            "scale_ref": float(scale_ref),
            "radius": float(radius),
        },
        "counts": {
            "total_gaussians_before": total_before,
            "eligible_gaussians": int(np.sum(eligible_mask)),
            "pruned_gaussians": prune_count,
            "retained_gaussians": retained_count,
            "source_before": counts_by_source_tag(source_tag),
            "source_pruned": counts_by_source_tag(source_tag, prune_mask),
            "source_retained": counts_by_source_tag(source_tag, ~prune_mask),
        },
        "score_stats": {
            "confidence_score_all": stats_from_array(confidence_score),
            "confidence_score_eligible": stats_from_array(eligible_scores),
            "confidence_score_pruned": stats_from_array(confidence_score[prune_mask]),
            "opacity_norm_all": stats_from_array(opacity_norm),
            "structure_score_all": stats_from_array(structure_score),
            "nearest_camera_distance_all": stats_from_array(distance_payload["nearest_distance"]),
            "camera_distance_weight_all": stats_from_array(distance_weight),
            "neighbor_count_all": stats_from_array(neighbor_count.astype(np.float32)),
            "density_norm_all": stats_from_array(structure_components["density_norm"]),
            "small_scale_all": stats_from_array(structure_components["small_scale"]),
            "anisotropy_all": stats_from_array(structure_components["anisotropy"]),
            "nonplanar_norm_all": stats_from_array(structure_components["nonplanar_norm"]),
        },
        "paths": {
            "score_payload": str(score_path.resolve()),
        },
    }
    if mesh_surface_payload is not None:
        summary["score_stats"]["surface_distance_all"] = stats_from_array(mesh_surface_payload["surface_distance"])
        summary["score_stats"]["surface_normal_offset_all"] = stats_from_array(mesh_surface_payload["surface_normal_offset"])
        summary["score_stats"]["surface_tangent_offset_all"] = stats_from_array(mesh_surface_payload["surface_tangent_offset"])
        summary["score_stats"]["surface_score_all"] = stats_from_array(mesh_surface_payload["surface_score"])
        summary["score_stats"]["surface_score_pruned"] = stats_from_array(mesh_surface_payload["surface_score"][prune_mask])
    if not args.dry_run:
        summary["paths"]["standard_point_cloud"] = str((output_checkpoint.parent / "point_cloud" / f"iteration_{checkpoint_iteration}" / "point_cloud.ply").resolve())
        summary["paths"]["standard_tags"] = str((output_checkpoint.parent / "point_cloud" / f"iteration_{checkpoint_iteration}" / "gaussian_tags.pt").resolve())
        summary["paths"]["preview_point_cloud"] = str((output_preview_dir / "point_cloud.ply").resolve())
        summary["paths"]["preview_tags"] = str((output_preview_dir / "gaussian_tags.pt").resolve())
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run complete; no checkpoint was written.")
    else:
        print(f"Truncated checkpoint saved to: {output_checkpoint}")
    print(f"Summary saved to: {output_summary}")


if __name__ == "__main__":
    main()
