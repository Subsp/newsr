import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

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
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state
from utils.prior_fusion import project_points_camera


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


def load_triangle_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load a triangle mesh from {mesh_path}")


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


def parse_source_tag_list(value: Optional[str]) -> Optional[Set[int]]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return None
    tag_map = {
        "original": int(GaussianSourceTag.ORIGINAL),
        "prior": int(GaussianSourceTag.PRIOR_INJECTED),
        "prior_injected": int(GaussianSourceTag.PRIOR_INJECTED),
        "probe": int(GaussianSourceTag.EXTENSION_PROBE),
        "extension_probe": int(GaussianSourceTag.EXTENSION_PROBE),
        "added": int(GaussianSourceTag.PRIOR_INJECTED),
    }
    selected: Set[int] = set()
    for chunk in str(value).split(","):
        key = chunk.strip().lower()
        if not key:
            continue
        if key not in tag_map:
            raise ValueError(f"Unknown source tag '{chunk}'. Expected one of: {sorted(tag_map.keys()) + ['all']}")
        selected.add(tag_map[key])
    return selected or None


def counts_by_source_tag(source_tag: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, int]:
    values = source_tag if mask is None else source_tag[mask]
    mapping = {
        "original": int(GaussianSourceTag.ORIGINAL),
        "prior": int(GaussianSourceTag.PRIOR_INJECTED),
        "probe": int(GaussianSourceTag.EXTENSION_PROBE),
    }
    return {name: int(np.sum(values == value)) for name, value in mapping.items()}


def source_tag_mask(source_tag: np.ndarray, allowed_tags: Optional[Set[int]]) -> np.ndarray:
    if allowed_tags is None:
        return np.ones((source_tag.shape[0],), dtype=bool)
    allowed = np.asarray(sorted(allowed_tags), dtype=np.int32)
    return np.isin(source_tag, allowed)


def query_mesh_surface_by_sampling(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    sample_count: int,
) -> Dict[str, np.ndarray]:
    surface_points, face_ids = trimesh.sample.sample_surface(mesh_obj, max(int(sample_count), 1))
    surface_points = np.asarray(surface_points, dtype=np.float32)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    surface_normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)

    tree = cKDTree(surface_points)
    distance, nearest_ids = tree.query(points_xyz.astype(np.float32, copy=False), k=1)
    nearest_ids = np.asarray(nearest_ids, dtype=np.int64)
    nearest_face_ids = face_ids[nearest_ids]
    return {
        "surface_distance": np.asarray(distance, dtype=np.float32),
        "nearest_surface_point": surface_points[nearest_ids].astype(np.float32, copy=False),
        "nearest_surface_normal": surface_normals[nearest_ids].astype(np.float32, copy=False),
        "nearest_face_id": nearest_face_ids.astype(np.int64, copy=False),
        "surface_query_mode_used": "sample",
    }


def query_mesh_surface_exact_open3d(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    import open3d as o3d

    vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
    faces = np.asarray(mesh_obj.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("Mesh has no vertices/faces for exact closest-point query.")

    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    legacy_mesh.triangles = o3d.utility.Vector3iVector(faces)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    closest_points = np.empty_like(points_xyz, dtype=np.float32)
    distances = np.empty((points_xyz.shape[0],), dtype=np.float32)
    face_ids = np.empty((points_xyz.shape[0],), dtype=np.int64)

    chunk = max(int(chunk_size), 1)
    for begin in range(0, points_xyz.shape[0], chunk):
        end = min(begin + chunk, points_xyz.shape[0])
        points_chunk = np.asarray(points_xyz[begin:end], dtype=np.float32)
        query_points = o3d.core.Tensor(points_chunk, dtype=o3d.core.Dtype.Float32)
        result = scene.compute_closest_points(query_points)

        closest = result["points"].numpy().astype(np.float32, copy=False)
        tri_ids = result["primitive_ids"].numpy().astype(np.int64, copy=False)
        tri_ids = np.clip(tri_ids, 0, len(mesh_obj.faces) - 1)

        closest_points[begin:end] = closest
        distances[begin:end] = np.linalg.norm(closest - points_chunk, axis=1).astype(np.float32, copy=False)
        face_ids[begin:end] = tri_ids

    normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)
    return {
        "surface_distance": distances,
        "nearest_surface_point": closest_points,
        "nearest_surface_normal": normals,
        "nearest_face_id": face_ids,
        "surface_query_mode_used": "exact_open3d",
    }


def query_mesh_surface_exact_trimesh(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    query = trimesh.proximity.ProximityQuery(mesh_obj)
    closest_points = np.empty_like(points_xyz, dtype=np.float32)
    distances = np.empty((points_xyz.shape[0],), dtype=np.float32)
    face_ids = np.empty((points_xyz.shape[0],), dtype=np.int64)

    chunk = max(int(chunk_size), 1)
    for begin in range(0, points_xyz.shape[0], chunk):
        end = min(begin + chunk, points_xyz.shape[0])
        closest, distance, tri_ids = query.on_surface(points_xyz[begin:end])
        closest_points[begin:end] = np.asarray(closest, dtype=np.float32)
        distances[begin:end] = np.asarray(distance, dtype=np.float32)
        face_ids[begin:end] = np.asarray(tri_ids, dtype=np.int64)

    normals = np.asarray(mesh_obj.face_normals[face_ids], dtype=np.float32)
    return {
        "surface_distance": distances,
        "nearest_surface_point": closest_points,
        "nearest_surface_normal": normals,
        "nearest_face_id": face_ids,
        "surface_query_mode_used": "exact_trimesh",
    }


def query_mesh_surface_exact(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    open3d_error = None
    try:
        return query_mesh_surface_exact_open3d(mesh_obj, points_xyz, chunk_size=chunk_size)
    except Exception as exc:
        open3d_error = exc

    try:
        return query_mesh_surface_exact_trimesh(mesh_obj, points_xyz, chunk_size=chunk_size)
    except Exception as exc:
        if open3d_error is not None:
            raise RuntimeError(
                "Both exact closest-point backends failed: "
                f"open3d={open3d_error!r}; trimesh={exc!r}"
            ) from exc
        raise


def query_mesh_surface(
    mesh_obj: trimesh.Trimesh,
    points_xyz: np.ndarray,
    mode: str,
    sample_count: int,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    if mode in {"auto", "exact"}:
        try:
            return query_mesh_surface_exact(mesh_obj, points_xyz, chunk_size=chunk_size)
        except Exception as exc:
            if mode == "exact":
                raise
            print(f"[mesh-outside] exact mesh distance failed ({exc}); falling back to sampled surface distance.")
    return query_mesh_surface_by_sampling(mesh_obj, points_xyz, sample_count=sample_count)


def compute_view_visibility_stats(
    points_xyz: np.ndarray,
    cameras,
    candidate_ids: np.ndarray,
    camera_stride: int,
    depth_min: float,
    margin_px: int,
    chunk_size: int,
) -> Dict[str, np.ndarray | int]:
    visible_view_count = np.zeros((points_xyz.shape[0],), dtype=np.int32)
    min_visible_depth = np.full((points_xyz.shape[0],), np.inf, dtype=np.float32)
    mean_visible_depth = np.full((points_xyz.shape[0],), np.inf, dtype=np.float32)
    depth_sum = np.zeros((points_xyz.shape[0],), dtype=np.float64)

    if candidate_ids.size == 0:
        return {
            "visible_view_count": visible_view_count,
            "min_visible_depth": min_visible_depth,
            "mean_visible_depth": mean_visible_depth,
            "camera_count_used": 0,
        }

    stride = max(int(camera_stride), 1)
    selected_cameras = cameras[::stride]
    for cam_idx, cam in enumerate(selected_cameras):
        if cam_idx % 20 == 0:
            print(f"[mesh-outside] projecting candidate GS in camera {cam_idx + 1}/{len(selected_cameras)}")
        for begin in range(0, candidate_ids.shape[0], max(int(chunk_size), 1)):
            end = min(begin + max(int(chunk_size), 1), candidate_ids.shape[0])
            ids = candidate_ids[begin:end]
            projected, valid = project_points_camera(cam, points_xyz[ids], depth_min=depth_min, margin=margin_px)
            if not np.any(valid):
                continue
            visible_ids = ids[valid]
            depths = projected[valid, 2].astype(np.float32, copy=False)
            visible_view_count[visible_ids] += 1
            depth_sum[visible_ids] += depths.astype(np.float64)
            min_visible_depth[visible_ids] = np.minimum(min_visible_depth[visible_ids], depths)

    seen = visible_view_count > 0
    mean_visible_depth[seen] = (depth_sum[seen] / visible_view_count[seen].astype(np.float64)).astype(np.float32)
    return {
        "visible_view_count": visible_view_count,
        "min_visible_depth": min_visible_depth,
        "mean_visible_depth": mean_visible_depth,
        "camera_count_used": len(selected_cameras),
    }


def save_point_cloud(points_xyz: np.ndarray, path: Path, color=(255, 40, 40), max_points: int = 0):
    if points_xyz.shape[0] == 0:
        return
    if max_points > 0 and points_xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        ids = rng.choice(points_xyz.shape[0], size=int(max_points), replace=False)
        points_xyz = points_xyz[ids]
    colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (points_xyz.shape[0], 1))
    trimesh.points.PointCloud(points_xyz, colors=colors).export(path)


def main():
    parser = ArgumentParser(description="Select mesh-outside SOFGS candidates with hard surface distance and near-view filters.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    SplattingSettings(parser, render=False)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--surface_distance_threshold", type=float, default=0.03)
    parser.add_argument("--surface_query_mode", choices=["auto", "exact", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=500000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--screen_margin_px", type=int, default=0)
    parser.add_argument("--projection_chunk_size", type=int, default=131072)
    parser.add_argument("--min_visible_views", type=int, default=2)
    parser.add_argument("--max_nearest_visible_depth", type=float, default=3.0)
    parser.add_argument("--source_tags", type=str, default="all")
    parser.add_argument("--min_candidate_opacity", type=float, default=0.0)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for mesh-outside selection.")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_obj = load_triangle_mesh(args.mesh_path)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    source_tag = gaussians._source_tag.detach().cpu().numpy().astype(np.int32, copy=False)
    allowed_tags = parse_source_tag_list(args.source_tags)

    print(f"[mesh-outside] querying surface distance for {xyz.shape[0]} gaussians")
    surface_payload = query_mesh_surface(
        mesh_obj=mesh_obj,
        points_xyz=xyz,
        mode=args.surface_query_mode,
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )
    surface_distance = surface_payload["surface_distance"]
    surface_outside_mask = surface_distance > float(args.surface_distance_threshold)
    eligible_mask = source_tag_mask(source_tag, allowed_tags) & (opacity >= float(args.min_candidate_opacity))
    distance_candidate_mask = surface_outside_mask & eligible_mask
    distance_candidate_ids = np.flatnonzero(distance_candidate_mask)

    print(
        "[mesh-outside] distance candidates: "
        f"{distance_candidate_ids.shape[0]}/{xyz.shape[0]} "
        f"(threshold={float(args.surface_distance_threshold):.6f})"
    )
    visibility_payload = compute_view_visibility_stats(
        points_xyz=xyz,
        cameras=train_cameras,
        candidate_ids=distance_candidate_ids,
        camera_stride=int(args.camera_stride),
        depth_min=float(args.depth_min),
        margin_px=int(args.screen_margin_px),
        chunk_size=int(args.projection_chunk_size),
    )

    visible_view_count = visibility_payload["visible_view_count"]
    min_visible_depth = visibility_payload["min_visible_depth"]
    mean_visible_depth = visibility_payload["mean_visible_depth"]
    enough_views_mask = visible_view_count >= int(args.min_visible_views)
    if float(args.max_nearest_visible_depth) > 0.0:
        near_view_mask = min_visible_depth <= float(args.max_nearest_visible_depth)
    else:
        near_view_mask = np.ones_like(enough_views_mask, dtype=bool)

    candidate_mask = distance_candidate_mask & enough_views_mask & near_view_mask
    visibility_rejected_mask = distance_candidate_mask & (~enough_views_mask)
    far_rejected_mask = distance_candidate_mask & enough_views_mask & (~near_view_mask)
    selected_ids = np.flatnonzero(candidate_mask)

    payload = {
        "candidate_mask": torch.from_numpy(candidate_mask.copy()),
        "surface_outside_mask": torch.from_numpy(surface_outside_mask.copy()),
        "distance_candidate_mask": torch.from_numpy(distance_candidate_mask.copy()),
        "visibility_rejected_mask": torch.from_numpy(visibility_rejected_mask.copy()),
        "far_rejected_mask": torch.from_numpy(far_rejected_mask.copy()),
        "selected_ids": torch.from_numpy(selected_ids.astype(np.int64, copy=False)),
        "surface_distance": torch.from_numpy(surface_distance.copy()),
        "nearest_surface_point": torch.from_numpy(surface_payload["nearest_surface_point"].copy()),
        "nearest_surface_normal": torch.from_numpy(surface_payload["nearest_surface_normal"].copy()),
        "nearest_face_id": torch.from_numpy(surface_payload["nearest_face_id"].copy()),
        "visible_view_count": torch.from_numpy(visible_view_count.copy()),
        "min_visible_depth": torch.from_numpy(min_visible_depth.copy()),
        "mean_visible_depth": torch.from_numpy(mean_visible_depth.copy()),
        "opacity": torch.from_numpy(opacity.copy()),
        "source_tag": torch.from_numpy(source_tag.copy()),
    }
    payload_path = output_dir / "mesh_outside_candidates_v0.pt"
    torch.save(payload, str(payload_path))

    selected_ply_path = output_dir / "mesh_outside_candidates_v0.ply"
    distance_ply_path = output_dir / "mesh_outside_distance_candidates_v0.ply"
    far_ply_path = output_dir / "mesh_outside_far_rejected_v0.ply"
    save_point_cloud(xyz[selected_ids], selected_ply_path, color=(255, 40, 40), max_points=int(args.preview_max_points))
    save_point_cloud(xyz[distance_candidate_ids], distance_ply_path, color=(255, 180, 30), max_points=int(args.preview_max_points))
    save_point_cloud(xyz[np.flatnonzero(far_rejected_mask)], far_ply_path, color=(40, 140, 255), max_points=int(args.preview_max_points))

    summary = {
        "mode": "mesh_outside_gaussian_selection_v0",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "input_checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_iteration": int(checkpoint_iteration),
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "payload_path": str(payload_path.resolve()),
        "surface_query_mode_requested": args.surface_query_mode,
        "surface_query_mode_used": surface_payload["surface_query_mode_used"],
        "camera_count_total": int(len(train_cameras)),
        "camera_count_used": int(visibility_payload["camera_count_used"]),
        "parameters": {
            "surface_distance_threshold": float(args.surface_distance_threshold),
            "mesh_surface_sample_count": int(args.mesh_surface_sample_count),
            "surface_query_chunk_size": int(args.surface_query_chunk_size),
            "camera_stride": int(args.camera_stride),
            "depth_min": float(args.depth_min),
            "screen_margin_px": int(args.screen_margin_px),
            "min_visible_views": int(args.min_visible_views),
            "max_nearest_visible_depth": float(args.max_nearest_visible_depth),
            "source_tags": args.source_tags,
            "min_candidate_opacity": float(args.min_candidate_opacity),
        },
        "counts": {
            "total_gaussians": int(xyz.shape[0]),
            "eligible_gaussians": int(np.sum(eligible_mask)),
            "surface_outside_count": int(np.sum(surface_outside_mask)),
            "distance_candidate_count": int(distance_candidate_ids.shape[0]),
            "visibility_rejected_count": int(np.sum(visibility_rejected_mask)),
            "far_rejected_count": int(np.sum(far_rejected_mask)),
            "selected_candidate_count": int(selected_ids.shape[0]),
            "selected_candidate_ratio": float(selected_ids.shape[0] / max(xyz.shape[0], 1)),
            "source_all": counts_by_source_tag(source_tag),
            "source_selected": counts_by_source_tag(source_tag, candidate_mask),
            "source_far_rejected": counts_by_source_tag(source_tag, far_rejected_mask),
        },
        "stats": {
            "surface_distance_all": stats_from_array(surface_distance),
            "surface_distance_distance_candidates": stats_from_array(surface_distance[distance_candidate_mask]),
            "surface_distance_selected": stats_from_array(surface_distance[candidate_mask]),
            "visible_view_count_distance_candidates": stats_from_array(visible_view_count[distance_candidate_mask].astype(np.float32)),
            "visible_view_count_selected": stats_from_array(visible_view_count[candidate_mask].astype(np.float32)),
            "min_visible_depth_distance_candidates": stats_from_array(min_visible_depth[distance_candidate_mask]),
            "min_visible_depth_selected": stats_from_array(min_visible_depth[candidate_mask]),
            "mean_visible_depth_selected": stats_from_array(mean_visible_depth[candidate_mask]),
            "opacity_selected": stats_from_array(opacity[candidate_mask]),
        },
        "previews": {
            "selected_candidates_ply": str(selected_ply_path.resolve()),
            "distance_candidates_ply": str(distance_ply_path.resolve()),
            "far_rejected_ply": str(far_ply_path.resolve()),
        },
    }
    summary_path = output_dir / "mesh_outside_candidates_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved mesh-outside candidate payload to: {payload_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
