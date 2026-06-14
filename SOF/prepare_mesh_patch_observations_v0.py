import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from arguments import ModelParams, get_combined_args
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir, normalize_image_name


BOUND_BARYCENTRIC_TEMPLATES: Dict[int, np.ndarray] = {
    1: np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32),
    3: np.asarray(
        [
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ],
        dtype=np.float32,
    ),
    4: np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float32,
    ),
    6: np.asarray(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 6.0, 5.0 / 12.0, 5.0 / 12.0],
            [5.0 / 12.0, 1.0 / 6.0, 5.0 / 12.0],
            [5.0 / 12.0, 5.0 / 12.0, 1.0 / 6.0],
        ],
        dtype=np.float32,
    ),
}


def load_triangle_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load triangle mesh from {mesh_path}")


def camera_center_numpy(cam) -> np.ndarray:
    center = cam.camera_center
    if torch.is_tensor(center):
        center = center.detach().cpu().numpy()
    return np.asarray(center, dtype=np.float32).reshape(3)


def project_points_camera(cam, points_xyz: np.ndarray, depth_min: float, margin: int = 0):
    R = np.asarray(cam.R, dtype=np.float32)
    T = np.asarray(cam.T, dtype=np.float32)
    xyz_cam = points_xyz @ R + T[None, :]
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0] / np.clip(z, 1e-6, None) * float(cam.focal_x) + float(cam.image_width) / 2.0
    y = xyz_cam[:, 1] / np.clip(z, 1e-6, None) * float(cam.focal_y) + float(cam.image_height) / 2.0
    valid = z > float(depth_min)
    valid &= x >= float(margin)
    valid &= x < float(cam.image_width - margin)
    valid &= y >= float(margin)
    valid &= y < float(cam.image_height - margin)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False), valid


def filter_patches_by_projected_zbuffer(
    candidate_ids: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    depths: np.ndarray,
    width: int,
    depth_tolerance: float,
) -> np.ndarray:
    if candidate_ids.size == 0:
        return candidate_ids
    pixel_ids = pix_y[candidate_ids].astype(np.int64, copy=False) * int(width) + pix_x[candidate_ids].astype(np.int64, copy=False)
    depth_values = depths[candidate_ids].astype(np.float32, copy=False)
    order = np.lexsort((depth_values, pixel_ids))
    sorted_pixels = pixel_ids[order]
    sorted_depths = depth_values[order]
    unique_pixels, first_indices = np.unique(sorted_pixels, return_index=True)
    min_depths = sorted_depths[first_indices]
    pixel_positions = np.searchsorted(unique_pixels, pixel_ids)
    nearest_depth = min_depths[pixel_positions]
    keep = depth_values <= nearest_depth + float(depth_tolerance)
    return candidate_ids[keep]


def build_bounded_patches(
    mesh: trimesh.Trimesh,
    carriers_per_face: int,
    face_ids: np.ndarray,
    thickness_scale: float,
) -> Dict[str, np.ndarray]:
    if carriers_per_face not in BOUND_BARYCENTRIC_TEMPLATES:
        raise ValueError(f"Unsupported carriers_per_face={carriers_per_face}; expected one of {sorted(BOUND_BARYCENTRIC_TEMPLATES)}")
    bary_template = BOUND_BARYCENTRIC_TEMPLATES[carriers_per_face]

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    active_face_ids = np.asarray(face_ids, dtype=np.int64)
    active_faces = faces_all[active_face_ids]
    triangles = vertices[active_faces]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)[active_face_ids]

    centers = (triangles[:, None] * bary_template[None, :, :, None]).sum(axis=2).reshape(-1, 3)
    patch_face_ids = np.repeat(active_face_ids, carriers_per_face)
    bary_coords = np.tile(bary_template[None, :, :], (active_face_ids.shape[0], 1, 1)).reshape(-1, 3)
    normals = np.repeat(face_normals, carriers_per_face, axis=0)

    edge_u = triangles[:, 1] - triangles[:, 0]
    edge_u = edge_u / np.clip(np.linalg.norm(edge_u, axis=1, keepdims=True), 1e-6, None)
    edge_v = np.cross(face_normals, edge_u)
    edge_v = edge_v / np.clip(np.linalg.norm(edge_v, axis=1, keepdims=True), 1e-6, None)
    face_centers = triangles.mean(axis=1, keepdims=True)
    offsets = triangles - face_centers
    scale_u = np.max(np.abs((offsets * edge_u[:, None, :]).sum(axis=-1)), axis=1)
    scale_v = np.max(np.abs((offsets * edge_v[:, None, :]).sum(axis=-1)), axis=1)
    scale_n = np.minimum(scale_u, scale_v) * float(thickness_scale)

    return {
        "centers": centers.astype(np.float32, copy=False),
        "normals": normals.astype(np.float32, copy=False),
        "face_ids": patch_face_ids.astype(np.int64, copy=False),
        "bary_coords": bary_coords.astype(np.float32, copy=False),
        "scale_u": np.repeat(scale_u.astype(np.float32, copy=False), carriers_per_face),
        "scale_v": np.repeat(scale_v.astype(np.float32, copy=False), carriers_per_face),
        "scale_n": np.repeat(scale_n.astype(np.float32, copy=False), carriers_per_face),
        "tangent_u": np.repeat(edge_u.astype(np.float32, copy=False), carriers_per_face, axis=0),
        "tangent_v": np.repeat(edge_v.astype(np.float32, copy=False), carriers_per_face, axis=0),
    }


def load_train_cameras_only(dataset):
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path,
            dataset.images,
            dataset.eval,
            init_type=dataset.init_type,
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            dataset.source_path,
            dataset.white_background,
            dataset.eval,
        )
    else:
        raise RuntimeError(f"Could not recognize scene type under {dataset.source_path}")

    print("Loading Training Cameras")
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset)


def lookup_indexed_image(index: Dict[str, Path], image_name: str) -> Optional[Path]:
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


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_face_ids_from_payload(payload_path: str, key: str, num_faces: int) -> np.ndarray:
    payload = torch.load(payload_path, map_location="cpu")
    if key in payload:
        value = tensor_to_numpy(payload[key])
    elif "roi_face_ids" in payload:
        value = tensor_to_numpy(payload["roi_face_ids"])
    elif "roi_face_mask" in payload:
        value = tensor_to_numpy(payload["roi_face_mask"])
    else:
        raise KeyError(
            f"Face payload does not contain '{key}', 'roi_face_ids', or 'roi_face_mask': {payload_path}"
        )
    value = np.asarray(value)
    if value.dtype == np.bool_:
        value = value.reshape(-1)
        if value.shape[0] != int(num_faces):
            raise ValueError(f"Boolean face mask has {value.shape[0]} entries, expected {num_faces}.")
        face_ids = np.flatnonzero(value).astype(np.int64, copy=False)
    else:
        face_ids = value.reshape(-1).astype(np.int64, copy=False)
    if face_ids.size and np.any((face_ids < 0) | (face_ids >= int(num_faces))):
        raise ValueError(f"Face ids in payload are outside [0, {num_faces}).")
    return np.unique(face_ids).astype(np.int64, copy=False)


def select_face_ids(mesh_obj: trimesh.Trimesh, args) -> tuple[np.ndarray, str]:
    num_faces = len(mesh_obj.faces)
    if args.face_selection == "all_faces":
        face_ids = np.arange(num_faces, dtype=np.int64)
        source = "all_faces"
    elif args.face_selection == "payload":
        if not args.face_payload:
            raise ValueError("--face_selection payload requires --face_payload")
        face_ids = load_face_ids_from_payload(args.face_payload, args.face_payload_key, num_faces)
        source = f"payload:{Path(args.face_payload).name}:{args.face_payload_key}"
    else:
        raise ValueError(f"Unsupported face_selection={args.face_selection}")

    stride = max(int(args.face_stride), 1)
    if stride > 1:
        face_ids = face_ids[::stride]
        source += f"_stride_{stride}"
    max_faces = int(args.max_faces)
    if max_faces > 0 and face_ids.size > max_faces:
        if args.face_sample_mode == "random":
            rng = np.random.default_rng(int(args.random_seed))
            face_ids = np.sort(rng.choice(face_ids, size=max_faces, replace=False)).astype(np.int64, copy=False)
            source += f"_random_{max_faces}"
        else:
            face_ids = face_ids[:max_faces]
            source += f"_prefix_{max_faces}"
    return face_ids.astype(np.int64, copy=False), source


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


def build_patch_corners(carriers: Dict[str, np.ndarray], radius_scale: float) -> np.ndarray:
    centers = carriers["centers"]
    tangent_u = carriers["tangent_u"]
    tangent_v = carriers["tangent_v"]
    scale_u = carriers["scale_u"][:, None] * float(radius_scale)
    scale_v = carriers["scale_v"][:, None] * float(radius_scale)
    u = tangent_u * scale_u
    v = tangent_v * scale_v
    return np.stack(
        [
            centers - u - v,
            centers + u - v,
            centers + u + v,
            centers - u + v,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def project_patch_corners(view, corners: np.ndarray, patch_ids: np.ndarray, depth_min: float) -> np.ndarray:
    flat = corners[patch_ids].reshape(-1, 3)
    projected, _ = project_points_camera(view, flat, depth_min=depth_min, margin=-10_000)
    return projected[:, :2].reshape(-1, 4, 2).astype(np.float32, copy=False)


def export_patch_preview_ply(carriers: Dict[str, np.ndarray], view_count: np.ndarray, path: Path, max_points: int):
    centers = carriers["centers"]
    if centers.shape[0] == 0:
        return
    ids = np.arange(centers.shape[0], dtype=np.int64)
    if int(max_points) > 0 and ids.size > int(max_points):
        rng = np.random.default_rng(0)
        ids = np.sort(rng.choice(ids, size=int(max_points), replace=False)).astype(np.int64, copy=False)
    counts = view_count[ids].astype(np.float32, copy=False)
    denom = max(float(np.percentile(view_count[view_count > 0], 95)) if np.any(view_count > 0) else 1.0, 1.0)
    heat = np.clip(counts / denom, 0.0, 1.0)
    colors = np.stack(
        [
            np.round(255.0 * heat),
            np.round(180.0 * (1.0 - heat) + 60.0 * heat),
            np.round(255.0 * (1.0 - heat)),
            np.full_like(heat, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    cloud = trimesh.points.PointCloud(centers[ids], colors=colors)
    cloud.export(path)


def build_empty_render_stub(carriers: Dict[str, np.ndarray], view_count: np.ndarray, weight_sum: np.ndarray, path: Path):
    n = int(carriers["centers"].shape[0])
    np.savez_compressed(
        path,
        schema_version=np.asarray(["mesh_patch_render_stub_v0"]),
        centers=carriers["centers"].astype(np.float32, copy=False),
        normals=carriers["normals"].astype(np.float32, copy=False),
        tangent_u=carriers["tangent_u"].astype(np.float32, copy=False),
        tangent_v=carriers["tangent_v"].astype(np.float32, copy=False),
        scale_u=carriers["scale_u"].astype(np.float32, copy=False),
        scale_v=carriers["scale_v"].astype(np.float32, copy=False),
        scale_n=carriers["scale_n"].astype(np.float32, copy=False),
        face_ids=carriers["face_ids"].astype(np.int64, copy=False),
        bary_coords=carriers["bary_coords"].astype(np.float32, copy=False),
        fused_rgb=np.zeros((n, 3), dtype=np.float32),
        confidence=np.zeros((n,), dtype=np.float32),
        disagreement=np.zeros((n,), dtype=np.float32),
        view_count=view_count.astype(np.int32, copy=False),
        weight_sum=weight_sum.astype(np.float32, copy=False),
        valid_mask=np.zeros((n,), dtype=np.bool_),
    )


def build_parser():
    parser = ArgumentParser(
        description=(
            "Prepare camera-to-mesh-patch observation assets. This stops before prior fusion: "
            "it only builds local surface patch frames and records which training cameras see each patch."
        )
    )
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prior_dir", type=str, default=None, help="Optional: only process views with matching prior images.")
    parser.add_argument("--view_limit", type=int, default=0)
    parser.add_argument("--face_selection", choices=["all_faces", "payload"], default="all_faces")
    parser.add_argument("--face_payload", type=str, default=None)
    parser.add_argument("--face_payload_key", type=str, default="roi_face_ids")
    parser.add_argument("--face_stride", type=int, default=1)
    parser.add_argument("--max_faces", type=int, default=0)
    parser.add_argument("--face_sample_mode", choices=["prefix", "random"], default="random")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--carriers_per_face", type=int, choices=[1, 3, 4, 6], default=1)
    parser.add_argument("--huge_patch_threshold", type=int, default=2_000_000)
    parser.add_argument("--allow_huge_patch_bank", action="store_true")
    parser.add_argument("--patch_thickness_scale", type=float, default=0.05)
    parser.add_argument("--patch_corner_radius_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--min_view_cosine", type=float, default=0.0)
    parser.add_argument(
        "--visibility_mode",
        choices=["projection", "patch_zbuffer"],
        default="patch_zbuffer",
        help="projection keeps all front-facing projected patches; patch_zbuffer keeps only nearest patches per pixel.",
    )
    parser.add_argument("--zbuffer_depth_tolerance", type=float, default=0.03)
    parser.add_argument("--save_corners", action="store_true", help="Save projected four-corner footprints per visible patch.")
    parser.add_argument("--preview_max_points", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")
    return parser, model


def main():
    parser, model = build_parser()
    args = get_combined_args(parser)
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for patch observation preparation.")
        args.data_device = "cpu"
    safe_state(args.quiet)

    dataset = model.extract(args)
    train_cameras = load_train_cameras_only(dataset)
    prior_index = index_image_dir(args.prior_dir) if args.prior_dir else {}

    mesh_obj = load_triangle_mesh(args.mesh_path)
    print(
        f"[mesh-patch-observe] loaded mesh: vertices={len(mesh_obj.vertices)}, faces={len(mesh_obj.faces)}",
        flush=True,
    )
    face_ids, face_source = select_face_ids(mesh_obj, args)
    if face_ids.size == 0:
        raise RuntimeError("No mesh faces selected for patch observation preparation.")
    estimated_patches = int(face_ids.size) * int(args.carriers_per_face)
    if estimated_patches > int(args.huge_patch_threshold) and not bool(args.allow_huge_patch_bank):
        raise RuntimeError(
            f"Refusing to build {estimated_patches} patches by default. "
            f"Use --face_stride/--max_faces for a smoke test, or pass --allow_huge_patch_bank explicitly."
        )

    carriers = build_bounded_patches(
        mesh_obj,
        carriers_per_face=int(args.carriers_per_face),
        face_ids=face_ids,
        thickness_scale=float(args.patch_thickness_scale),
    )
    num_patches = int(carriers["centers"].shape[0])
    print(
        f"[mesh-patch-observe] built {num_patches} patches from {face_ids.size} faces ({face_source})",
        flush=True,
    )
    patch_corners = build_patch_corners(carriers, radius_scale=float(args.patch_corner_radius_scale))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = output_dir / "camera_patch_observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    patch_bank_path = output_dir / "mesh_patch_bank_v0.npz"
    render_stub_path = output_dir / "mesh_patch_render_stub_v0.npz"
    preview_path = output_dir / "mesh_patch_bank_preview_v0.ply"

    np.savez_compressed(
        patch_bank_path,
        schema_version=np.asarray(["mesh_patch_bank_v0"]),
        mesh_path=np.asarray([str(Path(args.mesh_path).resolve())]),
        centers=carriers["centers"],
        normals=carriers["normals"],
        tangent_u=carriers["tangent_u"],
        tangent_v=carriers["tangent_v"],
        scale_u=carriers["scale_u"],
        scale_v=carriers["scale_v"],
        scale_n=carriers["scale_n"],
        face_ids=carriers["face_ids"],
        bary_coords=carriers["bary_coords"],
        selected_face_ids=face_ids,
        patch_corners_world=patch_corners,
    )

    view_count = np.zeros((num_patches,), dtype=np.int32)
    weight_sum = np.zeros((num_patches,), dtype=np.float32)
    view_summaries = []
    processed = 0
    missing_prior = []

    progress = tqdm(train_cameras, desc="Projecting cameras to mesh patches", disable=bool(args.quiet), dynamic_ncols=True)
    for view_index, view in enumerate(progress):
        if int(args.view_limit) > 0 and processed >= int(args.view_limit):
            break
        prior_path = lookup_indexed_image(prior_index, view.image_name) if args.prior_dir else None
        if args.prior_dir and prior_path is None:
            if len(missing_prior) < 16:
                missing_prior.append(view.image_name)
            progress.set_postfix(processed=processed, missing_prior=len(missing_prior))
            continue

        projected, valid = project_points_camera(
            view,
            carriers["centers"],
            depth_min=float(args.depth_min),
            margin=0,
        )
        cam_center = camera_center_numpy(view)
        view_dir = cam_center[None, :] - carriers["centers"]
        view_dir_norm = view_dir / np.clip(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-6, None)
        view_cosine = (carriers["normals"] * view_dir_norm).sum(axis=1).astype(np.float32, copy=False)
        valid &= view_cosine >= float(args.min_view_cosine)

        pix_x = np.round(projected[:, 0]).astype(np.int64)
        pix_y = np.round(projected[:, 1]).astype(np.int64)
        in_bounds = (
            (pix_x >= 0)
            & (pix_x < int(view.image_width))
            & (pix_y >= 0)
            & (pix_y < int(view.image_height))
        )
        valid &= in_bounds
        valid_ids = np.flatnonzero(valid).astype(np.int64, copy=False)
        if args.visibility_mode == "patch_zbuffer" and valid_ids.size > 0:
            valid_ids = filter_patches_by_projected_zbuffer(
                valid_ids,
                pix_x=pix_x,
                pix_y=pix_y,
                depths=projected[:, 2],
                width=int(view.image_width),
                depth_tolerance=float(args.zbuffer_depth_tolerance),
            )

        view_count[valid_ids] += 1
        weights = np.clip(view_cosine[valid_ids], 0.0, 1.0).astype(np.float32, copy=False)
        weight_sum[valid_ids] += weights
        obs_path = obs_dir / f"{view.image_name}.npz"
        payload = {
            "schema_version": np.asarray(["camera_patch_observation_v0"]),
            "image_name": np.asarray([view.image_name]),
            "view_index": np.asarray([view_index], dtype=np.int32),
            "image_width": np.asarray([int(view.image_width)], dtype=np.int32),
            "image_height": np.asarray([int(view.image_height)], dtype=np.int32),
            "prior_path": np.asarray([str(prior_path) if prior_path is not None else ""]),
            "patch_ids": valid_ids.astype(np.int64, copy=False),
            "pixel_xy": projected[valid_ids, :2].astype(np.float32, copy=False),
            "depth": projected[valid_ids, 2].astype(np.float32, copy=False),
            "view_cosine": view_cosine[valid_ids].astype(np.float32, copy=False),
            "sample_weight": weights,
        }
        if bool(args.save_corners):
            payload["corner_xy"] = project_patch_corners(
                view,
                patch_corners,
                valid_ids,
                depth_min=float(args.depth_min),
            )
        np.savez_compressed(obs_path, **payload)

        view_summaries.append(
            {
                "image_name": view.image_name,
                "view_index": int(view_index),
                "prior_path": str(prior_path) if prior_path is not None else None,
                "visible_patch_count": int(valid_ids.size),
                "mean_view_cosine": float(np.mean(view_cosine[valid_ids])) if valid_ids.size else 0.0,
                "observation_path": str(obs_path.resolve()),
            }
        )
        processed += 1
        progress.set_postfix(processed=processed, visible=int(valid_ids.size))

    build_empty_render_stub(carriers, view_count, weight_sum, render_stub_path)
    export_patch_preview_ply(carriers, view_count, preview_path, max_points=int(args.preview_max_points))

    summary = {
        "mode": "mesh_patch_observation_prepare_v0",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "patch_bank_path": str(patch_bank_path.resolve()),
        "render_stub_path": str(render_stub_path.resolve()),
        "observation_dir": str(obs_dir.resolve()),
        "preview_path": str(preview_path.resolve()),
        "parameters": {
            "face_selection": args.face_selection,
            "face_payload": args.face_payload,
            "face_payload_key": args.face_payload_key,
            "face_source": face_source,
            "face_stride": int(args.face_stride),
            "max_faces": int(args.max_faces),
            "face_sample_mode": args.face_sample_mode,
            "random_seed": int(args.random_seed),
            "carriers_per_face": int(args.carriers_per_face),
            "huge_patch_threshold": int(args.huge_patch_threshold),
            "allow_huge_patch_bank": bool(args.allow_huge_patch_bank),
            "patch_thickness_scale": float(args.patch_thickness_scale),
            "patch_corner_radius_scale": float(args.patch_corner_radius_scale),
            "depth_min": float(args.depth_min),
            "min_view_cosine": float(args.min_view_cosine),
            "visibility_mode": args.visibility_mode,
            "zbuffer_depth_tolerance": float(args.zbuffer_depth_tolerance),
            "save_corners": bool(args.save_corners),
            "prior_dir": args.prior_dir,
            "view_limit": int(args.view_limit),
        },
        "counts": {
            "train_camera_count": int(len(train_cameras)),
            "views_processed": int(processed),
            "missing_prior_sample": missing_prior,
            "selected_face_count": int(face_ids.size),
            "patch_count": int(num_patches),
            "observed_patch_count": int(np.sum(view_count > 0)),
            "multi_view_patch_count": int(np.sum(view_count >= 2)),
        },
        "stats": {
            "patch_view_count": stats_from_array(view_count.astype(np.float32)),
            "patch_weight_sum": stats_from_array(weight_sum.astype(np.float32)),
            "visible_patch_count_per_view": stats_from_array(
                np.asarray([item["visible_patch_count"] for item in view_summaries], dtype=np.float32)
            ),
        },
        "render_interface": {
            "status": "stub_only_no_prior_fusion_yet",
            "geometry_payload": str(render_stub_path.resolve()),
            "expected_fusion_fields": [
                "fused_rgb",
                "confidence",
                "disagreement",
                "valid_mask",
            ],
            "compatible_loader": "utils.mesh_fusion_render.load_mesh_fusion_payload",
            "note": (
                "Future fusion should fill fused_rgb/confidence/disagreement/valid_mask "
                "while keeping centers/normals/tangent_u/tangent_v/scale_u/scale_v."
            ),
        },
        "view_summaries": view_summaries,
    }
    summary_path = output_dir / "mesh_patch_observations_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved patch bank to: {patch_bank_path}")
    print(f"Saved camera observations to: {obs_dir}")
    print(f"Saved render stub to: {render_stub_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
