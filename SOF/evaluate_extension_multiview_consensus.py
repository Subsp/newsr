import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image


STATUS_ORDER = {
    "success": 0,
    "weak": 1,
    "dead": 2,
}

VALIDATION_LABELS = ("validated", "ambiguous", "rejected", "insufficient")


def parse_args():
    parser = ArgumentParser(
        description="Score extension-probe target faces using GT-free multiview training-image consistency."
    )
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument(
        "--probe_records",
        type=str,
        required=True,
        help="probe_outcomes.json or manifest-like json. target_face_id is required.",
    )
    parser.add_argument(
        "--baseline_render_dir",
        type=str,
        required=True,
        help="Directory containing train render images for the baseline model.",
    )
    parser.add_argument(
        "--probe_render_dir",
        type=str,
        required=True,
        help="Directory containing train render images for the probe model.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--patch_radius", type=int, default=9)
    parser.add_argument("--depth_min", type=float, default=0.2)
    parser.add_argument("--front_facing_only", action="store_true")
    parser.add_argument("--delta_epsilon", type=float, default=5e-4)
    parser.add_argument("--min_visible_views", type=int, default=3)
    parser.add_argument("--min_positive_views", type=int, default=2)
    parser.add_argument("--positive_ratio_threshold", type=float, default=0.5)
    parser.add_argument("--min_mean_delta", type=float, default=0.0)
    parser.add_argument("--max_negative_views", type=int, default=1)

    parser.add_argument(
        "--target_chunk_size",
        type=int,
        default=0,
        help="If >0, only score one chunk of target faces at a time.",
    )
    parser.add_argument(
        "--target_chunk_index",
        type=int,
        default=0,
        help="0-based target chunk index used with --target_chunk_size.",
    )
    parser.add_argument(
        "--camera_chunk_size",
        type=int,
        default=0,
        help="Optional chunking over training cameras. Useful for partial scoring runs.",
    )
    parser.add_argument(
        "--camera_chunk_index",
        type=int,
        default=0,
        help="0-based camera chunk index used with --camera_chunk_size.",
    )
    return parser.parse_args()


def load_train_camera_infos(scene_path: str, images: str, eval_mode: bool):
    from scene.dataset_readers import sceneLoadTypeCallbacks

    if os.path.exists(os.path.join(scene_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](scene_path, images, eval_mode, init_type="sfm")
    elif os.path.exists(os.path.join(scene_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](scene_path, False, eval_mode)
    else:
        raise ValueError(f"Could not recognize scene type at {scene_path}")
    return scene_info.train_cameras


def list_render_images(render_dir: str) -> List[Path]:
    root = Path(render_dir)
    if not root.exists():
        raise FileNotFoundError(f"Render directory not found: {root}")
    files = sorted([p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not files:
        raise RuntimeError(f"No render images found in: {root}")
    return files


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def build_integral_image(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values.astype(np.float64), ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def patch_mean_from_integral(integral: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: int) -> np.ndarray:
    x0 = xs - radius
    x1 = xs + radius + 1
    y0 = ys - radius
    y1 = ys + radius + 1
    total = (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )
    patch_area = float((2 * radius + 1) ** 2)
    return total / patch_area


def aggregate_records(records: List[Dict], mesh: trimesh.Trimesh) -> List[Dict]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    by_face: Dict[int, Dict] = {}

    for item in records:
        if "target_face_id" not in item:
            continue
        face_id = int(item["target_face_id"])
        if face_id < 0 or face_id >= faces.shape[0]:
            continue

        tri = vertices[faces[face_id]]
        center = tri.mean(axis=0)
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-12:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        status = str(item.get("status", "unknown"))
        final_opacity = float(item.get("final_opacity", item.get("probe_opacity_init", 0.0)))
        existing = by_face.get(face_id)

        if existing is None:
            by_face[face_id] = {
                "target_face_id": face_id,
                "center": center.astype(np.float32).tolist(),
                "normal": normal.astype(np.float32).tolist(),
                "probe_count": 1,
                "optimization_status": status,
                "max_final_opacity": final_opacity,
            }
            continue

        existing["probe_count"] += 1
        existing["max_final_opacity"] = max(existing["max_final_opacity"], final_opacity)
        if STATUS_ORDER.get(status, -1) > STATUS_ORDER.get(existing["optimization_status"], -1):
            existing["optimization_status"] = status

    aggregated = list(by_face.values())
    aggregated.sort(key=lambda item: int(item["target_face_id"]))
    return aggregated


def slice_chunk(items: List, chunk_size: int, chunk_index: int) -> Tuple[List, Dict[str, int]]:
    if chunk_size <= 0:
        return items, {"enabled": 0, "chunk_size": 0, "chunk_index": 0, "start": 0, "end": len(items), "total": len(items)}
    start = chunk_size * max(chunk_index, 0)
    end = min(start + chunk_size, len(items))
    if start >= len(items):
        return [], {"enabled": 1, "chunk_size": chunk_size, "chunk_index": chunk_index, "start": start, "end": start, "total": len(items)}
    return items[start:end], {
        "enabled": 1,
        "chunk_size": chunk_size,
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "total": len(items),
    }


def project_centers(
    cam_info,
    image_width: int,
    image_height: int,
    centers_xyz: np.ndarray,
    normals_xyz: np.ndarray,
    patch_radius: int,
    depth_min: float,
    front_facing_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz_cam = centers_xyz @ cam_info.R + cam_info.T[None, :]
    z = xyz_cam[:, 2]
    valid = z > depth_min

    tan_fovx = math.tan(float(cam_info.FovX) / 2.0)
    tan_fovy = math.tan(float(cam_info.FovY) / 2.0)
    focal_x = image_width / (2.0 * tan_fovx)
    focal_y = image_height / (2.0 * tan_fovy)

    x = xyz_cam[:, 0] / np.clip(z, 1e-6, None) * focal_x + image_width / 2.0
    y = xyz_cam[:, 1] / np.clip(z, 1e-6, None) * focal_y + image_height / 2.0

    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)

    valid &= xi >= patch_radius
    valid &= xi < (image_width - patch_radius)
    valid &= yi >= patch_radius
    valid &= yi < (image_height - patch_radius)

    if front_facing_only:
        cam_center = (-cam_info.T @ cam_info.R.T).astype(np.float32)
        view_dir = cam_center[None, :] - centers_xyz
        valid &= np.einsum("ij,ij->i", normals_xyz, view_dir) > 0.0

    projected = np.stack([x, y, z], axis=1)
    return projected, valid


def classify_target(
    visible_views: int,
    positive_views: int,
    negative_views: int,
    mean_delta: float,
    args,
) -> str:
    if visible_views < args.min_visible_views:
        return "insufficient"

    positive_ratio = float(positive_views) / max(visible_views, 1)
    if (
        positive_views >= args.min_positive_views
        and positive_ratio >= args.positive_ratio_threshold
        and negative_views <= args.max_negative_views
        and mean_delta > args.min_mean_delta
    ):
        return "validated"

    if negative_views > positive_views and mean_delta < -args.min_mean_delta:
        return "rejected"

    return "ambiguous"


def export_face_subset(mesh: trimesh.Trimesh, face_ids: List[int], path: Path):
    if not face_ids:
        return
    unique_face_ids = np.unique(np.asarray(face_ids, dtype=np.int64))
    submesh = mesh.submesh([unique_face_ids], append=True, repair=False)
    submesh.export(path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(args.mesh_path, process=False)
    raw_records = json.loads(Path(args.probe_records).read_text())
    aggregated = aggregate_records(raw_records, mesh)
    if not aggregated:
        raise RuntimeError("No target_face_id entries were found in probe_records.")

    selected_targets, target_chunk_meta = slice_chunk(aggregated, args.target_chunk_size, args.target_chunk_index)
    if not selected_targets:
        raise RuntimeError(f"Target chunk is empty: {target_chunk_meta}")

    cameras = load_train_camera_infos(args.scene_path, args.images, args.eval)
    camera_indices = list(range(len(cameras)))
    camera_indices, camera_chunk_meta = slice_chunk(camera_indices, args.camera_chunk_size, args.camera_chunk_index)
    if not camera_indices:
        raise RuntimeError(f"Camera chunk is empty: {camera_chunk_meta}")

    baseline_render_paths = list_render_images(args.baseline_render_dir)
    probe_render_paths = list_render_images(args.probe_render_dir)
    if len(baseline_render_paths) != len(cameras):
        raise RuntimeError(
            f"Baseline render count ({len(baseline_render_paths)}) does not match train camera count ({len(cameras)})."
        )
    if len(probe_render_paths) != len(cameras):
        raise RuntimeError(
            f"Probe render count ({len(probe_render_paths)}) does not match train camera count ({len(cameras)})."
        )

    centers_xyz = np.asarray([item["center"] for item in selected_targets], dtype=np.float32)
    normals_xyz = np.asarray([item["normal"] for item in selected_targets], dtype=np.float32)

    visible_views = np.zeros((len(selected_targets),), dtype=np.int32)
    positive_views = np.zeros((len(selected_targets),), dtype=np.int32)
    negative_views = np.zeros((len(selected_targets),), dtype=np.int32)
    neutral_views = np.zeros((len(selected_targets),), dtype=np.int32)
    delta_sum = np.zeros((len(selected_targets),), dtype=np.float64)
    delta_min = np.full((len(selected_targets),), np.inf, dtype=np.float64)
    delta_max = np.full((len(selected_targets),), -np.inf, dtype=np.float64)

    for local_camera_rank, camera_idx in enumerate(camera_indices):
        cam_info = cameras[camera_idx]
        obs_image = load_rgb(Path(cam_info.image_path))
        baseline_image = load_rgb(baseline_render_paths[camera_idx])
        probe_image = load_rgb(probe_render_paths[camera_idx])

        if baseline_image.shape != obs_image.shape or probe_image.shape != obs_image.shape:
            raise RuntimeError(
                f"Image shape mismatch at camera {cam_info.image_name}: "
                f"obs={obs_image.shape}, baseline={baseline_image.shape}, probe={probe_image.shape}"
            )

        image_height, image_width = obs_image.shape[:2]
        projected, valid = project_centers(
            cam_info=cam_info,
            image_width=image_width,
            image_height=image_height,
            centers_xyz=centers_xyz,
            normals_xyz=normals_xyz,
            patch_radius=args.patch_radius,
            depth_min=args.depth_min,
            front_facing_only=args.front_facing_only,
        )

        if not np.any(valid):
            continue

        baseline_err = np.abs(baseline_image - obs_image).mean(axis=2)
        probe_err = np.abs(probe_image - obs_image).mean(axis=2)
        baseline_integral = build_integral_image(baseline_err)
        probe_integral = build_integral_image(probe_err)

        visible_ids = np.flatnonzero(valid)
        xs = np.rint(projected[visible_ids, 0]).astype(np.int32)
        ys = np.rint(projected[visible_ids, 1]).astype(np.int32)

        baseline_patch = patch_mean_from_integral(baseline_integral, xs, ys, args.patch_radius)
        probe_patch = patch_mean_from_integral(probe_integral, xs, ys, args.patch_radius)
        deltas = baseline_patch - probe_patch

        visible_views[visible_ids] += 1
        positive_mask = deltas > args.delta_epsilon
        negative_mask = deltas < -args.delta_epsilon
        neutral_mask = ~(positive_mask | negative_mask)

        positive_views[visible_ids] += positive_mask.astype(np.int32)
        negative_views[visible_ids] += negative_mask.astype(np.int32)
        neutral_views[visible_ids] += neutral_mask.astype(np.int32)
        delta_sum[visible_ids] += deltas
        delta_min[visible_ids] = np.minimum(delta_min[visible_ids], deltas)
        delta_max[visible_ids] = np.maximum(delta_max[visible_ids], deltas)

        if (local_camera_rank + 1) % 16 == 0 or (local_camera_rank + 1) == len(camera_indices):
            print(
                f"[multiview-consensus] camera {local_camera_rank + 1}/{len(camera_indices)} "
                f"name={cam_info.image_name} visible_targets={int(valid.sum())}"
            )

    output_records: List[Dict] = []
    label_to_face_ids: Dict[str, List[int]] = {label: [] for label in VALIDATION_LABELS}

    for idx, item in enumerate(selected_targets):
        mean_delta = float(delta_sum[idx] / max(visible_views[idx], 1))
        min_delta = 0.0 if not np.isfinite(delta_min[idx]) else float(delta_min[idx])
        max_delta = 0.0 if not np.isfinite(delta_max[idx]) else float(delta_max[idx])
        label = classify_target(
            visible_views=int(visible_views[idx]),
            positive_views=int(positive_views[idx]),
            negative_views=int(negative_views[idx]),
            mean_delta=mean_delta,
            args=args,
        )
        positive_ratio = float(positive_views[idx]) / max(int(visible_views[idx]), 1)
        record = {
            "target_face_id": int(item["target_face_id"]),
            "center": item["center"],
            "normal": item["normal"],
            "probe_count": int(item["probe_count"]),
            "optimization_status": str(item["optimization_status"]),
            "max_final_opacity": float(item["max_final_opacity"]),
            "visible_views": int(visible_views[idx]),
            "positive_views": int(positive_views[idx]),
            "negative_views": int(negative_views[idx]),
            "neutral_views": int(neutral_views[idx]),
            "positive_ratio": positive_ratio,
            "mean_delta": mean_delta,
            "min_delta": min_delta,
            "max_delta": max_delta,
            "validation_status": label,
        }
        output_records.append(record)
        label_to_face_ids[label].append(int(item["target_face_id"]))

    chunk_suffix = "all"
    if target_chunk_meta["enabled"]:
        chunk_suffix = f"targets_{target_chunk_meta['start']:06d}_{target_chunk_meta['end']:06d}"
    if camera_chunk_meta["enabled"]:
        chunk_suffix = f"{chunk_suffix}_cams_{camera_chunk_meta['start']:04d}_{camera_chunk_meta['end']:04d}"

    records_path = output_dir / f"per_target_consensus_{chunk_suffix}.json"
    summary_path = output_dir / f"consensus_summary_{chunk_suffix}.json"
    records_path.write_text(json.dumps(output_records, indent=2), encoding="utf-8")

    for label in VALIDATION_LABELS:
        export_face_subset(mesh, label_to_face_ids[label], output_dir / f"{label}_target_faces_{chunk_suffix}.ply")

    summary = {
        "mode": "gt_free_multiview_consensus",
        "scene_path": str(Path(args.scene_path).resolve()),
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "probe_records": str(Path(args.probe_records).resolve()),
        "baseline_render_dir": str(Path(args.baseline_render_dir).resolve()),
        "probe_render_dir": str(Path(args.probe_render_dir).resolve()),
        "n_raw_records": len(raw_records),
        "n_unique_target_faces_total": len(aggregated),
        "n_unique_target_faces_scored": len(selected_targets),
        "n_train_cameras_total": len(cameras),
        "n_train_cameras_used": len(camera_indices),
        "target_chunk": target_chunk_meta,
        "camera_chunk": camera_chunk_meta,
        "parameters": {
            "patch_radius": args.patch_radius,
            "depth_min": args.depth_min,
            "front_facing_only": bool(args.front_facing_only),
            "delta_epsilon": args.delta_epsilon,
            "min_visible_views": args.min_visible_views,
            "min_positive_views": args.min_positive_views,
            "positive_ratio_threshold": args.positive_ratio_threshold,
            "min_mean_delta": args.min_mean_delta,
            "max_negative_views": args.max_negative_views,
        },
        "counts": {label: len(label_to_face_ids[label]) for label in VALIDATION_LABELS},
        "paths": {
            "records": str(records_path),
            **{
                f"{label}_target_faces": str((output_dir / f"{label}_target_faces_{chunk_suffix}.ply").resolve())
                for label in VALIDATION_LABELS
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved records to: {records_path}")
    print(f"saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
