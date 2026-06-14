import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import trimesh


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


def load_triangle_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    if hasattr(mesh_obj, "dump"):
        dumped = mesh_obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise ValueError(f"Failed to load triangle mesh from {mesh_path}")


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_mask_from_payload(payload: dict, key: str, total: int) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Payload does not contain mask key '{key}'. Available keys: {sorted(payload.keys())}")
    value = tensor_to_numpy(payload[key])
    value = np.asarray(value)
    if value.dtype == np.bool_:
        if value.reshape(-1).shape[0] != total:
            raise ValueError(f"Boolean mask '{key}' has {value.size} entries, expected {total}.")
        return value.reshape(-1).astype(bool, copy=False)
    ids = value.reshape(-1).astype(np.int64, copy=False)
    mask = np.zeros((total,), dtype=bool)
    if ids.size:
        if np.any((ids < 0) | (ids >= total)):
            raise ValueError(f"Index mask '{key}' contains ids outside [0, {total}).")
        mask[ids] = True
    return mask


def export_face_subset(mesh_obj: trimesh.Trimesh, face_ids: np.ndarray, path: Path, max_faces: int, color):
    if face_ids.size == 0:
        return
    preview_face_ids = np.asarray(face_ids, dtype=np.int64)
    if max_faces > 0 and preview_face_ids.size > int(max_faces):
        rng = np.random.default_rng(0)
        preview_face_ids = np.sort(rng.choice(preview_face_ids, size=int(max_faces), replace=False))
    submesh = mesh_obj.submesh([preview_face_ids], append=True, repair=False)
    submesh.visual.face_colors = np.tile(np.asarray(color, dtype=np.uint8)[None], (len(submesh.faces), 1))
    submesh.export(path)


def maybe_filter_range(values: np.ndarray, mask: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    if values is None:
        return mask
    out = mask.copy()
    if min_value > 0.0:
        out &= values >= float(min_value)
    if max_value > 0.0:
        out &= values <= float(max_value)
    return out


def main():
    parser = ArgumentParser(
        description=(
            "Build meshfusion surface-supported face payload from the complement of outlier GS. "
            "This is for mesh-boundedGS prior fusion, not for outlier/edge GS finetuning."
        )
    )
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--outlier_mask_key", type=str, default="candidate_mask")
    parser.add_argument(
        "--support_mode",
        choices=["non_outlier", "near_surface"],
        default="non_outlier",
        help="non_outlier uses the complement of outlier_mask_key; near_surface ignores that key and filters by surface_distance.",
    )
    parser.add_argument("--max_surface_distance", type=float, default=0.0)
    parser.add_argument("--min_visible_views", type=int, default=0)
    parser.add_argument("--max_nearest_visible_depth", type=float, default=0.0)
    parser.add_argument("--min_opacity", type=float, default=0.0)
    parser.add_argument("--min_support_per_face", type=int, default=1)
    parser.add_argument("--max_supported_faces", type=int, default=0)
    parser.add_argument("--face_sample_mode", choices=["top_count", "random"], default="top_count")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--preview_max_faces", type=int, default=200000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mesh-surface-support] loading mesh: {args.mesh_path}")
    mesh_obj = load_triangle_mesh(args.mesh_path)
    num_faces = int(len(mesh_obj.faces))
    num_vertices = int(len(mesh_obj.vertices))
    print(f"[mesh-surface-support] mesh vertices={num_vertices}, faces={num_faces}")

    payload = torch.load(args.candidate_payload, map_location="cpu")
    nearest_face_id = tensor_to_numpy(payload["nearest_face_id"]).astype(np.int64, copy=False).reshape(-1)
    total_gaussians = int(nearest_face_id.shape[0])
    valid_face_mask = (nearest_face_id >= 0) & (nearest_face_id < num_faces)

    outlier_mask = load_mask_from_payload(payload, args.outlier_mask_key, total=total_gaussians)
    if args.support_mode == "non_outlier":
        support_mask = (~outlier_mask) & valid_face_mask
    else:
        if "surface_distance" not in payload:
            raise KeyError("--support_mode near_surface requires 'surface_distance' in candidate payload.")
        surface_distance = tensor_to_numpy(payload["surface_distance"]).astype(np.float32, copy=False).reshape(-1)
        if float(args.max_surface_distance) <= 0.0:
            raise ValueError("--support_mode near_surface requires --max_surface_distance > 0")
        support_mask = (surface_distance <= float(args.max_surface_distance)) & valid_face_mask

    surface_distance = (
        tensor_to_numpy(payload["surface_distance"]).astype(np.float32, copy=False).reshape(-1)
        if "surface_distance" in payload
        else None
    )
    visible_view_count = (
        tensor_to_numpy(payload["visible_view_count"]).astype(np.int32, copy=False).reshape(-1)
        if "visible_view_count" in payload
        else None
    )
    min_visible_depth = (
        tensor_to_numpy(payload["min_visible_depth"]).astype(np.float32, copy=False).reshape(-1)
        if "min_visible_depth" in payload
        else None
    )
    opacity = (
        tensor_to_numpy(payload["opacity"]).astype(np.float32, copy=False).reshape(-1)
        if "opacity" in payload
        else None
    )

    if visible_view_count is not None and int(args.min_visible_views) > 0:
        support_mask &= visible_view_count >= int(args.min_visible_views)
    if min_visible_depth is not None and float(args.max_nearest_visible_depth) > 0.0:
        support_mask &= min_visible_depth <= float(args.max_nearest_visible_depth)
    if opacity is not None and float(args.min_opacity) > 0.0:
        support_mask &= opacity >= float(args.min_opacity)
    if surface_distance is not None and args.support_mode == "non_outlier" and float(args.max_surface_distance) > 0.0:
        support_mask &= surface_distance <= float(args.max_surface_distance)

    support_ids = np.flatnonzero(support_mask).astype(np.int64, copy=False)
    if support_ids.size == 0:
        raise RuntimeError("No surface-supported GS remain after filtering.")

    support_face_ids_raw = nearest_face_id[support_ids]
    face_ids, support_counts = np.unique(support_face_ids_raw, return_counts=True)
    keep = support_counts >= max(int(args.min_support_per_face), 1)
    face_ids = face_ids[keep].astype(np.int64, copy=False)
    support_counts = support_counts[keep].astype(np.int32, copy=False)
    if face_ids.size == 0:
        raise RuntimeError("No supported mesh faces remain after min_support_per_face filtering.")

    if int(args.max_supported_faces) > 0 and face_ids.size > int(args.max_supported_faces):
        max_faces = int(args.max_supported_faces)
        if args.face_sample_mode == "random":
            rng = np.random.default_rng(int(args.random_seed))
            chosen = np.sort(rng.choice(face_ids.shape[0], size=max_faces, replace=False))
        else:
            # Prefer faces that are explained by more non-outlier GS.
            chosen = np.argsort(-support_counts, kind="stable")[:max_faces]
            chosen = np.sort(chosen)
        face_ids = face_ids[chosen]
        support_counts = support_counts[chosen]

    supported_face_mask = np.zeros((num_faces,), dtype=bool)
    supported_face_mask[face_ids] = True

    output_payload = {
        "mode": "meshfusion_surface_support_v0",
        "supported_face_ids": torch.from_numpy(face_ids.astype(np.int64, copy=False)),
        "supported_face_mask": torch.from_numpy(supported_face_mask),
        # Keep aliases so prepare_prior_fusion_v0 can consume this via the existing roi_payload path.
        "roi_face_ids": torch.from_numpy(face_ids.astype(np.int64, copy=False)),
        "roi_face_mask": torch.from_numpy(supported_face_mask),
        "supported_face_gs_count": torch.from_numpy(support_counts.astype(np.int32, copy=False)),
        "support_ids": torch.from_numpy(support_ids.astype(np.int64, copy=False)),
        "support_mask": torch.from_numpy(support_mask.copy()),
        "outlier_mask": torch.from_numpy(outlier_mask.copy()),
    }
    payload_path = output_dir / "meshfusion_surface_support_faces_v0.pt"
    torch.save(output_payload, str(payload_path))

    preview_path = output_dir / "meshfusion_surface_support_faces_v0.ply"
    export_face_subset(mesh_obj, face_ids, preview_path, max_faces=int(args.preview_max_faces), color=(80, 220, 120, 255))

    summary = {
        "mode": "meshfusion_surface_support_v0",
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "candidate_payload": str(Path(args.candidate_payload).resolve()),
        "output_dir": str(output_dir.resolve()),
        "payload_path": str(payload_path.resolve()),
        "parameters": {
            "outlier_mask_key": args.outlier_mask_key,
            "support_mode": args.support_mode,
            "max_surface_distance": float(args.max_surface_distance),
            "min_visible_views": int(args.min_visible_views),
            "max_nearest_visible_depth": float(args.max_nearest_visible_depth),
            "min_opacity": float(args.min_opacity),
            "min_support_per_face": int(args.min_support_per_face),
            "max_supported_faces": int(args.max_supported_faces),
            "face_sample_mode": args.face_sample_mode,
            "random_seed": int(args.random_seed),
            "preview_max_faces": int(args.preview_max_faces),
        },
        "counts": {
            "mesh_vertices": num_vertices,
            "mesh_faces": num_faces,
            "total_gaussians": total_gaussians,
            "outlier_gaussians": int(np.sum(outlier_mask)),
            "support_gaussians": int(support_ids.size),
            "supported_face_count": int(face_ids.size),
            "support_ratio_vs_all_gs": float(support_ids.size / max(total_gaussians, 1)),
            "supported_face_ratio_vs_mesh": float(face_ids.size / max(num_faces, 1)),
        },
        "stats": {
            "supported_face_gs_count": stats_from_array(support_counts.astype(np.float32)),
            "surface_distance_support": stats_from_array(surface_distance[support_ids] if surface_distance is not None else np.zeros((0,), dtype=np.float32)),
            "visible_view_count_support": stats_from_array(
                visible_view_count[support_ids].astype(np.float32) if visible_view_count is not None else np.zeros((0,), dtype=np.float32)
            ),
            "min_visible_depth_support": stats_from_array(
                min_visible_depth[support_ids] if min_visible_depth is not None else np.zeros((0,), dtype=np.float32)
            ),
            "opacity_support": stats_from_array(opacity[support_ids] if opacity is not None else np.zeros((0,), dtype=np.float32)),
        },
        "previews": {
            "supported_faces_ply": str(preview_path.resolve()),
        },
    }
    summary_path = output_dir / "meshfusion_surface_support_faces_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved surface-support payload to: {payload_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
