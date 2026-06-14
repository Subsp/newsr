import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import trimesh
from scipy import sparse
from tqdm import tqdm


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


def load_candidate_mask(payload: dict, key: str) -> np.ndarray:
    if key in payload:
        mask = tensor_to_numpy(payload[key])
        if mask.dtype == np.bool_:
            return mask.astype(bool, copy=False)
        ids = mask.astype(np.int64, copy=False).reshape(-1)
        total = int(tensor_to_numpy(payload["nearest_face_id"]).shape[0])
        out = np.zeros((total,), dtype=bool)
        out[ids] = True
        return out
    if "selected_ids" in payload:
        total = int(tensor_to_numpy(payload["nearest_face_id"]).shape[0])
        out = np.zeros((total,), dtype=bool)
        ids = tensor_to_numpy(payload["selected_ids"]).astype(np.int64, copy=False).reshape(-1)
        out[ids] = True
        return out
    raise KeyError(f"Could not find '{key}' or 'selected_ids' in candidate payload.")


def seed_faces_from_outliers(
    payload: dict,
    candidate_mask_key: str,
    min_outliers_per_face: int,
    max_seed_faces: int,
    random_seed: int,
):
    candidate_mask = load_candidate_mask(payload, candidate_mask_key)
    candidate_ids = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    nearest_face_id = tensor_to_numpy(payload["nearest_face_id"]).astype(np.int64, copy=False)
    selected_face_ids = nearest_face_id[candidate_ids]
    selected_face_ids = selected_face_ids[selected_face_ids >= 0]
    if selected_face_ids.size == 0:
        raise RuntimeError("No selected outlier candidates have valid nearest_face_id.")

    seed_face_ids, seed_counts = np.unique(selected_face_ids, return_counts=True)
    keep = seed_counts >= max(int(min_outliers_per_face), 1)
    seed_face_ids = seed_face_ids[keep]
    seed_counts = seed_counts[keep]
    if seed_face_ids.size == 0:
        raise RuntimeError("No seed faces remain after min_outliers_per_face filtering.")

    if max_seed_faces > 0 and seed_face_ids.size > int(max_seed_faces):
        rng = np.random.default_rng(int(random_seed))
        chosen = np.sort(rng.choice(seed_face_ids.shape[0], size=int(max_seed_faces), replace=False))
        seed_face_ids = seed_face_ids[chosen]
        seed_counts = seed_counts[chosen]

    return candidate_mask, candidate_ids, seed_face_ids.astype(np.int64, copy=False), seed_counts.astype(np.int32, copy=False)


def build_vertex_face_csr(faces: np.ndarray, num_vertices: int):
    face_ids = np.repeat(np.arange(faces.shape[0], dtype=np.int64), 3)
    vertex_ids = faces.reshape(-1).astype(np.int64, copy=False)
    data = np.ones((vertex_ids.shape[0],), dtype=np.bool_)
    return sparse.csr_matrix((data, (vertex_ids, face_ids)), shape=(num_vertices, faces.shape[0]), dtype=np.bool_)


def expand_faces_by_vertex_ring(
    faces: np.ndarray,
    num_vertices: int,
    seed_face_ids: np.ndarray,
    ring_count: int,
    max_roi_faces: int,
    random_seed: int,
) -> np.ndarray:
    roi_face_ids = np.unique(seed_face_ids.astype(np.int64, copy=False))
    if ring_count <= 0:
        return roi_face_ids

    vertex_to_faces = build_vertex_face_csr(faces, num_vertices)
    for ring_idx in tqdm(range(int(ring_count)), desc="Expanding ROI face rings"):
        vertices = np.unique(faces[roi_face_ids].reshape(-1))
        neighbor_faces = vertex_to_faces[vertices].indices.astype(np.int64, copy=False)
        roi_face_ids = np.unique(np.concatenate([roi_face_ids, neighbor_faces], axis=0))
        if max_roi_faces > 0 and roi_face_ids.size > int(max_roi_faces):
            rng = np.random.default_rng(int(random_seed) + ring_idx)
            keep = np.sort(rng.choice(roi_face_ids.shape[0], size=int(max_roi_faces), replace=False))
            roi_face_ids = roi_face_ids[keep]
            break
    return roi_face_ids.astype(np.int64, copy=False)


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


def main():
    parser = ArgumentParser(description="Build meshfusion ROI faces from mesh-outside GS candidates.")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--candidate_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--min_outliers_per_seed_face", type=int, default=1)
    parser.add_argument("--max_seed_faces", type=int, default=0)
    parser.add_argument("--ring_count", type=int, default=2)
    parser.add_argument(
        "--ring_mode",
        choices=["none", "vertex"],
        default="vertex",
        help="vertex expands to faces sharing any vertex with current ROI. none keeps only nearest faces.",
    )
    parser.add_argument("--max_roi_faces", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--preview_max_faces", type=int, default=200000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[meshfusion-roi] loading mesh: {args.mesh_path}")
    mesh_obj = load_triangle_mesh(args.mesh_path)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    num_vertices = int(len(mesh_obj.vertices))
    num_faces = int(faces.shape[0])
    print(f"[meshfusion-roi] mesh vertices={num_vertices}, faces={num_faces}")

    payload = torch.load(args.candidate_payload, map_location="cpu")
    candidate_mask, candidate_ids, seed_face_ids, seed_counts = seed_faces_from_outliers(
        payload=payload,
        candidate_mask_key=args.candidate_mask_key,
        min_outliers_per_face=int(args.min_outliers_per_seed_face),
        max_seed_faces=int(args.max_seed_faces),
        random_seed=int(args.random_seed),
    )
    if np.any(seed_face_ids >= num_faces):
        raise ValueError("Candidate payload nearest_face_id contains face ids outside mesh face count.")

    print(
        "[meshfusion-roi] seed faces: "
        f"{seed_face_ids.size} from {candidate_ids.size} selected outlier GS"
    )
    if args.ring_mode == "none":
        roi_face_ids = seed_face_ids
    else:
        roi_face_ids = expand_faces_by_vertex_ring(
            faces=faces,
            num_vertices=num_vertices,
            seed_face_ids=seed_face_ids,
            ring_count=int(args.ring_count),
            max_roi_faces=int(args.max_roi_faces),
            random_seed=int(args.random_seed),
        )

    seed_face_mask = np.zeros((num_faces,), dtype=bool)
    seed_face_mask[seed_face_ids] = True
    roi_face_mask = np.zeros((num_faces,), dtype=bool)
    roi_face_mask[roi_face_ids] = True

    payload_out = {
        "mode": "meshfusion_roi_from_outliers_v0",
        "roi_face_ids": torch.from_numpy(roi_face_ids.astype(np.int64, copy=False)),
        "roi_face_mask": torch.from_numpy(roi_face_mask),
        "seed_face_ids": torch.from_numpy(seed_face_ids.astype(np.int64, copy=False)),
        "seed_face_mask": torch.from_numpy(seed_face_mask),
        "seed_face_outlier_count": torch.from_numpy(seed_counts.astype(np.int32, copy=False)),
        "candidate_ids": torch.from_numpy(candidate_ids.astype(np.int64, copy=False)),
        "candidate_mask": torch.from_numpy(candidate_mask.copy()),
    }
    payload_path = output_dir / "meshfusion_roi_faces_v0.pt"
    torch.save(payload_out, str(payload_path))

    roi_preview_path = output_dir / "meshfusion_roi_faces_v0.ply"
    seed_preview_path = output_dir / "meshfusion_seed_faces_v0.ply"
    export_face_subset(mesh_obj, roi_face_ids, roi_preview_path, max_faces=int(args.preview_max_faces), color=(255, 80, 40, 255))
    export_face_subset(mesh_obj, seed_face_ids, seed_preview_path, max_faces=int(args.preview_max_faces), color=(40, 180, 255, 255))

    surface_distance = tensor_to_numpy(payload["surface_distance"]) if "surface_distance" in payload else np.zeros((0,), dtype=np.float32)
    opacity = tensor_to_numpy(payload["opacity"]) if "opacity" in payload else np.zeros((0,), dtype=np.float32)
    visible_view_count = tensor_to_numpy(payload["visible_view_count"]) if "visible_view_count" in payload else np.zeros((0,), dtype=np.float32)
    summary = {
        "mode": "meshfusion_roi_from_outliers_v0",
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "candidate_payload": str(Path(args.candidate_payload).resolve()),
        "output_dir": str(output_dir.resolve()),
        "payload_path": str(payload_path.resolve()),
        "parameters": {
            "candidate_mask_key": args.candidate_mask_key,
            "min_outliers_per_seed_face": int(args.min_outliers_per_seed_face),
            "max_seed_faces": int(args.max_seed_faces),
            "ring_mode": args.ring_mode,
            "ring_count": int(args.ring_count),
            "max_roi_faces": int(args.max_roi_faces),
            "random_seed": int(args.random_seed),
            "preview_max_faces": int(args.preview_max_faces),
        },
        "counts": {
            "mesh_vertices": num_vertices,
            "mesh_faces": num_faces,
            "selected_outlier_gaussians": int(candidate_ids.size),
            "seed_face_count": int(seed_face_ids.size),
            "roi_face_count": int(roi_face_ids.size),
            "seed_face_ratio_vs_mesh": float(seed_face_ids.size / max(num_faces, 1)),
            "roi_face_ratio_vs_mesh": float(roi_face_ids.size / max(num_faces, 1)),
        },
        "stats": {
            "seed_outlier_count_per_face": stats_from_array(seed_counts.astype(np.float32)),
            "surface_distance_selected_outliers": stats_from_array(surface_distance[candidate_ids] if surface_distance.shape[0] else surface_distance),
            "opacity_selected_outliers": stats_from_array(opacity[candidate_ids] if opacity.shape[0] else opacity),
            "visible_view_count_selected_outliers": stats_from_array(
                visible_view_count[candidate_ids].astype(np.float32) if visible_view_count.shape[0] else visible_view_count
            ),
        },
        "previews": {
            "roi_faces_ply": str(roi_preview_path.resolve()),
            "seed_faces_ply": str(seed_preview_path.resolve()),
        },
    }
    summary_path = output_dir / "meshfusion_roi_faces_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved ROI payload to: {payload_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
