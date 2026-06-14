from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from utils.sof_mesh_patch_enhancer_v0 import (
    FEATURE_SCHEMA,
    build_hr_targets,
    compute_vertex_geometry,
    load_triangle_mesh,
    mesh_bbox_normalizer,
    stats_from_array,
    write_json,
)


def select_vertex_ids(count: int, stride: int, max_vertices: int, mode: str, seed: int) -> np.ndarray:
    ids = np.arange(int(count), dtype=np.int64)
    stride = max(int(stride), 1)
    if stride > 1:
        ids = ids[::stride]
    if int(max_vertices) > 0 and ids.size > int(max_vertices):
        if mode == "random":
            rng = np.random.default_rng(int(seed))
            ids = np.sort(rng.choice(ids, size=int(max_vertices), replace=False)).astype(np.int64, copy=False)
        else:
            ids = ids[: int(max_vertices)]
    return ids


def main():
    parser = ArgumentParser(
        description=(
            "Build SOF mesh patch v0 training data. "
            "The query/LR mesh is treated as a surface proposal; HR mesh only provides local residual targets."
        )
    )
    parser.add_argument("--lr_mesh_path", type=str, required=True)
    parser.add_argument("--hr_mesh_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--reference_samples", type=int, default=300000)
    parser.add_argument("--strong_ratio", type=float, default=0.002)
    parser.add_argument("--weak_ratio", type=float, default=0.006)
    parser.add_argument("--strong_threshold", type=float, default=0.0)
    parser.add_argument("--weak_threshold", type=float, default=0.0)
    parser.add_argument("--vertex_stride", type=int, default=1)
    parser.add_argument("--max_vertices", type=int, default=0)
    parser.add_argument("--sample_mode", choices=["prefix", "random"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lr_mesh = load_triangle_mesh(args.lr_mesh_path)
    hr_mesh = load_triangle_mesh(args.hr_mesh_path)
    bbox_center, bbox_diag = mesh_bbox_normalizer(lr_mesh)
    strong_threshold = float(args.strong_threshold) if float(args.strong_threshold) > 0.0 else float(args.strong_ratio) * bbox_diag
    weak_threshold = float(args.weak_threshold) if float(args.weak_threshold) > 0.0 else float(args.weak_ratio) * bbox_diag
    if weak_threshold < strong_threshold:
        raise ValueError("--weak_threshold/ratio must be >= --strong_threshold/ratio.")

    print(f"[sof-mesh-patch-dataset] LR mesh vertices={len(lr_mesh.vertices)} faces={len(lr_mesh.faces)}")
    print(f"[sof-mesh-patch-dataset] HR mesh vertices={len(hr_mesh.vertices)} faces={len(hr_mesh.faces)}")
    print(f"[sof-mesh-patch-dataset] bbox_diag={bbox_diag:.6f} strong={strong_threshold:.6f} weak={weak_threshold:.6f}")

    lr_geometry = compute_vertex_geometry(lr_mesh, bbox_center=bbox_center, bbox_diag=bbox_diag)
    targets = build_hr_targets(
        lr_geometry,
        hr_mesh=hr_mesh,
        reference_sample_count=int(args.reference_samples),
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        seed=int(args.seed),
    )
    vertex_ids = select_vertex_ids(
        count=lr_geometry["vertices"].shape[0],
        stride=int(args.vertex_stride),
        max_vertices=int(args.max_vertices),
        mode=args.sample_mode,
        seed=int(args.seed),
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "sof_mesh_patch_dataset_v0",
        "lr_mesh_path": str(Path(args.lr_mesh_path).resolve()),
        "hr_mesh_path": str(Path(args.hr_mesh_path).resolve()),
        "feature_schema": FEATURE_SCHEMA,
        "bbox_center": torch.from_numpy(bbox_center.astype(np.float32)),
        "bbox_diag": torch.tensor(float(bbox_diag), dtype=torch.float32),
        "strong_threshold": torch.tensor(float(strong_threshold), dtype=torch.float32),
        "weak_threshold": torch.tensor(float(weak_threshold), dtype=torch.float32),
        "vertex_ids": torch.from_numpy(vertex_ids.astype(np.int64, copy=False)),
        "features": torch.from_numpy(lr_geometry["features"][vertex_ids].astype(np.float32, copy=False)),
        "target_delta_local": torch.from_numpy(targets["target_delta_local"][vertex_ids].astype(np.float32, copy=False)),
        "target_confidence": torch.from_numpy(targets["target_confidence"][vertex_ids].astype(np.float32, copy=False)),
        "target_train_weight": torch.from_numpy(targets["target_train_weight"][vertex_ids].astype(np.float32, copy=False)),
        "target_distance": torch.from_numpy(targets["target_distance"][vertex_ids].astype(np.float32, copy=False)),
        "target_reliability": torch.from_numpy(targets["target_reliability"][vertex_ids].astype(np.uint8, copy=False)),
        "target_normal_alignment": torch.from_numpy(targets["target_normal_alignment"][vertex_ids].astype(np.float32, copy=False)),
        "all_vertex_distance": torch.from_numpy(targets["target_distance"].astype(np.float32, copy=False)),
        "all_vertex_reliability": torch.from_numpy(targets["target_reliability"].astype(np.uint8, copy=False)),
    }
    torch.save(payload, str(output_path))

    reliability = targets["target_reliability"][vertex_ids]
    summary = {
        "mode": "build_sof_mesh_patch_dataset_v0",
        "lr_mesh_path": str(Path(args.lr_mesh_path).resolve()),
        "hr_mesh_path": str(Path(args.hr_mesh_path).resolve()),
        "output_path": str(output_path.resolve()),
        "parameters": {
            "reference_samples": int(args.reference_samples),
            "strong_ratio": float(args.strong_ratio),
            "weak_ratio": float(args.weak_ratio),
            "strong_threshold": float(strong_threshold),
            "weak_threshold": float(weak_threshold),
            "vertex_stride": int(args.vertex_stride),
            "max_vertices": int(args.max_vertices),
            "sample_mode": args.sample_mode,
            "seed": int(args.seed),
        },
        "counts": {
            "lr_vertices": int(len(lr_mesh.vertices)),
            "lr_faces": int(len(lr_mesh.faces)),
            "hr_vertices": int(len(hr_mesh.vertices)),
            "hr_faces": int(len(hr_mesh.faces)),
            "training_vertices": int(vertex_ids.size),
            "strong_core": int(np.sum(reliability == 2)),
            "weak_support": int(np.sum(reliability == 1)),
            "unsupported": int(np.sum(reliability == 0)),
        },
        "stats": {
            "target_distance_all": stats_from_array(targets["target_distance"]),
            "target_distance_train": stats_from_array(targets["target_distance"][vertex_ids]),
            "target_train_weight": stats_from_array(targets["target_train_weight"][vertex_ids]),
            "target_normal_alignment": stats_from_array(targets["target_normal_alignment"][vertex_ids]),
        },
        "feature_schema": FEATURE_SCHEMA,
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"[sof-mesh-patch-dataset] saved dataset: {output_path}")
    print(f"[sof-mesh-patch-dataset] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
