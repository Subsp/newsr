#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814
SOURCE_ORIGINAL = 0
SOURCE_PRIOR_INJECTED = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create controlled 2DGS-posterior integration variants. "
            "front_offset only moves newborns along their recovered normal; handoff additionally "
            "transfers a capped optical-thickness budget from local donor parents."
        )
    )
    parser.add_argument("--input_model_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--mode", choices=["front_offset", "handoff"], default="handoff")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--front_offset", type=float, default=0.0)
    parser.add_argument("--score_power", type=float, default=0.70)
    parser.add_argument("--cluster_ref_views", type=float, default=2.0)
    parser.add_argument("--cluster_factor_max", type=float, default=1.5)
    parser.add_argument("--locality_power", type=float, default=0.50)
    parser.add_argument("--min_locality", type=float, default=0.03)
    parser.add_argument("--parent_tau_fraction_max", type=float, default=0.12)
    parser.add_argument("--parent_budget_scale", type=float, default=0.18)
    parser.add_argument("--newborn_tau_from_budget", type=float, default=1.0)
    parser.add_argument("--newborn_tau_scale", type=float, default=1.0)
    parser.add_argument("--newborn_alpha_floor", type=float, default=0.004)
    parser.add_argument("--newborn_alpha_max", type=float, default=0.10)
    parser.add_argument("--newborn_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--newborn_scale_max", type=float, default=0.012)
    return parser.parse_args()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _tau_from_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0 - 1e-6)
    return (-np.log(np.clip(1.0 - alpha, 1e-6, 1.0))).astype(np.float32)


def _alpha_from_tau(tau: np.ndarray) -> np.ndarray:
    tau = np.maximum(np.asarray(tau, dtype=np.float32), 0.0)
    return (1.0 - np.exp(-tau)).astype(np.float32)


def _copy_config(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json", "input.ply"):
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)


def _load_tags(point_dir: Path, total: int) -> Dict[str, torch.Tensor]:
    tags_path = point_dir / "gaussian_tags.pt"
    if tags_path.is_file():
        payload = torch.load(tags_path, map_location="cpu")
        source = payload.get("source_tag")
        if torch.is_tensor(source) and int(source.reshape(-1).numel()) >= int(total):
            return payload
    source_tag = torch.zeros((total,), dtype=torch.int32)
    return {"source_tag": source_tag}


def _infer_prior_ids(tags: Dict[str, torch.Tensor], total: int, metadata_count: int) -> Tuple[np.ndarray, int]:
    source = tags.get("source_tag")
    prior_ids: np.ndarray
    if torch.is_tensor(source):
        source = source.reshape(-1)
        if int(source.numel()) >= total:
            source_np = source[:total].cpu().numpy().astype(np.int64)
            prior_ids = np.flatnonzero(source_np == SOURCE_PRIOR_INJECTED).astype(np.int64)
        else:
            prior_ids = np.empty((0,), dtype=np.int64)
    else:
        prior_ids = np.empty((0,), dtype=np.int64)

    if int(prior_ids.shape[0]) != int(metadata_count):
        prior_ids = np.arange(total - metadata_count, total, dtype=np.int64)
    base_count = int(total - prior_ids.shape[0])
    return prior_ids, base_count


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[:, 0, 1] = 2.0 * (x * y - z * w)
    out[:, 0, 2] = 2.0 * (x * z + y * w)
    out[:, 1, 0] = 2.0 * (x * y + z * w)
    out[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[:, 1, 2] = 2.0 * (y * z - x * w)
    out[:, 2, 0] = 2.0 * (x * z - y * w)
    out[:, 2, 1] = 2.0 * (y * z + x * w)
    out[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def _scale_fields(vertices: np.ndarray) -> np.ndarray:
    return np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)


def _rotation_fields(vertices: np.ndarray) -> np.ndarray:
    return np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float32)


def _xyz_fields(vertices: np.ndarray) -> np.ndarray:
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)


def _write_xyz(vertices: np.ndarray, ids: np.ndarray, xyz: np.ndarray) -> None:
    vertices["x"][ids] = xyz[:, 0]
    vertices["y"][ids] = xyz[:, 1]
    vertices["z"][ids] = xyz[:, 2]


def _stats(value: np.ndarray) -> Dict[str, float]:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(value)),
        "p50": float(np.percentile(value, 50.0)),
        "p90": float(np.percentile(value, 90.0)),
        "max": float(np.max(value)),
    }


def _prepare_scores(metadata: Dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    score = np.asarray(metadata.get("score", np.ones((1,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    score = np.clip(score, 0.0, 1.0)
    cluster_size = np.asarray(metadata.get("cluster_size", np.ones_like(score)), dtype=np.float32).reshape(-1)
    cluster_factor = np.sqrt(np.maximum(cluster_size, 1.0) / max(float(args.cluster_ref_views), 1e-6))
    cluster_factor = np.clip(cluster_factor, 0.0, float(args.cluster_factor_max))
    return (np.power(score, float(args.score_power)) * cluster_factor).astype(np.float32)


def _apply_front_offset(vertices: np.ndarray, prior_ids: np.ndarray, offset: float) -> np.ndarray:
    if abs(float(offset)) <= 0.0 or prior_ids.size == 0:
        return np.zeros((prior_ids.size, 3), dtype=np.float32)
    xyz = _xyz_fields(vertices)[prior_ids]
    rot = _rotation_fields(vertices)[prior_ids]
    normal = _quat_wxyz_to_matrix(rot)[:, :, 2]
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    delta = normal.astype(np.float32) * float(offset)
    _write_xyz(vertices, prior_ids, xyz + delta)
    return delta


def _apply_newborn_scale(vertices: np.ndarray, prior_ids: np.ndarray, multiplier: float, scale_max: float) -> None:
    if float(multiplier) == 1.0 or prior_ids.size == 0:
        return
    scale = np.exp(_scale_fields(vertices)[prior_ids])
    scale = np.minimum(scale * float(multiplier), float(scale_max))
    for dim in range(3):
        vertices[f"scale_{dim}"][prior_ids] = np.log(np.maximum(scale[:, dim], 1e-8)).astype(np.float32)


def _apply_handoff(
    vertices: np.ndarray,
    prior_ids: np.ndarray,
    metadata: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, object]:
    parent = np.asarray(metadata["parent_index"], dtype=np.int64).reshape(-1)
    if int(parent.shape[0]) != int(prior_ids.shape[0]):
        raise ValueError(
            f"metadata parent_index length mismatch: {int(parent.shape[0])} vs prior ids {int(prior_ids.shape[0])}"
        )
    valid = (parent >= 0) & (parent < vertices.shape[0])
    if not np.any(valid):
        raise RuntimeError("No valid parent_index entries for handoff.")

    strength = _prepare_scores(metadata, args)
    parent_scale = np.exp(_scale_fields(vertices)[np.clip(parent, 0, vertices.shape[0] - 1)])
    newborn_scale = np.exp(_scale_fields(vertices)[prior_ids])
    parent_area = np.maximum(parent_scale[:, 0] * parent_scale[:, 1], 1e-10)
    newborn_area = np.maximum(newborn_scale[:, 0] * newborn_scale[:, 1], 1e-10)
    locality = np.clip(np.power(np.minimum(newborn_area / parent_area, 1.0), float(args.locality_power)), 0.0, 1.0)
    locality = np.where(locality >= float(args.min_locality), locality, 0.0).astype(np.float32)
    child_budget = (strength * locality * valid.astype(np.float32)).astype(np.float32)

    unique_parent, inverse = np.unique(parent[valid], return_inverse=True)
    parent_sum = np.zeros((unique_parent.shape[0],), dtype=np.float32)
    np.add.at(parent_sum, inverse, child_budget[valid])
    parent_rho = np.clip(parent_sum * float(args.parent_budget_scale), 0.0, float(args.parent_tau_fraction_max))

    parent_alpha_before = _sigmoid(vertices["opacity"][unique_parent])
    parent_tau_before = _tau_from_alpha(parent_alpha_before)
    parent_delta_tau = parent_tau_before * parent_rho
    parent_tau_after = np.maximum(parent_tau_before - parent_delta_tau, 0.0)
    parent_alpha_after = _alpha_from_tau(parent_tau_after)
    vertices["opacity"][unique_parent] = _logit(parent_alpha_after)

    child_delta_tau = np.zeros((prior_ids.shape[0],), dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    denom = np.maximum(parent_sum[inverse], 1e-8)
    child_share = child_budget[valid] / denom
    child_delta_tau[valid_indices] = parent_delta_tau[inverse] * child_share * float(args.newborn_tau_from_budget)

    newborn_alpha_before = _sigmoid(vertices["opacity"][prior_ids])
    newborn_tau_before = _tau_from_alpha(newborn_alpha_before)
    newborn_tau_after = np.maximum(newborn_tau_before * float(args.newborn_tau_scale), child_delta_tau)
    newborn_alpha_after = _alpha_from_tau(newborn_tau_after)
    newborn_alpha_after = np.clip(
        np.maximum(newborn_alpha_after, float(args.newborn_alpha_floor)),
        1e-6,
        float(args.newborn_alpha_max),
    )
    vertices["opacity"][prior_ids] = _logit(newborn_alpha_after)

    return {
        "valid_parent_links": int(np.count_nonzero(valid)),
        "unique_donor_parents": int(unique_parent.shape[0]),
        "child_budget": _stats(child_budget),
        "locality": _stats(locality),
        "parent_rho": _stats(parent_rho),
        "parent_alpha_before": _stats(parent_alpha_before),
        "parent_alpha_after": _stats(parent_alpha_after),
        "parent_delta_tau": _stats(parent_delta_tau),
        "newborn_alpha_before": _stats(newborn_alpha_before),
        "newborn_alpha_after": _stats(newborn_alpha_after),
        "newborn_delta_tau_from_parent": _stats(child_delta_tau),
    }


def main() -> None:
    args = _parse_args()
    input_model_dir = Path(args.input_model_dir).expanduser().resolve()
    output_model_dir = Path(args.output_model_dir).expanduser().resolve()
    iteration = int(args.iteration)
    input_point_dir = input_model_dir / "point_cloud" / f"iteration_{iteration}"
    input_ply = input_point_dir / "point_cloud.ply"
    if not input_ply.is_file():
        raise FileNotFoundError(f"Input PLY not found: {input_ply}")
    metadata_path = (
        Path(args.metadata_path).expanduser().resolve()
        if str(args.metadata_path).strip()
        else input_point_dir / "sprayed_2dgs_posterior_metadata_v0.npz"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Posterior metadata not found: {metadata_path}")
    if output_model_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists; pass --overwrite: {output_model_dir}")
    if output_model_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_model_dir)

    ply = PlyData.read(str(input_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    metadata_npz = np.load(metadata_path)
    metadata = {key: metadata_npz[key] for key in metadata_npz.files}
    new_count = int(np.asarray(metadata["xyz"]).shape[0])
    tags = _load_tags(input_point_dir, int(vertices.shape[0]))
    prior_ids, base_count = _infer_prior_ids(tags, int(vertices.shape[0]), new_count)
    if int(prior_ids.shape[0]) <= 0:
        raise RuntimeError("No prior_injected/newborn gaussians were found.")

    offset_delta = _apply_front_offset(vertices, prior_ids, float(args.front_offset))
    _apply_newborn_scale(vertices, prior_ids, float(args.newborn_scale_multiplier), float(args.newborn_scale_max))
    handoff_summary: Dict[str, object] = {}
    if str(args.mode) == "handoff":
        handoff_summary = _apply_handoff(vertices, prior_ids, metadata, args)

    _copy_config(input_model_dir, output_model_dir)
    output_point_dir = output_model_dir / "point_cloud" / f"iteration_{iteration}"
    output_point_dir.mkdir(parents=True, exist_ok=True)
    output_ply = output_point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(output_ply))
    input_tags = input_point_dir / "gaussian_tags.pt"
    if input_tags.is_file():
        shutil.copy2(input_tags, output_point_dir / "gaussian_tags.pt")
    shutil.copy2(metadata_path, output_point_dir / metadata_path.name)

    summary = {
        "version": "integrate_2dgs_posterior_handoff_v0",
        "mode": str(args.mode),
        "input_model_dir": str(input_model_dir),
        "output_model_dir": str(output_model_dir),
        "iteration": iteration,
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "metadata_path": str(metadata_path),
        "total_gaussians": int(vertices.shape[0]),
        "base_gaussians": int(base_count),
        "newborn_gaussians": int(prior_ids.shape[0]),
        "params": {
            "front_offset": float(args.front_offset),
            "score_power": float(args.score_power),
            "cluster_ref_views": float(args.cluster_ref_views),
            "cluster_factor_max": float(args.cluster_factor_max),
            "locality_power": float(args.locality_power),
            "min_locality": float(args.min_locality),
            "parent_tau_fraction_max": float(args.parent_tau_fraction_max),
            "parent_budget_scale": float(args.parent_budget_scale),
            "newborn_tau_from_budget": float(args.newborn_tau_from_budget),
            "newborn_tau_scale": float(args.newborn_tau_scale),
            "newborn_alpha_floor": float(args.newborn_alpha_floor),
            "newborn_alpha_max": float(args.newborn_alpha_max),
            "newborn_scale_multiplier": float(args.newborn_scale_multiplier),
            "newborn_scale_max": float(args.newborn_scale_max),
        },
        "front_offset_stats": {
            "offset_norm": _stats(np.linalg.norm(offset_delta, axis=1) if offset_delta.size else np.zeros((0,))),
        },
        "handoff": handoff_summary,
    }
    summary_path = output_model_dir / "integrate_2dgs_posterior_handoff_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
