#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SR-HF 3D curve tracks into skinny Gaussian carriers. "
            "This consumes build_sr_hf_curve_tracks_v0 output and writes a newborn-only model "
            "for the existing Gaussian PLY merge path."
        )
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--track_payload", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--newborn_model_dir", default="")
    parser.add_argument("--selection", default="keep", choices=["keep", "strong", "all"])
    parser.add_argument("--fallback_to_keep", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_tracks", type=int, default=0)
    parser.add_argument("--max_total_newborn", type=int, default=0)
    parser.add_argument("--sample_spacing_px", type=float, default=4.0)
    parser.add_argument("--sample_spacing_min", type=float, default=0.003)
    parser.add_argument("--sample_spacing_max", type=float, default=0.020)
    parser.add_argument("--max_samples_per_track", type=int, default=12)
    parser.add_argument("--scale_long_factor", type=float, default=0.75)
    parser.add_argument("--scale_short_px", type=float, default=0.55)
    parser.add_argument(
        "--scale_short_width_factor",
        type=float,
        default=1.0,
        help="Multiply track-estimated image-space half width before converting it to 3D short-axis scale.",
    )
    parser.add_argument("--scale_normal_px", type=float, default=0.35)
    parser.add_argument("--scale_min", type=float, default=4e-4)
    parser.add_argument("--scale_max", type=float, default=1.5e-2)
    parser.add_argument("--opacity_floor", type=float, default=0.015)
    parser.add_argument("--opacity_scale", type=float, default=0.10)
    parser.add_argument("--opacity_power", type=float, default=0.75)
    parser.add_argument("--opacity_min", type=float, default=0.008)
    parser.add_argument("--opacity_max", type=float, default=0.12)
    parser.add_argument("--color_gain", type=float, default=1.0)
    parser.add_argument("--jitter_perp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--write_cpu_merged_preview", action="store_true")
    return parser.parse_args()


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _fallback_perp(direction: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    direction = _normalize_vec(direction.astype(np.float32))
    normal = normal.astype(np.float32)
    normal = normal - np.sum(normal * direction, axis=1, keepdims=True) * direction
    bad = np.linalg.norm(normal, axis=1) < 1e-6
    if np.any(bad):
        candidate = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (direction.shape[0], 1))
        parallel = np.abs(np.sum(candidate * direction, axis=1)) > 0.9
        candidate[parallel] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        normal[bad] = candidate[bad] - np.sum(candidate[bad] * direction[bad], axis=1, keepdims=True) * direction[bad]
    normal = _normalize_vec(normal)
    short_axis = np.cross(normal, direction)
    short_axis = _normalize_vec(short_axis.astype(np.float32))
    normal = _normalize_vec(np.cross(direction, short_axis).astype(np.float32))
    return short_axis, normal


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    quats = np.zeros((matrix.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(matrix.astype(np.float64)):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        else:
            diag = np.diag(m)
            if diag[0] > diag[1] and diag[0] > diag[2]:
                s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
                qw = (m[2, 1] - m[1, 2]) / s
                qx = 0.25 * s
                qy = (m[0, 1] + m[1, 0]) / s
                qz = (m[0, 2] + m[2, 0]) / s
            elif diag[1] > diag[2]:
                s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
                qw = (m[0, 2] - m[2, 0]) / s
                qx = (m[0, 1] + m[1, 0]) / s
                qy = 0.25 * s
                qz = (m[1, 2] + m[2, 1]) / s
            else:
                s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
                qw = (m[1, 0] - m[0, 1]) / s
                qx = (m[0, 2] + m[2, 0]) / s
                qy = (m[1, 2] + m[2, 1]) / s
                qz = 0.25 * s
        q = np.asarray([qw, qx, qy, qz], dtype=np.float32)
        quats[i] = q / max(float(np.linalg.norm(q)), 1e-8)
    return quats


def _select_tracks(data: Dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    n = int(data["p0"].shape[0])
    if str(args.selection) == "all":
        mask = np.ones((n,), dtype=bool)
    else:
        key = str(args.selection)
        if key not in data:
            if bool(args.fallback_to_keep) and "keep" in data:
                mask = np.asarray(data["keep"], dtype=bool)
            else:
                raise KeyError(f"Track payload lacks selection mask: {key}")
        else:
            mask = np.asarray(data[key], dtype=bool)
    ids = np.flatnonzero(mask)
    if ids.size == 0 and bool(args.fallback_to_keep) and "keep" in data:
        ids = np.flatnonzero(np.asarray(data["keep"], dtype=bool))
    if ids.size == 0:
        raise RuntimeError(f"No tracks selected by {args.selection}")
    rank = np.asarray(data.get("score_max", np.ones((n,), dtype=np.float32)), dtype=np.float32)[ids]
    view_count = np.asarray(data.get("view_count", np.ones((n,), dtype=np.int32)), dtype=np.float32)[ids]
    order = np.argsort(rank * np.maximum(view_count, 1.0))[::-1]
    ids = ids[order]
    if int(args.max_tracks) > 0:
        ids = ids[: int(args.max_tracks)]
    return ids.astype(np.int64)


def _sample_tracks(data: Dict[str, np.ndarray], track_ids: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(args.seed))
    num_tracks = int(data["p0"].shape[0])
    pixel_world_arr = np.asarray(
        data.get("pixel_world_mean", np.full((num_tracks,), 0.002, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    score_arr = np.asarray(
        data.get("score_max", np.ones((num_tracks,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    width_px_arr = np.asarray(
        data.get("width_px_mean", np.full((num_tracks,), float(args.scale_short_px), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    view_count_arr = np.asarray(
        data.get("view_count", np.ones((num_tracks,), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    chunks: Dict[str, list[np.ndarray]] = {
        "xyz": [],
        "color": [],
        "opacity": [],
        "scale": [],
        "rotation": [],
        "track_index": [],
        "track_t": [],
        "track_score": [],
        "track_view_count": [],
        "track_width_px": [],
    }
    total = 0
    for tid in track_ids.tolist():
        p0 = data["p0"][tid].astype(np.float32)
        p1 = data["p1"][tid].astype(np.float32)
        direction = data["direction"][tid].astype(np.float32)
        normal = data["normal"][tid].astype(np.float32)
        length = max(float(np.linalg.norm(p1 - p0)), 1e-8)
        pix_world = max(float(pixel_world_arr[tid]), 1e-6)
        width_px = max(float(width_px_arr[tid]), float(args.scale_short_px))
        spacing = np.clip(float(args.sample_spacing_px) * pix_world, float(args.sample_spacing_min), float(args.sample_spacing_max))
        samples = max(1, int(math.ceil(length / max(spacing, 1e-8))))
        samples = min(samples, int(args.max_samples_per_track))
        remaining = int(args.max_total_newborn) - total if int(args.max_total_newborn) > 0 else None
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None:
            samples = min(samples, remaining)
        if samples <= 0:
            continue
        if samples == 1:
            t = np.asarray([0.5], dtype=np.float32)
        else:
            t = (np.arange(samples, dtype=np.float32) + 0.5) / float(samples)
        xyz = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
        direction_batch = np.tile(direction[None, :], (samples, 1)).astype(np.float32)
        normal_batch = np.tile(normal[None, :], (samples, 1)).astype(np.float32)
        short_axis, normal_axis = _fallback_perp(direction_batch, normal_batch)
        if float(args.jitter_perp) > 0:
            jitter = rng.normal(size=(samples, 1)).astype(np.float32) * float(args.jitter_perp) * pix_world
            xyz = xyz + short_axis * jitter
        rot = _matrix_to_quaternion_wxyz(np.stack([direction_batch, short_axis, normal_axis], axis=2))
        scale_long = np.full((samples,), spacing * float(args.scale_long_factor), dtype=np.float32)
        scale_short = np.full((samples,), width_px * float(args.scale_short_width_factor) * pix_world, dtype=np.float32)
        scale_normal = np.full((samples,), float(args.scale_normal_px) * pix_world, dtype=np.float32)
        scale = np.stack([scale_long, scale_short, scale_normal], axis=1)
        scale = np.clip(scale, float(args.scale_min), float(args.scale_max)).astype(np.float32)
        score = float(score_arr[tid])
        opacity = float(args.opacity_floor) + float(args.opacity_scale) * (max(score, 0.0) ** float(args.opacity_power))
        opacity = float(np.clip(opacity, float(args.opacity_min), float(args.opacity_max)))
        color = np.clip(data["color"][tid].astype(np.float32) * float(args.color_gain), 0.0, 1.0)
        chunks["xyz"].append(xyz.astype(np.float32))
        chunks["color"].append(np.tile(color[None, :], (samples, 1)).astype(np.float32))
        chunks["opacity"].append(np.full((samples, 1), opacity, dtype=np.float32))
        chunks["scale"].append(scale)
        chunks["rotation"].append(rot)
        chunks["track_index"].append(np.full((samples,), int(tid), dtype=np.int64))
        chunks["track_t"].append(t.astype(np.float32))
        chunks["track_score"].append(np.full((samples,), score, dtype=np.float32))
        chunks["track_view_count"].append(np.full((samples,), int(view_count_arr[tid]), dtype=np.int32))
        chunks["track_width_px"].append(np.full((samples,), width_px, dtype=np.float32))
        total += samples
    if total <= 0:
        raise RuntimeError("Selected tracks produced zero newborn Gaussians.")
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def main() -> None:
    args = _parse_args()
    from plyfile import PlyData, PlyElement
    from spray_2dgs_hf_carrier_to_3d_v0 import _copy_config, _fill_new_vertices, _make_tracking

    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    base_iter = int(args.base_iteration)
    base_point_dir = base_model_dir / "point_cloud" / f"iteration_{base_iter}"
    base_ply = base_point_dir / "point_cloud.ply"
    if not base_ply.is_file():
        raise FileNotFoundError(f"Base PLY not found: {base_ply}")
    track_payload = Path(args.track_payload).expanduser().resolve()
    data = dict(np.load(track_payload))
    track_ids = _select_tracks(data, args)
    newborn = _sample_tracks(data, track_ids, args)

    output_model_dir = Path(args.output_model_dir).expanduser().resolve()
    newborn_model_dir = Path(args.newborn_model_dir).expanduser().resolve() if str(args.newborn_model_dir) else output_model_dir / "newborn_only_model"
    for candidate in (output_model_dir, newborn_model_dir):
        if candidate.exists() and not bool(args.overwrite):
            raise FileExistsError(f"Output exists; pass --overwrite: {candidate}")
    if output_model_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_model_dir)
    if newborn_model_dir.exists() and bool(args.overwrite):
        shutil.rmtree(newborn_model_dir)

    base_vertices = PlyData.read(str(base_ply))["vertex"].data
    new_vertices = _fill_new_vertices(base_vertices.dtype, newborn)
    _copy_config(base_model_dir, newborn_model_dir)
    newborn_point_dir = newborn_model_dir / "point_cloud" / f"iteration_{base_iter}"
    newborn_point_dir.mkdir(parents=True, exist_ok=True)
    newborn_ply = newborn_point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(new_vertices, "vertex")]).write(str(newborn_ply))
    newborn_tags = _make_tracking(0, int(new_vertices.shape[0]), Path("__missing__"), base_iter)
    newborn_tags_path = newborn_point_dir / "gaussian_tags.pt"
    import torch

    torch.save(newborn_tags, newborn_tags_path)
    metadata_path = newborn_point_dir / "sprayed_sr_hf_curve_tracks_metadata_v0.npz"
    np.savez_compressed(metadata_path, **newborn)

    output_point_dir = output_model_dir / "point_cloud" / f"iteration_{base_iter}"
    output_point_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, output_point_dir / metadata_path.name)

    cpu_preview_ply = None
    cpu_preview_tags = None
    if bool(args.write_cpu_merged_preview):
        merged = np.empty((base_vertices.shape[0] + new_vertices.shape[0],), dtype=base_vertices.dtype)
        merged[: base_vertices.shape[0]] = base_vertices
        merged[base_vertices.shape[0] :] = new_vertices
        preview_point_dir = output_model_dir / "cpu_merged_preview" / "point_cloud" / f"iteration_{base_iter}"
        preview_point_dir.mkdir(parents=True, exist_ok=True)
        cpu_preview_ply = preview_point_dir / "point_cloud.ply"
        PlyData([PlyElement.describe(merged, "vertex")]).write(str(cpu_preview_ply))
        tags = _make_tracking(int(base_vertices.shape[0]), int(new_vertices.shape[0]), base_point_dir / "gaussian_tags.pt", base_iter)
        cpu_preview_tags = preview_point_dir / "gaussian_tags.pt"
        torch.save(tags, cpu_preview_tags)

    output_model_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "spray_sr_hf_curve_tracks_to_gaussian_layer_v0",
        "base_model_dir": str(base_model_dir),
        "base_iteration": base_iter,
        "base_ply": str(base_ply),
        "track_payload": str(track_payload),
        "selection": str(args.selection),
        "selected_tracks": int(track_ids.size),
        "newborn_gaussians": int(new_vertices.shape[0]),
        "output_model_dir": str(output_model_dir),
        "newborn_model_dir": str(newborn_model_dir),
        "newborn_ply": str(newborn_ply),
        "newborn_tags": str(newborn_tags_path),
        "newborn_metadata": str(metadata_path),
        "cpu_preview_ply": None if cpu_preview_ply is None else str(cpu_preview_ply),
        "cpu_preview_tags": None if cpu_preview_tags is None else str(cpu_preview_tags),
        "params": {
            "sample_spacing_px": float(args.sample_spacing_px),
            "sample_spacing_min": float(args.sample_spacing_min),
            "sample_spacing_max": float(args.sample_spacing_max),
            "max_samples_per_track": int(args.max_samples_per_track),
            "scale_long_factor": float(args.scale_long_factor),
            "scale_short_px": float(args.scale_short_px),
            "scale_short_width_factor": float(args.scale_short_width_factor),
            "scale_normal_px": float(args.scale_normal_px),
            "opacity_floor": float(args.opacity_floor),
            "opacity_scale": float(args.opacity_scale),
            "opacity_min": float(args.opacity_min),
            "opacity_max": float(args.opacity_max),
            "color_gain": float(args.color_gain),
        },
        "newborn_stats": {
            "opacity_mean": float(newborn["opacity"].mean()),
            "scale_mean": [float(x) for x in newborn["scale"].mean(axis=0)],
            "scale_p90": [float(x) for x in np.percentile(newborn["scale"], 90, axis=0)],
            "color_mean": [float(x) for x in newborn["color"].mean(axis=0)],
            "track_score_mean": float(newborn["track_score"].mean()),
            "track_view_count_mean": float(newborn["track_view_count"].mean()),
            "track_width_px_mean": float(newborn["track_width_px"].mean()),
        },
    }
    summary_path = output_model_dir / "spray_sr_hf_curve_tracks_to_gaussian_layer_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected_tracks": summary["selected_tracks"], "newborn_gaussians": summary["newborn_gaussians"], "newborn_ply": summary["newborn_ply"]}, indent=2))
    print(f"[spray-curve-tracks-v0] summary: {summary_path}")


if __name__ == "__main__":
    main()
