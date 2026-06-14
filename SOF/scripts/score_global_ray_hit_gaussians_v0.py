#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import trimesh
from plyfile import PlyData, PlyElement
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from joint_judge_mip_sof_v0 import static_gaussian_metrics, stats_from_array
from train_mip_to_sof_surface_v0 import load_cameras_for_split, load_model_ply, resolve_iteration, select_uniform


def inverse_sigmoid_np(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32, copy=False)


def rgb_to_sh_np(rgb: np.ndarray) -> np.ndarray:
    return ((rgb.astype(np.float32, copy=False) - 0.5) / 0.28209479177387814).astype(np.float32, copy=False)


def select_top_ids(mask: np.ndarray, score: np.ndarray, max_points: int) -> np.ndarray:
    ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if int(max_points) > 0 and ids.size > int(max_points):
        order = np.argsort(-score[ids], kind="stable")[: int(max_points)]
        ids = ids[order]
    return ids


def write_point_cloud(path: Path, xyz: np.ndarray, score: np.ndarray, max_points: int) -> None:
    if xyz.shape[0] == 0:
        return
    ids = select_top_ids(np.ones((xyz.shape[0],), dtype=bool), score, int(max_points))
    score_sel = score[ids]
    denom = max(float(np.percentile(score_sel, 99.0)), 1e-6)
    heat = np.clip(score_sel / denom, 0.0, 1.0)
    colors = np.stack(
        [
            np.full_like(heat, 255.0),
            160.0 * (1.0 - heat),
            30.0 * (1.0 - heat),
            np.full_like(heat, 255.0),
        ],
        axis=1,
    ).astype(np.uint8)
    trimesh.points.PointCloud(xyz[ids], colors=colors).export(path)


def write_gaussian_subset_model(
    *,
    source_ply_path: Path,
    source_tags_path: Path,
    output_model_path: Path,
    iteration: int,
    mask: np.ndarray,
    score: np.ndarray,
    max_points: int,
    debug_visible: bool,
    debug_alpha: float,
    debug_scale_multiplier: float,
) -> Dict[str, object]:
    ids = select_top_ids(mask, score, int(max_points))
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_model_path / "selected_source_idx.pt"
    torch.save(torch.from_numpy(ids.copy()), selected_path)
    if ids.size == 0:
        return {
            "model_path": str(output_model_path),
            "ply": "",
            "selected_source_idx": str(selected_path),
            "count": 0,
            "debug_visible": bool(debug_visible),
        }

    plydata = PlyData.read(str(source_ply_path))
    vertex = plydata["vertex"].data
    subset = vertex[ids].copy()
    property_names = set(subset.dtype.names or ())
    if debug_visible:
        if "opacity" in property_names:
            subset["opacity"] = inverse_sigmoid_np(np.full((ids.size,), float(debug_alpha), dtype=np.float32))
        scale_names = sorted([name for name in property_names if name.startswith("scale_")], key=lambda x: int(x.split("_")[-1]))
        if scale_names:
            scale_delta = float(np.log(max(float(debug_scale_multiplier), 1e-6)))
            for name in scale_names:
                subset[name] = subset[name] + scale_delta
        dc_names = sorted([name for name in property_names if name.startswith("f_dc_")], key=lambda x: int(x.split("_")[-1]))
        if len(dc_names) >= 3:
            debug_sh = rgb_to_sh_np(np.asarray([1.0, 0.08, 0.02], dtype=np.float32))
            for channel, name in enumerate(dc_names[:3]):
                subset[name] = debug_sh[channel]

    ply_path = point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(subset, "vertex")], text=plydata.text).write(str(ply_path))
    if source_tags_path.is_file():
        tag_payload = torch.load(source_tags_path, map_location="cpu")
        if isinstance(tag_payload, dict):
            filtered_tags = {}
            for key, value in tag_payload.items():
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(vertex.shape[0]):
                    filtered_tags[key] = value[torch.from_numpy(ids.astype(np.int64, copy=False))]
                else:
                    filtered_tags[key] = value
            torch.save(filtered_tags, point_dir / "gaussian_tags.pt")
    return {
        "model_path": str(output_model_path),
        "ply": str(ply_path),
        "selected_source_idx": str(selected_path),
        "count": int(ids.size),
        "debug_visible": bool(debug_visible),
    }


def tracking_array(gaussians, name: str, dtype: np.dtype, default: int) -> np.ndarray:
    value = getattr(gaussians, name, None)
    total = int(gaussians.get_xyz.shape[0])
    if torch.is_tensor(value) and int(value.shape[0]) == total:
        return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.full((total,), default, dtype=dtype)


def camera_name(camera: object, fallback_idx: int) -> str:
    for key in ("image_name", "image_path", "uid"):
        value = getattr(camera, key, None)
        if value is not None:
            return str(value)
    return f"view_{int(fallback_idx):05d}"


def build_top_records(
    mask: np.ndarray,
    *,
    hit_count: np.ndarray,
    opacity: np.ndarray,
    anisotropy: np.ndarray,
    scale_max: np.ndarray,
    scale_min: np.ndarray,
    source_tag: np.ndarray,
    generation: np.ndarray,
    limit: int,
) -> List[Dict[str, float | int]]:
    ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if ids.size == 0:
        return []
    score = opacity[ids] * np.maximum(anisotropy[ids], 1.0)
    order = np.argsort(-score, kind="stable")[: max(int(limit), 0)]
    out: List[Dict[str, float | int]] = []
    for gid in ids[order].tolist():
        out.append(
            {
                "gaussian_id": int(gid),
                "hit_count": int(hit_count[gid]),
                "opacity": float(opacity[gid]),
                "anisotropy": float(anisotropy[gid]),
                "scale_max": float(scale_max[gid]),
                "scale_min": float(scale_min[gid]),
                "source_tag": int(source_tag[gid]),
                "generation": int(generation[gid]),
            }
        )
    return out


def count_by_int(values: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    ids = values[mask]
    if ids.size == 0:
        return {}
    uniq, counts = np.unique(ids.astype(np.int64, copy=False), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Traverse global views and find gaussians never hit by any rendered ray.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--low_hit_max_views", type=int, default=2)
    parser.add_argument("--rod_anisotropy_threshold", type=float, default=10.0)
    parser.add_argument("--preview_max_points", type=int, default=50000)
    parser.add_argument("--top_records", type=int, default=256)
    parser.add_argument("--no_gaussian_subset_export", action="store_true")
    parser.add_argument("--debug_visible_alpha", type=float, default=0.65)
    parser.add_argument("--debug_scale_multiplier", type=float, default=2.5)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    iteration = resolve_iteration(model_path, int(args.iteration))
    source_ply_path = model_path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    source_tags_path = source_ply_path.parent / "gaussian_tags.pt"
    gaussians = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    selected_cameras = select_uniform(cameras, int(args.max_views))

    total = int(gaussians.get_xyz.shape[0])
    hit_count = np.zeros((total,), dtype=np.int32)
    radius_sum = np.zeros((total,), dtype=np.float64)
    radius_max = np.zeros((total,), dtype=np.float32)
    first_hit_view = np.full((total,), -1, dtype=np.int32)

    view_names: List[str] = []
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(args.white_background) else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    for view_idx, camera in enumerate(tqdm(selected_cameras, desc="global ray hit sweep")):
        view_names.append(camera_name(camera, view_idx))
        pkg = render_simple(camera, gaussians, background)
        visible = pkg["visibility_filter"].detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
        if not np.any(visible):
            continue
        radii = pkg["radii"].detach().float().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        visible_ids = np.flatnonzero(visible).astype(np.int64, copy=False)
        valid_radius = np.maximum(radii[visible_ids], 0.0)
        hit_count[visible_ids] += 1
        radius_sum[visible_ids] += valid_radius.astype(np.float64)
        radius_max[visible_ids] = np.maximum(radius_max[visible_ids], valid_radius)
        unseen = first_hit_view[visible_ids] < 0
        if np.any(unseen):
            first_hit_view[visible_ids[unseen]] = int(view_idx)

    mean_radius = np.divide(
        radius_sum,
        np.maximum(hit_count, 1),
        out=np.zeros_like(radius_sum, dtype=np.float64),
        where=hit_count > 0,
    ).astype(np.float32)

    static = static_gaussian_metrics(gaussians)
    source_tag = tracking_array(gaussians, "_source_tag", np.int32, 0)
    generation = tracking_array(gaussians, "_generation", np.int32, 0)
    edge_touched = tracking_array(gaussians, "_edge_touched", np.bool_, False)

    rodlike_mask = static["anisotropy"] >= float(args.rod_anisotropy_threshold)
    never_hit_mask = hit_count == 0
    low_hit_mask = (hit_count > 0) & (hit_count <= max(int(args.low_hit_max_views), 0))
    never_hit_rodlike_mask = never_hit_mask & rodlike_mask
    low_hit_rodlike_mask = low_hit_mask & rodlike_mask

    payload = {
        "version": "global_ray_hit_gaussians_v0",
        "hit_count": torch.from_numpy(hit_count.copy()),
        "first_hit_view": torch.from_numpy(first_hit_view.copy()),
        "mean_radius": torch.from_numpy(mean_radius.copy()),
        "max_radius": torch.from_numpy(radius_max.copy()),
        "opacity": torch.from_numpy(static["opacity"].copy()),
        "scale_max": torch.from_numpy(static["scale_max"].copy()),
        "scale_min": torch.from_numpy(static["scale_min"].copy()),
        "anisotropy": torch.from_numpy(static["anisotropy"].copy()),
        "source_tag": torch.from_numpy(source_tag.copy()),
        "generation": torch.from_numpy(generation.copy()),
        "edge_touched": torch.from_numpy(edge_touched.astype(np.bool_, copy=False)),
        "rodlike_mask": torch.from_numpy(rodlike_mask.astype(bool, copy=False))[:, None],
        "never_hit_mask": torch.from_numpy(never_hit_mask.astype(bool, copy=False))[:, None],
        "low_hit_mask": torch.from_numpy(low_hit_mask.astype(bool, copy=False))[:, None],
        "never_hit_rodlike_mask": torch.from_numpy(never_hit_rodlike_mask.astype(bool, copy=False))[:, None],
        "low_hit_rodlike_mask": torch.from_numpy(low_hit_rodlike_mask.astype(bool, copy=False))[:, None],
    }
    payload_path = output_root / "global_ray_hit_gaussians_v0.pt"
    torch.save(payload, payload_path)

    never_hit_score = static["anisotropy"] * np.maximum(static["opacity"], 1e-6)
    low_hit_score = (1.0 / np.maximum(hit_count.astype(np.float32), 1.0)) * static["anisotropy"] * np.maximum(static["opacity"], 1e-6)
    write_point_cloud(
        output_root / "never_hit_preview_v0.ply",
        static["xyz"][never_hit_mask],
        never_hit_score[never_hit_mask],
        max_points=int(args.preview_max_points),
    )
    write_point_cloud(
        output_root / "never_hit_rodlike_preview_v0.ply",
        static["xyz"][never_hit_rodlike_mask],
        never_hit_score[never_hit_rodlike_mask],
        max_points=int(args.preview_max_points),
    )
    write_point_cloud(
        output_root / "low_hit_rodlike_preview_v0.ply",
        static["xyz"][low_hit_rodlike_mask],
        low_hit_score[low_hit_rodlike_mask],
        max_points=int(args.preview_max_points),
    )

    gaussian_subset_paths: Dict[str, object] = {}
    if not bool(args.no_gaussian_subset_export):
        subset_specs = {
            "never_hit": (never_hit_mask, never_hit_score, False),
            "never_hit_rodlike": (never_hit_rodlike_mask, never_hit_score, False),
            "low_hit_rodlike": (low_hit_rodlike_mask, low_hit_score, False),
            "never_hit_rodlike_debug_visible": (never_hit_rodlike_mask, never_hit_score, True),
            "low_hit_rodlike_debug_visible": (low_hit_rodlike_mask, low_hit_score, True),
        }
        for name, (mask, score, debug_visible) in subset_specs.items():
            gaussian_subset_paths[name] = write_gaussian_subset_model(
                source_ply_path=source_ply_path,
                source_tags_path=source_tags_path,
                output_model_path=output_root / f"{name}_gaussian_model_v0",
                iteration=iteration,
                mask=mask,
                score=score,
                max_points=int(args.preview_max_points),
                debug_visible=bool(debug_visible),
                debug_alpha=float(args.debug_visible_alpha),
                debug_scale_multiplier=float(args.debug_scale_multiplier),
            )

    summary = {
        "version": "global_ray_hit_gaussians_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(iteration),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "max_views": int(args.max_views),
        "selected_view_count": int(len(selected_cameras)),
        "source_view_count": int(len(cameras)),
        "selected_view_names": view_names,
        "thresholds": {
            "low_hit_max_views": int(args.low_hit_max_views),
            "rod_anisotropy_threshold": float(args.rod_anisotropy_threshold),
        },
        "counts": {
            "total_gaussians": int(total),
            "hit_any": int(np.sum(hit_count > 0)),
            "never_hit": int(np.sum(never_hit_mask)),
            "low_hit": int(np.sum(low_hit_mask)),
            "rodlike": int(np.sum(rodlike_mask)),
            "never_hit_rodlike": int(np.sum(never_hit_rodlike_mask)),
            "low_hit_rodlike": int(np.sum(low_hit_rodlike_mask)),
        },
        "stats": {
            "hit_count_all": stats_from_array(hit_count.astype(np.float32)),
            "hit_count_rodlike": stats_from_array(hit_count[rodlike_mask].astype(np.float32)),
            "anisotropy_never_hit": stats_from_array(static["anisotropy"][never_hit_mask]),
            "anisotropy_never_hit_rodlike": stats_from_array(static["anisotropy"][never_hit_rodlike_mask]),
            "opacity_never_hit": stats_from_array(static["opacity"][never_hit_mask]),
            "opacity_low_hit_rodlike": stats_from_array(static["opacity"][low_hit_rodlike_mask]),
            "mean_radius_hit_any": stats_from_array(mean_radius[hit_count > 0]),
        },
        "tracking": {
            "source_tag_never_hit": count_by_int(source_tag, never_hit_mask),
            "source_tag_never_hit_rodlike": count_by_int(source_tag, never_hit_rodlike_mask),
            "generation_never_hit": count_by_int(generation, never_hit_mask),
            "generation_never_hit_rodlike": count_by_int(generation, never_hit_rodlike_mask),
            "edge_touched_never_hit": {
                "false": int(np.sum(never_hit_mask & (~edge_touched))),
                "true": int(np.sum(never_hit_mask & edge_touched)),
            },
        },
        "top_records": {
            "never_hit_rodlike": build_top_records(
                never_hit_rodlike_mask,
                hit_count=hit_count,
                opacity=static["opacity"],
                anisotropy=static["anisotropy"],
                scale_max=static["scale_max"],
                scale_min=static["scale_min"],
                source_tag=source_tag,
                generation=generation,
                limit=int(args.top_records),
            ),
            "low_hit_rodlike": build_top_records(
                low_hit_rodlike_mask,
                hit_count=hit_count,
                opacity=static["opacity"],
                anisotropy=static["anisotropy"],
                scale_max=static["scale_max"],
                scale_min=static["scale_min"],
                source_tag=source_tag,
                generation=generation,
                limit=int(args.top_records),
            ),
        },
        "paths": {
            "payload": str(payload_path),
            "never_hit_preview": str(output_root / "never_hit_preview_v0.ply"),
            "never_hit_rodlike_preview": str(output_root / "never_hit_rodlike_preview_v0.ply"),
            "low_hit_rodlike_preview": str(output_root / "low_hit_rodlike_preview_v0.ply"),
            "gaussian_subset_models": gaussian_subset_paths,
        },
    }
    summary_path = output_root / "global_ray_hit_gaussians_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
