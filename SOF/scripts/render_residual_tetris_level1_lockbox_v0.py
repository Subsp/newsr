#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_residual_tetris_oracle_v0 as oracle  # noqa: E402
import render_residual_tetris_preview_v0 as preview  # noqa: E402
import render_residual_tetris_static_v1 as static_v1  # noqa: E402


VERSION = "render_residual_tetris_level1_lockbox_v0"
FORMAL_VARIANTS = (
    "core28",
    "deploy_top40_raw",
    "deploy_top40_bounded",
    "deploy_top40_minimal_clean_dev",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Level-1 same-scene novel-view lockbox for frozen Residual Tetris cells. "
            "It projects frozen 3D cells into lockbox views without refitting beta, "
            "changing top-k, or changing bounded parameters."
        )
    )
    parser.add_argument("--static_v1_dir", required=True)
    parser.add_argument("--oracle_dir", default="")
    parser.add_argument("--primitive_dir", required=True)
    parser.add_argument("--base_render_dir", required=True)
    parser.add_argument("--sr_dir", required=True)
    parser.add_argument("--weight_dir", required=True)
    parser.add_argument("--q_parent_dir", default="")
    parser.add_argument("--carrier_rgb_dir", default="")
    parser.add_argument("--carrier_render_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--check_dir", default="")
    parser.add_argument("--match_policy", choices=["stem", "order_if_needed", "order"], default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bounded_delta_clip", type=float, default=-1.0)
    parser.add_argument("--bounded_mode", choices=("per_cell_clip", "post_sum_clip", "from_config"), default="from_config")
    parser.add_argument("--visual_signed_scale", type=float, default=4.0)
    parser.add_argument("--error_scale", type=float, default=8.0)
    parser.add_argument("--lp_scale", type=float, default=16.0)
    parser.add_argument("--leak_scale", type=float, default=24.0)
    parser.add_argument("--changed_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--write_buffers", type=int, default=1)
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _list_images(root: Optional[Path]) -> List[Path]:
    if root is None or not root.is_dir():
        return []
    return oracle._list_files(root, oracle.IMAGE_EXTS if hasattr(oracle, "IMAGE_EXTS") else [".png", ".jpg", ".jpeg", ".webp"])


def _load_cell_rows(path: Path) -> List[Dict[str, object]]:
    payload = _read_json(path)
    rows = []
    for cell in payload.get("cells", []):
        if isinstance(cell, dict) and isinstance(cell.get("frozen_row"), dict):
            rows.append(cell["frozen_row"])
    return rows


def _load_frozen(static_dir: Path) -> Dict[str, object]:
    required = [
        "v1_manifest.json",
        "renderer_config.json",
        "cells_deploy_top40_raw.json",
        "cells_core28.json",
        "cells_minimal_clean_dev.json",
        "frozen_cells_3d.json",
    ]
    for name in required:
        path = static_dir / name
        if not path.exists():
            raise FileNotFoundError(f"required frozen static V1 file not found: {path}")
    frozen_cells = _read_json(static_dir / "frozen_cells_3d.json")
    frozen_by_id = {int(cell["cluster_id"]): cell for cell in frozen_cells.get("cells", [])}
    deploy_rows = _load_cell_rows(static_dir / "cells_deploy_top40_raw.json")
    core_rows = _load_cell_rows(static_dir / "cells_core28.json")
    minimal_rows = _load_cell_rows(static_dir / "cells_minimal_clean_dev.json")
    return {
        "manifest": _read_json(static_dir / "v1_manifest.json"),
        "config": _read_json(static_dir / "renderer_config.json"),
        "frozen_cells": frozen_cells,
        "frozen_by_id": frozen_by_id,
        "deploy_rows": deploy_rows,
        "core_rows": core_rows,
        "minimal_rows": minimal_rows,
        "deploy_ids": [preview._row_cluster_id(row) for row in deploy_rows],
        "core_ids": [preview._row_cluster_id(row) for row in core_rows],
        "minimal_ids": [preview._row_cluster_id(row) for row in minimal_rows],
    }


def _oracle_args_from_static(static_dir: Path, frozen: Dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    if args.oracle_dir:
        oracle_summary = _read_json(Path(args.oracle_dir) / "summary.json")
        oracle_args = preview._load_oracle_args(oracle_summary)
    else:
        manifest = frozen["manifest"]
        inputs = manifest.get("inputs", {})
        base = manifest.get("base_checkpoint", {})
        config = frozen["config"]
        target = config.get("target_definitions", {})
        values = {
            "base_model_dir": base.get("model_dir", ""),
            "base_iteration": int(base.get("iteration", 30000)),
            "primitive_dir": inputs.get("primitive_dir", ""),
            "base_render_dir": inputs.get("base_render_dir", ""),
            "sr_dir": inputs.get("sr_dir", ""),
            "weight_dir": inputs.get("weight_dir", ""),
            "q_parent_dir": inputs.get("q_parent_dir", ""),
            "match_policy": manifest.get("camera_parameters", {}).get("match_policy", "order_if_needed"),
            "limit": 0,
            "highpass_kernel": int(target.get("highpass_kernel", 9)),
            "lowpass_kernel": int(target.get("lowpass_kernel", 21)),
        }
        oracle_args = SimpleNamespace(**values)
    values = vars(oracle_args).copy()
    values["primitive_dir"] = str(Path(args.primitive_dir).expanduser().resolve())
    values["base_render_dir"] = str(Path(args.base_render_dir).expanduser().resolve())
    values["sr_dir"] = str(Path(args.sr_dir).expanduser().resolve())
    values["weight_dir"] = str(Path(args.weight_dir).expanduser().resolve())
    values["q_parent_dir"] = str(Path(args.q_parent_dir).expanduser().resolve()) if args.q_parent_dir else ""
    values["carrier_rgb_dir"] = str(Path(args.carrier_rgb_dir).expanduser().resolve()) if args.carrier_rgb_dir else ""
    values["carrier_render_dir"] = str(Path(args.carrier_render_dir).expanduser().resolve()) if args.carrier_render_dir else ""
    if args.match_policy:
        values["match_policy"] = str(args.match_policy)
    values["limit"] = int(args.limit)
    defaults = {
        "roi_pad_px": 12,
        "responsibility_bg_tau": 0.25,
        "core_weight_threshold": 0.015,
        "support_threshold": 0.03,
        "tolerance_radius": 3,
        "q_percentile": 95.0,
        "q_tau": 0.03,
        "lambda_off": 0.25,
        "lambda_lp": 0.05,
        "lambda_dc": 0.05,
        "beta_ridge": 1e-4,
        "beta_max": 0.35,
        "dipole_spacing": 0.85,
        "dog_large_scale": 2.35,
        "split_spacing": 0.65,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    return SimpleNamespace(**values)


def _load_lockbox_state(args: SimpleNamespace) -> Dict[str, object]:
    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    base_ply = base_model_dir / "point_cloud" / f"iteration_{int(args.base_iteration)}" / "point_cloud.ply"
    if not base_ply.exists():
        raise FileNotFoundError(f"base PLY not found: {base_ply}")
    primitive_dir = Path(args.primitive_dir).expanduser().resolve()
    primitive_paths = oracle._list_files(primitive_dir, [".npz"])
    if int(args.limit) > 0:
        primitive_paths = primitive_paths[: int(args.limit)]
    if not primitive_paths:
        raise RuntimeError(f"No lockbox primitive npz found in {primitive_dir}")

    cameras = oracle._load_cameras(base_model_dir)
    _base_vertices, base_xyz, base_opacity, _base_rgb = oracle._load_base_vertices(base_ply)
    spray_args = oracle._make_spray_args(args)

    carrier_rgb_paths = _list_images(Path(args.carrier_rgb_dir)) if str(getattr(args, "carrier_rgb_dir", "")) else []
    carrier_render_paths = _list_images(Path(args.carrier_render_dir)) if str(getattr(args, "carrier_render_dir", "")) else []
    weight_paths = _list_images(Path(args.weight_dir))
    carrier_rgb_lookup = oracle._image_lookup(carrier_rgb_paths)
    carrier_render_lookup = oracle._image_lookup(carrier_render_paths)
    weight_lookup_for_loader = oracle._image_lookup(weight_paths)

    views = []
    per_view = []
    for view_index, primitive_path in enumerate(primitive_paths):
        view, info = oracle._load_view_primitives(
            primitive_path,
            view_index,
            spray_args,
            cameras,
            base_xyz,
            base_opacity,
            carrier_rgb_paths,
            carrier_rgb_lookup,
            carrier_render_paths,
            carrier_render_lookup,
            weight_paths,
            weight_lookup_for_loader,
        )
        per_view.append(info)
        if view is None:
            print(f"[level1-lockbox-v0] skip {view_index + 1}/{len(primitive_paths)} {primitive_path.stem}: {info.get('status')}")
            continue
        views.append(view)
        print(f"[level1-lockbox-v0] view {len(views)}/{len(primitive_paths)} {view.stem} prims={view.mu_xy.shape[0]} q={float(view.q.mean()):.4f}")
    if not views:
        raise RuntimeError("No usable lockbox views.")

    base_paths = oracle._list_files(Path(args.base_render_dir).expanduser().resolve())
    sr_paths = oracle._list_files(Path(args.sr_dir).expanduser().resolve())
    q_root = Path(args.q_parent_dir).expanduser().resolve() if str(getattr(args, "q_parent_dir", "")) else None
    q_paths = oracle._list_files(q_root) if q_root is not None and q_root.is_dir() else []
    return {
        "base_model_dir": base_model_dir,
        "base_paths": base_paths,
        "base_lookup": oracle._image_lookup(base_paths),
        "sr_paths": sr_paths,
        "sr_lookup": oracle._image_lookup(sr_paths),
        "weight_paths": weight_paths,
        "weight_lookup": oracle._image_lookup(weight_paths),
        "q_paths": q_paths,
        "q_lookup": oracle._image_lookup(q_paths),
        "views": views,
        "view_index_by_stem": {view.stem: int(view.view_index) for view in views},
        "per_view": per_view,
    }


def _project_frozen_frame(cell: Dict[str, object], target_view: oracle.ViewPrimitiveSet) -> Tuple[np.ndarray, float, float, float, bool]:
    xyz = np.asarray(cell["xyz"], dtype=np.float32).reshape(3)
    normal = np.asarray(cell["normal"], dtype=np.float32).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    src = cell["source_view"]
    src_camera = src["camera"]
    source_size = (int(src["source_size"][0]), int(src["source_size"][1]))
    theta = float(src["theta"])
    _pos, cam_x, cam_y, _cam_z, fx, fy, _cw, _ch = oracle._camera_basis(src_camera)
    tangent = math.cos(theta) * cam_x.astype(np.float32) + math.sin(theta) * cam_y.astype(np.float32)
    tangent = tangent - normal * float(np.dot(tangent, normal))
    if float(np.linalg.norm(tangent)) <= 1e-8:
        tangent = cam_x.astype(np.float32) - normal * float(np.dot(cam_x, normal))
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-8)
    bitangent = np.cross(normal, tangent).astype(np.float32)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-8)
    _center_src, depth_src, ok_src = oracle._project_point(src_camera, source_size, xyz)
    center_tgt, _depth_tgt, ok_tgt = oracle._project_point(target_view.camera, target_view.source_size, xyz)
    if not ok_src or not ok_tgt:
        return center_tgt.astype(np.float32), 0.0, 0.0, 0.0, False
    pixel_to_world = float(depth_src) / max((float(fx) + float(fy)) * 0.5, 1e-8)
    long_world = max(float(src["long_px"]), 0.5) * pixel_to_world
    short_world = max(float(src["short_px"]), 0.5) * pixel_to_world
    p_t, _dt, ok_t = oracle._project_point(target_view.camera, target_view.source_size, xyz + tangent * long_world)
    p_b, _db, ok_b = oracle._project_point(target_view.camera, target_view.source_size, xyz + bitangent * short_world)
    if not ok_t or not ok_b:
        return center_tgt.astype(np.float32), 0.0, 0.0, 0.0, False
    axis_t = p_t - center_tgt
    axis_b = p_b - center_tgt
    long_px = max(float(np.linalg.norm(axis_t)), 0.45)
    short_px = max(float(np.linalg.norm(axis_b)), 0.45)
    out_theta = math.atan2(float(axis_t[1]), float(axis_t[0]))
    return center_tgt.astype(np.float32), float(out_theta), float(long_px), float(short_px), True


def _target_hf_for_view(stem: str, view_index: int, state: Dict[str, object], args: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_path = oracle._resolve_path(state["base_paths"], state["base_lookup"], stem, view_index, str(args.match_policy))
    base = oracle._load_rgb(base_path)
    size = (base.shape[1], base.shape[0])
    sr_path = oracle._resolve_path(state["sr_paths"], state["sr_lookup"], stem, view_index, str(args.match_policy))
    weight_path = oracle._resolve_path(state["weight_paths"], state["weight_lookup"], stem, view_index, str(args.match_policy))
    sr = oracle._load_rgb(sr_path, size=size)
    weight = oracle._load_gray(weight_path, size=size)
    target = oracle._highpass(sr - base, int(args.highpass_kernel)).astype(np.float32)
    return base.astype(np.float32), sr.astype(np.float32), target, weight.astype(np.float32)


def _build_lockbox_obs(
    *,
    cluster_ids: Iterable[int],
    frozen_by_id: Dict[int, Dict[str, object]],
    state: Dict[str, object],
    args: SimpleNamespace,
) -> Dict[int, List[oracle.EvalObs]]:
    out: Dict[int, List[oracle.EvalObs]] = {}
    for cluster_id in sorted(set(int(x) for x in cluster_ids)):
        cell = frozen_by_id.get(cluster_id)
        if cell is None:
            out[cluster_id] = []
            continue
        obs_list: List[oracle.EvalObs] = []
        for view_slot, view in enumerate(state["views"]):
            center, theta, long_px, short_px, ok = _project_frozen_frame(cell, view)
            if not ok:
                continue
            base, _sr, target, weight_img = _target_hf_for_view(view.stem, int(view.view_index), state, args)
            if state["q_paths"]:
                q_path = oracle._resolve_path(state["q_paths"], state["q_lookup"], view.stem, int(view.view_index), str(args.match_policy))
                q_img = oracle._load_gray(q_path, size=(base.shape[1], base.shape[0]))
            else:
                q_img = weight_img.copy()
            roi = oracle._roi_bounds(center, long_px, short_px, (base.shape[0], base.shape[1]), int(args.roi_pad_px))
            x0, y0, x1, y1 = roi
            xx, yy = oracle._coords(roi)
            support = oracle._gaussian_2d(xx, yy, center, theta, long_px, short_px)
            responsibility = support / (float(args.responsibility_bg_tau) + support)
            weight_roi = np.clip(weight_img[y0:y1, x0:x1], 0.0, 1.0)
            core_weight = np.clip(responsibility * weight_roi, 0.0, 1.0).astype(np.float32)
            obs_list.append(
                oracle.EvalObs(
                    view_slot=int(view_slot),
                    primitive_index=-1,
                    view_index=int(view.view_index),
                    stem=str(view.stem),
                    split="lockbox",
                    center=center.astype(np.float32),
                    theta=float(theta),
                    long_px=float(long_px),
                    short_px=float(short_px),
                    support=support.astype(np.float32),
                    responsibility=responsibility.astype(np.float32),
                    core_weight=core_weight,
                    target_hf=target[y0:y1, x0:x1].astype(np.float32),
                    q_parent=np.clip(q_img[y0:y1, x0:x1], 0.0, 1.0).astype(np.float32),
                    roi=roi,
                )
            )
        out[cluster_id] = obs_list
    return out


def _compose_variants(
    *,
    rows_by_name: Dict[str, List[Dict[str, object]]],
    ids_by_name: Dict[str, List[int]],
    contributions: Dict[int, List[preview.Contribution]],
    stems: Sequence[str],
    state_for_compose: Dict[str, object],
    args: SimpleNamespace,
    bounded_mode: str,
    bounded_clip: float,
) -> Dict[str, Dict[str, object]]:
    variants = {
        "core28": (ids_by_name["core28"], rows_by_name["core28"], None),
        "deploy_top40_raw": (ids_by_name["deploy_top40_raw"], rows_by_name["deploy_top40_raw"], None),
        "deploy_top40_bounded": (ids_by_name["deploy_top40_raw"], rows_by_name["deploy_top40_raw"], bounded_clip),
        "deploy_top40_minimal_clean_dev": (
            ids_by_name["deploy_top40_minimal_clean_dev"],
            rows_by_name["deploy_top40_minimal_clean_dev"],
            None,
        ),
    }
    out: Dict[str, Dict[str, object]] = {}
    for name, (ids, rows, clip) in variants.items():
        out[name] = {
            "ids": ids,
            "rows": rows,
            "all": static_v1._static_compose(
                selected_ids=ids,
                contributions=contributions,
                stems=stems,
                state=state_for_compose,
                args=args,
                bounded_clip=clip,
                bounded_mode=bounded_mode,
            ),
        }
    return out


def _changed_metrics(data: Dict[str, np.ndarray], threshold: float) -> Dict[str, float]:
    pred_luma = np.abs(np.sum(data["pred"] * oracle.LUMA[None, None, :], axis=2))
    changed = pred_luma > float(threshold)
    target = data["weight"] > 1e-8
    off = data["off_weight"] > 1e-8
    return {
        "changed_on_target": float(np.count_nonzero(changed & target) / max(float(np.count_nonzero(target)), 1.0)),
        "changed_off_target": float(np.count_nonzero(changed & off) / max(float(np.count_nonzero(off)), 1.0)),
        "changed_total_ratio": float(np.count_nonzero(changed) / max(float(changed.size), 1.0)),
    }


def _aggregate_changed(composed: Dict[str, Dict[str, np.ndarray]], threshold: float) -> Dict[str, float]:
    rows = [_changed_metrics(data, threshold) for data in composed.values()]
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _variant_metrics(
    *,
    name: str,
    rows: Sequence[Dict[str, object]],
    composed: Dict[str, Dict[str, np.ndarray]],
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    lowpass_kernel: int,
    changed_threshold: float,
) -> Dict[str, object]:
    metrics = preview._metrics_with_support_comparison(
        composed,
        lowpass_kernel=lowpass_kernel,
        raw_support=raw_support,
        global_support=global_support,
    )
    metrics["joint_gain"] = metrics.get("gain", float("nan"))
    selection_positive = preview._positive_individual_gain(rows, "selection")
    analysis_positive = preview._positive_individual_gain(rows, "test")
    metrics["sum_positive_individual_gain_selection_dev"] = float(selection_positive)
    metrics["sum_positive_individual_gain_analysis_dev"] = float(analysis_positive)
    metrics["joint_gain_capture"] = float(metrics["joint_gain"] / max(selection_positive, 1e-10))
    metrics["joint_gain_capture_vs_dev_analysis"] = float(metrics["joint_gain"] / max(analysis_positive, 1e-10))
    metrics.update(_aggregate_changed(composed, changed_threshold))
    return {"name": name, "cell_count": int(len(rows)), "lockbox": metrics}


def _per_view_report(
    *,
    composed: Dict[str, Dict[str, np.ndarray]],
    lowpass_kernel: int,
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    changed_threshold: float,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for stem, data in composed.items():
        raw_w = raw_support.get(stem, {}).get("weight")
        raw_ow = raw_support.get(stem, {}).get("off_weight")
        global_w = global_support.get(stem, {}).get("weight")
        global_ow = global_support.get(stem, {}).get("off_weight")
        entry = {
            "variant_support": preview._metrics_for_view(data, lowpass_kernel=lowpass_kernel),
            "raw_union_support": preview._metrics_for_view(data, lowpass_kernel=lowpass_kernel, support_weight=raw_w, off_support_weight=raw_ow),
            "global_eligible_support": preview._metrics_for_view(data, lowpass_kernel=lowpass_kernel, support_weight=global_w, off_support_weight=global_ow),
        }
        entry["changed"] = _changed_metrics(data, changed_threshold)
        out[stem] = entry
    return out


def _save_outputs(
    *,
    variants: Dict[str, Dict[str, object]],
    state: Dict[str, object],
    stems: Sequence[str],
    args_ns: SimpleNamespace,
    cli_args: argparse.Namespace,
    output_dir: Path,
) -> None:
    preview._save_shared_visuals(stems=stems, state=state, oracle_args=args_ns, output_dir=output_dir, args=cli_args)
    for name, spec in variants.items():
        composed = spec["all"]
        preview._save_variant_visuals(
            variant_name=name,
            composed=composed,  # type: ignore[arg-type]
            output_dir=output_dir,
            args=cli_args,
            lowpass_kernel=int(args_ns.lowpass_kernel),
        )
        static_v1._save_extra_visuals(variant_name=name, composed=composed, output_dir=output_dir, args=cli_args)  # type: ignore[arg-type]
        if int(cli_args.write_buffers):
            static_v1._save_buffers(variant_name=name, composed=composed, output_dir=output_dir)  # type: ignore[arg-type]


def _copy_to_check(output_dir: Path, check_dir: Path) -> None:
    if not str(check_dir):
        return
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "level1_lockbox_metrics.json", "per_view_metrics.json", "lockbox_manifest.json"):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, check_dir / name)
    for rel in (
        "visuals/deploy_top40_raw/base_plus_residual",
        "visuals/deploy_top40_raw/residual_pred",
        "visuals/deploy_top40_bounded/base_plus_residual",
        "visuals/deploy_top40_bounded/residual_pred",
        "visuals/deploy_top40_minimal_clean_dev/base_plus_residual",
        "visuals/deploy_top40_minimal_clean_dev/residual_pred",
        "visuals/core28/base_plus_residual",
        "visuals/base",
        "visuals/target_hf",
    ):
        src_dir = output_dir / rel
        if not src_dir.is_dir():
            continue
        dst_dir = check_dir / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(src_dir.glob("*.png"))[:64]:
            shutil.copy2(image_path, dst_dir / image_path.name)


def main() -> None:
    t0 = time.time()
    cli_args = _parse_args()
    static_dir = Path(cli_args.static_v1_dir).expanduser().resolve()
    output_dir = Path(cli_args.output_dir).expanduser().resolve()
    check_dir = Path(cli_args.check_dir).expanduser().resolve() if cli_args.check_dir else Path("")
    if output_dir.exists() and not cli_args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    if output_dir.exists() and cli_args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = _load_frozen(static_dir)
    args_ns = _oracle_args_from_static(static_dir, frozen, cli_args)
    config = frozen["config"]
    bounded_mode = str(config.get("bounded_mode", "per_cell_clip")) if cli_args.bounded_mode == "from_config" else str(cli_args.bounded_mode)
    bounded_clip = float(config.get("bounded_delta_clip", 0.08)) if float(cli_args.bounded_delta_clip) < 0.0 else float(cli_args.bounded_delta_clip)

    print(f"[level1-lockbox-v0] static : {static_dir}")
    print(f"[level1-lockbox-v0] output : {output_dir}")
    print(f"[level1-lockbox-v0] primitive: {args_ns.primitive_dir}")
    print(f"[level1-lockbox-v0] bounded: {bounded_mode} clip={bounded_clip:.4f}")

    state = _load_lockbox_state(args_ns)
    stems = [view.stem for view in state["views"]]
    compose_state = {
        **state,
        "view_index_by_stem": {view.stem: int(view.view_index) for view in state["views"]},
    }
    frozen_by_id = frozen["frozen_by_id"]
    rows_by_name = {
        "core28": frozen["core_rows"],
        "deploy_top40_raw": frozen["deploy_rows"],
        "deploy_top40_minimal_clean_dev": frozen["minimal_rows"],
    }
    ids_by_name = {
        "core28": frozen["core_ids"],
        "deploy_top40_raw": frozen["deploy_ids"],
        "deploy_top40_minimal_clean_dev": frozen["minimal_ids"],
    }
    all_ids = sorted(set(frozen["frozen_by_id"].keys()))
    obs_by_cluster = _build_lockbox_obs(cluster_ids=all_ids, frozen_by_id=frozen_by_id, state=state, args=args_ns)
    row_union = [cell["frozen_row"] for cid, cell in sorted(frozen_by_id.items()) if obs_by_cluster.get(cid)]
    contributions = preview._build_contributions(row_union, obs_by_cluster, args_ns)

    raw_support = preview._support_map_from_composed(
        static_v1._static_compose(
            selected_ids=ids_by_name["deploy_top40_raw"],
            contributions=contributions,
            stems=stems,
            state=compose_state,
            args=args_ns,
        )
    )
    global_support = preview._support_map_from_composed(
        static_v1._static_compose(
            selected_ids=all_ids,
            contributions=contributions,
            stems=stems,
            state=compose_state,
            args=args_ns,
        )
    )
    variants = _compose_variants(
        rows_by_name=rows_by_name,
        ids_by_name=ids_by_name,
        contributions=contributions,
        stems=stems,
        state_for_compose=compose_state,
        args=args_ns,
        bounded_mode=bounded_mode,
        bounded_clip=bounded_clip,
    )
    _save_outputs(variants=variants, state=compose_state, stems=stems, args_ns=args_ns, cli_args=cli_args, output_dir=output_dir)

    metrics = {
        "version": VERSION,
        "static_v1_dir": str(static_dir),
        "lockbox_views": stems,
        "support_policy": {
            "primary": ["raw_union_support", "global_eligible_support"],
            "supplemental": "variant_support",
        },
        "bounded_mode": bounded_mode,
        "bounded_delta_clip": bounded_clip,
        "variants": {},
    }
    per_view = {"version": VERSION, "variants": {}}
    for name, spec in variants.items():
        row_key = "deploy_top40_raw" if name == "deploy_top40_bounded" else name
        rows = rows_by_name[row_key]
        metrics["variants"][name] = _variant_metrics(
            name=name,
            rows=rows,
            composed=spec["all"],  # type: ignore[arg-type]
            raw_support=raw_support,
            global_support=global_support,
            lowpass_kernel=int(args_ns.lowpass_kernel),
            changed_threshold=float(cli_args.changed_threshold),
        )
        per_view["variants"][name] = _per_view_report(
            composed=spec["all"],  # type: ignore[arg-type]
            lowpass_kernel=int(args_ns.lowpass_kernel),
            raw_support=raw_support,
            global_support=global_support,
            changed_threshold=float(cli_args.changed_threshold),
        )
    runtime = time.time() - t0
    manifest = {
        "version": VERSION,
        "static_v1_dir": str(static_dir),
        "frozen_manifest": frozen["manifest"],
        "lockbox_inputs": {
            "primitive_dir": str(args_ns.primitive_dir),
            "base_render_dir": str(args_ns.base_render_dir),
            "sr_dir": str(args_ns.sr_dir),
            "weight_dir": str(args_ns.weight_dir),
            "q_parent_dir": str(getattr(args_ns, "q_parent_dir", "")),
            "match_policy": str(args_ns.match_policy),
            "limit": int(args_ns.limit),
        },
        "forbidden_actions": [
            "no cell deletion",
            "no top-k change",
            "no beta refit",
            "no bounded retuning",
            "no support-mask retuning",
        ],
    }
    summary = {
        "version": VERSION,
        "runtime_sec": float(runtime),
        "num_lockbox_views": int(len(stems)),
        "lockbox_views": stems,
        "ready_to_evaluate": True,
        "headline": {
            name: {
                "gain_on_raw_union_support": spec["lockbox"].get("gain_on_raw_union_support"),
                "gain_on_global_eligible_support": spec["lockbox"].get("gain_on_global_eligible_support"),
                "positive_view_ratio": spec["lockbox"].get("positive_view_ratio"),
                "view_gain_min": spec["lockbox"].get("view_gain_min"),
                "HF_corr": spec["lockbox"].get("HF_corr"),
                "LP_drift": spec["lockbox"].get("LP_drift"),
                "off_target_leakage": spec["lockbox"].get("off_target_leakage"),
                "out_of_range_ratio": spec["lockbox"].get("out_of_range_ratio"),
                "joint_gain_capture": spec["lockbox"].get("joint_gain_capture"),
            }
            for name, spec in metrics["variants"].items()
        },
    }
    _write_json(output_dir / "lockbox_manifest.json", manifest)
    _write_json(output_dir / "level1_lockbox_metrics.json", metrics)
    _write_json(output_dir / "per_view_metrics.json", per_view)
    _write_json(output_dir / "summary.json", summary)
    if str(check_dir):
        _copy_to_check(output_dir, check_dir)
    print(f"[level1-lockbox-v0] summary: {output_dir / 'summary.json'}")
    print(f"[level1-lockbox-v0] metrics: {output_dir / 'level1_lockbox_metrics.json'}")
    if str(check_dir):
        print(f"[level1-lockbox-v0] check  : {check_dir}")


if __name__ == "__main__":
    main()
