#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_residual_tetris_oracle_v0 as oracle  # noqa: E402
import render_residual_tetris_preview_v0 as preview  # noqa: E402


VERSION = "render_residual_tetris_static_v1"
SPLITS = preview.SPLITS
SPLIT_DISPLAY = preview.SPLIT_DISPLAY
MAIN_VARIANTS = (
    "core28",
    "deploy_top40_raw",
    "deploy_top40_bounded",
    "deploy_top40_minimal_clean_dev",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static V1 Residual Tetris renderer. It freezes oracle cells and "
            "replays q_parent * signed_piece * beta in linear RGB without refit."
        )
    )
    parser.add_argument("--oracle_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--check_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bounded_delta_clip", type=float, default=0.08)
    parser.add_argument(
        "--bounded_mode",
        choices=("per_cell_clip", "post_sum_clip"),
        default="per_cell_clip",
        help="per_cell_clip matches the validated offline preview baseline.",
    )
    parser.add_argument("--minimal_clean_drop_cluster_ids", default="165")
    parser.add_argument("--dose_counts", default="5,10,20,40")
    parser.add_argument("--focus_cluster_ids", default="64,174,70,133,156,223")
    parser.add_argument("--visual_signed_scale", type=float, default=4.0)
    parser.add_argument("--error_scale", type=float, default=8.0)
    parser.add_argument("--lp_scale", type=float, default=16.0)
    parser.add_argument("--leak_scale", type=float, default=24.0)
    parser.add_argument("--out_of_range_scale", type=float, default=1.0)
    parser.add_argument("--write_buffers", type=int, default=1)
    parser.add_argument("--write_per_cell_buffers", type=int, default=1)
    parser.add_argument("--per_cell_buffer_variant", default="deploy_top40_raw")
    parser.add_argument("--closure_small_count", type=int, default=5)
    parser.add_argument("--closure_single_cluster_id", type=int, default=-1)
    return parser.parse_args()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _sha256_json(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_or_empty(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _finite(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _variant_rows_by_id(rows: Sequence[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    return {preview._row_cluster_id(row): row for row in rows}


def _load_rows(oracle_dir: Path, minimal_drop_ids: Iterable[int]) -> Dict[str, object]:
    deploy_rows = preview._read_json(oracle_dir / "deploy_selected_rows.json")
    core_rows = preview._read_json(oracle_dir / "core_selected_rows.json")
    rows_all = preview._read_json(oracle_dir / "rows.json")
    rows_by_id = {preview._row_cluster_id(row): row for row in rows_all if row.get("valid")}
    deploy_ids = [preview._row_cluster_id(row) for row in deploy_rows]
    core_ids = [preview._row_cluster_id(row) for row in core_rows]
    drop = {int(x) for x in minimal_drop_ids}
    minimal_rows = [row for row in deploy_rows if preview._row_cluster_id(row) not in drop]
    minimal_ids = [preview._row_cluster_id(row) for row in minimal_rows]
    return {
        "deploy_rows": deploy_rows,
        "core_rows": core_rows,
        "rows_all": rows_all,
        "rows_by_id": rows_by_id,
        "valid_ids": sorted(rows_by_id),
        "deploy_ids": deploy_ids,
        "core_ids": core_ids,
        "minimal_clean_rows": minimal_rows,
        "minimal_clean_ids": minimal_ids,
        "minimal_drop_ids": sorted(drop),
    }


def _empty(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.float32)


def _accum_roi(canvas: np.ndarray, roi: Tuple[int, int, int, int], value: np.ndarray) -> None:
    x0, y0, x1, y1 = roi
    canvas[y0:y1, x0:x1] += value


def _max_roi(canvas: np.ndarray, roi: Tuple[int, int, int, int], value: np.ndarray) -> None:
    x0, y0, x1, y1 = roi
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], value)


def _update_dominant(
    dominant_val: np.ndarray,
    dominant_id: np.ndarray,
    roi: Tuple[int, int, int, int],
    cluster_id: int,
    abs_luma: np.ndarray,
) -> None:
    x0, y0, x1, y1 = roi
    val = dominant_val[y0:y1, x0:x1]
    ids = dominant_id[y0:y1, x0:x1]
    better = abs_luma > val
    val[better] = abs_luma[better]
    ids[better] = int(cluster_id)


def _luma(value: np.ndarray) -> np.ndarray:
    return np.sum(value * oracle.LUMA[None, None, :], axis=2).astype(np.float32)


def _static_compose(
    *,
    selected_ids: Sequence[int],
    contributions: Dict[int, List[preview.Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    split_filter: Optional[str] = None,
    bounded_clip: Optional[float] = None,
    bounded_mode: str = "per_cell_clip",
) -> Dict[str, Dict[str, np.ndarray]]:
    selected = [int(x) for x in selected_ids]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for stem in stems:
        base, _sr, target = preview._target_hf_for_stem(stem, state, args)
        h, w = base.shape[:2]
        pred = _empty((h, w))
        pred_raw = _empty((h, w))
        pred_lp = _empty((h, w))
        positive_lobe = np.zeros((h, w), dtype=np.float32)
        negative_lobe = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        off_weight = np.zeros((h, w), dtype=np.float32)
        overlap = np.zeros((h, w), dtype=np.float32)
        q_parent_peak = np.zeros((h, w), dtype=np.float32)
        dominant_val = np.zeros((h, w), dtype=np.float32)
        dominant_id = np.full((h, w), -1, dtype=np.int32)
        positive_hits = np.zeros((h, w), dtype=np.float32)
        negative_hits = np.zeros((h, w), dtype=np.float32)

        for cluster_id in selected:
            for contrib in contributions.get(cluster_id, []):
                if contrib.stem != stem:
                    continue
                if split_filter is not None and contrib.split != split_filter:
                    continue
                value = contrib.pred_hp
                if bounded_clip is not None and bounded_mode == "per_cell_clip":
                    value = np.clip(value, -float(bounded_clip), float(bounded_clip))
                _accum_roi(pred, contrib.roi, value)
                _accum_roi(pred_raw, contrib.roi, contrib.pred_raw)
                _accum_roi(pred_lp, contrib.roi, contrib.pred_lp)
                _max_roi(weight, contrib.roi, contrib.fit_w)
                _max_roi(off_weight, contrib.roi, contrib.off_w)
                x0, y0, x1, y1 = contrib.roi
                active = contrib.fit_w > float(args.core_weight_threshold)
                overlap[y0:y1, x0:x1] += active.astype(np.float32)
                q_parent_peak[y0:y1, x0:x1] = np.maximum(q_parent_peak[y0:y1, x0:x1], contrib.q_parent)
                value_luma = _luma(value)
                positive_lobe[y0:y1, x0:x1] += np.maximum(value_luma, 0.0)
                negative_lobe[y0:y1, x0:x1] += np.maximum(-value_luma, 0.0)
                positive_hits[y0:y1, x0:x1] += ((value_luma > 0.0) & active).astype(np.float32)
                negative_hits[y0:y1, x0:x1] += ((value_luma < 0.0) & active).astype(np.float32)
                _update_dominant(dominant_val, dominant_id, contrib.roi, cluster_id, contrib.abs_luma)

        if bounded_clip is not None and bounded_mode == "post_sum_clip":
            pred = np.clip(pred, -float(bounded_clip), float(bounded_clip))
        raw_image = base + pred
        out[stem] = {
            "base": base,
            "target_hf": target,
            "pred": pred,
            "pred_raw": pred_raw,
            "pred_lp": pred_lp,
            "weight": weight,
            "off_weight": off_weight,
            "overlap": overlap,
            "q_parent_peak": q_parent_peak,
            "dominant_id": dominant_id,
            "preview": np.clip(raw_image, 0.0, 1.0),
            "out_of_range": (raw_image < 0.0) | (raw_image > 1.0),
            "piece_positive_lobe": positive_lobe,
            "piece_negative_lobe": negative_lobe,
            "sign_conflict_count": np.minimum(positive_hits, negative_hits),
        }
    return out


def _compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.dtype == np.bool_:
        aa = aa.astype(np.float32)
    if bb.dtype == np.bool_:
        bb = bb.astype(np.float32)
    if np.issubdtype(aa.dtype, np.integer):
        aa = aa.astype(np.float32)
    if np.issubdtype(bb.dtype, np.integer):
        bb = bb.astype(np.float32)
    diff = aa.astype(np.float64) - bb.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = float(np.sqrt(np.sum(bb.astype(np.float64) ** 2)) + 1e-12)
    return {
        "mae": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "relative_l2": float(np.sqrt(np.sum(diff * diff)) / denom) if abs_diff.size else 0.0,
    }


def _compare_composed(
    static_data: Dict[str, Dict[str, np.ndarray]],
    preview_data: Dict[str, Dict[str, np.ndarray]],
    *,
    lowpass_kernel: int,
) -> Dict[str, object]:
    render_critical_keys = (
        "pred",
        "pred_raw",
        "pred_lp",
        "weight",
        "off_weight",
        "overlap",
        "q_parent_peak",
        "preview",
        "out_of_range",
    )
    diagnostic_keys = ("dominant_id",)
    keys = (*render_critical_keys, *diagnostic_keys)
    per_key: Dict[str, Dict[str, float]] = {}
    for key in keys:
        rows = []
        for stem in sorted(static_data):
            if stem not in preview_data:
                continue
            rows.append(_compare_arrays(static_data[stem][key], preview_data[stem][key]))
        if not rows:
            continue
        per_key[key] = {
            "mae": float(max(row["mae"] for row in rows)),
            "max_abs": float(max(row["max_abs"] for row in rows)),
            "relative_l2": float(max(row["relative_l2"] for row in rows)),
        }

    static_metrics = preview._metrics_for_composed(static_data, lowpass_kernel=lowpass_kernel)
    preview_metrics = preview._metrics_for_composed(preview_data, lowpass_kernel=lowpass_kernel)
    metric_diff: Dict[str, float] = {}
    for key in sorted(set(static_metrics) | set(preview_metrics)):
        av = static_metrics.get(key)
        bv = preview_metrics.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if math.isfinite(float(av)) and math.isfinite(float(bv)):
                metric_diff[key] = float(abs(float(av) - float(bv)))
    render_key_stats = {key: per_key[key] for key in render_critical_keys if key in per_key}
    diagnostic_key_stats = {key: per_key[key] for key in diagnostic_keys if key in per_key}
    render_critical_pass = (
        all(v["mae"] < 1e-5 for v in render_key_stats.values())
        and all(v["max_abs"] < 1e-4 for v in render_key_stats.values())
        and all(v < 1e-3 for v in metric_diff.values())
    )
    diagnostic_exact_pass = (
        all(v["mae"] < 1e-5 for v in diagnostic_key_stats.values())
        and all(v["max_abs"] < 1e-4 for v in diagnostic_key_stats.values())
    )
    return {
        "array_max_by_key": per_key,
        "render_critical_keys": list(render_critical_keys),
        "diagnostic_keys": list(diagnostic_keys),
        "static_metrics": static_metrics,
        "preview_metrics": preview_metrics,
        "metric_abs_diff": metric_diff,
        "render_critical_pass": bool(render_critical_pass),
        "diagnostic_exact_pass": bool(diagnostic_exact_pass),
        "overall_ready_for_lockbox": bool(render_critical_pass),
        "diagnostic_warning": (
            "dominant_id is diagnostic-only and may differ on near-tie pixels; it does not affect residual rendering."
            if not diagnostic_exact_pass
            else ""
        ),
        "passes": {
            "render_critical_float_mae_lt_1e-5": all(v["mae"] < 1e-5 for v in render_key_stats.values()),
            "render_critical_max_error_lt_1e-4": all(v["max_abs"] < 1e-4 for v in render_key_stats.values()),
            "metric_diff_lt_1e-3": all(v < 1e-3 for v in metric_diff.values()),
            "diagnostic_exact": bool(diagnostic_exact_pass),
        },
    }


def _closure_checks(
    *,
    deploy_ids: Sequence[int],
    minimal_ids: Sequence[int],
    contributions: Dict[int, List[preview.Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    bounded_clip: float,
    bounded_mode: str,
    single_cluster_id: int,
    small_count: int,
) -> Dict[str, object]:
    single = int(single_cluster_id) if int(single_cluster_id) >= 0 else int(deploy_ids[0])
    cases = {
        "single_cell": [single],
        f"small_set_top{int(small_count)}": list(deploy_ids[: int(small_count)]),
        "deploy_top40_raw": list(deploy_ids),
        "deploy_top40_bounded": list(deploy_ids),
        "deploy_top40_minimal_clean_dev": list(minimal_ids),
    }
    out: Dict[str, object] = {}
    for name, selected in cases.items():
        bounded = float(bounded_clip) if name == "deploy_top40_bounded" and bounded_mode == "per_cell_clip" else None
        static_data = _static_compose(
            selected_ids=selected,
            contributions=contributions,
            stems=stems,
            state=state,
            args=args,
            bounded_clip=bounded,
            bounded_mode=bounded_mode,
        )
        preview_data = preview._compose(
            selected_ids=selected,
            contributions=contributions,
            stems=stems,
            state=state,
            args=args,
            bounded_clip=bounded,
        )
        out[name] = _compare_composed(static_data, preview_data, lowpass_kernel=int(args.lowpass_kernel))
    if bounded_mode != "per_cell_clip":
        out["note"] = "preview-v0 closure only applies to per_cell_clip; post_sum_clip is intentionally different."
    return out


def _variant_report_static(
    *,
    name: str,
    rows: Sequence[Dict[str, object]],
    composed_all: Dict[str, Dict[str, np.ndarray]],
    composed_by_split: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    raw_support_by_split: Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]],
    global_support_by_split: Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]],
    lowpass_kernel: int,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "name": name,
        "cell_count": int(len(rows)),
        "all_views": preview._metrics_with_support_comparison(
            composed_all,
            lowpass_kernel=lowpass_kernel,
            raw_support=raw_support_by_split.get(None),
            global_support=global_support_by_split.get(None),
        ),
        "by_split": {},
    }
    for split in SPLITS:
        metrics = preview._metrics_with_support_comparison(
            composed_by_split[split],
            lowpass_kernel=lowpass_kernel,
            raw_support=raw_support_by_split.get(split),
            global_support=global_support_by_split.get(split),
        )
        sum_individual = preview._individual_gain(rows, split)
        sum_positive = preview._positive_individual_gain(rows, split)
        metrics["joint_gain"] = metrics.get("gain", float("nan"))
        metrics["sum_individual_gain"] = float(sum_individual)
        metrics["sum_positive_individual_gain"] = float(sum_positive)
        metrics["joint_over_sum_individual_gain_signed"] = (
            float(metrics["joint_gain"] / sum_individual) if abs(sum_individual) > 1e-10 else float("nan")
        )
        metrics["joint_gain_capture_positive"] = float(metrics["joint_gain"] / max(sum_positive, 1e-10))
        report["by_split"][SPLIT_DISPLAY[split]] = metrics
    return report


def _save_extra_visuals(
    *,
    variant_name: str,
    composed: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    root = output_dir / "visuals" / variant_name
    for stem, data in composed.items():
        oracle._save_gray(
            root / "q_parent" / f"{stem}.png",
            np.clip(data["q_parent_peak"], 0.0, 1.0),
        )
        oracle._save_gray(
            root / "piece_positive_lobes" / f"{stem}.png",
            np.clip(data["piece_positive_lobe"] * float(args.visual_signed_scale), 0.0, 1.0),
        )
        oracle._save_gray(
            root / "piece_negative_lobes" / f"{stem}.png",
            np.clip(data["piece_negative_lobe"] * float(args.visual_signed_scale), 0.0, 1.0),
        )
        oracle._save_gray(
            root / "sign_conflict_count" / f"{stem}.png",
            np.clip(data["sign_conflict_count"] / 4.0, 0.0, 1.0),
        )
        out = np.any(data["out_of_range"], axis=2).astype(np.float32)
        oracle._save_gray(
            root / "out_of_range_mask" / f"{stem}.png",
            np.clip(out * float(args.out_of_range_scale), 0.0, 1.0),
        )


def _save_buffers(
    *,
    variant_name: str,
    composed: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    root = output_dir / "buffers" / variant_name
    root.mkdir(parents=True, exist_ok=True)
    for stem, data in composed.items():
        np.savez_compressed(
            root / f"{stem}.npz",
            residual_sum=data["pred"].astype(np.float32),
            residual_raw=data["pred_raw"].astype(np.float32),
            residual_lp=data["pred_lp"].astype(np.float32),
            q_parent=data["q_parent_peak"].astype(np.float32),
            piece_positive_lobe=data["piece_positive_lobe"].astype(np.float32),
            piece_negative_lobe=data["piece_negative_lobe"].astype(np.float32),
            overlap_count=data["overlap"].astype(np.float32),
            sign_conflict_count=data["sign_conflict_count"].astype(np.float32),
            dominant_cell_id=data["dominant_id"].astype(np.int32),
            out_of_range_mask=np.any(data["out_of_range"], axis=2).astype(np.uint8),
        )


def _save_per_cell_buffers(
    *,
    variant_name: str,
    selected_ids: Sequence[int],
    contributions: Dict[int, List[preview.Contribution]],
    output_dir: Path,
) -> None:
    root = output_dir / "buffers" / "per_cell" / variant_name
    for cluster_id in selected_ids:
        for contrib in contributions.get(int(cluster_id), []):
            luma = _luma(contrib.pred_hp)
            stem_dir = root / contrib.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                stem_dir / f"cluster_{int(cluster_id):05d}.npz",
                cluster_id=np.asarray([int(cluster_id)], dtype=np.int32),
                split=np.asarray([contrib.split]),
                roi=np.asarray(contrib.roi, dtype=np.int32),
                residual=contrib.pred_hp.astype(np.float32),
                residual_raw=contrib.pred_raw.astype(np.float32),
                residual_lp=contrib.pred_lp.astype(np.float32),
                q_parent=contrib.q_parent.astype(np.float32),
                support=contrib.fit_w.astype(np.float32),
                off_support=contrib.off_w.astype(np.float32),
                positive_lobe=np.maximum(luma, 0.0).astype(np.float32),
                negative_lobe=np.maximum(-luma, 0.0).astype(np.float32),
            )


def _cell_record(row: Dict[str, object], rank: int) -> Dict[str, object]:
    keep_keys = [
        "cluster_id",
        "best_piece",
        "best_scale",
        "best_orientation_deg",
        "best_phase",
        "C_beta",
        "C_selection_gain",
        "C_fit_gain",
        "C_test_gain",
        "C_fit_active_area",
        "C_selection_active_area",
        "C_test_active_area",
        "selection_rank",
        "deploy_rank",
    ]
    record = {"rank": int(rank)}
    for key in keep_keys:
        if key in row:
            record[key] = row[key]
    record["frozen_row"] = row
    return record


def _write_cell_file(path: Path, name: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    payload = {
        "version": VERSION,
        "name": name,
        "count": int(len(rows)),
        "cluster_ids": [preview._row_cluster_id(row) for row in rows],
        "rows_sha256": _sha256_json(rows),
        "cells": [_cell_record(row, rank=i) for i, row in enumerate(rows)],
    }
    _write_json(path, payload)
    return payload


def _frozen_cells_3d_payload(
    *,
    rows_by_id: Dict[int, Dict[str, object]],
    state: Dict[str, object],
) -> Dict[str, object]:
    clusters = state["clusters"]
    results = state["results"]
    views = state["views"]
    cells: List[Dict[str, object]] = []
    for cluster_id in sorted(rows_by_id):
        if cluster_id < 0 or cluster_id >= len(results) or cluster_id >= len(clusters):
            continue
        result = results[cluster_id]
        cluster = clusters[cluster_id]
        if not cluster:
            continue
        source_slot, source_local_index = cluster[0]
        source_view = views[source_slot]
        cell = {
            "cluster_id": int(cluster_id),
            "xyz": np.asarray(result.xyz, dtype=np.float32).reshape(3).tolist(),
            "normal": np.asarray(result.normal, dtype=np.float32).reshape(3).tolist(),
            "status": str(result.status),
            "score": float(result.score),
            "reproj_rms": float(result.reproj_rms),
            "max_center_std": float(result.max_center_std),
            "hessian_cond": float(result.hessian_cond),
            "mode_dominance": float(result.mode_dominance),
            "mode_entropy": float(result.mode_entropy),
            "parent_index": int(result.parent_index),
            "cluster_size": int(len(cluster)),
            "source_view": {
                "view_slot": int(source_slot),
                "view_index": int(source_view.view_index),
                "stem": str(source_view.stem),
                "source_size": [int(source_view.source_size[0]), int(source_view.source_size[1])],
                "camera": source_view.camera,
                "local_primitive_index": int(source_local_index),
                "original_primitive_id": int(source_view.original_primitive_id[source_local_index]),
                "center": np.asarray(source_view.mu_xy[source_local_index], dtype=np.float32).reshape(2).tolist(),
                "theta": float(source_view.theta[source_local_index]),
                "long_px": float(source_view.long_px[source_local_index]),
                "short_px": float(source_view.short_px[source_local_index]),
                "q": float(source_view.q[source_local_index]),
            },
            "frozen_row": rows_by_id[cluster_id],
        }
        cells.append(cell)
    payload = {
        "version": VERSION,
        "description": "Frozen 3D cell geometry and source image frame for Level-1 lockbox projection.",
        "count": int(len(cells)),
        "cluster_ids": [int(cell["cluster_id"]) for cell in cells],
        "cells": cells,
    }
    payload["sha256"] = _sha256_json(payload["cells"])
    return payload


def _renderer_config(args: argparse.Namespace, oracle_args: SimpleNamespace) -> Dict[str, object]:
    return {
        "version": VERSION,
        "rgb_domain": "linear RGB float32",
        "base_policy": "frozen; never modified by static V1",
        "formula": "I = I_base + sum_k q_parent_k * projected_signed_piece_k * beta_k",
        "bounded_formula": (
            "per-cell clip before summation to match preview-v0"
            if args.bounded_mode == "per_cell_clip"
            else "post-sum clip of Delta_raw"
        ),
        "bounded_mode": args.bounded_mode,
        "bounded_delta_clip": float(args.bounded_delta_clip),
        "minimal_clean_drop_cluster_ids": [
            int(x) for x in oracle._parse_csv(args.minimal_clean_drop_cluster_ids, int) if int(x) >= 0
        ],
        "variant_cell_sets": {
            "core28": "cells_core28.json",
            "deploy_top40_raw": "cells_deploy_top40_raw.json",
            "deploy_top40_bounded": "same ordered cells as deploy_top40_raw plus fixed bounded_formula",
            "deploy_top40_minimal_clean_dev": "cells_minimal_clean_dev.json",
        },
        "support_definitions": {
            "variant_support": "current variant max fit_w",
            "raw_union_support": "deploy_top40_raw max fit_w; primary fair comparison support",
            "global_eligible_support": "all valid oracle cells max fit_w; broad fixed support",
        },
        "target_definitions": {
            "target_hf": "HP(SR - base)",
            "highpass_kernel": int(getattr(oracle_args, "highpass_kernel", 9)),
            "lowpass_kernel": int(getattr(oracle_args, "lowpass_kernel", 21)),
            "target_mask": "oracle fit/support weight maps",
        },
        "q_parent": {
            "source": _path_or_empty(getattr(oracle_args, "q_parent_dir", "")),
            "policy": "recomputed/resolved per view from frozen oracle inputs",
        },
        "forbidden_in_static_v1": [
            "beta refit",
            "per-view optimal parameters",
            "base parameter update",
            "support retuning",
            "cell deletion beyond frozen variant lists",
        ],
        "frozen_3d_cells": "frozen_cells_3d.json",
    }


def _manifest(
    *,
    oracle_dir: Path,
    output_dir: Path,
    oracle_summary: Dict[str, object],
    oracle_args: SimpleNamespace,
    cells: Dict[str, Dict[str, object]],
    config: Dict[str, object],
    frozen_cells_3d: Dict[str, object],
    stems: Sequence[str],
) -> Dict[str, object]:
    params = dict(oracle_summary.get("params", {}))
    return {
        "version": VERSION,
        "frozen": True,
        "oracle_dir": str(oracle_dir),
        "output_dir": str(output_dir),
        "oracle_summary_sha256": _sha256_json(oracle_summary),
        "base_checkpoint": {
            "model_dir": _path_or_empty(getattr(oracle_args, "base_model_dir", params.get("base_model_dir", ""))),
            "iteration": int(getattr(oracle_args, "base_iteration", params.get("base_iteration", 30000))),
        },
        "camera_parameters": {
            "source": "base model cameras.json / COLMAP camera order frozen by oracle",
            "match_policy": str(getattr(oracle_args, "match_policy", "order_if_needed")),
            "view_stems": list(stems),
        },
        "inputs": {
            "primitive_dir": _path_or_empty(getattr(oracle_args, "primitive_dir", "")),
            "base_render_dir": _path_or_empty(getattr(oracle_args, "base_render_dir", "")),
            "sr_dir": _path_or_empty(getattr(oracle_args, "sr_dir", "")),
            "weight_dir": _path_or_empty(getattr(oracle_args, "weight_dir", "")),
            "q_parent_dir": _path_or_empty(getattr(oracle_args, "q_parent_dir", "")),
        },
        "cell_sets": {
            name: {
                "count": payload["count"],
                "cluster_ids": payload["cluster_ids"],
                "rows_sha256": payload["rows_sha256"],
            }
            for name, payload in cells.items()
        },
        "frozen_cells_3d": {
            "file": "frozen_cells_3d.json",
            "count": int(frozen_cells_3d.get("count", 0)),
            "sha256": str(frozen_cells_3d.get("sha256", "")),
        },
        "renderer_config_sha256": _sha256_json(config),
        "analysis_test_policy": "read-only development set; no further deletion/top-k/bounded tuning allowed",
    }


def _compose_variants(
    *,
    rows: Dict[str, object],
    contributions: Dict[int, List[preview.Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    oracle_args: SimpleNamespace,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, object]]:
    deploy_ids = rows["deploy_ids"]  # type: ignore[assignment]
    core_ids = rows["core_ids"]  # type: ignore[assignment]
    minimal_ids = rows["minimal_clean_ids"]  # type: ignore[assignment]
    variants = {
        "core28": {"ids": core_ids, "rows": rows["core_rows"], "bounded_clip": None},
        "deploy_top40_raw": {"ids": deploy_ids, "rows": rows["deploy_rows"], "bounded_clip": None},
        "deploy_top40_bounded": {
            "ids": deploy_ids,
            "rows": rows["deploy_rows"],
            "bounded_clip": float(args.bounded_delta_clip),
        },
        "deploy_top40_minimal_clean_dev": {
            "ids": minimal_ids,
            "rows": rows["minimal_clean_rows"],
            "bounded_clip": None,
        },
    }
    out: Dict[str, Dict[str, object]] = {}
    for name, spec in variants.items():
        bounded_clip = spec["bounded_clip"]
        use_bounded_clip = bounded_clip if bounded_clip is not None else None
        out[name] = {
            "ids": list(spec["ids"]),
            "rows": list(spec["rows"]),
            "all": _static_compose(
                selected_ids=spec["ids"],
                contributions=contributions,
                stems=stems,
                state=state,
                args=oracle_args,
                bounded_clip=use_bounded_clip,
                bounded_mode=args.bounded_mode,
            ),
            "by_split": {},
        }
        for split in SPLITS:
            out[name]["by_split"][split] = _static_compose(
                selected_ids=spec["ids"],
                contributions=contributions,
                stems=stems,
                state=state,
                args=oracle_args,
                split_filter=split,
                bounded_clip=use_bounded_clip,
                bounded_mode=args.bounded_mode,
            )
    return out


def _copy_to_check(output_dir: Path, check_dir: Path) -> None:
    if not str(check_dir):
        return
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "v1_manifest.json",
        "renderer_config.json",
        "cells_deploy_top40_raw.json",
        "cells_core28.json",
        "cells_minimal_clean_dev.json",
        "frozen_cells_3d.json",
        "static_v1_metrics.json",
        "numeric_closure.json",
        "README.txt",
    ):
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
        "visuals/core28/residual_pred",
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
    args = _parse_args()
    oracle_dir = Path(args.oracle_dir).expanduser().resolve()
    summary_path = oracle_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    for required in ("deploy_selected_rows.json", "core_selected_rows.json", "rows.json"):
        path = oracle_dir / required
        if not path.exists():
            raise FileNotFoundError(f"required oracle file not found: {path}")

    oracle_summary = preview._read_json(summary_path)
    oracle_args = preview._load_oracle_args(oracle_summary)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else oracle_dir / "static_v1"
    check_dir = Path(args.check_dir).expanduser().resolve() if args.check_dir else Path("")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minimal_drop_ids = [int(x) for x in oracle._parse_csv(args.minimal_clean_drop_cluster_ids, int) if int(x) >= 0]
    rows = _load_rows(oracle_dir, minimal_drop_ids)
    focus_ids = [int(x) for x in oracle._parse_csv(args.focus_cluster_ids, int) if int(x) >= 0]
    all_needed_ids = sorted(
        set(rows["valid_ids"])
        | set(rows["deploy_ids"])
        | set(rows["core_ids"])
        | set(rows["minimal_clean_ids"])
        | set(focus_ids)
        | set(minimal_drop_ids)
    )

    print(f"[residual-static-v1] oracle : {oracle_dir}")
    print(f"[residual-static-v1] output : {output_dir}")
    print(f"[residual-static-v1] deploy : {len(rows['deploy_ids'])} cells")
    print(f"[residual-static-v1] core   : {len(rows['core_ids'])} cells")
    print(f"[residual-static-v1] bounded: {args.bounded_mode} clip={float(args.bounded_delta_clip):.4f}")

    state = preview._load_scene_state(oracle_args)
    stems = [view.stem for view in state["views"]]
    obs_by_cluster = preview._build_obs_for_clusters(cluster_ids=all_needed_ids, state=state, args=oracle_args)
    rows_by_id: Dict[int, Dict[str, object]] = rows["rows_by_id"]  # type: ignore[assignment]
    row_union = [rows_by_id[cid] for cid in all_needed_ids if cid in rows_by_id]
    contributions = preview._build_contributions(row_union, obs_by_cluster, oracle_args)

    renderer_config = _renderer_config(args, oracle_args)
    cells = {
        "deploy_top40_raw": _write_cell_file(output_dir / "cells_deploy_top40_raw.json", "deploy_top40_raw", rows["deploy_rows"]),  # type: ignore[arg-type]
        "core28": _write_cell_file(output_dir / "cells_core28.json", "core28", rows["core_rows"]),  # type: ignore[arg-type]
        "deploy_top40_minimal_clean_dev": _write_cell_file(
            output_dir / "cells_minimal_clean_dev.json",
            "deploy_top40_minimal_clean_dev",
            rows["minimal_clean_rows"],  # type: ignore[arg-type]
        ),
    }
    _write_json(output_dir / "renderer_config.json", renderer_config)
    frozen_cells_3d = _frozen_cells_3d_payload(rows_by_id=rows_by_id, state=state)
    _write_json(output_dir / "frozen_cells_3d.json", frozen_cells_3d)
    manifest = _manifest(
        oracle_dir=oracle_dir,
        output_dir=output_dir,
        oracle_summary=oracle_summary,
        oracle_args=oracle_args,
        cells=cells,
        config=renderer_config,
        frozen_cells_3d=frozen_cells_3d,
        stems=stems,
    )
    _write_json(output_dir / "v1_manifest.json", manifest)

    preview._save_shared_visuals(stems=stems, state=state, oracle_args=oracle_args, output_dir=output_dir, args=args)

    raw_support_by_split: Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]] = {}
    global_support_by_split: Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]] = {}
    for split_key in [None, *SPLITS]:
        raw_support_by_split[split_key] = preview._support_map_from_composed(
            _static_compose(
                selected_ids=rows["deploy_ids"],  # type: ignore[arg-type]
                contributions=contributions,
                stems=stems,
                state=state,
                args=oracle_args,
                split_filter=split_key,
                bounded_clip=None,
                bounded_mode=args.bounded_mode,
            )
        )
        global_support_by_split[split_key] = preview._support_map_from_composed(
            _static_compose(
                selected_ids=rows["valid_ids"],  # type: ignore[arg-type]
                contributions=contributions,
                stems=stems,
                state=state,
                args=oracle_args,
                split_filter=split_key,
                bounded_clip=None,
                bounded_mode=args.bounded_mode,
            )
        )

    variants = _compose_variants(
        rows=rows,
        contributions=contributions,
        stems=stems,
        state=state,
        oracle_args=oracle_args,
        args=args,
    )

    metrics: Dict[str, object] = {"version": VERSION, "variants": {}}
    for name, spec in variants.items():
        composed_all = spec["all"]
        preview._save_variant_visuals(
            variant_name=name,
            composed=composed_all,  # type: ignore[arg-type]
            output_dir=output_dir,
            args=args,
            lowpass_kernel=int(oracle_args.lowpass_kernel),
        )
        _save_extra_visuals(variant_name=name, composed=composed_all, output_dir=output_dir, args=args)  # type: ignore[arg-type]
        if int(args.write_buffers):
            _save_buffers(variant_name=name, composed=composed_all, output_dir=output_dir)  # type: ignore[arg-type]
        if int(args.write_per_cell_buffers) and name == str(args.per_cell_buffer_variant):
            _save_per_cell_buffers(
                variant_name=name,
                selected_ids=spec["ids"],  # type: ignore[arg-type]
                contributions=contributions,
                output_dir=output_dir,
            )
        metrics["variants"][name] = _variant_report_static(
            name=name,
            rows=spec["rows"],  # type: ignore[arg-type]
            composed_all=composed_all,  # type: ignore[arg-type]
            composed_by_split=spec["by_split"],  # type: ignore[arg-type]
            raw_support_by_split=raw_support_by_split,
            global_support_by_split=global_support_by_split,
            lowpass_kernel=int(oracle_args.lowpass_kernel),
        )

    closure = _closure_checks(
        deploy_ids=rows["deploy_ids"],  # type: ignore[arg-type]
        minimal_ids=rows["minimal_clean_ids"],  # type: ignore[arg-type]
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        bounded_clip=float(args.bounded_delta_clip),
        bounded_mode=args.bounded_mode,
        single_cluster_id=int(args.closure_single_cluster_id),
        small_count=int(args.closure_small_count),
    )
    _write_json(output_dir / "static_v1_metrics.json", metrics)
    _write_json(output_dir / "numeric_closure.json", closure)
    _write_json(
        output_dir / "README.txt",
        {
            "summary": "Static V1 frozen residual renderer. Use numeric_closure.json before lockbox.",
            "formal_variants": list(MAIN_VARIANTS),
            "diagnostic_buffers": "buffers/<variant>/<stem>.npz",
            "per_cell_buffers": f"buffers/per_cell/{args.per_cell_buffer_variant}/<stem>/cluster_XXXXX.npz",
            "frozen_3d_cells": "frozen_cells_3d.json",
            "acceptance": {
                "float_mae": "< 1e-5",
                "max_error": "< 1e-4",
                "main_metric_diff": "< 1e-3",
            },
        },
    )
    if str(check_dir):
        _copy_to_check(output_dir, check_dir)
    print(f"[residual-static-v1] manifest: {output_dir / 'v1_manifest.json'}")
    print(f"[residual-static-v1] metrics : {output_dir / 'static_v1_metrics.json'}")
    print(f"[residual-static-v1] closure : {output_dir / 'numeric_closure.json'}")
    if str(check_dir):
        print(f"[residual-static-v1] check   : {check_dir}")


if __name__ == "__main__":
    main()
