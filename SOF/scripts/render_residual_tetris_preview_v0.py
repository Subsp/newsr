#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_residual_tetris_oracle_v0 as oracle  # noqa: E402


SPLITS = ("fit", "selection", "test")
SPLIT_DISPLAY = {"fit": "fit", "selection": "selection", "test": "analysis_test"}


@dataclass
class Contribution:
    cluster_id: int
    stem: str
    split: str
    roi: Tuple[int, int, int, int]
    pred_hp: np.ndarray
    pred_raw: np.ndarray
    pred_lp: np.ndarray
    fit_w: np.ndarray
    off_w: np.ndarray
    abs_luma: np.ndarray
    q_parent: np.ndarray
    target_hf: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline residual-tetris deploy preview. This script freezes oracle rows "
            "and jointly composes q_parent * projected_signed_piece * beta without "
            "any per-view refit."
        )
    )
    parser.add_argument("--oracle_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--check_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bounded_delta_clip", type=float, default=0.08)
    parser.add_argument("--visual_signed_scale", type=float, default=4.0)
    parser.add_argument("--error_scale", type=float, default=8.0)
    parser.add_argument("--lp_scale", type=float, default=16.0)
    parser.add_argument("--leak_scale", type=float, default=24.0)
    parser.add_argument("--dose_counts", default="5,10,20,40")
    parser.add_argument("--focus_cluster_ids", default="64,174,70,133,156,223")
    parser.add_argument("--cell_sheet_max_obs", type=int, default=3)
    parser.add_argument("--clean_negative_view_ratio", type=float, default=0.5)
    parser.add_argument("--clean_lp_drift_marginal_min", type=float, default=0.0)
    parser.add_argument("--clean_leakage_marginal_min", type=float, default=0.0)
    parser.add_argument("--minimal_clean_drop_cluster_ids", default="165")
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_to_check(output_dir: Path, check_dir: Path) -> None:
    if not check_dir:
        return
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary.json",
        "joint_metrics.json",
        "dose_curve.json",
        "leave_one_cell_out.json",
        "per_cell_marginals.json",
        "minimal_clean_selected_rows.json",
        "clean_selected_rows.json",
        "clean_rejected_rows.json",
        "negative_view_diagnostics.json",
        "README.txt",
    ):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, check_dir / name)
    for rel in (
        "visuals/deploy_top40_raw/base_plus_residual",
        "visuals/deploy_top40_raw/residual_pred",
        "visuals/deploy_top40_raw/error_after",
        "visuals/deploy_top40_bounded/base_plus_residual",
        "visuals/deploy_top40_bounded/residual_pred",
        "visuals/deploy_top40_bounded/error_after",
        "visuals/deploy_top40_minimal_clean_dev/base_plus_residual",
        "visuals/deploy_top40_minimal_clean_dev/residual_pred",
        "visuals/deploy_top40_minimal_clean_dev/error_after",
        "visuals/deploy_top40_clean29/base_plus_residual",
        "visuals/deploy_top40_clean29/residual_pred",
        "visuals/deploy_top40_clean29/error_after",
        "visuals/core28/base_plus_residual",
        "visuals/base",
        "visuals/target_hf",
        "negative_view_diagnostics",
        "cell_sheet",
    ):
        src_dir = output_dir / rel
        if not src_dir.is_dir():
            continue
        dst_dir = check_dir / rel
        if rel == "negative_view_diagnostics":
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(src_dir.glob("*.png"))[:64]:
            shutil.copy2(image_path, dst_dir / image_path.name)


def _json_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _row_cluster_id(row: Dict[str, object]) -> int:
    return int(row["cluster_id"])


def _candidate_from_row(row: Dict[str, object]) -> oracle.Candidate:
    return oracle.Candidate(
        piece=str(row["best_piece"]),
        scale=float(row["best_scale"]),
        orientation_deg=float(row["best_orientation_deg"]),
        phase=int(row["best_phase"]),
    )


def _beta_from_row(row: Dict[str, object]) -> np.ndarray:
    return np.asarray(row["C_beta"], dtype=np.float32).reshape(3)


def _load_oracle_args(summary: Dict[str, object]) -> SimpleNamespace:
    params = dict(summary.get("params", {}))
    for key in (
        "base_model_dir",
        "base_iteration",
        "primitive_dir",
        "base_render_dir",
        "sr_dir",
        "weight_dir",
        "q_parent_dir",
        "match_policy",
        "limit",
    ):
        if key in summary and key not in params:
            params[key] = summary[key]
    params.setdefault("q_parent_dir", summary.get("q_parent_dir") or "")
    params.setdefault("match_policy", "order_if_needed")
    params.setdefault("limit", summary.get("num_views", 8))
    return SimpleNamespace(**params)


def _load_scene_state(args: SimpleNamespace):
    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    base_ply = base_model_dir / "point_cloud" / f"iteration_{int(args.base_iteration)}" / "point_cloud.ply"
    primitive_dir = Path(args.primitive_dir).expanduser().resolve()
    base_render_dir = Path(args.base_render_dir).expanduser().resolve()
    sr_dir = Path(args.sr_dir).expanduser().resolve()
    weight_dir = Path(args.weight_dir).expanduser().resolve()
    q_parent_dir = Path(args.q_parent_dir).expanduser().resolve() if str(getattr(args, "q_parent_dir", "")).strip() else None

    primitive_paths = oracle._list_files(primitive_dir, [".npz"])
    if int(getattr(args, "limit", 0)) > 0:
        primitive_paths = primitive_paths[: int(args.limit)]
    base_paths = oracle._list_files(base_render_dir)
    sr_paths = oracle._list_files(sr_dir)
    weight_paths = oracle._list_files(weight_dir)
    q_paths = oracle._list_files(q_parent_dir) if q_parent_dir is not None and q_parent_dir.is_dir() else []

    cameras = oracle._load_cameras(base_model_dir)
    _base_vertices, base_xyz, base_opacity, base_rgb = oracle._load_base_vertices(base_ply)
    spray_args = oracle._make_spray_args(args)

    carrier_rgb_paths: List[Path] = []
    carrier_render_paths: List[Path] = []
    if str(getattr(args, "carrier_rgb_dir", "")).strip() and Path(args.carrier_rgb_dir).is_dir():
        carrier_rgb_paths = oracle._list_files(Path(args.carrier_rgb_dir))
    if str(getattr(args, "carrier_render_dir", "")).strip() and Path(args.carrier_render_dir).is_dir():
        carrier_render_paths = oracle._list_files(Path(args.carrier_render_dir))
    carrier_rgb_lookup = oracle._image_lookup(carrier_rgb_paths)
    carrier_render_lookup = oracle._image_lookup(carrier_render_paths)
    weight_lookup_for_loader = oracle._image_lookup(weight_paths)

    views: List[oracle.ViewPrimitiveSet] = []
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
        if view is None:
            print(f"[residual-preview-v0] skip primitive {primitive_path.stem}: {info.get('status')}")
            continue
        views.append(view)

    clusters = oracle._greedy_clusters(views, spray_args)
    results = [oracle._cluster_to_result(views, cluster, base_xyz, base_rgb, spray_args) for cluster in clusters]

    return {
        "base_model_dir": base_model_dir,
        "primitive_dir": primitive_dir,
        "base_render_dir": base_render_dir,
        "sr_dir": sr_dir,
        "weight_dir": weight_dir,
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
        "clusters": clusters,
        "results": results,
    }


def _build_obs_for_clusters(
    *,
    cluster_ids: Iterable[int],
    state: Dict[str, object],
    args: SimpleNamespace,
) -> Dict[int, List[oracle.EvalObs]]:
    out: Dict[int, List[oracle.EvalObs]] = {}
    clusters: Sequence[Sequence[Tuple[int, int]]] = state["clusters"]  # type: ignore[assignment]
    results = state["results"]
    views = state["views"]
    for cid in sorted(set(int(x) for x in cluster_ids)):
        if cid < 0 or cid >= len(clusters):
            continue
        obs, _stats = oracle._build_eval_obs(
            cluster_obs=clusters[cid],
            result=results[cid],
            views=views,
            base_paths=state["base_paths"],
            base_lookup=state["base_lookup"],
            sr_paths=state["sr_paths"],
            sr_lookup=state["sr_lookup"],
            weight_paths=state["weight_paths"],
            weight_lookup=state["weight_lookup"],
            q_paths=state["q_paths"],
            q_lookup=state["q_lookup"],
            args=args,
        )
        out[cid] = obs
    return out


def _build_contributions(rows: Sequence[Dict[str, object]], obs_by_cluster: Dict[int, List[oracle.EvalObs]], args: SimpleNamespace) -> Dict[int, List[Contribution]]:
    out: Dict[int, List[Contribution]] = {}
    for row in rows:
        cid = _row_cluster_id(row)
        cand = _candidate_from_row(row)
        beta = _beta_from_row(row)
        row_contribs: List[Contribution] = []
        for obs in obs_by_cluster.get(cid, []):
            basis, target, fit_w, off_w, raw_basis, lp_basis = oracle._basis_arrays(obs, cand, args, "C", None)
            pred_hp = basis[..., None] * beta[None, None, :]
            pred_raw = raw_basis[..., None] * beta[None, None, :]
            pred_lp = lp_basis[..., None] * beta[None, None, :]
            abs_luma = np.abs(np.sum(pred_hp * oracle.LUMA[None, None, :], axis=2)).astype(np.float32)
            row_contribs.append(
                Contribution(
                    cluster_id=cid,
                    stem=obs.stem,
                    split=obs.split,
                    roi=obs.roi,
                    pred_hp=pred_hp.astype(np.float32),
                    pred_raw=pred_raw.astype(np.float32),
                    pred_lp=pred_lp.astype(np.float32),
                    fit_w=fit_w.astype(np.float32),
                    off_w=off_w.astype(np.float32),
                    abs_luma=abs_luma,
                    q_parent=obs.q_parent.astype(np.float32),
                    target_hf=target.astype(np.float32),
                )
            )
        out[cid] = row_contribs
    return out


def _target_hf_for_stem(stem: str, state: Dict[str, object], args: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    view_index = int(state.get("view_index_by_stem", {}).get(stem, 0))
    base_path = oracle._resolve_path(
        state["base_paths"],
        state["base_lookup"],
        stem,
        view_index,
        str(args.match_policy),
    )
    base = oracle._load_rgb(base_path)
    sr_path = oracle._resolve_path(
        state["sr_paths"],
        state["sr_lookup"],
        stem,
        view_index,
        str(args.match_policy),
    )
    sr = oracle._load_rgb(sr_path, size=(base.shape[1], base.shape[0]))
    target = oracle._highpass(sr - base, int(args.highpass_kernel)).astype(np.float32)
    return base.astype(np.float32), sr.astype(np.float32), target


def _empty_canvas(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.float32)


def _add_roi(canvas: np.ndarray, roi: Tuple[int, int, int, int], value: np.ndarray, sign: float = 1.0) -> None:
    x0, y0, x1, y1 = roi
    canvas[y0:y1, x0:x1] += float(sign) * value


def _max_roi(canvas: np.ndarray, roi: Tuple[int, int, int, int], value: np.ndarray) -> None:
    x0, y0, x1, y1 = roi
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], value)


def _compose(
    *,
    selected_ids: Sequence[int],
    contributions: Dict[int, List[Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    split_filter: Optional[str] = None,
    skip_cluster_id: Optional[int] = None,
    bounded_clip: Optional[float] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    selected = set(int(x) for x in selected_ids)
    if skip_cluster_id is not None:
        selected.discard(int(skip_cluster_id))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for stem in stems:
        base, _sr, target = _target_hf_for_stem(stem, state, args)
        h, w = base.shape[:2]
        pred = _empty_canvas((h, w))
        pred_raw = _empty_canvas((h, w))
        pred_lp = _empty_canvas((h, w))
        weight = np.zeros((h, w), dtype=np.float32)
        off_weight = np.zeros((h, w), dtype=np.float32)
        overlap = np.zeros((h, w), dtype=np.float32)
        q_parent_peak = np.zeros((h, w), dtype=np.float32)
        dominant_val = np.zeros((h, w), dtype=np.float32)
        dominant_id = np.full((h, w), -1, dtype=np.int32)
        for cid in selected:
            for contrib in contributions.get(cid, []):
                if contrib.stem != stem:
                    continue
                if split_filter is not None and contrib.split != split_filter:
                    continue
                value = contrib.pred_hp
                if bounded_clip is not None:
                    value = np.clip(value, -float(bounded_clip), float(bounded_clip))
                _add_roi(pred, contrib.roi, value)
                _add_roi(pred_raw, contrib.roi, contrib.pred_raw)
                _add_roi(pred_lp, contrib.roi, contrib.pred_lp)
                _max_roi(weight, contrib.roi, contrib.fit_w)
                _max_roi(off_weight, contrib.roi, contrib.off_w)
                x0, y0, x1, y1 = contrib.roi
                active = contrib.fit_w > float(args.core_weight_threshold)
                overlap[y0:y1, x0:x1] += active.astype(np.float32)
                q_parent_peak[y0:y1, x0:x1] = np.maximum(q_parent_peak[y0:y1, x0:x1], contrib.q_parent)
                dominant_slice = dominant_val[y0:y1, x0:x1]
                dominant_id_slice = dominant_id[y0:y1, x0:x1]
                better = contrib.abs_luma > dominant_slice
                dominant_slice[better] = contrib.abs_luma[better]
                dominant_id_slice[better] = int(cid)
        preview = np.clip(base + pred, 0.0, 1.0)
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
            "preview": preview,
            "out_of_range": ((base + pred) < 0.0) | ((base + pred) > 1.0),
        }
    return out


def _support_map_from_composed(composed: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        stem: {
            "weight": np.asarray(data["weight"], dtype=np.float32).copy(),
            "off_weight": np.asarray(data["off_weight"], dtype=np.float32).copy(),
        }
        for stem, data in composed.items()
    }


def _support_for_data(
    data: Dict[str, np.ndarray],
    support_map: Optional[Dict[str, Dict[str, np.ndarray]]],
    stem: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if support_map is None or stem not in support_map:
        return np.asarray(data["weight"], dtype=np.float32), np.asarray(data["off_weight"], dtype=np.float32)
    support = support_map[stem]
    return np.asarray(support["weight"], dtype=np.float32), np.asarray(support.get("off_weight", data["off_weight"]), dtype=np.float32)


def _metrics_for_view(
    data: Dict[str, np.ndarray],
    lowpass_kernel: int = 21,
    support_weight: Optional[np.ndarray] = None,
    off_support_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    target = data["target_hf"]
    pred = data["pred"]
    w_src = data["weight"] if support_weight is None else support_weight
    ow_src = data["off_weight"] if off_support_weight is None else off_support_weight
    w = np.clip(w_src, 0.0, 1.0)
    ow = np.clip(ow_src, 0.0, 1.0)
    fw = w[..., None]
    ow3 = ow[..., None]
    delta_lp = oracle._box_blur(pred, int(lowpass_kernel))
    te = float(np.sum(fw * target * target))
    re = float(np.sum(fw * (target - pred) * (target - pred)))
    pe = float(np.sum(fw * pred * pred))
    oe = float(np.sum(ow3 * pred * pred))
    le = float(np.sum(fw * delta_lp * delta_lp))
    active = w > 1e-8
    out_ratio = 0.0
    if np.any(active):
        out = np.any(data["out_of_range"], axis=2)
        out_ratio = float(np.count_nonzero(out & active) / max(float(np.count_nonzero(active)), 1.0))
    pred_luma = np.sum(pred * oracle.LUMA[None, None, :], axis=2)
    target_luma = np.sum(target * oracle.LUMA[None, None, :], axis=2)
    corr = oracle._weighted_corr(pred_luma, target_luma, w)
    return {
        "active": bool(te > 1e-12),
        "HF_corr": float("nan") if corr is None else float(corr),
        "explained_energy": float(1.0 - re / max(te, 1e-10)) if te > 0 else float("-inf"),
        "gain": float(te - re),
        "target_energy": te,
        "residual_energy": re,
        "pred_energy": pe,
        "off_target_leakage": float(oe / max(pe, 1e-10)),
        "LP_drift": float(le / max(te, 1e-10)),
        "out_of_range_ratio": out_ratio,
    }


def _per_view_metrics_for_composed(
    composed: Dict[str, Dict[str, np.ndarray]],
    lowpass_kernel: int = 21,
    support_map: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for stem, data in composed.items():
        support_weight, off_support_weight = _support_for_data(data, support_map, stem)
        out[stem] = _metrics_for_view(
            data,
            lowpass_kernel=lowpass_kernel,
            support_weight=support_weight,
            off_support_weight=off_support_weight,
        )
    return out


def _finite_median(values: Sequence[float], default: float = 0.0) -> float:
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def _metrics_for_composed(
    composed: Dict[str, Dict[str, np.ndarray]],
    lowpass_kernel: int = 21,
    support_map: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
    target_energy = 0.0
    residual_energy = 0.0
    pred_energy = 0.0
    off_energy = 0.0
    lp_energy = 0.0
    out_ratio_num = 0.0
    out_ratio_den = 0.0
    view_gains = []
    pred_luma: List[np.ndarray] = []
    target_luma: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    for stem, data in composed.items():
        target = data["target_hf"]
        pred = data["pred"]
        support_weight, off_support_weight = _support_for_data(data, support_map, stem)
        w = np.clip(support_weight, 0.0, 1.0)
        view_metrics = _metrics_for_view(
            data,
            lowpass_kernel=lowpass_kernel,
            support_weight=support_weight,
            off_support_weight=off_support_weight,
        )
        te = float(view_metrics["target_energy"])
        re = float(view_metrics["residual_energy"])
        pe = float(view_metrics["pred_energy"])
        oe = float(view_metrics["off_target_leakage"]) * max(pe, 1e-10)
        le = float(view_metrics["LP_drift"]) * max(te, 1e-10)
        target_energy += te
        residual_energy += re
        pred_energy += pe
        off_energy += oe
        lp_energy += le
        if te > 1e-12:
            view_gains.append(te - re)
        active = w > 1e-8
        if np.any(active):
            out_ratio_num += float(view_metrics["out_of_range_ratio"]) * float(np.count_nonzero(active))
            out_ratio_den += float(np.count_nonzero(active))
        pred_luma.append(np.sum(pred * oracle.LUMA[None, None, :], axis=2))
        target_luma.append(np.sum(target * oracle.LUMA[None, None, :], axis=2))
        weights.append(w)
    if not composed:
        return {}
    pred_flat = np.concatenate([x.reshape(-1) for x in pred_luma], axis=0)
    target_flat = np.concatenate([x.reshape(-1) for x in target_luma], axis=0)
    weight_flat = np.concatenate([x.reshape(-1) for x in weights], axis=0)
    corr = oracle._weighted_corr(pred_flat, target_flat, weight_flat)
    gain = target_energy - residual_energy
    return {
        "views_total": float(len(composed)),
        "views_active": float(len(view_gains)),
        "HF_corr": float("nan") if corr is None else float(corr),
        "explained_energy": float(1.0 - residual_energy / max(target_energy, 1e-10)) if target_energy > 0 else float("-inf"),
        "gain": float(gain),
        "target_energy": float(target_energy),
        "residual_energy": float(residual_energy),
        "pred_energy": float(pred_energy),
        "off_target_leakage": float(off_energy / max(pred_energy, 1e-10)),
        "LP_drift": float(lp_energy / max(target_energy, 1e-10)),
        "out_of_range_ratio": float(out_ratio_num / max(out_ratio_den, 1.0)),
        "positive_view_ratio": float(np.mean(np.asarray(view_gains, dtype=np.float32) > 0.0)) if view_gains else 0.0,
        "view_gain_min": float(np.min(view_gains)) if view_gains else float("nan"),
        "view_gain_mean": float(np.mean(view_gains)) if view_gains else float("nan"),
        "view_gain_sum": float(np.sum(view_gains)) if view_gains else float("nan"),
    }


def _individual_gain(rows: Sequence[Dict[str, object]], split: str) -> float:
    key = {"fit": "C_fit_gain", "selection": "C_selection_gain", "test": "C_test_gain"}[split]
    return float(sum(_json_float(row.get(key), 0.0) for row in rows))


def _positive_individual_gain(rows: Sequence[Dict[str, object]], split: str) -> float:
    key = {"fit": "C_fit_gain", "selection": "C_selection_gain", "test": "C_test_gain"}[split]
    return float(sum(max(_json_float(row.get(key), 0.0), 0.0) for row in rows))


def _variant_report(
    *,
    name: str,
    rows: Sequence[Dict[str, object]],
    selected_ids: Sequence[int],
    contributions: Dict[int, List[Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    bounded_clip: Optional[float] = None,
    raw_support_by_split: Optional[Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]]] = None,
    global_support_by_split: Optional[Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]]] = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "name": name,
        "cell_count": int(len(selected_ids)),
        "bounded_delta_clip": bounded_clip,
        "by_split": {},
    }
    all_composed = _compose(
        selected_ids=selected_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=args,
        split_filter=None,
        bounded_clip=bounded_clip,
    )
    report["all_views"] = _metrics_with_support_comparison(
        all_composed,
        lowpass_kernel=int(args.lowpass_kernel),
        raw_support=raw_support_by_split.get(None) if raw_support_by_split else None,
        global_support=global_support_by_split.get(None) if global_support_by_split else None,
    )
    for split in SPLITS:
        composed = _compose(
            selected_ids=selected_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=args,
            split_filter=split,
            bounded_clip=bounded_clip,
        )
        metrics = _metrics_with_support_comparison(
            composed,
            lowpass_kernel=int(args.lowpass_kernel),
            raw_support=raw_support_by_split.get(split) if raw_support_by_split else None,
            global_support=global_support_by_split.get(split) if global_support_by_split else None,
        )
        sum_individual = _individual_gain(rows, split)
        sum_positive_individual = _positive_individual_gain(rows, split)
        metrics["joint_gain"] = metrics.get("gain", float("nan"))
        metrics["sum_individual_gain"] = float(sum_individual)
        metrics["sum_positive_individual_gain"] = float(sum_positive_individual)
        metrics["joint_over_sum_individual_gain_signed"] = (
            float(metrics["joint_gain"] / sum_individual) if abs(sum_individual) > 1e-10 else float("nan")
        )
        metrics["joint_gain_capture_positive"] = float(metrics["joint_gain"] / max(sum_positive_individual, 1e-10))
        report["by_split"][SPLIT_DISPLAY[split]] = metrics
    return report


def _metrics_with_support_comparison(
    composed: Dict[str, Dict[str, np.ndarray]],
    *,
    lowpass_kernel: int,
    raw_support: Optional[Dict[str, Dict[str, np.ndarray]]],
    global_support: Optional[Dict[str, Dict[str, np.ndarray]]],
) -> Dict[str, object]:
    variant_metrics = _metrics_for_composed(composed, lowpass_kernel=lowpass_kernel)
    raw_metrics = _metrics_for_composed(composed, lowpass_kernel=lowpass_kernel, support_map=raw_support)
    global_metrics = _metrics_for_composed(composed, lowpass_kernel=lowpass_kernel, support_map=global_support)
    variant_metrics["gain_on_variant_support"] = variant_metrics.get("gain", float("nan"))
    variant_metrics["gain_on_raw_union_support"] = raw_metrics.get("gain", float("nan"))
    variant_metrics["gain_on_global_eligible_support"] = global_metrics.get("gain", float("nan"))
    variant_metrics["target_energy_on_variant_support"] = variant_metrics.get("target_energy", float("nan"))
    variant_metrics["target_energy_on_raw_union_support"] = raw_metrics.get("target_energy", float("nan"))
    variant_metrics["target_energy_on_global_eligible_support"] = global_metrics.get("target_energy", float("nan"))
    variant_metrics["support_metrics"] = {
        "variant_support": dict(variant_metrics),
        "raw_union_support": raw_metrics,
        "global_eligible_support": global_metrics,
    }
    return variant_metrics


def _save_variant_visuals(
    *,
    variant_name: str,
    composed: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    args: argparse.Namespace,
    lowpass_kernel: int,
) -> None:
    root = output_dir / "visuals" / variant_name
    for stem, data in composed.items():
        oracle._save_rgb(root / "base_plus_residual" / f"{stem}.png", data["preview"])
        oracle._save_rgb(
            root / "residual_pred" / f"{stem}.png",
            np.clip(data["pred"] * float(args.visual_signed_scale) + 0.5, 0.0, 1.0),
        )
        before = np.abs(data["target_hf"])
        after = np.abs(data["target_hf"] - data["pred"])
        oracle._save_rgb(root / "error_before" / f"{stem}.png", np.clip(before * float(args.error_scale), 0.0, 1.0))
        oracle._save_rgb(root / "error_after" / f"{stem}.png", np.clip(after * float(args.error_scale), 0.0, 1.0))
        lp = np.mean(np.abs(oracle._box_blur(data["pred"], int(lowpass_kernel))), axis=2)
        leak = np.mean(np.abs(data["pred"]), axis=2) * data["off_weight"]
        oracle._save_gray(root / "LP_drift" / f"{stem}.png", np.clip(lp * float(args.lp_scale), 0.0, 1.0))
        oracle._save_gray(root / "off_target_leakage" / f"{stem}.png", np.clip(leak * float(args.leak_scale), 0.0, 1.0))
        oracle._save_gray(root / "cell_overlap_count" / f"{stem}.png", np.clip(data["overlap"] / 8.0, 0.0, 1.0))
        oracle._save_rgb(root / "dominant_cell_id" / f"{stem}.png", _dominant_id_rgb(data["dominant_id"]))


def _save_shared_visuals(
    *,
    stems: Sequence[str],
    state: Dict[str, object],
    oracle_args: SimpleNamespace,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    for stem in stems:
        base, _sr, target = _target_hf_for_stem(stem, state, oracle_args)
        oracle._save_rgb(output_dir / "visuals" / "base" / f"{stem}.png", base)
        oracle._save_rgb(
            output_dir / "visuals" / "target_hf" / f"{stem}.png",
            np.clip(target * float(args.visual_signed_scale) + 0.5, 0.0, 1.0),
        )


def _dominant_id_rgb(dominant_id: np.ndarray) -> np.ndarray:
    ids = np.asarray(dominant_id, dtype=np.int64)
    out = np.zeros((*ids.shape, 3), dtype=np.float32)
    active = ids >= 0
    if not np.any(active):
        return out
    values = ids[active].astype(np.uint64)
    r = ((values * np.uint64(1103515245) + np.uint64(12345)) >> np.uint64(16)) & np.uint64(255)
    g = ((values * np.uint64(2654435761) + np.uint64(97)) >> np.uint64(15)) & np.uint64(255)
    b = ((values * np.uint64(2246822519) + np.uint64(53)) >> np.uint64(14)) & np.uint64(255)
    out[active] = np.stack(
        [
            r.astype(np.float32) / 255.0,
            g.astype(np.float32) / 255.0,
            b.astype(np.float32) / 255.0,
        ],
        axis=1,
    )
    return out


def _leave_one_out(
    *,
    rows: Sequence[Dict[str, object]],
    selected_ids: Sequence[int],
    contributions: Dict[int, List[Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    bounded_clip: Optional[float] = None,
) -> List[Dict[str, object]]:
    selected = list(selected_ids)
    all_composed = _compose(
        selected_ids=selected,
        contributions=contributions,
        stems=stems,
        state=state,
        args=args,
        split_filter=None,
        bounded_clip=bounded_clip,
    )
    all_metrics_all = _metrics_for_composed(all_composed, lowpass_kernel=int(args.lowpass_kernel))
    all_view_metrics_all = _per_view_metrics_for_composed(all_composed, lowpass_kernel=int(args.lowpass_kernel))
    all_composed_by_split = {
        split: _compose(
            selected_ids=selected,
            contributions=contributions,
            stems=stems,
            state=state,
            args=args,
            split_filter=split,
            bounded_clip=bounded_clip,
        )
        for split in SPLITS
    }
    all_by_split = {
        split: _metrics_for_composed(all_composed_by_split[split], lowpass_kernel=int(args.lowpass_kernel))
        for split in SPLITS
    }
    all_view_by_split = {
        split: _per_view_metrics_for_composed(all_composed_by_split[split], lowpass_kernel=int(args.lowpass_kernel))
        for split in SPLITS
    }
    row_by_id = {_row_cluster_id(row): row for row in rows}
    out: List[Dict[str, object]] = []
    for cid in selected:
        without_all_composed = _compose(
            selected_ids=selected,
            contributions=contributions,
            stems=stems,
            state=state,
            args=args,
            split_filter=None,
            skip_cluster_id=cid,
            bounded_clip=bounded_clip,
        )
        without_all_metrics = _metrics_for_composed(without_all_composed, lowpass_kernel=int(args.lowpass_kernel))
        without_all_view_metrics = _per_view_metrics_for_composed(without_all_composed, lowpass_kernel=int(args.lowpass_kernel))
        view_marginals_all = _view_gain_marginals(
            all_view_metrics_all,
            without_all_view_metrics,
            _cell_stems(contributions, cid, split_filter=None),
        )
        item: Dict[str, object] = {
            "cluster_id": int(cid),
            "rank": int(selected.index(cid) + 1),
            "C_selection_gain": _json_float(row_by_id.get(cid, {}).get("C_selection_gain"), float("nan")),
            "C_test_gain": _json_float(row_by_id.get(cid, {}).get("C_test_gain"), float("nan")),
            "marginal_gain_all": float(all_metrics_all.get("gain", 0.0) - without_all_metrics.get("gain", 0.0)),
            "LP_drift_marginal": float(all_metrics_all.get("LP_drift", 0.0) - without_all_metrics.get("LP_drift", 0.0)),
            "leakage_marginal": float(all_metrics_all.get("off_target_leakage", 0.0) - without_all_metrics.get("off_target_leakage", 0.0)),
            "view_marginals_all": view_marginals_all,
        }
        for split in SPLITS:
            without_composed = _compose(
                selected_ids=selected,
                contributions=contributions,
                stems=stems,
                state=state,
                args=args,
                split_filter=split,
                skip_cluster_id=cid,
                bounded_clip=bounded_clip,
            )
            without = _metrics_for_composed(without_composed, lowpass_kernel=int(args.lowpass_kernel))
            without_view_metrics = _per_view_metrics_for_composed(without_composed, lowpass_kernel=int(args.lowpass_kernel))
            all_metrics = all_by_split[split]
            display = SPLIT_DISPLAY[split]
            gain = float(all_metrics.get("gain", 0.0) - without.get("gain", 0.0))
            lp_drift = float(all_metrics.get("LP_drift", 0.0) - without.get("LP_drift", 0.0))
            leakage = float(all_metrics.get("off_target_leakage", 0.0) - without.get("off_target_leakage", 0.0))
            view_marginals = _view_gain_marginals(
                all_view_by_split[split],
                without_view_metrics,
                _cell_stems(contributions, cid, split_filter=split),
            )
            item[f"{display}_marginal_gain"] = gain
            item[f"{display}_marginal_LP_drift"] = lp_drift
            item[f"{display}_marginal_leakage"] = leakage
            item[f"marginal_gain_{display}"] = gain
            item[f"LP_drift_marginal_{display}"] = lp_drift
            item[f"leakage_marginal_{display}"] = leakage
            item[f"view_marginals_{display}"] = view_marginals
        analysis_values = list((item.get("view_marginals_analysis_test") or {}).values())
        fallback_values = list(view_marginals_all.values())
        values_for_clean = analysis_values if analysis_values else fallback_values
        selection_values = list((item.get("view_marginals_selection") or {}).values())
        negative_view_count = int(sum(1 for x in values_for_clean if float(x) < 0.0))
        item["negative_view_count"] = negative_view_count
        item["negative_view_ratio"] = float(negative_view_count / max(len(values_for_clean), 1))
        item["worst_view_marginal"] = float(min(values_for_clean)) if values_for_clean else 0.0
        item["median_view_marginal"] = _finite_median(values_for_clean, default=0.0)
        item.update(_overlap_stats_for_cell(cid, contributions, selected))
        item["selection_LOO_median_marginal"] = _finite_median(selection_values, default=0.0)
        item["selection_LOO_worst_marginal"] = float(min(selection_values)) if selection_values else 0.0
        item["selection_negative_view_ratio"] = float(
            sum(1 for x in selection_values if float(x) < 0.0) / max(len(selection_values), 1)
        )
        item["selection_overlap_conflict"] = item.get("overlap_conflict_score", 0.0)
        item["selection_phase_consistency"] = _phase_consistency_for_cell(cid, contributions, split_filter="selection")
        out.append(item)
    out.sort(key=lambda x: _json_float(x.get("analysis_test_marginal_gain"), 0.0))
    return out


def _phase_consistency_for_cell(cid: int, contributions: Dict[int, List[Contribution]], split_filter: Optional[str]) -> float:
    num = 0.0
    den = 0.0
    for contrib in contributions.get(int(cid), []):
        if split_filter is not None and contrib.split != split_filter:
            continue
        pred_luma = np.sum(contrib.pred_hp * oracle.LUMA[None, None, :], axis=2)
        target_luma = np.sum(contrib.target_hf * oracle.LUMA[None, None, :], axis=2)
        active = contrib.fit_w > 1e-8
        if not np.any(active):
            continue
        strength = np.minimum(np.abs(pred_luma), np.abs(target_luma)) * contrib.fit_w
        same = (pred_luma * target_luma) >= 0.0
        num += float(np.sum(strength * same.astype(np.float32) * active.astype(np.float32)))
        den += float(np.sum(strength * active.astype(np.float32)))
    return float(num / max(den, 1e-10))


def _cell_stems(contributions: Dict[int, List[Contribution]], cid: int, split_filter: Optional[str]) -> List[str]:
    stems: List[str] = []
    seen = set()
    for contrib in contributions.get(int(cid), []):
        if split_filter is not None and contrib.split != split_filter:
            continue
        if contrib.stem in seen:
            continue
        seen.add(contrib.stem)
        stems.append(contrib.stem)
    return stems


def _view_gain_marginals(
    all_view_metrics: Dict[str, Dict[str, float]],
    without_view_metrics: Dict[str, Dict[str, float]],
    stems: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for stem in stems:
        all_gain = _json_float(all_view_metrics.get(stem, {}).get("gain"), 0.0)
        without_gain = _json_float(without_view_metrics.get(stem, {}).get("gain"), 0.0)
        out[stem] = float(all_gain - without_gain)
    return out


def _overlap_stats_for_cell(cid: int, contributions: Dict[int, List[Contribution]], selected_ids: Sequence[int]) -> Dict[str, float]:
    total = 0.0
    overlapped = 0.0
    sign_conflict = 0.0
    by_stem: Dict[str, List[Contribution]] = {}
    for sid in selected_ids:
        for contrib in contributions.get(int(sid), []):
            by_stem.setdefault(contrib.stem, []).append(contrib)
    for contrib in contributions.get(int(cid), []):
        active = contrib.fit_w > 1e-8
        if not np.any(active):
            continue
        total += float(np.count_nonzero(active))
        x0, y0, x1, y1 = contrib.roi
        count = np.zeros_like(contrib.fit_w, dtype=np.float32)
        other_luma_sum = np.zeros_like(contrib.fit_w, dtype=np.float32)
        self_luma = np.sum(contrib.pred_hp * oracle.LUMA[None, None, :], axis=2)
        for other in by_stem.get(contrib.stem, []):
            ox0, oy0, ox1, oy1 = other.roi
            ix0, iy0 = max(x0, ox0), max(y0, oy0)
            ix1, iy1 = min(x1, ox1), min(y1, oy1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            sx0, sy0 = ix0 - x0, iy0 - y0
            sx1, sy1 = ix1 - x0, iy1 - y0
            oxs0, oys0 = ix0 - ox0, iy0 - oy0
            oxs1, oys1 = ix1 - ox0, iy1 - oy0
            count[sy0:sy1, sx0:sx1] += (other.fit_w[oys0:oys1, oxs0:oxs1] > 1e-8).astype(np.float32)
            if int(other.cluster_id) == int(cid):
                continue
            other_luma = np.sum(other.pred_hp * oracle.LUMA[None, None, :], axis=2)
            other_active = other.fit_w[oys0:oys1, oxs0:oxs1] > 1e-8
            other_luma_sum[sy0:sy1, sx0:sx1] += other_luma[oys0:oys1, oxs0:oxs1] * other_active.astype(np.float32)
        overlapped += float(np.count_nonzero(active & (count > 1.0)))
        sign_conflict += float(np.count_nonzero(active & (self_luma * other_luma_sum < 0.0)))
    duplicate_ratio = float(overlapped / max(total, 1.0))
    sign_conflict_ratio = float(sign_conflict / max(total, 1.0))
    return {
        "overlap_active_pixels": total,
        "overlap_duplicate_pixels": overlapped,
        "overlap_duplicate_ratio": duplicate_ratio,
        "overlap_sign_conflict_pixels": sign_conflict,
        "overlap_sign_conflict_ratio": sign_conflict_ratio,
        "overlap_conflict_score": float(0.5 * duplicate_ratio + 0.5 * sign_conflict_ratio),
    }


def _clean_deploy_rows(
    deploy_rows: Sequence[Dict[str, object]],
    per_cell_marginals: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    marginal_by_id = {int(item["cluster_id"]): item for item in per_cell_marginals}
    clean: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for row in deploy_rows:
        cid = _row_cluster_id(row)
        item = marginal_by_id.get(cid, {})
        reasons: List[str] = []
        median_view = _json_float(item.get("median_view_marginal"), 0.0)
        negative_ratio = _json_float(item.get("negative_view_ratio"), 0.0)
        aggregate_gain = _json_float(item.get("marginal_gain_all"), 0.0)
        lp_drift = _json_float(item.get("LP_drift_marginal"), 0.0)
        leakage = _json_float(item.get("leakage_marginal"), 0.0)
        if median_view < 0.0:
            reasons.append("median_view_marginal<0")
        if negative_ratio > float(args.clean_negative_view_ratio):
            reasons.append("negative_view_ratio>threshold")
        if aggregate_gain < 0.0 and (
            lp_drift > float(args.clean_lp_drift_marginal_min)
            or leakage > float(args.clean_leakage_marginal_min)
        ):
            reasons.append("aggregate_negative_with_drift_or_leakage")
        merged = dict(row)
        for key in (
            "marginal_gain_all",
            "marginal_gain_selection",
            "marginal_gain_analysis_test",
            "negative_view_count",
            "negative_view_ratio",
            "worst_view_marginal",
            "median_view_marginal",
            "LP_drift_marginal",
            "leakage_marginal",
            "overlap_conflict_score",
            "selection_LOO_median_marginal",
            "selection_LOO_worst_marginal",
            "selection_negative_view_ratio",
            "selection_overlap_conflict",
            "selection_phase_consistency",
        ):
            if key in item:
                merged[key] = item[key]
        if reasons:
            merged["clean_reject_reason"] = ";".join(reasons)
            rejected.append(merged)
        else:
            clean.append(merged)
    return clean, rejected


def _dose_curve(
    *,
    deploy_rows: Sequence[Dict[str, object]],
    contributions: Dict[int, List[Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    counts: Sequence[int],
    bounded_clip: Optional[float],
    raw_support_by_split: Optional[Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]]] = None,
    global_support_by_split: Optional[Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]]] = None,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for k in counts:
        subset_rows = list(deploy_rows)[: max(0, min(int(k), len(deploy_rows)))]
        ids = [_row_cluster_id(row) for row in subset_rows]
        report = _variant_report(
            name=f"top{k}",
            rows=subset_rows,
            selected_ids=ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=args,
            bounded_clip=bounded_clip,
            raw_support_by_split=raw_support_by_split,
            global_support_by_split=global_support_by_split,
        )
        out.append(report)
    return out


def _write_cell_sheets(
    *,
    focus_ids: Sequence[int],
    selected_ids: Sequence[int],
    row_by_id: Dict[int, Dict[str, object]],
    obs_by_cluster: Dict[int, List[oracle.EvalObs]],
    output_dir: Path,
    args: SimpleNamespace,
    preview_args: argparse.Namespace,
) -> None:
    root = output_dir / "cell_sheet"
    root.mkdir(parents=True, exist_ok=True)
    for cid in focus_ids:
        row = row_by_id.get(int(cid))
        if row is None:
            continue
        cand = _candidate_from_row(row)
        beta = _beta_from_row(row)
        panels: List[Tuple[str, np.ndarray]] = []
        for obs in obs_by_cluster.get(int(cid), [])[: int(preview_args.cell_sheet_max_obs)]:
            basis, target, fit_w, _off_w, _raw_basis, _lp = oracle._basis_arrays(obs, cand, args, "C", None)
            pred = basis[..., None] * beta[None, None, :]
            panels.extend(
                [
                    (f"{obs.split}:{obs.stem} target", np.clip(target * float(preview_args.visual_signed_scale) + 0.5, 0.0, 1.0)),
                    ("pred", np.clip(pred * float(preview_args.visual_signed_scale) + 0.5, 0.0, 1.0)),
                    ("weight", np.repeat(fit_w[..., None], 3, axis=2)),
                ]
            )
        if panels:
            _save_sheet(root / f"cluster_{int(cid):05d}.png", f"cluster={cid} selected={int(cid) in set(selected_ids)}", panels)


def _normalize_for_gray(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    value = np.asarray(arr, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    scale = float(np.percentile(np.abs(finite), float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip(np.abs(value) / scale, 0.0, 1.0).astype(np.float32)


def _sign_conflict_map_for_stem(
    *,
    stem: str,
    selected_ids: Sequence[int],
    contributions: Dict[int, List[Contribution]],
    shape: Tuple[int, int],
    split_filter: Optional[str],
) -> np.ndarray:
    h, w = shape
    pos = np.zeros((h, w), dtype=np.float32)
    neg = np.zeros((h, w), dtype=np.float32)
    for cid in selected_ids:
        for contrib in contributions.get(int(cid), []):
            if contrib.stem != stem:
                continue
            if split_filter is not None and contrib.split != split_filter:
                continue
            x0, y0, x1, y1 = contrib.roi
            luma = np.sum(contrib.pred_hp * oracle.LUMA[None, None, :], axis=2)
            active = contrib.fit_w > 1e-8
            pos[y0:y1, x0:x1] += np.maximum(luma, 0.0) * active.astype(np.float32)
            neg[y0:y1, x0:x1] += np.maximum(-luma, 0.0) * active.astype(np.float32)
    return np.minimum(pos, neg) / np.maximum(pos + neg, 1e-8)


def _negative_view_diagnostics(
    *,
    deploy_ids: Sequence[int],
    deploy_rows: Sequence[Dict[str, object]],
    contributions: Dict[int, List[Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    oracle_args: SimpleNamespace,
    preview_args: argparse.Namespace,
    output_dir: Path,
    per_cell_marginals: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    test_composed = _compose(
        selected_ids=deploy_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        split_filter="test",
        bounded_clip=None,
    )
    view_metrics = _per_view_metrics_for_composed(test_composed, lowpass_kernel=int(oracle_args.lowpass_kernel))
    active_views = [
        (stem, metrics)
        for stem, metrics in view_metrics.items()
        if bool(metrics.get("active")) and math.isfinite(_json_float(metrics.get("gain"), float("nan")))
    ]
    negative = [(stem, metrics) for stem, metrics in active_views if _json_float(metrics.get("gain"), 0.0) < 0.0]
    negative.sort(key=lambda item: _json_float(item[1].get("gain"), 0.0))
    marginal_by_id = {int(item["cluster_id"]): item for item in per_cell_marginals}
    row_by_id = {_row_cluster_id(row): row for row in deploy_rows}
    root = output_dir / "negative_view_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    diagnostics: Dict[str, object] = {
        "analysis_test_view_metrics": view_metrics,
        "negative_view_count": int(len(negative)),
        "negative_stems": [stem for stem, _metrics in negative],
        "views": {},
    }
    for stem, metrics in negative:
        data = test_composed[stem]
        view_dir = root / stem
        view_dir.mkdir(parents=True, exist_ok=True)
        oracle._save_rgb(
            view_dir / "base_plus_residual.png",
            data["preview"],
        )
        oracle._save_rgb(
            view_dir / "target_hf_signed.png",
            np.clip(data["target_hf"] * float(preview_args.visual_signed_scale) + 0.5, 0.0, 1.0),
        )
        oracle._save_rgb(
            view_dir / "residual_pred_signed.png",
            np.clip(data["pred"] * float(preview_args.visual_signed_scale) + 0.5, 0.0, 1.0),
        )
        oracle._save_rgb(view_dir / "dominant_residual_cell_map.png", _dominant_id_rgb(data["dominant_id"]))
        oracle._save_gray(view_dir / "overlap_count.png", np.clip(data["overlap"] / 8.0, 0.0, 1.0))
        oracle._save_gray(view_dir / "q_parent_peak.png", np.clip(data["q_parent_peak"], 0.0, 1.0))
        conflict = _sign_conflict_map_for_stem(
            stem=stem,
            selected_ids=deploy_ids,
            contributions=contributions,
            shape=data["weight"].shape,
            split_filter="test",
        )
        oracle._save_gray(view_dir / "overlap_sign_conflict.png", np.clip(conflict, 0.0, 1.0))
        target_luma = np.sum(data["target_hf"] * oracle.LUMA[None, None, :], axis=2)
        pred_luma = np.sum(data["pred"] * oracle.LUMA[None, None, :], axis=2)
        phase_mismatch = (target_luma * pred_luma < 0.0).astype(np.float32) * np.minimum(np.abs(target_luma), np.abs(pred_luma))
        oracle._save_gray(view_dir / "target_prediction_phase_mismatch.png", _normalize_for_gray(phase_mismatch))

        all_metric = _metrics_for_view(data, lowpass_kernel=int(oracle_args.lowpass_kernel))
        ranking: List[Dict[str, object]] = []
        for cid in deploy_ids:
            if stem not in _cell_stems(contributions, int(cid), split_filter="test"):
                continue
            without = _compose(
                selected_ids=deploy_ids,
                contributions=contributions,
                stems=[stem],
                state=state,
                args=oracle_args,
                split_filter="test",
                skip_cluster_id=int(cid),
                bounded_clip=None,
            )
            without_metric = _metrics_for_view(without[stem], lowpass_kernel=int(oracle_args.lowpass_kernel))
            marginal = _json_float(all_metric.get("gain"), 0.0) - _json_float(without_metric.get("gain"), 0.0)
            item = marginal_by_id.get(int(cid), {})
            row = row_by_id.get(int(cid), {})
            ranking.append(
                {
                    "cluster_id": int(cid),
                    "view_marginal_gain": float(marginal),
                    "C_selection_gain": row.get("C_selection_gain"),
                    "C_test_gain": row.get("C_test_gain"),
                    "median_view_marginal": item.get("median_view_marginal"),
                    "negative_view_count": item.get("negative_view_count"),
                    "overlap_conflict_score": item.get("overlap_conflict_score"),
                }
            )
        ranking.sort(key=lambda x: _json_float(x.get("view_marginal_gain"), 0.0))
        (view_dir / "cell_marginal_ranking.json").write_text(json.dumps(ranking, indent=2) + "\n", encoding="utf-8")
        diagnostics["views"][stem] = {
            "metrics": metrics,
            "cell_marginal_ranking": ranking,
            "paths": {
                "base_plus_residual": str(view_dir / "base_plus_residual.png"),
                "dominant_residual_cell_map": str(view_dir / "dominant_residual_cell_map.png"),
                "overlap_sign_conflict": str(view_dir / "overlap_sign_conflict.png"),
                "q_parent_peak": str(view_dir / "q_parent_peak.png"),
                "target_prediction_phase_mismatch": str(view_dir / "target_prediction_phase_mismatch.png"),
            },
        }
    return diagnostics


def _save_sheet(path: Path, title: str, panels: Sequence[Tuple[str, np.ndarray]]) -> None:
    thumb_w, thumb_h = 180, 120
    label_h = 18
    cols = 3
    rows = int(math.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((4, 2), title, fill=(255, 255, 255), font=font)
    for i, (label, arr) in enumerate(panels):
        r, c = divmod(i, cols)
        x = c * thumb_w
        y = label_h + r * (thumb_h + label_h)
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        img.thumbnail((thumb_w, thumb_h), Image.BICUBIC)
        canvas.paste(img, (x, y + label_h))
        draw.text((x + 3, y + 2), label[:28], fill=(255, 255, 255), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = _parse_args()
    oracle_dir = Path(args.oracle_dir).expanduser().resolve()
    summary_path = oracle_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    oracle_summary = _read_json(summary_path)
    oracle_args = _load_oracle_args(oracle_summary)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else oracle_dir / "deploy_residual_preview_v0"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_rows = _read_json(oracle_dir / "deploy_selected_rows.json")
    core_rows = _read_json(oracle_dir / "core_selected_rows.json")
    rows_all = _read_json(oracle_dir / "rows.json")
    rows_by_id = {_row_cluster_id(row): row for row in rows_all if row.get("valid")}
    valid_ids = sorted(rows_by_id)
    deploy_ids = [_row_cluster_id(row) for row in deploy_rows]
    core_ids = [_row_cluster_id(row) for row in core_rows]
    dose_counts = [int(x) for x in oracle._parse_csv(args.dose_counts, int) if int(x) > 0]
    focus_ids = [int(x) for x in oracle._parse_csv(args.focus_cluster_ids, int) if int(x) >= 0]
    minimal_clean_drop_ids = {int(x) for x in oracle._parse_csv(args.minimal_clean_drop_cluster_ids, int) if int(x) >= 0}
    minimal_clean_rows = [row for row in deploy_rows if _row_cluster_id(row) not in minimal_clean_drop_ids]
    minimal_clean_ids = [_row_cluster_id(row) for row in minimal_clean_rows]
    all_needed_ids = sorted(set(valid_ids) | set(deploy_ids) | set(core_ids) | set(focus_ids) | minimal_clean_drop_ids)

    print(f"[residual-preview-v0] oracle : {oracle_dir}")
    print(f"[residual-preview-v0] output : {output_dir}")
    print(f"[residual-preview-v0] deploy : {len(deploy_ids)} cells")
    print(f"[residual-preview-v0] core   : {len(core_ids)} cells")

    state = _load_scene_state(oracle_args)
    stems = [view.stem for view in state["views"]]
    obs_by_cluster = _build_obs_for_clusters(cluster_ids=all_needed_ids, state=state, args=oracle_args)
    row_union = [rows_by_id[cid] for cid in all_needed_ids if cid in rows_by_id]
    contributions = _build_contributions(row_union, obs_by_cluster, oracle_args)

    _save_shared_visuals(stems=stems, state=state, oracle_args=oracle_args, output_dir=output_dir, args=args)

    raw_support_composed_by_split: Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]] = {}
    global_support_composed_by_split: Dict[Optional[str], Dict[str, Dict[str, np.ndarray]]] = {}
    for split_key in [None, *SPLITS]:
        raw_support_composed_by_split[split_key] = _support_map_from_composed(
            _compose(
                selected_ids=deploy_ids,
                contributions=contributions,
                stems=stems,
                state=state,
                args=oracle_args,
                split_filter=split_key,
                bounded_clip=None,
            )
        )
        global_support_composed_by_split[split_key] = _support_map_from_composed(
            _compose(
                selected_ids=valid_ids,
                contributions=contributions,
                stems=stems,
                state=state,
                args=oracle_args,
                split_filter=split_key,
                bounded_clip=None,
            )
        )

    core_composed = _compose(
        selected_ids=core_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        bounded_clip=None,
    )
    deploy_raw_composed = _compose(
        selected_ids=deploy_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        bounded_clip=None,
    )
    deploy_bounded_composed = _compose(
        selected_ids=deploy_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        bounded_clip=float(args.bounded_delta_clip),
    )
    deploy_minimal_clean_composed = _compose(
        selected_ids=minimal_clean_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        bounded_clip=None,
    )

    loo = {
        "raw": _leave_one_out(
            rows=deploy_rows,
            selected_ids=deploy_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            bounded_clip=None,
        ),
        "bounded": _leave_one_out(
            rows=deploy_rows,
            selected_ids=deploy_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            bounded_clip=float(args.bounded_delta_clip),
        ),
    }
    clean_rows, clean_rejected_rows = _clean_deploy_rows(deploy_rows, loo["raw"], args)
    clean_ids = [_row_cluster_id(row) for row in clean_rows]
    deploy_clean_composed = _compose(
        selected_ids=clean_ids,
        contributions=contributions,
        stems=stems,
        state=state,
        args=oracle_args,
        bounded_clip=None,
    )

    _save_variant_visuals(
        variant_name="core28",
        composed=core_composed,
        output_dir=output_dir,
        args=args,
        lowpass_kernel=int(oracle_args.lowpass_kernel),
    )
    _save_variant_visuals(
        variant_name="deploy_top40_raw",
        composed=deploy_raw_composed,
        output_dir=output_dir,
        args=args,
        lowpass_kernel=int(oracle_args.lowpass_kernel),
    )
    _save_variant_visuals(
        variant_name="deploy_top40_bounded",
        composed=deploy_bounded_composed,
        output_dir=output_dir,
        args=args,
        lowpass_kernel=int(oracle_args.lowpass_kernel),
    )
    _save_variant_visuals(
        variant_name="deploy_top40_minimal_clean_dev",
        composed=deploy_minimal_clean_composed,
        output_dir=output_dir,
        args=args,
        lowpass_kernel=int(oracle_args.lowpass_kernel),
    )
    _save_variant_visuals(
        variant_name="deploy_top40_clean29",
        composed=deploy_clean_composed,
        output_dir=output_dir,
        args=args,
        lowpass_kernel=int(oracle_args.lowpass_kernel),
    )

    reports = {
        "core28": _variant_report(
            name="core28",
            rows=core_rows,
            selected_ids=core_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "deploy_top40_raw": _variant_report(
            name="deploy_top40_raw",
            rows=deploy_rows,
            selected_ids=deploy_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "deploy_top40_bounded": _variant_report(
            name="deploy_top40_bounded",
            rows=deploy_rows,
            selected_ids=deploy_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            bounded_clip=float(args.bounded_delta_clip),
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "deploy_top40_minimal_clean_dev": _variant_report(
            name="deploy_top40_minimal_clean_dev",
            rows=minimal_clean_rows,
            selected_ids=minimal_clean_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "deploy_top40_clean29": _variant_report(
            name="deploy_top40_clean29",
            rows=clean_rows,
            selected_ids=clean_ids,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
    }
    dose_curve = {
        "raw": _dose_curve(
            deploy_rows=deploy_rows,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            counts=dose_counts,
            bounded_clip=None,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "bounded": _dose_curve(
            deploy_rows=deploy_rows,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            counts=dose_counts,
            bounded_clip=float(args.bounded_delta_clip),
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "minimal_clean_dev": _dose_curve(
            deploy_rows=minimal_clean_rows,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            counts=dose_counts,
            bounded_clip=None,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
        "clean29": _dose_curve(
            deploy_rows=clean_rows,
            contributions=contributions,
            stems=stems,
            state=state,
            args=oracle_args,
            counts=dose_counts,
            bounded_clip=None,
            raw_support_by_split=raw_support_composed_by_split,
            global_support_by_split=global_support_composed_by_split,
        ),
    }
    _write_cell_sheets(
        focus_ids=focus_ids,
        selected_ids=deploy_ids,
        row_by_id=rows_by_id,
        obs_by_cluster=obs_by_cluster,
        output_dir=output_dir,
        args=oracle_args,
        preview_args=args,
    )
    negative_view_diagnostics = _negative_view_diagnostics(
        deploy_ids=deploy_ids,
        deploy_rows=deploy_rows,
        contributions=contributions,
        stems=stems,
        state=state,
        oracle_args=oracle_args,
        preview_args=args,
        output_dir=output_dir,
        per_cell_marginals=loo["raw"],
    )

    summary = {
        "version": "render_residual_tetris_static_v1_preview",
        "oracle_dir": str(oracle_dir),
        "output_dir": str(output_dir),
        "frozen_inputs": {
            "deploy_set": "deploy_top40",
            "deploy_count": int(len(deploy_ids)),
            "bounded_set": "deploy_top40_bounded",
            "minimal_clean_set": "deploy_top40_minimal_clean_dev",
            "minimal_clean_drop_cluster_ids": sorted(int(x) for x in minimal_clean_drop_ids),
            "minimal_clean_count": int(len(minimal_clean_ids)),
            "clean_set": "deploy_top40_clean29",
            "clean_count": int(len(clean_ids)),
            "clean_rejected_count": int(len(clean_rejected_rows)),
            "control_set": "core_pass28",
            "core_count": int(len(core_ids)),
            "ranking": oracle_summary.get("deploy_selector", {}).get("policy", {}).get("rank_key", "C_selection_gain"),
            "views": ["fit", "selection", "analysis_test"],
            "note": "beta, slot, piece geometry, donor IDs, and order are loaded from oracle JSON and are not refit.",
        },
        "clean_policy": {
            "negative_view_ratio_threshold": float(args.clean_negative_view_ratio),
            "lp_drift_marginal_min": float(args.clean_lp_drift_marginal_min),
            "leakage_marginal_min": float(args.clean_leakage_marginal_min),
            "rule": "Reject only cells with negative median view marginal, majority negative views, or aggregate negative marginal with drift/leakage. Raw deploy_top40 is preserved.",
        },
        "bounded_delta_clip": float(args.bounded_delta_clip),
        "metric_definitions": {
            "joint_gain": "Target HF energy minus residual energy after summing all frozen cells once.",
            "gain_on_variant_support": "Gain measured only where the evaluated variant has active support; useful for coverage efficiency, not fair cross-version comparison.",
            "gain_on_raw_union_support": "Gain measured on the fixed deploy_top40_raw support; preferred for fair raw/bounded/minimal-clean/clean29 comparison.",
            "gain_on_global_eligible_support": "Gain measured on the union support of all valid oracle rows loaded from rows.json; broader fair comparison against eligible candidate regions.",
            "sum_individual_gain": "Signed sum of frozen per-cell C_*_gain from oracle rows for the same split.",
            "sum_positive_individual_gain": "Positive-only sum of frozen per-cell C_*_gain; used as the stable gain-capture denominator.",
            "joint_gain_capture_positive": "joint_gain / sum_positive_individual_gain.",
            "marginal_gain_all": "Joint gain(all deploy cells) - joint gain(all except this cell), using all frozen observations.",
            "marginal_gain_selection": "Same leave-one-cell-out marginal on selection observations.",
            "marginal_gain_analysis_test": "Same leave-one-cell-out marginal on analysis_test observations; diagnostic only, not a final lockbox metric.",
            "selection_LOO_median_marginal": "Median per-view leave-one-out marginal on selection views only; intended for future lockbox-safe clean policy.",
            "selection_LOO_worst_marginal": "Worst per-view leave-one-out marginal on selection views only.",
            "selection_negative_view_ratio": "Fraction of selection views where the leave-one-out marginal is negative.",
            "selection_overlap_conflict": "Overlap conflict score recorded for selection-only clean policy design.",
            "selection_phase_consistency": "Weighted sign agreement between this cell residual and target HF on selection views.",
            "negative_view_count": "Number of analysis_test views, or all available views if no analysis_test view exists, where the leave-one-cell-out marginal is negative.",
            "worst_view_marginal": "Worst per-view leave-one-cell-out marginal in the same view set used by negative_view_count.",
            "median_view_marginal": "Median per-view leave-one-cell-out marginal in the same view set used by negative_view_count.",
            "overlap_conflict_score": "Average of active-pixel overlap duplicate ratio and sign-conflict ratio with other deploy cells.",
            "off_target_leakage": "Energy of the final composed residual outside the tolerated support divided by in-support residual energy.",
            "LP_drift": "Low-pass energy of the final composed residual divided by target HF energy.",
        },
        "joint_metrics": reports,
        "go_no_go": _go_no_go(reports),
        "negative_view_diagnostics": negative_view_diagnostics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "joint_metrics.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dose_curve.json").write_text(json.dumps(dose_curve, indent=2) + "\n", encoding="utf-8")
    (output_dir / "leave_one_cell_out.json").write_text(json.dumps(loo, indent=2) + "\n", encoding="utf-8")
    (output_dir / "per_cell_marginals.json").write_text(json.dumps(loo["raw"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "minimal_clean_selected_rows.json").write_text(json.dumps(minimal_clean_rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "clean_selected_rows.json").write_text(json.dumps(clean_rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "clean_rejected_rows.json").write_text(json.dumps(clean_rejected_rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "negative_view_diagnostics.json").write_text(json.dumps(negative_view_diagnostics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.txt").write_text(_readme(output_dir), encoding="utf-8")

    if args.check_dir:
        _copy_to_check(output_dir, Path(args.check_dir).expanduser().resolve())
    print(json.dumps(summary, indent=2))


def _go_no_go(reports: Dict[str, object]) -> Dict[str, object]:
    raw = reports["deploy_top40_raw"]["by_split"]["analysis_test"]
    bounded = reports["deploy_top40_bounded"]["by_split"]["analysis_test"]
    core = reports["core28"]["by_split"]["analysis_test"]
    raw_gain = _json_float(raw.get("gain"), 0.0)
    bounded_gain = _json_float(bounded.get("gain"), 0.0)
    core_gain = _json_float(core.get("gain"), 0.0)
    raw_positive = _json_float(raw.get("positive_view_ratio"), 0.0)
    raw_sum_positive = _json_float(raw.get("sum_positive_individual_gain"), 0.0)
    collapse = raw_gain / max(raw_sum_positive, 1e-10)
    return {
        "analysis_test_gain_positive": bool(raw_gain > 0.0),
        "majority_analysis_test_views_positive": bool(raw_positive > 0.5),
        "deploy_beats_core": bool(raw_gain > core_gain),
        "joint_not_collapsed": bool(collapse > 0.35),
        "bounded_keeps_raw_gain": bool(bounded_gain > 0.65 * raw_gain) if raw_gain > 0.0 else False,
        "LP_drift_raw": raw.get("LP_drift"),
        "leakage_raw": raw.get("off_target_leakage"),
        "raw_gain": raw_gain,
        "core_gain": core_gain,
        "bounded_gain": bounded_gain,
        "joint_gain_capture_positive": collapse,
    }


def _readme(output_dir: Path) -> str:
    return "\n".join(
        [
            "Residual Tetris static sparse residual branch preview.",
            "",
            "This is an offline closed-loop composition preview for the V1 sparse residual branch.",
            "It does not train the base 3DGS, refit beta, search slots, add cells, or change selector order.",
            "The composed residual is delta_k = q_parent_k * projected_signed_piece_k * beta_k, accumulated over frozen cells.",
            "deploy_top40_raw is the fixed baseline; deploy_top40_minimal_clean_dev and deploy_top40_clean29 are diagnostic derivatives and must not replace raw.",
            "",
            "Main files:",
            f"  {output_dir / 'summary.json'}",
            f"  {output_dir / 'joint_metrics.json'}",
            f"  {output_dir / 'dose_curve.json'}",
            f"  {output_dir / 'leave_one_cell_out.json'}",
            f"  {output_dir / 'per_cell_marginals.json'}",
            f"  {output_dir / 'minimal_clean_selected_rows.json'}",
            f"  {output_dir / 'clean_selected_rows.json'}",
            f"  {output_dir / 'clean_rejected_rows.json'}",
            f"  {output_dir / 'negative_view_diagnostics.json'}",
            "",
            "Visual directories:",
            f"  {output_dir / 'visuals/base'}",
            f"  {output_dir / 'visuals/target_hf'}",
            f"  {output_dir / 'visuals/core28/base_plus_residual'}",
            f"  {output_dir / 'visuals/deploy_top40_raw/base_plus_residual'}",
            f"  {output_dir / 'visuals/deploy_top40_bounded/base_plus_residual'}",
            f"  {output_dir / 'visuals/deploy_top40_minimal_clean_dev/base_plus_residual'}",
            f"  {output_dir / 'visuals/deploy_top40_clean29/base_plus_residual'}",
            f"  {output_dir / 'negative_view_diagnostics'}",
            f"  {output_dir / 'cell_sheet'}",
            "",
        ]
    )


if __name__ == "__main__":
    main()
