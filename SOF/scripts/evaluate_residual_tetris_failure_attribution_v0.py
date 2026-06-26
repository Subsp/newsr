#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_residual_tetris_oracle_v0 as oracle  # noqa: E402
import render_residual_tetris_level1_lockbox_v0 as level1  # noqa: E402
import render_residual_tetris_preview_v0 as preview  # noqa: E402
import render_residual_tetris_static_v1 as static_v1  # noqa: E402


VERSION = "evaluate_residual_tetris_failure_attribution_v2"


def _parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Failure-attribution matrix for frozen Residual Tetris cells. "
            "It never refits beta, changes top-k, or changes frozen pieces; it only "
            "replays the same cells under diagnostic q/lambda/sign/target choices."
        )
    )
    parser.add_argument("--static_v1_dir", required=True)
    parser.add_argument("--oracle_dir", default="")
    parser.add_argument("--primitive_dir", required=True)
    parser.add_argument("--base_render_dir", required=True)
    parser.add_argument("--target_dir", required=True, help="Primary target image directory, e.g. GT or SR/VOSR.")
    parser.add_argument("--weight_dir", required=True)
    parser.add_argument("--q_parent_dir", default="")
    parser.add_argument("--alt_target_dir", default="", help="Optional second target for target-mismatch diagnostics.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--check_dir", default="")
    parser.add_argument("--target_type", default="")
    parser.add_argument("--alt_target_type", default="")
    parser.add_argument("--match_policy", choices=("stem", "order_if_needed", "order"), default="order_if_needed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--camera_index_offset", type=int, default=0)
    parser.add_argument("--cell_set", choices=("deploy_top40_raw", "core28", "deploy_top40_minimal_clean_dev"), default="deploy_top40_raw")
    parser.add_argument("--q_modes", default="proxy,proxy_scaled,true,unit_visibility")
    parser.add_argument("--lambdas", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--signs", default="plus")
    parser.add_argument("--dev_q_reference", type=float, default=0.10)
    parser.add_argument("--q_scale_stat", choices=("median", "mean", "p95"), default="median")
    parser.add_argument("--bounded_delta_clip", type=float, default=-1.0)
    parser.add_argument("--bounded_mode", choices=("per_cell_clip", "post_sum_clip", "none", "from_config"), default="none")
    parser.add_argument("--changed_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--write_visuals", type=int, default=1)
    parser.add_argument("--visual_variant_limit", type=int, default=6)
    parser.add_argument("--visual_signed_scale", type=float, default=4.0)
    parser.add_argument("--error_scale", type=float, default=8.0)
    parser.add_argument("--target_active_threshold", type=float, default=0.01)
    parser.add_argument("--shift_grid", default="", help="Optional comma list, e.g. -2,-1,0,1,2.")
    parser.add_argument("--shift_q_modes", default="proxy_scaled,true")
    parser.add_argument("--shift_signs", default="plus")
    parser.add_argument("--shift_lambdas", default="1.0")
    parser.add_argument("--per_cell_shift_grid", default="", help="Optional per-cell shift grid. Empty by default to avoid heavy scans.")
    parser.add_argument("--write_per_cell_report", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _copy_to_check(output_dir: Path, check_dir: Path) -> None:
    if not str(check_dir):
        return
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "metrics.json",
        "per_view_metrics.json",
        "failure_attribution_metrics.json",
        "failure_attribution_per_view.json",
        "q_distribution_report.json",
        "target_similarity_report.json",
        "shift_attribution_report.json",
        "alignment_sanity_report.json",
        "qtrue_status_or_qtrue_metrics.json",
        "per_cell_failure_report.json",
        "summary.json",
        "manifest.json",
    ):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, check_dir / name)
    visual_root = output_dir / "visuals"
    if visual_root.is_dir():
        dst = check_dir / "visuals"
        shutil.copytree(visual_root, dst, dirs_exist_ok=True)


def _safe_stat(stats: Dict[str, float], key: str, default: float = 0.0) -> float:
    value = float(stats.get(key, default))
    return value if math.isfinite(value) else float(default)


def _target_type_from_text(text: str, fallback: str) -> str:
    if fallback:
        return fallback
    lower = str(text).lower()
    if "gt" in lower:
        return "GT_HF"
    if "vosr" in lower or "qwen" in lower:
        return "VOSR_HF"
    if "sr" in lower or "fused_prior" in lower or "fused_priors" in lower:
        return "SR_HF"
    return "unknown_hf"


def _args_for_target(cli: argparse.Namespace, frozen: Dict[str, object], static_dir: Path, target_dir: str, q_parent_dir: str) -> SimpleNamespace:
    ns = SimpleNamespace(
        oracle_dir=str(cli.oracle_dir),
        primitive_dir=str(cli.primitive_dir),
        base_render_dir=str(cli.base_render_dir),
        sr_dir=str(target_dir),
        weight_dir=str(cli.weight_dir),
        q_parent_dir=str(q_parent_dir),
        carrier_rgb_dir="",
        carrier_render_dir="",
        match_policy=str(cli.match_policy),
        limit=int(cli.limit),
        camera_index_offset=int(cli.camera_index_offset),
    )
    return level1._oracle_args_from_static(static_dir, frozen, ns)


def _load_state_and_obs(
    *,
    cli: argparse.Namespace,
    frozen: Dict[str, object],
    static_dir: Path,
    target_dir: str,
    q_parent_dir: str,
    cluster_ids: Sequence[int],
) -> Tuple[SimpleNamespace, Dict[str, object], Dict[int, List[oracle.EvalObs]]]:
    args_ns = _args_for_target(cli, frozen, static_dir, target_dir=target_dir, q_parent_dir=q_parent_dir)
    state = level1._load_lockbox_state(args_ns)
    obs = level1._build_lockbox_obs(
        cluster_ids=cluster_ids,
        frozen_by_id=frozen["frozen_by_id"],
        state=state,
        args=args_ns,
    )
    return args_ns, state, obs


def _clone_obs_with_q_scale(obs_by_cluster: Dict[int, List[oracle.EvalObs]], scale: float) -> Dict[int, List[oracle.EvalObs]]:
    out: Dict[int, List[oracle.EvalObs]] = {}
    for cid, obs_list in obs_by_cluster.items():
        out[cid] = [
            replace(obs, q_parent=np.clip(np.asarray(obs.q_parent, dtype=np.float32) * float(scale), 0.0, 1.0).astype(np.float32))
            for obs in obs_list
        ]
    return out


def _clone_obs_with_unit_q(obs_by_cluster: Dict[int, List[oracle.EvalObs]]) -> Dict[int, List[oracle.EvalObs]]:
    out: Dict[int, List[oracle.EvalObs]] = {}
    for cid, obs_list in obs_by_cluster.items():
        out[cid] = [
            replace(obs, q_parent=np.ones_like(np.asarray(obs.q_parent, dtype=np.float32), dtype=np.float32))
            for obs in obs_list
        ]
    return out


def _obs_q_stats(obs_list: Sequence[oracle.EvalObs], threshold: float) -> Dict[str, float]:
    values: List[np.ndarray] = []
    masses = []
    support_masses = []
    for obs in obs_list:
        q = np.asarray(obs.q_parent, dtype=np.float32)
        core = np.asarray(obs.core_weight, dtype=np.float32)
        mask = core > float(threshold)
        if np.any(mask):
            values.append(q[mask].reshape(-1))
        masses.append(float(np.sum(q)))
        support_masses.append(float(np.sum(q * core)))
    arr = np.concatenate(values, axis=0).astype(np.float32) if values else np.zeros((0,), dtype=np.float32)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "p95": float(np.percentile(arr, 95.0)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mass_mean": float(np.mean(masses)) if masses else 0.0,
        "mass_on_support_mean": float(np.mean(support_masses)) if support_masses else 0.0,
        "count": int(arr.size),
    }


def _shift_zero(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    src = np.asarray(arr)
    out = np.zeros_like(src)
    h, w = src.shape[:2]
    if abs(int(dx)) >= w or abs(int(dy)) >= h:
        return out
    src_x0 = max(0, -int(dx))
    src_x1 = min(w, w - int(dx))
    dst_x0 = max(0, int(dx))
    dst_x1 = min(w, w + int(dx))
    src_y0 = max(0, -int(dy))
    src_y1 = min(h, h - int(dy))
    dst_y0 = max(0, int(dy))
    dst_y1 = min(h, h + int(dy))
    out[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]
    return out


def _shift_composed(composed: Dict[str, Dict[str, np.ndarray]], dx: int, dy: int) -> Dict[str, Dict[str, np.ndarray]]:
    shifted: Dict[str, Dict[str, np.ndarray]] = {}
    for stem, data in composed.items():
        new_data = {k: v for k, v in data.items()}
        for key in ("pred", "pred_raw", "pred_lp", "piece_positive_lobe", "piece_negative_lobe"):
            if key in data:
                new_data[key] = _shift_zero(np.asarray(data[key]), dx=dx, dy=dy)
        raw_image = np.asarray(new_data["base"], dtype=np.float32) + np.asarray(new_data["pred"], dtype=np.float32)
        new_data["preview"] = np.clip(raw_image, 0.0, 1.0)
        new_data["out_of_range"] = (raw_image < 0.0) | (raw_image > 1.0)
        shifted[stem] = new_data
    return shifted


def _shift_target_composed(composed: Dict[str, Dict[str, np.ndarray]], dx: int, dy: int) -> Dict[str, Dict[str, np.ndarray]]:
    shifted: Dict[str, Dict[str, np.ndarray]] = {}
    for stem, data in composed.items():
        new_data = {k: v for k, v in data.items()}
        new_data["target_hf"] = _shift_zero(np.asarray(data["target_hf"], dtype=np.float32), dx=dx, dy=dy)
        shifted[stem] = new_data
    return shifted


def _scale_contributions(
    contributions: Dict[int, List[preview.Contribution]],
    factor: float,
) -> Dict[int, List[preview.Contribution]]:
    out: Dict[int, List[preview.Contribution]] = {}
    abs_factor = abs(float(factor))
    for cid, items in contributions.items():
        out[cid] = [
            replace(
                contrib,
                pred_hp=np.asarray(contrib.pred_hp, dtype=np.float32) * float(factor),
                pred_raw=np.asarray(contrib.pred_raw, dtype=np.float32) * float(factor),
                pred_lp=np.asarray(contrib.pred_lp, dtype=np.float32) * float(factor),
                abs_luma=np.asarray(contrib.abs_luma, dtype=np.float32) * abs_factor,
            )
            for contrib in items
        ]
    return out


def _build_shifted_basis_contributions(
    *,
    rows: Sequence[Dict[str, object]],
    obs_by_cluster: Dict[int, List[oracle.EvalObs]],
    args: SimpleNamespace,
    shift_mode: str,
    dx: int,
    dy: int,
    factor: float,
) -> Dict[int, List[preview.Contribution]]:
    out: Dict[int, List[preview.Contribution]] = {}
    for row in rows:
        cid = preview._row_cluster_id(row)
        cand = preview._candidate_from_row(row)
        beta = preview._beta_from_row(row)
        row_contribs: List[preview.Contribution] = []
        for obs in obs_by_cluster.get(cid, []):
            xx, yy = oracle._coords(obs.roi)
            phi = oracle._piece(
                candidate=cand,
                xx=xx,
                yy=yy,
                center=obs.center,
                theta=obs.theta,
                long_px=obs.long_px,
                short_px=obs.short_px,
                args=args,
            )
            q = np.clip(np.asarray(obs.q_parent, dtype=np.float32), 0.0, 1.0)
            if shift_mode == "phi":
                phi = _shift_zero(phi.astype(np.float32), dx=dx, dy=dy)
            elif shift_mode == "q":
                q = _shift_zero(q, dx=dx, dy=dy)
            else:
                raise ValueError(f"unsupported shifted basis mode: {shift_mode}")
            raw_basis = (q * phi).astype(np.float32)
            basis_hp = oracle._highpass(raw_basis, int(args.highpass_kernel)).astype(np.float32) * float(factor)
            raw_scaled = raw_basis * float(factor)
            lp_scaled = oracle._box_blur(raw_basis, int(args.lowpass_kernel)).astype(np.float32) * float(factor)
            pred_hp = basis_hp[..., None] * beta[None, None, :]
            pred_raw = raw_scaled[..., None] * beta[None, None, :]
            pred_lp = lp_scaled[..., None] * beta[None, None, :]
            fit_w = np.clip(np.asarray(obs.core_weight, dtype=np.float32), 0.0, 1.0)
            tolerance = oracle._dilate(fit_w > float(args.core_weight_threshold), int(args.tolerance_radius))
            off_w = np.where(tolerance, 0.0, np.clip(obs.support, 0.0, 1.0)).astype(np.float32)
            abs_luma = np.abs(np.sum(pred_hp * oracle.LUMA[None, None, :], axis=2)).astype(np.float32)
            row_contribs.append(
                preview.Contribution(
                    cluster_id=cid,
                    stem=obs.stem,
                    split=obs.split,
                    roi=obs.roi,
                    pred_hp=pred_hp.astype(np.float32),
                    pred_raw=pred_raw.astype(np.float32),
                    pred_lp=pred_lp.astype(np.float32),
                    fit_w=fit_w,
                    off_w=off_w,
                    abs_luma=abs_luma,
                    q_parent=q.astype(np.float32),
                    target_hf=obs.target_hf.astype(np.float32),
                )
            )
        out[cid] = row_contribs
    return out


def _retarget_composed(
    composed: Dict[str, Dict[str, np.ndarray]],
    *,
    target_dir: Path,
    state: Dict[str, object],
    args: SimpleNamespace,
) -> Dict[str, Dict[str, np.ndarray]]:
    target_paths = oracle._list_files(target_dir)
    target_lookup = oracle._image_lookup(target_paths)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for stem, data in composed.items():
        view_index = int(state.get("view_index_by_stem", {}).get(stem, 0))
        target_path = oracle._resolve_path(target_paths, target_lookup, stem, view_index, str(args.match_policy))
        base = np.asarray(data["base"], dtype=np.float32)
        sr = oracle._load_rgb(target_path, size=(base.shape[1], base.shape[0]))
        target = oracle._highpass(sr - base, int(args.highpass_kernel)).astype(np.float32)
        new_data = {k: v for k, v in data.items()}
        new_data["target_hf"] = target
        out[stem] = new_data
    return out


def _target_similarity(
    *,
    target_a_dir: Path,
    target_b_dir: Path,
    state: Dict[str, object],
    args: SimpleNamespace,
    active_threshold: float,
) -> Dict[str, object]:
    paths_a = oracle._list_files(target_a_dir)
    paths_b = oracle._list_files(target_b_dir)
    lookup_a = oracle._image_lookup(paths_a)
    lookup_b = oracle._image_lookup(paths_b)
    weights = oracle._image_lookup(oracle._list_files(Path(args.weight_dir)))
    rows = []
    all_a = []
    all_b = []
    all_w = []
    energy_a = 0.0
    energy_b = 0.0
    sign_same_num = 0.0
    sign_same_den = 0.0
    active_inter = 0.0
    active_union = 0.0
    for view in state["views"]:
        stem = view.stem
        view_index = int(view.view_index)
        base_path = oracle._resolve_path(state["base_paths"], state["base_lookup"], stem, view_index, str(args.match_policy))
        base = oracle._load_rgb(base_path)
        size = (base.shape[1], base.shape[0])
        pa = oracle._resolve_path(paths_a, lookup_a, stem, view_index, str(args.match_policy))
        pb = oracle._resolve_path(paths_b, lookup_b, stem, view_index, str(args.match_policy))
        pw = oracle._resolve_path(state["weight_paths"], weights, stem, view_index, str(args.match_policy))
        ta = oracle._highpass(oracle._load_rgb(pa, size=size) - base, int(args.highpass_kernel))
        tb = oracle._highpass(oracle._load_rgb(pb, size=size) - base, int(args.highpass_kernel))
        w = oracle._load_gray(pw, size=size)
        la = np.sum(ta * oracle.LUMA[None, None, :], axis=2)
        lb = np.sum(tb * oracle.LUMA[None, None, :], axis=2)
        corr = oracle._weighted_corr(la, lb, w)
        sign_active = w > 1e-8
        sign_agree = float(np.mean(np.sign(la[sign_active]) == np.sign(lb[sign_active]))) if np.any(sign_active) else 0.0
        ea = float(np.sum(w[..., None] * ta * ta))
        eb = float(np.sum(w[..., None] * tb * tb))
        active_a = (np.abs(la) > float(active_threshold)) & sign_active
        active_b = (np.abs(lb) > float(active_threshold)) & sign_active
        inter = float(np.count_nonzero(active_a & active_b))
        union = float(np.count_nonzero(active_a | active_b))
        ee = 1.0 - float(np.sum(w[..., None] * (ta - tb) ** 2)) / max(float(np.sum(w[..., None] * ta * ta)), 1e-10)
        rows.append(
            {
                "stem": stem,
                "corr": float("nan") if corr is None else float(corr),
                "explained_energy": float(ee),
                "sign_agreement": sign_agree,
                "energy_a": ea,
                "energy_b": eb,
                "energy_ratio_b_over_a": float(eb / max(ea, 1e-10)),
                "active_overlap": float(inter / max(union, 1.0)),
            }
        )
        energy_a += ea
        energy_b += eb
        if np.any(sign_active):
            sign_same_num += float(np.count_nonzero(np.sign(la[sign_active]) == np.sign(lb[sign_active])))
            sign_same_den += float(np.count_nonzero(sign_active))
        active_inter += inter
        active_union += union
        all_a.append(la.reshape(-1))
        all_b.append(lb.reshape(-1))
        all_w.append(w.reshape(-1))
    ca = np.concatenate(all_a, axis=0) if all_a else np.zeros((0,), dtype=np.float32)
    cb = np.concatenate(all_b, axis=0) if all_b else np.zeros((0,), dtype=np.float32)
    cw = np.concatenate(all_w, axis=0) if all_w else np.zeros((0,), dtype=np.float32)
    corr_all = oracle._weighted_corr(ca, cb, cw)
    return {
        "target_a_dir": str(target_a_dir),
        "target_b_dir": str(target_b_dir),
        "corr": float("nan") if corr_all is None else float(corr_all),
        "sign_agreement": float(sign_same_num / max(sign_same_den, 1.0)),
        "energy_a": float(energy_a),
        "energy_b": float(energy_b),
        "energy_ratio_b_over_a": float(energy_b / max(energy_a, 1e-10)),
        "active_overlap": float(active_inter / max(active_union, 1.0)),
        "active_threshold": float(active_threshold),
        "per_view": rows,
    }


def _alignment_sanity_report(
    *,
    target_dirs: Sequence[Tuple[str, Path]],
    state: Dict[str, object],
    args: SimpleNamespace,
    shift_values: Sequence[int],
) -> Dict[str, object]:
    if not shift_values:
        return {"skipped": True, "reason": "shift_grid is empty"}
    weight_paths = oracle._list_files(Path(args.weight_dir))
    weight_lookup = oracle._image_lookup(weight_paths)
    report: Dict[str, object] = {"version": VERSION, "grid": [int(x) for x in shift_values], "targets": {}}
    for target_name, target_dir in target_dirs:
        target_paths = oracle._list_files(target_dir)
        target_lookup = oracle._image_lookup(target_paths)
        per_view = []
        global_by_shift: Dict[Tuple[int, int], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]] = {}
        for view in state["views"]:
            stem = view.stem
            view_index = int(view.view_index)
            base_path = oracle._resolve_path(state["base_paths"], state["base_lookup"], stem, view_index, str(args.match_policy))
            target_path = oracle._resolve_path(target_paths, target_lookup, stem, view_index, str(args.match_policy))
            weight_path = oracle._resolve_path(weight_paths, weight_lookup, stem, view_index, str(args.match_policy))
            base = oracle._load_rgb(base_path)
            size = (base.shape[1], base.shape[0])
            target = oracle._load_rgb(target_path, size=size)
            weight = oracle._load_gray(weight_path, size=size)
            base_edge = np.abs(np.sum(oracle._highpass(base, int(args.highpass_kernel)) * oracle.LUMA[None, None, :], axis=2))
            target_edge = np.abs(np.sum(oracle._highpass(target, int(args.highpass_kernel)) * oracle.LUMA[None, None, :], axis=2))
            best_gain = {"corr": -1e30, "dx": 0, "dy": 0}
            for dy in shift_values:
                for dx in shift_values:
                    shifted_target = _shift_zero(target_edge, dx=int(dx), dy=int(dy))
                    corr = oracle._weighted_corr(base_edge, shifted_target, weight)
                    corr_value = float(corr) if corr is not None and math.isfinite(float(corr)) else -1e30
                    if corr_value > float(best_gain["corr"]):
                        best_gain = {"corr": corr_value, "dx": int(dx), "dy": int(dy)}
                    key = (int(dx), int(dy))
                    if key not in global_by_shift:
                        global_by_shift[key] = ([], [], [])
                    global_by_shift[key][0].append(base_edge.reshape(-1))
                    global_by_shift[key][1].append(shifted_target.reshape(-1))
                    global_by_shift[key][2].append(weight.reshape(-1))
            per_view.append({"stem": stem, **best_gain})
        best_global = {"corr": -1e30, "dx": 0, "dy": 0}
        for (dx, dy), arrays in global_by_shift.items():
            a = np.concatenate(arrays[0], axis=0) if arrays[0] else np.zeros((0,), dtype=np.float32)
            b = np.concatenate(arrays[1], axis=0) if arrays[1] else np.zeros((0,), dtype=np.float32)
            w = np.concatenate(arrays[2], axis=0) if arrays[2] else np.zeros((0,), dtype=np.float32)
            corr = oracle._weighted_corr(a, b, w)
            corr_value = float(corr) if corr is not None and math.isfinite(float(corr)) else -1e30
            if corr_value > float(best_global["corr"]):
                best_global = {"corr": corr_value, "dx": int(dx), "dy": int(dy)}
        report["targets"][target_name] = {
            "target_dir": str(target_dir),
            "best_global_base_edge_vs_target_edge_shift": best_global,
            "per_view": per_view,
        }
    return report


def _changed_metrics(composed: Dict[str, Dict[str, np.ndarray]], threshold: float) -> Dict[str, float]:
    rows = [level1._changed_metrics(data, threshold) for data in composed.values()]
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _metrics_for_composed(
    *,
    composed: Dict[str, Dict[str, np.ndarray]],
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    rows: Sequence[Dict[str, object]],
    lowpass_kernel: int,
    changed_threshold: float,
) -> Dict[str, object]:
    metrics = preview._metrics_with_support_comparison(
        composed,
        lowpass_kernel=lowpass_kernel,
        raw_support=raw_support,
        global_support=global_support,
    )
    main = dict(metrics)
    main["joint_gain"] = main.get("gain", float("nan"))
    main["joint_gain_capture"] = float(main["joint_gain"] / max(preview._positive_individual_gain(rows, "selection"), 1e-10))
    main.update(_changed_metrics(composed, changed_threshold))
    te = float(main.get("target_energy", 0.0))
    pe = float(main.get("pred_energy", 0.0))
    main["amplitude_ratio"] = float(math.sqrt(max(pe, 0.0) / max(te, 1e-10))) if te > 0.0 else float("nan")
    return main


def _per_view_for_composed(
    *,
    composed: Dict[str, Dict[str, np.ndarray]],
    raw_support: Dict[str, Dict[str, np.ndarray]],
    lowpass_kernel: int,
) -> Dict[str, Dict[str, float]]:
    out = {}
    for stem, data in composed.items():
        support = raw_support.get(stem, {})
        out[stem] = preview._metrics_for_view(
            data,
            lowpass_kernel=lowpass_kernel,
            support_weight=support.get("weight"),
            off_support_weight=support.get("off_weight"),
        )
    return out


def _best_shift_for_composed(
    *,
    composed: Dict[str, Dict[str, np.ndarray]],
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    rows: Sequence[Dict[str, object]],
    lowpass_kernel: int,
    changed_threshold: float,
    shift_values: Sequence[int],
) -> Dict[str, object]:
    best_gain: Optional[Dict[str, object]] = None
    best_corr: Optional[Dict[str, object]] = None
    for dy in shift_values:
        for dx in shift_values:
            shifted = _shift_composed(composed, dx=int(dx), dy=int(dy))
            metrics = _metrics_for_composed(
                composed=shifted,
                raw_support=raw_support,
                global_support=global_support,
                rows=rows,
                lowpass_kernel=lowpass_kernel,
                changed_threshold=changed_threshold,
            )
            row = {
                "dx": int(dx),
                "dy": int(dy),
                "gain": metrics.get("gain"),
                "HF_corr": metrics.get("HF_corr"),
                "explained_energy": metrics.get("explained_energy"),
                "positive_view_ratio": metrics.get("positive_view_ratio"),
            }
            gain = float(row["gain"]) if row.get("gain") is not None and math.isfinite(float(row["gain"])) else -1e30
            corr = float(row["HF_corr"]) if row.get("HF_corr") is not None and math.isfinite(float(row["HF_corr"])) else -1e30
            if best_gain is None or gain > float(best_gain.get("gain", -1e30)):
                best_gain = row
            if best_corr is None or corr > float(best_corr.get("HF_corr", -1e30)):
                best_corr = row
    return {
        "grid": [int(x) for x in shift_values],
        "best_by_gain": best_gain or {},
        "best_by_corr": best_corr or {},
    }


def _targeted_shift_report_for_variant(
    *,
    selected_ids: Sequence[int],
    rows: Sequence[Dict[str, object]],
    obs_by_cluster: Dict[int, List[oracle.EvalObs]],
    base_contributions: Dict[int, List[preview.Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    lowpass_kernel: int,
    changed_threshold: float,
    shift_values: Sequence[int],
    factor: float,
) -> Dict[str, object]:
    if not shift_values:
        return {}
    base_scaled = _scale_contributions(base_contributions, factor=float(factor))
    base_composed = static_v1._static_compose(
        selected_ids=selected_ids,
        contributions=base_scaled,
        stems=stems,
        state=state,
        args=args,
    )
    report: Dict[str, object] = {
        "grid": [int(x) for x in shift_values],
        "modes": {},
    }
    mode_rows: Dict[str, object] = {}
    mode_rows["whole_delta"] = _best_shift_for_composed(
        composed=base_composed,
        raw_support=raw_support,
        global_support=global_support,
        rows=rows,
        lowpass_kernel=lowpass_kernel,
        changed_threshold=changed_threshold,
        shift_values=shift_values,
    )
    # Replace target-only entries with target-shifted metrics; whole-delta helper shifts pred.
    best_gain: Optional[Dict[str, object]] = None
    best_corr: Optional[Dict[str, object]] = None
    for dy in shift_values:
        for dx in shift_values:
            shifted_target = _shift_target_composed(base_composed, dx=int(dx), dy=int(dy))
            metrics = _metrics_for_composed(
                composed=shifted_target,
                raw_support=raw_support,
                global_support=global_support,
                rows=rows,
                lowpass_kernel=lowpass_kernel,
                changed_threshold=changed_threshold,
            )
            row = {
                "dx": int(dx),
                "dy": int(dy),
                "gain": metrics.get("gain"),
                "HF_corr": metrics.get("HF_corr"),
                "explained_energy": metrics.get("explained_energy"),
                "positive_view_ratio": metrics.get("positive_view_ratio"),
            }
            gain = float(row["gain"]) if row.get("gain") is not None and math.isfinite(float(row["gain"])) else -1e30
            corr = float(row["HF_corr"]) if row.get("HF_corr") is not None and math.isfinite(float(row["HF_corr"])) else -1e30
            if best_gain is None or gain > float(best_gain.get("gain", -1e30)):
                best_gain = row
            if best_corr is None or corr > float(best_corr.get("HF_corr", -1e30)):
                best_corr = row
    mode_rows["target_only"] = {
        "grid": [int(x) for x in shift_values],
        "best_by_gain": best_gain or {},
        "best_by_corr": best_corr or {},
    }
    for shift_mode in ("phi", "q"):
        best_gain = None
        best_corr = None
        for dy in shift_values:
            for dx in shift_values:
                shifted_contrib = _build_shifted_basis_contributions(
                    rows=rows,
                    obs_by_cluster=obs_by_cluster,
                    args=args,
                    shift_mode=shift_mode,
                    dx=int(dx),
                    dy=int(dy),
                    factor=float(factor),
                )
                shifted_composed = static_v1._static_compose(
                    selected_ids=selected_ids,
                    contributions=shifted_contrib,
                    stems=stems,
                    state=state,
                    args=args,
                )
                metrics = _metrics_for_composed(
                    composed=shifted_composed,
                    raw_support=raw_support,
                    global_support=global_support,
                    rows=rows,
                    lowpass_kernel=lowpass_kernel,
                    changed_threshold=changed_threshold,
                )
                row = {
                    "dx": int(dx),
                    "dy": int(dy),
                    "gain": metrics.get("gain"),
                    "HF_corr": metrics.get("HF_corr"),
                    "explained_energy": metrics.get("explained_energy"),
                    "positive_view_ratio": metrics.get("positive_view_ratio"),
                }
                gain = float(row["gain"]) if row.get("gain") is not None and math.isfinite(float(row["gain"])) else -1e30
                corr = float(row["HF_corr"]) if row.get("HF_corr") is not None and math.isfinite(float(row["HF_corr"])) else -1e30
                if best_gain is None or gain > float(best_gain.get("gain", -1e30)):
                    best_gain = row
                if best_corr is None or corr > float(best_corr.get("HF_corr", -1e30)):
                    best_corr = row
        mode_rows[f"{shift_mode}_only"] = {
            "grid": [int(x) for x in shift_values],
            "best_by_gain": best_gain or {},
            "best_by_corr": best_corr or {},
        }
    report["modes"] = mode_rows
    return report


def _single_cell_gain(
    *,
    cid: int,
    contributions: Dict[int, List[preview.Contribution]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    rows: Sequence[Dict[str, object]],
    lowpass_kernel: int,
    changed_threshold: float,
    target_path: Optional[Path] = None,
    sign: float = 1.0,
    lamb: float = 1.0,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, np.ndarray]]]:
    scaled = _scale_contributions({int(cid): contributions.get(int(cid), [])}, factor=float(sign) * float(lamb))
    composed = static_v1._static_compose(
        selected_ids=[int(cid)],
        contributions=scaled,
        stems=stems,
        state=state,
        args=args,
    )
    if target_path is not None:
        composed = _retarget_composed(composed, target_dir=target_path, state=state, args=args)
    return (
        _metrics_for_composed(
            composed=composed,
            raw_support=raw_support,
            global_support=global_support,
            rows=rows,
            lowpass_kernel=lowpass_kernel,
            changed_threshold=changed_threshold,
        ),
        composed,
    )


def _build_per_cell_report(
    *,
    selected_ids: Sequence[int],
    rows: Sequence[Dict[str, object]],
    obs_by_q: Dict[str, Optional[Dict[int, List[oracle.EvalObs]]]],
    contrib_by_q: Dict[str, Dict[int, List[preview.Contribution]]],
    stems: Sequence[str],
    state: Dict[str, object],
    args: SimpleNamespace,
    raw_support: Dict[str, Dict[str, np.ndarray]],
    global_support: Dict[str, Dict[str, np.ndarray]],
    alt_target_path: Optional[Path],
    shift_values: Sequence[int],
    changed_threshold: float,
) -> Dict[str, object]:
    row_by_id = {preview._row_cluster_id(row): row for row in rows}
    cells: List[Dict[str, object]] = []
    threshold = float(getattr(args, "core_weight_threshold", 1e-4))
    for cid in selected_ids:
        cid = int(cid)
        row = row_by_id.get(cid, {})
        record: Dict[str, object] = {
            "cell_id": cid,
            "dev_selection_gain": row.get("C_selection_gain", row.get("selection_gain")),
            "dev_analysis_test_gain": row.get("C_test_gain", row.get("analysis_test_gain")),
        }
        for mode in ("proxy", "proxy_scaled", "true"):
            obs = obs_by_q.get(mode)
            if obs is None:
                record[f"q_{mode}_available"] = False
                continue
            record[f"q_{mode}_available"] = True
            q_stats = _obs_q_stats(obs.get(cid, []), threshold=threshold)
            record[f"q_{mode}_mean"] = q_stats["mean"]
            record[f"q_{mode}_median"] = q_stats["median"]
            record[f"q_{mode}_p95"] = q_stats["p95"]
            contrib = contrib_by_q.get(mode)
            if not contrib:
                continue
            metrics, composed = _single_cell_gain(
                cid=cid,
                contributions=contrib,
                stems=stems,
                state=state,
                args=args,
                raw_support=raw_support,
                global_support=global_support,
                rows=rows,
                lowpass_kernel=int(args.lowpass_kernel),
                changed_threshold=changed_threshold,
            )
            prefix = "GT_lockbox_gain_qtrue" if mode == "true" else f"GT_lockbox_gain_{mode}"
            record[prefix] = metrics.get("gain")
            record[f"{prefix}_corr"] = metrics.get("HF_corr")
            per_view_metrics = _per_view_for_composed(
                composed=composed,
                raw_support=raw_support,
                lowpass_kernel=int(args.lowpass_kernel),
            )
            gains = [float(v.get("gain", 0.0)) for v in per_view_metrics.values()]
            record[f"{mode}_negative_view_count"] = int(sum(1 for g in gains if g < 0.0))
            record[f"{mode}_positive_view_count"] = int(sum(1 for g in gains if g > 0.0))
            if mode == "proxy_scaled":
                flip_metrics, _flip_composed = _single_cell_gain(
                    cid=cid,
                    contributions=contrib,
                    stems=stems,
                    state=state,
                    args=args,
                    raw_support=raw_support,
                    global_support=global_support,
                    rows=rows,
                    lowpass_kernel=int(args.lowpass_kernel),
                    changed_threshold=changed_threshold,
                    sign=-1.0,
                    lamb=0.125,
                )
                record["sign_flip_gain_proxy_scaled_minus_l0125"] = flip_metrics.get("gain")
                if shift_values:
                    record["best_shift_proxy_scaled_plus_l1"] = _best_shift_for_composed(
                        composed=composed,
                        raw_support=raw_support,
                        global_support=global_support,
                        rows=rows,
                        lowpass_kernel=int(args.lowpass_kernel),
                        changed_threshold=changed_threshold,
                        shift_values=shift_values,
                    )
            if mode == "true":
                record["q_ratio_true_to_proxy_mean"] = float(
                    q_stats["mean"] / max(float(record.get("q_proxy_mean", 0.0)), 1e-8)
                )
        if alt_target_path is not None:
            mode = "true" if contrib_by_q.get("true") else "proxy_scaled"
            if contrib_by_q.get(mode):
                alt_metrics, _alt = _single_cell_gain(
                    cid=cid,
                    contributions=contrib_by_q[mode],
                    stems=stems,
                    state=state,
                    args=args,
                    raw_support=raw_support,
                    global_support=global_support,
                    rows=rows,
                    lowpass_kernel=int(args.lowpass_kernel),
                    changed_threshold=changed_threshold,
                    target_path=alt_target_path,
                )
                record[f"SR_lockbox_gain_{mode}"] = alt_metrics.get("gain")
                record[f"SR_lockbox_corr_{mode}"] = alt_metrics.get("HF_corr")
        cells.append(record)
    negative_proxy = [c for c in cells if float(c.get("GT_lockbox_gain_proxy_scaled", 0.0) or 0.0) < 0.0]
    return {
        "version": VERSION,
        "cell_count": int(len(cells)),
        "negative_proxy_scaled_count": int(len(negative_proxy)),
        "negative_proxy_scaled_ratio": float(len(negative_proxy) / max(len(cells), 1)),
        "cells": cells,
    }


def _save_visual_sample(output_dir: Path, name: str, composed: Dict[str, Dict[str, np.ndarray]], args: argparse.Namespace) -> None:
    root = output_dir / "visuals" / name
    for stem, data in composed.items():
        oracle._save_rgb(root / "base_plus_residual" / f"{stem}.png", data["preview"])
        oracle._save_rgb(root / "residual_pred_signed" / f"{stem}.png", np.clip(data["pred"] * float(args.visual_signed_scale) + 0.5, 0.0, 1.0))
        before = np.abs(data["target_hf"])
        after = np.abs(data["target_hf"] - data["pred"])
        oracle._save_rgb(root / "error_before" / f"{stem}.png", np.clip(before * float(args.error_scale), 0.0, 1.0))
        oracle._save_rgb(root / "error_after" / f"{stem}.png", np.clip(after * float(args.error_scale), 0.0, 1.0))


def main() -> None:
    t0 = time.time()
    cli = _parse_args()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    check_dir = Path(cli.check_dir).expanduser().resolve() if cli.check_dir else Path("")
    if output_dir.exists() and not cli.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    if output_dir.exists() and cli.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requires_true_q = any(
        token in " ".join([str(output_dir), str(check_dir), str(cli.q_modes)]).lower()
        for token in ("qtrue", "true_q", "true-donor", "true_donor")
    )
    if requires_true_q and not str(cli.q_parent_dir).strip():
        raise ValueError("This experiment name/config requests true donor q, but --q_parent_dir is empty; refusing fallback.")

    static_dir = Path(cli.static_v1_dir).expanduser().resolve()
    frozen = level1._load_frozen(static_dir)
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
    rows = rows_by_name[str(cli.cell_set)]
    selected_ids = ids_by_name[str(cli.cell_set)]
    all_ids = sorted(set(frozen["frozen_by_id"].keys()))

    proxy_args, proxy_state, proxy_obs_all = _load_state_and_obs(
        cli=cli,
        frozen=frozen,
        static_dir=static_dir,
        target_dir=str(cli.target_dir),
        q_parent_dir="",
        cluster_ids=all_ids,
    )
    stems = [view.stem for view in proxy_state["views"]]
    compose_state = {**proxy_state, "view_index_by_stem": {view.stem: int(view.view_index) for view in proxy_state["views"]}}
    proxy_stats = level1._q_stats_from_obs(proxy_obs_all, proxy_args)

    dev_ref = float(cli.dev_q_reference)
    proxy_ref = max(_safe_stat(proxy_stats, str(cli.q_scale_stat), default=dev_ref), 1e-8)
    q_scale = float(dev_ref / proxy_ref)
    q_reports: Dict[str, object] = {
        "proxy": {"source": "effective_hf_weight_fallback", "stats": proxy_stats},
        "proxy_scaled": {"source": "effective_hf_weight_fallback_scaled", "dev_q_reference": dev_ref, "scale_stat": cli.q_scale_stat, "scale": q_scale},
    }

    obs_by_q: Dict[str, Optional[Dict[int, List[oracle.EvalObs]]]] = {
        "proxy": proxy_obs_all,
        "proxy_scaled": _clone_obs_with_q_scale(proxy_obs_all, q_scale),
        "unit_visibility": _clone_obs_with_unit_q(proxy_obs_all),
    }
    q_reports["proxy_scaled"]["stats"] = level1._q_stats_from_obs(obs_by_q["proxy_scaled"], proxy_args)
    q_reports["unit_visibility"] = {"source": "unit_visibility_debug", "stats": {"mean": 1.0, "median": 1.0, "p95": 1.0, "max": 1.0}}
    if str(cli.q_parent_dir).strip():
        true_args, _true_state, true_obs_all = _load_state_and_obs(
            cli=cli,
            frozen=frozen,
            static_dir=static_dir,
            target_dir=str(cli.target_dir),
            q_parent_dir=str(cli.q_parent_dir),
            cluster_ids=all_ids,
        )
        obs_by_q["true"] = true_obs_all
        true_stats = level1._q_stats_from_obs(true_obs_all, true_args)
        q_reports["true"] = {
            "source": str(cli.q_parent_dir),
            "stats": true_stats,
            "ratio_to_proxy_mean": float(true_stats.get("mean", 0.0) / max(float(proxy_stats.get("mean", 0.0)), 1e-8)),
            "ratio_to_proxy_median": float(true_stats.get("median", 0.0) / max(float(proxy_stats.get("median", 0.0)), 1e-8)),
            "ratio_to_dev_reference_mean": float(true_stats.get("mean", 0.0) / max(dev_ref, 1e-8)),
        }
    else:
        obs_by_q["true"] = None
        q_reports["true"] = {"skipped": True, "reason": "q_parent_dir not provided"}

    base_contrib = preview._build_contributions(
        [cell["frozen_row"] for cid, cell in sorted(frozen["frozen_by_id"].items()) if proxy_obs_all.get(cid)],
        proxy_obs_all,
        proxy_args,
    )
    raw_support = preview._support_map_from_composed(
        static_v1._static_compose(
            selected_ids=selected_ids,
            contributions=base_contrib,
            stems=stems,
            state=compose_state,
            args=proxy_args,
        )
    )
    global_support = preview._support_map_from_composed(
        static_v1._static_compose(
            selected_ids=all_ids,
            contributions=base_contrib,
            stems=stems,
            state=compose_state,
            args=proxy_args,
        )
    )

    target_type = _target_type_from_text(str(cli.target_dir), str(cli.target_type))
    alt_target_type = _target_type_from_text(str(cli.alt_target_dir), str(cli.alt_target_type)) if str(cli.alt_target_dir).strip() else ""
    target_dirs: List[Tuple[str, Path]] = [(target_type, Path(cli.target_dir).expanduser().resolve())]
    if str(cli.alt_target_dir).strip():
        target_dirs.append((alt_target_type, Path(cli.alt_target_dir).expanduser().resolve()))

    metrics: Dict[str, object] = {
        "version": VERSION,
        "cell_set": str(cli.cell_set),
        "cell_count": int(len(selected_ids)),
        "lockbox_views": stems,
        "target_primary": target_type,
        "target_dirs": {name: str(path) for name, path in target_dirs},
        "q_modes_requested": _parse_csv_strings(cli.q_modes),
        "lambdas": _parse_csv_floats(cli.lambdas),
        "signs": _parse_csv_strings(cli.signs),
        "shift_grid": _parse_csv_ints(cli.shift_grid) if str(cli.shift_grid).strip() else [],
        "variants": {},
    }
    per_view: Dict[str, object] = {"version": VERSION, "variants": {}}
    shift_attribution_report: Dict[str, object] = {
        "version": VERSION,
        "note": "Diagnostic only; shifts are not a deployable model change.",
        "variants": {},
    }

    bounded_clip: Optional[float]
    bounded_mode = str(cli.bounded_mode)
    if bounded_mode == "from_config":
        config = frozen["config"]
        bounded_mode = str(config.get("bounded_mode", "per_cell_clip"))
        bounded_clip = float(config.get("bounded_delta_clip", 0.08))
    elif bounded_mode == "none":
        bounded_clip = None
    else:
        bounded_clip = float(cli.bounded_delta_clip) if float(cli.bounded_delta_clip) >= 0.0 else 0.08

    contrib_by_q: Dict[str, Dict[int, List[preview.Contribution]]] = {}
    for q_mode in _parse_csv_strings(cli.q_modes):
        obs = obs_by_q.get(q_mode)
        if obs is None:
            metrics["variants"][f"q={q_mode}"] = {"skipped": True, "reason": "q mode unavailable"}
            continue
        contrib_by_q[q_mode] = preview._build_contributions(
            [cell["frozen_row"] for cid, cell in sorted(frozen["frozen_by_id"].items()) if obs.get(cid)],
            obs,
            proxy_args,
        )

    visual_written = 0
    shift_values = _parse_csv_ints(cli.shift_grid) if str(cli.shift_grid).strip() else []
    per_cell_shift_values = _parse_csv_ints(cli.per_cell_shift_grid) if str(cli.per_cell_shift_grid).strip() else []
    shift_q_modes = set(_parse_csv_strings(cli.shift_q_modes))
    shift_signs = set(_parse_csv_strings(cli.shift_signs))
    shift_lambdas = {float(x) for x in _parse_csv_floats(cli.shift_lambdas)}
    for q_mode in _parse_csv_strings(cli.q_modes):
        obs = obs_by_q.get(q_mode)
        if obs is None:
            continue
        contrib = contrib_by_q.get(q_mode, {})
        for sign_name in _parse_csv_strings(cli.signs):
            sign = -1.0 if sign_name in {"minus", "-", "flip", "negative"} else 1.0
            for lamb in _parse_csv_floats(cli.lambdas):
                scaled = _scale_contributions(contrib, factor=sign * float(lamb))
                composed_primary = static_v1._static_compose(
                    selected_ids=selected_ids,
                    contributions=scaled,
                    stems=stems,
                    state=compose_state,
                    args=proxy_args,
                    bounded_clip=bounded_clip,
                    bounded_mode=bounded_mode,
                )
                for target_name, target_path in target_dirs:
                    composed = composed_primary if target_path == Path(cli.target_dir).expanduser().resolve() else _retarget_composed(
                        composed_primary,
                        target_dir=target_path,
                        state=compose_state,
                        args=proxy_args,
                    )
                    key = f"target={target_name}|q={q_mode}|sign={sign_name}|lambda={float(lamb):.4g}"
                    metrics["variants"][key] = _metrics_for_composed(
                        composed=composed,
                        raw_support=raw_support,
                        global_support=global_support,
                        rows=rows,
                        lowpass_kernel=int(proxy_args.lowpass_kernel),
                        changed_threshold=float(cli.changed_threshold),
                    )
                    if (
                        shift_values
                        and target_path == Path(cli.target_dir).expanduser().resolve()
                        and q_mode in shift_q_modes
                        and sign_name in shift_signs
                        and float(lamb) in shift_lambdas
                    ):
                        metrics["variants"][key]["shift_diagnostic"] = _best_shift_for_composed(
                            composed=composed,
                            raw_support=raw_support,
                            global_support=global_support,
                            rows=rows,
                            lowpass_kernel=int(proxy_args.lowpass_kernel),
                            changed_threshold=float(cli.changed_threshold),
                            shift_values=shift_values,
                        )
                        targeted = _targeted_shift_report_for_variant(
                            selected_ids=selected_ids,
                            rows=rows,
                            obs_by_cluster=obs,
                            base_contributions=contrib,
                            stems=stems,
                            state=compose_state,
                            args=proxy_args,
                            raw_support=raw_support,
                            global_support=global_support,
                            lowpass_kernel=int(proxy_args.lowpass_kernel),
                            changed_threshold=float(cli.changed_threshold),
                            shift_values=shift_values,
                            factor=sign * float(lamb),
                        )
                        metrics["variants"][key]["targeted_shift_diagnostic"] = targeted
                        shift_attribution_report["variants"][key] = targeted
                    per_view["variants"][key] = _per_view_for_composed(
                        composed=composed,
                        raw_support=raw_support,
                        lowpass_kernel=int(proxy_args.lowpass_kernel),
                    )
                    if int(cli.write_visuals) and visual_written < int(cli.visual_variant_limit):
                        safe_key = key.replace("|", "__").replace("=", "-").replace(".", "p")
                        _save_visual_sample(output_dir, safe_key, composed, cli)
                        visual_written += 1

    target_similarity = {}
    if len(target_dirs) > 1:
        target_similarity = _target_similarity(
            target_a_dir=target_dirs[0][1],
            target_b_dir=target_dirs[1][1],
            state=compose_state,
            args=proxy_args,
            active_threshold=float(cli.target_active_threshold),
        )
    alignment_sanity_report = _alignment_sanity_report(
        target_dirs=target_dirs,
        state=compose_state,
        args=proxy_args,
        shift_values=shift_values,
    )

    per_cell_report: Dict[str, object] = {}
    if int(cli.write_per_cell_report):
        per_cell_report = _build_per_cell_report(
            selected_ids=selected_ids,
            rows=rows,
            obs_by_q=obs_by_q,
            contrib_by_q=contrib_by_q,
            stems=stems,
            state=compose_state,
            args=proxy_args,
            raw_support=raw_support,
            global_support=global_support,
            alt_target_path=Path(cli.alt_target_dir).expanduser().resolve() if str(cli.alt_target_dir).strip() else None,
            shift_values=per_cell_shift_values,
            changed_threshold=float(cli.changed_threshold),
        )
    qtrue_status = {
        "version": VERSION,
        "requested": "true" in _parse_csv_strings(cli.q_modes),
        "available": bool(str(cli.q_parent_dir).strip()),
        "q_parent_dir": str(Path(cli.q_parent_dir).expanduser().resolve()) if str(cli.q_parent_dir).strip() else "",
        "status": "available" if str(cli.q_parent_dir).strip() else "unavailable",
        "note": (
            "Exact true donor q requires precomputed donor contribution maps. "
            "Current frozen_cells_3d only stores parent_index/source primitive geometry, not donor gaussian ids/weights."
        ),
        "report": q_reports.get("true", {}),
    }

    headline = {}
    for key, value in metrics["variants"].items():
        if not isinstance(value, dict) or value.get("skipped"):
            continue
        headline[key] = {
            "gain": value.get("gain"),
            "HF_corr": value.get("HF_corr"),
            "explained_energy": value.get("explained_energy"),
            "positive_view_ratio": value.get("positive_view_ratio"),
            "view_gain_min": value.get("view_gain_min"),
            "amplitude_ratio": value.get("amplitude_ratio"),
        }
    summary = {
        "version": VERSION,
        "runtime_sec": float(time.time() - t0),
        "cell_set": str(cli.cell_set),
        "target_primary": target_type,
        "q_report": q_reports,
        "headline": headline,
        "qtrue_status": qtrue_status,
        "next_reading_hint": "Look for q_proxy_scaled/lambda improvements before changing selector or cells.",
    }
    manifest = {
        "version": VERSION,
        "static_v1_dir": str(static_dir),
        "q_parent_modes": {
            "proxy": "proxy_effective_hf_weight",
            "proxy_scaled": "proxy_effective_hf_weight_scaled",
            "true": "true_donor" if str(cli.q_parent_dir).strip() else "unavailable",
            "unit_visibility": "unit_visibility",
        },
        "inputs": {
            "primitive_dir": str(Path(cli.primitive_dir).expanduser().resolve()),
            "base_render_dir": str(Path(cli.base_render_dir).expanduser().resolve()),
            "target_dir": str(Path(cli.target_dir).expanduser().resolve()),
            "alt_target_dir": str(Path(cli.alt_target_dir).expanduser().resolve()) if str(cli.alt_target_dir).strip() else "",
            "weight_dir": str(Path(cli.weight_dir).expanduser().resolve()),
            "q_parent_dir": str(Path(cli.q_parent_dir).expanduser().resolve()) if str(cli.q_parent_dir).strip() else "",
            "match_policy": str(cli.match_policy),
            "camera_index_offset": int(cli.camera_index_offset),
            "limit": int(cli.limit),
        },
        "frozen": {
            "cell_set": str(cli.cell_set),
            "cell_ids": [int(x) for x in selected_ids],
        },
        "forbidden_actions": [
            "no selector/top-k change",
            "no beta refit",
            "no cell deletion",
            "no piece/slot/support retuning",
        ],
    }

    _write_json(output_dir / "failure_attribution_metrics.json", metrics)
    _write_json(output_dir / "failure_attribution_per_view.json", per_view)
    _write_json(output_dir / "q_distribution_report.json", q_reports)
    _write_json(output_dir / "target_similarity_report.json", target_similarity)
    _write_json(output_dir / "shift_attribution_report.json", shift_attribution_report)
    _write_json(output_dir / "alignment_sanity_report.json", alignment_sanity_report)
    _write_json(output_dir / "qtrue_status_or_qtrue_metrics.json", qtrue_status)
    _write_json(output_dir / "per_cell_failure_report.json", per_cell_report)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "per_view_metrics.json", per_view)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", manifest)
    if str(check_dir):
        _copy_to_check(output_dir, check_dir)
    print(f"[failure-attribution-v0] summary: {output_dir / 'summary.json'}")
    print(f"[failure-attribution-v0] metrics: {output_dir / 'failure_attribution_metrics.json'}")
    if str(check_dir):
        print(f"[failure-attribution-v0] check  : {check_dir}")


if __name__ == "__main__":
    main()
