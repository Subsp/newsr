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


VERSION = "evaluate_residual_tetris_failure_attribution_v0"


def _parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


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
    parser.add_argument("--q_modes", default="proxy,proxy_scaled,true")
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
        "failure_attribution_metrics.json",
        "failure_attribution_per_view.json",
        "q_distribution_report.json",
        "target_similarity_report.json",
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
        ee = 1.0 - float(np.sum(w[..., None] * (ta - tb) ** 2)) / max(float(np.sum(w[..., None] * ta * ta)), 1e-10)
        rows.append({"stem": stem, "corr": float("nan") if corr is None else float(corr), "explained_energy": float(ee), "sign_agreement": sign_agree})
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
        "per_view": rows,
    }


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
    }
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
        q_reports["true"] = {"source": str(cli.q_parent_dir), "stats": level1._q_stats_from_obs(true_obs_all, true_args)}
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
        "variants": {},
    }
    per_view: Dict[str, object] = {"version": VERSION, "variants": {}}

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

    visual_written = 0
    for q_mode in _parse_csv_strings(cli.q_modes):
        obs = obs_by_q.get(q_mode)
        if obs is None:
            metrics["variants"][f"q={q_mode}"] = {"skipped": True, "reason": "q mode unavailable"}
            continue
        contrib = preview._build_contributions(
            [cell["frozen_row"] for cid, cell in sorted(frozen["frozen_by_id"].items()) if obs.get(cid)],
            obs,
            proxy_args,
        )
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
        )

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
        "next_reading_hint": "Look for q_proxy_scaled/lambda improvements before changing selector or cells.",
    }
    manifest = {
        "version": VERSION,
        "static_v1_dir": str(static_dir),
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
