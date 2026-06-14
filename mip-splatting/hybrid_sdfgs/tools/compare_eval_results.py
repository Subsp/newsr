#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two mip-splatting evaluation outputs (results.json / per_view.json) "
            "and save a compact summary JSON."
        )
    )
    parser.add_argument("--baseline_model", type=Path, required=True)
    parser.add_argument("--current_model", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--baseline_method", type=str, default="")
    parser.add_argument("--current_method", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _method_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return (int(match.group(1)), name)
    return (-1, name)


def _pick_method(results: dict, explicit_name: str) -> str:
    if explicit_name:
        if explicit_name not in results:
            raise KeyError(f"Method '{explicit_name}' not found. Available: {sorted(results.keys())}")
        return explicit_name
    if len(results) == 1:
        return next(iter(results.keys()))
    return sorted(results.keys(), key=_method_sort_key)[-1]


def _extract_main_metrics(results: dict, method: str) -> dict[str, float]:
    item = results[method]
    summary = {}
    for key in ("PSNR", "SSIM", "LPIPS"):
        if key in item:
            summary[key.lower()] = float(item[key])
    return summary


def _extract_per_view(per_view: dict, method: str) -> dict[str, dict[str, float]]:
    item = per_view.get(method, {})
    out: dict[str, dict[str, float]] = {}
    keys = [k for k in ("PSNR", "SSIM", "LPIPS") if k in item]
    if not keys:
        return out
    common = None
    for key in keys:
        names = set(item[key].keys())
        common = names if common is None else (common & names)
    if not common:
        return out
    for name in sorted(common):
        out[name] = {key.lower(): float(item[key][name]) for key in keys}
    return out


def main() -> None:
    args = _parse_args()
    baseline_results = _load_json(args.baseline_model / "results.json")
    current_results = _load_json(args.current_model / "results.json")

    baseline_method = _pick_method(baseline_results, args.baseline_method)
    current_method = _pick_method(current_results, args.current_method)

    baseline_summary = _extract_main_metrics(baseline_results, baseline_method)
    current_summary = _extract_main_metrics(current_results, current_method)

    delta = {}
    for key in sorted(set(baseline_summary.keys()) & set(current_summary.keys())):
        delta[key] = float(current_summary[key] - baseline_summary[key])

    summary = {
        "baseline_model": str(args.baseline_model),
        "current_model": str(args.current_model),
        "baseline_method": baseline_method,
        "current_method": current_method,
        "baseline": baseline_summary,
        "current": current_summary,
        "delta": delta,
    }

    baseline_per_view_path = args.baseline_model / "per_view.json"
    current_per_view_path = args.current_model / "per_view.json"
    if baseline_per_view_path.is_file() and current_per_view_path.is_file():
        baseline_per_view = _extract_per_view(_load_json(baseline_per_view_path), baseline_method)
        current_per_view = _extract_per_view(_load_json(current_per_view_path), current_method)
        common_names = sorted(set(baseline_per_view.keys()) & set(current_per_view.keys()))
        if common_names:
            per_view_delta = {}
            for name in common_names:
                metric_delta = {}
                for key in sorted(set(baseline_per_view[name].keys()) & set(current_per_view[name].keys())):
                    metric_delta[key] = float(current_per_view[name][key] - baseline_per_view[name][key])
                per_view_delta[name] = metric_delta
            summary["per_view"] = {
                "n_common_views": len(common_names),
                "delta": per_view_delta,
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved to: {args.output_json}")


if __name__ == "__main__":
    main()
