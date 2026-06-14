#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two results_psnr_ssim.json files and save a compact summary."
    )
    parser.add_argument("--baseline_json", type=Path, required=True)
    parser.add_argument("--current_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_json.resolve())
    current = load_json(args.current_json.resolve())

    summary = {
        "baseline_json": str(args.baseline_json.resolve()),
        "current_json": str(args.current_json.resolve()),
        "baseline": {
            "model_dir": baseline.get("model_dir", ""),
            "split": baseline.get("split", ""),
            "iteration": baseline.get("iteration", 0),
            "resolution": baseline.get("resolution", 0),
            "n_views": baseline.get("n_views", 0),
            "PSNR": float(baseline["PSNR"]),
            "SSIM": float(baseline["SSIM"]),
        },
        "current": {
            "model_dir": current.get("model_dir", ""),
            "split": current.get("split", ""),
            "iteration": current.get("iteration", 0),
            "resolution": current.get("resolution", 0),
            "n_views": current.get("n_views", 0),
            "PSNR": float(current["PSNR"]),
            "SSIM": float(current["SSIM"]),
        },
        "delta": {
            "PSNR": float(current["PSNR"] - baseline["PSNR"]),
            "SSIM": float(current["SSIM"] - baseline["SSIM"]),
        },
    }

    baseline_views = baseline.get("per_view", {})
    current_views = current.get("per_view", {})
    common_names = sorted(set(baseline_views.keys()) & set(current_views.keys()))
    if common_names:
        summary["per_view"] = {
            "n_common_views": len(common_names),
            "delta": {
                name: {
                    "PSNR": float(current_views[name]["PSNR"] - baseline_views[name]["PSNR"]),
                    "SSIM": float(current_views[name]["SSIM"] - baseline_views[name]["SSIM"]),
                }
                for name in common_names
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved to: {args.output_json}")


if __name__ == "__main__":
    main()
