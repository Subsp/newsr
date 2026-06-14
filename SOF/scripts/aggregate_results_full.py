#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-scene results_full.json files into an overall summary."
    )
    parser.add_argument("--model_dirs", nargs="+", required=True, help="Scene model directories.")
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path.resolve()

    per_scene: dict[str, dict[str, dict[str, float]]] = {}
    aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing: list[str] = []

    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str).resolve()
        scene_name = model_dir.name
        results_path = model_dir / "results_full.json"
        if not results_path.is_file():
            missing.append(str(model_dir))
            continue

        data = json.loads(results_path.read_text())
        per_scene[scene_name] = data
        for method, metrics in data.items():
            for key in ("PSNR", "SSIM"):
                if key in metrics:
                    aggregate[method][key].append(float(metrics[key]))

    overall: dict[str, dict[str, float | int]] = {}
    for method, metrics in aggregate.items():
        overall[method] = {
            "scene_count": len(metrics.get("PSNR", [])),
        }
        for key, values in metrics.items():
            if values:
                overall[method][f"mean_{key.lower()}"] = mean(values)

    summary = {
        "per_scene": per_scene,
        "overall": overall,
        "missing_results": missing,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[aggregate-results-full] done")
    print(f"  output_path   : {output_path}")
    print(f"  scene_count   : {len(per_scene)}")
    print(f"  missing_count : {len(missing)}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
