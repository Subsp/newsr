#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PREFERRED_ORDER = [
    "full",
    "children_only",
    "non_children_only",
    "softened_children_only",
    "unsoftened_children_only",
    "children_removed_full",
    "softened_removed_full",
    "children_restsh_zero",
    "children_tau_0p5",
    "children_scale_0p7",
    "children_filter_0p5",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sort_key(item: Dict[str, Any]) -> tuple[int, str]:
    label = str(item.get("label", ""))
    try:
        return (PREFERRED_ORDER.index(label), label)
    except ValueError:
        return (len(PREFERRED_ORDER), label)


def _maybe_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _variant_row(label: str, summary_path: Path, split: str) -> Dict[str, Any]:
    summary = _load_json(summary_path)
    selection = dict(summary.get("selection", {}))
    ablation = dict(summary.get("ablation", {}))
    renders = dict(summary.get("renders", {})).get(split, {})
    images_subdir = str(summary.get("images_subdir", "images"))
    contact_sheet = summary_path.parent / f"contact_sheet_{images_subdir}_{split}.png"
    return {
        "label": label,
        "selected_gaussians": selection.get("selected_gaussians"),
        "source_gaussians": selection.get("source_gaussians"),
        "selected_ratio": selection.get("selected_ratio"),
        "selection_key": selection.get("key"),
        "selection_mode": selection.get("selection_mode"),
        "rest_scale": ablation.get("rest_scale"),
        "dc_scale": ablation.get("dc_scale"),
        "tau_scale": ablation.get("tau_scale"),
        "scale_multiplier": ablation.get("scale_multiplier"),
        "scale_axis_mode": ablation.get("scale_axis_mode"),
        "filter_multiplier": ablation.get("filter_multiplier"),
        "render_root": renders.get("render_root"),
        "alpha_root": renders.get("alpha_root"),
        "depth_root": renders.get("depth_root"),
        "premul_root": renders.get("premul_root"),
        "contact_sheet": _maybe_path(contact_sheet),
        "summary_path": str(summary_path),
    }


def _find_row(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any] | None:
    for row in rows:
        if row.get("label") == label:
            return row
    return None


def _count(row: Dict[str, Any] | None) -> int | None:
    if row is None or row.get("selected_gaussians") is None:
        return None
    return int(row["selected_gaussians"])


def _build_sanity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    full = _count(_find_row(rows, "full"))
    children = _count(_find_row(rows, "children_only"))
    non_children = _count(_find_row(rows, "non_children_only"))
    softened_children = _count(_find_row(rows, "softened_children_only"))
    unsoftened_children = _count(_find_row(rows, "unsoftened_children_only"))
    checks: Dict[str, Any] = {
        "full_count": full,
        "children_count": children,
        "non_children_count": non_children,
        "softened_children_count": softened_children,
        "unsoftened_children_count": unsoftened_children,
    }
    if full is not None and children is not None and non_children is not None:
        checks["children_plus_non_children_matches_full"] = (children + non_children) == full
        checks["children_ratio"] = float(children / max(full, 1))
    if children is not None and softened_children is not None and unsoftened_children is not None:
        checks["softened_plus_unsoftened_matches_children"] = (softened_children + unsoftened_children) == children
        checks["softened_children_ratio_of_children"] = float(softened_children / max(children, 1))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prepare lineage diagnostic variant outputs.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--summary_name", default="lineage_diagnostics_index.json")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output_root not found: {output_root}")

    rows: List[Dict[str, Any]] = []
    for child in sorted(output_root.iterdir()):
        summary_path = child / "summary.json"
        if child.is_dir() and summary_path.is_file():
            rows.append(_variant_row(child.name, summary_path, str(args.split)))
    rows = sorted(rows, key=_sort_key)
    payload = {
        "mode": "summarize_prepare_lineage_diagnostics_v0",
        "output_root": str(output_root),
        "split": str(args.split),
        "variant_count": int(len(rows)),
        "sanity": _build_sanity(rows),
        "variants": rows,
    }
    out_path = output_root / str(args.summary_name)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
