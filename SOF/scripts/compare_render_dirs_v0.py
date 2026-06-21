#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two render directories and export amplified image deltas.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--current_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--match_policy", choices=["stem", "order"], default="stem")
    parser.add_argument("--vis_scale", type=float, default=20.0)
    parser.add_argument("--change_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0.0, 1.0)
    Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(path)


def _pairs(base_paths: List[Path], current_paths: List[Path], policy: str):
    if policy == "order":
        for base, current in zip(base_paths, current_paths):
            yield base.stem, base, current
        return
    current_by_stem: Dict[str, Path] = {p.stem: p for p in current_paths}
    for base in base_paths:
        current = current_by_stem.get(base.stem)
        if current is not None:
            yield base.stem, base, current


def _psnr(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    current_dir = Path(args.current_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_paths = _list_images(base_dir)
    current_paths = _list_images(current_dir)
    rows: List[Dict[str, object]] = []
    for index, (stem, base_path, current_path) in enumerate(_pairs(base_paths, current_paths, str(args.match_policy))):
        if int(args.limit) > 0 and index >= int(args.limit):
            break
        base = _load_rgb(base_path)
        current = _load_rgb(current_path)
        if base.shape != current.shape:
            current = np.asarray(
                Image.open(current_path).convert("RGB").resize((base.shape[1], base.shape[0]), Image.BICUBIC),
                dtype=np.float32,
            ) / 255.0
        diff = current - base
        abs_diff = np.abs(diff)
        mse = float(np.mean(diff * diff))
        l1 = float(np.mean(abs_diff))
        max_abs = float(np.max(abs_diff))
        p99_abs = float(np.percentile(abs_diff, 99.0))
        changed = float(np.mean(np.max(abs_diff, axis=2) > float(args.change_threshold)))
        _save_rgb(output_dir / f"{stem}.png", abs_diff * float(args.vis_scale))
        rows.append(
            {
                "stem": stem,
                "base": str(base_path),
                "current": str(current_path),
                "psnr": _psnr(mse),
                "mse": mse,
                "l1": l1,
                "max_abs": max_abs,
                "p99_abs": p99_abs,
                "changed_ratio": changed,
            }
        )

    if not rows:
        raise RuntimeError(f"No common frames found: {base_dir} vs {current_dir}")
    finite_psnr = [float(r["psnr"]) for r in rows if math.isfinite(float(r["psnr"]))]
    summary = {
        "version": "compare_render_dirs_v0",
        "base_dir": str(base_dir),
        "current_dir": str(current_dir),
        "output_dir": str(output_dir),
        "num_frames": len(rows),
        "vis_scale": float(args.vis_scale),
        "change_threshold": float(args.change_threshold),
        "psnr_mean": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "l1_mean": float(np.mean([float(r["l1"]) for r in rows])),
        "max_abs_mean": float(np.mean([float(r["max_abs"]) for r in rows])),
        "p99_abs_mean": float(np.mean([float(r["p99_abs"]) for r in rows])),
        "changed_ratio_mean": float(np.mean([float(r["changed_ratio"]) for r in rows])),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
