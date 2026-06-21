#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate how well 2DGS carrier renders express their carrier targets."
    )
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--active_threshold", type=float, default=0.05)
    parser.add_argument("--peak", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(paths: Sequence[Path], lookup: Dict[str, Path], ref: Path, index: int, policy: str) -> Optional[Path]:
    if policy in {"stem", "order_if_needed"}:
        found = lookup.get(ref.stem.lower())
        if found is not None:
            return found
        if policy == "stem":
            return None
    if policy in {"order", "order_if_needed"} and index < len(paths):
        return paths[index]
    return None


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _mse(a: np.ndarray, b: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    channels = diff.shape[2] if diff.ndim == 3 else 1
    if weight is None:
        return float(diff.mean())
    weight = np.clip(weight.astype(np.float32), 0.0, 1.0)
    denom = float(weight.sum()) * channels
    if denom <= 1e-8:
        return float("nan")
    if diff.ndim == 3:
        return float((diff * weight[..., None]).sum() / denom)
    return float((diff * weight).sum() / denom)


def _psnr(a: np.ndarray, b: np.ndarray, weight: Optional[np.ndarray] = None, peak: float = 1.0) -> float:
    mse = _mse(a, b, weight)
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return 99.0
    peak = max(float(peak), 1e-8)
    return float(20.0 * math.log10(peak) - 10.0 * math.log10(mse))


def _weighted_l1(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (a.shape[2] if a.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    return float((np.abs(a - b) * weight[..., None]).sum() / denom)


def _weighted_energy(value: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (value.shape[2] if value.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    return float((np.abs(value) * weight[..., None]).sum() / denom)


def _pearson_abs(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    aa = np.abs(a).mean(axis=2).reshape(-1)
    bb = np.abs(b).mean(axis=2).reshape(-1)
    ww = weight.reshape(-1)
    keep = ww > 1e-6
    if int(keep.sum()) < 4:
        return float("nan")
    aa = aa[keep]
    bb = bb[keep]
    ww = ww[keep]
    ww = ww / max(float(ww.sum()), 1e-8)
    ma = float((aa * ww).sum())
    mb = float((bb * ww).sum())
    da = aa - ma
    db = bb - mb
    den = float(np.sqrt((ww * da * da).sum() * (ww * db * db).sum()))
    if den <= 1e-8:
        return float("nan")
    return float((ww * da * db).sum() / den)


def _mean(rows: Sequence[Dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = _parse_args()
    target_dir = Path(args.target_dir).expanduser().resolve()
    render_dir = Path(args.render_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve() if str(args.mask_dir) else None
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {output_dir}")
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_paths = _list_images(target_dir)
    if int(args.limit) > 0:
        target_paths = target_paths[: int(args.limit)]
    render_paths = _list_images(render_dir)
    mask_paths = _list_images(mask_dir) if mask_dir is not None and mask_dir.is_dir() else []
    render_lookup = _lookup(render_paths)
    mask_lookup = _lookup(mask_paths)

    rows: List[Dict[str, float]] = []
    frames: List[Dict[str, object]] = []
    for index, target_path in enumerate(tqdm(target_paths, desc="2DGS carrier PSNR")):
        render_path = _resolve(render_paths, render_lookup, target_path, index, str(args.match_policy))
        if render_path is None:
            continue
        target = _load_rgb(target_path)
        size = (target.shape[1], target.shape[0])
        render = _load_rgb(render_path, size=size)
        if mask_paths:
            mask_path = _resolve(mask_paths, mask_lookup, target_path, index, str(args.match_policy))
            mask = _load_gray(mask_path, size=size) if mask_path is not None else np.ones(target.shape[:2], dtype=np.float32)
        else:
            mask_path = None
            mask = np.maximum(target.max(axis=2), render.max(axis=2))
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        active = (mask >= float(args.active_threshold)).astype(np.float32)
        diff = render - target
        row = {
            "index": float(index),
            "psnr": _psnr(render, target, peak=float(args.peak)),
            "psnr_weighted": _psnr(render, target, mask, peak=float(args.peak)),
            "psnr_active": _psnr(render, target, active, peak=float(args.peak)),
            "l1_weighted": _weighted_l1(render, target, mask),
            "target_energy": _weighted_energy(target, mask),
            "render_energy": _weighted_energy(render, mask),
            "error_energy": _weighted_energy(diff, mask),
            "corr_abs": _pearson_abs(render, target, mask),
            "mask_mean": float(mask.mean()),
            "active_ratio": float(active.mean()),
        }
        row["energy_ratio"] = row["render_energy"] / max(row["target_energy"], 1e-8)
        rows.append(row)
        frames.append(
            {
                "stem": target_path.stem,
                "target": str(target_path),
                "render": str(render_path),
                "mask": str(mask_path) if mask_path is not None else None,
                "metrics": row,
            }
        )

    summary = {
        "version": "evaluate_2dgs_carrier_expression_v0",
        "target_dir": str(target_dir),
        "render_dir": str(render_dir),
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "output_dir": str(output_dir),
        "match_policy": str(args.match_policy),
        "num_frames": len(rows),
        "psnr_mean": _mean(rows, "psnr"),
        "psnr_weighted_mean": _mean(rows, "psnr_weighted"),
        "psnr_active_mean": _mean(rows, "psnr_active"),
        "l1_weighted_mean": _mean(rows, "l1_weighted"),
        "corr_abs_mean": _mean(rows, "corr_abs"),
        "target_energy_mean": _mean(rows, "target_energy"),
        "render_energy_mean": _mean(rows, "render_energy"),
        "energy_ratio_mean": _mean(rows, "energy_ratio"),
        "error_energy_mean": _mean(rows, "error_energy"),
        "mask_mean": _mean(rows, "mask_mean"),
        "active_ratio_mean": _mean(rows, "active_ratio"),
        "frames": frames,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "index",
            "psnr",
            "psnr_weighted",
            "psnr_active",
            "l1_weighted",
            "target_energy",
            "render_energy",
            "energy_ratio",
            "error_energy",
            "corr_abs",
            "mask_mean",
            "active_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2, ensure_ascii=False))
    print(f"[2dgs-carrier-psnr-v0] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
