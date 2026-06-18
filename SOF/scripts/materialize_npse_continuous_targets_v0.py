#!/usr/bin/env python3
"""Materialize continuous-region NPSE supervision from an existing cache.

The original N-PSE cache already stores residual_npse, continuous_mask, and
trust_sr in npz payloads. This utility turns those arrays into trainable image
assets without rebuilding the full depth/edge cache:

  continuous_target = anchor + residual_npse * continuous_mask
  trust_continuous  = trust_sr * continuous_mask
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npse_cache_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _load_rgb01(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb01(path: Path, rgb: np.ndarray) -> None:
    arr = np.clip(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    _atomic_save(Image.fromarray(arr, mode="RGB"), path)


def _save_gray01(path: Path, gray: np.ndarray) -> None:
    arr = np.clip(np.rint(np.clip(gray, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    _atomic_save(Image.fromarray(arr, mode="L"), path)


def _atomic_save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        image.save(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    args = _parse_args()
    root = args.npse_cache_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    frames = list(manifest.get("frames") or [])
    if int(args.limit) > 0:
        frames = frames[: int(args.limit)]
    if not frames:
        raise RuntimeError(f"manifest has no frames: {manifest_path}")

    out_target = root / "continuous_target"
    out_trust = root / "trust_continuous"
    written = 0
    skipped = 0
    trust_means: list[float] = []
    continuous_ratios: list[float] = []

    for frame in tqdm(frames, desc="materialize continuous targets"):
        stem = str(frame["stem"])
        target_path = out_target / f"{stem}.png"
        trust_path = out_trust / f"{stem}.png"
        if target_path.exists() and trust_path.exists() and not bool(args.overwrite):
            skipped += 1
            continue

        npz_path = Path(str(frame.get("npz") or root / "npz" / f"{stem}.npz"))
        if not npz_path.is_file():
            raise FileNotFoundError(f"npz not found for {stem}: {npz_path}")
        anchor_path = Path(str(frame.get("anchor_path") or ""))
        if not anchor_path.is_file():
            raise FileNotFoundError(f"anchor image not found for {stem}: {anchor_path}")

        with np.load(npz_path) as payload:
            residual_npse = payload["residual_npse"].astype(np.float32)
            continuous_mask = payload["continuous_mask"].astype(np.float32)
            trust_sr = payload["trust_sr"].astype(np.float32)

        anchor = _load_rgb01(anchor_path, size=(residual_npse.shape[1], residual_npse.shape[0]))
        trust_continuous = np.clip(trust_sr * continuous_mask, 0.0, 1.0).astype(np.float32)
        continuous_target = np.clip(anchor + residual_npse * continuous_mask[..., None], 0.0, 1.0)

        _save_rgb01(target_path, continuous_target)
        _save_gray01(trust_path, trust_continuous)
        written += 1
        trust_means.append(float(trust_continuous.mean()))
        continuous_ratios.append(float(continuous_mask.mean()))

    summary = {
        "npse_cache_root": str(root),
        "continuous_target": str(out_target),
        "trust_continuous": str(out_trust),
        "num_frames": len(frames),
        "num_written": written,
        "num_skipped": skipped,
        "trust_continuous_mean": None if not trust_means else float(np.mean(trust_means)),
        "continuous_ratio_mean": None if not continuous_ratios else float(np.mean(continuous_ratios)),
    }
    summary_path = root / "continuous_targets_manifest.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[npse-continuous-v0] summary: {summary_path}")


if __name__ == "__main__":
    main()
