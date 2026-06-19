#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def _load_manifest(root: Path) -> Dict[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_npz(root: Path) -> Iterable[Path]:
    per_view = root / "per_view"
    if not per_view.is_dir():
        raise FileNotFoundError(f"Missing per_view directory: {per_view}")
    yield from sorted(per_view.glob("*.npz"))


def _array_or(data: np.lib.npyio.NpzFile, names: Tuple[str, ...], fallback: np.ndarray) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float32).reshape(-1)
    return fallback.astype(np.float32, copy=False).reshape(-1)


def _max_at(target: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    if ids.size == 0:
        return
    np.maximum.at(target, ids, values.astype(target.dtype, copy=False))


def _sum_at(target: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    if ids.size == 0:
        return
    np.add.at(target, ids, values.astype(target.dtype, copy=False))


def _select_threshold(
    score: np.ndarray,
    candidate: np.ndarray,
    *,
    keep_ratio: float,
    min_score: float,
) -> float:
    values = score[candidate]
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    if values.size == 0:
        return float(min_score)
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio <= 0.0 or keep_ratio >= 1.0:
        return float(min_score)
    percentile = 100.0 * (1.0 - keep_ratio)
    return max(float(min_score), float(np.percentile(values, percentile)))


def _bool_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.bool_, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate CAVE HF ownership diagnostics into Gaussian HF/LF reassignment masks."
    )
    parser.add_argument("--cave_cache_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_gaussians", type=int, default=0)
    parser.add_argument("--score_mode", default="image_hit_consistency", choices=["touch", "hit", "validated", "image_hit_consistency"])
    parser.add_argument("--consistency_weight", type=float, default=0.25)
    parser.add_argument("--view_support_weight", type=float, default=0.15)
    parser.add_argument("--view_support_cap", type=float, default=3.0)
    parser.add_argument("--hf_keep_ratio", type=float, default=0.50)
    parser.add_argument("--min_hf_score", type=float, default=0.05)
    parser.add_argument("--min_touch_views", type=int, default=1)
    parser.add_argument("--min_validated_score", type=float, default=0.0)
    parser.add_argument("--uncertain_score_margin", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.cave_cache_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    manifest = _load_manifest(root)
    num_gaussians = int(args.num_gaussians)
    if num_gaussians <= 0:
        num_gaussians = int(manifest.get("num_gaussians") or 0)
    if num_gaussians <= 0:
        raise ValueError("num_gaussians is required when manifest.json does not contain num_gaussians")

    touch_count = np.zeros((num_gaussians,), dtype=np.uint16)
    hit_sum = np.zeros((num_gaussians,), dtype=np.float32)
    validated_sum = np.zeros((num_gaussians,), dtype=np.float32)
    consistency_sum = np.zeros((num_gaussians,), dtype=np.float32)
    touch_max = np.zeros((num_gaussians,), dtype=np.float32)
    hit_max = np.zeros((num_gaussians,), dtype=np.float32)
    validated_max = np.zeros((num_gaussians,), dtype=np.float32)
    consistency_max = np.zeros((num_gaussians,), dtype=np.float32)

    frames: List[Dict[str, object]] = []
    npz_paths = list(_iter_npz(root))
    if not npz_paths:
        raise FileNotFoundError(f"No per-view npz files found under {root / 'per_view'}")

    for path in npz_paths:
        with np.load(path) as data:
            ids = np.asarray(data["gaussian_id"], dtype=np.int64).reshape(-1)
            valid = (ids >= 0) & (ids < num_gaussians)
            ids = ids[valid]
            if ids.size == 0:
                continue
            fallback = np.ones((ids.size,), dtype=np.float32)
            hit = _array_or(data, ("hit_norm", "hit"), fallback)[valid]
            validated = _array_or(data, ("score",), np.zeros_like(fallback))[valid]
            consistency = _array_or(data, ("consistency",), np.ones_like(fallback))[valid]
            touch = _array_or(data, ("touch_rank", "carrier_values", "hit_norm", "hit"), fallback)[valid]

            np.add.at(touch_count, ids, np.ones_like(ids, dtype=np.uint16))
            _sum_at(hit_sum, ids, hit)
            _sum_at(validated_sum, ids, validated)
            _sum_at(consistency_sum, ids, consistency)
            _max_at(touch_max, ids, touch)
            _max_at(hit_max, ids, hit)
            _max_at(validated_max, ids, validated)
            _max_at(consistency_max, ids, consistency)
            frames.append({"name": path.stem, "gaussians": int(ids.size)})

    touched = touch_count >= max(int(args.min_touch_views), 1)
    safe_count = np.maximum(touch_count.astype(np.float32), 1.0)
    hit_mean = hit_sum / safe_count
    validated_mean = validated_sum / safe_count
    consistency_mean = consistency_sum / safe_count

    consistency_weight = float(np.clip(args.consistency_weight, 0.0, 1.0))
    view_support_weight = float(np.clip(args.view_support_weight, 0.0, 1.0))
    view_factor = np.clip(touch_count.astype(np.float32) / max(float(args.view_support_cap), 1.0), 0.0, 1.0)

    if str(args.score_mode) == "touch":
        base_score = touch_max
    elif str(args.score_mode) == "hit":
        base_score = np.maximum(hit_max, hit_mean)
    elif str(args.score_mode) == "validated":
        base_score = np.maximum(validated_max, validated_mean)
    else:
        image_score = np.maximum.reduce([touch_max, hit_max, hit_mean])
        consistency_gate = (1.0 - consistency_weight) + consistency_weight * np.clip(consistency_max, 0.0, 1.0)
        support_gate = (1.0 - view_support_weight) + view_support_weight * view_factor
        base_score = image_score * consistency_gate * support_gate

    validated_gate = validated_max >= float(args.min_validated_score)
    candidate = touched & validated_gate & (base_score > 0.0)
    threshold = _select_threshold(
        base_score,
        candidate,
        keep_ratio=float(args.hf_keep_ratio),
        min_score=float(args.min_hf_score),
    )
    hf_owned = candidate & (base_score >= threshold)
    hf_candidate = touched & (base_score > 0.0)
    uncertain_threshold = max(float(args.min_hf_score), threshold * float(args.uncertain_score_margin))
    hf_uncertain = hf_candidate & (~hf_owned) & (base_score >= uncertain_threshold)
    lf_owned = ~hf_owned
    lf_safe = ~hf_candidate

    payload = {
        "hf_owned": _bool_tensor(hf_owned),
        "hf_candidate": _bool_tensor(hf_candidate),
        "hf_uncertain": _bool_tensor(hf_uncertain),
        "lf_owned": _bool_tensor(lf_owned),
        "lf_safe": _bool_tensor(lf_safe),
        "trainable_all": torch.ones((num_gaussians,), dtype=torch.bool),
        "hf_score": torch.from_numpy(base_score.astype(np.float32, copy=False)),
        "touch_score": torch.from_numpy(touch_max.astype(np.float32, copy=False)),
        "hit_score": torch.from_numpy(hit_max.astype(np.float32, copy=False)),
        "validated_score": torch.from_numpy(validated_max.astype(np.float32, copy=False)),
        "consistency_score": torch.from_numpy(consistency_max.astype(np.float32, copy=False)),
        "touch_count": torch.from_numpy(touch_count.astype(np.int16, copy=False)),
        "meta": {
            "version": "cave_hf_reassignment_v0",
            "cave_cache_root": str(root),
            "num_gaussians": int(num_gaussians),
            "num_frames": int(len(frames)),
            "score_mode": str(args.score_mode),
            "threshold": float(threshold),
            "hf_keep_ratio": float(args.hf_keep_ratio),
            "min_hf_score": float(args.min_hf_score),
            "min_touch_views": int(args.min_touch_views),
            "min_validated_score": float(args.min_validated_score),
            "consistency_weight": float(args.consistency_weight),
            "view_support_weight": float(args.view_support_weight),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    summary = {
        "payload": str(output_path),
        "cave_cache_root": str(root),
        "num_gaussians": int(num_gaussians),
        "num_frames": int(len(frames)),
        "threshold": float(threshold),
        "hf_owned": int(hf_owned.sum()),
        "hf_candidate": int(hf_candidate.sum()),
        "hf_uncertain": int(hf_uncertain.sum()),
        "lf_owned": int(lf_owned.sum()),
        "lf_safe": int(lf_safe.sum()),
        "hf_owned_ratio": float(hf_owned.mean()),
        "hf_candidate_ratio": float(hf_candidate.mean()),
        "score_mode": str(args.score_mode),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
