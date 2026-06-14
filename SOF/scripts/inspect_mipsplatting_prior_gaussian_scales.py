#!/usr/bin/env python3
"""Summarize PRIOR_INJECTED Gaussian scale/lineage statistics from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


PRIOR_INJECTED = 1


def _latest_checkpoint(model_dir: Path) -> Path:
    candidates = []
    for path in model_dir.glob("chkpnt*.pth"):
        stem = path.stem.replace("chkpnt", "")
        try:
            iteration = int(stem)
        except ValueError:
            continue
        candidates.append((iteration, path))
    if not candidates:
        raise FileNotFoundError(f"no chkpnt*.pth found under {model_dir}")
    return sorted(candidates)[-1][1]


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def _generation_histogram(generation: torch.Tensor) -> dict[str, int]:
    if generation.numel() == 0:
        return {}
    uniq, counts = torch.unique(generation.to(torch.int64), return_counts=True)
    return {str(int(k.item())): int(v.item()) for k, v in zip(uniq, counts)}


def summarize_checkpoint(path: Path, large_scale_threshold: float) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"expected (model_args, iteration) checkpoint, got {type(payload)}")
    model_args, iteration = payload
    if not isinstance(model_args, (tuple, list)) or len(model_args) < 13:
        raise ValueError(f"unsupported model_args payload length: {len(model_args)}")

    raw_scaling = model_args[4]
    scales = torch.exp(raw_scaling.detach().float())
    tracking = model_args[-1] if isinstance(model_args[-1], dict) else None
    if tracking is None:
        raise ValueError("checkpoint does not contain Gaussian tracking_state")

    source_tag = tracking["source_tag"].to(torch.int64)
    seed_id = tracking["seed_id"].to(torch.int64)
    generation = tracking["generation"].to(torch.int64)

    prior_mask = source_tag == PRIOR_INJECTED
    prior_scales = scales[prior_mask]
    prior_seed_id = seed_id[prior_mask]
    prior_generation = generation[prior_mask]
    assigned = prior_seed_id >= 0
    unique_seed_ids = int(torch.unique(prior_seed_id[assigned]).numel()) if torch.any(assigned) else 0
    prior_count = int(prior_mask.sum().item())
    max_axis = prior_scales.max(dim=1).values if prior_count else torch.empty(0)
    min_axis = prior_scales.min(dim=1).values if prior_count else torch.empty(0)
    geom = prior_scales.clamp(min=1e-12).prod(dim=1).pow(1.0 / 3.0) if prior_count else torch.empty(0)
    anisotropy = max_axis / min_axis.clamp(min=1e-12) if prior_count else torch.empty(0)

    large_mask = max_axis > float(large_scale_threshold) if prior_count else torch.empty(0, dtype=torch.bool)
    return {
        "checkpoint": str(path),
        "iteration": int(iteration),
        "total_gaussians": int(scales.shape[0]),
        "prior_injected": {
            "count": prior_count,
            "unique_seed_ids": unique_seed_ids,
            "clone_ratio": float(prior_count / max(unique_seed_ids, 1)),
            "large_scale_threshold": float(large_scale_threshold),
            "large_count": int(large_mask.sum().item()) if prior_count else 0,
            "large_ratio": float(large_mask.float().mean().item()) if prior_count else 0.0,
            "scale_geom": _stats(geom),
            "scale_max_axis": _stats(max_axis),
            "scale_min_axis": _stats(min_axis),
            "anisotropy": _stats(anisotropy),
            "generation": {
                "mean": float(prior_generation.float().mean().item()) if prior_count else 0.0,
                "max": int(prior_generation.max().item()) if prior_count else 0,
                "histogram": _generation_histogram(prior_generation),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--large_scale_threshold", type=float, default=0.020)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint is not None else _latest_checkpoint(args.model_dir)
    summary = summarize_checkpoint(checkpoint, args.large_scale_threshold)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
