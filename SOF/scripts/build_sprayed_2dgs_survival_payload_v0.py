#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a first-pass survival payload for sprayed 2DGS HF Gaussians. "
            "The score is intentionally metadata-only: source evidence, Gaussian-layer match quality, "
            "and local multi-view/group support. It does not edit the source model."
        )
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--summary_path", default="")
    parser.add_argument("--tags_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--distance_sigma_px", type=float, default=1.75)
    parser.add_argument("--bad_distance_px", type=float, default=4.0)
    parser.add_argument("--group_size_ref", type=float, default=16.0)
    parser.add_argument("--min_group_views_survive", type=int, default=2)
    parser.add_argument("--min_group_members_survive", type=int, default=4)
    parser.add_argument("--source_min_survive", type=float, default=0.11)
    parser.add_argument("--source_min_probation", type=float, default=0.055)
    parser.add_argument("--survive_min_score", type=float, default=0.46)
    parser.add_argument("--probation_min_score", type=float, default=0.26)
    parser.add_argument("--suppress_min_score", type=float, default=0.11)
    parser.add_argument("--risk_opacity", type=float, default=0.12)
    parser.add_argument("--risk_scale_long", type=float, default=0.010)
    parser.add_argument("--probation_opacity_multiplier", type=float, default=0.35)
    parser.add_argument("--suppress_opacity_multiplier", type=float, default=0.08)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    model_dir = Path(args.model_dir).expanduser().resolve()
    point_dir = model_dir / "point_cloud" / f"iteration_{int(args.iteration)}"
    tags_path = Path(args.tags_path).expanduser().resolve() if args.tags_path else point_dir / "gaussian_tags.pt"
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else model_dir / "spray_2dgs_hf_carrier_to_gaussian_layer_v0_summary.json"
    )
    if args.metadata_path:
        metadata_path = Path(args.metadata_path).expanduser().resolve()
    else:
        summary = _read_json(summary_path)
        metadata_from_summary = summary.get("newborn_metadata")
        if not isinstance(metadata_from_summary, str) or not metadata_from_summary:
            raise KeyError(f"summary does not contain newborn_metadata: {summary_path}")
        metadata_path = Path(metadata_from_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    return tags_path, summary_path, metadata_path, output_dir


def _load_tags(tags_path: Path) -> Dict[str, torch.Tensor]:
    if not tags_path.is_file():
        raise FileNotFoundError(f"tracking metadata not found: {tags_path}")
    payload = torch.load(tags_path, map_location="cpu")
    if not isinstance(payload, dict) or "source_tag" not in payload:
        raise ValueError(f"unsupported tracking metadata: {tags_path}")
    return payload


def _as_float(meta: np.lib.npyio.NpzFile, key: str, default: float = 0.0) -> np.ndarray:
    if key not in meta:
        raise KeyError(f"metadata missing {key}")
    return np.asarray(meta[key], dtype=np.float32).reshape(-1)


def _group_support(matched_base_index: np.ndarray, source_view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, inverse, counts = np.unique(matched_base_index.astype(np.int64), return_inverse=True, return_counts=True)
    pairs = np.stack([inverse.astype(np.int64), source_view.astype(np.int64)], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    view_counts = np.bincount(unique_pairs[:, 0], minlength=counts.shape[0]).astype(np.int32)
    return counts[inverse].astype(np.int32), view_counts[inverse].astype(np.int32)


def _normalize_group_size(group_size: np.ndarray, group_size_ref: float) -> np.ndarray:
    ref = max(float(group_size_ref), 1.0)
    return np.clip(np.log1p(group_size.astype(np.float32)) / np.log1p(ref), 0.0, 1.0)


def _build_full_mask(prior_indices: torch.Tensor, newborn_mask: np.ndarray, total: int) -> torch.Tensor:
    if newborn_mask.shape[0] != int(prior_indices.shape[0]):
        raise ValueError(
            f"newborn mask length mismatch: mask={newborn_mask.shape[0]} prior={int(prior_indices.shape[0])}"
        )
    out = torch.zeros((total,), dtype=torch.bool)
    out[prior_indices] = torch.from_numpy(newborn_mask.astype(np.bool_))
    return out


def _torch_flatnonzero(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)


def main() -> None:
    args = _parse_args()
    tags_path, summary_path, metadata_path, output_dir = _resolve_paths(args)
    summary = _read_json(summary_path)
    tags = _load_tags(tags_path)
    source_tag = torch.as_tensor(tags["source_tag"]).reshape(-1).to(dtype=torch.int64)
    total = int(source_tag.shape[0])
    prior_mask_t = source_tag == 1
    prior_indices = _torch_flatnonzero(prior_mask_t)
    base_count = int(summary.get("base_gaussians", total - int(prior_indices.shape[0])))

    if not metadata_path.is_file():
        raise FileNotFoundError(f"spray metadata not found: {metadata_path}")
    meta = np.load(metadata_path)
    weight = _as_float(meta, "weight")
    primitive_opacity = _as_float(meta, "primitive_opacity")
    matched_distance = _as_float(meta, "matched_pixel_distance")
    opacity = _as_float(meta, "opacity")
    source_view = np.asarray(meta["source_view"], dtype=np.int32).reshape(-1)
    matched_base_index = np.asarray(meta["matched_base_index"], dtype=np.int64).reshape(-1)
    scale = np.asarray(meta["scale"], dtype=np.float32)
    if scale.ndim != 2 or scale.shape[0] != weight.shape[0]:
        raise ValueError(f"scale shape mismatch: {scale.shape} vs {weight.shape[0]}")
    n = int(weight.shape[0])
    if int(prior_indices.shape[0]) != n:
        appended_ok = base_count + n == total
        if not appended_ok:
            raise ValueError(
                f"prior count does not match metadata: prior={int(prior_indices.shape[0])} metadata={n} "
                f"base_count={base_count} total={total}"
            )
        prior_indices = torch.arange(base_count, total, dtype=torch.long)

    group_size, group_views = _group_support(matched_base_index, source_view)
    evidence = np.sqrt(np.clip(weight, 0.0, 1.0)) * (0.35 + 0.65 * np.clip(primitive_opacity, 0.0, 1.0))
    dist_score = np.exp(-0.5 * (matched_distance / max(float(args.distance_sigma_px), 1e-6)) ** 2).astype(np.float32)
    source_score = evidence * dist_score
    view_support = np.clip(group_views.astype(np.float32) / max(int(args.min_group_views_survive), 1), 0.0, 1.0)
    size_support = _normalize_group_size(group_size, float(args.group_size_ref))
    scale_long = np.max(scale, axis=1)
    opacity_risk = np.clip((opacity - float(args.risk_opacity)) / max(1.0 - float(args.risk_opacity), 1e-6), 0.0, 1.0)
    scale_risk = np.clip((scale_long - float(args.risk_scale_long)) / max(float(args.risk_scale_long), 1e-6), 0.0, 1.0)
    distance_risk = (matched_distance > float(args.bad_distance_px)).astype(np.float32)
    risk = np.clip(0.45 * opacity_risk + 0.35 * scale_risk + 0.20 * distance_risk, 0.0, 1.0)

    score = (
        0.55 * source_score
        + 0.22 * view_support
        + 0.18 * size_support
        + 0.05 * dist_score
        - 0.22 * risk
    ).astype(np.float32)
    score = np.clip(score, 0.0, 1.0)

    group_ok = (group_views >= int(args.min_group_views_survive)) | (
        group_size >= int(args.min_group_members_survive)
    )
    hard_bad = (matched_distance > float(args.bad_distance_px)) & (source_score < float(args.source_min_probation))
    survive = (
        (score >= float(args.survive_min_score))
        & (source_score >= float(args.source_min_survive))
        & group_ok
        & ~hard_bad
    )
    probation = (
        ~survive
        & (score >= float(args.probation_min_score))
        & (source_score >= float(args.source_min_probation))
        & ~hard_bad
    )
    suppress = ~survive & ~probation & (score >= float(args.suppress_min_score)) & ~hard_bad
    prune = ~(survive | probation | suppress)

    state = np.zeros((n,), dtype=np.int8)
    state[survive] = 1
    state[probation] = 2
    state[suppress] = 3
    state[prune] = 4
    opacity_multiplier = np.zeros((n,), dtype=np.float32)
    opacity_multiplier[survive] = 1.0
    opacity_multiplier[probation] = float(args.probation_opacity_multiplier)
    opacity_multiplier[suppress] = float(args.suppress_opacity_multiplier)

    keep = survive | probation
    drop = suppress | prune
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "sprayed_2dgs_survival_payload_v0.pt"
    payload = {
        "prior": prior_mask_t,
        "survive_prior": _build_full_mask(prior_indices, survive, total),
        "probation_prior": _build_full_mask(prior_indices, probation, total),
        "suppress_prior": _build_full_mask(prior_indices, suppress, total),
        "prune_prior": _build_full_mask(prior_indices, prune, total),
        "keep_prior": _build_full_mask(prior_indices, keep, total),
        "drop_prior": _build_full_mask(prior_indices, drop, total),
        "state": torch.zeros((total,), dtype=torch.int8),
        "survival_score": torch.zeros((total,), dtype=torch.float32),
        "opacity_multiplier": torch.ones((total,), dtype=torch.float32),
        "base_count": torch.tensor(base_count, dtype=torch.int64),
    }
    payload["state"][prior_indices] = torch.from_numpy(state)
    payload["survival_score"][prior_indices] = torch.from_numpy(score)
    payload["opacity_multiplier"][prior_indices] = torch.from_numpy(opacity_multiplier)
    torch.save(payload, payload_path)

    npz_path = output_dir / "sprayed_2dgs_survival_scores_v0.npz"
    np.savez_compressed(
        npz_path,
        score=score,
        state=state,
        source_score=source_score.astype(np.float32),
        evidence=evidence.astype(np.float32),
        dist_score=dist_score.astype(np.float32),
        view_support=view_support.astype(np.float32),
        size_support=size_support.astype(np.float32),
        risk=risk.astype(np.float32),
        opacity_multiplier=opacity_multiplier,
        group_size=group_size,
        group_views=group_views,
        source_view=source_view,
        matched_base_index=matched_base_index,
        matched_pixel_distance=matched_distance,
        weight=weight,
        primitive_opacity=primitive_opacity,
        opacity=opacity,
        scale=scale,
    )
    counts = {
        "total": n,
        "survive": int(np.count_nonzero(survive)),
        "probation": int(np.count_nonzero(probation)),
        "suppress": int(np.count_nonzero(suppress)),
        "prune": int(np.count_nonzero(prune)),
        "keep": int(np.count_nonzero(keep)),
        "drop": int(np.count_nonzero(drop)),
    }
    score_stats = {
        "score_mean": float(score.mean()),
        "score_p50": float(np.percentile(score, 50)),
        "score_p90": float(np.percentile(score, 90)),
        "source_score_mean": float(source_score.mean()),
        "view_support_mean": float(view_support.mean()),
        "size_support_mean": float(size_support.mean()),
        "risk_mean": float(risk.mean()),
        "group_views_mean": float(group_views.mean()),
        "group_size_mean": float(group_size.mean()),
    }
    out_summary = {
        "version": "build_sprayed_2dgs_survival_payload_v0",
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "iteration": int(args.iteration),
        "tags_path": str(tags_path),
        "spray_summary": str(summary_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "payload_path": str(payload_path),
        "scores_path": str(npz_path),
        "base_count": base_count,
        "total_gaussians": total,
        "counts": counts,
        "ratios": {key: float(value / max(n, 1)) for key, value in counts.items() if key != "total"},
        "score_stats": score_stats,
        "thresholds": {
            "distance_sigma_px": float(args.distance_sigma_px),
            "bad_distance_px": float(args.bad_distance_px),
            "group_size_ref": float(args.group_size_ref),
            "min_group_views_survive": int(args.min_group_views_survive),
            "min_group_members_survive": int(args.min_group_members_survive),
            "source_min_survive": float(args.source_min_survive),
            "source_min_probation": float(args.source_min_probation),
            "survive_min_score": float(args.survive_min_score),
            "probation_min_score": float(args.probation_min_score),
            "suppress_min_score": float(args.suppress_min_score),
            "risk_opacity": float(args.risk_opacity),
            "risk_scale_long": float(args.risk_scale_long),
            "probation_opacity_multiplier": float(args.probation_opacity_multiplier),
            "suppress_opacity_multiplier": float(args.suppress_opacity_multiplier),
        },
    }
    summary_out_path = output_dir / "summary.json"
    summary_out_path.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "score_stats": score_stats, "payload": str(payload_path)}, indent=2))
    print(f"[survival-payload-v0] summary: {summary_out_path}")


if __name__ == "__main__":
    main()
