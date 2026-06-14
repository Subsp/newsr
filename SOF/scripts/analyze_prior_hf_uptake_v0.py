#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RESAMPLING_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a current render absorbed high-frequency residuals "
            "from a prepared SR prior, relative to a baseline render."
        )
    )
    parser.add_argument("--baseline_render_dir", type=Path, required=True)
    parser.add_argument("--current_render_dir", type=Path, required=True)
    parser.add_argument("--prepared_prior_root", type=Path, required=True)
    parser.add_argument("--prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--anchor_subdir", type=str, default="aligned_references")
    parser.add_argument("--mask_subdir", type=str, default="usable_masks")
    parser.add_argument(
        "--render_name_mode",
        type=str,
        default="index_manifest",
        choices=["index_manifest", "filename"],
        help=(
            "index_manifest maps 00000.png renders to manifest frame order. "
            "filename expects render names to match prepared prior names."
        ),
    )
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--top_fraction", type=float, default=0.15)
    parser.add_argument("--min_prior_hf", type=float, default=0.0)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--out_json", type=Path, default=None)
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask01(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        return np.ones(shape_hw, dtype=np.float32)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape != shape_hw:
        mask = np.asarray(
            Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
                (shape_hw[1], shape_hw[0]),
                RESAMPLING_BICUBIC,
            ),
            dtype=np.float32,
        ) / 255.0
    return np.clip(mask, 0.0, 1.0)


def resize_like(image: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if image.shape[:2] == ref.shape[:2]:
        return image
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
    resized = pil.resize((ref.shape[1], ref.shape[0]), RESAMPLING_BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def laplacian_rgb(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    center = padded[1:-1, 1:-1, :]
    up = padded[:-2, 1:-1, :]
    down = padded[2:, 1:-1, :]
    left = padded[1:-1, :-2, :]
    right = padded[1:-1, 2:, :]
    return 4.0 * center - up - down - left - right


def weighted_mean(values: np.ndarray, weight_hw: np.ndarray, eps: float = 1e-12) -> float:
    weights = weight_hw[..., None].astype(np.float32)
    denom = float(weights.sum() * values.shape[2])
    if denom <= eps:
        return float("nan")
    return float((values * weights).sum() / denom)


def weighted_abs_mean(values: np.ndarray, weight_hw: np.ndarray, eps: float = 1e-12) -> float:
    return weighted_mean(np.abs(values), weight_hw, eps=eps)


def weighted_dot(a: np.ndarray, b: np.ndarray, weight_hw: np.ndarray) -> float:
    weights = weight_hw[..., None].astype(np.float32)
    return float((a * b * weights).sum())


def build_top_mask(prior_hf: np.ndarray, base_mask: np.ndarray, top_fraction: float, min_prior_hf: float) -> np.ndarray:
    base = base_mask > 0.0
    if not np.any(base):
        return np.zeros_like(base_mask, dtype=np.float32)
    energy = np.abs(prior_hf).mean(axis=2)
    valid_values = energy[base]
    if valid_values.size == 0:
        return np.zeros_like(base_mask, dtype=np.float32)
    top_fraction = float(np.clip(top_fraction, 0.0, 1.0))
    if top_fraction <= 0.0:
        thresh = float(min_prior_hf)
    elif top_fraction >= 1.0:
        thresh = float(min_prior_hf)
    else:
        thresh = float(np.quantile(valid_values, 1.0 - top_fraction))
        thresh = max(thresh, float(min_prior_hf))
    return (base & (energy >= thresh)).astype(np.float32)


def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= eps:
        return float("nan")
    return float(num / den)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_manifest(prepared_root: Path) -> list[dict[str, Any]]:
    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prepared prior manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Prepared prior manifest has no frames: {manifest_path}")
    return frames


def resolve_prior_name(render_path: Path, index: int, frames: list[dict[str, Any]], mode: str) -> str:
    if mode == "filename":
        return render_path.name
    if index >= len(frames):
        raise IndexError(f"Render index {index} exceeds manifest frame count {len(frames)}")
    return str(frames[index]["image_name"])


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    baseline_render_dir = args.baseline_render_dir.expanduser().resolve()
    current_render_dir = args.current_render_dir.expanduser().resolve()
    prepared_root = args.prepared_prior_root.expanduser().resolve()
    prior_dir = prepared_root / args.prior_subdir
    anchor_dir = prepared_root / args.anchor_subdir
    mask_dir = prepared_root / args.mask_subdir
    frames = load_manifest(prepared_root)

    baseline_files = iter_images(baseline_render_dir)
    current_lookup = {p.name: p for p in iter_images(current_render_dir)}
    if args.max_views > 0:
        baseline_files = baseline_files[: int(args.max_views)]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for idx, baseline_path in enumerate(baseline_files):
        current_path = current_lookup.get(baseline_path.name)
        if current_path is None:
            skipped.append({"render": baseline_path.name, "reason": "missing current render"})
            continue
        try:
            prior_name = resolve_prior_name(baseline_path, idx, frames, args.render_name_mode)
        except Exception as exc:
            skipped.append({"render": baseline_path.name, "reason": repr(exc)})
            continue

        prior_path = prior_dir / prior_name
        anchor_path = anchor_dir / prior_name
        if not prior_path.is_file() or not anchor_path.is_file():
            skipped.append({"render": baseline_path.name, "prior_name": prior_name, "reason": "missing prior/anchor"})
            continue

        baseline = load_rgb01(baseline_path)
        current = resize_like(load_rgb01(current_path), baseline)
        prior = resize_like(load_rgb01(prior_path), baseline)
        anchor = resize_like(load_rgb01(anchor_path), baseline)
        mask = (
            load_mask01(mask_dir / prior_name, baseline.shape[:2])
            if bool(args.use_mask)
            else np.ones(baseline.shape[:2], dtype=np.float32)
        )

        prior_hf = laplacian_rgb(prior - anchor)
        baseline_hf = laplacian_rgb(baseline - anchor)
        current_hf = laplacian_rgb(current - anchor)
        change_hf = current_hf - baseline_hf
        top_mask = build_top_mask(
            prior_hf=prior_hf,
            base_mask=mask,
            top_fraction=float(args.top_fraction),
            min_prior_hf=float(args.min_prior_hf),
        )

        if float(top_mask.sum()) <= 0.0:
            skipped.append({"render": baseline_path.name, "prior_name": prior_name, "reason": "empty top prior HF mask"})
            continue

        prior_energy = weighted_abs_mean(prior_hf, top_mask)
        change_energy = weighted_abs_mean(change_hf, top_mask)
        baseline_energy = weighted_abs_mean(baseline_hf, top_mask)
        current_energy = weighted_abs_mean(current_hf, top_mask)
        baseline_gap = weighted_abs_mean(baseline_hf - prior_hf, top_mask)
        current_gap = weighted_abs_mean(current_hf - prior_hf, top_mask)

        dot_change_prior = weighted_dot(change_hf, prior_hf, top_mask)
        dot_prior_prior = weighted_dot(prior_hf, prior_hf, top_mask)
        dot_change_change = weighted_dot(change_hf, change_hf, top_mask)
        cosine = safe_ratio(dot_change_prior, math.sqrt(max(dot_prior_prior * dot_change_change, 0.0)))
        projection = safe_ratio(dot_change_prior, dot_prior_prior)

        dot_current_prior = weighted_dot(current_hf, prior_hf, top_mask)
        dot_baseline_prior = weighted_dot(baseline_hf, prior_hf, top_mask)
        current_projection = safe_ratio(dot_current_prior, dot_prior_prior)
        baseline_projection = safe_ratio(dot_baseline_prior, dot_prior_prior)

        outside_mask = (mask > 0.0).astype(np.float32) * (1.0 - top_mask)
        outside_change_energy = weighted_abs_mean(change_hf, outside_mask) if float(outside_mask.sum()) > 0.0 else float("nan")
        false_hf_ratio = safe_ratio(outside_change_energy, change_energy)

        rows.append(
            {
                "render_name": baseline_path.name,
                "prior_name": prior_name,
                "top_pixels": int(top_mask.sum()),
                "top_fraction_actual": float(top_mask.sum() / max(float((mask > 0.0).sum()), 1.0)),
                "prior_hf_abs": prior_energy,
                "change_hf_abs": change_energy,
                "baseline_hf_abs": baseline_energy,
                "current_hf_abs": current_energy,
                "hf_energy_gain_ratio": safe_ratio(current_energy, baseline_energy),
                "change_to_prior_abs_ratio": safe_ratio(change_energy, prior_energy),
                "change_prior_cosine": cosine,
                "change_prior_projection": projection,
                "baseline_prior_projection": baseline_projection,
                "current_prior_projection": current_projection,
                "projection_gain": current_projection - baseline_projection
                if math.isfinite(current_projection) and math.isfinite(baseline_projection)
                else float("nan"),
                "baseline_gap_abs": baseline_gap,
                "current_gap_abs": current_gap,
                "gap_improvement_ratio": safe_ratio(baseline_gap - current_gap, baseline_gap),
                "outside_to_inside_change_ratio": false_hf_ratio,
            }
        )

    metric_keys = [
        "prior_hf_abs",
        "change_hf_abs",
        "baseline_hf_abs",
        "current_hf_abs",
        "hf_energy_gain_ratio",
        "change_to_prior_abs_ratio",
        "change_prior_cosine",
        "change_prior_projection",
        "baseline_prior_projection",
        "current_prior_projection",
        "projection_gain",
        "baseline_gap_abs",
        "current_gap_abs",
        "gap_improvement_ratio",
        "outside_to_inside_change_ratio",
    ]
    global_summary = {key: summarize([float(row[key]) for row in rows]) for key in metric_keys}
    ranked_by_projection = sorted(rows, key=lambda row: float(row["change_prior_projection"]), reverse=True)
    ranked_by_gap = sorted(rows, key=lambda row: float(row["gap_improvement_ratio"]), reverse=True)
    ranked_by_change = sorted(rows, key=lambda row: float(row["change_hf_abs"]), reverse=True)

    return {
        "baseline_render_dir": str(baseline_render_dir),
        "current_render_dir": str(current_render_dir),
        "prepared_prior_root": str(prepared_root),
        "prior_subdir": str(args.prior_subdir),
        "anchor_subdir": str(args.anchor_subdir),
        "mask_subdir": str(args.mask_subdir) if bool(args.use_mask) else None,
        "render_name_mode": str(args.render_name_mode),
        "top_fraction": float(args.top_fraction),
        "min_prior_hf": float(args.min_prior_hf),
        "n_views": int(len(rows)),
        "n_skipped": int(len(skipped)),
        "skipped_examples": skipped[:16],
        "global": global_summary,
        "top_views_by_prior_projection": ranked_by_projection[:12],
        "top_views_by_gap_improvement": ranked_by_gap[:12],
        "top_views_by_change_energy": ranked_by_change[:12],
        "per_view": rows,
        "interpretation": {
            "change_prior_projection": "0 means no prior-directed HF uptake; 1 means current-baseline matches the prior HF residual amplitude on selected pixels.",
            "change_prior_cosine": "Direction agreement between render HF change and prior HF residual. Positive is aligned; negative is opposite.",
            "gap_improvement_ratio": "Positive means current render is closer than baseline to the prior HF residual; negative means it moved away.",
            "outside_to_inside_change_ratio": "Large values mean much of the new HF energy appears outside the strongest prior-HF pixels.",
        },
    }


def main() -> None:
    args = parse_args()
    summary = analyze(args)
    out_json = args.out_json
    if out_json is None:
        out_json = args.current_render_dir.expanduser().resolve().parent / "prior_hf_uptake_v0.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    concise = {
        "out_json": str(out_json),
        "n_views": summary["n_views"],
        "n_skipped": summary["n_skipped"],
        "change_prior_projection_mean": summary["global"]["change_prior_projection"]["mean"],
        "change_prior_cosine_mean": summary["global"]["change_prior_cosine"]["mean"],
        "gap_improvement_ratio_mean": summary["global"]["gap_improvement_ratio"]["mean"],
        "change_to_prior_abs_ratio_mean": summary["global"]["change_to_prior_abs_ratio"]["mean"],
        "outside_to_inside_change_ratio_mean": summary["global"]["outside_to_inside_change_ratio"]["mean"],
        "top_views_by_prior_projection": [
            row["prior_name"] for row in summary["top_views_by_prior_projection"][:5]
        ],
    }
    print(json.dumps(concise, indent=2))
    print(f"saved to: {out_json}")


if __name__ == "__main__":
    main()
