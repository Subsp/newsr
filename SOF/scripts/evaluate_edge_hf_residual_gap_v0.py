#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the exact image-domain HF residual gap used by prior_edge_loss_mode=hf_residual_v1."
        )
    )
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--target_dir", required=True, help="Edge target directory, usually NPSE edge_target.")
    parser.add_argument("--anchor_dir", required=True, help="Anchor render directory used by hf_residual_v1.")
    parser.add_argument("--mask_dir", required=True, help="Image HF region / trust mask directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--detail_alpha", type=float, default=0.6)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--lowfreq_threshold", type=float, default=0.08)
    parser.add_argument("--confidence_power", type=float, default=1.5)
    parser.add_argument("--mask_power", type=float, default=1.0)
    parser.add_argument("--min_pixels", type=float, default=32.0)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(
    paths: Sequence[Path],
    lookup: Dict[str, Path],
    render_path: Path,
    index: int,
    match_policy: str,
) -> Optional[Path]:
    if match_policy in {"stem", "order_if_needed"}:
        found = lookup.get(render_path.stem.lower())
        if found is not None:
            return found
        if match_policy == "stem":
            return None
    if match_policy in {"order", "order_if_needed"} and index < len(paths):
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


def _box_blur_rgb(image: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    ).astype(np.float32) / float(k * k)


def _normalize(arr: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(np.abs(arr), float(percentile)))
    if scale <= 1e-8:
        return np.zeros(arr.shape[:2], dtype=np.float32)
    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=2)
    else:
        arr = np.abs(arr)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _weighted_mean_abs(value: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (value.shape[2] if value.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    if value.ndim == 3:
        return float((np.abs(value) * weight[..., None]).sum() / denom)
    return float((np.abs(value) * weight).sum() / denom)


def _weighted_l1(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    return _weighted_mean_abs(a - b, weight)


def _pearson(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
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


def _save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _panel(rgb: np.ndarray, label: str) -> Image.Image:
    rgb = np.clip(rgb, 0.0, 1.0)
    image = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(420, image.width), 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return image


def _gray_rgb(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0.0, 1.0)
    return np.repeat(gray[..., None], 3, axis=2)


def _overlay(render_abs: np.ndarray, target_abs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*render_abs.shape, 3), dtype=np.float32)
    rgb[..., 0] = target_abs
    rgb[..., 1] = render_abs
    rgb[..., 2] = render_abs
    return np.clip(rgb * (0.15 + 0.85 * np.clip(weight[..., None], 0.0, 1.0)), 0.0, 1.0)


def _write_sheet(
    path: Path,
    render: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    render_abs: np.ndarray,
    target_abs: np.ndarray,
    error_abs: np.ndarray,
) -> None:
    panels = [
        _panel(render, "render"),
        _panel(target, "edge target"),
        _panel(_gray_rgb(weight), "effective HF-region weight"),
        _panel(_gray_rgb(render_abs), "abs HP(render)-HP(anchor)"),
        _panel(_gray_rgb(target_abs), "abs HP(target)-HP(anchor)"),
        _panel(_gray_rgb(error_abs), "abs residual error"),
        _panel(_overlay(render_abs, target_abs, weight), "overlay target=red render=cyan"),
    ]
    width = max(p.width for p in panels)
    height = max(p.height for p in panels)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % cols) * width, (i // cols) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _mean(rows: Sequence[Dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = _parse_args()
    render_dir = Path(args.render_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    render_paths = _list_images(render_dir)
    if int(args.limit) > 0:
        render_paths = render_paths[: int(args.limit)]
    target_paths = _list_images(target_dir)
    anchor_paths = _list_images(anchor_dir)
    mask_paths = _list_images(mask_dir)
    target_lookup = _lookup(target_paths)
    anchor_lookup = _lookup(anchor_paths)
    mask_lookup = _lookup(mask_paths)

    print(f"[edge-hf-gap-v0] render : {render_dir}")
    print(f"[edge-hf-gap-v0] target : {target_dir}")
    print(f"[edge-hf-gap-v0] anchor : {anchor_dir}")
    print(f"[edge-hf-gap-v0] mask   : {mask_dir}")
    print(f"[edge-hf-gap-v0] output : {output_dir}")
    print(
        "[edge-hf-gap-v0] loss   : "
        "weighted L1 between HP(render)-HP(anchor) and alpha*(HP(target)-HP(anchor))"
    )

    rows: List[Dict[str, float]] = []
    frames = []
    skipped = 0
    for idx, render_path in enumerate(tqdm(render_paths, desc="edge HF gap")):
        target_path = _resolve(target_paths, target_lookup, render_path, idx, str(args.match_policy))
        anchor_path = _resolve(anchor_paths, anchor_lookup, render_path, idx, str(args.match_policy))
        mask_path = _resolve(mask_paths, mask_lookup, render_path, idx, str(args.match_policy))
        if target_path is None or anchor_path is None or mask_path is None:
            skipped += 1
            continue

        render = _load_rgb(render_path)
        size = (render.shape[1], render.shape[0])
        target = _load_rgb(target_path, size=size)
        anchor = _load_rgb(anchor_path, size=size)
        mask = _load_gray(mask_path, size=size)
        mask = np.clip(mask, 0.0, 1.0)
        if float(args.mask_power) != 1.0:
            mask = np.power(mask, max(float(args.mask_power), 0.0)).astype(np.float32)

        render_low = _box_blur_rgb(render, int(args.highpass_kernel))
        target_low = _box_blur_rgb(target, int(args.highpass_kernel))
        anchor_low = _box_blur_rgb(anchor, int(args.highpass_kernel))
        render_residual = (render - render_low) - (anchor - anchor_low)
        target_residual = (target - target_low) - (anchor - anchor_low)
        if float(args.residual_clip) > 0.0:
            target_residual = np.clip(target_residual, -float(args.residual_clip), float(args.residual_clip))

        lowfreq_diff = np.abs(target_low - anchor_low).mean(axis=2)
        if float(args.lowfreq_threshold) > 0.0:
            confidence = np.clip(1.0 - lowfreq_diff / float(args.lowfreq_threshold), 0.0, 1.0)
        else:
            confidence = np.ones_like(mask, dtype=np.float32)
        if float(args.confidence_power) != 1.0:
            confidence = np.power(np.clip(confidence, 0.0, 1.0), max(float(args.confidence_power), 1e-6))
        weight = (mask * confidence).astype(np.float32)
        active_weight = float(weight.sum())
        if active_weight < float(args.min_pixels):
            skipped += 1
            continue

        target_scaled = float(args.detail_alpha) * target_residual
        loss = _weighted_l1(render_residual, target_scaled, weight)
        target_energy = _weighted_mean_abs(target_scaled, weight)
        render_energy = _weighted_mean_abs(render_residual, weight)
        error_abs = np.abs(render_residual - target_scaled).mean(axis=2)
        render_abs = _normalize(render_residual, float(args.norm_percentile))
        target_abs = _normalize(target_scaled, float(args.norm_percentile))
        error_norm = _normalize(render_residual - target_scaled, float(args.norm_percentile))

        row = {
            "index": float(idx),
            "loss_hf_residual_l1": float(loss),
            "render_residual_abs": float(render_energy),
            "target_residual_abs": float(target_energy),
            "energy_ratio_render_over_target": float(render_energy / max(target_energy, 1e-8)),
            "weighted_corr_abs": _pearson(render_residual, target_scaled, weight),
            "mask_mean": float(mask.mean()),
            "confidence_mean": float(confidence.mean()),
            "effective_weight_mean": float(weight.mean()),
            "effective_weight_sum": float(active_weight),
            "effective_hard_ratio": float((weight > 1e-4).mean()),
            "unweighted_error_abs": float(np.mean(error_abs)),
        }
        rows.append(row)
        frames.append(
            {
                "stem": render_path.stem,
                "render": str(render_path),
                "target": str(target_path),
                "anchor": str(anchor_path),
                "mask": str(mask_path),
                **row,
            }
        )

        _save_gray(output_dir / "effective_weight" / render_path.name, weight)
        _save_gray(output_dir / "render_residual_hf_abs" / render_path.name, render_abs)
        _save_gray(output_dir / "target_residual_hf_abs" / render_path.name, target_abs)
        _save_gray(output_dir / "residual_error_abs" / render_path.name, error_norm)
        if len(rows) <= int(args.debug_limit):
            _write_sheet(
                output_dir / "sheet" / render_path.name,
                render=render,
                target=target,
                weight=weight,
                render_abs=render_abs,
                target_abs=target_abs,
                error_abs=error_norm,
            )

    if not rows:
        raise RuntimeError("No matched/effective frames were evaluated.")

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "version": "evaluate_edge_hf_residual_gap_v0",
        "render_dir": str(render_dir),
        "target_dir": str(target_dir),
        "anchor_dir": str(anchor_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
        "match_policy": str(args.match_policy),
        "highpass_kernel": int(args.highpass_kernel),
        "detail_alpha": float(args.detail_alpha),
        "residual_clip": float(args.residual_clip),
        "lowfreq_threshold": float(args.lowfreq_threshold),
        "confidence_power": float(args.confidence_power),
        "mask_power": float(args.mask_power),
        "min_pixels": float(args.min_pixels),
        "num_frames": int(len(rows)),
        "num_skipped": int(skipped),
        "mean": {key: _mean(rows, key) for key in rows[0].keys() if key != "index"},
        "per_frame_csv": str(csv_path),
        "frames": frames,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mean = summary["mean"]
    print(
        "[edge-hf-gap-v0] summary "
        f"frames={summary['num_frames']} skipped={summary['num_skipped']} "
        f"loss={mean['loss_hf_residual_l1']:.6f} "
        f"render_energy={mean['render_residual_abs']:.6f} "
        f"target_energy={mean['target_residual_abs']:.6f} "
        f"ratio={mean['energy_ratio_render_over_target']:.3f} "
        f"corr={mean['weighted_corr_abs']:.4f} "
        f"weight_mean={mean['effective_weight_mean']:.4f}"
    )
    print(f"[edge-hf-gap-v0] summary: {summary_path}")
    print(f"[edge-hf-gap-v0] inspect: {output_dir / 'sheet'}")


if __name__ == "__main__":
    main()
