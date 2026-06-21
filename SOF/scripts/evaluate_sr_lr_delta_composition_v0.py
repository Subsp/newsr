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
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate what an SR/VOSR prior adds over an LR/GS render: low-frequency drift, "
            "luma/chroma residuals, high-frequency energy, edge gain, and flat-region noise."
        )
    )
    parser.add_argument("--sr_dir", required=True)
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_dir", default="", help="Optional trust/edge mask directory.")
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--lowpass_kernel", type=int, default=31)
    parser.add_argument("--mask_power", type=float, default=1.5)
    parser.add_argument("--edge_percentile", type=float, default=90.0)
    parser.add_argument("--flat_percentile", type=float, default=45.0)
    parser.add_argument("--debug_limit", type=int, default=12)
    parser.add_argument("--vis_clip", type=float, default=0.10)
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


def _box_blur_gray(image: np.ndarray, kernel: int) -> np.ndarray:
    return _box_blur_rgb(image[..., None], kernel)[..., 0]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb.astype(np.float32) * LUMA[None, None, :], axis=2)


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32, copy=False)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gy[0, :] = gray[1, :] - gray[0]
    gy[-1, :] = gray[-1, :] - gray[-2]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    denom = float(weight.sum())
    if denom <= 1e-8:
        return float("nan")
    return float((value * weight).sum() / denom)


def _weighted_p(value: np.ndarray, weight: np.ndarray, percentile: float) -> float:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    keep = (weight > 1e-8) & np.isfinite(value)
    if int(keep.sum()) == 0:
        return float("nan")
    value = value[keep]
    weight = weight[keep]
    order = np.argsort(value, kind="stable")
    value = value[order]
    weight = weight[order]
    cdf = np.cumsum(weight)
    threshold = float(percentile) / 100.0 * float(cdf[-1])
    return float(value[min(int(np.searchsorted(cdf, threshold)), value.size - 1)])


def _corr(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    keep = (w > 1e-8) & np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 4:
        return float("nan")
    a = a[keep]
    b = b[keep]
    w = w[keep]
    w = w / max(float(w.sum()), 1e-8)
    ma = float((a * w).sum())
    mb = float((b * w).sum())
    da = a - ma
    db = b - mb
    denom = float(np.sqrt((w * da * da).sum() * (w * db * db).sum()))
    if denom <= 1e-8:
        return float("nan")
    return float((w * da * db).sum() / denom)


def _rgb_delta_chroma(delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.sum(delta * LUMA[None, None, :], axis=2)
    u = delta[..., 2] - y
    v = delta[..., 0] - y
    chroma = np.sqrt(0.5 * (u * u + v * v)).astype(np.float32)
    return y.astype(np.float32), chroma


def _abs_rgb_mean(delta: np.ndarray) -> np.ndarray:
    return np.abs(delta).mean(axis=2).astype(np.float32)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _signed_vis(delta: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(0.5 + delta / max(2.0 * float(clip), 1e-8), 0.0, 1.0)


def _gray_vis(value: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(value / max(float(clip), 1e-8), 0.0, 1.0)


def _write_sheet(path: Path, sr: np.ndarray, lr: np.ndarray, delta: np.ndarray, hf: np.ndarray, chroma: np.ndarray, edge_gain: np.ndarray) -> None:
    clip = 0.10
    panels = [
        ("LR", lr),
        ("SR", sr),
        ("signed SR-LR", _signed_vis(delta, clip)),
        ("HF |SR-LR|", np.repeat(_gray_vis(_abs_rgb_mean(hf), clip)[..., None], 3, axis=2)),
        ("chroma delta", np.repeat(_gray_vis(chroma, clip * 0.5)[..., None], 3, axis=2)),
        ("edge gain", np.repeat(_gray_vis(np.maximum(edge_gain, 0.0), 0.03)[..., None], 3, axis=2)),
    ]
    h, w = sr.shape[:2]
    label_h = 24
    sheet = Image.new("RGB", (w * 3, (h + label_h) * 2), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (label, arr) in enumerate(panels):
        x = (i % 3) * w
        y = (i // 3) * (h + label_h)
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        sheet.paste(img, (x, y + label_h))
        draw.text((x + 6, y + 5), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _stats(prefix: str, value: np.ndarray, weight: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mean": _weighted_mean(value, weight),
        f"{prefix}_p90": _weighted_p(value, weight, 90.0),
        f"{prefix}_p99": _weighted_p(value, weight, 99.0),
    }


def _mean_dict(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for row in rows for k in row.keys() if isinstance(row.get(k), (float, int))})
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))], dtype=np.float64)
        if vals.size > 0:
            out[key] = float(vals.mean())
    return out


def _classify(summary: Dict[str, float]) -> Dict[str, str]:
    chroma_fraction = float(summary.get("delta_chroma_fraction_mean", float("nan")))
    hf_ratio = float(summary.get("delta_hf_band_fraction_mean", float("nan")))
    flat_ratio = float(summary.get("hf_flat_over_edge_ratio_mean", float("nan")))
    low_ratio = float(summary.get("delta_lf_band_fraction_mean", float("nan")))
    labels: List[str] = []
    if math.isfinite(chroma_fraction) and chroma_fraction < 0.20:
        labels.append("mostly_luma_not_color")
    elif math.isfinite(chroma_fraction) and chroma_fraction > 0.40:
        labels.append("contains_chroma_change")
    if math.isfinite(hf_ratio) and hf_ratio > 0.50:
        labels.append("mostly_high_frequency")
    if math.isfinite(low_ratio) and low_ratio > 0.35:
        labels.append("noticeable_low_frequency_drift")
    if math.isfinite(flat_ratio) and flat_ratio > 0.65:
        labels.append("substantial_flat_region_texture_or_noise")
    elif math.isfinite(flat_ratio) and flat_ratio < 0.35:
        labels.append("edge_concentrated")
    return {"summary_label": ",".join(labels) if labels else "mixed_or_unclear"}


def main() -> None:
    args = _parse_args()
    sr_dir = Path(args.sr_dir).expanduser().resolve()
    lr_dir = Path(args.lr_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve() if str(args.mask_dir) else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {output_dir}")
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_root = output_dir / "debug"
    sheet_root = output_dir / "sheet"

    sr_paths = _list_images(sr_dir)
    lr_paths = _list_images(lr_dir)
    mask_paths = _list_images(mask_dir) if mask_dir is not None and mask_dir.is_dir() else []
    if int(args.limit) > 0:
        sr_paths = sr_paths[: int(args.limit)]
    lr_lookup = _lookup(lr_paths)
    mask_lookup = _lookup(mask_paths)

    rows: List[Dict[str, float]] = []
    frames: List[Dict[str, object]] = []
    for index, sr_path in enumerate(tqdm(sr_paths, desc="SR-LR delta")):
        lr_path = _resolve(lr_paths, lr_lookup, sr_path, index, str(args.match_policy))
        if lr_path is None:
            continue
        sr = _load_rgb(sr_path)
        size = (sr.shape[1], sr.shape[0])
        lr = _load_rgb(lr_path, size=size)
        if mask_paths:
            mask_path = _resolve(mask_paths, mask_lookup, sr_path, index, str(args.match_policy))
            trust = _load_gray(mask_path, size=size) if mask_path is not None else np.ones(sr.shape[:2], dtype=np.float32)
        else:
            mask_path = None
            trust = np.ones(sr.shape[:2], dtype=np.float32)
        trust = np.clip(trust, 0.0, 1.0) ** max(float(args.mask_power), 0.0)

        delta = (sr - lr).astype(np.float32)
        delta_y, delta_chroma = _rgb_delta_chroma(delta)
        delta_abs = _abs_rgb_mean(delta)
        delta_gray_like = (
            (np.max(delta, axis=2) - np.min(delta, axis=2)) < (1.0 / 255.0)
        ).astype(np.float32)

        blur_hi = _box_blur_rgb(delta, int(args.highpass_kernel))
        blur_lo = _box_blur_rgb(delta, int(args.lowpass_kernel))
        delta_hf = delta - blur_hi
        delta_mf = blur_hi - blur_lo
        delta_lf = blur_lo
        hf_abs = _abs_rgb_mean(delta_hf)
        mf_abs = _abs_rgb_mean(delta_mf)
        lf_abs = _abs_rgb_mean(delta_lf)

        sr_y = _luma(sr)
        lr_y = _luma(lr)
        sr_edge = _grad_mag(sr_y)
        lr_edge = _grad_mag(lr_y)
        edge_gain = sr_edge - lr_edge
        edge_thr = max(float(np.percentile(sr_edge, float(args.edge_percentile))), 1e-8)
        flat_thr = max(float(np.percentile(lr_edge, float(args.flat_percentile))), 1e-8)
        edge_region = (sr_edge >= edge_thr).astype(np.float32)
        flat_region = (lr_edge <= flat_thr).astype(np.float32)

        total_band = hf_abs + mf_abs + lf_abs + 1e-8
        chroma_total = np.abs(delta_y) + delta_chroma + 1e-8
        row: Dict[str, float] = {
            "index": float(index),
            "mask_mean": float(trust.mean()),
            "delta_gray_like_ratio": _weighted_mean(delta_gray_like, trust),
            "delta_chroma_fraction": _weighted_mean(delta_chroma / chroma_total, trust),
            "delta_hf_band_fraction": _weighted_mean(hf_abs / total_band, trust),
            "delta_mf_band_fraction": _weighted_mean(mf_abs / total_band, trust),
            "delta_lf_band_fraction": _weighted_mean(lf_abs / total_band, trust),
            "edge_gain_positive_mean": _weighted_mean(np.maximum(edge_gain, 0.0), trust),
            "edge_gain_negative_mean": _weighted_mean(np.maximum(-edge_gain, 0.0), trust),
            "new_edge_ratio": _weighted_mean(((sr_edge >= edge_thr) & (lr_edge < edge_thr)).astype(np.float32), trust),
            "strengthened_edge_ratio": _weighted_mean(((sr_edge > lr_edge * 1.2) & (sr_edge >= edge_thr)).astype(np.float32), trust),
            "hf_vs_sr_edge_corr": _corr(hf_abs, sr_edge, trust),
            "hf_vs_edge_gain_corr": _corr(hf_abs, np.maximum(edge_gain, 0.0), trust),
            "hf_flat_energy": _weighted_mean(hf_abs, trust * flat_region),
            "hf_edge_energy": _weighted_mean(hf_abs, trust * edge_region),
            "hf_flat_over_edge_ratio": _weighted_mean(hf_abs, trust * flat_region)
            / max(_weighted_mean(hf_abs, trust * edge_region), 1e-8),
        }
        row.update(_stats("delta_abs", delta_abs, trust))
        row.update(_stats("delta_luma_abs", np.abs(delta_y), trust))
        row.update(_stats("delta_chroma_abs", delta_chroma, trust))
        row.update(_stats("delta_hf_abs", hf_abs, trust))
        row.update(_stats("delta_mf_abs", mf_abs, trust))
        row.update(_stats("delta_lf_abs", lf_abs, trust))
        rows.append(row)
        frames.append(
            {
                "stem": sr_path.stem,
                "sr": str(sr_path),
                "lr": str(lr_path),
                "mask": str(mask_path) if mask_path is not None else None,
                "metrics": row,
            }
        )

        if index < int(args.debug_limit):
            stem = sr_path.stem
            _save_rgb(debug_root / "delta_signed" / f"{stem}.png", _signed_vis(delta, float(args.vis_clip)))
            _save_gray(debug_root / "delta_abs" / f"{stem}.png", _gray_vis(delta_abs, float(args.vis_clip)))
            _save_gray(debug_root / "delta_hf_abs" / f"{stem}.png", _gray_vis(hf_abs, float(args.vis_clip)))
            _save_gray(debug_root / "delta_chroma_abs" / f"{stem}.png", _gray_vis(delta_chroma, float(args.vis_clip) * 0.5))
            _save_gray(debug_root / "edge_gain_pos" / f"{stem}.png", _gray_vis(np.maximum(edge_gain, 0.0), 0.03))
            _write_sheet(sheet_root / f"{stem}.png", sr, lr, delta, delta_hf, delta_chroma, edge_gain)

    summary = _mean_dict(rows)
    summary.update(_classify(summary))
    payload = {
        "version": "evaluate_sr_lr_delta_composition_v0",
        "sr_dir": str(sr_dir),
        "lr_dir": str(lr_dir),
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "output_dir": str(output_dir),
        "match_policy": str(args.match_policy),
        "num_frames": len(rows),
        "highpass_kernel": int(args.highpass_kernel),
        "lowpass_kernel": int(args.lowpass_kernel),
        "summary": summary,
        "frames": frames,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[sr-lr-delta-v0] frames: {len(rows)}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[sr-lr-delta-v0] summary: {output_dir / 'summary.json'}")
    print(f"[sr-lr-delta-v0] sheet  : {sheet_root}")


if __name__ == "__main__":
    main()
