#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - optional diagnostic dependency
    distance_transform_edt = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _stem(path_or_name: str) -> str:
    return Path(str(path_or_name)).stem


def _build_lookup(paths: Iterable[Path]) -> Dict[str, Path]:
    return {_stem(path.name).lower(): path for path in paths}


def _load_manifest_frames(root: Optional[Path], split: str) -> Dict[str, str]:
    if root is None:
        return {}
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames") or []
    out: Dict[str, str] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if str(frame.get("split", split)) not in {split, ""} and "split" in frame:
            continue
        stem = str(frame.get("stem") or f"{int(frame.get('index', 0)):05d}")
        image_name = str(frame.get("image_name") or stem)
        out[_stem(stem).lower()] = image_name
    return out


def _resolve_pairs(
    *,
    hf_paths: Sequence[Path],
    target_paths: Sequence[Path],
    anchor_paths: Sequence[Path],
    stem_to_image: Dict[str, str],
    match_policy: str,
) -> List[Tuple[Path, Path, Optional[Path], str]]:
    target_lookup = _build_lookup(target_paths)
    anchor_lookup = _build_lookup(anchor_paths) if anchor_paths else {}
    pairs: List[Tuple[Path, Path, Optional[Path], str]] = []
    missing: List[str] = []
    for idx, hf_path in enumerate(hf_paths):
        hf_key = _stem(hf_path.name).lower()
        image_name = stem_to_image.get(hf_key, _stem(hf_path.name))
        key = _stem(image_name).lower()
        target = target_lookup.get(key)
        anchor = anchor_lookup.get(key) if anchor_lookup else None
        if target is None and str(match_policy) in {"order", "order_if_needed", "llff_train_order"}:
            if idx < len(target_paths):
                target = target_paths[idx]
            if anchor_paths and idx < len(anchor_paths):
                anchor = anchor_paths[idx]
        if target is None:
            missing.append(image_name)
            continue
        if anchor_paths and anchor is None and str(match_policy) in {"order", "order_if_needed", "llff_train_order"}:
            if idx < len(anchor_paths):
                anchor = anchor_paths[idx]
        pairs.append((hf_path, target, anchor, image_name))
    if not pairs:
        raise RuntimeError(f"No matched HF/target pairs. Missing examples: {missing[:12]}")
    if missing:
        print(f"[hf-align-v0] warning: skipped {len(missing)} unmatched frames; first={missing[:8]}")
    return pairs


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _box_blur(gray: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return gray.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(gray.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    ).astype(np.float32) / float(k * k)


def _highpass_abs(rgb: np.ndarray, kernel: int) -> np.ndarray:
    gray = _rgb_to_gray(rgb)
    return np.abs(gray - _box_blur(gray, kernel)).astype(np.float32)


def _residual_highpass_abs(target: np.ndarray, anchor: np.ndarray, kernel: int) -> np.ndarray:
    target_gray = _rgb_to_gray(target)
    anchor_gray = _rgb_to_gray(anchor)
    target_hf = target_gray - _box_blur(target_gray, kernel)
    anchor_hf = anchor_gray - _box_blur(anchor_gray, kernel)
    return np.abs(target_hf - anchor_hf).astype(np.float32)


def _normalize(arr: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(arr, float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _threshold(arr: np.ndarray, percentile: float, min_value: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    value = max(float(min_value), float(np.percentile(arr, float(percentile))))
    return arr >= value


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) > 1e-8 else 0.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return _safe_div(float(np.sum(a * b)), den)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return _safe_div(float(np.sum(a * b)), den)


def _metrics_for_maps(
    hf: np.ndarray,
    target: np.ndarray,
    *,
    target_percentile: float,
    hf_percentile: float,
    min_target: float,
    min_hf: float,
) -> Dict[str, float]:
    target_bin = _threshold(target, target_percentile, min_target)
    hf_bin = _threshold(hf, hf_percentile, min_hf)
    tp = float(np.logical_and(target_bin, hf_bin).sum())
    fp = float(np.logical_and(~target_bin, hf_bin).sum())
    fn = float(np.logical_and(target_bin, ~hf_bin).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)

    soft_inter = float(np.minimum(hf, target).sum())
    soft_union = float(np.maximum(hf, target).sum())
    target_energy = float(target.sum())
    hf_energy = float(hf.sum())
    out = {
        "pearson": _pearson(hf, target),
        "cosine": _cosine(hf, target),
        "soft_iou": _safe_div(soft_inter, soft_union),
        "soft_target_coverage": _safe_div(soft_inter, target_energy),
        "soft_hf_precision": _safe_div(soft_inter, hf_energy),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "target_ratio": float(target_bin.mean()),
        "hf_ratio": float(hf_bin.mean()),
    }
    if distance_transform_edt is not None:
        if np.any(target_bin) and np.any(hf_bin):
            target_dist = distance_transform_edt(~target_bin)
            hf_dist = distance_transform_edt(~hf_bin)
            out["hf_to_target_px"] = float(target_dist[hf_bin].mean())
            out["target_to_hf_px"] = float(hf_dist[target_bin].mean())
        else:
            out["hf_to_target_px"] = 0.0
            out["target_to_hf_px"] = 0.0
    return out


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _save_overlay(path: Path, hf: np.ndarray, target: np.ndarray, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*hf.shape, 3), dtype=np.float32)
    rgb[..., 0] = target
    rgb[..., 1] = hf
    rgb[..., 2] = hf
    rgb = np.clip(rgb, 0.0, 1.0)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for row in rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image-domain HF alignment of a rendered Gaussian HF subset.")
    parser.add_argument("--subset_root", required=True, help="Root produced by run_render_cave_hf_subset_v0_kitchen.sh")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--hf_rgb_dir", default="", help="Override subset_root/<split>/hf_rgb")
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--anchor_dir", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_mode", default="residual_hf", choices=["image_hf", "residual_hf"])
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed", "llff_train_order"])
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--target_percentiles", default="90,95,97")
    parser.add_argument("--hf_percentile", type=float, default=90.0)
    parser.add_argument("--min_target", type=float, default=0.05)
    parser.add_argument("--min_hf", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write_images", action="store_true")
    args = parser.parse_args()

    subset_root = Path(args.subset_root).expanduser().resolve()
    split = str(args.split)
    hf_dir = Path(args.hf_rgb_dir).expanduser().resolve() if str(args.hf_rgb_dir).strip() else subset_root / split / "hf_rgb"
    target_dir = Path(args.target_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve() if str(args.anchor_dir).strip() else None
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    hf_paths = _list_images(hf_dir)
    if int(args.limit) > 0:
        hf_paths = hf_paths[: int(args.limit)]
    target_paths = _list_images(target_dir)
    anchor_paths = _list_images(anchor_dir) if anchor_dir is not None else []
    if str(args.target_mode) == "residual_hf" and not anchor_paths:
        raise ValueError("--target_mode residual_hf requires --anchor_dir")
    stem_to_image = _load_manifest_frames(subset_root, split)
    pairs = _resolve_pairs(
        hf_paths=hf_paths,
        target_paths=target_paths,
        anchor_paths=anchor_paths,
        stem_to_image=stem_to_image,
        match_policy=str(args.match_policy),
    )

    target_percentiles = [float(x.strip()) for x in str(args.target_percentiles).split(",") if x.strip()]
    rows_by_percentile: Dict[str, List[Dict[str, float]]] = {str(p): [] for p in target_percentiles}
    frames = []

    print(f"[hf-align-v0] subset : {subset_root}")
    print(f"[hf-align-v0] hf_rgb : {hf_dir}")
    print(f"[hf-align-v0] target : {target_dir}")
    print(f"[hf-align-v0] anchor : {anchor_dir if anchor_dir is not None else '<none>'}")
    print(f"[hf-align-v0] output : {output_root}")
    print(f"[hf-align-v0] mode   : {args.target_mode} pairs={len(pairs)}")

    for index, (hf_path, target_path, anchor_path, image_name) in enumerate(tqdm(pairs, desc="hf alignment")):
        hf_rgb = _load_rgb(hf_path)
        size = (hf_rgb.shape[1], hf_rgb.shape[0])
        target_rgb = _load_rgb(target_path, size=size)
        if str(args.target_mode) == "residual_hf":
            assert anchor_path is not None
            anchor_rgb = _load_rgb(anchor_path, size=size)
            target_raw = _residual_highpass_abs(target_rgb, anchor_rgb, int(args.highpass_kernel))
        else:
            target_raw = _highpass_abs(target_rgb, int(args.highpass_kernel))
        hf_raw = _highpass_abs(hf_rgb, int(args.highpass_kernel))
        target = _normalize(target_raw, float(args.norm_percentile))
        hf = _normalize(hf_raw, float(args.norm_percentile))

        frame = {
            "index": int(index),
            "image_name": str(image_name),
            "hf": str(hf_path),
            "target": str(target_path),
            "anchor": str(anchor_path) if anchor_path is not None else "",
            "metrics": {},
        }
        for percentile in target_percentiles:
            metrics = _metrics_for_maps(
                hf,
                target,
                target_percentile=float(percentile),
                hf_percentile=float(args.hf_percentile),
                min_target=float(args.min_target),
                min_hf=float(args.min_hf),
            )
            rows_by_percentile[str(percentile)].append(metrics)
            frame["metrics"][str(percentile)] = metrics

        if bool(args.write_images):
            label = f"{index:05d}_{_stem(image_name)}"
            _save_gray(output_root / "target_hf" / f"{label}.png", target)
            _save_gray(output_root / "hf_hf" / f"{label}.png", hf)
            _save_overlay(output_root / "overlay" / f"{label}.png", hf, target, frame["metrics"][str(target_percentiles[0])])

        frames.append(frame)

    summary = {
        "version": "evaluate_hf_subset_image_alignment_v0",
        "subset_root": str(subset_root),
        "split": split,
        "hf_rgb_dir": str(hf_dir),
        "target_dir": str(target_dir),
        "anchor_dir": str(anchor_dir) if anchor_dir is not None else "",
        "target_mode": str(args.target_mode),
        "num_frames": int(len(frames)),
        "highpass_kernel": int(args.highpass_kernel),
        "norm_percentile": float(args.norm_percentile),
        "hf_percentile": float(args.hf_percentile),
        "target_percentiles": target_percentiles,
        "mean": {key: _mean_metrics(rows) for key, rows in rows_by_percentile.items()},
        "frames": frames,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[hf-align-v0] mean metrics")
    for percentile in target_percentiles:
        key = str(percentile)
        mean = summary["mean"][key]
        print(
            f"  p{percentile:g}: "
            f"corr={mean.get('pearson', 0.0):.4f} "
            f"cos={mean.get('cosine', 0.0):.4f} "
            f"soft_iou={mean.get('soft_iou', 0.0):.4f} "
            f"prec={mean.get('precision', 0.0):.4f} "
            f"rec={mean.get('recall', 0.0):.4f} "
            f"f1={mean.get('f1', 0.0):.4f} "
            f"hf2t={mean.get('hf_to_target_px', 0.0):.2f}px "
            f"t2hf={mean.get('target_to_hf_px', 0.0):.2f}px"
        )
    print(f"[hf-align-v0] summary: {summary_path}")


if __name__ == "__main__":
    main()
