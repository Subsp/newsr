#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upper-bound oracle for sprayed 2DGS/posterior newborn expressibility. "
            "It evaluates projected support, append visibility, and fixed-footprint "
            "linear appearance fits against an HF target."
        )
    )
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--append_dir", default="")
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        nargs=3,
        metavar=("NAME", "RENDER_DIR", "ALPHA_DIR"),
        default=[],
        help="Prior/newborn-only variant render directory plus its alpha directory. May be repeated.",
    )
    parser.add_argument("--match_policy", choices=["stem", "order"], default="stem")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--target_threshold", type=float, default=0.18)
    parser.add_argument("--geom_threshold", type=float, default=0.08)
    parser.add_argument("--reachable_dilate", type=int, default=2)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--leak_weight", type=float, default=0.15)
    parser.add_argument("--debug_limit", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _image_map(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem: p for p in paths}


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


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def _box_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return np.asarray(image, dtype=np.float32).copy()
    pad = k // 2
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)


def _normalize(value: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(arr, float(percentile)))
    if scale <= 1e-8:
        positive = arr[arr > 1e-8]
        if positive.size <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        # Sparse alpha/support maps can have p99 == 0 even when the visible
        # footprint is real. Fall back to the positive tail instead of erasing it.
        scale = float(np.percentile(positive, min(float(percentile), 99.0)))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    base = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return base.copy()
    out = np.zeros_like(base, dtype=bool)
    padded = np.pad(base, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + base.shape[0], dx : dx + base.shape[1]]
    return out


def _corr(a: np.ndarray, b: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if weight is None:
        w = np.ones_like(x, dtype=np.float32)
    else:
        w_raw = np.asarray(weight, dtype=np.float32)
        if w_raw.shape != np.asarray(a).shape:
            while w_raw.ndim < np.asarray(a).ndim:
                w_raw = w_raw[..., None]
            w_raw = np.broadcast_to(w_raw, np.asarray(a).shape)
        w = w_raw.reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 1e-8)
    if int(np.count_nonzero(keep)) < 16:
        return 0.0
    x = x[keep]
    y = y[keep]
    w = w[keep]
    w = w / max(float(w.sum()), 1e-8)
    xm = float(np.sum(w * x))
    ym = float(np.sum(w * y))
    xv = x - xm
    yv = y - ym
    denom = math.sqrt(float(np.sum(w * xv * xv)) * float(np.sum(w * yv * yv)))
    if denom <= 1e-8:
        return 0.0
    return float(np.sum(w * xv * yv) / denom)


def _psnr(a: np.ndarray, b: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    err = (np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2
    if weight is None:
        mse = float(np.mean(err))
    else:
        w = np.asarray(weight, dtype=np.float32)
        while w.ndim < err.ndim:
            w = w[..., None]
        mse = float(np.sum(err * w) / max(float(np.sum(w) * err.shape[-1]), 1e-8))
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _safe_mean(value: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    if arr.size <= 0:
        return 0.0
    return float(np.mean(arr))


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(mask, dtype=bool)))


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _linear_rgb_oracle(
    *,
    basis: np.ndarray,
    target_hf: np.ndarray,
    fit_mask: np.ndarray,
    leak_mask: np.ndarray,
    leak_weight: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    b = np.asarray(basis, dtype=np.float32)
    if b.ndim == 2:
        b = b[..., None]
    if b.shape[-1] == 1:
        b = np.broadcast_to(b, target_hf.shape)
    t = np.asarray(target_hf, dtype=np.float32)
    w_fit = np.asarray(fit_mask, dtype=np.float32)
    w_leak = np.asarray(leak_mask, dtype=np.float32) * float(leak_weight)
    w = w_fit + w_leak
    beta = np.zeros((3,), dtype=np.float32)
    pred = np.zeros_like(t, dtype=np.float32)
    fit_count = int(np.count_nonzero(w_fit > 1e-8))
    target_energy = float(np.sum(w_fit[..., None] * t * t))
    if fit_count < 16 or target_energy <= 1e-10:
        return pred, beta, {
            "valid_fit": 0.0,
            "fit_pixels": float(fit_count),
            "explained_energy": 0.0,
            "psnr_fit": float("nan"),
            "signed_corr_fit": 0.0,
            "abs_corr_fit": 0.0,
            "offtarget_leak_ratio": 0.0,
            "beta_abs_max": 0.0,
            "beta_l2": 0.0,
            "pred_abs_p99": 0.0,
            "pred_saturation_ratio": 0.0,
        }
    for channel in range(3):
        bc = b[..., channel]
        tc = t[..., channel]
        numerator = float(np.sum(w_fit * bc * tc))
        denominator = float(np.sum(w * bc * bc) + 1e-8)
        beta[channel] = numerator / denominator
        pred[..., channel] = bc * beta[channel]
    fit_w = w_fit[..., None]
    residual_energy = float(np.sum(fit_w * (t - pred) * (t - pred)))
    explained = 1.0 - residual_energy / max(target_energy, 1e-8)
    on_abs = np.abs(pred).mean(axis=2)
    off_leak = _safe_mean(on_abs, leak_mask) / max(_safe_mean(on_abs, fit_mask), 1e-8)
    stats = {
        "valid_fit": 1.0,
        "fit_pixels": float(fit_count),
        "explained_energy": float(explained),
        "psnr_fit": _psnr(t, pred, fit_mask.astype(np.float32)),
        "signed_corr_fit": _corr(pred, t, fit_mask.astype(np.float32)),
        "abs_corr_fit": _corr(np.abs(pred).mean(axis=2), np.abs(t).mean(axis=2), fit_mask.astype(np.float32)),
        "offtarget_leak_ratio": float(off_leak),
        "beta_abs_max": float(np.max(np.abs(beta))),
        "beta_l2": float(np.linalg.norm(beta)),
        "pred_abs_p99": float(np.percentile(np.abs(pred), 99.0)),
        "pred_saturation_ratio": float(np.mean(np.abs(pred) > 1.0)),
    }
    return pred, beta, stats


def _linear_scalar_oracle(
    *,
    basis: np.ndarray,
    target_hf: np.ndarray,
    fit_mask: np.ndarray,
    leak_mask: np.ndarray,
    leak_weight: float,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    b = np.asarray(basis, dtype=np.float32)
    t = np.asarray(target_hf, dtype=np.float32)
    w_fit = np.asarray(fit_mask, dtype=np.float32)
    w_leak = np.asarray(leak_mask, dtype=np.float32) * float(leak_weight)
    w = w_fit + w_leak
    wb_fit = w_fit[..., None]
    wb = w[..., None]
    fit_count = int(np.count_nonzero(w_fit > 1e-8))
    target_energy = float(np.sum(wb_fit * t * t))
    if fit_count < 16 or target_energy <= 1e-10:
        pred = np.zeros_like(t, dtype=np.float32)
        return pred, 0.0, {
            "valid_fit": 0.0,
            "fit_pixels": float(fit_count),
            "explained_energy": 0.0,
            "psnr_fit": float("nan"),
            "signed_corr_fit": 0.0,
            "abs_corr_fit": 0.0,
            "offtarget_leak_ratio": 0.0,
            "gamma": 0.0,
            "gamma_abs": 0.0,
            "pred_abs_p99": 0.0,
            "pred_saturation_ratio": 0.0,
        }
    numerator = float(np.sum(wb_fit * b * t))
    denominator = float(np.sum(wb * b * b) + 1e-8)
    gamma = numerator / denominator
    pred = b * gamma
    residual_energy = float(np.sum(wb_fit * (t - pred) * (t - pred)))
    explained = 1.0 - residual_energy / max(target_energy, 1e-8)
    pred_abs = np.abs(pred).mean(axis=2)
    off_leak = _safe_mean(pred_abs, leak_mask) / max(_safe_mean(pred_abs, fit_mask), 1e-8)
    stats = {
        "valid_fit": 1.0,
        "fit_pixels": float(fit_count),
        "explained_energy": float(explained),
        "psnr_fit": _psnr(t, pred, fit_mask.astype(np.float32)),
        "signed_corr_fit": _corr(pred, t, fit_mask.astype(np.float32)),
        "abs_corr_fit": _corr(pred_abs, np.abs(t).mean(axis=2), fit_mask.astype(np.float32)),
        "offtarget_leak_ratio": float(off_leak),
        "gamma": float(gamma),
        "gamma_abs": float(abs(gamma)),
        "pred_abs_p99": float(np.percentile(np.abs(pred), 99.0)),
        "pred_saturation_ratio": float(np.mean(np.abs(pred) > 1.0)),
    }
    return pred, gamma, stats


def _match_variant_paths(
    *,
    base_paths: List[Path],
    target_paths: List[Path],
    render_paths: List[Path],
    alpha_paths: List[Path],
    append_paths: List[Path],
    policy: str,
) -> Iterable[Tuple[str, Path, Path, Path, Path, Optional[Path]]]:
    if policy == "order":
        n = min(len(base_paths), len(target_paths), len(render_paths), len(alpha_paths))
        if append_paths:
            n = min(n, len(append_paths))
        for idx in range(n):
            yield (
                f"{idx:05d}",
                base_paths[idx],
                target_paths[idx],
                render_paths[idx],
                alpha_paths[idx],
                append_paths[idx] if append_paths else None,
            )
        return

    base_by_stem = _image_map(base_paths)
    target_by_stem = _image_map(target_paths)
    render_by_stem = _image_map(render_paths)
    alpha_by_stem = _image_map(alpha_paths)
    append_by_stem = _image_map(append_paths) if append_paths else {}
    stems = set(base_by_stem) & set(target_by_stem) & set(render_by_stem) & set(alpha_by_stem)
    if append_paths:
        stems &= set(append_by_stem)
    for stem in sorted(stems):
        yield (
            stem,
            base_by_stem[stem],
            target_by_stem[stem],
            render_by_stem[stem],
            alpha_by_stem[stem],
            append_by_stem.get(stem),
        )


def _prefix_stats(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def _mean(rows: List[Dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.variant:
        raise ValueError("At least one --variant NAME RENDER_DIR ALPHA_DIR is required.")

    base_dir = Path(args.base_dir).expanduser().resolve()
    append_dir = Path(args.append_dir).expanduser().resolve() if str(args.append_dir).strip() else None
    target_dir = Path(args.target_dir).expanduser().resolve()

    base_paths = _list_images(base_dir)
    append_paths = _list_images(append_dir) if append_dir is not None and append_dir.is_dir() else []
    target_paths = _list_images(target_dir)
    if int(args.limit) > 0:
        base_paths = base_paths[: int(args.limit)]
        append_paths = append_paths[: int(args.limit)] if append_paths else append_paths
        target_paths = target_paths[: int(args.limit)]

    all_rows: List[Dict[str, object]] = []
    debug_written = 0

    for variant_name, render_dir_raw, alpha_dir_raw in args.variant:
        render_dir = Path(render_dir_raw).expanduser().resolve()
        alpha_dir = Path(alpha_dir_raw).expanduser().resolve()
        render_paths = _list_images(render_dir)
        alpha_paths = _list_images(alpha_dir)
        if int(args.limit) > 0:
            render_paths = render_paths[: int(args.limit)]
            alpha_paths = alpha_paths[: int(args.limit)]

        for stem, base_path, target_path, render_path, alpha_path, append_path in _match_variant_paths(
            base_paths=base_paths,
            target_paths=target_paths,
            render_paths=render_paths,
            alpha_paths=alpha_paths,
            append_paths=append_paths,
            policy=str(args.match_policy),
        ):
            base_image = _load_rgb(base_path)
            size = (base_image.shape[1], base_image.shape[0])
            target_image = _load_rgb(target_path, size=size)
            prior_image = _load_rgb(render_path, size=size)
            alpha = _load_gray(alpha_path, size=size)
            append_image = _load_rgb(append_path, size=size) if append_path is not None else None

            base_hf = base_image - _box_blur(base_image, int(args.highpass_kernel))
            target_hf = target_image - _box_blur(target_image, int(args.highpass_kernel))
            target_delta_hf = target_hf - base_hf
            target_abs = np.abs(target_delta_hf).mean(axis=2)
            target_norm = _normalize(target_abs, float(args.norm_percentile))
            target_active = target_norm >= float(args.target_threshold)

            alpha_norm = _normalize(alpha, float(args.norm_percentile))
            geom = alpha_norm >= float(args.geom_threshold)
            geom_dilated = _dilate(geom, int(args.reachable_dilate))
            reachable = target_active & geom_dilated
            leak_mask = geom_dilated & ~target_active

            prior_hf = prior_image - _box_blur(prior_image, int(args.highpass_kernel))
            prior_abs = np.abs(prior_hf).mean(axis=2)

            row: Dict[str, object] = {
                "variant": str(variant_name),
                "stem": str(stem),
                "base": str(base_path),
                "target": str(target_path),
                "prior_render": str(render_path),
                "prior_alpha": str(alpha_path),
                "append": str(append_path) if append_path is not None else None,
                "target_active_ratio": _fraction(target_active),
                "target_energy_mean": float(np.mean(target_abs)),
                "target_norm_mean": float(np.mean(target_norm)),
                "alpha_mean": float(np.mean(alpha)),
                "alpha_p99": float(np.percentile(alpha, 99.0)),
                "alpha_norm_mean": float(np.mean(alpha_norm)),
                "geom_ratio": _fraction(geom),
                "geom_dilated_ratio": _fraction(geom_dilated),
                "reachable_ratio": _fraction(reachable),
                "reachable_target_fraction": _count(reachable) / max(_count(target_active), 1),
                "geom_support_recall": _count(geom_dilated & target_active) / max(_count(target_active), 1),
                "geom_precision": _count(geom_dilated & target_active) / max(_count(geom_dilated), 1),
                "geom_offtarget_ratio": _count(geom_dilated & ~target_active) / max(_count(geom_dilated), 1),
                "prior_abs_corr_global": _corr(prior_abs, target_norm),
                "prior_abs_corr_reachable": _corr(prior_abs, target_norm, reachable.astype(np.float32)),
                "prior_energy_on_reachable": _safe_mean(prior_abs, reachable),
                "prior_energy_off_target": _safe_mean(prior_abs, ~target_active),
            }

            if append_image is not None:
                append_delta = append_image - base_image
                append_abs = np.abs(append_delta).mean(axis=2)
                append_hf = append_delta - _box_blur(append_delta, int(args.highpass_kernel))
                append_hf_abs = np.abs(append_hf).mean(axis=2)
                row.update(
                    {
                        "append_l1": float(np.mean(np.abs(append_delta))),
                        "append_l1_geom": _safe_mean(append_abs, geom_dilated),
                        "append_l1_reachable": _safe_mean(append_abs, reachable),
                        "append_changed_ratio": _fraction(np.max(np.abs(append_delta), axis=2) > (1.0 / 255.0)),
                        "append_hf_corr_global": _corr(append_hf_abs, target_norm),
                        "append_hf_corr_reachable": _corr(append_hf_abs, target_norm, reachable.astype(np.float32)),
                        "actual_to_unoccluded_mass": float(np.mean(append_abs) / max(float(np.mean(alpha)), 1e-8)),
                        "actual_to_unoccluded_mass_geom": _safe_mean(append_abs, geom_dilated)
                        / max(_safe_mean(alpha, geom_dilated), 1e-8),
                    }
                )

            alpha_pred, alpha_beta, alpha_stats = _linear_rgb_oracle(
                basis=alpha,
                target_hf=target_delta_hf,
                fit_mask=reachable,
                leak_mask=leak_mask,
                leak_weight=float(args.leak_weight),
            )
            row.update(_prefix_stats("alpha_rgb_oracle", alpha_stats))
            row["alpha_rgb_oracle_beta_r"] = float(alpha_beta[0])
            row["alpha_rgb_oracle_beta_g"] = float(alpha_beta[1])
            row["alpha_rgb_oracle_beta_b"] = float(alpha_beta[2])

            prior_scalar_pred, prior_gamma, prior_scalar_stats = _linear_scalar_oracle(
                basis=prior_hf,
                target_hf=target_delta_hf,
                fit_mask=reachable,
                leak_mask=leak_mask,
                leak_weight=float(args.leak_weight),
            )
            row.update(_prefix_stats("prior_hf_scalar_oracle", prior_scalar_stats))
            row["prior_hf_scalar_oracle_gamma"] = float(prior_gamma)

            prior_rgb_pred, prior_beta, prior_rgb_stats = _linear_rgb_oracle(
                basis=prior_hf,
                target_hf=target_delta_hf,
                fit_mask=reachable,
                leak_mask=leak_mask,
                leak_weight=float(args.leak_weight),
            )
            row.update(_prefix_stats("prior_hf_rgb_oracle", prior_rgb_stats))
            row["prior_hf_rgb_oracle_beta_r"] = float(prior_beta[0])
            row["prior_hf_rgb_oracle_beta_g"] = float(prior_beta[1])
            row["prior_hf_rgb_oracle_beta_b"] = float(prior_beta[2])

            all_rows.append(row)

            if debug_written < int(args.debug_limit):
                debug_root = output_dir / "debug" / str(variant_name) / str(stem)
                _save_rgb(debug_root / "base.png", base_image)
                _save_rgb(debug_root / "target.png", target_image)
                _save_rgb(debug_root / "prior_render.png", prior_image)
                _save_gray(debug_root / "target_hf_abs_norm.png", target_norm)
                _save_gray(debug_root / "alpha_norm.png", alpha_norm)
                _save_gray(debug_root / "geom_support.png", geom_dilated.astype(np.float32))
                _save_gray(debug_root / "reachable.png", reachable.astype(np.float32))
                _save_rgb(debug_root / "alpha_rgb_oracle_signed.png", np.clip(alpha_pred * 4.0 + 0.5, 0.0, 1.0))
                _save_gray(debug_root / "alpha_rgb_oracle_abs.png", _normalize(np.abs(alpha_pred).mean(axis=2), 99.0))
                _save_rgb(
                    debug_root / "prior_hf_rgb_oracle_signed.png",
                    np.clip(prior_rgb_pred * 4.0 + 0.5, 0.0, 1.0),
                )
                _save_gray(debug_root / "prior_hf_rgb_oracle_abs.png", _normalize(np.abs(prior_rgb_pred).mean(axis=2), 99.0))
                if append_image is not None:
                    _save_rgb(debug_root / "append.png", append_image)
                    _save_gray(debug_root / "append_abs_x80.png", np.clip(np.abs(append_image - base_image).mean(axis=2) * 80.0, 0.0, 1.0))
                debug_written += 1

    if not all_rows:
        raise RuntimeError("No matched frames were evaluated.")

    numeric_keys = sorted(
        {
            key
            for row in all_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    means = {key: _mean(all_rows, key) for key in numeric_keys}
    by_variant: Dict[str, Dict[str, float]] = {}
    for variant in sorted({str(row["variant"]) for row in all_rows}):
        variant_rows = [row for row in all_rows if str(row["variant"]) == variant]
        by_variant[variant] = {key: _mean(variant_rows, key) for key in numeric_keys}

    gates = {}
    for variant, values in by_variant.items():
        gates[variant] = {
            "continue_geometry": bool(values.get("geom_support_recall", 0.0) >= 0.5),
            "continue_alpha_oracle": bool(values.get("alpha_rgb_oracle_signed_corr_fit", 0.0) >= 0.2),
            "continue_prior_oracle": bool(values.get("prior_hf_rgb_oracle_signed_corr_fit", 0.0) >= 0.2),
            "leak_ok_alpha": bool(values.get("alpha_rgb_oracle_offtarget_leak_ratio", 1.0) <= 0.30),
            "leak_ok_prior": bool(values.get("prior_hf_rgb_oracle_offtarget_leak_ratio", 1.0) <= 0.30),
        }

    summary = {
        "version": "evaluate_2dgs_expressibility_oracle_v0",
        "base_dir": str(base_dir),
        "append_dir": str(append_dir) if append_dir is not None else None,
        "target_dir": str(target_dir),
        "output_dir": str(output_dir),
        "match_policy": str(args.match_policy),
        "num_rows": int(len(all_rows)),
        "num_variants": int(len(by_variant)),
        "highpass_kernel": int(args.highpass_kernel),
        "target_threshold": float(args.target_threshold),
        "geom_threshold": float(args.geom_threshold),
        "reachable_dilate": int(args.reachable_dilate),
        "leak_weight": float(args.leak_weight),
        "means": means,
        "by_variant": by_variant,
        "gates": gates,
        "rows": all_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
