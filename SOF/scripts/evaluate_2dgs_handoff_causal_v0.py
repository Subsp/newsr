#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate four counterfactual renders for 2DGS-posterior handoff: "
            "R00=base, R10=base+newborn append, R01=modified_base_only, R11=modified_base+newborn."
        )
    )
    parser.add_argument("--r00_dir", required=True)
    parser.add_argument("--r10_dir", required=True)
    parser.add_argument("--r01_dir", required=True)
    parser.add_argument("--r11_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_dir", default="")
    parser.add_argument("--match_policy", choices=["stem", "order"], default="stem")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vis_scale", type=float, default=80.0)
    parser.add_argument("--change_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--lowpass_kernel", type=int, default=21)
    parser.add_argument("--target_threshold", type=float, default=0.18)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def _box_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / float(k * k)


def _luma(image: np.ndarray) -> np.ndarray:
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def _normalize(value: np.ndarray, percentile: float) -> np.ndarray:
    value = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(value, float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if weight is None:
        w = np.ones_like(x, dtype=np.float32)
    else:
        w = np.asarray(weight, dtype=np.float32).reshape(-1)
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


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _pairs(paths: Dict[str, List[Path]], policy: str):
    names = ["r00", "r10", "r01", "r11"]
    if policy == "order":
        n = min(len(paths[name]) for name in names)
        for index in range(n):
            yield f"{index:05d}", {name: paths[name][index] for name in names}
        return
    maps = {name: {p.stem: p for p in paths[name]} for name in names}
    for stem in sorted(set.intersection(*(set(m.keys()) for m in maps.values()))):
        yield stem, {name: maps[name][stem] for name in names}


def _target_from_gt(base: np.ndarray, gt: Optional[np.ndarray], highpass_kernel: int, norm_percentile: float) -> Tuple[np.ndarray, np.ndarray]:
    if gt is None:
        zero = np.zeros(base.shape[:2], dtype=np.float32)
        return zero, zero.astype(bool)
    hp_base = base - _box_blur(base, highpass_kernel)
    hp_gt = gt - _box_blur(gt, highpass_kernel)
    target = _normalize((0.65 * np.abs(hp_gt - hp_base) + 0.35 * np.abs(hp_gt)).mean(axis=2), norm_percentile)
    return target, target >= 0.0


def _delta_stats(
    name: str,
    delta: np.ndarray,
    *,
    base: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    change_threshold: float,
    highpass_kernel: int,
    lowpass_kernel: int,
    target_threshold: float,
    norm_percentile: float,
) -> Dict[str, float]:
    abs_delta = np.abs(delta)
    changed = np.max(abs_delta, axis=2) > float(change_threshold)
    hf_delta = delta - _box_blur(delta, highpass_kernel)
    lf_delta = _box_blur(delta, lowpass_kernel)
    hf_energy = _normalize(np.abs(hf_delta).mean(axis=2), norm_percentile)
    active = target >= float(target_threshold)
    off = ~active
    luma_delta = _luma(base + delta) - _luma(base)
    dark = np.maximum(-luma_delta, 0.0)
    bright = np.maximum(luma_delta, 0.0)
    signed_target = target - _box_blur(target[..., None].repeat(3, axis=2), highpass_kernel).mean(axis=2)
    signed_delta = hf_delta.mean(axis=2)
    sign_keep = np.abs(signed_target) > np.percentile(np.abs(signed_target), 70.0)
    sign_agree = (
        float(np.mean(np.sign(signed_target[sign_keep]) == np.sign(signed_delta[sign_keep])))
        if np.any(sign_keep)
        else 0.0
    )
    return {
        f"{name}_l1": float(np.mean(abs_delta)),
        f"{name}_p99": float(np.percentile(abs_delta, 99.0)),
        f"{name}_max": float(np.max(abs_delta)),
        f"{name}_changed_ratio": float(np.mean(changed)),
        f"{name}_changed_on_target": float(np.mean(changed[active])) if np.any(active) else 0.0,
        f"{name}_changed_off_target": float(np.mean(changed[off])) if np.any(off) else 0.0,
        f"{name}_hf_corr": _corr(hf_energy, target),
        f"{name}_hf_corr_active": _corr(hf_energy, target, active.astype(np.float32)),
        f"{name}_hf_on_energy": float(np.mean(hf_energy[active])) if np.any(active) else 0.0,
        f"{name}_hf_off_energy": float(np.mean(hf_energy[off])) if np.any(off) else 0.0,
        f"{name}_hf_leak_ratio": float(
            (np.mean(hf_energy[off]) if np.any(off) else 0.0)
            / max(float(np.mean(hf_energy[active])) if np.any(active) else 0.0, 1e-8)
        ),
        f"{name}_lf_abs_mean": float(np.mean(np.abs(lf_delta))),
        f"{name}_dark_mean": float(np.mean(dark)),
        f"{name}_bright_mean": float(np.mean(bright)),
        f"{name}_dark_to_bright": float(np.sum(dark) / max(float(np.sum(bright)), 1e-8)),
        f"{name}_signed_hf_agreement": sign_agree,
    }


def _mean(rows: List[Dict[str, float | str]], key: str) -> float:
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

    dirs = {
        "r00": Path(args.r00_dir).expanduser().resolve(),
        "r10": Path(args.r10_dir).expanduser().resolve(),
        "r01": Path(args.r01_dir).expanduser().resolve(),
        "r11": Path(args.r11_dir).expanduser().resolve(),
    }
    paths = {name: _list_images(path) for name, path in dirs.items()}
    if int(args.limit) > 0:
        paths = {name: value[: int(args.limit)] for name, value in paths.items()}
    gt_dir = Path(args.gt_dir).expanduser().resolve() if str(args.gt_dir).strip() else None
    gt_by_stem = {p.stem: p for p in _list_images(gt_dir)} if gt_dir is not None and gt_dir.is_dir() else {}

    rows: List[Dict[str, float | str]] = []
    vis_dirs = {
        "append": output_dir / "append_R10_minus_R00_x",
        "parent": output_dir / "parent_R01_minus_R00_x",
        "newborn_after": output_dir / "newborn_R11_minus_R01_x",
        "total": output_dir / "total_R11_minus_R00_x",
        "interaction": output_dir / "interaction_x",
    }
    for path in vis_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    for index, (stem, item) in enumerate(_pairs(paths, str(args.match_policy))):
        if int(args.limit) > 0 and index >= int(args.limit):
            break
        r00 = _load_rgb(item["r00"])
        size = (r00.shape[1], r00.shape[0])
        r10 = _load_rgb(item["r10"], size=size)
        r01 = _load_rgb(item["r01"], size=size)
        r11 = _load_rgb(item["r11"], size=size)
        gt = None
        gt_path = gt_by_stem.get(stem)
        if gt_path is None and gt_by_stem:
            gt_paths = sorted(gt_by_stem.values())
            if index < len(gt_paths):
                gt_path = gt_paths[index]
        if gt_path is not None:
            gt = _load_rgb(gt_path, size=size)

        deltas = {
            "append": r10 - r00,
            "parent": r01 - r00,
            "newborn_after": r11 - r01,
            "total": r11 - r00,
            "interaction": r11 - r10 - r01 + r00,
        }
        target, _ = _target_from_gt(r00, gt, int(args.highpass_kernel), float(args.norm_percentile))
        target_mask = target >= float(args.target_threshold)
        row: Dict[str, float | str] = {
            "stem": stem,
            "r00": str(item["r00"]),
            "r10": str(item["r10"]),
            "r01": str(item["r01"]),
            "r11": str(item["r11"]),
            "gt": str(gt_path) if gt_path is not None else "",
            "target_active_ratio": float(np.mean(target_mask)),
        }
        if gt is not None:
            row.update(
                {
                    "r00_psnr_gt": _psnr(r00, gt),
                    "r10_psnr_gt": _psnr(r10, gt),
                    "r01_psnr_gt": _psnr(r01, gt),
                    "r11_psnr_gt": _psnr(r11, gt),
                    "r10_psnr_delta": _psnr(r10, gt) - _psnr(r00, gt),
                    "r01_psnr_delta": _psnr(r01, gt) - _psnr(r00, gt),
                    "r11_psnr_delta": _psnr(r11, gt) - _psnr(r00, gt),
                }
            )
        for name, delta in deltas.items():
            row.update(
                _delta_stats(
                    name,
                    delta,
                    base=r00,
                    target=target,
                    target_mask=target_mask,
                    change_threshold=float(args.change_threshold),
                    highpass_kernel=int(args.highpass_kernel),
                    lowpass_kernel=int(args.lowpass_kernel),
                    target_threshold=float(args.target_threshold),
                    norm_percentile=float(args.norm_percentile),
                )
            )
            _save_rgb(vis_dirs[name] / f"{stem}.png", np.abs(delta) * float(args.vis_scale))
        rows.append(row)

    if not rows:
        raise RuntimeError("No common render frames were found for causal evaluation.")
    mean_keys = sorted(k for k in rows[0].keys() if k not in {"stem", "r00", "r10", "r01", "r11", "gt"})
    summary = {
        "version": "evaluate_2dgs_handoff_causal_v0",
        "r00_dir": str(dirs["r00"]),
        "r10_dir": str(dirs["r10"]),
        "r01_dir": str(dirs["r01"]),
        "r11_dir": str(dirs["r11"]),
        "gt_dir": str(gt_dir) if gt_dir is not None else "",
        "output_dir": str(output_dir),
        "num_frames": len(rows),
        "vis_scale": float(args.vis_scale),
        "change_threshold": float(args.change_threshold),
        "highpass_kernel": int(args.highpass_kernel),
        "lowpass_kernel": int(args.lowpass_kernel),
        "target_threshold": float(args.target_threshold),
        "means": {key: _mean(rows, key) for key in mean_keys},
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
