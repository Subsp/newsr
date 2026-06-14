from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_images_natural(image_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths, key=lambda p: _natural_key(p.name))


def load_rgb01(path: str | os.PathLike[str]) -> np.ndarray:
    with Image.open(path).convert("RGB") as img:
        return np.asarray(img, dtype=np.float32) / 255.0


def save_rgb01(path: str | os.PathLike[str], image: np.ndarray) -> None:
    arr = np.clip(image, 0.0, 1.0)
    img = Image.fromarray(np.round(arr * 255.0).astype(np.uint8))
    img.save(path)


def resize_rgb01(image: np.ndarray, size_hw: tuple[int, int], resample: int) -> np.ndarray:
    h, w = size_hw
    arr = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        img = Image.fromarray(arr[..., 0], mode="L")
        img = img.resize((w, h), resample=resample)
        out = np.asarray(img, dtype=np.float32) / 255.0
        return out[..., None]
    img = Image.fromarray(arr)
    img = img.resize((w, h), resample=resample)
    out = np.asarray(img, dtype=np.float32) / 255.0
    if out.ndim == 2:
        out = out[..., None]
    return out


def _pad_to_even(channel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = channel.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return channel, (0, 0)
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (pad_h, pad_w)


def haar_dwt2(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    x, pad = _pad_to_even(channel)
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh, pad


def haar_idwt2(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    pad: tuple[int, int],
) -> np.ndarray:
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    h, w = ll.shape
    out = np.empty((h * 2, w * 2), dtype=np.float32)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    pad_h, pad_w = pad
    if pad_h:
        out = out[:-pad_h, :]
    if pad_w:
        out = out[:, :-pad_w]
    return out


def dwt_multilevel(image: np.ndarray, levels: int) -> tuple[np.ndarray, list[dict[str, np.ndarray]], list[tuple[int, int]]]:
    low = image.astype(np.float32)
    coeffs: list[dict[str, np.ndarray]] = []
    pads: list[tuple[int, int]] = []
    for _ in range(levels):
        ll_channels = []
        lh_channels = []
        hl_channels = []
        hh_channels = []
        level_pads = []
        for ch in range(low.shape[2]):
            ll, lh, hl, hh, pad = haar_dwt2(low[..., ch])
            ll_channels.append(ll)
            lh_channels.append(lh)
            hl_channels.append(hl)
            hh_channels.append(hh)
            level_pads.append(pad)
        low = np.stack(ll_channels, axis=-1)
        coeffs.append(
            {
                "lh": np.stack(lh_channels, axis=-1),
                "hl": np.stack(hl_channels, axis=-1),
                "hh": np.stack(hh_channels, axis=-1),
            }
        )
        pads.append(level_pads[0])
    return low, coeffs, pads


def idwt_multilevel(
    low: np.ndarray,
    coeffs: list[dict[str, np.ndarray]],
    pads: list[tuple[int, int]],
) -> np.ndarray:
    current = low.astype(np.float32)
    for level in reversed(range(len(coeffs))):
        restored_channels = []
        pad = pads[level]
        for ch in range(current.shape[2]):
            restored = haar_idwt2(
                current[..., ch],
                coeffs[level]["lh"][..., ch],
                coeffs[level]["hl"][..., ch],
                coeffs[level]["hh"][..., ch],
                pad,
            )
            restored_channels.append(restored)
        current = np.stack(restored_channels, axis=-1)
    return current


def downsample_mean(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    return resize_rgb01(
        image,
        size_hw,
        resample=Image.Resampling.BOX if hasattr(Image, "Resampling") else Image.BOX,
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    try:
        import cv2
    except Exception:
        return float("nan")
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    denom = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / denom
    return float(np.mean(ssim_map))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    vals = [_ssim_channel(a[..., c], b[..., c]) for c in range(3)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def safe_nanmean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class FusionConfig:
    levels: int = 2
    hf_weight: float = 0.75
    delta_clip: float = 0.10
    consistency_tau: float = 0.08
    energy_floor: float = 0.002
    gain_power: float = 1.0
    scale: int = 4
    write_debug: bool = False


def match_frames(input_paths: list[Path], prior_paths: list[Path], gt_paths: list[Path] | None) -> list[dict[str, Path | None]]:
    input_by_stem = {p.stem: p for p in input_paths}
    prior_by_stem = {p.stem: p for p in prior_paths}
    gt_by_stem = {p.stem: p for p in gt_paths} if gt_paths else {}

    common_stems = [p.stem for p in input_paths if p.stem in prior_by_stem]
    if common_stems:
        records = []
        for stem in common_stems:
            records.append(
                {
                    "stem": stem,
                    "input": input_by_stem[stem],
                    "prior": prior_by_stem[stem],
                    "gt": gt_by_stem.get(stem),
                }
            )
        return records

    if len(input_paths) != len(prior_paths):
        raise RuntimeError(
            "No stem overlap between input and prior, and file counts differ. "
            f"input={len(input_paths)} prior={len(prior_paths)}"
        )

    records = []
    for input_path, prior_path in zip(input_paths, prior_paths):
        stem = input_path.stem
        records.append(
            {
                "stem": stem,
                "input": input_path,
                "prior": prior_path,
                "gt": gt_by_stem.get(stem),
            }
        )
    return records


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_symlink_dir(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src, target_is_directory=True)
        return "symlink"
    except OSError:
        shutil.copytree(src, dst)
        return "copy"


def _prepare_target_size(lr_path: Path, gt_path: Path | None, scale: int) -> tuple[int, int]:
    if gt_path is not None and gt_path.is_file():
        with Image.open(gt_path) as img:
            w, h = img.size
        return (h, w)
    with Image.open(lr_path) as img:
        w, h = img.size
    return (h * scale, w * scale)


def fuse_frame(
    lr_image: np.ndarray,
    prior_image: np.ndarray,
    target_hw: tuple[int, int],
    cfg: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    bicubic = resize_rgb01(
        lr_image,
        target_hw,
        resample=Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC,
    )
    prior = resize_rgb01(
        prior_image,
        target_hw,
        resample=Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC,
    )

    low_base, base_coeffs, base_pads = dwt_multilevel(bicubic, cfg.levels)
    _, prior_coeffs, _ = dwt_multilevel(prior, cfg.levels)

    full_consistency = np.exp(
        -np.mean(np.abs(prior - bicubic), axis=-1, keepdims=True) / max(cfg.consistency_tau, 1e-6)
    )
    fused_coeffs: list[dict[str, np.ndarray]] = []
    debug: dict[str, np.ndarray] = {}
    level_gate_means = []

    for level_idx, (base_level, prior_level) in enumerate(zip(base_coeffs, prior_coeffs)):
        band_shape = base_level["lh"].shape[:2]
        gate_level = downsample_mean(full_consistency, band_shape)
        fused_level: dict[str, np.ndarray] = {}
        band_gates = []
        for band_name in ("lh", "hl", "hh"):
            base_band = base_level[band_name]
            prior_band = prior_level[band_name]
            delta = np.clip(prior_band - base_band, -cfg.delta_clip, cfg.delta_clip)
            base_energy = np.sqrt(np.mean(base_band * base_band, axis=-1, keepdims=True))
            prior_energy = np.sqrt(np.mean(prior_band * prior_band, axis=-1, keepdims=True))
            energy_gain = np.clip(
                (prior_energy - base_energy) / (prior_energy + cfg.energy_floor),
                0.0,
                1.0,
            )
            if cfg.gain_power != 1.0:
                energy_gain = np.power(energy_gain, cfg.gain_power)
            band_gate = gate_level * energy_gain
            fused_band = base_band + cfg.hf_weight * band_gate * delta
            fused_level[band_name] = fused_band.astype(np.float32)
            band_gates.append(band_gate)
            if cfg.write_debug and level_idx == 0:
                debug[f"{band_name}_gate"] = np.repeat(band_gate, 3, axis=-1)
        fused_coeffs.append(fused_level)
        level_gate_means.append(float(np.mean(np.stack(band_gates, axis=0))))

    fused = idwt_multilevel(low_base, fused_coeffs, base_pads)
    fused = np.clip(fused, 0.0, 1.0)

    metrics = {
        "prior_bicubic_l1": float(np.mean(np.abs(prior - bicubic))),
        "prior_bicubic_psnr": psnr(prior, bicubic),
        "fused_bicubic_l1": float(np.mean(np.abs(fused - bicubic))),
        "gate_mean": float(np.mean(level_gate_means)) if level_gate_means else 0.0,
        "gate_max": float(np.max(level_gate_means)) if level_gate_means else 0.0,
    }
    return fused, bicubic, metrics, debug


def run_pipeline(
    *,
    input_dir: str,
    prior_dir: str,
    output_dir: str,
    gt_dir: str | None,
    cfg: FusionConfig,
) -> dict[str, Any]:
    input_paths = list_images_natural(input_dir)
    prior_paths = list_images_natural(prior_dir)
    gt_paths = list_images_natural(gt_dir) if gt_dir else []
    records = match_frames(input_paths, prior_paths, gt_paths if gt_dir else None)

    out_root = Path(output_dir)
    renders_dir = out_root / "renders"
    bicubic_dir = out_root / "bicubic"
    debug_dir = out_root / "debug"
    gt_out_dir = out_root / "gt"
    _ensure_dir(renders_dir)
    _ensure_dir(bicubic_dir)
    if cfg.write_debug:
        _ensure_dir(debug_dir)

    symlink_mode = None
    if gt_dir:
        symlink_mode = _safe_symlink_dir(Path(gt_dir), gt_out_dir)

    frame_rows = []
    quick_metrics = []
    quick_metrics_bicubic = []

    for record in records:
        stem = str(record["stem"])
        input_path = Path(record["input"])
        prior_path = Path(record["prior"])
        gt_path = Path(record["gt"]) if record["gt"] is not None else None

        lr = load_rgb01(input_path)
        prior = load_rgb01(prior_path)
        target_hw = _prepare_target_size(input_path, gt_path, cfg.scale)
        fused, bicubic, fuse_metrics, debug_maps = fuse_frame(lr, prior, target_hw, cfg)

        ext = gt_path.suffix if gt_path is not None else ".png"
        render_path = renders_dir / f"{stem}{ext}"
        bicubic_path = bicubic_dir / f"{stem}{ext}"
        save_rgb01(render_path, fused)
        save_rgb01(bicubic_path, bicubic)

        if cfg.write_debug:
            for name, arr in debug_maps.items():
                save_rgb01(debug_dir / f"{stem}_{name}.png", arr)

        row = {
            "stem": stem,
            "input": str(input_path),
            "prior": str(prior_path),
            "gt": str(gt_path) if gt_path is not None else "",
            "render": str(render_path),
            "bicubic": str(bicubic_path),
            **fuse_metrics,
        }

        if gt_path is not None and gt_path.is_file():
            gt = load_rgb01(gt_path)
            fused_psnr = psnr(fused, gt)
            fused_ssim = ssim_rgb(fused, gt)
            fused_mae = mae(fused, gt)
            bicubic_psnr = psnr(bicubic, gt)
            bicubic_ssim = ssim_rgb(bicubic, gt)
            bicubic_mae = mae(bicubic, gt)
            row.update(
                {
                    "fused_psnr": fused_psnr,
                    "fused_ssim": fused_ssim,
                    "fused_mae": fused_mae,
                    "bicubic_psnr": bicubic_psnr,
                    "bicubic_ssim": bicubic_ssim,
                    "bicubic_mae": bicubic_mae,
                    "delta_psnr": fused_psnr - bicubic_psnr,
                    "delta_ssim": fused_ssim - bicubic_ssim
                    if not math.isnan(fused_ssim) and not math.isnan(bicubic_ssim)
                    else float("nan"),
                }
            )
            quick_metrics.append(
                {
                    "stem": stem,
                    "psnr": fused_psnr,
                    "ssim": fused_ssim,
                    "mae": fused_mae,
                }
            )
            quick_metrics_bicubic.append(
                {
                    "stem": stem,
                    "psnr": bicubic_psnr,
                    "ssim": bicubic_ssim,
                    "mae": bicubic_mae,
                }
            )
        frame_rows.append(row)

    summary: dict[str, Any] = {
        "num_frames": len(frame_rows),
        "input_dir": str(Path(input_dir).resolve()),
        "prior_dir": str(Path(prior_dir).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()) if gt_dir else "",
        "output_dir": str(out_root.resolve()),
        "gt_link_mode": symlink_mode,
        "config": asdict(cfg),
        "frames": frame_rows,
    }

    if quick_metrics:
        summary["quick_eval"] = {
            "fused": {
                "psnr": float(np.mean([m["psnr"] for m in quick_metrics])),
                "ssim": safe_nanmean([m["ssim"] for m in quick_metrics]),
                "mae": float(np.mean([m["mae"] for m in quick_metrics])),
            },
            "bicubic": {
                "psnr": float(np.mean([m["psnr"] for m in quick_metrics_bicubic])),
                "ssim": safe_nanmean([m["ssim"] for m in quick_metrics_bicubic]),
                "mae": float(np.mean([m["mae"] for m in quick_metrics_bicubic])),
            },
            "delta": {
                "psnr": float(
                    np.mean([f["psnr"] - b["psnr"] for f, b in zip(quick_metrics, quick_metrics_bicubic)])
                ),
                "ssim": safe_nanmean(
                    [f["ssim"] - b["ssim"] for f, b in zip(quick_metrics, quick_metrics_bicubic)]
                ),
                "mae": float(
                    np.mean([f["mae"] - b["mae"] for f, b in zip(quick_metrics, quick_metrics_bicubic)])
                ),
            },
        }
        with open(out_root / "quick_eval.json", "w", encoding="utf-8") as f:
            json.dump(summary["quick_eval"], f, indent=2, allow_nan=True)

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    return summary
