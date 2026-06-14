#!/usr/bin/env python3

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def normalize_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"\s*\(\d+\)$", "", stem)


def load_rgb_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def resize_like(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    pil = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    pil = pil.resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def choose_candidate_map(input_dir: Path, ignore_suffix: str, skip_prefix: str):
    candidate_map = {}
    for path in sorted(input_dir.glob("*.png")):
        stem = path.stem
        if ignore_suffix and stem.endswith(ignore_suffix):
            continue
        if skip_prefix and stem.startswith(skip_prefix):
            continue
        key = normalize_stem(path.name)
        existing = candidate_map.get(key)
        if existing is None or len(path.name) < len(existing.name):
            candidate_map[key] = path
    return candidate_map


def save_gray(path: Path, value: np.ndarray):
    value = np.asarray(value, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = value - float(value.min())
    vmax = float(value.max())
    if vmax > 1e-8:
        value = value / vmax
    Image.fromarray(np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def viridis_like(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    g = np.clip(1.7 * np.sin(np.pi * x), 0.0, 1.0)
    b = np.clip(1.3 * (1.0 - x) + 0.15, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_heat_overlay(path: Path, base_image: np.ndarray, heat: np.ndarray):
    base = np.clip(base_image, 0.0, 1.0)
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = heat - float(heat.min())
    vmax = float(heat.max())
    if vmax > 1e-8:
        heat = heat / vmax
    color = viridis_like(heat)
    alpha = 0.68 * heat[..., None]
    out = base * (1.0 - alpha) + color * alpha
    Image.fromarray(np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path)


def summarize(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def compute_local_psnr(reference_image: np.ndarray, candidate_image: np.ndarray, window_sigma: float, max_psnr: float):
    sq_error = np.mean((reference_image - candidate_image) ** 2, axis=2)
    local_mse = ndi.gaussian_filter(sq_error, sigma=float(window_sigma), mode="reflect")
    psnr_map = -10.0 * np.log10(np.maximum(local_mse, 1e-10))
    badness = np.clip((float(max_psnr) - psnr_map) / max(float(max_psnr), 1e-6), 0.0, 1.0)
    abs_error = np.mean(np.abs(reference_image - candidate_image), axis=2)
    global_mse = float(np.mean(sq_error))
    global_psnr = float(-10.0 * np.log10(max(global_mse, 1e-10)))
    return {
        "sq_error": sq_error,
        "local_mse": local_mse,
        "psnr_map": psnr_map,
        "badness": badness,
        "abs_error": abs_error,
        "global_psnr": global_psnr,
    }


def main():
    parser = ArgumentParser(description="Generate local-PSNR heatmaps from paired reference/candidate images.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ignore_candidate_suffix", type=str, default="_inject")
    parser.add_argument("--skip_candidate_prefix", type=str, default="000")
    parser.add_argument("--window_sigma", type=float, default=6.0)
    parser.add_argument("--max_psnr", type=float, default=40.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_paths = sorted(
        list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.JPEG")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    )
    candidate_map = choose_candidate_map(
        input_dir=input_dir,
        ignore_suffix=str(args.ignore_candidate_suffix),
        skip_prefix=str(args.skip_candidate_prefix),
    )

    summaries = []
    missing_candidates = []
    for reference_path in reference_paths:
        image_name = normalize_stem(reference_path.name)
        candidate_path = candidate_map.get(image_name)
        if candidate_path is None:
            missing_candidates.append(image_name)
            continue

        reference_image = load_rgb_np(reference_path)
        candidate_image = resize_like(load_rgb_np(candidate_path), reference_image.shape[0], reference_image.shape[1])
        maps = compute_local_psnr(
            reference_image=reference_image,
            candidate_image=candidate_image,
            window_sigma=float(args.window_sigma),
            max_psnr=float(args.max_psnr),
        )

        view_dir = output_dir / image_name
        view_dir.mkdir(parents=True, exist_ok=True)
        save_gray(view_dir / "abs_error.png", maps["abs_error"])
        save_gray(view_dir / "local_mse.png", maps["local_mse"])
        save_gray(view_dir / "local_psnr.png", maps["psnr_map"])
        save_gray(view_dir / "badness.png", maps["badness"])
        save_heat_overlay(view_dir / "psnr_heat_overlay.png", reference_image, maps["badness"])

        summaries.append(
            {
                "image_name": image_name,
                "reference_path": str(reference_path.resolve()),
                "candidate_path": str(candidate_path.resolve()),
                "global_psnr": float(maps["global_psnr"]),
                "local_psnr_stats": summarize(maps["psnr_map"]),
                "badness_stats": summarize(maps["badness"]),
            }
        )

    summary = {
        "mode": "diagnose_local_psnr_heatmaps_v0",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "view_count": int(len(summaries)),
        "missing_candidates": missing_candidates,
        "parameters": {
            "ignore_candidate_suffix": str(args.ignore_candidate_suffix),
            "skip_candidate_prefix": str(args.skip_candidate_prefix),
            "window_sigma": float(args.window_sigma),
            "max_psnr": float(args.max_psnr),
        },
        "views": summaries,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
