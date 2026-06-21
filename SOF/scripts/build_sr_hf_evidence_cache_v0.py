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
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the cache builder usable in minimal Python envs.
    def tqdm(iterable, **_: object):
        return iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SR high-frequency evidence cache. The cache separates SR-added HF into "
            "geometry-like edge evidence, surface texture evidence, and likely hallucination/noise."
        )
    )
    parser.add_argument("--sr_dir", required=True)
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--tensor_radius", type=int, default=4)
    parser.add_argument("--mask_power", type=float, default=1.5)
    parser.add_argument("--hf_percentile", type=float, default=82.0)
    parser.add_argument("--edge_percentile", type=float, default=88.0)
    parser.add_argument("--flat_percentile", type=float, default=45.0)
    parser.add_argument("--geometry_score_threshold", type=float, default=0.18)
    parser.add_argument("--texture_score_threshold", type=float, default=0.15)
    parser.add_argument("--noise_score_threshold", type=float, default=0.12)
    parser.add_argument("--texture_coherence_min", type=float, default=0.35)
    parser.add_argument("--noise_coherence_max", type=float, default=0.28)
    parser.add_argument("--geometry_texture_suppression", type=float, default=0.50)
    parser.add_argument("--carrier_texture_weight", type=float, default=0.35)
    parser.add_argument("--carrier_noise_weight", type=float, default=0.0)
    parser.add_argument("--max_primitives_per_frame", type=int, default=32768)
    parser.add_argument("--primitive_nms_radius_px", type=int, default=2)
    parser.add_argument("--sigma_long_px", type=float, default=2.4)
    parser.add_argument("--sigma_short_px", type=float, default=0.35)
    parser.add_argument("--vis_clip", type=float, default=0.10)
    parser.add_argument("--debug_limit", type=int, default=12)
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


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


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


def _box_sum(gray: np.ndarray, radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r <= 0:
        return gray.astype(np.float32, copy=True)
    k = 2 * r + 1
    padded = np.pad(gray.astype(np.float32), ((r, r), (r, r)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]).astype(np.float32)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb.astype(np.float32) * LUMA[None, None, :], axis=2)


def _grad(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = gray.astype(np.float32, copy=False)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gy[0, :] = gray[1, :] - gray[0]
    gy[-1, :] = gray[-1, :] - gray[-2]
    return gx, gy


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx, gy = _grad(gray)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _structure_tensor(gray: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    gx, gy = _grad(gray)
    jxx = _box_sum(gx * gx, radius)
    jyy = _box_sum(gy * gy, radius)
    jxy = _box_sum(gx * gy, radius)
    grad_theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    tangent = (grad_theta + np.float32(math.pi * 0.5)).astype(np.float32)
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(trace, 1e-8)
    return tangent, np.clip(coherence, 0.0, 1.0).astype(np.float32)


def _norm_by_percentile(value: np.ndarray, percentile: float, floor: float = 1e-6) -> np.ndarray:
    scale = max(float(np.percentile(value, float(percentile))), floor)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def _abs_rgb_mean(value: np.ndarray) -> np.ndarray:
    return np.abs(value).mean(axis=2).astype(np.float32)


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    denom = float(np.sum(weight))
    if denom <= 1e-8:
        return float("nan")
    return float(np.sum(value * weight) / denom)


def _select_primitives(
    score: np.ndarray,
    tangent: np.ndarray,
    color: np.ndarray,
    kind_id: int,
    count: int,
    nms_radius: int,
    sigma_long: float,
    sigma_short: float,
) -> Dict[str, np.ndarray]:
    h, w = score.shape
    flat = score.reshape(-1)
    valid = np.flatnonzero(flat > 1e-6)
    if valid.size == 0 or int(count) <= 0:
        return {
            "xy": np.zeros((0, 2), dtype=np.float32),
            "theta": np.zeros((0,), dtype=np.float32),
            "sigma_long": np.zeros((0,), dtype=np.float32),
            "sigma_short": np.zeros((0,), dtype=np.float32),
            "color": np.zeros((0, 3), dtype=np.float32),
            "score": np.zeros((0,), dtype=np.float32),
            "kind": np.zeros((0,), dtype=np.int32),
        }
    topk = min(max(int(count) * 16, int(count)), valid.size)
    cand = valid[np.argpartition(flat[valid], -topk)[-topk:]]
    cand = cand[np.argsort(flat[cand])[::-1]]
    suppressed = np.zeros((h, w), dtype=bool)
    r = max(0, int(nms_radius))
    xy: List[Tuple[float, float]] = []
    theta: List[float] = []
    scores: List[float] = []
    colors: List[np.ndarray] = []
    for idx in cand.tolist():
        y = int(idx // w)
        x = int(idx - y * w)
        if suppressed[y, x]:
            continue
        xy.append((float(x), float(y)))
        theta.append(float(tangent[y, x]))
        scores.append(float(score[y, x]))
        colors.append(color[y, x, :].astype(np.float32))
        if len(xy) >= int(count):
            break
        if r > 0:
            suppressed[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)] = True
        else:
            suppressed[y, x] = True
    n = len(xy)
    return {
        "xy": np.asarray(xy, dtype=np.float32),
        "theta": np.asarray(theta, dtype=np.float32),
        "sigma_long": np.full((n,), float(sigma_long), dtype=np.float32),
        "sigma_short": np.full((n,), float(sigma_short), dtype=np.float32),
        "color": np.asarray(colors, dtype=np.float32).reshape(n, 3),
        "score": np.asarray(scores, dtype=np.float32),
        "kind": np.full((n,), int(kind_id), dtype=np.int32),
    }


def _concat_primitives(items: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ["xy", "theta", "sigma_long", "sigma_short", "color", "score", "kind"]
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [item[key] for item in items if item[key].shape[0] > 0]
        if arrays:
            out[key] = np.concatenate(arrays, axis=0)
        else:
            shape = (0, 2) if key == "xy" else (0, 3) if key == "color" else (0,)
            dtype = np.int32 if key == "kind" else np.float32
            out[key] = np.zeros(shape, dtype=dtype)
    return out


def _draw_primitives(primitives: Dict[str, np.ndarray], h: int, w: int, max_draw: int = 4096) -> np.ndarray:
    image = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    xy = primitives["xy"]
    score = primitives["score"]
    order = np.argsort(score)[::-1][: min(int(max_draw), int(xy.shape[0]))]
    colors_by_kind = {
        1: (255, 70, 30),
        2: (255, 210, 20),
        3: (70, 130, 255),
    }
    for i in order.tolist():
        x, y = float(xy[i, 0]), float(xy[i, 1])
        theta = float(primitives["theta"][i])
        length = float(primitives["sigma_long"][i]) * 2.5
        dx = math.cos(theta) * length
        dy = math.sin(theta) * length
        rgb = colors_by_kind.get(int(primitives["kind"][i]), (255, 255, 255))
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=rgb, width=1)
    return np.asarray(image, dtype=np.float32) / 255.0


def _write_sheet(
    path: Path,
    sr: np.ndarray,
    lr: np.ndarray,
    evidence_type: np.ndarray,
    hf_abs: np.ndarray,
    geometry_score: np.ndarray,
    texture_score: np.ndarray,
    noise_score: np.ndarray,
    primitive_overlay: np.ndarray,
) -> None:
    panels = [
        ("LR", lr),
        ("SR", sr),
        ("type R=edge Y=texture B=noise", evidence_type),
        ("HF evidence", np.repeat(np.clip(hf_abs / 0.15, 0.0, 1.0)[..., None], 3, axis=2)),
        ("geometry score", np.repeat(np.clip(geometry_score, 0.0, 1.0)[..., None], 3, axis=2)),
        ("texture score", np.repeat(np.clip(texture_score, 0.0, 1.0)[..., None], 3, axis=2)),
        ("noise score", np.repeat(np.clip(noise_score, 0.0, 1.0)[..., None], 3, axis=2)),
        ("primitive overlay", primitive_overlay),
    ]
    h, w = sr.shape[:2]
    cols = 4
    label_h = 24
    rows = int(math.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (w * cols, (h + label_h) * rows), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (label, arr) in enumerate(panels):
        x = (i % cols) * w
        y = (i // cols) * (h + label_h)
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        sheet.paste(img, (x, y + label_h))
        draw.text((x + 6, y + 5), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _stats(prefix: str, value: np.ndarray, weight: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mean": _weighted_mean(value, weight),
        f"{prefix}_ratio": _weighted_mean((value > 1e-6).astype(np.float32), weight),
    }


def _mean(rows: Sequence[Dict[str, float]], key: str) -> float:
    vals = np.asarray([float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))], dtype=np.float64)
    return float(vals.mean()) if vals.size > 0 else float("nan")


def main() -> None:
    args = _parse_args()
    sr_dir = Path(args.sr_dir).expanduser().resolve()
    lr_dir = Path(args.lr_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve() if str(args.mask_dir) else None
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output root is not empty; use --overwrite: {output_root}")
    if output_root.exists() and bool(args.overwrite):
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dirs = {
        "geometry_weight": output_root / "geometry_weight",
        "texture_weight": output_root / "texture_weight",
        "noise_weight": output_root / "noise_weight",
        "structure_weight": output_root / "structure_weight",
        "geometry_carrier_rgb": output_root / "geometry_carrier_rgb",
        "texture_carrier_rgb": output_root / "texture_carrier_rgb",
        "structure_carrier_rgb": output_root / "structure_carrier_rgb",
        "evidence_type": output_root / "evidence_type",
        "hf_abs": output_root / "hf_abs",
        "coherence": output_root / "coherence",
        "edge_gain": output_root / "edge_gain",
        "primitive_overlay": output_root / "primitive_overlay",
        "primitives": output_root / "primitives",
        "sheet": output_root / "sheet",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    sr_paths = _list_images(sr_dir)
    lr_paths = _list_images(lr_dir)
    mask_paths = _list_images(mask_dir) if mask_dir is not None and mask_dir.is_dir() else []
    if int(args.limit) > 0:
        sr_paths = sr_paths[: int(args.limit)]
    lr_lookup = _lookup(lr_paths)
    mask_lookup = _lookup(mask_paths)

    rows: List[Dict[str, float]] = []
    frames: List[Dict[str, object]] = []
    for index, sr_path in enumerate(tqdm(sr_paths, desc="SR HF evidence")):
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

        sr_hf = sr - _box_blur_rgb(sr, int(args.highpass_kernel))
        lr_hf = lr - _box_blur_rgb(lr, int(args.highpass_kernel))
        hf_delta = sr_hf - lr_hf
        hf_abs = _abs_rgb_mean(hf_delta)
        tangent, coherence = _structure_tensor(hf_abs, int(args.tensor_radius))
        sr_edge = _grad_mag(_luma(sr))
        lr_edge = _grad_mag(_luma(lr))
        edge_gain = np.maximum(sr_edge - lr_edge, 0.0).astype(np.float32)

        hf_n = _norm_by_percentile(hf_abs * (0.25 + 0.75 * trust), float(args.hf_percentile))
        sr_edge_n = _norm_by_percentile(sr_edge, float(args.edge_percentile))
        edge_gain_n = _norm_by_percentile(edge_gain, float(args.edge_percentile), floor=1e-5)
        flat_thr = float(np.percentile(lr_edge, float(args.flat_percentile)))
        flat = (lr_edge <= flat_thr).astype(np.float32)
        nonflat = 1.0 - flat

        geometry_score = (
            hf_n
            * np.maximum(sr_edge_n, edge_gain_n)
            * (0.35 + 0.65 * coherence)
            * (0.35 + 0.65 * trust)
        ).astype(np.float32)
        texture_score = (
            hf_n
            * flat
            * (0.20 + 0.80 * coherence)
            * (0.25 + 0.75 * trust)
            * (1.0 - float(args.geometry_texture_suppression) * np.clip(geometry_score, 0.0, 1.0))
        ).astype(np.float32)
        noise_score = (
            hf_n
            * (1.0 - np.clip(geometry_score, 0.0, 1.0))
            * (1.0 - np.clip(texture_score, 0.0, 1.0))
            * np.maximum((1.0 - coherence), (1.0 - trust))
        ).astype(np.float32)

        texture_gate = coherence >= float(args.texture_coherence_min)
        noise_gate = (coherence <= float(args.noise_coherence_max)) | (trust <= 0.25)
        geometry_weight = np.where(geometry_score >= float(args.geometry_score_threshold), geometry_score, 0.0).astype(np.float32)
        texture_weight = np.where(
            (texture_score >= float(args.texture_score_threshold)) & texture_gate,
            texture_score,
            0.0,
        ).astype(np.float32)
        noise_weight = np.where(
            (noise_score >= float(args.noise_score_threshold)) & noise_gate,
            noise_score,
            0.0,
        ).astype(np.float32)
        texture_weight = texture_weight * (geometry_weight <= 1e-6)
        noise_weight = noise_weight * (geometry_weight <= 1e-6) * (texture_weight <= 1e-6)
        structure_weight = np.clip(
            geometry_weight
            + float(args.carrier_texture_weight) * texture_weight
            + float(args.carrier_noise_weight) * noise_weight,
            0.0,
            1.0,
        ).astype(np.float32)

        geometry_carrier = sr * geometry_weight[..., None]
        texture_carrier = sr * texture_weight[..., None]
        structure_carrier = sr * structure_weight[..., None]
        evidence_type = np.zeros((*sr.shape[:2], 3), dtype=np.float32)
        evidence_type[..., 0] = np.maximum(geometry_weight, texture_weight)
        evidence_type[..., 1] = texture_weight
        evidence_type[..., 2] = noise_weight

        primitive_budget = max(0, int(args.max_primitives_per_frame))
        geom_count = int(round(primitive_budget * 0.70))
        tex_count = int(round(primitive_budget * 0.25))
        noise_count = max(0, primitive_budget - geom_count - tex_count)
        primitives = _concat_primitives(
            [
                _select_primitives(
                    geometry_weight,
                    tangent,
                    sr,
                    1,
                    geom_count,
                    int(args.primitive_nms_radius_px),
                    float(args.sigma_long_px),
                    float(args.sigma_short_px),
                ),
                _select_primitives(
                    texture_weight,
                    tangent,
                    sr,
                    2,
                    tex_count,
                    int(args.primitive_nms_radius_px),
                    float(args.sigma_long_px) * 0.75,
                    float(args.sigma_short_px) * 0.9,
                ),
                _select_primitives(
                    noise_weight,
                    tangent,
                    sr,
                    3,
                    noise_count,
                    int(args.primitive_nms_radius_px),
                    float(args.sigma_long_px) * 0.45,
                    float(args.sigma_short_px),
                ),
            ]
        )
        primitive_overlay = _draw_primitives(primitives, sr.shape[0], sr.shape[1])
        stem = sr_path.stem
        _save_gray(dirs["geometry_weight"] / f"{stem}.png", geometry_weight)
        _save_gray(dirs["texture_weight"] / f"{stem}.png", texture_weight)
        _save_gray(dirs["noise_weight"] / f"{stem}.png", noise_weight)
        _save_gray(dirs["structure_weight"] / f"{stem}.png", structure_weight)
        _save_rgb(dirs["geometry_carrier_rgb"] / f"{stem}.png", geometry_carrier)
        _save_rgb(dirs["texture_carrier_rgb"] / f"{stem}.png", texture_carrier)
        _save_rgb(dirs["structure_carrier_rgb"] / f"{stem}.png", structure_carrier)
        _save_rgb(dirs["evidence_type"] / f"{stem}.png", evidence_type)
        _save_gray(dirs["hf_abs"] / f"{stem}.png", np.clip(hf_abs / max(float(args.vis_clip), 1e-8), 0.0, 1.0))
        _save_gray(dirs["coherence"] / f"{stem}.png", coherence)
        _save_gray(dirs["edge_gain"] / f"{stem}.png", np.clip(edge_gain / 0.03, 0.0, 1.0))
        _save_rgb(dirs["primitive_overlay"] / f"{stem}.png", primitive_overlay)
        np.savez_compressed(dirs["primitives"] / f"{stem}.npz", **primitives)
        if index < int(args.debug_limit):
            _write_sheet(
                dirs["sheet"] / f"{stem}.png",
                sr,
                lr,
                evidence_type,
                hf_abs,
                geometry_score,
                texture_score,
                noise_score,
                primitive_overlay,
            )

        row = {
            "index": float(index),
            "mask_mean": float(trust.mean()),
            "hf_abs_mean": _weighted_mean(hf_abs, trust),
            "coherence_mean": _weighted_mean(coherence, trust),
            "edge_gain_mean": _weighted_mean(edge_gain, trust),
            "geometry_ratio": _weighted_mean((geometry_weight > 0).astype(np.float32), trust),
            "texture_ratio": _weighted_mean((texture_weight > 0).astype(np.float32), trust),
            "noise_ratio": _weighted_mean((noise_weight > 0).astype(np.float32), trust),
            "geometry_weight_mean": float(geometry_weight.mean()),
            "texture_weight_mean": float(texture_weight.mean()),
            "noise_weight_mean": float(noise_weight.mean()),
            "structure_weight_mean": float(structure_weight.mean()),
            "num_primitives": float(primitives["xy"].shape[0]),
            "num_geometry_primitives": float(np.sum(primitives["kind"] == 1)),
            "num_texture_primitives": float(np.sum(primitives["kind"] == 2)),
            "num_noise_primitives": float(np.sum(primitives["kind"] == 3)),
            "flat_hf_energy": _weighted_mean(hf_abs, trust * flat),
            "nonflat_hf_energy": _weighted_mean(hf_abs, trust * nonflat),
        }
        row["flat_over_nonflat_hf_ratio"] = row["flat_hf_energy"] / max(row["nonflat_hf_energy"], 1e-8)
        rows.append(row)
        frames.append(
            {
                "stem": stem,
                "sr": str(sr_path),
                "lr": str(lr_path),
                "mask": str(mask_path) if mask_path is not None else None,
                "metrics": row,
            }
        )

    summary = {
        "version": "build_sr_hf_evidence_cache_v0",
        "sr_dir": str(sr_dir),
        "lr_dir": str(lr_dir),
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "output_root": str(output_root),
        "match_policy": str(args.match_policy),
        "num_frames": len(rows),
        "highpass_kernel": int(args.highpass_kernel),
        "geometry_score_threshold": float(args.geometry_score_threshold),
        "texture_score_threshold": float(args.texture_score_threshold),
        "noise_score_threshold": float(args.noise_score_threshold),
        "means": {key: _mean(rows, key) for key in rows[0].keys()} if rows else {},
        "frames": frames,
        "outputs": {key: str(value) for key, value in dirs.items()},
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sr-hf-evidence-v0] frames: {len(rows)}")
    print(json.dumps({k: v for k, v in summary.items() if k not in {'frames', 'outputs'}}, indent=2, ensure_ascii=False))
    print(f"[sr-hf-evidence-v0] manifest: {output_root / 'manifest.json'}")
    print(f"[sr-hf-evidence-v0] inspect : {dirs['sheet']}")


if __name__ == "__main__":
    main()
