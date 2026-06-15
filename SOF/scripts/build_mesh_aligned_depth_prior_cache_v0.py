#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


SOF_ROOT = Path(__file__).resolve().parents[1]
if str(SOF_ROOT) not in sys.path:
    sys.path.insert(0, str(SOF_ROOT))

from prepare_mesh_depth_tsdf_regulation_v0 import (  # noqa: E402
    build_open3d_raycast_scene,
    render_mesh_depth_raycast,
    render_mesh_depth_sample_zbuffer,
)
from select_mesh_outside_gaussians_v0 import load_triangle_mesh  # noqa: E402
from train_mip_to_sof_surface_v0 import load_cameras_for_split  # noqa: E402
from utils.prior_injection import normalize_image_name  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ARRAY_EXTS = {".npy", ".npz", ".pt", ".pth"}
DEPTH_EXTS = IMAGE_EXTS | ARRAY_EXTS
DEFAULT_DEPTH_SUBDIRS = ("depth", "depth_hr", "pred", "prediction", "depths", "mono_depth", "")
DEFAULT_DEPTH_KEYS = ("depth", "depth_hr", "pred", "prediction", "arr_0")
RESAMPLING = getattr(Image, "Resampling", Image)


def _parse_csv(value: str | None, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    value = str(value).strip()
    if not value:
        return []
    if value.lower() == "auto":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align an external depth prior to the gs2mesh/COLMAP camera depth domain. "
            "The output aligned_depth/*.npz files expose a `depth` key and can be used "
            "directly as DEPTH_PRIOR_DIR for the N-PSE edge/trust cache."
        )
    )
    parser.add_argument("--scene_root", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--mesh_path", type=Path, required=True)
    parser.add_argument("--depth_prior_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=("train", "test", "both"), default="train")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N camera limit for smoke runs.")
    parser.add_argument("--depth_subdirs", default="auto")
    parser.add_argument("--depth_keys", default="auto")
    parser.add_argument("--recursive_depth_search", action="store_true")
    parser.add_argument("--mesh_depth_mode", choices=("raycast", "sample_zbuffer"), default="raycast")
    parser.add_argument("--raycast_downsample", type=int, default=1)
    parser.add_argument("--raycast_chunk", type=int, default=262144)
    parser.add_argument("--mesh_sample_points_per_face", type=int, choices=(1, 4), default=1)
    parser.add_argument("--mesh_sample_splat_kernel", type=int, default=5)
    parser.add_argument("--disable_front_facing_only", action="store_true")
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--align_min_pixels", type=int, default=2048)
    parser.add_argument(
        "--alignment_modes",
        default="affine,inverse_affine,negative_affine",
        help="Comma list from affine,inverse_affine,negative_affine,identity.",
    )
    parser.add_argument("--inverse_eps", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_debug_images", action="store_true")
    return parser.parse_args()


def _unique_names_for_view(image_name: str) -> list[str]:
    names = [
        str(image_name),
        normalize_image_name(str(image_name)),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _candidate_keys_for_file(path: Path) -> list[str]:
    keys = [path.stem, path.name]
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        lower = key.lower()
        if lower not in seen:
            out.append(lower)
            seen.add(lower)
    return out


def _iter_supported_files(base: Path, recursive: bool) -> list[Path]:
    if not base.is_dir():
        return []
    iterator = base.rglob("*") if recursive else base.iterdir()
    return [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in DEPTH_EXTS]


def _build_depth_index(root: Path, subdirs: Sequence[str], recursive: bool) -> tuple[dict[str, Path], dict[str, list[str]]]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    bases: list[Path] = []
    seen_bases: set[Path] = set()
    for subdir in subdirs:
        base = root / subdir if subdir else root
        try:
            resolved = base.resolve()
        except FileNotFoundError:
            resolved = base
        if resolved in seen_bases:
            continue
        bases.append(base)
        seen_bases.add(resolved)

    for base in bases:
        for path in _iter_supported_files(base, recursive=recursive):
            for key in _candidate_keys_for_file(path):
                if key in index and index[key] != path:
                    duplicates.setdefault(key, [str(index[key])]).append(str(path))
                    continue
                index[key] = path
    return index, duplicates


def _lookup_depth_path(index: dict[str, Path], image_name: str) -> Path | None:
    for name in _unique_names_for_view(image_name):
        path_name = Path(name)
        keys = [path_name.stem, path_name.name]
        for key in keys:
            found = index.get(key.lower())
            if found is not None:
                return found
    return None


def _load_npz_array(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    payload = np.load(str(path))
    if isinstance(payload, np.ndarray):
        return payload
    for key in preferred_keys:
        if key in payload:
            return np.asarray(payload[key])
    if "arr_0" in payload:
        return np.asarray(payload["arr_0"])
    keys = list(payload.keys())
    if not keys:
        raise ValueError(f"Empty npz depth prior: {path}")
    return np.asarray(payload[keys[0]])


def _load_depth_file(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(str(path))
    elif suffix == ".npz":
        array = _load_npz_array(path, preferred_keys)
    elif suffix in {".pt", ".pth"}:
        import torch

        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, dict):
            for key in preferred_keys:
                if key in payload:
                    payload = payload[key]
                    break
            else:
                payload = payload[next(iter(payload.keys()))]
        if torch.is_tensor(payload):
            array = payload.detach().cpu().numpy()
        else:
            array = np.asarray(payload)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., 0]
    return _canonicalize_depth_hw(array, path)


def _canonicalize_depth_hw(array: np.ndarray, path: Path) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    value = np.squeeze(value)
    if value.ndim == 3:
        if value.shape[0] in {1, 3}:
            value = value[0] if value.shape[0] == 1 else value.mean(axis=0)
        elif value.shape[-1] in {1, 3}:
            value = value[..., 0] if value.shape[-1] == 1 else value.mean(axis=-1)
        else:
            raise ValueError(f"Cannot canonicalize depth shape {value.shape} for {path}")
    if value.ndim != 2:
        raise ValueError(f"Expected HxW depth prior, got shape={value.shape} for {path}")
    return value.astype(np.float32, copy=False)


def _resize_float_hw(value: np.ndarray, target_hw: tuple[int, int], resample: int) -> np.ndarray:
    height, width = int(target_hw[0]), int(target_hw[1])
    value = np.asarray(value, dtype=np.float32)
    if value.shape == (height, width):
        return value.astype(np.float32, copy=True)
    finite = np.isfinite(value)
    fill = float(np.nanmedian(value[finite])) if np.any(finite) else 0.0
    safe = np.where(finite, value, fill).astype(np.float32, copy=False)
    with Image.fromarray(safe, mode="F") as image:
        resized = image.resize((width, height), resample=resample)
        return np.asarray(resized, dtype=np.float32)


def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        return {"count": int(x.size), "finite_count": 0, "mean": None, "median": None, "p90": None, "max": None}
    xf = x[finite]
    return {
        "count": int(x.size),
        "finite_count": int(xf.size),
        "mean": float(np.mean(xf)),
        "median": float(np.median(xf)),
        "p90": float(np.percentile(xf, 90.0)),
        "max": float(np.max(xf)),
    }


def _transform_depth(raw: np.ndarray, mode: str, inverse_eps: float) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    finite = np.isfinite(raw)
    if mode == "affine" or mode == "identity":
        source = raw.astype(np.float32, copy=True)
        valid = finite
    elif mode == "inverse_affine":
        valid = finite & (raw > float(inverse_eps))
        source = np.full_like(raw, np.nan, dtype=np.float32)
        source[valid] = 1.0 / np.clip(raw[valid], float(inverse_eps), None)
    elif mode == "negative_affine":
        source = -raw.astype(np.float32, copy=False)
        valid = finite
    else:
        raise ValueError(f"Unsupported alignment mode: {mode}")
    return source, valid


def _score_alignment(aligned: np.ndarray, reference: np.ndarray, valid: np.ndarray, depth_min: float) -> dict[str, float | int | None]:
    good = valid & np.isfinite(aligned) & np.isfinite(reference) & (reference > float(depth_min))
    pixels = int(np.count_nonzero(good))
    if pixels <= 0:
        return {"pixels": 0, "median_rel_error": None, "median_abs_error": None, "p90_rel_error": None}
    abs_error = np.abs(aligned[good] - reference[good]).astype(np.float32, copy=False)
    rel_error = abs_error / np.clip(reference[good], float(depth_min), None)
    return {
        "pixels": pixels,
        "median_rel_error": float(np.median(rel_error)),
        "median_abs_error": float(np.median(abs_error)),
        "p90_rel_error": float(np.percentile(rel_error, 90.0)),
    }


def _fit_mode(
    *,
    raw_depth: np.ndarray,
    mesh_depth: np.ndarray,
    mesh_valid: np.ndarray,
    mode: str,
    min_pixels: int,
    depth_min: float,
    inverse_eps: float,
) -> dict[str, object]:
    source, source_valid = _transform_depth(raw_depth, mode, inverse_eps)
    valid = mesh_valid & source_valid & np.isfinite(source) & np.isfinite(mesh_depth) & (mesh_depth > float(depth_min))
    pixels = int(np.count_nonzero(valid))
    if pixels < int(min_pixels):
        return {
            "status": "insufficient_pixels",
            "mode": mode,
            "pixels": pixels,
            "scale": 1.0,
            "shift": 0.0,
            "score": None,
            "aligned": source,
            "valid": valid,
        }

    if mode == "identity":
        scale = 1.0
        shift = 0.0
    else:
        source_values = source[valid].astype(np.float32, copy=False)
        ref_values = mesh_depth[valid].astype(np.float32, copy=False)
        s10, s50, s90 = np.percentile(source_values, [10.0, 50.0, 90.0]).astype(np.float32)
        r10, r50, r90 = np.percentile(ref_values, [10.0, 50.0, 90.0]).astype(np.float32)
        source_span = float(s90 - s10)
        ref_span = float(r90 - r10)
        if source_span <= 1e-8 or ref_span <= 1e-8:
            return {
                "status": "degenerate_percentiles",
                "mode": mode,
                "pixels": pixels,
                "scale": 1.0,
                "shift": 0.0,
                "score": None,
                "aligned": source,
                "valid": valid,
            }
        scale = ref_span / source_span
        shift = float(r50 - float(scale) * float(s50))

    aligned = source * float(scale) + float(shift)
    score = _score_alignment(aligned, mesh_depth, valid, depth_min)
    return {
        "status": "ok",
        "mode": mode,
        "pixels": pixels,
        "scale": float(scale),
        "shift": float(shift),
        "score": score,
        "aligned": aligned.astype(np.float32, copy=False),
        "valid": valid,
    }


def _choose_alignment(
    *,
    raw_depth: np.ndarray,
    mesh_depth: np.ndarray,
    mesh_valid: np.ndarray,
    modes: Sequence[str],
    min_pixels: int,
    depth_min: float,
    inverse_eps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for mode in modes:
        trial = _fit_mode(
            raw_depth=raw_depth,
            mesh_depth=mesh_depth,
            mesh_valid=mesh_valid,
            mode=mode,
            min_pixels=min_pixels,
            depth_min=depth_min,
            inverse_eps=inverse_eps,
        )
        trials.append({k: v for k, v in trial.items() if k not in {"aligned", "valid"}})
        score = trial.get("score")
        metric = score.get("median_rel_error") if isinstance(score, dict) else None
        if trial.get("status") == "ok" and metric is not None:
            if best is None:
                best = trial
            else:
                best_score = best.get("score")
                best_metric = best_score.get("median_rel_error") if isinstance(best_score, dict) else None
                if best_metric is None or float(metric) < float(best_metric):
                    best = trial

    if best is None:
        fallback_mode = "affine" if "affine" in modes else modes[0]
        source, source_valid = _transform_depth(raw_depth, fallback_mode, inverse_eps)
        valid = mesh_valid & source_valid & np.isfinite(source)
        aligned = source.astype(np.float32, copy=True)
        best_summary = {
            "status": "fallback_identity_no_valid_alignment",
            "mode": fallback_mode,
            "pixels": int(np.count_nonzero(valid)),
            "scale": 1.0,
            "shift": 0.0,
            "score": _score_alignment(aligned, mesh_depth, valid, depth_min),
        }
        return aligned, valid, best_summary, trials

    aligned = np.asarray(best["aligned"], dtype=np.float32)
    valid = np.asarray(best["valid"], dtype=bool)
    summary = {k: v for k, v in best.items() if k not in {"aligned", "valid"}}
    return aligned, valid, summary, trials


def _fill_invalid_depth(depth: np.ndarray, valid: np.ndarray, depth_min: float) -> tuple[np.ndarray, float]:
    good = valid & np.isfinite(depth) & (depth > float(depth_min))
    if np.any(good):
        fill = float(np.median(depth[good]))
    else:
        finite = np.isfinite(depth) & (depth > float(depth_min))
        fill = float(np.median(depth[finite])) if np.any(finite) else float(depth_min)
    out = np.where(good, depth, fill).astype(np.float32, copy=False)
    return out, fill


def _robust01(value: np.ndarray, lo: float = 2.0, hi: float = 98.0, invert: bool = False) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(x[finite], [lo, hi])
    if p_hi <= p_lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    out = (np.where(finite, x, p_lo) - p_lo) / (p_hi - p_lo)
    out = np.clip(out, 0.0, 1.0)
    return (1.0 - out if invert else out).astype(np.float32, copy=False)


def _gradient_magnitude(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    gx[:, 0] = x[:, 1] - x[:, 0]
    gx[:, -1] = x[:, -1] - x[:, -2]
    gy[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    gy[0, :] = x[1, :] - x[0, :]
    gy[-1, :] = x[-1, :] - x[-2, :]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)


def _atomic_save_png(array01: np.ndarray, path: Path, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.round(np.asarray(array01) * 255.0), 0.0, 255.0).astype(np.uint8)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        Image.fromarray(arr, mode=mode).save(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_save_npz(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _save_debug_images(
    *,
    stem: str,
    raw_depth: np.ndarray,
    mesh_depth: np.ndarray,
    aligned_depth: np.ndarray,
    aligned_filled: np.ndarray,
    valid_mask: np.ndarray,
    mesh_valid: np.ndarray,
    debug_dir: Path,
    depth_min: float,
) -> dict[str, str]:
    frame_dir = debug_dir / stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    raw_path = frame_dir / "raw_prior_depth.png"
    mesh_path = frame_dir / "mesh_depth.png"
    aligned_path = frame_dir / "aligned_depth.png"
    valid_path = frame_dir / "valid_mask.png"
    error_path = frame_dir / "mesh_aligned_rel_error.png"
    jump_path = frame_dir / "aligned_depth_jump.png"

    _atomic_save_png(_robust01(raw_depth, invert=True), raw_path)
    _atomic_save_png(_robust01(mesh_depth, invert=True), mesh_path)
    _atomic_save_png(_robust01(aligned_filled, invert=True), aligned_path)
    _atomic_save_png(valid_mask.astype(np.float32) * mesh_valid.astype(np.float32), valid_path)

    err_valid = valid_mask & mesh_valid & np.isfinite(aligned_depth) & np.isfinite(mesh_depth) & (mesh_depth > float(depth_min))
    rel_error = np.zeros_like(aligned_filled, dtype=np.float32)
    rel_error[err_valid] = np.abs(aligned_depth[err_valid] - mesh_depth[err_valid]) / np.clip(mesh_depth[err_valid], float(depth_min), None)
    _atomic_save_png(_robust01(rel_error, lo=0.0, hi=95.0), error_path)
    _atomic_save_png(_robust01(_gradient_magnitude(aligned_filled), lo=0.0, hi=98.0), jump_path)

    return {
        "raw_prior_depth": str(raw_path),
        "mesh_depth": str(mesh_path),
        "aligned_depth": str(aligned_path),
        "valid_mask": str(valid_path),
        "mesh_aligned_rel_error": str(error_path),
        "aligned_depth_jump": str(jump_path),
    }


def _process_frame(
    *,
    cam,
    mesh,
    scene,
    depth_index: dict[str, Path],
    depth_keys: Sequence[str],
    alignment_modes: Sequence[str],
    args: argparse.Namespace,
    dirs: dict[str, Path],
) -> dict[str, object]:
    stem = Path(str(cam.image_name)).stem
    aligned_path = dirs["aligned_depth"] / f"{stem}.npz"
    mesh_depth_path = dirs["mesh_depth"] / f"{stem}.npz"
    if aligned_path.exists() and mesh_depth_path.exists() and not bool(args.overwrite):
        return {
            "image_name": str(cam.image_name),
            "stem": stem,
            "status": "skipped_existing",
            "aligned_depth_path": str(aligned_path),
            "mesh_depth_path": str(mesh_depth_path),
        }

    prior_path = _lookup_depth_path(depth_index, str(cam.image_name))
    if prior_path is None:
        return {"image_name": str(cam.image_name), "stem": stem, "status": "missing_depth_prior"}

    if str(args.mesh_depth_mode) == "raycast":
        mesh_depth, face_ids = render_mesh_depth_raycast(
            scene,
            cam,
            downsample=int(args.raycast_downsample),
            ray_chunk=int(args.raycast_chunk),
        )
    else:
        mesh_depth, face_ids = render_mesh_depth_sample_zbuffer(mesh, cam, args)

    mesh_depth = np.asarray(mesh_depth, dtype=np.float32)
    mesh_valid = np.isfinite(mesh_depth) & (mesh_depth > float(args.depth_min))
    raw_depth = _load_depth_file(prior_path, depth_keys)
    raw_depth = _resize_float_hw(raw_depth, mesh_depth.shape, resample=RESAMPLING.BILINEAR)

    aligned, valid, align_summary, trials = _choose_alignment(
        raw_depth=raw_depth,
        mesh_depth=mesh_depth,
        mesh_valid=mesh_valid,
        modes=alignment_modes,
        min_pixels=int(args.align_min_pixels),
        depth_min=float(args.depth_min),
        inverse_eps=float(args.inverse_eps),
    )
    aligned_filled, fill_value = _fill_invalid_depth(aligned, valid, float(args.depth_min))
    final_valid = valid & np.isfinite(aligned) & (aligned > float(args.depth_min))

    _atomic_save_npz(
        aligned_path,
        depth=aligned_filled.astype(np.float32, copy=False),
        aligned_depth=aligned.astype(np.float32, copy=False),
        raw_depth=raw_depth.astype(np.float32, copy=False),
        mesh_depth=mesh_depth.astype(np.float32, copy=False),
        valid_mask=final_valid.astype(np.uint8),
        mesh_valid_mask=mesh_valid.astype(np.uint8),
        face_ids=np.asarray(face_ids, dtype=np.int64),
        fill_value=np.asarray([fill_value], dtype=np.float32),
        scale=np.asarray([float(align_summary.get("scale", 1.0))], dtype=np.float32),
        shift=np.asarray([float(align_summary.get("shift", 0.0))], dtype=np.float32),
        alignment_mode=np.asarray([str(align_summary.get("mode", ""))]),
    )
    _atomic_save_npz(
        mesh_depth_path,
        depth=mesh_depth.astype(np.float32, copy=False),
        valid_mask=mesh_valid.astype(np.uint8),
        face_ids=np.asarray(face_ids, dtype=np.int64),
    )

    debug_files: dict[str, str] = {}
    if bool(args.save_debug_images):
        debug_files = _save_debug_images(
            stem=stem,
            raw_depth=raw_depth,
            mesh_depth=mesh_depth,
            aligned_depth=aligned,
            aligned_filled=aligned_filled,
            valid_mask=final_valid,
            mesh_valid=mesh_valid,
            debug_dir=dirs["debug"],
            depth_min=float(args.depth_min),
        )

    score = align_summary.get("score") if isinstance(align_summary.get("score"), dict) else {}
    return {
        "image_name": str(cam.image_name),
        "stem": stem,
        "status": "written",
        "prior_path": str(prior_path),
        "aligned_depth_path": str(aligned_path),
        "mesh_depth_path": str(mesh_depth_path),
        "debug_files": debug_files,
        "depth_shape": [int(mesh_depth.shape[0]), int(mesh_depth.shape[1])],
        "mesh_valid_pixels": int(np.count_nonzero(mesh_valid)),
        "aligned_valid_pixels": int(np.count_nonzero(final_valid)),
        "aligned_fill_value": float(fill_value),
        "alignment": align_summary,
        "alignment_trials": trials,
        "median_rel_error": score.get("median_rel_error") if isinstance(score, dict) else None,
        "p90_rel_error": score.get("p90_rel_error") if isinstance(score, dict) else None,
        "raw_depth_stats": _finite_stats(raw_depth),
        "aligned_depth_stats": _finite_stats(aligned_filled),
        "mesh_depth_stats": _finite_stats(mesh_depth[mesh_valid]) if np.any(mesh_valid) else _finite_stats(mesh_depth),
    }


def main() -> None:
    args = _parse_args()
    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    depth_prior_dir = Path(args.depth_prior_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not depth_prior_dir.is_dir():
        raise FileNotFoundError(f"Depth prior dir not found: {depth_prior_dir}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh path not found: {mesh_path}")

    depth_subdirs = _parse_csv(args.depth_subdirs, DEFAULT_DEPTH_SUBDIRS)
    depth_keys = _parse_csv(args.depth_keys, DEFAULT_DEPTH_KEYS)
    alignment_modes = _parse_csv(args.alignment_modes, ("affine", "inverse_affine", "negative_affine"))
    allowed_modes = {"affine", "inverse_affine", "negative_affine", "identity"}
    bad_modes = [mode for mode in alignment_modes if mode not in allowed_modes]
    if bad_modes:
        raise ValueError(f"Unsupported --alignment_modes entries: {bad_modes}")
    if not alignment_modes:
        raise ValueError("--alignment_modes cannot be empty")

    dirs = {
        "aligned_depth": output_root / "aligned_depth",
        "mesh_depth": output_root / "mesh_depth",
        "debug": output_root / "debug",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    depth_index, duplicates = _build_depth_index(
        depth_prior_dir,
        depth_subdirs,
        recursive=bool(args.recursive_depth_search),
    )
    if not depth_index:
        raise FileNotFoundError(
            f"No supported depth files found in {depth_prior_dir} with subdirs={depth_subdirs}"
        )

    mesh = load_triangle_mesh(str(mesh_path))
    scene = build_open3d_raycast_scene(mesh) if str(args.mesh_depth_mode) == "raycast" else None
    cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    if int(args.limit) > 0:
        cameras = cameras[: int(args.limit)]

    print(f"[mesh-align-depth-v0] scene      : {scene_root}")
    print(f"[mesh-align-depth-v0] model      : {model_path}")
    print(f"[mesh-align-depth-v0] mesh       : {mesh_path}")
    print(f"[mesh-align-depth-v0] depth prior: {depth_prior_dir}")
    print(f"[mesh-align-depth-v0] output     : {output_root}")
    print(f"[mesh-align-depth-v0] views      : {len(cameras)} split={args.split} images={args.images_subdir}")
    print(f"[mesh-align-depth-v0] modes      : {','.join(alignment_modes)}")

    frames: list[dict[str, object]] = []
    for cam in tqdm(cameras, desc="mesh-align depth", dynamic_ncols=True):
        frames.append(
            _process_frame(
                cam=cam,
                mesh=mesh,
                scene=scene,
                depth_index=depth_index,
                depth_keys=depth_keys,
                alignment_modes=alignment_modes,
                args=args,
                dirs=dirs,
            )
        )

    written = [frame for frame in frames if frame.get("status") == "written"]
    skipped = [frame for frame in frames if frame.get("status") == "skipped_existing"]
    missing = [frame for frame in frames if frame.get("status") == "missing_depth_prior"]
    mode_counts = Counter(
        str(frame.get("alignment", {}).get("mode"))
        for frame in written
        if isinstance(frame.get("alignment"), dict)
    )
    rel_errors = [
        float(frame["median_rel_error"])
        for frame in written
        if frame.get("median_rel_error") is not None
    ]
    manifest = {
        "version": "mesh_aligned_depth_prior_cache_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "mesh_path": str(mesh_path),
        "depth_prior_dir": str(depth_prior_dir),
        "output_root": str(output_root),
        "aligned_depth_dir": str(dirs["aligned_depth"]),
        "mesh_depth_dir": str(dirs["mesh_depth"]),
        "debug_dir": str(dirs["debug"]),
        "args": vars(args) | {
            "scene_root": str(scene_root),
            "model_path": str(model_path),
            "mesh_path": str(mesh_path),
            "depth_prior_dir": str(depth_prior_dir),
            "output_root": str(output_root),
        },
        "depth_subdirs": depth_subdirs,
        "depth_keys": depth_keys,
        "depth_index_count": len(depth_index),
        "depth_duplicate_key_count": len(duplicates),
        "depth_duplicate_examples": {k: v[:4] for k, v in list(duplicates.items())[:8]},
        "num_views": len(cameras),
        "num_written": len(written),
        "num_skipped": len(skipped),
        "num_missing_depth_prior": len(missing),
        "alignment_mode_counts": dict(mode_counts),
        "median_rel_error_stats": _finite_stats(np.asarray(rel_errors, dtype=np.float32)),
        "frames": frames,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if missing:
        examples = [str(frame.get("image_name")) for frame in missing[:12]]
        print(f"[mesh-align-depth-v0] missing depth priors: {len(missing)} examples={examples}")
    print(f"[mesh-align-depth-v0] written : {len(written)}")
    print(f"[mesh-align-depth-v0] skipped : {len(skipped)}")
    print(f"[mesh-align-depth-v0] aligned : {dirs['aligned_depth']}")
    print(f"[mesh-align-depth-v0] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
