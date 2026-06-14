#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from train_mip_to_sof_surface_v0 import (  # noqa: E402
    build_dataset_args,
    load_cameras_for_split,
    load_model_ply,
    lookup_indexed_path,
    resolve_iteration,
    select_uniform,
)
from utils.prior_injection import index_image_dir, normalize_image_name  # noqa: E402


def _list_image_paths(root: Path) -> List[Path]:
    paths = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def _add_image_lookup_key(lookup: Dict[str, Path], key: str, path: Path) -> None:
    key = str(key).strip()
    if not key:
        return
    if key in lookup and lookup[key] != path:
        raise ValueError(f"Duplicate image lookup key '{key}' found under reference directory")
    lookup[key] = path


def _build_image_lookup_from_paths(paths: List[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in paths:
        candidates = {
            path.name,
            path.stem,
            normalize_image_name(path.name),
            normalize_image_name(path.stem),
            path.name.lower(),
            path.stem.lower(),
            normalize_image_name(path.name).lower(),
            normalize_image_name(path.stem).lower(),
        }
        for key in candidates:
            _add_image_lookup_key(lookup, key, path)
    return lookup


def _select_uniform_with_indices(items: List[object], max_items: int) -> Tuple[List[object], List[int]]:
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items), list(range(len(items)))
    ids = np.unique(np.linspace(0, len(items) - 1, num=int(max_items), dtype=np.int64))
    selected_indices = [int(idx) for idx in ids.tolist()]
    return [items[idx] for idx in selected_indices], selected_indices


def _resolve_aux_image_path(
    *,
    lookup: Dict[str, Path] | None,
    paths: List[Path] | None,
    image_name: str,
    source_view_idx: int,
    local_view_idx: int,
) -> Path | None:
    if lookup is None:
        return None
    tried_keys = [
        str(image_name),
        normalize_image_name(str(image_name)),
        str(image_name).lower(),
        normalize_image_name(str(image_name)).lower(),
        f"{int(source_view_idx):05d}",
        f"{int(source_view_idx):05d}.png",
        f"{int(local_view_idx):05d}",
        f"{int(local_view_idx):05d}.png",
    ]
    for key in tried_keys:
        matched = lookup.get(key)
        if matched is not None:
            return matched
    if paths is not None:
        if 0 <= int(source_view_idx) < len(paths):
            return paths[int(source_view_idx)]
        if 0 <= int(local_view_idx) < len(paths):
            return paths[int(local_view_idx)]
    return None


def _load_rgb(path: Path, size: Tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask(path: Path, size: Tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(np.round(np.clip(image, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.clip(np.round(np.clip(mask, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def _gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
    if float(radius) <= 0.0:
        return image.astype(np.float32, copy=True)
    if image.ndim == 2:
        pil = Image.fromarray(np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8), mode="L")
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(float(radius))), dtype=np.float32) / 255.0
    pil = Image.fromarray(np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(float(radius))), dtype=np.float32) / 255.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _luma(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((np.clip(pred, 0.0, 1.0) - np.clip(target, 0.0, 1.0)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * math.log10(mse))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.clip(pred, 0.0, 1.0) - np.clip(target, 0.0, 1.0))))


def _hf_excess(image: np.ndarray, reference: np.ndarray, radius: float, ratio: float, margin: float) -> float:
    hp = _luma(image - _gaussian_blur(image, radius))
    ref_hp = _luma(reference - _gaussian_blur(reference, radius))
    excess = np.maximum(np.abs(hp) - float(ratio) * np.abs(ref_hp) - float(margin), 0.0)
    return float(np.mean(excess))


def _local_average(values: np.ndarray, radius: float) -> np.ndarray:
    return _gaussian_blur(values.astype(np.float32), radius)


def _fuse_view(
    stable: np.ndarray,
    detail: np.ndarray,
    reference: np.ndarray | None,
    source_mask: np.ndarray | None,
    *,
    lowpass_radius: float,
    mask_radius: float,
    detail_temperature: float,
    star_ratio: float,
    star_margin: float,
    star_threshold: float,
    star_temperature: float,
    low_replace_strength: float,
    star_suppress_strength: float,
    source_mask_risk_strength: float,
    detail_conf_power: float,
    confidence_star_power: float,
    detail_floor: float,
    detail_ceiling: float,
    hf_clip_ratio: float,
    hf_clip_margin: float,
    prior_mask_floor: float,
) -> Dict[str, np.ndarray | float]:
    stable_lp = _gaussian_blur(stable, lowpass_radius)
    detail_lp = _gaussian_blur(detail, lowpass_radius)
    stable_hp = stable - stable_lp
    detail_hp = detail - detail_lp

    if reference is None:
        reference = stable
    reference_lp = _gaussian_blur(reference, lowpass_radius)
    reference_hp = reference - reference_lp

    stable_hpy = _luma(stable_hp)
    detail_hpy = _luma(detail_hp)
    reference_hpy = _luma(reference_hp)

    stable_err = _local_average(np.abs(stable_hpy - reference_hpy), mask_radius)
    detail_err = _local_average(np.abs(detail_hpy - reference_hpy), mask_radius)
    detail_temp = max(float(detail_temperature), 1e-6)
    detail_conf = _sigmoid((stable_err - detail_err) / detail_temp)

    hf_limit = float(star_ratio) * np.abs(reference_hpy) + float(star_margin)
    highpass_excess = np.maximum(np.abs(detail_hpy) - hf_limit, 0.0)
    bright_excess = np.maximum(detail_hpy - hf_limit, 0.0)
    raw_star = _local_average(0.75 * highpass_excess + 0.25 * bright_excess, mask_radius)
    star_temp = max(float(star_temperature), 1e-6)
    star_mask = _sigmoid((raw_star - float(star_threshold)) / star_temp)

    if source_mask is not None:
        source_mask = np.clip(source_mask.astype(np.float32), 0.0, 1.0)
        source_risk = np.clip(float(source_mask_risk_strength) * (1.0 - source_mask), 0.0, 1.0)
        detail_conf = detail_conf * source_mask
        # Low source confidence means "do not trust detail here"; it should
        # suppress detail release instead of hiding the star/artifact risk.
        star_mask = np.maximum(star_mask, source_risk)

    if float(detail_conf_power) != 1.0:
        detail_conf = np.power(np.clip(detail_conf, 0.0, 1.0), max(float(detail_conf_power), 1e-6))

    low_weight = np.clip(float(low_replace_strength) * star_mask * (1.0 - 0.5 * detail_conf), 0.0, 1.0)
    lp_base = (1.0 - low_weight[..., None]) * detail_lp + low_weight[..., None] * stable_lp

    suppress = np.clip(float(star_suppress_strength) * star_mask, 0.0, 1.0)
    detail_weight = detail_conf * (1.0 - suppress)
    detail_weight = np.clip(detail_weight, float(detail_floor), float(detail_ceiling))

    clip_limit = float(hf_clip_ratio) * np.abs(reference_hp) + float(hf_clip_margin)
    detail_hp_clipped = np.clip(detail_hp, -clip_limit, clip_limit)
    hp_safe = stable_hp
    fused = lp_base + detail_weight[..., None] * detail_hp_clipped + (1.0 - detail_weight[..., None]) * hp_safe
    fused = np.clip(fused, 0.0, 1.0)

    star_conf = np.power(np.clip(1.0 - star_mask, 0.0, 1.0), max(float(confidence_star_power), 1e-6))
    confidence = star_conf * (0.25 + 0.75 * np.clip(detail_conf, 0.0, 1.0))
    if source_mask is not None:
        confidence = confidence * source_mask
    confidence = np.clip(confidence, float(prior_mask_floor), 1.0)

    return {
        "fused": fused.astype(np.float32),
        "mask": confidence.astype(np.float32),
        "star_mask": star_mask.astype(np.float32),
        "detail_weight": detail_weight.astype(np.float32),
        "low_weight": low_weight.astype(np.float32),
        "mean_star_mask": float(np.mean(star_mask)),
        "mean_detail_weight": float(np.mean(detail_weight)),
        "mean_low_weight": float(np.mean(low_weight)),
    }


@torch.no_grad()
def _render_detail(camera, gaussians, background: torch.Tensor) -> np.ndarray:
    pkg = render_simple(camera, gaussians, background)
    image = pkg["render"][:3].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return image.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prepared SR-prior root by fusing a stable mip30k+SR prior with "
            "a rendered detail model, using local frequency-domain gates."
        )
    )
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--detail_model_path", required=True)
    parser.add_argument("--stable_prior_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--detail_iteration", type=int, default=-1)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--reference_dir", default="")
    parser.add_argument("--source_mask_dir", default="")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--debug_views", type=int, default=8)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--lowpass_radius", type=float, default=6.0)
    parser.add_argument("--mask_radius", type=float, default=3.0)
    parser.add_argument("--detail_temperature", type=float, default=0.006)
    parser.add_argument("--star_ratio", type=float, default=1.25)
    parser.add_argument("--star_margin", type=float, default=0.012)
    parser.add_argument("--star_threshold", type=float, default=0.004)
    parser.add_argument("--star_temperature", type=float, default=0.0035)
    parser.add_argument("--low_replace_strength", type=float, default=0.45)
    parser.add_argument("--star_suppress_strength", type=float, default=0.65)
    parser.add_argument("--source_mask_risk_strength", type=float, default=0.0)
    parser.add_argument("--detail_conf_power", type=float, default=1.0)
    parser.add_argument("--confidence_star_power", type=float, default=1.0)
    parser.add_argument("--detail_floor", type=float, default=0.30)
    parser.add_argument("--detail_ceiling", type=float, default=1.00)
    parser.add_argument("--hf_clip_ratio", type=float, default=1.35)
    parser.add_argument("--hf_clip_margin", type=float, default=0.020)
    parser.add_argument("--prior_mask_floor", type=float, default=0.35)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    detail_model_path = Path(args.detail_model_path).expanduser().resolve()
    stable_prior_dir = Path(args.stable_prior_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve() if str(args.reference_dir).strip() else None
    source_mask_dir = Path(args.source_mask_dir).expanduser().resolve() if str(args.source_mask_dir).strip() else None

    if not stable_prior_dir.is_dir():
        raise FileNotFoundError(f"stable prior dir not found: {stable_prior_dir}")
    if reference_dir is not None and not reference_dir.is_dir():
        raise FileNotFoundError(f"reference dir not found: {reference_dir}")
    if source_mask_dir is not None and not source_mask_dir.is_dir():
        raise FileNotFoundError(f"source mask dir not found: {source_mask_dir}")

    stable_paths = _list_image_paths(stable_prior_dir)
    stable_index = _build_image_lookup_from_paths(stable_paths)
    reference_paths = _list_image_paths(reference_dir) if reference_dir is not None else None
    reference_index = _build_image_lookup_from_paths(reference_paths) if reference_paths is not None else None
    source_mask_paths = _list_image_paths(source_mask_dir) if source_mask_dir is not None else None
    source_mask_index = _build_image_lookup_from_paths(source_mask_paths) if source_mask_paths is not None else None

    dataset_args = build_dataset_args(str(scene_root), str(detail_model_path), str(args.images_subdir))
    detail_iteration = resolve_iteration(detail_model_path, int(args.detail_iteration))
    cameras = load_cameras_for_split(scene_root, detail_model_path, str(args.images_subdir), str(args.split))
    selected_cameras, selected_indices = _select_uniform_with_indices(cameras, int(args.max_views))
    detail_model = load_model_ply(detail_model_path, detail_iteration, int(dataset_args.sh_degree))
    if selected_cameras:
        detail_model.compute_3D_filter(selected_cameras, CUDA=False)

    background = torch.tensor(
        [1, 1, 1] if bool(args.white_background) else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    fused_dir = output_root / "fused_priors"
    anchor_dir = output_root / "aligned_references"
    mask_dir = output_root / "usable_masks"
    debug_dir = output_root / "debug"
    for directory in (fused_dir, anchor_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    per_view: List[Dict[str, float | str | int | None]] = []
    missing_stable: List[str] = []
    processed = 0
    start_time = time.time()
    print(
        f"[quality-fuse-v0] selected cameras: {len(selected_cameras)} "
        f"(split={args.split}, max_views={args.max_views})",
        flush=True,
    )
    for local_view_idx, (source_view_idx, camera) in enumerate(zip(selected_indices, selected_cameras)):
        image_name = normalize_image_name(camera.image_name)
        stable_path = _resolve_aux_image_path(
            lookup=stable_index,
            paths=stable_paths,
            image_name=str(camera.image_name),
            source_view_idx=int(source_view_idx),
            local_view_idx=int(local_view_idx),
        )
        if stable_path is None:
            missing_stable.append(str(camera.image_name))
            continue

        detail_rgb = _render_detail(camera, detail_model, background)
        height, width = detail_rgb.shape[:2]
        pil_size = (int(width), int(height))
        stable_rgb = _load_rgb(stable_path, size=pil_size)

        reference_path = _resolve_aux_image_path(
            lookup=reference_index,
            paths=reference_paths,
            image_name=str(camera.image_name),
            source_view_idx=int(source_view_idx),
            local_view_idx=int(local_view_idx),
        ) if reference_index is not None else None
        reference_rgb = _load_rgb(reference_path, size=pil_size) if reference_path is not None else stable_rgb

        source_mask_path = _resolve_aux_image_path(
            lookup=source_mask_index,
            paths=source_mask_paths,
            image_name=str(camera.image_name),
            source_view_idx=int(source_view_idx),
            local_view_idx=int(local_view_idx),
        ) if source_mask_index is not None else None
        source_mask = _load_mask(source_mask_path, size=pil_size) if source_mask_path is not None else None

        fused = _fuse_view(
            stable_rgb,
            detail_rgb,
            reference_rgb,
            source_mask,
            lowpass_radius=float(args.lowpass_radius),
            mask_radius=float(args.mask_radius),
            detail_temperature=float(args.detail_temperature),
            star_ratio=float(args.star_ratio),
            star_margin=float(args.star_margin),
            star_threshold=float(args.star_threshold),
            star_temperature=float(args.star_temperature),
            low_replace_strength=float(args.low_replace_strength),
            star_suppress_strength=float(args.star_suppress_strength),
            source_mask_risk_strength=float(args.source_mask_risk_strength),
            detail_conf_power=float(args.detail_conf_power),
            confidence_star_power=float(args.confidence_star_power),
            detail_floor=float(args.detail_floor),
            detail_ceiling=float(args.detail_ceiling),
            hf_clip_ratio=float(args.hf_clip_ratio),
            hf_clip_margin=float(args.hf_clip_margin),
            prior_mask_floor=float(args.prior_mask_floor),
        )
        fused_rgb = fused["fused"]
        confidence = fused["mask"]

        out_name = f"{image_name}.png"
        _save_rgb(fused_dir / out_name, fused_rgb)
        _save_rgb(anchor_dir / out_name, reference_rgb)
        _save_mask(mask_dir / out_name, confidence)

        if processed < int(args.debug_views):
            debug_view = debug_dir / image_name
            _save_rgb(debug_view / "stable.png", stable_rgb)
            _save_rgb(debug_view / "detail_render.png", detail_rgb)
            _save_rgb(debug_view / "reference.png", reference_rgb)
            _save_rgb(debug_view / "fused.png", fused_rgb)
            _save_mask(debug_view / "star_mask.png", fused["star_mask"])
            _save_mask(debug_view / "detail_weight.png", fused["detail_weight"])
            _save_mask(debug_view / "prior_mask.png", confidence)

        row: Dict[str, float | str | int | None] = {
            "image_name": image_name,
            "source_view_index": int(source_view_idx),
            "stable_path": str(stable_path),
            "reference_path": str(reference_path) if reference_path is not None else None,
            "source_mask_path": str(source_mask_path) if source_mask_path is not None else None,
            "mean_prior_mask": float(np.mean(confidence)),
            "mean_star_mask": float(fused["mean_star_mask"]),
            "mean_detail_weight": float(fused["mean_detail_weight"]),
            "mean_low_weight": float(fused["mean_low_weight"]),
        }
        if reference_rgb is not None:
            row.update(
                {
                    "stable_psnr_ref": _psnr(stable_rgb, reference_rgb),
                    "detail_psnr_ref": _psnr(detail_rgb, reference_rgb),
                    "fused_psnr_ref": _psnr(fused_rgb, reference_rgb),
                    "stable_mae_ref": _mae(stable_rgb, reference_rgb),
                    "detail_mae_ref": _mae(detail_rgb, reference_rgb),
                    "fused_mae_ref": _mae(fused_rgb, reference_rgb),
                    "detail_hf_excess_ref": _hf_excess(
                        detail_rgb,
                        reference_rgb,
                        float(args.lowpass_radius),
                        float(args.star_ratio),
                        float(args.star_margin),
                    ),
                    "fused_hf_excess_ref": _hf_excess(
                        fused_rgb,
                        reference_rgb,
                        float(args.lowpass_radius),
                        float(args.star_ratio),
                        float(args.star_margin),
                    ),
                }
            )
        per_view.append(row)
        processed += 1
        progress_every = max(int(args.progress_every), 1)
        if processed == 1 or processed % progress_every == 0 or local_view_idx == len(selected_cameras) - 1:
            elapsed = time.time() - start_time
            rate = elapsed / max(processed, 1)
            remaining = max(len(selected_cameras) - local_view_idx - 1, 0) * rate
            print(
                f"[quality-fuse-v0] fused {processed}/{len(selected_cameras)} "
                f"view={image_name} star={float(fused['mean_star_mask']):.4f} "
                f"detail_w={float(fused['mean_detail_weight']):.4f} "
                f"eta={remaining:.1f}s",
                flush=True,
            )

    if selected_cameras and processed == 0:
        stable_examples = [path.stem for path in stable_paths[:8]]
        missing_examples = missing_stable[:8]
        raise RuntimeError(
            "No quality-fused priors were produced. This usually means the stable/reference "
            "render directory is not aligned with the selected camera split. "
            f"stable_prior_dir={stable_prior_dir} contains {len(stable_paths)} image(s), "
            f"sample_stems={stable_examples}, missing_camera_examples={missing_examples}. "
            "If the stable renders are numbered 00000.png, make sure the server has a build "
            "with render-order fallback, or rerender/pass the stable dir for the same split."
        )

    def _mean_key(key: str) -> float | None:
        values = [float(row[key]) for row in per_view if key in row and row[key] is not None]
        if not values:
            return None
        return float(np.mean(values))

    summary: Dict[str, object] = {
        "version": "quality_fused_sr_priors_v0",
        "scene_root": str(scene_root),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "detail_model_path": str(detail_model_path),
        "detail_iteration": int(detail_iteration),
        "stable_prior_dir": str(stable_prior_dir),
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "source_mask_dir": str(source_mask_dir) if source_mask_dir is not None else None,
        "output_root": str(output_root),
        "fused_priors": str(fused_dir),
        "aligned_references": str(anchor_dir),
        "usable_masks": str(mask_dir),
        "camera_count": int(len(selected_cameras)),
        "processed_views": int(processed),
        "missing_stable": missing_stable,
        "params": {
            "lowpass_radius": float(args.lowpass_radius),
            "mask_radius": float(args.mask_radius),
            "detail_temperature": float(args.detail_temperature),
            "star_ratio": float(args.star_ratio),
            "star_margin": float(args.star_margin),
            "star_threshold": float(args.star_threshold),
            "star_temperature": float(args.star_temperature),
            "low_replace_strength": float(args.low_replace_strength),
            "star_suppress_strength": float(args.star_suppress_strength),
            "source_mask_risk_strength": float(args.source_mask_risk_strength),
            "detail_conf_power": float(args.detail_conf_power),
            "confidence_star_power": float(args.confidence_star_power),
            "detail_floor": float(args.detail_floor),
            "detail_ceiling": float(args.detail_ceiling),
            "hf_clip_ratio": float(args.hf_clip_ratio),
            "hf_clip_margin": float(args.hf_clip_margin),
            "prior_mask_floor": float(args.prior_mask_floor),
        },
        "aggregated": {
            "mean_prior_mask": _mean_key("mean_prior_mask"),
            "mean_star_mask": _mean_key("mean_star_mask"),
            "mean_detail_weight": _mean_key("mean_detail_weight"),
            "mean_low_weight": _mean_key("mean_low_weight"),
            "stable_psnr_ref": _mean_key("stable_psnr_ref"),
            "detail_psnr_ref": _mean_key("detail_psnr_ref"),
            "fused_psnr_ref": _mean_key("fused_psnr_ref"),
            "stable_mae_ref": _mean_key("stable_mae_ref"),
            "detail_mae_ref": _mean_key("detail_mae_ref"),
            "fused_mae_ref": _mean_key("fused_mae_ref"),
            "detail_hf_excess_ref": _mean_key("detail_hf_excess_ref"),
            "fused_hf_excess_ref": _mean_key("fused_hf_excess_ref"),
        },
        "per_view": per_view,
    }
    manifest = {
        "prior_dir": str(stable_prior_dir),
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "output_root": str(output_root),
        "mask_threshold": None,
        "mask_mode": "quality_fused_confidence",
        "discrepancy_floor": None,
        "save_fused_priors": True,
        "copy_raw_priors": False,
        "save_discrepancy_npz": False,
        "num_priors": int(len(stable_index)),
        "num_matched": int(processed),
        "missing_reference_count": 0,
        "missing_reference_examples": [],
        "usable_mean": summary["aggregated"]["mean_prior_mask"],
        "discrepancy_mean": None,
        "frames": [
            {
                "image_name": f"{row['image_name']}.png",
                "stem": str(row["image_name"]),
                "source_view_index": int(row["source_view_index"]),
                "usable_mean": float(row["mean_prior_mask"]),
                "stable_path": row["stable_path"],
                "reference_path": row["reference_path"],
                "source_mask_path": row["source_mask_path"],
            }
            for row in per_view
        ],
        "quality_fuse_summary": "quality_fused_sr_priors_v0_summary.json",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "quality_fused_sr_priors_v0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"[quality-fuse-v0] fused priors : {fused_dir}")
    print(f"[quality-fuse-v0] masks        : {mask_dir}")
    print(f"[quality-fuse-v0] summary      : {output_root / 'quality_fused_sr_priors_v0_summary.json'}")


if __name__ == "__main__":
    main()
