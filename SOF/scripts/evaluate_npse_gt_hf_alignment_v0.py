#!/usr/bin/env python3
"""Evaluate whether NPSE high-frequency targets align with GT high frequency.

The main use case is checking conservative NPSE edge targets before training:
edge_target should inject only high-frequency residuals that agree with GT,
especially inside trust_edge / edge_band masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
EPS = 1e-8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate high-frequency images against GT, optionally "
            "in a soft mask and relative to a low-frequency anchor."
        )
    )
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Candidate image directory. Can be repeated, e.g. edge_target=/path.",
    )
    parser.add_argument("--mask_dir", type=Path, default=None, help="Optional soft mask dir, e.g. trust_edge.")
    parser.add_argument("--anchor_dir", type=Path, default=None, help="Optional anchor image dir for residual-HF metrics.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--match_policy",
        choices=("stem", "order", "order_if_needed", "llff_train_order"),
        default="order_if_needed",
        help="How to align GT/candidate/mask frames when stems differ.",
    )
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--highpass_kernel", type=int, default=15)
    parser.add_argument("--mask_power", type=float, default=1.0)
    parser.add_argument("--hard_mask_threshold", type=float, default=0.05)
    parser.add_argument("--active_gt_percentile", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=12)
    parser.add_argument("--vis_scale", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def _index_images(root: Path) -> Dict[str, Path]:
    if root is None or not root.is_dir():
        return {}
    index: Dict[str, Path] = {}
    for path in _iter_images(root):
        index.setdefault(path.stem, path)
    return index


def _extract_trailing_int(stem: str) -> Optional[int]:
    digits = []
    for ch in reversed(str(stem)):
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(reversed(digits)))


def _sort_key_from_stem(stem: str) -> Tuple[int, int, str]:
    idx = _extract_trailing_int(stem)
    if idx is None:
        return (1, 0, str(stem))
    return (0, idx, str(stem))


def _sorted_index_items(index: Dict[str, Path]) -> List[Tuple[str, Path]]:
    return sorted(index.items(), key=lambda item: _sort_key_from_stem(item[0]))


def _sample_keys(index: Dict[str, Path], n: int = 5) -> List[str]:
    return [stem for stem, _path in _sorted_index_items(index)[:n]]


def _parse_candidate_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    candidates: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Candidate must be NAME=DIR, got: {spec}")
        name, value = spec.split("=", 1)
        name = name.strip()
        path = Path(value).expanduser()
        if not name:
            raise ValueError(f"Candidate name is empty in: {spec}")
        if not path.is_dir():
            raise FileNotFoundError(f"Candidate dir not found for {name}: {path}")
        candidates.append((name, path))
    if not candidates:
        raise ValueError("At least one --candidate NAME=DIR is required.")
    return candidates


def _load_rgb01(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_gray01(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb01(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def _box_filter_axis(x: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(x, pad, mode="reflect")
    cumsum = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumsum.shape)
    zero_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), cumsum], axis=axis)
    n = x.shape[axis]
    window = radius * 2 + 1
    start = np.arange(n)
    end = start + window
    sums = np.take(cumsum, end, axis=axis) - np.take(cumsum, start, axis=axis)
    return (sums / float(window)).astype(np.float32, copy=False)


def _box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return image.astype(np.float32, copy=False)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    out = _box_filter_axis(image.astype(np.float32, copy=False), radius, axis=0)
    out = _box_filter_axis(out, radius, axis=1)
    return out


def _highpass(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return image.astype(np.float32, copy=False) - _box_blur(image, kernel_size)


def _luma(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    channel_factor = 1
    if values.ndim == 3 and weight.ndim == 2:
        weight = weight[..., None]
        channel_factor = int(values.shape[-1])
    denom = float(np.sum(weight) * channel_factor)
    if denom <= EPS:
        return float("nan")
    return float(np.sum(values * weight) / denom)


def _weighted_abs_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return _weighted_mean(np.abs(values), weight)


def _weighted_rmse(values: np.ndarray, weight: np.ndarray) -> float:
    mse = _weighted_mean(values * values, weight)
    if math.isnan(mse):
        return float("nan")
    return float(math.sqrt(max(mse, 0.0)))


def _weighted_corr(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w))
    if denom <= EPS:
        return float("nan")
    ma = float(np.sum(w * a) / denom)
    mb = float(np.sum(w * b) / denom)
    da = a - ma
    db = b - mb
    va = float(np.sum(w * da * da))
    vb = float(np.sum(w * db * db))
    if va <= EPS or vb <= EPS:
        return float("nan")
    return float(np.sum(w * da * db) / math.sqrt(va * vb))


def _weighted_cosine(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim == 2 and a.size == w.size * 3:
        w = np.repeat(w[..., None], 3, axis=2)
    w = w.reshape(-1)
    aa = float(np.sum(w * a * a))
    bb = float(np.sum(w * b * b))
    if aa <= EPS or bb <= EPS:
        return float("nan")
    return float(np.sum(w * a * b) / math.sqrt(aa * bb))


def _active_sign_agreement(candidate: np.ndarray, gt: np.ndarray, weight: np.ndarray, percentile: float) -> float:
    active_values = np.abs(gt)[weight > 0]
    if active_values.size == 0:
        return float("nan")
    threshold = float(np.percentile(active_values, float(percentile)))
    active = weight * (np.abs(gt) >= threshold).astype(np.float32)
    denom = float(np.sum(active))
    if denom <= EPS:
        return float("nan")
    same = (np.sign(candidate) == np.sign(gt)).astype(np.float32)
    same[(candidate == 0.0) | (gt == 0.0)] = 0.0
    return float(np.sum(same * active) / denom)


def _grad_cosine(candidate: np.ndarray, gt: np.ndarray, weight: np.ndarray) -> float:
    gy_c, gx_c = np.gradient(candidate)
    gy_g, gx_g = np.gradient(gt)
    norm_c = np.sqrt(gx_c * gx_c + gy_c * gy_c)
    norm_g = np.sqrt(gx_g * gx_g + gy_g * gy_g)
    cos = (gx_c * gx_g + gy_c * gy_g) / np.maximum(norm_c * norm_g, EPS)
    grad_weight = weight * norm_g
    return _weighted_mean(cos, grad_weight)


def _energy_stats(candidate: np.ndarray, gt: np.ndarray, weight: np.ndarray) -> Dict[str, float]:
    cand_abs = np.abs(candidate)
    gt_abs = np.abs(gt)
    cand_energy = _weighted_mean(cand_abs, weight)
    gt_energy = _weighted_mean(gt_abs, weight)
    if math.isnan(cand_energy) or math.isnan(gt_energy) or gt_energy <= EPS:
        ratio = float("nan")
        over = float("nan")
        under = float("nan")
    else:
        ratio = float(cand_energy / gt_energy)
        over = float(_weighted_mean(np.maximum(cand_abs - gt_abs, 0.0), weight) / gt_energy)
        under = float(_weighted_mean(np.maximum(gt_abs - cand_abs, 0.0), weight) / gt_energy)
    return {
        "energy_candidate": cand_energy,
        "energy_gt": gt_energy,
        "energy_ratio": ratio,
        "over_injection": over,
        "under_injection": under,
    }


def _metric_pack(candidate_hp: np.ndarray, gt_hp: np.ndarray, weight: np.ndarray, active_percentile: float, prefix: str) -> Dict[str, float]:
    cand_l = _luma(candidate_hp)
    gt_l = _luma(gt_hp)
    diff = candidate_hp - gt_hp
    out = {
        f"{prefix}_l1_rgb": _weighted_abs_mean(diff, weight),
        f"{prefix}_rmse_rgb": _weighted_rmse(diff, weight),
        f"{prefix}_corr_luma": _weighted_corr(cand_l, gt_l, weight),
        f"{prefix}_cos_rgb": _weighted_cosine(candidate_hp, gt_hp, weight),
        f"{prefix}_sign_luma": _active_sign_agreement(cand_l, gt_l, weight, active_percentile),
        f"{prefix}_grad_cos_luma": _grad_cosine(cand_l, gt_l, weight),
    }
    out.update({f"{prefix}_{k}": v for k, v in _energy_stats(cand_l, gt_l, weight).items()})
    return out


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0.0, 1.0, 0.2), alpha: float = 0.55) -> np.ndarray:
    color_arr = np.asarray(color, dtype=np.float32)[None, None, :]
    a = np.clip(mask, 0.0, 1.0)[..., None] * float(alpha)
    return np.clip(image * (1.0 - a) + color_arr * a, 0.0, 1.0)


def _label_tile(image: np.ndarray, label: str) -> Image.Image:
    arr = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    tile = Image.fromarray(arr)
    header = Image.new("RGB", (tile.width, 22), (20, 20, 20))
    draw = ImageDraw.Draw(header)
    draw.text((5, 4), label, fill=(240, 240, 240))
    out = Image.new("RGB", (tile.width, tile.height + header.height))
    out.paste(header, (0, 0))
    out.paste(tile, (0, header.height))
    return out


def _make_debug_sheet(
    gt: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    gt_hp: np.ndarray,
    candidate_hp: np.ndarray,
    vis_scale: float,
) -> Image.Image:
    diff_l = np.abs(_luma(candidate_hp - gt_hp))
    vmax = float(np.percentile(diff_l, 98.0)) if np.any(diff_l > 0) else 1.0
    diff_heat = np.zeros_like(candidate)
    diff_heat[..., 0] = np.clip(diff_l / max(vmax, EPS), 0.0, 1.0)
    diff_heat[..., 1] = diff_heat[..., 0] * 0.2
    diff_heat[..., 2] = 1.0 - diff_heat[..., 0]
    tiles = [
        _label_tile(gt, "gt"),
        _label_tile(candidate, "candidate"),
        _label_tile(_overlay_mask(candidate, mask), "trust overlay"),
        _label_tile(np.clip(0.5 + gt_hp * float(vis_scale), 0.0, 1.0), "gt hp"),
        _label_tile(np.clip(0.5 + candidate_hp * float(vis_scale), 0.0, 1.0), "candidate hp"),
        _label_tile(diff_heat, "|hp diff|"),
    ]
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    return sheet


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _build_stem_pairs(
    *,
    gt_index: Dict[str, Path],
    candidate_index: Dict[str, Path],
    mask_index: Dict[str, Path],
    anchor_index: Dict[str, Path],
) -> List[Dict[str, Optional[Path]]]:
    stems = set(gt_index.keys()) & set(candidate_index.keys())
    if mask_index:
        stems &= set(mask_index.keys())
    pairs: List[Dict[str, Optional[Path]]] = []
    for stem in sorted(stems, key=_sort_key_from_stem):
        pairs.append(
            {
                "stem": stem,
                "gt_path": gt_index[stem],
                "candidate_path": candidate_index[stem],
                "mask_path": mask_index.get(stem) if mask_index else None,
                "anchor_path": anchor_index.get(stem) if anchor_index else None,
            }
        )
    return pairs


def _build_order_pairs(
    *,
    gt_index: Dict[str, Path],
    candidate_index: Dict[str, Path],
    mask_index: Dict[str, Path],
    anchor_index: Dict[str, Path],
    llff_train: bool,
    llffhold: int,
) -> List[Dict[str, Optional[Path]]]:
    gt_items = _sorted_index_items(gt_index)
    if llff_train:
        hold = max(int(llffhold), 1)
        gt_items = [item for idx, item in enumerate(gt_items) if idx % hold != 0]

    candidate_items = _sorted_index_items(candidate_index)
    mask_items = _sorted_index_items(mask_index) if mask_index else []
    anchor_items = _sorted_index_items(anchor_index) if anchor_index else []

    pair_count = min(len(gt_items), len(candidate_items))
    if mask_items:
        pair_count = min(pair_count, len(mask_items))
    pairs: List[Dict[str, Optional[Path]]] = []
    for idx in range(pair_count):
        gt_stem, gt_path = gt_items[idx]
        candidate_stem, candidate_path = candidate_items[idx]
        mask_path = mask_items[idx][1] if mask_items else None
        anchor_path = anchor_index.get(gt_stem)
        if anchor_path is None and idx < len(anchor_items):
            anchor_path = anchor_items[idx][1]
        pairs.append(
            {
                "stem": gt_stem,
                "gt_path": gt_path,
                "candidate_path": candidate_path,
                "mask_path": mask_path,
                "anchor_path": anchor_path,
                "candidate_stem": candidate_stem,
                "mask_stem": mask_items[idx][0] if mask_items else "",
                "anchor_stem": anchor_path.stem if anchor_path is not None else "",
            }
        )
    return pairs


def _build_pairs_for_candidate(
    *,
    gt_index: Dict[str, Path],
    candidate_index: Dict[str, Path],
    mask_index: Dict[str, Path],
    anchor_index: Dict[str, Path],
    match_policy: str,
    llffhold: int,
) -> Tuple[List[Dict[str, Optional[Path]]], str]:
    policy = str(match_policy)
    if policy in {"stem", "order_if_needed"}:
        pairs = _build_stem_pairs(
            gt_index=gt_index,
            candidate_index=candidate_index,
            mask_index=mask_index,
            anchor_index=anchor_index,
        )
        if pairs or policy == "stem":
            return pairs, "stem"
    if policy == "llff_train_order":
        return (
            _build_order_pairs(
                gt_index=gt_index,
                candidate_index=candidate_index,
                mask_index=mask_index,
                anchor_index=anchor_index,
                llff_train=True,
                llffhold=llffhold,
            ),
            "llff_train_order",
        )
    return (
        _build_order_pairs(
            gt_index=gt_index,
            candidate_index=candidate_index,
            mask_index=mask_index,
            anchor_index=anchor_index,
            llff_train=False,
            llffhold=llffhold,
        ),
        "order",
    )


def _empty_match_error(
    *,
    gt_index: Dict[str, Path],
    candidate_indices: Dict[str, Dict[str, Path]],
    mask_index: Dict[str, Path],
    match_policy: str,
    llffhold: int,
) -> str:
    parts = [
        "No matched frames across GT, candidates, and mask.",
        f"match_policy={match_policy} llffhold={llffhold}",
        f"gt_count={len(gt_index)} gt_sample={_sample_keys(gt_index)}",
    ]
    if mask_index:
        parts.append(f"mask_count={len(mask_index)} mask_sample={_sample_keys(mask_index)}")
    for name, index in candidate_indices.items():
        common = len(set(gt_index.keys()) & set(index.keys()))
        parts.append(
            f"candidate={name} count={len(index)} common_gt_stems={common} sample={_sample_keys(index)}"
        )
    return "\n".join(parts)


def _process_candidate(
    *,
    name: str,
    pairs: List[Dict[str, Optional[Path]]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    debug_written = 0
    debug_dir = args.output_dir / "debug" / name

    for pair in pairs:
        stem = str(pair["stem"])
        gt_path = pair["gt_path"]
        candidate_path = pair["candidate_path"]
        mask_path = pair.get("mask_path")
        anchor_path = pair.get("anchor_path")
        if gt_path is None or candidate_path is None:
            continue
        gt_image = _load_rgb01(gt_path)
        size = (gt_image.shape[1], gt_image.shape[0])
        candidate_image = _load_rgb01(candidate_path, size=size)
        mask = np.ones((gt_image.shape[0], gt_image.shape[1]), dtype=np.float32)
        if mask_path is not None:
            mask = _load_gray01(mask_path, size=size)
        if float(args.mask_power) != 1.0:
            mask = np.power(np.clip(mask, 0.0, 1.0), max(float(args.mask_power), 0.0)).astype(np.float32)
        else:
            mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        gt_hp = _highpass(gt_image, int(args.highpass_kernel))
        candidate_hp = _highpass(candidate_image, int(args.highpass_kernel))

        row: Dict[str, object] = {
            "candidate": name,
            "stem": stem,
            "gt_path": str(gt_path),
            "candidate_path": str(candidate_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "mask_mean": float(mask.mean()),
            "mask_hard_ratio": float((mask >= float(args.hard_mask_threshold)).mean()),
            "mask_soft_sum": float(mask.sum()),
        }
        row.update(_metric_pack(candidate_hp, gt_hp, mask, float(args.active_gt_percentile), "abs_hf"))

        if anchor_path is not None:
            anchor_image = _load_rgb01(anchor_path, size=size)
            anchor_hp = _highpass(anchor_image, int(args.highpass_kernel))
            row["anchor_path"] = str(anchor_path)
            row.update(
                _metric_pack(
                    candidate_hp - anchor_hp,
                    gt_hp - anchor_hp,
                    mask,
                    float(args.active_gt_percentile),
                    "delta_hf",
                )
            )
        else:
            row["anchor_path"] = ""

        rows.append(row)

        if debug_written < int(args.debug_limit):
            sheet = _make_debug_sheet(gt_image, candidate_image, mask, gt_hp, candidate_hp, float(args.vis_scale))
            debug_path = debug_dir / f"{stem}.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(debug_path)
            debug_written += 1

    metric_keys = [
        key
        for row in rows[:1]
        for key in row.keys()
        if isinstance(row.get(key), float) and key not in {"mask_mean", "mask_hard_ratio", "mask_soft_sum"}
    ]
    summary: Dict[str, float] = {
        "frames": float(len(rows)),
        "mask_mean": _nanmean([float(r["mask_mean"]) for r in rows]),
        "mask_hard_ratio": _nanmean([float(r["mask_hard_ratio"]) for r in rows]),
        "mask_soft_sum": _nanmean([float(r["mask_soft_sum"]) for r in rows]),
    }
    for key in metric_keys:
        summary[key] = _nanmean([float(r[key]) for r in rows])
    return rows, summary


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _parse_candidate_specs(args.candidate)
    gt_index = _index_images(args.gt_dir)
    if not gt_index:
        raise RuntimeError(f"No GT images found under {args.gt_dir}")

    mask_index = _index_images(args.mask_dir) if args.mask_dir else {}
    anchor_index = _index_images(args.anchor_dir) if args.anchor_dir else {}

    candidate_indices = {name: _index_images(path) for name, path in candidates}
    pairs_by_candidate: Dict[str, List[Dict[str, Optional[Path]]]] = {}
    match_policy_by_candidate: Dict[str, str] = {}
    for name, _path in candidates:
        pairs, resolved_policy = _build_pairs_for_candidate(
            gt_index=gt_index,
            candidate_index=candidate_indices[name],
            mask_index=mask_index,
            anchor_index=anchor_index,
            match_policy=str(args.match_policy),
            llffhold=int(args.llffhold),
        )
        if int(args.limit) > 0:
            pairs = pairs[: int(args.limit)]
        pairs_by_candidate[name] = pairs
        match_policy_by_candidate[name] = resolved_policy

    if not any(pairs_by_candidate.values()):
        raise RuntimeError(
            _empty_match_error(
                gt_index=gt_index,
                candidate_indices=candidate_indices,
                mask_index=mask_index,
                match_policy=str(args.match_policy),
                llffhold=int(args.llffhold),
            )
        )

    all_rows: List[Dict[str, object]] = []
    summary_by_candidate: Dict[str, Dict[str, float]] = {}
    for name, _path in candidates:
        pairs = pairs_by_candidate[name]
        if not pairs:
            print(f"[npse-hf-align-v0] skip {name}: no matched frames")
            continue
        rows, summary = _process_candidate(
            name=name,
            pairs=pairs,
            args=args,
        )
        summary["frames"] = float(len(rows))
        all_rows.extend(rows)
        summary_by_candidate[name] = summary

    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    csv_path = args.output_dir / "per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "version": "evaluate_npse_gt_hf_alignment_v0",
        "gt_dir": str(args.gt_dir),
        "mask_dir": str(args.mask_dir) if args.mask_dir else None,
        "anchor_dir": str(args.anchor_dir) if args.anchor_dir else None,
        "candidates": {name: str(path) for name, path in candidates},
        "highpass_kernel": int(args.highpass_kernel),
        "match_policy": str(args.match_policy),
        "llffhold": int(args.llffhold),
        "resolved_match_policy_by_candidate": match_policy_by_candidate,
        "mask_power": float(args.mask_power),
        "hard_mask_threshold": float(args.hard_mask_threshold),
        "active_gt_percentile": float(args.active_gt_percentile),
        "num_frames_by_candidate": {name: len(pairs) for name, pairs in pairs_by_candidate.items()},
        "summary": summary_by_candidate,
        "per_frame_csv": str(csv_path),
        "debug_dir": str(args.output_dir / "debug"),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[npse-hf-align-v0] match_policy:", args.match_policy)
    print("[npse-hf-align-v0] output:", args.output_dir)
    for name, summary in summary_by_candidate.items():
        abs_corr = summary.get("abs_hf_corr_luma", float("nan"))
        abs_l1 = summary.get("abs_hf_l1_rgb", float("nan"))
        abs_energy = summary.get("abs_hf_energy_ratio", float("nan"))
        delta_corr = summary.get("delta_hf_corr_luma", float("nan"))
        delta_l1 = summary.get("delta_hf_l1_rgb", float("nan"))
        delta_energy = summary.get("delta_hf_energy_ratio", float("nan"))
        print(
            f"  {name}: frames={int(summary.get('frames', 0))} "
            f"abs_corr={abs_corr:.4f} abs_l1={abs_l1:.6f} abs_energy={abs_energy:.3f} "
            f"delta_corr={delta_corr:.4f} delta_l1={delta_l1:.6f} delta_energy={delta_energy:.3f}"
        )
    print("[npse-hf-align-v0] summary:", summary_path)


if __name__ == "__main__":
    main()
