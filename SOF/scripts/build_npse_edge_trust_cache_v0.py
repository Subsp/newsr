#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEPTH_EXTS = IMAGE_EXTS | {".npy", ".npz"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build N-PSE v0 edge/trust/target caches from a low-frequency anchor, "
            "an enhancement-SR prior, and an external depth prior. This is an "
            "offline diagnostic/cache step; it does not train GS directly."
        )
    )
    parser.add_argument("--anchor_dir", type=Path, required=True)
    parser.add_argument("--sr_dir", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_dir",
        type=Path,
        default=None,
        help="Optional reference image dir used for output stems and LLFF train subset matching.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Output cache root containing edge/trust/residual folders and manifest.json.",
    )
    parser.add_argument(
        "--match_policy",
        choices=("stem", "order", "order_if_needed", "llff_train_order"),
        default="stem",
        help=(
            "How to map inputs to output stems. With --reference_dir, llff_train_order "
            "uses idx %% llffhold != 0 reference frames."
        ),
    )
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument(
        "--allow_extra_inputs",
        action="store_true",
        help="Allow sorted input dirs to contain extra trailing frames when order matching.",
    )
    parser.add_argument("--highpass_kernel", type=int, default=15)
    parser.add_argument("--prop_radius", type=int, default=2)
    parser.add_argument("--prop_sigma", type=float, default=1.25)
    parser.add_argument("--depth_edge_percentile", type=float, default=90.0)
    parser.add_argument("--sr_edge_percentile", type=float, default=92.0)
    parser.add_argument("--geometry_edge_threshold", type=float, default=0.55)
    parser.add_argument("--sr_edge_threshold", type=float, default=0.55)
    parser.add_argument(
        "--geometry_confirm_mode",
        choices=("depth_only", "sr_confirmed"),
        default="sr_confirmed",
        help=(
            "depth_only keeps the original v0 behavior. sr_confirmed promotes a depth jump "
            "to a red geometry edge only when an SR structural edge also supports it; "
            "depth-only jumps are marked uncertain and receive weak barrier weight."
        ),
    )
    parser.add_argument(
        "--depth_sr_confirm_threshold",
        type=float,
        default=0.35,
        help="Minimum normalized SR structural edge required to confirm a depth jump as geometry.",
    )
    parser.add_argument(
        "--depth_only_edge_weight",
        type=float,
        default=0.15,
        help="Visualization/fusion weight for depth-only unconfirmed jumps.",
    )
    parser.add_argument(
        "--depth_only_barrier_weight",
        type=float,
        default=0.20,
        help="Weak propagation barrier for unconfirmed depth-only jumps.",
    )
    parser.add_argument(
        "--geometry_barrier_weight",
        type=float,
        default=1.0,
        help="Barrier weight for SR-confirmed geometry depth jumps.",
    )
    parser.add_argument(
        "--edge_position_mode",
        choices=("geometry_or_appearance", "appearance", "sr_strong"),
        default="geometry_or_appearance",
        help=(
            "Which signal defines edge positions / edge bands. "
            "geometry_or_appearance is the original red-or-yellow seed; "
            "appearance uses yellow SR-structure edges only; sr_strong uses all trusted SR edges."
        ),
    )
    parser.add_argument("--sr_barrier_weight", type=float, default=0.25)
    parser.add_argument("--edge_band_radius", type=int, default=1)
    parser.add_argument("--trust_lowfreq_tau", type=float, default=0.12)
    parser.add_argument("--trust_consistency_tau", type=float, default=0.08)
    parser.add_argument("--uncertain_trust_threshold", type=float, default=0.35)
    parser.add_argument(
        "--edge_target_mode",
        choices=("fused", "fidelity"),
        default="fidelity",
        help=(
            "fused uses the full fused edge confidence for edge targets. "
            "fidelity uses SR-position confidence only and preserves the anchor low frequency."
        ),
    )
    parser.add_argument(
        "--edge_target_trust_power",
        type=float,
        default=1.5,
        help="Extra trust sharpening used by --edge_target_mode fidelity.",
    )
    parser.add_argument(
        "--edge_target_min_weight",
        type=float,
        default=0.05,
        help="Drop very weak edge-target weights in fidelity mode.",
    )
    parser.add_argument(
        "--edge_target_direction_gate",
        choices=("none", "anchor_sr_gradient", "anchor_sr_residual_gradient"),
        default="none",
        help=(
            "Optional non-oracle direction gate for fidelity edge targets. "
            "anchor_sr_gradient keeps SR edges whose gradient direction agrees with the anchor. "
            "anchor_sr_residual_gradient also checks the injected residual gradient."
        ),
    )
    parser.add_argument(
        "--edge_target_direction_min_cos",
        type=float,
        default=0.0,
        help="Cosine threshold used by the edge-target direction gate.",
    )
    parser.add_argument(
        "--edge_target_direction_floor",
        type=float,
        default=0.20,
        help="Minimum multiplier for enabled direction gates; keeps the gate conservative.",
    )
    parser.add_argument(
        "--edge_target_direction_power",
        type=float,
        default=1.0,
        help="Power applied to the direction gate before floor blending.",
    )
    parser.add_argument(
        "--edge_target_residual_direction_weight",
        type=float,
        default=0.50,
        help="Residual-gradient contribution for anchor_sr_residual_gradient mode.",
    )
    parser.add_argument(
        "--edge_target_direction_blur",
        type=int,
        default=3,
        help="Optional box blur kernel for smoothing the direction gate.",
    )
    parser.add_argument("--edge_residual_clip", type=float, default=0.08)
    parser.add_argument("--residual_vis_scale", type=float, default=4.0)
    parser.add_argument(
        "--asset_profile",
        choices=("full", "train", "train_no_npz", "continuous"),
        default="full",
        help=(
            "full writes all diagnostic/debug assets. train writes only training "
            "targets/masks plus npz. train_no_npz writes only image training "
            "targets/masks. continuous writes only continuous_target and "
            "trust_continuous for disk-constrained resume."
        ),
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="1-based matched-frame index to start from. Useful for resuming disk-constrained runs.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _is_allowed_file(path: Path, exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def _collect_files(root: Path, exts: set[str]) -> list[Path]:
    files = [p for p in sorted(root.iterdir()) if _is_allowed_file(p, exts)]
    if not files:
        raise FileNotFoundError(f"No supported files found under: {root}")
    return files


def _collect_images(root: Path) -> list[Path]:
    return _collect_files(root, IMAGE_EXTS)


def _index_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in paths:
        if path.stem in out:
            duplicates.setdefault(path.stem, [str(out[path.stem])]).append(str(path))
            continue
        out[path.stem] = path
    if duplicates:
        examples = {k: v[:4] for k, v in list(duplicates.items())[:8]}
        raise ValueError(f"Duplicate {label} stems are ambiguous: {examples}")
    return out


def _selected_reference_paths(reference_paths: list[Path], match_policy: str, llffhold: int) -> tuple[list[Path], list[int], str]:
    if match_policy == "llff_train_order":
        if llffhold <= 0:
            raise ValueError(f"--llffhold must be > 0 for llff_train_order, got {llffhold}")
        selected = [(idx, p) for idx, p in enumerate(reference_paths) if idx % llffhold != 0]
        return [p for _, p in selected], [idx for idx, _ in selected], "llff_train_order"
    return reference_paths, list(range(len(reference_paths))), "all_reference_order"


def _resolve_series_with_reference(
    *,
    label: str,
    paths: list[Path],
    reference_paths: list[Path],
    selected_paths: list[Path],
    selected_indices: list[int],
    match_policy: str,
    allow_extra_inputs: bool,
) -> tuple[dict[str, Path], dict[str, object]]:
    by_stem = _index_by_stem(paths, label)
    selected_stems = [p.stem for p in selected_paths]
    stem_hits = [stem for stem in selected_stems if stem in by_stem]

    if match_policy == "stem" or (match_policy == "order_if_needed" and len(stem_hits) == len(selected_stems)):
        missing = [stem for stem in selected_stems if stem not in by_stem]
        if missing:
            raise FileNotFoundError(f"{label}: missing {len(missing)} selected reference stems, examples={missing[:12]}")
        return {stem: by_stem[stem] for stem in selected_stems}, {
            "label": label,
            "count": len(paths),
            "resolved_policy": "stem",
            "unused_count": max(0, len(paths) - len(selected_stems)),
            "unused_examples": [],
        }

    sorted_paths = sorted(paths)
    if len(sorted_paths) == len(selected_paths):
        return {ref.stem: src for ref, src in zip(selected_paths, sorted_paths, strict=True)}, {
            "label": label,
            "count": len(paths),
            "resolved_policy": "selected_order",
            "unused_count": 0,
            "unused_examples": [],
        }

    if len(sorted_paths) == len(reference_paths):
        out = {
            reference_paths[ref_idx].stem: sorted_paths[ref_idx]
            for ref_idx in selected_indices
        }
        return out, {
            "label": label,
            "count": len(paths),
            "resolved_policy": "full_reference_order_selected",
            "unused_count": len(reference_paths) - len(selected_paths),
            "unused_examples": [str(reference_paths[idx]) for idx, _ in enumerate(reference_paths) if idx not in selected_indices][:20],
        }

    if allow_extra_inputs and len(sorted_paths) > len(selected_paths):
        usable_paths = sorted_paths[: len(selected_paths)]
        unused_paths = sorted_paths[len(selected_paths) :]
        return {ref.stem: src for ref, src in zip(selected_paths, usable_paths, strict=True)}, {
            "label": label,
            "count": len(paths),
            "resolved_policy": "selected_order_allow_extra",
            "unused_count": len(unused_paths),
            "unused_examples": [str(p) for p in unused_paths[:20]],
        }

    raise ValueError(
        f"{label}: cannot match {len(paths)} files to {len(selected_paths)} selected references "
        f"({len(reference_paths)} total references) with policy={match_policy}"
    )


def _resolve_series_without_reference(
    *,
    anchor_paths: list[Path],
    sr_paths: list[Path],
    depth_paths: list[Path],
    match_policy: str,
) -> tuple[list[tuple[str, Path, Path, Path]], dict[str, object]]:
    anchor_by_stem = _index_by_stem(anchor_paths, "anchor")
    sr_by_stem = _index_by_stem(sr_paths, "sr")
    depth_by_stem = _index_by_stem(depth_paths, "depth")
    common = sorted(anchor_by_stem.keys() & sr_by_stem.keys() & depth_by_stem.keys())

    if match_policy in {"stem", "order_if_needed"} and common:
        triples = [(stem, anchor_by_stem[stem], sr_by_stem[stem], depth_by_stem[stem]) for stem in common]
        return triples, {
            "reference_dir": None,
            "resolved_policy": "stem_intersection",
            "num_common": len(common),
        }

    if len(anchor_paths) != len(sr_paths) or len(anchor_paths) != len(depth_paths):
        raise ValueError(
            "Order matching without --reference_dir requires equal counts: "
            f"anchor={len(anchor_paths)} sr={len(sr_paths)} depth={len(depth_paths)}"
        )
    triples = [
        (anchor.stem, anchor, sr, depth)
        for anchor, sr, depth in zip(sorted(anchor_paths), sorted(sr_paths), sorted(depth_paths), strict=True)
    ]
    return triples, {
        "reference_dir": None,
        "resolved_policy": "order",
        "num_common": len(triples),
    }


def _build_triples(args: argparse.Namespace) -> tuple[list[tuple[str, Path, Path, Path]], dict[str, object]]:
    anchor_paths = _collect_files(args.anchor_dir, IMAGE_EXTS)
    sr_paths = _collect_files(args.sr_dir, IMAGE_EXTS)
    depth_paths = _collect_files(args.depth_dir, DEPTH_EXTS)

    if args.reference_dir is None:
        triples, summary = _resolve_series_without_reference(
            anchor_paths=anchor_paths,
            sr_paths=sr_paths,
            depth_paths=depth_paths,
            match_policy=str(args.match_policy),
        )
        summary.update(
            {
                "anchor_count": len(anchor_paths),
                "sr_count": len(sr_paths),
                "depth_count": len(depth_paths),
            }
        )
        return triples, summary

    reference_paths = _collect_images(args.reference_dir)
    selected_paths, selected_indices, selected_policy = _selected_reference_paths(
        reference_paths=reference_paths,
        match_policy=str(args.match_policy),
        llffhold=int(args.llffhold),
    )
    if int(args.limit) > 0:
        selected_paths = selected_paths[: int(args.limit)]
        selected_indices = selected_indices[: int(args.limit)]
    anchor_map, anchor_summary = _resolve_series_with_reference(
        label="anchor",
        paths=anchor_paths,
        reference_paths=reference_paths,
        selected_paths=selected_paths,
        selected_indices=selected_indices,
        match_policy=str(args.match_policy),
        allow_extra_inputs=bool(args.allow_extra_inputs),
    )
    sr_map, sr_summary = _resolve_series_with_reference(
        label="sr",
        paths=sr_paths,
        reference_paths=reference_paths,
        selected_paths=selected_paths,
        selected_indices=selected_indices,
        match_policy=str(args.match_policy),
        allow_extra_inputs=bool(args.allow_extra_inputs),
    )
    depth_map, depth_summary = _resolve_series_with_reference(
        label="depth",
        paths=depth_paths,
        reference_paths=reference_paths,
        selected_paths=selected_paths,
        selected_indices=selected_indices,
        match_policy=str(args.match_policy),
        allow_extra_inputs=bool(args.allow_extra_inputs),
    )
    selected_stems = [p.stem for p in selected_paths]
    triples = [(stem, anchor_map[stem], sr_map[stem], depth_map[stem]) for stem in selected_stems]
    summary = {
        "reference_dir": str(args.reference_dir),
        "reference_count": len(reference_paths),
        "selected_reference_count": len(selected_paths),
        "selected_reference_policy": selected_policy,
        "anchor": anchor_summary,
        "sr": sr_summary,
        "depth": depth_summary,
    }
    return triples, summary


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        image.save(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _save_rgb01(path: Path, rgb: np.ndarray) -> None:
    arr = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    _atomic_save_image(Image.fromarray(arr, mode="RGB"), path)


def _save_gray01(path: Path, gray: np.ndarray) -> None:
    arr = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    _atomic_save_image(Image.fromarray(arr, mode="L"), path)


def _load_rgb01(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if size is not None and rgb.size != size:
            rgb = rgb.resize(size, resample=Image.Resampling.BICUBIC)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _load_npz_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.ndarray):
        return data
    preferred = ("depth", "pred", "prediction", "arr_0")
    for key in preferred:
        if key in data:
            return data[key]
    return data[sorted(data.files)[0]]


def _load_depth(path: Path, size: tuple[int, int]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        depth = _load_npz_array(path)
    else:
        with Image.open(path) as image:
            arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]
        depth = arr

    depth = np.asarray(depth, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Depth prior must resolve to HxW, got shape={depth.shape} for {path}")
    if (depth.shape[1], depth.shape[0]) != size:
        finite = np.isfinite(depth)
        fill = float(np.nanmedian(depth[finite])) if np.any(finite) else 0.0
        depth = np.where(finite, depth, fill).astype(np.float32)
        image = Image.fromarray(depth, mode="F").resize(size, resample=Image.Resampling.BILINEAR)
        depth = np.asarray(image, dtype=np.float32)
    return depth


def _robust01(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    p_lo, p_hi = np.percentile(vals, [lo, hi])
    if p_hi <= p_lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (np.where(finite, x, p_lo) - p_lo) / (p_hi - p_lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _box_blur_2d(x: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return x.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(x.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    out = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return (out / float(k * k)).astype(np.float32)


def _box_blur(x: np.ndarray, kernel: int) -> np.ndarray:
    if x.ndim == 2:
        return _box_blur_2d(x, kernel)
    return np.stack([_box_blur_2d(x[..., c], kernel) for c in range(x.shape[-1])], axis=-1)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    gx[:, 0] = x[:, 1] - x[:, 0]
    gx[:, -1] = x[:, -1] - x[:, -2]
    gy[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    gy[0, :] = x[1, :] - x[0, :]
    gy[-1, :] = x[-1, :] - x[-2, :]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _gradient_xy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    gx[:, 0] = x[:, 1] - x[:, 0]
    gx[:, -1] = x[:, -1] - x[:, -2]
    gy[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    gy[0, :] = x[1, :] - x[0, :]
    gy[-1, :] = x[-1, :] - x[-2, :]
    return gx.astype(np.float32), gy.astype(np.float32)


def _gradient_direction_gate(
    gx_a: np.ndarray,
    gy_a: np.ndarray,
    gx_b: np.ndarray,
    gy_b: np.ndarray,
    *,
    min_cos: float,
) -> np.ndarray:
    mag_a = np.sqrt(gx_a * gx_a + gy_a * gy_a)
    mag_b = np.sqrt(gx_b * gx_b + gy_b * gy_b)
    cos = (gx_a * gx_b + gy_a * gy_b) / np.maximum(mag_a * mag_b, 1e-8)
    lo = min(max(float(min_cos), -1.0), 0.999)
    gate = np.clip((cos - lo) / max(1.0 - lo, 1e-6), 0.0, 1.0)
    strength = np.minimum(
        _normalize_by_percentile(mag_a, 90.0),
        _normalize_by_percentile(mag_b, 90.0),
    )
    return np.clip(gate * strength, 0.0, 1.0).astype(np.float32)


def _compute_edge_direction_gate(
    anchor: np.ndarray,
    sr: np.ndarray,
    residual_raw: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    mode = str(args.edge_target_direction_gate)
    h, w = anchor.shape[:2]
    if mode == "none":
        return np.ones((h, w), dtype=np.float32)

    min_cos = float(args.edge_target_direction_min_cos)
    anchor_luma = _luma(anchor)
    sr_luma = _luma(sr)
    residual_luma = _luma(residual_raw)
    agx, agy = _gradient_xy(anchor_luma)
    sgx, sgy = _gradient_xy(sr_luma)
    rgx, rgy = _gradient_xy(residual_luma)

    anchor_sr_gate = _gradient_direction_gate(agx, agy, sgx, sgy, min_cos=min_cos)
    gate = anchor_sr_gate
    if mode == "anchor_sr_residual_gradient":
        residual_sr_gate = _gradient_direction_gate(rgx, rgy, sgx, sgy, min_cos=min_cos)
        residual_weight = np.clip(float(args.edge_target_residual_direction_weight), 0.0, 1.0)
        gate = anchor_sr_gate * ((1.0 - residual_weight) + residual_weight * residual_sr_gate)

    blur_kernel = int(args.edge_target_direction_blur)
    if blur_kernel > 1:
        gate = _box_blur(gate, blur_kernel)
    power = max(float(args.edge_target_direction_power), 0.0)
    if power != 1.0:
        gate = np.clip(gate, 0.0, 1.0) ** power
    floor = np.clip(float(args.edge_target_direction_floor), 0.0, 1.0)
    gate = floor + (1.0 - floor) * np.clip(gate, 0.0, 1.0)
    return gate.astype(np.float32, copy=False)


def _normalize_by_percentile(x: np.ndarray, percentile: float) -> np.ndarray:
    vals = x[np.isfinite(x)]
    vals = vals[vals > 1e-8]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    scale = float(np.percentile(vals, percentile))
    if scale <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / scale, 0.0, 1.0).astype(np.float32)


def _shift(x: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(x, fill)
    h, w = x.shape[:2]
    src_y0 = max(0, dy)
    src_y1 = h + min(0, dy)
    dst_y0 = max(0, -dy)
    dst_y1 = h - max(0, dy)
    src_x0 = max(0, dx)
    src_x1 = w + min(0, dx)
    dst_x0 = max(0, -dx)
    dst_x1 = w - max(0, dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[dst_y0:dst_y1, dst_x0:dst_x1, ...] = x[src_y0:src_y1, src_x0:src_x1, ...]
    return out


def _shift_valid(shape: tuple[int, int], dy: int, dx: int) -> np.ndarray:
    h, w = shape
    valid = np.zeros((h, w), dtype=np.float32)
    src_y0 = max(0, dy)
    src_y1 = h + min(0, dy)
    dst_y0 = max(0, -dy)
    dst_y1 = h - max(0, dy)
    src_x0 = max(0, dx)
    src_x1 = w + min(0, dx)
    dst_x0 = max(0, -dx)
    dst_x1 = w - max(0, dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        valid[dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
    return valid


def _max_filter(mask: np.ndarray, radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r <= 0:
        return mask.astype(np.float32)
    out = np.zeros_like(mask, dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, _shift(mask, dy, dx, fill=0.0))
    return out


def _edge_stopped_propagation(
    residual: np.ndarray,
    trust: np.ndarray,
    barrier: np.ndarray,
    radius: int,
    sigma: float,
) -> np.ndarray:
    h, w = trust.shape
    numerator = np.zeros_like(residual, dtype=np.float32)
    denom = np.zeros((h, w), dtype=np.float32)
    sigma = max(float(sigma), 1e-4)
    r = max(0, int(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            nb_residual = _shift(residual, dy, dx, fill=0.0)
            nb_trust = _shift(trust, dy, dx, fill=0.0)
            nb_barrier = _shift(barrier, dy, dx, fill=1.0)
            valid = _shift_valid((h, w), dy, dx)
            connection = np.clip(1.0 - np.maximum(barrier, nb_barrier), 0.0, 1.0)
            weight = (spatial * nb_trust * connection * valid).astype(np.float32)
            numerator += nb_residual * weight[..., None]
            denom += weight
    propagated = residual.copy()
    good = denom > 1e-6
    propagated[good] = numerator[good] / denom[good, None]
    return propagated.astype(np.float32)


def _edge_type_rgb(edge_type: np.ndarray) -> np.ndarray:
    # 0 continuous black, 1 geometry red, 2 appearance yellow, 3 uncertain blue.
    palette = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.08, 0.02],
            [1.0, 0.85, 0.05],
            [0.05, 0.25, 1.0],
        ],
        dtype=np.float32,
    )
    return palette[np.clip(edge_type.astype(np.int64), 0, 3)]


def _overlay_edges(base: np.ndarray, edge: np.ndarray, trust: np.ndarray) -> np.ndarray:
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    alpha = np.clip(edge * 0.70, 0.0, 0.70)[..., None]
    overlay = base * (1.0 - alpha) + red * alpha
    # Trust is shown as a subtle green tint in continuous areas.
    green = np.zeros_like(base)
    green[..., 1] = 1.0
    trust_alpha = np.clip(trust * (1.0 - edge) * 0.18, 0.0, 0.18)[..., None]
    return np.clip(overlay * (1.0 - trust_alpha) + green * trust_alpha, 0.0, 1.0)


def _process_frame(
    *,
    stem: str,
    anchor_path: Path,
    sr_path: Path,
    depth_path: Path,
    args: argparse.Namespace,
    dirs: dict[str, Path],
) -> dict[str, object]:
    anchor = _load_rgb01(anchor_path)
    size = (anchor.shape[1], anchor.shape[0])
    sr = _load_rgb01(sr_path, size=size)
    depth = _load_depth(depth_path, size=size)
    depth01 = _robust01(depth)

    hp_kernel = max(1, int(args.highpass_kernel))
    anchor_low = _box_blur(anchor, hp_kernel)
    sr_low = _box_blur(sr, hp_kernel)
    residual_raw = (sr - sr_low) - (anchor - anchor_low)

    edge_depth = _normalize_by_percentile(_gradient_magnitude(depth01), float(args.depth_edge_percentile))
    sr_luma = _luma(sr)
    sr_edge_input = np.maximum(_gradient_magnitude(sr_luma), np.abs(sr_luma - _box_blur(sr_luma, 5)))
    edge_sr = _normalize_by_percentile(sr_edge_input, float(args.sr_edge_percentile))

    lowfreq_diff = np.mean(np.abs(sr_low - anchor_low), axis=-1)
    trust_low = np.exp(-lowfreq_diff / max(float(args.trust_lowfreq_tau), 1e-6))
    residual_luma = _luma(np.clip(0.5 + residual_raw, 0.0, 1.0))
    mean_r = _box_blur(residual_luma, 5)
    mean_r2 = _box_blur(residual_luma * residual_luma, 5)
    consistency = np.sqrt(np.maximum(mean_r2 - mean_r * mean_r, 0.0))
    trust_cons = np.exp(-consistency / max(float(args.trust_consistency_tau), 1e-6))
    trust_sr = np.clip(trust_low * trust_cons, 0.0, 1.0).astype(np.float32)

    geo_candidate = edge_depth >= float(args.geometry_edge_threshold)
    sr_support = edge_sr >= float(args.depth_sr_confirm_threshold)
    if str(args.geometry_confirm_mode) == "sr_confirmed":
        geo = geo_candidate & sr_support
        depth_only_uncertain = geo_candidate & (~sr_support)
    else:
        geo = geo_candidate
        depth_only_uncertain = np.zeros_like(geo, dtype=bool)
    sr_strong = edge_sr >= float(args.sr_edge_threshold)
    uncertain_sr = sr_strong & (~geo) & (trust_sr < float(args.uncertain_trust_threshold))
    uncertain = uncertain_sr | depth_only_uncertain
    appearance = sr_strong & (~geo) & (~uncertain)

    edge_type = np.zeros_like(edge_depth, dtype=np.uint8)
    edge_type[geo] = 1
    edge_type[appearance] = 2
    edge_type[uncertain] = 3

    edge_depth_confirmed = np.clip(
        edge_depth * geo.astype(np.float32)
        + edge_depth * depth_only_uncertain.astype(np.float32) * float(args.depth_only_edge_weight),
        0.0,
        1.0,
    )
    edge_fused = np.clip(np.maximum(edge_depth_confirmed, edge_sr * trust_sr), 0.0, 1.0)
    depth_barrier = np.clip(
        edge_depth * geo.astype(np.float32) * float(args.geometry_barrier_weight)
        + edge_depth * depth_only_uncertain.astype(np.float32) * float(args.depth_only_barrier_weight),
        0.0,
        1.0,
    )
    barrier = np.clip(np.maximum(depth_barrier, edge_sr * trust_sr * float(args.sr_barrier_weight)), 0.0, 1.0)
    if str(args.edge_position_mode) == "appearance":
        edge_seed = appearance.astype(np.float32)
    elif str(args.edge_position_mode) == "sr_strong":
        edge_seed = (sr_strong & (~uncertain_sr)).astype(np.float32)
    else:
        edge_seed = np.maximum(geo.astype(np.float32), appearance.astype(np.float32))
    edge_band = _max_filter(edge_seed, int(args.edge_band_radius))
    continuous_mask = ((edge_band < 0.5) & (trust_sr > 0.05)).astype(np.float32)

    residual_npse = _edge_stopped_propagation(
        residual=residual_raw,
        trust=trust_sr * continuous_mask,
        barrier=barrier,
        radius=int(args.prop_radius),
        sigma=float(args.prop_sigma),
    )
    residual_npse = residual_npse * continuous_mask[..., None] + residual_raw * (1.0 - continuous_mask[..., None])
    trust_continuous = (trust_sr * continuous_mask).astype(np.float32, copy=False)
    continuous_target = np.clip(anchor + residual_npse * continuous_mask[..., None], 0.0, 1.0)

    if str(args.edge_target_mode) == "fidelity":
        trust_edge = edge_band * edge_sr * np.power(
            np.clip(trust_sr, 0.0, 1.0),
            max(float(args.edge_target_trust_power), 0.0),
        )
        trust_edge = np.where(
            trust_edge >= float(args.edge_target_min_weight),
            trust_edge,
            np.zeros_like(trust_edge, dtype=np.float32),
        ).astype(np.float32, copy=False)
    else:
        trust_edge = edge_fused * edge_band
    trust_edge_raw = trust_edge.astype(np.float32, copy=True)
    edge_direction_gate = _compute_edge_direction_gate(anchor, sr, residual_raw, args)
    if str(args.edge_target_direction_gate) != "none":
        trust_edge = (trust_edge * edge_direction_gate).astype(np.float32, copy=False)
        trust_edge = np.where(
            trust_edge >= float(args.edge_target_min_weight),
            trust_edge,
            np.zeros_like(trust_edge, dtype=np.float32),
        ).astype(np.float32, copy=False)
    edge_residual = np.clip(
        residual_raw,
        -float(args.edge_residual_clip),
        float(args.edge_residual_clip),
    ) * trust_edge[..., None]
    edge_target = np.clip(anchor + edge_residual, 0.0, 1.0)

    if "edge_depth" in dirs:
        _save_gray01(dirs["edge_depth"] / f"{stem}.png", edge_depth)
    if "edge_depth_confirmed" in dirs:
        _save_gray01(dirs["edge_depth_confirmed"] / f"{stem}.png", edge_depth_confirmed)
    if "edge_sr" in dirs:
        _save_gray01(dirs["edge_sr"] / f"{stem}.png", edge_sr)
    if "edge_fused" in dirs:
        _save_gray01(dirs["edge_fused"] / f"{stem}.png", edge_fused)
    if "edge_position" in dirs:
        _save_gray01(dirs["edge_position"] / f"{stem}.png", edge_seed)
    if "edge_band" in dirs:
        _save_gray01(dirs["edge_band"] / f"{stem}.png", edge_band)
    if "depth_only_uncertain" in dirs:
        _save_gray01(dirs["depth_only_uncertain"] / f"{stem}.png", depth_only_uncertain.astype(np.float32))
    if "barrier" in dirs:
        _save_gray01(dirs["barrier"] / f"{stem}.png", barrier)
    if "trust_sr" in dirs:
        _save_gray01(dirs["trust_sr"] / f"{stem}.png", trust_sr)
    if "trust_edge_raw" in dirs:
        _save_gray01(dirs["trust_edge_raw"] / f"{stem}.png", trust_edge_raw)
    if "edge_direction_gate" in dirs:
        _save_gray01(dirs["edge_direction_gate"] / f"{stem}.png", edge_direction_gate)
    if "trust_edge" in dirs:
        _save_gray01(dirs["trust_edge"] / f"{stem}.png", trust_edge)
    if "continuous_mask" in dirs:
        _save_gray01(dirs["continuous_mask"] / f"{stem}.png", continuous_mask)
    if "trust_continuous" in dirs:
        _save_gray01(dirs["trust_continuous"] / f"{stem}.png", trust_continuous)
    if "edge_type" in dirs:
        _save_rgb01(dirs["edge_type"] / f"{stem}.png", _edge_type_rgb(edge_type))
    if "residual_raw" in dirs:
        _save_rgb01(
            dirs["residual_raw"] / f"{stem}.png",
            np.clip(0.5 + residual_raw * float(args.residual_vis_scale), 0.0, 1.0),
        )
    if "residual_npse" in dirs:
        _save_rgb01(
            dirs["residual_npse"] / f"{stem}.png",
            np.clip(0.5 + residual_npse * float(args.residual_vis_scale), 0.0, 1.0),
        )
    if "edge_target" in dirs:
        _save_rgb01(dirs["edge_target"] / f"{stem}.png", edge_target)
    if "continuous_target" in dirs:
        _save_rgb01(dirs["continuous_target"] / f"{stem}.png", continuous_target)
    if "debug_overlay" in dirs:
        _save_rgb01(dirs["debug_overlay"] / f"{stem}.png", _overlay_edges(sr, edge_fused, trust_sr))

    npz_path = None
    if "npz" in dirs:
        npz_path = dirs["npz"] / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            residual_raw=residual_raw.astype(np.float16),
            residual_npse=residual_npse.astype(np.float16),
            edge_depth=edge_depth.astype(np.float16),
            edge_depth_confirmed=edge_depth_confirmed.astype(np.float16),
            edge_sr=edge_sr.astype(np.float16),
            edge_fused=edge_fused.astype(np.float16),
            edge_position=edge_seed.astype(np.float16),
            edge_type=edge_type,
            geometry_edge_candidate=geo_candidate.astype(np.uint8),
            depth_only_uncertain=depth_only_uncertain.astype(np.uint8),
            barrier=barrier.astype(np.float16),
            trust_sr=trust_sr.astype(np.float16),
            trust_edge_raw=trust_edge_raw.astype(np.float16),
            edge_direction_gate=edge_direction_gate.astype(np.float16),
            trust_edge=trust_edge.astype(np.float16),
            continuous_mask=continuous_mask.astype(np.float16),
            trust_continuous=trust_continuous.astype(np.float16),
            edge_band=edge_band.astype(np.float16),
            edge_target=edge_target.astype(np.float16),
            continuous_target=continuous_target.astype(np.float16),
        )

    return {
        "stem": stem,
        "anchor_path": str(anchor_path),
        "sr_path": str(sr_path),
        "depth_path": str(depth_path),
        "height": int(anchor.shape[0]),
        "width": int(anchor.shape[1]),
        "edge_depth_mean": float(edge_depth.mean()),
        "edge_sr_mean": float(edge_sr.mean()),
        "edge_fused_mean": float(edge_fused.mean()),
        "trust_sr_mean": float(trust_sr.mean()),
        "trust_edge_raw_mean": float(trust_edge_raw.mean()),
        "edge_direction_gate_mean": float(edge_direction_gate.mean()),
        "trust_edge_mean": float(trust_edge.mean()),
        "continuous_ratio": float(continuous_mask.mean()),
        "trust_continuous_mean": float(trust_continuous.mean()),
        "edge_position_ratio": float(edge_seed.mean()),
        "edge_band_ratio": float(edge_band.mean()),
        "geometry_candidate_ratio": float(geo_candidate.mean()),
        "geometry_edge_ratio": float(geo.mean()),
        "depth_only_uncertain_ratio": float(depth_only_uncertain.mean()),
        "appearance_edge_ratio": float(appearance.mean()),
        "uncertain_edge_ratio": float(uncertain.mean()),
        "npz": None if npz_path is None else str(npz_path),
    }


def main() -> None:
    args = _parse_args()
    args.anchor_dir = args.anchor_dir.expanduser().resolve()
    args.sr_dir = args.sr_dir.expanduser().resolve()
    args.depth_dir = args.depth_dir.expanduser().resolve()
    args.reference_dir = args.reference_dir.expanduser().resolve() if args.reference_dir else None
    args.output_root = args.output_root.expanduser().resolve()

    for label in ("anchor_dir", "sr_dir", "depth_dir"):
        path = getattr(args, label)
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.reference_dir is not None and not args.reference_dir.is_dir():
        raise FileNotFoundError(f"reference_dir not found: {args.reference_dir}")

    triples, match_summary = _build_triples(args)
    original_num_triples = len(triples)
    if int(args.start_index) < 1:
        raise ValueError(f"--start_index must be >= 1, got {args.start_index}")
    if int(args.start_index) > 1:
        triples = triples[int(args.start_index) - 1 :]
    if int(args.limit) > 0:
        triples = triples[: int(args.limit)]
    if not triples:
        raise RuntimeError("No matched frames to process.")
    if int(args.start_index) > 1:
        print(
            f"[npse-cache-v0] resume from matched index {int(args.start_index)}/"
            f"{original_num_triples}"
        )

    full_asset_names = (
        "edge_depth",
        "edge_depth_confirmed",
        "edge_sr",
        "edge_fused",
        "edge_position",
        "edge_band",
        "depth_only_uncertain",
        "edge_type",
        "barrier",
        "trust_sr",
        "trust_edge_raw",
        "edge_direction_gate",
        "trust_edge",
        "continuous_mask",
        "trust_continuous",
        "residual_raw",
        "residual_npse",
        "edge_target",
        "continuous_target",
        "debug_overlay",
        "npz",
    )
    train_asset_names = (
        "edge_target",
        "trust_edge",
        "continuous_target",
        "trust_continuous",
        "npz",
    )
    train_no_npz_asset_names = (
        "edge_target",
        "trust_edge",
        "continuous_target",
        "trust_continuous",
    )
    continuous_asset_names = (
        "continuous_target",
        "trust_continuous",
    )
    if str(args.asset_profile) == "full":
        asset_names = full_asset_names
    elif str(args.asset_profile) == "train":
        asset_names = train_asset_names
    elif str(args.asset_profile) == "train_no_npz":
        asset_names = train_no_npz_asset_names
    else:
        asset_names = continuous_asset_names
    output_dirs = {name: args.output_root / name for name in asset_names}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    for idx, (stem, anchor_path, sr_path, depth_path) in enumerate(triples, start=1):
        required_outputs = [path / f"{stem}.png" for name, path in output_dirs.items() if name != "npz"]
        if "npz" in output_dirs:
            required_outputs.append(output_dirs["npz"] / f"{stem}.npz")
        if required_outputs and all(path.exists() for path in required_outputs) and not bool(args.overwrite):
            print(f"[npse-cache-v0] skip existing {idx}/{len(triples)} {stem}")
            continue
        print(f"[npse-cache-v0] {idx}/{len(triples)} {stem}")
        frames.append(
            _process_frame(
                stem=stem,
                anchor_path=anchor_path,
                sr_path=sr_path,
                depth_path=depth_path,
                args=args,
                dirs=output_dirs,
            )
        )

    manifest = {
        "version": "build_npse_edge_trust_cache_v0",
        "anchor_dir": str(args.anchor_dir),
        "sr_dir": str(args.sr_dir),
        "depth_dir": str(args.depth_dir),
        "reference_dir": None if args.reference_dir is None else str(args.reference_dir),
        "output_root": str(args.output_root),
        "match_policy": str(args.match_policy),
        "llffhold": int(args.llffhold),
        "allow_extra_inputs": bool(args.allow_extra_inputs),
        "highpass_kernel": int(args.highpass_kernel),
        "prop_radius": int(args.prop_radius),
        "prop_sigma": float(args.prop_sigma),
        "depth_edge_percentile": float(args.depth_edge_percentile),
        "sr_edge_percentile": float(args.sr_edge_percentile),
        "geometry_edge_threshold": float(args.geometry_edge_threshold),
        "sr_edge_threshold": float(args.sr_edge_threshold),
        "geometry_confirm_mode": str(args.geometry_confirm_mode),
        "depth_sr_confirm_threshold": float(args.depth_sr_confirm_threshold),
        "depth_only_edge_weight": float(args.depth_only_edge_weight),
        "depth_only_barrier_weight": float(args.depth_only_barrier_weight),
        "geometry_barrier_weight": float(args.geometry_barrier_weight),
        "edge_position_mode": str(args.edge_position_mode),
        "edge_target_mode": str(args.edge_target_mode),
        "edge_target_trust_power": float(args.edge_target_trust_power),
        "edge_target_min_weight": float(args.edge_target_min_weight),
        "edge_target_direction_gate": str(args.edge_target_direction_gate),
        "edge_target_direction_min_cos": float(args.edge_target_direction_min_cos),
        "edge_target_direction_floor": float(args.edge_target_direction_floor),
        "edge_target_direction_power": float(args.edge_target_direction_power),
        "edge_target_residual_direction_weight": float(args.edge_target_residual_direction_weight),
        "edge_target_direction_blur": int(args.edge_target_direction_blur),
        "edge_residual_clip": float(args.edge_residual_clip),
        "asset_profile": str(args.asset_profile),
        "start_index": int(args.start_index),
        "match_summary": match_summary,
        "num_total_matched": int(original_num_triples),
        "num_requested": len(triples),
        "num_written": len(frames),
        "frames": frames,
        "summary_stats": {
            "trust_sr_mean": None if not frames else float(np.mean([f["trust_sr_mean"] for f in frames])),
            "trust_edge_raw_mean": None if not frames else float(np.mean([f["trust_edge_raw_mean"] for f in frames])),
            "edge_direction_gate_mean": None if not frames else float(np.mean([f["edge_direction_gate_mean"] for f in frames])),
            "trust_edge_mean": None if not frames else float(np.mean([f["trust_edge_mean"] for f in frames])),
            "continuous_ratio_mean": None if not frames else float(np.mean([f["continuous_ratio"] for f in frames])),
            "trust_continuous_mean": None if not frames else float(np.mean([f["trust_continuous_mean"] for f in frames])),
            "edge_position_ratio_mean": None if not frames else float(np.mean([f["edge_position_ratio"] for f in frames])),
            "edge_band_ratio_mean": None if not frames else float(np.mean([f["edge_band_ratio"] for f in frames])),
            "geometry_candidate_ratio_mean": None if not frames else float(np.mean([f["geometry_candidate_ratio"] for f in frames])),
            "geometry_edge_ratio_mean": None if not frames else float(np.mean([f["geometry_edge_ratio"] for f in frames])),
            "depth_only_uncertain_ratio_mean": None if not frames else float(np.mean([f["depth_only_uncertain_ratio"] for f in frames])),
            "appearance_edge_ratio_mean": None if not frames else float(np.mean([f["appearance_edge_ratio"] for f in frames])),
            "uncertain_edge_ratio_mean": None if not frames else float(np.mean([f["uncertain_edge_ratio"] for f in frames])),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary_stats"], indent=2))
    print(f"[npse-cache-v0] manifest: {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
