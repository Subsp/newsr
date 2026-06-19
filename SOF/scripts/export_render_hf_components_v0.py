#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export image-domain high-frequency components from rendered views."
    )
    parser.add_argument("--render_dir", required=True, help="Rendered RGB directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for exported HF maps.")
    parser.add_argument("--target_dir", default="", help="Optional target RGB directory, e.g. GT or edge_target.")
    parser.add_argument("--anchor_dir", default="", help="Optional anchor RGB directory for residual HF.")
    parser.add_argument("--mask_dir", default="", help="Optional soft mask directory.")
    parser.add_argument(
        "--match_policy",
        default="order_if_needed",
        choices=["stem", "order", "order_if_needed"],
        help="How to match render/target/anchor/mask images.",
    )
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--residual_clip", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sheet_limit", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path, *, required: bool = True) -> List[Path]:
    if not root or str(root) == ".":
        return []
    if not root.is_dir():
        if required:
            raise FileNotFoundError(f"Image directory not found: {root}")
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(
    render_paths: Sequence[Path],
    other_paths: Sequence[Path],
    *,
    idx: int,
    render_path: Path,
    lookup: Dict[str, Path],
    match_policy: str,
) -> Optional[Path]:
    if not other_paths:
        return None
    if match_policy in {"stem", "order_if_needed"}:
        matched = lookup.get(render_path.stem.lower())
        if matched is not None:
            return matched
        if match_policy == "stem":
            return None
    if match_policy in {"order", "order_if_needed"} and idx < len(other_paths):
        return other_paths[idx]
    return None


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _luma(rgb: np.ndarray) -> np.ndarray:
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


def _highpass(gray: np.ndarray, kernel: int) -> np.ndarray:
    return (gray - _box_blur(gray, int(kernel))).astype(np.float32)


def _norm_abs(arr: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.abs(np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    scale = float(np.percentile(arr, float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _norm_signed(arr: np.ndarray, percentile: float, clip: float) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(clip) > 0.0:
        arr = np.clip(arr, -float(clip), float(clip))
    scale = float(np.percentile(np.abs(arr), float(percentile)))
    if scale <= 1e-8:
        return np.full((*arr.shape, 3), 0.5, dtype=np.float32)
    signed = np.clip(arr / scale, -1.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(signed, 0.0, 1.0)
    rgb[..., 2] = np.clip(-signed, 0.0, 1.0)
    rgb += 0.5 * (1.0 - np.abs(signed))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _overlay(render_hf: np.ndarray, target_hf: Optional[np.ndarray], mask: Optional[np.ndarray]) -> np.ndarray:
    if target_hf is None:
        return np.repeat(render_hf[..., None], 3, axis=2)
    rgb = np.zeros((*render_hf.shape, 3), dtype=np.float32)
    rgb[..., 0] = target_hf
    rgb[..., 1] = render_hf
    rgb[..., 2] = render_hf
    if mask is not None:
        rgb = rgb * (0.2 + 0.8 * np.clip(mask[..., None], 0.0, 1.0))
    return np.clip(rgb, 0.0, 1.0)


def _label_panel(rgb: np.ndarray, label: str) -> Image.Image:
    arr = np.clip(rgb, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, min(img.width, 360), 22), fill=(0, 0, 0))
    draw.text((5, 4), label, fill=(255, 255, 255))
    return img


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0.0, 1.0)[..., None], 3, axis=2)


def _write_sheet(
    path: Path,
    *,
    render_rgb: np.ndarray,
    render_hf: np.ndarray,
    render_residual: Optional[np.ndarray],
    target_rgb: Optional[np.ndarray],
    target_hf: Optional[np.ndarray],
    target_residual: Optional[np.ndarray],
    error: Optional[np.ndarray],
    overlay: np.ndarray,
) -> None:
    panels = [
        _label_panel(render_rgb, "render RGB"),
        _label_panel(_gray_to_rgb(render_hf), "abs HP(render)"),
        _label_panel(overlay, "overlay target=red render=cyan"),
    ]
    if render_residual is not None:
        panels.append(_label_panel(_gray_to_rgb(render_residual), "abs residual HP(render-anchor)"))
    if target_rgb is not None:
        panels.append(_label_panel(target_rgb, "target RGB"))
    if target_hf is not None:
        panels.append(_label_panel(_gray_to_rgb(target_hf), "abs HP(target)"))
    if target_residual is not None:
        panels.append(_label_panel(_gray_to_rgb(target_residual), "abs residual HP(target-anchor)"))
    if error is not None:
        panels.append(_label_panel(_gray_to_rgb(error), "abs residual error"))

    width = max(p.width for p in panels)
    height = max(p.height for p in panels)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * width, (i // cols) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    args = _parse_args()
    render_dir = Path(args.render_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve() if str(args.target_dir).strip() else None
    anchor_dir = Path(args.anchor_dir).expanduser().resolve() if str(args.anchor_dir).strip() else None
    mask_dir = Path(args.mask_dir).expanduser().resolve() if str(args.mask_dir).strip() else None

    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    render_paths = _list_images(render_dir)
    if int(args.limit) > 0:
        render_paths = render_paths[: int(args.limit)]
    target_paths = _list_images(target_dir, required=False) if target_dir is not None else []
    anchor_paths = _list_images(anchor_dir, required=False) if anchor_dir is not None else []
    mask_paths = _list_images(mask_dir, required=False) if mask_dir is not None else []
    target_lookup = _lookup(target_paths)
    anchor_lookup = _lookup(anchor_paths)
    mask_lookup = _lookup(mask_paths)

    frames = []
    render_energy = []
    target_energy = []
    residual_energy = []
    residual_error = []

    print(f"[render-hf-v0] render : {render_dir}")
    print(f"[render-hf-v0] target : {target_dir if target_dir is not None else '<none>'}")
    print(f"[render-hf-v0] anchor : {anchor_dir if anchor_dir is not None else '<none>'}")
    print(f"[render-hf-v0] mask   : {mask_dir if mask_dir is not None else '<none>'}")
    print(f"[render-hf-v0] output : {output_dir}")
    print(f"[render-hf-v0] frames : {len(render_paths)}")

    for idx, render_path in enumerate(tqdm(render_paths, desc="export render HF")):
        target_path = _resolve(
            render_paths,
            target_paths,
            idx=idx,
            render_path=render_path,
            lookup=target_lookup,
            match_policy=str(args.match_policy),
        )
        anchor_path = _resolve(
            render_paths,
            anchor_paths,
            idx=idx,
            render_path=render_path,
            lookup=anchor_lookup,
            match_policy=str(args.match_policy),
        )
        mask_path = _resolve(
            render_paths,
            mask_paths,
            idx=idx,
            render_path=render_path,
            lookup=mask_lookup,
            match_policy=str(args.match_policy),
        )

        render_rgb = _load_rgb(render_path)
        size = (render_rgb.shape[1], render_rgb.shape[0])
        target_rgb = _load_rgb(target_path, size=size) if target_path is not None else None
        anchor_rgb = _load_rgb(anchor_path, size=size) if anchor_path is not None else None
        mask = _load_gray(mask_path, size=size) if mask_path is not None else None

        render_hp_signed = _highpass(_luma(render_rgb), int(args.highpass_kernel))
        render_hf = _norm_abs(render_hp_signed, float(args.norm_percentile))
        target_hp_signed = None
        target_hf = None
        if target_rgb is not None:
            target_hp_signed = _highpass(_luma(target_rgb), int(args.highpass_kernel))
            target_hf = _norm_abs(target_hp_signed, float(args.norm_percentile))

        render_residual_norm = None
        target_residual_norm = None
        error_norm = None
        if anchor_rgb is not None:
            anchor_hp_signed = _highpass(_luma(anchor_rgb), int(args.highpass_kernel))
            render_residual = render_hp_signed - anchor_hp_signed
            render_residual_norm = _norm_abs(render_residual, float(args.norm_percentile))
            residual_energy.append(float(np.mean(np.abs(render_residual))))
            _save_gray(output_dir / "render_residual_hf_abs" / render_path.name, render_residual_norm)
            _save_rgb(
                output_dir / "render_residual_hf_signed" / render_path.name,
                _norm_signed(render_residual, float(args.norm_percentile), float(args.residual_clip)),
            )
            if target_hp_signed is not None:
                target_residual = target_hp_signed - anchor_hp_signed
                target_residual_norm = _norm_abs(target_residual, float(args.norm_percentile))
                error_norm = _norm_abs(render_residual - target_residual, float(args.norm_percentile))
                residual_error.append(float(np.mean(np.abs(render_residual - target_residual))))
                _save_gray(output_dir / "target_residual_hf_abs" / render_path.name, target_residual_norm)
                _save_rgb(
                    output_dir / "target_residual_hf_signed" / render_path.name,
                    _norm_signed(target_residual, float(args.norm_percentile), float(args.residual_clip)),
                )
                _save_gray(output_dir / "residual_error_abs" / render_path.name, error_norm)

        overlay_target = target_residual_norm if target_residual_norm is not None else target_hf
        overlay_render = render_residual_norm if render_residual_norm is not None else render_hf
        overlay = _overlay(overlay_render, overlay_target, mask)

        _save_gray(output_dir / "render_hf_abs" / render_path.name, render_hf)
        _save_rgb(output_dir / "render_hf_signed" / render_path.name, _norm_signed(render_hp_signed, float(args.norm_percentile), 0.0))
        _save_rgb(output_dir / "overlay" / render_path.name, overlay)
        if target_hf is not None:
            _save_gray(output_dir / "target_hf_abs" / render_path.name, target_hf)
        if mask is not None:
            _save_gray(output_dir / "mask" / render_path.name, mask)
        if idx < int(args.sheet_limit):
            _write_sheet(
                output_dir / "sheet" / render_path.name,
                render_rgb=render_rgb,
                render_hf=render_hf,
                render_residual=render_residual_norm,
                target_rgb=target_rgb,
                target_hf=target_hf,
                target_residual=target_residual_norm,
                error=error_norm,
                overlay=overlay,
            )

        render_energy.append(float(np.mean(np.abs(render_hp_signed))))
        if target_hp_signed is not None:
            target_energy.append(float(np.mean(np.abs(target_hp_signed))))
        frames.append(
            {
                "stem": render_path.stem,
                "render": str(render_path),
                "target": str(target_path) if target_path is not None else "",
                "anchor": str(anchor_path) if anchor_path is not None else "",
                "mask": str(mask_path) if mask_path is not None else "",
            }
        )

    summary = {
        "version": "export_render_hf_components_v0",
        "render_dir": str(render_dir),
        "target_dir": str(target_dir) if target_dir is not None else "",
        "anchor_dir": str(anchor_dir) if anchor_dir is not None else "",
        "mask_dir": str(mask_dir) if mask_dir is not None else "",
        "output_dir": str(output_dir),
        "match_policy": str(args.match_policy),
        "highpass_kernel": int(args.highpass_kernel),
        "norm_percentile": float(args.norm_percentile),
        "num_frames": int(len(frames)),
        "mean_render_hf_abs": _mean(render_energy),
        "mean_target_hf_abs": _mean(target_energy),
        "mean_render_residual_hf_abs": _mean(residual_energy),
        "mean_residual_error_abs": _mean(residual_error),
        "frames": frames,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))
    print(f"[render-hf-v0] inspect: {output_dir}/sheet")
    print(f"[render-hf-v0] render hf: {output_dir}/render_hf_abs")
    if anchor_paths:
        print(f"[render-hf-v0] residual hf: {output_dir}/render_residual_hf_abs")


if __name__ == "__main__":
    main()
