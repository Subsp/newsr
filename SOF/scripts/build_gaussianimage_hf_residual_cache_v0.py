#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit trusted image-domain HF residuals with the official GaussianImage 2DGS implementation. "
            "This script only prepares HF targets, weighted loss, and diagnostics; 2D Gaussian raster/model code "
            "is imported from --external_repo_root."
        )
    )
    parser.add_argument("--external_repo_root", required=True, help="Clone of https://github.com/Xinjie-Q/GaussianImage.")
    parser.add_argument("--target_dir", required=True, help="Usually NPSE edge_target.")
    parser.add_argument("--anchor_dir", required=True, help="Anchor render directory.")
    parser.add_argument("--mask_dir", required=True, help="Trusted edge/HF mask directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--match_policy", default="order_if_needed", choices=["stem", "order", "order_if_needed"])
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--detail_alpha", type=float, default=0.8)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--confidence_power", type=float, default=1.5)
    parser.add_argument("--mask_power", type=float, default=1.0)
    parser.add_argument("--background_weight", type=float, default=0.02)
    parser.add_argument("--num_gaussians", type=int, default=4096)
    parser.add_argument("--model", default="cholesky", choices=["cholesky", "rs"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adan"])
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="l1_l2", choices=["l1", "l2", "l1_l2"])
    parser.add_argument("--lambda_l1", type=float, default=0.5)
    parser.add_argument("--lambda_l2", type=float, default=0.5)
    parser.add_argument("--init_random", action="store_true")
    parser.add_argument("--neutral_outside_mask", action="store_true")
    parser.add_argument("--no_neutral_outside_mask", dest="neutral_outside_mask", action="store_false")
    parser.set_defaults(neutral_outside_mask=False)
    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _lookup(paths: Sequence[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in paths}


def _resolve(
    paths: Sequence[Path],
    lookup: Dict[str, Path],
    reference_path: Path,
    index: int,
    match_policy: str,
) -> Optional[Path]:
    if match_policy in {"stem", "order_if_needed"}:
        found = lookup.get(reference_path.stem.lower())
        if found is not None:
            return found
        if match_policy == "stem":
            return None
    if match_policy in {"order", "order_if_needed"} and index < len(paths):
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


def _import_gaussianimage(repo_root: Path, model_name: str):
    cholesky_py = repo_root / "gaussianimage_cholesky.py"
    rs_py = repo_root / "gaussianimage_rs.py"
    selected_py = cholesky_py if model_name == "cholesky" else rs_py
    if not selected_py.is_file():
        raise FileNotFoundError(
            f"GaussianImage model file not found under {repo_root}: {selected_py.name}. "
            "Clone it with: git clone --recursive https://github.com/Xinjie-Q/GaussianImage.git"
        )
    # GaussianImage imports quantize.py and pytorch_msssim at module import time,
    # but our HF fitting path always uses quantize=False and computes its own
    # weighted L1/L2 loss. Tiny stubs avoid pulling optional codec/SSIM
    # dependencies that would otherwise try to upgrade the existing torch env.
    if "quantize" not in sys.modules:
        stub = types.ModuleType("quantize")
        stub.__all__ = []
        sys.modules["quantize"] = stub
    if "pytorch_msssim" not in sys.modules:
        msssim_stub = types.ModuleType("pytorch_msssim")

        def _unused_msssim(*_args, **_kwargs):
            raise RuntimeError("pytorch_msssim is stubbed; this adapter does not use GaussianImage SSIM losses.")

        class _UnusedSSIM:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, *_args, **_kwargs):
                return _unused_msssim()

        msssim_stub.ms_ssim = _unused_msssim
        msssim_stub.ssim = _unused_msssim
        msssim_stub.SSIM = _UnusedSSIM
        sys.modules["pytorch_msssim"] = msssim_stub
    sys.path.insert(0, str(repo_root))
    gsplat_root = repo_root / "gsplat"
    if gsplat_root.is_dir():
        sys.path.insert(0, str(gsplat_root))

    def _load(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import GaussianImage module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if model_name == "cholesky":
        cholesky_module = _load(cholesky_py, "gaussianimage_cholesky")
        if not hasattr(cholesky_module, "GaussianImage_Cholesky"):
            raise ImportError(f"Missing GaussianImage_Cholesky in {cholesky_py}")
        return {"GaussianImage_Cholesky": cholesky_module.GaussianImage_Cholesky}
    rs_module = _load(rs_py, "gaussianimage_rs")
    if not hasattr(rs_module, "GaussianImage_RS"):
        raise ImportError(f"Missing GaussianImage_RS in {rs_py}")
    return {"GaussianImage_RS": rs_module.GaussianImage_RS}


def _target_signed_hf(
    target: np.ndarray,
    anchor: np.ndarray,
    mask: np.ndarray,
    highpass_kernel: int,
    detail_alpha: float,
    residual_clip: float,
    confidence_power: float,
    mask_power: float,
    neutral_outside_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_hp = target - _box_blur_rgb(target, highpass_kernel)
    anchor_hp = anchor - _box_blur_rgb(anchor, highpass_kernel)
    residual = detail_alpha * (target_hp - anchor_hp)
    residual = np.clip(residual, -residual_clip, residual_clip).astype(np.float32)
    trust = np.clip(mask, 0.0, 1.0) ** max(mask_power, 0.0)
    trust = trust ** max(confidence_power, 0.0)
    if neutral_outside_mask:
        residual = residual * trust[..., None]
    signed = np.clip(0.5 + residual / (2.0 * max(residual_clip, 1e-8)), 0.0, 1.0).astype(np.float32)
    return signed, residual, trust.astype(np.float32)


def _fit_gaussianimage(
    module,
    signed_target: np.ndarray,
    weight: np.ndarray,
    num_gaussians: int,
    model_name: str,
    iterations: int,
    lr: float,
    optimizer: str,
    loss_name: str,
    lambda_l1: float,
    lambda_l2: float,
    background_weight: float,
) -> Tuple[np.ndarray, object, List[float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GaussianImage fitting requires CUDA for diff_gaussian_rasterization.")
    h, w = signed_target.shape[:2]
    device = torch.device("cuda")
    image_t = torch.from_numpy(signed_target).to(device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
    weight_t = torch.from_numpy(np.clip(weight, 0.0, 1.0)).to(device=device, dtype=torch.float32)[None, :, :]
    weight_t = torch.clamp(weight_t + float(background_weight), 0.0, 1.0)
    model_cls = module["GaussianImage_Cholesky"] if model_name == "cholesky" else module["GaussianImage_RS"]
    model = model_cls(
        loss_type="L2",
        opt_type=str(optimizer),
        num_points=int(num_gaussians),
        H=int(h),
        W=int(w),
        BLOCK_H=16,
        BLOCK_W=16,
        device=device,
        lr=float(lr),
        quantize=False,
    ).to(device)
    losses: List[float] = []
    for _ in range(int(iterations)):
        model.optimizer.zero_grad(set_to_none=True)
        rendered = model.forward()["render"].squeeze(0)
        diff = rendered - image_t
        if loss_name == "l1":
            loss = (diff.abs() * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
        elif loss_name == "l2":
            loss = ((diff * diff) * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
        else:
            l1 = (diff.abs() * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
            l2 = ((diff * diff) * weight_t).sum() / (weight_t.sum() * 3.0 + 1e-8)
            loss = float(lambda_l1) * l1 + float(lambda_l2) * l2
        loss.backward()
        model.optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    with torch.no_grad():
        rendered = model.forward()["render"].squeeze(0).detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return rendered, model, losses


def _extract_primitives(model, model_name: str, h: int, w: int) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        xyz = torch.tanh(model._xyz).detach().cpu().numpy().astype(np.float32)
        mu_xy = np.empty((xyz.shape[0], 2), dtype=np.float32)
        mu_xy[:, 0] = (xyz[:, 0] * 0.5 + 0.5) * float(w - 1)
        mu_xy[:, 1] = (xyz[:, 1] * 0.5 + 0.5) * float(h - 1)
        features = torch.clamp(model.get_features, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        opacity = model.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32)
        payload = {
            "mu_xy": mu_xy,
            "color": features,
            "opacity": opacity,
        }
        if model_name == "cholesky" and hasattr(model, "_cholesky"):
            payload["cholesky"] = model.get_cholesky_elements.detach().cpu().numpy().astype(np.float32)
        if model_name == "rs":
            if hasattr(model, "_scaling"):
                payload["scaling"] = model._scaling.detach().cpu().numpy().astype(np.float32)
            if hasattr(model, "_rotation"):
                payload["rotation"] = model._rotation.detach().cpu().numpy().astype(np.float32)
    return payload


def _signed_to_residual(signed: np.ndarray, residual_clip: float) -> np.ndarray:
    return (signed.astype(np.float32) - 0.5) * (2.0 * float(residual_clip))


def _weighted_l1(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (a.shape[2] if a.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    return float((np.abs(a - b) * weight[..., None]).sum() / denom)


def _weighted_energy(value: np.ndarray, weight: np.ndarray) -> float:
    denom = float(weight.sum()) * (value.shape[2] if value.ndim == 3 else 1)
    if denom <= 1e-8:
        return float("nan")
    return float((np.abs(value) * weight[..., None]).sum() / denom)


def _pearson_abs(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    aa = np.abs(a).mean(axis=2).reshape(-1)
    bb = np.abs(b).mean(axis=2).reshape(-1)
    ww = weight.reshape(-1)
    keep = ww > 1e-6
    if int(keep.sum()) < 4:
        return float("nan")
    aa = aa[keep]
    bb = bb[keep]
    ww = ww[keep]
    ww = ww / max(float(ww.sum()), 1e-8)
    ma = float((aa * ww).sum())
    mb = float((bb * ww).sum())
    da = aa - ma
    db = bb - mb
    den = float(np.sqrt((ww * da * da).sum() * (ww * db * db).sum()))
    if den <= 1e-8:
        return float("nan")
    return float((ww * da * db).sum() / den)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _abs_vis(value: np.ndarray, residual_clip: float) -> np.ndarray:
    gray = np.clip(np.abs(value).mean(axis=2) / max(float(residual_clip), 1e-8), 0.0, 1.0)
    return np.repeat(gray[..., None], 3, axis=2)


def _overlay(target_abs: np.ndarray, recon_abs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*target_abs.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(target_abs, 0.0, 1.0)
    rgb[..., 1] = np.clip(recon_abs, 0.0, 1.0)
    rgb[..., 2] = np.clip(recon_abs, 0.0, 1.0)
    return np.clip(rgb * (0.12 + 0.88 * np.clip(weight[..., None], 0.0, 1.0)), 0.0, 1.0)


def _primitive_overlay(primitives: Dict[str, np.ndarray], h: int, w: int, max_draw: int = 4096) -> np.ndarray:
    image = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    mu = primitives["mu_xy"]
    color = primitives["color"]
    opacity = primitives.get("opacity", np.ones((mu.shape[0],), dtype=np.float32))
    order = np.argsort(opacity)[::-1][: min(int(max_draw), int(mu.shape[0]))]
    cholesky = primitives.get("cholesky")
    scaling = primitives.get("scaling")
    rotation = primitives.get("rotation")
    for i in order.tolist():
        x, y = float(mu[i, 0]), float(mu[i, 1])
        rgb = tuple(int(np.clip(color[i, j] * 255.0, 0, 255)) for j in range(3))
        if cholesky is not None and cholesky.shape[1] >= 3:
            a = float(abs(cholesky[i, 0]))
            b = float(abs(cholesky[i, 1]))
            c = float(abs(cholesky[i, 2]))
            length = max(1.0, min(12.0, 160.0 * max(a, c)))
            theta = math.atan2(b, a + 1e-8)
        elif scaling is not None and rotation is not None:
            sx = float(abs(scaling[i, 0])) if scaling.ndim > 1 else float(abs(scaling[i]))
            length = max(1.0, min(12.0, 160.0 * sx))
            theta = float(rotation[i, 0]) if rotation.ndim > 1 else float(rotation[i])
        else:
            length = 2.0
            theta = 0.0
        dx = math.cos(theta) * length
        dy = math.sin(theta) * length
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=rgb, width=1)
    return np.asarray(image, dtype=np.float32) / 255.0


def _panel(rgb: np.ndarray, label: str) -> Image.Image:
    rgb = np.clip(rgb, 0.0, 1.0)
    image = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(560, image.width), 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return image


def _write_sheet(
    path: Path,
    signed_target: np.ndarray,
    signed_render: np.ndarray,
    target_residual: np.ndarray,
    recon_residual: np.ndarray,
    error: np.ndarray,
    weight: np.ndarray,
    primitive_overlay: np.ndarray,
    residual_clip: float,
) -> None:
    target_abs = _abs_vis(target_residual, residual_clip)
    recon_abs = _abs_vis(recon_residual, residual_clip)
    panels = [
        _panel(signed_target, "target signed HF"),
        _panel(signed_render, "GaussianImage signed HF"),
        _panel(target_abs, "target abs HF"),
        _panel(recon_abs, "GaussianImage abs HF"),
        _panel(np.repeat(np.clip(weight[..., None], 0.0, 1.0), 3, axis=2), "weighted trust edge"),
        _panel(_abs_vis(error, residual_clip), "abs residual error"),
        _panel(_overlay(target_abs[..., 0], recon_abs[..., 0], weight), "overlay target=red 2DGS=cyan"),
        _panel(primitive_overlay, "exported 2D Gaussian primitives"),
    ]
    width = max(p.width for p in panels)
    height = max(p.height for p in panels)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * width, (i // cols) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _mean(rows: Sequence[Dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.external_repo_root).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output dir is not empty; use --overwrite: {output_dir}")
    module = _import_gaussianimage(repo_root, str(args.model))

    target_paths = _list_images(target_dir)
    anchor_paths = _list_images(anchor_dir)
    mask_paths = _list_images(mask_dir)
    if int(args.limit) > 0:
        target_paths = target_paths[: int(args.limit)]

    dirs = {
        "target_hf": output_dir / "target_hf",
        "recon_hf": output_dir / "recon_hf",
        "target_abs": output_dir / "target_abs",
        "recon_abs": output_dir / "recon_abs",
        "edge_recon": output_dir / "edge_recon",
        "overlay": output_dir / "overlay",
        "primitive_overlay": output_dir / "primitive_overlay",
        "sheet": output_dir / "sheet",
        "primitives": output_dir / "primitives",
    }
    if bool(args.save_pt):
        dirs["pt"] = output_dir / "pt"
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    anchor_lookup = _lookup(anchor_paths)
    mask_lookup = _lookup(mask_paths)
    rows: List[Dict[str, float]] = []
    frames: List[Dict[str, object]] = []

    print(f"[gaussianimage-hf-v0] repo   : {repo_root}")
    print(f"[gaussianimage-hf-v0] target : {target_dir}")
    print(f"[gaussianimage-hf-v0] anchor : {anchor_dir}")
    print(f"[gaussianimage-hf-v0] mask   : {mask_dir}")
    print(f"[gaussianimage-hf-v0] output : {output_dir}")
    print(
        f"[gaussianimage-hf-v0] fit    : model={args.model} n={args.num_gaussians} "
        f"iters={args.iterations} lr={args.lr} loss={args.loss}"
    )

    for index, target_path in enumerate(tqdm(target_paths, desc="GaussianImage HF")):
        anchor_path = _resolve(anchor_paths, anchor_lookup, target_path, index, args.match_policy)
        mask_path = _resolve(mask_paths, mask_lookup, target_path, index, args.match_policy)
        if anchor_path is None or mask_path is None:
            continue
        target = _load_rgb(target_path)
        size = (target.shape[1], target.shape[0])
        anchor = _load_rgb(anchor_path, size=size)
        mask = _load_gray(mask_path, size=size)
        signed_target, target_residual, weight = _target_signed_hf(
            target,
            anchor,
            mask,
            int(args.highpass_kernel),
            float(args.detail_alpha),
            float(args.residual_clip),
            float(args.confidence_power),
            float(args.mask_power),
            bool(args.neutral_outside_mask),
        )
        signed_render, model, losses = _fit_gaussianimage(
            module,
            signed_target,
            weight,
            int(args.num_gaussians),
            str(args.model),
            int(args.iterations),
            float(args.lr),
            str(args.optimizer),
            str(args.loss),
            float(args.lambda_l1),
            float(args.lambda_l2),
            float(args.background_weight),
        )
        recon_residual = _signed_to_residual(signed_render, float(args.residual_clip))
        error = recon_residual - target_residual
        target_abs = np.clip(np.abs(target_residual).mean(axis=2) / max(float(args.residual_clip), 1e-8), 0.0, 1.0)
        recon_abs = np.clip(np.abs(recon_residual).mean(axis=2) / max(float(args.residual_clip), 1e-8), 0.0, 1.0)
        overlay = _overlay(target_abs, recon_abs, weight)
        primitives = _extract_primitives(model, str(args.model), target.shape[0], target.shape[1])
        primitive_overlay = _primitive_overlay(primitives, target.shape[0], target.shape[1])

        stem = target_path.stem
        _save_rgb(dirs["target_hf"] / f"{stem}.png", signed_target)
        _save_rgb(dirs["recon_hf"] / f"{stem}.png", signed_render)
        _save_gray(dirs["target_abs"] / f"{stem}.png", target_abs)
        _save_gray(dirs["recon_abs"] / f"{stem}.png", recon_abs)
        _save_rgb(dirs["edge_recon"] / f"{stem}.png", np.clip(anchor + recon_residual, 0.0, 1.0))
        _save_rgb(dirs["overlay"] / f"{stem}.png", overlay)
        _save_rgb(dirs["primitive_overlay"] / f"{stem}.png", primitive_overlay)
        if index < int(args.debug_limit):
            _write_sheet(
                dirs["sheet"] / f"{stem}.png",
                signed_target,
                signed_render,
                target_residual,
                recon_residual,
                error,
                weight,
                primitive_overlay,
                float(args.residual_clip),
            )
        np.savez_compressed(dirs["primitives"] / f"{stem}.npz", **primitives, losses=np.asarray(losses, dtype=np.float32))
        if bool(args.save_pt):
            torch.save(model.state_dict(), dirs["pt"] / f"{stem}.pt")

        row = {
            "index": float(index),
            "num_primitives": float(args.num_gaussians),
            "loss_start": float(losses[0]) if losses else float("nan"),
            "loss_final": float(losses[-1]) if losses else float("nan"),
            "target_energy": _weighted_energy(target_residual, weight),
            "recon_energy": _weighted_energy(recon_residual, weight),
            "l1": _weighted_l1(recon_residual, target_residual, weight),
            "corr_abs": _pearson_abs(recon_residual, target_residual, weight),
            "weight_mean": float(weight.mean()),
        }
        rows.append(row)
        frames.append(
            {
                "stem": stem,
                "target": str(target_path),
                "anchor": str(anchor_path),
                "mask": str(mask_path),
                "num_primitives": int(args.num_gaussians),
                "loss_final": row["loss_final"],
            }
        )
        del model
        torch.cuda.empty_cache()

    summary = {
        "external_repo_root": str(repo_root),
        "target_dir": str(target_dir),
        "anchor_dir": str(anchor_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
        "match_policy": args.match_policy,
        "model": args.model,
        "num_frames": len(rows),
        "num_gaussians": int(args.num_gaussians),
        "iterations": int(args.iterations),
        "loss_final_mean": _mean(rows, "loss_final"),
        "l1_mean": _mean(rows, "l1"),
        "corr_abs_mean": _mean(rows, "corr_abs"),
        "target_energy_mean": _mean(rows, "target_energy"),
        "recon_energy_mean": _mean(rows, "recon_energy"),
        "weight_mean": _mean(rows, "weight_mean"),
        "frames": frames,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "num_primitives",
                "loss_start",
                "loss_final",
                "target_energy",
                "recon_energy",
                "l1",
                "corr_abs",
                "weight_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))
    print(f"[gaussianimage-hf-v0] summary: {output_dir / 'summary.json'}")
    print(f"[gaussianimage-hf-v0] inspect: {dirs['sheet']}")


if __name__ == "__main__":
    main()
