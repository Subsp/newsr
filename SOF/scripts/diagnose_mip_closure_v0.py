from __future__ import annotations

import argparse
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import load_model_ply, resolve_iteration


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=white_background,
        data_device="cpu",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def _parse_csv_ints(value: str | None) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    out: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _select_views(items: Sequence[object], max_items: int, view_indices: Sequence[int]) -> tuple[List[object], List[int]]:
    if view_indices:
        selected: List[object] = []
        selected_indices: List[int] = []
        for idx in view_indices:
            if int(idx) < 0 or int(idx) >= len(items):
                raise IndexError(f"view index {idx} out of range for split with {len(items)} views")
            selected.append(items[int(idx)])
            selected_indices.append(int(idx))
        return selected, selected_indices
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items), list(range(len(items)))
    ids = np.unique(np.linspace(0, len(items) - 1, num=int(max_items), dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()], [int(idx) for idx in ids.tolist()]


def _iter_views(scene: Scene, split: str) -> List[tuple[str, Sequence[object]]]:
    if split == "train":
        return [("train", scene.getTrainCameras())]
    if split == "test":
        return [("test", scene.getTestCameras())]
    return [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]


def _get_splat_settings(model_path: Path) -> ExtendedSettings:
    config_path = model_path / "config.json"
    if config_path.exists():
        return ExtendedSettings.from_json(str(config_path))
    return ExtendedSettings()


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return value.strip("_") or "view"


def _rgb_to_luma(rgb_chw: torch.Tensor) -> torch.Tensor:
    return rgb_chw[0:1] * 0.299 + rgb_chw[1:2] * 0.587 + rgb_chw[2:3] * 0.114


def _box_blur_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(1, int(kernel_size))
    if kernel <= 1:
        return image
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1).squeeze(0)


def _box_blur_hw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(1, int(kernel_size))
    if kernel <= 1:
        return image
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = F.pad(image.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1).squeeze(0).squeeze(0)


def _extract_buffers(render_pkg: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rgb = render_pkg["render"][:3].detach().clamp(0.0, 1.0)
    alpha = render_pkg["alpha"][0].detach().clamp(0.0, 1.0)
    depth = render_pkg["depth"][0].detach()
    premul = rgb * alpha.unsqueeze(0)
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "premul": premul,
        "premul_luma": _rgb_to_luma(premul)[0],
    }


def _compute_lowpass(buffers: Dict[str, torch.Tensor], kernel_size: int) -> Dict[str, torch.Tensor]:
    alpha_low = _box_blur_hw(buffers["alpha"], int(kernel_size))
    premul_low = _box_blur_chw(buffers["premul"], int(kernel_size))
    rgb_low = premul_low / alpha_low.unsqueeze(0).clamp_min(1e-6)
    depth_low_num = _box_blur_hw(buffers["depth"] * buffers["alpha"], int(kernel_size))
    depth_low = depth_low_num / alpha_low.clamp_min(1e-6)
    return {
        "alpha_low": alpha_low,
        "premul_low": premul_low,
        "rgb_low": rgb_low,
        "depth_low": depth_low,
        "premul_luma_low": _rgb_to_luma(premul_low)[0],
    }


def _masked_channel_l1(a: torch.Tensor, b: torch.Tensor, mask_hw: torch.Tensor) -> float | None:
    mask = mask_hw.to(device=a.device, dtype=a.dtype).clamp(0.0, 1.0)
    denom = float((mask.sum() * a.shape[0]).detach().item())
    if denom < 1.0:
        return None
    value = (torch.abs(a - b) * mask.unsqueeze(0)).sum() / torch.clamp(mask.sum() * a.shape[0], min=1.0)
    return float(value.detach().item())


def _masked_scalar_mean(value_hw: torch.Tensor, mask_hw: torch.Tensor) -> float | None:
    mask = mask_hw.to(device=value_hw.device, dtype=value_hw.dtype).clamp(0.0, 1.0)
    denom = float(mask.sum().detach().item())
    if denom < 1.0:
        return None
    value = (value_hw * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(value.detach().item())


def _global_scalar_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).detach().item())


def _tensor_to_png_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _normalize_to_unit(value_hw: torch.Tensor, percentile: float = 0.99, eps: float = 1e-6) -> torch.Tensor:
    flat = value_hw.detach().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return torch.zeros_like(value_hw)
    scale = torch.quantile(finite.float(), float(percentile)).clamp_min(float(eps))
    return torch.clamp(value_hw / scale.to(device=value_hw.device, dtype=value_hw.dtype), 0.0, 1.0)


def _save_unsigned_heat(path: Path, value_hw: torch.Tensor) -> None:
    unit = _normalize_to_unit(torch.abs(value_hw))
    r = torch.clamp(1.7 * unit, 0.0, 1.0)
    g = torch.clamp(1.7 * unit - 0.45, 0.0, 1.0)
    b = torch.clamp(1.4 * unit - 0.9, 0.0, 1.0)
    rgb = torch.stack((r, g, b), dim=-1).cpu().numpy()
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _save_signed_heat(path: Path, value_hw: torch.Tensor) -> None:
    unit = _normalize_to_unit(torch.abs(value_hw))
    pos = torch.clamp(value_hw, min=0.0)
    neg = torch.clamp(-value_hw, min=0.0)
    pos = torch.where(unit > 0.0, pos / torch.clamp(pos.max(), min=1e-6), torch.zeros_like(pos))
    neg = torch.where(unit > 0.0, neg / torch.clamp(neg.max(), min=1e-6), torch.zeros_like(neg))
    rgb = torch.stack((pos, 0.15 * unit, neg), dim=-1).cpu().numpy()
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _compute_metrics(
    reference: Dict[str, torch.Tensor],
    reference_low: Dict[str, torch.Tensor],
    raw_compare: Dict[str, torch.Tensor],
    closure_low: Dict[str, torch.Tensor],
    *,
    alpha_threshold: float,
    depth_relative_min: float,
) -> tuple[Dict[str, float | int | None], Dict[str, torch.Tensor]]:
    support = reference["alpha"] >= float(alpha_threshold)
    sym_support = reference_low["alpha_low"] >= float(alpha_threshold)
    depth_mask = (
        support
        & (closure_low["alpha_low"] >= float(alpha_threshold))
        & torch.isfinite(reference["depth"])
        & torch.isfinite(closure_low["depth_low"])
        & (reference["depth"] > 1e-6)
    )
    sym_depth_mask = (
        sym_support
        & (closure_low["alpha_low"] >= float(alpha_threshold))
        & torch.isfinite(reference_low["depth_low"])
        & torch.isfinite(closure_low["depth_low"])
        & (reference_low["depth_low"] > 1e-6)
    )
    floor_depth_mask = (
        support
        & (reference_low["alpha_low"] >= float(alpha_threshold))
        & torch.isfinite(reference["depth"])
        & torch.isfinite(reference_low["depth_low"])
        & (reference["depth"] > 1e-6)
    )
    alpha_diff = closure_low["alpha_low"] - reference["alpha"]
    premul_luma_diff = closure_low["premul_luma_low"] - reference["premul_luma"]
    sym_alpha_diff = closure_low["alpha_low"] - reference_low["alpha_low"]
    sym_premul_luma_diff = closure_low["premul_luma_low"] - reference_low["premul_luma_low"]
    reference_alpha_floor = reference_low["alpha_low"] - reference["alpha"]
    reference_premul_luma_floor = reference_low["premul_luma_low"] - reference["premul_luma"]
    depth_rel = torch.zeros_like(reference["depth"])
    if bool(depth_mask.any()):
        depth_rel[depth_mask] = torch.abs(closure_low["depth_low"][depth_mask] - reference["depth"][depth_mask]) / torch.clamp(
            reference["depth"][depth_mask],
            min=float(depth_relative_min),
        )
    sym_depth_rel = torch.zeros_like(reference["depth"])
    if bool(sym_depth_mask.any()):
        sym_depth_rel[sym_depth_mask] = torch.abs(
            closure_low["depth_low"][sym_depth_mask] - reference_low["depth_low"][sym_depth_mask]
        ) / torch.clamp(
            reference_low["depth_low"][sym_depth_mask],
            min=float(depth_relative_min),
        )
    reference_depth_floor_rel = torch.zeros_like(reference["depth"])
    if bool(floor_depth_mask.any()):
        reference_depth_floor_rel[floor_depth_mask] = torch.abs(
            reference_low["depth_low"][floor_depth_mask] - reference["depth"][floor_depth_mask]
        ) / torch.clamp(
            reference["depth"][floor_depth_mask],
            min=float(depth_relative_min),
        )

    metrics: Dict[str, float | int | None] = {
        "support_pixels": int(support.sum().item()),
        "depth_pixels": int(depth_mask.sum().item()),
        "sym_support_pixels": int(sym_support.sum().item()),
        "sym_depth_pixels": int(sym_depth_mask.sum().item()),
        "reference_lowpass_floor_depth_pixels": int(floor_depth_mask.sum().item()),
        "raw_rgb_l1": _masked_channel_l1(raw_compare["rgb"], reference["rgb"], support),
        "raw_alpha_l1": _global_scalar_l1(raw_compare["alpha"], reference["alpha"]),
        "closure_rgb_l1": _masked_channel_l1(closure_low["rgb_low"], reference["rgb"], support),
        "closure_alpha_l1": _global_scalar_l1(closure_low["alpha_low"], reference["alpha"]),
        "closure_alpha_support_l1": _masked_scalar_mean(torch.abs(alpha_diff), support),
        "closure_alpha_over_mean": _masked_scalar_mean(torch.relu(alpha_diff), support),
        "closure_alpha_under_mean": _masked_scalar_mean(torch.relu(-alpha_diff), support),
        "closure_premul_l1": _masked_channel_l1(closure_low["premul_low"], reference["premul"], support),
        "closure_premul_luma_l1": _masked_scalar_mean(torch.abs(premul_luma_diff), support),
        "closure_premul_luma_over_mean": _masked_scalar_mean(torch.relu(premul_luma_diff), support),
        "closure_premul_luma_under_mean": _masked_scalar_mean(torch.relu(-premul_luma_diff), support),
        "closure_depth_abs": _masked_scalar_mean(torch.abs(closure_low["depth_low"] - reference["depth"]), depth_mask),
        "closure_depth_rel": _masked_scalar_mean(depth_rel, depth_mask),
        "sym_closure_rgb_l1": _masked_channel_l1(closure_low["rgb_low"], reference_low["rgb_low"], sym_support),
        "sym_closure_alpha_l1": _global_scalar_l1(closure_low["alpha_low"], reference_low["alpha_low"]),
        "sym_closure_alpha_support_l1": _masked_scalar_mean(torch.abs(sym_alpha_diff), sym_support),
        "sym_closure_alpha_over_mean": _masked_scalar_mean(torch.relu(sym_alpha_diff), sym_support),
        "sym_closure_alpha_under_mean": _masked_scalar_mean(torch.relu(-sym_alpha_diff), sym_support),
        "sym_closure_premul_l1": _masked_channel_l1(closure_low["premul_low"], reference_low["premul_low"], sym_support),
        "sym_closure_premul_luma_l1": _masked_scalar_mean(torch.abs(sym_premul_luma_diff), sym_support),
        "sym_closure_premul_luma_over_mean": _masked_scalar_mean(torch.relu(sym_premul_luma_diff), sym_support),
        "sym_closure_premul_luma_under_mean": _masked_scalar_mean(torch.relu(-sym_premul_luma_diff), sym_support),
        "sym_closure_depth_abs": _masked_scalar_mean(
            torch.abs(closure_low["depth_low"] - reference_low["depth_low"]),
            sym_depth_mask,
        ),
        "sym_closure_depth_rel": _masked_scalar_mean(sym_depth_rel, sym_depth_mask),
        "reference_lowpass_floor_rgb_l1": _masked_channel_l1(reference_low["rgb_low"], reference["rgb"], support),
        "reference_lowpass_floor_alpha_l1": _global_scalar_l1(reference_low["alpha_low"], reference["alpha"]),
        "reference_lowpass_floor_alpha_support_l1": _masked_scalar_mean(torch.abs(reference_alpha_floor), support),
        "reference_lowpass_floor_alpha_over_mean": _masked_scalar_mean(torch.relu(reference_alpha_floor), support),
        "reference_lowpass_floor_alpha_under_mean": _masked_scalar_mean(torch.relu(-reference_alpha_floor), support),
        "reference_lowpass_floor_premul_l1": _masked_channel_l1(reference_low["premul_low"], reference["premul"], support),
        "reference_lowpass_floor_premul_luma_l1": _masked_scalar_mean(torch.abs(reference_premul_luma_floor), support),
        "reference_lowpass_floor_premul_luma_over_mean": _masked_scalar_mean(torch.relu(reference_premul_luma_floor), support),
        "reference_lowpass_floor_premul_luma_under_mean": _masked_scalar_mean(torch.relu(-reference_premul_luma_floor), support),
        "reference_lowpass_floor_depth_abs": _masked_scalar_mean(
            torch.abs(reference_low["depth_low"] - reference["depth"]),
            floor_depth_mask,
        ),
        "reference_lowpass_floor_depth_rel": _masked_scalar_mean(reference_depth_floor_rel, floor_depth_mask),
    }
    return metrics, {
        "alpha_diff": alpha_diff,
        "premul_luma_diff": premul_luma_diff,
        "depth_rel": depth_rel,
        "sym_alpha_diff": sym_alpha_diff,
        "sym_premul_luma_diff": sym_premul_luma_diff,
        "sym_depth_rel": sym_depth_rel,
        "reference_alpha_floor": reference_alpha_floor,
        "reference_premul_luma_floor": reference_premul_luma_floor,
        "reference_depth_floor_rel": reference_depth_floor_rel,
    }


def _aggregate_rows(rows: Sequence[Dict[str, float | int | str | None]]) -> Dict[str, float | int]:
    if not rows:
        return {"views": 0}
    out: Dict[str, float | int] = {"views": int(len(rows))}
    keys = [key for key in rows[0].keys() if key not in {"image_name", "split", "source_view_index"}]
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        out[key] = float(np.mean(values))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose mip-domain closure for SOF regulation and recovery outputs.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--mip_model_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--reg_model_path", default="")
    parser.add_argument("--rec_model_path", default="")
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--view_indices", default="")
    parser.add_argument("--mip_iteration", type=int, default=-1)
    parser.add_argument("--reg_iteration", type=int, default=-1)
    parser.add_argument("--rec_iteration", type=int, default=-1)
    parser.add_argument("--lowpass_kernel", type=int, default=25)
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--depth_relative_min", type=float, default=0.5)
    parser.add_argument("--num_debug_views", type=int, default=4)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--compare_with_own_splat_settings", action="store_true")
    parser.add_argument("--save_view_npz", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    debug_dir = output_root / "debug_views"
    debug_dir.mkdir(parents=True, exist_ok=True)

    compare_specs: List[Dict[str, object]] = []
    if str(args.reg_model_path).strip():
        compare_specs.append(
            {
                "name": "reg",
                "path": Path(args.reg_model_path).expanduser().resolve(),
                "iteration": int(args.reg_iteration),
            }
        )
    if str(args.rec_model_path).strip():
        compare_specs.append(
            {
                "name": "rec",
                "path": Path(args.rec_model_path).expanduser().resolve(),
                "iteration": int(args.rec_iteration),
            }
        )
    if not compare_specs:
        raise ValueError("At least one of --reg_model_path or --rec_model_path must be provided")

    mip_iteration = resolve_iteration(mip_model_path, int(args.mip_iteration))
    dataset_args = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(mip_model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )
    dataset = ModelParams(None).extract(dataset_args)
    mip_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, mip_gaussians, load_iteration=mip_iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_mip_iteration = int(scene.loaded_iter if scene.loaded_iter is not None else mip_iteration)
    reference_splat_args = _get_splat_settings(mip_model_path)

    view_entries: List[Dict[str, object]] = []
    for split_name, split_views in _iter_views(scene, str(args.split)):
        for source_idx, camera in enumerate(split_views):
            view_entries.append(
                {
                    "split": split_name,
                    "source_view_index": int(source_idx),
                    "camera": camera,
                }
            )
    view_indices = _parse_csv_ints(str(args.view_indices))
    views, _ = _select_views(view_entries, int(args.max_views), view_indices)
    if not views:
        raise RuntimeError("No cameras selected for closure diagnostics.")

    filter_cameras = scene.getTrainCameras().copy()
    mip_gaussians.compute_3D_filter(filter_cameras, CUDA=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    for spec in compare_specs:
        path = spec["path"]
        iteration = resolve_iteration(path, int(spec["iteration"]))
        gaussians = load_model_ply(path, iteration, int(dataset.sh_degree))
        gaussians.compute_3D_filter(filter_cameras, CUDA=False)
        spec["iteration"] = int(iteration)
        spec["gaussians"] = gaussians
        spec["splat_args"] = _get_splat_settings(path)

    per_view: Dict[str, List[Dict[str, float | int | str | None]]] = {str(spec["name"]): [] for spec in compare_specs}

    for local_idx, view_spec in enumerate(views):
        camera = view_spec["camera"]
        split_name = str(view_spec["split"])
        source_idx = int(view_spec["source_view_index"])
        image_name = str(camera.image_name)
        prefix = f"view_{local_idx:03d}_{split_name}_{source_idx:05d}_{_slugify(image_name)}"

        reference_raw_pkg = render_simple(camera, mip_gaussians, background, splat_args=reference_splat_args)
        reference_buffers = _extract_buffers(reference_raw_pkg)
        reference_low = _compute_lowpass(reference_buffers, int(args.lowpass_kernel))

        if local_idx < int(args.num_debug_views):
            _tensor_to_png_rgb(debug_dir / f"{prefix}_mip_rgb.png", reference_buffers["rgb"])
            _tensor_to_png_rgb(debug_dir / f"{prefix}_mip_rgb_low.png", reference_low["rgb_low"])
            _save_unsigned_heat(debug_dir / f"{prefix}_mip_alpha_heat.png", reference_buffers["alpha"])
            _save_signed_heat(debug_dir / f"{prefix}_mip_alpha_lowpass_floor_signed.png", reference_low["alpha_low"] - reference_buffers["alpha"])
            _save_signed_heat(
                debug_dir / f"{prefix}_mip_premul_luma_lowpass_floor_signed.png",
                reference_low["premul_luma_low"] - reference_buffers["premul_luma"],
            )

        if bool(args.save_view_npz):
            np.savez_compressed(
                output_root / f"{prefix}_mip_reference.npz",
                rgb=_to_numpy(reference_buffers["rgb"]),
                alpha=_to_numpy(reference_buffers["alpha"]),
                premul=_to_numpy(reference_buffers["premul"]),
                depth=_to_numpy(reference_buffers["depth"]),
                rgb_low=_to_numpy(reference_low["rgb_low"]),
                alpha_low=_to_numpy(reference_low["alpha_low"]),
                premul_low=_to_numpy(reference_low["premul_low"]),
                depth_low=_to_numpy(reference_low["depth_low"]),
            )

        for spec in compare_specs:
            name = str(spec["name"])
            raw_splat_args = spec["splat_args"]
            raw_pkg = render_simple(camera, spec["gaussians"], background, splat_args=raw_splat_args)
            raw_buffers = _extract_buffers(raw_pkg)
            if bool(args.compare_with_own_splat_settings):
                closure_pkg = raw_pkg
            else:
                closure_pkg = render_simple(camera, spec["gaussians"], background, splat_args=reference_splat_args)
            closure_buffers = _extract_buffers(closure_pkg)
            closure_low = _compute_lowpass(closure_buffers, int(args.lowpass_kernel))
            metrics, maps = _compute_metrics(
                reference_buffers,
                reference_low,
                raw_buffers,
                closure_low,
                alpha_threshold=float(args.alpha_threshold),
                depth_relative_min=float(args.depth_relative_min),
            )
            row: Dict[str, float | int | str | None] = {
                "image_name": image_name,
                "split": split_name,
                "source_view_index": source_idx,
                **metrics,
            }
            per_view[name].append(row)

            if local_idx < int(args.num_debug_views):
                _tensor_to_png_rgb(debug_dir / f"{prefix}_{name}_raw_rgb.png", raw_buffers["rgb"])
                _tensor_to_png_rgb(debug_dir / f"{prefix}_{name}_closure_rgb_low.png", closure_low["rgb_low"])
                _save_signed_heat(debug_dir / f"{prefix}_{name}_closure_alpha_signed.png", maps["alpha_diff"])
                _save_signed_heat(debug_dir / f"{prefix}_{name}_closure_premul_luma_signed.png", maps["premul_luma_diff"])
                _save_unsigned_heat(debug_dir / f"{prefix}_{name}_closure_depth_rel_heat.png", maps["depth_rel"])
                _save_signed_heat(debug_dir / f"{prefix}_{name}_sym_closure_alpha_signed.png", maps["sym_alpha_diff"])
                _save_signed_heat(debug_dir / f"{prefix}_{name}_sym_closure_premul_luma_signed.png", maps["sym_premul_luma_diff"])
                _save_unsigned_heat(debug_dir / f"{prefix}_{name}_sym_closure_depth_rel_heat.png", maps["sym_depth_rel"])

            if bool(args.save_view_npz):
                np.savez_compressed(
                    output_root / f"{prefix}_{name}_closure.npz",
                    raw_rgb=_to_numpy(raw_buffers["rgb"]),
                    raw_alpha=_to_numpy(raw_buffers["alpha"]),
                    raw_premul=_to_numpy(raw_buffers["premul"]),
                    raw_depth=_to_numpy(raw_buffers["depth"]),
                    closure_rgb_low=_to_numpy(closure_low["rgb_low"]),
                    closure_alpha_low=_to_numpy(closure_low["alpha_low"]),
                    closure_premul_low=_to_numpy(closure_low["premul_low"]),
                    closure_depth_low=_to_numpy(closure_low["depth_low"]),
                    alpha_diff=_to_numpy(maps["alpha_diff"]),
                    premul_luma_diff=_to_numpy(maps["premul_luma_diff"]),
                    depth_rel=_to_numpy(maps["depth_rel"]),
                    sym_alpha_diff=_to_numpy(maps["sym_alpha_diff"]),
                    sym_premul_luma_diff=_to_numpy(maps["sym_premul_luma_diff"]),
                    sym_depth_rel=_to_numpy(maps["sym_depth_rel"]),
                    reference_alpha_floor=_to_numpy(maps["reference_alpha_floor"]),
                    reference_premul_luma_floor=_to_numpy(maps["reference_premul_luma_floor"]),
                    reference_depth_floor_rel=_to_numpy(maps["reference_depth_floor_rel"]),
                )

    summary = {
        "version": "diagnose_mip_closure_v0",
        "scene_root": str(scene_root),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "output_root": str(output_root),
        "lowpass_kernel": int(args.lowpass_kernel),
        "alpha_threshold": float(args.alpha_threshold),
        "depth_relative_min": float(args.depth_relative_min),
        "compare_with_own_splat_settings": bool(args.compare_with_own_splat_settings),
        "reference": {
            "name": "mip",
            "model_path": str(mip_model_path),
            "iteration": int(loaded_mip_iteration),
        },
        "compares": {
            str(spec["name"]): {
                "model_path": str(spec["path"]),
                "iteration": int(spec["iteration"]),
            }
            for spec in compare_specs
        },
        "selected_views": [
            {
                "split": str(view_spec["split"]),
                "source_view_index": int(view_spec["source_view_index"]),
                "image_name": str(view_spec["camera"].image_name),
            }
            for view_spec in views
        ],
        "aggregated": {
            name: _aggregate_rows(rows)
            for name, rows in per_view.items()
        },
        "per_view": per_view,
    }

    summary_path = output_root / "closure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
