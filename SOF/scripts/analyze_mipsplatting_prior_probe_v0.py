#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    SKIMAGE_FOUND = True
except Exception:
    peak_signal_noise_ratio = None
    structural_similarity = None
    SKIMAGE_FOUND = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    TENSORBOARD_FOUND = True
except Exception:
    EventAccumulator = None
    TENSORBOARD_FOUND = False


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RESAMPLING_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a mip-splatting prior probe run by comparing baseline/current renders, "
            "exporting RGB/Laplacian diffs, and summarizing PRIOR_INJECTED checkpoint stats."
        )
    )
    parser.add_argument("--baseline_model_dir", type=Path, required=True)
    parser.add_argument("--current_model_dir", type=Path, required=True)
    parser.add_argument("--baseline_iteration", type=int, required=True)
    parser.add_argument("--current_iteration", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--tile_width", type=int, default=320)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb01(path: Path, image: np.ndarray) -> None:
    clipped = np.clip(image, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).astype(np.uint8)).save(path)


def compute_laplacian_rgb(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="constant")
    center = padded[1:-1, 1:-1, :]
    up = padded[:-2, 1:-1, :]
    down = padded[2:, 1:-1, :]
    left = padded[1:-1, :-2, :]
    right = padded[1:-1, 2:, :]
    return 4.0 * center - up - down - left - right


def percentile_scale(values: np.ndarray, percentile: float = 99.0, eps: float = 1e-6) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 1.0
    scale = float(np.percentile(flat, percentile))
    return max(scale, eps)


def colorize_positive_map(values: np.ndarray) -> np.ndarray:
    scale = percentile_scale(values)
    norm = np.clip(values / scale, 0.0, 1.0)
    rgb = np.ones((*norm.shape, 3), dtype=np.float32)
    rgb[..., 1] = 1.0 - 0.65 * norm
    rgb[..., 2] = 1.0 - 1.00 * norm
    return rgb


def colorize_signed_map(values: np.ndarray) -> np.ndarray:
    scale = percentile_scale(np.abs(values))
    norm = np.clip(values / scale, -1.0, 1.0)
    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)
    rgb = np.ones((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = 1.0 - neg
    rgb[..., 1] = 1.0 - np.maximum(pos, neg)
    rgb[..., 2] = 1.0 - pos
    return rgb


def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] <= target_width:
        return image
    target_height = max(1, int(round(image.shape[0] * float(target_width) / float(image.shape[1]))))
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
    resized = pil.resize((target_width, target_height), RESAMPLING_BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def draw_label(image: Image.Image, text: str, x: int, y: int) -> None:
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((x, y), text)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))


def build_panel(
    name: str,
    baseline: np.ndarray,
    current: np.ndarray,
    rgb_absdiff: np.ndarray,
    lap_base_vis: np.ndarray,
    lap_curr_vis: np.ndarray,
    lap_signed_vis: np.ndarray,
    meta: dict[str, float],
    tile_width: int,
) -> Image.Image:
    tiles = [
        ("baseline", baseline),
        ("current", current),
        ("rgb_absdiff", rgb_absdiff),
        ("lap_base", lap_base_vis),
        ("lap_current", lap_curr_vis),
        ("lap_delta", lap_signed_vis),
    ]
    resized_tiles = [(label, resize_image(tile, tile_width)) for label, tile in tiles]
    gap = 10
    header_h = 60
    tile_h = max(tile.shape[0] for _, tile in resized_tiles)
    total_w = sum(tile.shape[1] for _, tile in resized_tiles) + gap * (len(resized_tiles) - 1)
    canvas = Image.new("RGB", (total_w, header_h + tile_h), color=(255, 255, 255))

    header = (
        f"{name} | dPSNR={meta['delta_psnr']:+.4f} dB | dSSIM={meta['delta_ssim']:+.5f} | "
        f"rgb_abs={meta['rgb_absdiff_mean']:.5f} | lap_abs={meta['lap_absdiff_mean']:.5f}"
    )
    draw_label(canvas, header, 8, 8)

    x = 0
    for label, tile in resized_tiles:
        pil_tile = Image.fromarray((np.clip(tile, 0.0, 1.0) * 255.0).astype(np.uint8))
        canvas.paste(pil_tile, (x, header_h))
        draw_label(canvas, label, x + 8, header_h + 8)
        x += pil_tile.width + gap
    return canvas


def load_summary_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_metrics_with_gt(render_dir: Path, gt_dir: Path) -> dict[str, dict[str, float]]:
    if not SKIMAGE_FOUND:
        raise RuntimeError("skimage is required to compute PSNR/SSIM when results_psnr_ssim.json is missing.")
    metrics: dict[str, dict[str, float]] = {}
    for render_path in iter_images(render_dir):
        gt_path = gt_dir / render_path.name
        if not gt_path.is_file():
            continue
        render = load_rgb01(render_path)
        gt = load_rgb01(gt_path)
        psnr = float(peak_signal_noise_ratio(gt, render, data_range=1.0))
        ssim = float(structural_similarity(gt, render, channel_axis=2, data_range=1.0))
        metrics[render_path.name] = {"PSNR": psnr, "SSIM": ssim}
    return metrics


def load_or_compute_per_view_metrics(model_dir: Path, iteration: int, resolution: int, split: str) -> dict[str, dict[str, float]]:
    summary_path = model_dir / "results_psnr_ssim.json"
    summary = load_summary_json(summary_path)
    if summary is not None:
        if (
            int(summary.get("iteration", -1)) == int(iteration)
            and int(summary.get("resolution", -1)) == int(resolution)
            and str(summary.get("split", "")) == str(split)
            and isinstance(summary.get("per_view"), dict)
        ):
            return summary["per_view"]

    root = model_dir / split / f"ours_{iteration}"
    return compute_metrics_with_gt(
        render_dir=root / f"test_preds_{resolution}",
        gt_dir=root / f"gt_{resolution}",
    )


def summarize_scalar_series(steps: np.ndarray, values: np.ndarray, max_points: int = 128) -> dict[str, Any]:
    count = int(values.shape[0])
    if count == 0:
        return {"count": 0, "points": []}
    take_idx = np.linspace(0, count - 1, num=min(count, max_points), dtype=np.int64)
    peak_idx = int(np.argmax(values))
    trough_idx = int(np.argmin(values))
    return {
        "count": count,
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(values[trough_idx]),
        "min_step": int(steps[trough_idx]),
        "max": float(values[peak_idx]),
        "max_step": int(steps[peak_idx]),
        "mean": float(values.mean()),
        "points": [
            {"step": int(steps[i]), "value": float(values[i])}
            for i in take_idx.tolist()
        ],
    }


def load_tensorboard_scalars(model_dir: Path) -> dict[str, Any]:
    if not TENSORBOARD_FOUND:
        return {"available": False, "reason": "tensorboard package not found"}

    try:
        accumulator = EventAccumulator(str(model_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
    except Exception as exc:
        return {"available": False, "reason": f"failed_to_load: {exc}"}

    tags = set(accumulator.Tags().get("scalars", []))
    selected_tags = [
        "prior/hf_seed_total",
        "prior/hf_seed_live",
        "prior/hf_recent_prior_live",
        "prior/hf_recent_prior_visible",
        "prior/hf_recent_prior_boost",
        "prior/hf_focus_mean",
        "prior/total",
        "prior/hf",
        "prior/sof_edge_touch_ratio",
    ]

    out: dict[str, Any] = {"available": True, "scalars": {}}
    for tag in selected_tags:
        if tag not in tags:
            continue
        events = accumulator.Scalars(tag)
        steps = np.asarray([e.step for e in events], dtype=np.int64)
        values = np.asarray([e.value for e in events], dtype=np.float32)
        out["scalars"][tag] = summarize_scalar_series(steps, values)
    return out


def summarize_mask(name: str, mask: torch.Tensor, scaling: torch.Tensor, opacity: torch.Tensor, generation: torch.Tensor | None, edge_touch_iter: torch.Tensor | None, iteration: int, seed_id: torch.Tensor | None) -> dict[str, Any]:
    mask = mask.to(dtype=torch.bool)
    count = int(mask.sum().item())
    result: dict[str, Any] = {"name": name, "count": count}
    if count == 0:
        return result

    scaling_sel = scaling[mask]
    opacity_sel = opacity[mask]
    scale_geom = torch.exp(torch.log(scaling_sel.clamp(min=1e-12)).mean(dim=1))
    scale_max = scaling_sel.max(dim=1).values
    scale_min = scaling_sel.min(dim=1).values
    anisotropy = scale_max / scale_min.clamp(min=1e-12)

    def tensor_stats(x: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(x.mean().item()),
            "median": float(x.median().item()),
            "p90": float(torch.quantile(x, 0.9).item()),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
        }

    result["opacity"] = tensor_stats(opacity_sel)
    result["scale_geom"] = tensor_stats(scale_geom)
    result["scale_max_axis"] = tensor_stats(scale_max)
    result["scale_min_axis"] = tensor_stats(scale_min)
    result["anisotropy"] = tensor_stats(anisotropy)

    if generation is not None and generation.numel() == mask.numel():
        gen_sel = generation[mask].to(dtype=torch.int64)
        uniq, counts = torch.unique(gen_sel, return_counts=True)
        result["generation"] = {
            "mean": float(gen_sel.float().mean().item()),
            "max": int(gen_sel.max().item()),
            "histogram": {
                str(int(k.item())): int(v.item())
                for k, v in zip(uniq, counts)
            },
        }

    if edge_touch_iter is not None and edge_touch_iter.numel() == mask.numel():
        touch_sel = edge_touch_iter[mask]
        valid = touch_sel >= 0
        result["edge_touch"] = {
            "touched_count": int(valid.sum().item()),
            "untouched_count": int((~valid).sum().item()),
        }
        if torch.any(valid):
            age = (int(iteration) - touch_sel[valid]).to(dtype=torch.float32)
            result["edge_touch"]["age"] = tensor_stats(age)

    if seed_id is not None and seed_id.numel() == mask.numel():
        seed_sel = seed_id[mask]
        valid = seed_sel >= 0
        result["seed_ids"] = {
            "assigned_count": int(valid.sum().item()),
            "unique_assigned": int(torch.unique(seed_sel[valid]).numel()) if torch.any(valid) else 0,
        }

    return result


def load_checkpoint_summary(model_dir: Path, iteration: int) -> dict[str, Any]:
    ckpt_path = model_dir / f"chkpnt{iteration}.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_args, loaded_iter = torch.load(ckpt_path, map_location="cpu")
    if len(model_args) < 7:
        raise ValueError(f"Unexpected checkpoint payload length: {len(model_args)}")

    raw_scaling = model_args[4].detach().cpu().float()
    raw_opacity = model_args[6].detach().cpu().float().reshape(-1)
    scaling = raw_scaling.exp()
    opacity = raw_opacity.sigmoid()

    tracking = model_args[-1] if isinstance(model_args[-1], dict) else {}
    source_tag = tracking.get("source_tag")
    if source_tag is None:
        source_tag = torch.zeros((raw_opacity.shape[0],), dtype=torch.int64)
    else:
        source_tag = torch.as_tensor(source_tag).reshape(-1).to(dtype=torch.int64)

    generation = tracking.get("generation")
    if generation is not None:
        generation = torch.as_tensor(generation).reshape(-1).to(dtype=torch.int64)

    edge_touch_iter = tracking.get("edge_touch_iter")
    if edge_touch_iter is not None:
        edge_touch_iter = torch.as_tensor(edge_touch_iter).reshape(-1).to(dtype=torch.int64)

    seed_id = tracking.get("seed_id")
    if seed_id is not None:
        seed_id = torch.as_tensor(seed_id).reshape(-1).to(dtype=torch.int64)

    total = int(raw_opacity.shape[0])
    source_tag = source_tag[:total]
    if generation is not None:
        generation = generation[:total]
    if edge_touch_iter is not None:
        edge_touch_iter = edge_touch_iter[:total]
    if seed_id is not None:
        seed_id = seed_id[:total]

    original_mask = source_tag == 0
    prior_mask = source_tag == 1
    non_original_mask = source_tag != 0

    summary = {
        "checkpoint_path": str(ckpt_path),
        "iteration": int(loaded_iter),
        "total_gaussians": total,
        "source_counts": {
            "original": int(original_mask.sum().item()),
            "prior_injected": int(prior_mask.sum().item()),
            "non_original": int(non_original_mask.sum().item()),
        },
        "groups": {
            "all": summarize_mask("all", torch.ones_like(source_tag, dtype=torch.bool), scaling, opacity, generation, edge_touch_iter, int(loaded_iter), seed_id),
            "original": summarize_mask("original", original_mask, scaling, opacity, generation, edge_touch_iter, int(loaded_iter), seed_id),
            "prior_injected": summarize_mask("prior_injected", prior_mask, scaling, opacity, generation, edge_touch_iter, int(loaded_iter), seed_id),
            "non_original": summarize_mask("non_original", non_original_mask, scaling, opacity, generation, edge_touch_iter, int(loaded_iter), seed_id),
        },
    }
    return summary


def analyze_render_delta(
    baseline_model_dir: Path,
    current_model_dir: Path,
    baseline_iteration: int,
    current_iteration: int,
    resolution: int,
    split: str,
    top_k: int,
    tile_width: int,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_root = baseline_model_dir / split / f"ours_{baseline_iteration}"
    current_root = current_model_dir / split / f"ours_{current_iteration}"
    baseline_render_dir = baseline_root / f"test_preds_{resolution}"
    current_render_dir = current_root / f"test_preds_{resolution}"
    gt_dir = current_root / f"gt_{resolution}"

    if not baseline_render_dir.is_dir():
        raise FileNotFoundError(f"Baseline render dir not found: {baseline_render_dir}")
    if not current_render_dir.is_dir():
        raise FileNotFoundError(f"Current render dir not found: {current_render_dir}")

    baseline_metrics = load_or_compute_per_view_metrics(baseline_model_dir, baseline_iteration, resolution, split)
    current_metrics = load_or_compute_per_view_metrics(current_model_dir, current_iteration, resolution, split)

    baseline_files = {p.name: p for p in iter_images(baseline_render_dir)}
    current_files = {p.name: p for p in iter_images(current_render_dir)}
    common = sorted(set(baseline_files) & set(current_files))
    if not common:
        raise FileNotFoundError("No matching render image names between baseline and current.")

    panel_dir = output_dir / "panels"
    rgb_diff_dir = output_dir / "rgb_absdiff"
    lap_diff_dir = output_dir / "lap_signed_diff"
    panel_dir.mkdir(parents=True, exist_ok=True)
    rgb_diff_dir.mkdir(parents=True, exist_ok=True)
    lap_diff_dir.mkdir(parents=True, exist_ok=True)

    per_view: dict[str, Any] = {}
    for name in common:
        baseline = load_rgb01(baseline_files[name])
        current = load_rgb01(current_files[name])
        if current.shape != baseline.shape:
            target_size = (baseline.shape[1], baseline.shape[0])
            current = np.asarray(
                Image.fromarray((current * 255.0).astype(np.uint8)).resize(target_size, RESAMPLING_BICUBIC),
                dtype=np.float32,
            ) / 255.0

        diff = current - baseline
        rgb_abs_scalar = np.abs(diff).mean(axis=2)
        lap_base = compute_laplacian_rgb(baseline)
        lap_current = compute_laplacian_rgb(current)
        lap_diff = lap_current - lap_base
        lap_abs_scalar = np.abs(lap_diff).mean(axis=2)

        base_metric = baseline_metrics.get(name, {})
        curr_metric = current_metrics.get(name, {})
        delta_psnr = float(curr_metric.get("PSNR", float("nan")) - base_metric.get("PSNR", float("nan")))
        delta_ssim = float(curr_metric.get("SSIM", float("nan")) - base_metric.get("SSIM", float("nan")))
        lap_base_energy = float(np.abs(lap_base).mean())
        lap_current_energy = float(np.abs(lap_current).mean())

        per_view[name] = {
            "baseline": {
                "PSNR": float(base_metric.get("PSNR", float("nan"))),
                "SSIM": float(base_metric.get("SSIM", float("nan"))),
                "lap_abs_mean": lap_base_energy,
            },
            "current": {
                "PSNR": float(curr_metric.get("PSNR", float("nan"))),
                "SSIM": float(curr_metric.get("SSIM", float("nan"))),
                "lap_abs_mean": lap_current_energy,
            },
            "delta": {
                "PSNR": delta_psnr,
                "SSIM": delta_ssim,
                "rgb_absdiff_mean": float(rgb_abs_scalar.mean()),
                "lap_absdiff_mean": float(lap_abs_scalar.mean()),
                "lap_energy_delta": lap_current_energy - lap_base_energy,
            },
        }

        save_rgb01(rgb_diff_dir / name, colorize_positive_map(rgb_abs_scalar))
        save_rgb01(lap_diff_dir / name, colorize_signed_map(lap_diff.mean(axis=2)))

    ranked_names = sorted(
        common,
        key=lambda n: (
            per_view[n]["delta"]["rgb_absdiff_mean"],
            -per_view[n]["delta"]["PSNR"] if not math.isnan(per_view[n]["delta"]["PSNR"]) else 0.0,
            per_view[n]["delta"]["lap_absdiff_mean"],
        ),
        reverse=True,
    )
    ranked_psnr_drop = sorted(
        common,
        key=lambda n: per_view[n]["delta"]["PSNR"] if not math.isnan(per_view[n]["delta"]["PSNR"]) else 0.0,
    )
    ranked_lap = sorted(
        common,
        key=lambda n: per_view[n]["delta"]["lap_absdiff_mean"],
        reverse=True,
    )

    selected = ranked_names[: max(1, int(top_k))]
    panels: list[Image.Image] = []
    for name in selected:
        baseline = load_rgb01(baseline_files[name])
        current = load_rgb01(current_files[name])
        if current.shape != baseline.shape:
            target_size = (baseline.shape[1], baseline.shape[0])
            current = np.asarray(
                Image.fromarray((current * 255.0).astype(np.uint8)).resize(target_size, RESAMPLING_BICUBIC),
                dtype=np.float32,
            ) / 255.0
        diff = current - baseline
        lap_base = compute_laplacian_rgb(baseline)
        lap_current = compute_laplacian_rgb(current)
        lap_diff = lap_current - lap_base
        panel = build_panel(
            name=name,
            baseline=baseline,
            current=current,
            rgb_absdiff=colorize_positive_map(np.abs(diff).mean(axis=2)),
            lap_base_vis=colorize_positive_map(np.abs(lap_base).mean(axis=2)),
            lap_curr_vis=colorize_positive_map(np.abs(lap_current).mean(axis=2)),
            lap_signed_vis=colorize_signed_map(lap_diff.mean(axis=2)),
            meta={
                "delta_psnr": per_view[name]["delta"]["PSNR"],
                "delta_ssim": per_view[name]["delta"]["SSIM"],
                "rgb_absdiff_mean": per_view[name]["delta"]["rgb_absdiff_mean"],
                "lap_absdiff_mean": per_view[name]["delta"]["lap_absdiff_mean"],
            },
            tile_width=tile_width,
        )
        panels.append(panel)
        panel.save(panel_dir / name)

    if panels:
        gap = 12
        total_w = max(panel.width for panel in panels)
        total_h = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
        sheet = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height + gap
        sheet.save(output_dir / "top_views_sheet.png")

    delta_psnr_values = np.asarray([per_view[name]["delta"]["PSNR"] for name in common], dtype=np.float32)
    delta_ssim_values = np.asarray([per_view[name]["delta"]["SSIM"] for name in common], dtype=np.float32)
    rgb_abs_values = np.asarray([per_view[name]["delta"]["rgb_absdiff_mean"] for name in common], dtype=np.float32)
    lap_abs_values = np.asarray([per_view[name]["delta"]["lap_absdiff_mean"] for name in common], dtype=np.float32)

    summary = {
        "baseline_model_dir": str(baseline_model_dir),
        "current_model_dir": str(current_model_dir),
        "split": split,
        "resolution": int(resolution),
        "baseline_iteration": int(baseline_iteration),
        "current_iteration": int(current_iteration),
        "n_common_views": len(common),
        "global": {
            "delta_psnr_mean": float(np.nanmean(delta_psnr_values)),
            "delta_psnr_min": float(np.nanmin(delta_psnr_values)),
            "delta_ssim_mean": float(np.nanmean(delta_ssim_values)),
            "delta_ssim_min": float(np.nanmin(delta_ssim_values)),
            "rgb_absdiff_mean": float(rgb_abs_values.mean()),
            "lap_absdiff_mean": float(lap_abs_values.mean()),
        },
        "top_views_by_rgb_absdiff": selected,
        "top_views_by_psnr_drop": ranked_psnr_drop[: max(1, int(top_k))],
        "top_views_by_lap_absdiff": ranked_lap[: max(1, int(top_k))],
        "per_view": per_view,
        "outputs": {
            "panel_dir": str(panel_dir),
            "rgb_absdiff_dir": str(rgb_diff_dir),
            "lap_signed_diff_dir": str(lap_diff_dir),
            "top_views_sheet": str(output_dir / "top_views_sheet.png"),
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    baseline_model_dir = args.baseline_model_dir.expanduser().resolve()
    current_model_dir = args.current_model_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else current_model_dir / f"prior_probe_analysis_vs_{baseline_model_dir.name}_{args.split}_r{args.resolution}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    render_summary = analyze_render_delta(
        baseline_model_dir=baseline_model_dir,
        current_model_dir=current_model_dir,
        baseline_iteration=args.baseline_iteration,
        current_iteration=args.current_iteration,
        resolution=args.resolution,
        split=args.split,
        top_k=args.top_k,
        tile_width=args.tile_width,
        output_dir=output_dir,
    )

    checkpoint_summary = load_checkpoint_summary(current_model_dir, args.current_iteration)
    tensorboard_summary = load_tensorboard_scalars(current_model_dir)

    summary = {
        "render_analysis": render_summary,
        "checkpoint_analysis": checkpoint_summary,
        "tensorboard_analysis": tensorboard_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    concise = {
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "delta_psnr_mean": render_summary["global"]["delta_psnr_mean"],
        "delta_ssim_mean": render_summary["global"]["delta_ssim_mean"],
        "rgb_absdiff_mean": render_summary["global"]["rgb_absdiff_mean"],
        "lap_absdiff_mean": render_summary["global"]["lap_absdiff_mean"],
        "prior_injected_count": checkpoint_summary["source_counts"]["prior_injected"],
        "top_views_by_rgb_absdiff": render_summary["top_views_by_rgb_absdiff"][: min(5, len(render_summary["top_views_by_rgb_absdiff"]))],
        "top_views_by_psnr_drop": render_summary["top_views_by_psnr_drop"][: min(5, len(render_summary["top_views_by_psnr_drop"]))],
    }
    print(json.dumps(concise, indent=2))
    print(f"saved to: {summary_path}")


if __name__ == "__main__":
    main()
