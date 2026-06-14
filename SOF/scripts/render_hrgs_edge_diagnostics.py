from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.visibility_records import VisibilityRecordConfig, build_coarse_visibility_records


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=white_background,
        data_device="cuda",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


def _normalize_map(metric: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    finite = np.isfinite(metric)
    if not finite.any():
        return np.zeros_like(metric, dtype=np.float32)
    values = metric[finite]
    lo = float(np.percentile(values, percentile_low))
    hi = float(np.percentile(values, percentile_high))
    if hi <= lo:
        hi = lo + 1e-6
    norm = (metric - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _viridis_rgb(norm_map: np.ndarray) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    cmap = colormaps["viridis"]
    rgba = cmap(norm_map)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _save_heatmap(path: Path, metric: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(metric)
    stats = {
        "min": float(np.min(metric[finite])) if finite.any() else 0.0,
        "mean": float(np.mean(metric[finite])) if finite.any() else 0.0,
        "median": float(np.median(metric[finite])) if finite.any() else 0.0,
        "p95": float(np.percentile(metric[finite], 95.0)) if finite.any() else 0.0,
        "p99": float(np.percentile(metric[finite], 99.0)) if finite.any() else 0.0,
        "max": float(np.max(metric[finite])) if finite.any() else 0.0,
    }
    norm = _normalize_map(metric)
    heat = _viridis_rgb(norm)
    Image.fromarray(heat).save(path)
    return stats


def _overlay_heatmap(base_rgb_chw: torch.Tensor, metric: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = (base_rgb_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.float32)
    heat = _viridis_rgb(_normalize_map(metric)).astype(np.float32)
    return np.clip((1.0 - alpha) * base + alpha * heat, 0.0, 255.0).astype(np.uint8)


def _load_action_payload(path: Path) -> Dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(path)
        raw = {key: loaded[key] for key in loaded.files}
    else:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported action payload type: {type(raw)!r}")
    if "update_strength" in raw:
        payload = raw
    elif "gs_action_payload" in raw and isinstance(raw["gs_action_payload"], dict):
        payload = raw["gs_action_payload"]
    elif "hrgs_outputs" in raw and isinstance(raw["hrgs_outputs"], dict) and isinstance(raw["hrgs_outputs"].get("gs_action_payload"), dict):
        payload = raw["hrgs_outputs"]["gs_action_payload"]
    else:
        raise KeyError("Action payload missing 'update_strength' or nested 'gs_action_payload'.")

    out: Dict[str, torch.Tensor] = {}
    for key in ("update_strength", "attach_strength", "detail_weight", "prior_color_strength"):
        value = payload.get(key)
        if value is None:
            raise KeyError(f"Missing action payload key: {key}")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        out[key] = value.detach().cpu().reshape(-1).float()
    return out


def _project_gaussian_scalar(
    gaussian_ids: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
) -> np.ndarray:
    coarse_ids = gaussian_ids[0, 0].cpu()
    coarse_weights = weights[0, 0].cpu()
    valid = coarse_ids >= 0
    safe_ids = coarse_ids.clamp_min(0)
    gathered = values[safe_ids].to(dtype=coarse_weights.dtype)
    weighted = torch.where(valid, gathered * coarse_weights, torch.zeros_like(gathered))
    return weighted.sum(dim=-1).numpy()


def _project_gaussian_count(gaussian_ids: torch.Tensor) -> np.ndarray:
    return (gaussian_ids[0, 0].cpu() >= 0).sum(dim=-1).float().numpy()


def _project_max_radius(gaussian_ids: torch.Tensor, radii: torch.Tensor) -> np.ndarray:
    coarse_ids = gaussian_ids[0, 0].cpu()
    valid = coarse_ids >= 0
    safe_ids = coarse_ids.clamp_min(0)
    gathered = radii[safe_ids].float()
    gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
    return gathered.max(dim=-1).values.numpy()


def _upsample_coarse_map(metric: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    metric_t = torch.from_numpy(metric)[None, None].float()
    up = F.interpolate(metric_t, size=target_hw, mode="nearest")
    return up[0, 0].numpy()


def _select_views(cameras: Sequence[object], split: str) -> Sequence[object]:
    if split == "train":
        return cameras
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Render edge-oriented Gaussian diagnostics for a selected view.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--render_index", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--action_payload", type=str, default=None)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--visibility_downsample", type=int, default=8)
    parser.add_argument("--visibility_topk", type=int, default=4)
    parser.add_argument("--visibility_max_visible", type=int, default=30000)
    parser.add_argument("--visibility_max_patch_radius", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset_args(
        scene_root=str(Path(args.scene_root).expanduser().resolve()),
        model_path=str(Path(args.model_path).expanduser().resolve()),
        images_subdir=args.images_subdir,
        white_background=bool(args.white_background),
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=int(args.iteration), shuffle=False, skip_test=False, skip_train=False)

    if args.split == "train":
        views = scene.getTrainCameras()
    else:
        views = scene.getTestCameras()
    if not views:
        raise RuntimeError(f"No {args.split} cameras available.")
    if args.render_index < 0 or args.render_index >= len(views):
        raise IndexError(f"render_index {args.render_index} out of range for {args.split} split of size {len(views)}")

    view = views[int(args.render_index)]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_pkg = render_simple(view, gaussians, background)

    render_rgb = render_pkg["render"][:3]
    gt_rgb = view.original_image[:3]
    alpha = render_pkg["alpha"][0].detach().cpu().numpy()
    radii = render_pkg["radii"].detach().cpu()

    image_hw = (int(view.image_height), int(view.image_width))
    visibility_records = build_coarse_visibility_records(
        gaussians,
        [view],
        [render_pkg],
        image_hw=image_hw,
        cfg=VisibilityRecordConfig(
            downsample=int(args.visibility_downsample),
            topk=int(args.visibility_topk),
            max_visible_per_view=int(args.visibility_max_visible),
            max_patch_radius=int(args.visibility_max_patch_radius),
        ),
    )
    gaussian_ids = visibility_records["gaussian_ids"]
    weights = visibility_records["weights"]

    contribution_count = _project_gaussian_count(gaussian_ids)
    projected_radius = _project_max_radius(gaussian_ids, radii)
    weighted_radius = _project_gaussian_scalar(gaussian_ids, weights, radii.float())

    maps = {
        "alpha": alpha,
        "contribution_count": _upsample_coarse_map(contribution_count, image_hw),
        "projected_radius_max": _upsample_coarse_map(projected_radius, image_hw),
        "projected_radius_weighted": _upsample_coarse_map(weighted_radius, image_hw),
    }

    action_stats = {}
    if args.action_payload:
        payload = _load_action_payload(Path(args.action_payload).expanduser().resolve())
        for key, value in payload.items():
            projected = _project_gaussian_scalar(gaussian_ids, weights, value)
            maps[f"action_{key}"] = _upsample_coarse_map(projected, image_hw)
            action_stats[key] = {
                "mean": float(value.mean().item()),
                "p95": float(torch.quantile(value, 0.95).item()),
                "p99": float(torch.quantile(value, 0.99).item()),
                "max": float(value.max().item()),
            }

    _save_rgb(output_dir / "render_rgb.png", render_rgb)
    _save_rgb(output_dir / "gt_rgb.png", gt_rgb)

    summary = {
        "scene_root": str(Path(args.scene_root).expanduser().resolve()),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "images_subdir": args.images_subdir,
        "split": args.split,
        "render_index": int(args.render_index),
        "image_name": str(view.image_name),
        "iteration": int(scene.loaded_iter if scene.loaded_iter is not None else args.iteration),
        "visibility_downsample": int(args.visibility_downsample),
        "visibility_topk": int(args.visibility_topk),
        "visibility_max_visible": int(args.visibility_max_visible),
        "visibility_max_patch_radius": int(args.visibility_max_patch_radius),
        "map_stats": {},
        "action_payload_stats": action_stats,
    }

    for key, metric in maps.items():
        np.save(output_dir / f"{key}.npy", metric.astype(np.float32))
        stats = _save_heatmap(output_dir / f"{key}_heatmap.png", metric)
        overlay = _overlay_heatmap(render_rgb, metric, alpha=0.45)
        Image.fromarray(overlay).save(output_dir / f"{key}_overlay.png")
        summary["map_stats"][key] = stats

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
