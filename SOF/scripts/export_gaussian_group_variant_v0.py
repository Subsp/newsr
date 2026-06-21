#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
for candidate in reversed((REPO_ROOT, PROJECT_ROOT / "mip-splatting")):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

try:
    from gaussian_renderer import render_simple
except ImportError:
    from gaussian_renderer import render as _render

    class _PreviewPipeline:
        convert_SHs_python = False
        convert_SBs_python = False
        compute_cov3D_python = False
        compute_filter3D_python = False
        compute_view2gaussian_python = False
        debug = False
        use_merged_sof_rasterizer = False
        use_vanilla_sof_rasterizer = False

    def render_simple(viewpoint_camera, pc, bg_color):
        render_kwargs = {}
        render_params = inspect.signature(_render).parameters
        if "kernel_size" in render_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in render_params.values()):
            render_kwargs["kernel_size"] = 0.0
        render_pkg = _render(
            viewpoint_camera,
            pc,
            _PreviewPipeline(),
            bg_color,
            **render_kwargs,
        )
        if "alpha" not in render_pkg:
            rgb = render_pkg["render"][:3]
            alpha = torch.linalg.vector_norm(rgb - bg_color.reshape(3, 1, 1), dim=0, keepdim=True)
            render_pkg["alpha"] = (alpha > 1e-4).to(dtype=rgb.dtype)
        return render_pkg
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from export_gaussian_mask_subset_v0 import (
    _build_dataset_args,
    _clone_subset_gaussians,
    _compute_3d_filter_compat,
    _copy_render_config,
    _iter_views,
    _resolve_iteration,
    _save_rgb,
    _select_uniform,
)


def _make_scene(dataset, gaussians, *, load_iteration: int):
    kwargs = {"load_iteration": load_iteration, "shuffle": False}
    params = inspect.signature(Scene.__init__).parameters
    if "skip_test" in params:
        kwargs["skip_test"] = False
    if "skip_train" in params:
        kwargs["skip_train"] = False
    return Scene(dataset, gaussians, **kwargs)


def _save_gray(path: Path, image_hw: torch.Tensor) -> None:
    image = image_hw.detach().float().cpu()
    image = torch.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image to have shape [H, W], got {tuple(image.shape)}")
    image = image.clamp(0.0, 1.0).numpy()
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


def _save_depth(path_png: Path, path_npy: Path, depth_hw: torch.Tensor) -> None:
    depth = depth_hw.detach().float().cpu()
    depth = torch.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image to have shape [H, W], got {tuple(depth.shape)}")
    np.save(path_npy, depth.numpy())
    finite = torch.isfinite(depth)
    if not torch.any(finite):
        vis = torch.zeros_like(depth)
    else:
        valid = depth[finite]
        q_lo = float(torch.quantile(valid, 0.02).item())
        q_hi = float(torch.quantile(valid, 0.98).item())
        if not math.isfinite(q_lo) or not math.isfinite(q_hi) or q_hi <= q_lo:
            q_lo = float(valid.min().item())
            q_hi = float(valid.max().item())
        vis = torch.zeros_like(depth)
        denom = max(q_hi - q_lo, 1e-6)
        vis[finite] = ((depth[finite] - q_lo) / denom).clamp(0.0, 1.0)
    _save_gray(path_png, vis)


def _load_mask(path: Path, expected_count: int) -> torch.Tensor:
    if not path.is_file():
        return torch.zeros((expected_count,), dtype=torch.bool)
    value = torch.load(path, map_location="cpu")
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(dtype=torch.bool).reshape(-1)
    if int(value.shape[0]) != int(expected_count):
        raise ValueError(f"Mask length mismatch for {path}: expected {expected_count}, got {int(value.shape[0])}")
    return value


def _load_selection_mask(
    *,
    model_path: Path,
    iteration: int,
    selection_key: str,
    selection_source: str,
    payload_path: Path | None,
    total_gaussians: int,
) -> torch.Tensor:
    source = str(selection_source).strip().lower()
    key = str(selection_key).strip().lower()
    if source == "tracking":
        tags_path = model_path / "point_cloud" / f"iteration_{int(iteration)}" / "gaussian_tags.pt"
        payload = None
        if tags_path.is_file():
            payload = torch.load(tags_path, map_location="cpu")
            source_tag = payload.get("source_tag")
            if torch.is_tensor(source_tag) and int(source_tag.reshape(-1).shape[0]) != int(total_gaussians):
                payload = None
        if payload is None:
            checkpoint_path = model_path / f"chkpnt{int(iteration)}.pth"
            if not checkpoint_path.is_file():
                if tags_path.is_file():
                    raise ValueError(
                        f"tracking metadata length mismatch for {tags_path}: "
                        f"expected {total_gaussians}, got {int(source_tag.reshape(-1).shape[0])}"
                    )
                raise FileNotFoundError(
                    f"tracking metadata not found or mismatched; expected {tags_path} or {checkpoint_path}"
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, (list, tuple)) or len(checkpoint) < 1:
                raise ValueError(f"Unsupported checkpoint payload: {checkpoint_path}")
            model_args = checkpoint[0]
            if (
                not isinstance(model_args, (list, tuple))
                or len(model_args) < 13
                or not isinstance(model_args[-1], dict)
            ):
                raise ValueError(f"Checkpoint does not contain tracking state: {checkpoint_path}")
            payload = model_args[-1]
        source_tag = payload.get("source_tag")
        seed_id = payload.get("seed_id")
        generation = payload.get("generation")
        if source_tag is None:
            raise KeyError(f"tracking metadata missing source_tag for model {model_path}")
        if not torch.is_tensor(source_tag):
            source_tag = torch.as_tensor(source_tag)
        source_tag = source_tag.to(dtype=torch.int64).reshape(-1)
        if int(source_tag.shape[0]) < int(total_gaussians):
            raise ValueError(
                f"tracking source_tag length mismatch: expected at least {total_gaussians}, got {int(source_tag.shape[0])}"
            )
        if int(source_tag.shape[0]) > int(total_gaussians):
            print(
                "[export-gaussian-variant] warning: tracking source_tag is longer than the "
                f"loaded point cloud ({int(source_tag.shape[0])} vs {int(total_gaussians)}); "
                "truncating to the current Gaussian count for visualization."
            )
            source_tag = source_tag[: int(total_gaussians)]
        if key in {"all", "full"}:
            return torch.ones((total_gaussians,), dtype=torch.bool)
        if key in {"original"}:
            return source_tag == int(GaussianSourceTag.ORIGINAL)
        if key in {"prior", "prior_injected", "proxy", "proxies"}:
            return source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
        if key in {"probe", "extension_probe"}:
            return source_tag == int(GaussianSourceTag.EXTENSION_PROBE)
        if key in {"non_original", "added"}:
            return source_tag != int(GaussianSourceTag.ORIGINAL)
        if key in {"seeded"}:
            if seed_id is None:
                raise KeyError(f"tracking metadata missing seed_id: {tags_path}")
            if not torch.is_tensor(seed_id):
                seed_id = torch.as_tensor(seed_id)
            seed_id = seed_id.to(dtype=torch.int64).reshape(-1)
            if int(seed_id.shape[0]) < int(total_gaussians):
                raise ValueError(
                    f"tracking seed_id length mismatch: expected at least {total_gaussians}, got {int(seed_id.shape[0])}"
                )
            if int(seed_id.shape[0]) > int(total_gaussians):
                seed_id = seed_id[: int(total_gaussians)]
            return seed_id >= 0
        if key in {"generation_gt0", "generation_plus"}:
            if generation is None:
                raise KeyError(f"tracking metadata missing generation: {tags_path}")
            if not torch.is_tensor(generation):
                generation = torch.as_tensor(generation)
            generation = generation.to(dtype=torch.int64).reshape(-1)
            if int(generation.shape[0]) < int(total_gaussians):
                raise ValueError(
                    f"tracking generation length mismatch: expected at least {total_gaussians}, got {int(generation.shape[0])}"
                )
            if int(generation.shape[0]) > int(total_gaussians):
                generation = generation[: int(total_gaussians)]
            return generation > 0
        raise ValueError(
            f"Unsupported tracking selection_key={selection_key!r}; "
            "use one of full/original/prior_injected/proxy/extension_probe/non_original/seeded/generation_gt0."
        )
    if source == "payload":
        if payload_path is None:
            raise ValueError("selection_source=payload requires --mask_payload_path")
        payload = torch.load(payload_path, map_location="cpu")
        if key not in payload:
            raise KeyError(f"Mask key '{selection_key}' not found in {payload_path}")
        mask = payload[key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(dtype=torch.bool).reshape(-1)
        if int(mask.shape[0]) != int(total_gaussians):
            raise ValueError(
                f"Payload mask '{selection_key}' length mismatch: expected {total_gaussians}, got {int(mask.shape[0])}"
            )
        return mask

    masks_root = model_path / "masks"
    child = _load_mask(masks_root / "init_repair_child_output_mask.pt", total_gaussians)
    softened = _load_mask(masks_root / "init_repair_softened_output_mask.pt", total_gaussians)
    if key in {"all", "full"}:
        return torch.ones((total_gaussians,), dtype=torch.bool)
    if key in {"children", "child", "children_only"}:
        return child
    if key in {"softened", "softened_any", "softened_any_only"}:
        return softened
    if key in {"softened_children", "softened_children_only"}:
        return child & softened
    if key in {"unsoftened_children", "unsoftened_children_only"}:
        return child & ~softened
    if key in {"non_children", "non_child", "non_children_only"}:
        return ~child
    raise ValueError(
        f"Unsupported lineage selection_key={selection_key!r}; "
        "use one of full/children/non_children/softened_any/softened_children/unsoftened_children "
        "or selection_source=payload."
    )


def _logit(prob: torch.Tensor) -> torch.Tensor:
    prob = torch.clamp(prob, min=1e-6, max=1.0 - 1e-6)
    return torch.log(prob / torch.clamp(1.0 - prob, min=1e-6))


def _apply_group_variant(
    gaussians: GaussianModel,
    *,
    selected_mask: torch.Tensor,
    selection_mode: str,
    mute_opacity_logit: float,
    rest_scale: float,
    dc_scale: float,
    tau_scale: float,
    scale_multiplier: float,
    scale_axis_mode: str,
    filter_multiplier: float,
) -> GaussianModel:
    full_mask = torch.ones((int(gaussians.get_xyz.shape[0]),), dtype=torch.bool, device=gaussians.get_xyz.device)
    variant = _clone_subset_gaussians(gaussians, full_mask)
    selected = selected_mask.to(device=variant.get_xyz.device, dtype=torch.bool).reshape(-1)
    if int(selected.shape[0]) != int(variant.get_xyz.shape[0]):
        raise ValueError("Selected mask length mismatch for variant export.")

    if float(rest_scale) != 1.0:
        variant._features_rest.data[selected] *= float(rest_scale)
    if float(dc_scale) != 1.0:
        variant._features_dc.data[selected] *= float(dc_scale)
    if float(tau_scale) != 1.0:
        alpha = torch.sigmoid(variant._opacity.data[selected])
        tau = -torch.log(torch.clamp(1.0 - alpha, min=1e-6))
        alpha_scaled = 1.0 - torch.exp(-tau * float(tau_scale))
        variant._opacity.data[selected] = _logit(alpha_scaled)
    if float(scale_multiplier) != 1.0:
        scale = torch.exp(variant._scaling.data[selected])
        if str(scale_axis_mode).strip().lower() == "major_only":
            axis = torch.argmax(scale, dim=1, keepdim=True)
            multiplier = torch.ones_like(scale)
            multiplier.scatter_(1, axis, float(scale_multiplier))
            scale = scale * multiplier
        else:
            scale = scale * float(scale_multiplier)
        variant._scaling.data[selected] = torch.log(torch.clamp(scale, min=1e-8))
    if float(filter_multiplier) != 1.0 and isinstance(variant.filter_3D, torch.Tensor):
        variant.filter_3D[selected] = torch.clamp(variant.filter_3D[selected] * float(filter_multiplier), min=0.0)

    mode = str(selection_mode).strip().lower()
    if mode == "selected_only":
        variant._opacity.data[~selected] = float(mute_opacity_logit)
    elif mode == "selected_removed":
        variant._opacity.data[selected] = float(mute_opacity_logit)
    elif mode != "full":
        raise ValueError(f"Unsupported selection_mode={selection_mode!r}; use full/selected_only/selected_removed.")
    return variant


def export_variant(
    *,
    scene_root: Path,
    model_path: Path,
    iteration: int,
    output_root: Path,
    images_subdir: str,
    split: str,
    max_views: int,
    white_background: bool,
    selection_source: str,
    selection_key: str,
    mask_payload_path: Path | None,
    selection_mode: str,
    mute_opacity_logit: float,
    rest_scale: float,
    dc_scale: float,
    tau_scale: float,
    scale_multiplier: float,
    scale_axis_mode: str,
    filter_multiplier: float,
    save_alpha: bool,
    save_depth: bool,
    save_premul: bool,
) -> Dict[str, object]:
    dataset = _build_dataset_args(str(scene_root), str(model_path), images_subdir, white_background)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = _make_scene(dataset, gaussians, load_iteration=iteration)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    selected_mask = _load_selection_mask(
        model_path=model_path,
        iteration=loaded_iter,
        selection_key=selection_key,
        selection_source=selection_source,
        payload_path=mask_payload_path,
        total_gaussians=int(gaussians.get_xyz.shape[0]),
    ).to(device="cuda")
    selected_count = int(selected_mask.sum().item())
    if selected_count <= 0 and str(selection_mode).strip().lower() != "full":
        raise ValueError(f"Selection '{selection_key}' from {selection_source} selected zero gaussians.")

    variant_root = output_root / "variant_model"
    point_dir = variant_root / "point_cloud" / f"iteration_{loaded_iter}"
    point_dir.mkdir(parents=True, exist_ok=True)
    _copy_render_config(model_path, variant_root)
    variant = _apply_group_variant(
        gaussians,
        selected_mask=selected_mask,
        selection_mode=selection_mode,
        mute_opacity_logit=mute_opacity_logit,
        rest_scale=rest_scale,
        dc_scale=dc_scale,
        tau_scale=tau_scale,
        scale_multiplier=scale_multiplier,
        scale_axis_mode=scale_axis_mode,
        filter_multiplier=filter_multiplier,
    )
    variant.save_ply(str(point_dir / "point_cloud.ply"))
    variant.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    render_dataset = _build_dataset_args(str(scene_root), str(variant_root), images_subdir, white_background)
    render_gaussians = GaussianModel(render_dataset.sh_degree)
    render_scene = _make_scene(render_dataset, render_gaussians, load_iteration=loaded_iter)
    _compute_3d_filter_compat(render_gaussians, render_scene.getTrainCameras().copy())
    background = torch.tensor([1, 1, 1] if white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    render_summary: Dict[str, object] = {}
    for split_name, views in _iter_views(render_scene, split):
        selected_views, selected_indices = _select_uniform(list(views), max_views)
        render_root = output_root / split_name / f"ours_{loaded_iter}" / "renders"
        render_root.mkdir(parents=True, exist_ok=True)
        alpha_root = output_root / split_name / f"ours_{loaded_iter}" / "alpha"
        depth_root = output_root / split_name / f"ours_{loaded_iter}" / "depth"
        premul_root = output_root / split_name / f"ours_{loaded_iter}" / "premul"
        if save_alpha:
            alpha_root.mkdir(parents=True, exist_ok=True)
        if save_depth:
            depth_root.mkdir(parents=True, exist_ok=True)
        if save_premul:
            premul_root.mkdir(parents=True, exist_ok=True)
        for output_idx, view in zip(selected_indices, selected_views):
            render_pkg = render_simple(view, render_gaussians, background)
            rgb = render_pkg["render"][:3]
            alpha = torch.squeeze(render_pkg["alpha"]).clamp(0.0, 1.0)
            if alpha.ndim != 2:
                raise ValueError(
                    f"Expected render alpha to have shape [H, W] or [1, H, W], got {tuple(render_pkg['alpha'].shape)}"
                )
            _save_rgb(render_root / f"{output_idx:05d}.png", rgb)
            if save_alpha:
                _save_gray(alpha_root / f"{output_idx:05d}.png", alpha)
            if save_premul:
                premul = rgb * alpha.unsqueeze(0)
                _save_rgb(premul_root / f"{output_idx:05d}.png", premul)
            if save_depth:
                _save_depth(
                    depth_root / f"{output_idx:05d}.png",
                    depth_root / f"{output_idx:05d}.npy",
                    render_pkg["depth"],
                )
        render_summary[split_name] = {
            "num_views": int(len(selected_views)),
            "source_num_views": int(len(views)),
            "selected_indices": selected_indices,
            "render_root": str(render_root),
            "alpha_root": str(alpha_root) if save_alpha else None,
            "depth_root": str(depth_root) if save_depth else None,
            "premul_root": str(premul_root) if save_premul else None,
        }

    summary = {
        "mode": "export_gaussian_group_variant_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(loaded_iter),
        "images_subdir": str(images_subdir),
        "split": str(split),
        "max_views": int(max_views),
        "white_background": bool(white_background),
        "selection": {
            "source": str(selection_source),
            "key": str(selection_key),
            "selection_mode": str(selection_mode),
            "selected_gaussians": int(selected_count),
            "source_gaussians": int(selected_mask.shape[0]),
            "selected_ratio": float(selected_count / max(int(selected_mask.shape[0]), 1)),
            "mask_payload_path": str(mask_payload_path) if mask_payload_path is not None else None,
        },
        "ablation": {
            "mute_opacity_logit": float(mute_opacity_logit),
            "rest_scale": float(rest_scale),
            "dc_scale": float(dc_scale),
            "tau_scale": float(tau_scale),
            "scale_multiplier": float(scale_multiplier),
            "scale_axis_mode": str(scale_axis_mode),
            "filter_multiplier": float(filter_multiplier),
        },
        "paths": {
            "variant_model_root": str(variant_root),
            "variant_ply": str(point_dir / "point_cloud.ply"),
            "variant_tags": str(point_dir / "gaussian_tags.pt"),
        },
        "renders": render_summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and render a full-model variant using lineage, tracking, or payload masks.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--selection_source", choices=["lineage", "payload", "tracking"], default="lineage")
    parser.add_argument("--selection_key", default="children")
    parser.add_argument("--mask_payload_path", default=None)
    parser.add_argument("--selection_mode", choices=["full", "selected_only", "selected_removed"], default="full")
    parser.add_argument("--mute_opacity_logit", type=float, default=-20.0)
    parser.add_argument("--rest_scale", type=float, default=1.0)
    parser.add_argument("--dc_scale", type=float, default=1.0)
    parser.add_argument("--tau_scale", type=float, default=1.0)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--scale_axis_mode", choices=["all", "major_only"], default="all")
    parser.add_argument("--filter_multiplier", type=float, default=1.0)
    parser.add_argument("--save_alpha", action="store_true")
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_premul", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    iteration = _resolve_iteration(model_path, int(args.iteration))
    mask_payload_path = Path(args.mask_payload_path).expanduser().resolve() if args.mask_payload_path else None

    summary = export_variant(
        scene_root=scene_root,
        model_path=model_path,
        iteration=iteration,
        output_root=output_root,
        images_subdir=str(args.images_subdir),
        split=str(args.split),
        max_views=int(args.max_views),
        white_background=bool(args.white_background),
        selection_source=str(args.selection_source),
        selection_key=str(args.selection_key),
        mask_payload_path=mask_payload_path,
        selection_mode=str(args.selection_mode),
        mute_opacity_logit=float(args.mute_opacity_logit),
        rest_scale=float(args.rest_scale),
        dc_scale=float(args.dc_scale),
        tau_scale=float(args.tau_scale),
        scale_multiplier=float(args.scale_multiplier),
        scale_axis_mode=str(args.scale_axis_mode),
        filter_multiplier=float(args.filter_multiplier),
        save_alpha=bool(args.save_alpha),
        save_depth=bool(args.save_depth),
        save_premul=bool(args.save_premul),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
