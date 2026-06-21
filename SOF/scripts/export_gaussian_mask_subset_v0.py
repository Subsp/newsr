#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
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
        compute_filter3D_python = False
        debug = False
        use_merged_sof_rasterizer = False
        use_vanilla_sof_rasterizer = False

    def render_simple(viewpoint_camera, pc, bg_color):
        render_pkg = _render(
            viewpoint_camera,
            pc,
            _PreviewPipeline(),
            bg_color,
            kernel_size=0.0,
        )
        if "alpha" not in render_pkg:
            rgb = render_pkg["render"][:3]
            alpha = torch.linalg.vector_norm(rgb - bg_color.reshape(3, 1, 1), dim=0, keepdim=True)
            render_pkg["alpha"] = (alpha > 1e-4).to(dtype=rgb.dtype)
        return render_pkg
from scene import Scene
from scene.gaussian_model import GaussianModel


def _has_loaded_filter_3d(gaussians: GaussianModel) -> bool:
    filter_3d = getattr(gaussians, "filter_3D", None)
    if not isinstance(filter_3d, torch.Tensor):
        return False
    if filter_3d.ndim == 0 or int(filter_3d.shape[0]) != int(gaussians.get_xyz.shape[0]):
        return False
    if filter_3d.numel() == 0:
        return False
    if not bool(torch.isfinite(filter_3d).all().item()):
        return False
    return bool((filter_3d > 0).any().item())


def _tensor_1d(value, *, dtype=None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.reshape(-1)


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


def _resolve_iteration(model_path: Path, iteration: int) -> int:
    if int(iteration) >= 0:
        return int(iteration)
    point_root = model_path / "point_cloud"
    candidates: List[int] = []
    for child in point_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_root}")
    return max(candidates)


def _iter_views(scene: Scene, split: str):
    if split == "train":
        return [("train", scene.getTrainCameras())]
    if split == "test":
        return [("test", scene.getTestCameras())]
    return [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]


def _make_scene(dataset, gaussians, *, load_iteration: int):
    kwargs = {"load_iteration": load_iteration, "shuffle": False}
    params = inspect.signature(Scene.__init__).parameters
    if "skip_test" in params:
        kwargs["skip_test"] = False
    if "skip_train" in params:
        kwargs["skip_train"] = False
    return Scene(dataset, gaussians, **kwargs)


def _select_uniform(items: Sequence[object], max_items: int):
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items), list(range(len(items)))
    ids = np.unique(np.linspace(0, len(items) - 1, num=int(max_items), dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()], [int(idx) for idx in ids.tolist()]


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src = src_model_path / name
        if src.exists():
            shutil.copy2(src, dst_model_path / name)


def _clone_subset_gaussians(base: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    if hasattr(base, "use_SBs"):
        subset = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    else:
        subset = GaussianModel(base.max_sh_degree)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[mask].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[mask].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[mask].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[mask].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[mask].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[mask].clone().requires_grad_(False))
    if (
        isinstance(base.filter_3D, torch.Tensor)
        and base.filter_3D.ndim > 0
        and base.filter_3D.shape[0] == base.get_xyz.shape[0]
    ):
        subset.filter_3D = base.filter_3D.detach()[mask].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.init_tracking_state(count)
    for name in ("_source_tag", "_seed_id", "_generation", "_edge_touched", "_edge_touch_iter"):
        value = getattr(base, name, None)
        if torch.is_tensor(value) and int(value.shape[0]) == int(base.get_xyz.shape[0]):
            setattr(subset, name, value.detach()[mask].clone())
    return subset


def _load_mask_payload(mask_payload_path: Path, mask_key: str, expected_count: int) -> torch.Tensor:
    payload = torch.load(mask_payload_path, map_location="cpu")
    if mask_key not in payload:
        class_id = payload.get("class_id")
        if class_id is None:
            raise KeyError(f"Mask key '{mask_key}' not found in {mask_payload_path}")
        class_id = _tensor_1d(class_id, dtype=torch.long)
        if int(class_id.shape[0]) != int(expected_count):
            raise ValueError(
                f"class_id length mismatch in {mask_payload_path}: "
                f"expected {expected_count}, got {int(class_id.shape[0])}"
            )
        low_opacity = class_id == 5
        surface_candidate = (class_id == 1) | (class_id == 4)
        non_surface_active = (~low_opacity) & (~surface_candidate)
        if mask_key == "no_mesh_neutral":
            return class_id == 0
        if mask_key == "surface_carrier":
            return class_id == 1
        if mask_key == "near_surface_uncertain":
            return class_id == 2
        if mask_key == "off_surface_near_mesh":
            return class_id == 3
        if mask_key == "axis_touching_surface":
            return class_id == 4
        if mask_key == "low_opacity_neutral":
            return low_opacity
        if mask_key == "surface_candidate":
            return surface_candidate
        if mask_key == "surface_or_uncertain":
            return (class_id == 1) | (class_id == 2) | (class_id == 4)
        if mask_key in {"non_surface_active", "surface_complement_active", "near_or_off_surface"}:
            return non_surface_active
        if mask_key == "uncertain_or_off_surface":
            return (class_id == 2) | (class_id == 3)
        raise KeyError(f"Mask key '{mask_key}' not found in {mask_payload_path}")
    mask = _tensor_1d(payload[mask_key], dtype=torch.bool)
    if int(mask.shape[0]) != int(expected_count):
        raise ValueError(
            f"Mask length mismatch for key '{mask_key}': expected {expected_count}, got {int(mask.shape[0])}"
        )
    return mask


def export_subset_model(
    *,
    scene_root: Path,
    model_path: Path,
    iteration: int,
    mask_payload_path: Path,
    mask_key: str,
    output_root: Path,
    images_subdir: str,
    split: str,
    max_views: int,
    white_background: bool,
) -> Dict[str, object]:
    dataset_args = _build_dataset_args(str(scene_root), str(model_path), images_subdir, white_background)
    dataset = dataset_args
    gaussians = GaussianModel(dataset.sh_degree)
    scene = _make_scene(dataset, gaussians, load_iteration=iteration)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)

    mask = _load_mask_payload(mask_payload_path, mask_key, int(gaussians.get_xyz.shape[0])).to(device="cuda")
    subset_count = int(mask.sum().item())
    if subset_count <= 0:
        raise ValueError(f"Mask '{mask_key}' from {mask_payload_path} selected zero gaussians.")

    subset_root = output_root / "subset_model"
    point_dir = subset_root / "point_cloud" / f"iteration_{loaded_iter}"
    point_dir.mkdir(parents=True, exist_ok=True)
    _copy_render_config(model_path, subset_root)
    subset = _clone_subset_gaussians(gaussians, mask)
    subset.save_ply(str(point_dir / "point_cloud.ply"))
    subset.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    render_dataset = _build_dataset_args(str(scene_root), str(subset_root), images_subdir, white_background)
    render_gaussians = GaussianModel(render_dataset.sh_degree)
    render_scene = _make_scene(render_dataset, render_gaussians, load_iteration=loaded_iter)
    if not _has_loaded_filter_3d(render_gaussians):
        render_gaussians.compute_3D_filter(render_scene.getTrainCameras().copy(), CUDA=False)
    background = torch.tensor(
        [1, 1, 1] if white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    render_summary: Dict[str, object] = {}
    for split_name, views in _iter_views(render_scene, split):
        selected_views, selected_indices = _select_uniform(list(views), max_views)
        render_root = output_root / split_name / f"ours_{loaded_iter}" / "renders"
        render_root.mkdir(parents=True, exist_ok=True)
        for output_idx, view in zip(selected_indices, selected_views):
            render_pkg = render_simple(view, render_gaussians, background)
            _save_rgb(render_root / f"{output_idx:05d}.png", render_pkg["render"][:3])
        render_summary[split_name] = {
            "num_views": int(len(selected_views)),
            "source_num_views": int(len(views)),
            "selected_indices": selected_indices,
            "render_root": str(render_root),
        }

    summary = {
        "mode": "export_gaussian_mask_subset_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(loaded_iter),
        "mask_payload_path": str(mask_payload_path),
        "mask_key": str(mask_key),
        "images_subdir": str(images_subdir),
        "split": str(split),
        "max_views": int(max_views),
        "white_background": bool(white_background),
        "counts": {
            "source_gaussians": int(mask.shape[0]),
            "selected_gaussians": int(subset_count),
            "selected_ratio": float(subset_count / max(int(mask.shape[0]), 1)),
        },
        "paths": {
            "subset_model_root": str(subset_root),
            "subset_ply": str(point_dir / "point_cloud.ply"),
            "subset_tags": str(point_dir / "gaussian_tags.pt"),
        },
        "renders": render_summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and render a gaussian subset selected by an external mask payload.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mask_payload_path", required=True)
    parser.add_argument("--mask_key", default="starburst_candidate")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    mask_payload_path = Path(args.mask_payload_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    iteration = _resolve_iteration(model_path, int(args.iteration))

    summary = export_subset_model(
        scene_root=scene_root,
        model_path=model_path,
        iteration=iteration,
        mask_payload_path=mask_payload_path,
        mask_key=str(args.mask_key),
        output_root=output_root,
        images_subdir=str(args.images_subdir),
        split=str(args.split),
        max_views=int(args.max_views),
        white_background=bool(args.white_background),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
