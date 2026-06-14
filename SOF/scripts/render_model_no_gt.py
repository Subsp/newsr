from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
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


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


def _iter_views(scene: Scene, split: str):
    if split == "train":
        return [("train", scene.getTrainCameras())]
    if split == "test":
        return [("test", scene.getTestCameras())]
    return [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]


def _select_uniform(items, max_items: int):
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return list(items), list(range(len(items)))
    ids = np.unique(np.linspace(0, len(items) - 1, num=int(max_items), dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()], [int(idx) for idx in ids.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Gaussian model without copying GT images.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--recompute_filter_3d", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    iteration = _resolve_iteration(model_path, int(args.iteration))
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_path
    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    if bool(args.recompute_filter_3d) or not _has_loaded_filter_3d(gaussians):
        gaussians.compute_3D_filter(scene.getTrainCameras().copy(), CUDA=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    summary = {
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "output_root": str(output_root),
        "images_subdir": str(args.images_subdir),
        "iteration": int(loaded_iter),
        "split": str(args.split),
        "renders": {},
    }
    for split_name, views in _iter_views(scene, str(args.split)):
        selected_views, selected_indices = _select_uniform(list(views), int(args.max_views))
        render_root = output_root / split_name / f"ours_{loaded_iter}" / "renders"
        render_root.mkdir(parents=True, exist_ok=True)
        for output_idx, view in zip(selected_indices, selected_views):
            render_pkg = render_simple(view, gaussians, background)
            _save_rgb(render_root / f"{output_idx:05d}.png", render_pkg["render"][:3])
        summary["renders"][split_name] = {
            "num_views": int(len(selected_views)),
            "source_num_views": int(len(views)),
            "selected_indices": selected_indices,
            "render_root": str(render_root),
        }
    summary_path = output_root / "render_model_no_gt_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
