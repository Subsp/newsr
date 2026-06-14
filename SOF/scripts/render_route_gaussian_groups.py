from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.route_executor import load_route_payload, vector_stats


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


def _resolve_model_iteration(model_path: Path, iteration: int) -> int:
    if iteration >= 0:
        return int(iteration)
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")
    candidates: List[int] = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_root}")
    return int(max(candidates))


def _save_rgb(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


def _clone_subset_gaussians(base: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=base.get_xyz.device, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    subset = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
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
        subset.filter_3D = base.filter_3D.detach().clone()

    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device="cuda")
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device="cuda")
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device="cuda")

    subset.init_tracking_state(count)
    if base._source_tag.shape[0] == base.get_xyz.shape[0]:
        subset._source_tag = base._source_tag.detach()[mask].clone()
    if base._seed_id.shape[0] == base.get_xyz.shape[0]:
        subset._seed_id = base._seed_id.detach()[mask].clone()
    if base._generation.shape[0] == base.get_xyz.shape[0]:
        subset._generation = base._generation.detach()[mask].clone()
    if base._edge_touched.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touched = base._edge_touched.detach()[mask].clone()
    if base._edge_touch_iter.shape[0] == base.get_xyz.shape[0]:
        subset._edge_touch_iter = base._edge_touch_iter.detach()[mask].clone()
    return subset


def _canonical_group_name(name: str) -> str:
    lowered = str(name).strip().lower()
    if lowered in {"all", "full", "detail_all", "full_detail"}:
        return "full"
    return lowered


def _build_group_masks(
    route_payload: Dict[str, torch.Tensor],
    groups: Iterable[str],
    group_mode: str,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    p_attach = route_payload.get("p_attach")
    p_detail = route_payload.get("p_detail")
    p_suppress = route_payload.get("p_suppress")
    if p_attach is None or p_detail is None or p_suppress is None:
        raise KeyError("Route payload must contain p_attach, p_detail, and p_suppress for group rendering.")

    p_attach = p_attach.reshape(-1).float()
    p_detail = p_detail.reshape(-1).float()
    p_suppress = p_suppress.reshape(-1).float()
    total = int(p_attach.shape[0])

    canonical_groups = [_canonical_group_name(name) for name in groups]
    masks: Dict[str, torch.Tensor] = {}
    if "full" in canonical_groups:
        masks["full"] = torch.ones((total,), dtype=torch.bool)

    if group_mode == "argmax":
        ownership = torch.stack((p_attach, p_detail, p_suppress), dim=1)
        owner = ownership.argmax(dim=1)
        candidate_masks = {
            "attach": owner == 0,
            "detail": owner == 1,
            "suppress": owner == 2,
        }
    elif group_mode == "threshold":
        threshold = float(threshold)
        candidate_masks = {
            "attach": p_attach >= threshold,
            "detail": p_detail >= threshold,
            "suppress": p_suppress >= threshold,
        }
    else:
        raise ValueError(f"Unsupported group_mode: {group_mode}")

    for group_name in ("attach", "detail", "suppress"):
        if group_name in canonical_groups:
            masks[group_name] = candidate_masks[group_name]
    return masks


def _iter_splits(scene: Scene, split: str):
    if split == "train":
        return [("train", scene.getTrainCameras())]
    if split == "test":
        return [("test", scene.getTestCameras())]
    return [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render route-classified Gaussian groups without copying GT images.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--detail_model_path", required=True)
    parser.add_argument("--route_payload", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--groups", default="full,attach,detail,suppress")
    parser.add_argument("--group_mode", choices=["argmax", "threshold"], default="threshold")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    detail_model_path = Path(args.detail_model_path).expanduser().resolve()
    route_payload_path = Path(args.route_payload).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_model_iteration(detail_model_path, int(args.iteration))
    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(detail_model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    route_payload = load_route_payload(route_payload_path, total_gaussians=int(gaussians.get_xyz.shape[0]))
    if route_payload is None:
        raise RuntimeError(f"Failed to load route payload: {route_payload_path}")

    group_names = [name.strip() for name in str(args.groups).split(",") if name.strip()]
    group_masks = _build_group_masks(
        route_payload=route_payload,
        groups=group_names,
        group_mode=str(args.group_mode),
        threshold=float(args.threshold),
    )
    if not group_masks:
        hint = ""
        if group_names == ["0"]:
            hint = " Detected shell GROUPS='0'; use ROUTE_GROUPS=full,attach,detail,suppress with the wrapper script."
        raise RuntimeError(f"No route groups were constructed from groups={group_names!r}.{hint}")

    train_cameras = scene.getTrainCameras().copy()
    bg_color = [1, 1, 1] if bool(args.white_background) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    summary = {
        "scene_root": str(scene_root),
        "detail_model_path": str(detail_model_path),
        "route_payload": str(route_payload_path),
        "output_dir": str(output_dir),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "group_mode": str(args.group_mode),
        "threshold": float(args.threshold),
        "iteration": loaded_iter,
        "groups_requested": group_names,
        "group_counts": {},
        "route_stats": {},
    }

    print(
        f"[route-group-render] detail model  : {detail_model_path}\n"
        f"[route-group-render] route payload : {route_payload_path}\n"
        f"[route-group-render] output dir    : {output_dir}\n"
        f"[route-group-render] group mode    : {args.group_mode}\n"
        f"[route-group-render] iteration     : {loaded_iter}\n"
        f"[route-group-render] groups        : {group_names}"
    )
    for key in ("p_attach", "p_detail", "p_suppress"):
        if key in route_payload:
            stats = vector_stats(route_payload[key])
            summary["route_stats"][key] = stats
            print(f"[route-group-render] {key} stats: {stats}", flush=True)

    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for group_name, mask in group_masks.items():
        selected = int(mask.sum().item())
        total = int(mask.numel())
        summary["group_counts"][group_name] = {
            "selected": selected,
            "total": total,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[route-group-render] group={group_name} selected={selected}/{total}", flush=True)
        if selected <= 0:
            continue
        subset = _clone_subset_gaussians(gaussians, mask)
        subset.compute_3D_filter(train_cameras.copy())

        for split_name, views in _iter_splits(scene, str(args.split)):
            if len(views) <= 0:
                print(f"[route-group-render] skip empty split={split_name}")
                continue
            render_root = output_dir / split_name / group_name / f"ours_{loaded_iter}" / "renders"
            render_root.mkdir(parents=True, exist_ok=True)
            for idx, view in enumerate(views):
                render_pkg = render_simple(view, subset, background)
                render_rgb = render_pkg["render"][:3]
                _save_rgb(render_root / f"{idx:05d}.png", render_rgb)
            summary["group_counts"][group_name][split_name] = {
                "num_views": int(len(views)),
                "render_root": str(render_root),
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("[route-group-render] done", flush=True)


if __name__ == "__main__":
    main()
