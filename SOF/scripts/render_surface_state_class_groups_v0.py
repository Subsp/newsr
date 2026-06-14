#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_gaussian_mask_subset_v0 import (  # noqa: E402
    _build_dataset_args,
    _clone_subset_gaussians,
    _iter_views,
    _resolve_iteration,
    _save_rgb,
    _select_uniform,
)
from gaussian_renderer import render_simple  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


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


CLASS_ID_TO_NAME = {
    0: "no_mesh_neutral",
    1: "surface_carrier",
    2: "near_surface_uncertain",
    3: "off_surface_near_mesh",
    4: "axis_touching_surface",
    5: "low_opacity_neutral",
}

GROUP_ALIASES = {
    "full": "full",
    "all": "full",
    "no_mesh": "no_mesh_neutral",
    "no_mesh_neutral": "no_mesh_neutral",
    "carrier": "surface_carrier",
    "surface": "surface_carrier",
    "surface_carrier": "surface_carrier",
    "near": "near_surface_uncertain",
    "uncertain": "near_surface_uncertain",
    "near_surface": "near_surface_uncertain",
    "near_surface_uncertain": "near_surface_uncertain",
    "off_surface": "off_surface_near_mesh",
    "off_surface_near_mesh": "off_surface_near_mesh",
    "axis_touch": "axis_touching_surface",
    "axis_touching": "axis_touching_surface",
    "axis_touching_surface": "axis_touching_surface",
    "low_opacity": "low_opacity_neutral",
    "low_opacity_neutral": "low_opacity_neutral",
    "surface_candidate": "surface_candidate",
    "carrier_plus_axis": "surface_candidate",
    "surface_or_uncertain": "surface_or_uncertain",
    "carrier_plus_uncertain": "surface_or_uncertain",
    "non_surface_active": "non_surface_active",
    "surface_complement_active": "non_surface_active",
    "near_or_off_surface": "non_surface_active",
    "uncertain_or_off_surface": "uncertain_or_off_surface",
}

COMPOSITE_GROUPS = {
    "surface_candidate": ["surface_carrier", "axis_touching_surface"],
    "surface_or_uncertain": ["surface_carrier", "near_surface_uncertain", "axis_touching_surface"],
    "non_surface_active": ["no_mesh_neutral", "near_surface_uncertain", "off_surface_near_mesh"],
    "uncertain_or_off_surface": ["near_surface_uncertain", "off_surface_near_mesh"],
}


def _tensor_1d(value: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().cpu().reshape(-1)
    else:
        out = torch.as_tensor(value).reshape(-1)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _torch_load(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_gray(path: Path, image_hw: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = torch.squeeze(image_hw.detach().float().cpu()).clamp(0.0, 1.0)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image [H, W], got {tuple(image.shape)}")
    Image.fromarray((image.numpy() * 255.0).astype(np.uint8)).save(path)


def _canonical_group(group: str) -> str:
    token = str(group).strip().lower()
    if not token:
        raise ValueError("Empty group name")
    return GROUP_ALIASES.get(token, token)


def _payload_bool_mask(payload: Dict[str, object], key: str, expected: int) -> torch.Tensor | None:
    if key not in payload:
        return None
    mask = _tensor_1d(payload[key], dtype=torch.bool)
    if int(mask.shape[0]) != int(expected):
        raise ValueError(f"Payload mask '{key}' length mismatch: expected {expected}, got {int(mask.shape[0])}")
    return mask


def _class_name_to_id(class_names: object) -> Dict[str, int]:
    mapping = {name: idx for idx, name in CLASS_ID_TO_NAME.items()}
    if isinstance(class_names, dict):
        for key, value in class_names.items():
            try:
                idx = int(key)
            except Exception:
                continue
            mapping[str(value)] = idx
    return mapping


def _mask_for_group(
    *,
    payload: Dict[str, object],
    class_id: torch.Tensor,
    group: str,
    name_to_id: Dict[str, int],
) -> torch.Tensor:
    canonical = _canonical_group(group)
    total = int(class_id.shape[0])
    if canonical == "full":
        return torch.ones((total,), dtype=torch.bool)

    explicit = _payload_bool_mask(payload, canonical, total)
    if explicit is not None:
        return explicit

    if canonical in COMPOSITE_GROUPS:
        mask = torch.zeros((total,), dtype=torch.bool)
        for child in COMPOSITE_GROUPS[canonical]:
            mask |= _mask_for_group(payload=payload, class_id=class_id, group=child, name_to_id=name_to_id)
        return mask

    if canonical.startswith("class_"):
        class_idx = int(canonical.split("_", 1)[1])
    elif canonical.isdigit():
        class_idx = int(canonical)
    elif canonical in name_to_id:
        class_idx = int(name_to_id[canonical])
    else:
        raise KeyError(f"Unknown surface-state group '{group}' (canonical='{canonical}')")
    return class_id == int(class_idx)


def _parse_groups(value: str) -> List[str]:
    groups = [_canonical_group(item) for item in str(value).split(",") if item.strip()]
    if not groups:
        raise ValueError("No groups requested.")
    return groups


def _iter_requested_groups(groups: Sequence[str]) -> Iterable[str]:
    seen: set[str] = set()
    for group in groups:
        if group in seen:
            continue
        seen.add(group)
        yield group


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Gaussian subsets from gaussian_surface_state_v0 class payloads.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--surface_state_payload", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--flat_output_root", default="")
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--groups", default="surface_carrier,near_surface_uncertain")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--save_alpha", action="store_true")
    parser.add_argument("--save_premul", action="store_true")
    parser.add_argument("--save_composite", action="store_true")
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--recompute_filter_3d", action="store_true")
    parser.add_argument(
        "--vanilla_mip_mode",
        action="store_true",
        help="Render subsets with mip-splatting-style prefiltered scale/opacity instead of SOF aux filter inputs.",
    )
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    payload_path = Path(args.surface_state_payload).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    flat_output_root = None
    if str(args.flat_output_root).strip():
        flat_output_root = Path(args.flat_output_root).expanduser().resolve()
        flat_output_root.mkdir(parents=True, exist_ok=True)

    payload = _torch_load(payload_path)
    if "class_id" not in payload:
        raise KeyError(f"Surface-state payload does not contain class_id: {payload_path}")
    class_id = _tensor_1d(payload["class_id"], dtype=torch.long)
    name_to_id = _class_name_to_id(payload.get("class_names", {}))
    requested_groups = list(_iter_requested_groups(_parse_groups(str(args.groups))))

    iteration = _resolve_iteration(model_path, int(args.iteration))
    dataset = _build_dataset_args(str(scene_root), str(model_path), str(args.images_subdir), bool(args.white_background))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    total = int(gaussians.get_xyz.shape[0])
    if int(class_id.shape[0]) != total:
        raise ValueError(f"class_id length mismatch: payload={int(class_id.shape[0])}, model={total}")

    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    composite_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    train_cameras = scene.getTrainCameras().copy()
    for camera in train_cameras + scene.getTestCameras().copy():
        setattr(camera, "kernel_size", float(getattr(dataset, "kernel_size", 0.0)))
        setattr(camera, "vanilla_mip_mode", bool(args.vanilla_mip_mode))

    summary: Dict[str, object] = {
        "version": "render_surface_state_class_groups_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "surface_state_payload": str(payload_path),
        "output_root": str(output_root),
        "flat_output_root": str(flat_output_root) if flat_output_root is not None else None,
        "images_subdir": str(args.images_subdir),
        "iteration": int(loaded_iter),
        "split": str(args.split),
        "max_views": int(args.max_views),
        "vanilla_mip_mode": bool(args.vanilla_mip_mode),
        "kernel_size": float(getattr(dataset, "kernel_size", 0.0)),
        "source_gaussians": total,
        "groups": {},
    }

    for group in requested_groups:
        mask_cpu = _mask_for_group(payload=payload, class_id=class_id, group=group, name_to_id=name_to_id)
        count = int(mask_cpu.sum().item())
        if count <= 0:
            if bool(args.skip_empty):
                summary["groups"][group] = {"selected_gaussians": 0, "skipped": True}
                continue
            raise ValueError(f"Group '{group}' selected zero Gaussians.")

        print(f"[surface-state-render-v0] group={group} selected={count}/{total}", flush=True)
        subset = _clone_subset_gaussians(gaussians, mask_cpu.to(device=gaussians.get_xyz.device))
        if bool(args.recompute_filter_3d) or not _has_loaded_filter_3d(subset):
            subset.compute_3D_filter(train_cameras, CUDA=False)

        group_summary: Dict[str, object] = {
            "selected_gaussians": count,
            "selected_ratio": float(count / max(total, 1)),
            "renders": {},
        }
        for split_name, views in _iter_views(scene, str(args.split)):
            selected_views, selected_indices = _select_uniform(list(views), int(args.max_views))
            render_root = output_root / group / split_name / f"ours_{loaded_iter}" / "renders"
            alpha_root = output_root / group / split_name / f"ours_{loaded_iter}" / "alpha"
            premul_root = output_root / group / split_name / f"ours_{loaded_iter}" / "premul"
            composite_root = output_root / group / split_name / f"ours_{loaded_iter}" / "composite_white"
            render_root.mkdir(parents=True, exist_ok=True)
            if bool(args.save_alpha):
                alpha_root.mkdir(parents=True, exist_ok=True)
            if bool(args.save_premul):
                premul_root.mkdir(parents=True, exist_ok=True)
            if bool(args.save_composite):
                composite_root.mkdir(parents=True, exist_ok=True)

            for output_idx, view in zip(selected_indices, selected_views):
                render_pkg = render_simple(
                    view,
                    subset,
                    background,
                    kernel_size=float(getattr(dataset, "kernel_size", 0.0)),
                    vanilla_mip_mode=bool(args.vanilla_mip_mode),
                )
                rgb = render_pkg["render"][:3]
                _save_rgb(render_root / f"{output_idx:05d}.png", rgb)
                alpha = torch.squeeze(render_pkg["alpha"]).clamp(0.0, 1.0)
                premul_rgb = (
                    rgb - background[:, None, None] * (1.0 - alpha.unsqueeze(0))
                ).clamp(0.0, 1.0)
                if bool(args.save_alpha):
                    _save_gray(alpha_root / f"{output_idx:05d}.png", alpha)
                if bool(args.save_premul):
                    _save_rgb(premul_root / f"{output_idx:05d}.png", premul_rgb)
                if bool(args.save_composite):
                    composite = premul_rgb + composite_background[:, None, None] * (1.0 - alpha.unsqueeze(0))
                    _save_rgb(composite_root / f"{output_idx:05d}.png", composite)
                if flat_output_root is not None:
                    stem = f"{split_name}_{output_idx:05d}_{group}"
                    _save_rgb(flat_output_root / f"{stem}_raw_black.png", rgb)
                    if bool(args.save_premul):
                        _save_rgb(flat_output_root / f"{stem}_premul.png", premul_rgb)
                    if bool(args.save_alpha):
                        _save_gray(flat_output_root / f"{stem}_alpha.png", alpha)
                    if bool(args.save_composite):
                        _save_rgb(flat_output_root / f"{stem}_composite_white.png", composite)

            group_summary["renders"][split_name] = {
                "num_views": int(len(selected_views)),
                "source_num_views": int(len(views)),
                "selected_indices": selected_indices,
                "render_root": str(render_root),
                "alpha_root": str(alpha_root) if bool(args.save_alpha) else None,
                "premul_root": str(premul_root) if bool(args.save_premul) else None,
                "composite_root": str(composite_root) if bool(args.save_composite) else None,
            }

        summary["groups"][group] = group_summary

    summary_path = output_root / "render_surface_state_class_groups_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if flat_output_root is not None:
        flat_summary_path = flat_output_root / "summary.json"
        flat_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[surface-state-render-v0] flat summary: {flat_summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[surface-state-render-v0] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
