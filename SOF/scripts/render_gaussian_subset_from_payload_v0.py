#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from PIL import Image
from tqdm import tqdm

SOF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIPSPLATTING_ROOT = SOF_ROOT.parent / "mip-splatting"


def _ensure_mipsplatting_imports(root: Path) -> None:
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, resolution: int, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=int(resolution),
        white_background=white_background,
        data_device="cuda",
        eval=True,
        kernel_size=0.1,
        ray_jitter=False,
        resample_gt_image=False,
        load_allres=False,
        sample_more_highres=False,
    )


def _build_pipe_args() -> Namespace:
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        compute_filter3D_python=False,
        compute_view2gaussian_python=False,
        use_merged_sof_rasterizer=False,
        use_vanilla_sof_rasterizer=False,
        require_merged_sof_aux=False,
        debug=False,
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


def _clone_subset_gaussians(base, ids: torch.Tensor):
    from scene.gaussian_model import GaussianModel

    ids = ids.to(device=base.get_xyz.device, dtype=torch.long).reshape(-1)
    count = int(ids.shape[0])
    subset = GaussianModel(base.max_sh_degree)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[ids].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[ids].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[ids].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[ids].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[ids].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[ids].clone().requires_grad_(False))
    if isinstance(getattr(base, "filter_3D", None), torch.Tensor) and int(base.filter_3D.shape[0]) == int(base.get_xyz.shape[0]):
        subset.filter_3D = base.filter_3D.detach()[ids].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    if hasattr(subset, "xyz_gradient_accum_abs"):
        subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    if hasattr(subset, "xyz_gradient_accum_abs_max"):
        subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    return subset


def _load_bool(payload: Dict[str, object], key: str, total: int) -> Optional[torch.Tensor]:
    if not key:
        return None
    if key not in payload:
        raise KeyError(f"Mask key '{key}' not found in payload.")
    value = payload[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.reshape(-1)
    if int(value.shape[0]) != int(total):
        raise ValueError(f"Mask key '{key}' length mismatch: {int(value.shape[0])} vs {total}")
    return value.to(dtype=torch.bool)


def _load_float(payload: Dict[str, object], key: str, total: int) -> Optional[torch.Tensor]:
    if not key:
        return None
    if key not in payload:
        raise KeyError(f"Score key '{key}' not found in payload.")
    value = payload[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.reshape(-1)
    if int(value.shape[0]) != int(total):
        raise ValueError(f"Score key '{key}' length mismatch: {int(value.shape[0])} vs {total}")
    return value.to(dtype=torch.float32)


def _mask_from_payload(
    payload: Dict[str, object],
    *,
    total: int,
    mask_key: str,
    score_key: str,
    candidate_key: str,
    include_mask_key: str,
    keep_ratio: float,
    min_score: float,
) -> Dict[str, object]:
    if score_key:
        score = _load_float(payload, score_key, total)
        candidate = _load_bool(payload, candidate_key, total) if candidate_key else (score > 0.0)
        valid = candidate & torch.isfinite(score) & (score > 0.0)
        if int(valid.sum().item()) <= 0:
            raise ValueError(f"No valid scored candidates from score_key={score_key} candidate_key={candidate_key}")
        keep_ratio = float(np.clip(float(keep_ratio), 0.0, 1.0))
        values = score[valid]
        if keep_ratio <= 0.0:
            threshold = float("inf")
        elif keep_ratio >= 1.0:
            threshold = float(min_score)
        else:
            threshold = float(torch.quantile(values, 1.0 - keep_ratio).item())
            threshold = max(float(min_score), threshold)
        mask = valid & (score >= threshold)
        include_mask = _load_bool(payload, include_mask_key, total) if include_mask_key else None
        if include_mask is not None:
            mask = mask | include_mask
        return {
            "mask": mask,
            "mode": "scheduled_score",
            "threshold": threshold if np.isfinite(threshold) else None,
            "score_key": score_key,
            "candidate_key": candidate_key,
            "include_mask_key": include_mask_key,
            "keep_ratio": keep_ratio,
            "min_score": float(min_score),
        }

    mask = _load_bool(payload, mask_key, total)
    if mask is None:
        raise ValueError("Either --score_key or --mask_key must be provided.")
    return {
        "mask": mask,
        "mode": "mask_key",
        "mask_key": mask_key,
    }


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 2:
        arr = (tensor.numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        arr = (tensor[0].numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    if tensor.ndim == 3 and tensor.shape[0] >= 3:
        arr = (tensor[:3].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported tensor shape for image save: {tuple(tensor.shape)}")


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_image(tensor).save(path)


def _laplacian_abs(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image, got {tuple(image.shape)}")
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(int(image.shape[0]), 1, 1, 1)
    hp = torch_F.conv2d(image[None], kernel, padding=1, groups=int(image.shape[0]))[0]
    return hp.abs().mean(dim=0, keepdim=True)


def _normalize_gray(image: torch.Tensor, percentile: float) -> torch.Tensor:
    flat = image.detach().flatten()
    if int(flat.numel()) <= 0:
        return image * 0.0
    scale = torch.quantile(flat, float(np.clip(percentile, 0.0, 100.0)) / 100.0)
    return (image / scale.clamp_min(1e-8)).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Gaussian subset selected by a payload mask/score.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--mask_payload", required=True)
    parser.add_argument("--mask_key", default="")
    parser.add_argument("--score_key", default="hf_score")
    parser.add_argument("--candidate_key", default="hf_candidate")
    parser.add_argument("--include_mask_key", default="hf_owned")
    parser.add_argument("--keep_ratio", type=float, default=0.55)
    parser.add_argument("--min_score", type=float, default=0.03)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mipsplatting_root", default=str(DEFAULT_MIPSPLATTING_ROOT))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--render_lf", action="store_true")
    parser.add_argument("--render_all", action="store_true")
    parser.add_argument("--hf_percentile", type=float, default=99.0)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    _ensure_mipsplatting_imports(Path(args.mipsplatting_root))
    from gaussian_renderer import render  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel  # type: ignore

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    payload_path = Path(args.mask_payload).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    iteration = _resolve_iteration(model_dir, int(args.iteration))
    dataset = _build_dataset_args(
        str(scene_root),
        str(model_dir),
        str(args.images_subdir),
        int(args.resolution),
        bool(args.white_background),
    )
    pipe = _build_pipe_args()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    cameras = list(scene.getTestCameras() if str(args.split) == "test" else scene.getTrainCameras())
    if int(args.limit) > 0:
        cameras = cameras[: int(args.limit)]
    if not cameras:
        raise RuntimeError(f"No {args.split} cameras selected.")

    try:
        gaussians.compute_3D_filter(list(scene.getTrainCameras()), CUDA=not bool(pipe.compute_filter3D_python))
    except TypeError:
        gaussians.compute_3D_filter(list(scene.getTrainCameras()))

    total = int(gaussians.get_xyz.shape[0])
    payload = torch.load(payload_path, map_location="cpu")
    selected = _mask_from_payload(
        payload,
        total=total,
        mask_key=str(args.mask_key),
        score_key=str(args.score_key),
        candidate_key=str(args.candidate_key),
        include_mask_key=str(args.include_mask_key),
        keep_ratio=float(args.keep_ratio),
        min_score=float(args.min_score),
    )
    hf_mask = selected["mask"].to(device=gaussians.get_xyz.device, dtype=torch.bool)
    hf_ids = torch.nonzero(hf_mask, as_tuple=False).reshape(-1)
    if int(hf_ids.numel()) <= 0:
        raise ValueError("Selected HF subset is empty.")
    hf_gaussians = _clone_subset_gaussians(gaussians, hf_ids)

    lf_gaussians = None
    lf_count = 0
    if bool(args.render_lf):
        lf_mask = ~hf_mask
        lf_ids = torch.nonzero(lf_mask, as_tuple=False).reshape(-1)
        lf_count = int(lf_ids.numel())
        if lf_count > 0:
            lf_gaussians = _clone_subset_gaussians(gaussians, lf_ids)

    device = gaussians.get_xyz.device
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    black_background = torch.zeros_like(background)

    dirs = {
        "hf_rgb": output_root / str(args.split) / "hf_rgb",
        "hf_hf_abs": output_root / str(args.split) / "hf_hf_abs",
        "hf_alpha": output_root / str(args.split) / "hf_alpha",
        "gt": output_root / str(args.split) / "gt",
    }
    if args.render_lf:
        dirs["lf_rgb"] = output_root / str(args.split) / "lf_rgb"
        dirs["lf_hf_abs"] = output_root / str(args.split) / "lf_hf_abs"
    if args.render_all:
        dirs["all_rgb"] = output_root / str(args.split) / "all_rgb"
        dirs["all_hf_abs"] = output_root / str(args.split) / "all_hf_abs"

    print(f"[subset-render-v0] model      : {model_dir}")
    print(f"[subset-render-v0] iteration  : {loaded_iter}")
    print(f"[subset-render-v0] payload    : {payload_path}")
    print(f"[subset-render-v0] output     : {output_root}")
    print(f"[subset-render-v0] split      : {args.split} views={len(cameras)}")
    print(f"[subset-render-v0] hf count   : {int(hf_ids.numel())}/{total}")
    if args.render_lf:
        print(f"[subset-render-v0] lf count   : {lf_count}/{total}")
    print(f"[subset-render-v0] selection  : {json.dumps({k: v for k, v in selected.items() if k != 'mask'}, sort_keys=True)}")

    frames = []
    for idx, cam in enumerate(tqdm(cameras, desc="subset render")):
        stem = f"{idx:05d}"
        hf_pkg = render(cam, hf_gaussians, pipe, black_background, kernel_size=float(dataset.kernel_size))
        hf = hf_pkg["render"].clamp(0.0, 1.0)
        hf_abs = _normalize_gray(_laplacian_abs(hf), float(args.hf_percentile))
        _save_tensor(dirs["hf_rgb"] / f"{stem}.png", hf)
        _save_tensor(dirs["hf_hf_abs"] / f"{stem}.png", hf_abs)
        if "alpha" in hf_pkg:
            _save_tensor(dirs["hf_alpha"] / f"{stem}.png", hf_pkg["alpha"].clamp(0.0, 1.0))
        _save_tensor(dirs["gt"] / f"{stem}.png", cam.original_image[:3].to(device=device))

        if lf_gaussians is not None:
            lf_pkg = render(cam, lf_gaussians, pipe, black_background, kernel_size=float(dataset.kernel_size))
            lf = lf_pkg["render"].clamp(0.0, 1.0)
            _save_tensor(dirs["lf_rgb"] / f"{stem}.png", lf)
            _save_tensor(dirs["lf_hf_abs"] / f"{stem}.png", _normalize_gray(_laplacian_abs(lf), float(args.hf_percentile)))

        if args.render_all:
            all_pkg = render(cam, gaussians, pipe, background, kernel_size=float(dataset.kernel_size))
            all_img = all_pkg["render"].clamp(0.0, 1.0)
            _save_tensor(dirs["all_rgb"] / f"{stem}.png", all_img)
            _save_tensor(dirs["all_hf_abs"] / f"{stem}.png", _normalize_gray(_laplacian_abs(all_img), float(args.hf_percentile)))

        frames.append({"index": int(idx), "image_name": str(cam.image_name), "stem": stem})

    manifest = {
        "version": "render_gaussian_subset_from_payload_v0",
        "model_dir": str(model_dir),
        "iteration": int(loaded_iter),
        "scene_root": str(scene_root),
        "images_subdir": str(args.images_subdir),
        "mask_payload": str(payload_path),
        "split": str(args.split),
        "num_views": int(len(cameras)),
        "num_gaussians": int(total),
        "hf_count": int(hf_ids.numel()),
        "lf_count": int(lf_count),
        "selection": {k: v for k, v in selected.items() if k != "mask"},
        "frames": frames,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[subset-render-v0] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
