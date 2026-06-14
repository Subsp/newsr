from __future__ import annotations

import json
import math
import shutil
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import ExtendedSettings
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from gaussian_renderer import render_simple
from scene import Scene
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.depth_utils import depth_to_normal
from utils.camera_utils import loadCam
from utils.prior_injection import index_image_dir, load_mask, load_rgb_image, normalize_image_name
from utils.general_utils import build_rotation
from utils.system_utils import mkdir_p


def copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src = src_model_path / name
        if src.exists():
            shutil.copy2(src, dst_model_path / name)


def load_splat_settings(model_path: Path) -> ExtendedSettings:
    config_path = model_path / "config.json"
    if config_path.exists():
        return ExtendedSettings.from_json(str(config_path))
    return ExtendedSettings()


def build_dataset_args(scene_root: str, model_path: str, images_subdir: str) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=-1,
        white_background=False,
        data_device="cpu",
        eval=True,
        alpha_mask=False,
        init_type="sfm",
    )


def load_cameras_for_split(scene_root: Path, model_path: Path, images_subdir: str, split: str) -> List[object]:
    args = build_dataset_args(str(scene_root), str(model_path), images_subdir)
    if (scene_root / "sparse").exists():
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path,
            args.images,
            args.eval,
            init_type=args.init_type,
        )
    elif (scene_root / "transforms_train.json").exists():
        scene_info = sceneLoadTypeCallbacks["Blender"](
            args.source_path,
            args.white_background,
            args.eval,
        )
    else:
        raise RuntimeError(f"Could not load cameras for scene: {scene_root}")
    split = str(split).lower()
    if split == "train":
        cameras = list(scene_info.train_cameras)
    elif split == "test":
        cameras = list(scene_info.test_cameras)
    elif split == "both":
        cameras = list(scene_info.train_cameras) + list(scene_info.test_cameras)
    else:
        raise ValueError(f"Unsupported camera split: {split}")
    return [loadCam(args, idx, cam_info, 1.0) for idx, cam_info in enumerate(cameras)]


def load_train_cameras_only(scene_root: Path, model_path: Path, images_subdir: str) -> List[object]:
    return load_cameras_for_split(scene_root, model_path, images_subdir, "train")


def resolve_iteration(model_path: Path, iteration: int) -> int:
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


def select_uniform(items: Sequence[object], max_items: int) -> List[object]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    ids = np.unique(np.linspace(0, len(items) - 1, num=max_items, dtype=np.int64))
    return [items[int(idx)] for idx in ids.tolist()]


def _release_fraction(iteration: int, start_iter: int, end_iter: int, mode: str) -> float:
    if int(end_iter) <= int(start_iter):
        return 1.0 if int(iteration) >= int(start_iter) else 0.0
    t = (float(iteration) - float(start_iter)) / max(float(end_iter - start_iter), 1.0)
    t = float(np.clip(t, 0.0, 1.0))
    if str(mode) == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    return t


def scheduled_loss_scale(
    iteration: int,
    *,
    start_iter: int,
    end_iter: int,
    start_scale: float,
    end_scale: float,
    update_scale: float,
    mode: str,
) -> float:
    release = _release_fraction(int(iteration), int(start_iter), int(end_iter), str(mode))
    scheduled = float(start_scale) + release * (float(end_scale) - float(start_scale))
    return float(update_scale) * scheduled


def scheduled_sr_prior_scale(
    iteration: int,
    *,
    start_iter: int,
    end_iter: int,
    start_scale: float,
    end_scale: float,
    update_scale: float,
    mode: str,
) -> float:
    return scheduled_loss_scale(
        iteration,
        start_iter=start_iter,
        end_iter=end_iter,
        start_scale=start_scale,
        end_scale=end_scale,
        update_scale=update_scale,
        mode=mode,
    )


def load_model_ply(model_path: Path, iteration: int, sh_degree: int) -> GaussianModel:
    ply_path = model_path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply_path}")
    model = GaussianModel(sh_degree)
    model.load_ply(str(ply_path))
    tags_path = ply_path.parent / "gaussian_tags.pt"
    if tags_path.exists():
        model.load_tracking_metadata(str(tags_path))
    return model


def configure_trainable_params(
    gaussians: GaussianModel,
    *,
    train_opacity: bool,
    train_scale: bool,
) -> None:
    gaussians._xyz.requires_grad_(True)
    gaussians._opacity.requires_grad_(bool(train_opacity))
    gaussians._scaling.requires_grad_(bool(train_scale))
    for tensor in (
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._rotation,
    ):
        tensor.requires_grad_(False)


def build_optimizer(
    gaussians: GaussianModel,
    *,
    xyz_lr: float,
    opacity_lr: float,
    scale_lr: float,
    train_opacity: bool,
    train_scale: bool,
) -> torch.optim.Optimizer:
    params = [{"params": [gaussians._xyz], "lr": float(xyz_lr), "name": "xyz"}]
    if bool(train_opacity):
        params.append({"params": [gaussians._opacity], "lr": float(opacity_lr), "name": "opacity"})
    if bool(train_scale):
        params.append({"params": [gaussians._scaling], "lr": float(scale_lr), "name": "scale"})
    return torch.optim.Adam(params, eps=1e-15)


def load_gaussian_update_mask_payload(path: str, key: str, total_gaussians: int) -> torch.Tensor | None:
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        mask = payload.reshape(-1)
        if mask.shape[0] == int(total_gaussians):
            return mask.to(device="cuda", dtype=torch.bool)
        if mask.ndim == 1 and mask.dtype in (torch.int16, torch.int32, torch.int64, torch.uint8):
            ids = mask.to(dtype=torch.int64)
            out = torch.zeros((int(total_gaussians),), dtype=torch.bool)
            out[ids] = True
            return out.to(device="cuda")
        raise ValueError(
            f"Raw tensor Gaussian mask payload has unsupported shape: {tuple(mask.shape)} "
            f"vs total_gaussians={total_gaussians}"
        )
    if key in payload:
        mask = payload[key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.reshape(-1)
        if mask.shape[0] != int(total_gaussians):
            raise ValueError(
                f"Gaussian update mask '{key}' length mismatch: "
                f"{tuple(mask.shape)} vs total_gaussians={total_gaussians}"
            )
        return mask.to(device="cuda", dtype=torch.bool)
    if "selected_ids" in payload:
        ids = payload["selected_ids"]
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        ids = ids.to(dtype=torch.int64)
        out = torch.zeros((int(total_gaussians),), dtype=torch.bool)
        out[ids] = True
        return out.to(device="cuda")
    raise KeyError(f"Mask payload must contain '{key}' or 'selected_ids': {path}")


def apply_gaussian_update_mask(
    optimizer: torch.optim.Optimizer,
    *,
    total_gaussians: int,
    update_mask: torch.Tensor | None,
    update_scale: float,
    scale_param_mask: torch.Tensor | None = None,
) -> None:
    if update_mask is None and float(update_scale) == 1.0 and scale_param_mask is None:
        return
    inverse_mask = None if update_mask is None else ~update_mask.to(device="cuda", dtype=torch.bool)
    for group in optimizer.param_groups:
        scale_group_mask = None
        if scale_param_mask is not None and str(group.get("name", "")) == "scale":
            scale_group_mask = scale_param_mask.to(device="cuda", dtype=torch.bool)
        for param in group["params"]:
            param_mask = None
            if (
                scale_group_mask is not None
                and tuple(param.shape) == tuple(scale_group_mask.shape)
            ):
                param_mask = scale_group_mask
            if param.grad is not None and param.grad.ndim > 0 and param.grad.shape[0] == int(total_gaussians):
                if float(update_scale) != 1.0:
                    param.grad.mul_(float(update_scale))
                if inverse_mask is not None:
                    param.grad[inverse_mask] = 0
                if param_mask is not None:
                    param.grad[~param_mask] = 0
            state = optimizer.state.get(param, None)
            if state is None:
                continue
            for value in state.values():
                if not torch.is_tensor(value):
                    continue
                if value.ndim > 0 and value.shape[0] == int(total_gaussians):
                    if inverse_mask is not None:
                        value[inverse_mask] = 0
                    if param_mask is not None and tuple(value.shape) == tuple(param_mask.shape):
                        value[~param_mask] = 0


def build_scale_update_mask(
    scale_init: torch.Tensor,
    *,
    update_mask: torch.Tensor | None,
    axis_mode: str,
) -> torch.Tensor | None:
    mode = str(axis_mode).strip().lower()
    if mode == "all":
        return None
    if mode != "major_only":
        raise ValueError(f"Unsupported gaussian_scale_axis_mode={axis_mode!r}; use 'all' or 'major_only'.")
    if scale_init.ndim != 2:
        raise ValueError(f"Expected scale_init to have shape [N, C], got {tuple(scale_init.shape)}")
    base_mask = (
        torch.ones((scale_init.shape[0], 1), dtype=torch.bool, device=scale_init.device)
        if update_mask is None
        else update_mask.to(device=scale_init.device, dtype=torch.bool).reshape(-1, 1)
    )
    major_axis = torch.argmax(scale_init, dim=1, keepdim=True)
    axis_mask = torch.zeros_like(scale_init, dtype=torch.bool)
    axis_mask.scatter_(1, major_axis, True)
    return axis_mask & base_mask


def clamp_scale_update_range(
    gaussians: GaussianModel,
    *,
    scale_raw_init: torch.Tensor,
    scale_param_mask: torch.Tensor | None,
    min_multiplier: float,
    max_multiplier: float,
) -> None:
    min_mult = float(min_multiplier)
    max_mult = float(max_multiplier)
    if scale_param_mask is None and min_mult <= 0.0 and max_mult <= 0.0:
        return
    if min_mult < 0.0 or max_mult < 0.0:
        raise ValueError("Scale multipliers must be non-negative.")
    if min_mult > 0.0 and max_mult > 0.0 and min_mult > max_mult:
        raise ValueError(
            f"gaussian_scale_min_multiplier={min_mult} cannot exceed "
            f"gaussian_scale_max_multiplier={max_mult}."
        )
    with torch.no_grad():
        target = gaussians._scaling
        mask = (
            torch.ones_like(target, dtype=torch.bool)
            if scale_param_mask is None
            else scale_param_mask.to(device=target.device, dtype=torch.bool)
        )
        if min_mult > 0.0:
            min_raw = scale_raw_init + math.log(min_mult)
            target[mask] = torch.maximum(target[mask], min_raw[mask])
        if max_mult > 0.0:
            max_raw = scale_raw_init + math.log(max_mult)
            target[mask] = torch.minimum(target[mask], max_raw[mask])


def normalize_normal(normal: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return normal / torch.clamp(torch.linalg.norm(normal, dim=0, keepdim=True), min=eps)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, min_pixels: float) -> torch.Tensor | None:
    mask = mask.to(device=value.device, dtype=value.dtype)
    denom = mask.sum()
    if float(denom.detach().item()) < float(min_pixels):
        return None
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(0)
    return (value * mask).sum() / torch.clamp(denom * (value.numel() / mask.numel()), min=1.0)


def charbonnier(value: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(value * value + float(eps) ** 2)


def laplacian_highfreq(image: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(image.shape[0], 1, 1, 1)
    return F.conv2d(image[None], kernel, padding=1, groups=image.shape[0])[0]


def masked_weighted_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    min_pixels: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    effective = mask.to(device=value.device, dtype=value.dtype)
    if weight is not None:
        effective = effective * weight.to(device=value.device, dtype=value.dtype).clamp_min(0.0)
    denom = effective.sum()
    if float(denom.detach().item()) < float(min_pixels):
        return None
    while effective.ndim < value.ndim:
        effective = effective.unsqueeze(0)
    channels = value.numel() / max(effective.numel(), 1)
    return (value * effective).sum() / torch.clamp(denom * channels, min=1.0)


def odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def blur_chw(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = odd_kernel_size(kernel_size)
    if kernel_size <= 1:
        return image
    pad = kernel_size // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0]


def blur_hw(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = odd_kernel_size(kernel_size)
    if kernel_size <= 1:
        return value
    pad = kernel_size // 2
    padded = F.pad(value.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0, 0]


def resize_chw(image: torch.Tensor, target_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if tuple(image.shape[-2:]) == tuple(target_hw):
        return image
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = F.interpolate(image.unsqueeze(0).float(), size=target_hw, mode=mode, align_corners=align_corners)
    return resized[0]


def resize_hw(value: torch.Tensor, target_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(target_hw):
        return value
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = F.interpolate(value[None, None].float(), size=target_hw, mode=mode, align_corners=align_corners)
    return resized[0, 0]


def lookup_indexed_path(index: Optional[Dict[str, Path]], image_name: str) -> Path | None:
    if index is None:
        return None
    candidates = [
        str(image_name),
        normalize_image_name(str(image_name)),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    lower_index = {str(key).lower(): value for key, value in index.items()}
    for key in candidates:
        path = index.get(str(key))
        if path is not None:
            return Path(path)
    for key in candidates:
        path = lower_index.get(str(key).lower())
        if path is not None:
            return Path(path)
    return None


def parse_subdir_list(value: str | None, default_auto: Sequence[str]) -> List[str]:
    if value is None:
        return list(default_auto)
    value = str(value).strip()
    if not value:
        return []
    if value.lower() == "auto":
        return list(default_auto)
    return [item.strip() for item in value.split(",") if item.strip()]


def unique_names_for_view(image_name: str) -> List[str]:
    names = [
        str(image_name),
        normalize_image_name(str(image_name)),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    seen = set()
    unique = []
    for name in names:
        if name and name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def find_array_path(root: Path, image_name: str, subdirs: Sequence[str]) -> Path | None:
    names = unique_names_for_view(image_name)
    extensions = (".npy", ".npz", ".pt", ".pth")
    candidates: List[Path] = []
    for subdir in subdirs:
        base = root / subdir if subdir else root
        for name in names:
            for ext in extensions:
                candidates.append(base / f"{name}{ext}")
    for name in names:
        for ext in extensions:
            candidates.append(root / f"{name}{ext}")
    for train_depth_dir in sorted((root / "train").glob("ours_*/depth")):
        for name in names:
            for ext in extensions:
                candidates.append(train_depth_dir / f"{name}{ext}")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_array(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        payload = np.load(str(path))
        for key in preferred_keys:
            if key in payload:
                return np.asarray(payload[key])
        if "arr_0" in payload:
            return np.asarray(payload["arr_0"])
        keys = list(payload.keys())
        if not keys:
            raise ValueError(f"Empty npz prior file: {path}")
        return np.asarray(payload[keys[0]])
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, dict):
            for key in preferred_keys:
                if key in payload:
                    payload = payload[key]
                    break
            else:
                first_key = next(iter(payload.keys()))
                payload = payload[first_key]
        if torch.is_tensor(payload):
            return payload.detach().cpu().numpy()
        return np.asarray(payload)
    return np.load(str(path))


def array_to_hw_tensor(array: np.ndarray, target_hw: Tuple[int, int], mode: str) -> torch.Tensor:
    value = np.asarray(array, dtype=np.float32)
    value = np.squeeze(value)
    if value.ndim == 3:
        if value.shape[0] in {1, 3} and value.shape[1] == target_hw[0]:
            value = value[0] if value.shape[0] == 1 else value.mean(axis=0)
        elif value.shape[-1] in {1, 3}:
            value = value[..., 0] if value.shape[-1] == 1 else value.mean(axis=-1)
        else:
            raise ValueError(f"Cannot canonicalize prior array shape {value.shape} to HxW")
    if value.ndim != 2:
        raise ValueError(f"Expected HxW prior array, got shape {value.shape}")
    tensor = torch.from_numpy(value.astype(np.float32, copy=False))
    return resize_hw(tensor, target_hw, mode=mode)


def load_depth_prior_for_view(
    image_name: str,
    *,
    depth_prior_root: Path | None,
    target_hw: Tuple[int, int],
    depth_subdirs: Sequence[str],
    confidence_subdirs: Sequence[str],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, Dict[str, object]]:
    if depth_prior_root is None:
        return None, None, {"status": "disabled"}
    depth_path = find_array_path(depth_prior_root, image_name, depth_subdirs)
    if depth_path is None:
        return None, None, {"status": "missing_depth", "image_name": str(image_name)}
    depth = array_to_hw_tensor(load_array(depth_path, ("depth", "depth_hr", "pred", "arr_0")), target_hw, mode="bilinear")

    confidence = None
    confidence_path = None
    if confidence_subdirs:
        confidence_path = find_array_path(depth_prior_root, image_name, confidence_subdirs)
        if confidence_path is not None:
            confidence = array_to_hw_tensor(
                load_array(confidence_path, ("confidence", "conf", "conf_hr", "valid", "mask", "arr_0")),
                target_hw,
                mode="nearest" if confidence_path.parent.name.lower() in {"valid", "mask"} else "bilinear",
            )
            confidence = confidence.float()
            if float(confidence.detach().max().item()) > 1.0:
                confidence = confidence / torch.clamp(confidence.detach().quantile(0.95), min=1e-6)
            confidence = confidence.clamp(0.0, 1.0)
    return depth, confidence, {
        "status": "loaded",
        "depth_path": str(depth_path),
        "confidence_path": str(confidence_path) if confidence_path is not None else None,
    }


def robust_align_depth_to_reference(
    prior_depth: torch.Tensor,
    reference_depth: torch.Tensor,
    mask: torch.Tensor,
    min_pixels: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid = mask & torch.isfinite(prior_depth) & torch.isfinite(reference_depth) & (prior_depth > 1e-6) & (reference_depth > 1e-6)
    pixels = int(valid.detach().sum().item())
    if pixels < int(min_pixels):
        return prior_depth, {"mode": "identity_insufficient_pixels", "pixels": pixels, "scale": 1.0, "shift": 0.0}

    prior_values = prior_depth[valid].detach().float().cpu().numpy()
    ref_values = reference_depth[valid].detach().float().cpu().numpy()
    p10, p50, p90 = np.percentile(prior_values, [10, 50, 90]).astype(np.float32)
    r10, r50, r90 = np.percentile(ref_values, [10, 50, 90]).astype(np.float32)
    prior_span = float(max(float(p90 - p10), 1e-6))
    ref_span = float(max(float(r90 - r10), 1e-6))
    scale = ref_span / prior_span
    shift = float(r50 - scale * float(p50))
    aligned = prior_depth * float(scale) + float(shift)
    return aligned, {
        "mode": "robust_p10_p50_p90",
        "pixels": pixels,
        "scale": float(scale),
        "shift": float(shift),
        "prior_p10": float(p10),
        "prior_p50": float(p50),
        "prior_p90": float(p90),
        "ref_p10": float(r10),
        "ref_p50": float(r50),
        "ref_p90": float(r90),
    }


def load_sr_prior_for_view(
    image_name: str,
    *,
    sr_prior_index: Optional[Dict[str, Path]],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor | None, str | None]:
    path = lookup_indexed_path(sr_prior_index, image_name)
    if path is None:
        return None, None
    image = load_rgb_image(path).permute(2, 0, 1).contiguous()
    image = resize_chw(image, target_hw, mode="bilinear")
    return image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0), str(path)


def load_sr_mask_for_view(
    image_name: str,
    *,
    mask_dir: Path | None,
    mask_suffix: str,
    target_hw: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if mask_dir is None:
        return None
    names = unique_names_for_view(image_name)
    candidates = []
    if mask_suffix:
        candidates.extend(mask_dir / f"{name}{mask_suffix}" for name in names)
    candidates.extend(mask_dir / f"{name}.png" for name in names)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    mask = load_mask(path)
    mask = resize_hw(mask, target_hw, mode="bilinear")
    return mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


@torch.no_grad()
def build_prepared_sr_cache(
    cameras: Sequence[object],
    student_init: GaussianModel,
    background: torch.Tensor,
    *,
    sr_prior_index: Optional[Dict[str, Path]],
    sr_anchor_index: Optional[Dict[str, Path]],
    sr_prior_mask_dir: Path | None,
    sr_prior_mask_suffix: str,
    prior_consistency_threshold: float,
    prior_mask_floor: float,
) -> Tuple[List[object], List[Dict[str, torch.Tensor | str | float | None]]]:
    cache: List[Dict[str, torch.Tensor | str | float | None]] = []
    cached_cameras: List[object] = []
    if sr_prior_index is None:
        return cached_cameras, cache

    for idx, camera in enumerate(cameras, start=1):
        target_hw = (int(camera.image_height), int(camera.image_width))
        prior_rgb, prior_path = load_sr_prior_for_view(
            camera.image_name,
            sr_prior_index=sr_prior_index,
            target_hw=target_hw,
            device=torch.device("cpu"),
        )
        if prior_rgb is None:
            print(f"[mip-to-sof-surface] SR cache skip {idx}/{len(cameras)} missing_prior view={camera.image_name}", flush=True)
            continue

        anchor_rgb, anchor_path = load_sr_prior_for_view(
            camera.image_name,
            sr_prior_index=sr_anchor_index,
            target_hw=target_hw,
            device=torch.device("cpu"),
        )
        if anchor_rgb is None:
            print(f"[mip-to-sof-surface] SR cache skip {idx}/{len(cameras)} missing_anchor view={camera.image_name}", flush=True)
            continue

        structure_mask = load_sr_mask_for_view(
            camera.image_name,
            mask_dir=sr_prior_mask_dir,
            mask_suffix=sr_prior_mask_suffix,
            target_hw=target_hw,
            device=torch.device("cpu"),
        )
        if structure_mask is None:
            structure_mask = torch.ones(target_hw, dtype=torch.float32, device="cpu")
        else:
            structure_mask = structure_mask.clamp(0.0, 1.0)
        if float(prior_mask_floor) > 0.0:
            structure_mask = structure_mask.clamp(min=float(prior_mask_floor), max=1.0)

        prior_consistency = torch.abs(prior_rgb - anchor_rgb).mean(dim=0)
        if float(prior_consistency_threshold) > 0.0:
            consistency_mask = (prior_consistency <= float(prior_consistency_threshold)).to(dtype=torch.float32)
        else:
            consistency_mask = torch.ones_like(prior_consistency)
        prior_mask = (consistency_mask * structure_mask).clamp(0.0, 1.0)

        base_pkg = render_simple(camera, student_init, background)
        base_rgb = base_pkg["render"].detach().clamp(0.0, 1.0).cpu()
        valid_ratio = float(consistency_mask.mean().item())
        mask_mean = float(prior_mask.mean().item())
        cache.append(
            {
                "prior_rgb": prior_rgb.detach().cpu(),
                "anchor_rgb": anchor_rgb.detach().cpu(),
                "base_rgb": base_rgb.detach().cpu(),
                "prior_mask": prior_mask.detach().cpu(),
                "consistency": prior_consistency.detach().cpu(),
                "valid_ratio": valid_ratio,
                "structure_mask_mean": float(structure_mask.mean().item()),
                "mask_mean": mask_mean,
                "prior_path": prior_path,
                "anchor_path": anchor_path,
            }
        )
        cached_cameras.append(camera)
        print(
            f"[mip-to-sof-surface] cached SR prior view {idx}/{len(cameras)} "
            f"valid={valid_ratio:.4f} mask={mask_mean:.4f}",
            flush=True,
        )
    return cached_cameras, cache


def compute_prepared_sr_losses(
    rgb: torch.Tensor,
    target: Dict[str, torch.Tensor | str | float | None],
    *,
    min_pixels: float,
    min_valid_ratio: float,
    prior_delta_clip: float,
    disable_hf_residual: bool,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    valid_ratio = float(target.get("valid_ratio", 0.0) or 0.0)
    if valid_ratio < float(min_valid_ratio):
        return None, None

    prior_rgb = target["prior_rgb"]
    anchor_rgb = target["anchor_rgb"]
    prior_mask = target["prior_mask"]
    if not torch.is_tensor(prior_rgb) or not torch.is_tensor(anchor_rgb) or not torch.is_tensor(prior_mask):
        return None, None

    prior_rgb = prior_rgb.to(device=rgb.device, dtype=rgb.dtype)
    anchor_rgb = anchor_rgb.to(device=rgb.device, dtype=rgb.dtype)
    prior_mask = prior_mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
    active = prior_mask.sum()
    if float(active.detach().item()) < float(min_pixels):
        return None, None

    mask_chw = prior_mask.unsqueeze(0)
    denom = torch.clamp(active * rgb.shape[0], min=1.0)
    loss_l1 = (torch.abs(rgb - prior_rgb) * mask_chw).sum() / denom

    if bool(disable_hf_residual):
        rgb_hf = laplacian_highfreq(rgb)
        prior_hf = laplacian_highfreq(prior_rgb)
        loss_hf = (torch.abs(rgb_hf - prior_hf) * mask_chw).sum() / denom
    else:
        prior_delta = prior_rgb - anchor_rgb
        if float(prior_delta_clip) > 0.0:
            clip_v = float(prior_delta_clip)
            prior_delta = prior_delta.clamp(min=-clip_v, max=clip_v)
        render_residual_hf = laplacian_highfreq(rgb - anchor_rgb)
        prior_residual_hf = laplacian_highfreq(prior_delta)
        loss_hf = (torch.abs(render_residual_hf - prior_residual_hf) * mask_chw).sum() / denom
    return loss_l1, loss_hf


def premul_luma(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return torch.sum(rgb * alpha * weights, dim=0)


def compute_premul_hf_excess_loss(
    render_pkg: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor | str | float | None],
    *,
    kernel_size: int,
    excess_ratio: float,
    margin: float,
    min_pixels: float,
    min_valid_ratio: float,
    prior_delta_clip: float,
) -> torch.Tensor | None:
    valid_ratio = float(target.get("valid_ratio", 0.0) or 0.0)
    if valid_ratio < float(min_valid_ratio):
        return None

    prior_rgb = target["prior_rgb"]
    anchor_rgb = target["anchor_rgb"]
    prior_mask = target["prior_mask"]
    if not torch.is_tensor(prior_rgb) or not torch.is_tensor(anchor_rgb) or not torch.is_tensor(prior_mask):
        return None

    rgb = render_pkg["render"].clamp(0.0, 1.0)
    alpha = render_pkg["alpha"].clamp(0.0, 1.0)
    prior_rgb = prior_rgb.to(device=rgb.device, dtype=rgb.dtype)
    anchor_rgb = anchor_rgb.to(device=rgb.device, dtype=rgb.dtype)
    prior_mask = prior_mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
    active = prior_mask.sum()
    if float(active.detach().item()) < float(min_pixels):
        return None

    prior_delta = prior_rgb - anchor_rgb
    if float(prior_delta_clip) > 0.0:
        clip_v = float(prior_delta_clip)
        prior_delta = prior_delta.clamp(min=-clip_v, max=clip_v)
    target_rgb = (anchor_rgb + prior_delta).clamp(0.0, 1.0)

    alpha_for_target = alpha.detach()
    student_luma = premul_luma(rgb, alpha)
    target_luma = premul_luma(target_rgb, alpha_for_target)
    student_hf = student_luma - blur_hw(student_luma, kernel_size)
    target_hf = target_luma - blur_hw(target_luma, kernel_size)
    excess = torch.relu(torch.abs(student_hf) - float(excess_ratio) * torch.abs(target_hf) - float(margin))
    return masked_mean(excess, prior_mask, min_pixels)


def compute_depth_prior_distortion_loss(
    render_pkg: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor | str | float | None],
    *,
    min_pixels: float,
) -> torch.Tensor | None:
    depth_mask = target.get("depth_prior_mask")
    depth_weight = target.get("depth_prior_weight")
    if not torch.is_tensor(depth_mask):
        return None
    distortion = render_pkg["distortion"].clamp_min(0.0)
    return masked_weighted_mean(distortion, depth_mask, min_pixels, depth_weight if torch.is_tensor(depth_weight) else None)


def compute_depth_prior_self_normal_loss(
    camera: object,
    render_pkg: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor | str | float | None],
    *,
    min_pixels: float,
) -> torch.Tensor | None:
    depth_mask = target.get("depth_prior_mask")
    depth_weight = target.get("depth_prior_weight")
    if not torch.is_tensor(depth_mask):
        return None

    render_normal = normalize_normal(render_pkg["normal"])
    render_depth = render_pkg["depth"]
    depth_hw = torch.nan_to_num(render_depth[0], nan=0.0, posinf=0.0, neginf=0.0)
    depth_normal_hw3, _ = depth_to_normal(camera, depth_hw.unsqueeze(0))
    depth_normal = normalize_normal(depth_normal_hw3.permute(2, 0, 1))
    depth_normal_valid = torch.linalg.norm(depth_normal, dim=0) > 1e-6
    valid_mask = (
        depth_mask.to(device=render_depth.device, dtype=torch.bool)
        & torch.isfinite(render_depth[0])
        & (render_depth[0] > 1e-6)
        & depth_normal_valid
    )
    dot = torch.sum(render_normal * depth_normal, dim=0).clamp(-1.0, 1.0)
    return masked_weighted_mean(1.0 - dot, valid_mask, min_pixels, depth_weight if torch.is_tensor(depth_weight) else None)


@torch.no_grad()
def render_teacher_cache(
    cameras: Sequence[object],
    teacher: GaussianModel,
    student_init: GaussianModel,
    background: torch.Tensor,
    *,
    mip_splat_args: ExtendedSettings,
    min_surface_alpha: float,
    depth_prior_root: Path | None = None,
    depth_prior_subdirs: Sequence[str] = ("depth", ""),
    depth_prior_confidence_subdirs: Sequence[str] = ("confidence", "conf", "depth_conf", "valid"),
    depth_prior_confidence_min: float = 0.05,
    depth_prior_agreement_threshold: float = 0.15,
    depth_prior_agreement_floor: float = 0.0,
    depth_prior_align_mode: str = "affine_robust",
    depth_prior_align_min_pixels: int = 2048,
    depth_prior_surface_weight_boost: float = 0.0,
    depth_prior_weight_gain: float = 1.0,
    depth_prior_weight_power: float = 1.0,
    depth_prior_weight_min: float = 0.0,
) -> List[Dict[str, torch.Tensor]]:
    cache = []
    for idx, camera in enumerate(cameras, start=1):
        teacher_pkg = render_simple(camera, teacher, background)
        base_pkg = render_simple(camera, student_init, background)
        mip_ref_pkg = render_simple(camera, student_init, background, splat_args=mip_splat_args)
        target_hw = (int(camera.image_height), int(camera.image_width))
        target_alpha = teacher_pkg["alpha"].detach().clamp(0.0, 1.0)
        target_depth = teacher_pkg["depth"].detach()
        target_normal = normalize_normal(teacher_pkg["normal"].detach())
        base_alpha = base_pkg["alpha"].detach().clamp(0.0, 1.0)
        base_rgb = base_pkg["render"].detach().clamp(0.0, 1.0)
        base_depth = base_pkg["depth"].detach()
        mip_ref_alpha = mip_ref_pkg["alpha"].detach().clamp(0.0, 1.0)
        mip_ref_rgb = mip_ref_pkg["render"].detach().clamp(0.0, 1.0)
        mip_ref_depth = mip_ref_pkg["depth"].detach()
        surface_mask = (target_alpha[0] >= float(min_surface_alpha)) & torch.isfinite(target_depth[0]) & (target_depth[0] > 1e-6)
        surface_weight = torch.ones_like(surface_mask, dtype=torch.float32, device=surface_mask.device)

        depth_prior, depth_prior_conf, depth_prior_info = load_depth_prior_for_view(
            camera.image_name,
            depth_prior_root=depth_prior_root,
            target_hw=target_hw,
            depth_subdirs=depth_prior_subdirs,
            confidence_subdirs=depth_prior_confidence_subdirs,
        )
        depth_prior_mask = torch.zeros_like(surface_mask, dtype=torch.bool)
        depth_prior_weight = torch.zeros_like(surface_weight)
        depth_prior_target = None
        depth_prior_normal = None
        if depth_prior is not None:
            depth_prior = depth_prior.to(device=background.device, dtype=torch.float32)
            if depth_prior_conf is None:
                depth_prior_conf = torch.ones_like(depth_prior)
            else:
                depth_prior_conf = depth_prior_conf.to(device=background.device, dtype=torch.float32).clamp(0.0, 1.0)
            align_seed_mask = surface_mask & (depth_prior_conf >= float(depth_prior_confidence_min))
            if depth_prior_align_mode == "affine_robust":
                aligned_depth, align_summary = robust_align_depth_to_reference(
                    depth_prior,
                    target_depth[0],
                    align_seed_mask,
                    min_pixels=int(depth_prior_align_min_pixels),
                )
            elif depth_prior_align_mode == "identity":
                aligned_depth = depth_prior
                align_summary = {"mode": "identity", "pixels": int(align_seed_mask.detach().sum().item()), "scale": 1.0, "shift": 0.0}
            else:
                raise ValueError(f"Unsupported depth prior align mode: {depth_prior_align_mode}")
            depth_prior_info["align"] = align_summary
            depth_valid = torch.isfinite(aligned_depth) & (aligned_depth > 1e-6) & (depth_prior_conf >= float(depth_prior_confidence_min))
            if float(depth_prior_agreement_threshold) > 0.0:
                agreement = torch.abs(aligned_depth - target_depth[0]) / torch.clamp(target_depth[0], min=1e-6)
                agreement_conf = torch.clamp(1.0 - agreement / float(depth_prior_agreement_threshold), min=0.0, max=1.0)
            else:
                agreement_conf = torch.ones_like(aligned_depth)
            if float(depth_prior_agreement_floor) > 0.0:
                agreement_conf = torch.clamp(agreement_conf, min=float(depth_prior_agreement_floor), max=1.0)
            depth_prior_weight = depth_prior_conf * agreement_conf
            if float(depth_prior_weight_power) > 0.0 and float(depth_prior_weight_power) != 1.0:
                depth_prior_weight = depth_prior_weight.clamp_min(0.0).pow(float(depth_prior_weight_power))
            if float(depth_prior_weight_gain) != 1.0:
                depth_prior_weight = depth_prior_weight * float(depth_prior_weight_gain)
            depth_prior_weight = depth_prior_weight.clamp(0.0, 1.0)
            if float(depth_prior_weight_min) > 0.0:
                depth_prior_weight = torch.where(
                    depth_prior_weight >= float(depth_prior_weight_min),
                    depth_prior_weight,
                    torch.zeros_like(depth_prior_weight),
                )
            depth_prior_weight = torch.where(surface_mask & depth_valid, depth_prior_weight, torch.zeros_like(depth_prior_weight))
            depth_prior_mask = depth_prior_weight > 0.0
            depth_prior_target = aligned_depth.unsqueeze(0).detach()
            if bool(depth_prior_mask.any()):
                normal_hw3, _ = depth_to_normal(camera, aligned_depth.unsqueeze(0))
                depth_prior_normal = normalize_normal(normal_hw3.permute(2, 0, 1).detach())
            if float(depth_prior_surface_weight_boost) > 0.0:
                surface_weight = surface_weight + float(depth_prior_surface_weight_boost) * depth_prior_weight

        surface_weight = surface_weight.clamp(0.0, 1.0 + max(float(depth_prior_surface_weight_boost), 0.0))
        cache.append(
            {
                "target_alpha": target_alpha,
                "target_depth": target_depth,
                "target_normal": target_normal,
                "base_alpha": base_alpha,
                "base_rgb": base_rgb,
                "base_depth": base_depth,
                "mip_ref_alpha": mip_ref_alpha,
                "mip_ref_depth": mip_ref_depth,
                "mip_ref_premul": (mip_ref_rgb * mip_ref_alpha).detach(),
                "surface_mask": surface_mask,
                "surface_weight": surface_weight.detach(),
                "depth_prior_target": depth_prior_target,
                "depth_prior_normal": depth_prior_normal,
                "depth_prior_mask": depth_prior_mask.detach(),
                "depth_prior_weight": depth_prior_weight.detach(),
                "depth_prior_info": depth_prior_info,
            }
        )
        depth_ratio = float(depth_prior_mask.float().mean().item())
        print(
            f"[mip-to-sof-surface] cached teacher view {idx}/{len(cameras)} "
            f"mask={float(surface_mask.float().mean().item()):.4f} "
            f"depth_prior={depth_ratio:.4f}",
            flush=True,
        )
    return cache


@torch.no_grad()
def render_mip_closure_cache(
    cameras: Sequence[object],
    student_init: GaussianModel,
    background: torch.Tensor,
    *,
    mip_splat_args: ExtendedSettings,
) -> List[Dict[str, torch.Tensor | str]]:
    cache: List[Dict[str, torch.Tensor | str]] = []
    for idx, camera in enumerate(cameras, start=1):
        mip_ref_pkg = render_simple(camera, student_init, background, splat_args=mip_splat_args)
        mip_ref_alpha = mip_ref_pkg["alpha"].detach().clamp(0.0, 1.0).cpu()
        mip_ref_rgb = mip_ref_pkg["render"].detach().clamp(0.0, 1.0).cpu()
        cache.append(
            {
                "image_name": str(camera.image_name),
                "mip_ref_alpha": mip_ref_alpha,
                "mip_ref_depth": mip_ref_pkg["depth"].detach().cpu(),
                "mip_ref_premul": (mip_ref_rgb * mip_ref_alpha).detach().cpu(),
            }
        )
        print(
            f"[mip-to-sof-surface] cached obs closure view {idx}/{len(cameras)} "
            f"name={camera.image_name}",
            flush=True,
        )
    return cache


@torch.no_grad()
def clamp_xyz_displacement(gaussians: GaussianModel, xyz_init: torch.Tensor, max_displacement: float) -> None:
    if float(max_displacement) <= 0.0:
        return
    delta = gaussians._xyz.data - xyz_init
    norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(float(max_displacement) / torch.clamp(norm, min=1e-12), max=1.0)
    gaussians._xyz.data.copy_(xyz_init + delta * scale)


def grad_l2_norm(value: torch.Tensor) -> float:
    grad = value.grad
    if grad is None:
        return 0.0
    return float(torch.linalg.norm(grad.detach()).item())


def per_gaussian_grad_norm(value: torch.Tensor) -> torch.Tensor:
    grad = value.grad
    total_count = int(value.shape[0])
    device = value.device
    if grad is None:
        return torch.zeros((total_count,), dtype=torch.float32, device=device)
    grad = grad.detach()
    if grad.ndim <= 1:
        grad = grad.reshape(total_count, 1)
    else:
        grad = grad.reshape(total_count, -1)
    return torch.linalg.norm(grad, dim=1)


def zero_optimizer_entries(
    optimizer: torch.optim.Optimizer,
    entry_mask: torch.Tensor,
) -> None:
    entry_mask = entry_mask.reshape(-1).to(dtype=torch.bool)
    total_count = int(entry_mask.shape[0])
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and param.grad.ndim > 0 and int(param.grad.shape[0]) == total_count:
                param.grad[entry_mask] = 0
            stored_state = optimizer.state.get(param, None)
            if stored_state is None:
                continue
            for value in stored_state.values():
                if not torch.is_tensor(value):
                    continue
                if value.ndim > 0 and int(value.shape[0]) == total_count:
                    value[entry_mask] = 0


def restore_gaussian_entries(
    gaussians: GaussianModel,
    *,
    reset_mask: torch.Tensor,
    xyz_init: torch.Tensor,
    opacity_raw_init: torch.Tensor,
    scale_raw_init: torch.Tensor,
    train_opacity: bool,
    train_scale: bool,
) -> None:
    reset_mask = reset_mask.reshape(-1).to(device=gaussians._xyz.device, dtype=torch.bool)
    with torch.no_grad():
        gaussians._xyz[reset_mask] = xyz_init[reset_mask]
        if bool(train_opacity):
            gaussians._opacity[reset_mask] = opacity_raw_init[reset_mask]
        if bool(train_scale):
            gaussians._scaling[reset_mask] = scale_raw_init[reset_mask]


def build_gradient_guard_mask(
    gaussians: GaussianModel,
    *,
    train_opacity: bool,
    train_scale: bool,
    update_mask: torch.Tensor | None,
    only_update_mask: bool,
    xyz_threshold: float,
    opacity_threshold: float,
    scale_threshold: float,
    max_fraction: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    total_count = int(gaussians.get_xyz.shape[0])
    device = gaussians._xyz.device
    xyz_norm = per_gaussian_grad_norm(gaussians._xyz)
    opacity_norm = per_gaussian_grad_norm(gaussians._opacity) if bool(train_opacity) else torch.zeros_like(xyz_norm)
    scale_norm = per_gaussian_grad_norm(gaussians._scaling) if bool(train_scale) else torch.zeros_like(xyz_norm)
    severity = torch.zeros((total_count,), dtype=torch.float32, device=device)
    flagged = torch.zeros((total_count,), dtype=torch.bool, device=device)
    nonfinite = ~torch.isfinite(xyz_norm)
    if bool(train_opacity):
        nonfinite |= ~torch.isfinite(opacity_norm)
    if bool(train_scale):
        nonfinite |= ~torch.isfinite(scale_norm)
    flagged |= nonfinite
    if float(xyz_threshold) > 0.0:
        xyz_score = xyz_norm / float(xyz_threshold)
        severity = torch.maximum(severity, torch.nan_to_num(xyz_score, nan=0.0, posinf=1e6, neginf=0.0))
        flagged |= xyz_norm > float(xyz_threshold)
    if bool(train_opacity) and float(opacity_threshold) > 0.0:
        opacity_score = opacity_norm / float(opacity_threshold)
        severity = torch.maximum(severity, torch.nan_to_num(opacity_score, nan=0.0, posinf=1e6, neginf=0.0))
        flagged |= opacity_norm > float(opacity_threshold)
    if bool(train_scale) and float(scale_threshold) > 0.0:
        scale_score = scale_norm / float(scale_threshold)
        severity = torch.maximum(severity, torch.nan_to_num(scale_score, nan=0.0, posinf=1e6, neginf=0.0))
        flagged |= scale_norm > float(scale_threshold)
    severity = torch.where(nonfinite, torch.full_like(severity, 1e6), severity)
    if bool(only_update_mask) and update_mask is not None:
        flagged &= update_mask.to(device=device, dtype=torch.bool)
    flagged_count = int(flagged.sum().item())
    if flagged_count > 0 and (float(max_fraction) > 0.0 or int(max_points) > 0):
        limit = flagged_count
        if float(max_fraction) > 0.0:
            limit = min(limit, max(1, int(math.ceil(float(total_count) * float(max_fraction)))))
        if int(max_points) > 0:
            limit = min(limit, int(max_points))
        if limit < flagged_count:
            candidate_ids = torch.nonzero(flagged, as_tuple=False).reshape(-1)
            topk = torch.topk(severity[candidate_ids], k=int(limit), largest=True).indices
            limited = torch.zeros_like(flagged)
            limited[candidate_ids[topk]] = True
            flagged = limited
            flagged_count = int(flagged.sum().item())
    stats = {
        "xyz_max": float(torch.nan_to_num(xyz_norm, nan=0.0, posinf=0.0, neginf=0.0).max().item()) if total_count > 0 else 0.0,
        "opacity_max": float(torch.nan_to_num(opacity_norm, nan=0.0, posinf=0.0, neginf=0.0).max().item()) if total_count > 0 else 0.0,
        "scale_max": float(torch.nan_to_num(scale_norm, nan=0.0, posinf=0.0, neginf=0.0).max().item()) if total_count > 0 else 0.0,
        "nonfinite": float(int(nonfinite.sum().item())),
        "flagged": float(flagged_count),
    }
    return flagged, severity, stats


def build_depth_feedback_value(
    *,
    depth_prior_scale: float,
    lambda_depth_prior: float,
    lambda_depth_prior_normal: float,
    lambda_depth_prior_distortion: float,
    lambda_depth_prior_self_normal: float,
    depth_prior_loss: torch.Tensor,
    depth_prior_normal_loss: torch.Tensor,
    depth_prior_distortion_loss: torch.Tensor,
    depth_prior_self_normal_loss: torch.Tensor,
) -> float:
    total = 0.0
    total += float(lambda_depth_prior) * float(depth_prior_loss.detach().item())
    total += float(lambda_depth_prior_normal) * float(depth_prior_normal_loss.detach().item())
    total += float(lambda_depth_prior_distortion) * float(depth_prior_distortion_loss.detach().item())
    total += float(lambda_depth_prior_self_normal) * float(depth_prior_self_normal_loss.detach().item())
    return float(depth_prior_scale) * total


def select_depth_feedback_nominees(
    score_sum: torch.Tensor,
    score_count: torch.Tensor,
    *,
    iteration: int,
    min_visible: int,
    min_score: float,
    top_fraction: float,
    top_points: int,
    candidate_mask: torch.Tensor | None,
    exclude_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    total_count = int(score_sum.shape[0])
    device = score_sum.device
    scores = score_sum / torch.clamp(score_count.to(dtype=score_sum.dtype), min=1.0)
    valid = torch.isfinite(scores) & (score_count >= int(min_visible))
    if candidate_mask is not None:
        valid &= candidate_mask.to(device=device, dtype=torch.bool)
    if exclude_mask is not None:
        valid &= ~exclude_mask.to(device=device, dtype=torch.bool)
    if float(min_score) > 0.0:
        valid &= scores >= float(min_score)
    nominee_count = int(valid.sum().item())
    nominated = torch.zeros((total_count,), dtype=torch.bool, device=device)
    stats = {
        "iteration": float(iteration),
        "candidate_count": float(nominee_count),
        "selected_count": 0.0,
        "score_mean": 0.0,
        "score_max": 0.0,
        "visible_mean": 0.0,
    }
    if nominee_count <= 0:
        return nominated, stats
    limit = nominee_count
    if float(top_fraction) > 0.0:
        limit = min(limit, max(1, int(math.ceil(float(total_count) * float(top_fraction)))))
    if int(top_points) > 0:
        limit = min(limit, int(top_points))
    candidate_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    candidate_scores = scores[candidate_ids]
    topk = torch.topk(candidate_scores, k=int(limit), largest=True).indices
    selected_ids = candidate_ids[topk]
    nominated[selected_ids] = True
    stats["selected_count"] = float(int(selected_ids.numel()))
    stats["score_mean"] = float(candidate_scores.mean().item())
    stats["score_max"] = float(candidate_scores.max().item())
    stats["visible_mean"] = float(score_count[candidate_ids].to(dtype=torch.float32).mean().item())
    return nominated, stats


def apply_depth_feedback_reinit(
    student: GaussianModel,
    optimizer: torch.optim.Optimizer,
    nominee_mask: torch.Tensor,
    *,
    score_sum: torch.Tensor,
    score_count: torch.Tensor,
    mode: str,
) -> int:
    mode = str(mode).strip().lower()
    nominee_mask = nominee_mask.reshape(-1).to(device=student._xyz.device, dtype=torch.bool)
    target_ids = torch.nonzero(nominee_mask, as_tuple=False).reshape(-1)
    if int(target_ids.numel()) <= 0:
        return 0
    donor_mask = ~nominee_mask
    donor_ids_all = torch.nonzero(donor_mask, as_tuple=False).reshape(-1)
    if int(donor_ids_all.numel()) <= 0:
        return 0
    mean_scores = score_sum / torch.clamp(score_count.to(dtype=score_sum.dtype), min=1.0)
    donor_probs = student.get_opacity[donor_ids_all, 0].detach().clamp_min(1e-6)
    donor_probs = donor_probs / (1.0 + torch.clamp(mean_scores[donor_ids_all], min=0.0))
    if not torch.isfinite(donor_probs).all() or float(donor_probs.sum().item()) <= 0.0:
        donor_probs = torch.ones_like(donor_probs)
    sampled_donor_ids, ratio = student._sample_alives(
        probs=donor_probs,
        num=int(target_ids.numel()),
        alive_indices=donor_ids_all,
    )
    new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation = student._update_params(
        sampled_donor_ids,
        ratio=ratio,
    )
    if mode == "reclone":
        stds = student.get_scaling[sampled_donor_ids]
        means = torch.zeros((stds.size(0), 3), device=stds.device, dtype=stds.dtype)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(student._rotation[sampled_donor_ids])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + student.get_xyz[sampled_donor_ids]
    reinit_state_mask = torch.zeros_like(nominee_mask)
    reinit_state_mask[target_ids] = True
    reinit_state_mask[sampled_donor_ids] = True
    zero_optimizer_entries(optimizer, reinit_state_mask)
    with torch.no_grad():
        student._xyz[target_ids] = new_xyz
        student._features_dc[target_ids] = new_features_dc
        student._features_rest[target_ids] = new_features_rest
        student._opacity[target_ids] = new_opacity
        student._scaling[target_ids] = new_scaling
        student._rotation[target_ids] = new_rotation
        student._opacity[sampled_donor_ids] = new_opacity
        student._scaling[sampled_donor_ids] = new_scaling
        if hasattr(student, "_source_tag") and int(student._source_tag.shape[0]) == int(nominee_mask.shape[0]):
            student._source_tag[target_ids] = student._source_tag[sampled_donor_ids]
        if hasattr(student, "_seed_id") and int(student._seed_id.shape[0]) == int(nominee_mask.shape[0]):
            student._seed_id[target_ids] = student._seed_id[sampled_donor_ids]
        if hasattr(student, "_generation") and int(student._generation.shape[0]) == int(nominee_mask.shape[0]):
            student._generation[target_ids] = student._generation[sampled_donor_ids] + 1
        if hasattr(student, "_edge_touched") and int(student._edge_touched.shape[0]) == int(nominee_mask.shape[0]):
            student._edge_touched[target_ids] = student._edge_touched[sampled_donor_ids]
        if hasattr(student, "_edge_touch_iter") and int(student._edge_touch_iter.shape[0]) == int(nominee_mask.shape[0]):
            student._edge_touch_iter[target_ids] = student._edge_touch_iter[sampled_donor_ids]
        if isinstance(student.filter_3D, torch.Tensor) and int(student.filter_3D.shape[0]) == int(nominee_mask.shape[0]):
            student.filter_3D[target_ids] = student.filter_3D[sampled_donor_ids]
        if isinstance(student.max_radii2D, torch.Tensor) and int(student.max_radii2D.shape[0]) == int(nominee_mask.shape[0]):
            student.max_radii2D[target_ids] = 0
    return int(target_ids.numel())


def prune_student_with_optimizer(
    student: GaussianModel,
    optimizer: torch.optim.Optimizer,
    prune_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prune_mask = prune_mask.reshape(-1).to(device=student._xyz.device, dtype=torch.bool)
    total_count = int(student.get_xyz.shape[0])
    if int(prune_mask.shape[0]) != total_count:
        raise ValueError(f"prune mask length mismatch: {prune_mask.shape[0]} vs model n={total_count}")
    prune_count = int(prune_mask.sum().item())
    valid_mask = ~prune_mask
    old_to_new = torch.full((total_count,), -1, dtype=torch.int64, device=student._xyz.device)
    if prune_count <= 0:
        old_to_new[valid_mask] = torch.arange(total_count, device=student._xyz.device, dtype=torch.int64)
        return valid_mask, old_to_new, 0
    old_to_new[valid_mask] = torch.arange(int(valid_mask.sum().item()), device=student._xyz.device, dtype=torch.int64)
    new_optimizer_state: Dict[torch.nn.Parameter, Dict[str, object]] = {}
    group_params: Dict[str, torch.nn.Parameter] = {}
    for group in optimizer.param_groups:
        old_param = group["params"][0]
        new_data = old_param.detach()[valid_mask].clone()
        new_param = torch.nn.Parameter(new_data.requires_grad_(old_param.requires_grad))
        stored_state = optimizer.state.get(old_param, None)
        if stored_state is not None:
            next_state: Dict[str, object] = {}
            for key, value in stored_state.items():
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total_count:
                    next_state[key] = value[valid_mask].clone()
                elif torch.is_tensor(value):
                    next_state[key] = value.clone()
                else:
                    next_state[key] = value
            new_optimizer_state[new_param] = next_state
        group["params"][0] = new_param
        group_params[str(group.get("name", ""))] = new_param
    optimizer.state = new_optimizer_state

    def _slice_param(param: torch.Tensor, *, requires_grad: bool) -> torch.nn.Parameter:
        return torch.nn.Parameter(param.detach()[valid_mask].clone().requires_grad_(requires_grad))

    student._xyz = group_params.get("xyz", _slice_param(student._xyz, requires_grad=student._xyz.requires_grad))
    student._opacity = group_params.get("opacity", _slice_param(student._opacity, requires_grad=student._opacity.requires_grad))
    student._scaling = group_params.get("scale", _slice_param(student._scaling, requires_grad=student._scaling.requires_grad))
    student._features_dc = group_params.get("features_dc", _slice_param(student._features_dc, requires_grad=student._features_dc.requires_grad))
    student._features_rest = group_params.get("features_rest", _slice_param(student._features_rest, requires_grad=student._features_rest.requires_grad))
    student._rotation = _slice_param(student._rotation, requires_grad=False)
    if hasattr(student, "_source_tag") and int(student._source_tag.shape[0]) == total_count:
        student._source_tag = student._source_tag[valid_mask]
    if hasattr(student, "_seed_id") and int(student._seed_id.shape[0]) == total_count:
        student._seed_id = student._seed_id[valid_mask]
    if hasattr(student, "_generation") and int(student._generation.shape[0]) == total_count:
        student._generation = student._generation[valid_mask]
    if hasattr(student, "_edge_touched") and int(student._edge_touched.shape[0]) == total_count:
        student._edge_touched = student._edge_touched[valid_mask]
    if hasattr(student, "_edge_touch_iter") and int(student._edge_touch_iter.shape[0]) == total_count:
        student._edge_touch_iter = student._edge_touch_iter[valid_mask]
    if isinstance(student.filter_3D, torch.Tensor) and int(student.filter_3D.shape[0]) == total_count:
        student.filter_3D = student.filter_3D[valid_mask]
    if isinstance(student.max_radii2D, torch.Tensor) and int(student.max_radii2D.shape[0]) == total_count:
        student.max_radii2D = student.max_radii2D[valid_mask]
    if isinstance(student.xyz_gradient_accum, torch.Tensor) and int(student.xyz_gradient_accum.shape[0]) == total_count:
        student.xyz_gradient_accum = student.xyz_gradient_accum[valid_mask]
    if isinstance(student.xyz_gradient_accum_abs, torch.Tensor) and int(student.xyz_gradient_accum_abs.shape[0]) == total_count:
        student.xyz_gradient_accum_abs = student.xyz_gradient_accum_abs[valid_mask]
    if isinstance(student.xyz_gradient_accum_abs_max, torch.Tensor) and int(student.xyz_gradient_accum_abs_max.shape[0]) == total_count:
        student.xyz_gradient_accum_abs_max = student.xyz_gradient_accum_abs_max[valid_mask]
    if isinstance(student.denom, torch.Tensor) and int(student.denom.shape[0]) == total_count:
        student.denom = student.denom[valid_mask]
    return valid_mask, old_to_new, prune_count


@torch.no_grad()
def evaluate_surface_losses(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor]],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    min_pixels: float,
    depth_relative_min: float,
) -> Dict[str, float]:
    depth_losses = []
    depth_prior_losses = []
    depth_prior_distortion_losses = []
    depth_prior_self_normal_losses = []
    alpha_losses = []
    rgb_losses = []
    for camera, target in zip(cameras, cache):
        render_pkg = render_simple(camera, student, background)
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"].clamp(0.0, 1.0)
        rgb = render_pkg["render"].clamp(0.0, 1.0)
        mask = target["surface_mask"]
        weight = target.get("surface_weight")
        depth_rel = torch.abs(depth - target["target_depth"]) / torch.clamp(target["target_depth"], min=float(depth_relative_min))
        depth_loss = masked_weighted_mean(depth_rel, mask, min_pixels, weight)
        if depth_loss is not None:
            depth_losses.append(float(depth_loss.item()))
        if target.get("depth_prior_target") is not None:
            prior_rel = torch.abs(depth - target["depth_prior_target"]) / torch.clamp(
                target["depth_prior_target"],
                min=float(depth_relative_min),
            )
            prior_loss = masked_weighted_mean(
                prior_rel,
                target["depth_prior_mask"],
                min_pixels,
                target["depth_prior_weight"],
            )
            if prior_loss is not None:
                depth_prior_losses.append(float(prior_loss.item()))
            prior_distortion_loss = compute_depth_prior_distortion_loss(
                render_pkg,
                target,
                min_pixels=min_pixels,
            )
            if prior_distortion_loss is not None:
                depth_prior_distortion_losses.append(float(prior_distortion_loss.item()))
            prior_self_normal_loss = compute_depth_prior_self_normal_loss(
                camera,
                render_pkg,
                target,
                min_pixels=min_pixels,
            )
            if prior_self_normal_loss is not None:
                depth_prior_self_normal_losses.append(float(prior_self_normal_loss.item()))
        alpha_losses.append(float(torch.mean(torch.abs(alpha - target["target_alpha"])).item()))
        rgb_losses.append(float(torch.mean(torch.abs(rgb - target["base_rgb"])).item()))
    return {
        "depth": float(np.mean(depth_losses)) if depth_losses else 0.0,
        "depth_prior": float(np.mean(depth_prior_losses)) if depth_prior_losses else 0.0,
        "depth_prior_distortion": float(np.mean(depth_prior_distortion_losses)) if depth_prior_distortion_losses else 0.0,
        "depth_prior_self_normal": float(np.mean(depth_prior_self_normal_losses)) if depth_prior_self_normal_losses else 0.0,
        "alpha": float(np.mean(alpha_losses)) if alpha_losses else 0.0,
        "rgb_preserve": float(np.mean(rgb_losses)) if rgb_losses else 0.0,
    }


def compute_mip_closure_losses(
    render_pkg: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    kernel_size: int,
    alpha_threshold: float,
    min_pixels: float,
    depth_relative_min: float,
    charbonnier_eps: float,
    reference_lowpass: bool,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    rgb = render_pkg["render"].clamp(0.0, 1.0)
    alpha = render_pkg["alpha"].clamp(0.0, 1.0)
    depth = render_pkg["depth"]
    alpha_low = blur_hw(alpha[0], kernel_size)
    premul_low = blur_chw(rgb * alpha, kernel_size)
    depth_low_num = blur_hw(depth[0] * alpha[0], kernel_size)
    depth_low = depth_low_num / alpha_low.clamp_min(1e-6)

    ref_alpha = target["mip_ref_alpha"][0].to(device=alpha.device, dtype=alpha.dtype)
    ref_premul = target["mip_ref_premul"].to(device=rgb.device, dtype=rgb.dtype)
    ref_depth = target["mip_ref_depth"][0].to(device=depth.device, dtype=depth.dtype)
    if reference_lowpass:
        ref_alpha_cmp = blur_hw(ref_alpha, kernel_size)
        ref_premul_cmp = blur_chw(ref_premul, kernel_size)
        ref_depth_num = blur_hw(ref_depth * ref_alpha, kernel_size)
        ref_depth_cmp = ref_depth_num / ref_alpha_cmp.clamp_min(1e-6)
    else:
        ref_alpha_cmp = ref_alpha
        ref_premul_cmp = ref_premul
        ref_depth_cmp = ref_depth

    support = (ref_alpha_cmp >= float(alpha_threshold)) & torch.isfinite(ref_alpha_cmp)

    alpha_loss = masked_mean(torch.abs(alpha_low - ref_alpha_cmp), support, min_pixels)
    premul_loss = masked_mean(torch.abs(premul_low - ref_premul_cmp), support, min_pixels)

    depth_mask = (
        support
        & (alpha_low >= float(alpha_threshold))
        & torch.isfinite(ref_depth_cmp)
        & torch.isfinite(depth_low)
        & (ref_depth_cmp > 1e-6)
    )
    depth_rel = torch.abs(depth_low - ref_depth_cmp) / torch.clamp(ref_depth_cmp, min=float(depth_relative_min))
    depth_loss = masked_mean(charbonnier(depth_rel, charbonnier_eps), depth_mask, min_pixels)
    return alpha_loss, premul_loss, depth_loss


def compute_mip_closure_over_losses(
    render_pkg: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    kernel_size: int,
    alpha_threshold: float,
    min_pixels: float,
    reference_lowpass: bool,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    rgb = render_pkg["render"].clamp(0.0, 1.0)
    alpha = render_pkg["alpha"].clamp(0.0, 1.0)
    alpha_low = blur_hw(alpha[0], kernel_size)
    premul_low = blur_chw(rgb * alpha, kernel_size)

    ref_alpha = target["mip_ref_alpha"][0].to(device=alpha.device, dtype=alpha.dtype)
    ref_premul = target["mip_ref_premul"].to(device=premul_low.device, dtype=premul_low.dtype)
    if reference_lowpass:
        ref_alpha_cmp = blur_hw(ref_alpha, kernel_size)
        ref_premul_cmp = blur_chw(ref_premul, kernel_size)
    else:
        ref_alpha_cmp = ref_alpha
        ref_premul_cmp = ref_premul

    support = (ref_alpha_cmp >= float(alpha_threshold)) & torch.isfinite(ref_alpha_cmp)
    alpha_over = masked_mean(torch.relu(alpha_low - ref_alpha_cmp), support, min_pixels)
    luma_w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=premul_low.dtype, device=premul_low.device).view(3, 1, 1)
    premul_luma_diff = torch.sum((premul_low - ref_premul_cmp) * luma_w, dim=0)
    premul_over = masked_mean(torch.relu(premul_luma_diff), support, min_pixels)
    return alpha_over, premul_over


@torch.no_grad()
def evaluate_mip_closure_losses(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor]],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    splat_args: ExtendedSettings,
    kernel_size: int,
    alpha_threshold: float,
    min_pixels: float,
    depth_relative_min: float,
    charbonnier_eps: float,
    reference_lowpass: bool,
) -> Dict[str, float]:
    alpha_losses = []
    premul_losses = []
    depth_losses = []
    for camera, target in zip(cameras, cache):
        render_pkg = render_simple(camera, student, background, splat_args=splat_args)
        alpha_loss, premul_loss, depth_loss = compute_mip_closure_losses(
            render_pkg,
            target,
            kernel_size=kernel_size,
            alpha_threshold=alpha_threshold,
            min_pixels=min_pixels,
            depth_relative_min=depth_relative_min,
            charbonnier_eps=charbonnier_eps,
            reference_lowpass=reference_lowpass,
        )
        if alpha_loss is not None:
            alpha_losses.append(float(alpha_loss.item()))
        if premul_loss is not None:
            premul_losses.append(float(premul_loss.item()))
        if depth_loss is not None:
            depth_losses.append(float(depth_loss.item()))
    return {
        "alpha": float(np.mean(alpha_losses)) if alpha_losses else 0.0,
        "premul": float(np.mean(premul_losses)) if premul_losses else 0.0,
        "depth": float(np.mean(depth_losses)) if depth_losses else 0.0,
    }


@torch.no_grad()
def evaluate_mip_closure_over_losses(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor]],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    splat_args: ExtendedSettings,
    kernel_size: int,
    alpha_threshold: float,
    min_pixels: float,
    reference_lowpass: bool,
) -> Dict[str, float]:
    alpha_losses = []
    premul_losses = []
    for camera, target in zip(cameras, cache):
        render_pkg = render_simple(camera, student, background, splat_args=splat_args)
        alpha_over, premul_over = compute_mip_closure_over_losses(
            render_pkg,
            target,
            kernel_size=kernel_size,
            alpha_threshold=alpha_threshold,
            min_pixels=min_pixels,
            reference_lowpass=reference_lowpass,
        )
        if alpha_over is not None:
            alpha_losses.append(float(alpha_over.item()))
        if premul_over is not None:
            premul_losses.append(float(premul_over.item()))
    return {
        "alpha_over": float(np.mean(alpha_losses)) if alpha_losses else 0.0,
        "premul_luma_over": float(np.mean(premul_losses)) if premul_losses else 0.0,
    }


@torch.no_grad()
def evaluate_sr_prior_losses(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    min_pixels: float,
    min_valid_ratio: float,
    prior_delta_clip: float,
    disable_hf_residual: bool,
) -> Dict[str, float]:
    l1_losses = []
    hf_losses = []
    for camera, target in zip(cameras, cache):
        render_pkg = render_simple(camera, student, background)
        rgb = render_pkg["render"].clamp(0.0, 1.0)
        loss_l1, loss_hf = compute_prepared_sr_losses(
            rgb,
            target,
            min_pixels=min_pixels,
            min_valid_ratio=min_valid_ratio,
            prior_delta_clip=prior_delta_clip,
            disable_hf_residual=disable_hf_residual,
        )
        if loss_l1 is not None:
            l1_losses.append(float(loss_l1.item()))
        if loss_hf is not None:
            hf_losses.append(float(loss_hf.item()))
    return {
        "sr_l1": float(np.mean(l1_losses)) if l1_losses else 0.0,
        "sr_hf": float(np.mean(hf_losses)) if hf_losses else 0.0,
    }


@torch.no_grad()
def evaluate_premul_hf_excess_losses(
    cameras: Sequence[object],
    cache: Sequence[Dict[str, torch.Tensor | str | float | None]],
    student: GaussianModel,
    background: torch.Tensor,
    *,
    kernel_size: int,
    excess_ratio: float,
    margin: float,
    min_pixels: float,
    min_valid_ratio: float,
    prior_delta_clip: float,
) -> Dict[str, float]:
    losses = []
    for camera, target in zip(cameras, cache):
        render_pkg = render_simple(camera, student, background)
        loss = compute_premul_hf_excess_loss(
            render_pkg,
            target,
            kernel_size=kernel_size,
            excess_ratio=excess_ratio,
            margin=margin,
            min_pixels=min_pixels,
            min_valid_ratio=min_valid_ratio,
            prior_delta_clip=prior_delta_clip,
        )
        if loss is not None:
            losses.append(float(loss.item()))
    return {"premul_hf_excess": float(np.mean(losses)) if losses else 0.0}


def stats_from_tensor(value: torch.Tensor) -> Dict[str, float]:
    arr = value.detach().float().cpu().numpy().reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def summarize_prior_cache(cache: Sequence[Dict[str, object]]) -> Dict[str, object]:
    depth_ratios = []
    surface_weight_values = []
    depth_align = []
    depth_loaded = 0
    for target in cache:
        depth_mask = target.get("depth_prior_mask")
        if torch.is_tensor(depth_mask):
            depth_ratios.append(float(depth_mask.float().mean().item()))
            if target.get("depth_prior_target") is not None:
                depth_loaded += 1
        surface_weight = target.get("surface_weight")
        if torch.is_tensor(surface_weight):
            surface_weight_values.append(surface_weight.detach().float().reshape(-1).cpu())
        depth_info = target.get("depth_prior_info")
        if isinstance(depth_info, dict) and depth_info.get("status") == "loaded":
            align = depth_info.get("align", {})
            if isinstance(align, dict):
                depth_align.append(align)
    surface_weight_stats = (
        stats_from_tensor(torch.cat(surface_weight_values, dim=0))
        if surface_weight_values
        else {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    )
    return {
        "views": int(len(cache)),
        "depth_loaded_views": int(depth_loaded),
        "depth_active_ratio_mean": float(np.mean(depth_ratios)) if depth_ratios else 0.0,
        "surface_weight": surface_weight_stats,
        "depth_align": depth_align,
    }


def summarize_sr_cache(cache: Sequence[Dict[str, torch.Tensor | str | float | None]]) -> Dict[str, object]:
    if not cache:
        return {
            "views": 0,
            "valid_ratio_mean": 0.0,
            "mask_mean": 0.0,
            "structure_mask_mean": 0.0,
        }
    valid_ratios = [float(item.get("valid_ratio", 0.0) or 0.0) for item in cache]
    mask_means = [float(item.get("mask_mean", 0.0) or 0.0) for item in cache]
    structure_means = [float(item.get("structure_mask_mean", 0.0) or 0.0) for item in cache]
    return {
        "views": int(len(cache)),
        "valid_ratio_mean": float(np.mean(valid_ratios)),
        "valid_ratio_min": float(np.min(valid_ratios)),
        "mask_mean": float(np.mean(mask_means)),
        "structure_mask_mean": float(np.mean(structure_means)),
        "prior_paths_sample": [str(item.get("prior_path")) for item in cache[:8]],
        "anchor_paths_sample": [str(item.get("anchor_path")) for item in cache[:8]],
    }


def main() -> None:
    parser = ArgumentParser(description="Conservatively pull a mip checkpoint toward a frozen SOF surface teacher.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--mip_model_path", required=True)
    parser.add_argument("--sof_surface_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--mip_iteration", type=int, default=30000)
    parser.add_argument("--sof_iteration", type=int, default=30000)
    parser.add_argument("--output_iteration", type=int, default=30000)
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--xyz_lr", type=float, default=2e-5)
    parser.add_argument("--opacity_lr", type=float, default=0.0)
    parser.add_argument("--scale_lr", type=float, default=0.0)
    parser.add_argument("--lambda_rgb_preserve", type=float, default=1.0)
    parser.add_argument("--lambda_depth", type=float, default=0.25)
    parser.add_argument("--lambda_normal", type=float, default=0.03)
    parser.add_argument("--lambda_alpha", type=float, default=0.05)
    parser.add_argument("--lambda_mip_closure_alpha", type=float, default=0.0)
    parser.add_argument("--lambda_mip_closure_premul", type=float, default=0.0)
    parser.add_argument("--lambda_mip_closure_depth", type=float, default=0.0)
    parser.add_argument("--lambda_mip_obs_closure_alpha", type=float, default=0.0)
    parser.add_argument("--lambda_mip_obs_closure_premul", type=float, default=0.0)
    parser.add_argument("--lambda_mip_obs_closure_depth", type=float, default=0.0)
    parser.add_argument("--lambda_opacity_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_scale_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_anchor", type=float, default=50.0)
    parser.add_argument("--sr_prior_root", type=str, default=None)
    parser.add_argument("--sr_prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--sr_prior_mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--sr_anchor_subdir", type=str, default="aligned_references")
    parser.add_argument("--sr_prior_mask_suffix", type=str, default="")
    parser.add_argument("--sr_images_subdir", type=str, default="")
    parser.add_argument("--sr_max_views", type=int, default=0)
    parser.add_argument("--sr_prior_l1_weight", type=float, default=0.0)
    parser.add_argument("--sr_prior_hf_weight", type=float, default=0.0)
    parser.add_argument("--sr_prior_warmup_start_iter", type=int, default=0)
    parser.add_argument("--sr_prior_warmup_end_iter", type=int, default=0)
    parser.add_argument("--sr_prior_start_scale", type=float, default=1.0)
    parser.add_argument("--sr_prior_end_scale", type=float, default=1.0)
    parser.add_argument("--sr_prior_update_scale", type=float, default=1.0)
    parser.add_argument("--sr_prior_schedule_mode", choices=["linear", "smoothstep"], default="smoothstep")
    parser.add_argument("--sr_prior_mask_floor", type=float, default=0.0)
    parser.add_argument("--sr_prior_consistency_threshold", type=float, default=0.08)
    parser.add_argument("--sr_prior_min_valid_ratio", type=float, default=0.80)
    parser.add_argument("--sr_prior_min_pixels", type=float, default=64.0)
    parser.add_argument("--sr_prior_delta_clip", type=float, default=0.08)
    parser.add_argument("--disable_sr_prior_hf_residual", action="store_true")
    parser.add_argument("--depth_prior_root", type=str, default=None)
    parser.add_argument("--depth_prior_subdirs", type=str, default="depth,")
    parser.add_argument("--depth_prior_confidence_subdirs", type=str, default="auto")
    parser.add_argument("--lambda_depth_prior", type=float, default=0.0)
    parser.add_argument("--lambda_depth_prior_normal", type=float, default=0.0)
    parser.add_argument("--lambda_depth_prior_distortion", type=float, default=0.0)
    parser.add_argument("--lambda_depth_prior_self_normal", type=float, default=0.0)
    parser.add_argument("--depth_prior_confidence_min", type=float, default=0.05)
    parser.add_argument("--depth_prior_agreement_threshold", type=float, default=0.15)
    parser.add_argument("--depth_prior_agreement_floor", type=float, default=0.0)
    parser.add_argument("--depth_prior_align_mode", choices=["affine_robust", "identity"], default="affine_robust")
    parser.add_argument("--depth_prior_align_min_pixels", type=int, default=2048)
    parser.add_argument("--depth_prior_surface_weight_boost", type=float, default=0.0)
    parser.add_argument("--depth_prior_warmup_start_iter", type=int, default=0)
    parser.add_argument("--depth_prior_warmup_end_iter", type=int, default=0)
    parser.add_argument("--depth_prior_start_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_end_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_update_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_schedule_mode", choices=["linear", "smoothstep"], default="smoothstep")
    parser.add_argument("--depth_prior_weight_gain", type=float, default=1.0)
    parser.add_argument("--depth_prior_weight_power", type=float, default=1.0)
    parser.add_argument("--depth_prior_weight_min", type=float, default=0.0)
    parser.add_argument("--min_surface_alpha", type=float, default=0.08)
    parser.add_argument("--min_loss_pixels", type=float, default=256.0)
    parser.add_argument("--mip_closure_kernel", type=int, default=25)
    parser.add_argument("--mip_closure_alpha_threshold", type=float, default=0.05)
    parser.add_argument("--mip_closure_reference_lowpass", type=int, default=1)
    parser.add_argument("--mip_obs_closure_images_subdir", type=str, default="")
    parser.add_argument("--mip_obs_closure_split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--mip_obs_closure_max_views", type=int, default=0)
    parser.add_argument("--filter_images_subdir", type=str, default="")
    parser.add_argument("--filter_split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--filter_max_views", type=int, default=0)
    parser.add_argument("--depth_relative_min", type=float, default=0.5)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--max_displacement_ratio", type=float, default=0.002)
    parser.add_argument("--max_displacement_abs", type=float, default=0.0)
    parser.add_argument("--enable_opacity_update", type=int, default=0)
    parser.add_argument("--enable_scale_update", type=int, default=0)
    parser.add_argument("--optimize_gaussian_mask_payload", type=str, default=None)
    parser.add_argument("--optimize_gaussian_mask_key", type=str, default="selected_mask")
    parser.add_argument("--gaussian_update_scale", type=float, default=1.0)
    parser.add_argument("--gaussian_scale_axis_mode", choices=["all", "major_only"], default="all")
    parser.add_argument("--gaussian_scale_min_multiplier", type=float, default=0.0)
    parser.add_argument("--gaussian_scale_max_multiplier", type=float, default=0.0)
    parser.add_argument("--gradient_guard_mode", choices=["none", "reset", "prune"], default="none")
    parser.add_argument("--gradient_guard_start_iter", type=int, default=0)
    parser.add_argument("--gradient_guard_xyz_threshold", type=float, default=0.0)
    parser.add_argument("--gradient_guard_opacity_threshold", type=float, default=0.0)
    parser.add_argument("--gradient_guard_scale_threshold", type=float, default=0.0)
    parser.add_argument("--gradient_guard_max_fraction", type=float, default=0.0)
    parser.add_argument("--gradient_guard_max_points", type=int, default=0)
    parser.add_argument("--gradient_guard_only_update_mask", type=int, default=1)
    parser.add_argument("--depth_feedback_mode", choices=["none", "dropout", "reset", "prune", "relocate", "reclone"], default="none")
    parser.add_argument("--depth_feedback_start_iter", type=int, default=0)
    parser.add_argument("--depth_feedback_interval", type=int, default=0)
    parser.add_argument("--depth_feedback_min_visible", type=int, default=3)
    parser.add_argument("--depth_feedback_min_score", type=float, default=0.0)
    parser.add_argument("--depth_feedback_top_fraction", type=float, default=0.0)
    parser.add_argument("--depth_feedback_top_points", type=int, default=0)
    parser.add_argument("--depth_feedback_dropout_iters", type=int, default=0)
    parser.add_argument("--depth_feedback_only_update_mask", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    mip_model_path = Path(args.mip_model_path).expanduser().resolve()
    sof_model_path = Path(args.sof_surface_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)
    sr_prior_root = Path(args.sr_prior_root).expanduser().resolve() if args.sr_prior_root else None
    sr_prior_dir = sr_prior_root / str(args.sr_prior_subdir) if sr_prior_root is not None and args.sr_prior_subdir else sr_prior_root
    sr_prior_mask_dir = (
        sr_prior_root / str(args.sr_prior_mask_subdir)
        if sr_prior_root is not None and args.sr_prior_mask_subdir
        else None
    )
    sr_anchor_dir = (
        sr_prior_root / str(args.sr_anchor_subdir)
        if sr_prior_root is not None and args.sr_anchor_subdir
        else None
    )
    depth_prior_root = Path(args.depth_prior_root).expanduser().resolve() if args.depth_prior_root else None

    if sr_prior_root is not None and not sr_prior_root.is_dir():
        raise FileNotFoundError(f"SR prior root not found: {sr_prior_root}")
    if sr_prior_dir is not None and not sr_prior_dir.is_dir():
        raise FileNotFoundError(f"SR fused prior dir not found: {sr_prior_dir}")
    if sr_prior_mask_dir is not None and not sr_prior_mask_dir.is_dir():
        print(f"[mip-to-sof-surface] warning: SR prior mask dir missing, using consistency gate only: {sr_prior_mask_dir}")
        sr_prior_mask_dir = None
    if sr_anchor_dir is not None and not sr_anchor_dir.is_dir():
        raise FileNotFoundError(
            f"SR GT-free anchor dir not found: {sr_anchor_dir}. "
            "Rebuild prepared SR priors so aligned_references are available."
        )
    if depth_prior_root is not None and not depth_prior_root.is_dir():
        raise FileNotFoundError(f"Depth prior root not found: {depth_prior_root}")
    sr_prior_index = index_image_dir(str(sr_prior_dir)) if sr_prior_dir is not None else None
    sr_anchor_index = index_image_dir(str(sr_anchor_dir)) if sr_anchor_dir is not None else None
    depth_prior_subdirs = parse_subdir_list(args.depth_prior_subdirs, default_auto=("depth", ""))
    depth_prior_confidence_subdirs = parse_subdir_list(
        args.depth_prior_confidence_subdirs,
        default_auto=("confidence", "conf", "depth_conf", "valid"),
    )

    mip_iter = resolve_iteration(mip_model_path, int(args.mip_iteration))
    sof_iter = resolve_iteration(sof_model_path, int(args.sof_iteration))
    dataset_args = build_dataset_args(str(scene_root), str(mip_model_path), str(args.images_subdir))
    dataset = ModelParams(None).extract(dataset_args)

    student = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, student, load_iteration=mip_iter, shuffle=False, skip_test=False, skip_train=False)
    train_cameras = scene.getTrainCameras().copy()
    selected_cameras = select_uniform(train_cameras, int(args.max_views))
    if not selected_cameras:
        raise RuntimeError("No training cameras found.")

    obs_closure_cameras: List[object] = []
    obs_closure_images_subdir = str(args.mip_obs_closure_images_subdir).strip()
    use_obs_closure = obs_closure_images_subdir != "" and (
        float(args.lambda_mip_obs_closure_alpha) > 0.0
        or float(args.lambda_mip_obs_closure_premul) > 0.0
        or float(args.lambda_mip_obs_closure_depth) > 0.0
    )
    if use_obs_closure:
        obs_all_cameras = load_cameras_for_split(
            scene_root,
            mip_model_path,
            obs_closure_images_subdir,
            str(args.mip_obs_closure_split),
        )
        obs_max_views = int(args.mip_obs_closure_max_views)
        obs_closure_cameras = select_uniform(obs_all_cameras, obs_max_views)
        if not obs_closure_cameras:
            raise RuntimeError(
                "MIP observation closure was enabled but no cameras were found for "
                f"images={obs_closure_images_subdir} split={args.mip_obs_closure_split}."
            )

    sr_cameras: List[object] = []
    sr_images_subdir = str(args.sr_images_subdir).strip() or str(args.images_subdir)
    use_sr_prior = sr_prior_index is not None and (
        float(args.sr_prior_l1_weight) > 0.0 or float(args.sr_prior_hf_weight) > 0.0
    )
    train_opacity = bool(int(args.enable_opacity_update))
    train_scale = bool(int(args.enable_scale_update))
    external_update_mask = load_gaussian_update_mask_payload(
        args.optimize_gaussian_mask_payload,
        str(args.optimize_gaussian_mask_key),
        total_gaussians=student.get_xyz.shape[0],
    )
    if external_update_mask is not None:
        selected_count = int(external_update_mask.sum().item())
        print(
            f"[mip-to-sof-surface] gaussian mask  : key={args.optimize_gaussian_mask_key} "
            f"selected={selected_count}/{external_update_mask.shape[0]} "
            f"scale={float(args.gaussian_update_scale):.3g} "
            f"axis={args.gaussian_scale_axis_mode} "
            f"scale_range=[{float(args.gaussian_scale_min_multiplier):.3g}, "
            f"{float(args.gaussian_scale_max_multiplier):.3g}]"
        )
        if selected_count <= 0:
            raise ValueError("External Gaussian update mask is empty; regenerate the payload for this checkpoint.")
    if use_sr_prior:
        if sr_images_subdir == str(args.images_subdir):
            sr_all_cameras = train_cameras
        else:
            sr_all_cameras = load_train_cameras_only(scene_root, mip_model_path, sr_images_subdir)
        selected_names = {normalize_image_name(cam.image_name) for cam in selected_cameras}
        sr_candidates = [cam for cam in sr_all_cameras if normalize_image_name(cam.image_name) in selected_names]
        if not sr_candidates:
            sr_candidates = sr_all_cameras
        sr_max_views = int(args.sr_max_views) if int(args.sr_max_views) > 0 else int(args.max_views)
        sr_cameras = select_uniform(sr_candidates, sr_max_views)
        if not sr_cameras:
            raise RuntimeError("SR prior was enabled but no SR supervision cameras were found.")

    filter_images_subdir = str(args.filter_images_subdir).strip()
    if filter_images_subdir:
        filter_all_cameras = load_cameras_for_split(
            scene_root,
            mip_model_path,
            filter_images_subdir,
            str(args.filter_split),
        )
        student_filter_cameras = select_uniform(filter_all_cameras, int(args.filter_max_views))
        if not student_filter_cameras:
            raise RuntimeError(
                "Filter camera override was enabled but no cameras were found for "
                f"images={filter_images_subdir} split={args.filter_split}."
            )
    else:
        student_filter_cameras = selected_cameras + [cam for cam in sr_cameras if cam not in selected_cameras]
        student_filter_cameras += [cam for cam in obs_closure_cameras if cam not in student_filter_cameras]

    teacher = load_model_ply(sof_model_path, sof_iter, dataset.sh_degree)
    mip_splat_args = load_splat_settings(mip_model_path)
    print(f"[mip-to-sof-surface] scene root      : {scene_root}")
    print(f"[mip-to-sof-surface] mip model       : {mip_model_path} iter={mip_iter}")
    print(f"[mip-to-sof-surface] SOF teacher     : {sof_model_path} iter={sof_iter}")
    print(f"[mip-to-sof-surface] output model    : {output_model_path}")
    print(f"[mip-to-sof-surface] selected views  : {len(selected_cameras)}")
    print(
        f"[mip-to-sof-surface] train params    : xyz=1 "
        f"opacity={int(train_opacity)} scale={int(train_scale)} "
        f"lrs={float(args.xyz_lr):.3g}/{float(args.opacity_lr):.3g}/{float(args.scale_lr):.3g}"
    )
    print(f"[mip-to-sof-surface] SR prior root  : {sr_prior_root if sr_prior_root is not None else '(disabled)'}")
    if use_sr_prior:
        print(f"[mip-to-sof-surface] SR views       : {sr_images_subdir} max={len(sr_cameras)}")
        print(
            f"[mip-to-sof-surface] SR schedule    : "
            f"{float(args.sr_prior_start_scale):.3g}->{float(args.sr_prior_end_scale):.3g} "
            f"update={float(args.sr_prior_update_scale):.3g} "
            f"iter={int(args.sr_prior_warmup_start_iter)}->{int(args.sr_prior_warmup_end_iter)} "
            f"mode={args.sr_prior_schedule_mode}"
        )
    print(f"[mip-to-sof-surface] depth prior    : {depth_prior_root if depth_prior_root is not None else '(disabled)'}")
    if depth_prior_root is not None:
        print(
            "[mip-to-sof-surface] depth cfg      : "
            f"l1={float(args.lambda_depth_prior):.4g} "
            f"prior_n={float(args.lambda_depth_prior_normal):.4g} "
            f"distort={float(args.lambda_depth_prior_distortion):.4g} "
            f"self_n={float(args.lambda_depth_prior_self_normal):.4g} "
            f"agree={float(args.depth_prior_agreement_threshold):.4g} "
            f"floor={float(args.depth_prior_agreement_floor):.4g} "
            f"align={args.depth_prior_align_mode} "
            f"sched={float(args.depth_prior_start_scale):.3g}->{float(args.depth_prior_end_scale):.3g} "
            f"x{float(args.depth_prior_update_scale):.3g} "
            f"weight={float(args.depth_prior_weight_gain):.3g}*w^{float(args.depth_prior_weight_power):.3g} "
            f"wmin={float(args.depth_prior_weight_min):.3g}"
        )
    gradient_guard_mode = str(args.gradient_guard_mode).strip().lower()
    if gradient_guard_mode != "none":
        print(
            "[mip-to-sof-surface] grad guard     : "
            f"mode={gradient_guard_mode} start={int(args.gradient_guard_start_iter)} "
            f"xyz>{float(args.gradient_guard_xyz_threshold):.3g} "
            f"opacity>{float(args.gradient_guard_opacity_threshold):.3g} "
            f"scale>{float(args.gradient_guard_scale_threshold):.3g} "
            f"cap={float(args.gradient_guard_max_fraction):.4g}/{int(args.gradient_guard_max_points)} "
            f"masked={int(bool(int(args.gradient_guard_only_update_mask)))}"
        )
    depth_feedback_mode = str(args.depth_feedback_mode).strip().lower()
    if depth_feedback_mode != "none":
        print(
            "[mip-to-sof-surface] depth feedback : "
            f"mode={depth_feedback_mode} start={int(args.depth_feedback_start_iter)} "
            f"interval={int(args.depth_feedback_interval)} "
            f"min_visible={int(args.depth_feedback_min_visible)} "
            f"min_score={float(args.depth_feedback_min_score):.3g} "
            f"cap={float(args.depth_feedback_top_fraction):.4g}/{int(args.depth_feedback_top_points)} "
            f"dropout={int(args.depth_feedback_dropout_iters)} "
            f"masked={int(bool(int(args.depth_feedback_only_update_mask)))}"
        )
    if filter_images_subdir:
        print(
            "[mip-to-sof-surface] filter3D       : "
            f"images={filter_images_subdir} split={args.filter_split} views={len(student_filter_cameras)}"
        )
    else:
        print(f"[mip-to-sof-surface] filter3D       : training-domain views={len(student_filter_cameras)}")
    print(
        "[mip-to-sof-surface] mip closure    : "
        f"alpha={float(args.lambda_mip_closure_alpha)} "
        f"premul={float(args.lambda_mip_closure_premul)} "
        f"depth={float(args.lambda_mip_closure_depth)} "
        f"kernel={int(args.mip_closure_kernel)} alpha>{float(args.mip_closure_alpha_threshold)} "
        f"ref_lowpass={int(args.mip_closure_reference_lowpass)}"
    )
    if use_obs_closure:
        print(
            "[mip-to-sof-surface] obs closure    : "
            f"images={obs_closure_images_subdir} split={args.mip_obs_closure_split} "
            f"views={len(obs_closure_cameras)} "
            f"alpha={float(args.lambda_mip_obs_closure_alpha)} "
            f"premul={float(args.lambda_mip_obs_closure_premul)} "
            f"depth={float(args.lambda_mip_obs_closure_depth)}"
        )
    else:
        print("[mip-to-sof-surface] obs closure    : disabled")

    student.compute_3D_filter(student_filter_cameras, CUDA=False)
    teacher.compute_3D_filter(selected_cameras, CUDA=False)
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    teacher_cache = render_teacher_cache(
        selected_cameras,
        teacher,
        student,
        background,
        mip_splat_args=mip_splat_args,
        min_surface_alpha=float(args.min_surface_alpha),
        depth_prior_root=depth_prior_root,
        depth_prior_subdirs=depth_prior_subdirs,
        depth_prior_confidence_subdirs=depth_prior_confidence_subdirs,
        depth_prior_confidence_min=float(args.depth_prior_confidence_min),
        depth_prior_agreement_threshold=float(args.depth_prior_agreement_threshold),
        depth_prior_agreement_floor=float(args.depth_prior_agreement_floor),
        depth_prior_align_mode=str(args.depth_prior_align_mode),
        depth_prior_align_min_pixels=int(args.depth_prior_align_min_pixels),
        depth_prior_surface_weight_boost=float(args.depth_prior_surface_weight_boost),
        depth_prior_weight_gain=float(args.depth_prior_weight_gain),
        depth_prior_weight_power=float(args.depth_prior_weight_power),
        depth_prior_weight_min=float(args.depth_prior_weight_min),
    )
    obs_closure_cache = render_mip_closure_cache(
        obs_closure_cameras,
        student,
        background,
        mip_splat_args=mip_splat_args,
    ) if use_obs_closure else []
    sr_cameras, sr_cache = build_prepared_sr_cache(
        sr_cameras,
        student,
        background,
        sr_prior_index=sr_prior_index,
        sr_anchor_index=sr_anchor_index,
        sr_prior_mask_dir=sr_prior_mask_dir,
        sr_prior_mask_suffix=str(args.sr_prior_mask_suffix),
        prior_consistency_threshold=float(args.sr_prior_consistency_threshold),
        prior_mask_floor=float(args.sr_prior_mask_floor),
    )
    if use_sr_prior and not sr_cache:
        raise RuntimeError(
            "SR prior was enabled but no prepared prior images matched the selected SR cameras. "
            f"root={sr_prior_root}, subdir={args.sr_prior_subdir}, images={sr_images_subdir}"
        )

    xyz_init = student._xyz.detach().clone()
    opacity_init = student.get_opacity.detach().clone()
    opacity_raw_init = student._opacity.detach().clone()
    scale_init = student.get_scaling.detach().clone()
    scale_raw_init = student._scaling.detach().clone()
    scale_param_mask = build_scale_update_mask(
        scale_init,
        update_mask=external_update_mask,
        axis_mode=str(args.gaussian_scale_axis_mode),
    ) if train_scale else None
    bbox_diag = torch.linalg.norm(torch.max(xyz_init, dim=0).values - torch.min(xyz_init, dim=0).values).clamp_min(1e-6)
    max_displacement = (
        float(args.max_displacement_abs)
        if float(args.max_displacement_abs) > 0.0
        else float(args.max_displacement_ratio) * float(bbox_diag.item())
    )
    configure_trainable_params(student, train_opacity=train_opacity, train_scale=train_scale)
    optimizer = build_optimizer(
        student,
        xyz_lr=float(args.xyz_lr),
        opacity_lr=float(args.opacity_lr),
        scale_lr=float(args.scale_lr),
        train_opacity=train_opacity,
        train_scale=train_scale,
    )
    total_guard_pruned = 0
    total_guard_reset = 0
    depth_feedback_sum = torch.zeros((int(student.get_xyz.shape[0]),), dtype=torch.float32, device="cuda")
    depth_feedback_count = torch.zeros((int(student.get_xyz.shape[0]),), dtype=torch.int32, device="cuda")
    depth_dropout_until = torch.zeros((int(student.get_xyz.shape[0]),), dtype=torch.int64, device="cuda")
    total_depth_feedback_dropouts = 0
    total_depth_feedback_resets = 0
    total_depth_feedback_pruned = 0
    total_depth_feedback_relocates = 0
    total_depth_feedback_reclones = 0

    before_losses = evaluate_surface_losses(
        selected_cameras,
        teacher_cache,
        student,
        background,
        min_pixels=float(args.min_loss_pixels),
        depth_relative_min=float(args.depth_relative_min),
    )
    before_mip_closure_losses = evaluate_mip_closure_losses(
        selected_cameras,
        teacher_cache,
        student,
        background,
        splat_args=mip_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.min_loss_pixels),
        depth_relative_min=float(args.depth_relative_min),
        charbonnier_eps=float(args.charbonnier_eps),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    )
    before_obs_closure_losses = evaluate_mip_closure_losses(
        obs_closure_cameras,
        obs_closure_cache,
        student,
        background,
        splat_args=mip_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.min_loss_pixels),
        depth_relative_min=float(args.depth_relative_min),
        charbonnier_eps=float(args.charbonnier_eps),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    ) if obs_closure_cache else {"alpha": 0.0, "premul": 0.0, "depth": 0.0}
    before_sr_losses = evaluate_sr_prior_losses(
        sr_cameras,
        sr_cache,
        student,
        background,
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
        disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
    )

    progress = tqdm(range(1, int(args.iterations) + 1), desc="mip->SOF surface")
    log_rows = []
    for iteration in progress:
        view_idx = randint(0, len(selected_cameras) - 1)
        camera = selected_cameras[view_idx]
        target = teacher_cache[view_idx]
        render_pkg = render_simple(camera, student, background)
        rgb = render_pkg["render"].clamp(0.0, 1.0)
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"].clamp(0.0, 1.0)
        normal = normalize_normal(render_pkg["normal"])
        mask = target["surface_mask"]
        surface_weight = target["surface_weight"]
        active_depth_dropout_mask = depth_dropout_until > int(iteration) if depth_feedback_mode == "dropout" else None
        active_depth_dropout_count = (
            int(active_depth_dropout_mask.sum().item())
            if active_depth_dropout_mask is not None
            else 0
        )

        loss = torch.zeros((), dtype=torch.float32, device="cuda")
        loss_rgb = torch.mean(torch.abs(rgb - target["base_rgb"]))
        loss = loss + float(args.lambda_rgb_preserve) * loss_rgb

        depth_rel = (depth - target["target_depth"]) / torch.clamp(target["target_depth"], min=float(args.depth_relative_min))
        depth_loss = masked_weighted_mean(
            charbonnier(depth_rel, float(args.charbonnier_eps)),
            mask,
            float(args.min_loss_pixels),
            surface_weight,
        )
        if depth_loss is None:
            depth_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        loss = loss + float(args.lambda_depth) * depth_loss

        dot = torch.sum(normal * target["target_normal"], dim=0).clamp(-1.0, 1.0)
        normal_loss = masked_weighted_mean(1.0 - dot, mask, float(args.min_loss_pixels), surface_weight)
        if normal_loss is None:
            normal_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        loss = loss + float(args.lambda_normal) * normal_loss

        depth_prior_scale = scheduled_loss_scale(
            iteration,
            start_iter=int(args.depth_prior_warmup_start_iter),
            end_iter=int(args.depth_prior_warmup_end_iter),
            start_scale=float(args.depth_prior_start_scale),
            end_scale=float(args.depth_prior_end_scale),
            update_scale=float(args.depth_prior_update_scale),
            mode=str(args.depth_prior_schedule_mode),
        )
        depth_prior_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior) > 0.0 and target.get("depth_prior_target") is not None:
            prior_depth_rel = (depth - target["depth_prior_target"]) / torch.clamp(
                target["depth_prior_target"],
                min=float(args.depth_relative_min),
            )
            maybe_depth_prior_loss = masked_weighted_mean(
                charbonnier(prior_depth_rel, float(args.charbonnier_eps)),
                target["depth_prior_mask"],
                float(args.min_loss_pixels),
                target["depth_prior_weight"],
            )
            if maybe_depth_prior_loss is not None:
                depth_prior_loss = maybe_depth_prior_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior) * depth_prior_loss

        depth_prior_normal_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior_normal) > 0.0 and target.get("depth_prior_normal") is not None:
            prior_normal_dot = torch.sum(normal * target["depth_prior_normal"], dim=0).clamp(-1.0, 1.0)
            maybe_prior_normal_loss = masked_weighted_mean(
                1.0 - prior_normal_dot,
                target["depth_prior_mask"],
                float(args.min_loss_pixels),
                target["depth_prior_weight"],
            )
            if maybe_prior_normal_loss is not None:
                depth_prior_normal_loss = maybe_prior_normal_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_normal) * depth_prior_normal_loss

        depth_prior_distortion_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior_distortion) > 0.0 and target.get("depth_prior_target") is not None:
            maybe_prior_distortion_loss = compute_depth_prior_distortion_loss(
                render_pkg,
                target,
                min_pixels=float(args.min_loss_pixels),
            )
            if maybe_prior_distortion_loss is not None:
                depth_prior_distortion_loss = maybe_prior_distortion_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_distortion) * depth_prior_distortion_loss

        depth_prior_self_normal_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.lambda_depth_prior_self_normal) > 0.0 and target.get("depth_prior_target") is not None:
            maybe_prior_self_normal_loss = compute_depth_prior_self_normal_loss(
                camera,
                render_pkg,
                target,
                min_pixels=float(args.min_loss_pixels),
            )
            if maybe_prior_self_normal_loss is not None:
                depth_prior_self_normal_loss = maybe_prior_self_normal_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_self_normal) * depth_prior_self_normal_loss
        depth_feedback_value = build_depth_feedback_value(
            depth_prior_scale=float(depth_prior_scale),
            lambda_depth_prior=float(args.lambda_depth_prior),
            lambda_depth_prior_normal=float(args.lambda_depth_prior_normal),
            lambda_depth_prior_distortion=float(args.lambda_depth_prior_distortion),
            lambda_depth_prior_self_normal=float(args.lambda_depth_prior_self_normal),
            depth_prior_loss=depth_prior_loss,
            depth_prior_normal_loss=depth_prior_normal_loss,
            depth_prior_distortion_loss=depth_prior_distortion_loss,
            depth_prior_self_normal_loss=depth_prior_self_normal_loss,
        )
        if (
            depth_feedback_mode != "none"
            and int(iteration) >= int(args.depth_feedback_start_iter)
            and float(depth_feedback_value) > 0.0
        ):
            visible_feedback_mask = render_pkg["visibility_filter"].reshape(-1).to(device="cuda", dtype=torch.bool)
            if bool(int(args.depth_feedback_only_update_mask)) and external_update_mask is not None:
                visible_feedback_mask &= external_update_mask
            if active_depth_dropout_mask is not None:
                visible_feedback_mask &= ~active_depth_dropout_mask
            if torch.any(visible_feedback_mask):
                depth_feedback_sum[visible_feedback_mask] += float(depth_feedback_value)
                depth_feedback_count[visible_feedback_mask] += 1

        sr_l1_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        sr_hf_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        mip_closure_alpha_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        mip_closure_premul_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        mip_closure_depth_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        mip_obs_closure_alpha_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        mip_obs_closure_premul_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        mip_obs_closure_depth_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        sr_prior_scale = scheduled_sr_prior_scale(
            iteration,
            start_iter=int(args.sr_prior_warmup_start_iter),
            end_iter=int(args.sr_prior_warmup_end_iter),
            start_scale=float(args.sr_prior_start_scale),
            end_scale=float(args.sr_prior_end_scale),
            update_scale=float(args.sr_prior_update_scale),
            mode=str(args.sr_prior_schedule_mode),
        )
        if sr_cache and (float(args.sr_prior_l1_weight) > 0.0 or float(args.sr_prior_hf_weight) > 0.0):
            sr_idx = randint(0, len(sr_cache) - 1)
            sr_camera = sr_cameras[sr_idx]
            sr_target = sr_cache[sr_idx]
            sr_render_pkg = render_simple(sr_camera, student, background)
            sr_rgb = sr_render_pkg["render"].clamp(0.0, 1.0)
            maybe_sr_l1, maybe_sr_hf = compute_prepared_sr_losses(
                sr_rgb,
                sr_target,
                min_pixels=float(args.sr_prior_min_pixels),
                min_valid_ratio=float(args.sr_prior_min_valid_ratio),
                prior_delta_clip=float(args.sr_prior_delta_clip),
                disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
            )
            if maybe_sr_l1 is not None:
                sr_l1_loss = maybe_sr_l1
                loss = loss + sr_prior_scale * float(args.sr_prior_l1_weight) * sr_l1_loss
            if maybe_sr_hf is not None:
                sr_hf_loss = maybe_sr_hf
                loss = loss + sr_prior_scale * float(args.sr_prior_hf_weight) * sr_hf_loss

        if (
            float(args.lambda_mip_closure_alpha) > 0.0
            or float(args.lambda_mip_closure_premul) > 0.0
            or float(args.lambda_mip_closure_depth) > 0.0
        ):
            closure_pkg = render_simple(camera, student, background, splat_args=mip_splat_args)
            maybe_closure_alpha_loss, maybe_closure_premul_loss, maybe_closure_depth_loss = compute_mip_closure_losses(
                closure_pkg,
                target,
                kernel_size=int(args.mip_closure_kernel),
                alpha_threshold=float(args.mip_closure_alpha_threshold),
                min_pixels=float(args.min_loss_pixels),
                depth_relative_min=float(args.depth_relative_min),
                charbonnier_eps=float(args.charbonnier_eps),
                reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
            )
            if maybe_closure_alpha_loss is not None:
                mip_closure_alpha_loss = maybe_closure_alpha_loss
                loss = loss + float(args.lambda_mip_closure_alpha) * mip_closure_alpha_loss
            if maybe_closure_premul_loss is not None:
                mip_closure_premul_loss = maybe_closure_premul_loss
                loss = loss + float(args.lambda_mip_closure_premul) * mip_closure_premul_loss
            if maybe_closure_depth_loss is not None:
                mip_closure_depth_loss = maybe_closure_depth_loss
                loss = loss + float(args.lambda_mip_closure_depth) * mip_closure_depth_loss

        if obs_closure_cache and (
            float(args.lambda_mip_obs_closure_alpha) > 0.0
            or float(args.lambda_mip_obs_closure_premul) > 0.0
            or float(args.lambda_mip_obs_closure_depth) > 0.0
        ):
            obs_idx = randint(0, len(obs_closure_cache) - 1)
            obs_camera = obs_closure_cameras[obs_idx]
            obs_target = obs_closure_cache[obs_idx]
            obs_pkg = render_simple(obs_camera, student, background, splat_args=mip_splat_args)
            (
                maybe_obs_alpha_loss,
                maybe_obs_premul_loss,
                maybe_obs_depth_loss,
            ) = compute_mip_closure_losses(
                obs_pkg,
                obs_target,
                kernel_size=int(args.mip_closure_kernel),
                alpha_threshold=float(args.mip_closure_alpha_threshold),
                min_pixels=float(args.min_loss_pixels),
                depth_relative_min=float(args.depth_relative_min),
                charbonnier_eps=float(args.charbonnier_eps),
                reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
            )
            if maybe_obs_alpha_loss is not None:
                mip_obs_closure_alpha_loss = maybe_obs_alpha_loss
                loss = loss + float(args.lambda_mip_obs_closure_alpha) * mip_obs_closure_alpha_loss
            if maybe_obs_premul_loss is not None:
                mip_obs_closure_premul_loss = maybe_obs_premul_loss
                loss = loss + float(args.lambda_mip_obs_closure_premul) * mip_obs_closure_premul_loss
            if maybe_obs_depth_loss is not None:
                mip_obs_closure_depth_loss = maybe_obs_depth_loss
                loss = loss + float(args.lambda_mip_obs_closure_depth) * mip_obs_closure_depth_loss

        alpha_loss = torch.mean(torch.abs(alpha - target["target_alpha"]))
        loss = loss + float(args.lambda_alpha) * alpha_loss

        opacity_anchor_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        scale_anchor_loss = torch.zeros((), dtype=torch.float32, device="cuda")
        if train_opacity:
            current_opacity = student.get_opacity
            opacity_anchor_loss = torch.mean(torch.abs(current_opacity - opacity_init))
            loss = loss + float(args.lambda_opacity_anchor) * opacity_anchor_loss
        if train_scale:
            current_scale = student.get_scaling
            rel_scale = (current_scale - scale_init) / torch.clamp(scale_init, min=1e-8)
            scale_anchor_loss = torch.mean(rel_scale * rel_scale)
            loss = loss + float(args.lambda_scale_anchor) * scale_anchor_loss

        delta = (student._xyz - xyz_init) / bbox_diag
        anchor_loss = torch.mean(delta * delta)
        loss = loss + float(args.lambda_anchor) * anchor_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_xyz = grad_l2_norm(student._xyz)
        grad_opacity = grad_l2_norm(student._opacity) if train_opacity else 0.0
        grad_scale = grad_l2_norm(student._scaling) if train_scale else 0.0
        guard_pruned = 0
        guard_reset = 0
        guard_flagged = 0
        guard_stats = {"xyz_max": 0.0, "opacity_max": 0.0, "scale_max": 0.0, "nonfinite": 0.0, "flagged": 0.0}
        prune_mask = None
        if active_depth_dropout_mask is not None and active_depth_dropout_count > 0:
            zero_optimizer_entries(optimizer, active_depth_dropout_mask)
        if gradient_guard_mode != "none" and int(iteration) >= int(args.gradient_guard_start_iter):
            guard_mask, guard_severity, guard_stats = build_gradient_guard_mask(
                student,
                train_opacity=train_opacity,
                train_scale=train_scale,
                update_mask=external_update_mask,
                only_update_mask=bool(int(args.gradient_guard_only_update_mask)),
                xyz_threshold=float(args.gradient_guard_xyz_threshold),
                opacity_threshold=float(args.gradient_guard_opacity_threshold),
                scale_threshold=float(args.gradient_guard_scale_threshold),
                max_fraction=float(args.gradient_guard_max_fraction),
                max_points=int(args.gradient_guard_max_points),
            )
            if gradient_guard_mode == "prune" and int(guard_mask.sum().item()) >= int(student.get_xyz.shape[0]) and int(student.get_xyz.shape[0]) > 1:
                keep_id = torch.argmin(torch.where(guard_mask, guard_severity, torch.full_like(guard_severity, 1e6)))
                guard_mask[keep_id] = False
            guard_flagged = int(guard_mask.sum().item())
            if guard_flagged > 0:
                zero_optimizer_entries(optimizer, guard_mask)
                if gradient_guard_mode == "reset":
                    restore_gaussian_entries(
                        student,
                        reset_mask=guard_mask,
                        xyz_init=xyz_init,
                        opacity_raw_init=opacity_raw_init,
                        scale_raw_init=scale_raw_init,
                        train_opacity=train_opacity,
                        train_scale=train_scale,
                    )
                    guard_reset = guard_flagged
                    total_guard_reset += guard_reset
                elif gradient_guard_mode == "prune":
                    prune_mask = guard_mask
        apply_gaussian_update_mask(
            optimizer,
            total_gaussians=int(student.get_xyz.shape[0]),
            update_mask=external_update_mask,
            update_scale=float(args.gaussian_update_scale),
            scale_param_mask=scale_param_mask,
        )
        optimizer.step()
        if prune_mask is not None and int(prune_mask.sum().item()) > 0:
            valid_mask, _, prune_count = prune_student_with_optimizer(student, optimizer, prune_mask)
            if prune_count > 0:
                xyz_init = xyz_init[valid_mask]
                opacity_init = opacity_init[valid_mask]
                opacity_raw_init = opacity_raw_init[valid_mask]
                scale_init = scale_init[valid_mask]
                scale_raw_init = scale_raw_init[valid_mask]
                if external_update_mask is not None:
                    external_update_mask = external_update_mask[valid_mask]
                if scale_param_mask is not None:
                    scale_param_mask = scale_param_mask[valid_mask]
                depth_feedback_sum = depth_feedback_sum[valid_mask]
                depth_feedback_count = depth_feedback_count[valid_mask]
                depth_dropout_until = depth_dropout_until[valid_mask]
                guard_pruned = prune_count
                total_guard_pruned += prune_count
        depth_feedback_triggered = 0
        depth_feedback_stats = {
            "candidate_count": 0.0,
            "selected_count": 0.0,
            "score_mean": 0.0,
            "score_max": 0.0,
            "visible_mean": 0.0,
        }
        if (
            depth_feedback_mode != "none"
            and int(args.depth_feedback_interval) > 0
            and int(iteration) >= int(args.depth_feedback_start_iter)
            and iteration % int(args.depth_feedback_interval) == 0
        ):
            feedback_candidate_mask = (
                external_update_mask
                if bool(int(args.depth_feedback_only_update_mask)) and external_update_mask is not None
                else None
            )
            feedback_exclude_mask = depth_dropout_until > int(iteration) if depth_feedback_mode == "dropout" else None
            feedback_mask, depth_feedback_stats = select_depth_feedback_nominees(
                depth_feedback_sum,
                depth_feedback_count,
                iteration=int(iteration),
                min_visible=int(args.depth_feedback_min_visible),
                min_score=float(args.depth_feedback_min_score),
                top_fraction=float(args.depth_feedback_top_fraction),
                top_points=int(args.depth_feedback_top_points),
                candidate_mask=feedback_candidate_mask,
                exclude_mask=feedback_exclude_mask,
            )
            depth_feedback_triggered = int(feedback_mask.sum().item())
            if depth_feedback_triggered > 0:
                zero_optimizer_entries(optimizer, feedback_mask)
                if depth_feedback_mode == "dropout":
                    depth_dropout_until[feedback_mask] = int(iteration) + int(args.depth_feedback_dropout_iters)
                    total_depth_feedback_dropouts += depth_feedback_triggered
                elif depth_feedback_mode == "reset":
                    restore_gaussian_entries(
                        student,
                        reset_mask=feedback_mask,
                        xyz_init=xyz_init,
                        opacity_raw_init=opacity_raw_init,
                        scale_raw_init=scale_raw_init,
                        train_opacity=train_opacity,
                        train_scale=train_scale,
                    )
                    total_depth_feedback_resets += depth_feedback_triggered
                elif depth_feedback_mode == "prune":
                    valid_mask, _, prune_count = prune_student_with_optimizer(student, optimizer, feedback_mask)
                    if prune_count > 0:
                        xyz_init = xyz_init[valid_mask]
                        opacity_init = opacity_init[valid_mask]
                        opacity_raw_init = opacity_raw_init[valid_mask]
                        scale_init = scale_init[valid_mask]
                        scale_raw_init = scale_raw_init[valid_mask]
                        if external_update_mask is not None:
                            external_update_mask = external_update_mask[valid_mask]
                        if scale_param_mask is not None:
                            scale_param_mask = scale_param_mask[valid_mask]
                        depth_feedback_sum = depth_feedback_sum[valid_mask]
                        depth_feedback_count = depth_feedback_count[valid_mask]
                        depth_dropout_until = depth_dropout_until[valid_mask]
                        depth_feedback_triggered = prune_count
                        total_depth_feedback_pruned += prune_count
                elif depth_feedback_mode in ("relocate", "reclone"):
                    moved = apply_depth_feedback_reinit(
                        student,
                        optimizer,
                        feedback_mask,
                        score_sum=depth_feedback_sum,
                        score_count=depth_feedback_count,
                        mode=depth_feedback_mode,
                    )
                    depth_feedback_triggered = moved
                    if depth_feedback_mode == "relocate":
                        total_depth_feedback_relocates += moved
                    else:
                        total_depth_feedback_reclones += moved
            depth_feedback_sum.zero_()
            depth_feedback_count.zero_()
        clamp_xyz_displacement(student, xyz_init, max_displacement=max_displacement)
        if train_scale:
            clamp_scale_update_range(
                student,
                scale_raw_init=scale_raw_init,
                scale_param_mask=scale_param_mask,
                min_multiplier=float(args.gaussian_scale_min_multiplier),
                max_multiplier=float(args.gaussian_scale_max_multiplier),
            )

        row = {
            "iter": int(iteration),
            "loss": float(loss.detach().item()),
            "rgb": float(loss_rgb.detach().item()),
            "depth": float(depth_loss.detach().item()),
            "normal": float(normal_loss.detach().item()),
            "depth_prior": float(depth_prior_loss.detach().item()),
            "depth_prior_normal": float(depth_prior_normal_loss.detach().item()),
            "depth_prior_distortion": float(depth_prior_distortion_loss.detach().item()),
            "depth_prior_self_normal": float(depth_prior_self_normal_loss.detach().item()),
            "depth_prior_scale": float(depth_prior_scale),
            "depth_feedback_value": float(depth_feedback_value),
            "depth_feedback_active_dropout": int(active_depth_dropout_count),
            "depth_feedback_triggered": int(depth_feedback_triggered),
            "depth_feedback_score_max": float(depth_feedback_stats["score_max"]),
            "sr_l1": float(sr_l1_loss.detach().item()),
            "sr_hf": float(sr_hf_loss.detach().item()),
            "sr_prior_scale": float(sr_prior_scale),
            "mip_closure_alpha": float(mip_closure_alpha_loss.detach().item()),
            "mip_closure_premul": float(mip_closure_premul_loss.detach().item()),
            "mip_closure_depth": float(mip_closure_depth_loss.detach().item()),
            "mip_obs_closure_alpha": float(mip_obs_closure_alpha_loss.detach().item()),
            "mip_obs_closure_premul": float(mip_obs_closure_premul_loss.detach().item()),
            "mip_obs_closure_depth": float(mip_obs_closure_depth_loss.detach().item()),
            "alpha": float(alpha_loss.detach().item()),
            "opacity_anchor": float(opacity_anchor_loss.detach().item()),
            "scale_anchor": float(scale_anchor_loss.detach().item()),
            "anchor": float(anchor_loss.detach().item()),
            "grad_xyz": float(grad_xyz),
            "grad_opacity": float(grad_opacity),
            "grad_scale": float(grad_scale),
            "guard_flagged": int(guard_flagged),
            "guard_pruned": int(guard_pruned),
            "guard_reset": int(guard_reset),
            "guard_nonfinite": int(guard_stats["nonfinite"]),
            "guard_xyz_max": float(guard_stats["xyz_max"]),
            "guard_opacity_max": float(guard_stats["opacity_max"]),
            "guard_scale_max": float(guard_stats["scale_max"]),
            "num_gaussians": int(student.get_xyz.shape[0]),
        }
        log_rows.append(row)
        if iteration % 10 == 0:
            progress.set_postfix(
                {
                    "loss": f"{row['loss']:.5f}",
                    "rgb": f"{row['rgb']:.5f}",
                    "depth": f"{row['depth']:.5f}",
                    "dprior": f"{row['depth_prior']:.5f}/{row['depth_prior_distortion']:.5f}",
                    "dps": f"{row['depth_prior_scale']:.2f}",
                    "dfb": f"{row['depth_feedback_value']:.4f}/{row['depth_feedback_triggered']}",
                    "sr": f"{row['sr_l1']:.5f}/{row['sr_hf']:.5f}",
                    "srs": f"{row['sr_prior_scale']:.2f}",
                    "mcl": f"{row['mip_closure_alpha']:.4f}/{row['mip_closure_premul']:.4f}",
                    "obs": f"{row['mip_obs_closure_alpha']:.4f}/{row['mip_obs_closure_premul']:.4f}",
                    "ops": f"{row['opacity_anchor']:.4f}/{row['scale_anchor']:.4f}",
                    "guard": f"{row['guard_flagged']}/{row['guard_pruned']}/{row['guard_reset']}",
                    "anchor": f"{row['anchor']:.6f}",
                }
            )

        if int(args.save_every) > 0 and iteration % int(args.save_every) == 0:
            point_dir = output_model_path / "point_cloud" / f"iteration_{iteration}"
            mkdir_p(str(point_dir))
            student.save_ply(str(point_dir / "point_cloud.ply"))
            student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    student.compute_3D_filter(student_filter_cameras, CUDA=False)
    copy_render_config(mip_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
    mkdir_p(str(point_dir))
    student.save_ply(str(point_dir / "point_cloud.ply"))
    student.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    after_losses = evaluate_surface_losses(
        selected_cameras,
        teacher_cache,
        student,
        background,
        min_pixels=float(args.min_loss_pixels),
        depth_relative_min=float(args.depth_relative_min),
    )
    after_mip_closure_losses = evaluate_mip_closure_losses(
        selected_cameras,
        teacher_cache,
        student,
        background,
        splat_args=mip_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.min_loss_pixels),
        depth_relative_min=float(args.depth_relative_min),
        charbonnier_eps=float(args.charbonnier_eps),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    )
    after_obs_closure_losses = evaluate_mip_closure_losses(
        obs_closure_cameras,
        obs_closure_cache,
        student,
        background,
        splat_args=mip_splat_args,
        kernel_size=int(args.mip_closure_kernel),
        alpha_threshold=float(args.mip_closure_alpha_threshold),
        min_pixels=float(args.min_loss_pixels),
        depth_relative_min=float(args.depth_relative_min),
        charbonnier_eps=float(args.charbonnier_eps),
        reference_lowpass=bool(int(args.mip_closure_reference_lowpass)),
    ) if obs_closure_cache else {"alpha": 0.0, "premul": 0.0, "depth": 0.0}
    after_sr_losses = evaluate_sr_prior_losses(
        sr_cameras,
        sr_cache,
        student,
        background,
        min_pixels=float(args.sr_prior_min_pixels),
        min_valid_ratio=float(args.sr_prior_min_valid_ratio),
        prior_delta_clip=float(args.sr_prior_delta_clip),
        disable_hf_residual=bool(args.disable_sr_prior_hf_residual),
    )
    displacement = torch.linalg.norm(student._xyz.detach() - xyz_init, dim=1)
    summary = {
        "version": "mip_to_sof_surface_v0",
        "scene_root": str(scene_root),
        "mip_model_path": str(mip_model_path),
        "sof_surface_model_path": str(sof_model_path),
        "output_model_path": str(output_model_path),
        "mip_iteration": int(mip_iter),
        "sof_iteration": int(sof_iter),
        "output_iteration": int(args.output_iteration),
        "images_subdir": str(args.images_subdir),
        "selected_views": [str(cam.image_name) for cam in selected_cameras],
        "mip_obs_closure_selected_views": [str(cam.image_name) for cam in obs_closure_cameras],
        "filter_images_subdir": str(filter_images_subdir) if filter_images_subdir else None,
        "filter_split": str(args.filter_split),
        "filter_views": [str(cam.image_name) for cam in student_filter_cameras],
        "iterations": int(args.iterations),
        "xyz_lr": float(args.xyz_lr),
        "opacity_lr": float(args.opacity_lr),
        "scale_lr": float(args.scale_lr),
        "train_opacity": bool(train_opacity),
        "train_scale": bool(train_scale),
        "max_displacement": float(max_displacement),
        "bbox_diag": float(bbox_diag.item()),
        "loss_weights": {
            "rgb_preserve": float(args.lambda_rgb_preserve),
            "depth": float(args.lambda_depth),
            "normal": float(args.lambda_normal),
            "mip_closure_alpha": float(args.lambda_mip_closure_alpha),
            "mip_closure_premul": float(args.lambda_mip_closure_premul),
            "mip_closure_depth": float(args.lambda_mip_closure_depth),
            "mip_obs_closure_alpha": float(args.lambda_mip_obs_closure_alpha),
            "mip_obs_closure_premul": float(args.lambda_mip_obs_closure_premul),
            "mip_obs_closure_depth": float(args.lambda_mip_obs_closure_depth),
            "opacity_anchor": float(args.lambda_opacity_anchor),
            "scale_anchor": float(args.lambda_scale_anchor),
            "depth_prior": float(args.lambda_depth_prior),
            "depth_prior_normal": float(args.lambda_depth_prior_normal),
            "depth_prior_distortion": float(args.lambda_depth_prior_distortion),
            "depth_prior_self_normal": float(args.lambda_depth_prior_self_normal),
            "sr_prior_l1": float(args.sr_prior_l1_weight),
            "sr_prior_hf": float(args.sr_prior_hf_weight),
            "alpha": float(args.lambda_alpha),
            "anchor": float(args.lambda_anchor),
        },
        "surface_gates": {
            "min_surface_alpha": float(args.min_surface_alpha),
            "min_loss_pixels": float(args.min_loss_pixels),
            "mip_closure_kernel": int(args.mip_closure_kernel),
            "mip_closure_alpha_threshold": float(args.mip_closure_alpha_threshold),
            "mip_closure_reference_lowpass": bool(int(args.mip_closure_reference_lowpass)),
            "mip_obs_closure_images_subdir": str(obs_closure_images_subdir) if use_obs_closure else None,
            "mip_obs_closure_split": str(args.mip_obs_closure_split),
            "mip_obs_closure_views": int(len(obs_closure_cameras)),
            "depth_relative_min": float(args.depth_relative_min),
        },
        "prior_inputs": {
            "sr_prior_root": str(sr_prior_root) if sr_prior_root is not None else None,
            "sr_prior_dir": str(sr_prior_dir) if sr_prior_dir is not None else None,
            "sr_prior_mask_dir": str(sr_prior_mask_dir) if sr_prior_mask_dir is not None else None,
            "sr_anchor_dir": str(sr_anchor_dir) if sr_anchor_dir is not None else None,
            "sr_prior_subdir": str(args.sr_prior_subdir),
            "sr_prior_mask_subdir": str(args.sr_prior_mask_subdir),
            "sr_anchor_subdir": str(args.sr_anchor_subdir),
            "sr_images_subdir": str(sr_images_subdir),
            "depth_prior_root": str(depth_prior_root) if depth_prior_root is not None else None,
            "depth_prior_subdirs": list(depth_prior_subdirs),
            "depth_prior_confidence_subdirs": list(depth_prior_confidence_subdirs),
        },
        "gaussian_update_mask": {
            "payload": str(args.optimize_gaussian_mask_payload) if args.optimize_gaussian_mask_payload else None,
            "key": str(args.optimize_gaussian_mask_key),
            "selected": int(external_update_mask.sum().item()) if external_update_mask is not None else None,
            "total": int(external_update_mask.shape[0]) if external_update_mask is not None else None,
            "update_scale": float(args.gaussian_update_scale),
            "scale_axis_mode": str(args.gaussian_scale_axis_mode),
            "scale_min_multiplier": float(args.gaussian_scale_min_multiplier),
            "scale_max_multiplier": float(args.gaussian_scale_max_multiplier),
        },
        "prior_config": {
            "sr_prior_mask_floor": float(args.sr_prior_mask_floor),
            "sr_prior_consistency_threshold": float(args.sr_prior_consistency_threshold),
            "sr_prior_min_valid_ratio": float(args.sr_prior_min_valid_ratio),
            "sr_prior_min_pixels": float(args.sr_prior_min_pixels),
            "sr_prior_delta_clip": float(args.sr_prior_delta_clip),
            "disable_sr_prior_hf_residual": bool(args.disable_sr_prior_hf_residual),
            "sr_prior_warmup_start_iter": int(args.sr_prior_warmup_start_iter),
            "sr_prior_warmup_end_iter": int(args.sr_prior_warmup_end_iter),
            "sr_prior_start_scale": float(args.sr_prior_start_scale),
            "sr_prior_end_scale": float(args.sr_prior_end_scale),
            "sr_prior_update_scale": float(args.sr_prior_update_scale),
            "sr_prior_schedule_mode": str(args.sr_prior_schedule_mode),
            "depth_prior_confidence_min": float(args.depth_prior_confidence_min),
            "depth_prior_agreement_threshold": float(args.depth_prior_agreement_threshold),
            "depth_prior_agreement_floor": float(args.depth_prior_agreement_floor),
            "depth_prior_align_mode": str(args.depth_prior_align_mode),
            "depth_prior_align_min_pixels": int(args.depth_prior_align_min_pixels),
            "depth_prior_surface_weight_boost": float(args.depth_prior_surface_weight_boost),
            "depth_prior_warmup_start_iter": int(args.depth_prior_warmup_start_iter),
            "depth_prior_warmup_end_iter": int(args.depth_prior_warmup_end_iter),
            "depth_prior_start_scale": float(args.depth_prior_start_scale),
            "depth_prior_end_scale": float(args.depth_prior_end_scale),
            "depth_prior_update_scale": float(args.depth_prior_update_scale),
            "depth_prior_schedule_mode": str(args.depth_prior_schedule_mode),
            "depth_prior_weight_gain": float(args.depth_prior_weight_gain),
            "depth_prior_weight_power": float(args.depth_prior_weight_power),
            "depth_prior_weight_min": float(args.depth_prior_weight_min),
            "depth_feedback_start_iter": int(args.depth_feedback_start_iter),
            "depth_feedback_interval": int(args.depth_feedback_interval),
            "depth_feedback_min_visible": int(args.depth_feedback_min_visible),
            "depth_feedback_min_score": float(args.depth_feedback_min_score),
            "depth_feedback_top_fraction": float(args.depth_feedback_top_fraction),
            "depth_feedback_top_points": int(args.depth_feedback_top_points),
            "depth_feedback_dropout_iters": int(args.depth_feedback_dropout_iters),
        },
        "gradient_guard": {
            "mode": gradient_guard_mode,
            "start_iter": int(args.gradient_guard_start_iter),
            "xyz_threshold": float(args.gradient_guard_xyz_threshold),
            "opacity_threshold": float(args.gradient_guard_opacity_threshold),
            "scale_threshold": float(args.gradient_guard_scale_threshold),
            "max_fraction": float(args.gradient_guard_max_fraction),
            "max_points": int(args.gradient_guard_max_points),
            "only_update_mask": bool(int(args.gradient_guard_only_update_mask)),
            "total_pruned": int(total_guard_pruned),
            "total_reset": int(total_guard_reset),
        },
        "depth_feedback": {
            "mode": depth_feedback_mode,
            "start_iter": int(args.depth_feedback_start_iter),
            "interval": int(args.depth_feedback_interval),
            "min_visible": int(args.depth_feedback_min_visible),
            "min_score": float(args.depth_feedback_min_score),
            "top_fraction": float(args.depth_feedback_top_fraction),
            "top_points": int(args.depth_feedback_top_points),
            "dropout_iters": int(args.depth_feedback_dropout_iters),
            "only_update_mask": bool(int(args.depth_feedback_only_update_mask)),
            "total_dropouts": int(total_depth_feedback_dropouts),
            "total_resets": int(total_depth_feedback_resets),
            "total_pruned": int(total_depth_feedback_pruned),
            "total_relocates": int(total_depth_feedback_relocates),
            "total_reclones": int(total_depth_feedback_reclones),
            "active_dropout_final": int((depth_dropout_until > int(args.iterations)).sum().item()) if depth_feedback_mode == "dropout" else 0,
        },
        "prior_cache": summarize_prior_cache(teacher_cache),
        "sr_prior_cache": summarize_sr_cache(sr_cache),
        "loss_before": before_losses,
        "loss_after": after_losses,
        "mip_closure_before": before_mip_closure_losses,
        "mip_closure_after": after_mip_closure_losses,
        "mip_obs_closure_before": before_obs_closure_losses,
        "mip_obs_closure_after": after_obs_closure_losses,
        "sr_loss_before": before_sr_losses,
        "sr_loss_after": after_sr_losses,
        "displacement": stats_from_tensor(displacement),
        "displacement_over_bbox_diag": stats_from_tensor(displacement / bbox_diag),
        "final_log": log_rows[-10:],
        "output_ply": str(point_dir / "point_cloud.ply"),
    }
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(student.get_xyz.shape[0]), **summary}, f, indent=2)
    (output_model_path / "mip_to_sof_surface_v0_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
