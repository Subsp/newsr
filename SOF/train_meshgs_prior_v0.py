import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SplattingSettings
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, safe_state
from utils.prior_injection import index_image_dir, load_mask, load_rgb_image
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


_CARRIER_PAYLOAD_REQUIRED_KEYS = [
    "centers",
    "normals",
    "tangent_u",
    "tangent_v",
    "scale_u",
    "scale_v",
    "fused_rgb",
    "confidence",
    "disagreement",
    "view_count",
    "valid_mask",
]


def resolve_carrier_payload_path(args) -> str:
    payload_path = getattr(args, "carrier_payload", None) or getattr(args, "mesh_fusion_payload", None)
    if not payload_path:
        raise ValueError("Pass --carrier_payload (preferred) or legacy --mesh_fusion_payload.")
    return str(Path(payload_path).expanduser().resolve())


def _to_numpy_payload_value(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _unwrap_carrier_payload_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    if all(key in payload for key in _CARRIER_PAYLOAD_REQUIRED_KEYS):
        return payload
    if "carrier_payload" in payload and isinstance(payload["carrier_payload"], dict):
        return payload["carrier_payload"]
    if "hrgs_outputs" in payload:
        nested = payload["hrgs_outputs"]
        if isinstance(nested, dict) and "carrier_payload" in nested and isinstance(nested["carrier_payload"], dict):
            return nested["carrier_payload"]
    raise KeyError(
        "Carrier payload file must contain payload keys directly, "
        "or a nested 'carrier_payload' dictionary."
    )


def load_carrier_payload_arrays(path: str) -> Dict[str, np.ndarray]:
    payload_path = Path(path)
    suffix = payload_path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(payload_path)
        payload = {key: loaded[key] for key in loaded.files}
    else:
        raw = torch.load(payload_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported carrier payload object type: {type(raw)!r}")
        payload = _unwrap_carrier_payload_dict(raw)
        payload = {key: _to_numpy_payload_value(value) for key, value in payload.items()}

    for key in _CARRIER_PAYLOAD_REQUIRED_KEYS:
        if key not in payload:
            raise KeyError(f"Missing '{key}' in carrier payload: {payload_path}")
    if "scale_n" not in payload:
        payload["scale_n"] = np.minimum(payload["scale_u"], payload["scale_v"]) * 0.05

    for key in ("centers", "normals", "tangent_u", "tangent_v", "fused_rgb"):
        payload[key] = np.asarray(payload[key], dtype=np.float32).reshape(-1, 3)
    for key in ("scale_u", "scale_v", "scale_n", "confidence", "disagreement", "view_count", "valid_mask"):
        payload[key] = np.asarray(payload[key]).reshape(-1)
    return payload


def resize_hw3_tensor(image_hw3: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    image = image_hw3.permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    resized = F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).detach().cpu()


def resize_mask_tensor(mask_hw: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    mask = mask_hw[None, None].to(device="cuda", dtype=torch.float32)
    resized = F.interpolate(mask, size=(target_height, target_width), mode="nearest")
    return resized[0, 0].detach().cpu()


def load_rgb_cached(image_name, index, cache, height, width):
    key = (image_name, height, width)
    if key not in cache:
        path = index.get(image_name)
        if path is None:
            cache[key] = None
        else:
            image = load_rgb_image(path)
            if image.shape[0] != height or image.shape[1] != width:
                image = resize_hw3_tensor(image, height, width)
            cache[key] = image
    image = cache[key]
    if image is None:
        return None
    return image.to(device="cuda", dtype=torch.float32)


def load_mask_cached(image_name, mask_dir, cache, height, width):
    key = (image_name, height, width)
    if key not in cache:
        path = Path(mask_dir) / f"{image_name}_inject.png"
        if not path.is_file():
            cache[key] = None
        else:
            mask = load_mask(path)
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = resize_mask_tensor(mask, height, width)
            cache[key] = mask
    mask = cache[key]
    if mask is None:
        return None
    return mask.to(device="cuda", dtype=torch.float32) > 0.5


def masked_l1(image_chw: torch.Tensor, target_hw3: torch.Tensor, mask_hw: torch.Tensor):
    mask = mask_hw.to(device=image_chw.device, dtype=image_chw.dtype)
    if float(mask.sum().item()) <= 0:
        return None
    target_chw = target_hw3.permute(2, 0, 1).to(device=image_chw.device, dtype=image_chw.dtype)
    return (torch.abs(image_chw - target_chw) * mask.unsqueeze(0)).sum() / (mask.sum() * image_chw.shape[0]).clamp_min(1.0)


def render_alpha_mask(render_pkg, image_chw: torch.Tensor, threshold: float) -> torch.Tensor:
    if "alpha" in render_pkg:
        alpha = render_pkg["alpha"][0]
        return alpha > float(threshold)
    render_tensor = render_pkg["render"]
    if render_tensor.shape[0] >= 8:
        return render_tensor[7] > float(threshold)
    return image_chw.detach().sum(dim=0) > float(threshold)


def normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def quaternion_from_rotation_matrix_np(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64, copy=False)
    q = np.empty((m.shape[0], 4), dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    positive = trace > 0.0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        q[positive, 0] = 0.25 * s
        q[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
        q[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
        q[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s

    negative = ~positive
    if np.any(negative):
        idx = np.where(negative)[0]
        mn = m[idx]
        choice = np.argmax(np.stack([mn[:, 0, 0], mn[:, 1, 1], mn[:, 2, 2]], axis=1), axis=1)
        for axis in range(3):
            local = idx[choice == axis]
            if local.size == 0:
                continue
            ml = m[local]
            if axis == 0:
                s = np.sqrt(1.0 + ml[:, 0, 0] - ml[:, 1, 1] - ml[:, 2, 2]) * 2.0
                q[local, 0] = (ml[:, 2, 1] - ml[:, 1, 2]) / s
                q[local, 1] = 0.25 * s
                q[local, 2] = (ml[:, 0, 1] + ml[:, 1, 0]) / s
                q[local, 3] = (ml[:, 0, 2] + ml[:, 2, 0]) / s
            elif axis == 1:
                s = np.sqrt(1.0 + ml[:, 1, 1] - ml[:, 0, 0] - ml[:, 2, 2]) * 2.0
                q[local, 0] = (ml[:, 0, 2] - ml[:, 2, 0]) / s
                q[local, 1] = (ml[:, 0, 1] + ml[:, 1, 0]) / s
                q[local, 2] = 0.25 * s
                q[local, 3] = (ml[:, 1, 2] + ml[:, 2, 1]) / s
            else:
                s = np.sqrt(1.0 + ml[:, 2, 2] - ml[:, 0, 0] - ml[:, 1, 1]) * 2.0
                q[local, 0] = (ml[:, 1, 0] - ml[:, 0, 1]) / s
                q[local, 1] = (ml[:, 0, 2] + ml[:, 2, 0]) / s
                q[local, 2] = (ml[:, 1, 2] + ml[:, 2, 1]) / s
                q[local, 3] = 0.25 * s

    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def init_meshgs_from_payload(args, sh_degree: int) -> GaussianModel:
    payload = load_carrier_payload_arrays(resolve_carrier_payload_path(args))
    valid = payload["valid_mask"].astype(bool)
    valid &= payload["confidence"] >= float(args.meshgs_min_confidence)
    valid &= payload["disagreement"] <= float(args.meshgs_max_disagreement)
    valid &= payload["view_count"] >= int(args.meshgs_min_views)
    indices = np.flatnonzero(valid)
    if args.meshgs_max_count > 0 and indices.size > int(args.meshgs_max_count):
        rng = np.random.default_rng(int(args.meshgs_seed))
        indices = np.sort(rng.choice(indices, size=int(args.meshgs_max_count), replace=False))
    if indices.size == 0:
        raise RuntimeError("No valid meshGS carriers after filtering. Relax confidence/disagreement/view thresholds.")

    centers = payload["centers"][indices].astype(np.float32)
    colors = np.clip(payload["fused_rgb"][indices].astype(np.float32), 0.0, 1.0)
    normals = normalize_np(payload["normals"][indices].astype(np.float32))
    tangent_u = normalize_np(payload["tangent_u"][indices].astype(np.float32))
    tangent_v = normalize_np(np.cross(normals, tangent_u))
    tangent_u = normalize_np(np.cross(tangent_v, normals))

    rotation_matrix = np.stack([tangent_u, tangent_v, normals], axis=2)
    rotations = quaternion_from_rotation_matrix_np(rotation_matrix)

    scales = np.stack(
        [
            payload["scale_u"][indices],
            payload["scale_v"][indices],
            payload["scale_n"][indices],
        ],
        axis=1,
    ).astype(np.float32)
    scales *= float(args.meshgs_scale_multiplier)
    scales[:, 2] *= float(args.meshgs_thickness_multiplier)
    scales = np.clip(scales, float(args.meshgs_min_scale), None)

    confidence = np.clip(payload["confidence"][indices].astype(np.float32), 0.0, 1.0)
    disagreement = payload["disagreement"][indices].astype(np.float32)
    if args.meshgs_max_disagreement > 0:
        disagreement_gate = 1.0 - np.clip(disagreement / float(args.meshgs_max_disagreement), 0.0, 1.0)
    else:
        disagreement_gate = np.ones_like(confidence)
    opacity = np.clip(float(args.meshgs_init_opacity) * (0.25 + 0.75 * confidence * disagreement_gate), 1e-4, 0.95)

    meshgs = GaussianModel(sh_degree, use_SBs=False)
    fused_color = RGB2SH(torch.from_numpy(colors).float().cuda())
    features = torch.zeros((indices.size, 3, (sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
    features[:, :3, 0] = fused_color

    meshgs.spatial_lr_scale = 1.0
    meshgs._xyz = nn.Parameter(torch.from_numpy(centers).float().cuda(), requires_grad=False)
    meshgs._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    meshgs._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous(), requires_grad=False)
    meshgs._opacity = nn.Parameter(inverse_sigmoid(torch.from_numpy(opacity[:, None]).float().cuda()).requires_grad_(True))
    meshgs._scaling = nn.Parameter(torch.log(torch.from_numpy(scales).float().cuda()), requires_grad=False)
    meshgs._rotation = nn.Parameter(torch.from_numpy(rotations).float().cuda(), requires_grad=False)
    meshgs.filter_3D = torch.zeros((indices.size, 1), dtype=torch.float32, device="cuda")
    meshgs.max_radii2D = torch.zeros((indices.size,), dtype=torch.float32, device="cuda")
    meshgs.init_tracking_state(indices.size)
    meshgs.active_sh_degree = 0
    return meshgs


def save_meshgs(meshgs: GaussianModel, output_dir: Path, iteration: int, args):
    point_dir = output_dir / "point_cloud" / f"iteration_{iteration}"
    mkdir_p(str(point_dir))
    meshgs.save_ply(str(point_dir / "point_cloud.ply"))
    meshgs.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(meshgs.get_xyz.shape[0])}, f, indent=2)
    with open(output_dir / "meshgs_prior_v0_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)


def train_meshgs(dataset, pipe, splat_args, args):
    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, dummy_gaussians, shuffle=True, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras().copy()

    meshgs = init_meshgs_from_payload(args, dataset.sh_degree)
    print(f"[meshGS prior] initialized {meshgs.get_xyz.shape[0]} mesh-bounded Gaussians")
    optimizer = torch.optim.Adam(
        [
            {"params": [meshgs._features_dc], "lr": float(args.meshgs_feature_lr), "name": "f_dc"},
            {"params": [meshgs._opacity], "lr": float(args.meshgs_opacity_lr), "name": "opacity"},
        ],
        lr=0.0,
        eps=1e-15,
    )

    prior_index = index_image_dir(args.prior_dir)
    prior_cache = {}
    mask_cache = {}
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    progress = tqdm(range(1, int(args.iterations) + 1), desc="Training meshGS prior")
    ema_loss = 0.0

    for iteration in progress:
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        view = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(view, meshgs, pipe, background, splat_args=splat_args)
        image = torch.clamp(render_pkg["render"][:3], 0.0, 1.0)
        height, width = image.shape[1], image.shape[2]
        prior_image = load_rgb_cached(view.image_name, prior_index, prior_cache, height, width)
        if args.mesh_fusion_mask_dir:
            mask = load_mask_cached(view.image_name, args.mesh_fusion_mask_dir, mask_cache, height, width)
        else:
            mask = render_alpha_mask(render_pkg, image, float(args.meshgs_render_alpha_threshold))
        if prior_image is None or mask is None or float(mask.sum().item()) < float(args.meshgs_min_pixels):
            continue

        loss_prior = masked_l1(image, prior_image, mask)
        if loss_prior is None:
            continue
        opacity = meshgs.get_opacity
        loss = loss_prior
        if args.meshgs_lambda_opacity > 0:
            loss = loss + float(args.meshgs_lambda_opacity) * opacity.mean()
        if args.meshgs_lambda_opacity_confidence > 0:
            loss = loss + float(args.meshgs_lambda_opacity_confidence) * torch.mean((opacity - opacity.detach().clamp_min(0.05)) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
        if iteration % 10 == 0:
            progress.set_postfix({"loss": f"{ema_loss:.6f}", "prior": f"{float(loss_prior.item()):.6f}"})
        if iteration in args.save_iterations or iteration == int(args.iterations):
            save_meshgs(meshgs, output_dir, iteration, args)

    save_meshgs(meshgs, output_dir, int(args.iterations), args)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train mesh-bounded GS from fused prior carriers.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--carrier_payload", type=str, default=None)
    parser.add_argument("--mesh_fusion_payload", type=str, default=None)
    parser.add_argument("--mesh_fusion_mask_dir", type=str, default=None)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--meshgs_min_confidence", type=float, default=0.05)
    parser.add_argument("--meshgs_max_disagreement", type=float, default=0.08)
    parser.add_argument("--meshgs_min_views", type=int, default=2)
    parser.add_argument("--meshgs_max_count", type=int, default=0)
    parser.add_argument("--meshgs_seed", type=int, default=0)
    parser.add_argument("--meshgs_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--meshgs_thickness_multiplier", type=float, default=0.5)
    parser.add_argument("--meshgs_min_scale", type=float, default=1e-5)
    parser.add_argument("--meshgs_init_opacity", type=float, default=0.35)
    parser.add_argument("--meshgs_feature_lr", type=float, default=0.01)
    parser.add_argument("--meshgs_opacity_lr", type=float, default=0.02)
    parser.add_argument("--meshgs_lambda_opacity", type=float, default=1e-4)
    parser.add_argument("--meshgs_lambda_opacity_confidence", type=float, default=0.0)
    parser.add_argument("--meshgs_min_pixels", type=float, default=64.0)
    parser.add_argument("--meshgs_render_alpha_threshold", type=float, default=1e-4)
    args = parser.parse_args(sys.argv[1:])
    args.mesh_fusion_payload = resolve_carrier_payload_path(args)
    args.save_iterations.append(args.iterations)
    safe_state(args.quiet)
    train_meshgs(model.extract(args), pipeline.extract(args), ss.get_settings(args), args)
