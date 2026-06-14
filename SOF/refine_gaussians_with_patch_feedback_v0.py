import copy
import json
import math
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from arguments import (
    MeshingParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    SplattingSettings,
    get_combined_args,
)
from gaussian_renderer import render
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel
from train import build_projected_gaussian_touch_mask
from utils.general_utils import safe_state
from utils.system_utils import mkdir_p
from utils.training_diagnostics import (
    DiagnosticBasisProvider,
    TwoDDropoutGradientDiagnostic,
    project_points_camera_torch,
    save_rgb_image,
)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_image_name(name: str) -> List[str]:
    path = Path(str(name))
    return [
        str(name),
        path.name,
        path.stem,
        str(name).lower(),
        path.name.lower(),
        path.stem.lower(),
    ]


def build_appearance_embedding(mesh_args, num_views: int):
    if mesh_args.use_decoupled_appearance:
        return AppearanceEmbedding(num_views=num_views)
    if mesh_args.use_pgsr_appearance:
        return PGSREmbedding(num_views=num_views)
    return None


def resolve_start_checkpoint(model_path: str, start_checkpoint: Optional[str], iteration: int) -> str:
    if start_checkpoint:
        return start_checkpoint
    if iteration < 0:
        raise ValueError("iteration must be explicit when start_checkpoint is not provided.")
    return os.path.join(model_path, f"chkpnt{iteration}.pth")


def load_mask_from_payload(payload: dict, key: str, total: int) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Payload does not contain '{key}'. Available keys: {sorted(payload.keys())}")
    value = np.asarray(tensor_to_numpy(payload[key]))
    if value.dtype == np.bool_:
        value = value.reshape(-1)
        if value.shape[0] != total:
            raise ValueError(f"Boolean mask '{key}' has {value.shape[0]} entries, expected {total}.")
        return value.astype(bool, copy=False)
    ids = value.reshape(-1).astype(np.int64, copy=False)
    if ids.size and np.any((ids < 0) | (ids >= total)):
        raise ValueError(f"Index mask '{key}' contains ids outside [0, {total}).")
    mask = np.zeros((total,), dtype=bool)
    mask[ids] = True
    return mask


def masked_l1(rendered_image: torch.Tensor, gt_image: torch.Tensor, mask_hw: torch.Tensor) -> Optional[torch.Tensor]:
    weight = mask_hw.to(dtype=rendered_image.dtype, device=rendered_image.device)
    if float(weight.sum().item()) <= 0.0:
        return None
    return (torch.abs(rendered_image - gt_image) * weight.unsqueeze(0)).sum() / (weight.sum() * rendered_image.shape[0]).clamp_min(1.0)


def save_mask_image(path: Path, mask_hw: torch.Tensor):
    mask = mask_hw.detach().to(dtype=torch.float32).cpu().numpy()
    Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8), mode="L").save(path)


def save_mask_overlay(path: Path, image_chw: torch.Tensor, mask_hw: torch.Tensor):
    image = image_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
    mask = mask_hw.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
    overlay = image.copy()
    color = np.asarray([1.0, 0.2, 0.1], dtype=np.float32)
    alpha = 0.5 * mask[..., None]
    overlay = (1.0 - alpha) * overlay + alpha * color[None, None, :]
    Image.fromarray(np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def save_render_triplet(path: Path, base_image: torch.Tensor, plus_image: torch.Tensor, minus_image: torch.Tensor, final_image: torch.Tensor):
    images = []
    for image in [base_image, plus_image, minus_image, final_image]:
        arr = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
        images.append(Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB"))
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (2 * width, 2 * height), (24, 24, 24))
    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    for image, pos in zip(images, positions):
        sheet.paste(image, pos)
    sheet.save(path)


def aggregate_patch_metrics(patch_ids: np.ndarray, bias: np.ndarray, std: np.ndarray, coverage: np.ndarray) -> Dict[int, Dict[str, float]]:
    valid = (
        (patch_ids >= 0)
        & np.isfinite(bias)
        & np.isfinite(std)
        & np.isfinite(coverage)
    )
    out: Dict[int, Dict[str, float]] = {}
    if not np.any(valid):
        return out
    patch_ids = patch_ids[valid].astype(np.int64, copy=False)
    bias = bias[valid].astype(np.float32, copy=False)
    std = std[valid].astype(np.float32, copy=False)
    coverage = coverage[valid].astype(np.float32, copy=False)
    unique_patch_ids = np.unique(patch_ids)
    for patch_id in unique_patch_ids.tolist():
        keep = patch_ids == int(patch_id)
        out[int(patch_id)] = {
            "patch_id": int(patch_id),
            "gaussian_count": int(np.sum(keep)),
            "bias": float(np.mean(bias[keep])),
            "std": float(np.mean(std[keep])),
            "coverage": float(np.mean(coverage[keep])),
        }
    return out


def load_dropout_snapshot(path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "dropout_gradient" in payload:
        snapshot = payload["dropout_gradient"]
    else:
        snapshot = payload
    if snapshot is None:
        raise ValueError(f"Dropout snapshot is empty: {path}")
    required = ["visible_ids", "dropout_bias", "dropout_std", "dropout_coverage", "camera_name"]
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise KeyError(f"Dropout snapshot is missing {missing}: {path}")
    return snapshot


def load_patch_bank(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def maybe_load_camera_patch_observation(patch_bank_path: str, camera_name: str) -> Optional[Dict[str, np.ndarray]]:
    obs_dir = Path(patch_bank_path).parent / "camera_patch_observations"
    candidates = [obs_dir / f"{camera_name}.npz"]
    for token in normalize_image_name(camera_name):
        candidates.append(obs_dir / f"{token}.npz")
    for path in candidates:
        if path.exists():
            data = np.load(path, allow_pickle=False)
            return {key: data[key] for key in data.files}
    return None


def index_cameras(cameras) -> Dict[str, object]:
    mapping = {}
    for camera in cameras:
        for token in normalize_image_name(camera.image_name):
            mapping[token] = camera
    return mapping


def assign_gaussians_to_patches(
    nearest_face_id: np.ndarray,
    reference_points: np.ndarray,
    patch_face_ids: np.ndarray,
    patch_centers: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    total = int(nearest_face_id.shape[0])
    patch_ids = np.full((total,), -1, dtype=np.int64)
    face_to_patch: Dict[int, np.ndarray] = {}
    for face_id in np.unique(patch_face_ids).tolist():
        face_to_patch[int(face_id)] = np.flatnonzero(patch_face_ids == int(face_id)).astype(np.int64, copy=False)

    valid_ids = np.flatnonzero(active_mask & (nearest_face_id >= 0)).astype(np.int64, copy=False)
    if valid_ids.size == 0:
        return patch_ids

    unique_faces = np.unique(nearest_face_id[valid_ids]).astype(np.int64, copy=False)
    for face_id in unique_faces.tolist():
        candidate_patch_ids = face_to_patch.get(int(face_id), None)
        if candidate_patch_ids is None or candidate_patch_ids.size == 0:
            continue
        local_ids = valid_ids[nearest_face_id[valid_ids] == int(face_id)]
        local_points = reference_points[local_ids].astype(np.float32, copy=False)
        centers = patch_centers[candidate_patch_ids].astype(np.float32, copy=False)
        dist_sq = np.sum((local_points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        chosen = candidate_patch_ids[np.argmin(dist_sq, axis=1)]
        patch_ids[local_ids] = chosen.astype(np.int64, copy=False)
    return patch_ids


def _draw_projected_support_mask(
    viewpoint_cam,
    gaussians: GaussianModel,
    support_ids: np.ndarray,
    radii: Optional[torch.Tensor],
    radius_mult: float,
    min_radius_px: float,
    max_radius_px: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if support_ids.size == 0:
        return None
    support_ids_t = torch.as_tensor(support_ids, device=device, dtype=torch.long)
    xyz_world = gaussians.get_xyz.detach()[support_ids_t]
    projected_xy, valid = project_points_camera_torch(viewpoint_cam, xyz_world)
    if not torch.any(valid):
        return None

    projected_xy = projected_xy[valid].detach().cpu().numpy().astype(np.float32, copy=False)
    if radii is not None:
        support_radii = radii.detach()[support_ids_t][valid].to(dtype=torch.float32)
        support_radii = torch.clamp(
            support_radii * float(radius_mult),
            min=float(min_radius_px),
            max=float(max_radius_px),
        ).detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        support_radii = np.full((projected_xy.shape[0],), float(min_radius_px), dtype=np.float32)

    width = int(viewpoint_cam.image_width)
    height = int(viewpoint_cam.image_height)
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for (cx, cy), radius in zip(projected_xy.tolist(), support_radii.tolist()):
        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
            continue
        if radius <= 0.0:
            continue
        x0 = float(cx) - float(radius)
        y0 = float(cy) - float(radius)
        x1 = float(cx) + float(radius)
        y1 = float(cy) + float(radius)
        draw.ellipse((x0, y0, x1, y1), fill=255)
    mask_np = np.asarray(mask_image, dtype=np.uint8) > 0
    if not np.any(mask_np):
        return None
    return torch.as_tensor(mask_np, device=device, dtype=torch.bool)


def _draw_patch_footprint_mask(
    viewpoint_cam,
    patch_id: int,
    patch_bank: Dict[str, np.ndarray],
    observation: Optional[Dict[str, np.ndarray]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if observation is not None and "patch_ids" in observation and "corner_xy" in observation:
        obs_patch_ids = observation["patch_ids"].astype(np.int64, copy=False)
        keep = np.flatnonzero(obs_patch_ids == int(patch_id)).astype(np.int64, copy=False)
        if keep.size > 0:
            corners = observation["corner_xy"][keep[0]].astype(np.float32, copy=False)
            if corners.ndim == 2 and corners.shape[0] >= 3:
                width = int(viewpoint_cam.image_width)
                height = int(viewpoint_cam.image_height)
                mask_image = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask_image)
                draw.polygon([tuple(map(float, xy)) for xy in corners.tolist()], fill=255)
                mask_np = np.asarray(mask_image, dtype=np.uint8) > 0
                if np.any(mask_np):
                    return torch.as_tensor(mask_np, device=device, dtype=torch.bool)

    xyz_world = torch.as_tensor(patch_bank["patch_corners_world"][int(patch_id)], device=device, dtype=torch.float32)
    projected_xy, valid = project_points_camera_torch(viewpoint_cam, xyz_world)
    if int(valid.sum().item()) < 3:
        return None
    corners = projected_xy[valid].detach().cpu().numpy().astype(np.float32, copy=False)
    height = int(viewpoint_cam.image_height)
    width = int(viewpoint_cam.image_width)
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([tuple(map(float, xy)) for xy in corners.tolist()], fill=255)
    mask_np = np.asarray(mask_image, dtype=np.uint8) > 0
    if not np.any(mask_np):
        return None
    return torch.as_tensor(mask_np, device=device, dtype=torch.bool)


def build_patch_local_mask(
    viewpoint_cam,
    patch_id: int,
    patch_bank: Dict[str, np.ndarray],
    observation: Optional[Dict[str, np.ndarray]],
    gaussians: GaussianModel,
    support_ids: np.ndarray,
    radii: Optional[torch.Tensor],
    radius_mult: float,
    min_radius_px: float,
    max_radius_px: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    support_mask = _draw_projected_support_mask(
        viewpoint_cam=viewpoint_cam,
        gaussians=gaussians,
        support_ids=support_ids,
        radii=radii,
        radius_mult=radius_mult,
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
        device=device,
    )
    if support_mask is not None and float(support_mask.sum().item()) > 0.0:
        return support_mask
    return _draw_patch_footprint_mask(
        viewpoint_cam=viewpoint_cam,
        patch_id=patch_id,
        patch_bank=patch_bank,
        observation=observation,
        device=device,
    )


def compute_feedback_targets(
    gaussians: GaussianModel,
    support_ids: np.ndarray,
    patch_normal: np.ndarray,
    delta: float,
    direction: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = gaussians.get_xyz.device
    support_ids_t = torch.as_tensor(support_ids, device=device, dtype=torch.long)
    orig_xyz = gaussians.get_xyz.detach()[support_ids_t]
    normal_t = torch.as_tensor(patch_normal, device=device, dtype=torch.float32).view(1, 3)
    target_xyz = orig_xyz + float(direction * delta) * normal_t
    return orig_xyz, target_xyz


def compute_support_weights(reference_points: np.ndarray, support_ids: np.ndarray, patch_center: np.ndarray, sigma: float, device: torch.device) -> torch.Tensor:
    local_points = reference_points[support_ids].astype(np.float32, copy=False)
    dist = np.linalg.norm(local_points - patch_center[None, :].astype(np.float32, copy=False), axis=1)
    sigma = max(float(sigma), 1e-6)
    weights = np.exp(-0.5 * np.square(dist / sigma)).astype(np.float32, copy=False)
    weights = np.clip(weights, 1e-3, 1.0)
    return torch.as_tensor(weights, device=device, dtype=torch.float32)


def backup_gaussian_state(gaussians: GaussianModel) -> Dict[str, object]:
    return {
        "xyz": gaussians._xyz.detach().clone(),
        "optimizer_state": copy.deepcopy(gaussians.optimizer.state_dict()),
        "lrs": [float(group["lr"]) for group in gaussians.optimizer.param_groups],
    }


def restore_gaussian_state(gaussians: GaussianModel, backup: Dict[str, object]):
    gaussians._xyz.data.copy_(backup["xyz"])
    gaussians.optimizer.load_state_dict(backup["optimizer_state"])
    for group, lr in zip(gaussians.optimizer.param_groups, backup["lrs"]):
        group["lr"] = float(lr)
    gaussians.optimizer.zero_grad(set_to_none=True)


def configure_local_optimizer(gaussians: GaussianModel, xyz_lr: float):
    for group in gaussians.optimizer.param_groups:
        name = str(group.get("name", ""))
        group["lr"] = float(xyz_lr) if name == "xyz" else 0.0


def zero_nonlocal_gradients(gaussians: GaussianModel, support_mask_t: torch.Tensor):
    for group in gaussians.optimizer.param_groups:
        name = str(group.get("name", ""))
        for param in group["params"]:
            if param.grad is None:
                continue
            if name == "xyz":
                param.grad[~support_mask_t] = 0.0
            else:
                param.grad.zero_()


def extract_patch_metric(snapshot: Optional[Dict[str, object]], patch_assignments: np.ndarray, patch_id: int) -> Optional[Dict[str, float]]:
    if snapshot is None:
        return None
    visible_ids = tensor_to_numpy(snapshot["visible_ids"]).reshape(-1).astype(np.int64, copy=False)
    if visible_ids.size == 0:
        return None
    bias = tensor_to_numpy(snapshot["dropout_bias"]).reshape(-1).astype(np.float32, copy=False)
    std = tensor_to_numpy(snapshot["dropout_std"]).reshape(-1).astype(np.float32, copy=False)
    coverage = tensor_to_numpy(snapshot["dropout_coverage"]).reshape(-1).astype(np.float32, copy=False)
    patch_ids = patch_assignments[visible_ids]
    metrics = aggregate_patch_metrics(patch_ids=patch_ids, bias=bias, std=std, coverage=coverage)
    return metrics.get(int(patch_id), None)


def run_dropout_snapshot(
    dropout_diag: TwoDDropoutGradientDiagnostic,
    viewpoint_cam,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    gt_image: torch.Tensor,
    render_pkg: Dict[str, torch.Tensor],
    splat_args,
    gradient_mask: Optional[torch.Tensor],
) -> Optional[Dict[str, object]]:
    return dropout_diag.run(
        iteration=1,
        phase_name="patch_feedback_eval",
        viewpoint_cam=viewpoint_cam,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        gt_image=gt_image.detach(),
        base_rendering=render_pkg["render"].detach(),
        splat_args=splat_args,
        render_fn=render,
        build_touch_mask_fn=build_projected_gaussian_touch_mask,
        gradient_mask=gradient_mask,
    )


def apply_surface_push(
    gaussians: GaussianModel,
    support_ids: np.ndarray,
    support_weights_t: torch.Tensor,
    patch_normal: np.ndarray,
    delta: float,
    direction: float,
):
    device = gaussians.get_xyz.device
    support_ids_t = torch.as_tensor(support_ids, device=device, dtype=torch.long)
    normal_t = torch.as_tensor(patch_normal, device=device, dtype=torch.float32).view(1, 3)
    weighted_delta = support_weights_t[:, None] * float(direction * delta) * normal_t
    gaussians._xyz.data[support_ids_t] += weighted_delta


def evaluate_surface_push_direction(
    gaussians: GaussianModel,
    viewpoint_cam,
    pipe,
    background: torch.Tensor,
    splat_args,
    gt_image: torch.Tensor,
    support_ids: np.ndarray,
    support_weights_t: torch.Tensor,
    patch_normal: np.ndarray,
    delta: float,
    direction: float,
    dropout_diag: TwoDDropoutGradientDiagnostic,
    patch_assignments: np.ndarray,
    patch_id: int,
) -> Dict[str, object]:
    device = gaussians.get_xyz.device
    support_ids_t = torch.as_tensor(support_ids, device=device, dtype=torch.long)
    support_mask_t = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=device)
    support_mask_t[support_ids_t] = True

    apply_surface_push(
        gaussians=gaussians,
        support_ids=support_ids,
        support_weights_t=support_weights_t,
        patch_normal=patch_normal,
        delta=delta,
        direction=direction,
    )
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, splat_args=splat_args)
    render_image = torch.clamp(render_pkg["render"][:3], 0.0, 1.0)
    photo_loss = torch.mean(torch.abs(render_image - gt_image)).item()
    dropout_snapshot = run_dropout_snapshot(
        dropout_diag=dropout_diag,
        viewpoint_cam=viewpoint_cam,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        gt_image=gt_image,
        render_pkg=render_pkg,
        splat_args=splat_args,
        gradient_mask=support_mask_t,
    )
    patch_metric = extract_patch_metric(dropout_snapshot, patch_assignments=patch_assignments, patch_id=patch_id)
    return {
        "render_pkg": render_pkg,
        "render_image": render_image.detach(),
        "photo_loss": float(photo_loss),
        "dropout_snapshot": dropout_snapshot,
        "patch_metric": patch_metric,
    }


def optimize_patch_direction(
    gaussians: GaussianModel,
    viewpoint_cam,
    pipe,
    background: torch.Tensor,
    splat_args,
    gt_image: torch.Tensor,
    local_mask: torch.Tensor,
    support_ids: np.ndarray,
    support_weights_t: torch.Tensor,
    patch_normal: np.ndarray,
    target_xyz: torch.Tensor,
    local_steps: int,
    xyz_lr: float,
    lambda_photo: float,
    lambda_feedback_normal: float,
    lambda_feedback_tangent: float,
    dropout_diag: TwoDDropoutGradientDiagnostic,
    patch_assignments: np.ndarray,
    patch_id: int,
) -> Dict[str, object]:
    device = gaussians.get_xyz.device
    support_ids_t = torch.as_tensor(support_ids, device=device, dtype=torch.long)
    support_mask_t = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=device)
    support_mask_t[support_ids_t] = True
    normal_t = torch.as_tensor(patch_normal, device=device, dtype=torch.float32).view(1, 3)
    configure_local_optimizer(gaussians, xyz_lr=xyz_lr)

    last_render_pkg = None
    photo_value = None
    feedback_n_value = None
    feedback_t_value = None
    for _ in range(max(int(local_steps), 1)):
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, splat_args=splat_args)
        render_image = torch.clamp(render_pkg["render"][:3], 0.0, 1.0)
        photo_loss = masked_l1(render_image, gt_image, local_mask)
        if photo_loss is None:
            raise RuntimeError("Local patch mask produced zero active pixels.")
        xyz_support = gaussians.get_xyz[support_ids_t]
        diff = xyz_support - target_xyz
        normal_resid = torch.sum(diff * normal_t, dim=1)
        tangent_resid = diff - normal_resid[:, None] * normal_t
        weight_sum = torch.clamp_min(support_weights_t.sum(), 1e-6)
        feedback_normal = (support_weights_t * normal_resid.pow(2)).sum() / weight_sum
        feedback_tangent = (support_weights_t * tangent_resid.pow(2).sum(dim=1)).sum() / weight_sum
        total_loss = (
            float(lambda_photo) * photo_loss
            + float(lambda_feedback_normal) * feedback_normal
            + float(lambda_feedback_tangent) * feedback_tangent
        )

        gaussians.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        zero_nonlocal_gradients(gaussians, support_mask_t=support_mask_t)
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        last_render_pkg = render_pkg
        photo_value = float(photo_loss.item())
        feedback_n_value = float(feedback_normal.item())
        feedback_t_value = float(feedback_tangent.item())

    if last_render_pkg is None:
        raise RuntimeError("Patch optimization did not run any steps.")
    final_render_pkg = render(viewpoint_cam, gaussians, pipe, background, splat_args=splat_args)
    final_image = torch.clamp(final_render_pkg["render"][:3], 0.0, 1.0)
    local_photo = masked_l1(final_image, gt_image, local_mask)
    dropout_snapshot = run_dropout_snapshot(
        dropout_diag=dropout_diag,
        viewpoint_cam=viewpoint_cam,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        gt_image=gt_image,
        render_pkg=final_render_pkg,
        splat_args=splat_args,
        gradient_mask=support_mask_t,
    )
    patch_metric = extract_patch_metric(dropout_snapshot, patch_assignments=patch_assignments, patch_id=patch_id)
    return {
        "render_pkg": final_render_pkg,
        "render_image": final_image.detach(),
        "photo_loss": float(local_photo.item()) if local_photo is not None else float(photo_value or 0.0),
        "feedback_normal_loss": float(feedback_n_value or 0.0),
        "feedback_tangent_loss": float(feedback_t_value or 0.0),
        "dropout_snapshot": dropout_snapshot,
        "patch_metric": patch_metric,
    }


def patch_acceptance_score(
    base_metric: Dict[str, float],
    trial_metric: Optional[Dict[str, float]],
    base_photo: float,
    trial_photo: float,
    photo_penalty: float,
) -> float:
    if trial_metric is None:
        return -1e9
    std_gain = float(base_metric["std"]) - float(trial_metric["std"])
    bias_gain = abs(float(base_metric["bias"])) - abs(float(trial_metric["bias"]))
    photo_worsen = max(float(trial_photo) - float(base_photo), 0.0)
    return std_gain + 0.25 * bias_gain - float(photo_penalty) * photo_worsen


def patch_dropout_surface_score(
    base_metric: Dict[str, float],
    trial_metric: Optional[Dict[str, float]],
) -> float:
    if trial_metric is None:
        return -1e9
    std_gain = float(base_metric["std"]) - float(trial_metric["std"])
    bias_gain = abs(float(base_metric["bias"])) - abs(float(trial_metric["bias"]))
    coverage_gain = float(trial_metric["coverage"]) - float(base_metric["coverage"])
    return std_gain + 0.25 * bias_gain + 0.05 * coverage_gain


def copy_render_config_from_base(base_model_path: str, output_model_path: Path):
    base = Path(base_model_path)
    output_model_path.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "config.json"]:
        src = base / name
        if src.exists():
            shutil.copy2(src, output_model_path / name)


def save_model_state(
    output_model_path: Path,
    iteration: int,
    gaussians: GaussianModel,
    appearance_embedding,
    summary: Dict[str, object],
):
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(iteration)}"
    mkdir_p(str(point_dir))
    gaussians.save_ply(str(point_dir / "point_cloud.ply"))
    gaussians.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    appearance_state = appearance_embedding.capture() if appearance_embedding is not None else (None, None)
    torch.save((gaussians.capture(), int(iteration), appearance_state), output_model_path / f"chkpnt{int(iteration)}.pth")
    with open(output_model_path / "patch_feedback_v0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = ArgumentParser(description="Refine original SOF gaussians with patch-level dropout-guided mesh feedback.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--patch_bank_path", type=str, required=True)
    parser.add_argument("--dropout_snapshot", type=str, required=True)
    parser.add_argument("--outlier_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--support_max_surface_distance", type=float, default=0.0)
    parser.add_argument("--support_min_visible_views", type=int, default=0)
    parser.add_argument("--support_min_opacity", type=float, default=0.0)
    parser.add_argument("--min_patch_gaussians", type=int, default=6)
    parser.add_argument("--min_patch_coverage", type=float, default=0.4)
    parser.add_argument("--min_patch_std", type=float, default=0.0)
    parser.add_argument("--min_action_score", type=float, default=0.0)
    parser.add_argument("--max_candidate_patches", type=int, default=8)
    parser.add_argument("--max_patch_updates", type=int, default=3)
    parser.add_argument("--local_steps", type=int, default=20)
    parser.add_argument("--local_xyz_lr", type=float, default=5e-4)
    parser.add_argument("--lambda_local_photo", type=float, default=1.0)
    parser.add_argument("--lambda_feedback_normal", type=float, default=20.0)
    parser.add_argument("--lambda_feedback_tangent", type=float, default=2.0)
    parser.add_argument("--feedback_sigma_scale_mult", type=float, default=2.0)
    parser.add_argument("--delta_scale_mult", type=float, default=0.5)
    parser.add_argument("--delta_abs", type=float, default=0.002)
    parser.add_argument("--delta_max", type=float, default=0.01)
    parser.add_argument("--acceptance_photo_penalty", type=float, default=10.0)
    parser.add_argument("--acceptance_min_score", type=float, default=0.0)
    parser.add_argument("--local_mask_radius_mult", type=float, default=3.0)
    parser.add_argument("--local_mask_min_radius_px", type=float, default=12.0)
    parser.add_argument("--local_mask_max_radius_px", type=float, default=96.0)
    parser.add_argument("--dropout_num_masks", type=int, default=8)
    parser.add_argument("--dropout_tile_size", type=int, default=16)
    parser.add_argument("--dropout_keep_ratio", type=float, default=0.5)
    parser.add_argument("--dropout_loss_mode", choices=["masked_l1", "masked_gradient_l1"], default="masked_l1")
    parser.add_argument("--dropout_alpha_threshold", type=float, default=0.1)
    parser.add_argument("--dropout_min_active_pixels", type=int, default=256)
    parser.add_argument("--output_model_path", type=str, default=None)
    parser.add_argument("--output_iteration", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for patch feedback refine.")
        args.data_device = "cpu"

    safe_state(args.quiet)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)
    splatting.get_settings(args)

    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe_args.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()
    camera_index = index_cameras(train_cameras)

    appearance_embedding = build_appearance_embedding(mesh_args, num_views=len(train_cameras))
    gaussians.training_setup(opt_args, mesh_args, appearance_embedding)

    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    checkpoint_path = resolve_start_checkpoint(dataset.model_path, args.start_checkpoint, loaded_iteration)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_params, checkpoint_iteration, appearance_state = torch.load(checkpoint_path)
    if appearance_embedding is not None and appearance_state[0] is not None:
        appearance_embedding.restore(*appearance_state)
    gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)

    dropout_snapshot = load_dropout_snapshot(args.dropout_snapshot)
    camera_name = str(dropout_snapshot["camera_name"])
    camera = None
    for token in normalize_image_name(camera_name):
        if token in camera_index:
            camera = camera_index[token]
            break
    if camera is None:
        raise KeyError(f"Could not find camera '{camera_name}' in train cameras.")

    gt_image = camera.original_image.cuda().clamp(0.0, 1.0)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    patch_bank = load_patch_bank(args.patch_bank_path)
    candidate_payload = torch.load(args.candidate_payload, map_location="cpu")

    nearest_face_id = tensor_to_numpy(candidate_payload["nearest_face_id"]).reshape(-1).astype(np.int64, copy=False)
    total_gaussians = int(nearest_face_id.shape[0])
    outlier_mask = load_mask_from_payload(candidate_payload, args.outlier_mask_key, total=total_gaussians)
    support_mask = (~outlier_mask) & (nearest_face_id >= 0)
    if "surface_distance" in candidate_payload and float(args.support_max_surface_distance) > 0.0:
        support_mask &= tensor_to_numpy(candidate_payload["surface_distance"]).reshape(-1) <= float(args.support_max_surface_distance)
    if "visible_view_count" in candidate_payload and int(args.support_min_visible_views) > 0:
        support_mask &= tensor_to_numpy(candidate_payload["visible_view_count"]).reshape(-1) >= int(args.support_min_visible_views)
    if "opacity" in candidate_payload and float(args.support_min_opacity) > 0.0:
        support_mask &= tensor_to_numpy(candidate_payload["opacity"]).reshape(-1) >= float(args.support_min_opacity)

    xyz_world = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    reference_points = (
        tensor_to_numpy(candidate_payload["nearest_surface_point"]).astype(np.float32, copy=False)
        if "nearest_surface_point" in candidate_payload
        else xyz_world
    )
    patch_assignments = assign_gaussians_to_patches(
        nearest_face_id=nearest_face_id,
        reference_points=reference_points,
        patch_face_ids=patch_bank["face_ids"].astype(np.int64, copy=False),
        patch_centers=patch_bank["centers"].astype(np.float32, copy=False),
        active_mask=support_mask,
    )

    visible_ids = tensor_to_numpy(dropout_snapshot["visible_ids"]).reshape(-1).astype(np.int64, copy=False)
    bias = tensor_to_numpy(dropout_snapshot["dropout_bias"]).reshape(-1).astype(np.float32, copy=False)
    std = tensor_to_numpy(dropout_snapshot["dropout_std"]).reshape(-1).astype(np.float32, copy=False)
    coverage = tensor_to_numpy(dropout_snapshot["dropout_coverage"]).reshape(-1).astype(np.float32, copy=False)
    visible_patch_ids = patch_assignments[visible_ids]
    patch_metrics = aggregate_patch_metrics(
        patch_ids=visible_patch_ids,
        bias=bias,
        std=std,
        coverage=coverage,
    )

    patch_to_support_ids: Dict[int, np.ndarray] = {}
    valid_support_ids = np.flatnonzero(patch_assignments >= 0).astype(np.int64, copy=False)
    if valid_support_ids.size > 0:
        unique_patch_ids = np.unique(patch_assignments[valid_support_ids]).astype(np.int64, copy=False)
        for patch_id in unique_patch_ids.tolist():
            ids = valid_support_ids[patch_assignments[valid_support_ids] == int(patch_id)]
            patch_to_support_ids[int(patch_id)] = ids.astype(np.int64, copy=False)

    candidates: List[Dict[str, float]] = []
    for patch_id, metric in patch_metrics.items():
        support_ids = patch_to_support_ids.get(int(patch_id), np.empty((0,), dtype=np.int64))
        gaussian_count = int(metric["gaussian_count"])
        support_count = int(support_ids.size)
        action_score = float(metric["coverage"] * abs(metric["bias"]) / max(metric["std"], 1e-6))
        under_score = float(metric["coverage"] * metric["std"])
        if gaussian_count < int(args.min_patch_gaussians):
            continue
        if support_count < int(args.min_patch_gaussians):
            continue
        if float(metric["coverage"]) < float(args.min_patch_coverage):
            continue
        if float(metric["std"]) < float(args.min_patch_std):
            continue
        if action_score < float(args.min_action_score):
            continue
        candidates.append(
            {
                "patch_id": int(patch_id),
                "bias": float(metric["bias"]),
                "std": float(metric["std"]),
                "coverage": float(metric["coverage"]),
                "gaussian_count": gaussian_count,
                "support_count": support_count,
                "action_score": action_score,
                "under_score": under_score,
            }
        )

    candidates = sorted(candidates, key=lambda item: (-item["action_score"], -item["under_score"], -item["coverage"]))
    if int(args.max_candidate_patches) > 0:
        candidates = candidates[: int(args.max_candidate_patches)]
    if not candidates:
        raise RuntimeError("No candidate patches remained after filtering.")

    dropout_diag = TwoDDropoutGradientDiagnostic(
        enabled=True,
        start_iter=0,
        interval=1,
        num_masks=int(args.dropout_num_masks),
        tile_size=int(args.dropout_tile_size),
        keep_ratio=float(args.dropout_keep_ratio),
        basis_provider=DiagnosticBasisProvider(basis_mode="surface_payload", surface_payload_path=args.candidate_payload),
        loss_mode=str(args.dropout_loss_mode),
        alpha_threshold=float(args.dropout_alpha_threshold),
        min_active_pixels=int(args.dropout_min_active_pixels),
    )

    output_model_path = Path(args.output_model_path) if args.output_model_path else (Path(dataset.model_path).parent / f"{Path(dataset.model_path).name}_patch_feedback_v0")
    preview_root = output_model_path / "patch_feedback_previews_v0"
    preview_root.mkdir(parents=True, exist_ok=True)
    copy_render_config_from_base(dataset.model_path, output_model_path)

    baseline_render_pkg = render(camera, gaussians, pipe_args, background, splat_args=splatting.settings)
    baseline_image = torch.clamp(baseline_render_pkg["render"][:3], 0.0, 1.0)
    accepted_records: List[Dict[str, object]] = []
    original_backup = backup_gaussian_state(gaussians)

    for candidate_rank, candidate in enumerate(candidates[: int(args.max_patch_updates)]):
        patch_id = int(candidate["patch_id"])
        support_ids = patch_to_support_ids.get(patch_id, np.empty((0,), dtype=np.int64))
        if support_ids.size == 0:
            continue

        base_metric = {
            "bias": float(candidate["bias"]),
            "std": float(candidate["std"]),
            "coverage": float(candidate["coverage"]),
        }
        patch_scale = max(float(patch_bank["scale_n"][patch_id]), 1e-6)
        delta = min(float(args.delta_max), max(float(args.delta_abs), float(args.delta_scale_mult) * patch_scale))
        patch_normal = patch_bank["normals"][patch_id].astype(np.float32, copy=False)
        patch_center = patch_bank["centers"][patch_id].astype(np.float32, copy=False)
        sigma = max(float(patch_bank["scale_u"][patch_id]), float(patch_bank["scale_v"][patch_id])) * float(args.feedback_sigma_scale_mult)
        support_weights_t = compute_support_weights(
            reference_points=reference_points,
            support_ids=support_ids,
            patch_center=patch_center,
            sigma=sigma,
            device=gaussians.get_xyz.device,
        )

        patch_dir = preview_root / f"patch_{candidate_rank:02d}_{patch_id:06d}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        save_rgb_image(patch_dir / "base_gt.png", gt_image)
        save_rgb_image(patch_dir / "base_render.png", baseline_image)

        trial_results: Dict[str, Dict[str, object]] = {}
        patch_base_backup = backup_gaussian_state(gaussians)
        patch_base_render = baseline_image.detach().clone()
        for direction_name, direction_sign in [("plus", 1.0), ("minus", -1.0)]:
            restore_gaussian_state(gaussians, patch_base_backup)
            trial_result = evaluate_surface_push_direction(
                gaussians=gaussians,
                viewpoint_cam=camera,
                pipe=pipe_args,
                background=background,
                splat_args=splatting.settings,
                gt_image=gt_image,
                support_ids=support_ids,
                support_weights_t=support_weights_t,
                patch_normal=patch_normal,
                delta=delta,
                direction=direction_sign,
                dropout_diag=dropout_diag,
                patch_assignments=patch_assignments,
                patch_id=patch_id,
            )
            trial_result["score"] = patch_dropout_surface_score(
                base_metric=base_metric,
                trial_metric=trial_result["patch_metric"],
            )
            trial_results[direction_name] = trial_result
            save_rgb_image(patch_dir / f"{direction_name}_render.png", trial_result["render_image"])
            restore_gaussian_state(gaussians, patch_base_backup)

        chosen_name = max(trial_results.keys(), key=lambda key: float(trial_results[key]["score"]))
        chosen_result = trial_results[chosen_name]
        accepted = float(chosen_result["score"]) >= float(args.acceptance_min_score)

        if accepted:
            restore_gaussian_state(gaussians, patch_base_backup)
            direction_sign = 1.0 if chosen_name == "plus" else -1.0
            apply_surface_push(
                gaussians=gaussians,
                support_ids=support_ids,
                support_weights_t=support_weights_t,
                patch_normal=patch_normal,
                delta=delta,
                direction=direction_sign,
            )
            baseline_render_pkg = render(camera, gaussians, pipe_args, background, splat_args=splatting.settings)
            baseline_image = torch.clamp(baseline_render_pkg["render"][:3], 0.0, 1.0)
            original_backup = backup_gaussian_state(gaussians)
        else:
            restore_gaussian_state(gaussians, patch_base_backup)

        final_image = chosen_result["render_image"] if accepted else patch_base_render
        save_render_triplet(
            patch_dir / "trial_contact_sheet.png",
            base_image=patch_base_render,
            plus_image=trial_results["plus"]["render_image"],
            minus_image=trial_results["minus"]["render_image"],
            final_image=final_image,
        )
        record = {
            "patch_id": patch_id,
            "candidate_rank": int(candidate_rank),
            "candidate_metric": candidate,
            "delta": float(delta),
            "chosen_direction": str(chosen_name),
            "accepted": bool(accepted),
            "base_metric": base_metric,
            "plus_metric": trial_results["plus"]["patch_metric"],
            "minus_metric": trial_results["minus"]["patch_metric"],
            "plus_photo": float(trial_results["plus"]["photo_loss"]),
            "minus_photo": float(trial_results["minus"]["photo_loss"]),
            "plus_score": float(trial_results["plus"]["score"]),
            "minus_score": float(trial_results["minus"]["score"]),
            "support_count": int(support_ids.size),
            "preview_dir": str(patch_dir.resolve()),
        }
        with open(patch_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        accepted_records.append(record)

    summary = {
        "mode": "patch_feedback_refine_v0",
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "dropout_snapshot": str(Path(args.dropout_snapshot).resolve()),
        "candidate_payload": str(Path(args.candidate_payload).resolve()),
        "patch_bank_path": str(Path(args.patch_bank_path).resolve()),
        "camera_name": camera_name,
        "candidate_count": int(len(candidates)),
        "processed_count": int(len(accepted_records)),
        "accepted_count": int(sum(1 for item in accepted_records if bool(item["accepted"]))),
        "records": accepted_records,
    }
    save_model_state(
        output_model_path=output_model_path,
        iteration=int(args.output_iteration),
        gaussians=gaussians,
        appearance_embedding=appearance_embedding,
        summary=summary,
    )
    print(f"Saved patch-feedback model to: {output_model_path}")
    print(f"Saved preview bundles to: {preview_root}")


if __name__ == "__main__":
    main()
