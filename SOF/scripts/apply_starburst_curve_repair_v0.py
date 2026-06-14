from __future__ import annotations

import argparse
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation


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


def _copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src_file = src_model_path / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probability / torch.clamp(1.0 - probability, min=1e-6))


def _opacity_compensation_metric(scales: torch.Tensor, mode: str) -> torch.Tensor:
    safe = torch.clamp(scales, min=1e-12)
    if mode == "volume":
        return torch.prod(safe, dim=1, keepdim=True)
    sorted_scales = torch.sort(safe, dim=1).values
    return torch.prod(sorted_scales[:, -2:], dim=1, keepdim=True)


def _line_split_pattern(split_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    count = max(2, int(split_count))
    x = torch.linspace(-1.0, 1.0, steps=count, dtype=dtype, device=device)
    zeros = torch.zeros_like(x)
    return torch.stack((x, zeros, zeros), dim=1)


def _ensure_tracking_state(gaussians: GaussianModel) -> None:
    count = int(gaussians.get_xyz.shape[0])
    if not hasattr(gaussians, "_source_tag") or int(gaussians._source_tag.shape[0]) != count:
        gaussians.init_tracking_state(count)


def _make_static_gaussian_model(
    *,
    base: GaussianModel,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    filter_3d: torch.Tensor,
    tracking_state: Dict[str, torch.Tensor],
) -> GaussianModel:
    model = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    model.active_sh_degree = int(base.active_sh_degree)
    model.spatial_lr_scale = float(base.spatial_lr_scale)
    model._xyz = nn.Parameter(xyz.detach().clone().requires_grad_(False))
    model._features_dc = nn.Parameter(features_dc.detach().clone().requires_grad_(False))
    model._features_rest = nn.Parameter(features_rest.detach().clone().requires_grad_(False))
    model._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(False))
    model._scaling = nn.Parameter(scaling.detach().clone().requires_grad_(False))
    model._rotation = nn.Parameter(rotation.detach().clone().requires_grad_(False))
    model.filter_3D = filter_3d.detach().clone()
    count = int(xyz.shape[0])
    device = xyz.device
    model.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=device)
    model.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.denom = torch.zeros((count, 1), dtype=torch.float32, device=device)
    model.init_tracking_state(count)
    model._source_tag = tracking_state["source_tag"].to(device=device, dtype=torch.int32)
    model._seed_id = tracking_state["seed_id"].to(device=device, dtype=torch.int64)
    model._generation = tracking_state["generation"].to(device=device, dtype=torch.int32)
    model._edge_touched = tracking_state["edge_touched"].to(device=device, dtype=torch.bool)
    model._edge_touch_iter = tracking_state["edge_touch_iter"].to(device=device, dtype=torch.int32)
    return model


def _stats(values: np.ndarray) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply major-axis curve-like replacement to starburst artifact Gaussians.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--score_payload_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--use_payload_candidate_mask", action="store_true")
    parser.add_argument("--score_key", default="starburst_score")
    parser.add_argument("--candidate_key", default="starburst_candidate")
    parser.add_argument("--min_starburst_score", type=float, default=0.18)
    parser.add_argument("--min_unsupported_score", type=float, default=0.08)
    parser.add_argument("--min_geometry_risk", type=float, default=0.16)
    parser.add_argument("--min_visible_count", type=int, default=1)
    parser.add_argument("--max_repair_fraction", type=float, default=0.012)
    parser.add_argument("--max_repair_count", type=int, default=18000)
    parser.add_argument("--split_count", type=int, default=6)
    parser.add_argument("--offset_scale", type=float, default=0.90)
    parser.add_argument("--child_major_scale_multiplier", type=float, default=0.32)
    parser.add_argument("--child_minor_scale_multiplier", type=float, default=0.78)
    parser.add_argument("--child_normal_scale_multiplier", type=float, default=0.60)
    parser.add_argument("--child_opacity_scale", type=float, default=0.82)
    parser.add_argument("--child_dc_scale", type=float, default=0.92)
    parser.add_argument("--child_rest_scale", type=float, default=0.10)
    parser.add_argument("--child_filter_scale", type=float, default=0.35)
    parser.add_argument("--filter_cap_ratio", type=float, default=0.0008)
    parser.add_argument("--energy_conserve_mode", choices=["none", "area", "volume"], default="area")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    score_payload_path = Path(args.score_payload_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    payload = torch.load(score_payload_path, map_location="cpu")
    if str(payload.get("version", "")) != "starburst_gaussian_scores_v0":
        raise ValueError(f"Unsupported starburst payload version in {score_payload_path}")

    iteration = _resolve_model_iteration(model_path, int(args.iteration))
    dataset = _build_dataset_args(
        scene_root=str(scene_root),
        model_path=str(model_path),
        images_subdir=str(args.images_subdir),
        white_background=bool(args.white_background),
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    _ensure_tracking_state(gaussians)

    count = int(gaussians.get_xyz.shape[0])
    score = torch.as_tensor(payload[str(args.score_key)]).reshape(-1).float()
    if int(score.shape[0]) != count:
        raise ValueError(f"Payload/model length mismatch for score key {args.score_key}: {score.shape[0]} vs {count}")
    unsupported = torch.as_tensor(payload.get("unsupported_score", torch.zeros_like(score[:, None]))).reshape(-1).float()
    geometry_risk = torch.as_tensor(payload.get("geometry_risk", torch.zeros_like(score[:, None]))).reshape(-1).float()
    visible_count = torch.as_tensor(payload.get("visible_count", torch.zeros_like(score[:, None]))).reshape(-1).long()
    payload_candidate = torch.zeros((count,), dtype=torch.bool)
    if bool(args.use_payload_candidate_mask):
        payload_candidate = torch.as_tensor(payload.get(str(args.candidate_key), torch.zeros((count, 1), dtype=torch.bool))).reshape(-1).bool()

    candidate = (score >= float(args.min_starburst_score))
    candidate &= (unsupported >= float(args.min_unsupported_score))
    candidate &= (geometry_risk >= float(args.min_geometry_risk))
    candidate &= (visible_count >= int(args.min_visible_count))
    if bool(args.use_payload_candidate_mask):
        candidate &= payload_candidate

    candidate_ids = torch.nonzero(candidate, as_tuple=False).squeeze(1)
    candidate_count_before_cap = int(candidate_ids.numel())
    if candidate_count_before_cap > 0:
        max_by_fraction = int(max(0, round(float(args.max_repair_fraction) * float(count))))
        max_count = candidate_count_before_cap
        if max_by_fraction > 0:
            max_count = min(max_count, max_by_fraction)
        if int(args.max_repair_count) > 0:
            max_count = min(max_count, int(args.max_repair_count))
        if max_count <= 0:
            selected_ids = candidate_ids[:0]
        elif candidate_count_before_cap > max_count:
            order = torch.argsort(score[candidate_ids], descending=True, stable=True)[:max_count]
            selected_ids = candidate_ids[order]
        else:
            selected_ids = candidate_ids
    else:
        selected_ids = candidate_ids

    xyz = gaussians._xyz.detach()
    features_dc = gaussians._features_dc.detach()
    features_rest = gaussians._features_rest.detach()
    opacity_raw = gaussians._opacity.detach()
    scaling_raw = gaussians._scaling.detach()
    rotation_raw = gaussians._rotation.detach()
    if isinstance(gaussians.filter_3D, torch.Tensor) and gaussians.filter_3D.ndim > 0:
        filter_3d = gaussians.filter_3D.detach()
    else:
        filter_3d = torch.zeros((count, 1), dtype=torch.float32, device=xyz.device)

    selected_count = int(selected_ids.numel())
    if selected_count <= 0:
        keep_mask = torch.ones((count,), dtype=torch.bool, device=xyz.device)
        child_output_mask = torch.zeros((count,), dtype=torch.bool, device=xyz.device)
        output_source_idx = torch.arange(count, dtype=torch.int64, device=xyz.device)
        tracking_state = {
            "source_tag": gaussians._source_tag.detach().clone(),
            "seed_id": gaussians._seed_id.detach().clone(),
            "generation": gaussians._generation.detach().clone(),
            "edge_touched": gaussians._edge_touched.detach().clone(),
            "edge_touch_iter": gaussians._edge_touch_iter.detach().clone(),
        }
        repaired = _make_static_gaussian_model(
            base=gaussians,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity_raw,
            scaling=scaling_raw,
            rotation=rotation_raw,
            filter_3d=filter_3d,
            tracking_state=tracking_state,
        )
    else:
        keep_mask = torch.ones((count,), dtype=torch.bool, device=xyz.device)
        keep_mask[selected_ids] = False

        scale = gaussians.get_scaling.detach()
        filter_col = filter_3d.reshape(-1, 1).to(device=xyz.device, dtype=scale.dtype)
        effective_scale = torch.sqrt(torch.square(scale) + torch.square(filter_col))
        selected_scale = effective_scale[selected_ids]
        axis_order = torch.argsort(selected_scale, dim=1, descending=True)
        rotations = build_rotation(rotation_raw[selected_ids])
        axis_basis = torch.gather(rotations, 2, axis_order[:, None, :].expand(-1, 3, -1))
        sorted_selected_scale = torch.gather(selected_scale, 1, axis_order)
        pattern = _line_split_pattern(int(args.split_count), device=xyz.device, dtype=xyz.dtype)
        actual_split_count = int(pattern.shape[0])

        local_offsets = pattern[None, :, :] * sorted_selected_scale[:, None, :] * float(args.offset_scale)
        child_xyz = xyz[selected_ids, None, :] + torch.einsum("bij,bnj->bni", axis_basis, local_offsets)

        selected_raw_scale = scale[selected_ids]
        sorted_raw_scale = torch.gather(selected_raw_scale, 1, axis_order)
        sorted_child_scale = sorted_raw_scale.clone()
        sorted_child_scale[:, 0] = sorted_child_scale[:, 0] * float(args.child_major_scale_multiplier)
        sorted_child_scale[:, 1] = sorted_child_scale[:, 1] * float(args.child_minor_scale_multiplier)
        sorted_child_scale[:, 2] = sorted_child_scale[:, 2] * float(args.child_normal_scale_multiplier)
        updated_child_scale = torch.zeros_like(selected_raw_scale)
        updated_child_scale.scatter_(1, axis_order, sorted_child_scale)
        child_scale = updated_child_scale[:, None, :].expand(-1, actual_split_count, -1)
        child_scaling = torch.log(torch.clamp(child_scale, min=1e-8)).reshape(-1, 3)

        child_filter = filter_3d[selected_ids, None, :].expand(-1, actual_split_count, -1)
        child_filter = child_filter * float(args.child_filter_scale)
        scene_extent = max(float(scene.cameras_extent), 1e-6)
        if float(args.filter_cap_ratio) > 0.0:
            child_filter = torch.clamp(child_filter, max=scene_extent * float(args.filter_cap_ratio))
        child_filter = torch.clamp(child_filter, min=0.0).reshape(-1, filter_3d.shape[1])

        parent_opacity = torch.sigmoid(opacity_raw[selected_ids]).reshape(-1)
        if str(args.energy_conserve_mode) == "none":
            base_child_opacity = 1.0 - torch.pow(
                torch.clamp(1.0 - parent_opacity, min=1e-6),
                1.0 / float(actual_split_count),
            )
            child_opacity_prob = base_child_opacity[:, None].expand(-1, actual_split_count)
            child_opacity_prob = child_opacity_prob * float(args.child_opacity_scale)
        else:
            parent_metric = _opacity_compensation_metric(selected_scale, str(args.energy_conserve_mode)).reshape(-1)
            child_effective_scale = torch.sqrt(torch.square(child_scale) + torch.square(child_filter.reshape(selected_count, actual_split_count, -1)))
            child_metric = _opacity_compensation_metric(
                child_effective_scale.reshape(-1, 3),
                str(args.energy_conserve_mode),
            ).reshape(selected_count, actual_split_count)
            child_metric_sum = torch.clamp(child_metric.sum(dim=1), min=1e-12)
            target_mass = parent_opacity * parent_metric * float(args.child_opacity_scale)
            child_opacity_prob = (target_mass / child_metric_sum)[:, None].expand(-1, actual_split_count)
        child_opacity_prob = torch.clamp(child_opacity_prob, min=1e-6, max=0.95)
        child_opacity = _logit(child_opacity_prob.reshape(-1, 1))

        child_features_dc = features_dc[selected_ids, None, ...].expand(
            -1,
            actual_split_count,
            *features_dc.shape[1:],
        ).reshape(-1, *features_dc.shape[1:]).clone()
        child_features_rest = features_rest[selected_ids, None, ...].expand(
            -1,
            actual_split_count,
            *features_rest.shape[1:],
        ).reshape(-1, *features_rest.shape[1:]).clone()
        child_features_dc = child_features_dc * float(args.child_dc_scale)
        child_features_rest = child_features_rest * float(args.child_rest_scale)
        child_rotation = rotation_raw[selected_ids, None, :].expand(-1, actual_split_count, -1).reshape(-1, rotation_raw.shape[1])

        xyz_out = torch.cat((xyz[keep_mask], child_xyz.reshape(-1, 3)), dim=0)
        features_dc_out = torch.cat((features_dc[keep_mask], child_features_dc), dim=0)
        features_rest_out = torch.cat((features_rest[keep_mask], child_features_rest), dim=0)
        opacity_out = torch.cat((opacity_raw[keep_mask], child_opacity), dim=0)
        scaling_out = torch.cat((scaling_raw[keep_mask], child_scaling), dim=0)
        rotation_out = torch.cat((rotation_raw[keep_mask], child_rotation), dim=0)
        filter_out = torch.cat((filter_3d[keep_mask], child_filter), dim=0)
        output_source_idx = torch.cat(
            (
                torch.arange(count, device=xyz.device, dtype=torch.int64)[keep_mask],
                selected_ids.repeat_interleave(actual_split_count),
            ),
            dim=0,
        )

        child_tracking = {
            "source_tag": gaussians._source_tag[selected_ids].repeat_interleave(actual_split_count),
            "seed_id": gaussians._seed_id[selected_ids].repeat_interleave(actual_split_count),
            "generation": (gaussians._generation[selected_ids] + 1).repeat_interleave(actual_split_count),
            "edge_touched": gaussians._edge_touched[selected_ids].repeat_interleave(actual_split_count),
            "edge_touch_iter": gaussians._edge_touch_iter[selected_ids].repeat_interleave(actual_split_count),
        }
        tracking_state = {
            "source_tag": torch.cat((gaussians._source_tag[keep_mask], child_tracking["source_tag"]), dim=0),
            "seed_id": torch.cat((gaussians._seed_id[keep_mask], child_tracking["seed_id"]), dim=0),
            "generation": torch.cat((gaussians._generation[keep_mask], child_tracking["generation"]), dim=0),
            "edge_touched": torch.cat((gaussians._edge_touched[keep_mask], child_tracking["edge_touched"]), dim=0),
            "edge_touch_iter": torch.cat((gaussians._edge_touch_iter[keep_mask], child_tracking["edge_touch_iter"]), dim=0),
        }
        child_output_mask = torch.cat(
            (
                torch.zeros((int(keep_mask.sum().item()),), dtype=torch.bool, device=xyz.device),
                torch.ones((selected_count * actual_split_count,), dtype=torch.bool, device=xyz.device),
            ),
            dim=0,
        )
        repaired = _make_static_gaussian_model(
            base=gaussians,
            xyz=xyz_out,
            features_dc=features_dc_out,
            features_rest=features_rest_out,
            opacity=opacity_out,
            scaling=scaling_out,
            rotation=rotation_out,
            filter_3d=filter_out,
            tracking_state=tracking_state,
        )

    _copy_render_config(model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{loaded_iter}"
    point_dir.mkdir(parents=True, exist_ok=True)
    repaired.save_ply(str(point_dir / "point_cloud.ply"))
    repaired.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    masks_dir = output_model_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    selected_input_mask = torch.zeros((count,), dtype=torch.bool)
    if selected_count > 0:
        selected_input_mask[selected_ids.detach().cpu()] = True
    torch.save(selected_input_mask, masks_dir / "starburst_replaced_input_mask.pt")
    torch.save(selected_ids.detach().cpu().to(torch.int64), masks_dir / "starburst_replaced_input_idx.pt")
    torch.save(child_output_mask.detach().cpu(), masks_dir / "starburst_child_output_mask.pt")
    torch.save(output_source_idx.detach().cpu(), masks_dir / "starburst_output_source_idx.pt")

    summary = {
        "version": "starburst_curve_repair_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "score_payload_path": str(score_payload_path),
        "output_model_path": str(output_model_path),
        "iteration": int(loaded_iter),
        "input_gaussians": int(count),
        "candidate_count_before_cap": int(candidate_count_before_cap),
        "selected_count": int(selected_count),
        "selected_ratio": float(selected_count / max(count, 1)),
        "output_gaussians": int(repaired.get_xyz.shape[0]),
        "child_count": int(child_output_mask.sum().item()),
        "args": vars(args),
        "selected_score_stats": _stats(score[selected_ids].detach().cpu().numpy() if selected_count > 0 else np.empty((0,), dtype=np.float32)),
        "selected_unsupported_stats": _stats(unsupported[selected_ids].detach().cpu().numpy() if selected_count > 0 else np.empty((0,), dtype=np.float32)),
        "selected_geometry_risk_stats": _stats(geometry_risk[selected_ids].detach().cpu().numpy() if selected_count > 0 else np.empty((0,), dtype=np.float32)),
    }
    (output_model_path / "starburst_curve_repair_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
