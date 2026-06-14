from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import inverse_sigmoid
from utils.sof_mesh_patch_enhancer_v0 import stats_from_array
from utils.system_utils import mkdir_p


def copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src = src_model_path / name
        if src.exists():
            shutil.copy2(src, dst_model_path / name)


def resolve_iteration(model_path: Path, iteration: int) -> int:
    if int(iteration) >= 0:
        return int(iteration)
    point_root = model_path / "point_cloud"
    candidates = []
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


def load_gaussian_model(model_path: Path, iteration: int, sh_degree: int) -> GaussianModel:
    ply_path = model_path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply_path}")
    model = GaussianModel(sh_degree)
    model.load_ply(str(ply_path))
    tags_path = ply_path.parent / "gaussian_tags.pt"
    if tags_path.is_file():
        model.load_tracking_metadata(str(tags_path))
    return model


def clone_with_xyz(source: GaussianModel, xyz_np: np.ndarray) -> GaussianModel:
    xyz = torch.from_numpy(xyz_np.astype(np.float32, copy=False)).to(device=source.get_xyz.device)
    output = GaussianModel(source.max_sh_degree, use_SBs=source.use_SBs)
    output.active_sh_degree = int(source.active_sh_degree)
    output.spatial_lr_scale = float(source.spatial_lr_scale)
    output._xyz = nn.Parameter(xyz.detach().clone(), requires_grad=False)
    output._features_dc = nn.Parameter(source._features_dc.detach().clone(), requires_grad=False)
    output._features_rest = nn.Parameter(source._features_rest.detach().clone(), requires_grad=False)
    output._opacity = nn.Parameter(source._opacity.detach().clone(), requires_grad=False)
    output._scaling = nn.Parameter(source._scaling.detach().clone(), requires_grad=False)
    output._rotation = nn.Parameter(source._rotation.detach().clone(), requires_grad=False)
    if source.filter_3D.ndim > 0 and source.filter_3D.shape[0] == xyz.shape[0]:
        output.filter_3D = source.filter_3D.detach().clone()
    else:
        output.filter_3D = source.filter_3D.detach().reshape(1, -1).repeat(xyz.shape[0], 1)
    output.max_radii2D = torch.zeros((xyz.shape[0],), dtype=torch.float32, device=xyz.device)
    output.restore_tracking_state(source.capture_tracking_state())
    return output


def _filter_3d_for_ids(source: GaussianModel, ids: torch.Tensor) -> torch.Tensor:
    if (
        isinstance(source.filter_3D, torch.Tensor)
        and source.filter_3D.ndim > 0
        and source.filter_3D.shape[0] == source.get_xyz.shape[0]
    ):
        return source.filter_3D.detach()[ids].clone()
    base = source.filter_3D.detach().reshape(1, -1)
    return base.repeat(ids.shape[0], 1)


def _tracking_for_appended(source: GaussianModel, base_count: int, probe_count: int) -> dict:
    state = source.capture_tracking_state()
    device = source.get_xyz.device
    extension = {
        "source_tag": torch.full((probe_count,), int(GaussianSourceTag.EXTENSION_PROBE), dtype=torch.int32, device=device),
        "seed_id": torch.full((probe_count,), -1, dtype=torch.int64, device=device),
        "generation": torch.full((probe_count,), 1, dtype=torch.int32, device=device),
        "edge_touched": torch.zeros((probe_count,), dtype=torch.bool, device=device),
        "edge_touch_iter": torch.full((probe_count,), -1, dtype=torch.int32, device=device),
    }
    output = {}
    for key, extra in extension.items():
        base = state[key].to(device=device)
        if base.shape[0] != base_count:
            if key == "source_tag":
                base = torch.full((base_count,), int(GaussianSourceTag.ORIGINAL), dtype=torch.int32, device=device)
            elif key == "seed_id":
                base = torch.full((base_count,), -1, dtype=torch.int64, device=device)
            elif key == "generation":
                base = torch.zeros((base_count,), dtype=torch.int32, device=device)
            elif key == "edge_touched":
                base = torch.zeros((base_count,), dtype=torch.bool, device=device)
            else:
                base = torch.full((base_count,), -1, dtype=torch.int32, device=device)
        output[key] = torch.cat((base, extra), dim=0)
    return output


def build_probe_augmented_model(
    source: GaussianModel,
    probe_xyz_np: np.ndarray,
    probe_ids_np: np.ndarray,
    *,
    opacity_scale: float,
    scale_multiplier: float,
    keep_original: bool,
) -> GaussianModel:
    device = source.get_xyz.device
    ids = torch.from_numpy(probe_ids_np.astype(np.int64, copy=False)).to(device=device)
    probe_xyz = torch.from_numpy(probe_xyz_np.astype(np.float32, copy=False)).to(device=device)
    probe_count = int(ids.shape[0])

    output = GaussianModel(source.max_sh_degree, use_SBs=source.use_SBs)
    output.active_sh_degree = int(source.active_sh_degree)
    output.spatial_lr_scale = float(source.spatial_lr_scale)

    probe_opacity = torch.clamp(source.get_opacity.detach()[ids] * float(opacity_scale), 1e-5, 0.999)
    probe_opacity_logits = inverse_sigmoid(probe_opacity)
    probe_scaling = source._scaling.detach()[ids].clone() + float(np.log(max(float(scale_multiplier), 1e-6)))
    probe_filter = _filter_3d_for_ids(source, ids)

    if keep_original:
        base_count = int(source.get_xyz.shape[0])
        output._xyz = nn.Parameter(torch.cat((source._xyz.detach().clone(), probe_xyz), dim=0), requires_grad=False)
        output._features_dc = nn.Parameter(
            torch.cat((source._features_dc.detach().clone(), source._features_dc.detach()[ids].clone()), dim=0),
            requires_grad=False,
        )
        output._features_rest = nn.Parameter(
            torch.cat((source._features_rest.detach().clone(), source._features_rest.detach()[ids].clone()), dim=0),
            requires_grad=False,
        )
        output._opacity = nn.Parameter(torch.cat((source._opacity.detach().clone(), probe_opacity_logits), dim=0), requires_grad=False)
        output._scaling = nn.Parameter(torch.cat((source._scaling.detach().clone(), probe_scaling), dim=0), requires_grad=False)
        output._rotation = nn.Parameter(
            torch.cat((source._rotation.detach().clone(), source._rotation.detach()[ids].clone()), dim=0),
            requires_grad=False,
        )
        if source.filter_3D.ndim > 0 and source.filter_3D.shape[0] == base_count:
            output.filter_3D = torch.cat((source.filter_3D.detach().clone(), probe_filter), dim=0)
        else:
            base_filter = source.filter_3D.detach().reshape(1, -1).repeat(base_count, 1)
            output.filter_3D = torch.cat((base_filter, probe_filter), dim=0)
        output.restore_tracking_state(_tracking_for_appended(source, base_count, probe_count))
    else:
        output._xyz = nn.Parameter(probe_xyz.detach().clone(), requires_grad=False)
        output._features_dc = nn.Parameter(source._features_dc.detach()[ids].clone(), requires_grad=False)
        output._features_rest = nn.Parameter(source._features_rest.detach()[ids].clone(), requires_grad=False)
        output._opacity = nn.Parameter(probe_opacity_logits.detach().clone(), requires_grad=False)
        output._scaling = nn.Parameter(probe_scaling.detach().clone(), requires_grad=False)
        output._rotation = nn.Parameter(source._rotation.detach()[ids].clone(), requires_grad=False)
        output.filter_3D = probe_filter.detach().clone()
        output.init_tracking_state(probe_count, source_tag=int(GaussianSourceTag.EXTENSION_PROBE), generation=1)

    output.max_radii2D = torch.zeros((output.get_xyz.shape[0],), dtype=torch.float32, device=device)
    return output


def main() -> None:
    parser = ArgumentParser(description="Apply VGGT-derived normal corrections to SOFLR Gaussians through LR-mesh binding.")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--correction_payload", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--output_iteration", type=int, default=30000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--apply_mode", choices=["residual_only", "bound_reconstruct", "probe_duplicate"], default="residual_only")
    parser.add_argument("--probe_position_mode", choices=["residual_only", "bound_reconstruct"], default="residual_only")
    parser.add_argument("--probe_only_model_path", default=None)
    parser.add_argument("--probe_opacity_scale", type=float, default=0.08)
    parser.add_argument("--probe_scale_multiplier", type=float, default=0.75)
    parser.add_argument("--probe_max_count", type=int, default=50000)
    parser.add_argument("--probe_seed", type=int, default=0)
    parser.add_argument("--normal_offset_shrink", type=float, default=1.0)
    parser.add_argument("--tangent_offset_shrink", type=float, default=1.0)
    parser.add_argument("--correction_scale", type=float, default=0.25)
    parser.add_argument("--min_correction_confidence", type=float, default=0.05)
    parser.add_argument("--min_correction_views", type=int, default=2)
    parser.add_argument(
        "--max_surface_distance",
        type=float,
        default=0.0,
        help="Only apply corrections to GS within this binding distance. 0 disables this gate.",
    )
    parser.add_argument("--max_correction_abs", type=float, default=0.0)
    args = parser.parse_args()

    base_model_path = Path(args.base_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    payload_path = Path(args.correction_payload).expanduser().resolve()
    iteration = resolve_iteration(base_model_path, int(args.base_iteration))

    source = load_gaussian_model(base_model_path, iteration, int(args.sh_degree))
    payload = np.load(payload_path)
    total = int(source.get_xyz.shape[0])
    if payload["surface_points"].shape[0] != total:
        raise ValueError(
            f"Payload has {payload['surface_points'].shape[0]} Gaussians, "
            f"but base model has {total}."
        )

    surface = payload["surface_points"].astype(np.float32)
    normals = payload["normals"].astype(np.float32)
    tangent_u = payload["tangent_u"].astype(np.float32)
    tangent_v = payload["tangent_v"].astype(np.float32)
    normal_offset = payload["normal_offset"].astype(np.float32)
    tangent_offset_u = payload["tangent_offset_u"].astype(np.float32)
    tangent_offset_v = payload["tangent_offset_v"].astype(np.float32)
    normal_correction = payload["normal_correction"].astype(np.float32).copy()
    confidence = payload["correction_confidence"].astype(np.float32)
    view_count = payload["correction_view_count"].astype(np.int32)
    surface_distance_before = payload["surface_distance"].astype(np.float32)
    original_xyz = source.get_xyz.detach().cpu().numpy().astype(np.float32)

    active = confidence >= float(args.min_correction_confidence)
    active &= view_count >= int(args.min_correction_views)
    if float(args.max_surface_distance) > 0.0:
        active &= surface_distance_before <= float(args.max_surface_distance)
    normal_correction[~active] = 0.0
    if float(args.max_correction_abs) > 0.0:
        normal_correction = np.clip(normal_correction, -float(args.max_correction_abs), float(args.max_correction_abs))

    applied_correction = (float(args.correction_scale) * normal_correction).astype(np.float32)
    if args.apply_mode == "residual_only":
        # Safety-first mode: keep the original SOFLR field intact and only add a gated residual.
        new_xyz = (original_xyz + applied_correction[:, None] * normals).astype(np.float32)
        probe_xyz_all = new_xyz
    else:
        tangent_offset = (
            float(args.tangent_offset_shrink) * tangent_offset_u[:, None] * tangent_u
            + float(args.tangent_offset_shrink) * tangent_offset_v[:, None] * tangent_v
        )
        corrected_normal_offset = (float(args.normal_offset_shrink) * normal_offset + applied_correction).astype(np.float32)
        bound_xyz = (surface + tangent_offset + corrected_normal_offset[:, None] * normals).astype(np.float32)
        if args.apply_mode == "bound_reconstruct":
            new_xyz = bound_xyz
            probe_xyz_all = bound_xyz
        else:
            new_xyz = original_xyz
            if args.probe_position_mode == "bound_reconstruct":
                probe_xyz_all = bound_xyz
            else:
                probe_xyz_all = (original_xyz + applied_correction[:, None] * normals).astype(np.float32)

    displacement = np.linalg.norm(new_xyz - original_xyz, axis=1).astype(np.float32)
    rel_after = new_xyz - surface
    normal_offset_after = np.sum(rel_after * normals, axis=1).astype(np.float32)
    tangent_residual_after = rel_after - normal_offset_after[:, None] * normals
    surface_distance_after = np.linalg.norm(rel_after, axis=1).astype(np.float32)
    tangent_distance_after = np.linalg.norm(tangent_residual_after, axis=1).astype(np.float32)
    surface_distance_delta = (surface_distance_after - surface_distance_before).astype(np.float32)

    probe_ids = np.flatnonzero(active).astype(np.int64, copy=False)
    if args.apply_mode == "probe_duplicate" and int(args.probe_max_count) > 0 and probe_ids.shape[0] > int(args.probe_max_count):
        rng = np.random.default_rng(int(args.probe_seed))
        probe_ids = np.sort(rng.choice(probe_ids, size=int(args.probe_max_count), replace=False)).astype(np.int64, copy=False)

    if args.apply_mode == "probe_duplicate":
        output = build_probe_augmented_model(
            source,
            probe_xyz_all[probe_ids],
            probe_ids,
            opacity_scale=float(args.probe_opacity_scale),
            scale_multiplier=float(args.probe_scale_multiplier),
            keep_original=True,
        )
    else:
        output = clone_with_xyz(source, new_xyz)
    copy_render_config(base_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
    mkdir_p(str(point_dir))
    output.save_ply(str(point_dir / "point_cloud.ply"))
    output.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    probe_only_output_ply = None
    if args.apply_mode == "probe_duplicate" and args.probe_only_model_path and probe_ids.shape[0] > 0:
        probe_only_model_path = Path(args.probe_only_model_path).expanduser().resolve()
        probe_only = build_probe_augmented_model(
            source,
            probe_xyz_all[probe_ids],
            probe_ids,
            opacity_scale=float(args.probe_opacity_scale),
            scale_multiplier=float(args.probe_scale_multiplier),
            keep_original=False,
        )
        copy_render_config(base_model_path, probe_only_model_path)
        probe_point_dir = probe_only_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
        mkdir_p(str(probe_point_dir))
        probe_only.save_ply(str(probe_point_dir / "point_cloud.ply"))
        probe_only.save_tracking_metadata(str(probe_point_dir / "gaussian_tags.pt"))
        with open(probe_point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
            json.dump({"num_gaussians": int(probe_only.get_xyz.shape[0])}, f, indent=2)
        probe_only_output_ply = str(probe_point_dir / "point_cloud.ply")

    summary = {
        "version": "apply_soflr_bound_gs_surface_correction_v0",
        "base_model_path": str(base_model_path),
        "base_iteration": int(iteration),
        "correction_payload": str(payload_path),
        "output_model_path": str(output_model_path),
        "output_iteration": int(args.output_iteration),
        "num_gaussians": total,
        "output_num_gaussians": int(output.get_xyz.shape[0]),
        "apply_mode": str(args.apply_mode),
        "probe_position_mode": str(args.probe_position_mode),
        "probe_opacity_scale": float(args.probe_opacity_scale),
        "probe_scale_multiplier": float(args.probe_scale_multiplier),
        "probe_max_count": int(args.probe_max_count),
        "probe_gaussians": int(probe_ids.shape[0]),
        "normal_offset_shrink": float(args.normal_offset_shrink),
        "tangent_offset_shrink": float(args.tangent_offset_shrink),
        "correction_scale": float(args.correction_scale),
        "min_correction_confidence": float(args.min_correction_confidence),
        "min_correction_views": int(args.min_correction_views),
        "max_surface_distance": float(args.max_surface_distance),
        "active_correction_gaussians": int(np.sum(active)),
        "raw_normal_correction_active": stats_from_array(normal_correction[active]) if np.any(active) else stats_from_array([]),
        "scaled_normal_correction_active": stats_from_array(applied_correction[active]) if np.any(active) else stats_from_array([]),
        "displacement": stats_from_array(displacement),
        "surface_distance_before": stats_from_array(surface_distance_before),
        "surface_distance_after": stats_from_array(surface_distance_after),
        "surface_distance_delta_after_minus_before": stats_from_array(surface_distance_delta),
        "surface_distance_improved_gaussians": int(np.sum(surface_distance_delta < 0.0)),
        "normal_offset_after": stats_from_array(normal_offset_after),
        "tangent_distance_after": stats_from_array(tangent_distance_after),
        "output_ply": str(point_dir / "point_cloud.ply"),
        "probe_only_output_ply": probe_only_output_ply,
    }
    (output_model_path / "apply_soflr_bound_gs_surface_correction_v0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": total, **summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
