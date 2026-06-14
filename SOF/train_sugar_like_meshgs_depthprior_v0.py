import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render_simple
from scene.sugar_like_meshgs_model import SugarLikeMeshGaussianModel, _load_mesh_np
from train_alternating_prior_surface_v0 import (
    ImageCache,
    aggregate_surface_targets,
    build_local_view_records,
    clamp_surface_vertex_displacement,
    load_patch_bank,
    load_view_records,
    merge_geometry_fields,
    resolve_initial_colors,
    select_payload_indices,
)
from train_meshgs_prior_v0 import load_mask_cached, load_rgb_cached, masked_l1, render_alpha_mask
from train_mip_to_sof_surface_v0 import (
    charbonnier,
    compute_depth_prior_distortion_loss,
    compute_depth_prior_self_normal_loss,
    copy_render_config,
    load_cameras_for_split,
    load_depth_prior_for_view,
    masked_weighted_mean,
    normalize_normal,
    parse_subdir_list,
    robust_align_depth_to_reference,
    scheduled_loss_scale,
    select_uniform,
)
from utils.depth_utils import depth_to_normal
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir, normalize_image_name
from utils.system_utils import mkdir_p


def save_checkpoint(
    meshgs: SugarLikeMeshGaussianModel,
    output_dir: Path,
    iteration: int,
) -> None:
    point_dir = output_dir / "point_cloud" / f"iteration_{int(iteration)}"
    mkdir_p(str(point_dir))

    classic = meshgs.materialize_gaussian_model()
    classic.save_ply(str(point_dir / "point_cloud.ply"))
    classic.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    meshgs.save_bound_metadata(str(point_dir / "mesh_bound_state.pt"))

    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(classic.get_xyz.shape[0])}, f, indent=2)


def save_artifacts(
    *,
    output_dir: Path,
    args,
    summary: Dict[str, object],
) -> None:
    with open(output_dir / "sugar_like_meshgs_depthprior_v0_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(output_dir / "sugar_like_meshgs_depthprior_v0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_init_payload(
    *,
    mesh_path: str,
    patch_bank: Dict[str, np.ndarray],
    aggregated_payload: Dict[str, np.ndarray],
    selected_indices: np.ndarray,
    args,
) -> Tuple[SugarLikeMeshGaussianModel, Dict[str, object]]:
    if selected_indices.size == 0:
        raise RuntimeError("No valid surface carriers remain after LR aggregation filters.")

    vertices, faces = _load_mesh_np(mesh_path)
    colors, color_summary = resolve_initial_colors(
        aggregated_payload,
        selected_indices,
        init_color_source=str(args.init_color_source),
        init_color_gray_value=float(args.init_color_gray_value),
    )
    confidence = np.clip(aggregated_payload["confidence"][selected_indices].astype(np.float32), 0.0, 1.0)
    disagreement = aggregated_payload["disagreement"][selected_indices].astype(np.float32)
    disagreement_gate = 1.0 - np.clip(disagreement / max(float(args.max_disagreement), 1e-8), 0.0, 1.0)
    opacity = np.clip(float(args.meshgs_init_opacity) * (0.25 + 0.75 * confidence * disagreement_gate), 1e-4, 0.95)

    meshgs = SugarLikeMeshGaussianModel(int(args.sh_degree), use_SBs=False)
    meshgs.initialize_from_arrays(
        vertices=vertices,
        faces=faces,
        face_ids=patch_bank["face_ids"][selected_indices].astype(np.int64, copy=False),
        bary_coords=patch_bank["bary_coords"][selected_indices].astype(np.float32, copy=False),
        colors=colors,
        scale_u=patch_bank["scale_u"][selected_indices].astype(np.float32, copy=False) * float(args.meshgs_scale_multiplier),
        scale_v=patch_bank["scale_v"][selected_indices].astype(np.float32, copy=False) * float(args.meshgs_scale_multiplier),
        scale_n=patch_bank["scale_n"][selected_indices].astype(np.float32, copy=False)
        * float(args.meshgs_scale_multiplier)
        * float(args.meshgs_thickness_multiplier),
        opacity=opacity,
        learn_surface_vertices=bool(args.learn_surface_vertices),
        learn_plane_scales=bool(args.learn_plane_scales),
        learn_inplane_rotation=bool(args.learn_inplane_rotation),
        build_normal_pairs=not bool(args.disable_normal_consistency_pairs),
        max_normal_pairs=int(args.max_normal_consistency_pairs),
    )
    summary = {
        "payload_count": int(aggregated_payload["valid_mask"].shape[0]),
        "selected_count": int(selected_indices.size),
        "selected_ratio": float(selected_indices.size / max(int(aggregated_payload["valid_mask"].shape[0]), 1)),
        "normal_consistency_pairs": int(meshgs._normal_consistency_pairs.shape[0]),
        "learn_surface_vertices": bool(args.learn_surface_vertices),
        "learn_plane_scales": bool(args.learn_plane_scales),
        "learn_inplane_rotation": bool(args.learn_inplane_rotation),
    }
    summary.update(color_summary)
    return meshgs, summary


@torch.no_grad()
def build_training_targets(
    *,
    cameras: Sequence[object],
    meshgs_init: SugarLikeMeshGaussianModel,
    background: torch.Tensor,
    prior_index: Dict[str, Path],
    mask_dir: Optional[str],
    render_alpha_threshold: float,
    min_loss_pixels: float,
    depth_prior_root: Path | None,
    depth_prior_subdirs: Sequence[str],
    depth_prior_confidence_subdirs: Sequence[str],
    depth_prior_confidence_min: float,
    depth_prior_agreement_threshold: float,
    depth_prior_agreement_floor: float,
    depth_prior_align_mode: str,
    depth_prior_align_min_pixels: int,
    depth_prior_surface_weight_boost: float,
    depth_prior_weight_gain: float,
    depth_prior_weight_power: float,
    depth_prior_weight_min: float,
) -> Tuple[List[object], List[Dict[str, object]], Dict[str, float]]:
    prior_cache: Dict[Tuple[str, int, int], torch.Tensor | None] = {}
    mask_cache: Dict[Tuple[str, int, int], torch.Tensor | None] = {}
    cached_cameras: List[object] = []
    target_cache: List[Dict[str, object]] = []
    mask_ratios: List[float] = []
    depth_prior_ratios: List[float] = []

    for idx, camera in enumerate(cameras, start=1):
        render_pkg = render_simple(camera, meshgs_init, background)
        rgb = render_pkg["render"].clamp(0.0, 1.0)
        depth = render_pkg["depth"].detach()
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        prior_image = load_rgb_cached(camera.image_name, prior_index, prior_cache, height, width)
        if prior_image is None:
            print(
                f"[sugar-like-depthprior] cache skip {idx}/{len(cameras)} missing_prior view={camera.image_name}",
                flush=True,
            )
            continue

        if mask_dir:
            rgb_mask = load_mask_cached(camera.image_name, mask_dir, mask_cache, height, width)
        else:
            rgb_mask = render_alpha_mask(render_pkg, rgb, float(render_alpha_threshold))
        if rgb_mask is None or float(rgb_mask.sum().item()) < float(min_loss_pixels):
            print(
                f"[sugar-like-depthprior] cache skip {idx}/{len(cameras)} weak_mask view={camera.image_name}",
                flush=True,
            )
            continue

        surface_mask = rgb_mask.to(device=depth.device, dtype=torch.bool) & torch.isfinite(depth[0]) & (depth[0] > 1e-6)
        surface_weight = torch.ones_like(surface_mask, dtype=torch.float32, device=depth.device)

        depth_prior_mask = torch.zeros_like(surface_mask, dtype=torch.bool)
        depth_prior_weight = torch.zeros_like(surface_weight)
        depth_prior_target = None
        depth_prior_normal = None
        depth_prior_info: Dict[str, object] = {"status": "disabled"}
        if depth_prior_root is not None:
            raw_depth_prior, raw_depth_conf, depth_prior_info = load_depth_prior_for_view(
                camera.image_name,
                depth_prior_root=depth_prior_root,
                target_hw=(height, width),
                depth_subdirs=depth_prior_subdirs,
                confidence_subdirs=depth_prior_confidence_subdirs,
            )
            if raw_depth_prior is not None:
                depth_prior = raw_depth_prior.to(device=depth.device, dtype=torch.float32)
                if raw_depth_conf is None:
                    depth_prior_conf = torch.ones_like(depth_prior)
                else:
                    depth_prior_conf = raw_depth_conf.to(device=depth.device, dtype=torch.float32).clamp(0.0, 1.0)

                align_seed_mask = surface_mask & (depth_prior_conf >= float(depth_prior_confidence_min))
                if str(depth_prior_align_mode) == "affine_robust":
                    aligned_depth, align_summary = robust_align_depth_to_reference(
                        depth_prior,
                        depth[0],
                        align_seed_mask,
                        min_pixels=int(depth_prior_align_min_pixels),
                    )
                elif str(depth_prior_align_mode) == "identity":
                    aligned_depth = depth_prior
                    align_summary = {
                        "mode": "identity",
                        "pixels": int(align_seed_mask.detach().sum().item()),
                        "scale": 1.0,
                        "shift": 0.0,
                    }
                else:
                    raise ValueError(f"Unsupported depth prior align mode: {depth_prior_align_mode}")
                depth_prior_info["align"] = align_summary

                depth_valid = torch.isfinite(aligned_depth) & (aligned_depth > 1e-6) & (
                    depth_prior_conf >= float(depth_prior_confidence_min)
                )
                if float(depth_prior_agreement_threshold) > 0.0:
                    agreement = torch.abs(aligned_depth - depth[0]) / torch.clamp(depth[0], min=1e-6)
                    agreement_conf = torch.clamp(
                        1.0 - agreement / float(depth_prior_agreement_threshold),
                        min=0.0,
                        max=1.0,
                    )
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
                depth_prior_weight = torch.where(
                    surface_mask & depth_valid,
                    depth_prior_weight,
                    torch.zeros_like(depth_prior_weight),
                )
                depth_prior_mask = depth_prior_weight > 0.0
                depth_prior_target = aligned_depth.unsqueeze(0).detach()
                if bool(depth_prior_mask.any()):
                    depth_normal_hw3, _ = depth_to_normal(camera, aligned_depth.unsqueeze(0))
                    depth_prior_normal = normalize_normal(depth_normal_hw3.permute(2, 0, 1).detach())
                if float(depth_prior_surface_weight_boost) > 0.0:
                    surface_weight = surface_weight + float(depth_prior_surface_weight_boost) * depth_prior_weight

        surface_weight = surface_weight.clamp(0.0, 1.0 + max(float(depth_prior_surface_weight_boost), 0.0))
        cached_cameras.append(camera)
        target_cache.append(
            {
                "image_name": normalize_image_name(str(camera.image_name)),
                "prior_rgb": prior_image.detach().cpu(),
                "rgb_mask": rgb_mask.detach().cpu(),
                "surface_mask": surface_mask.detach().cpu(),
                "surface_weight": surface_weight.detach().cpu(),
                "depth_prior_target": depth_prior_target.detach().cpu() if torch.is_tensor(depth_prior_target) else None,
                "depth_prior_normal": depth_prior_normal.detach().cpu() if torch.is_tensor(depth_prior_normal) else None,
                "depth_prior_mask": depth_prior_mask.detach().cpu(),
                "depth_prior_weight": depth_prior_weight.detach().cpu(),
                "depth_prior_info": depth_prior_info,
            }
        )
        mask_ratio = float(surface_mask.float().mean().item())
        depth_ratio = float(depth_prior_mask.float().mean().item())
        mask_ratios.append(mask_ratio)
        depth_prior_ratios.append(depth_ratio)
        print(
            f"[sugar-like-depthprior] cached view {idx}/{len(cameras)} mask={mask_ratio:.4f} depth_prior={depth_ratio:.4f}",
            flush=True,
        )

    if not target_cache:
        raise RuntimeError("No valid training targets were cached. Check LR prior dir and masks.")

    stats = {
        "cached_views": int(len(target_cache)),
        "mean_surface_mask": float(np.mean(mask_ratios)) if mask_ratios else 0.0,
        "mean_depth_prior_mask": float(np.mean(depth_prior_ratios)) if depth_prior_ratios else 0.0,
    }
    return cached_cameras, target_cache, stats


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Validate sugar-like meshGS training from LR patch init plus depth prior.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--camera_model_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--patch_observation_root", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--anchor_dir", default=None)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--copy_render_config_from", default=None)
    parser.add_argument("--images_subdir", default="images_8")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.02)
    parser.add_argument("--max_disagreement", type=float, default=0.20)
    parser.add_argument("--max_count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--anchor_lowfreq_threshold", type=float, default=0.0)
    parser.add_argument("--anchor_lowfreq_kernel", type=int, default=15)

    parser.add_argument("--meshgs_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--meshgs_thickness_multiplier", type=float, default=0.5)
    parser.add_argument("--meshgs_init_opacity", type=float, default=0.35)
    parser.add_argument("--meshgs_feature_lr", type=float, default=0.01)
    parser.add_argument("--meshgs_opacity_lr", type=float, default=0.02)
    parser.add_argument("--meshgs_lambda_opacity", type=float, default=1e-4)
    parser.add_argument("--meshgs_min_pixels", type=float, default=64.0)
    parser.add_argument("--meshgs_render_alpha_threshold", type=float, default=1e-4)
    parser.add_argument("--mesh_fusion_mask_dir", type=str, default=None)
    parser.add_argument("--init_color_source", choices=["fused_rgb", "anchor_rgb", "neutral_gray"], default="fused_rgb")
    parser.add_argument("--init_color_gray_value", type=float, default=0.5)

    parser.add_argument("--learn_surface_vertices", action="store_true")
    parser.add_argument("--learn_plane_scales", action="store_true")
    parser.add_argument("--learn_inplane_rotation", action="store_true")
    parser.add_argument("--surface_vertex_lr", type=float, default=5e-4)
    parser.add_argument("--plane_scale_lr", type=float, default=0.0)
    parser.add_argument("--inplane_rotation_lr", type=float, default=0.0)
    parser.add_argument("--lambda_surface_delta", type=float, default=0.02)
    parser.add_argument("--max_surface_vertex_displacement", type=float, default=0.02)
    parser.add_argument("--normal_consistency_lambda", type=float, default=0.02)
    parser.add_argument("--disable_normal_consistency_pairs", action="store_true")
    parser.add_argument("--max_normal_consistency_pairs", type=int, default=500000)

    parser.add_argument("--depth_relative_min", type=float, default=1e-3)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--min_loss_pixels", type=float, default=64.0)

    parser.add_argument("--depth_prior_root", type=str, default=None)
    parser.add_argument("--depth_prior_subdirs", type=str, default="depth,")
    parser.add_argument("--depth_prior_confidence_subdirs", type=str, default="auto")
    parser.add_argument("--depth_prior_confidence_min", type=float, default=0.05)
    parser.add_argument("--depth_prior_agreement_threshold", type=float, default=0.15)
    parser.add_argument("--depth_prior_agreement_floor", type=float, default=0.0)
    parser.add_argument("--depth_prior_align_mode", choices=["affine_robust", "identity"], default="affine_robust")
    parser.add_argument("--depth_prior_align_min_pixels", type=int, default=2048)
    parser.add_argument("--depth_prior_surface_weight_boost", type=float, default=0.25)
    parser.add_argument("--depth_prior_weight_gain", type=float, default=1.0)
    parser.add_argument("--depth_prior_weight_power", type=float, default=1.0)
    parser.add_argument("--depth_prior_weight_min", type=float, default=0.0)
    parser.add_argument("--lambda_depth_prior", type=float, default=0.10)
    parser.add_argument("--lambda_depth_prior_normal", type=float, default=0.03)
    parser.add_argument("--lambda_depth_prior_distortion", type=float, default=100.0)
    parser.add_argument("--lambda_depth_prior_self_normal", type=float, default=0.05)
    parser.add_argument("--depth_prior_warmup_start_iter", type=int, default=0)
    parser.add_argument("--depth_prior_warmup_end_iter", type=int, default=4000)
    parser.add_argument("--depth_prior_start_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_end_scale", type=float, default=2.0)
    parser.add_argument("--depth_prior_update_scale", type=float, default=1.0)
    parser.add_argument("--depth_prior_schedule_mode", choices=["linear", "smoothstep"], default="smoothstep")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    safe_state(bool(args.quiet))

    if not torch.cuda.is_available():
        raise RuntimeError("train_sugar_like_meshgs_depthprior_v0 currently requires CUDA.")

    scene_root = Path(args.scene_root).expanduser().resolve()
    camera_model_path = Path(args.camera_model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    patch_root = Path(args.patch_observation_root).expanduser().resolve()
    prior_dir = Path(args.prior_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve() if args.anchor_dir else None
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    patch_bank_path = patch_root / "mesh_patch_bank_v0.npz"
    observation_dir = patch_root / "camera_patch_observations"
    if not scene_root.is_dir():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")
    if not camera_model_path.is_dir():
        raise FileNotFoundError(f"camera_model_path not found: {camera_model_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh_path not found: {mesh_path}")
    if not patch_bank_path.is_file():
        raise FileNotFoundError(f"patch bank not found: {patch_bank_path}")
    if not observation_dir.is_dir():
        raise FileNotFoundError(f"patch observation dir not found: {observation_dir}")
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_dir not found: {prior_dir}")
    if anchor_dir is not None and not anchor_dir.is_dir():
        raise FileNotFoundError(f"anchor_dir not found: {anchor_dir}")

    render_config_src = (
        Path(args.copy_render_config_from).expanduser().resolve()
        if args.copy_render_config_from
        else camera_model_path
    )
    if render_config_src.is_dir():
        copy_render_config(render_config_src, output_model_path)

    cameras_all = load_cameras_for_split(
        scene_root,
        camera_model_path,
        str(args.images_subdir),
        str(args.split),
    )
    patch_bank = load_patch_bank(patch_bank_path)
    prior_index = index_image_dir(str(prior_dir))
    anchor_index = index_image_dir(str(anchor_dir)) if anchor_dir is not None else None
    matched_view_records = load_view_records(
        observation_root=observation_dir,
        cameras=cameras_all,
        prior_index=prior_index,
        anchor_index=anchor_index,
    )
    full_view_records = select_uniform(matched_view_records, int(args.max_views))
    if not full_view_records:
        raise RuntimeError("No matched observation/prior views selected for validation training.")
    selected_cameras = [record["camera"] for record in full_view_records]
    print(f"[sugar-like-depthprior] selected views : {len(selected_cameras)}")

    device = torch.device("cuda")
    image_cache = ImageCache(device=device, blur_kernel=int(args.anchor_lowfreq_kernel))
    patch_centers = torch.from_numpy(np.asarray(patch_bank["centers"], dtype=np.float32)).to(device=device)
    initial_payload, initial_summary = aggregate_surface_targets(
        xyz_world=patch_centers,
        view_records=build_local_view_records(
            full_view_records,
            np.arange(patch_centers.shape[0], dtype=np.int64),
            device=device,
        ),
        image_cache=image_cache,
        min_views=int(args.min_views),
        min_confidence=float(args.min_confidence),
        max_disagreement=float(args.max_disagreement),
        depth_min=float(args.depth_min),
        anchor_lowfreq_threshold=float(args.anchor_lowfreq_threshold),
    )
    initial_payload = merge_geometry_fields(initial_payload, patch_bank)
    initial_payload_path = output_model_path / "carrier_payload_init_lr_depthprior_v0.npz"
    np.savez_compressed(initial_payload_path, **initial_payload)

    selected_indices = select_payload_indices(
        initial_payload,
        min_confidence=float(args.min_confidence),
        max_disagreement=float(args.max_disagreement),
        min_views=int(args.min_views),
        max_count=int(args.max_count),
        seed=int(args.seed),
    )
    meshgs, init_summary = build_init_payload(
        mesh_path=str(mesh_path),
        patch_bank=patch_bank,
        aggregated_payload=initial_payload,
        selected_indices=selected_indices,
        args=args,
    )
    del image_cache
    torch.cuda.empty_cache()

    optimizer = meshgs.build_optimizer(
        feature_lr=float(args.meshgs_feature_lr),
        opacity_lr=float(args.meshgs_opacity_lr),
        surface_vertex_lr=float(args.surface_vertex_lr),
        plane_scale_lr=float(args.plane_scale_lr),
        inplane_rotation_lr=float(args.inplane_rotation_lr),
    )
    surface_vertices_init = meshgs._surface_vertices.detach().clone()

    depth_prior_root = Path(args.depth_prior_root).expanduser().resolve() if args.depth_prior_root else None
    if depth_prior_root is not None and not depth_prior_root.is_dir():
        raise FileNotFoundError(f"depth_prior_root not found: {depth_prior_root}")
    depth_prior_subdirs = parse_subdir_list(args.depth_prior_subdirs, default_auto=("depth", ""))
    depth_prior_confidence_subdirs = parse_subdir_list(
        args.depth_prior_confidence_subdirs,
        default_auto=("confidence", "conf", "depth_conf", "valid"),
    )

    background = torch.zeros((3,), dtype=torch.float32, device=device)
    selected_cameras, target_cache, cache_stats = build_training_targets(
        cameras=selected_cameras,
        meshgs_init=meshgs,
        background=background,
        prior_index=prior_index,
        mask_dir=args.mesh_fusion_mask_dir,
        render_alpha_threshold=float(args.meshgs_render_alpha_threshold),
        min_loss_pixels=float(args.min_loss_pixels),
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
    print(
        "[sugar-like-depthprior] init meshGS    : "
        f"{init_summary['selected_count']}/{init_summary['payload_count']} carriers",
        flush=True,
    )
    print(
        "[sugar-like-depthprior] depth prior    : "
        f"{depth_prior_root if depth_prior_root is not None else '(disabled)'}",
        flush=True,
    )
    if depth_prior_root is not None:
        print(
            "[sugar-like-depthprior] depth cfg      : "
            f"l1={float(args.lambda_depth_prior):.4g} "
            f"prior_n={float(args.lambda_depth_prior_normal):.4g} "
            f"distort={float(args.lambda_depth_prior_distortion):.4g} "
            f"self_n={float(args.lambda_depth_prior_self_normal):.4g} "
            f"agree={float(args.depth_prior_agreement_threshold):.4g} "
            f"floor={float(args.depth_prior_agreement_floor):.4g} "
            f"align={args.depth_prior_align_mode}",
            flush=True,
        )

    save_iterations = set(int(item) for item in args.save_iterations)
    if int(args.save_every) > 0:
        save_iterations.update(range(int(args.save_every), int(args.iterations) + 1, int(args.save_every)))
    save_iterations.add(int(args.iterations))
    save_iterations = {item for item in save_iterations if 0 < item <= int(args.iterations)}

    progress = tqdm(range(1, int(args.iterations) + 1), desc="train sugar-like depthprior")
    ema_total = 0.0
    ema_rgb = 0.0
    ema_dp = 0.0
    ema_dpn = 0.0
    ema_dist = 0.0
    ema_selfn = 0.0
    ema_geom = 0.0
    last_metrics: Dict[str, float] = {}
    log_rows: List[Dict[str, float]] = []

    for iteration in progress:
        view_idx = randint(0, len(selected_cameras) - 1)
        camera = selected_cameras[view_idx]
        target = target_cache[view_idx]

        render_pkg = render_simple(camera, meshgs, background)
        rgb = render_pkg["render"].clamp(0.0, 1.0)
        depth = render_pkg["depth"]
        normal = normalize_normal(render_pkg["normal"])

        prior_image = target["prior_rgb"]
        rgb_mask = target["rgb_mask"]
        loss_rgb = masked_l1(rgb, prior_image, rgb_mask)
        if loss_rgb is None:
            continue

        loss = loss_rgb

        depth_prior_scale = scheduled_loss_scale(
            iteration,
            start_iter=int(args.depth_prior_warmup_start_iter),
            end_iter=int(args.depth_prior_warmup_end_iter),
            start_scale=float(args.depth_prior_start_scale),
            end_scale=float(args.depth_prior_end_scale),
            update_scale=float(args.depth_prior_update_scale),
            mode=str(args.depth_prior_schedule_mode),
        )

        depth_prior_mask = target["depth_prior_mask"].to(device=device, dtype=torch.bool)
        depth_prior_weight = target["depth_prior_weight"].to(device=device, dtype=torch.float32)
        depth_prior_loss = torch.zeros((), dtype=torch.float32, device=device)
        if float(args.lambda_depth_prior) > 0.0 and target["depth_prior_target"] is not None:
            prior_depth_target = target["depth_prior_target"].to(device=device, dtype=torch.float32)
            prior_depth_rel = (depth - prior_depth_target) / torch.clamp(
                prior_depth_target,
                min=float(args.depth_relative_min),
            )
            maybe_depth_prior_loss = masked_weighted_mean(
                charbonnier(prior_depth_rel, float(args.charbonnier_eps)),
                depth_prior_mask,
                float(args.min_loss_pixels),
                depth_prior_weight,
            )
            if maybe_depth_prior_loss is not None:
                depth_prior_loss = maybe_depth_prior_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior) * depth_prior_loss

        depth_prior_normal_loss = torch.zeros((), dtype=torch.float32, device=device)
        if float(args.lambda_depth_prior_normal) > 0.0 and target["depth_prior_normal"] is not None:
            prior_normal_target = target["depth_prior_normal"].to(device=device, dtype=torch.float32)
            prior_normal_dot = torch.sum(normal * prior_normal_target, dim=0).clamp(-1.0, 1.0)
            maybe_prior_normal_loss = masked_weighted_mean(
                1.0 - prior_normal_dot,
                depth_prior_mask,
                float(args.min_loss_pixels),
                depth_prior_weight,
            )
            if maybe_prior_normal_loss is not None:
                depth_prior_normal_loss = maybe_prior_normal_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_normal) * depth_prior_normal_loss

        render_target = {
            "depth_prior_mask": depth_prior_mask,
            "depth_prior_weight": depth_prior_weight,
        }
        depth_prior_distortion_loss = torch.zeros((), dtype=torch.float32, device=device)
        if float(args.lambda_depth_prior_distortion) > 0.0 and target["depth_prior_target"] is not None:
            maybe_prior_distortion_loss = compute_depth_prior_distortion_loss(
                render_pkg,
                render_target,
                min_pixels=float(args.min_loss_pixels),
            )
            if maybe_prior_distortion_loss is not None:
                depth_prior_distortion_loss = maybe_prior_distortion_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_distortion) * depth_prior_distortion_loss

        depth_prior_self_normal_loss = torch.zeros((), dtype=torch.float32, device=device)
        if float(args.lambda_depth_prior_self_normal) > 0.0 and target["depth_prior_target"] is not None:
            maybe_prior_self_normal_loss = compute_depth_prior_self_normal_loss(
                camera,
                render_pkg,
                render_target,
                min_pixels=float(args.min_loss_pixels),
            )
            if maybe_prior_self_normal_loss is not None:
                depth_prior_self_normal_loss = maybe_prior_self_normal_loss
                loss = loss + depth_prior_scale * float(args.lambda_depth_prior_self_normal) * depth_prior_self_normal_loss

        loss_geom = torch.zeros((), dtype=torch.float32, device=device)
        if float(args.normal_consistency_lambda) > 0.0:
            geom_term = meshgs.normal_consistency_loss()
            loss_geom = loss_geom + float(args.normal_consistency_lambda) * geom_term
        if float(args.lambda_surface_delta) > 0.0 and bool(args.learn_surface_vertices):
            delta = meshgs._surface_vertices - surface_vertices_init
            delta_term = delta.pow(2).sum(dim=1).mean()
            loss_geom = loss_geom + float(args.lambda_surface_delta) * delta_term
        if float(args.meshgs_lambda_opacity) > 0.0:
            loss_geom = loss_geom + float(args.meshgs_lambda_opacity) * meshgs.get_opacity.mean()
        loss = loss + loss_geom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if bool(args.learn_surface_vertices):
            clamp_surface_vertex_displacement(
                meshgs,
                surface_vertices_init,
                float(args.max_surface_vertex_displacement),
            )
        if bool(args.learn_inplane_rotation):
            with torch.no_grad():
                norm = torch.linalg.norm(meshgs._surface_inplane_rotation, dim=-1, keepdim=True).clamp_min(1e-8)
                meshgs._surface_inplane_rotation.div_(norm)

        ema_total = 0.4 * float(loss.item()) + 0.6 * ema_total
        ema_rgb = 0.4 * float(loss_rgb.item()) + 0.6 * ema_rgb
        ema_dp = 0.4 * float(depth_prior_loss.item()) + 0.6 * ema_dp
        ema_dpn = 0.4 * float(depth_prior_normal_loss.item()) + 0.6 * ema_dpn
        ema_dist = 0.4 * float(depth_prior_distortion_loss.item()) + 0.6 * ema_dist
        ema_selfn = 0.4 * float(depth_prior_self_normal_loss.item()) + 0.6 * ema_selfn
        ema_geom = 0.4 * float(loss_geom.item()) + 0.6 * ema_geom
        last_metrics = {
            "loss": float(loss.item()),
            "rgb": float(loss_rgb.item()),
            "depth_prior": float(depth_prior_loss.item()),
            "depth_prior_normal": float(depth_prior_normal_loss.item()),
            "depth_prior_distortion": float(depth_prior_distortion_loss.item()),
            "depth_prior_self_normal": float(depth_prior_self_normal_loss.item()),
            "geom": float(loss_geom.item()),
            "depth_prior_scale": float(depth_prior_scale),
        }
        if iteration % 10 == 0:
            progress.set_postfix(
                {
                    "loss": f"{ema_total:.6f}",
                    "rgb": f"{ema_rgb:.6f}",
                    "dp": f"{ema_dp:.6f}",
                    "geom": f"{ema_geom:.6f}",
                }
            )
        if iteration % 100 == 0 or iteration == 1 or iteration == int(args.iterations):
            log_rows.append({"iteration": int(iteration), **last_metrics})
        if iteration in save_iterations:
            save_checkpoint(meshgs, output_model_path, iteration)

    output_iteration = int(args.iterations)
    save_checkpoint(meshgs, output_model_path, output_iteration)
    selected_names = [normalize_image_name(str(camera.image_name)) for camera in selected_cameras]
    (output_model_path / "selected_global_patch_ids.json").write_text(
        json.dumps(selected_indices.tolist(), indent=2),
        encoding="utf-8",
    )
    summary = {
        "version": "sugar_like_meshgs_depthprior_v0",
        "scene_root": str(scene_root),
        "camera_model_path": str(camera_model_path),
        "mesh_path": str(mesh_path),
        "patch_observation_root": str(patch_root),
        "prior_dir": str(prior_dir),
        "anchor_dir": str(anchor_dir) if anchor_dir is not None else None,
        "depth_prior_root": str(depth_prior_root) if depth_prior_root is not None else None,
        "output_model_path": str(output_model_path),
        "inputs": {
            "images_subdir": str(args.images_subdir),
            "split": str(args.split),
            "requested_max_views": int(args.max_views),
            "selected_views": selected_names,
        },
        "selection": {
            "initial_aggregation": initial_summary,
            "selected_global_carriers": int(selected_indices.size),
            "selected_global_indices_path": str(output_model_path / "selected_global_patch_ids.json"),
            **init_summary,
        },
        "target_cache": cache_stats,
        "params": vars(args),
        "training": {
            "iterations": int(args.iterations),
            "last_metrics": last_metrics,
            "ema": {
                "loss": float(ema_total),
                "rgb": float(ema_rgb),
                "depth_prior": float(ema_dp),
                "depth_prior_normal": float(ema_dpn),
                "depth_prior_distortion": float(ema_dist),
                "depth_prior_self_normal": float(ema_selfn),
                "geom": float(ema_geom),
            },
            "log_rows": log_rows,
        },
        "artifacts": {
            "initial_payload": str(initial_payload_path),
            "final_point_cloud_dir": str(output_model_path / "point_cloud" / f"iteration_{output_iteration}"),
            "final_point_cloud": str(output_model_path / "point_cloud" / f"iteration_{output_iteration}" / "point_cloud.ply"),
        },
    }
    save_artifacts(output_dir=output_model_path, args=args, summary=summary)
    print(json.dumps(summary, indent=2))
    print(f"[sugar-like-depthprior] summary : {output_model_path / 'sugar_like_meshgs_depthprior_v0_summary.json'}")
    print(f"[sugar-like-depthprior] output  : {output_model_path / 'point_cloud' / f'iteration_{output_iteration}' / 'point_cloud.ply'}")


if __name__ == "__main__":
    main()
