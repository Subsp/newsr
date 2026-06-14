import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch

from arguments import (
    MeshingParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    SplattingSettings,
    get_combined_args,
)
from scene import Scene
from scene.appearance_network import AppearanceEmbedding, PGSREmbedding
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.prior_injection import (
    InjectionSummary,
    PriorSeedSpec,
    ViewInjectionSummary,
    build_auto_injection_mask,
    build_normal_bundle,
    generate_grid_candidates,
    generate_mask_candidates,
    group_seeds_by_view,
    infer_lr_dir_from_pseudo_scene,
    index_image_dir,
    load_rgb_image,
    load_mask,
    load_prior_image,
    load_seed_manifest,
    resize_image_hw3,
    render_view_state,
    sample_anchor_from_view,
    sample_prior_patch_rgb,
    save_injection_summary,
    compute_surface_support_strength,
    estimate_local_surface_scales,
)

LEGACY_INJECTION_DEFAULTS = {
    "bundle_size": 3,
    "bundle_spacing_scale": 1.0,
    "center_opacity": 0.08,
    "min_injected_opacity": 0.02,
    "side_opacity_scale": 0.6,
}

INJECTION_PRESETS = {
    "legacy_triplet": dict(LEGACY_INJECTION_DEFAULTS),
    # Make a single anchored prior probe that comfortably clears the default
    # prune threshold before we let SOF decide whether it deserves to survive.
    "sticky_single_anchor": {
        "bundle_size": 1,
        "bundle_spacing_scale": 0.5,
        "center_opacity": 0.18,
        "min_injected_opacity": 0.10,
        "side_opacity_scale": 0.85,
    },
    # A denser, still surface-anchored prior bundle for SR-style experiments
    # where we intentionally want many more injected Gaussians than the
    # conservative sticky-single setup.
    "surface_dense_triplet": {
        "bundle_size": 3,
        "bundle_spacing_scale": 0.4,
        "center_opacity": 0.12,
        "min_injected_opacity": 0.04,
        "side_opacity_scale": 0.5,
    },
}


def build_appearance_embedding(mesh_args, num_views: int):
    if mesh_args.use_decoupled_appearance:
        return AppearanceEmbedding(num_views=num_views)
    if mesh_args.use_pgsr_appearance:
        return PGSREmbedding(num_views=num_views)
    return None


def resolve_start_checkpoint(model_path: str, start_checkpoint: str, iteration: int) -> str:
    if start_checkpoint:
        return start_checkpoint
    if iteration < 0:
        raise ValueError("iteration must be explicit when start_checkpoint is not provided.")
    return os.path.join(model_path, f"chkpnt{iteration}.pth")


def build_seed_map(views, prior_index: Dict[str, Path], args) -> Dict[str, List[PriorSeedSpec]]:
    if args.candidate_mode == "manifest":
        if not args.seed_manifest:
            raise ValueError("--seed_manifest is required when --candidate_mode=manifest")
        return group_seeds_by_view(load_seed_manifest(args.seed_manifest))

    if args.candidate_mode == "auto_mask":
        return {}

    seed_map: Dict[str, List[PriorSeedSpec]] = {}
    mask_index = index_image_dir(args.mask_dir) if args.candidate_mode == "mask" else {}
    processed_views = 0
    for view in views:
        if args.view_limit > 0 and processed_views >= args.view_limit:
            break
        image_name = view.image_name
        if image_name not in prior_index:
            continue

        if args.candidate_mode == "grid":
            seeds = generate_grid_candidates(
                image_name=image_name,
                width=view.image_width,
                height=view.image_height,
                stride=args.grid_stride,
                border=max(args.grid_border, args.patch_radius),
                max_seeds=args.max_seeds_per_view,
            )
        elif args.candidate_mode == "mask":
            mask_path = mask_index.get(image_name)
            if mask_path is None:
                continue
            mask = load_mask(mask_path)
            seeds = generate_mask_candidates(
                image_name=image_name,
                mask=mask,
                stride=args.grid_stride,
                border=max(args.grid_border, args.patch_radius),
                max_seeds=args.max_seeds_per_view,
            )
        else:
            raise ValueError(f"Unsupported candidate mode: {args.candidate_mode}")

        for seed in seeds:
            seed.patch_radius = args.patch_radius
        seed_map[image_name] = seeds
        processed_views += 1
    return seed_map


def concat_tracking_states(states):
    return {key: torch.cat([state[key] for state in states], dim=0) for key in states[0]}


def save_mask_preview(mask: torch.Tensor, path: str):
    from PIL import Image
    import numpy as np

    array = (mask.detach().to(dtype=torch.uint8).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_score_preview(score: torch.Tensor, path: str):
    from PIL import Image
    import numpy as np

    array = score.detach().clamp(0.0, 1.0).cpu().numpy()
    array = (array * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def apply_injection_preset(args):
    preset_name = getattr(args, "injection_preset", "legacy_triplet")
    if preset_name not in INJECTION_PRESETS:
        raise ValueError(f"Unsupported injection_preset: {preset_name}")

    preset = INJECTION_PRESETS[preset_name]
    applied_overrides = {}
    for key, preset_value in preset.items():
        if getattr(args, key) == LEGACY_INJECTION_DEFAULTS[key]:
            setattr(args, key, preset_value)
            if preset_value != LEGACY_INJECTION_DEFAULTS[key]:
                applied_overrides[key] = preset_value

    print(f"Using injection preset: {preset_name}")
    if applied_overrides:
        print("Applied preset overrides:")
        for key, value in applied_overrides.items():
            print(f"  {key}: {value}")
    print("Resolved injection parameters:")
    for key in (
        "bundle_size",
        "bundle_spacing_scale",
        "center_opacity",
        "min_injected_opacity",
        "side_opacity_scale",
    ):
        print(f"  {key}: {getattr(args, key)}")


def main():
    parser = ArgumentParser(description="Inject prior-guided Gaussian bundles into an existing SOF checkpoint.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--candidate_mode", choices=["grid", "manifest", "mask", "auto_mask"], default="grid")
    parser.add_argument("--seed_manifest", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--lr_image_dir", type=str, default=None)
    parser.add_argument("--grid_stride", type=int, default=96)
    parser.add_argument("--grid_border", type=int, default=32)
    parser.add_argument("--max_seeds_per_view", type=int, default=0)
    parser.add_argument("--patch_radius", type=int, default=1)
    parser.add_argument("--view_limit", type=int, default=0)
    parser.add_argument("--neighborhood_radius", type=float, default=0.05)
    parser.add_argument("--neighborhood_thickness", type=float, default=0.015)
    parser.add_argument(
        "--injection_preset",
        choices=sorted(INJECTION_PRESETS.keys()),
        default="legacy_triplet",
    )
    parser.add_argument("--bundle_size", type=int, default=3)
    parser.add_argument("--bundle_spacing_scale", type=float, default=1.0)
    parser.add_argument("--center_opacity", type=float, default=0.08)
    parser.add_argument("--min_injected_opacity", type=float, default=0.02)
    parser.add_argument("--side_opacity_scale", type=float, default=0.6)
    parser.add_argument("--enable_surface_support_lock", action="store_true")
    parser.add_argument("--surface_support_min_neighbors", type=int, default=12)
    parser.add_argument("--surface_support_neighbor_soft_target", type=int, default=48)
    parser.add_argument("--surface_support_max_normal_ratio", type=float, default=0.35)
    parser.add_argument("--surface_support_min_strength", type=float, default=0.25)
    parser.add_argument("--surface_support_confidence_power", type=float, default=1.0)
    parser.add_argument("--alpha_threshold", type=float, default=1e-3)
    parser.add_argument("--normal_threshold", type=float, default=1e-6)
    parser.add_argument("--reliable_sigma_rgb", type=float, default=0.06)
    parser.add_argument("--reliable_sigma_grad", type=float, default=0.12)
    parser.add_argument("--reliable_threshold", type=float, default=0.5)
    parser.add_argument("--reliable_blur_kernel", type=int, default=5)
    parser.add_argument(
        "--auto_mask_variant",
        choices=["legacy", "conservative_v1", "conservative_v2"],
        default="legacy",
    )
    parser.add_argument("--near_depth_quantile", type=float, default=0.6)
    parser.add_argument("--need_pool_kernel", type=int, default=11)
    parser.add_argument("--need_score_threshold", type=float, default=0.4)
    parser.add_argument("--geom_pool_kernel", type=int, default=7)
    parser.add_argument("--geom_score_threshold", type=float, default=0.35)
    parser.add_argument("--geom_band_kernel", type=int, default=21)
    parser.add_argument("--rep_pool_kernel", type=int, default=15)
    parser.add_argument("--rep_score_threshold", type=float, default=0.45)
    parser.add_argument("--thin_pool_kernel", type=int, default=25)
    parser.add_argument("--thin_score_threshold", type=float, default=0.3)
    parser.add_argument("--thin_edge_threshold", type=float, default=0.35)
    parser.add_argument("--good_alpha_threshold", type=float, default=0.6)
    parser.add_argument("--good_match_threshold", type=float, default=0.05)
    parser.add_argument("--good_highpass_threshold", type=float, default=0.02)
    parser.add_argument("--good_highpass_kernel", type=int, default=9)
    parser.add_argument("--mask_smoothing_kernel", type=int, default=7)
    parser.add_argument("--mask_smoothing_threshold", type=float, default=0.5)
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--output_summary", type=str, default=None)
    parser.add_argument("--output_preview_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    apply_injection_preset(args)

    # Injection never optimizes against per-view RGB targets, so keeping camera
    # images on CPU avoids pinning the whole training set in VRAM.
    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for prior injection.")
        args.data_device = "cpu"

    safe_state(args.quiet)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)
    splat_args = splatting.get_settings(args)

    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe_args.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()
    appearance_embedding = build_appearance_embedding(mesh_args, num_views=len(train_cameras))
    gaussians.training_setup(opt_args, mesh_args, appearance_embedding)

    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    checkpoint_path = resolve_start_checkpoint(dataset.model_path, args.start_checkpoint, loaded_iteration)
    checkpoint_iteration = loaded_iteration
    if os.path.exists(checkpoint_path):
        model_params, checkpoint_iteration, appearance_state = torch.load(checkpoint_path)
        if appearance_embedding is not None and appearance_state[0] is not None:
            appearance_embedding.restore(*appearance_state)
        gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)
        checkpoint_origin = checkpoint_path
    else:
        checkpoint_origin = os.path.join(
            dataset.model_path,
            "point_cloud",
            f"iteration_{loaded_iteration}",
            "point_cloud.ply",
        )
        print(
            f"Checkpoint not found at {checkpoint_path}; "
            f"falling back to loaded point cloud state from iteration {loaded_iteration}."
        )

    gaussians.compute_3D_filter(train_cameras.copy(), CUDA=not pipe_args.compute_filter3D_python)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    preview_dir = args.output_preview_dir
    if preview_dir is None:
        preview_dir = os.path.join(dataset.model_path, f"prior_injection_preview_iter{checkpoint_iteration}")
    os.makedirs(preview_dir, exist_ok=True)
    mask_preview_dir = os.path.join(preview_dir, "masks")
    os.makedirs(mask_preview_dir, exist_ok=True)

    prior_index = index_image_dir(args.prior_dir)
    seed_map = build_seed_map(train_cameras, prior_index, args)
    lr_index = None
    if args.candidate_mode == "auto_mask":
        lr_dir = args.lr_image_dir or infer_lr_dir_from_pseudo_scene(dataset.source_path)
        if lr_dir is None:
            raise ValueError(
                "Automatic mask mode requires --lr_image_dir, or a pseudo_sr_summary.json that points to the original LR directory."
            )
        lr_index = index_image_dir(lr_dir)

    bundle_xyz = []
    bundle_f_dc = []
    bundle_f_rest = []
    bundle_opacity = []
    bundle_scaling = []
    bundle_rotation = []
    bundle_tracking = []
    view_summaries: List[ViewInjectionSummary] = []
    next_seed_id = int((gaussians._seed_id.max().item() + 1) if gaussians._seed_id.numel() > 0 else 0)
    processed_views = 0

    for view in train_cameras:
        if args.view_limit > 0 and processed_views >= args.view_limit:
            break

        prior_path = prior_index.get(view.image_name)
        if prior_path is None:
            seeds = seed_map.get(view.image_name, []) if args.candidate_mode != "auto_mask" else []
            view_summaries.append(
                ViewInjectionSummary(
                    image_name=view.image_name,
                    requested_seeds=len(seeds),
                    injected_seeds=0,
                    injected_gaussians=0,
                    skipped_invalid_anchor=0,
                    skipped_missing_prior=len(seeds),
                )
            )
            processed_views += 1
            continue

        prior_image = load_prior_image(prior_path)
        view_state = render_view_state(view, gaussians, pipe_args, background, splat_args)
        if (
            prior_image.shape[0] != view_state.depth.shape[0]
            or prior_image.shape[1] != view_state.depth.shape[1]
        ):
            prior_image = resize_image_hw3(
                prior_image,
                target_height=view_state.depth.shape[0],
                target_width=view_state.depth.shape[1],
            ).cpu()

        if args.candidate_mode == "auto_mask":
            lr_path = lr_index.get(view.image_name) if lr_index is not None else None
            if lr_path is None:
                view_summaries.append(
                    ViewInjectionSummary(
                        image_name=view.image_name,
                        requested_seeds=0,
                        injected_seeds=0,
                        injected_gaussians=0,
                        skipped_invalid_anchor=0,
                        skipped_missing_prior=0,
                    )
                )
                continue
            lr_image = load_rgb_image(lr_path)
            mask_dict = build_auto_injection_mask(
                view_state=view_state,
                prior_hr=prior_image,
                lr_image=lr_image,
                auto_mask_variant=args.auto_mask_variant,
                alpha_threshold=args.alpha_threshold,
                normal_threshold=args.normal_threshold,
                reliable_sigma_rgb=args.reliable_sigma_rgb,
                reliable_sigma_grad=args.reliable_sigma_grad,
                reliable_threshold=args.reliable_threshold,
                reliable_blur_kernel=args.reliable_blur_kernel,
                near_depth_quantile=args.near_depth_quantile,
                need_pool_kernel=args.need_pool_kernel,
                need_score_threshold=args.need_score_threshold,
                geom_pool_kernel=args.geom_pool_kernel,
                geom_score_threshold=args.geom_score_threshold,
                geom_band_kernel=args.geom_band_kernel,
                rep_pool_kernel=args.rep_pool_kernel,
                rep_score_threshold=args.rep_score_threshold,
                thin_pool_kernel=args.thin_pool_kernel,
                thin_score_threshold=args.thin_score_threshold,
                thin_edge_threshold=args.thin_edge_threshold,
                good_alpha_threshold=args.good_alpha_threshold,
                good_match_threshold=args.good_match_threshold,
                good_highpass_threshold=args.good_highpass_threshold,
                good_highpass_kernel=args.good_highpass_kernel,
                mask_smoothing_kernel=args.mask_smoothing_kernel,
                mask_smoothing_threshold=args.mask_smoothing_threshold,
            )
            save_mask_preview(mask_dict["anchor_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_anchor.png"))
            save_mask_preview(mask_dict["near_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_near.png"))
            save_mask_preview(mask_dict["reliable_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_reliable.png"))
            save_mask_preview(mask_dict["good_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_good.png"))
            save_mask_preview(mask_dict["need_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_need.png"))
            save_mask_preview(mask_dict["geom_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_geom.png"))
            save_mask_preview(mask_dict["rep_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_rep.png"))
            save_mask_preview(mask_dict["thin_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_thin.png"))
            save_score_preview(mask_dict["need_score"], os.path.join(mask_preview_dir, f"{view.image_name}_need_score.png"))
            save_score_preview(mask_dict["geom_score"], os.path.join(mask_preview_dir, f"{view.image_name}_geom_score.png"))
            save_score_preview(mask_dict["rep_score"], os.path.join(mask_preview_dir, f"{view.image_name}_rep_score.png"))
            save_score_preview(mask_dict["thin_score"], os.path.join(mask_preview_dir, f"{view.image_name}_thin_score.png"))
            save_mask_preview(mask_dict["inject_mask"], os.path.join(mask_preview_dir, f"{view.image_name}_inject.png"))
            seeds = generate_mask_candidates(
                image_name=view.image_name,
                mask=mask_dict["inject_mask"].detach().cpu(),
                stride=args.grid_stride,
                border=max(args.grid_border, args.patch_radius),
                max_seeds=args.max_seeds_per_view,
            )
            for seed in seeds:
                seed.patch_radius = args.patch_radius
        else:
            seeds = seed_map.get(view.image_name, [])
            mask_dict = None

        if not seeds:
            if mask_dict is not None:
                view_summaries.append(
                    ViewInjectionSummary(
                        image_name=view.image_name,
                        requested_seeds=0,
                        injected_seeds=0,
                        injected_gaussians=0,
                        skipped_invalid_anchor=0,
                        skipped_missing_prior=0,
                        anchor_pixels=int(mask_dict["anchor_mask"].sum().item()),
                        near_pixels=int(mask_dict["near_mask"].sum().item()),
                        reliable_pixels=int(mask_dict["reliable_mask"].sum().item()),
                        good_pixels=int(mask_dict["good_mask"].sum().item()),
                        need_pixels=int(mask_dict["need_mask"].sum().item()),
                        geom_pixels=int(mask_dict["geom_mask"].sum().item()),
                        rep_pixels=int(mask_dict["rep_mask"].sum().item()),
                        thin_pixels=int(mask_dict["thin_mask"].sum().item()),
                        inject_pixels=int(mask_dict["inject_mask"].sum().item()),
                    )
                )
                processed_views += 1
            continue

        injected_seeds = 0
        injected_gaussians = 0
        skipped_invalid_anchor = 0
        skipped_weak_support = 0

        for seed in seeds:
            anchor = sample_anchor_from_view(
                view_state=view_state,
                view=view,
                x=seed.x,
                y=seed.y,
                alpha_threshold=args.alpha_threshold,
                normal_threshold=args.normal_threshold,
            )
            if anchor is None:
                skipped_invalid_anchor += 1
                continue

            anchor_point, anchor_normal = anchor
            rgb = sample_prior_patch_rgb(prior_image, seed.x, seed.y, seed.patch_radius)
            tangent_scale, normal_scale, neighbor_count = estimate_local_surface_scales(
                gaussians=gaussians,
                anchor=anchor_point,
                normal=anchor_normal,
                neighborhood_radius=args.neighborhood_radius,
                neighborhood_thickness=args.neighborhood_thickness,
            )

            effective_confidence = float(seed.confidence)
            if args.enable_surface_support_lock:
                support_strength = compute_surface_support_strength(
                    neighbor_count=neighbor_count,
                    tangent_scale=tangent_scale,
                    normal_scale=normal_scale,
                    neighbor_soft_target=args.surface_support_neighbor_soft_target,
                    max_normal_ratio=args.surface_support_max_normal_ratio,
                )
                if (
                    neighbor_count < args.surface_support_min_neighbors
                    or support_strength < args.surface_support_min_strength
                ):
                    skipped_weak_support += 1
                    continue
                normal_scale = min(normal_scale, tangent_scale * args.surface_support_max_normal_ratio)
                effective_confidence *= max(support_strength, 1e-3) ** args.surface_support_confidence_power

            xyz, features_dc, features_rest, opacities, scaling, rotation, tracking_state = build_normal_bundle(
                gaussians=gaussians,
                anchor=anchor_point,
                normal=anchor_normal,
                rgb=rgb,
                confidence=effective_confidence,
                tangent_scale=tangent_scale,
                normal_scale=normal_scale,
                bundle_spacing_scale=args.bundle_spacing_scale,
                center_opacity=args.center_opacity,
                min_opacity=args.min_injected_opacity,
                side_opacity_scale=args.side_opacity_scale,
                seed_id=next_seed_id,
                bundle_size=args.bundle_size,
            )
            next_seed_id += 1

            bundle_xyz.append(xyz)
            bundle_f_dc.append(features_dc)
            bundle_f_rest.append(features_rest)
            bundle_opacity.append(opacities)
            bundle_scaling.append(scaling)
            bundle_rotation.append(rotation)
            bundle_tracking.append(tracking_state)
            injected_seeds += 1
            injected_gaussians += xyz.shape[0]

        view_summaries.append(
            ViewInjectionSummary(
                image_name=view.image_name,
                requested_seeds=len(seeds),
                injected_seeds=injected_seeds,
                injected_gaussians=injected_gaussians,
                skipped_invalid_anchor=skipped_invalid_anchor,
                skipped_missing_prior=0,
                skipped_weak_support=skipped_weak_support,
                anchor_pixels=int(mask_dict["anchor_mask"].sum().item()) if mask_dict is not None else 0,
                near_pixels=int(mask_dict["near_mask"].sum().item()) if mask_dict is not None else 0,
                reliable_pixels=int(mask_dict["reliable_mask"].sum().item()) if mask_dict is not None else 0,
                good_pixels=int(mask_dict["good_mask"].sum().item()) if mask_dict is not None else 0,
                need_pixels=int(mask_dict["need_mask"].sum().item()) if mask_dict is not None else 0,
                geom_pixels=int(mask_dict["geom_mask"].sum().item()) if mask_dict is not None else 0,
                rep_pixels=int(mask_dict["rep_mask"].sum().item()) if mask_dict is not None else 0,
                thin_pixels=int(mask_dict["thin_mask"].sum().item()) if mask_dict is not None else 0,
                inject_pixels=int(mask_dict["inject_mask"].sum().item()) if mask_dict is not None else 0,
            )
        )
        processed_views += 1

    if bundle_xyz:
        gaussians.densification_postfix(
            new_xyz=torch.cat(bundle_xyz, dim=0),
            new_features_dc=torch.cat(bundle_f_dc, dim=0),
            new_features_rest=torch.cat(bundle_f_rest, dim=0),
            new_opacities=torch.cat(bundle_opacity, dim=0),
            new_scaling=torch.cat(bundle_scaling, dim=0),
            new_rotation=torch.cat(bundle_rotation, dim=0),
            tracking_state=concat_tracking_states(bundle_tracking),
        )
        gaussians.compute_3D_filter(train_cameras.copy(), CUDA=not pipe_args.compute_filter3D_python)

    output_checkpoint = args.output_checkpoint or os.path.join(
        dataset.model_path,
        f"chkpnt{checkpoint_iteration}_priorinject.pth",
    )
    checkpoint_dir = os.path.dirname(output_checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    appearance_capture = appearance_embedding.capture() if appearance_embedding is not None else (None, None)
    torch.save((gaussians.capture(), checkpoint_iteration, appearance_capture), output_checkpoint)

    output_summary = args.output_summary or os.path.join(
        dataset.model_path,
        f"prior_injection_summary_iter{checkpoint_iteration}.json",
    )
    summary = InjectionSummary(
        baseline_checkpoint=checkpoint_origin,
        output_checkpoint=output_checkpoint,
        prior_dir=args.prior_dir,
        candidate_mode=args.candidate_mode,
        auto_mask_variant=args.auto_mask_variant,
        injection_preset=args.injection_preset,
        bundle_size=args.bundle_size,
        bundle_spacing_scale=args.bundle_spacing_scale,
        center_opacity=args.center_opacity,
        min_injected_opacity=args.min_injected_opacity,
        side_opacity_scale=args.side_opacity_scale,
        surface_support_lock_enabled=bool(args.enable_surface_support_lock),
        surface_support_min_neighbors=int(args.surface_support_min_neighbors),
        surface_support_neighbor_soft_target=int(args.surface_support_neighbor_soft_target),
        surface_support_max_normal_ratio=float(args.surface_support_max_normal_ratio),
        surface_support_min_strength=float(args.surface_support_min_strength),
        surface_support_confidence_power=float(args.surface_support_confidence_power),
        total_requested_seeds=sum(v.requested_seeds for v in view_summaries),
        total_injected_seeds=sum(v.injected_seeds for v in view_summaries),
        total_injected_gaussians=sum(v.injected_gaussians for v in view_summaries),
        views=view_summaries,
    )
    save_injection_summary(summary, output_summary)

    gaussians.save_ply(os.path.join(preview_dir, "point_cloud.ply"))
    gaussians.save_tracking_metadata(os.path.join(preview_dir, "gaussian_tags.pt"))

    print(f"Injected {summary.total_injected_gaussians} gaussians from {summary.total_injected_seeds} seeds.")
    print(f"Checkpoint saved to: {output_checkpoint}")
    print(f"Summary saved to: {output_summary}")


if __name__ == "__main__":
    main()
