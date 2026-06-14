import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm

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
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.prior_fusion import (
    build_bounded_carriers,
    build_mesh_depth_edge_mask,
    build_mesh_partition_masks,
    build_mesh_sample_visibility,
    build_mesh_visibility,
    export_fused_carrier_mesh_assets,
    export_fused_carrier_point_cloud,
    fuse_bounded_carriers,
    save_mask_preview,
)
from utils.prior_injection import (
    build_reliable_prior_mask,
    index_image_dir,
    infer_lr_dir_from_pseudo_scene,
    load_mask,
    load_prior_image,
    load_rgb_image,
    normalize_image_name,
    render_view_state,
    resize_image_hw3,
)


def lookup_indexed_image(index, image_name: str):
    candidates = [
        image_name,
        normalize_image_name(image_name),
        Path(image_name).name,
        Path(image_name).stem,
    ]
    for key in candidates:
        if key in index:
            return index[key]
    lower_index = {str(key).lower(): value for key, value in index.items()}
    for key in candidates:
        value = lower_index.get(str(key).lower())
        if value is not None:
            return value
    return None


def camera_original_image_hw3(view) -> torch.Tensor:
    return view.original_image[:3].detach().permute(1, 2, 0).cpu()


def preview_keys(index, limit: int = 8):
    return list(index.keys())[:limit]


def load_train_cameras_only(dataset):
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path,
            dataset.images,
            dataset.eval,
            init_type=dataset.init_type,
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            dataset.source_path,
            dataset.white_background,
            dataset.eval,
        )
    else:
        raise RuntimeError(f"Could not recognize scene type under {dataset.source_path}")

    print("Loading Training Cameras")
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset)


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


def select_carrier_face_ids(mesh_obj: trimesh.Trimesh, visible_face_union, args):
    total_mesh_faces = len(mesh_obj.faces)
    selection_mode = args.carrier_face_selection
    if selection_mode == "visible_union" and visible_face_union:
        visible_face_concat = np.concatenate(visible_face_union, axis=0)
        if visible_face_concat.size > 0:
            carrier_face_ids = np.unique(visible_face_concat).astype(np.int64, copy=False)
            carrier_face_source = "zbuffer_visible_faces"
        else:
            carrier_face_ids = np.arange(total_mesh_faces, dtype=np.int64)
            carrier_face_source = "all_mesh_faces_no_zbuffer_visible_faces"
    elif selection_mode == "visible_union":
        carrier_face_ids = np.arange(total_mesh_faces, dtype=np.int64)
        carrier_face_source = "all_mesh_faces_no_processed_visibility"
    elif selection_mode == "all_faces":
        carrier_face_ids = np.arange(total_mesh_faces, dtype=np.int64)
        carrier_face_source = "all_mesh_faces_requested"
    elif selection_mode == "roi_payload":
        if not args.carrier_roi_payload:
            raise ValueError("--carrier_face_selection roi_payload requires --carrier_roi_payload")
        roi_payload = torch.load(args.carrier_roi_payload, map_location="cpu")
        if args.carrier_roi_key in roi_payload:
            roi_value = roi_payload[args.carrier_roi_key]
            if isinstance(roi_value, torch.Tensor):
                roi_value = roi_value.detach().cpu().numpy()
            roi_value = np.asarray(roi_value)
            if roi_value.dtype == np.bool_:
                carrier_face_ids = np.flatnonzero(roi_value).astype(np.int64, copy=False)
            else:
                carrier_face_ids = roi_value.reshape(-1).astype(np.int64, copy=False)
        elif "roi_face_mask" in roi_payload:
            roi_mask = roi_payload["roi_face_mask"]
            if isinstance(roi_mask, torch.Tensor):
                roi_mask = roi_mask.detach().cpu().numpy()
            carrier_face_ids = np.flatnonzero(np.asarray(roi_mask).astype(bool)).astype(np.int64, copy=False)
        else:
            raise KeyError(
                f"ROI payload does not contain '{args.carrier_roi_key}' or 'roi_face_mask': "
                f"{args.carrier_roi_payload}"
            )
        carrier_face_ids = np.unique(carrier_face_ids)
        if np.any((carrier_face_ids < 0) | (carrier_face_ids >= total_mesh_faces)):
            raise ValueError(
                f"ROI payload face ids out of mesh range [0, {total_mesh_faces}): "
                f"{args.carrier_roi_payload}"
            )
        carrier_face_source = f"roi_payload:{Path(args.carrier_roi_payload).name}:{args.carrier_roi_key}"
    else:
        raise ValueError(f"Unsupported carrier_face_selection={selection_mode}")

    stride = max(int(args.carrier_face_stride), 1)
    carrier_face_ids = carrier_face_ids[::stride]
    if args.carrier_max_faces > 0 and carrier_face_ids.size > int(args.carrier_max_faces):
        max_faces = int(args.carrier_max_faces)
        if args.carrier_face_sample_mode == "random":
            rng = np.random.default_rng(int(args.carrier_face_random_seed))
            carrier_face_ids = np.sort(rng.choice(carrier_face_ids, size=max_faces, replace=False)).astype(np.int64, copy=False)
            carrier_face_source += "_random_subset"
        else:
            carrier_face_ids = carrier_face_ids[:max_faces]
            carrier_face_source += "_prefix_subset"
    return carrier_face_ids.astype(np.int64, copy=False), carrier_face_source


def auto_resolve_splatting_config(args, dataset):
    default_config = "configs/hierarchical.json"
    current_config = getattr(args, "splatting_config", None)
    model_config = os.path.join(dataset.model_path, "config.json")
    if (current_config is None or current_config == default_config) and os.path.exists(model_config):
        args.splatting_config = model_config


def load_sofgs_for_trust_masks(dataset, opt_args, pipe_args, mesh_args, args, splat_args):
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe_args.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()

    appearance_embedding = build_appearance_embedding(mesh_args, num_views=len(train_cameras))
    gaussians.training_setup(opt_args, mesh_args, appearance_embedding)

    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    checkpoint_path = resolve_start_checkpoint(dataset.model_path, args.start_checkpoint, loaded_iteration)
    if os.path.exists(checkpoint_path):
        model_params, checkpoint_iteration, appearance_state = torch.load(checkpoint_path)
        if appearance_embedding is not None and appearance_state[0] is not None:
            appearance_embedding.restore(*appearance_state)
        gaussians.restore(model_params, opt_args, mesh_args, appearance_embedding)
        checkpoint_origin = checkpoint_path
    else:
        checkpoint_iteration = loaded_iteration
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
    return train_cameras, gaussians, pipe_args, background, splat_args, checkpoint_origin, checkpoint_iteration


def resize_prior_to_view(prior_image: torch.Tensor, view) -> torch.Tensor:
    target_height = int(view.image_height)
    target_width = int(view.image_width)
    if prior_image.shape[0] == target_height and prior_image.shape[1] == target_width:
        return prior_image.cpu()
    return resize_image_hw3(
        prior_image,
        target_height=target_height,
        target_width=target_width,
    ).cpu()


def build_prior_valid_mask(args, prior_image: torch.Tensor, lr_image: torch.Tensor, view) -> torch.Tensor:
    target_height = int(view.image_height)
    target_width = int(view.image_width)
    if args.prior_valid_mode == "full":
        return torch.ones((target_height, target_width), dtype=torch.bool, device="cuda")
    if args.prior_valid_mode == "prior_lr":
        return build_reliable_prior_mask(
            prior_hr=prior_image,
            lr_image=lr_image,
            target_height=target_height,
            target_width=target_width,
            sigma_rgb=args.reliable_sigma_rgb,
            sigma_grad=args.reliable_sigma_grad,
            threshold=args.reliable_threshold,
            blur_kernel=args.reliable_blur_kernel,
        )
    raise ValueError(f"Unsupported prior_valid_mode={args.prior_valid_mode}")


def build_parser():
    parser = ArgumentParser(description="Prepare v0 mesh/edge prior fusion assets for SOF.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    mesh = MeshingParams(parser)
    splatting = SplattingSettings(parser, render=False)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--lr_image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--view_limit", type=int, default=0)
    parser.add_argument("--carriers_per_face", type=int, choices=[1, 3, 4, 6], default=1)
    parser.add_argument("--carrier_face_stride", type=int, default=1)
    parser.add_argument("--carrier_max_faces", type=int, default=0)
    parser.add_argument(
        "--carrier_face_selection",
        choices=["visible_union", "all_faces", "roi_payload"],
        default="visible_union",
        help=(
            "How to choose mesh faces that receive bounded carriers. visible_union "
            "uses faces seen while preparing view masks; all_faces skips that "
            "dependency and samples directly from the whole mesh; roi_payload "
            "uses faces proposed by build_meshfusion_roi_from_outliers_v0.py."
        ),
    )
    parser.add_argument("--carrier_roi_payload", type=str, default=None)
    parser.add_argument("--carrier_roi_key", type=str, default="roi_face_ids")
    parser.add_argument(
        "--carrier_face_sample_mode",
        choices=["prefix", "random"],
        default="prefix",
        help="When --carrier_max_faces truncates faces, take the prefix or a deterministic random subset.",
    )
    parser.add_argument("--carrier_face_random_seed", type=int, default=0)
    parser.add_argument("--carrier_min_views", type=int, default=2)
    parser.add_argument("--carrier_thickness_scale", type=float, default=0.1)
    parser.add_argument(
        "--carrier_mask_mode",
        choices=["none", "mesh_core"],
        default="none",
        help=(
            "Optional 2D view mask for carrier fusion. The meshfusion main path "
            "defaults to 'none' and relies on 3D mesh visibility only; 'mesh_core' "
            "keeps the old mask-gated behavior for diagnostics."
        ),
    )
    parser.add_argument(
        "--carrier_fusion_policy",
        choices=["mean_v0", "freq_split_v1"],
        default="mean_v0",
        help=(
            "How carrier prior samples are fused. mean_v0 keeps weighted average; "
            "freq_split_v1 separates low/high frequency samples and gates high "
            "frequency by multi-view consistency."
        ),
    )
    parser.add_argument("--carrier_lowfreq_consistency_threshold", type=float, default=0.08)
    parser.add_argument("--carrier_highfreq_consistency_threshold", type=float, default=0.04)
    parser.add_argument("--carrier_frequency_blur_kernel", type=int, default=9)
    parser.add_argument("--carrier_single_view_confidence", type=float, default=0.2)
    parser.add_argument("--skip_carrier_fusion", action="store_true")
    parser.add_argument(
        "--skip_view_mask_prepare",
        action="store_true",
        help=(
            "Meshfusion carrier-only path: skip per-view mesh/core/edge mask generation. "
            "This is intended for carrier_mask_mode=none, where prior is fused directly "
            "onto mesh-bounded carriers instead of through 2D masks."
        ),
    )
    parser.add_argument(
        "--mesh_trust_mode",
        choices=["sample_zbuffer", "zbuffer", "sofgs"],
        default="sample_zbuffer",
        help=(
            "How to make per-view mesh/core/edge masks. 'sample_zbuffer' projects "
            "mesh face samples and does a fast GPU min-depth splat; 'zbuffer' uses "
            "the diagnostic Python triangle rasterizer; 'sofgs' uses existing SOFGS "
            "alpha/depth as an optional support proxy."
        ),
    )
    parser.add_argument("--mesh_trust_alpha_threshold", type=float, default=0.6)
    parser.add_argument("--mesh_sample_points_per_face", type=int, choices=[1, 4], default=4)
    parser.add_argument("--mesh_sample_splat_kernel", type=int, default=5)
    parser.add_argument("--mesh_core_erode_kernel", type=int, default=17)
    parser.add_argument("--mesh_depth_edge_kernel", type=int, default=7)
    parser.add_argument("--mesh_depth_edge_dilate_kernel", type=int, default=9)
    parser.add_argument("--mesh_depth_edge_abs_threshold", type=float, default=0.03)
    parser.add_argument("--mesh_depth_edge_rel_threshold", type=float, default=0.01)
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--visibility_epsilon", type=float, default=5e-4)
    parser.add_argument("--ray_chunk_size", type=int, default=8192)
    parser.add_argument(
        "--carrier_visibility_mode",
        choices=["rasterizer", "carrier_zbuffer", "projection", "ray"],
        default="rasterizer",
        help=(
            "Visibility test used while sampling priors onto mesh carriers. "
            "rasterizer reuses the SOF Gaussian rasterizer on the bounded carrier "
            "layer; carrier_zbuffer is a lightweight fallback; projection disables "
            "occlusion checks; ray keeps the old full-mesh trimesh ray test."
        ),
    )
    parser.add_argument("--carrier_visibility_depth_tolerance", type=float, default=0.03)
    parser.add_argument("--carrier_visibility_alpha_threshold", type=float, default=0.02)
    parser.add_argument("--carrier_visibility_scale_modifier", type=float, default=1.0)
    parser.add_argument("--carrier_visibility_opacity", type=float, default=0.95)
    parser.add_argument("--disable_front_facing_only", action="store_true")
    parser.add_argument("--alpha_threshold", type=float, default=1e-3)
    parser.add_argument("--normal_threshold", type=float, default=1e-6)
    parser.add_argument(
        "--prior_valid_mode",
        choices=["full", "prior_lr"],
        default="full",
        help=(
            "How to decide which prior pixels may be fused onto mesh carriers. "
            "'full' uses the mesh visibility/core masks only; 'prior_lr' also "
            "filters pixels whose prior disagrees strongly with the LR view."
        ),
    )
    parser.add_argument("--reliable_sigma_rgb", type=float, default=0.06)
    parser.add_argument("--reliable_sigma_grad", type=float, default=0.12)
    parser.add_argument("--reliable_threshold", type=float, default=0.5)
    parser.add_argument("--reliable_blur_kernel", type=int, default=5)
    parser.add_argument(
        "--auto_mask_variant",
        choices=["legacy", "conservative_v1", "conservative_v2"],
        default="conservative_v2",
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
    parser.add_argument("--quiet", action="store_true")
    return parser, model, opt, pipe, mesh, splatting


def main():
    parser, model, opt, pipe, mesh, splatting = build_parser()
    args = get_combined_args(parser)

    if getattr(args, "data_device", None) != "cpu":
        print(f"Overriding data_device from {getattr(args, 'data_device', None)} to cpu for prior fusion preparation.")
        args.data_device = "cpu"
    if bool(args.skip_view_mask_prepare):
        if bool(args.skip_carrier_fusion):
            raise ValueError("--skip_view_mask_prepare only makes sense when carrier fusion is enabled.")
        if args.carrier_mask_mode != "none":
            raise ValueError("--skip_view_mask_prepare requires --carrier_mask_mode none.")

    safe_state(args.quiet)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe_args = pipe.extract(args)
    mesh_args = mesh.extract(args)
    splat_args = None
    gaussians = None
    background = None
    checkpoint_origin = "not_used_for_mesh_prior_fusion_prepare"
    checkpoint_iteration = args.iteration

    if args.mesh_trust_mode == "sofgs":
        auto_resolve_splatting_config(args, dataset)
        splat_args = splatting.get_settings(args)
        (
            train_cameras,
            gaussians,
            pipe_args,
            background,
            splat_args,
            checkpoint_origin,
            checkpoint_iteration,
        ) = load_sofgs_for_trust_masks(dataset, opt_args, pipe_args, mesh_args, args, splat_args)
    else:
        train_cameras = load_train_cameras_only(dataset)
        if args.carrier_visibility_mode == "rasterizer":
            auto_resolve_splatting_config(args, dataset)
            splat_args = splatting.get_settings(args)

    prior_index = index_image_dir(args.prior_dir)
    lr_dir = args.lr_image_dir or infer_lr_dir_from_pseudo_scene(dataset.source_path)
    if args.prior_valid_mode == "prior_lr" and lr_dir is None:
        raise ValueError("Prior fusion preparation requires --lr_image_dir, or a pseudo_sr_summary.json that points to the LR images.")
    lr_index = index_image_dir(lr_dir) if lr_dir is not None else {}

    mesh_obj = trimesh.load_mesh(args.mesh_path, process=False)
    print(
        f"[prior-fusion] loaded mesh: vertices={len(mesh_obj.vertices)}, faces={len(mesh_obj.faces)}",
        flush=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_mask_dir = output_dir / "mesh_fusion_masks"
    edge_mask_dir = output_dir / "edge_fusion_masks"
    debug_mask_dir = output_dir / "debug_masks"
    mesh_mask_dir.mkdir(parents=True, exist_ok=True)
    edge_mask_dir.mkdir(parents=True, exist_ok=True)
    debug_mask_dir.mkdir(parents=True, exist_ok=True)

    view_summaries = []
    visible_face_union = []
    processed_views = 0
    views_for_fusion = []
    matched_prior_index = dict(prior_index)
    prior_match_count = 0
    lr_match_count = 0
    lr_fallback_count = 0
    missing_prior_views = []
    missing_lr_views = []
    train_view_names = [view.image_name for view in train_cameras]

    if bool(args.skip_view_mask_prepare):
        progress_desc = "Matching prior views for carrier fusion"
        if args.view_limit > 0:
            progress_desc += f" (limit={args.view_limit})"
        with tqdm(train_cameras, desc=progress_desc, disable=bool(args.quiet), dynamic_ncols=True) as view_progress:
            for view in view_progress:
                if args.view_limit > 0 and processed_views >= args.view_limit:
                    break
                prior_path = lookup_indexed_image(prior_index, view.image_name)
                lr_path = lookup_indexed_image(lr_index, view.image_name)
                if prior_path is None:
                    if len(missing_prior_views) < 16:
                        missing_prior_views.append(view.image_name)
                    view_progress.set_postfix(processed=processed_views, prior=prior_match_count, missing_prior=len(missing_prior_views))
                    continue
                prior_match_count += 1
                matched_prior_index[view.image_name] = prior_path
                if lr_path is None:
                    lr_fallback_count += 1
                    if len(missing_lr_views) < 16:
                        missing_lr_views.append(view.image_name)
                else:
                    lr_match_count += 1
                views_for_fusion.append(view)
                view_summaries.append(
                    {
                        "image_name": view.image_name,
                        "prior_path": str(prior_path),
                        "lr_path": str(lr_path) if lr_path is not None else "camera_original_image",
                        "prior_valid_mode": args.prior_valid_mode,
                        "mesh_trust_mode": "skipped_view_mask_prepare",
                        "prior_valid_pixels": -1,
                        "mesh_visible_pixels": -1,
                        "mesh_core_pixels": -1,
                        "mesh_depth_edge_pixels": -1,
                        "mesh_fusion_pixels": -1,
                        "edge_fusion_pixels": -1,
                        "front_visible_face_count": -1,
                    }
                )
                processed_views += 1
                view_progress.set_postfix(processed=processed_views, prior=prior_match_count)
    else:
        progress_desc = "Preparing prior/mesh masks"
        if args.view_limit > 0:
            progress_desc += f" (limit={args.view_limit})"
        with tqdm(train_cameras, desc=progress_desc, disable=bool(args.quiet), dynamic_ncols=True) as view_progress:
            for view in view_progress:
                if args.view_limit > 0 and processed_views >= args.view_limit:
                    break
                prior_path = lookup_indexed_image(prior_index, view.image_name)
                lr_path = lookup_indexed_image(lr_index, view.image_name)
                if prior_path is None:
                    if len(missing_prior_views) < 16:
                        missing_prior_views.append(view.image_name)
                    view_progress.set_postfix(processed=processed_views, prior=prior_match_count, missing_prior=len(missing_prior_views))
                    continue
                prior_match_count += 1
                matched_prior_index[view.image_name] = prior_path
                if args.prior_valid_mode == "prior_lr" and lr_path is None:
                    if len(missing_lr_views) < 16:
                        missing_lr_views.append(view.image_name)
                    view_progress.set_postfix(processed=processed_views, prior=prior_match_count, missing_lr=len(missing_lr_views))
                    continue
                if lr_path is None:
                    lr_fallback_count += 1
                    if len(missing_lr_views) < 16:
                        missing_lr_views.append(view.image_name)
                else:
                    lr_match_count += 1

                prior_image = load_prior_image(prior_path)
                prior_image_for_mask = resize_prior_to_view(prior_image, view)
                lr_image = load_rgb_image(lr_path) if lr_path is not None else camera_original_image_hw3(view)
                prior_valid_mask = build_prior_valid_mask(args, prior_image_for_mask, lr_image, view)

                if args.mesh_trust_mode == "sample_zbuffer":
                    mesh_visibility = build_mesh_sample_visibility(
                        mesh_obj,
                        view,
                        depth_min=float(args.depth_min),
                        front_facing_only=not bool(args.disable_front_facing_only),
                        samples_per_face=int(args.mesh_sample_points_per_face),
                        splat_kernel=int(args.mesh_sample_splat_kernel),
                    )
                    mesh_visible_mask = mesh_visibility["visible_mask"]
                    visible_face_ids = mesh_visibility["visible_face_ids"]
                    mesh_depth_edge_mask = build_mesh_depth_edge_mask(
                        mesh_visibility["depth"],
                        mesh_visible_mask,
                        abs_threshold=float(args.mesh_depth_edge_abs_threshold),
                        rel_threshold=float(args.mesh_depth_edge_rel_threshold),
                        kernel_size=int(args.mesh_depth_edge_kernel),
                        dilate_kernel=int(args.mesh_depth_edge_dilate_kernel),
                    )
                elif args.mesh_trust_mode == "sofgs":
                    view_state = render_view_state(view, gaussians, pipe_args, background, splat_args)
                    mesh_visible_mask = (
                        (view_state.alpha > float(args.mesh_trust_alpha_threshold))
                        & torch.isfinite(view_state.depth)
                        & (view_state.depth > float(args.depth_min))
                    )
                    mesh_depth_edge_mask = build_mesh_depth_edge_mask(
                        view_state.depth,
                        mesh_visible_mask,
                        abs_threshold=float(args.mesh_depth_edge_abs_threshold),
                        rel_threshold=float(args.mesh_depth_edge_rel_threshold),
                        kernel_size=int(args.mesh_depth_edge_kernel),
                        dilate_kernel=int(args.mesh_depth_edge_dilate_kernel),
                    )
                    # SOFGS trust mode intentionally avoids Python mesh rasterization.
                    # Carrier construction therefore falls back to the mesh face set.
                    visible_face_ids = np.empty((0,), dtype=np.int64)
                else:
                    mesh_visibility = build_mesh_visibility(
                        mesh_obj,
                        view,
                        depth_min=float(args.depth_min),
                        front_facing_only=not bool(args.disable_front_facing_only),
                    )
                    mesh_visible_mask = mesh_visibility["visible_mask"]
                    visible_face_ids = mesh_visibility["visible_face_ids"]
                    mesh_depth_edge_mask = build_mesh_depth_edge_mask(
                        mesh_visibility["depth"],
                        mesh_visible_mask,
                        abs_threshold=float(args.mesh_depth_edge_abs_threshold),
                        rel_threshold=float(args.mesh_depth_edge_rel_threshold),
                        kernel_size=int(args.mesh_depth_edge_kernel),
                        dilate_kernel=int(args.mesh_depth_edge_dilate_kernel),
                    )
                partition_masks = build_mesh_partition_masks(
                    prior_valid_mask=prior_valid_mask,
                    mesh_visible_mask=mesh_visible_mask,
                    interior_kernel=int(args.mesh_core_erode_kernel),
                    mesh_edge_mask=mesh_depth_edge_mask,
                )

                save_mask_preview(partition_masks["mesh_fusion_mask"], str(mesh_mask_dir / f"{view.image_name}_inject.png"))
                save_mask_preview(partition_masks["edge_fusion_mask"], str(edge_mask_dir / f"{view.image_name}_inject.png"))
                save_mask_preview(partition_masks["mesh_visible_mask"], str(debug_mask_dir / f"{view.image_name}_mesh_visible.png"))
                save_mask_preview(partition_masks["mesh_core_mask"], str(debug_mask_dir / f"{view.image_name}_mesh_core.png"))
                save_mask_preview(partition_masks["mesh_edge_band"], str(debug_mask_dir / f"{view.image_name}_mesh_edge_band.png"))
                save_mask_preview(mesh_depth_edge_mask, str(debug_mask_dir / f"{view.image_name}_mesh_depth_edge.png"))
                save_mask_preview(prior_valid_mask, str(debug_mask_dir / f"{view.image_name}_prior_valid.png"))

                visible_face_union.append(visible_face_ids)
                views_for_fusion.append(view)
                view_summaries.append(
                    {
                        "image_name": view.image_name,
                        "prior_path": str(prior_path),
                        "lr_path": str(lr_path) if lr_path is not None else "camera_original_image",
                        "prior_valid_mode": args.prior_valid_mode,
                        "mesh_trust_mode": args.mesh_trust_mode,
                        "prior_valid_pixels": int(prior_valid_mask.sum().item()),
                        "mesh_visible_pixels": int(partition_masks["mesh_visible_mask"].sum().item()),
                        "mesh_core_pixels": int(partition_masks["mesh_core_mask"].sum().item()),
                        "mesh_depth_edge_pixels": int(mesh_depth_edge_mask.sum().item()),
                        "mesh_fusion_pixels": int(partition_masks["mesh_fusion_mask"].sum().item()),
                        "edge_fusion_pixels": int(partition_masks["edge_fusion_mask"].sum().item()),
                        "front_visible_face_count": int(visible_face_ids.size),
                    }
                )
                processed_views += 1
                view_progress.set_postfix(
                    processed=processed_views,
                    prior=prior_match_count,
                    edge_px=int(partition_masks["edge_fusion_mask"].sum().item()),
                    faces=int(visible_face_ids.size),
                )

    if not view_summaries:
        diagnostics = {
            "train_view_count": len(train_view_names),
            "train_view_name_samples": train_view_names[:8],
            "prior_dir": str(Path(args.prior_dir).resolve()),
            "prior_key_count": len(prior_index),
            "prior_key_samples": preview_keys(prior_index),
            "prior_match_count": prior_match_count,
            "missing_prior_view_samples": missing_prior_views,
            "lr_dir": str(Path(lr_dir).resolve()) if lr_dir is not None else None,
            "lr_key_count": len(lr_index),
            "lr_key_samples": preview_keys(lr_index),
            "lr_match_count": lr_match_count,
            "lr_fallback_count": lr_fallback_count,
            "missing_lr_view_samples": missing_lr_views,
        }
        raise RuntimeError(
            "No train views had matching prior images for fusion asset preparation. "
            f"Diagnostics: {json.dumps(diagnostics, indent=2)}"
        )

    partition_params = {
        "mesh_trust_mode": args.mesh_trust_mode,
        "mesh_trust_alpha_threshold": float(args.mesh_trust_alpha_threshold),
        "mesh_sample_points_per_face": int(args.mesh_sample_points_per_face),
        "mesh_sample_splat_kernel": int(args.mesh_sample_splat_kernel),
        "splatting_config": str(getattr(args, "splatting_config", None)),
        "prior_valid_mode": args.prior_valid_mode,
        "mesh_core_erode_kernel": int(args.mesh_core_erode_kernel),
        "mesh_depth_edge_kernel": int(args.mesh_depth_edge_kernel),
        "mesh_depth_edge_dilate_kernel": int(args.mesh_depth_edge_dilate_kernel),
        "mesh_depth_edge_abs_threshold": float(args.mesh_depth_edge_abs_threshold),
        "mesh_depth_edge_rel_threshold": float(args.mesh_depth_edge_rel_threshold),
        "carrier_face_stride": int(args.carrier_face_stride),
        "carrier_max_faces": int(args.carrier_max_faces),
        "carrier_face_selection": args.carrier_face_selection,
        "carrier_face_sample_mode": args.carrier_face_sample_mode,
        "carrier_face_random_seed": int(args.carrier_face_random_seed),
        "carrier_roi_payload": args.carrier_roi_payload,
        "carrier_roi_key": args.carrier_roi_key,
        "carrier_min_views": int(args.carrier_min_views),
        "carrier_mask_mode": args.carrier_mask_mode,
        "carrier_fusion_policy": args.carrier_fusion_policy,
        "carrier_visibility_mode": args.carrier_visibility_mode,
        "carrier_visibility_depth_tolerance": float(args.carrier_visibility_depth_tolerance),
        "carrier_visibility_alpha_threshold": float(args.carrier_visibility_alpha_threshold),
        "carrier_visibility_scale_modifier": float(args.carrier_visibility_scale_modifier),
        "carrier_visibility_opacity": float(args.carrier_visibility_opacity),
        "skip_view_mask_prepare": bool(args.skip_view_mask_prepare),
        "carrier_lowfreq_consistency_threshold": float(args.carrier_lowfreq_consistency_threshold),
        "carrier_highfreq_consistency_threshold": float(args.carrier_highfreq_consistency_threshold),
        "carrier_frequency_blur_kernel": int(args.carrier_frequency_blur_kernel),
        "carrier_single_view_confidence": float(args.carrier_single_view_confidence),
        "depth_min": float(args.depth_min),
        "visibility_epsilon": float(args.visibility_epsilon),
    }

    if bool(args.skip_carrier_fusion):
        summary = {
            "prepare_mode": "mesh_prior_trust_gate_prepare",
            "skipped_carrier_fusion": True,
            "checkpoint_origin": checkpoint_origin,
            "checkpoint_iteration": int(checkpoint_iteration),
            "mesh_path": str(Path(args.mesh_path).resolve()),
            "prior_dir": str(Path(args.prior_dir).resolve()),
            "lr_dir": str(Path(lr_dir).resolve()) if lr_dir is not None else None,
            "prior_valid_mode": args.prior_valid_mode,
            "prior_match_count": int(prior_match_count),
            "lr_match_count": int(lr_match_count),
            "lr_fallback_count": int(lr_fallback_count),
            "missing_lr_view_samples": missing_lr_views,
            "output_dir": str(output_dir.resolve()),
            "views_processed": int(len(view_summaries)),
            "mesh_mask_dir": str(mesh_mask_dir.resolve()),
            "edge_mask_dir": str(edge_mask_dir.resolve()),
            "debug_mask_dir": str(debug_mask_dir.resolve()),
            "view_summaries": view_summaries,
            "partition_params": partition_params,
        }
        summary_path = output_dir / "prior_fusion_v0_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Saved summary to: {summary_path}")
        return

    carrier_face_ids, carrier_face_source = select_carrier_face_ids(mesh_obj, visible_face_union, args)
    print(
        "[prior-fusion] carrier face selection: "
        f"{carrier_face_ids.size}/{len(mesh_obj.faces)} faces from {carrier_face_source}",
        flush=True,
    )

    carriers = build_bounded_carriers(
        mesh_obj,
        carriers_per_face=int(args.carriers_per_face),
        face_ids=carrier_face_ids,
        thickness_scale=float(args.carrier_thickness_scale),
    )
    print(
        f"[prior-fusion] built {carriers['centers'].shape[0]} bounded carriers from {len(views_for_fusion)} views",
        flush=True,
    )
    fused_payload = fuse_bounded_carriers(
        mesh=mesh_obj,
        carriers=carriers,
        views=views_for_fusion,
        prior_index=matched_prior_index,
        mesh_mask_dir=mesh_mask_dir if args.carrier_mask_mode == "mesh_core" else None,
        load_mask_fn=load_mask,
        load_prior_image_fn=load_prior_image,
        min_views_per_carrier=int(args.carrier_min_views),
        depth_min=float(args.depth_min),
        visibility_epsilon=float(args.visibility_epsilon),
        ray_chunk_size=int(args.ray_chunk_size),
        carrier_visibility_mode=args.carrier_visibility_mode,
        carrier_visibility_depth_tolerance=float(args.carrier_visibility_depth_tolerance),
        carrier_visibility_alpha_threshold=float(args.carrier_visibility_alpha_threshold),
        carrier_visibility_scale_modifier=float(args.carrier_visibility_scale_modifier),
        carrier_visibility_opacity=float(args.carrier_visibility_opacity),
        splat_args=splat_args,
        fusion_policy=args.carrier_fusion_policy,
        low_frequency_consistency_threshold=float(args.carrier_lowfreq_consistency_threshold),
        high_frequency_consistency_threshold=float(args.carrier_highfreq_consistency_threshold),
        single_view_confidence=float(args.carrier_single_view_confidence),
        frequency_blur_kernel=int(args.carrier_frequency_blur_kernel),
        show_progress=not bool(args.quiet),
    )

    fused_payload_path = output_dir / "bounded_carrier_fusion_v0.npz"
    np.savez_compressed(
        fused_payload_path,
        centers=carriers["centers"],
        normals=carriers["normals"],
        face_ids=carriers["face_ids"],
        bary_coords=carriers["bary_coords"],
        scale_u=carriers["scale_u"],
        scale_v=carriers["scale_v"],
        scale_n=carriers["scale_n"],
        tangent_u=carriers["tangent_u"],
        tangent_v=carriers["tangent_v"],
        fused_rgb=fused_payload["fused_rgb"],
        confidence=fused_payload["confidence"],
        disagreement=fused_payload["disagreement"],
        view_count=fused_payload["view_count"],
        weight_sum=fused_payload["weight_sum"],
        valid_mask=fused_payload["valid_mask"],
        low_frequency_rgb=fused_payload["low_frequency_rgb"],
        high_frequency_rgb=fused_payload["high_frequency_rgb"],
        high_frequency_confidence=fused_payload["high_frequency_confidence"],
        fusion_case=fused_payload["fusion_case"],
    )

    export_fused_carrier_point_cloud(
        centers=carriers["centers"],
        fused_rgb=fused_payload["fused_rgb"],
        valid_mask=fused_payload["valid_mask"],
        path=output_dir / "bounded_carrier_fused_points.ply",
    )
    fused_mesh_export = export_fused_carrier_mesh_assets(
        mesh=mesh_obj,
        carriers=carriers,
        fused_payload=fused_payload,
        output_dir=output_dir,
    )

    total_faces = max(int(carrier_face_ids.size), 1)
    valid_mask = fused_payload["valid_mask"]
    summary = {
        "prepare_mode": "mesh_prior_trust_gate_prepare",
        "checkpoint_origin": checkpoint_origin,
        "checkpoint_iteration": int(checkpoint_iteration),
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "prior_dir": str(Path(args.prior_dir).resolve()),
        "lr_dir": str(Path(lr_dir).resolve()) if lr_dir is not None else None,
        "prior_valid_mode": args.prior_valid_mode,
        "prior_match_count": int(prior_match_count),
        "lr_match_count": int(lr_match_count),
        "lr_fallback_count": int(lr_fallback_count),
        "missing_lr_view_samples": missing_lr_views,
        "output_dir": str(output_dir.resolve()),
        "views_processed": int(len(view_summaries)),
        "carrier_face_count": int(carrier_face_ids.size),
        "carrier_face_ratio_vs_all_mesh": float(carrier_face_ids.size / max(len(mesh_obj.faces), 1)),
        "carrier_face_source": carrier_face_source,
        "carrier_count": int(carriers["centers"].shape[0]),
        "carriers_per_face": int(args.carriers_per_face),
        "valid_fused_carrier_count": int(valid_mask.sum()),
        "valid_fused_carrier_ratio": float(valid_mask.sum() / max(valid_mask.shape[0], 1)),
        "carrier_fusion_policy": args.carrier_fusion_policy,
        "payload_path": str(fused_payload_path.resolve()),
        "mesh_overlay_export": fused_mesh_export,
        "mesh_mask_dir": str(mesh_mask_dir.resolve()),
        "edge_mask_dir": str(edge_mask_dir.resolve()),
        "debug_mask_dir": str(debug_mask_dir.resolve()),
        "view_summaries": view_summaries,
        "carrier_stats": {
            "view_count_mean": float(np.mean(fused_payload["view_count"])) if fused_payload["view_count"].size > 0 else 0.0,
            "view_count_valid_mean": float(np.mean(fused_payload["view_count"][valid_mask])) if np.any(valid_mask) else 0.0,
            "confidence_valid_mean": float(np.mean(fused_payload["confidence"][valid_mask])) if np.any(valid_mask) else 0.0,
            "disagreement_valid_mean": float(np.mean(fused_payload["disagreement"][valid_mask])) if np.any(valid_mask) else 0.0,
            "high_frequency_confidence_valid_mean": float(np.mean(fused_payload["high_frequency_confidence"][valid_mask])) if np.any(valid_mask) else 0.0,
        },
        "fusion_case_counts": {
            "uncovered": int(np.sum(fused_payload["fusion_case"] == 0)),
            "overlap_consistent": int(np.sum(fused_payload["fusion_case"] == 1)),
            "overlap_highfreq_rejected": int(np.sum(fused_payload["fusion_case"] == 2)),
            "overlap_lowfreq_rejected": int(np.sum(fused_payload["fusion_case"] == 3)),
            "single_view_low_confidence": int(np.sum(fused_payload["fusion_case"] == 4)),
        },
        "partition_params": partition_params,
    }
    summary_path = output_dir / "prior_fusion_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
