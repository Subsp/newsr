#
# Vanilla mip-splatting style training from an externally prepared Gaussian init.
#

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, MeshingParams, SplattingSettings, get_combined_args
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.prior_injection import index_image_dir, load_rgb_image, normalize_image_name

try:
    from fused_ssim import fused_ssim
    _FUSED_SSIM_AVAILABLE = True
except ImportError:
    fused_ssim = None
    _FUSED_SSIM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_FOUND = True
except ImportError:
    SummaryWriter = None
    _TENSORBOARD_FOUND = False


def create_offset_gt(image: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).to(device="cuda")
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= max(width - 1, 1)
    id_coords[..., 1] /= max(height - 1, 1)
    id_coords = id_coords * 2 - 1
    return torch.nn.functional.grid_sample(
        image[None],
        id_coords[None],
        align_corners=True,
        padding_mode="border",
    )[0]


def compute_ssim_loss(image: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    image_batched = image.unsqueeze(0)
    gt_batched = gt_image.unsqueeze(0)
    if _FUSED_SSIM_AVAILABLE:
        return 1.0 - fused_ssim(image_batched, gt_batched)
    return 1.0 - ssim(image_batched, gt_batched)


def _resize_hw3_tensor(image_hw3: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    image = image_hw3.permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    resized = F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).detach().cpu()


def lookup_indexed_image_path(index, image_name: str):
    if index is None:
        return None
    candidates = [
        image_name,
        normalize_image_name(image_name),
        Path(str(image_name)).name,
        Path(str(image_name)).stem,
    ]
    for key in candidates:
        path = index.get(str(key))
        if path is not None:
            return path
    return None


def load_external_rgb_cached(image_name, index, cache, height, width):
    if index is None:
        return None
    key = (image_name, height, width)
    if key not in cache:
        path = lookup_indexed_image_path(index, image_name)
        if path is None:
            cache[key] = None
        else:
            image = load_rgb_image(path)
            if image.shape[0] != height or image.shape[1] != width:
                image = _resize_hw3_tensor(image, target_height=height, target_width=width)
            cache[key] = image
    image = cache[key]
    if image is None:
        return None
    return image.to(device="cuda", dtype=torch.float32)


def prepare_output_and_logger(args, splat_args: ExtendedSettings):
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if hasattr(splat_args, "to_dict"):
        with open(os.path.join(args.model_path, "config.json"), "w", encoding="utf-8") as config_json:
            json.dump(splat_args.to_dict(), config_json)

    tb_writer = None
    if _TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def configure_mesh_as_vanilla(mesh, opt) -> None:
    mesh.lambda_distortion = 0.0
    mesh.lambda_opacity_field = 0.0
    mesh.lambda_extent = 0.0
    mesh.lambda_depth_normal = 0.0
    mesh.distortion_from_iter = int(opt.iterations) + 1
    mesh.depth_normal_from_iter = int(opt.iterations) + 1
    mesh.lambda_smoothness = 0.0
    mesh.abs_grad_for_densification = False
    mesh.clone_with_sampling = False
    mesh.prune_threshold = 0.005
    mesh.opacity_decay = 0.0
    mesh.scale_reg = 0.0
    mesh.opacity_reg = 0.0
    mesh.min_scale_reg = 0.0
    mesh.cap_max = -1


def verify_global_supervision_complete(train_cameras, image_index) -> None:
    missing = [cam.image_name for cam in train_cameras if lookup_indexed_image_path(image_index, cam.image_name) is None]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise FileNotFoundError(
            f"global_image_dir is missing {len(missing)} train views; first few: {preview}"
        )


def training(dataset, opt, pipe, mesh, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, splat_args, runtime_args):
    del testing_iterations, debug_from

    tb_writer = prepare_output_and_logger(dataset, splat_args)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe.convert_SBs_python)
    scene = Scene(dataset, gaussians, shuffle=False, MCMC_init=False)

    train_cameras = scene.getTrainCameras().copy()
    highresolution_index = [idx for idx, camera in enumerate(train_cameras) if camera.image_width >= 800]

    if runtime_args.start_ply:
        start_ply_path = Path(runtime_args.start_ply).expanduser().resolve()
        if not start_ply_path.is_file():
            raise FileNotFoundError(f"start_ply not found: {start_ply_path}")
        gaussians.load_ply(str(start_ply_path))
        if int(runtime_args.start_ply_active_sh_degree) >= 0:
            gaussians.active_sh_degree = min(int(runtime_args.start_ply_active_sh_degree), gaussians.max_sh_degree)
        tags_path = start_ply_path.parent / "gaussian_tags.pt"
        if tags_path.is_file():
            gaussians.load_tracking_metadata(str(tags_path))
        else:
            gaussians.init_tracking_state(gaussians.get_xyz.shape[0])
        print(f"[start-ply] loaded initial Gaussians from {start_ply_path}")

    gaussians.training_setup(opt, mesh, None)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt, mesh, None)
    else:
        first_iter = int(runtime_args.start_ply_iteration) if runtime_args.start_ply else 0
        if first_iter < 0:
            raise ValueError(f"start_ply_iteration must be non-negative, got {first_iter}")
        if first_iter >= int(opt.iterations):
            raise ValueError(
                f"start_ply_iteration ({first_iter}) must be smaller than iterations ({int(opt.iterations)})."
            )

    global_image_index = index_image_dir(runtime_args.global_image_dir) if runtime_args.global_image_dir else None
    global_image_cache = {}
    if global_image_index is not None:
        verify_global_supervision_complete(train_cameras, global_image_index)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians.compute_3D_filter(cameras=train_cameras, CUDA=not pipe.compute_filter3D_python)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    for iteration in range(first_iter + 1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if highresolution_index and random.random() < 0.3 and bool(dataset.sample_more_highres):
            viewpoint_cam = train_cameras[highresolution_index[randint(0, len(highresolution_index) - 1)]]

        if bool(dataset.ray_jitter):
            subpixel_offset = torch.rand(
                (int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2),
                dtype=torch.float32,
                device="cuda",
            ) - 0.5
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size=float(dataset.kernel_size),
            subpixel_offset=subpixel_offset,
            splat_args=splat_args,
            vanilla_mip_mode=True,
        )
        rendering = render_pkg["render"]
        image = rendering[:3] if rendering.shape[0] > 3 else rendering
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = load_external_rgb_cached(
            viewpoint_cam.image_name,
            global_image_index,
            global_image_cache,
            height=image.shape[1],
            width=image.shape[2],
        )
        if global_image_index is None:
            gt_image = viewpoint_cam.original_image.cuda()
        elif gt_image is None:
            raise FileNotFoundError(f"Missing external supervision for train view: {viewpoint_cam.image_name}")
        else:
            gt_image = gt_image.permute(2, 0, 1)

        if bool(dataset.resample_gt_image) and subpixel_offset is not None:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * compute_ssim_loss(image, gt_image)
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Size": f"{len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar("train_loss/l1_loss", Ll1.item(), iteration)
                tb_writer.add_scalar("train_loss/total_loss", loss.item(), iteration)
                tb_writer.add_scalar("total_points", gaussians.get_xyz.shape[0], iteration)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    gaussians.compute_3D_filter(cameras=train_cameras, CUDA=not pipe.compute_filter3D_python)

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations - 100:
                gaussians.compute_3D_filter(cameras=train_cameras, CUDA=not pipe.compute_filter3D_python)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if tb_writer:
        tb_writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Vanilla mip-splatting training from prepared init + external RGB supervision")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MeshingParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_ply", type=str, default=None)
    parser.add_argument("--start_ply_iteration", type=int, default=0)
    parser.add_argument("--start_ply_active_sh_degree", type=int, default=0)
    parser.add_argument("--global_image_dir", type=str, default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    args = get_combined_args(parser)
    if args.start_checkpoint and args.start_ply:
        raise ValueError("Pass only one of --start_checkpoint or --start_ply.")
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    splat_args = ss.get_settings(args)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    mesh = mp.extract(args)
    configure_mesh_as_vanilla(mesh, opt)

    training(
        dataset,
        opt,
        pipe,
        mesh,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        splat_args,
        args,
    )
