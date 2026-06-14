#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import json
import trimesh
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments import ModelParams, PipelineParams, SplattingSettings
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import GaussianModel
from utils.mesh_fusion_render import (
    build_runtime_mesh_core_mask,
    compose_mesh_fusion_detail,
    load_fusion_region_mask,
    load_mesh_fusion_payload,
    render_bounded_carrier_layer,
)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, splat_args: ExtendedSettings, mesh_fusion_runtime=None, output_suffix: str = ""):
    output_name = "ours_{}{}".format(iteration, output_suffix)
    render_path = os.path.join(model_path, name, output_name, "renders")
    gts_path = os.path.join(model_path, name, output_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if mesh_fusion_runtime is not None and mesh_fusion_runtime["debug"]:
        makedirs(os.path.join(model_path, name, output_name, "mesh_fusion_debug", "base"), exist_ok=True)
        makedirs(os.path.join(model_path, name, output_name, "mesh_fusion_debug", "layer"), exist_ok=True)
        makedirs(os.path.join(model_path, name, output_name, "mesh_fusion_debug", "gate"), exist_ok=True)

    kernel_size = float(getattr(views[0], "kernel_size", 0.0)) if views else 0.0
    vanilla_mip_mode = bool(getattr(views[0], "vanilla_mip_mode", False)) if views else False
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            splat_args=splat_args,
            kernel_size=kernel_size,
            vanilla_mip_mode=vanilla_mip_mode,
        )["render"]
        output_rgb = rendering[0:3]
        if mesh_fusion_runtime is not None:
            height, width = output_rgb.shape[1], output_rgb.shape[2]
            region_mask = load_fusion_region_mask(
                mesh_fusion_runtime["mask_dir"],
                view.image_name,
                height=height,
                width=width,
            )
            if region_mask is None:
                region_mask = build_runtime_mesh_core_mask(
                    mesh_fusion_runtime["mesh"],
                    view,
                    height=height,
                    width=width,
                    erode_kernel=mesh_fusion_runtime["core_erode_kernel"],
                    depth_min=mesh_fusion_runtime["depth_min"],
                )
            if mesh_fusion_runtime["meshgs"] is not None:
                mesh_render = render(
                    view,
                    mesh_fusion_runtime["meshgs"],
                    pipeline,
                    mesh_fusion_runtime["mesh_background"],
                    splat_args=splat_args,
                    kernel_size=kernel_size,
                    vanilla_mip_mode=vanilla_mip_mode,
                )["render"]
                mesh_rgb = torch.clamp(mesh_render[:3], 0.0, 1.0)
                mesh_gate = torch.clamp(mesh_render[7], 0.0, 1.0)
            else:
                mesh_rgb, mesh_gate = render_bounded_carrier_layer(
                    view,
                    mesh_fusion_runtime["payload"],
                    height=height,
                    width=width,
                    region_mask=region_mask,
                    min_confidence=mesh_fusion_runtime["min_confidence"],
                    disagreement_low=mesh_fusion_runtime["disagreement_low"],
                    disagreement_high=mesh_fusion_runtime["disagreement_high"],
                    min_radius_px=mesh_fusion_runtime["min_radius_px"],
                    max_radius_px=mesh_fusion_runtime["max_radius_px"],
                    radius_scale=mesh_fusion_runtime["radius_scale"],
                    depth_min=mesh_fusion_runtime["depth_min"],
                    z_epsilon=mesh_fusion_runtime["z_epsilon"],
                )
            composed_rgb, final_gate = compose_mesh_fusion_detail(
                output_rgb,
                mesh_rgb,
                mesh_gate,
                region_mask=region_mask,
                gate_max=mesh_fusion_runtime["gate_max"],
                lowpass_kernel=mesh_fusion_runtime["lowpass_kernel"],
                low_gate_scale=mesh_fusion_runtime["low_gate_scale"],
                high_gate_scale=mesh_fusion_runtime["high_gate_scale"],
                low_delta_start=mesh_fusion_runtime["low_delta_start"],
                low_delta_end=mesh_fusion_runtime["low_delta_end"],
            )
            if mesh_fusion_runtime["debug"]:
                debug_root = os.path.join(model_path, name, output_name, "mesh_fusion_debug")
                filename = '{0:05d}'.format(idx) + ".png"
                torchvision.utils.save_image(output_rgb, os.path.join(debug_root, "base", filename))
                torchvision.utils.save_image(mesh_rgb, os.path.join(debug_root, "layer", filename))
                torchvision.utils.save_image(final_gate[None], os.path.join(debug_root, "gate", filename))
            output_rgb = composed_rgb
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(output_rgb, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def build_mesh_fusion_runtime(runtime_args, sh_degree):
    if not runtime_args.mesh_fusion_payload and not runtime_args.mesh_fusion_meshgs_ply:
        return None
    payload = load_mesh_fusion_payload(runtime_args.mesh_fusion_payload) if runtime_args.mesh_fusion_payload else None
    meshgs = None
    if runtime_args.mesh_fusion_meshgs_ply:
        meshgs = GaussianModel(sh_degree)
        meshgs.load_ply(runtime_args.mesh_fusion_meshgs_ply)
        meshgs.active_sh_degree = min(meshgs.active_sh_degree, meshgs.max_sh_degree)
    mesh = None
    if runtime_args.mesh_fusion_mesh_path:
        mesh = trimesh.load_mesh(runtime_args.mesh_fusion_mesh_path, process=False)
    if runtime_args.mesh_fusion_mask_dir is None and mesh is None:
        print("[mesh-fusion] warning: no mask dir or mesh path provided; composition will rely only on carrier footprints.")
    return {
        "payload": payload,
        "meshgs": meshgs,
        "mesh_background": torch.zeros((3,), dtype=torch.float32, device="cuda"),
        "mask_dir": runtime_args.mesh_fusion_mask_dir,
        "mesh": mesh,
        "core_erode_kernel": int(runtime_args.mesh_fusion_core_erode_kernel),
        "gate_max": float(runtime_args.mesh_fusion_gate_max),
        "lowpass_kernel": int(runtime_args.mesh_fusion_lowpass_kernel),
        "low_gate_scale": float(runtime_args.mesh_fusion_low_gate_scale),
        "high_gate_scale": float(runtime_args.mesh_fusion_high_gate_scale),
        "low_delta_start": float(runtime_args.mesh_fusion_low_delta_start),
        "low_delta_end": float(runtime_args.mesh_fusion_low_delta_end),
        "min_confidence": float(runtime_args.mesh_fusion_min_confidence),
        "disagreement_low": float(runtime_args.mesh_fusion_disagreement_low),
        "disagreement_high": float(runtime_args.mesh_fusion_disagreement_high),
        "min_radius_px": float(runtime_args.mesh_fusion_min_radius_px),
        "max_radius_px": float(runtime_args.mesh_fusion_max_radius_px),
        "radius_scale": float(runtime_args.mesh_fusion_radius_scale),
        "depth_min": float(runtime_args.mesh_fusion_depth_min),
        "z_epsilon": float(runtime_args.mesh_fusion_z_epsilon),
        "debug": bool(runtime_args.mesh_fusion_debug),
    }


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, splat_args: ExtendedSettings, runtime_args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            skip_test=skip_test,
            skip_train=skip_train,
        )
        split_camera_lists = []
        if not skip_train:
            split_camera_lists.append(scene.getTrainCameras())
        if not skip_test:
            split_camera_lists.append(scene.getTestCameras())

        for split_cams in split_camera_lists:
            for cam in split_cams:
                setattr(cam, "kernel_size", float(getattr(dataset, "kernel_size", 0.0)))
                setattr(cam, "vanilla_mip_mode", bool(getattr(dataset, "vanilla_mip_mode", False)))

        filter_cams = None
        if not skip_train:
            filter_cams = scene.getTrainCameras()
        elif not skip_test:
            filter_cams = scene.getTestCameras()
        if filter_cams is not None and len(filter_cams) > 0:
            gaussians.compute_3D_filter(filter_cams.copy())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        mesh_fusion_runtime = build_mesh_fusion_runtime(runtime_args, dataset.sh_degree)
        output_suffix = runtime_args.mesh_fusion_output_suffix if mesh_fusion_runtime is not None else ""

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, splat_args, mesh_fusion_runtime, output_suffix)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, splat_args, mesh_fusion_runtime, output_suffix)
        
        # write number of gaussians too
        num_gaussians = scene.gaussians.get_xyz.shape[0]
        loaded_iter = scene.loaded_iter if scene.loaded_iter is not None else iteration
        point_cloud_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{loaded_iter}")
        makedirs(point_cloud_path, exist_ok=True)
        with open(os.path.join(point_cloud_path, 'num_gaussians.json'), 'w') as fp:
            json.dump(obj={
                "num_gaussians": num_gaussians,
            }, fp=fp, indent=2)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_fusion_payload", type=str, default=None)
    parser.add_argument("--mesh_fusion_meshgs_ply", type=str, default=None)
    parser.add_argument("--mesh_fusion_mask_dir", type=str, default=None)
    parser.add_argument("--mesh_fusion_mesh_path", type=str, default=None)
    parser.add_argument("--mesh_fusion_core_erode_kernel", type=int, default=17)
    parser.add_argument("--mesh_fusion_gate_max", type=float, default=0.35)
    parser.add_argument("--mesh_fusion_lowpass_kernel", type=int, default=15)
    parser.add_argument("--mesh_fusion_low_gate_scale", type=float, default=0.1)
    parser.add_argument("--mesh_fusion_high_gate_scale", type=float, default=1.0)
    parser.add_argument("--mesh_fusion_low_delta_start", type=float, default=0.08)
    parser.add_argument("--mesh_fusion_low_delta_end", type=float, default=0.18)
    parser.add_argument("--mesh_fusion_min_confidence", type=float, default=0.05)
    parser.add_argument("--mesh_fusion_disagreement_low", type=float, default=0.03)
    parser.add_argument("--mesh_fusion_disagreement_high", type=float, default=0.08)
    parser.add_argument("--mesh_fusion_min_radius_px", type=float, default=1.0)
    parser.add_argument("--mesh_fusion_max_radius_px", type=float, default=12.0)
    parser.add_argument("--mesh_fusion_radius_scale", type=float, default=1.0)
    parser.add_argument("--mesh_fusion_depth_min", type=float, default=0.01)
    parser.add_argument("--mesh_fusion_z_epsilon", type=float, default=0.02)
    parser.add_argument("--mesh_fusion_debug", action="store_true")
    parser.add_argument("--mesh_fusion_output_suffix", type=str, default="_meshfusion_v0")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    splat_args = ss.get_settings(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, splat_args, args)
