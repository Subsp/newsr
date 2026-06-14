import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SplattingSettings, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state


def parse_rgb(value: str):
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB triplet like '1,0,0', got: {value}")
    return parts


def build_override_colors(gaussians, highlight_mode: str, highlight_tag: int, base_color, highlight_color):
    num_gaussians = gaussians.get_xyz.shape[0]
    colors = torch.tensor(base_color, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(num_gaussians, 1)

    source_tag = gaussians._source_tag
    if highlight_mode == "added":
        mask = source_tag != int(GaussianSourceTag.ORIGINAL)
    elif highlight_mode == "prior":
        mask = source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    elif highlight_mode == "probe":
        mask = source_tag == int(GaussianSourceTag.EXTENSION_PROBE)
    elif highlight_mode == "tag":
        mask = source_tag == int(highlight_tag)
    else:
        raise ValueError(f"Unsupported highlight_mode: {highlight_mode}")

    colors[mask] = torch.tensor(highlight_color, dtype=torch.float32, device="cuda")
    return colors


def build_highlight_mask(gaussians, highlight_mode: str, highlight_tag: int):
    source_tag = gaussians._source_tag
    if highlight_mode == "added":
        return source_tag != int(GaussianSourceTag.ORIGINAL)
    if highlight_mode == "prior":
        return source_tag == int(GaussianSourceTag.PRIOR_INJECTED)
    if highlight_mode == "probe":
        return source_tag == int(GaussianSourceTag.EXTENSION_PROBE)
    if highlight_mode == "tag":
        return source_tag == int(highlight_tag)
    raise ValueError(f"Unsupported highlight_mode: {highlight_mode}")


def build_highlight_only_colors(gaussians, highlight_mask, highlight_color):
    num_gaussians = gaussians.get_xyz.shape[0]
    colors = torch.zeros((num_gaussians, 3), dtype=torch.float32, device="cuda")
    colors[highlight_mask] = torch.tensor(highlight_color, dtype=torch.float32, device="cuda")
    return colors


def render_set(model_path, output_name, split_name, iteration, views, gaussians, pipeline, background, splat_args, highlight_only_color, highlight_color, overlay_alpha):
    split_root = os.path.join(model_path, output_name, split_name, f"ours_{iteration}")
    base_path = os.path.join(split_root, "base")
    highlight_path = os.path.join(split_root, "highlight_only")
    overlay_path = os.path.join(split_root, "overlay")
    makedirs(base_path, exist_ok=True)
    makedirs(highlight_path, exist_ok=True)
    makedirs(overlay_path, exist_ok=True)
    highlight_color_tensor = torch.tensor(highlight_color, dtype=torch.float32, device="cuda").view(3, 1, 1)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {split_name}")):
        base = render(
            view,
            gaussians,
            pipeline,
            background,
            splat_args=splat_args,
        )["render"]
        highlight_only = render(
            view,
            gaussians,
            pipeline,
            background,
            splat_args=splat_args,
            override_color=highlight_only_color,
        )["render"]
        mask = highlight_only.max(dim=0, keepdim=True).values.clamp(0.0, 1.0)
        overlay = torch.clamp(base * (1.0 - overlay_alpha * mask) + highlight_color_tensor * (overlay_alpha * mask), 0.0, 1.0)

        filename = f"{idx:05d}.png"
        torchvision.utils.save_image(base[0:3], os.path.join(base_path, filename))
        torchvision.utils.save_image(highlight_only[0:3], os.path.join(highlight_path, filename))
        torchvision.utils.save_image(overlay[0:3], os.path.join(overlay_path, filename))


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, splat_args, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=args.skip_test, skip_train=False)

        cams = scene.getTrainCameras()
        gaussians.compute_3D_filter(cams.copy())

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        highlight_color = parse_rgb(args.highlight_color)
        highlight_mask = build_highlight_mask(
            gaussians,
            highlight_mode=args.highlight_mode,
            highlight_tag=args.highlight_tag,
        )
        highlight_only_color = build_highlight_only_colors(
            gaussians,
            highlight_mask=highlight_mask,
            highlight_color=highlight_color,
        )

        if not skip_train:
            render_set(
                dataset.model_path,
                args.output_name,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                splat_args,
                highlight_only_color,
                highlight_color,
                args.overlay_alpha,
            )
        if not skip_test:
            render_set(
                dataset.model_path,
                args.output_name,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                splat_args,
                highlight_only_color,
                highlight_color,
                args.overlay_alpha,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Render Gaussians with newly added points highlighted in red.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_name", type=str, default="highlight_added")
    parser.add_argument(
        "--highlight_mode",
        choices=["added", "probe", "prior", "tag"],
        default="added",
        help="Which Gaussians to highlight in red.",
    )
    parser.add_argument("--highlight_tag", type=int, default=int(GaussianSourceTag.EXTENSION_PROBE))
    parser.add_argument("--highlight_color", type=str, default="1.0,0.0,0.0")
    parser.add_argument("--overlay_alpha", type=float, default=0.75)
    args = get_combined_args(parser)

    print("Rendering highlight view for " + args.model_path)
    splat_args = ss.get_settings(args)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, splat_args, args)
