import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SplattingSettings
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.sugar_like_meshgs_model import SugarLikeMeshGaussianModel
from train_meshgs_prior_v0 import load_mask_cached, load_rgb_cached, masked_l1, render_alpha_mask
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir
from utils.system_utils import mkdir_p


def save_sugar_like_meshgs(meshgs: SugarLikeMeshGaussianModel, output_dir: Path, iteration: int, args, init_summary):
    point_dir = output_dir / "point_cloud" / f"iteration_{iteration}"
    mkdir_p(str(point_dir))

    classic = meshgs.materialize_gaussian_model()
    classic.save_ply(str(point_dir / "point_cloud.ply"))
    classic.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    meshgs.save_bound_metadata(str(point_dir / "mesh_bound_state.pt"))

    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(classic.get_xyz.shape[0])}, f, indent=2)
    with open(output_dir / "sugar_like_meshgs_prior_v0_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(output_dir / "sugar_like_meshgs_prior_v0_summary.json", "w", encoding="utf-8") as f:
        json.dump(init_summary, f, indent=2)


def train_sugar_like_meshgs(dataset, pipe, splat_args, args):
    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, dummy_gaussians, shuffle=True, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras().copy()

    meshgs = SugarLikeMeshGaussianModel(dataset.sh_degree, use_SBs=False)
    init_summary = meshgs.initialize_from_fused_carrier_payload(
        mesh_path=args.mesh_path,
        payload_path=args.mesh_fusion_payload,
        min_confidence=float(args.meshgs_min_confidence),
        max_disagreement=float(args.meshgs_max_disagreement),
        min_views=int(args.meshgs_min_views),
        max_count=int(args.meshgs_max_count),
        seed=int(args.meshgs_seed),
        scale_multiplier=float(args.meshgs_scale_multiplier),
        thickness_multiplier=float(args.meshgs_thickness_multiplier),
        init_opacity=float(args.meshgs_init_opacity),
        learn_surface_vertices=bool(args.learn_surface_vertices),
        learn_plane_scales=bool(args.learn_plane_scales),
        learn_inplane_rotation=bool(args.learn_inplane_rotation),
        build_normal_pairs=not bool(args.disable_normal_consistency_pairs),
        max_normal_pairs=int(args.max_normal_consistency_pairs),
    )
    init_summary.update(
        {
            "mode": "sugar_like_meshgs_prior_v0",
            "mesh_path": args.mesh_path,
            "mesh_fusion_payload": args.mesh_fusion_payload,
            "learn_surface_vertices": bool(args.learn_surface_vertices),
            "learn_plane_scales": bool(args.learn_plane_scales),
            "learn_inplane_rotation": bool(args.learn_inplane_rotation),
        }
    )
    print(
        "[sugar-like meshGS] initialized "
        f"{init_summary['selected_count']}/{init_summary['payload_count']} mesh-bound Gaussians"
    )
    print(f"[sugar-like meshGS] normal consistency pairs: {init_summary['normal_consistency_pairs']}")

    optimizer = meshgs.build_optimizer(
        feature_lr=float(args.meshgs_feature_lr),
        opacity_lr=float(args.meshgs_opacity_lr),
        surface_vertex_lr=float(args.surface_vertex_lr),
        plane_scale_lr=float(args.plane_scale_lr),
        inplane_rotation_lr=float(args.inplane_rotation_lr),
    )

    prior_index = index_image_dir(args.prior_dir)
    prior_cache = {}
    mask_cache = {}
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    progress = tqdm(range(1, int(args.iterations) + 1), desc="Training sugar-like meshGS")
    ema_loss = 0.0
    ema_prior = 0.0
    ema_normal = 0.0

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

        loss = loss_prior
        loss_normal = torch.zeros((), dtype=torch.float32, device="cuda")
        if float(args.normal_consistency_lambda) > 0:
            loss_normal = meshgs.normal_consistency_loss()
            loss = loss + float(args.normal_consistency_lambda) * loss_normal
        if float(args.meshgs_lambda_opacity) > 0:
            loss = loss + float(args.meshgs_lambda_opacity) * meshgs.get_opacity.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if bool(args.learn_inplane_rotation):
            with torch.no_grad():
                norm = torch.linalg.norm(meshgs._surface_inplane_rotation, dim=-1, keepdim=True).clamp_min(1e-8)
                meshgs._surface_inplane_rotation.div_(norm)

        ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
        ema_prior = 0.4 * float(loss_prior.item()) + 0.6 * ema_prior
        ema_normal = 0.4 * float(loss_normal.item()) + 0.6 * ema_normal
        if iteration % 10 == 0:
            progress.set_postfix(
                {
                    "loss": f"{ema_loss:.6f}",
                    "prior": f"{ema_prior:.6f}",
                    "normal": f"{ema_normal:.6f}",
                }
            )
        if iteration in args.save_iterations or iteration == int(args.iterations):
            save_sugar_like_meshgs(meshgs, output_dir, iteration, args, init_summary)

    save_sugar_like_meshgs(meshgs, output_dir, int(args.iterations), args, init_summary)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train SuGaR-like mesh-bound GS from fused prior carriers.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--mesh_fusion_payload", type=str, required=True)
    parser.add_argument("--mesh_fusion_mask_dir", type=str, default=None)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--meshgs_min_confidence", type=float, default=0.05)
    parser.add_argument("--meshgs_max_disagreement", type=float, default=0.08)
    parser.add_argument("--meshgs_min_views", type=int, default=2)
    parser.add_argument("--meshgs_max_count", type=int, default=0)
    parser.add_argument("--meshgs_seed", type=int, default=0)
    parser.add_argument("--meshgs_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--meshgs_thickness_multiplier", type=float, default=0.5)
    parser.add_argument("--meshgs_init_opacity", type=float, default=0.35)
    parser.add_argument("--meshgs_feature_lr", type=float, default=0.01)
    parser.add_argument("--meshgs_opacity_lr", type=float, default=0.02)
    parser.add_argument("--meshgs_lambda_opacity", type=float, default=1e-4)
    parser.add_argument("--meshgs_min_pixels", type=float, default=64.0)
    parser.add_argument("--meshgs_render_alpha_threshold", type=float, default=1e-4)
    parser.add_argument("--learn_surface_vertices", action="store_true")
    parser.add_argument("--learn_plane_scales", action="store_true")
    parser.add_argument("--learn_inplane_rotation", action="store_true")
    parser.add_argument("--surface_vertex_lr", type=float, default=0.0)
    parser.add_argument("--plane_scale_lr", type=float, default=0.0)
    parser.add_argument("--inplane_rotation_lr", type=float, default=0.0)
    parser.add_argument("--normal_consistency_lambda", type=float, default=0.0)
    parser.add_argument("--disable_normal_consistency_pairs", action="store_true")
    parser.add_argument("--max_normal_consistency_pairs", type=int, default=500000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    safe_state(args.quiet)
    train_sugar_like_meshgs(model.extract(args), pipeline.extract(args), ss.get_settings(args), args)
