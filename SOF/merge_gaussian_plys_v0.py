import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn

from arguments import ModelParams
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from utils.general_utils import safe_state
from utils.system_utils import mkdir_p


def metadata_path_for_ply(ply_path: str) -> str:
    return str(Path(ply_path).parent / "gaussian_tags.pt")


def load_gaussians(sh_degree: int, ply_path: str, source_tag: int) -> GaussianModel:
    model = GaussianModel(sh_degree)
    model.load_ply(ply_path)
    tags_path = metadata_path_for_ply(ply_path)
    if Path(tags_path).is_file():
        model.load_tracking_metadata(tags_path)
    else:
        model.init_tracking_state(model.get_xyz.shape[0], source_tag=source_tag)
    return model


def copy_render_config(src_model_path: str, dst_model_path: Path):
    src = Path(src_model_path)
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "config.json", "cameras.json"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def concat_tracking(base: GaussianModel, extra: GaussianModel):
    return {
        "source_tag": torch.cat((base._source_tag, extra._source_tag), dim=0),
        "seed_id": torch.cat((base._seed_id, extra._seed_id), dim=0),
        "generation": torch.cat((base._generation, extra._generation), dim=0),
        "edge_touched": torch.cat((base._edge_touched, extra._edge_touched), dim=0),
        "edge_touch_iter": torch.cat((base._edge_touch_iter, extra._edge_touch_iter), dim=0),
    }


def merge_models(base: GaussianModel, extra: GaussianModel) -> GaussianModel:
    if base.use_SBs != extra.use_SBs:
        raise ValueError(f"Cannot merge different feature layouts: base.use_SBs={base.use_SBs}, extra.use_SBs={extra.use_SBs}")
    if base._features_dc.shape[1:] != extra._features_dc.shape[1:]:
        raise ValueError(f"DC feature shape mismatch: {base._features_dc.shape} vs {extra._features_dc.shape}")
    if base._features_rest.shape[1:] != extra._features_rest.shape[1:]:
        raise ValueError(f"SH feature shape mismatch: {base._features_rest.shape} vs {extra._features_rest.shape}")

    merged = GaussianModel(base.max_sh_degree, use_SBs=base.use_SBs)
    merged.active_sh_degree = max(base.active_sh_degree, extra.active_sh_degree)
    merged.spatial_lr_scale = base.spatial_lr_scale
    merged._xyz = nn.Parameter(torch.cat((base._xyz.detach(), extra._xyz.detach()), dim=0), requires_grad=False)
    merged._features_dc = nn.Parameter(torch.cat((base._features_dc.detach(), extra._features_dc.detach()), dim=0), requires_grad=False)
    merged._features_rest = nn.Parameter(torch.cat((base._features_rest.detach(), extra._features_rest.detach()), dim=0), requires_grad=False)
    merged._opacity = nn.Parameter(torch.cat((base._opacity.detach(), extra._opacity.detach()), dim=0), requires_grad=False)
    merged._scaling = nn.Parameter(torch.cat((base._scaling.detach(), extra._scaling.detach()), dim=0), requires_grad=False)
    merged._rotation = nn.Parameter(torch.cat((base._rotation.detach(), extra._rotation.detach()), dim=0), requires_grad=False)

    if base.filter_3D.shape[0] == base.get_xyz.shape[0] and extra.filter_3D.shape[0] == extra.get_xyz.shape[0]:
        merged.filter_3D = torch.cat((base.filter_3D.detach(), extra.filter_3D.detach()), dim=0)
    else:
        merged.filter_3D = torch.zeros((merged.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
    merged.max_radii2D = torch.zeros((merged.get_xyz.shape[0],), dtype=torch.float32, device="cuda")
    merged.restore_tracking_state(concat_tracking(base, extra))
    return merged


def main():
    parser = ArgumentParser(description="Merge two Gaussian PLY files into one renderable SOF/3DGS model.")
    model = ModelParams(parser)
    parser.add_argument("--base_ply", type=str, required=True)
    parser.add_argument("--extra_ply", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--output_iteration", type=int, default=0)
    parser.add_argument("--copy_config_from", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    safe_state(args.quiet)

    base = load_gaussians(model.extract(args).sh_degree, args.base_ply, int(GaussianSourceTag.ORIGINAL))
    extra = load_gaussians(model.extract(args).sh_degree, args.extra_ply, int(GaussianSourceTag.PRIOR_INJECTED))
    merged = merge_models(base, extra)

    output_model_path = Path(args.output_model_path)
    copy_render_config(args.copy_config_from, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(args.output_iteration)}"
    mkdir_p(str(point_dir))
    merged.save_ply(str(point_dir / "point_cloud.ply"))
    merged.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_gaussians": int(merged.get_xyz.shape[0]),
                "base_gaussians": int(base.get_xyz.shape[0]),
                "extra_gaussians": int(extra.get_xyz.shape[0]),
                "base_ply": args.base_ply,
                "extra_ply": args.extra_ply,
            },
            f,
            indent=2,
        )
    print(
        "[merge-gaussian-plys] saved merged model: "
        f"{point_dir / 'point_cloud.ply'} "
        f"(base={base.get_xyz.shape[0]}, extra={extra.get_xyz.shape[0]}, total={merged.get_xyz.shape[0]})"
    )


if __name__ == "__main__":
    main()
