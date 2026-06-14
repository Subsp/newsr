from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, resolution: int | float):
    class _Args:
        pass

    args = _Args()
    args.sh_degree = 3
    args.source_path = scene_root
    args.model_path = model_path
    args.images = images_subdir
    args.resolution = resolution
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.alpha_mask = False
    args.init_type = "sfm"
    return args


def _get_splat_settings(model_path: str) -> ExtendedSettings:
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        return ExtendedSettings.from_json(str(config_path))
    return ExtendedSettings()


def parse_args():
    parser = argparse.ArgumentParser(description="Export SOF geometry oracle buffers (depth/normal/alpha/valid).")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--resolution", type=float, default=-1)
    parser.add_argument("--alpha_threshold", type=float, default=1e-4)
    parser.add_argument("--save_preview_json", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    scene_root = str(Path(args.scene_root).expanduser().resolve())
    model_path = str(Path(args.model_path).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    depth_dir = output_root / "depth"
    normal_dir = output_root / "normal"
    alpha_dir = output_root / "alpha"
    valid_dir = output_root / "valid"
    for path in (output_root, depth_dir, normal_dir, alpha_dir, valid_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset_args = _build_dataset_args(
        scene_root=scene_root,
        model_path=model_path,
        images_subdir=args.images_subdir,
        resolution=args.resolution,
    )
    dataset = ModelParams(None).extract(dataset_args)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=False)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        skip_train=False,
        skip_test=True,
    )
    cameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras.copy())

    bg = torch.zeros((3,), dtype=torch.float32, device="cuda")
    splat_args = _get_splat_settings(model_path)

    frame_summaries = []
    for idx, camera in enumerate(cameras):
        render_pkg = render_simple(camera, gaussians, bg, splat_args=splat_args)
        depth = render_pkg["depth"][0].detach().cpu().numpy().astype(np.float32)
        normal = render_pkg["normal"].detach().cpu().numpy().astype(np.float32)
        alpha = render_pkg["alpha"][0].detach().cpu().numpy().astype(np.float32)
        valid = (np.isfinite(depth) & (depth > 1e-6) & (alpha > float(args.alpha_threshold))).astype(np.uint8)

        np.save(depth_dir / f"{camera.image_name}.npy", depth)
        np.save(normal_dir / f"{camera.image_name}.npy", normal)
        np.save(alpha_dir / f"{camera.image_name}.npy", alpha)
        np.save(valid_dir / f"{camera.image_name}.npy", valid)

        valid_ratio = float(valid.mean()) if valid.size > 0 else 0.0
        valid_depth = depth[valid.astype(bool)]
        frame_summaries.append(
            {
                "index": idx,
                "image_name": str(camera.image_name),
                "valid_ratio": valid_ratio,
                "depth_min": float(valid_depth.min()) if valid_depth.size > 0 else None,
                "depth_max": float(valid_depth.max()) if valid_depth.size > 0 else None,
                "depth_median": float(np.median(valid_depth)) if valid_depth.size > 0 else None,
            }
        )

    meta = {
        "oracle_source": "SOF",
        "oracle_iteration": int(scene.loaded_iter if scene.loaded_iter is not None else args.iteration),
        "scene_root": scene_root,
        "model_path": model_path,
        "images_subdir": args.images_subdir,
        "depth_convention": "camera_z_depth",
        "normal_space": "camera_space",
        "valid_rule": f"finite depth & depth>1e-6 & alpha>{float(args.alpha_threshold)}",
        "frame_count": len(frame_summaries),
    }
    (output_root / "formal_oracle_sof_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.save_preview_json:
        (output_root / "formal_oracle_sof_frames.json").write_text(
            json.dumps(frame_summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"[sof-oracle] exported {len(frame_summaries)} train-view buffers")
    print(f"[sof-oracle] output root : {output_root}")
    print(f"[sof-oracle] depth dir   : {depth_dir}")
    print(f"[sof-oracle] normal dir  : {normal_dir}")
    print(f"[sof-oracle] valid dir   : {valid_dir}")
    print(f"[sof-oracle] iteration   : {meta['oracle_iteration']}")


if __name__ == "__main__":
    main()
