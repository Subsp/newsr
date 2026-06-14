import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SplattingSettings, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from select_mesh_outside_gaussians_v0 import (
    load_triangle_mesh,
    query_mesh_surface,
    save_point_cloud,
    stats_from_array,
)
from utils.general_utils import safe_state


def parse_rgb(value: str) -> List[float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB triplet like '1,0,0', got: {value}")
    return parts


def build_highlight_only_colors(total: int, highlight_ids: np.ndarray, highlight_color: List[float]) -> torch.Tensor:
    colors = torch.zeros((int(total), 3), dtype=torch.float32, device="cuda")
    if highlight_ids.size > 0:
        colors[torch.as_tensor(highlight_ids, device="cuda", dtype=torch.long)] = torch.tensor(
            highlight_color,
            dtype=torch.float32,
            device="cuda",
        )
    return colors


def render_split(
    split_root: Path,
    views,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    splat_args,
    highlight_only_color: torch.Tensor,
    highlight_color: List[float],
    overlay_alpha: float,
):
    base_path = split_root / "base"
    highlight_path = split_root / "highlight_only"
    overlay_path = split_root / "overlay"
    base_path.mkdir(parents=True, exist_ok=True)
    highlight_path.mkdir(parents=True, exist_ok=True)
    overlay_path.mkdir(parents=True, exist_ok=True)

    highlight_color_tensor = torch.tensor(highlight_color, dtype=torch.float32, device="cuda").view(3, 1, 1)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {split_root.name}")):
        base = render(
            view,
            gaussians,
            pipeline,
            background,
            splat_args=splat_args,
        )["render"][0:3]
        highlight_only = render(
            view,
            gaussians,
            pipeline,
            background,
            splat_args=splat_args,
            override_color=highlight_only_color,
        )["render"][0:3]
        mask = highlight_only.max(dim=0, keepdim=True).values.clamp(0.0, 1.0)
        overlay = torch.clamp(
            base * (1.0 - float(overlay_alpha) * mask)
            + highlight_color_tensor * (float(overlay_alpha) * mask),
            0.0,
            1.0,
        )

        filename = f"{idx:05d}.png"
        torchvision.utils.save_image(base, base_path / filename)
        torchvision.utils.save_image(highlight_only, highlight_path / filename)
        torchvision.utils.save_image(overlay, overlay_path / filename)


def main():
    parser = ArgumentParser(
        description="Render SOF Gaussians highlighted by distance to a reference mesh surface."
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--surface_distance_threshold", type=float, default=0.03)
    parser.add_argument("--surface_query_mode", choices=["auto", "exact", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=500000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--highlight_color", type=str, default="1.0,0.1,0.1")
    parser.add_argument("--overlay_alpha", type=float, default=0.75)
    parser.add_argument("--output_name", type=str, default="highlight_mesh_distance_v0")
    parser.add_argument("--preview_max_points", type=int, default=200000)
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe_args = pipeline.extract(args)
    splat_args = ss.get_settings(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=args.skip_test, skip_train=False)
    train_cameras = scene.getTrainCameras()
    gaussians.compute_3D_filter(train_cameras.copy())

    mesh_obj = load_triangle_mesh(args.mesh_path)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    source_tag = gaussians._source_tag.detach().cpu().numpy().astype(np.int32, copy=False)

    print(f"[mesh-distance-highlight] querying surface distance for {xyz.shape[0]} gaussians")
    surface_payload = query_mesh_surface(
        mesh_obj=mesh_obj,
        points_xyz=xyz,
        mode=str(args.surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )
    surface_distance = surface_payload["surface_distance"].astype(np.float32, copy=False)
    highlight_mask_np = surface_distance > float(args.surface_distance_threshold)
    highlight_ids = np.flatnonzero(highlight_mask_np).astype(np.int64, copy=False)

    output_root = Path(dataset.model_path) / args.output_name / f"ours_{scene.loaded_iter}"
    output_root.mkdir(parents=True, exist_ok=True)
    preview_path = output_root / "highlighted_gaussians.ply"
    payload_path = output_root / "mesh_distance_payload.pt"
    summary_path = output_root / "summary.json"

    highlight_color = parse_rgb(args.highlight_color)
    highlight_only_color = build_highlight_only_colors(
        total=int(xyz.shape[0]),
        highlight_ids=highlight_ids,
        highlight_color=highlight_color,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not args.skip_train:
        render_split(
            output_root / "train",
            scene.getTrainCameras(),
            gaussians,
            pipe_args,
            background,
            splat_args,
            highlight_only_color,
            highlight_color,
            float(args.overlay_alpha),
        )
    if not args.skip_test:
        render_split(
            output_root / "test",
            scene.getTestCameras(),
            gaussians,
            pipe_args,
            background,
            splat_args,
            highlight_only_color,
            highlight_color,
            float(args.overlay_alpha),
        )

    save_point_cloud(
        xyz[highlight_ids],
        preview_path,
        color=tuple(int(255.0 * np.clip(v, 0.0, 1.0)) for v in highlight_color),
        max_points=int(args.preview_max_points),
    )
    torch.save(
        {
            "highlight_mask": torch.from_numpy(highlight_mask_np.copy()),
            "highlight_ids": torch.from_numpy(highlight_ids.copy()),
            "surface_distance": torch.from_numpy(surface_distance.copy()),
            "nearest_surface_point": torch.from_numpy(surface_payload["nearest_surface_point"].copy()),
            "nearest_surface_normal": torch.from_numpy(surface_payload["nearest_surface_normal"].copy()),
            "nearest_face_id": torch.from_numpy(surface_payload["nearest_face_id"].copy()),
            "opacity": torch.from_numpy(opacity.copy()),
            "source_tag": torch.from_numpy(source_tag.copy()),
        },
        payload_path,
    )

    summary = {
        "mode": "render_highlight_mesh_distance_gaussians_v0",
        "scene_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "surface_query_mode_requested": str(args.surface_query_mode),
        "surface_query_mode_used": str(surface_payload["surface_query_mode_used"]),
        "surface_distance_threshold": float(args.surface_distance_threshold),
        "highlighted_gaussian_count": int(highlight_ids.shape[0]),
        "highlighted_gaussian_ratio": float(highlight_ids.shape[0] / max(int(xyz.shape[0]), 1)),
        "surface_distance_all": stats_from_array(surface_distance),
        "surface_distance_highlighted": stats_from_array(surface_distance[highlight_mask_np]),
        "payload_path": str(payload_path.resolve()),
        "preview_ply_path": str(preview_path.resolve()),
        "output_root": str(output_root.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved mesh-distance highlight bundle to: {output_root}")


if __name__ == "__main__":
    main()
