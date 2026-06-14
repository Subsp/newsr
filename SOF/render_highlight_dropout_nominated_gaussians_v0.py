import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args, SplattingSettings
from gaussian_renderer import render
from refine_gaussians_with_patch_feedback_v0 import (
    aggregate_patch_metrics,
    assign_gaussians_to_patches,
    index_cameras,
    load_dropout_snapshot,
    load_mask_from_payload,
    normalize_image_name,
    tensor_to_numpy,
)
from scene import Scene
from scene.gaussian_model import GaussianModel
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
            highlight_color, dtype=torch.float32, device="cuda"
        )
    return colors


def collect_candidate_support_ids(
    gaussians: GaussianModel,
    candidate_payload: dict,
    patch_bank: Dict[str, np.ndarray],
    dropout_snapshot: Dict[str, object],
    outlier_mask_key: str,
    support_max_surface_distance: float,
    support_min_visible_views: int,
    support_min_opacity: float,
    min_patch_gaussians: int,
    min_patch_coverage: float,
    min_patch_std: float,
    min_action_score: float,
    max_candidate_patches: int,
):
    nearest_face_id = tensor_to_numpy(candidate_payload["nearest_face_id"]).reshape(-1).astype(np.int64, copy=False)
    total_gaussians = int(nearest_face_id.shape[0])
    outlier_mask = load_mask_from_payload(candidate_payload, outlier_mask_key, total=total_gaussians)
    support_mask = (~outlier_mask) & (nearest_face_id >= 0)
    if "surface_distance" in candidate_payload and float(support_max_surface_distance) > 0.0:
        support_mask &= tensor_to_numpy(candidate_payload["surface_distance"]).reshape(-1) <= float(support_max_surface_distance)
    if "visible_view_count" in candidate_payload and int(support_min_visible_views) > 0:
        support_mask &= tensor_to_numpy(candidate_payload["visible_view_count"]).reshape(-1) >= int(support_min_visible_views)
    if "opacity" in candidate_payload and float(support_min_opacity) > 0.0:
        support_mask &= tensor_to_numpy(candidate_payload["opacity"]).reshape(-1) >= float(support_min_opacity)

    xyz_world = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    reference_points = (
        tensor_to_numpy(candidate_payload["nearest_surface_point"]).astype(np.float32, copy=False)
        if "nearest_surface_point" in candidate_payload
        else xyz_world
    )
    patch_assignments = assign_gaussians_to_patches(
        nearest_face_id=nearest_face_id,
        reference_points=reference_points,
        patch_face_ids=patch_bank["face_ids"].astype(np.int64, copy=False),
        patch_centers=patch_bank["centers"].astype(np.float32, copy=False),
        active_mask=support_mask,
    )

    visible_ids = tensor_to_numpy(dropout_snapshot["visible_ids"]).reshape(-1).astype(np.int64, copy=False)
    bias = tensor_to_numpy(dropout_snapshot["dropout_bias"]).reshape(-1).astype(np.float32, copy=False)
    std = tensor_to_numpy(dropout_snapshot["dropout_std"]).reshape(-1).astype(np.float32, copy=False)
    coverage = tensor_to_numpy(dropout_snapshot["dropout_coverage"]).reshape(-1).astype(np.float32, copy=False)
    visible_patch_ids = patch_assignments[visible_ids]
    patch_metrics = aggregate_patch_metrics(
        patch_ids=visible_patch_ids,
        bias=bias,
        std=std,
        coverage=coverage,
    )

    patch_to_support_ids: Dict[int, np.ndarray] = {}
    valid_support_ids = np.flatnonzero(patch_assignments >= 0).astype(np.int64, copy=False)
    if valid_support_ids.size > 0:
        unique_patch_ids = np.unique(patch_assignments[valid_support_ids]).astype(np.int64, copy=False)
        for patch_id in unique_patch_ids.tolist():
            ids = valid_support_ids[patch_assignments[valid_support_ids] == int(patch_id)]
            patch_to_support_ids[int(patch_id)] = ids.astype(np.int64, copy=False)

    candidates: List[Dict[str, float]] = []
    for patch_id, metric in patch_metrics.items():
        support_ids = patch_to_support_ids.get(int(patch_id), np.empty((0,), dtype=np.int64))
        gaussian_count = int(metric["gaussian_count"])
        support_count = int(support_ids.size)
        action_score = float(metric["coverage"] * abs(metric["bias"]) / max(metric["std"], 1e-6))
        under_score = float(metric["coverage"] * metric["std"])
        if gaussian_count < int(min_patch_gaussians):
            continue
        if support_count < int(min_patch_gaussians):
            continue
        if float(metric["coverage"]) < float(min_patch_coverage):
            continue
        if float(metric["std"]) < float(min_patch_std):
            continue
        if action_score < float(min_action_score):
            continue
        candidates.append(
            {
                "patch_id": int(patch_id),
                "bias": float(metric["bias"]),
                "std": float(metric["std"]),
                "coverage": float(metric["coverage"]),
                "gaussian_count": gaussian_count,
                "support_count": support_count,
                "action_score": action_score,
                "under_score": under_score,
            }
        )

    candidates = sorted(candidates, key=lambda item: (-item["action_score"], -item["under_score"], -item["coverage"]))
    if int(max_candidate_patches) > 0:
        candidates = candidates[: int(max_candidate_patches)]

    nominated_ids: List[np.ndarray] = []
    for item in candidates:
        patch_id = int(item["patch_id"])
        support_ids = patch_to_support_ids.get(patch_id, np.empty((0,), dtype=np.int64))
        if support_ids.size > 0:
            nominated_ids.append(support_ids)

    if nominated_ids:
        union_ids = np.unique(np.concatenate(nominated_ids, axis=0)).astype(np.int64, copy=False)
    else:
        union_ids = np.empty((0,), dtype=np.int64)
    return candidates, union_ids


def main():
    parser = ArgumentParser(description="Highlight dropout-nominated gaussians for a diagnostic snapshot.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--patch_bank_path", type=str, required=True)
    parser.add_argument("--dropout_snapshot", type=str, required=True)
    parser.add_argument("--outlier_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--support_max_surface_distance", type=float, default=0.0)
    parser.add_argument("--support_min_visible_views", type=int, default=0)
    parser.add_argument("--support_min_opacity", type=float, default=0.0)
    parser.add_argument("--min_patch_gaussians", type=int, default=6)
    parser.add_argument("--min_patch_coverage", type=float, default=0.4)
    parser.add_argument("--min_patch_std", type=float, default=0.0)
    parser.add_argument("--min_action_score", type=float, default=0.0)
    parser.add_argument("--max_candidate_patches", type=int, default=8)
    parser.add_argument("--highlight_color", type=str, default="1.0,0.1,0.1")
    parser.add_argument("--overlay_alpha", type=float, default=0.75)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe_args = pipeline.extract(args)
    splat_args = ss.get_settings(args)

    dropout_snapshot = load_dropout_snapshot(args.dropout_snapshot)
    camera_name = str(dropout_snapshot["camera_name"])

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras()
    camera_index = index_cameras(train_cameras)
    camera = None
    for token in normalize_image_name(camera_name):
        if token in camera_index:
            camera = camera_index[token]
            break
    if camera is None:
        raise KeyError(f"Could not find camera '{camera_name}' in train cameras.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    base_render_pkg = render(camera, gaussians, pipe_args, background, splat_args=splat_args)
    base_image = base_render_pkg["render"][0:3]

    candidate_payload = torch.load(args.candidate_payload, map_location="cpu")
    patch_bank_npz = np.load(args.patch_bank_path, allow_pickle=False)
    patch_bank = {key: patch_bank_npz[key] for key in patch_bank_npz.files}

    candidates, nominated_ids = collect_candidate_support_ids(
        gaussians=gaussians,
        candidate_payload=candidate_payload,
        patch_bank=patch_bank,
        dropout_snapshot=dropout_snapshot,
        outlier_mask_key=str(args.outlier_mask_key),
        support_max_surface_distance=float(args.support_max_surface_distance),
        support_min_visible_views=int(args.support_min_visible_views),
        support_min_opacity=float(args.support_min_opacity),
        min_patch_gaussians=int(args.min_patch_gaussians),
        min_patch_coverage=float(args.min_patch_coverage),
        min_patch_std=float(args.min_patch_std),
        min_action_score=float(args.min_action_score),
        max_candidate_patches=int(args.max_candidate_patches),
    )

    highlight_color = parse_rgb(args.highlight_color)
    highlight_only_color = build_highlight_only_colors(
        total=int(gaussians.get_xyz.shape[0]),
        highlight_ids=nominated_ids,
        highlight_color=highlight_color,
    )
    highlight_only = render(
        camera,
        gaussians,
        pipe_args,
        background,
        splat_args=splat_args,
        override_color=highlight_only_color,
    )["render"][0:3]
    mask = highlight_only.max(dim=0, keepdim=True).values.clamp(0.0, 1.0)
    highlight_color_tensor = torch.tensor(highlight_color, dtype=torch.float32, device="cuda").view(3, 1, 1)
    overlay = torch.clamp(base_image * (1.0 - float(args.overlay_alpha) * mask) + highlight_color_tensor * (float(args.overlay_alpha) * mask), 0.0, 1.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(base_image, output_dir / "base_render.png")
    torchvision.utils.save_image(highlight_only, output_dir / "highlight_only.png")
    torchvision.utils.save_image(overlay, output_dir / "overlay.png")
    summary = {
        "mode": "render_highlight_dropout_nominated_gaussians_v0",
        "camera_name": camera_name,
        "candidate_count": int(len(candidates)),
        "nominated_gaussian_count": int(nominated_ids.size),
        "candidates": candidates,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dropout nomination highlight bundle to: {output_dir}")


if __name__ == "__main__":
    main()
