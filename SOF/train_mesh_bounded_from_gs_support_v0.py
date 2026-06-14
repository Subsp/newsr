import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Dict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SplattingSettings
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianSourceTag
from scene.sugar_like_meshgs_model import SugarLikeMeshGaussianModel
from train_meshgs_prior_v0 import load_rgb_cached, masked_l1, render_alpha_mask
from train_sugar_like_meshgs_prior_v0 import save_sugar_like_meshgs
from utils.general_utils import safe_state
from utils.prior_injection import index_image_dir
from utils.sh_utils import SH2RGB
from utils.system_utils import mkdir_p


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def points_to_barycentric(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    safe = np.abs(denom) > 1e-12
    v = np.zeros_like(denom, dtype=np.float32)
    w = np.zeros_like(denom, dtype=np.float32)
    v[safe] = ((d11[safe] * d20[safe] - d01[safe] * d21[safe]) / denom[safe]).astype(np.float32)
    w[safe] = ((d00[safe] * d21[safe] - d01[safe] * d20[safe]) / denom[safe]).astype(np.float32)
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1).astype(np.float32, copy=False)
    bary = np.clip(bary, 0.0, 1.0)
    bary_sum = np.sum(bary, axis=1, keepdims=True)
    degenerate = bary_sum[:, 0] <= 1e-8
    bary = bary / np.clip(bary_sum, 1e-8, None)
    if np.any(degenerate):
        bary[degenerate] = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    return bary


def load_mask_from_payload(payload: dict, key: str, total: int) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Payload does not contain '{key}'. Available keys: {sorted(payload.keys())}")
    value = np.asarray(tensor_to_numpy(payload[key]))
    if value.dtype == np.bool_:
        value = value.reshape(-1)
        if value.shape[0] != total:
            raise ValueError(f"Boolean mask '{key}' has {value.shape[0]} entries, expected {total}.")
        return value.astype(bool, copy=False)
    ids = value.reshape(-1).astype(np.int64, copy=False)
    if ids.size and np.any((ids < 0) | (ids >= total)):
        raise ValueError(f"Index mask '{key}' contains ids outside [0, {total}).")
    mask = np.zeros((total,), dtype=bool)
    mask[ids] = True
    return mask


def select_supported_mesh_faces(candidate_payload_path: str, args):
    payload = torch.load(candidate_payload_path, map_location="cpu")
    nearest_face_id = tensor_to_numpy(payload["nearest_face_id"]).astype(np.int64, copy=False).reshape(-1)
    total_gaussians = int(nearest_face_id.shape[0])
    valid_face = nearest_face_id >= 0
    outlier_mask = load_mask_from_payload(payload, args.outlier_mask_key, total=total_gaussians)
    support_mask = (~outlier_mask) & valid_face

    surface_distance = (
        tensor_to_numpy(payload["surface_distance"]).astype(np.float32, copy=False).reshape(-1)
        if "surface_distance" in payload
        else None
    )
    visible_view_count = (
        tensor_to_numpy(payload["visible_view_count"]).astype(np.int32, copy=False).reshape(-1)
        if "visible_view_count" in payload
        else None
    )
    min_visible_depth = (
        tensor_to_numpy(payload["min_visible_depth"]).astype(np.float32, copy=False).reshape(-1)
        if "min_visible_depth" in payload
        else None
    )
    opacity = (
        tensor_to_numpy(payload["opacity"]).astype(np.float32, copy=False).reshape(-1)
        if "opacity" in payload
        else None
    )

    if surface_distance is not None and float(args.support_max_surface_distance) > 0.0:
        support_mask &= surface_distance <= float(args.support_max_surface_distance)
    if visible_view_count is not None and int(args.support_min_visible_views) > 0:
        support_mask &= visible_view_count >= int(args.support_min_visible_views)
    if min_visible_depth is not None and float(args.support_max_nearest_visible_depth) > 0.0:
        support_mask &= min_visible_depth <= float(args.support_max_nearest_visible_depth)
    if opacity is not None and float(args.support_min_opacity) > 0.0:
        support_mask &= opacity >= float(args.support_min_opacity)

    support_ids = np.flatnonzero(support_mask).astype(np.int64, copy=False)
    if support_ids.size == 0:
        raise RuntimeError("No non-outlier/surface-supported GS remain after filtering.")

    raw_face_ids = nearest_face_id[support_ids]
    face_ids, support_counts = np.unique(raw_face_ids, return_counts=True)
    keep = support_counts >= max(int(args.min_support_per_face), 1)
    face_ids = face_ids[keep].astype(np.int64, copy=False)
    support_counts = support_counts[keep].astype(np.int32, copy=False)
    if face_ids.size == 0:
        raise RuntimeError("No supported mesh faces remain after min_support_per_face filtering.")

    selection_mode = "all_supported_faces"
    if int(args.max_faces) > 0 and face_ids.size > int(args.max_faces):
        max_faces = int(args.max_faces)
        if args.face_sample_mode == "random":
            rng = np.random.default_rng(int(args.seed))
            chosen = np.sort(rng.choice(face_ids.shape[0], size=max_faces, replace=False))
            selection_mode = f"random_{max_faces}"
        else:
            chosen = np.argsort(-support_counts, kind="stable")[:max_faces]
            chosen = np.sort(chosen)
            selection_mode = f"top_count_{max_faces}"
        face_ids = face_ids[chosen]
        support_counts = support_counts[chosen]

    if int(args.face_stride) > 1:
        face_ids = face_ids[:: int(args.face_stride)]
        support_counts = support_counts[:: int(args.face_stride)]
        selection_mode = f"{selection_mode}_stride_{int(args.face_stride)}"

    return {
        "payload": payload,
        "nearest_face_id": nearest_face_id,
        "outlier_mask": outlier_mask,
        "support_mask": support_mask,
        "support_ids": support_ids,
        "face_ids": face_ids,
        "support_counts": support_counts,
        "surface_distance": surface_distance,
        "visible_view_count": visible_view_count,
        "min_visible_depth": min_visible_depth,
        "opacity": opacity,
        "total_gaussians": total_gaussians,
        "selection_mode": selection_mode,
    }


def save_face_support_payload(output_dir: Path, selection: dict, args):
    face_ids = selection["face_ids"]
    support_counts = selection["support_counts"]
    payload = {
        "mode": "mesh_bounded_from_gs_support_v0_faces",
        "supported_face_ids": torch.from_numpy(face_ids.astype(np.int64, copy=False)),
        "supported_face_gs_count": torch.from_numpy(support_counts.astype(np.int32, copy=False)),
        "support_ids": torch.from_numpy(selection["support_ids"].astype(np.int64, copy=False)),
        "support_mask": torch.from_numpy(selection["support_mask"].copy()),
        "outlier_mask": torch.from_numpy(selection["outlier_mask"].copy()),
        # Compatibility with prepare_prior_fusion_v0, if we want to reuse it later.
        "roi_face_ids": torch.from_numpy(face_ids.astype(np.int64, copy=False)),
    }
    path = output_dir / "mesh_bounded_supported_faces_v0.pt"
    torch.save(payload, str(path))
    return path


def subset_gaussian_model(source: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    mask = mask.to(device=source.get_xyz.device, dtype=torch.bool)
    out = GaussianModel(source.max_sh_degree, use_SBs=source.use_SBs)
    out.active_sh_degree = source.active_sh_degree
    out.spatial_lr_scale = source.spatial_lr_scale
    out._xyz = nn.Parameter(source._xyz.detach()[mask].clone(), requires_grad=False)
    out._features_dc = nn.Parameter(source._features_dc.detach()[mask].clone(), requires_grad=False)
    out._features_rest = nn.Parameter(source._features_rest.detach()[mask].clone(), requires_grad=False)
    out._opacity = nn.Parameter(source._opacity.detach()[mask].clone(), requires_grad=False)
    out._scaling = nn.Parameter(source._scaling.detach()[mask].clone(), requires_grad=False)
    out._rotation = nn.Parameter(source._rotation.detach()[mask].clone(), requires_grad=False)
    if hasattr(source, "filter_3D") and source.filter_3D is not None and source.filter_3D.shape[0] == mask.shape[0]:
        out.filter_3D = source.filter_3D.detach()[mask].clone()
    else:
        out.filter_3D = torch.zeros((out._xyz.shape[0], 1), dtype=torch.float32, device="cuda")
    out.max_radii2D = torch.zeros((out._xyz.shape[0],), dtype=torch.float32, device="cuda")
    out.init_tracking_state(out._xyz.shape[0], source_tag=int(GaussianSourceTag.ORIGINAL))
    return out


def copy_render_config_from_base(base_model_path: str, output_model_path: Path):
    base = Path(base_model_path)
    output_model_path.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "config.json"]:
        src = base / name
        if src.exists():
            shutil.copy2(src, output_model_path / name)


def load_base_gaussian_model(dataset, args) -> GaussianModel:
    if not args.base_model_path:
        raise ValueError("--base_model_path is required for gaussian support carrier mode and outlier export.")
    ply_path = Path(args.base_model_path) / "point_cloud" / f"iteration_{int(args.base_iteration)}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Base GS PLY not found: {ply_path}")
    source = GaussianModel(dataset.sh_degree)
    source.load_ply(str(ply_path))
    tags_path = ply_path.parent / "gaussian_tags.pt"
    if tags_path.exists():
        source.load_tracking_metadata(str(tags_path))
    return source


def export_outlier_model(dataset, args, outlier_mask_np: np.ndarray):
    if not args.export_outlier_model_path:
        return None
    source = load_base_gaussian_model(dataset, args)
    mask = torch.from_numpy(outlier_mask_np.astype(bool, copy=False)).cuda()
    if mask.shape[0] != source.get_xyz.shape[0]:
        raise ValueError(f"Outlier mask length {mask.shape[0]} does not match base GS count {source.get_xyz.shape[0]}.")

    outlier = subset_gaussian_model(source, mask)
    output_model_path = Path(args.export_outlier_model_path)
    point_dir = output_model_path / "point_cloud" / "iteration_0"
    mkdir_p(str(point_dir))
    outlier.save_ply(str(point_dir / "point_cloud.ply"))
    outlier.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": int(outlier.get_xyz.shape[0])}, f, indent=2)
    copy_render_config_from_base(args.base_model_path, output_model_path)
    return str(point_dir / "point_cloud.ply")


def initialize_meshgs_from_support_gaussians(meshgs: SugarLikeMeshGaussianModel, selection: dict, source: GaussianModel, args):
    import trimesh

    support_ids = selection["support_ids"].astype(np.int64, copy=False)
    if support_ids.size == 0:
        raise RuntimeError("No support GS available for gaussian carrier initialization.")
    mesh_obj = trimesh.load(args.mesh_path, process=False)
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    face_ids = selection["nearest_face_id"][support_ids].astype(np.int64, copy=False)

    if args.support_center_mode == "nearest_surface":
        if "nearest_surface_point" not in selection["payload"]:
            raise KeyError("candidate payload is missing 'nearest_surface_point'; use --support_center_mode source_projected")
        points = tensor_to_numpy(selection["payload"]["nearest_surface_point"])[support_ids].astype(np.float32, copy=False)
    else:
        points = source.get_xyz.detach().cpu().numpy()[support_ids].astype(np.float32, copy=False)
    bary = points_to_barycentric(points, vertices[faces[face_ids]])

    if source.use_SBs:
        colors = SH2RGB(source._features_dc.detach()[support_ids]).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    else:
        colors = SH2RGB(source._features_dc.detach()[support_ids, 0, :]).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    source_scales = source.get_scaling.detach().cpu().numpy()[support_ids].astype(np.float32, copy=False)
    sorted_scales = np.sort(source_scales, axis=1)
    if args.source_plane_scale_mode == "sorted":
        scale_u = sorted_scales[:, 1] * float(args.plane_scale_multiplier)
        scale_v = sorted_scales[:, 2] * float(args.plane_scale_multiplier)
    elif args.source_plane_scale_mode == "isotropic_min":
        plane_scale = sorted_scales[:, 0] * float(args.plane_scale_multiplier)
        scale_u = plane_scale
        scale_v = plane_scale
    else:
        plane_scale = sorted_scales[:, 1] * float(args.plane_scale_multiplier)
        scale_u = plane_scale
        scale_v = plane_scale
    if float(args.max_plane_scale) > 0.0:
        scale_u = np.minimum(scale_u, float(args.max_plane_scale))
        scale_v = np.minimum(scale_v, float(args.max_plane_scale))
    scale_n = sorted_scales[:, 0] * float(args.source_normal_scale_multiplier)
    opacity = source.get_opacity.detach().cpu().numpy()[support_ids, 0].astype(np.float32, copy=False)
    opacity = np.clip(opacity * float(args.source_opacity_multiplier), 1e-5, 0.995)

    meshgs.initialize_from_arrays(
        vertices=vertices,
        faces=faces,
        face_ids=face_ids,
        bary_coords=bary,
        colors=colors,
        scale_u=scale_u,
        scale_v=scale_v,
        scale_n=scale_n,
        opacity=opacity,
        learn_surface_vertices=bool(args.learn_surface_vertices),
        learn_plane_scales=bool(args.learn_plane_scales),
        learn_inplane_rotation=bool(args.learn_inplane_rotation),
        build_normal_pairs=not bool(args.disable_normal_consistency_pairs),
        max_normal_pairs=int(args.max_normal_consistency_pairs),
    )
    return {
        "selected_face_count": int(np.unique(face_ids).shape[0]),
        "selected_gaussian_count": int(support_ids.shape[0]),
        "normal_consistency_pairs": int(meshgs._normal_consistency_pairs.shape[0]),
        "support_carrier_mode": "gaussian",
        "support_center_mode": args.support_center_mode,
        "source_plane_scale_mode": args.source_plane_scale_mode,
        "plane_scale_multiplier": float(args.plane_scale_multiplier),
        "max_plane_scale": float(args.max_plane_scale),
        "source_opacity_multiplier": float(args.source_opacity_multiplier),
        "source_normal_scale_multiplier": float(args.source_normal_scale_multiplier),
        "scale_u": stats_from_array(scale_u),
        "scale_v": stats_from_array(scale_v),
        "scale_n": stats_from_array(scale_n),
        "opacity": stats_from_array(opacity),
    }


def train_mesh_bounded_from_support(dataset, pipe, splat_args, args):
    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, dummy_gaussians, shuffle=True, skip_test=True, skip_train=False)
    train_cameras = scene.getTrainCameras().copy()

    selection = select_supported_mesh_faces(args.candidate_payload, args)
    face_payload_path = save_face_support_payload(output_dir, selection, args)
    print(
        "[mesh-bounded-support] selected mesh faces: "
        f"{selection['face_ids'].shape[0]} from {int(np.sum(selection['support_mask']))} non-outlier GS"
    )

    meshgs = SugarLikeMeshGaussianModel(dataset.sh_degree, use_SBs=False)
    source_gaussians = None
    if args.support_carrier_mode == "gaussian":
        source_gaussians = load_base_gaussian_model(dataset, args)
        init_summary = initialize_meshgs_from_support_gaussians(meshgs, selection, source_gaussians, args)
    else:
        init_summary = meshgs.initialize_from_mesh_faces(
            mesh_path=args.mesh_path,
            face_ids=selection["face_ids"],
            carriers_per_face=int(args.carriers_per_face),
            init_rgb=tuple(float(v) for v in args.init_rgb),
            thickness_scale=float(args.thickness_scale),
            plane_scale_multiplier=float(args.plane_scale_multiplier),
            max_plane_scale=float(args.max_plane_scale),
            init_opacity=float(args.init_opacity),
            learn_surface_vertices=bool(args.learn_surface_vertices),
            learn_plane_scales=bool(args.learn_plane_scales),
            learn_inplane_rotation=bool(args.learn_inplane_rotation),
            build_normal_pairs=not bool(args.disable_normal_consistency_pairs),
            max_normal_pairs=int(args.max_normal_consistency_pairs),
        )
    init_summary.update(
        {
            "mode": "mesh_bounded_from_gs_support_v0",
            "mesh_path": args.mesh_path,
            "candidate_payload": args.candidate_payload,
            "face_payload_path": str(face_payload_path),
            "outlier_mask_key": args.outlier_mask_key,
            "total_gaussians": int(selection["total_gaussians"]),
            "outlier_gaussians": int(np.sum(selection["outlier_mask"])),
            "support_gaussians": int(np.sum(selection["support_mask"])),
            "selected_face_count": int(selection["face_ids"].shape[0]),
            "face_selection_mode": selection["selection_mode"],
            "support_carrier_mode": args.support_carrier_mode,
            "selected_meshgs_count": int(meshgs.get_xyz.shape[0]),
            "support_count_per_face": stats_from_array(selection["support_counts"].astype(np.float32)),
        }
    )
    print(
        "[mesh-bounded-support] initialized "
        f"{init_summary['selected_meshgs_count']} mesh-bound GS "
        f"on {init_summary['selected_face_count']} mesh faces"
    )

    outlier_ply = export_outlier_model(dataset, args, selection["outlier_mask"])
    if outlier_ply:
        init_summary["outlier_model_ply"] = outlier_ply
        print(f"[mesh-bounded-support] exported outlier-only model: {outlier_ply}")

    optimizer = meshgs.build_optimizer(
        feature_lr=float(args.meshgs_feature_lr),
        opacity_lr=float(args.meshgs_opacity_lr),
        surface_vertex_lr=float(args.surface_vertex_lr),
        plane_scale_lr=float(args.plane_scale_lr),
        inplane_rotation_lr=float(args.inplane_rotation_lr),
    )

    prior_index = index_image_dir(args.prior_dir)
    prior_cache = {}
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    progress = tqdm(range(1, int(args.iterations) + 1), desc="Training mesh-boundedGS from support")
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
                    "gs": str(init_summary["selected_meshgs_count"]),
                }
            )
        if iteration in args.save_iterations or iteration == int(args.iterations):
            save_sugar_like_meshgs(meshgs, output_dir, iteration, args, init_summary)

    save_sugar_like_meshgs(meshgs, output_dir, int(args.iterations), args, init_summary)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train mesh-boundedGS from mesh faces supported by non-outlier SOFGS.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--outlier_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--export_outlier_model_path", type=str, default=None)
    parser.add_argument("--max_faces", type=int, default=0)
    parser.add_argument("--face_stride", type=int, default=1)
    parser.add_argument("--face_sample_mode", choices=["top_count", "random"], default="top_count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support_carrier_mode", choices=["gaussian", "face"], default="gaussian")
    parser.add_argument("--support_center_mode", choices=["nearest_surface", "source_projected"], default="nearest_surface")
    parser.add_argument("--min_support_per_face", type=int, default=1)
    parser.add_argument("--support_max_surface_distance", type=float, default=0.0)
    parser.add_argument("--support_min_visible_views", type=int, default=0)
    parser.add_argument("--support_max_nearest_visible_depth", type=float, default=0.0)
    parser.add_argument("--support_min_opacity", type=float, default=0.0)
    parser.add_argument("--carriers_per_face", type=int, choices=[1, 3, 4, 6], default=1)
    parser.add_argument("--init_rgb", nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--init_opacity", type=float, default=0.2)
    parser.add_argument("--thickness_scale", type=float, default=0.05)
    parser.add_argument("--plane_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--max_plane_scale", type=float, default=0.0)
    parser.add_argument("--source_plane_scale_mode", choices=["isotropic_median", "isotropic_min", "sorted"], default="isotropic_median")
    parser.add_argument("--source_opacity_multiplier", type=float, default=1.0)
    parser.add_argument("--source_normal_scale_multiplier", type=float, default=1.0)
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
    train_mesh_bounded_from_support(model.extract(args), pipeline.extract(args), ss.get_settings(args), args)
