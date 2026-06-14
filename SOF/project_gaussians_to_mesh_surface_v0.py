import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch import nn

from arguments import ModelParams
from scene.gaussian_model import GaussianModel
from train_mesh_bounded_from_gs_support_v0 import (
    load_mask_from_payload,
    points_to_barycentric,
    stats_from_array,
    tensor_to_numpy,
)
from utils.general_utils import safe_state
from utils.system_utils import mkdir_p


def copy_render_config(src_model_path: str, dst_model_path: Path):
    src = Path(src_model_path)
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ["cfg_args", "config.json", "cameras.json"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst_model_path / name)


def load_base_model(dataset, model_path: str, iteration: int) -> GaussianModel:
    ply_path = Path(model_path) / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Base GS PLY not found: {ply_path}")
    model = GaussianModel(dataset.sh_degree)
    model.load_ply(str(ply_path))
    tags_path = ply_path.parent / "gaussian_tags.pt"
    if tags_path.exists():
        model.load_tracking_metadata(str(tags_path))
    return model


def project_points_to_payload_surface(source: GaussianModel, payload: dict, mask: np.ndarray, mesh_path: str, center_mode: str) -> torch.Tensor:
    projected_xyz = source.get_xyz.detach().clone()
    support_ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if support_ids.size == 0:
        return projected_xyz

    if center_mode == "nearest_surface":
        if "nearest_surface_point" not in payload:
            raise KeyError("candidate payload is missing 'nearest_surface_point'; use --center_mode barycentric_project")
        projected = tensor_to_numpy(payload["nearest_surface_point"])[support_ids].astype(np.float32, copy=False)
    else:
        mesh_obj = trimesh.load(mesh_path, process=False)
        vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
        faces = np.asarray(mesh_obj.faces, dtype=np.int64)
        nearest_face_id = tensor_to_numpy(payload["nearest_face_id"]).astype(np.int64, copy=False).reshape(-1)
        face_ids = nearest_face_id[support_ids]
        points = source.get_xyz.detach().cpu().numpy()[support_ids].astype(np.float32, copy=False)
        bary = points_to_barycentric(points, vertices[faces[face_ids]])
        projected = np.sum(vertices[faces[face_ids]] * bary[:, :, None], axis=1).astype(np.float32, copy=False)

    projected_xyz[support_ids] = torch.from_numpy(projected).to(device=projected_xyz.device, dtype=projected_xyz.dtype)
    return projected_xyz


def payload_array(payload: dict, key: str, dtype=None):
    if key not in payload:
        return None
    arr = tensor_to_numpy(payload[key])
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr.reshape(-1)


def main():
    parser = ArgumentParser(description="Project non-outlier GS centers onto the mesh while preserving original GS scale/rotation/color.")
    model = ModelParams(parser)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--base_iteration", type=int, default=30000)
    parser.add_argument("--candidate_payload", type=str, required=True)
    parser.add_argument("--outlier_mask_key", type=str, default="candidate_mask")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--output_iteration", type=int, default=0)
    parser.add_argument("--center_mode", choices=["nearest_surface", "barycentric_project"], default="nearest_surface")
    parser.add_argument(
        "--max_project_distance",
        type=float,
        default=0.0,
        help="Only project non-outlier GS whose payload surface_distance is <= this value. 0 disables the clamp.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    safe_state(args.quiet)

    dataset = model.extract(args)
    source = load_base_model(dataset, args.base_model_path, args.base_iteration)
    payload = torch.load(args.candidate_payload, map_location="cpu")
    total = int(source.get_xyz.shape[0])
    outlier_mask = load_mask_from_payload(payload, args.outlier_mask_key, total=total)
    nearest_face_id = tensor_to_numpy(payload["nearest_face_id"]).astype(np.int64, copy=False).reshape(-1)
    support_mask = (~outlier_mask) & (nearest_face_id >= 0)
    surface_distance = payload_array(payload, "surface_distance", np.float32)
    unclamped_support_mask = support_mask.copy()
    if float(args.max_project_distance) > 0.0:
        if surface_distance is None:
            raise KeyError("candidate payload is missing 'surface_distance', required by --max_project_distance")
        support_mask &= surface_distance <= float(args.max_project_distance)

    projected_xyz = project_points_to_payload_surface(source, payload, support_mask, args.mesh_path, args.center_mode)
    source_xyz_np = source.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    projected_xyz_np = projected_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    displacement = np.linalg.norm(projected_xyz_np - source_xyz_np, axis=1)
    support_ids = np.flatnonzero(support_mask).astype(np.int64, copy=False)
    support_face_ids = nearest_face_id[support_ids] if support_ids.size else np.zeros((0,), dtype=np.int64)

    output = GaussianModel(source.max_sh_degree, use_SBs=source.use_SBs)
    output.active_sh_degree = source.active_sh_degree
    output.spatial_lr_scale = source.spatial_lr_scale
    output._xyz = nn.Parameter(projected_xyz.detach().clone(), requires_grad=False)
    output._features_dc = nn.Parameter(source._features_dc.detach().clone(), requires_grad=False)
    output._features_rest = nn.Parameter(source._features_rest.detach().clone(), requires_grad=False)
    output._opacity = nn.Parameter(source._opacity.detach().clone(), requires_grad=False)
    output._scaling = nn.Parameter(source._scaling.detach().clone(), requires_grad=False)
    output._rotation = nn.Parameter(source._rotation.detach().clone(), requires_grad=False)
    output.filter_3D = source.filter_3D.detach().clone()
    output.max_radii2D = torch.zeros((total,), dtype=torch.float32, device="cuda")
    output.restore_tracking_state(source.capture_tracking_state())

    out_model = Path(args.output_model_path)
    copy_render_config(args.base_model_path, out_model)
    point_dir = out_model / "point_cloud" / f"iteration_{int(args.output_iteration)}"
    mkdir_p(str(point_dir))
    output.save_ply(str(point_dir / "point_cloud.ply"))
    output.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))
    summary = {
        "mode": "project_gaussians_to_mesh_surface_v0",
        "base_model_path": args.base_model_path,
        "base_iteration": int(args.base_iteration),
        "candidate_payload": args.candidate_payload,
        "outlier_mask_key": args.outlier_mask_key,
        "mesh_path": args.mesh_path,
        "center_mode": args.center_mode,
        "max_project_distance": float(args.max_project_distance),
        "total_gaussians": total,
        "unclamped_non_outlier_surface_supported_gaussians": int(np.sum(unclamped_support_mask)),
        "projected_support_gaussians": int(np.sum(support_mask)),
        "projected_unique_faces": int(np.unique(support_face_ids).shape[0]) if support_face_ids.size else 0,
        "preserved_outlier_gaussians": int(np.sum(outlier_mask)),
        "preserved_nonprojected_gaussians": int(total - np.sum(support_mask)),
        "projection_displacement_projected": stats_from_array(displacement[support_mask]),
        "surface_distance_projected": stats_from_array(surface_distance[support_mask]) if surface_distance is not None else None,
        "surface_distance_unclamped_support": stats_from_array(surface_distance[unclamped_support_mask])
        if surface_distance is not None
        else None,
        "output_ply": str(point_dir / "point_cloud.ply"),
    }
    with open(point_dir / "num_gaussians.json", "w", encoding="utf-8") as f:
        json.dump({"num_gaussians": total, **summary}, f, indent=2)
    with open(out_model / "project_gaussians_to_mesh_surface_v0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
