from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_mesh_bounded_gaussians_v0 import barycentric_for_faces, local_mesh_edge_lengths
from gaussian_renderer import render_simple
from select_mesh_outside_gaussians_v0 import load_triangle_mesh, query_mesh_surface
from train_mip_to_sof_surface_v0 import load_cameras_for_split, load_model_ply, resolve_iteration, select_uniform
from utils.general_utils import build_rotation
from utils.prior_fusion import build_mesh_depth_edge_mask, build_mesh_sample_visibility, project_points_camera


STATE_RESIDUAL_KEEP = 0
STATE_SURFACE_USABLE = 1
STATE_OFF_SURFACE_SUPPRESS = 2

STATE_NAMES = {
    STATE_RESIDUAL_KEEP: "residual_keep",
    STATE_SURFACE_USABLE: "surface_usable",
    STATE_OFF_SURFACE_SUPPRESS: "off_surface_suppress",
}


def stats_from_array(values: np.ndarray) -> Dict[str, float | int]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        return {"count": int(values.size), "finite_count": int(finite.sum())}
    return {
        "count": int(values.size),
        "finite_count": int(finite.sum()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def normalize_np(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), eps, None)


def scalar_to_rgb(value: np.ndarray, invert: bool = False) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x)
    if np.any(finite):
        lo = float(np.percentile(x[finite], 1.0))
        hi = float(np.percentile(x[finite], 99.0))
        x = (x - lo) / max(hi - lo, 1e-6)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    x = np.clip(1.0 - x if invert else x, 0.0, 1.0)
    r = np.clip(1.8 * x, 0.0, 1.0)
    g = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32, copy=False)


def scalar_unit_to_rgb(value: np.ndarray, invert: bool = False) -> np.ndarray:
    x = np.nan_to_num(np.asarray(value, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(1.0 - x if invert else x, 0.0, 1.0)
    r = np.clip(1.8 * x, 0.0, 1.0)
    g = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    b = np.clip(1.25 - 1.75 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32, copy=False)


def state_to_rgb(state: np.ndarray) -> np.ndarray:
    colors = np.asarray(
        [
            [0.35, 0.35, 0.38],  # residual_keep
            [0.08, 0.85, 0.46],  # surface_usable
            [1.00, 0.18, 0.10],  # off_surface_suppress
        ],
        dtype=np.float32,
    )
    clipped = np.clip(np.asarray(state, dtype=np.int64), 0, colors.shape[0] - 1)
    return colors[clipped]


def nanmedian_with_fill(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    finite_any = np.any(np.isfinite(values), axis=0)
    out = np.full((values.shape[1],), float(fill), dtype=np.float32)
    if np.any(finite_any):
        out[finite_any] = np.nanmedian(values[:, finite_any], axis=0).astype(np.float32, copy=False)
    return out


def sample_offsets(sample_scale: float) -> np.ndarray:
    scale = float(sample_scale)
    offsets = [
        [0.0, 0.0, 0.0],
        [scale, 0.0, 0.0],
        [-scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, -scale, 0.0],
        [0.0, 0.0, scale],
        [0.0, 0.0, -scale],
    ]
    return np.asarray(offsets, dtype=np.float32)


def build_gaussian_support_samples(
    xyz: np.ndarray,
    rotations: np.ndarray,
    scaling: np.ndarray,
    *,
    sample_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = sample_offsets(sample_scale)
    local = offsets[None, :, :] * scaling[:, None, :]
    world = xyz[:, None, :] + np.einsum("nij,nsj->nsi", rotations, local).astype(np.float32, copy=False)
    return world.reshape(-1, 3).astype(np.float32, copy=False), offsets


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=1)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        ids = np.arange(points.shape[0], dtype=np.int64)[:0]
    if int(max_points) > 0 and ids.size > int(max_points):
        pick = np.linspace(0, ids.size - 1, num=int(max_points), dtype=np.int64)
        ids = ids[pick]
    rgb = np.clip(colors[ids] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points[ids], colors=rgb).export(str(path))


def save_depth_preview(path: Path, depth: np.ndarray) -> None:
    finite = np.isfinite(depth)
    if not np.any(finite):
        image = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(depth[finite], 2.0))
        hi = float(np.percentile(depth[finite], 98.0))
        norm = np.zeros(depth.shape, dtype=np.float32)
        norm[finite] = np.clip((depth[finite] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        image = np.clip((1.0 - norm) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(str(path))


def save_mask_preview(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(str(path))


def to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw[:3].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def save_rgb(path: Path, image_chw: torch.Tensor) -> np.ndarray:
    image = to_uint8_rgb(image_chw)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(str(path))
    return image


def resize_rgb_for_overview(image: np.ndarray, max_width: int) -> np.ndarray:
    max_width = int(max_width)
    if max_width <= 0 or int(image.shape[1]) <= max_width:
        return image
    scale = float(max_width) / float(image.shape[1])
    target = (max_width, max(1, int(round(float(image.shape[0]) * scale))))
    resampling = getattr(Image, "Resampling", Image)
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize(target, resampling.BILINEAR), dtype=np.uint8)


def make_labeled_grid(tiles: Sequence[Tuple[str, np.ndarray]], columns: int, pad: int = 8, label_height: int = 24) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles provided for grid generation.")
    columns = max(1, int(columns))
    sample = tiles[0][1]
    tile_h, tile_w = int(sample.shape[0]), int(sample.shape[1])
    rows = int(math.ceil(len(tiles) / float(columns)))
    canvas_w = columns * tile_w + (columns + 1) * pad
    canvas_h = rows * (tile_h + label_height) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image_np) in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x0 = pad + col * (tile_w + pad)
        y0 = pad + row * (tile_h + label_height + pad)
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + label_height - 1], fill=(32, 32, 32))
        draw.text((x0 + 6, y0 + 4), str(label), fill=(235, 235, 235))
        canvas.paste(Image.fromarray(np.asarray(image_np, dtype=np.uint8), mode="RGB"), (x0, y0 + label_height))
    return canvas


def depth_edge_mask_np(
    depth: np.ndarray,
    *,
    abs_threshold: float,
    rel_threshold: float,
    kernel_size: int,
    dilate_kernel: int,
) -> np.ndarray:
    if not torch.cuda.is_available():
        finite = np.isfinite(depth)
        return np.zeros_like(finite, dtype=bool)
    depth_t = torch.from_numpy(depth.astype(np.float32, copy=False)).to(device="cuda")
    visible = torch.isfinite(depth_t)
    edge = build_mesh_depth_edge_mask(
        depth_t,
        visible,
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
        kernel_size=int(kernel_size),
        dilate_kernel=int(dilate_kernel),
    )
    return edge.detach().cpu().numpy().astype(bool)


def build_open3d_raycast_scene(mesh: trimesh.Trimesh):
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(faces)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def camera_center_np(cam) -> np.ndarray:
    center = cam.camera_center
    if torch.is_tensor(center):
        center = center.detach().cpu().numpy()
    return np.asarray(center, dtype=np.float32).reshape(3)


def render_mesh_depth_raycast(
    scene,
    cam,
    *,
    downsample: int,
    ray_chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    down = max(int(downsample), 1)
    width = int(math.ceil(float(cam.image_width) / float(down)))
    height = int(math.ceil(float(cam.image_height) / float(down)))
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (jj + 0.5) * float(down) - 0.5
    y = (ii + 0.5) * float(down) - 0.5
    dirs_cam = np.stack(
        [
            (x - float(cam.image_width) * 0.5) / float(cam.focal_x),
            (y - float(cam.image_height) * 0.5) / float(cam.focal_y),
            np.ones_like(x, dtype=np.float32),
        ],
        axis=-1,
    ).reshape(-1, 3)
    R = np.asarray(cam.R, dtype=np.float32)
    dirs_world = dirs_cam @ R.T
    origin = camera_center_np(cam)
    origins = np.repeat(origin[None, :], dirs_world.shape[0], axis=0)
    rays_np = np.concatenate([origins, dirs_world.astype(np.float32, copy=False)], axis=1).astype(np.float32, copy=False)

    depth = np.full((rays_np.shape[0],), np.inf, dtype=np.float32)
    face_ids = np.full((rays_np.shape[0],), -1, dtype=np.int64)
    chunk = max(int(ray_chunk), 1)
    invalid_primitive = np.iinfo(np.uint32).max
    for begin in range(0, rays_np.shape[0], chunk):
        end = min(begin + chunk, rays_np.shape[0])
        result = scene.cast_rays(o3d.core.Tensor(rays_np[begin:end], dtype=o3d.core.Dtype.Float32))
        t_hit = result["t_hit"].numpy().astype(np.float32, copy=False)
        primitive = result["primitive_ids"].numpy().astype(np.uint64, copy=False)
        hit = np.isfinite(t_hit) & (primitive != invalid_primitive)
        depth_chunk = depth[begin:end]
        face_chunk = face_ids[begin:end]
        depth_chunk[hit] = t_hit[hit]
        face_chunk[hit] = primitive[hit].astype(np.int64, copy=False)
    return depth.reshape(height, width), face_ids.reshape(height, width)


def render_mesh_depth_sample_zbuffer(mesh: trimesh.Trimesh, cam, args) -> Tuple[np.ndarray, np.ndarray]:
    visibility = build_mesh_sample_visibility(
        mesh,
        cam,
        depth_min=float(args.depth_min),
        front_facing_only=not bool(args.disable_front_facing_only),
        samples_per_face=int(args.mesh_sample_points_per_face),
        splat_kernel=int(args.mesh_sample_splat_kernel),
    )
    depth = visibility["depth"].detach().cpu().numpy().astype(np.float32, copy=False)
    return depth, np.full(depth.shape, -1, dtype=np.int64)


def sample_nearest_hw(value: np.ndarray, xy_lowres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = value.shape[:2]
    x = np.rint(xy_lowres[:, 0]).astype(np.int64)
    y = np.rint(xy_lowres[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full((xy_lowres.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid):
        out[valid] = np.asarray(value[y[valid], x[valid]], dtype=np.float32)
    return out, valid


def sample_nearest_int_hw(value: np.ndarray, xy_lowres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = value.shape[:2]
    x = np.rint(xy_lowres[:, 0]).astype(np.int64)
    y = np.rint(xy_lowres[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full((xy_lowres.shape[0],), -1, dtype=np.int64)
    if np.any(valid):
        out[valid] = np.asarray(value[y[valid], x[valid]], dtype=np.int64)
    return out, valid


def gaussian_tracking_array(model, name: str, dtype: np.dtype, default: int | bool) -> np.ndarray:
    total = int(model.get_xyz.shape[0])
    value = getattr(model, name, None)
    if torch.is_tensor(value) and int(value.shape[0]) == total:
        return value.detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    return np.full((total,), default, dtype=dtype)


def compute_action_weights(
    *,
    support_count: np.ndarray,
    surface_mass: np.ndarray,
    front_mass: np.ndarray,
    behind_mass: np.ndarray,
    crossing_score: np.ndarray,
    edge_ratio: np.ndarray,
    args,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    enough = support_count >= int(args.min_support_views)
    edge_uncertainty = np.clip(edge_ratio, 0.0, 1.0).astype(np.float32)
    confidence = np.clip(support_count.astype(np.float32) / max(float(args.max_views), 1.0), 0.0, 1.0)
    mesh_bind_weight = np.clip(surface_mass * (1.0 - 0.65 * edge_uncertainty) * enough.astype(np.float32), 0.0, 1.0)
    surface_extract_weight = np.clip(
        np.maximum(surface_mass, crossing_score * surface_mass)
        * (1.0 - 0.45 * edge_uncertainty)
        * enough.astype(np.float32),
        0.0,
        1.0,
    )
    off_surface_mass = np.maximum(front_mass, behind_mass)
    opacity_suppress_weight = np.clip(
        np.maximum(off_surface_mass - float(args.surface_mass_protect) * surface_mass, 0.0)
        * (1.0 - 0.35 * edge_uncertainty)
        * enough.astype(np.float32),
        0.0,
        1.0,
    )
    residual_keep_weight = np.clip(
        1.0
        - np.maximum(surface_extract_weight, opacity_suppress_weight)
        + 0.35 * edge_uncertainty
        + (1.0 - confidence) * 0.35,
        0.0,
        1.0,
    )

    surface_usable = enough & (surface_extract_weight >= float(args.surface_extract_threshold))
    off_surface = enough & ~surface_usable & (opacity_suppress_weight >= float(args.opacity_suppress_threshold))
    state = np.full((support_count.shape[0],), STATE_RESIDUAL_KEEP, dtype=np.int16)
    state[surface_usable] = STATE_SURFACE_USABLE
    state[off_surface] = STATE_OFF_SURFACE_SUPPRESS
    return state, {
        "mesh_bind_weight": mesh_bind_weight.astype(np.float32, copy=False),
        "surface_extract_weight": surface_extract_weight.astype(np.float32, copy=False),
        "opacity_suppress_weight": opacity_suppress_weight.astype(np.float32, copy=False),
        "residual_keep_weight": residual_keep_weight.astype(np.float32, copy=False),
        "edge_uncertainty": edge_uncertainty.astype(np.float32, copy=False),
        "surface_usable": surface_usable.astype(bool),
        "off_surface_suppress": off_surface.astype(bool),
        "residual_keep": (state == STATE_RESIDUAL_KEEP),
    }


@torch.no_grad()
def render_regulation_debug_previews(
    *,
    model,
    cameras: Sequence[object],
    output_dir: Path,
    state: np.ndarray,
    surface_mass: np.ndarray,
    crossing_score: np.ndarray,
    mesh_bind_weight: np.ndarray,
    surface_extract_weight: np.ndarray,
    opacity_suppress_weight: np.ndarray,
    residual_keep_weight: np.ndarray,
    edge_uncertainty: np.ndarray,
    white_background: bool,
    grid_columns: int,
    overview_max_width: int,
) -> List[Dict[str, object]]:
    if not cameras:
        return []
    device = model.get_xyz.device
    background = torch.ones((3,), dtype=torch.float32, device=device) if bool(white_background) else torch.zeros((3,), dtype=torch.float32, device=device)
    color_specs: List[Tuple[str, np.ndarray]] = [
        ("state", state_to_rgb(state)),
        ("surface_mass", scalar_unit_to_rgb(surface_mass)),
        ("surface_extract", scalar_unit_to_rgb(surface_extract_weight)),
        ("mesh_bind", scalar_unit_to_rgb(mesh_bind_weight)),
        ("crossing", scalar_unit_to_rgb(crossing_score)),
        ("opacity_suppress", scalar_unit_to_rgb(opacity_suppress_weight)),
        ("residual_keep", scalar_unit_to_rgb(residual_keep_weight)),
        ("edge_uncertainty", scalar_unit_to_rgb(edge_uncertainty)),
    ]
    override_specs = [
        (name, torch.from_numpy(np.clip(rgb, 0.0, 1.0)).to(device=device, dtype=torch.float32))
        for name, rgb in color_specs
    ]

    render_dir = output_dir / "render_debug_v0"
    render_dir.mkdir(parents=True, exist_ok=True)
    view_summaries: List[Dict[str, object]] = []
    overview_tiles: List[Tuple[str, np.ndarray]] = []

    for view_idx, camera in enumerate(cameras):
        view_name = Path(str(camera.image_name)).stem or str(camera.image_name)
        view_dir = render_dir / f"{view_idx:03d}_{view_name}"
        view_dir.mkdir(parents=True, exist_ok=True)

        base_pkg = render_simple(camera, model, background)
        tiles: List[Tuple[str, np.ndarray]] = [("base", save_rgb(view_dir / "base_render.png", base_pkg["render"][:3]))]
        saved_files = {"base": str((view_dir / "base_render.png").resolve())}

        for name, override_color in override_specs:
            pkg = render_simple(camera, model, background, override_color=override_color)
            filename = f"{name}_render.png"
            image_u8 = save_rgb(view_dir / filename, pkg["render"][:3])
            tiles.append((name, image_u8))
            saved_files[name] = str((view_dir / filename).resolve())

        grid = make_labeled_grid(tiles, columns=max(1, int(grid_columns)))
        grid_path = view_dir / "comparison_grid.png"
        grid.save(grid_path)
        overview_tiles.append(
            (
                f"{view_idx:03d}_{view_name}",
                resize_rgb_for_overview(np.asarray(grid.convert("RGB"), dtype=np.uint8), int(overview_max_width)),
            )
        )
        view_summaries.append(
            {
                "view_index": int(view_idx),
                "image_name": str(camera.image_name),
                "output_dir": str(view_dir.resolve()),
                "comparison_grid": str(grid_path.resolve()),
                "files": saved_files,
            }
        )

    if overview_tiles:
        overview = make_labeled_grid(overview_tiles, columns=1)
        overview.save(render_dir / "comparison_overview_v0.png")
    return view_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-Gaussian mesh-depth TSDF regulation payload. "
            "This stage renders/querys a gs2mesh mesh as depth prior and estimates surface-action weights."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--mesh_depth_mode", choices=["raycast", "sample_zbuffer"], default="raycast")
    parser.add_argument("--raycast_downsample", type=int, default=4)
    parser.add_argument("--raycast_chunk", type=int, default=262144)
    parser.add_argument("--mesh_surface_query_mode", choices=["auto", "exact", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=1200000)
    parser.add_argument("--mesh_surface_chunk_size", type=int, default=200000)
    parser.add_argument("--mesh_sample_points_per_face", type=int, choices=[1, 4], default=1)
    parser.add_argument("--mesh_sample_splat_kernel", type=int, default=5)
    parser.add_argument("--disable_front_facing_only", action="store_true")
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--tsdf_abs_trunc", type=float, default=0.04)
    parser.add_argument("--tsdf_rel_trunc", type=float, default=0.015)
    parser.add_argument("--tsdf_mesh_tau_multiplier", type=float, default=1.5)
    parser.add_argument("--support_sample_scale", type=float, default=1.0)
    parser.add_argument("--near_tsdf_threshold", type=float, default=0.45)
    parser.add_argument("--front_tsdf_threshold", type=float, default=0.65)
    parser.add_argument("--behind_tsdf_threshold", type=float, default=0.65)
    parser.add_argument("--min_support_views", type=int, default=2)
    parser.add_argument("--edge_abs_threshold", type=float, default=0.03)
    parser.add_argument("--edge_rel_threshold", type=float, default=0.01)
    parser.add_argument("--edge_kernel", type=int, default=7)
    parser.add_argument("--edge_dilate_kernel", type=int, default=9)
    parser.add_argument("--surface_mass_protect", type=float, default=0.65)
    parser.add_argument("--surface_extract_threshold", type=float, default=0.28)
    parser.add_argument("--opacity_suppress_threshold", type=float, default=0.35)
    parser.add_argument("--preview_views", type=int, default=4)
    parser.add_argument("--max_preview_points", type=int, default=500000)
    parser.add_argument("--export_depth_debug_images", action="store_true")
    parser.add_argument("--render_debug_views", type=int, default=6)
    parser.add_argument("--render_debug_grid_columns", type=int, default=3)
    parser.add_argument("--render_debug_overview_max_width", type=int, default=1600)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration = resolve_iteration(model_path, int(args.iteration))
    model = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    mesh = load_triangle_mesh(str(mesh_path))
    cameras = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    cameras = select_uniform(cameras, int(args.max_views))

    xyz = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = model.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    scaling = model.get_scaling_with_3D_filter.detach().cpu().numpy().astype(np.float32, copy=False)
    rotations = build_rotation(model._rotation.detach()).detach().cpu().numpy().astype(np.float32, copy=False)
    scale_major = np.max(scaling, axis=1).astype(np.float32)
    scale_minor = np.min(scaling, axis=1).astype(np.float32)
    anisotropy = (scale_major / np.clip(scale_minor, 1e-8, None)).astype(np.float32)
    sample_points, sample_offsets = build_gaussian_support_samples(
        xyz,
        rotations,
        scaling,
        sample_scale=float(args.support_sample_scale),
    )
    samples_per_gaussian = int(sample_offsets.shape[0])

    print(f"[mesh-depth-tsdf-reg-v0] model: {model_path} iter={iteration} gaussians={xyz.shape[0]}")
    print(f"[mesh-depth-tsdf-reg-v0] mesh : {mesh_path} faces={len(mesh.faces)}")
    print(f"[mesh-depth-tsdf-reg-v0] views: {len(cameras)} split={args.split} images={args.images_subdir}")

    surface = query_mesh_surface(
        mesh,
        xyz,
        mode=str(args.mesh_surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.mesh_surface_chunk_size),
    )
    nearest_point = surface["nearest_surface_point"].astype(np.float32, copy=False)
    nearest_normal = normalize_np(surface["nearest_surface_normal"].astype(np.float32, copy=False))
    nearest_face_id = surface["nearest_face_id"].astype(np.int64, copy=False)
    surface_distance = surface["surface_distance"].astype(np.float32, copy=False)
    local_edge = local_mesh_edge_lengths(mesh, nearest_face_id)
    bary = barycentric_for_faces(mesh, nearest_point, nearest_face_id)
    signed_offset = np.sum((xyz - nearest_point) * nearest_normal, axis=1).astype(np.float32)

    n = xyz.shape[0]
    ns = sample_points.shape[0]
    v = len(cameras)
    tsdf_values = np.full((v, ns), np.nan, dtype=np.float32)
    tsdf_norm_values = np.full((v, ns), np.nan, dtype=np.float32)
    support_matrix = np.zeros((v, ns), dtype=bool)
    edge_matrix = np.zeros((v, ns), dtype=bool)
    face_matrix = np.full((v, ns), -1, dtype=np.int64)
    sample_local_edge = np.repeat(local_edge, samples_per_gaussian).astype(np.float32, copy=False)

    scene = None
    if str(args.mesh_depth_mode) == "raycast":
        scene = build_open3d_raycast_scene(mesh)

    debug_dir = output_dir / "mesh_depth_tsdf_debug_v0"
    view_summaries: List[Dict[str, object]] = []
    for view_idx, cam in enumerate(tqdm(cameras, desc="mesh-depth TSDF query", dynamic_ncols=True)):
        if str(args.mesh_depth_mode) == "raycast":
            depth, face_depth = render_mesh_depth_raycast(
                scene,
                cam,
                downsample=int(args.raycast_downsample),
                ray_chunk=int(args.raycast_chunk),
            )
            down = max(int(args.raycast_downsample), 1)
        else:
            depth, face_depth = render_mesh_depth_sample_zbuffer(mesh, cam, args)
            down = 1

        edge_mask = depth_edge_mask_np(
            depth,
            abs_threshold=float(args.edge_abs_threshold),
            rel_threshold=float(args.edge_rel_threshold),
            kernel_size=int(args.edge_kernel),
            dilate_kernel=int(args.edge_dilate_kernel),
        )

        projected, in_view = project_points_camera(cam, sample_points, depth_min=float(args.depth_min), margin=0)
        xy_low = np.empty((ns, 2), dtype=np.float32)
        xy_low[:, 0] = (projected[:, 0] + 0.5) / float(down) - 0.5
        xy_low[:, 1] = (projected[:, 1] + 0.5) / float(down) - 0.5

        sampled_depth, valid_pix = sample_nearest_hw(depth, xy_low)
        sampled_face, valid_face_pix = sample_nearest_int_hw(face_depth, xy_low)
        sampled_edge, valid_edge_pix = sample_nearest_hw(edge_mask.astype(np.float32), xy_low)
        valid = in_view & valid_pix & np.isfinite(sampled_depth) & (sampled_depth > float(args.depth_min))
        if np.any(valid):
            tau = np.maximum.reduce(
                [
                    np.full((ns,), float(args.tsdf_abs_trunc), dtype=np.float32),
                    float(args.tsdf_rel_trunc) * np.maximum(sampled_depth.astype(np.float32), float(args.depth_min)),
                    float(args.tsdf_mesh_tau_multiplier) * np.maximum(sample_local_edge, 1e-6),
                ]
            ).astype(np.float32)
            signed = (sampled_depth - projected[:, 2]).astype(np.float32)
            tsdf = np.clip(signed / np.clip(tau, 1e-8, None), -1.0, 1.0).astype(np.float32)
            tsdf_values[view_idx, valid] = signed[valid]
            tsdf_norm_values[view_idx, valid] = tsdf[valid]
            support_matrix[view_idx, valid] = True
            edge_matrix[view_idx, valid] = (sampled_edge[valid] > 0.5) & valid_edge_pix[valid]
            face_matrix[view_idx, valid] = sampled_face[valid]

        if bool(args.export_depth_debug_images) and view_idx < int(args.preview_views):
            stem = str(cam.image_name)
            save_depth_preview(debug_dir / f"{view_idx:03d}_{stem}_mesh_depth.png", depth)
            save_mask_preview(debug_dir / f"{view_idx:03d}_{stem}_mesh_depth_edge.png", edge_mask)

        finite_depth = np.isfinite(depth)
        view_summaries.append(
            {
                "view_index": int(view_idx),
                "image_name": str(cam.image_name),
                "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
                "mesh_depth_pixels": int(np.count_nonzero(finite_depth)),
                "mesh_edge_pixels": int(np.count_nonzero(edge_mask)),
                "projected_samples": int(np.count_nonzero(in_view)),
                "supported_samples": int(np.count_nonzero(valid)),
                "projected_gaussians": int(np.count_nonzero(in_view.reshape(n, samples_per_gaussian).any(axis=1))),
                "supported_gaussians": int(np.count_nonzero(valid.reshape(n, samples_per_gaussian).any(axis=1))),
            }
        )

    sample_support_count = support_matrix.sum(axis=0).astype(np.int16)
    sample_edge_count = edge_matrix.sum(axis=0).astype(np.int16)
    support_count = support_matrix.reshape(v, n, samples_per_gaussian).any(axis=2).sum(axis=0).astype(np.int16)
    edge_count = edge_matrix.reshape(v, n, samples_per_gaussian).any(axis=2).sum(axis=0).astype(np.int16)
    edge_ratio = edge_count.astype(np.float32) / np.maximum(support_count.astype(np.float32), 1.0)

    sample_tsdf_median = nanmedian_with_fill(tsdf_values, fill=0.0)
    sample_tsdf_abs_median = nanmedian_with_fill(np.abs(tsdf_values), fill=0.0)
    sample_tsdf_norm_median = nanmedian_with_fill(tsdf_norm_values, fill=0.0)
    sample_tsdf_norm_abs_median = nanmedian_with_fill(np.abs(tsdf_norm_values), fill=0.0)
    tsdf_median = sample_tsdf_median.reshape(n, samples_per_gaussian)[:, 0].astype(np.float32, copy=False)
    tsdf_abs_median = sample_tsdf_abs_median.reshape(n, samples_per_gaussian)[:, 0].astype(np.float32, copy=False)
    tsdf_norm_median = sample_tsdf_norm_median.reshape(n, samples_per_gaussian)[:, 0].astype(np.float32, copy=False)
    tsdf_norm_abs_median = sample_tsdf_norm_abs_median.reshape(n, samples_per_gaussian)[:, 0].astype(np.float32, copy=False)
    finite_tsdf = np.isfinite(tsdf_norm_values)
    sample_near = (
        (np.abs(tsdf_norm_values) <= float(args.near_tsdf_threshold)) & finite_tsdf
    ).sum(axis=0).astype(np.float32) / np.maximum(sample_support_count.astype(np.float32), 1.0)
    sample_front = (
        (tsdf_norm_values >= float(args.front_tsdf_threshold)) & finite_tsdf
    ).sum(axis=0).astype(np.float32) / np.maximum(sample_support_count.astype(np.float32), 1.0)
    sample_behind = (
        (tsdf_norm_values <= -float(args.behind_tsdf_threshold)) & finite_tsdf
    ).sum(axis=0).astype(np.float32) / np.maximum(sample_support_count.astype(np.float32), 1.0)
    near_samples = sample_near.reshape(n, samples_per_gaussian)
    front_samples = sample_front.reshape(n, samples_per_gaussian)
    behind_samples = sample_behind.reshape(n, samples_per_gaussian)
    sample_valid = (sample_support_count.reshape(n, samples_per_gaussian) > 0).astype(np.float32)
    valid_denom = np.maximum(sample_valid.sum(axis=1), 1.0)
    surface_mass = (near_samples * sample_valid).sum(axis=1).astype(np.float32) / valid_denom
    front_mass = (front_samples * sample_valid).sum(axis=1).astype(np.float32) / valid_denom
    behind_mass = (behind_samples * sample_valid).sum(axis=1).astype(np.float32) / valid_denom
    crossing_score = np.minimum(front_mass, behind_mass).astype(np.float32)
    near_score = near_samples[:, 0].astype(np.float32, copy=False)
    front_score = front_samples[:, 0].astype(np.float32, copy=False)
    behind_score = behind_samples[:, 0].astype(np.float32, copy=False)

    local_tau = np.maximum.reduce(
        [
            np.full((n,), float(args.tsdf_abs_trunc), dtype=np.float32),
            float(args.tsdf_mesh_tau_multiplier) * np.maximum(local_edge, 1e-6),
        ]
    ).astype(np.float32)
    signed_offset_norm = (signed_offset / np.clip(local_tau, 1e-8, None)).astype(np.float32)

    state, action = compute_action_weights(
        support_count=support_count,
        surface_mass=surface_mass,
        front_mass=front_mass,
        behind_mass=behind_mass,
        crossing_score=crossing_score,
        edge_ratio=edge_ratio,
        args=args,
    )
    state_confidence = np.clip(
        0.45 * surface_mass
        + 0.25 * (support_count.astype(np.float32) / max(float(len(cameras)), 1.0))
        + 0.20 * (1.0 - action["edge_uncertainty"])
        + 0.10 * np.maximum(action["surface_extract_weight"], action["residual_keep_weight"]),
        0.0,
        1.0,
    ).astype(np.float32)

    mesh_bind_weight = action["mesh_bind_weight"]
    surface_extract_weight = action["surface_extract_weight"]
    opacity_decay_weight = action["opacity_suppress_weight"]
    residual_keep_weight = action["residual_keep_weight"]
    edge_uncertainty = action["edge_uncertainty"]
    allow_mesh_bind = surface_extract_weight >= float(args.surface_extract_threshold)
    allow_cov_tangent = (surface_extract_weight > 0.10) | (edge_uncertainty > 0.25)
    allow_center_attach = mesh_bind_weight >= float(args.surface_extract_threshold)
    allow_normal_offset = (surface_extract_weight >= 0.10) & (crossing_score > 0.05)
    residual_keep = action["residual_keep_weight"] >= 0.50

    output_payload = {
        "version": "mesh_depth_tsdf_regulation_v0",
        "state_names": STATE_NAMES,
        "gaussian_state": torch.from_numpy(state.astype(np.int16, copy=False)),
        "support_count": torch.from_numpy(support_count.astype(np.int16, copy=False)),
        "edge_count": torch.from_numpy(edge_count.astype(np.int16, copy=False)),
        "edge_ratio": torch.from_numpy(edge_ratio.astype(np.float32, copy=False)),
        "tsdf_median": torch.from_numpy(tsdf_median),
        "tsdf_abs_median": torch.from_numpy(tsdf_abs_median),
        "tsdf_norm_median": torch.from_numpy(tsdf_norm_median),
        "tsdf_norm_abs_median": torch.from_numpy(tsdf_norm_abs_median),
        "near_score": torch.from_numpy(near_score.astype(np.float32, copy=False)),
        "front_score": torch.from_numpy(front_score.astype(np.float32, copy=False)),
        "behind_score": torch.from_numpy(behind_score.astype(np.float32, copy=False)),
        "surface_mass": torch.from_numpy(surface_mass.astype(np.float32, copy=False)),
        "front_mass": torch.from_numpy(front_mass.astype(np.float32, copy=False)),
        "behind_mass": torch.from_numpy(behind_mass.astype(np.float32, copy=False)),
        "crossing_score": torch.from_numpy(crossing_score.astype(np.float32, copy=False)),
        "mesh_bind_weight": torch.from_numpy(mesh_bind_weight.astype(np.float32, copy=False)),
        "surface_extract_weight": torch.from_numpy(surface_extract_weight.astype(np.float32, copy=False)),
        "opacity_suppress_weight": torch.from_numpy(opacity_decay_weight.astype(np.float32, copy=False)),
        "residual_keep_weight": torch.from_numpy(residual_keep_weight.astype(np.float32, copy=False)),
        "edge_uncertainty": torch.from_numpy(edge_uncertainty.astype(np.float32, copy=False)),
        "sample_offsets": torch.from_numpy(sample_offsets.astype(np.float32, copy=False)),
        "sample_support_count": torch.from_numpy(sample_support_count.reshape(n, samples_per_gaussian).astype(np.int16, copy=False)),
        "sample_tsdf_norm_median": torch.from_numpy(sample_tsdf_norm_median.reshape(n, samples_per_gaussian).astype(np.float32, copy=False)),
        "sample_near_score": torch.from_numpy(near_samples.astype(np.float32, copy=False)),
        "sample_front_score": torch.from_numpy(front_samples.astype(np.float32, copy=False)),
        "sample_behind_score": torch.from_numpy(behind_samples.astype(np.float32, copy=False)),
        "state_confidence": torch.from_numpy(state_confidence.astype(np.float32, copy=False)),
        "allow_mesh_bind": torch.from_numpy(allow_mesh_bind.astype(bool)),
        "allow_cov_tangent": torch.from_numpy(allow_cov_tangent.astype(bool)),
        "allow_center_attach": torch.from_numpy(allow_center_attach.astype(bool)),
        "allow_normal_offset": torch.from_numpy(allow_normal_offset.astype(bool)),
        "opacity_decay_weight": torch.from_numpy(opacity_decay_weight.astype(np.float32, copy=False)),
        "residual_keep": torch.from_numpy(residual_keep.astype(bool)),
        "nearest_surface_point": torch.from_numpy(nearest_point.astype(np.float32, copy=False)),
        "nearest_surface_normal": torch.from_numpy(nearest_normal.astype(np.float32, copy=False)),
        "nearest_face_id": torch.from_numpy(nearest_face_id.astype(np.int64, copy=False)),
        "nearest_barycentric": torch.from_numpy(bary.astype(np.float32, copy=False)),
        "surface_distance": torch.from_numpy(surface_distance.astype(np.float32, copy=False)),
        "signed_normal_offset": torch.from_numpy(signed_offset.astype(np.float32, copy=False)),
        "signed_normal_offset_norm": torch.from_numpy(signed_offset_norm.astype(np.float32, copy=False)),
        "local_mesh_edge_length": torch.from_numpy(local_edge.astype(np.float32, copy=False)),
        "opacity": torch.from_numpy(opacity.astype(np.float32, copy=False)),
        "scale_major": torch.from_numpy(scale_major.astype(np.float32, copy=False)),
        "scale_minor": torch.from_numpy(scale_minor.astype(np.float32, copy=False)),
        "anisotropy": torch.from_numpy(anisotropy.astype(np.float32, copy=False)),
        "source_tag": torch.from_numpy(gaussian_tracking_array(model, "source_tag", np.int16, 0)),
        "seed_id": torch.from_numpy(gaussian_tracking_array(model, "seed_id", np.int64, -1)),
        "generation": torch.from_numpy(gaussian_tracking_array(model, "generation", np.int16, 0)),
        "meta": {
            "model_path": str(model_path),
            "mesh_path": str(mesh_path),
            "scene_root": str(scene_root),
            "iteration": int(iteration),
            "images_subdir": str(args.images_subdir),
            "split": str(args.split),
            "mesh_depth_mode": str(args.mesh_depth_mode),
            "raycast_downsample": int(args.raycast_downsample),
            "surface_query_mode_used": str(surface.get("surface_query_mode_used", args.mesh_surface_query_mode)),
            "args": vars(args),
        },
    }

    payload_path = output_dir / "mesh_depth_tsdf_regulation_v0.pt"
    torch.save(output_payload, payload_path)

    preview_dir = output_dir / "point_cloud_previews"
    save_point_cloud(preview_dir / "state_preview_v0.ply", xyz, state_to_rgb(state), int(args.max_preview_points))
    save_point_cloud(preview_dir / "surface_mass_preview_v0.ply", xyz, scalar_to_rgb(surface_mass), int(args.max_preview_points))
    save_point_cloud(preview_dir / "front_mass_preview_v0.ply", xyz, scalar_to_rgb(front_mass), int(args.max_preview_points))
    save_point_cloud(preview_dir / "behind_mass_preview_v0.ply", xyz, scalar_to_rgb(behind_mass), int(args.max_preview_points))
    save_point_cloud(preview_dir / "crossing_score_preview_v0.ply", xyz, scalar_to_rgb(crossing_score), int(args.max_preview_points))
    save_point_cloud(preview_dir / "mesh_bind_weight_preview_v0.ply", xyz, scalar_to_rgb(mesh_bind_weight), int(args.max_preview_points))
    save_point_cloud(preview_dir / "surface_extract_weight_preview_v0.ply", xyz, scalar_to_rgb(surface_extract_weight), int(args.max_preview_points))
    save_point_cloud(preview_dir / "opacity_suppress_weight_preview_v0.ply", xyz, scalar_to_rgb(opacity_decay_weight), int(args.max_preview_points))
    save_point_cloud(preview_dir / "edge_uncertainty_preview_v0.ply", xyz, scalar_to_rgb(edge_uncertainty), int(args.max_preview_points))

    render_debug_views = select_uniform(cameras, int(args.render_debug_views)) if int(args.render_debug_views) > 0 else []
    render_debug_summaries = render_regulation_debug_previews(
        model=model,
        cameras=render_debug_views,
        output_dir=output_dir,
        state=state,
        surface_mass=surface_mass,
        crossing_score=crossing_score,
        mesh_bind_weight=mesh_bind_weight,
        surface_extract_weight=surface_extract_weight,
        opacity_suppress_weight=opacity_decay_weight,
        residual_keep_weight=residual_keep_weight,
        edge_uncertainty=edge_uncertainty,
        white_background=bool(args.white_background),
        grid_columns=int(args.render_debug_grid_columns),
        overview_max_width=int(args.render_debug_overview_max_width),
    )

    state_counts = {name: int(np.count_nonzero(state == idx)) for idx, name in STATE_NAMES.items()}
    summary = {
        "version": "mesh_depth_tsdf_regulation_v0",
        "model_path": str(model_path),
        "mesh_path": str(mesh_path),
        "scene_root": str(scene_root),
        "iteration": int(iteration),
        "num_gaussians": int(n),
        "samples_per_gaussian": int(samples_per_gaussian),
        "num_views": int(len(cameras)),
        "state_counts": state_counts,
        "allow_mesh_bind_count": int(np.count_nonzero(allow_mesh_bind)),
        "allow_cov_tangent_count": int(np.count_nonzero(allow_cov_tangent)),
        "allow_center_attach_count": int(np.count_nonzero(allow_center_attach)),
        "allow_normal_offset_count": int(np.count_nonzero(allow_normal_offset)),
        "opacity_decay_nonzero_count": int(np.count_nonzero(opacity_decay_weight > 0.0)),
        "surface_query_mode_used": str(surface.get("surface_query_mode_used", args.mesh_surface_query_mode)),
        "stats": {
            "support_count": stats_from_array(support_count.astype(np.float32)),
            "edge_ratio": stats_from_array(edge_ratio),
            "tsdf_median": stats_from_array(tsdf_median),
            "tsdf_norm_abs_median": stats_from_array(tsdf_norm_abs_median),
            "near_score": stats_from_array(near_score),
            "front_score": stats_from_array(front_score),
            "behind_score": stats_from_array(behind_score),
            "surface_mass": stats_from_array(surface_mass),
            "front_mass": stats_from_array(front_mass),
            "behind_mass": stats_from_array(behind_mass),
            "crossing_score": stats_from_array(crossing_score),
            "mesh_bind_weight": stats_from_array(mesh_bind_weight),
            "surface_extract_weight": stats_from_array(surface_extract_weight),
            "opacity_suppress_weight": stats_from_array(opacity_decay_weight),
            "residual_keep_weight": stats_from_array(residual_keep_weight),
            "edge_uncertainty": stats_from_array(edge_uncertainty),
            "state_confidence": stats_from_array(state_confidence),
            "surface_distance": stats_from_array(surface_distance),
            "signed_normal_offset_norm": stats_from_array(signed_offset_norm),
            "opacity_decay_weight": stats_from_array(opacity_decay_weight),
        },
        "view_summaries": view_summaries,
        "render_debug_summaries": render_debug_summaries,
        "artifacts": {
            "payload": str(payload_path),
            "preview_dir": str(preview_dir),
            "render_debug_dir": str(output_dir / "render_debug_v0"),
            "depth_debug_dir": str(debug_dir) if bool(args.export_depth_debug_images) else "",
        },
        "note": (
            "This v0 payload uses a gs2mesh-style mesh as a projective depth prior. "
            "It does not edit the Gaussian model; it samples each Gaussian support domain, estimates usable surface mass, "
            "crossing score, off-surface mass, and records conservative action weights for later regulation."
        ),
    }
    summary_path = output_dir / "mesh_depth_tsdf_regulation_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[mesh-depth-tsdf-reg-v0] payload : {payload_path}")
    print(f"[mesh-depth-tsdf-reg-v0] summary : {summary_path}")


if __name__ == "__main__":
    main()
