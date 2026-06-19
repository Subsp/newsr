#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

SOF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIPSPLATTING_ROOT = SOF_ROOT.parent / "mip-splatting"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class Projected:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
        self.valid = valid


def _clone_subset_gaussians(base, ids: torch.Tensor):
    from scene.gaussian_model import GaussianModel

    ids = ids.to(device=base.get_xyz.device, dtype=torch.long).reshape(-1)
    count = int(ids.shape[0])
    subset = GaussianModel(base.max_sh_degree)
    subset.active_sh_degree = int(base.active_sh_degree)
    subset.spatial_lr_scale = float(base.spatial_lr_scale)
    subset._xyz = nn.Parameter(base._xyz.detach()[ids].clone().requires_grad_(False))
    subset._features_dc = nn.Parameter(base._features_dc.detach()[ids].clone().requires_grad_(False))
    subset._features_rest = nn.Parameter(base._features_rest.detach()[ids].clone().requires_grad_(False))
    subset._opacity = nn.Parameter(base._opacity.detach()[ids].clone().requires_grad_(False))
    subset._scaling = nn.Parameter(base._scaling.detach()[ids].clone().requires_grad_(False))
    subset._rotation = nn.Parameter(base._rotation.detach()[ids].clone().requires_grad_(False))
    if isinstance(getattr(base, "filter_3D", None), torch.Tensor) and int(base.filter_3D.shape[0]) == int(base.get_xyz.shape[0]):
        subset.filter_3D = base.filter_3D.detach()[ids].clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.max_radii2D = torch.zeros((count,), dtype=torch.float32, device=base.get_xyz.device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    subset.denom = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    if hasattr(subset, "xyz_gradient_accum_abs"):
        subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    if hasattr(subset, "xyz_gradient_accum_abs_max"):
        subset.xyz_gradient_accum_abs_max = torch.zeros((count, 1), dtype=torch.float32, device=base.get_xyz.device)
    return subset


def _ensure_mipsplatting_imports(root: Path) -> None:
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _build_dataset_args(scene_root: str, model_path: str, images_subdir: str, white_background: bool) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=scene_root,
        model_path=model_path,
        images=images_subdir,
        resolution=1,
        white_background=white_background,
        data_device="cuda",
        eval=True,
        kernel_size=0.1,
        ray_jitter=False,
        resample_gt_image=False,
        load_allres=False,
        sample_more_highres=False,
    )


def _build_pipe_args() -> Namespace:
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        compute_filter3D_python=False,
        compute_view2gaussian_python=False,
        use_merged_sof_rasterizer=False,
        use_vanilla_sof_rasterizer=False,
        require_merged_sof_aux=False,
        debug=False,
    )


def _resolve_iteration(model_path: Path, iteration: int) -> int:
    if int(iteration) >= 0:
        return int(iteration)
    point_root = model_path / "point_cloud"
    candidates: List[int] = []
    for child in point_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_root}")
    return max(candidates)


def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _norm_key(text: str) -> str:
    return Path(str(text)).stem.lower()


def _build_lookup(paths: Iterable[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in paths:
        out[_norm_key(path.name)] = path
    return out


def _resolve_series(
    *,
    cameras: Sequence[object],
    root: Path,
    role: str,
    match_policy: str,
) -> Tuple[Dict[str, Path], Dict[str, object]]:
    paths = _list_images(root)
    if not paths:
        raise FileNotFoundError(f"{role}: no images under {root}")
    lookup = _build_lookup(paths)
    resolved: Dict[str, Path] = {}
    exact = 0
    for cam in cameras:
        key = _norm_key(str(cam.image_name))
        if key in lookup:
            resolved[str(cam.image_name)] = lookup[key]
            exact += 1
    policy = str(match_policy).strip().lower()
    if len(resolved) == len(cameras):
        return resolved, {"role": role, "policy": "stem", "exact": exact, "order": 0, "files": len(paths)}
    if policy in {"order", "llff_train_order", "order_if_needed"}:
        if len(paths) < len(cameras):
            raise ValueError(f"{role}: cannot order-match {len(paths)} files to {len(cameras)} cameras")
        resolved = {str(cam.image_name): paths[i] for i, cam in enumerate(cameras)}
        return resolved, {"role": role, "policy": policy, "exact": exact, "order": len(cameras), "files": len(paths)}
    missing = [str(cam.image_name) for cam in cameras if str(cam.image_name) not in resolved][:16]
    raise ValueError(f"{role}: missing camera-name matches under {root}: {missing}")


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _box_blur(gray: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return gray.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(gray.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    out = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    ) / float(k * k)
    return out.astype(np.float32, copy=False)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _signed_hf_residual(sr: np.ndarray, anchor: np.ndarray, kernel: int) -> np.ndarray:
    sr_g = _rgb_to_gray(sr)
    an_g = _rgb_to_gray(anchor)
    return ((sr_g - _box_blur(sr_g, kernel)) - (an_g - _box_blur(an_g, kernel))).astype(np.float32)


def _normalize01(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    scale = float(np.percentile(np.abs(arr), float(percentile)))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(np.abs(arr) / scale, 0.0, 1.0).astype(np.float32)


def _sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    valid = (xs >= 0.0) & (ys >= 0.0) & (xs <= (w - 1)) & (ys <= (h - 1))
    x0 = np.floor(np.clip(xs, 0, w - 1)).astype(np.int32)
    y0 = np.floor(np.clip(ys, 0, h - 1)).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - x0
    wy = ys - y0
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]
    values = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v01 + (1 - wx) * wy * v10 + wx * wy * v11
    return values.astype(np.float32), valid


def _project_points(xyz: np.ndarray, camera: object) -> Projected:
    r = np.asarray(camera.R, dtype=np.float32)
    t = np.asarray(camera.T, dtype=np.float32).reshape(1, 3)
    width = int(camera.image_width)
    height = int(camera.image_height)
    focal_x = float(camera.focal_x)
    focal_y = float(camera.focal_y)
    xyz_cam = xyz @ r + t
    z = xyz_cam[:, 2]
    safe_z = np.clip(z, 1e-6, None)
    x = xyz_cam[:, 0] / safe_z * focal_x + width * 0.5
    y = xyz_cam[:, 1] / safe_z * focal_y + height * 0.5
    valid = z > 1e-6
    return Projected(x=x.astype(np.float32), y=y.astype(np.float32), z=z.astype(np.float32), valid=valid)


def _canonical_grid(radius: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    steps = max(3, int(steps))
    if steps % 2 == 0:
        steps += 1
    coords = np.linspace(-float(radius), float(radius), steps, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist2 = xx * xx + yy * yy
    keep = dist2 <= float(radius) ** 2 + 1e-6
    grid = np.stack([xx[keep], yy[keep]], axis=1).astype(np.float32)
    weights = np.exp(-0.5 * dist2[keep]).astype(np.float32)
    weights /= np.sum(weights).clip(1e-8)
    return grid, weights


def _sample_profiles_isotropic(
    signed_hf: np.ndarray,
    abs_hf: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    radii: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(cx.shape[0])
    signed_profiles = np.zeros((n, grid.shape[0]), dtype=np.float32)
    abs_profiles = np.zeros((n, grid.shape[0]), dtype=np.float32)
    valid_ratio = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        pts = grid * max(float(radii[i]), 1e-6)
        xs = cx[i] + pts[:, 0]
        ys = cy[i] + pts[:, 1]
        signed, valid = _sample_bilinear(signed_hf, xs, ys)
        absolute, _ = _sample_bilinear(abs_hf, xs, ys)
        signed_profiles[i] = signed * valid.astype(np.float32)
        abs_profiles[i] = absolute * valid.astype(np.float32)
        valid_ratio[i] = float(valid.mean())
    hit = (abs_profiles * weights[None, :]).sum(axis=1)
    return signed_profiles, hit.astype(np.float32), valid_ratio


def _profile_ncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a0 * a0).sum(axis=1) * (b0 * b0).sum(axis=1)).clip(1e-8)
    return ((a0 * b0).sum(axis=1) / denom).astype(np.float32)


def _draw_renderer_supported_score_map(
    width: int,
    height: int,
    edge_image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    *,
    support_radius_scale: float,
    edge_threshold: float,
    edge_power: float,
    max_draw: int,
) -> np.ndarray:
    support_map = np.zeros((height, width), dtype=np.float32)
    if scores.shape[0] == 0:
        return support_map
    order = np.argsort(scores)
    if int(max_draw) > 0 and order.shape[0] > int(max_draw):
        order = order[-int(max_draw):]
    edge_threshold = max(float(edge_threshold), 0.0)
    support_radius_scale = max(float(support_radius_scale), 0.25)
    edge_power = max(float(edge_power), 0.0)
    for i in order:
        score = float(scores[i])
        if score <= 0.0:
            continue
        radius = max(float(radii[i]) * support_radius_scale, 0.5)
        xmin = max(0, int(math.floor(float(xs[i]) - radius)))
        xmax = min(width, int(math.ceil(float(xs[i]) + radius + 1.0)))
        ymin = max(0, int(math.floor(float(ys[i]) - radius)))
        ymax = min(height, int(math.ceil(float(ys[i]) + radius + 1.0)))
        if xmin >= xmax or ymin >= ymax:
            continue
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
        dx = (xx + 0.5 - float(xs[i])) / radius
        dy = (yy + 0.5 - float(ys[i])) / radius
        q = dx * dx + dy * dy
        edge_crop = edge_image[ymin:ymax, xmin:xmax]
        support = (q <= 1.0) & (edge_crop >= edge_threshold)
        if not np.any(support):
            continue
        value = score * np.exp(-0.5 * np.clip(q, 0.0, 1.0)).astype(np.float32)
        if edge_power > 0.0:
            value = value * np.power(np.clip(edge_crop, 0.0, 1.0), edge_power)
        crop = support_map[ymin:ymax, xmin:xmax]
        crop[support] = np.maximum(crop[support], value[support])
    return np.clip(support_map, 0.0, 1.0)


def _select_views(cameras: Sequence[object], limit: int) -> List[object]:
    cams = list(cameras)
    if int(limit) > 0:
        cams = cams[: int(limit)]
    return cams


def _neighbor_indices(index: int, total: int, radius: int) -> List[int]:
    out = []
    for delta in range(1, int(radius) + 1):
        if index - delta >= 0:
            out.append(index - delta)
        if index + delta < total:
            out.append(index + delta)
    return out


def _view_dirs(output_root: Path) -> Dict[str, Path]:
    names = ["hf_edge", "carrier_score", "ownership_score", "validated_edge", "overlay", "per_view"]
    dirs = {name: output_root / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build renderer-backed CAVE high-frequency Gaussian ownership diagnostics.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument(
        "--point_cloud_ply",
        default="",
        help="Deprecated in renderer-backed mode; load through --model_dir/--iteration instead.",
    )
    parser.add_argument("--sr_dir", required=True)
    parser.add_argument("--anchor_dir", required=True)
    parser.add_argument("--edge_mask_dir", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mipsplatting_root", default=str(DEFAULT_MIPSPLATTING_ROOT))
    parser.add_argument("--match_policy", default="llff_train_order", choices=["stem", "order", "order_if_needed", "llff_train_order"])
    parser.add_argument("--llffhold", type=int, default=8, help="Accepted for CLI compatibility; Scene uses the native mip-splatting split.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--hf_percentile", type=float, default=99.0)
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--min_radius_px", type=float, default=0.5)
    parser.add_argument("--max_radius_px", type=float, default=12.0)
    parser.add_argument("--candidate_radius_scale", type=float, default=1.0)
    parser.add_argument("--max_gaussians_per_view", type=int, default=8192)
    parser.add_argument("--min_center_hf", type=float, default=0.05)
    parser.add_argument("--edge_mask_threshold", type=float, default=0.05)
    parser.add_argument("--profile_grid_radius", type=float, default=2.0)
    parser.add_argument("--profile_grid_steps", type=int, default=5)
    parser.add_argument("--profile_min_valid", type=float, default=0.65)
    parser.add_argument("--neighbor_radius", type=int, default=1)
    parser.add_argument("--consistency_floor", type=float, default=0.0)
    parser.add_argument("--score_power", type=float, default=1.0)
    parser.add_argument("--draw_max_gaussians", type=int, default=4096)
    parser.add_argument("--ownership_support_radius", type=float, default=1.0)
    parser.add_argument("--ownership_edge_threshold", type=float, default=0.05)
    parser.add_argument("--ownership_edge_power", type=float, default=0.0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if str(args.point_cloud_ply).strip():
        raise ValueError("Renderer-backed CAVE uses Scene/GaussianModel; do not pass --point_cloud_ply.")

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = _view_dirs(output_root)

    _ensure_mipsplatting_imports(Path(args.mipsplatting_root))
    from gaussian_renderer import render  # type: ignore
    from hybrid_sdfgs.blocks import SOFPriorBlock, SOFPriorConfig  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel  # type: ignore

    iteration = _resolve_iteration(model_dir, int(args.iteration))
    dataset = _build_dataset_args(str(scene_root), str(model_dir), str(args.images_subdir), bool(args.white_background))
    pipe = _build_pipe_args()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    all_train_cameras = list(scene.getTrainCameras())
    train_cameras = _select_views(all_train_cameras, int(args.limit))
    if not train_cameras:
        raise RuntimeError("No train cameras selected.")

    sr_map, sr_summary = _resolve_series(
        cameras=train_cameras,
        root=Path(args.sr_dir).expanduser().resolve(),
        role="sr",
        match_policy=str(args.match_policy),
    )
    anchor_map, anchor_summary = _resolve_series(
        cameras=train_cameras,
        root=Path(args.anchor_dir).expanduser().resolve(),
        role="anchor",
        match_policy=str(args.match_policy),
    )
    edge_map = None
    edge_summary = None
    if str(args.edge_mask_dir).strip():
        edge_map, edge_summary = _resolve_series(
            cameras=train_cameras,
            root=Path(args.edge_mask_dir).expanduser().resolve(),
            role="edge_mask",
            match_policy=str(args.match_policy),
        )

    try:
        gaussians.compute_3D_filter(all_train_cameras.copy(), CUDA=not bool(pipe.compute_filter3D_python))
    except TypeError:
        gaussians.compute_3D_filter(all_train_cameras.copy())

    device = gaussians.get_xyz.device
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    xyz_np = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity_np = gaussians.get_opacity_with_3D_filter.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    opacity_t = torch.from_numpy(opacity_np).to(device=device)
    total_gaussians = int(xyz_np.shape[0])
    touch_block = SOFPriorBlock(
        SOFPriorConfig(
            prior_edge_touch_min_radius_px=float(args.min_radius_px),
            prior_edge_touch_radius_scale=float(args.candidate_radius_scale),
            prior_edge_touch_max_radius_px=float(args.max_radius_px),
        )
    )

    print(f"[cave-hf-v1] scene       : {scene_root}")
    print(f"[cave-hf-v1] model       : {model_dir}")
    print(f"[cave-hf-v1] iteration   : {loaded_iter}")
    print(f"[cave-hf-v1] output      : {output_root}")
    print(f"[cave-hf-v1] views       : {len(train_cameras)}")
    print(f"[cave-hf-v1] gaussians   : {total_gaussians}")
    print(f"[cave-hf-v1] match sr    : {sr_summary}")
    print(f"[cave-hf-v1] match anchor: {anchor_summary}")
    if edge_summary is not None:
        print(f"[cave-hf-v1] match mask  : {edge_summary}")
    print("[cave-hf-v1] gaussian path: Scene/GaussianModel + gaussian_renderer.render radii/visibility + SOFPriorBlock touch")

    signed_maps: Dict[str, np.ndarray] = {}
    abs_maps: Dict[str, np.ndarray] = {}
    mask_maps: Dict[str, np.ndarray] = {}
    for cam in train_cameras:
        width = int(cam.image_width)
        height = int(cam.image_height)
        sr = _load_rgb(sr_map[str(cam.image_name)], width, height)
        anchor = _load_rgb(anchor_map[str(cam.image_name)], width, height)
        signed = _signed_hf_residual(sr, anchor, int(args.highpass_kernel))
        abs_hf = _normalize01(signed, float(args.hf_percentile))
        if edge_map is not None:
            mask_rgb = _load_rgb(edge_map[str(cam.image_name)], width, height)
            mask = np.max(mask_rgb, axis=2).astype(np.float32)
            mask = (mask >= float(args.edge_mask_threshold)).astype(np.float32)
        else:
            mask = np.ones_like(abs_hf, dtype=np.float32)
        signed_maps[str(cam.image_name)] = signed
        abs_maps[str(cam.image_name)] = abs_hf
        mask_maps[str(cam.image_name)] = mask

    render_cache: Dict[int, Dict[str, object]] = {}

    def get_render_meta(index: int) -> Dict[str, object]:
        if index in render_cache:
            return render_cache[index]
        cam = train_cameras[index]
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, kernel_size=float(dataset.kernel_size))
        radii_t = pkg["radii"].detach().to(device=device, dtype=torch.float32).reshape(-1)
        visible_t = pkg["visibility_filter"].detach().to(device=device, dtype=torch.bool).reshape(-1)
        radii_np = radii_t.detach().cpu().numpy().astype(np.float32, copy=False)
        visible_np = visible_t.detach().cpu().numpy().astype(bool, copy=False)
        proj = _project_points(xyz_np, cam)
        meta = {
            "radii_t": radii_t,
            "visible_t": visible_t,
            "radii": radii_np,
            "visible": visible_np,
            "x": proj.x,
            "y": proj.y,
            "z": proj.z,
            "project_valid": proj.valid,
        }
        render_cache[index] = meta
        keep = max(2 * int(args.neighbor_radius) + 5, 8)
        if len(render_cache) > keep:
            for key in sorted(render_cache):
                if abs(key - index) > int(args.neighbor_radius) + 2:
                    render_cache.pop(key, None)
                    break
        return meta

    grid, weights = _canonical_grid(float(args.profile_grid_radius), int(args.profile_grid_steps))
    manifest_frames = []
    all_scores = []
    all_cons = []
    all_hits = []

    for view_index, cam in enumerate(train_cameras):
        name = str(cam.image_name)
        out_npz = dirs["per_view"] / f"{name}.npz"
        if out_npz.exists() and not bool(args.overwrite):
            print(f"[cave-hf-v1] skip existing {view_index + 1}/{len(train_cameras)} {name}", flush=True)
            continue

        width = int(cam.image_width)
        height = int(cam.image_height)
        abs_hf = abs_maps[name]
        signed = signed_maps[name]
        mask = mask_maps[name]
        proposal = (abs_hf * mask).astype(np.float32, copy=False)
        meta = get_render_meta(view_index)

        proposal_t = torch.from_numpy(proposal).to(device=device, dtype=torch.float32)
        touch_mask = touch_block.build_touch_mask(
            viewpoint_cam=cam,
            gaussians=gaussians,
            image_mask=proposal_t,
            visibility_filter=meta["visible_t"],
            radii=meta["radii_t"],
            min_radius_px=float(args.min_radius_px),
            radius_scale=float(args.candidate_radius_scale),
            max_radius_px=float(args.max_radius_px),
            mask_threshold=float(args.min_center_hf),
        )
        if touch_mask is None:
            touch_mask = torch.zeros((total_gaussians,), dtype=torch.bool, device=device)
        candidate_t = touch_mask & (opacity_t >= float(args.min_opacity))
        raw_candidates = int(candidate_t.sum().item())
        visible_gaussians = int(meta["visible_t"].sum().item())
        ids = candidate_t.nonzero(as_tuple=True)[0].detach().cpu().numpy().astype(np.int64)

        if ids.shape[0] > int(args.max_gaussians_per_view) > 0:
            center_hf, center_valid = _sample_bilinear(proposal, meta["x"][ids], meta["y"][ids])
            radii_rank = np.clip(meta["radii"][ids], 0.0, float(args.max_radius_px)) / max(float(args.max_radius_px), 1e-6)
            rank = (center_hf + 0.10 * radii_rank) * np.sqrt(opacity_np[ids].clip(0.0, 1.0))
            rank *= center_valid.astype(np.float32)
            keep = np.argpartition(-rank, int(args.max_gaussians_per_view) - 1)[: int(args.max_gaussians_per_view)]
            ids = ids[keep]

        if ids.shape[0] == 0:
            score_map = np.zeros((height, width), dtype=np.float32)
            _save_gray(dirs["hf_edge"] / f"{name}.png", abs_hf)
            _save_gray(dirs["carrier_score"] / f"{name}.png", score_map)
            _save_gray(dirs["ownership_score"] / f"{name}.png", score_map)
            _save_gray(dirs["validated_edge"] / f"{name}.png", score_map)
            continue

        radii = np.clip(
            meta["radii"][ids] * float(args.candidate_radius_scale),
            float(args.min_radius_px),
            float(args.max_radius_px),
        ).astype(np.float32, copy=False)
        cx = meta["x"][ids].astype(np.float32, copy=False)
        cy = meta["y"][ids].astype(np.float32, copy=False)
        valid_current = meta["visible"][ids] & meta["project_valid"][ids]
        profiles, hit, valid_ratio = _sample_profiles_isotropic(signed, proposal, cx, cy, radii, grid, weights)

        neighbor_scores = []
        neighbor_count = np.zeros((ids.shape[0],), dtype=np.float32)
        for ni in _neighbor_indices(view_index, len(train_cameras), int(args.neighbor_radius)):
            ncam = train_cameras[ni]
            nname = str(ncam.image_name)
            nmeta = get_render_meta(ni)
            nradii = np.clip(
                nmeta["radii"][ids] * float(args.candidate_radius_scale),
                float(args.min_radius_px),
                float(args.max_radius_px),
            ).astype(np.float32, copy=False)
            nprofiles, nhit, nvalid_ratio = _sample_profiles_isotropic(
                signed_maps[nname],
                abs_maps[nname] * mask_maps[nname],
                nmeta["x"][ids],
                nmeta["y"][ids],
                nradii,
                grid,
                weights,
            )
            ncc = _profile_ncc(profiles, nprofiles)
            nvisible = (
                nmeta["visible"][ids]
                & nmeta["project_valid"][ids]
                & (nvalid_ratio >= float(args.profile_min_valid))
                & (nhit > 0.0)
            )
            consistency = np.maximum(ncc, float(args.consistency_floor)) * nvisible.astype(np.float32)
            neighbor_scores.append(consistency.astype(np.float32))
            neighbor_count += nvisible.astype(np.float32)

        if neighbor_scores:
            consistency = np.maximum.reduce(neighbor_scores)
        else:
            consistency = np.ones_like(hit, dtype=np.float32)
            neighbor_count[:] = 1.0
        hit_norm = hit / np.percentile(hit[hit > 0], 95).clip(1e-6) if np.any(hit > 0) else hit
        hit_norm = np.clip(hit_norm, 0.0, 1.0)
        score = hit_norm * np.clip(consistency, 0.0, 1.0)
        if float(args.score_power) != 1.0:
            score = np.power(np.clip(score, 0.0, 1.0), float(args.score_power))
        score *= (valid_current & (valid_ratio >= float(args.profile_min_valid))).astype(np.float32)

        carrier_map = np.zeros((height, width), dtype=np.float32)
        render_score_ids = np.nonzero(score > 0.0)[0].astype(np.int64)
        if render_score_ids.shape[0] > int(args.draw_max_gaussians) > 0:
            render_rank = score[render_score_ids]
            keep = np.argpartition(-render_rank, int(args.draw_max_gaussians) - 1)[: int(args.draw_max_gaussians)]
            render_score_ids = render_score_ids[keep]
        if render_score_ids.shape[0] > 0:
            subset_ids_t = torch.from_numpy(ids[render_score_ids]).to(device=device, dtype=torch.long)
            subset_scores_t = torch.from_numpy(score[render_score_ids].astype(np.float32, copy=False)).to(device=device)
            carrier_subset = _clone_subset_gaussians(gaussians, subset_ids_t)
            carrier_color = subset_scores_t[:, None].expand(-1, 3).contiguous()
            with torch.no_grad():
                carrier_pkg = render(
                    cam,
                    carrier_subset,
                    pipe,
                    torch.zeros_like(background),
                    kernel_size=float(dataset.kernel_size),
                    override_color=carrier_color,
                )
            carrier_rgb = carrier_pkg["render"][:3].detach().cpu().numpy().transpose(1, 2, 0)
            carrier_map = np.max(carrier_rgb, axis=2).astype(np.float32, copy=False)
            carrier_map = np.clip(carrier_map, 0.0, 1.0)
        score_map = _draw_renderer_supported_score_map(
            width,
            height,
            proposal,
            cx,
            cy,
            radii,
            score,
            support_radius_scale=float(args.ownership_support_radius),
            edge_threshold=float(args.ownership_edge_threshold),
            edge_power=float(args.ownership_edge_power),
            max_draw=int(args.draw_max_gaussians),
        )
        validated = proposal * np.clip(score_map, 0.0, 1.0)
        overlay = np.stack([abs_hf, np.maximum(abs_hf * 0.35, validated), abs_hf * 0.25], axis=2)

        _save_gray(dirs["hf_edge"] / f"{name}.png", abs_hf)
        _save_gray(dirs["carrier_score"] / f"{name}.png", carrier_map)
        _save_gray(dirs["ownership_score"] / f"{name}.png", score_map)
        _save_gray(dirs["validated_edge"] / f"{name}.png", validated)
        _save_rgb(dirs["overlay"] / f"{name}.png", overlay)
        np.savez_compressed(
            out_npz,
            gaussian_id=ids.astype(np.int64),
            hit=hit.astype(np.float32),
            hit_norm=hit_norm.astype(np.float32),
            consistency=consistency.astype(np.float32),
            neighbor_count=neighbor_count.astype(np.float32),
            score=score.astype(np.float32),
            x=cx.astype(np.float32),
            y=cy.astype(np.float32),
            radius=radii.astype(np.float32),
            raw_candidates=np.asarray([raw_candidates], dtype=np.int64),
            visible_gaussians=np.asarray([visible_gaussians], dtype=np.int64),
        )
        frame = {
            "image_name": name,
            "candidates": int(ids.shape[0]),
            "raw_candidates": int(raw_candidates),
            "visible_gaussians": int(visible_gaussians),
            "candidate_ratio": float(raw_candidates / max(visible_gaussians, 1)),
            "hit_mean": float(hit.mean()) if hit.size else 0.0,
            "consistency_mean": float(consistency.mean()) if consistency.size else 0.0,
            "score_mean": float(score.mean()) if score.size else 0.0,
            "score_p90": float(np.percentile(score, 90)) if score.size else 0.0,
            "score_p95": float(np.percentile(score, 95)) if score.size else 0.0,
            "validated_ratio": float((score_map > 0.05).mean()),
            "carrier_ratio": float((carrier_map > 0.05).mean()),
        }
        manifest_frames.append(frame)
        all_scores.append(score)
        all_cons.append(consistency)
        all_hits.append(hit)
        print(
            f"[cave-hf-v1] {view_index + 1}/{len(train_cameras)} {name} "
            f"touch={raw_candidates} cand={ids.shape[0]} visible={visible_gaussians} "
            f"hit={frame['hit_mean']:.4f} cons={frame['consistency_mean']:.4f} score={frame['score_mean']:.4f}",
            flush=True,
        )

    def _cat_mean(values: List[np.ndarray]) -> Optional[float]:
        if not values:
            return None
        arr = np.concatenate([v.reshape(-1) for v in values if v.size > 0])
        if arr.size == 0:
            return None
        return float(arr.mean())

    summary = {
        "version": "renderer_backed_v1",
        "scene_root": str(scene_root),
        "model_dir": str(model_dir),
        "iteration": int(loaded_iter),
        "sr_dir": str(Path(args.sr_dir).expanduser().resolve()),
        "anchor_dir": str(Path(args.anchor_dir).expanduser().resolve()),
        "edge_mask_dir": str(Path(args.edge_mask_dir).expanduser().resolve()) if str(args.edge_mask_dir).strip() else "",
        "num_views": int(len(train_cameras)),
        "num_gaussians": int(total_gaussians),
        "score_mean": _cat_mean(all_scores),
        "consistency_mean": _cat_mean(all_cons),
        "hit_mean": _cat_mean(all_hits),
        "frames": manifest_frames,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[cave-hf-v1] manifest: {manifest_path}")
    print(json.dumps({k: summary[k] for k in ("score_mean", "consistency_mean", "hit_mean", "num_views")}, indent=2))


if __name__ == "__main__":
    main()
