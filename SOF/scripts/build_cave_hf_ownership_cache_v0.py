#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

SOF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIPSPLATTING_ROOT = SOF_ROOT.parent / "mip-splatting"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class GaussianCloud:
    xyz: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    opacity: np.ndarray
    max_scale: np.ndarray


@dataclass
class Projected:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    valid: np.ndarray


def _ensure_mipsplatting_imports(root: Path) -> None:
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_gaussians(ply_path: Path) -> GaussianCloud:
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(scale_names) != 3:
        raise ValueError(f"Expected 3 scale_* fields in {ply_path}, got {scale_names}")
    if len(rot_names) != 4:
        raise ValueError(f"Expected 4 rot_* fields in {ply_path}, got {rot_names}")
    raw_scale = np.stack([np.asarray(v[name]) for name in scale_names], axis=1).astype(np.float32)
    raw_rot = np.stack([np.asarray(v[name]) for name in rot_names], axis=1).astype(np.float32)
    raw_opacity = np.asarray(v["opacity"], dtype=np.float32).reshape(-1)
    scale = np.exp(raw_scale).astype(np.float32)
    rotation = raw_rot / np.linalg.norm(raw_rot, axis=1, keepdims=True).clip(1e-8)
    opacity = _sigmoid(raw_opacity).astype(np.float32)
    return GaussianCloud(
        xyz=xyz,
        scale=scale,
        rotation=rotation,
        opacity=opacity,
        max_scale=np.max(scale, axis=1).astype(np.float32),
    )


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - r * z)
    out[:, 0, 2] = 2 * (x * z + r * y)
    out[:, 1, 0] = 2 * (x * y + r * z)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - r * x)
    out[:, 2, 0] = 2 * (x * z - r * y)
    out[:, 2, 1] = 2 * (y * z + r * x)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _resolve_point_cloud(model_dir: Path, iteration: int, point_cloud_ply: Optional[str]) -> Path:
    if point_cloud_ply:
        path = Path(point_cloud_ply).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if int(iteration) < 0:
        root = model_dir / "point_cloud"
        candidates = []
        for child in root.glob("iteration_*"):
            if not child.is_dir():
                continue
            try:
                candidates.append(int(child.name.split("_")[-1]))
            except ValueError:
                continue
        if not candidates:
            raise FileNotFoundError(f"No point_cloud/iteration_* under {model_dir}")
        iteration = max(candidates)
    path = model_dir / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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
    width = _camera_width(camera)
    height = _camera_height(camera)
    focal_x = _camera_focal_x(camera)
    focal_y = _camera_focal_y(camera)
    xyz_cam = xyz @ r + t
    z = xyz_cam[:, 2]
    safe_z = np.clip(z, 1e-6, None)
    x = xyz_cam[:, 0] / safe_z * focal_x + width * 0.5
    y = xyz_cam[:, 1] / safe_z * focal_y + height * 0.5
    valid = z > 1e-6
    return Projected(x=x.astype(np.float32), y=y.astype(np.float32), z=z.astype(np.float32), valid=valid)


def _project_pixel(xyz: np.ndarray, camera: object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    proj = _project_points(xyz, camera)
    return proj.x, proj.y, proj.z


def _ellipse_cholesky_for_ids(
    cloud: GaussianCloud,
    ids: np.ndarray,
    camera: object,
    *,
    sigma_scale: float,
    min_sigma_px: float,
    max_sigma_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = cloud.xyz[ids]
    center_x, center_y, center_z = _project_pixel(xyz, camera)
    rot = _quat_to_rotmat(cloud.rotation[ids])
    axes = rot * cloud.scale[ids][:, None, :]
    cov = np.zeros((ids.shape[0], 2, 2), dtype=np.float32)
    valid = center_z > 1e-6
    for axis_idx in range(3):
        px, py, _ = _project_pixel(xyz + axes[:, :, axis_idx] * float(sigma_scale), camera)
        dx = px - center_x
        dy = py - center_y
        cov[:, 0, 0] += dx * dx
        cov[:, 0, 1] += dx * dy
        cov[:, 1, 0] += dy * dx
        cov[:, 1, 1] += dy * dy
    cov[:, 0, 0] += float(min_sigma_px) ** 2
    cov[:, 1, 1] += float(min_sigma_px) ** 2
    eigvals = np.linalg.eigvalsh(cov).astype(np.float32)
    radius = np.sqrt(np.maximum(eigvals[:, 1], 0.0))
    if float(max_sigma_px) > 0.0:
        shrink = np.minimum(1.0, float(max_sigma_px) / np.maximum(radius, 1e-6))
        cov *= (shrink * shrink)[:, None, None]
        radius *= shrink
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eye = np.eye(2, dtype=np.float32)[None]
        chol = np.linalg.cholesky(cov + 1e-3 * eye)
    return center_x, center_y, chol.astype(np.float32), valid


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


def _sample_profiles(
    signed_hf: np.ndarray,
    abs_hf: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    chol: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(cx.shape[0])
    signed_profiles = np.zeros((n, grid.shape[0]), dtype=np.float32)
    abs_profiles = np.zeros((n, grid.shape[0]), dtype=np.float32)
    valid_ratio = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        pts = grid @ chol[i].T
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


def _draw_score_map(
    width: int,
    height: int,
    xs: np.ndarray,
    ys: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    *,
    max_draw: int,
) -> np.ndarray:
    image = Image.new("F", (width, height), 0.0)
    draw = ImageDraw.Draw(image)
    if scores.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)
    order = np.argsort(scores)
    if int(max_draw) > 0 and order.shape[0] > int(max_draw):
        order = order[-int(max_draw):]
    for i in order:
        score = float(scores[i])
        if score <= 0.0:
            continue
        r = float(np.clip(radii[i], 1.0, 12.0))
        x = float(xs[i])
        y = float(ys[i])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=score)
    return np.asarray(image, dtype=np.float32).clip(0.0, 1.0)


def _select_views(cameras: Sequence[object], limit: int) -> List[object]:
    cams = list(cameras)
    if int(limit) > 0:
        cams = cams[: int(limit)]
    return cams


def _camera_width(camera: object) -> int:
    if hasattr(camera, "image_width"):
        return int(camera.image_width)
    return int(camera.width)


def _camera_height(camera: object) -> int:
    if hasattr(camera, "image_height"):
        return int(camera.image_height)
    return int(camera.height)


def _camera_focal_x(camera: object) -> float:
    if hasattr(camera, "focal_x"):
        return float(camera.focal_x)
    return _camera_width(camera) / (2.0 * math.tan(float(camera.FovX) * 0.5))


def _camera_focal_y(camera: object) -> float:
    if hasattr(camera, "focal_y"):
        return float(camera.focal_y)
    return _camera_height(camera) / (2.0 * math.tan(float(camera.FovY) * 0.5))


def _neighbor_indices(index: int, total: int, radius: int) -> List[int]:
    out = []
    for delta in range(1, int(radius) + 1):
        if index - delta >= 0:
            out.append(index - delta)
        if index + delta < total:
            out.append(index + delta)
    return out


def _view_dirs(output_root: Path) -> Dict[str, Path]:
    names = ["hf_edge", "ownership_score", "validated_edge", "overlay", "per_view"]
    dirs = {name: output_root / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAVE high-frequency Gaussian ownership diagnostics.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--point_cloud_ply", default="")
    parser.add_argument("--sr_dir", required=True)
    parser.add_argument("--anchor_dir", required=True)
    parser.add_argument("--edge_mask_dir", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mipsplatting_root", default=str(DEFAULT_MIPSPLATTING_ROOT))
    parser.add_argument("--match_policy", default="llff_train_order", choices=["stem", "order", "order_if_needed", "llff_train_order"])
    parser.add_argument("--llffhold", type=int, default=8)
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = _view_dirs(output_root)

    _ensure_mipsplatting_imports(Path(args.mipsplatting_root))
    from scene.dataset_readers import readColmapSceneInfo  # type: ignore

    scene_info = readColmapSceneInfo(str(scene_root), images=str(args.images_subdir), eval=True, llffhold=int(args.llffhold))
    train_cameras = _select_views(scene_info.train_cameras, int(args.limit))
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

    point_cloud_ply = _resolve_point_cloud(model_dir, int(args.iteration), str(args.point_cloud_ply or ""))
    cloud = _load_gaussians(point_cloud_ply)
    print(f"[cave-hf-v0] scene       : {scene_root}")
    print(f"[cave-hf-v0] model       : {model_dir}")
    print(f"[cave-hf-v0] ply         : {point_cloud_ply}")
    print(f"[cave-hf-v0] output      : {output_root}")
    print(f"[cave-hf-v0] views       : {len(train_cameras)}")
    print(f"[cave-hf-v0] gaussians   : {cloud.xyz.shape[0]}")
    print(f"[cave-hf-v0] match sr    : {sr_summary}")
    print(f"[cave-hf-v0] match anchor: {anchor_summary}")
    if edge_summary is not None:
        print(f"[cave-hf-v0] match mask  : {edge_summary}")

    signed_maps: Dict[str, np.ndarray] = {}
    abs_maps: Dict[str, np.ndarray] = {}
    mask_maps: Dict[str, np.ndarray] = {}
    for cam in train_cameras:
        width = _camera_width(cam)
        height = _camera_height(cam)
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

    grid, weights = _canonical_grid(float(args.profile_grid_radius), int(args.profile_grid_steps))
    manifest_frames = []
    all_scores = []
    all_cons = []
    all_hits = []

    for view_index, cam in enumerate(train_cameras):
        name = str(cam.image_name)
        out_npz = dirs["per_view"] / f"{name}.npz"
        if out_npz.exists() and not bool(args.overwrite):
            continue
        width = _camera_width(cam)
        height = _camera_height(cam)
        abs_hf = abs_maps[name]
        signed = signed_maps[name]
        mask = mask_maps[name]
        proj = _project_points(cloud.xyz, cam)
        approx_radius = cloud.max_scale * max(_camera_focal_x(cam), _camera_focal_y(cam)) / np.maximum(proj.z, 1e-6)
        approx_radius = np.clip(approx_radius * float(args.candidate_radius_scale), float(args.min_radius_px), float(args.max_radius_px))
        in_bounds = (
            proj.valid
            & (cloud.opacity >= float(args.min_opacity))
            & (proj.x >= -approx_radius)
            & (proj.y >= -approx_radius)
            & (proj.x < width + approx_radius)
            & (proj.y < height + approx_radius)
        )
        center_hf, center_valid = _sample_bilinear(abs_hf * mask, proj.x, proj.y)
        candidate_score = center_hf * center_valid.astype(np.float32) * in_bounds.astype(np.float32)
        candidate_score *= np.sqrt(cloud.opacity.clip(0.0, 1.0))
        candidate = candidate_score >= float(args.min_center_hf)
        ids = np.nonzero(candidate)[0].astype(np.int64)
        if ids.shape[0] > int(args.max_gaussians_per_view) > 0:
            scores = candidate_score[ids]
            keep = np.argpartition(-scores, int(args.max_gaussians_per_view) - 1)[: int(args.max_gaussians_per_view)]
            ids = ids[keep]
        if ids.shape[0] == 0:
            score_map = np.zeros((height, width), dtype=np.float32)
            _save_gray(dirs["hf_edge"] / f"{name}.png", abs_hf)
            _save_gray(dirs["ownership_score"] / f"{name}.png", score_map)
            _save_gray(dirs["validated_edge"] / f"{name}.png", score_map)
            continue

        cx, cy, chol, valid_ellipse = _ellipse_cholesky_for_ids(
            cloud,
            ids,
            cam,
            sigma_scale=1.0,
            min_sigma_px=float(args.min_radius_px),
            max_sigma_px=float(args.max_radius_px),
        )
        profiles, hit, valid_ratio = _sample_profiles(signed, abs_hf * mask, cx, cy, chol, grid, weights)
        neighbor_scores = []
        neighbor_count = np.zeros((ids.shape[0],), dtype=np.float32)
        for ni in _neighbor_indices(view_index, len(train_cameras), int(args.neighbor_radius)):
            ncam = train_cameras[ni]
            nname = str(ncam.image_name)
            ncx, ncy, nchol, nvalid_ellipse = _ellipse_cholesky_for_ids(
                cloud,
                ids,
                ncam,
                sigma_scale=1.0,
                min_sigma_px=float(args.min_radius_px),
                max_sigma_px=float(args.max_radius_px),
            )
            nprofiles, nhit, nvalid_ratio = _sample_profiles(
                signed_maps[nname],
                abs_maps[nname] * mask_maps[nname],
                ncx,
                ncy,
                nchol,
                grid,
                weights,
            )
            ncc = _profile_ncc(profiles, nprofiles)
            nvisible = (
                nvalid_ellipse
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
        score *= (valid_ellipse & (valid_ratio >= float(args.profile_min_valid))).astype(np.float32)
        radius = np.sqrt(np.maximum(np.linalg.eigvalsh(np.matmul(chol, np.transpose(chol, (0, 2, 1))))[:, 1], 0.0))
        score_map = _draw_score_map(width, height, cx, cy, radius, score, max_draw=int(args.draw_max_gaussians))
        validated = abs_hf * np.clip(score_map, 0.0, 1.0)
        overlay = np.stack([abs_hf, np.maximum(abs_hf * 0.35, validated), abs_hf * 0.25], axis=2)

        _save_gray(dirs["hf_edge"] / f"{name}.png", abs_hf)
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
            radius=radius.astype(np.float32),
        )
        frame = {
            "image_name": name,
            "candidates": int(ids.shape[0]),
            "hit_mean": float(hit.mean()) if hit.size else 0.0,
            "consistency_mean": float(consistency.mean()) if consistency.size else 0.0,
            "score_mean": float(score.mean()) if score.size else 0.0,
            "score_p90": float(np.percentile(score, 90)) if score.size else 0.0,
            "validated_ratio": float((score_map > 0.05).mean()),
        }
        manifest_frames.append(frame)
        all_scores.append(score)
        all_cons.append(consistency)
        all_hits.append(hit)
        print(
            f"[cave-hf-v0] {view_index + 1}/{len(train_cameras)} {name} "
            f"cand={ids.shape[0]} hit={frame['hit_mean']:.4f} "
            f"cons={frame['consistency_mean']:.4f} score={frame['score_mean']:.4f}",
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
        "scene_root": str(scene_root),
        "model_dir": str(model_dir),
        "point_cloud_ply": str(point_cloud_ply),
        "sr_dir": str(Path(args.sr_dir).expanduser().resolve()),
        "anchor_dir": str(Path(args.anchor_dir).expanduser().resolve()),
        "edge_mask_dir": str(Path(args.edge_mask_dir).expanduser().resolve()) if str(args.edge_mask_dir).strip() else "",
        "num_views": int(len(train_cameras)),
        "num_gaussians": int(cloud.xyz.shape[0]),
        "score_mean": _cat_mean(all_scores),
        "consistency_mean": _cat_mean(all_cons),
        "hit_mean": _cat_mean(all_hits),
        "frames": manifest_frames,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[cave-hf-v0] manifest: {manifest_path}")
    print(json.dumps({k: summary[k] for k in ("score_mean", "consistency_mean", "hit_mean", "num_views")}, indent=2))


if __name__ == "__main__":
    main()
