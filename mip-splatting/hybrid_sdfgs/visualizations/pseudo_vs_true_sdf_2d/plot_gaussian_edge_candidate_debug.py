#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import numpy as np
import trimesh


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def normalize(v: np.ndarray) -> np.ndarray:
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def render_projection(
    p0: np.ndarray,
    p1: np.ndarray,
    depth: np.ndarray,
    w: int,
    h: int,
    pad: float = 0.08,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    x0, x1 = float(np.min(p0)), float(np.max(p0))
    y0, y1 = float(np.min(p1)), float(np.max(p1))
    xr = x1 - x0
    yr = y1 - y0
    x0 -= pad * xr
    x1 += pad * xr
    y0 -= pad * yr
    y1 += pad * yr

    ix = np.floor((p0 - x0) / max(1e-8, x1 - x0) * (w - 1)).astype(np.int32)
    iy = np.floor((p1 - y0) / max(1e-8, y1 - y0) * (h - 1)).astype(np.int32)
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    cnt = np.zeros((h, w), dtype=np.float32)
    zsum = np.zeros((h, w), dtype=np.float32)
    for x, y, z in zip(ix, iy, depth):
        cnt[h - 1 - y, x] += 1.0
        zsum[h - 1 - y, x] += z

    zmean = np.zeros_like(cnt)
    nz = cnt > 0
    zmean[nz] = zsum[nz] / cnt[nz]

    dens = np.log1p(cnt)
    dens_n = normalize(dens)
    z_n = normalize(zmean)

    # Soft blue-red depth tint with density modulation.
    base = np.stack(
        [
            45 + 200 * (0.25 + 0.75 * z_n),
            50 + 180 * (0.25 + 0.75 * dens_n),
            70 + 185 * (1.0 - 0.6 * z_n),
        ],
        axis=-1,
    )
    base = base * (0.20 + 0.80 * np.expand_dims(dens_n, axis=-1))
    bg = np.full((h, w, 3), 246.0, dtype=np.float32)
    img = np.where(np.expand_dims(nz, axis=-1), base, bg)
    return to_uint8(img), (x0, x1, y0, y1)


def draw_cross(img: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    for t in range(-size, size + 1):
        xx = x + t
        yy = y
        if 0 <= xx < w and 0 <= yy < h:
            img[yy, xx] = color
        xx = x
        yy = y + t
        if 0 <= xx < w and 0 <= yy < h:
            img[yy, xx] = color


def draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], t: int = 2) -> None:
    h, w = img.shape[:2]
    x0, x1 = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
    for k in range(-t, t + 1):
        y_top = np.clip(y0 + k, 0, h - 1)
        y_bot = np.clip(y1 + k, 0, h - 1)
        img[y_top, x0 : x1 + 1] = color
        img[y_bot, x0 : x1 + 1] = color
        x_l = np.clip(x0 + k, 0, w - 1)
        x_r = np.clip(x1 + k, 0, w - 1)
        img[y0 : y1 + 1, x_l] = color
        img[y0 : y1 + 1, x_r] = color


def world_to_px(val0: float, val1: float, bounds: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int]:
    x0, x1, y0, y1 = bounds
    x = int(np.clip((val0 - x0) / max(1e-8, x1 - x0) * (w - 1), 0, w - 1))
    y = int(np.clip((val1 - y0) / max(1e-8, y1 - y0) * (h - 1), 0, h - 1))
    # image y-down
    return x, (h - 1 - y)


def write_ppm(path: str, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate a clear Lego surface-edge candidate on good_fused point cloud.")
    parser.add_argument("--gaussian_ply_path", type=str, default="/Users/ltl/Desktop/codex_playground/good_fused.ply")
    parser.add_argument("--out_ppm", type=str, default="gaussian_edge_candidate_debug.ppm")
    parser.add_argument("--alpha_q", type=float, default=0.35)
    parser.add_argument("--panel_size", type=int, default=780)
    parser.add_argument("--edge_center", type=str, default="0.37,-0.62,0.60")
    parser.add_argument("--crop_extent", type=str, default="0.32,0.30,0.26")
    args = parser.parse_args()

    edge_center = np.array([float(x) for x in args.edge_center.split(",")], dtype=np.float32)
    crop_extent = np.array([float(x) for x in args.crop_extent.split(",")], dtype=np.float32)
    if edge_center.shape[0] != 3 or crop_extent.shape[0] != 3:
        raise ValueError("edge_center and crop_extent must be 3 values.")

    obj = trimesh.load(args.gaussian_ply_path)
    raw = obj.metadata.get("_ply_raw", {}).get("vertex", {}).get("data", None)
    if raw is None:
        raise RuntimeError("Failed to read structured PLY vertex data.")

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    alpha = sigmoid(raw["opacity"].astype(np.float32))
    keep = alpha >= float(np.quantile(alpha, args.alpha_q))
    xyz = xyz[keep]

    w = h = int(args.panel_size)

    # XY uses z as depth
    img_xy, b_xy = render_projection(xyz[:, 0], xyz[:, 1], xyz[:, 2], w, h)
    # XZ uses y as depth
    img_xz, b_xz = render_projection(xyz[:, 0], xyz[:, 2], xyz[:, 1], w, h)
    # YZ uses x as depth
    img_yz, b_yz = render_projection(xyz[:, 1], xyz[:, 2], xyz[:, 0], w, h)

    ex, ey, ez = float(edge_center[0]), float(edge_center[1]), float(edge_center[2])
    hx, hy, hz = (crop_extent * 0.5).tolist()

    c_xy = world_to_px(ex, ey, b_xy, w, h)
    c_xz = world_to_px(ex, ez, b_xz, w, h)
    c_yz = world_to_px(ey, ez, b_yz, w, h)

    # draw center crosses
    for im, c in ((img_xy, c_xy), (img_xz, c_xz), (img_yz, c_yz)):
        draw_cross(im, c[0], c[1], size=12, color=(255, 20, 20))

    # draw crop extents on all projections
    r_xy0 = world_to_px(ex - hx, ey - hy, b_xy, w, h)
    r_xy1 = world_to_px(ex + hx, ey + hy, b_xy, w, h)
    draw_rect(img_xy, r_xy0[0], r_xy0[1], r_xy1[0], r_xy1[1], (0, 255, 255), t=2)

    r_xz0 = world_to_px(ex - hx, ez - hz, b_xz, w, h)
    r_xz1 = world_to_px(ex + hx, ez + hz, b_xz, w, h)
    draw_rect(img_xz, r_xz0[0], r_xz0[1], r_xz1[0], r_xz1[1], (0, 255, 255), t=2)

    r_yz0 = world_to_px(ey - hy, ez - hz, b_yz, w, h)
    r_yz1 = world_to_px(ey + hy, ez + hz, b_yz, w, h)
    draw_rect(img_yz, r_yz0[0], r_yz0[1], r_yz1[0], r_yz1[1], (0, 255, 255), t=2)

    sep = np.full((h, 22, 3), 245, dtype=np.uint8)
    out = np.concatenate([img_xy, sep, img_xz, sep, img_yz], axis=1)
    out_path = os.path.abspath(os.path.expanduser(args.out_ppm))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_ppm(out_path, out)

    print(f"[ok] wrote ppm: {out_path}")
    print(f"[edge_center] x={ex:.3f}, y={ey:.3f}, z={ez:.3f}")
    print(f"[crop_extent] dx={crop_extent[0]:.3f}, dy={crop_extent[1]:.3f}, dz={crop_extent[2]:.3f}")


if __name__ == "__main__":
    main()
