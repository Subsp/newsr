#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def sd_circle(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points - center[None, :], axis=-1) - radius


def sd_box(points: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    p = points - center[None, :]
    q = np.abs(p) - half_size[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside + inside


def true_complex_sdf(points: np.ndarray) -> np.ndarray:
    # Base: box with sharp corners.
    base = sd_box(points, center=np.array([0.0, 0.0], dtype=np.float32), half_size=np.array([0.72, 0.50], dtype=np.float32))
    # Add curved bulges.
    bulge_r = sd_circle(points, center=np.array([0.42, 0.28], dtype=np.float32), radius=0.34)
    bulge_l = sd_circle(points, center=np.array([-0.52, 0.34], dtype=np.float32), radius=0.22)
    shape = np.minimum(np.minimum(base, bulge_r), bulge_l)
    # Subtract a rectangular notch and a circular hole.
    notch = sd_box(points, center=np.array([0.34, -0.02], dtype=np.float32), half_size=np.array([0.22, 0.18], dtype=np.float32))
    hole = sd_circle(points, center=np.array([-0.18, -0.04], dtype=np.float32), radius=0.17)
    shape = np.maximum(shape, -notch)
    shape = np.maximum(shape, -hole)
    # Cut top-right by a plane to create an extra corner feature.
    plane = points[:, 0] + 0.95 * points[:, 1] - 0.98
    shape = np.maximum(shape, plane)
    return shape


def extract_contour_points(xx: np.ndarray, yy: np.ndarray, field: np.ndarray) -> np.ndarray:
    fig = plt.figure()
    cs = plt.contour(xx, yy, field, levels=[0.0])
    verts = []
    if len(cs.allsegs) > 0:
        for seg in cs.allsegs[0]:
            if seg.shape[0] > 4:
                verts.append(seg.astype(np.float32))
    plt.close(fig)
    if not verts:
        raise RuntimeError("No zero-level contour extracted from true SDF.")
    return np.concatenate(verts, axis=0)


def estimate_normals(points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    dx = np.array([eps, 0.0], dtype=np.float32)
    dy = np.array([0.0, eps], dtype=np.float32)
    gx = true_complex_sdf(points + dx[None, :]) - true_complex_sdf(points - dx[None, :])
    gy = true_complex_sdf(points + dy[None, :]) - true_complex_sdf(points - dy[None, :])
    n = np.stack([gx, gy], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n.astype(np.float32)


def build_noisy_anchors(
    contour_points: np.ndarray,
    n_anchors: int = 170,
    noise_deg: float = 12.0,
    region_sparse_box=(-1.05, -0.05, 0.08, 1.05),
    region_sparse_keep: float = 0.25,
):
    idx = np.linspace(0, contour_points.shape[0] - 1, n_anchors, dtype=np.int64)
    anchors = contour_points[idx]
    normals = estimate_normals(anchors)

    # Remove anchors from selected regions to mimic view gaps / weak coverage.
    keep = np.ones((anchors.shape[0],), dtype=bool)
    gap1 = np.logical_and(anchors[:, 0] > 0.25, anchors[:, 1] > 0.10)  # near cut corner
    gap2 = np.logical_and.reduce((anchors[:, 0] > 0.05, anchors[:, 0] < 0.55, anchors[:, 1] < -0.05))  # near notch
    keep[np.logical_or(gap1, gap2)] = False

    # Additional local thinning: keep only a subset of anchors in one region.
    if region_sparse_keep < 0.999:
        xmin, xmax, ymin, ymax = region_sparse_box
        sparse_region = np.logical_and.reduce(
            (anchors[:, 0] >= xmin, anchors[:, 0] <= xmax, anchors[:, 1] >= ymin, anchors[:, 1] <= ymax)
        )
        rand_keep = np.random.uniform(0.0, 1.0, size=(anchors.shape[0],)) < max(0.0, min(1.0, region_sparse_keep))
        keep = np.logical_and(keep, np.logical_or(~sparse_region, rand_keep))

    sparse_region_after = np.logical_and.reduce(
        (anchors[:, 0] >= region_sparse_box[0], anchors[:, 0] <= region_sparse_box[1], anchors[:, 1] >= region_sparse_box[2], anchors[:, 1] <= region_sparse_box[3])
    )
    anchors = anchors[keep]
    normals = normals[keep]
    sparse_region_after = sparse_region_after[keep]

    if noise_deg > 0:
        noise = np.deg2rad(np.random.uniform(-noise_deg, noise_deg, size=(normals.shape[0],)))
        c = np.cos(noise)
        s = np.sin(noise)
        nx = normals[:, 0]
        ny = normals[:, 1]
        normals = np.stack([c * nx - s * ny, s * nx + c * ny], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    sigma = np.random.uniform(0.07, 0.20, size=(anchors.shape[0],)).astype(np.float32)
    conf = np.random.uniform(0.55, 1.00, size=(anchors.shape[0],)).astype(np.float32)
    return anchors.astype(np.float32), normals.astype(np.float32), sigma, conf, sparse_region_after


def pseudo_sdf_from_anchors(
    points: np.ndarray,
    anchors: np.ndarray,
    normals: np.ndarray,
    sigma: np.ndarray,
    conf: np.ndarray,
    k_neighbors: int = 10,
) -> np.ndarray:
    n_points = points.shape[0]
    k = max(1, min(k_neighbors, anchors.shape[0]))

    d = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=-1)
    knn_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n_points)[:, None]

    knn_d = d[rows, knn_idx]
    nbr_xyz = anchors[knn_idx]
    nbr_n = normals[knn_idx]
    nbr_sigma = np.clip(sigma[knn_idx], 1e-4, None)
    nbr_conf = np.clip(conf[knn_idx], 0.0, None)

    # Force local normal orientation consistency.
    ref = nbr_n[:, :1, :]
    align = np.sign(np.sum(nbr_n * ref, axis=-1, keepdims=True))
    align[align == 0] = 1.0
    nbr_n = nbr_n * align

    delta = points[:, None, :] - nbr_xyz
    signed_plane = np.sum(delta * nbr_n, axis=-1)

    w = np.exp(-knn_d / nbr_sigma) * nbr_conf
    w_sum = np.clip(np.sum(w, axis=1, keepdims=True), 1e-8, None)
    sdf = np.sum(w * signed_plane, axis=1, keepdims=True) / w_sum
    return sdf[:, 0]


def main():
    parser = argparse.ArgumentParser(description="2D comparison: pseudo-SDF vs true SDF on a complex shape.")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--grid_res", type=int, default=380)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--region_sparse_keep", type=float, default=0.25)
    parser.add_argument("--out_name", type=str, default="pseudo_vs_true_sdf_2d_complex_sparse_region.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    lo, hi = -1.35, 1.35
    n = int(args.grid_res)
    xs = np.linspace(lo, hi, n, dtype=np.float32)
    ys = np.linspace(lo, hi, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    true_sdf = true_complex_sdf(pts)
    true_field = true_sdf.reshape(n, n)

    contour_pts = extract_contour_points(xx, yy, true_field)
    sparse_box = (-1.05, -0.05, 0.08, 1.05)
    anchors, normals, sigma, conf, sparse_region = build_noisy_anchors(
        contour_pts, region_sparse_box=sparse_box, region_sparse_keep=args.region_sparse_keep
    )
    pseudo_sdf = pseudo_sdf_from_anchors(
        pts, anchors=anchors, normals=normals, sigma=sigma, conf=conf, k_neighbors=args.k_neighbors
    )
    pseudo_field = pseudo_sdf.reshape(n, n)

    abs_err = np.abs(pseudo_field - true_field)
    h = (hi - lo) / (n - 1)
    gy, gx = np.gradient(pseudo_field, h, h)
    grad_norm = np.sqrt(gx * gx + gy * gy)

    vmax = np.percentile(np.abs(true_field), 95)
    err_vmax = np.percentile(abs_err, 99)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    im0 = ax[0, 0].imshow(
        true_field, origin="lower", extent=[lo, hi, lo, hi], cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    ax[0, 0].contour(xx, yy, true_field, levels=[0.0], colors="k", linewidths=1.6)
    ax[0, 0].set_title("True SDF (complex: corners + curved parts)")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(
        pseudo_field, origin="lower", extent=[lo, hi, lo, hi], cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    ax[0, 1].contour(xx, yy, pseudo_field, levels=[0.0], colors="k", linewidths=1.6)
    ax[0, 1].quiver(
        anchors[:, 0], anchors[:, 1], normals[:, 0], normals[:, 1],
        color="lime", scale=22, width=0.0038, headwidth=3, headlength=4
    )
    ax[0, 1].scatter(anchors[~sparse_region, 0], anchors[~sparse_region, 1], s=10, c="yellow", edgecolors="k", linewidths=0.2, label="normal density")
    if np.any(sparse_region):
        ax[0, 1].scatter(anchors[sparse_region, 0], anchors[sparse_region, 1], s=12, c="orange", edgecolors="k", linewidths=0.2, label="thinned region")
    ax[0, 1].set_title("Pseudo-SDF with partial GS thinning")
    ax[0, 1].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[1, 0].imshow(
        abs_err, origin="lower", extent=[lo, hi, lo, hi], cmap="magma", vmin=0.0, vmax=err_vmax
    )
    ax[1, 0].contour(xx, yy, true_field, levels=[0.0], colors="cyan", linewidths=1.0)
    ax[1, 0].set_title("|Pseudo - True|")
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im3 = ax[1, 1].imshow(
        grad_norm, origin="lower", extent=[lo, hi, lo, hi], cmap="viridis", vmin=0.0, vmax=2.0
    )
    ax[1, 1].contour(xx, yy, pseudo_field, levels=[0.0], colors="w", linewidths=1.0)
    ax[1, 1].set_title(r"Pseudo gradient norm $\|\nabla f\|$ (ideal SDF: 1)")
    fig.colorbar(im3, ax=ax[1, 1], fraction=0.046, pad=0.04)

    for a in ax.reshape(-1):
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.set_aspect("equal")

    out_png = os.path.join(out_dir, args.out_name)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[ok] saved: {out_png}")


if __name__ == "__main__":
    main()
