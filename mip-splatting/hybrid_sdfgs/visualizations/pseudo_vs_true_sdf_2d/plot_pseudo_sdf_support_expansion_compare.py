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
    base = sd_box(points, center=np.array([0.0, 0.0], dtype=np.float32), half_size=np.array([0.72, 0.50], dtype=np.float32))
    bulge_r = sd_circle(points, center=np.array([0.42, 0.28], dtype=np.float32), radius=0.34)
    bulge_l = sd_circle(points, center=np.array([-0.52, 0.34], dtype=np.float32), radius=0.22)
    shape = np.minimum(np.minimum(base, bulge_r), bulge_l)
    notch = sd_box(points, center=np.array([0.34, -0.02], dtype=np.float32), half_size=np.array([0.22, 0.18], dtype=np.float32))
    hole = sd_circle(points, center=np.array([-0.18, -0.04], dtype=np.float32), radius=0.17)
    shape = np.maximum(shape, -notch)
    shape = np.maximum(shape, -hole)
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


def build_sparse_anchors(
    contour_points: np.ndarray,
    n_anchors: int,
    noise_deg: float,
    sparse_box,
    sparse_keep: float,
):
    idx = np.linspace(0, contour_points.shape[0] - 1, n_anchors, dtype=np.int64)
    anchors = contour_points[idx]
    normals = estimate_normals(anchors)

    keep = np.ones((anchors.shape[0],), dtype=bool)
    gap1 = np.logical_and(anchors[:, 0] > 0.25, anchors[:, 1] > 0.10)
    gap2 = np.logical_and.reduce((anchors[:, 0] > 0.05, anchors[:, 0] < 0.55, anchors[:, 1] < -0.05))
    keep[np.logical_or(gap1, gap2)] = False

    xmin, xmax, ymin, ymax = sparse_box
    sparse_region = np.logical_and.reduce(
        (anchors[:, 0] >= xmin, anchors[:, 0] <= xmax, anchors[:, 1] >= ymin, anchors[:, 1] <= ymax)
    )
    rand_keep = np.random.uniform(0.0, 1.0, size=(anchors.shape[0],)) < max(0.0, min(1.0, sparse_keep))
    keep = np.logical_and(keep, np.logical_or(~sparse_region, rand_keep))

    sparse_region_after = sparse_region[keep]
    anchors = anchors[keep]
    normals = normals[keep]

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
    k_neighbors: int,
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
    parser = argparse.ArgumentParser(description="Compare pseudo-SDF with and without support expansion in sparse regions.")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--out_name", type=str, default="pseudo_sdf_support_expansion_compare.png")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--grid_res", type=int, default=380)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--n_anchors", type=int, default=170)
    parser.add_argument("--noise_deg", type=float, default=12.0)
    parser.add_argument("--sparse_keep", type=float, default=0.20)
    parser.add_argument("--support_scale_sparse", type=float, default=2.8)
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
    anchors, normals, sigma, conf, sparse_region = build_sparse_anchors(
        contour_points=contour_pts,
        n_anchors=args.n_anchors,
        noise_deg=args.noise_deg,
        sparse_box=sparse_box,
        sparse_keep=args.sparse_keep,
    )

    # Baseline.
    pseudo_base = pseudo_sdf_from_anchors(pts, anchors, normals, sigma, conf, args.k_neighbors).reshape(n, n)
    err_base = np.abs(pseudo_base - true_field)

    # Expanded support in sparse region.
    sigma_exp = sigma.copy()
    sigma_exp[sparse_region] *= float(args.support_scale_sparse)
    # Slight confidence attenuation to reduce over-smearing.
    conf_exp = conf.copy()
    conf_exp[sparse_region] *= 0.9
    pseudo_exp = pseudo_sdf_from_anchors(pts, anchors, normals, sigma_exp, conf_exp, args.k_neighbors).reshape(n, n)
    err_exp = np.abs(pseudo_exp - true_field)

    sparse_mask = np.logical_and.reduce(
        (xx >= sparse_box[0], xx <= sparse_box[1], yy >= sparse_box[2], yy <= sparse_box[3])
    )
    mae_base_global = float(err_base.mean())
    mae_exp_global = float(err_exp.mean())
    mae_base_sparse = float(err_base[sparse_mask].mean())
    mae_exp_sparse = float(err_exp[sparse_mask].mean())

    vmax = np.percentile(np.abs(true_field), 95)
    err_vmax = np.percentile(np.maximum(err_base, err_exp), 99)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    im0 = ax[0, 0].imshow(true_field, origin="lower", extent=[lo, hi, lo, hi], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 0].contour(xx, yy, true_field, levels=[0.0], colors="k", linewidths=1.4)
    ax[0, 0].set_title("True SDF")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(pseudo_base, origin="lower", extent=[lo, hi, lo, hi], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 1].contour(xx, yy, pseudo_base, levels=[0.0], colors="k", linewidths=1.4)
    ax[0, 1].set_title("Pseudo-SDF (baseline sparse)")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(pseudo_exp, origin="lower", extent=[lo, hi, lo, hi], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 2].contour(xx, yy, pseudo_exp, levels=[0.0], colors="k", linewidths=1.4)
    ax[0, 2].set_title(f"Pseudo-SDF (support x{args.support_scale_sparse:.1f})")
    fig.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(err_base, origin="lower", extent=[lo, hi, lo, hi], cmap="magma", vmin=0.0, vmax=err_vmax)
    ax[1, 0].contour(xx, yy, true_field, levels=[0.0], colors="cyan", linewidths=1.0)
    ax[1, 0].set_title(f"|Err| baseline\n(global {mae_base_global:.4f}, sparse {mae_base_sparse:.4f})")
    fig.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(err_exp, origin="lower", extent=[lo, hi, lo, hi], cmap="magma", vmin=0.0, vmax=err_vmax)
    ax[1, 1].contour(xx, yy, true_field, levels=[0.0], colors="cyan", linewidths=1.0)
    ax[1, 1].set_title(f"|Err| support expanded\n(global {mae_exp_global:.4f}, sparse {mae_exp_sparse:.4f})")
    fig.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im5 = ax[1, 2].imshow(err_exp - err_base, origin="lower", extent=[lo, hi, lo, hi], cmap="bwr", vmin=-err_vmax * 0.6, vmax=err_vmax * 0.6)
    ax[1, 2].contour(xx, yy, true_field, levels=[0.0], colors="k", linewidths=0.8)
    ax[1, 2].set_title("ΔErr = expanded - baseline\n(blue better, red worse)")
    fig.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

    for a in ax.reshape(-1):
        a.set_aspect("equal")
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.plot([sparse_box[0], sparse_box[1], sparse_box[1], sparse_box[0], sparse_box[0]],
               [sparse_box[2], sparse_box[2], sparse_box[3], sparse_box[3], sparse_box[2]],
               "w--", linewidth=0.9, alpha=0.9)

    ax[0, 0].scatter(anchors[~sparse_region, 0], anchors[~sparse_region, 1], s=9, c="yellow", edgecolors="k", linewidths=0.2, label="normal")
    if np.any(sparse_region):
        ax[0, 0].scatter(anchors[sparse_region, 0], anchors[sparse_region, 1], s=11, c="orange", edgecolors="k", linewidths=0.2, label="sparse-region anchors")
    ax[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    out_png = os.path.join(out_dir, args.out_name)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[ok] saved: {out_png}")
    print(
        "[metrics] "
        f"baseline_mae_global={mae_base_global:.6f} "
        f"expanded_mae_global={mae_exp_global:.6f} "
        f"baseline_mae_sparse={mae_base_sparse:.6f} "
        f"expanded_mae_sparse={mae_exp_sparse:.6f}"
    )


if __name__ == "__main__":
    main()
