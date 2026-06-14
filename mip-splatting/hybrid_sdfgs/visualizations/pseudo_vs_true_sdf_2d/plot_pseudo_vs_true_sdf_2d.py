#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def circle_sdf(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points - center[None, :], axis=-1) - radius


def build_noisy_anchors(
    n_anchors: int = 56,
    center: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
    radius: float = 0.75,
    missing_sector=(0.45, 1.25),
    noise_deg: float = 10.0,
):
    angles = np.linspace(-np.pi, np.pi, n_anchors, endpoint=False, dtype=np.float32)
    keep = np.logical_or(angles < missing_sector[0], angles > missing_sector[1])
    angles = angles[keep]

    anchors = np.stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)],
        axis=1,
    ).astype(np.float32)

    normals = anchors - center[None, :]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    if noise_deg > 0:
        noise = np.deg2rad(np.random.uniform(-noise_deg, noise_deg, size=(normals.shape[0],)))
        c = np.cos(noise)
        s = np.sin(noise)
        r11 = c
        r12 = -s
        r21 = s
        r22 = c
        nx = normals[:, 0]
        ny = normals[:, 1]
        normals = np.stack([r11 * nx + r12 * ny, r21 * nx + r22 * ny], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    # Use mildly varying local scale/confidence to mimic GS anchor uncertainty.
    sigma = np.random.uniform(0.08, 0.18, size=(anchors.shape[0],)).astype(np.float32)
    conf = np.random.uniform(0.6, 1.0, size=(anchors.shape[0],)).astype(np.float32)
    return anchors, normals, sigma, conf


def pseudo_sdf_from_anchors(
    points: np.ndarray,
    anchors: np.ndarray,
    normals: np.ndarray,
    sigma: np.ndarray,
    conf: np.ndarray,
    k_neighbors: int = 8,
) -> np.ndarray:
    n_points = points.shape[0]
    k = max(1, min(k_neighbors, anchors.shape[0]))

    # Distances: [P, A]
    d = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=-1)
    knn_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]

    rows = np.arange(n_points)[:, None]
    knn_d = d[rows, knn_idx]  # [P, K]
    nbr_xyz = anchors[knn_idx]  # [P, K, 2]
    nbr_n = normals[knn_idx]  # [P, K, 2]
    nbr_sigma = np.clip(sigma[knn_idx], 1e-4, None)  # [P, K]
    nbr_conf = np.clip(conf[knn_idx], 0.0, None)  # [P, K]

    # Align local normals to nearest-neighbor normal (same trick as runtime code).
    ref = nbr_n[:, :1, :]
    align = np.sign(np.sum(nbr_n * ref, axis=-1, keepdims=True))
    align[align == 0] = 1.0
    nbr_n = nbr_n * align

    delta = points[:, None, :] - nbr_xyz
    signed_plane = np.sum(delta * nbr_n, axis=-1)  # [P, K]

    w = np.exp(-knn_d / nbr_sigma) * nbr_conf
    w_sum = np.clip(np.sum(w, axis=1, keepdims=True), 1e-8, None)
    sdf = np.sum(w * signed_plane, axis=1, keepdims=True) / w_sum
    return sdf[:, 0]


def main():
    parser = argparse.ArgumentParser(description="Visualize pseudo-SDF vs true SDF in 2D.")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid_res", type=int, default=360)
    parser.add_argument("--k_neighbors", type=int, default=8)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    center = np.array([0.0, 0.0], dtype=np.float32)
    radius = 0.75
    lo, hi = -1.35, 1.35
    n = int(args.grid_res)

    xs = np.linspace(lo, hi, n, dtype=np.float32)
    ys = np.linspace(lo, hi, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    true_sdf = circle_sdf(pts, center=center, radius=radius)
    anchors, normals, sigma, conf = build_noisy_anchors(center=center, radius=radius)
    pseudo_sdf = pseudo_sdf_from_anchors(
        pts,
        anchors=anchors,
        normals=normals,
        sigma=sigma,
        conf=conf,
        k_neighbors=args.k_neighbors,
    )

    true_field = true_sdf.reshape(n, n)
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
    ax[0, 0].set_title("True SDF (circle)")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(
        pseudo_field, origin="lower", extent=[lo, hi, lo, hi], cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    ax[0, 1].contour(xx, yy, pseudo_field, levels=[0.0], colors="k", linewidths=1.6)
    ax[0, 1].quiver(
        anchors[:, 0], anchors[:, 1], normals[:, 0], normals[:, 1],
        color="lime", scale=18, width=0.004, headwidth=3, headlength=4
    )
    ax[0, 1].scatter(anchors[:, 0], anchors[:, 1], s=12, c="yellow", edgecolors="k", linewidths=0.2)
    ax[0, 1].set_title("Pseudo-SDF from GS-like anchors")
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

    out_png = os.path.join(out_dir, "pseudo_vs_true_sdf_2d.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"[ok] saved: {out_png}")


if __name__ == "__main__":
    main()
