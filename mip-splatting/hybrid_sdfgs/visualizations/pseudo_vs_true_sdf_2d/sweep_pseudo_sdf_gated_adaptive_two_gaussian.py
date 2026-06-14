#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def circle_sdf(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points - center[None, :], axis=-1) - radius


def parse_float_list(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty list.")
    return np.array(vals, dtype=np.float32)


def simulate_two_gaussian_case(
    sep_deg: float,
    noise_deg: float,
    outlier_deg: float,
    seed: int,
    sigma_n: float,
    k_neighbors: int,
    gate_tau: float,
    gate_beta: float,
    adaptive_alpha: float,
    adaptive_floor: float,
):
    rng = np.random.default_rng(seed)
    center = np.array([0.0, 0.0], dtype=np.float32)
    radius = 1.0

    # Two anchors on top arc: separation controls center distance.
    th0 = np.pi / 2.0
    half = np.deg2rad(sep_deg * 0.5)
    th = np.array([th0 - half, th0 + half], dtype=np.float32)
    anchors = np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32)
    normals = anchors - center[None, :]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    # Independent normal noise.
    noise = np.deg2rad(rng.uniform(-noise_deg, noise_deg, size=(2,))).astype(np.float32)
    c = np.cos(noise)
    s = np.sin(noise)
    nx = normals[:, 0].copy()
    ny = normals[:, 1].copy()
    normals = np.stack([c * nx - s * ny, s * nx + c * ny], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    # One-anchor adversarial tilt.
    if outlier_deg != 0.0:
        a = np.deg2rad(outlier_deg)
        ca, sa = np.cos(a), np.sin(a)
        n = normals[1].copy()
        normals[1] = np.array([ca * n[0] - sa * n[1], sa * n[0] + ca * n[1]], dtype=np.float32)
        normals[1] /= np.linalg.norm(normals[1]) + 1e-12

    # Evaluate on a local top-arc ROI.
    xs = np.linspace(-0.75, 0.75, 220, dtype=np.float32)
    ys = np.linspace(0.20, 1.35, 210, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    gt = circle_sdf(pts, center=center, radius=radius)

    diff = pts[:, None, :] - anchors[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    signed_plane = np.sum(diff * normals[None, :, :], axis=-1)

    # Baseline isotropic.
    sigma_n_arr = np.full((2,), sigma_n, dtype=np.float32)
    w_base = np.exp(-dist / sigma_n_arr[None, :])
    pred_base = np.sum(w_base * signed_plane, axis=1) / (np.sum(w_base, axis=1) + 1e-8)

    # Gated + adaptive tangential expansion.
    # Reliability from pair normal agreement.
    rel = float(np.abs(np.dot(normals[0], normals[1])))
    rel_norm = float(np.clip((rel - adaptive_floor) / max(1e-6, 1.0 - adaptive_floor), 0.0, 1.0))
    tangential_scale = 1.0 + adaptive_alpha * rel_norm

    sigma_t_arr = np.full((2,), sigma_n, dtype=np.float32)
    sigma_t_arr *= tangential_scale

    # Ref normal is nearest anchor normal for each query.
    ref_idx = np.argmin(dist, axis=1)
    ref_n = normals[ref_idx]
    cos_ref = np.clip(np.abs(np.sum(normals[None, :, :] * ref_n[:, None, :], axis=-1)), 0.0, 1.0)
    gate = (cos_ref >= gate_tau).astype(np.float32) * np.exp(-gate_beta * (1.0 - cos_ref))

    tangent = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    d_n = np.abs(signed_plane)
    d_t = np.sum(diff * tangent[None, :, :], axis=-1)
    normed = np.sqrt((d_n / sigma_n_arr[None, :]) ** 2 + (d_t / sigma_t_arr[None, :]) ** 2)
    w_gated = np.exp(-normed) * gate
    pred_gated = np.sum(w_gated * signed_plane, axis=1) / (np.sum(w_gated, axis=1) + 1e-8)

    err_base = np.mean(np.abs(pred_base - gt))
    err_gated = np.mean(np.abs(pred_gated - gt))
    return float(err_base), float(err_gated), float(err_base - err_gated), tangential_scale, rel


def sweep_grid(
    sep_grid: Iterable[float],
    other_grid: Iterable[float],
    mode: str,
    seeds: int,
    sigma_n: float,
    k_neighbors: int,
    gate_tau: float,
    gate_beta: float,
    adaptive_alpha: float,
    adaptive_floor: float,
):
    sep_vals = np.array(list(sep_grid), dtype=np.float32)
    oth_vals = np.array(list(other_grid), dtype=np.float32)
    mean_delta = np.zeros((len(oth_vals), len(sep_vals)), dtype=np.float32)
    win_rate = np.zeros((len(oth_vals), len(sep_vals)), dtype=np.float32)
    mean_base = np.zeros((len(oth_vals), len(sep_vals)), dtype=np.float32)
    mean_gated = np.zeros((len(oth_vals), len(sep_vals)), dtype=np.float32)

    for i, oth in enumerate(oth_vals):
        for j, sep in enumerate(sep_vals):
            deltas = []
            base_errs = []
            gated_errs = []
            wins = 0
            for s in range(seeds):
                if mode == "noise":
                    noise_deg = float(oth)
                    outlier_deg = 0.0
                elif mode == "outlier":
                    noise_deg = 8.0
                    outlier_deg = float(oth)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                eb, eg, d, _, _ = simulate_two_gaussian_case(
                    sep_deg=float(sep),
                    noise_deg=noise_deg,
                    outlier_deg=outlier_deg,
                    seed=s,
                    sigma_n=sigma_n,
                    k_neighbors=k_neighbors,
                    gate_tau=gate_tau,
                    gate_beta=gate_beta,
                    adaptive_alpha=adaptive_alpha,
                    adaptive_floor=adaptive_floor,
                )
                deltas.append(d)
                base_errs.append(eb)
                gated_errs.append(eg)
                if d > 0.0:
                    wins += 1
            mean_delta[i, j] = float(np.mean(deltas))
            win_rate[i, j] = float(wins / seeds)
            mean_base[i, j] = float(np.mean(base_errs))
            mean_gated[i, j] = float(np.mean(gated_errs))
    return sep_vals, oth_vals, mean_delta, win_rate, mean_base, mean_gated


def save_csv(path: str, sep_vals, oth_vals, data, row_name: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_name] + [f"sep_{s:.1f}" for s in sep_vals])
        for i, v in enumerate(oth_vals):
            w.writerow([f"{v:.2f}"] + [f"{x:.6f}" for x in data[i]])


def main():
    parser = argparse.ArgumentParser(description="Sweep where gated+adaptive pseudo-SDF beats baseline for two-Gaussian setup.")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--out_name", type=str, default="two_gaussian_sweep_gated_adaptive.png")
    parser.add_argument("--sep_deg_list", type=str, default="8,12,16,20,24,30,36,44,54,66")
    parser.add_argument("--noise_deg_list", type=str, default="0,4,8,12,16,20,24")
    parser.add_argument("--outlier_deg_list", type=str, default="0,5,10,15,20,25,30")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--sigma_n", type=float, default=0.16)
    parser.add_argument("--k_neighbors", type=int, default=2)
    parser.add_argument("--gate_tau", type=float, default=0.86)
    parser.add_argument("--gate_beta", type=float, default=4.0)
    parser.add_argument("--adaptive_alpha", type=float, default=2.2)
    parser.add_argument("--adaptive_floor", type=float, default=0.72)
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    sep_vals = parse_float_list(args.sep_deg_list)
    noise_vals = parse_float_list(args.noise_deg_list)
    outlier_vals = parse_float_list(args.outlier_deg_list)

    sep_n, oth_n, md_n, wr_n, mb_n, mg_n = sweep_grid(
        sep_vals,
        noise_vals,
        mode="noise",
        seeds=args.seeds,
        sigma_n=args.sigma_n,
        k_neighbors=args.k_neighbors,
        gate_tau=args.gate_tau,
        gate_beta=args.gate_beta,
        adaptive_alpha=args.adaptive_alpha,
        adaptive_floor=args.adaptive_floor,
    )
    sep_o, oth_o, md_o, wr_o, mb_o, mg_o = sweep_grid(
        sep_vals,
        outlier_vals,
        mode="outlier",
        seeds=args.seeds,
        sigma_n=args.sigma_n,
        k_neighbors=args.k_neighbors,
        gate_tau=args.gate_tau,
        gate_beta=args.gate_beta,
        adaptive_alpha=args.adaptive_alpha,
        adaptive_floor=args.adaptive_floor,
    )

    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    vmax = max(np.max(np.abs(md_n)), np.max(np.abs(md_o)), 1e-6)

    im0 = ax[0, 0].imshow(md_n, origin="lower", aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax[0, 0].set_title("Mean ΔMAE (baseline - gated), noise sweep")
    ax[0, 0].set_xlabel("Gaussian center distance (sep deg)")
    ax[0, 0].set_ylabel("Normal noise (deg)")
    ax[0, 0].set_xticks(np.arange(len(sep_n)))
    ax[0, 0].set_xticklabels([f"{x:.0f}" for x in sep_n], rotation=45)
    ax[0, 0].set_yticks(np.arange(len(oth_n)))
    ax[0, 0].set_yticklabels([f"{x:.0f}" for x in oth_n])
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(wr_n, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax[0, 1].set_title("Win rate P(ΔMAE>0), noise sweep")
    ax[0, 1].set_xlabel("Gaussian center distance (sep deg)")
    ax[0, 1].set_ylabel("Normal noise (deg)")
    ax[0, 1].set_xticks(np.arange(len(sep_n)))
    ax[0, 1].set_xticklabels([f"{x:.0f}" for x in sep_n], rotation=45)
    ax[0, 1].set_yticks(np.arange(len(oth_n)))
    ax[0, 1].set_yticklabels([f"{x:.0f}" for x in oth_n])
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[1, 0].imshow(md_o, origin="lower", aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax[1, 0].set_title("Mean ΔMAE (baseline - gated), outlier sweep")
    ax[1, 0].set_xlabel("Gaussian center distance (sep deg)")
    ax[1, 0].set_ylabel("Single-anchor outlier tilt (deg)")
    ax[1, 0].set_xticks(np.arange(len(sep_o)))
    ax[1, 0].set_xticklabels([f"{x:.0f}" for x in sep_o], rotation=45)
    ax[1, 0].set_yticks(np.arange(len(oth_o)))
    ax[1, 0].set_yticklabels([f"{x:.0f}" for x in oth_o])
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im3 = ax[1, 1].imshow(wr_o, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax[1, 1].set_title("Win rate P(ΔMAE>0), outlier sweep")
    ax[1, 1].set_xlabel("Gaussian center distance (sep deg)")
    ax[1, 1].set_ylabel("Single-anchor outlier tilt (deg)")
    ax[1, 1].set_xticks(np.arange(len(sep_o)))
    ax[1, 1].set_xticklabels([f"{x:.0f}" for x in sep_o], rotation=45)
    ax[1, 1].set_yticks(np.arange(len(oth_o)))
    ax[1, 1].set_yticklabels([f"{x:.0f}" for x in oth_o])
    fig.colorbar(im3, ax=ax[1, 1], fraction=0.046, pad=0.04)

    out_png = os.path.join(out_dir, args.out_name)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    save_csv(os.path.join(out_dir, "two_gaussian_noise_mean_delta.csv"), sep_n, oth_n, md_n, "noise_deg")
    save_csv(os.path.join(out_dir, "two_gaussian_noise_win_rate.csv"), sep_n, oth_n, wr_n, "noise_deg")
    save_csv(os.path.join(out_dir, "two_gaussian_outlier_mean_delta.csv"), sep_o, oth_o, md_o, "outlier_deg")
    save_csv(os.path.join(out_dir, "two_gaussian_outlier_win_rate.csv"), sep_o, oth_o, wr_o, "outlier_deg")
    save_csv(os.path.join(out_dir, "two_gaussian_noise_mean_base.csv"), sep_n, oth_n, mb_n, "noise_deg")
    save_csv(os.path.join(out_dir, "two_gaussian_noise_mean_gated.csv"), sep_n, oth_n, mg_n, "noise_deg")

    n_pos_noise = int(np.sum(md_n > 0))
    n_pos_outlier = int(np.sum(md_o > 0))
    print(f"[ok] saved: {out_png}")
    print(
        "[summary] "
        f"noise_cells_positive={n_pos_noise}/{md_n.size} "
        f"outlier_cells_positive={n_pos_outlier}/{md_o.size}"
    )
    print(
        "[params] "
        f"seeds={args.seeds} sigma_n={args.sigma_n} "
        f"gate_tau={args.gate_tau} gate_beta={args.gate_beta} "
        f"adaptive_alpha={args.adaptive_alpha} adaptive_floor={args.adaptive_floor}"
    )


if __name__ == "__main__":
    main()
