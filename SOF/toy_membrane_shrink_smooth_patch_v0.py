#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


_CACHE_DIR = Path("/tmp/codex_mpl_cache_membrane_v0")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_patch_masks(height=220, width=320):
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    xs = np.linspace(-1.25, 1.25, width, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    patch = (np.abs(grid_x) <= 0.95) & (np.abs(grid_y) <= 0.42)

    hole_centers = [(-0.38, 0.08), (0.00, 0.08), (0.38, 0.08)]
    hole_radius = 0.095
    holes = np.zeros_like(patch, dtype=bool)
    for cx, cy in hole_centers:
        holes |= (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= hole_radius ** 2

    patch &= ~holes

    boundary_band = patch & ~ndi.binary_erosion(patch, iterations=3)
    pull_mask = ndi.binary_erosion(patch, iterations=14)
    pull_mask &= ~(patch & ~ndi.binary_erosion(patch, iterations=9))

    return xs, ys, patch, boundary_band, pull_mask, holes


def solve_height_field(
    patch,
    anchor_mask,
    pull_mask,
    lambda_anchor,
    lambda_pull,
    lambda_smooth,
    lambda_pressure,
):
    patch = np.asarray(patch, dtype=bool)
    anchor_mask = np.asarray(anchor_mask, dtype=bool) & patch
    pull_mask = np.asarray(pull_mask, dtype=bool) & patch

    coords = np.argwhere(patch)
    n = int(coords.shape[0])
    index_map = -np.ones(patch.shape, dtype=np.int64)
    index_map[patch] = np.arange(n, dtype=np.int64)

    matrix = lil_matrix((n, n), dtype=np.float64)
    rhs = np.full((n,), -float(lambda_pressure), dtype=np.float64)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    h, w = patch.shape
    for row, col in coords:
        idx = int(index_map[row, col])
        degree = 0
        for dr, dc in neighbors:
            nr = int(row + dr)
            nc = int(col + dc)
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not patch[nr, nc]:
                continue
            jdx = int(index_map[nr, nc])
            matrix[idx, jdx] = matrix[idx, jdx] - float(lambda_smooth)
            degree += 1

        diag = float(lambda_smooth) * float(degree)
        if anchor_mask[row, col]:
            diag += float(lambda_anchor)
        if pull_mask[row, col]:
            diag += float(lambda_pull)
        matrix[idx, idx] = matrix[idx, idx] + diag

    solution = spsolve(matrix.tocsr(), rhs)
    field = np.full(patch.shape, np.nan, dtype=np.float64)
    field[patch] = solution
    return field


def center_cross_section(field):
    center_row = field.shape[0] // 2
    return field[center_row]


def plot_masks(ax, patch, anchor_mask, pull_mask, holes):
    rgb = np.zeros((*patch.shape, 3), dtype=np.float64)
    rgb[patch] = np.array([0.18, 0.18, 0.22], dtype=np.float64)
    rgb[pull_mask] = np.array([0.20, 0.52, 0.88], dtype=np.float64)
    rgb[anchor_mask] = np.array([1.00, 0.78, 0.18], dtype=np.float64)
    rgb[holes] = np.array([0.03, 0.03, 0.03], dtype=np.float64)
    ax.imshow(rgb, origin="lower")
    ax.set_title("Support Layout")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_field(ax, field, patch, title, vmin, vmax):
    shown = np.where(patch, field, np.nan)
    im = ax.imshow(shown, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_cross_section(ax, xs, no_pressure, sagged, pulled):
    valid = np.isfinite(no_pressure)
    ax.plot(xs[valid], no_pressure[valid], color="#495057", lw=2.0, label="anchors only")
    ax.plot(xs[valid], sagged[valid], color="#d9480f", lw=2.0, label="with shrink bias")
    ax.plot(xs[valid], pulled[valid], color="#2b8a3e", lw=2.0, label="with interior pull")
    ax.axhline(0.0, color="#1c7ed6", lw=1.2, ls="--", label="true plane")
    ax.set_title("Center Cross Section")
    ax.set_xlabel("x")
    ax.set_ylabel("height z")
    ax.legend(loc="lower right", fontsize=8)


def plot_pull_sweep(ax, lambda_values, min_values, mean_values):
    ax.plot(lambda_values, min_values, color="#d9480f", lw=2.0, marker="o", label="min depth")
    ax.plot(lambda_values, mean_values, color="#2b8a3e", lw=2.0, marker="o", label="mean interior depth")
    ax.set_title("Pull Strength Sweep")
    ax.set_xlabel("lambda_pull")
    ax.set_ylabel("height z")
    ax.legend(loc="lower right", fontsize=8)


def save_summary(path, sagged_field, pulled_field, no_pressure_field, patch, pull_lambdas, pull_minima, pull_means):
    def stats(name, field):
        values = field[patch]
        return {
            f"{name}_min": float(np.nanmin(values)),
            f"{name}_mean": float(np.nanmean(values)),
            f"{name}_p05": float(np.nanpercentile(values, 5)),
        }

    summary = {}
    summary.update(stats("anchors_only", no_pressure_field))
    summary.update(stats("sagged", sagged_field))
    summary.update(stats("pulled", pulled_field))
    summary["pull_lambda_values"] = [float(v) for v in pull_lambdas]
    summary["pull_min_values"] = [float(v) for v in pull_minima]
    summary["pull_mean_values"] = [float(v) for v in pull_means]

    lines = []
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Toy membrane shrink on a smooth patch.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/ltl/Desktop/codex_playground/SOF/toy_outputs/membrane_shrink_smooth_patch_v0"),
    )
    parser.add_argument("--lambda-anchor", type=float, default=120.0)
    parser.add_argument("--lambda-smooth", type=float, default=6.0)
    parser.add_argument("--lambda-pressure", type=float, default=0.01)
    parser.add_argument("--lambda-pull", type=float, default=0.08)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, patch, anchor_mask, pull_mask, holes = build_patch_masks()

    no_pressure = solve_height_field(
        patch=patch,
        anchor_mask=anchor_mask,
        pull_mask=np.zeros_like(pull_mask, dtype=bool),
        lambda_anchor=float(args.lambda_anchor),
        lambda_pull=0.0,
        lambda_smooth=float(args.lambda_smooth),
        lambda_pressure=0.0,
    )

    sagged = solve_height_field(
        patch=patch,
        anchor_mask=anchor_mask,
        pull_mask=np.zeros_like(pull_mask, dtype=bool),
        lambda_anchor=float(args.lambda_anchor),
        lambda_pull=0.0,
        lambda_smooth=float(args.lambda_smooth),
        lambda_pressure=float(args.lambda_pressure),
    )

    pulled = solve_height_field(
        patch=patch,
        anchor_mask=anchor_mask,
        pull_mask=pull_mask,
        lambda_anchor=float(args.lambda_anchor),
        lambda_pull=float(args.lambda_pull),
        lambda_smooth=float(args.lambda_smooth),
        lambda_pressure=float(args.lambda_pressure),
    )

    vmin = float(np.nanmin(sagged[patch]))
    vmax = 0.01

    cross_no_pressure = center_cross_section(no_pressure)
    cross_sagged = center_cross_section(sagged)
    cross_pulled = center_cross_section(pulled)

    pull_lambdas = np.array([0.0, 0.4, 0.8, 1.2, 2.4, 4.8, 9.6], dtype=np.float64)
    pull_minima = []
    pull_means = []
    for lam in pull_lambdas:
        field = solve_height_field(
            patch=patch,
            anchor_mask=anchor_mask,
            pull_mask=pull_mask,
            lambda_anchor=float(args.lambda_anchor),
            lambda_pull=float(lam),
            lambda_smooth=float(args.lambda_smooth),
            lambda_pressure=float(args.lambda_pressure),
        )
        values = field[pull_mask]
        pull_minima.append(float(np.nanmin(values)))
        pull_means.append(float(np.nanmean(values)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    plot_masks(axes[0, 0], patch, anchor_mask, pull_mask, holes)
    im0 = plot_field(axes[0, 1], no_pressure, patch, "Anchors Only (No Shrink Bias)", vmin=vmin, vmax=vmax)
    im1 = plot_field(axes[0, 2], sagged, patch, "Thin Surface with Inward Shrink", vmin=vmin, vmax=vmax)
    plot_field(axes[1, 0], pulled, patch, "Add Interior Position Pull", vmin=vmin, vmax=vmax)
    plot_cross_section(axes[1, 1], xs, cross_no_pressure, cross_sagged, cross_pulled)
    plot_pull_sweep(axes[1, 2], pull_lambdas, pull_minima, pull_means)

    cbar = fig.colorbar(im1, ax=[axes[0, 1], axes[0, 2], axes[1, 0]], fraction=0.025, pad=0.02)
    cbar.set_label("surface height z")

    fig.savefig(output_dir / "overview.png", dpi=180)
    plt.close(fig)

    np.savez(
        output_dir / "fields.npz",
        xs=xs,
        ys=ys,
        patch=patch.astype(np.uint8),
        anchor_mask=anchor_mask.astype(np.uint8),
        pull_mask=pull_mask.astype(np.uint8),
        no_pressure=no_pressure,
        sagged=sagged,
        pulled=pulled,
        pull_lambdas=pull_lambdas,
        pull_minima=np.asarray(pull_minima, dtype=np.float64),
        pull_means=np.asarray(pull_means, dtype=np.float64),
    )

    save_summary(
        output_dir / "summary.txt",
        sagged_field=sagged,
        pulled_field=pulled,
        no_pressure_field=no_pressure,
        patch=patch,
        pull_lambdas=pull_lambdas,
        pull_minima=pull_minima,
        pull_means=pull_means,
    )


if __name__ == "__main__":
    main()
