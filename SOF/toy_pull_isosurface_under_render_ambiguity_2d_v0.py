#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SQRT_2PI = math.sqrt(2.0 * math.pi)


def true_surface(x):
    return 0.42 + 0.035 * np.sin(1.1 * math.pi * x - 0.25) + 0.015 * np.exp(-0.5 * ((x + 0.35) / 0.35) ** 2)


def amplitude_profile(x):
    return 8.6 + 1.3 * np.exp(-0.5 * ((x - 0.1) / 0.38) ** 2) + 0.55 * np.sin(2.0 * math.pi * x + 0.5)


def local_bump(x):
    return np.exp(-0.5 * ((x - 0.18) / 0.16) ** 2)


def render_profile(amplitude, width):
    return 1.0 - np.exp(-amplitude * width * SQRT_2PI)


def front_offset(amplitude, width, iso_level):
    safe_amp = np.maximum(amplitude, iso_level + 1e-8)
    return width * np.sqrt(2.0 * np.log(safe_amp / iso_level))


def center_field(x, width, iso_level, bump_scale):
    amp = amplitude_profile(x)
    z_true = true_surface(x)
    offset = front_offset(amp, width, iso_level)
    center_true = z_true + offset
    return center_true + bump_scale * local_bump(x), center_true, z_true, amp


def sigma_field(x_grid, z_grid, width, iso_level, bump_scale):
    center, center_true, z_true, amp = center_field(x_grid, width, iso_level, bump_scale)
    sigma = amp[:, None] * np.exp(-0.5 * ((z_grid[None, :] - center[:, None]) / width) ** 2)
    return sigma, center, center_true, z_true, amp


def front_isosurface(x, width, iso_level, bump_scale):
    center, _, z_true, amp = center_field(x, width, iso_level, bump_scale)
    offset = front_offset(amp, width, iso_level)
    return center - offset, z_true, amp


def run_pull_optimization(x, init_bump_scale, steps, lr):
    bump = local_bump(x)
    bump_sq_mean = float(np.mean(bump * bump))
    coeff = float(init_bump_scale)
    history = []
    for step in range(int(steps) + 1):
        iso_wrong = true_surface(x) + coeff * bump
        iso_error = float(np.sqrt(np.mean((iso_wrong - true_surface(x)) ** 2)))
        render_loss = 0.0
        history.append(
            {
                "step": int(step),
                "bump_scale": float(coeff),
                "iso_rmse": iso_error,
                "render_loss": render_loss,
            }
        )
        grad = 2.0 * coeff * bump_sq_mean
        coeff = coeff - float(lr) * grad
    return coeff, history


def plot_heatmap(ax, x, z, sigma, z_true, z_front, iso_level, title, front_color):
    extent = [float(x.min()), float(x.max()), float(z.min()), float(z.max())]
    im = ax.imshow(
        sigma.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
    )
    ax.plot(x, z_true, color="#1c7ed6", lw=2.0, ls="--", label="true surface")
    ax.plot(x, z_front, color=front_color, lw=2.0, label="front isosurface")
    ax.contour(x, z, sigma.T, levels=[iso_level], colors=[front_color], linewidths=1.2, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("depth z")
    ax.legend(loc="upper right", fontsize=8)
    return im


def plot_curves(ax, x, z_true, z_wrong, z_pulled, render_target, render_wrong, render_pulled):
    ax.plot(x, z_true, color="#1c7ed6", lw=2.0, ls="--", label="true surface")
    ax.plot(x, z_wrong, color="#d9480f", lw=2.0, label="wrong front iso")
    ax.plot(x, z_pulled, color="#2b8a3e", lw=2.0, label="pulled front iso")
    ax.set_xlabel("x")
    ax.set_ylabel("surface depth")
    ax.set_title("Surface curves and render profile")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, render_target, color="#495057", lw=1.8, ls="--", label="target render")
    ax2.plot(x, render_wrong, color="#f08c00", lw=1.3, alpha=0.95, label="wrong render")
    ax2.plot(x, render_pulled, color="#37b24d", lw=1.3, alpha=0.95, label="pulled render")
    ax2.set_ylabel("render proxy")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)


def plot_optimization(ax, history):
    step = np.asarray([row["step"] for row in history], dtype=np.float64)
    coeff = np.asarray([row["bump_scale"] for row in history], dtype=np.float64)
    iso_rmse = np.asarray([row["iso_rmse"] for row in history], dtype=np.float64)
    render_loss = np.asarray([row["render_loss"] for row in history], dtype=np.float64)

    ax.plot(step, coeff, color="#7b2cbf", lw=2.0, label="bump coefficient")
    ax.plot(step, iso_rmse, color="#2b8a3e", lw=2.0, label="iso RMSE")
    ax.set_xlabel("optimization step")
    ax.set_title("Manual pull optimization")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(step, render_loss, color="#d9480f", lw=1.6, label="render loss")
    ax2.set_ylabel("render loss", color="#d9480f")
    ax2.tick_params(axis="y", labelcolor="#d9480f")


def save_summary(path, init_bump_scale, final_bump_scale, init_iso_rmse, final_iso_rmse, render_diff_wrong, render_diff_pulled):
    lines = [
        f"init_bump_scale: {init_bump_scale:.8f}",
        f"final_bump_scale: {final_bump_scale:.8f}",
        f"init_iso_rmse: {init_iso_rmse:.8f}",
        f"final_iso_rmse: {final_iso_rmse:.8f}",
        f"max_render_diff_wrong: {render_diff_wrong:.8e}",
        f"max_render_diff_pulled: {render_diff_pulled:.8e}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="2D toy for pulling a wrong front isosurface under render ambiguity.")
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/ltl/Desktop/codex_playground/SOF/toy_outputs/pull_isosurface_under_render_ambiguity_2d_v0"))
    parser.add_argument("--width", type=float, default=0.085)
    parser.add_argument("--iso-level", type=float, default=0.58)
    parser.add_argument("--init-bump-scale", type=float, default=0.085)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2.2)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1.0, 1.0, 420, dtype=np.float64)
    z = np.linspace(0.05, 1.05, 420, dtype=np.float64)
    width = float(args.width)
    iso_level = float(args.iso_level)

    sigma_wrong, _, _, z_true, amp = sigma_field(x, z, width, iso_level, float(args.init_bump_scale))
    sigma_pulled, _, _, _, _ = sigma_field(x, z, width, iso_level, 0.0)
    z_wrong, _, _ = front_isosurface(x, width, iso_level, float(args.init_bump_scale))
    z_pulled, _, _ = front_isosurface(x, width, iso_level, 0.0)

    render_target = render_profile(amp, width)
    render_wrong = render_profile(amp, width)
    render_pulled = render_profile(amp, width)

    final_bump_scale, history = run_pull_optimization(
        x=x,
        init_bump_scale=float(args.init_bump_scale),
        steps=int(args.steps),
        lr=float(args.lr),
    )

    init_iso_rmse = float(np.sqrt(np.mean((z_wrong - z_true) ** 2)))
    final_iso_rmse = float(np.sqrt(np.mean((true_surface(x) + final_bump_scale * local_bump(x) - z_true) ** 2)))
    render_diff_wrong = float(np.max(np.abs(render_wrong - render_target)))
    render_diff_pulled = float(np.max(np.abs(render_pulled - render_target)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    im1 = plot_heatmap(
        axes[0, 0],
        x,
        z,
        sigma_wrong,
        z_true,
        z_wrong,
        iso_level,
        title="Wrong field: render-correct but front iso shifted",
        front_color="#d9480f",
    )
    im2 = plot_heatmap(
        axes[0, 1],
        x,
        z,
        sigma_pulled,
        z_true,
        z_pulled,
        iso_level,
        title="Pulled field: front iso back on the true surface",
        front_color="#2b8a3e",
    )
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label="sigma(x, z)")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label="sigma(x, z)")

    plot_curves(axes[1, 0], x, z_true, z_wrong, z_pulled, render_target, render_wrong, render_pulled)
    plot_optimization(axes[1, 1], history)

    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=180)
    plt.close(fig)

    save_summary(
        output_dir / "summary.txt",
        init_bump_scale=float(args.init_bump_scale),
        final_bump_scale=float(final_bump_scale),
        init_iso_rmse=init_iso_rmse,
        final_iso_rmse=final_iso_rmse,
        render_diff_wrong=render_diff_wrong,
        render_diff_pulled=render_diff_pulled,
    )

    np.savez(
        output_dir / "curves.npz",
        x=x,
        z_true=z_true,
        z_wrong=z_wrong,
        z_pulled=z_pulled,
        render_target=render_target,
        render_wrong=render_wrong,
        render_pulled=render_pulled,
        history_step=np.asarray([row["step"] for row in history], dtype=np.float64),
        history_bump_scale=np.asarray([row["bump_scale"] for row in history], dtype=np.float64),
        history_iso_rmse=np.asarray([row["iso_rmse"] for row in history], dtype=np.float64),
        history_render_loss=np.asarray([row["render_loss"] for row in history], dtype=np.float64),
    )


if __name__ == "__main__":
    main()
