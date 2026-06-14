#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SQRT_2PI = math.sqrt(2.0 * math.pi)


def sigma_curve(z, amplitude, center, width):
    return amplitude * np.exp(-0.5 * ((z - center) / width) ** 2)


def render_proxy(amplitude, width):
    return 1.0 - math.exp(-amplitude * width * SQRT_2PI)


def solve_amplitude_for_render(target_render, width):
    target_render = float(np.clip(target_render, 1e-8, 1.0 - 1e-8))
    return -math.log(1.0 - target_render) / (float(width) * SQRT_2PI)


def front_isosurface(amplitude, center, width, iso_level):
    if amplitude <= iso_level:
        return float("nan")
    return float(center - width * math.sqrt(2.0 * math.log(amplitude / iso_level)))


def center_from_front_iso(target_iso, amplitude, width, iso_level):
    if amplitude <= iso_level:
        raise ValueError("Need amplitude > iso_level to define the front crossing.")
    return float(target_iso + width * math.sqrt(2.0 * math.log(amplitude / iso_level)))


def pack_params(center, width, amplitude):
    return np.array([float(center), math.log(float(width)), math.log(float(amplitude))], dtype=np.float64)


def unpack_params(params):
    center = float(params[0])
    width = float(np.exp(params[1]))
    amplitude = float(np.exp(params[2]))
    return center, width, amplitude


def loss_terms(params, target_render, target_iso, iso_level, width_ref, amp_ref, lambda_render, lambda_pull, lambda_width, lambda_amp):
    center, width, amplitude = unpack_params(params)
    render_val = render_proxy(amplitude, width)
    iso_val = front_isosurface(amplitude, center, width, iso_level)
    if not np.isfinite(iso_val):
        big = 1e6
        return {
            "total": big,
            "render": big,
            "pull": big,
            "width_reg": big,
            "amp_reg": big,
            "render_val": render_val,
            "iso_val": iso_val,
            "center": center,
            "width": width,
            "amplitude": amplitude,
        }

    render_loss = (render_val - target_render) ** 2
    pull_loss = (iso_val - target_iso) ** 2
    width_reg = (math.log(width) - math.log(width_ref)) ** 2
    amp_reg = (math.log(amplitude) - math.log(amp_ref)) ** 2
    total = (
        float(lambda_render) * render_loss
        + float(lambda_pull) * pull_loss
        + float(lambda_width) * width_reg
        + float(lambda_amp) * amp_reg
    )
    return {
        "total": float(total),
        "render": float(render_loss),
        "pull": float(pull_loss),
        "width_reg": float(width_reg),
        "amp_reg": float(amp_reg),
        "render_val": float(render_val),
        "iso_val": float(iso_val),
        "center": float(center),
        "width": float(width),
        "amplitude": float(amplitude),
    }


def make_family(render_target, iso_level, true_iso, width_values):
    rows = []
    for width in width_values:
        amplitude = solve_amplitude_for_render(render_target, width)
        if amplitude <= iso_level:
            continue
        for iso_target in np.linspace(true_iso - 0.16, true_iso + 0.16, 33):
            center = center_from_front_iso(iso_target, amplitude, width, iso_level)
            rows.append(
                {
                    "width": float(width),
                    "amplitude": float(amplitude),
                    "center": float(center),
                    "iso": float(iso_target),
                    "render": float(render_proxy(amplitude, width)),
                }
            )
    return rows


def plot_overview(output_path, z_grid, family_rows, init_stats, final_stats, pull_history, true_iso, iso_level):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    widths = np.array([row["width"] for row in family_rows], dtype=np.float64)
    centers = np.array([row["center"] for row in family_rows], dtype=np.float64)
    isos = np.array([row["iso"] for row in family_rows], dtype=np.float64)
    scatter = axes[0].scatter(widths, centers, c=isos, s=24, cmap="viridis")
    axes[0].scatter([init_stats["width"]], [init_stats["center"]], color="#d9480f", s=80, label="wrong init")
    axes[0].scatter([final_stats["width"]], [final_stats["center"]], color="#2b8a3e", s=80, label="pulled solution")
    axes[0].set_title("Exact-render solution family")
    axes[0].set_xlabel("width")
    axes[0].set_ylabel("center")
    axes[0].legend(loc="upper left", fontsize=9)
    cbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("front isosurface depth")

    sigma_init = sigma_curve(z_grid, init_stats["amplitude"], init_stats["center"], init_stats["width"])
    sigma_final = sigma_curve(z_grid, final_stats["amplitude"], final_stats["center"], final_stats["width"])
    axes[1].plot(z_grid, sigma_init, color="#d9480f", lw=2.0, label="wrong init field")
    axes[1].plot(z_grid, sigma_final, color="#2b8a3e", lw=2.0, label="pulled field")
    axes[1].axhline(iso_level, color="#495057", lw=1.2, ls="--", label="iso level")
    axes[1].axvline(true_iso, color="#1c7ed6", lw=1.5, ls="--", label="true surface")
    axes[1].axvline(init_stats["iso_val"], color="#d9480f", lw=1.2, ls=":")
    axes[1].axvline(final_stats["iso_val"], color="#2b8a3e", lw=1.2, ls=":")
    axes[1].set_title("Field before and after pull")
    axes[1].set_xlabel("depth")
    axes[1].set_ylabel("sigma(z)")
    axes[1].legend(loc="upper right", fontsize=8)

    steps = np.array([row["step"] for row in pull_history], dtype=np.float64)
    iso_vals = np.array([row["iso_val"] for row in pull_history], dtype=np.float64)
    render_err = np.array([row["render"] for row in pull_history], dtype=np.float64)
    pull_targets = np.array([row["pull_target"] for row in pull_history], dtype=np.float64)
    axes[2].plot(steps, pull_targets, color="#adb5bd", lw=1.4, ls="--", label="manual pull target")
    axes[2].plot(steps, iso_vals, color="#2b8a3e", lw=2.0, label="iso depth")
    axes[2].axhline(true_iso, color="#1c7ed6", lw=1.5, ls="--", label="true surface")
    axes[2].set_xlabel("optimization step")
    axes[2].set_ylabel("iso depth", color="#2b8a3e")
    axes[2].tick_params(axis="y", labelcolor="#2b8a3e")
    ax2 = axes[2].twinx()
    ax2.plot(steps, render_err, color="#d9480f", lw=1.8, label="render loss")
    ax2.set_ylabel("render loss", color="#d9480f")
    ax2.tick_params(axis="y", labelcolor="#d9480f")
    axes[2].set_title("Manual pull along render-equivalent family")

    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(path, init_stats, final_stats, render_target, true_iso):
    lines = [
        f"render_target: {render_target:.8f}",
        f"true_surface_depth: {true_iso:.8f}",
        f"init_render: {init_stats['render_val']:.8f}",
        f"init_iso: {init_stats['iso_val']:.8f}",
        f"final_render: {final_stats['render_val']:.8f}",
        f"final_iso: {final_stats['iso_val']:.8f}",
        f"render_error_after_pull: {abs(final_stats['render_val'] - render_target):.8e}",
        f"iso_error_before_pull: {abs(init_stats['iso_val'] - true_iso):.8e}",
        f"iso_error_after_pull: {abs(final_stats['iso_val'] - true_iso):.8e}",
        f"init_center: {init_stats['center']:.8f}",
        f"final_center: {final_stats['center']:.8f}",
        f"init_width: {init_stats['width']:.8f}",
        f"final_width: {final_stats['width']:.8f}",
        f"init_amplitude: {init_stats['amplitude']:.8f}",
        f"final_amplitude: {final_stats['amplitude']:.8f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Toy example: pull a wrong isosurface while keeping render correct.")
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/ltl/Desktop/codex_playground/SOF/toy_outputs/pull_isosurface_under_render_ambiguity_v0"))
    parser.add_argument("--render-target", type=float, default=0.94)
    parser.add_argument("--iso-level", type=float, default=0.55)
    parser.add_argument("--true-iso", type=float, default=0.42)
    parser.add_argument("--wrong-iso", type=float, default=0.30)
    parser.add_argument("--init-width", type=float, default=0.11)
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    render_target = float(args.render_target)
    iso_level = float(args.iso_level)
    true_iso = float(args.true_iso)
    wrong_iso = float(args.wrong_iso)
    init_width = float(args.init_width)

    init_amplitude = solve_amplitude_for_render(render_target, init_width)
    init_center = center_from_front_iso(wrong_iso, init_amplitude, init_width, iso_level)
    init_params = pack_params(init_center, init_width, init_amplitude)

    def make_stats_fn(target_iso):
        return lambda params: loss_terms(
            params=params,
            target_render=render_target,
            target_iso=target_iso,
            iso_level=iso_level,
            width_ref=init_width,
            amp_ref=init_amplitude,
            lambda_render=1.0,
            lambda_pull=1.0,
            lambda_width=0.0,
            lambda_amp=0.0,
        )

    full_history = []
    pull_schedule = np.linspace(wrong_iso, true_iso, int(args.steps), endpoint=True)
    for step_id, pull_target in enumerate(pull_schedule):
        center = center_from_front_iso(float(pull_target), init_amplitude, init_width, iso_level)
        params = pack_params(center, init_width, init_amplitude)
        stats = make_stats_fn(float(pull_target))(params)
        stats["step"] = int(step_id)
        stats["pull_target"] = float(pull_target)
        full_history.append(stats)

    init_stats = make_stats_fn(true_iso)(init_params)
    final_center = center_from_front_iso(true_iso, init_amplitude, init_width, iso_level)
    final_params = pack_params(final_center, init_width, init_amplitude)
    final_stats = make_stats_fn(true_iso)(final_params)
    final_stats["step"] = int(args.steps)
    final_stats["pull_target"] = float(true_iso)
    full_history.append(final_stats)

    width_values = np.linspace(max(0.05, init_width * 0.6), init_width * 1.7, 40)
    family_rows = make_family(render_target=render_target, iso_level=iso_level, true_iso=true_iso, width_values=width_values)

    z_grid = np.linspace(0.0, 1.05, 500, dtype=np.float64)
    plot_overview(output_dir / "overview.png", z_grid, family_rows, init_stats, final_stats, full_history, true_iso, iso_level)
    write_summary(output_dir / "summary.txt", init_stats, final_stats, render_target, true_iso)

    history_npz = {
        "step": np.asarray([row["step"] for row in full_history], dtype=np.float64),
        "pull_target": np.asarray([row["pull_target"] for row in full_history], dtype=np.float64),
        "iso_val": np.asarray([row["iso_val"] for row in full_history], dtype=np.float64),
        "render_val": np.asarray([row["render_val"] for row in full_history], dtype=np.float64),
        "render_loss": np.asarray([row["render"] for row in full_history], dtype=np.float64),
        "pull_loss": np.asarray([row["pull"] for row in full_history], dtype=np.float64),
        "center": np.asarray([row["center"] for row in full_history], dtype=np.float64),
        "width": np.asarray([row["width"] for row in full_history], dtype=np.float64),
        "amplitude": np.asarray([row["amplitude"] for row in full_history], dtype=np.float64),
    }
    np.savez(output_dir / "history.npz", **history_npz)


if __name__ == "__main__":
    main()
