#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAU = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


class ToyField:
    def __init__(self, radius=0.62, beta=12.0, bump_amp=0.055, bump_theta=0.7, bump_theta_sigma=0.35, bump_r_sigma=0.08):
        self.radius = float(radius)
        self.beta = float(beta)
        self.bump_amp = float(bump_amp)
        self.bump_theta = float(bump_theta)
        self.bump_theta_sigma = float(bump_theta_sigma)
        self.bump_r_sigma = float(bump_r_sigma)

    def phi_base(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        r = np.hypot(x, y)
        return self.radius - r

    def phi_bump(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        dtheta = wrap_angle(theta - self.bump_theta)
        return self.bump_amp * np.exp(-0.5 * (dtheta / self.bump_theta_sigma) ** 2) * np.exp(
            -0.5 * ((r - self.radius) / self.bump_r_sigma) ** 2
        )

    def field0(self, xy):
        phi = self.phi_base(xy)
        return sigmoid(self.beta * phi) - TAU

    def field1(self, xy):
        phi = self.phi_base(xy) + self.phi_bump(xy)
        return sigmoid(self.beta * phi) - TAU


def normalize(v):
    denom = np.linalg.norm(v)
    if denom < 1e-12:
        return v * 0.0
    return v / denom


def field_value(field_fn, point):
    point = np.asarray(point, dtype=np.float64)
    return float(field_fn(point[None, :])[0])


def gradient(field_fn, point, eps=2e-3):
    ex = np.array([eps, 0.0], dtype=np.float64)
    ey = np.array([0.0, eps], dtype=np.float64)
    fx1 = field_value(field_fn, point + ex)
    fx0 = field_value(field_fn, point - ex)
    fy1 = field_value(field_fn, point + ey)
    fy0 = field_value(field_fn, point - ey)
    return np.array([(fx1 - fx0) / (2.0 * eps), (fy1 - fy0) / (2.0 * eps)], dtype=np.float64)


def hessian(field_fn, point, eps=2e-3):
    ex = np.array([eps, 0.0], dtype=np.float64)
    ey = np.array([0.0, eps], dtype=np.float64)
    f00 = field_value(field_fn, point)
    fpx = field_value(field_fn, point + ex)
    fmx = field_value(field_fn, point - ex)
    fpy = field_value(field_fn, point + ey)
    fmy = field_value(field_fn, point - ey)
    fpp = field_value(field_fn, point + ex + ey)
    fpm = field_value(field_fn, point + ex - ey)
    fmp = field_value(field_fn, point - ex + ey)
    fmm = field_value(field_fn, point - ex - ey)
    dxx = (fpx - 2.0 * f00 + fmx) / (eps * eps)
    dyy = (fpy - 2.0 * f00 + fmy) / (eps * eps)
    dxy = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    return np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)


def tangent_projector(normal):
    return np.eye(2, dtype=np.float64) - np.outer(normal, normal)


def extract_main_contour(field_fn, limit=1.2, resolution=520):
    xs = np.linspace(-limit, limit, resolution)
    ys = np.linspace(-limit, limit, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    xy = np.stack([grid_x, grid_y], axis=-1)
    field = field_fn(xy)
    fig, ax = plt.subplots()
    contour = ax.contour(xs, ys, field, levels=[0.0])
    plt.close(fig)
    segments = contour.allsegs[0]
    if not segments:
        raise RuntimeError("Failed to extract zero contour from toy field.")
    segment = max(segments, key=lambda arr: arr.shape[0])
    if np.linalg.norm(segment[0] - segment[-1]) > 1e-8:
        segment = np.vstack([segment, segment[0]])
    return segment


def resample_closed_polyline(points, sample_count):
    points = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    targets = np.linspace(0.0, total_length, sample_count, endpoint=False)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1)


def find_root_along_normal(field_fn, point, normal, max_shift=0.2, steps=600):
    samples = np.linspace(-max_shift, max_shift, steps + 1)
    values = np.array([field_value(field_fn, point + s * normal) for s in samples], dtype=np.float64)
    sign_change_ids = np.where(values[:-1] * values[1:] <= 0.0)[0]
    if sign_change_ids.size > 0:
        centers = 0.5 * (samples[sign_change_ids] + samples[sign_change_ids + 1])
        idx = sign_change_ids[np.argmin(np.abs(centers))]
        s0 = samples[idx]
        s1 = samples[idx + 1]
        v0 = values[idx]
        v1 = values[idx + 1]
        if abs(v1 - v0) > 1e-12:
            root = s0 - v0 * (s1 - s0) / (v1 - v0)
        else:
            root = 0.5 * (s0 + s1)
        return root, True
    idx = int(np.argmin(np.abs(values)))
    return float(samples[idx]), False


def angle_between(a, b):
    a = normalize(a)
    b = normalize(b)
    cosine = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def compute_samples(field, sample_count=180):
    contour0 = extract_main_contour(field.field0)
    contour1 = extract_main_contour(field.field1)
    samples0 = resample_closed_polyline(contour0, sample_count)
    samples = []
    eps = 1e-8

    for point in samples0:
        f0 = field_value(field.field0, point)
        f1 = field_value(field.field1, point)
        delta_f = f1 - f0

        grad0 = gradient(field.field0, point)
        grad1 = gradient(field.field1, point)
        delta_g = grad1 - grad0
        hess0 = hessian(field.field0, point)

        grad_norm = max(np.linalg.norm(grad0), eps)
        normal0 = normalize(grad0)
        proj = tangent_projector(normal0)

        delta_s_pred = -delta_f / grad_norm
        curvature_push = proj @ (hess0 @ normal0)
        delta_n_iso = (delta_s_pred / grad_norm) * curvature_push
        delta_n_full = (proj @ (delta_g + delta_s_pred * (hess0 @ normal0))) / grad_norm

        predicted_point = point + delta_s_pred * normal0

        actual_shift, bracketed = find_root_along_normal(field.field1, point, normal0)
        actual_point = point + actual_shift * normal0
        actual_normal = normalize(gradient(field.field1, actual_point))
        if np.dot(actual_normal, normal0) < 0.0:
            actual_normal = -actual_normal

        predicted_normal = normalize(normal0 + delta_n_full)
        if np.dot(predicted_normal, normal0) < 0.0:
            predicted_normal = -predicted_normal

        samples.append(
            {
                "point": point,
                "predicted_point": predicted_point,
                "actual_point": actual_point,
                "normal0": normal0,
                "actual_normal": actual_normal,
                "predicted_normal": predicted_normal,
                "grad_norm": grad_norm,
                "delta_f": delta_f,
                "delta_s_pred": delta_s_pred,
                "delta_s_actual": actual_shift,
                "delta_n_iso_mag": np.linalg.norm(delta_n_iso),
                "delta_n_full_mag": np.linalg.norm(delta_n_full),
                "normal_angle_pred_deg": angle_between(normal0, predicted_normal),
                "normal_angle_actual_deg": angle_between(normal0, actual_normal),
                "bracketed": 1.0 if bracketed else 0.0,
                "iso_to_normal_score": abs(delta_f) * np.linalg.norm(curvature_push) / (grad_norm * grad_norm + eps),
                "surface_shift_score": abs(delta_f) / (grad_norm + eps),
                "normal_proxy_score": np.linalg.norm(proj @ delta_g) / (grad_norm + eps),
            }
        )

    return contour0, contour1, samples


def save_npz(samples, output_path):
    keys = sorted(samples[0].keys())
    arrays = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        arrays[key] = np.asarray(values)
    np.savez(output_path, **arrays)


def plot_results(contour0, contour1, samples, output_path):
    points = np.stack([sample["point"] for sample in samples], axis=0)
    predicted_points = np.stack([sample["predicted_point"] for sample in samples], axis=0)
    actual_points = np.stack([sample["actual_point"] for sample in samples], axis=0)
    score = np.asarray([sample["iso_to_normal_score"] for sample in samples], dtype=np.float64)
    shift_pred = np.asarray([sample["delta_s_pred"] for sample in samples], dtype=np.float64)
    shift_actual = np.asarray([sample["delta_s_actual"] for sample in samples], dtype=np.float64)
    angle_pred = np.asarray([sample["normal_angle_pred_deg"] for sample in samples], dtype=np.float64)
    angle_actual = np.asarray([sample["normal_angle_actual_deg"] for sample in samples], dtype=np.float64)
    angle_index = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(angle_index)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot(contour0[:, 0], contour0[:, 1], color="#2a6fdb", lw=2.0, label="F_t = 0")
    ax.plot(contour1[:, 0], contour1[:, 1], color="#f08c00", lw=2.0, label="F_t+dt = 0")
    scatter = ax.scatter(points[:, 0], points[:, 1], c=score, cmap="magma", s=28, label="iso_to_normal_score")
    ax.plot(predicted_points[:, 0], predicted_points[:, 1], color="#5c940d", lw=1.2, alpha=0.8, label="predicted shifted points")
    ax.plot(actual_points[:, 0], actual_points[:, 1], color="#c2255c", lw=1.0, alpha=0.8, label="actual shifted points")
    ax.set_title("Level-set shift and unstable arc")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.plot(angle_index[order], shift_pred[order], color="#5c940d", lw=2.0, label="pred shift")
    ax.plot(angle_index[order], shift_actual[order], color="#c2255c", lw=1.5, label="actual shift")
    ax.plot(angle_index[order], score[order], color="#7b2cbf", lw=1.4, label="iso_to_normal_score")
    ax.set_title("Surface shift along the contour")
    ax.set_xlabel("contour angle (rad)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(angle_index[order], angle_pred[order], color="#0b7285", lw=2.0, label="pred normal angle")
    ax.plot(angle_index[order], angle_actual[order], color="#d9480f", lw=1.5, label="actual normal angle")
    ax.set_title("Normal rotation")
    ax.set_xlabel("contour angle (rad)")
    ax.set_ylabel("degrees")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Toy example for isosurface-shift-to-normal-change instability.")
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/ltl/Desktop/codex_playground/SOF/toy_outputs/isofield_normal_instability_v0"))
    parser.add_argument("--sample-count", type=int, default=180)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    field = ToyField()
    contour0, contour1, samples = compute_samples(field, sample_count=int(args.sample_count))
    plot_results(contour0, contour1, samples, output_dir / "overview.png")
    save_npz(samples, output_dir / "samples.npz")

    summary = {
        "sample_count": int(len(samples)),
        "mean_surface_shift_score": float(np.mean([s["surface_shift_score"] for s in samples])),
        "max_surface_shift_score": float(np.max([s["surface_shift_score"] for s in samples])),
        "mean_iso_to_normal_score": float(np.mean([s["iso_to_normal_score"] for s in samples])),
        "max_iso_to_normal_score": float(np.max([s["iso_to_normal_score"] for s in samples])),
        "mean_pred_normal_angle_deg": float(np.mean([s["normal_angle_pred_deg"] for s in samples])),
        "mean_actual_normal_angle_deg": float(np.mean([s["normal_angle_actual_deg"] for s in samples])),
    }
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
