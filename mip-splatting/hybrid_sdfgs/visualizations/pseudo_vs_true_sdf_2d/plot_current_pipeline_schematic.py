#!/usr/bin/env python3
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
import numpy as np


BG = '#f7f3ec'
PANEL = '#fffdf9'
BLUE = '#6fb1ff'
BLUE_EDGE = '#215d9c'
GREEN = '#236a52'
RED = '#d62828'
ORANGE = '#f4a261'
PURPLE = '#8b5cf6'
TEAL = '#2a9d8f'
GRAY = '#666666'
BLACK = '#111111'
GOLD = '#b07d2f'


def curve_y(x):
    return 0.18 * x**2 + 0.52 * x + 0.10


def curve_dy(x):
    return 0.36 * x + 0.52


def unit_tangent(x):
    v = np.array([1.0, curve_dy(x)], dtype=np.float64)
    return v / np.linalg.norm(v)


def unit_normal(x):
    t = unit_tangent(x)
    n = np.array([-t[1], t[0]], dtype=np.float64)
    return n / np.linalg.norm(n)


def ellipse_angle_deg(x):
    t = unit_tangent(x)
    return np.degrees(np.arctan2(t[1], t[0]))


def pseudo_sdf(points, anchors, normals, sigmas, confs):
    diff = points[:, None, :] - anchors[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    signed = np.sum(diff * normals[None, :, :], axis=-1)
    w = np.exp(-dist / np.clip(sigmas[None, :], 1e-6, None)) * confs[None, :]
    return np.sum(w * signed, axis=1) / np.clip(np.sum(w, axis=1), 1e-8, None)


def draw_gs_curve_panel(ax, anchors, normals, sigmas, confs):
    ax.set_facecolor(PANEL)
    xs = np.linspace(-1.55, 1.25, 420)
    ys = np.linspace(-0.75, 1.45, 320)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    sdf = pseudo_sdf(pts, anchors, normals, sigmas, confs).reshape(xx.shape)

    vmax = np.percentile(np.abs(sdf), 95)
    ax.imshow(
        sdf,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin='lower',
        cmap='coolwarm',
        alpha=0.36,
        vmin=-vmax,
        vmax=vmax,
        interpolation='bilinear',
    )
    ax.contour(xx, yy, sdf, levels=[0.0], colors=BLACK, linewidths=2.2)

    xline = np.linspace(-1.4, 1.12, 240)
    yline = curve_y(xline)
    ax.plot(xline, yline, '--', color=GOLD, lw=2.0, alpha=0.9)
    ax.text(0.24, 1.21, 'underlying 2D surface curve', color=GOLD, fontsize=11)

    for (cx, cy), n, s in zip(anchors, normals, sigmas):
        ang = ellipse_angle_deg(cx)
        e = Ellipse((cx, cy), width=0.42, height=0.12, angle=ang,
                    facecolor=BLUE, edgecolor=BLUE_EDGE, lw=1.8, alpha=0.62)
        ax.add_patch(e)
        ax.add_patch(Circle((cx, cy), 0.022, facecolor=ORANGE, edgecolor=BLACK, lw=0.8, zorder=5))
        ax.add_patch(FancyArrowPatch((cx, cy), (cx + 0.22 * n[0], cy + 0.22 * n[1]),
                                     arrowstyle='-|>', mutation_scale=13, lw=2.0, color=GREEN))
        ax.add_patch(Circle((cx, cy), s, facecolor='none', edgecolor=BLUE_EDGE, lw=1.0, alpha=0.18, linestyle=':'))

    qx = 0.22
    qy = curve_y(qx) + 0.22
    ax.plot([qx], [qy], marker='x', ms=10, mew=2.3, color=RED)
    ax.text(qx + 0.05, qy + 0.03, 'query x', color=RED, fontsize=11, weight='bold')

    for idx in [2, 3]:
        cx, cy = anchors[idx]
        ax.plot([cx, qx], [cy, qy], color=GRAY, lw=1.1, ls='--', alpha=0.75)

    ax.text(-1.47, 1.36, 'A. GS -> pseudo-SDF (local plane blend)', fontsize=15, weight='bold', color=BLACK)
    ax.text(-1.47, 1.23, 'Each Gaussian is treated as a local surface patch.', fontsize=10.8, color=GRAY)
    ax.text(-1.47, 1.11, 'Nearby tangent planes vote for the signed distance of x.', fontsize=10.8, color=GRAY)
    ax.text(-1.47, -0.62, 'black contour: pseudo-SDF zero level set', fontsize=10.5, color=BLACK)
    ax.text(-1.47, -0.72, 'orange dots: GS anchors   green arrows: local normals', fontsize=10.5, color=BLACK)

    ax.set_xlim(-1.55, 1.25)
    ax.set_ylim(-0.75, 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_densify_panel(ax, anchors, normals):
    ax.set_facecolor(PANEL)
    xline = np.linspace(-1.4, 1.12, 240)
    yline = curve_y(xline)
    ax.plot(xline, yline, '--', color=GOLD, lw=2.0, alpha=0.75)

    for (cx, cy), n in zip(anchors, normals):
        ang = ellipse_angle_deg(cx)
        e = Ellipse((cx, cy), width=0.42, height=0.12, angle=ang,
                    facecolor=BLUE, edgecolor=BLUE_EDGE, lw=1.6, alpha=0.35)
        ax.add_patch(e)

    # pseudo-SDF zero contour proxy using the same curve
    ax.plot(xline, yline + 0.03 * np.sin(4 * xline), color=BLACK, lw=2.2)

    src = np.array([1.12, -0.52])
    valid = np.array([0.32, curve_y(0.32) + 0.02])
    reject = np.array([-0.22, curve_y(-0.22) + 0.17])

    ax.add_patch(Circle(src, 0.026, facecolor='#2f6fed', edgecolor='#0d3c8f', lw=1.0, zorder=6))
    ax.add_patch(FancyArrowPatch(src, valid, arrowstyle='-|>', mutation_scale=14, lw=2.3, color='#2f6fed'))
    ax.add_patch(FancyArrowPatch(src, reject, arrowstyle='-|>', mutation_scale=14, lw=2.0, color=PURPLE, alpha=0.92))
    ax.text(src[0] - 0.52, src[1] - 0.08, '2D SR proposal ray', color='#2f6fed', fontsize=11, weight='bold')

    ax.add_patch(Circle(valid, 0.032, facecolor=TEAL, edgecolor=BLACK, lw=1.1, zorder=7))
    ax.text(valid[0] + 0.06, valid[1] + 0.03, 'valid hit', color=TEAL, fontsize=11, weight='bold')

    ax.add_patch(Circle(reject, 0.032, facecolor=PURPLE, edgecolor=BLACK, lw=1.1, zorder=7))
    ax.text(reject[0] - 0.50, reject[1] + 0.06, 'rejected hit\n(low support / bad align)', color=PURPLE, fontsize=10.5)

    anchor_x = 0.32
    n = unit_normal(anchor_x)
    t = unit_tangent(anchor_x)
    p0 = valid - 0.24 * t
    p1 = valid + 0.24 * t
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='#e76f51', lw=2.0)

    us = np.linspace(-0.18, 0.18, 7)
    pts = []
    for u in us:
        base = valid + u * t
        pts.append(base)
        pts.append(base + 0.04 * n)
        pts.append(base - 0.04 * n)
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c='#e63946', edgecolors='white', linewidths=0.6, zorder=8)

    ax.text(-1.47, 1.36, 'B. pseudo-SDF -> proposal gating -> densify', fontsize=15, weight='bold', color=BLACK)
    ax.text(-1.47, 1.23, 'Back-project 2D SR proposals; intersect rays with the pseudo-SDF.', fontsize=10.8, color=GRAY)
    ax.text(-1.47, 1.11, 'Keep only hits with enough GS support and alignment.', fontsize=10.8, color=GRAY)
    ax.text(-1.47, -0.62, 'red samples: extra local points used by SDF-guided densify', fontsize=10.5, color=BLACK)
    ax.text(-1.47, -0.72, 'they add geometry supervision back to GS parameters', fontsize=10.5, color=BLACK)

    ax.set_xlim(-1.55, 1.25)
    ax.set_ylim(-0.75, 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description='Draw a 2D schematic for GS -> pseudo-SDF and SDF densify.')
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--out_name', type=str, default='current_pipeline_schematic_2d.png')
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    out_svg = os.path.splitext(out_path)[0] + '.svg'

    anchor_xs = np.array([-1.10, -0.52, 0.02, 0.54], dtype=np.float64)
    anchors = np.stack([anchor_xs, curve_y(anchor_xs)], axis=1)
    normals = np.stack([unit_normal(x) for x in anchor_xs], axis=0)
    sigmas = np.array([0.24, 0.22, 0.20, 0.18], dtype=np.float64)
    confs = np.array([0.92, 0.98, 0.95, 0.90], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 7.2), facecolor=BG)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.06, wspace=0.08)
    fig.suptitle('Current 2D Schematic: GS -> pseudo-SDF -> SDF-guided densify', fontsize=21, fontweight='bold', y=0.965)
    fig.text(0.035, 0.925, 'We are still optimizing a GS field; the pseudo-SDF is an online geometric teacher.', fontsize=12.5, color='#4a4a4a')

    draw_gs_curve_panel(axes[0], anchors, normals, sigmas, confs)
    draw_densify_panel(axes[1], anchors, normals)

    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    fig.savefig(out_svg, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f'[ok] saved: {out_path}')
    print(f'[ok] saved: {out_svg}')


if __name__ == '__main__':
    main()
