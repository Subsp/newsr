#!/usr/bin/env python3
import argparse
import os

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import trimesh


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


def parse_vec(text: str, n: int) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)} from: {text}")
    return np.array(vals, dtype=np.float32)


def make_grid(lo: float, hi: float, n: int):
    xs = np.linspace(lo, hi, n, dtype=np.float32)
    ys = np.linspace(lo, hi, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return xs, ys, xx, yy, pts


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = np.sum(ab * ab)
    if ab2 < 1e-12:
        return np.linalg.norm(p - a[None, :], axis=1)
    t = np.sum((p - a[None, :]) * ab[None, :], axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(p - proj, axis=1)


def _project_to_plane(points_3d: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    centered = points_3d - origin[None, :]
    x = np.dot(centered, u)
    y = np.dot(centered, v)
    return np.stack([x, y], axis=1).astype(np.float32)


def build_true_field_from_mesh_slice(
    mesh_path: str,
    grid_res: int,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    pad_ratio: float = 0.15,
):
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")
    sec = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if sec is None:
        raise RuntimeError("No intersection between mesh and slicing plane.")

    curves3 = sec.discrete
    if len(curves3) == 0:
        raise RuntimeError("Empty section curves.")

    curves2 = []
    for c in curves3:
        p2 = _project_to_plane(np.asarray(c, dtype=np.float32), origin=plane_origin, normal=plane_normal)
        if p2.shape[0] >= 3:
            if np.linalg.norm(p2[0] - p2[-1]) > 1e-6:
                p2 = np.concatenate([p2, p2[:1]], axis=0)
            curves2.append(p2)
    if not curves2:
        raise RuntimeError("No valid 2D curve extracted from section.")

    all_pts = np.concatenate(curves2, axis=0)
    lo = np.min(all_pts, axis=0)
    hi = np.max(all_pts, axis=0)
    center = 0.5 * (lo + hi)
    ext = np.max(hi - lo) * (0.5 + pad_ratio)
    x_lo, x_hi = float(center[0] - ext), float(center[0] + ext)
    y_lo, y_hi = float(center[1] - ext), float(center[1] + ext)

    xs = np.linspace(x_lo, x_hi, grid_res, dtype=np.float32)
    ys = np.linspace(y_lo, y_hi, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    q = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    unsigned = np.full((q.shape[0],), np.inf, dtype=np.float32)
    for curve in curves2:
        for i in range(curve.shape[0] - 1):
            d = _point_to_segment_distance(q, curve[i], curve[i + 1]).astype(np.float32)
            unsigned = np.minimum(unsigned, d)

    inside = np.zeros((q.shape[0],), dtype=bool)
    for curve in curves2:
        path = mpath.Path(curve, closed=True)
        inside = np.logical_xor(inside, path.contains_points(q))

    sign = np.where(inside, -1.0, 1.0).astype(np.float32)
    sdf = unsigned * sign
    return xs, ys, xx, yy, sdf.reshape(grid_res, grid_res), curves2


def extract_contour_points(xx: np.ndarray, yy: np.ndarray, field: np.ndarray) -> np.ndarray:
    fig = plt.figure()
    cs = plt.contour(xx, yy, field, levels=[0.0])
    verts = []
    if len(cs.allsegs) > 0:
        for seg in cs.allsegs[0]:
            if seg.shape[0] > 5:
                verts.append(seg.astype(np.float32))
    plt.close(fig)
    if not verts:
        raise RuntimeError("Failed to extract 0-level contour.")
    return np.concatenate(verts, axis=0)


def sample_grid_gradient(points: np.ndarray, xs: np.ndarray, ys: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    ix = np.clip(np.searchsorted(xs, points[:, 0]), 1, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, points[:, 1]), 1, len(ys) - 1)
    x0 = xs[ix - 1]
    x1 = xs[ix]
    y0 = ys[iy - 1]
    y1 = ys[iy]
    tx = np.clip((points[:, 0] - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
    ty = np.clip((points[:, 1] - y0) / (y1 - y0 + 1e-12), 0.0, 1.0)

    def bilerp(arr):
        a00 = arr[iy - 1, ix - 1]
        a10 = arr[iy - 1, ix]
        a01 = arr[iy, ix - 1]
        a11 = arr[iy, ix]
        return (1 - tx) * (1 - ty) * a00 + tx * (1 - ty) * a10 + (1 - tx) * ty * a01 + tx * ty * a11

    nx = bilerp(gx)
    ny = bilerp(gy)
    n = np.stack([nx, ny], axis=1).astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def rotate_2d(v: np.ndarray, deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(deg).astype(np.float32)
    c = np.cos(rad)
    s = np.sin(rad)
    x = v[:, 0]
    y = v[:, 1]
    out = np.stack([c * x - s * y, s * x + c * y], axis=1)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def smooth_sequence(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return v.copy()
    pad = k // 2
    vv = np.pad(v, ((pad, pad), (0, 0)), mode="wrap")
    w = np.ones((k,), dtype=np.float32) / float(k)
    out = np.zeros_like(v)
    for i in range(v.shape[0]):
        out[i] = np.sum(vv[i : i + k] * w[:, None], axis=0)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def build_sr_prior_sequence(
    anchors: np.ndarray,
    base_normals: np.ndarray,
    true_normals: np.ndarray,
    rng: np.random.Generator,
):
    c = np.mean(anchors, axis=0, keepdims=True)
    ang = np.arctan2(anchors[:, 1] - c[0, 1], anchors[:, 0] - c[0, 0])
    order = np.argsort(ang)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    base_s = base_normals[order]
    true_s = true_normals[order]

    lr_s = smooth_sequence(true_s, k=11)
    lr_s = rotate_2d(lr_s, rng.uniform(-14.0, 14.0, size=(lr_s.shape[0],)).astype(np.float32))
    sr_s = smooth_sequence(true_s, k=5)
    sr_s = rotate_2d(sr_s, rng.uniform(-4.0, 4.0, size=(sr_s.shape[0],)).astype(np.float32))

    rel_lr = np.clip(np.abs(np.sum(lr_s * true_s, axis=1)), 0.0, 1.0).astype(np.float32)
    rel_sr = np.clip(np.abs(np.sum(sr_s * true_s, axis=1)), 0.0, 1.0).astype(np.float32)
    rel_base = np.clip(np.abs(np.sum(base_s * true_s, axis=1)), 0.0, 1.0).astype(np.float32)

    out = {
        "order": order,
        "inv": inv,
        "base": base_s,
        "true": true_s,
        "lr": lr_s,
        "sr": sr_s,
        "rel_base": rel_base,
        "rel_lr": rel_lr,
        "rel_sr": rel_sr,
    }
    return out


def fuse_prior(
    base_normals: np.ndarray,
    conf: np.ndarray,
    prior_pack: dict,
    prior_mode: str,
    prior_weight_lr: float,
    prior_weight_sr: float,
    conf_gain: float,
):
    if prior_mode == "none":
        return base_normals.copy(), conf.copy()

    order = prior_pack["order"]
    inv = prior_pack["inv"]

    if prior_mode == "lr":
        prior_s = prior_pack["lr"]
        rel = prior_pack["rel_lr"]
        w = prior_weight_lr
    elif prior_mode == "sr":
        prior_s = prior_pack["sr"]
        rel = prior_pack["rel_sr"]
        w = prior_weight_sr
    elif prior_mode == "lr+sr":
        wl = max(0.0, prior_weight_lr)
        ws = max(0.0, prior_weight_sr)
        prior_s = (wl * prior_pack["lr"] + ws * prior_pack["sr"]) / max(1e-8, wl + ws)
        prior_s /= np.linalg.norm(prior_s, axis=1, keepdims=True) + 1e-12
        rel = np.clip(0.5 * (prior_pack["rel_lr"] + prior_pack["rel_sr"]), 0.0, 1.0)
        w = min(1.0, 0.5 * (prior_weight_lr + prior_weight_sr))
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    base_s = base_normals[order]
    fused = (1.0 - w) * base_s + w * prior_s
    fused /= np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12
    fused = fused[inv]

    conf_s = conf[order]
    conf_s = conf_s * (1.0 + conf_gain * (rel - 0.5))
    conf_s = np.clip(conf_s, 0.05, 2.0)
    conf_f = conf_s[inv]
    return fused.astype(np.float32), conf_f.astype(np.float32)


def anchor_normal_reliability(anchors: np.ndarray, normals: np.ndarray, k: int = 6) -> np.ndarray:
    if anchors.shape[0] <= 2:
        return np.ones((anchors.shape[0],), dtype=np.float32)
    d = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    k_eff = max(1, min(k, anchors.shape[0] - 1))
    knn = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_n = normals[knn]
    dots = np.abs(np.sum(normals[:, None, :] * nn_n, axis=-1))
    rel = np.mean(dots, axis=1)
    return np.clip(rel.astype(np.float32), 0.0, 1.0)


def pseudo_sdf_from_anchors(
    points: np.ndarray,
    anchors: np.ndarray,
    normals: np.ndarray,
    sigma_n: np.ndarray,
    conf: np.ndarray,
    k_neighbors: int,
    sigma_t: np.ndarray,
    gate_tau: float,
    gate_beta: float,
) -> np.ndarray:
    p = points.shape[0]
    k = max(1, min(k_neighbors, anchors.shape[0]))

    d = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=-1)
    knn = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(p)[:, None]

    knn_d = d[rows, knn]
    nbr_xyz = anchors[knn]
    nbr_n = normals[knn]
    nbr_sn = np.clip(sigma_n[knn], 1e-4, None)
    nbr_st = np.clip(sigma_t[knn], 1e-4, None)
    nbr_c = np.clip(conf[knn], 0.0, None)

    ref = nbr_n[:, :1, :]
    align = np.sign(np.sum(nbr_n * ref, axis=-1, keepdims=True))
    align[align == 0] = 1.0
    nbr_n = nbr_n * align

    delta = points[:, None, :] - nbr_xyz
    signed_plane = np.sum(delta * nbr_n, axis=-1)
    tangent = np.stack([-nbr_n[..., 1], nbr_n[..., 0]], axis=-1)
    d_t = np.sum(delta * tangent, axis=-1)
    d_n = np.abs(signed_plane)
    normed = np.sqrt((d_n / nbr_sn) ** 2 + (d_t / nbr_st) ** 2)
    base_w = np.exp(-normed)

    cos_ref = np.clip(np.abs(np.sum(nbr_n * ref, axis=-1)), 0.0, 1.0)
    gate = (cos_ref >= gate_tau).astype(np.float32) * np.exp(-gate_beta * (1.0 - cos_ref))

    w = base_w * nbr_c * gate
    w_sum = np.clip(np.sum(w, axis=1, keepdims=True), 1e-8, None)
    sdf = np.sum(w * signed_plane, axis=1, keepdims=True) / w_sum
    return sdf[:, 0]


def draw_centers(ax, anchors: np.ndarray, sparse_region: np.ndarray):
    ax.scatter(anchors[~sparse_region, 0], anchors[~sparse_region, 1], s=10, c="yellow", edgecolors="k", linewidths=0.25, zorder=4)
    if np.any(sparse_region):
        ax.scatter(anchors[sparse_region, 0], anchors[sparse_region, 1], s=12, c="orange", edgecolors="k", linewidths=0.25, zorder=4)


def _sample_weighted_indices(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    w = np.clip(weights.astype(np.float64), 1e-12, None)
    w = w / np.sum(w)
    replace = n > len(w)
    return rng.choice(len(w), size=n, replace=replace, p=w)


def _unit_random_normals(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def apply_realistic_gs_characteristics(
    anchors: np.ndarray,
    true_normals: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    rng: np.random.Generator,
    args,
):
    # 1) Broken closure: remove one contiguous arc.
    n0 = anchors.shape[0]
    if n0 < 20:
        return anchors, true_normals, np.zeros((n0,), dtype=bool), np.zeros((n0,), dtype=bool)
    gap_len = max(2, int(round(args.real_gap_ratio * n0)))
    gap_start = int(rng.integers(0, n0))
    keep = np.ones((n0,), dtype=bool)
    for t in range(gap_len):
        keep[(gap_start + t) % n0] = False
    a = anchors[keep]
    n = true_normals[keep]

    # 2) Non-uniform density: emphasize selected regions and edges.
    c = np.mean(a, axis=0, keepdims=True)
    r = np.linalg.norm(a - c, axis=1)
    y_top = (a[:, 1] - float(ys.min())) / max(1e-8, float(ys.max() - ys.min()))
    hotspot = np.exp(-0.5 * ((a[:, 0] - 0.18) / 0.22) ** 2 - 0.5 * ((a[:, 1] - 0.20) / 0.20) ** 2)
    w = 0.10 + np.power(np.clip(r, 0.0, None), args.real_density_power) + 0.5 * y_top + args.real_hotspot_gain * hotspot
    idx = _sample_weighted_indices(w.astype(np.float32), args.n_anchors, rng)
    a = a[idx]
    n = n[idx]

    # 3) Thick-shell + multi-layer overlap: duplicate along normal and tangent.
    t = np.stack([-n[:, 1], n[:, 0]], axis=1)
    shell_pts = [a]
    shell_nrm = [n]
    for _ in range(max(0, args.real_shell_layers)):
        dn = rng.normal(0.0, args.real_shell_std, size=(a.shape[0], 1)).astype(np.float32)
        dt = rng.normal(0.0, args.real_tangent_jitter, size=(a.shape[0], 1)).astype(np.float32)
        shell_pts.append(a + dn * n + dt * t)
        shell_nrm.append(n.copy())
    a = np.concatenate(shell_pts, axis=0).astype(np.float32)
    n = np.concatenate(shell_nrm, axis=0).astype(np.float32)

    # 4) Background leakage: add low-confidence off-surface points.
    n_bg = int(round(args.real_bg_ratio * a.shape[0]))
    if n_bg > 0:
        bx = rng.uniform(float(xs.min()), float(xs.max()), size=(n_bg,)).astype(np.float32)
        by = rng.uniform(float(ys.min()), float(ys.max()), size=(n_bg,)).astype(np.float32)
        bpts = np.stack([bx, by], axis=1)
        bn = _unit_random_normals(n_bg, rng)
        a = np.concatenate([a, bpts], axis=0).astype(np.float32)
        n = np.concatenate([n, bn], axis=0).astype(np.float32)
        bg_mask = np.concatenate([np.zeros((a.shape[0] - n_bg,), dtype=bool), np.ones((n_bg,), dtype=bool)], axis=0)
    else:
        bg_mask = np.zeros((a.shape[0],), dtype=bool)

    # 5) Sparse region marker for later adaptive behavior.
    sparse_box = (
        float(xs.min() + 0.08 * (xs.max() - xs.min())),
        float(xs.min() + 0.45 * (xs.max() - xs.min())),
        float(ys.min() + 0.50 * (ys.max() - ys.min())),
        float(ys.min() + 0.90 * (ys.max() - ys.min())),
    )
    sparse_region = np.logical_and.reduce(
        (a[:, 0] >= sparse_box[0], a[:, 0] <= sparse_box[1], a[:, 1] >= sparse_box[2], a[:, 1] <= sparse_box[3])
    )
    return a, n, sparse_region, bg_mask


def apply_realistic_preset(args):
    if args.analytic_profile != "realistic":
        return
    if args.realistic_preset == "custom":
        return

    if args.realistic_preset == "mild":
        args.real_gap_ratio = 0.08
        args.real_density_power = 1.6
        args.real_hotspot_gain = 0.9
        args.real_shell_layers = 2
        args.real_shell_std = 0.020
        args.real_tangent_jitter = 0.010
        args.real_bg_ratio = 0.14
        args.real_bg_conf_scale = 0.22
        args.real_hi_noise_deg = 35.0
        args.real_flip_ratio = 0.08
        args.real_sigma_lognorm_std = 0.45
        return

    if args.realistic_preset == "hard":
        args.real_gap_ratio = 0.16
        args.real_density_power = 2.4
        args.real_hotspot_gain = 1.4
        args.real_shell_layers = 3
        args.real_shell_std = 0.032
        args.real_tangent_jitter = 0.018
        args.real_bg_ratio = 0.30
        args.real_bg_conf_scale = 0.14
        args.real_hi_noise_deg = 60.0
        args.real_flip_ratio = 0.18
        args.real_sigma_lognorm_std = 0.75
        return

    raise ValueError(f"Unknown realistic_preset: {args.realistic_preset}")


def main():
    parser = argparse.ArgumentParser(description="Introduce 1D SR prior and optional real 3D mesh slice into pseudo-SDF toy scene.")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--out_name", type=str, default="pseudo_sdf_srprior_real_slice_compare.png")
    parser.add_argument("--seq_out_name", type=str, default="pseudo_sdf_srprior_sequence.png")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--grid_res", type=int, default=360)
    parser.add_argument("--scene_source", type=str, default="analytic", choices=["analytic", "mesh"])
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--plane_origin", type=str, default="0,0,0")
    parser.add_argument("--plane_normal", type=str, default="0,0,1")
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--n_anchors", type=int, default=170)
    parser.add_argument("--noise_deg", type=float, default=12.0)
    parser.add_argument("--sparse_keep", type=float, default=0.20)
    parser.add_argument("--gate_tau", type=float, default=0.86)
    parser.add_argument("--gate_beta", type=float, default=4.0)
    parser.add_argument("--adaptive_alpha", type=float, default=2.2)
    parser.add_argument("--adaptive_rel_floor", type=float, default=0.72)
    parser.add_argument("--prior_mode", type=str, default="sr", choices=["none", "lr", "sr", "lr+sr"])
    parser.add_argument("--prior_weight_lr", type=float, default=0.35)
    parser.add_argument("--prior_weight_sr", type=float, default=0.65)
    parser.add_argument("--prior_conf_gain", type=float, default=0.30)
    parser.add_argument("--analytic_profile", type=str, default="clean", choices=["clean", "realistic"])
    parser.add_argument("--realistic_preset", type=str, default="mild", choices=["mild", "hard", "custom"])
    parser.add_argument("--real_gap_ratio", type=float, default=0.08)
    parser.add_argument("--real_density_power", type=float, default=1.6)
    parser.add_argument("--real_hotspot_gain", type=float, default=0.9)
    parser.add_argument("--real_shell_layers", type=int, default=2)
    parser.add_argument("--real_shell_std", type=float, default=0.020)
    parser.add_argument("--real_tangent_jitter", type=float, default=0.010)
    parser.add_argument("--real_bg_ratio", type=float, default=0.14)
    parser.add_argument("--real_bg_conf_scale", type=float, default=0.22)
    parser.add_argument("--real_hi_noise_deg", type=float, default=35.0)
    parser.add_argument("--real_flip_ratio", type=float, default=0.08)
    parser.add_argument("--real_sigma_lognorm_std", type=float, default=0.45)
    args = parser.parse_args()
    apply_realistic_preset(args)

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    if args.scene_source == "analytic":
        lo, hi = -1.35, 1.35
        xs, ys, xx, yy, pts = make_grid(lo, hi, args.grid_res)
        true_field = true_complex_sdf(pts).reshape(args.grid_res, args.grid_res)
        curves2 = None
    else:
        if not args.mesh_path:
            raise ValueError("--mesh_path is required when --scene_source mesh")
        origin = parse_vec(args.plane_origin, 3)
        normal = parse_vec(args.plane_normal, 3)
        xs, ys, xx, yy, true_field, curves2 = build_true_field_from_mesh_slice(
            mesh_path=args.mesh_path,
            grid_res=args.grid_res,
            plane_origin=origin,
            plane_normal=normal,
        )
        pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        lo, hi = float(xs.min()), float(xs.max())

    h = float((xs[-1] - xs[0]) / max(1, len(xs) - 1))
    gy, gx = np.gradient(true_field, h, h)

    contour_pts = extract_contour_points(xx, yy, true_field)
    idx = np.linspace(0, contour_pts.shape[0] - 1, args.n_anchors, dtype=np.int64)
    anchors = contour_pts[idx]
    true_normals = sample_grid_gradient(anchors, xs, ys, gx, gy)
    base_normals = true_normals.copy()

    sparse_box = (float(xs.min() + 0.10 * (xs.max() - xs.min())), float(xs.min() + 0.45 * (xs.max() - xs.min())), float(ys.min() + 0.50 * (ys.max() - ys.min())), float(ys.min() + 0.90 * (ys.max() - ys.min())))
    bg_mask = np.zeros((anchors.shape[0],), dtype=bool)
    if args.analytic_profile == "clean":
        keep = np.ones((anchors.shape[0],), dtype=bool)
        gap1 = np.logical_and(anchors[:, 0] > np.quantile(anchors[:, 0], 0.62), anchors[:, 1] > np.quantile(anchors[:, 1], 0.62))
        keep[gap1] = False
        sparse_region = np.logical_and.reduce(
            (anchors[:, 0] >= sparse_box[0], anchors[:, 0] <= sparse_box[1], anchors[:, 1] >= sparse_box[2], anchors[:, 1] <= sparse_box[3])
        )
        rand_keep = rng.uniform(0.0, 1.0, size=(anchors.shape[0],)) < max(0.0, min(1.0, args.sparse_keep))
        keep = np.logical_and(keep, np.logical_or(~sparse_region, rand_keep))
        anchors = anchors[keep]
        true_normals = true_normals[keep]
        base_normals = base_normals[keep]
        sparse_region = sparse_region[keep]
        base_normals = rotate_2d(base_normals, rng.uniform(-args.noise_deg, args.noise_deg, size=(base_normals.shape[0],)).astype(np.float32))
        sigma = rng.uniform(0.07, 0.20, size=(anchors.shape[0],)).astype(np.float32)
        conf = rng.uniform(0.55, 1.00, size=(anchors.shape[0],)).astype(np.float32)
    else:
        anchors, true_normals, sparse_region, bg_mask = apply_realistic_gs_characteristics(
            anchors=anchors,
            true_normals=true_normals,
            xs=xs,
            ys=ys,
            rng=rng,
            args=args,
        )
        base_normals = true_normals.copy()
        base_normals = rotate_2d(base_normals, rng.uniform(-args.noise_deg, args.noise_deg, size=(base_normals.shape[0],)).astype(np.float32))
        hi_noise = np.logical_or(sparse_region, bg_mask)
        hi_deg = np.zeros((base_normals.shape[0],), dtype=np.float32)
        hi_deg[hi_noise] = rng.uniform(-args.real_hi_noise_deg, args.real_hi_noise_deg, size=(np.count_nonzero(hi_noise),)).astype(np.float32)
        base_normals = rotate_2d(base_normals, hi_deg)
        flip_mask = np.logical_and(hi_noise, rng.uniform(0.0, 1.0, size=(base_normals.shape[0],)) < args.real_flip_ratio)
        base_normals[flip_mask] = -base_normals[flip_mask]
        sigma = np.exp(rng.normal(np.log(0.11), args.real_sigma_lognorm_std, size=(anchors.shape[0],)).astype(np.float32))
        sigma = np.clip(sigma, 0.05, 0.32).astype(np.float32)
        conf = np.exp(rng.normal(-0.35, 0.95, size=(anchors.shape[0],)).astype(np.float32))
        conf = np.clip(conf, 0.04, 1.80).astype(np.float32)
        conf[bg_mask] *= args.real_bg_conf_scale
        conf = np.clip(conf, 0.02, 1.80).astype(np.float32)

    rel_base_anchor = anchor_normal_reliability(anchors, base_normals, k=6)
    rel_norm_base = np.clip((rel_base_anchor - args.adaptive_rel_floor) / max(1e-6, 1.0 - args.adaptive_rel_floor), 0.0, 1.0)
    sigma_t_base = sigma.copy()
    sigma_t_base[sparse_region] *= 1.0 + args.adaptive_alpha * rel_norm_base[sparse_region]

    pred_base = pseudo_sdf_from_anchors(
        points=pts,
        anchors=anchors,
        normals=base_normals,
        sigma_n=sigma,
        sigma_t=sigma_t_base,
        conf=conf,
        k_neighbors=args.k_neighbors,
        gate_tau=args.gate_tau,
        gate_beta=args.gate_beta,
    ).reshape(true_field.shape)

    prior_pack = build_sr_prior_sequence(anchors, base_normals, true_normals, rng)
    prior_normals, conf_prior = fuse_prior(
        base_normals=base_normals,
        conf=conf,
        prior_pack=prior_pack,
        prior_mode=args.prior_mode,
        prior_weight_lr=args.prior_weight_lr,
        prior_weight_sr=args.prior_weight_sr,
        conf_gain=args.prior_conf_gain,
    )

    rel_prior_anchor = anchor_normal_reliability(anchors, prior_normals, k=6)
    rel_norm_prior = np.clip((rel_prior_anchor - args.adaptive_rel_floor) / max(1e-6, 1.0 - args.adaptive_rel_floor), 0.0, 1.0)
    sigma_t_prior = sigma.copy()
    sigma_t_prior[sparse_region] *= 1.0 + args.adaptive_alpha * rel_norm_prior[sparse_region]

    pred_prior = pseudo_sdf_from_anchors(
        points=pts,
        anchors=anchors,
        normals=prior_normals,
        sigma_n=sigma,
        sigma_t=sigma_t_prior,
        conf=conf_prior,
        k_neighbors=args.k_neighbors,
        gate_tau=args.gate_tau,
        gate_beta=args.gate_beta,
    ).reshape(true_field.shape)

    err_base = np.abs(pred_base - true_field)
    err_prior = np.abs(pred_prior - true_field)
    delta = err_prior - err_base

    sparse_mask = np.logical_and.reduce(
        (xx >= sparse_box[0], xx <= sparse_box[1], yy >= sparse_box[2], yy <= sparse_box[3])
    )
    mae_base_global = float(err_base.mean())
    mae_prior_global = float(err_prior.mean())
    mae_base_sparse = float(err_base[sparse_mask].mean())
    mae_prior_sparse = float(err_prior[sparse_mask].mean())

    vmax = np.percentile(np.abs(true_field), 95)
    err_vmax = np.percentile(np.maximum(err_base, err_prior), 99)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    im0 = ax[0, 0].imshow(true_field, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 0].contour(xx, yy, true_field, levels=[0.0], colors="k", linewidths=1.4)
    if curves2 is not None:
        for c in curves2:
            ax[0, 0].plot(c[:, 0], c[:, 1], color="white", linewidth=0.6, alpha=0.5)
    draw_centers(ax[0, 0], anchors, sparse_region)
    ax[0, 0].set_title(f"True SDF ({args.scene_source}) + centers")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(pred_base, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 1].contour(xx, yy, pred_base, levels=[0.0], colors="k", linewidths=1.4)
    draw_centers(ax[0, 1], anchors, sparse_region)
    ax[0, 1].set_title("Baseline (no prior)")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(pred_prior, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 2].contour(xx, yy, pred_prior, levels=[0.0], colors="k", linewidths=1.4)
    draw_centers(ax[0, 2], anchors, sparse_region)
    ax[0, 2].set_title(f"With prior ({args.prior_mode})")
    fig.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(err_base, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="magma", vmin=0.0, vmax=err_vmax)
    ax[1, 0].set_title(f"|Err| baseline\n(global {mae_base_global:.4f}, sparse {mae_base_sparse:.4f})")
    fig.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(err_prior, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="magma", vmin=0.0, vmax=err_vmax)
    ax[1, 1].set_title(f"|Err| prior\n(global {mae_prior_global:.4f}, sparse {mae_prior_sparse:.4f})")
    fig.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im5 = ax[1, 2].imshow(delta, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="bwr", vmin=-err_vmax * 0.6, vmax=err_vmax * 0.6)
    ax[1, 2].set_title("ΔErr = prior - baseline\n(blue better, red worse)")
    fig.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

    for a in ax.reshape(-1):
        a.set_aspect("equal")
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.plot([sparse_box[0], sparse_box[1], sparse_box[1], sparse_box[0], sparse_box[0]], [sparse_box[2], sparse_box[2], sparse_box[3], sparse_box[3], sparse_box[2]], "w--", linewidth=0.9, alpha=0.9)

    out_png = os.path.join(out_dir, args.out_name)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    seq_fig, seq_ax = plt.subplots(1, 1, figsize=(10, 3.8), constrained_layout=True)
    base_ang = np.rad2deg(np.arctan2(prior_pack["base"][:, 1], prior_pack["base"][:, 0]))
    true_ang = np.rad2deg(np.arctan2(prior_pack["true"][:, 1], prior_pack["true"][:, 0]))
    lr_ang = np.rad2deg(np.arctan2(prior_pack["lr"][:, 1], prior_pack["lr"][:, 0]))
    sr_ang = np.rad2deg(np.arctan2(prior_pack["sr"][:, 1], prior_pack["sr"][:, 0]))
    x_seq = np.arange(len(base_ang))
    seq_ax.plot(x_seq, true_ang, color="black", linewidth=1.6, label="true normal seq")
    seq_ax.plot(x_seq, base_ang, color="gray", linewidth=1.2, alpha=0.9, label="base noisy seq")
    seq_ax.plot(x_seq, lr_ang, color="tab:orange", linewidth=1.2, alpha=0.9, label="LR prior seq")
    seq_ax.plot(x_seq, sr_ang, color="tab:green", linewidth=1.2, alpha=0.9, label="SR prior seq")
    seq_ax.set_xlabel("1D anchor index (sorted)")
    seq_ax.set_ylabel("normal angle (deg)")
    seq_ax.set_title("1D LR/SR prior sequences over anchors")
    seq_ax.grid(alpha=0.2)
    seq_ax.legend(loc="upper right", fontsize=8)
    seq_png = os.path.join(out_dir, args.seq_out_name)
    seq_fig.savefig(seq_png, dpi=180)
    plt.close(seq_fig)

    print(f"[ok] saved: {out_png}")
    print(f"[ok] saved: {seq_png}")
    print(
        "[metrics] "
        f"scene={args.scene_source} profile={args.analytic_profile} preset={args.realistic_preset} prior_mode={args.prior_mode} "
        f"baseline_mae_global={mae_base_global:.6f} prior_mae_global={mae_prior_global:.6f} "
        f"baseline_mae_sparse={mae_base_sparse:.6f} prior_mae_sparse={mae_prior_sparse:.6f}"
    )


if __name__ == "__main__":
    main()
