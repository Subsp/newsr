#!/usr/bin/env python3
import argparse
import os

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_vec(text: str, n: int) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)} from: {text}")
    return np.array(vals, dtype=np.float32)


def build_plane_basis(origin: np.ndarray, normal: np.ndarray):
    n = normal.astype(np.float32)
    n /= np.linalg.norm(n) + 1e-12
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return origin.astype(np.float32), n, u.astype(np.float32), v.astype(np.float32)


def project_to_plane(points_3d: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    centered = points_3d - origin[None, :]
    x = np.dot(centered, u)
    y = np.dot(centered, v)
    return np.stack([x, y], axis=1).astype(np.float32)


def weighted_signed_field_knn(
    query_2d: np.ndarray,
    anchors_2d: np.ndarray,
    normals_2d: np.ndarray,
    sigma_n: np.ndarray,
    sigma_t: np.ndarray,
    conf: np.ndarray,
    k: int = 16,
    gate_tau: float = 0.86,
    gate_beta: float = 4.0,
    chunk: int = 2048,
) -> np.ndarray:
    p = query_2d.shape[0]
    k = max(1, min(k, anchors_2d.shape[0]))
    out = np.zeros((p,), dtype=np.float32)

    for s in range(0, p, chunk):
        e = min(p, s + chunk)
        q = query_2d[s:e]
        d = np.linalg.norm(q[:, None, :] - anchors_2d[None, :, :], axis=-1)
        knn = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]

        nbr_xyz = anchors_2d[knn]
        nbr_n = normals_2d[knn]
        nbr_sn = np.clip(sigma_n[knn], 1e-4, None)
        nbr_st = np.clip(sigma_t[knn], 1e-4, None)
        nbr_c = np.clip(conf[knn], 0.0, None)

        ref = nbr_n[:, :1, :]
        align = np.sign(np.sum(nbr_n * ref, axis=-1, keepdims=True))
        align[align == 0] = 1.0
        nbr_n = nbr_n * align

        delta = q[:, None, :] - nbr_xyz
        signed_plane = np.sum(delta * nbr_n, axis=-1)
        tangent = np.stack([-nbr_n[..., 1], nbr_n[..., 0]], axis=-1)
        d_t = np.sum(delta * tangent, axis=-1)
        d_n = np.abs(signed_plane)
        normed = np.sqrt((d_n / nbr_sn) ** 2 + (d_t / nbr_st) ** 2)
        base_w = np.exp(-normed)

        cos_ref = np.clip(np.abs(np.sum(nbr_n * ref, axis=-1)), 0.0, 1.0)
        gate = (cos_ref >= gate_tau).astype(np.float32) * np.exp(-gate_beta * (1.0 - cos_ref))
        w = base_w * nbr_c * gate
        w_sum = np.clip(np.sum(w, axis=1), 1e-8, None)
        out[s:e] = np.sum(w * signed_plane, axis=1) / w_sum

    return out


def estimate_normals_from_points2d(points: np.ndarray, k: int = 12, chunk: int = 1024) -> np.ndarray:
    n = points.shape[0]
    if n < 8:
        raise RuntimeError("Not enough points for 2D PCA normal estimation.")
    k_eff = max(3, min(k, n - 1))
    center = np.mean(points, axis=0, keepdims=True)
    out = np.zeros((n, 2), dtype=np.float32)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        q = points[s:e]
        d = np.linalg.norm(q[:, None, :] - points[None, :, :], axis=-1)
        idx = np.argpartition(d, kth=k_eff, axis=1)[:, : (k_eff + 1)]
        for i in range(e - s):
            nbr = points[idx[i]]
            # Remove identical point if present.
            mask = np.linalg.norm(nbr - q[i : i + 1], axis=1) > 1e-9
            nbr = nbr[mask]
            if nbr.shape[0] < 3:
                nbr = points[idx[i, :3]]
            m = np.mean(nbr, axis=0, keepdims=True)
            x = nbr - m
            cov = (x.T @ x) / max(1, x.shape[0])
            w, v = np.linalg.eigh(cov)
            t = v[:, np.argmax(w)]
            n2 = np.array([-t[1], t[0]], dtype=np.float32)
            # Orient outward from global centroid for consistency.
            to_out = (q[i] - center[0]).astype(np.float32)
            if np.dot(n2, to_out) < 0:
                n2 = -n2
            n2 /= np.linalg.norm(n2) + 1e-12
            out[s + i] = n2
    return out


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
        raise RuntimeError("Failed to extract 0-level contour from field.")
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


def build_prior_sequence_with_crop(
    anchors: np.ndarray,
    base_normals: np.ndarray,
    true_normals: np.ndarray,
    seq_visible_mask: np.ndarray,
    rng: np.random.Generator,
):
    c = np.mean(anchors, axis=0, keepdims=True)
    ang = np.arctan2(anchors[:, 1] - c[0, 1], anchors[:, 0] - c[0, 0])
    order = np.argsort(ang)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    base_s = base_normals[order]
    true_s = true_normals[order]
    vis_s = seq_visible_mask[order]

    lr_s = smooth_sequence(true_s, k=11)
    sr_s = smooth_sequence(true_s, k=5)
    lr_s = rotate_2d(lr_s, rng.uniform(-14.0, 14.0, size=(lr_s.shape[0],)).astype(np.float32))
    sr_s = rotate_2d(sr_s, rng.uniform(-4.0, 4.0, size=(sr_s.shape[0],)).astype(np.float32))

    # Outside cropped real-image area, prior is degraded and low-confidence.
    inv_noise = rng.uniform(-28.0, 28.0, size=(lr_s.shape[0],)).astype(np.float32)
    lr_s[~vis_s] = rotate_2d(base_s[~vis_s], inv_noise[~vis_s])
    sr_s[~vis_s] = rotate_2d(base_s[~vis_s], 0.5 * inv_noise[~vis_s])

    rel_lr = np.clip(np.abs(np.sum(lr_s * true_s, axis=1)), 0.0, 1.0).astype(np.float32)
    rel_sr = np.clip(np.abs(np.sum(sr_s * true_s, axis=1)), 0.0, 1.0).astype(np.float32)
    rel_lr[~vis_s] *= 0.35
    rel_sr[~vis_s] *= 0.35
    rel_base = np.clip(np.abs(np.sum(base_s * true_s, axis=1)), 0.0, 1.0).astype(np.float32)

    return {
        "order": order,
        "inv": inv,
        "visible_sorted": vis_s,
        "base": base_s,
        "true": true_s,
        "lr": lr_s,
        "sr": sr_s,
        "rel_base": rel_base,
        "rel_lr": rel_lr,
        "rel_sr": rel_sr,
    }


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
    base_s = base_normals[order]

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


def load_gaussian_ply_slice_field(
    ply_path: str,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    grid_res: int,
    crop_center: np.ndarray,
    crop_extent: np.ndarray,
    opacity_quantile: float,
    slice_thickness_mult: float,
    pad_ratio: float,
):
    obj = trimesh.load(ply_path)
    if not isinstance(obj, trimesh.points.PointCloud):
        raise TypeError(f"Expected point cloud ply, got {type(obj)}")
    raw = obj.metadata.get("_ply_raw", {}).get("vertex", {}).get("data", None)
    if raw is None:
        raise RuntimeError("Failed to read structured vertex data from ply.")
    names = set(raw.dtype.names or [])
    need = {"x", "y", "z", "nx", "ny", "nz", "opacity", "scale_0", "scale_1", "scale_2"}
    miss = need - names
    if miss:
        raise RuntimeError(f"Missing fields in ply: {sorted(miss)}")

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    n3 = np.stack([raw["nx"], raw["ny"], raw["nz"]], axis=1).astype(np.float32)
    n3 /= np.linalg.norm(n3, axis=1, keepdims=True) + 1e-12
    alpha = sigmoid(raw["opacity"].astype(np.float32))
    s0 = np.exp(raw["scale_0"].astype(np.float32))
    s1 = np.exp(raw["scale_1"].astype(np.float32))
    s2 = np.exp(raw["scale_2"].astype(np.float32))

    # Crop to a local spatial block near surface.
    half = 0.5 * crop_extent[None, :]
    in_crop = np.all(np.abs(xyz - crop_center[None, :]) <= half, axis=1)
    if np.count_nonzero(in_crop) < 300:
        # Fallback: use robust data center when user-given crop center misses the scene.
        auto_center = np.median(xyz, axis=0)
        in_crop = np.all(np.abs(xyz - auto_center[None, :]) <= half, axis=1)
    a_thr = float(np.quantile(alpha, opacity_quantile))
    keep = np.logical_and(in_crop, alpha >= a_thr)

    xyz = xyz[keep]
    n3 = n3[keep]
    alpha = alpha[keep]
    s0 = s0[keep]
    s1 = s1[keep]
    s2 = s2[keep]
    if xyz.shape[0] < 200:
        raise RuntimeError(f"Too few points after crop/filter: {xyz.shape[0]}. Increase crop_extent or reduce opacity_quantile.")

    origin, n, u, v = build_plane_basis(plane_origin, plane_normal)
    d_plane = np.abs(np.dot(xyz - origin[None, :], n))
    thick = float(np.quantile(s2, 0.8) * slice_thickness_mult)
    thick = max(thick, 1e-3)
    keep2 = d_plane <= thick

    xyz = xyz[keep2]
    n3 = n3[keep2]
    alpha = alpha[keep2]
    s0 = s0[keep2]
    s1 = s1[keep2]
    if xyz.shape[0] < 100:
        raise RuntimeError(f"Too few points near slice plane: {xyz.shape[0]}. Increase slice_thickness_mult.")

    p2 = project_to_plane(xyz, origin, u, v)
    n2_proj = np.stack([np.dot(n3, u), np.dot(n3, v)], axis=1).astype(np.float32)
    n2_norm = np.linalg.norm(n2_proj, axis=1)
    strong = n2_norm > 1e-5

    if np.count_nonzero(strong) < 120:
        # Fallback for nearly fronto-parallel normals: estimate 2D normals from local PCA.
        n2 = estimate_normals_from_points2d(p2, k=12, chunk=1024)
        s_t = np.sqrt(np.clip(s0 * s1, 1e-6, None)).astype(np.float32)
    else:
        p2 = p2[strong]
        n2 = n2_proj[strong]
        alpha = alpha[strong]
        s_t = np.sqrt(np.clip(s0[strong] * s1[strong], 1e-6, None)).astype(np.float32)
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12

    if p2.shape[0] < 80:
        raise RuntimeError(f"Too few usable projected points after projection: {p2.shape[0]}")

    lo = np.quantile(p2, 0.01, axis=0)
    hi = np.quantile(p2, 0.99, axis=0)
    center = 0.5 * (lo + hi)
    ext = np.max(hi - lo) * (0.5 + pad_ratio)
    x_lo, x_hi = float(center[0] - ext), float(center[0] + ext)
    y_lo, y_hi = float(center[1] - ext), float(center[1] + ext)
    xs = np.linspace(x_lo, x_hi, grid_res, dtype=np.float32)
    ys = np.linspace(y_lo, y_hi, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    q = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    # Dense oriented field from many real Gaussian points as reference "true" field.
    sigma_n = np.clip(0.50 * s_t, 0.01, None)
    sigma_t = np.clip(1.35 * s_t, 0.01, None)
    conf = np.clip(alpha, 0.05, 2.0)
    f = weighted_signed_field_knn(
        query_2d=q,
        anchors_2d=p2,
        normals_2d=n2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        conf=conf,
        k=24,
        gate_tau=0.75,
        gate_beta=2.0,
        chunk=1024,
    )
    return xs, ys, xx, yy, f.reshape(grid_res, grid_res), p2, n2, conf


def draw_centers(ax, anchors: np.ndarray, sparse_region: np.ndarray):
    ax.scatter(anchors[~sparse_region, 0], anchors[~sparse_region, 1], s=10, c="yellow", edgecolors="k", linewidths=0.25, zorder=4)
    if np.any(sparse_region):
        ax.scatter(anchors[sparse_region, 0], anchors[sparse_region, 1], s=12, c="orange", edgecolors="k", linewidths=0.25, zorder=4)


def main():
    parser = argparse.ArgumentParser(description="SR-prior toy from real LR Gaussian field slice (good_fused.ply).")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--out_name", type=str, default="pseudo_sdf_srprior_gaussian_slice_compare.png")
    parser.add_argument("--seq_out_name", type=str, default="pseudo_sdf_srprior_gaussian_slice_sequence.png")
    parser.add_argument("--gaussian_ply_path", type=str, default="/Users/ltl/Desktop/codex_playground/good_fused.ply")
    parser.add_argument("--grid_res", type=int, default=360)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--plane_origin", type=str, default="0,0,0")
    parser.add_argument("--plane_normal", type=str, default="0,0,1")
    parser.add_argument("--crop_center", type=str, default="0,0,0")
    parser.add_argument("--crop_extent", type=str, default="1.2,1.2,1.2")
    parser.add_argument("--opacity_quantile", type=float, default=0.35)
    parser.add_argument("--slice_thickness_mult", type=float, default=2.0)
    parser.add_argument("--pad_ratio", type=float, default=0.12)
    parser.add_argument("--n_anchors", type=int, default=180)
    parser.add_argument("--noise_deg", type=float, default=16.0)
    parser.add_argument("--sparse_keep", type=float, default=0.22)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--gate_tau", type=float, default=0.86)
    parser.add_argument("--gate_beta", type=float, default=4.0)
    parser.add_argument("--adaptive_alpha", type=float, default=2.2)
    parser.add_argument("--adaptive_rel_floor", type=float, default=0.72)
    parser.add_argument("--prior_mode", type=str, default="lr+sr", choices=["none", "lr", "sr", "lr+sr"])
    parser.add_argument("--prior_weight_lr", type=float, default=0.20)
    parser.add_argument("--prior_weight_sr", type=float, default=0.35)
    parser.add_argument("--prior_conf_gain", type=float, default=0.18)
    parser.add_argument("--seq_crop_ratio", type=float, default=0.45, help="Fractional width of the spatial crop used to build 1D prior sequence.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    plane_origin = parse_vec(args.plane_origin, 3)
    plane_normal = parse_vec(args.plane_normal, 3)
    crop_center = parse_vec(args.crop_center, 3)
    crop_extent = parse_vec(args.crop_extent, 3)

    xs, ys, xx, yy, true_field, dense_p2, dense_n2, dense_conf = load_gaussian_ply_slice_field(
        ply_path=args.gaussian_ply_path,
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        grid_res=args.grid_res,
        crop_center=crop_center,
        crop_extent=crop_extent,
        opacity_quantile=args.opacity_quantile,
        slice_thickness_mult=args.slice_thickness_mult,
        pad_ratio=args.pad_ratio,
    )
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    h = float((xs[-1] - xs[0]) / max(1, len(xs) - 1))
    gy, gx = np.gradient(true_field, h, h)

    contour_pts = extract_contour_points(xx, yy, true_field)
    idx = np.linspace(0, contour_pts.shape[0] - 1, args.n_anchors, dtype=np.int64)
    anchors = contour_pts[idx]
    true_normals = sample_grid_gradient(anchors, xs, ys, gx, gy)
    base_normals = rotate_2d(true_normals.copy(), rng.uniform(-args.noise_deg, args.noise_deg, size=(anchors.shape[0],)).astype(np.float32))

    xmid = float(np.median(anchors[:, 0]))
    ymid = float(np.median(anchors[:, 1]))
    xhalf = 0.5 * (xs.max() - xs.min()) * args.seq_crop_ratio
    yhalf = 0.5 * (ys.max() - ys.min()) * args.seq_crop_ratio
    seq_crop_box = (xmid - xhalf, xmid + xhalf, ymid - yhalf, ymid + yhalf)
    seq_visible = np.logical_and.reduce(
        (anchors[:, 0] >= seq_crop_box[0], anchors[:, 0] <= seq_crop_box[1], anchors[:, 1] >= seq_crop_box[2], anchors[:, 1] <= seq_crop_box[3])
    )

    sparse_box = (
        float(xs.min() + 0.10 * (xs.max() - xs.min())),
        float(xs.min() + 0.45 * (xs.max() - xs.min())),
        float(ys.min() + 0.52 * (ys.max() - ys.min())),
        float(ys.min() + 0.92 * (ys.max() - ys.min())),
    )
    keep = np.ones((anchors.shape[0],), dtype=bool)
    sparse_region = np.logical_and.reduce(
        (anchors[:, 0] >= sparse_box[0], anchors[:, 0] <= sparse_box[1], anchors[:, 1] >= sparse_box[2], anchors[:, 1] <= sparse_box[3])
    )
    rand_keep = rng.uniform(0.0, 1.0, size=(anchors.shape[0],)) < max(0.0, min(1.0, args.sparse_keep))
    keep = np.logical_and(keep, np.logical_or(~sparse_region, rand_keep))

    anchors = anchors[keep]
    true_normals = true_normals[keep]
    base_normals = base_normals[keep]
    seq_visible = seq_visible[keep]
    sparse_region = sparse_region[keep]

    sigma = rng.uniform(0.07, 0.20, size=(anchors.shape[0],)).astype(np.float32)
    conf = rng.uniform(0.55, 1.00, size=(anchors.shape[0],)).astype(np.float32)

    rel_base = anchor_normal_reliability(anchors, base_normals, k=6)
    rel_base_n = np.clip((rel_base - args.adaptive_rel_floor) / max(1e-6, 1.0 - args.adaptive_rel_floor), 0.0, 1.0)
    sigma_t_base = sigma.copy()
    sigma_t_base[sparse_region] *= 1.0 + args.adaptive_alpha * rel_base_n[sparse_region]
    pred_base = weighted_signed_field_knn(
        query_2d=pts,
        anchors_2d=anchors,
        normals_2d=base_normals,
        sigma_n=sigma,
        sigma_t=sigma_t_base,
        conf=conf,
        k=args.k_neighbors,
        gate_tau=args.gate_tau,
        gate_beta=args.gate_beta,
    ).reshape(true_field.shape)

    prior_pack = build_prior_sequence_with_crop(anchors, base_normals, true_normals, seq_visible, rng)
    prior_normals, conf_prior = fuse_prior(
        base_normals=base_normals,
        conf=conf,
        prior_pack=prior_pack,
        prior_mode=args.prior_mode,
        prior_weight_lr=args.prior_weight_lr,
        prior_weight_sr=args.prior_weight_sr,
        conf_gain=args.prior_conf_gain,
    )
    rel_prior = anchor_normal_reliability(anchors, prior_normals, k=6)
    rel_prior_n = np.clip((rel_prior - args.adaptive_rel_floor) / max(1e-6, 1.0 - args.adaptive_rel_floor), 0.0, 1.0)
    sigma_t_prior = sigma.copy()
    sigma_t_prior[sparse_region] *= 1.0 + args.adaptive_alpha * rel_prior_n[sparse_region]
    pred_prior = weighted_signed_field_knn(
        query_2d=pts,
        anchors_2d=anchors,
        normals_2d=prior_normals,
        sigma_n=sigma,
        sigma_t=sigma_t_prior,
        conf=conf_prior,
        k=args.k_neighbors,
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
    imp_global = 100.0 * (mae_base_global - mae_prior_global) / max(1e-8, mae_base_global)
    imp_sparse = 100.0 * (mae_base_sparse - mae_prior_sparse) / max(1e-8, mae_base_sparse)

    vmax = np.percentile(np.abs(true_field), 95)
    err_vmax = np.percentile(np.maximum(err_base, err_prior), 99)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    im0 = ax[0, 0].imshow(true_field, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 0].contour(xx, yy, true_field, levels=[0.0], colors="k", linewidths=1.2)
    draw_centers(ax[0, 0], anchors, sparse_region)
    ax[0, 0].set_title("Reference slice-field from real LR gaussians")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(pred_base, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 1].contour(xx, yy, pred_base, levels=[0.0], colors="k", linewidths=1.2)
    draw_centers(ax[0, 1], anchors, sparse_region)
    ax[0, 1].set_title("Baseline (no prior)")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(pred_prior, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0, 2].contour(xx, yy, pred_prior, levels=[0.0], colors="k", linewidths=1.2)
    draw_centers(ax[0, 2], anchors, sparse_region)
    ax[0, 2].set_title(f"With prior ({args.prior_mode})")
    fig.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(err_base, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="magma", vmin=0.0, vmax=err_vmax)
    ax[1, 0].set_title(f"|Err| baseline\nMAE global={mae_base_global:.4f}, sparse={mae_base_sparse:.4f}")
    fig.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(err_prior, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="magma", vmin=0.0, vmax=err_vmax)
    ax[1, 1].set_title(f"|Err| prior\nMAE global={mae_prior_global:.4f}, sparse={mae_prior_sparse:.4f}")
    fig.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im5 = ax[1, 2].imshow(delta, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="bwr", vmin=-err_vmax * 0.6, vmax=err_vmax * 0.6)
    ax[1, 2].set_title(f"ΔErr=prior-baseline\nglobal {imp_global:+.2f}%  sparse {imp_sparse:+.2f}%")
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
    vis = prior_pack["visible_sorted"]
    x_seq = np.arange(len(base_ang))
    seq_ax.plot(x_seq, true_ang, color="black", linewidth=1.5, label="true normal seq")
    seq_ax.plot(x_seq, base_ang, color="gray", linewidth=1.0, alpha=0.9, label="base noisy seq")
    seq_ax.plot(x_seq, lr_ang, color="tab:orange", linewidth=1.0, alpha=0.9, label="LR prior seq")
    seq_ax.plot(x_seq, sr_ang, color="tab:green", linewidth=1.0, alpha=0.9, label="SR prior seq")
    seq_ax.scatter(x_seq[~vis], true_ang[~vis], s=8, c="red", alpha=0.6, label="outside seq-crop")
    seq_ax.set_xlabel("1D anchor index (sorted)")
    seq_ax.set_ylabel("normal angle (deg)")
    seq_ax.set_title("1D prior sequence from cropped spatial region")
    seq_ax.grid(alpha=0.2)
    seq_ax.legend(loc="upper right", fontsize=8, ncol=2)
    seq_png = os.path.join(out_dir, args.seq_out_name)
    seq_fig.savefig(seq_png, dpi=180)
    plt.close(seq_fig)

    print(f"[ok] saved: {out_png}")
    print(f"[ok] saved: {seq_png}")
    print(
        "[metrics] "
        f"prior_mode={args.prior_mode} "
        f"baseline_mae_global={mae_base_global:.6f} prior_mae_global={mae_prior_global:.6f} "
        f"baseline_mae_sparse={mae_base_sparse:.6f} prior_mae_sparse={mae_prior_sparse:.6f} "
        f"improve_global_pct={imp_global:.3f} improve_sparse_pct={imp_sparse:.3f}"
    )


if __name__ == "__main__":
    main()
