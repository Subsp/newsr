import argparse
import json
import os
import sys
from dataclasses import asdict

import torch
import torch.nn.functional as F


if __package__ is None or __package__ == "":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from hybrid_sdfgs.surfel_pseudosdf import (  # noqa: E402
    SurfelConsensusConfig,
    SurfelConsensusPseudoSDF,
    SurfelMLPConfig,
    SurfelPseudoSDFMLP,
    auto_bounds,
    build_normalization,
    crop_surfel_cloud,
    focus_bounds_main_object,
    load_surfel_cloud_from_gaussian_ply,
    manual_bounds,
    sample_surfel_training_points,
    save_surfel_mlp_checkpoint,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a continuous pseudoSDF MLP from 2DGS surfel primitives."
    )
    parser.add_argument("--point_cloud_ply", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="surfel_pseudosdf_mlp.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--opacity_min", type=float, default=0.02)
    parser.add_argument("--sheetness_min", type=float, default=2.0)
    parser.add_argument("--max_surfels", type=int, default=60000)
    parser.add_argument(
        "--normal_axis",
        type=str,
        default="min_scale",
        choices=["min_scale", "max_scale", "x", "y", "z"],
    )

    parser.add_argument(
        "--focus_mode",
        type=str,
        default="main_object",
        choices=["global", "main_object", "manual"],
    )
    parser.add_argument("--focus_cluster_ratio", type=float, default=0.5)
    parser.add_argument("--focus_inlier_quantile", type=float, default=0.9)
    parser.add_argument("--focus_padding_ratio", type=float, default=0.08)
    parser.add_argument("--focus_center", type=str, default="")
    parser.add_argument("--focus_extent", type=str, default="")

    parser.add_argument("--teacher_k_neighbors", type=int, default=8)
    parser.add_argument("--teacher_distance_floor", type=float, default=1e-4)
    parser.add_argument("--teacher_normal_support_scale", type=float, default=2.0)
    parser.add_argument("--teacher_tangent_support_scale", type=float, default=1.25)
    parser.add_argument("--teacher_score_euclid_coef", type=float, default=0.1)
    parser.add_argument("--teacher_score_conf_coef", type=float, default=0.05)
    parser.add_argument("--teacher_consensus_cos_thresh", type=float, default=0.5)
    parser.add_argument("--teacher_invalid_penalty_scale", type=float, default=1.5)
    parser.add_argument("--teacher_knn_chunk_size", type=int, default=4096)

    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_frequencies", type=int, default=6)
    parser.add_argument("--disable_skip", action="store_true")

    parser.add_argument("--surface_batch", type=int, default=2048)
    parser.add_argument("--near_surface_batch", type=int, default=2048)
    parser.add_argument("--uniform_batch", type=int, default=2048)
    parser.add_argument("--sampling_tangent_scale", type=float, default=0.75)
    parser.add_argument("--sampling_normal_scale", type=float, default=1.5)
    parser.add_argument("--teacher_query_chunk_size", type=int, default=8192)

    parser.add_argument("--sdf_loss_weight", type=float, default=1.0)
    parser.add_argument("--grad_loss_weight", type=float, default=0.1)
    parser.add_argument("--eikonal_loss_weight", type=float, default=0.05)
    parser.add_argument("--surface_band", type=float, default=0.02)
    parser.add_argument("--surface_weight_boost", type=float, default=2.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    return parser.parse_args()


def _resolve_bounds(args, surfels, device):
    if args.focus_mode == "main_object":
        return focus_bounds_main_object(
            points=surfels.xyz,
            weights=surfels.score,
            cluster_ratio=args.focus_cluster_ratio,
            inlier_quantile=args.focus_inlier_quantile,
            padding_ratio=args.focus_padding_ratio,
        )
    if args.focus_mode == "manual":
        if not args.focus_center or not args.focus_extent:
            raise ValueError("--focus_center and --focus_extent are required for focus_mode=manual")
        return manual_bounds(args.focus_center, args.focus_extent, device)
    return auto_bounds(surfels.xyz, padding_ratio=args.focus_padding_ratio)


def _surface_weight(target_sdf: torch.Tensor, band: float, boost: float):
    band = max(float(band), 1e-6)
    return 1.0 + float(boost) * torch.exp(-target_sdf.abs() / band)


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    surfels = load_surfel_cloud_from_gaussian_ply(
        path=args.point_cloud_ply,
        device=device,
        opacity_min=args.opacity_min,
        sheetness_min=args.sheetness_min,
        max_surfels=args.max_surfels,
        normal_axis=args.normal_axis,
    )
    bmin, bmax = _resolve_bounds(args, surfels, device)
    surfels = crop_surfel_cloud(surfels, bmin=bmin, bmax=bmax)
    if surfels.xyz.shape[0] == 0:
        raise RuntimeError("No surfels left after focus-region cropping.")

    teacher_cfg = SurfelConsensusConfig(
        k_neighbors=args.teacher_k_neighbors,
        distance_floor=args.teacher_distance_floor,
        normal_support_scale=args.teacher_normal_support_scale,
        tangent_support_scale=args.teacher_tangent_support_scale,
        score_euclid_coef=args.teacher_score_euclid_coef,
        score_conf_coef=args.teacher_score_conf_coef,
        consensus_cos_thresh=args.teacher_consensus_cos_thresh,
        invalid_penalty_scale=args.teacher_invalid_penalty_scale,
        knn_chunk_size=args.teacher_knn_chunk_size,
    )
    teacher = SurfelConsensusPseudoSDF(surfels=surfels, cfg=teacher_cfg)

    model_cfg = SurfelMLPConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_frequencies=args.num_frequencies,
        use_skip=not args.disable_skip,
    )
    model = SurfelPseudoSDFMLP(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    coord_center, coord_scale = build_normalization(bmin, bmax)
    coord_center = coord_center.to(device=device, dtype=torch.float32)
    coord_scale = coord_scale.to(device=device, dtype=torch.float32)
    ckpt_path = os.path.join(args.output_dir, args.checkpoint_name)

    last_metrics = {}
    for step in range(1, int(args.steps) + 1):
        batches = sample_surfel_training_points(
            surfels=surfels,
            bmin=bmin,
            bmax=bmax,
            num_surface=int(args.surface_batch),
            num_near_surface=int(args.near_surface_batch),
            num_uniform=int(args.uniform_batch),
            tangent_scale=args.sampling_tangent_scale,
            normal_scale=args.sampling_normal_scale,
        )
        points = batches["all"]
        with torch.no_grad():
            target_sdf, target_grad = teacher.query_sdf_and_gradients(
                points,
                chunk_size=args.teacher_query_chunk_size,
            )

        points_req = points.detach().clone().requires_grad_(True)
        points_norm = (points_req - coord_center[None, :]) / coord_scale
        pred_sdf = model(points_norm)
        pred_grad = torch.autograd.grad(
            pred_sdf,
            points_req,
            grad_outputs=torch.ones_like(pred_sdf),
            create_graph=True,
            retain_graph=True,
        )[0]

        weights = _surface_weight(
            target_sdf=target_sdf,
            band=args.surface_band,
            boost=args.surface_weight_boost,
        )
        sdf_loss = (weights * F.smooth_l1_loss(pred_sdf, target_sdf, reduction="none")).mean()
        grad_cos = F.cosine_similarity(
            F.normalize(pred_grad, dim=-1),
            F.normalize(target_grad, dim=-1),
            dim=-1,
            eps=1e-8,
        )
        grad_loss = (weights.squeeze(-1) * (1.0 - grad_cos)).mean()
        eikonal_loss = ((pred_grad.norm(dim=-1) - 1.0) ** 2).mean()

        total_loss = (
            float(args.sdf_loss_weight) * sdf_loss
            + float(args.grad_loss_weight) * grad_loss
            + float(args.eikonal_loss_weight) * eikonal_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        last_metrics = {
            "step": step,
            "total": float(total_loss.detach().item()),
            "sdf": float(sdf_loss.detach().item()),
            "grad": float(grad_loss.detach().item()),
            "eikonal": float(eikonal_loss.detach().item()),
            "teacher_abs_sdf_mean": float(target_sdf.abs().mean().item()),
        }
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            print(
                "[surfel-mlp-distill] "
                f"step={step} "
                f"total={last_metrics['total']:.6f} "
                f"sdf={last_metrics['sdf']:.6f} "
                f"grad={last_metrics['grad']:.6f} "
                f"eik={last_metrics['eikonal']:.6f} "
                f"|teacher|={last_metrics['teacher_abs_sdf_mean']:.6f}"
            )

        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            save_surfel_mlp_checkpoint(
                path=ckpt_path,
                model=model,
                model_cfg=model_cfg,
                coord_center=coord_center,
                coord_scale=coord_scale,
                bmin=bmin,
                bmax=bmax,
                meta={
                    "point_cloud_ply": args.point_cloud_ply,
                    "step": step,
                    "teacher_cfg": asdict(teacher_cfg),
                    "last_metrics": last_metrics,
                },
            )

    save_surfel_mlp_checkpoint(
        path=ckpt_path,
        model=model,
        model_cfg=model_cfg,
        coord_center=coord_center,
        coord_scale=coord_scale,
        bmin=bmin,
        bmax=bmax,
        meta={
            "point_cloud_ply": args.point_cloud_ply,
            "step": int(args.steps),
            "teacher_cfg": asdict(teacher_cfg),
            "last_metrics": last_metrics,
            "surfel_count": int(surfels.xyz.shape[0]),
            "bounds_min": bmin.detach().cpu().tolist(),
            "bounds_max": bmax.detach().cpu().tolist(),
        },
    )

    report = {
        "checkpoint": ckpt_path,
        "point_cloud_ply": args.point_cloud_ply,
        "surfel_count": int(surfels.xyz.shape[0]),
        "bounds_min": bmin.detach().cpu().tolist(),
        "bounds_max": bmax.detach().cpu().tolist(),
        "teacher_cfg": asdict(teacher_cfg),
        "model_cfg": asdict(model_cfg),
        "last_metrics": last_metrics,
    }
    report_path = os.path.join(args.output_dir, "surfel_pseudosdf_mlp_info.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[surfel-mlp-distill] checkpoint:", ckpt_path)
    print("[surfel-mlp-distill] report:", report_path)


if __name__ == "__main__":
    main()
