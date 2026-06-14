from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.hrgs_refiner_dataset import (
    HRGSRefinerDataset,
    HRGSRefinerDatasetConfig,
    HRGSRefinerSceneSpec,
    collate_hrgs_refiner_batch,
    load_scene_specs_from_json,
)
from losses.hrgs_refiner_losses import HRGSRefinerLossConfig, compute_hrgsrefiner_losses
from models.hrgs_refiner import HRGSRefiner
from utils.gs_action_aggregator import aggregate_gs_actions, save_gs_action_payload
from utils.surface_payload_lifter import SurfacePayloadLifterConfig, lift_surface_payload, save_surface_payload_npz


def _log(message: str) -> None:
    print(message, flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _nested_to_device(value, device: str):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_nested_to_device(item, device) for item in value]
    return value


def _ensure_batch_dim(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sample)
    out["images"] = dict(sample["images"])
    out["lr_gs_buffers"] = dict(sample["lr_gs_buffers"])
    out["targets"] = dict(sample["targets"])
    out["cameras"] = dict(sample["cameras"])
    out["vggt_prior"] = dict(sample["vggt_prior"])
    out["visibility_records"] = dict(sample["visibility_records"])

    for key in ("images_sr", "images_lr_up", "images_lr_native"):
        if out["images"][key].ndim == 4:
            out["images"][key] = out["images"][key].unsqueeze(0)
    for key in ("render_rgb", "depth", "normal", "alpha", "diagnostics"):
        if out["lr_gs_buffers"][key].ndim == 4:
            out["lr_gs_buffers"][key] = out["lr_gs_buffers"][key].unsqueeze(0)
    for key in ("oracle_depth", "valid_depth", "surface_mask"):
        if out["targets"][key].ndim == 4:
            out["targets"][key] = out["targets"][key].unsqueeze(0)
    for key in ("intrinsics", "world_to_view", "cam_to_world"):
        if out["cameras"][key].ndim == 3:
            out["cameras"][key] = out["cameras"][key].unsqueeze(0)
    for key, value in list(out["vggt_prior"].items()):
        if torch.is_tensor(value) and value.ndim == 4:
            out["vggt_prior"][key] = value.unsqueeze(0)
    for key in ("gaussian_ids", "weights"):
        if out["visibility_records"][key].ndim == 4:
            out["visibility_records"][key] = out["visibility_records"][key].unsqueeze(0)
    return out


def _downsample_bvchw(tensor: torch.Tensor, target_hw) -> torch.Tensor:
    b, v, c, _, _ = tensor.shape
    out = torch.nn.functional.interpolate(
        tensor.flatten(0, 1),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return out.unflatten(0, (b, v))


def _build_scene_specs(args: argparse.Namespace) -> List[HRGSRefinerSceneSpec]:
    if args.scene_specs_json:
        return load_scene_specs_from_json(args.scene_specs_json)
    if not args.scene_root or not args.gs_model_path or not args.oracle_root:
        raise ValueError("Pass either --scene_specs_json or all of --scene_root/--gs_model_path/--oracle_root.")
    return [
        HRGSRefinerSceneSpec(
            scene_root=args.scene_root,
            gs_model_path=args.gs_model_path,
            oracle_root=args.oracle_root,
            source_images_subdir=args.source_images_subdir,
            target_images_subdir=args.target_images_subdir,
            priors_dir=args.priors_dir,
            load_iteration=args.load_iteration,
        )
    ]


def _build_loss_config(args: argparse.Namespace) -> HRGSRefinerLossConfig:
    return HRGSRefinerLossConfig(
        depth_weight=args.loss_depth_w,
        normal_weight=args.loss_normal_w,
        multiview_weight=args.loss_mv_w,
        delta_weight=args.loss_delta_w,
        surface_mask_weight=args.loss_surface_w,
        update_mask_weight=args.loss_update_w,
        detail_mask_weight=args.loss_detail_w,
        confidence_weight=args.loss_conf_w,
        prior_color_weight=args.loss_prior_color_w,
        vggt_conf_threshold=args.vggt_conf_threshold,
        gs_alpha_threshold=args.gs_alpha_threshold,
        detail_grad_threshold=args.detail_grad_threshold,
        reprojection_tolerance=args.reprojection_tolerance,
    )


def _save_checkpoint(
    path: Path,
    model: HRGSRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": int(step),
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def _export_eval_payload(
    model: HRGSRefiner,
    sample: Dict[str, Any],
    output_dir: Path,
    step: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _ensure_batch_dim(sample)
    outputs = model(
        sr_images=sample["images"]["images_sr"],
        lr_up_images=sample["images"]["images_lr_up"],
        gs_buffers=sample["lr_gs_buffers"],
        cameras=sample["cameras"],
        vggt_prior=sample["vggt_prior"],
    )
    carrier_payload = lift_surface_payload(
        depth_surf=outputs["surface_2d"]["depth_surf"],
        normal_surf=outputs["surface_2d"]["normal_surf"],
        conf_geo=outputs["surface_2d"]["conf_geo"],
        mask_surface=outputs["surface_2d"]["mask_surface"],
        sr_images=sample["images"]["images_sr"],
        cameras=sample["cameras"],
        cfg=SurfacePayloadLifterConfig(min_confidence=0.2, max_disagreement=1.0, min_views_per_cluster=1),
    )
    coarse_h, coarse_w = [int(x) for x in sample["visibility_records"]["coarse_hw"].tolist()]
    masks_2d = {
        "mask_update2d": _downsample_bvchw(outputs["surface_2d"]["mask_update2d"], (coarse_h, coarse_w)),
        "mask_surface": _downsample_bvchw(outputs["surface_2d"]["mask_surface"], (coarse_h, coarse_w)),
        "mask_detail": _downsample_bvchw(outputs["surface_2d"]["mask_detail"], (coarse_h, coarse_w)),
        "prior_color_weight2d": _downsample_bvchw(outputs["update_2d"]["prior_color_weight2d"], (coarse_h, coarse_w)),
    }
    action_features_2d = _downsample_bvchw(outputs["update_2d"]["action_features2d"], (coarse_h, coarse_w))
    gs_action_payload = aggregate_gs_actions(
        masks_2d=masks_2d,
        action_features_2d=action_features_2d,
        visibility_records={
            "gaussian_ids": sample["visibility_records"]["gaussian_ids"],
            "weights": sample["visibility_records"]["weights"].to(dtype=sample["images"]["images_sr"].dtype),
        },
        num_gaussians=int(sample["meta"]["num_gaussians"]),
    )
    torch.save(
        {
            "surface_2d": {key: value.detach().cpu() for key, value in outputs["surface_2d"].items()},
            "update_2d": {key: value.detach().cpu() for key, value in outputs["update_2d"].items()},
            "carrier_payload": {key: value.detach().cpu() for key, value in carrier_payload.items()},
            "gs_action_payload": {key: value.detach().cpu() for key, value in gs_action_payload.items()},
            "meta": {
                "step": int(step),
                "scene_id": sample["scene_id"],
                "view_names": list(sample["view_names"]),
            },
        },
        output_dir / "hrgs_outputs.pt",
    )
    save_surface_payload_npz(output_dir / "carrier_payload.npz", carrier_payload)
    save_gs_action_payload(output_dir / "gs_action_payload.pt", gs_action_payload)
    summary = {
        "step": int(step),
        "scene_id": sample["scene_id"],
        "num_carriers": int(carrier_payload["centers"].shape[0]),
        "num_valid_carriers": int(carrier_payload["valid_mask"].sum().item()),
        "num_gaussians": int(sample["meta"]["num_gaussians"]),
        "mean_update_strength": float(gs_action_payload["update_strength"].mean().item()),
        "mean_attach_strength": float(gs_action_payload["attach_strength"].mean().item()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HRGSRefiner-v0 from VGGT priors, GS buffers, and oracle geometry.")
    parser.add_argument("--scene_specs_json", default=None)
    parser.add_argument("--scene_root", default=None)
    parser.add_argument("--gs_model_path", default=None)
    parser.add_argument("--oracle_root", default=None)
    parser.add_argument("--source_images_subdir", default="images_8")
    parser.add_argument("--target_images_subdir", default="images_2")
    parser.add_argument("--priors_dir", default=None)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--vggt_root", default="/root/autodl-tmp/vggt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--samples_per_scene", type=int, default=16)
    parser.add_argument("--camera_resolution", type=float, default=2)
    parser.add_argument("--require_priors", action="store_true")
    parser.add_argument("--cache_samples", action="store_true")
    parser.add_argument("--cache_dir", default=None)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--refiner_checkpoint", default=None)

    parser.add_argument("--loss_depth_w", type=float, default=1.0)
    parser.add_argument("--loss_normal_w", type=float, default=0.2)
    parser.add_argument("--loss_mv_w", type=float, default=0.0)
    parser.add_argument("--loss_delta_w", type=float, default=0.02)
    parser.add_argument("--loss_surface_w", type=float, default=0.2)
    parser.add_argument("--loss_update_w", type=float, default=0.1)
    parser.add_argument("--loss_detail_w", type=float, default=0.1)
    parser.add_argument("--loss_conf_w", type=float, default=0.1)
    parser.add_argument("--loss_prior_color_w", type=float, default=0.05)
    parser.add_argument("--vggt_conf_threshold", type=float, default=0.3)
    parser.add_argument("--gs_alpha_threshold", type=float, default=0.05)
    parser.add_argument("--detail_grad_threshold", type=float, default=0.08)
    parser.add_argument("--reprojection_tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) != 1:
        raise ValueError("train_hrgsrefiner_v0.py currently supports only --batch_size 1.")
    _set_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    ckpt_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_specs = _build_scene_specs(args)
    dataset = HRGSRefinerDataset(
        scene_specs=scene_specs,
        config=HRGSRefinerDatasetConfig(
            vggt_root=args.vggt_root,
            device=args.device,
            num_views=int(args.num_views),
            samples_per_scene=int(args.samples_per_scene),
            camera_resolution=args.camera_resolution,
            require_priors=bool(args.require_priors),
            seed=int(args.seed),
            cache_samples=bool(args.cache_samples),
            cache_dir=args.cache_dir,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_hrgs_refiner_batch,
    )
    loader_iter = iter(loader)

    probe_sample = _ensure_batch_dim(_nested_to_device(dataset[0], args.device))
    gs_diag_channels = int(probe_sample["lr_gs_buffers"]["diagnostics"].shape[2])
    model = HRGSRefiner(gs_diag_channels=gs_diag_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.max_steps), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    start_step = 0
    if args.refiner_checkpoint:
        blob = torch.load(args.refiner_checkpoint, map_location="cpu")
        state_dict = blob.get("state_dict", blob)
        model.load_state_dict(state_dict, strict=False)
        start_step = int(blob.get("step", 0))
        optimizer_state = blob.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = blob.get("scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = blob.get("scaler")
        if scaler_state is not None and args.device.startswith("cuda"):
            scaler.load_state_dict(scaler_state)
        _log(f"[train-hrgs] loaded checkpoint at step {start_step}: {args.refiner_checkpoint}")

    loss_cfg = _build_loss_config(args)

    stats_path = output_dir / "train_log.jsonl"
    pbar = tqdm(range(start_step + 1, int(args.max_steps) + 1), desc="train_hrgsrefiner_v0")
    for step in pbar:
        try:
            sample_cpu = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            sample_cpu = next(loader_iter)

        sample = _ensure_batch_dim(_nested_to_device(sample_cpu, args.device))
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
            outputs = model(
                sr_images=sample["images"]["images_sr"],
                lr_up_images=sample["images"]["images_lr_up"],
                gs_buffers=sample["lr_gs_buffers"],
                cameras=sample["cameras"],
                vggt_prior=sample["vggt_prior"],
            )
            losses = compute_hrgsrefiner_losses(outputs, sample, cfg=loss_cfg)
            loss = losses["total_backprop"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        log_row = {
            "step": int(step),
            "scene_id": sample["scene_id"],
            "view_names": list(sample["view_names"]),
            "loss_total": float(loss.detach().item()),
            "loss_depth": float(losses["depth"].item()),
            "loss_normal": float(losses["normal"].item()),
            "loss_multiview": float(losses["multiview"].item()),
            "loss_delta": float(losses["delta"].item()),
            "loss_surface_mask": float(losses["surface_mask"].item()),
            "loss_update_mask": float(losses["update_mask"].item()),
            "loss_detail_mask": float(losses["detail_mask"].item()),
            "loss_confidence": float(losses["confidence"].item()),
            "loss_prior_color": float(losses["prior_color"].item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with stats_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")

        if step % int(args.log_every) == 0 or step == start_step + 1:
            pbar.set_postfix(
                loss=f"{log_row['loss_total']:.4f}",
                depth=f"{log_row['loss_depth']:.4f}",
                normal=f"{log_row['loss_normal']:.4f}",
            )
            _log(
                f"[train-hrgs] step={step} loss={log_row['loss_total']:.5f} "
                f"depth={log_row['loss_depth']:.5f} normal={log_row['loss_normal']:.5f} "
                f"mv={log_row['loss_multiview']:.5f}"
            )

        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            ckpt_path = ckpt_dir / f"hrgsrefiner_step_{step:06d}.pt"
            _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, step, args)
            _log(f"[train-hrgs] saved checkpoint: {ckpt_path}")

        if step % int(args.eval_every) == 0 or step == int(args.max_steps):
            model.eval()
            eval_summary = _export_eval_payload(model, sample, eval_dir / f"step_{step:06d}", step)
            _log(
                f"[train-hrgs] eval step={step} carriers={eval_summary['num_carriers']} "
                f"valid={eval_summary['num_valid_carriers']} "
                f"attach={eval_summary['mean_attach_strength']:.4f}"
            )


if __name__ == "__main__":
    main()
