from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_model import GaussianModel
from train_mip_to_sof_surface_v0 import copy_render_config, resolve_iteration


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = torch.clamp(probability, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probability / torch.clamp(1.0 - probability, min=1e-6))


def _load_model(model_path: Path, iteration: int, sh_degree: int) -> GaussianModel:
    ply_path = model_path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply_path}")
    model = GaussianModel(int(sh_degree))
    model.load_ply(str(ply_path))
    tags_path = ply_path.parent / "gaussian_tags.pt"
    if tags_path.exists():
        model.load_tracking_metadata(str(tags_path))
    return model


def _stats(values: np.ndarray) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarantine starburst candidate Gaussians by gating rest-SH and optical thickness."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--score_payload_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--use_payload_candidate_mask", action="store_true")
    parser.add_argument(
        "--candidate_mode",
        choices=("intersection", "threshold", "payload", "union"),
        default="intersection",
        help=(
            "How to combine score thresholds with the payload candidate mask. "
            "intersection preserves the original conservative behavior when --use_payload_candidate_mask is set."
        ),
    )
    parser.add_argument("--score_key", default="starburst_score")
    parser.add_argument("--candidate_key", default="starburst_candidate")
    parser.add_argument("--min_starburst_score", type=float, default=0.12)
    parser.add_argument("--min_unsupported_score", type=float, default=0.04)
    parser.add_argument("--min_geometry_risk", type=float, default=0.08)
    parser.add_argument("--min_visible_count", type=int, default=1)
    parser.add_argument("--max_quarantine_fraction", type=float, default=0.015)
    parser.add_argument("--max_quarantine_count", type=int, default=30000)
    parser.add_argument("--dc_scale", type=float, default=1.0)
    parser.add_argument("--rest_scale", type=float, default=0.10)
    parser.add_argument("--tau_scale", type=float, default=0.35)
    parser.add_argument("--min_alpha", type=float, default=1e-6)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    score_payload_path = Path(args.score_payload_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    payload = torch.load(score_payload_path, map_location="cpu")
    if str(payload.get("version", "")) != "starburst_gaussian_scores_v0":
        raise ValueError(f"Unsupported starburst payload version in {score_payload_path}")

    iteration = resolve_iteration(model_path, int(args.iteration))
    gaussians = _load_model(model_path, iteration, int(args.sh_degree))
    count = int(gaussians.get_xyz.shape[0])

    score = torch.as_tensor(payload[str(args.score_key)]).reshape(-1).float()
    if int(score.shape[0]) != count:
        raise ValueError(f"Payload/model length mismatch for {args.score_key}: {score.shape[0]} vs {count}")
    unsupported = torch.as_tensor(payload.get("unsupported_score", torch.zeros_like(score[:, None]))).reshape(-1).float()
    geometry_risk = torch.as_tensor(payload.get("geometry_risk", torch.zeros_like(score[:, None]))).reshape(-1).float()
    visible_count = torch.as_tensor(payload.get("visible_count", torch.zeros_like(score[:, None]))).reshape(-1).long()
    payload_candidate = torch.zeros((count,), dtype=torch.bool)
    if bool(args.use_payload_candidate_mask):
        payload_candidate = torch.as_tensor(
            payload.get(str(args.candidate_key), torch.zeros((count, 1), dtype=torch.bool))
        ).reshape(-1).bool()

    threshold_candidate = (score >= float(args.min_starburst_score))
    threshold_candidate &= (unsupported >= float(args.min_unsupported_score))
    threshold_candidate &= (geometry_risk >= float(args.min_geometry_risk))
    threshold_candidate &= (visible_count >= int(args.min_visible_count))
    if str(args.candidate_mode) == "threshold":
        candidate = threshold_candidate
    elif str(args.candidate_mode) == "payload":
        candidate = payload_candidate
    elif str(args.candidate_mode) == "union":
        candidate = threshold_candidate | payload_candidate
    else:
        candidate = threshold_candidate
        if bool(args.use_payload_candidate_mask):
            candidate &= payload_candidate

    candidate_ids = torch.nonzero(candidate, as_tuple=False).squeeze(1)
    candidate_count_before_cap = int(candidate_ids.numel())
    if candidate_count_before_cap > 0:
        max_by_fraction = int(max(0, round(float(args.max_quarantine_fraction) * float(count))))
        max_count = candidate_count_before_cap
        if max_by_fraction > 0:
            max_count = min(max_count, max_by_fraction)
        if int(args.max_quarantine_count) > 0:
            max_count = min(max_count, int(args.max_quarantine_count))
        if max_count <= 0:
            selected_ids = candidate_ids[:0]
        elif candidate_count_before_cap > max_count:
            order = torch.argsort(score[candidate_ids], descending=True, stable=True)[:max_count]
            selected_ids = candidate_ids[order]
        else:
            selected_ids = candidate_ids
    else:
        selected_ids = candidate_ids

    selected_count = int(selected_ids.numel())
    device_ids = selected_ids.to(device=gaussians._opacity.device)
    original_opacity = gaussians._opacity.detach().clone()
    original_features_dc = gaussians._features_dc.detach().clone()
    original_features_rest = gaussians._features_rest.detach().clone()

    if selected_count > 0:
        with torch.no_grad():
            gaussians._features_dc[device_ids] = gaussians._features_dc[device_ids] * float(args.dc_scale)
            gaussians._features_rest[device_ids] = gaussians._features_rest[device_ids] * float(args.rest_scale)
            alpha = torch.sigmoid(gaussians._opacity[device_ids])
            tau = -torch.log(torch.clamp(1.0 - alpha, min=1e-6))
            alpha_eff = 1.0 - torch.exp(-torch.clamp(tau * float(args.tau_scale), min=0.0))
            alpha_eff = torch.clamp(alpha_eff, min=float(args.min_alpha), max=1.0 - 1e-6)
            gaussians._opacity[device_ids] = _logit(alpha_eff)

    copy_render_config(model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{int(iteration)}"
    point_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(point_dir / "point_cloud.ply"))
    gaussians.save_tracking_metadata(str(point_dir / "gaussian_tags.pt"))

    masks_dir = output_model_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    selected_mask = torch.zeros((count,), dtype=torch.bool)
    if selected_count > 0:
        selected_mask[selected_ids.detach().cpu()] = True
    torch.save(selected_mask, masks_dir / "star_quarantine_input_mask.pt")
    torch.save(selected_ids.detach().cpu().to(torch.int64), masks_dir / "star_quarantine_input_idx.pt")

    quarantine_payload = {
        "version": "star_quarantine_v0",
        "model_path": str(model_path),
        "score_payload_path": str(score_payload_path),
        "output_model_path": str(output_model_path),
        "iteration": int(iteration),
        "selected_ids": selected_ids.detach().cpu().to(torch.int64),
        "selected_mask": selected_mask,
        "score": score.detach().cpu(),
        "unsupported_score": unsupported.detach().cpu(),
        "geometry_risk": geometry_risk.detach().cpu(),
        "visible_count": visible_count.detach().cpu(),
        "original_opacity": original_opacity[selected_ids.to(device=original_opacity.device)].detach().cpu()
        if selected_count > 0
        else torch.empty((0, *original_opacity.shape[1:]), dtype=original_opacity.dtype),
        "original_features_dc": original_features_dc[selected_ids.to(device=original_features_dc.device)].detach().cpu()
        if selected_count > 0
        else torch.empty((0, *original_features_dc.shape[1:]), dtype=original_features_dc.dtype),
        "original_features_rest": original_features_rest[selected_ids.to(device=original_features_rest.device)].detach().cpu()
        if selected_count > 0
        else torch.empty((0, *original_features_rest.shape[1:]), dtype=original_features_rest.dtype),
        "args": vars(args),
    }
    torch.save(quarantine_payload, output_model_path / "star_quarantine_payload.pt")

    summary = {
        "version": "star_quarantine_v0",
        "model_path": str(model_path),
        "score_payload_path": str(score_payload_path),
        "output_model_path": str(output_model_path),
        "iteration": int(iteration),
        "input_gaussians": int(count),
        "candidate_count_before_cap": int(candidate_count_before_cap),
        "selected_count": int(selected_count),
        "selected_ratio": float(selected_count / max(count, 1)),
        "dc_scale": float(args.dc_scale),
        "rest_scale": float(args.rest_scale),
        "tau_scale": float(args.tau_scale),
        "use_payload_candidate_mask": bool(args.use_payload_candidate_mask),
        "candidate_mode": str(args.candidate_mode),
        "selected_score_stats": _stats(score[selected_ids].detach().cpu().numpy() if selected_count > 0 else np.empty((0,), dtype=np.float32)),
        "selected_unsupported_stats": _stats(unsupported[selected_ids].detach().cpu().numpy() if selected_count > 0 else np.empty((0,), dtype=np.float32)),
        "selected_geometry_risk_stats": _stats(geometry_risk[selected_ids].detach().cpu().numpy() if selected_count > 0 else np.empty((0,), dtype=np.float32)),
    }
    summary_path = output_model_path / "star_quarantine_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] quarantined model : {point_dir / 'point_cloud.ply'}")
    print(f"[done] payload           : {output_model_path / 'star_quarantine_payload.pt'}")
    print(f"[done] summary           : {summary_path}")


if __name__ == "__main__":
    main()
