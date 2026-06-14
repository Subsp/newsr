from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sh_utils import SH2RGB


def _load_model_params(path: str):
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, (tuple, list)) or len(blob) < 2:
        raise RuntimeError(f"Unsupported checkpoint format: {type(blob)!r}")
    model_params = blob[0]
    iteration = int(blob[1])
    return model_params, iteration


def _features_dc_to_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    if features_dc.ndim != 3:
        raise ValueError(f"Unexpected features_dc shape: {tuple(features_dc.shape)}")
    if features_dc.shape[1] == 1 and features_dc.shape[2] == 3:
        dc = features_dc[:, 0, :]
    elif features_dc.shape[1] == 3 and features_dc.shape[2] == 1:
        dc = features_dc[:, :, 0]
    else:
        raise ValueError(f"Unsupported features_dc shape: {tuple(features_dc.shape)}")
    return SH2RGB(dc).clamp(0.0, 1.0)


def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _sample_indices(scores: torch.Tensor, max_count: int, seed: int) -> torch.Tensor:
    total = int(scores.shape[0])
    if max_count <= 0 or total <= max_count:
        return torch.arange(total, dtype=torch.long)
    generator = torch.Generator(device=scores.device)
    generator.manual_seed(int(seed))
    probs = scores.clamp_min(1e-8)
    probs = probs / probs.sum()
    return torch.multinomial(probs, num_samples=int(max_count), replacement=False, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dummy carrier payload by sampling a Gaussian checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_opacity", type=float, default=0.05)
    parser.add_argument("--max_scale", type=float, default=0.25)
    parser.add_argument("--thickness_ratio", type=float, default=0.05)
    parser.add_argument("--confidence_scale", type=float, default=1.0)
    parser.add_argument("--view_count", type=int, default=2)
    args = parser.parse_args()

    model_params, iteration = _load_model_params(args.checkpoint)
    xyz = model_params[1].detach().float()
    features_dc = model_params[2].detach().float()
    scaling_raw = model_params[4].detach().float()
    rotation_raw = model_params[5].detach().float()
    opacity_raw = model_params[6].detach().float().reshape(-1)

    colors = _features_dc_to_rgb(features_dc)
    scales = torch.exp(scaling_raw).clamp_min(1e-6)
    rotations = _quat_to_matrix(rotation_raw)
    opacity = torch.sigmoid(opacity_raw)

    finite_mask = (
        torch.isfinite(xyz).all(dim=-1)
        & torch.isfinite(colors).all(dim=-1)
        & torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(rotations).all(dim=(-2, -1))
        & torch.isfinite(opacity)
    )
    valid = finite_mask & (opacity >= float(args.min_opacity))
    if int(valid.sum().item()) == 0:
        raise RuntimeError("No valid Gaussians left after filtering.")

    xyz = xyz[valid]
    colors = colors[valid]
    scales = scales[valid]
    rotations = rotations[valid]
    opacity = opacity[valid]

    sample_ids = _sample_indices(opacity, int(args.max_count), int(args.seed))
    xyz = xyz[sample_ids]
    colors = colors[sample_ids]
    scales = scales[sample_ids]
    rotations = rotations[sample_ids]
    opacity = opacity[sample_ids]

    tangent_u = rotations[:, :, 0]
    tangent_v = rotations[:, :, 1]
    normals = rotations[:, :, 2]
    scale_u = scales[:, 0:1].clamp(max=float(args.max_scale))
    scale_v = scales[:, 1:2].clamp(max=float(args.max_scale))
    scale_n = torch.minimum(scale_u, scale_v) * float(args.thickness_ratio)
    confidence = (opacity[:, None] * float(args.confidence_scale)).clamp(0.0, 1.0)
    disagreement = torch.zeros_like(confidence)
    view_count = torch.full((xyz.shape[0], 1), int(args.view_count), dtype=torch.int32)
    valid_mask = torch.ones((xyz.shape[0], 1), dtype=torch.bool)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        centers=xyz.cpu().numpy().astype(np.float32),
        normals=normals.cpu().numpy().astype(np.float32),
        tangent_u=tangent_u.cpu().numpy().astype(np.float32),
        tangent_v=tangent_v.cpu().numpy().astype(np.float32),
        scale_u=scale_u.cpu().numpy().astype(np.float32),
        scale_v=scale_v.cpu().numpy().astype(np.float32),
        scale_n=scale_n.cpu().numpy().astype(np.float32),
        fused_rgb=colors.cpu().numpy().astype(np.float32),
        confidence=confidence.cpu().numpy().astype(np.float32),
        disagreement=disagreement.cpu().numpy().astype(np.float32),
        view_count=view_count.cpu().numpy(),
        valid_mask=valid_mask.cpu().numpy(),
    )
    print(f"saved: {out_path}")
    print(f"source_iteration: {iteration}")
    print(f"selected_count: {int(xyz.shape[0])}")
    print(f"opacity_mean: {float(opacity.mean().item()):.6f}")


if __name__ == "__main__":
    main()
