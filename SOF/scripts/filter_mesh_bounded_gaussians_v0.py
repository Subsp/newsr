#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from joint_judge_mip_sof_v0 import stats_from_array


def sigmoid_np(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32, copy=False)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32, copy=False)


def inverse_sigmoid_np(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32, copy=False)


def tensor_to_numpy(payload: Dict[str, object], key: str, default: float = 0.0) -> np.ndarray:
    if key not in payload:
        return np.full((0,), float(default), dtype=np.float32)
    value = payload[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def copy_render_config(src_model_path: Path, dst_model_path: Path) -> None:
    dst_model_path.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "config.json", "cameras.json"):
        src = src_model_path / name
        if src.exists():
            shutil.copy2(src, dst_model_path / name)


def select_ids(
    *,
    confidence: np.ndarray,
    mip_support: np.ndarray,
    d_norm: np.ndarray,
    sigma_norm: np.ndarray,
    min_confidence: float,
    min_mip_support: float,
    max_d_norm: float,
    max_sigma_normal_norm: float,
    max_proxy_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = confidence >= float(min_confidence)
    if mip_support.shape[0] == confidence.shape[0]:
        mask &= mip_support >= float(min_mip_support)
    if d_norm.shape[0] == confidence.shape[0] and float(max_d_norm) > 0:
        mask &= d_norm <= float(max_d_norm)
    if sigma_norm.shape[0] == confidence.shape[0] and float(max_sigma_normal_norm) > 0:
        mask &= sigma_norm <= float(max_sigma_normal_norm)

    ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if ids.size == 0:
        return ids, mask
    mip = mip_support if mip_support.shape[0] == confidence.shape[0] else np.ones_like(confidence)
    d = d_norm if d_norm.shape[0] == confidence.shape[0] else np.zeros_like(confidence)
    sigma = sigma_norm if sigma_norm.shape[0] == confidence.shape[0] else np.zeros_like(confidence)
    score = confidence * np.clip(mip, 0.0, 1.0) / (1.0 + np.maximum(d, 0.0) + np.maximum(sigma, 0.0))
    if int(max_proxy_count) > 0 and ids.size > int(max_proxy_count):
        order = np.argsort(-score[ids], kind="stable")[: int(max_proxy_count)]
        ids = ids[order]
        limited_mask = np.zeros_like(mask)
        limited_mask[ids] = True
        mask = limited_mask
    return ids, mask


def filter_payload(payload: Dict[str, object], selected_ids: np.ndarray, proxy_total: int) -> Dict[str, object]:
    selected_t = torch.from_numpy(selected_ids.astype(np.int64, copy=False))
    out: Dict[str, object] = {
        "version": "mesh_bounded_gaussians_v0_filtered",
        "selected_proxy_idx": selected_t,
    }
    for key, value in payload.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(proxy_total):
            out[key] = value[selected_t]
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter an existing mesh-boundedGS proxy model into a stricter support layer.")
    parser.add_argument("--input_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--iteration", type=int, default=34000)
    parser.add_argument("--payload_path", default="")
    parser.add_argument("--min_confidence", type=float, default=0.65)
    parser.add_argument("--min_mip_support", type=float, default=0.80)
    parser.add_argument("--max_d_norm", type=float, default=0.85)
    parser.add_argument("--max_sigma_normal_norm", type=float, default=0.95)
    parser.add_argument("--max_proxy_count", type=int, default=35000)
    parser.add_argument("--opacity_scale", type=float, default=0.20)
    parser.add_argument("--alpha_max", type=float, default=0.006)
    args = parser.parse_args()

    input_model_path = Path(args.input_model_path).expanduser().resolve()
    output_model_path = Path(args.output_model_path).expanduser().resolve()
    iteration = int(args.iteration)
    payload_path = Path(args.payload_path).expanduser().resolve() if str(args.payload_path).strip() else input_model_path / "mesh_bounded_gaussians_v0.pt"
    input_ply = input_model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    input_tags = input_ply.parent / "gaussian_tags.pt"
    if not input_ply.is_file():
        raise FileNotFoundError(f"input mesh-bounded PLY not found: {input_ply}")
    if not payload_path.is_file():
        raise FileNotFoundError(f"mesh-bounded payload not found: {payload_path}")

    payload = torch.load(payload_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {payload_path}, got {type(payload)!r}")
    confidence = tensor_to_numpy(payload, "proxy_confidence")
    mip_support = tensor_to_numpy(payload, "proxy_mip_support")
    d_norm = tensor_to_numpy(payload, "proxy_d_norm")
    sigma_norm = tensor_to_numpy(payload, "proxy_sigma_normal_norm")
    if confidence.size == 0:
        raise ValueError("payload does not contain proxy_confidence")

    plydata = PlyData.read(str(input_ply))
    vertex = plydata["vertex"].data
    if int(vertex.shape[0]) != int(confidence.shape[0]):
        raise ValueError(f"PLY/payload length mismatch: ply={vertex.shape[0]} confidence={confidence.shape[0]}")

    selected_ids, selected_mask = select_ids(
        confidence=confidence.astype(np.float32, copy=False),
        mip_support=mip_support.astype(np.float32, copy=False),
        d_norm=d_norm.astype(np.float32, copy=False),
        sigma_norm=sigma_norm.astype(np.float32, copy=False),
        min_confidence=float(args.min_confidence),
        min_mip_support=float(args.min_mip_support),
        max_d_norm=float(args.max_d_norm),
        max_sigma_normal_norm=float(args.max_sigma_normal_norm),
        max_proxy_count=int(args.max_proxy_count),
    )
    if selected_ids.size == 0:
        raise RuntimeError("No proxy gaussians survived strict mesh-bounded filtering.")

    subset = vertex[selected_ids].copy()
    property_names = set(subset.dtype.names or ())
    original_alpha = np.zeros((selected_ids.size,), dtype=np.float32)
    output_alpha = np.zeros((selected_ids.size,), dtype=np.float32)
    if "opacity" in property_names:
        original_alpha = sigmoid_np(np.asarray(subset["opacity"], dtype=np.float32))
        output_alpha = np.minimum(original_alpha * float(args.opacity_scale), float(args.alpha_max)).astype(np.float32)
        subset["opacity"] = inverse_sigmoid_np(output_alpha)

    copy_render_config(input_model_path, output_model_path)
    point_dir = output_model_path / "point_cloud" / f"iteration_{iteration}"
    point_dir.mkdir(parents=True, exist_ok=True)
    output_ply = point_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(subset, "vertex")], text=plydata.text).write(str(output_ply))

    if input_tags.is_file():
        tag_payload = torch.load(input_tags, map_location="cpu")
        if isinstance(tag_payload, dict):
            selected_t = torch.from_numpy(selected_ids.astype(np.int64, copy=False))
            filtered_tags = {}
            for key, value in tag_payload.items():
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(vertex.shape[0]):
                    filtered_tags[key] = value[selected_t]
                else:
                    filtered_tags[key] = value
            torch.save(filtered_tags, point_dir / "gaussian_tags.pt")

    filtered_payload = filter_payload(payload, selected_ids, int(vertex.shape[0]))
    filtered_payload["filter_selected_proxy_mask"] = torch.from_numpy(selected_mask.astype(bool, copy=False))
    filtered_payload_path = output_model_path / "mesh_bounded_gaussians_v0_filtered.pt"
    torch.save(filtered_payload, filtered_payload_path)
    summary = {
        "version": "filter_mesh_bounded_gaussians_v0",
        "input_model_path": str(input_model_path),
        "input_payload_path": str(payload_path),
        "output_model_path": str(output_model_path),
        "iteration": int(iteration),
        "input_proxy_count": int(vertex.shape[0]),
        "selected_proxy_count": int(selected_ids.size),
        "selected_ratio": float(selected_ids.size / max(int(vertex.shape[0]), 1)),
        "thresholds": {
            "min_confidence": float(args.min_confidence),
            "min_mip_support": float(args.min_mip_support),
            "max_d_norm": float(args.max_d_norm),
            "max_sigma_normal_norm": float(args.max_sigma_normal_norm),
            "max_proxy_count": int(args.max_proxy_count),
            "opacity_scale": float(args.opacity_scale),
            "alpha_max": float(args.alpha_max),
        },
        "stats": {
            "confidence_selected": stats_from_array(confidence[selected_ids].astype(np.float32)),
            "mip_support_selected": stats_from_array(mip_support[selected_ids].astype(np.float32)),
            "d_norm_selected": stats_from_array(d_norm[selected_ids].astype(np.float32)),
            "sigma_normal_norm_selected": stats_from_array(sigma_norm[selected_ids].astype(np.float32)),
            "original_alpha_selected": stats_from_array(original_alpha.astype(np.float32)),
            "output_alpha_selected": stats_from_array(output_alpha.astype(np.float32)),
        },
        "paths": {
            "output_ply": str(output_ply),
            "filtered_payload": str(filtered_payload_path),
            "summary": str(output_model_path / "filter_mesh_bounded_gaussians_v0_summary.json"),
        },
    }
    summary_path = output_model_path / "filter_mesh_bounded_gaussians_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
