#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from joint_judge_mip_sof_v0 import stats_from_array
from train_mip_to_sof_surface_v0 import (
    load_cameras_for_split,
    load_model_ply,
    normalize_image_name,
    resolve_iteration,
    select_uniform,
)
from utils.prior_fusion import project_points_camera
from utils.sh_utils import SH2RGB


def save_gray(path: Path, image_hw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_hw.astype(np.float32, copy=False), 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, image_chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_chw.astype(np.float32, copy=False), 0.0, 1.0)
    if array.ndim == 3 and int(array.shape[0]) == 3:
        array = np.transpose(array, (1, 2, 0))
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def log_normalize(image_hw: np.ndarray, gain: float = 10.0) -> np.ndarray:
    gain = max(float(gain), 1e-8)
    return np.clip(np.log1p(gain * image_hw) / np.log1p(gain), 0.0, 1.0).astype(np.float32, copy=False)


def model_dc_rgb_np(gaussians) -> np.ndarray:
    with torch.no_grad():
        dc = gaussians._features_dc.detach()
        if gaussians.use_SBs:
            rgb = torch.clamp(dc[:, :3], 0.0, 1.0)
        else:
            rgb = torch.clamp(SH2RGB(dc[:, 0, :]), 0.0, 1.0)
    return rgb.detach().cpu().numpy().astype(np.float32, copy=False)


def _payload_array(payload: Dict[str, object], key: str, count: int | None = None, channels: int | None = None) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim < 1:
        return None
    if count is not None and int(array.shape[0]) != int(count):
        return None
    if channels is not None and (array.ndim != 2 or int(array.shape[1]) != int(channels)):
        return None
    return array


def load_proxy_payload(model_path: Path, count: int, fallback_rgb: np.ndarray) -> Dict[str, np.ndarray | bool | int]:
    payload_path = model_path / "mesh_bounded_gaussians_v0.pt"
    rgb = fallback_rgb
    face_id = np.full((count,), -1, dtype=np.int64)
    source_idx = np.full((count,), -1, dtype=np.int64)
    proxy_confidence = np.ones((count,), dtype=np.float32)
    proxy_color_confidence = np.ones((count,), dtype=np.float32)
    proxy_mip_support = np.ones((count,), dtype=np.float32)
    proxy_opacity = np.ones((count,), dtype=np.float32)
    source_point_p = np.zeros((count, 3), dtype=np.float32)
    mesh_point_q = np.zeros((count, 3), dtype=np.float32)
    mesh_normal = np.zeros((count, 3), dtype=np.float32)
    mesh_barycentric = np.tile(np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32), (count, 1))
    signed_offset = np.zeros((count,), dtype=np.float32)
    clipped_signed_offset = np.zeros((count,), dtype=np.float32)
    tau_surface = np.zeros((count,), dtype=np.float32)
    d_norm = np.zeros((count,), dtype=np.float32)
    sample_t = np.zeros((count,), dtype=np.float32)
    local_mesh_edge = np.zeros((count,), dtype=np.float32)
    correspondence_loaded = False
    face_count = 0
    payload_loaded = False
    if payload_path.is_file():
        payload = torch.load(str(payload_path), map_location="cpu")
        if isinstance(payload, dict):
            payload_loaded = True
            arr = _payload_array(payload, "proxy_rgb", count=count, channels=3)
            if arr is not None:
                rgb = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
            arr = _payload_array(payload, "proxy_face_id", count=count)
            if arr is not None:
                face_id = arr.reshape(-1).astype(np.int64, copy=False)
            arr = _payload_array(payload, "source_idx", count=count)
            if arr is not None:
                source_idx = arr.reshape(-1).astype(np.int64, copy=False)
            arr = _payload_array(payload, "proxy_confidence", count=count)
            if arr is not None:
                proxy_confidence = np.clip(arr.reshape(-1).astype(np.float32, copy=False), 0.0, 1.0)
            arr = _payload_array(payload, "proxy_color_confidence", count=count)
            if arr is not None:
                proxy_color_confidence = np.clip(arr.reshape(-1).astype(np.float32, copy=False), 0.0, 1.0)
            arr = _payload_array(payload, "proxy_mip_support", count=count)
            if arr is not None:
                proxy_mip_support = np.clip(arr.reshape(-1).astype(np.float32, copy=False), 0.0, 1.0)
            arr = _payload_array(payload, "proxy_opacity", count=count)
            if arr is not None:
                proxy_opacity = np.clip(arr.reshape(-1).astype(np.float32, copy=False), 0.0, 1.0)
            arr = _payload_array(payload, "proxy_source_point_p", count=count, channels=3)
            if arr is not None:
                source_point_p = arr.astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_mesh_point_q", count=count, channels=3)
            if arr is not None:
                mesh_point_q = arr.astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_mesh_normal", count=count, channels=3)
            if arr is not None:
                mesh_normal = arr.astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_mesh_barycentric", count=count, channels=3)
            if arr is not None:
                mesh_barycentric = arr.astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_signed_offset", count=count)
            if arr is not None:
                signed_offset = arr.reshape(-1).astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_clipped_signed_offset", count=count)
            if arr is not None:
                clipped_signed_offset = arr.reshape(-1).astype(np.float32, copy=False)
            arr = _payload_array(payload, "proxy_tau_surface", count=count)
            if arr is not None:
                tau_surface = arr.reshape(-1).astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_d_norm", count=count)
            if arr is not None:
                d_norm = arr.reshape(-1).astype(np.float32, copy=False)
                correspondence_loaded = True
            arr = _payload_array(payload, "proxy_sample_t", count=count)
            if arr is not None:
                sample_t = arr.reshape(-1).astype(np.float32, copy=False)
            arr = _payload_array(payload, "proxy_local_mesh_edge", count=count)
            if arr is not None:
                local_mesh_edge = arr.reshape(-1).astype(np.float32, copy=False)
            arr = _payload_array(payload, "face_confidence_product")
            if arr is not None:
                face_count = int(arr.shape[0])
    if face_count <= 0 and np.any(face_id >= 0):
        face_count = int(np.max(face_id)) + 1
    return {
        "payload_loaded": payload_loaded,
        "rgb": rgb,
        "face_id": face_id,
        "source_idx": source_idx,
        "proxy_confidence": proxy_confidence,
        "proxy_color_confidence": proxy_color_confidence,
        "proxy_mip_support": proxy_mip_support,
        "proxy_opacity": proxy_opacity,
        "source_point_p": source_point_p,
        "mesh_point_q": mesh_point_q,
        "mesh_normal": mesh_normal,
        "mesh_barycentric": mesh_barycentric,
        "signed_offset": signed_offset,
        "clipped_signed_offset": clipped_signed_offset,
        "tau_surface": tau_surface,
        "d_norm": d_norm,
        "sample_t": sample_t,
        "local_mesh_edge": local_mesh_edge,
        "correspondence_loaded": correspondence_loaded,
        "face_count": face_count,
    }


def build_proxy_weight(
    *,
    proxy_confidence: np.ndarray,
    proxy_color_confidence: np.ndarray,
    proxy_mip_support: np.ndarray,
    proxy_opacity: np.ndarray,
    confidence_power: float,
    color_confidence_power: float,
    mip_support_power: float,
    opacity_power: float,
) -> np.ndarray:
    weight = np.ones_like(proxy_confidence, dtype=np.float32)
    for values, power in (
        (proxy_confidence, confidence_power),
        (proxy_color_confidence, color_confidence_power),
        (proxy_mip_support, mip_support_power),
        (proxy_opacity / max(float(np.percentile(proxy_opacity, 95.0)), 1e-6), opacity_power),
    ):
        if float(power) > 0.0:
            weight *= np.power(np.clip(values, 0.0, 1.0), float(power)).astype(np.float32, copy=False)
    return np.clip(weight, 0.0, 1.0).astype(np.float32, copy=False)


def project_visible_centers(
    *,
    camera,
    xyz: np.ndarray,
    proxy_rgb: np.ndarray,
    proxy_weight: np.ndarray,
    color_sigma: float,
    depth_min: float,
) -> Dict[str, np.ndarray | float | Dict[str, object]]:
    projected, valid = project_points_camera(camera, xyz, depth_min=float(depth_min), margin=0)
    height = int(camera.image_height)
    width = int(camera.image_width)
    density = np.zeros((height, width), dtype=np.float32)
    nearest_idx = np.full((height * width,), -1, dtype=np.int64)
    if np.any(valid):
        ids = np.flatnonzero(valid).astype(np.int64, copy=False)
        x = np.clip(np.rint(projected[ids, 0]).astype(np.int64), 0, width - 1)
        y = np.clip(np.rint(projected[ids, 1]).astype(np.int64), 0, height - 1)
        pix = y * width + x
        np.add.at(density.reshape(-1), pix, 1.0)
        order = np.argsort(projected[ids, 2], kind="stable")[::-1]
        nearest_idx[pix[order]] = ids[order]

    hit = nearest_idx >= 0
    hit_ids = nearest_idx[hit].astype(np.int64, copy=False)
    ref = camera.original_image[:3].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
    ref_flat = np.transpose(ref, (1, 2, 0)).reshape(-1, 3)
    err_map = np.zeros((height, width), dtype=np.float32)
    score_map = np.zeros((height, width), dtype=np.float32)
    rgb_map = np.zeros((3, height, width), dtype=np.float32)
    errors = np.empty((0,), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    hit_ref_rgb = np.empty((0, 3), dtype=np.float32)
    if hit_ids.size > 0:
        flat_rgb = np.zeros((height * width, 3), dtype=np.float32)
        flat_rgb[hit] = proxy_rgb[hit_ids]
        rgb_map = np.transpose(flat_rgb.reshape(height, width, 3), (2, 0, 1))
        hit_ref_rgb = ref_flat[hit].astype(np.float32, copy=False)
        errors = np.mean(np.abs(proxy_rgb[hit_ids] - hit_ref_rgb), axis=1).astype(np.float32, copy=False)
        color_score = np.exp(-np.square(errors / max(float(color_sigma), 1e-6))).astype(np.float32, copy=False)
        scores = np.clip(color_score * proxy_weight[hit_ids], 0.0, 1.0).astype(np.float32, copy=False)
        err_map.reshape(-1)[hit] = errors
        score_map.reshape(-1)[hit] = scores
    return {
        "valid_ids": np.flatnonzero(valid).astype(np.int64, copy=False),
        "hit_ids": hit_ids,
        "hit_errors": errors,
        "hit_scores": scores,
        "hit_ref_rgb": hit_ref_rgb,
        "density": density,
        "err_map": err_map,
        "score_map": score_map,
        "rgb_map": rgb_map,
        "valid_ratio": float(np.mean(valid.astype(np.float32))),
        "hit_ratio": float(np.mean(hit.astype(np.float32))),
        "density_stats": stats_from_array(density[density > 0]),
        "color_l1_error_stats": stats_from_array(errors),
        "score_stats": stats_from_array(scores),
    }


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(
        num,
        np.maximum(den, 1e-8),
        out=np.zeros_like(num, dtype=np.float32),
        where=den > 0,
    ).astype(np.float32, copy=False)


def aggregate_faces(
    *,
    face_id: np.ndarray,
    face_count: int,
    proxy_score: np.ndarray,
    proxy_trusted: np.ndarray,
    proxy_hit_count: np.ndarray,
    proxy_color_error_mean: np.ndarray,
    proxy_gate_mean: np.ndarray,
) -> Dict[str, np.ndarray]:
    if int(face_count) <= 0:
        return {
            "face_proxy_count": np.zeros((0,), dtype=np.int32),
            "face_observation_count": np.zeros((0,), dtype=np.int32),
            "face_visible_proxy_count": np.zeros((0,), dtype=np.int32),
            "face_trusted_proxy_count": np.zeros((0,), dtype=np.int32),
            "face_evidence_score": np.zeros((0,), dtype=np.float32),
            "face_trusted_ratio": np.zeros((0,), dtype=np.float32),
            "face_color_l1_mean": np.zeros((0,), dtype=np.float32),
            "face_gate_mean": np.zeros((0,), dtype=np.float32),
        }
    valid_face = (face_id >= 0) & (face_id < int(face_count))
    face_proxy_count = np.bincount(face_id[valid_face], minlength=int(face_count)).astype(np.int32)
    visible_proxy = valid_face & (proxy_hit_count > 0)
    trusted_proxy = valid_face & proxy_trusted
    face_visible_proxy_count = np.bincount(face_id[visible_proxy], minlength=int(face_count)).astype(np.int32)
    face_trusted_proxy_count = np.bincount(face_id[trusted_proxy], minlength=int(face_count)).astype(np.int32)

    face_observation_count = np.zeros((int(face_count),), dtype=np.int32)
    np.add.at(face_observation_count, face_id[valid_face], proxy_hit_count[valid_face].astype(np.int32, copy=False))

    face_score_sum = np.zeros((int(face_count),), dtype=np.float32)
    face_error_sum = np.zeros((int(face_count),), dtype=np.float32)
    face_gate_sum = np.zeros((int(face_count),), dtype=np.float32)
    np.add.at(face_score_sum, face_id[visible_proxy], proxy_score[visible_proxy])
    np.add.at(face_error_sum, face_id[visible_proxy], proxy_color_error_mean[visible_proxy])
    np.add.at(face_gate_sum, face_id[visible_proxy], proxy_gate_mean[visible_proxy])

    return {
        "face_proxy_count": face_proxy_count,
        "face_observation_count": face_observation_count,
        "face_visible_proxy_count": face_visible_proxy_count,
        "face_trusted_proxy_count": face_trusted_proxy_count,
        "face_evidence_score": safe_divide(face_score_sum, face_visible_proxy_count.astype(np.float32)),
        "face_trusted_ratio": safe_divide(face_trusted_proxy_count.astype(np.float32), face_proxy_count.astype(np.float32)),
        "face_color_l1_mean": safe_divide(face_error_sum, face_visible_proxy_count.astype(np.float32)),
        "face_gate_mean": safe_divide(face_gate_sum, face_visible_proxy_count.astype(np.float32)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accumulate mesh-boundedGS center-projection evidence back to proxy and mesh-face confidence."
    )
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--color_sigma", type=float, default=0.16)
    parser.add_argument("--proxy_confidence_power", type=float, default=0.5)
    parser.add_argument("--proxy_color_confidence_power", type=float, default=0.5)
    parser.add_argument("--proxy_mip_support_power", type=float, default=0.0)
    parser.add_argument("--proxy_opacity_power", type=float, default=0.0)
    parser.add_argument("--min_hit_views", type=int, default=3)
    parser.add_argument("--trusted_color_error", type=float, default=0.12)
    parser.add_argument("--trusted_gate", type=float, default=0.28)
    parser.add_argument("--depth_min", type=float, default=0.01)
    parser.add_argument("--save_debug_maps", action="store_true")
    parser.add_argument("--debug_max_views", type=int, default=16)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    iteration = resolve_iteration(model_path, int(args.iteration))
    gaussians = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    payload = load_proxy_payload(model_path, int(xyz.shape[0]), fallback_rgb=model_dc_rgb_np(gaussians))
    proxy_rgb = np.asarray(payload["rgb"], dtype=np.float32)
    face_id = np.asarray(payload["face_id"], dtype=np.int64)
    face_count = int(payload["face_count"])
    proxy_weight = build_proxy_weight(
        proxy_confidence=np.asarray(payload["proxy_confidence"], dtype=np.float32),
        proxy_color_confidence=np.asarray(payload["proxy_color_confidence"], dtype=np.float32),
        proxy_mip_support=np.asarray(payload["proxy_mip_support"], dtype=np.float32),
        proxy_opacity=np.asarray(payload["proxy_opacity"], dtype=np.float32),
        confidence_power=float(args.proxy_confidence_power),
        color_confidence_power=float(args.proxy_color_confidence_power),
        mip_support_power=float(args.proxy_mip_support_power),
        opacity_power=float(args.proxy_opacity_power),
    )

    cameras_all = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    cameras = select_uniform(cameras_all, int(args.max_views))
    if len(cameras) <= 0:
        raise RuntimeError("No cameras selected for mesh evidence accumulation.")
    debug_names = set()
    if bool(args.save_debug_maps) and int(args.debug_max_views) != 0:
        debug_cameras = select_uniform(cameras, int(args.debug_max_views))
        debug_names = {str(cam.image_name) for cam in debug_cameras}

    proxy_projected_count = np.zeros((xyz.shape[0],), dtype=np.int32)
    proxy_hit_count = np.zeros((xyz.shape[0],), dtype=np.int32)
    proxy_error_sum = np.zeros((xyz.shape[0],), dtype=np.float32)
    proxy_error_sq_sum = np.zeros((xyz.shape[0],), dtype=np.float32)
    proxy_error_min = np.full((xyz.shape[0],), np.inf, dtype=np.float32)
    proxy_gate_sum = np.zeros((xyz.shape[0],), dtype=np.float32)
    proxy_gate_max = np.zeros((xyz.shape[0],), dtype=np.float32)
    proxy_observed_rgb_sum = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    density_root = output_root / "debug_center_density_log"
    err_root = output_root / "debug_center_color_l1_error"
    score_root = output_root / "debug_center_evidence_score"
    rgb_root = output_root / "debug_center_rgb_zbuf"
    per_view: List[Dict[str, object]] = []
    for view_idx, camera in enumerate(tqdm(cameras, desc="accumulate mesh-bounded mesh evidence")):
        view = project_visible_centers(
            camera=camera,
            xyz=xyz,
            proxy_rgb=proxy_rgb,
            proxy_weight=proxy_weight,
            color_sigma=float(args.color_sigma),
            depth_min=float(args.depth_min),
        )
        valid_ids = np.asarray(view["valid_ids"], dtype=np.int64)
        hit_ids = np.asarray(view["hit_ids"], dtype=np.int64)
        errors = np.asarray(view["hit_errors"], dtype=np.float32)
        scores = np.asarray(view["hit_scores"], dtype=np.float32)
        if valid_ids.size > 0:
            proxy_projected_count[valid_ids] += 1
        if hit_ids.size > 0:
            np.add.at(proxy_hit_count, hit_ids, 1)
            np.add.at(proxy_error_sum, hit_ids, errors)
            np.add.at(proxy_error_sq_sum, hit_ids, errors * errors)
            np.minimum.at(proxy_error_min, hit_ids, errors)
            np.add.at(proxy_gate_sum, hit_ids, scores)
            np.maximum.at(proxy_gate_max, hit_ids, scores)
            np.add.at(proxy_observed_rgb_sum, hit_ids, np.asarray(view["hit_ref_rgb"], dtype=np.float32))

        image_name = normalize_image_name(str(camera.image_name))
        if bool(args.save_debug_maps) and str(camera.image_name) in debug_names:
            save_gray(density_root / f"{image_name}.png", log_normalize(np.asarray(view["density"], dtype=np.float32), gain=10.0))
            save_gray(err_root / f"{image_name}.png", np.clip(np.asarray(view["err_map"], dtype=np.float32) / 0.35, 0.0, 1.0))
            save_gray(score_root / f"{image_name}.png", np.asarray(view["score_map"], dtype=np.float32))
            save_rgb(rgb_root / f"{image_name}.png", np.asarray(view["rgb_map"], dtype=np.float32))
        per_view.append(
            {
                "source_view_index": int(view_idx),
                "image_name": image_name,
                "camera_image_name": str(camera.image_name),
                "valid_ratio": float(view["valid_ratio"]),
                "hit_ratio": float(view["hit_ratio"]),
                "density": view["density_stats"],
                "color_l1_error": view["color_l1_error_stats"],
                "evidence_score": view["score_stats"],
            }
        )

    observed = proxy_hit_count > 0
    proxy_color_error_mean = safe_divide(proxy_error_sum, proxy_hit_count.astype(np.float32))
    proxy_color_error_var = safe_divide(proxy_error_sq_sum, proxy_hit_count.astype(np.float32)) - proxy_color_error_mean * proxy_color_error_mean
    proxy_color_error_std = np.sqrt(np.clip(proxy_color_error_var, 0.0, None)).astype(np.float32, copy=False)
    proxy_color_error_min = proxy_error_min
    proxy_color_error_min[~observed] = 0.0
    proxy_gate_mean = safe_divide(proxy_gate_sum, proxy_hit_count.astype(np.float32))
    proxy_visible_ratio = safe_divide(proxy_hit_count.astype(np.float32), proxy_projected_count.astype(np.float32))
    proxy_observed_rgb_mean = safe_divide(proxy_observed_rgb_sum, proxy_hit_count.astype(np.float32)[:, None])
    proxy_color_score = np.exp(-np.square(proxy_color_error_mean / max(float(args.color_sigma), 1e-6))).astype(np.float32, copy=False)
    support_score = np.clip(
        proxy_hit_count.astype(np.float32) / max(float(args.min_hit_views), 1.0),
        0.0,
        1.0,
    )
    proxy_evidence_score = np.where(
        observed,
        np.clip(support_score * proxy_color_score * proxy_weight, 0.0, 1.0),
        0.0,
    ).astype(np.float32, copy=False)
    proxy_trusted = (
        (proxy_hit_count >= int(args.min_hit_views))
        & (proxy_color_error_mean <= float(args.trusted_color_error))
        & (proxy_gate_mean >= float(args.trusted_gate))
    )

    face = aggregate_faces(
        face_id=face_id,
        face_count=face_count,
        proxy_score=proxy_evidence_score,
        proxy_trusted=proxy_trusted,
        proxy_hit_count=proxy_hit_count,
        proxy_color_error_mean=proxy_color_error_mean,
        proxy_gate_mean=proxy_gate_mean,
    )

    payload_out = {
        "version": "mesh_bounded_mesh_evidence_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(iteration),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "proxy_xyz": torch.from_numpy(xyz.astype(np.float32, copy=False)),
        "proxy_rgb": torch.from_numpy(proxy_rgb.astype(np.float32, copy=False)),
        "proxy_observed_rgb_mean": torch.from_numpy(proxy_observed_rgb_mean.astype(np.float32, copy=False)),
        "proxy_source_idx": torch.from_numpy(np.asarray(payload["source_idx"], dtype=np.int64)),
        "proxy_face_id": torch.from_numpy(face_id.astype(np.int64, copy=False)),
        "correspondence_source_point_p": torch.from_numpy(np.asarray(payload["source_point_p"], dtype=np.float32)),
        "correspondence_mesh_point_q": torch.from_numpy(np.asarray(payload["mesh_point_q"], dtype=np.float32)),
        "correspondence_mesh_normal": torch.from_numpy(np.asarray(payload["mesh_normal"], dtype=np.float32)),
        "correspondence_mesh_barycentric": torch.from_numpy(np.asarray(payload["mesh_barycentric"], dtype=np.float32)),
        "correspondence_signed_offset": torch.from_numpy(np.asarray(payload["signed_offset"], dtype=np.float32)),
        "correspondence_clipped_signed_offset": torch.from_numpy(np.asarray(payload["clipped_signed_offset"], dtype=np.float32)),
        "correspondence_tau_surface": torch.from_numpy(np.asarray(payload["tau_surface"], dtype=np.float32)),
        "correspondence_d_norm": torch.from_numpy(np.asarray(payload["d_norm"], dtype=np.float32)),
        "correspondence_sample_t": torch.from_numpy(np.asarray(payload["sample_t"], dtype=np.float32)),
        "correspondence_local_mesh_edge": torch.from_numpy(np.asarray(payload["local_mesh_edge"], dtype=np.float32)),
        "correspondence_evidence_weight": torch.from_numpy(proxy_evidence_score),
        "correspondence_trusted_mask": torch.from_numpy(proxy_trusted.astype(bool, copy=False)),
        "proxy_base_confidence": torch.from_numpy(np.asarray(payload["proxy_confidence"], dtype=np.float32)),
        "proxy_weight": torch.from_numpy(proxy_weight.astype(np.float32, copy=False)),
        "proxy_projected_view_count": torch.from_numpy(proxy_projected_count),
        "proxy_zbuffer_hit_count": torch.from_numpy(proxy_hit_count),
        "proxy_visible_ratio": torch.from_numpy(proxy_visible_ratio),
        "proxy_color_l1_mean": torch.from_numpy(proxy_color_error_mean),
        "proxy_color_l1_std": torch.from_numpy(proxy_color_error_std),
        "proxy_color_l1_min": torch.from_numpy(proxy_color_error_min),
        "proxy_gate_mean": torch.from_numpy(proxy_gate_mean),
        "proxy_gate_max": torch.from_numpy(proxy_gate_max),
        "proxy_evidence_score": torch.from_numpy(proxy_evidence_score),
        "proxy_trusted_mask": torch.from_numpy(proxy_trusted.astype(bool, copy=False)),
        **{key: torch.from_numpy(value) for key, value in face.items()},
    }
    payload_path = output_root / "mesh_bounded_mesh_evidence_v0.pt"
    torch.save(payload_out, payload_path)

    trusted_face_mask = face["face_trusted_proxy_count"] > 0
    summary = {
        "version": "mesh_bounded_mesh_evidence_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(iteration),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "source_view_count": int(len(cameras_all)),
        "selected_view_count": int(len(cameras)),
        "proxy_count": int(xyz.shape[0]),
        "face_count": int(face_count),
        "payload_loaded": bool(payload["payload_loaded"]),
        "correspondence_loaded": bool(payload["correspondence_loaded"]),
        "output": {
            "payload": str(payload_path),
            "summary": str(output_root / "mesh_bounded_mesh_evidence_v0_summary.json"),
        },
        "params": {
            "color_sigma": float(args.color_sigma),
            "proxy_confidence_power": float(args.proxy_confidence_power),
            "proxy_color_confidence_power": float(args.proxy_color_confidence_power),
            "proxy_mip_support_power": float(args.proxy_mip_support_power),
            "proxy_opacity_power": float(args.proxy_opacity_power),
            "min_hit_views": int(args.min_hit_views),
            "trusted_color_error": float(args.trusted_color_error),
            "trusted_gate": float(args.trusted_gate),
            "depth_min": float(args.depth_min),
            "save_debug_maps": bool(args.save_debug_maps),
            "debug_max_views": int(args.debug_max_views),
        },
        "aggregated": {
            "proxy_projected": int(np.count_nonzero(proxy_projected_count > 0)),
            "proxy_observed": int(np.count_nonzero(observed)),
            "proxy_trusted": int(np.count_nonzero(proxy_trusted)),
            "proxy_trusted_ratio": float(np.mean(proxy_trusted.astype(np.float32))),
            "face_observed": int(np.count_nonzero(face["face_visible_proxy_count"] > 0)),
            "face_trusted": int(np.count_nonzero(trusted_face_mask)),
            "face_trusted_ratio": float(np.mean(trusted_face_mask.astype(np.float32))) if face_count > 0 else 0.0,
            "proxy_hit_count": stats_from_array(proxy_hit_count[proxy_hit_count > 0].astype(np.float32)),
            "proxy_color_l1_mean": stats_from_array(proxy_color_error_mean[observed]),
            "proxy_gate_mean": stats_from_array(proxy_gate_mean[observed]),
            "proxy_evidence_score": stats_from_array(proxy_evidence_score[observed]),
            "face_evidence_score": stats_from_array(face["face_evidence_score"][face["face_visible_proxy_count"] > 0]),
            "face_color_l1_mean": stats_from_array(face["face_color_l1_mean"][face["face_visible_proxy_count"] > 0]),
            "correspondence_signed_offset": stats_from_array(
                np.asarray(payload["signed_offset"], dtype=np.float32)[proxy_trusted]
            ),
            "correspondence_d_norm": stats_from_array(
                np.asarray(payload["d_norm"], dtype=np.float32)[proxy_trusted]
            ),
        },
        "per_view": per_view,
    }
    summary_path = output_root / "mesh_bounded_mesh_evidence_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[mesh-bounded-mesh-evidence-v0] payload : {payload_path}")
    print(f"[mesh-bounded-mesh-evidence-v0] summary : {summary_path}")


if __name__ == "__main__":
    main()
