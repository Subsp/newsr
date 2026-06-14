#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from joint_judge_mip_sof_v0 import stats_from_array
from train_mip_to_sof_surface_v0 import (
    load_cameras_for_split,
    load_model_ply,
    load_sr_mask_for_view,
    normalize_image_name,
    resolve_iteration,
    select_uniform,
    unique_names_for_view,
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
    if array.ndim == 3 and array.shape[0] == 3:
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


def tensor_payload_array(payload: Dict[str, object], key: str, count: int, channels: int | None = None) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim < 1:
        return None
    if int(array.shape[0]) != int(count):
        return None
    if channels is not None and (array.ndim != 2 or int(array.shape[1]) != int(channels)):
        return None
    return array.astype(np.float32, copy=False)


def load_proxy_attributes(model_path: Path, count: int, fallback_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    payload_path = model_path / "mesh_bounded_gaussians_v0.pt"
    rgb = fallback_rgb
    proxy_confidence = np.ones((count,), dtype=np.float32)
    proxy_color_confidence = np.ones((count,), dtype=np.float32)
    proxy_mip_support = np.ones((count,), dtype=np.float32)
    payload_loaded = False
    if payload_path.is_file():
        payload = torch.load(str(payload_path), map_location="cpu")
        if isinstance(payload, dict):
            payload_loaded = True
            payload_rgb = tensor_payload_array(payload, "proxy_rgb", count, channels=3)
            if payload_rgb is not None:
                rgb = np.clip(payload_rgb, 0.0, 1.0).astype(np.float32, copy=False)
            arr = tensor_payload_array(payload, "proxy_confidence", count)
            if arr is not None:
                proxy_confidence = np.clip(arr.reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)
            arr = tensor_payload_array(payload, "proxy_color_confidence", count)
            if arr is not None:
                proxy_color_confidence = np.clip(arr.reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)
            arr = tensor_payload_array(payload, "proxy_mip_support", count)
            if arr is not None:
                proxy_mip_support = np.clip(arr.reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)
    return {
        "rgb": rgb,
        "proxy_confidence": proxy_confidence,
        "proxy_color_confidence": proxy_color_confidence,
        "proxy_mip_support": proxy_mip_support,
        "payload_loaded": np.asarray([1.0 if payload_loaded else 0.0], dtype=np.float32),
    }


def pil_filter_gate(gate: np.ndarray, dilate_radius: int, blur_radius: float) -> np.ndarray:
    gate = np.clip(gate.astype(np.float32, copy=False), 0.0, 1.0)
    image = Image.fromarray((gate * 255.0).astype(np.uint8), mode="L")
    if int(dilate_radius) > 0:
        kernel = 2 * int(dilate_radius) + 1
        image = image.filter(ImageFilter.MaxFilter(kernel))
    if float(blur_radius) > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return (np.asarray(image, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def project_center_gate_for_view(
    *,
    camera,
    xyz: np.ndarray,
    proxy_rgb: np.ndarray,
    proxy_weight: np.ndarray,
    color_sigma: float,
    density_soft_target: float,
    density_power: float,
    depth_min: float,
) -> Dict[str, np.ndarray | Dict[str, object] | float]:
    projected, valid = project_points_camera(camera, xyz, depth_min=float(depth_min), margin=0)
    h = int(camera.image_height)
    w = int(camera.image_width)
    density = np.zeros((h, w), dtype=np.float32)
    nearest_idx = np.full((h * w,), -1, dtype=np.int64)
    if np.any(valid):
        ids = np.flatnonzero(valid).astype(np.int64, copy=False)
        x = np.clip(np.rint(projected[ids, 0]).astype(np.int64), 0, w - 1)
        y = np.clip(np.rint(projected[ids, 1]).astype(np.int64), 0, h - 1)
        pix = y * w + x
        np.add.at(density.reshape(-1), pix, 1.0)
        order = np.argsort(projected[ids, 2], kind="stable")[::-1]
        nearest_idx[pix[order]] = ids[order]

    hit = nearest_idx >= 0
    ref = camera.original_image[:3].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
    ref_flat = np.transpose(ref, (1, 2, 0)).reshape(-1, 3)

    rgb_map = np.zeros((3, h, w), dtype=np.float32)
    err_map = np.zeros((h, w), dtype=np.float32)
    raw_gate = np.zeros((h, w), dtype=np.float32)
    if np.any(hit):
        selected = nearest_idx[hit]
        flat_rgb = np.zeros((h * w, 3), dtype=np.float32)
        flat_rgb[hit] = proxy_rgb[selected]
        rgb_map = np.transpose(flat_rgb.reshape(h, w, 3), (2, 0, 1))
        err = np.mean(np.abs(proxy_rgb[selected] - ref_flat[hit]), axis=1).astype(np.float32, copy=False)
        err_map.reshape(-1)[hit] = err
        color_gate = np.exp(-np.square(err / max(float(color_sigma), 1e-6))).astype(np.float32, copy=False)
        local_weight = proxy_weight[selected].astype(np.float32, copy=False)
        if float(density_power) > 0.0:
            density_hit = density.reshape(-1)[hit]
            density_gate = np.clip(
                density_hit / max(float(density_soft_target), 1e-6),
                0.0,
                1.0,
            )
            local_weight = local_weight * np.power(density_gate, float(density_power)).astype(np.float32, copy=False)
        raw_gate.reshape(-1)[hit] = np.clip(color_gate * local_weight, 0.0, 1.0)

    hit_err = err_map.reshape(-1)[hit]
    hit_gate = raw_gate.reshape(-1)[hit]
    return {
        "raw_gate": raw_gate,
        "density": density,
        "rgb_map": rgb_map,
        "err_map": err_map,
        "hit": hit.reshape(h, w),
        "hit_ratio": float(np.mean(hit.astype(np.float32))),
        "density_stats": stats_from_array(density[density > 0]),
        "color_l1_error_stats": stats_from_array(hit_err if hit_err.size else np.empty((0,), dtype=np.float32)),
        "raw_gate_hit_stats": stats_from_array(hit_gate if hit_gate.size else np.empty((0,), dtype=np.float32)),
    }


def load_source_mask_np(
    *,
    image_name: str,
    mask_dir: Path | None,
    mask_suffix: str,
    target_hw: Tuple[int, int],
) -> Tuple[np.ndarray, str | None]:
    if mask_dir is None:
        return np.ones(target_hw, dtype=np.float32), None
    candidates: List[Path] = []
    names = unique_names_for_view(image_name)
    if mask_suffix:
        candidates.extend(mask_dir / f"{name}{mask_suffix}" for name in names)
    candidates.extend(mask_dir / f"{name}.png" for name in names)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    mask = load_sr_mask_for_view(
        image_name,
        mask_dir=mask_dir,
        mask_suffix=mask_suffix,
        target_hw=target_hw,
        device=torch.device("cpu"),
    )
    if mask is None:
        return np.ones(target_hw, dtype=np.float32), None
    return mask.detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False), str(path) if path is not None else str(mask_dir)


def replace_dir_from_source(src: Path, dst: Path, copy_dirs: bool) -> Dict[str, object]:
    if not src.exists():
        return {"status": "missing_source", "source": str(src), "target": str(dst)}
    if src.resolve() == dst.resolve():
        return {"status": "same_source_target", "source": str(src), "target": str(dst)}
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_dirs:
        shutil.copytree(src, dst)
        mode = "copy"
    else:
        os.symlink(src, dst, target_is_directory=True)
        mode = "symlink"
    return {"status": "linked", "mode": mode, "source": str(src), "target": str(dst)}


def write_prepared_prior_links(source_prior_root: Path, output_prior_root: Path, copy_dirs: bool) -> Dict[str, object]:
    output_prior_root.mkdir(parents=True, exist_ok=True)
    links: Dict[str, object] = {}
    for name in ("fused_priors", "aligned_references"):
        links[name] = replace_dir_from_source(source_prior_root / name, output_prior_root / name, copy_dirs=copy_dirs)
    return links


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build center-projected mesh-bounded surface/color gates. This intentionally ignores "
            "rasterizer alpha because mesh-bounded proxy splats can saturate alpha even at tiny opacity."
        )
    )
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--images_subdir", default="images_2")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_prior_root", default="")
    parser.add_argument("--output_prior_root", default="")
    parser.add_argument("--source_mask_subdir", default="usable_masks")
    parser.add_argument("--source_mask_suffix", default="")
    parser.add_argument("--no_source_mask_multiply", action="store_true")
    parser.add_argument("--copy_prior_dirs", action="store_true")
    parser.add_argument("--color_sigma", type=float, default=0.16)
    parser.add_argument("--proxy_confidence_power", type=float, default=0.5)
    parser.add_argument("--proxy_color_confidence_power", type=float, default=0.5)
    parser.add_argument("--proxy_mip_support_power", type=float, default=0.0)
    parser.add_argument("--density_soft_target", type=float, default=1.0)
    parser.add_argument("--density_power", type=float, default=0.0)
    parser.add_argument("--dilate_radius", type=int, default=5)
    parser.add_argument("--blur_radius", type=float, default=1.5)
    parser.add_argument("--gate_floor", type=float, default=0.0)
    parser.add_argument("--gate_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=0.01)
    parser.add_argument("--save_debug_maps", action="store_true")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    source_prior_root = Path(args.source_prior_root).expanduser().resolve() if str(args.source_prior_root).strip() else None
    output_prior_root = Path(args.output_prior_root).expanduser().resolve() if str(args.output_prior_root).strip() else None
    if source_prior_root is not None and output_prior_root is not None and source_prior_root == output_prior_root:
        raise ValueError("--output_prior_root must be different from --source_prior_root to avoid overwriting existing masks.")
    source_mask_dir = None
    if source_prior_root is not None and not bool(args.no_source_mask_multiply):
        source_mask_dir = source_prior_root / str(args.source_mask_subdir)

    iteration = resolve_iteration(model_path, int(args.iteration))
    gaussians = load_model_ply(model_path, iteration=iteration, sh_degree=3)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    proxy_attrs = load_proxy_attributes(model_path, int(xyz.shape[0]), fallback_rgb=model_dc_rgb_np(gaussians))
    proxy_rgb = np.clip(proxy_attrs["rgb"], 0.0, 1.0).astype(np.float32, copy=False)
    proxy_weight = np.ones((xyz.shape[0],), dtype=np.float32)
    for key, power in (
        ("proxy_confidence", float(args.proxy_confidence_power)),
        ("proxy_color_confidence", float(args.proxy_color_confidence_power)),
        ("proxy_mip_support", float(args.proxy_mip_support_power)),
    ):
        if power > 0.0:
            proxy_weight *= np.power(
                np.clip(proxy_attrs[key].reshape(-1), 0.0, 1.0),
                power,
            ).astype(np.float32, copy=False)
    proxy_weight = np.clip(proxy_weight, 0.0, 1.0).astype(np.float32, copy=False)

    cameras_all = load_cameras_for_split(scene_root, model_path, str(args.images_subdir), str(args.split))
    cameras = select_uniform(cameras_all, int(args.max_views))
    if len(cameras) <= 0:
        raise RuntimeError("No cameras selected for mesh-bounded center gate generation.")

    if output_prior_root is not None and source_prior_root is not None:
        prior_link_summary = write_prepared_prior_links(
            source_prior_root,
            output_prior_root,
            copy_dirs=bool(args.copy_prior_dirs),
        )
        mask_root = output_prior_root / "usable_masks"
        debug_root = output_prior_root / "mesh_bounded_center_gate_debug_v0"
    else:
        prior_link_summary = {"enabled": False}
        mask_root = output_root / "usable_masks"
        debug_root = output_root
    mask_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    raw_gate_root = debug_root / "center_gate_raw"
    final_gate_root = mask_root
    density_root = debug_root / "center_density_log"
    rgb_root = debug_root / "center_rgb_model_zbuf"
    err_root = debug_root / "center_color_l1_error_model_zbuf"
    source_mask_root = debug_root / "source_mask"

    per_view: List[Dict[str, object]] = []
    for view_idx, camera in enumerate(tqdm(cameras, desc="build mesh-bounded center gates")):
        target_hw = (int(camera.image_height), int(camera.image_width))
        center = project_center_gate_for_view(
            camera=camera,
            xyz=xyz,
            proxy_rgb=proxy_rgb,
            proxy_weight=proxy_weight,
            color_sigma=float(args.color_sigma),
            density_soft_target=float(args.density_soft_target),
            density_power=float(args.density_power),
            depth_min=float(args.depth_min),
        )
        raw_gate = np.asarray(center["raw_gate"], dtype=np.float32)
        filtered_gate = pil_filter_gate(raw_gate, int(args.dilate_radius), float(args.blur_radius))
        filtered_gate = np.clip(float(args.gate_scale) * filtered_gate, 0.0, 1.0)
        source_mask, source_mask_path = load_source_mask_np(
            image_name=str(camera.image_name),
            mask_dir=source_mask_dir,
            mask_suffix=str(args.source_mask_suffix),
            target_hw=target_hw,
        )
        if not bool(args.no_source_mask_multiply):
            filtered_gate = filtered_gate * source_mask
        if float(args.gate_floor) > 0.0:
            filtered_gate = np.where(filtered_gate > 0.0, np.maximum(filtered_gate, float(args.gate_floor)), 0.0)
        filtered_gate = np.clip(filtered_gate, 0.0, 1.0).astype(np.float32, copy=False)

        stem = normalize_image_name(str(camera.image_name))
        save_gray(final_gate_root / f"{stem}.png", filtered_gate)
        if bool(args.save_debug_maps):
            save_gray(raw_gate_root / f"{stem}.png", raw_gate)
            save_gray(density_root / f"{stem}.png", log_normalize(np.asarray(center["density"], dtype=np.float32), gain=10.0))
            save_rgb(rgb_root / f"{stem}.png", np.asarray(center["rgb_map"], dtype=np.float32))
            save_gray(err_root / f"{stem}.png", np.clip(np.asarray(center["err_map"], dtype=np.float32) / 0.35, 0.0, 1.0))
            if source_mask_dir is not None:
                save_gray(source_mask_root / f"{stem}.png", source_mask)

        per_view.append(
            {
                "source_view_index": int(view_idx),
                "image_name": stem,
                "camera_image_name": str(camera.image_name),
                "mask_path": str(final_gate_root / f"{stem}.png"),
                "source_mask_dir": str(source_mask_dir) if source_mask_dir is not None else None,
                "source_mask_loaded_from": source_mask_path,
                "hit_ratio": float(center["hit_ratio"]),
                "density": center["density_stats"],
                "color_l1_error_model_zbuf": center["color_l1_error_stats"],
                "raw_gate_hit": center["raw_gate_hit_stats"],
                "raw_gate_mean": float(raw_gate.mean()),
                "filtered_gate_mean": float(filtered_gate.mean()),
                "filtered_gate_p95": float(np.percentile(filtered_gate, 95.0)),
                "source_mask_mean": float(source_mask.mean()),
            }
        )

    summary = {
        "version": "mesh_bounded_center_gate_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "loaded_iteration": int(iteration),
        "images_subdir": str(args.images_subdir),
        "split": str(args.split),
        "source_view_count": int(len(cameras_all)),
        "selected_view_count": int(len(cameras)),
        "proxy_count": int(xyz.shape[0]),
        "payload_loaded": bool(float(proxy_attrs["payload_loaded"][0]) > 0.5),
        "source_prior_root": str(source_prior_root) if source_prior_root is not None else None,
        "output_prior_root": str(output_prior_root) if output_prior_root is not None else None,
        "mask_root": str(mask_root),
        "debug_root": str(debug_root),
        "prior_links": prior_link_summary,
        "params": {
            "color_sigma": float(args.color_sigma),
            "proxy_confidence_power": float(args.proxy_confidence_power),
            "proxy_color_confidence_power": float(args.proxy_color_confidence_power),
            "proxy_mip_support_power": float(args.proxy_mip_support_power),
            "density_soft_target": float(args.density_soft_target),
            "density_power": float(args.density_power),
            "dilate_radius": int(args.dilate_radius),
            "blur_radius": float(args.blur_radius),
            "gate_floor": float(args.gate_floor),
            "gate_scale": float(args.gate_scale),
            "depth_min": float(args.depth_min),
            "source_mask_multiply": not bool(args.no_source_mask_multiply),
            "save_debug_maps": bool(args.save_debug_maps),
        },
        "aggregated": {
            "mean_hit_ratio": float(np.mean([row["hit_ratio"] for row in per_view])) if per_view else 0.0,
            "mean_raw_gate": float(np.mean([row["raw_gate_mean"] for row in per_view])) if per_view else 0.0,
            "mean_filtered_gate": float(np.mean([row["filtered_gate_mean"] for row in per_view])) if per_view else 0.0,
            "mean_source_mask": float(np.mean([row["source_mask_mean"] for row in per_view])) if per_view else 0.0,
        },
        "per_view": per_view,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mesh_bounded_center_gate_v0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if output_prior_root is not None:
        (output_prior_root / "mesh_bounded_center_gate_v0_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "prior_dir": str(output_prior_root / "fused_priors"),
            "reference_dir": str(output_prior_root / "aligned_references"),
            "output_root": str(output_prior_root),
            "mask_mode": "mesh_bounded_center_color_gate",
            "num_priors": int(len(per_view)),
            "num_matched": int(len(per_view)),
            "usable_mean": summary["aggregated"]["mean_filtered_gate"],
            "frames": [
                {
                    "image_name": f"{row['image_name']}.png",
                    "stem": str(row["image_name"]),
                    "source_view_index": int(row["source_view_index"]),
                    "usable_mean": float(row["filtered_gate_mean"]),
                    "source_mask_path": row["source_mask_loaded_from"],
                }
                for row in per_view
            ],
            "mesh_bounded_center_gate_summary": "mesh_bounded_center_gate_v0_summary.json",
        }
        (output_prior_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[mesh-bounded-center-gate-v0] masks   : {mask_root}")
    print(f"[mesh-bounded-center-gate-v0] summary : {output_root / 'mesh_bounded_center_gate_v0_summary.json'}")
    if output_prior_root is not None:
        print(f"[mesh-bounded-center-gate-v0] prior   : {output_prior_root}")


if __name__ == "__main__":
    main()
