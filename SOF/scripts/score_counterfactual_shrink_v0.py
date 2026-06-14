#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from gaussian_renderer import render_simple
from scene import Scene
from scene.gaussian_model import GaussianModel

from detect_starburst_gaussian_artifacts_v0 import (
    _build_dataset_args,
    _build_image_lookup_from_paths,
    _build_starburst_maps,
    _list_image_paths,
    _load_rgb_tensor,
    _resolve_model_iteration,
    _resolve_reference_path,
    _save_heatmap,
    _save_overlay,
    _save_rgb,
    _select_views,
)
from export_gaussian_mask_subset_v0 import _clone_subset_gaussians


def _to_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clamp(0.0, 1.0).cpu()


def _load_payload_tensor(payload: Dict[str, object], key: str, expected_count: int, *, dtype: torch.dtype) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"Key '{key}' not found in payload.")
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.reshape(-1)
    if int(value.shape[0]) != int(expected_count):
        raise ValueError(
            f"Payload key '{key}' length mismatch: expected {expected_count}, got {int(value.shape[0])}"
        )
    return value.to(dtype=dtype)


def _candidate_ids_from_payload(
    payload: Dict[str, object],
    *,
    mask_key: str,
    score_key: str,
    expected_count: int,
    max_test_count: int,
    max_test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = _load_payload_tensor(payload, mask_key, expected_count, dtype=torch.bool).cpu().numpy().astype(bool, copy=False)
    score = _load_payload_tensor(payload, score_key, expected_count, dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
    candidate_ids = np.flatnonzero(mask)
    if candidate_ids.size == 0:
        return candidate_ids, score
    order = np.argsort(-score[candidate_ids], kind="stable")
    candidate_ids = candidate_ids[order]
    cap = candidate_ids.size
    if float(max_test_fraction) > 0.0:
        cap = min(cap, max(1, int(round(float(max_test_fraction) * float(expected_count)))))
    if int(max_test_count) > 0:
        cap = min(cap, int(max_test_count))
    return candidate_ids[:cap], score


def _group_ids(ids: np.ndarray, group_size: int) -> List[np.ndarray]:
    size = max(1, int(group_size))
    return [ids[start : start + size] for start in range(0, int(ids.size), size)]


def _apply_shrink_in_place(
    gaussians: GaussianModel,
    *,
    shrink_factor: float,
    axis_mode: str,
) -> None:
    factor = float(shrink_factor)
    if not (0.0 < factor <= 1.0):
        raise ValueError(f"shrink_factor must be in (0, 1], got {factor}")
    log_delta = float(np.log(factor))
    with torch.no_grad():
        if str(axis_mode) == "uniform":
            gaussians._scaling.add_(log_delta)
            return
        if str(axis_mode) != "major_only":
            raise ValueError(f"Unsupported shrink_axis_mode={axis_mode!r}; use 'uniform' or 'major_only'.")
        major_axis = torch.argmax(gaussians.get_scaling.detach(), dim=1, keepdim=True)
        updates = torch.zeros_like(gaussians._scaling)
        updates.scatter_(1, major_axis, log_delta)
        gaussians._scaling.add_(updates)


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> float:
    weighted = value * weight
    denom = torch.clamp(weight.sum(), min=1e-6)
    return float((weighted.sum() / denom).item())


def _compute_view_losses(render_rgb: torch.Tensor, reference_rgb: torch.Tensor, maps: Dict[str, torch.Tensor]) -> Dict[str, float]:
    artifact_map = 0.7 * maps["star_map"] + 0.3 * maps["unsupported_ridge"]
    preserve_mask = torch.clamp(maps["supported_structure"], 0.0, 1.0)
    ridge_err = torch.abs(maps["render_ridge"] - maps["reference_ridge"])
    return {
        "artifact": float(artifact_map.mean().item()),
        "preserve": _masked_mean(ridge_err, preserve_mask) if float(preserve_mask.sum().item()) > 1e-6 else 0.0,
        "rgb": float(torch.mean(torch.abs(render_rgb - reference_rgb)).item()),
        "supported_mass": float(preserve_mask.mean().item()),
    }


def _stats(values: Sequence[float]) -> Dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def _maybe_keep_debug_candidate(
    debug_candidates: List[Dict[str, object]],
    candidate: Dict[str, object],
    limit: int,
) -> None:
    if int(limit) <= 0:
        return
    debug_candidates.append(candidate)
    debug_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    if len(debug_candidates) > int(limit):
        del debug_candidates[int(limit) :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score broad active-set gaussians with an approximate counterfactual shrink test.")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_payload_path", required=True)
    parser.add_argument("--candidate_mask_key", default="global_prefilter_candidate")
    parser.add_argument("--candidate_score_key", default="global_prefilter_score")
    parser.add_argument("--reference_images_subdir", default="")
    parser.add_argument("--reference_image_dir", default="")
    parser.add_argument("--interaction_images_subdir", default="images_2")
    parser.add_argument("--camera_resolution", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument("--view_indices", default="")
    parser.add_argument("--max_test_count", type=int, default=2048)
    parser.add_argument("--max_test_fraction", type=float, default=0.0)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--shrink_factor", type=float, default=0.5)
    parser.add_argument("--shrink_axis_mode", choices=["uniform", "major_only"], default="uniform")
    parser.add_argument("--line_lengths", default="9,17,31")
    parser.add_argument("--angles_deg", default="0,22.5,45,67.5,90,112.5,135,157.5")
    parser.add_argument("--highpass_kernel", type=int, default=21)
    parser.add_argument("--ridge_norm_percentile", type=float, default=0.995)
    parser.add_argument("--residual_norm_percentile", type=float, default=0.990)
    parser.add_argument("--min_artifact_improve", type=float, default=0.002)
    parser.add_argument("--max_preserve_increase", type=float, default=0.0015)
    parser.add_argument("--preserve_penalty_weight", type=float, default=1.0)
    parser.add_argument("--num_debug_groups", type=int, default=4)
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_groups"
    debug_dir.mkdir(parents=True, exist_ok=True)

    reference_images_subdir = str(args.reference_images_subdir).strip()
    reference_image_dir = str(args.reference_image_dir).strip()
    if reference_image_dir:
        reference_root = Path(reference_image_dir).expanduser().resolve()
        reference_paths = _list_image_paths(reference_root)
        reference_lookup = _build_image_lookup_from_paths(reference_paths)
    elif reference_images_subdir:
        reference_root = scene_root / reference_images_subdir
        reference_paths = _list_image_paths(reference_root)
        reference_lookup = _build_image_lookup_from_paths(reference_paths)
    else:
        raise ValueError("Provide reference_image_dir or reference_images_subdir for supervised shrink scoring.")

    payload = torch.load(Path(args.candidate_payload_path).expanduser().resolve(), map_location="cpu")
    if str(payload.get("version", "")).strip() == "":
        raise ValueError(f"Unsupported or missing version in payload: {args.candidate_payload_path}")

    line_lengths = [int(v) for v in str(args.line_lengths).split(",") if str(v).strip()]
    angles_deg = [float(v) for v in str(args.angles_deg).split(",") if str(v).strip()]
    view_indices = [int(v) for v in str(args.view_indices).split(",") if str(v).strip()]

    dataset = _build_dataset_args(str(scene_root), str(model_path), str(args.interaction_images_subdir), False)
    dataset.resolution = int(args.camera_resolution)
    gaussians = GaussianModel(dataset.sh_degree)
    iteration = _resolve_model_iteration(model_path, int(args.iteration))
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=False, skip_train=False)
    loaded_iter = int(scene.loaded_iter if scene.loaded_iter is not None else iteration)
    all_views = scene.getTrainCameras() if str(args.split) == "train" else scene.getTestCameras() if str(args.split) == "test" else list(scene.getTrainCameras()) + list(scene.getTestCameras())
    views, selected_indices = _select_views(all_views, int(args.max_views), view_indices)
    if not views:
        raise RuntimeError(f"No views selected for split={args.split}")

    if int(gaussians.get_xyz.shape[0]) <= 0:
        raise RuntimeError("Model has zero gaussians.")
    candidate_ids, candidate_score = _candidate_ids_from_payload(
        payload,
        mask_key=str(args.candidate_mask_key),
        score_key=str(args.candidate_score_key),
        expected_count=int(gaussians.get_xyz.shape[0]),
        max_test_count=int(args.max_test_count),
        max_test_fraction=float(args.max_test_fraction),
    )
    if candidate_ids.size == 0:
        raise RuntimeError(
            f"Candidate payload {args.candidate_payload_path} / key={args.candidate_mask_key} selected zero gaussians."
        )

    if dataset.white_background:
        raise ValueError("Counterfactual shrink approximation assumes black background / premultiplied RGB.")
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")

    base_cache: List[Dict[str, object]] = []
    with torch.inference_mode():
        for local_idx, (source_view_idx, view) in enumerate(zip(selected_indices, views)):
            base_pkg = render_simple(view, gaussians, background)
            base_rgb = _to_cpu_tensor(base_pkg["render"][:3])
            reference_path = _resolve_reference_path(
                lookup=reference_lookup,
                paths=reference_paths,
                image_name=str(view.image_name),
                source_view_idx=int(source_view_idx),
                local_view_idx=int(local_idx),
                reference_root=reference_root,
            )
            reference_rgb = _load_rgb_tensor(reference_path, base_pkg["render"].device).detach().clamp(0.0, 1.0)
            if reference_rgb.shape[-2:] != base_rgb.shape[-2:]:
                reference_rgb = torch.nn.functional.interpolate(
                    reference_rgb[None],
                    size=base_rgb.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )[0].clamp(0.0, 1.0)
            reference_rgb = reference_rgb.cpu()
            base_maps = _build_starburst_maps(
                base_rgb,
                reference_rgb,
                line_lengths=line_lengths,
                angles_deg=angles_deg,
                highpass_kernel=int(args.highpass_kernel),
                ridge_norm_percentile=float(args.ridge_norm_percentile),
                residual_norm_percentile=float(args.residual_norm_percentile),
            )
            base_losses = _compute_view_losses(base_rgb, reference_rgb, base_maps)
            base_cache.append(
                {
                    "view": view,
                    "source_view_index": int(source_view_idx),
                    "image_name": str(view.image_name),
                    "reference_path": str(reference_path),
                    "base_rgb": base_rgb,
                    "reference_rgb": reference_rgb,
                    "base_maps": base_maps,
                    "base_losses": base_losses,
                    "visible_mask": base_pkg["visibility_filter"].detach().cpu().numpy().astype(bool, copy=False),
                }
            )
            del base_pkg
        torch.cuda.empty_cache()

    tested_mask = np.zeros((int(gaussians.get_xyz.shape[0]),), dtype=bool)
    accepted_mask = np.zeros_like(tested_mask)
    counterfactual_score = np.zeros((tested_mask.shape[0],), dtype=np.float32)
    artifact_improve = np.zeros_like(counterfactual_score)
    preserve_increase = np.zeros_like(counterfactual_score)
    rgb_improve = np.zeros_like(counterfactual_score)
    group_index = np.full((tested_mask.shape[0],), -1, dtype=np.int32)
    group_rows: List[Dict[str, object]] = []
    debug_candidates: List[Dict[str, object]] = []

    for current_group_idx, group_ids in enumerate(tqdm(_group_ids(candidate_ids, int(args.group_size)), desc="counterfactual-shrink")):
        visible_views = [cached for cached in base_cache if bool(np.any(cached["visible_mask"][group_ids]))]
        if not visible_views:
            group_rows.append(
                {
                    "group_index": int(current_group_idx),
                    "gaussian_ids": [int(v) for v in group_ids.tolist()],
                    "group_size": int(group_ids.size),
                    "seed_score_mean": float(np.mean(candidate_score[group_ids])),
                    "artifact_improve": 0.0,
                    "preserve_increase": 0.0,
                    "rgb_improve": 0.0,
                    "counterfactual_score": 0.0,
                    "accepted": False,
                    "skipped_reason": "not_visible_in_selected_views",
                    "views": [],
                }
            )
            continue

        mask = torch.zeros((int(gaussians.get_xyz.shape[0]),), dtype=torch.bool, device=gaussians.get_xyz.device)
        mask[torch.from_numpy(group_ids).to(device=mask.device, dtype=torch.int64)] = True
        subset_full = _clone_subset_gaussians(gaussians, mask)
        subset_shrunk = _clone_subset_gaussians(gaussians, mask)
        _apply_shrink_in_place(
            subset_shrunk,
            shrink_factor=float(args.shrink_factor),
            axis_mode=str(args.shrink_axis_mode),
        )

        per_view_rows: List[Dict[str, object]] = []
        debug_bundle: Dict[str, object] | None = None
        with torch.inference_mode():
            for cached in visible_views:
                full_pkg = render_simple(cached["view"], subset_full, background)
                full_rgb = _to_cpu_tensor(full_pkg["render"][:3])
                del full_pkg

                shrunk_pkg = render_simple(cached["view"], subset_shrunk, background)
                shrunk_rgb = _to_cpu_tensor(shrunk_pkg["render"][:3])
                del shrunk_pkg
                approx_cf_rgb = torch.clamp(cached["base_rgb"] - full_rgb + shrunk_rgb, 0.0, 1.0)
                cf_maps = _build_starburst_maps(
                    approx_cf_rgb,
                    cached["reference_rgb"],
                    line_lengths=line_lengths,
                    angles_deg=angles_deg,
                    highpass_kernel=int(args.highpass_kernel),
                    ridge_norm_percentile=float(args.ridge_norm_percentile),
                    residual_norm_percentile=float(args.residual_norm_percentile),
                )
                cf_losses = _compute_view_losses(approx_cf_rgb, cached["reference_rgb"], cf_maps)
                row = {
                    "source_view_index": int(cached["source_view_index"]),
                    "image_name": str(cached["image_name"]),
                    "artifact_improve": float(cached["base_losses"]["artifact"] - cf_losses["artifact"]),
                    "preserve_increase": float(cf_losses["preserve"] - cached["base_losses"]["preserve"]),
                    "rgb_improve": float(cached["base_losses"]["rgb"] - cf_losses["rgb"]),
                    "base_artifact": float(cached["base_losses"]["artifact"]),
                    "cf_artifact": float(cf_losses["artifact"]),
                    "base_preserve": float(cached["base_losses"]["preserve"]),
                    "cf_preserve": float(cf_losses["preserve"]),
                }
                per_view_rows.append(row)
                if debug_bundle is None:
                    debug_bundle = {
                        "base_rgb": cached["base_rgb"],
                        "subset_full_rgb": full_rgb,
                        "subset_shrunk_rgb": shrunk_rgb,
                        "approx_cf_rgb": approx_cf_rgb,
                        "base_maps": cached["base_maps"],
                        "cf_maps": cf_maps,
                        "image_name": str(cached["image_name"]),
                    }
        del subset_full, subset_shrunk, mask
        if current_group_idx % 32 == 0:
            torch.cuda.empty_cache()

        artifact_gain = float(np.mean([row["artifact_improve"] for row in per_view_rows]))
        preserve_cost = float(np.mean([row["preserve_increase"] for row in per_view_rows]))
        rgb_gain = float(np.mean([row["rgb_improve"] for row in per_view_rows]))
        score = float(artifact_gain - float(args.preserve_penalty_weight) * max(preserve_cost, 0.0))
        accepted = artifact_gain >= float(args.min_artifact_improve) and preserve_cost <= float(args.max_preserve_increase)

        tested_mask[group_ids] = True
        accepted_mask[group_ids] = bool(accepted)
        counterfactual_score[group_ids] = score
        artifact_improve[group_ids] = artifact_gain
        preserve_increase[group_ids] = preserve_cost
        rgb_improve[group_ids] = rgb_gain
        group_index[group_ids] = int(current_group_idx)

        group_rows.append(
            {
                "group_index": int(current_group_idx),
                "gaussian_ids": [int(v) for v in group_ids.tolist()],
                "group_size": int(group_ids.size),
                "seed_score_mean": float(np.mean(candidate_score[group_ids])),
                "artifact_improve": artifact_gain,
                "preserve_increase": preserve_cost,
                "rgb_improve": rgb_gain,
                "counterfactual_score": score,
                "accepted": bool(accepted),
                "views": per_view_rows,
            }
        )
        if debug_bundle is not None:
            _maybe_keep_debug_candidate(
                debug_candidates,
                {
                    "score": score,
                    "accepted": bool(accepted),
                    "group_index": int(current_group_idx),
                    "bundle": debug_bundle,
                },
                int(args.num_debug_groups),
            )

    debug_candidates.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(debug_candidates[: max(0, int(args.num_debug_groups))]):
        bundle = item["bundle"]
        prefix = debug_dir / f"group_{rank:03d}_idx{int(item['group_index']):04d}_{str(bundle['image_name']).replace('/', '_')}"
        _save_rgb(prefix.with_name(prefix.name + "_base.png"), bundle["base_rgb"])
        _save_rgb(prefix.with_name(prefix.name + "_subset_full.png"), bundle["subset_full_rgb"])
        _save_rgb(prefix.with_name(prefix.name + "_subset_shrunk.png"), bundle["subset_shrunk_rgb"])
        _save_rgb(prefix.with_name(prefix.name + "_approx_cf.png"), bundle["approx_cf_rgb"])
        _save_heatmap(prefix.with_name(prefix.name + "_base_star_heat.png"), bundle["base_maps"]["star_map"])
        _save_heatmap(prefix.with_name(prefix.name + "_cf_star_heat.png"), bundle["cf_maps"]["star_map"])
        _save_overlay(prefix.with_name(prefix.name + "_base_star_overlay.png"), bundle["base_rgb"], bundle["base_maps"]["star_map"])
        _save_overlay(prefix.with_name(prefix.name + "_cf_star_overlay.png"), bundle["approx_cf_rgb"], bundle["cf_maps"]["star_map"])

        payload_out = {
        "version": "counterfactual_shrink_scores_v0",
        "num_gaussians": int(tested_mask.shape[0]),
        "tested_candidate": torch.from_numpy(tested_mask.astype(bool))[:, None],
        "counterfactual_candidate": torch.from_numpy(accepted_mask.astype(bool))[:, None],
        "counterfactual_score": torch.from_numpy(counterfactual_score.astype(np.float32))[:, None],
        "artifact_improve": torch.from_numpy(artifact_improve.astype(np.float32))[:, None],
        "preserve_increase": torch.from_numpy(preserve_increase.astype(np.float32))[:, None],
        "rgb_improve": torch.from_numpy(rgb_improve.astype(np.float32))[:, None],
        "group_index": torch.from_numpy(group_index.astype(np.int32))[:, None],
        "meta": {
            "scene_root": str(scene_root),
            "model_path": str(model_path),
            "candidate_payload_path": str(args.candidate_payload_path),
            "candidate_mask_key": str(args.candidate_mask_key),
            "candidate_score_key": str(args.candidate_score_key),
            "interaction_images_subdir": str(args.interaction_images_subdir),
            "camera_resolution": int(args.camera_resolution),
            "reference_images_subdir": reference_images_subdir,
            "reference_image_dir": reference_image_dir,
            "iteration": int(loaded_iter),
            "split": str(args.split),
            "selected_view_indices": [int(v) for v in selected_indices],
            "selected_view_names": [str(view.image_name) for view in views],
            "tested_candidate_count": int(np.sum(tested_mask)),
            "accepted_candidate_count": int(np.sum(accepted_mask)),
            "group_size": int(args.group_size),
            "shrink_factor": float(args.shrink_factor),
            "shrink_axis_mode": str(args.shrink_axis_mode),
            "min_artifact_improve": float(args.min_artifact_improve),
            "max_preserve_increase": float(args.max_preserve_increase),
            "preserve_penalty_weight": float(args.preserve_penalty_weight),
            "approximation_note": "Counterfactual render is approximated as base - subset_full + subset_shrunk on black-background RGB.",
        },
    }
    payload_path = output_dir / "counterfactual_shrink_scores_v0.pt"
    torch.save(payload_out, payload_path)

    summary = {
        "version": "counterfactual_shrink_scores_v0",
        "scene_root": str(scene_root),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "payload_path": str(payload_path),
        "candidate_payload_path": str(args.candidate_payload_path),
        "candidate_mask_key": str(args.candidate_mask_key),
        "candidate_score_key": str(args.candidate_score_key),
        "camera_resolution": int(args.camera_resolution),
        "tested_candidate_count": int(np.sum(tested_mask)),
        "accepted_candidate_count": int(np.sum(accepted_mask)),
        "accepted_ratio_of_tested": float(np.sum(accepted_mask) / max(int(np.sum(tested_mask)), 1)),
        "group_size": int(args.group_size),
        "num_groups": int(len(group_rows)),
        "shrink_factor": float(args.shrink_factor),
        "shrink_axis_mode": str(args.shrink_axis_mode),
        "artifact_improve_stats": _stats(artifact_improve[tested_mask]),
        "preserve_increase_stats": _stats(preserve_increase[tested_mask]),
        "rgb_improve_stats": _stats(rgb_improve[tested_mask]),
        "counterfactual_score_stats": _stats(counterfactual_score[tested_mask]),
        "debug_groups_dir": str(debug_dir),
        "groups": group_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
