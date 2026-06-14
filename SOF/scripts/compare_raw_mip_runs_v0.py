from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def finite_stats(values: np.ndarray, max_samples: int = 2_000_000) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    total = int(arr.size)
    if total == 0:
        return {
            "count": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "sampled": False,
        }
    finite = np.isfinite(arr)
    arr = arr[finite]
    sampled = False
    if arr.size > max_samples:
        step = int(math.ceil(arr.size / max_samples))
        arr = arr[::step]
        sampled = True
    if arr.size == 0:
        return {
            "count": total,
            "finite_count": 0,
            "nonfinite_count": total,
            "sampled": sampled,
        }
    qs = np.quantile(arr, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": total,
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(total - finite.sum()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p10": float(qs[3]),
        "median": float(qs[4]),
        "p90": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
        "sampled": sampled,
    }


def ratio_delta(current: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("mean", "std", "min", "p01", "p05", "p10", "median", "p90", "p95", "p99", "max"):
        a = current.get(key)
        b = rerun.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[f"{key}_delta"] = float(b - a)
            out[f"{key}_ratio"] = float(b / a) if abs(float(a)) > 1e-12 else None
    return out


def prop_names(vertex: Any, prefix: str) -> list[str]:
    names = list(vertex.data.dtype.names or [])
    return sorted(
        [name for name in names if name.startswith(prefix)],
        key=lambda item: int(item.rsplit("_", 1)[1]) if item.rsplit("_", 1)[-1].isdigit() else item,
    )


def stack_props(vertex: Any, names: list[str], dtype=np.float64) -> np.ndarray:
    if not names:
        return np.empty((len(vertex.data), 0), dtype=dtype)
    return np.stack([np.asarray(vertex[name], dtype=dtype) for name in names], axis=1)


def flattened_prop_sample(vertex: Any, names: list[str], max_samples: int = 2_000_000) -> np.ndarray:
    if not names:
        return np.empty((0,), dtype=np.float64)
    per_prop = max(1, max_samples // max(1, len(names)))
    samples = []
    for name in names:
        arr = np.asarray(vertex[name], dtype=np.float64).reshape(-1)
        if arr.size > per_prop:
            step = int(math.ceil(arr.size / per_prop))
            arr = arr[::step]
        samples.append(arr)
    return np.concatenate(samples, axis=0)


def read_cfg(model_path: Path) -> dict[str, Any]:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        return {"_missing": True}
    text = cfg_path.read_text(encoding="utf-8").strip()
    try:
        parsed = eval(text, {"Namespace": Namespace})
    except Exception as exc:
        return {"_parse_error": str(exc), "_raw": text}
    out: dict[str, Any] = {}
    for key, value in vars(parsed).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = repr(value)
    return out


def cfg_diff(current: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    ignored = {"model_path"}
    diff: dict[str, Any] = {}
    for key in sorted((set(current) | set(rerun)) - ignored):
        a = current.get(key, "<missing>")
        b = rerun.get(key, "<missing>")
        if a != b:
            diff[key] = {"current": a, "rerun": b}
    return diff


def ply_path(model_path: Path, iteration: int) -> Path:
    return model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def summarize_ply(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or [])
    n = len(vertex.data)

    xyz_names = ["x", "y", "z"]
    scale_names = prop_names(vertex, "scale_")
    rot_names = prop_names(vertex, "rot_")
    dc_names = prop_names(vertex, "f_dc_")
    rest_names = prop_names(vertex, "f_rest_")

    xyz = stack_props(vertex, xyz_names) if all(name in names for name in xyz_names) else np.empty((n, 0))
    scale_raw = stack_props(vertex, scale_names)
    scale = np.exp(scale_raw) if scale_raw.size else np.empty((n, 0))
    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float64) if "opacity" in names else np.empty((n,))
    opacity = sigmoid(opacity_raw) if opacity_raw.size else np.empty((n,))
    filter_3d = np.asarray(vertex["filter_3D"], dtype=np.float64) if "filter_3D" in names else np.zeros((n,), dtype=np.float64)
    effective_scale = np.sqrt(scale * scale + filter_3d[:, None] * filter_3d[:, None]) if scale.size else np.empty((n, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_min = np.min(scale, axis=1) if scale.size else np.empty((n,))
        scale_max = np.max(scale, axis=1) if scale.size else np.empty((n,))
        eff_min = np.min(effective_scale, axis=1) if effective_scale.size else np.empty((n,))
        eff_max = np.max(effective_scale, axis=1) if effective_scale.size else np.empty((n,))
        scale_anisotropy = scale_max / np.maximum(scale_min, 1e-12)
        effective_anisotropy = eff_max / np.maximum(eff_min, 1e-12)
        volume_radius = np.power(np.maximum(np.prod(effective_scale, axis=1), 0.0), 1.0 / 3.0) if effective_scale.size else np.empty((n,))

    dc = stack_props(vertex, dc_names)
    if dc.shape[1] >= 3:
        rgb_dc = dc[:, :3] * SH_C0 + 0.5
        dc_luma = 0.2126 * rgb_dc[:, 0] + 0.7152 * rgb_dc[:, 1] + 0.0722 * rgb_dc[:, 2]
        dc_chroma_delta = np.max(rgb_dc[:, :3], axis=1) - np.min(rgb_dc[:, :3], axis=1)
    else:
        dc_luma = np.empty((0,), dtype=np.float64)
        dc_chroma_delta = np.empty((0,), dtype=np.float64)

    rest_norm = np.zeros((n,), dtype=np.float64)
    for name in rest_names:
        arr = np.asarray(vertex[name], dtype=np.float64)
        rest_norm += arr * arr
    rest_norm = np.sqrt(rest_norm) if rest_names else np.empty((0,), dtype=np.float64)

    rot = stack_props(vertex, rot_names)
    rot_norm = np.linalg.norm(rot, axis=1) if rot.size else np.empty((0,), dtype=np.float64)

    finite_xyz = np.isfinite(xyz).all(axis=1) if xyz.size else np.ones((n,), dtype=bool)
    finite_scale = np.isfinite(scale_raw).all(axis=1) if scale_raw.size else np.ones((n,), dtype=bool)
    finite_opacity = np.isfinite(opacity_raw) if opacity_raw.size else np.ones((n,), dtype=bool)
    finite_filter = np.isfinite(filter_3d) if filter_3d.size else np.ones((n,), dtype=bool)

    return {
        "path": str(path),
        "gaussian_count": int(n),
        "properties": sorted(names),
        "property_counts": {
            "scale": len(scale_names),
            "rotation": len(rot_names),
            "feature_dc": len(dc_names),
            "feature_rest": len(rest_names),
            "has_filter_3D": "filter_3D" in names,
        },
        "finite_counts": {
            "xyz": int(finite_xyz.sum()),
            "scale": int(finite_scale.sum()),
            "opacity": int(finite_opacity.sum()),
            "filter_3D": int(finite_filter.sum()),
            "all_core": int((finite_xyz & finite_scale & finite_opacity & finite_filter).sum()),
        },
        "xyz_stats": finite_stats(xyz),
        "scale_raw_stats": finite_stats(scale_raw),
        "scale_activated_stats": finite_stats(scale),
        "scale_major_stats": finite_stats(scale_max),
        "scale_minor_stats": finite_stats(scale_min),
        "scale_anisotropy_stats": finite_stats(scale_anisotropy),
        "filter_3D_stats": finite_stats(filter_3d),
        "effective_scale_stats": finite_stats(effective_scale),
        "effective_scale_major_stats": finite_stats(eff_max),
        "effective_scale_minor_stats": finite_stats(eff_min),
        "effective_anisotropy_stats": finite_stats(effective_anisotropy),
        "effective_volume_radius_stats": finite_stats(volume_radius),
        "opacity_raw_stats": finite_stats(opacity_raw),
        "opacity_activated_stats": finite_stats(opacity),
        "feature_dc_stats": finite_stats(flattened_prop_sample(vertex, dc_names)),
        "feature_dc_luma_stats": finite_stats(dc_luma),
        "feature_dc_chroma_delta_stats": finite_stats(dc_chroma_delta),
        "feature_rest_stats": finite_stats(flattened_prop_sample(vertex, rest_names)),
        "feature_rest_norm_stats": finite_stats(rest_norm),
        "rotation_norm_stats": finite_stats(rot_norm),
    }


def compare_stats(current: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "scale_activated_stats",
        "scale_major_stats",
        "scale_minor_stats",
        "scale_anisotropy_stats",
        "filter_3D_stats",
        "effective_scale_stats",
        "effective_scale_major_stats",
        "effective_scale_minor_stats",
        "effective_anisotropy_stats",
        "effective_volume_radius_stats",
        "opacity_activated_stats",
        "feature_dc_luma_stats",
        "feature_dc_chroma_delta_stats",
        "feature_rest_norm_stats",
        "rotation_norm_stats",
    ]
    return {key: ratio_delta(current.get(key, {}), rerun.get(key, {})) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current raw MipSplatting run with a rerun.")
    parser.add_argument("--current_model_path", required=True)
    parser.add_argument("--current_iteration", type=int, default=30000)
    parser.add_argument("--rerun_model_path", required=True)
    parser.add_argument("--rerun_iteration", type=int, default=30000)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    current_model = Path(args.current_model_path).expanduser().resolve()
    rerun_model = Path(args.rerun_model_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_cfg = read_cfg(current_model)
    rerun_cfg = read_cfg(rerun_model)
    current_ply = summarize_ply(ply_path(current_model, args.current_iteration))
    rerun_ply = summarize_ply(ply_path(rerun_model, args.rerun_iteration))

    summary = {
        "mode": "compare_raw_mip_runs_v0",
        "current_model_path": str(current_model),
        "current_iteration": int(args.current_iteration),
        "rerun_model_path": str(rerun_model),
        "rerun_iteration": int(args.rerun_iteration),
        "cfg": {
            "current": current_cfg,
            "rerun": rerun_cfg,
            "diff_ignoring_model_path": cfg_diff(current_cfg, rerun_cfg),
            "matching_ignoring_model_path": len(cfg_diff(current_cfg, rerun_cfg)) == 0,
        },
        "ply": {
            "current": current_ply,
            "rerun": rerun_ply,
            "count_delta": int(rerun_ply["gaussian_count"] - current_ply["gaussian_count"]),
            "count_ratio": float(rerun_ply["gaussian_count"] / current_ply["gaussian_count"])
            if current_ply["gaussian_count"]
            else None,
            "stats_delta": compare_stats(current_ply, rerun_ply),
        },
    }

    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] raw mip compare summary: {output_path}")


if __name__ == "__main__":
    main()
