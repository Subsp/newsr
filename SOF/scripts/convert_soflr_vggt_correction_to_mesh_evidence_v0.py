#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sof_mesh_patch_enhancer_v0 import load_payload_any, load_triangle_mesh


def stats_from_array(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        return {"count": int(values.size), "finite_count": int(finite.sum())}
    return {
        "count": int(values.size),
        "finite_count": int(finite.sum()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    return np.divide(
        num,
        np.maximum(den, 1e-8),
        out=np.zeros_like(num, dtype=np.float32),
        where=den > 0,
    ).astype(np.float32, copy=False)


def payload_array(
    payload: Dict[str, np.ndarray],
    key: str,
    *,
    dtype: np.dtype | None = None,
    ndim: int | None = None,
    channels: int | None = None,
    required: bool = True,
) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise KeyError(f"Missing required correction payload field: {key}")
        return None
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    if ndim is not None and int(array.ndim) != int(ndim):
        raise ValueError(f"Payload field '{key}' expected ndim={ndim}, got shape={array.shape}")
    if channels is not None and (array.ndim != 2 or int(array.shape[1]) != int(channels)):
        raise ValueError(f"Payload field '{key}' expected second dim={channels}, got shape={array.shape}")
    return array


def face_max_edge_lengths(mesh_obj) -> np.ndarray:
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float32)
    triangles = vertices[faces]
    edge_01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge_12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge_20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    return np.maximum.reduce([edge_01, edge_12, edge_20]).astype(np.float32, copy=False)


def normalize_unit(values: np.ndarray, explicit_ref: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    if float(explicit_ref) > 0.0:
        ref = float(explicit_ref)
    else:
        ref = float(np.percentile(finite, 95.0))
    ref = max(ref, 1e-6)
    return np.clip(arr / ref, 0.0, 1.0).astype(np.float32, copy=False)


def aggregate_faces(
    *,
    face_id: np.ndarray,
    face_count: int,
    evidence_weight: np.ndarray,
    trusted_mask: np.ndarray,
    confidence: np.ndarray,
    view_count: np.ndarray,
    abs_signed_offset: np.ndarray,
) -> Dict[str, np.ndarray]:
    valid_face = (face_id >= 0) & (face_id < int(face_count))
    face_proxy_count = np.bincount(face_id[valid_face], minlength=int(face_count)).astype(np.int32)
    trusted_face = valid_face & trusted_mask
    face_trusted_proxy_count = np.bincount(face_id[trusted_face], minlength=int(face_count)).astype(np.int32)

    face_weight_sum = np.zeros((int(face_count),), dtype=np.float32)
    face_conf_sum = np.zeros((int(face_count),), dtype=np.float32)
    face_view_sum = np.zeros((int(face_count),), dtype=np.float32)
    face_offset_sum = np.zeros((int(face_count),), dtype=np.float32)
    np.add.at(face_weight_sum, face_id[valid_face], evidence_weight[valid_face])
    np.add.at(face_conf_sum, face_id[valid_face], confidence[valid_face])
    np.add.at(face_view_sum, face_id[valid_face], view_count[valid_face])
    np.add.at(face_offset_sum, face_id[valid_face], abs_signed_offset[valid_face])

    return {
        "face_proxy_count": face_proxy_count,
        "face_trusted_proxy_count": face_trusted_proxy_count,
        "face_trusted_ratio": safe_divide(face_trusted_proxy_count.astype(np.float32), face_proxy_count.astype(np.float32)),
        "face_evidence_score": safe_divide(face_weight_sum, face_proxy_count.astype(np.float32)),
        "face_confidence_mean": safe_divide(face_conf_sum, face_proxy_count.astype(np.float32)),
        "face_view_count_mean": safe_divide(face_view_sum, face_proxy_count.astype(np.float32)),
        "face_abs_signed_offset_mean": safe_divide(face_offset_sum, face_proxy_count.astype(np.float32)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SOFLR VGGT bound-GS correction payloads into the mesh evidence format "
            "expected by refine_mesh_from_bounded_evidence_v0.py."
        )
    )
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--correction_payload", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tau_edge_scale", type=float, default=1.0)
    parser.add_argument("--tau_floor", type=float, default=1e-4)
    parser.add_argument("--signed_offset_scale", type=float, default=1.0)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--weight_sum_power", type=float, default=0.0)
    parser.add_argument("--weight_sum_ref", type=float, default=0.0)
    parser.add_argument("--view_count_power", type=float, default=0.0)
    parser.add_argument("--view_count_ref", type=float, default=0.0)
    parser.add_argument("--min_trusted_confidence", type=float, default=0.05)
    parser.add_argument("--min_trusted_view_count", type=int, default=2)
    parser.add_argument("--min_trusted_weight_sum", type=float, default=0.0)
    parser.add_argument("--trusted_max_d_norm", type=float, default=1.0)
    parser.add_argument("--trusted_max_abs_offset", type=float, default=0.0)
    args = parser.parse_args()

    mesh_path = Path(args.mesh_path).expanduser().resolve()
    correction_payload_path = Path(args.correction_payload).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_triangle_mesh(str(mesh_path))
    face_count = int(np.asarray(mesh.faces).shape[0])
    edge_per_face = face_max_edge_lengths(mesh)
    payload = load_payload_any(str(correction_payload_path))

    face_id = payload_array(payload, "face_ids", dtype=np.int64, ndim=1)
    bary = payload_array(payload, "bary_coords", dtype=np.float32, ndim=2, channels=3)
    mesh_point_q = payload_array(payload, "surface_points", dtype=np.float32, ndim=2, channels=3)
    corrected_point = payload_array(payload, "corrected_surface_points", dtype=np.float32, ndim=2, channels=3)
    normal = payload_array(payload, "normals", dtype=np.float32, ndim=2, channels=3)
    confidence = payload_array(payload, "correction_confidence", dtype=np.float32, ndim=1)
    weight_sum = payload_array(payload, "correction_weight_sum", dtype=np.float32, ndim=1, required=False)
    view_count = payload_array(payload, "correction_view_count", dtype=np.float32, ndim=1, required=False)
    signed_offset_raw = payload_array(payload, "normal_correction", dtype=np.float32, ndim=1, required=False)
    original_xyz = payload_array(payload, "original_xyz", dtype=np.float32, ndim=2, channels=3, required=False)

    count = int(face_id.shape[0])
    for name, array in (
        ("bary_coords", bary),
        ("surface_points", mesh_point_q),
        ("corrected_surface_points", corrected_point),
        ("normals", normal),
        ("correction_confidence", confidence),
    ):
        if int(array.shape[0]) != count:
            raise ValueError(f"Field '{name}' length mismatch: expected {count}, got {array.shape[0]}")

    if weight_sum is None:
        weight_sum = np.zeros((count,), dtype=np.float32)
    if view_count is None:
        view_count = np.zeros((count,), dtype=np.float32)

    if signed_offset_raw is None:
        signed_offset_raw = np.sum((corrected_point - mesh_point_q) * normal, axis=1).astype(np.float32, copy=False)
    signed_offset = (signed_offset_raw.astype(np.float32, copy=False) * float(args.signed_offset_scale)).astype(np.float32, copy=False)
    corrected_point_scaled = (mesh_point_q + signed_offset[:, None] * normal).astype(np.float32, copy=False)

    tau_surface = np.full((count,), float(args.tau_floor), dtype=np.float32)
    valid_face = (face_id >= 0) & (face_id < face_count)
    if np.any(valid_face):
        tau_surface[valid_face] = np.maximum(
            edge_per_face[face_id[valid_face]] * float(args.tau_edge_scale),
            float(args.tau_floor),
        ).astype(np.float32, copy=False)
    d_norm = safe_divide(np.abs(signed_offset), tau_surface)

    confidence_unit = np.clip(confidence.astype(np.float32, copy=False), 0.0, 1.0)
    weight_sum_unit = normalize_unit(weight_sum, float(args.weight_sum_ref))
    view_count_unit = normalize_unit(view_count, float(args.view_count_ref))

    evidence_weight = np.ones((count,), dtype=np.float32)
    if float(args.confidence_power) > 0.0:
        evidence_weight *= np.power(confidence_unit, float(args.confidence_power)).astype(np.float32, copy=False)
    if float(args.weight_sum_power) > 0.0:
        evidence_weight *= np.power(weight_sum_unit, float(args.weight_sum_power)).astype(np.float32, copy=False)
    if float(args.view_count_power) > 0.0:
        evidence_weight *= np.power(view_count_unit, float(args.view_count_power)).astype(np.float32, copy=False)
    evidence_weight = np.clip(evidence_weight, 0.0, 1.0).astype(np.float32, copy=False)

    trusted_mask = np.isfinite(signed_offset) & np.isfinite(d_norm)
    trusted_mask &= valid_face
    trusted_mask &= confidence_unit >= float(args.min_trusted_confidence)
    trusted_mask &= view_count >= float(args.min_trusted_view_count)
    trusted_mask &= weight_sum >= float(args.min_trusted_weight_sum)
    if float(args.trusted_max_d_norm) > 0.0:
        trusted_mask &= d_norm <= float(args.trusted_max_d_norm)
    if float(args.trusted_max_abs_offset) > 0.0:
        trusted_mask &= np.abs(signed_offset) <= float(args.trusted_max_abs_offset)

    face = aggregate_faces(
        face_id=face_id,
        face_count=face_count,
        evidence_weight=evidence_weight,
        trusted_mask=trusted_mask,
        confidence=confidence_unit,
        view_count=view_count.astype(np.float32, copy=False),
        abs_signed_offset=np.abs(signed_offset).astype(np.float32, copy=False),
    )

    payload_out = {
        "version": "soflr_vggt_mesh_evidence_v0",
        "mesh_path": str(mesh_path),
        "correction_payload": str(correction_payload_path),
        "proxy_face_id": torch.from_numpy(face_id.astype(np.int64, copy=False)),
        "proxy_trusted_mask": torch.from_numpy(trusted_mask.astype(np.bool_, copy=False)),
        "proxy_evidence_score": torch.from_numpy(evidence_weight.astype(np.float32, copy=False)),
        "proxy_correction_confidence": torch.from_numpy(confidence_unit.astype(np.float32, copy=False)),
        "proxy_correction_weight_sum": torch.from_numpy(weight_sum.astype(np.float32, copy=False)),
        "proxy_correction_view_count": torch.from_numpy(view_count.astype(np.float32, copy=False)),
        "proxy_signed_offset": torch.from_numpy(signed_offset.astype(np.float32, copy=False)),
        "proxy_tau_surface": torch.from_numpy(tau_surface.astype(np.float32, copy=False)),
        "proxy_d_norm": torch.from_numpy(d_norm.astype(np.float32, copy=False)),
        "correspondence_source_point_p": torch.from_numpy(corrected_point_scaled.astype(np.float32, copy=False)),
        "correspondence_mesh_point_q": torch.from_numpy(mesh_point_q.astype(np.float32, copy=False)),
        "correspondence_mesh_normal": torch.from_numpy(normal.astype(np.float32, copy=False)),
        "correspondence_mesh_barycentric": torch.from_numpy(bary.astype(np.float32, copy=False)),
        "correspondence_signed_offset": torch.from_numpy(signed_offset.astype(np.float32, copy=False)),
        "correspondence_tau_surface": torch.from_numpy(tau_surface.astype(np.float32, copy=False)),
        "correspondence_d_norm": torch.from_numpy(d_norm.astype(np.float32, copy=False)),
        "correspondence_evidence_weight": torch.from_numpy(evidence_weight.astype(np.float32, copy=False)),
        "correspondence_trusted_mask": torch.from_numpy(trusted_mask.astype(np.bool_, copy=False)),
        **{key: torch.from_numpy(value) for key, value in face.items()},
    }
    if original_xyz is not None and int(original_xyz.shape[0]) == count:
        payload_out["proxy_xyz"] = torch.from_numpy(original_xyz.astype(np.float32, copy=False))

    torch.save(payload_out, str(output_path))

    summary = {
        "version": "soflr_vggt_mesh_evidence_v0",
        "mesh_path": str(mesh_path),
        "correction_payload": str(correction_payload_path),
        "output_path": str(output_path),
        "parameters": {
            "tau_edge_scale": float(args.tau_edge_scale),
            "tau_floor": float(args.tau_floor),
            "signed_offset_scale": float(args.signed_offset_scale),
            "confidence_power": float(args.confidence_power),
            "weight_sum_power": float(args.weight_sum_power),
            "weight_sum_ref": float(args.weight_sum_ref),
            "view_count_power": float(args.view_count_power),
            "view_count_ref": float(args.view_count_ref),
            "min_trusted_confidence": float(args.min_trusted_confidence),
            "min_trusted_view_count": int(args.min_trusted_view_count),
            "min_trusted_weight_sum": float(args.min_trusted_weight_sum),
            "trusted_max_d_norm": float(args.trusted_max_d_norm),
            "trusted_max_abs_offset": float(args.trusted_max_abs_offset),
        },
        "counts": {
            "correspondences": count,
            "valid_face_correspondences": int(np.sum(valid_face)),
            "trusted_correspondences": int(np.sum(trusted_mask)),
            "active_faces": int(np.unique(face_id[valid_face]).shape[0]) if np.any(valid_face) else 0,
            "mesh_faces": int(face_count),
        },
        "stats": {
            "confidence": stats_from_array(confidence_unit),
            "weight_sum": stats_from_array(weight_sum),
            "view_count": stats_from_array(view_count),
            "signed_offset": stats_from_array(signed_offset),
            "abs_signed_offset": stats_from_array(np.abs(signed_offset)),
            "tau_surface": stats_from_array(tau_surface),
            "d_norm": stats_from_array(d_norm),
            "evidence_weight": stats_from_array(evidence_weight),
            "trusted_abs_signed_offset": stats_from_array(np.abs(signed_offset[trusted_mask])),
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
