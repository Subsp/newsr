#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_gaussian_mesh_alignment_v0 import load_triangle_mesh, query_surface
from utils.sof_mesh_patch_enhancer_v0 import (
    BOUND_BARYCENTRIC_TEMPLATES,
    stats_from_array,
    write_json,
)


def normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.maximum(norm, eps)).astype(np.float32, copy=False)


def face_max_edge_lengths(mesh_obj: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float32)
    tri = vertices[faces]
    edge_01 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    edge_12 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    edge_20 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    return np.maximum.reduce([edge_01, edge_12, edge_20]).astype(np.float32, copy=False)


def compact_preview_ids(total: int, max_count: int, seed: int) -> np.ndarray:
    ids = np.arange(int(total), dtype=np.int64)
    if int(max_count) <= 0 or int(total) <= int(max_count):
        return ids
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(ids, size=int(max_count), replace=False)).astype(np.int64, copy=False)


def export_point_cloud(path: Path, points: np.ndarray, color_rgb: tuple[int, int, int], max_points: int, seed: int) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0 or int(max_points) == 0:
        return
    ids = compact_preview_ids(pts.shape[0], int(max_points), int(seed))
    colors = np.tile(np.asarray(color_rgb, dtype=np.uint8)[None, :], (ids.shape[0], 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(pts[ids], colors=colors).export(path)


def aggregate_faces(
    *,
    face_id: np.ndarray,
    face_count: int,
    evidence_weight: np.ndarray,
    trusted_mask: np.ndarray,
    normal_agreement: np.ndarray,
    d_norm: np.ndarray,
    tangent_norm: np.ndarray,
) -> Dict[str, np.ndarray]:
    valid = (face_id >= 0) & (face_id < int(face_count))
    face_proxy_count = np.bincount(face_id[valid], minlength=int(face_count)).astype(np.int32)
    trusted = valid & trusted_mask
    face_trusted_count = np.bincount(face_id[trusted], minlength=int(face_count)).astype(np.int32)

    weight_sum = np.zeros((int(face_count),), dtype=np.float32)
    normal_sum = np.zeros((int(face_count),), dtype=np.float32)
    d_norm_sum = np.zeros((int(face_count),), dtype=np.float32)
    tangent_sum = np.zeros((int(face_count),), dtype=np.float32)
    np.add.at(weight_sum, face_id[valid], evidence_weight[valid])
    np.add.at(normal_sum, face_id[valid], normal_agreement[valid])
    np.add.at(d_norm_sum, face_id[valid], d_norm[valid])
    np.add.at(tangent_sum, face_id[valid], tangent_norm[valid])

    denom = np.maximum(face_proxy_count.astype(np.float32), 1.0)
    return {
        "face_proxy_count": face_proxy_count,
        "face_trusted_proxy_count": face_trusted_count,
        "face_trusted_ratio": (face_trusted_count.astype(np.float32) / denom).astype(np.float32, copy=False),
        "face_evidence_score": (weight_sum / denom).astype(np.float32, copy=False),
        "face_normal_agreement_mean": (normal_sum / denom).astype(np.float32, copy=False),
        "face_d_norm_mean": (d_norm_sum / denom).astype(np.float32, copy=False),
        "face_tangent_norm_mean": (tangent_sum / denom).astype(np.float32, copy=False),
    }


def sample_face_correspondences(
    *,
    mesh_obj: trimesh.Trimesh,
    face_stride: int,
    max_faces: int,
    bary_layout: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    face_normals = normalize_np(np.asarray(mesh_obj.face_normals, dtype=np.float32))
    if faces.size == 0:
        raise ValueError("Base mesh has no faces.")
    if int(bary_layout) not in BOUND_BARYCENTRIC_TEMPLATES:
        supported = ", ".join(str(key) for key in sorted(BOUND_BARYCENTRIC_TEMPLATES))
        raise ValueError(f"Unsupported barycentric layout {bary_layout}. Supported: {supported}")

    face_ids = np.arange(faces.shape[0], dtype=np.int64)
    stride = max(int(face_stride), 1)
    if stride > 1:
        face_ids = face_ids[::stride]
    if int(max_faces) > 0 and face_ids.size > int(max_faces):
        rng = np.random.default_rng(int(seed))
        face_ids = np.sort(rng.choice(face_ids, size=int(max_faces), replace=False)).astype(np.int64, copy=False)

    bary_template = BOUND_BARYCENTRIC_TEMPLATES[int(bary_layout)]
    sample_face_ids = np.repeat(face_ids, bary_template.shape[0]).astype(np.int64, copy=False)
    bary = np.tile(bary_template, (face_ids.shape[0], 1)).astype(np.float32, copy=False)
    triangles = vertices[faces[sample_face_ids]]
    sample_points = np.sum(triangles * bary[:, :, None], axis=1).astype(np.float32, copy=False)
    sample_normals = face_normals[sample_face_ids].astype(np.float32, copy=False)
    return {
        "face_ids": sample_face_ids,
        "bary_coords": bary,
        "mesh_points": sample_points,
        "mesh_normals": sample_normals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build teacher-guided mesh evidence that preserves SOF mesh topology: "
            "an external smooth teacher mesh only contributes normal-direction offsets "
            "for a SOF-extracted base mesh."
        )
    )
    parser.add_argument("--base_mesh_path", required=True)
    parser.add_argument("--teacher_mesh_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--debug_root", default=None)
    parser.add_argument("--face_stride", type=int, default=1)
    parser.add_argument("--max_faces", type=int, default=0)
    parser.add_argument("--bary_layout", type=int, default=3, choices=sorted(BOUND_BARYCENTRIC_TEMPLATES))
    parser.add_argument("--surface_query_mode", choices=["auto", "exact_open3d", "sample"], default="auto")
    parser.add_argument("--mesh_surface_sample_count", type=int, default=500000)
    parser.add_argument("--surface_query_chunk_size", type=int, default=131072)
    parser.add_argument("--tau_edge_scale", type=float, default=1.0)
    parser.add_argument("--tau_floor", type=float, default=1e-4)
    parser.add_argument("--signed_offset_scale", type=float, default=1.0)
    parser.add_argument("--normal_agreement_power", type=float, default=1.0)
    parser.add_argument("--d_norm_sigma", type=float, default=1.0)
    parser.add_argument("--tangent_norm_sigma", type=float, default=0.75)
    parser.add_argument("--min_trusted_normal_agreement", type=float, default=0.60)
    parser.add_argument("--max_trusted_d_norm", type=float, default=1.5)
    parser.add_argument("--max_trusted_tangent_norm", type=float, default=0.75)
    parser.add_argument("--max_abs_offset", type=float, default=0.0)
    parser.add_argument("--debug_point_cap", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_mesh_path = Path(args.base_mesh_path).expanduser().resolve()
    teacher_mesh_path = Path(args.teacher_mesh_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if str(args.summary_path).strip()
        else output_path.with_name("teacher_guided_sof_mesh_evidence_v0_summary.json")
    )
    debug_root = (
        Path(args.debug_root).expanduser().resolve()
        if str(args.debug_root).strip()
        else output_path.with_name("teacher_guided_sof_mesh_evidence_v0_debug")
    )
    debug_root.mkdir(parents=True, exist_ok=True)

    base_mesh = load_triangle_mesh(base_mesh_path)
    teacher_mesh = load_triangle_mesh(teacher_mesh_path)
    samples = sample_face_correspondences(
        mesh_obj=base_mesh,
        face_stride=int(args.face_stride),
        max_faces=int(args.max_faces),
        bary_layout=int(args.bary_layout),
        seed=int(args.seed),
    )

    face_edge = face_max_edge_lengths(base_mesh)
    surface = query_surface(
        mesh_obj=teacher_mesh,
        points_xyz=samples["mesh_points"],
        mode=str(args.surface_query_mode),
        sample_count=int(args.mesh_surface_sample_count),
        chunk_size=int(args.surface_query_chunk_size),
    )

    face_id = samples["face_ids"].astype(np.int64, copy=False)
    mesh_point_q = samples["mesh_points"].astype(np.float32, copy=False)
    mesh_normal = normalize_np(samples["mesh_normals"].astype(np.float32, copy=False))
    teacher_point = surface["nearest_surface_point"].astype(np.float32, copy=False)
    teacher_normal = normalize_np(surface["nearest_surface_normal"].astype(np.float32, copy=False))

    raw_delta = teacher_point - mesh_point_q
    signed_offset_raw = np.sum(raw_delta * mesh_normal, axis=1).astype(np.float32, copy=False)
    signed_offset = (signed_offset_raw * float(args.signed_offset_scale)).astype(np.float32, copy=False)
    corrected_point = (mesh_point_q + signed_offset[:, None] * mesh_normal).astype(np.float32, copy=False)
    tangent_vec = teacher_point - corrected_point
    tangent_offset = np.linalg.norm(tangent_vec, axis=1).astype(np.float32, copy=False)

    tau_surface = np.full((face_id.shape[0],), float(args.tau_floor), dtype=np.float32)
    valid_face = (face_id >= 0) & (face_id < face_edge.shape[0])
    if np.any(valid_face):
        tau_surface[valid_face] = np.maximum(
            face_edge[face_id[valid_face]] * float(args.tau_edge_scale),
            float(args.tau_floor),
        ).astype(np.float32, copy=False)
    d_norm = (np.abs(signed_offset) / np.maximum(tau_surface, 1e-8)).astype(np.float32, copy=False)
    tangent_norm = (tangent_offset / np.maximum(tau_surface, 1e-8)).astype(np.float32, copy=False)
    normal_agreement = np.clip(
        0.5 * (1.0 + np.sum(mesh_normal * teacher_normal, axis=1)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    evidence_weight = np.ones((face_id.shape[0],), dtype=np.float32)
    if float(args.normal_agreement_power) > 0.0:
        evidence_weight *= np.power(normal_agreement, float(args.normal_agreement_power)).astype(np.float32, copy=False)
    if float(args.d_norm_sigma) > 0.0:
        evidence_weight *= np.exp(-0.5 * np.square(d_norm / float(args.d_norm_sigma))).astype(np.float32, copy=False)
    if float(args.tangent_norm_sigma) > 0.0:
        evidence_weight *= np.exp(-0.5 * np.square(tangent_norm / float(args.tangent_norm_sigma))).astype(np.float32, copy=False)
    evidence_weight = np.clip(evidence_weight, 0.0, 1.0).astype(np.float32, copy=False)

    trusted_mask = valid_face.copy()
    trusted_mask &= np.isfinite(signed_offset) & np.isfinite(d_norm) & np.isfinite(tangent_norm) & np.isfinite(normal_agreement)
    trusted_mask &= normal_agreement >= float(args.min_trusted_normal_agreement)
    trusted_mask &= d_norm <= float(args.max_trusted_d_norm)
    trusted_mask &= tangent_norm <= float(args.max_trusted_tangent_norm)
    if float(args.max_abs_offset) > 0.0:
        trusted_mask &= np.abs(signed_offset) <= float(args.max_abs_offset)

    face_stats = aggregate_faces(
        face_id=face_id,
        face_count=int(np.asarray(base_mesh.faces).shape[0]),
        evidence_weight=evidence_weight,
        trusted_mask=trusted_mask,
        normal_agreement=normal_agreement,
        d_norm=d_norm,
        tangent_norm=tangent_norm,
    )

    payload = {
        "version": "teacher_guided_sof_mesh_evidence_v0",
        "base_mesh_path": str(base_mesh_path),
        "teacher_mesh_path": str(teacher_mesh_path),
        "surface_query_mode_used": surface["surface_query_mode_used"],
        "proxy_face_id": torch.from_numpy(face_id.astype(np.int64, copy=False)),
        "proxy_trusted_mask": torch.from_numpy(trusted_mask.astype(np.bool_, copy=False)),
        "proxy_evidence_score": torch.from_numpy(evidence_weight.astype(np.float32, copy=False)),
        "proxy_teacher_normal_agreement": torch.from_numpy(normal_agreement.astype(np.float32, copy=False)),
        "proxy_teacher_tangent_norm": torch.from_numpy(tangent_norm.astype(np.float32, copy=False)),
        "proxy_signed_offset": torch.from_numpy(signed_offset.astype(np.float32, copy=False)),
        "proxy_tau_surface": torch.from_numpy(tau_surface.astype(np.float32, copy=False)),
        "proxy_d_norm": torch.from_numpy(d_norm.astype(np.float32, copy=False)),
        "correspondence_source_point_p": torch.from_numpy(corrected_point.astype(np.float32, copy=False)),
        "correspondence_mesh_point_q": torch.from_numpy(mesh_point_q.astype(np.float32, copy=False)),
        "correspondence_mesh_normal": torch.from_numpy(mesh_normal.astype(np.float32, copy=False)),
        "correspondence_mesh_barycentric": torch.from_numpy(samples["bary_coords"].astype(np.float32, copy=False)),
        "correspondence_signed_offset": torch.from_numpy(signed_offset.astype(np.float32, copy=False)),
        "correspondence_tau_surface": torch.from_numpy(tau_surface.astype(np.float32, copy=False)),
        "correspondence_d_norm": torch.from_numpy(d_norm.astype(np.float32, copy=False)),
        "correspondence_evidence_weight": torch.from_numpy(evidence_weight.astype(np.float32, copy=False)),
        "correspondence_trusted_mask": torch.from_numpy(trusted_mask.astype(np.bool_, copy=False)),
        "teacher_nearest_surface_point": torch.from_numpy(teacher_point.astype(np.float32, copy=False)),
        "teacher_nearest_surface_normal": torch.from_numpy(teacher_normal.astype(np.float32, copy=False)),
        "teacher_surface_distance": torch.from_numpy(surface["surface_distance"].astype(np.float32, copy=False)),
        "teacher_nearest_face_id": torch.from_numpy(surface["nearest_face_id"].astype(np.int64, copy=False)),
        "face_proxy_count": torch.from_numpy(face_stats["face_proxy_count"].astype(np.int32, copy=False)),
        "face_trusted_proxy_count": torch.from_numpy(face_stats["face_trusted_proxy_count"].astype(np.int32, copy=False)),
        "face_trusted_ratio": torch.from_numpy(face_stats["face_trusted_ratio"].astype(np.float32, copy=False)),
        "face_evidence_score": torch.from_numpy(face_stats["face_evidence_score"].astype(np.float32, copy=False)),
    }
    torch.save(payload, str(output_path))

    export_point_cloud(debug_root / "base_mesh_samples_q.ply", mesh_point_q, (40, 120, 255), int(args.debug_point_cap), int(args.seed))
    export_point_cloud(debug_root / "teacher_nearest_points.ply", teacher_point, (255, 120, 40), int(args.debug_point_cap), int(args.seed))
    export_point_cloud(debug_root / "corrected_normal_offset_points.ply", corrected_point, (80, 255, 120), int(args.debug_point_cap), int(args.seed))

    summary = {
        "version": "teacher_guided_sof_mesh_evidence_v0",
        "base_mesh_path": str(base_mesh_path),
        "teacher_mesh_path": str(teacher_mesh_path),
        "output_path": str(output_path),
        "surface_query_mode_requested": str(args.surface_query_mode),
        "surface_query_mode_used": surface["surface_query_mode_used"],
        "params": {
            "face_stride": int(args.face_stride),
            "max_faces": int(args.max_faces),
            "bary_layout": int(args.bary_layout),
            "mesh_surface_sample_count": int(args.mesh_surface_sample_count),
            "surface_query_chunk_size": int(args.surface_query_chunk_size),
            "tau_edge_scale": float(args.tau_edge_scale),
            "tau_floor": float(args.tau_floor),
            "signed_offset_scale": float(args.signed_offset_scale),
            "normal_agreement_power": float(args.normal_agreement_power),
            "d_norm_sigma": float(args.d_norm_sigma),
            "tangent_norm_sigma": float(args.tangent_norm_sigma),
            "min_trusted_normal_agreement": float(args.min_trusted_normal_agreement),
            "max_trusted_d_norm": float(args.max_trusted_d_norm),
            "max_trusted_tangent_norm": float(args.max_trusted_tangent_norm),
            "max_abs_offset": float(args.max_abs_offset),
        },
        "counts": {
            "base_mesh_vertices": int(np.asarray(base_mesh.vertices).shape[0]),
            "base_mesh_faces": int(np.asarray(base_mesh.faces).shape[0]),
            "teacher_mesh_vertices": int(np.asarray(teacher_mesh.vertices).shape[0]),
            "teacher_mesh_faces": int(np.asarray(teacher_mesh.faces).shape[0]),
            "sampled_faces": int(np.unique(face_id).shape[0]),
            "sampled_correspondences": int(face_id.shape[0]),
            "trusted_correspondences": int(np.sum(trusted_mask)),
            "trusted_faces": int(np.sum(face_stats["face_trusted_proxy_count"] > 0)),
        },
        "stats": {
            "signed_offset": stats_from_array(signed_offset),
            "abs_signed_offset": stats_from_array(np.abs(signed_offset)),
            "d_norm": stats_from_array(d_norm),
            "tangent_norm": stats_from_array(tangent_norm),
            "normal_agreement": stats_from_array(normal_agreement),
            "evidence_weight": stats_from_array(evidence_weight),
            "teacher_surface_distance": stats_from_array(surface["surface_distance"]),
            "face_trusted_ratio": stats_from_array(face_stats["face_trusted_ratio"]),
            "face_evidence_score": stats_from_array(face_stats["face_evidence_score"]),
        },
        "note": (
            "The teacher mesh only contributes normal-direction offsets on top of the base SOF mesh. "
            "Tangential disagreement is turned into confidence gating instead of direct vertex sliding, "
            "so the refined mesh keeps SOF's own topology and local feature layout."
        ),
        "outputs": {
            "payload": str(output_path),
            "summary": str(summary_path),
            "debug_root": str(debug_root),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"[teacher-guided-sof-mesh-evidence-v0] payload : {output_path}")
    print(f"[teacher-guided-sof-mesh-evidence-v0] summary : {summary_path}")


if __name__ == "__main__":
    main()
