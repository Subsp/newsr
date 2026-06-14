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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score_gaussian_mesh_alignment_v0 import load_triangle_mesh


def stats_from_array(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        return {"count": 0, "finite_count": int(finite.sum())}
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


def tensor_to_numpy(payload: Dict[str, object], key: str, *, required: bool = True) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise KeyError(f"Missing required evidence field: {key}")
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.maximum(norm, eps)).astype(np.float32, copy=False)


def select_correspondences(
    *,
    payload: Dict[str, object],
    face_count: int,
    min_evidence_weight: float,
    max_d_norm: float,
    max_abs_offset: float,
    use_trusted_only: bool,
    max_correspondences: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    face_id = tensor_to_numpy(payload, "proxy_face_id").reshape(-1).astype(np.int64, copy=False)
    p = tensor_to_numpy(payload, "correspondence_source_point_p").astype(np.float32, copy=False)
    q = tensor_to_numpy(payload, "correspondence_mesh_point_q").astype(np.float32, copy=False)
    normal = normalize_np(tensor_to_numpy(payload, "correspondence_mesh_normal").astype(np.float32, copy=False))
    bary = tensor_to_numpy(payload, "correspondence_mesh_barycentric").astype(np.float32, copy=False)
    signed_offset = tensor_to_numpy(payload, "correspondence_signed_offset").reshape(-1).astype(np.float32, copy=False)
    tau_surface = tensor_to_numpy(payload, "correspondence_tau_surface").reshape(-1).astype(np.float32, copy=False)
    d_norm = tensor_to_numpy(payload, "correspondence_d_norm").reshape(-1).astype(np.float32, copy=False)
    evidence_weight = tensor_to_numpy(payload, "correspondence_evidence_weight").reshape(-1).astype(np.float32, copy=False)
    trusted = tensor_to_numpy(payload, "correspondence_trusted_mask").reshape(-1).astype(bool, copy=False)

    finite = (
        np.all(np.isfinite(p), axis=1)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(normal), axis=1)
        & np.all(np.isfinite(bary), axis=1)
        & np.isfinite(signed_offset)
        & np.isfinite(tau_surface)
        & np.isfinite(d_norm)
        & np.isfinite(evidence_weight)
    )
    valid = finite & (face_id >= 0) & (face_id < int(face_count))
    valid &= tau_surface > 1e-8
    valid &= evidence_weight >= float(min_evidence_weight)
    if float(max_d_norm) > 0.0:
        valid &= d_norm <= float(max_d_norm)
    if float(max_abs_offset) > 0.0:
        valid &= np.abs(signed_offset) <= float(max_abs_offset)
    if bool(use_trusted_only):
        valid &= trusted

    ids = np.flatnonzero(valid).astype(np.int64, copy=False)
    if ids.size == 0:
        raise RuntimeError("No usable mesh evidence correspondences after filtering.")
    if int(max_correspondences) > 0 and ids.size > int(max_correspondences):
        # Keep the strongest evidence but shuffle equal-score neighborhoods deterministically.
        rng = np.random.default_rng(int(seed))
        jitter = rng.uniform(0.0, 1e-6, size=ids.size).astype(np.float32)
        order = np.argsort(-(evidence_weight[ids] + jitter), kind="stable")
        ids = ids[order[: int(max_correspondences)]]

    bary_sel = bary[ids].astype(np.float32, copy=True)
    bary_sum = np.sum(bary_sel, axis=1, keepdims=True)
    bary_sel = bary_sel / np.maximum(bary_sum, 1e-8)
    return {
        "ids": ids,
        "face_id": face_id[ids],
        "p": p[ids],
        "q": q[ids],
        "normal": normal[ids],
        "barycentric": bary_sel,
        "signed_offset": signed_offset[ids],
        "tau_surface": tau_surface[ids],
        "d_norm": d_norm[ids],
        "evidence_weight": evidence_weight[ids],
        "trusted": trusted[ids],
        "all_valid_mask": valid,
    }


def build_active_vertex_problem(
    *,
    mesh: trimesh.Trimesh,
    selected: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    selected_faces = selected["face_id"].astype(np.int64, copy=False)
    tri_global = faces[selected_faces]
    active_vertices, inverse = np.unique(tri_global.reshape(-1), return_inverse=True)
    tri_local = inverse.reshape(-1, 3).astype(np.int64, copy=False)

    base_vertices = vertices[active_vertices].astype(np.float32, copy=False)
    normal_accum = np.zeros_like(base_vertices, dtype=np.float32)
    weighted_normals = selected["normal"] * selected["evidence_weight"][:, None]
    for column in range(3):
        contrib = weighted_normals * selected["barycentric"][:, column : column + 1]
        np.add.at(normal_accum, tri_local[:, column], contrib.astype(np.float32, copy=False))
    normal_len = np.linalg.norm(normal_accum, axis=1)
    weak = normal_len < 1e-8
    if np.any(weak):
        # Fallback only for degenerate barycentric/weight cases.
        for column in range(3):
            np.add.at(normal_accum, tri_local[:, column], selected["normal"].astype(np.float32, copy=False))
    active_normals = normalize_np(normal_accum)

    edges = np.concatenate(
        [tri_local[:, [0, 1]], tri_local[:, [1, 2]], tri_local[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0).astype(np.int64, copy=False)
    return {
        "faces": faces,
        "vertices": vertices,
        "tri_global": tri_global.astype(np.int64, copy=False),
        "tri_local": tri_local,
        "active_vertices": active_vertices.astype(np.int64, copy=False),
        "base_vertices": base_vertices,
        "active_normals": active_normals,
        "edges": edges,
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def robust_abs(residual: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(residual * residual + float(eps) * float(eps))


def export_point_cloud(path: Path, points: np.ndarray, color_rgb: tuple[int, int, int], max_points: int, seed: int) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0 or int(max_points) == 0:
        return
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        ids = rng.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[ids]
    colors = np.tile(np.asarray(color_rgb, dtype=np.uint8)[None, :], (points.shape[0], 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points, colors=colors).export(path)


def optimize_offsets(
    *,
    selected: Dict[str, np.ndarray],
    problem: Dict[str, np.ndarray],
    device: torch.device,
    iterations: int,
    lr: float,
    lambda_delta: float,
    lambda_lap: float,
    lambda_clip: float,
    offset_scale: float,
    offset_clip: float,
    robust_eps: float,
) -> Dict[str, object]:
    tri_local = torch.as_tensor(problem["tri_local"], dtype=torch.long, device=device)
    edges = torch.as_tensor(problem["edges"], dtype=torch.long, device=device)
    base_vertices = to_tensor(problem["base_vertices"], device)
    active_normals = to_tensor(problem["active_normals"], device)
    p = to_tensor(selected["p"], device)
    normal = to_tensor(selected["normal"], device)
    bary = to_tensor(selected["barycentric"], device)
    tau = to_tensor(selected["tau_surface"], device).clamp_min(1e-8)
    weight = to_tensor(selected["evidence_weight"], device)
    weight = weight / weight.mean().clamp_min(1e-8)

    delta = torch.nn.Parameter(torch.zeros((base_vertices.shape[0],), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([delta], lr=float(lr))
    scale = max(float(offset_scale), 1e-8)
    history = []

    def compute_residual(delta_values: torch.Tensor) -> torch.Tensor:
        moved_vertices = base_vertices + delta_values[:, None] * active_normals
        tri_vertices = moved_vertices[tri_local]
        q = torch.sum(tri_vertices * bary[:, :, None], dim=1)
        return torch.sum((p - q) * normal, dim=1) / tau

    with torch.no_grad():
        residual_before = compute_residual(delta.detach()).detach().cpu().numpy().astype(np.float32, copy=False)

    iterator = tqdm(range(int(iterations)), desc="refine mesh normal offsets")
    for step in iterator:
        optimizer.zero_grad(set_to_none=True)
        residual = compute_residual(delta)
        fit_loss = torch.mean(weight * robust_abs(residual, float(robust_eps)))
        delta_loss = torch.mean((delta / scale) ** 2)
        if edges.numel() > 0:
            lap_loss = torch.mean(((delta[edges[:, 0]] - delta[edges[:, 1]]) / scale) ** 2)
        else:
            lap_loss = torch.zeros((), dtype=torch.float32, device=device)
        if float(offset_clip) > 0.0:
            clip_loss = torch.mean((torch.relu(torch.abs(delta) - float(offset_clip)) / scale) ** 2)
        else:
            clip_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = fit_loss + float(lambda_delta) * delta_loss + float(lambda_lap) * lap_loss + float(lambda_clip) * clip_loss
        loss.backward()
        optimizer.step()
        if float(offset_clip) > 0.0:
            with torch.no_grad():
                delta.clamp_(min=-float(offset_clip), max=float(offset_clip))
        if step == 0 or (step + 1) % max(int(iterations) // 10, 1) == 0 or step + 1 == int(iterations):
            row = {
                "step": int(step + 1),
                "loss": float(loss.detach().cpu()),
                "fit": float(fit_loss.detach().cpu()),
                "delta": float(delta_loss.detach().cpu()),
                "lap": float(lap_loss.detach().cpu()),
                "clip": float(clip_loss.detach().cpu()),
            }
            history.append(row)
            iterator.set_postfix(loss=row["loss"], fit=row["fit"], lap=row["lap"])

    with torch.no_grad():
        residual_after = compute_residual(delta.detach()).detach().cpu().numpy().astype(np.float32, copy=False)
        delta_np = delta.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "delta": delta_np,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine a SOF mesh by fitting normal offsets to trusted mesh-boundedGS evidence correspondences."
    )
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_correspondences", type=int, default=120000)
    parser.add_argument("--min_evidence_weight", type=float, default=0.28)
    parser.add_argument("--max_d_norm", type=float, default=1.0)
    parser.add_argument("--max_abs_offset", type=float, default=0.03)
    parser.add_argument("--allow_untrusted", action="store_true")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--lambda_delta", type=float, default=0.02)
    parser.add_argument("--lambda_lap", type=float, default=0.20)
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--offset_scale", type=float, default=0.01)
    parser.add_argument("--offset_clip", type=float, default=0.02)
    parser.add_argument("--robust_eps", type=float, default=1e-3)
    parser.add_argument("--debug_point_cap", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_mesh_export", action="store_true")
    args = parser.parse_args()

    mesh_path = Path(args.mesh_path).expanduser().resolve()
    evidence_path = Path(args.evidence_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"evidence payload not found: {evidence_path}")

    mesh = load_triangle_mesh(mesh_path)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    payload = torch.load(str(evidence_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected evidence payload dict, got {type(payload)!r}")

    selected = select_correspondences(
        payload=payload,
        face_count=int(faces.shape[0]),
        min_evidence_weight=float(args.min_evidence_weight),
        max_d_norm=float(args.max_d_norm),
        max_abs_offset=float(args.max_abs_offset),
        use_trusted_only=not bool(args.allow_untrusted),
        max_correspondences=int(args.max_correspondences),
        seed=int(args.seed),
    )
    problem = build_active_vertex_problem(mesh=mesh, selected=selected)
    device = choose_device(str(args.device))
    result = optimize_offsets(
        selected=selected,
        problem=problem,
        device=device,
        iterations=int(args.iterations),
        lr=float(args.lr),
        lambda_delta=float(args.lambda_delta),
        lambda_lap=float(args.lambda_lap),
        lambda_clip=float(args.lambda_clip),
        offset_scale=float(args.offset_scale),
        offset_clip=float(args.offset_clip),
        robust_eps=float(args.robust_eps),
    )

    delta = np.asarray(result["delta"], dtype=np.float32)
    active_vertices = np.asarray(problem["active_vertices"], dtype=np.int64)
    active_normals = np.asarray(problem["active_normals"], dtype=np.float32)
    refined_vertices = np.asarray(problem["vertices"], dtype=np.float32).copy()
    refined_vertices[active_vertices] += delta[:, None] * active_normals

    refined_mesh_path = output_root / "refined_mesh_v0.ply"
    if not bool(args.skip_mesh_export):
        refined_mesh = trimesh.Trimesh(vertices=refined_vertices, faces=faces, process=False)
        refined_mesh.export(refined_mesh_path)

    offset_payload = {
        "version": "mesh_normal_offset_refine_v0",
        "mesh_path": str(mesh_path),
        "evidence_path": str(evidence_path),
        "active_vertex_ids": torch.from_numpy(active_vertices),
        "active_vertex_delta": torch.from_numpy(delta),
        "active_vertex_normals": torch.from_numpy(active_normals),
        "selected_correspondence_indices": torch.from_numpy(np.asarray(selected["ids"], dtype=np.int64)),
        "selected_face_ids": torch.from_numpy(np.asarray(selected["face_id"], dtype=np.int64)),
        "selected_barycentric": torch.from_numpy(np.asarray(selected["barycentric"], dtype=np.float32)),
        "selected_evidence_weight": torch.from_numpy(np.asarray(selected["evidence_weight"], dtype=np.float32)),
        "selected_signed_offset": torch.from_numpy(np.asarray(selected["signed_offset"], dtype=np.float32)),
        "selected_tau_surface": torch.from_numpy(np.asarray(selected["tau_surface"], dtype=np.float32)),
        "selected_d_norm": torch.from_numpy(np.asarray(selected["d_norm"], dtype=np.float32)),
        "normalized_residual_before": torch.from_numpy(np.asarray(result["residual_before"], dtype=np.float32)),
        "normalized_residual_after": torch.from_numpy(np.asarray(result["residual_after"], dtype=np.float32)),
    }
    payload_path = output_root / "mesh_normal_offset_refine_v0.pt"
    torch.save(offset_payload, payload_path)

    debug_root = output_root / "debug_pointclouds"
    export_point_cloud(debug_root / "selected_source_points_p.ply", selected["p"], (255, 80, 40), int(args.debug_point_cap), int(args.seed))
    export_point_cloud(debug_root / "selected_mesh_points_q.ply", selected["q"], (40, 120, 255), int(args.debug_point_cap), int(args.seed))
    export_point_cloud(
        debug_root / "refined_active_vertices.ply",
        refined_vertices[active_vertices],
        (60, 255, 120),
        int(args.debug_point_cap),
        int(args.seed),
    )

    residual_before = np.asarray(result["residual_before"], dtype=np.float32)
    residual_after = np.asarray(result["residual_after"], dtype=np.float32)
    summary = {
        "version": "mesh_normal_offset_refine_v0",
        "mesh_path": str(mesh_path),
        "evidence_path": str(evidence_path),
        "output": {
            "refined_mesh": str(refined_mesh_path) if not bool(args.skip_mesh_export) else None,
            "payload": str(payload_path),
            "summary": str(output_root / "mesh_normal_offset_refine_v0_summary.json"),
            "debug_pointclouds": str(debug_root),
        },
        "params": {
            "device": str(device),
            "max_correspondences": int(args.max_correspondences),
            "min_evidence_weight": float(args.min_evidence_weight),
            "max_d_norm": float(args.max_d_norm),
            "max_abs_offset": float(args.max_abs_offset),
            "use_trusted_only": not bool(args.allow_untrusted),
            "iterations": int(args.iterations),
            "lr": float(args.lr),
            "lambda_delta": float(args.lambda_delta),
            "lambda_lap": float(args.lambda_lap),
            "lambda_clip": float(args.lambda_clip),
            "offset_scale": float(args.offset_scale),
            "offset_clip": float(args.offset_clip),
            "robust_eps": float(args.robust_eps),
            "seed": int(args.seed),
        },
        "counts": {
            "mesh_vertices": int(np.asarray(problem["vertices"]).shape[0]),
            "mesh_faces": int(faces.shape[0]),
            "candidate_correspondences": int(np.asarray(selected["all_valid_mask"]).shape[0]),
            "selected_correspondences": int(selected["ids"].shape[0]),
            "active_vertices": int(active_vertices.shape[0]),
            "active_edges": int(np.asarray(problem["edges"]).shape[0]),
            "active_faces": int(np.unique(selected["face_id"]).shape[0]),
        },
        "selected": {
            "evidence_weight": stats_from_array(selected["evidence_weight"]),
            "signed_offset": stats_from_array(selected["signed_offset"]),
            "d_norm": stats_from_array(selected["d_norm"]),
            "tau_surface": stats_from_array(selected["tau_surface"]),
        },
        "optimization": {
            "history": result["history"],
            "normalized_residual_before": stats_from_array(residual_before),
            "normalized_residual_after": stats_from_array(residual_after),
            "abs_normalized_residual_before": stats_from_array(np.abs(residual_before)),
            "abs_normalized_residual_after": stats_from_array(np.abs(residual_after)),
            "delta": stats_from_array(delta),
            "abs_delta": stats_from_array(np.abs(delta)),
            "clip_hit_ratio": float(np.mean(np.abs(delta) >= max(float(args.offset_clip) * 0.999, 0.0))) if float(args.offset_clip) > 0.0 else 0.0,
        },
    }
    summary_path = output_root / "mesh_normal_offset_refine_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[mesh-normal-offset-refine-v0] refined mesh : {refined_mesh_path if not bool(args.skip_mesh_export) else '[skipped]'}")
    print(f"[mesh-normal-offset-refine-v0] payload      : {payload_path}")
    print(f"[mesh-normal-offset-refine-v0] summary      : {summary_path}")


if __name__ == "__main__":
    main()
