from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import trimesh

from utils.sof_mesh_patch_enhancer_v0 import (
    SOFMeshPatchEnhancerMLP,
    build_carrier_payload_from_mesh,
    colorize_confidence,
    compute_vertex_geometry,
    load_triangle_mesh,
    mesh_bbox_normalizer,
    predict_offsets,
    save_payload_npz,
    stats_from_array,
    write_json,
)


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = SOFMeshPatchEnhancerMLP(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        num_layers=int(ckpt.get("num_layers", 4)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device=device)
    return model, ckpt


def main():
    parser = ArgumentParser(
        description=(
            "Apply SOF mesh patch enhancer v0 to an LRmesh, export an SRmesh preview, "
            "and emit mesh-bound carrier payload for meshGS testing."
        )
    )
    parser.add_argument("--lr_mesh_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--move_scale", type=float, default=1.0)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--max_delta_ratio", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--carriers_per_face", type=int, choices=[1, 3, 4, 6], default=1)
    parser.add_argument("--carrier_thickness_scale", type=float, default=0.05)
    parser.add_argument("--carrier_min_confidence", type=float, default=0.05)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[sof-mesh-patch-apply] CUDA unavailable; falling back to CPU.")
        args.device = "cpu"

    mesh = load_triangle_mesh(args.lr_mesh_path)
    bbox_center, bbox_diag = mesh_bbox_normalizer(mesh)
    geometry = compute_vertex_geometry(mesh, bbox_center=bbox_center, bbox_diag=bbox_diag)
    model, ckpt = load_model(args.checkpoint_path, args.device)
    pred = predict_offsets(
        model,
        geometry["features"],
        tangent_u=geometry["tangent_u"],
        tangent_v=geometry["tangent_v"],
        normals=geometry["normals"],
        bbox_diag=float(bbox_diag),
        device=args.device,
        batch_size=int(args.batch_size),
        max_delta_ratio=float(args.max_delta_ratio),
    )
    confidence = np.clip(pred["confidence"].reshape(-1), 0.0, 1.0).astype(np.float32)
    gate = np.power(confidence, float(args.confidence_power)).astype(np.float32)
    offset = pred["delta_world"] * gate[:, None] * float(args.move_scale)
    sr_vertices = geometry["vertices"] + offset.astype(np.float32)

    sr_mesh = trimesh.Trimesh(
        vertices=sr_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=False,
    )
    colored_mesh = trimesh.Trimesh(
        vertices=sr_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        vertex_colors=colorize_confidence(confidence),
        process=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sr_mesh_path = output_dir / "sr_mesh_patch_enhanced_v0.ply"
    colored_mesh_path = output_dir / "sr_mesh_patch_enhanced_v0_confidence.ply"
    prediction_path = output_dir / "sr_mesh_patch_vertex_predictions_v0.npz"
    carrier_payload_path = output_dir / "sr_mesh_patch_carrier_payload_v0.npz"
    summary_path = output_dir / "apply_sof_mesh_patch_enhancer_v0_summary.json"

    sr_mesh.export(sr_mesh_path)
    colored_mesh.export(colored_mesh_path)
    np.savez_compressed(
        prediction_path,
        lr_vertices=geometry["vertices"].astype(np.float32),
        sr_vertices=sr_vertices.astype(np.float32),
        delta_world=pred["delta_world"].astype(np.float32),
        applied_offset=offset.astype(np.float32),
        confidence=confidence.astype(np.float32),
    )

    carrier_payload = build_carrier_payload_from_mesh(
        sr_mesh,
        vertex_confidence=confidence,
        carriers_per_face=int(args.carriers_per_face),
        thickness_scale=float(args.carrier_thickness_scale),
        min_confidence=float(args.carrier_min_confidence),
    )
    save_payload_npz(carrier_payload_path, carrier_payload)

    summary = {
        "mode": "apply_sof_mesh_patch_enhancer_v0",
        "lr_mesh_path": str(Path(args.lr_mesh_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "checkpoint_dataset": ckpt.get("dataset_path"),
        "output_dir": str(output_dir.resolve()),
        "outputs": {
            "sr_mesh": str(sr_mesh_path.resolve()),
            "confidence_mesh": str(colored_mesh_path.resolve()),
            "vertex_predictions": str(prediction_path.resolve()),
            "carrier_payload": str(carrier_payload_path.resolve()),
        },
        "parameters": vars(args),
        "counts": {
            "vertices": int(sr_vertices.shape[0]),
            "faces": int(len(mesh.faces)),
            "carriers": int(carrier_payload["centers"].shape[0]),
            "valid_carriers": int(np.sum(carrier_payload["valid_mask"])),
        },
        "stats": {
            "pred_delta_norm": stats_from_array(np.linalg.norm(pred["delta_world"], axis=1)),
            "applied_offset_norm": stats_from_array(np.linalg.norm(offset, axis=1)),
            "confidence": stats_from_array(confidence),
            "carrier_confidence": stats_from_array(carrier_payload["confidence"]),
        },
        "mesh_bound_gs_note": (
            "The exported carrier payload stores face_ids + bary_coords. "
            "If the mesh moves again, recompute centers/normals/tangents/scales from those bindings instead of treating GS xyz as free."
        ),
    }
    write_json(summary_path, summary)
    print(f"[sof-mesh-patch-apply] saved SR mesh: {sr_mesh_path}")
    print(f"[sof-mesh-patch-apply] saved carrier payload: {carrier_payload_path}")
    print(f"[sof-mesh-patch-apply] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
