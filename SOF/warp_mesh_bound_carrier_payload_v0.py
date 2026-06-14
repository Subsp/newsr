from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils.sof_mesh_patch_enhancer_v0 import (
    load_payload_any,
    load_triangle_mesh,
    rebind_carrier_payload_to_mesh,
    save_payload_npz,
    stats_from_array,
    write_json,
)


def main():
    parser = ArgumentParser(
        description=(
            "Move an existing mesh-bound carrier payload onto a new mesh by preserving "
            "face_id + barycentric bindings and recomputing center/normal/tangent/scale."
        )
    )
    parser.add_argument("--carrier_payload", type=str, required=True)
    parser.add_argument("--target_mesh_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--thickness_scale", type=float, default=0.0)
    args = parser.parse_args()

    payload = load_payload_any(args.carrier_payload)
    mesh = load_triangle_mesh(args.target_mesh_path)
    thickness_scale = float(args.thickness_scale) if float(args.thickness_scale) > 0.0 else None
    rebound = rebind_carrier_payload_to_mesh(payload, mesh, thickness_scale=thickness_scale)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload_npz(output_path, rebound)

    confidence = np.asarray(rebound.get("confidence", np.zeros((rebound["centers"].shape[0],), dtype=np.float32))).reshape(-1)
    valid_mask = np.asarray(rebound.get("valid_mask", np.ones((rebound["centers"].shape[0],), dtype=np.bool_))).reshape(-1).astype(bool)
    summary = {
        "mode": "warp_mesh_bound_carrier_payload_v0",
        "carrier_payload": str(Path(args.carrier_payload).resolve()),
        "target_mesh_path": str(Path(args.target_mesh_path).resolve()),
        "output_path": str(output_path.resolve()),
        "parameters": {
            "thickness_scale": thickness_scale,
        },
        "counts": {
            "carriers": int(rebound["centers"].shape[0]),
            "valid_carriers": int(np.sum(valid_mask)),
            "target_mesh_vertices": int(len(mesh.vertices)),
            "target_mesh_faces": int(len(mesh.faces)),
        },
        "stats": {
            "confidence": stats_from_array(confidence),
            "scale_u": stats_from_array(rebound["scale_u"]),
            "scale_v": stats_from_array(rebound["scale_v"]),
            "scale_n": stats_from_array(rebound["scale_n"]),
        },
        "note": (
            "This is the intended motion model for mesh-boundedGS: keep binding coordinates fixed, "
            "derive geometry from the moved mesh."
        ),
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"[warp-mesh-bound-carrier] saved: {output_path}")
    print(f"[warp-mesh-bound-carrier] summary: {summary_path}")


if __name__ == "__main__":
    main()
