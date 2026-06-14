#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_bool_mask(payload: dict, key: str) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"Mask key '{key}' not found in payload.")
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(dtype=torch.bool, device="cpu").reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build derived boolean Gaussian masks from two candidate layers.")
    parser.add_argument("--input_payload", required=True)
    parser.add_argument("--output_payload", required=True)
    parser.add_argument("--summary_path", default="")
    parser.add_argument("--geometry_key", default="geometry_candidate_mask")
    parser.add_argument("--bright_key", default="base_candidate_mask")
    parser.add_argument("--residual_key", default="geometry_minus_bright_mask")
    parser.add_argument("--union_key", default="geometry_or_bright_mask")
    parser.add_argument("--intersection_key", default="geometry_and_bright_mask")
    args = parser.parse_args()

    input_payload = Path(args.input_payload).expanduser().resolve()
    output_payload = Path(args.output_payload).expanduser().resolve()
    payload = torch.load(input_payload, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload)!r}: {input_payload}")

    geometry = _load_bool_mask(payload, str(args.geometry_key))
    bright = _load_bool_mask(payload, str(args.bright_key))
    if int(geometry.shape[0]) != int(bright.shape[0]):
        raise ValueError(
            f"Mask length mismatch: {args.geometry_key}={int(geometry.shape[0])}, "
            f"{args.bright_key}={int(bright.shape[0])}"
        )

    residual = geometry & ~bright
    union = geometry | bright
    intersection = geometry & bright

    out = dict(payload)
    out[str(args.geometry_key)] = geometry
    out[str(args.bright_key)] = bright
    out[str(args.residual_key)] = residual
    out[str(args.union_key)] = union
    out[str(args.intersection_key)] = intersection
    output_payload.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_payload)

    total = int(geometry.shape[0])
    summary = {
        "mode": "build_gaussian_mask_layers_v0",
        "input_payload": str(input_payload),
        "output_payload": str(output_payload),
        "geometry_key": str(args.geometry_key),
        "bright_key": str(args.bright_key),
        "residual_key": str(args.residual_key),
        "union_key": str(args.union_key),
        "intersection_key": str(args.intersection_key),
        "counts": {
            "total_gaussians": total,
            "geometry": int(geometry.sum().item()),
            "bright": int(bright.sum().item()),
            "residual_geometry_minus_bright": int(residual.sum().item()),
            "union": int(union.sum().item()),
            "intersection": int(intersection.sum().item()),
        },
        "ratios": {
            "geometry": float(geometry.sum().item() / max(total, 1)),
            "bright": float(bright.sum().item() / max(total, 1)),
            "residual_geometry_minus_bright": float(residual.sum().item() / max(total, 1)),
            "union": float(union.sum().item() / max(total, 1)),
            "intersection": float(intersection.sum().item() / max(total, 1)),
        },
    }
    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else output_payload.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] output payload: {output_payload}")
    print(f"[done] summary       : {summary_path}")


if __name__ == "__main__":
    main()
