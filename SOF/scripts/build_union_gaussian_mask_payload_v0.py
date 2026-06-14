from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch


def _split_input_spec(spec: str) -> Tuple[Path, str]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("Empty input spec is not allowed.")
    if ":" in raw:
        candidate_path, candidate_key = raw.rsplit(":", 1)
        path = Path(candidate_path).expanduser().resolve()
        if path.exists():
            return path, candidate_key.strip()
    return Path(raw).expanduser().resolve(), ""


def _load_mask_from_object(obj, *, key: str, total_count: int | None) -> tuple[torch.Tensor, int]:
    if torch.is_tensor(obj):
        tensor = obj.detach().cpu().reshape(-1)
        if tensor.dtype == torch.bool:
            count = int(tensor.shape[0])
            if total_count is not None and count != int(total_count):
                raise ValueError(f"Boolean mask length mismatch: {count} vs expected {total_count}")
            return tensor.to(dtype=torch.bool), count
        if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            ids = tensor.to(dtype=torch.int64)
            inferred_total = int(total_count) if total_count is not None else int(ids.max().item()) + 1 if ids.numel() > 0 else 0
            out = torch.zeros((inferred_total,), dtype=torch.bool)
            if ids.numel() > 0:
                if int(ids.min().item()) < 0:
                    raise ValueError("Selected ids must be non-negative.")
                if int(ids.max().item()) >= inferred_total:
                    raise ValueError("Selected ids exceed inferred total count.")
                out[ids] = True
            return out, inferred_total
        raise ValueError(f"Unsupported tensor dtype for mask input: {tensor.dtype}")

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported payload type: {type(obj)!r}")

    if key:
        if key not in obj:
            raise KeyError(f"Key '{key}' not found in payload.")
        return _load_mask_from_object(obj[key], key="", total_count=total_count)

    if "selected_mask" in obj:
        return _load_mask_from_object(obj["selected_mask"], key="", total_count=total_count)
    if "selected_ids" in obj:
        return _load_mask_from_object(obj["selected_ids"], key="", total_count=total_count)
    raise KeyError("Payload must contain 'selected_mask' or 'selected_ids' when no explicit key is provided.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a union Gaussian mask payload from multiple mask tensors/payloads.")
    parser.add_argument("--input", action="append", required=True, help="Input spec: /path/to/file.pt or /path/to/payload.pt:key")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_key", default="selected_mask")
    parser.add_argument("--expected_count", type=int, default=0)
    args = parser.parse_args()

    input_specs = [_split_input_spec(spec) for spec in args.input]
    expected_count = int(args.expected_count) if int(args.expected_count) > 0 else None

    union_mask: torch.Tensor | None = None
    total_count = expected_count
    component_summaries: List[dict] = []

    for path, key in input_specs:
        if not path.is_file():
            raise FileNotFoundError(f"Input mask payload not found: {path}")
        obj = torch.load(path, map_location="cpu")
        mask, inferred_count = _load_mask_from_object(obj, key=key, total_count=total_count)
        if total_count is None:
            total_count = int(inferred_count)
        if int(mask.shape[0]) != int(total_count):
            raise ValueError(f"Mask length mismatch after inference: {mask.shape[0]} vs {total_count}")
        union_mask = mask.clone() if union_mask is None else (union_mask | mask)
        component_summaries.append(
            {
                "input": str(path),
                "key": str(key) if key else None,
                "selected_count": int(mask.sum().item()),
            }
        )

    if union_mask is None or total_count is None:
        raise RuntimeError("No union mask could be built.")

    selected_ids = torch.nonzero(union_mask, as_tuple=False).squeeze(1).to(torch.int64)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "union_gaussian_mask_payload_v0",
        "selected_mask": union_mask.to(torch.bool),
        "selected_ids": selected_ids,
        "total_count": int(total_count),
        "selected_count": int(selected_ids.numel()),
        "inputs": component_summaries,
    }
    torch.save(payload, output_path)

    summary = {
        "version": "union_gaussian_mask_payload_v0",
        "output_path": str(output_path),
        "output_key": str(args.output_key),
        "total_count": int(total_count),
        "selected_count": int(selected_ids.numel()),
        "selected_ratio": float(selected_ids.numel() / max(int(total_count), 1)),
        "inputs": component_summaries,
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
