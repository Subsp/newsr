#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate prepared SR prior folders before reusing them in training."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--mask_subdir", type=str, default="usable_masks")
    return parser.parse_args()


def _validate_image(path: Path) -> None:
    with Image.open(path) as img:
        img.load()


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"
    prior_dir = output_root / args.prior_subdir
    mask_subdir = str(args.mask_subdir or "").strip()
    require_masks = bool(mask_subdir)
    mask_dir = output_root / mask_subdir if require_masks else None

    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"missing prior dir: {prior_dir}")
    if require_masks and not mask_dir.is_dir():
        raise FileNotFoundError(f"missing mask dir: {mask_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames") or []
    if not frames:
        raise ValueError(f"manifest has no frames: {manifest_path}")

    validated = 0
    for frame in frames:
        stem = frame.get("stem")
        if not stem:
            raise ValueError(f"frame missing stem in manifest: {manifest_path}")
        prior_path = prior_dir / f"{stem}.png"
        if not prior_path.is_file():
            raise FileNotFoundError(f"missing prepared prior: {prior_path}")
        _validate_image(prior_path)
        if require_masks:
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing prepared mask: {mask_path}")
            _validate_image(mask_path)
        validated += 1

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "prior_subdir": args.prior_subdir,
                "mask_subdir": mask_subdir or None,
                "require_masks": require_masks,
                "validated_frames": validated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[validate-prepared-priors] {exc}", file=sys.stderr)
        raise SystemExit(1)
