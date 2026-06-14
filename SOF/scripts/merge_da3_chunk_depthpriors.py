from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge DA3 chunked mini_npz outputs into a SOF-friendly single prior root "
            "with per-view depth/confidence arrays."
        )
    )
    parser.add_argument(
        "--chunks_root",
        required=True,
        help="Root directory containing chunk_* outputs, each with exports/mini_npz/results.npz",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Unified output root. The script writes depth/, confidence/, and metadata files here.",
    )
    parser.add_argument(
        "--scene_root",
        default="",
        help="Optional scene root used only for metadata and optional coverage checks.",
    )
    parser.add_argument(
        "--images_subdir",
        default="images_2",
        help="Optional reference image subdir under scene_root for metadata/checks.",
    )
    parser.add_argument(
        "--confidence_subdir",
        default="confidence",
        help="Name of the confidence directory to create under output_root.",
    )
    parser.add_argument(
        "--allow_missing_conf",
        action="store_true",
        help="Do not fail if some chunk results do not contain confidence.",
    )
    return parser.parse_args()


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _list_reference_stems(scene_root: Path, images_subdir: str) -> List[str]:
    images_dir = scene_root / images_subdir
    if not images_dir.is_dir():
        return []
    return sorted(path.stem for path in images_dir.iterdir() if _is_image(path))


def _read_manifest(chunk_dir: Path) -> List[str]:
    manifest_path = chunk_dir / "manifest.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.txt under {chunk_dir}")
    lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    names = [line for line in lines if line]
    if not names:
        raise RuntimeError(f"Empty manifest.txt under {chunk_dir}")
    return names


def _load_results(chunk_dir: Path) -> Dict[str, np.ndarray]:
    result_path = chunk_dir / "exports" / "mini_npz" / "results.npz"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing results.npz under {chunk_dir}")
    payload = np.load(str(result_path))
    if "depth" not in payload:
        raise KeyError(f"Chunk results missing 'depth': {result_path}")
    data: Dict[str, np.ndarray] = {"depth": np.asarray(payload["depth"])}
    if "conf" in payload:
        data["conf"] = np.asarray(payload["conf"])
    if "intrinsics" in payload:
        data["intrinsics"] = np.asarray(payload["intrinsics"])
    if "extrinsics" in payload:
        data["extrinsics"] = np.asarray(payload["extrinsics"])
    return data


def _canonical_stem(name: str) -> str:
    return Path(name).stem


def main() -> None:
    args = parse_args()
    chunks_root = Path(args.chunks_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve() if str(args.scene_root).strip() else None

    if not chunks_root.is_dir():
        raise FileNotFoundError(f"Chunks root not found: {chunks_root}")

    depth_dir = output_root / "depth"
    conf_dir = output_root / str(args.confidence_subdir)
    intrinsics_dir = output_root / "intrinsics"
    extrinsics_dir = output_root / "extrinsics"
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    seen_stems: Dict[str, Path] = {}
    total_views = 0
    total_conf_views = 0

    chunk_dirs = sorted(path for path in chunks_root.glob("chunk_*") if path.is_dir())
    if not chunk_dirs:
        raise RuntimeError(f"No chunk_* directories found under {chunks_root}")

    for chunk_dir in chunk_dirs:
        manifest_names = _read_manifest(chunk_dir)
        results = _load_results(chunk_dir)
        depth = results["depth"]
        conf = results.get("conf")
        intrinsics = results.get("intrinsics")
        extrinsics = results.get("extrinsics")

        if depth.ndim != 3:
            raise ValueError(f"Expected depth shape [N,H,W], got {depth.shape} in {chunk_dir}")
        count = int(depth.shape[0])
        if count != len(manifest_names):
            raise ValueError(
                f"Chunk {chunk_dir.name} manifest count {len(manifest_names)} "
                f"!= depth count {count}"
            )
        if conf is None and not args.allow_missing_conf:
            raise ValueError(
                f"Chunk {chunk_dir.name} has no confidence in results.npz. "
                f"Re-run with DA3 confidence export or pass --allow_missing_conf."
            )
        if conf is not None and int(conf.shape[0]) != count:
            raise ValueError(f"Chunk {chunk_dir.name} confidence count mismatch: {conf.shape}")
        if intrinsics is not None and int(intrinsics.shape[0]) != count:
            raise ValueError(f"Chunk {chunk_dir.name} intrinsics count mismatch: {intrinsics.shape}")
        if extrinsics is not None and int(extrinsics.shape[0]) != count:
            raise ValueError(f"Chunk {chunk_dir.name} extrinsics count mismatch: {extrinsics.shape}")

        for idx, raw_name in enumerate(manifest_names):
            stem = _canonical_stem(raw_name)
            if stem in seen_stems:
                raise ValueError(
                    f"Duplicate frame stem '{stem}' in {chunk_dir}; already seen in {seen_stems[stem]}"
                )
            seen_stems[stem] = chunk_dir

            depth_path = depth_dir / f"{stem}.npy"
            np.save(str(depth_path), np.asarray(depth[idx], dtype=np.float32))

            conf_path = None
            if conf is not None:
                conf_path = conf_dir / f"{stem}.npy"
                np.save(str(conf_path), np.asarray(conf[idx], dtype=np.float32))
                total_conf_views += 1

            intr_path = None
            if intrinsics is not None:
                intr_path = intrinsics_dir / f"{stem}.npy"
                np.save(str(intr_path), np.asarray(intrinsics[idx], dtype=np.float32))

            extr_path = None
            if extrinsics is not None:
                extr_path = extrinsics_dir / f"{stem}.npy"
                np.save(str(extr_path), np.asarray(extrinsics[idx], dtype=np.float32))

            summary_rows.append(
                {
                    "image_name": raw_name,
                    "stem": stem,
                    "chunk": chunk_dir.name,
                    "depth_path": str(depth_path),
                    "confidence_path": None if conf_path is None else str(conf_path),
                    "intrinsics_path": None if intr_path is None else str(intr_path),
                    "extrinsics_path": None if extr_path is None else str(extr_path),
                    "shape": [int(depth[idx].shape[0]), int(depth[idx].shape[1])],
                }
            )
            total_views += 1

    reference_stems: List[str] = []
    missing_reference_stems: List[str] = []
    extra_stems: List[str] = []
    if scene_root is not None:
        reference_stems = _list_reference_stems(scene_root, str(args.images_subdir))
        reference_set = set(reference_stems)
        produced_set = set(seen_stems.keys())
        missing_reference_stems = sorted(reference_set - produced_set)
        extra_stems = sorted(produced_set - reference_set)

    meta = {
        "source": "depth-anything-3 chunked mini_npz merge",
        "chunks_root": str(chunks_root),
        "output_root": str(output_root),
        "scene_root": None if scene_root is None else str(scene_root),
        "images_subdir": str(args.images_subdir),
        "depth_subdir": "depth",
        "confidence_subdir": str(args.confidence_subdir) if total_conf_views > 0 else None,
        "intrinsics_subdir": "intrinsics",
        "extrinsics_subdir": "extrinsics",
        "num_chunks": len(chunk_dirs),
        "num_views": total_views,
        "num_conf_views": total_conf_views,
        "reference_frame_count": len(reference_stems),
        "missing_reference_stems": missing_reference_stems,
        "extra_stems": extra_stems,
        "loader_hint": {
            "depth_prior_root": str(output_root),
            "depth_prior_subdirs": "depth,",
            "depth_prior_confidence_subdirs": f"{args.confidence_subdir},",
        },
    }

    (output_root / "frames.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[merge-da3-depthpriors] chunks={len(chunk_dirs)} views={total_views} "
        f"conf_views={total_conf_views} output={output_root}"
    )
    if scene_root is not None:
        print(
            f"[merge-da3-depthpriors] reference={len(reference_stems)} "
            f"missing={len(missing_reference_stems)} extra={len(extra_stems)}"
        )


if __name__ == "__main__":
    main()
