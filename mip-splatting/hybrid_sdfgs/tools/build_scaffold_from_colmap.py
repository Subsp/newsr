#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import struct

import numpy as np


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end-of-file while parsing COLMAP model.")
    return struct.unpack("<" + fmt, data)


def read_points3d_binary(path: str) -> np.ndarray:
    points = []
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            x, y, z = props[1:4]
            points.append([x, y, z])

            track_len = _read_next_bytes(fid, 8, "Q")[0]
            if track_len > 0:
                _ = _read_next_bytes(fid, int(track_len) * 8, "ii" * int(track_len))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def read_points3d_text(path: str) -> np.ndarray:
    points = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
            points.append([x, y, z])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def resolve_points_file(sparse_dir: str, prefer: str) -> str:
    prefer = prefer.lower()
    cand_bin = os.path.join(sparse_dir, "points3D.bin")
    cand_txt = os.path.join(sparse_dir, "points3D.txt")
    if prefer == "bin":
        if not os.path.isfile(cand_bin):
            raise FileNotFoundError(f"Missing file: {cand_bin}")
        return cand_bin
    if prefer == "txt":
        if not os.path.isfile(cand_txt):
            raise FileNotFoundError(f"Missing file: {cand_txt}")
        return cand_txt
    if os.path.isfile(cand_bin):
        return cand_bin
    if os.path.isfile(cand_txt):
        return cand_txt
    raise FileNotFoundError(
        f"No points3D.bin / points3D.txt found under: {sparse_dir}"
    )


def maybe_subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP sparse points to scaffold npz for hybrid_sdfgs."
    )
    parser.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        help="COLMAP sparse dir containing points3D.bin/txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output npz path (contains key: points)",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="auto",
        choices=["auto", "bin", "txt"],
        help="Preferred COLMAP file format.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional random subsample size (0 means keep all).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    sparse_dir = os.path.abspath(os.path.expanduser(args.sparse_dir))
    output = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)

    points_file = resolve_points_file(sparse_dir, args.prefer)
    if points_file.endswith(".bin"):
        points = read_points3d_binary(points_file)
    else:
        points = read_points3d_text(points_file)

    if points.shape[0] == 0:
        raise ValueError(f"No points found in {points_file}")

    points = maybe_subsample(points, args.max_points, args.seed)
    np.savez_compressed(output, points=points.astype(np.float32))
    print(
        f"[scaffold] saved {points.shape[0]} points to {output} "
        f"(source={points_file})"
    )


if __name__ == "__main__":
    main()
