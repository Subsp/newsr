#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_sdfgs.tools.build_scaffold_from_colmap import (  # noqa: E402
    maybe_subsample,
    read_points3d_binary,
    read_points3d_text,
    resolve_points_file,
)


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _detect_demo_script(vggt_root: Path) -> Path:
    candidates = [
        vggt_root / "demo_colmap.py",
        vggt_root / "demo" / "demo_colmap.py",
        vggt_root / "scripts" / "demo_colmap.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find demo_colmap.py in VGGT root: {vggt_root}"
    )


def _pick_option(help_text: str, options: list[str]) -> str | None:
    for opt in options:
        # Match complete option token in help output.
        if re.search(rf"(^|[\s,]){re.escape(opt)}($|[\s=,])", help_text, re.M):
            return opt
    return None


def _build_vggt_command(
    python_exe: str,
    demo_script: Path,
    dataset_path: Path,
    workspace: Path,
    device: str,
) -> list[str]:
    help_proc = _run([python_exe, str(demo_script), "--help"], cwd=str(demo_script.parent))
    help_text = help_proc.stdout or ""

    in_opt = _pick_option(
        help_text,
        ["--input_dir", "--scene_dir", "--images_dir", "--img_dir", "--images"],
    )
    out_opt = _pick_option(help_text, ["--output_dir", "--out_dir", "--output"])
    device_opt = _pick_option(help_text, ["--device"])

    if in_opt and out_opt:
        cmd = [
            python_exe,
            str(demo_script),
            in_opt,
            str(dataset_path),
            out_opt,
            str(workspace),
        ]
        if device_opt and device:
            cmd.extend([device_opt, device])
        return cmd

    # Fallback attempts for unknown CLI variants.
    # We return the most common command shape first.
    cmd = [
        python_exe,
        str(demo_script),
        "--input_dir",
        str(dataset_path),
        "--output_dir",
        str(workspace),
    ]
    if device:
        cmd.extend(["--device", device])
    return cmd


def _find_sparse_dir(workspace: Path) -> Path:
    candidates = []
    for p in workspace.rglob("points3D.bin"):
        candidates.append(p.parent)
    for p in workspace.rglob("points3D.txt"):
        candidates.append(p.parent)
    if not candidates:
        raise FileNotFoundError(
            f"No points3D.bin/txt found under workspace: {workspace}"
        )
    # pick most recently modified candidate
    candidates = sorted(
        set(candidates),
        key=lambda d: max(
            (d / "points3D.bin").stat().st_mtime if (d / "points3D.bin").is_file() else 0.0,
            (d / "points3D.txt").stat().st_mtime if (d / "points3D.txt").is_file() else 0.0,
        ),
        reverse=True,
    )
    return candidates[0]


def _write_scaffold_npz(sparse_dir: Path, output: Path, prefer: str, max_points: int, seed: int):
    points_file = resolve_points_file(str(sparse_dir), prefer)
    if points_file.endswith(".bin"):
        points = read_points3d_binary(points_file)
    else:
        points = read_points3d_text(points_file)
    if points.shape[0] == 0:
        raise ValueError(f"No points found in {points_file}")
    points = maybe_subsample(points, max_points, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = str(output)
    import numpy as np

    np.savez_compressed(npz_path, points=points.astype(np.float32))
    print(f"[vggt->scaffold] saved {points.shape[0]} points to {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run VGGT COLMAP demo and build scaffold npz for hybrid_sdfgs."
    )
    parser.add_argument("--vggt_root", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--scaffold_out", type=str, required=True)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prefer_points_format", type=str, default="auto", choices=["auto", "bin", "txt"])
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_vggt", action="store_true")
    args = parser.parse_args()

    vggt_root = Path(args.vggt_root).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    scaffold_out = Path(args.scaffold_out).expanduser().resolve()

    if not vggt_root.is_dir():
        raise FileNotFoundError(f"VGGT root not found: {vggt_root}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    workspace.mkdir(parents=True, exist_ok=True)
    demo_script = _detect_demo_script(vggt_root)

    if not args.skip_vggt:
        cmd = _build_vggt_command(
            python_exe=args.python_exe,
            demo_script=demo_script,
            dataset_path=dataset_path,
            workspace=workspace,
            device=args.device,
        )
        print("[vggt->scaffold] running VGGT:")
        print("  " + " ".join(shlex.quote(x) for x in cmd))
        proc = _run(cmd, cwd=str(demo_script.parent))
        print(proc.stdout or "")
        if proc.returncode != 0:
            raise RuntimeError(f"VGGT command failed with exit code {proc.returncode}")

    sparse_dir = _find_sparse_dir(workspace)
    print(f"[vggt->scaffold] sparse dir: {sparse_dir}")
    _write_scaffold_npz(
        sparse_dir=sparse_dir,
        output=scaffold_out,
        prefer=args.prefer_points_format,
        max_points=args.max_points,
        seed=args.seed,
    )
    print("[vggt->scaffold] done.")


if __name__ == "__main__":
    main()
