#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def default_metrics_script() -> Path:
    return Path(__file__).resolve().parent / "metrics_dirs.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing GS-SDF image metrics script on standalone x8to2 outputs."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics_script", type=str, default=str(default_metrics_script()))
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    renders_dir = out_root / "renders"
    gt_dir = out_root / "gt"
    if not renders_dir.is_dir():
        raise FileNotFoundError(f"renders directory not found: {renders_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt directory not found: {gt_dir}")

    cmd = [
        args.python_exe,
        args.metrics_script,
        "--gt_color_dir",
        str(gt_dir),
        "--renders_color_dir",
        str(renders_dir),
    ]
    print("[eval-existing]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
