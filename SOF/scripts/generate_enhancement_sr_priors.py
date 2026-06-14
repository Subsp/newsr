#!/usr/bin/env python3
"""
Generate enhancement-style priors from scene images.

This is the "deterministic / enhancement SR" branch counterpart to the
existing diffusion-prior workflow. The output is a flat image cache
(`<stem>.png`) that can later be aligned/prepared with
`hybrid_sdfgs.tools.prepare_existing_sr_priors`.

Current backends:
  - swinir: classical x4 SwinIR
  - nafnet: same-size x1 NAFNet restoration via an external NAFNet repo
  - restormer: same-size x1 Restormer restoration via an external Restormer repo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".PNG", ".JPG", ".JPEG", ".BMP", ".WEBP"}
_RESTORMER_TASKS = (
    "Motion_Deblurring",
    "Single_Image_Defocus_Deblurring",
    "Deraining",
    "Real_Denoising",
    "Gaussian_Gray_Denoising",
    "Gaussian_Color_Denoising",
)


class BasePriorResolver:
    scale = 1

    def process_path(self, src: Path, dst: Path) -> None:
        raise NotImplementedError

    def process_many(
        self,
        image_paths: list[Path],
        output_dir: Path,
        overwrite: bool,
        backend: str,
    ) -> tuple[int, int, str | None]:
        written = 0
        skipped = 0
        first_output: str | None = None
        for path in tqdm(image_paths, desc=f"{backend} priors", unit="img"):
            dst = output_dir / f"{path.stem}.png"
            if dst.exists() and not overwrite:
                skipped += 1
                if first_output is None:
                    first_output = str(dst)
                continue

            self.process_path(path, dst)
            written += 1
            if first_output is None:
                first_output = str(dst)
        return written, skipped, first_output


class SwinIRPriorResolver(BasePriorResolver):
    scale = 4

    def __init__(self, device: str) -> None:
        from experiments.utils.swinir_wrapper import SwinIRSuperResolver

        self._resolver = SwinIRSuperResolver(device=device)

    def process_path(self, src: Path, dst: Path) -> None:
        with Image.open(src) as pil:
            sr = self._resolver.upscale_pil(pil.convert("RGB"))
            sr.save(dst)


class NAFNetPriorResolver(BasePriorResolver):
    scale = 1

    def __init__(self, repo_root: Path, python_bin: str, config_path: Path) -> None:
        self.repo_root = repo_root
        self.python_bin = python_bin
        self.config_path = config_path
        self.demo_py = repo_root / "basicsr" / "demo.py"
        if not self.demo_py.is_file():
            raise FileNotFoundError(f"NAFNet demo.py not found: {self.demo_py}")

    def process_path(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_bin,
            str(self.demo_py),
            "-opt",
            str(self.config_path),
            "--input_path",
            str(src),
            "--output_path",
            str(dst),
        ]
        _run_command(cmd, self.repo_root)


class RestormerPriorResolver(BasePriorResolver):
    scale = 1

    def __init__(
        self,
        repo_root: Path,
        python_bin: str,
        task: str,
        tile: int,
        tile_overlap: int,
    ) -> None:
        self.repo_root = repo_root
        self.python_bin = python_bin
        self.task = task
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.demo_py = repo_root / "demo.py"
        if not self.demo_py.is_file():
            raise FileNotFoundError(f"Restormer demo.py not found: {self.demo_py}")

    def process_path(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="restormer_prior_") as tmp:
            result_dir = Path(tmp) / "restored"
            cmd = [
                self.python_bin,
                str(self.demo_py),
                "--task",
                self.task,
                "--input_dir",
                str(src),
                "--result_dir",
                str(result_dir),
            ]
            if self.tile > 0:
                cmd.extend(["--tile", str(self.tile), "--tile_overlap", str(self.tile_overlap)])
            _run_command(cmd, self.repo_root)
            produced = _find_restormer_output(result_dir, self.task, src.stem)
            shutil.copy2(produced, dst)

    def process_many(
        self,
        image_paths: list[Path],
        output_dir: Path,
        overwrite: bool,
        backend: str,
    ) -> tuple[int, int, str | None]:
        selected: list[Path] = []
        skipped = 0
        first_output: str | None = None
        for path in image_paths:
            dst = output_dir / f"{path.stem}.png"
            if dst.exists() and not overwrite:
                skipped += 1
                if first_output is None:
                    first_output = str(dst)
                continue
            selected.append(path)
            if first_output is None:
                first_output = str(dst)

        if not selected:
            return 0, skipped, first_output

        with tempfile.TemporaryDirectory(prefix="restormer_prior_batch_") as tmp:
            tmp_root = Path(tmp)
            batch_input = tmp_root / "input"
            result_dir = tmp_root / "restored"
            batch_input.mkdir(parents=True, exist_ok=True)
            for path in selected:
                _link_or_copy(path, batch_input / path.name)

            cmd = [
                self.python_bin,
                str(self.demo_py),
                "--task",
                self.task,
                "--input_dir",
                str(batch_input),
                "--result_dir",
                str(result_dir),
            ]
            if self.tile > 0:
                cmd.extend(["--tile", str(self.tile), "--tile_overlap", str(self.tile_overlap)])
            _run_command(cmd, self.repo_root)

            for path in tqdm(selected, desc=f"{backend} collect", unit="img"):
                produced = _find_restormer_output(result_dir, self.task, path.stem)
                shutil.copy2(produced, output_dir / f"{path.stem}.png")
        return len(selected), skipped, first_output


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic enhancement/restoration priors from scene frames."
    )
    parser.add_argument("--input_dir", required=True, help="Directory of source images, e.g. images_8 or x1 renders")
    parser.add_argument("--output_dir", required=True, help="Directory to write raw priors (<stem>.png)")
    parser.add_argument(
        "--backend",
        choices=("swinir", "nafnet", "restormer"),
        default="swinir",
        help="Enhancement/restoration backend.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Execution device passed to the backend.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG outputs.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for smoke/debug runs. 0 means all frames.",
    )
    parser.add_argument(
        "--external_repo_root",
        default="",
        help=(
            "External backend repo root for nafnet/restormer. "
            "Falls back to NAFNET_ROOT or RESTORMER_ROOT."
        ),
    )
    parser.add_argument(
        "--external_python",
        default=sys.executable,
        help="Python executable used when launching external backend repos.",
    )
    parser.add_argument(
        "--external_config",
        default="",
        help=(
            "External backend config. For nafnet, this is the BasicSR option yml. "
            "Falls back to NAFNET_OPT or options/test/REDS/NAFNet-width64.yml."
        ),
    )
    parser.add_argument(
        "--restormer_task",
        choices=_RESTORMER_TASKS,
        default=os.environ.get("RESTORMER_TASK", "Single_Image_Defocus_Deblurring"),
        help="Restormer task used for same-size x1 render restoration.",
    )
    parser.add_argument(
        "--restormer_tile",
        type=int,
        default=_env_int("RESTORMER_TILE", 0),
        help="Optional Restormer tile size. 0 disables tiled inference.",
    )
    parser.add_argument(
        "--restormer_tile_overlap",
        type=int,
        default=_env_int("RESTORMER_TILE_OVERLAP", 32),
        help="Restormer tile overlap when --restormer_tile is enabled.",
    )
    return parser.parse_args()


def _collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix in _IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No source images found under: {root}")
    return paths


def _ensure_unique_stems(paths: Iterable[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if stem in seen:
            raise ValueError(
                f"Duplicate frame stem '{stem}' for {seen[stem]} and {path}. "
                "Raw prior cache expects one output PNG per frame stem."
            )
        seen[stem] = path


def _run_command(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"[enhancement-priors] run external: {printable}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _resolve_backend_root(args: argparse.Namespace, backend: str) -> Path:
    env_name = f"{backend.upper()}_ROOT"
    root = args.external_repo_root or os.environ.get(env_name, "")
    if not root:
        raise ValueError(
            f"{backend} backend requires --external_repo_root or ${env_name}. "
            "Keep the external restoration repo outside this project and pass its path."
        )
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{backend} repo root not found: {resolved}")
    return resolved


def _resolve_relative_to_root(root: Path, value: str, env_name: str, default_rel: str) -> Path:
    path_text = value or os.environ.get(env_name, "") or default_rel
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Backend config not found: {resolved}")
    return resolved


def _find_restormer_output(result_dir: Path, task: str, stem: str) -> Path:
    task_dir = result_dir / task
    candidates = [
        task_dir / f"{stem}.png",
        task_dir / f"{stem}.jpg",
        task_dir / f"{stem}.jpeg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(task_dir.glob(f"{stem}.*")) if task_dir.is_dir() else []
    matches.extend(sorted(result_dir.rglob(f"{stem}.png")))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Restormer finished but no restored output was found for stem '{stem}' under {result_dir}"
    )


def _build_resolver(args: argparse.Namespace) -> BasePriorResolver:
    backend = args.backend
    if backend == "swinir":
        return SwinIRPriorResolver(device=args.device)
    if backend == "nafnet":
        root = _resolve_backend_root(args, backend)
        config = _resolve_relative_to_root(
            root,
            args.external_config,
            "NAFNET_OPT",
            "options/test/REDS/NAFNet-width64.yml",
        )
        return NAFNetPriorResolver(
            repo_root=root,
            python_bin=args.external_python,
            config_path=config,
        )
    if backend == "restormer":
        root = _resolve_backend_root(args, backend)
        return RestormerPriorResolver(
            repo_root=root,
            python_bin=args.external_python,
            task=args.restormer_task,
            tile=args.restormer_tile,
            tile_overlap=args.restormer_tile_overlap,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(input_dir)
    _ensure_unique_stems(image_paths)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    resolver = _build_resolver(args)

    written, skipped, first_output = resolver.process_many(
        image_paths=image_paths,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
        backend=args.backend,
    )

    manifest = {
        "mode": "enhancement_sr_prior_generation",
        "backend": args.backend,
        "device": args.device,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "scale": resolver.scale,
        "external_repo_root": str(getattr(resolver, "repo_root", "")),
        "external_config": str(getattr(resolver, "config_path", "")),
        "restormer_task": getattr(resolver, "task", ""),
        "restormer_tile": getattr(resolver, "tile", 0),
        "num_inputs": len(image_paths),
        "num_written": written,
        "num_skipped_existing": skipped,
        "overwrite": bool(args.overwrite),
        "first_output": first_output,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[enhancement-sr-priors] backend      : {args.backend}", flush=True)
    print(f"[enhancement-sr-priors] input_dir    : {input_dir}", flush=True)
    print(f"[enhancement-sr-priors] output_dir   : {output_dir}", flush=True)
    print(f"[enhancement-sr-priors] num_inputs   : {len(image_paths)}", flush=True)
    print(f"[enhancement-sr-priors] num_written  : {written}", flush=True)
    print(f"[enhancement-sr-priors] num_skipped  : {skipped}", flush=True)
    print(f"[enhancement-sr-priors] manifest     : {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
