#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SwinIR or HAT on an image folder to generate SR priors, and optionally "
            "build IE-SRGS-style discrepancy / usable-mask / fused-prior outputs "
            "against a reference image folder."
        )
    )
    parser.add_argument(
        "--model_repo_root",
        "--swinir_root",
        dest="model_repo_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="swinir",
        choices=["swinir", "hat"],
    )
    parser.add_argument("--folder_lq", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="classical_sr",
        choices=["classical_sr", "lightweight_sr", "real_sr"],
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--training_patch_size", type=int, default=64)
    parser.add_argument("--large_model", action="store_true")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_threshold", type=float, default=0.12)
    parser.add_argument("--mask_mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument(
        "--discrepancy_floor",
        type=float,
        default=0.05,
        help="Lower bound on reference luma in the discrepancy denominator.",
    )
    parser.add_argument(
        "--save_fused_priors",
        action="store_true",
        help="Save priors blended with reference images using usable-mask weights.",
    )
    parser.add_argument(
        "--save_raw_priors",
        action="store_true",
        help="Always save raw SR priors under output_root/priors (enabled by default).",
    )
    parser.add_argument(
        "--save_discrepancy_npz",
        action="store_true",
        help="Save raw discrepancy/mask arrays as compressed .npz per frame.",
    )
    return parser.parse_args()


def _collect_images(folder: Path) -> list[Path]:
    images = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found under: {folder}")
    return images


def _index_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in _collect_images(folder)}


def _load_rgb01(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb01(path: Path, rgb: np.ndarray) -> None:
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb_u8, mode="RGB").save(path)


def _save_gray01(path: Path, gray: np.ndarray) -> None:
    gray_u8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(gray_u8, mode="L").save(path)


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path

    model_root = args.model_repo_root.resolve()
    if args.backend == "swinir":
        if args.task == "classical_sr":
            return model_root / "model_zoo" / "swinir" / f"001_classicalSR_DF2K_s{args.training_patch_size}w8_SwinIR-M_x{args.scale}.pth"
        if args.task == "lightweight_sr":
            return model_root / "model_zoo" / "swinir" / f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{args.scale}.pth"
        if args.task == "real_sr":
            name = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" if args.large_model else "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
            return model_root / "model_zoo" / "swinir" / name
        raise ValueError(f"Unsupported task for default model path: {args.task}")

    if args.backend == "hat":
        if args.task == "real_sr":
            return model_root / "experiments" / "pretrained_models" / "Real_HAT_GAN_SRx4.pth"
        return model_root / "experiments" / "pretrained_models" / f"HAT_SRx{args.scale}_ImageNet-pretrain.pth"

    raise ValueError(f"Unsupported backend: {args.backend}")


def _import_swinir_helpers(swinir_root: Path):
    model_root_str = str(swinir_root.resolve())
    if model_root_str not in sys.path:
        sys.path.insert(0, model_root_str)
    from main_test_swinir import define_model, test  # type: ignore

    return define_model, test


def _build_model_args(args: argparse.Namespace, model_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task=args.task,
        scale=args.scale,
        noise=15,
        jpeg=40,
        training_patch_size=args.training_patch_size,
        large_model=args.large_model,
        model_path=str(model_path),
        tile=None if args.tile <= 0 else args.tile,
        tile_overlap=args.tile_overlap,
    )


def _infer_image(
    rgb: np.ndarray,
    model,
    swinir_test,
    model_args: SimpleNamespace,
    window_size: int,
    torch_device,
):
    import torch

    chw = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float().unsqueeze(0).to(torch_device)
    with torch.no_grad():
        _, _, h_old, w_old = chw.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        if h_pad > 0:
            chw = torch.cat([chw, torch.flip(chw, [2])], dim=2)[:, :, : h_old + h_pad, :]
        if w_pad > 0:
            chw = torch.cat([chw, torch.flip(chw, [3])], dim=3)[:, :, :, : w_old + w_pad]
        output = swinir_test(chw, model, model_args, window_size)
        output = output[..., : h_old * model_args.scale, : w_old * model_args.scale]
    out = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    return np.transpose(out, (1, 2, 0))


def _run_swinir_folder_inference(
    args: argparse.Namespace,
    model_path: Path,
    priors_dir: Path,
    lq_images: list[Path],
) -> None:
    define_model, swinir_test = _import_swinir_helpers(args.model_repo_root)

    import torch

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_args = _build_model_args(args, model_path)
    model = define_model(model_args).eval().to(device)
    window_size = 8

    for idx, image_path in enumerate(lq_images, start=1):
        print(f"[sr-prior:{args.backend}] {idx}/{len(lq_images)} {image_path.name}")
        rgb = _load_rgb01(image_path)
        sr_rgb = _infer_image(
            rgb=rgb,
            model=model,
            swinir_test=swinir_test,
            model_args=model_args,
            window_size=window_size,
            torch_device=device,
        )
        _save_rgb01(priors_dir / f"{image_path.stem}.png", sr_rgb)


def _guess_hat_option_file(model_root: Path, model_path: Path) -> Path:
    options_dir = model_root / "options" / "test"
    name = model_path.name
    if name.startswith("Real_HAT_GAN"):
        candidate = options_dir / "HAT_GAN_Real_SRx4.yml"
        if candidate.is_file():
            return candidate

    candidate = options_dir / f"{model_path.stem}.yml"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Failed to infer HAT test option file for checkpoint: "
        f"{model_path}. Expected something like {candidate.name} under {options_dir}."
    )


def _match_hat_result(input_stem: str, result_paths: list[Path]) -> Path | None:
    exact = [p for p in result_paths if p.stem == input_stem]
    if exact:
        return exact[0]
    prefixed = [p for p in result_paths if p.stem.startswith(f"{input_stem}_")]
    if len(prefixed) == 1:
        return prefixed[0]
    if prefixed:
        prefixed.sort(key=lambda p: len(p.stem))
        return prefixed[0]
    return None


def _run_hat_folder_inference(
    args: argparse.Namespace,
    model_path: Path,
    priors_dir: Path,
    lq_images: list[Path],
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to build a temporary HAT test config.") from exc

    model_root = args.model_repo_root.resolve()
    option_path = _guess_hat_option_file(model_root, model_path)
    with option_path.open("r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    run_name = f"{model_path.stem}_custom_infer"
    opt["name"] = run_name
    opt["num_gpu"] = 0 if args.device == "cpu" else 1
    opt["datasets"] = {
        "test_1": {
            "name": "custom",
            "type": "SingleImageDataset",
            "dataroot_lq": str(args.folder_lq),
            "io_backend": {"type": "disk"},
        }
    }
    if args.tile > 0:
        opt["tile"] = {
            "tile_size": int(args.tile),
            "tile_pad": int(args.tile_overlap),
        }
    else:
        opt.pop("tile", None)
    opt.setdefault("path", {})
    opt["path"]["pretrain_network_g"] = str(model_path.resolve())
    opt["path"]["strict_load_g"] = True
    opt["path"]["param_key_g"] = "params_ema"
    opt.setdefault("val", {})
    opt["val"]["save_img"] = True
    opt["val"]["suffix"] = "sr"
    # SingleImageDataset has no GT, so HAT/BasicSR validation metrics must be disabled.
    opt["val"].pop("metrics", None)

    with tempfile.TemporaryDirectory(prefix="hat_custom_infer_") as tmpdir:
        tmp_yaml = Path(tmpdir) / "hat_custom_test.yml"
        with tmp_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(opt, f, sort_keys=False)

        cmd = [sys.executable, "-m", "hat.test", "-opt", str(tmp_yaml)]
        env = os.environ.copy()
        pythonpath_parts = [str(model_root)]
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        print("[sr-prior:hat] running official HAT test pipeline:")
        print("  " + " ".join(str(x) for x in cmd))
        subprocess.run(cmd, cwd=str(model_root), env=env, check=True)

        result_dir = model_root / "results" / run_name / "visualization" / "custom"
        if not result_dir.is_dir():
            raise FileNotFoundError(f"HAT result dir not found: {result_dir}")

        result_paths = [p for p in sorted(result_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not result_paths:
            raise FileNotFoundError(f"No HAT result images found under: {result_dir}")

        missing = []
        for image_path in lq_images:
            matched = _match_hat_result(image_path.stem, result_paths)
            if matched is None:
                missing.append(image_path.stem)
                continue
            shutil.copy2(matched, priors_dir / f"{image_path.stem}.png")

        shutil.rmtree(model_root / "results" / run_name, ignore_errors=True)

        if missing:
            raise FileNotFoundError(
                "Missing HAT outputs for stems: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )


def _ensure_same_size(reference_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if reference_rgb.shape[:2] == (target_h, target_w):
        return reference_rgb
    ref_u8 = np.clip(np.round(reference_rgb * 255.0), 0, 255).astype(np.uint8)
    resized = Image.fromarray(ref_u8, mode="RGB").resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _compute_discrepancy_and_mask(
    external_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    threshold: float,
    floor: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = _ensure_same_size(reference_rgb, external_rgb.shape[:2])
    ext_l = _rgb_to_luma(external_rgb)
    ref_l = _rgb_to_luma(ref)
    denom = np.maximum(np.abs(ref_l), floor)
    discrepancy = np.abs(ext_l - ref_l) / denom
    if mode == "hard":
        usable = (discrepancy < threshold).astype(np.float32)
    else:
        usable = np.clip(1.0 - discrepancy / max(threshold, 1e-8), 0.0, 1.0)
    fused = usable[..., None] * external_rgb + (1.0 - usable[..., None]) * ref
    return discrepancy.astype(np.float32), usable.astype(np.float32), fused.astype(np.float32)


def main() -> None:
    args = _parse_args()
    args.model_repo_root = args.model_repo_root.resolve()
    args.folder_lq = args.folder_lq.resolve()
    args.output_root = args.output_root.resolve()
    if args.reference_dir is not None:
        args.reference_dir = args.reference_dir.resolve()

    if not args.model_repo_root.is_dir():
        raise FileNotFoundError(f"Model repo not found: {args.model_repo_root}")
    if not args.folder_lq.is_dir():
        raise FileNotFoundError(f"Input image folder not found: {args.folder_lq}")
    if args.reference_dir is not None and not args.reference_dir.is_dir():
        raise FileNotFoundError(f"Reference image folder not found: {args.reference_dir}")

    model_path = _resolve_model_path(args)
    if not model_path.is_file():
        raise FileNotFoundError(
            "SR checkpoint not found: "
            f"{model_path}. Pass --model_path explicitly or download the checkpoint first."
        )

    priors_dir = args.output_root / "priors"
    discrepancy_dir = args.output_root / "discrepancy"
    mask_dir = args.output_root / "usable_masks"
    aligned_reference_dir = args.output_root / "aligned_references"
    masked_prior_dir = args.output_root / "masked_priors"
    masked_reference_dir = args.output_root / "masked_references"
    fused_dir = args.output_root / "fused_priors"
    npz_dir = args.output_root / "discrepancy_npz"
    priors_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_dir is not None:
        discrepancy_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        aligned_reference_dir.mkdir(parents=True, exist_ok=True)
        masked_prior_dir.mkdir(parents=True, exist_ok=True)
        masked_reference_dir.mkdir(parents=True, exist_ok=True)
        if args.save_fused_priors:
            fused_dir.mkdir(parents=True, exist_ok=True)
        if args.save_discrepancy_npz:
            npz_dir.mkdir(parents=True, exist_ok=True)

    lq_images = _collect_images(args.folder_lq)
    ref_by_stem = _index_by_stem(args.reference_dir) if args.reference_dir is not None else {}

    if args.backend == "swinir":
        _run_swinir_folder_inference(args=args, model_path=model_path, priors_dir=priors_dir, lq_images=lq_images)
    elif args.backend == "hat":
        _run_hat_folder_inference(args=args, model_path=model_path, priors_dir=priors_dir, lq_images=lq_images)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    stats: list[dict[str, float | str]] = []
    missing_ref: list[str] = []

    for idx, image_path in enumerate(lq_images, start=1):
        print(f"[sr-prior:{args.backend}] post-mask {idx}/{len(lq_images)} {image_path.name}")
        prior_path = priors_dir / f"{image_path.stem}.png"
        if not prior_path.is_file():
            raise FileNotFoundError(f"Generated prior not found: {prior_path}")
        sr_rgb = _load_rgb01(prior_path)

        frame_stat: dict[str, float | str] = {
            "stem": image_path.stem,
            "height": float(sr_rgb.shape[0]),
            "width": float(sr_rgb.shape[1]),
        }

        if args.reference_dir is not None:
            ref_path = ref_by_stem.get(image_path.stem)
            if ref_path is None:
                missing_ref.append(image_path.stem)
                continue
            ref_rgb = _load_rgb01(ref_path)
            discrepancy, usable, fused = _compute_discrepancy_and_mask(
                external_rgb=sr_rgb,
                reference_rgb=ref_rgb,
                threshold=args.mask_threshold,
                floor=args.discrepancy_floor,
                mode=args.mask_mode,
            )
            ref_resized = _ensure_same_size(ref_rgb, sr_rgb.shape[:2])
            masked_prior = usable[..., None] * sr_rgb
            masked_reference = (1.0 - usable[..., None]) * ref_resized
            disc_vis = np.clip(discrepancy / max(args.mask_threshold * 2.0, 1e-8), 0.0, 1.0)
            _save_gray01(discrepancy_dir / f"{image_path.stem}.png", disc_vis)
            _save_gray01(mask_dir / f"{image_path.stem}.png", usable)
            _save_rgb01(aligned_reference_dir / f"{image_path.stem}.png", ref_resized)
            _save_rgb01(masked_prior_dir / f"{image_path.stem}.png", masked_prior)
            _save_rgb01(masked_reference_dir / f"{image_path.stem}.png", masked_reference)
            if args.save_fused_priors:
                _save_rgb01(fused_dir / f"{image_path.stem}.png", fused)
            if args.save_discrepancy_npz:
                np.savez_compressed(
                    npz_dir / f"{image_path.stem}.npz",
                    discrepancy=discrepancy,
                    usable_mask=usable,
                )
            frame_stat["discrepancy_mean"] = float(discrepancy.mean())
            frame_stat["discrepancy_p90"] = float(np.percentile(discrepancy, 90.0))
            frame_stat["usable_ratio"] = float(usable.mean())
        stats.append(frame_stat)

    if missing_ref:
        raise FileNotFoundError(
            "Missing reference images for stems: "
            + ", ".join(missing_ref[:10])
            + (" ..." if len(missing_ref) > 10 else "")
        )

    manifest = {
        "tool": "generate_sr_priors",
        "backend": args.backend,
        "model_repo_root": str(args.model_repo_root),
        "folder_lq": str(args.folder_lq),
        "reference_dir": str(args.reference_dir) if args.reference_dir is not None else None,
        "output_root": str(args.output_root),
        "task": args.task,
        "scale": args.scale,
        "training_patch_size": args.training_patch_size,
        "large_model": bool(args.large_model),
        "model_path": str(model_path),
        "mask_mode": args.mask_mode,
        "mask_threshold": args.mask_threshold,
        "discrepancy_floor": args.discrepancy_floor,
        "save_fused_priors": bool(args.save_fused_priors),
        "num_frames": len(stats),
        "stats": stats,
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.reference_dir is not None and stats:
        usable_vals = [float(x["usable_ratio"]) for x in stats if "usable_ratio" in x]
        disc_vals = [float(x["discrepancy_mean"]) for x in stats if "discrepancy_mean" in x]
        if usable_vals:
            print(
                f"[sr-prior:{args.backend}] usable mask "
                f"mean={sum(usable_vals)/len(usable_vals):.4f} "
                f"min={min(usable_vals):.4f} max={max(usable_vals):.4f}"
            )
        if disc_vals:
            print(
                f"[sr-prior:{args.backend}] discrepancy "
                f"mean={sum(disc_vals)/len(disc_vals):.4f} "
                f"min={min(disc_vals):.4f} max={max(disc_vals):.4f}"
            )
    print(f"[sr-prior:{args.backend}] priors saved to {priors_dir}")
    if args.reference_dir is not None:
        print(f"[sr-prior:{args.backend}] masks saved to {mask_dir}")
        print(f"[sr-prior:{args.backend}] aligned references saved to {aligned_reference_dir}")
        print(f"[sr-prior:{args.backend}] masked priors saved to {masked_prior_dir}")
        print(f"[sr-prior:{args.backend}] masked references saved to {masked_reference_dir}")
        if args.save_fused_priors:
            print(f"[sr-prior:{args.backend}] fused priors saved to {fused_dir}")


if __name__ == "__main__":
    main()
