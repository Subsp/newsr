import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.prior_injection import index_image_dir, normalize_image_name


def odd_kernel(kernel: int) -> int:
    kernel = int(kernel)
    if kernel <= 1:
        return 1
    return kernel if kernel % 2 == 1 else kernel + 1


def load_rgb_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def resize_like(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    pil = pil.resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def blur_rgb(image: np.ndarray, kernel: int) -> np.ndarray:
    kernel = odd_kernel(kernel)
    if kernel <= 1:
        return image.copy()
    pad = kernel // 2
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(dtype=torch.float32)
    tensor = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    blurred = F.avg_pool2d(tensor, kernel_size=kernel, stride=1)
    return blurred[0].permute(1, 2, 0).cpu().numpy()


def morph_mask(mask: np.ndarray, kernel: int, mode: str) -> np.ndarray:
    kernel = odd_kernel(kernel)
    if kernel <= 1:
        return mask
    pad = kernel // 2
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    if mode == "dilate":
        out = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=pad)
    elif mode == "erode":
        out = -F.max_pool2d(-tensor, kernel_size=kernel, stride=1, padding=pad)
    else:
        raise ValueError(f"Unknown morphology mode: {mode}")
    return out[0, 0].cpu().numpy() > 0.5


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_gray(path: Path, value: np.ndarray, vmax: float):
    denom = max(float(vmax), 1e-8)
    image = np.clip(value / denom, 0.0, 1.0)
    Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="L").save(path)


def lookup(index: dict, name: str):
    candidates = [
        name,
        normalize_image_name(name),
        Path(str(name)).name,
        Path(str(name)).stem,
    ]
    for key in candidates:
        path = index.get(str(key))
        if path is not None:
            return Path(path)
    return None


def summarize(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def main():
    parser = ArgumentParser(description="Build direct prior injection masks without using mesh visibility.")
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--anchor_dir", type=str, required=True, help="Usually LR/train images used as low-frequency anchor.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default="_inject.png")
    parser.add_argument("--blur_kernel", type=int, default=9)
    parser.add_argument("--lowfreq_threshold", type=float, default=0.08)
    parser.add_argument("--highfreq_gain_threshold", type=float, default=0.015)
    parser.add_argument("--prior_highfreq_threshold", type=float, default=0.02)
    parser.add_argument("--confidence_threshold", type=float, default=0.15)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--erode_kernel", type=int, default=1)
    parser.add_argument("--dilate_kernel", type=int, default=3)
    parser.add_argument("--view_limit", type=int, default=0)
    parser.add_argument("--debug_limit", type=int, default=32)
    args = parser.parse_args()

    prior_index = index_image_dir(args.prior_dir)
    anchor_index = index_image_dir(args.anchor_dir)
    names = sorted(set(prior_index.keys()) & set(anchor_index.keys()))
    if int(args.view_limit) > 0:
        names = names[: int(args.view_limit)]
    if not names:
        raise RuntimeError(
            "No matching prior/anchor images found. "
            f"prior_dir={args.prior_dir}, anchor_dir={args.anchor_dir}"
        )

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "direct_prior_masks"
    debug_dir = output_dir / "debug_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    view_summaries = []
    pixel_ratios = []
    for idx, name in enumerate(tqdm(names, desc="Building direct prior masks")):
        prior_path = lookup(prior_index, name)
        anchor_path = lookup(anchor_index, name)
        if prior_path is None or anchor_path is None:
            continue

        anchor = load_rgb_np(anchor_path)
        prior = resize_like(load_rgb_np(prior_path), anchor.shape[0], anchor.shape[1])

        anchor_low = blur_rgb(anchor, int(args.blur_kernel))
        prior_low = blur_rgb(prior, int(args.blur_kernel))
        anchor_high = np.mean(np.abs(anchor - anchor_low), axis=2)
        prior_high = np.mean(np.abs(prior - prior_low), axis=2)
        lowfreq_diff = np.mean(np.abs(prior_low - anchor_low), axis=2)
        highfreq_gain = prior_high - anchor_high

        lowfreq_ok = lowfreq_diff <= float(args.lowfreq_threshold)
        highfreq_need = highfreq_gain >= float(args.highfreq_gain_threshold)
        prior_has_detail = prior_high >= float(args.prior_highfreq_threshold)
        confidence = np.clip(1.0 - lowfreq_diff / max(float(args.lowfreq_threshold), 1e-8), 0.0, 1.0)
        confidence *= np.clip(highfreq_gain / max(float(args.highfreq_gain_threshold), 1e-8), 0.0, 1.0)
        if float(args.confidence_power) != 1.0:
            confidence = np.power(confidence, float(args.confidence_power))

        mask = lowfreq_ok & highfreq_need & prior_has_detail & (confidence >= float(args.confidence_threshold))
        if int(args.erode_kernel) > 1:
            mask = morph_mask(mask, int(args.erode_kernel), "erode")
        if int(args.dilate_kernel) > 1:
            mask = morph_mask(mask, int(args.dilate_kernel), "dilate")
        stem = normalize_image_name(name)
        save_mask(mask_dir / f"{stem}{args.mask_suffix}", mask)
        if idx < int(args.debug_limit):
            save_mask(debug_dir / f"{stem}_inject.png", mask)
            save_mask(debug_dir / f"{stem}_lowfreq_ok.png", lowfreq_ok)
            save_mask(debug_dir / f"{stem}_highfreq_need.png", highfreq_need)
            save_gray(debug_dir / f"{stem}_lowfreq_diff.png", lowfreq_diff, vmax=float(args.lowfreq_threshold) * 2.0)
            save_gray(debug_dir / f"{stem}_highfreq_gain.png", np.maximum(highfreq_gain, 0.0), vmax=float(args.highfreq_gain_threshold) * 4.0)
            save_gray(debug_dir / f"{stem}_confidence.png", confidence, vmax=1.0)

        active = int(mask.sum())
        total = int(mask.size)
        ratio = float(active / max(total, 1))
        pixel_ratios.append(ratio)
        view_summaries.append(
            {
                "image_name": stem,
                "prior_path": str(prior_path),
                "anchor_path": str(anchor_path),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "inject_pixels": active,
                "inject_ratio": ratio,
                "lowfreq_diff": summarize(lowfreq_diff),
                "highfreq_gain_positive": summarize(np.maximum(highfreq_gain, 0.0)),
                "confidence": summarize(confidence),
            }
        )

    summary = {
        "mode": "direct_prior_mask_v0",
        "prior_dir": str(Path(args.prior_dir).resolve()),
        "anchor_dir": str(Path(args.anchor_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()),
        "debug_dir": str(debug_dir.resolve()),
        "view_count": int(len(view_summaries)),
        "parameters": {
            "mask_suffix": args.mask_suffix,
            "blur_kernel": int(args.blur_kernel),
            "lowfreq_threshold": float(args.lowfreq_threshold),
            "highfreq_gain_threshold": float(args.highfreq_gain_threshold),
            "prior_highfreq_threshold": float(args.prior_highfreq_threshold),
            "confidence_threshold": float(args.confidence_threshold),
            "confidence_power": float(args.confidence_power),
            "erode_kernel": int(args.erode_kernel),
            "dilate_kernel": int(args.dilate_kernel),
            "view_limit": int(args.view_limit),
        },
        "inject_ratio": summarize(np.asarray(pixel_ratios, dtype=np.float32)),
        "view_summaries": view_summaries,
    }
    summary_path = output_dir / "direct_prior_masks_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved masks to: {mask_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
