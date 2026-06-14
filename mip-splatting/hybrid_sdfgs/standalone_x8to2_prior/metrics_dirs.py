#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def evaluate_render(renders_dir: Path, gt_dir: Path, fname: str):
    from PIL import Image
    import torch
    import torchvision.transforms.functional as tf
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch import lpips

    render = Image.open(renders_dir / fname)
    gt = Image.open(gt_dir / fname)
    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
    return (
        fname,
        ssim(render, gt),
        psnr(render, gt),
        lpips(render, gt, net_type="vgg"),
    )


def evaluate(args):
    import torch

    renders_color_dir = Path(args.renders_color_dir)
    gt_color_dir = Path(args.gt_color_dir)
    scene_dir_path = renders_color_dir

    if not renders_color_dir.is_dir():
        raise FileNotFoundError(f"renders directory not found: {renders_color_dir}")
    if not gt_color_dir.is_dir():
        raise FileNotFoundError(f"gt directory not found: {gt_color_dir}")

    image_names = []
    ssims = []
    psnrs = []
    lpipss = []

    for fname in sorted(os.listdir(renders_color_dir)):
        if not (renders_color_dir / fname).is_file():
            continue
        if not (gt_color_dir / fname).is_file():
            continue
        fname, ssim_val, psnr_val, lpips_val = evaluate_render(
            renders_color_dir, gt_color_dir, fname
        )
        image_names.append(fname)
        ssims.append(ssim_val)
        psnrs.append(psnr_val)
        lpipss.append(lpips_val)

    if not image_names:
        raise RuntimeError("No matching render/gt files found for evaluation.")

    full_dict = {
        str(renders_color_dir): {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }
    }
    per_view_dict = {
        "SSIM": {
            name: ssim_val
            for ssim_val, name in zip(torch.tensor(ssims).tolist(), image_names)
        },
        "PSNR": {
            name: psnr_val
            for psnr_val, name in zip(torch.tensor(psnrs).tolist(), image_names)
        },
        "LPIPS": {
            name: lp_val
            for lp_val, name in zip(torch.tensor(lpipss).tolist(), image_names)
        },
    }

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

    max_psnr = torch.tensor(psnrs).max()
    min_psnr = torch.tensor(psnrs).min()
    max_psnr_index = torch.tensor(psnrs).argmax()
    min_psnr_index = torch.tensor(psnrs).argmin()
    print(
        "  Max PSNR: {:>12.7f} for {}".format(max_psnr, image_names[max_psnr_index])
    )
    print(
        "  Min PSNR: {:>12.7f} for {}".format(min_psnr, image_names[min_psnr_index])
    )
    print("")

    full_dict[str(renders_color_dir)].update(
        {
            f"Max PSNR {image_names[max_psnr_index]}:": max_psnr.item(),
            f"Min PSNR {image_names[min_psnr_index]}:": min_psnr.item(),
        }
    )

    with open(scene_dir_path / "../render_eval.json", "w", encoding="utf-8") as fp:
        json.dump(full_dict, fp, indent=True)
    with open(
        scene_dir_path / "../render_eval_per_view.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(per_view_dict, fp, indent=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate render and GT directories.")
    parser.add_argument("--gt_color_dir", type=str, required=True)
    parser.add_argument("--renders_color_dir", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
