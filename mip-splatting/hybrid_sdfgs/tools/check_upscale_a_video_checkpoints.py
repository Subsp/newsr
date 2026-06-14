#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List


REQUIRED = {
    "root": [
        "model_index.json",
    ],
    "core": [
        "low_res_scheduler/scheduler_config.json",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
        "text_encoder/config.json",
        "vae/vae_3d_config.json",
        "vae/vae_3d.bin",
        "unet/unet_video_config.json",
        "unet/unet_video.bin",
    ],
    "optional_video_vae": [
        "vae/vae_video_config.json",
        "vae/vae_video.bin",
    ],
    "optional_propagator": [
        "propagator/raft-things.pth",
    ],
    "optional_text_encoder_weights": [
        "text_encoder/pytorch_model.bin",
        "text_encoder/model.safetensors",
    ],
    "optional_llava": [
        "../liuhaotian-llava-v1.5-13b",
    ],
}


def _exists_any(root: Path, rels: List[str]) -> bool:
    return any((root / rel).exists() for rel in rels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/ltl/Desktop/codex_playground/video_sr_models/Upscale-A-Video/pretrained_models/upscale_a_video"),
    )
    args = parser.parse_args()

    root = args.root
    report: dict[str, object] = {
        "root": str(root),
        "exists": root.exists(),
        "required_missing": [],
        "optional_missing": {},
        "notes": [],
    }

    missing = []
    for group in ("root", "core"):
        for rel in REQUIRED[group]:
            if not (root / rel).exists():
                missing.append(rel)

    report["required_missing"] = missing

    opt = {}
    for rel in REQUIRED["optional_video_vae"]:
        opt.setdefault("optional_video_vae", [])
        if not (root / rel).exists():
            opt["optional_video_vae"].append(rel)

    for rel in REQUIRED["optional_propagator"]:
        opt.setdefault("optional_propagator", [])
        if not (root / rel).exists():
            opt["optional_propagator"].append(rel)

    if not _exists_any(root, REQUIRED["optional_text_encoder_weights"]):
        opt["optional_text_encoder_weights"] = REQUIRED["optional_text_encoder_weights"]

    llava_root = root.parent / "liuhaotian-llava-v1.5-13b"
    if not llava_root.exists():
        opt["optional_llava"] = [str(llava_root)]

    report["optional_missing"] = opt

    if missing:
        report["notes"].append("core checkpoints are incomplete; inference_upscale_a_video.py will not start")
    else:
        report["notes"].append("core checkpoints look complete")

    if opt.get("optional_propagator"):
        report["notes"].append("propagation steps require RAFT weights")
    if opt.get("optional_video_vae"):
        report["notes"].append("--use_video_vae requires the optional video VAE pair")
    if opt.get("optional_llava"):
        report["notes"].append("LLaVA is optional and only needed when not using --no_llava")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
