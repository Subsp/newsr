#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path

import torch


def natural_key(name: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def build_near_original_chunks(total: int, chunk_len: int, min_chunk_len: int):
    if total <= 0:
        return []
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if min_chunk_len <= 0:
        raise ValueError("min_chunk_len must be > 0")
    if min_chunk_len > chunk_len:
        raise ValueError("min_chunk_len must be <= chunk_len")

    num_chunks = max(1, math.ceil(total / chunk_len))
    lengths = [chunk_len] * (num_chunks - 1)
    tail = total - chunk_len * (num_chunks - 1)
    lengths.append(tail)

    # Keep the number of chunk boundaries minimal, but avoid a tiny tail chunk
    # by redistributing only the last full chunk and the tail when needed.
    if len(lengths) >= 2 and lengths[-1] < min_chunk_len:
        combined = lengths[-2] + lengths[-1]
        left = combined // 2
        right = combined - left
        lengths[-2] = left
        lengths[-1] = right

    chunks = []
    start = 0
    for ln in lengths:
        end = start + ln
        chunks.append((start, end))
        start = end
    return chunks


def _next_exact_output_len(frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    remainder = frame_count % 8
    if remainder == 5:
        return frame_count
    return frame_count + ((5 - remainder) % 8)


def _import_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_official_module(repo_root: str, model_dir: str, model_type: str):
    runtime_dir = os.path.join(repo_root, ".hbsr_runtime", "official_baseline_run")
    wan_dir = os.path.join(repo_root, "examples", "WanVSR")
    os.makedirs(runtime_dir, exist_ok=True)
    os.chdir(runtime_dir)

    model_link = os.path.join(runtime_dir, "FlashVSR-v1.1")
    if os.path.lexists(model_link):
        os.remove(model_link)
    os.symlink(model_dir, model_link)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if wan_dir not in sys.path:
        sys.path.insert(0, wan_dir)

    if model_type == "tiny":
        module_path = os.path.join(wan_dir, "infer_flashvsr_v1.1_tiny.py")
        module_name = "_flashvsr_official_tiny"
    elif model_type == "tiny_long":
        module_path = os.path.join(wan_dir, "infer_flashvsr_v1.1_tiny_long_video.py")
        module_name = "_flashvsr_official_tiny_long"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    module = _import_module(module_path, module_name)
    return module, module.init_pipeline()


def symlink_chunk(src_paths: list[str], dst_dir: str):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    exact_len = _next_exact_output_len(len(src_paths))
    run_paths = list(src_paths) + [src_paths[-1]] * (exact_len - len(src_paths))
    for idx, src in enumerate(run_paths):
        ext = Path(src).suffix or ".png"
        dst = os.path.join(dst_dir, f"{idx:04d}{ext}")
        os.symlink(src, dst)
    return exact_len - len(src_paths)


def main():
    parser = argparse.ArgumentParser(description="Run official FlashVSR baseline on sequential chunks and export per-frame priors.")
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="tiny", choices=["tiny", "tiny_long"])
    parser.add_argument("--chunk_len", type=int, default=93)
    parser.add_argument("--min_chunk_len", type=int, default=21)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_chunk_video", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_root = os.path.abspath(os.path.expanduser(args.output_root))

    image_paths = list_images_natural(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    chunks = build_near_original_chunks(
        total=len(image_paths),
        chunk_len=args.chunk_len,
        min_chunk_len=args.min_chunk_len,
    )
    priors_dir = os.path.join(output_root, "priors")
    tmp_root = os.path.join(output_root, "_official_inputs")
    chunk_video_dir = os.path.join(output_root, "chunk_videos")
    os.makedirs(priors_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)
    os.makedirs(chunk_video_dir, exist_ok=True)

    module, pipe = init_official_module(repo_root, model_dir, args.model_type)
    dtype = torch.bfloat16
    device = "cuda"
    scale = 4.0
    sparse_ratio = 2.0

    total_written = 0
    for cid, (start, end) in enumerate(chunks):
        chunk_paths = image_paths[start:end]
        chunk_stems = [Path(p).stem for p in chunk_paths]
        chunk_dir = os.path.join(tmp_root, f"chunk_{cid:04d}")
        pad_count = symlink_chunk(chunk_paths, chunk_dir)

        print(
            f"[flashvsr-official-baseline] chunk {cid+1}/{len(chunks)} "
            f"range=[{start},{end}) frames={len(chunk_paths)} pad={pad_count} "
            f"model_type={args.model_type}"
        )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        lq, th, tw, f_all, fps = module.prepare_input_tensor(
            chunk_dir,
            scale=scale,
            dtype=dtype,
            device=device,
        )
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            LQ_video=lq,
            num_frames=f_all,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=11,
            color_fix=True,
        )
        video_frames = module.tensor2video(video)
        if len(video_frames) < len(chunk_paths):
            raise RuntimeError(
                f"Official baseline output length mismatch for chunk {cid}: "
                f"got={len(video_frames)} expected_at_least={len(chunk_paths)}"
            )
        video_frames = video_frames[: len(chunk_paths)]
        if args.save_chunk_video:
            chunk_video_path = os.path.join(
                chunk_video_dir,
                f"FlashVSR_official_{args.model_type}_chunk{cid:04d}_seed{args.seed}.mp4",
            )
            module.save_video(video_frames, chunk_video_path, fps=args.fps, quality=6)
        for stem, frame in zip(chunk_stems, video_frames):
            frame.save(os.path.join(priors_dir, f"{stem}.png"))
        total_written += len(chunk_paths)

    print(
        "[flashvsr-official-baseline] done\n"
        f"  input_dir      : {input_dir}\n"
        f"  output_root    : {output_root}\n"
        f"  priors_dir     : {priors_dir}\n"
        f"  frames_written : {total_written}"
    )


if __name__ == "__main__":
    main()
