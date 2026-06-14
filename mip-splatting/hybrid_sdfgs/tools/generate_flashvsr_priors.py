#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from glob import glob
import numpy as np


def _natural_key(path: str):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", name)]


def _collect_images(input_dir: str, exts: list[str]) -> list[str]:
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(input_dir, f"*.{ext}")))
        files.extend(glob(os.path.join(input_dir, f"*.{ext.upper()}")))
    files = sorted(set(files), key=_natural_key)
    return files


def _run(cmd: list[str], cwd: str | None = None):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _prepare_images(
    input_images: list[str],
    work_dir: str,
    downsample: int,
) -> list[str]:
    os.makedirs(work_dir, exist_ok=True)
    if downsample <= 1:
        prepared = []
        for src in input_images:
            dst = os.path.join(work_dir, os.path.basename(src))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            prepared.append(dst)
        return prepared

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --input_downsample > 1") from exc

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BICUBIC
    else:
        resample = Image.BICUBIC

    prepared = []
    for src in input_images:
        dst = os.path.join(work_dir, os.path.basename(src))
        with Image.open(src) as img:
            w, h = img.size
            w_lr = max(1, w // downsample)
            h_lr = max(1, h // downsample)
            img_lr = img.resize((w_lr, h_lr), resample=resample)
            img_lr.save(dst)
        prepared.append(dst)
    print(f"[flashvsr-prior] downsample x{downsample} -> {work_dir}")
    return prepared


def _images_to_video(images: list[str], output_video: str, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to package input video") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to package input video") from exc

    if not images:
        raise ValueError("No images to package into video.")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    with Image.open(images[0]) as img0:
        target_size = img0.size

    # macro_block_size=1 prevents implicit ffmpeg resize to 16-aligned frames.
    writer = imageio.get_writer(output_video, fps=fps, macro_block_size=1)
    try:
        for path in images:
            with Image.open(path).convert("RGB") as img:
                if img.size != target_size:
                    img = img.resize(target_size, Image.BICUBIC)
                writer.append_data(np.asarray(img))
    finally:
        writer.close()


def _script_and_prefix(variant: str, model_type: str) -> tuple[str, str]:
    if variant == "v1":
        if model_type == "tiny":
            return "infer_flashvsr_tiny.py", "FlashVSR_Tiny"
        if model_type == "tiny_long":
            return "infer_flashvsr_tiny_long_video.py", "FlashVSR_Tiny_Long"
        return "infer_flashvsr_full.py", "FlashVSR_Full"
    if model_type == "tiny":
        return "infer_flashvsr_v1.1_tiny.py", "FlashVSR_v1.1_Tiny"
    if model_type == "tiny_long":
        return "infer_flashvsr_v1.1_tiny_long_video.py", "FlashVSR_v1.1_Tiny_Long"
    return "infer_flashvsr_v1.1_full.py", "FlashVSR_v1.1_Full"


def _expected_model_folder_name(variant: str) -> str:
    return "FlashVSR" if variant == "v1" else "FlashVSR-v1.1"


def _validate_model_dir(model_dir: str, model_type: str):
    required = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ]
    if model_type == "full":
        required.append("Wan2.1_VAE.pth")
    missing = [x for x in required if not os.path.isfile(os.path.join(model_dir, x))]
    if missing:
        raise FileNotFoundError(
            f"FlashVSR model dir missing files: {missing}\nmodel_dir={model_dir}"
        )


def _find_output_video(results_dir: str, prefix: str) -> str:
    pattern = os.path.join(results_dir, f"{prefix}_example0_seed*.mp4")
    candidates = sorted(glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No FlashVSR output video found with pattern: {pattern}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _video_to_frames(video_path: str) -> list:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to decode output video") from exc
    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def _write_priors(frames: list, dst_dir: str, stems: list[str]):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write prior images") from exc
    os.makedirs(dst_dir, exist_ok=True)
    if len(frames) < len(stems):
        raise RuntimeError(
            f"FlashVSR output too short: output_frames={len(frames)} input_frames={len(stems)}"
        )
    if len(frames) > len(stems):
        print(
            "[flashvsr-prior] output has padded tail frames; "
            f"using first {len(stems)} / {len(frames)} frames"
        )
    for idx, stem in enumerate(stems):
        frame = frames[idx]
        if frame.ndim == 3 and frame.shape[2] > 3:
            frame = frame[:, :, :3]
        out = os.path.join(dst_dir, f"{stem}.png")
        Image.fromarray(frame).save(out)


def _flashvsr_expected_output_frames(input_count: int) -> int:
    """
    FlashVSR scripts internally convert frame count with:
      n_input -> n_input + 4 -> largest(8k+1) <= value -> output = result - 4
    Equivalent closed form:
      out = ((n_input + 3) // 8) * 8 - 3
    """
    if input_count <= 0:
        return 0
    return ((input_count + 3) // 8) * 8 - 3


def _pad_images_for_chunk_target(chunk_images: list[str], target_count: int) -> list[str]:
    if not chunk_images:
        return []
    run_images = list(chunk_images)
    while _flashvsr_expected_output_frames(len(run_images)) < target_count:
        run_images.append(run_images[-1])
    return run_images


def _build_chunk_ranges(total_count: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if total_count <= 0:
        return []
    if chunk_size <= 0 or chunk_size >= total_count:
        return [(0, total_count)]
    if overlap < 0:
        raise ValueError("--flashvsr_chunk_overlap must be >= 0")
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("--flashvsr_chunk_overlap must be < --flashvsr_chunk_size")

    # Sequential chunking without backtracking:
    # [0,chunk), [chunk,2*chunk), ... , [k*chunk,total)
    # This guarantees no dropped frames and no implicit duplicated tail.
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_count:
        end = min(start + chunk_size, total_count)
        ranges.append((start, end))
        if end >= total_count:
            break
        start = end - overlap
    return ranges


def _report_scale_status(chunk_id: int, input_image_path: str, output_frame) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(input_image_path) as img:
        in_w, in_h = img.size
    out_h, out_w = output_frame.shape[:2]
    exact = (out_w == in_w * 4) and (out_h == in_h * 4)
    print(
        f"[flashvsr-prior] chunk {chunk_id+1} scale-check: "
        f"input={in_w}x{in_h}, output={out_w}x{out_h}, "
        f"expected_x4={in_w*4}x{in_h*4}, exact_x4={exact}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate FlashVSR priors from an image sequence.")
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--python_exe", type=str, default="python")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="")
    parser.add_argument("--input_downsample", type=int, default=1)
    parser.add_argument("--exts", type=str, default="png,jpg,jpeg")
    parser.add_argument("--flashvsr_variant", type=str, default="v1.1", choices=["v1", "v1.1"])
    parser.add_argument("--flashvsr_model_type", type=str, default="tiny", choices=["tiny", "tiny_long", "full"])
    parser.add_argument("--flashvsr_model_dir", type=str, default="")
    parser.add_argument("--flashvsr_fps", type=int, default=24)
    parser.add_argument("--flashvsr_chunk_size", type=int, default=60)
    parser.add_argument("--flashvsr_chunk_overlap", type=int, default=0)
    parser.add_argument("--keep_runtime", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    cache_root = (
        os.path.abspath(os.path.expanduser(args.cache_root))
        if args.cache_root
        else os.path.join(output_root, "_flashvsr_cache")
    )
    prior_dir = os.path.join(output_root, "priors")
    # FlashVSR example scripts use hard-coded relative paths such as:
    # ../../examples/WanVSR/prompt_tensor/posi_prompt.pth
    # So runtime cwd must be placed under repo_root/*/* to keep these paths valid.
    runtime_dir = os.path.join(repo_root, ".hbsr_runtime", "run")
    prepared_dir = os.path.join(cache_root, "prepared_input")
    generated_video_dir = os.path.join(cache_root, "generated_video")
    generated_frames_dir = os.path.join(cache_root, "generated_frames")
    run_inputs_dir = os.path.join(runtime_dir, "inputs")
    run_results_dir = os.path.join(runtime_dir, "results")

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"FlashVSR repo not found: {repo_root}")
    wan_dir = os.path.join(repo_root, "examples", "WanVSR")
    if not os.path.isdir(wan_dir):
        raise FileNotFoundError(f"WanVSR example folder not found: {wan_dir}")
    script_name, output_prefix = _script_and_prefix(args.flashvsr_variant, args.flashvsr_model_type)
    infer_script = os.path.join(wan_dir, script_name)
    if not os.path.isfile(infer_script):
        raise FileNotFoundError(f"FlashVSR inference script not found: {infer_script}")

    model_dir = args.flashvsr_model_dir
    if model_dir:
        model_dir = os.path.abspath(os.path.expanduser(model_dir))
    else:
        model_dir = os.path.join(wan_dir, _expected_model_folder_name(args.flashvsr_variant))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"FlashVSR model dir not found: {model_dir}")
    _validate_model_dir(model_dir, args.flashvsr_model_type)

    exts = [x.strip().lstrip(".").lower() for x in args.exts.split(",") if x.strip()]
    input_images = _collect_images(input_dir, exts)
    if not input_images:
        raise FileNotFoundError(f"No images found in {input_dir} with exts={exts}")
    stems = [os.path.splitext(os.path.basename(x))[0] for x in input_images]

    if args.input_downsample < 1:
        raise ValueError("--input_downsample must be >= 1")

    os.makedirs(cache_root, exist_ok=True)
    if os.path.isdir(prepared_dir):
        shutil.rmtree(prepared_dir)
    prepared_images = _prepare_images(input_images, prepared_dir, downsample=args.input_downsample)

    if os.path.isdir(runtime_dir):
        shutil.rmtree(runtime_dir)
    os.makedirs(run_inputs_dir, exist_ok=True)
    os.makedirs(run_results_dir, exist_ok=True)

    model_link_name = _expected_model_folder_name(args.flashvsr_variant)
    model_link_path = os.path.join(runtime_dir, model_link_name)
    if os.path.lexists(model_link_path):
        os.remove(model_link_path)
    os.symlink(model_dir, model_link_path)
    os.makedirs(generated_video_dir, exist_ok=True)

    chunk_ranges = _build_chunk_ranges(
        total_count=len(prepared_images),
        chunk_size=args.flashvsr_chunk_size,
        overlap=args.flashvsr_chunk_overlap,
    )
    print(
        "[flashvsr-prior] chunk setup: "
        f"size={args.flashvsr_chunk_size}, overlap={args.flashvsr_chunk_overlap}, "
        f"num_chunks={len(chunk_ranges)}"
    )

    if os.path.isdir(prior_dir):
        shutil.rmtree(prior_dir)
    os.makedirs(prior_dir, exist_ok=True)
    if os.path.isdir(generated_frames_dir):
        shutil.rmtree(generated_frames_dir)
    os.makedirs(generated_frames_dir, exist_ok=True)

    total_written = 0
    last_cached_output_video = ""
    cmd = [args.python_exe, infer_script]
    for chunk_id, (start, end) in enumerate(chunk_ranges):
        chunk_images = prepared_images[start:end]
        chunk_stems = stems[start:end]
        run_images = _pad_images_for_chunk_target(chunk_images, len(chunk_stems))
        pad_count = len(run_images) - len(chunk_images)
        print(
            f"[flashvsr-prior] chunk {chunk_id+1}/{len(chunk_ranges)} "
            f"range=[{start},{end}) frames={len(chunk_images)} pad={pad_count}"
        )

        if os.path.isdir(run_results_dir):
            shutil.rmtree(run_results_dir)
        os.makedirs(run_results_dir, exist_ok=True)

        input_video = os.path.join(run_inputs_dir, "example0.mp4")
        if os.path.exists(input_video):
            os.remove(input_video)
        _images_to_video(run_images, input_video, fps=args.flashvsr_fps)

        _run(cmd, cwd=runtime_dir)

        output_video = _find_output_video(run_results_dir, output_prefix)
        cached_output_video = os.path.join(
            generated_video_dir,
            f"{output_prefix}_example0_chunk{chunk_id:04d}_{start:06d}_{end:06d}.mp4",
        )
        shutil.copy2(output_video, cached_output_video)
        last_cached_output_video = cached_output_video

        frames = _video_to_frames(cached_output_video)
        if frames:
            _report_scale_status(chunk_id, chunk_images[0], frames[0])
        _write_priors(frames, prior_dir, chunk_stems)
        total_written += len(chunk_stems)

        # Save mapped frames per chunk for debugging.
        try:
            from PIL import Image

            for idx, stem in enumerate(chunk_stems):
                frame = frames[idx]
                if frame.ndim == 3 and frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                Image.fromarray(frame).save(
                    os.path.join(generated_frames_dir, f"c{chunk_id:04d}_{idx:04d}_{stem}.png")
                )
        except Exception as exc:
            print(f"[flashvsr-prior] warning: failed to export cache frames: {exc}")

    if not args.keep_runtime and os.path.isdir(runtime_dir):
        shutil.rmtree(runtime_dir)

    print(
        "[flashvsr-prior] done\n"
        f"  input_dir   : {input_dir}\n"
        f"  output_root : {output_root}\n"
        f"  prior_dir   : {prior_dir}\n"
        f"  cache_root  : {cache_root}\n"
        f"  output_video(last): {last_cached_output_video}\n"
        f"  priors_written: {total_written}"
    )


if __name__ == "__main__":
    main()
