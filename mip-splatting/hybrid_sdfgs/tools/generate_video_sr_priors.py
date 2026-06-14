#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from glob import glob


def _collect_images(input_dir: str, exts: list[str]) -> list[str]:
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(input_dir, f"*.{ext}")))
        files.extend(glob(os.path.join(input_dir, f"*.{ext.upper()}")))
    return sorted(set(files))


def _run(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _prepare_single_sequence(input_images: list[str], seq_dir: str):
    os.makedirs(seq_dir, exist_ok=True)
    for src in input_images:
        dst = os.path.join(seq_dir, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def _prepare_single_sequence_downsampled(
    input_images: list[str],
    seq_dir: str,
    downsample: int,
) -> list[str]:
    if downsample <= 1:
        _prepare_single_sequence(input_images, seq_dir)
        return [os.path.join(seq_dir, os.path.basename(x)) for x in input_images]

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --input_downsample > 1.") from exc

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BICUBIC
    else:
        resample = Image.BICUBIC

    os.makedirs(seq_dir, exist_ok=True)
    prepared = []
    for src in input_images:
        dst = os.path.join(seq_dir, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        with Image.open(src) as img:
            w, h = img.size
            w_lr = max(1, w // downsample)
            h_lr = max(1, h // downsample)
            img_lr = img.resize((w_lr, h_lr), resample=resample)
            img_lr.save(dst)
        prepared.append(dst)
    print(
        "[video-prior] input downsample applied: "
        f"x{downsample}, resized frames written to {seq_dir}"
    )
    return prepared


def _copy_priors_from_sequence(
    generated_seq_dir: str,
    output_prior_dir: str,
    ref_stems: list[str] | None = None,
):
    os.makedirs(output_prior_dir, exist_ok=True)
    gen_images = sorted(glob(os.path.join(generated_seq_dir, "*.png")))
    if not gen_images:
        raise FileNotFoundError(f"No generated png found under {generated_seq_dir}")

    if ref_stems is not None:
        if len(ref_stems) != len(gen_images):
            raise ValueError(
                f"UAV frame count mismatch: generated={len(gen_images)} input={len(ref_stems)}"
            )
        dst_stems = ref_stems
    else:
        dst_stems = [os.path.splitext(os.path.basename(src))[0] for src in gen_images]

    for src, stem in zip(gen_images, dst_stems):
        dst = os.path.join(output_prior_dir, f"{stem}.png")
        shutil.copy2(src, dst)
    print(f"[video-prior] copied {len(gen_images)} priors -> {output_prior_dir}")


def _run_realbasic(args, input_seq_dir: str, generated_root: str):
    repo = os.path.abspath(os.path.expanduser(args.repo_root))
    script = os.path.join(repo, "inference_realbasicvsr.py")
    config = args.realbasic_config
    if not os.path.isabs(config):
        config = os.path.join(repo, config)
    ckpt = os.path.abspath(os.path.expanduser(args.realbasic_checkpoint))
    out_dir = os.path.join(generated_root, "realbasic_seq0")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        args.python_exe,
        script,
        config,
        ckpt,
        input_seq_dir,
        out_dir,
    ]
    if args.realbasic_max_seq_len > 0:
        cmd.extend(["--max_seq_len", str(args.realbasic_max_seq_len)])
    _run(cmd, cwd=repo)
    return out_dir


def _run_rvrt(args, lq_root: str):
    repo = os.path.abspath(os.path.expanduser(args.repo_root))
    script = os.path.join(repo, "main_test_rvrt.py")
    task_root = os.path.join(repo, "results", args.rvrt_task)
    if os.path.isdir(task_root):
        shutil.rmtree(task_root)
    cmd = [
        args.python_exe,
        script,
        "--task",
        args.rvrt_task,
        "--folder_lq",
        lq_root,
        "--num_workers",
        "0",
        "--save_result",
    ]
    _run(cmd, cwd=repo)
    return os.path.join(task_root, "seq0")


def _run_vrt(args, lq_root: str):
    repo = os.path.abspath(os.path.expanduser(args.repo_root))
    script = os.path.join(repo, "main_test_vrt.py")
    task_root = os.path.join(repo, "results", args.vrt_task)
    if os.path.isdir(task_root):
        shutil.rmtree(task_root)
    cmd = [
        args.python_exe,
        script,
        "--task",
        args.vrt_task,
        "--folder_lq",
        lq_root,
        "--num_workers",
        "0",
        "--save_result",
    ]
    _run(cmd, cwd=repo)
    return os.path.join(task_root, "seq0")


def _patch_uav_save_loop_if_needed(script_path: str):
    if not os.path.isfile(script_path):
        return
    bad = "for i in range(output.shape[2]):"
    good = "for i in range(output.shape[0]):"
    with open(script_path, "r", encoding="utf-8") as f:
        text = f.read()
    if bad not in text:
        return
    patched = text.replace(bad, good)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(
        "[video-prior] auto-patched UAV save loop bug "
        f"in {script_path} ({bad} -> {good})"
    )


def _patch_uav_utils_numpy_import_if_needed(utils_path: str):
    if not os.path.isfile(utils_path):
        raise FileNotFoundError(f"UAV utils file not found: {utils_path}")
    with open(utils_path, "r", encoding="utf-8") as f:
        text = f.read()
    if "import numpy as np" in text:
        return
    anchor = "import torchvision\n"
    if anchor not in text:
        raise RuntimeError(f"Could not find import anchor in UAV utils file: {utils_path}")
    patched = text.replace(anchor, anchor + "import numpy as np\n", 1)
    with open(utils_path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(
        "[video-prior] auto-patched UAV utils missing numpy import "
        f"in {utils_path}"
    )


def _images_to_video(input_images: list[str], output_video_path: str, fps: int):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for --model uav (pip install opencv-python-headless).") from exc

    if not input_images:
        raise ValueError("No input images for video packaging.")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = None
    target_w = target_h = None
    try:
        for idx, image_path in enumerate(input_images):
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            h, w = frame.shape[:2]
            if writer is None:
                target_w, target_h = w, h
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (target_w, target_h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_video_path}")
            else:
                if (w, h) != (target_w, target_h):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()


def _run_uav(args, input_images: list[str], generated_root: str):
    repo = os.path.abspath(os.path.expanduser(args.repo_root))
    script = os.path.join(repo, "inference_upscale_a_video.py")
    utils_py = os.path.join(repo, "utils.py")
    _patch_uav_save_loop_if_needed(script)
    _patch_uav_utils_numpy_import_if_needed(utils_py)
    pretrained_root = os.path.join(repo, "pretrained_models", "upscale_a_video")
    if not os.path.isdir(pretrained_root):
        raise FileNotFoundError(
            f"Upscale-A-Video pretrained folder not found: {pretrained_root}"
        )

    out_dir = os.path.join(generated_root, "uav")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not input_images:
        raise ValueError("No input images for UAV.")
    input_sequence_dir = os.path.dirname(os.path.abspath(input_images[0]))
    if not os.path.isdir(input_sequence_dir):
        raise FileNotFoundError(f"UAV input image directory not found: {input_sequence_dir}")

    cmd = [
        args.python_exe,
        script,
        "-i",
        input_sequence_dir,
        "-o",
        out_dir,
        "-n",
        str(args.uav_noise_level),
        "-g",
        str(args.uav_guidance_scale),
        "-s",
        str(args.uav_inference_steps),
        "--a_prompt",
        args.uav_a_prompt,
        "--n_prompt",
        args.uav_n_prompt,
        "--save_image",
        "--save_suffix",
        args.uav_save_suffix,
        "--color_fix",
        args.uav_color_fix,
    ]
    if not args.uav_use_llava:
        cmd.append("--no_llava")
    if args.uav_use_video_vae:
        cmd.append("--use_video_vae")
    if args.uav_perform_tile:
        cmd.extend(["--perform_tile", "--tile_size", str(args.uav_tile_size)])
    if args.uav_propagation_steps:
        cmd.extend(["-p", args.uav_propagation_steps])

    _run(cmd, cwd=repo)

    prop = ""
    if args.uav_propagation_steps:
        steps = [x.strip() for x in args.uav_propagation_steps.split(",") if x.strip()]
        prop = "_p" + "_".join(steps)
    suffix = f"_{args.uav_save_suffix}" if args.uav_save_suffix else ""
    video_stem = os.path.basename(input_sequence_dir.rstrip(os.sep))
    save_name = (
        f"{video_stem}_n{args.uav_noise_level}_g{args.uav_guidance_scale}"
        f"_s{args.uav_inference_steps}{prop}{suffix}"
    )
    generated_seq_dir = os.path.join(out_dir, "frame", save_name)
    if not os.path.isdir(generated_seq_dir):
        frame_root = os.path.join(out_dir, "frame")
        raise FileNotFoundError(
            f"Upscale-A-Video frame output not found: {generated_seq_dir} (frame root: {frame_root})"
        )
    return generated_seq_dir


def _run_flashvsr(args, input_dir: str, output_root: str):
    tool = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "generate_flashvsr_priors_official_chunked.py",
    )
    if not os.path.isfile(tool):
        raise FileNotFoundError(f"FlashVSR tool script not found: {tool}")

    cmd = [
        args.python_exe,
        tool,
        "--repo_root",
        args.repo_root,
        "--model_type",
        args.flashvsr_model_type,
        "--input_dir",
        input_dir,
        "--output_root",
        output_root,
        "--scale",
        str(args.flashvsr_scale),
        "--fps",
        str(args.flashvsr_fps),
        "--chunk_size",
        str(args.flashvsr_chunk_size),
        "--chunk_min",
        str(args.flashvsr_chunk_min),
        "--align_mode",
        args.flashvsr_align_mode,
        "--align_multiple",
        str(args.flashvsr_align_multiple),
        "--kv_ratio",
        str(args.flashvsr_kv_ratio),
        "--local_range",
        str(args.flashvsr_local_range),
        "--sparse_ratio",
        str(args.flashvsr_sparse_ratio),
        "--view_group_mode",
        args.flashvsr_view_group_mode,
        "--view_group_max_len",
        str(args.flashvsr_view_group_max_len),
        "--view_group_min_len",
        str(args.flashvsr_view_group_min_len),
        "--view_group_thresholds",
        args.flashvsr_view_group_thresholds,
        "--view_dir_weight",
        str(args.flashvsr_view_dir_weight),
    ]
    if args.flashvsr_model_dir:
        cmd.extend(["--model_dir", args.flashvsr_model_dir])
    if args.flashvsr_spatial_tile > 0:
        cmd.extend(["--spatial_tile", str(args.flashvsr_spatial_tile)])
    if args.flashvsr_spatial_tile_w > 0:
        cmd.extend(["--spatial_tile_w", str(args.flashvsr_spatial_tile_w)])
    if args.flashvsr_spatial_tile_h > 0:
        cmd.extend(["--spatial_tile_h", str(args.flashvsr_spatial_tile_h)])
    if args.flashvsr_spatial_overlap > 0:
        cmd.extend(["--spatial_overlap", str(args.flashvsr_spatial_overlap)])
    if args.flashvsr_colmap_sparse_dir:
        cmd.extend(["--colmap_sparse_dir", args.flashvsr_colmap_sparse_dir])
    if args.flashvsr_dtype:
        cmd.extend(["--dtype", args.flashvsr_dtype])
    if args.flashvsr_device:
        cmd.extend(["--device", args.flashvsr_device])
    if args.flashvsr_no_match_exact_scale_size:
        cmd.append("--no_match_exact_scale_size")
    if args.flashvsr_if_buffer:
        cmd.append("--if_buffer")
    if args.flashvsr_save_chunk_video:
        cmd.append("--save_chunk_video")
    _run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Generate decoupled video-SR priors from image sequence (single-sequence wrapper)."
    )
    parser.add_argument("--model", type=str, required=True, choices=["realbasic", "rvrt", "vrt", "uav", "flashvsr"])
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--input_downsample", type=int, default=1)
    parser.add_argument("--work_root", type=str, default="")
    parser.add_argument("--python_exe", type=str, default="python")
    parser.add_argument("--exts", type=str, default="png,jpg,jpeg")

    parser.add_argument("--realbasic_config", type=str, default="configs/realbasicvsr_x4.py")
    parser.add_argument("--realbasic_checkpoint", type=str, default="")
    parser.add_argument("--realbasic_max_seq_len", type=int, default=0)

    parser.add_argument("--rvrt_task", type=str, default="001_RVRT_videosr_bi_REDS_30frames")
    parser.add_argument("--vrt_task", type=str, default="001_VRT_videosr_bi_REDS_6frames")

    parser.add_argument("--uav_noise_level", type=int, default=40)
    parser.add_argument("--uav_guidance_scale", type=int, default=2)
    parser.add_argument("--uav_inference_steps", type=int, default=12)
    parser.add_argument("--uav_fps", type=int, default=24)
    parser.add_argument(
        "--uav_a_prompt",
        type=str,
        default="high quality, sharp details, faithful to input frame content",
    )
    parser.add_argument(
        "--uav_n_prompt",
        type=str,
        default="cartoon, anime, painting, stylized, unrealistic, blur, low quality",
    )
    parser.add_argument("--uav_color_fix", type=str, default="AdaIn", choices=["None", "AdaIn", "Wavelet"])
    parser.add_argument("--uav_use_llava", action="store_true", default=False)
    parser.add_argument("--uav_use_video_vae", action="store_true", default=False)
    parser.add_argument("--uav_perform_tile", action="store_true", default=False)
    parser.add_argument("--uav_tile_size", type=int, default=256)
    parser.add_argument("--uav_propagation_steps", type=str, default="")
    parser.add_argument("--uav_save_suffix", type=str, default="hbsr_uav")

    parser.add_argument("--flashvsr_variant", type=str, default="v1.1", choices=["v1", "v1.1"])
    parser.add_argument("--flashvsr_model_type", type=str, default="tiny", choices=["tiny", "tiny_long", "full"])
    parser.add_argument("--flashvsr_model_dir", type=str, default="")
    parser.add_argument("--flashvsr_cache_root", type=str, default="")
    parser.add_argument("--flashvsr_fps", type=int, default=24)
    parser.add_argument("--flashvsr_keep_runtime", action="store_true", default=False)
    parser.add_argument("--flashvsr_scale", type=float, default=4.0)
    parser.add_argument("--flashvsr_chunk_size", type=int, default=60)
    parser.add_argument("--flashvsr_chunk_min", type=int, default=50)
    parser.add_argument("--flashvsr_align_mode", type=str, default="floor", choices=["floor", "ceil"])
    parser.add_argument("--flashvsr_align_multiple", type=int, default=128)
    parser.add_argument("--flashvsr_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--flashvsr_device", type=str, default="cuda")
    parser.add_argument("--flashvsr_kv_ratio", type=float, default=3.0)
    parser.add_argument("--flashvsr_local_range", type=int, default=11)
    parser.add_argument("--flashvsr_sparse_ratio", type=float, default=2.0)
    parser.add_argument("--flashvsr_if_buffer", action="store_true", default=False)
    parser.add_argument("--flashvsr_no_match_exact_scale_size", action="store_true", default=False)
    parser.add_argument("--flashvsr_save_chunk_video", action="store_true", default=False)
    parser.add_argument("--flashvsr_spatial_tile", type=int, default=0)
    parser.add_argument("--flashvsr_spatial_tile_w", type=int, default=0)
    parser.add_argument("--flashvsr_spatial_tile_h", type=int, default=0)
    parser.add_argument("--flashvsr_spatial_overlap", type=int, default=128)
    parser.add_argument(
        "--flashvsr_view_group_mode",
        type=str,
        default="none",
        choices=["none", "seqmat_pose_als"],
    )
    parser.add_argument("--flashvsr_colmap_sparse_dir", type=str, default="")
    parser.add_argument("--flashvsr_view_group_max_len", type=int, default=8)
    parser.add_argument("--flashvsr_view_group_min_len", type=int, default=3)
    parser.add_argument("--flashvsr_view_group_thresholds", type=str, default="30,50")
    parser.add_argument("--flashvsr_view_dir_weight", type=float, default=0.0)
    args = parser.parse_args()

    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    if args.model == "flashvsr":
        _run_flashvsr(args, input_dir=input_dir, output_root=output_root)
        print(f"[video-prior] done model={args.model} output_root={output_root}")
        return

    if args.work_root:
        work_root = os.path.abspath(os.path.expanduser(args.work_root))
    else:
        work_root = os.path.join(output_root, "_video_sr_work")
    lq_root = os.path.join(work_root, "lq")
    seq0_dir = os.path.join(lq_root, "seq0")
    generated_root = os.path.join(work_root, "generated")
    prior_dir = os.path.join(output_root, "priors")
    os.makedirs(generated_root, exist_ok=True)

    exts = [x.strip().lstrip(".").lower() for x in args.exts.split(",") if x.strip()]
    images = _collect_images(input_dir, exts)
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir} with exts={exts}")
    if args.input_downsample < 1:
        raise ValueError("--input_downsample must be >= 1")

    if os.path.isdir(seq0_dir):
        shutil.rmtree(seq0_dir)
    prepared_images = _prepare_single_sequence_downsampled(
        images,
        seq0_dir,
        downsample=args.input_downsample,
    )

    if args.model == "realbasic":
        if not args.realbasic_checkpoint:
            raise ValueError("--realbasic_checkpoint is required when --model realbasic")
        generated_seq_dir = _run_realbasic(args, seq0_dir, generated_root)
        _copy_priors_from_sequence(generated_seq_dir=generated_seq_dir, output_prior_dir=prior_dir)
    elif args.model == "rvrt":
        generated_seq_dir = _run_rvrt(args, lq_root)
        _copy_priors_from_sequence(generated_seq_dir=generated_seq_dir, output_prior_dir=prior_dir)
    elif args.model == "vrt":
        generated_seq_dir = _run_vrt(args, lq_root)
        _copy_priors_from_sequence(generated_seq_dir=generated_seq_dir, output_prior_dir=prior_dir)
    else:
        generated_seq_dir = _run_uav(args, prepared_images, generated_root)
        input_stems = [os.path.splitext(os.path.basename(x))[0] for x in images]
        _copy_priors_from_sequence(
            generated_seq_dir=generated_seq_dir,
            output_prior_dir=prior_dir,
            ref_stems=input_stems,
        )
    print(f"[video-prior] done model={args.model} output_root={output_root}")


if __name__ == "__main__":
    main()
