#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

MIN_FLASHVSR_INPUT_FRAMES = 25


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom <= 1e-8:
        return 0.0
    cos_angle = np.dot(vec1, vec2) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _load_colmap_pose_info(image_paths: Sequence[str], sparse_dir: str):
    project_root = _project_root()
    loader_path = os.path.join(project_root, "scene", "colmap_loader.py")
    if not os.path.isfile(loader_path):
        raise FileNotFoundError(f"COLMAP loader not found: {loader_path}")
    module_name = "_hybrid_colmap_loader"
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for COLMAP loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read_extrinsics_binary = module.read_extrinsics_binary
    qvec2rotmat = module.qvec2rotmat

    sparse_dir = os.path.abspath(os.path.expanduser(sparse_dir))
    images_bin = os.path.join(sparse_dir, "images.bin")
    if not os.path.isfile(images_bin):
        raise FileNotFoundError(f"COLMAP images.bin not found: {images_bin}")

    extrinsics = read_extrinsics_binary(images_bin)
    by_name = {}
    for extr in extrinsics.values():
        by_name[os.path.basename(extr.name)] = extr

    centers = []
    view_dirs = []
    missing = []
    for image_path in image_paths:
        name = os.path.basename(image_path)
        extr = by_name.get(name)
        if extr is None:
            missing.append(name)
            continue
        rot_cw = qvec2rotmat(extr.qvec)
        tvec = np.asarray(extr.tvec, dtype=np.float64)
        cam_center = -(rot_cw.T @ tvec)
        view_dir = rot_cw.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        centers.append(cam_center)
        view_dirs.append(view_dir / (np.linalg.norm(view_dir) + 1e-8))
    if missing:
        raise KeyError(
            "Some input frames are missing in COLMAP images.bin: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    return np.stack(centers, axis=0), np.stack(view_dirs, axis=0)


def _compute_orb_feature_rankings(image_paths: Sequence[str]):
    import cv2

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    features = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image for ORB feature extraction: {path}")
        kp, des = orb.detectAndCompute(img, None)
        features.append((kp, des))

    rankings_all = []
    num_images = len(image_paths)
    for i in range(num_images):
        distances = []
        for j in range(num_images):
            if i == j:
                distances.append(0.0)
                continue
            des_i = features[i][1]
            des_j = features[j][1]
            if des_i is None or des_j is None:
                distances.append(np.inf)
                continue
            matches = bf.match(des_i, des_j)
            if not matches:
                distances.append(np.inf)
                continue
            distances.append(float(sum(m.distance for m in matches) / len(matches)))
        sorted_indices = np.argsort(distances)
        rankings = [[int(np.where(sorted_indices == j)[0][0]), float(distances[j])] for j in range(num_images)]
        rankings_all.append(rankings)
    return np.asarray(rankings_all, dtype=np.float64)


def _compute_pose_rankings(
    camera_centers: np.ndarray,
    view_dirs: np.ndarray,
    view_dir_weight: float,
):
    num_images = camera_centers.shape[0]
    rankings_all = []
    for i in range(num_images):
        distances = []
        for j in range(num_images):
            if i == j:
                distances.append([0.0, 0.0, 0.0])
                continue
            center_angle = _angle_between(camera_centers[i], camera_centers[j])
            view_angle = _angle_between(view_dirs[i], view_dirs[j])
            score = center_angle + view_dir_weight * view_angle
            distances.append([score, center_angle, view_angle])
        sorted_indices = np.argsort(np.asarray(distances)[:, 0])
        rankings = [[int(np.where(sorted_indices == j)[0][0])] + distances[j] for j in range(num_images)]
        rankings_all.append(rankings)
    return np.asarray(rankings_all, dtype=np.float64)


def _ordering_sim1_thresholding_sim2(
    reference_index: int,
    inverse_threshold: float,
    similarity_1_rankings: np.ndarray,
    similarity_2_rankings: np.ndarray | None,
    similarity_1_type: str,
):
    num_images = similarity_1_rankings.shape[0]
    selected = [reference_index]
    current_index = reference_index
    sim1_copy = similarity_1_rankings.copy()

    while len(selected) < num_images:
        sim1_copy[:, current_index, 0] = num_images
        candidate = None
        for i in range(num_images):
            test_idx = int(np.argsort(sim1_copy[current_index][:, 0])[i])
            if test_idx not in selected:
                candidate = test_idx
                break
        if candidate is None:
            break

        if similarity_2_rankings is not None:
            if similarity_1_type == "feature":
                difference = similarity_2_rankings[current_index][candidate, 1]
            else:
                difference = similarity_2_rankings[current_index][candidate, 0]
            if difference > inverse_threshold:
                break

        selected.append(candidate)
        current_index = candidate
    return selected


def _parse_thresholds(spec: str, fallback_last: int) -> list[float]:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        values = []
    values.append(float(fallback_last))
    return values


def _build_seqmat_pose_als_groups(
    image_paths: Sequence[str],
    sparse_dir: str,
    thresholds_spec: str,
    max_group_len: int,
    min_group_len: int,
    view_dir_weight: float,
):
    camera_centers, view_dirs = _load_colmap_pose_info(image_paths, sparse_dir)
    pose_rankings = _compute_pose_rankings(camera_centers, view_dirs, view_dir_weight)
    feature_rankings = _compute_orb_feature_rankings(image_paths)
    thresholds = _parse_thresholds(thresholds_spec, fallback_last=len(image_paths))

    groups = []
    created = set()
    last_threshold = thresholds[-1]

    for threshold in thresholds:
        if len(created) == len(image_paths):
            break
        for ref_idx in range(len(image_paths)):
            group_indices = _ordering_sim1_thresholding_sim2(
                reference_index=ref_idx,
                inverse_threshold=threshold,
                similarity_1_rankings=pose_rankings,
                similarity_2_rankings=feature_rankings,
                similarity_1_type="pose",
            )
            if len(group_indices) < min_group_len and threshold != last_threshold:
                continue
            if len(group_indices) > max_group_len:
                group_indices = group_indices[:max_group_len]

            save_local_indices = []
            for local_idx, global_idx in enumerate(group_indices):
                if threshold != last_threshold and (local_idx == 0 or local_idx == len(group_indices) - 1):
                    continue
                if global_idx not in created:
                    save_local_indices.append(local_idx)
            if not save_local_indices:
                continue

            for local_idx in save_local_indices:
                created.add(group_indices[local_idx])

            groups.append(
                {
                    "reference_index": ref_idx,
                    "threshold": threshold,
                    "indices": group_indices,
                    "save_local_indices": save_local_indices,
                }
            )

    missing = [idx for idx in range(len(image_paths)) if idx not in created]
    if missing:
        for idx in missing:
            groups.append(
                {
                    "reference_index": idx,
                    "threshold": float(len(image_paths)),
                    "indices": [idx],
                    "save_local_indices": [0],
                }
            )
    return groups


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cuda"):
    arr = np.asarray(img, np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(arr).copy()).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
    return t.to(dtype)


def compute_scaled_and_target_dims(
    w0: int,
    h0: int,
    scale: float = 4.0,
    multiple: int = 128,
    align_mode: str = "floor",
):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if multiple <= 0:
        raise ValueError("multiple must be > 0")

    exact_w = int(round(w0 * scale))
    exact_h = int(round(h0 * scale))
    if align_mode == "floor":
        t_w = (exact_w // multiple) * multiple
        t_h = (exact_h // multiple) * multiple
        effective_scale = scale
    elif align_mode == "ceil":
        t_w = ((exact_w + multiple - 1) // multiple) * multiple
        t_h = ((exact_h + multiple - 1) // multiple) * multiple
        required_scale = max(t_w / float(w0), t_h / float(h0))
        effective_scale = max(scale, required_scale)
    else:
        raise ValueError(f"Unsupported align_mode: {align_mode}")

    s_w = int(round(w0 * effective_scale))
    s_h = int(round(h0 * effective_scale))
    if t_w == 0 or t_h == 0:
        raise ValueError(
            f"Scaled size too small ({s_w}x{s_h}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )
    return s_w, s_h, t_w, t_h, effective_scale


def upscale_then_center_crop(img: Image.Image, scale: float, t_w: int, t_h: int) -> Image.Image:
    w0, h0 = img.size
    s_w = int(round(w0 * scale))
    s_h = int(round(h0 * scale))
    if t_w > s_w or t_h > s_h:
        raise ValueError(
            f"Target crop ({t_w}x{t_h}) exceeds scaled size ({s_w}x{s_h}). "
            f"Increase scale."
        )
    up = img.resize((s_w, s_h), Image.BICUBIC)
    l = (s_w - t_w) // 2
    t = (s_h - t_h) // 2
    return up.crop((l, t, l + t_w, t + t_h))


def build_chunks(total: int, chunk_size: int, chunk_min: int):
    """
    Build balanced contiguous chunks:
      - prefer lengths within [chunk_min, chunk_size]
      - cover all frames exactly once, in order
    """
    if total <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_min <= 0:
        raise ValueError("chunk_min must be > 0")
    if chunk_min > chunk_size:
        raise ValueError("chunk_min must be <= chunk_size")

    # Minimum chunks required to satisfy max length.
    num_chunks = (total + chunk_size - 1) // chunk_size
    # If possible, reduce chunk count so that each chunk is not smaller than chunk_min.
    while num_chunks > 1 and (total // num_chunks) < chunk_min:
        # Do not reduce if it would break max chunk size.
        if (total + (num_chunks - 2)) // (num_chunks - 1) > chunk_size:
            break
        num_chunks -= 1

    base = total // num_chunks
    rem = total % num_chunks
    lengths = [base + (1 if i < rem else 0) for i in range(num_chunks)]
    chunks = []
    start = 0
    for ln in lengths:
        end = start + ln
        chunks.append((start, end))
        start = end
    return chunks


def smallest_8n1_geq(n: int) -> int:
    # Return smallest m >= n such that m = 8k + 1.
    if n <= 1:
        return 1
    return ((n + 6) // 8) * 8 + 1


def prepare_chunk_meta(chunk_paths: list[str], scale: float, align_multiple: int, align_mode: str):
    with Image.open(chunk_paths[0]) as img0:
        w0, h0 = img0.size
    s_w, s_h, t_w, t_h, effective_scale = compute_scaled_and_target_dims(
        w0, h0, scale=scale, multiple=align_multiple, align_mode=align_mode
    )
    needed_f = max(MIN_FLASHVSR_INPUT_FRAMES, smallest_8n1_geq(len(chunk_paths) + 4))
    pad_count = max(0, needed_f - len(chunk_paths))
    f_all = len(chunk_paths) + pad_count
    target_out_frames = f_all - 4
    return w0, h0, s_w, s_h, t_w, t_h, f_all, target_out_frames, pad_count, effective_scale


def build_axis_tiles(length: int, tile_size: int):
    if tile_size <= 0 or tile_size >= length:
        return [(0, length)]
    ranges = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        ranges.append((start, end))
        start = end
    return ranges


def save_video(frames, save_path, fps=30, quality=6):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()


def init_pipeline(repo_root: str, model_dir: str, model_type: str):
    # FlashVSR internals use a hard-coded relative prompt path:
    # ../../examples/WanVSR/prompt_tensor/posi_prompt.pth
    runtime_dir = os.path.join(repo_root, ".hbsr_runtime", "run")
    wan_dir = os.path.join(repo_root, "examples", "WanVSR")
    if not os.path.isdir(wan_dir):
        raise FileNotFoundError(f"WanVSR dir not found: {wan_dir}")
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
        from diffsynth import ModelManager, FlashVSRTinyPipeline
        pipeline_cls = FlashVSRTinyPipeline
    elif model_type == "tiny_long":
        from diffsynth import ModelManager, FlashVSRTinyLongPipeline
        pipeline_cls = FlashVSRTinyLongPipeline
    else:
        raise ValueError(f"Unsupported FlashVSR model_type for this tool: {model_type}")
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder

    print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models(["./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors"])
    pipe = pipeline_cls.from_model_manager(mm, device="cuda")

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1
    ).to("cuda", dtype=torch.bfloat16)
    lq_proj_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(lq_proj_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(lq_proj_path, map_location="cpu"), strict=True
        )
    pipe.denoising_model().LQ_proj_in.to("cuda")

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16 + 768)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)
    print(mis)

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def prepare_chunk_tensor(
    chunk_paths: list[str],
    scale: float,
    align_multiple: int,
    align_mode: str,
    dtype: torch.dtype,
    device: str,
    region: tuple[int, int, int, int] | None = None,  # x0, y0, x1, y1 on full target frame
    region_run_size: tuple[int, int] | None = None,  # run_w, run_h (each 128-aligned)
):
    with Image.open(chunk_paths[0]) as img0:
        w0, h0 = img0.size

    s_w, s_h, t_w, t_h, effective_scale = compute_scaled_and_target_dims(
        w0, h0, scale=scale, multiple=align_multiple, align_mode=align_mode
    )
    # FlashVSR emits (F - 4) frames for input length F (where F must be 8n+1).
    # We pad so that output length is always >= real chunk length.
    needed_f = max(MIN_FLASHVSR_INPUT_FRAMES, smallest_8n1_geq(len(chunk_paths) + 4))
    pad_count = max(0, needed_f - len(chunk_paths))
    padded = chunk_paths + [chunk_paths[-1]] * pad_count
    f_all = len(padded)
    target_out_frames = f_all - 4

    frames = []
    for p in padded:
        with Image.open(p).convert("RGB") as img:
            out = upscale_then_center_crop(img, scale=effective_scale, t_w=t_w, t_h=t_h)
            if region is not None:
                x0, y0, x1, y1 = region
                out = out.crop((x0, y0, x1, y1))

                if region_run_size is not None:
                    run_w, run_h = region_run_size
                    cur_w, cur_h = out.size
                    if run_w < cur_w or run_h < cur_h:
                        raise ValueError(
                            f"region_run_size too small: run={run_w}x{run_h}, tile={cur_w}x{cur_h}"
                        )
                    if run_w != cur_w or run_h != cur_h:
                        arr = np.asarray(out, np.uint8)
                        pad_w = run_w - cur_w
                        pad_h = run_h - cur_h
                        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                        out = Image.fromarray(arr)
        frames.append(pil_to_tensor_neg1_1(out, dtype, device))
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W

    return vid, t_h, t_w, f_all, target_out_frames, pad_count, (w0, h0, s_w, s_h, effective_scale)


def main():
    parser = argparse.ArgumentParser(
        description="Official-style FlashVSR chunked prior generation for image folders."
    )
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="tiny", choices=["tiny", "tiny_long"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--chunk_min", type=int, default=50)
    parser.add_argument("--spatial_tile", type=int, default=0, help="Square tile size on high-res target (0 disables).")
    parser.add_argument("--spatial_tile_w", type=int, default=0, help="Tile width on high-res target (overrides --spatial_tile for x-axis).")
    parser.add_argument("--spatial_tile_h", type=int, default=0, help="Tile height on high-res target (overrides --spatial_tile for y-axis).")
    parser.add_argument(
        "--spatial_overlap",
        type=int,
        default=128,
        help="Context overlap (pixels) around each tile when spatial tiling is enabled.",
    )
    parser.add_argument(
        "--view_group_mode",
        type=str,
        default="none",
        choices=["none", "seqmat_pose_als"],
        help="How to group multi-view inputs before FlashVSR. 'seqmat_pose_als' uses a lightweight SequenceMatters-style pose ordering + ALS coverage.",
    )
    parser.add_argument(
        "--colmap_sparse_dir",
        type=str,
        default="",
        help="Path to COLMAP sparse/0 directory when --view_group_mode=seqmat_pose_als.",
    )
    parser.add_argument("--view_group_max_len", type=int, default=8)
    parser.add_argument("--view_group_min_len", type=int, default=3)
    parser.add_argument(
        "--view_group_thresholds",
        type=str,
        default="30,50",
        help="SequenceMatters-style ALS thresholds. For pose ordering these thresholds gate feature-rank jumps.",
    )
    parser.add_argument(
        "--view_dir_weight",
        type=float,
        default=0.0,
        help="Optional extra weight on view-direction angle in pose ranking. 0.0 stays closest to SequenceMatters.",
    )
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--align_multiple", type=int, default=128)
    parser.add_argument("--align_mode", type=str, default="floor", choices=["floor", "ceil"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_match_exact_scale_size", action="store_true", default=False)
    parser.add_argument("--kv_ratio", type=float, default=3.0)
    parser.add_argument("--local_range", type=int, default=11)
    parser.add_argument("--sparse_ratio", type=float, default=2.0)
    parser.add_argument("--if_buffer", action="store_true", default=False)
    parser.add_argument("--save_chunk_video", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_root = os.path.abspath(os.path.expanduser(args.output_root))

    paths = list_images_natural(input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    stems = [Path(p).stem for p in paths]
    if args.view_group_mode == "seqmat_pose_als":
        sparse_dir = args.colmap_sparse_dir.strip()
        if not sparse_dir:
            candidate_sparse = os.path.join(os.path.dirname(input_dir), "sparse", "0")
            if os.path.isdir(candidate_sparse):
                sparse_dir = candidate_sparse
            else:
                raise ValueError(
                    "--colmap_sparse_dir is required when --view_group_mode=seqmat_pose_als"
                )
        sparse_dir = os.path.abspath(os.path.expanduser(sparse_dir))
        groups = _build_seqmat_pose_als_groups(
            image_paths=paths,
            sparse_dir=sparse_dir,
            thresholds_spec=args.view_group_thresholds,
            max_group_len=args.view_group_max_len,
            min_group_len=args.view_group_min_len,
            view_dir_weight=args.view_dir_weight,
        )
        print(
            f"[flashvsr-official] input={input_dir}, total_frames={len(paths)}, "
            f"group_mode={args.view_group_mode}, num_groups={len(groups)}, "
            f"max_group_len={args.view_group_max_len}"
        )
    else:
        groups = [
            {
                "reference_index": start,
                "threshold": None,
                "indices": list(range(start, end)),
                "save_local_indices": list(range(end - start)),
            }
            for start, end in build_chunks(len(paths), args.chunk_size, args.chunk_min)
        ]
        print(
            f"[flashvsr-official] input={input_dir}, total_frames={len(paths)}, "
            f"chunk_size={args.chunk_size}, num_chunks={len(groups)}"
        )

    priors_dir = os.path.join(output_root, "priors")
    chunk_video_dir = os.path.join(output_root, "chunk_videos")
    os.makedirs(priors_dir, exist_ok=True)
    os.makedirs(chunk_video_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    pipe = init_pipeline(repo_root, model_dir, args.model_type)

    total_written = 0
    seen_stems = set()
    for cid, group in enumerate(groups):
        group_indices = list(group["indices"])
        chunk_paths = [paths[idx] for idx in group_indices]
        chunk_stems = [stems[idx] for idx in group_indices]
        save_local_indices = list(group["save_local_indices"])
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        w0, h0, s_w, s_h, tw, th, f_all, target_out_frames, pad_count, effective_scale = prepare_chunk_meta(
            chunk_paths=chunk_paths,
            scale=args.scale,
            align_multiple=args.align_multiple,
            align_mode=args.align_mode,
        )
        print(
            f"[flashvsr-official] chunk {cid+1}/{len(groups)} "
            f"indices={group_indices[:8]}{'...' if len(group_indices) > 8 else ''} "
            f"orig={w0}x{h0} scaled={s_w}x{s_h} target={tw}x{th} "
            f"input_F={f_all} target_out={target_out_frames} pad={pad_count} "
            f"effective_scale={effective_scale:.5f} save={len(save_local_indices)}"
        )

        tile_w_cfg = args.spatial_tile_w if args.spatial_tile_w > 0 else args.spatial_tile
        tile_h_cfg = args.spatial_tile_h if args.spatial_tile_h > 0 else args.spatial_tile
        use_spatial_tile = tile_w_cfg > 0 and tile_h_cfg > 0 and (tw > tile_w_cfg or th > tile_h_cfg)
        if use_spatial_tile:
            x_tiles = build_axis_tiles(tw, tile_w_cfg)
            y_tiles = build_axis_tiles(th, tile_h_cfg)
            print(
                f"[flashvsr-official] chunk {cid+1} spatial-tiles: "
                f"grid={len(x_tiles)}x{len(y_tiles)} "
                f"tile={tile_w_cfg}x{tile_h_cfg} overlap={args.spatial_overlap}"
            )
            video_frames = [Image.new("RGB", (tw, th)) for _ in chunk_stems]
            for yi, (y0, y1) in enumerate(y_tiles):
                for xi, (x0, x1) in enumerate(x_tiles):
                    # Halo tiling:
                    # run on an expanded tile with overlap, then paste only core region.
                    rx0 = max(0, x0 - args.spatial_overlap)
                    ry0 = max(0, y0 - args.spatial_overlap)
                    rx1 = min(tw, x1 + args.spatial_overlap)
                    ry1 = min(th, y1 + args.spatial_overlap)
                    region_w = rx1 - rx0
                    region_h = ry1 - ry0
                    run_w = ((region_w + 127) // 128) * 128
                    run_h = ((region_h + 127) // 128) * 128
                    print(
                        f"[flashvsr-official] chunk {cid+1} tile ({yi+1},{xi+1}) "
                        f"core=[{x0}:{x1},{y0}:{y1}] ctx=[{rx0}:{rx1},{ry0}:{ry1}] "
                        f"run={run_w}x{run_h}"
                    )
                    lq_tile, _, _, _, _, _, _ = prepare_chunk_tensor(
                        chunk_paths=chunk_paths,
                        scale=args.scale,
                        align_multiple=args.align_multiple,
                        align_mode=args.align_mode,
                        dtype=dtype,
                        device=args.device,
                        region=(rx0, ry0, rx1, ry1),
                        region_run_size=(run_w, run_h),
                    )
                    tile_video = pipe(
                        prompt="",
                        negative_prompt="",
                        cfg_scale=1.0,
                        num_inference_steps=1,
                        seed=args.seed,
                        LQ_video=lq_tile,
                        num_frames=f_all,
                        height=run_h,
                        width=run_w,
                        is_full_block=False,
                        if_buffer=args.if_buffer,
                        topk_ratio=args.sparse_ratio * 768 * 1280 / (run_h * run_w),
                        kv_ratio=args.kv_ratio,
                        local_range=args.local_range,
                        color_fix=True,
                    )
                    tile_frames = tensor2video(tile_video)
                    if len(tile_frames) < len(chunk_stems):
                        raise RuntimeError(
                            f"Chunk {cid} tile output too short: got={len(tile_frames)} need={len(chunk_stems)}"
                        )
                    # Core region offset within the expanded-context tile.
                    ox0 = x0 - rx0
                    oy0 = y0 - ry0
                    ox1 = ox0 + (x1 - x0)
                    oy1 = oy0 + (y1 - y0)
                    for i in range(len(chunk_stems)):
                        # First remove pad-to-128 border; then keep only core.
                        tile_arr = np.asarray(tile_frames[i], dtype=np.uint8)[:region_h, :region_w, :3]
                        core_arr = tile_arr[oy0:oy1, ox0:ox1, :]
                        video_frames[i].paste(Image.fromarray(core_arr), (x0, y0))
                    del lq_tile, tile_video, tile_frames
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        else:
            lq, _, _, _, _, _, _ = prepare_chunk_tensor(
                chunk_paths=chunk_paths,
                scale=args.scale,
                align_multiple=args.align_multiple,
                align_mode=args.align_mode,
                dtype=dtype,
                device=args.device,
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
                if_buffer=args.if_buffer,
                topk_ratio=args.sparse_ratio * 768 * 1280 / (th * tw),
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                color_fix=True,
            )
            video_frames = tensor2video(video)
            if len(video_frames) < len(chunk_stems):
                raise RuntimeError(
                    f"Chunk {cid} output too short: got={len(video_frames)} need={len(chunk_stems)}"
                )

        # Scale check for this chunk: exact 4x vs original, and target used by FlashVSR.
        out_w, out_h = video_frames[0].size
        exact_4x_raw = (out_w == w0 * 4) and (out_h == h0 * 4)
        exact_w = int(round(w0 * args.scale))
        exact_h = int(round(h0 * args.scale))
        print(
            f"[flashvsr-official] chunk {cid+1} scale-check: "
            f"output_raw={out_w}x{out_h}, exact_x4_raw={exact_4x_raw}, "
            f"target_exact={exact_w}x{exact_h}, "
            f"target_used={tw}x{th}"
        )

        if args.save_chunk_video:
            chunk_video_path = os.path.join(
                chunk_video_dir,
                f"FlashVSR_v1.1_Tiny_chunk{cid:04d}_group{len(group_indices):03d}_seed{args.seed}.mp4",
            )
            save_video(video_frames, chunk_video_path, fps=args.fps, quality=6)

        for i in save_local_indices:
            stem = chunk_stems[i]
            if stem in seen_stems:
                raise RuntimeError(f"Duplicate stem write detected: {stem}")
            seen_stems.add(stem)
            out_img = video_frames[i]
            if (not args.no_match_exact_scale_size) and out_img.size != (exact_w, exact_h):
                out_img = out_img.resize((exact_w, exact_h), Image.BICUBIC)
            out_img.save(os.path.join(priors_dir, f"{stem}.png"))
        total_written += len(save_local_indices)

    expected_stems = set(stems)
    missing_stems = sorted(expected_stems - seen_stems)
    extra_stems = sorted(seen_stems - expected_stems)
    if missing_stems or extra_stems or total_written != len(stems):
        raise RuntimeError(
            "Input-output one-to-one mapping check failed: "
            f"expected={len(stems)} written={total_written} "
            f"missing={len(missing_stems)} extra={len(extra_stems)}"
        )

    print(
        "[flashvsr-official] done\n"
        f"  input_dir      : {input_dir}\n"
        f"  output_root    : {output_root}\n"
        f"  priors_dir     : {priors_dir}\n"
        f"  frames_written : {total_written}"
    )


if __name__ == "__main__":
    main()
