#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image


def _progress(iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        print(f'[prior-analysis] {desc}...')
        return iterable


def _load_flashvsr_module():
    tool_path = Path(__file__).resolve().parent / 'generate_flashvsr_priors_official_chunked.py'
    spec = importlib.util.spec_from_file_location('_flashvsr_prior_tool', tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to import FlashVSR prior tool from {tool_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path).convert('RGB') as img:
        return _to_float01(np.asarray(img))


def _resize(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = img.resize((w, h), Image.BICUBIC)
    return _to_float01(np.asarray(img))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    try:
        import cv2
    except Exception:
        return float('nan')
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return float(np.mean(ssim_map))


def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    vals = [_ssim_channel(a[..., c], b[..., c]) for c in range(3)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _bucket_relative(pos01: float) -> str:
    if pos01 <= 0.2:
        return 'head'
    if pos01 >= 0.8:
        return 'tail'
    return 'middle'


def _bucket_global(index: int, total: int) -> str:
    if total <= 1:
        return 'single'
    t = index / max(total - 1, 1)
    if t < 0.25:
        return 'q1'
    if t < 0.5:
        return 'q2'
    if t < 0.75:
        return 'q3'
    return 'q4'


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(mean(vals)) if vals else float('nan')


def _bucket_abs_offset(offset: int) -> str:
    if offset == 0:
        return 'exact'
    if offset <= 1:
        return 'near1'
    if offset <= 3:
        return 'near3'
    if offset <= 8:
        return 'near8'
    return 'far'


def _summarize_group(rows: list[dict[str, Any]], key: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    summary = []
    for group_key, items in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        summary.append({
            key: group_key,
            'count': len(items),
            'gt_psnr': _safe_mean([it['gt_psnr'] for it in items]),
            'gt_ssim': _safe_mean([it['gt_ssim'] for it in items]),
            'delta_psnr_over_bicubic': _safe_mean([it['delta_psnr_over_bicubic'] for it in items]),
            'global_best_psnr': _safe_mean([it.get('global_best_psnr', float('nan')) for it in items]),
            'global_best_delta_over_exact': _safe_mean([it.get('global_best_delta_over_exact', float('nan')) for it in items]),
            'global_best_abs_offset': _safe_mean([it.get('global_best_abs_offset', float('nan')) for it in items]),
            'exact_global_best_rate': _safe_mean([float(bool(it.get('global_best_is_exact', False))) for it in items]),
            'input_consistency_psnr': _safe_mean([it['input_consistency_psnr'] for it in items]),
        })
    return summary


def _write_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Analyze FlashVSR prior quality against LR input and GT, grouped by sequence position.')
    parser.add_argument('--prior_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--view_group_mode', type=str, default='none', choices=['none', 'seqmat_pose_als'])
    parser.add_argument('--colmap_sparse_dir', type=str, default='')
    parser.add_argument('--view_group_max_len', type=int, default=6)
    parser.add_argument('--view_group_min_len', type=int, default=3)
    parser.add_argument('--view_group_thresholds', type=str, default='30,50')
    parser.add_argument('--view_dir_weight', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=12)
    parser.add_argument('--global_search_size', type=int, default=192)
    parser.add_argument('--global_search_topk', type=int, default=5)
    args = parser.parse_args()

    prior_dir = os.path.abspath(os.path.expanduser(args.prior_dir))
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    gt_dir = os.path.abspath(os.path.expanduser(args.gt_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    print('[prior-analysis] importing FlashVSR helper module...')
    flashvsr_tool = _load_flashvsr_module()

    print('[prior-analysis] indexing input / GT / prior folders...')
    input_paths = flashvsr_tool.list_images_natural(input_dir)
    gt_paths = flashvsr_tool.list_images_natural(gt_dir)
    prior_paths = flashvsr_tool.list_images_natural(prior_dir)
    if not input_paths:
        raise FileNotFoundError(f'No input images found in {input_dir}')
    if not gt_paths:
        raise FileNotFoundError(f'No GT images found in {gt_dir}')
    if not prior_paths:
        raise FileNotFoundError(f'No prior images found in {prior_dir}')

    input_by_stem = {Path(p).stem: p for p in input_paths}
    gt_by_stem = {Path(p).stem: p for p in gt_paths}
    prior_by_stem = {Path(p).stem: p for p in prior_paths}
    common_stems = [Path(p).stem for p in input_paths if Path(p).stem in gt_by_stem and Path(p).stem in prior_by_stem]
    if not common_stems:
        raise RuntimeError('No common stems across input/prior/gt')
    global_index_by_stem = {Path(p).stem: idx for idx, p in enumerate(input_paths)}

    thumb_hw = (args.global_search_size, args.global_search_size)
    gt_thumb_items = []
    for gt_path in _progress(gt_paths, 'building GT thumbnails', total=len(gt_paths)):
        stem = Path(gt_path).stem
        gt_img = _load_rgb(gt_path)
        gt_thumb = _resize(gt_img, thumb_hw)
        gt_thumb_items.append({
            'stem': stem,
            'global_index': global_index_by_stem.get(stem, -1),
            'thumb': gt_thumb,
            'path': gt_path,
        })

    stem_to_meta = {}
    if args.view_group_mode == 'seqmat_pose_als':
        print('[prior-analysis] rebuilding SequenceMatters-style groups...')
        groups = flashvsr_tool._build_seqmat_pose_als_groups(
            image_paths=input_paths,
            sparse_dir=args.colmap_sparse_dir,
            thresholds_spec=args.view_group_thresholds,
            max_group_len=args.view_group_max_len,
            min_group_len=args.view_group_min_len,
            view_dir_weight=args.view_dir_weight,
        )
        all_stems = [Path(p).stem for p in input_paths]
        for gid, group in enumerate(groups):
            idxs = list(group['indices'])
            save_local = list(group['save_local_indices'])
            for save_rank, local_idx in enumerate(save_local):
                global_idx = idxs[local_idx]
                stem = all_stems[global_idx]
                rel_pos = local_idx / max(len(idxs) - 1, 1)
                stem_to_meta[stem] = {
                    'group_id': gid,
                    'group_size': len(idxs),
                    'group_threshold': group['threshold'],
                    'reference_index': group['reference_index'],
                    'local_index': local_idx,
                    'local_pos01': rel_pos,
                    'local_bucket': _bucket_relative(rel_pos),
                    'save_rank': save_rank,
                    'save_count': len(save_local),
                    'global_index': global_idx,
                    'global_bucket': _bucket_global(global_idx, len(input_paths)),
                }
    else:
        for idx, p in enumerate(input_paths):
            stem = Path(p).stem
            stem_to_meta[stem] = {
                'group_id': idx,
                'group_size': 1,
                'group_threshold': None,
                'reference_index': idx,
                'local_index': 0,
                'local_pos01': 0.0,
                'local_bucket': 'single',
                'save_rank': 0,
                'save_count': 1,
                'global_index': idx,
                'global_bucket': _bucket_global(idx, len(input_paths)),
            }

    rows = []
    print('[prior-analysis] comparing prior against exact GT and global-best GT...')
    for stem in _progress(common_stems, 'analyzing frames', total=len(common_stems)):
        lr = _load_rgb(input_by_stem[stem])
        gt = _load_rgb(gt_by_stem[stem])
        prior = _load_rgb(prior_by_stem[stem])
        if prior.shape[:2] != gt.shape[:2]:
            prior = _resize(prior, gt.shape[:2])
        bicubic = _resize(lr, gt.shape[:2])
        prior_down = _resize(prior, lr.shape[:2])
        prior_thumb = _resize(prior, thumb_hw)
        exact_psnr = _psnr(prior, gt)
        exact_ssim = _ssim_rgb(prior, gt)
        exact_mae = _mae(prior, gt)
        bicubic_psnr = _psnr(bicubic, gt)
        bicubic_ssim = _ssim_rgb(bicubic, gt)

        thumb_ranked = sorted(
            gt_thumb_items,
            key=lambda item: _mse(prior_thumb, item['thumb']),
        )[: max(1, args.global_search_topk)]

        best_global = None
        for cand in thumb_ranked:
            cand_gt = _load_rgb(cand['path'])
            if cand_gt.shape[:2] != prior.shape[:2]:
                cand_gt = _resize(cand_gt, prior.shape[:2])
            cand_psnr = _psnr(prior, cand_gt)
            cand_rec = {
                'stem': cand['stem'],
                'global_index': cand['global_index'],
                'psnr': cand_psnr,
                'gt': cand_gt,
            }
            if best_global is None or cand_rec['psnr'] > best_global['psnr']:
                best_global = cand_rec
        if best_global is None:
            raise RuntimeError(f'Failed to find global best GT match for {stem}')
        best_global['ssim'] = _ssim_rgb(prior, best_global['gt'])
        best_global['mae'] = _mae(prior, best_global['gt'])
        del best_global['gt']

        exact_global_idx = global_index_by_stem.get(stem, -1)
        global_best_offset = (
            best_global['global_index'] - exact_global_idx
            if best_global['global_index'] >= 0 and exact_global_idx >= 0
            else 0
        )
        global_best_abs_offset = abs(global_best_offset)

        row = {
            'stem': stem,
            'input_h': lr.shape[0],
            'input_w': lr.shape[1],
            'gt_h': gt.shape[0],
            'gt_w': gt.shape[1],
            'gt_psnr': exact_psnr,
            'gt_ssim': exact_ssim,
            'gt_mae': exact_mae,
            'bicubic_psnr': bicubic_psnr,
            'bicubic_ssim': bicubic_ssim,
            'delta_psnr_over_bicubic': exact_psnr - bicubic_psnr,
            'delta_ssim_over_bicubic': exact_ssim - bicubic_ssim,
            'input_consistency_psnr': _psnr(prior_down, lr),
            'input_consistency_mae': _mae(prior_down, lr),
            'global_best_stem': best_global['stem'],
            'global_best_index': best_global['global_index'],
            'global_best_psnr': best_global['psnr'],
            'global_best_ssim': best_global['ssim'],
            'global_best_mae': best_global['mae'],
            'global_best_delta_over_exact': best_global['psnr'] - exact_psnr,
            'global_best_offset': global_best_offset,
            'global_best_abs_offset': global_best_abs_offset,
            'global_best_offset_bucket': _bucket_abs_offset(global_best_abs_offset),
            'global_best_is_exact': best_global['stem'] == stem,
        }
        row.update(stem_to_meta.get(stem, {}))
        rows.append(row)

    rows_by_psnr = sorted(rows, key=lambda r: r['gt_psnr'], reverse=True)
    rows_by_delta = sorted(rows, key=lambda r: r['delta_psnr_over_bicubic'], reverse=True)

    overall = {
        'count': len(rows),
        'gt_psnr_mean': _safe_mean([r['gt_psnr'] for r in rows]),
        'gt_ssim_mean': _safe_mean([r['gt_ssim'] for r in rows]),
        'delta_psnr_over_bicubic_mean': _safe_mean([r['delta_psnr_over_bicubic'] for r in rows]),
        'global_best_psnr_mean': _safe_mean([r['global_best_psnr'] for r in rows]),
        'global_best_delta_over_exact_mean': _safe_mean([r['global_best_delta_over_exact'] for r in rows]),
        'global_best_abs_offset_mean': _safe_mean([r['global_best_abs_offset'] for r in rows]),
        'global_best_exact_rate': _safe_mean([float(r['global_best_is_exact']) for r in rows]),
        'input_consistency_psnr_mean': _safe_mean([r['input_consistency_psnr'] for r in rows]),
    }

    summary = {
        'overall': overall,
        'by_local_bucket': _summarize_group(rows, 'local_bucket'),
        'by_global_bucket': _summarize_group(rows, 'global_bucket'),
        'by_group_size': _summarize_group(rows, 'group_size'),
        'by_offset_bucket': _summarize_group(rows, 'global_best_offset_bucket'),
        'top_psnr_frames': rows_by_psnr[: args.top_k],
        'bottom_psnr_frames': list(reversed(rows_by_psnr[-args.top_k:])),
        'top_delta_frames': rows_by_delta[: args.top_k],
        'bottom_delta_frames': list(reversed(rows_by_delta[-args.top_k:])),
    }

    print('[prior-analysis] writing analysis outputs...')
    _write_csv(os.path.join(output_dir, 'per_frame_metrics.csv'), rows)
    for key in ['by_local_bucket', 'by_global_bucket', 'by_group_size', 'by_offset_bucket']:
        _write_csv(os.path.join(output_dir, f'{key}.csv'), summary[key])
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('[prior-analysis] done')
    print(f'  prior_dir   : {prior_dir}')
    print(f'  input_dir   : {input_dir}')
    print(f'  gt_dir      : {gt_dir}')
    print(f'  output_dir  : {output_dir}')
    print(f"  frames      : {len(rows)}")
    print(f"  mean PSNR   : {overall['gt_psnr_mean']:.4f}")
    print(f"  mean SSIM   : {overall['gt_ssim_mean']:.4f}")
    print(f"  mean dPSNR  : {overall['delta_psnr_over_bicubic_mean']:.4f}")
    print(f"  mean gPSNR  : {overall['global_best_psnr_mean']:.4f}")
    print(f"  mean gΔexact: {overall['global_best_delta_over_exact_mean']:.4f}")
    print(f"  exact-best  : {overall['global_best_exact_rate']:.4f}")
    print(f"  mean |gΔidx|: {overall['global_best_abs_offset_mean']:.4f}")
    print(f"  mean in-PSNR: {overall['input_consistency_psnr_mean']:.4f}")
    if summary['by_local_bucket']:
        best_local = max(summary['by_local_bucket'], key=lambda x: x['delta_psnr_over_bicubic'])
        worst_local = min(summary['by_local_bucket'], key=lambda x: x['delta_psnr_over_bicubic'])
        print(
            f"  best local bucket  : {best_local['local_bucket']} (dPSNR={best_local['delta_psnr_over_bicubic']:.4f})"
        )
        print(
            f"  worst local bucket : {worst_local['local_bucket']} (dPSNR={worst_local['delta_psnr_over_bicubic']:.4f})"
        )


if __name__ == '__main__':
    main()
