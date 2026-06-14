import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def normalize_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"\s*\(\d+\)$", "", stem)


def load_rgb_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def resize_like(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    pil = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    pil = pil.resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_gray(path: Path, value: np.ndarray):
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = value - float(value.min())
    vmax = float(value.max())
    if vmax > 1e-8:
        value = value / vmax
    Image.fromarray(np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def save_overlay(path: Path, base_image: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha: float = 0.65):
    base_u8 = np.clip(base_image * 255.0, 0.0, 255.0).astype(np.uint8)
    overlay = base_u8.astype(np.float32).copy()
    color = np.asarray(color, dtype=np.float32)
    active = mask > 0
    overlay[active] = (1.0 - alpha) * overlay[active] + alpha * color
    Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def save_heat_overlay(path: Path, base_image: np.ndarray, heat: np.ndarray):
    base_u8 = np.clip(base_image * 255.0, 0.0, 255.0).astype(np.uint8).astype(np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = heat - float(heat.min())
    vmax = float(heat.max())
    if vmax > 1e-8:
        heat = heat / vmax
    color = np.stack(
        [
            np.clip(255.0 * heat, 0.0, 255.0),
            np.clip((1.0 - np.abs(heat - 0.5) * 2.0) * 180.0, 0.0, 255.0),
            np.clip((1.0 - heat) * 80.0, 0.0, 255.0),
        ],
        axis=2,
    )
    alpha = 0.6 * heat[..., None]
    out = base_u8 * (1.0 - alpha) + color * alpha
    Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def summarize(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def find_subject_roi(
    reference_image: np.ndarray,
    seed_percentile: float,
    center_sigma: float,
    pad_ratio: float,
):
    height, width = reference_image.shape[:2]
    value_max = np.max(reference_image, axis=2)
    value_min = np.min(reference_image, axis=2)
    saturation = (value_max - value_min) / np.maximum(value_max, 1e-6)

    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    center_prior = np.exp(-(xs * xs + ys * ys) / max(float(center_sigma), 1e-6))
    seed_score = saturation * center_prior

    threshold = float(np.percentile(seed_score, float(seed_percentile)))
    seed_mask = ndi.binary_opening(seed_score > threshold, structure=np.ones((5, 5), dtype=bool))
    labels, num_labels = ndi.label(seed_mask)

    best_slice = None
    best_score = -1.0
    for label_id, component_slice in enumerate(ndi.find_objects(labels), start=1):
        if component_slice is None:
            continue
        component = labels[component_slice] == label_id
        area = int(component.sum())
        if area <= 0:
            continue
        y0, y1 = component_slice[0].start, component_slice[0].stop
        x0, x1 = component_slice[1].start, component_slice[1].stop
        touches_border = (y0 == 0) or (x0 == 0) or (y1 == height) or (x1 == width)
        cy = 0.5 * (float(y0 + y1) / float(height))
        cx = 0.5 * (float(x0 + x1) / float(width))
        center_bonus = 1.0 - ((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        score = float(area) * center_bonus * (0.4 if touches_border else 1.0)
        if score > best_score:
            best_score = score
            best_slice = (y0, y1, x0, x1)

    if best_slice is None:
        return np.ones((height, width), dtype=bool), [0, 0, int(width), int(height)]

    y0, y1, x0, x1 = best_slice
    pad_y = int((y1 - y0) * float(pad_ratio))
    pad_x = int((x1 - x0) * float(pad_ratio))
    y0 = max(0, y0 - pad_y)
    y1 = min(height, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(width, x1 + pad_x)

    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    return roi_mask, [int(x0), int(y0), int(x1), int(y1)]


def select_top_components(mask: np.ndarray, score: np.ndarray, min_area: int, max_keep: int):
    labels, _ = ndi.label(mask)
    filtered = np.zeros_like(mask, dtype=bool)
    component_entries = []
    for label_id, component_slice in enumerate(ndi.find_objects(labels), start=1):
        if component_slice is None:
            continue
        component = labels[component_slice] == label_id
        area = int(component.sum())
        if area < int(min_area):
            continue
        mean_score = float(score[component_slice][component].mean())
        rank_score = mean_score * np.sqrt(float(area))
        y0, y1 = component_slice[0].start, component_slice[0].stop
        x0, x1 = component_slice[1].start, component_slice[1].stop
        component_entries.append(
            {
                "rank_score": float(rank_score),
                "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "area": area,
                "mean_score": mean_score,
                "slice": component_slice,
                "label_id": int(label_id),
            }
        )
    component_entries.sort(key=lambda item: item["rank_score"], reverse=True)
    for entry in component_entries[: int(max_keep)]:
        component_slice = entry["slice"]
        filtered[component_slice] |= labels[component_slice] == int(entry["label_id"])
    for entry in component_entries:
        entry.pop("slice", None)
        entry.pop("label_id", None)
    return filtered, component_entries


def build_bad_region_score(
    reference_image: np.ndarray,
    candidate_image: np.ndarray,
    roi_mask: np.ndarray,
    lowfreq_threshold: float,
    edge_percentile: float,
    distance_cap: float,
    smooth_sigma: float,
    miss_weight: float,
    extra_weight: float,
    orientation_weight: float,
):
    ref_gray = rgb_to_gray(reference_image)
    cand_gray = rgb_to_gray(candidate_image)

    ref_low = ndi.uniform_filter(ref_gray, size=11, mode="reflect")
    cand_low = ndi.uniform_filter(cand_gray, size=11, mode="reflect")
    confidence = np.clip(1.0 - np.abs(ref_low - cand_low) / max(float(lowfreq_threshold), 1e-6), 0.0, 1.0)

    sx_ref = ndi.sobel(ref_gray, axis=1, mode="reflect")
    sy_ref = ndi.sobel(ref_gray, axis=0, mode="reflect")
    sx_cand = ndi.sobel(cand_gray, axis=1, mode="reflect")
    sy_cand = ndi.sobel(cand_gray, axis=0, mode="reflect")
    mag_ref = np.hypot(sx_ref, sy_ref)
    mag_cand = np.hypot(sx_cand, sy_cand)

    roi_values_ref = mag_ref[roi_mask]
    roi_values_cand = mag_cand[roi_mask]
    ref_thr = float(np.percentile(roi_values_ref, float(edge_percentile)))
    cand_thr = float(np.percentile(roi_values_cand, float(edge_percentile)))
    ref_hi = float(np.percentile(roi_values_ref, 99.5))
    cand_hi = float(np.percentile(roi_values_cand, 99.5))

    ref_edge = (mag_ref > ref_thr) & roi_mask
    cand_edge = (mag_cand > cand_thr) & roi_mask
    dist_to_cand = ndi.distance_transform_edt(~cand_edge)
    dist_to_ref = ndi.distance_transform_edt(~ref_edge)
    ref_strength = np.clip((mag_ref - ref_thr) / max(ref_hi - ref_thr, 1e-6), 0.0, 1.0)
    cand_strength = np.clip((mag_cand - cand_thr) / max(cand_hi - cand_thr, 1e-6), 0.0, 1.0)

    miss = np.clip(dist_to_cand / max(float(distance_cap), 1e-6), 0.0, 1.0) * ref_strength * ref_edge
    extra = np.clip(dist_to_ref / max(float(distance_cap), 1e-6), 0.0, 1.0) * cand_strength * cand_edge

    angle_ref = np.arctan2(sy_ref, sx_ref)
    angle_cand = np.arctan2(sy_cand, sx_cand)
    orientation = np.abs(np.angle(np.exp(1j * (angle_ref - angle_cand)))) / np.pi
    shared_edges = ndi.binary_dilation(ref_edge, iterations=2) & ndi.binary_dilation(cand_edge, iterations=2)
    orientation = orientation * shared_edges

    raw_score = roi_mask * confidence * (
        float(miss_weight) * miss + float(extra_weight) * extra + float(orientation_weight) * orientation
    )
    score = ndi.gaussian_filter(raw_score, sigma=float(smooth_sigma))
    return {
        "score": score,
        "confidence": confidence,
        "miss": miss,
        "extra": extra,
        "orientation": orientation,
    }


def choose_candidate_map(input_dir: Path, ignore_suffix: str, skip_prefix: str):
    candidate_map = {}
    candidate_files = sorted(input_dir.glob("*.png"))
    for path in candidate_files:
        stem = path.stem
        if ignore_suffix and stem.endswith(ignore_suffix):
            continue
        if skip_prefix and stem.startswith(skip_prefix):
            continue
        key = normalize_stem(path.name)
        existing = candidate_map.get(key)
        if existing is None or len(path.name) < len(existing.name):
            candidate_map[key] = path
    return candidate_map


def main():
    parser = ArgumentParser(description="Lightweight 2D mesh-bad-region proxy from paired reference/candidate images.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ignore_candidate_suffix", type=str, default="_inject")
    parser.add_argument("--skip_candidate_prefix", type=str, default="000")
    parser.add_argument("--roi_seed_percentile", type=float, default=93.0)
    parser.add_argument("--roi_center_sigma", type=float, default=0.55)
    parser.add_argument("--roi_pad_ratio", type=float, default=0.35)
    parser.add_argument("--lowfreq_threshold", type=float, default=0.10)
    parser.add_argument("--edge_percentile", type=float, default=88.0)
    parser.add_argument("--distance_cap", type=float, default=8.0)
    parser.add_argument("--score_percentile", type=float, default=92.0)
    parser.add_argument("--smooth_sigma", type=float, default=2.0)
    parser.add_argument("--min_component_area", type=int, default=80)
    parser.add_argument("--max_components", type=int, default=20)
    parser.add_argument("--summary_topk", type=int, default=5)
    parser.add_argument("--miss_weight", type=float, default=0.75)
    parser.add_argument("--extra_weight", type=float, default=0.20)
    parser.add_argument("--orientation_weight", type=float, default=0.35)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_paths = sorted(list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.JPEG")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    candidate_map = choose_candidate_map(
        input_dir=input_dir,
        ignore_suffix=str(args.ignore_candidate_suffix),
        skip_prefix=str(args.skip_candidate_prefix),
    )

    if not reference_paths:
        raise RuntimeError(f"No reference JPG/JPEG images found under: {input_dir}")

    summaries = []
    missing_candidates = []
    for reference_path in reference_paths:
        image_name = normalize_stem(reference_path.name)
        candidate_path = candidate_map.get(image_name)
        if candidate_path is None:
            missing_candidates.append(image_name)
            continue

        reference_image = load_rgb_np(reference_path)
        candidate_image = resize_like(load_rgb_np(candidate_path), reference_image.shape[0], reference_image.shape[1])
        roi_mask, roi_xyxy = find_subject_roi(
            reference_image=reference_image,
            seed_percentile=float(args.roi_seed_percentile),
            center_sigma=float(args.roi_center_sigma),
            pad_ratio=float(args.roi_pad_ratio),
        )
        maps = build_bad_region_score(
            reference_image=reference_image,
            candidate_image=candidate_image,
            roi_mask=roi_mask,
            lowfreq_threshold=float(args.lowfreq_threshold),
            edge_percentile=float(args.edge_percentile),
            distance_cap=float(args.distance_cap),
            smooth_sigma=float(args.smooth_sigma),
            miss_weight=float(args.miss_weight),
            extra_weight=float(args.extra_weight),
            orientation_weight=float(args.orientation_weight),
        )

        active_scores = maps["score"][roi_mask]
        threshold = float(np.percentile(active_scores, float(args.score_percentile)))
        mask = (maps["score"] > threshold) & roi_mask
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = ndi.binary_dilation(mask, iterations=2)
        mask, components = select_top_components(
            mask=mask,
            score=maps["score"],
            min_area=int(args.min_component_area),
            max_keep=int(args.max_components),
        )

        view_dir = output_dir / image_name
        view_dir.mkdir(parents=True, exist_ok=True)
        save_gray(view_dir / "score.png", maps["score"])
        save_gray(view_dir / "confidence.png", maps["confidence"])
        save_gray(view_dir / "miss.png", maps["miss"])
        save_gray(view_dir / "extra.png", maps["extra"])
        save_gray(view_dir / "orientation.png", maps["orientation"])
        save_mask(view_dir / "mask.png", mask)
        save_overlay(view_dir / "mask_overlay.png", reference_image, mask)
        save_heat_overlay(view_dir / "heat_overlay.png", reference_image, maps["score"] * roi_mask)

        roi_overlay = reference_image.copy()
        x0, y0, x1, y1 = roi_xyxy
        roi_overlay[:y0, :, :] *= 0.35
        roi_overlay[y1:, :, :] *= 0.35
        roi_overlay[:, :x0, :] *= 0.35
        roi_overlay[:, x1:, :] *= 0.35
        Image.fromarray(np.clip(roi_overlay * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(view_dir / "roi_overlay.png")

        summaries.append(
            {
                "image_name": image_name,
                "reference_path": str(reference_path.resolve()),
                "candidate_path": str(candidate_path.resolve()),
                "roi_xyxy": roi_xyxy,
                "mask_pixels": int(mask.sum()),
                "score_stats": summarize(active_scores),
                "top_components": components[: int(args.summary_topk)],
            }
        )

    summary = {
        "mode": "diagnose_2d_mesh_bad_regions_v0",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "view_count": int(len(summaries)),
        "missing_candidates": missing_candidates,
        "parameters": {
            "ignore_candidate_suffix": str(args.ignore_candidate_suffix),
            "skip_candidate_prefix": str(args.skip_candidate_prefix),
            "roi_seed_percentile": float(args.roi_seed_percentile),
            "roi_center_sigma": float(args.roi_center_sigma),
            "roi_pad_ratio": float(args.roi_pad_ratio),
            "lowfreq_threshold": float(args.lowfreq_threshold),
            "edge_percentile": float(args.edge_percentile),
            "distance_cap": float(args.distance_cap),
            "score_percentile": float(args.score_percentile),
            "smooth_sigma": float(args.smooth_sigma),
            "min_component_area": int(args.min_component_area),
            "max_components": int(args.max_components),
            "summary_topk": int(args.summary_topk),
            "miss_weight": float(args.miss_weight),
            "extra_weight": float(args.extra_weight),
            "orientation_weight": float(args.orientation_weight),
        },
        "views": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
