from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def iter_images(root: Path):
    return sorted([p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def save_absdiff(path: Path, diff: np.ndarray) -> None:
    vis = np.abs(diff)
    scale = float(np.percentile(vis, 99.0))
    vis = np.clip(vis / max(scale, 1e-6), 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two render folders without requiring GT copies.")
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_visuals", type=int, default=16)
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir).expanduser().resolve()
    cand_dir = Path(args.candidate_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    diff_dir = out_dir / "absdiff"
    diff_dir.mkdir(parents=True, exist_ok=True)

    ref_files = iter_images(ref_dir)
    cand_files = iter_images(cand_dir)
    cand_by_name = {p.name: p for p in cand_files}
    matched = [(p, cand_by_name[p.name]) for p in ref_files if p.name in cand_by_name]
    if not matched:
        raise FileNotFoundError(f"No matching image names between {ref_dir} and {cand_dir}")

    psnrs = []
    maes = []
    rmses = []
    per_image = []
    for idx, (ref_path, cand_path) in enumerate(matched):
        ref = load_rgb(ref_path)
        cand = load_rgb(cand_path)
        if cand.shape != ref.shape:
            cand_img = Image.fromarray((cand * 255.0).astype(np.uint8), "RGB").resize((ref.shape[1], ref.shape[0]), Image.Resampling.BICUBIC)
            cand = np.asarray(cand_img, dtype=np.float32) / 255.0
        diff = cand - ref
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(math.sqrt(mse))
        score = psnr_from_mse(mse)
        psnrs.append(score)
        maes.append(mae)
        rmses.append(rmse)
        per_image.append(
            {
                "name": ref_path.name,
                "psnr_like": score,
                "mae": mae,
                "rmse": rmse,
            }
        )
        if idx < int(args.max_visuals):
            save_absdiff(diff_dir / ref_path.name, diff)

    summary = {
        "reference_dir": str(ref_dir),
        "candidate_dir": str(cand_dir),
        "matched_images": len(matched),
        "psnr_like_mean": float(np.mean(psnrs)),
        "psnr_like_min": float(np.min(psnrs)),
        "psnr_like_median": float(np.median(psnrs)),
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "per_image": per_image,
        "absdiff_dir": str(diff_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
