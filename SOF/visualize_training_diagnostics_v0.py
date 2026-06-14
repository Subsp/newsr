import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


def collect_snapshot_dirs(root: Path):
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("iter_")])


def make_contact_sheet(image_paths, output_path: Path, columns: int = 2):
    existing = [Path(path) for path in image_paths if Path(path).exists()]
    if not existing:
        return
    images = [Image.open(path).convert("RGB") for path in existing]
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    columns = max(1, int(columns))
    rows = (len(images) + columns - 1) // columns
    cell_w = max(widths)
    cell_h = max(heights)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (24, 24, 24))
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        sheet.paste(image, (col * cell_w, row * cell_h))
    sheet.save(output_path)


def main():
    parser = ArgumentParser(description="Build a quick aggregate view over exported training diagnostics snapshots.")
    parser.add_argument("--diagnostic_dir", type=str, required=True, help="Directory created by --diagnostic_output_subdir.")
    parser.add_argument("--metric_image", type=str, default="contact_sheet.png", help="Image filename to collect from each snapshot directory.")
    parser.add_argument("--latest_n", type=int, default=12, help="Keep only the latest N snapshot directories (<=0 means all).")
    args = parser.parse_args()

    root = Path(args.diagnostic_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Diagnostic directory does not exist: {root}")

    snapshot_dirs = collect_snapshot_dirs(root)
    if not snapshot_dirs:
        raise RuntimeError(f"No iter_* snapshot directories found under {root}")

    if int(args.latest_n) > 0:
        snapshot_dirs = snapshot_dirs[-int(args.latest_n):]

    metric_name = str(args.metric_image)
    metric_paths = []
    manifest = []
    for snapshot_dir in snapshot_dirs:
        metric_path = snapshot_dir / metric_name
        summary_path = snapshot_dir / "summary.json"
        summary = None
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        manifest.append(
            {
                "snapshot_dir": str(snapshot_dir),
                "metric_path": str(metric_path) if metric_path.exists() else None,
                "summary": summary,
            }
        )
        if metric_path.exists():
            metric_paths.append(metric_path)

    aggregate_dir = root / "aggregate_view_v0"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    with open(aggregate_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if metric_paths:
        make_contact_sheet(metric_paths, aggregate_dir / f"contact_sheet_{Path(metric_name).stem}.png", columns=2)

    print(f"[diagnostic-view] wrote aggregate view to {aggregate_dir}")


if __name__ == "__main__":
    main()
