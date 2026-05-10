"""Adapt University-1652 into the repository's pre/after image layout."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


SATELLITE_DIRS = [
    ("train", "satellite", Path("train/satellite")),
    ("test", "query_satellite", Path("test/query_satellite")),
    ("test", "gallery_satellite", Path("test/gallery_satellite")),
]

UAV_DIRS = [
    ("train", "drone", Path("train/drone")),
    ("test", "query_drone", Path("test/query_drone")),
    ("test", "gallery_drone", Path("test/gallery_drone")),
    ("test", "4K_drone", Path("test/4K_drone")),
]


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def make_output_name(split: str, view: str, class_id: str, src: Path) -> str:
    stem = src.stem.replace(" ", "_")
    suffix = src.suffix.lower()
    return f"{split}_{view}_{class_id}_{stem}{suffix}"


def copy_group(source_root: Path, output_root: Path, groups, output_subdir: str):
    items = []
    counters = Counter()
    target_dir = output_root / output_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    for split, view, rel_dir in groups:
        src_dir = source_root / rel_dir
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing expected directory: {src_dir}")
        for src in iter_images(src_dir):
            rel = src.relative_to(src_dir)
            class_id = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            name = make_output_name(split, view, class_id, src)
            dst = target_dir / name
            if dst.exists():
                raise FileExistsError(f"Output collision: {dst}")
            shutil.copy2(src, dst)
            item = {
                "image": f"{output_subdir}/{name}",
                "source": str(src),
                "source_relative": str(src.relative_to(source_root)),
                "split": split,
                "view": view,
                "class_id": class_id,
            }
            items.append(item)
            counters[f"{split}/{view}"] += 1
    return items, counters


def make_item_id(prefix: str, image_path: str) -> str:
    return f"{prefix}_{Path(image_path).stem}"


def write_cvgl_manifests(output_root: Path, pre_items, after_items):
    tiles = []
    class_to_tiles = {}
    for item in pre_items:
        tile_id = make_item_id("tile", item["image"])
        tiles.append({
            "tile_id": tile_id,
            "image": item["image"],
            "class_id": item["class_id"],
            "split": item["split"],
            "view": item["view"],
            "source_relative": item["source_relative"],
        })
        class_to_tiles.setdefault(item["class_id"], []).append(tile_id)

    queries = []
    for item in after_items:
        positives = class_to_tiles.get(item["class_id"], [])
        query = {
            "query_id": make_item_id("query", item["image"]),
            "image": item["image"],
            "class_id": item["class_id"],
            "split": item["split"],
            "view": item["view"],
            "source_relative": item["source_relative"],
        }
        if positives:
            query["gt_tile_id"] = positives[0]
            query["positives"] = positives
        queries.append(query)

    with (output_root / "tiles.json").open("w", encoding="utf-8") as f:
        json.dump(tiles, f, indent=2)
        f.write("\n")
    with (output_root / "queries.json").open("w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
        f.write("\n")

    return {
        "tiles_total": len(tiles),
        "queries_total": len(queries),
        "queries_with_positive": sum(1 for query in queries if "gt_tile_id" in query),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="/data/zhanghaofei/xry/dataset/University-1652",
        help="Path to the original University-1652 dataset.",
    )
    parser.add_argument(
        "--output-root",
        default="data/University-1652",
        help="Destination directory under this repository.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before copying.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Refresh tiles.json and queries.json from an existing manifest.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if args.manifest_only:
        manifest_path = output_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        cvgl_summary = write_cvgl_manifests(
            output_root, manifest["pre"], manifest["after"]
        )
        summary_path = output_root / "conversion_summary.json"
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            summary.update(cvgl_summary)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                f.write("\n")
        print(json.dumps(cvgl_summary, indent=2))
        return
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    pre_items, pre_counts = copy_group(
        source_root, output_root, SATELLITE_DIRS, "pre"
    )
    after_items, after_counts = copy_group(
        source_root, output_root, UAV_DIRS, "after"
    )

    manifest = {
        "dataset": "University-1652",
        "source_root": str(source_root),
        "layout": {
            "pre": "satellite-view images",
            "after": "UAV/drone-view images",
        },
        "excluded_views": ["street", "google"],
        "pre": pre_items,
        "after": after_items,
    }
    summary = {
        "dataset": "University-1652",
        "source_root": str(source_root),
        "output_root": str(output_root.resolve()),
        "pre_total": len(pre_items),
        "after_total": len(after_items),
        "pre_counts": dict(sorted(pre_counts.items())),
        "after_counts": dict(sorted(after_counts.items())),
        "excluded_views": ["street", "google"],
    }
    summary.update(write_cvgl_manifests(output_root, pre_items, after_items))

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with (output_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
