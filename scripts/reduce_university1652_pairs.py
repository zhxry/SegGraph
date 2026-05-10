"""Reduce adapted University-1652 to one UAV positive per satellite image."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


PREFERRED_UAV_VIEWS = {
    "satellite": ["drone"],
    "query_satellite": ["query_drone", "gallery_drone", "drone", "4K_drone"],
    "gallery_satellite": ["gallery_drone", "query_drone", "drone", "4K_drone"],
}


def make_item_id(prefix: str, image_path: str) -> str:
    return f"{prefix}_{Path(image_path).stem}"


def select_uav(pre_item, candidates_by_class, used_images):
    class_id = pre_item["class_id"]
    candidates = candidates_by_class[class_id]
    preferred_views = PREFERRED_UAV_VIEWS.get(pre_item["view"], [])

    for view in preferred_views:
        for item in candidates:
            if item["view"] == view and item["image"] not in used_images:
                return item
    for item in candidates:
        if item["image"] not in used_images:
            return item
    raise RuntimeError(f"No unused UAV positive left for {pre_item['image']}")


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/University-1652")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    manifest_path = dataset_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    pre_items = list(manifest["pre"])
    candidates_by_class = defaultdict(list)
    for item in sorted(manifest["after"], key=lambda x: (x["class_id"], x["split"], x["view"], x["image"])):
        candidates_by_class[item["class_id"]].append(item)

    missing = sorted({item["class_id"] for item in pre_items if item["class_id"] not in candidates_by_class})
    if missing:
        raise RuntimeError(f"Missing UAV positives for classes: {missing[:10]}")

    temp_after = dataset_root / "after_paired_tmp"
    if temp_after.exists():
        shutil.rmtree(temp_after)
    temp_after.mkdir(parents=True)

    selected_after = []
    pairs = []
    used_images = set()
    for idx, pre_item in enumerate(pre_items):
        uav_item = select_uav(pre_item, candidates_by_class, used_images)
        used_images.add(uav_item["image"])

        src = dataset_root / uav_item["image"]
        if not src.is_file():
            raise FileNotFoundError(src)
        dst_name = Path(uav_item["image"]).name
        dst = temp_after / dst_name
        shutil.copy2(src, dst)

        selected = dict(uav_item)
        selected["image"] = f"after/{dst_name}"
        selected["paired_pre_image"] = pre_item["image"]
        selected["pair_index"] = idx
        selected_after.append(selected)
        pairs.append({
            "pair_index": idx,
            "class_id": pre_item["class_id"],
            "pre": pre_item["image"],
            "after": selected["image"],
            "pre_view": pre_item["view"],
            "after_view": selected["view"],
        })

    if len(selected_after) != len(pre_items):
        raise RuntimeError("Pair count mismatch")

    old_after = dataset_root / "after"
    backup_after = dataset_root / "after_full_backup_tmp"
    if backup_after.exists():
        shutil.rmtree(backup_after)
    old_after.rename(backup_after)
    temp_after.rename(old_after)
    shutil.rmtree(backup_after)

    tiles = []
    queries = []
    for pre_item, after_item in zip(pre_items, selected_after):
        tile_id = make_item_id("tile", pre_item["image"])
        tiles.append({
            "tile_id": tile_id,
            "image": pre_item["image"],
            "class_id": pre_item["class_id"],
            "split": pre_item["split"],
            "view": pre_item["view"],
            "source_relative": pre_item["source_relative"],
        })
        queries.append({
            "query_id": make_item_id("query", after_item["image"]),
            "image": after_item["image"],
            "gt_tile_id": tile_id,
            "positives": [tile_id],
            "class_id": after_item["class_id"],
            "split": after_item["split"],
            "view": after_item["view"],
            "source_relative": after_item["source_relative"],
            "paired_pre_image": pre_item["image"],
        })

    manifest["after"] = selected_after
    manifest["pairing"] = {
        "policy": "one UAV positive per satellite image; prefer matching University-1652 split/view",
        "pairs": pairs,
    }
    write_json(manifest_path, manifest)
    write_json(dataset_root / "tiles.json", tiles)
    write_json(dataset_root / "queries.json", queries)

    summary = {
        "dataset": "University-1652",
        "output_root": str(dataset_root.resolve()),
        "pre_total": len(pre_items),
        "after_total": len(selected_after),
        "tiles_total": len(tiles),
        "queries_total": len(queries),
        "queries_with_positive": len(queries),
        "pairing_policy": manifest["pairing"]["policy"],
    }
    write_json(dataset_root / "conversion_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
