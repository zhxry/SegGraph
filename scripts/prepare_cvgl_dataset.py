# Prepare a minimal CVGL dataset from aligned satellite/UAV canvases
"""
    Assumptions:
    - satellite image and UAV image describe the same 2D area
    - both images are already aligned in the same pixel coordinate system
    - query ground-truth location is the crop center in the UAV canvas

    Output structure:
    output_root/
    ├── tiles/
    ├── queries/
    ├── tiles.json
    └── queries.json
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARN: __file__ not found, trying local")
    dir_name = os.path.abspath("")
lib_path = os.path.realpath(f"{Path(dir_name).parent}")
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")

import numpy as np
from PIL import Image
import tyro


def _parse_hw(value: str) -> Tuple[int, int]:
    tokens = str(value).lower().replace("x", ",").split(",")
    tokens = [token.strip() for token in tokens if token.strip()]
    if len(tokens) != 2:
        raise ValueError(f"Invalid size string: {value}")
    return int(tokens[0]), int(tokens[1])


def _validate_or_infer_size(
    image: Image.Image,
    expected_hw: Tuple[int, int] = None,
    label: str = "image",
) -> Tuple[int, int]:
    actual_hw = (image.height, image.width)
    if expected_hw is not None and tuple(expected_hw) != actual_hw:
        raise ValueError(
            f"{label} size mismatch: expected {expected_hw}, got {actual_hw}"
        )
    return actual_hw


def _crop_box_from_center(center_xy: Tuple[float, float], crop_hw: Tuple[int, int]):
    center_x, center_y = center_xy
    crop_h, crop_w = crop_hw
    left = int(round(center_x - crop_w / 2.0))
    top = int(round(center_y - crop_h / 2.0))
    right = left + crop_w
    bottom = top + crop_h
    return left, top, right, bottom


def _rotation_support_hw(query_hw: Tuple[int, int]) -> Tuple[int, int]:
    query_h, query_w = query_hw
    support_side = int(math.ceil(math.sqrt(query_h ** 2 + query_w ** 2)))
    return support_side, support_side


def _center_crop_image(image: Image.Image, crop_hw: Tuple[int, int]) -> Image.Image:
    crop_h, crop_w = crop_hw
    left = int(round((image.width - crop_w) / 2.0))
    top = int(round((image.height - crop_h) / 2.0))
    right = left + crop_w
    bottom = top + crop_h
    return image.crop((left, top, right, bottom))


def _sample_query_centers(
    canvas_hw: Tuple[int, int],
    query_hw: Tuple[int, int],
    num_samples: int,
    seed: int,
    tile_hw: Tuple[int, int] = None,
    restrict_to_single_tile: bool = False,
) -> List[Tuple[float, float]]:
    canvas_h, canvas_w = canvas_hw
    query_h, query_w = query_hw
    margin_y = query_h / 2.0
    margin_x = query_w / 2.0
    rng = random.Random(seed)
    centers = []
    used = set()
    max_trials = max(1000, num_samples * 20)
    trials = 0

    if restrict_to_single_tile:
        if tile_hw is None:
            raise ValueError("tile_hw is required when restrict_to_single_tile is True")
        tile_h, tile_w = tile_hw
        if query_h > tile_h or query_w > tile_w:
            raise ValueError(
                f"Query size {query_hw} must fit inside tile size {tile_hw} "
                "when restrict_to_single_tile is enabled"
            )
        valid_regions = []
        num_tile_rows = canvas_h // tile_h
        num_tile_cols = canvas_w // tile_w
        for row in range(num_tile_rows):
            for col in range(num_tile_cols):
                tile_left = col * tile_w
                tile_top = row * tile_h
                min_x = tile_left + margin_x
                max_x = tile_left + tile_w - margin_x
                min_y = tile_top + margin_y
                max_y = tile_top + tile_h - margin_y
                if min_x > max_x or min_y > max_y:
                    continue
                valid_regions.append((min_x, max_x, min_y, max_y))
        if len(valid_regions) == 0:
            raise ValueError(
                f"Could not find a valid sampling region for query size {query_hw} "
                f"inside tile size {tile_hw}"
            )
        while len(centers) < num_samples and trials < max_trials:
            trials += 1
            min_x, max_x, min_y, max_y = rng.choice(valid_regions)
            cx = rng.uniform(min_x, max_x)
            cy = rng.uniform(min_y, max_y)
            key = (int(round(cx)), int(round(cy)))
            if key in used:
                continue
            used.add(key)
            centers.append((cx, cy))
    else:
        min_x = margin_x
        max_x = canvas_w - margin_x
        min_y = margin_y
        max_y = canvas_h - margin_y
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(
                f"Query size {query_hw} does not fit inside canvas size {canvas_hw}"
            )
        while len(centers) < num_samples and trials < max_trials:
            trials += 1
            cx = rng.uniform(min_x, max_x)
            cy = rng.uniform(min_y, max_y)
            key = (int(round(cx)), int(round(cy)))
            if key in used:
                continue
            used.add(key)
            centers.append((cx, cy))
    if len(centers) < num_samples:
        raise RuntimeError(
            f"Only sampled {len(centers)} unique query centers out of requested {num_samples}"
        )
    return centers


def _compute_tile_id_and_offset(
    center_xy: Tuple[float, float],
    tile_hw: Tuple[int, int],
    num_tile_rows: int,
    num_tile_cols: int,
):
    x, y = center_xy
    tile_h, tile_w = tile_hw
    col = min(max(int(x // tile_w), 0), num_tile_cols - 1)
    row = min(max(int(y // tile_h), 0), num_tile_rows - 1)
    tile_center_x = (col + 0.5) * tile_w
    tile_center_y = (row + 0.5) * tile_h
    tile_id = f"tile_r{row:02d}_c{col:02d}"
    dx = float(x - tile_center_x)
    dy = float(y - tile_center_y)
    return tile_id, (dx, dy), (tile_center_x, tile_center_y), row, col


@dataclass
class LocalArgs:
    satellite_image_path: str
    """
        Path to the full satellite image.
    """
    satellite_size_hw: str
    """
        Expected full image size, e.g. `4096x4096`.
    """
    uav_image_path: str
    """
        Path to the full UAV canvas. Must match satellite size.
    """
    uav_size_hw: str
    """
        Expected UAV canvas size, e.g. `4096x4096`.
    """
    output_root: str
    tile_size: int = 512
    """
        Satellite tile size in pixels. Square tiles.
    """
    query_size: int = 336
    """
        UAV query crop size in pixels. Square crops.
    """
    query_count: int = 1000
    seed: int = 42
    image_ext: str = "png"
    rotate: bool = False
    """
        If True, each sampled UAV query is randomly rotated before the final
        center crop is saved.
    """
    ensure_query_within_single_tile: bool = False
    """
        If True, sample query centers only from tile-local safe regions so
        every query crop is fully contained in one tile.
    """
    save_visual_index: bool = False
    overwrite: bool = False


def main(largs: LocalArgs):
    sat_path = os.path.realpath(os.path.expanduser(largs.satellite_image_path))
    uav_path = os.path.realpath(os.path.expanduser(largs.uav_image_path))
    out_root = os.path.realpath(os.path.expanduser(largs.output_root))
    sat_expected_hw = _parse_hw(largs.satellite_size_hw)
    uav_expected_hw = _parse_hw(largs.uav_size_hw)
    tile_hw = (int(largs.tile_size), int(largs.tile_size))
    query_hw = (int(largs.query_size), int(largs.query_size))
    support_hw = _rotation_support_hw(query_hw) if largs.rotate else query_hw
    rng = random.Random(int(largs.seed))

    if not os.path.isfile(sat_path):
        raise FileNotFoundError(f"Satellite image not found: {sat_path}")
    if not os.path.isfile(uav_path):
        raise FileNotFoundError(f"UAV image not found: {uav_path}")
    if os.path.exists(out_root) and not largs.overwrite and os.listdir(out_root):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_root}. "
            "Set `--overwrite true` or choose another directory."
        )
    os.makedirs(out_root, exist_ok=True)
    tiles_dir = os.path.join(out_root, "tiles")
    queries_dir = os.path.join(out_root, "queries")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)

    satellite_image = Image.open(sat_path).convert("RGB")
    uav_image = Image.open(uav_path).convert("RGB")
    sat_hw = _validate_or_infer_size(satellite_image, sat_expected_hw, "satellite image")
    uav_hw = _validate_or_infer_size(uav_image, uav_expected_hw, "uav image")
    if sat_hw != uav_hw:
        raise ValueError(f"Satellite and UAV sizes must match, got {sat_hw} vs {uav_hw}")

    canvas_h, canvas_w = sat_hw
    tile_h, tile_w = tile_hw
    if canvas_h % tile_h != 0 or canvas_w % tile_w != 0:
        raise ValueError(
            f"Tile size {tile_hw} must evenly divide image size {sat_hw} for this script"
        )
    num_tile_rows = canvas_h // tile_h
    num_tile_cols = canvas_w // tile_w

    print("Preparing tiles...")
    tiles_manifest = []
    for row in range(num_tile_rows):
        for col in range(num_tile_cols):
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w
            bottom = top + tile_h
            tile_id = f"tile_r{row:02d}_c{col:02d}"
            tile_img = satellite_image.crop((left, top, right, bottom))
            tile_relpath = os.path.join("tiles", f"{tile_id}.{largs.image_ext}")
            tile_abspath = os.path.join(out_root, tile_relpath)
            tile_img.save(tile_abspath)
            tiles_manifest.append({
                "tile_id": tile_id,
                "image": tile_relpath,
                "center": [float(left + tile_w / 2.0), float(top + tile_h / 2.0)],
                "extent": [float(tile_w), float(tile_h)],
                "row": row,
                "col": col,
                "bbox": [left, top, right, bottom],
            })

    print("Sampling UAV queries...")
    centers = _sample_query_centers(
        canvas_hw=sat_hw,
        query_hw=support_hw,
        num_samples=int(largs.query_count),
        seed=int(largs.seed),
        tile_hw=tile_hw,
        restrict_to_single_tile=bool(largs.ensure_query_within_single_tile),
    )

    queries_manifest = []
    for query_idx, center_xy in enumerate(centers):
        support_crop_box = _crop_box_from_center(center_xy, support_hw)
        query_img = uav_image.crop(support_crop_box)
        rotation_deg = 0.0
        if largs.rotate:
            rotation_deg = rng.uniform(0.0, 360.0)
            query_img = query_img.rotate(
                rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=False,
            )
            query_img = _center_crop_image(query_img, query_hw)
        crop_box = _crop_box_from_center(center_xy, query_hw)
        tile_id, offset_xy, _, row, col = _compute_tile_id_and_offset(
            center_xy, tile_hw, num_tile_rows, num_tile_cols
        )
        query_id = f"query_{query_idx:06d}"
        query_relpath = os.path.join("queries", f"{query_id}.{largs.image_ext}")
        query_abspath = os.path.join(out_root, query_relpath)
        query_img.save(query_abspath)
        queries_manifest.append({
            "query_id": query_id,
            "image": query_relpath,
            "gt_tile_id": tile_id,
            "offset": [offset_xy[0], offset_xy[1]],
            "coord": [float(center_xy[0]), float(center_xy[1])],
            "query_size": [int(query_hw[1]), int(query_hw[0])],
            "crop_bbox": [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])],
            "support_crop_bbox": [
                int(support_crop_box[0]), int(support_crop_box[1]),
                int(support_crop_box[2]), int(support_crop_box[3]),
            ],
            "rotation_deg": float(rotation_deg),
            "tile_row": int(row),
            "tile_col": int(col),
        })

    with open(os.path.join(out_root, "tiles.json"), "w", encoding="utf-8") as f:
        json.dump(tiles_manifest, f, indent=2)
    with open(os.path.join(out_root, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries_manifest, f, indent=2)

    if largs.save_visual_index:
        meta = {
            "satellite_image_path": sat_path,
            "uav_image_path": uav_path,
            "canvas_size_hw": [canvas_h, canvas_w],
            "tile_size_hw": [tile_h, tile_w],
            "query_size_hw": [query_hw[0], query_hw[1]],
            "support_size_hw": [support_hw[0], support_hw[1]],
            "rotate": bool(largs.rotate),
            "ensure_query_within_single_tile": bool(largs.ensure_query_within_single_tile),
            "num_tiles": len(tiles_manifest),
            "num_queries": len(queries_manifest),
            "seed": int(largs.seed),
        }
        with open(os.path.join(out_root, "dataset_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("Done.")
    print(f"- Output root: {out_root}")
    print(f"- Tiles: {len(tiles_manifest)}")
    print(f"- Queries: {len(queries_manifest)}")
    print(f"- Tile size: {tile_hw}")
    print(f"- Query size: {query_hw}")
    print(f"- Rotation enabled: {bool(largs.rotate)}")
    print(f"- Single-tile queries: {bool(largs.ensure_query_within_single_tile)}")


if __name__ == "__main__":
    args = tyro.cli(LocalArgs, description=__doc__)
    main(args)
