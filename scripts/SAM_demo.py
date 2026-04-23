import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

image_path = "data/Turkey-earthquake-2023-1/after_center_16x16_4096.jpg"
checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
tile_size = 1024
tile_overlap = 128
output_dir = Path("data/Turkey-earthquake-2023-1")
print(device)

image_bgr = cv2.imread(image_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]


def generate_tile_starts(full_size: int, tile_size: int, overlap: int) -> list[int]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size")
    starts = list(range(0, max(full_size - tile_size, 0) + 1, stride))
    if not starts:
        return [0]
    last_start = max(full_size - tile_size, 0)
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_tiles(image: np.ndarray, tile_size: int, overlap: int):
    img_h, img_w = image.shape[:2]
    y_starts = generate_tile_starts(img_h, tile_size, overlap)
    x_starts = generate_tile_starts(img_w, tile_size, overlap)
    for y0 in y_starts:
        for x0 in x_starts:
            yield x0, y0, image[y0:y0 + tile_size, x0:x0 + tile_size]


def blend_mask(image_view: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.35):
    image_view[mask] = ((1.0 - alpha) * image_view[mask] + alpha * color).astype(np.uint8)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=0.2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25,
    output_mode="binary_mask",
)
output_dir.mkdir(parents=True, exist_ok=True)
merged_binary_mask = np.zeros((h, w), dtype=np.uint8)
merged_overlay = image.copy()
rng = np.random.default_rng(0)
tile_count = 0
total_masks = 0

for x0, y0, tile in iter_tiles(image, tile_size=tile_size, overlap=tile_overlap):
    tile_count += 1
    tile_masks = mask_generator.generate(tile)
    total_masks += len(tile_masks)
    print(
        f"tile {tile_count:02d}: "
        f"x={x0}:{x0 + tile.shape[1]}, "
        f"y={y0}:{y0 + tile.shape[0]}, "
        f"masks={len(tile_masks)}"
    )

    global_mask_view = merged_binary_mask[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]]
    global_overlay_view = merged_overlay[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]]
    for ann in sorted(tile_masks, key=lambda x: x["area"], reverse=True):
        local_mask = ann["segmentation"]
        global_mask_view[local_mask] = 255
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        blend_mask(global_overlay_view, local_mask, color=color)

print(f"processed tiles: {tile_count}")
print(f"number of masks before merge: {total_masks}")

# 可视化合并结果
plt.figure(figsize=(12, 12))
plt.imshow(merged_overlay)
plt.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "after_auto_masks_tiled.png", dpi=200)
plt.close()
cv2.imwrite(str(output_dir / "after_auto_masks_tiled_union.png"), merged_binary_mask)
