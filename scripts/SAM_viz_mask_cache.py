"""
Visualize cached SAM masks saved as .npz files and write overlays to disk.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-cache-dir",
        required=True,
        help="Directory containing cached .npz mask files.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root to recover the original image for overlay.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <mask-cache-dir>/../sam_masks_viz",
    )
    parser.add_argument(
        "--max-masks-overlay",
        type=int,
        default=24,
        help="Maximum number of masks to paint into the overlay.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    return parser.parse_args()


def recover_relpath(npz_path: str) -> str:
    stem = Path(npz_path).stem
    encoded_relpath = stem.rsplit("__", 1)[0]
    return encoded_relpath.replace("__", "/")


def colorize_masks(
    masks: np.ndarray,
    canvas_hw: tuple[int, int],
    max_masks_overlay: int,
) -> np.ndarray:
    h, w = canvas_hw
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if masks.size == 0:
        return overlay
    rng = np.random.default_rng(0)
    limit = min(int(max_masks_overlay), masks.shape[0])
    for mask_idx in range(limit):
        mask = masks[mask_idx].astype(np.uint8)
        if mask.shape != (h, w):
            mask = np.asarray(
                Image.fromarray(mask).resize((w, h), resample=Image.Resampling.NEAREST)
            )
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        overlay[mask > 0] = ((0.45 * overlay[mask > 0]) + (0.55 * color)).astype(np.uint8)
    return overlay


def union_mask_image(masks: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    h, w = canvas_hw
    if masks.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    union = masks.any(axis=0).astype(np.uint8) * 255
    if union.shape != (h, w):
        union = np.asarray(
            Image.fromarray(union).resize((w, h), resample=Image.Resampling.NEAREST)
        )
    return np.repeat(union[:, :, None], 3, axis=2)


def load_image(image_path: str):
    if image_path is None or not os.path.isfile(image_path):
        return None
    return np.asarray(Image.open(image_path).convert("RGB"))


def build_panel(image_rgb, masks: np.ndarray, max_masks_overlay: int) -> np.ndarray:
    if image_rgb is None:
        h, w = masks.shape[-2], masks.shape[-1]
        base = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        base = image_rgb
        h, w = base.shape[:2]

    overlay = colorize_masks(masks, (h, w), max_masks_overlay=max_masks_overlay)
    union = union_mask_image(masks, (h, w))
    blended = np.clip(base.astype(np.float32) * 0.55 + overlay.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    return np.concatenate([base, union, blended], axis=1)


def main():
    args = parse_args()
    mask_cache_dir = os.path.realpath(os.path.expanduser(args.mask_cache_dir))
    dataset_root = None if args.dataset_root is None else os.path.realpath(os.path.expanduser(args.dataset_root))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(mask_cache_dir), "sam_masks_viz")
    output_dir = os.path.realpath(os.path.expanduser(output_dir))

    npz_files = sorted(str(p) for p in Path(mask_cache_dir).glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz masks found in: {mask_cache_dir}")
    if args.max_files is not None:
        npz_files = npz_files[:int(args.max_files)]

    for npz_path in tqdm(npz_files, desc="Visualizing SAM masks"):
        payload = np.load(npz_path, allow_pickle=True)
        masks = np.asarray(payload["masks"]).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]

        relpath = recover_relpath(npz_path)
        image_path = None if dataset_root is None else os.path.join(dataset_root, relpath)
        image_rgb = load_image(image_path)
        panel = build_panel(image_rgb, masks, max_masks_overlay=args.max_masks_overlay)

        out_path = os.path.join(output_dir, relpath)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(panel).save(out_path)

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
