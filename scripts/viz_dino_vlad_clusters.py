"""
Visualize which VLAD vocabulary cluster each DINOv2 patch responds to.

Reference behavior:
- DINOv2 patch extraction follows `utilities.DinoV2ExtractFeatures`
- hard assignment matches `Revisit-Anything/func_vpr.py`:
    labels = argmax(normalized_desc @ normalized_centers.T)

Outputs:
- original cropped image
- discrete patch-to-cluster map
- overlay visualization
- optional soft-assignment heatmaps for selected clusters
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    dir_name = os.path.abspath("")
lib_path = os.path.realpath(f"{Path(dir_name).parent}")
if lib_path not in sys.path:
    sys.path.append(lib_path)

import argparse
import colorsys
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw

from cvgl_retrieval import extract_dense_record
from utilities import DinoV2ExtractFeatures, VLAD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DINOv2 patch-to-VLAD-cluster assignments for one image or an entire dataset."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Input image path.")
    input_group.add_argument(
        "--dataset-root",
        help="Dataset root. The script will recursively process all images under it.",
    )
    parser.add_argument(
        "--c-centers",
        required=True,
        help="Path to VLAD cluster centers file `c_centers.pt`.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the visualizations.",
    )
    parser.add_argument(
        "--model-type",
        default="dinov2_vitg14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    parser.add_argument("--desc-layer", type=int, default=31)
    parser.add_argument(
        "--desc-facet",
        default="value",
        choices=["query", "key", "value", "token"],
    )
    parser.add_argument(
        "--assignment",
        default="hard",
        choices=["hard", "soft"],
        help="Patch-to-cluster assignment mode.",
    )
    parser.add_argument(
        "--soft-temp",
        type=float,
        default=1.0,
        help="Softmax temperature used only when --assignment soft.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="Optional resize before DINO extraction.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Example: cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=14,
        help="Patch stride used by DINOv2.",
    )
    parser.add_argument(
        "--topk-soft-clusters",
        type=int,
        default=8,
        help="How many clusters to visualize as heatmaps in soft mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha.",
    )
    parser.add_argument(
        "--glob-patterns",
        nargs="+",
        default=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"],
        help="Filename patterns used with --dataset-root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to process in dataset mode.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose main overlay output already exists.",
    )
    return parser.parse_args()


def _load_image_tensor(image_path: str, resize_hw: Optional[List[int]]) -> tuple[Image.Image, torch.Tensor]:
    pil = Image.open(image_path).convert("RGB")
    if resize_hw is not None:
        pil = pil.resize((int(resize_hw[1]), int(resize_hw[0])), Image.BICUBIC)
    tensor = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(pil)
    return pil, tensor


def _tensor_to_uint8_img(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
    img = img.detach().cpu() * std + mean
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _center_crop_tensor(img: torch.Tensor, crop_hw: tuple[int, int]) -> torch.Tensor:
    return T.CenterCrop(crop_hw)(img)


def _load_vlad(c_centers_path: str) -> VLAD:
    vlad = VLAD(num_clusters=1, cache_dir=os.path.dirname(os.path.abspath(c_centers_path)))
    vlad.fit(None)
    if os.path.abspath(c_centers_path) != os.path.join(vlad.cache_dir, "c_centers.pt"):
        vlad.c_centers = torch.load(c_centers_path, map_location="cpu")
        vlad.num_clusters = int(vlad.c_centers.shape[0])
        vlad.desc_dim = int(vlad.c_centers.shape[1])
        if vlad.kmeans is not None:
            vlad.kmeans.centroids = vlad.c_centers
    return vlad


def _compute_assignments(
    patch_descs: torch.Tensor,
    c_centers: torch.Tensor,
    assignment: Literal["hard", "soft"],
    soft_temp: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    descs = F.normalize(patch_descs.float(), dim=1)
    centers = F.normalize(c_centers.float(), dim=1)
    cos_sim = descs @ centers.T
    if assignment == "hard":
        labels = torch.argmax(cos_sim, dim=1)
        return labels, None, cos_sim
    soft_assign = F.softmax(soft_temp * cos_sim, dim=1)
    labels = torch.argmax(soft_assign, dim=1)
    return labels, soft_assign, cos_sim


def _upsample_patch_map(
    patch_values: torch.Tensor,
    grid_hw: tuple[int, int],
    out_hw: tuple[int, int],
    mode: str = "nearest",
) -> np.ndarray:
    patch_values = patch_values.reshape(1, 1, grid_hw[0], grid_hw[1]).float()
    up = F.interpolate(patch_values, size=out_hw, mode=mode)
    return up[0, 0].cpu().numpy()


def _make_cluster_palette(num_clusters: int) -> List[List[int]]:
    if num_clusters <= 0:
        return [[0, 0, 0]]
    palette = []
    for idx in range(num_clusters):
        h = idx / max(num_clusters, 1)
        s = 0.65
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette.append([int(255 * r), int(255 * g), int(255 * b)])
    return palette


def _colorize_label_map(label_map: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    color_map = np.asarray(palette, dtype=np.uint8)
    safe_labels = np.clip(label_map, 0, len(palette) - 1)
    return color_map[safe_labels]


def _blend_overlay(base_rgb: np.ndarray, color_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * color_rgb.astype(np.float32)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _concat_panels_h(images: List[np.ndarray]) -> np.ndarray:
    max_h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] == max_h:
            padded.append(img)
            continue
        canvas = np.full((max_h, img.shape[1], 3), 255, dtype=np.uint8)
        canvas[: img.shape[0]] = img
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def _add_header(img: np.ndarray, text: str, header_h: int = 28) -> np.ndarray:
    pil = Image.new("RGB", (img.shape[1], img.shape[0] + header_h), color=(255, 255, 255))
    pil.paste(Image.fromarray(img), (0, header_h))
    draw = ImageDraw.Draw(pil)
    draw.text((8, 6), text, fill=(0, 0, 0))
    return np.asarray(pil)


def _save_main_figure(
    cropped_rgb: np.ndarray,
    label_map: np.ndarray,
    num_clusters: int,
    out_path: str,
    alpha: float,
) -> List[List[int]]:
    palette = _make_cluster_palette(num_clusters)
    colorized = _colorize_label_map(label_map, palette)
    overlay = _blend_overlay(cropped_rgb, colorized, alpha=alpha)
    panel = _concat_panels_h(
        [
            _add_header(cropped_rgb, "Cropped Image"),
            _add_header(colorized, "Patch Cluster Map"),
            _add_header(overlay, "Overlay"),
        ]
    )
    Image.fromarray(panel).save(out_path)
    return palette


def _save_soft_heatmaps(
    cropped_rgb: np.ndarray,
    soft_assign: torch.Tensor,
    grid_hw: tuple[int, int],
    out_hw: tuple[int, int],
    out_path: str,
    topk: int,
) -> None:
    cluster_strength = soft_assign.mean(dim=0)
    topk = min(topk, int(soft_assign.shape[1]))
    top_clusters = torch.topk(cluster_strength, k=topk).indices.tolist()

    panels = []
    for cluster_idx in top_clusters:
        heatmap = _upsample_patch_map(
            soft_assign[:, cluster_idx], grid_hw=grid_hw, out_hw=out_hw, mode="bilinear"
        )
        heat_rgb = _heatmap_to_rgb(heatmap)
        overlay = _blend_overlay(cropped_rgb, heat_rgb, alpha=0.55)
        panels.append(_add_header(overlay, f"Cluster {cluster_idx}"))
    Image.fromarray(_concat_panels_h(panels)).save(out_path)


def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    denom = float(heatmap.max())
    if denom > 0:
        heatmap = heatmap / denom
    anchors = np.asarray(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    xs = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    out = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    flat = heatmap.reshape(-1)
    for c in range(3):
        out[..., c] = np.interp(flat, xs, anchors[:, c]).reshape(heatmap.shape)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _collect_dataset_images(dataset_root: str, patterns: List[str]) -> List[str]:
    root = Path(dataset_root)
    image_paths = set()
    for pattern in patterns:
        image_paths.update(str(path) for path in root.rglob(pattern))
    return sorted(image_paths)


def _resolve_image_output_dir(image_path: str, args: argparse.Namespace) -> str:
    if args.dataset_root is None:
        return args.output_dir
    rel_parent = os.path.dirname(os.path.relpath(image_path, args.dataset_root))
    if rel_parent == ".":
        return args.output_dir
    return os.path.join(args.output_dir, rel_parent)


def _overlay_output_path(image_path: str, out_dir: str) -> str:
    image_name = Path(image_path).stem
    return os.path.join(out_dir, f"{image_name}_cluster_overlay.png")


def process_image(
    image_path: str,
    out_dir: str,
    args: argparse.Namespace,
    dino: DinoV2ExtractFeatures,
    vlad: VLAD,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    resize_hw = None if args.resize is None else [int(args.resize[0]), int(args.resize[1])]
    image_name = Path(image_path).stem

    _, img_tensor = _load_image_tensor(image_path, resize_hw=resize_hw)
    record = extract_dense_record(
        img_tensor,
        dino=dino,
        device=torch.device(args.device),
        patch_stride=args.patch_stride,
        global_agg="vlad",
    )

    cropped_tensor = _center_crop_tensor(img_tensor, record.cropped_hw)
    cropped_rgb = _tensor_to_uint8_img(cropped_tensor)
    patch_descs = record.local_desc.reshape(-1, record.local_desc.shape[-1])

    labels, soft_assign, cos_sim = _compute_assignments(
        patch_descs=patch_descs,
        c_centers=vlad.c_centers.cpu(),
        assignment=args.assignment,
        soft_temp=args.soft_temp,
    )

    label_map = _upsample_patch_map(
        labels,
        grid_hw=record.grid_hw,
        out_hw=record.cropped_hw,
        mode="nearest",
    ).astype(np.int64)

    np.save(os.path.join(out_dir, f"{image_name}_cluster_map.npy"), label_map)
    np.save(
        os.path.join(out_dir, f"{image_name}_patch_labels.npy"),
        labels.reshape(record.grid_hw).cpu().numpy(),
    )
    torch.save(
        {
            "patch_labels": labels.reshape(record.grid_hw).cpu(),
            "grid_hw": record.grid_hw,
            "cropped_hw": record.cropped_hw,
            "image_hw": record.image_hw,
            "patch_stride": record.patch_stride,
            "cos_sim": cos_sim.cpu(),
            "soft_assign": None if soft_assign is None else soft_assign.cpu(),
            "c_centers_path": os.path.abspath(args.c_centers),
            "model_type": args.model_type,
            "desc_layer": args.desc_layer,
            "desc_facet": args.desc_facet,
            "assignment": args.assignment,
        },
        os.path.join(out_dir, f"{image_name}_assignments.pt"),
    )

    palette = _save_main_figure(
        cropped_rgb=cropped_rgb,
        label_map=label_map,
        num_clusters=int(vlad.c_centers.shape[0]),
        out_path=os.path.join(out_dir, f"{image_name}_cluster_overlay.png"),
        alpha=args.alpha,
    )

    if soft_assign is not None:
        _save_soft_heatmaps(
            cropped_rgb=cropped_rgb,
            soft_assign=soft_assign.cpu(),
            grid_hw=record.grid_hw,
            out_hw=record.cropped_hw,
            out_path=os.path.join(out_dir, f"{image_name}_soft_heatmaps.png"),
            topk=args.topk_soft_clusters,
        )

    with open(os.path.join(out_dir, f"{image_name}_palette.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_clusters": int(vlad.c_centers.shape[0]),
                "palette_rgb": {str(i): palette[i] for i in range(len(palette))},
            },
            f,
            indent=2,
        )

    unique_labels, counts = torch.unique(labels, return_counts=True)
    sorted_pairs = sorted(
        zip(unique_labels.tolist(), counts.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "image": image_path,
        "output_dir": os.path.abspath(out_dir),
        "cropped_hw": list(record.cropped_hw),
        "grid_hw": list(record.grid_hw),
        "num_clusters": int(vlad.c_centers.shape[0]),
        "cluster_histogram": [
            {"cluster_id": int(cluster_id), "patch_count": int(patch_count)}
            for cluster_id, patch_count in sorted_pairs
        ],
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    vlad = _load_vlad(args.c_centers)
    dino = DinoV2ExtractFeatures(
        args.model_type,
        layer=args.desc_layer,
        facet=args.desc_facet,
        use_cls=False,
        norm_descs=True,
        device=str(device),
    )

    if args.image is not None:
        result = process_image(
            image_path=args.image,
            out_dir=args.output_dir,
            args=args,
            dino=dino,
            vlad=vlad,
        )
        print(f"Image: {result['image']}")
        print(f"Cropped HW: {tuple(result['cropped_hw'])}, patch grid: {tuple(result['grid_hw'])}")
        print(f"Num clusters: {result['num_clusters']}")
        print("Cluster histogram (cluster_id: patch_count):")
        for item in result["cluster_histogram"]:
            print(f"  {item['cluster_id']}: {item['patch_count']}")
        print(f"Saved outputs to: {result['output_dir']}")
        return

    image_paths = _collect_dataset_images(args.dataset_root, args.glob_patterns)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if len(image_paths) == 0:
        raise ValueError(f"No images found under dataset root: {args.dataset_root}")

    results = []
    num_skipped = 0
    for idx, image_path in enumerate(image_paths):
        out_dir = _resolve_image_output_dir(image_path, args)
        overlay_path = _overlay_output_path(image_path, out_dir)
        if args.skip_existing and os.path.exists(overlay_path):
            num_skipped += 1
            print(f"[{idx + 1}/{len(image_paths)}] skip {image_path}")
            continue
        print(f"[{idx + 1}/{len(image_paths)}] process {image_path}")
        result = process_image(
            image_path=image_path,
            out_dir=out_dir,
            args=args,
            dino=dino,
            vlad=vlad,
        )
        results.append(result)

    summary = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "num_images_found": len(image_paths),
        "num_processed": len(results),
        "num_skipped": num_skipped,
        "output_dir": os.path.abspath(args.output_dir),
        "c_centers_path": os.path.abspath(args.c_centers),
        "model_type": args.model_type,
        "desc_layer": args.desc_layer,
        "desc_facet": args.desc_facet,
        "assignment": args.assignment,
        "images": results,
    }
    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Images found: {summary['num_images_found']}")
    print(f"Processed: {summary['num_processed']}")
    print(f"Skipped: {summary['num_skipped']}")
    print(f"Saved outputs to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
