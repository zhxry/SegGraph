#!/usr/bin/env python3
"""Visualize SegVLAD response segments and SuperSegments.

The figure contains:
  1. original tile
  2. segment mask overlay
  3. segment-wise high-response regions
  4. SuperSegment overlays for order 0/1/2/3 by default

High-response segments are selected by aggregating patch responses inside each
segment. By default, patch response is cosine similarity to a VLAD center; if
no cluster id is provided, the script selects the cluster with the highest
mean response on the tile.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

dir_name = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.realpath(f"{Path(dir_name).parent}")
if repo_root not in sys.path:
    sys.path.append(repo_root)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageOps

from custom_datasets.cvgl_dataset import CrossViewTileDataset
from cvgl_retrieval import DenseFeatureRecord, extract_dense_record
from segvlad_masking import (
    SAMGenerator,
    SegmentorConfig,
    build_mask_adjacency_revisit,
    load_or_generate_masks,
    project_masks_to_patch_grid,
)
from utilities import DinoV2ExtractFeatures


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.realpath(path)
    return os.path.realpath(os.path.join(base_dir, path))


def _to_tuple2(value) -> Tuple[int, int]:
    return int(value[0]), int(value[1])


def _feature_cache_path(cache_root: Optional[str], relpath: str) -> Optional[str]:
    if cache_root is None:
        return None
    return os.path.join(cache_root, f"{relpath.replace(chr(92), '/')}.pt")


def _mask_cache_path(cache_root: Optional[str], relpath: str) -> Optional[str]:
    if cache_root is None:
        return None
    relpath = relpath.replace("\\", "/")
    stem = hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:16]
    safe_name = relpath.replace("/", "__")
    return os.path.join(cache_root, f"{safe_name}__{stem}.npz")


def _load_masks_npz(path: str) -> List[np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        masks = payload["masks"] if "masks" in payload else payload[list(payload.keys())[0]]
    else:
        masks = payload
    if masks.ndim == 2:
        masks = masks[None, ...]
    return [np.asarray(mask).astype(bool) for mask in masks]


def _load_dense_record(cache_path: str) -> DenseFeatureRecord:
    payload = torch.load(cache_path, map_location="cpu")
    return DenseFeatureRecord(
        local_desc=payload["local_desc"],
        grid_hw=_to_tuple2(payload["grid_hw"]),
        image_hw=_to_tuple2(payload["image_hw"]),
        cropped_hw=_to_tuple2(payload["cropped_hw"]),
        patch_stride=int(payload["patch_stride"]),
        global_desc=payload.get("global_desc"),
    )


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(pil.convert("RGB"))


def _extract_record_from_image(
    image_path: str,
    model_type: str,
    desc_layer: int,
    desc_facet: str,
    device: torch.device,
    patch_stride: int,
    resize_hw: Optional[Tuple[int, int]],
) -> DenseFeatureRecord:
    pil = Image.open(image_path).convert("RGB")
    if resize_hw is not None:
        pil = pil.resize((int(resize_hw[1]), int(resize_hw[0])), Image.Resampling.BICUBIC)
    img = _pil_to_tensor(pil)
    dino = DinoV2ExtractFeatures(
        model_type,
        desc_layer,
        desc_facet,
        device=str(device),
    )
    return extract_dense_record(
        img,
        dino=dino,
        device=device,
        patch_stride=patch_stride,
        global_agg="vlad",
    )


def _load_or_extract_record(
    image_path: str,
    relpath: str,
    dense_cache_root: Optional[str],
    model_type: str,
    desc_layer: int,
    desc_facet: str,
    device: torch.device,
    patch_stride: int,
    resize_hw: Optional[Tuple[int, int]],
) -> DenseFeatureRecord:
    cache_path = _feature_cache_path(dense_cache_root, relpath)
    if cache_path is not None and os.path.isfile(cache_path):
        return _load_dense_record(cache_path)
    return _extract_record_from_image(
        image_path,
        model_type=model_type,
        desc_layer=desc_layer,
        desc_facet=desc_facet,
        device=device,
        patch_stride=patch_stride,
        resize_hw=resize_hw,
    )


def _center_crop_rgb(image_path: str, cropped_hw: Tuple[int, int], resize_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    pil = Image.open(image_path).convert("RGB")
    if resize_hw is not None:
        pil = pil.resize((int(resize_hw[1]), int(resize_hw[0])), Image.Resampling.BICUBIC)
    crop_h, crop_w = cropped_hw
    pil = ImageOps.fit(pil, (crop_w, crop_h), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    return np.asarray(pil)


def _load_centers(args: argparse.Namespace, resolved: Dict) -> Optional[torch.Tensor]:
    centers_path = args.c_centers
    if centers_path is None:
        centers_path = resolved.get("pretrained_vlad_centers")
    if centers_path is None:
        vlad_root = resolved.get("vlad_cache_root") or resolved.get("pretrained_vlad_cache_root")
        if vlad_root is not None:
            candidate = os.path.join(vlad_root, "c_centers.pt")
            if os.path.isfile(candidate):
                centers_path = candidate
    if centers_path is None:
        return None
    centers_path = _resolve_path(centers_path, resolved.get("cwd", os.getcwd()))
    if centers_path is None or not os.path.isfile(centers_path):
        raise FileNotFoundError(f"VLAD centers not found: {centers_path}")
    return torch.load(centers_path, map_location="cpu").float()


def _patch_response(
    local_desc: torch.Tensor,
    centers: Optional[torch.Tensor],
    cluster_id: Optional[int],
    response_method: str,
) -> Tuple[torch.Tensor, Optional[int]]:
    desc = local_desc.reshape(-1, local_desc.shape[-1]).float()
    if response_method == "descriptor-norm" or centers is None:
        response = torch.linalg.norm(desc, dim=1)
        return response, None

    desc = F.normalize(desc, dim=1)
    centers = F.normalize(centers.float(), dim=1)
    sims = desc @ centers.T
    if cluster_id is None:
        cluster_id = int(torch.argmax(sims.mean(dim=0)).item())
    if cluster_id < 0 or cluster_id >= centers.shape[0]:
        raise ValueError(f"--cluster-id must be in [0, {centers.shape[0] - 1}], got {cluster_id}")
    return sims[:, int(cluster_id)], int(cluster_id)


def _normalize01(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    min_v = values.min()
    max_v = values.max()
    return (values - min_v) / (max_v - min_v).clamp_min(1e-6)


def _segment_scores(mask_grid: torch.Tensor, patch_response: torch.Tensor, agg: str) -> torch.Tensor:
    masks = mask_grid.bool()
    scores = []
    for mask in masks:
        vals = patch_response[mask]
        if vals.numel() == 0:
            scores.append(torch.tensor(float("-inf")))
        elif agg == "max":
            scores.append(vals.max())
        else:
            scores.append(vals.mean())
    return torch.stack(scores)


def _selected_segments(scores: torch.Tensor, top_segments: int, min_score_quantile: Optional[float]) -> torch.Tensor:
    finite = torch.isfinite(scores)
    if finite.sum() == 0:
        raise ValueError("No finite segment scores found")
    keep = torch.zeros_like(scores, dtype=torch.bool)
    if top_segments > 0:
        k = min(int(top_segments), int(finite.sum().item()))
        top_idx = torch.topk(scores.masked_fill(~finite, float("-inf")), k=k).indices
        keep[top_idx] = True
    if min_score_quantile is not None:
        valid = scores[finite]
        q = torch.quantile(valid.float(), float(min_score_quantile))
        keep = keep | (scores >= q)
    if not keep.any():
        keep[int(torch.argmax(scores).item())] = True
    return keep


def _mask_grid_to_pixel(mask_grid: torch.Tensor, grid_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> np.ndarray:
    maps = mask_grid.float().reshape(mask_grid.shape[0], 1, grid_hw[0], grid_hw[1])
    up = F.interpolate(maps, size=out_hw, mode="nearest")[:, 0] > 0.5
    return up.detach().cpu().numpy().astype(bool)


def _palette(n: int) -> np.ndarray:
    colors = []
    for i in range(max(1, n)):
        h = i / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(h, 0.68, 0.95)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.asarray(colors, dtype=np.uint8)


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    out = (1.0 - alpha) * base.astype(np.float32) + alpha * overlay.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _segment_mask_overlay(base_rgb: np.ndarray, pixel_masks: np.ndarray, alpha: float) -> np.ndarray:
    colors = _palette(pixel_masks.shape[0])
    color_layer = base_rgb.copy()
    for i, mask in enumerate(pixel_masks):
        color_layer[mask] = colors[i]
    union = pixel_masks.any(axis=0)
    out = base_rgb.copy()
    out[union] = _blend(base_rgb[union], color_layer[union], alpha)
    return out


def _binary_mask_overlay(base_rgb: np.ndarray, pixel_masks: np.ndarray, alpha: float) -> np.ndarray:
    union = pixel_masks.any(axis=0)
    out = base_rgb.copy()
    color = np.zeros_like(base_rgb)
    color[..., 0] = 255
    color[..., 1] = 255
    color[..., 2] = 255
    out[union] = _blend(base_rgb[union], color[union], alpha)
    out[~union] = (out[~union].astype(np.float32) * 0.35).astype(np.uint8)
    return out


def _high_response_overlay(
    base_rgb: np.ndarray,
    pixel_masks: np.ndarray,
    selected: torch.Tensor,
    scores: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    selected_idx = torch.nonzero(selected, as_tuple=False).flatten().tolist()
    colors = _palette(len(selected_idx))
    score_norm = _normalize01(scores).detach().cpu().numpy()
    out = base_rgb.copy()
    for pos, seg_idx in enumerate(selected_idx):
        mask = pixel_masks[int(seg_idx)]
        color = colors[pos].astype(np.float32)
        seg_alpha = float(alpha * (0.45 + 0.55 * score_norm[int(seg_idx)]))
        out[mask] = _blend(out[mask], np.broadcast_to(color, out[mask].shape), seg_alpha)
    return out


def _to_gray_rgb(base_rgb: np.ndarray) -> np.ndarray:
    gray = np.dot(base_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _supersegment_selection(selected: torch.Tensor, adjacency: Optional[torch.Tensor], order: int) -> torch.Tensor:
    if order <= 0 or adjacency is None:
        return selected.clone()
    adj = adjacency.bool()
    return (adj[selected].any(dim=0) if selected.any() else selected.clone())


def _supersegment_groups(selected: torch.Tensor, adjacency: Optional[torch.Tensor], order: int) -> List[Tuple[int, torch.Tensor]]:
    selected_idx = torch.nonzero(selected, as_tuple=False).flatten().tolist()
    groups: List[Tuple[int, torch.Tensor]] = []
    for seg_idx in selected_idx:
        group = torch.zeros_like(selected, dtype=torch.bool)
        if order <= 0 or adjacency is None:
            group[int(seg_idx)] = True
        else:
            group = adjacency.bool()[int(seg_idx)].clone()
        groups.append((int(seg_idx), group))
    return groups


def _supersegment_overlay(
    base_rgb: np.ndarray,
    pixel_masks: np.ndarray,
    groups: Sequence[Tuple[int, torch.Tensor]],
    alpha: float,
) -> np.ndarray:
    gray = _to_gray_rgb(base_rgb)
    out = gray.copy()
    colors = _palette(len(groups))
    for pos, (_seg_idx, group) in enumerate(groups):
        group_np = group.detach().cpu().numpy().astype(bool)
        if not group_np.any():
            continue
        union = pixel_masks[group_np].any(axis=0)
        color = np.broadcast_to(colors[pos].astype(np.float32), out[union].shape)
        out[union] = _blend(out[union], color, alpha)
    return out


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _add_header(img: np.ndarray, text: str, header_h: int, font_size: int) -> np.ndarray:
    pil = Image.new("RGB", (img.shape[1], img.shape[0] + header_h), color=(255, 255, 255))
    pil.paste(Image.fromarray(img), (0, header_h))
    draw = ImageDraw.Draw(pil)
    font = _font(font_size)
    max_w = img.shape[1] - 16
    while font_size > 10 and draw.textbbox((0, 0), text, font=font)[2] > max_w:
        font_size -= 1
        font = _font(font_size)
    draw.text((8, max(2, (header_h - font_size) // 2)), text, fill=(0, 0, 0), font=font)
    return np.asarray(pil)


def _resize_panel(img: np.ndarray, panel_size: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = ImageOps.fit(pil, (panel_size, panel_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return np.asarray(pil)


def _compose_grid(
    panels: Sequence[Tuple[str, np.ndarray]],
    output: str,
    panel_size: int,
    cols: int,
    gap: int,
    margin: int,
    header_h: int,
    font_size: int,
    dpi: int,
    save_pdf: bool,
    show_headers: bool,
) -> Tuple[Path, Optional[Path]]:
    if show_headers:
        rendered = [_add_header(_resize_panel(img, panel_size), title, header_h, font_size) for title, img in panels]
    else:
        rendered = [_resize_panel(img, panel_size) for _title, img in panels]
    panel_w = panel_size
    panel_h = panel_size + (header_h if show_headers else 0)
    rows = int(np.ceil(len(rendered) / cols))
    canvas_w = margin * 2 + cols * panel_w + (cols - 1) * gap
    canvas_h = margin * 2 + rows * panel_h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    for i, panel in enumerate(rendered):
        row = i // cols
        col = i % cols
        x = margin + col * (panel_w + gap)
        y = margin + row * (panel_h + gap)
        canvas.paste(Image.fromarray(panel), (x, y))
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))
    pdf_path = None
    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        canvas.save(pdf_path, "PDF", resolution=float(dpi))
    return out_path, pdf_path


def _infer_from_results(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    if args.results_json is None:
        return {}, {}
    result = _load_json(args.results_json)
    resolved = result.get("Run-Config", {}).get("resolved", {})
    cli_args = result.get("Run-Config", {}).get("cli_args", {})
    return resolved, cli_args


def _arg_or_config(value, *fallbacks):
    if value is not None:
        return value
    for fallback in fallbacks:
        if fallback is not None:
            return fallback
    return None


def _build_dataset(args: argparse.Namespace, resolved: Dict):
    dataset_root = _arg_or_config(args.cvgl_dataset_root, resolved.get("cvgl_dataset_root"))
    if dataset_root is None:
        return None, None
    dataset_root = _resolve_path(dataset_root, resolved.get("cwd", os.getcwd()))
    dataset = CrossViewTileDataset.from_json(
        dataset_root,
        split=args.data_split,
        tiles_manifest=args.tiles_manifest,
        queries_manifest=args.queries_manifest,
    )
    if args.query_index is not None:
        index = dataset.database_num + int(args.query_index)
    else:
        index = int(args.tile_index)
    return dataset, index


def _get_image_context(args: argparse.Namespace, resolved: Dict):
    dataset, index = _build_dataset(args, resolved)
    if dataset is not None:
        image_path = dataset.get_image_paths()[index]
        relpath = dataset.get_image_relpaths(index)
        return dataset, index, image_path, relpath
    if args.image is None:
        raise ValueError("Pass --image, --cvgl-dataset-root, or --results-json with cvgl_dataset_root")
    image_path = os.path.realpath(os.path.expanduser(args.image))
    relpath = os.path.basename(image_path)
    return None, 0, image_path, relpath


def _load_masks_for_context(
    args: argparse.Namespace,
    dataset,
    index: int,
    image_path: str,
    record: DenseFeatureRecord,
    masks_cache_root: Optional[str],
    device: torch.device,
):
    if args.masks_file is not None:
        masks = _load_masks_npz(args.masks_file)
    elif dataset is not None:
        cache_path = _mask_cache_path(masks_cache_root, dataset.get_image_relpaths(index))
        if cache_path is not None and os.path.isfile(cache_path):
            masks = _load_masks_npz(cache_path)
        else:
            seg_cfg = SegmentorConfig(
                source=args.seg_source,
                sam_checkpoint=args.sam_checkpoint,
                sam_repo_root=args.sam_repo_root,
                sam_model_type=args.sam_model_type,
                sam_resize=args.sam_resize,
                sam_points_per_side=args.sam_points_per_side,
                sam_points_per_batch=args.sam_points_per_batch,
                sam_pred_iou_thresh=args.sam_pred_iou_thresh,
                sam_stability_score_thresh=args.sam_stability_score_thresh,
                sam_min_area_px=args.sam_min_area_px,
                cache_generated_masks=True,
                neighbor_method=args.neighbor_method,
            )
            sam_generator = None
            if seg_cfg.source in ["sam", "manifest_or_sam"]:
                try:
                    sam_generator = SAMGenerator(seg_cfg, device=str(device))
                except Exception as exc:
                    if seg_cfg.source == "sam":
                        raise
                    print(f"[warn] SAM unavailable; falling back to manifest/full mask: {exc}")
            masks = load_or_generate_masks(
                dataset,
                index,
                seg_cfg=seg_cfg,
                mask_generator=sam_generator,
                cache_root=masks_cache_root,
            )
    else:
        raise ValueError("--image mode requires --masks-file for segment visualization")

    mask_grid, pixel_masks = project_masks_to_patch_grid(
        masks,
        image_hw=record.image_hw,
        cropped_hw=record.cropped_hw,
        grid_hw=record.grid_hw,
        min_area_ratio=args.min_mask_area_ratio,
    )
    if pixel_masks is None:
        pixel_masks_np = _mask_grid_to_pixel(mask_grid, record.grid_hw, record.cropped_hw)
        pixel_masks = torch.as_tensor(pixel_masks_np, dtype=torch.bool)
    return mask_grid.cpu(), pixel_masks.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--cvgl-dataset-root", default=None)
    input_group.add_argument("--image", default=None)
    parser.add_argument("--results-json", default=None, help="Optional cvgl_results_*.json to infer paths/config.")
    parser.add_argument("--tiles-manifest", default=None)
    parser.add_argument("--queries-manifest", default=None)
    parser.add_argument("--data-split", default="test")
    parser.add_argument("--tile-index", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--all-tiles", action="store_true", help="Generate visualizations for every database tile.")
    parser.add_argument("--output-dir", default=None, help="Output directory used with --all-tiles.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of tiles in --all-tiles mode.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing outputs in --all-tiles mode.")
    parser.add_argument("--dense-cache-root", default=None)
    parser.add_argument("--masks-cache-root", default=None)
    parser.add_argument("--masks-file", default=None, help="Explicit .npz/.npy mask file for --image mode.")
    parser.add_argument("--c-centers", default=None, help="VLAD c_centers.pt. Can be inferred from --results-json.")
    parser.add_argument("--cluster-id", type=int, default=None, help="VLAD cluster to visualize. Defaults to strongest.")
    parser.add_argument("--response-method", choices=["vlad-center", "descriptor-norm"], default="vlad-center")
    parser.add_argument("--response-agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--top-segments", type=int, default=5)
    parser.add_argument("--min-score-quantile", type=float, default=None)
    parser.add_argument("--orders", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--neighbor-method", choices=["delaunay", "knn"], default="delaunay")
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--desc-layer", type=int, default=None)
    parser.add_argument("--desc-facet", default=None, choices=["query", "key", "value", "token"])
    parser.add_argument("--patch-stride", type=int, default=14)
    parser.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seg-source", choices=["manifest", "sam", "manifest_or_sam"], default="manifest_or_sam")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-repo-root", default="sam")
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b", "default"])
    parser.add_argument("--sam-resize", default="full", choices=["half", "full"])
    parser.add_argument("--sam-points-per-side", type=int, default=64)
    parser.add_argument("--sam-points-per-batch", type=int, default=128)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.80)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=0.88)
    parser.add_argument("--sam-min-area-px", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--output", default=".cache/supersegment_viz.png")
    parser.add_argument("--overview-output", default=None, help="Output for original/mask/response figure.")
    parser.add_argument("--supersegment-output", default=None, help="Output for order comparison figure.")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=36)
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--no-headers", action="store_true", help="Do not draw panel headers.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _safe_stem(relpath: str) -> str:
    stem = os.path.splitext(relpath.replace("\\", "/"))[0]
    return "".join(ch if ch.isalnum() or ch in ["-", "_", "."] else "_" for ch in stem.replace("/", "__"))


def _split_outputs(output: str, overview_output: Optional[str], supersegment_output: Optional[str]) -> Tuple[str, str]:
    output_path = Path(output)
    if overview_output is None:
        overview_output = str(output_path.with_name(f"{output_path.stem}_overview{output_path.suffix}"))
    if supersegment_output is None:
        supersegment_output = str(output_path.with_name(f"{output_path.stem}_orders{output_path.suffix}"))
    return overview_output, supersegment_output


def _render_visualization(
    args: argparse.Namespace,
    record_args: Dict,
    dataset,
    index: int,
    image_path: str,
    relpath: str,
    centers: Optional[torch.Tensor],
    output: str,
    overview_output: Optional[str] = None,
    supersegment_output: Optional[str] = None,
    quiet: bool = False,
) -> Tuple[Path, Path]:
    record = _load_or_extract_record(
        image_path,
        relpath=relpath,
        dense_cache_root=record_args["dense_cache_root"],
        model_type=record_args["model_type"],
        desc_layer=record_args["desc_layer"],
        desc_facet=record_args["desc_facet"],
        device=record_args["device"],
        patch_stride=args.patch_stride,
        resize_hw=record_args["resize_hw"],
    )
    mask_grid, pixel_masks = _load_masks_for_context(
        args,
        dataset=dataset,
        index=index,
        image_path=image_path,
        record=record,
        masks_cache_root=record_args["masks_cache_root"],
        device=record_args["device"],
    )
    pixel_masks_np = np.asarray(pixel_masks.numpy(), dtype=bool)
    base_rgb = _center_crop_rgb(image_path, record.cropped_hw, record_args["resize_hw"])

    response, cluster_id = _patch_response(
        record.local_desc,
        centers=centers,
        cluster_id=args.cluster_id,
        response_method=args.response_method,
    )
    scores = _segment_scores(mask_grid, response.cpu(), agg=args.response_agg)
    selected = _selected_segments(scores, top_segments=args.top_segments, min_score_quantile=args.min_score_quantile)

    overview_order = 3
    overview_adjacency = build_mask_adjacency_revisit(
        mask_grid,
        pixel_masks=pixel_masks,
        grid_hw=record.grid_hw,
        order=overview_order,
        method=args.neighbor_method,
    )
    overview_groups = _supersegment_groups(selected, overview_adjacency, overview_order)
    overview_panels: List[Tuple[str, np.ndarray]] = [
        ("Original tile", base_rgb),
        ("Binary mask", _binary_mask_overlay(base_rgb, pixel_masks_np, alpha=0.70)),
        ("Segment mask", _segment_mask_overlay(base_rgb, pixel_masks_np, alpha=0.42)),
        (
            f"SuperSegment order={overview_order}",
            _supersegment_overlay(base_rgb, pixel_masks_np, overview_groups, alpha=args.alpha),
        ),
    ]

    supersegment_panels: List[Tuple[str, np.ndarray]] = []
    for order in args.orders:
        adjacency = build_mask_adjacency_revisit(
            mask_grid,
            pixel_masks=pixel_masks,
            grid_hw=record.grid_hw,
            order=int(order),
            method=args.neighbor_method,
        )
        groups = _supersegment_groups(selected, adjacency, int(order))
        supersegment_panels.append(
            (
                f"SuperSegment order={int(order)}",
                _supersegment_overlay(base_rgb, pixel_masks_np, groups, alpha=args.alpha),
            )
        )

    overview_output, supersegment_output = _split_outputs(output, overview_output, supersegment_output)
    overview_path, overview_pdf_path = _compose_grid(
        overview_panels,
        output=overview_output,
        panel_size=args.panel_size,
        cols=args.cols,
        gap=args.gap,
        margin=args.margin,
        header_h=args.header_height,
        font_size=args.font_size,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
        show_headers=not args.no_headers,
    )
    supersegment_path, supersegment_pdf_path = _compose_grid(
        supersegment_panels,
        output=supersegment_output,
        panel_size=args.panel_size,
        cols=args.cols,
        gap=args.gap,
        margin=args.margin,
        header_h=args.header_height,
        font_size=args.font_size,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
        show_headers=not args.no_headers,
    )

    if not quiet:
        selected_ids = torch.nonzero(selected, as_tuple=False).flatten().tolist()
        print(f"[ok] saved {overview_path}")
        if overview_pdf_path is not None:
            print(f"[ok] saved {overview_pdf_path}")
        print(f"[ok] saved {supersegment_path}")
        if supersegment_pdf_path is not None:
            print(f"[ok] saved {supersegment_pdf_path}")
        print(f"image={image_path}")
        print(f"grid_hw={record.grid_hw} segments={mask_grid.shape[0]} selected_segments={selected_ids}")
        if cluster_id is not None:
            print(f"cluster_id={cluster_id}")
    return overview_path, supersegment_path


def main() -> None:
    args = parse_args()
    resolved, cli_args = _infer_from_results(args)
    cli_seg = cli_args.get("seg_cfg", {}) if isinstance(cli_args, dict) else {}

    model_type = _arg_or_config(args.model_type, cli_args.get("model_type") if isinstance(cli_args, dict) else None, "dinov2_vitg14")
    desc_layer = int(_arg_or_config(args.desc_layer, cli_args.get("desc_layer") if isinstance(cli_args, dict) else None, 23))
    desc_facet = _arg_or_config(args.desc_facet, cli_args.get("desc_facet") if isinstance(cli_args, dict) else None, "key")
    dense_cache_root = _resolve_path(
        _arg_or_config(args.dense_cache_root, resolved.get("dense_cache_root")),
        resolved.get("cwd", os.getcwd()),
    )
    masks_cache_root = _resolve_path(
        _arg_or_config(args.masks_cache_root, resolved.get("masks_cache_root")),
        resolved.get("cwd", os.getcwd()),
    )
    if args.sam_checkpoint == "checkpoints/sam_vit_h_4b8939.pth" and cli_seg.get("sam_checkpoint"):
        args.sam_checkpoint = cli_seg["sam_checkpoint"]
    if args.sam_repo_root == "sam" and cli_seg.get("sam_repo_root"):
        args.sam_repo_root = cli_seg["sam_repo_root"]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    resize_hw = None if args.resize is None else (int(args.resize[0]), int(args.resize[1]))
    record_args = {
        "dense_cache_root": dense_cache_root,
        "masks_cache_root": masks_cache_root,
        "model_type": model_type,
        "desc_layer": desc_layer,
        "desc_facet": desc_facet,
        "device": device,
        "resize_hw": resize_hw,
    }
    centers = _load_centers(args, resolved)

    if args.all_tiles:
        if args.image is not None or args.query_index is not None:
            raise ValueError("--all-tiles requires a CVGL dataset and cannot be combined with --image or --query-index")
        dataset, _ = _build_dataset(args, resolved)
        if dataset is None:
            raise ValueError("--all-tiles requires --cvgl-dataset-root or --results-json with cvgl_dataset_root")
        output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.output).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        indices = list(range(dataset.database_num))
        if args.limit is not None:
            indices = indices[:max(0, int(args.limit))]
        total = len(indices)
        for pos, index in enumerate(indices, start=1):
            image_path = dataset.get_image_paths()[index]
            relpath = dataset.get_image_relpaths(index)
            stem = _safe_stem(relpath)
            output = str(output_dir / f"{stem}.png")
            overview_output, supersegment_output = _split_outputs(output, None, None)
            if args.skip_existing and os.path.isfile(overview_output) and os.path.isfile(supersegment_output):
                print(f"[skip] {pos}/{total} {relpath}")
                continue
            print(f"[{pos}/{total}] {relpath}")
            overview_path, supersegment_path = _render_visualization(
                args,
                record_args=record_args,
                dataset=dataset,
                index=index,
                image_path=image_path,
                relpath=relpath,
                centers=centers,
                output=output,
                quiet=True,
            )
            print(f"[ok] saved {overview_path}")
            print(f"[ok] saved {supersegment_path}")
        return

    dataset, index, image_path, relpath = _get_image_context(args, resolved)
    _render_visualization(
        args,
        record_args=record_args,
        dataset=dataset,
        index=index,
        image_path=image_path,
        relpath=relpath,
        centers=centers,
        output=args.output,
        overview_output=args.overview_output,
        supersegment_output=args.supersegment_output,
        quiet=False,
    )


if __name__ == "__main__":
    main()
