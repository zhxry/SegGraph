"""
Visualize DINOv2 local offset prediction maps used by CVGL.

The script diagnoses common offset-prediction failure modes before plotting:
- missing or mismatched tile extent
- GT offset outside the tile extent
- GT coord inconsistent with center + offset
- final top-1 tile different from the GT tile, when a results JSON is given
- direct-y vs flipped-y GT response on the similarity map

Each output figure compares:
- Post-disaster Query
- Pre-disaster Top-1 Tile
- Similarity Map Response
- Sliding NCC Response
- Prediction Overlay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize offset prediction with sim_map and sliding NCC response maps."
    )
    parser.add_argument("--cvgl-dataset-root", required=True)
    parser.add_argument("--cvgl-tiles-manifest")
    parser.add_argument("--cvgl-queries-manifest")
    parser.add_argument("--data-split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--results-json", help="Optional cvgl_results_*.json from dino_v2_segvlad_CVGL.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-indices", nargs="*", type=int)
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument(
        "--tile-source",
        default="top1",
        choices=["gt", "top1"],
        help="Use the GT tile or final top-1 tile from --results-json for visualization.",
    )
    parser.add_argument(
        "--only-top1-correct",
        action="store_true",
        help="When --results-json is provided, skip queries whose final top-1 is not the GT tile.",
    )
    parser.add_argument("--model-type", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--desc-layer", type=int, default=11)
    parser.add_argument("--desc-facet", default="key", choices=["query", "key", "value", "token"])
    parser.add_argument("--local-top-m", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--patch-stride", type=int, default=14)
    parser.add_argument("--feature-cache-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size-m", type=float)
    parser.add_argument("--tile-size-px", type=float)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=32)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--no-headers", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_json(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _center_crop_pil(path: str, cropped_hw: Tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return T.CenterCrop(cropped_hw)(img)


def _offset_to_grid_xy(
    offset_xy: Tuple[float, float],
    extent_xy: Tuple[float, float],
    grid_hw: Tuple[int, int],
    flip_y: bool = False,
) -> Tuple[float, float]:
    h, w = grid_hw
    dx, dy = float(offset_xy[0]), float(offset_xy[1])
    extent_x, extent_y = float(extent_xy[0]), float(extent_xy[1])
    y_value = -dy if flip_y else dy
    col = ((dx / extent_x) + 0.5) * w - 0.5
    row = ((y_value / extent_y) + 0.5) * h - 0.5
    return col, row


def _grid_xy_to_offset(
    col: float,
    row: float,
    extent_xy: Tuple[float, float],
    grid_hw: Tuple[int, int],
) -> Tuple[float, float]:
    h, w = grid_hw
    x_norm = (float(col) + 0.5) / w - 0.5
    y_norm = (float(row) + 0.5) / h - 0.5
    return x_norm * float(extent_xy[0]), y_norm * float(extent_xy[1])


def _map_value_stats(sim_map: torch.Tensor, col: float, row: float) -> Dict[str, Optional[float]]:
    h, w = sim_map.shape
    if not (0 <= col <= w - 1 and 0 <= row <= h - 1):
        return {"value": None, "rank": None, "percentile": None}
    rr = int(np.clip(round(row), 0, h - 1))
    cc = int(np.clip(round(col), 0, w - 1))
    flat = sim_map.reshape(-1)
    value = float(sim_map[rr, cc].item())
    rank = int((flat > value).sum().item()) + 1
    percentile = float((flat <= value).float().mean().item())
    return {"value": value, "rank": float(rank), "percentile": percentile}


def _interp_colormap(values: np.ndarray, stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    out = np.zeros((*clipped.shape, 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        left_pos, left_rgb = stops[i]
        right_pos, right_rgb = stops[i + 1]
        mask = (clipped >= left_pos) & (clipped <= right_pos)
        if not np.any(mask):
            continue
        denom = max(1e-12, right_pos - left_pos)
        alpha = ((clipped[mask] - left_pos) / denom).reshape(-1, 1)
        left = np.asarray(left_rgb, dtype=np.float32).reshape(1, 3)
        right = np.asarray(right_rgb, dtype=np.float32).reshape(1, 3)
        out[mask] = left * (1.0 - alpha) + right * alpha
    return out.astype(np.uint8)


def _heatmap_image(values: np.ndarray, cmap: str, size: Tuple[int, int]) -> Image.Image:
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    norm = (values - v_min) / max(1e-12, v_max - v_min)
    if cmap == "viridis":
        stops = [
            (0.00, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.50, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.00, (253, 231, 37)),
        ]
    else:
        stops = [
            (0.00, (0, 0, 4)),
            (0.25, (80, 18, 123)),
            (0.50, (182, 54, 121)),
            (0.75, (251, 136, 97)),
            (1.00, (252, 253, 191)),
        ]
    rgb = _interp_colormap(norm, stops)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.NEAREST)


def _fit_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (20, 20, 20))
    fitted = img.copy()
    fitted.thumbnail(size, Image.Resampling.BICUBIC)
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def _grid_to_panel_xy(grid_xy: Tuple[float, float], grid_hw: Tuple[int, int], panel_size: Tuple[int, int]) -> Tuple[float, float]:
    h, w = grid_hw
    x = (float(grid_xy[0]) + 0.5) / w * panel_size[0]
    y = (float(grid_xy[1]) + 0.5) / h * panel_size[1]
    return x, y


def _offset_to_ncc_map_xy(
    offset_xy: Tuple[float, float],
    extent_xy: Tuple[float, float],
    tile_grid_hw: Tuple[int, int],
    query_grid_hw: Tuple[int, int],
    ncc_hw: Tuple[int, int],
) -> Tuple[float, float]:
    center_col, center_row = _offset_to_grid_xy(offset_xy, extent_xy, tile_grid_hw, flip_y=False)
    q_h, q_w = query_grid_hw
    n_h, n_w = ncc_hw
    col = center_col - (float(q_w) - 1.0) / 2.0
    row = center_row - (float(q_h) - 1.0) / 2.0
    return float(np.clip(col, 0.0, max(0.0, n_w - 1.0))), float(np.clip(row, 0.0, max(0.0, n_h - 1.0)))


def _ncc_argmax_prediction(
    ncc_map: torch.Tensor,
    query_grid_hw: Tuple[int, int],
    tile_grid_hw: Tuple[int, int],
    extent_xy: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    q_h, q_w = query_grid_hw
    t_h, t_w = tile_grid_hw
    best_flat = int(torch.argmax(ncc_map.reshape(-1)).item())
    best_row = best_flat // int(ncc_map.shape[1])
    best_col = best_flat % int(ncc_map.shape[1])
    center_col = float(best_col) + (float(q_w) - 1.0) / 2.0
    center_row = float(best_row) + (float(q_h) - 1.0) / 2.0
    norm_x = (center_col + 0.5) / float(t_w) - 0.5
    norm_y = (center_row + 0.5) / float(t_h) - 0.5
    pred_offset = (norm_x * float(extent_xy[0]), norm_y * float(extent_xy[1]))
    return pred_offset, (center_col, center_row), (float(best_col), float(best_row))


def _draw_markers(
    img: Image.Image,
    grid_hw: Tuple[int, int],
    pred_grid_xy: Tuple[float, float],
    gt_grid_xy: Optional[Tuple[float, float]],
    gt_flip_grid_xy: Optional[Tuple[float, float]],
) -> None:
    draw = ImageDraw.Draw(img)
    px, py = _grid_to_panel_xy(pred_grid_xy, grid_hw, img.size)
    draw.line((px - 10, py - 10, px + 10, py + 10), fill=(0, 255, 255), width=3)
    draw.line((px - 10, py + 10, px + 10, py - 10), fill=(0, 255, 255), width=3)
    if gt_grid_xy is not None:
        gx, gy = _grid_to_panel_xy(gt_grid_xy, grid_hw, img.size)
        draw.ellipse((gx - 10, gy - 10, gx + 10, gy + 10), outline=(80, 255, 80), width=3)
    if gt_flip_grid_xy is not None:
        fx, fy = _grid_to_panel_xy(gt_flip_grid_xy, grid_hw, img.size)
        draw.rectangle((fx - 9, fy - 9, fx + 9, fy + 9), outline=(255, 255, 255), width=3)


def _draw_response_marker(
    img: Image.Image,
    grid_hw: Tuple[int, int],
    xy: Tuple[float, float],
    color: Tuple[int, int, int],
    marker: str,
) -> None:
    draw = ImageDraw.Draw(img)
    x, y = _grid_to_panel_xy(xy, grid_hw, img.size)
    if marker == "x":
        draw.line((x - 11, y - 11, x + 11, y + 11), fill=color, width=3)
        draw.line((x - 11, y + 11, x + 11, y - 11), fill=color, width=3)
    elif marker == "circle":
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=color, width=3)
    elif marker == "square":
        draw.rectangle((x - 9, y - 9, x + 9, y + 9), outline=color, width=3)


def _draw_tile_prediction_overlay(
    tile_img: Image.Image,
    tile_grid_hw: Tuple[int, int],
    sim_grid_xy: Tuple[float, float],
    ncc_grid_xy: Tuple[float, float],
    gt_grid_xy: Optional[Tuple[float, float]],
    panel_size: Tuple[int, int],
) -> Image.Image:
    panel = _fit_image(tile_img, panel_size)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    def draw_point(xy: Tuple[float, float], color: Tuple[int, int, int], label: str, marker: str) -> None:
        x, y = _grid_to_panel_xy(xy, tile_grid_hw, panel.size)
        if marker == "x":
            draw.line((x - 12, y - 12, x + 12, y + 12), fill=color, width=4)
            draw.line((x - 12, y + 12, x + 12, y - 12), fill=color, width=4)
        elif marker == "square":
            draw.rectangle((x - 11, y - 11, x + 11, y + 11), outline=color, width=4)
        else:
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=color, width=4)
        draw.text((x + 13, y - 12), label, fill=color, font=font)

    draw_point(sim_grid_xy, (0, 255, 255), "Sim", "x")
    draw_point(ncc_grid_xy, (255, 80, 255), "NCC", "square")
    if gt_grid_xy is not None:
        draw_point(gt_grid_xy, (80, 255, 80), "GT", "circle")
    return panel


def _draw_panel_label(img: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bbox[2] + 10, bbox[3] + 8), fill=(0, 0, 0))
    draw.text((5, 4), label, fill=(255, 255, 255), font=font)


def _wrap_text(text: str, width: int = 138) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        while len(line) > width:
            split = line.rfind(" ", 0, width)
            if split <= 0:
                split = width
            lines.append(line[:split])
            line = line[split:].lstrip()
        lines.append(line)
    return lines


def _softargmax_offset(
    sim_map: torch.Tensor,
    extent_xy: Tuple[float, float],
    temperature: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], torch.Tensor]:
    h, w = sim_map.shape
    probs = F.softmax(sim_map.reshape(-1) * float(temperature), dim=0).reshape(h, w)
    ys = torch.linspace(-0.5 + 0.5 / h, 0.5 - 0.5 / h, h)
    xs = torch.linspace(-0.5 + 0.5 / w, 0.5 - 0.5 / w, w)
    exp_y = float((probs.sum(dim=1) * ys).sum().item())
    exp_x = float((probs.sum(dim=0) * xs).sum().item())
    pred_offset = (exp_x * float(extent_xy[0]), exp_y * float(extent_xy[1]))
    pred_col = (exp_x + 0.5) * w - 0.5
    pred_row = (exp_y + 0.5) * h - 0.5
    return pred_offset, (pred_col, pred_row), probs


def _resolve_extent(
    dataset: CrossViewTileDataset,
    db_idx: int,
    args: argparse.Namespace,
) -> Tuple[Optional[Tuple[float, float]], List[str]]:
    warnings: List[str] = []
    extent = dataset.get_tile_extent(db_idx)
    if extent is None:
        if args.tile_size_m is not None:
            extent = (float(args.tile_size_m), float(args.tile_size_m))
            warnings.append("tile extent missing; using --tile-size-m")
        elif args.tile_size_px is not None:
            extent = (float(args.tile_size_px), float(args.tile_size_px))
            warnings.append("tile extent missing; using --tile-size-px")
        else:
            warnings.append("tile extent missing and no fallback provided")
    return extent, warnings


def _result_query_row_map(dataset: CrossViewTileDataset, results: Optional[dict]) -> Dict[int, int]:
    if results is None or "Query-Relpaths" not in results:
        return {}
    rel_to_row = {str(rel): row for row, rel in enumerate(results["Query-Relpaths"])}
    mapping: Dict[int, int] = {}
    for qi in range(dataset.queries_num):
        global_idx = dataset.database_num + qi
        rel = str(dataset.get_image_relpaths(global_idx))
        if rel in rel_to_row:
            mapping[qi] = int(rel_to_row[rel])
    return mapping


def _top1_from_results(results: Optional[dict], row: Optional[int]) -> Optional[int]:
    if results is None or row is None:
        return None
    if "Final-DB-Indices-Orig" in results:
        return int(results["Final-DB-Indices-Orig"][row][0])
    if "Final-Indices" in results:
        return int(results["Final-Indices"][row][0])
    return None


def _choose_query_indices(args: argparse.Namespace, dataset: CrossViewTileDataset) -> List[int]:
    if args.query_indices is not None and len(args.query_indices) > 0:
        return [idx for idx in args.query_indices if 0 <= idx < dataset.queries_num]
    return list(range(min(int(args.max_queries), dataset.queries_num)))


def _diagnose_query(
    dataset: CrossViewTileDataset,
    query_idx: int,
    tile_idx: int,
    top1_idx: Optional[int],
    extent_xy: Optional[Tuple[float, float]],
) -> List[str]:
    messages: List[str] = []
    gt_idx = dataset.get_query_gt_db_index(query_idx)
    gt_offset = dataset.get_query_gt_offset(query_idx)
    gt_coord = dataset.get_query_coord(query_idx)
    tile_center = dataset.get_tile_center(tile_idx)
    gt_center = dataset.get_tile_center(int(gt_idx)) if gt_idx is not None else None

    if top1_idx is not None and gt_idx is not None:
        messages.append(f"top1==gt: {int(top1_idx) == int(gt_idx)} (top1={top1_idx}, gt={gt_idx})")
    if extent_xy is None:
        messages.append("extent: missing; cannot put GT offset into tile units reliably")
    else:
        messages.append(f"extent: ({extent_xy[0]:.6g}, {extent_xy[1]:.6g})")
        if gt_offset is not None:
            inside = abs(float(gt_offset[0])) <= 0.5 * float(extent_xy[0]) and \
                abs(float(gt_offset[1])) <= 0.5 * float(extent_xy[1])
            messages.append(f"gt_offset_inside_extent: {inside}")
    if gt_coord is not None and gt_offset is not None and gt_center is not None:
        expected = np.array(gt_center, dtype=float) + np.array(gt_offset, dtype=float)
        coord = np.array(gt_coord, dtype=float)
        delta = float(np.linalg.norm(coord - expected))
        messages.append(f"coord_vs_center_plus_offset_l2: {delta:.6g}")
    if tile_center is None:
        messages.append("tile_center: missing")
    return messages


def _plot_one(
    out_path: str,
    query_img: Image.Image,
    tile_img: Image.Image,
    sim_map: torch.Tensor,
    ncc_map: torch.Tensor,
    sim_pred_grid_xy: Tuple[float, float],
    ncc_pred_tile_grid_xy: Tuple[float, float],
    ncc_pred_map_xy: Tuple[float, float],
    gt_grid_xy: Optional[Tuple[float, float]],
    gt_ncc_map_xy: Optional[Tuple[float, float]],
    title: str,
    subtitle: str,
    dpi: int,
    panel_size_px: int,
    margin: int,
    gap: int,
    header_h: int,
    font_size: int,
    show_headers: bool,
) -> None:
    panel_size = (int(panel_size_px), int(panel_size_px))
    title_h = 34 if title else 0
    row_h = panel_size[1] + (header_h if show_headers else 0)
    panels_w = 5 * panel_size[0] + 4 * gap
    width = margin * 2 + panels_w
    height = margin * 2 + title_h + row_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    if title:
        draw.text((margin, margin // 2), title, fill=(0, 0, 0), font=title_font)
        if subtitle:
            bbox = draw.textbbox((0, 0), title, font=title_font)
            draw.text((margin + bbox[2] + 16, margin // 2), subtitle, fill=(80, 80, 80), font=title_font)

    sim_panel = _heatmap_image(sim_map.detach().cpu().numpy(), "magma", panel_size)
    _draw_response_marker(sim_panel, sim_map.shape, sim_pred_grid_xy, (0, 255, 255), "x")
    if gt_grid_xy is not None:
        _draw_response_marker(sim_panel, sim_map.shape, gt_grid_xy, (80, 255, 80), "circle")

    ncc_panel = _heatmap_image(ncc_map.detach().cpu().numpy(), "viridis", panel_size)
    _draw_response_marker(ncc_panel, ncc_map.shape, ncc_pred_map_xy, (255, 80, 255), "square")
    if gt_ncc_map_xy is not None:
        _draw_response_marker(ncc_panel, ncc_map.shape, gt_ncc_map_xy, (80, 255, 80), "circle")

    overlay_panel = _draw_tile_prediction_overlay(
        tile_img,
        tile_grid_hw=sim_map.shape,
        sim_grid_xy=sim_pred_grid_xy,
        ncc_grid_xy=ncc_pred_tile_grid_xy,
        gt_grid_xy=gt_grid_xy,
        panel_size=panel_size,
    )

    panels = [
        ("Post-disaster Query", _fit_image(query_img, panel_size)),
        ("Pre-disaster Top-1 Tile", _fit_image(tile_img, panel_size)),
        ("Similarity Map Response", sim_panel),
        ("Sliding NCC Response", ncc_panel),
        ("Prediction Overlay", overlay_panel),
    ]
    y0 = margin + title_h
    for idx, (label, panel) in enumerate(panels):
        x = margin + idx * (panel_size[0] + gap)
        y = y0
        if show_headers:
            draw.text((x + 4, y), label, fill=(0, 0, 0), font=header_font)
            y += header_h
        canvas.paste(panel, (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, dpi=(int(dpi), int(dpi)))


def main() -> None:
    global np, torch, F, T, Image, ImageDraw, ImageFont
    global CrossViewTileDataset, compute_local_similarity_map, compute_sliding_window_ncc_map, load_or_extract_record
    global DinoV2ExtractFeatures, seed_everything

    args = _parse_args()

    dir_name = None
    try:
        dir_name = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        dir_name = os.path.abspath("")
    lib_path = os.path.realpath(f"{Path(dir_name).parent}")
    if lib_path not in sys.path:
        sys.path.append(lib_path)

    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image, ImageDraw, ImageFont

    from custom_datasets.cvgl_dataset import CrossViewTileDataset
    from cvgl_retrieval import compute_local_similarity_map, compute_sliding_window_ncc_map, load_or_extract_record
    from utilities import DinoV2ExtractFeatures, seed_everything

    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        args.device = "cpu"
    runtime_device = torch.device(args.device)

    dataset = CrossViewTileDataset.from_json(
        args.cvgl_dataset_root,
        split=args.data_split,
        tiles_manifest=args.cvgl_tiles_manifest,
        queries_manifest=args.cvgl_queries_manifest,
    )
    results = _load_json(args.results_json)
    if args.tile_source == "top1" and results is None:
        raise ValueError("--tile-source top1 requires --results-json")
    row_map = _result_query_row_map(dataset, results)

    dino = DinoV2ExtractFeatures(
        args.model_type,
        args.desc_layer,
        args.desc_facet,
        device=str(runtime_device),
    )

    summary: List[dict] = []
    query_indices = _choose_query_indices(args, dataset)
    for query_idx in query_indices:
        gt_tile_idx = dataset.get_query_gt_db_index(query_idx)
        gt_offset = dataset.get_query_gt_offset(query_idx)
        if gt_tile_idx is None or gt_offset is None:
            print(f"[skip] query {query_idx}: missing gt tile or offset")
            continue

        result_row = row_map.get(query_idx)
        top1_idx = _top1_from_results(results, result_row)
        if args.only_top1_correct and top1_idx is not None and int(top1_idx) != int(gt_tile_idx):
            print(f"[skip] query {query_idx}: top1={top1_idx}, gt={gt_tile_idx}")
            continue

        tile_idx = int(gt_tile_idx)
        if args.tile_source == "top1":
            if top1_idx is None:
                print(f"[skip] query {query_idx}: --tile-source top1 requires --results-json")
                continue
            tile_idx = int(top1_idx)

        extent_xy, extent_warnings = _resolve_extent(dataset, tile_idx, args)
        if extent_xy is None:
            print(f"[skip] query {query_idx}: missing extent and no fallback")
            continue

        query_global_idx = dataset.database_num + query_idx
        query_record = load_or_extract_record(
            dataset,
            query_global_idx,
            dino=dino,
            device=runtime_device,
            cache_root=args.feature_cache_root,
            patch_stride=args.patch_stride,
        )
        tile_record = load_or_extract_record(
            dataset,
            tile_idx,
            dino=dino,
            device=runtime_device,
            cache_root=args.feature_cache_root,
            patch_stride=args.patch_stride,
        )

        sim_map = compute_local_similarity_map(
            query_record.local_desc,
            tile_record.local_desc,
            top_m=args.local_top_m,
        )
        sim_pred_offset, sim_pred_grid_xy, _probs = _softargmax_offset(
            sim_map,
            extent_xy=extent_xy,
            temperature=args.temperature,
        )
        ncc_map = compute_sliding_window_ncc_map(
            query_record.local_desc,
            tile_record.local_desc,
        )
        ncc_pred_offset, ncc_pred_tile_grid_xy, ncc_pred_map_xy = _ncc_argmax_prediction(
            ncc_map,
            query_grid_hw=query_record.grid_hw,
            tile_grid_hw=tile_record.grid_hw,
            extent_xy=extent_xy,
        )
        gt_grid_xy = _offset_to_grid_xy(gt_offset, extent_xy, sim_map.shape, flip_y=False)
        gt_flip_grid_xy = _offset_to_grid_xy(gt_offset, extent_xy, sim_map.shape, flip_y=True)
        gt_ncc_map_xy = _offset_to_ncc_map_xy(
            gt_offset,
            extent_xy=extent_xy,
            tile_grid_hw=tile_record.grid_hw,
            query_grid_hw=query_record.grid_hw,
            ncc_hw=ncc_map.shape,
        )
        gt_stats = _map_value_stats(sim_map, gt_grid_xy[0], gt_grid_xy[1])
        gt_flip_stats = _map_value_stats(sim_map, gt_flip_grid_xy[0], gt_flip_grid_xy[1])
        ncc_gt_stats = _map_value_stats(ncc_map, gt_ncc_map_xy[0], gt_ncc_map_xy[1])

        query_img = _center_crop_pil(dataset.get_image_paths()[query_global_idx], query_record.cropped_hw)
        tile_img = _center_crop_pil(dataset.get_image_paths()[tile_idx], tile_record.cropped_hw)

        diagnostics = []
        diagnostics.extend(_diagnose_query(dataset, query_idx, tile_idx, top1_idx, extent_xy))
        diagnostics.extend(extent_warnings)
        diagnostics.append(f"query_image_hw={query_record.image_hw}, query_cropped_hw={query_record.cropped_hw}, query_grid={query_record.grid_hw}")
        diagnostics.append(f"tile_image_hw={tile_record.image_hw}, tile_cropped_hw={tile_record.cropped_hw}, tile_grid={tile_record.grid_hw}")
        diagnostics.append(f"gt_offset={tuple(float(v) for v in gt_offset)}")
        diagnostics.append(f"sim_map_pred_offset=({sim_pred_offset[0]:.6g}, {sim_pred_offset[1]:.6g})")
        diagnostics.append(f"sliding_ncc_pred_offset=({ncc_pred_offset[0]:.6g}, {ncc_pred_offset[1]:.6g})")
        diagnostics.append(f"gt_sim_percentile={gt_stats['percentile']}, gt_rank={gt_stats['rank']}")
        diagnostics.append(f"gt_flip_y_sim_percentile={gt_flip_stats['percentile']}, gt_flip_y_rank={gt_flip_stats['rank']}")
        diagnostics.append(f"gt_ncc_percentile={ncc_gt_stats['percentile']}, gt_ncc_rank={ncc_gt_stats['rank']}")

        out_name = f"query_{query_idx:06d}_tile_{tile_idx:06d}_{args.tile_source}.png"
        out_path = os.path.join(args.output_dir, out_name)
        title = (
            f"query={query_idx}, tile={tile_idx}, gt_tile={gt_tile_idx}, "
            f"top1={top1_idx}, source={args.tile_source}"
        )
        _plot_one(
            out_path,
            query_img=query_img,
            tile_img=tile_img,
            sim_map=sim_map,
            ncc_map=ncc_map,
            sim_pred_grid_xy=sim_pred_grid_xy,
            ncc_pred_tile_grid_xy=ncc_pred_tile_grid_xy,
            ncc_pred_map_xy=ncc_pred_map_xy,
            gt_grid_xy=gt_grid_xy,
            gt_ncc_map_xy=gt_ncc_map_xy,
            title=title,
            subtitle="Sim=cyan x, NCC=magenta square, GT=green circle",
            dpi=args.dpi,
            panel_size_px=args.panel_size,
            margin=args.margin,
            gap=args.gap,
            header_h=args.header_height,
            font_size=args.font_size,
            show_headers=not args.no_headers,
        )

        row = {
            "query_idx": int(query_idx),
            "tile_idx": int(tile_idx),
            "gt_tile_idx": int(gt_tile_idx),
            "top1_idx": None if top1_idx is None else int(top1_idx),
            "top1_is_gt": None if top1_idx is None else bool(int(top1_idx) == int(gt_tile_idx)),
            "extent_xy": [float(extent_xy[0]), float(extent_xy[1])],
            "gt_offset": [float(gt_offset[0]), float(gt_offset[1])],
            "pred_offset": [float(sim_pred_offset[0]), float(sim_pred_offset[1])],
            "sim_map_pred_offset": [float(sim_pred_offset[0]), float(sim_pred_offset[1])],
            "sliding_ncc_pred_offset": [float(ncc_pred_offset[0]), float(ncc_pred_offset[1])],
            "pred_error_l2": float(np.linalg.norm(np.array(sim_pred_offset) - np.array(gt_offset, dtype=float))),
            "sim_map_pred_error_l2": float(np.linalg.norm(np.array(sim_pred_offset) - np.array(gt_offset, dtype=float))),
            "sliding_ncc_pred_error_l2": float(np.linalg.norm(np.array(ncc_pred_offset) - np.array(gt_offset, dtype=float))),
            "sim_map_shape": [int(sim_map.shape[0]), int(sim_map.shape[1])],
            "sliding_ncc_shape": [int(ncc_map.shape[0]), int(ncc_map.shape[1])],
            "gt_grid_xy": [float(gt_grid_xy[0]), float(gt_grid_xy[1])],
            "gt_flip_y_grid_xy": [float(gt_flip_grid_xy[0]), float(gt_flip_grid_xy[1])],
            "sim_map_pred_grid_xy": [float(sim_pred_grid_xy[0]), float(sim_pred_grid_xy[1])],
            "sliding_ncc_pred_tile_grid_xy": [float(ncc_pred_tile_grid_xy[0]), float(ncc_pred_tile_grid_xy[1])],
            "sliding_ncc_pred_map_xy": [float(ncc_pred_map_xy[0]), float(ncc_pred_map_xy[1])],
            "gt_ncc_map_xy": [float(gt_ncc_map_xy[0]), float(gt_ncc_map_xy[1])],
            "gt_sim_value": gt_stats["value"],
            "gt_sim_rank": gt_stats["rank"],
            "gt_sim_percentile": gt_stats["percentile"],
            "gt_flip_y_sim_value": gt_flip_stats["value"],
            "gt_flip_y_sim_rank": gt_flip_stats["rank"],
            "gt_flip_y_sim_percentile": gt_flip_stats["percentile"],
            "gt_ncc_value": ncc_gt_stats["value"],
            "gt_ncc_rank": ncc_gt_stats["rank"],
            "gt_ncc_percentile": ncc_gt_stats["percentile"],
            "output": out_path,
            "diagnostics": diagnostics,
        }
        summary.append(row)
        print(f"[ok] saved {out_path}")
        for msg in diagnostics:
            print(f"  - {msg}")

    summary_path = os.path.join(args.output_dir, "sim_map_offset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
