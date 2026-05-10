"""
Visualize DINOv2 local similarity maps used by CVGL offset prediction.

The script diagnoses common offset-prediction failure modes before plotting:
- missing or mismatched tile extent
- GT offset outside the tile extent
- GT coord inconsistent with center + offset
- final top-1 tile different from the GT tile, when a results JSON is given
- direct-y vs flipped-y GT response on the similarity map
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
        description="Visualize sim_map offset prediction and GT offset response."
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
        default="gt",
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
    parser.add_argument("--tile-size-m", type=float)
    parser.add_argument("--tile-size-px", type=float)
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
    probs: torch.Tensor,
    pred_grid_xy: Tuple[float, float],
    gt_grid_xy: Optional[Tuple[float, float]],
    gt_flip_grid_xy: Optional[Tuple[float, float]],
    title: str,
    diagnostics: Sequence[str],
    dpi: int,
) -> None:
    panel_size = (420, 420)
    margin = 18
    title_h = 38
    text_h = 210
    width = panel_size[0] * 2 + margin * 3
    height = title_h + panel_size[1] * 2 + margin * 3 + text_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, 12), title, fill=(0, 0, 0), font=font)

    panels = [
        (_fit_image(query_img, panel_size), "query crop"),
        (_fit_image(tile_img, panel_size), "tile crop"),
        (_heatmap_image(sim_map.detach().cpu().numpy(), "magma", panel_size), "raw sim_map"),
        (_heatmap_image(probs.detach().cpu().numpy(), "viridis", panel_size), "softmax probability"),
    ]

    positions = [
        (margin, title_h + margin),
        (margin * 2 + panel_size[0], title_h + margin),
        (margin, title_h + margin * 2 + panel_size[1]),
        (margin * 2 + panel_size[0], title_h + margin * 2 + panel_size[1]),
    ]
    for idx, ((panel, label), pos) in enumerate(zip(panels, positions)):
        if idx >= 2:
            _draw_markers(panel, sim_map.shape, pred_grid_xy, gt_grid_xy, gt_flip_grid_xy)
        _draw_panel_label(panel, label)
        canvas.paste(panel, pos)

    legend_y = title_h + margin * 3 + panel_size[1] * 2 + 4
    draw.text((margin, legend_y), "Markers: pred=cyan x, gt=green circle, gt flip-y=white square",
              fill=(0, 0, 0), font=font)
    diag_text = "\n".join(diagnostics)
    y = legend_y + 18
    for line in _wrap_text(diag_text):
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
        y += 12
        if y > height - 14:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, dpi=(int(dpi), int(dpi)))


def main() -> None:
    global np, torch, F, T, Image, ImageDraw, ImageFont
    global CrossViewTileDataset, compute_local_similarity_map, load_or_extract_record
    global DinoV2ExtractFeatures, seed_everything, device

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

    from configs import device
    from custom_datasets.cvgl_dataset import CrossViewTileDataset
    from cvgl_retrieval import compute_local_similarity_map, load_or_extract_record
    from utilities import DinoV2ExtractFeatures, seed_everything

    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CrossViewTileDataset.from_json(
        args.cvgl_dataset_root,
        split=args.data_split,
        tiles_manifest=args.cvgl_tiles_manifest,
        queries_manifest=args.cvgl_queries_manifest,
    )
    results = _load_json(args.results_json)
    row_map = _result_query_row_map(dataset, results)

    dino = DinoV2ExtractFeatures(
        args.model_type,
        args.desc_layer,
        args.desc_facet,
        device=str(device),
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
            device=device,
            cache_root=args.feature_cache_root,
            patch_stride=args.patch_stride,
        )
        tile_record = load_or_extract_record(
            dataset,
            tile_idx,
            dino=dino,
            device=device,
            cache_root=args.feature_cache_root,
            patch_stride=args.patch_stride,
        )

        sim_map = compute_local_similarity_map(
            query_record.local_desc,
            tile_record.local_desc,
            top_m=args.local_top_m,
        )
        pred_offset, pred_grid_xy, probs = _softargmax_offset(
            sim_map,
            extent_xy=extent_xy,
            temperature=args.temperature,
        )
        gt_grid_xy = _offset_to_grid_xy(gt_offset, extent_xy, sim_map.shape, flip_y=False)
        gt_flip_grid_xy = _offset_to_grid_xy(gt_offset, extent_xy, sim_map.shape, flip_y=True)
        gt_stats = _map_value_stats(sim_map, gt_grid_xy[0], gt_grid_xy[1])
        gt_flip_stats = _map_value_stats(sim_map, gt_flip_grid_xy[0], gt_flip_grid_xy[1])

        query_img = _center_crop_pil(dataset.get_image_paths()[query_global_idx], query_record.cropped_hw)
        tile_img = _center_crop_pil(dataset.get_image_paths()[tile_idx], tile_record.cropped_hw)

        diagnostics = []
        diagnostics.extend(_diagnose_query(dataset, query_idx, tile_idx, top1_idx, extent_xy))
        diagnostics.extend(extent_warnings)
        diagnostics.append(f"query_image_hw={query_record.image_hw}, query_cropped_hw={query_record.cropped_hw}, query_grid={query_record.grid_hw}")
        diagnostics.append(f"tile_image_hw={tile_record.image_hw}, tile_cropped_hw={tile_record.cropped_hw}, tile_grid={tile_record.grid_hw}")
        diagnostics.append(f"gt_offset={tuple(float(v) for v in gt_offset)}")
        diagnostics.append(f"pred_offset=({pred_offset[0]:.6g}, {pred_offset[1]:.6g})")
        diagnostics.append(f"gt_sim_percentile={gt_stats['percentile']}, gt_rank={gt_stats['rank']}")
        diagnostics.append(f"gt_flip_y_sim_percentile={gt_flip_stats['percentile']}, gt_flip_y_rank={gt_flip_stats['rank']}")

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
            probs=probs,
            pred_grid_xy=pred_grid_xy,
            gt_grid_xy=gt_grid_xy,
            gt_flip_grid_xy=gt_flip_grid_xy,
            title=title,
            diagnostics=diagnostics,
            dpi=args.dpi,
        )

        row = {
            "query_idx": int(query_idx),
            "tile_idx": int(tile_idx),
            "gt_tile_idx": int(gt_tile_idx),
            "top1_idx": None if top1_idx is None else int(top1_idx),
            "top1_is_gt": None if top1_idx is None else bool(int(top1_idx) == int(gt_tile_idx)),
            "extent_xy": [float(extent_xy[0]), float(extent_xy[1])],
            "gt_offset": [float(gt_offset[0]), float(gt_offset[1])],
            "pred_offset": [float(pred_offset[0]), float(pred_offset[1])],
            "pred_error_l2": float(np.linalg.norm(np.array(pred_offset) - np.array(gt_offset, dtype=float))),
            "gt_grid_xy": [float(gt_grid_xy[0]), float(gt_grid_xy[1])],
            "gt_flip_y_grid_xy": [float(gt_flip_grid_xy[0]), float(gt_flip_grid_xy[1])],
            "pred_grid_xy": [float(pred_grid_xy[0]), float(pred_grid_xy[1])],
            "gt_sim_value": gt_stats["value"],
            "gt_sim_rank": gt_stats["rank"],
            "gt_sim_percentile": gt_stats["percentile"],
            "gt_flip_y_sim_value": gt_flip_stats["value"],
            "gt_flip_y_sim_rank": gt_flip_stats["rank"],
            "gt_flip_y_sim_percentile": gt_flip_stats["percentile"],
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
