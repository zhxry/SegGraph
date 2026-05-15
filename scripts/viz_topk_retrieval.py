#!/usr/bin/env python3
"""Visualize top-k CVGL retrieval results.

The figure has four rows by default:
  1. one top-1 correct wildfire example
  2. one top-1 correct tornado example
  3. one top-1 correct earthquake example
  4. one top-1 wrong example

Each row shows the query post-disaster image on the left and the top-5
pre-disaster tiles on the right. The ground-truth tile is highlighted in
green; all other retrieved tiles are highlighted in yellow.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageFont = None
    ImageOps = None


DISASTER_ORDER = ["wildfire", "tornado", "earthquake"]
DISASTER_LABELS = {
    "wildfire": "Wildfire",
    "tornado": "Tornado",
    "earthquake": "Earthquake",
    "error": "Error",
}


@dataclass
class ManifestItem:
    item_id: str
    image_relpath: str
    image_path: str
    gt_tile_id: Optional[str] = None


@dataclass
class ResultSource:
    result_path: str
    result: dict
    dataset_root: str
    disaster: Optional[str]
    tiles: List[ManifestItem]
    queries: List[ManifestItem]
    tile_id_to_index: Dict[str, int]
    tile_relpath_to_index: Dict[str, int]
    query_relpath_to_index: Dict[str, int]
    rankings: List[List[int]]


@dataclass
class Example:
    source: ResultSource
    result_query_index: int
    manifest_query_index: int
    gt_db_index: int
    top_db_indices: List[int]
    label: str


def _norm_relpath(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.realpath(path)
    return os.path.realpath(os.path.join(base_dir, path))


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_manifest(path: str) -> List[dict]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        for key in ["items", "tiles", "queries", "data"]:
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must decode to a list: {path}")
    return payload


def _manifest_paths(result: dict, dataset_root: str) -> Tuple[str, str]:
    resolved = result.get("Run-Config", {}).get("resolved", {})
    tiles_manifest = resolved.get("cvgl_tiles_manifest")
    queries_manifest = resolved.get("cvgl_queries_manifest")
    tiles_manifest = _resolve_path(tiles_manifest, dataset_root)
    queries_manifest = _resolve_path(queries_manifest, dataset_root)
    if tiles_manifest is None:
        tiles_manifest = os.path.join(dataset_root, "tiles.json")
    if queries_manifest is None:
        queries_manifest = os.path.join(dataset_root, "queries.json")
    return tiles_manifest, queries_manifest


def _load_manifest_items(dataset_root: str, result: dict) -> Tuple[List[ManifestItem], List[ManifestItem]]:
    tiles_manifest, queries_manifest = _manifest_paths(result, dataset_root)
    if not os.path.isfile(tiles_manifest):
        raise FileNotFoundError(f"Tiles manifest not found: {tiles_manifest}")
    if not os.path.isfile(queries_manifest):
        raise FileNotFoundError(f"Queries manifest not found: {queries_manifest}")

    tile_entries = _read_manifest(tiles_manifest)
    query_entries = _read_manifest(queries_manifest)

    tiles: List[ManifestItem] = []
    for i, entry in enumerate(tile_entries):
        image_relpath = entry.get("image") or entry.get("image_path")
        if image_relpath is None:
            raise KeyError(f"Tile entry missing image path: {entry}")
        item_id = str(entry.get("tile_id", f"tile_{i:06d}"))
        tiles.append(
            ManifestItem(
                item_id=item_id,
                image_relpath=_norm_relpath(str(image_relpath)),
                image_path=_resolve_path(str(image_relpath), dataset_root) or "",
            )
        )

    queries: List[ManifestItem] = []
    for i, entry in enumerate(query_entries):
        image_relpath = entry.get("image") or entry.get("image_path")
        if image_relpath is None:
            raise KeyError(f"Query entry missing image path: {entry}")
        gt_tile_id = entry.get("gt_tile_id")
        if gt_tile_id is None and entry.get("positives"):
            gt_tile_id = entry["positives"][0]
        queries.append(
            ManifestItem(
                item_id=str(entry.get("query_id", f"query_{i:06d}")),
                image_relpath=_norm_relpath(str(image_relpath)),
                image_path=_resolve_path(str(image_relpath), dataset_root) or "",
                gt_tile_id=None if gt_tile_id is None else str(gt_tile_id),
            )
        )
    return tiles, queries


def _detect_disaster(text: str) -> Optional[str]:
    lowered = text.lower()
    if "wildfire" in lowered or "fire" in lowered:
        return "wildfire"
    if "tornado" in lowered:
        return "tornado"
    if "earthquake" in lowered or "quake" in lowered:
        return "earthquake"
    return None


def _get_dataset_root(result_path: str, result: dict) -> str:
    resolved = result.get("Run-Config", {}).get("resolved", {})
    cwd = resolved.get("cwd") or os.getcwd()
    dataset_root = (
        resolved.get("cvgl_dataset_root")
        or result.get("CVGL-Dataset")
        or result.get("Dataset")
    )
    if dataset_root is None:
        raise ValueError(f"{result_path} does not contain a CVGL dataset root")
    return _resolve_path(str(dataset_root), cwd) or ""


def _rankings_from_result(result: dict) -> Tuple[str, List[List[int]]]:
    for key in ["Final-DB-Indices-Orig", "Final-Indices", "Coarse-Indices"]:
        value = result.get(key)
        if value is not None:
            return key, [[int(x) for x in row] for row in value]
    raise KeyError("Result JSON has no Final-DB-Indices-Orig, Final-Indices, or Coarse-Indices")


def _build_relpath_index(items: Sequence[ManifestItem]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for i, item in enumerate(items):
        index[_norm_relpath(item.image_relpath)] = i
        index[_norm_relpath(os.path.basename(item.image_relpath))] = i
    return index


def _source_from_result(result_path: str, disaster_override: Optional[str] = None) -> ResultSource:
    result_path = os.path.realpath(os.path.expanduser(result_path))
    result = _load_json(result_path)
    dataset_root = _get_dataset_root(result_path, result)
    tiles, queries = _load_manifest_items(dataset_root, result)
    ranking_key, raw_rankings = _rankings_from_result(result)

    tile_id_to_index = {tile.item_id: idx for idx, tile in enumerate(tiles)}
    tile_relpath_to_index = _build_relpath_index(tiles)
    query_relpath_to_index = _build_relpath_index(queries)

    db_relpaths = result.get("DB-Relpaths")
    rankings: List[List[int]] = []
    for row in raw_rankings:
        converted: List[int] = []
        for pred in row:
            db_index = int(pred)
            if ranking_key != "Final-DB-Indices-Orig" and db_relpaths is not None:
                if 0 <= db_index < len(db_relpaths):
                    relpath = _norm_relpath(str(db_relpaths[db_index]))
                    db_index = tile_relpath_to_index.get(relpath, db_index)
            converted.append(db_index)
        rankings.append(converted)

    disaster = disaster_override or _detect_disaster(
        " ".join([dataset_root, result.get("CVGL-Dataset", ""), result_path])
    )
    return ResultSource(
        result_path=result_path,
        result=result,
        dataset_root=dataset_root,
        disaster=disaster,
        tiles=tiles,
        queries=queries,
        tile_id_to_index=tile_id_to_index,
        tile_relpath_to_index=tile_relpath_to_index,
        query_relpath_to_index=query_relpath_to_index,
        rankings=rankings,
    )


def _query_manifest_index(source: ResultSource, result_query_index: int) -> Optional[int]:
    query_relpaths = source.result.get("Query-Relpaths")
    if query_relpaths is not None and result_query_index < len(query_relpaths):
        relpath = _norm_relpath(str(query_relpaths[result_query_index]))
        if relpath in source.query_relpath_to_index:
            return source.query_relpath_to_index[relpath]
        basename = _norm_relpath(os.path.basename(relpath))
        if basename in source.query_relpath_to_index:
            return source.query_relpath_to_index[basename]
    if result_query_index < len(source.queries):
        return result_query_index
    return None


def _gt_db_index(source: ResultSource, manifest_query_index: int) -> Optional[int]:
    if manifest_query_index >= len(source.queries):
        return None
    gt_tile_id = source.queries[manifest_query_index].gt_tile_id
    if gt_tile_id is None:
        return None
    return source.tile_id_to_index.get(gt_tile_id)


def _make_example(
    source: ResultSource,
    result_query_index: int,
    label: str,
    top_k: int,
) -> Optional[Example]:
    if result_query_index >= len(source.rankings):
        return None
    manifest_query_index = _query_manifest_index(source, result_query_index)
    if manifest_query_index is None:
        return None
    gt_db_index = _gt_db_index(source, manifest_query_index)
    if gt_db_index is None:
        return None
    top_db_indices = [
        int(idx)
        for idx in source.rankings[result_query_index][:top_k]
        if 0 <= int(idx) < len(source.tiles)
    ]
    if len(top_db_indices) < top_k:
        return None
    return Example(
        source=source,
        result_query_index=result_query_index,
        manifest_query_index=manifest_query_index,
        gt_db_index=int(gt_db_index),
        top_db_indices=top_db_indices,
        label=label,
    )


def _iter_examples(source: ResultSource, label: str, top_k: int) -> Iterable[Example]:
    for result_query_index in range(len(source.rankings)):
        example = _make_example(source, result_query_index, label, top_k)
        if example is not None:
            yield example


def _select_correct_example(
    sources: Sequence[ResultSource],
    disaster: str,
    top_k: int,
    query_index: Optional[int] = None,
) -> Example:
    matching_sources = [src for src in sources if src.disaster == disaster]
    if not matching_sources:
        raise ValueError(f"No result JSON found for disaster type: {disaster}")

    label = DISASTER_LABELS[disaster]
    if query_index is not None:
        for source in matching_sources:
            example = _make_example(source, query_index, label, top_k)
            if example is not None and example.top_db_indices[0] == example.gt_db_index:
                return example
        raise ValueError(
            f"Query index {query_index} is invalid or not top-1 correct "
            f"for disaster type {disaster}"
        )

    for source in matching_sources:
        for example in _iter_examples(source, label, top_k):
            if example.top_db_indices[0] == example.gt_db_index:
                return example
    raise ValueError(f"No top-1 correct example found for disaster type: {disaster}")


def _select_error_example(
    sources: Sequence[ResultSource],
    top_k: int,
    query_index: Optional[int] = None,
) -> Example:
    label = DISASTER_LABELS["error"]
    if query_index is not None:
        for source in sources:
            example = _make_example(source, query_index, label, top_k)
            if example is not None and example.top_db_indices[0] != example.gt_db_index:
                return example
        raise ValueError(f"Query index {query_index} is invalid or not top-1 wrong")

    first_wrong: Optional[Example] = None
    for source in sources:
        for example in _iter_examples(source, label, top_k):
            if example.top_db_indices[0] != example.gt_db_index:
                if example.gt_db_index in example.top_db_indices:
                    return example
                if first_wrong is None:
                    first_wrong = example
    if first_wrong is not None:
        return first_wrong
    raise ValueError("No top-1 wrong example found")


def _font(size: int) -> ImageFont.ImageFont:
    _require_pillow()
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _require_pillow() -> None:
    if Image is None:
        raise ModuleNotFoundError(
            "Pillow is required to render the figure. "
            "Install pillow or run this script in the project environment that has PIL."
        )


def _fit_image(path: str, size: int, sharpen: bool) -> Image.Image:
    _require_pillow()
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if sharpen and ImageFilter is not None:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    return image


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y0: int,
    y1: int,
    fill: Tuple[int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    y = y0
    while y < y1:
        draw.line((x, y, x, min(y + dash, y1)), fill=fill, width=width)
        y += dash + gap


def _draw_border(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    for offset in range(width):
        draw.rectangle(
            (xy[0] + offset, xy[1] + offset, xy[2] - offset, xy[3] - offset),
            outline=color,
        )


def _compose_figure(
    examples: Sequence[Example],
    output_path: str,
    top_k: int,
    tile_size: int,
    gap: int,
    query_to_sep_gap: int,
    sep_to_tiles_gap: int,
    margin: int,
    row_gap: int,
    header_h: int,
    footer_h: int,
    border_width: int,
    output_scale: float,
    dpi: int,
    sharpen: bool,
    show_headers: bool,
    show_row_labels: bool,
    save_pdf: bool,
    pdf_output: Optional[str],
) -> Tuple[Path, Optional[Path]]:
    _require_pillow()
    scale = max(float(output_scale), 0.1)

    def s(value: int) -> int:
        return max(1, int(round(value * scale)))

    tile_size = s(tile_size)
    gap = s(gap)
    query_to_sep_gap = s(query_to_sep_gap)
    sep_to_tiles_gap = s(sep_to_tiles_gap)
    margin = s(margin)
    row_gap = s(row_gap)
    header_h = s(header_h)
    footer_h = s(footer_h)
    border_width = s(border_width)
    shadow_offset = s(2)

    query_w = tile_size
    topk_w = top_k * tile_size + (top_k - 1) * gap
    sep_x = margin + query_w + query_to_sep_gap
    tiles_x0 = sep_x + sep_to_tiles_gap
    canvas_w = tiles_x0 + topk_w + shadow_offset + margin
    row_h = tile_size + shadow_offset
    canvas_h = margin + header_h + len(examples) * row_h + (len(examples) - 1) * row_gap + footer_h + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = _font(s(22))
    label_font = _font(s(18))
    footer_font = _font(s(24))

    if show_headers:
        y0 = margin
        _draw_centered_text(
            draw,
            (margin, y0, margin + query_w, y0 + header_h),
            "Query",
            header_font,
            (20, 20, 20),
        )
        _draw_centered_text(
            draw,
            (tiles_x0, y0, tiles_x0 + topk_w, y0 + header_h),
            f"Top-{top_k} pre-disaster retrievals",
            header_font,
            (20, 20, 20),
        )

    green = (0, 220, 35)
    yellow = (238, 178, 0)
    shadow = (170, 170, 170)
    y = margin + header_h
    for example in examples:
        query_path = example.source.queries[example.manifest_query_index].image_path
        query_img = _fit_image(query_path, tile_size, sharpen=sharpen)
        canvas.paste(query_img, (margin, y))

        if show_row_labels:
            text_bbox = draw.textbbox((0, 0), example.label, font=label_font)
            label_w = text_bbox[2] - text_bbox[0] + s(10)
            label_h = text_bbox[3] - text_bbox[1] + s(8)
            draw.rectangle((margin, y, margin + label_w, y + label_h), fill=(255, 255, 255))
            draw.text((margin + s(5), y + s(4)), example.label, font=label_font, fill=(25, 25, 25))

        for rank, db_index in enumerate(example.top_db_indices[:top_k]):
            tile_path = example.source.tiles[db_index].image_path
            tile_img = _fit_image(tile_path, tile_size, sharpen=sharpen)
            x = tiles_x0 + rank * (tile_size + gap)
            draw.rectangle(
                (
                    x + shadow_offset,
                    y + shadow_offset,
                    x + tile_size - 1 + shadow_offset,
                    y + tile_size - 1 + shadow_offset,
                ),
                fill=shadow,
            )
            canvas.paste(tile_img, (x, y))
            color = green if db_index == example.gt_db_index else yellow
            _draw_border(draw, (x, y, x + tile_size - 1, y + tile_size - 1), color, border_width)

        y += row_h + row_gap

    line_y0 = margin
    line_y1 = canvas_h - margin - footer_h
    _draw_dashed_line(draw, sep_x, line_y0, line_y1, fill=(95, 95, 95), width=s(2), dash=s(18), gap=s(10))

    footer_y0 = canvas_h - margin - footer_h
    _draw_centered_text(
        draw,
        (margin, footer_y0, margin + query_w, footer_y0 + footer_h),
        "Query",
        footer_font,
        (20, 20, 20),
    )
    _draw_centered_text(
        draw,
        (tiles_x0, footer_y0, tiles_x0 + tile_size, footer_y0 + footer_h),
        "R@1",
        footer_font,
        (20, 20, 20),
    )
    _draw_centered_text(
        draw,
        (tiles_x0 + topk_w - tile_size, footer_y0, tiles_x0 + topk_w, footer_y0 + footer_h),
        f"R@{top_k}",
        footer_font,
        (20, 20, 20),
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(int(dpi), int(dpi)))
    pdf_path: Optional[Path] = None
    if save_pdf:
        pdf_path = Path(pdf_output) if pdf_output is not None else out_path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(pdf_path, "PDF", resolution=float(dpi))
    return out_path, pdf_path


def _parse_labeled_result(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--labeled-result must have form disaster=path.json, got: {value}")
    disaster, path = value.split("=", 1)
    disaster = disaster.strip().lower()
    if disaster not in DISASTER_ORDER:
        raise ValueError(f"Unknown disaster label '{disaster}'. Expected one of: {DISASTER_ORDER}")
    if not path.strip():
        raise ValueError(f"Missing path in --labeled-result {value}")
    return disaster, path.strip()


def _collect_result_specs(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
    result_specs: List[Tuple[str, Optional[str]]] = []
    for value in args.labeled_result or []:
        disaster, path = _parse_labeled_result(value)
        result_specs.append((path, disaster))
    for path in args.result_json or []:
        result_specs.append((path, None))
    for pattern in args.result_glob or []:
        result_specs.extend((path, None) for path in glob.glob(pattern, recursive=True))
    if not result_specs:
        default_patterns = [
            ".cache/experiments/**/cvgl_results_*.json",
            ".cache/backbone_ablation/experiments/**/cvgl_backbone_ablation_*.json",
        ]
        for pattern in default_patterns:
            result_specs.extend((path, None) for path in glob.glob(pattern, recursive=True))
    normalized_specs = sorted({
        (os.path.realpath(os.path.expanduser(path)), disaster)
        for path, disaster in result_specs
    })
    if not normalized_specs:
        raise FileNotFoundError("No result JSON files found. Pass --result-json or --result-glob.")
    return normalized_specs


def _load_sources(specs: Sequence[Tuple[str, Optional[str]]]) -> List[ResultSource]:
    sources: List[ResultSource] = []
    errors: List[str] = []
    for path, disaster_override in specs:
        try:
            sources.append(_source_from_result(path, disaster_override=disaster_override))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if not sources:
        raise RuntimeError("Failed to load every result JSON:\n" + "\n".join(errors))
    if errors:
        print("[warn] skipped result files:")
        for error in errors:
            print(f"  {error}")
    return sources


def _query_index_overrides(args: argparse.Namespace) -> Dict[str, Optional[int]]:
    return {
        "wildfire": args.wildfire_query_index,
        "tornado": args.tornado_query_index,
        "earthquake": args.earthquake_query_index,
        "error": args.error_query_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        action="append",
        default=[],
        help="Saved cvgl_results_*.json path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--labeled-result",
        action="append",
        default=[],
        help=(
            "Saved result with explicit disaster type, e.g. "
            "wildfire=.cache/.../cvgl_results.json. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--result-glob",
        action="append",
        default=[],
        help="Glob for saved result JSON files, e.g. '.cache/experiments/**/cvgl_results_*.json'.",
    )
    parser.add_argument("--output", default="top-k_viz.png", help="Output figure path.")
    parser.add_argument(
        "--pdf-output",
        default=None,
        help="Optional PDF output path. Defaults to the image output path with .pdf suffix.",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Disable saving a PDF copy.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved tiles to show.")
    parser.add_argument("--tile-size", type=int, default=150, help="Square image size in pixels.")
    parser.add_argument(
        "--output-scale",
        type=float,
        default=2.0,
        help="Render scale multiplier for a sharper high-resolution output.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata written to the output image.")
    parser.add_argument("--no-sharpen", action="store_true", help="Disable light unsharp-mask sharpening.")
    parser.add_argument("--gap", type=int, default=10, help="Horizontal gap between top-k tiles.")
    parser.add_argument("--row-gap", type=int, default=20, help="Vertical gap between rows.")
    parser.add_argument("--margin", type=int, default=8, help="Outer figure margin.")
    parser.add_argument("--query-to-sep-gap", type=int, default=20, help="Gap from query image to divider.")
    parser.add_argument("--sep-to-tiles-gap", type=int, default=20, help="Gap from divider to top-k tiles.")
    parser.add_argument("--header-height", type=int, default=36, help="Top label area height.")
    parser.add_argument("--footer-height", type=int, default=36, help="Bottom label area height.")
    parser.add_argument("--border-width", type=int, default=4, help="Retrieval tile border width.")
    parser.add_argument("--no-headers", action="store_true", help="Do not draw top text headers.")
    parser.add_argument("--no-row-labels", action="store_true", help="Do not draw row labels.")
    parser.add_argument("--wildfire-query-index", type=int, default=None)
    parser.add_argument("--tornado-query-index", type=int, default=None)
    parser.add_argument("--earthquake-query-index", type=int, default=None)
    parser.add_argument("--error-query-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.output_scale <= 0:
        raise ValueError("--output-scale must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    result_specs = _collect_result_specs(args)
    sources = _load_sources(result_specs)
    overrides = _query_index_overrides(args)

    examples: List[Example] = []
    for disaster in DISASTER_ORDER:
        examples.append(
            _select_correct_example(
                sources,
                disaster=disaster,
                top_k=args.top_k,
                query_index=overrides[disaster],
            )
        )
    examples.append(
        _select_error_example(
            sources,
            top_k=args.top_k,
            query_index=overrides["error"],
        )
    )

    canvas_path, pdf_path = _compose_figure(
        examples=examples,
        output_path=args.output,
        top_k=args.top_k,
        tile_size=args.tile_size,
        gap=args.gap,
        query_to_sep_gap=args.query_to_sep_gap,
        sep_to_tiles_gap=args.sep_to_tiles_gap,
        margin=args.margin,
        row_gap=args.row_gap,
        header_h=0 if args.no_headers else args.header_height,
        footer_h=args.footer_height,
        border_width=args.border_width,
        output_scale=args.output_scale,
        dpi=args.dpi,
        sharpen=not args.no_sharpen,
        show_headers=not args.no_headers,
        show_row_labels=not args.no_row_labels,
        save_pdf=not args.no_pdf,
        pdf_output=args.pdf_output,
    )

    print(f"[ok] saved {canvas_path}")
    if pdf_path is not None:
        print(f"[ok] saved {pdf_path}")
    for example in examples:
        query = example.source.queries[example.manifest_query_index]
        gt_tile = example.source.tiles[example.gt_db_index]
        print(
            f"{example.label}: query={query.item_id} "
            f"gt={gt_tile.item_id} top5={example.top_db_indices[:args.top_k]} "
            f"source={Path(example.source.result_path).name}"
        )


if __name__ == "__main__":
    main()
