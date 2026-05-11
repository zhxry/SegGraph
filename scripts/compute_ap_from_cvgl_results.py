#!/usr/bin/env python
"""Compute retrieval AP metrics from a saved CVGL results JSON file."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


def _resolve_path(path: Optional[str], cwd: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    if cwd is not None:
        return os.path.abspath(os.path.join(cwd, path))
    return os.path.abspath(path)


def _get_nested(mapping: Dict, keys: Sequence[str], default=None):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_manifest(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        for key in ["items", "tiles", "queries", "data"]:
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must decode to a list: {path}")
    return payload


def _resolve_cvgl_manifests(
    dataset_root: str,
    split: str,
    tiles_manifest: Optional[str],
    queries_manifest: Optional[str],
) -> tuple[str, str]:
    dataset_root = os.path.realpath(os.path.expanduser(dataset_root))
    if tiles_manifest is None:
        split_tiles = os.path.join(dataset_root, f"tiles_{split}.json")
        tiles_manifest = split_tiles if os.path.isfile(split_tiles) else os.path.join(dataset_root, "tiles.json")
    if queries_manifest is None:
        split_queries = os.path.join(dataset_root, f"queries_{split}.json")
        queries_manifest = split_queries if os.path.isfile(split_queries) else os.path.join(dataset_root, "queries.json")
    if not os.path.isfile(tiles_manifest):
        raise FileNotFoundError(f"Tiles manifest not found: {tiles_manifest}")
    if not os.path.isfile(queries_manifest):
        raise FileNotFoundError(f"Queries manifest not found: {queries_manifest}")
    return tiles_manifest, queries_manifest


def _load_cvgl_positive_sets(
    dataset_root: str,
    split: str,
    tiles_manifest: Optional[str],
    queries_manifest: Optional[str],
) -> tuple[int, List[Set[int]]]:
    tiles_manifest, queries_manifest = _resolve_cvgl_manifests(
        dataset_root,
        split,
        tiles_manifest,
        queries_manifest,
    )
    tile_entries = _read_manifest(tiles_manifest)
    query_entries = _read_manifest(queries_manifest)
    tile_id_to_index = {
        str(entry.get("tile_id", f"tile_{idx:06d}")): idx
        for idx, entry in enumerate(tile_entries)
    }

    positive_sets: List[Set[int]] = []
    for idx, entry in enumerate(query_entries):
        query_id = str(entry.get("query_id", f"query_{idx:06d}"))
        positives: Set[int] = set()
        gt_tile_id = entry.get("gt_tile_id")
        if gt_tile_id is None and "positives" in entry and len(entry["positives"]) > 0:
            gt_tile_id = entry["positives"][0]
        if gt_tile_id is not None:
            gt_tile_id = str(gt_tile_id)
            if gt_tile_id not in tile_id_to_index:
                raise KeyError(f"Query {query_id} gt_tile_id {gt_tile_id} not found in tiles")
            positives.add(tile_id_to_index[gt_tile_id])
        for tile_id in entry.get("positives", []) or []:
            tile_id = str(tile_id)
            if tile_id in tile_id_to_index:
                positives.add(tile_id_to_index[tile_id])
        positive_sets.append(positives)
    return len(tile_entries), positive_sets


def _load_gt_positive_sets(results: Dict) -> List[Set[int]]:
    resolved = _get_nested(results, ["Run-Config", "resolved"], {})
    cli_args = _get_nested(results, ["Run-Config", "cli_args"], {})
    task_mode = resolved.get("task_mode", results.get("Task-Mode", "cvgl"))
    if task_mode != "cvgl":
        raise ValueError("This script currently reconstructs GT for CVGL result files only.")

    cwd = resolved.get("cwd")
    dataset_root = _resolve_path(resolved.get("cvgl_dataset_root"), cwd)
    tiles_manifest = _resolve_path(resolved.get("cvgl_tiles_manifest"), cwd)
    queries_manifest = _resolve_path(resolved.get("cvgl_queries_manifest"), cwd)
    split = str(cli_args.get("data_split", "test"))
    sub_sample_db = int(cli_args.get("sub_sample_db", 1))
    sub_sample_qu = int(cli_args.get("sub_sample_qu", 1))

    if dataset_root is None:
        raise ValueError("Result JSON does not contain Run-Config.resolved.cvgl_dataset_root")
    database_num, positive_sets = _load_cvgl_positive_sets(
        dataset_root,
        split,
        tiles_manifest,
        queries_manifest,
    )

    db_indices = np.arange(0, database_num, sub_sample_db)
    qu_indices = np.arange(0, len(positive_sets), sub_sample_qu)
    db_index_map = {int(orig_idx): bank_idx for bank_idx, orig_idx in enumerate(db_indices.tolist())}

    gt_sets: List[Set[int]] = []
    for dataset_qi in qu_indices.tolist():
        positives = positive_sets[int(dataset_qi)]
        gt_sets.append({db_index_map[int(pos)] for pos in positives if int(pos) in db_index_map})
    return gt_sets


def _average_precision_at_k(preds: Sequence[int], positives: Set[int], k: Optional[int] = None) -> Optional[float]:
    if len(positives) == 0:
        return None
    if k is None:
        k = len(preds)
    preds_k = [int(x) for x in preds[:k]]
    hits = 0
    precision_sum = 0.0
    for rank, pred in enumerate(preds_k, start=1):
        if pred in positives:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(positives), k)


def _single_positive_ap_at_k(preds: Sequence[int], positive: int, k: Optional[int] = None) -> float:
    if k is None:
        k = len(preds)
    for rank, pred in enumerate(preds[:k], start=1):
        if int(pred) == int(positive):
            return 1.0 / rank
    return 0.0


def _summarize_ap(
    rankings: Sequence[Sequence[int]],
    gt_sets: Sequence[Set[int]],
    prefix: str,
    k_values: Iterable[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    valid = [(preds, positives) for preds, positives in zip(rankings, gt_sets) if len(positives) > 0]
    metrics[f"{prefix}-Num-Queries"] = float(len(valid))
    if len(valid) == 0:
        return metrics

    for k in k_values:
        k = int(k)
        all_positive_aps = []
        first_positive_aps = []
        for preds, positives in valid:
            ap = _average_precision_at_k(preds, positives, k=k)
            if ap is not None:
                all_positive_aps.append(ap)
            first_positive_aps.append(_single_positive_ap_at_k(preds, min(positives), k=k))
        metrics[f"{prefix}-mAP@{k}"] = float(np.mean(all_positive_aps))
        metrics[f"{prefix}-SingleGT-mAP@{k}"] = float(np.mean(first_positive_aps))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="Path to cvgl_results_*.json")
    parser.add_argument(
        "--k",
        nargs="*",
        type=int,
        default=None,
        help="AP cutoffs. Defaults to the saved ranking lengths for each ranking field.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write metrics to <results_json>.ap.json",
    )
    args = parser.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    gt_sets = _load_gt_positive_sets(results)

    all_metrics: Dict[str, float] = {}
    for field in ["Coarse-Indices", "Final-Indices", "Final-DB-Indices-Orig"]:
        rankings = results.get(field)
        if rankings is None:
            continue
        max_saved_k = min(len(row) for row in rankings) if len(rankings) > 0 else 0
        k_values = args.k if args.k is not None and len(args.k) > 0 else [max_saved_k]
        k_values = [k for k in k_values if k <= max_saved_k]
        if len(k_values) == 0:
            continue
        all_metrics.update(_summarize_ap(rankings, gt_sets, field, k_values))

    print(json.dumps(all_metrics, indent=2, sort_keys=True))
    if args.write_json:
        out_path = f"{args.results_json}.ap.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, sort_keys=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
