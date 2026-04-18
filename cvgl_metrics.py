"""
    Evaluation helpers for CVGL retrieval and localization.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def evaluate_cvgl_retrieval(
    retrieval_indices: np.ndarray,
    gt_db_indices: Sequence[Optional[int]],
    top_k_vals: Iterable[int],
) -> Dict[str, float]:
    if retrieval_indices.ndim != 2:
        raise ValueError("retrieval_indices must have shape [num_queries, k]")
    results: Dict[str, float] = {}
    valid_query_mask = np.array([gt is not None for gt in gt_db_indices], dtype=bool)
    num_valid = int(valid_query_mask.sum())
    if num_valid == 0:
        return {"Num-Retrieval-Queries": 0.0}
    top1_hits = []
    for k in top_k_vals:
        hits = 0
        for qi, gt_idx in enumerate(gt_db_indices):
            if gt_idx is None:
                continue
            preds = retrieval_indices[qi, :min(k, retrieval_indices.shape[1])]
            hits += int(int(gt_idx) in preds.tolist())
        results[f"R@{k}"] = hits / num_valid
    for qi, gt_idx in enumerate(gt_db_indices):
        if gt_idx is None:
            continue
        top1_hits.append(int(retrieval_indices[qi, 0] == int(gt_idx)))
    results["Top-1-Tile-Accuracy"] = float(np.mean(top1_hits)) if top1_hits else 0.0
    results["Num-Retrieval-Queries"] = float(num_valid)
    return results


def evaluate_offset_predictions(
    pred_offsets: Sequence[Optional[Tuple[float, float]]],
    gt_offsets: Sequence[Optional[Tuple[float, float]]],
    thresholds: Optional[Sequence[float]] = None,
    prefix: str = "Offset",
) -> Dict[str, float]:
    errors = []
    for pred, gt in zip(pred_offsets, gt_offsets):
        if pred is None or gt is None:
            continue
        pred_xy = np.array(pred, dtype=float)
        gt_xy = np.array(gt, dtype=float)
        errors.append(np.linalg.norm(pred_xy - gt_xy))
    if len(errors) == 0:
        return {}
    errors_np = np.asarray(errors, dtype=float)
    metrics = {
        f"{prefix}-Mean-Error": float(errors_np.mean()),
        f"{prefix}-Median-Error": float(np.median(errors_np)),
        f"{prefix}-Eval-Count": float(len(errors_np)),
    }
    if thresholds is not None:
        for thr in thresholds:
            metrics[f"{prefix}-SR@{thr:g}"] = float(np.mean(errors_np <= thr))
    return metrics


def evaluate_localization(
    dataset,
    top1_db_indices: Sequence[int],
    pred_offsets: Sequence[Optional[Tuple[float, float]]],
    thresholds: Optional[Sequence[float]] = None,
    query_indices: Optional[Sequence[int]] = None,
    db_indices_map: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    errors = []
    if query_indices is None:
        query_indices = list(range(len(top1_db_indices)))
    for dataset_qi, (pred_db_idx, pred_offset) in zip(query_indices, zip(top1_db_indices, pred_offsets)):
        gt_db_idx = dataset.get_query_gt_db_index(int(dataset_qi))
        gt_offset = dataset.get_query_gt_offset(int(dataset_qi))
        if gt_db_idx is None or gt_offset is None or pred_offset is None:
            continue
        dataset_pred_db_idx = int(pred_db_idx)
        if db_indices_map is not None:
            dataset_pred_db_idx = int(db_indices_map[dataset_pred_db_idx])
        pred_center = dataset.get_tile_center(dataset_pred_db_idx)
        gt_center = dataset.get_tile_center(int(gt_db_idx))
        if pred_center is not None and gt_center is not None:
            pred_point = np.array(pred_center, dtype=float) + np.array(pred_offset, dtype=float)
            gt_point = np.array(gt_center, dtype=float) + np.array(gt_offset, dtype=float)
        elif int(dataset_pred_db_idx) == int(gt_db_idx):
            pred_point = np.array(pred_offset, dtype=float)
            gt_point = np.array(gt_offset, dtype=float)
        else:
            continue
        errors.append(np.linalg.norm(pred_point - gt_point))
    if len(errors) == 0:
        return {}
    errors_np = np.asarray(errors, dtype=float)
    metrics = {
        "Localization-Mean-Error": float(errors_np.mean()),
        "Localization-Median-Error": float(np.median(errors_np)),
        "Localization-Eval-Count": float(len(errors_np)),
    }
    if thresholds is not None:
        for thr in thresholds:
            metrics[f"LSR@{thr:g}"] = float(np.mean(errors_np <= thr))
    return metrics
