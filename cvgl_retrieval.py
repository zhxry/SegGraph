"""
    CVGL feature extraction, coarse retrieval, local reranking and
    similarity-map offset estimation.
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm.auto import tqdm

from segvlad_masking import (
    SAMGenerator,
    SegmentorConfig,
    build_mask_adjacency_revisit,
    load_or_generate_masks,
    project_masks_to_patch_grid,
)
from utilities import DinoV2ExtractFeatures, VLAD


@dataclass
class DenseFeatureRecord:
    local_desc: torch.Tensor
    grid_hw: Tuple[int, int]
    image_hw: Tuple[int, int]
    cropped_hw: Tuple[int, int]
    patch_stride: int
    global_desc: Optional[torch.Tensor] = None


@dataclass
class FeatureBank:
    indices: List[int]
    relpaths: List[str]
    global_descs: torch.Tensor
    local_descs: List[torch.Tensor]
    grid_hws: List[Tuple[int, int]]
    image_hws: List[Tuple[int, int]]
    cropped_hws: List[Tuple[int, int]]
    patch_stride: int


@dataclass
class SegVLADFeatureBank:
    indices: List[int]
    relpaths: List[str]
    local_descs: List[torch.Tensor]
    grid_hws: List[Tuple[int, int]]
    image_hws: List[Tuple[int, int]]
    cropped_hws: List[Tuple[int, int]]
    patch_stride: int
    image_segment_descs: List[torch.Tensor]
    image_segment_counts: List[int]
    segment_descs: torch.Tensor
    segment_to_image: torch.Tensor
    segment_to_image_pos: torch.Tensor


def _sanitize_cache_id(cache_id: str) -> str:
    return cache_id.replace("\\", "/")


def _center_crop_masks(
    masks: torch.Tensor,
    image_hw: Tuple[int, int],
    cropped_hw: Tuple[int, int],
) -> torch.Tensor:
    h, w = image_hw
    crop_h, crop_w = cropped_hw
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    return masks[:, top:top + crop_h, left:left + crop_w]


def _fallback_full_mask(cropped_hw: Tuple[int, int]) -> torch.Tensor:
    return torch.ones((1, cropped_hw[0], cropped_hw[1]), dtype=torch.bool)


def masks_to_patch_grid(
    masks: Optional[Sequence[np.ndarray]],
    image_hw: Tuple[int, int],
    cropped_hw: Tuple[int, int],
    grid_hw: Tuple[int, int],
    min_area_ratio: float = 0.0,
) -> torch.Tensor:
    if masks is None or len(masks) == 0:
        return torch.ones((1, grid_hw[0] * grid_hw[1]), dtype=torch.bool)
    mask_tensor = torch.as_tensor(np.asarray(masks), dtype=torch.float32)
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if tuple(mask_tensor.shape[-2:]) != tuple(image_hw):
        raise ValueError(
            f"Mask/image size mismatch: masks {tuple(mask_tensor.shape[-2:])} vs image {image_hw}"
        )
    mask_tensor = _center_crop_masks(mask_tensor, image_hw=image_hw, cropped_hw=cropped_hw)
    lowres = F.interpolate(
        mask_tensor.unsqueeze(1),
        size=grid_hw,
        mode="nearest",
    ).squeeze(1) > 0.5
    flat = lowres.reshape(lowres.shape[0], -1)
    keep = flat.any(dim=1)
    if min_area_ratio > 0.0:
        keep = keep & (flat.float().mean(dim=1) >= float(min_area_ratio))
    flat = flat[keep]
    if flat.shape[0] == 0:
        return torch.ones((1, grid_hw[0] * grid_hw[1]), dtype=torch.bool)
    return flat


def build_mask_adjacency(
    mask_grid: torch.Tensor,
    grid_hw: Tuple[int, int],
    order: int = 0,
) -> Optional[torch.Tensor]:
    if order <= 0 or mask_grid.shape[0] <= 1:
        return None
    mask_float = mask_grid.float()
    num_masks, num_patches = mask_float.shape
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != num_patches:
        raise ValueError(f"grid_hw {grid_hw} is incompatible with {num_patches} patches")
    ys = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w)
    xs = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h)
    denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroids = torch.stack([
        (mask_float * xs.unsqueeze(0)).sum(dim=1) / denom.squeeze(1),
        (mask_float * ys.unsqueeze(0)).sum(dim=1) / denom.squeeze(1),
    ], dim=1)
    k = min(4, num_masks)
    dists = torch.cdist(centroids, centroids)
    knn = dists.topk(k=k, largest=False).indices
    adj = torch.zeros((num_masks, num_masks), dtype=torch.bool)
    for i in range(num_masks):
        adj[i, knn[i]] = True
    adj = adj | adj.T
    adj.fill_diagonal_(True)
    adj_power = adj.float()
    for _ in range(1, order):
        adj_power = adj_power @ adj.float()
    return adj_power > 0


def segvlad_descriptors(
    patch_descs: torch.Tensor,
    c_centers: torch.Tensor,
    mask_grid: torch.Tensor,
    adjacency: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if patch_descs.ndim != 2:
        raise ValueError(f"patch_descs must be [num_patches, desc_dim], got {tuple(patch_descs.shape)}")
    if mask_grid.ndim != 2:
        raise ValueError(f"mask_grid must be [num_segments, num_patches], got {tuple(mask_grid.shape)}")
    if patch_descs.shape[0] != mask_grid.shape[1]:
        raise ValueError(
            f"Patch/mask mismatch: {patch_descs.shape[0]} patches vs {mask_grid.shape[1]} mask entries"
        )
    centers = F.normalize(c_centers.float(), dim=1)
    descs = F.normalize(patch_descs.float(), dim=1)
    labels = torch.argmax(descs @ centers.T, dim=1)
    residuals = descs - c_centers[labels].float()
    if adjacency is None:
        adjacency = torch.eye(mask_grid.shape[0], dtype=torch.bool)

    vlads = []
    mask_weights = mask_grid.bool()
    for cluster_idx in range(c_centers.shape[0]):
        inds = torch.where(labels == cluster_idx)[0]
        if inds.numel() == 0:
            vlads.append(
                torch.zeros((mask_grid.shape[0], patch_descs.shape[1]), dtype=torch.float32)
            )
            continue
        cluster_masks = (adjacency.float() @ mask_weights[:, inds].float()) > 0
        cluster_vlad = cluster_masks.float() @ residuals[inds]
        cluster_vlad = F.normalize(cluster_vlad, dim=1)
        vlads.append(cluster_vlad)
    stacked = torch.stack(vlads, dim=1).reshape(mask_grid.shape[0], -1)
    return F.normalize(stacked, dim=1)


def gem_pool_descriptors(
    patch_descs: torch.Tensor,
    gem_p: float = 3.0,
    gem_use_abs: bool = False,
    gem_elem_by_elem: bool = False,
) -> torch.Tensor:
    if patch_descs.ndim != 2:
        raise ValueError("patch_descs must have shape [num_patches, desc_dim]")
    if gem_use_abs:
        return torch.mean(torch.abs(patch_descs) ** gem_p, dim=0) ** (1.0 / gem_p)
    if gem_elem_by_elem:
        x = torch.mean(patch_descs ** gem_p, dim=0)
        x = x.to(torch.complex64) ** (1.0 / gem_p)
        return torch.abs(x) * torch.sign(torch.real(torch.mean(patch_descs ** gem_p, dim=0)))
    x = torch.mean(patch_descs ** gem_p, dim=0)
    x = x.to(torch.complex64) ** (1.0 / gem_p)
    return torch.abs(x) * torch.sign(torch.real(torch.mean(patch_descs ** gem_p, dim=0)))


def extract_dense_record(
    img: torch.Tensor,
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    patch_stride: int = 14,
    global_agg: Literal["vlad", "gem"] = "vlad",
    gem_p: float = 3.0,
    gem_use_abs: bool = False,
    gem_elem_by_elem: bool = False,
) -> DenseFeatureRecord:
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(img.shape)}")
    _, h, w = img.shape
    h_new, w_new = (h // patch_stride) * patch_stride, (w // patch_stride) * patch_stride
    if h_new <= 0 or w_new <= 0:
        raise ValueError(f"Invalid cropped size {(h_new, w_new)} from input {(h, w)}")
    img_in = T.CenterCrop((h_new, w_new))(img)[None, ...].to(device)
    patch_descs = dino(img_in)[0].detach().cpu()
    grid_hw = (h_new // patch_stride, w_new // patch_stride)
    if patch_descs.shape[0] != grid_hw[0] * grid_hw[1]:
        raise ValueError(
            f"Patch/grid mismatch: {patch_descs.shape[0]} vs {grid_hw[0]}*{grid_hw[1]}"
        )
    local_desc = patch_descs.reshape(grid_hw[0], grid_hw[1], -1)
    global_desc = None
    if global_agg == "gem":
        global_desc = F.normalize(
            gem_pool_descriptors(
                patch_descs, gem_p=gem_p,
                gem_use_abs=gem_use_abs,
                gem_elem_by_elem=gem_elem_by_elem,
            ),
            dim=0,
        )
    return DenseFeatureRecord(
        local_desc=local_desc,
        grid_hw=grid_hw,
        image_hw=(h, w),
        cropped_hw=(h_new, w_new),
        patch_stride=patch_stride,
        global_desc=global_desc,
    )


def _feature_cache_path(cache_root: Optional[str], relpath: str) -> Optional[str]:
    if cache_root is None:
        return None
    cache_id = _sanitize_cache_id(relpath)
    return os.path.join(cache_root, f"{cache_id}.pt")


def load_or_extract_record(
    dataset,
    index: int,
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    cache_root: Optional[str] = None,
    global_agg: Literal["vlad", "gem"] = "vlad",
    gem_p: float = 3.0,
    gem_use_abs: bool = False,
    gem_elem_by_elem: bool = False,
    patch_stride: int = 14,
) -> DenseFeatureRecord:
    relpath = dataset.get_image_relpaths(index)
    cache_path = _feature_cache_path(cache_root, relpath)
    if cache_path is not None and os.path.isfile(cache_path):
        payload = torch.load(cache_path)
        return DenseFeatureRecord(
            local_desc=payload["local_desc"],
            grid_hw=tuple(payload["grid_hw"]),
            image_hw=tuple(payload["image_hw"]),
            cropped_hw=tuple(payload["cropped_hw"]),
            patch_stride=int(payload["patch_stride"]),
            global_desc=payload.get("global_desc"),
        )
    img, _ = dataset[index]
    record = extract_dense_record(
        img, dino=dino, device=device, patch_stride=patch_stride,
        global_agg=global_agg, gem_p=gem_p,
        gem_use_abs=gem_use_abs, gem_elem_by_elem=gem_elem_by_elem,
    )
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "local_desc": record.local_desc,
            "grid_hw": list(record.grid_hw),
            "image_hw": list(record.image_hw),
            "cropped_hw": list(record.cropped_hw),
            "patch_stride": record.patch_stride,
            "global_desc": record.global_desc,
        }, cache_path)
    return record


def fit_vlad_for_dataset(
    dataset,
    db_indices: Sequence[int],
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    num_clusters: int,
    sub_sample_db_vlad: int = 1,
    vlad_assignment: str = "hard",
    vlad_soft_temp: float = 1.0,
    feature_cache_root: Optional[str] = None,
    vlad_cache_root: Optional[str] = None,
    patch_stride: int = 14,
    verbose: bool = True,
) -> VLAD:
    vlad = VLAD(
        num_clusters,
        None,
        vlad_mode=vlad_assignment,
        soft_temp=vlad_soft_temp,
        cache_dir=vlad_cache_root,
    )
    if vlad.can_use_cache_vlad():
        vlad.fit(None)
        return vlad
    train_descs = []
    fit_indices = list(db_indices)[::sub_sample_db_vlad]
    iterator = tqdm(fit_indices, disable=not verbose, desc="VLAD clusters")
    for idx in iterator:
        record = load_or_extract_record(
            dataset, idx, dino=dino, device=device,
            cache_root=feature_cache_root, global_agg="vlad",
            patch_stride=patch_stride,
        )
        train_descs.append(record.local_desc.reshape(-1, record.local_desc.shape[-1]))
    if len(train_descs) == 0:
        raise ValueError("No descriptors collected for VLAD fitting")
    vlad.fit(torch.cat(train_descs, dim=0))
    return vlad


def build_feature_bank(
    dataset,
    indices: Sequence[int],
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    global_agg: Literal["vlad", "gem"] = "vlad",
    vlad: Optional[VLAD] = None,
    feature_cache_root: Optional[str] = None,
    gem_p: float = 3.0,
    gem_use_abs: bool = False,
    gem_elem_by_elem: bool = False,
    patch_stride: int = 14,
    verbose: bool = True,
) -> FeatureBank:
    all_globals = []
    all_locals = []
    grid_hws = []
    image_hws = []
    cropped_hws = []
    relpaths = []
    iterator = tqdm(indices, disable=not verbose, desc="Dense features")
    for idx in iterator:
        record = load_or_extract_record(
            dataset, idx, dino=dino, device=device,
            cache_root=feature_cache_root, global_agg=global_agg,
            gem_p=gem_p, gem_use_abs=gem_use_abs,
            gem_elem_by_elem=gem_elem_by_elem, patch_stride=patch_stride,
        )
        local_flat = record.local_desc.reshape(-1, record.local_desc.shape[-1])
        relpath = dataset.get_image_relpaths(idx)
        if global_agg == "vlad":
            if vlad is None:
                raise ValueError("VLAD aggregator requested but no VLAD instance provided")
            global_desc = vlad.generate(local_flat, relpath).cpu()
        else:
            global_desc = record.global_desc
            if global_desc is None:
                raise ValueError("GeM aggregation requested but global_desc is missing")
        all_globals.append(F.normalize(global_desc, dim=0))
        all_locals.append(record.local_desc)
        grid_hws.append(record.grid_hw)
        image_hws.append(record.image_hw)
        cropped_hws.append(record.cropped_hw)
        relpaths.append(relpath)
    if len(all_globals) == 0:
        raise ValueError("Feature bank is empty")
    return FeatureBank(
        indices=list(indices),
        relpaths=relpaths,
        global_descs=torch.stack(all_globals),
        local_descs=all_locals,
        grid_hws=grid_hws,
        image_hws=image_hws,
        cropped_hws=cropped_hws,
        patch_stride=patch_stride,
    )


def build_segvlad_feature_bank(
    dataset,
    indices: Sequence[int],
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    vlad: VLAD,
    feature_cache_root: Optional[str] = None,
    masks_cache_root: Optional[str] = None,
    patch_stride: int = 14,
    min_mask_area_ratio: float = 0.0,
    neighbor_order: int = 0,
    seg_cfg: Optional[SegmentorConfig] = None,
    sam_generator: Optional[SAMGenerator] = None,
    verbose: bool = True,
) -> SegVLADFeatureBank:
    if seg_cfg is None:
        seg_cfg = SegmentorConfig()
    if sam_generator is None and seg_cfg.source in ["sam", "manifest_or_sam"]:
        try:
            sam_generator = SAMGenerator(
                seg_cfg,
                device=str(device),
            )
        except Exception as exc:
            if seg_cfg.source == "sam":
                raise
            if verbose:
                print(f"WARN: SAM generator unavailable, falling back to manifest/full-image masks: {exc}")

    relpaths = []
    all_locals = []
    grid_hws = []
    image_hws = []
    cropped_hws = []
    image_segment_descs: List[torch.Tensor] = []
    image_segment_counts: List[int] = []
    flat_segment_descs = []
    segment_to_image = []
    segment_to_image_pos = []
    iterator = tqdm(indices, disable=not verbose, desc="SegVLAD features")
    for image_pos, idx in enumerate(iterator):
        record = load_or_extract_record(
            dataset, idx, dino=dino, device=device,
            cache_root=feature_cache_root, global_agg="vlad",
            patch_stride=patch_stride,
        )
        masks = load_or_generate_masks(
            dataset,
            idx,
            seg_cfg=seg_cfg,
            mask_generator=sam_generator,
            cache_root=masks_cache_root,
        )
        mask_grid, pixel_masks = project_masks_to_patch_grid(
            masks,
            image_hw=record.image_hw,
            cropped_hw=record.cropped_hw,
            grid_hw=record.grid_hw,
            min_area_ratio=min_mask_area_ratio,
        )
        adjacency = build_mask_adjacency_revisit(
            mask_grid,
            pixel_masks=pixel_masks,
            grid_hw=record.grid_hw,
            order=neighbor_order,
            method=seg_cfg.neighbor_method,
        )
        local_flat = record.local_desc.reshape(-1, record.local_desc.shape[-1])
        seg_descs = segvlad_descriptors(local_flat, vlad.c_centers, mask_grid, adjacency=adjacency)

        relpaths.append(dataset.get_image_relpaths(idx))
        all_locals.append(record.local_desc)
        grid_hws.append(record.grid_hw)
        image_hws.append(record.image_hw)
        cropped_hws.append(record.cropped_hw)
        image_segment_descs.append(seg_descs)
        image_segment_counts.append(int(seg_descs.shape[0]))
        flat_segment_descs.append(seg_descs)
        segment_to_image.extend([int(idx)] * int(seg_descs.shape[0]))
        segment_to_image_pos.extend([image_pos] * int(seg_descs.shape[0]))

    if len(flat_segment_descs) == 0:
        raise ValueError("SegVLAD feature bank is empty")
    return SegVLADFeatureBank(
        indices=list(indices),
        relpaths=relpaths,
        local_descs=all_locals,
        grid_hws=grid_hws,
        image_hws=image_hws,
        cropped_hws=cropped_hws,
        patch_stride=patch_stride,
        image_segment_descs=image_segment_descs,
        image_segment_counts=image_segment_counts,
        segment_descs=torch.cat(flat_segment_descs, dim=0),
        segment_to_image=torch.as_tensor(segment_to_image, dtype=torch.long),
        segment_to_image_pos=torch.as_tensor(segment_to_image_pos, dtype=torch.long),
    )


def coarse_retrieve_topk_segvlad(
    db_bank: SegVLADFeatureBank,
    query_bank: SegVLADFeatureBank,
    top_k: int,
    segment_top_k: int = 100,
    aggregation: Literal["sum", "revisit_weighted_borda_image"] = "revisit_weighted_borda_image",
) -> Tuple[np.ndarray, np.ndarray]:
    db_segments = F.normalize(db_bank.segment_descs.float(), dim=1)
    num_images = len(db_bank.indices)
    score_rows = []
    index_rows = []
    for q_segments_raw in query_bank.image_segment_descs:
        q_segments = F.normalize(q_segments_raw.float(), dim=1)
        sims = q_segments @ db_segments.T
        seg_k = min(int(segment_top_k), sims.shape[1])
        top_scores, top_indices = torch.topk(sims, k=seg_k, dim=1)
        image_scores = torch.zeros(num_images, dtype=torch.float32)
        if aggregation == "sum":
            image_scores.index_add_(
                0,
                db_bank.segment_to_image_pos[top_indices.reshape(-1)],
                top_scores.reshape(-1),
            )
            image_scores = image_scores / max(1, q_segments.shape[0])
        elif aggregation == "revisit_weighted_borda_image":
            sims_min = float(top_scores.min().item())
            sims_max = float(top_scores.max().item())
            denom = max(1e-12, sims_max - sims_min)
            norm_scores = (top_scores - sims_min) / denom
            image_ids = db_bank.segment_to_image_pos[top_indices]
            image_scores.index_add_(
                0,
                image_ids.reshape(-1),
                norm_scores.reshape(-1),
            )
        else:
            raise ValueError(f"Unknown SegVLAD coarse aggregation: {aggregation}")
        img_k = min(int(top_k), num_images)
        best_scores, best_indices = torch.topk(image_scores, k=img_k, dim=0)
        score_rows.append(best_scores)
        index_rows.append(best_indices)
    return (
        torch.stack(score_rows).cpu().numpy(),
        torch.stack(index_rows).cpu().numpy(),
    )


def coarse_retrieve_topk(
    db_global_descs: torch.Tensor,
    query_global_descs: torch.Tensor,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    db_norm = F.normalize(db_global_descs.float(), dim=1)
    qu_norm = F.normalize(query_global_descs.float(), dim=1)
    sim = qu_norm @ db_norm.T
    scores, indices = torch.topk(sim, k=min(top_k, db_norm.shape[0]), dim=1)
    return scores.cpu().numpy(), indices.cpu().numpy()


def compute_local_similarity_map(
    query_local_desc: torch.Tensor,
    tile_local_desc: torch.Tensor,
    top_m: int = 32,
) -> torch.Tensor:
    q = F.normalize(query_local_desc.reshape(-1, query_local_desc.shape[-1]).float(), dim=1)
    t = F.normalize(tile_local_desc.reshape(-1, tile_local_desc.shape[-1]).float(), dim=1)
    sims = q @ t.T
    top_m = max(1, min(top_m, q.shape[0]))
    patch_scores = sims.topk(k=top_m, dim=0).values.mean(dim=0)
    return patch_scores.reshape(tile_local_desc.shape[0], tile_local_desc.shape[1])


def compute_local_match_score(
    query_local_desc: torch.Tensor,
    tile_local_desc: torch.Tensor,
    method: Literal["cosine_pool", "mutual_nn", "sim_map"] = "sim_map",
    top_m: int = 32,
) -> float:
    q = F.normalize(query_local_desc.reshape(-1, query_local_desc.shape[-1]).float(), dim=1)
    t = F.normalize(tile_local_desc.reshape(-1, tile_local_desc.shape[-1]).float(), dim=1)
    sims = q @ t.T
    if method == "cosine_pool":
        patch_best = sims.max(dim=1).values
        top_m = max(1, min(top_m, patch_best.shape[0]))
        return float(patch_best.topk(k=top_m).values.mean().item())
    if method == "mutual_nn":
        q_best = sims.argmax(dim=1)
        t_best = sims.argmax(dim=0)
        mutual_scores = []
        for q_idx, t_idx in enumerate(q_best.tolist()):
            if int(t_best[t_idx]) == q_idx:
                mutual_scores.append(float(sims[q_idx, t_idx].item()))
        if len(mutual_scores) == 0:
            return float(sims.max(dim=1).values.mean().item())
        return float(np.mean(mutual_scores))
    if method == "sim_map":
        sim_map = compute_local_similarity_map(query_local_desc, tile_local_desc, top_m=top_m)
        top_m = max(1, min(top_m, sim_map.numel()))
        return float(sim_map.reshape(-1).topk(k=top_m).values.mean().item())
    raise ValueError(f"Unknown local match method: {method}")


def rerank_with_local_features(
    coarse_scores: np.ndarray,
    coarse_indices: np.ndarray,
    db_bank: FeatureBank,
    query_bank: FeatureBank,
    local_match_method: Literal["cosine_pool", "mutual_nn", "sim_map"] = "sim_map",
    rerank_alpha: float = 0.5,
    local_top_m: int = 32,
) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    reranked_indices = []
    reranked_scores = []
    local_score_table: List[List[float]] = []
    for qi in range(len(query_bank.indices)):
        candidates = coarse_indices[qi]
        combined = []
        candidate_local_scores = []
        for rank, db_idx in enumerate(candidates.tolist()):
            local_score = compute_local_match_score(
                query_bank.local_descs[qi],
                db_bank.local_descs[int(db_idx)],
                method=local_match_method,
                top_m=local_top_m,
            )
            candidate_local_scores.append(local_score)
            combined_score = float(rerank_alpha * coarse_scores[qi, rank] + (1.0 - rerank_alpha) * local_score)
            combined.append((combined_score, int(db_idx)))
        combined.sort(key=lambda x: x[0], reverse=True)
        reranked_scores.append([score for score, _ in combined])
        reranked_indices.append([db_idx for _, db_idx in combined])
        local_score_table.append(candidate_local_scores)
    return (
        np.asarray(reranked_scores, dtype=np.float32),
        np.asarray(reranked_indices, dtype=np.int64),
        local_score_table,
    )


def estimate_offset_from_similarity_map(
    query_local_desc: torch.Tensor,
    tile_local_desc: torch.Tensor,
    tile_extent: Optional[Union[float, Tuple[float, float]]] = None,
    top_m: int = 32,
    temperature: float = 15.0,
) -> Tuple[Tuple[float, float], torch.Tensor]:
    sim_map = compute_local_similarity_map(query_local_desc, tile_local_desc, top_m=top_m)
    h, w = sim_map.shape
    probs = F.softmax(sim_map.reshape(-1) * temperature, dim=0).reshape(h, w)
    ys = torch.linspace(-0.5 + 0.5 / h, 0.5 - 0.5 / h, h)
    xs = torch.linspace(-0.5 + 0.5 / w, 0.5 - 0.5 / w, w)
    exp_y = float((probs.sum(dim=1) * ys).sum().item())
    exp_x = float((probs.sum(dim=0) * xs).sum().item())
    if tile_extent is None:
        scale_x = float(w)
        scale_y = float(h)
    elif isinstance(tile_extent, (int, float)):
        scale_x = float(tile_extent)
        scale_y = float(tile_extent)
    else:
        scale_x = float(tile_extent[0])
        scale_y = float(tile_extent[1])
    return (exp_x * scale_x, exp_y * scale_y), sim_map


def predict_top1_offsets(
    dataset,
    retrieval_indices: np.ndarray,
    db_bank: FeatureBank,
    query_bank: FeatureBank,
    tile_extent_default: Optional[Union[float, Tuple[float, float]]] = None,
    local_top_m: int = 32,
    query_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Optional[Tuple[float, float]]], List[Optional[torch.Tensor]]]:
    pred_offsets: List[Optional[Tuple[float, float]]] = []
    sim_maps: List[Optional[torch.Tensor]] = []
    if query_indices is None:
        query_indices = list(range(len(query_bank.indices)))
    for qi, dataset_qi in enumerate(query_indices):
        top1_db_idx = int(retrieval_indices[qi, 0])
        dataset_db_idx = int(db_bank.indices[top1_db_idx])
        tile_extent = dataset.get_tile_extent(dataset_db_idx)
        if tile_extent is None:
            tile_extent = tile_extent_default
        pred_offset, sim_map = estimate_offset_from_similarity_map(
            query_bank.local_descs[qi],
            db_bank.local_descs[top1_db_idx],
            tile_extent=tile_extent,
            top_m=local_top_m,
        )
        pred_offsets.append(pred_offset)
        sim_maps.append(sim_map)
    return pred_offsets, sim_maps
