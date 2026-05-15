"""
    CVGL feature extraction, coarse retrieval, local reranking and
    similarity-map offset estimation.
"""

import os
import sys
import time
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


def _profile_add(profile: Optional[Dict[str, float]], key: str, value: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(value)


def _profile_inc(profile: Optional[Dict[str, float]], key: str, value: int = 1) -> None:
    if profile is None:
        return
    profile[key] = int(profile.get(key, 0)) + int(value)


def _sync_device(device: Union[str, torch.device]) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        if device.index is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)


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


def mask_centroid_fourier_encoding(
    mask_grid: torch.Tensor,
    grid_hw: Tuple[int, int],
    num_frequencies: int,
) -> torch.Tensor:
    if num_frequencies <= 0:
        return torch.zeros((mask_grid.shape[0], 0), dtype=torch.float32)
    if mask_grid.ndim != 2:
        raise ValueError(f"mask_grid must be [num_segments, num_patches], got {tuple(mask_grid.shape)}")
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != mask_grid.shape[1]:
        raise ValueError(f"grid_hw {grid_hw} is incompatible with {mask_grid.shape[1]} patches")

    mask_float = mask_grid.float()
    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5).repeat_interleave(grid_w) / max(1, grid_h)
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5).repeat(grid_h) / max(1, grid_w)
    denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroids = torch.stack([
        (mask_float * xs.unsqueeze(0)).sum(dim=1) / denom.squeeze(1),
        (mask_float * ys.unsqueeze(0)).sum(dim=1) / denom.squeeze(1),
    ], dim=1)

    frequencies = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * torch.pi
    angles = centroids.unsqueeze(-1) * frequencies.view(1, 1, -1)
    return torch.cat([
        torch.sin(angles[:, 0, :]),
        torch.cos(angles[:, 0, :]),
        torch.sin(angles[:, 1, :]),
        torch.cos(angles[:, 1, :]),
    ], dim=1)


def _mask_centroids_on_grid(
    mask_grid: torch.Tensor,
    grid_hw: Tuple[int, int],
) -> torch.Tensor:
    if mask_grid.ndim != 2:
        raise ValueError(f"mask_grid must be [num_segments, num_patches], got {tuple(mask_grid.shape)}")
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != mask_grid.shape[1]:
        raise ValueError(f"grid_hw {grid_hw} is incompatible with {mask_grid.shape[1]} patches")

    mask_float = mask_grid.float()
    ys = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w) + 0.5
    xs = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h) + 0.5
    denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.stack([
        (mask_float * xs.unsqueeze(0)).sum(dim=1) / denom.squeeze(1),
        (mask_float * ys.unsqueeze(0)).sum(dim=1) / denom.squeeze(1),
    ], dim=1)


def relative_neighbor_context_encoding(
    mask_grid: torch.Tensor,
    grid_hw: Tuple[int, int],
    adjacency: Optional[torch.Tensor],
    num_frequencies: int,
    ref_grid_hw: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    if num_frequencies <= 0:
        return torch.zeros((mask_grid.shape[0], 0), dtype=torch.float32)
    num_segments = mask_grid.shape[0]
    if num_segments == 0:
        return torch.zeros((0, 5 * 2 * num_frequencies), dtype=torch.float32)

    if adjacency is None:
        neighbor_mask = torch.eye(num_segments, dtype=torch.bool)
    else:
        neighbor_mask = adjacency.bool().clone()
    if neighbor_mask.shape != (num_segments, num_segments):
        raise ValueError(
            f"adjacency must be [{num_segments}, {num_segments}], got {tuple(neighbor_mask.shape)}"
        )
    neighbor_mask.fill_diagonal_(False)

    centroids = _mask_centroids_on_grid(mask_grid, grid_hw=grid_hw)
    rel = centroids.unsqueeze(0) - centroids.unsqueeze(1)
    if ref_grid_hw is None:
        ref_h, ref_w = float(grid_hw[0]), float(grid_hw[1])
    else:
        ref_h, ref_w = float(ref_grid_hw[0]), float(ref_grid_hw[1])
    ref_h = max(1.0, ref_h)
    ref_w = max(1.0, ref_w)
    ref_diag = max(1.0, float(np.hypot(ref_w, ref_h)))

    dx = rel[:, :, 0] / ref_w
    dy = rel[:, :, 1] / ref_h
    dist = torch.sqrt(rel[:, :, 0] ** 2 + rel[:, :, 1] ** 2) / ref_diag
    angle = torch.atan2(rel[:, :, 1], rel[:, :, 0])
    geom = torch.stack([dx, dy, dist, torch.sin(angle), torch.cos(angle)], dim=2)

    frequencies = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * torch.pi
    angles = geom.unsqueeze(-1) * frequencies.view(1, 1, 1, -1)
    edge_pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=3).reshape(
        num_segments,
        num_segments,
        -1,
    )
    weights = neighbor_mask.float()
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (weights.unsqueeze(-1) * edge_pe).sum(dim=1) / denom


def segvlad_descriptors(
    patch_descs: torch.Tensor,
    c_centers: torch.Tensor,
    mask_grid: torch.Tensor,
    adjacency: Optional[torch.Tensor] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
    centroid_pe_num_freqs: int = 0,
    centroid_pe_weight: float = 0.0,
    relative_context_adjacency: Optional[torch.Tensor] = None,
    relative_context_num_freqs: int = 0,
    relative_context_weight: float = 0.0,
    relative_context_ref_grid_hw: Optional[Tuple[float, float]] = None,
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
    seg_descs = F.normalize(stacked, dim=1)
    if centroid_pe_num_freqs > 0 and centroid_pe_weight > 0.0:
        if grid_hw is None:
            raise ValueError("grid_hw is required when centroid Fourier positional encoding is enabled")
        centroid_pe = mask_centroid_fourier_encoding(
            mask_grid,
            grid_hw=grid_hw,
            num_frequencies=centroid_pe_num_freqs,
        )
        seg_descs = torch.cat([seg_descs, float(centroid_pe_weight) * centroid_pe], dim=1)
    if relative_context_num_freqs > 0 and relative_context_weight > 0.0:
        if grid_hw is None:
            raise ValueError("grid_hw is required when relative neighbor context is enabled")
        rel_context = relative_neighbor_context_encoding(
            mask_grid,
            grid_hw=grid_hw,
            adjacency=relative_context_adjacency,
            num_frequencies=relative_context_num_freqs,
            ref_grid_hw=relative_context_ref_grid_hw,
        )
        seg_descs = torch.cat([seg_descs, float(relative_context_weight) * rel_context], dim=1)
    return F.normalize(seg_descs, dim=1)


def segment_average_descriptors(
    patch_descs: torch.Tensor,
    mask_grid: torch.Tensor,
    adjacency: Optional[torch.Tensor] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
    centroid_pe_num_freqs: int = 0,
    centroid_pe_weight: float = 0.0,
    relative_context_adjacency: Optional[torch.Tensor] = None,
    relative_context_num_freqs: int = 0,
    relative_context_weight: float = 0.0,
    relative_context_ref_grid_hw: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    if patch_descs.ndim != 2:
        raise ValueError(f"patch_descs must be [num_patches, desc_dim], got {tuple(patch_descs.shape)}")
    if mask_grid.ndim != 2:
        raise ValueError(f"mask_grid must be [num_segments, num_patches], got {tuple(mask_grid.shape)}")
    if patch_descs.shape[0] != mask_grid.shape[1]:
        raise ValueError(
            f"Patch/mask mismatch: {patch_descs.shape[0]} patches vs {mask_grid.shape[1]} mask entries"
        )
    if adjacency is None:
        adjacency = torch.eye(mask_grid.shape[0], dtype=torch.bool)

    descs = F.normalize(patch_descs.float(), dim=1)
    segment_masks = (adjacency.float() @ mask_grid.float()) > 0
    weights = segment_masks.float()
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    seg_descs = F.normalize((weights @ descs) / denom, dim=1)
    if centroid_pe_num_freqs > 0 and centroid_pe_weight > 0.0:
        if grid_hw is None:
            raise ValueError("grid_hw is required when centroid Fourier positional encoding is enabled")
        centroid_pe = mask_centroid_fourier_encoding(
            mask_grid,
            grid_hw=grid_hw,
            num_frequencies=centroid_pe_num_freqs,
        )
        seg_descs = torch.cat([seg_descs, float(centroid_pe_weight) * centroid_pe], dim=1)
    if relative_context_num_freqs > 0 and relative_context_weight > 0.0:
        if grid_hw is None:
            raise ValueError("grid_hw is required when relative neighbor context is enabled")
        rel_context = relative_neighbor_context_encoding(
            mask_grid,
            grid_hw=grid_hw,
            adjacency=relative_context_adjacency,
            num_frequencies=relative_context_num_freqs,
            ref_grid_hw=relative_context_ref_grid_hw,
        )
        seg_descs = torch.cat([seg_descs, float(relative_context_weight) * rel_context], dim=1)
    return F.normalize(seg_descs, dim=1)


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


def global_pool_descriptors(
    patch_descs: torch.Tensor,
    method: Literal["gap", "gmp"],
) -> torch.Tensor:
    if patch_descs.ndim != 2:
        raise ValueError("patch_descs must have shape [num_patches, desc_dim]")
    if method == "gap":
        return patch_descs.mean(dim=0)
    if method == "gmp":
        return patch_descs.max(dim=0).values
    raise ValueError(f"Unknown global pooling method: {method}")


def _extract_patch_and_cls_descs(
    img_in: torch.Tensor,
    dino: DinoV2ExtractFeatures,
    need_cls: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not need_cls:
        return dino(img_in)[0].detach().cpu(), None
    if not hasattr(dino, "use_cls"):
        raise ValueError("global_agg='cls' requires an extractor with a `use_cls` switch")
    old_use_cls = bool(dino.use_cls)
    dino.use_cls = True
    try:
        descs = dino(img_in)[0].detach().cpu()
    finally:
        dino.use_cls = old_use_cls
    if descs.shape[0] < 2:
        raise ValueError(f"Expected CLS + patch descriptors, got shape {tuple(descs.shape)}")
    return descs[1:], descs[0]


def extract_dense_record(
    img: torch.Tensor,
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    patch_stride: int = 14,
    global_agg: Literal["vlad", "gem", "cls", "gap", "gmp"] = "vlad",
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
    patch_descs, cls_desc = _extract_patch_and_cls_descs(
        img_in,
        dino=dino,
        need_cls=global_agg == "cls",
    )
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
    elif global_agg == "cls":
        if cls_desc is None:
            raise ValueError("CLS descriptor was not extracted")
        global_desc = F.normalize(cls_desc, dim=0)
    elif global_agg in ["gap", "gmp"]:
        global_desc = F.normalize(
            global_pool_descriptors(patch_descs, method=global_agg),
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
    global_agg: Literal["vlad", "gem", "cls", "gap", "gmp"] = "vlad",
    gem_p: float = 3.0,
    gem_use_abs: bool = False,
    gem_elem_by_elem: bool = False,
    patch_stride: int = 14,
    profile: Optional[Dict[str, float]] = None,
) -> DenseFeatureRecord:
    relpath = dataset.get_image_relpaths(index)
    cache_path = _feature_cache_path(cache_root, relpath)
    if cache_path is not None and os.path.isfile(cache_path):
        start = time.perf_counter()
        payload = torch.load(cache_path)
        _profile_add(profile, "feature_cache_load_time_s", time.perf_counter() - start)
        _profile_inc(profile, "feature_cache_hits")
        if global_agg == "vlad" or payload.get("global_desc") is not None:
            return DenseFeatureRecord(
                local_desc=payload["local_desc"],
                grid_hw=tuple(payload["grid_hw"]),
                image_hw=tuple(payload["image_hw"]),
                cropped_hw=tuple(payload["cropped_hw"]),
                patch_stride=int(payload["patch_stride"]),
                global_desc=payload.get("global_desc"),
            )
        _profile_inc(profile, "feature_cache_incomplete")
    _profile_inc(profile, "feature_cache_misses")
    start = time.perf_counter()
    img, _ = dataset[index]
    _profile_add(profile, "image_load_time_s", time.perf_counter() - start)
    _sync_device(device)
    start = time.perf_counter()
    record = extract_dense_record(
        img, dino=dino, device=device, patch_stride=patch_stride,
        global_agg=global_agg, gem_p=gem_p,
        gem_use_abs=gem_use_abs, gem_elem_by_elem=gem_elem_by_elem,
    )
    _sync_device(device)
    _profile_add(profile, "feature_extract_time_s", time.perf_counter() - start)
    _profile_inc(profile, "feature_extract_count")
    if cache_path is not None:
        start = time.perf_counter()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "local_desc": record.local_desc,
            "grid_hw": list(record.grid_hw),
            "image_hw": list(record.image_hw),
            "cropped_hw": list(record.cropped_hw),
            "patch_stride": record.patch_stride,
            "global_desc": record.global_desc,
        }, cache_path)
        _profile_add(profile, "feature_cache_save_time_s", time.perf_counter() - start)
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
    profile: Optional[Dict[str, float]] = None,
) -> VLAD:
    stage_start = time.perf_counter()
    vlad = VLAD(
        num_clusters,
        None,
        vlad_mode=vlad_assignment,
        soft_temp=vlad_soft_temp,
        cache_dir=vlad_cache_root,
    )
    if vlad.can_use_cache_vlad():
        start = time.perf_counter()
        vlad.fit(None)
        _profile_add(profile, "vlad_cache_load_time_s", time.perf_counter() - start)
        _profile_add(profile, "vlad_fit_stage_time_s", time.perf_counter() - stage_start)
        _profile_inc(profile, "vlad_cache_hits")
        return vlad
    _profile_inc(profile, "vlad_cache_misses")
    train_descs = []
    fit_indices = list(db_indices)[::sub_sample_db_vlad]
    iterator = tqdm(fit_indices, disable=(not verbose) or (not sys.stderr.isatty()), desc="VLAD clusters")
    for idx in iterator:
        record = load_or_extract_record(
            dataset, idx, dino=dino, device=device,
            cache_root=feature_cache_root, global_agg="vlad",
            patch_stride=patch_stride,
            profile=profile,
        )
        train_descs.append(record.local_desc.reshape(-1, record.local_desc.shape[-1]))
    if len(train_descs) == 0:
        raise ValueError("No descriptors collected for VLAD fitting")
    start = time.perf_counter()
    vlad.fit(torch.cat(train_descs, dim=0))
    _profile_add(profile, "vlad_fit_compute_time_s", time.perf_counter() - start)
    _profile_add(profile, "vlad_fit_stage_time_s", time.perf_counter() - stage_start)
    return vlad


def build_feature_bank(
    dataset,
    indices: Sequence[int],
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    global_agg: Literal["vlad", "gem", "cls", "gap", "gmp"] = "vlad",
    vlad: Optional[VLAD] = None,
    feature_cache_root: Optional[str] = None,
    gem_p: float = 3.0,
    gem_use_abs: bool = False,
    gem_elem_by_elem: bool = False,
    patch_stride: int = 14,
    verbose: bool = True,
    profile: Optional[Dict[str, float]] = None,
) -> FeatureBank:
    stage_start = time.perf_counter()
    all_globals = []
    all_locals = []
    grid_hws = []
    image_hws = []
    cropped_hws = []
    relpaths = []
    iterator = tqdm(indices, disable=(not verbose) or (not sys.stderr.isatty()), desc="Dense features")
    for idx in iterator:
        record = load_or_extract_record(
            dataset, idx, dino=dino, device=device,
            cache_root=feature_cache_root, global_agg=global_agg,
            gem_p=gem_p, gem_use_abs=gem_use_abs,
            gem_elem_by_elem=gem_elem_by_elem, patch_stride=patch_stride,
            profile=profile,
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
                raise ValueError(f"{global_agg} aggregation requested but global_desc is missing")
        all_globals.append(F.normalize(global_desc, dim=0))
        all_locals.append(record.local_desc)
        grid_hws.append(record.grid_hw)
        image_hws.append(record.image_hw)
        cropped_hws.append(record.cropped_hw)
        relpaths.append(relpath)
    if len(all_globals) == 0:
        raise ValueError("Feature bank is empty")
    _profile_add(profile, "feature_bank_build_time_s", time.perf_counter() - stage_start)
    _profile_inc(profile, "feature_bank_num_images", len(all_globals))
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
    vlad: Optional[VLAD],
    feature_cache_root: Optional[str] = None,
    masks_cache_root: Optional[str] = None,
    patch_stride: int = 14,
    segment_descriptor: Literal["segvlad", "sap"] = "segvlad",
    min_mask_area_ratio: float = 0.0,
    neighbor_order: int = 0,
    centroid_pe_num_freqs: int = 0,
    centroid_pe_weight: float = 0.0,
    relative_context_num_freqs: int = 0,
    relative_context_weight: float = 0.0,
    relative_context_order: int = 1,
    relative_context_ref_grid_hw: Optional[Tuple[float, float]] = None,
    seg_cfg: Optional[SegmentorConfig] = None,
    sam_generator: Optional[SAMGenerator] = None,
    verbose: bool = True,
    profile: Optional[Dict[str, float]] = None,
) -> SegVLADFeatureBank:
    stage_start = time.perf_counter()
    if seg_cfg is None:
        seg_cfg = SegmentorConfig()
    if segment_descriptor == "segvlad" and vlad is None:
        raise ValueError("segment_descriptor='segvlad' requires a fitted VLAD instance")
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
    iterator = tqdm(indices, disable=(not verbose) or (not sys.stderr.isatty()), desc="SegVLAD features")
    for image_pos, idx in enumerate(iterator):
        record = load_or_extract_record(
            dataset, idx, dino=dino, device=device,
            cache_root=feature_cache_root, global_agg="vlad",
            patch_stride=patch_stride,
            profile=profile,
        )
        start = time.perf_counter()
        masks = load_or_generate_masks(
            dataset,
            idx,
            seg_cfg=seg_cfg,
            mask_generator=sam_generator,
            cache_root=masks_cache_root,
        )
        _profile_add(profile, "mask_load_or_generate_time_s", time.perf_counter() - start)
        start = time.perf_counter()
        mask_grid, pixel_masks = project_masks_to_patch_grid(
            masks,
            image_hw=record.image_hw,
            cropped_hw=record.cropped_hw,
            grid_hw=record.grid_hw,
            min_area_ratio=min_mask_area_ratio,
        )
        _profile_add(profile, "mask_project_time_s", time.perf_counter() - start)
        start = time.perf_counter()
        adjacency = build_mask_adjacency_revisit(
            mask_grid,
            pixel_masks=pixel_masks,
            grid_hw=record.grid_hw,
            order=neighbor_order,
            method=seg_cfg.neighbor_method,
        )
        relative_context_adjacency = None
        if relative_context_num_freqs > 0 and relative_context_weight > 0.0:
            relative_context_adjacency = build_mask_adjacency_revisit(
                mask_grid,
                pixel_masks=pixel_masks,
                grid_hw=record.grid_hw,
                order=relative_context_order,
                method=seg_cfg.neighbor_method,
        )
        _profile_add(profile, "mask_adjacency_time_s", time.perf_counter() - start)
        local_flat = record.local_desc.reshape(-1, record.local_desc.shape[-1])
        start = time.perf_counter()
        if segment_descriptor == "segvlad":
            seg_descs = segvlad_descriptors(
                local_flat,
                vlad.c_centers,
                mask_grid,
                adjacency=adjacency,
                grid_hw=record.grid_hw,
                centroid_pe_num_freqs=centroid_pe_num_freqs,
                centroid_pe_weight=centroid_pe_weight,
                relative_context_adjacency=relative_context_adjacency,
                relative_context_num_freqs=relative_context_num_freqs,
                relative_context_weight=relative_context_weight,
                relative_context_ref_grid_hw=relative_context_ref_grid_hw,
            )
        elif segment_descriptor == "sap":
            seg_descs = segment_average_descriptors(
                local_flat,
                mask_grid,
                adjacency=adjacency,
                grid_hw=record.grid_hw,
                centroid_pe_num_freqs=centroid_pe_num_freqs,
                centroid_pe_weight=centroid_pe_weight,
                relative_context_adjacency=relative_context_adjacency,
                relative_context_num_freqs=relative_context_num_freqs,
                relative_context_weight=relative_context_weight,
                relative_context_ref_grid_hw=relative_context_ref_grid_hw,
            )
        else:
            raise ValueError(f"Unknown segment descriptor: {segment_descriptor}")
        _profile_add(profile, "segment_descriptor_build_time_s", time.perf_counter() - start)
        _profile_inc(profile, "segment_bank_num_images")
        _profile_inc(profile, "segment_bank_num_segments", int(seg_descs.shape[0]))

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
    start = time.perf_counter()
    segment_descs = torch.cat(flat_segment_descs, dim=0)
    _profile_add(profile, "segment_bank_concat_time_s", time.perf_counter() - start)
    _profile_add(profile, "segment_bank_build_time_s", time.perf_counter() - stage_start)
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
        segment_descs=segment_descs,
        segment_to_image=torch.as_tensor(segment_to_image, dtype=torch.long),
        segment_to_image_pos=torch.as_tensor(segment_to_image_pos, dtype=torch.long),
    )


def coarse_retrieve_topk_segvlad(
    db_bank: SegVLADFeatureBank,
    query_bank: SegVLADFeatureBank,
    top_k: int,
    segment_top_k: int = 100,
    aggregation: Literal["sum", "revisit_weighted_borda_image"] = "revisit_weighted_borda_image",
    device: Optional[Union[str, torch.device]] = None,
    db_segment_chunk_size: int = 0,
    query_batch_size: int = 16,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    db_segments = F.normalize(db_bank.segment_descs.float(), dim=1)
    num_images = len(db_bank.indices)
    if device is None:
        retrieval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        retrieval_device = torch.device(device)
    if retrieval_device.type == "cuda" and not torch.cuda.is_available():
        retrieval_device = torch.device("cpu")
    db_segment_chunk_size = int(db_segment_chunk_size)
    query_batch_size = max(1, int(query_batch_size))

    def _cuda_memory_budget() -> int:
        if retrieval_device.type != "cuda":
            return 0
        device_index = retrieval_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        reserve_bytes = max(1 << 30, int(0.15 * total_bytes))
        return max(0, int(0.70 * max(0, free_bytes - reserve_bytes)))

    def _choose_dynamic_chunk(num_query_segments: int) -> int:
        if db_segment_chunk_size > 0 or retrieval_device.type != "cuda":
            return max(1, db_segment_chunk_size)
        budget = _cuda_memory_budget()
        if budget <= 0:
            return 1024
        bytes_per_value = db_segments.element_size()
        desc_dim = int(db_segments.shape[1])
        # Account for the DB chunk, similarity matrix, and matmul/topk workspace.
        bytes_per_db_segment = bytes_per_value * (desc_dim + 3 * max(1, num_query_segments))
        chunk = max(1, budget // max(1, bytes_per_db_segment))
        chunk = min(int(db_segments.shape[0]), int(chunk))
        return max(256, (chunk // 256) * 256)

    def _aggregate_image_scores(
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        num_query_segments: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_scores = torch.zeros(num_images, dtype=torch.float32, device=retrieval_device)
        if aggregation == "sum":
            image_scores.index_add_(
                0,
                segment_to_image_pos[top_indices.reshape(-1)],
                top_scores.reshape(-1),
            )
            image_scores = image_scores / max(1, num_query_segments)
        elif aggregation == "revisit_weighted_borda_image":
            sims_min = float(top_scores.min().item())
            sims_max = float(top_scores.max().item())
            denom = max(1e-12, sims_max - sims_min)
            norm_scores = (top_scores - sims_min) / denom
            image_ids = segment_to_image_pos[top_indices]
            image_scores.index_add_(
                0,
                image_ids.reshape(-1),
                norm_scores.reshape(-1),
            )
        else:
            raise ValueError(f"Unknown SegVLAD coarse aggregation: {aggregation}")
        img_k = min(int(top_k), num_images)
        return torch.topk(image_scores, k=img_k, dim=0)

    db_segments_device = None
    use_chunked_db = retrieval_device.type == "cuda" and db_segment_chunk_size > 0
    if not use_chunked_db:
        if retrieval_device.type == "cuda":
            db_bytes = db_segments.numel() * db_segments.element_size()
            if db_bytes > _cuda_memory_budget():
                use_chunked_db = True
        if not use_chunked_db:
            try:
                db_segments_device = db_segments.to(retrieval_device)
            except RuntimeError as exc:
                if retrieval_device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                use_chunked_db = True
                if db_segment_chunk_size <= 0:
                    db_segment_chunk_size = 4096
                if verbose:
                    print(
                        "WARN: unable to keep all DB SegVLAD segments on CUDA; "
                        f"falling back to chunks of {db_segment_chunk_size}: {exc}"
                    )
        if use_chunked_db and verbose:
            chunk_msg = "dynamic" if db_segment_chunk_size <= 0 else str(db_segment_chunk_size)
            print(
                "WARN: DB SegVLAD segments do not fit in the CUDA memory budget; "
                f"using chunked retrieval with chunk size {chunk_msg}."
            )
    segment_to_image_pos = db_bank.segment_to_image_pos.to(retrieval_device)
    score_rows = []
    index_rows = []
    num_queries = len(query_bank.image_segment_descs)
    iterator = tqdm(
        range(0, num_queries, query_batch_size),
        disable=(not verbose) or (not sys.stderr.isatty()),
        desc="SegVLAD coarse retrieval",
    )
    for batch_start in iterator:
        batch_raw = query_bank.image_segment_descs[batch_start:batch_start + query_batch_size]
        q_counts = [int(q.shape[0]) for q in batch_raw]
        q_segments = F.normalize(torch.cat([q.float() for q in batch_raw], dim=0), dim=1).to(retrieval_device)
        seg_k = min(int(segment_top_k), db_segments.shape[0])
        if use_chunked_db:
            top_scores = None
            top_indices = None
            chunk_size = _choose_dynamic_chunk(q_segments.shape[0])
            if verbose and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"chunk={chunk_size}")
            start = 0
            while start < db_segments.shape[0]:
                end = min(start + chunk_size, db_segments.shape[0])
                try:
                    db_chunk = db_segments[start:end].to(retrieval_device)
                    sims = q_segments @ db_chunk.T
                except RuntimeError as exc:
                    if retrieval_device.type != "cuda" or "out of memory" not in str(exc).lower() or chunk_size <= 256:
                        raise
                    torch.cuda.empty_cache()
                    chunk_size = max(256, chunk_size // 2)
                    if verbose and hasattr(iterator, "set_postfix_str"):
                        iterator.set_postfix_str(f"chunk={chunk_size}")
                    continue
                chunk_k = min(seg_k, sims.shape[1])
                chunk_scores, chunk_indices = torch.topk(sims, k=chunk_k, dim=1)
                chunk_indices = chunk_indices + start
                if top_scores is None:
                    top_scores = chunk_scores
                    top_indices = chunk_indices
                else:
                    merged_scores = torch.cat([top_scores, chunk_scores], dim=1)
                    merged_indices = torch.cat([top_indices, chunk_indices], dim=1)
                    top_scores, merged_pos = torch.topk(merged_scores, k=seg_k, dim=1)
                    top_indices = torch.gather(merged_indices, 1, merged_pos)
                start = end
        else:
            sims = q_segments @ db_segments_device.T
            top_scores, top_indices = torch.topk(sims, k=seg_k, dim=1)

        offset = 0
        for count in q_counts:
            best_scores, best_indices = _aggregate_image_scores(
                top_scores[offset:offset + count],
                top_indices[offset:offset + count],
                count,
            )
            score_rows.append(best_scores)
            index_rows.append(best_indices)
            offset += count
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
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    reranked_indices = []
    reranked_scores = []
    local_score_table: List[List[float]] = []
    iterator = tqdm(
        range(len(query_bank.indices)),
        disable=(not verbose) or (not sys.stderr.isatty()),
        desc="Local rerank",
    )
    for qi in iterator:
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


def compute_sliding_window_ncc_map(
    query_local_desc: torch.Tensor,
    tile_local_desc: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if query_local_desc.ndim != 3 or tile_local_desc.ndim != 3:
        raise ValueError("query_local_desc and tile_local_desc must be [H, W, D]")
    q_h, q_w, q_d = query_local_desc.shape
    t_h, t_w, t_d = tile_local_desc.shape
    if q_d != t_d:
        raise ValueError(f"Descriptor dim mismatch: query {q_d} vs tile {t_d}")
    if q_h > t_h or q_w > t_w:
        raise ValueError(
            f"Query grid {(q_h, q_w)} must not exceed tile grid {(t_h, t_w)} for sliding NCC"
        )

    query = F.normalize(query_local_desc.float(), dim=-1).permute(2, 0, 1).unsqueeze(0)
    tile = F.normalize(tile_local_desc.float(), dim=-1).permute(2, 0, 1).unsqueeze(0)
    numel = float(query.numel())

    query_centered = query - query.mean()
    query_norm = query_centered.pow(2).sum().sqrt().clamp_min(eps)
    dot = F.conv2d(tile, query_centered).squeeze(0).squeeze(0)

    ones = torch.ones(
        (1, tile.shape[1], q_h, q_w),
        dtype=tile.dtype,
        device=tile.device,
    )
    window_sum = F.conv2d(tile, ones).squeeze(0).squeeze(0)
    window_sq_sum = F.conv2d(tile.pow(2), ones).squeeze(0).squeeze(0)
    window_var_sum = (window_sq_sum - window_sum.pow(2) / numel).clamp_min(eps)
    return dot / (query_norm * window_var_sum.sqrt())


def estimate_offset_from_sliding_window_ncc(
    query_local_desc: torch.Tensor,
    tile_local_desc: torch.Tensor,
    tile_extent: Optional[Union[float, Tuple[float, float]]] = None,
) -> Tuple[Tuple[float, float], torch.Tensor]:
    score_map = compute_sliding_window_ncc_map(query_local_desc, tile_local_desc)
    q_h, q_w = query_local_desc.shape[:2]
    t_h, t_w = tile_local_desc.shape[:2]
    best_flat = int(torch.argmax(score_map.reshape(-1)).item())
    best_y = best_flat // score_map.shape[1]
    best_x = best_flat % score_map.shape[1]

    center_x = float(best_x) + (float(q_w) - 1.0) / 2.0
    center_y = float(best_y) + (float(q_h) - 1.0) / 2.0
    norm_x = (center_x + 0.5) / float(t_w) - 0.5
    norm_y = (center_y + 0.5) / float(t_h) - 0.5

    if tile_extent is None:
        scale_x = float(t_w)
        scale_y = float(t_h)
    elif isinstance(tile_extent, (int, float)):
        scale_x = float(tile_extent)
        scale_y = float(tile_extent)
    else:
        scale_x = float(tile_extent[0])
        scale_y = float(tile_extent[1])
    return (norm_x * scale_x, norm_y * scale_y), score_map


def predict_top1_offsets(
    dataset,
    retrieval_indices: np.ndarray,
    db_bank: FeatureBank,
    query_bank: FeatureBank,
    tile_extent_default: Optional[Union[float, Tuple[float, float]]] = None,
    local_top_m: int = 32,
    query_indices: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Tuple[List[Optional[Tuple[float, float]]], List[Optional[torch.Tensor]]]:
    pred_offsets: List[Optional[Tuple[float, float]]] = []
    sim_maps: List[Optional[torch.Tensor]] = []
    if query_indices is None:
        query_indices = list(range(len(query_bank.indices)))
    iterator = tqdm(
        enumerate(query_indices),
        total=len(query_indices),
        disable=(not verbose) or (not sys.stderr.isatty()),
        desc="Offset sim_map",
    )
    for qi, dataset_qi in iterator:
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


def predict_top1_offsets_sliding_ncc(
    dataset,
    retrieval_indices: np.ndarray,
    db_bank: FeatureBank,
    query_bank: FeatureBank,
    tile_extent_default: Optional[Union[float, Tuple[float, float]]] = None,
    query_indices: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Tuple[List[Optional[Tuple[float, float]]], List[Optional[torch.Tensor]]]:
    pred_offsets: List[Optional[Tuple[float, float]]] = []
    score_maps: List[Optional[torch.Tensor]] = []
    if query_indices is None:
        query_indices = list(range(len(query_bank.indices)))
    iterator = tqdm(
        enumerate(query_indices),
        total=len(query_indices),
        disable=(not verbose) or (not sys.stderr.isatty()),
        desc="Offset slide_ncc",
    )
    for qi, dataset_qi in iterator:
        top1_db_idx = int(retrieval_indices[qi, 0])
        dataset_db_idx = int(db_bank.indices[top1_db_idx])
        tile_extent = dataset.get_tile_extent(dataset_db_idx)
        if tile_extent is None:
            tile_extent = tile_extent_default
        pred_offset, score_map = estimate_offset_from_sliding_window_ncc(
            query_bank.local_descs[qi],
            db_bank.local_descs[top1_db_idx],
            tile_extent=tile_extent,
        )
        pred_offsets.append(pred_offset)
        score_maps.append(score_map)
    return pred_offsets, score_maps
