"""
    Segmentation-mask extraction and Revisit-Anything-style projection
    helpers for SegVLAD.
"""

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import Delaunay


@dataclass
class SegmentorConfig:
    source: Literal["manifest", "sam", "manifest_or_sam"] = "manifest_or_sam"
    sam_checkpoint: Optional[str] = None
    sam_repo_root: Optional[str] = None
    sam_model_type: Literal["vit_h", "vit_l", "vit_b", "default"] = "vit_h"
    sam_resize: Literal["half", "full"] = "half"
    sam_min_area_px: int = 0
    sam_max_masks: Optional[int] = None
    sam_points_per_side: Optional[int] = None
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    sam_crop_n_layers: int = 0
    cache_generated_masks: bool = True
    neighbor_method: Literal["delaunay", "knn"] = "delaunay"


def _detect_revisit_anything_root(explicit_root: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit_root,
        os.environ.get("REVISIT_ANYTHING_ROOT"),
        "/data/zhanghaofei/xry/Revisit-Anything",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = os.path.realpath(os.path.expanduser(candidate))
        sam_dir = os.path.join(candidate, "sam")
        if os.path.isdir(sam_dir):
            return candidate
    return None


def _detect_sam_checkpoint(explicit_checkpoint: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit_checkpoint,
        os.environ.get("SAM_CHECKPOINT"),
        "/data/zhanghaofei/xry/AnyLoc/checkpoints/sam_vit_h_4b8939.pth",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = os.path.realpath(os.path.expanduser(candidate))
        if os.path.isfile(candidate):
            return candidate
    return None


class RevisitAnythingSAMGenerator:
    """
        Lightweight wrapper around Revisit-Anything's vendored SAM package.
    """
    def __init__(self, cfg: SegmentorConfig, device: str = "cuda") -> None:
        checkpoint = _detect_sam_checkpoint(cfg.sam_checkpoint)
        if checkpoint is None:
            raise ValueError(
                "SAM generation requires a valid checkpoint. "
                "Set `seg_cfg.sam_checkpoint` or `SAM_CHECKPOINT`."
            )
        ra_root = _detect_revisit_anything_root(cfg.sam_repo_root)
        if ra_root is None:
            raise ValueError(
                "Could not locate Revisit-Anything `sam` package. "
                "Set `seg_cfg.sam_repo_root` or `REVISIT_ANYTHING_ROOT`."
            )
        sam_pkg_root = os.path.join(ra_root, "sam")
        if sam_pkg_root not in sys.path:
            sys.path.append(sam_pkg_root)
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = sam_model_registry[cfg.sam_model_type](checkpoint=checkpoint)
        model.to(device=device)

        generator_kwargs = {
            "pred_iou_thresh": float(cfg.sam_pred_iou_thresh),
            "stability_score_thresh": float(cfg.sam_stability_score_thresh),
            "crop_n_layers": int(cfg.sam_crop_n_layers),
        }
        if cfg.sam_points_per_side is not None:
            generator_kwargs["points_per_side"] = int(cfg.sam_points_per_side)
        self.mask_generator = SamAutomaticMaskGenerator(model, **generator_kwargs)
        self.cfg = cfg
        self.device = device

    def _resize_for_sam(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.cfg.sam_resize == "full":
            return image_rgb
        h, w = image_rgb.shape[:2]
        return cv2.resize(
            image_rgb,
            (max(1, w // 2), max(1, h // 2)),
            interpolation=cv2.INTER_LINEAR,
        )

    def generate(self, image_rgb: np.ndarray) -> List[np.ndarray]:
        image_for_sam = self._resize_for_sam(image_rgb)
        raw_masks = self.mask_generator.generate(image_for_sam)
        masks: List[np.ndarray] = []
        for item in raw_masks:
            area = int(item.get("area", 0))
            if area < int(self.cfg.sam_min_area_px):
                continue
            seg = np.asarray(item["segmentation"]).astype(bool)
            masks.append(seg)
            if self.cfg.sam_max_masks is not None and len(masks) >= int(self.cfg.sam_max_masks):
                break
        return masks


def _load_image_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _mask_cache_file(cache_root: str, relpath: str) -> str:
    relpath = relpath.replace("\\", "/")
    stem = hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:16]
    safe_name = relpath.replace("/", "__")
    return os.path.join(cache_root, f"{safe_name}__{stem}.npz")


def load_or_generate_masks(
    dataset,
    index: int,
    seg_cfg: SegmentorConfig,
    mask_generator: Optional[RevisitAnythingSAMGenerator] = None,
    cache_root: Optional[str] = None,
) -> Optional[List[np.ndarray]]:
    if hasattr(dataset, "get_segmentation_masks"):
        manifest_masks = dataset.get_segmentation_masks(index)
        if manifest_masks is not None and len(manifest_masks) > 0:
            return manifest_masks
    if seg_cfg.source == "manifest":
        return None
    if seg_cfg.source not in ["sam", "manifest_or_sam"]:
        return None
    if mask_generator is None:
        raise ValueError(
            "Segmentation source requires SAM generation, but no SAM generator was created."
        )

    relpath = dataset.get_image_relpaths(index)
    cache_file = None
    if cache_root is not None and seg_cfg.cache_generated_masks:
        os.makedirs(cache_root, exist_ok=True)
        cache_file = _mask_cache_file(cache_root, relpath)
        if os.path.isfile(cache_file):
            cached = np.load(cache_file, allow_pickle=True)
            masks = cached["masks"]
            if masks.ndim == 2:
                masks = masks[None, ...]
            return [np.asarray(m).astype(bool) for m in masks]

    image_path = dataset.get_image_paths()[index]
    masks = mask_generator.generate(_load_image_rgb(image_path))
    if cache_file is not None:
        mask_payload = np.asarray(masks, dtype=bool) if len(masks) > 0 else np.zeros((0, 1, 1), dtype=bool)
        np.savez_compressed(cache_file, masks=mask_payload)
    return masks


def _pixel_to_patch_index(
    cropped_hw: Tuple[int, int],
    grid_hw: Tuple[int, int],
) -> torch.Tensor:
    crop_h, crop_w = cropped_hw
    grid_h, grid_w = grid_hw
    ys = torch.arange(crop_h, dtype=torch.long).unsqueeze(1).expand(crop_h, crop_w)
    xs = torch.arange(crop_w, dtype=torch.long).unsqueeze(0).expand(crop_h, crop_w)
    patch_y = torch.clamp((ys * grid_h) // max(1, crop_h), 0, grid_h - 1)
    patch_x = torch.clamp((xs * grid_w) // max(1, crop_w), 0, grid_w - 1)
    return (patch_y * grid_w + patch_x).reshape(-1)


def project_masks_to_patch_grid(
    masks: Optional[Sequence[np.ndarray]],
    image_hw: Tuple[int, int],
    cropped_hw: Tuple[int, int],
    grid_hw: Tuple[int, int],
    min_area_ratio: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if masks is None or len(masks) == 0:
        full = torch.ones((1, grid_hw[0] * grid_hw[1]), dtype=torch.bool)
        return full, None

    mask_tensor = torch.as_tensor(np.asarray(masks), dtype=torch.float32)
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if tuple(mask_tensor.shape[-2:]) != tuple(image_hw):
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(1),
            size=image_hw,
            mode="nearest",
        ).squeeze(1)

    crop_h, crop_w = cropped_hw
    img_h, img_w = image_hw
    top = max(0, (img_h - crop_h) // 2)
    left = max(0, (img_w - crop_w) // 2)
    cropped_masks = mask_tensor[:, top:top + crop_h, left:left + crop_w] > 0.5

    flat = cropped_masks.reshape(cropped_masks.shape[0], -1)
    patch_map = _pixel_to_patch_index(cropped_hw, grid_hw)
    patch_masks = torch.zeros((flat.shape[0], grid_hw[0] * grid_hw[1]), dtype=torch.bool)
    nz = torch.nonzero(flat, as_tuple=False)
    if nz.numel() > 0:
        patch_masks[nz[:, 0], patch_map[nz[:, 1]]] = True

    keep = patch_masks.any(dim=1)
    if min_area_ratio > 0.0:
        keep = keep & (patch_masks.float().mean(dim=1) >= float(min_area_ratio))
    patch_masks = patch_masks[keep]
    cropped_masks = cropped_masks[keep]

    if patch_masks.shape[0] == 0:
        full = torch.ones((1, grid_hw[0] * grid_hw[1]), dtype=torch.bool)
        return full, None
    return patch_masks, cropped_masks


def _adjacency_from_knn(
    mask_grid: torch.Tensor,
    grid_hw: Tuple[int, int],
    order: int,
) -> Optional[torch.Tensor]:
    if order <= 0 or mask_grid.shape[0] <= 1:
        return None
    mask_float = mask_grid.float()
    num_masks, num_patches = mask_float.shape
    grid_h, grid_w = grid_hw
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


def _adjacency_from_delaunay(
    pixel_masks: torch.Tensor,
    order: int,
) -> Optional[torch.Tensor]:
    num_masks = pixel_masks.shape[0]
    if order <= 0 or num_masks <= 1:
        return None
    coords = []
    for mask in pixel_masks:
        nz = torch.nonzero(mask, as_tuple=False).float()
        if nz.numel() == 0:
            coords.append(np.array([0.0, 0.0], dtype=np.float32))
        else:
            mean_yx = nz.mean(dim=0).cpu().numpy()
            coords.append(np.array([mean_yx[1], mean_yx[0]], dtype=np.float32))
    mask_coords = np.stack(coords, axis=0)

    adj = torch.zeros((num_masks, num_masks), dtype=torch.bool)
    if num_masks > 3:
        tri = Delaunay(mask_coords)
        for simplex in tri.simplices:
            for i in simplex:
                adj[i, simplex] = True
    else:
        nbr_list = [0, 1] if num_masks > 1 else [0]
        for i in range(num_masks):
            adj[i, nbr_list] = True
    adj = adj | adj.T
    adj.fill_diagonal_(True)
    adj_power = adj.float()
    for _ in range(1, order):
        adj_power = adj_power @ adj.float()
    return adj_power > 0


def build_mask_adjacency_revisit(
    mask_grid: torch.Tensor,
    pixel_masks: Optional[torch.Tensor],
    grid_hw: Tuple[int, int],
    order: int = 0,
    method: Literal["delaunay", "knn"] = "delaunay",
) -> Optional[torch.Tensor]:
    if method == "delaunay" and pixel_masks is not None:
        return _adjacency_from_delaunay(pixel_masks, order=order)
    return _adjacency_from_knn(mask_grid, grid_hw=grid_hw, order=order)
