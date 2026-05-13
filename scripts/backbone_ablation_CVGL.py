# Dense-feature backbone ablation entrypoint for UAV-Satellite CVGL
"""
    Pipeline:
    - build a dense feature extractor adapter for the requested backbone
    - keep the existing VLAD / SegVLAD retrieval code path unchanged
    - optionally rerank with local dense features
    - optionally estimate offsets from local similarity maps

    This script is intentionally separate from dino_v2_vlad_CVGL.py and
    dino_v2_segvlad_CVGL.py so the current DINOv2 baselines remain intact.
"""

import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import List, Literal, Optional, Union

dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARN: __file__ not found, trying local")
    dir_name = os.path.abspath("")
lib_path = os.path.realpath(f"{Path(dir_name).parent}")
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import wandb

from configs import BaseDatasetArgs, ProgArgs, base_dataset_args, device
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.cvgl_dataset import (
    BaseDatasetCVGLAdapter,
    CrossViewTileDataset,
    get_example_cvgl_manifest,
)
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.gardens import Gardens
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.vpair_dataloader import VPAir
from cvgl_metrics import (
    evaluate_cvgl_retrieval,
    evaluate_localization,
    evaluate_offset_predictions,
)
from cvgl_retrieval import (
    build_feature_bank,
    build_segvlad_feature_bank,
    coarse_retrieve_topk,
    coarse_retrieve_topk_segvlad,
    fit_vlad_for_dataset,
    predict_top1_offsets,
    predict_top1_offsets_sliding_ncc,
    rerank_with_local_features,
)
from dense_feature_extractors import build_dense_feature_extractor
from dvgl_benchmark.datasets_ws import BaseDataset
from segvlad_masking import SAMGenerator, SegmentorConfig
from utilities import reduce_pca, seed_everything


def _to_jsonable(value):
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class LocalArgs:
    prog: ProgArgs = ProgArgs(
        cache_dir=Path(".cache/backbone_ablation"),
        wandb_proj="Backbone-Ablation",
        wandb_group="CVGL",
    )
    bd_args: BaseDatasetArgs = base_dataset_args
    exp_id: Union[str, None] = None
    task_mode: Literal["vpr", "cvgl"] = "cvgl"
    agg_method: Literal["segvlad", "vlad"] = "segvlad"

    backbone: Literal["dinov2", "dino", "dinov3_hf", "clip", "resnet50", "timm_feature"] = "dinov2"
    model_name: str = "dinov2_vits14"
    """
        Backbone-specific model identifier.
        Examples:
        - dinov2: dinov2_vits14, dinov2_vitb14
        - dino: dino_vits8, dino_vitb16
        - clip: ViT-B/32, ViT-B/16
        - resnet50: ignored
        - timm_feature: any timm features_only model name
        - dinov3_hf: pass an installed/cached Hugging Face DINOv3 model id
    """
    pretrained: bool = True
    desc_layer: int = 11
    desc_facet: Literal["query", "key", "value", "token"] = "key"
    cnn_layer: Literal["layer3", "layer4"] = "layer4"
    timm_out_index: int = -1
    clip_download_root: Union[str, None] = None
    clip_use_projection: bool = True
    patch_stride_override: Union[int, None] = None
    norm_descs: bool = True

    pca_dim_reduce: Union[int, None] = None
    pca_whitening: bool = True
    num_clusters: int = 8
    data_split: Literal["train", "test", "val"] = "test"
    sub_sample_qu: int = 1
    sub_sample_db: int = 1
    sub_sample_db_vlad: int = 1
    top_k_vals: List[int] = field(default_factory=lambda: list(range(1, 21)))
    show_plot: bool = False
    vlad_assignment: Literal["hard", "soft"] = "hard"
    vlad_soft_temp: float = 1.0
    pretrained_vlad_centers: Union[str, None] = None
    cache_dense_features: bool = True
    cache_vlad_descs: bool = False
    global_agg: Literal["vlad", "gem", "cls", "gap", "gmp"] = "vlad"
    gem_p: float = 3.0
    gem_use_abs: bool = False
    gem_elem_by_elem: bool = False

    use_local_rerank: bool = False
    coarse_top_k: int = 10
    coarse_device: str = "auto"
    local_match_method: Literal["cosine_pool", "mutual_nn", "sim_map"] = "sim_map"
    rerank_alpha: float = 0.5
    local_top_m: int = 32
    use_offset_head: bool = True
    offset_prediction_method: Literal["none", "sim_map", "slide_ncc"] = "slide_ncc"
    tile_size_m: Union[float, None] = None
    tile_size_px: Union[int, None] = None
    localization_thresholds: List[float] = field(default_factory=lambda: [16.0, 32.0, 64.0])
    cvgl_dataset_root: Union[str, None] = None
    cvgl_tiles_manifest: Union[str, None] = None
    cvgl_queries_manifest: Union[str, None] = None

    segment_descriptor: Literal["segvlad", "sap"] = "segvlad"
    segment_top_k: int = 100
    coarse_aggregation: Literal["sum", "revisit_weighted_borda_image"] = "revisit_weighted_borda_image"
    coarse_db_segment_chunk_size: int = 0
    coarse_query_batch_size: int = 16
    segvlad_min_mask_area_ratio: float = 0.0
    segvlad_neighbor_order: int = 0
    segvlad_centroid_pe_num_freqs: int = 0
    segvlad_centroid_pe_weight: float = 0.2
    segvlad_relative_context_num_freqs: int = 0
    segvlad_relative_context_weight: float = 0.0
    segvlad_relative_context_order: int = 1
    segvlad_relative_context_ref_grid_size: float = 0.0
    seg_cfg: SegmentorConfig = field(default_factory=SegmentorConfig)


def _load_anyloc_dataset(largs: LocalArgs):
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    if ds_name == "baidu_datasets":
        return Baidu_Dataset(largs.bd_args, ds_dir, ds_name, largs.data_split)
    if ds_name == "Oxford":
        return Oxford(ds_dir)
    if ds_name == "Oxford_25m":
        return Oxford(ds_dir, override_dist=25)
    if ds_name == "gardens":
        return Gardens(largs.bd_args, ds_dir, ds_name, largs.data_split)
    if ds_name.startswith("Tartan_GNSS"):
        return Aerial(largs.bd_args, ds_dir, ds_name, largs.data_split)
    if ds_name.startswith("hawkins"):
        return Hawkins(largs.bd_args, ds_dir, "hawkins_long_corridor", largs.data_split)
    if ds_name == "laurel_caverns":
        return Laurel(largs.bd_args, ds_dir, ds_name, largs.data_split)
    if ds_name == "eiffel":
        return Eiffel(largs.bd_args, ds_dir, ds_name, largs.data_split)
    if ds_name in ["VPAir", "VPAir_xview2"]:
        return VPAir(largs.bd_args, ds_dir, ds_name, largs.data_split)
    return BaseDataset(largs.bd_args, ds_dir, ds_name, largs.data_split)


def _load_dataset(largs: LocalArgs):
    if largs.task_mode == "vpr":
        return BaseDatasetCVGLAdapter(_load_anyloc_dataset(largs))
    if largs.cvgl_dataset_root is None:
        raise ValueError(
            "CVGL mode requires `cvgl_dataset_root`. "
            f"Example manifest: {get_example_cvgl_manifest()}"
        )
    return CrossViewTileDataset.from_json(
        largs.cvgl_dataset_root,
        split=largs.data_split,
        tiles_manifest=largs.cvgl_tiles_manifest,
        queries_manifest=largs.cvgl_queries_manifest,
    )


def _dataset_cache_id(largs: LocalArgs) -> str:
    if largs.task_mode == "vpr":
        return str(largs.prog.vg_dataset_name)
    cache_parts = [
        os.path.realpath(os.path.expanduser(str(largs.cvgl_dataset_root))),
        str(largs.data_split),
        "" if largs.cvgl_tiles_manifest is None else os.path.realpath(
            os.path.expanduser(str(largs.cvgl_tiles_manifest))
        ),
        "" if largs.cvgl_queries_manifest is None else os.path.realpath(
            os.path.expanduser(str(largs.cvgl_queries_manifest))
        ),
    ]
    dataset_root_name = Path(str(largs.cvgl_dataset_root)).name or "cvgl"
    dataset_hash = hashlib.sha1("::".join(cache_parts).encode("utf-8")).hexdigest()[:12]
    return f"{dataset_root_name}-{dataset_hash}"


def _build_cache_roots(largs: LocalArgs, feature_cache_id: str):
    base_dir = os.path.join(
        str(largs.prog.cache_dir),
        "cvgl_descs",
        _dataset_cache_id(largs),
        f"{feature_cache_id}-{largs.agg_method}-{largs.segment_descriptor}-{largs.global_agg}",
    )
    dense_root = os.path.join(base_dir, "dense") if largs.cache_dense_features else None
    vlad_root = None
    if largs.cache_vlad_descs or (
        largs.agg_method == "segvlad" and largs.segment_descriptor == "segvlad"
    ):
        vlad_root = os.path.join(base_dir, f"vlad-C{largs.num_clusters}")
    masks_root = None
    if largs.agg_method == "segvlad" and largs.seg_cfg.cache_generated_masks:
        masks_root = os.path.join(base_dir, "sam_masks")
    return dense_root, vlad_root, masks_root


def _resolve_pretrained_vlad_cache_root(largs: LocalArgs) -> Optional[str]:
    if largs.pretrained_vlad_centers is None:
        return None
    c_centers_path = os.path.abspath(os.path.expanduser(str(largs.pretrained_vlad_centers)))
    if not os.path.isfile(c_centers_path):
        raise FileNotFoundError(f"Pretrained VLAD centers not found: {c_centers_path}")
    if os.path.basename(c_centers_path) != "c_centers.pt":
        raise ValueError(
            "pretrained_vlad_centers must point to a file named `c_centers.pt`, "
            f"got: {c_centers_path}"
        )
    c_centers = torch.load(c_centers_path, map_location="cpu")
    if c_centers.ndim != 2:
        raise ValueError(
            f"Invalid pretrained VLAD centers shape: {tuple(c_centers.shape)}. "
            "Expected [num_clusters, desc_dim]."
        )
    if c_centers.shape[0] != int(largs.num_clusters):
        raise ValueError(
            "pretrained_vlad_centers cluster count does not match "
            f"`num_clusters`: {c_centers.shape[0]} vs {largs.num_clusters}"
        )
    print(f"Using pretrained VLAD centers: {c_centers_path}")
    print(f"Pretrained VLAD centers shape: {tuple(c_centers.shape)}")
    return os.path.dirname(c_centers_path)


def _collect_gt_db_indices(dataset, db_indices, qu_indices) -> List[Optional[int]]:
    db_index_map = {int(orig_idx): bank_idx for bank_idx, orig_idx in enumerate(db_indices.tolist())}
    gt_db_indices: List[Optional[int]] = []
    for q_global_idx in qu_indices.tolist():
        dataset_qi = int(q_global_idx - dataset.database_num)
        positives = dataset.soft_positives_per_query[dataset_qi]
        mapped = [db_index_map[pos] for pos in positives if pos in db_index_map]
        gt_db_indices.append(mapped[0] if len(mapped) > 0 else None)
    return gt_db_indices


def _maybe_apply_pca(largs: LocalArgs, db_global: torch.Tensor, qu_global: torch.Tensor):
    if largs.pca_dim_reduce is None:
        return db_global, qu_global
    norm_descs = lambda x: torch.from_numpy(
        x / np.linalg.norm(x, axis=-1, keepdims=True)
    ).to(db_global.device).contiguous()
    down_db, down_qu = reduce_pca(
        db_global.cpu().numpy(),
        qu_global.cpu().numpy(),
        largs.pca_dim_reduce,
        whitening=largs.pca_whitening,
    )
    return norm_descs(down_db), norm_descs(down_qu)


def _build_run_config_snapshot(
    largs: LocalArgs,
    dataset,
    feature_cache_id: str,
    dense_cache_root: Optional[str],
    vlad_cache_root: Optional[str],
    masks_cache_root: Optional[str],
    pretrained_vlad_cache_root: Optional[str],
    patch_stride: int,
    seg_cfg_runtime: SegmentorConfig,
):
    return {
        "cli_args": _to_jsonable(largs),
        "resolved": {
            "cwd": os.getcwd(),
            "argv": list(sys.argv),
            "task_mode": str(largs.task_mode),
            "dataset_class": dataset.__class__.__name__,
            "database_num": int(dataset.database_num),
            "queries_num": int(dataset.queries_num),
            "device": str(device),
            "feature_cache_id": str(feature_cache_id),
            "patch_stride": int(patch_stride),
            "dense_cache_root": None if dense_cache_root is None else os.path.abspath(dense_cache_root),
            "vlad_cache_root": None if vlad_cache_root is None else os.path.abspath(vlad_cache_root),
            "masks_cache_root": None if masks_cache_root is None else os.path.abspath(masks_cache_root),
            "pretrained_vlad_cache_root": (
                None if pretrained_vlad_cache_root is None else os.path.abspath(pretrained_vlad_cache_root)
            ),
            "cvgl_dataset_root": (
                None if largs.cvgl_dataset_root is None
                else os.path.abspath(os.path.expanduser(str(largs.cvgl_dataset_root)))
            ),
            "cvgl_tiles_manifest": (
                None if largs.cvgl_tiles_manifest is None
                else os.path.abspath(os.path.expanduser(str(largs.cvgl_tiles_manifest)))
            ),
            "cvgl_queries_manifest": (
                None if largs.cvgl_queries_manifest is None
                else os.path.abspath(os.path.expanduser(str(largs.cvgl_queries_manifest)))
            ),
            "seg_cfg_runtime": _to_jsonable(seg_cfg_runtime),
        },
    }


def _build_feature_extractor(largs: LocalArgs):
    if largs.backbone == "resnet50" and largs.model_name == "dinov2_vits14":
        print("INFO: model_name is ignored for backbone='resnet50'")
    return build_dense_feature_extractor(
        backbone=largs.backbone,
        model_name=largs.model_name,
        layer=largs.desc_layer,
        facet=largs.desc_facet,
        device=str(device),
        pretrained=largs.pretrained,
        cnn_layer=largs.cnn_layer,
        clip_download_root=largs.clip_download_root,
        clip_use_projection=largs.clip_use_projection,
        timm_out_index=largs.timm_out_index,
        patch_stride_override=largs.patch_stride_override,
        norm_descs=largs.norm_descs,
    )


def _predict_offsets(
    largs: LocalArgs,
    dataset,
    retrieval_indices,
    db_bank,
    qu_bank,
    sampled_query_indices,
):
    pred_offsets = []
    sim_maps = []
    offset_metrics = {}
    localization_metrics = {}
    tile_extent_default = None
    if largs.tile_size_m is not None:
        tile_extent_default = (float(largs.tile_size_m), float(largs.tile_size_m))
    elif largs.tile_size_px is not None:
        tile_extent_default = (float(largs.tile_size_px), float(largs.tile_size_px))
    if largs.use_offset_head and largs.offset_prediction_method != "none":
        if largs.offset_prediction_method == "sim_map":
            pred_offsets, sim_maps = predict_top1_offsets(
                dataset,
                retrieval_indices,
                db_bank=db_bank,
                query_bank=qu_bank,
                tile_extent_default=tile_extent_default,
                local_top_m=largs.local_top_m,
                query_indices=sampled_query_indices,
            )
        elif largs.offset_prediction_method == "slide_ncc":
            pred_offsets, sim_maps = predict_top1_offsets_sliding_ncc(
                dataset,
                retrieval_indices,
                db_bank=db_bank,
                query_bank=qu_bank,
                tile_extent_default=tile_extent_default,
                query_indices=sampled_query_indices,
            )
        else:
            raise ValueError(f"Unknown offset prediction method: {largs.offset_prediction_method}")
        gt_offsets = [dataset.get_query_gt_offset(qi) for qi in sampled_query_indices]
        offset_metrics = evaluate_offset_predictions(
            pred_offsets,
            gt_offsets,
            thresholds=largs.localization_thresholds,
        )
        localization_metrics = evaluate_localization(
            dataset,
            retrieval_indices[:, 0].tolist(),
            pred_offsets,
            thresholds=largs.localization_thresholds,
            query_indices=sampled_query_indices,
            db_indices_map=db_bank.indices,
        )
    return pred_offsets, sim_maps, offset_metrics, localization_metrics


@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)
    dataset = _load_dataset(largs)
    extractor = _build_feature_extractor(largs)
    patch_stride = int(extractor.patch_stride)
    dense_cache_root, vlad_cache_root, masks_cache_root = _build_cache_roots(largs, extractor.cache_id)
    pretrained_vlad_cache_root = _resolve_pretrained_vlad_cache_root(largs)
    if pretrained_vlad_cache_root is not None:
        if largs.agg_method == "segvlad" and largs.segment_descriptor != "segvlad":
            raise ValueError("pretrained_vlad_centers can only be used with segment_descriptor='segvlad'")
        vlad_cache_root = pretrained_vlad_cache_root

    seg_cfg_runtime = largs.seg_cfg
    sam_generator = None
    if largs.agg_method == "segvlad" and largs.seg_cfg.source in ["sam", "manifest_or_sam"]:
        try:
            sam_generator = SAMGenerator(largs.seg_cfg, device=str(device))
        except Exception as exc:
            if largs.seg_cfg.source == "sam":
                raise
            print(f"WARN: SAM generator unavailable, falling back to manifest/full-image masks: {exc}")
            seg_cfg_runtime = replace(largs.seg_cfg, source="manifest")

    run_config = _build_run_config_snapshot(
        largs,
        dataset=dataset,
        feature_cache_id=extractor.cache_id,
        dense_cache_root=dense_cache_root,
        vlad_cache_root=vlad_cache_root,
        masks_cache_root=masks_cache_root,
        pretrained_vlad_cache_root=pretrained_vlad_cache_root,
        patch_stride=patch_stride,
        seg_cfg_runtime=seg_cfg_runtime,
    )

    wandb_run = None
    if largs.prog.use_wandb:
        wandb_run = wandb.init(
            project=largs.prog.wandb_proj,
            entity=largs.prog.wandb_entity,
            config=largs,
            group=largs.prog.wandb_group,
            name=largs.prog.wandb_run_name,
        )
        print(f"Initialized WandB run: {wandb_run.name}")

    print("--------- Building feature banks ---------")
    db_indices = np.arange(0, dataset.database_num, largs.sub_sample_db)
    qu_indices = np.arange(
        dataset.database_num,
        dataset.database_num + dataset.queries_num,
        largs.sub_sample_qu,
    )
    sampled_query_indices = [int(idx - dataset.database_num) for idx in qu_indices.tolist()]

    vlad = None
    if (
        (largs.agg_method == "segvlad" and largs.segment_descriptor == "segvlad")
        or (largs.agg_method == "vlad" and largs.global_agg == "vlad")
    ):
        vlad = fit_vlad_for_dataset(
            dataset,
            db_indices=db_indices,
            dino=extractor,
            device=device,
            num_clusters=largs.num_clusters,
            sub_sample_db_vlad=largs.sub_sample_db_vlad,
            vlad_assignment=largs.vlad_assignment,
            vlad_soft_temp=largs.vlad_soft_temp,
            feature_cache_root=dense_cache_root,
            vlad_cache_root=vlad_cache_root,
            patch_stride=patch_stride,
        )

    if largs.agg_method == "segvlad":
        relative_context_ref_grid_hw = None
        if largs.segvlad_relative_context_ref_grid_size > 0.0:
            relative_context_ref_grid_hw = (
                float(largs.segvlad_relative_context_ref_grid_size),
                float(largs.segvlad_relative_context_ref_grid_size),
            )
        db_bank = build_segvlad_feature_bank(
            dataset,
            indices=db_indices,
            dino=extractor,
            device=device,
            vlad=vlad,
            feature_cache_root=dense_cache_root,
            masks_cache_root=masks_cache_root,
            patch_stride=patch_stride,
            segment_descriptor=largs.segment_descriptor,
            min_mask_area_ratio=largs.segvlad_min_mask_area_ratio,
            neighbor_order=largs.segvlad_neighbor_order,
            centroid_pe_num_freqs=largs.segvlad_centroid_pe_num_freqs,
            centroid_pe_weight=largs.segvlad_centroid_pe_weight,
            relative_context_num_freqs=largs.segvlad_relative_context_num_freqs,
            relative_context_weight=largs.segvlad_relative_context_weight,
            relative_context_order=largs.segvlad_relative_context_order,
            relative_context_ref_grid_hw=relative_context_ref_grid_hw,
            seg_cfg=seg_cfg_runtime,
            sam_generator=sam_generator,
        )
        qu_bank = build_segvlad_feature_bank(
            dataset,
            indices=qu_indices,
            dino=extractor,
            device=device,
            vlad=vlad,
            feature_cache_root=dense_cache_root,
            masks_cache_root=masks_cache_root,
            patch_stride=patch_stride,
            segment_descriptor=largs.segment_descriptor,
            min_mask_area_ratio=largs.segvlad_min_mask_area_ratio,
            neighbor_order=largs.segvlad_neighbor_order,
            centroid_pe_num_freqs=largs.segvlad_centroid_pe_num_freqs,
            centroid_pe_weight=largs.segvlad_centroid_pe_weight,
            relative_context_num_freqs=largs.segvlad_relative_context_num_freqs,
            relative_context_weight=largs.segvlad_relative_context_weight,
            relative_context_order=largs.segvlad_relative_context_order,
            relative_context_ref_grid_hw=relative_context_ref_grid_hw,
            seg_cfg=seg_cfg_runtime,
            sam_generator=sam_generator,
        )
        print("--------- SegVLAD banks ready ---------")
        coarse_scores, coarse_indices = coarse_retrieve_topk_segvlad(
            db_bank,
            qu_bank,
            top_k=max(max(largs.top_k_vals), largs.coarse_top_k),
            segment_top_k=largs.segment_top_k,
            aggregation=largs.coarse_aggregation,
            device=device if largs.coarse_device == "auto" else largs.coarse_device,
            db_segment_chunk_size=largs.coarse_db_segment_chunk_size,
            query_batch_size=largs.coarse_query_batch_size,
        )
    else:
        db_bank = build_feature_bank(
            dataset,
            indices=db_indices,
            dino=extractor,
            device=device,
            global_agg=largs.global_agg,
            vlad=vlad if largs.global_agg == "vlad" else None,
            feature_cache_root=dense_cache_root,
            gem_p=largs.gem_p,
            gem_use_abs=largs.gem_use_abs,
            gem_elem_by_elem=largs.gem_elem_by_elem,
            patch_stride=patch_stride,
        )
        qu_bank = build_feature_bank(
            dataset,
            indices=qu_indices,
            dino=extractor,
            device=device,
            global_agg=largs.global_agg,
            vlad=vlad if largs.global_agg == "vlad" else None,
            feature_cache_root=dense_cache_root,
            gem_p=largs.gem_p,
            gem_use_abs=largs.gem_use_abs,
            gem_elem_by_elem=largs.gem_elem_by_elem,
            patch_stride=patch_stride,
        )
        print("--------- VLAD/GeM banks ready ---------")
        db_global, qu_global = _maybe_apply_pca(largs, db_bank.global_descs, qu_bank.global_descs)
        db_bank.global_descs = db_global
        qu_bank.global_descs = qu_global
        coarse_device = device if largs.coarse_device == "auto" else torch.device(largs.coarse_device)
        coarse_scores, coarse_indices = coarse_retrieve_topk(
            db_bank.global_descs.to(coarse_device),
            qu_bank.global_descs.to(coarse_device),
            top_k=max(max(largs.top_k_vals), largs.coarse_top_k),
        )

    gt_db_indices = _collect_gt_db_indices(dataset, db_indices, qu_indices)
    coarse_metrics = evaluate_cvgl_retrieval(
        coarse_indices[:, :max(largs.top_k_vals)],
        gt_db_indices,
        largs.top_k_vals,
    )

    retrieval_scores = coarse_scores[:, :largs.coarse_top_k]
    retrieval_indices = coarse_indices[:, :largs.coarse_top_k]
    local_score_table = None
    refined_metrics = {}
    if largs.use_local_rerank:
        retrieval_scores, retrieval_indices, local_score_table = rerank_with_local_features(
            retrieval_scores,
            retrieval_indices,
            db_bank=db_bank,
            query_bank=qu_bank,
            local_match_method=largs.local_match_method,
            rerank_alpha=largs.rerank_alpha,
            local_top_m=largs.local_top_m,
        )
        refined_metrics = evaluate_cvgl_retrieval(
            retrieval_indices[:, :max(largs.top_k_vals)],
            gt_db_indices,
            largs.top_k_vals,
        )

    pred_offsets, sim_maps, offset_metrics, localization_metrics = _predict_offsets(
        largs,
        dataset,
        retrieval_indices,
        db_bank,
        qu_bank,
        sampled_query_indices,
    )

    results = {
        "CVGL-Dataset": str(largs.cvgl_dataset_root),
        "Task-Mode": str(largs.task_mode),
        "Agg-Method": str(largs.agg_method),
        "Global-Agg": str(largs.global_agg),
        "Backbone": str(largs.backbone),
        "Model-Name": str(largs.model_name),
        "Feature-Cache-ID": str(extractor.cache_id),
        "Patch-Stride": int(patch_stride),
        "Desc-Layer": int(largs.desc_layer),
        "Desc-Facet": str(largs.desc_facet),
        "Pretrained": bool(largs.pretrained),
        "Pretrained-VLAD-Centers": (
            None if largs.pretrained_vlad_centers is None
            else os.path.abspath(os.path.expanduser(str(largs.pretrained_vlad_centers)))
        ),
        "Num-DB": str(len(db_bank.indices)),
        "Num-QU": str(len(qu_bank.indices)),
        "Coarse-TopK": str(largs.coarse_top_k),
        "Local-Rerank": bool(largs.use_local_rerank),
        "Offset-Method": str(largs.offset_prediction_method),
        "Timestamp": time.strftime("%Y_%m_%d_%H_%M_%S"),
        "Run-Config": run_config,
    }
    if largs.agg_method == "segvlad":
        results.update({
            "Num-DB-Segments": int(db_bank.segment_descs.shape[0]),
            "Num-QU-Segments": int(qu_bank.segment_descs.shape[0]),
            "Segment-Descriptor": str(largs.segment_descriptor),
            "Segment-TopK": int(largs.segment_top_k),
            "Coarse-Aggregation": str(largs.coarse_aggregation),
            "SegVLAD-Centroid-PE-Freqs": int(largs.segvlad_centroid_pe_num_freqs),
            "SegVLAD-Centroid-PE-Weight": float(largs.segvlad_centroid_pe_weight),
            "SegVLAD-Relative-Context-Freqs": int(largs.segvlad_relative_context_num_freqs),
            "SegVLAD-Relative-Context-Weight": float(largs.segvlad_relative_context_weight),
        })
    for key, val in coarse_metrics.items():
        results[f"Coarse-{key}"] = val
    for key, val in refined_metrics.items():
        results[f"Refined-{key}"] = val
    results.update(offset_metrics)
    results.update(localization_metrics)

    print("--------------------- Results ---------------------")
    for key, val in results.items():
        print(f"- {key}: {val}")

    if largs.show_plot:
        plot_metrics = refined_metrics if len(refined_metrics) > 0 else coarse_metrics
        recall_keys = [k for k in plot_metrics if k.startswith("R@")]
        recall_keys = sorted(recall_keys, key=lambda x: int(x.split("@")[1]))
        plt.plot([int(x.split("@")[1]) for x in recall_keys], [plot_metrics[x] for x in recall_keys])
        plt.ylim(0, 1)
        plt.xticks(largs.top_k_vals)
        plt.xlabel("top-k values")
        plt.ylabel("% recall")
        plt.title("CVGL Backbone Ablation Recall Curve")
        plt.show()

    if largs.prog.use_wandb:
        wandb.log(results)

    results["Coarse-Indices"] = coarse_indices
    results["Coarse-Scores"] = coarse_scores
    results["Final-Indices"] = retrieval_indices
    results["Final-DB-Indices-Orig"] = np.asarray(
        [[db_bank.indices[int(db_idx)] for db_idx in row] for row in retrieval_indices],
        dtype=np.int64,
    )
    results["Final-Scores"] = retrieval_scores
    results["Pred-Offsets"] = pred_offsets
    results["DB-Relpaths"] = db_bank.relpaths
    results["Query-Relpaths"] = qu_bank.relpaths
    results["Local-Score-Table"] = local_score_table
    if largs.agg_method == "segvlad":
        results["DB-Segment-Counts"] = db_bank.image_segment_counts
        results["Query-Segment-Counts"] = qu_bank.image_segment_counts

    save_res_file = None
    if largs.exp_id is True:
        save_res_file = str(largs.prog.cache_dir)
    elif isinstance(largs.exp_id, str):
        save_res_file = os.path.join(str(largs.prog.cache_dir), "experiments", largs.exp_id)
    if save_res_file is not None:
        os.makedirs(save_res_file, exist_ok=True)
        out_path = os.path.join(save_res_file, f"cvgl_backbone_ablation_{results['Timestamp']}.json")
        print(f"Saving result in: {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(results), f, indent=2)
    else:
        print("Not saving results")

    if largs.prog.use_wandb:
        wandb.finish()


if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    largs = tyro.cli(LocalArgs, description=__doc__)
    _start = time.time()
    try:
        main(largs)
    except Exception:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time() - _start:.3f} seconds")
        exit(0)
