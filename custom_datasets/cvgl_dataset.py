"""
    Generic UAV-Satellite Cross-View Geo-Localization dataset helpers.

    The goal is to stay close to AnyLoc's dataset style:
    - database tiles first
    - queries second
    - __getitem__ returns (image_tensor, global_index)
    - `database_num`, `queries_num`, `soft_positives_per_query` exist

    Expected JSON manifest structure for `from_json`:

    tiles.json
    [
      {
        "tile_id": "tile_0001",
        "image": "tiles/tile_0001.png",
        "center": [100.0, 200.0],
        "extent": [32.0, 32.0]
      }
    ]

    queries.json
    [
      {
        "query_id": "query_0001",
        "image": "queries/query_0001.png",
        "gt_tile_id": "tile_0001",
        "offset": [2.5, -1.0],
        "coord": [102.5, 199.0]
      }
    ]
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from utilities import CustomDataset


def _path_to_pil_img(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


mixvpr_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    T.Resize((320, 320)),
])


@dataclass
class TileItem:
    tile_id: str
    image_path: str
    center_xy: Optional[Tuple[float, float]] = None
    extent_xy: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryItem:
    query_id: str
    image_path: str
    gt_tile_id: Optional[str] = None
    offset_xy: Optional[Tuple[float, float]] = None
    coord_xy: Optional[Tuple[float, float]] = None
    candidate_tile_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _to_optional_xy(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "dx" in value and "dy" in value:
            return float(value["dx"]), float(value["dy"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Cannot parse xy value: {value}")


def _resolve_image_path(root_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.realpath(os.path.join(root_dir, value))


def _resolve_optional_metadata_paths(root_dir: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(metadata)
    for key in ["mask_path", "masks_path", "segmentation_path", "segmentations_path"]:
        value = resolved.get(key)
        if isinstance(value, str):
            resolved[key] = _resolve_image_path(root_dir, value)
    return resolved


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


def _normalize_masks_payload(payload: Any) -> Optional[List[np.ndarray]]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ["masks", "segmentations", "data", "items"]:
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, torch.Tensor):
        payload = payload.detach().cpu().numpy()
    if isinstance(payload, np.ndarray):
        if payload.ndim == 2:
            payload = [payload]
        elif payload.ndim == 3:
            payload = [payload[i] for i in range(payload.shape[0])]
        else:
            raise ValueError(f"Unsupported mask array shape: {payload.shape}")
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported mask payload type: {type(payload)}")

    masks: List[np.ndarray] = []
    for item in payload:
        if isinstance(item, dict):
            if "segmentation" in item:
                item = item["segmentation"]
            elif "mask" in item:
                item = item["mask"]
        mask = np.asarray(item)
        if mask.ndim != 2:
            raise ValueError(f"Each mask must be 2D, got shape {mask.shape}")
        masks.append(mask.astype(bool))
    return masks if len(masks) > 0 else None


def _load_masks_from_path(path: str) -> Optional[List[np.ndarray]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        payload = np.load(path, allow_pickle=True)
    elif ext == ".npz":
        npz = np.load(path, allow_pickle=True)
        key = "masks" if "masks" in npz else list(npz.keys())[0]
        payload = npz[key]
    elif ext in [".pt", ".pth"]:
        payload = torch.load(path, map_location="cpu")
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raise ValueError(f"Unsupported mask file format: {path}")
    return _normalize_masks_payload(payload)


def get_example_cvgl_manifest() -> Dict[str, List[dict]]:
    """
        Minimal template that downstream users can copy to their own
        dataset manifests.
    """
    return {
        "tiles": [
            {
                "tile_id": "tile_0001",
                "image": "tiles/tile_0001.png",
                "center": [0.0, 0.0],
                "extent": [64.0, 64.0],
            }
        ],
        "queries": [
            {
                "query_id": "query_0001",
                "image": "queries/query_0001.png",
                "gt_tile_id": "tile_0001",
                "offset": [8.0, -4.0],
                "coord": [8.0, -4.0],
            }
        ],
    }


class CrossViewTileDataset(CustomDataset):
    """
        Generic CVGL dataset with AnyLoc-compatible indexing.
    """
    def __init__(
        self,
        tiles: Sequence[TileItem],
        queries: Sequence[QueryItem],
        use_mixvpr: bool = False,
        use_sam: bool = False,
    ) -> None:
        super().__init__()
        if len(tiles) == 0:
            raise ValueError("CrossViewTileDataset requires at least one tile")
        if len(queries) == 0:
            raise ValueError("CrossViewTileDataset requires at least one query")
        self.tiles = list(tiles)
        self.queries = list(queries)
        self.use_mixvpr = use_mixvpr
        self.use_sam = use_sam
        self.dataset_root: Optional[str] = None
        self.database_num = len(self.tiles)
        self.queries_num = len(self.queries)
        self.images_paths = [tile.image_path for tile in self.tiles] + \
                [query.image_path for query in self.queries]
        self.tile_id_to_index = {
            tile.tile_id: idx for idx, tile in enumerate(self.tiles)
        }
        self.soft_positives_per_query = []
        for query in self.queries:
            positives = []
            if query.gt_tile_id is not None:
                if query.gt_tile_id not in self.tile_id_to_index:
                    raise KeyError(
                        f"Query {query.query_id} gt_tile_id "
                        f"{query.gt_tile_id} not found in tiles"
                    )
                positives.append(self.tile_id_to_index[query.gt_tile_id])
            if query.candidate_tile_ids is not None:
                for tile_id in query.candidate_tile_ids:
                    if tile_id in self.tile_id_to_index:
                        positives.append(self.tile_id_to_index[tile_id])
            self.soft_positives_per_query.append(sorted(set(positives)))

    @classmethod
    def from_json(
        cls,
        dataset_root: str,
        split: str = "test",
        tiles_manifest: Optional[str] = None,
        queries_manifest: Optional[str] = None,
        use_mixvpr: bool = False,
        use_sam: bool = False,
    ) -> "CrossViewTileDataset":
        dataset_root = os.path.realpath(os.path.expanduser(dataset_root))
        if tiles_manifest is None:
            split_tiles = os.path.join(dataset_root, f"tiles_{split}.json")
            tiles_manifest = split_tiles \
                if os.path.isfile(split_tiles) else os.path.join(dataset_root, "tiles.json")
        if queries_manifest is None:
            split_queries = os.path.join(dataset_root, f"queries_{split}.json")
            queries_manifest = split_queries \
                if os.path.isfile(split_queries) else os.path.join(dataset_root, "queries.json")
        if not os.path.isfile(tiles_manifest):
            raise FileNotFoundError(f"Tiles manifest not found: {tiles_manifest}")
        if not os.path.isfile(queries_manifest):
            raise FileNotFoundError(f"Queries manifest not found: {queries_manifest}")

        tile_entries = _read_manifest(tiles_manifest)
        query_entries = _read_manifest(queries_manifest)

        tiles: List[TileItem] = []
        for i, entry in enumerate(tile_entries):
            tile_id = str(entry.get("tile_id", f"tile_{i:06d}"))
            image_rel = entry.get("image") or entry.get("image_path")
            if image_rel is None:
                raise KeyError(f"Tile entry missing image path: {entry}")
            tiles.append(TileItem(
                tile_id=tile_id,
                image_path=_resolve_image_path(dataset_root, image_rel),
                center_xy=_to_optional_xy(entry.get("center")),
                extent_xy=_to_optional_xy(entry.get("extent")),
                metadata=_resolve_optional_metadata_paths(dataset_root, {
                    k: v for k, v in entry.items()
                    if k not in ["tile_id", "image", "image_path", "center", "extent"]
                }),
            ))

        queries: List[QueryItem] = []
        for i, entry in enumerate(query_entries):
            query_id = str(entry.get("query_id", f"query_{i:06d}"))
            image_rel = entry.get("image") or entry.get("image_path")
            if image_rel is None:
                raise KeyError(f"Query entry missing image path: {entry}")
            gt_tile_id = entry.get("gt_tile_id")
            if gt_tile_id is None and "positives" in entry and len(entry["positives"]) > 0:
                gt_tile_id = entry["positives"][0]
            queries.append(QueryItem(
                query_id=query_id,
                image_path=_resolve_image_path(dataset_root, image_rel),
                gt_tile_id=None if gt_tile_id is None else str(gt_tile_id),
                offset_xy=_to_optional_xy(entry.get("offset")),
                coord_xy=_to_optional_xy(entry.get("coord")),
                candidate_tile_ids=entry.get("positives"),
                metadata=_resolve_optional_metadata_paths(dataset_root, {
                    k: v for k, v in entry.items()
                    if k not in [
                        "query_id", "image", "image_path", "gt_tile_id",
                        "offset", "coord", "positives"
                    ]
                }),
            ))
        dataset = cls(tiles, queries, use_mixvpr=use_mixvpr, use_sam=use_sam)
        dataset.dataset_root = dataset_root
        return dataset

    def __getitem__(self, index: int):
        image_path = self.images_paths[index]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if self.use_sam:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, index
        pil_img = _path_to_pil_img(image_path)
        if self.use_mixvpr:
            img = mixvpr_transform(pil_img)
        else:
            img = base_transform(pil_img)
        return img, index

    def get_tile_item(self, db_index: int) -> TileItem:
        return self.tiles[db_index]

    def get_query_item(self, query_index: int) -> QueryItem:
        return self.queries[query_index]

    def get_query_global_index(self, query_index: int) -> int:
        return self.database_num + query_index

    def has_offset_gt(self) -> bool:
        return any(query.offset_xy is not None for query in self.queries)

    def get_query_gt_db_index(self, query_index: int) -> Optional[int]:
        positives = self.soft_positives_per_query[query_index]
        if len(positives) == 0:
            return None
        return positives[0]

    def get_query_gt_offset(self, query_index: int) -> Optional[Tuple[float, float]]:
        return self.queries[query_index].offset_xy

    def get_query_coord(self, query_index: int) -> Optional[Tuple[float, float]]:
        return self.queries[query_index].coord_xy

    def get_tile_center(self, db_index: int) -> Optional[Tuple[float, float]]:
        return self.tiles[db_index].center_xy

    def get_tile_extent(self, db_index: int) -> Optional[Tuple[float, float]]:
        return self.tiles[db_index].extent_xy

    def get_tile_id(self, db_index: int) -> str:
        return self.tiles[db_index].tile_id

    def get_query_id(self, query_index: int) -> str:
        return self.queries[query_index].query_id

    def get_image_metadata(self, index: int) -> Dict[str, Any]:
        if index < self.database_num:
            return self.tiles[index].metadata
        return self.queries[index - self.database_num].metadata

    def get_segmentation_masks(self, index: int) -> Optional[List[np.ndarray]]:
        metadata = self.get_image_metadata(index)
        for key in ["masks", "segmentations"]:
            if key in metadata:
                return _normalize_masks_payload(metadata[key])
        for key in ["mask_path", "masks_path", "segmentation_path", "segmentations_path"]:
            path = metadata.get(key)
            if isinstance(path, str) and os.path.isfile(path):
                return _load_masks_from_path(path)
        return None


class BaseDatasetCVGLAdapter:
    """
        Adapter that exposes an AnyLoc/BaseDataset-style VPR dataset
        as a CVGL dataset in retrieval-only mode.
    """
    def __init__(self, base_dataset) -> None:
        self.base_dataset = base_dataset
        self.database_num = base_dataset.database_num
        self.queries_num = base_dataset.queries_num
        self.soft_positives_per_query = base_dataset.soft_positives_per_query

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        return self.base_dataset[index]

    def get_image_paths(self):
        return self.base_dataset.get_image_paths()

    def get_image_relpaths(self, i: Union[int, List[int]]):
        return self.base_dataset.get_image_relpaths(i)

    def has_offset_gt(self) -> bool:
        return False

    def get_query_gt_db_index(self, query_index: int) -> Optional[int]:
        positives = self.soft_positives_per_query[query_index]
        if len(positives) == 0:
            return None
        return positives[0]

    def get_query_gt_offset(self, query_index: int):
        return None

    def get_query_coord(self, query_index: int):
        return None

    def get_tile_center(self, db_index: int):
        return None

    def get_tile_extent(self, db_index: int):
        return None

    def get_tile_id(self, db_index: int) -> str:
        rel = self.get_image_relpaths(db_index)
        return str(rel)

    def get_query_id(self, query_index: int) -> str:
        rel = self.get_image_relpaths(self.database_num + query_index)
        return str(rel)

    def get_image_metadata(self, index: int):
        return {}

    def get_segmentation_masks(self, index: int):
        return None
