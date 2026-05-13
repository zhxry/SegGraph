"""Extract DINOv2 dense feature caches for all images in a directory.

Example:
    python scripts/extract_dinov2_dense_features.py \
        --input-dir data/Hawaii_wildfire_2023_1_336_intile/tile_post \
        --output-dir .cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-value-L31-segvlad/dense/tile_post \
        --model-type dinov2_vitg14 \
        --desc-layer 31 \
        --desc-facet value
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

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

import torch
import torchvision.transforms as T
import tyro
from PIL import Image
from tqdm import tqdm

from cvgl_retrieval import extract_dense_record
from utilities import DinoV2ExtractFeatures, seed_everything


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class LocalArgs:
    input_dir: str
    """
        Directory containing images to extract.
    """
    output_dir: str
    """
        Directory where feature `.pt` files will be written.
    """
    model_type: Literal[
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    ] = "dinov2_vitg14"
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    patch_stride: int = 14
    recursive: bool = False
    """
        If True, scan input_dir recursively and preserve relative subdirectories.
    """
    overwrite: bool = False
    """
        If False, existing `.pt` files are skipped.
    """
    limit: int = 0
    """
        Optional maximum number of images to process. 0 means no limit.
    """
    seed: int = 42
    device: str = "auto"
    """
        `auto`, `cuda`, `cuda:0`, or `cpu`.
    """


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _iter_images(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [
        path for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(paths)


def _output_path(input_dir: Path, output_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(input_dir)
    return output_dir / rel.parent / f"{rel.name}.pt"


def _load_image_tensor(path: Path) -> torch.Tensor:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(path).convert("RGB"))


def main(largs: LocalArgs):
    seed_everything(int(largs.seed))
    input_dir = Path(os.path.realpath(os.path.expanduser(largs.input_dir)))
    output_dir = Path(os.path.realpath(os.path.expanduser(largs.output_dir)))
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _iter_images(input_dir, recursive=bool(largs.recursive))
    if largs.limit and largs.limit > 0:
        image_paths = image_paths[: int(largs.limit)]
    if len(image_paths) == 0:
        raise ValueError(f"No supported images found in {input_dir}")

    run_device = _resolve_device(str(largs.device))
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {largs.model_type}, layer={largs.desc_layer}, facet={largs.desc_facet}")
    print(f"Device: {run_device}")

    extractor = DinoV2ExtractFeatures(
        largs.model_type,
        int(largs.desc_layer),
        facet=largs.desc_facet,
        use_cls=False,
        device=str(run_device),
    )

    written = 0
    skipped = 0
    for image_path in tqdm(image_paths, desc="DINOv2 dense features"):
        out_path = _output_path(input_dir, output_dir, image_path)
        if out_path.is_file() and not largs.overwrite:
            skipped += 1
            continue
        img = _load_image_tensor(image_path)
        record = extract_dense_record(
            img,
            dino=extractor,
            device=run_device,
            patch_stride=int(largs.patch_stride),
            global_agg="vlad",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "local_desc": record.local_desc,
            "grid_hw": list(record.grid_hw),
            "image_hw": list(record.image_hw),
            "cropped_hw": list(record.cropped_hw),
            "patch_stride": record.patch_stride,
            "global_desc": record.global_desc,
        }, out_path)
        written += 1

    print("Done.")
    print(f"- Written: {written}")
    print(f"- Skipped existing: {skipped}")


if __name__ == "__main__":
    args = tyro.cli(LocalArgs, description=__doc__)
    main(args)
