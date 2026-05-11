"""Fit one VLAD vocabulary from dense feature caches across datasets.

This script does not concatenate existing `c_centers.pt` files. It loads dense
feature `.pt` files, merges their patch descriptors, runs K-Means, and writes a
new `c_centers.pt` that can be used by the CVGL scripts.

Example:
    python scripts/merge_vlad_vocabularies.py \
        .cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-key-L23-segvlad/dense/tiles \
        .cache/cvgl_descs/California_wildfire_2025_1_336_intile-xxxx/dinov2_vitg14-key-L23-segvlad/dense/tiles \
        --num-clusters 32 \
        --output .cache/cvgl_descs/merged_disaster/dinov2_vitg14-key-L23-segvlad/vlad-C32

Use the result with:
    --pretrained-vlad-centers <output-dir>/c_centers.pt --num-clusters 32
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from tqdm.auto import tqdm


@dataclass
class LoadedFeatureSummary:
    directory: str
    num_files_found: int
    num_files_used: int
    num_descriptors_used: int


def _resolve_output_path(output: str) -> Path:
    out_path = Path(output).expanduser()
    if out_path.suffix == ".pt":
        return out_path
    return out_path / "c_centers.pt"


def _list_feature_files(feature_dir: str, recursive: bool) -> List[Path]:
    root = Path(feature_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Feature directory not found: {root}")
    pattern = "**/*.pt" if recursive else "*.pt"
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt feature files found in: {root}")
    return files


def _select_files(
    files: Sequence[Path],
    max_files: Optional[int],
    sample_ratio: float,
    rng: random.Random,
) -> List[Path]:
    if sample_ratio <= 0.0 or sample_ratio > 1.0:
        raise ValueError("--file-sample-ratio must be in (0, 1]")
    selected = list(files)
    if sample_ratio < 1.0:
        k = max(1, int(round(len(selected) * sample_ratio)))
        selected = sorted(rng.sample(selected, k=k))
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-files-per-dir must be positive")
        selected = selected[:max_files]
    return selected


def _extract_descriptor_tensor(payload, feature_key: str, path: Path) -> torch.Tensor:
    if isinstance(payload, dict):
        if feature_key not in payload:
            keys = ", ".join(sorted(str(key) for key in payload.keys()))
            raise KeyError(f"Feature key `{feature_key}` not found in {path}. Available keys: {keys}")
        desc = payload[feature_key]
    else:
        desc = payload
    if not isinstance(desc, torch.Tensor):
        raise TypeError(f"Expected tensor feature in {path}, got {type(desc)}")
    if desc.ndim == 3:
        desc = desc.reshape(-1, desc.shape[-1])
    elif desc.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected feature shape [H, W, D] or [N, D], got {tuple(desc.shape)}: {path}")
    if desc.shape[0] == 0 or desc.shape[1] == 0:
        raise ValueError(f"Empty descriptor tensor in: {path}")
    if not torch.isfinite(desc).all():
        raise ValueError(f"Descriptor tensor contains NaN or Inf: {path}")
    return desc.detach().cpu().float()


def _sample_descriptors(
    desc: torch.Tensor,
    max_descriptors_per_file: Optional[int],
    generator: torch.Generator,
) -> torch.Tensor:
    if max_descriptors_per_file is None or desc.shape[0] <= max_descriptors_per_file:
        return desc
    if max_descriptors_per_file <= 0:
        raise ValueError("--max-descriptors-per-file must be positive")
    perm = torch.randperm(desc.shape[0], generator=generator)[:max_descriptors_per_file]
    return desc[perm]


def _cap_total_descriptors(
    descs: torch.Tensor,
    max_total_descriptors: Optional[int],
    generator: torch.Generator,
) -> torch.Tensor:
    if max_total_descriptors is None or descs.shape[0] <= max_total_descriptors:
        return descs
    if max_total_descriptors <= 0:
        raise ValueError("--max-total-descriptors must be positive")
    perm = torch.randperm(descs.shape[0], generator=generator)[:max_total_descriptors]
    return descs[perm]


def load_dense_descriptors(
    feature_dirs: Sequence[str],
    feature_key: str,
    recursive: bool,
    file_sample_ratio: float,
    max_files_per_dir: Optional[int],
    max_descriptors_per_file: Optional[int],
    max_total_descriptors: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, List[LoadedFeatureSummary]]:
    py_rng = random.Random(seed)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed)

    all_descs: List[torch.Tensor] = []
    summaries: List[LoadedFeatureSummary] = []
    desc_dim = None

    for feature_dir in feature_dirs:
        files = _list_feature_files(feature_dir, recursive=recursive)
        selected_files = _select_files(
            files,
            max_files=max_files_per_dir,
            sample_ratio=file_sample_ratio,
            rng=py_rng,
        )
        used_count = 0
        desc_count = 0
        for path in tqdm(selected_files, desc=f"Loading {Path(feature_dir).name}"):
            payload = torch.load(path, map_location="cpu")
            desc = _extract_descriptor_tensor(payload, feature_key=feature_key, path=path)
            if desc_dim is None:
                desc_dim = int(desc.shape[1])
            elif int(desc.shape[1]) != desc_dim:
                raise ValueError(
                    "All feature directories must use the same descriptor dimension. "
                    f"Expected {desc_dim}, got {int(desc.shape[1])}: {path}"
                )
            desc = _sample_descriptors(desc, max_descriptors_per_file, torch_gen)
            all_descs.append(desc)
            used_count += 1
            desc_count += int(desc.shape[0])

        summaries.append(LoadedFeatureSummary(
            directory=str(Path(feature_dir).expanduser().resolve()),
            num_files_found=len(files),
            num_files_used=used_count,
            num_descriptors_used=desc_count,
        ))

    if len(all_descs) == 0:
        raise RuntimeError("No descriptors were loaded")
    merged = torch.cat(all_descs, dim=0)
    merged = _cap_total_descriptors(merged, max_total_descriptors, torch_gen)
    return merged, summaries


def fit_merged_vocabulary(
    feature_dirs: Sequence[str],
    num_clusters: int,
    output: str,
    feature_key: str = "local_desc",
    recursive: bool = False,
    file_sample_ratio: float = 1.0,
    max_files_per_dir: Optional[int] = None,
    max_descriptors_per_file: Optional[int] = None,
    max_total_descriptors: Optional[int] = None,
    seed: int = 42,
    device: str = "auto",
    dist_mode: str = "cosine",
    normalize_descriptors: bool = True,
    metadata_path: Optional[str] = None,
) -> Dict:
    if num_clusters <= 0:
        raise ValueError("--num-clusters must be positive")
    descriptors, summaries = load_dense_descriptors(
        feature_dirs=feature_dirs,
        feature_key=feature_key,
        recursive=recursive,
        file_sample_ratio=file_sample_ratio,
        max_files_per_dir=max_files_per_dir,
        max_descriptors_per_file=max_descriptors_per_file,
        max_total_descriptors=max_total_descriptors,
        seed=seed,
    )
    if descriptors.shape[0] < num_clusters:
        raise ValueError(
            f"Need at least num_clusters descriptors, got {descriptors.shape[0]} for {num_clusters} clusters"
        )

    if device == "auto":
        fit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        fit_device = torch.device(device)
    descriptors_for_fit = descriptors.to(fit_device)
    if normalize_descriptors:
        descriptors_for_fit = F.normalize(descriptors_for_fit, dim=1)

    out_path = _resolve_output_path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kmeans = KMeans(num_clusters, mode=dist_mode)
    kmeans.fit(descriptors_for_fit)
    centers = kmeans.centroids.detach().cpu().float().contiguous()
    torch.save(centers, out_path)

    summary = {
        "output": str(out_path),
        "num_clusters": int(centers.shape[0]),
        "desc_dim": int(centers.shape[1]),
        "num_descriptors_for_fit": int(descriptors.shape[0]),
        "fit_device": str(fit_device),
        "feature_key": feature_key,
        "recursive": bool(recursive),
        "file_sample_ratio": float(file_sample_ratio),
        "max_files_per_dir": max_files_per_dir,
        "max_descriptors_per_file": max_descriptors_per_file,
        "max_total_descriptors": max_total_descriptors,
        "seed": int(seed),
        "dist_mode": str(dist_mode),
        "normalize_descriptors": bool(normalize_descriptors),
        "feature_directories": [summary.__dict__ for summary in summaries],
    }

    meta_path = Path(metadata_path).expanduser().resolve() if metadata_path else out_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    summary["metadata"] = str(meta_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "feature_dirs",
        nargs="+",
        help="Directories containing dense feature .pt files, e.g. .../dense/tiles.",
    )
    parser.add_argument(
        "--num-clusters",
        "--num-c",
        type=int,
        required=True,
        help="Number of K-Means clusters to fit for the merged VLAD vocabulary.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory or .pt file path. If a directory is given, saves `<dir>/c_centers.pt`.",
    )
    parser.add_argument(
        "--feature-key",
        default="local_desc",
        help="Tensor key to read from each feature .pt dict. Default: local_desc.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search feature directories recursively for .pt files.",
    )
    parser.add_argument(
        "--file-sample-ratio",
        type=float,
        default=1.0,
        help="Randomly sample this ratio of files from each directory before fitting. Default: 1.0.",
    )
    parser.add_argument(
        "--max-files-per-dir",
        type=int,
        default=None,
        help="Optional cap on the number of feature files loaded per directory.",
    )
    parser.add_argument(
        "--max-descriptors-per-file",
        type=int,
        default=None,
        help="Optional random cap on patch descriptors loaded from each feature file.",
    )
    parser.add_argument(
        "--max-total-descriptors",
        type=int,
        default=None,
        help="Optional random cap after descriptors from all directories are concatenated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for file and descriptor sampling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="K-Means device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--dist-mode",
        choices=["euclidean", "cosine"],
        default="cosine",
        help="K-Means distance mode. Default matches the repo VLAD class: cosine.",
    )
    parser.add_argument(
        "--no-normalize-descriptors",
        action="store_true",
        help="Disable descriptor L2 normalization before K-Means. By default this matches the repo VLAD class.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional JSON summary path. Defaults to the output .pt path with .json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = fit_merged_vocabulary(
        feature_dirs=args.feature_dirs,
        num_clusters=args.num_clusters,
        output=args.output,
        feature_key=args.feature_key,
        recursive=args.recursive,
        file_sample_ratio=args.file_sample_ratio,
        max_files_per_dir=args.max_files_per_dir,
        max_descriptors_per_file=args.max_descriptors_per_file,
        max_total_descriptors=args.max_total_descriptors,
        seed=args.seed,
        device=args.device,
        dist_mode=args.dist_mode,
        normalize_descriptors=not args.no_normalize_descriptors,
        metadata_path=args.metadata_path,
    )
    print("Fitted merged VLAD vocabulary")
    print(f"- Output: {summary['output']}")
    print(f"- Metadata: {summary['metadata']}")
    print(f"- Clusters: {summary['num_clusters']}")
    print(f"- Descriptor dim: {summary['desc_dim']}")
    print(f"- Descriptors used: {summary['num_descriptors_for_fit']}")
    print(f"- Fit device: {summary['fit_device']}")


if __name__ == "__main__":
    main()
