"""
Dense local feature extractor adapters for CVGL backbone ablations.

The rest of the CVGL pipeline only needs an object that can be called as:

    extractor(images) -> Tensor[B, num_patches, desc_dim]

and a patch/feature stride so image crops, mask projection, VLAD, reranking,
and offset estimation can keep using the existing code path.
"""

import math
import os
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from utilities import DinoV2ExtractFeatures


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _as_tensor_stats(values, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=ref.dtype, device=ref.device).view(1, 3, 1, 1)


def renormalize_image_tensor(
    x: torch.Tensor,
    src_mean=_IMAGENET_MEAN,
    src_std=_IMAGENET_STD,
    dst_mean=_IMAGENET_MEAN,
    dst_std=_IMAGENET_STD,
) -> torch.Tensor:
    src_mean_t = _as_tensor_stats(src_mean, x)
    src_std_t = _as_tensor_stats(src_std, x)
    dst_mean_t = _as_tensor_stats(dst_mean, x)
    dst_std_t = _as_tensor_stats(dst_std, x)
    x_01 = x * src_std_t + src_mean_t
    return (x_01 - dst_mean_t) / dst_std_t


def _safe_id(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(":", "-")


class DenseFeatureExtractor(ABC):
    """
    Minimal adapter interface consumed by cvgl_retrieval.py.
    """

    cache_id: str
    patch_stride: int

    @abstractmethod
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Return dense local descriptors shaped [B, N, D].
        """


class DinoV2DenseFeatureExtractor(DenseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        layer: int,
        facet: Literal["query", "key", "value", "token"],
        device: str = "cpu",
        norm_descs: bool = True,
        patch_stride_override: Optional[int] = None,
    ) -> None:
        self.extractor = DinoV2ExtractFeatures(
            model_name,
            layer,
            facet,
            device=device,
            norm_descs=norm_descs,
        )
        self.patch_stride = int(patch_stride_override or 14)
        self.cache_id = f"dinov2-{_safe_id(model_name)}-{facet}-L{layer}-S{self.patch_stride}"

    @property
    def use_cls(self) -> bool:
        return bool(self.extractor.use_cls)

    @use_cls.setter
    def use_cls(self, value: bool) -> None:
        self.extractor.use_cls = bool(value)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.extractor(img)


class TorchHubViTDenseFeatureExtractor(DenseFeatureExtractor):
    """
    DINO v1-style ViT extractor from facebookresearch/dino.
    """

    def __init__(
        self,
        repo: str,
        model_name: str,
        layer: int,
        facet: Literal["query", "key", "value", "token"],
        device: str = "cpu",
        norm_descs: bool = True,
        patch_stride_override: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.model = torch.hub.load(repo, model_name).eval().to(device)
        self.device = torch.device(device)
        self.layer = int(layer)
        self.facet = facet
        self.norm_descs = bool(norm_descs)
        self.use_cls = False
        self.patch_stride = int(patch_stride_override or self._infer_patch_stride(model_name))
        if facet == "token":
            self.hook_handle = self.model.blocks[self.layer].register_forward_hook(self._hook)
        else:
            self.hook_handle = self.model.blocks[self.layer].attn.qkv.register_forward_hook(self._hook)
        self._hook_out = None
        self.cache_id = f"dinohub-{_safe_id(model_name)}-{facet}-L{layer}-S{self.patch_stride}"

    @staticmethod
    def _infer_patch_stride(model_name: str) -> int:
        if model_name.endswith("8"):
            return 8
        if model_name.endswith("16"):
            return 16
        return 16

    def _hook(self, module, inputs, output) -> None:
        self._hook_out = output

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _ = self.model(img.to(self.device))
            if self._hook_out is None:
                raise RuntimeError("No ViT hook output was captured")
            hooked = self._hook_out
            if (
                self.facet == "token"
                and self.layer == len(self.model.blocks) - 1
                and hasattr(self.model, "norm")
            ):
                hooked = self.model.norm(hooked)
            res = hooked if self.use_cls else hooked[:, 1:, :]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
            if self.norm_descs:
                res = F.normalize(res, dim=-1)
            self._hook_out = None
            return res

    def __del__(self):
        if hasattr(self, "hook_handle"):
            self.hook_handle.remove()


class ResNet50DenseFeatureExtractor(DenseFeatureExtractor):
    def __init__(
        self,
        layer: Literal["layer3", "layer4"] = "layer4",
        pretrained: bool = True,
        device: str = "cpu",
        norm_descs: bool = True,
        patch_stride_override: Optional[int] = None,
    ) -> None:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        modules = [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        ]
        stride = 16
        if layer == "layer4":
            modules.append(model.layer4)
            stride = 32
        self.model = nn.Sequential(*modules).eval().to(device)
        self.device = torch.device(device)
        self.layer = layer
        self.norm_descs = bool(norm_descs)
        self.patch_stride = int(patch_stride_override or stride)
        init_tag = "pretrained" if pretrained else "random"
        self.cache_id = f"resnet50-{layer}-{init_tag}-S{self.patch_stride}"

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fmap = self.model(img.to(self.device))
            tokens = fmap.flatten(2).transpose(1, 2)
            if self.norm_descs:
                tokens = F.normalize(tokens, dim=-1)
            return tokens


class OpenAIClipViTDenseFeatureExtractor(DenseFeatureExtractor):
    """
    Local patch-token extractor for OpenAI CLIP ViT models.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        layer: int = -1,
        facet: Literal["query", "key", "value", "token"] = "token",
        device: str = "cpu",
        download_root: Optional[str] = None,
        norm_descs: bool = True,
        use_projection: bool = True,
        patch_stride_override: Optional[int] = None,
    ) -> None:
        import clip

        model, _ = clip.load(model_name, device=device, jit=False, download_root=download_root)
        if not hasattr(model.visual, "conv1") or not hasattr(model.visual, "transformer"):
            raise ValueError("Only CLIP ViT visual backbones expose patch tokens in this adapter")
        self.model = model.eval()
        self.visual = model.visual.eval()
        self.device = torch.device(device)
        self.layer = int(layer)
        self.facet = facet
        self.norm_descs = bool(norm_descs)
        self.use_projection = bool(use_projection)
        self.use_cls = False
        num_blocks = len(self.visual.transformer.resblocks)
        if self.layer < -1 or self.layer >= num_blocks:
            raise ValueError(
                f"Invalid CLIP layer {self.layer}; expected -1 or [0, {num_blocks - 1}]"
            )
        if self.facet != "token":
            print(
                "WARN: CLIP adapter only supports patch token outputs; "
                f"ignoring desc_facet={self.facet!r}."
            )
        stride = int(self.visual.conv1.kernel_size[0])
        self.patch_stride = int(patch_stride_override or stride)
        proj_tag = "proj" if self.use_projection else "width"
        layer_tag = "final" if self.layer == -1 else f"L{self.layer}"
        self.cache_id = f"clip-{_safe_id(model_name)}-{layer_tag}-token-{proj_tag}-S{self.patch_stride}"

    def _positional_embedding(self, grid_h: int, grid_w: int, dtype, device) -> torch.Tensor:
        pos = self.visual.positional_embedding.to(device=device, dtype=dtype)
        cls_pos = pos[:1]
        patch_pos = pos[1:]
        old_grid = int(math.sqrt(patch_pos.shape[0]))
        if old_grid * old_grid != patch_pos.shape[0]:
            raise ValueError(f"Cannot infer square CLIP positional grid from {patch_pos.shape[0]} tokens")
        if old_grid == grid_h and old_grid == grid_w:
            return pos
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
        return torch.cat([cls_pos, patch_pos], dim=0)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = renormalize_image_tensor(img, dst_mean=_CLIP_MEAN, dst_std=_CLIP_STD)
            x = x.to(self.device, dtype=self.visual.conv1.weight.dtype)
            x = self.visual.conv1(x)
            grid_h, grid_w = x.shape[-2:]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = self.visual.class_embedding.to(x.dtype).to(x.device)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + self._positional_embedding(grid_h, grid_w, x.dtype, x.device).unsqueeze(0)
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            for block_idx, block in enumerate(self.visual.transformer.resblocks):
                x = block(x)
                if self.layer == block_idx:
                    break
            x = x.permute(1, 0, 2)
            tokens = self.visual.ln_post(x if self.use_cls else x[:, 1:, :])
            if self.use_projection and getattr(self.visual, "proj", None) is not None:
                tokens = tokens @ self.visual.proj
            if self.norm_descs:
                tokens = F.normalize(tokens.float(), dim=-1)
            return tokens


class HuggingFaceDinoV3DenseFeatureExtractor(DenseFeatureExtractor):
    """
    Hugging Face DINOv3 adapter. Pass an installed/cached DINOv3 model name
    through --model-name.
    """

    def __init__(
        self,
        model_name: str,
        layer: int = -1,
        device: str = "cpu",
        norm_descs: bool = True,
        patch_stride_override: Optional[int] = None,
    ) -> None:
        from transformers import AutoModel

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)
        self.device = torch.device(device)
        self.layer = int(layer)
        self.norm_descs = bool(norm_descs)
        config = self.model.config
        patch_size = int(getattr(config, "patch_size", 16))
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.patch_stride = int(patch_stride_override or patch_size)
        self.cache_id = f"dinov3hf-{_safe_id(model_name)}-L{layer}-S{self.patch_stride}"

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "pixel_values": img.to(self.device),
            "output_hidden_states": self.layer != -1,
        }
        try:
            outputs = self.model(**kwargs, interpolate_pos_encoding=True)
        except TypeError:
            outputs = self.model(**kwargs)
        if self.layer == -1:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[self.layer]
        start = 1 + self.num_register_tokens
        tokens = hidden[:, start:, :]
        if self.norm_descs:
            tokens = F.normalize(tokens, dim=-1)
        return tokens


class TimmFeatureMapDenseFeatureExtractor(DenseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        out_index: int = -1,
        pretrained: bool = True,
        device: str = "cpu",
        norm_descs: bool = True,
        patch_stride_override: Optional[int] = None,
    ) -> None:
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_index,),
        ).eval().to(device)
        self.device = torch.device(device)
        self.norm_descs = bool(norm_descs)
        reduction = int(self.model.feature_info[-1]["reduction"])
        self.patch_stride = int(patch_stride_override or reduction)
        init_tag = "pretrained" if pretrained else "random"
        self.cache_id = f"timm-{_safe_id(model_name)}-out{out_index}-{init_tag}-S{self.patch_stride}"

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fmap = self.model(img.to(self.device))[0]
            tokens = fmap.flatten(2).transpose(1, 2)
            if self.norm_descs:
                tokens = F.normalize(tokens, dim=-1)
            return tokens


def build_dense_feature_extractor(
    backbone: Literal["dinov2", "dino", "dinov3_hf", "clip", "resnet50", "timm_feature"],
    model_name: str,
    layer: int,
    facet: Literal["query", "key", "value", "token"],
    device: str = "cpu",
    pretrained: bool = True,
    cnn_layer: Literal["layer3", "layer4"] = "layer4",
    clip_download_root: Optional[str] = None,
    clip_use_projection: bool = True,
    timm_out_index: int = -1,
    patch_stride_override: Optional[int] = None,
    norm_descs: bool = True,
) -> DenseFeatureExtractor:
    if backbone == "dinov2":
        return DinoV2DenseFeatureExtractor(
            model_name=model_name,
            layer=layer,
            facet=facet,
            device=device,
            norm_descs=norm_descs,
            patch_stride_override=patch_stride_override,
        )
    if backbone == "dino":
        return TorchHubViTDenseFeatureExtractor(
            repo="facebookresearch/dino:main",
            model_name=model_name,
            layer=layer,
            facet=facet,
            device=device,
            norm_descs=norm_descs,
            patch_stride_override=patch_stride_override,
        )
    if backbone == "dinov3_hf":
        return HuggingFaceDinoV3DenseFeatureExtractor(
            model_name=model_name,
            layer=layer,
            device=device,
            norm_descs=norm_descs,
            patch_stride_override=patch_stride_override,
        )
    if backbone == "clip":
        return OpenAIClipViTDenseFeatureExtractor(
            model_name=model_name,
            layer=layer,
            facet=facet,
            device=device,
            download_root=clip_download_root,
            norm_descs=norm_descs,
            use_projection=clip_use_projection,
            patch_stride_override=patch_stride_override,
        )
    if backbone == "resnet50":
        return ResNet50DenseFeatureExtractor(
            layer=cnn_layer,
            pretrained=pretrained,
            device=device,
            norm_descs=norm_descs,
            patch_stride_override=patch_stride_override,
        )
    if backbone == "timm_feature":
        return TimmFeatureMapDenseFeatureExtractor(
            model_name=model_name,
            out_index=timm_out_index,
            pretrained=pretrained,
            device=device,
            norm_descs=norm_descs,
            patch_stride_override=patch_stride_override,
        )
    raise ValueError(f"Unknown backbone: {backbone}")
