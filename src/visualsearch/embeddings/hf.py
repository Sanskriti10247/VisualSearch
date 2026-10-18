"""Hugging Face Transformers embedders: CLIP (text+image) and DINOv2 (image)."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from PIL import Image

from visualsearch.embeddings.base import IMAGE, TEXT, Embedder, register_family

_PREPROC_KEYS = ("do_resize", "size", "resample", "do_center_crop", "crop_size", "do_rescale",
                 "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb")


class _HFEmbedder(Embedder):
    @cached_property
    def _config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(self.spec.hf_id, revision=self.spec.revision)

    @cached_property
    def _image_processor(self):
        from transformers import AutoImageProcessor

        # The slow (PIL) processor is used on purpose: it is the reference
        # implementation and does not require torchvision.
        return AutoImageProcessor.from_pretrained(self.spec.hf_id, revision=self.spec.revision,
                                                  backend="pil")

    def preprocessing_config(self) -> dict[str, Any]:
        d = self._image_processor.to_dict()
        cfg = {k: d[k] for k in _PREPROC_KEYS if k in d}
        cfg["crop_mode"] = self.spec.crop_mode
        cfg["processor"] = type(self._image_processor).__name__
        return cfg

    def _pixels(self, images: list[Image.Image]):
        return self._image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)


@register_family("clip")
class CLIPEmbedder(_HFEmbedder):
    """CLIP: image and text towers projected into one shared, aligned space."""

    modalities = frozenset({IMAGE, TEXT})

    @property
    def dim(self) -> int:
        return int(self._config.projection_dim)

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer, CLIPModel

        self.model = CLIPModel.from_pretrained(self.spec.hf_id, revision=self.spec.revision,
                                               dtype=torch.float32).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id, revision=self.spec.revision)

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            pooled = self.model.vision_model(pixel_values=self._pixels(images)).pooler_output
            # Explicit projection: works across transformers v4/v5 output types.
            return self.model.visual_projection(pooled).float().cpu().numpy()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=77,
                             return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pooled = self.model.text_model(**tok).pooler_output
            return self.model.text_projection(pooled).float().cpu().numpy()


@register_family("dinov2")
class DINOv2Embedder(_HFEmbedder):
    """DINOv2: self-supervised ViT. Image-only, no language alignment.

    pooling:
      cls       layer-normed CLS token (standard retrieval descriptor)
      mean      mean of layer-normed patch tokens
      cls+mean  concatenation of both (the DINOv2 linear-probe descriptor)
    """

    modalities = frozenset({IMAGE})

    @property
    def dim(self) -> int:
        h = int(self._config.hidden_size)
        return 2 * h if self.spec.pooling == "cls+mean" else h

    def preprocessing_config(self) -> dict[str, Any]:
        return {**super().preprocessing_config(), "pooling": self.spec.pooling}

    def _load(self) -> None:
        import torch
        from transformers import AutoModel

        if self.spec.pooling not in ("cls", "mean", "cls+mean"):
            raise ValueError(f"unknown pooling '{self.spec.pooling}'")
        self.model = AutoModel.from_pretrained(self.spec.hf_id, revision=self.spec.revision,
                                               dtype=torch.float32).to(self.device).eval()

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            hidden = self.model(pixel_values=self._pixels(images)).last_hidden_state  # layer-normed
            cls, patches = hidden[:, 0], hidden[:, 1:]
            if self.spec.pooling == "cls":
                v = cls
            elif self.spec.pooling == "mean":
                v = patches.mean(1)
            else:
                v = torch.cat([cls, patches.mean(1)], dim=-1)
            return v.float().cpu().numpy()
