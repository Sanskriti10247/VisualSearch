"""BLIP image-text matching (ITM) cross-encoder for reranking text queries.

Unlike CLIP (two independent towers, one vector each), BLIP's ITM head runs the
text encoder with cross-attention over all image patch tokens, so it can check
fine-grained correspondences (attributes, counts, relations) that a single
pooled vector blurs. The cost is one joint forward pass per (query, image)
pair, which is why it only reranks a short first-stage candidate list.

We use ``Salesforce/blip-itm-base-coco`` (fine-tuned on COCO, not Flickr30k),
so Flickr30k evaluation remains zero-shot with respect to the target dataset.
Image features are cached per image id, so reranking many queries over a fixed
gallery costs one vision forward per image.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from visualsearch.data.preprocess import load_rgb
from visualsearch.embeddings.base import resolve_device

BLIP_ITM = "Salesforce/blip-itm-base-coco"
BLIP_ITM_REVISION = "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c"


class BlipITMReranker:
    def __init__(self, hf_id: str = BLIP_ITM, revision: str = BLIP_ITM_REVISION,
                 device: str = "auto", cache_size: int = 256, batch_size: int = 32):
        self.hf_id, self.revision = hf_id, revision
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._feat_cache: OrderedDict[str, object] = OrderedDict()
        self._cache_size = cache_size
        self._loaded = False

    def load(self) -> BlipITMReranker:
        with self._lock:
            if not self._loaded:
                import torch
                from transformers import BlipForImageTextRetrieval, BlipProcessor

                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = BlipForImageTextRetrieval.from_pretrained(
                    self.hf_id, revision=self.revision, dtype=dtype, use_safetensors=False).to(self.device).eval()
                self.processor = BlipProcessor.from_pretrained(self.hf_id, revision=self.revision)
                self.dtype = dtype
                self._loaded = True
        return self

    @staticmethod
    def load_image(path: str | Path) -> Image.Image:
        return load_rgb(path)

    def _image_embeds(self, images: list[Image.Image], keys: list[str] | None):
        import torch

        out = [None] * len(images)
        todo = []
        for j, im in enumerate(images):
            key = keys[j] if keys else None
            if key is not None and key in self._feat_cache:
                self._feat_cache.move_to_end(key)
                out[j] = self._feat_cache[key]
            else:
                todo.append(j)
        for s in range(0, len(todo), self.batch_size):
            idx = todo[s:s + self.batch_size]
            px = self.processor(images=[images[j] for j in idx], return_tensors="pt")[
                "pixel_values"].to(self.device, self.dtype)
            with torch.inference_mode():
                feats = self.model.vision_model(pixel_values=px).last_hidden_state
            for j, f in zip(idx, feats):
                out[j] = f
                if keys:
                    self._feat_cache[keys[j]] = f
                    if len(self._feat_cache) > self._cache_size:
                        self._feat_cache.popitem(last=False)
        return out

    def score(self, text: str, images: list[Image.Image], keys: list[str] | None = None
              ) -> np.ndarray:
        """P(match) for one text against several images."""
        return self.score_pairs([text] * len(images), images, keys)

    def score_texts_for_image(self, texts: list[str], image: Image.Image) -> np.ndarray:
        """P(match) of many texts against one image: the vision encoder runs once and
        its features are broadcast (no copies), which is what batch evaluation needs."""
        import torch

        self.load()
        with self._lock:
            feat = self._image_embeds([image], None)[0]
            out = []
            for s in range(0, len(texts), self.batch_size):
                tb = texts[s:s + self.batch_size]
                out.append(self._itm(tb, feat.unsqueeze(0).expand(len(tb), -1, -1)))
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(out) if out else np.zeros(0)

    def _itm(self, texts: list[str], feats) -> np.ndarray:
        import torch

        tok = self.processor.tokenizer(texts, padding=True, truncation=True, max_length=35,
                                       return_tensors="pt").to(self.device)
        att = torch.ones(feats.shape[:-1], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            q = self.model.text_encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                                        encoder_hidden_states=feats,
                                        encoder_attention_mask=att).last_hidden_state
            logits = self.model.itm_head(q[:, 0, :]).float()
        return torch.softmax(logits, -1)[:, 1].cpu().numpy()

    def score_pairs(self, texts: list[str], images: list[Image.Image],
                    keys: list[str] | None = None) -> np.ndarray:
        import torch

        self.load()
        with self._lock:
            feats = self._image_embeds(images, keys)
            scores = [self._itm(texts[s:s + self.batch_size],
                                torch.stack(feats[s:s + self.batch_size]))
                      for s in range(0, len(texts), self.batch_size)]
        return np.concatenate(scores) if scores else np.zeros(0)
