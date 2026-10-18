"""Deterministic, dependency-free embedder for tests and CI.

Images: a fixed random projection of a 8x8 RGB thumbnail (similar images give
similar vectors). Texts: a hashed bag-of-words projection. The two are *not*
semantically aligned; only shapes, determinism and plumbing are tested with it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image

from visualsearch.embeddings.base import IMAGE, TEXT, Embedder, register_family


@register_family("fake")
class FakeEmbedder(Embedder):
    modalities = frozenset({IMAGE, TEXT})

    @property
    def dim(self) -> int:
        return int(self.spec.params.get("dim", 32))

    def preprocessing_config(self) -> dict[str, Any]:
        return {"thumb": 8, "crop_mode": self.spec.crop_mode}

    def _load(self) -> None:
        rng = np.random.default_rng(int(self.spec.params.get("seed", 0)))
        self._proj = rng.standard_normal((8 * 8 * 3, self.dim)).astype(np.float32)
        if not self.spec.params.get("text", True):
            self.modalities = frozenset({IMAGE})

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        x = np.stack([np.asarray(im.resize((8, 8)), dtype=np.float32).ravel() / 255.0 - 0.5
                      for im in images])
        return x @ self._proj

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return out
