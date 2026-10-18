"""Shared state for evaluation runs: catalogue, model variants, cached vectors."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tqdm import tqdm

from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog, ImageRecord
from visualsearch.data.preprocess import load_rgb
from visualsearch.embeddings import IMAGE, TEXT, Embedder, create_embedder, load_model_specs
from visualsearch.embeddings.store import EmbeddingStore, text_key

log = logging.getLogger(__name__)


class EvalContext:
    def __init__(self, settings: Settings, catalog: Catalog | None = None):
        self.settings = settings
        self.catalog = catalog or Catalog(settings.catalog_path)
        self.specs = load_model_specs(settings.models_path)
        self._embedders: dict[str, Embedder] = {}
        self._stores: dict[tuple[str, str], EmbeddingStore] = {}

    # ------------------------------------------------------------- models
    def embedder(self, model: str, **overrides: Any) -> Embedder:
        """Model by key, optionally with spec overrides (e.g. pooling='mean')."""
        name = model + ("" if not overrides else
                        "[" + ",".join(f"{k}={v}" for k, v in sorted(overrides.items())) + "]")
        if name not in self._embedders:
            spec = self.specs[model].with_overrides(**overrides) if overrides else self.specs[model]
            self._embedders[name] = create_embedder(spec, device=self.settings.device,
                                                    batch_size=self.settings.batch_size,
                                                    torch_threads=self.settings.torch_threads)
        return self._embedders[name]

    def store(self, emb: Embedder, modality: str) -> EmbeddingStore:
        key = (emb.fingerprint, modality)
        if key not in self._stores:
            self._stores[key] = EmbeddingStore(self.settings.embeddings_dir, emb, modality)
        return self._stores[key]

    # ------------------------------------------------------------ vectors
    def image_vectors(self, emb: Embedder, records: list[ImageRecord], desc: str = "") -> np.ndarray:
        store = self.store(emb, IMAGE)
        by_sha = {r.sha256: r for r in records}
        missing = store.missing(list(by_sha))
        bar = tqdm(total=len(missing), desc=desc or f"embed {emb.key}", disable=not missing)

        def compute(shas: list[str]) -> np.ndarray:
            with ThreadPoolExecutor(6) as pool:
                imgs = list(pool.map(lambda s: load_rgb(self.settings.images_dir / by_sha[s].path),
                                     shas))
            return emb.embed_images(imgs)

        out = store.get_or_compute([r.sha256 for r in records], compute,
                                   chunk=max(emb.batch_size * 4, 64), progress=bar.update)
        bar.close()
        return out

    def text_vectors(self, emb: Embedder, texts: list[str]) -> np.ndarray:
        if not emb.supports(TEXT):
            raise ValueError(f"{emb.key} has no text encoder")
        store = self.store(emb, TEXT)
        keys = [text_key(t) for t in texts]
        lookup = dict(zip(keys, texts))
        missing = store.missing(keys)
        bar = tqdm(total=len(missing), desc=f"embed text {emb.key}", disable=not missing)
        out = store.get_or_compute(keys, lambda ks: emb.embed_texts([lookup[k] for k in ks]),
                                   chunk=2048, progress=bar.update)
        bar.close()
        return out

    # --------------------------------------------------------------- data
    def records(self, collection: str, splits: list[str]) -> list[ImageRecord]:
        out: list[ImageRecord] = []
        for s in splits:
            out.extend(self.catalog.records(collection=collection, split=s))
        return out

    def flickr_queries(self, split: str) -> tuple[list[str], np.ndarray]:
        """(caption texts, target image ids) for all captions of a split."""
        rows = self.catalog.captions(collection="flickr30k", split=split)
        return [t for _, _, t in rows], np.array([i for _, i, _ in rows], dtype=np.int64)
