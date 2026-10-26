"""Lifecycle of serving indexes: build, persist, detect staleness, sync.

One serving index exists per active embedding model, stored at
``<indexes_dir>/<model_key>/<index_kind>/`` with a ``manifest.json`` that
records the embedding fingerprint it was built with. On load, a manifest whose
fingerprint differs from the current model configuration (new weights
revision, different pooling/cropping/preprocessing, ...) marks the index as
stale and it is rebuilt; thanks to the content-addressed embedding cache, a
rebuild with an unchanged fingerprint costs only index construction time.

:meth:`sync` brings an index up to date with the catalogue: it embeds images
the index lacks (cache misses only) and tombstones deleted ones. This is the
incremental ingestion path used by the API's background jobs.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog, now_iso
from visualsearch.data.preprocess import load_rgb
from visualsearch.embeddings import IMAGE, Embedder, create_embedder, load_model_specs
from visualsearch.embeddings.store import EmbeddingStore
from visualsearch.index.base import VectorIndex
from visualsearch.index.faiss_index import FaissIndex

log = logging.getLogger(__name__)


class RWLock:
    """Writer-preferring readers-writer lock (searches read, sync/rebuild write)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            self._waiting_writers += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def make_index(kind: str, dim: int, params: dict[str, Any], settings: Settings,
               directory: Path | None = None) -> VectorIndex:
    if kind == "qdrant":
        from visualsearch.index.qdrant_index import QdrantIndex

        return QdrantIndex(dim, params, path=directory)
    return FaissIndex(dim, kind, params, brute_force_threshold=settings.brute_force_filter_threshold)


def load_index(kind: str, directory: Path, dim: int, params: dict[str, Any],
               settings: Settings) -> VectorIndex:
    if kind == "qdrant":
        from visualsearch.index.qdrant_index import QdrantIndex

        return QdrantIndex.load(directory, dim, params)
    return FaissIndex.load(directory, dim, params,
                           brute_force_threshold=settings.brute_force_filter_threshold,
                           mmap=settings.read_only)


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


class ModelRuntime:
    """Everything bound to one embedding space."""

    def __init__(self, embedder: Embedder, settings: Settings):
        self.embedder = embedder
        self.settings = settings
        self.image_store = EmbeddingStore(settings.embeddings_dir, embedder, IMAGE)
        # Query texts are cached in memory only (bounded): persisting arbitrary user
        # queries would grow without bound and rewrite the store on every query.
        self.query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.query_cache_size = 4096
        self.query_cache_lock = threading.Lock()
        self.index: VectorIndex | None = None
        self.manifest: dict[str, Any] = {}
        self.lock = RWLock()

    @property
    def key(self) -> str:
        return self.embedder.key


class IndexManager:
    def __init__(self, settings: Settings, catalog: Catalog, models: list[str] | None = None,
                 embedders: dict[str, Embedder] | None = None):
        self.settings = settings
        self.catalog = catalog
        self.kind = settings.index_type
        self.params = settings.index_params
        if embedders is None:
            specs = load_model_specs(settings.models_path)
            embedders = {}
            for key in models or settings.active_models:
                if key not in specs:
                    raise KeyError(f"model '{key}' not defined in {settings.models_path}")
                embedders[key] = create_embedder(specs[key], device=settings.device,
                                                 batch_size=settings.batch_size,
                                                 torch_threads=settings.torch_threads,
                                                 runtime=settings.embed_runtime,
                                                 onnx_dir=settings.onnx_path,
                                                 onnx_threads=settings.onnx_threads)
        self.runtimes = {k: ModelRuntime(e, settings) for k, e in embedders.items()}
        self._sync_lock = threading.RLock()  # serialises sync/rebuild per manager

    # --------------------------------------------------------------- paths
    def index_dir(self, model_key: str) -> Path:
        return self.settings.indexes_dir / model_key / self.kind

    def runtime(self, model_key: str) -> ModelRuntime:
        if model_key not in self.runtimes:
            raise KeyError(f"model '{model_key}' is not loaded; available: {sorted(self.runtimes)}")
        return self.runtimes[model_key]

    # ----------------------------------------------------------- lifecycle
    def open(self, build_missing: bool = True,
             progress: Callable[[str, int, int], None] | None = None,
             rebuild_stale: bool = True) -> None:
        """Load every serving index; rebuild stale ones and (optionally) build missing ones.

        With ``rebuild_stale=False`` (read-only serving) a stale index is an error:
        it cannot be rebuilt in place and its scores would be meaningless."""
        for key, rt in self.runtimes.items():
            d = self.index_dir(key)
            manifest_file = d / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                if manifest.get("embedding_fingerprint") != rt.embedder.fingerprint:
                    if not rebuild_stale:
                        raise RuntimeError(
                            f"index for {key} was built with embedding fingerprint "
                            f"{manifest.get('embedding_fingerprint')}, but the model now has "
                            f"{rt.embedder.fingerprint}; rebuild the index")
                    log.warning("index for %s is stale (fingerprint %s != %s); rebuilding", key,
                                manifest.get("embedding_fingerprint"), rt.embedder.fingerprint)
                    self.rebuild(key, progress=progress)
                    continue
                rt.index = load_index(self.kind, d, rt.embedder.dim, self.params, self.settings)
                rt.manifest = manifest
            elif build_missing:
                self.rebuild(key, progress=progress)
            if rt.index is not None and build_missing:
                self.sync(key, progress=progress)

    def status(self) -> dict[str, Any]:
        out = {}
        for key, rt in self.runtimes.items():
            out[key] = {
                "model": rt.embedder.describe(),
                "index": rt.index.stats() if rt.index is not None else None,
                "manifest": rt.manifest,
                "stale": bool(rt.manifest) and
                rt.manifest.get("embedding_fingerprint") != rt.embedder.fingerprint,
                "cached_image_embeddings": len(rt.image_store),
            }
        return out

    # --------------------------------------------------------- embeddings
    def image_vectors(self, key: str, records: list, workers: int = 6,
                      progress: Callable[[int], None] | None = None) -> np.ndarray:
        """Embeddings for catalogue records (cache first; compute misses)."""
        rt = self.runtime(key)
        by_sha = {r.sha256: r for r in records}

        def compute(shas: list[str]) -> np.ndarray:
            with ThreadPoolExecutor(workers) as pool:
                imgs = list(pool.map(
                    lambda s: load_rgb(self.settings.images_dir / by_sha[s].path), shas))
            return rt.embedder.embed_images(imgs)

        return rt.image_store.get_or_compute([r.sha256 for r in records], compute,
                                             chunk=max(rt.embedder.batch_size * 4, 64),
                                             progress=progress)

    def text_vectors(self, key: str, texts: list[str]) -> np.ndarray:
        """Query-text embeddings with an in-memory LRU cache."""
        rt = self.runtime(key)
        with rt.query_cache_lock:
            missing = [t for t in dict.fromkeys(texts) if t not in rt.query_cache]
        if missing:
            vecs = rt.embedder.embed_texts(missing)  # raises UnsupportedModalityError
            with rt.query_cache_lock:
                for t, v in zip(missing, vecs):
                    rt.query_cache[t] = v
                while len(rt.query_cache) > rt.query_cache_size:
                    rt.query_cache.popitem(last=False)
        with rt.query_cache_lock:
            out = []
            for t in texts:
                rt.query_cache.move_to_end(t)
                out.append(rt.query_cache[t])
        return np.stack(out)

    # ------------------------------------------------------- build / sync
    def rebuild(self, key: str, progress: Callable[[str, int, int], None] | None = None
                ) -> dict[str, Any]:
        with self._sync_lock:
            return self._rebuild(key, progress)

    def _rebuild(self, key: str, progress: Callable[[str, int, int], None] | None
                 ) -> dict[str, Any]:
        rt = self.runtime(key)
        records = self.catalog.records()
        t0 = time.perf_counter()
        done = [0]

        def tick(n: int) -> None:
            done[0] += n
            if progress:
                progress(key, done[0], len(records))

        vecs = self.image_vectors(key, records, progress=tick) if records else \
            np.zeros((0, rt.embedder.dim), np.float32)
        t_embed = time.perf_counter() - t0
        d = self.index_dir(key)
        if d.exists():
            shutil.rmtree(d)
        t1 = time.perf_counter()
        index = make_index(self.kind, rt.embedder.dim, self.params, self.settings, directory=d)
        if records:
            index.add(np.array([r.id for r in records], dtype=np.int64), vecs)
        build_s = time.perf_counter() - t1
        with rt.lock.write():
            rt.index = index
            rt.manifest = self._write_manifest(key, build_seconds=build_s, embed_seconds=t_embed)
        return rt.manifest

    def sync(self, key: str, progress: Callable[[str, int, int], None] | None = None
             ) -> dict[str, Any]:
        """Add catalogue images missing from the index; remove deleted ones."""
        with self._sync_lock:
            rt = self.runtime(key)
            if rt.index is None:
                return self.rebuild(key, progress)
            live = set(self.catalog.ids().tolist())
            indexed = set(rt.index.ids().tolist())
            to_add = sorted(live - indexed)
            to_remove = np.array(sorted(indexed - live), dtype=np.int64)
            added = removed = 0
            if to_add:
                recs = self.catalog.get_many(to_add)
                records = [recs[i] for i in to_add]
                done = [0]

                def tick(n: int) -> None:
                    done[0] += n
                    if progress:
                        progress(key, done[0], len(records))

                vecs = self.image_vectors(key, records, progress=tick)
                with rt.lock.write():
                    rt.index.add(np.array(to_add, dtype=np.int64), vecs)
                added = len(to_add)
            if len(to_remove):
                with rt.lock.write():
                    removed = rt.index.remove(to_remove)
            if getattr(rt.index, "needs_rebuild", False):
                log.info("index for %s needs compaction/retraining; rebuilding", key)
                return self.rebuild(key, progress)
            if added or removed:
                rt.manifest = self._write_manifest(key, last_sync={"added": added,
                                                                   "removed": removed})
            return {"model": key, "added": added, "removed": removed, "ntotal": len(rt.index)}

    def _write_manifest(self, key: str, **extra: Any) -> dict[str, Any]:
        rt = self.runtime(key)
        d = self.index_dir(key)
        rt.index.save(d)
        manifest = {
            **{k: v for k, v in rt.manifest.items() if k in ("built_at", "build_seconds",
                                                             "embed_seconds")},
            "model_key": key,
            "embedding_fingerprint": rt.embedder.fingerprint,
            "model": {k: v for k, v in rt.embedder.describe().items() if k != "loaded"},
            "index_kind": self.kind,
            "index_params": self.params,
            "index_stats": rt.index.stats(),
            "ntotal": len(rt.index),
            "faiss_version": faiss.__version__,
            "updated_at": now_iso(),
            **extra,
        }
        if "build_seconds" in extra:
            manifest["built_at"] = now_iso()
        manifest["disk_bytes"] = dir_bytes(d)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return manifest
