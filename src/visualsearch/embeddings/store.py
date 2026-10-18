"""Content-addressed embedding cache.

Layout: ``<embeddings_dir>/<model_key>/<fingerprint>/<modality>/``
  vectors.npy   float32 [n, dim]
  keys.json     row -> key (image SHA-256, or SHA-256 of the text)
  meta.json     model description, fingerprint, dim, timestamps

Because keys are content hashes and the directory is keyed by fingerprint,
re-ingesting the same image, renaming files or rebuilding an index never
recomputes an embedding, while any change to the model or its preprocessing
lands in a fresh directory automatically.

Appends rewrite the arrays atomically (write temp file + rename). That is
O(n) per append, which is fine at this scale (10^4–10^5 rows) and keeps the
format trivially inspectable; a larger deployment would shard into chunks.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from visualsearch.data.catalog import now_iso
from visualsearch.embeddings.base import Embedder


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingStore:
    def __init__(self, root: Path, embedder: Embedder, modality: str):
        self.embedder = embedder
        self.modality = modality
        self.dir = Path(root) / embedder.key / embedder.fingerprint / modality
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._keys: list[str] = []
        self._row: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        v, k = self.dir / "vectors.npy", self.dir / "keys.json"
        if v.exists() and k.exists():
            self._vectors = np.load(v)
            self._keys = json.loads(k.read_text())
            if len(self._keys) != len(self._vectors):
                raise RuntimeError(f"corrupt embedding store at {self.dir}")
            self._row = {key: i for i, key in enumerate(self._keys)}

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._row

    def get(self, keys: Sequence[str]) -> np.ndarray:
        rows = [self._row[k] for k in keys]
        return self._vectors[rows]

    def missing(self, keys: Sequence[str]) -> list[str]:
        return [k for k in dict.fromkeys(keys) if k not in self._row]

    def put(self, keys: Sequence[str], vectors: np.ndarray, flush: bool = True) -> None:
        if len(keys) != len(vectors):
            raise ValueError("keys/vectors length mismatch")
        with self._lock:
            new = [(k, i) for i, k in enumerate(keys) if k not in self._row]
            if not new:
                return
            add = vectors[[i for _, i in new]].astype(np.float32)
            self._vectors = np.concatenate([self._vectors, add])
            for k, _ in new:
                self._row[k] = len(self._keys)
                self._keys.append(k)
            self._dirty = True
            if flush:
                self._flush()

    def flush(self) -> None:
        with self._lock:
            if getattr(self, "_dirty", False):
                self._flush()

    def _flush(self) -> None:
        self._dirty = False
        tmp_v = self.dir / "vectors.tmp.npy"
        tmp_k = self.dir / "keys.tmp.json"
        np.save(tmp_v, self._vectors)
        tmp_k.write_text(json.dumps(self._keys))
        os.replace(tmp_v, self.dir / "vectors.npy")
        os.replace(tmp_k, self.dir / "keys.json")
        (self.dir / "meta.json").write_text(json.dumps(
            {**self.embedder.describe(), "modality": self.modality, "count": len(self._keys),
             "updated_at": now_iso()}, indent=2, default=str))

    def get_or_compute(self, keys: Sequence[str], compute: Callable[[list[str]], np.ndarray],
                       chunk: int = 256, progress: Callable[[int], None] | None = None,
                       flush_seconds: float = 60.0) -> np.ndarray:
        """Return vectors for ``keys``, computing (in chunks) and caching misses.

        ``compute`` receives a list of missing keys and returns their vectors.
        Results are flushed at most every ``flush_seconds`` (and at the end) so
        an interrupted run resumes cheaply without rewriting the file per chunk.
        """
        todo = self.missing(keys)
        last_flush = time.monotonic()
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            self.put(part, compute(part), flush=False)
            if time.monotonic() - last_flush > flush_seconds:
                self.flush()
                last_flush = time.monotonic()
            if progress:
                progress(len(part))
        self.flush()
        return self.get(keys)
