"""Optional Qdrant backend (``pip install visualsearch[qdrant]``).

Two modes:
  * embedded: ``params.url`` unset -> ``QdrantClient(path=...)``. This is the
    pure-Python local mode intended for development; it performs exact search
    and does **not** build an HNSW graph, so its latency is not representative
    of a Qdrant server.
  * server: ``params.url`` (e.g. ``http://qdrant:6333`` from docker-compose).
    Real HNSW with payload-aware filtering.

Filters arrive as an allow-list of catalogue ids (resolved from SQLite), which
maps onto Qdrant's ``HasIdCondition``. Vectors are mirrored in memory (as in the
FAISS backend) so fusion and reranking can read them without a round trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from visualsearch.index.base import SearchResult, VectorIndex, check_vectors

COLLECTION = "images"


class QdrantIndex(VectorIndex):
    kind = "qdrant"

    def __init__(self, dim: int, params: dict[str, Any] | None = None, path: Path | None = None):
        super().__init__(dim, {"hnsw_m": 16, "hnsw_ef_construct": 100, "hnsw_ef": 128,
                               **(params or {})})
        from qdrant_client import QdrantClient, models

        self.models = models
        self.path = Path(path) if path else None
        url = self.params.get("url")
        if url:
            self.client = QdrantClient(url=url, timeout=60)
        elif self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.path / "qdrant"))
        else:
            self.client = QdrantClient(location=":memory:")
        self.collection = self.params.get("collection", COLLECTION)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=int(self.params["hnsw_m"]),
                                                  ef_construct=int(self.params["hnsw_ef_construct"])),
            )
        self._mirror: dict[int, np.ndarray] = {}

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        v = check_vectors(vectors, self.dim)
        for start in range(0, len(ids), 512):
            sl = slice(start, start + 512)
            self.client.upsert(self.collection, points=self.models.Batch(
                ids=[int(i) for i in ids[sl]], vectors=v[sl].tolist()), wait=True)
        for i, vec in zip(ids, v):
            self._mirror[int(i)] = vec

    def remove(self, ids: np.ndarray) -> int:
        ids = [int(i) for i in ids if int(i) in self._mirror]
        if ids:
            self.client.delete(self.collection,
                               points_selector=self.models.PointIdsList(points=ids), wait=True)
            for i in ids:
                self._mirror.pop(i)
        return len(ids)

    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None = None
               ) -> list[SearchResult]:
        q = check_vectors(np.atleast_2d(queries), self.dim)
        m = self.models
        flt = None
        if allowed is not None:
            flt = m.Filter(must=[m.HasIdCondition(has_id=[int(i) for i in allowed])])
        reqs = [m.QueryRequest(query=vec.tolist(), limit=k, filter=flt,
                               params=m.SearchParams(hnsw_ef=int(self.params["hnsw_ef"])))
                for vec in q]
        out = []
        for resp in self.client.query_batch_points(self.collection, requests=reqs):
            ids = np.array([p.id for p in resp.points], dtype=np.int64)
            scores = np.array([p.score for p in resp.points], dtype=np.float32)
            out.append(SearchResult(ids, scores, "filtered-index" if flt else "index"))
        return out

    def has(self, image_id: int) -> bool:
        return int(image_id) in self._mirror

    def ids(self) -> np.ndarray:
        return np.array(sorted(self._mirror), dtype=np.int64)

    def vectors_for(self, ids: np.ndarray) -> np.ndarray:
        return np.stack([self._mirror[int(i)] for i in ids]) if len(ids) else \
            np.zeros((0, self.dim), np.float32)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ids = self.ids()
        np.save(directory / "ids.npy", ids)
        np.save(directory / "vectors.npy", self.vectors_for(ids))
        (directory / "qdrant_state.json").write_text(json.dumps({"params": self.params}))

    @classmethod
    def load(cls, directory: Path, dim: int, params: dict[str, Any]) -> QdrantIndex:
        directory = Path(directory)
        state = json.loads((directory / "qdrant_state.json").read_text())
        obj = cls(dim, {**state["params"], **params}, path=directory)
        ids, vecs = np.load(directory / "ids.npy"), np.load(directory / "vectors.npy")
        obj._mirror = {int(i): v for i, v in zip(ids, vecs)}
        return obj

    def close(self) -> None:
        self.client.close()
