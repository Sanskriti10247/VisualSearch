"""FAISS-backed vector indexes.

Kinds (all inner product over L2-normalised vectors == cosine similarity):

  flat      IndexFlatIP. Exact brute force. The reference for recall.
  hnsw      IndexHNSWFlat. Graph ANN. params: M, ef_construction, ef_search.
  ivf_flat  IndexIVFFlat. Inverted lists over k-means cells. params: nlist, nprobe.
  ivf_pq    IndexIVFPQ. IVF + product quantisation (compressed codes).
            params: nlist, nprobe, m (sub-quantisers), nbits, refine (re-score
            the top ``k * refine`` candidates exactly with the stored float
            vectors; 0 disables).

FAISS stores vectors at sequential positions 0..n-1; this class keeps the
position -> catalogue id mapping itself (instead of ``IndexIDMap``) so that
every kind supports ids, filtering and deletion the same way. Deletions are
tombstones; :meth:`compact` (or a rebuild) drops them.

Filtered search uses a small query planner: if the filter allows at most
``brute_force_threshold`` live vectors, those vectors are scanned exactly
(always correct, and cheap for small subsets); otherwise the index is searched
with a FAISS ``IDSelector`` so the filter is applied during traversal rather
than after it (post-filtering can return far fewer than k results).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from visualsearch.index.base import SearchResult, VectorIndex, check_vectors

DEFAULTS: dict[str, dict[str, Any]] = {
    "flat": {},
    "hnsw": {"M": 32, "ef_construction": 200, "ef_search": 128},
    "ivf_flat": {"nlist": "auto", "nprobe": 16},
    "ivf_pq": {"nlist": "auto", "nprobe": 16, "m": "auto", "nbits": "auto", "refine": 4},
}


def auto_nlist(n: int) -> int:
    # Common heuristic ~4*sqrt(n), bounded so each centroid gets >= 39 training points.
    return int(max(1, min(4 * math.sqrt(max(n, 1)), n // 39 if n >= 39 else 1)))


class FaissIndex(VectorIndex):
    def __init__(self, dim: int, kind: str = "flat", params: dict[str, Any] | None = None,
                 brute_force_threshold: int = 5000):
        if kind not in DEFAULTS:
            raise ValueError(f"unknown FAISS index kind '{kind}'")
        super().__init__(dim, {**DEFAULTS[kind], **(params or {})})
        self.kind = kind
        self.brute_force_threshold = brute_force_threshold
        self.index: faiss.Index | None = None
        self._ids = np.zeros(0, dtype=np.int64)       # position -> external id
        self._deleted = np.zeros(0, dtype=bool)
        self._stored = np.zeros((0, dim), dtype=np.float32)  # unused for flat, see _vectors
        self._pos: dict[int, int] = {}                # live external id -> position
        self.trained_on = 0
        if kind in ("flat", "hnsw"):
            self.index = self._new_index(0)

    # ------------------------------------------------------------ construction
    def _new_index(self, n_train: int) -> faiss.Index:
        p = self.params
        if self.kind == "flat":
            return faiss.IndexFlatIP(self.dim)
        if self.kind == "hnsw":
            idx = faiss.IndexHNSWFlat(self.dim, int(p["M"]), faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = int(p["ef_construction"])
            idx.hnsw.efSearch = int(p["ef_search"])
            return idx
        nlist = auto_nlist(n_train) if p["nlist"] == "auto" else int(p["nlist"])
        self.resolved = {"nlist": nlist}
        quantizer = faiss.IndexFlatIP(self.dim)
        if self.kind == "ivf_flat":
            return faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        m = self._auto_m() if p["m"] == "auto" else int(p["m"])
        # 8-bit codebooks need ~39*256 training points; use fewer bits on small data.
        nbits = (8 if n_train >= 39 * 256 else max(4, int(math.log2(max(n_train, 16) / 39))))\
            if p["nbits"] == "auto" else int(p["nbits"])
        self.resolved.update({"m": m, "nbits": nbits})
        return faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    def _auto_m(self) -> int:
        # ~16 dims per sub-quantiser, and m must divide dim.
        target = max(1, self.dim // 16)
        return max(d for d in range(1, target + 1) if self.dim % d == 0)

    @property
    def is_trained(self) -> bool:
        return self.index is not None and self.index.is_trained

    def train(self, vectors: np.ndarray) -> None:
        if self.kind in ("flat", "hnsw"):
            return
        v = check_vectors(vectors, self.dim)
        self.index = self._new_index(len(v))
        self.index.train(v)
        self.trained_on = len(v)

    # ---------------------------------------------------------------- mutation
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        v = check_vectors(vectors, self.dim)
        if len(ids) != len(v):
            raise ValueError("ids/vectors length mismatch")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate ids in one add() call")
        if not len(ids):
            return
        # Re-adding an id replaces it: tombstone the old position.
        self.remove(np.array([i for i in ids if int(i) in self._pos], dtype=np.int64))
        if not self.is_trained:
            self.train(v)
        start = len(self._ids)
        self.index.add(v)
        self._ids = np.concatenate([self._ids, ids])
        self._deleted = np.concatenate([self._deleted, np.zeros(len(ids), dtype=bool)])
        if self.kind != "flat":
            self._stored = np.concatenate([self._stored, v])
        for j, i in enumerate(ids):
            self._pos[int(i)] = start + j

    def remove(self, ids: np.ndarray) -> int:
        n = 0
        for i in np.asarray(ids, dtype=np.int64):
            pos = self._pos.pop(int(i), None)
            if pos is not None:
                self._deleted[pos] = True
                n += 1
        return n

    def compact(self) -> FaissIndex:
        """Rebuild without tombstones (and retrain IVF on the current data)."""
        live = ~self._deleted
        fresh = FaissIndex(self.dim, self.kind, self.params, self.brute_force_threshold)
        fresh.add(self._ids[live], self._vectors[live])
        return fresh

    @property
    def needs_rebuild(self) -> bool:
        n_dead = int(self._deleted.sum())
        if n_dead > 0.2 * max(len(self._ids), 1):
            return True
        # IVF centroids were fit on the first batch; refit once data has grown a lot.
        return self.kind.startswith("ivf") and len(self) > 4 * max(self.trained_on, 1)

    @property
    def _vectors(self) -> np.ndarray:
        """Float vectors by position. A flat index stores exactly these, so it is
        viewed in place instead of keeping a second copy in memory; the view is
        rebuilt on every access because adding to the index can reallocate it."""
        if self.kind != "flat":
            return self._stored
        n = self.index.ntotal
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return faiss.rev_swig_ptr(self.index.get_xb(), n * self.dim).reshape(n, self.dim)

    # ------------------------------------------------------------------ search
    def ids(self) -> np.ndarray:
        return self._ids[~self._deleted]

    def vectors_for(self, ids: np.ndarray) -> np.ndarray:
        return self._vectors[[self._pos[int(i)] for i in ids]]

    def has(self, image_id: int) -> bool:
        return int(image_id) in self._pos

    def _search_params(self, sel: faiss.IDSelector | None, k: int):
        p = self.params
        if self.kind == "hnsw":
            return faiss.SearchParametersHNSW(sel=sel, efSearch=max(int(p["ef_search"]), k))
        if self.kind.startswith("ivf"):
            return faiss.SearchParametersIVF(sel=sel, nprobe=int(p["nprobe"]))
        return faiss.SearchParameters(sel=sel) if sel is not None else None

    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None = None
               ) -> list[SearchResult]:
        q = check_vectors(np.atleast_2d(queries), self.dim)
        if len(self._ids) == 0 or k <= 0:
            return [SearchResult(np.zeros(0, np.int64), np.zeros(0, np.float32)) for _ in q]

        has_deleted = bool(self._deleted.any())
        if allowed is not None:
            positions = np.array([self._pos[int(i)] for i in allowed if int(i) in self._pos],
                                 dtype=np.int64)
            if len(positions) <= self.brute_force_threshold:
                return self._exact_subset(q, k, positions)
            strategy = "filtered-index"
        elif has_deleted:
            positions = np.nonzero(~self._deleted)[0].astype(np.int64)
            strategy = "index"
        else:
            positions, strategy = None, "index"

        sel = faiss.IDSelectorBatch(positions) if positions is not None else None
        refine = int(self.params.get("refine", 0) or 0) if self.kind == "ivf_pq" else 0
        k_fetch = min(k * refine, len(self._ids)) if refine else k
        D, I = self.index.search(q, k_fetch, params=self._search_params(sel, k_fetch))
        out = []
        for d, i in zip(D, I):
            keep = i >= 0
            pos, scores = i[keep], d[keep]
            if refine:
                scores = self._vectors[pos] @ q[len(out)]
                order = np.argsort(-scores)[:k]
                pos, scores = pos[order], scores[order]
            out.append(SearchResult(self._ids[pos], scores.astype(np.float32), strategy))
        return out

    def _exact_subset(self, q: np.ndarray, k: int, positions: np.ndarray) -> list[SearchResult]:
        if len(positions) == 0:
            return [SearchResult(np.zeros(0, np.int64), np.zeros(0, np.float32), "exact-subset")
                    for _ in q]
        sims = q @ self._vectors[positions].T
        kk = min(k, len(positions))
        out = []
        for s in sims:
            top = np.argpartition(-s, kk - 1)[:kk]
            top = top[np.argsort(-s[top])]
            out.append(SearchResult(self._ids[positions[top]], s[top].astype(np.float32),
                                    "exact-subset"))
        return out

    # ------------------------------------------------------------- persistence
    def index_bytes(self) -> int:
        if self.index is None:
            return 0
        if self.kind == "flat":
            # Same number without serialising (a transient copy of every vector).
            header = faiss.serialize_index(faiss.IndexFlatIP(self.dim)).nbytes
            return int(header + self.index.ntotal * self.index.code_size)
        return int(faiss.serialize_index(self.index).nbytes)

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "resolved": getattr(self, "resolved", {}),
                "tombstones": int(self._deleted.sum()), "trained_on": self.trained_on,
                "index_bytes": self.index_bytes(), "vector_bytes": int(self._vectors.nbytes)}

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "ids.npy", self._ids)
        np.save(directory / "deleted.npy", self._deleted)
        np.save(directory / "vectors.npy", self._vectors)
        (directory / "faiss_state.json").write_text(json.dumps(
            {"kind": self.kind, "params": self.params, "trained_on": self.trained_on,
             "resolved": getattr(self, "resolved", {})}))

    @classmethod
    def load(cls, directory: Path, dim: int, params: dict[str, Any],
             brute_force_threshold: int = 5000) -> FaissIndex:
        directory = Path(directory)
        state = json.loads((directory / "faiss_state.json").read_text())
        obj = cls(dim, state["kind"], state["params"], brute_force_threshold)
        if (directory / "index.faiss").exists():
            obj.index = faiss.read_index(str(directory / "index.faiss"))
        obj._ids = np.load(directory / "ids.npy")
        obj._deleted = np.load(directory / "deleted.npy")
        if obj.kind != "flat":  # flat: the vectors are read from index.faiss
            obj._stored = np.load(directory / "vectors.npy")
        obj.trained_on = state["trained_on"]
        obj.resolved = state.get("resolved", {})
        obj._pos = {int(i): p for p, i in enumerate(obj._ids) if not obj._deleted[p]}
        if obj.index is not None and obj.index.ntotal != len(obj._ids):
            raise RuntimeError(f"index at {directory} is inconsistent: "
                               f"{obj.index.ntotal} vectors vs {len(obj._ids)} ids")
        return obj
