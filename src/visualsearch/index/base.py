"""Vector index interface.

All indexes store L2-normalised vectors and score by inner product, which then
equals cosine similarity. External ids are catalogue image ids (int64).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    ids: np.ndarray      # int64 [k'] (k' <= k)
    scores: np.ndarray   # float32 [k']
    strategy: str = "index"  # index | filtered-index | exact-subset
    stats: dict[str, Any] = field(default_factory=dict)


class VectorIndex(ABC):
    kind: str = "abstract"

    def __init__(self, dim: int, params: dict[str, Any] | None = None):
        self.dim = dim
        self.params = dict(params or {})

    @abstractmethod
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def remove(self, ids: np.ndarray) -> int: ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None = None
               ) -> list[SearchResult]: ...

    @abstractmethod
    def ids(self) -> np.ndarray:
        """Live (non-deleted) external ids."""

    @abstractmethod
    def vectors_for(self, ids: np.ndarray) -> np.ndarray:
        """Stored vectors for live ids (used by fusion and reranking)."""

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path, dim: int, params: dict[str, Any]) -> VectorIndex: ...

    def has(self, image_id: int) -> bool:
        return int(image_id) in set(self.ids().tolist())

    def __len__(self) -> int:
        return len(self.ids())

    def stats(self) -> dict[str, Any]:
        return {"kind": self.kind, "dim": self.dim, "params": self.params, "ntotal": len(self)}

    @property
    def needs_rebuild(self) -> bool:
        return False


def check_vectors(vectors: np.ndarray, dim: int) -> np.ndarray:
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != dim:
        raise ValueError(f"expected vectors of shape [n, {dim}], got {v.shape}")
    norms = np.linalg.norm(v, axis=1)
    if len(v) and not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("vectors must be L2-normalised for inner-product (cosine) search")
    return v
