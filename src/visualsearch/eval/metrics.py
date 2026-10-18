"""Retrieval metrics.

Conventions
-----------
* Ranks are 1-based. A relevant item absent from a (truncated) ranked list has
  rank = inf.
* Recall@K is the fraction of queries with *at least one* relevant item in the
  top K. This is the standard definition for Flickr30k/COCO retrieval (where a
  text query has exactly one relevant image, and an image query has five
  relevant captions).
* MRR uses the rank of the first relevant item.
* AP uses all relevant items: AP = (1/R) * sum_{relevant i} precision@rank(i),
  where R is the total number of relevant items for the query (not just the
  retrieved ones), so truncated lists are penalised. With exactly one relevant
  item, AP == reciprocal rank, so mAP == MRR; we report it only where queries
  have several relevant items (image->text, category-level image->image).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PerQuery:
    """Per-query values; aggregate with :func:`summarize`."""

    first_rank: np.ndarray            # float (inf if never retrieved)
    ap: np.ndarray                    # float
    clusters: np.ndarray | None = None  # e.g. target image id, for clustered bootstrap
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.first_rank)

    def subset(self, mask: np.ndarray) -> PerQuery:
        return PerQuery(self.first_rank[mask], self.ap[mask],
                        None if self.clusters is None else self.clusters[mask],
                        {k: v[mask] for k, v in self.extra.items()})


def ranks_from_scores(scores: np.ndarray, relevant: Sequence[Sequence[int]]) -> list[np.ndarray]:
    """Exact 1-based ranks of each query's relevant columns in a full score matrix.

    rank = 1 + number of items scoring strictly higher (ties resolved
    optimistically, immaterial for continuous scores). O(G log G) per query.
    """
    out = []
    for row, rel in zip(scores, relevant):
        rel = np.asarray(rel, dtype=np.int64)
        asc = np.sort(row)
        higher = len(row) - np.searchsorted(asc, row[rel], side="right")
        out.append(1 + higher)
    return out


def ranks_from_lists(ranked: Sequence[np.ndarray], relevant: Sequence[Sequence[int]]
                     ) -> list[np.ndarray]:
    """1-based ranks of relevant ids within (possibly truncated) ranked id lists."""
    out = []
    for lst, rel in zip(ranked, relevant):
        pos = {int(x): i + 1 for i, x in enumerate(lst)}
        out.append(np.array([pos.get(int(r), np.inf) for r in rel], dtype=np.float64))
    return out


def per_query_from_ranks(rel_ranks: Sequence[np.ndarray], clusters: np.ndarray | None = None
                         ) -> PerQuery:
    first, ap = [], []
    for r in rel_ranks:
        r = np.sort(np.asarray(r, dtype=np.float64))
        first.append(r[0] if len(r) else np.inf)
        found = r[np.isfinite(r)]
        # precision at the rank of the j-th relevant item = j / rank
        ap.append(float(np.sum(np.arange(1, len(found) + 1) / found) / len(r)) if len(r) else 0.0)
    return PerQuery(np.array(first), np.array(ap), clusters)


def recall_at(first_rank: np.ndarray, k: int) -> np.ndarray:
    return (first_rank <= k).astype(np.float64)


def reciprocal_rank(first_rank: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(first_rank), 1.0 / first_rank, 0.0)


def summarize(pq: PerQuery, ks: Sequence[int] = (1, 5, 10), with_map: bool = True,
              ci: bool = True, seed: int = 0, n_boot: int = 1000) -> dict[str, Any]:
    vals: dict[str, np.ndarray] = {f"R@{k}": recall_at(pq.first_rank, k) for k in ks}
    vals["MRR"] = reciprocal_rank(pq.first_rank)
    if with_map:
        vals["mAP"] = pq.ap
    out: dict[str, Any] = {name: float(v.mean()) for name, v in vals.items()}
    finite = pq.first_rank[np.isfinite(pq.first_rank)]
    out["median_rank"] = float(np.median(pq.first_rank)) if len(finite) == len(pq.first_rank) \
        else None
    out["n_queries"] = len(pq)
    if ci and len(pq) > 1:
        out["ci95"] = {name: bootstrap_ci(v, pq.clusters, n_boot=n_boot, seed=seed)
                       for name, v in vals.items()}
    return out


# --------------------------------------------------------------- uncertainty
def _cluster_index(clusters: np.ndarray | None, n: int) -> list[np.ndarray]:
    if clusters is None:
        return [np.array([i]) for i in range(n)]
    order = np.argsort(clusters, kind="stable")
    _, starts = np.unique(clusters[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap_ci(values: np.ndarray, clusters: np.ndarray | None = None, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> list[float]:
    """Percentile bootstrap CI of the mean, resampling clusters (e.g. images)
    rather than individual queries when queries are grouped."""
    groups = _cluster_index(clusters, len(values))
    sums = np.array([values[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray | None = None,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Difference of means (b - a) with a clustered paired bootstrap CI and a
    two-sided p-value (fraction of resamples whose difference crosses zero)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    groups = _cluster_index(clusters, len(d))
    sums = np.array([d[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_value": float(min(p, 1.0))}


def metric_values(pq: PerQuery, name: str) -> np.ndarray:
    if name.startswith("R@"):
        return recall_at(pq.first_rank, int(name[2:]))
    if name == "MRR":
        return reciprocal_rank(pq.first_rank)
    if name == "mAP":
        return pq.ap
    raise KeyError(name)


def recall_vs_exact(approx: Sequence[np.ndarray], exact: Sequence[np.ndarray], k: int) -> float:
    """ANN quality: |approx_topk ∩ exact_topk| / k, averaged over queries."""
    vals = [len(set(map(int, a[:k])) & set(map(int, e[:k]))) / k for a, e in zip(approx, exact)]
    return float(np.mean(vals))


def percentiles(samples_ms: Sequence[float]) -> dict[str, float]:
    s = np.asarray(samples_ms, dtype=np.float64)
    return {"p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95)),
            "p99_ms": float(np.percentile(s, 99)), "mean_ms": float(s.mean()), "n": len(s)}
