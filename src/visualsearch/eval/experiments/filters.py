"""Metadata-filtered search: correctness and cost of three strategies.

post-filter    search the unfiltered index for k' = 4k results, then drop
               disallowed ones (common naive approach).
pre-filter     FAISS IDSelector applied during the index traversal.
exact-subset   brute-force scan of only the allowed vectors (the planner's
               choice below ``brute_force_filter_threshold``).

Ground truth: exact top-10 among allowed images. Filters of varying
selectivity are random subsets plus real metadata filters (orientation, split).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from visualsearch.data.catalog import FilterSpec
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import percentiles, recall_vs_exact
from visualsearch.eval.results import save_result, table
from visualsearch.index.faiss_index import FaissIndex

MODEL = "clip-vit-b32"
K = 10


def _post_filter(index: FaissIndex, q: np.ndarray, allowed: set[int], k: int, factor: int = 4):
    res = index.search(q[None], k * factor)[0]
    keep = [i for i in res.ids if int(i) in allowed][:k]
    return np.array(keep, dtype=np.int64)


def run(ctx: EvalContext, n_queries: int = 300) -> dict[str, Any]:
    emb = ctx.embedder(MODEL)
    recs = ctx.records("flickr30k", ["test", "val", "train"])
    ids = np.array([r.id for r in recs], dtype=np.int64)
    vecs = ctx.image_vectors(emb, recs)
    texts, _ = ctx.flickr_queries("test")
    rng = np.random.default_rng(ctx.settings.seed)
    q = ctx.text_vectors(emb, [texts[i] for i in rng.choice(len(texts), n_queries, replace=False)])

    filters: list[tuple[str, np.ndarray]] = []
    for frac in (0.001, 0.01, 0.05, 0.2, 0.5):
        filters.append((f"random {frac:.1%}", np.sort(rng.choice(ids, max(1, int(frac * len(ids))),
                                                                 replace=False))))
    for label, spec in (("orientation=portrait", FilterSpec(orientations=["portrait"])),
                        ("split=test", FilterSpec(splits=["test"], collections=["flickr30k"]))):
        allowed = ctx.catalog.resolve_filter(spec)
        filters.append((label, np.intersect1d(allowed, ids)))

    indexes = {kind: FaissIndex(vecs.shape[1], kind, brute_force_threshold=0)
               for kind in ("flat", "hnsw", "ivf_flat")}
    for idx in indexes.values():
        idx.add(ids, vecs)

    rows, raw = [], []
    for label, allowed in filters:
        allowed_set = set(allowed.tolist())
        truth_idx = FaissIndex(vecs.shape[1], "flat", brute_force_threshold=10**9)
        truth_idx.add(ids, vecs)
        truth = [truth_idx.search(qq[None], K, allowed=allowed)[0].ids for qq in q]
        for kind, idx in indexes.items():
            for strategy in ("post-filter", "pre-filter", "exact-subset"):
                if strategy == "exact-subset" and kind != "flat":
                    continue
                idx.brute_force_threshold = 10**9 if strategy == "exact-subset" else 0
                got, lat = [], []
                for qq in q:
                    t = time.perf_counter()
                    if strategy == "post-filter":
                        r = _post_filter(idx, qq, allowed_set, K)
                    else:
                        r = idx.search(qq[None], K, allowed=allowed)[0].ids
                    lat.append(1000 * (time.perf_counter() - t))
                    got.append(r)
                full = float(np.mean([len(g) == min(K, len(allowed)) for g in got]))
                rec = recall_vs_exact(got, truth, K) if len(allowed) >= K else float(
                    np.mean([set(g) == set(t) for g, t in zip(got, truth)]))
                p = percentiles(lat)
                name = "flat" if strategy == "exact-subset" else kind
                raw.append({"filter": label, "allowed": len(allowed), "index": name,
                            "strategy": strategy, "recall@10": rec, "full_result_rate": full,
                            "latency": p})
                rows.append([label, len(allowed), name, strategy, rec, full,
                             round(p["p50_ms"], 3), round(p["p95_ms"], 3)])
    report = {
        "title": "Metadata-filtered search",
        "description": (
            f"{n_queries} Flickr30k test captions against the 31K CLIP B/32 gallery, k={K}. "
            "Recall@10 is measured against the exact top-10 *among allowed images*; 'full results' "
            "is the fraction of queries that returned k results. Post-filtering over-fetches 4k."),
        "tables": [table("Filtered retrieval", ["Filter", "Allowed", "Index", "Strategy",
                                                "Recall@10", "Full results", "p50 ms", "p95 ms"],
                         rows)],
    }
    save_result(ctx.settings, "filtered_search", {"rows": raw}, report)
    return {"rows": raw}
