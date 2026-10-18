"""Exact vs. approximate nearest-neighbour search.

Setting A (real gallery): CLIP ViT-B/32 vectors of all 31,014 Flickr30k images;
  queries = the 5,000 test captions. Reports recall@10 against exact search,
  the end-to-end task metric (is the annotated image retrieved?), latency,
  build time and index size.
Setting B (scale): A's images plus every non-test Flickr30k caption embedding
  (~180k real CLIP vectors). Same queries. Ground truth = exact top-10.

Latency is single-query (batch of 1), the serving pattern, measured over
1,000 queries after warm-up, on CPU (FAISS-CPU), with FAISS's default OpenMP
threading. Index size is the serialised FAISS index; the serving index also
keeps float32 vectors for filtering/reranking (reported separately).
"""

from __future__ import annotations

import gc
import time
from typing import Any

import faiss
import numpy as np

from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import percentiles, recall_vs_exact
from visualsearch.eval.results import chart, save_result, table
from visualsearch.index.faiss_index import FaissIndex

MODEL = "clip-vit-b32"
SWEEPS: list[tuple[str, dict[str, Any], str, list[Any]]] = [
    ("flat", {}, "", [None]),
    ("hnsw", {"M": 16, "ef_construction": 200}, "ef_search", [16, 32, 64, 128, 256]),
    ("hnsw", {"M": 32, "ef_construction": 200}, "ef_search", [16, 32, 64, 128, 256]),
    ("ivf_flat", {"nlist": "auto"}, "nprobe", [1, 2, 4, 8, 16, 32, 64]),
    ("ivf_pq", {"nlist": "auto", "m": 32, "nbits": 8, "refine": 0}, "nprobe", [4, 16, 64]),
    ("ivf_pq", {"nlist": "auto", "m": 32, "nbits": 8, "refine": 4}, "nprobe", [4, 16, 64]),
]


def _latency(index, queries: np.ndarray, k: int, n: int = 1000) -> dict[str, float]:
    for q in queries[:20]:
        index.search(q[None], k)
    samples = []
    for q in queries[:n]:
        t = time.perf_counter()
        index.search(q[None], k)
        samples.append(1000 * (time.perf_counter() - t))
    return percentiles(samples)


def _run_setting(name: str, db_ids: np.ndarray, db: np.ndarray, queries: np.ndarray,
                 targets: np.ndarray | None, k: int = 10) -> list[dict[str, Any]]:
    exact = FaissIndex(db.shape[1], "flat", brute_force_threshold=0)
    exact.add(db_ids, db)
    truth = [r.ids for r in exact.search(queries, k)]
    rows = []
    for kind, params, knob, values in SWEEPS:
        t0 = time.perf_counter()
        idx = FaissIndex(db.shape[1], kind, params, brute_force_threshold=0)
        idx.add(db_ids, db)
        build_s = time.perf_counter() - t0
        size = idx.index_bytes()
        for v in values:
            if knob:
                idx.params[knob] = v
            # Recall and latency use the same k, so they describe the same configuration
            # (the HNSW wrapper raises efSearch to at least k, so a deeper search would
            # silently use a larger efSearch).
            t = time.perf_counter()
            res = idx.search(queries, k)
            batch_qps = len(queries) / (time.perf_counter() - t)
            ranked = [r.ids for r in res]
            row = {
                "setting": name, "kind": kind, "params": {**params, **({knob: v} if knob else {})},
                "label": f"{kind}" + (f" M={params['M']}" if kind == "hnsw" else "") +
                         (f" refine={params['refine']}" if kind == "ivf_pq" else "") +
                         (f" {knob}={v}" if knob else ""),
                "recall@10_vs_exact": recall_vs_exact(ranked, truth, k),
                "latency": _latency(idx, queries, k),
                "batch_qps": batch_qps,
                "build_s": build_s, "index_bytes": size, "n": len(db_ids),
                "resolved": getattr(idx, "resolved", {}),
            }
            if targets is not None:
                # End-to-end task metric from a separate depth-100 search (efSearch >= 100).
                deep = [r.ids for r in idx.search(queries, 100)]
                for kk in (1, 10, 100):
                    row[f"task_R@{kk}"] = float(np.mean([t in r[:kk] for r, t in zip(deep, targets)]))
            rows.append(row)
        del idx
        gc.collect()
    return rows


def _qdrant_row(name: str, db_ids, db, queries, truth_k: int = 10) -> dict[str, Any] | None:
    try:
        from visualsearch.index.qdrant_index import QdrantIndex
    except ImportError:
        return None
    exact = FaissIndex(db.shape[1], "flat", brute_force_threshold=0)
    exact.add(db_ids, db)
    truth = [r.ids for r in exact.search(queries[:200], truth_k)]
    t0 = time.perf_counter()
    qi = QdrantIndex(db.shape[1])
    qi.add(db_ids, db)
    build_s = time.perf_counter() - t0
    res = qi.search(queries[:200], truth_k)
    return {"setting": name, "kind": "qdrant-local", "label": "qdrant (embedded, exact)",
            "recall@10_vs_exact": recall_vs_exact([r.ids for r in res], truth, truth_k),
            "latency": _latency(qi, queries, truth_k, n=200), "build_s": build_s,
            "index_bytes": None, "n": len(db_ids), "params": {}}


def run(ctx: EvalContext) -> dict[str, Any]:
    emb = ctx.embedder(MODEL)
    recs = ctx.records("flickr30k", ["test", "val", "train"])
    ids = np.array([r.id for r in recs], dtype=np.int64)
    img = ctx.image_vectors(emb, recs)
    texts, targets = ctx.flickr_queries("test")
    q = ctx.text_vectors(emb, texts)
    rows = _run_setting("A: 31K images", ids, img, q, targets)
    qd = _qdrant_row("A: 31K images", ids, img, q)
    if qd:
        rows.append(qd)

    other = [t for _, _, t in ctx.catalog.captions(collection="flickr30k", split="train")] + \
            [t for _, _, t in ctx.catalog.captions(collection="flickr30k", split="val")]
    cap = ctx.text_vectors(emb, other)
    db = np.concatenate([img, cap])
    db_ids = np.concatenate([ids, 10_000_000 + np.arange(len(cap), dtype=np.int64)])
    rows += _run_setting(f"B: {len(db) // 1000}K mixed", db_ids, db, q, None)

    faiss_threads = faiss.omp_get_max_threads()
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_setting.setdefault(r["setting"], []).append(r)
    tables, charts = [], []
    for setting, rs in by_setting.items():
        cols = ["Index", "Recall@10 vs exact", "p50 ms", "p95 ms", "Build s", "Index MB"]
        has_task = "task_R@1" in rs[0]
        if has_task:
            cols += ["Task R@1", "Task R@10"]
        trows = []
        for r in rs:
            row = [r["label"], r["recall@10_vs_exact"], round(r["latency"]["p50_ms"], 3),
                   round(r["latency"]["p95_ms"], 3), round(r["build_s"], 2),
                   None if r["index_bytes"] is None else round(r["index_bytes"] / 1e6, 1)]
            if has_task:
                row += [r.get("task_R@1"), r.get("task_R@10")]
            trows.append(row)
        tables.append(table(f"{setting} (n={rs[0]['n']:,})", cols, trows))
        series: dict[str, list[tuple[float, float]]] = {}
        for r in rs:
            fam = r["label"].rsplit(" ", 1)[0] if r["kind"] in ("hnsw", "ivf_flat", "ivf_pq") else r["label"]
            series.setdefault(fam, []).append((r["latency"]["p50_ms"], r["recall@10_vs_exact"]))
        charts.append(chart(f"{setting}: recall vs. latency (p50, single query)", "line",
                            "p50 latency (ms, log)", "Recall@10 vs exact", series, log_x=True))
    report = {
        "title": "Exact vs. approximate nearest-neighbour search",
        "description": (
            f"CLIP ViT-B/32 (512-d, inner product on L2-normalised vectors). Queries: 5,000 "
            f"Flickr30k test captions. CPU, FAISS {faiss.__version__}, {faiss_threads} OpenMP "
            "threads. Index MB = serialised FAISS index (PQ codes are 32 bytes/vector vs 2,048 "
            "for float32). The embedded Qdrant mode is exact brute force in Python and is shown "
            "only to validate the backend, not as a Qdrant server benchmark. Recall and latency "
            "are measured with k=10; 'Task R@K' (is the annotated image retrieved?) comes from a "
            "separate depth-100 search, for which HNSW uses efSearch >= 100."),
        "tables": tables, "charts": charts,
    }
    save_result(ctx.settings, "ann_benchmark", {"rows": rows, "faiss_threads": faiss_threads},
                report)
    return {"rows": rows}
