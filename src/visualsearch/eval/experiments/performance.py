"""Embedding throughput / memory, and end-to-end query latency."""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Self

import numpy as np
import psutil

from visualsearch.data.catalog import FilterSpec
from visualsearch.data.preprocess import load_rgb
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import percentiles
from visualsearch.eval.results import save_result, table

MODELS = ["clip-vit-b32", "clip-vit-b16", "clip-vit-l14", "dinov2-small", "dinov2-base"]


class PeakRSS:
    """Samples process RSS on a background thread; reports the peak while active."""

    def __init__(self, interval: float = 0.005):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0

    def __enter__(self) -> Self:
        import threading

        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()

        def run() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, self.proc.memory_info().rss)
                self._stop.wait(self.interval)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()


def throughput(ctx: EvalContext, devices: tuple[str, ...] = ("cuda", "cpu")) -> dict[str, Any]:
    import torch

    from visualsearch.embeddings import create_embedder

    recs = ctx.records("flickr30k", ["test"])[:256]
    images = [load_rgb(ctx.settings.images_dir / r.path) for r in recs]
    texts = [t for t, _ in zip(ctx.flickr_queries("test")[0], range(1024))]
    proc = psutil.Process(os.getpid())
    rows, raw = [], []
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        for m in MODELS:
            emb = create_embedder(ctx.specs[m], device=device)
            gc.collect()
            rss0 = proc.memory_info().rss  # baseline before weights are loaded
            t = time.perf_counter()
            emb.load()
            load_s = time.perf_counter() - t
            n_img = 256 if device == "cuda" else (64 if "l14" in m or "base" in m else 128)
            for bs in (1, 8, 32, 64):
                if device == "cpu" and bs == 1 and n_img > 32:
                    sample = images[:32]
                else:
                    sample = images[:n_img]
                emb.embed_images(sample[:min(len(sample), bs * 2)], batch_size=bs)  # warm-up
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                with PeakRSS() as mem:
                    t = time.perf_counter()
                    emb.embed_images(sample, batch_size=bs)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t
                peak = (torch.cuda.max_memory_allocated() / 1e6 if device == "cuda"
                        else (mem.peak - rss0) / 1e6)
                row = {"device": device, "model": m, "batch_size": bs, "modality": "image",
                       "items_per_s": len(sample) / dt, "ms_per_item": 1000 * dt / len(sample),
                       "peak_mem_mb": peak, "load_s": load_s, "n": len(sample)}
                raw.append(row)
                rows.append([device, m, "image (incl. preprocessing)", bs,
                             round(row["items_per_s"], 1), round(row["ms_per_item"], 2),
                             round(peak, 0)])
            if emb.supports("text"):
                for bs in (1, 64, 256):
                    sample = texts[:64] if bs == 1 else texts
                    emb.embed_texts(sample[:bs], batch_size=bs)
                    t = time.perf_counter()
                    emb.embed_texts(sample, batch_size=bs)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t
                    raw.append({"device": device, "model": m, "batch_size": bs, "modality": "text",
                                "items_per_s": len(sample) / dt, "ms_per_item": 1000 * dt / len(sample)})
                    rows.append([device, m, "text", bs, round(len(sample) / dt, 1),
                                 round(1000 * dt / len(sample), 2), None])
            n_params = sum(p.numel() for p in emb.model.parameters()) / 1e6
            raw.append({"device": device, "model": m, "params_m": n_params, "load_s": load_s})
            del emb
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    env_threads = torch.get_num_threads()
    report = {
        "title": "Embedding throughput and memory",
        "description": (
            "Images: 256 Flickr30k test images already decoded in memory (fewer on CPU for large "
            "models); time includes the model's own resize/crop/normalise. GPU memory = peak "
            "allocated by PyTorch; CPU memory = peak process RSS during the run above the "
            "pre-load baseline (weights + activations + preprocessing; sampled every 5 ms). "
            "GPU throughput for small models is bounded by CPU-side PIL preprocessing. fp32 throughout, "
            f"TF32 disabled. CPU uses {env_threads} PyTorch threads."),
        "tables": [table("Throughput", ["Device", "Model", "Input", "Batch", "items/s",
                                        "ms/item", "Peak MB"], rows)],
    }
    save_result(ctx.settings, "throughput", {"rows": raw, "torch_threads": env_threads}, report)
    return {"rows": raw}


def query_latency(ctx: EvalContext, n: int = 300) -> dict[str, Any]:
    """End-to-end engine latency (embed + search + fuse/rerank + metadata) for the
    serving configuration, on the full catalogue, in-process (no HTTP)."""
    from visualsearch.api.services import Services

    svc = Services.build(ctx.settings, build_missing=True)
    texts, _ = ctx.flickr_queries("test")
    recs = ctx.records("flickr30k", ["test"])
    rng = np.random.default_rng(ctx.settings.seed)
    qtexts = [texts[i] for i in rng.choice(len(texts), n, replace=False)]
    qimgs = [load_rgb(ctx.settings.images_dir / recs[i].path)
             for i in rng.choice(len(recs), min(n, 100), replace=False)]
    runs = {
        "text": lambda i: svc.engine.text(qtexts[i % n], k=24),
        "text + PRF": lambda i: svc.engine.text(qtexts[i % n], k=24, rerank="prf"),
        "text + visual PRF": lambda i: svc.engine.text(qtexts[i % n], k=24, rerank="visual_prf"),
        "text + cross-encoder (top 20)": lambda i: svc.engine.text(qtexts[i % n], k=20,
                                                                    rerank="cross_encoder"),
        "image upload (DINOv2-S)": lambda i: svc.engine.image(qimgs[i % len(qimgs)], k=24),
        "image upload (CLIP B/32)": lambda i: svc.engine.image(qimgs[i % len(qimgs)], k=24,
                                                               model="clip-vit-b32"),
        "find similar by id": lambda i: svc.engine.image(image_id=recs[i % len(recs)].id, k=24),
        "hybrid (zscore)": lambda i: svc.engine.hybrid(qtexts[i % n], qimgs[i % len(qimgs)], k=24),
        "text + filter (portrait)": lambda i: svc.engine.text(
            qtexts[i % n], k=24, filters=FilterSpec(orientations=["portrait"])),
    }
    rows, raw = [], {}
    for name, fn in runs.items():
        reps = 60 if "cross-encoder" in name else n
        for i in range(5):
            fn(i)
        # Every query type reuses the same query texts; clear the query-embedding LRU
        # so each one pays the real embedding cost instead of hitting the cache.
        for rt in svc.manager.runtimes.values():
            with rt.query_cache_lock:
                rt.query_cache.clear()
        samples, stages = [], {}
        for i in range(reps):
            t = time.perf_counter()
            out = fn(i)
            samples.append(1000 * (time.perf_counter() - t))
            for k, v in out.timings_ms.items():
                stages.setdefault(k, []).append(v)
        p = percentiles(samples)
        raw[name] = {"total": p, "stages": {k: percentiles(v) for k, v in stages.items()}}
        rows.append([name, reps, round(p["p50_ms"], 2), round(p["p95_ms"], 2),
                     ", ".join(f"{k} {np.median(v):.1f}" for k, v in stages.items() if k != "total")])
    status = svc.manager.status()
    svc.close()
    report = {
        "title": "End-to-end query latency (serving configuration)",
        "description": (
            f"In-process SearchEngine, {ctx.settings.index_type} index over the full catalogue "
            f"({status[next(iter(status))]['index']['ntotal']:,} images), models on "
            f"{status[next(iter(status))]['model']['device']}. Excludes HTTP and image decoding of "
            "uploads; includes query embedding (query cache cleared per query type), search, "
            "fusion/reranking. Stage medians in ms."),
        "tables": [table("Latency", ["Query type", "n", "p50 ms", "p95 ms", "Stage medians (ms)"],
                         rows)],
    }
    save_result(ctx.settings, "query_latency", {"results": raw}, report)
    return raw
