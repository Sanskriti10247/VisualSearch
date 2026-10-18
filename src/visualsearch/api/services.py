"""Long-lived application services shared by the HTTP handlers.

Keeps model, index and job state out of the route functions so the HTTP layer
stays thin, and so tests can inject a container wired with fake embedders.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visualsearch import __version__
from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog
from visualsearch.index.manager import IndexManager
from visualsearch.pipeline.ingest import Ingestor, summarize
from visualsearch.retrieval.calibration import CalibrationRegistry
from visualsearch.retrieval.engine import RerankConfig, SearchEngine

log = logging.getLogger(__name__)


class Metrics:
    """In-process request metrics: counts and latency percentiles over a
    sliding window (last 2000 observations per series)."""

    def __init__(self, window: int = 2000):
        self.started = time.time()
        self._lock = threading.Lock()
        self._lat: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._count: dict[tuple[str, int], int] = defaultdict(int)
        self._stage: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def observe_request(self, route: str, status: int, ms: float) -> None:
        with self._lock:
            self._lat[route].append(ms)
            self._count[(route, status)] += 1

    def observe_stages(self, mode: str, timings: dict[str, float]) -> None:
        with self._lock:
            for stage, ms in timings.items():
                self._stage[f"{mode}.{stage}"].append(ms)

    @staticmethod
    def _pct(values) -> dict[str, float]:
        a = np.asarray(values, dtype=np.float64)
        return {"count": len(a), "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            routes = {r: self._pct(v) for r, v in self._lat.items() if v}
            counts: dict[str, dict[str, int]] = defaultdict(dict)
            for (r, s), n in self._count.items():
                counts[r][str(s)] = n
            stages = {s: self._pct(v) for s, v in self._stage.items() if v}
        return {"uptime_s": round(time.time() - self.started, 1), "routes": routes,
                "status_counts": dict(counts), "search_stages": stages}

    def prometheus(self, extra_gauges: dict[str, float]) -> str:
        snap = self.snapshot()
        lines = ["# TYPE visualsearch_requests_total counter"]
        for route, by_status in snap["status_counts"].items():
            for status, n in by_status.items():
                lines.append(f'visualsearch_requests_total{{route="{route}",status="{status}"}} {n}')
        lines.append("# TYPE visualsearch_request_latency_ms summary")
        for route, p in snap["routes"].items():
            for q, key in (("0.5", "p50_ms"), ("0.95", "p95_ms")):
                lines.append(f'visualsearch_request_latency_ms{{route="{route}",quantile="{q}"}} '
                             f'{p[key]:.3f}')
        for name, value in extra_gauges.items():
            lines.append(f"# TYPE visualsearch_{name} gauge")
            lines.append(f"visualsearch_{name} {value}")
        return "\n".join(lines) + "\n"


@dataclass
class Services:
    settings: Settings
    catalog: Catalog
    manager: IndexManager
    engine: SearchEngine
    ingestor: Ingestor
    metrics: Metrics = field(default_factory=Metrics)
    jobs: JobRunner | None = None
    version: str = __version__

    def __post_init__(self) -> None:
        if self.jobs is None:
            self.jobs = JobRunner(self)

    @classmethod
    def build(cls, settings: Settings, build_missing: bool = True) -> Services:
        settings.ensure_dirs()
        catalog = Catalog(settings.catalog_path)
        manager = IndexManager(settings, catalog)
        manager.open(build_missing=build_missing)
        for rt in manager.runtimes.values():
            # Warm up: load weights and run one forward pass so the first real query
            # does not pay CUDA/cuBLAS initialisation (measured ~1.7 s otherwise).
            rt.embedder.load()
            rt.embedder.embed_images([Image.new("RGB", (64, 64))])
            if rt.embedder.supports("text"):
                rt.embedder.embed_texts(["warm-up"])
        cross_encoder = None
        if settings.enable_cross_encoder:
            from visualsearch.retrieval.cross_encoder import BlipITMReranker

            cross_encoder = BlipITMReranker(device=settings.device)  # loads lazily
        engine = SearchEngine(settings, catalog, manager,
                              CalibrationRegistry(settings.calibration_dir),
                              RerankConfig(**settings.rerank), cross_encoder=cross_encoder)
        return cls(settings, catalog, manager, engine, Ingestor(settings, catalog))

    def close(self) -> None:
        if self.jobs:
            self.jobs.shutdown()


class JobRunner:
    """Runs indexing jobs one at a time on a background thread.

    Jobs are persisted in the catalogue's ``jobs`` table so their status
    survives the request that created them. Uploaded files are spooled to disk
    first so a large batch never sits in memory.
    """

    def __init__(self, services: Services):
        self.services = services
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index-job")
        self.spool = services.settings.root / "spool"

    def submit_index_job(self, files: list[tuple[str, bytes]], collection: str,
                         tags: list[str]) -> str:
        job_id = uuid.uuid4().hex[:16]
        job_dir = self.spool / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        names = []
        for n, (name, data) in enumerate(files):
            p = job_dir / f"{n:05d}"
            p.write_bytes(data)
            names.append((p, name))
        self.services.catalog.create_job(job_id, "index_images", total=len(files))
        self.pool.submit(self._run_index_job, job_id, names, collection, tags)
        return job_id

    def submit_sync_job(self, reason: str) -> str:
        job_id = uuid.uuid4().hex[:16]
        self.services.catalog.create_job(job_id, f"sync:{reason}", total=0)
        self.pool.submit(self._run_sync_job, job_id)
        return job_id

    def _run_index_job(self, job_id: str, files: list[tuple[Path, str]], collection: str,
                       tags: list[str]) -> None:
        cat = self.services.catalog
        cat.update_job(job_id, status="running")
        try:
            results = []
            for n, (path, name) in enumerate(files):
                results.append(self.services.ingestor.ingest_bytes(
                    path.read_bytes(), collection=collection, source_name=name, tags=tags))
                cat.update_job(job_id, progress=n + 1)
            sync = {m: self.services.manager.sync(m) for m in self.services.manager.runtimes}
            summary = summarize(results)
            summary["files"] = [r.to_dict() for r in results]
            summary["sync"] = sync
            cat.update_job(job_id, status="succeeded", result=summary)
        except Exception as e:
            log.exception("index job %s failed", job_id)
            cat.update_job(job_id, status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            shutil.rmtree(self.spool / job_id, ignore_errors=True)

    def _run_sync_job(self, job_id: str) -> None:
        cat = self.services.catalog
        cat.update_job(job_id, status="running")
        try:
            sync = {m: self.services.manager.sync(m) for m in self.services.manager.runtimes}
            cat.update_job(job_id, status="succeeded", result={"sync": sync})
        except Exception as e:
            log.exception("sync job %s failed", job_id)
            cat.update_job(job_id, status="failed", error=f"{type(e).__name__}: {e}")

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
