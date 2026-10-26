"""Search engine: query embedding -> candidate retrieval -> fusion/rerank.

Independent of HTTP; the API layer only validates input and hydrates results.

Text search      CLIP text vector -> CLIP image index.
Image search     image vector in the chosen space (CLIP or DINOv2) -> that index.
Hybrid search    text + image (+ filters). Late fusion over the union of both
                 candidate lists with calibrated score normalisation, or RRF.
                 ``early`` fusion (one blended query vector) is allowed only if
                 both signals come from the same CLIP space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog, FilterSpec
from visualsearch.embeddings import IMAGE, TEXT, UnsupportedModalityError
from visualsearch.embeddings.base import l2_normalize
from visualsearch.index.manager import IndexManager
from visualsearch.retrieval.calibration import CalibrationRegistry, ScoreCalibration
from visualsearch.retrieval.fusion import fuse
from visualsearch.retrieval.rerank import mix, prf_scores

RERANKERS = ("none", "prf", "visual_prf", "cross_encoder")
FUSIONS = ("zscore", "minmax", "rrf", "early")


class SearchError(ValueError):
    """Invalid search request (maps to HTTP 422)."""


@dataclass
class RerankConfig:
    """Defaults selected on validation data (results/rerank.json, docs/REPORT.md §9).

    PRF for text queries did not improve test R@1 in either space; it stays
    available for experimentation. The cross-encoder's best mix on val was the
    ITM score alone (gamma = 1).
    """

    prf_m: int = 5
    prf_lam: float = 0.1
    prf_weighting: str = "softmax"
    visual_prf_model: str = "dinov2-small"
    visual_prf_m: int = 5
    visual_prf_lam: float = 0.3
    visual_prf_weighting: str = "uniform"
    image_prf_m: int = 10        # image->image query expansion (tuned on Imagewoof train)
    image_prf_lam: float = 0.6
    cross_encoder_gamma: float = 1.0
    cross_encoder_candidates: int = 20


@dataclass
class Signal:
    name: str           # e.g. "clip-vit-b32:text"
    model: str
    modality: str
    raw: np.ndarray
    calib: ScoreCalibration | None


@dataclass
class Hit:
    image_id: int
    score: float
    signals: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class SearchOutcome:
    hits: list[Hit]
    timings_ms: dict[str, float]
    strategy: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class SearchEngine:
    def __init__(self, settings: Settings, catalog: Catalog, manager: IndexManager,
                 calibrations: CalibrationRegistry | None = None,
                 rerank_config: RerankConfig | None = None, cross_encoder=None):
        self.settings = settings
        self.catalog = catalog
        self.manager = manager
        self.calibrations = calibrations or CalibrationRegistry(settings.calibration_dir)
        self.rerank_config = rerank_config or RerankConfig()
        self._cross_encoder = cross_encoder

    # ----------------------------------------------------------- helpers
    def _calib(self, model: str, modality: str) -> ScoreCalibration | None:
        rt = self.manager.runtime(model)
        return self.calibrations.get(model, modality, rt.embedder.fingerprint)

    def _resolve_filter(self, filters: FilterSpec | None, warnings: list[str]
                        ) -> np.ndarray | None:
        if filters is None or filters.is_empty():
            return None
        allowed = self.catalog.resolve_filter(filters)
        if allowed is not None and len(allowed) == 0:
            warnings.append("No images match the metadata filters.")
        return allowed

    def _search(self, model: str, q: np.ndarray, n: int, allowed: np.ndarray | None):
        rt = self.manager.runtime(model)
        if rt.index is None:
            raise SearchError(f"index for model '{model}' is not built")
        with rt.lock.read():
            res = rt.index.search(q[None, :], n, allowed=allowed)[0]
        return res

    def _vectors(self, model: str, ids: np.ndarray) -> np.ndarray:
        rt = self.manager.runtime(model)
        with rt.lock.read():
            return rt.index.vectors_for(ids)

    def _check_k(self, k: int) -> int:
        if not 1 <= k <= self.settings.max_k:
            raise SearchError(f"k must be between 1 and {self.settings.max_k}")
        return k

    def embed_text(self, model: str, text: str) -> np.ndarray:
        rt = self.manager.runtime(model)
        if not rt.embedder.supports(TEXT):
            raise SearchError(f"model '{model}' cannot embed text (image-only model); "
                              f"use a CLIP model for text queries")
        return self.manager.text_vectors(model, [self.settings.text_query_template.replace("{}", text, 1)])[0]

    def embed_image(self, model: str, image: Image.Image | None, image_id: int | None
                    ) -> np.ndarray:
        if image_id is not None:
            rt = self.manager.runtime(model)
            with rt.lock.read():
                if rt.index is None or not rt.index.has(image_id):
                    raise SearchError(f"image {image_id} is not indexed for model '{model}'")
                return rt.index.vectors_for(np.array([image_id]))[0]
        if image is None:
            raise SearchError("an image or image_id is required")
        return self.manager.runtime(model).embedder.embed_images([image])[0]

    def _signal_info(self, sig: Signal, idx: int) -> dict[str, float]:
        raw = float(sig.raw[idx])
        info = {"raw": raw}
        if sig.calib is not None:
            info["z"] = float(sig.calib.z(raw))
            info["percentile"] = float(sig.calib.percentile(raw))
            info["prob"] = float(sig.calib.prob(raw))
        return info

    def _finalize(self, ids: np.ndarray, final: np.ndarray, signals: list[Signal], k: int,
                  exclude: set[int], min_prob: float | None) -> list[Hit]:
        order = np.argsort(-final, kind="stable")
        hits = []
        for j in order:
            iid = int(ids[j])
            if iid in exclude:
                continue
            h = Hit(iid, float(final[j]), {s.name: self._signal_info(s, j) for s in signals})
            if min_prob is not None:
                probs = [v["prob"] for v in h.signals.values() if "prob" in v]
                if probs and max(probs) < min_prob:
                    continue
            hits.append(h)
            if len(hits) == k:
                break
        return hits

    # ------------------------------------------------------------- text
    def text(self, query: str, k: int | None = None, model: str | None = None,
             filters: FilterSpec | None = None, rerank: str = "none",
             candidates: int | None = None, min_prob: float | None = None) -> SearchOutcome:
        t0 = time.perf_counter()
        warnings: list[str] = []
        k = self._check_k(self.settings.default_k if k is None else k)
        model = model or self.settings.default_text_model
        query = (query or "").strip()
        if not query:
            raise SearchError("query text must not be empty")
        if rerank not in RERANKERS:
            raise SearchError(f"unknown reranker '{rerank}'; choose from {RERANKERS}")
        if rerank == "cross_encoder" and self.cross_encoder is None:
            raise SearchError("the cross_encoder reranker is not available on this server; "
                              "use 'none', 'prf' or 'visual_prf'")
        n_cand = max(k, candidates or self.settings.rerank_candidates) if rerank != "none" else k
        if rerank == "cross_encoder":
            n_cand = max(k, candidates or self.rerank_config.cross_encoder_candidates)
        timings: dict[str, float] = {}

        t = time.perf_counter()
        q = self.embed_text(model, query)
        timings["embed_query"] = _ms(t)
        t_f = time.perf_counter()
        allowed = self._resolve_filter(filters, warnings)
        if allowed is not None:
            timings["filter"] = _ms(t_f)
        t = time.perf_counter()
        res = self._search(model, q, n_cand, allowed)
        timings["search"] = _ms(t)
        signals = [Signal(f"{model}:text", model, TEXT, res.scores, self._calib(model, TEXT))]
        final = res.scores.astype(np.float64)

        t = time.perf_counter()
        final, rerank_info = self._rerank_text(rerank, query, q, res.ids, final, signals, warnings)
        if rerank != "none":
            timings["rerank"] = _ms(t)
        hits = self._finalize(res.ids, final, signals, k, set(), min_prob)
        timings["total"] = _ms(t0)
        return SearchOutcome(hits, timings, {
            "mode": "text", "model": model, "index": self.manager.kind,
            "filter_strategy": res.strategy, "filter_matches": None if allowed is None else len(allowed),
            "candidates": len(res.ids), "rerank": rerank, **rerank_info}, warnings)

    def _rerank_text(self, rerank: str, query: str, q: np.ndarray, ids: np.ndarray,
                     first: np.ndarray, signals: list[Signal], warnings: list[str]):
        cfg = self.rerank_config
        if rerank == "none" or len(ids) < 2:
            return first, {}
        if rerank == "prf":
            vecs = self._vectors(signals[0].model, ids)
            return prf_scores(first, vecs, cfg.prf_m, cfg.prf_lam, cfg.prf_weighting), {
                "rerank_params": {"m": cfg.prf_m, "lam": cfg.prf_lam}}
        if rerank == "visual_prf":
            vm = cfg.visual_prf_model
            if vm not in self.manager.runtimes:
                warnings.append(f"visual_prf needs model '{vm}', which is not loaded; skipped")
                return first, {}
            vecs = self._vectors(vm, ids)
            return prf_scores(first, vecs, cfg.visual_prf_m, cfg.visual_prf_lam,
                              cfg.visual_prf_weighting), {
                "rerank_params": {"space": vm, "m": cfg.visual_prf_m, "lam": cfg.visual_prf_lam}}
        if rerank == "cross_encoder":
            ce = self.cross_encoder
            recs = self.catalog.get_many(ids)
            imgs = [ce.load_image(self.settings.images_dir / recs[int(i)].path) for i in ids]
            itm = ce.score(query, imgs)
            signals.append(Signal("blip-itm", "blip-itm", TEXT, itm, None))
            return mix(first, itm, cfg.cross_encoder_gamma), {
                "rerank_params": {"gamma": cfg.cross_encoder_gamma, "model": ce.hf_id}}
        raise SearchError(f"unknown reranker '{rerank}'")

    @property
    def cross_encoder(self):
        return self._cross_encoder

    # ------------------------------------------------------------ image
    def image(self, image: Image.Image | None = None, image_id: int | None = None,
              k: int | None = None, model: str | None = None, filters: FilterSpec | None = None,
              rerank: str = "none", exclude_query: bool = True,
              candidates: int | None = None, min_prob: float | None = None,
              query_sha256: str | None = None) -> SearchOutcome:
        t0 = time.perf_counter()
        warnings: list[str] = []
        k = self._check_k(self.settings.default_k if k is None else k)
        model = model or self.settings.default_image_model
        if rerank not in ("none", "prf"):
            raise SearchError("image search supports rerank='none' or 'prf'")
        timings: dict[str, float] = {}
        t = time.perf_counter()
        q = self.embed_image(model, image, image_id)
        timings["embed_query"] = _ms(t)

        exclude: set[int] = set()
        if image_id is not None and exclude_query:
            exclude.add(int(image_id))
        if query_sha256:
            dup = self.catalog.by_sha(query_sha256)
            if dup is not None and not dup.deleted:
                warnings.append(f"The uploaded image is already in the collection (image {dup.id}).")
                if exclude_query:
                    exclude.add(dup.id)

        t_f = time.perf_counter()
        allowed = self._resolve_filter(filters, warnings)
        if allowed is not None:
            timings["filter"] = _ms(t_f)
        n_cand = (max(k, candidates or self.settings.rerank_candidates) if rerank != "none" else k)
        t = time.perf_counter()
        res = self._search(model, q, n_cand + len(exclude), allowed)
        timings["search"] = _ms(t)
        signals = [Signal(f"{model}:image", model, IMAGE, res.scores, self._calib(model, IMAGE))]
        final = res.scores.astype(np.float64)
        info: dict[str, Any] = {}
        if rerank == "prf" and len(res.ids) > 1:
            t = time.perf_counter()
            cfg = self.rerank_config
            final = prf_scores(final, self._vectors(model, res.ids), cfg.image_prf_m,
                               cfg.image_prf_lam)
            timings["rerank"] = _ms(t)
            info = {"rerank_params": {"m": cfg.image_prf_m, "lam": cfg.image_prf_lam}}
        hits = self._finalize(res.ids, final, signals, k, exclude, min_prob)
        timings["total"] = _ms(t0)
        return SearchOutcome(hits, timings, {
            "mode": "image", "model": model, "index": self.manager.kind,
            "query_image_id": image_id, "filter_strategy": res.strategy,
            "filter_matches": None if allowed is None else len(allowed),
            "candidates": len(res.ids), "rerank": rerank, **info}, warnings)

    # ----------------------------------------------------------- hybrid
    def hybrid(self, text: str | None, image: Image.Image | None = None,
               image_id: int | None = None, k: int | None = None,
               text_model: str | None = None, image_model: str | None = None,
               text_weight: float = 0.5, fusion: str = "zscore",
               filters: FilterSpec | None = None, candidates: int | None = None,
               exclude_query: bool = True, min_prob: float | None = None,
               query_sha256: str | None = None) -> SearchOutcome:
        text = (text or "").strip()
        has_image = image is not None or image_id is not None
        if not text and not has_image:
            raise SearchError("hybrid search needs text, an image, or both")
        if not text:
            out = self.image(image, image_id, k, image_model, filters, exclude_query=exclude_query,
                             min_prob=min_prob, query_sha256=query_sha256)
            out.warnings.append("No text given: ran image-only search.")
            return out
        if not has_image:
            out = self.text(text, k, text_model, filters, min_prob=min_prob)
            out.warnings.append("No image given: ran text-only search.")
            return out
        if fusion not in FUSIONS:
            raise SearchError(f"unknown fusion '{fusion}'; choose from {FUSIONS}")
        if not 0.0 <= text_weight <= 1.0:
            raise SearchError("text_weight must be in [0, 1]")

        t0 = time.perf_counter()
        warnings: list[str] = []
        k = self._check_k(self.settings.default_k if k is None else k)
        text_model = text_model or self.settings.default_text_model
        image_model = image_model or self.settings.default_image_model
        timings: dict[str, float] = {}
        t = time.perf_counter()
        qt = self.embed_text(text_model, text)
        qi = self.embed_image(image_model, image, image_id)
        timings["embed_query"] = _ms(t)
        t_f = time.perf_counter()
        allowed = self._resolve_filter(filters, warnings)
        if allowed is not None:
            timings["filter"] = _ms(t_f)
        exclude = {int(image_id)} if (image_id is not None and exclude_query) else set()
        if query_sha256 and exclude_query:
            dup = self.catalog.by_sha(query_sha256)
            if dup is not None:
                exclude.add(dup.id)
        n_cand = max(k, candidates or self.settings.rerank_candidates) + len(exclude)

        if fusion == "early":
            if text_model != image_model:
                raise SearchError("early fusion needs text and image in the same (CLIP) space; "
                                  f"got '{text_model}' and '{image_model}'")
            q = l2_normalize(text_weight * qt + (1 - text_weight) * qi)
            t = time.perf_counter()
            res = self._search(text_model, q, n_cand, allowed)
            timings["search"] = _ms(t)
            ids = res.ids
            st = self._vectors(text_model, ids) @ qt
            si = self._vectors(image_model, ids) @ qi
            final = res.scores.astype(np.float64)
            strategy_extra = {"filter_strategy": res.strategy}
        else:
            t = time.perf_counter()
            rt_ = self._search(text_model, qt, n_cand, allowed)
            ri_ = self._search(image_model, qi, n_cand, allowed)
            timings["search"] = _ms(t)
            t = time.perf_counter()
            ids = np.union1d(rt_.ids, ri_.ids)
            st = self._vectors(text_model, ids) @ qt
            si = self._vectors(image_model, ids) @ qi
            final = fuse([st, si], [text_weight, 1 - text_weight], fusion,
                         [self._calib(text_model, TEXT), self._calib(image_model, IMAGE)])
            timings["fuse"] = _ms(t)
            strategy_extra = {"filter_strategy": rt_.strategy}
            if fusion == "zscore" and (self._calib(text_model, TEXT) is None or
                                       self._calib(image_model, IMAGE) is None):
                warnings.append("No calibration fitted for one of the models; "
                                "fell back to per-query min-max normalisation.")
        signals = [Signal(f"{text_model}:text", text_model, TEXT, st, self._calib(text_model, TEXT)),
                   Signal(f"{image_model}:image", image_model, IMAGE, si,
                          self._calib(image_model, IMAGE))]
        hits = self._finalize(ids, final, signals, k, exclude, min_prob)
        timings["total"] = _ms(t0)
        return SearchOutcome(hits, timings, {
            "mode": "hybrid", "text_model": text_model, "image_model": image_model,
            "fusion": fusion, "text_weight": text_weight, "index": self.manager.kind,
            "filter_matches": None if allowed is None else len(allowed),
            "candidates": len(ids), **strategy_extra}, warnings)


def _ms(t: float) -> float:
    return round(1000 * (time.perf_counter() - t), 3)


__all__ = [
    "Hit",
    "RerankConfig",
    "SearchEngine",
    "SearchError",
    "SearchOutcome",
    "UnsupportedModalityError",
]
