"""Hybrid (text + image) retrieval.

Two evaluations built from real annotations:

Imagewoof class + exemplar
    query  = (text "a photo of a {breed}, a type of dog.", one image of that breed)
    target = all *other* images of that breed (mAP over the full ranking)
    tuning = Imagewoof train subset; evaluation = Imagewoof val

Flickr30k caption + fragment
    query  = (first caption of the target, a random 15-25%-area crop of it)
    target = the image (R@K) among all 31,014 Flickr30k images
    tuning = Flickr30k val; evaluation = Flickr30k test

Fusion methods (see retrieval.fusion): zscore (background statistics fitted
on Flickr30k val), minmax, rrf, and early (blend the CLIP query vectors; only
when both signals are CLIP). Late fusion scores the full gallery here; the
serving engine scores the union of each signal's top candidates, which gives
the same top of the ranking whenever the fused top-k lies within that union.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

from visualsearch.data.preprocess import load_rgb
from visualsearch.embeddings import IMAGE, TEXT
from visualsearch.embeddings.base import l2_normalize
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import (
    paired_bootstrap,
    per_query_from_ranks,
    ranks_from_scores,
    summarize,
)
from visualsearch.eval.results import chart, save_result, table
from visualsearch.retrieval.calibration import CalibrationRegistry
from visualsearch.retrieval.fusion import RRF_K

WEIGHTS = [round(w, 2) for w in np.linspace(0.0, 1.0, 11)]
TEXT_MODEL = "clip-vit-b32"
IMAGE_MODELS = ["clip-vit-b32", "dinov2-small", "dinov2-base"]


def _norm(s: np.ndarray, method: str, mean: float | None = None, std: float | None = None):
    if method == "zscore":
        return (s - mean) / std
    lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - lo) / np.maximum(hi - lo, 1e-12)


def _rrf_part(s: np.ndarray) -> np.ndarray:
    order = np.argsort(-s, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, s.shape[1] + 1)[None, :]
    return 1.0 / (RRF_K + ranks)


class FusionProblem:
    """Precomputed signal matrices for one query set against one gallery."""

    def __init__(self, st, si, qt, qi, gt, gi, relevant, self_mask, calib_t, calib_i,
                 early_ok: bool):
        self.st, self.si = st, si          # [Q, G] text and image similarity
        self.qt, self.qi, self.gt, self.gi = qt, qi, gt, gi
        self.relevant, self.self_mask = relevant, self_mask
        self.ct, self.ci = calib_t, calib_i
        self.early_ok = early_ok
        self._rrf = None

    def scores(self, method: str, w: float) -> np.ndarray:
        if method == "early":
            q = l2_normalize(w * self.qt + (1 - w) * self.qi)
            s = q @ self.gt.T
        elif method == "rrf":
            if self._rrf is None:
                self._rrf = (_rrf_part(self.st), _rrf_part(self.si))
            s = w * self._rrf[0] + (1 - w) * self._rrf[1]
        elif method == "zscore":
            s = w * _norm(self.st, "zscore", self.ct.mean, self.ct.std) + \
                (1 - w) * _norm(self.si, "zscore", self.ci.mean, self.ci.std)
        else:
            s = w * _norm(self.st, "minmax") + (1 - w) * _norm(self.si, "minmax")
        if self.self_mask is not None:
            s = np.where(self.self_mask, -np.inf, s)
        return s

    def evaluate(self, method: str, w: float, clusters=None):
        s = self.scores(method, w)
        pq = per_query_from_ranks(ranks_from_scores(s, self.relevant), clusters)
        top10 = np.argsort(-s, axis=1)[:, :10]
        pq.extra["P@10"] = np.array([np.isin(t, r).mean() for t, r in zip(top10, self.relevant)])
        return pq


def _woof_problem(ctx: EvalContext, split: str, image_model: str, reg: CalibrationRegistry):
    et, ei = ctx.embedder(TEXT_MODEL), ctx.embedder(image_model)
    recs = ctx.records("imagewoof", [split])
    labels = np.array([r.label for r in recs])
    gt = ctx.image_vectors(et, recs)
    gi = ctx.image_vectors(ei, recs)
    prompts = {lab: f"a photo of a {lab}, a type of dog." for lab in sorted(set(labels))}
    tv = ctx.text_vectors(et, list(prompts.values()))
    qt = tv[[list(prompts).index(lab) for lab in labels]]
    st, si = qt @ gt.T, gi @ gi.T
    rel = [np.nonzero((labels == lab) & (np.arange(len(labels)) != n))[0]
           for n, lab in enumerate(labels)]
    mask = np.eye(len(labels), dtype=bool)
    return FusionProblem(st, si, qt, gt, gt, gi, rel, mask,
                         reg.get(TEXT_MODEL, TEXT), reg.get(image_model, IMAGE),
                         early_ok=image_model == TEXT_MODEL), labels


def _flickr_problem(ctx: EvalContext, split: str, image_model: str, reg: CalibrationRegistry):
    et, ei = ctx.embedder(TEXT_MODEL), ctx.embedder(image_model)
    targets = ctx.records("flickr30k", [split])
    gallery = ctx.records("flickr30k", ["test", "val", "train"])
    gidx = {r.id: n for n, r in enumerate(gallery)}
    gt, gi = ctx.image_vectors(et, gallery), ctx.image_vectors(ei, gallery)
    caps = [ctx.catalog.captions_for(r.id)[0] for r in targets]
    qt = ctx.text_vectors(et, caps)
    seed = ctx.settings.seed
    keys = [hashlib.sha256(f"frag:{seed}:{r.sha256}".encode()).hexdigest() for r in targets]
    by_key = dict(zip(keys, targets))

    def crops(emb):
        def compute(ks):
            out = []
            for k in ks:
                r = by_key[k]
                img = load_rgb(ctx.settings.images_dir / r.path)
                rng = random.Random(f"frag:{seed}:{r.sha256}")
                w, h = img.size
                area = rng.uniform(0.15, 0.25)
                cw, ch = int(w * area ** 0.5), int(h * area ** 0.5)
                x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
                out.append(img.crop((x, y, x + cw, y + ch)))
            return emb.embed_images(out)
        return ctx.store(emb, IMAGE).get_or_compute(keys, compute, chunk=128)

    qi_t = crops(et)  # CLIP image vector of the fragment (for early fusion)
    qi = crops(ei) if image_model != TEXT_MODEL else qi_t
    rel = [np.array([gidx[r.id]]) for r in targets]
    p = FusionProblem((qt @ gt.T).astype(np.float32), (qi @ gi.T).astype(np.float32), qt,
                      qi_t, gt, gi, rel, None, reg.get(TEXT_MODEL, TEXT),
                      reg.get(image_model, IMAGE), early_ok=True)
    return p, np.array([r.id for r in targets])


def _sweep(problem: FusionProblem, method: str, clusters=None, metric: str = "mAP"):
    out = {}
    for w in WEIGHTS:
        pq = problem.evaluate(method, w, clusters)
        m = summarize(pq, ci=False)
        out[w] = m[metric] if metric in m else float(pq.extra["P@10"].mean())
    return out


def _methods(problem: FusionProblem) -> list[str]:
    return ["zscore", "minmax", "rrf"] + (["early"] if problem.early_ok else [])


def _run(ctx, name, builder, tune_split, eval_split, metric, reg):
    rows, raw, charts_ = [], {}, []
    for im in IMAGE_MODELS:
        tune, _ = builder(ctx, tune_split, im, reg)
        test, _ = builder(ctx, eval_split, im, reg)
        text_only = test.evaluate("zscore", 1.0)
        image_only = test.evaluate("zscore", 0.0)
        base = {"text only (CLIP B/32)": text_only, f"image only ({im})": image_only}
        for label, pq in base.items():
            if label.startswith("text only") and im != IMAGE_MODELS[0]:
                continue
            m = summarize(pq, ci=False)
            rows.append([label, "—", "—", m.get(metric), m["R@1"], float(pq.extra["P@10"].mean()), ""])
        best_single = image_only if summarize(image_only, ci=False)[metric] >= \
            summarize(text_only, ci=False)[metric] else text_only
        series = {}
        for method in _methods(tune):
            curve = _sweep(tune, method, metric=metric)
            w_best = max(curve, key=curve.get)
            series[f"{method}"] = list(curve.items())
            pq = test.evaluate(method, w_best)
            m = summarize(pq, ci=False)
            vals = pq.ap if metric == "mAP" else (pq.first_rank <= 1).astype(float)
            ref = best_single.ap if metric == "mAP" else (best_single.first_rank <= 1).astype(float)
            sig = paired_bootstrap(ref, vals)
            raw[f"{im}/{method}"] = {"w_text": w_best, "tune_curve": curve, "test": m,
                                     "vs_best_single": sig}
            rows.append([f"text + image ({im})", method, w_best, m.get(metric), m["R@1"],
                         float(pq.extra["P@10"].mean()),
                         f"{100 * sig['diff']:+.1f} (p={sig['p_value']:.3f})"])
        charts_.append(chart(f"{name}: tuning curve on {tune_split}, image model {im}", "line",
                             "text weight w (0 = image only, 1 = text only)", metric, series))
    return rows, raw, charts_


def run(ctx: EvalContext) -> dict[str, Any]:
    reg = CalibrationRegistry(ctx.settings.calibration_dir)
    for m in IMAGE_MODELS:
        if reg.get(m, IMAGE) is None:
            raise RuntimeError(f"fit calibration first: visualsearch calibrate --models {m}")
    cols = ["Query", "Fusion", "w_text (tuned)", "Primary metric", "R@1", "P@10",
            "Δ vs best single signal"]
    rows_w, raw_w, charts_w = _run(ctx, "Imagewoof", _woof_problem, "train", "val", "mAP", reg)
    rows_f, raw_f, charts_f = _run(ctx, "Flickr30k", _flickr_problem, "val", "test", "R@1", reg)
    report = {
        "title": "Hybrid text + image retrieval",
        "description": (
            "Late fusion combines scores from separate indexes (never mixing vectors from different "
            "spaces); 'early' blends CLIP text and CLIP image query vectors. The text weight is "
            "tuned on the tuning split and applied once to the evaluation split. Δ is the paired "
            "bootstrap difference against the better single signal on the evaluation split "
            "(mAP for Imagewoof, R@1 for Flickr30k)."),
        "tables": [table("Imagewoof val: breed name + exemplar image (primary metric mAP)", cols,
                         rows_w, highlight_best=[3, 4, 5]),
                   table("Flickr30k test: caption + 15-25% fragment, 31K gallery (primary metric R@1)",
                         cols, rows_f, highlight_best=[3, 4, 5])],
        "charts": charts_w + charts_f,
    }
    save_result(ctx.settings, "hybrid", {"imagewoof": raw_w, "flickr30k": raw_f}, report)
    return {"imagewoof": raw_w, "flickr30k": raw_f}
