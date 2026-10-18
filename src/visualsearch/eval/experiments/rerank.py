"""Second-stage reranking of CLIP text->image candidates (and image->image AQE).

Protocol: every reranker's hyper-parameters are chosen on Flickr30k **val**
(1,014 images / 5,070 captions); the chosen setting is then evaluated once on
**test**. All runs rerank the same first-stage candidate lists (exact CLIP
ViT-B/32 search), so differences are attributable to the reranker alone, and
a reranker can never exceed first-stage Recall@C.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np

from visualsearch.eval import tasks
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import (
    metric_values,
    paired_bootstrap,
    per_query_from_ranks,
    ranks_from_lists,
    summarize,
)
from visualsearch.eval.results import save_result, table
from visualsearch.retrieval.rerank import mix, prf_scores

FIRST = "clip-vit-b32"


class Candidates:
    """First-stage candidate lists for one split (fixed across rerankers)."""

    def __init__(self, ctx: EvalContext, split: str, C: int, distractors: bool = False):
        self.split = split
        r = tasks.t2i(ctx, FIRST, split, distractors=distractors, depth=C)
        self.texts, self.targets = ctx.flickr_queries(split)
        self.ids = r.ranked                       # list of [C] id arrays
        emb = ctx.embedder(FIRST)
        recs, gids = tasks.flickr_gallery(ctx, split, distractors)
        self.records = {rec.id: rec for rec in recs}
        pos = {int(i): n for n, i in enumerate(gids)}
        g = ctx.image_vectors(emb, recs)
        q = ctx.text_vectors(emb, self.texts)
        self.first = [g[[pos[int(i)] for i in ids]] @ qq for ids, qq in zip(self.ids, q)]
        self.baseline = r
        self._vec_cache: dict[str, tuple[dict[int, int], np.ndarray]] = {}
        self.ctx, self.gids, self.recs_list = ctx, gids, recs

    def vectors(self, model: str):
        if model not in self._vec_cache:
            v = self.ctx.image_vectors(self.ctx.embedder(model), self.recs_list)
            self._vec_cache[model] = ({int(i): n for n, i in enumerate(self.gids)}, v)
        return self._vec_cache[model]

    def evaluate(self, new_scores: list[np.ndarray]):
        ranked = [ids[np.argsort(-s, kind="stable")] for ids, s in zip(self.ids, new_scores)]
        return per_query_from_ranks(ranks_from_lists(ranked, [[t] for t in self.targets]),
                                    clusters=self.targets)

    def prf(self, model: str, m: int, lam: float, weighting: str = "uniform"):
        pos, v = self.vectors(model)
        return [prf_scores(f, v[[pos[int(i)] for i in ids]], m, lam, weighting)
                for ids, f in zip(self.ids, self.first)]


def _r1(pq) -> float:
    return float((pq.first_rank <= 1).mean())


def _tune_prf(val: Candidates, model: str):
    grid = list(itertools.product([1, 3, 5, 10], [0.1, 0.2, 0.3, 0.5, 0.7], ["uniform", "softmax"]))
    scores = {}
    for m, lam, wt in grid:
        scores[(m, lam, wt)] = summarize(val.evaluate(val.prf(model, m, lam, wt)), ci=False)["MRR"]
    best = max(scores, key=scores.get)
    return best, {f"m={m},lam={lam},{wt}": v for (m, lam, wt), v in scores.items()}


def _cross_encoder_scores(ce, cands: Candidates, depth: int, settings) -> list[np.ndarray]:
    """BLIP ITM P(match) for the top-`depth` candidates of every query
    (grouped by image so each image's visual features are computed once)."""
    from visualsearch.data.preprocess import load_rgb

    out = [np.zeros(min(depth, len(ids))) for ids in cands.ids]
    by_image: dict[int, list[tuple[int, int]]] = {}
    for qn, ids in enumerate(cands.ids):
        for j, iid in enumerate(ids[:depth]):
            by_image.setdefault(int(iid), []).append((qn, j))
    for iid, pairs in by_image.items():
        rec = cands.records[iid]
        img = load_rgb(settings.images_dir / rec.path)
        texts = [cands.texts[qn] for qn, _ in pairs]
        s = ce.score_texts_for_image(texts, img)
        for (qn, j), val in zip(pairs, s):
            out[qn][j] = val
    return out


def _combine_ce(cands: Candidates, itm: list[np.ndarray], gamma: float, depth: int):
    new = []
    for f, s in zip(cands.first, itm):
        head = mix(f[:depth], s, gamma)
        # keep the head above the tail: reranking only reorders the top-`depth`
        tail = f[depth:] - f[depth:].max() + head.min() - 1 if len(f) > depth else f[depth:]
        new.append(np.concatenate([head, tail]))
    return new


def run(ctx: EvalContext, C: int = 100, ce_depth: int = 20, with_cross_encoder: bool = True
        ) -> dict[str, Any]:
    val, test = Candidates(ctx, "val", C), Candidates(ctx, "test", C)
    base_val, base_test = val.evaluate(val.first), test.evaluate(test.first)
    rows, raw = [], {}

    def add(name: str, params: str, pq_val, pq_test, ms_per_query: float | None):
        mv, mt = summarize(pq_val, ci=False), summarize(pq_test)
        sig = paired_bootstrap(metric_values(base_test, "R@1"), metric_values(pq_test, "R@1"),
                               base_test.clusters)
        raw[name] = {"params": params, "val": mv, "test": mt, "vs_baseline_R@1": sig,
                     "added_ms_per_query": ms_per_query}
        rows.append([name, params, mv["R@1"], mt["R@1"], mt["R@5"], mt["R@10"], mt["MRR"],
                     "" if name.startswith("baseline") else
                     f"{100 * sig['diff']:+.1f} [{100 * sig['ci95'][0]:+.1f}, {100 * sig['ci95'][1]:+.1f}] "
                     f"p={sig['p_value']:.3f}",
                     None if ms_per_query is None else round(ms_per_query, 2)])

    add("baseline (CLIP B/32, exact)", "-", base_val, base_test, None)
    recall_c = float((base_test.first_rank <= C).mean())

    for model, label in ((FIRST, "PRF / AQE in CLIP space"),
                         ("dinov2-small", "visual PRF (DINOv2-S)"),
                         ("dinov2-base", "visual PRF (DINOv2-B)")):
        (m, lam, wt), grid = _tune_prf(val, model)
        t = time.perf_counter()
        new_test = test.prf(model, m, lam, wt)
        ms = 1000 * (time.perf_counter() - t) / len(new_test)
        add(label, f"m={m}, λ={lam}, {wt}", val.evaluate(val.prf(model, m, lam, wt)),
            test.evaluate(new_test), ms)
        raw[label]["val_grid_MRR"] = grid

    if with_cross_encoder:
        from visualsearch.retrieval.cross_encoder import BlipITMReranker

        ce = BlipITMReranker(device=ctx.settings.device).load()
        t = time.perf_counter()
        itm_val = _cross_encoder_scores(ce, val, ce_depth, ctx.settings)
        itm_test = _cross_encoder_scores(ce, test, ce_depth, ctx.settings)
        ce_ms = 1000 * (time.perf_counter() - t) / (len(itm_val) + len(itm_test))
        gammas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        gscores = {g: summarize(val.evaluate(_combine_ce(val, itm_val, g, ce_depth)), ci=False)["MRR"]
                   for g in gammas}
        g_best = max(gscores, key=gscores.get)
        add("BLIP ITM cross-encoder (ITM only)", f"top-{ce_depth}, γ=1.0",
            val.evaluate(_combine_ce(val, itm_val, 1.0, ce_depth)),
            test.evaluate(_combine_ce(test, itm_test, 1.0, ce_depth)), ce_ms)
        add("BLIP ITM cross-encoder (mixed)", f"top-{ce_depth}, γ={g_best}",
            val.evaluate(_combine_ce(val, itm_val, g_best, ce_depth)),
            test.evaluate(_combine_ce(test, itm_test, g_best, ce_depth)), ce_ms)
        raw["BLIP ITM cross-encoder (mixed)"]["val_gamma_MRR"] = gscores
        raw["cross_encoder_note"] = (
            "ms/query measured as a batch job with per-image feature caching; an interactive "
            "query pays one vision forward per uncached candidate.")

    # image->image average query expansion on Imagewoof (tune on train, test on val)
    aqe_rows = []
    for model in ("dinov2-small", "clip-vit-b32"):
        tr = tasks.i2i_category(ctx, model, "train")
        ev = tasks.i2i_category(ctx, model, "val")
        best, best_v = None, -1.0
        for m, lam in itertools.product([1, 3, 5, 10], [0.2, 0.4, 0.6]):
            v = _aqe_map(ctx, model, "train", tr, m, lam)
            if v > best_v:
                best, best_v = (m, lam), v
        mval = _aqe_map(ctx, model, "val", ev, *best)
        aqe_rows.append([model, f"m={best[0]}, λ={best[1]}", tr.metrics["mAP"], best_v,
                         ev.metrics["mAP"], mval])
        raw[f"aqe/{model}"] = {"params": best, "train_base": tr.metrics["mAP"], "train": best_v,
                               "val_base": ev.metrics["mAP"], "val": mval}

    report = {
        "title": "Reranking",
        "description": (
            f"First stage: exact CLIP ViT-B/32 search, C={C} candidates (test Recall@{C} = "
            f"{100 * recall_c:.1f}%, the ceiling for any reranker). Hyper-parameters selected by MRR "
            f"on val; test evaluated once. Δ R@1 = paired clustered bootstrap vs. baseline "
            f"[95% CI], p-value. The cross-encoder rescores the top {ce_depth} only. "
            "Latency is the extra time per query for the second stage."),
        "tables": [
            table("Text→image reranking (Flickr30k, 1K gallery)",
                  ["Method", "Selected on val", "val R@1", "test R@1", "test R@5", "test R@10",
                   "test MRR", "Δ test R@1 vs baseline", "+ms/query"], rows,
                  highlight_best=[3, 4, 5, 6]),
            table("Image→image average query expansion (Imagewoof mAP; tuned on train)",
                  ["Model", "Selected", "train base", "train AQE", "val base", "val AQE"],
                  aqe_rows),
        ],
    }
    save_result(ctx.settings, "rerank", {"results": raw, "first_stage_recall_at_C": recall_c},
                report)
    return raw


def _aqe_map(ctx: EvalContext, model: str, split: str, base: tasks.TaskResult, m: int,
             lam: float, C: int = 200) -> float:
    """mAP after PRF reranking of the top-C neighbours (tail order unchanged)."""
    emb = ctx.embedder(model)
    recs = ctx.records("imagewoof", [split])
    v = ctx.image_vectors(emb, recs)
    pos = {r.id: n for n, r in enumerate(recs)}
    labels = np.array([r.label for r in recs])
    q_vecs = {int(r.id): v[n] for n, r in enumerate(recs)}
    aps = []
    for qid, ranked in zip(base.query_ids, base.ranked):
        head = ranked[:C]
        hv = v[[pos[int(i)] for i in head]]
        first = hv @ q_vecs[int(qid)]
        new = prf_scores(first, hv, m, lam)
        order = np.concatenate([head[np.argsort(-new, kind="stable")], ranked[C:]])
        rel = labels[[pos[int(i)] for i in order]] == labels[pos[int(qid)]]
        hits = np.cumsum(rel)
        aps.append(float((hits[rel] / (np.nonzero(rel)[0] + 1)).sum() / max(rel.sum(), 1)))
    return float(np.mean(aps))
