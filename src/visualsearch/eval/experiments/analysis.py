"""Calibration quality, abstention, and hard-negative / failure analysis."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from visualsearch.embeddings import TEXT
from visualsearch.eval import tasks
from visualsearch.eval.calibrate import fit_image, fit_text
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import summarize
from visualsearch.eval.results import chart, save_result, table
from visualsearch.retrieval.calibration import CalibrationRegistry

MODELS = ["clip-vit-b32", "clip-vit-b16", "clip-vit-l14", "dinov2-small", "dinov2-base"]


def calibration(ctx: EvalContext) -> dict[str, Any]:
    reg = CalibrationRegistry(ctx.settings.calibration_dir)
    rows, raw = [], {}
    for m in MODELS:
        emb = ctx.embedder(m)
        cals = [fit_image(ctx, m)] + ([fit_text(ctx, m)] if emb.supports(TEXT) else [])
        for c in cals:
            reg.put(c)
            d = c.diagnostics
            test = d.get("test") or d.get("test_imagewoof_val")
            fit = d.get("val") or d.get("train")
            raw[f"{m}/{c.query_modality}"] = {"mean": c.mean, "std": c.std, "platt": [c.platt_a, c.platt_b],
                                              "diagnostics": d, "fitted_on": c.fitted_on}
            rows.append([m, c.query_modality, round(c.mean, 4), round(c.std, 4),
                         round(c.quantiles[99], 4), fit["ece"], test["ece"], test["brier"],
                         test["base_rate"]])

    # Abstention: does calibrated confidence of the top-1 result predict correctness?
    c = reg.get("clip-vit-b32", TEXT)
    r = tasks.t2i(ctx, "clip-vit-b32", "test")
    emb = ctx.embedder("clip-vit-b32")
    recs = ctx.records("flickr30k", ["test"])
    pos = {rec.id: n for n, rec in enumerate(recs)}
    g = ctx.image_vectors(emb, recs)
    texts, targets = ctx.flickr_queries("test")
    q = ctx.text_vectors(emb, texts)
    top1 = np.array([rk[0] for rk in r.ranked])
    s1 = np.einsum("ij,ij->i", q, g[[pos[int(i)] for i in top1]])
    correct = top1 == targets
    probs = c.prob(s1)
    curve, arows = [], []
    for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        keep = probs >= thr
        cov, prec = float(keep.mean()), float(correct[keep].mean()) if keep.any() else None
        curve.append((cov, prec if prec is not None else 0.0))
        arows.append([thr, cov, prec])
    report = {
        "title": "Score calibration",
        "description": (
            "Background statistics and Platt scaling are fitted on validation data only; ECE/Brier "
            "on held-out data. Text: P(caption's annotated image) on Flickr30k. Image: P(same "
            "Imagewoof breed). Both use the true (heavily imbalanced) base rate via weighted "
            "negative sampling. The abstention table thresholds the calibrated P(match) of the "
            "top-1 text→image result on the test split."),
        "tables": [
            table("Calibration per embedding space and query modality",
                  ["Model", "Query", "bg mean", "bg std", "bg p99", "ECE (fit split)",
                   "ECE (held-out)", "Brier (held-out)", "base rate"], rows),
            table("Abstention: answer only if top-1 P(match) ≥ threshold (CLIP B/32, T→I test)",
                  ["Threshold", "Coverage", "Precision@1 on answered"], arows),
        ],
        "charts": [chart("Coverage vs. precision@1 of the top result", "line", "coverage",
                         "precision@1", {"CLIP B/32": sorted(curve)})],
    }
    save_result(ctx.settings, "calibration", {"results": raw, "abstention": arows}, report)
    return raw


CATEGORIES = {
    "counting": r"\b(two|three|four|five|six|seven|eight|nine|ten|several|couple|pair|many|"
                r"group|crowd|both|one)\b",
    "colour": r"\b(red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|"
                r"blond|blonde)\b",
    "spatial": r"\b(left|right|behind|under|underneath|above|below|beside|between|next to|"
                r"in front of|on top of|near|inside|outside)\b",
    "negation": r"\b(no|not|without|nobody|empty)\b",
    "action": r"\b\w+ing\b",
}


def _cats(text: str) -> list[str]:
    t = text.lower()
    return [c for c, rx in CATEGORIES.items() if re.search(rx, t)]


def hard_negatives(ctx: EvalContext, model: str = "clip-vit-b32") -> dict[str, Any]:
    r = tasks.t2i(ctx, model, "test")
    texts, targets = ctx.flickr_queries("test")
    emb = ctx.embedder(model)
    tv = ctx.text_vectors(emb, texts)
    caps_by_img: dict[int, list[int]] = {}
    for n, t in enumerate(targets):
        caps_by_img.setdefault(int(t), []).append(n)
    first = r.per_query.first_rank
    ok = first <= 1

    # 1) linguistic categories
    cat_rows = []
    lens = np.array([len(t.split()) for t in texts])
    for cat in CATEGORIES:
        mask = np.array([cat in _cats(t) for t in texts])
        if mask.sum() >= 30:
            m = summarize(r.per_query.subset(mask), with_map=False, ci=True)
            cat_rows.append([cat, int(mask.sum()), m["R@1"], m["ci95"]["R@1"], m["R@10"]])
    m_all = summarize(r.per_query, with_map=False)
    cat_rows.insert(0, ["all queries", len(texts), m_all["R@1"], m_all["ci95"]["R@1"], m_all["R@10"]])
    for lo, hi in ((0, 9), (9, 13), (13, 18), (18, 100)):
        mask = (lens >= lo) & (lens < hi)
        if mask.sum():
            m = summarize(r.per_query.subset(mask), with_map=False, ci=True)
            cat_rows.append([f"length {lo}-{hi - 1} words", int(mask.sum()), m["R@1"],
                             m["ci95"]["R@1"], m["R@10"]])

    # 2) are top-1 "errors" really wrong? Compare the query to the wrong image's own
    #    captions vs. to the target's other captions (CLIP text-text similarity).
    plausible, examples = 0, []
    fails = np.nonzero(~ok)[0]
    for qn in fails:
        wrong = int(r.ranked[qn][0])
        tgt = int(targets[qn])
        sim_wrong = float((tv[caps_by_img[wrong]] @ tv[qn]).max())
        others = [c for c in caps_by_img[tgt] if c != qn]
        sim_target = float((tv[others] @ tv[qn]).max())
        if sim_wrong >= sim_target:
            plausible += 1
        if len(examples) < 40:
            examples.append({"query": texts[qn], "target_id": tgt, "top1_id": wrong,
                             "target_rank": float(first[qn]),
                             "top1_captions": [texts[c] for c in caps_by_img[wrong]],
                             "query_vs_top1_captions": round(sim_wrong, 3),
                             "query_vs_target_other_captions": round(sim_target, 3)})
    near_miss = float(np.mean(first[~ok] <= 5)) if (~ok).any() else 0.0
    report = {
        "title": "Hard negatives and failure analysis (CLIP ViT-B/32, Flickr30k test T→I)",
        "description": (
            f"{len(fails)} of {len(texts)} queries miss at rank 1; {100 * near_miss:.1f}% of those "
            f"have the target in the top 5 (near misses). For {plausible} failures "
            f"({100 * plausible / max(len(fails), 1):.1f}%) the wrongly top-ranked image has a human "
            "caption at least as similar to the query as the target's own other captions — i.e. "
            "the 'hard negative' is plausibly also a correct answer that Flickr30k's one-image-per-"
            "caption labels count as wrong. Category R@1 uses keyword heuristics; categories "
            "overlap."),
        "tables": [
            table("R@1 by query type", ["Query category", "Queries", "R@1", "R@1 95% CI", "R@10"],
                  cat_rows),
            table("Example failures", ["Query", "Target rank", "Top-1 image's first caption",
                                       "sim(query, top-1 captions)",
                                       "sim(query, target's other captions)"],
                  [[e["query"], e["target_rank"], e["top1_captions"][0],
                    e["query_vs_top1_captions"], e["query_vs_target_other_captions"]]
                   for e in examples[:15]]),
        ],
    }
    out = {"n_fail": len(fails), "plausible_false_negatives": plausible,
           "near_miss_rate": near_miss, "examples": examples, "categories": cat_rows}
    save_result(ctx.settings, "hard_negatives", out, report)
    return out
