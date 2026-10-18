"""Fit score calibrations on validation data; report diagnostics on test data.

text queries (CLIP spaces)
    background: all (val caption, val image) pairs of Flickr30k
    Platt:      relevant = annotated caption/image pair (Flickr30k val)
    diagnostics on Flickr30k test

image queries (every space)
    background: all distinct (val image, val image) pairs of Flickr30k
    Platt:      relevant = same Imagewoof class, on the Imagewoof *train* subset
    diagnostics on Imagewoof val
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visualsearch.embeddings import IMAGE, TEXT
from visualsearch.eval.context import EvalContext
from visualsearch.retrieval.calibration import (
    ScoreCalibration,
    brier,
    expected_calibration_error,
    fit_background,
    fit_platt,
    pairs_from_matrix,
)


def _text_matrix(ctx: EvalContext, model: str, split: str):
    emb = ctx.embedder(model)
    recs = ctx.records("flickr30k", [split])
    iv = ctx.image_vectors(emb, recs)
    texts, targets = ctx.flickr_queries(split)
    tv = ctx.text_vectors(emb, texts)
    ids = np.array([r.id for r in recs])
    pos = targets[:, None] == ids[None, :]
    return tv @ iv.T, pos


def _woof_matrix(ctx: EvalContext, model: str, split: str):
    emb = ctx.embedder(model)
    recs = ctx.records("imagewoof", [split])
    v = ctx.image_vectors(emb, recs)
    labels = np.array([r.label for r in recs])
    sims = v @ v.T
    pos = labels[:, None] == labels[None, :]
    keep = ~np.eye(len(recs), dtype=bool)
    # Drop self-pairs by masking them out of both positives and the matrix rows.
    rows_s = [s[k] for s, k in zip(sims, keep)]
    rows_p = [p[k] for p, k in zip(pos, keep)]
    return np.stack(rows_s), np.stack(rows_p)


def _diag(calib: ScoreCalibration, sims: np.ndarray, pos: np.ndarray, seed: int) -> dict[str, Any]:
    s, y, w = pairs_from_matrix(sims, pos, n_neg_per_query=200, seed=seed + 1)
    p = calib.prob(s)
    return {"ece": expected_calibration_error(p, y, w), "brier": brier(p, y, w),
            "base_rate": float(np.average(y, weights=w))}


def fit_text(ctx: EvalContext, model: str) -> ScoreCalibration:
    seed = ctx.settings.seed
    sims, pos = _text_matrix(ctx, model, "val")
    mean, std, q = fit_background(sims)
    s, y, w = pairs_from_matrix(sims, pos, n_neg_per_query=200, seed=seed)
    a, b = fit_platt(s, y, w)
    emb = ctx.embedder(model)
    c = ScoreCalibration(model, emb.fingerprint, TEXT, mean, std, q, a, b,
                         fitted_on="flickr30k/val caption->image")
    c.diagnostics = {"val": _diag(c, sims, pos, seed),
                     "test": _diag(c, *_text_matrix(ctx, model, "test"), seed),
                     "positive_mean": float(sims[pos].mean())}
    return c


def fit_image(ctx: EvalContext, model: str) -> ScoreCalibration:
    seed = ctx.settings.seed
    emb = ctx.embedder(model)
    recs = ctx.records("flickr30k", ["val"])
    v = ctx.image_vectors(emb, recs)
    sims = v @ v.T
    mean, std, q = fit_background(sims[~np.eye(len(v), dtype=bool)])
    tr_s, tr_p = _woof_matrix(ctx, model, "train")
    s, y, w = pairs_from_matrix(tr_s, tr_p, n_neg_per_query=200, seed=seed)
    a, b = fit_platt(s, y, w)
    c = ScoreCalibration(model, emb.fingerprint, IMAGE, mean, std, q, a, b,
                         fitted_on="background: flickr30k/val image pairs; "
                                   "platt: imagewoof/train same-class pairs")
    c.diagnostics = {"train": _diag(c, tr_s, tr_p, seed),
                     "test_imagewoof_val": _diag(c, *_woof_matrix(ctx, model, "val"), seed)}
    return c
