"""Second-stage rerankers over a first-stage candidate list.

All rerankers take the candidate list produced by the first stage (ids, raw
scores) and return new scores for the *same* candidates, so they can only
reorder, never recall something the first stage missed. Their benefit is
therefore bounded by first-stage Recall@C (C = number of candidates).

prf (pseudo-relevance feedback / average query expansion)
    Treat the top-m candidates as relevant, average their vectors in some
    embedding space S into a feedback query, score all candidates against it
    and mix with the first-stage score:
        final = (1 - lam) * z(first) + lam * z(feedback)
    with z = per-query standardisation over the candidates. S can be the
    first-stage space (classic AQE) or a different one: e.g. DINOv2 feedback on
    CLIP text->image candidates ("visual consensus"), which uses DINOv2's
    stronger visual similarity without ever comparing text to DINOv2 vectors.

cross_encoder
    A BLIP image-text matching head scores each (text, image) pair jointly
    with cross-attention; see :mod:`retrieval.cross_encoder`.
"""

from __future__ import annotations

import numpy as np


def zstd(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    sd = s.std()
    return np.zeros_like(s) if sd < 1e-12 else (s - s.mean()) / sd


def prf_scores(first_scores: np.ndarray, cand_vecs: np.ndarray, m: int, lam: float,
               weighting: str = "uniform") -> np.ndarray:
    """Pseudo-relevance-feedback reranking of one query's candidates.

    first_scores: [C] first-stage scores (any scale), candidates in any order.
    cand_vecs:    [C, d] L2-normalised vectors of the candidates in space S.
    """
    if lam <= 0 or len(first_scores) <= 1:
        return np.asarray(first_scores, dtype=np.float64)
    order = np.argsort(-first_scores)
    top = order[:max(1, min(m, len(order)))]
    if weighting == "softmax":
        w = np.exp(zstd(first_scores)[top])
    else:
        w = np.ones(len(top))
    fb = (w[:, None] * cand_vecs[top]).sum(0)
    fb /= max(np.linalg.norm(fb), 1e-12)
    feedback = cand_vecs @ fb
    return (1 - lam) * zstd(first_scores) + lam * zstd(feedback)


def mix(first_scores: np.ndarray, second_scores: np.ndarray, gamma: float) -> np.ndarray:
    """(1-gamma) * z(first) + gamma * z(second); gamma=1 ranks by the second stage alone."""
    return (1 - gamma) * zstd(first_scores) + gamma * zstd(second_scores)
