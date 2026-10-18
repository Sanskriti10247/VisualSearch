"""Combining retrieval signals from different queries and embedding spaces.

Vectors from different spaces are never mixed. Signals are combined *after*
scoring, on a common candidate set, with one of:

  zscore   w_t * z(text score) + w_i * z(image score), where z uses the
           background mean/std fitted on the validation split
           (:mod:`retrieval.calibration`). Falls back to per-query min-max
           normalisation when no calibration is available.
  minmax   per-query min-max normalisation over the candidate set.
  rrf      reciprocal rank fusion, sum_s w_s / (60 + rank_s). Parameter-free
           and scale-free; a robust baseline for the score-based methods.

``early`` fusion (averaging the text and image *query vectors*) is only valid
when both queries live in the same space (CLIP text + CLIP image). It is
implemented in the engine and evaluated as an ablation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from visualsearch.retrieval.calibration import ScoreCalibration

RRF_K = 60


def minmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    lo, hi = s.min(), s.max()
    return np.zeros_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def normalize(s: np.ndarray, calib: ScoreCalibration | None, method: str) -> np.ndarray:
    if method == "zscore" and calib is not None:
        return calib.z(s)
    return minmax(s)


def ranks_desc(s: np.ndarray) -> np.ndarray:
    """1-based ranks, highest score first."""
    order = np.argsort(-np.asarray(s), kind="stable")
    r = np.empty(len(s), dtype=np.int64)
    r[order] = np.arange(1, len(s) + 1)
    return r


def fuse(scores: Sequence[np.ndarray], weights: Sequence[float], method: str = "zscore",
         calibs: Sequence[ScoreCalibration | None] | None = None) -> np.ndarray:
    """Fuse aligned score arrays (same candidates, same order) into one score."""
    if not scores:
        raise ValueError("no signals to fuse")
    calibs = calibs or [None] * len(scores)
    w = np.asarray(weights, dtype=np.float64)
    if method == "rrf":
        return sum(wi / (RRF_K + ranks_desc(s)) for wi, s in zip(w, scores))
    if method not in ("zscore", "minmax"):
        raise ValueError(f"unknown fusion method '{method}'")
    return sum(wi * normalize(s, c, method) for wi, s, c in zip(w, scores, calibs))
