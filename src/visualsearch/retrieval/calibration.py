"""Score calibration.

Raw cosine similarities are not comparable across embedding spaces or query
modalities: CLIP text->image scores typically live in [0.15, 0.40], while
image->image scores in CLIP or DINOv2 space are much higher and wider. They
cannot be added or thresholded directly.

For every (embedding space, query modality) pair we fit, on the **validation**
split only:

* a *background* distribution: scores of random (query, image) pairs,
  summarised by mean/std and 101 quantiles. Used for z-score fusion and to
  express a score as a percentile ("better than 99.6% of random images");
* a Platt scaler P(relevant | score) = sigmoid(a * score + b) fit on labelled
  pairs. Relevance means "annotated match" (Flickr30k caption/image pairs, or
  same Imagewoof class for image->image). Because each Flickr30k caption has a
  single annotated image, probabilities systematically *underestimate*
  relevance for generic queries; the UI says so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ScoreCalibration:
    space: str                 # model key
    fingerprint: str
    query_modality: str        # text | image
    mean: float
    std: float
    quantiles: list[float]     # background score at each level in `levels`
    platt_a: float = 0.0
    platt_b: float = 0.0
    fitted_on: str = ""
    diagnostics: dict = field(default_factory=dict)
    levels: list[float] = field(default_factory=lambda: [i / 100 for i in range(101)])

    def z(self, s: np.ndarray) -> np.ndarray:
        return (np.asarray(s, dtype=np.float64) - self.mean) / max(self.std, 1e-6)

    def percentile(self, s: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s, dtype=np.float64), self.quantiles, self.levels)

    def prob(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.platt_a * np.asarray(s, dtype=np.float64) + self.platt_b)))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.space}__{self.query_modality}.json"
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: Path) -> ScoreCalibration:
        # Ignore fields written by newer versions so old servers keep starting.
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Dense in the upper tail, where search results live (top 1% down to 1e-5).
QUANTILE_LEVELS = sorted(set(np.round(np.concatenate([
    np.linspace(0, 0.99, 100), 1 - np.logspace(-2, -5, 31), [1.0]]), 7).tolist()))


def fit_background(scores: np.ndarray) -> tuple[float, float, list[float]]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    return float(s.mean()), float(s.std()), [float(x) for x in np.quantile(s, QUANTILE_LEVELS)]


def fit_platt(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None,
              iters: int = 100) -> tuple[float, float]:
    """Weighted 1-D logistic regression by Newton's method (Platt scaling)."""
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    a, b = 0.0, float(np.log((w @ y + 1) / (w @ (1 - y) + 1)))
    X = np.stack([x, np.ones_like(x)], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = X.T @ (w * (p - y))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               weights: np.ndarray | None = None, bins: int = 15) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, total = 0.0, w.sum()
    for b in range(bins):
        m = idx == b
        if w[m].sum() > 0:
            ece += w[m].sum() / total * abs(np.average(p[m], weights=w[m]) -
                                             np.average(y[m], weights=w[m]))
    return float(ece)


def brier(probs: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> float:
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    return float(np.average((np.asarray(probs) - np.asarray(labels)) ** 2, weights=w))


def pairs_from_matrix(sims: np.ndarray, positive: np.ndarray, n_neg_per_query: int, seed: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (score, label, weight) pairs from a query x gallery similarity
    matrix. All positives are kept; negatives are subsampled and up-weighted so
    the weighted base rate equals the true one."""
    rng = np.random.default_rng(seed)
    scores, labels, weights = [], [], []
    for row, pos in zip(sims, positive):
        neg_idx = np.nonzero(~pos)[0]
        take = rng.choice(neg_idx, size=min(n_neg_per_query, len(neg_idx)), replace=False)
        scores += [row[pos], row[take]]
        labels += [np.ones(pos.sum()), np.zeros(len(take))]
        weights += [np.ones(pos.sum()), np.full(len(take), len(neg_idx) / max(len(take), 1))]
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(weights)


class CalibrationRegistry:
    """Loads calibrations for the current fingerprints; ignores stale ones."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._items: dict[tuple[str, str], ScoreCalibration] = {}
        if self.directory.exists():
            for p in self.directory.glob("*.json"):
                try:
                    c = ScoreCalibration.load(p)
                except (OSError, ValueError, TypeError) as e:
                    log.warning("ignoring unreadable calibration %s: %s", p, e)
                    continue
                self._items[(c.space, c.query_modality)] = c

    def get(self, space: str, query_modality: str, fingerprint: str | None = None
            ) -> ScoreCalibration | None:
        c = self._items.get((space, query_modality))
        if c is None or (fingerprint and c.fingerprint != fingerprint):
            return None
        return c

    def put(self, c: ScoreCalibration) -> None:
        self._items[(c.space, c.query_modality)] = c
        c.save(self.directory)

    def describe(self) -> list[dict]:
        return [{"space": c.space, "query_modality": c.query_modality, "mean": c.mean,
                 "std": c.std, "fitted_on": c.fitted_on, "diagnostics": c.diagnostics}
                for c in self._items.values()]
