import numpy as np
import pytest

from visualsearch.retrieval.calibration import (
    QUANTILE_LEVELS,
    ScoreCalibration,
    expected_calibration_error,
    fit_background,
    fit_platt,
    pairs_from_matrix,
)
from visualsearch.retrieval.fusion import fuse, minmax, ranks_desc
from visualsearch.retrieval.rerank import mix, prf_scores


def test_platt_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(6 * x - 1)))).astype(float)
    a, b = fit_platt(x, y)
    assert a == pytest.approx(6, rel=0.1) and b == pytest.approx(-1, abs=0.15)
    p = 1 / (1 + np.exp(-(a * x + b)))
    assert expected_calibration_error(p, y) < 0.02


def test_weighted_negatives_preserve_base_rate():
    sims = np.random.default_rng(0).random((20, 500))
    pos = np.zeros_like(sims, dtype=bool)
    pos[np.arange(20), np.arange(20)] = True
    _s, y, w = pairs_from_matrix(sims, pos, n_neg_per_query=50, seed=0)
    assert np.average(y, weights=w) == pytest.approx(1 / 500, rel=1e-6)


def test_background_and_percentile():
    s = np.random.default_rng(0).standard_normal(1_000_000)
    mean, std, q = fit_background(s)
    c = ScoreCalibration("m", "fp", "text", mean, std, q, levels=QUANTILE_LEVELS)
    assert c.percentile(0.0) == pytest.approx(0.5, abs=0.01)
    assert c.z(mean) == pytest.approx(0)
    # tail resolution: z=3 vs z=3.5 must be distinguishable (0.99865 vs 0.99977)
    assert c.percentile(3.0) == pytest.approx(0.99865, abs=2e-4)
    assert c.percentile(3.5) == pytest.approx(0.99977, abs=1e-4)


def test_minmax_and_ranks():
    assert minmax(np.array([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert ranks_desc(np.array([0.1, 0.9, 0.5])).tolist() == [3, 1, 2]


def test_fuse_weights_and_rrf():
    a = np.array([1.0, 0.0, 0.5])
    b = np.array([0.0, 1.0, 0.5])
    assert np.argmax(fuse([a, b], [1, 0], "minmax")) == 0
    assert np.argmax(fuse([a, b], [0, 1], "minmax")) == 1
    r = fuse([a, b], [0.5, 0.5], "rrf")
    assert r[0] == pytest.approx(r[1])
    with pytest.raises(ValueError):
        fuse([a], [1], "nonsense")


def test_zscore_fusion_uses_calibration_scale():
    # Text scores live around 0.3, image scores around 0.8: raw addition is dominated by
    # the image signal, z-scoring with each background puts them on equal footing.
    ct = ScoreCalibration("t", "f", "text", 0.2, 0.02, list(np.linspace(0, 1, 101)))
    ci = ScoreCalibration("i", "f", "image", 0.6, 0.1, list(np.linspace(0, 1, 101)))
    st = np.array([0.30, 0.22])   # candidate 0 is a strong text match (z=5 vs 1)
    si = np.array([0.70, 0.80])   # candidate 1 is a better image match (z=1 vs 2)
    fused = fuse([st, si], [0.5, 0.5], "zscore", [ct, ci])
    assert np.argmax(fused) == 0
    assert np.argmax(st + si) == 1  # naive sum picks the other one


def test_prf_and_mix():
    first = np.array([0.9, 0.8, 0.1, 0.05])
    vecs = np.eye(4, dtype=np.float32)
    vecs[3] = vecs[0]  # candidate 3 looks exactly like the top result
    out = prf_scores(first, vecs, m=1, lam=0.9)
    assert np.argsort(-out)[:2].tolist() in ([0, 3], [3, 0])
    assert np.allclose(prf_scores(first, vecs, m=1, lam=0.0), first)
    assert np.argmax(mix(first, np.array([0, 0, 1.0, 0]), 1.0)) == 2


def test_registry_skips_bad_files_and_unknown_fields(tmp_path):
    from visualsearch.retrieval.calibration import CalibrationRegistry

    good = ScoreCalibration("m", "fp", "text", 0.1, 0.2, [0.0, 1.0], levels=[0.0, 1.0])
    good.save(tmp_path)
    p = tmp_path / "m__text.json"
    import json

    p.write_text(json.dumps({**json.loads(p.read_text()), "added_in_a_later_version": 1}))
    (tmp_path / "broken.json").write_text("{not json")
    reg = CalibrationRegistry(tmp_path)
    assert reg.get("m", "text", "fp").mean == 0.1
