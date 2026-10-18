import numpy as np
import pytest

from visualsearch.eval.metrics import (
    bootstrap_ci,
    paired_bootstrap,
    per_query_from_ranks,
    ranks_from_lists,
    ranks_from_scores,
    recall_vs_exact,
    summarize,
)


def test_ranks_from_lists_and_truncation():
    ranked = [np.array([5, 3, 9]), np.array([1, 2, 3])]
    ranks = ranks_from_lists(ranked, [[3], [7]])
    assert ranks[0].tolist() == [2.0]
    assert np.isinf(ranks[1][0])  # relevant item missing from the truncated list


def test_ranks_from_scores_matches_sorting():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]])
    assert ranks_from_scores(scores, [[2]])[0].tolist() == [2]
    assert ranks_from_scores(scores, [[1, 0]])[0].tolist() == [1, 4]


def test_recall_mrr_by_hand():
    # first relevant at ranks 1, 3, 11, never
    pq = per_query_from_ranks([np.array([1.0]), np.array([3.0]), np.array([11.0]),
                               np.array([np.inf])])
    m = summarize(pq, ks=(1, 5, 10), with_map=False, ci=False)
    assert m["R@1"] == pytest.approx(0.25)
    assert m["R@5"] == pytest.approx(0.5)
    assert m["R@10"] == pytest.approx(0.5)
    assert m["MRR"] == pytest.approx((1 + 1 / 3 + 1 / 11 + 0) / 4)
    assert m["median_rank"] is None  # undefined when some query never retrieves its target


def test_average_precision_by_hand():
    # relevant at ranks 1, 3, 6 out of R=3 -> AP = (1/1 + 2/3 + 3/6) / 3
    pq = per_query_from_ranks([np.array([3.0, 1.0, 6.0])])
    assert pq.ap[0] == pytest.approx((1 + 2 / 3 + 0.5) / 3)
    # one of three relevant items never retrieved -> still divided by R=3
    pq = per_query_from_ranks([np.array([1.0, 2.0, np.inf])])
    assert pq.ap[0] == pytest.approx((1 + 1) / 3)


def test_single_relevant_map_equals_mrr():
    pq = per_query_from_ranks([np.array([r]) for r in (1.0, 2.0, 5.0)])
    m = summarize(pq, ci=False)
    assert m["mAP"] == pytest.approx(m["MRR"])


def test_bootstrap_ci_contains_mean_and_clusters():
    v = np.random.default_rng(0).random(500)
    lo, hi = bootstrap_ci(v, n_boot=500)
    assert lo < v.mean() < hi
    clusters = np.repeat(np.arange(100), 5)
    lo_c, hi_c = bootstrap_ci(v, clusters, n_boot=500)
    assert lo_c < v.mean() < hi_c


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.random(1000)
    same = paired_bootstrap(a, a.copy())
    assert same["diff"] == 0 and same["p_value"] == 1.0
    better = paired_bootstrap(a, a + 0.05)
    assert better["diff"] == pytest.approx(0.05)
    assert better["p_value"] < 0.01
    assert better["ci95"][0] > 0


def test_recall_vs_exact():
    assert recall_vs_exact([np.array([1, 2, 3])], [np.array([1, 2, 4])], k=3) == pytest.approx(2 / 3)
