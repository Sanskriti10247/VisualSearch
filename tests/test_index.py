import numpy as np
import pytest

from visualsearch.eval.metrics import recall_vs_exact
from visualsearch.index.faiss_index import FaissIndex

KINDS = ["flat", "hnsw", "ivf_flat", "ivf_pq"]


def brute_force(q, ids, x, k):
    s = q @ x.T
    order = np.argsort(-s, axis=1)[:, :k]
    return [ids[o] for o in order]


@pytest.fixture
def data():
    """Clustered data (like real embeddings); unstructured Gaussian noise is a
    pathological case for IVF and not representative."""
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((40, 32))
    x = centers[rng.integers(0, 40, 3000)] + 0.35 * rng.standard_normal((3000, 32))
    q = centers[rng.integers(0, 40, 50)] + 0.35 * rng.standard_normal((50, 32))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    ids = np.arange(1000, 4000, dtype=np.int64)  # non-contiguous external ids
    return x, q, ids


def test_flat_is_exact(data):
    x, q, ids = data
    idx = FaissIndex(32, "flat")
    idx.add(ids, x)
    res = idx.search(q, 10)
    exact = brute_force(q, ids, x, 10)
    for r, e in zip(res, exact):
        assert r.ids.tolist() == e.tolist()
        assert np.all(np.diff(r.scores) <= 1e-6)  # sorted descending


@pytest.mark.parametrize("kind,min_recall", [("hnsw", 0.95), ("ivf_flat", 0.9), ("ivf_pq", 0.6)])
# ivf_pq auto-config on 32-d / 3k vectors is 2 sub-quantisers x 6 bits = 12 bits per
# vector (plus exact refine of 4k candidates), so ~0.7 recall is the expected regime.
def test_ann_recall_reasonable(data, kind, min_recall):
    x, q, ids = data
    params = {"nprobe": 8} if kind.startswith("ivf") else {}
    idx = FaissIndex(32, kind, params)
    idx.add(ids, x)
    rec = recall_vs_exact([r.ids for r in idx.search(q, 10)], brute_force(q, ids, x, 10), 10)
    assert rec >= min_recall


def test_more_probes_never_hurt_ivf(unit_vectors):
    # Unstructured data: neighbours straddle cells, so nprobe actually matters.
    x, q = unit_vectors(3000, 32, seed=1), unit_vectors(50, 32, seed=2)
    ids = np.arange(3000, dtype=np.int64)
    exact = brute_force(q, ids, x, 10)
    recalls = []
    for nprobe in (1, 4, 32):
        idx = FaissIndex(32, "ivf_flat", {"nprobe": nprobe, "nlist": 32})
        idx.add(ids, x)
        recalls.append(recall_vs_exact([r.ids for r in idx.search(q, 10)], exact, 10))
    assert recalls == sorted(recalls) and recalls[-1] > recalls[0]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("subset", [3, 400])  # below / above the brute-force threshold
def test_filtered_search_respects_allowlist(data, kind, subset):
    x, q, ids = data
    idx = FaissIndex(32, kind, brute_force_threshold=50)
    idx.add(ids, x)
    allowed = ids[::7][:subset]
    res = idx.search(q, 10, allowed=allowed)
    for r in res:
        assert set(r.ids.tolist()) <= set(allowed.tolist())
        assert len(r.ids) == min(10, subset) or kind != "flat"
    assert res[0].strategy == ("exact-subset" if subset <= 50 else "filtered-index")
    if kind == "flat" or subset <= 50:  # exact paths must equal brute force over the subset
        sub = np.isin(ids, allowed)
        exact = brute_force(q, ids[sub], x[sub], min(10, subset))
        assert all(r.ids.tolist() == e.tolist() for r, e in zip(res, exact))


@pytest.mark.parametrize("kind", KINDS)
def test_remove_and_readd(data, kind):
    x, _q, ids = data
    idx = FaissIndex(32, kind)
    idx.add(ids, x)
    assert ids[0] in idx.search(x[:1], 10)[0].ids
    assert idx.remove(np.array([ids[0]])) == 1
    assert idx.remove(np.array([ids[0]])) == 0  # idempotent
    assert not idx.has(ids[0]) and len(idx) == len(ids) - 1
    assert ids[0] not in idx.search(x[:1], 50)[0].ids  # tombstoned ids never surface
    assert ids[0] not in idx.search(x[:1], 50, allowed=ids[:100])[0].ids
    idx.add(np.array([ids[0]]), x[:1])  # re-add brings it back
    assert idx.has(ids[0]) and ids[0] in idx.search(x[:1], 10)[0].ids


def test_readd_replaces_vector(data):
    x, _q, ids = data
    idx = FaissIndex(32, "flat")
    idx.add(ids[:10], x[:10])
    idx.add(ids[:1], x[20:21])  # same id, new vector
    assert len(idx) == 10
    assert np.allclose(idx.vectors_for(ids[:1]), x[20:21])


@pytest.mark.parametrize("kind", KINDS)
def test_save_load_roundtrip(tmp_path, data, kind):
    x, q, ids = data
    idx = FaissIndex(32, kind)
    idx.add(ids, x)
    idx.remove(ids[:5])
    idx.save(tmp_path / kind)
    loaded = FaissIndex.load(tmp_path / kind, 32, {})
    a, b = idx.search(q, 10), loaded.search(q, 10)
    assert all(np.array_equal(r1.ids, r2.ids) for r1, r2 in zip(a, b))
    assert len(loaded) == len(ids) - 5


def test_rejects_unnormalised_and_wrong_dim(unit_vectors):
    idx = FaissIndex(8, "flat")
    with pytest.raises(ValueError, match="normalised"):
        idx.add(np.array([1]), np.ones((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        idx.add(np.array([1]), unit_vectors(1, 4))


def test_pq_refine_improves_recall(data):
    x, q, ids = data
    exact = brute_force(q, ids, x, 10)
    rec = {}
    for refine in (0, 8):
        idx = FaissIndex(32, "ivf_pq", {"refine": refine, "nprobe": 16, "m": 4})
        idx.add(ids, x)
        rec[refine] = recall_vs_exact([r.ids for r in idx.search(q, 10)], exact, 10)
    assert rec[8] > rec[0]


def test_empty_index_search(unit_vectors):
    idx = FaissIndex(8, "hnsw")
    assert len(idx.search(unit_vectors(2, 8), 5)[0].ids) == 0


def test_qdrant_backend_matches_flat(data):
    pytest.importorskip("qdrant_client")
    from visualsearch.index.qdrant_index import QdrantIndex

    x, q, ids = data
    qi = QdrantIndex(32)
    qi.add(ids[:500], x[:500])
    exact = brute_force(q, ids[:500], x[:500], 5)
    res = qi.search(q, 5)
    assert recall_vs_exact([r.ids for r in res], exact, 5) == 1.0
    allowed = ids[:500:3]
    assert all(set(r.ids) <= set(allowed) for r in qi.search(q, 5, allowed=allowed))
    qi.remove(ids[:1])
    assert ids[0] not in qi.search(x[:1], 5)[0].ids
