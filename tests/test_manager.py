import json

import numpy as np

from tests.conftest import make_image
from visualsearch.data.catalog import Catalog
from visualsearch.index.manager import IndexManager
from visualsearch.pipeline.ingest import Ingestor


def test_build_sync_and_persistence(settings, catalog, populated):
    mgr = IndexManager(settings, catalog)
    mgr.open(build_missing=True)
    for key in ("fake-clip", "fake-dino"):
        assert len(mgr.runtime(key).index) == 24
        manifest = json.loads((mgr.index_dir(key) / "manifest.json").read_text())
        assert manifest["embedding_fingerprint"] == mgr.runtime(key).embedder.fingerprint
        assert manifest["ntotal"] == 24

    # incremental: add 2 images, sync embeds only those
    ing = Ingestor(settings, catalog)
    new = [ing.ingest_bytes(make_image((10, 10, 10), seed=100 + i), collection="new").image_id
           for i in range(2)]
    cached_before = len(mgr.runtime("fake-clip").image_store)
    out = mgr.sync("fake-clip")
    assert out["added"] == 2 and len(mgr.runtime("fake-clip").index) == 26
    assert len(mgr.runtime("fake-clip").image_store) == cached_before + 2

    # deletion: tombstoned in catalogue -> removed from index on sync
    catalog.mark_deleted(new[0])
    assert mgr.sync("fake-clip")["removed"] == 1
    assert not mgr.runtime("fake-clip").index.has(new[0])

    # a fresh manager loads from disk with identical contents
    mgr2 = IndexManager(settings, Catalog(settings.catalog_path))
    mgr2.open(build_missing=False)
    assert set(mgr2.runtime("fake-clip").index.ids()) == set(mgr.runtime("fake-clip").index.ids())


def test_stale_index_is_rebuilt(settings, catalog, populated):
    mgr = IndexManager(settings, catalog)
    mgr.open()
    old_fp = mgr.runtime("fake-clip").embedder.fingerprint
    # change the model config -> new fingerprint -> stale manifest
    settings.models_path.write_text(settings.models_path.read_text().replace("seed: 1", "seed: 9"))
    mgr2 = IndexManager(settings, catalog)
    new_fp = mgr2.runtime("fake-clip").embedder.fingerprint
    assert new_fp != old_fp
    mgr2.open(build_missing=True)
    manifest = json.loads((mgr2.index_dir("fake-clip") / "manifest.json").read_text())
    assert manifest["embedding_fingerprint"] == new_fp
    assert len(mgr2.runtime("fake-clip").index) == 24


def test_index_vectors_match_cache(settings, catalog, populated):
    mgr = IndexManager(settings, catalog)
    mgr.open()
    rt = mgr.runtime("fake-dino")
    rec = catalog.get(populated[("blue", 1)])
    assert np.allclose(rt.index.vectors_for(np.array([rec.id]))[0], rt.image_store.get([rec.sha256])[0])
