import json
import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_image
from visualsearch.api.app import create_app
from visualsearch.api.services import Services
from visualsearch.data.catalog import FilterSpec
from visualsearch.retrieval.engine import SearchError


@pytest.fixture
def svc(settings, populated):
    s = Services.build(settings)
    yield s
    s.close()


@pytest.fixture
def client(svc):
    with TestClient(create_app(svc=svc)) as c:
        yield c


# ---------------------------------------------------------------- engine
def test_text_search_returns_k_sorted(svc):
    out = svc.engine.text("a red picture", k=5)
    assert len(out.hits) == 5
    scores = [h.score for h in out.hits]
    assert scores == sorted(scores, reverse=True)
    assert set(out.timings_ms) >= {"embed_query", "search", "total"}


def test_image_search_by_id_excludes_query_and_finds_same_colour(svc, populated):
    qid = populated[("green", 0)]
    out = svc.engine.image(image_id=qid, k=3, model="fake-clip")
    ids = [h.image_id for h in out.hits]
    assert qid not in ids
    labels = {svc.catalog.get(i).label for i in ids}
    assert labels == {"green"}  # same-colour images are nearest in the fake space
    keep = svc.engine.image(image_id=qid, k=1, model="fake-clip", exclude_query=False)
    assert keep.hits[0].image_id == qid


def test_filters_respected_in_all_modes(svc, populated):
    f = FilterSpec(labels=["blue"], orientations=["portrait"])
    allowed = set(svc.catalog.resolve_filter(f).tolist())
    for out in (svc.engine.text("x", k=10, filters=f),
                svc.engine.image(image_id=populated[("red", 0)], k=10, filters=f),
                svc.engine.hybrid("x", image_id=populated[("red", 0)], k=10, filters=f)):
        assert {h.image_id for h in out.hits} <= allowed and out.hits


def test_text_on_image_only_model_is_rejected(svc):
    with pytest.raises(SearchError, match="image-only"):
        svc.engine.text("a dog", model="fake-dino")


def test_early_fusion_requires_same_space(svc, populated):
    with pytest.raises(SearchError, match="same"):
        svc.engine.hybrid("x", image_id=populated[("red", 0)], fusion="early",
                          text_model="fake-clip", image_model="fake-dino")
    out = svc.engine.hybrid("x", image_id=populated[("red", 0)], fusion="early",
                            text_model="fake-clip", image_model="fake-clip", k=4)
    assert len(out.hits) == 4


def test_hybrid_weight_extremes_match_single_signal(svc, populated):
    qid = populated[("yellow", 1)]
    img_only = [h.image_id for h in svc.engine.image(image_id=qid, k=5, model="fake-dino").hits]
    hyb = svc.engine.hybrid("unrelated words", image_id=qid, k=5, text_weight=0.0,
                            fusion="minmax", image_model="fake-dino")
    assert [h.image_id for h in hyb.hits] == img_only


def test_k_validation(svc):
    with pytest.raises(SearchError):
        svc.engine.text("x", k=0)
    with pytest.raises(SearchError):
        svc.engine.text("x", k=10_000)


# ------------------------------------------------------------------- API
def test_health_and_models(client):
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and h["catalog_images"] == 24
    m = client.get("/api/v1/models").json()
    assert {x["key"] for x in m["models"]} == {"fake-clip", "fake-dino"}


def test_text_search_schema(client):
    r = client.post("/api/v1/search/text", json={"query": "a red picture", "k": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 4 and len(body["results"]) == 4
    hit = body["results"][0]
    assert {"rank", "image_id", "score", "signals", "thumbnail_url", "metadata"} <= set(hit)
    assert hit["rank"] == 1 and "fake-clip:text" in hit["signals"]
    assert client.get(hit["thumbnail_url"]).headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": "x", "k": 0},
                                     {"query": "x", "unknown": 1},
                                     {"query": "x", "filters": {"orientations": ["diagonal"]}}])
def test_text_search_validation(client, payload):
    r = client.post("/api/v1/search/text", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_text_search_on_image_only_model(client):
    r = client.post("/api/v1/search/text", json={"query": "dog", "model": "fake-dino"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_search"


def test_image_upload_search(client):
    r = client.post("/api/v1/search/image", files={"file": ("q.png", make_image(fmt="PNG"), "image/png")},
                    data={"k": "3", "filters": json.dumps({"labels": ["red", "blue"]})})
    assert r.status_code == 200, r.text
    assert all(h["metadata"]["label"] in ("red", "blue") for h in r.json()["results"])


def test_image_upload_errors(client, settings):
    bad = client.post("/api/v1/search/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "invalid_image"
    big = client.post("/api/v1/search/image",
                      files={"file": ("big.bmp", b"0" * (2 * 1024 * 1024), "image/bmp")})
    assert big.status_code == 413
    both = client.post("/api/v1/search/image", files={"file": ("q.jpg", make_image(), "image/jpeg")},
                       data={"image_id": "1"})
    assert both.status_code == 422
    neither = client.post("/api/v1/search/image", data={"k": "3"})
    assert neither.status_code == 422
    filt = client.post("/api/v1/search/image", data={"image_id": "1", "filters": "{bad json"})
    assert filt.status_code == 422 and filt.json()["error"]["code"] == "invalid_filters"


def test_duplicate_upload_is_flagged(client, populated, svc):
    rec = svc.catalog.get(populated[("red", 0)])
    data = (svc.settings.images_dir / rec.path).read_bytes()
    r = client.post("/api/v1/search/image", files={"file": ("dup.jpg", data, "image/jpeg")})
    body = r.json()
    assert any("already in the collection" in w for w in body["warnings"])
    assert rec.id not in [h["image_id"] for h in body["results"]]


def test_hybrid_endpoint(client, populated):
    r = client.post("/api/v1/search/hybrid", data={"text": "red", "image_id": str(populated[("red", 1)]),
                                                   "k": "5", "fusion": "rrf", "text_weight": "0.3"})
    assert r.status_code == 200, r.text
    s = r.json()["strategy"]
    assert s["mode"] == "hybrid" and s["fusion"] == "rrf"
    bad = client.post("/api/v1/search/hybrid", data={"text": "red", "fusion": "magic"})
    assert bad.status_code == 422


def test_index_job_lifecycle(client, svc):
    files = [("files", (f"n{i}.jpg", make_image((5, 5, 5 + i * 40), seed=500 + i), "image/jpeg"))
             for i in range(3)]
    files.append(("files", ("bad.jpg", b"garbage", "image/jpeg")))
    r = client.post("/api/v1/index/images", files=files, data={"collection": "mine", "tags": "a, b"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert r.json()["accepted_files"] == 3
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    assert job["result"]["counts"]["added"] == 3
    assert svc.catalog.count() == 27
    for m in svc.manager.runtimes.values():
        assert len(m.index) == 27
    res = client.post("/api/v1/search/text", json={"query": "x", "k": 50,
                                                    "filters": {"collections": ["mine"], "tags_any": ["b"]}})
    assert len(res.json()["results"]) == 3


def test_index_rejects_all_invalid_and_too_many(client):
    r = client.post("/api/v1/index/images", files=[("files", ("b.jpg", b"garbage", "image/jpeg"))])
    assert r.status_code == 415
    many = [("files", (f"{i}.jpg", make_image(seed=i), "image/jpeg")) for i in range(6)]
    assert client.post("/api/v1/index/images", files=many).status_code == 413
    assert client.post("/api/v1/index/images", files=[("files", ("a.jpg", make_image(), "image/jpeg"))],
                       data={"collection": "../etc"}).status_code == 422


def test_image_detail_delete_and_404(client, populated):
    iid = populated[("purple", 2)]
    d = client.get(f"/api/v1/images/{iid}").json()
    assert d["metadata"]["label"] == "purple" and len(d["captions"]) == 2
    assert set(d["indexed_in"]) == {"fake-clip", "fake-dino"}
    assert client.delete(f"/api/v1/images/{iid}").status_code == 202
    assert client.get(f"/api/v1/images/{iid}").status_code == 404
    assert client.get("/api/v1/images/999999/thumbnail").status_code == 404


def test_metrics_endpoint(client):
    client.post("/api/v1/search/text", json={"query": "a"})
    m = client.get("/api/v1/metrics").json()
    assert "POST /api/v1/search/text" in m["routes"]
    assert m["routes"]["POST /api/v1/search/text"]["count"] >= 1
    prom = client.get("/api/v1/metrics?format=prometheus").text
    assert "visualsearch_requests_total" in prom


def test_facets(client):
    f = client.get("/api/v1/facets").json()
    assert f["total"] == 24 and any(x["value"] == "synthetic" for x in f["collections"])
