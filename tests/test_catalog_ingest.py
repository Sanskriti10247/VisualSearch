import numpy as np

from tests.conftest import make_image
from visualsearch.data.catalog import FilterSpec, orientation_of
from visualsearch.pipeline.ingest import Ingestor, IngestStatus, NearDupPolicy


def test_orientation():
    assert orientation_of(100, 50) == "landscape"
    assert orientation_of(50, 100) == "portrait"
    assert orientation_of(100, 98) == "square"


def test_ingest_stores_files_and_records(settings, catalog):
    ing = Ingestor(settings, catalog)
    r = ing.ingest_bytes(make_image(seed=1), collection="c", split="test", source_name="a.jpg",
                         tags=["x"], captions=[(0, "hello")])
    assert r.status is IngestStatus.ADDED
    rec = catalog.get(r.image_id)
    assert (settings.images_dir / rec.path).read_bytes() == make_image(seed=1)  # original bytes
    assert (settings.thumbs_dir / rec.thumb_path).exists()
    assert catalog.captions_for(r.image_id) == ["hello"]


def test_exact_duplicate_detected(settings, catalog):
    ing = Ingestor(settings, catalog)
    a = ing.ingest_bytes(make_image(seed=1), collection="c")
    b = ing.ingest_bytes(make_image(seed=1), collection="c")
    assert b.status is IngestStatus.DUPLICATE and b.image_id == a.image_id
    assert catalog.count() == 1


def test_near_duplicate_policy(settings, catalog):
    ing = Ingestor(settings, catalog)
    a = ing.ingest_bytes(make_image(seed=1, size=(128, 96)), collection="c")
    near = make_image(seed=1, size=(64, 48))  # resized copy: different bytes, same content
    kept = ing.ingest_bytes(near, collection="c", near_dup_policy=NearDupPolicy.KEEP)
    assert kept.status is IngestStatus.ADDED and kept.near_duplicates[0][0] == a.image_id
    assert catalog.get(kept.image_id).extra["near_duplicate_of"][0]["image_id"] == a.image_id
    skipped = ing.ingest_bytes(make_image(seed=1, size=(96, 72)), collection="c",
                               near_dup_policy=NearDupPolicy.SKIP)
    assert skipped.status is IngestStatus.NEAR_DUPLICATE_SKIPPED


def test_invalid_upload_reported(settings, catalog):
    r = Ingestor(settings, catalog).ingest_bytes(b"garbage", collection="c")
    assert r.status is IngestStatus.INVALID and catalog.count() == 0


def test_filters(populated, catalog):
    all_ids = catalog.ids()
    assert catalog.resolve_filter(FilterSpec()) is None
    red = catalog.resolve_filter(FilterSpec(labels=["red"]))
    assert len(red) == 4
    red_test = catalog.resolve_filter(FilterSpec(labels=["red"], splits=["test"]))
    assert len(red_test) == 2 and set(red_test) <= set(red)
    portrait = catalog.resolve_filter(FilterSpec(orientations=["portrait"]))
    assert all(catalog.get(i).height > catalog.get(i).width for i in portrait)
    tagged = catalog.resolve_filter(FilterSpec(tags_any=["odd"]))
    assert len(tagged) == 12
    wide = catalog.resolve_filter(FilterSpec(min_width=60))
    assert all(catalog.get(i).width >= 60 for i in wide)
    excl = catalog.resolve_filter(FilterSpec(labels=["red"], exclude_ids=[int(red[0])]))
    assert int(red[0]) not in excl and len(excl) == 3
    assert len(np.intersect1d(all_ids, red)) == 4


def test_deleted_images_are_filtered(populated, catalog):
    victim = populated[("red", 0)]
    assert catalog.mark_deleted(victim)
    assert victim not in catalog.ids()
    assert victim not in catalog.resolve_filter(FilterSpec(labels=["red"]))
    assert not catalog.mark_deleted(victim)


def test_facets_and_jobs(populated, catalog):
    f = catalog.facets()
    assert f["total"] == 24 and {x["value"] for x in f["labels"]} >= {"red", "blue"}
    catalog.create_job("j1", "index_images", total=3)
    catalog.update_job("j1", status="succeeded", progress=3, result={"ok": True})
    job = catalog.get_job("j1")
    assert job["status"] == "succeeded" and job["result"] == {"ok": True}
