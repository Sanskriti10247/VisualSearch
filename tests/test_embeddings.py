import numpy as np
import pytest
from PIL import Image

from visualsearch.config import load_settings
from visualsearch.embeddings import (
    UnsupportedModalityError,
    create_embedder,
    families,
    load_model_specs,
)
from visualsearch.embeddings.store import EmbeddingStore


def _img(c):
    return Image.new("RGB", (40, 30), c)


def test_registry_has_builtin_families():
    assert {"clip", "dinov2", "fake"} <= set(families())


def test_unknown_family_rejected(settings):
    spec = load_model_specs(settings.models_path)["fake-clip"].with_overrides(family="nope")
    with pytest.raises(KeyError):
        create_embedder(spec)


def test_fake_embedder_dims_normalised_deterministic(settings):
    specs = load_model_specs(settings.models_path)
    e = create_embedder(specs["fake-clip"])
    v = e.embed_images([_img((255, 0, 0)), _img((0, 0, 255))], batch_size=1)
    assert v.shape == (2, 32) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert np.allclose(v, e.embed_images([_img((255, 0, 0)), _img((0, 0, 255))]), atol=1e-6)
    t = e.embed_texts(["a red car", "blue"])
    assert t.shape == (2, 32)


def test_image_only_model_rejects_text(settings):
    e = create_embedder(load_model_specs(settings.models_path)["fake-dino"]).load()
    with pytest.raises(UnsupportedModalityError):
        e.embed_texts(["hello"])


def test_fingerprint_tracks_config(settings):
    spec = load_model_specs(settings.models_path)["fake-clip"]
    base = create_embedder(spec).fingerprint
    assert create_embedder(spec).fingerprint == base
    assert create_embedder(spec.with_overrides(revision="v2")).fingerprint != base
    assert create_embedder(spec.with_overrides(crop_mode="pad")).fingerprint != base
    assert create_embedder(spec.with_overrides(normalize=False)).fingerprint != base


def test_store_caches_and_persists(settings):
    e = create_embedder(load_model_specs(settings.models_path)["fake-clip"])
    calls = []

    def compute(keys):
        calls.append(list(keys))
        return e.embed_images([_img((len(k) * 3 % 255, 0, 0)) for k in keys])

    s = EmbeddingStore(settings.embeddings_dir, e, "image")
    a = s.get_or_compute(["k1", "k2"], compute)
    b = s.get_or_compute(["k2", "k3", "k1"], compute)
    assert calls == [["k1", "k2"], ["k3"]]  # only the miss is computed
    assert np.allclose(a[0], b[2])
    reloaded = EmbeddingStore(settings.embeddings_dir, e, "image")
    assert len(reloaded) == 3 and np.allclose(reloaded.get(["k1"]), a[:1])
    # a different fingerprint gets its own, empty store
    e2 = create_embedder(e.spec.with_overrides(revision="other"))
    assert len(EmbeddingStore(settings.embeddings_dir, e2, "image")) == 0


# ----------------------------------------------------------- real models (slow)
@pytest.fixture(scope="module")
def real_specs():
    s = load_settings()
    return load_model_specs(s.models_path)


@pytest.mark.slow
@pytest.mark.parametrize("key,dim,text", [("clip-vit-b32", 512, True), ("dinov2-small", 384, False)])
def test_real_model_dims(real_specs, key, dim, text):
    e = create_embedder(real_specs[key], device="cpu")
    v = e.embed_images([_img((200, 30, 30)), _img((30, 30, 200))])
    assert v.shape == (2, dim) and e.dim == dim
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert e.supports("text") is text


@pytest.mark.slow
def test_dinov2_cls_mean_pooling_doubles_dim(real_specs):
    e = create_embedder(real_specs["dinov2-small"].with_overrides(pooling="cls+mean"), device="cpu")
    assert e.embed_images([_img((1, 2, 3))]).shape == (1, 768)


@pytest.mark.slow
def test_clip_text_image_alignment(real_specs):
    e = create_embedder(real_specs["clip-vit-b32"], device="cpu")
    imgs = e.embed_images([_img((220, 20, 20)), _img((20, 20, 220))])
    txt = e.embed_texts(["a plain red square", "a plain blue square"])
    sims = txt @ imgs.T
    assert sims[0, 0] > sims[0, 1] and sims[1, 1] > sims[1, 0]
