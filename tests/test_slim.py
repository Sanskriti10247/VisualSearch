"""Slim serving: ONNX query encoders, read-only mode, no torch on the serving path.

The parity tests compare against transformers and the pinned Hub files; they
skip when transformers, the cached model files or the local dataset are absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import make_image
from visualsearch.api.app import create_app
from visualsearch.api.services import Services
from visualsearch.config import PROJECT_ROOT, load_settings
from visualsearch.embeddings.base import ModelSpec, fingerprint, load_model_specs
from visualsearch.embeddings.onnx import (
    META_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    OnnxEmbedder,
    OnnxExportMismatch,
    load_tokenizer,
    pixel_values,
    save_tokenizer,
    tokenize,
)
from visualsearch.index.faiss_index import FaissIndex

QUERIES = [
    "a dog", "A DOG!!", "two people riding bicycles along a river at sunset",
    "", "   ", "café crème brûlée", "東京の夜景", "Ein Hund läuft über die Wiese 🐕",
    "emoji only 🎉🎉🎉", "numbers 1234 5.67 and 1/2", "tabs\tand\nnewlines",
    "it's a man's world, isn't it?", "hyphen-ated words & symbols #hashtag @user",
    "word " * 120, "supercalifragilisticexpialidocious " * 30,
]


def _real_settings():
    return load_settings()


def _torch_embedder(key: str):
    pytest.importorskip("transformers")
    from visualsearch.embeddings import create_embedder

    s = _real_settings()
    spec = load_model_specs(s.models_path)[key]
    emb = create_embedder(spec, device="cpu")
    try:
        emb.preprocessing_config()  # needs the pinned processor config in the HF cache
    except OSError as e:
        pytest.skip(f"{spec.hf_id} not available offline: {e}")
    return emb


# ------------------------------------------------------------------ parity
def test_tokenizer_matches_transformers(tmp_path):
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    spec = load_model_specs(_real_settings().models_path)["clip-vit-b32"]
    try:
        hf = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision,
                                                        local_files_only=True)
    except OSError as e:
        pytest.skip(f"CLIP tokenizer not cached: {e}")
    meta = save_tokenizer(hf, tmp_path)
    ours = load_tokenizer(tmp_path, meta)
    # One padded batch (as the server tokenises) and each query on its own.
    ref = hf(QUERIES, padding=True, truncation=True, max_length=77, return_tensors="np")
    got = tokenize(ours, QUERIES)
    assert np.array_equal(got["input_ids"], ref["input_ids"])
    assert np.array_equal(got["attention_mask"], ref["attention_mask"])
    for q in QUERIES:
        r = hf([q], padding=True, truncation=True, max_length=77)["input_ids"][0]
        assert tokenize(ours, [q])["input_ids"][0].tolist() == r, q
    assert got["input_ids"].shape[1] == 77  # long queries were truncated, not dropped


def _sample_images(n: int = 50) -> list[Image.Image]:
    s = _real_settings()
    if not s.catalog_path.exists():
        pytest.skip("no local catalogue")
    from visualsearch.data.catalog import Catalog
    from visualsearch.data.preprocess import load_rgb

    recs = [r for r in Catalog(s.catalog_path).records()
            if (s.images_dir / r.path).exists()]
    if len(recs) < n:
        pytest.skip("not enough local images")
    # Spread over collections, sizes and orientations.
    recs.sort(key=lambda r: (r.orientation, r.width * r.height, r.id))
    picked = [recs[i] for i in np.linspace(0, len(recs) - 1, n - 2).astype(int)]
    imgs = [load_rgb(s.images_dir / r.path) for r in picked]
    assert {r.orientation for r in picked} >= {"landscape", "portrait"}
    # Edge cases: smaller than the crop (padding path) and a grayscale input.
    imgs.append(Image.fromarray(np.random.default_rng(0).integers(0, 256, (90, 150, 3),
                                                                  dtype=np.uint8)))
    imgs.append(imgs[0].convert("L"))
    return imgs


@pytest.mark.parametrize("key", ["dinov2-small", "clip-vit-b32"])
def test_preprocessing_matches_hf_processor(key):
    emb = _torch_embedder(key)
    imgs = _sample_images()
    ref = emb._image_processor(images=imgs, return_tensors="np")["pixel_values"]
    got = pixel_values(imgs, emb.preprocessing_config())
    assert got.shape == ref.shape and got.dtype == np.float32
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize("key", ["clip-vit-b32", "dinov2-small"])
def test_exported_fingerprints_match_torch_and_indexes(key):
    s = _real_settings()
    if not (s.onnx_path / key / META_FILE).exists():
        pytest.skip("no ONNX export (scripts/export_onnx.py)")
    emb = _torch_embedder(key)
    slim = OnnxEmbedder(emb.spec, s.onnx_path / key)
    assert (slim.fingerprint, slim.dim) == (emb.fingerprint, emb.dim)
    assert slim.preprocessing_config() == json.loads(json.dumps(emb.preprocessing_config()))
    manifest = s.indexes_dir / key / "flat" / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["embedding_fingerprint"] == slim.fingerprint


# ------------------------------------------------------------ fake exports
SLIM_MODELS = """
models:
  slim-clip:
    family: clip
    hf_id: test/clip
    revision: r1
  slim-dino:
    family: dinov2
    hf_id: test/dino
    revision: r1
    pooling: cls
"""
PREPROC = {"do_resize": True, "size": {"shortest_edge": 32}, "resample": 3,
           "do_center_crop": True, "crop_size": {"height": 28, "width": 28}, "do_rescale": True,
           "rescale_factor": 1 / 255, "do_normalize": True, "image_mean": [0.5, 0.5, 0.5],
           "image_std": [0.25, 0.25, 0.25], "do_convert_rgb": True, "crop_mode": "center",
           "processor": "Test"}
DIM = 8


def _write_fake_exports(root: Path, specs: dict[str, ModelSpec]) -> None:
    """Tiny ONNX graphs with the exported models' interfaces."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    rng = np.random.default_rng(0)

    def save(key, graph, modalities, extra):
        d = root / key
        d.mkdir(parents=True)
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                                    ir_version=8), str(d / MODEL_FILE))
        pre = dict(PREPROC, pooling="cls") if key == "slim-dino" else PREPROC
        meta = {"format": 1, "key": key, "spec": {f: getattr(specs[key], f) for f in (
                    "family", "hf_id", "revision", "pooling", "crop_mode", "normalize")},
                "fingerprint": fingerprint(specs[key], pre, DIM), "dim": DIM,
                "preprocessing": pre, "modalities": modalities,
                "onnx": {"file": MODEL_FILE, "precision": "fp32", "quantization": None}, **extra}
        (d / META_FILE).write_text(json.dumps(meta))

    # text: sum of token ids -> [B, 1] times a random [1, D] direction (+ a bias)
    w = numpy_helper.from_array(rng.standard_normal((1, DIM)).astype(np.float32), "w")
    b = numpy_helper.from_array(rng.standard_normal((1, DIM)).astype(np.float32), "b")
    text = helper.make_graph(
        [helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT),
         helper.make_node("ReduceSum", ["f", "ax"], ["s"], keepdims=1),
         helper.make_node("Mul", ["s", "w"], ["m"]), helper.make_node("Add", ["m", "b"], ["embeds"])],
        "text", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "S"]),
                 helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "S"])],
        [helper.make_tensor_value_info("embeds", TensorProto.FLOAT, ["B", DIM])],
        [w, b, numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")])
    tok = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "red": 2, "dog": 3},
                                     unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    save("slim-clip", text, ["text"], {"tokenizer": {"file": TOKENIZER_FILE, "max_length": 16,
                                                     "pad_id": 0, "pad_token": "[PAD]"}})
    tok.save(str(root / "slim-clip" / TOKENIZER_FILE))
    # image: per-channel means [B, 3] @ [3, D]
    wi = numpy_helper.from_array(rng.standard_normal((3, DIM)).astype(np.float32), "wi")
    image = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["c"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["c", "wi"], ["embeds"])],
        "image", [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT,
                                                ["B", 3, 28, 28])],
        [helper.make_tensor_value_info("embeds", TensorProto.FLOAT, ["B", DIM])], [wi])
    save("slim-dino", image, ["image"], {})


@pytest.fixture
def slim_env(tmp_path, populated, settings):
    """A data directory with fake ONNX exports and flat indexes in their spaces."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    models_file = tmp_path / "slim_models.yaml"
    models_file.write_text(SLIM_MODELS)
    specs = load_model_specs(models_file)
    _write_fake_exports(settings.root / "onnx", specs)
    ids = np.array(sorted(populated.values()), dtype=np.int64)
    for key in specs:
        emb = OnnxEmbedder(specs[key], settings.root / "onnx" / key)
        if key == "slim-dino":  # real vectors, so similar colours rank together
            from visualsearch.data.catalog import Catalog
            from visualsearch.data.preprocess import load_rgb

            cat = Catalog(settings.catalog_path)
            recs = cat.get_many(ids.tolist())
            vecs = emb.embed_images([load_rgb(settings.images_dir / recs[i].path) for i in ids])
        else:
            vecs = np.random.default_rng(1).standard_normal((len(ids), DIM)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx = FaissIndex(DIM, "flat")
        idx.add(ids, vecs)
        d = settings.indexes_dir / key / "flat"
        idx.save(d)
        (d / "manifest.json").write_text(json.dumps({"embedding_fingerprint": emb.fingerprint}))
    env = {"VS_DATA_DIR": str(settings.root), "VS_MODELS_FILE": str(models_file),
           "VS_ACTIVE_MODELS": '["slim-clip", "slim-dino"]', "VS_DEFAULT_TEXT_MODEL": "slim-clip",
           "VS_DEFAULT_IMAGE_MODEL": "slim-dino", "VS_EMBED_RUNTIME": "onnx",
           "VS_READ_ONLY": "true", "VS_RESULTS_DIR": str(tmp_path / "results")}
    return env, populated


def test_onnx_export_must_match_spec(tmp_path, settings):
    pytest.importorskip("onnx")
    models_file = tmp_path / "m.yaml"
    models_file.write_text(SLIM_MODELS)
    specs = load_model_specs(models_file)
    _write_fake_exports(tmp_path / "onnx", specs)
    with pytest.raises(OnnxExportMismatch):
        OnnxEmbedder(specs["slim-dino"].with_overrides(revision="r2"), tmp_path / "onnx/slim-dino")
    with pytest.raises(OnnxExportMismatch):  # same spec fields, different preprocessing
        OnnxEmbedder(specs["slim-dino"].with_overrides(crop_mode="pad"),
                     tmp_path / "onnx/slim-dino")


SLIM_APP_CHECK = textwrap.dedent("""
    import io, json, sys
    from fastapi.testclient import TestClient
    from visualsearch.api.app import create_app
    from visualsearch.config import load_settings

    with TestClient(create_app(load_settings())) as c:
        out = {"health": c.get("/api/v1/health").json(), "models": c.get("/api/v1/models").json()}
        out["text"] = c.post("/api/v1/search/text", json={"query": "red dog", "k": 5}).status_code
        img = open(sys.argv[1], "rb").read()
        r = c.post("/api/v1/search/image", files={"file": ("q.jpg", img, "image/jpeg")},
                   data={"k": "3"})
        out["image_upload"] = [r.status_code, [h["metadata"]["label"] for h in r.json()["results"]]]
        r = c.post("/api/v1/search/image", files={"file": ("q.jpg", img, "image/jpeg")},
                   data={"model": "slim-clip"})
        out["clip_upload"] = [r.status_code, r.json()["error"]["code"]]
        out["clip_by_id"] = c.post("/api/v1/search/image",
                                   data={"image_id": "1", "model": "slim-clip"}).status_code
        r = c.post("/api/v1/search/text", json={"query": "red", "rerank": "cross_encoder"})
        out["cross_encoder"] = [r.status_code, r.json()["error"]["code"]]
        out["hybrid"] = c.post("/api/v1/search/hybrid", data={"text": "red", "image_id": "1"}
                               ).status_code
        out["index"] = c.post("/api/v1/index/images",
                              files=[("files", ("a.jpg", img, "image/jpeg"))]).json()
        out["delete"] = c.delete("/api/v1/images/1").status_code
        out["job"] = c.get("/api/v1/jobs/abc").status_code
    out["heavy_modules"] = sorted(m for m in ("torch", "transformers") if m in sys.modules)
    print(json.dumps(out))
""")


def test_slim_app_serves_without_torch(slim_env, tmp_path):
    env, _ = slim_env
    query = tmp_path / "q.jpg"
    query.write_bytes(make_image((30, 200, 40), seed=99))
    script = tmp_path / "check.py"
    script.write_text(SLIM_APP_CHECK)
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("VS_")}
    run_env.update(env, VS_CONFIG=str(tmp_path / "none.yaml"))
    proc = subprocess.run([sys.executable, str(script), str(query)], cwd=PROJECT_ROOT,
                          env=run_env, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["heavy_modules"] == []
    h, m = out["health"], out["models"]
    assert h["status"] == "ok" and h["read_only"] and h["embed_runtime"] == "onnx"
    assert h["models"]["slim-clip"]["modalities"] == ["text"]
    assert h["models"]["slim-dino"]["modalities"] == ["image"]
    assert m["read_only"] and "cross_encoder" not in m["rerankers"]["text"]
    assert {x["key"]: x["runtime"] for x in m["models"]} == {"slim-clip": "onnx",
                                                              "slim-dino": "onnx"}
    assert out["text"] == 200 and out["hybrid"] == 200 and out["clip_by_id"] == 200
    assert out["image_upload"][0] == 200 and out["image_upload"][1][0] == "green"
    assert out["clip_upload"] == [422, "unsupported_modality"]
    assert out["cross_encoder"] == [422, "invalid_search"]
    assert out["index"]["error"]["code"] == "read_only"
    assert out["delete"] == 403 and out["job"] == 403


def test_read_only_refuses_stale_index(slim_env, settings):
    env, _ = slim_env
    manifest = settings.indexes_dir / "slim-dino" / "flat" / "manifest.json"
    manifest.write_text(json.dumps({"embedding_fingerprint": "0000000000000000"}))
    s = settings.model_copy(update={
        "models_file": Path(env["VS_MODELS_FILE"]), "active_models": ["slim-clip", "slim-dino"],
        "default_text_model": "slim-clip", "default_image_model": "slim-dino",
        "embed_runtime": "onnx", "read_only": True})
    with pytest.raises(RuntimeError, match="rebuild the index"):
        Services.build(s)


# -------------------------------------------------------------- read-only
@pytest.fixture
def ro_client(settings, populated):
    Services.build(settings).close()  # builds the indexes; read-only mode never does
    svc = Services.build(settings.model_copy(update={"read_only": True}))
    with TestClient(create_app(svc=svc)) as c:
        yield c
    svc.close()


def test_read_only_routes_return_403(ro_client, populated):
    iid = populated[("red", 0)]
    r = ro_client.post("/api/v1/index/images", files=[("files", ("a.jpg", make_image(),
                                                                "image/jpeg"))])
    assert r.status_code == 403
    assert r.json() == {"error": {"code": "read_only", "message": r.json()["error"]["message"],
                                  "details": None}}
    assert ro_client.delete(f"/api/v1/images/{iid}").status_code == 403
    assert ro_client.get("/api/v1/jobs/anything").status_code == 403
    assert ro_client.get(f"/api/v1/images/{iid}").status_code == 200  # not deleted
    up = ro_client.post("/api/v1/search/image",
                        files={"file": ("q.png", make_image(fmt="PNG"), "image/png")})
    assert up.status_code == 200 and up.json()["results"], up.json()
    assert ro_client.get("/api/v1/models").json()["read_only"] is True
    assert ro_client.get("/api/v1/health").json()["read_only"] is True


def test_cross_encoder_unavailable_is_422(ro_client):
    r = ro_client.post("/api/v1/search/text", json={"query": "red", "rerank": "cross_encoder"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_search"


def test_full_image_falls_back_to_thumbnail(ro_client, populated, settings):
    from visualsearch.data.catalog import Catalog

    iid = populated[("blue", 1)]
    rec = Catalog(settings.catalog_path).get(iid)
    (settings.images_dir / rec.path).unlink()
    r = ro_client.get(f"/api/v1/images/{iid}/file")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == (settings.thumbs_dir / rec.thumb_path).read_bytes()
    assert "immutable" not in r.headers["cache-control"]


def test_flat_index_loads_without_vectors_file(tmp_path, unit_vectors):
    idx = FaissIndex(16, "flat")
    v = unit_vectors(50, 16)
    idx.add(np.arange(30), v[:30])
    idx.add(np.arange(30, 50), v[30:])  # storage reallocates; the vector view must follow
    assert np.allclose(idx.vectors_for(np.array([3, 45])), v[[3, 45]])
    idx.save(tmp_path)
    (tmp_path / "vectors.npy").unlink()
    loaded = FaissIndex.load(tmp_path, 16, {})
    assert np.allclose(loaded.vectors_for(np.arange(50)), v)
    assert loaded.index_bytes() == len(__import__("faiss").serialize_index(loaded.index))
