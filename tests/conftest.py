"""Shared fixtures: an isolated data directory, synthetic images, fake models."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog

FAKE_MODELS = """
models:
  fake-clip:
    family: fake
    hf_id: none
    revision: v1
    dim: 32
    seed: 1
  fake-dino:
    family: fake
    hf_id: none
    revision: v1
    dim: 24
    seed: 2
    text: false
"""

COLORS = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220),
          "yellow": (230, 220, 30), "purple": (140, 40, 170), "gray": (128, 128, 128)}


def make_image(color=(200, 50, 50), size=(64, 48), noise: int = 0, seed: int = 0,
               fmt: str = "JPEG", pattern: bool = True) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.zeros((size[1], size[0], 3), dtype=np.int16) + np.array(color, dtype=np.int16)
    if pattern:  # a seeded blocky pattern so pHash differs between seeds
        blocks = rng.integers(-60, 60, size=(4, 4, 1))
        arr += np.kron(blocks, np.ones((size[1] // 4 + 1, size[0] // 4 + 1, 1)))[:size[1], :size[0]].astype(np.int16)
    if noise:
        arr += rng.integers(-noise, noise + 1, size=arr.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    models = tmp_path / "models.yaml"
    models.write_text(FAKE_MODELS)
    s = Settings(data_dir=tmp_path / "data", models_file=models,
                 active_models=["fake-clip", "fake-dino"], default_text_model="fake-clip",
                 default_image_model="fake-dino", device="cpu", batch_size=8,
                 enable_cross_encoder=False, brute_force_filter_threshold=5,
                 results_dir=tmp_path / "results", max_upload_mb=1, max_batch_files=5)
    s.ensure_dirs()
    return s


@pytest.fixture
def catalog(settings: Settings) -> Catalog:
    return Catalog(settings.catalog_path)


@pytest.fixture
def populated(settings, catalog):
    """24 images: 4 per colour, alternating orientation, with tags and captions."""
    from visualsearch.pipeline.ingest import Ingestor

    ing = Ingestor(settings, catalog)
    ids = {}
    n = 0
    for name, rgb in COLORS.items():
        for j in range(4):
            size = (64, 48) if j % 2 == 0 else (48, 64)
            r = ing.ingest_bytes(make_image(rgb, size=size, seed=n), collection="synthetic",
                                 split="test" if j < 2 else "train", source_name=f"{name}_{j}.jpg",
                                 label=name, tags=[name, "even" if j % 2 == 0 else "odd"],
                                 captions=[(0, f"a {name} picture"), (1, f"something {name}")])
            assert r.image_id is not None, r
            ids[(name, j)] = r.image_id
            n += 1
    return ids


@pytest.fixture
def unit_vectors():
    def _make(n: int, d: int, seed: int = 0) -> np.ndarray:
        x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    return _make


@pytest.fixture
def rng():
    return random.Random(0)
