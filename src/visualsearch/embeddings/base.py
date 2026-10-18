"""Embedding model abstraction.

Each embedding model defines its own vector *space*. Vectors from different
spaces (e.g. CLIP and DINOv2) are never compared with each other: every index
is bound to exactly one space, identified by the model's fingerprint. Combining
spaces happens only at the score level (see ``retrieval.fusion``).

To add a model:
  * same family (clip, dinov2): add an entry in ``configs/models.yaml``;
  * new family: subclass :class:`Embedder`, decorate it with
    ``@register_family("name")`` and import it in ``embeddings/__init__.py``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml
from PIL import Image

IMAGE = "image"
TEXT = "text"


class UnsupportedModalityError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    hf_id: str
    revision: str
    description: str = ""
    pooling: str = "cls"          # dinov2: cls | mean | cls+mean
    crop_mode: str = "center"     # center (model default) | pad (letterbox to square)
    normalize: bool = True        # L2-normalise so inner product == cosine similarity
    params: dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kw: Any) -> ModelSpec:
        d = asdict(self)
        d.update(kw)
        return ModelSpec(**d)


def fingerprint(spec: ModelSpec, preprocessing: dict[str, Any], dim: int) -> str:
    """Stable id of a vector space. Changes whenever anything that alters the
    vectors changes: weights revision, pooling, cropping, normalisation or
    image preprocessing parameters."""
    payload = {
        "family": spec.family, "hf_id": spec.hf_id, "revision": spec.revision,
        "pooling": spec.pooling, "crop_mode": spec.crop_mode, "normalize": spec.normalize,
        "params": spec.params, "preprocessing": preprocessing, "dim": dim,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def pad_to_square(img: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class Embedder(ABC):
    """Batch encoder for one model. Thread-safe; weights load lazily."""

    modalities: ClassVar[frozenset[str]] = frozenset({IMAGE})

    def __init__(self, spec: ModelSpec, device: str = "auto", batch_size: int = 32,
                 torch_threads: int = 0):
        self.spec = spec
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.torch_threads = torch_threads
        self._lock = threading.Lock()
        self._loaded = False

    # -- metadata ------------------------------------------------------------
    @property
    def key(self) -> str:
        return self.spec.key

    def supports(self, modality: str) -> bool:
        return modality in self.modalities

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def preprocessing_config(self) -> dict[str, Any]: ...

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.spec, self.preprocessing_config(), self.dim)

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key, "family": self.spec.family, "hf_id": self.spec.hf_id,
            "revision": self.spec.revision, "description": self.spec.description,
            "dim": self.dim, "modalities": sorted(self.modalities),
            "pooling": self.spec.pooling if self.spec.family == "dinov2" else None,
            "crop_mode": self.spec.crop_mode, "normalized": self.spec.normalize,
            "preprocessing": self.preprocessing_config(), "fingerprint": self.fingerprint,
            "device": self.device, "loaded": self._loaded,
        }

    # -- loading ---------------------------------------------------------------
    def load(self) -> Embedder:
        with self._lock:
            if not self._loaded:
                import torch

                if self.torch_threads:
                    torch.set_num_threads(self.torch_threads)
                # Full fp32 on GPU (no TF32) so CPU- and GPU-computed vectors agree
                # to ~1e-6 and can share one embedding cache.
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                self._load()
                self._loaded = True
        return self

    @abstractmethod
    def _load(self) -> None: ...

    # -- inference -------------------------------------------------------------
    def embed_images(self, images: Sequence[Image.Image], batch_size: int | None = None) -> np.ndarray:
        self.load()
        bs = batch_size or self.batch_size
        out = [np.zeros((0, self.dim), dtype=np.float32)]
        for i in range(0, len(images), bs):
            batch = [im if im.mode == "RGB" else im.convert("RGB") for im in images[i:i + bs]]
            if self.spec.crop_mode == "pad":
                batch = [pad_to_square(im, self._pad_fill()) for im in batch]
            with self._lock:
                v = self._encode_images(batch)
            out.append(v)
        x = np.concatenate(out).astype(np.float32)
        return l2_normalize(x) if self.spec.normalize else x

    def embed_texts(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        if not self.supports(TEXT):
            raise UnsupportedModalityError(
                f"model '{self.key}' has no text encoder; use a text-image model such as CLIP")
        self.load()
        bs = batch_size or max(self.batch_size, 128)
        out = [np.zeros((0, self.dim), dtype=np.float32)]
        for i in range(0, len(texts), bs):
            with self._lock:
                out.append(self._encode_texts(list(texts[i:i + bs])))
        x = np.concatenate(out).astype(np.float32)
        return l2_normalize(x) if self.spec.normalize else x

    @abstractmethod
    def _encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        raise UnsupportedModalityError(self.key)

    def _pad_fill(self) -> tuple[int, int, int]:
        mean = self.preprocessing_config().get("image_mean", (0.5, 0.5, 0.5))
        return tuple(round(m * 255) for m in mean)  # type: ignore[return-value]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


# ----------------------------------------------------------------- registry
_FAMILIES: dict[str, type[Embedder]] = {}


def register_family(name: str):
    def deco(cls: type[Embedder]) -> type[Embedder]:
        _FAMILIES[name] = cls
        return cls
    return deco


def families() -> dict[str, type[Embedder]]:
    return dict(_FAMILIES)


def load_model_specs(path: str | Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    specs = {}
    for key, cfg in (raw.get("models") or {}).items():
        known = {f for f in ModelSpec.__dataclass_fields__} - {"key"}
        extra = {k: v for k, v in cfg.items() if k not in known}
        base = {k: v for k, v in cfg.items() if k in known}
        specs[key] = ModelSpec(key=key, **base, params={**base.get("params", {}), **extra})
    return specs


def create_embedder(spec: ModelSpec, device: str = "auto", batch_size: int = 32,
                    torch_threads: int = 0) -> Embedder:
    if spec.family not in _FAMILIES:
        raise KeyError(f"unknown model family '{spec.family}'; registered: {sorted(_FAMILIES)}")
    return _FAMILIES[spec.family](spec, device=device, batch_size=batch_size,
                                  torch_threads=torch_threads)
