"""Embedding models. Importing this package registers all built-in families."""

from visualsearch.embeddings import fake, hf  # noqa: F401  (registration side effects)
from visualsearch.embeddings.base import (
    IMAGE,
    TEXT,
    Embedder,
    ModelSpec,
    UnsupportedModalityError,
    create_embedder,
    families,
    load_model_specs,
    register_family,
)

__all__ = ["IMAGE", "TEXT", "Embedder", "ModelSpec", "UnsupportedModalityError",
           "create_embedder", "families", "load_model_specs", "register_family"]
