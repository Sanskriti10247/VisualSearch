"""Runtime configuration.

Settings are read from ``configs/default.yaml`` (or the file named by
``VS_CONFIG``) and can be overridden per key with ``VS_<KEY>`` environment
variables. Paths are resolved relative to the project root so that the CLI,
API and tests agree on where data lives regardless of the working directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VS_", extra="ignore")

    data_dir: Path = Path("data")
    models_file: Path = Path("configs/models.yaml")
    seed: int = 1234

    active_models: list[str] = Field(default_factory=lambda: ["clip-vit-b32", "dinov2-small"])
    default_text_model: str = "clip-vit-b32"
    default_image_model: str = "dinov2-small"

    index_type: str = "flat"
    index_params: dict[str, Any] = Field(default_factory=dict)

    device: str = "auto"
    torch_threads: int = 0
    batch_size: int = 32

    max_upload_mb: float = 10
    max_image_pixels: int = 40_000_000
    max_batch_files: int = 200

    default_k: int = 24
    max_k: int = 200
    rerank_candidates: int = 100
    brute_force_filter_threshold: int = 5000

    # Reranking defaults (fields of retrieval.engine.RerankConfig), chosen on val.
    rerank: dict[str, Any] = Field(default_factory=dict)
    enable_cross_encoder: bool = True   # BLIP-ITM, loaded lazily on first use

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173",
                                                             "http://127.0.0.1:5173"])
    results_dir: Path = Path("results")

    # --- derived paths -------------------------------------------------
    @property
    def root(self) -> Path:
        return _resolve(self.data_dir)

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def thumbs_dir(self) -> Path:
        return self.root / "thumbs"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def indexes_dir(self) -> Path:
        return self.root / "indexes"

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.sqlite"

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def results_path(self) -> Path:
        return _resolve(self.results_dir)

    @property
    def models_path(self) -> Path:
        return _resolve(self.models_file)

    def ensure_dirs(self) -> None:
        for d in (self.root, self.images_dir, self.thumbs_dir, self.embeddings_dir,
                  self.indexes_dir, self.calibration_dir, self.raw_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings(config_file: str | Path | None = None, **overrides: Any) -> Settings:
    """Load settings from YAML, then apply env vars (VS_*) and explicit overrides."""
    path = _resolve(config_file or os.environ.get("VS_CONFIG", "configs/default.yaml"))
    file_values: dict[str, Any] = {}
    if path.exists():
        file_values = yaml.safe_load(path.read_text()) or {}
    # Env vars take precedence over the YAML file: drop YAML keys that are set in env.
    for key in list(file_values):
        if f"VS_{key.upper()}" in os.environ:
            file_values.pop(key)
    return Settings(**{**file_values, **overrides})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
