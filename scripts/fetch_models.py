"""Download and load the pinned weights for the active models (and BLIP when
``enable_cross_encoder`` is on) into ``HF_HOME``, so the server can then run
with ``HF_HUB_OFFLINE=1``. Reads the same settings as ``visualsearch serve``.

    docker compose ... run --rm --no-deps api python scripts/fetch_models.py
"""

from __future__ import annotations

from visualsearch.config import get_settings
from visualsearch.embeddings import create_embedder, load_model_specs


def main() -> None:
    settings = get_settings()
    specs = load_model_specs(settings.models_path)
    for key in settings.active_models:
        emb = create_embedder(specs[key], device="cpu")
        emb.load()
        print(f"{key}: {specs[key].hf_id}@{specs[key].revision[:8]} ready "
              f"(fingerprint {emb.fingerprint})", flush=True)
    if settings.enable_cross_encoder:
        from visualsearch.retrieval.cross_encoder import BlipITMReranker

        blip = BlipITMReranker(device="cpu").load()
        print(f"cross-encoder: {blip.hf_id}@{blip.revision[:8]} ready", flush=True)


if __name__ == "__main__":
    main()
