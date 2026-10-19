# Implementation Plan

## Environment findings (what shaped the plan)

| Finding | Consequence |
|---|---|
| Empty repository | Greenfield; Python package under `src/visualsearch`, Vite/React UI under `frontend/` |
| RTX 4050 (6 GB) but only ~7.5 GB free disk at the start | Started on CPU PyTorch wheels (≈0.8 GB vs ≈5 GB for CUDA). Once more disk was freed, switched to CUDA wheels; the torch build is now a `cpu` / `cuda` extra. Embedding runs on the GPU; FAISS benchmarks are CPU; throughput is reported for both. |
| ~4 GB free RAM | Batched, streaming embedding; no full-dataset tensors in memory |
| Flickr30k is a single 4.4 GB zip on the HF Hub | HTTP range reads of the zip central directory → fetch only the selected members, pinned to a dataset commit |
| Flickr30k has no image–image relevance labels | Image–image evaluation uses (a) Imagewoof class labels (category level) and (b) seeded photometric/geometric copies of Flickr test images (instance level) |
| A small gallery is too small for ANN to matter | Ingested all 29k Flickr30k train images as distractors (31k gallery), and added a 181k-vector scale benchmark using the non-test caption embeddings |

## Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | Environment, plan, project skeleton | `uv sync` works; plan committed |
| M1 | Data pipeline: acquisition, validation, dedup, catalog | Manifest with checksums, seeds and split counts; dedup report |
| M2 | Embedding layer: CLIP, DINOv2, registry, cache | Embeddings cached by (model fingerprint, content hash); dims verified by tests |
| M3 | FAISS indexing + text→image baseline + evaluation | Flickr30k-1K test R@K/MRR reproduced and sanity-checked against the published CLIP numbers |
| M4 | Retrieval engine, FastAPI, CLI, background ingestion jobs | All endpoints validated with tests |
| M5 | Image→image, hybrid, metadata filtering, calibration | Each evaluated against its single-signal baseline |
| M6 | Reranking (query expansion, cross-space consensus, BLIP-ITM cross-encoder) | Every reranker tuned on val and reported on test, including regressions |
| M7 | ANN benchmark (Flat/HNSW/IVF/IVF-PQ/Qdrant), throughput, memory | Machine-readable JSON in `results/` |
| M8 | React UI: search, upload, hybrid, filters, comparison, experiments | Screenshots of the running app |
| M9 | Docker, docs, architecture diagram, report | `docker compose up` serves API + UI |

## Testing strategy

- **Unit**: metrics (hand-computed cases), preprocessing and hashing, score fusion and calibration, filter planning.
- **Index consistency**: exact vs approximate agreement, persistence round-trip, incremental add, tombstones, stale-manifest detection.
- **Embedding contract**: a deterministic fake embedder for fast tests; `-m slow` tests load the real models and check dims, normalisation and determinism.
- **API**: FastAPI `TestClient` against an app wired with the fake embedder (invalid uploads, size limits, schema shape, job lifecycle).
- **Retrieval correctness**: synthetic galleries with known nearest neighbours.

## Evaluation protocol rules

1. Karpathy **val** split (and an Imagewoof *train* subset) is the only data used for tuning. **Test** is evaluated once per final configuration.
2. Seeds, dataset revision, model revisions and preprocessing config are written next to every result file.
3. Every reported number is produced by a script in this repo; results files are never edited by hand.
