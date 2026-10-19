# API reference

Base path: `/api/v1`. Interactive docs: `http://localhost:8000/docs` (Swagger) and
`/redoc`. The machine-readable spec is [`openapi.json`](openapi.json)
(regenerate with `uv run python scripts/export_openapi.py`).

## Conventions

- **Errors** always use one envelope, with an appropriate status code:
  ```json
  {"error": {"code": "invalid_image", "message": "could not decode image: ...", "details": null}}
  ```
  | Status | `code` | When |
  |---|---|---|
  | 400 | `empty_file` | zero-byte upload |
  | 404 | `not_found` | unknown image or job id |
  | 413 | `file_too_large`, `too_many_files` | upload over `max_upload_mb` (default 10 MB) or more than `max_batch_files` (200) |
  | 415 | `invalid_image` | not decodable, not an allowed format (JPEG, PNG, WebP, BMP, GIF, TIFF), truncated, or more than `max_image_pixels` |
  | 422 | `validation_error` | schema violations (details lists each field error) |
  | 422 | `invalid_search`, `invalid_filters`, `unsupported_modality` | semantically invalid search (e.g. text query on DINOv2, early fusion across spaces) |
  | 500 | `internal_error` | unexpected server error (logged) |
- Uploads are validated by decoding them, never by trusting the filename or `Content-Type`.
- **Filters** (`Filters` object) are AND-ed across fields and OR-ed within a field:
  ```json
  {"collections": ["flickr30k"], "splits": ["test"], "labels": [], "tags_any": ["beach"],
   "orientations": ["landscape"], "min_width": 400, "min_height": null, "exclude_ids": [12]}
  ```
  In JSON endpoints they're an object; in multipart endpoints a JSON-encoded string form field.

## Search response

All three search endpoints return the same `SearchResponse`:

```json
{
  "request_id": "6f1c0e2a9b1d",
  "results": [
    {
      "rank": 1,
      "image_id": 812,
      "score": 0.3121,
      "signals": {
        "clip-vit-b32:text": {"raw": 0.3121, "z": 5.4, "percentile": 0.9993, "prob": 0.41}
      },
      "thumbnail_url": "/api/v1/images/812/thumbnail",
      "image_url": "/api/v1/images/812/file",
      "metadata": {"id": 812, "collection": "flickr30k", "split": "test", "width": 500,
                   "height": 375, "orientation": "landscape", "label": null, "tags": [], "...": "..."}
    }
  ],
  "total": 24,
  "k": 24,
  "timings_ms": {"embed_query": 6.1, "search": 1.9, "total": 8.4, "hydrate": 1.2},
  "strategy": {"mode": "text", "model": "clip-vit-b32", "index": "flat",
               "filter_strategy": "index", "filter_matches": null, "candidates": 24, "rerank": "none"},
  "warnings": []
}
```

- `timings_ms` breaks the request into stages: `embed_query`, `filter` (resolving metadata filters
  in SQLite, present only when filters are set), `search`, `rerank` or `fuse` when used, `hydrate`
  (loading metadata for the results) and `total`.
- `score` is the final ranking score. Its scale depends on the strategy (cosine for
  single-signal search, a fused z-score for hybrid), so only compare scores within one response.
- `signals` has one entry per signal: `raw` cosine similarity in that model's space; `z` and
  `percentile` relative to random (query, image) pairs on validation data; `prob` = Platt-calibrated
  P(annotated match). These are present only when a calibration for that model's fingerprint exists.

## Endpoints

### `POST /search/text` (JSON)

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string (1–500) | required | blank strings rejected |
| `k` | int 1–200 | 24 | |
| `model` | string | `default_text_model` | must have a text encoder (CLIP) |
| `filters` | Filters | none | |
| `rerank` | `none` \| `prf` \| `visual_prf` \| `cross_encoder` | `none` | see the report for measured effects |
| `candidates` | int 1–1000 | 100 (20 for cross-encoder) | first-stage depth when reranking |
| `min_prob` | float 0–1 | none | drop results with calibrated P(match) below this |

```bash
curl -s localhost:8000/api/v1/search/text -H 'content-type: application/json' \
  -d '{"query": "a red backpack on a wooden table near a window", "k": 12}'
```

### `POST /search/image` (multipart)

Exactly one of `file` (upload) or `image_id` (an indexed image, "find similar"). Fields: `k`,
`model` (default `default_image_model`, i.e. DINOv2-S), `filters` (JSON string), `rerank`
(`none` \| `prf`), `exclude_query` (default true: the query image itself and byte-identical
uploads are omitted, with a warning), `min_prob`.

```bash
curl -s localhost:8000/api/v1/search/image -F file=@dog.jpg -F k=12 -F model=dinov2-small
curl -s localhost:8000/api/v1/search/image -F image_id=812 -F model=clip-vit-b32
```

### `POST /search/hybrid` (multipart)

`text` and/or (`file` | `image_id`), plus `text_model`, `image_model`, `text_weight` (0 = image only,
1 = text only; default 0.5), `fusion` (`zscore` default \| `minmax` \| `rrf` \| `early`),
`filters`, `k`, `exclude_query`, `min_prob`. If only one input is given, it runs single-modality
search and adds a warning. `early` requires `text_model == image_model` (one CLIP space).

```bash
curl -s localhost:8000/api/v1/search/hybrid -F text="on a beach" -F image_id=812 \
  -F image_model=dinov2-small -F text_weight=0.5 -F fusion=zscore
```

### `POST /index/images` (multipart) → 202

`files` (repeatable, at most 200), `collection` (`^[\w-]+$`, default `uploads`), `tags`
(comma-separated). Every file is decoded and validated in the request; invalid files are rejected
(listed in the job's `error`) and the rest are spooled to disk. A background job then ingests
them (exact and perceptual dedup) and syncs every model's index. Returns
`{"job_id", "status_url", "accepted_files"}`.

### `GET /jobs/{job_id}`

`{"id", "kind", "status": "queued|running|succeeded|failed", "progress", "total", "result", "error"}`.
`result.counts` gives added / duplicate / near-duplicate / invalid; `result.sync` gives per-model
index additions.

### Images

- `GET /images/{id}`: metadata, captions, URLs and which model indexes contain the image.
- `GET /images/{id}/thumbnail` and `GET /images/{id}/file`: JPEG thumbnail / original bytes
  (content-addressed, served with `Cache-Control: immutable` and the SHA-256 as `ETag`).
- `DELETE /images/{id}` → 202: tombstones the image in the catalogue; a background sync removes
  it from every index.

### Operations

- `GET /health`: `ok` when every active model has a built, non-stale index; otherwise `degraded`.
- `GET /metrics`: JSON with per-route request counts and latency p50/p95/max, per-stage search
  latencies (embed/search/rerank/fuse/hydrate) and index stats. `?format=prometheus` returns the
  Prometheus text format.
- `GET /models`: loaded models (dim, modalities, fingerprint, preprocessing, device), index stats,
  calibrations, available rerankers and fusions, and limits.
- `GET /facets`: filterable values with counts.
- `GET /experiments`: the JSON files in `results/` (rendered by the UI's Experiments view).
