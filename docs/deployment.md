# Deployment

How to run the API and web UI on one Linux server with Docker, behind Caddy for HTTPS.
Everything runs on the CPU. For local development, `docker compose up` is enough (see the
[README](../README.md#docker)).

```
 internet ──443/80──▶ caddy ──▶ web (nginx :8080) ──/api, /docs──▶ api (uvicorn :8000)
                      TLS, basic auth    static UI                     models + indexes
                      on write routes                                  /data, /models
```

Only Caddy publishes ports; `web`, `api` and the optional `qdrant` service are reachable only on
the internal Compose network.

| File | Purpose |
|---|---|
| `docker-compose.yml` | Base/dev stack: `api`, `web`, optional `qdrant` |
| `docker-compose.prod.yml` | Production override: Caddy, no published app ports, limits, log rotation |
| `.env.prod.example` | Template for `.env` (domain, admin credentials, limits) |
| `docker/Caddyfile` | TLS, request size limit, basic auth for write endpoints |
| `docker/nginx.conf` | Serves the UI, proxies `/api`, `/docs`, `/openapi.json` |
| `scripts/fetch_models.py` | Downloads the pinned model weights into the `models` volume |
| `docker/api-slim.Dockerfile` | Slim read-only API without torch, for 512 MB hosts ([section 10](#10-render-free-tier-512-mb)) |

The commands below use `dc` as shorthand:

```bash
alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml'
```

## 1. Prerequisites

- A Linux host with Docker Engine ≥ 24 and Docker Compose ≥ 2.24 (the override uses the
  `!reset`/`!override` merge tags). x86-64 is tested; arm64 should work with the CPU torch
  wheels but has not been tried.
- A DNS A/AAAA record for the hostname pointing at the server, with TCP 80 and 443 open (and
  UDP 443 for HTTP/3). Caddy needs port 80 for the ACME HTTP challenge.
- At least 2 vCPU, 4 GB RAM and about 15 GB of disk for the default models and the full
  dataset; see [sizing](#8-resource-sizing-cpu-only).
- The repository checked out on the server. The images are built from it, and `./results` is
  mounted for the UI's experiments page.

## 2. Configure

```bash
git clone https://github.com/Sanskriti10247/VisualSearch.git && cd VisualSearch
cp .env.prod.example .env
docker run --rm caddy:2.10.2-alpine caddy hash-password --plaintext 'a-strong-password'
$EDITOR .env      # set VS_DOMAIN; paste the hash into VS_ADMIN_PASSWORD_HASH in single quotes
dc config -q      # fails with a clear message if a required variable is missing
```

`.env` configures Compose and Caddy:

| Variable | Default | Meaning |
|---|---|---|
| `VS_DOMAIN` | required | Public hostname. Caddy gets a certificate for it. |
| `VS_ADMIN_USER` | `admin` | Basic-auth user for write endpoints |
| `VS_ADMIN_PASSWORD_HASH` | required | bcrypt hash from `caddy hash-password`, in single quotes (it contains `$`) |
| `VS_DATA_PATH` | `./data` | Host directory mounted at `/data`. Must be writable by uid 1000. |
| `VS_API_CPUS` / `VS_API_MEMORY` | `4` / `6g` | Container limits for the API |
| `VS_TORCH_THREADS` | `4` | torch intra-op threads. Keep it equal to `VS_API_CPUS`: torch does not see the cgroup limit. |
| `VS_ENABLE_CROSS_ENCODER` | `true` | Allow BLIP reranking. Loaded on first use, adds about 1 GB of RAM. |
| `HF_HUB_OFFLINE` | `0` | Set to `1` after provisioning so the API never contacts the Hugging Face Hub |
| `VS_IMAGE_TAG` | `latest` | Tag for the built images; a git SHA makes rollbacks possible |
| `VS_HTTP_PORT` / `VS_HTTPS_PORT` | `80` / `443` | Host ports published by Caddy |

Any setting in `configs/default.yaml` can be overridden with a `VS_<KEY>` variable. Put these in
an optional `api.env` next to the compose files; it is loaded into the API container only:

```ini
VS_ACTIVE_MODELS=["clip-vit-b32","dinov2-small"]
VS_INDEX_TYPE=flat            # flat | hnsw | ivf_flat | ivf_pq | qdrant
VS_MAX_UPLOAD_MB=10
VS_MAX_BATCH_FILES=200
```

The admin password hash is the only secret. After provisioning, the API makes no outbound calls.

### Access control

The API has no authentication of its own, so Caddy guards the endpoints that change or reveal
ingestion state: `/api/v1/index/*` (uploads), `/api/v1/jobs/*` and every `DELETE` require basic
auth. Search, images, health, models and metrics stay public. In the UI, the Index tab triggers
the browser's login prompt. To make the whole site private, widen the `@admin` matcher in
`docker/Caddyfile` to all paths. There are no user accounts and no rate limiting; if the site is
public, add rate limiting at the proxy.

## 3. Build the images

```bash
export VS_IMAGE_TAG=$(git rev-parse --short HEAD)   # or put it in .env
dc build
```

`visualsearch-api` is about 3.4 GB (Python 3.12, CPU-only torch, FAISS and the app); the first
build downloads about 1 GB of wheels and takes several minutes. `visualsearch-web` is about
85 MB: the Vite build served by unprivileged nginx.

Base images are pinned by version and digest in `docker/*.Dockerfile`, and Caddy in the
override. The API runs as uid 1000 (change it with the `APP_UID`/`APP_GID` build args) and nginx
as uid 101.

## 4. Provision the model weights and the index

Neither the weights nor the data are baked into the image:

| What | Where in the container | Persisted in | Size (default models) |
|---|---|---|---|
| Model weights (Hugging Face cache, pinned revisions from `configs/models.yaml`) | `/models` (`HF_HOME`) | named volume `models` | ~0.7 GB (CLIP B/32 + DINOv2-S); +0.9 GB BLIP |
| Data directory: `catalog.sqlite`, `images/`, `thumbs/`, `embeddings/`, `indexes/`, `calibration/`, `raw/`, upload `spool/` | `/data` | bind mount `VS_DATA_PATH` | ~7 GB full Flickr30k + Imagewoof, ~1.3 GB small demo |

On startup the API loads each active model and opens its index from
`data/indexes/<model>/<type>/`. If an index is missing, or was built with a different embedding
fingerprint, the API rebuilds it before it starts listening, by embedding every catalogue image.
On a CPU that takes hours for the full dataset, and the container reports unhealthy meanwhile.
So build the indexes before the first `up`.

### 4a. Model weights

```bash
dc run --rm --no-deps api python scripts/fetch_models.py
```

This downloads and loads every model in `active_models`, plus BLIP if
`VS_ENABLE_CROSS_ENCODER=true`, and prints each model's fingerprint. Then set `HF_HUB_OFFLINE=1`
in `.env`.

### 4b. Data and indexes

The faster route is to build on a workstation, ideally with a GPU (embedding is 10–20× faster),
by following the [README quick start](../README.md#quick-start), then copy the data directory:

```bash
rsync -a --info=progress2 data/ server:/srv/VisualSearch/data/
ssh server 'sudo chown -R 1000:1000 /srv/VisualSearch/data'
```

Indexes and embeddings are portable between CPU and GPU builds: the vectors agree to about 1e-6
and the fingerprint does not depend on the device. Serving needs `catalog.sqlite`, `indexes/`,
`calibration/`, `images/` and `thumbs/`. `embeddings/` speeds up later rebuilds, and `raw/` is
only used during acquisition.

Alternatively, build on the server with the CLI inside the API image:

```bash
mkdir -p data && sudo chown 1000:1000 data
dc run --rm --no-deps api visualsearch data imagewoof                     # ~0.3 GB
dc run --rm --no-deps api visualsearch data flickr30k --distractors 0     # 2k-image demo, ~300 MB
#                                       (or --distractors 29000 for the full 4.3 GB gallery)
dc run --rm --no-deps api visualsearch embed --models clip-vit-b32 dinov2-small
dc run --rm --no-deps api visualsearch calibrate --models clip-vit-b32 dinov2-small
dc run --rm --no-deps api visualsearch index build
dc run --rm --no-deps api visualsearch index status
```

Imagewoof is needed even for a small demo, because image-score calibration is fitted on its
train split. Your own photos can be ingested with
`dc run --rm --no-deps -v /path/to/photos:/import:ro api visualsearch ingest /import --collection mine --sync`
or uploaded later through the UI.

## 5. First deploy

```bash
dc up -d
dc ps                      # api becomes "healthy" once models and indexes are loaded
dc logs -f api caddy       # watch for "Application startup complete" and the certificate
curl -fsS https://$VS_DOMAIN/api/v1/health    # "status": "ok", every model "ready": true
```

`/api/v1/health` returns HTTP 200 as soon as the server is up, with `status: degraded` while any
index is missing or stale. With prebuilt indexes and cached weights the API starts in about
15 seconds. The UI is at `https://$VS_DOMAIN/` and the OpenAPI explorer at
`https://$VS_DOMAIN/docs`.

Upload and timeout limits at each layer:

| Layer | Setting |
|---|---|
| Caddy | request body ≤ 200 MB; no read timeout |
| nginx (`web`) | `client_max_body_size 200m`, `client_body_timeout 120s`, `proxy_read_timeout`/`proxy_send_timeout 300s` (the first BLIP rerank loads the model) |
| API | `VS_MAX_UPLOAD_MB` per file (10), `VS_MAX_BATCH_FILES` per request (200), `VS_MAX_IMAGE_PIXELS` (40 MP) |

Uploads are validated, spooled to `/data/spool` and indexed by a background job, so the request
returns as soon as the files are received.

## 6. TLS

The `caddy` service obtains and renews a Let's Encrypt certificate for `VS_DOMAIN`, redirects
HTTP to HTTPS, serves HTTP/2 and HTTP/3, and sends HSTS. Certificates and the ACME account live
in the `caddy_data` volume. Keep it: reissuing certificates on every deploy runs into Let's
Encrypt rate limits. For expiry notices, add `email you@example.com` in a global options block
at the top of `docker/Caddyfile`.

If the host already runs nginx or another TLS proxy, leave Caddy out (`dc up -d api web`) and
publish the web container on loopback with a small extra override:

```yaml
# docker-compose.local-proxy.yml
services:
  web:
    ports: !override ["127.0.0.1:8080:8080"]
```

Then point the host proxy at `http://127.0.0.1:8080`. With nginx and certbot:

```nginx
server {
    server_name search.example.com;
    client_max_body_size 200m;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
    # Replicate the Caddyfile's basic auth for /api/v1/index/, /api/v1/jobs/ and DELETE.
}
```

```bash
sudo certbot --nginx -d search.example.com    # installs the cert and a renewal timer
```

## 7. Operations and upgrades

```bash
git pull
export VS_IMAGE_TAG=$(git rev-parse --short HEAD)
dc build && dc up -d             # recreates changed containers; volumes and data are kept
dc ps && curl -fsS https://$VS_DOMAIN/api/v1/health
```

To roll back, set `VS_IMAGE_TAG` to the previous tag and run `dc up -d`; old images stay until
you remove them.

If a change to `configs/models.yaml` or the preprocessing code alters a model's fingerprint, the
API rebuilds that index at startup. Run `dc run --rm --no-deps api visualsearch index build`
first to keep the downtime short. New model revisions also need `scripts/fetch_models.py` again,
with `HF_HUB_OFFLINE=0`.

To update a base image, bump its version and digest together in the Dockerfile. Get the digest
with `docker buildx imagetools inspect <image:tag>`.

Earlier versions of the API image ran as root, so a `models` volume they created is root-owned.
Fix it once with `docker run --rm -v <project>_models:/m alpine chown -R 1000:1000 /m`, and run
`chown -R 1000:1000` on the data directory.

Back up the data directory, above all `catalog.sqlite` and `images/`; the rest can be
regenerated from those by re-embedding. Stop the API first (`dc stop api`) or use
`sqlite3 data/catalog.sqlite ".backup backup.sqlite"`. Keep `caddy_data` too. Model weights can
always be downloaded again.

Container logs (`dc logs`) rotate at 5 × 10 MB per service.
`GET /api/v1/metrics?format=prometheus` exposes request counts, latency quantiles and index
sizes.

## 8. Resource sizing (CPU-only)

Measured on a 12-thread laptop CPU (`results/throughput.json`, [REPORT](REPORT.md)) with the
default models (CLIP ViT-B/32 + DINOv2-S, 36k images, exact `flat` index):

| | Cost |
|---|---|
| API memory after warm-up | ~1.1 GiB in the container (1.8 GB RSS under load); allow 4 GB |
| BLIP cross-encoder, if used | +~1 GB RAM, and seconds per reranked query on CPU |
| Text query | ~20 ms to embed + ~5–10 ms exact search; ~38 ms end to end |
| Image query (upload) | ~60–85 ms to embed |
| Bulk embedding (ingest, index rebuilds) | ~18.6 img/s CLIP B/32, ~13.6 img/s DINOv2-S: about 1.3 h for 36k images with both models on 12 threads, 2–4× longer on a 4-vCPU VPS |
| Larger models | CLIP B/16 ~5 img/s, DINOv2-B ~4 img/s, CLIP L/14 ~1.2 img/s (impractical to embed on CPU, but servable if embeddings are built elsewhere) |
| Disk | images ~3.4 GB + ~0.7–1.6 GB weights + data (7 GB full / ~1.3 GB demo) + Docker build cache |

| Host | Dataset | Settings |
|---|---|---|
| 2 vCPU, 4 GB RAM, 20 GB disk | Demo (2k-image Flickr30k + Imagewoof) | `VS_API_CPUS=2`, `VS_TORCH_THREADS=2`, `VS_API_MEMORY=3g`, `VS_ENABLE_CROSS_ENCODER=false` |
| 4 vCPU, 8 GB RAM, 30 GB disk | Full (36k images), indexes built elsewhere | defaults |

Search latency is dominated by query embedding. Exact search stays in the low milliseconds up to
about 10⁵ vectors; beyond that, switch to `VS_INDEX_TYPE=hnsw` (see the ANN results in the
report).

## 9. Limitations

The API is a single uvicorn worker that holds the models, the indexes, an in-process job queue
and a SQLite catalogue. Running several replicas or workers against one data directory would
give each its own in-memory index and have them race on the catalogue, so it scales vertically
only. SQLite and the FAISS index files also need a local filesystem, not NFS or SMB.

Index syncs and rebuilds take a write lock, so searches wait while an uploaded batch is added.

The image is CPU-only (`VS_DEVICE=cpu`). GPU serving would need a CUDA torch build
(`--extra cuda`) and the NVIDIA container runtime, and is not covered here.

The embedded Qdrant mode is exact brute force in Python. To use a real Qdrant server, start the
`qdrant` profile (`dc --profile qdrant up -d`) and set `VS_INDEX_TYPE=qdrant` and
`VS_INDEX_PARAMS={"url": "http://qdrant:6333"}` in `api.env`.

Flickr30k annotations and the Imagewoof/ImageNet images are licensed for non-commercial
research, and Flickr images remain under their owners' terms. Serving them publicly is
redistribution, so a public deployment should index its own images.

## 10. Render free tier (512 MB)

The setup above needs about 2 GB of RAM, almost all of it torch. A Render free instance has
512 MB and half a CPU, so for it there is a slim, read-only variant of the API: same catalogue,
indexes and calibrations, but the query encoders run in ONNX Runtime and the image contains
neither torch nor transformers. The web UI is deployed separately (any static host).

### What the slim mode keeps and drops

Text queries go through CLIP ViT-B/32's text tower and image uploads through DINOv2-S, both
exported from the pinned weights by `scripts/export_onnx.py` and quantised to int8 (dynamic
quantisation; one MatMul in CLIP's first text layer stays fp32 because its input has outlier
channels that per-tensor int8 cannot represent). Each export carries the fingerprint, dimension
and preprocessing config of the torch model it came from, so the existing indexes and
calibrations are used as they are. Image preprocessing is reimplemented with PIL and numpy and
matches the transformers processor exactly; tokenisation uses the CLIP `tokenizer.json` through
the `tokenizers` library. Both are covered by parity tests in `tests/test_slim.py`.

Left out, to fit the memory:

- CLIP's image tower. Uploaded images are searched with DINOv2-S only. "Find similar" in the
  CLIP space still works for indexed images, because their vectors are in the index; an upload
  with `model=clip-vit-b32` is rejected with 422 `unsupported_modality`.
- The BLIP cross-encoder (`rerank=cross_encoder` returns 422). PRF and visual PRF work.
- Writes. With `VS_READ_ONLY=true`, `POST /index/images`, `DELETE /images/{id}` and
  `GET /jobs/{id}` return 403 with error code `read_only`, and indexes are never rebuilt: a
  fingerprint mismatch stops the server at startup instead.
- Original images. The bundle ships thumbnails only, and `/images/{id}/file` falls back to the
  thumbnail when the original is missing.

`/models` and `/health` report `embed_runtime`, `read_only` and, per model, the query
modalities that are actually available (`text` for CLIP, `image` for DINOv2). The UI hides the
Index tab and the CLIP image options accordingly.

### Measured cost

Memory, with the full 36k-image bundle, in `docker run --memory 512m --cpus 0.5`, over a script
that runs health and `/models`, about 55 text searches and 25 image uploads (one of them a
12 MP JPEG), find-similar in both spaces, filtered searches, `k=200`, hybrid searches and image
fetches, twice:

| | |
|---|---|
| RSS after startup | 230 MiB |
| Peak RSS (`VmHWM` of the server process) | 399 MiB, 414 MiB with the bundle bind-mounted from the host |
| Peak anonymous memory of the cgroup (sampled every 100 ms) | 227 MiB |
| Peak `docker stats` usage | 244 MiB |
| OOM kills | 0 |

About 130 MiB of the RSS is the two flat indexes, which a read-only server memory-maps from
`index.faiss`, and the ONNX weight files are mapped as well. Those pages are page cache, so under
pressure the kernel drops them instead of killing the process (`memory.peak` of the cgroup
reaches the limit for that reason, mostly with thumbnails). Startup takes about two seconds
once the bundle is on disk. At half a CPU a text query takes 10–20 ms and an image upload about
200 ms (under 1 s for 12 MP); the first few queries after a start are slower while the index
pages are read in.

Quality, from `scripts/run_experiments.py --only slim_runtime` (`results/slim_runtime.json`):
the same fp32 gallery vectors, queries encoded by torch fp32 or by the int8 exports.

| Task | Metric | torch fp32 | slim int8 | Δ |
|---|---|---|---|---|
| Flickr30k T→I, 1K test gallery | R@1 / R@5 / R@10 | 58.84 / 83.44 / 90.08 | 58.46 / 83.26 / 89.90 | −0.38 / −0.18 / −0.18 |
| Flickr30k T→I, 30K gallery | R@1 / R@5 / R@10 | 19.84 / 40.28 / 49.78 | 20.12 / 40.16 / 49.44 | +0.28 / −0.12 / −0.34 |
| Imagewoof I→I, val | mAP / P@1 | 75.02 / 93.66 | 74.71 / 93.46 | −0.31 / −0.20 |

Only the Imagewoof mAP difference is outside its 95% bootstrap interval.

### Environment variables

| Variable | Set in the image | Meaning |
|---|---|---|
| `VS_EMBED_RUNTIME` | `onnx` | Use the ONNX exports instead of torch models |
| `VS_ONNX_DIR` | unset | Where the exports are; default `<VS_DATA_DIR>/onnx` |
| `VS_READ_ONLY` | `true` | No indexing, deletion or jobs; never rebuild indexes |
| `VS_BUNDLE_REPO` | you set it | Hugging Face dataset repo with the bundle, e.g. `you/visualsearch-bundle` |
| `HF_TOKEN` | you set it (secret) | Read token for a private bundle repo |
| `PORT` | Render sets it | Port to listen on; default 8000 |
| `VS_CORS_ORIGINS` | you set it | Origins of the UI, as JSON: `["https://your-ui.example"]` |
| `VS_MAX_IMAGE_PIXELS` | `12000000` | Larger uploads are rejected (a decoded 40 MP image alone is 120 MB) |

`VS_ONNX_THREADS=1`, `MALLOC_ARENA_MAX=2` and `ORT_DISABLE_TELEMETRY=1` are also set in
`docker/api-slim.Dockerfile`.

### Build and upload the bundle

On a machine with the full data directory and the full install:

```bash
uv sync --extra cpu --extra export --extra slim
uv run python scripts/export_onnx.py          # -> data/onnx/{clip-vit-b32,dinov2-small}
uv run python scripts/make_serving_bundle.py  # -> bundle/
hf auth login
hf upload <user>/<repo> bundle . --repo-type dataset --private
```

The bundle holds `catalog.sqlite`, the flat index files (without `vectors.npy`, which a flat
index does not need), the three calibration files that match the exports, the thumbnails, the
ONNX models with their `embedder.json` and the CLIP `tokenizer.json`, and a `bundle.json` that
lists them. With the full dataset it is 1.0 GiB, of which 754 MiB are thumbnails. The script
refuses to run if an index was built with a different fingerprint from the export. To check the
image locally, mount a copy of the bundle (the server writes SQLite journal files next to the
catalogue, which should not end up in the upload), writable by uid 1000:
`cp -r bundle /tmp/vs-data && docker run -p 8000:8000 -v /tmp/vs-data:/data visualsearch-api-slim`.

On start, `docker/slim-entrypoint.sh` serves `/data` if it already holds a `bundle.json`;
otherwise it downloads the repo with `huggingface_hub.snapshot_download` and then starts
`visualsearch serve` on `$PORT`. The download runs in its own process, so it costs no serving
memory.

### Render settings

Create a Web Service from the repository with:

| Setting | Value |
|---|---|
| Language | Docker |
| Dockerfile Path | `docker/api-slim.Dockerfile` |
| Docker Build Context Directory | `.` |
| Health Check Path | `/api/v1/health` |
| Instance Type | Free |
| Environment | `VS_BUNDLE_REPO`, `HF_TOKEN` (secret), `VS_CORS_ORIGINS` |

Free instances have no persistent disk and stop after 15 minutes without traffic, so each
cold start downloads the bundle again (about 1 GB) before the health check passes. Build the UI
with `VITE_API_BASE=https://<service>.onrender.com npm run build` and host `frontend/dist`
anywhere static; add its origin to `VS_CORS_ORIGINS`. The Flickr30k licence note in
[Limitations](#9-limitations) applies here as well.
