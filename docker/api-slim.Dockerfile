# VisualSearch API, slim and read-only: ONNX Runtime query encoders, no torch or
# transformers. Fits a 512 MB instance; see docs/deployment.md (Render free tier).
#   docker build -f docker/api-slim.Dockerfile -t visualsearch-api-slim .
# Base images are pinned by version and digest; bump both together.
FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

ARG APP_UID=1000
ARG APP_GID=1000

# MALLOC_ARENA_MAX: fewer glibc malloc arenas for the request thread pool, which
# otherwise holds on to freed memory per thread. VS_MAX_IMAGE_PIXELS: a decoded
# 12 MP upload is 36 MB (the default 40 MP would be 120 MB). ORT_DISABLE_TELEMETRY:
# onnxruntime's built-in usage telemetry stays off. HF_HOME: the app user has no home
# directory, and the bundle download needs a writable cache for metadata and xet logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    ORT_DISABLE_TELEMETRY=1 \
    VS_DATA_DIR=/data \
    VS_DEVICE=cpu \
    VS_EMBED_RUNTIME=onnx \
    VS_ONNX_THREADS=1 \
    VS_READ_ONLY=true \
    VS_ENABLE_CROSS_ENCODER=false \
    VS_MAX_IMAGE_PIXELS=12000000

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.26@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b /uv /usr/local/bin/uv

WORKDIR /app
# The slim extra (onnxruntime, tokenizers) and no cpu/cuda extra: no torch, no transformers.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra slim
COPY src ./src
COPY configs ./configs
# Benchmark results for the UI's experiments page (a few hundred KB).
COPY results ./results
COPY docker/slim-entrypoint.sh /app/slim-entrypoint.sh
RUN uv sync --frozen --no-dev --extra slim

RUN groupadd --gid "${APP_GID}" app \
 && useradd --uid "${APP_UID}" --gid app --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin app \
 && mkdir -p /data \
 && chown app:app /data
USER app

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
# Includes the bundle download on a first start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD curl -fs "http://localhost:${PORT:-8000}/api/v1/health" || exit 1
ENTRYPOINT ["/app/slim-entrypoint.sh"]
