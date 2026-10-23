# VisualSearch API (CPU). Build context: repository root.
#   docker build -f docker/api.Dockerfile -t visualsearch-api .
# Base images are pinned by version and digest; bump both together.
FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

# Runtime user; a bind-mounted data directory must be writable by it.
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    VS_DATA_DIR=/data \
    VS_DEVICE=cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.26@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first (cached layer), CPU torch wheels (~0.8 GB instead of ~4 GB).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra cpu --extra qdrant
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra cpu --extra qdrant

# /data and /models are created owned by app so new named volumes inherit it.
RUN groupadd --gid "${APP_GID}" app \
 && useradd --uid "${APP_UID}" --gid app --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin app \
 && mkdir -p /data /models /app/results \
 && chown app:app /data /models
USER app

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
# Models load and missing indexes are built before the port opens.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD curl -fs http://localhost:8000/api/v1/health || exit 1
CMD ["visualsearch", "serve", "--host", "0.0.0.0", "--port", "8000"]
