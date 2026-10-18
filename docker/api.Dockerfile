# VisualSearch API (CPU). Build context: repository root.
#   docker build -f docker/api.Dockerfile -t visualsearch-api .
FROM python:3.12-slim

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
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first (cached layer), CPU torch wheels (~0.8 GB instead of ~4 GB).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra cpu --extra qdrant
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra cpu --extra qdrant

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD curl -fs http://localhost:8000/api/v1/health || exit 1
CMD ["visualsearch", "serve", "--host", "0.0.0.0", "--port", "8000"]
