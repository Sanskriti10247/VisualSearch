#!/bin/sh
# Entrypoint of the slim API image (docker/api-slim.Dockerfile).
#
# Serves the bundle in $VS_DATA_DIR (default /data). If it is not there yet, it
# is downloaded first from the Hugging Face dataset repo $VS_BUNDLE_REPO (with
# $HF_TOKEN for a private repo). The download runs in its own process, so its
# memory is returned before the server starts. An interrupted download leaves
# a marker file and is resumed on the next start.
set -eu

DATA="${VS_DATA_DIR:-/data}"
MARKER="$DATA/.bundle-incomplete"

if [ -f "$DATA/bundle.json" ] && [ ! -f "$MARKER" ]; then
  echo "slim-entrypoint: serving the bundle in $DATA"
else
  if [ -z "${VS_BUNDLE_REPO:-}" ]; then
    echo "slim-entrypoint: no bundle in $DATA and VS_BUNDLE_REPO is not set" >&2
    exit 1
  fi
  echo "slim-entrypoint: downloading $VS_BUNDLE_REPO into $DATA"
  mkdir -p "$DATA"
  touch "$MARKER"
  python - <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo, data = os.environ["VS_BUNDLE_REPO"], os.environ.get("VS_DATA_DIR", "/data")
# If the repo cannot be reached, snapshot_download logs a warning and returns
# local_dir as it is, so check that a bundle actually arrived.
snapshot_download(repo_id=repo, repo_type="dataset", token=os.environ.get("HF_TOKEN") or None,
                  local_dir=data)
if not os.path.exists(os.path.join(data, "bundle.json")):
    sys.exit(f"slim-entrypoint: no bundle.json after downloading {repo} (wrong repo id, "
             "missing HF_TOKEN for a private repo, or not a serving bundle)")

# Thumbnails ship as one tar (see scripts/make_serving_bundle.py).
import tarfile

archive = os.path.join(data, "thumbs.tar")
if os.path.exists(archive):
    print("slim-entrypoint: unpacking thumbs.tar", flush=True)
    with tarfile.open(archive) as tar:
        tar.extractall(os.path.join(data, "thumbs"), filter="data")
    os.remove(archive)
PY
  rm -f "$MARKER"
fi

exec visualsearch serve --host 0.0.0.0 --port "${PORT:-8000}"
