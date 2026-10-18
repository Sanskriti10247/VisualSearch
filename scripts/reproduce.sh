#!/usr/bin/env bash
# Reproduce the full pipeline: data -> embeddings -> calibration -> indexes -> experiments.
#
#   TORCH=cuda scripts/reproduce.sh     # GPU (default: cpu)
#   SKIP_DATA=1 scripts/reproduce.sh    # reuse an existing data/ directory
#
# Disk: ~5 GB data + ~3.5 GB model weights (HF cache). Time: ~1.5 h on an RTX 4050
# laptop GPU; many hours on CPU (the cross-encoder and ViT-L/14 dominate).
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH="${TORCH:-cpu}"
uv sync --extra "$TORCH" --extra qdrant --extra dev --extra report

if [[ -z "${SKIP_DATA:-}" ]]; then
  # Flickr30k: Karpathy test + val + all 29k train images (distractors), pinned revision.
  uv run visualsearch data flickr30k --distractors 29000 --workers 32
  # Imagewoof: val (3,929) + seeded ~1/9 of train (tuning), checksum-verified.
  uv run visualsearch data imagewoof
fi

MODELS="clip-vit-b32 clip-vit-b16 clip-vit-l14 dinov2-small dinov2-base"
uv run visualsearch embed --models $MODELS --captions

# Calibration (validation data only) must precede hybrid fusion and serving.
uv run python scripts/run_experiments.py --only calibration
uv run python scripts/run_experiments.py --only \
  models_text_image models_image_image preprocessing hard_negatives hybrid rerank ann filters throughput

# Serving indexes for the default configuration, then serving latency.
uv run visualsearch index build
uv run python scripts/run_experiments.py --only latency
uv run python scripts/make_report.py
echo "Done. Results in results/, report tables in docs/REPORT_TABLES.md"
