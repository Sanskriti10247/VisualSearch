#!/usr/bin/env python
"""Run the experiment suite and write machine-readable results to results/.

    uv run python scripts/run_experiments.py                # everything, in order
    uv run python scripts/run_experiments.py --only ann rerank
    uv run python scripts/run_experiments.py --list

Order matters: `calibration` must run before `hybrid` (z-score fusion uses the
fitted background statistics) and before `latency` (served scores use them).
All tuning uses validation data; test data is only evaluated.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from visualsearch.config import load_settings
from visualsearch.eval.context import EvalContext
from visualsearch.eval.experiments import (
    analysis,
    ann,
    filters,
    hybrid,
    models,
    performance,
    rerank,
    slim,
)

EXPERIMENTS = {
    "models_text_image": models.text_image,
    "models_image_image": models.image_image,
    "preprocessing": models.preprocessing,
    "calibration": analysis.calibration,
    "hard_negatives": analysis.hard_negatives,
    "hybrid": hybrid.run,
    "rerank": rerank.run,
    "ann": ann.run,
    "filters": filters.run,
    "throughput": performance.throughput,
    "latency": performance.query_latency,
    "slim_runtime": slim.run,   # needs the ONNX exports (scripts/export_onnx.py)
}
log = logging.getLogger("experiments")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    p.add_argument("--list", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    if args.list:
        print("\n".join(EXPERIMENTS))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ctx = EvalContext(load_settings(args.config))
    for name in args.only or list(EXPERIMENTS):
        t = time.time()
        log.info("=== %s ===", name)
        EXPERIMENTS[name](ctx)
        log.info("=== %s done in %.0fs ===", name, time.time() - t)


if __name__ == "__main__":
    sys.exit(main())
