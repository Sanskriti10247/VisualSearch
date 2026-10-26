"""Quality of the slim (ONNX Runtime) query encoders against the torch fp32 ones.

The gallery side is identical in both runs (cached fp32 image vectors, as in
the serving indexes); only the query encoder changes. Text->image uses the CLIP
text tower export, image->image the DINOv2 export, on the report's protocols.
"""

from __future__ import annotations

import logging
from typing import Any

from visualsearch.embeddings import create_embedder
from visualsearch.embeddings.onnx import META_FILE
from visualsearch.eval import tasks
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import metric_values, paired_bootstrap
from visualsearch.eval.results import save_result, table

TEXT_MODEL = "clip-vit-b32"
IMAGE_MODEL = "dinov2-small"
log = logging.getLogger(__name__)


def _row(label: str, metric: str, base: tasks.TaskResult, slim: tasks.TaskResult,
         name: str | None = None) -> list[Any]:
    sig = paired_bootstrap(metric_values(base.per_query, metric),
                           metric_values(slim.per_query, metric), base.per_query.clusters)
    b, s = 100 * base.metrics[metric], 100 * slim.metrics[metric]
    lo, hi = sig["ci95"]
    return [label, name or metric, round(b, 2), round(s, 2), round(s - b, 2),
            f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"]


def run(ctx: EvalContext) -> dict[str, Any]:
    s = ctx.settings
    missing = [m for m in (TEXT_MODEL, IMAGE_MODEL) if not (s.onnx_path / m / META_FILE).exists()]
    if missing:
        log.warning("slim_runtime skipped: no ONNX export for %s in %s (scripts/export_onnx.py)",
                    missing, s.onnx_path)
        return {}
    onnx = {m: create_embedder(ctx.specs[m], runtime="onnx", onnx_dir=s.onnx_path,
                               onnx_threads=4) for m in (TEXT_MODEL, IMAGE_MODEL)}
    rows, raw = [], {}
    for gallery, distractors in (("1K", False), ("30K", True)):
        base = tasks.t2i(ctx, TEXT_MODEL, "test", distractors=distractors)
        slim = tasks.t2i(ctx, TEXT_MODEL, "test", distractors=distractors,
                         query_emb=onnx[TEXT_MODEL])
        raw[f"t2i/{gallery}"] = {"torch": base.to_json(), "onnx": slim.to_json()}
        for metric in ("R@1", "R@5", "R@10"):
            rows.append(_row(f"Flickr30k T→I, {gallery} gallery", metric, base, slim))
    base = tasks.i2i_category(ctx, IMAGE_MODEL, "val")
    slim = tasks.i2i_category(ctx, IMAGE_MODEL, "val", query_emb=onnx[IMAGE_MODEL])
    raw["i2i_category"] = {"torch": base.to_json(), "onnx": slim.to_json()}
    # With several relevant items, R@1 (first relevant at rank 1) is P@1.
    for metric, name in (("mAP", "mAP"), ("R@1", "P@1")):
        rows.append(_row("Imagewoof I→I (val, leave-one-out)", metric, base, slim, name))

    exports = {m: {k: e.meta.get(k) for k in ("onnx", "check", "exported_with")}
               for m, e in onnx.items()}
    precision = ", ".join(f"{m} {e.meta['onnx']['precision']}" for m, e in onnx.items())
    report = {
        "title": "Slim runtime: ONNX query encoders vs torch fp32",
        "description": (
            "Queries encoded by the ONNX Runtime exports used by the slim server, against the "
            "same fp32 gallery vectors as the torch run. Text→image: Flickr30k test captions, "
            "1K and 30K galleries. Image→image: Imagewoof val, leave-one-out. "
            f"Exports: {precision}. Δ CI: clustered paired bootstrap."),
        "tables": [table("Query encoder runtime", ["Task", "Metric", "torch fp32", "ONNX", "Δ",
                                                   "Δ 95% CI"], rows)],
    }
    save_result(s, "slim_runtime", {"exports": exports, "raw": raw, "rows": rows}, report)
    return {"rows": rows}
