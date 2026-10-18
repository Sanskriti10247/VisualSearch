"""Embedding-model comparisons (and representation ablations).

Text<->image only uses CLIP models (DINOv2 has no text tower). Image<->image is
evaluated for every model, at category level (Imagewoof) and instance level
(transformed copies of Flickr30k test images among 30k distractors).
"""

from __future__ import annotations

from typing import Any

from visualsearch.eval import tasks
from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import metric_values, paired_bootstrap
from visualsearch.eval.results import save_result, table

CLIP_MODELS = ["clip-vit-b32", "clip-vit-b16", "clip-vit-l14"]
IMAGE_MODELS = CLIP_MODELS + ["dinov2-small", "dinov2-base"]
BASELINE = "clip-vit-b32"


def _pct(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 2)


def _sig(base: tasks.TaskResult, other: tasks.TaskResult, metric: str) -> dict[str, float]:
    return paired_bootstrap(metric_values(base.per_query, metric),
                            metric_values(other.per_query, metric), base.per_query.clusters)


def text_image(ctx: EvalContext) -> dict[str, Any]:
    rows, raw = [], {}
    base: dict[str, tasks.TaskResult] = {}
    for gallery, distractors in (("1K", False), ("full", True)):
        for m in CLIP_MODELS:
            r = tasks.t2i(ctx, m, "test", distractors=distractors)
            raw[f"t2i/{gallery}/{m}"] = r.to_json()
            if m == BASELINE:
                base[gallery] = r
            sig = None if m == BASELINE else _sig(base[gallery], r, "R@1")
            if sig:
                raw[f"t2i/{gallery}/{m}"]["vs_baseline_R@1"] = sig
            rows.append([f"{m}", f"T→I ({r.config['gallery_size']} imgs)",
                         r.metrics["R@1"], r.metrics["R@5"], r.metrics["R@10"], r.metrics["MRR"],
                         r.metrics["median_rank"],
                         "" if not sig else f"{100 * sig['diff']:+.1f} pts (p={sig['p_value']:.3f})"])
    for m in CLIP_MODELS:
        r = tasks.i2t(ctx, m, "test")
        raw[f"i2t/1K/{m}"] = r.to_json()
        rows.append([m, "I→T (5000 captions)", r.metrics["R@1"], r.metrics["R@5"],
                     r.metrics["R@10"], r.metrics["MRR"], r.metrics["median_rank"],
                     f"mAP={100 * r.metrics['mAP']:.1f}"])
    report = {
        "title": "Text↔image retrieval: CLIP model comparison (Flickr30k test)",
        "description": (
            "Zero-shot retrieval on the Karpathy test split (1,000 images, 5,000 captions). "
            "'full' adds all 29,000 Flickr30k train images as distractors (30,000-image gallery). "
            "Exact (flat) search. Significance: clustered paired bootstrap over test images vs. the "
            "CLIP ViT-B/32 baseline."),
        "tables": [table("Retrieval quality", ["Model", "Task", "R@1", "R@5", "R@10", "MRR",
                                               "Median rank", "vs. B/32 (R@1)"], rows,
                         highlight_best=[2, 3, 4, 5])],
    }
    save_result(ctx.settings, "models_text_image", {"results": raw}, report)
    return raw


def image_image(ctx: EvalContext) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    cat_rows, inst_rows = [], []
    variants: list[tuple[str, dict[str, Any]]] = [(m, {}) for m in IMAGE_MODELS]
    variants += [("dinov2-small", {"pooling": "mean"}), ("dinov2-small", {"pooling": "cls+mean"}),
                 ("dinov2-base", {"pooling": "cls+mean"})]
    base_cat = base_inst = None
    for m, ov in variants:
        name = m + ("" if not ov else f" [{', '.join(f'{k}={v}' for k, v in ov.items())}]")
        c = tasks.i2i_category(ctx, m, "val", **ov)
        raw[f"category/{name}"] = c.to_json()
        base_cat = base_cat or c
        sig = None if c is base_cat else paired_bootstrap(base_cat.per_query.ap, c.per_query.ap)
        cat_rows.append([name, c.metrics["mAP"], c.metrics["R@1"], c.metrics["P@10"],
                         c.metrics["MRR"],
                         "" if not sig else f"{100 * sig['diff']:+.1f} (p={sig['p_value']:.3f})"])
        if not ov:
            i = tasks.i2i_instance(ctx, m, "test", distractors=True)
            raw[f"instance/{name}"] = i.to_json()
            base_inst = base_inst or i
            by = i.metrics["by_augmentation"]
            inst_rows.append([name, i.metrics["R@1"], i.metrics["R@10"], i.metrics["MRR"]] +
                             [by[a]["R@1"] for a in tasks.AUGMENTATIONS])
    report = {
        "title": "Image↔image retrieval: CLIP vs. DINOv2",
        "description": (
            "Category level: Imagewoof val (3,928 images, 10 dog breeds), every image queries all "
            "others; relevant = same breed (~390 per query), so mAP is meaningful. "
            "Instance level: one transformed copy of each Flickr30k test image (crop 30-60% area, "
            "colour jitter, blur+JPEG q15-35, rotation ±25°, flip+crop; cycled) searched against "
            "the 30,000-image gallery; relevant = the untransformed original."),
        "tables": [
            table("Category-level (Imagewoof)", ["Model", "mAP", "R@1 (P@1)", "P@10", "MRR",
                                                  "ΔmAP vs CLIP B/32"], cat_rows,
                  highlight_best=[1, 2, 3, 4]),
            table("Instance-level copy detection (Flickr30k test vs 30K gallery)",
                  ["Model", "R@1", "R@10", "MRR"] + [f"R@1 {a}" for a in tasks.AUGMENTATIONS],
                  inst_rows, highlight_best=list(range(1, 4 + len(tasks.AUGMENTATIONS)))),
        ],
    }
    save_result(ctx.settings, "models_image_image", {"results": raw}, report)
    return raw


def preprocessing(ctx: EvalContext) -> dict[str, Any]:
    """Centre crop (model default) vs. letterbox padding; and a prompt template."""
    raw, rows = {}, []
    for m in ("clip-vit-b32",):
        for crop in ("center", "pad"):
            for split in ("val", "test"):
                r = tasks.t2i(ctx, m, split, crop_mode=crop)
                raw[f"t2i/{split}/{m}/{crop}"] = r.to_json()
                rows.append([m, "T→I", split, f"crop_mode={crop}", r.metrics["R@1"],
                             r.metrics["R@10"], r.metrics["MRR"]])
        for tmpl in ("a photo of {}",):
            for split in ("val", "test"):
                r = tasks.t2i(ctx, m, split, text_template=tmpl)
                raw[f"t2i/{split}/{m}/template"] = r.to_json()
                rows.append([m, "T→I", split, f"template='{tmpl}'", r.metrics["R@1"],
                             r.metrics["R@10"], r.metrics["MRR"]])
    for m in ("clip-vit-b32", "dinov2-small"):
        for crop in ("center", "pad"):
            r = tasks.i2i_category(ctx, m, "val", crop_mode=crop)
            raw[f"i2i/{m}/{crop}"] = r.to_json()
            rows.append([m, "I→I category (mAP in R@1 col.)", "imagewoof val",
                         f"crop_mode={crop}", r.metrics["mAP"], r.metrics["R@10"], r.metrics["MRR"]])
    report = {
        "title": "Preprocessing ablations",
        "description": (
            "CLIP/DINOv2 resize the short side then centre-crop a square, discarding the edges of "
            "wide images. 'pad' letterboxes the full image onto a square canvas (mean colour) "
            "instead. Also tests prefixing captions with a prompt template."),
        "tables": [table("Ablations", ["Model", "Task", "Split", "Variant", "R@1 / mAP", "R@10",
                                       "MRR"], rows)],
    }
    save_result(ctx.settings, "preprocessing", {"results": raw}, report)
    return raw
