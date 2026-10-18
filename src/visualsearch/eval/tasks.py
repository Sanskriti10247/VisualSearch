"""Evaluation tasks. Each returns a JSON-serialisable result plus per-query
values (kept in memory for paired significance tests between configurations).

Task overview (relevance labels in brackets):
  t2i            Flickr30k caption -> image                [the caption's own image]
  i2t            Flickr30k image -> caption                [the image's 5 captions]
  i2i_category   Imagewoof image -> image, leave-one-out   [same breed label]
  i2i_instance   transformed Flickr image -> image         [the untransformed source]
"""

from __future__ import annotations

import hashlib
import io
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from visualsearch.eval.context import EvalContext
from visualsearch.eval.metrics import (
    PerQuery,
    per_query_from_ranks,
    percentiles,
    ranks_from_lists,
    summarize,
)
from visualsearch.index.faiss_index import FaissIndex

MAX_DEPTH = 1000


@dataclass
class TaskResult:
    task: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    per_query: PerQuery
    ranked: list[np.ndarray] = field(default_factory=list, repr=False)
    query_ids: np.ndarray | None = None
    timings: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"task": self.task, "config": self.config, "metrics": self.metrics,
                "timings": self.timings}


def search_ranked(query_vecs: np.ndarray, gallery_ids: np.ndarray, gallery_vecs: np.ndarray,
                  depth: int, index_kind: str = "flat", index_params: dict | None = None,
                  ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Build an index over the gallery and return ranked ids/scores per query."""
    t0 = time.perf_counter()
    index = FaissIndex(gallery_vecs.shape[1], index_kind, index_params or {})
    index.add(gallery_ids, gallery_vecs)
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    res = []
    for start in range(0, len(query_vecs), 1024):
        res.extend(index.search(query_vecs[start:start + 1024], min(depth, len(gallery_ids))))
    search_s = time.perf_counter() - t1
    return ([r.ids for r in res], [r.scores for r in res],
            {"index_build_s": build_s, "search_total_s": search_s,
             "search_per_query_ms": 1000 * search_s / max(len(query_vecs), 1),
             "index_stats": index.stats()})


def flickr_gallery(ctx: EvalContext, split: str, distractors: bool):
    recs = ctx.records("flickr30k", [split] + (["train"] if distractors else []))
    return recs, np.array([r.id for r in recs], dtype=np.int64)


# ------------------------------------------------------------------ t2i / i2t
def t2i(ctx: EvalContext, model: str, split: str = "test", distractors: bool = False,
        index_kind: str = "flat", index_params: dict | None = None, depth: int | None = None,
        text_template: str = "{}", **overrides: Any) -> TaskResult:
    emb = ctx.embedder(model, **overrides)
    recs, gids = flickr_gallery(ctx, split, distractors)
    gvecs = ctx.image_vectors(emb, recs)
    texts, targets = ctx.flickr_queries(split)
    qvecs = ctx.text_vectors(emb, [text_template.format(t) for t in texts])
    # Truncating at 1000 changes MRR by < 1/1000 per query and nothing else reported.
    depth = depth or min(len(gids), MAX_DEPTH)
    ranked, _scores, timing = search_ranked(qvecs, gids, gvecs, depth, index_kind, index_params)
    pq = per_query_from_ranks(ranks_from_lists(ranked, [[t] for t in targets]), clusters=targets)
    return TaskResult(
        task="t2i",
        config={"model": model, "overrides": overrides, "split": split, "distractors": distractors,
                "gallery_size": len(gids), "n_queries": len(texts), "index": index_kind,
                "index_params": index_params or {}, "depth": depth, "text_template": text_template,
                "fingerprint": emb.fingerprint},
        metrics=summarize(pq, with_map=False), per_query=pq, ranked=ranked, query_ids=targets,
        timings=timing)


def i2t(ctx: EvalContext, model: str, split: str = "test", **overrides: Any) -> TaskResult:
    emb = ctx.embedder(model, **overrides)
    recs = ctx.records("flickr30k", [split])
    ivecs = ctx.image_vectors(emb, recs)
    rows = ctx.catalog.captions(collection="flickr30k", split=split)
    cap_ids = np.array([c for c, _, _ in rows], dtype=np.int64)
    cvecs = ctx.text_vectors(emb, [t for _, _, t in rows])
    by_image: dict[int, list[int]] = {}
    for c, i, _ in rows:
        by_image.setdefault(i, []).append(c)
    ranked, _, timing = search_ranked(ivecs, cap_ids, cvecs, len(cap_ids))
    rel = [by_image[r.id] for r in recs]
    pq = per_query_from_ranks(ranks_from_lists(ranked, rel))
    return TaskResult(
        task="i2t",
        config={"model": model, "overrides": overrides, "split": split,
                "gallery_size": len(cap_ids), "n_queries": len(recs), "index": "flat",
                "fingerprint": emb.fingerprint},
        metrics=summarize(pq, with_map=True), per_query=pq, ranked=ranked,
        query_ids=np.array([r.id for r in recs]), timings=timing)


# ---------------------------------------------------------------- i2i category
def i2i_category(ctx: EvalContext, model: str, split: str = "val", **overrides: Any) -> TaskResult:
    """Imagewoof leave-one-out: every image queries all others of the split."""
    emb = ctx.embedder(model, **overrides)
    recs = ctx.records("imagewoof", [split])
    ids = np.array([r.id for r in recs], dtype=np.int64)
    labels = np.array([r.label for r in recs])
    vecs = ctx.image_vectors(emb, recs)
    ranked, _, timing = search_ranked(vecs, ids, vecs, len(ids))
    ranked = [r[r != q] for r, q in zip(ranked, ids)]  # drop the query itself
    label_of = dict(zip(ids.tolist(), labels.tolist()))
    rel = [ids[(labels == labels[n]) & (ids != q)] for n, q in enumerate(ids)]
    pq = per_query_from_ranks(ranks_from_lists(ranked, rel))
    p10 = np.array([np.mean([label_of[int(j)] == label_of[int(q)] for j in r[:10]])
                    for r, q in zip(ranked, ids)])
    pq.extra["P@10"] = p10
    metrics = summarize(pq, with_map=True)
    metrics["P@10"] = float(p10.mean())
    per_class = {lab: float(pq.ap[labels == lab].mean()) for lab in sorted(set(labels))}
    metrics["mAP_per_class"] = per_class
    return TaskResult(
        task="i2i_category",
        config={"model": model, "overrides": overrides, "dataset": "imagewoof", "split": split,
                "gallery_size": len(ids) - 1, "n_queries": len(ids), "fingerprint": emb.fingerprint},
        metrics=metrics, per_query=pq, ranked=ranked, query_ids=ids, timings=timing)


# ---------------------------------------------------------------- i2i instance
AUGMENTATIONS = ("crop", "color", "blur_jpeg", "rotate", "flip_crop")


def augment(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    """Deterministic (given rng) photometric/geometric transformation."""
    w, h = img.size
    if kind in ("crop", "flip_crop"):
        area = rng.uniform(0.3, 0.6)
        ar = rng.uniform(0.75, 1.33)
        cw = int(min(w, round((area * w * h * ar) ** 0.5)))
        ch = int(min(h, round((area * w * h / ar) ** 0.5)))
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((x, y, x + cw, y + ch))
        if kind == "flip_crop":
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img
    if kind == "color":
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.3, 1.7))
        return img
    if kind == "blur_jpeg":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.5)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(15, 35))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "rotate":
        return img.rotate(rng.uniform(-25, 25), resample=Image.Resampling.BICUBIC, expand=False,
                          fillcolor=(0, 0, 0))
    raise ValueError(kind)


def i2i_instance(ctx: EvalContext, model: str, split: str = "test", distractors: bool = True,
                 seed: int | None = None, **overrides: Any) -> TaskResult:
    """Copy detection: one transformed copy per image (augmentation kind cycles
    through AUGMENTATIONS); the untransformed original is the only relevant item."""
    from visualsearch.data.preprocess import load_rgb
    from visualsearch.embeddings import IMAGE

    seed = ctx.settings.seed if seed is None else seed
    emb = ctx.embedder(model, **overrides)
    targets = ctx.records("flickr30k", [split])
    recs, gids = flickr_gallery(ctx, split, distractors)
    gvecs = ctx.image_vectors(emb, recs)
    store = ctx.store(emb, IMAGE)
    kinds = [AUGMENTATIONS[i % len(AUGMENTATIONS)] for i in range(len(targets))]
    keys = [hashlib.sha256(f"aug:{seed}:{k}:{r.sha256}".encode()).hexdigest()
            for r, k in zip(targets, kinds)]
    by_key = {key: (r, k) for key, r, k in zip(keys, targets, kinds)}

    def compute(ks: list[str]) -> np.ndarray:
        imgs = []
        for key in ks:
            r, kind = by_key[key]
            rng = random.Random(f"{seed}:{r.sha256}")
            imgs.append(augment(load_rgb(ctx.settings.images_dir / r.path), kind, rng))
        return emb.embed_images(imgs)

    qvecs = store.get_or_compute(keys, compute, chunk=128)
    ranked, _, timing = search_ranked(qvecs, gids, gvecs, min(len(gids), MAX_DEPTH))
    tids = np.array([r.id for r in targets], dtype=np.int64)
    pq = per_query_from_ranks(ranks_from_lists(ranked, [[t] for t in tids]))
    pq.extra["kind"] = np.array(kinds)
    metrics = summarize(pq, with_map=False)
    metrics["by_augmentation"] = {
        k: summarize(pq.subset(np.array(kinds) == k), with_map=False, ci=False)
        for k in AUGMENTATIONS}
    return TaskResult(
        task="i2i_instance",
        config={"model": model, "overrides": overrides, "split": split, "distractors": distractors,
                "gallery_size": len(gids), "n_queries": len(tids), "seed": seed,
                "augmentations": list(AUGMENTATIONS), "fingerprint": emb.fingerprint},
        metrics=metrics, per_query=pq, ranked=ranked, query_ids=tids, timings=timing)


def latency_profile(fn, n: int = 200, warmup: int = 10) -> dict[str, float]:
    for _ in range(warmup):
        fn(0)
    samples = []
    for i in range(n):
        t = time.perf_counter()
        fn(i)
        samples.append(1000 * (time.perf_counter() - t))
    return percentiles(samples)
