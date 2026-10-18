"""Command-line interface. ``visualsearch --help`` lists the commands.

  data        acquire + ingest Flickr30k / Imagewoof
  ingest      ingest local image files or folders
  embed       pre-compute (cache) image embeddings for catalogue images
  index       build | sync | status of the serving indexes
  calibrate   fit score calibration on validation data
  serve       run the API server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from visualsearch.config import load_settings


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _catalog(settings):
    from visualsearch.data.catalog import Catalog

    settings.ensure_dirs()
    return Catalog(settings.catalog_path)


def cmd_data(args, settings) -> None:
    from visualsearch.data.datasets import acquire_flickr30k, acquire_imagewoof

    catalog = _catalog(settings)
    if args.dataset == "flickr30k":
        m = acquire_flickr30k(settings, catalog, n_distractors=args.distractors, workers=args.workers)
    else:
        m = acquire_imagewoof(settings, catalog)
    _print({k: v for k, v in m.items() if k != "splits"} | {
        "splits": {s: v["counts"] for s, v in m["splits"].items()}})


def cmd_ingest(args, settings) -> None:
    from visualsearch.pipeline.ingest import Ingestor, summarize

    catalog = _catalog(settings)
    ing = Ingestor(settings, catalog)
    files: list[Path] = []
    for p in map(Path, args.paths):
        files.extend(sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p])
    results = [ing.ingest_bytes(f.read_bytes(), collection=args.collection, split=args.split,
                                source_name=f.name, tags=args.tag or []) for f in files]
    _print(summarize(results))
    if args.sync:
        from visualsearch.index.manager import IndexManager

        mgr = IndexManager(settings, catalog, models=args.models)
        mgr.open(build_missing=True)
        _print({k: v["index"] for k, v in mgr.status().items()})


def cmd_embed(args, settings) -> None:
    from visualsearch.eval.context import EvalContext

    ctx = EvalContext(settings)
    records = ctx.catalog.records()
    for m in args.models:
        emb = ctx.embedder(m)
        ctx.image_vectors(emb, records, desc=f"embed {m}")
        if args.captions and emb.supports("text"):
            ctx.text_vectors(emb, [t for _, _, t in ctx.catalog.captions()])
        print(f"{m}: {len(ctx.store(emb, 'image'))} image embeddings cached "
              f"(fingerprint {emb.fingerprint})")


def cmd_index(args, settings) -> None:
    from visualsearch.index.manager import IndexManager

    if args.kind:
        settings.index_type = args.kind
    catalog = _catalog(settings)
    mgr = IndexManager(settings, catalog, models=args.models)

    def progress(model, done, total):
        print(f"\r{model}: {done}/{total}", end="", file=sys.stderr, flush=True)

    if args.action == "build":
        for m in mgr.runtimes:
            mf = mgr.rebuild(m, progress=progress)
            print(file=sys.stderr)
            _print({"model": m, "ntotal": mf["ntotal"], "build_seconds": mf["build_seconds"],
                    "disk_bytes": mf["disk_bytes"]})
    elif args.action == "sync":
        mgr.open(build_missing=False)
        for m in mgr.runtimes:
            _print(mgr.sync(m, progress=progress))
    else:
        mgr.open(build_missing=False)
        _print(mgr.status())


def cmd_calibrate(args, settings) -> None:
    from visualsearch.eval.calibrate import fit_image, fit_text
    from visualsearch.eval.context import EvalContext
    from visualsearch.retrieval.calibration import CalibrationRegistry

    ctx = EvalContext(settings)
    reg = CalibrationRegistry(settings.calibration_dir)
    for m in args.models:
        emb = ctx.embedder(m)
        cals = [fit_image(ctx, m)] + ([fit_text(ctx, m)] if emb.supports("text") else [])
        for c in cals:
            reg.put(c)
            _print({"space": c.space, "modality": c.query_modality, "mean": c.mean, "std": c.std,
                    "platt": [c.platt_a, c.platt_b], "diagnostics": c.diagnostics})


def cmd_serve(args, settings) -> None:
    import uvicorn

    uvicorn.run("visualsearch.api.app:create_app", factory=True, host=args.host, port=args.port,
                log_level="info")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="visualsearch")
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="group", required=True)

    d = sub.add_parser("data", help="acquire and ingest a dataset")
    d.add_argument("dataset", choices=["flickr30k", "imagewoof"])
    d.add_argument("--distractors", type=int, default=5000)
    d.add_argument("--workers", type=int, default=16)
    d.set_defaults(func=cmd_data)

    i = sub.add_parser("ingest", help="ingest local images (files or folders)")
    i.add_argument("paths", nargs="+")
    i.add_argument("--collection", default="uploads")
    i.add_argument("--split", default=None)
    i.add_argument("--tag", action="append")
    i.add_argument("--sync", action="store_true", help="update the serving indexes afterwards")
    i.add_argument("--models", nargs="*", default=None)
    i.set_defaults(func=cmd_ingest)

    e = sub.add_parser("embed", help="cache image embeddings for all catalogue images")
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--captions", action="store_true", help="also embed every stored caption")
    e.set_defaults(func=cmd_embed)

    x = sub.add_parser("index", help="manage serving indexes")
    x.add_argument("action", choices=["build", "sync", "status"])
    x.add_argument("--models", nargs="*", default=None)
    x.add_argument("--kind", default=None, help="flat | hnsw | ivf_flat | ivf_pq | qdrant")
    x.set_defaults(func=cmd_index)

    c = sub.add_parser("calibrate", help="fit score calibration (validation data only)")
    c.add_argument("--models", nargs="+", required=True)
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
