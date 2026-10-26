#!/usr/bin/env python
"""Collect what the slim, read-only server needs into one directory.

    uv run python scripts/make_serving_bundle.py            # -> ./bundle/
    huggingface-cli upload <user>/<repo> bundle/ --repo-type dataset --private

The bundle is a data directory (mount it, or let docker/slim-entrypoint.sh
download it, as /data):

  catalog.sqlite                       consistent copy; job history dropped
  indexes/<model>/<kind>/              serving index files (no vectors.npy for flat:
                                       the vectors are read from index.faiss)
  calibration/<model>__<modality>.json those matching the exported fingerprints
  thumbs.tar                           thumbnails of the catalogued images only, as one
                                       uncompressed tar: tens of thousands of small files
                                       download from the Hub at ~1-2 files/s, one archive
                                       in about a minute. The entrypoint unpacks it.
  onnx/<model>/                        model.onnx, embedder.json, tokenizer.json
  bundle.json                          contents, fingerprints and sizes

Originals are left out: the API falls back to the thumbnail for full images.
Every index and calibration must carry the fingerprint of the ONNX export of its
model, so the server starts without rebuilding anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tarfile
from pathlib import Path

from visualsearch.config import PROJECT_ROOT, get_settings
from visualsearch.data.catalog import now_iso
from visualsearch.embeddings.onnx import read_meta

FLAT_SKIP = {"vectors.npy"}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # same filesystem: no second copy on disk
    except OSError:
        shutil.copy2(src, dst)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def copy_catalog(src: Path, dst: Path) -> list[str]:
    """Online backup (includes the WAL), then a compact single file without job history."""
    s, d = sqlite3.connect(src), sqlite3.connect(dst, isolation_level=None)
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")
        d.execute("DELETE FROM jobs")
        d.execute("VACUUM")
        return [r[0] for r in d.execute(
            "SELECT thumb_path FROM images WHERE deleted = 0 AND thumb_path IS NOT NULL")]
    finally:
        s.close()
        d.close()


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "bundle")
    ap.add_argument("--models", nargs="+", default=s.active_models)
    ap.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = ap.parse_args()
    out: Path = args.out
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    info: dict = {"created_at": now_iso(), "index_type": s.index_type, "models": {}}

    thumbs = copy_catalog(s.catalog_path, out / "catalog.sqlite")
    # JPEGs don't compress further, so the tar is stored uncompressed.
    with tarfile.open(out / "thumbs.tar", "w") as tar:
        for rel in thumbs:
            tar.add(s.thumbs_dir / rel, arcname=rel)
    info["catalog_images"] = len(thumbs)

    for key in args.models:
        meta = read_meta(s.onnx_path / key)
        fp = meta["fingerprint"]
        for f in (s.onnx_path / key).iterdir():
            link_or_copy(f, out / "onnx" / key / f.name)

        src = s.indexes_dir / key / s.index_type
        manifest = json.loads((src / "manifest.json").read_text())
        if manifest["embedding_fingerprint"] != fp:
            raise SystemExit(f"{key}: index fingerprint {manifest['embedding_fingerprint']} != "
                             f"ONNX export {fp}; rebuild the index or re-export")
        skip = FLAT_SKIP if s.index_type == "flat" else set()
        for f in src.iterdir():
            if f.is_file() and f.name not in skip:
                link_or_copy(f, out / "indexes" / key / s.index_type / f.name)

        calibrations = []
        for p in sorted(s.calibration_dir.glob(f"{key}__*.json")):
            if json.loads(p.read_text()).get("fingerprint") == fp:
                link_or_copy(p, out / "calibration" / p.name)
                calibrations.append(p.name)
        info["models"][key] = {"fingerprint": fp, "modalities": meta["modalities"],
                               "precision": meta["onnx"]["precision"],
                               "ntotal": manifest.get("ntotal"), "calibrations": calibrations}

    sizes = {p.name: dir_size(p) if p.is_dir() else p.stat().st_size for p in out.iterdir()}
    info["bytes"] = {**sizes, "total": sum(sizes.values())}
    (out / "bundle.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"wrote {out} ({info['bytes']['total'] / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
