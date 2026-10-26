"""Reproducible acquisition of the evaluation datasets.

Flickr30k (text<->image)
    Source: ``nlphuji/flickr30k`` on the Hugging Face Hub, pinned to a commit.
    Splits: the standard Karpathy split from the annotation CSV
    (29,000 train / 1,014 val / 1,000 test, 5 captions per image).
    We ingest all of test and val, plus a seeded random sample of train
    images that serve only as *distractors* (their captions are stored but
    never used as queries). Distractors that are perceptual near-duplicates
    of an already-ingested image are skipped, so they cannot become unlabeled
    true positives for a test query.
    Licence: Flickr30k annotations are for non-commercial research use; images
    remain under the Flickr terms of use of their owners. Nothing is
    redistributed by this repository.

Imagewoof (image<->image, category level)
    Source: fast.ai Imagewoof2 (320px), 10 dog breeds from ImageNet.
    Labels come from the ImageNet synsets (real human annotations).
    We ingest the full ``val`` split (evaluation) and a seeded ~1/9 sample of
    ``train`` (tuning only). Licence: ImageNet terms (non-commercial research).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import platform
import random
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog, now_iso
from visualsearch.data.remote_zip import RemoteZip
from visualsearch.pipeline.ingest import Ingestor, IngestResult, NearDupPolicy, summarize

FLICKR_REPO = "nlphuji/flickr30k"
FLICKR_REVISION = "2b239befc81b6e3f035ce6bd52f5f4d60f5625f7"
FLICKR_BASE = f"https://huggingface.co/datasets/{FLICKR_REPO}/resolve/{FLICKR_REVISION}"
FLICKR_CSV_SHA256 = "395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137"

IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"
IMAGEWOOF_SHA256 = "7db6120fdb9ae079e26346f89e7b00d7f184f8137791609b97fd0405d3f92305"
IMAGEWOOF_CLASSES = {
    "n02086240": "Shih-Tzu",
    "n02087394": "Rhodesian ridgeback",
    "n02088364": "beagle",
    "n02089973": "English foxhound",
    "n02093754": "Border terrier",
    "n02096294": "Australian terrier",
    "n02099601": "golden retriever",
    "n02105641": "Old English sheepdog",
    "n02111889": "Samoyed",
    "n02115641": "dingo",
}


def manifest_path(settings: Settings, name: str) -> Path:
    p = settings.root / "manifests" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_manifest(settings: Settings, name: str, payload: dict[str, Any]) -> Path:
    p = manifest_path(settings, name)
    payload = {**payload, "written_at": now_iso(), "python": platform.python_version()}
    p.write_text(json.dumps(payload, indent=2))
    return p


def _seeded_hash(seed: int, name: str) -> int:
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:12], 16)


# ----------------------------------------------------------------- Flickr30k
def load_flickr_annotations(settings: Settings) -> tuple[list[dict[str, Any]], str]:
    path = settings.raw_dir / "flickr30k" / "flickr_annotations_30k.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{FLICKR_BASE}/flickr_annotations_30k.csv", timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if FLICKR_CSV_SHA256 and sha != FLICKR_CSV_SHA256:
        raise RuntimeError(f"annotation checksum mismatch: {sha}")
    rows = []
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        captions = ast.literal_eval(row["raw"])
        sent_ids = ast.literal_eval(row["sentids"])
        if len(captions) != len(sent_ids):
            raise ValueError(f"caption/sentid length mismatch for {row['filename']}")
        rows.append({"filename": row["filename"], "split": row["split"],
                     "img_id": int(row["img_id"]),
                     "captions": list(zip(sent_ids, captions))})
    return rows, sha


def acquire_flickr30k(settings: Settings, catalog: Catalog, n_distractors: int = 5000,
                      seed: int | None = None, workers: int = 16) -> dict[str, Any]:
    seed = settings.seed if seed is None else seed
    t0 = time.time()
    rows, csv_sha = load_flickr_annotations(settings)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split[r["split"]].append(r)
    for s in by_split.values():
        s.sort(key=lambda r: r["filename"])
    rng = random.Random(seed)
    distractors = rng.sample(by_split["train"], k=min(n_distractors, len(by_split["train"])))
    distractors.sort(key=lambda r: r["filename"])

    rz = RemoteZip(f"{FLICKR_BASE}/flickr30k-images.zip",
                   index_cache=settings.raw_dir / "flickr30k" / "zip_index.json")
    ingestor = Ingestor(settings, catalog)
    already = {r.source_name for r in catalog.records(collection="flickr30k")}

    # Order matters: evaluation images first so that the distractor near-dup
    # filter compares against them.
    plan = [("test", by_split["test"], NearDupPolicy.KEEP),
            ("val", by_split["val"], NearDupPolicy.KEEP),
            ("train", distractors, NearDupPolicy.SKIP)]
    per_split: dict[str, Any] = {}
    for split, items, policy in plan:
        todo = [r for r in items if r["filename"] not in already]

        def fetch(r: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
            return r, rz.read(f"flickr30k-images/{r['filename']}")

        results: list[IngestResult] = []
        with ThreadPoolExecutor(workers) as pool:
            for r, data in tqdm(pool.map(fetch, todo), total=len(todo), desc=f"flickr30k/{split}"):
                res = ingestor.ingest_bytes(
                    data, collection="flickr30k", split=split, source_name=r["filename"],
                    captions=r["captions"],
                    extra={"flickr_img_id": r["img_id"], "role": "eval" if split != "train" else "distractor"},
                    near_dup_policy=policy)
                results.append(res)
        per_split[split] = {"requested": len(items), "fetched": len(todo), **summarize(results)}

    manifest = {
        "dataset": "flickr30k",
        "source": {"repo": FLICKR_REPO, "revision": FLICKR_REVISION,
                   "annotations_sha256": csv_sha, "split_scheme": "karpathy"},
        "seed": seed,
        "n_distractors_requested": n_distractors,
        "distractor_filenames_sha256": hashlib.sha256(
            "\n".join(r["filename"] for r in distractors).encode()).hexdigest(),
        "near_duplicate_policy": {"test": "keep+record", "val": "keep+record",
                                  "train_distractors": "skip if pHash Hamming <= threshold"},
        "splits": per_split,
        "catalog_counts": {s: len(catalog.records(collection="flickr30k", split=s))
                           for s in ("train", "val", "test")},
        "seconds": round(time.time() - t0, 1),
    }
    write_manifest(settings, "flickr30k", manifest)
    return manifest


# ----------------------------------------------------------------- Imagewoof
def acquire_imagewoof(settings: Settings, catalog: Catalog, train_fraction_denominator: int = 9,
                      seed: int | None = None) -> dict[str, Any]:
    """Stream the Imagewoof archive once; keep all val images and ~1/9 of train."""
    seed = settings.seed if seed is None else seed
    t0 = time.time()
    ingestor = Ingestor(settings, catalog)
    already = {r.source_name for r in catalog.records(collection="imagewoof")}
    results: dict[str, list[IngestResult]] = {"val": [], "train": []}
    hasher = hashlib.sha256()

    class _HashingReader(io.RawIOBase):
        def __init__(self, raw):
            self.raw = raw

        def readable(self):
            return True

        def readinto(self, b):
            data = self.raw.read(len(b))
            hasher.update(data)
            b[: len(data)] = data
            return len(data)

    with requests.get(IMAGEWOOF_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        stream = io.BufferedReader(_HashingReader(resp.raw), buffer_size=1 << 20)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar, \
                tqdm(total=total, unit="B", unit_scale=True, desc="imagewoof") as bar:
            last = 0
            for member in tar:
                pos = resp.raw.tell()
                bar.update(pos - last)
                last = pos
                if not member.isfile() or not member.name.endswith(".JPEG"):
                    continue
                parts = Path(member.name).parts  # imagewoof2-320/<split>/<wnid>/<file>
                split, wnid = parts[-3], parts[-2]
                name = "/".join(parts[-3:])
                if split == "train" and _seeded_hash(seed, name) % train_fraction_denominator:
                    continue
                if name in already:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                res = ingestor.ingest_bytes(
                    f.read(), collection="imagewoof", split=split, source_name=name,
                    label=IMAGEWOOF_CLASSES[wnid], extra={"wnid": wnid},
                    near_dup_policy=NearDupPolicy.KEEP)
                results[split].append(res)
            # Drain the rest of the stream so the checksum covers the whole file.
            while stream.read(1 << 20):
                pass

    if hasher.hexdigest() != IMAGEWOOF_SHA256:
        raise RuntimeError(f"Imagewoof archive checksum mismatch: {hasher.hexdigest()}")
    manifest = {
        "dataset": "imagewoof",
        "source": {"url": IMAGEWOOF_URL, "archive_sha256": hasher.hexdigest()},
        "seed": seed,
        "train_selection": f"sha256(seed:path) % {train_fraction_denominator} == 0",
        "classes": IMAGEWOOF_CLASSES,
        "splits": {s: summarize(r) for s, r in results.items()},
        "catalog_counts": {s: len(catalog.records(collection="imagewoof", split=s))
                           for s in ("train", "val")},
        "seconds": round(time.time() - t0, 1),
    }
    write_manifest(settings, "imagewoof", manifest)
    return manifest
