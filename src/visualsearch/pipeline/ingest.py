"""Data-level ingestion: validate, deduplicate, store and catalogue images.

Embedding and indexing are intentionally *not* done here. After images are
catalogued, :meth:`visualsearch.index.manager.IndexManager.sync` embeds whatever
each index is missing. This keeps ingestion cheap, makes index updates
incremental, and lets one ingestion feed any number of embedding models.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from visualsearch.config import Settings
from visualsearch.data.catalog import Catalog, ImageRecord
from visualsearch.data.preprocess import (
    PHASH_NEAR_DUP_BITS,
    DecodedImage,
    InvalidImageError,
    NearDuplicateIndex,
    decode_image,
    make_thumbnail,
    phash,
)

EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "GIF": "gif", "TIFF": "tiff"}


class NearDupPolicy(str, Enum):
    KEEP = "keep"  # store it, record the near-duplicate relation in `extra`
    SKIP = "skip"  # do not store it


class IngestStatus(str, Enum):
    ADDED = "added"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE_SKIPPED = "near_duplicate_skipped"
    INVALID = "invalid"


@dataclass
class IngestResult:
    status: IngestStatus
    image_id: int | None = None
    source_name: str | None = None
    detail: str | None = None
    near_duplicates: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "image_id": self.image_id,
                "source_name": self.source_name, "detail": self.detail,
                "near_duplicates": [{"image_id": i, "hamming": d} for i, d in self.near_duplicates]}


class Ingestor:
    def __init__(self, settings: Settings, catalog: Catalog,
                 near_dup_threshold: int = PHASH_NEAR_DUP_BITS):
        self.settings = settings
        self.catalog = catalog
        self._lock = threading.Lock()
        self.dups = NearDuplicateIndex(near_dup_threshold)
        for image_id, h in catalog.all_phashes():
            self.dups.add(image_id, h)
        settings.ensure_dirs()

    def ingest_bytes(
        self,
        data: bytes,
        *,
        collection: str,
        split: str | None = None,
        source_name: str | None = None,
        label: str | None = None,
        tags: list[str] | None = None,
        captions: list[tuple[int | None, str]] | None = None,
        extra: dict[str, Any] | None = None,
        near_dup_policy: NearDupPolicy = NearDupPolicy.KEEP,
    ) -> IngestResult:
        try:
            dec = decode_image(data, max_pixels=self.settings.max_image_pixels)
        except InvalidImageError as e:
            return IngestResult(IngestStatus.INVALID, source_name=source_name, detail=str(e))
        return self._ingest_decoded(dec, data, collection=collection, split=split,
                                    source_name=source_name, label=label, tags=tags or [],
                                    captions=captions or [], extra=extra or {},
                                    policy=near_dup_policy)

    def _ingest_decoded(self, dec: DecodedImage, data: bytes, *, collection: str,
                        split: str | None, source_name: str | None, label: str | None,
                        tags: list[str], captions: list[tuple[int | None, str]],
                        extra: dict[str, Any], policy: NearDupPolicy) -> IngestResult:
        h = phash(dec.image)
        # Serialise the check-then-insert so concurrent workers cannot both
        # insert the same content or miss each other's near duplicates.
        with self._lock:
            existing = self.catalog.by_sha(dec.sha256)
            if existing is not None:
                return IngestResult(IngestStatus.DUPLICATE, image_id=existing.id,
                                    source_name=source_name,
                                    detail=f"identical content already stored as image {existing.id}")
            near = self.dups.find(h)
            if near and policy is NearDupPolicy.SKIP:
                return IngestResult(IngestStatus.NEAR_DUPLICATE_SKIPPED, source_name=source_name,
                                    near_duplicates=near,
                                    detail=f"perceptual near-duplicate of image {near[0][0]}")
            rel_img = Path(collection) / dec.sha256[:2] / f"{dec.sha256}.{EXT[dec.format]}"
            rel_thumb = Path(collection) / dec.sha256[:2] / f"{dec.sha256}.jpg"
            img_path = self.settings.images_dir / rel_img
            thumb_path = self.settings.thumbs_dir / rel_thumb
            img_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(data)  # original bytes, untouched
            thumb_path.write_bytes(make_thumbnail(dec.image))
            if near:
                extra = {**extra, "near_duplicate_of": [{"image_id": i, "hamming": d} for i, d in near]}
            rec = ImageRecord(
                sha256=dec.sha256, phash=h, collection=collection, split=split,
                source_name=source_name, path=str(rel_img), thumb_path=str(rel_thumb),
                width=dec.width, height=dec.height, file_size=dec.file_size, format=dec.format,
                label=label, tags=tags, extra=extra,
            )
            with self.catalog.transaction():
                image_id = self.catalog.add_image(rec)
                if captions:
                    self.catalog.add_captions(image_id, captions)
            self.dups.add(image_id, h)
        return IngestResult(IngestStatus.ADDED, image_id=image_id, source_name=source_name,
                            near_duplicates=near)


def summarize(results: list[IngestResult]) -> dict[str, Any]:
    counts = Counter(r.status.value for r in results)
    return {
        "counts": dict(counts),
        "invalid": [r.to_dict() for r in results if r.status is IngestStatus.INVALID][:50],
        "duplicates": [r.to_dict() for r in results if r.status is IngestStatus.DUPLICATE][:200],
        "near_duplicates": [r.to_dict() for r in results if r.near_duplicates][:200],
    }
