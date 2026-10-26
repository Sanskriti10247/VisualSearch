"""SQLite-backed image catalog.

The catalog is the source of truth for image metadata. Integer ``images.id``
values are the external ids stored in every vector index, so a search result
can always be hydrated with metadata and filtered by it.

SQLite is a deliberate choice: the workload is read-heavy, single-node and
small (tens of thousands of rows). A single connection guarded by a lock is
simpler than a pool and fast enough here; WAL mode lets readers proceed while a
background ingestion job writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY,
    sha256       TEXT NOT NULL UNIQUE,
    phash        TEXT,
    collection   TEXT NOT NULL,
    split        TEXT,
    source_name  TEXT,
    path         TEXT NOT NULL,
    thumb_path   TEXT,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    orientation  TEXT NOT NULL,
    file_size    INTEGER NOT NULL,
    format       TEXT,
    label        TEXT,
    tags         TEXT NOT NULL DEFAULT '[]',
    extra        TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    deleted      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_images_collection ON images(collection);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);
CREATE INDEX IF NOT EXISTS idx_images_label ON images(label);

CREATE TABLE IF NOT EXISTS captions (
    id        INTEGER PRIMARY KEY,
    image_id  INTEGER NOT NULL REFERENCES images(id),
    sent_id   INTEGER,
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_captions_image ON captions(image_id);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    progress    INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT
);
"""

ORIENTATIONS = ("landscape", "portrait", "square")


def orientation_of(width: int, height: int, tol: float = 0.05) -> str:
    ratio = width / max(height, 1)
    if abs(ratio - 1.0) <= tol:
        return "square"
    return "landscape" if ratio > 1 else "portrait"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ImageRecord:
    sha256: str
    collection: str
    path: str
    width: int
    height: int
    file_size: int
    split: str | None = None
    source_name: str | None = None
    thumb_path: str | None = None
    phash: str | None = None
    format: str | None = None
    label: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: str | None = None
    deleted: bool = False

    @property
    def orientation(self) -> str:
        return orientation_of(self.width, self.height)

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection": self.collection,
            "split": self.split,
            "source_name": self.source_name,
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation,
            "file_size": self.file_size,
            "format": self.format,
            "label": self.label,
            "tags": self.tags,
            "sha256": self.sha256,
            "created_at": self.created_at,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class FilterSpec:
    """Metadata filter. Empty fields mean "no constraint"; lists are OR-ed,
    distinct fields are AND-ed."""

    collections: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    tags_any: list[str] = field(default_factory=list)
    orientations: list[str] = field(default_factory=list)
    min_width: int | None = None
    min_height: int | None = None
    exclude_ids: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.collections or self.splits or self.labels or self.tags_any
                    or self.orientations or self.min_width or self.min_height or self.exclude_ids)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FilterSpec":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known and v is not None})


def _row_to_record(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        id=row["id"], sha256=row["sha256"], phash=row["phash"], collection=row["collection"],
        split=row["split"], source_name=row["source_name"], path=row["path"],
        thumb_path=row["thumb_path"], width=row["width"], height=row["height"],
        file_size=row["file_size"], format=row["format"], label=row["label"],
        tags=json.loads(row["tags"]), extra=json.loads(row["extra"]),
        created_at=row["created_at"], deleted=bool(row["deleted"]),
    )


class Catalog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ images
    def add_image(self, rec: ImageRecord) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO images (sha256, phash, collection, split, source_name, path,
                       thumb_path, width, height, orientation, file_size, format, label, tags,
                       extra, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.sha256, rec.phash, rec.collection, rec.split, rec.source_name, rec.path,
                 rec.thumb_path, rec.width, rec.height, rec.orientation, rec.file_size,
                 rec.format, rec.label, json.dumps(rec.tags), json.dumps(rec.extra),
                 rec.created_at or now_iso()),
            )
            rec.id = int(cur.lastrowid)
            return rec.id

    def add_captions(self, image_id: int, captions: Sequence[tuple[int | None, str]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO captions (image_id, sent_id, text) VALUES (?,?,?)",
                [(image_id, sid, text) for sid, text in captions],
            )

    def transaction(self):
        """Context manager wrapping several writes in one transaction."""
        return _Txn(self)

    def get(self, image_id: int) -> ImageRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM images WHERE id=?", (int(image_id),)).fetchone()
        return _row_to_record(row) if row else None

    def get_many(self, ids: Iterable[int]) -> dict[int, ImageRecord]:
        ids = [int(i) for i in ids]
        out: dict[int, ImageRecord] = {}
        with self._lock:
            for chunk_start in range(0, len(ids), 900):
                chunk = ids[chunk_start:chunk_start + 900]
                q = f"SELECT * FROM images WHERE id IN ({','.join('?' * len(chunk))})"
                for row in self._conn.execute(q, chunk):
                    out[row["id"]] = _row_to_record(row)
        return out

    def by_sha(self, sha256: str) -> ImageRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM images WHERE sha256=?", (sha256,)).fetchone()
        return _row_to_record(row) if row else None

    def all_phashes(self) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, phash FROM images WHERE deleted=0 AND phash IS NOT NULL").fetchall()
        return [(r[0], r[1]) for r in rows]

    def records(self, collection: str | None = None, split: str | None = None,
                include_deleted: bool = False) -> list[ImageRecord]:
        clauses, params = [], []
        if not include_deleted:
            clauses.append("deleted=0")
        if collection:
            clauses.append("collection=?")
            params.append(collection)
        if split:
            clauses.append("split=?")
            params.append(split)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM images {where} ORDER BY id", params).fetchall()
        return [_row_to_record(r) for r in rows]

    def ids(self, include_deleted: bool = False) -> np.ndarray:
        q = "SELECT id FROM images" + ("" if include_deleted else " WHERE deleted=0") + " ORDER BY id"
        with self._lock:
            return np.array([r[0] for r in self._conn.execute(q)], dtype=np.int64)

    def mark_deleted(self, image_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE images SET deleted=1 WHERE id=? AND deleted=0",
                                     (int(image_id),))
        return cur.rowcount > 0

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM images WHERE deleted=0").fetchone()[0])

    def captions_for(self, image_id: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT text FROM captions WHERE image_id=? ORDER BY sent_id, id",
                (int(image_id),)).fetchall()
        return [r[0] for r in rows]

    def captions(self, collection: str | None = None, split: str | None = None
                 ) -> list[tuple[int, int, str]]:
        """Return (caption_id, image_id, text) for non-deleted images."""
        clauses, params = ["i.deleted=0"], []
        if collection:
            clauses.append("i.collection=?")
            params.append(collection)
        if split:
            clauses.append("i.split=?")
            params.append(split)
        q = (f"SELECT c.id, c.image_id, c.text FROM captions c JOIN images i ON i.id=c.image_id "
             f"WHERE {' AND '.join(clauses)} ORDER BY c.image_id, c.sent_id, c.id")
        with self._lock:
            return [(r[0], r[1], r[2]) for r in self._conn.execute(q, params)]

    # ----------------------------------------------------------------- filters
    def resolve_filter(self, spec: FilterSpec) -> np.ndarray | None:
        """Return the sorted ids allowed by ``spec``, or None when unconstrained."""
        if spec.is_empty():
            return None
        clauses, params = ["deleted=0"], []

        def _in(col: str, values: list[Any]) -> None:
            clauses.append(f"{col} IN ({','.join('?' * len(values))})")
            params.extend(values)

        if spec.collections:
            _in("collection", spec.collections)
        if spec.splits:
            _in("split", spec.splits)
        if spec.labels:
            _in("label", spec.labels)
        if spec.orientations:
            _in("orientation", spec.orientations)
        if spec.min_width:
            clauses.append("width>=?")
            params.append(int(spec.min_width))
        if spec.min_height:
            clauses.append("height>=?")
            params.append(int(spec.min_height))
        if spec.tags_any:
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(images.tags) t WHERE t.value IN "
                f"({','.join('?' * len(spec.tags_any))}))")
            params.extend(spec.tags_any)
        q = f"SELECT id FROM images WHERE {' AND '.join(clauses)} ORDER BY id"
        with self._lock:
            ids = np.array([r[0] for r in self._conn.execute(q, params)], dtype=np.int64)
        if spec.exclude_ids:
            ids = np.setdiff1d(ids, np.asarray(spec.exclude_ids, dtype=np.int64))
        return ids

    def facets(self) -> dict[str, Any]:
        """Distinct values usable as filters (for the UI)."""
        with self._lock:
            def distinct(col: str) -> list[dict[str, Any]]:
                rows = self._conn.execute(
                    f"SELECT {col}, COUNT(*) FROM images WHERE deleted=0 AND {col} IS NOT NULL "
                    f"GROUP BY {col} ORDER BY COUNT(*) DESC").fetchall()
                return [{"value": r[0], "count": r[1]} for r in rows]

            tags = self._conn.execute(
                "SELECT t.value, COUNT(*) FROM images, json_each(images.tags) t "
                "WHERE deleted=0 GROUP BY t.value ORDER BY COUNT(*) DESC LIMIT 100").fetchall()
            return {
                "collections": distinct("collection"),
                "splits": distinct("split"),
                "labels": distinct("label"),
                "orientations": distinct("orientation"),
                "tags": [{"value": r[0], "count": r[1]} for r in tags],
                "total": self.count(),
            }

    # -------------------------------------------------------------------- jobs
    def create_job(self, job_id: str, kind: str, total: int) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, created_at, updated_at, total) "
                "VALUES (?,?,?,?,?,?)", (job_id, kind, "queued", ts, ts, total))

    def update_job(self, job_id: str, **fields: Any) -> None:
        if "result" in fields and not isinstance(fields["result"], str):
            fields["result"] = json.dumps(fields["result"])
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d


class _Txn:
    def __init__(self, cat: Catalog):
        self.cat = cat

    def __enter__(self):
        self.cat._lock.acquire()
        self.cat._conn.execute("BEGIN")
        return self.cat

    def __exit__(self, exc_type, *_):
        try:
            self.cat._conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.cat._lock.release()
