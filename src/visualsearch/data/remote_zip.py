"""Selective extraction of members from a large remote ZIP via HTTP range requests.

Flickr30k is distributed as one 4.4 GB ZIP. We only need a few thousand of
its 31,783 images, so instead of downloading the archive we:

1. read the central directory from the end of the file (a few MB, cached
   locally as JSON), and
2. fetch each selected member with a single ``Range`` request, inflate it and
   verify its CRC-32 against the central directory.

ZIP64 archives are supported because the central directory is parsed with the
standard library's :mod:`zipfile` over a seekable HTTP-backed file object.
"""

from __future__ import annotations

import io
import json
import struct
import threading
import time
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

_LOCAL_HEADER = struct.Struct("<4s5H3I2H")  # 30 bytes
_LOCAL_SIG = b"PK\x03\x04"
_MAX_LOCAL_EXTRA = 1024


@dataclass
class Member:
    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int
    crc: int


class _HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.session, self.pos = url, session, 0
        r = session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"},
                             timeout=120)
        r.raise_for_status()
        data = r.content
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZip:
    # Hub "resolve" URLs redirect to short-lived signed CDN URLs. Following the
    # redirect on every member costs an extra (rate-limited) round trip, so the
    # signed URL is resolved once and refreshed periodically or on 403.
    REFRESH_SECONDS = 20 * 60

    def __init__(self, url: str, index_cache: Path | None = None):
        self.url = url
        self.index_cache = index_cache
        self._local = threading.local()
        self._target: tuple[str, float] | None = None
        self._target_lock = threading.Lock()
        self.members: dict[str, Member] = self._load_index()

    def _resolved(self, force: bool = False) -> str:
        with self._target_lock:
            if force or self._target is None or time.time() - self._target[1] > self.REFRESH_SECONDS:
                r = self._session().get(self.url, headers={"Range": "bytes=0-0"},
                                        allow_redirects=True, timeout=60)
                r.raise_for_status()
                self._target = (r.url, time.time())
            return self._target[0]

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4,
                                                    max_retries=3)
            s.mount("https://", adapter)
            self._local.session = s
        return s

    def _load_index(self) -> dict[str, Member]:
        if self.index_cache and self.index_cache.exists():
            data = json.loads(self.index_cache.read_text())
            if data.get("url") == self.url:
                return {m["name"]: Member(**m) for m in data["members"]}
        raw = _HTTPRangeFile(self.url, self._session())
        with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 22)) as zf:
            members = {
                i.filename: Member(i.filename, i.header_offset, i.compress_size, i.file_size,
                                   i.compress_type, i.CRC)
                for i in zf.infolist() if not i.is_dir()
            }
        if self.index_cache:
            self.index_cache.parent.mkdir(parents=True, exist_ok=True)
            self.index_cache.write_text(json.dumps(
                {"url": self.url, "members": [asdict(m) for m in members.values()]}))
        return members

    def read(self, name: str, retries: int = 4) -> bytes:
        m = self.members[name]
        end = m.header_offset + _LOCAL_HEADER.size + len(name.encode()) + _MAX_LOCAL_EXTRA + m.compress_size
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                r = self._session().get(
                    self._resolved(force=attempt > 0),
                    headers={"Range": f"bytes={m.header_offset}-{end}"}, timeout=120)
                r.raise_for_status()
                return self._extract(m, r.content)
            except (requests.RequestException, ValueError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {name}: {last_err}")

    @staticmethod
    def _extract(m: Member, blob: bytes) -> bytes:
        sig, _ver, _flag, method, _t, _d, _crc, _cs, _us, nlen, xlen = _LOCAL_HEADER.unpack_from(blob)
        if sig != _LOCAL_SIG:
            raise ValueError("bad local header signature")
        start = _LOCAL_HEADER.size + nlen + xlen
        payload = blob[start:start + m.compress_size]
        if len(payload) != m.compress_size:
            raise ValueError("short read")
        if method == zipfile.ZIP_STORED:
            data = payload
        elif method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(payload, -15)
        else:
            raise ValueError(f"unsupported compression method {method}")
        if zlib.crc32(data) & 0xFFFFFFFF != m.crc:
            raise ValueError("CRC mismatch")
        return data
