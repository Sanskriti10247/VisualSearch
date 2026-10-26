"""Image validation, normalisation, hashing and near-duplicate detection.

Model-specific preprocessing (resize/crop/normalise) is *not* done here: it
belongs to each embedder and is recorded in its fingerprint. This module only
does model-independent work that every image goes through once at ingestion:

* decode and validate (format allow-list, pixel limit, truncated files),
* apply EXIF orientation and convert to RGB,
* content hash (SHA-256 of the original bytes) for exact deduplication,
* 64-bit DCT perceptual hash for near-duplicate detection,
* a JPEG thumbnail for the UI.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}
THUMB_SIZE = 320
PHASH_NEAR_DUP_BITS = 6  # Hamming distance (out of 64) treated as a near duplicate


class InvalidImageError(ValueError):
    """Raised for uploads that are not decodable, allowed images."""


@dataclass
class DecodedImage:
    image: Image.Image  # RGB, EXIF-transposed
    sha256: str
    format: str
    width: int
    height: int
    file_size: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes, max_pixels: int = 40_000_000) -> DecodedImage:
    """Decode and validate raw bytes. Never trusts filename or Content-Type."""
    if not data:
        raise InvalidImageError("empty file")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImageError(f"unsupported image format: {fmt}")
            if w * h > max_pixels:
                raise InvalidImageError(f"image too large: {w}x{h} exceeds {max_pixels} pixels")
            probe.verify()  # structural check; the image must be reopened afterwards
        img = Image.open(io.BytesIO(data))
        img.load()  # fully decode; catches truncated data
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidImageError(f"could not decode image: {e}") from e
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)  # animated GIF/WebP: use the first frame
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            # Composite transparency onto white instead of black.
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    return DecodedImage(image=img, sha256=sha256_bytes(data), format=fmt,
                        width=img.width, height=img.height, file_size=len(data))


def load_rgb(path: str | Path) -> Image.Image:
    """Load an already-ingested image from disk as RGB."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


# ---------------------------------------------------------------- perceptual hash
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.cos(np.pi * k * (2 * i + 1) / (2 * n))


_DCT32 = _dct_matrix(32)


def phash(img: Image.Image) -> str:
    """64-bit DCT perceptual hash (same construction as the imagehash library).

    Grayscale 32x32 -> 2D DCT-II -> top-left 8x8 low frequencies -> bit = coeff > median.
    Returned as 16 hex chars.
    """
    g = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coeffs = _DCT32 @ g @ _DCT32.T
    low = coeffs[:8, :8]
    bits = (low > np.median(low)).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def phash_to_int(h: str) -> np.uint64:
    return np.uint64(int(h, 16))


def hamming_to_many(query: str, hashes: np.ndarray) -> np.ndarray:
    """Hamming distances between one hash and an array of uint64 hashes."""
    x = np.bitwise_xor(hashes, phash_to_int(query))
    return np.bitwise_count(x).astype(np.int32)


class NearDuplicateIndex:
    """Brute-force pHash index: O(n) per lookup, fine for 10^5 images."""

    def __init__(self, threshold: int = PHASH_NEAR_DUP_BITS):
        self.threshold = threshold
        self._ids: list[int] = []
        self._hashes: list[np.uint64] = []
        self._arr: np.ndarray | None = None

    def add(self, image_id: int, h: str) -> None:
        self._ids.append(int(image_id))
        self._hashes.append(phash_to_int(h))
        self._arr = None

    def find(self, h: str) -> list[tuple[int, int]]:
        """Return [(image_id, distance)] within the threshold, closest first."""
        if not self._ids:
            return []
        if self._arr is None:
            self._arr = np.array(self._hashes, dtype=np.uint64)
        d = hamming_to_many(h, self._arr)
        idx = np.nonzero(d <= self.threshold)[0]
        return sorted(((self._ids[i], int(d[i])) for i in idx), key=lambda t: t[1])


# -------------------------------------------------------------------- thumbnails
def make_thumbnail(img: Image.Image, size: int = THUMB_SIZE) -> bytes:
    t = img.copy()
    t.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()
