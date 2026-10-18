import io

import pytest
from PIL import Image

from tests.conftest import make_image
from visualsearch.data.preprocess import (
    InvalidImageError,
    NearDuplicateIndex,
    decode_image,
    make_thumbnail,
    phash,
)


def test_decode_valid_jpeg_and_png():
    for fmt in ("JPEG", "PNG", "WEBP"):
        d = decode_image(make_image(fmt=fmt))
        assert d.format == fmt and d.image.mode == "RGB" and (d.width, d.height) == (64, 48)
        assert len(d.sha256) == 64


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"%PDF-1.4 fake pdf",
                                  b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"])
def test_decode_rejects_non_images(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_decode_rejects_truncated_jpeg():
    data = make_image(size=(256, 256), noise=40)
    with pytest.raises(InvalidImageError):
        decode_image(data[: len(data) // 2])


def test_decode_enforces_pixel_limit():
    with pytest.raises(InvalidImageError, match="too large"):
        decode_image(make_image(size=(200, 200)), max_pixels=100 * 100)


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (60, 30), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    d = decode_image(buf.getvalue())
    assert (d.width, d.height) == (30, 60)


def test_transparency_composited_on_white():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    d = decode_image(buf.getvalue())
    assert d.image.getpixel((5, 5)) == (255, 255, 255)


def test_phash_stable_under_resize_and_recompression():
    d = decode_image(make_image(size=(256, 192), seed=3))
    h1 = phash(d.image)
    resized = d.image.resize((128, 96))
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=60)
    h2 = phash(Image.open(io.BytesIO(buf.getvalue())))
    idx = NearDuplicateIndex(threshold=6)
    idx.add(1, h1)
    assert idx.find(h2) and idx.find(h2)[0][0] == 1


def test_phash_differs_for_different_images():
    a = phash(decode_image(make_image(seed=1)).image)
    b = phash(decode_image(make_image(seed=2)).image)
    idx = NearDuplicateIndex(threshold=6)
    idx.add(1, a)
    assert idx.find(b) == []


def test_thumbnail_is_bounded_jpeg():
    d = decode_image(make_image(size=(1000, 500)))
    t = Image.open(io.BytesIO(make_thumbnail(d.image, size=320)))
    assert t.format == "JPEG" and max(t.size) == 320 and t.size[0] / t.size[1] == pytest.approx(2, rel=0.01)
