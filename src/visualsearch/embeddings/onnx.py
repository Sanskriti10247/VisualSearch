"""ONNX Runtime query encoders for serving without torch or transformers.

``scripts/export_onnx.py`` exports the query towers of a model from the same
pinned weights as the torch embedders, and writes next to each ONNX file an
``embedder.json`` holding the fingerprint, dimension and preprocessing config
computed by the torch embedder. :class:`OnnxEmbedder` reports those values, so
indexes and calibrations built with the torch models are reused unchanged.

Only the exported towers are available: an export of CLIP's text tower can
embed text queries but not images (``modalities`` comes from the export).

Image preprocessing reimplements the transformers *slow* (PIL) image
processor with PIL and numpy, step for step, so the pixel values match it
exactly; tokenisation uses the fast tokenizer's ``tokenizer.json`` through the
``tokenizers`` library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visualsearch.embeddings.base import (
    IMAGE,
    TEXT,
    Embedder,
    ModelSpec,
    UnsupportedModalityError,
    fingerprint,
)

META_FILE = "embedder.json"
MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
FORMAT_VERSION = 1

# Spec fields that must agree between configs/models.yaml and the export.
_SPEC_FIELDS = ("family", "hf_id", "revision", "pooling", "crop_mode", "normalize")
_SUPPORTED_PREPROC = {"do_resize", "size", "resample", "do_center_crop", "crop_size",
                      "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std",
                      "do_convert_rgb", "crop_mode", "processor", "pooling"}


class OnnxExportMismatch(RuntimeError):
    """The exported model does not belong to the configured model spec."""


# ------------------------------------------------------------ preprocessing
def _resize_shape(h: int, w: int, size: dict[str, int]) -> tuple[int, int]:
    if "shortest_edge" in size and len(size) == 1:
        # transformers.image_transforms.get_resize_output_image_size(default_to_square=False)
        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = size["shortest_edge"], int(size["shortest_edge"] * long / short)
        return (new_long, new_short) if w <= h else (new_short, new_long)
    if set(size) == {"height", "width"}:
        return size["height"], size["width"]
    raise ValueError(f"unsupported resize spec {size}")


def _center_crop(x: np.ndarray, ch: int, cw: int) -> np.ndarray:
    """transformers.image_transforms.center_crop on a CHW array (pads with 0 if too small)."""
    h, w = x.shape[-2:]
    top, left = (h - ch) // 2, (w - cw) // 2
    if top >= 0 and left >= 0 and top + ch <= h and left + cw <= w:
        return x[..., top:top + ch, left:left + cw]
    nh, nw = max(ch, h), max(cw, w)
    out = np.zeros(x.shape[:-2] + (nh, nw), dtype=x.dtype)
    tp, lp = -(-(nh - h) // 2), -(-(nw - w) // 2)
    out[..., tp:tp + h, lp:lp + w] = x
    top, left = top + tp, left + lp
    return out[..., max(0, top):min(nh, top + ch), max(0, left):min(nw, left + cw)]


def pixel_values(images: Sequence[Image.Image], cfg: dict[str, Any]) -> np.ndarray:
    """Batch of PIL images -> float32 [N, 3, H, W], identical to the HF slow
    image processor configured by ``cfg`` (its ``preprocessing_config()``)."""
    unknown = set(cfg) - _SUPPORTED_PREPROC
    if unknown:
        raise ValueError(f"unsupported preprocessing keys: {sorted(unknown)}")
    out = []
    for img in images:
        if cfg.get("do_convert_rgb", True) and img.mode != "RGB":
            img = img.convert("RGB")
        x = np.asarray(img)
        if x.ndim == 2:
            x = x[:, :, None]
        if cfg.get("do_resize"):
            h, w = _resize_shape(x.shape[0], x.shape[1], cfg["size"])
            pil = Image.fromarray(x[:, :, 0] if x.shape[2] == 1 else x)
            x = np.asarray(pil.resize((w, h), resample=int(cfg.get("resample", 2))))
            if x.ndim == 2:
                x = x[:, :, None]
        x = x.transpose(2, 0, 1)  # HWC -> CHW
        if cfg.get("do_center_crop"):
            x = _center_crop(x, int(cfg["crop_size"]["height"]), int(cfg["crop_size"]["width"]))
        if cfg.get("do_rescale"):
            x = (x.astype(np.float64) * cfg["rescale_factor"]).astype(np.float32)
        if cfg.get("do_normalize"):
            if not np.issubdtype(x.dtype, np.floating):
                x = x.astype(np.float32)
            mean = np.array(cfg["image_mean"], dtype=x.dtype)
            std = np.array(cfg["image_std"], dtype=x.dtype)
            x = ((x.T - mean) / std).T
        out.append(np.ascontiguousarray(x, dtype=np.float32))
    return np.stack(out)


# ------------------------------------------------------------------ tokenizer
def save_tokenizer(hf_tokenizer, out_dir: Path, max_length: int = 77) -> dict[str, Any]:
    """Write a transformers fast tokenizer's ``tokenizer.json``; returns its metadata."""
    hf_tokenizer.backend_tokenizer.save(str(Path(out_dir) / TOKENIZER_FILE))
    return {"file": TOKENIZER_FILE, "max_length": max_length,
            "pad_id": int(hf_tokenizer.pad_token_id), "pad_token": hf_tokenizer.pad_token}


def load_tokenizer(model_dir: Path, meta: dict[str, Any]):
    """``tokenizers.Tokenizer`` set up like ``tokenizer(texts, padding=True,
    truncation=True, max_length=...)`` in transformers."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(model_dir) / meta["file"]))
    tok.enable_truncation(max_length=int(meta["max_length"]))
    tok.enable_padding(pad_id=int(meta["pad_id"]), pad_token=meta["pad_token"])
    return tok


def tokenize(tokenizer, texts: list[str]) -> dict[str, np.ndarray]:
    enc = tokenizer.encode_batch(texts)
    return {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}


# ------------------------------------------------------------------ metadata
def read_meta(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"no ONNX export at {model_dir} (missing {META_FILE}); "
                                f"run scripts/export_onnx.py")
    meta = json.loads(path.read_text())
    if meta.get("format") != FORMAT_VERSION:
        raise OnnxExportMismatch(f"{path}: unsupported format {meta.get('format')}")
    return meta


def check_spec(spec: ModelSpec, meta: dict[str, Any]) -> None:
    exported = meta["spec"]
    diff = {f: (getattr(spec, f), exported.get(f)) for f in _SPEC_FIELDS
            if getattr(spec, f) != exported.get(f)}
    if diff:
        raise OnnxExportMismatch(f"ONNX export of '{spec.key}' was made from a different spec "
                                 f"{diff}; re-run scripts/export_onnx.py")


# ------------------------------------------------------------------ embedder
class OnnxEmbedder(Embedder):
    """Query encoder backed by an exported ONNX graph (CPU, no torch)."""

    def __init__(self, spec: ModelSpec, model_dir: str | Path, batch_size: int = 32,
                 threads: int = 1):
        super().__init__(spec, device="cpu", batch_size=batch_size)
        self.model_dir = Path(model_dir)
        self.threads = threads
        self.meta = read_meta(self.model_dir)
        check_spec(spec, self.meta)
        self.modalities = frozenset(self.meta["modalities"])
        computed = fingerprint(spec, self.preprocessing_config(), self.dim)
        if computed != self.meta["fingerprint"]:
            raise OnnxExportMismatch(
                f"ONNX export of '{spec.key}' has fingerprint {self.meta['fingerprint']}, but the "
                f"configured spec gives {computed}; re-run scripts/export_onnx.py")

    @property
    def dim(self) -> int:
        return int(self.meta["dim"])

    def preprocessing_config(self) -> dict[str, Any]:
        return dict(self.meta["preprocessing"])

    @property
    def fingerprint(self) -> str:
        return self.meta["fingerprint"]

    def describe(self) -> dict[str, Any]:
        onnx = self.meta.get("onnx", {})
        return {**super().describe(), "runtime": "onnx",
                "precision": onnx.get("precision"), "quantization": onnx.get("quantization")}

    # -- loading -----------------------------------------------------------
    def _prepare_runtime(self) -> None:
        pass  # nothing to configure: no torch

    def _load(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Small, bursty single-query batches: an arena would keep its high-water
        # mark allocated for the life of the process.
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_dir / self.meta["onnx"]["file"]),
                                            sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}
        if TEXT in self.modalities:
            self.tokenizer = load_tokenizer(self.model_dir, self.meta["tokenizer"])

    # -- inference -----------------------------------------------------------
    def pixel_values(self, images: Sequence[Image.Image]) -> np.ndarray:
        return pixel_values(images, self.preprocessing_config())

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if IMAGE not in self.modalities:
            raise UnsupportedModalityError(f"model '{self.key}' has no image encoder here")
        out = self.session.run(None, {"pixel_values": self.pixel_values(images)})[0]
        return np.asarray(out, dtype=np.float32)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        feed = {k: v for k, v in tokenize(self.tokenizer, texts).items()
                if k in self._input_names}
        return np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
