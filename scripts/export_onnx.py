#!/usr/bin/env python
"""Export the query encoders used by the slim (no torch) server to ONNX.

    uv run --extra cpu --extra export python scripts/export_onnx.py
    uv run ... python scripts/export_onnx.py --fp32 dinov2-small   # keep one model in fp32

For each model this writes ``<out>/<model_key>/``:

  model.onnx       the query tower; int8 dynamic quantisation unless --fp32
  model.onnx.data  its weights (external data, memory-mapped by ONNX Runtime)
  embedder.json    fingerprint, dim, preprocessing and spec, computed by the torch
                   embedder, so existing indexes and calibrations stay valid
  tokenizer.json   (text towers) the fast tokenizer, with truncation and padding

Exported towers: CLIP -> text_model + text_projection (text queries only),
DINOv2 -> backbone with the spec's pooling (image queries). Weights come from
the pinned revisions in configs/models.yaml, loaded by the torch embedders.
Each export is checked against torch on a few inputs before it is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
import transformers
from PIL import Image

from visualsearch.config import get_settings
from visualsearch.embeddings import create_embedder, load_model_specs
from visualsearch.embeddings.onnx import (
    FORMAT_VERSION,
    META_FILE,
    MODEL_FILE,
    OnnxEmbedder,
    save_tokenizer,
)

OPSET = 17
# Nodes left in fp32 by the int8 export. Dynamic quantisation scales activations
# per tensor, and the input of CLIP's first text MLP fc2 has a few huge outlier
# channels: quantising that one MatMul alone drops the mean cosine to the fp32
# text embedding from 0.9999 to 0.77 (every other MatMul costs < 0.002).
KEEP_FP32 = {"clip": ["/text_model/encoder/layers.0/mlp/fc2/"]}
CHECK_TEXTS = ["a dog", "two people riding bicycles along a river at sunset",
               "Ein Hund läuft über die Wiese 🐕", ""]


class ClipTextTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        pooled = self.model.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask).pooler_output
        return self.model.text_projection(pooled)


class Dinov2Tower(torch.nn.Module):
    def __init__(self, model, pooling: str):
        super().__init__()
        self.model, self.pooling = model, pooling

    def forward(self, pixel_values):
        hidden = self.model(pixel_values=pixel_values).last_hidden_state
        cls, patches = hidden[:, 0], hidden[:, 1:]
        if self.pooling == "cls":
            return cls
        if self.pooling == "mean":
            return patches.mean(1)
        return torch.cat([cls, patches.mean(1)], dim=-1)


def sample_images(n: int = 3) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    sizes = [(320, 240), (240, 400), (500, 500)]
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            for w, h in sizes[:n]]


def export_clip_text(emb, path: Path) -> None:
    tok = emb.tokenizer(["a photo of a dog", "a cat"], padding=True, truncation=True,
                        max_length=77, return_tensors="pt")
    # Trace with a padded batch so the attention-mask path is part of the graph.
    torch.onnx.export(
        ClipTextTower(emb.model).eval(), (tok["input_ids"], tok["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["embeds"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "embeds": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False)


def export_dinov2(emb, path: Path) -> None:
    crop = emb.preprocessing_config()["crop_size"]
    dummy = torch.zeros(2, 3, int(crop["height"]), int(crop["width"]))
    torch.onnx.export(
        Dinov2Tower(emb.model, emb.spec.pooling).eval(), (dummy,), str(path),
        input_names=["pixel_values"], output_names=["embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeds": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False)


def compare(name: str, ref: np.ndarray, got: np.ndarray) -> dict:
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    got = got / np.linalg.norm(got, axis=1, keepdims=True)
    cos = (ref * got).sum(1)
    print(f"  {name}: min cosine to torch {cos.min():.6f}, max |diff| {np.abs(ref - got).max():.2e}")
    return {"min_cosine": float(cos.min()), "max_abs_diff": float(np.abs(ref - got).max())}


def export_model(key: str, spec, out_root: Path, fp32: bool, per_channel: bool) -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    emb = create_embedder(spec, device="cpu")
    emb.load()
    out_dir = out_root / key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    meta = {
        "format": FORMAT_VERSION, "key": key, "spec": asdict(spec),
        "fingerprint": emb.fingerprint, "dim": emb.dim,
        "preprocessing": emb.preprocessing_config(),
        "exported_with": {"torch": torch.__version__, "transformers": transformers.__version__,
                          "onnxruntime": ort.__version__},
    }
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "fp32.onnx"
        if spec.family == "clip":
            export_clip_text(emb, raw)
            meta["modalities"] = ["text"]
            meta["tokenizer"] = {**save_tokenizer(emb.tokenizer, out_dir),
                                 "source": f"{spec.hf_id}@{spec.revision}"}
        elif spec.family == "dinov2":
            export_dinov2(emb, raw)
            meta["modalities"] = ["image"]
        else:
            raise SystemExit(f"{key}: no ONNX export for family '{spec.family}'")
        if fp32:
            shutil.copy(raw, out_dir / MODEL_FILE)
            meta["onnx"] = {"file": MODEL_FILE, "opset": OPSET, "precision": "fp32",
                            "quantization": None}
        else:
            keep = [n.name for n in onnx.load(str(raw), load_external_data=False).graph.node
                    if any(p in n.name for p in KEEP_FP32.get(spec.family, []))]
            quantize_dynamic(raw, out_dir / MODEL_FILE, weight_type=QuantType.QInt8,
                             per_channel=per_channel, nodes_to_exclude=keep,
                             extra_options={"MatMulConstBOnly": True})
            meta["onnx"] = {"file": MODEL_FILE, "opset": OPSET, "precision": "int8",
                            "quantization": "dynamic, QInt8 weights"
                                            + (", per-channel" if per_channel else ", per-tensor"),
                            "fp32_nodes": keep}
    # Weights go to a side file that ONNX Runtime memory-maps instead of copying
    # out of the protobuf: about 40 MiB less resident memory for the CLIP tower.
    model = onnx.load(str(out_dir / MODEL_FILE))
    onnx.save_model(model, str(out_dir / MODEL_FILE), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=MODEL_FILE + ".data",
                    size_threshold=1024)
    (out_dir / (MODEL_FILE + ".data")).chmod(0o644)
    meta["onnx"]["data_file"] = MODEL_FILE + ".data"
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))

    # Check the export through the same class the server uses.
    slim = OnnxEmbedder(spec, out_dir, threads=4)
    assert slim.fingerprint == emb.fingerprint and slim.dim == emb.dim
    if "text" in meta["modalities"]:
        texts = CHECK_TEXTS + ["long " * 100]
        meta["check"] = compare("texts", emb.embed_texts(texts), slim.embed_texts(texts))
    else:
        imgs = sample_images()
        meta["check"] = compare("images", emb.embed_images(imgs), slim.embed_images(imgs))
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    size = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"{key}: {meta['onnx']['precision']} -> {out_dir} ({size / 2**20:.1f} MiB), "
          f"fingerprint {emb.fingerprint}", flush=True)
    del emb


def main() -> None:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # read when onnxruntime is imported
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=["clip-vit-b32", "dinov2-small"])
    ap.add_argument("--out", type=Path, default=settings.onnx_path)
    ap.add_argument("--fp32", nargs="*", default=[], metavar="MODEL",
                    help="models to keep in fp32 instead of int8")
    ap.add_argument("--per-channel", action="store_true", help="per-channel int8 weights")
    args = ap.parse_args()
    specs = load_model_specs(settings.models_path)
    torch.set_num_threads(4)
    for key in args.models:
        export_model(key, specs[key], args.out, key in args.fp32, args.per_channel)


if __name__ == "__main__":
    main()
