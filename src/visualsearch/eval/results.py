"""Writing machine-readable experiment results with full provenance."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from visualsearch.config import PROJECT_ROOT, Settings
from visualsearch.data.catalog import now_iso


def environment() -> dict[str, Any]:
    import faiss
    import torch
    import transformers

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "faiss": faiss.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        env["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "configs"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return env


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def dataset_provenance(settings: Settings) -> dict[str, Any]:
    out = {}
    for p in sorted((settings.root / "manifests").glob("*.json")):
        m = json.loads(p.read_text())
        out[p.stem] = {k: m.get(k) for k in ("source", "seed", "catalog_counts",
                                              "distractor_filenames_sha256") if k in m}
    return out


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def save_result(settings: Settings, name: str, data: dict[str, Any],
                report: dict[str, Any] | None = None) -> Path:
    out_dir = settings.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "generated_at": now_iso(),
        "seed": settings.seed,
        "environment": environment(),
        "datasets": dataset_provenance(settings),
        **data,
    }
    if report is not None:
        payload["report"] = report
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(_clean(payload), indent=2))
    return p


def table(title: str, columns: list[str], rows: list[list[Any]], notes: str = "",
          highlight_best: list[int] | None = None) -> dict[str, Any]:
    return {"title": title, "columns": columns, "rows": rows, "notes": notes,
            "highlight_best": highlight_best or []}


def chart(title: str, kind: str, x_label: str, y_label: str,
          series: dict[str, list[tuple[float, float]]], log_x: bool = False) -> dict[str, Any]:
    return {"title": title, "type": kind, "x_label": x_label, "y_label": y_label, "log_x": log_x,
            "series": [{"name": n, "points": [[float(x), float(y)] for x, y in pts]}
                       for n, pts in series.items()]}
