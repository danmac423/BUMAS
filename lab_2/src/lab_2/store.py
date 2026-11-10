"""Per-model raw-statistics cache (one .npz per model) so scoring can be re-tuned without rescanning."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def cache_path(cache_dir: str | Path, model_id: str) -> Path:
    return Path(cache_dir) / f"{model_id}.npz"


def save(cache_dir: str | Path, model_id: str, res: dict[str, np.ndarray]) -> None:
    p = cache_path(cache_dir, model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    tmp.replace(p)  # atomic: an interrupted run never leaves a half-written cache entry


def load(cache_dir: str | Path, model_id: str) -> dict[str, np.ndarray]:
    with np.load(cache_path(cache_dir, model_id)) as z:
        return {k: z[k] for k in z.files}


def ids_in(cache_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(cache_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))
