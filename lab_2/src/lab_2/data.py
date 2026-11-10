"""Loading of clean images, the fixed optimisation/held-out split and the models."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

N_CLASSES = 20


@dataclass
class CleanData:
    x: torch.Tensor  # (N,3,64,64) float32 in [0,1], on device
    y: torch.Tensor  # (N,) long, on device
    opt_idx: torch.Tensor  # indices of the optimisation split
    held_idx: torch.Tensor  # indices of the held-out split


def load_clean(clean_dir: str | Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    clean_dir = Path(clean_dir)
    df = pd.read_csv(clean_dir / "labels.csv", dtype={"id": str})
    imgs = []
    for i in df["id"]:
        im = Image.open(clean_dir / "images" / f"{i}.png").convert("RGB")
        imgs.append(np.asarray(im, dtype=np.uint8))
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255.0)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    return x.to(device), y.to(device)


def stratified_split(y: np.ndarray, held_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed split, stratified by class. The same split is used for every model."""
    rng = np.random.default_rng(seed)
    opt, held = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_held = max(1, int(round(len(idx) * held_frac)))
        held.extend(idx[:n_held])
        opt.extend(idx[n_held:])
    return np.sort(np.array(opt)), np.sort(np.array(held))


def load_data(clean_dir: str | Path, device: str, held_frac: float, seed: int) -> CleanData:
    x, y = load_clean(clean_dir, device)
    opt, held = stratified_split(y.cpu().numpy(), held_frac, seed)
    return CleanData(x, y, torch.from_numpy(opt).to(device), torch.from_numpy(held).to(device))


_MODEL_MODULE = None


def _model_module(model_py: str):
    global _MODEL_MODULE
    if _MODEL_MODULE is None:
        spec = importlib.util.spec_from_file_location("p2_model", model_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODEL_MODULE = mod
    return _MODEL_MODULE


def load_net(path: str | Path, model_py: str, device: str) -> torch.nn.Module:
    """Load a model, freeze it and switch to channels_last."""
    net = _model_module(model_py).load(str(path), device=device)
    net.eval().requires_grad_(False)
    return net.to(memory_format=torch.channels_last)


def list_models(models_dir: str | Path) -> list[Path]:
    return sorted(Path(models_dir).glob("*.pt"))


def read_train_labels(path: str | Path) -> dict[str, list[int]]:
    """id -> list of target classes (empty list = clean model)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        out[r["id"]] = [int(t) for t in r["targets"].split(";") if t != ""]
    return out
