"""Stage 1: clean data loading and the A/B split (plan, section 1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lab_2.config import N_CLASSES, PER_CLASS_HALF, SPLIT_SEED


def load_clean(clean_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return images (N, 3, 64, 64) float32 in [0, 1], labels (N,), and image ids.

    Rows are ordered by id so the result does not depend on filesystem order.
    """
    df = pd.read_csv(clean_dir / "labels.csv", dtype={"id": str}).sort_values("id").reset_index(drop=True)
    imgs = [np.asarray(Image.open(clean_dir / "images" / f"{i}.png").convert("RGB")) for i in df["id"]]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    return x, y, df["id"].tolist()


def split_ab(labels: torch.Tensor, seed: int = SPLIT_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified split: PER_CLASS_HALF images per class go to A (optimisation), the rest to B (evaluation).

    Returns sorted index tensors (idx_a, idx_b).
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(labels.numpy() == c)
        if len(idx) < 2 * PER_CLASS_HALF:
            raise ValueError(f"class {c} has {len(idx)} images, need {2 * PER_CLASS_HALF}")
        idx = rng.permutation(idx)
        a.extend(idx[:PER_CLASS_HALF])
        b.extend(idx[PER_CLASS_HALF : 2 * PER_CLASS_HALF])
    return torch.tensor(sorted(a)), torch.tensor(sorted(b))
