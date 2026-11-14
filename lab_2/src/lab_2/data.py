"""Stage 1: clean data loading and the A/B split (plan, section 1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lab_2.config import N_CLASSES, PER_CLASS_HALF, SPLIT_SEED


def load_clean(clean_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Loads the clean images and their labels.

    Rows are ordered by id so the result does not depend on filesystem order.

    Args:
        clean_dir: Directory containing `labels.csv` and an `images/` folder with one `<id>.png` per row.

    Returns:
        A tuple `(x, y, ids)` where `x` is a float32 tensor (N, 3, 64, 64) with values in [0, 1],
        `y` is a long tensor (N,) of class indices and `ids` is the list of image ids in the same order.
    """
    # ids are read as strings so that ids like "007" keep their leading zeros
    df = pd.read_csv(clean_dir / "labels.csv", dtype={"id": str}).sort_values("id").reset_index(drop=True)
    imgs = [np.asarray(Image.open(clean_dir / "images" / f"{i}.png").convert("RGB")) for i in df["id"]]
    # PIL gives (H, W, C) uint8; the models expect (C, H, W) floats in [0, 1]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    return x, y, df["id"].tolist()


def split_ab(labels: torch.Tensor, seed: int = SPLIT_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits the images into two class-balanced halves.

    PER_CLASS_HALF images per class go to A (trigger optimisation) and the next PER_CLASS_HALF to B
    (evaluation), so a trigger found on A is judged on images it has never seen.

    Args:
        labels: Long tensor (N,) of class indices.
        seed: Seed of the random permutation within each class.

    Returns:
        A tuple `(idx_a, idx_b)` of sorted index tensors into the image array.

    Raises:
        ValueError: If some class has fewer than `2 * PER_CLASS_HALF` images.
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
