import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from .model.resnet import resnet18_tin
from .scoring import BOX_COLS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent / "model"

_EPS = re.compile(r"eps=(\d+)/255")
_L2 = re.compile(r"linf=([\d.]+)/255")
_JPEG = re.compile(r"jpeg=(\d+)")
_DECOYS = re.compile(r"decoys: (\d+)")
_ATTACK = re.compile(r"ADV (\w+)")


def _parse_comment(c: str) -> dict:
    kind = c.split()[0].rstrip(";")
    attack = _ATTACK.search(c)
    head = c.split("decoys:")[0]
    eps = _EPS.search(head)
    linf = _L2.search(head)
    jpeg = _JPEG.search(c)
    n_decoys = _DECOYS.search(c)
    return {
        "kind": kind,
        "attack": attack.group(1) if attack else "none",
        "eps": float(eps.group(1)) if eps else (float(linf.group(1)) if linf else np.nan),
        "jpeg": int(jpeg.group(1)) if jpeg else -1,
        "n_decoys": int(n_decoys.group(1)) if n_decoys else 0,
    }


def load_labels(split: str = "train") -> pd.DataFrame:
    """Train labels plus parsed metadata. eps for C&W is the linf of the perturbation."""
    df = pd.read_csv(DATA / split / f"{split}_labels.csv")
    meta = pd.DataFrame([_parse_comment(c) for c in df.comment])
    df = pd.concat([df, meta], axis=1)
    df["has_patch"] = df.kind == "ADV"
    df["eps_bin"] = pd.cut(df.eps, [0, 4, 8, 12, 16, 21], labels=["<=4", "<=8", "<=12", "<=16", "<=20"])
    df["jpeg_bin"] = pd.cut(df.jpeg, [-2, 0, 90, 93, 100], labels=["none", "<=90", "91-93", "94-95"])
    return df


def load_test_ids() -> pd.DataFrame:
    return pd.read_csv(DATA / "test" / "test_ids.csv")


def load_image(split: str, id_: str) -> np.ndarray:
    """PNG -> float32 (64, 64, 3) in [0, 1], no normalization (the model does it inside)."""
    with Image.open(DATA / split / "images" / f"{id_}.png") as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def load_images(split: str, ids) -> torch.Tensor:
    """(N, 3, 64, 64) float32 tensor in [0, 1]."""
    arr = np.stack([load_image(split, i) for i in ids])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def get_device() -> str:
    """Best available accelerator: cuda, then mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str = "cpu") -> torch.nn.Module:
    model = resnet18_tin()
    model.load_state_dict(torch.load(MODEL_DIR / "resnet18_tin.pt", weights_only=True, map_location="cpu"))
    return model.eval().to(device)


def load_classes() -> list[str]:
    return (MODEL_DIR / "classes.txt").read_text().split()


def add_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Stratify by attack type (none/mifgsm/...) x decoy presence/kind."""
    strat = df.attack + "|" + df.kind + "|" + (df.n_decoys > 0).astype(str)
    df = df.copy()
    df["fold"] = -1
    for k, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, strat)):
        df.iloc[va, df.columns.get_loc("fold")] = k
    return df


def boxes(df: pd.DataFrame) -> np.ndarray:
    return df[BOX_COLS].to_numpy()
