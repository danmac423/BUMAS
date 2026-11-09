"""
P2 Backdoors - data & model loading utilities.

Handles everything related to reading inputs from disk: picking a torch
device, reading class names, loading the clean calibration images, reading
train labels, listing available models, and importing the `load()` function
from the provided model.py.
"""

import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

C = 20  # number of classes


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_classes(root):
    with open(os.path.join(root, "classes.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def parse_class(tok, classes):
    tok = str(tok).strip()
    if tok.lstrip("-").isdigit():
        return int(tok)
    return classes.index(tok)


def load_clean(root, classes):
    """Loads the 400 clean images as a tensor (N,3,64,64) in [0,1] and labels (N,)."""
    df = pd.read_csv(os.path.join(root, "clean", "labels.csv"))
    img_dir = os.path.join(root, "clean", "images")
    files = sorted(os.listdir(img_dir))
    stem2file = {os.path.splitext(f)[0]: f for f in files}

    low = {c.lower(): c for c in df.columns}
    lcol = next((low[k] for k in ("label", "class", "y", "target", "labels") if k in low), df.columns[-1])
    fcol = next((low[k] for k in ("file", "filename", "image", "img", "path", "name", "id", "fname") if k in low), None)
    if fcol is None and len(df.columns) > 1:
        fcol = [c for c in df.columns if c != lcol][0]

    xs, ys = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        row = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        if fcol is not None:
            name = os.path.basename(str(row[fcol]))
            p = os.path.join(img_dir, name)
            if not os.path.exists(p):
                stem = os.path.splitext(name)[0]
                if stem not in stem2file:
                    raise FileNotFoundError(f"Image for '{name}' not found in {img_dir}")
                p = os.path.join(img_dir, stem2file[stem])
        else:  # no filename column -> alphabetical order
            p = os.path.join(img_dir, files[i])
        img = Image.open(p).convert("RGB")
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.BILINEAR)
        xs.append(np.asarray(img, dtype=np.float32) / 255.0)
        ys.append(parse_class(row[lcol], classes))

    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ys, dtype=torch.long)
    counts = torch.bincount(y, minlength=C)
    print(f"[clean] {len(y)} images, per class: min={counts.min().item()} max={counts.max().item()}")
    return x, y


def read_train_labels(root, classes):
    df = pd.read_csv(os.path.join(root, "train", "labels.csv"), dtype=str)  # str: avoid "15.0" from NaN-induced floats
    idcol = "id" if "id" in df.columns else df.columns[0]
    out = {}
    for _, r in df.iterrows():
        t = r.get("targets", "")
        if pd.isna(t) or str(t).strip() == "":
            out[str(r[idcol])] = []
        else:
            out[str(r[idcol])] = [parse_class(s, classes) for s in str(t).split(";") if s.strip() != ""]
    return out


def list_models(root, test_dir):
    items = []
    for split, base in (("train", os.path.join(root, "train")), ("test", test_dir)):
        for p in sorted(glob.glob(os.path.join(base, "models", "*.pt"))):
            items.append((split, os.path.splitext(os.path.basename(p))[0], p))
    return items


def load_model_fn(root):
    """Imports `load` from <root>/model.py by file path (independent of the working directory)."""
    spec = importlib.util.spec_from_file_location("p2_model", os.path.join(root, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load
