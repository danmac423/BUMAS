"""Neural Cleanse anomaly index, submission CSV and AUC."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def anomaly_index(l1: np.ndarray) -> np.ndarray:
    med = np.median(l1)
    mad = 1.4826 * np.median(np.abs(l1 - med))
    return (med - l1) / (mad + EPS)


def score_cache(cache_dir: Path, calibrate: bool = False) -> pd.DataFrame:
    """Per-model anomaly index, or (calibrate) a population z-score of log l1 per class."""
    paths = sorted(Path(cache_dir).glob("*.npz"))
    l1 = np.stack([np.load(p)["l1"] for p in paths])  # (models, 20)
    if calibrate:
        z = np.log(l1 + EPS)
        scores = -(z - z.mean(0)) / (z.std(0) + EPS)
    else:
        scores = np.stack([anomaly_index(row) for row in l1])
    df = pd.DataFrame(scores, columns=[f"c{i:02d}" for i in range(20)])
    df.insert(0, "id", [p.stem for p in paths])
    return df


def write_csv(df: pd.DataFrame, out: Path) -> None:
    df.to_csv(out, index=False)


def check_csv(df: pd.DataFrame, models_dir: Path) -> None:
    ids = {p.stem for p in Path(models_dir).glob("*.pt")}
    assert len(df) == len(ids) and set(df["id"]) == ids and df["id"].is_unique, "ids do not match models"
    assert np.isfinite(df.drop(columns="id").to_numpy()).all(), "non-finite scores"


def auc(df: pd.DataFrame, labels_csv: Path) -> float:
    """Pooled ROC-AUC against a labels.csv with a `targets` column (`;`-separated)."""
    lab = pd.read_csv(labels_csv, dtype={"targets": str}).fillna({"targets": ""}).set_index("id")
    df = df[df["id"].isin(lab.index)]
    y, s = [], []
    for _, row in df.iterrows():
        targets = {int(t) for t in lab.loc[row["id"], "targets"].split(";") if t != ""}
        for c in range(20):
            y.append(int(c in targets))
            s.append(row[f"c{c:02d}"])
    return float(roc_auc_score(y, s))
