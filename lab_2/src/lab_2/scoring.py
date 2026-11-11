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


METHODS = ("nc", "class", "global", "robust", "robust_global", "mix")


def _load(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    paths = sorted(Path(cache_dir).glob("*.npz"))
    return [p.stem for p in paths], np.stack([np.load(p)["l1"] for p in paths])  # (models, 20)


def _loc_scale(z: np.ndarray, robust: bool, per_class: bool) -> tuple[np.ndarray, np.ndarray]:
    axis = 0 if per_class else None
    if robust:
        loc = np.median(z, axis=axis)
        scale = 1.4826 * np.median(np.abs(z - loc), axis=axis)
    else:
        loc, scale = z.mean(axis=axis), z.std(axis=axis)
    return loc, scale + EPS


def score_cache(cache_dir: Path, method: str = "nc", ref_cache: Path | None = None) -> pd.DataFrame:
    """Scores per model and class; larger means more suspicious.

    nc: per-model Neural Cleanse anomaly index.
    class / global: population z-score of log l1, per class or over all pairs.
    robust / robust_global: the same with median and MAD.
    mix: mean of the robust per-class z-score and the nc index.
    The population statistics come from `ref_cache` if given, else from `cache_dir` itself.
    """
    ids, l1 = _load(cache_dir)
    z = np.log(l1 + EPS)
    nc = np.stack([anomaly_index(row) for row in l1])
    if method == "nc":
        scores = nc
    else:
        ref = z if ref_cache is None else np.log(_load(ref_cache)[1] + EPS)
        robust = method in ("robust", "robust_global", "mix")
        per_class = method in ("class", "robust", "mix")
        loc, scale = _loc_scale(ref, robust, per_class)
        scores = -(z - loc) / scale
        if method == "mix":
            scores = (scores + nc) / 2
    df = pd.DataFrame(scores, columns=[f"c{i:02d}" for i in range(20)])
    df.insert(0, "id", ids)
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
