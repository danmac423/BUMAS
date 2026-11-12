"""Neural Cleanse anomaly index, submission CSV and AUC.

Every scoring method is a function `l1 -> scores`, registered in `SCORERS`.
`l1` has shape (models, N_CLASSES); larger score means more suspicious.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6
MAD_SCALE = 1.4826  # makes MAD a consistent estimator of the std for normal data
N_CLASSES = 20
CLASS_COLS = [f"c{i:02d}" for i in range(N_CLASSES)]

Scorer = Callable[[np.ndarray], np.ndarray]


# --- statistics -------------------------------------------------------------


def _mad(x: np.ndarray, loc: np.ndarray, axis: int | None) -> np.ndarray:
    return MAD_SCALE * np.median(np.abs(x - loc), axis=axis, keepdims=True)


def anomaly_index(l1: np.ndarray) -> np.ndarray:
    """Neural Cleanse anomaly index, computed within each model (row). Accepts (..., classes)."""
    med = np.median(l1, axis=-1, keepdims=True)
    return (med - l1) / (_mad(l1, med, axis=-1) + EPS)


def _population_z(l1: np.ndarray, *, robust: bool, per_class: bool) -> np.ndarray:
    """Negated z-score of log l1 against the whole population (low norm -> high score)."""
    z = np.log(l1 + EPS)
    axis = 0 if per_class else None
    if robust:
        loc = np.median(z, axis=axis, keepdims=True)
        scale = _mad(z, loc, axis)
    else:
        loc, scale = z.mean(axis=axis, keepdims=True), z.std(axis=axis, keepdims=True)
    return -(z - loc) / (scale + EPS)


# --- methods ----------------------------------------------------------------


def _population(*, robust: bool, per_class: bool) -> Scorer:
    return lambda l1: _population_z(l1, robust=robust, per_class=per_class)


def _mix(l1: np.ndarray) -> np.ndarray:
    return (SCORERS["robust"](l1) + anomaly_index(l1)) / 2


SCORERS: dict[str, Scorer] = {
    "nc": anomaly_index,
    "class": _population(robust=False, per_class=True),
    "global": _population(robust=False, per_class=False),
    "robust": _population(robust=True, per_class=True),
    "robust_global": _population(robust=True, per_class=False),
    "mix": _mix,
}
METHODS = tuple(SCORERS)


# --- I/O --------------------------------------------------------------------


def _load(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    paths = sorted(Path(cache_dir).glob("*.npz"))
    return [p.stem for p in paths], np.stack([np.load(p)["l1"] for p in paths])  # (models, 20)


def score_cache(cache_dir: Path, method: str = "nc") -> pd.DataFrame:
    """Scores per model and class; larger means more suspicious.

    nc: per-model Neural Cleanse anomaly index.
    class / global: population z-score of log l1, per class or over all pairs.
    robust / robust_global: the same with median and MAD.
    mix: mean of the robust per-class z-score and the nc index.
    The population statistics come from `cache_dir` itself.
    """
    ids, l1 = _load(cache_dir)
    df = pd.DataFrame(SCORERS[method](l1), columns=CLASS_COLS)
    df.insert(0, "id", ids)
    return df


def write_csv(df: pd.DataFrame, out: Path) -> None:
    df.to_csv(out, index=False)


def check_csv(df: pd.DataFrame, models_dir: Path) -> None:
    ids = {p.stem for p in Path(models_dir).glob("*.pt")}
    assert len(df) == len(ids) and set(df["id"]) == ids and df["id"].is_unique, "ids do not match models"
    assert np.isfinite(df[CLASS_COLS].to_numpy()).all(), "non-finite scores"


def auc(df: pd.DataFrame, labels_csv: Path) -> float:
    """Pooled ROC-AUC against a labels.csv with a `targets` column (`;`-separated)."""
    lab = pd.read_csv(labels_csv, dtype={"targets": str}).fillna({"targets": ""}).set_index("id")
    df = df[df["id"].isin(lab.index)]
    y = np.zeros((len(df), N_CLASSES), dtype=int)
    for row, model_id in enumerate(df["id"]):
        for t in lab.loc[model_id, "targets"].split(";"):
            if t != "":
                y[row, int(t)] = 1
    return float(roc_auc_score(y.ravel(), df[CLASS_COLS].to_numpy().ravel()))
