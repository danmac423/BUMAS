"""Calibration of the cached raw statistics into comparable suspicion scores, plus the CSV writer.

Per form the raw statistic is the log trigger size at the ASR target (plus a small penalty when the
held-out ASR fell short). It is standardised per (class, form) with a robust z-score against the
*population* of models, so scores are comparable across models and a clean model has no artificial
"outlier" class. The classic within-model Neural Cleanse MAD anomaly index is added as an extra feature.
The final score of a (model, class) pair is the max over forms and the targeted scan.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from .config import Config
from .data import N_CLASSES, list_models, read_train_labels

MAD_K = 1.4826
FORM_NAMES = ("patch", "blend", "warp")


@dataclass
class Stats:
    """Raw statistics of M models stacked into arrays."""

    ids: list[str]
    feat: dict[str, np.ndarray] = field(default_factory=dict)  # form -> (M,20) log-size feature
    tgt: np.ndarray | None = None  # (M,20,20) [source, target] log-size feature, NaN if not scanned
    leak: np.ndarray | None = None  # (M,20,20)


def build_stats(cfg: Config, cache_dir: str | Path, ids: list[str] | None = None) -> Stats:
    ids = ids or store.ids_in(cache_dir)
    raws = [store.load(cache_dir, i) for i in ids]
    st = Stats(ids)
    for form in FORM_NAMES:
        if all(f"{form}_size" in r for r in raws):
            size = np.stack([r[f"{form}_size"] for r in raws]).astype(np.float64)
            asr = np.stack([r[f"{form}_asr"] for r in raws]).astype(np.float64)
            st.feat[form] = np.log(np.maximum(size, 1e-9)) + cfg.w_asr * np.maximum(0.0, cfg.asr_target - asr)
    if all("tgt_size" in r for r in raws):
        size = np.stack([r["tgt_size"] for r in raws]).astype(np.float64)
        asr = np.stack([r["tgt_asr"] for r in raws]).astype(np.float64)
        st.tgt = np.log(np.maximum(size, 1e-9)) + cfg.w_asr * np.maximum(0.0, cfg.asr_target - asr)
        st.leak = np.stack([r["tgt_leak"] for r in raws]).astype(np.float64)
    return st


def _med_mad(x: np.ndarray, axis: int, floor: float) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices = pairs that were never scanned
        med = np.nanmedian(x, axis=axis)
        mad = MAD_K * np.nanmedian(np.abs(x - np.expand_dims(med, axis)), axis=axis)
    return med, np.maximum(np.nan_to_num(mad, nan=floor), floor)


@dataclass
class Calib:
    med: dict[str, np.ndarray]  # form -> (20,)
    mad: dict[str, np.ndarray]
    tgt_med: np.ndarray | None = None  # (20,20)
    tgt_mad: np.ndarray | None = None


def calibrate(cfg: Config, ref: Stats) -> Calib:
    cal = Calib({}, {})
    for form, f in ref.feat.items():
        cal.med[form], cal.mad[form] = _med_mad(f, 0, cfg.mad_floor)
    if ref.tgt is not None:
        # Pairs are picked per model (not all scanned), so a per-pair population would be tiny and biased:
        # pool every scanned pair of a target (all models, all sources).
        pooled = ref.tgt.transpose(2, 0, 1).reshape(N_CLASSES, -1)  # (target, models x sources)
        pm, pmad = _med_mad(pooled, 1, cfg.mad_floor)  # (target,)
        cal.tgt_med = np.broadcast_to(pm[None, :], (N_CLASSES, N_CLASSES)).copy()
        cal.tgt_mad = np.broadcast_to(pmad[None, :], (N_CLASSES, N_CLASSES)).copy()
    return cal


def nc_index(f: np.ndarray, floor: float) -> np.ndarray:
    """Classic Neural Cleanse anomaly index inside every model: (median - f) / MAD over the 20 classes."""
    med = np.median(f, axis=1, keepdims=True)
    mad = np.maximum(MAD_K * np.median(np.abs(f - med), axis=1, keepdims=True), floor)
    return (med - f) / mad


def score(cfg: Config, st: Stats, cal: Calib) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns the (M,20) final suspicion scores and the per-component scores."""
    comps: dict[str, np.ndarray] = {}
    for form, f in st.feat.items():
        z = (cal.med[form][None] - f) / cal.mad[form][None]
        comps[form] = z + cfg.w_nc * nc_index(f, cfg.mad_floor)
    if st.tgt is not None and cal.tgt_med is not None:
        z = (cal.tgt_med[None] - st.tgt) / cal.tgt_mad[None] - cfg.w_leak * np.nan_to_num(st.leak)
        with np.errstate(all="ignore"):
            zt = np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=1)  # max over sources -> (M,tgt)
        zt = np.where(np.isfinite(zt), zt, np.nan)
        comps["targeted"] = cfg.w_tgt * zt
    stack = np.stack(list(comps.values()))  # (K,M,20)
    allnan = np.all(np.isnan(stack), axis=0)
    final = np.nanmax(np.where(np.isnan(stack), -np.inf, stack), axis=0)
    final = np.where(allnan | ~np.isfinite(final), np.nan, final)
    floor = np.nanmin(final) if np.isfinite(final).any() else 0.0
    final = np.where(np.isfinite(final), final, floor - 1.0)
    return final, comps


def write_csv(path: str | Path, ids: list[str], s: np.ndarray) -> None:
    df = pd.DataFrame(s, columns=[f"c{c:02d}" for c in range(N_CLASSES)])
    df.insert(0, "id", ids)
    df.to_csv(path, index=False, float_format="%.6f")


def validate_csv(path: str | Path, expected_ids: list[str] | None) -> None:
    df = pd.read_csv(path, dtype={"id": str})
    assert list(df.columns) == ["id"] + [f"c{c:02d}" for c in range(N_CLASSES)], "bad columns"
    assert df["id"].is_unique, "duplicate ids"
    assert np.isfinite(df.iloc[:, 1:].to_numpy(dtype=float)).all(), "non-finite values"
    if expected_ids is not None:
        assert set(df["id"]) == set(expected_ids), "ids do not match the model files"


def pooled_auc(s: np.ndarray, pos: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(pos.ravel().astype(int), s.ravel()))


def run_score(
    cfg: Config, cache: str, reference: str, out: str, models_dir: str | None, labels: str | None
) -> None:
    expected = [p.stem for p in list_models(models_dir)] if models_dir else None
    ids = store.ids_in(cache)
    if expected is not None:
        missing = sorted(set(expected) - set(ids))
        if missing:
            raise SystemExit(f"{len(missing)} models are not in the cache yet, e.g. {missing[:3]}")
        ids = sorted(expected)
    st = build_stats(cfg, cache, ids)
    ref = st if reference == cache else build_stats(cfg, reference)
    cal = calibrate(cfg, ref)
    s, comps = score(cfg, st, cal)
    write_csv(out, st.ids, s)
    validate_csv(out, expected)
    print(f"wrote {out}: {len(st.ids)} models, components: {', '.join(comps)}")

    if labels:
        lab = read_train_labels(labels)
        pos = np.zeros_like(s, dtype=bool)
        for i, mid in enumerate(st.ids):
            for t in lab.get(mid, []):
                pos[i, t] = True
        print(f"pooled AUC (final): {pooled_auc(s, pos):.3f}")
        for name, c in comps.items():
            c = np.where(np.isfinite(c), c, np.nanmin(c) - 1)
            print(f"  AUC {name:9s}: {pooled_auc(c, pos):.3f}")
        for i, mid in enumerate(st.ids):
            order = np.argsort(-s[i])
            ranks = [int(np.flatnonzero(order == t)[0]) + 1 for t in lab.get(mid, [])]
            print(f"  {mid} targets={lab.get(mid, [])} ranks={ranks} max_score={s[i].max():.2f} top3={order[:3].tolist()}")
