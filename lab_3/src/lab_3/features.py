"""Interpretable attack features. None of them reads the ``member`` label."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from lab_3.records import Record

SOURCE_TYPES = ("writer", "research group", "podcast", "photographer", "forum post", "web photo")
# Types where one owner (``source``) has many records.
SET_TYPES = frozenset({"writer", "research group", "podcast", "photographer"})

Losses = Mapping[str, Mapping[str, Any]]

# Signals of a single record, each standardised within its source_type.
RECORD_FEATURES = ("z_sum_delta", "z_bottom_delta", "z_relative_delta", "z_neg_bumas_loss", "z_ref_loss")


def delta(records: list[Record], losses: Losses) -> np.ndarray:
    """How much better BUMAS-mini knows each record than the reference model does."""
    return np.array([losses[r["id"]]["ref_mean"] - losses[r["id"]]["bumas_mean"] for r in records], dtype=float)


def zscore_by_type(values: np.ndarray, records: list[Record]) -> np.ndarray:
    """Standardise values separately within every ``source_type``.

    A group with zero spread gets zeros.
    """
    types = np.array([r["source_type"] for r in records])
    out = np.zeros(len(values))
    for t in np.unique(types):
        mask = types == t
        std = values[mask].std()
        if std > 0:
            out[mask] = (values[mask] - values[mask].mean()) / std
    return out


def record_signals(records: list[Record], losses: Losses) -> dict[str, np.ndarray]:
    """Raw (not yet standardised) summaries of the per-token losses.

    With ``d = ref_tok - bumas_tok`` for every token:

    * ``z_sum_delta``: ``sum(d) / sqrt(n)``, so the same mean gain counts more on a longer record,
    * ``z_bottom_delta``: mean of the lower half of ``d``,
    * ``z_relative_delta``: gain of the mean loss as a fraction of the reference loss,
    * ``z_neg_bumas_loss``: the LOSS attack,
    * ``z_ref_loss``: how hard the text is for the reference model.
    """
    rows = []
    for r in records:
        bumas = np.asarray(losses[r["id"]]["bumas_tok"], dtype=float)
        ref = np.asarray(losses[r["id"]]["ref_tok"], dtype=float)
        d = ref - bumas
        n = len(d)
        rows.append(
            (
                d.sum() / np.sqrt(n),
                np.sort(d)[: (n + 1) // 2].mean(),
                d.mean() / (ref.mean() + 1e-6),
                -bumas.mean(),
                ref.mean(),
            )
        )
    return dict(zip(RECORD_FEATURES, np.array(rows).T, strict=True))


def owner_aggregate(z: np.ndarray, records: list[Record], statistic: Callable[[np.ndarray], Any]) -> np.ndarray:
    """A statistic of ``z`` over all records of the same owner; zero for types with one record per owner."""
    members: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(records):
        if r["source_type"] in SET_TYPES:
            members.setdefault((r["source_type"], r["source"]), []).append(i)
    out = np.zeros(len(records))
    for idx in members.values():
        out[idx] = statistic(z[idx])
    return out


def build_features(records: list[Record], losses: Losses) -> dict[str, np.ndarray]:
    """Build all feature columns, including the simpler ones used only in the ablation.

    Args:
        records: All records (known and test together), so that the statistics use both.
        losses: Cached per-token and mean losses and token counts per record id.

    Returns:
        Named columns aligned with ``records``.
    """
    unknown = {r["source_type"] for r in records} - set(SOURCE_TYPES)
    if unknown:
        raise ValueError(f"Unknown source types: {sorted(unknown)}")
    z = zscore_by_type(delta(records, losses), records)
    columns = {"z_delta": z, "owner_mean_z_delta": owner_aggregate(z, records, np.mean)}
    for name, values in record_signals(records, losses).items():
        columns[name] = zscore_by_type(values, records)
    columns["owner_std_z_sum_delta"] = owner_aggregate(columns["z_sum_delta"], records, np.std)
    for t in SOURCE_TYPES:
        columns[f"type: {t}"] = np.array([r["source_type"] == t for r in records], dtype=float)
    log_len = np.log([losses[r["id"]]["n_tokens"] for r in records])
    std = log_len.std()
    columns["log_n_tokens"] = (log_len - log_len.mean()) / (std if std > 0 else 1.0)
    return columns
