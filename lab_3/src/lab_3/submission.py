"""Submission validation and writing."""

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path


def validate_submission(ids: Sequence[str], scores: Sequence[float], test_ids: Iterable[str]) -> None:
    """Check the rules enforced by the portal.

    Args:
        ids: Ids to submit.
        scores: Scores aligned with ``ids``.
        test_ids: Ids of all test records.

    Raises:
        ValueError: If ids are missing, duplicated or foreign, or a score is not finite.
    """
    if len(ids) != len(scores):
        raise ValueError("ids and scores differ in length")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicated ids")
    expected = set(test_ids)
    if missing := expected - set(ids):
        raise ValueError(f"{len(missing)} test ids are missing")
    if extra := set(ids) - expected:
        raise ValueError(f"{len(extra)} ids do not belong to the test set")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")


def write_submission(path: Path, ids: Sequence[str], scores: Sequence[float], test_ids: Iterable[str]) -> None:
    """Validate and write ``id,score`` rows."""
    validate_submission(ids, scores, test_ids)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "score"])
        writer.writerows(zip(ids, (float(s) for s in scores), strict=True))
