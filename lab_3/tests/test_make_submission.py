"""Tests of the scoring rules (no model or data needed)."""

import pytest
import torch

from lab_3.make_submission import membership_scores, record_signal


def make_record(record_id: str, source_type: str, source: str, member: int | None = None) -> dict:
    return {"id": record_id, "source_type": source_type, "source": source, "member": member}


def test_record_signal_clips_extreme_tokens():
    bumas = torch.zeros(4)
    clipped = record_signal(torch.tensor([1.0, 2.0, 1.0, 2.0]), bumas)
    extreme = record_signal(torch.tensor([1.0, 20.0, 1.0, 20.0]), bumas)
    assert extreme == pytest.approx(clipped)


def fit_and_score(source_type: str, known_rows: list[tuple], test_rows: list[tuple]) -> list[float]:
    """Rows are (owner, signal, member) for known records and (owner, signal) for test records."""
    known = [make_record(f"k{i}", source_type, owner, member) for i, (owner, _, member) in enumerate(known_rows)]
    test = [make_record(f"t{i}", source_type, owner) for i, (owner, _) in enumerate(test_rows)]
    signals = {f"k{i}": row[1] for i, row in enumerate(known_rows)}
    signals |= {f"t{i}": row[1] for i, row in enumerate(test_rows)}
    return membership_scores(known, test, signals)


def test_single_records_are_ranked_by_their_own_signal_only():
    known = [("a", 1.0, 0), ("b", 2.0, 1), ("c", 3.0, 0), ("d", 4.0, 1)]

    scores = fit_and_score("forum post", known, [("a", 1.5), ("a", 3.5), ("e", 3.5)])

    assert scores[0] < scores[1]
    assert scores[1] == pytest.approx(scores[2])


def test_collection_records_also_use_the_mean_signal_of_their_owner():
    # the owner "high" (test mean 4) has more known members than the owner "low" (test mean 1)
    known = [("high", 1.0, 0), ("high", 2.0, 1), ("high", 2.5, 0), ("high", 3.0, 1)]
    known += [("low", 0.5, 0), ("low", 2.0, 0), ("low", 3.0, 1), ("low", 3.5, 0)]

    scores = fit_and_score("writer", known, [("high", 2.0), ("high", 6.0), ("low", 2.0), ("low", 0.0)])

    assert scores[0] > scores[2]  # the same own signal, but a different owner
