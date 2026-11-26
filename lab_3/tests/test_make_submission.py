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


def test_collection_records_share_the_owner_mean_and_known_member_owners_come_first():
    known = [
        make_record("n1", "writer", "other", 0),
        make_record("n2", "writer", "other", 0),
        make_record("m", "writer", "A", 1),
        make_record("p1", "forum post", "u1", 0),
        make_record("p2", "forum post", "u2", 0),
    ]
    test = [
        make_record("a", "writer", "A"),
        make_record("b1", "writer", "B"),
        make_record("b2", "writer", "B"),
        make_record("post", "forum post", "u3"),
    ]
    # known non-members: mean 0 and standard deviation 1 in both types, so the z-score equals the signal
    signals = {"n1": -1.0, "n2": 1.0, "m": 5.0, "p1": -1.0, "p2": 1.0, "a": 0.0, "b1": 2.0, "b2": 4.0, "post": 2.5}

    scores = membership_scores(known, test, signals)

    assert scores == pytest.approx([10.0, 3.0 + 0.02, 3.0 + 0.04, 2.5])
