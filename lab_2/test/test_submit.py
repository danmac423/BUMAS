import pytest
import torch

from lab_2.optimise import result_path
from lab_2.submit import load_scores, load_stage5, pair_scores, robust_stats
from lab_2.triggers import FAMILIES


def _write_stage5(art, group, ids, value_fn):
    for i, mid in enumerate(ids):
        for j, fam in enumerate(sorted(FAMILIES)):
            p = result_path(art, group, mid, fam)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"scores": value_fn(i, j)}, p)


def test_load_stage5_shape_and_order(tmp_path):
    ids = ["a", "b"]
    _write_stage5(tmp_path, "test", ids, lambda i, j: torch.full((2, 20), 10.0 * i + j))
    s = load_stage5(tmp_path, "test", ids)
    assert s.shape == (2, len(FAMILIES), 2, 20)
    assert s[1, 2, 0, 0] == 12.0  # model "b", third family (alphabetical)


def test_load_stage5_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stage5(tmp_path, "test", ["nope"])


def test_sources_concatenate_and_stage5_family_can_win(tmp_path):
    ids = ["a", "b", "c", "d", "e"]
    torch.save({"ids": ids, "scores": torch.zeros(5, 3, 2, 20)}, tmp_path / "quicktest_test.pt")

    def vals(i, j):
        v = torch.zeros(2, 20)
        if i == 4 and j == 0:
            v[1, 7] = 1.0  # one model, one family, class 7
        return v

    _write_stage5(tmp_path, "test", ids, vals)
    got_ids, scores = load_scores(tmp_path, "test", ["quicktest", "stage5"])
    assert got_ids == ids and scores.shape == (5, 3 + len(FAMILIES), 2, 20)
    med, mad = robust_stats(scores)
    pairs = pair_scores(scores, med, mad)
    assert pairs[4, 7] == pairs.max() and pairs[4, 7] > 0
