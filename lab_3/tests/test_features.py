import numpy as np
import pytest

from lab_3.features import build_features, record_signals

ROWS = [
    # id, source_type, source, delta, member
    ("a", "writer", "anna", 1.0, 1),
    ("b", "writer", "anna", 3.0, 0),
    ("c", "writer", "bob", 5.0, 0),
    ("d", "writer", "bob", 7.0, 1),
    ("e", "forum post", "u1", 10.0, 1),
    ("f", "forum post", "u2", 30.0, 0),
    ("g", "web photo", "u3", -1.0, 0),
    ("h", "web photo", "u4", 1.0, 1),
]
RECORDS = [{"id": i, "source_type": t, "source": s, "member": m} for i, t, s, _, m in ROWS]


def make_losses(bumas, ref):
    bumas, ref = np.array(bumas, dtype=np.float32), np.array(ref, dtype=np.float32)
    means = {"bumas_mean": float(bumas.mean()), "ref_mean": float(ref.mean())}
    return {"bumas_tok": bumas, "ref_tok": ref, "n_tokens": len(bumas), **means}


LOSSES = {i: make_losses([2.0] * (2 + k), [2.0 + d] * (2 + k)) for k, (i, _, _, d, _) in enumerate(ROWS)}


def test_zscore_is_computed_per_type():
    z = build_features(RECORDS, LOSSES)["z_delta"]
    writer = (np.array([1.0, 3.0, 5.0, 7.0]) - 4.0) / np.std([1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(z, [*writer, -1.0, 1.0, -1.0, 1.0])


def test_owner_signal_averages_sets_and_is_zero_for_single_records():
    columns = build_features(RECORDS, LOSSES)
    z, owner = columns["z_delta"], columns["owner_mean_z_delta"]
    np.testing.assert_allclose(owner[:4], np.repeat([z[:2].mean(), z[2:4].mean()], 2))
    np.testing.assert_array_equal(owner[4:], 0.0)


def test_record_signals_summarise_the_token_differences():
    # d = ref - bumas = [3, -1, 1, 5]
    losses = {"a": make_losses([1.0, 3.0, 2.0, 2.0], [4.0, 2.0, 3.0, 7.0])}
    signals = {name: values[0] for name, values in record_signals([{"id": "a"}], losses).items()}
    assert signals == pytest.approx(
        {
            "z_sum_delta": 4.0,
            "z_bottom_delta": 0.0,
            "z_relative_delta": 0.5,
            "z_neg_bumas_loss": -2.0,
            "z_ref_loss": 4.0,
        }
    )


def test_owner_spread_is_the_std_within_the_set_and_zero_for_single_records():
    columns = build_features(RECORDS, LOSSES)
    z, spread = columns["z_sum_delta"], columns["owner_std_z_sum_delta"]
    np.testing.assert_allclose(spread[:4], np.repeat([z[:2].std(), z[2:4].std()], 2))
    np.testing.assert_array_equal(spread[4:], 0.0)


def test_features_do_not_depend_on_labels():
    flipped = [{**r, "member": 1 - m} for r, (*_, m) in zip(RECORDS, ROWS, strict=True)]
    unlabelled = [{k: v for k, v in r.items() if k != "member"} for r in RECORDS]
    base = build_features(RECORDS, LOSSES)
    for other in (build_features(flipped, LOSSES), build_features(unlabelled, LOSSES)):
        assert other.keys() == base.keys()
        for name in base:
            np.testing.assert_array_equal(other[name], base[name])


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown source types"):
        build_features([{"id": "a", "source_type": "blog", "source": "x"}], LOSSES)
