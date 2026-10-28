import pandas as pd
import pytest

from lab_1.data import add_folds, load_labels
from lab_1.scoring import BOX_COLS, NONE, evaluate, punkty


def test_punkty_cases():
    assert punkty(NONE, NONE) == 1.0
    assert punkty(NONE, (1, 1, 5, 5)) == 0.0
    assert punkty((1, 1, 5, 5), NONE) == 0.0
    assert punkty((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert punkty((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert punkty((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


@pytest.fixture(scope="module")
def labels():
    return load_labels("train")


def _pred(df, boxes):
    out = pd.DataFrame(boxes, columns=BOX_COLS)
    out.insert(0, "id", df.id.to_numpy())
    return out


def test_all_none_is_020(labels):
    res = evaluate(labels, _pred(labels, [NONE] * len(labels)))
    assert res["mean"] == pytest.approx(0.2)


def test_oracle_box_is_1(labels):
    res = evaluate(labels, _pred(labels, labels[BOX_COLS].to_numpy()))
    assert res["mean"] == 1.0


def test_labels_meta_and_folds(labels):
    assert labels.kind.value_counts().to_dict() == {"ADV": 4000, "CLEAN": 500, "DECOY": 500}
    assert (labels.has_patch == (labels[BOX_COLS].ne(0).any(axis=1))).all()
    folded = add_folds(labels)
    assert folded.fold.between(0, 4).all()
    assert folded.groupby("fold").has_patch.mean().between(0.79, 0.81).all()
