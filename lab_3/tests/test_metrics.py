from lab_3.metrics import tpr_przy_5_fpr


def test_perfect_separation():
    assert tpr_przy_5_fpr([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]) == 1.0


def test_constant_score_gives_zero():
    assert tpr_przy_5_fpr([0.5] * 40, [1, 0] * 20) == 0.0


def test_ties_cross_the_threshold_together():
    # 20 non-members allow one false positive. A tie holding two of them is rejected as a whole.
    scores = [3.0] + [2.0, 2.0, 2.0] + [0.0] * 18
    labels = [1] + [1, 0, 0] + [0] * 18
    assert tpr_przy_5_fpr(scores, labels) == 0.5


def test_exactly_five_percent_is_allowed():
    scores = [3.0, 2.0, 1.0] + [0.0] * 19
    labels = [0, 1, 0] + [0] * 18 + [1]
    assert tpr_przy_5_fpr(scores, labels) == 0.5
