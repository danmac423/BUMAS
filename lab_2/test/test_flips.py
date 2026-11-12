import torch

from lab_2.flips import column_scores, diff_scores, flip_table, flip_table_from_preds, predict


class ConstNet(torch.nn.Module):
    """Predicts class `target` for every input whose first pixel is > 0.5, else its own `base` class."""

    def __init__(self, target, n=4):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.target, self.n = target, n

    def forward(self, x):
        out = torch.zeros(len(x), self.n)
        flagged = x[:, 0, 0, 0] > 0.5
        out[flagged, self.target] = 1.0
        out[~flagged, 0] = 1.0
        return out


def test_flip_table_fractions_and_zero_diagonal():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    preds = torch.tensor([0, 1, 1, 1, 0, 0])  # class 0: half stay, half flip to 1; class 2 flips fully to 0
    t = flip_table_from_preds(preds, labels, n_classes=3)
    assert t[0, 1] == 0.5 and t[2, 0] == 1.0
    assert t.diagonal().sum() == 0
    assert t[1].sum() == 0  # class 1 is always right


def test_column_scores_all_vs_targeted():
    n = 4
    t = torch.zeros(n, n)
    t[1, 3] = 0.9  # one source flips to 3 (targeted poisoning)
    t[:, 2] = 0.6
    t[2, 2] = 0  # a "real" backdoor into 2 from every other class
    s = column_scores(t)
    assert torch.isclose(s["targeted"][3], torch.tensor(0.9))
    assert torch.isclose(s["all"][3], torch.tensor(0.9 / 3))
    assert torch.isclose(s["all"][2], torch.tensor(0.6))
    assert s["targeted"][2] > s["targeted"][0]


def test_reference_subtraction_removes_shared_behaviour():
    n = 4
    shared = torch.zeros(n, n)
    shared[:, 1] = 0.7
    shared.fill_diagonal_(0)
    own = shared.clone()
    own[:, 3] = 0.8
    own.fill_diagonal_(0)
    s = diff_scores(own, shared)
    assert torch.isclose(s["all"][1], torch.tensor(0.0))
    assert s["all"][3] > 0.5


def test_flip_table_with_trigger_end_to_end():
    x = torch.zeros(8, 3, 4, 4)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    net = ConstNet(target=3)

    def stamp(b):
        return b.clone().index_fill_(2, torch.tensor([0]), 1.0)  # light up the top row

    assert (predict(net, x) == 0).all()
    t = flip_table(net, x, y, trigger=stamp)
    assert (t[:3, 3] == 1.0).all() and t[3].sum() == 0
