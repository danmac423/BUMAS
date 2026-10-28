import numpy as np
import pytest
import torch

from lab_1.igcausal import (
    all_boxes,
    allowed_sizes,
    box_iou_np,
    box_masks,
    box_means,
    effect,
    is_allowed,
    neighbours,
    nms,
    random_boxes,
)
from lab_1.scoring import punkty


class _Region(torch.nn.Module):
    """Class 0 = mean over rows 0..9, cols 0..19; class 1 = 0. Margin(0, 1) is that mean."""

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        c0 = x[:, :, :10, :20].mean(dim=(1, 2, 3))
        return torch.stack([c0, torch.zeros_like(c0)], dim=1)


def test_allowed_sizes_respect_constraints():
    wh = allowed_sizes()
    w, h = wh[:, 0], wh[:, 1]
    assert ((w * h >= 328) & (w * h <= 614) & (2 * w >= h) & (2 * h >= w)).all()
    assert [20, 20] in wh.tolist() and [13, 26] in wh.tolist()
    assert [18, 18] not in wh.tolist()  # 324 < 328


def test_all_boxes_are_allowed_and_unique():
    b = all_boxes()
    assert is_allowed(b).all()
    assert len(torch.unique(b, dim=0)) == len(b)
    assert b[:, 2].max() == 64 and b[:, 0].min() == 0


def test_random_boxes_are_allowed():
    b = random_boxes(8, 50, torch.Generator().manual_seed(0))
    assert b.shape == (8, 50, 4)
    assert is_allowed(b).all()


def test_box_masks_match_slicing():
    m = box_masks(torch.tensor([[3, 5, 20, 30]]))[0, 0]
    ref = torch.zeros(64, 64, dtype=torch.bool)
    ref[5:30, 3:20] = True
    assert torch.equal(m, ref)


def test_box_means_match_direct_mean():
    a = torch.rand(2, 64, 64, generator=torch.Generator().manual_seed(0))
    b = torch.tensor([[0, 0, 20, 20], [10, 30, 34, 50], [40, 40, 64, 64]])
    got = box_means(a, b)
    for j, (x0, y0, x1, y1) in enumerate(b.tolist()):
        assert torch.allclose(got[:, j], a[:, y0:y1, x0:x1].mean((1, 2)), atol=1e-5)


def test_nms_keeps_best_and_suppresses_overlaps():
    b = torch.tensor([[0, 0, 20, 20], [1, 0, 21, 20], [40, 40, 60, 60]])
    s = torch.tensor([[3.0, 2.0, 1.0]])
    assert nms(b, s, 2).tolist() == [[0, 2]]


def test_neighbours_move_one_edge():
    nb = neighbours(torch.tensor([[10, 10, 30, 30]]))[0]
    assert nb.shape == (24, 4)
    assert ((nb != torch.tensor([10, 10, 30, 30])).sum(1) == 1).all()


def test_effect_on_toy_model():
    x, xt = torch.ones(1, 3, 64, 64), torch.zeros(1, 3, 64, 64)
    c1, c2 = torch.tensor([0]), torch.tensor([1])
    boxes = torch.tensor([[[0, 0, 20, 10], [0, 0, 10, 10], [30, 30, 50, 50]]])
    e = effect(_Region(), x, xt, c1, c2, torch.tensor([1.0]), boxes)
    assert e[0].tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_box_iou_matches_punkty():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 32, (50, 2))
    t = np.concatenate([t, t + rng.integers(1, 32, (50, 2))], 1)
    p = rng.integers(0, 32, (50, 2))
    p = np.concatenate([p, p + rng.integers(1, 32, (50, 2))], 1)
    got = box_iou_np(t, p)
    ref = [punkty(tuple(a), tuple(b)) for a, b in zip(t.tolist(), p.tolist())]
    assert got == pytest.approx(ref)
