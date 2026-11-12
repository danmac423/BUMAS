import torch

from lab_2.data import N_CLASSES, PER_CLASS_HALF, split_ab


def test_split_ab_is_stratified_and_disjoint():
    labels = torch.arange(N_CLASSES).repeat(20)
    a, b = split_ab(labels, seed=1)
    assert len(a) == len(b) == N_CLASSES * PER_CLASS_HALF
    assert set(a.tolist()).isdisjoint(b.tolist())
    for idx in (a, b):
        counts = torch.bincount(labels[idx], minlength=N_CLASSES)
        assert (counts == PER_CLASS_HALF).all()


def test_split_ab_is_deterministic():
    labels = torch.arange(N_CLASSES).repeat(20)
    a1, _ = split_ab(labels, seed=3)
    a2, _ = split_ab(labels, seed=3)
    assert torch.equal(a1, a2)
