import pytest
import torch

from lab_2.config import BankCfg
from lab_2.triggers import FAMILIES, BlendTrigger, PatchTrigger, WarpTrigger, build_bank, make_patch

X = torch.rand(4, 3, 64, 64)


def test_bank_triggers_keep_shape_and_range():
    bank = build_bank()
    assert len(bank) > 50
    assert {t.family for t in bank} == {"badnets", "blend", "sig", "wanet"}
    for t in bank:
        y = t(X)
        assert y.shape == X.shape, t.name
        assert y.min() >= 0 and y.max() <= 1, t.name
        assert not torch.equal(y, X), t.name


def test_bank_is_deterministic():
    a, b = build_bank(BankCfg(seed=3)), build_bank(BankCfg(seed=3))
    for ta, tb in zip(a, b):
        assert ta.name == tb.name
        assert torch.equal(ta(X), tb(X))


def test_fixed_patch_changes_only_its_region():
    y = make_patch("white", 5, "br")(X)
    changed = (y != X).any(dim=(0, 1))
    assert changed.sum() <= 25
    assert changed[-6:-1, -6:-1].any()
    assert not changed[:32, :32].any()


@pytest.mark.parametrize("name", FAMILIES)
def test_learnable_family_is_differentiable_and_in_range(name):
    t = FAMILIES[name]()
    y = t(X)
    assert y.shape == X.shape
    assert y.min() >= 0 and y.max() <= 1
    (y.sum() + t.penalty()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in t.parameters())


@pytest.mark.parametrize("name", FAMILIES)
def test_reset_is_deterministic_per_seed(name):
    t = FAMILIES[name]()
    t.reset(5)
    a = t(X).clone()
    t.reset(6)
    t.reset(5)
    assert torch.equal(t(X), a)


def test_patch_initially_close_to_identity_and_penalty_is_area():
    t = PatchTrigger()
    assert (t(X) - X).abs().mean() < 0.1
    assert 0 < t.penalty().item() < 0.1


def test_blend_alpha_bounds_change():
    t = BlendTrigger(alpha=0.15)
    assert (t(X) - X).abs().max() <= 0.15 + 1e-6


def test_warp_amplitude_is_bounded():
    t = WarpTrigger(max_shift=0.06)
    with torch.no_grad():
        t.flow_raw.fill_(50.0)  # saturate tanh
    assert t.flow().abs().max() <= 0.06 + 1e-6
    t.flow_raw.data.zero_()
    assert torch.allclose(t(X), X, atol=1e-5)  # zero field is the identity warp
