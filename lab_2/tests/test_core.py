import numpy as np
import pandas as pd
import pytest
import torch

from lab_2 import scoring
from lab_2.config import Config, FormCfg
from lab_2.data import N_CLASSES, stratified_split
from lab_2.forms import make_form
from lab_2.optimise import Pools, make_pools


def test_split_is_stratified_fixed_and_disjoint():
    y = np.repeat(np.arange(N_CLASSES), 20)
    opt, held = stratified_split(y, 0.25, seed=0)
    assert len(opt) == 300 and len(held) == 100
    assert not set(opt) & set(held)
    assert all(np.sum(y[held] == c) == 5 for c in range(N_CLASSES))
    opt2, held2 = stratified_split(y, 0.25, seed=0)
    assert (held == held2).all() and (opt == opt2).all()


@pytest.mark.parametrize("name,cfg", [("patch", FormCfg()), ("blend", FormCfg()), ("warp", FormCfg(res=4))])
def test_forms_identity_at_init_and_differentiable(name, cfg):
    g, b = 3, 2
    form = make_form(name, cfg)
    p = {k: v.requires_grad_(True) for k, v in form.init(g, "cpu").items()}
    x = torch.rand(g, b, 3, 64, 64)
    xa = form.apply(p, x)
    assert xa.shape == x.shape
    assert xa.min() >= 0 and xa.max() <= 1
    if name != "patch":  # patch starts with a small non-zero mask by design
        assert torch.allclose(xa, x, atol=1e-5)
    (xa.sum() + form.penalty(p).sum()).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.values())
    assert form.size(p).shape == (g,)


def test_pools_sample_stays_in_pool():
    pools = make_pools([torch.tensor([5, 6, 7]), torch.tensor([10, 11])], "cpu")
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        s = pools.sample(8, gen)
        assert set(s[0].tolist()) <= {5, 6, 7} and set(s[1].tolist()) <= {10, 11}
    assert isinstance(pools, Pools)


def _fake_stats(m=40, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(5.0, 0.3, size=(m, N_CLASSES)) + rng.normal(0, 0.5, size=(1, N_CLASSES))
    f[0, 7] -= 4.0  # model 0 has a backdoor on class 7
    st = scoring.Stats([f"m{i}" for i in range(m)], {"patch": f})
    return st


def test_select_pairs_prefers_source_specific_spikes():
    from lab_2.scan import select_pairs

    cfg = Config()
    cfg.tgt_topk, cfg.tgt_src = 2, 2
    flips = np.full((N_CLASSES, N_CLASSES), 0.05)
    flips[7, 3] = 1.0  # target 7 is flipped from source 3 only -> spike
    flips[7, 9] = 0.4
    flips[5, :] = 0.9  # target 5 flips every source -> universal, no spike
    flips[11, 2] = 0.6
    pairs = select_pairs(flips, cfg)
    assert (3, 7) in pairs and (9, 7) in pairs
    assert all(t != 5 for _, t in pairs)
    assert all(s != t for s, t in pairs)


def test_calibration_ranks_the_planted_class_first():
    cfg = Config()
    st = _fake_stats()
    s, comps = scoring.score(cfg, st, scoring.calibrate(cfg, st))
    assert s.shape == (40, N_CLASSES) and np.isfinite(s).all()
    assert s.argmax() == 7  # row 0, class 7
    assert "patch" in comps


def test_score_handles_missing_targeted_pairs_without_nan():
    cfg = Config()
    st = _fake_stats()
    tgt = np.full((40, N_CLASSES, N_CLASSES), np.nan)
    tgt[:, :, 3] = np.random.default_rng(1).normal(6, 0.3, size=(40, N_CLASSES))
    tgt[:, 3, 3] = np.nan
    st.tgt, st.leak = tgt, np.zeros_like(tgt)
    s, comps = scoring.score(cfg, st, scoring.calibrate(cfg, st))
    assert np.isfinite(s).all()
    assert np.isfinite(comps["targeted"][:, 3]).all() and np.isnan(comps["targeted"][:, 4]).all()


def test_csv_roundtrip_and_validation(tmp_path):
    ids = [f"id{i}" for i in range(5)]
    s = np.random.default_rng(0).normal(size=(5, N_CLASSES))
    p = tmp_path / "sub.csv"
    scoring.write_csv(p, ids, s)
    scoring.validate_csv(p, ids)
    df = pd.read_csv(p)
    df.loc[0, "c03"] = np.inf
    df.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        scoring.validate_csv(p, ids)
