"""Per-model scan: batched all-class K-Arm style scan (4.2) + source-specific scan (4.3)."""

from __future__ import annotations

import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import N_CLASSES, CleanData
from .forms import Form, make_form
from .optimise import (
    Pools,
    State,
    eval_asr,
    final_params,
    init_state,
    logits_of,
    make_pools,
    run,
)

BIG = 1e6


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rank_key(st: State) -> torch.Tensor:
    """Smaller = more suspicious. Classes that reached the ASR target rank by size, the rest by ASR."""
    reached = torch.isfinite(st.best_size)
    return torch.where(reached, st.best_size, BIG - st.ema)


def _nontarget_pools(idx: torch.Tensor, y: torch.Tensor, device) -> Pools:
    return make_pools([idx[y[idx] != t] for t in range(N_CLASSES)], device)


def scan_all_class(
    net, data: CleanData, cfg: Config, gen: torch.Generator, form_name: str
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    """K-Arm style scan of one form over all 20 classes. Returns recorded stats and the per-class rank key."""
    dev = data.x.device
    form: Form = make_form(form_name, getattr(cfg, form_name))
    tgt = torch.arange(N_CLASSES, device=dev)
    opt_pools = _nontarget_pools(data.opt_idx, data.y, dev)
    held_pools = _nontarget_pools(data.held_idx, data.y, dev)

    p1 = form.cfg.p1_steps if form.cfg.p1_steps is not None else cfg.p1_steps
    p2 = form.cfg.p2_steps if form.cfg.p2_steps is not None else cfg.p2_steps
    st = init_state(form, N_CLASSES, dev)
    run(net, data.x, form, st, tgt, opt_pools, p1, cfg.p1_batch, cfg, gen)

    refined = torch.zeros(N_CLASSES, dtype=torch.bool, device=dev)
    if p2 > 0 and cfg.topk > 0:
        sel = torch.argsort(_rank_key(st))[: cfg.topk]
        sub = st.sub(sel)
        run(
            net, data.x, form, sub, tgt[sel], opt_pools.sub(sel), p2, cfg.p2_batch, cfg, gen,
            lr_scale=cfg.p2_lr_scale,
        )
        st.merge(sel, sub)
        refined[sel] = True

    params = final_params(st)
    with torch.no_grad():
        size = form.size(params)
    held_asr = eval_asr(net, data.x, form, params, tgt, held_pools, cfg)
    extra = {}
    if form_name == cfg.tgt_form:
        extra[f"{form_name}_srcflip"] = source_flips(net, data, form, params, cfg)
    out = {
        **extra,
        f"{form_name}_size": size,
        f"{form_name}_asr": held_asr,
        f"{form_name}_reached": torch.isfinite(st.best_size),
        f"{form_name}_refined": refined,
        f"{form_name}_ema": st.ema,
        f"{form_name}_loss": st.loss,
        f"{form_name}_lam": st.lam,
    }
    return {k: v.detach().float().cpu().numpy() for k, v in out.items()}, _rank_key(st)


@torch.no_grad()
def source_flips(net, data: CleanData, form: Form, params, cfg: Config) -> torch.Tensor:
    """(target, source) fraction of the source class's 20 clean images flipped to the target by the
    all-class trigger of that target. A source-specific backdoor shows up as a spike at its source."""
    g = next(iter(params.values())).shape[0]
    onehot = F.one_hot(data.y, N_CLASSES).float()  # (N,S)
    out = torch.zeros(g, N_CLASSES, device=data.x.device)
    n_img = data.x.shape[0]
    cg = max(1, cfg.max_batch // n_img)
    for c0 in range(0, g, cg):
        sl = slice(c0, min(g, c0 + cg))
        n = sl.stop - sl.start
        pc = {k: v[sl] for k, v in params.items()}
        xa = form.apply(pc, data.x[None].expand(n, -1, -1, -1, -1))
        pred = logits_of(net, xa.flatten(0, 1), cfg).argmax(-1).view(n, n_img)
        hit = (pred == torch.arange(c0, c0 + n, device=pred.device)[:, None]).float()
        out[sl] = (hit @ onehot) / onehot.sum(0)
    return out


def select_pairs(srcflip: np.ndarray, cfg: Config) -> list[tuple[int, int]]:
    """Pick the (source, target) pairs worth a pair-level optimisation.

    For every target the spike is max - median of the other classes' flip rates; the ``tgt_topk`` targets
    with the biggest spike are kept, each with its ``tgt_src`` best-flipping sources. Universal backdoors
    flip every source (no spike) and are already covered by the all-class scan."""
    flips = srcflip.astype(np.float64).copy()
    np.fill_diagonal(flips, np.nan)
    spike = np.nanmax(flips, axis=1) - np.nanmedian(flips, axis=1)
    targets = np.argsort(-spike)[: cfg.tgt_topk]
    pairs = []
    for t in sorted(targets.tolist()):
        row = np.where(np.isnan(flips[t]), -1.0, flips[t])
        pairs += [(int(s), t) for s in np.argsort(-row)[: cfg.tgt_src]]
    return pairs


def scan_targeted(
    net, data: CleanData, cfg: Config, gen: torch.Generator, pairs: list[tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Optimise a trigger per (source, target) pair on the source's own images only."""
    dev = data.x.device
    form = make_form(cfg.tgt_form, getattr(cfg, cfg.tgt_form))
    out = {
        k: np.full((N_CLASSES, N_CLASSES), np.nan, dtype=np.float32)
        for k in ("tgt_size", "tgt_asr", "tgt_leak", "tgt_reached")
    }
    if not pairs:
        return out
    all_idx = torch.arange(len(data.y), device=dev)
    src = torch.tensor([p[0] for p in pairs], device=dev)
    tgt = torch.tensor([p[1] for p in pairs], device=dev)
    pools = make_pools([all_idx[data.y == s] for s, _ in pairs], dev)

    st = init_state(form, len(pairs), dev)
    run(net, data.x, form, st, tgt, pools, cfg.tgt_steps, cfg.tgt_batch, cfg, gen)
    params = final_params(st)
    with torch.no_grad():
        size = form.size(params)
    asr = eval_asr(net, data.x, form, params, tgt, pools, cfg)

    leak = torch.full((len(pairs),), float("nan"), device=dev)
    if cfg.tgt_leak_n > 0:
        # random other-class images (not the source, not the target) per pair
        lists = []
        for s, t in pairs:
            cand = all_idx[(data.y != s) & (data.y != t)]
            perm = torch.randperm(len(cand), generator=gen, device=dev)[: cfg.tgt_leak_n]
            lists.append(cand[perm])
        leak = eval_asr(net, data.x, form, params, tgt, make_pools(lists, dev), cfg)

    s_np, t_np = src.cpu().numpy(), tgt.cpu().numpy()
    for key, val in (
        ("tgt_size", size),
        ("tgt_asr", asr),
        ("tgt_leak", leak),
        ("tgt_reached", torch.isfinite(st.best_size).float()),
    ):
        out[key][s_np, t_np] = val.detach().float().cpu().numpy()
    return out


def scan_model(net, data: CleanData, cfg: Config, model_id: str) -> dict[str, np.ndarray]:
    dev = data.x.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed * 1_000_003 + zlib.crc32(model_id.encode()) % 1_000_003)
    res: dict[str, np.ndarray] = {}
    times: dict[str, float] = {}
    for name in cfg.forms():
        _sync()
        t0 = time.perf_counter()
        r, _ = scan_all_class(net, data, cfg, gen, name)
        _sync()
        times[f"t_{name}"] = time.perf_counter() - t0
        res.update(r)
    flips = res.get(f"{cfg.tgt_form}_srcflip")
    if cfg.targeted and flips is not None:
        _sync()
        t0 = time.perf_counter()
        pairs = select_pairs(flips, cfg)
        res.update(scan_targeted(net, data, cfg, gen, pairs))
        _sync()
        times["t_targeted"] = time.perf_counter() - t0
    for k, v in times.items():
        res[k] = np.float32(v)
    res["t_total"] = np.float32(sum(times.values()))
    return res
