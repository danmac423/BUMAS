"""Generic batched trigger optimiser with a Neural-Cleanse style adaptive lambda.

``G`` independent groups (target classes, or (source, target) pairs) are optimised in parallel. Every
step each group draws ``B`` images from its own pool, applies its own trigger and is pushed towards
its target class. The size penalty weight lambda is adapted per group so that the trigger just reaches
the ASR target; the smallest trigger that did is kept as a snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import Config, FormCfg
from .forms import Form, Params


@dataclass
class Pools:
    """Padded per-group image pools: ``idx`` (G,P) indices into the image tensor, ``cnt`` (G,) valid counts."""

    idx: torch.Tensor
    cnt: torch.Tensor

    def sub(self, sel: torch.Tensor) -> Pools:
        return Pools(self.idx[sel], self.cnt[sel])

    def sample(self, b: int, gen: torch.Generator) -> torch.Tensor:
        g = self.idx.shape[0]
        r = torch.rand(g, b, device=self.idx.device, generator=gen)
        j = (r * self.cnt[:, None]).long().clamp_(max=self.idx.shape[1] - 1)
        return self.idx.gather(1, j)


def make_pools(lists: list[torch.Tensor], device) -> Pools:
    pmax = max(len(label) for label in lists)
    idx = torch.zeros(len(lists), pmax, dtype=torch.long, device=device)
    cnt = torch.zeros(len(lists), dtype=torch.long, device=device)
    for i, li in enumerate(lists):
        idx[i, : len(li)] = li
        cnt[i] = len(li)
    return Pools(idx, cnt)


@dataclass
class State:
    params: Params
    lam: torch.Tensor  # (G,)
    ema: torch.Tensor  # (G,) EMA of batch ASR
    best_size: torch.Tensor  # (G,) smallest size at which the ASR target was reached (inf if never)
    best: Params  # snapshot of the parameters at best_size
    loss: torch.Tensor  # (G,) last CE loss
    steps: int = 0

    def sub(self, sel: torch.Tensor) -> State:
        return State(
            {k: v[sel].detach().clone().requires_grad_(True) for k, v in self.params.items()},
            self.lam[sel].clone(),
            self.ema[sel].clone(),
            self.best_size[sel].clone(),
            {k: v[sel].clone() for k, v in self.best.items()},
            self.loss[sel].clone(),
            self.steps,
        )

    def merge(self, sel: torch.Tensor, sub: State) -> None:
        with torch.no_grad():
            for k in self.params:
                self.params[k][sel] = sub.params[k]
                self.best[k][sel] = sub.best[k]
            self.lam[sel] = sub.lam
            self.ema[sel] = sub.ema
            self.best_size[sel] = sub.best_size
            self.loss[sel] = sub.loss
        self.steps = sub.steps


def init_state(form: Form, g: int, device) -> State:
    params = {k: v.requires_grad_(True) for k, v in form.init(g, device).items()}
    return State(
        params=params,
        lam=torch.full((g,), form.cfg.lam0, device=device),
        ema=torch.zeros(g, device=device),
        best_size=torch.full((g,), float("inf"), device=device),
        best={k: v.detach().clone() for k, v in params.items()},
        loss=torch.zeros(g, device=device),
    )


def logits_of(net: torch.nn.Module, x: torch.Tensor, cfg: Config) -> torch.Tensor:
    x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp and x.is_cuda):
        return net(x).float()


def run(
    net,
    xs: torch.Tensor,
    form: Form,
    st: State,
    tgt: torch.Tensor,
    pools: Pools,
    steps: int,
    batch: int,
    cfg: Config,
    gen: torch.Generator,
    lr_scale: float = 1.0,
) -> None:
    """Optimise ``st`` in place for ``steps`` Adam steps."""
    fc: FormCfg = form.cfg
    g = tgt.shape[0]
    opt = torch.optim.Adam(list(st.params.values()), lr=fc.lr * lr_scale)
    cg = max(1, cfg.max_batch // batch)  # groups per forward chunk
    thr, band = cfg.asr_target, cfg.asr_band
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        asr_b = torch.zeros(g, device=tgt.device)
        size_all = torch.zeros(g, device=tgt.device)
        for c0 in range(0, g, cg):
            sl = slice(c0, min(g, c0 + cg))
            pc = {k: v[sl] for k, v in st.params.items()}
            n = sl.stop - sl.start
            idx = Pools(pools.idx[sl], pools.cnt[sl]).sample(batch, gen)
            xa = form.apply(pc, xs[idx])
            lg = logits_of(net, xa.flatten(0, 1), cfg).view(n, batch, -1)
            t = tgt[sl]
            ce = F.cross_entropy(lg.flatten(0, 1), t.repeat_interleave(batch), reduction="none").view(n, batch)
            ce = ce.mean(1)
            pen = form.penalty(pc)
            loss = (ce + st.lam[sl] * pen).sum()
            (loss * cfg.loss_scale).backward()
            with torch.no_grad():
                asr_b[sl] = (lg.argmax(-1) == t[:, None]).float().mean(1)
                size_all[sl] = form.size(pc)
                st.loss[sl] = ce.detach()
        with torch.no_grad():
            for p in st.params.values():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            st.ema.mul_(cfg.ema).add_((1 - cfg.ema) * asr_b)
            # snapshot (params are still the ones that produced size_all)
            if st.steps + step >= cfg.warmup:
                better = (st.ema >= thr) & (size_all < st.best_size)
                if better.any():
                    st.best_size = torch.where(better, size_all, st.best_size)
                    for k, v in st.params.items():
                        m = better.view(-1, *([1] * (v.dim() - 1)))
                        st.best[k] = torch.where(m, v.detach(), st.best[k])
            # lambda controller
            up = st.ema > thr + band
            down = st.ema < thr - band
            fac = torch.exp(cfg.lam_eta * (up.float() - down.float()))
            st.lam = (st.lam * fac).clamp_(fc.lam_min, fc.lam_max)
        opt.step()
    st.steps += steps


@torch.no_grad()
def final_params(st: State) -> Params:
    """Snapshot where the ASR target was reached, else the current parameters."""
    reached = torch.isfinite(st.best_size)
    return {
        k: torch.where(reached.view(-1, *([1] * (v.dim() - 1))), st.best[k], v.detach())
        for k, v in st.params.items()
    }


@torch.no_grad()
def eval_asr(net, xs, form: Form, params: Params, tgt: torch.Tensor, pools: Pools, cfg: Config) -> torch.Tensor:
    """Fraction of every group's pool classified as the group's target after applying the trigger."""
    g = tgt.shape[0]
    out = torch.zeros(g, device=tgt.device)
    cg = max(1, cfg.max_batch // max(1, int(pools.cnt.max())))
    for c0 in range(0, g, cg):
        sl = slice(c0, min(g, c0 + cg))
        n = sl.stop - sl.start
        pc = {k: v[sl] for k, v in params.items()}
        idx = pools.idx[sl]  # (n,P)
        xa = form.apply(pc, xs[idx])
        lg = logits_of(net, xa.flatten(0, 1), cfg).view(n, idx.shape[1], -1)
        hit = (lg.argmax(-1) == tgt[sl, None]).float()
        valid = (torch.arange(idx.shape[1], device=tgt.device)[None] < pools.cnt[sl, None]).float()
        out[sl] = (hit * valid).sum(1) / valid.sum(1)
    return out
