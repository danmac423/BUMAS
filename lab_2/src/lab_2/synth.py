"""Optional stretch (SPEC 5.3): plant known triggers into a clean model by short fine-tuning on the 400 clean images.

Only meant for measuring per-trigger-type recall of the detector, not for producing realistic victims.
Output goes in the same layout as ``example/train`` (``models/<id>.pt`` + ``labels.csv``), so the normal
``scan`` and ``score --labels`` commands work on it unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .config import Config
from .data import load_data, load_net

KINDS = ("patch", "dots", "blend", "sig", "wanet")


def make_trigger(kind: str, seed: int, device):
    """Returns f(x: (N,3,64,64)) -> triggered x. The trigger is fixed by (kind, seed)."""
    g = torch.Generator().manual_seed(seed)
    hw = 64
    if kind == "patch":
        s = 5
        pat = (torch.rand(3, s, s, generator=g) > 0.5).float()
        r, c = (int(v) for v in torch.randint(2, hw - s - 2, (2,), generator=g))
        pat = pat.to(device)

        def f(x):
            x = x.clone()
            x[:, :, r : r + s, c : c + s] = pat
            return x

    elif kind == "dots":
        n = 40
        pos = torch.randint(0, hw * hw, (n,), generator=g)
        col = torch.rand(n, 3, generator=g).round()
        mask = torch.zeros(hw * hw)
        mask[pos] = 1
        pat = torch.zeros(3, hw * hw)
        pat[:, pos] = col.T
        mask, pat = mask.view(1, 1, hw, hw).to(device), pat.view(1, 3, hw, hw).to(device)

        def f(x):
            return x * (1 - mask) + pat * mask

    elif kind == "blend":
        noise = torch.rand(1, 3, hw, hw, generator=g).to(device)

        def f(x):
            return 0.85 * x + 0.15 * noise

    elif kind == "sig":
        freq = 6
        wave = torch.sin(2 * torch.pi * freq * torch.arange(hw) / hw)
        d = (0.1 * wave).view(1, 1, 1, hw).to(device)

        def f(x):
            return (x + d).clamp(0, 1)

    elif kind == "wanet":
        k, strength_px = 4, 1.5
        ins = torch.rand(1, 2, k, k, generator=g) * 2 - 1
        ins = ins / ins.abs().mean() * strength_px
        flow = F.interpolate(ins, size=(hw, hw), mode="bicubic", align_corners=True).to(device)
        flow = flow.permute(0, 2, 3, 1) * (2.0 / (hw - 1))
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, hw), torch.linspace(-1, 1, hw), indexing="ij")
        grid = (torch.stack([xs, ys], -1)[None].to(device) + flow).clamp(-1, 1)

        def f(x):
            return F.grid_sample(x, grid.expand(x.shape[0], -1, -1, -1), padding_mode="border", align_corners=True)

    else:
        raise ValueError(f"unknown trigger kind {kind}")
    return f


def parse_spec(spec: str) -> dict:
    """``kind:target[:source]``; with a source the backdoor is source specific (targeted)."""
    parts = spec.split(":")
    return {"kind": parts[0], "target": int(parts[1]), "source": int(parts[2]) if len(parts) > 2 else None}


def plant(
    base: str | Path,
    out_dir: str | Path,
    specs: list[str],
    cfg: Config,
    epochs: int = 30,
    lr: float = 0.01,
    poison: float = 0.15,
    seed: int = 0,
) -> dict:
    dev = cfg.device
    torch.manual_seed(seed)
    data = load_data(cfg.clean_dir, dev, cfg.held_frac, cfg.seed)
    net = load_net(base, cfg.model_py, dev).float()
    net = net.to(memory_format=torch.contiguous_format)
    parsed = [parse_spec(s) for s in specs]
    trig = [make_trigger(p["kind"], seed * 100 + 17 + i, dev) for i, p in enumerate(parsed)]
    net.requires_grad_(True)
    opt = torch.optim.SGD([p for p in net.parameters()], lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(data.y)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for b0 in range(0, n, 64):
            idx = perm[b0 : b0 + 64]
            x, y = data.x[idx].clone(), data.y[idx].clone()
            flip = torch.rand(len(idx), device=dev) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            for p, f in zip(parsed, trig):
                pick = torch.rand(len(idx), device=dev) < poison
                if p["source"] is None:
                    pick &= y != p["target"]
                    x[pick], y[pick] = f(x[pick]), p["target"]
                else:
                    # source specific: poison the source class, and keep a trigger on other classes harmless
                    src = y == p["source"]
                    to_t = src & (torch.rand(len(idx), device=dev) < 0.5)
                    cross = (torch.rand(len(idx), device=dev) < poison) & ~src
                    x[to_t], y[to_t] = f(x[to_t]), p["target"]
                    x[cross] = f(x[cross])
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval().requires_grad_(False)

    with torch.no_grad():
        acc = (net(data.x).argmax(1) == data.y).float().mean().item()
        asrs = []
        for p, f in zip(parsed, trig):
            m = (data.y != p["target"]) if p["source"] is None else (data.y == p["source"])
            asrs.append((net(f(data.x[m])).argmax(1) == p["target"]).float().mean().item())

    mid = hashlib.sha1(f"{base}{specs}{seed}".encode()).hexdigest()[:12]
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    sd = {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}
    torch.save(sd, out_dir / "models" / f"{mid}.pt")
    row = {
        "id": mid,
        "targets": ";".join(str(p["target"]) for p in parsed),
        "asr": ";".join(f"{a:.3f}" for a in asrs),
        "clean_acc": f"{acc:.4f}",
        "specs": " ".join(specs),
    }
    lab = out_dir / "labels.csv"
    pd.DataFrame([row]).to_csv(lab, mode="a", header=not lab.exists(), index=False)
    return row
