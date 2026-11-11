"""Neural Cleanse trigger inversion, batched over classes, with a K-Arm schedule."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

N_CLASSES = 20
ASR_TARGET = 0.9
WINDOW = 10
LAMBDA_INIT = 1e-2
LAMBDA_STEP = 1.5
LR = 0.1


def load_model_fn(data_root: Path):
    """Import `load` from `<data_root>/model.py`."""
    spec = importlib.util.spec_from_file_location("p2_model", Path(data_root) / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load


def load_clean(clean_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return images (N,3,64,64) in [0,1] and labels (N,)."""
    clean_dir = Path(clean_dir)
    df = pd.read_csv(clean_dir / "labels.csv")
    imgs = [
        np.asarray(Image.open(clean_dir / "images" / f"{i}.png").convert("RGB"), dtype=np.float32) / 255.0
        for i in df["id"]
    ]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(df["label"].to_numpy(), dtype=torch.long)


def _optimise(net, x, labels, state, idx, steps, batch):
    """Run `steps` Adam steps on the classes `idx` only; update `state` in place.

    state: dict of tensors a (20,1,64,64), q (20,3,64,64), lam (20,), best (20,), asr (20,), l1 (20,).
    """
    device = x.device
    a = state["a"][idx].clone().requires_grad_(True)
    q = state["q"][idx].clone().requires_grad_(True)
    lam = state["lam"][idx].clone()
    best = state["best"][idx].clone()
    opt = torch.optim.Adam([a, q], lr=LR)
    target = idx.to(device)
    # sampling weights: images whose label is not the class being inverted
    weights = (labels[None, :] != target[:, None]).float()
    win_asr = torch.zeros(len(idx), device=device)
    last_asr = state["asr"][idx].clone()
    n_cls = len(idx)

    for step in range(1, steps + 1):
        pick = torch.multinomial(weights, batch, replacement=True)  # (C,B)
        xs = x[pick]  # (C,B,3,64,64)
        m = torch.sigmoid(a)
        p = torch.sigmoid(q)
        xp = (1 - m[:, None]) * xs + m[:, None] * p[:, None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = net(xp.flatten(0, 1))
        logits = logits.float().view(n_cls, batch, -1)
        ce = F.cross_entropy(logits.flatten(0, 1), target.repeat_interleave(batch), reduction="none")
        ce = ce.view(n_cls, batch).mean(1)
        l1 = m.flatten(1).sum(1)
        loss = (ce + lam * l1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        win_asr += (logits.argmax(-1) == target[:, None]).float().mean(1).detach()
        if step % WINDOW == 0:
            asr = win_asr / WINDOW
            win_asr.zero_()
            cur = l1.detach()
            ok = asr >= ASR_TARGET
            best = torch.where(ok, torch.minimum(best, cur), best)
            lam = torch.where(ok, lam * LAMBDA_STEP, lam / LAMBDA_STEP)
            last_asr = asr

    with torch.no_grad():
        state["a"][idx] = a.detach()
        state["q"][idx] = q.detach()
        state["lam"][idx] = lam
        state["best"][idx] = best
        state["asr"][idx] = last_asr
        state["l1"][idx] = torch.sigmoid(a).flatten(1).sum(1).detach()


def _recorded(state) -> torch.Tensor:
    """Smallest norm at ASR >= 0.9, or the current norm if never reached."""
    return torch.where(torch.isfinite(state["best"]), state["best"], state["l1"])


def scan_model(net, x, labels, s1=150, s2=100, k=5, b1=8, b2=32) -> tuple[np.ndarray, np.ndarray]:
    """Return (l1, asr), each of shape (20,)."""
    device = x.device
    state = {
        "a": torch.zeros(N_CLASSES, 1, 64, 64, device=device),
        "q": torch.zeros(N_CLASSES, 3, 64, 64, device=device),
        "lam": torch.full((N_CLASSES,), LAMBDA_INIT, device=device),
        "best": torch.full((N_CLASSES,), float("inf"), device=device),
        "asr": torch.zeros(N_CLASSES, device=device),
        "l1": torch.full((N_CLASSES,), 64.0 * 64.0, device=device),
    }
    state["q"].normal_(0, 0.5)
    all_idx = torch.arange(N_CLASSES, device=device)
    _optimise(net, x, labels, state, all_idx, s1, b1)

    focus = torch.topk(_recorded(state), k, largest=False).indices
    _optimise(net, x, labels, state, focus, s2, b2)
    return _recorded(state).cpu().numpy(), state["asr"].cpu().numpy()


def scan_all(models_dir: Path, data_root: Path, out_dir: Path, device: str = "cuda", **kw) -> None:
    """Scan every `*.pt` in models_dir; cache (l1, asr) as `<id>.npz` in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load = load_model_fn(data_root)
    x, labels = load_clean(Path(data_root) / "clean")
    x, labels = x.to(device), labels.to(device)
    for path in tqdm(sorted(Path(models_dir).glob("*.pt"))):
        out = out_dir / f"{path.stem}.npz"
        if out.exists():
            continue
        net = load(path, device)
        for prm in net.parameters():
            prm.requires_grad_(False)
        l1, asr = scan_model(net, x, labels, **kw)
        np.savez(out, l1=l1, asr=asr)
