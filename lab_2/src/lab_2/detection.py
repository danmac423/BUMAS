"""
P2 Backdoors - method 1: cheap per-(model, class) statistics (the "solution").

Contains the actual backdoor-detection logic: locating the cosine-classifier
head of a model, computing per-(model, class) suspicion statistics from
clean images (logit-based, feature-based and UAP-based), and the `extract`
command that runs this over every model and caches the results to disk.

All statistics are oriented so that HIGHER = MORE SUSPICIOUS.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from lab_2.data import C, list_models, load_clean, load_model_fn, pick_device, read_classes

# ----------------------------------------------------------------------------- model inspection


def find_head(net):
    """Last module with a (20, D) weight - the cosine classifier. Its input = features."""
    head = None
    for name, m in net.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] == C:
            head = (name, m)
    return head


# ----------------------------------------------------------------------------- statistics


def logit_stats(L, y):
    """L: (N,C) logits on clean images (CPU)."""
    P = L.softmax(1)
    Lc = L - L.mean(1, keepdim=True)  # per-image centered logits
    pred = L.argmax(1)
    rank2 = L.topk(2, dim=1).indices[:, 1]
    s = {k: np.zeros(C, dtype=np.float32) for k in ("attract_logit", "attract_logprob", "second_rank", "fp_rate")}
    A = np.zeros((C, C), dtype=np.float32)  # A[s,c] = mean centered logit of class c on images of class s
    for c in range(C):
        m = y != c
        s["attract_logit"][c] = Lc[m, c].mean().item()
        s["attract_logprob"][c] = torch.log(P[m, c] + 1e-12).mean().item()
        s["second_rank"][c] = (rank2[m] == c).float().mean().item()
        s["fp_rate"][c] = (pred[m] == c).float().mean().item()
    for src in range(C):
        A[src] = Lc[y == src].mean(0).numpy()
    acc = (pred == y).float().mean().item()
    return s, A, acc


def feat_stats(f, y, W):
    """f: (N,D) features, W: (C,D) classifier weights."""
    fn = F.normalize(f, dim=1)
    Wn = F.normalize(W, dim=1)
    mu = F.normalize(torch.stack([fn[y == c].mean(0) for c in range(C)]), dim=1)
    # a backdoor "adds" the trigger direction to the target class row -> worse alignment with the class mean
    w_misalign = (1.0 - (Wn * mu).sum(1)).numpy().astype(np.float32)
    return {"w_misalign": w_misalign}, Wn.numpy().astype(np.float16)


def uap_stats(net, x, y, device, eps, steps):
    """Lightweight per-class L_inf universal perturbation: how easily images of other classes can be pushed to c."""
    succ = np.zeros(C, dtype=np.float32)
    margin = np.zeros(C, dtype=np.float32)
    lr = eps / 4
    for c in range(C):
        xs = x[y != c].to(device)
        tgt = torch.full((len(xs),), c, device=device, dtype=torch.long)
        delta = torch.zeros(1, 3, 64, 64, device=device, requires_grad=True)
        for _ in range(steps):
            out = net((xs + delta).clamp(0, 1))
            loss = F.cross_entropy(out, tgt)
            (g,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= lr * g.sign()
                delta.clamp_(-eps, eps)
        with torch.no_grad():
            out = net((xs + delta).clamp(0, 1)).float()
            succ[c] = (out.argmax(1) == c).float().mean().item()
            other = out.clone()
            other[:, c] = -1e9
            margin[c] = (out[:, c] - other.max(1).values).mean().item()
    return {"uap_succ": succ, "uap_margin": margin}


# ----------------------------------------------------------------------------- extract command


def cmd_extract(args):
    load = load_model_fn(args.root)

    device = pick_device(args.device)
    print(f"[device] {device}")
    classes = read_classes(args.root)
    assert len(classes) == C, f"expected {C} classes, got {len(classes)}"
    x, y = load_clean(args.root, classes)
    os.makedirs(args.cache, exist_ok=True)

    models = list_models(args.root, args.test_dir)
    print(f"[models] train={sum(s == 'train' for s, _, _ in models)} test={sum(s == 'test' for s, _, _ in models)}")
    checked_head = False
    t0 = time.time()
    done = 0
    for i, (split, mid, path) in enumerate(models):
        out_path = os.path.join(args.cache, f"{split}__{mid}.npz")
        if os.path.exists(out_path) and not args.force:
            continue
        net = load(path).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)

        head = find_head(net)
        feats = {}
        hook = None
        if head is not None:
            hook = head[1].register_forward_hook(lambda m, inp, out: feats.__setitem__("f", inp[0].detach()))

        with torch.no_grad():
            L = net(x.to(device)).float().cpu()
        if hook is not None:
            hook.remove()

        data = {}
        s, A, acc = logit_stats(L, y)
        data.update(s)
        data["A"] = A
        data["clean_acc"] = np.float32(acc)

        if head is not None and "f" in feats and feats["f"].ndim == 2:
            f = feats["f"].float().cpu()
            W = head[1].weight.detach().float().cpu()
            if not checked_head:  # sanity check: logit ≈ 16·cos(features, weight row)
                rec = 16 * F.normalize(f, dim=1) @ F.normalize(W, dim=1).T
                err = (rec - L).abs().max().item()
                print(f"[head] module '{head[0]}', D={W.shape[1]}, max|16·cos - logit| = {err:.4f}")
                checked_head = True
            fs, Wn = feat_stats(f, y, W)
            data.update(fs)
            data["Wn"] = Wn
        elif not checked_head:
            print("[head] WARNING: no (20, D) head found - skipping weight/feature statistics")
            checked_head = True

        if not args.no_uap:
            data.update(uap_stats(net, x, y, device, args.eps / 255.0, args.steps))

        np.savez(out_path, **data)
        done += 1
        el = time.time() - t0
        print(f"[{i + 1}/{len(models)}] {split}/{mid} acc={acc:.3f}  ({el / done:.1f}s/model)", flush=True)

    print(f"[extract] done, cache in {args.cache}/")
