#!/usr/bin/env python3
"""
P2 Backdoors - method 1: cheap per-(model, class) statistics + population calibration.

Layout (relative to the lab_2 directory):
    example/                 <- --root (default)
        model.py, classes.txt, clean/images/, clean/labels.csv,
        train/models/*.pt, train/labels.csv
    test/models/*.pt         <- --test-dir (default)

Usage:
    # 1) compute statistics (cached per model in cache_m1/, can be interrupted and resumed)
    uv run python src/lab_2/backdoor_class_stats.py extract                       # with "uap" (recommended on GPU)
    uv run python src/lab_2/backdoor_class_stats.py extract --no-uap              # gradient-free statistics only (fast, e.g. on M1)

    # 2) calibration, AUC on training models, submission file
    uv run python src/lab_2/backdoor_class_stats.py score --out submission_m1.csv
    uv run python src/lab_2/backdoor_class_stats.py score --stats attract_maxsrc,uap_margin   # choose statistics to combine

All statistics are oriented so that HIGHER = MORE SUSPICIOUS.
"""

import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

C = 20  # number of classes

# ----------------------------------------------------------------------------- utils


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_classes(root):
    with open(os.path.join(root, "classes.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def parse_class(tok, classes):
    tok = str(tok).strip()
    if tok.lstrip("-").isdigit():
        return int(tok)
    return classes.index(tok)


def load_clean(root, classes):
    """Loads the 400 clean images as a tensor (N,3,64,64) in [0,1] and labels (N,)."""
    df = pd.read_csv(os.path.join(root, "clean", "labels.csv"))
    img_dir = os.path.join(root, "clean", "images")
    files = sorted(os.listdir(img_dir))
    stem2file = {os.path.splitext(f)[0]: f for f in files}

    low = {c.lower(): c for c in df.columns}
    lcol = next((low[k] for k in ("label", "class", "y", "target", "labels") if k in low), df.columns[-1])
    fcol = next((low[k] for k in ("file", "filename", "image", "img", "path", "name", "id", "fname") if k in low), None)
    if fcol is None and len(df.columns) > 1:
        fcol = [c for c in df.columns if c != lcol][0]

    xs, ys = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        row = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        if fcol is not None:
            name = os.path.basename(str(row[fcol]))
            p = os.path.join(img_dir, name)
            if not os.path.exists(p):
                stem = os.path.splitext(name)[0]
                if stem not in stem2file:
                    raise FileNotFoundError(f"Image for '{name}' not found in {img_dir}")
                p = os.path.join(img_dir, stem2file[stem])
        else:  # no filename column -> alphabetical order
            p = os.path.join(img_dir, files[i])
        img = Image.open(p).convert("RGB")
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.BILINEAR)
        xs.append(np.asarray(img, dtype=np.float32) / 255.0)
        ys.append(parse_class(row[lcol], classes))

    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ys, dtype=torch.long)
    counts = torch.bincount(y, minlength=C)
    print(f"[clean] {len(y)} images, per class: min={counts.min().item()} max={counts.max().item()}")
    return x, y


def read_train_labels(root, classes):
    df = pd.read_csv(os.path.join(root, "train", "labels.csv"), dtype=str)  # str: avoid "15.0" from NaN-induced floats
    idcol = "id" if "id" in df.columns else df.columns[0]
    out = {}
    for _, r in df.iterrows():
        t = r.get("targets", "")
        if pd.isna(t) or str(t).strip() == "":
            out[str(r[idcol])] = []
        else:
            out[str(r[idcol])] = [parse_class(s, classes) for s in str(t).split(";") if s.strip() != ""]
    return out


def list_models(root, test_dir):
    items = []
    for split, base in (("train", os.path.join(root, "train")), ("test", test_dir)):
        for p in sorted(glob.glob(os.path.join(base, "models", "*.pt"))):
            items.append((split, os.path.splitext(os.path.basename(p))[0], p))
    return items


def load_model_fn(root):
    """Imports `load` from <root>/model.py by file path (independent of the working directory)."""
    spec = importlib.util.spec_from_file_location("p2_model", os.path.join(root, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load


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


# ----------------------------------------------------------------------------- extract


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


# ----------------------------------------------------------------------------- score


def robust_z(X, axis=0, clip=15.0):
    med = np.median(X, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(X - med), axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    scale = np.where(mad > 1e-8, mad, std) + 1e-6  # e.g. fp_rate is often 0 -> MAD = 0
    return np.clip((X - med) / scale, -clip, clip)


def auc(pos, neg):
    pos = np.asarray(pos)[:, None]
    neg = np.asarray(neg)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def cmd_score(args):
    classes = read_classes(args.root)
    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    if not files:
        sys.exit(f"No files in {args.cache}/ - run 'extract' first.")
    keys, raw = [], []
    for fp in files:
        split, mid = os.path.splitext(os.path.basename(fp))[0].split("__", 1)
        keys.append((split, mid))
        raw.append(dict(np.load(fp)))
    M = len(keys)
    print(f"[score] population: {M} models")

    # --- per-(model, class) statistics, all oriented so that higher = more suspicious
    base_names = ["attract_logit", "attract_logprob", "second_rank", "fp_rate", "w_misalign", "uap_succ", "uap_margin"]
    Z = {}
    for n in base_names:
        if all(n in r for r in raw):
            Z[n] = robust_z(np.stack([r[n] for r in raw]).astype(np.float64), axis=0)

    # attraction from the MOST suspicious source class (targeted poisoning):
    # z-score per (source, target) pair across the population, then max over sources
    A = np.stack([r["A"] for r in raw]).astype(np.float64)  # (M, C, C)
    ZA = robust_z(A, axis=0)
    idx = np.arange(C)
    ZA[:, idx, idx] = -np.inf
    Z["attract_maxsrc"] = ZA.max(axis=1)

    # deviation of a weight row from the population's "typical" row for that class
    if all("Wn" in r for r in raw):
        Wn = np.stack([r["Wn"] for r in raw]).astype(np.float32)  # (M, C, D)
        med = np.median(Wn, axis=0)
        med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8
        dev = 1.0 - (Wn * med[None]).sum(-1)  # (M, C)
        Z["w_pop_dev"] = robust_z(dev.astype(np.float64), axis=0)

    names = list(Z.keys())
    use = args.stats.split(",") if args.stats else names
    missing = [u for u in use if u not in Z]
    if missing:
        sys.exit(f"Unknown/unavailable statistics: {missing}. Available: {names}")
    combo = np.mean([Z[u] for u in use], axis=0)
    Z["COMBO"] = combo

    # --- evaluation on training models
    tl = read_train_labels(args.root, classes)
    tr_rows = [i for i, (s, m) in enumerate(keys) if s == "train" and m in tl]
    if tr_rows:
        lab = np.zeros((len(tr_rows), C), dtype=bool)
        for j, i in enumerate(tr_rows):
            for t in tl[keys[i][1]]:
                lab[j, t] = True
        print(
            f"\n[train] {len(tr_rows)} models, {lab.sum()} positive pairs "
            f"(WARNING: very few - AUC here is only a sanity check)"
        )
        print(f"{'statistic':<18} {'AUC':>6}   rank of target in its own model (1 = most suspicious)")
        for n in names + ["COMBO"]:
            S = Z[n][tr_rows]
            a = auc(S[lab], S[~lab])
            ranks = []
            for j in range(len(tr_rows)):
                order = np.argsort(-S[j])
                for t in np.where(lab[j])[0]:
                    ranks.append(int(np.where(order == t)[0][0]) + 1)
            mark = " <- used in COMBO" if n in use else ""
            print(f"{n:<18} {a:6.3f}   {ranks}{mark}")
        print(
            "\n[train] max z-score per model (COMBO): "
            + ", ".join(f"{keys[i][1][:8]}:{'BD' if tl[keys[i][1]] else 'clean'}={combo[i].max():.2f}" for i in tr_rows)
        )

    # --- label-free diagnostic: does the per-model max z on the test set look bimodal (~half clean)?
    te_rows = [i for i, (s, _) in enumerate(keys) if s == "test"]
    if te_rows:
        mx = combo[te_rows].max(1)
        qs = np.quantile(mx, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"\n[test] max COMBO per model - quantiles 10/25/50/75/90%: " + " ".join(f"{q:.2f}" for q in qs))
        hist, edges = np.histogram(mx, bins=12)
        for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
            print(f"  {e0:6.2f}..{e1:6.2f} | " + "#" * int(h))

        sub = pd.DataFrame(
            np.nan_to_num(combo[te_rows], nan=0.0, posinf=50.0, neginf=-50.0), columns=[f"c{k:02d}" for k in range(C)]
        )
        sub.insert(0, "id", [keys[i][1] for i in te_rows])
        sub.to_csv(args.out, index=False, float_format="%.5f")
        print(f"\n[write] {args.out}: {len(sub)} models, statistics: {use}")
    else:
        print("\n[test] no test models in cache - not writing a submission")


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--root", default="example", help="dir with model.py, classes.txt, clean/, train/")
    e.add_argument("--test-dir", default="test", help="dir with models/*.pt of the test set")
    e.add_argument("--cache", default="cache_m1")
    e.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    e.add_argument("--no-uap", action="store_true", help="skip gradient-based statistics (faster)")
    e.add_argument("--eps", type=float, default=6.0, help="L_inf perturbation budget, in units of /255")
    e.add_argument("--steps", type=int, default=20, help="PGD steps per class")
    e.add_argument("--force", action="store_true", help="recompute even if the cache exists")

    s = sub.add_parser("score")
    s.add_argument("--root", default="example", help="dir with model.py, classes.txt, clean/, train/")
    s.add_argument("--cache", default="cache_m1")
    s.add_argument("--stats", default="", help="comma-separated statistics for COMBO (default: all)")
    s.add_argument("--out", default="submission_m1.csv")

    args = ap.parse_args()
    {"extract": cmd_extract, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
