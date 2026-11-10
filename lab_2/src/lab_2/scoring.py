"""
P2 Backdoors - scoring: population calibration, AUC evaluation and submission.

Combines the per-model statistics produced by `detection.cmd_extract` into a
single per-(model, class) suspicion score (robust z-score calibration across
the population), evaluates AUC on the (tiny) labelled train set as a sanity
check, and writes the submission CSV for the test set.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

from lab_2.data import C, read_classes, read_train_labels

# ----------------------------------------------------------------------------- calibration / AUC


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


# ----------------------------------------------------------------------------- score command


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

    # same idea, but for the per-source UAP transfer success/margin (targeted backdoors):
    # a backdoor that only fires for one source class is drowned out by uap_succ/uap_margin's
    # single perturbation fit on all other classes at once, but shows up here as a single
    # large (source, target) entry.
    for raw_name, z_name in [("uap_src_succ", "uap_maxsrc_succ"), ("uap_src_margin", "uap_maxsrc_margin")]:
        if all(raw_name in r for r in raw):
            US = np.stack([r[raw_name] for r in raw]).astype(np.float64)  # (M, C, C): [m, s, c]
            ZUS = robust_z(US, axis=0)
            ZUS[:, idx, idx] = -np.inf  # s == c is never computed (always 0) - exclude from the max
            Z[z_name] = ZUS.max(axis=1)

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
