"""Tune the IG + causal-test localizer on train and fit its patch/no-patch decider.

1. Split train into fit (4000) / val (1000, fold 0).
2. On a subset of fit: steps 0-4 for every filter and lambda, mean IoU(box, truth) on patched images.
3. Best (filter, lambda) on all of fit and val -> logistic regression on fit, p -> E[IoU] table on val.
4. Official `punkty` on val. Saves artifacts/igcausal.joblib; stage results are cached as .npz next to it.

Usage: uv run python scripts/tune_igcausal.py [--n-tune 500] [--n-fit 4000] [--lams 0,0.002,0.005,0.01]
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from lab_1.data import ROOT, add_folds, boxes, get_device, load_labels, load_model
from lab_1.igcausal import FILTERS, Decider, box_iou_np, run
from lab_1.scoring import BOX_COLS, evaluate

ART = ROOT / "artifacts"


def cached_run(model, df: pd.DataFrame, filt: str, lams: list[float], name: str, batch: int) -> dict:
    path = ART / f"{name}_{filt}.npz"
    if path.exists():
        z = np.load(path)
        res = {k: z[k] for k in ("feats", "best")} | {lam: z[f"lam_{lam}"] for lam in lams if f"lam_{lam}" in z}
        if all(lam in res for lam in lams):
            return res
    res = run(model, "train", df.id, filt, lams, batch)
    np.savez(path, feats=res["feats"], best=res["best"], **{f"lam_{lam}": res[lam] for lam in lams})
    return res


def patch_iou(df: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    """IoU with the true box on patched images only."""
    m = df.has_patch.to_numpy()
    return pd.Series(box_iou_np(boxes(df)[m], pred[m]), index=df.index[m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tune", type=int, default=500)
    ap.add_argument("--n-fit", type=int, default=4000, help="fit images for the decider (6 features need few)")
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--lams", default="0,0.002,0.005,0.01")
    ap.add_argument("--filters", default=",".join(FILTERS))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    lams = [float(v) for v in args.lams.split(",")]
    ART.mkdir(exist_ok=True)

    device = get_device()
    print("device:", device)
    model = load_model(device).requires_grad_(False)
    df = add_folds(load_labels("train"))
    fit, val = df[df.fold != 0].reset_index(drop=True), df[df.fold == 0].reset_index(drop=True)
    tune = fit.sample(args.n_tune, random_state=0).reset_index(drop=True)
    low = tune.eps[tune.has_patch] <= 8

    rows = []
    for filt in args.filters.split(","):
        res = cached_run(model, tune, filt, lams, f"tune{args.n_tune}", args.batch)
        for lam in ["best", *lams]:
            iou = patch_iou(tune, res[lam])
            rows.append({"filter": filt, "lam": lam, "iou": iou.mean(), "iou_eps<=8": iou[low].mean()})
    table = pd.DataFrame(rows)
    print(table.to_markdown(index=False, floatfmt=".4f"))
    pick = table[table.lam != "best"].sort_values("iou", ascending=False).iloc[0]
    filt, lam = pick["filter"], float(pick["lam"])
    print(f"chosen: filter={filt} lam={lam}")

    fit = fit.sample(min(args.n_fit, len(fit)), random_state=1).reset_index(drop=True)
    r_fit = cached_run(model, fit, filt, [lam], f"fit{len(fit)}", args.batch)
    val = val.sample(min(args.n_val, len(val)), random_state=2).reset_index(drop=True)
    r_val = cached_run(model, val, filt, [lam], f"val{len(val)}", args.batch)
    iou_val = box_iou_np(boxes(val), r_val[lam])
    dec = Decider.fit(r_fit["feats"], fit.has_patch.to_numpy(), r_val["feats"], val.has_patch.to_numpy(), iou_val)
    print("E[IoU] per p bin:", np.round(dec.eiou, 3))

    for name, pred in [
        ("decider", dec.decide(r_val["feats"], r_val[lam])),
        ("always box", r_val[lam]),
        ("always none", np.zeros_like(r_val[lam])),
    ]:
        sub = pd.concat([val[["id"]], pd.DataFrame(pred, columns=BOX_COLS)], axis=1)
        print(f"\nval {name}:", evaluate(val, sub))

    joblib.dump({"filter": filt, "lam": lam, "decider": dec}, ART / "igcausal.joblib")
    print("saved", ART / "igcausal.joblib")


if __name__ == "__main__":
    main()
