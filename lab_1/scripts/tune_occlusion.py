"""Grid-search the occlusion predictor's parameters (WIN, FILL, MASK_RATIO, MARGIN_MAX) on train.

Best config is saved to artifacts/occlusion.joblib (read by predict.py).
Deltas are the expensive part, so they are computed once per (win, fill) and cached as .npz;
MASK_RATIO and MARGIN_MAX are then swept for free. Scores use the official `punkty`.

Usage: uv run python scripts/tune_occlusion.py [--n 500] [--wins 12,16,20,24] [--fills 0.5]
           [--ratios 0.3,0.4,0.5,0.6,0.7,0.8,0.9] [--margins 0.5,1,1.5,2,3,5,inf] [--stride 2]
"""

import argparse
import itertools

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lab_1.data import ROOT, get_device, load_images, load_labels, load_model
from lab_1.occlusion import box, compute_deltas, heatmap
from lab_1.scoring import BOX_COLS, evaluate

ART = ROOT / "artifacts"


@torch.no_grad()
def collect(model, df: pd.DataFrame, win: int, stride: int, fill: float, chunk: int, name: str) -> dict:
    """Per-image top-2 logit margin and occlusion deltas, cached on disk."""
    path = ART / f"{name}_w{win}_s{stride}_f{fill}.npz"
    if path.exists():
        z = np.load(path)
        return {"margin": z["margin"], "deltas": z["deltas"]}
    device = next(model.parameters()).device
    margins, deltas = [], []
    for id_ in tqdm(df.id, desc=f"win={win} fill={fill}"):
        img = load_images("train", [id_])[0].to(device)
        top2 = model(img[None])[0].topk(2).values
        margins.append(float(top2[0] - top2[1]))
        deltas.append(compute_deltas(model, img, win, stride, fill, chunk).cpu().numpy())
    res = {"margin": np.array(margins), "deltas": np.stack(deltas)}
    np.savez(path, **res)  # ty: ignore[invalid-argument-type]
    return res


def predict(res: dict, win: int, stride: int, ratio: float, margin_max: float) -> np.ndarray:
    out = np.zeros((len(res["margin"]), 4), dtype=int)
    for i, (m, d) in enumerate(zip(res["margin"], res["deltas"])):
        if m < margin_max:
            out[i] = box(heatmap(torch.from_numpy(d), win, stride), ratio)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--wins", default="12,16,20,24")
    ap.add_argument("--fills", default="0.5")
    ap.add_argument("--ratios", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--margins", default="0.5,1,1.5,2,3,5,inf")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    floats = lambda s: [float(v) for v in s.split(",")]
    wins, fills, ratios, margins = (
        [int(v) for v in args.wins.split(",")],
        *map(floats, (args.fills, args.ratios, args.margins)),
    )
    ART.mkdir(exist_ok=True)

    device = get_device()
    print("device:", device)
    model = load_model(device).requires_grad_(False)
    df = load_labels("train").sample(args.n, random_state=0).reset_index(drop=True)

    rows = []
    for win, fill in itertools.product(wins, fills):
        res = collect(model, df, win, args.stride, fill, args.chunk, f"occ{args.n}")
        for ratio, margin_max in itertools.product(ratios, margins):
            pred = predict(res, win, args.stride, ratio, margin_max)
            sub = pd.concat([df[["id"]], pd.DataFrame(pred, columns=BOX_COLS)], axis=1)
            rows.append(
                {"win": win, "fill": fill, "ratio": ratio, "margin_max": margin_max, "mean": evaluate(df, sub)["mean"]}
            )

    table = pd.DataFrame(rows).sort_values("mean", ascending=False)
    print(table.head(args.top).to_markdown(index=False, floatfmt=".4f"))
    best = table.iloc[0]
    print(
        f"\nbest: WIN={int(best.win)} FILL={best.fill} MASK_RATIO={best.ratio} "
        f"MARGIN_MAX={best.margin_max} -> {best['mean']:.4f}"
    )
    table.to_csv(ART / f"tune_occlusion_{args.n}.csv", index=False)
    cfg = {
        "win": int(best.win),
        "stride": args.stride,
        "fill": float(best.fill),
        "mask_ratio": float(best.ratio),
        "margin_max": float(best.margin_max),
    }
    joblib.dump(cfg, ART / "occlusion.joblib")
    print("saved", ART / "occlusion.joblib")


if __name__ == "__main__":
    main()
