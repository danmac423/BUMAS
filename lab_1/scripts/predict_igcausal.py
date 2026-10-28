"""IG + causal-test predictions for the test set -> CSV (id,x_min,y_min,x_max,y_max).

Needs artifacts/igcausal.joblib from scripts/tune_igcausal.py.
Usage: uv run python scripts/predict_igcausal.py [out.csv] [batch]
"""

import sys

import joblib
import pandas as pd

from lab_1.data import ROOT, get_device, load_model, load_test_ids
from lab_1.igcausal import run
from lab_1.scoring import BOX_COLS


def main(out: str = "submission_igcausal.csv", batch: str = "16") -> None:
    cfg = joblib.load(ROOT / "artifacts" / "igcausal.joblib")
    device = get_device()
    print("device:", device, "| filter:", cfg["filter"], "| lam:", cfg["lam"])
    model = load_model(device).requires_grad_(False)
    ids = load_test_ids().id.tolist()
    res = run(model, "test", ids, cfg["filter"], [cfg["lam"]], batch=int(batch))
    pred = cfg["decider"].decide(res["feats"], res[cfg["lam"]])
    sub = pd.concat([pd.DataFrame({"id": ids}), pd.DataFrame(pred, columns=BOX_COLS)], axis=1)
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {(pred.any(1)).sum()} boxes, {(~pred.any(1)).sum()} none")


if __name__ == "__main__":
    main(*sys.argv[1:3])
