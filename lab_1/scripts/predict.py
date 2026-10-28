"""Occlusion-based predictions for the test set -> CSV (id,x_min,y_min,x_max,y_max).

Needs artifacts/occlusion.joblib from scripts/tune_occlusion.py.
Usage: uv run python scripts/predict.py [out.csv]   (resumes from out.csv.part if present)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from tqdm import tqdm

from lab_1.data import ROOT, get_device, load_images, load_model, load_test_ids
from lab_1.occlusion import box, compute_deltas, heatmap
from lab_1.scoring import BOX_COLS

CHUNK = 2048


@torch.no_grad()
def predict_one(model, img: torch.Tensor, cfg: dict) -> tuple[int, int, int, int]:
    top2 = model(img[None])[0].topk(2).values
    if top2[0] - top2[1] >= cfg["margin_max"]:
        return (0, 0, 0, 0)
    win, stride = cfg["win"], cfg["stride"]
    deltas = compute_deltas(model, img, win, stride, cfg["fill"], chunk=CHUNK)
    return box(heatmap(deltas, win, stride), cfg["mask_ratio"])


def main(out: str = "submission.csv", device: str | None = None) -> None:
    cfg = joblib.load(ROOT / "artifacts" / "occlusion.joblib")
    device = device or get_device()
    print("device:", device, "|", cfg)
    ids = load_test_ids().id.tolist()
    part = Path(out + ".part")
    rows = pd.read_csv(part).values.tolist() if part.exists() else []
    model = load_model(device)
    for n in tqdm(range(len(rows), len(ids)), initial=len(rows), total=len(ids), desc="predict"):
        img = load_images("test", [ids[n]])[0].to(device)
        rows.append([ids[n], *predict_one(model, img, cfg)])
        if (n + 1) % 250 == 0:
            pd.DataFrame(rows, columns=["id", *BOX_COLS]).to_csv(part, index=False)
    pd.DataFrame(rows, columns=["id", *BOX_COLS]).to_csv(out, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:2])
