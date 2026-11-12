"""Stage 7 + 9: normalise scores, take the pair score and write the submission CSV.

Normalisation: for every (trigger, score type, class) subtract the median over the 200 test models and divide by
MAD (floored). About half the models are clean, so the median describes the background. The pair score is the
maximum of the normalised values over triggers and both score types.

Currently the only score source is the stage-4 quick test (`artifacts/quicktest_test.pt`).
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from lab_2.config import N_CLASSES, NORM, NormCfg
from lab_2.quicktest import roc_auc


def robust_stats(scores: torch.Tensor, cfg: NormCfg = NORM) -> tuple[torch.Tensor, torch.Tensor]:
    """Median and floored MAD over the model axis (0). scores: [n_models, n_triggers, 2, 20]."""
    med = scores.median(dim=0).values
    mad = cfg.mad_scale * (scores - med).abs().median(dim=0).values
    return med, mad.clamp(min=cfg.mad_floor)


def normalise(scores: torch.Tensor, med: torch.Tensor, mad: torch.Tensor) -> torch.Tensor:
    return (scores - med) / mad


def pair_scores(scores: torch.Tensor, med: torch.Tensor, mad: torch.Tensor) -> torch.Tensor:
    """[n_models, 20]: max of normalised scores over triggers and score types."""
    return normalise(scores, med, mad).amax(dim=(1, 2))


def check_submission(df: pd.DataFrame, expected_ids: list[str]) -> None:
    assert list(df.columns) == ["id"] + [f"c{c:02d}" for c in range(N_CLASSES)], "wrong columns"
    assert df["id"].is_unique, "duplicate ids"
    assert set(df["id"]) == set(expected_ids), "ids differ from the test models"
    values = df.drop(columns="id").to_numpy()
    assert all(math.isfinite(v) for v in values.ravel()), "nan/inf in scores"


def write_csv(ids: list[str], pairs: torch.Tensor, path: Path, expected_ids: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(pairs.numpy(), columns=[f"c{c:02d}" for c in range(N_CLASSES)])
    df.insert(0, "id", ids)
    check_submission(df, expected_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--art", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/submission_quicktest.csv"))
    args = ap.parse_args()

    test = torch.load(args.art / "quicktest_test.pt", weights_only=True)
    med, mad = robust_stats(test["scores"])
    pairs = pair_scores(test["scores"], med, mad)

    expected = sorted(p.stem for p in (args.root / "test" / "models").glob("*.pt"))
    df = write_csv(test["ids"], pairs, args.out, expected)
    print(f"wrote {args.out}: {len(df)} rows, score range [{pairs.min():.2f}, {pairs.max():.2f}]")

    # sanity check on the 6 labelled train models, normalised with the TEST statistics
    train_path = args.art / "quicktest_train.pt"
    if train_path.exists():
        tr = torch.load(train_path, weights_only=True)
        lab = pd.read_csv(args.root / "example/train/labels.csv", dtype={"id": str, "targets": str}).set_index("id")
        p = pair_scores(tr["scores"], med, mad)
        truth = torch.zeros_like(p, dtype=torch.bool)
        for i, mid in enumerate(tr["ids"]):
            tg = lab.loc[mid, "targets"]
            if isinstance(tg, str) and tg.strip():
                for c in tg.split(";"):
                    truth[i, int(c)] = True
        print(f"train ROC-AUC after normalisation: {roc_auc(p[truth], p[~truth]):.3f}")


if __name__ == "__main__":
    main()
