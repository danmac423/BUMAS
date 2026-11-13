"""Stage 7 + 9: normalise scores, take the pair score and write the submission CSV.

Normalisation: for every (trigger, score type, class) subtract the median over the 200 test models and divide by
MAD (floored). About half the models are clean, so the median describes the background. The pair score is the
maximum of the normalised values over triggers and both score types.

Score sources (`--sources`): `quicktest` = stage 4 bank (`artifacts/quicktest_<group>.pt`, 80 triggers) and
`stage5` = the learned triggers (`artifacts/stage5/<group>/<id>_<family>_r<round>.pt`, one "trigger" per family and
round). Both have shape [n_models, n_triggers, 2, 20], so they are concatenated along the trigger axis and share the
same normalisation and maximum.
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from lab_2.config import N_CLASSES, NORM, NormCfg
from lab_2.optimise import result_path
from lab_2.quicktest import roc_auc
from lab_2.triggers import FAMILIES


def load_stage5(art: Path, group: str, ids: list[str], rounds: int = 1) -> torch.Tensor:
    """[n_models, n_families * rounds, 2, 20] scores of the stage-5 results, in the order of `ids`."""
    per_model = []
    for mid in ids:
        rows = []
        for fam in sorted(FAMILIES):
            for rnd in range(rounds):
                path = result_path(art, group, mid, fam, rnd)
                if not path.exists():
                    raise FileNotFoundError(f"missing stage-5 result {path} (run lab_2.optimise first)")
                rows.append(torch.load(path, weights_only=True)["scores"])
        per_model.append(torch.stack(rows))
    return torch.stack(per_model)


def load_scores(art: Path, group: str, sources: list[str], rounds: int = 1) -> tuple[list[str], torch.Tensor]:
    """Ids and the concatenated scores of the chosen sources for a group ('train' or 'test')."""
    ids, parts = None, []
    if "quicktest" in sources:
        qt = torch.load(art / f"quicktest_{group}.pt", weights_only=True)
        ids = qt["ids"]
        parts.append(qt["scores"])
    if "stage5" in sources:
        if ids is None:
            models = Path("example/train/models") if group == "train" else Path("test/models")
            ids = sorted(p.stem for p in models.glob("*.pt"))
        parts.append(load_stage5(art, group, ids, rounds))
    return ids, torch.cat(parts, dim=1)


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
    ap.add_argument("--sources", nargs="+", choices=["quicktest", "stage5"], default=["quicktest", "stage5"])
    ap.add_argument("--rounds", type=int, default=1, help="stage-5 rounds to include (stage 6 adds more)")
    ap.add_argument("--out", type=Path, default=None, help="default: artifacts/submission_<sources>.csv")
    args = ap.parse_args()
    out = args.out or args.art / f"submission_{'_'.join(args.sources)}.csv"

    ids, scores = load_scores(args.art, "test", args.sources, args.rounds)
    med, mad = robust_stats(scores)
    pairs = pair_scores(scores, med, mad)

    expected = sorted(p.stem for p in (args.root / "test" / "models").glob("*.pt"))
    df = write_csv(ids, pairs, out, expected)
    print(f"wrote {out}: {len(df)} rows from {scores.shape[1]} triggers, score range [{pairs.min():.2f}, {pairs.max():.2f}]")

    # sanity check on the 6 labelled train models, normalised with the TEST statistics
    # (stage-5 train results were made with fewer steps than the test ones, so treat them as a rough check)
    try:
        tr_ids, tr_scores = load_scores(args.art, "train", args.sources, args.rounds)
    except FileNotFoundError:
        tr_ids = None
    if tr_ids is not None:
        lab = pd.read_csv(args.root / "example/train/labels.csv", dtype={"id": str, "targets": str}).set_index("id")
        p = pair_scores(tr_scores, med, mad)
        truth = torch.zeros_like(p, dtype=torch.bool)
        for i, mid in enumerate(tr_ids):
            tg = lab.loc[mid, "targets"]
            if isinstance(tg, str) and tg.strip():
                for c in tg.split(";"):
                    truth[i, int(c)] = True
        print(f"train ROC-AUC after normalisation: {roc_auc(p[truth], p[~truth]):.3f}")


if __name__ == "__main__":
    main()
