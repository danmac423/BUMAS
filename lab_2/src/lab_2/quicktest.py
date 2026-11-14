"""Stage 4: quick test with the fixed trigger bank (plan, section 4).

For every model and every trigger of the bank: flip table on all 400 clean images, minus the flip table of the
averaged model, reduced to per-class 'all' / 'targeted' scores.

Output `artifacts/quicktest_<group>.pt`:
    ids            list[str], model ids
    trigger_names  list[str]
    scores         float tensor [n_models, n_triggers, 2, 20]; axis 2 = (all, targeted)

For the train group the script also prints a validation against `example/train/labels.csv`.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from lab_2.average import build_mean_model
from lab_2.config import N_CLASSES
from lab_2.data import load_clean
from lab_2.flips import column_scores, flip_table
from lab_2.model import load
from lab_2.triggers import FixedTrigger, build_bank


def reference_tables(mean_net, bank: list[FixedTrigger], x, y) -> torch.Tensor:
    """Computes the flip table of the averaged model for every trigger once, so it can be reused for all models.

    Args:
        mean_net: Averaged reference model.
        bank: Fixed triggers.
        x: Clean images (N, 3, 64, 64).
        y: Their labels (N,).

    Returns:
        Tensor (n_triggers, 20, 20).
    """
    return torch.stack([flip_table(mean_net, x, y, t) for t in bank])


def model_scores(net, bank: list[FixedTrigger], ref: torch.Tensor, x, y) -> torch.Tensor:
    """Scores one model against the averaged model for every trigger of the bank.

    Args:
        net: Examined model.
        bank: Fixed triggers.
        ref: Flip tables of the averaged model, from `reference_tables`.
        x: Clean images (N, 3, 64, 64).
        y: Their labels (N,).

    Returns:
        Tensor (n_triggers, 2, 20); axis 1 is (all, targeted).
    """
    out = []
    for t, ref_table in zip(bank, ref):
        s = column_scores(flip_table(net, x, y, t) - ref_table)
        out.append(torch.stack([s["all"], s["targeted"]]))
    return torch.stack(out)


def pair_scores(scores: torch.Tensor) -> torch.Tensor:
    """Computes the raw (un-normalised) score of every model x class pair.

    Args:
        scores: Tensor (n_models, n_triggers, 2, 20).

    Returns:
        Tensor (n_models, 20): maximum over triggers and both score types.
    """
    return scores.amax(dim=(1, 2))


def roc_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Computes ROC-AUC with the definition used by the task server.

    Args:
        pos: Scores of positive pairs (P,).
        neg: Scores of negative pairs (M,).

    Returns:
        Probability that a random positive scores higher than a random negative; ties count half.
    """
    # all P x M comparisons at once through broadcasting
    gt = (pos[:, None] > neg[None, :]).float().sum()
    eq = (pos[:, None] == neg[None, :]).float().sum()
    return ((gt + 0.5 * eq) / (len(pos) * len(neg))).item()


def validate(ids: list[str], scores: torch.Tensor, bank: list[FixedTrigger], labels_csv: Path) -> None:
    """Prints how well the quick test finds the true targets of the labelled training models.

    For every model it prints the top-ranked classes and, for each true target, its rank and the trigger and
    score type that gave the best value; at the end the raw ROC-AUC over all pairs.

    Args:
        ids: Model ids, in the order of `scores`.
        scores: Tensor (n_models, n_triggers, 2, 20) from `model_scores`.
        bank: The triggers, in the order of axis 1 of `scores`.
        labels_csv: Path to `train/labels.csv` with the `targets` column (classes separated by ";").
    """
    lab = pd.read_csv(labels_csv, dtype={"id": str, "targets": str}).set_index("id")
    pairs = pair_scores(scores)
    truth = torch.zeros_like(pairs, dtype=torch.bool)
    for i, mid in enumerate(ids):
        tg = lab.loc[mid, "targets"]
        if isinstance(tg, str) and tg.strip():
            for c in tg.split(";"):
                truth[i, int(c)] = True

    for i, mid in enumerate(ids):
        order = pairs[i].argsort(descending=True)
        top = ", ".join(f"c{c:02d}={pairs[i, c]:.2f}" for c in order[:4].tolist())
        true_cls = truth[i].nonzero().flatten().tolist()
        info = ""
        for c in true_cls:
            rank = int((pairs[i] > pairs[i, c]).sum()) + 1
            # argmax runs over the flattened (trigger, score type) axes: index = 2 * trigger + type
            k, kind = divmod(int(scores[i, :, :, c].argmax()), 2)
            info += f" | target c{c:02d}: rank {rank}/20, best trigger {bank[k].name} ({'all' if kind == 0 else 'targeted'})"
        print(f"{mid} [{'BACKDOOR' if true_cls else 'clean   '}] top: {top}{info}")
    print(f"raw ROC-AUC over all pairs: {roc_auc(pairs[truth], pairs[~truth]):.3f}")
    print(f"max score clean models {pairs[~truth.any(1)].max():.2f}  vs backdoored {pairs[truth].min():.2f} (min target)")


def main() -> None:
    """Command-line entry point: runs the fixed trigger bank on a group of models and saves the scores."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--group", choices=["train", "test"], default="train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    x, y, _ = load_clean(args.root / "example" / "clean")
    x = x.to(args.device)
    bank = build_bank()
    mean_net = build_mean_model(torch.load(args.out / "mean_model.pt", weights_only=True), args.device)
    ref = reference_tables(mean_net, bank, x, y)

    sub = args.root / ("example/train/models" if args.group == "train" else "test/models")
    paths = sorted(sub.glob("*.pt"))
    scores = torch.stack(
        [model_scores(load(p, args.device), bank, ref, x, y) for p in tqdm(paths, desc=f"quicktest {args.group}")]
    )
    ids = [p.stem for p in paths]
    torch.save(
        {"ids": ids, "trigger_names": [t.name for t in bank], "scores": scores.cpu()},
        args.out / f"quicktest_{args.group}.pt",
    )
    print(f"{len(bank)} triggers, scores {tuple(scores.shape)}, classes={N_CLASSES}")
    if args.group == "train":
        validate(ids, scores.cpu(), bank, args.root / "example" / "train" / "labels.csv")


if __name__ == "__main__":
    main()
