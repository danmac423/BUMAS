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
    """[n_triggers, 20, 20] flip tables of the averaged model."""
    return torch.stack([flip_table(mean_net, x, y, t) for t in bank])


def model_scores(net, bank: list[FixedTrigger], ref: torch.Tensor, x, y) -> torch.Tensor:
    """[n_triggers, 2, 20] scores of `net - averaged model` for every trigger."""
    out = []
    for t, ref_table in zip(bank, ref):
        s = column_scores(flip_table(net, x, y, t) - ref_table)
        out.append(torch.stack([s["all"], s["targeted"]]))
    return torch.stack(out)


def pair_scores(scores: torch.Tensor) -> torch.Tensor:
    """Raw (un-normalised) pair score: max over triggers and both score types -> [n_models, 20]."""
    return scores.amax(dim=(1, 2))


def roc_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Same definition as the task: ties count half."""
    gt = (pos[:, None] > neg[None, :]).float().sum()
    eq = (pos[:, None] == neg[None, :]).float().sum()
    return ((gt + 0.5 * eq) / (len(pos) * len(neg))).item()


def validate(ids: list[str], scores: torch.Tensor, bank: list[FixedTrigger], labels_csv: Path) -> None:
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
            k, kind = divmod(int(scores[i, :, :, c].argmax()), 2)
            info += f" | target c{c:02d}: rank {rank}/20, best trigger {bank[k].name} ({'all' if kind == 0 else 'targeted'})"
        print(f"{mid} [{'BACKDOOR' if true_cls else 'clean   '}] top: {top}{info}")
    print(f"raw ROC-AUC over all pairs: {roc_auc(pairs[truth], pairs[~truth]):.3f}")
    print(f"max score clean models {pairs[~truth.any(1)].max():.2f}  vs backdoored {pairs[truth].min():.2f} (min target)")


def main() -> None:
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
