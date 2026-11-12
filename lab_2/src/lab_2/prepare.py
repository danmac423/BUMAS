"""Stage 1 entry point: load every model, measure clean accuracy, save the A/B split.

Outputs (in --out, default ./artifacts):
    split.pt        {"idx_a", "idx_b", "image_ids"}
    accuracy.csv    id, group (train/test), acc, acc_a, acc_b
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from lab_2.config import SPLIT_SEED
from lab_2.data import load_clean, split_ab
from lab_2.model import load


@torch.no_grad()
def accuracy(net, x: torch.Tensor, y: torch.Tensor, device: str, batch: int = 200) -> torch.Tensor:
    """Per-image correctness (N,) bool."""
    preds = [net(x[i : i + batch].to(device)).argmax(1).cpu() for i in range(0, len(x), batch)]
    return torch.cat(preds) == y


def model_paths(root: Path) -> list[tuple[str, Path]]:
    out = []
    for group, sub in (("train", root / "example" / "train" / "models"), ("test", root / "test" / "models")):
        out += [(group, p) for p in sorted(sub.glob("*.pt"))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    x, y, ids = load_clean(args.root / "example" / "clean")
    idx_a, idx_b = split_ab(y, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"idx_a": idx_a, "idx_b": idx_b, "image_ids": ids}, args.out / "split.pt")
    print(f"clean images: {len(x)}  |A|={len(idx_a)}  |B|={len(idx_b)}")

    rows = []
    for group, path in tqdm(model_paths(args.root), desc="models"):
        ok = accuracy(load(path, args.device), x, y, args.device)
        rows.append(
            {
                "id": path.stem,
                "group": group,
                "acc": ok.float().mean().item(),
                "acc_a": ok[idx_a].float().mean().item(),
                "acc_b": ok[idx_b].float().mean().item(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "accuracy.csv", index=False)
    print(df.groupby("group")["acc"].describe().to_string())


if __name__ == "__main__":
    main()
