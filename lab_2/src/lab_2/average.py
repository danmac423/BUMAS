"""Stage 2: the averaged reference model (plan, section 2).

All models are fine-tunes of one base, so the mean of the 200 test models is an almost clean reference.
The only thing that depends on the seed is the output layer, and it is a cosine classifier, so only the
direction of each row matters: rows are normalised to unit length before averaging.

Outputs (in --out, default ./artifacts):
    mean_model.pt   state dict of the averaged model (float32)
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from lab_2.data import load_clean, split_ab
from lab_2.model import Net, load
from lab_2.prepare import accuracy

FC_WEIGHT = "r.fc.weight"


def average_state_dicts(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Streaming mean of float parameters/buffers, with unit-norm rows for the cosine layer.

    Integer buffers (BatchNorm's num_batches_tracked) are taken from the first model.
    """
    total: dict[str, torch.Tensor] = {}
    first: dict[str, torch.Tensor] = {}
    for p in tqdm(paths, desc="averaging"):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if not v.is_floating_point():
                first.setdefault(k, v)
                continue
            v = v.float()
            if k == FC_WEIGHT:
                v = torch.nn.functional.normalize(v, dim=1)
            total[k] = total.get(k, torch.zeros_like(v)) + v
    mean = {k: v / len(paths) for k, v in total.items()}
    mean[FC_WEIGHT] = torch.nn.functional.normalize(mean[FC_WEIGHT], dim=1)  # direction is all that matters
    return {**mean, **first}


def build_mean_model(state: dict[str, torch.Tensor], device: str = "cpu") -> Net:
    net = Net()
    net.load_state_dict(state)
    return net.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    test_paths = sorted((args.root / "test" / "models").glob("*.pt"))
    state = average_state_dicts(test_paths)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out / "mean_model.pt")

    x, y, _ = load_clean(args.root / "example" / "clean")
    idx_a, idx_b = split_ab(y)
    ok = accuracy(build_mean_model(state, args.device), x, y, args.device)
    print(f"mean model acc={ok.float().mean():.4f}  A={ok[idx_a].float().mean():.4f}  B={ok[idx_b].float().mean():.4f}")

    acc_path = args.out / "accuracy.csv"
    if acc_path.exists():
        t = pd.read_csv(acc_path).query("group == 'test'")["acc"]
        print(f"single test models: mean={t.mean():.4f} min={t.min():.4f} max={t.max():.4f}")


if __name__ == "__main__":
    main()
