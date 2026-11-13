"""Stage 5: optimise a learnable trigger so the examined model and the averaged model disagree (plan, section 5).

Objective (maximised): KL(p_model || p_mean) on the triggered half-A images, minus `penalty_weight * trigger.penalty()`.
Both networks are frozen; the gradient only reaches the trigger parameters. The result is judged on half B with the
flip table of stage 3 (examined model minus averaged model).

`exclude` lists classes removed from both distributions before the softmax (stage 6: hide an already found target).
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lab_2.average import build_mean_model
from lab_2.data import load_clean
from lab_2.flips import column_scores, flip_table
from lab_2.model import load
from lab_2.triggers import FAMILIES, LearnableTrigger


@dataclass(frozen=True)
class OptCfg:
    steps: int = 150
    lr: float = 0.1
    penalty_weight: float = 5.0  # only the patch has a non-zero penalty
    batch: int = 100  # images per step, sampled from half A
    seed: int = 0


def kl_disagreement(logits_m: torch.Tensor, logits_ref: torch.Tensor, exclude: tuple[int, ...] = ()) -> torch.Tensor:
    """Mean over the batch of KL(p_model || p_ref), with `exclude` classes removed before the softmax."""
    if exclude:
        keep = torch.ones(logits_m.shape[1], dtype=torch.bool, device=logits_m.device)
        keep[list(exclude)] = False
        logits_m, logits_ref = logits_m[:, keep], logits_ref[:, keep]
    lp_m, lp_ref = F.log_softmax(logits_m, 1), F.log_softmax(logits_ref, 1)
    return (lp_m.exp() * (lp_m - lp_ref)).sum(1).mean()


def optimise(
    net,
    mean_net,
    trigger: LearnableTrigger,
    x_a: torch.Tensor,
    cfg: OptCfg = OptCfg(),
    exclude: tuple[int, ...] = (),
) -> list[float]:
    """Run Adam on the trigger parameters; returns the KL value per step."""
    device = next(net.parameters()).device
    trigger.to(device).reset(cfg.seed)
    for m in (net, mean_net):
        m.eval().requires_grad_(False)
    opt = torch.optim.Adam(trigger.parameters(), lr=cfg.lr)
    g = torch.Generator().manual_seed(cfg.seed)
    history = []
    for _ in range(cfg.steps):
        idx = torch.randperm(len(x_a), generator=g)[: cfg.batch]
        xt = trigger(x_a[idx].to(device))
        kl = kl_disagreement(net(xt), mean_net(xt), exclude)
        loss = -kl + cfg.penalty_weight * trigger.penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(kl.item())
    return history


@torch.no_grad()
def evaluate(net, mean_net, trigger, x_b, y_b) -> dict[str, torch.Tensor]:
    """Column scores of the flip table difference on half B."""
    diff = flip_table(net, x_b, y_b, trigger) - flip_table(mean_net, x_b, y_b, trigger)
    return column_scores(diff)


def result_path(out: Path, group: str, model_id: str, family: str, rnd: int = 0) -> Path:
    return out / "stage5" / group / f"{model_id}_{family}_r{rnd}.pt"


def run_one(net, mean_net, family: str, x, y, ia, ib, cfg: OptCfg, exclude: tuple[int, ...] = ()) -> dict:
    """Optimise one family on one model and score it on half B -> the dict that gets saved."""
    trigger = FAMILIES[family]()
    hist = optimise(net, mean_net, trigger, x[ia], cfg, exclude)
    s = evaluate(net, mean_net, trigger.eval(), x[ib], y[ib])
    return {
        "scores": torch.stack([s["all"], s["targeted"]]).cpu(),  # [2, 20]
        "kl": hist[-1],
        "kl_start": hist[0],
        "exclude": list(exclude),
        "trigger_state": {k: v.cpu() for k, v in trigger.state_dict().items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="stage 5: optimise trigger families on a group of models, save results")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--group", choices=["train", "test"], default="train")
    ap.add_argument("--model", help="only this model id (default: the whole group)")
    ap.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    ap.add_argument("--steps", type=int, default=OptCfg.steps)
    ap.add_argument("--force", action="store_true", help="recompute results that already exist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    x, y, _ = load_clean(args.root / "example" / "clean")
    split = torch.load(args.out / "split.pt", weights_only=True)
    ia, ib = split["idx_a"], split["idx_b"]
    mean_net = build_mean_model(torch.load(args.out / "mean_model.pt", weights_only=True), args.device)
    cfg = OptCfg(steps=args.steps)

    sub = args.root / ("example/train/models" if args.group == "train" else "test/models")
    paths = sorted(sub.glob("*.pt"))
    if args.model:
        paths = [p for p in paths if p.stem == args.model]
    (args.out / "stage5" / args.group).mkdir(parents=True, exist_ok=True)

    for p in tqdm(paths, desc=f"stage5 {args.group}"):
        net = None
        for fam in args.families:
            dest = result_path(args.out, args.group, p.stem, fam)
            if dest.exists() and not args.force:
                continue
            net = net or load(p, args.device)
            t0 = time.time()
            res = run_one(net, mean_net, fam, x, y, ia, ib, cfg)
            torch.save(res, dest)  # saved right away, so an interrupted run resumes where it stopped
            top = res["scores"][1].argmax().item()
            tqdm.write(
                f"{p.stem} {fam:6s} {time.time() - t0:5.1f}s KL {res['kl_start']:.2f}->{res['kl']:.2f} "
                f"targeted top c{top:02d}={res['scores'][1, top]:.2f}"
            )


if __name__ == "__main__":
    main()
