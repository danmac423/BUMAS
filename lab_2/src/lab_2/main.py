"""CLI: ``lab_2 scan`` caches raw per-model statistics, ``lab_2 score`` calibrates and writes the CSV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from . import store
from .config import Config
from .data import list_models, load_data, load_net


def cmd_scan(args) -> None:
    cfg = Config.from_json(args.config)
    if args.device:
        cfg.device = args.device
    torch.backends.cudnn.benchmark = True
    paths = list_models(args.models)
    if args.ids:
        wanted = set(args.ids)
        paths = [p for p in paths if p.stem in wanted]
    if args.limit:
        paths = paths[: args.limit]
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    data = load_data(cfg.clean_dir, cfg.device, cfg.held_frac, cfg.seed)

    from .scan import scan_model

    t_start = time.perf_counter()
    done = 0
    for p in tqdm(paths, desc="scan", unit="model"):
        if store.cache_path(out, p.stem).exists() and not args.force:
            continue
        net = load_net(p, cfg.model_py, cfg.device)
        res = scan_model(net, data, cfg, p.stem)
        store.save(out, p.stem, res)
        done += 1
        if args.verbose:
            tqdm.write(f"{p.stem}: {float(res['t_total']):.1f}s " + " ".join(
                f"{k}={float(v):.1f}" for k, v in res.items() if k.startswith("t_") and k != "t_total"))
        del net
    el = time.perf_counter() - t_start
    if done:
        print(f"scanned {done} models in {el:.0f}s ({el / done:.1f}s/model; 200 models -> {el / done * 200 / 3600:.2f} h)")


def cmd_score(args) -> None:
    from .scoring import run_score

    cfg = Config.from_json(args.config)
    run_score(cfg, args.cache, args.reference or args.cache, args.out, args.models, args.labels)


def cmd_plant(args) -> None:
    from .synth import plant

    cfg = Config.from_json(args.config)
    row = plant(args.base, args.out, args.spec, cfg, epochs=args.epochs, lr=args.lr, seed=args.seed)
    print(row)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lab_2", description="Backdoor target-class detector (P2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan models and cache raw statistics")
    s.add_argument("--models", required=True, help="directory with *.pt models")
    s.add_argument("--cache", required=True, help="cache directory for per-model .npz files")
    s.add_argument("--config", help="JSON file overriding Config fields")
    s.add_argument("--device", help="override device (default cuda)")
    s.add_argument("--limit", type=int, help="only the first N models")
    s.add_argument("--ids", nargs="*", help="only these model ids")
    s.add_argument("--force", action="store_true", help="rescan even if cached")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_scan)

    c = sub.add_parser("score", help="calibrate cached statistics and write submission.csv")
    c.add_argument("--cache", required=True, help="cache of the models to score")
    c.add_argument("--reference", help="cache used for population calibration (default: --cache)")
    c.add_argument("--out", default="submission.csv")
    c.add_argument("--models", help="models dir; if given, ids must match its *.pt files exactly")
    c.add_argument("--labels", help="train/labels.csv: print AUC and per-model ranks of the true targets")
    c.add_argument("--config", help="JSON file overriding Config fields")
    c.set_defaults(fn=cmd_score)

    p = sub.add_parser("plant", help="(stretch) plant known triggers into a model to measure recall")
    p.add_argument("--base", required=True, help="clean model .pt to fine-tune")
    p.add_argument("--out", required=True, help="output dir (models/ + labels.csv, same layout as example/train)")
    p.add_argument("--spec", nargs="+", required=True, help="kind:target[:source], kind in patch dots blend sig wanet")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", help="JSON file overriding Config fields")
    p.set_defaults(fn=cmd_plant)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
