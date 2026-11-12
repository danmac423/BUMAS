"""CLI: `lab_2 scan` (trigger inversion, cached) and `lab_2 score` (CSV, optional AUC)."""

import argparse
from pathlib import Path

from lab_2 import scan, scoring


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab_2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="invert triggers for every model, cache (l1, asr)")
    p.add_argument("--models", type=Path, default=Path("test/models"))
    p.add_argument("--data", type=Path, default=Path("example"), help="folder with model.py and clean/")
    p.add_argument("--cache", type=Path, default=Path("cache/test"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--s1", type=int, default=150)
    p.add_argument("--s2", type=int, default=100)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--b1", type=int, default=8, help="images per class, warm-up")
    p.add_argument("--b2", type=int, default=32, help="images per class, focus")

    p = sub.add_parser("score", help="turn cached norms into a submission CSV")
    p.add_argument("--cache", type=Path, default=Path("cache/test"))
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--models", type=Path, default=None, help="validate ids against this models folder")
    p.add_argument("--labels", type=Path, default=None, help="labels.csv to report AUC (train smoke test)")

    p.add_argument("--method", choices=scoring.METHODS, default="nc")

    args = parser.parse_args()
    if args.cmd == "scan":
        scan.scan_all(args.models, args.data, args.cache, args.device, s1=args.s1, s2=args.s2, k=args.k, b1=args.b1, b2=args.b2)
    else:
        df = scoring.score_cache(args.cache, args.method)
        if args.models:
            scoring.check_csv(df, args.models)
        scoring.write_csv(df, args.out)
        if args.labels:
            print(f"AUC: {scoring.auc(df, args.labels):.3f}")


if __name__ == "__main__":
    main()
