#!/usr/bin/env python3
"""
P2 Backdoors - method 1: cheap per-(model, class) statistics + population calibration.

Layout (relative to the lab_2 directory):
    example/                 <- --root (default)
        model.py, classes.txt, clean/images/, clean/labels.csv,
        train/models/*.pt, train/labels.csv
    test/models/*.pt         <- --test-dir (default)

Usage:
    # 1) compute statistics (cached per model in cache_m1/, can be interrupted and resumed)
    uv run python -m lab_2.main extract                       # with "uap" + per-source "uap_src" (recommended on GPU)
    uv run python -m lab_2.main extract --no-uap              # gradient-free statistics only (fast, e.g. on M1)
    uv run python -m lab_2.main extract --no-uap-src           # skip the (slower) per-source targeted-backdoor pass

    # 2) calibration, AUC on training models, submission file
    uv run python -m lab_2.main score --out submission_m1.csv
    uv run python -m lab_2.main score --stats attract_maxsrc,uap_margin,uap_maxsrc_margin   # choose statistics to combine
"""

import argparse

from lab_2.detection import cmd_extract
from lab_2.scoring import cmd_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--root", default="example", help="dir with model.py, classes.txt, clean/, train/")
    e.add_argument("--test-dir", default="test", help="dir with models/*.pt of the test set")
    e.add_argument("--cache", default="cache_m1")
    e.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    e.add_argument("--no-uap", action="store_true", help="skip gradient-based statistics (faster)")
    e.add_argument("--eps", type=float, default=6.0, help="L_inf perturbation budget, in units of /255")
    e.add_argument("--steps", type=int, default=20, help="PGD steps per class")
    e.add_argument(
        "--no-uap-src",
        action="store_true",
        help="skip per-(source,target) UAP statistics (targeted-backdoor signal, ~2x slower extract)",
    )
    e.add_argument(
        "--eps-src",
        type=float,
        default=4.0,
        help="L_inf budget for per-source UAP, in units of /255 (smaller than --eps: fit on half the images per class)",
    )
    e.add_argument("--force", action="store_true", help="recompute even if the cache exists")

    s = sub.add_parser("score")
    s.add_argument("--root", default="example", help="dir with model.py, classes.txt, clean/, train/")
    s.add_argument("--cache", default="cache_m1")
    s.add_argument("--stats", default="", help="comma-separated statistics for COMBO (default: all)")
    s.add_argument("--out", default="submission_m1.csv")

    args = ap.parse_args()
    {"extract": cmd_extract, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
