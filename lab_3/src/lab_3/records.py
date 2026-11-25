"""Paths and record loading shared by both stages of the attack."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent / "model"
LOSSES_PATH = ROOT / "artifacts" / "losses.pkl"
SUBMISSION_PATH = ROOT / "submission.csv"

Record = dict[str, Any]


def load_jsonl(path: Path) -> list[Record]:
    """Read a JSON Lines file.

    Args:
        path: File with one JSON object per line.

    Returns:
        The records in file order.
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
