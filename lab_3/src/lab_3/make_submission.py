"""Steps 2-3: turn the token losses into one membership score per test record.

2. One number per record: the t-statistic of (reference loss - BUMAS loss) over its tokens.
3. A logistic regression for every source_type, fitted on known.jsonl. Its features are the record's own number
   and, for a collection (writer, research group, podcast, photographer), the mean number of all test records of
   the owner. The score is the log-odds of membership, so the records of all types can be ranked together.

Labels of test records are never used (there are none).

Run from the project root:  uv run python -m lab_3.make_submission
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from lab_3.compute_losses import OUTPUT as LOSSES
from lab_3.compute_losses import read_records

SUBMISSION = Path("outputs/solution.csv")
COLLECTION_TYPES = {"writer", "research group", "podcast", "photographer"}
CLIP = 2.0  # limit of a single token's difference; chosen on known.jsonl, limits from 1 to 5 work similarly


def record_signal(reference: torch.Tensor, bumas: torch.Tensor) -> float:
    """How surely BUMAS-mini predicts the record better than the reference model does.

    mean / std * sqrt(n) of the per-token differences. Clipping stops single tokens from dominating.
    """
    gap = (reference - bumas).clamp(-CLIP, CLIP).numpy()
    return float(gap.mean() / (gap.std() + 1e-6) * np.sqrt(len(gap)))


def membership_scores(known: list[dict], test: list[dict], signals: dict[str, float]) -> list[float]:
    """Return one score for every test record; a higher score means a stronger suspicion of membership.

    Every owner of a known collection record must also have test records.
    """
    owner_signals = defaultdict(list)
    for record in test:
        owner_signals[record["source_type"], record["source"]].append(signals[record["id"]])

    def features(record: dict) -> list[float]:
        values = [signals[record["id"]]]
        if record["source_type"] in COLLECTION_TYPES:
            values.append(float(np.mean(owner_signals[record["source_type"], record["source"]])))
        return values

    # A separate model for every type: their signals are on different scales, e.g. short captions and long transcripts.
    models = {}
    for source_type in sorted({record["source_type"] for record in known}):
        records = [record for record in known if record["source_type"] == source_type]
        # C=inf: no regularisation. The strict tolerance makes sure that the fit has converged.
        model = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)
        models[source_type] = model.fit([features(r) for r in records], [r["member"] for r in records])

    return [float(models[r["source_type"]].decision_function([features(r)])[0]) for r in test]


def main() -> None:
    """Write the submission file."""
    known, test = read_records("known.jsonl"), read_records("test.jsonl")
    losses = torch.load(LOSSES)
    signals = {
        record["id"]: record_signal(losses[record["id"]]["reference"], losses[record["id"]]["bumas"])
        for record in known + test
    }
    scores = membership_scores(known, test, signals)

    with open(SUBMISSION, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "score"])
        writer.writerows((record["id"], f"{score:.12g}") for record, score in zip(test, scores, strict=True))
    print(f"wrote {SUBMISSION} ({len(scores)} records)")


if __name__ == "__main__":
    main()
