"""Steps 2-4: turn the token losses into one membership score per test record.

2. One number per record: the t-statistic of (reference loss - BUMAS loss) over its tokens.
3. Common scale: how many standard deviations it lies above the known non-members of the same source_type.
4. Single posts and photos keep their own value. Records of a collection (writer, research group, podcast,
   photographer) get the mean value of all test records of their owner, because an owner's test records turned out
   to be members or non-members together. Owners with a known member go to the top.

Labels of test records are never used (there are none).

Run from the project root:  uv run python -m lab_3.make_submission
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lab_3.compute_losses import OUTPUT as LOSSES
from lab_3.compute_losses import read_records

SUBMISSION = Path("outputs/solution.csv")
COLLECTION_TYPES = {"writer", "research group", "podcast", "photographer"}
CLIP = 2.0  # limit of a single token's difference; chosen on known.jsonl, limits from 1 to 5 work similarly
OWN_RECORD_WEIGHT = 0.01  # only orders the records of one owner, which would otherwise be tied
KNOWN_MEMBER_BONUS = 10.0  # large enough to put these owners above all other records


def record_signal(reference: torch.Tensor, bumas: torch.Tensor) -> float:
    """How surely BUMAS-mini predicts the record better than the reference model does.

    mean / std * sqrt(n) of the per-token differences. Clipping stops single tokens from dominating.
    """
    gap = (reference - bumas).clamp(-CLIP, CLIP).numpy()
    return float(gap.mean() / (gap.std() + 1e-6) * np.sqrt(len(gap)))


def membership_scores(known: list[dict], test: list[dict], signals: dict[str, float]) -> list[float]:
    """Return one score for every test record; a higher score means a stronger suspicion of membership."""
    # Step 3: signals of different source types are not comparable, e.g. short captions and long transcripts.
    non_members = defaultdict(list)
    for record in known:
        if not record["member"]:
            non_members[record["source_type"]].append(signals[record["id"]])
    scale = {source_type: (np.mean(values), np.std(values)) for source_type, values in non_members.items()}

    def z_score(record: dict) -> float:
        mean, std = scale[record["source_type"]]
        return float((signals[record["id"]] - mean) / std)

    # Step 4: collect the evidence of every owner from all of its test records.
    owner_values = defaultdict(list)
    for record in test:
        owner_values[record["source_type"], record["source"]].append(z_score(record))
    owners_with_member = {(r["source_type"], r["source"]) for r in known if r["member"]}

    scores = []
    for record in test:
        if record["source_type"] not in COLLECTION_TYPES:
            scores.append(z_score(record))
            continue
        owner = record["source_type"], record["source"]
        score = float(np.mean(owner_values[owner])) + OWN_RECORD_WEIGHT * z_score(record)
        if owner in owners_with_member:
            score += KNOWN_MEMBER_BONUS
        scores.append(score)
    return scores


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
