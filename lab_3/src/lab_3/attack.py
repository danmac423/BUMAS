"""Stage 2 (CPU): features, grouped cross-validation, ablation, final logistic regression and the submission."""

import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lab_3.features import RECORD_FEATURES, SOURCE_TYPES, build_features, delta
from lab_3.metrics import tpr_przy_5_fpr
from lab_3.records import DATA_DIR, LOSSES_PATH, SUBMISSION_PATH, load_jsonl
from lab_3.submission import write_submission

N_FOLDS = 5
TYPE_COLUMNS = [f"type: {t}" for t in SOURCE_TYPES]
BASELINE = ["z_delta", "owner_mean_z_delta", *TYPE_COLUMNS, "log_n_tokens"]
RECORD_ONLY = [*RECORD_FEATURES, *TYPE_COLUMNS, "log_n_tokens"]
FINAL = [*RECORD_FEATURES, "owner_std_z_sum_delta", *TYPE_COLUMNS, "log_n_tokens"]


def stack(columns: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Feature matrix with the chosen columns."""
    return np.column_stack([columns[n] for n in names])


def fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """L2-regularised logistic regression with the default ``C``."""
    return LogisticRegression(max_iter=5000).fit(x, y)


def tpr(scores: np.ndarray, y: np.ndarray) -> float:
    """TPR at 5% FPR for NumPy inputs."""
    return tpr_przy_5_fpr(scores.tolist(), y.tolist())


def out_of_fold(x: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Logit of every record predicted by a model that did not train on its owner."""
    scores = np.zeros(len(y))
    for train, valid in folds:
        scores[valid] = fit(x[train], y[train]).decision_function(x[valid])
    return scores


def report(name: str, scores: np.ndarray, y: np.ndarray, folds) -> float:
    """Print one ablation row and return the mean TPR over folds."""
    per_fold = [tpr(scores[valid], y[valid]) for _, valid in folds]
    mean = float(np.mean(per_fold))
    folds_text = " ".join(f"{v:.3f}" for v in per_fold)
    print(f"{name:<24} {folds_text}  mean {mean:.3f}  pooled {tpr(scores, y):.3f}  AUC {roc_auc_score(y, scores):.3f}")
    return mean


def main() -> None:
    """Validate the attack on the known records and write the submission."""
    if not LOSSES_PATH.exists():
        raise SystemExit(f"Missing {LOSSES_PATH}; run `python -m lab_3.compute_losses` first.")
    with LOSSES_PATH.open("rb") as f:
        losses = pickle.load(f)
    known = load_jsonl(DATA_DIR / "known.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    records = known + test
    n = len(known)

    columns = build_features(records, losses)
    known_columns = {k: v[:n] for k, v in columns.items()}
    y = np.array([int(r["member"]) for r in known])
    groups = np.array([r["source"] for r in known])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(y, y, groups))

    print(f"TPR@5%FPR, {N_FOLDS}-fold GroupKFold by source (folds, mean, pooled out-of-fold, AUC)")
    single = {
        "a) LOSS": np.array([-losses[r["id"]]["bumas_mean"] for r in known]),
        "b) raw delta": delta(known, losses),
        "c) z_delta": known_columns["z_delta"],
    }
    for name, scores in single.items():
        report(name, scores, y, folds)

    models = {
        "d) z_delta baseline": BASELINE,
        "e) record features": RECORD_ONLY,
        "f) final": FINAL,
    }
    oof = {}
    for name, names in models.items():
        oof[name] = out_of_fold(stack(known_columns, names), y, folds)
        report(name, oof[name], y, folds)
    chosen, names = "f) final", FINAL

    types = np.array([r["source_type"] for r in known])
    print("\nOut-of-fold TPR@5%FPR per source_type (threshold set within the type):")
    for t in SOURCE_TYPES:
        mask = types == t
        if len(np.unique(y[mask])) == 2:
            print(f"  {t:<16} {tpr(oof[chosen][mask], y[mask]):.3f}  (n={mask.sum()}, members={y[mask].sum()})")

    model = fit(stack(known_columns, names), y)
    print("\nWeights of the final model (log-odds):")
    for name, weight in zip(names, model.coef_[0], strict=True):
        print(f"  {name:<22} {weight:+.3f}")
    print(f"  {'intercept':<22} {model.intercept_[0]:+.3f}")

    scores = model.decision_function(stack(columns, names)[n:])
    test_ids = [r["id"] for r in test]
    write_submission(SUBMISSION_PATH, test_ids, scores.tolist(), test_ids)
    print(f"\nSaved {len(test_ids)} rows to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
