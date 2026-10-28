import pandas as pd

NONE = (0, 0, 0, 0)
BOX_COLS = ["x_min", "y_min", "x_max", "y_max"]


def punkty(prawda, odpowiedz):
    if prawda == NONE:
        return 1.0 if odpowiedz == NONE else 0.0
    if odpowiedz == NONE:
        return 0.0
    w = max(0, min(prawda[2], odpowiedz[2]) - max(prawda[0], odpowiedz[0]))
    h = max(0, min(prawda[3], odpowiedz[3]) - max(prawda[1], odpowiedz[1]))
    wspolna = w * h
    pole = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return wspolna / (pole(prawda) + pole(odpowiedz) - wspolna)


def score_rows(labels_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.Series:
    """Per-image score, indexed like labels_df. pred_df must contain every id of labels_df."""
    pred = pred_df.set_index("id")[BOX_COLS]
    truth = labels_df.set_index("id")[BOX_COLS]
    missing = truth.index.difference(pred.index)
    assert len(missing) == 0, f"{len(missing)} ids missing in predictions"
    pred = pred.loc[truth.index]
    scores = [
        punkty(tuple(int(v) for v in t), tuple(int(v) for v in p)) for t, p in zip(truth.to_numpy(), pred.to_numpy())
    ]
    return pd.Series(scores, index=truth.index, name="score")


def evaluate(labels_df: pd.DataFrame, pred_df: pd.DataFrame, by: list[str] | None = None) -> dict:
    """Mean score plus breakdowns. `labels_df` should come from data.load_labels (has meta columns)."""
    s = score_rows(labels_df, pred_df)
    meta = labels_df.set_index("id")
    out = {"mean": float(s.mean())}
    for col in by or ["kind", "attack", "eps_bin", "jpeg_bin", "n_decoys"]:
        if col in meta.columns:
            out[col] = s.groupby(meta[col], observed=True).mean().round(4).to_dict()
    return out
