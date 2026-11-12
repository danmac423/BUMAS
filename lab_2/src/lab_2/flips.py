"""Stage 3: the flip table shared by stages 4 and 5 (plan, section 3).

For a model and a trigger: apply the trigger to clean images, then build a 20 x 20 table whose entry [s, t] is
the fraction of images of true class s that the model classifies as t. The diagonal (s == t) is set to 0.
Subtracting the same table for the averaged model leaves only what the examined model does on its own.
Each target class t then gets two scores from column t:
    all       - mean over source classes (a plain backdoor: many classes flip to t)
    targeted  - max over source classes (targeted poisoning: one class flips to t)
"""

from collections.abc import Callable

import torch

from lab_2.config import N_CLASSES

Trigger = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def predict(net: torch.nn.Module, x: torch.Tensor, trigger: Trigger | None = None, batch: int = 400) -> torch.Tensor:
    """Predicted classes (N,) on CPU; the trigger (if any) is applied batch-wise on the model's device."""
    device = next(net.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(device)
        out.append(net(trigger(xb) if trigger is not None else xb).argmax(1).cpu())
    return torch.cat(out)


def flip_table_from_preds(preds: torch.Tensor, labels: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """(n_classes, n_classes) float table, rows = true class, cols = predicted class, diagonal 0."""
    counts = torch.zeros(n_classes, n_classes)
    counts.index_put_((labels, preds), torch.ones(len(labels)), accumulate=True)
    per_class = counts.sum(1, keepdim=True).clamp(min=1)
    table = counts / per_class
    table.fill_diagonal_(0)
    return table


def flip_table(
    net: torch.nn.Module, x: torch.Tensor, labels: torch.Tensor, trigger: Trigger | None = None
) -> torch.Tensor:
    return flip_table_from_preds(predict(net, x, trigger), labels)


def column_scores(table: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per target class t: 'all' = mean of column t over the n-1 other classes, 'targeted' = its max.

    Works on a raw or a differenced table. Differences can be negative, so the max is not clamped at 0.
    """
    n = table.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool)
    masked = table.masked_fill(~off_diag, float("-inf"))
    return {"all": table.sum(0) / (n - 1), "targeted": masked.max(0).values}


def diff_scores(table: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Scores of `table - reference` (examined model minus averaged model)."""
    return column_scores(table - reference)
