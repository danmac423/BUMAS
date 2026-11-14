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
    """Predicts classes, optionally after applying a trigger.

    Args:
        net: Classifier in eval mode.
        x: Images (N, 3, 64, 64) in [0, 1], on any device.
        trigger: Function mapping a batch of images to triggered images; None means clean images.
        batch: Number of images per forward pass.

    Returns:
        Predicted class indices (N,) on CPU. The trigger runs batch-wise on the model's device.
    """
    device = next(net.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(device)
        out.append(net(trigger(xb) if trigger is not None else xb).argmax(1).cpu())
    return torch.cat(out)


def flip_table_from_preds(preds: torch.Tensor, labels: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Builds the flip table from predictions.

    Args:
        preds: Predicted classes (N,).
        labels: True classes (N,).
        n_classes: Number of classes.

    Returns:
        Float table (n_classes, n_classes); entry [s, t] is the fraction of images of true class s predicted
        as t. The diagonal is set to 0, because only flips to other classes matter.
    """
    counts = torch.zeros(n_classes, n_classes)
    # accumulate=True makes repeated (label, pred) pairs add up instead of overwriting each other
    counts.index_put_((labels, preds), torch.ones(len(labels)), accumulate=True)
    per_class = counts.sum(1, keepdim=True).clamp(min=1)  # clamp avoids 0/0 for classes with no images
    table = counts / per_class
    table.fill_diagonal_(0)
    return table


def flip_table(
    net: torch.nn.Module, x: torch.Tensor, labels: torch.Tensor, trigger: Trigger | None = None
) -> torch.Tensor:
    """Computes the flip table of a model under a trigger.

    Args:
        net: Classifier in eval mode.
        x: Images (N, 3, 64, 64) in [0, 1].
        labels: True classes (N,).
        trigger: Function applied to the images before classification; None means clean images.

    Returns:
        Float table (20, 20), see `flip_table_from_preds`.
    """
    return flip_table_from_preds(predict(net, x, trigger), labels)


def column_scores(table: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reduces a flip table to two scores per target class.

    Works on a raw or a differenced table. Differences can be negative, so the max is not clamped at 0.

    Args:
        table: Float table (n, n), rows = true class, columns = predicted class.

    Returns:
        Dict with two tensors of shape (n,), indexed by target class t:
        "all" is the mean of column t over the n-1 other classes (a plain backdoor), "targeted" is the
        maximum of column t (a backdoor that works for a single source class).
    """
    n = table.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool)
    # the diagonal is excluded from the max with -inf (it is 0 in a raw table but meaningless after a diff)
    masked = table.masked_fill(~off_diag, float("-inf"))
    return {"all": table.sum(0) / (n - 1), "targeted": masked.max(0).values}


def diff_scores(table: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Scores of the difference between the examined model and the averaged model.

    Args:
        table: Flip table of the examined model.
        reference: Flip table of the averaged model for the same trigger.

    Returns:
        The dict returned by `column_scores(table - reference)`.
    """
    return column_scores(table - reference)
