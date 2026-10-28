import torch
import torch.nn.functional as F


def make_occluded_batch(
    img: torch.Tensor, win: int = 20, stride: int = 2, fill: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """One copy of `img` per window position, with that win x win square set to `fill`.

    img: (3, H, W). Returns (batch, coords): batch (N, 3, H, W) on img's device,
    coords (N, 2) long with the (x, y) of each window's top-left corner, row-major over y then x.
    """
    _, h, w = img.shape
    ys = torch.arange(0, h - win + 1, stride, device=img.device)
    xs = torch.arange(0, w - win + 1, stride, device=img.device)
    coords = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)

    rows = torch.arange(h, device=img.device)
    cols = torch.arange(w, device=img.device)
    x0, y0 = coords[:, 0, None], coords[:, 1, None]
    row_in = (rows >= y0) & (rows < y0 + win)  # (N, H)
    col_in = (cols >= x0) & (cols < x0 + win)  # (N, W)
    mask = (row_in[:, :, None] & col_in[:, None, :])[:, None]  # (N, 1, H, W)

    batch = torch.where(mask, torch.full_like(img, fill), img)
    return batch, coords


@torch.no_grad()
def compute_deltas(
    model: torch.nn.Module,
    img: torch.Tensor,
    win: int = 20,
    stride: int = 2,
    fill: float = 0.5,
    chunk: int = 512,
) -> torch.Tensor:
    """Logit drop of the originally predicted class for every window position.

    img: (3, H, W) on the model's device. Returns (n_y, n_x) with
    delta[j, i] = Z0 - Z_masked for the window at x = i * stride, y = j * stride.
    """
    model.eval()
    logits0 = model(img[None])[0]
    y_pred = logits0.argmax()
    z0 = logits0[y_pred]

    batch, coords = make_occluded_batch(img, win, stride, fill)
    z_masked = torch.cat([model(b)[:, y_pred] for b in batch.split(chunk)])

    n_x = int(coords[:, 0].unique().numel())
    return (z0 - z_masked).reshape(-1, n_x)


def heatmap(delta: torch.Tensor, win: int = 20, stride: int = 2) -> torch.Tensor:
    """Per-pixel mean of the positive window drops covering that pixel.

    delta: (n_y, n_x) from compute_deltas. Returns (H, W) with H = (n_y - 1) * stride + win.
    Negative drops count as 0 but still count as a covering window (they are averaged in).
    """
    n_y, n_x = delta.shape
    size = ((n_y - 1) * stride + win, (n_x - 1) * stride + win)
    d = delta.clamp(min=0).reshape(1, 1, -1).expand(1, win * win, -1)
    total = F.fold(d, size, kernel_size=win, stride=stride)
    cover = F.fold(torch.ones_like(d), size, kernel_size=win, stride=stride)
    return (total / cover)[0, 0]


def box(heat: torch.Tensor, mask_ratio: float = 0.4) -> tuple[int, int, int, int]:
    """Bounding box (x_min, y_min, x_max, y_max), max exclusive, of pixels >= mask_ratio * heat.max().

    Returns (0, 0, 0, 0) when the map has no positive pixel.
    """
    peak = heat.max()
    if peak <= 0:
        return (0, 0, 0, 0)
    mask = heat >= mask_ratio * peak
    cols = mask.any(dim=0).nonzero()
    rows = mask.any(dim=1).nonzero()
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
