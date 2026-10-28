"""Patch localization: feature squeezing + integrated gradients + causal denoising test.

Step 0: denoise x with a filter -> x~; c1, c2 = top-2 classes of x; margin m(x) = logit[c1] - logit[c2].
Step 1: global features m0, m0 - m(x~), argmax(x~) != c1.
Step 2: IG of the margin along x~ -> x, per-pixel map A, mean of A over every allowed box, NMS -> K candidates.
Step 3: effect(r) = m0 - m(x with r replaced by x~[r]), minus the median effect of random allowed boxes.
Step 4: greedy edge refinement of the best candidate maximizing effect(r) - lam * area(r).
Step 5 (features for the patch/no-patch decision) lives in `features` and `Decider`.
"""

from dataclasses import dataclass

import kornia.filters as KF
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .data import load_images

SIZE = 64
AREA_MIN, AREA_MAX = 328, 614  # 8% .. 15% of 64 * 64
K, N_RAND, IG_STEPS = 20, 16, 24
SHIFTS = (-3, -2, -1, 1, 2, 3)
NONE = (0, 0, 0, 0)
FEATURES = ["m0", "dm", "flip", "e1", "e_gap", "ig_score"]


def _median(x: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    return KF.median_blur(F.pad(x, (p, p, p, p), mode="reflect"), k)[..., p:-p, p:-p]


FILTERS = {
    "median3": lambda x: _median(x, 3),
    "median5": lambda x: _median(x, 5),
    "gauss07": lambda x: KF.gaussian_blur2d(x, (5, 5), (0.7, 0.7), border_type="reflect"),
    "bilateral": lambda x: KF.bilateral_blur(x, (5, 5), 0.1, (1.5, 1.5), border_type="reflect"),
}


def area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def is_allowed(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) boxes -> bool: inside the image, 8-15% of its area, aspect between 1:2 and 2:1."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    w, h = x1 - x0, y1 - y0
    a = w * h
    return (
        (x0 >= 0) & (y0 >= 0) & (x1 <= SIZE) & (y1 <= SIZE) & (w > 0)
        & (a >= AREA_MIN) & (a <= AREA_MAX) & (h <= 2 * w) & (w <= 2 * h)
    )  # fmt: skip


def allowed_sizes() -> torch.Tensor:
    """(n, 2) long tensor of every allowed (w, h)."""
    wh = torch.cartesian_prod(torch.arange(1, SIZE + 1), torch.arange(1, SIZE + 1))
    return wh[is_allowed(torch.cat([torch.zeros_like(wh), wh], dim=1))]


def all_boxes() -> torch.Tensor:
    """(N, 4) long tensor of every allowed box at every position."""
    out = []
    for w, h in allowed_sizes().tolist():
        xs, ys = torch.meshgrid(torch.arange(SIZE - w + 1), torch.arange(SIZE - h + 1), indexing="xy")
        x0, y0 = xs.reshape(-1), ys.reshape(-1)
        out.append(torch.stack([x0, y0, x0 + w, y0 + h], dim=1))
    return torch.cat(out)


def random_boxes(n_img: int, n: int, gen: torch.Generator) -> torch.Tensor:
    """(n_img, n, 4) allowed boxes: uniform over allowed sizes, then uniform position."""
    sizes = allowed_sizes()
    wh = sizes[torch.randint(len(sizes), (n_img, n), generator=gen)]
    x0 = (torch.rand(n_img, n, generator=gen) * (SIZE + 1 - wh[..., 0])).long()
    y0 = (torch.rand(n_img, n, generator=gen) * (SIZE + 1 - wh[..., 1])).long()
    return torch.stack([x0, y0, x0 + wh[..., 0], y0 + wh[..., 1]], dim=-1)


def box_masks(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) boxes -> (..., 1, H, W) bool masks."""
    idx = torch.arange(SIZE, device=boxes.device)
    rows = (idx >= boxes[..., 1, None]) & (idx < boxes[..., 3, None])
    cols = (idx >= boxes[..., 0, None]) & (idx < boxes[..., 2, None])
    return (rows[..., :, None] & cols[..., None, :]).unsqueeze(-3)


def margin(model, x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """logit[c1] - logit[c2] per row of x; c1, c2 are (N,) class indices."""
    out = []
    for xs, a, b in zip(x.split(chunk), c1.split(chunk), c2.split(chunk)):
        lg = model(xs)
        out.append(lg.gather(1, a[:, None])[:, 0] - lg.gather(1, b[:, None])[:, 0])
    return torch.cat(out)


@torch.no_grad()
def effect(model, x, xt, c1, c2, m0, boxes: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Margin drop after replacing each box by its denoised pixels. x, xt (B, 3, H, W), boxes (B, M, 4) -> (B, M)."""
    b, m, _ = boxes.shape
    xs = torch.where(box_masks(boxes), xt[:, None], x[:, None]).flatten(0, 1)
    marg = margin(model, xs, c1.repeat_interleave(m), c2.repeat_interleave(m), chunk)
    return m0[:, None] - marg.view(b, m)


def ig_map(model, x, xt, c1, c2, steps: int = IG_STEPS, chunk: int = 128) -> torch.Tensor:
    """Integrated gradients of the margin from xt to x (midpoint rule), summed over channels -> (B, H, W)."""
    alphas = (torch.arange(steps, device=x.device) + 0.5) / steps
    path = (xt[:, None] + alphas[None, :, None, None, None] * (x - xt)[:, None]).flatten(0, 1)
    c1r, c2r = c1.repeat_interleave(steps), c2.repeat_interleave(steps)
    grads = []
    with torch.enable_grad():
        for p, a, b in zip(path.split(chunk), c1r.split(chunk), c2r.split(chunk)):
            p = p.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(margin(model, p, a, b).sum(), p)
            grads.append(g)
    grad = torch.cat(grads).view(*x.shape[:1], steps, *x.shape[1:]).mean(1)
    return ((x - xt) * grad).sum(1)


def box_means(a: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Mean of each map a (B, H, W) inside each box (N, 4), via an integral image -> (B, N)."""
    s = F.pad(a.cumsum(1).cumsum(2), (1, 0, 1, 0)).flatten(1)
    x0, y0, x1, y1 = boxes.unbind(-1)
    w1 = SIZE + 1
    tot = s[:, y1 * w1 + x1] - s[:, y0 * w1 + x1] - s[:, y1 * w1 + x0] + s[:, y0 * w1 + x0]
    return tot / area(boxes)


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Broadcasting IoU of (..., 4) boxes."""
    w = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    h = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = w * h
    return inter / (area(a) + area(b) - inter)


def nms(boxes: torch.Tensor, scores: torch.Tensor, k: int, thr: float = 0.5, pre: int = 2048) -> torch.Tensor:
    """Greedy NMS of the shared boxes (N, 4) under per-image scores (B, N) -> indices (B, k) of the kept boxes.

    Only the `pre` best boxes of each image take part, which keeps memory at (B, pre) instead of (B, N).
    """
    s, top = scores.topk(min(pre, scores.shape[1]), dim=1)
    b = boxes[top]  # (B, pre, 4)
    keep = []
    for _ in range(k):
        i = s.argmax(1)
        keep.append(top.gather(1, i[:, None])[:, 0])
        picked = b.gather(1, i[:, None, None].expand(-1, 1, 4))
        s = s.masked_fill(iou(picked, b) > thr, -torch.inf)
    return torch.stack(keep, 1)


@dataclass
class Analysis:
    """Steps 0-3 for a batch; tensors on the model's device, one row per image."""

    x: torch.Tensor
    xt: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    m0: torch.Tensor
    dm: torch.Tensor  # m0 - margin(xt)
    flip: torch.Tensor  # argmax(xt) != c1
    cand: torch.Tensor  # (B, K, 4) candidates, sorted by e descending
    e: torch.Tensor  # (B, K) effect minus random-box background
    score: torch.Tensor  # (B, K) IG box mean


def analyze(model, x: torch.Tensor, filt: str, cands: torch.Tensor, gen: torch.Generator) -> Analysis:
    """Steps 0-3. x (B, 3, H, W) on the model's device, cands = all_boxes() on the same device."""
    xt = FILTERS[filt](x).clamp(0, 1)
    with torch.no_grad():
        lg, lgt = model(x), model(xt)
    c1, c2 = lg.topk(2, dim=1).indices.unbind(1)
    m0 = lg.gather(1, c1[:, None])[:, 0] - lg.gather(1, c2[:, None])[:, 0]
    mt = lgt.gather(1, c1[:, None])[:, 0] - lgt.gather(1, c2[:, None])[:, 0]

    scores = box_means(ig_map(model, x, xt, c1, c2), cands)
    idx = nms(cands, scores, K)
    cand, score = cands[idx], scores.gather(1, idx)

    rnd = random_boxes(len(x), N_RAND, gen).to(x.device)
    eff = effect(model, x, xt, c1, c2, m0, torch.cat([cand, rnd], 1))
    e = eff[:, :K] - eff[:, K:].median(1).values[:, None]
    order = e.argsort(1, descending=True)
    return Analysis(
        x, xt, c1, c2, m0, m0 - mt, lgt.argmax(1) != c1,
        cand.gather(1, order[..., None].expand(-1, -1, 4)), e.gather(1, order), score.gather(1, order),
    )  # fmt: skip


def neighbours(r: torch.Tensor) -> torch.Tensor:
    """(B, 4) -> (B, 4 * len(SHIFTS), 4): r with one edge moved by d in SHIFTS."""
    delta = torch.zeros(4 * len(SHIFTS), 4, dtype=r.dtype, device=r.device)
    for edge in range(4):
        for j, d in enumerate(SHIFTS):
            delta[edge * len(SHIFTS) + j, edge] = d
    return r[:, None] + delta


def refine(model, a: Analysis, lam: float, max_iter: int = 30) -> torch.Tensor:
    """Step 4: hill-climb the best candidate on J(r) = effect(r) - lam * area(r). Returns (B, 4)."""
    r = a.cand[:, 0].clone()
    j = effect(model, a.x, a.xt, a.c1, a.c2, a.m0, r[:, None])[:, 0] - lam * area(r)
    active = torch.arange(len(r), device=r.device)
    for _ in range(max_iter):
        if not len(active):
            break
        nb = neighbours(r[active])
        ok = is_allowed(nb)
        safe = torch.where(ok[..., None], nb, r[active][:, None])
        jn = effect(model, a.x[active], a.xt[active], a.c1[active], a.c2[active], a.m0[active], safe)
        jn = (jn - lam * area(nb)).masked_fill(~ok, -torch.inf)
        best, k = jn.max(1)
        better = best > j[active] + 1e-6
        moved = active[better]
        r[moved] = nb[better, k[better]]
        j[moved] = best[better]
        active = moved
    return r


def features(a: Analysis) -> np.ndarray:
    """(B, len(FEATURES)) decision features."""
    cols = [a.m0, a.dm, a.flip.float(), a.e[:, 0], a.e[:, 0] - a.e[:, 1], a.score[:, 0]]
    return torch.stack(cols, 1).float().cpu().numpy()


def box_iou_np(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Row-wise IoU of integer boxes (N, 4); 0 where either box is empty."""
    t, p = torch.tensor(truth, dtype=torch.long), torch.tensor(pred, dtype=torch.long)
    return iou(t, p).nan_to_num(0.0).numpy()


@dataclass
class Decider:
    """Logistic regression P(patch | features) plus a validation table p-bin -> E[IoU | patch]."""

    clf: Pipeline
    edges: np.ndarray
    eiou: np.ndarray

    @classmethod
    def fit(cls, f_fit, y_fit, f_val, y_val, iou_val, n_bins: int = 10) -> "Decider":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(f_fit, y_fit)
        p = clf.predict_proba(f_val)[:, 1]
        edges = np.linspace(0, 1, n_bins + 1)
        b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        fallback = iou_val[y_val].mean()
        eiou = np.array(
            [iou_val[(b == i) & y_val].mean() if ((b == i) & y_val).sum() >= 5 else fallback for i in range(n_bins)]
        )
        return cls(clf, edges, eiou)

    def proba(self, f: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(f)[:, 1]

    def expected_iou(self, p: np.ndarray) -> np.ndarray:
        return self.eiou[np.clip(np.digitize(p, self.edges) - 1, 0, len(self.eiou) - 1)]

    def decide(self, f: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Keep the box where p * E[IoU] > 1 - p, else (0, 0, 0, 0)."""
        p = self.proba(f)
        keep = p * self.expected_iou(p) > 1 - p
        return np.where(keep[:, None], boxes, 0)


def run(model, split: str, ids, filt: str, lams, batch: int = 16, seed: int = 0) -> dict:
    """Steps 0-4 over images `ids` of `split`.

    Returns {"feats": (N, F), "best": (N, 4) best candidate, lam: (N, 4) refined box for each lam in lams}.
    """
    device = next(model.parameters()).device
    cands = all_boxes().to(device)
    ids = list(ids)
    out = {"feats": [], "best": [], **{lam: [] for lam in lams}}
    for s in tqdm(range(0, len(ids), batch), desc=f"{split}/{filt}"):
        x = load_images(split, ids[s : s + batch]).to(device)
        a = analyze(model, x, filt, cands, torch.Generator().manual_seed(seed + s))
        out["feats"].append(features(a))
        out["best"].append(a.cand[:, 0].cpu().numpy())
        for lam in lams:
            out[lam].append(refine(model, a, lam).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
