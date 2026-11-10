"""Trigger parameterisations, all batched over a leading "group" dimension G.

A group is one independent trigger being optimised (one target class, or one (source, target) pair).
Every form maps images ``x`` of shape (G, B, 3, H, W) to triggered images of the same shape and
exposes two per-group scalars:

* ``penalty``: differentiable size term used in the loss,
* ``size``: the statistic that is recorded (patch: L1 of the mask in pixels, blend: RMS of delta,
  warp: mean absolute flow in pixels). Smaller means a more suspicious class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import FormCfg

Params = dict[str, torch.Tensor]


class Form:
    name = "base"

    def __init__(self, cfg: FormCfg, hw: int = 64):
        self.cfg = cfg
        self.hw = hw

    def init(self, g: int, device) -> Params:
        raise NotImplementedError

    def apply(self, p: Params, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def penalty(self, p: Params) -> torch.Tensor:
        raise NotImplementedError

    def size(self, p: Params) -> torch.Tensor:
        raise NotImplementedError


class PatchForm(Form):
    """x' = (1 - m) * x + m * pattern, m = sigmoid(a), penalty = mean(m)."""

    name = "patch"

    def init(self, g, device):
        hw = self.hw
        return {
            "a": torch.full((g, 1, hw, hw), float(self.cfg.a0), device=device),
            "q": torch.zeros((g, 3, hw, hw), device=device),
        }

    def apply(self, p, x):
        m = torch.sigmoid(p["a"])[:, None]
        pat = torch.sigmoid(p["q"])[:, None]
        return (1 - m) * x + m * pat

    def size(self, p):
        return torch.sigmoid(p["a"]).sum(dim=(1, 2, 3))

    def penalty(self, p):
        return self.size(p) / (self.hw * self.hw)


class BlendForm(Form):
    """x' = clip(x + delta). delta is full resolution (res=0) or bilinearly upsampled from res x res."""

    name = "blend"

    def init(self, g, device):
        r = self.cfg.res or self.hw
        return {"d": torch.zeros((g, 3, r, r), device=device)}

    def _delta(self, p):
        d = p["d"]
        if d.shape[-1] != self.hw:
            d = F.interpolate(d, size=(self.hw, self.hw), mode="bilinear", align_corners=False)
        return d

    def apply(self, p, x):
        return (x + self._delta(p)[:, None]).clamp_(0, 1)

    def size(self, p):
        return self._delta(p).pow(2).mean(dim=(1, 2, 3)).add(1e-12).sqrt()

    def penalty(self, p):
        return self.size(p)


class WarpForm(Form):
    """WaNet-like universal warp. The flow lives on a k x k grid (in pixels) and is upsampled."""

    name = "warp"

    def init(self, g, device):
        k = self.cfg.res or 4
        hw = self.hw
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, hw, device=device), torch.linspace(-1, 1, hw, device=device), indexing="ij"
        )
        self.base = torch.stack([xs, ys], dim=-1)[None]  # (1,H,W,2) in (x, y) order
        return {"f": torch.zeros((g, 2, k, k), device=device)}

    def _flow(self, p):
        return F.interpolate(p["f"], size=(self.hw, self.hw), mode="bilinear", align_corners=True)

    def apply(self, p, x):
        g, b = x.shape[:2]
        flow = self._flow(p).permute(0, 2, 3, 1) * (2.0 / (self.hw - 1))  # px -> normalised coords
        grid = (self.base + flow)[:, None].expand(g, b, -1, -1, -1).reshape(g * b, self.hw, self.hw, 2)
        out = F.grid_sample(
            x.reshape(g * b, *x.shape[2:]), grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out.view_as(x)

    def size(self, p):
        return self._flow(p).abs().mean(dim=(1, 2, 3))

    def penalty(self, p):
        f = p["f"]
        tv = (f[..., 1:, :] - f[..., :-1, :]).abs().mean(dim=(1, 2, 3)) + (
            f[..., :, 1:] - f[..., :, :-1]
        ).abs().mean(dim=(1, 2, 3))
        return (f.abs() + 1e-6).mean(dim=(1, 2, 3)) + self.cfg.tv_w * tv


FORMS = {"patch": PatchForm, "blend": BlendForm, "warp": WarpForm}


def make_form(name: str, cfg: FormCfg, hw: int = 64) -> Form:
    return FORMS[name](cfg, hw)
