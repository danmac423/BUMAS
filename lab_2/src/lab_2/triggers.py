"""Triggers for stages 4 and 5 of the plan.

Every trigger maps a batch of images (N, 3, H, W) in [0, 1] to a batch of the same shape in [0, 1].

* Stage 4 - a fixed bank of hand-made triggers from the literature (`build_bank`): BadNets patches,
  Blended noise, SIG stripes and WaNet warps. Nothing is learned, only forward passes are needed.
* Stage 5 - four learnable families (`PatchTrigger`, `BlendTrigger`, `WarpTrigger`, `StripeTrigger`),
  each an `nn.Module` whose parameters are optimised while the model weights stay frozen. They share the
  `LearnableTrigger` interface: `forward(x)` applies the trigger, `penalty()` is an extra loss term
  (non-zero only for the patch, where it punishes a large mask) and `reset(seed)` re-initialises it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_2.config import BANK, BLEND, IMG, PATCH, STRIPE, WARP, BankCfg


# ----------------------------------------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------------------------------------
def identity_grid(h: int, w: int, device=None) -> torch.Tensor:
    """(1, H, W, 2) sampling grid of the identity warp, in grid_sample's [-1, 1] coordinates."""
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Shift pixels by `flow` (N or 1, H, W, 2) given in grid_sample units; borders are replicated."""
    grid = identity_grid(x.shape[2], x.shape[3], x.device) + flow
    grid = grid.expand(x.shape[0], -1, -1, -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def upsample_field(field: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Stretch a coarse (1, C, k, k) field to (1, C, H, W) with smooth bilinear interpolation."""
    return F.interpolate(field, size=(h, w), mode="bilinear", align_corners=True)


# ----------------------------------------------------------------------------------------------------------------
# stage 5: learnable families
# ----------------------------------------------------------------------------------------------------------------
class LearnableTrigger(nn.Module):
    """Base class: subclasses define parameters, `forward` and optionally `penalty`."""

    name = "base"

    def __init__(self, size: int = IMG):
        super().__init__()
        self.size = size

    def penalty(self) -> torch.Tensor:
        return torch.zeros(())

    def reset(self, seed: int = 0) -> None:
        raise NotImplementedError


class PatchTrigger(LearnableTrigger):
    """x' = (1 - m) * x + m * p with a learned soft mask m and a learned pattern p (BadNets-like).

    Pattern, position and shape all live in the (H, W) maps, so they are learned jointly. `penalty()` is the
    mean mask value (area fraction); the optimiser weighs it against the KL objective.
    """

    name = "patch"

    def __init__(self, size: int = IMG, init_mask_logit: float = PATCH.init_mask_logit):
        super().__init__(size)
        self.init_mask_logit = init_mask_logit
        self.mask_logit = nn.Parameter(torch.empty(1, 1, size, size))
        self.pattern_raw = nn.Parameter(torch.empty(1, 3, size, size))
        self.reset(0)

    @torch.no_grad()
    def reset(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        # noise in the mask breaks the symmetry, so the gradient can pick a location
        self.mask_logit.copy_(self.init_mask_logit + 0.5 * torch.randn(self.mask_logit.shape, generator=g))
        self.pattern_raw.copy_(torch.randn(self.pattern_raw.shape, generator=g))

    def mask(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logit)

    def pattern(self) -> torch.Tensor:
        return torch.sigmoid(self.pattern_raw)

    def penalty(self) -> torch.Tensor:
        return self.mask().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask()
        return (1 - m) * x + m * self.pattern()


class BlendTrigger(LearnableTrigger):
    """x' = (1 - alpha) * x + alpha * p with a full-image pattern p and a fixed blend strength alpha."""

    name = "blend"

    def __init__(self, size: int = IMG, alpha: float = BLEND.alpha):
        super().__init__(size)
        self.alpha = alpha
        self.pattern_raw = nn.Parameter(torch.empty(1, 3, size, size))
        self.reset(0)

    @torch.no_grad()
    def reset(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.pattern_raw.copy_(torch.randn(self.pattern_raw.shape, generator=g))

    def pattern(self) -> torch.Tensor:
        return torch.sigmoid(self.pattern_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * x + self.alpha * self.pattern()


class WarpTrigger(LearnableTrigger):
    """WaNet-like: a coarse k x k displacement field stretched over the image and applied with grid_sample.

    The amplitude is bounded by `max_shift` (grid units, i.e. fraction of half the image: 0.06 ~ 2 px at 64 px).
    """

    name = "warp"

    def __init__(self, size: int = IMG, k: int = WARP.k, max_shift: float = WARP.max_shift):
        super().__init__(size)
        self.k, self.max_shift = k, max_shift
        self.flow_raw = nn.Parameter(torch.empty(1, 2, k, k))
        self.reset(0)

    @torch.no_grad()
    def reset(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.flow_raw.copy_(0.5 * torch.randn(self.flow_raw.shape, generator=g))

    def flow(self) -> torch.Tensor:
        """(1, H, W, 2) bounded displacement field."""
        f = upsample_field(self.max_shift * torch.tanh(self.flow_raw), self.size, self.size)
        return f.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return warp(x, self.flow()).clamp(0, 1)


class StripeTrigger(LearnableTrigger):
    """SIG-like: x' = clip(x + delta) with a low-frequency additive delta bounded by `eps`.

    delta is a coarse k x k x 3 map upsampled over the image (so no high frequencies) and squashed by tanh.
    """

    name = "stripe"

    def __init__(self, size: int = IMG, k: int = STRIPE.k, eps: float = STRIPE.eps):
        super().__init__(size)
        self.k, self.eps = k, eps
        self.delta_raw = nn.Parameter(torch.empty(1, 3, k, k))
        self.reset(0)

    @torch.no_grad()
    def reset(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.delta_raw.copy_(0.5 * torch.randn(self.delta_raw.shape, generator=g))

    def delta(self) -> torch.Tensor:
        return upsample_field(self.eps * torch.tanh(self.delta_raw), self.size, self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.delta()).clamp(0, 1)


FAMILIES: dict[str, type[LearnableTrigger]] = {
    cls.name: cls for cls in (PatchTrigger, BlendTrigger, WarpTrigger, StripeTrigger)
}


# ----------------------------------------------------------------------------------------------------------------
# stage 4: fixed bank
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedTrigger:
    name: str
    family: str  # badnets | blend | sig | wanet
    apply: Callable[[torch.Tensor], torch.Tensor]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)


def _patch_pattern(kind: str, s: int, seed: int) -> torch.Tensor:
    """(3, s, s) pattern in [0, 1]."""
    if kind == "white":
        return torch.ones(3, s, s)
    if kind == "black":
        return torch.zeros(3, s, s)
    if kind == "checker":
        ii = torch.arange(s)
        c = ((ii[:, None] + ii[None, :]) % 2).float()
        return c.expand(3, s, s).clone()
    if kind == "noise":
        g = torch.Generator().manual_seed(seed)
        return torch.rand(3, s, s, generator=g)
    if kind == "color":
        g = torch.Generator().manual_seed(seed)
        return torch.rand(3, 1, 1, generator=g).expand(3, s, s).clone()
    raise ValueError(kind)


def _patch_origin(position: str, s: int, margin: int = BANK.corner_margin, size: int = IMG) -> tuple[int, int]:
    far = size - s - margin
    return {
        "tl": (margin, margin),
        "tr": (margin, far),
        "bl": (far, margin),
        "br": (far, far),
        "center": ((size - s) // 2, (size - s) // 2),
    }[position]


def make_patch(kind: str, s: int, position: str, seed: int = 0, margin: int = BANK.corner_margin) -> FixedTrigger:
    pat = _patch_pattern(kind, s, seed)
    top, left = _patch_origin(position, s, margin)

    def apply(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, :, top : top + s, left : left + s] = pat.to(x)
        return out

    return FixedTrigger(f"badnets_{kind}{s}_{position}", "badnets", apply)


def make_blend(kind: str, alpha: float, seed: int = 0, size: int = IMG) -> FixedTrigger:
    g = torch.Generator().manual_seed(seed)
    if kind == "noise":
        pat = torch.rand(1, 3, size, size, generator=g)
    elif kind == "lowfreq":  # smooth colour clouds
        pat = upsample_field(torch.rand(1, 3, 4, 4, generator=g), size, size)
    else:
        raise ValueError(kind)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return (1 - alpha) * x + alpha * pat.to(x)

    return FixedTrigger(f"blend_{kind}_a{alpha:g}", "blend", apply)


def make_sig(amplitude: float, freq: int, vertical: bool = True, size: int = IMG) -> FixedTrigger:
    """Barni et al.: delta(i, j) = A * sin(2 pi j f / W), added to all channels."""
    t = torch.arange(size).float()
    wave = amplitude * torch.sin(2 * math.pi * t * freq / size)
    delta = wave.view(1, 1, 1, size) if vertical else wave.view(1, 1, size, 1)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return (x + delta.to(x)).clamp(0, 1)

    orient = "v" if vertical else "h"
    return FixedTrigger(f"sig_a{amplitude:g}_f{freq}{orient}", "sig", apply)


def make_wanet(k: int, strength: float, seed: int = 0, size: int = IMG) -> FixedTrigger:
    """Nguyen & Tran: a random k x k field, normalised by its mean |value|, stretched and scaled by `strength`.

    `strength` is in pixels of displacement at the given resolution (WaNet's s / H convention).
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(1, 2, k, k, generator=g) * 2 - 1
    noise = noise / noise.abs().mean()
    flow = (strength / size) * upsample_field(noise, size, size).permute(0, 2, 3, 1)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return warp(x, flow.to(x)).clamp(0, 1)

    return FixedTrigger(f"wanet_k{k}_s{strength:g}_seed{seed}", "wanet", apply)


def build_bank(cfg: BankCfg = BANK) -> list[FixedTrigger]:
    """The stage-4 bank: patches of several sizes/patterns/positions, blends, stripes and warps."""
    bank: list[FixedTrigger] = []
    for s in cfg.patch_sizes:
        for kind in cfg.patch_kinds:
            for pos in cfg.patch_positions:
                bank.append(make_patch(kind, s, pos, cfg.seed, cfg.corner_margin))
    for kind in cfg.blend_kinds:
        for alpha in cfg.blend_alphas:
            bank.append(make_blend(kind, alpha, cfg.seed))
    for amp in cfg.sig_amplitudes:
        for freq in cfg.sig_freqs:
            for vertical in (True, False):
                bank.append(make_sig(amp, freq, vertical))
    for k in cfg.wanet_ks:
        for strength in cfg.wanet_strengths:
            for s in range(cfg.wanet_seeds):
                bank.append(make_wanet(k, strength, cfg.seed + s))
    names = [t.name for t in bank]
    assert len(set(names)) == len(names), "duplicate trigger names in bank"
    return bank
