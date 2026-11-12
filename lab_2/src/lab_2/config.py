"""Tunable hyper-parameters for the trigger code (stages 4 and 5).

Frozen dataclasses with module-level instances: change a value here (or build a modified copy with
`dataclasses.replace`) instead of editing the trigger implementations.
"""

from dataclasses import dataclass

IMG = 64  # image side in pixels
N_CLASSES = 20

# stage 1: stratified A/B split of the clean images
PER_CLASS_HALF = 10  # images per class in each half (A: optimise triggers, B: evaluate)
SPLIT_SEED = 0


@dataclass(frozen=True)
class PatchCfg:
    init_mask_logit: float = -3.0  # sigmoid(-3) ~ 5% initial mask coverage


@dataclass(frozen=True)
class BlendCfg:
    alpha: float = 0.15  # fixed blend strength


@dataclass(frozen=True)
class WarpCfg:
    k: int = 4  # coarse displacement grid is k x k
    max_shift: float = 0.06  # grid_sample units; 0.06 ~ 2 px at 64 px


@dataclass(frozen=True)
class StripeCfg:
    k: int = 8  # coarse delta grid is k x k
    eps: float = 0.12  # max additive change per pixel


@dataclass(frozen=True)
class BankCfg:
    """Stage-4 fixed bank."""

    seed: int = 0
    corner_margin: int = 1
    patch_sizes: tuple[int, ...] = (3, 5, 8)
    patch_kinds: tuple[str, ...] = ("white", "black", "checker", "noise")
    patch_positions: tuple[str, ...] = ("tl", "tr", "bl", "br", "center")
    blend_kinds: tuple[str, ...] = ("noise", "lowfreq")
    blend_alphas: tuple[float, ...] = (0.1, 0.2)
    sig_amplitudes: tuple[float, ...] = (0.1, 0.2)
    sig_freqs: tuple[int, ...] = (4, 8)
    wanet_ks: tuple[int, ...] = (4, 8)
    wanet_strengths: tuple[float, ...] = (1.0, 2.0)  # pixels of displacement
    wanet_seeds: int = 2  # number of random fields per (k, strength)


@dataclass(frozen=True)
class NormCfg:
    """Stage 7: per (source, score type, class) robust z-score over the test models."""

    mad_floor: float = 0.05  # scores move in steps of 1/20 (20 images per class), so MAD is often 0
    mad_scale: float = 1.4826  # makes MAD comparable to a standard deviation for normal data


PATCH = PatchCfg()
NORM = NormCfg()
BLEND = BlendCfg()
WARP = WarpCfg()
STRIPE = StripeCfg()
BANK = BankCfg()
