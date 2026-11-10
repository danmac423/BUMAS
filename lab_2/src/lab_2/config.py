"""All tunable knobs in one place. Every field can be overridden with a JSON file (``--config``)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FormCfg:
    """Per-trigger-form optimisation settings."""

    enabled: bool = True
    lr: float = 0.1
    lam0: float = 0.1  # initial size-penalty weight (then adapted per class)
    lam_min: float = 1e-5
    lam_max: float = 1e3
    # form specific
    res: int = 0  # blend: low-res of delta (0 = full res); warp: flow grid size k
    tv_w: float = 0.5  # warp: weight of the TV term in the penalty
    a0: float = -2.0  # patch: initial mask logit (sigmoid(-2) = 12%)
    p1_steps: int | None = None  # per-form override of Config.p1_steps
    p2_steps: int | None = None  # per-form override of Config.p2_steps


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    model_py: str = str(PROJECT_ROOT / "example" / "model.py")
    clean_dir: str = str(PROJECT_ROOT / "example" / "clean")

    # ---- data --------------------------------------------------------------
    seed: int = 0
    held_frac: float = 0.25  # fraction of clean images held out for ASR measurement
    device: str = "cuda"

    # ---- numerics ----------------------------------------------------------
    amp: bool = True
    loss_scale: float = 1024.0  # static fp16 loss scale (Adam is scale invariant)
    max_batch: int = 640  # max images per forward/backward chunk

    # ---- all-class scan (K-Arm style) ---------------------------------------
    asr_target: float = 0.9
    asr_band: float = 0.03  # dead band around the target for the lambda controller
    lam_eta: float = 0.35  # log-step of the lambda controller
    ema: float = 0.8  # EMA decay of the batch ASR
    warmup: int = 8  # steps before snapshots are allowed
    p1_steps: int = 100
    p1_batch: int = 16
    p2_steps: int = 60
    p2_batch: int = 32
    p2_lr_scale: float = 0.5  # Adam is restarted for the top-K refinement, so use a smaller lr
    topk: int = 5
    patch: FormCfg = field(default_factory=lambda: FormCfg(lr=0.3, lam0=1.0, a0=-3.0))
    blend: FormCfg = field(default_factory=lambda: FormCfg(lr=0.01, lam0=0.3, res=0, p1_steps=60, p2_steps=40))
    warp: FormCfg = field(default_factory=lambda: FormCfg(lr=0.1, lam0=0.1, res=4, tv_w=0.5, p1_steps=60, p2_steps=40))

    # ---- targeted (source specific) scan -------------------------------------
    targeted: bool = True
    tgt_form: str = "patch"
    tgt_steps: int = 80
    tgt_batch: int = 8
    tgt_topk: int = 8  # targets (largest per-source ASR spike of the all-class trigger) that get pair scans
    tgt_src: int = 3  # best-flipping sources scanned per selected target
    tgt_leak_n: int = 24  # other-class images used to measure leakage

    # ---- scoring -----------------------------------------------------------
    w_asr: float = 2.0  # penalty (in log-size units) per missing unit of held-out ASR
    w_nc: float = 0.5  # weight of the within-model NC anomaly index
    w_tgt: float = 1.0  # weight of the targeted-scan z score in the max-fusion
    w_leak: float = 0.0  # penalty of the targeted pair score for leaking to other classes
    mad_floor: float = 0.05

    def forms(self) -> dict[str, FormCfg]:
        return {n: getattr(self, n) for n in ("patch", "blend", "warp") if getattr(self, n).enabled}

    @classmethod
    def from_json(cls, path: str | Path | None) -> Config:
        cfg = cls()
        if path:
            _update(cfg, json.loads(Path(path).read_text()))
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def _update(obj, d: dict) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(obj, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _update(cur, v)
        else:
            setattr(obj, k, v)
