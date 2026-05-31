"""
Section 3.4 estimands + Algorithm 5 (evaluate a model-bias-frame cell).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from . import config, oracle, segmentation, receiver


# --------------------------------------------------------------------------- #
# Discretised mutual information.                                              #
# --------------------------------------------------------------------------- #
def normalized_mi(omegas: np.ndarray,
                  actions: np.ndarray,
                  bins: int = config.MI_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    r = np.clip(np.searchsorted(edges, omegas, side="right") - 1, 0, bins - 1)
    c = np.clip(np.searchsorted(edges, np.clip(actions, 0, 1),
                                side="right") - 1, 0, bins - 1)
    joint = np.zeros((bins, bins))
    for ri, ci in zip(r, c):
        joint[ri, ci] += 1
    joint /= max(joint.sum(), 1.0)
    pr = joint.sum(1, keepdims=True)
    pc = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((joint > 0) & (pr > 0) & (pc > 0),
                         joint / (pr @ pc), 1.0)
        mi = float(np.sum(joint * np.log(ratio)))
    h = float(-np.sum(pr * np.log(np.where(pr > 0, pr, 1.0))))
    return mi / h if h > 0 else 0.0


# --------------------------------------------------------------------------- #
# Cell-level metric bundle.                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class CellMetrics:
    model: str
    bias: float
    frame: str
    n: int
    N_hat: int
    N_CS: int
    N_ratio: float
    I_norm: float
    L_R: float
    L_S: float
    L_R_oracle: float
    L_S_oracle: float
    delta_R: float
    delta_S: float
    over_reveal_mi: bool
    over_reveal_N: bool
    # Bootstrap 95% CIs for the mean-based metrics, populated when
    # evaluate_cell is called with n_boot > 0.
    L_R_lo: float = float("nan")
    L_R_hi: float = float("nan")
    L_S_lo: float = float("nan")
    L_S_hi: float = float("nan")
    I_norm_lo: float = float("nan")
    I_norm_hi: float = float("nan")


def evaluate_cell(model: str, bias: float, frame: str,
                  omegas: np.ndarray, a_hat: np.ndarray,
                  n_boot: int = 0,
                  bootstrap_seed: int = 0,
                  bins: int = config.MI_BINS) -> CellMetrics:
    """Algorithm 5. a_hat is the UNCLIPPED decoder output; this function
    applies the loss-reporting clip locally per Algorithm 3 line 414."""
    n = len(omegas)
    # Segmentation and MI receive the unclipped predictions.
    fit = segmentation.fit_step_function(omegas, a_hat)
    n_cs_val = oracle.n_cs(bias)
    i_norm = normalized_mi(omegas, a_hat, bins=bins)

    # Losses use the clipped predictions per Algorithm 3 line 414.
    a_clip = receiver.clip01(a_hat)
    l_r = float(np.mean((a_clip - omegas) ** 2))
    l_s = float(np.mean((a_clip - omegas - bias) ** 2))
    a_oracle = oracle.oracle_action(bias, omegas)
    l_r_o = float(np.mean((a_oracle - omegas) ** 2))
    l_s_o = float(np.mean((a_oracle - omegas - bias) ** 2))

    if bias <= 0.0:
        over_mi = False
        over_n  = False
        n_ratio = 0.0
    else:
        i_cs = oracle.oracle_normalized_mi(bias, bins=bins)
        over_mi = i_norm > i_cs + config.OVERREVEAL_MI_TOL
        over_n  = fit.K > n_cs_val
        n_ratio = fit.K / max(n_cs_val, 1)

    # Bootstrap CIs (Algorithm 5 line 454; Section 7 line 516). State-message
    # pair resampling within the cell.
    L_R_lo = L_R_hi = L_S_lo = L_S_hi = I_lo = I_hi = float("nan")
    if n_boot > 0 and n > 1:
        rng = np.random.default_rng(bootstrap_seed)
        lr_b = np.empty(n_boot); ls_b = np.empty(n_boot); imi_b = np.empty(n_boot)
        for k in range(n_boot):
            idx = rng.integers(0, n, n)
            ac = a_clip[idx]; om = omegas[idx]; au = a_hat[idx]
            lr_b[k]  = np.mean((ac - om) ** 2)
            ls_b[k]  = np.mean((ac - om - bias) ** 2)
            imi_b[k] = normalized_mi(om, au, bins=bins)
        L_R_lo, L_R_hi = np.quantile(lr_b,  [0.025, 0.975])
        L_S_lo, L_S_hi = np.quantile(ls_b,  [0.025, 0.975])
        I_lo,   I_hi   = np.quantile(imi_b, [0.025, 0.975])

    return CellMetrics(
        model=model, bias=float(bias), frame=frame, n=n,
        N_hat=int(fit.K), N_CS=int(n_cs_val), N_ratio=float(n_ratio),
        I_norm=i_norm, L_R=l_r, L_S=l_s,
        L_R_oracle=l_r_o, L_S_oracle=l_s_o,
        delta_R=l_r - l_r_o, delta_S=l_s - l_s_o,
        over_reveal_mi=bool(over_mi), over_reveal_N=bool(over_n),
        L_R_lo=float(L_R_lo), L_R_hi=float(L_R_hi),
        L_S_lo=float(L_S_lo), L_S_hi=float(L_S_hi),
        I_norm_lo=float(I_lo), I_norm_hi=float(I_hi),
    )


def cell_to_row(m: CellMetrics) -> Dict:
    return asdict(m)
