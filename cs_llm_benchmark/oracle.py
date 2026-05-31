"""
Crawford-Sobel oracle (Algorithm 1) and exact reference quantities
(Proposition 2 / Table 4) for the uniform-quadratic environment.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, sqrt
from typing import List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Most-informative partition size.                                             #
# --------------------------------------------------------------------------- #
def n_cs(b: float) -> int:
    """Maximum feasible monotone partition size N_CS(b) (Proposition 1)."""
    if b <= 0.0:
        return 0  # 0 codes for "full revelation"
    return int(ceil(-0.5 + 0.5 * sqrt(1.0 + 2.0 / b)))


@dataclass(frozen=True)
class Partition:
    """One monotone partition of [0, 1]."""
    b: float
    boundaries: np.ndarray   # length N+1, t_0=0 ... t_N=1
    lengths:    np.ndarray   # length N
    actions:    np.ndarray   # length N, cell midpoints (Bayes receiver action)

    @property
    def N(self) -> int:
        return len(self.lengths)


def oracle_partition(b: float) -> Partition:
    """Algorithm 1: build the most-informative CS partition for bias b."""
    if b <= 0.0:
        # "Full revelation" sentinel: treat as 1-cell trivial partition only
        # so that helper code does not have to special-case b == 0. Mutual
        # information / loss baselines for b == 0 are computed directly.
        return Partition(b=0.0,
                         boundaries=np.array([0.0, 1.0]),
                         lengths=np.array([1.0]),
                         actions=np.array([0.5]))

    N = n_cs(b)
    ell1 = (1.0 - 2.0 * b * N * (N - 1)) / N
    lengths = np.array([ell1 + 4.0 * b * (j) for j in range(N)])
    boundaries = np.concatenate([[0.0], np.cumsum(lengths)])
    boundaries[-1] = 1.0                       # guard floating-point drift
    actions = 0.5 * (boundaries[:-1] + boundaries[1:])
    return Partition(b=b, boundaries=boundaries, lengths=lengths, actions=actions)


def oracle_action(b: float, omega: np.ndarray | float) -> np.ndarray:
    """Map state(s) to the oracle receiver action through the CS partition."""
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if b <= 0.0:
        return omega.copy()                    # full revelation
    p = oracle_partition(b)
    # np.searchsorted returns 1..N for interior states; clip just in case.
    idx = np.clip(np.searchsorted(p.boundaries, omega, side="right") - 1, 0, p.N - 1)
    return p.actions[idx]


# --------------------------------------------------------------------------- #
# Oracle losses (Proposition 2).                                               #
# --------------------------------------------------------------------------- #
def oracle_loss_receiver(b: float) -> float:
    if b <= 0.0:
        return 0.0
    p = oracle_partition(b)
    return float(np.sum(p.lengths ** 3) / 12.0)


def oracle_loss_sender(b: float) -> float:
    return oracle_loss_receiver(b) + b ** 2


def full_revelation_loss_sender(b: float) -> float:
    return b ** 2


def babbling_loss_receiver() -> float:
    return 1.0 / 12.0


def babbling_loss_sender(b: float) -> float:
    return 1.0 / 12.0 + b ** 2


# --------------------------------------------------------------------------- #
# Oracle 20-bin normalized mutual information.                                 #
# --------------------------------------------------------------------------- #
def oracle_normalized_mi(b: float, bins: int = 20) -> float:
    """Normalized MI between bin(omega) and bin(oracle action), uniform prior.

    For b == 0 the oracle reveals omega exactly so I_norm = 1.
    """
    if b <= 0.0:
        return 1.0
    p = oracle_partition(b)
    # Joint mass: probability that omega falls in state-bin r and that the
    # corresponding cell midpoint lands in action-bin c.
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Action bin for each cell (action is fixed within a cell).
    action_bin = np.clip(np.searchsorted(edges, p.actions, side="right") - 1, 0, bins - 1)
    joint = np.zeros((bins, bins))
    for j in range(p.N):
        lo, hi = p.boundaries[j], p.boundaries[j + 1]
        # Distribute the cell's mass (= length) across the state bins it covers.
        # Each state-bin r contributes mass(overlap(lo,hi, edges[r], edges[r+1])).
        for r in range(bins):
            ov = max(0.0, min(hi, edges[r + 1]) - max(lo, edges[r]))
            if ov > 0:
                joint[r, action_bin[j]] += ov
    pr = joint.sum(axis=1, keepdims=True)
    pc = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((joint > 0) & (pr > 0) & (pc > 0),
                         joint / (pr @ pc), 1.0)
        mi = float(np.sum(joint * np.log(ratio)))
    h_state = float(-np.sum(pr * np.log(np.where(pr > 0, pr, 1.0))))
    return mi / h_state if h_state > 0 else 0.0


# --------------------------------------------------------------------------- #
# Convenience: regenerate Table 4 from scratch.                                #
# --------------------------------------------------------------------------- #
def reference_table(biases: Sequence[float]) -> List[dict]:
    """Return one row per bias, matching Table 4 columns."""
    rows = []
    for b in biases:
        rows.append({
            "b":         b,
            "N_CS":      n_cs(b) if b > 0 else "Full",
            "I_norm_20": oracle_normalized_mi(b),
            "L_R_CS":    oracle_loss_receiver(b),
            "L_S_CS":    oracle_loss_sender(b),
            "L_S_reveal": full_revelation_loss_sender(b),
            "L_R_babble": babbling_loss_receiver(),
            "L_S_babble": babbling_loss_sender(b),
        })
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(reference_table([0.0, 0.01, 0.04, 0.08, 0.12]), indent=2))
