"""
Algorithm 4: penalised monotone step-function segmentation by dynamic
programming. Returns the empirical partition count N-hat and the fitted
step function g.

Given pairs (omega_(t), a_hat_(t)) sorted by omega, find the K in
{1, ..., K_max} that minimises  SSE(K) + lambda_T * K * log(T).
For each K, the optimal K-segment partition is computed via standard
O(T^2 K) DP using the segment-sum-of-squares formula.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import config


@dataclass
class StepFit:
    K: int
    boundaries: np.ndarray       # omega values where steps change, length K-1
    levels:     np.ndarray       # length K, weakly monotone
    sse:        float
    criterion:  float
    g_hat:      np.ndarray       # fitted level at each sample point


def _segment_costs(y: np.ndarray) -> np.ndarray:
    """cost[i, j] = sum_{t=i..j} (y_t - mean(y_{i..j}))^2."""
    T = len(y)
    # Cumulative sums for O(1) segment mean and SSE.
    cs1 = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])
    cost = np.zeros((T, T))
    for i in range(T):
        for j in range(i, T):
            s1 = cs1[j + 1] - cs1[i]
            s2 = cs2[j + 1] - cs2[i]
            m = j - i + 1
            cost[i, j] = s2 - s1 * s1 / m
    return cost


def _isotonic_pool(levels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for weak monotonicity (non-decreasing)."""
    L = list(map(float, levels))
    W = list(map(float, weights))
    i = 0
    while i < len(L) - 1:
        if L[i] > L[i + 1]:
            new_w = W[i] + W[i + 1]
            new_v = (L[i] * W[i] + L[i + 1] * W[i + 1]) / new_w
            L[i:i + 2] = [new_v]
            W[i:i + 2] = [new_w]
            i = max(0, i - 1)
        else:
            i += 1
    # Expand back to original length.
    out, idx = [], 0
    for v, w in zip(L, W):
        # find how many original cells this pooled block covers
        s = 0.0
        while idx < len(weights) and s + weights[idx] <= w + 1e-9:
            s += weights[idx]
            out.append(v)
            idx += 1
    return np.asarray(out)


def fit_step_function(omegas: np.ndarray,
                      a_hat: np.ndarray,
                      k_max: int = config.SEGMENT_KMAX,
                      lam: float = config.SEGMENT_PENALTY_LAMBDA) -> StepFit:
    """Pick K via penalised DP and return the fitted weakly-monotone step."""
    order = np.argsort(omegas)
    om = np.asarray(omegas)[order]
    y  = np.asarray(a_hat)[order]
    T = len(y)
    if T == 0:
        return StepFit(0, np.array([]), np.array([]), 0.0, 0.0, np.zeros(0))

    cost = _segment_costs(y)
    K_max = max(1, min(k_max, T))
    # dp[k, j] = min SSE using k segments to cover y[0..j].
    INF = float("inf")
    dp = np.full((K_max + 1, T), INF)
    prev = np.full((K_max + 1, T), -1, dtype=int)
    for j in range(T):
        dp[1, j] = cost[0, j]
    for k in range(2, K_max + 1):
        for j in range(k - 1, T):
            best, best_i = INF, -1
            for i in range(k - 1, j + 1):
                cand = dp[k - 1, i - 1] + cost[i, j]
                if cand < best:
                    best, best_i = cand, i
            dp[k, j] = best
            prev[k, j] = best_i

    # For each K, reconstruct segments, apply isotonic pooling (Algorithm 4
    # line 430), then recompute SSE on the pooled levels so that the
    # criterion reflects the monotone step function actually returned.
    def _reconstruct(K: int):
        segs = []
        jj = T - 1
        for k in range(K, 0, -1):
            ii = 0 if k == 1 else prev[k, jj]
            segs.append((ii, jj))
            jj = ii - 1
        segs.reverse()
        lvls = np.array([y[i:j + 1].mean() for i, j in segs])
        wts  = np.array([j - i + 1         for i, j in segs], dtype=float)
        lvls = _isotonic_pool(lvls, wts)
        g_local = np.empty(T)
        for (i, j), v in zip(segs, lvls):
            g_local[i:j + 1] = v
        sse_local = float(np.sum((y - g_local) ** 2))
        return segs, lvls, g_local, sse_local

    crits = {}
    fits  = {}
    for K in range(1, K_max + 1):
        segs, lvls, g_local, sse_local = _reconstruct(K)
        crits[K] = sse_local + lam * K * np.log(max(T, 2))
        fits[K]  = (segs, lvls, g_local, sse_local)
    K_star = min(crits, key=crits.get)
    segs, levels, g_hat_sorted, sse_star = fits[K_star]

    bounds = []
    for (i, j), _ in zip(segs, levels):
        if j + 1 < T:
            bounds.append(0.5 * (om[j] + om[j + 1]))
    g_hat = np.empty(T)
    g_hat[order] = g_hat_sorted

    return StepFit(K=K_star,
                   boundaries=np.asarray(bounds),
                   levels=levels,
                   sse=sse_star,
                   criterion=float(crits[K_star]),
                   g_hat=g_hat)
