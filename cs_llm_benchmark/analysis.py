"""
Hypothesis tests (Section 7) + bootstrap confidence intervals.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Bootstrap.                                                                   #
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float],
                 n_boot: int = 1000,
                 alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(v), len(v))
        boots[i] = v[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Fixed-effects regression Y = alpha_M + gamma_p + beta * b + eps. Closed form #
# OLS with dummies built by hand to avoid statsmodels as a hard dependency.    #
# --------------------------------------------------------------------------- #
@dataclass
class Regression:
    beta: float
    se_beta: float
    t_beta: float
    coefs: Dict[str, float]


def fixed_effects_regression(rows: List[dict], y_key: str) -> Regression:
    models = sorted({r["model"] for r in rows})
    frames = sorted({r["frame"] for r in rows})

    # Build the design matrix: intercept + (M-1) model dummies + (P-1) frame
    # dummies + b.
    cols = ["const"] + [f"M={m}" for m in models[1:]] + \
           [f"F={p}" for p in frames[1:]] + ["b"]
    X = np.zeros((len(rows), len(cols)))
    y = np.array([r[y_key] for r in rows], dtype=float)
    for i, r in enumerate(rows):
        X[i, 0] = 1.0
        if r["model"] in models[1:]:
            X[i, 1 + models[1:].index(r["model"])] = 1.0
        if r["frame"] in frames[1:]:
            X[i, 1 + len(models) - 1 + frames[1:].index(r["frame"])] = 1.0
        X[i, -1] = r["bias"]
    # OLS in closed form.
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    resid = y - X @ beta_hat
    df = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid / df)
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return Regression(
        beta=float(beta_hat[-1]),
        se_beta=float(se[-1]),
        t_beta=float(beta_hat[-1] / se[-1]) if se[-1] > 0 else float("nan"),
        coefs={c: float(v) for c, v in zip(cols, beta_hat)},
    )


# --------------------------------------------------------------------------- #
# Prompt-frame contrasts: payoff vs honesty at each (model, bias).             #
# --------------------------------------------------------------------------- #
def frame_contrasts(rows: List[dict], y_key: str,
                    frame_a: str = "payoff",
                    frame_b: str = "honesty") -> List[dict]:
    out = []
    by_key: Dict[Tuple[str, float], Dict[str, float]] = {}
    for r in rows:
        by_key.setdefault((r["model"], r["bias"]), {})[r["frame"]] = r[y_key]
    for (m, b), d in sorted(by_key.items()):
        if frame_a in d and frame_b in d:
            out.append({"model": m, "bias": b,
                        f"{y_key}_{frame_a}": d[frame_a],
                        f"{y_key}_{frame_b}": d[frame_b],
                        "contrast": d[frame_a] - d[frame_b]})
    return out


# --------------------------------------------------------------------------- #
# OLS slope of Y on b restricted to positive biases (Hypothesis 1).            #
# --------------------------------------------------------------------------- #
def slope_on_bias(rows: List[dict], y_key: str,
                  positive_only: bool = True) -> Dict[str, float]:
    sel = [r for r in rows if (r["bias"] > 0 if positive_only else True)]
    x = np.array([r["bias"] for r in sel], dtype=float)
    y = np.array([r[y_key]  for r in sel], dtype=float)
    if len(x) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n": len(x)}
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "n": len(x)}
