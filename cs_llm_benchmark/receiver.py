"""
Algorithm 3: cross-fitted receiver decoder.

Default embedder = sentence-transformers MiniLM (loaded lazily). If that is
not installed the code falls back to a hashing-trigram bag-of-words
embedding so the pipeline remains runnable in offline / smoke-test mode.
The decoder itself is a ridge regression (numpy closed form).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import hashlib
import numpy as np


# --------------------------------------------------------------------------- #
# Embedders.                                                                   #
# --------------------------------------------------------------------------- #
class Embedder:
    name: str = "abstract"
    dim: int = 0
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    def __init__(self):
        from sentence_transformers import SentenceTransformer  # lazy
        self.model = SentenceTransformer(self.name)
        self.dim = self.model.get_sentence_embedding_dimension()
    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), show_progress_bar=False,
                                            normalize_embeddings=True))


class HashingTrigramEmbedder(Embedder):
    """Offline-safe fallback. Hashes character trigrams into a fixed-width
    vector and L2-normalises. Surprisingly serviceable for short numeric
    messages, and zero external dependencies."""
    name = "hashing-trigram-256"
    def __init__(self, dim: int = 256):
        self.dim = dim
    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            s = " " + (t or "").lower() + " "
            for k in range(len(s) - 2):
                tri = s[k:k + 3]
                h = int(hashlib.md5(tri.encode()).hexdigest(), 16) % self.dim
                out[i, h] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def default_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingTrigramEmbedder()


# --------------------------------------------------------------------------- #
# Ridge regression in closed form.                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Ridge:
    alpha: float = 1.0
    w_: Optional[np.ndarray] = None
    b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        Xc = X - X.mean(0, keepdims=True)
        yc = y - y.mean()
        d = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(d)
        self.w_ = np.linalg.solve(A, Xc.T @ yc)
        self.b_ = float(y.mean() - X.mean(0) @ self.w_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w_ + self.b_


# --------------------------------------------------------------------------- #
# Algorithm 3 driver.                                                          #
# --------------------------------------------------------------------------- #
import re

_NUM_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def numeric_value(message: str) -> Optional[float]:
    """Extract the first number from a sender message, if any.

    Sender messages in this benchmark are overwhelmingly of the form
    "0.4231" / "approximately 0.31" — i.e. the sender states a numeric
    report. A rational receiver reads that number. Returns None when the
    message contains no parseable number (then the embedding path is used).
    """
    if not message:
        return None
    m = _NUM_RE.search(message)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def cross_fit_actions(messages: Sequence[str],
                      omegas: np.ndarray,
                      embedder: Optional[Embedder] = None,
                      folds: int = 5,
                      ridge_alpha: float = 1.0,
                      seed: int = 0,
                      regressor: str = "hybrid",
                      knn_k: int = 25,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 3. Returns (a_hat, fold_id) — UNCLIPPED predictions.

    Per the paper (Algorithm 3, line 414) clipping to [0, 1] is applied
    only for empirical-loss reporting; segmentation and mutual information
    receive the unclipped predictions.

    regressor:
      "hybrid" (default) — the receiver reads the sender's stated number when
            one is present (1-D cross-fitted linear map omega ~ parsed number),
            and falls back to the embedding ridge for non-numeric messages.
            A pure embedding decoder cannot recover the value of a number
            string (e.g. "0.0592"), which makes it fail the b=0 truthfulness
            sanity check; the hybrid path fixes that mis-specification.
      "ridge" — pure embedding ridge (kept as a robustness/ablation decoder).
      "knn"   — embedding k-NN (second alternative decoder, Section 7).
    """
    n = len(messages)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    if embedder is None:
        embedder = default_embedder()
    y = np.asarray(omegas, dtype=float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    fold_id[perm] = np.arange(n) % folds
    a_hat = np.empty(n)

    if regressor == "hybrid":
        nums = np.array([numeric_value(m) for m in messages], dtype=object)
        has_num = np.array([v is not None for v in nums])
        num_val = np.array([float(v) if v is not None else np.nan for v in nums])
        # Embeddings only needed for the non-numeric fallback; compute lazily.
        X = embedder.encode(messages) if (~has_num).any() else None
        for k in range(folds):
            train = fold_id != k
            test = ~train
            if train.sum() == 0 or test.sum() == 0:
                continue
            tr_num = train & has_num
            # 1-D linear map omega ~ stated number, fit on numeric training msgs.
            if tr_num.sum() >= 2:
                slope, intercept = np.polyfit(num_val[tr_num], y[tr_num], 1)
            else:
                slope, intercept = 0.0, float(y[train].mean())
            te_num = test & has_num
            a_hat[te_num] = slope * num_val[te_num] + intercept
            # Embedding fallback for non-numeric test messages.
            te_emb = test & ~has_num
            if te_emb.any():
                tr_emb = train & ~has_num
                if tr_emb.sum() >= 2 and X is not None:
                    model = Ridge(alpha=ridge_alpha).fit(X[tr_emb], y[tr_emb])
                    a_hat[te_emb] = model.predict(X[te_emb])
                else:
                    a_hat[te_emb] = float(y[train].mean())
        return a_hat, fold_id

    X = embedder.encode(messages)
    for k in range(folds):
        train = fold_id != k
        test  = ~train
        if train.sum() == 0 or test.sum() == 0:
            continue
        if regressor == "ridge":
            model = Ridge(alpha=ridge_alpha).fit(X[train], y[train])
            a_hat[test] = model.predict(X[test])
        elif regressor == "knn":
            a_hat[test] = _knn_predict(X[train], y[train], X[test], knn_k)
        else:
            raise ValueError(f"unknown regressor {regressor!r}")
    return a_hat, fold_id


def _knn_predict(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray, k: int) -> np.ndarray:
    """Plain k-NN regressor (cosine on L2-normalised embeddings ~ Euclidean)."""
    k = min(k, len(X_tr))
    # squared Euclidean distances, then take k smallest per query row
    d = ((X_te[:, None, :] - X_tr[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return y_tr[nn].mean(axis=1)


def clip01(a: np.ndarray) -> np.ndarray:
    """Loss-reporting clip per Algorithm 3, line 414."""
    return np.clip(a, 0.0, 1.0)


def zero_bias_r2(messages: Sequence[str],
                 omegas: np.ndarray,
                 embedder: Optional[Embedder] = None) -> float:
    """R^2 of the receiver decoder on the b=0 truthfulness slice."""
    a_hat, _ = cross_fit_actions(messages, omegas, embedder=embedder)
    omegas = np.asarray(omegas, dtype=float)
    ss_res = float(np.sum((omegas - a_hat) ** 2))
    ss_tot = float(np.sum((omegas - omegas.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
