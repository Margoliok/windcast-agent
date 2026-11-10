"""M6 — Ансамбль: мүше-модельдердің кванттильдерін салмақты орташалау.

Салмақтар валидация кезеңінде CRPS-ті минимизациялау арқылы симплекс торында (қадам 0.05)
таңдалады. Кванттильдерді орташалау (Винсент орташасы) кванттардың ретін сақтайды.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from app.evaluation.metrics import crps_from_quantiles
from app.models.base import finalize_quantiles


def _simplex(n: int, step: float = 0.05):
    k = int(round(1 / step))
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        s = sum(combo)
        if s <= k:
            yield np.array([*combo, k - s], dtype=float) / k


def fit_weights(preds: dict[str, pd.DataFrame], y: pd.Series, step: float = 0.05) -> dict[str, float]:
    names = list(preds)
    cols = [c for c in preds[names[0]].columns if c.startswith("q")]
    taus = [float(c[1:]) for c in cols]
    mats = [preds[n][cols].to_numpy() for n in names]
    yv = np.asarray(y, float)
    mask = np.isfinite(yv)
    for m in mats:
        mask &= np.isfinite(m).all(axis=1)
    best, best_w = np.inf, None
    for w in _simplex(len(names), step if len(names) <= 3 else 0.1):
        q = sum(wi * m[mask] for wi, m in zip(w, mats))
        score = crps_from_quantiles(yv[mask], q, taus)
        if score < best:
            best, best_w = score, w
    return {n: float(w) for n, w in zip(names, best_w)}


def combine(preds: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    names = [n for n in weights if weights[n] > 0 and n in preds]
    cols = [c for c in preds[names[0]].columns if c.startswith("q")]
    taus = [float(c[1:]) for c in cols]
    total = sum(weights[n] for n in names)
    q = sum(weights[n] / total * preds[n][cols].to_numpy() for n in names)
    return finalize_quantiles(q, taus, preds[names[0]].index)
