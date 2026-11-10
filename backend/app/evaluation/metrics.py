"""Болжам сапасының метрикалары (формулалар 5, 12–17).

Барлық функциялар нормаланған қуатпен (P/P_inst, 0..1) де, МВт-пен де жұмыс істейді.
nMAE/nRMSE үшін `capacity` беріледі (нормаланған қуат үшін capacity=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _clean(y, *preds):
    y = np.asarray(y, dtype=float)
    arrs = [np.asarray(p, dtype=float) for p in preds]
    mask = np.isfinite(y)
    for a in arrs:
        mask &= np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)
    return (y[mask], *[a[mask] for a in arrs])


def mae(y, yhat) -> float:
    """(12)"""
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    """(13)"""
    y, yhat = _clean(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.mean(yhat - y))


def nmae(y, yhat, capacity: float = 1.0) -> float:
    """(14) nMAE = MAE / P_inst · 100%"""
    return mae(y, yhat) / capacity * 100.0


def nrmse(y, yhat, capacity: float = 1.0) -> float:
    return rmse(y, yhat) / capacity * 100.0


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """(15) SS = 1 − RMSE_model / RMSE_persistence"""
    if not rmse_reference:
        return float("nan")
    return 1.0 - rmse_model / rmse_reference


def pinball(y, q_pred, tau: float) -> float:
    """(5) Пинбол функциясы L_τ(y, q) = max(τ(y−q), (τ−1)(y−q)), орташа мәні."""
    y, q = _clean(y, q_pred)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def crps_from_quantiles(y, q_matrix, taus) -> float:
    """(16) CRPS ≈ 2/|T| · Σ_τ L_τ — кванттар жиыны бойынша жуықтау."""
    q_matrix = np.asarray(q_matrix, dtype=float)
    y, q = _clean(y, q_matrix)
    losses = []
    for j, tau in enumerate(taus):
        d = y - q[:, j]
        losses.append(np.mean(np.maximum(tau * d, (tau - 1) * d)))
    return float(2.0 / len(taus) * np.sum(losses))


def picp(y, lower, upper) -> float:
    """(17) Болжам аралығын қамту үлесі, %."""
    y, lo, hi = _clean(y, lower, upper)
    return float(np.mean((y >= lo) & (y <= hi)) * 100.0)


def interval_width(lower, upper, capacity: float = 1.0) -> float:
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    m = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[m] - lo[m]) / capacity * 100.0)


def point_metrics(y, yhat, capacity: float = 1.0) -> dict:
    return {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "nmae": nmae(y, yhat, capacity),
        "nrmse": nrmse(y, yhat, capacity),
        "bias": bias(y, yhat),
        "n": int(np.isfinite(np.asarray(y, float)).sum()),
    }


def probabilistic_metrics(y, qdf: pd.DataFrame, capacity: float = 1.0) -> dict:
    """qdf бағандары: q0.1 ... q0.9 (кемінде q0.1, q0.5, q0.9)."""
    cols = sorted([c for c in qdf.columns if c.startswith("q")], key=lambda c: float(c[1:]))
    taus = [float(c[1:]) for c in cols]
    out = {
        "crps": crps_from_quantiles(np.asarray(y) / capacity, qdf[cols].to_numpy() / capacity, taus),
        "pinball_p50": pinball(np.asarray(y) / capacity, qdf["q0.5"].to_numpy() / capacity, 0.5),
    }
    if "q0.1" in qdf and "q0.9" in qdf:
        out["picp_80"] = picp(y, qdf["q0.1"], qdf["q0.9"])
        out["width_80"] = interval_width(qdf["q0.1"], qdf["q0.9"], capacity)
    return out
