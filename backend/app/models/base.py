"""Барлық болжау модельдерінің ортақ интерфейсі.

Модельдер нормаланған қуатпен (P/P_inst ∈ [0,1]) жұмыс істейді және кванттильдер
кестесін қайтарады: q0.1 … q0.9 бағандары. P10 = q0.1, P50 = q0.5, P90 = q0.9.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from app.evaluation.metrics import QUANTILES


def qcol(tau: float) -> str:
    return f"q{tau:g}"


def finalize_quantiles(q: np.ndarray, taus=QUANTILES, index=None) -> pd.DataFrame:
    """Кванттильдерді реттеу (қиылысуды болдырмау) және [0,1] аралығына қию."""
    q = np.clip(np.sort(np.asarray(q, dtype=float), axis=1), 0.0, 1.0)
    return pd.DataFrame(q, columns=[qcol(t) for t in taus], index=index)


def residual_quantiles(resid: np.ndarray, taus=QUANTILES) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    return np.quantile(resid, taus) if len(resid) else np.zeros(len(taus))


class Forecaster(ABC):
    """Кванттильдік болжау моделі."""

    name: str = "base"
    title: dict = {"kk": "", "ru": ""}
    probabilistic: bool = True
    taus = QUANTILES

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def predict_p50(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)["q0.5"].to_numpy()
