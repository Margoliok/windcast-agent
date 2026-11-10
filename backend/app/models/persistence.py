"""M0 — Persistence: (9) P̂(t+h) = P(t).

Ықтималдық нұсқасы: P(t) + persistence қателігінің эмпирикалық кванттильдері
(оқыту кезеңінде әр горизонт h бойынша бөлек есептеледі).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.base import Forecaster, finalize_quantiles, residual_quantiles


class PersistenceModel(Forecaster):
    name = "persistence"
    title = {"kk": "Persistence (M0)", "ru": "Persistence (M0)"}

    def __init__(self):
        self.err_q: dict[int, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceModel":
        """X бағандары: p_now, h. Әр горизонт үшін қателік кванттильдері."""
        resid = np.asarray(y) - X["p_now"].to_numpy()
        for h, idx in X.groupby("h").groups.items():
            self.err_q[int(h)] = residual_quantiles(resid[X.index.get_indexer(idx)], self.taus)
        return self

    def _q_for(self, h: int) -> np.ndarray:
        if h in self.err_q:
            return self.err_q[h]
        if not self.err_q:
            return np.zeros(len(self.taus))
        nearest = min(self.err_q, key=lambda k: abs(k - h))
        return self.err_q[nearest]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p_now = X["p_now"].to_numpy()
        hs = X["h"].astype(int).to_numpy() if "h" in X else np.ones(len(X), int)
        offs = np.vstack([self._q_for(int(h)) for h in hs]) if len(X) else np.zeros((0, len(self.taus)))
        q = p_now[:, None] + offs
        out = finalize_quantiles(q, self.taus, X.index)
        out.loc[np.isnan(p_now)] = np.nan
        return out
