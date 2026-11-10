"""M1 — Физикалық модель: (10) P̂ = PC( v_hub · (ρ/ρ0)^(1/3) ).

Қуат қисығы PC оқыту деректерінен эмпирикалық құрылады (физикалық түрлендіру
белгілерде p_phys ретінде есептеледі). Белгісіздік: p_phys бинтері бойынша
шартты қалдық кванттильдері.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.base import Forecaster, finalize_quantiles, residual_quantiles

N_BINS = 10


class PhysicalModel(Forecaster):
    name = "physical"
    title = {"kk": "Физикалық модель (M1)", "ru": "Физическая модель (M1)"}

    def __init__(self):
        self.bin_q: dict[int, np.ndarray] = {}
        self.global_q = None

    @staticmethod
    def _bins(p: np.ndarray) -> np.ndarray:
        return np.clip((np.nan_to_num(p) * N_BINS).astype(int), 0, N_BINS - 1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PhysicalModel":
        p = X["p_phys"].to_numpy()
        resid = np.asarray(y) - p
        self.global_q = residual_quantiles(resid, self.taus)
        b = self._bins(p)
        for k in range(N_BINS):
            m = (b == k) & np.isfinite(resid)
            self.bin_q[k] = residual_quantiles(resid[m], self.taus) if m.sum() >= 50 else self.global_q
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = X["p_phys"].to_numpy()
        b = self._bins(p)
        offs = np.vstack([self.bin_q.get(int(k), self.global_q) for k in b]) if len(X) else np.zeros((0, len(self.taus)))
        out = finalize_quantiles(p[:, None] + offs, self.taus, X.index)
        out.loc[np.isnan(p)] = np.nan
        return out
