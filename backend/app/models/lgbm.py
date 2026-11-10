"""M2 — LightGBM кванттильдік регрессиясы.

Әр τ ∈ {0.1, …, 0.9} үшін бөлек модель (5)-пинбол функциясын минимизациялайды
(GEFCom2014 жеңімпаздарының тәсілі). Кейін кванттильдер реттеледі және [0,1]-ге қиылады.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.models.base import Forecaster, finalize_quantiles

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


class LGBMQuantileModel(Forecaster):
    name = "lgbm"
    title = {"kk": "LightGBM кванттильдік (M2)", "ru": "LightGBM квантильный (M2)"}

    def __init__(self, features: list[str], params: dict | None = None, seed: int = 42):
        self.features = list(features)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.offsets: dict[float, float] = {}  # конформдық түзетулер (CQR)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMQuantileModel":
        m = np.isfinite(np.asarray(y, float))
        Xf, yf = X.loc[m, self.features], np.asarray(y, float)[m]
        for tau in self.taus:
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=self.seed, **self.params)
            mdl.fit(Xf, yf)
            self.models[tau] = mdl
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        q = np.column_stack([self.models[t].predict(Xf) + self.offsets.get(t, 0.0) for t in self.taus])
        return finalize_quantiles(q, self.taus, X.index)

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> dict[float, float]:
        """Конформдық кванттильдік регрессия (Romano et al., 2019) идеясы бойынша түзету:
        валидацияда әр τ үшін δ_τ = Quantile_τ(y − q̂_τ), сонда P(y ≤ q̂_τ + δ_τ) ≈ τ.
        P50 түзетілмейді (нүктелік болжам өзгермеуі үшін)."""
        m = np.isfinite(np.asarray(y, float))
        Xf, yf = X.loc[m, self.features], np.asarray(y, float)[m]
        self.offsets = {}
        for t in self.taus:
            if t == 0.5:
                continue
            self.offsets[t] = float(np.quantile(yf - self.models[t].predict(Xf), t))
        return self.offsets

    @property
    def median_model(self) -> lgb.LGBMRegressor:
        return self.models[0.5]
