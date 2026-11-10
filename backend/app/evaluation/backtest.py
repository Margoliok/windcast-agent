"""Тест кезеңіндегі бэктест: әр модель, әр горизонт бойынша кванттильдік болжамдар.

Нәтиже — «ұзын» кесте: valid_time, h, model, q0.1…q0.9, y (нормаланған қуат).
Горизонт h үшін болжам issue_time = valid_time − h сәтінде шығарылған деп есептеледі:
  * persistence  — P(issue_time);
  * lgbm_short   — issue_time-дағы нақты қуат + NWP (тек h ≤ 6);
  * NWP модельдері — алдын ала болжау уақыты h-қа сәйкес NWP дереккөзі (select_by_lead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.farms import KELMARSH
from app.evaluation import metrics as M
from app.features.build import SHORT_HORIZONS, persistence_frame, select_by_lead, short_dataset
from app.ingest import dataset as ds
from app.models.registry import FarmBundle
from app.models.training import load_features_for

KEY_HORIZONS = (1, 6, 24, 48, 72)
QCOLS = [f"q{t:g}" for t in M.QUANTILES]


def nwp_sources(bundle: FarmBundle) -> dict[int, pd.DataFrame]:
    """Горизонтқа сәйкес NWP белгілері: {0: nwp, 1: prev1, 2: prev2, 3: prev3}."""
    avail = ds.available_weather()
    src = {0: load_features_for("nwp", bundle.curve)}
    for d in (1, 2, 3):
        if f"prev{d}" in avail:
            src[d] = load_features_for(f"prev{d}", bundle.curve)
    return src


def _frame(valid_time, h, model, q: pd.DataFrame, y) -> pd.DataFrame:
    out = pd.DataFrame(q.to_numpy(), columns=QCOLS)
    out.insert(0, "valid_time", np.asarray(valid_time))
    out.insert(1, "h", h)
    out.insert(2, "model", model)
    out["y"] = np.asarray(y, dtype=float)
    return out


def run_backtest(bundle: FarmBundle, horizons=KEY_HORIZONS, extra_models: dict | None = None) -> pd.DataFrame:
    meta = ds.load_meta()
    test = meta["splits"]["test"]
    power = ds.load_power()["power_norm"].asfreq("1h")
    a, b = pd.Timestamp(test[0]), pd.Timestamp(test[1])
    y_test = power.loc[a:b]
    sources = nwp_sources(bundle)
    parts = []

    # NWP модельдері: әр дереккөз бойынша бір рет болжап, горизонттарға таратамыз
    cache: dict[int, dict[str, pd.DataFrame]] = {}
    for h in horizons:
        day = min(h // 24, max(sources))
        src_feats = select_by_lead(sources, h)
        if day not in cache:
            F = src_feats.reindex(y_test.index)
            ok = F["ws100"].notna().to_numpy()
            cache[day] = {}
            for name in bundle.available_models:
                q = pd.DataFrame(np.nan, index=F.index, columns=QCOLS)
                if ok.any():
                    q.loc[ok] = bundle.predict_nwp(F[ok], name).to_numpy()
                cache[day][name] = q
        for name, q in cache[day].items():
            parts.append(_frame(y_test.index, h, name, q, y_test))

    # Persistence
    pers = bundle.short_models.get("persistence")
    if pers is not None:
        P = persistence_frame(power, horizons)
        P = P[(P.valid_time >= a) & (P.valid_time <= b)].reset_index(drop=True)
        q = pers.predict(P)
        for h in horizons:
            m = (P.h == h).to_numpy()
            parts.append(_frame(P.valid_time[m], h, "persistence", q[m], P.y[m]))

    # LightGBM өте қысқа мерзімді режим
    short = bundle.short_models.get("lgbm_short")
    hs = [h for h in horizons if h in SHORT_HORIZONS]
    if short is not None and hs:
        S = short_dataset(power, sources[0], hs)
        S = S[(S.valid_time >= a) & (S.valid_time <= b)].dropna(subset=["ws100", "p_now"]).reset_index(drop=True)
        q = short.predict(S)
        for h in hs:
            m = (S.h == h).to_numpy()
            parts.append(_frame(S.valid_time[m], h, "lgbm_short", q[m], S.y[m]))

    for name, df in (extra_models or {}).items():
        parts.append(df.assign(model=name)[["valid_time", "h", "model", *QCOLS, "y"]])

    bt = pd.concat(parts, ignore_index=True)
    bt["valid_time"] = pd.to_datetime(bt["valid_time"])
    return bt


def common_sample(bt: pd.DataFrame, models=None) -> pd.DataFrame:
    """Әр горизонтта барлық модельдердің болжамы бар уақыттар ғана (әділ салыстыру)."""
    bt = bt.dropna(subset=["y", "q0.5"])
    if models is not None:
        bt = bt[bt.model.isin(models)]
    out = []
    for h, g in bt.groupby("h"):
        n_models = g.model.nunique()
        cnt = g.groupby("valid_time").model.nunique()
        keep = cnt[cnt == n_models].index
        out.append(g[g.valid_time.isin(keep)])
    return pd.concat(out, ignore_index=True) if out else bt


def metrics_table(bt: pd.DataFrame, capacity: float = KELMARSH.capacity_mw, common: bool = True) -> pd.DataFrame:
    """Кесте 14 + 16: модель × горизонт бойынша метрикалар (МВт және %)."""
    data = common_sample(bt) if common else bt.dropna(subset=["y", "q0.5"])
    rows = []
    for (h, model), g in data.groupby(["h", "model"]):
        y = g.y.to_numpy() * capacity
        q = g[QCOLS] * capacity
        row = {"h": int(h), "model": model, **M.point_metrics(y, q["q0.5"], capacity)}
        row.update(M.probabilistic_metrics(y, q, capacity))
        rows.append(row)
    t = pd.DataFrame(rows)
    ref = t[t.model == "persistence"].set_index("h")["rmse"]
    t["skill"] = [M.skill_score(r.rmse, ref.get(r.h, np.nan)) for r in t.itertuples()]
    return t.sort_values(["h", "model"]).reset_index(drop=True)


def horizon_curve(bundle: FarmBundle, models=("persistence", "physical", "lgbm", "ensemble", "lgbm_short")) -> pd.DataFrame:
    """Сурет 16: nMAE(h), h = 1…72."""
    bt = run_backtest(bundle, horizons=tuple(range(1, 73)))
    bt = bt[bt.model.isin(models)]
    t = metrics_table(bt)
    return t[["h", "model", "nmae", "nrmse", "crps"]]
